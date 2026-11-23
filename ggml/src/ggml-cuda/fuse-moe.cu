#include "fuse-moe.cuh"
#include "quantize.cuh"
#include "unary.cuh"
#include "vecdotq.cuh"

#include <cmath>
#include <cstdint>
#include <type_traits>

// Plan 17 section C, group 3: Inkling 2's routed-expert chain at decode rows.
//  - moe_router_chain: the F32 router product, the sigmoid/bias top-k selection, the routes sorted by the card's local
//    experts and the q8_1 copy of the expert input, one launch a layer.
//  - moe_gate_up_situ: the merged gate/up product with the SITU activation in its epilogue.
//  - moe_experts_weighted_sum: the down product with the weighted sum of the selected experts in its epilogue.
// Each keeps the arithmetic and its order of the launches it replaces: the router product f32_cols' summation order
// (f32-few.cu), the selection topk_moe_cuda's loop (topk-moe.cu), the routes mmvq_sort_routes' sort, the q8_1 copies
// quantize_q8_1's arithmetic, the dots mul_mat_vec_q_moe's, the SITU glu_fn_kernel's and the weighted sum
// k_moe_weighted_sum's. So every output is bit-identical.

// f32_cols' shape (f32-few.cu): 12 warps split K, 8 lanes of a warp hold K's float4s, 4 lane groups hold the columns,
// and the lanes' sums then the warps' run in a fixed order.
#define MOE_CHAIN_WARPS      12
#define MOE_CHAIN_KLANES     8
#define MOE_CHAIN_ROWS       3
#define MOE_CHAIN_MAX_TOKENS 32

struct moe_chain_args {
    float   clamp_val;
    float   scale_val;
    bool    norm_eps;
    int32_t first; // the card's first global expert (the expert partition), 0 when not partitioned
    int32_t count; // and how many it owns
};

// the local index of a global expert, as ggml_cuda_local_expert_ids maps it (-1 for another card's expert)
__device__ __forceinline__ int32_t moe_chain_local(int32_t expert, int32_t first, int32_t count) {
    return expert >= first && expert < first + count ? expert - first : -1;
}

template <int n_expert, int cpl, bool has_bias, bool tma_mode>
__global__ __launch_bounds__(MOE_CHAIN_WARPS*WARP_SIZE, 1) void moe_router_chain(
        const float * cur, const float * router, const float * bias, float * logits,
        int32_t * ids, float * weights, int32_t * routes, block_q8_1 * q8, unsigned int * flag,
        int32_t * local_ids_out, const int n_embd, const int n_tokens, const int n_used,
        const int ids_stride, const moe_chain_args args) {
    constexpr int ncol = 4*cpl;
    __shared__ float                     partial[MOE_CHAIN_WARPS][MOE_CHAIN_ROWS][ncol];
    [[maybe_unused]] __shared__ uint32_t keys[tma_mode ? 1 : 256];
    __shared__ bool                      is_last;

    ggml_cuda_pdl_sync();

    const int warp = threadIdx.x/WARP_SIZE;
    const int lane = threadIdx.x%WARP_SIZE;
    const int kl   = lane%MOE_CHAIN_KLANES;
    const int cg   = lane/MOE_CHAIN_KLANES;
    const int tid  = threadIdx.x;

    // the expert input as q8_1, a warp a block, as quantize_q8_1 writes it (n_embd a multiple of QK8_1, so a block never
    // crosses a row)
    if constexpr (!tma_mode) {
        const int per_row = n_embd/QK8_1;
        const int n_blk   = n_tokens*per_row;
        const int stride  = gridDim.x*MOE_CHAIN_WARPS;
        for (int b = blockIdx.x*MOE_CHAIN_WARPS + warp; b < n_blk; b += stride) {
            const int   row = b/per_row;
            const int   col = (b%per_row)*QK8_1;
            const float xi  = cur[(int64_t) row*n_embd + col + lane];

            float amax = warp_reduce_max<QK8_1>(fabsf(xi));
            const float  d = amax/127.0f;
            const int8_t q = amax == 0.0f ? 0 : roundf(xi/d);
            const int  sum = warp_reduce_sum<QK8_1>((int) q);

            q8[b].qs[lane] = q;
            if (lane == 0) {
                q8[b].ds = make_half2(d, d*sum);
            }
        }
    } else {
        GGML_UNUSED(q8);
    }

    // the router product, f32_cols' loop over this block's rows and the tokens as its columns
    const int64_t row0 = int64_t(blockIdx.x)*MOE_CHAIN_ROWS;
    const float * ar[MOE_CHAIN_ROWS];
#pragma unroll
    for (int r = 0; r < MOE_CHAIN_ROWS; ++r) {
        ar[r] = router + min(row0 + r, int64_t(n_expert - 1))*n_embd;
    }
    const float * xc[cpl];
#pragma unroll
    for (int j = 0; j < cpl; ++j) {
        xc[j] = cur + min(int64_t(cg*cpl + j), int64_t(n_tokens - 1))*n_embd;
    }

    float acc[MOE_CHAIN_ROWS][cpl] = {};
    const uint64_t policy = ggml_cuda_l2_policy_evict_first();
    constexpr int64_t kstep = MOE_CHAIN_WARPS*MOE_CHAIN_KLANES*4;
#pragma unroll 4
    for (int64_t k0 = (warp*MOE_CHAIN_KLANES + kl)*4; k0 < n_embd; k0 += kstep) {
        float4 av[MOE_CHAIN_ROWS];
#pragma unroll
        for (int r = 0; r < MOE_CHAIN_ROWS; ++r) {
            const int4 raw = ggml_cuda_ldg_128_evict_first((const int4 *) (ar[r] + k0), policy);
            memcpy(&av[r], &raw, sizeof(float4));
        }
#pragma unroll
        for (int j = 0; j < cpl; ++j) {
            const float4 xv = __ldg((const float4 *) (xc[j] + k0));
#pragma unroll
            for (int r = 0; r < MOE_CHAIN_ROWS; ++r) {
                acc[r][j] = fmaf(av[r].x, xv.x, acc[r][j]);
                acc[r][j] = fmaf(av[r].y, xv.y, acc[r][j]);
                acc[r][j] = fmaf(av[r].z, xv.z, acc[r][j]);
                acc[r][j] = fmaf(av[r].w, xv.w, acc[r][j]);
            }
        }
    }

#pragma unroll
    for (int r = 0; r < MOE_CHAIN_ROWS; ++r) {
#pragma unroll
        for (int j = 0; j < cpl; ++j) {
#pragma unroll
            for (int offset = MOE_CHAIN_KLANES/2; offset > 0; offset >>= 1) {
                acc[r][j] += ggml_cuda_shfl_xor(acc[r][j], offset, WARP_SIZE);
            }
            if (kl == 0) {
                partial[warp][r][cg*cpl + j] = acc[r][j];
            }
        }
    }
    __syncthreads();

    for (int o = tid; o < MOE_CHAIN_ROWS*ncol; o += MOE_CHAIN_WARPS*WARP_SIZE) {
        const int r = o%MOE_CHAIN_ROWS;
        const int c = o/MOE_CHAIN_ROWS;
        if (row0 + r < n_expert && c < n_tokens) {
            float sum = 0.0f;
#pragma unroll
            for (int w = 0; w < MOE_CHAIN_WARPS; ++w) {
                sum += partial[w][r][c];
            }
            logits[c*n_expert + row0 + r] = sum;
        }
    }

    // every block's logits are whole before the last one selects; it also leaves the flag zero for the next launch
    __syncthreads();
    __threadfence();
    if (tid == 0) {
        is_last = atomicAdd(flag, 1u) == gridDim.x - 1;
    }
    __syncthreads();
    if (!is_last) {
        return;
    }

    constexpr int ept    = (n_expert + WARP_SIZE - 1)/WARP_SIZE;
    constexpr int rounds = (MOE_CHAIN_MAX_TOKENS + MOE_CHAIN_WARPS - 1)/MOE_CHAIN_WARPS;

    // ids and weights can alias the logits, so every row a warp selects reads all of its logits first
    float wt[rounds][ept];
    [[maybe_unused]] float sel[has_bias ? rounds : 1][has_bias ? ept : 1];
#pragma unroll
    for (int t = 0; t < rounds; ++t) {
#pragma unroll
        for (int i = 0; i < ept; ++i) {
            wt[t][i] = -INFINITY;
        }
    }
#pragma unroll
    for (int t = 0; t < rounds; ++t) {
        const int row = warp + t*MOE_CHAIN_WARPS;
        if (row >= n_tokens) {
            continue;
        }
        const float * lg = logits + (int64_t) row*n_expert;
#pragma unroll
        for (int i = 0; i < ept; ++i) {
            const int expert = i*WARP_SIZE + lane;
            wt[t][i] = n_expert % WARP_SIZE == 0 || expert < n_expert ? __ldcg(lg + expert) : -INFINITY;
        }
    }
    __syncthreads();

#pragma unroll
    for (int t = 0; t < rounds; ++t) {
        const int row = warp + t*MOE_CHAIN_WARPS;
        if (row >= n_tokens) {
            continue;
        }

#pragma unroll
        for (int i = 0; i < ept; ++i) {
            const int expert = i*WARP_SIZE + lane;
            if (n_expert % WARP_SIZE == 0 || expert < n_expert) {
                wt[t][i] = 1.f/(1.f + expf(-wt[t][i]));
            }
        }
        // NaN logits select nothing under an iterative argmax, so they become -FLT_MAX
#pragma unroll
        for (int i = 0; i < ept; ++i) {
            if (__isnanf(wt[t][i])) {
                wt[t][i] = -FLT_MAX;
            }
        }

        if constexpr (has_bias) {
#pragma unroll
            for (int i = 0; i < ept; ++i) {
                const int expert = i*WARP_SIZE + lane;
                sel[t][i] = (n_expert % WARP_SIZE == 0 || expert < n_expert) ? wt[t][i] + bias[expert] : -INFINITY;
                if (__isnanf(sel[t][i])) {
                    sel[t][i] = -FLT_MAX;
                }
            }
        }

        float wt_sum = 0.f;

        float output_weights[ept];
#pragma unroll
        for (int i = 0; i < ept; ++i) {
            output_weights[i] = 0.f;
        }

        for (int k = 0; k < n_used; ++k) {
            float max_val    = wt[t][0];
            int   max_expert = lane;
            float max_val_s  = 0.0f;

            if constexpr (has_bias) {
                max_val_s = sel[t][0];
#pragma unroll
                for (int i = 1; i < ept; ++i) {
                    const int expert = lane + i*WARP_SIZE;
                    if ((n_expert % WARP_SIZE == 0 || expert < n_expert) && sel[t][i] > max_val_s) {
                        max_val    = wt[t][i];
                        max_val_s  = sel[t][i];
                        max_expert = expert;
                    }
                }
#pragma unroll
                for (int mask = WARP_SIZE/2; mask > 0; mask /= 2) {
                    const float val    = ggml_cuda_shfl_xor(max_val, mask, WARP_SIZE);
                    const float val_s  = ggml_cuda_shfl_xor(max_val_s, mask, WARP_SIZE);
                    const int   expert = ggml_cuda_shfl_xor(max_expert, mask, WARP_SIZE);
                    if (val_s > max_val_s || (val_s == max_val_s && expert < max_expert)) {
                        max_val    = val;
                        max_val_s  = val_s;
                        max_expert = expert;
                    }
                }
                if ((max_expert & (WARP_SIZE - 1)) == lane) {
                    sel[t][max_expert/WARP_SIZE] = -INFINITY;
                }
            } else {
#pragma unroll
                for (int i = 1; i < ept; ++i) {
                    const int expert = lane + i*WARP_SIZE;
                    if ((n_expert % WARP_SIZE == 0 || expert < n_expert) && wt[t][i] > max_val) {
                        max_val    = wt[t][i];
                        max_expert = expert;
                    }
                }
#pragma unroll
                for (int mask = WARP_SIZE/2; mask > 0; mask /= 2) {
                    const float val    = ggml_cuda_shfl_xor(max_val, mask, WARP_SIZE);
                    const int   expert = ggml_cuda_shfl_xor(max_expert, mask, WARP_SIZE);
                    if (val > max_val || (val == max_val && expert < max_expert)) {
                        max_val    = val;
                        max_expert = expert;
                    }
                }
                if ((max_expert & (WARP_SIZE - 1)) == lane) {
                    wt[t][max_expert/WARP_SIZE] = -INFINITY;
                }
            }

            if ((k & (WARP_SIZE - 1)) == lane) {
                output_weights[k/WARP_SIZE] = max_val;
            }
            if ((max_expert & (WARP_SIZE - 1)) == lane) {
                ids[row*ids_stride + k] = max_expert;
                if (local_ids_out != nullptr) {
                    local_ids_out[row*n_used + k] = moe_chain_local(max_expert, args.first, args.count);
                }
                wt_sum += max_val;
            }
        }

        wt_sum              = warp_reduce_sum<WARP_SIZE>(wt_sum);
        wt_sum              = args.norm_eps ? wt_sum + args.clamp_val : max(wt_sum, args.clamp_val);
        const float inv_sum = 1.0f/wt_sum;

#pragma unroll
        for (int i = 0; i < ept; ++i) {
            output_weights[i] *= inv_sum;
        }
#pragma unroll
        for (int i = 0; i < ept; ++i) {
            const int idx = i*WARP_SIZE + lane;
            if (idx < n_used) {
                weights[row*n_used + idx] = output_weights[i]*args.scale_val;
            }
        }
    }

    if constexpr (!tma_mode) {
        // every warp's selected ids must be visible before sorting the layer's routes
        __syncthreads();

        // the layer's routes, sorted by the card's local expert as mmvq_sort_routes sorts them
        const uint32_t n_routes = uint32_t(n_tokens*n_used);
        uint32_t key = UINT32_MAX;
        if (tid < n_routes) {
            const int32_t expert = moe_chain_local(ids[(tid/uint32_t(n_used))*uint32_t(ids_stride) + tid%uint32_t(n_used)],
                                                   args.first, args.count);
            key = (uint32_t(max(expert, -1) + 1) << 8) | tid;
        }
        for (uint32_t width = 2; width <= 256; width *= 2) {
            for (uint32_t distance = width/2; distance; distance /= 2) {
                uint32_t peer = UINT32_MAX;
                if (distance < 32) {
                    peer = ggml_cuda_shfl_xor(key, distance, 32);
                } else {
                    if (tid < 256) {
                        keys[tid] = key;
                    }
                    __syncthreads();
                    if (tid < 256) {
                        peer = keys[tid ^ distance];
                    }
                    __syncthreads();
                }
                if (tid < 256) {
                    const bool take_min = ((tid & width) == 0) == ((tid & distance) == 0);
                    key = take_min ? min(peer, key) : max(peer, key);
                }
            }
        }
        if (tid < n_routes) {
            routes[tid] = int32_t(key);
        }
    } else {
        GGML_UNUSED(routes);
    }

    __syncthreads();
    if (tid == 0) {
        *flag = 0;
    }
}

// The merged gate/up product with the SITU of glu_fn_kernel (unary.cu) in its epilogue: a warp computes the gate rows and
// the up rows of one route (a token and one of its experts) as mul_mat_vec_q_moe computes them, and the SITU of the two
// tensors' halves is written instead of the two halves.
template <ggml_type type, int c_rows_per_block, bool sorted>
__launch_bounds__(8*ggml_cuda_get_physical_warp_size(), 1)
__global__ void moe_gate_up_situ(
        const void * vx_ptr, const void * vy_ptr,
        const int32_t * ids_ptr, const int32_t * routes_ptr,
        float * dst_ptr,
        const uint32_t ncols_x, const uint32_t n_ff, const uint32_t nrows_x,
        const uint32_t stride_row_x, const uint32_t stride_col_y, const uint32_t stride_col_dst,
        const uint32_t stride_channel_x, const uint32_t stride_channel_y, const uint32_t stride_channel_dst,
        const uint32_t nchannels_y, const uint32_t ids_stride, const int32_t first, const int32_t count,
        const float beta, const float lb) {
    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = get_vdr_mmvq(type);
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    constexpr vec_dot_q_cuda_t vec_dot_q_cuda = get_vec_dot_q_cuda(type);

    const int row0 = c_rows_per_block*blockIdx.x;
    const int blocks_per_row_x = ncols_x/qk;
    constexpr int blocks_per_iter = vdr*warp_size/qi;

    ggml_cuda_pdl_sync();

    uint32_t token_idx;
    uint32_t channel_dst;
    int32_t  channel_x;
    if constexpr (sorted) {
        const uint32_t route = uint32_t(routes_ptr[blockIdx.y*blockDim.y + threadIdx.y]);
        token_idx   = (route & 255u)/8u;
        channel_dst = route%8u;
        channel_x   = int32_t(route >> 8) - 1;
    } else {
        token_idx   = blockIdx.y;
        channel_dst = threadIdx.y;
        channel_x   = moe_chain_local(ids_ptr[channel_dst + token_idx*ids_stride], first, count);
    }

    if (channel_x < 0) {
        // another card's expert: the launch it replaces zeroes its rows, and the SITU of two zeros is zero
        if (threadIdx.x < c_rows_per_block && row0 + threadIdx.x < nrows_x) {
            dst_ptr[channel_dst*stride_channel_dst + token_idx*stride_col_dst + row0 + threadIdx.x] = 0.0f;
        }
        ggml_cuda_pdl_lc();
        return;
    }

    const block_q8_1 * y = ((const block_q8_1 *) vy_ptr) + (channel_dst%nchannels_y)*stride_channel_y +
        token_idx*stride_col_y;
    const int kbx_offset = int(channel_x*stride_channel_x) + int(row0*stride_row_x);
    const int kbx_up     = kbx_offset + int(n_ff*stride_row_x);

    float tmp_gate[c_rows_per_block] = {0.0f};
    float tmp_up[c_rows_per_block]   = {0.0f};

    for (int kbx = threadIdx.x/(qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx*(qk/QK8_1);
        const int kqs = vdr*(threadIdx.x%(qi/vdr));

#pragma unroll
        for (int i = 0; i < c_rows_per_block; ++i) {
            const int row = uint32_t(row0 + i) < nrows_x ? i : 0;
            tmp_gate[i] += vec_dot_q_cuda(vx_ptr, &y[kby], kbx_offset + row*stride_row_x + kbx, kqs);
            tmp_up[i]   += vec_dot_q_cuda(vx_ptr, &y[kby], kbx_up     + row*stride_row_x + kbx, kqs);
        }
    }

    ggml_cuda_pdl_lc();

#pragma unroll
    for (int i = 0; i < c_rows_per_block; ++i) {
        tmp_gate[i] = warp_reduce_sum<warp_size>(tmp_gate[i]);
        tmp_up[i]   = warp_reduce_sum<warp_size>(tmp_up[i]);
    }

    if (threadIdx.x < c_rows_per_block && (c_rows_per_block == 1 || uint32_t(row0 + threadIdx.x) < nrows_x)) {
        const float r = ggml_cuda_op_situ_single(tmp_gate[threadIdx.x], tmp_up[threadIdx.x], beta, lb);
        dst_ptr[channel_dst*stride_channel_dst + token_idx*stride_col_dst + row0 + threadIdx.x] = r;
    }
}

// The down product and k_moe_weighted_sum (binbcast.cu) as one launch: the eight slots of a token are the block's warps,
// so the sum runs in its slot order with each product and each sum rounded on its own.
template <ggml_type type, int c_rows_per_block>
__launch_bounds__(8*ggml_cuda_get_physical_warp_size(), 1)
__global__ void moe_experts_weighted_sum(
        const void * vx_ptr, const void * vy_ptr,
        const int32_t * ids_ptr, const float * weights_ptr,
        float * dst_ptr,
        const uint32_t ncols_x, const uint32_t nrows_x, const uint32_t stride_row_x,
        const uint32_t stride_col_y, const uint32_t stride_channel_x, const uint32_t stride_channel_y,
        const uint32_t nchannels_y, const uint32_t ids_stride, const uint32_t n_used, const uint32_t dst_col_stride,
        const uint32_t weights_s2, const int32_t first, const int32_t count) {
    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = get_vdr_mmvq(type);
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    constexpr vec_dot_q_cuda_t vec_dot_q_cuda = get_vec_dot_q_cuda(type);

    ggml_cuda_pdl_sync();

    const int      row0 = c_rows_per_block*blockIdx.x;
    const uint32_t token_idx   = blockIdx.y;
    const uint32_t channel_dst = threadIdx.y;
    const int32_t  channel_x   = moe_chain_local(ids_ptr[channel_dst + token_idx*ids_stride], first, count);

    float tmp[c_rows_per_block] = {0.0f};

    if (channel_x >= 0) {
        const block_q8_1 * y = ((const block_q8_1 *) vy_ptr) + (channel_dst%nchannels_y)*stride_channel_y +
            token_idx*stride_col_y;
        const int kbx_offset = int(channel_x*stride_channel_x) + int(row0*stride_row_x);

        for (int kbx = threadIdx.x/(qi/vdr); kbx < int(ncols_x/qk); kbx += vdr*warp_size/qi) {
            const int kby = kbx*(qk/QK8_1);
            const int kqs = vdr*(threadIdx.x%(qi/vdr));
#pragma unroll
            for (int i = 0; i < c_rows_per_block; ++i) {
                const int row = uint32_t(row0 + i) < nrows_x ? i : 0;
                tmp[i] += vec_dot_q_cuda(vx_ptr, &y[kby], kbx_offset + row*stride_row_x + kbx, kqs);
            }
        }
        ggml_cuda_pdl_lc();
#pragma unroll
        for (int i = 0; i < c_rows_per_block; ++i) {
            tmp[i] = warp_reduce_sum<warp_size>(tmp[i]);
        }
    }

    __shared__ float slot_val[c_rows_per_block][8];
    if (threadIdx.x < c_rows_per_block) {
        slot_val[threadIdx.x][channel_dst] = tmp[threadIdx.x];
    }
    __syncthreads();

    if (threadIdx.y == 0 && threadIdx.x < c_rows_per_block &&
            (c_rows_per_block == 1 || uint32_t(row0 + threadIdx.x) < nrows_x)) {
        const int i = threadIdx.x;
        const float * w = weights_ptr + token_idx*weights_s2;
        float acc = __fmul_rn(slot_val[i][0], w[0]);
        for (uint32_t s = 1; s < n_used; ++s) {
            acc = __fadd_rn(acc, __fmul_rn(slot_val[i][s], w[s]));
        }
        dst_ptr[token_idx*dst_col_stride + row0 + i] = acc;
    }
}

// ===== host side =====

static int moe_chain_cpl(int n_tokens) {
    if (n_tokens <= 4)  { return 1; }
    if (n_tokens <= 8)  { return 2; }
    if (n_tokens <= 16) { return 4; }
    if (n_tokens <= 24) { return 6; }
    return 8;
}

void ggml_cuda_op_moe_router_chain(ggml_backend_cuda_context & ctx, const ggml_tensor * router, const ggml_tensor * cur,
        const ggml_tensor * bias, ggml_tensor * logits, ggml_tensor * ids, ggml_tensor * weights,
        int32_t * routes, block_q8_1 * q8, unsigned int * flag, int32_t first, int32_t count,
        float clamp_val, float scale_val, bool norm_eps, int32_t * local_ids_out, bool tma_mode) {
    constexpr int n_expert = 384;
    GGML_ASSERT(router->ne[0] == 4608 && router->ne[1] == n_expert && cur->type == GGML_TYPE_F32);
    GGML_ASSERT(tma_mode || (routes != nullptr && q8 != nullptr));

    const int n_embd   = (int) router->ne[0];
    const int n_tokens = (int) ggml_nrows(cur);
    const int n_used   = (int) ids->ne[0];
    GGML_ASSERT(n_tokens >= 1 && n_tokens <= MOE_CHAIN_MAX_TOKENS && n_used >= 1 && n_used <= 32 && n_tokens * n_used <= 256);

    const moe_chain_args args = { clamp_val, scale_val, norm_eps, first, count };

    // ids is the argsort's narrowed view: a row of n_used ids every ids_stride elements, as topk_moe_cuda writes it
    const int ids_stride = (int) (ids->nb[1]/ggml_type_size(ids->type));
    GGML_ASSERT(ids->nb[0] == sizeof(int32_t) && ids_stride >= n_used);

    const dim3 grid((n_expert + MOE_CHAIN_ROWS - 1)/MOE_CHAIN_ROWS, 1, 1);
    const dim3 block(MOE_CHAIN_WARPS*WARP_SIZE, 1, 1);
    const ggml_cuda_kernel_launch_params params(grid, block, 0, ctx.stream());

    const auto launch_impl = [&](auto cpl_tag, auto bias_tag, auto tma_tag) {
        ggml_cuda_kernel_launch(
            moe_router_chain<n_expert, decltype(cpl_tag)::value, decltype(bias_tag)::value, decltype(tma_tag)::value>,
            params, (const float *) cur->data, (const float *) router->data,
            decltype(bias_tag)::value ? (const float *) bias->data : (const float *) nullptr,
            (float *) logits->data, (int32_t *) ids->data, (float *) weights->data,
            routes, q8, flag, local_ids_out, n_embd, n_tokens, n_used, ids_stride, args);
    };
    const auto launch_cpl = [&](auto cpl_tag) {
        if (bias != nullptr) {
            if (tma_mode) {
                launch_impl(cpl_tag, std::true_type{}, std::true_type{});
            } else {
                launch_impl(cpl_tag, std::true_type{}, std::false_type{});
            }
        } else {
            if (tma_mode) {
                launch_impl(cpl_tag, std::false_type{}, std::true_type{});
            } else {
                launch_impl(cpl_tag, std::false_type{}, std::false_type{});
            }
        }
    };

    switch (moe_chain_cpl(n_tokens)) {
        case 1: launch_cpl(std::integral_constant<int, 1>{}); break;
        case 2: launch_cpl(std::integral_constant<int, 2>{}); break;
        case 4: launch_cpl(std::integral_constant<int, 4>{}); break;
        case 6: launch_cpl(std::integral_constant<int, 6>{}); break;
        default: launch_cpl(std::integral_constant<int, 8>{}); break;
    }
}

void ggml_cuda_op_moe_gate_up_situ(ggml_backend_cuda_context & ctx, const ggml_tensor * w, const ggml_tensor * x,
        const ggml_tensor * ids, ggml_tensor * glu, const block_q8_1 * q8, block_q8_1 * q8_dst,
        const int32_t * routes, int32_t first, int32_t count, float beta, float lb, bool sorted) {
    GGML_ASSERT(w->ne[0] == x->ne[0] && w->ne[1] % 2 == 0);

    const int64_t n_ff         = w->ne[1]/2;
    const int64_t n_tokens     = glu->ne[2];
    const int64_t n_used       = glu->ne[1];
    GGML_ASSERT(n_used == 8 && (!sorted || n_tokens * n_used <= 256));
    const int64_t ne10_pad     = GGML_PAD(x->ne[0], MATRIX_ROW_PADDING);
    const int64_t s11          = ne10_pad/QK8_1;
    const int64_t s12          = x->ne[1]*s11;

    const int64_t ts_w       = ggml_type_size(w->type);
    const int64_t ts_glu     = ggml_type_size(glu->type);
    const int64_t s01        = w->nb[1]/ts_w;
    const int64_t s02        = w->nb[2]/ts_w;
    const int64_t s1         = glu->nb[1]/ts_glu;
    const int64_t s2         = glu->nb[2]/ts_glu;
    const int64_t ids_stride = ids->nb[1]/ggml_type_size(ids->type);

    const dim3 grid((unsigned) ((n_ff + 1)/2), (unsigned) n_tokens, 1);
    const dim3 block(WARP_SIZE, (unsigned) n_used, 1);
    const ggml_cuda_kernel_launch_params params(grid, block, 0, ctx.stream());

    const auto launch = [&](auto type) {
        if (sorted) {
            ggml_cuda_kernel_launch(moe_gate_up_situ<decltype(type)::value, 2, true>, params, w->data, q8, (const int32_t *) ids->data,
                routes, (float *) glu->data, (uint32_t) w->ne[0], (uint32_t) n_ff, (uint32_t) n_ff,
                (uint32_t) s01, (uint32_t) s12, (uint32_t) s2, (uint32_t) s02, (uint32_t) s11, (uint32_t) s1,
                (uint32_t) x->ne[1], (uint32_t) ids_stride, first, count, beta, lb);
        } else {
            ggml_cuda_kernel_launch(moe_gate_up_situ<decltype(type)::value, 2, false>, params, w->data, q8, (const int32_t *) ids->data,
                (const int32_t *) nullptr, (float *) glu->data, (uint32_t) w->ne[0], (uint32_t) n_ff, (uint32_t) n_ff,
                (uint32_t) s01, (uint32_t) s12, (uint32_t) s2, (uint32_t) s02, (uint32_t) s11, (uint32_t) s1,
                (uint32_t) x->ne[1], (uint32_t) ids_stride, first, count, beta, lb);
        }
    };
    switch (w->type) {
        case GGML_TYPE_Q3_K: launch(std::integral_constant<ggml_type, GGML_TYPE_Q3_K>{}); break;
        case GGML_TYPE_Q4_K: launch(std::integral_constant<ggml_type, GGML_TYPE_Q4_K>{}); break;
        case GGML_TYPE_Q5_K: launch(std::integral_constant<ggml_type, GGML_TYPE_Q5_K>{}); break;
        default: GGML_ABORT("moe_gate_up_situ has no launch for %s", ggml_type_name(w->type));
    }

    // the down product's quantized activation, as its staging launch writes it (the padded tail zeroed)
    const int64_t glu_pad = GGML_PAD(glu->ne[0], MATRIX_ROW_PADDING);
    quantize_row_q8_1_cuda((const float *) glu->data, q8_dst, glu->ne[0], glu->nb[1]/ts_glu,
        glu->nb[2]/ts_glu, glu->nb[3]/ts_glu, glu_pad, n_used, n_tokens, 1, ctx.stream());
}

void ggml_cuda_op_moe_experts_weighted_sum(ggml_backend_cuda_context & ctx, const ggml_tensor * w, const ggml_tensor * x,
        const ggml_tensor * ids, const ggml_tensor * weights, const block_q8_1 * q8, ggml_tensor * dst,
        int32_t first, int32_t count) {
    const int64_t n_tokens = ids->ne[1];
    const int64_t n_used   = ids->ne[0];
    GGML_ASSERT(n_used >= 1 && n_used <= 8);
    const int64_t ne10_pad = GGML_PAD(x->ne[0], MATRIX_ROW_PADDING);
    const int64_t s11      = ne10_pad/QK8_1;
    const int64_t s12      = x->ne[1]*s11;

    const int64_t ts_w       = ggml_type_size(w->type);
    const int64_t ts_dst     = ggml_type_size(dst->type);
    const int64_t s01        = w->nb[1]/ts_w;
    const int64_t s02        = w->nb[2]/ts_w;
    const int64_t ids_stride = ids->nb[1]/ggml_type_size(ids->type);
    const int64_t dst_stride = dst->nb[1]/ts_dst;

    GGML_ASSERT(weights->ne[0] == 1 && weights->nb[1] == (size_t) ggml_type_size(weights->type));

    const dim3 grid((unsigned) ((w->ne[1] + 1)/2), (unsigned) n_tokens, 1);
    const dim3 block(WARP_SIZE, (unsigned) n_used, 1);
    const ggml_cuda_kernel_launch_params params(grid, block, 0, ctx.stream());

    const auto launch = [&](auto type) {
        ggml_cuda_kernel_launch(moe_experts_weighted_sum<decltype(type)::value, 2>, params, w->data, q8, (const int32_t *) ids->data,
            (const float *) weights->data, (float *) dst->data, (uint32_t) w->ne[0], (uint32_t) w->ne[1],
            (uint32_t) s01, (uint32_t) s12, (uint32_t) s02, (uint32_t) s11, (uint32_t) x->ne[1],
            (uint32_t) ids_stride, (uint32_t) n_used, (uint32_t) dst_stride, (uint32_t) (weights->nb[2]/ggml_type_size(weights->type)),
            first, count);
    };
    switch (w->type) {
        case GGML_TYPE_Q3_K: launch(std::integral_constant<ggml_type, GGML_TYPE_Q3_K>{}); break;
        case GGML_TYPE_Q4_K: launch(std::integral_constant<ggml_type, GGML_TYPE_Q4_K>{}); break;
        case GGML_TYPE_Q5_K: launch(std::integral_constant<ggml_type, GGML_TYPE_Q5_K>{}); break;
        default: GGML_ABORT("moe_experts_weighted_sum has no launch for %s", ggml_type_name(w->type));
    }
}
