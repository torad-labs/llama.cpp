#include "moe-fuse.cuh"
#include "mmq.cuh"
#include "quantize.cuh"

#include <cmath>

// A3: Inkling 2's router product and top-k selection as one launch at decode shapes. The experts' dots are spread over
// MOE_ROUTER_CHUNK-blocks per token, each leaving its chunk's logits in a scratch row; the last block to arrive for a
// token runs the same warp-level argmax loop ggml_cuda_op_topk_moe does, so ids and weights come out byte for byte what
// mul_mat_id and k_moe_weighted_sum read from the mat-vec + topk pair today.
constexpr int MOE_ROUTER_WARP     = 32;
constexpr int MOE_ROUTER_NWARPS   = 8;
constexpr int MOE_ROUTER_CHUNK    = 32;  // experts per block, whole across warps
constexpr int MOE_ROUTER_PER_WARP = MOE_ROUTER_CHUNK / MOE_ROUTER_NWARPS;
static_assert(MOE_ROUTER_CHUNK % MOE_ROUTER_NWARPS == 0, "experts must split whole across warps");

// as calc_experts_per_thread of topk-moe.cu decides it
static constexpr __device__ int moe_router_calc_ept(int n_experts, int warp_size) {
    return n_experts <= warp_size ? 1 : (n_experts % warp_size == 0 ? n_experts / warp_size : n_experts / warp_size + 1);
}

// the warp-local sigmoid of the probs, as topk_moe_cuda applies it before the selection
template <int experts_per_thread>
__device__ void moe_router_sigmoid(float (&vals)[experts_per_thread]) {
#pragma unroll
    for (int i = 0; i < experts_per_thread; ++i) {
        vals[i] = 1.f / (1.f + expf(-vals[i]));
    }
}

struct moe_router_norm_args {
    bool  with_norm;
    bool  norm_eps;
    float clamp_val;
    float scale_val;
};

template <int n_expert, bool has_bias>
__global__ void moe_router_topk_cuda(const float * cur, const float * router,
                                     int32_t * ids, float * weights,
                                     const float * bias, float * logits_scratch,
                                     unsigned int * flags,
                                     const int n_embd, const int n_used, const int ids_stride,
                                     const moe_router_norm_args norm) {
    constexpr int chunks_per_token = n_expert / MOE_ROUTER_CHUNK;

    ggml_cuda_pdl_sync();

    const int t     = blockIdx.y;
    const int e0    = blockIdx.x * MOE_ROUTER_CHUNK;
    const int tid   = threadIdx.x + threadIdx.y * MOE_ROUTER_WARP;
    const int lane  = tid % MOE_ROUTER_WARP;
    const int warp  = tid / MOE_ROUTER_WARP;

    cur    += (int64_t) t * n_embd;
    router += (int64_t) e0 * n_embd;

    float acc[MOE_ROUTER_PER_WARP];
#pragma unroll
    for (int i = 0; i < MOE_ROUTER_PER_WARP; ++i) {
        acc[i] = 0.0f;
    }

#pragma unroll 4
    for (int sb = 0; sb < n_embd; sb += MOE_ROUTER_COL_TILE_C) {
        // the lanes' float4s tile the columns, so both operands stream coalesced
        const float4 c = *(const float4 *) (cur + sb + lane*4);

#pragma unroll
        for (int ei = 0; ei < MOE_ROUTER_PER_WARP; ++ei) {
            const float * w_row = router + (int64_t) (warp*MOE_ROUTER_PER_WARP + ei) * n_embd;
            const float4 w = *(const float4 *) (w_row + sb + lane*4);
            acc[ei] += w.x*c.x + w.y*c.y + w.z*c.z + w.w*c.w;
        }
    }

#pragma unroll
    for (int ei = 0; ei < MOE_ROUTER_PER_WARP; ++ei) {
        const float sum = warp_reduce_sum<MOE_ROUTER_WARP>(acc[ei]);
        if (lane == 0) {
            logits_scratch[(int64_t) t * n_expert + e0 + warp*MOE_ROUTER_PER_WARP + ei] = sum;
        }
    }

    // every writer's fence precedes its add and the last arrival sees all of them, so that block may select at once
    __threadfence();
    __shared__ bool is_last;
    if (tid == 0) {
        is_last = atomicAdd(&flags[t], 1u) == chunks_per_token - 1;
    }
    __syncthreads();

    if (!is_last || warp != 0) {
        return;
    }

    // the arriving count orders every writer's fenced stores ahead of these reads
    __threadfence();

    // the selection: one warp over all experts, mirroring topk_moe_cuda's loop exactly
    constexpr int experts_per_thread = moe_router_calc_ept(n_expert, MOE_ROUTER_WARP);

    const float * logits = logits_scratch + (int64_t) t * n_expert;

    float wt[experts_per_thread];
#pragma unroll
    for (int i = 0; i < experts_per_thread; ++i) {
        wt[i] = -INFINITY;
    }
#pragma unroll
    for (int i = 0; i < n_expert; i += MOE_ROUTER_WARP) {
        const int expert = i + lane;
        wt[i / MOE_ROUTER_WARP] = (n_expert % MOE_ROUTER_WARP == 0 || expert < n_expert) ? logits[expert] : -INFINITY;
    }

    moe_router_sigmoid<experts_per_thread>(wt);

    // NaN logits select nothing under an iterative argmax, so they become -FLT_MAX as there too
#pragma unroll
    for (int i = 0; i < experts_per_thread; ++i) {
        if (__isnanf(wt[i])) {
            wt[i] = -FLT_MAX;
        }
    }

    [[maybe_unused]] float selection_wt[has_bias ? experts_per_thread : 1];

    if (has_bias) {
#pragma unroll
        for (int i = 0; i < experts_per_thread; ++i) {
            selection_wt[i] = -INFINITY;
        }
#pragma unroll
        for (int i = 0; i < n_expert; i += MOE_ROUTER_WARP) {
            const int expert = i + lane;
            selection_wt[i / MOE_ROUTER_WARP] =
                (n_expert % MOE_ROUTER_WARP == 0 || expert < n_expert) ? wt[i / MOE_ROUTER_WARP] + bias[expert] : -INFINITY;
            if (__isnanf(selection_wt[i / MOE_ROUTER_WARP])) {
                selection_wt[i / MOE_ROUTER_WARP] = -FLT_MAX;
            }
        }
    }

    ids     += (int64_t) t * ids_stride;
    weights += (int64_t) t * n_used;

    float wt_sum = 0.f;

    float output_weights[experts_per_thread];
#pragma unroll
    for (int i = 0; i < experts_per_thread; ++i) {
        output_weights[i] = 0.f;
    }

    for (int k = 0; k < n_used; ++k) {
        float max_val    = wt[0];
        float max_val_s  = has_bias ? selection_wt[0] : 0.0f;
        int   max_expert = lane;

#pragma unroll
        for (int i = 1; i < experts_per_thread; ++i) {
            const int expert = lane + i * MOE_ROUTER_WARP;
            if ((n_expert % MOE_ROUTER_WARP == 0 || expert < n_expert)) {
                if (!has_bias) {
                    if (wt[i] > max_val) {
                        max_val    = wt[i];
                        max_expert = expert;
                    }
                } else if (selection_wt[i] > max_val_s) {
                    max_val    = wt[i];
                    max_val_s  = selection_wt[i];
                    max_expert = expert;
                }
            }
        }

#pragma unroll
        for (int mask = MOE_ROUTER_WARP / 2; mask > 0; mask /= 2) {
            const float val    = ggml_cuda_shfl_xor(max_val, mask, MOE_ROUTER_WARP);
            const float val_s  = has_bias ? ggml_cuda_shfl_xor(max_val_s, mask, MOE_ROUTER_WARP) : 0.0f;
            const int   expert = ggml_cuda_shfl_xor(max_expert, mask, MOE_ROUTER_WARP);
            if (!has_bias) {
                if (val > max_val || (val == max_val && expert < max_expert)) {
                    max_val    = val;
                    max_expert = expert;
                }
            } else {
                if (val_s > max_val_s || (val_s == max_val_s && expert < max_expert)) {
                    max_val    = val;
                    max_val_s  = val_s;
                    max_expert = expert;
                }
            }
        }

        // the weight written is the value without the bias shift
        const float w_out = max_val;

        if ((max_expert & (MOE_ROUTER_WARP - 1)) == lane) {
            if (has_bias) {
                selection_wt[max_expert / MOE_ROUTER_WARP] = -INFINITY;
            } else {
                wt[max_expert / MOE_ROUTER_WARP] = -INFINITY;
            }
        }

        if ((k & (MOE_ROUTER_WARP - 1)) == lane) {
            output_weights[k / MOE_ROUTER_WARP] = w_out;
        }

        if ((max_expert & (MOE_ROUTER_WARP - 1)) == lane) {
            ids[k] = max_expert;
            if (norm.with_norm) {
                wt_sum += w_out;
            }
        }
    }

    if (norm.with_norm) {
        wt_sum              = warp_reduce_sum<MOE_ROUTER_WARP>(wt_sum);
        wt_sum              = norm.norm_eps ? wt_sum + norm.clamp_val : max(wt_sum, norm.clamp_val);
        const float inv_sum = 1.0f / wt_sum;

#pragma unroll
        for (int i = 0; i < experts_per_thread; ++i) {
            output_weights[i] *= inv_sum;
        }
    }

#pragma unroll
    for (int i = 0; i < experts_per_thread; ++i) {
        const int idx = i * MOE_ROUTER_WARP + lane;
        if (idx < n_used) {
            weights[idx] = output_weights[i] * norm.scale_val;
        }
    }

    // the flags go back to zero for the next pass over them (a replay has no memset)
    __syncwarp();
    if (lane == 0) {
        flags[t] = 0;
    }
}

void ggml_cuda_op_moe_router_topk(ggml_backend_cuda_context & ctx,
        const ggml_tensor * cur, const ggml_tensor * router, ggml_tensor * ids, ggml_tensor * weights,
        const ggml_tensor * bias, bool with_norm, bool norm_eps, float clamp_val, float scale_val) {
    GGML_ASSERT(cur->type == GGML_TYPE_F32 && router->type == GGML_TYPE_F32);
    GGML_ASSERT(ids->type == GGML_TYPE_I32 && ids->nb[0] == sizeof(int32_t) && weights->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->ne[0] == 1 && weights->ne[1] == ids->ne[0] && weights->ne[2] == cur->ne[1] && ggml_is_contiguous(weights));

    constexpr int n_expert = 128;

    const int n_embd     = (int) router->ne[0];
    const int n_tokens   = (int) cur->ne[1];
    const int n_used     = (int) ids->ne[0];
    const int ids_stride = (int) (ids->nb[1] / sizeof(int32_t));
    GGML_ASSERT(n_used >= 1 && n_used <= n_expert && ids_stride >= n_used);

    constexpr int chunks_per_token = n_expert / MOE_ROUTER_CHUNK;

    const size_t scratch_bytes = n_tokens*n_expert*sizeof(float) + n_tokens*sizeof(unsigned int);
    ggml_cuda_pool_alloc<char> scratch(ctx.pool(), scratch_bytes);
    float *        logits_scratch = (float *)        scratch.get();
    unsigned int * flags          = (unsigned int *) (scratch.get() + n_tokens*n_expert*sizeof(float));
    CUDA_CHECK(cudaMemsetAsync(flags, 0, n_tokens*sizeof(unsigned int), ctx.stream()));

    const dim3 grid(chunks_per_token, n_tokens, 1);
    const dim3 block(MOE_ROUTER_WARP, MOE_ROUTER_NWARPS, 1);
    const ggml_cuda_kernel_launch_params launch_params(grid, block, 0, ctx.stream());

    moe_router_norm_args norm = { with_norm, norm_eps && with_norm, clamp_val, scale_val };

    // the bias shifts the selection but not the weight it selects, as in the topk_moe kernel
    if (bias) {
        ggml_cuda_kernel_launch(moe_router_topk_cuda<n_expert, true>, launch_params,
            (const float *) cur->data, (const float *) router->data, (int32_t *) ids->data,
            (float *) weights->data, (const float *) bias->data, logits_scratch, flags,
            n_embd, n_used, ids_stride, norm);
    } else {
        ggml_cuda_kernel_launch(moe_router_topk_cuda<n_expert, false>, launch_params,
            (const float *) cur->data, (const float *) router->data, (int32_t *) ids->data,
            (float *) weights->data, (const float *) nullptr, logits_scratch, flags,
            n_embd, n_used, ids_stride, norm);
    }
}

// D2: the attention output tail after build_attn. One launch carries the FWHT that the v_rot Hadamard-hinted MUL_MAT
// stands for, the SIGLU gate against it, and the q8_1 copy the wo mat-vec reads instead of its own staging quantize.

template <int N, int NT>
static __device__ __forceinline__ void moe_fwht_block_transform(float (&reg)[N / NT], float * s) {
    constexpr int warp_size = 32;
    constexpr int NE        = N / NT;
    static_assert(NE >= 1 && N % NT == 0 && NT % warp_size == 0, "bad block shape");

    const int tid  = threadIdx.x;
    const int lane = tid % warp_size;

#pragma unroll
    for (int h = 1; h < warp_size; h *= 2) {
#pragma unroll
        for (int j = 0; j < NE; ++j) {
            const float val  = reg[j];
            const float val2 = ggml_cuda_shfl_xor(val, h, warp_size);
            reg[j] = (lane & h) == 0 ? val + val2 : val2 - val;
        }
    }

#pragma unroll
    for (int h = warp_size; h < NT; h *= 2) {
#pragma unroll
        for (int j = 0; j < NE; ++j) {
            s[j * NT + tid] = reg[j];
        }
        __syncthreads();
#pragma unroll
        for (int j = 0; j < NE; ++j) {
            const float val  = reg[j];
            const float val2 = s[j * NT + (tid ^ h)];
            reg[j] = (tid & h) == 0 ? val + val2 : val2 - val;
        }
        __syncthreads();
    }

#pragma unroll
    for (int h = NT; h < N; h *= 2) {
        const int step = h / NT;
#pragma unroll
        for (int j = 0; j < NE; j += 2 * step) {
#pragma unroll
            for (int k = 0; k < step; ++k) {
                const float x = reg[j + k];
                const float y = reg[j + k + step];
                reg[j + k]        = x + y;
                reg[j + k + step] = x - y;
            }
        }
    }
}

// one block per token row: the transform of fwht.cu's block path, then up/(1+exp(-gate)) per element as glu_fn_kernel
// computes it, staged in shared memory and quantized to q8_1 blocks as the staging kernel would write them
template <int N>
__global__ void fwht_siglu_q8_cuda(const float * x, const float * gate, float * dst_f32, block_q8_1 * q8,
                                   const int64_t tokens, const int q8_row_blocks) {
    constexpr int NT = 256;
    constexpr int NE = N / NT;

    __shared__ float s[N];

    ggml_cuda_pdl_sync();

    const int64_t r = blockIdx.x;
    if (r >= tokens) {
        return;
    }

    x         += r * N;
    gate      += r * N;
    dst_f32   += r * N;
    block_q8_1 * q8_row = q8 + (int64_t) r * q8_row_blocks;

    const int tid = threadIdx.x;

    float reg[NE];
#pragma unroll
    for (int i = 0; i < NE; ++i) {
        // the FWHT's scale, as the Hadamard-hinted matmul applies it
        reg[i] = x[i * NT + tid] * (1.0f / sqrtf((float) N));
    }

    moe_fwht_block_transform<N, NT>(reg, s);

    float v[NE];
#pragma unroll
    for (int i = 0; i < NE; ++i) {
        const int e = i * NT + tid;
        v[i] = reg[i] / (1.0f + expf(-gate[e]));
        dst_f32[e] = v[i];
        s[e]       = v[i];
    }
    __syncthreads();

    constexpr int row_blocks_full = N / QK8_1;

    if (tid < 32) {
        // one warp per q8_1 block, matching the staging kernel lane by lane; the padded tail quantizes to zeros
        for (int b = 0; b < q8_row_blocks; ++b) {
            const bool real = b < row_blocks_full;
            const float xi = real ? s[b * QK8_1 + tid] : 0.0f;

            float amax = fabsf(xi);
            amax = warp_reduce_max<32>(amax);

            const float d   = amax / 127.0f;
            const int   q   = amax == 0.0f ? 0 : roundf(xi / d);
            const int   sum = warp_reduce_sum<32>(q);

            q8_row[b].qs[tid] = q;

            if (real && tid == 0) {
                q8_row[b].ds = make_half2(d, d * sum);
            } else if (!real && tid == 0) {
                // every lane of a padding block holds zero, so d and the sum are zero too
                q8_row[b].ds = make_half2(0.0f, 0.0f);
            }
        }
    }
}

void ggml_cuda_op_fwht_siglu_q8(ggml_backend_cuda_context & ctx,
        const ggml_tensor * x, const ggml_tensor * gate, ggml_tensor * dst_f32,
        block_q8_1 * q8_dst, int64_t q8_row_blocks) {
    GGML_ASSERT(x->type == GGML_TYPE_F32 && gate->type == GGML_TYPE_F32 && dst_f32->type == GGML_TYPE_F32);

    const int     n      = (int) dst_f32->ne[0];
    const int64_t tokens = ggml_nelements(dst_f32) / n;

    GGML_ASSERT(q8_row_blocks >= (n + QK8_1 - 1) / QK8_1); // the padded tail is written whole

    cudaStream_t stream = ctx.stream();

#define MOE_FWHT_SIGLU_CASE(NN) \
    case NN: { \
        const dim3 grid((unsigned) tokens, 1, 1), b(256, 1, 1); \
        const ggml_cuda_kernel_launch_params lp(grid, b, 0, stream); \
        ggml_cuda_kernel_launch(fwht_siglu_q8_cuda<NN>, lp, (const float *) x->data, (const float *) gate->data, \
            (float *) dst_f32->data, q8_dst, tokens, (int) q8_row_blocks); \
        return; \
    }
    switch (n) {
        MOE_FWHT_SIGLU_CASE(512)
        MOE_FWHT_SIGLU_CASE(1024)
        MOE_FWHT_SIGLU_CASE(2048)
        MOE_FWHT_SIGLU_CASE(4096)
        default:
            GGML_ABORT("the matcher only takes block-path widths");
    }
#undef MOE_FWHT_SIGLU_CASE
}

// One warp computes 128 channels in registers (two 64-wide or one 128-wide FWHT block, or unrotated) before any store.
template <int rot_width>
static __device__ __forceinline__ void attn_siglu_eval_128(
        const float * x_base, const float * gate_base, const float scale, float (&reg)[4]) {
    static_assert(rot_width == 0 || rot_width == 64 || rot_width == 128, "invalid rot_width");
    const int lane = threadIdx.x % WARP_SIZE;

    float g[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float xv = x_base[j * WARP_SIZE + lane];
        reg[j] = rot_width > 0 ? xv * scale : xv;
        g[j]   = gate_base[j * WARP_SIZE + lane];
    }

    if constexpr (rot_width > 0) {
#pragma unroll
        for (int h = 1; h < WARP_SIZE; h *= 2) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float val  = reg[j];
                const float val2 = ggml_cuda_shfl_xor(val, h, WARP_SIZE);
                reg[j] = (lane & h) == 0 ? val + val2 : val2 - val;
            }
        }
#pragma unroll
        for (int j = 0; j < 4; j += 2) {
            const float x0 = reg[j];
            const float y0 = reg[j + 1];
            reg[j]     = x0 + y0;
            reg[j + 1] = x0 - y0;
        }
        if constexpr (rot_width == 128) {
#pragma unroll
            for (int k = 0; k < 2; ++k) {
                const float x0 = reg[k];
                const float y0 = reg[k + 2];
                reg[k]     = x0 + y0;
                reg[k + 2] = x0 - y0;
            }
        }
    }

#pragma unroll
    for (int j = 0; j < 4; ++j) {
        reg[j] = reg[j] / (1.0f + expf(-g[j]));
    }
}

// Matches quantize_mmq_few_store in quantize.cu instruction for instruction.
template <mmq_q8_1_ds_layout ds_layout>
static __device__ __forceinline__ void attn_siglu_mmq_few_store(const float4 xi, int4 * yg, const int iu, const int g) {
    float amax = fmaxf(fmaxf(fabsf(xi.x), fabsf(xi.y)), fmaxf(fabsf(xi.z), fabsf(xi.w)));
    amax = warp_reduce_max<8>(amax);

    const float d_inv = 127.0f / amax;
    char4 q;
    q.x = roundf(xi.x*d_inv);
    q.y = roundf(xi.y*d_inv);
    q.z = roundf(xi.z*d_inv);
    q.w = roundf(xi.w*d_inv);
    const float d = 1.0f / d_inv;

    [[maybe_unused]] float sum = 0.0f;
    if constexpr (ds_layout != MMQ_Q8_1_DS_LAYOUT_D4) {
        sum = d*warp_reduce_sum<8>(float(q.x + q.y + q.z + q.w));
    }

    const int s = iu / QK8_1;
    const int k = iu % QK8_1;
    ((int *) (yg + (s/2)*WARP_SIZE + 4*g + (k % 16)/4))[2*(s % 2) + k/16] = *(const int *) &q;
    if (k == 0) {
        int4 * ds = yg + 4*WARP_SIZE + 4*(g/2) + s/2;
        if constexpr (ds_layout == MMQ_Q8_1_DS_LAYOUT_D4) {
            ((float *) ds)[2*(s % 2) + g % 2] = d;
        } else {
            ((half2 *) ds)[2*(s % 2) + g % 2] = make_half2(d, sum);
        }
    }
}

template <int rot_width>
__launch_bounds__(64)
static __global__ void attn_siglu_f32_cuda(
        const float * x, const float * gate, float * dst_f32, const int ncols, const float scale) {
    ggml_cuda_pdl_sync();

    const int64_t row_off = (int64_t) blockIdx.x * ncols;
    const int     base    = (int) blockIdx.y * QK_K + (threadIdx.x / WARP_SIZE) * 128;
    const int     lane    = threadIdx.x % WARP_SIZE;

    float reg[4];
    attn_siglu_eval_128<rot_width>(x + row_off + base, gate + row_off + base, scale, reg);

    float * dst_base = dst_f32 + row_off + base;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        dst_base[j * WARP_SIZE + lane] = reg[j];
    }
}

template <int rot_width>
__launch_bounds__(64)
static __global__ void attn_siglu_q8_1_cuda(
        const float * x, const float * gate, float * dst_f32, block_q8_1 * q8,
        const int ncols, const int real_units, const int q8_row_blocks, const float scale) {
    ggml_cuda_pdl_lc();
    ggml_cuda_pdl_sync();

    const int64_t col  = blockIdx.x;
    const int     u    = blockIdx.y;
    const int     warp = threadIdx.x / WARP_SIZE;
    const int     lane = threadIdx.x % WARP_SIZE;

    block_q8_1 * q8_w = q8 + col * q8_row_blocks + u * 8 + warp * 4;
    if (u >= real_units) {
#pragma unroll
        for (int b = 0; b < 4; ++b) {
            q8_w[b].qs[lane] = 0;
            if (lane == 0) {
                q8_w[b].ds = make_half2(0.0f, 0.0f);
            }
        }
        return;
    }

    const int64_t row_off = col * ncols;
    const int     base    = u * QK_K + warp * 128;

    float reg[4];
    attn_siglu_eval_128<rot_width>(x + row_off + base, gate + row_off + base, scale, reg);

    if (dst_f32 != nullptr) {
        float * dst_base = dst_f32 + row_off + base;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            dst_base[j * WARP_SIZE + lane] = reg[j];
        }
    }

#pragma unroll
    for (int b = 0; b < 4; ++b) {
        const float  xi   = reg[b];
        const float  amax = warp_reduce_max<QK8_1>(fabsf(xi));
        const float  d    = amax / 127.0f;
        const int8_t q    = amax == 0.0f ? 0 : roundf(xi / d);
        const int    sum  = warp_reduce_sum<QK8_1>((int) q);
        q8_w[b].qs[lane] = q;
        if (lane == 0) {
            q8_w[b].ds = make_half2(d, d * sum);
        }
    }
}

template <int rot_width, mmq_q8_1_ds_layout ds_layout>
__launch_bounds__(64)
static __global__ void attn_siglu_mmq_few_cuda(
        const float * x, const float * gate, float * dst_f32, int4 * vy,
        const int ncols, const int tokens, const float scale) {
    __shared__ __align__(16) float s_unit[QK_K];

    ggml_cuda_pdl_lc();
    ggml_cuda_pdl_sync();

    const int col  = blockIdx.x;
    const int u    = blockIdx.y;
    const int tid  = threadIdx.x;
    const int warp = tid / WARP_SIZE;
    const int lane = tid % WARP_SIZE;

    float4 xi = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (col < tokens) {
        const int64_t row_off = (int64_t) col * ncols;
        const int     base    = u * QK_K + warp * 128;

        float reg[4];
        attn_siglu_eval_128<rot_width>(x + row_off + base, gate + row_off + base, scale, reg);

        if (dst_f32 != nullptr) {
            float * dst_base = dst_f32 + row_off + base;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                dst_base[j * WARP_SIZE + lane] = reg[j];
            }
        }

        float * s_w = s_unit + warp * 128;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            s_w[j * WARP_SIZE + lane] = reg[j];
        }
        __syncwarp();
        xi = ((const float4 *) s_w)[lane];
    }

    const int ncg = gridDim.x / MMQ_FEW_NCOLS;
    int4 * yg = vy + ((int64_t) u * ncg + col / MMQ_FEW_NCOLS) * MMQ_FEW_Q8_1_INT4;
    attn_siglu_mmq_few_store<ds_layout>(xi, yg, tid * 4, col % MMQ_FEW_NCOLS);
}

template <int rot_width>
static void attn_siglu_fuse_launch_rot(ggml_backend_cuda_context & ctx,
        const float * x_d, const float * gate_d, float * dst_f32,
        const int ncols, const int64_t tokens, const float scale,
        const ggml_cuda_siglu_q8_mode q8_mode, const ggml_type wo_type, void * q8_dst) {
    cudaStream_t stream = ctx.stream();
    const dim3 block(64, 1, 1);

    if (q8_mode == GGML_CUDA_SIGLU_Q8_NONE) {
        GGML_ASSERT(dst_f32 != nullptr);
        const dim3 grid((unsigned) tokens, (unsigned) (ncols / QK_K), 1);
        const ggml_cuda_kernel_launch_params lp(grid, block, 0, stream);
        ggml_cuda_kernel_launch(attn_siglu_f32_cuda<rot_width>, lp, x_d, gate_d, dst_f32, ncols, scale);
        return;
    }

    GGML_ASSERT(q8_dst != nullptr);
    if (q8_mode == GGML_CUDA_SIGLU_Q8_MMVQ) {
        static_assert(MATRIX_ROW_PADDING % QK_K == 0, "MATRIX_ROW_PADDING must be a multiple of QK_K");
        const int real_units    = ncols / QK_K;
        const int pad_units     = GGML_PAD(ncols, MATRIX_ROW_PADDING) / QK_K;
        const int q8_row_blocks = pad_units * (QK_K / QK8_1);
        const dim3 grid((unsigned) tokens, (unsigned) pad_units, 1);
        const ggml_cuda_kernel_launch_params lp(grid, block, 0, stream);
        ggml_cuda_kernel_launch(attn_siglu_q8_1_cuda<rot_width>, lp,
            x_d, gate_d, dst_f32, (block_q8_1 *) q8_dst, ncols, real_units, q8_row_blocks, scale);
        return;
    }

    GGML_ASSERT(q8_mode == GGML_CUDA_SIGLU_Q8_MMQ_FEW && tokens <= INT_MAX - MMQ_FEW_NCOLS);
    const dim3 grid((unsigned) GGML_PAD(tokens, MMQ_FEW_NCOLS), (unsigned) (ncols / QK_K), 1);
    const ggml_cuda_kernel_launch_params lp(grid, block, 0, stream);
    switch (mmq_get_q8_1_ds_layout(wo_type)) {
        case MMQ_Q8_1_DS_LAYOUT_D4:
            ggml_cuda_kernel_launch((attn_siglu_mmq_few_cuda<rot_width, MMQ_Q8_1_DS_LAYOUT_D4>), lp,
                x_d, gate_d, dst_f32, (int4 *) q8_dst, ncols, (int) tokens, scale);
            break;
        case MMQ_Q8_1_DS_LAYOUT_DS4:
            ggml_cuda_kernel_launch((attn_siglu_mmq_few_cuda<rot_width, MMQ_Q8_1_DS_LAYOUT_DS4>), lp,
                x_d, gate_d, dst_f32, (int4 *) q8_dst, ncols, (int) tokens, scale);
            break;
        default:
            GGML_ABORT("unsupported ds_layout for attn_siglu_mmq_few_cuda");
    }
}

void ggml_cuda_op_attn_siglu_fuse(ggml_backend_cuda_context & ctx,
        const ggml_tensor * x, const ggml_tensor * gate, float * dst_f32,
        const int rot_width, const int ncols, const int64_t tokens,
        const ggml_cuda_siglu_q8_mode q8_mode, const ggml_type wo_type, void * q8_dst) {
    GGML_ASSERT(x->type == GGML_TYPE_F32 && gate->type == GGML_TYPE_F32);
    GGML_ASSERT(ncols >= QK_K && ncols % QK_K == 0 && tokens >= 1);
    GGML_ASSERT(rot_width == 0 || ((rot_width == 64 || rot_width == 128) && ncols % rot_width == 0));

    const float * x_d    = (const float *) x->data;
    const float * gate_d = (const float *) gate->data;
    const float   scale  = rot_width > 0 ? 1.0f / sqrtf((float) rot_width) : 1.0f;

    switch (rot_width) {
        case 0:
            attn_siglu_fuse_launch_rot<0>(ctx, x_d, gate_d, dst_f32, ncols, tokens, scale, q8_mode, wo_type, q8_dst);
            break;
        case 64:
            attn_siglu_fuse_launch_rot<64>(ctx, x_d, gate_d, dst_f32, ncols, tokens, scale, q8_mode, wo_type, q8_dst);
            break;
        case 128:
            attn_siglu_fuse_launch_rot<128>(ctx, x_d, gate_d, dst_f32, ncols, tokens, scale, q8_mode, wo_type, q8_dst);
            break;
        default:
            GGML_ABORT("unsupported rot_width");
    }
}
