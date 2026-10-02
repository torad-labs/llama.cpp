#include "common.cuh"
#include "fattn-common.cuh"
#include "fattn-mma-f16.cuh"
#include "fattn-tile.cuh"
#include "fattn-vec.cuh"
#include "fattn.cuh"

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
// A block a mask row: its finite entries' cells, ascending, into the row's n_kv_max indices, -1 past its count (a count
// over n_kv_max keeps the first n_kv_max: the bound is the graph's to keep).
// Upstream's scan (8e93a9773): 256 threads, 2048 columns a round of scalar loads. A decode's one row is one block, so
// a round is a memory round trip: 11 us at 32K columns. GGML_CUDA_FATTN_SPARSE_SCAN_LEGACY=1, or a mask row not
// 16-byte aligned.
__launch_bounds__(256, 1)
static __global__ void flash_attn_mask_to_sparse_indices_legacy(
        const half * mask_ptr, int32_t * indices_ptr, const int ne30, const int n_kv_max,
        const int64_t s31, const int64_t s33) {
    ggml_cuda_pdl_sync();

    constexpr int values_per_lane = 8;
    const int tid      = threadIdx.x;
    const int warp     = tid / WARP_SIZE;
    const int lane     = tid % WARP_SIZE;
    const int sequence = blockIdx.y;
    const int query    = blockIdx.x;

    const half * mask = mask_ptr + sequence*s33 + query*s31;
    int32_t * indices = indices_ptr + (int64_t(sequence)*gridDim.x + query)*n_kv_max;

    __shared__ int warp_offsets[256/WARP_SIZE];
    __shared__ int row_count;
    __shared__ int chunk_count;

    if (tid == 0) {
        row_count = 0;
    }
    __syncthreads();

    for (int i0 = 0; i0 < ne30; i0 += blockDim.x*values_per_lane) {
        uint32_t selected_warp[values_per_lane];
        int warp_count = 0;
#pragma unroll
        for (int item = 0; item < values_per_lane; ++item) {
            const int i = i0 + (warp*values_per_lane + item)*WARP_SIZE + lane;
            const bool selected = i < ne30 && isfinite(__half2float(mask[i]));
            selected_warp[item] = __ballot_sync(0xFFFFFFFF, selected);
            warp_count += __popc(selected_warp[item]);
        }

        if (lane == 0) {
            warp_offsets[warp] = warp_count;
        }
        __syncthreads();

        if (tid == 0) {
            int offset = 0;
#pragma unroll
            for (int iw = 0; iw < 256/WARP_SIZE; ++iw) {
                const int count = warp_offsets[iw];
                warp_offsets[iw] = offset;
                offset += count;
            }
            chunk_count = offset;
        }
        __syncthreads();

        const uint32_t lane_mask = lane == 0 ? 0 : (1u << lane) - 1;
        int warp_item_offset = 0;
#pragma unroll
        for (int item = 0; item < values_per_lane; ++item) {
            const int i = i0 + (warp*values_per_lane + item)*WARP_SIZE + lane;
            const int dst = row_count + warp_offsets[warp] + warp_item_offset + __popc(selected_warp[item] & lane_mask);
            if ((selected_warp[item] & (uint32_t(1) << lane)) && dst < n_kv_max) {
                indices[dst] = i;
            }
            warp_item_offset += __popc(selected_warp[item]);
        }
        __syncthreads();

        if (tid == 0) {
            row_count += chunk_count;
        }
        __syncthreads();
    }

    const int count = row_count;
    for (int i = count + tid; i < n_kv_max; i += blockDim.x) {
        indices[i] = -1;
    }
    __syncthreads();

    // the dependent grid reads indices, signal once the row is complete
    ggml_cuda_pdl_lc();
}

// The same lists, one block of 1024 threads a row: a thread tests 32 consecutive columns from four 16-byte loads issued
// together, so up to 32768 columns are one round trip, and a block-wide scan of the threads' counts places each
// thread's cells after the lower columns'. The row is 16-byte aligned and ne30 % 8 == 0.
static constexpr int sparse_scan_threads = 1024;
static constexpr int sparse_scan_cols    = 32;

__launch_bounds__(sparse_scan_threads, 1)
static __global__ void flash_attn_mask_to_sparse_indices(
        const half * mask_ptr, int32_t * indices_ptr, const int ne30, const int n_kv_max,
        const int64_t s31, const int64_t s33) {
    ggml_cuda_pdl_sync();

    constexpr int nwarps = sparse_scan_threads/WARP_SIZE;
    static_assert(nwarps == WARP_SIZE, "one warp scans the warps' sums");
    const int tid      = threadIdx.x;
    const int warp     = tid / WARP_SIZE;
    const int lane     = tid % WARP_SIZE;
    const int sequence = blockIdx.y;
    const int query    = blockIdx.x;

    const half * mask = mask_ptr + sequence*s33 + query*s31;
    int32_t * indices = indices_ptr + (int64_t(sequence)*gridDim.x + query)*n_kv_max;

    __shared__ int warp_sums[nwarps];
    int row_count = 0;

    for (int i0 = 0; i0 < ne30; i0 += sparse_scan_threads*sparse_scan_cols) {
        const int c0 = i0 + tid*sparse_scan_cols;

        // 8 halves a load, -inf past the row (ne30 % 8 == 0: a load is all in or all out)
        uint4 v[sparse_scan_cols/8];
#pragma unroll
        for (int j = 0; j < sparse_scan_cols/8; ++j) {
            v[j] = c0 + 8*j < ne30 ? ((const uint4 *) (mask + c0))[j] : make_uint4(0xFC00FC00u, 0xFC00FC00u, 0xFC00FC00u, 0xFC00FC00u);
        }

        // bit c: column c0 + c is finite (a half is inf or nan exactly when its exponent bits are all set)
        uint32_t finite = 0;
#pragma unroll
        for (int j = 0; j < sparse_scan_cols/8; ++j) {
            const uint32_t w[4] = {v[j].x, v[j].y, v[j].z, v[j].w};
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                finite |= uint32_t((w[k] & 0x00007C00u) != 0x00007C00u) << (8*j + 2*k + 0);
                finite |= uint32_t((w[k] & 0x7C000000u) != 0x7C000000u) << (8*j + 2*k + 1);
            }
        }
        const int count = __popc(finite);

        // inclusive scan of the counts in the warp, then of the warps' sums
        int incl = count;
#pragma unroll
        for (int offset = 1; offset < WARP_SIZE; offset *= 2) {
            const int t = __shfl_up_sync(0xFFFFFFFF, incl, offset);
            incl += lane >= offset ? t : 0;
        }
        if (lane == WARP_SIZE - 1) {
            warp_sums[warp] = incl;
        }
        __syncthreads();
        if (warp == 0) {
            int s = warp_sums[lane];
#pragma unroll
            for (int offset = 1; offset < WARP_SIZE; offset *= 2) {
                const int t = __shfl_up_sync(0xFFFFFFFF, s, offset);
                s += lane >= offset ? t : 0;
            }
            warp_sums[lane] = s;
        }
        __syncthreads();

        int dst = row_count + (warp == 0 ? 0 : warp_sums[warp - 1]) + incl - count;
        while (finite != 0 && dst < n_kv_max) {
            indices[dst++] = c0 + __ffs(finite) - 1;
            finite &= finite - 1;
        }
        row_count += warp_sums[nwarps - 1];
        __syncthreads(); // the next chunk rewrites warp_sums
    }

    for (int i = row_count + tid; i < n_kv_max; i += sparse_scan_threads) {
        indices[i] = -1;
    }
    __syncthreads();

    // the dependent grid reads indices, signal once the row is complete
    ggml_cuda_pdl_lc();
}
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

void ggml_cuda_flash_attn_ext_compact_mask(
        const ggml_tensor * mask, int32_t * indices, int32_t n_kv_max, cudaStream_t stream) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    GGML_UNUSED_VARS(mask, indices, n_kv_max, stream);
    GGML_ABORT("sparse flash attention is only supported on NVIDIA CUDA");
#else
    GGML_ASSERT(mask->type == GGML_TYPE_F16);
    static const bool scan_legacy = ggml_env_switch("GGML_CUDA_FATTN_SPARSE_SCAN_LEGACY");
    const int64_t s31 = mask->nb[1] / sizeof(half);
    const int64_t s33 = mask->nb[3] / sizeof(half);
    const bool aligned = (uintptr_t) mask->data % 16 == 0 && mask->ne[0] % 8 == 0 && s31 % 8 == 0 && s33 % 8 == 0;
    const bool wide = !scan_legacy && aligned;
    const dim3 blocks_num(mask->ne[1], mask->ne[3], 1);
    const dim3 block_dim(wide ? sparse_scan_threads : 256, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params(blocks_num, block_dim, 0, stream);
    ggml_cuda_kernel_launch(wide ? flash_attn_mask_to_sparse_indices : flash_attn_mask_to_sparse_indices_legacy, launch_params,
        (const half *) mask->data, indices, int(mask->ne[0]), n_kv_max, s31, s33);
    CUDA_CHECK(cudaGetLastError());
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
}

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
// One thread a q8_0 block of one slot of one (query row, sequence): the slot's cell, dequantized with the converter's
// arithmetic (dequantize_block_q8_0_f16: one half multiply of the exact int8 by the block's scale) into the cell's place in
// the dense f16 copy. Slots [0, n_kv_max) are the row's indices (-1: none), slot n_kv_max is cell 0, which the K tile load
// reads for a slot with no cell when it copies asynchronously (flash_attn_ext_f16_load_tile, the single stage's): left
// unconverted it is whatever the scratch held, and the masked cell's 0 x V turns a non-finite one into NaN.
static __global__ void flash_attn_gather_k_q8_0(
        const char * __restrict__ K, const int32_t * __restrict__ indices, half * __restrict__ K_f16,
        const size_t nb11, const size_t nb13, const int64_t ne1, const int nblocks, const int n_kv_max, const int ne31) {
    const int64_t item = int64_t(blockIdx.x)*blockDim.x + threadIdx.x;
    const int64_t slot = item / nblocks;
    if (slot > n_kv_max) {
        return;
    }
    const int block    = item % nblocks;
    const int row      = blockIdx.y;
    const int sequence = blockIdx.z;

    const int32_t cell = slot < n_kv_max ? indices[(int64_t(sequence)*ne31 + row)*n_kv_max + slot] : 0;
    if (cell < 0) {
        return;
    }

    const block_q8_0 * b = (const block_q8_0 *) (K + sequence*nb13 + cell*nb11) + block;
    const half d  = b->d;
    const char2 * qs = (const char2 *) b->qs;
    half2 * y = (half2 *) (K_f16 + (sequence*ne1 + cell)*(int64_t(nblocks)*QK8_0) + block*QK8_0);
#pragma unroll
    for (int i = 0; i < QK8_0/2; ++i) {
        y[i] = __hmul2(make_half2(qs[i].x, qs[i].y), __half2half2(d));
    }
}
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

void ggml_cuda_flash_attn_ext_gather_k_q8_0(
        const char * K, size_t nb11, size_t nb13, int64_t ne0, int64_t ne1, half * K_f16, const int32_t * indices,
        int32_t n_kv_max, int64_t ne31, int64_t n_tokens, int64_t n_seq, cudaStream_t stream) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    GGML_UNUSED_VARS(K, nb11, nb13, ne0, ne1, K_f16, indices, n_kv_max, ne31, n_tokens, n_seq, stream);
    GGML_ABORT("sparse flash attention is only supported on NVIDIA CUDA");
#else
    GGML_ASSERT(ne0 % QK8_0 == 0);
    const int nblocks = ne0 / QK8_0;
    const int64_t items = int64_t(n_kv_max + 1)*nblocks;
    const int threads = 256;
    const dim3 blocks_num((items + threads - 1)/threads, n_tokens, n_seq);
    flash_attn_gather_k_q8_0<<<blocks_num, threads, 0, stream>>>(K, indices, K_f16, nb11, nb13, ne1, nblocks, n_kv_max, int(ne31));
    CUDA_CHECK(cudaGetLastError());
#endif // defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
}

// GGML_CUDA_FATTN_SPARSE_LEGACY=1: the hint is ignored, the kernel reads the whole cache under the mask.
// The gather reads n_kv_max cells a query; the dense kernel reads the cache once for up to 64/ncols2 queries. The gather
// wins where the cache is at least those queries' cells (GLM-5.3's DSA shape, 32 heads on the latent, RTX 5080 and
// 5070 Ti: at 0.25-0.99x the dense time wherever K >= n_gather, at 1.13-1.27x wherever it is under; upstream's 2x margin
// gave up a verify's 3 tokens at 8K cells, 0.73x, and a prefill at 16K, 0.63-0.72x).
// GGML_CUDA_FATTN_SPARSE_HEADS_SWITCHOVER=1: where the 32-head tile takes the launch (sparse_heads_take), the gather reads a
// query's cells once for 32 heads, so it wins from 2 queries' cells, 4160 for GLM's 2080. Between that and the 8-head
// tile's switch-over it replaces the masked dense kernel, a different summation: not bit for bit.
bool ggml_cuda_flash_attn_ext_mma_f16_shall_use_sparse(ggml_backend_cuda_context & ctx, const ggml_tensor * dst, const int ncols2) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    GGML_UNUSED_VARS(ctx, dst, ncols2);
    return false;
#else
    static const bool sparse_legacy    = ggml_env_switch("GGML_CUDA_FATTN_SPARSE_LEGACY");
    static const bool heads_switchover = ggml_env_switch("GGML_CUDA_FATTN_SPARSE_HEADS_SWITCHOVER");

    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * mask = dst->src[3];
    const int cc = ggml_cuda_info().devices[ctx.device].cc;

    float max_bias = 0.0f;
    float logit_softcap = 0.0f;
    memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));

    const int32_t n_kv_max = ggml_flash_attn_ext_get_n_kv_max(dst);
    const int ncols2_gather = heads_switchover && ncols2 == 8 && ggml_cuda_flash_attn_ext_mma_f16_sparse_heads_take(ctx, dst) ? 32 : ncols2;
    const int64_t n_gather = std::min<int64_t>(Q->ne[1], 64/ncols2_gather) * n_kv_max;
    return !sparse_legacy && GGML_CUDA_CC_IS_NVIDIA(cc) && turing_mma_available(cc) &&
        mask != nullptr && mask->type == GGML_TYPE_F16 && n_kv_max > 0 && max_bias == 0.0f && logit_softcap == 0.0f &&
        mask->ne[0] == K->ne[1] && mask->ne[1] >= Q->ne[1] && mask->ne[2] == 1 &&
        K->ne[1] >= std::max<int64_t>(4096, n_gather);
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
}

// The 32-head tile takes a sparse launch of GLM's 512-wide latent with 32 heads or a multiple on it, where the 8-head tiles
// number at least 4 for each block of the 8-head launch's stream-k grid: the tiles that grid splits, which its kernel redoes
// after the head tile, are then a few of them. The block count is the bound the shared memory sets on the blocks an SM holds,
// at least the occupancy launch_fattn finds, so the test only errs toward the 8-head launch.
// GGML_CUDA_FATTN_SPARSE_HEADS_LEGACY=1: the 8-head launch alone.
bool ggml_cuda_flash_attn_ext_mma_f16_sparse_heads_take(ggml_backend_cuda_context & ctx, const ggml_tensor * dst) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    GGML_UNUSED_VARS(ctx, dst);
    return false;
#else
    static const bool heads_legacy = ggml_env_switch("GGML_CUDA_FATTN_SPARSE_HEADS_LEGACY");

    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    const int id = ctx.device;
    const ggml_cuda_device_info::cuda_device_info & info = ggml_cuda_info().devices[id];

    const int64_t gqa_ratio = Q->ne[2] / K->ne[2];
    if (heads_legacy || !GGML_CUDA_CC_IS_NVIDIA(info.cc) || !turing_mma_available(info.cc) ||
            Q->ne[0] != 512 || V->ne[0] != 512 || gqa_ratio % 32 != 0) {
        return false;
    }

    static int smem_sm[GGML_CUDA_MAX_DEVICES]    = {0}; // shared memory an SM holds, and what each block reserves of it
    static int smem_block[GGML_CUDA_MAX_DEVICES] = {0};
    if (smem_sm[id] == 0) {
        CUDA_CHECK(cudaDeviceGetAttribute(&smem_block[id], cudaDevAttrReservedSharedMemoryPerBlock, ggml_cuda_get_device()));
        CUDA_CHECK(cudaDeviceGetAttribute(&smem_sm[id], cudaDevAttrMaxSharedMemoryPerMultiprocessor, ggml_cuda_get_device()));
    }
    const fattn_mma_geometry geom = ggml_cuda_fattn_mma_get_geometry<512, 512, 1, 8, GGML_TYPE_F16, GGML_TYPE_F16>(info.cc, info.warp_size);
    const int64_t blocks = int64_t(smem_sm[id] / (geom.nbytes_shared + smem_block[id])) * info.nsm;
    const int64_t ntiles = Q->ne[1] * (gqa_ratio/8) * K->ne[2] * Q->ne[3];
    return ntiles >= 4*blocks;
#endif // defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
}

template <int DKQ, int DV, int ncols2>
static void ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const ggml_tensor * Q = dst->src[0];

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    // the sparse variant is one token a tile, whatever the batch
    if constexpr (ggml_cuda_flash_attn_ext_mma_f16_may_use_sparse(DKQ, DV, 1, ncols2)) {
        if (ggml_cuda_flash_attn_ext_mma_f16_shall_use_sparse(ctx, dst, ncols2)) {
            ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 1, ncols2>(ctx, dst);
            return;
        }
    }
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

    if constexpr (ncols2 <= 8) {
        if (turing_mma_available(cc) && Q->ne[1] <= 8/ncols2) {
            ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 8/ncols2, ncols2>(ctx, dst);
            return;
        }
    }

    if constexpr (ncols2 <= 16) {
        if (Q->ne[1] <= 16/ncols2) {
            ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 16/ncols2, ncols2>(ctx, dst);
            return;
        }
    }

    if (Q->ne[1] <= 32/ncols2 || (GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) == GGML_CUDA_CC_TURING) ||
            (GGML_CUDA_CC_IS_AMD(cc) && DKQ > 256)) {
        ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 32/ncols2, ncols2>(ctx, dst);
        return;
    }

    ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 64/ncols2, ncols2>(ctx, dst);
}

template <int DKQ, int DV>
static void ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const ggml_tensor * KQV  = dst;
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    float max_bias = 0.0f;
    memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

    // Edge cases like no mask, ALiBi, unpadded K/V, or misaligned addresses for large data transfers
    //     are put into the template specialization without GQA optimizations.
    bool use_gqa_opt = mask && max_bias == 0.0f && K->ne[1] % FATTN_KQ_STRIDE == 0;
    for (const ggml_tensor * t : {Q, K, V, mask}) {
        if (t == nullptr || ggml_is_quantized(t->type)) {
            continue;
        }
        for (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
            if (t->nb[i] % 16 != 0) {
                use_gqa_opt = false;
                break;
            }
        }
    }

    GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
    const int gqa_ratio = Q->ne[2] / K->ne[2];

    // On Volta the GQA optimizations aren't as impactful vs. minimizing wasted compute:
    if (cc == GGML_CUDA_CC_VOLTA) {
        if (use_gqa_opt && gqa_ratio % 8 == 0) {
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 8>(ctx, dst);
            return;
        }

        if (use_gqa_opt && gqa_ratio % 4 == 0) {
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 4>(ctx, dst);
            return;
        }

        if constexpr (DKQ <= 256) {
            if (use_gqa_opt && gqa_ratio % 2 == 0) {
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 2>(ctx, dst);
                return;
            }

            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 1>(ctx, dst);
            return;
        } else {
            GGML_ABORT("fatal error");
        }
    }

    if (use_gqa_opt && gqa_ratio > 4) {
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 8>(ctx, dst);
        return;
    }

    if (use_gqa_opt && gqa_ratio > 2) {
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 4>(ctx, dst);
        return;
    }

    if (use_gqa_opt && gqa_ratio > 1) {
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 2>(ctx, dst);
        return;
    }

    if constexpr (DKQ <= 256) {
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 1>(ctx, dst);
    } else {
        GGML_ABORT("fatal error");
    }
}

// q4_0 or q8_0 K/V (both of one type) with head size 256 and GQA > 4 are read by the MMA kernel directly, without the
// f16 copy of K and V. Must match the ncols2 == 8 choice of ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2, it also
// sizes the f16 buffer. GGML_CUDA_FATTN_Q4_0_LEGACY / GGML_CUDA_FATTN_Q8_0_LEGACY restore the stock kernels for that
// type (the f16 copy, the vector kernel for decode): the A/B for each path within one binary, and its off switch.
static bool ggml_cuda_fattn_mma_raw(const int cc, const ggml_tensor * dst) {
    static const bool legacy_q4_0 = ggml_env_switch("GGML_CUDA_FATTN_Q4_0_LEGACY");
    static const bool legacy_q8_0 = ggml_env_switch("GGML_CUDA_FATTN_Q8_0_LEGACY");

    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    if (K->type != V->type || K->ne[0] != 256 || V->ne[0] != 256) {
        return false;
    }
    switch (K->type) {
        case GGML_TYPE_Q4_0:
            if (legacy_q4_0) {
                return false;
            }
            break;
        case GGML_TYPE_Q8_0:
            if (legacy_q8_0) {
                return false;
            }
            break;
        default:
            return false;
    }
    if (!GGML_CUDA_CC_IS_NVIDIA(cc) || !ampere_mma_available(cc)) {
        return false; // needs cp.async and the int8 m16n8k32 MMA
    }

    float max_bias = 0.0f;
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));
    if (!mask || max_bias != 0.0f || K->ne[1] % FATTN_KQ_STRIDE != 0 || Q->ne[2] / K->ne[2] <= 4) {
        return false;
    }
    for (const ggml_tensor * t : {Q, K, V, mask}) {
        for (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
            if (t->nb[i] % 16 != 0) {
                return false; // Q and mask as in use_gqa_opt, q4_0 rows are copied in 16 byte chunks
            }
        }
    }
    return true;
}

template <int DKQ, int DV, ggml_type type_KV>
static void ggml_cuda_flash_attn_ext_mma_raw_switch_ncols1(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    constexpr int ncols2 = 8;
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];

    GGML_ASSERT((uintptr_t) dst->src[1]->data % 16 == 0 && (uintptr_t) dst->src[2]->data % 16 == 0);

    // An 8 head tile pads GQA 6 with 2 empty heads, a quarter of its work. With 32 or more tokens a 32 token x 2 head
    // tile of the same width has no padding, and every K/V tile it reads serves 64 Q columns instead of 48.
    const int gqa_ratio = Q->ne[2] / K->ne[2];
    if (gqa_ratio % 8 != 0 && gqa_ratio % 2 == 0 && Q->ne[1] >= 32) {
        ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 32, 2, type_KV, type_KV>(ctx, dst);
        return;
    }

    // A single token also takes the 2 token tile: K*Q with 16 Q columns per warp runs on int8 tensor cores
    // (flash_attn_ext_raw_KQ), which beat the 1 token tile at every context length tried (4096 to 262144).
    if (Q->ne[1] <= 16/ncols2) {
        ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 16/ncols2, ncols2, type_KV, type_KV>(ctx, dst);
        return;
    }
    if (Q->ne[1] <= 32/ncols2) {
        ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 32/ncols2, ncols2, type_KV, type_KV>(ctx, dst);
        return;
    }
    ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 64/ncols2, ncols2, type_KV, type_KV>(ctx, dst);
}

static void ggml_cuda_flash_attn_ext_mma_f16(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const ggml_tensor * KQV  = dst;
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    switch (Q->ne[0]) {
        case 64:
            GGML_ASSERT(V->ne[0] == 64);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2< 64,  64>(ctx, dst);
            break;
        case 80:
            GGML_ASSERT(V->ne[0] == 80);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2< 80,  80>(ctx, dst);
            break;
        case 96:
            GGML_ASSERT(V->ne[0] == 96);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2< 96,  96>(ctx, dst);
            break;
        case 112:
            GGML_ASSERT(V->ne[0] == 112);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<112, 112>(ctx, dst);
            break;
        case 128:
            GGML_ASSERT(V->ne[0] == 128);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<128, 128>(ctx, dst);
            break;
        case 192: {
            // MiMo-V2.5 / V2.5-Pro / V2-Flash: gqa_ratio is 8 (SWA) or 16 (full attn)
            GGML_ASSERT(V->ne[0] == 128);
            float max_bias = 0.0f;
            memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));
            const bool use_gqa_opt = mask && max_bias == 0.0f;
            GGML_ASSERT(use_gqa_opt);
            GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
            const int gqa_ratio = Q->ne[2] / K->ne[2];
            if (gqa_ratio % 16 == 0) {
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<192, 128, 16>(ctx, dst);
            } else {
                GGML_ASSERT(gqa_ratio % 8 == 0);
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<192, 128,  8>(ctx, dst);
            }
        } break;
        case 256:
            GGML_ASSERT(V->ne[0] == 256);
            if (ggml_cuda_fattn_mma_raw(cc, dst)) {
                if (K->type == GGML_TYPE_Q4_0) {
                    ggml_cuda_flash_attn_ext_mma_raw_switch_ncols1<256, 256, GGML_TYPE_Q4_0>(ctx, dst);
                } else {
                    ggml_cuda_flash_attn_ext_mma_raw_switch_ncols1<256, 256, GGML_TYPE_Q8_0>(ctx, dst);
                }
                break;
            }
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<256, 256>(ctx, dst);
            break;
        case 320:
            // For Mistral Small 4, go straight to the ncols1 switch (ncols2=32-only build).
            GGML_ASSERT(V->ne[0] == 256);
            {
                float max_bias = 0.0f;
                memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

                const bool use_gqa_opt = mask && max_bias == 0.0f;
                GGML_ASSERT(use_gqa_opt);
                GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
                const int gqa_ratio = Q->ne[2] / K->ne[2];
                GGML_ASSERT(gqa_ratio % 32 == 0);

                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<320, 256, 32>(ctx, dst);
            }
            break;
        case 512:
            GGML_ASSERT(V->ne[0] == 512);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<512, 512>(ctx, dst);
            break;
        case 576: {
            // For Deepseek, go straight to the ncols1 switch to avoid compiling unnecessary kernels.
            GGML_ASSERT(V->ne[0] == 512);
            float max_bias = 0.0f;
            memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

            const bool use_gqa_opt = mask && max_bias == 0.0f;
            GGML_ASSERT(use_gqa_opt);

            GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
            const int gqa_ratio = Q->ne[2] / K->ne[2];
            if (gqa_ratio == 20) { // GLM 4.7 Flash
                if (cc >= GGML_CUDA_CC_DGX_SPARK) {
                    if (Q->ne[1] <= 8) {
                        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
                        break;
                    }
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
                    break;
                }
                if (cc >= GGML_CUDA_CC_BLACKWELL) {
                    if (Q->ne[1] <= 4 && K->ne[1] >= 65536) {
                        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
                        break;
                    }
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
                    break;
                }
                if (cc >= GGML_CUDA_CC_ADA_LOVELACE) {
                    if (Q->ne[1] <= 4) {
                        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
                        break;
                    }
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
                    break;
                }
                if (cc >= GGML_CUDA_CC_TURING) {
                    if (Q->ne[1] <= 4) {
                        if (K->ne[1] <= 16384) {
                            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
                            break;
                        }
                        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 32>(ctx, dst);
                        break;
                    }
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
                    break;
                }
                // Volta:
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
            } else if (gqa_ratio % 16 == 0) {
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
            } else {
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512,  4>(ctx, dst);
            }
        } break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

#define FATTN_VEC_CASE(D, type_K, type_V)                                                                        \
    {                                                                                                            \
        const bool type_K_okay = K->type == (type_K) || (K->type == GGML_TYPE_F32 && (type_K) == GGML_TYPE_F16); \
        const bool type_V_okay = V->type == (type_V) || (V->type == GGML_TYPE_F32 && (type_V) == GGML_TYPE_F16); \
        if (Q->ne[0] == (D) && type_K_okay && type_V_okay) {                                                     \
            ggml_cuda_flash_attn_ext_vec_case<D, type_K, type_V>(ctx, dst);                                      \
            return;                                                                                              \
        }                                                                                                        \
    }                                                                                                            \

#define FATTN_VEC_CASES_ALL_D(type_K, type_V) \
    FATTN_VEC_CASE( 64, type_K, type_V)       \
    FATTN_VEC_CASE(128, type_K, type_V)       \
    FATTN_VEC_CASE(256, type_K, type_V)       \

static void ggml_cuda_flash_attn_ext_vec(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_tensor * Q = dst->src[0];
    ggml_tensor * K = dst->src[1];
    ggml_tensor * V = dst->src[2];

#ifdef GGML_CUDA_FA_ALL_QUANTS
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_F16)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_Q4_0)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_Q4_1)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_Q5_0)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_Q5_1)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_Q8_0)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_BF16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_BF16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_BF16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_BF16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_BF16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_BF16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_BF16)
#else
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_BF16)
#endif // GGML_CUDA_FA_ALL_QUANTS

    GGML_ABORT("fatal error");
}

// Best FlashAttention kernel for a specific GPU:
enum best_fattn_kernel {
    BEST_FATTN_KERNEL_NONE    =   0,
    BEST_FATTN_KERNEL_TILE    = 200,
    BEST_FATTN_KERNEL_VEC     = 100,
    BEST_FATTN_KERNEL_MMA_F16 = 400,
};

static bool ggml_cuda_fattn_kv_type_supported(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
            return true;
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
#ifndef GGML_CUDA_FA_ALL_QUANTS
            return false;
#endif // GGML_CUDA_FA_ALL_QUANTS
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_BF16:
            return true;
        default:
            return false;
    }
}

static best_fattn_kernel ggml_cuda_get_best_fattn_kernel(const int device, const ggml_tensor * dst) {
#ifndef FLASH_ATTN_AVAILABLE
    GGML_UNUSED(device); GGML_UNUSED(dst);
    return BEST_FATTN_KERNEL_NONE;
#endif// FLASH_ATTN_AVAILABLE

    const ggml_tensor * KQV   = dst;
    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];

    const int gqa_ratio = Q->ne[2] / K->ne[2];
    GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);

    // The kernels read Q, float K/V and the mask in chunks of up to 16 bytes (float4/int4 loads, cp.async), so one
    // whose data does not start at a multiple of 16 bytes (a view offset by an odd number of elements, say) would fault
    // with a misaligned address. There is no kernel for it here; this also runs in supports_op, before allocation, so
    // such an operand goes to another backend.
    for (const ggml_tensor * t : {Q, K, V, mask}) {
        if (t != nullptr && !ggml_is_quantized(t->type) && !ggml_cuda_data_is_aligned(t, 16)) {
            return BEST_FATTN_KERNEL_NONE;
        }
    }

    float max_bias = 0.0f;
    memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

    // The effective batch size for the kernel can be increased by gqa_ratio.
    // The kernel versions without this optimization are also used for ALiBi, if there is no mask, or if the KV cache is not padded,
    bool gqa_opt_applies = gqa_ratio >= 2 && mask && max_bias == 0.0f && K->ne[1] % FATTN_KQ_STRIDE == 0;
    for (const ggml_tensor * t : {Q, K, V, mask}) {
        if (t == nullptr || ggml_is_quantized(t->type)) {
            continue;
        }
        for (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
            if (t->nb[i] % 16 != 0) {
                gqa_opt_applies = false;
                break;
            }
        }
    }

    const int cc = ggml_cuda_info().devices[device].cc;

    switch (K->ne[0]) {
        case  40:
        case  64:
        case  72:
        case  80:
        case  96:
        case 128:
        case 112:
        case 256:
            if (V->ne[0] != K->ne[0]) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 192:
            if (V->ne[0] != 128 || !gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (gqa_ratio % 8 != 0) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 320:
            if (V->ne[0] != 256 || !gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (gqa_ratio % 32 != 0) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 512:
            if (V->ne[0] != K->ne[0]) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (!gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 576:
            if (V->ne[0] != 512) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (!gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        default:
            return BEST_FATTN_KERNEL_NONE;
    }

#ifndef GGML_CUDA_FA_ALL_QUANTS
    if (K->type != V->type) {
        return BEST_FATTN_KERNEL_NONE;
    }
#endif // GGML_CUDA_FA_ALL_QUANTS

    if (!ggml_cuda_fattn_kv_type_supported(K->type) || !ggml_cuda_fattn_kv_type_supported(V->type)) {
        return BEST_FATTN_KERNEL_NONE;
    }

    if (mask && mask->ne[2] != 1) {
        return BEST_FATTN_KERNEL_NONE;
    }

    // For small batch sizes the vector kernel may be preferable over the kernels optimized for large batch sizes:
    // 192 satisfies % 64 == 0 but has no vec instance (DKQ != DV); force it onto the MMA path.
    const bool can_use_vector_kernel = Q->ne[0] <= 256 && Q->ne[0] % 64 == 0 && Q->ne[0] != 192 && K->ne[1] % FATTN_KQ_STRIDE == 0;

    // If Turing tensor cores are available, use them:
    if (turing_mma_available(cc) && Q->ne[0] != 40 && Q->ne[0] != 72) {
        if (can_use_vector_kernel) {
            if (!ggml_is_quantized(K->type) && !ggml_is_quantized(V->type)) {
                if (cc >= GGML_CUDA_CC_ADA_LOVELACE && Q->ne[1] == 1 && Q->ne[3] == 1 && !(gqa_ratio > 4 && K->ne[1] >= 8192)) {
                    return BEST_FATTN_KERNEL_VEC;
                }
            } else {
                if (cc >= GGML_CUDA_CC_ADA_LOVELACE) {
                    // The vector kernel reads K and V once per Q head, the raw q4_0 / q8_0 MMA kernel once per GQA group.
                    // Decoding on an RTX 5080 the MMA kernel is ahead from 4096 tokens of context on (llama-bench tg64);
                    // q8_0 on an RTX 5090 too: level at 4096, +4 % at 8192, +10 % at 16384, 1.86x at 245760.
                    if (Q->ne[1] <= 2 && !(K->ne[1] >= 4096 && ggml_cuda_fattn_mma_raw(cc, dst))) {
                        return BEST_FATTN_KERNEL_VEC;
                    }
                } else {
                    if (Q->ne[1] == 1) {
                        return BEST_FATTN_KERNEL_VEC;
                    }
                }
            }
            if (!gqa_opt_applies && Q->ne[1] == 1) {
                return BEST_FATTN_KERNEL_VEC;
            }
        }
        return BEST_FATTN_KERNEL_MMA_F16;
    }

    const int ncols2_max = Q->ne[0] == 320 ? 32 : ((Q->ne[0] == 576 || Q->ne[0] == 192) ? 16 : 8);
    int gqa_ratio_eff = 1;
    while (gqa_ratio % (2*gqa_ratio_eff) == 0 && gqa_ratio_eff < ncols2_max) {
        gqa_ratio_eff *= 2;
    }

    if (volta_mma_available(cc) && Q->ne[0] != 40 && Q->ne[0] != 72) {
        if (can_use_vector_kernel && Q->ne[1] * gqa_ratio_eff <= 2) {
            return BEST_FATTN_KERNEL_VEC;
        }
        if (Q->ne[1] * gqa_ratio_eff <= 16) {
            return BEST_FATTN_KERNEL_TILE; // On Volta tensor cores are only faster for sufficiently large matrices.
        }
        return BEST_FATTN_KERNEL_MMA_F16;
    }

    // AMD MFMA needs a certain minimum batch size to outscale the tile kernel for large head sizes.
    if ((amd_mfma_available(cc) && Q->ne[0] <= 256) && Q->ne[0] != 40 && Q->ne[0] != 72) {
        if ((Q->ne[0] <= 64 && Q->ne[1] * gqa_ratio_eff > 8)) {
            return BEST_FATTN_KERNEL_MMA_F16;
        }
        if ((Q->ne[0] <= 128 && Q->ne[1] * gqa_ratio_eff > 16)) {
            return BEST_FATTN_KERNEL_MMA_F16;
        }
        if ((Q->ne[0] <= 256 && Q->ne[1] * gqa_ratio_eff > 64)) {
            return BEST_FATTN_KERNEL_MMA_F16;
        }
    }

    // AMD WMMA is always faster than the tile kernel if the full tile width of 16 can be utilized.
    if ((amd_wmma_available(cc) && gqa_opt_applies && Q->ne[0] <= 128) && Q->ne[0] != 40 && Q->ne[0] != 72 && Q->ne[1] * gqa_ratio_eff > 8) {
        return BEST_FATTN_KERNEL_MMA_F16;
    }

    // If there are no tensor cores available, use the generic tile kernel:
    if (can_use_vector_kernel) {
        if (!ggml_is_quantized(K->type) && !ggml_is_quantized(V->type)) {
            if (Q->ne[1] == 1) {
                if (!gqa_opt_applies) {
                    return BEST_FATTN_KERNEL_VEC;
                }
            }
        } else {
            if (Q->ne[1] <= 2) {
                return BEST_FATTN_KERNEL_VEC;
            }
        }
    }
    return BEST_FATTN_KERNEL_TILE;
}

size_t ggml_cuda_flash_attn_ext_get_alloc_size(int device, const ggml_tensor * dst) {
    GGML_ASSERT(dst->op == GGML_OP_FLASH_ATTN_EXT);

    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    GGML_ASSERT(K != nullptr);
    GGML_ASSERT(V != nullptr);

    const best_fattn_kernel kernel = ggml_cuda_get_best_fattn_kernel(device, dst);

    bool need_f16_K = false;
    bool need_f16_V = false;

    switch (kernel) {
        case BEST_FATTN_KERNEL_TILE:
            need_f16_K = true;
            need_f16_V = true;
            break;
        case BEST_FATTN_KERNEL_MMA_F16:
            need_f16_K = !ggml_cuda_fattn_mma_raw(ggml_cuda_info().devices[device].cc, dst);
            need_f16_V = need_f16_K;
            break;
        case BEST_FATTN_KERNEL_VEC:
            need_f16_K = K->type == GGML_TYPE_F32;
            need_f16_V = V->type == GGML_TYPE_F32;
            break;
        case BEST_FATTN_KERNEL_NONE:
            break;
    }

    const ggml_cuda_flash_attn_ext_f16_extra_data f16_extra =
        ggml_cuda_flash_attn_ext_get_f16_extra_data(dst, need_f16_K, need_f16_V);

    return f16_extra.end - (uintptr_t) dst->data;
}

void ggml_cuda_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_set_device(ctx.device);
    switch (ggml_cuda_get_best_fattn_kernel(ggml_cuda_get_device(), dst)) {
        case BEST_FATTN_KERNEL_NONE:
            GGML_ABORT("fatal error");
        case BEST_FATTN_KERNEL_TILE:
            ggml_cuda_flash_attn_ext_tile(ctx, dst);
            break;
        case BEST_FATTN_KERNEL_VEC:
            ggml_cuda_flash_attn_ext_vec(ctx, dst);
            break;
        case BEST_FATTN_KERNEL_MMA_F16:
            ggml_cuda_flash_attn_ext_mma_f16(ctx, dst);
            break;
    }
}

bool ggml_cuda_flash_attn_ext_supported(int device, const ggml_tensor * dst) {
    return ggml_cuda_get_best_fattn_kernel(device, dst) != BEST_FATTN_KERNEL_NONE;
}
