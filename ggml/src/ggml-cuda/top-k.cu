#include "argsort.cuh"
#include "top-k.cuh"

#ifdef GGML_CUDA_USE_CUB
#    include <cub/cub.cuh>
#    if (CCCL_MAJOR_VERSION >= 3 && CCCL_MINOR_VERSION >= 2)
#        define CUB_TOP_K_AVAILABLE
#        include <cuda/iterator>
using namespace cub;
#    endif  // CCCL_MAJOR_VERSION >= 3 && CCCL_MINOR_VERSION >= 2
#endif      // GGML_CUDA_USE_CUB

#ifdef CUB_TOP_K_AVAILABLE

static void top_k_cub(ggml_cuda_pool & pool,
                      const float *    src,
                      int *            dst,
                      const int        ncols,
                      const int        k,
                      cudaStream_t     stream) {
    auto requirements = cuda::execution::require(cuda::execution::determinism::not_guaranteed,
                                                 cuda::execution::output_ordering::unsorted);
    auto stream_env   = cuda::stream_ref{ stream };
    auto env          = cuda::std::execution::env{ stream_env, requirements };

    auto indexes_in = cuda::make_counting_iterator(0);

    size_t temp_storage_bytes = 0;
    CUDA_CHECK(DeviceTopK::MaxPairs(nullptr, temp_storage_bytes, src, cuda::discard_iterator(), indexes_in, dst, ncols, k,
                         env));

    ggml_cuda_pool_alloc<uint8_t> temp_storage_alloc(pool, temp_storage_bytes);
    void *                        d_temp_storage = temp_storage_alloc.get();

    CUDA_CHECK(DeviceTopK::MaxPairs(d_temp_storage, temp_storage_bytes, src, cuda::discard_iterator(), indexes_in, dst,
                         ncols, k, env));
}

#elif defined(GGML_CUDA_USE_CUB)  // CUB_TOP_K_AVAILABLE

static int next_power_of_2(int x) {
    int n = 1;
    while (n < x) {
        n *= 2;
    }
    return n;
}

#endif                            // CUB_TOP_K_AVAILABLE


// Two-stage top-k for wide rows: a global top-k element has at most k-1 larger
// elements, so at most k-1 inside its own tile and tiling cannot drop a winner.
#define TOPK_CAND  1024   // argsort_f32_i32_cuda_bitonic's row limit

// measured on H200 and A10G
#define TOPK_BLOCK     256
#define TOPK_TILE_WIDE 8192
#define TOPK_TILE      4096

template <int TILE, int BLOCK>
static __global__ void topk_tile(const float * src, float * cand_val, int * cand_idx,
                                 const int ncols, const int ntiles, const int k) {
    __shared__ uint64_t smem[BLOCK];

    const int     row     = blockIdx.x / ntiles;
    const int     tile    = blockIdx.x % ntiles;
    const float * row_ptr = src + (size_t) row * ncols;

    uint64_t keys[TILE / BLOCK];
#pragma unroll
    for (int i = 0; i < TILE / BLOCK; ++i) {
        const int col = tile * TILE + threadIdx.x + i * BLOCK;
        uint32_t  b   = col < ncols ? __float_as_uint(row_ptr[col]) : 0;
        b = (b & 0x80000000u) ? ~b : (b | 0x80000000u);
        keys[i] = col < ncols ? (((uint64_t) b << 32) | (uint32_t) (ncols - 1 - col)) : 0;
    }

    const size_t out = ((size_t) row * ntiles + tile) * k;
    for (int j = 0; j < k; ++j) {
        uint64_t local = 0;
#pragma unroll
        for (int i = 0; i < TILE / BLOCK; ++i) {
            local = max(local, keys[i]);
        }
        smem[threadIdx.x] = local;
        __syncthreads();
        for (int s = BLOCK / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) {
                smem[threadIdx.x] = max(smem[threadIdx.x], smem[threadIdx.x + s]);
            }
            __syncthreads();
        }
        const uint64_t best = smem[0];
        if (threadIdx.x == 0) {
            const int col = ncols - 1 - (int) (best & 0xFFFFFFFFu);
            cand_val[out + j] = best ? row_ptr[col] : -INFINITY;
            cand_idx[out + j] = best ? col : 0;
        }
#pragma unroll
        for (int i = 0; i < TILE / BLOCK; ++i) {
            if (keys[i] == best) {
                keys[i] = 0;
            }
        }
        __syncthreads();
    }
}

// The argsort ranks candidates; turn its positions back into columns.
static __global__ void topk_unmap(const int * cand_idx, const int * order, int * dst,
                                  const int ncand, const int k) {
    for (int i = threadIdx.x; i < k; i += blockDim.x) {
        dst[(size_t) blockIdx.x * k + i] = cand_idx[(size_t) blockIdx.x * ncand + order[(size_t) blockIdx.x * ncand + i]];
    }
}

static bool ggml_cuda_top_k_tiled(ggml_cuda_pool & pool, const float * src, int * dst,
                                  const int ncols, const int nrows, const int k,
                                  cudaStream_t stream) {
    // Narrow rows are already handled whole by the bitonic sort below.
    const int tile   = ncols >= 65536 ? TOPK_TILE_WIDE : TOPK_TILE;
    const int ntiles = (ncols + tile - 1) / tile;
    const int ncand  = ntiles * k;
    if (ncols <= TOPK_CAND || ncand > TOPK_CAND) {
        return false;
    }

    ggml_cuda_pool_alloc<float> cand_val(pool, (size_t) nrows * ncand);
    ggml_cuda_pool_alloc<int>   cand_idx(pool, (size_t) nrows * ncand);
    ggml_cuda_pool_alloc<int>   order   (pool, (size_t) nrows * ncand);

    if (tile == TOPK_TILE_WIDE) {
        topk_tile<TOPK_TILE_WIDE, TOPK_BLOCK><<<nrows * ntiles, TOPK_BLOCK, 0, stream>>>(
                src, cand_val.get(), cand_idx.get(), ncols, ntiles, k);
    } else {
        topk_tile<TOPK_TILE, TOPK_BLOCK><<<nrows * ntiles, TOPK_BLOCK, 0, stream>>>(
                src, cand_val.get(), cand_idx.get(), ncols, ntiles, k);
    }
    argsort_f32_i32_cuda_bitonic(cand_val.get(), order.get(), ncand, nrows,
            GGML_SORT_ORDER_DESC, stream);
    topk_unmap<<<nrows, TOPK_BLOCK, 0, stream>>>(cand_idx.get(), order.get(), dst, ncand, k);
    return true;
}

// Radix select for a large k: one block a row finds the k-th largest key in four passes of 8 bits (a histogram of
// the next digit among the keys that share the digits found so far), then one pass writes, in column order, every
// column above that key and the lowest columns equal to it. Its time does not grow with k, where the tiled path runs
// k block reductions one after another (a DSA indexer's 512 pools of thousands, one launch a layer).
#define TOPK_RADIX_BLOCK 1024
#define TOPK_RADIX_MIN_K 64
#define TOPK_RADIX_TILE  8192 // RTX 5080, k 512: 3 rows of 109,020 in 21.7 us (4,096: 26.3; 16,384: 27.3)

static __device__ __forceinline__ uint32_t topk_radix_key(const float x) {
    const uint32_t b = __float_as_uint(x);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

static __device__ __forceinline__ uint32_t topk_radix_warp_incl_scan(uint32_t v, const int lane) {
#pragma unroll
    for (int o = 1; o < WARP_SIZE; o <<= 1) {
        const uint32_t t = __shfl_up_sync(0xFFFFFFFFu, v, o);
        if (lane >= o) {
            v += t;
        }
    }
    return v;
}

// One block a segment of seg_len columns, nseg segments a row: a whole row (nseg 1), or a tile of a wide row whose k
// best, keys and columns, a second launch selects among (CAND: src_key/src_col are that first launch's out_key/out_col).
// A tile narrower than k pads its candidates with key 0, below every float's key, so they are never taken.
// ITEMS > 0: the segment's keys stay in registers (seg_len <= ITEMS*BLOCK); 0: every pass reads the segment again.
template <int ITEMS, bool CAND>
static __global__ void __launch_bounds__(TOPK_RADIX_BLOCK) topk_radix(
        const float * src, const uint32_t * src_key, const int * src_col, int * out_col, uint32_t * out_key,
        const int64_t row_stride, const int seg_len, const int nseg, const int ncols, const int k) {
    constexpr int BLOCK  = TOPK_RADIX_BLOCK;
    constexpr int NWARPS = BLOCK / WARP_SIZE;
    static_assert(NWARPS <= WARP_SIZE, "one warp scans the warps' sums");

    __shared__ uint32_t hist[256];
    __shared__ uint32_t warp_sums[NWARPS];
    __shared__ uint32_t s_digit;
    __shared__ uint32_t s_krem;

    const int tid  = threadIdx.x;
    const int lane = tid % WARP_SIZE;
    const int warp = tid / WARP_SIZE;

    const int     row  = blockIdx.x / nseg;
    const int     col0 = (blockIdx.x % nseg) * seg_len;
    const int     n    = min(seg_len, ncols - col0);
    const int     kn   = min(k, n);
    const int64_t base = row*row_stride + col0;
    int      * out   = out_col + (size_t) blockIdx.x * k;
    uint32_t * out_k = out_key ? out_key + (size_t) blockIdx.x * k : nullptr;

    auto load_key = [&](const int col) -> uint32_t {
        if (col >= n) {
            return 0;
        }
        if constexpr (CAND) {
            return src_key[base + col];
        } else {
            return topk_radix_key(src[base + col]);
        }
    };

    const int n_iter = ITEMS > 0 ? ITEMS : (n + BLOCK - 1) / BLOCK;

    uint32_t keys[ITEMS > 0 ? ITEMS : 1];
    if constexpr (ITEMS > 0) {
#pragma unroll
        for (int i = 0; i < ITEMS; ++i) {
            keys[i] = load_key(i*BLOCK + tid);
        }
    }

    uint32_t prefix = 0;
    uint32_t mask   = 0;
    uint32_t krem   = kn; // how many of the keys matching prefix/mask are still to be taken

#pragma unroll 1
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int b = tid; b < 256; b += BLOCK) {
            hist[b] = 0;
        }
        __syncthreads();

#pragma unroll
        for (int i = 0; i < n_iter; ++i) {
            const int col = i*BLOCK + tid;
            uint32_t key;
            if constexpr (ITEMS > 0) {
                key = keys[i];
            } else {
                key = load_key(col);
            }
            const bool     in    = col < n && (key & mask) == prefix;
            const uint32_t digit = (key >> shift) & 0xFFu;
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
            // keys close in value share their high digits: one atomic a digit a warp, not one a key
            const uint32_t active = __ballot_sync(0xFFFFFFFFu, in);
            if (in) {
                const uint32_t peers = __match_any_sync(active, digit);
                if (lane == __ffs(peers) - 1) {
                    atomicAdd(&hist[digit], (uint32_t) __popc(peers));
                }
            }
#else
            if (in) {
                atomicAdd(&hist[digit], 1u);
            }
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
        }
        __syncthreads();

        // the digit d with (keys above d) < krem <= (keys at or above d), scanning the bins from the top
        const int      b = 255 - tid;
        const uint32_t h = tid < 256 ? hist[b] : 0;
        const uint32_t incl = topk_radix_warp_incl_scan(h, lane);
        if (tid < 256 && lane == WARP_SIZE - 1) {
            warp_sums[warp] = incl;
        }
        __syncthreads();
        if (tid < 256) {
            uint32_t ge = incl;
            for (int w = 0; w < warp; ++w) {
                ge += warp_sums[w];
            }
            const uint32_t gt = ge - h;
            if (gt < krem && ge >= krem) {
                s_digit = b;
                s_krem  = krem - gt;
            }
        }
        __syncthreads();

        prefix |= s_digit << shift;
        mask   |= 0xFFu   << shift;
        krem    = s_krem;
    }

    // prefix is now the kn-th largest key; krem of the keys equal to it are taken, the lowest columns first
    const uint32_t n_gt = kn - krem;
    uint32_t base_gt = 0;
    uint32_t base_eq = 0;

    auto emit = [&](const uint32_t pos, const int col, const uint32_t key) {
        if constexpr (CAND) {
            out[pos] = src_col[base + col];
        } else {
            out[pos] = col0 + col;
        }
        if (out_k) {
            out_k[pos] = key;
        }
    };
    if (out_k) {
        for (int pos = kn + tid; pos < k; pos += BLOCK) {
            out[pos]   = -1;
            out_k[pos] = 0;
        }
    }

#pragma unroll
    for (int i = 0; i < n_iter; ++i) {
        const int col = i*BLOCK + tid;
        uint32_t key;
        if constexpr (ITEMS > 0) {
            key = keys[i];
        } else {
            key = load_key(col);
        }
        const bool gt = col < n && key >  prefix;
        const bool eq = col < n && key == prefix;

        // a block's counts fit in 16 bits: the column's rank among the chunk's gt keys low, among its eq keys high
        const uint32_t v    = (uint32_t) gt | ((uint32_t) eq << 16);
        const uint32_t incl = topk_radix_warp_incl_scan(v, lane);
        if (lane == WARP_SIZE - 1) {
            warp_sums[warp] = incl;
        }
        __syncthreads();
        if (warp == 0) {
            const uint32_t w = topk_radix_warp_incl_scan(lane < NWARPS ? warp_sums[lane] : 0, lane);
            if (lane < NWARPS) {
                warp_sums[lane] = w;
            }
        }
        __syncthreads();

        const uint32_t excl  = (warp > 0 ? warp_sums[warp - 1] : 0) + incl - v;
        const uint32_t total = warp_sums[NWARPS - 1];
        if (gt) {
            emit(base_gt + (excl & 0xFFFFu), col, key);
        }
        if (eq) {
            const uint32_t r = base_eq + (excl >> 16);
            if (r < krem) {
                emit(n_gt + r, col, key);
            }
        }
        base_gt += total & 0xFFFFu;
        base_eq += total >> 16;
        if (base_gt == n_gt && base_eq >= krem) {
            break; // uniform: every thread read the same total
        }
        __syncthreads(); // warp_sums is written again by the next chunk
    }
}

template <bool CAND>
static void topk_radix_launch(const int nblocks, const float * src, const uint32_t * src_key, const int * src_col,
                              int * out_col, uint32_t * out_key, const int64_t row_stride, const int seg_len,
                              const int nseg, const int ncols, const int k, cudaStream_t stream) {
    if (seg_len <= 4*TOPK_RADIX_BLOCK) {
        topk_radix<4, CAND><<<nblocks, TOPK_RADIX_BLOCK, 0, stream>>>(src, src_key, src_col, out_col, out_key, row_stride, seg_len, nseg, ncols, k);
    } else if (seg_len <= 8*TOPK_RADIX_BLOCK) {
        topk_radix<8, CAND><<<nblocks, TOPK_RADIX_BLOCK, 0, stream>>>(src, src_key, src_col, out_col, out_key, row_stride, seg_len, nseg, ncols, k);
    } else if (seg_len <= 16*TOPK_RADIX_BLOCK) {
        topk_radix<16, CAND><<<nblocks, TOPK_RADIX_BLOCK, 0, stream>>>(src, src_key, src_col, out_col, out_key, row_stride, seg_len, nseg, ncols, k);
    } else {
        topk_radix<0, CAND><<<nblocks, TOPK_RADIX_BLOCK, 0, stream>>>(src, src_key, src_col, out_col, out_key, row_stride, seg_len, nseg, ncols, k);
    }
}

static bool ggml_cuda_top_k_radix(ggml_cuda_pool & pool, const float * src, int * dst, const int ncols, const int nrows,
                                  const int k, const int cc, cudaStream_t stream) {
    // A wide row with a small k stays with the tiled path, whose blocks split the row. A narrow row (no tiling) at any k
    // is taken here: the other route is CUB's top-k one row after another, four launches a row (a 512-token ubatch of a
    // DSA indexer under 4,096 cells: 2,048 launches a layer). GGML_CUDA_TOPK_RADIX_LEGACY=1: never taken.
    static const bool legacy = ggml_env_switch("GGML_CUDA_TOPK_RADIX_LEGACY");
    if (legacy || (ncols > TOPK_CAND && k < TOPK_RADIX_MIN_K) || !GGML_CUDA_CC_IS_NVIDIA(cc) || cc < GGML_CUDA_CC_VOLTA) {
        return false;
    }
    if (ncols <= 16*TOPK_RADIX_BLOCK) {
        topk_radix_launch<false>(nrows, src, nullptr, nullptr, dst, nullptr, ncols, ncols, 1, ncols, k, stream);
        return true;
    }
#ifdef CUB_TOP_K_AVAILABLE
    // one wide row: CUB's device-wide top-k spreads it over the card (RTX 5080, k 512: 12.7 us at 109,020 columns
    // against the two stages' 22.4; at 3 rows its launch a row costs 68.4 against 21.7)
    if (nrows == 1) {
        return false;
    }
#endif // CUB_TOP_K_AVAILABLE
    // A row too wide for one block's registers is cut into tiles whose blocks select their k in parallel: a key among
    // the row's k best has fewer than k better keys (by value, then lower column), so its tile keeps it too.
    const int tile = TOPK_RADIX_TILE;
    if (2*k > tile) {
        topk_radix_launch<false>(nrows, src, nullptr, nullptr, dst, nullptr, ncols, ncols, 1, ncols, k, stream);
        return true;
    }
    const int ntiles = (ncols + tile - 1) / tile;
    const int ncand  = ntiles * k;
    ggml_cuda_pool_alloc<uint32_t> cand_key(pool, (size_t) nrows * ncand);
    ggml_cuda_pool_alloc<int>      cand_col(pool, (size_t) nrows * ncand);
    topk_radix_launch<false>(nrows * ntiles, src, nullptr, nullptr, cand_col.get(), cand_key.get(), ncols, tile, ntiles, ncols, k, stream);
    topk_radix_launch<true>(nrows, nullptr, cand_key.get(), cand_col.get(), dst, nullptr, ncand, ncand, 1, ncand, k, stream);
    return true;
}

void ggml_cuda_op_top_k(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0   = dst->src[0];
    const float *       src0_d = (const float *) src0->data;
    int *               dst_d  = (int *) dst->data;
    cudaStream_t        stream = ctx.stream();

    // are these asserts truly necessary?
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    const int64_t    ncols = src0->ne[0];
    const int64_t    nrows = ggml_nrows(src0);
    const int64_t    k     = dst->ne[0];
    ggml_cuda_pool & pool  = ctx.pool();

    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    if (ggml_cuda_top_k_radix(pool, src0_d, dst_d, ncols, nrows, k, cc, stream)) {
        return;
    }

    if (ggml_cuda_top_k_tiled(pool, src0_d, dst_d, ncols, nrows, k, stream)) {
        return;
    }

#ifdef CUB_TOP_K_AVAILABLE
    // TODO: Switch to `DeviceSegmentedTopK` for multi-row TopK once implemented
    // https://github.com/NVIDIA/cccl/issues/6391
    // TODO: investigate if there exists a point where parallelized argsort is faster than sequential top-k
    for (int i = 0; i < nrows; i++) {
        top_k_cub(pool, src0_d + i * ncols, dst_d + i * k, ncols, k, stream);
    }
#elif defined(GGML_CUDA_USE_CUB)  // CUB_TOP_K_AVAILABLE
    // Fall back to argsort + copy
    const int    ncols_pad      = next_power_of_2(ncols);
    const size_t shared_mem     = ncols_pad * sizeof(int);
    const size_t max_shared_mem = ggml_cuda_info().devices[ggml_cuda_get_device()].smpb;
    const bool   use_bitonic    = shared_mem <= max_shared_mem && ncols <= 1024;
    const int    chunk_nrows    = argsort_f32_i32_cuda_cub_chunk_nrows(src0->nb[1], nrows);

    ggml_cuda_pool_alloc<int> temp_dst_alloc(pool, ncols * chunk_nrows);
    int *                     tmp_dst = temp_dst_alloc.get();

    for (int64_t i = 0; i < nrows; i += chunk_nrows) {
        int iter_nrows = std::min((int64_t) chunk_nrows, nrows - i);

        if (use_bitonic) {
            argsort_f32_i32_cuda_bitonic(src0_d, tmp_dst, ncols, iter_nrows, GGML_SORT_ORDER_DESC, stream);
        } else {
            argsort_f32_i32_cuda_cub(pool, src0_d, tmp_dst, ncols, iter_nrows, GGML_SORT_ORDER_DESC, stream);
        }
        CUDA_CHECK(cudaMemcpy2DAsync(dst_d, k * sizeof(int), tmp_dst, ncols * sizeof(int), k * sizeof(int), iter_nrows,
                                     cudaMemcpyDeviceToDevice, stream));

        src0_d += ncols * iter_nrows;
        dst_d  += k     * iter_nrows;
    }
#else                             // GGML_CUDA_USE_CUB
    ggml_cuda_pool_alloc<int> temp_dst_alloc(pool, ncols * nrows);
    int *                     tmp_dst = temp_dst_alloc.get();
    argsort_f32_i32_cuda_bitonic(src0_d, tmp_dst, ncols, nrows, GGML_SORT_ORDER_DESC, stream);
    CUDA_CHECK(cudaMemcpy2DAsync(dst_d, k * sizeof(int), tmp_dst, ncols * sizeof(int), k * sizeof(int), nrows,
                                 cudaMemcpyDeviceToDevice, stream));
#endif
}
