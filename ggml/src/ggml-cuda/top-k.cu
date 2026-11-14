#include "argsort.cuh"
#include "top-k.cuh"

// Adjusted implementation thresholds from #28547, can be overridden at build time
#ifndef GGML_CUDA_TOP_K_NCOLS_THRESHOLD_BITONIC
#    if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
// not measured on HIP/MUSA, keep the old split
#        define GGML_CUDA_TOP_K_NCOLS_THRESHOLD_BITONIC 1024
#    else
#        define GGML_CUDA_TOP_K_NCOLS_THRESHOLD_BITONIC 512
#    endif
#endif // GGML_CUDA_TOP_K_NCOLS_THRESHOLD_BITONIC

#ifndef GGML_CUDA_TOP_K_NCOLS_THRESHOLD_ARGSORT
#    define GGML_CUDA_TOP_K_NCOLS_THRESHOLD_ARGSORT 4096
#endif // GGML_CUDA_TOP_K_NCOLS_THRESHOLD_ARGSORT

// bitonic up to this width while nrows fits in one wave of SMs, 0 disables
#ifndef GGML_CUDA_TOP_K_NCOLS_THRESHOLD_BITONIC_FEW_ROWS
#    if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
#        define GGML_CUDA_TOP_K_NCOLS_THRESHOLD_BITONIC_FEW_ROWS 0
#    else
#        define GGML_CUDA_TOP_K_NCOLS_THRESHOLD_BITONIC_FEW_ROWS 1024
#    endif
#endif // GGML_CUDA_TOP_K_NCOLS_THRESHOLD_BITONIC_FEW_ROWS

#ifdef GGML_CUDA_USE_CUB
#    include <cub/cub.cuh>
// DeviceTopK has a race condition before CCCL 3.4.3.
// https://github.com/NVIDIA/cccl/pull/10627
#    if (CCCL_MAJOR_VERSION > 3 || \
         (CCCL_MAJOR_VERSION == 3 && CCCL_MINOR_VERSION > 4) || \
         (CCCL_MAJOR_VERSION == 3 && CCCL_MINOR_VERSION == 4 && CCCL_PATCH_VERSION >= 3))
#        define CUB_TOP_K_AVAILABLE
#        include <cuda/iterator>
using namespace cub;
#    endif  // CCCL >= 3.4.3
#endif      // GGML_CUDA_USE_CUB

// max rows for the per-row DeviceTopK / CUB argsort path before switching to radix / bitonic
#ifndef GGML_CUDA_TOP_K_NROWS_THRESHOLD
#    ifdef CUB_TOP_K_AVAILABLE
#        define GGML_CUDA_TOP_K_NROWS_THRESHOLD 2
#    else
#        define GGML_CUDA_TOP_K_NROWS_THRESHOLD 1
#    endif
#endif // GGML_CUDA_TOP_K_NROWS_THRESHOLD

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

#endif                            // CUB_TOP_K_AVAILABLE

static int next_power_of_2(int x) {
    int n = 1;
    while (n < x) {
        n *= 2;
    }
    return n;
}

static __device__ __forceinline__ uint32_t top_k_float_to_ordered(float value) {
    const uint32_t bits = __float_as_uint(value);
    const uint32_t mask = (uint32_t) (-(int32_t) (bits >> 31)) | 0x80000000U;
    return bits ^ mask;
}

// one state per cache line, so the gather atomics of different rows do not compete
struct alignas(128) top_k_radix_state {
    uint32_t prefix;
    uint32_t prefix_mask;
    int rank;
    int greater_count;
};

static __global__ void top_k_radix_init(top_k_radix_state * states, int nrows, int k) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < nrows) {
        states[row] = {0, 0, k, 0};
    }
}

template<int BLOCK_SIZE, int RADIX_BITS>
static __global__ void top_k_radix_histogram(
        const float * __restrict__ src,
        const top_k_radix_state * __restrict__ states,
        int * __restrict__ block_histograms,
        int ncols,
        int block_cols,
        int blocks_per_row,
        int shift) {
    constexpr int NBINS = 1 << RADIX_BITS;

    const int row = blockIdx.x / blocks_per_row;
    const int row_block = blockIdx.x % blocks_per_row;
    const int tid = threadIdx.x;
    const float * row_src = src + (size_t) row * ncols;
    const int64_t col_begin = (int64_t) row_block * block_cols;
    const int64_t col_end   = min(col_begin + block_cols, (int64_t) ncols);
    __shared__ int histogram[NBINS];

    histogram[tid] = 0;
    __syncthreads();

    const top_k_radix_state state = states[row];
    for (int64_t col = col_begin + tid; col < col_end; col += BLOCK_SIZE) {
        const uint32_t key = top_k_float_to_ordered(row_src[col]);
        if ((key & state.prefix_mask) == state.prefix) {
            atomicAdd(&histogram[(key >> shift) & (NBINS - 1)], 1);
        }
    }
    __syncthreads();

    const size_t histogram_offset =
        ((size_t) row * blocks_per_row + row_block) * NBINS;
    block_histograms[histogram_offset + tid] = histogram[tid];
}

template<int BLOCK_SIZE, int RADIX_BITS>
static __global__ void top_k_radix_select(
        const int * __restrict__ block_histograms,
        top_k_radix_state * __restrict__ states,
        int blocks_per_row,
        int shift) {
    constexpr int NBINS = 1 << RADIX_BITS;

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    __shared__ int histogram[NBINS];

    int count = 0;
    for (int row_block = 0; row_block < blocks_per_row; ++row_block) {
        const size_t offset = ((size_t) row * blocks_per_row + row_block) * NBINS;
        count += block_histograms[offset + tid];
    }
    histogram[tid] = count;
    __syncthreads();

    if (tid == 0) {
        top_k_radix_state state = states[row];
        int bin = NBINS - 1;
        while (bin > 0 && histogram[bin] < state.rank) {
            state.rank -= histogram[bin--];
        }
        state.prefix |= (uint32_t) bin << shift;
        state.prefix_mask |= (uint32_t) (NBINS - 1) << shift;
        states[row] = state;
    }
}

template<int BLOCK_SIZE, int RADIX_BITS>
static __global__ void top_k_radix_gather(
        const float * __restrict__ src,
        int * __restrict__ dst,
        const int * __restrict__ block_histograms,
        top_k_radix_state * __restrict__ states,
        int ncols,
        int block_cols,
        int k,
        int blocks_per_row) {
    constexpr int NBINS     = 1 << RADIX_BITS;
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps    = BLOCK_SIZE / warp_size;
    using lane_mask_t = std::conditional_t<warp_size == 64, uint64_t, uint32_t>;
    // HIP requires a 64-bit mask also for a 32-wide warp
    constexpr uint64_t ballot_mask = ~(lane_mask_t) 0;
    const auto popc = [](const lane_mask_t mask) {
        if constexpr (warp_size == 64) {
            return __popcll(mask);
        } else {
            return __popc(mask);
        }
    };
    __shared__ int warp_ties[nwarps];

    const int row = blockIdx.x / blocks_per_row;
    const int row_block = blockIdx.x % blocks_per_row;
    const int lane = threadIdx.x % warp_size;
    const int warp = threadIdx.x / warp_size;
    const float * row_src = src + (size_t) row * ncols;
    int * row_dst = dst + (size_t) row * k;
    top_k_radix_state * state = &states[row];
    const uint32_t threshold = state->prefix;
    const int rank = state->rank;

    // each warp reads a contiguous part of the contiguous range of the block
    const int     warp_cols  = block_cols / nwarps;
    const int64_t warp_begin = (int64_t) row_block * block_cols + warp * warp_cols;
    const int64_t warp_end   = min(warp_begin + warp_cols, (int64_t) ncols);

    int nties = 0;
    for (int64_t col = warp_begin + lane; col < warp_end; col += warp_size) {
        const uint32_t key = top_k_float_to_ordered(row_src[col]);
        if (key > threshold) {
            const int pos = atomicAdd(&state->greater_count, 1);
            row_dst[pos] = col;
        } else if (key == threshold) {
            nties++;
        }
    }

    // the ties fill the last rank slots in column order, the lowest columns win.
    // the histograms of the last pass count the ties of each block in the bin of the threshold.
    const int * block_ties = block_histograms + (size_t) row * blocks_per_row * NBINS + (threshold & (NBINS - 1));
    int pos = 0;
    for (int b = lane; b < row_block; b += warp_size) {
        pos += block_ties[b * NBINS];
    }
    pos = warp_reduce_sum<warp_size>(pos);
    if (pos >= rank || block_ties[row_block * NBINS] == 0) {
        return; // the same result in all threads of the block
    }

    nties = warp_reduce_sum<warp_size>(nties);
    if (lane == 0) {
        warp_ties[warp] = nties;
    }
    __syncthreads();
    for (int w = 0; w < warp; ++w) {
        pos += warp_ties[w];
    }

    // pos is the same in all lanes of the warp, so the lanes leave the loop together
    const int pos_end = min(pos + nties, rank);
    for (int64_t tile = warp_begin; tile < warp_end && pos < pos_end; tile += warp_size) {
        const int64_t col    = tile + lane;
        const bool    is_tie = col < warp_end && top_k_float_to_ordered(row_src[col]) == threshold;

        const lane_mask_t ties    = (lane_mask_t) __ballot_sync(ballot_mask, is_tie);
        const int         tie_pos = pos + popc(ties & (((lane_mask_t) 1 << lane) - 1));
        if (is_tie && tie_pos < pos_end) {
            row_dst[k - rank + tie_pos] = col;
        }
        pos += popc(ties);
    }
}

static void top_k_radix_cuda(
        ggml_cuda_pool & pool,
        const float * src, int * dst, int ncols, int64_t nrows, int k, cudaStream_t stream) {
    constexpr int BLOCK_SIZE = 256;
    constexpr int RADIX_BITS = 8;
    constexpr int NBINS = 1 << RADIX_BITS;
    const int blocks_per_row = (int) std::min<int64_t>(((int64_t) ncols + 1023) / 1024, 64);
    // each block of a row reads a contiguous range of whole tiles
    const int block_cols = (int) (((int64_t) ncols + blocks_per_row * BLOCK_SIZE - 1) / (blocks_per_row * BLOCK_SIZE) * BLOCK_SIZE);

    // chunk the rows to bound the histogram memory to 64 MB
    const int64_t chunk_nrows = ggml_cuda_chunk_nrows((size_t) blocks_per_row * NBINS * sizeof(int), nrows);

    ggml_cuda_pool_alloc<top_k_radix_state> states_alloc(pool, chunk_nrows);
    ggml_cuda_pool_alloc<int> histograms_alloc(pool, (size_t) chunk_nrows * blocks_per_row * NBINS);
    top_k_radix_state * states = states_alloc.get();
    int * histograms = histograms_alloc.get();

    for (int64_t i = 0; i < nrows; i += chunk_nrows) {
        const int iter_nrows = std::min(chunk_nrows, nrows - i);

        top_k_radix_init<<<(iter_nrows + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(states, iter_nrows, k);

        const dim3 row_grid(blocks_per_row * iter_nrows);
        for (int shift = 32 - RADIX_BITS; shift >= 0; shift -= RADIX_BITS) {
            top_k_radix_histogram<BLOCK_SIZE, RADIX_BITS>
                <<<row_grid, BLOCK_SIZE, 0, stream>>>(
                    src, states, histograms, ncols, block_cols, blocks_per_row, shift);
            top_k_radix_select<BLOCK_SIZE, RADIX_BITS>
                <<<iter_nrows, BLOCK_SIZE, 0, stream>>>(histograms, states, blocks_per_row, shift);
        }

        top_k_radix_gather<BLOCK_SIZE, RADIX_BITS>
            <<<row_grid, BLOCK_SIZE, 0, stream>>>(
                src, dst, histograms, states, ncols, block_cols, k, blocks_per_row);

        src += (size_t) ncols * iter_nrows;
        dst += (size_t) k     * iter_nrows;
    }
}

static void top_k_argsort_cuda(
        ggml_cuda_pool & pool,
        const float * src, int * dst, int ncols, int64_t nrows, int k, bool use_cub, cudaStream_t stream) {
    const int64_t chunk_nrows = ggml_cuda_chunk_nrows((size_t) ncols * sizeof(int), nrows);

    ggml_cuda_pool_alloc<int> tmp_alloc(pool, (size_t) ncols * chunk_nrows);
    int * tmp = tmp_alloc.get();

    for (int64_t i = 0; i < nrows; i += chunk_nrows) {
        const int iter_nrows = std::min(chunk_nrows, nrows - i);

        if (use_cub) {
#ifdef GGML_CUDA_USE_CUB
            argsort_f32_i32_cuda_cub(pool, src, tmp, ncols, iter_nrows, GGML_SORT_ORDER_DESC, stream);
#else
            GGML_ABORT("CUB is not available");
#endif // GGML_CUDA_USE_CUB
        } else {
            argsort_f32_i32_cuda_bitonic(src, tmp, ncols, iter_nrows, GGML_SORT_ORDER_DESC, stream);
        }
        CUDA_CHECK(cudaMemcpy2DAsync(dst, k * sizeof(int), tmp, ncols * sizeof(int), k * sizeof(int), iter_nrows,
                                     cudaMemcpyDeviceToDevice, stream));

        src += (size_t) ncols * iter_nrows;
        dst += (size_t) k     * iter_nrows;
    }
}


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
template <int ITEMS, bool CAND, bool CANONICAL = false>
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
            const float value = src[base + col];
            if constexpr (CANONICAL) {
                // Keep numeric ties equal without changing the bits packed in the result.
                if (isnan(value)) { return 1; }
                return topk_radix_key(value == 0.0f ? 0.0f : value);
            }
            return topk_radix_key(value);
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

template <bool CAND, bool CANONICAL = false>
static void topk_radix_launch(const int nblocks, const float * src, const uint32_t * src_key, const int * src_col,
                              int * out_col, uint32_t * out_key, const int64_t row_stride, const int seg_len,
                              const int nseg, const int ncols, const int k, cudaStream_t stream) {
    if (seg_len <= 4*TOPK_RADIX_BLOCK) {
        topk_radix<4, CAND, CANONICAL><<<nblocks, TOPK_RADIX_BLOCK, 0, stream>>>(src, src_key, src_col, out_col, out_key, row_stride, seg_len, nseg, ncols, k);
    } else if (seg_len <= 8*TOPK_RADIX_BLOCK) {
        topk_radix<8, CAND, CANONICAL><<<nblocks, TOPK_RADIX_BLOCK, 0, stream>>>(src, src_key, src_col, out_col, out_key, row_stride, seg_len, nseg, ncols, k);
    } else if (seg_len <= 16*TOPK_RADIX_BLOCK) {
        topk_radix<16, CAND, CANONICAL><<<nblocks, TOPK_RADIX_BLOCK, 0, stream>>>(src, src_key, src_col, out_col, out_key, row_stride, seg_len, nseg, ncols, k);
    } else {
        topk_radix<0, CAND, CANONICAL><<<nblocks, TOPK_RADIX_BLOCK, 0, stream>>>(src, src_key, src_col, out_col, out_key, row_stride, seg_len, nseg, ncols, k);
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

static __global__ void topk_pack_candidates(const float * src, const int * ids, uint64_t * dst,
        const int ncols, const int k, const int count, const int parts, const int offset, const int64_t total) {
    const int64_t i = int64_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (i >= total) { return; }
    const int candidate = i % k;
    const int part = (i / k) % parts;
    const int64_t row = i / (int64_t(k)*parts);
    uint64_t record = UINT32_MAX;
    if (part == 0 && candidate < count) {
        const int id = ids[row*count + candidate];
        record = (uint64_t(__float_as_uint(src[row*ncols + id])) << 32) | uint32_t(offset + id);
    }
    dst[i] = record;
}

void ggml_cuda_op_top_k_candidates(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src = dst->src[0];
    GGML_ASSERT(src->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_I64);
    GGML_ASSERT(ggml_is_contiguous(src) && ggml_is_contiguous(dst));
    const int ncols = src->ne[0];
    const int nrows = dst->ne[2];
    const int k = dst->ne[0];
    const int count = std::min(k, ncols);
    const float * values = (const float *) src->data;
    cudaStream_t stream = ctx.stream();
    ggml_cuda_pool_alloc<int> ids(ctx.pool(), std::max(size_t(1), size_t(nrows)*count));
    if (count > 0 && nrows > 0) {
        if (ncols <= 16*TOPK_RADIX_BLOCK || 2*int64_t(count) > TOPK_RADIX_TILE) {
            topk_radix_launch<false, true>(nrows, values, nullptr, nullptr, ids.get(), nullptr,
                                          ncols, ncols, 1, ncols, count, stream);
        } else {
            const int tile = TOPK_RADIX_TILE;
            const int ntiles = (ncols + tile - 1) / tile;
            const int ncand = ntiles*count;
            ggml_cuda_pool_alloc<uint32_t> keys(ctx.pool(), size_t(nrows)*ncand);
            ggml_cuda_pool_alloc<int> cols(ctx.pool(), size_t(nrows)*ncand);
            topk_radix_launch<false, true>(nrows*ntiles, values, nullptr, nullptr, cols.get(), keys.get(),
                                          ncols, tile, ntiles, ncols, count, stream);
            topk_radix_launch<true>(nrows, nullptr, keys.get(), cols.get(), ids.get(), nullptr,
                                   ncand, ncand, 1, ncand, count, stream);
        }
    }
    const int64_t total = ggml_nelements(dst);
    if (total > 0) {
        topk_pack_candidates<<<(total + 255)/256, 256, 0, stream>>>(values, ids.get(), (uint64_t *) dst->data,
            ncols, k, count, dst->ne[1], ggml_get_op_params_i32(dst, 0), total);
    }
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

    const int device = ggml_cuda_get_device();

#ifdef CUB_TOP_K_AVAILABLE
    // a single row always uses DeviceTopK if available
    const bool bitonic_short    = nrows > 1 && ncols <= GGML_CUDA_TOP_K_NCOLS_THRESHOLD_BITONIC;
#else
    const bool bitonic_short    = ncols <= GGML_CUDA_TOP_K_NCOLS_THRESHOLD_BITONIC;
#endif // CUB_TOP_K_AVAILABLE
    const bool bitonic_few_rows = nrows > GGML_CUDA_TOP_K_NROWS_THRESHOLD &&
                                  ncols <= GGML_CUDA_TOP_K_NCOLS_THRESHOLD_BITONIC_FEW_ROWS &&
                                  nrows <= ggml_cuda_info().devices[device].nsm;

    if (bitonic_short || bitonic_few_rows) {
        // the padded row must fit in shared memory
        const int ncols_pad = next_power_of_2(ncols);
        if (ncols_pad * sizeof(int) <= ggml_cuda_info().devices[device].smpb) {
            top_k_argsort_cuda(pool, src0_d, dst_d, ncols, nrows, k, false, stream);
            return;
        }
    }

    if (nrows > GGML_CUDA_TOP_K_NROWS_THRESHOLD) {
        top_k_radix_cuda(pool, src0_d, dst_d, ncols, nrows, k, stream);
        return;
    }

#ifdef CUB_TOP_K_AVAILABLE
    // TODO: Assess perf of `DeviceBatchedTopK` for multi-row TopK & CCCL >= 3.5.0, re-running perf sweep of https://github.com/ggml-org/llama.cpp/pull/28713
    for (int64_t i = 0; i < nrows; i++) {
        top_k_cub(pool, src0_d + i * ncols, dst_d + i * k, ncols, k, stream);
    }
#elif defined(GGML_CUDA_USE_CUB)  // CUB_TOP_K_AVAILABLE
    if (ncols <= GGML_CUDA_TOP_K_NCOLS_THRESHOLD_ARGSORT) {
        top_k_argsort_cuda(pool, src0_d, dst_d, ncols, nrows, k, true, stream);
    } else {
        top_k_radix_cuda(pool, src0_d, dst_d, ncols, nrows, k, stream);
    }
#else                             // GGML_CUDA_USE_CUB
    top_k_radix_cuda(pool, src0_d, dst_d, ncols, nrows, k, stream);
#endif                            // CUB_TOP_K_AVAILABLE
}
