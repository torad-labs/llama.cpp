#include "mmvq-pq2-mma.cuh"

#include <cuda.h>
#include <cudaTypedefs.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <utility>

// The weights stream through one ring per warp: each slot is one TMA box of 16*RG rows x 8 PQ2_0 blocks (8 x 34 B =
// 272 B of a row, 1,024 weights), 4,352 B per 16 rows, which a K that is a multiple of 1024 always starts on a 16-byte
// boundary. A 272 B row stride leaves the int16 reads below conflict-free without a swizzle (rows g land on banks 4g).
// The warp's lane 0 fills its own slots (QGRE's ring: no producer warp, no cross-warp barrier), a slot is refilled right
// after the warp has read it, and the tokens (q8_1, at most 8 columns, shared by every warp) are read through L1.
//
// Tensor cores: mma.m16n8k32 s8, weights in A (16 rows), tokens in B (8 columns, those past ncols zero). A PQ2_0 int16
// holds 8 weights as 2-bit codes (0 -> -1, 1 -> 0, 2 -> 1, 3 -> 2); __byte_perm on it gives the even and the odd four as
// int8, which go to the fragment's two k halves, and the token's 8 int8 are split the same way, so each MMA is exactly
// one 32-weight chunk's integer dot for 16 rows x 8 columns, rescaled per chunk as vec_dot_pq2_0_q8_1 does (d2 * d8 * sumi).
//
// Work: the (16*RG-row tile, box) iterations split evenly over all warps of a one-block-per-SM grid (stream-K). A warp
// whose share covers a whole tile writes it; a tile shared by several warps is summed from their partials by the last to
// arrive, in warp order, so the result does not depend on timing.

#define PQ2_MMA_BOX_BLOCKS 8
#define PQ2_MMA_ROW_BYTES  (PQ2_MMA_BOX_BLOCKS * (int) sizeof(block_pq2_0))
#define PQ2_MMA_MAX_COLS   8

static_assert(sizeof(block_pq2_0) == 34, "PQ2_0 block layout");
static_assert(PQ2_MMA_ROW_BYTES % 16 == 0, "a box row must be a whole number of 16-byte TMA units");

static constexpr int pq2_mma_slot_bytes(int rg) {
    return 16 * rg * PQ2_MMA_ROW_BYTES;
}

static constexpr int pq2_mma_slot_stride(int rg) {
    return (pq2_mma_slot_bytes(rg) + 127) / 128 * 128; // TMA destinations are 128-byte aligned
}

static constexpr size_t pq2_mma_smem_bytes(int nwarps, int nslots, int rg) {
    return (size_t) nwarps * nslots * pq2_mma_slot_stride(rg) + (size_t) nwarps * nslots * sizeof(uint64_t);
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_HOPPER && !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#define PQ2_MMA_AVAILABLE
#endif

#ifdef PQ2_MMA_AVAILABLE
static __device__ __forceinline__ uint32_t pq2_smem_u32(const void * p) {
    return (uint32_t) __cvta_generic_to_shared(p);
}

static __device__ __forceinline__ void pq2_mbar_init(uint64_t * bar, uint32_t count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(pq2_smem_u32(bar)), "r"(count) : "memory");
}

static __device__ __forceinline__ void pq2_mbar_arrive_expect_tx(uint64_t * bar, uint32_t bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(pq2_smem_u32(bar)), "r"(bytes) : "memory");
}

static __device__ __forceinline__ void pq2_mbar_wait(uint64_t * bar, uint32_t parity) {
    const uint32_t addr = pq2_smem_u32(bar);
    uint32_t done = 0;
    do {
        asm volatile(
            "{\n"
            ".reg .pred p;\n"
            "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
            "selp.u32 %0, 1, 0, p;\n"
            "}\n"
            : "=r"(done) : "r"(addr), "r"(parity) : "memory");
    } while (!done);
}

template <bool evict_first>
static __device__ __forceinline__ void pq2_tma_load_2d(void * dst, const CUtensorMap * tmap, int c0, int c1, uint64_t * bar,
                                                       uint64_t policy) {
    if constexpr (evict_first) {
        asm volatile(
            "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint"
            " [%0], [%1, {%2, %3}], [%4], %5;"
            :: "r"(pq2_smem_u32(dst)), "l"((uint64_t) tmap), "r"(c0), "r"(c1), "r"(pq2_smem_u32(bar)), "l"(policy)
            : "memory");
    } else {
        GGML_UNUSED(policy);
        asm volatile(
            "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
            " [%0], [%1, {%2, %3}], [%4];"
            :: "r"(pq2_smem_u32(dst)), "l"((uint64_t) tmap), "r"(c0), "r"(c1), "r"(pq2_smem_u32(bar))
            : "memory");
    }
}

static __device__ __forceinline__ void pq2_mma_s8(int & c0, int & c1, int & c2, int & c3,
                                                  const int a0, const int a1, const int a2, const int a3,
                                                  const int b0, const int b1) {
    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %10, %10, %10};"
        : "=r"(c0), "=r"(c1), "=r"(c2), "=r"(c3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(0));
}

// The warp whose share [w*T/W, (w+1)*T/W) holds iteration x: the largest w with floor(w*T/W) <= x.
static __device__ __forceinline__ int64_t pq2_warp_of(int64_t x, int64_t T, int64_t W) {
    return ((x + 1) * W + T - 1) / T - 1;
}

static __device__ __forceinline__ int64_t pq2_share_begin(int64_t w, int64_t T, int64_t W) {
    return w * T / W;
}
#endif // PQ2_MMA_AVAILABLE

template <int nwarps, int nslots, int rg, bool evict_first, bool pre_sync_issue, bool has_bias>
__launch_bounds__(nwarps*32, 1)
static __global__ void mmvq_pq2_mma(
        const __grid_constant__ CUtensorMap tmap, const block_q8_1 * __restrict__ y, const float * __restrict__ x_bias,
        float * __restrict__ dst, float * __restrict__ ws, int * __restrict__ counters,
        const int nrows, const int ncols, const int nk, const int64_t total_iters,
        const int stride_col_y, const int stride_col_dst) {
#ifdef PQ2_MMA_AVAILABLE
    constexpr int tile_rows   = 16 * rg;
    constexpr int slot_bytes  = pq2_mma_slot_bytes(rg);
    constexpr int slot_stride = pq2_mma_slot_stride(rg);
    constexpr int pool        = 0x020100FF; // byte i of a code's selector: 0 -> 0xFF (-1), 1 -> 0, 2 -> 1, 3 -> 2

    extern __shared__ __align__(128) char smem[];

    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int g    = lane >> 2;
    const int t    = lane & 3;

    char     * ring = smem + warp * nslots * slot_stride;
    uint64_t * bars = (uint64_t *) (smem + nwarps * nslots * slot_stride) + warp * nslots;

    const int64_t W        = (int64_t) gridDim.x * nwarps;
    const int64_t w        = (int64_t) blockIdx.x * nwarps + warp;
    const int64_t it_begin = pq2_share_begin(w,     total_iters, W);
    const int64_t it_end   = pq2_share_begin(w + 1, total_iters, W);
    const int64_t n_iters  = it_end - it_begin;

    ggml_cuda_pdl_lc();

    uint64_t policy = 0;
    if constexpr (evict_first) {
        asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(policy));
    }

    if (lane == 0) {
#pragma unroll
        for (int s = 0; s < nslots; ++s) {
            pq2_mbar_init(&bars[s], 1);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncwarp();

    // warp-local iteration j is (tile, box) = divmod(it_begin + j, nk), in slot j % nslots
    const auto issue = [&](const int64_t j) {
        const int64_t it   = it_begin + j;
        const int     slot = j % nslots;
        pq2_mbar_arrive_expect_tx(&bars[slot], slot_bytes);
        pq2_tma_load_2d<evict_first>(ring + slot*slot_stride, &tmap, (int) (it % nk) * (PQ2_MMA_ROW_BYTES/2),
                                     (int) (it / nk) * tile_rows, &bars[slot], policy);
    };

    // the weights are constant, so the first slots are requested before the wait for the previous kernel; the tokens,
    // the bias and the partials are read after it
    if constexpr (pre_sync_issue) {
        if (lane == 0) {
            for (int64_t j = 0; j < nslots && j < n_iters; ++j) {
                issue(j);
            }
        }
        ggml_cuda_pdl_sync();
    } else {
        ggml_cuda_pdl_sync();
        if (lane == 0) {
            for (int64_t j = 0; j < nslots && j < n_iters; ++j) {
                issue(j);
            }
        }
    }

    const block_q8_1 * y_g  = y + (int64_t) g       * stride_col_y; // this lane's B column
    const block_q8_1 * y_c0 = y + (int64_t) (2*t)   * stride_col_y; // the columns of its C values
    const block_q8_1 * y_c1 = y + (int64_t) (2*t+1) * stride_col_y;

    int64_t j = 0;
    while (j < n_iters) {
        const int64_t it0  = it_begin + j;
        const int     tile = (int) (it0 / nk);
        const int     k0   = (int) (it0 % nk);
        const int     k1   = (int) min((int64_t) nk, k0 + (n_iters - j));

        float acc[rg][4] = {{0.0f}};

        for (int k = k0; k < k1; ++k, ++j) {
            const int slot = j % nslots;
            pq2_mbar_wait(&bars[slot], (uint32_t) ((j / nslots) & 1));
            const char * sw  = ring + slot*slot_stride;
            const int    kb0 = k * PQ2_MMA_BOX_BLOCKS; // the slot's first PQ2_0 block along the row

#pragma unroll
            for (int b = 0; b < PQ2_MMA_BOX_BLOCKS; ++b) {
                float d2[rg][2];
#pragma unroll
                for (int r = 0; r < rg; ++r) {
                    d2[r][0] = __half2float(*(const half *) (sw + (r*16 + g    )*PQ2_MMA_ROW_BYTES + b*34));
                    d2[r][1] = __half2float(*(const half *) (sw + (r*16 + g + 8)*PQ2_MMA_ROW_BYTES + b*34));
                }
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    const int kq = (kb0 + b)*4 + q; // the q8_1 block of these 32 weights' tokens

                    int b0 = 0;
                    int b1 = 0;
                    if (g < ncols) {
                        const int * qs = (const int *) y_g[kq].qs;
                        const int   u  = qs[2*t + 0]; // tokens 8t..8t+3
                        const int   v  = qs[2*t + 1]; // tokens 8t+4..8t+7
                        b0 = __byte_perm(u, v, 0x6420); // the even ones, against the codes' even weights
                        b1 = __byte_perm(u, v, 0x7531); // the odd ones
                    }
                    const float d8_0 = 2*t     < ncols ? __low2float(y_c0[kq].ds) : 0.0f;
                    const float d8_1 = 2*t + 1 < ncols ? __low2float(y_c1[kq].ds) : 0.0f;

#pragma unroll
                    for (int r = 0; r < rg; ++r) {
                        const int off = b*34 + 2 + 2*(4*q + t); // int16 4q+t of the block: weights 32q+8t..+7
                        const int qlo = *(const uint16_t *) (sw + (r*16 + g    )*PQ2_MMA_ROW_BYTES + off);
                        const int qhi = *(const uint16_t *) (sw + (r*16 + g + 8)*PQ2_MMA_ROW_BYTES + off);

                        int c0, c1, c2, c3;
                        pq2_mma_s8(c0, c1, c2, c3,
                            __byte_perm(pool, pool, qlo), __byte_perm(pool, pool, qhi),
                            __byte_perm(pool, pool, qlo >> 2), __byte_perm(pool, pool, qhi >> 2), b0, b1);

                        acc[r][0] += d2[r][0] * d8_0 * (float) c0;
                        acc[r][1] += d2[r][0] * d8_1 * (float) c1;
                        acc[r][2] += d2[r][1] * d8_0 * (float) c2;
                        acc[r][3] += d2[r][1] * d8_1 * (float) c3;
                    }
                }
            }

            __syncwarp(); // every lane has read this slot: refill it
            if (lane == 0 && j + nslots < n_iters) {
                issue(j + nslots);
            }
        }

        const auto store = [&](const float (&v)[rg][4]) {
#pragma unroll
            for (int r = 0; r < rg; ++r) {
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const int row = tile*tile_rows + r*16 + g + (i >= 2 ? 8 : 0);
                    const int col = 2*t + (i & 1);
                    if (row < nrows && col < ncols) {
                        float out = v[r][i];
                        if constexpr (has_bias) {
                            out += x_bias[(int64_t) col*stride_col_dst + row];
                        }
                        dst[(int64_t) col*stride_col_dst + row] = out;
                    }
                }
            }
        };

        if (k0 == 0 && k1 == nk) {
            store(acc);
            continue;
        }

        // a tile shared with other warps: this warp's partial goes to its slot (0: the tile its share starts in, 1: the
        // one it ends in); the last contributor to arrive sums them in warp order
        const int  first_tile = (int) (it_begin / nk);
        float    * part       = ws + ((w*2 + (tile == first_tile ? 0 : 1))*32 + lane) * (rg*4);
#pragma unroll
        for (int r = 0; r < rg; ++r) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                part[r*4 + i] = acc[r][i];
            }
        }
        __threadfence();
        __syncwarp();

        const int64_t w_lo = pq2_warp_of((int64_t) tile*nk,          total_iters, W);
        const int64_t w_hi = pq2_warp_of((int64_t) (tile + 1)*nk - 1, total_iters, W);
        int last = 0;
        if (lane == 0) {
            int ncontrib = 0;
            for (int64_t cw = w_lo; cw <= w_hi; ++cw) {
                ncontrib += pq2_share_begin(cw + 1, total_iters, W) > pq2_share_begin(cw, total_iters, W);
            }
            last = atomicAdd(&counters[tile], 1) == ncontrib - 1;
        }
        last = __shfl_sync(0xFFFFFFFF, last, 0);
        if (!last) {
            continue;
        }
        __threadfence();

        float sum[rg][4] = {{0.0f}};
        for (int64_t cw = w_lo; cw <= w_hi; ++cw) {
            const int64_t cb = pq2_share_begin(cw, total_iters, W);
            if (pq2_share_begin(cw + 1, total_iters, W) == cb) {
                continue;
            }
            const float * p = ws + ((cw*2 + (tile == (int) (cb / nk) ? 0 : 1))*32 + lane) * (rg*4);
#pragma unroll
            for (int r = 0; r < rg; ++r) {
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    sum[r][i] += __ldcg(p + r*4 + i);
                }
            }
        }
        store(sum);
        if (lane == 0) {
            counters[tile] = 0; // ready for the next launch, which reads it after its PDL wait
        }
    }
#else
    GGML_UNUSED_VARS(tmap, y, x_bias, dst, ws, counters, nrows, ncols, nk, total_iters, stride_col_y, stride_col_dst);
    NO_DEVICE_CODE;
#endif // PQ2_MMA_AVAILABLE
}

// ---------------------------------------------------------------------------------------------------------------------
// host

struct pq2_mma_config {
    int  nwarps         = 8;
    int  nslots         = 2;
    int  rg             = 1;
    bool evict_first    = true;
    bool pre_sync_issue = true;
};

// GGML_CUDA_PQ2_MMA_CFG="nwarps,nslots,rg,evict_first,pre_sync_issue" overrides the default for a sweep
static pq2_mma_config pq2_mma_get_config() {
    static const pq2_mma_config cfg = [] {
        pq2_mma_config c;
        const char * s = getenv("GGML_CUDA_PQ2_MMA_CFG");
        if (s != nullptr) {
            int v[5] = { c.nwarps, c.nslots, c.rg, c.evict_first, c.pre_sync_issue };
            if (sscanf(s, "%d,%d,%d,%d,%d", &v[0], &v[1], &v[2], &v[3], &v[4]) == 5) {
                c = { v[0], v[1], v[2], v[3] != 0, v[4] != 0 };
            } else {
                GGML_LOG_WARN("%s: GGML_CUDA_PQ2_MMA_CFG=%s is not nwarps,nslots,rg,evict_first,pre_sync_issue\n", __func__, s);
            }
        }
        return c;
    }();
    return cfg;
}

static bool pq2_mma_legacy() {
    static const bool legacy = [] {
        const char * s = getenv("GGML_CUDA_PQ2_MMA_LEGACY");
        return s != nullptr && atoi(s) != 0;
    }();
    return legacy;
}

bool ggml_cuda_mmvq_pq2_mma_usable(int cc, const void * vx, int64_t ncols_x, int64_t nrows_x, int64_t stride_row_x,
                                   int64_t ncols_dst) {
    return !pq2_mma_legacy() && GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_BLACKWELL && cc < GGML_CUDA_CC_RUBIN &&
        ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_BLACKWELL &&
        ncols_dst >= 2 && ncols_dst <= PQ2_MMA_MAX_COLS &&
        ncols_x % 1024 == 0 && stride_row_x*(int64_t) sizeof(block_pq2_0) % 16 == 0 && (uintptr_t) vx % 16 == 0 &&
        nrows_x < (int64_t) 1 << 31 && ncols_x*(int64_t) sizeof(block_pq2_0)/QK_PQ2_0 < ((int64_t) 1 << 32);
}

static PFN_cuTensorMapEncodeTiled_v12000 pq2_mma_encode_fn() {
    static PFN_cuTensorMapEncodeTiled_v12000 fn = [] {
        void * p = nullptr;
        cudaDriverEntryPointQueryResult q;
        CUDA_CHECK(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &p, 12000, cudaEnableDefault, &q));
        GGML_ASSERT(q == cudaDriverEntryPointSuccess && p != nullptr);
        return (PFN_cuTensorMapEncodeTiled_v12000) p;
    }();
    return fn;
}

// the stream-K fixup's per-tile arrival counters: zeroed once, left zeroed by every launch's last contributors, one
// buffer per stream (two streams' launches never share it), grown only outside a graph capture
static int * pq2_mma_counters(cudaStream_t stream, int64_t n_tiles) {
    static std::mutex mtx;
    static std::map<std::pair<int, cudaStream_t>, std::pair<int *, int64_t>> buffers;
    std::lock_guard<std::mutex> lock(mtx);
    auto & b = buffers[{ ggml_cuda_get_device(), stream }];
    if (b.second >= n_tiles) {
        return b.first;
    }
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    CUDA_CHECK(cudaStreamIsCapturing(stream, &capture));
    if (capture != cudaStreamCaptureStatusNone) {
        return nullptr;
    }
    const int64_t n = std::max<int64_t>(n_tiles, 4096);
    if (b.first != nullptr) {
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaFree(b.first));
    }
    CUDA_CHECK(cudaMalloc(&b.first, n*sizeof(int)));
    CUDA_CHECK(cudaMemset(b.first, 0, n*sizeof(int)));
    b.second = n;
    return b.first;
}

template <int nwarps, int nslots, int rg, bool evict_first, bool pre_sync_issue>
static void pq2_mma_launch(const CUtensorMap & tmap, const block_q8_1 * y, const float * x_bias, float * dst, float * ws,
        int * counters, int nrows, int ncols, int nk, int64_t total_iters, int stride_col_y, int stride_col_dst,
        int nblocks, cudaStream_t stream) {
    constexpr size_t smem = pq2_mma_smem_bytes(nwarps, nslots, rg);
    const ggml_cuda_kernel_launch_params params(dim3(nblocks), dim3(nwarps*32), smem, stream);
    if (x_bias != nullptr) {
        CUDA_SET_SHARED_MEMORY_LIMIT((mmvq_pq2_mma<nwarps, nslots, rg, evict_first, pre_sync_issue, true>), smem);
        ggml_cuda_kernel_launch(mmvq_pq2_mma<nwarps, nslots, rg, evict_first, pre_sync_issue, true>, params,
            tmap, y, x_bias, dst, ws, counters, nrows, ncols, nk, total_iters, stride_col_y, stride_col_dst);
    } else {
        CUDA_SET_SHARED_MEMORY_LIMIT((mmvq_pq2_mma<nwarps, nslots, rg, evict_first, pre_sync_issue, false>), smem);
        ggml_cuda_kernel_launch(mmvq_pq2_mma<nwarps, nslots, rg, evict_first, pre_sync_issue, false>, params,
            tmap, y, x_bias, dst, ws, counters, nrows, ncols, nk, total_iters, stride_col_y, stride_col_dst);
    }
}

template <int nwarps, int nslots, int rg>
static void pq2_mma_launch_flags(const pq2_mma_config & c, const CUtensorMap & tmap, const block_q8_1 * y,
        const float * x_bias, float * dst, float * ws, int * counters, int nrows, int ncols, int nk, int64_t total_iters,
        int stride_col_y, int stride_col_dst, int nblocks, cudaStream_t stream) {
#define PQ2_MMA_LAUNCH(EF, PRE) pq2_mma_launch<nwarps, nslots, rg, EF, PRE>(tmap, y, x_bias, dst, ws, counters, nrows, \
        ncols, nk, total_iters, stride_col_y, stride_col_dst, nblocks, stream)
    if (c.evict_first) {
        if (c.pre_sync_issue) { PQ2_MMA_LAUNCH(true,  true);  } else { PQ2_MMA_LAUNCH(true,  false); }
    } else {
        if (c.pre_sync_issue) { PQ2_MMA_LAUNCH(false, true);  } else { PQ2_MMA_LAUNCH(false, false); }
    }
#undef PQ2_MMA_LAUNCH
}

void ggml_cuda_mmvq_pq2_mma(ggml_backend_cuda_context & ctx, const void * vx, const void * vy, const float * x_bias,
                            float * dst, int64_t ncols_x, int64_t nrows_x, int64_t ncols_dst, int64_t stride_row_x,
                            int64_t stride_col_y, int64_t stride_col_dst, cudaStream_t stream) {
    const pq2_mma_config c = pq2_mma_get_config();
    const int tile_rows = 16 * c.rg;

    const int64_t row_bytes   = ncols_x / QK_PQ2_0 * (int64_t) sizeof(block_pq2_0);
    const int     nk          = (int) (ncols_x / (QK_PQ2_0 * PQ2_MMA_BOX_BLOCKS));
    const int64_t n_tiles     = (nrows_x + tile_rows - 1) / tile_rows;
    const int64_t total_iters = n_tiles * nk;

    // weights as 2-byte elements, [row_bytes/2, nrows] with the row stride; one box = tile_rows rows x 272 bytes
    CUtensorMap tmap;
    const cuuint64_t dims[2]      = { (cuuint64_t) (row_bytes / 2), (cuuint64_t) nrows_x };
    const cuuint64_t strides[1]   = { (cuuint64_t) (stride_row_x * (int64_t) sizeof(block_pq2_0)) };
    const cuuint32_t box[2]       = { (cuuint32_t) (PQ2_MMA_ROW_BYTES / 2), (cuuint32_t) tile_rows };
    const cuuint32_t elem_str[2]  = { 1, 1 };
    const CUresult res = pq2_mma_encode_fn()(&tmap, CU_TENSOR_MAP_DATA_TYPE_UINT16, 2, const_cast<void *>(vx), dims,
        strides, box, elem_str, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    GGML_ASSERT(res == CUDA_SUCCESS && "cuTensorMapEncodeTiled for the PQ2_0 weights");

    const int nsm     = ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;
    const int nblocks = (int) std::min<int64_t>(nsm, (total_iters + c.nwarps - 1) / c.nwarps);

    int * counters = pq2_mma_counters(stream, n_tiles);
    GGML_ASSERT(counters != nullptr && "the PQ2_0 MMA counters are allocated by an uncaptured run before a capture");
    ggml_cuda_pool_alloc<float> ws(ctx.pool(), (size_t) nblocks * c.nwarps * 2 * 32 * 4 * c.rg);

#define PQ2_MMA_CASE(NW, NS, RG)                                                                                        \
    if (c.nwarps == (NW) && c.nslots == (NS) && c.rg == (RG)) {                                                        \
        pq2_mma_launch_flags<NW, NS, RG>(c, tmap, (const block_q8_1 *) vy, x_bias, dst, ws.get(), counters,            \
            (int) nrows_x, (int) ncols_dst, nk, total_iters, (int) stride_col_y, (int) stride_col_dst, nblocks, stream); \
        return;                                                                                                        \
    }
    // (nwarps, nslots, rg) within the 99 KB a block may take: 8x2x1 72 KB, 10x2x1 90, 6x3x1 81, 4x2x2 70, 2x2x2 35
    PQ2_MMA_CASE(8,  2, 1)
    PQ2_MMA_CASE(10, 2, 1)
    PQ2_MMA_CASE(6,  2, 1)
    PQ2_MMA_CASE(6,  3, 1)
    PQ2_MMA_CASE(4,  2, 1)
    PQ2_MMA_CASE(4,  3, 1)
    PQ2_MMA_CASE(4,  2, 2)
    PQ2_MMA_CASE(2,  2, 2)
#undef PQ2_MMA_CASE
    GGML_ABORT("GGML_CUDA_PQ2_MMA_CFG: no instance for nwarps %d nslots %d rg %d", c.nwarps, c.nslots, c.rg);
}
