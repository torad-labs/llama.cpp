#include "mmvq-pq2-mma.cuh"
#include "unary.cuh"

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
// one 32-weight chunk's integer dot for 16 rows x 8 columns. vec_dot_pq2_0_q8_1's d2 * d8 * sumi is regrouped: a block's
// four chunks sum d8 * sumi, and its d2 scales that sum once.
//
// The pointers carry no __restrict__: with PDL a restrict load may compile to ld.global.nc, which the compiler can move
// above the grid dependency wait (upstream #24030), and the tokens and the bias are written by the kernels before this one.
//
// Work: the (16*RG-row tile, box) iterations split evenly over all warps of a one-block-per-SM grid (stream-K). A warp
// whose share covers a whole tile writes it; a tile shared by several warps is summed from their partials by the last to
// arrive, in warp order, so the result does not depend on timing.
//
// A gated FFN (NMAT 2): the up and the gate matrices' boxes of the same rows land in one slot under one barrier, their
// MMAs share each token fragment (one quantized src1 for both), and the tile's output is up * silu(gate), as
// mul_mat_vec_q's fused SWIGLU writes it; the two products and the GLU are one launch at any 1-8 columns.

#define PQ2_MMA_BOX_BLOCKS 8
#define PQ2_MMA_ROW_BYTES  (PQ2_MMA_BOX_BLOCKS * (int) sizeof(block_pq2_0))
#define PQ2_MMA_MAX_COLS   8

static_assert(sizeof(block_pq2_0) == 34, "PQ2_0 block layout");
static_assert(PQ2_MMA_ROW_BYTES % 16 == 0, "a box row must be a whole number of 16-byte TMA units");
static_assert(16 * PQ2_MMA_ROW_BYTES % 128 == 0, "a matrix's box within a slot starts 128-byte aligned");

// one matrix's box: 16*rg rows x 272 bytes
static constexpr __host__ __device__ int pq2_mma_slot_bytes(int rg) {
    return 16 * rg * PQ2_MMA_ROW_BYTES;
}

// a slot holds nmat boxes of the same rows
static constexpr __host__ __device__ int pq2_mma_slot_stride(int rg, int nmat) {
    return (nmat * pq2_mma_slot_bytes(rg) + 127) / 128 * 128; // TMA destinations are 128-byte aligned
}

static constexpr size_t pq2_mma_smem_bytes(int nwarps, int nslots, int rg, int nmat) {
    return (size_t) nwarps * nslots * pq2_mma_slot_stride(rg, nmat) + (size_t) nwarps * nslots * sizeof(uint64_t);
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

// The accumulator starts at 0x4B400000, the bits of 12582912.0f (1.5 * 2^23), so each s32 result is the bits of
// 12582912 + dot as an fp32 (exact: |dot| <= 32 * 2 * 127 < 2^22), and one FADD converts it instead of a quarter-rate I2F.
#define PQ2_MMA_F32_MAGIC_BITS 0x4B400000
#define PQ2_MMA_F32_MAGIC      12582912.0f

static __device__ __forceinline__ void pq2_mma_s8(int & c0, int & c1, int & c2, int & c3,
                                                  const int a0, const int a1, const int a2, const int a3,
                                                  const int b0, const int b1) {
    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %10, %10, %10};"
        : "=r"(c0), "=r"(c1), "=r"(c2), "=r"(c3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(PQ2_MMA_F32_MAGIC_BITS));
}

static __device__ __forceinline__ float pq2_mma_dot(const int c) {
    return __int_as_float(c) - PQ2_MMA_F32_MAGIC;
}

// The warp whose share [w*T/W, (w+1)*T/W) holds iteration x: the largest w with floor(w*T/W) <= x.
static __device__ __forceinline__ int64_t pq2_warp_of(int64_t x, int64_t T, int64_t W) {
    return ((x + 1) * W + T - 1) / T - 1;
}

static __device__ __forceinline__ int64_t pq2_share_begin(int64_t w, int64_t T, int64_t W) {
    return w * T / W;
}
#endif // PQ2_MMA_AVAILABLE

// nmat 1: dst = W x (+ x_bias); nmat 2: dst = (W x) * silu(G x), W's map in tmap and G's in tmap_gate
template <int nwarps, int nslots, int rg, int nmat, bool evict_first, bool pre_sync_issue, bool has_bias>
__launch_bounds__(nwarps*32, 1)
static __global__ void mmvq_pq2_mma(
        const __grid_constant__ CUtensorMap tmap, const __grid_constant__ CUtensorMap tmap_gate, const block_q8_1 * y,
        const float * x_bias, float * dst, float * ws, int * counters,
        const int nrows, const int ncols, const int nk, const int64_t total_iters,
        const int stride_col_y, const int stride_col_dst) {
    static_assert(nmat == 1 || (nmat == 2 && !has_bias), "a gated product fuses no bias");
#ifdef PQ2_MMA_AVAILABLE
    constexpr int tile_rows   = 16 * rg;
    constexpr int slot_bytes  = pq2_mma_slot_bytes(rg); // one matrix's box
    constexpr int slot_stride = pq2_mma_slot_stride(rg, nmat);
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

    // warp-local iteration j is (tile, box) = divmod(it_begin + j, nk), in slot j % nslots: the up box, then the gate's
    const auto issue = [&](const int64_t j) {
        const int64_t it   = it_begin + j;
        const int     slot = j % nslots;
        const int     c0   = (int) (it % nk) * (PQ2_MMA_ROW_BYTES/2);
        const int     c1   = (int) (it / nk) * tile_rows;
        pq2_mbar_arrive_expect_tx(&bars[slot], nmat * slot_bytes);
        pq2_tma_load_2d<evict_first>(ring + slot*slot_stride, &tmap, c0, c1, &bars[slot], policy);
        if constexpr (nmat == 2) {
            pq2_tma_load_2d<evict_first>(ring + slot*slot_stride + slot_bytes, &tmap_gate, c0, c1, &bars[slot], policy);
        }
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

        float acc[nmat][rg][4] = {{{0.0f}}};

        for (int k = k0; k < k1; ++k, ++j) {
            const int slot = j % nslots;
            pq2_mbar_wait(&bars[slot], (uint32_t) ((j / nslots) & 1));
            const char * sw  = ring + slot*slot_stride;
            const int    kb0 = k * PQ2_MMA_BOX_BLOCKS; // the slot's first PQ2_0 block along the row

#pragma unroll
            for (int b = 0; b < PQ2_MMA_BOX_BLOCKS; ++b) {
                float blk[nmat][rg][4] = {{{0.0f}}}; // the block's sum over its 4 chunks of d8 * dot; d2 scales it once
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
                    for (int m = 0; m < nmat; ++m) {
#pragma unroll
                        for (int r = 0; r < rg; ++r) {
                            const char * sm  = sw + m*slot_bytes;
                            const int    off = b*34 + 2 + 2*(4*q + t); // int16 4q+t of the block: weights 32q+8t..+7
                            const int    qlo = *(const uint16_t *) (sm + (r*16 + g    )*PQ2_MMA_ROW_BYTES + off);
                            const int    qhi = *(const uint16_t *) (sm + (r*16 + g + 8)*PQ2_MMA_ROW_BYTES + off);

                            int c0, c1, c2, c3;
                            pq2_mma_s8(c0, c1, c2, c3,
                                __byte_perm(pool, pool, qlo), __byte_perm(pool, pool, qhi),
                                __byte_perm(pool, pool, qlo >> 2), __byte_perm(pool, pool, qhi >> 2), b0, b1);

                            blk[m][r][0] += d8_0 * pq2_mma_dot(c0);
                            blk[m][r][1] += d8_1 * pq2_mma_dot(c1);
                            blk[m][r][2] += d8_0 * pq2_mma_dot(c2);
                            blk[m][r][3] += d8_1 * pq2_mma_dot(c3);
                        }
                    }
                }
#pragma unroll
                for (int m = 0; m < nmat; ++m) {
#pragma unroll
                    for (int r = 0; r < rg; ++r) {
                        const char * sm    = sw + m*slot_bytes;
                        const float  d2_lo = __half2float(*(const half *) (sm + (r*16 + g    )*PQ2_MMA_ROW_BYTES + b*34));
                        const float  d2_hi = __half2float(*(const half *) (sm + (r*16 + g + 8)*PQ2_MMA_ROW_BYTES + b*34));
                        acc[m][r][0] += d2_lo * blk[m][r][0];
                        acc[m][r][1] += d2_lo * blk[m][r][1];
                        acc[m][r][2] += d2_hi * blk[m][r][2];
                        acc[m][r][3] += d2_hi * blk[m][r][3];
                    }
                }
            }

            __syncwarp(); // every lane has read this slot: refill it
            if (lane == 0 && j + nslots < n_iters) {
                issue(j + nslots);
            }
        }

        const auto store = [&](const float (&v)[nmat][rg][4]) {
#pragma unroll
            for (int r = 0; r < rg; ++r) {
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const int row = tile*tile_rows + r*16 + g + (i >= 2 ? 8 : 0);
                    const int col = 2*t + (i & 1);
                    if (row < nrows && col < ncols) {
                        float out = v[0][r][i];
                        if constexpr (nmat == 2) {
                            out *= ggml_cuda_op_silu_single(v[1][r][i]);
                        }
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
        constexpr int nacc       = nmat * rg * 4; // a lane's floats in a partial
        const int     first_tile = (int) (it_begin / nk);
        float       * part       = ws + ((w*2 + (tile == first_tile ? 0 : 1))*32 + lane) * nacc;
#pragma unroll
        for (int m = 0; m < nmat; ++m) {
#pragma unroll
            for (int r = 0; r < rg; ++r) {
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    part[(m*rg + r)*4 + i] = acc[m][r][i];
                }
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

        float sum[nmat][rg][4] = {{{0.0f}}};
        for (int64_t cw = w_lo; cw <= w_hi; ++cw) {
            const int64_t cb = pq2_share_begin(cw, total_iters, W);
            if (pq2_share_begin(cw + 1, total_iters, W) == cb) {
                continue;
            }
            const float * p = ws + ((cw*2 + (tile == (int) (cb / nk) ? 0 : 1))*32 + lane) * nacc;
#pragma unroll
            for (int m = 0; m < nmat; ++m) {
#pragma unroll
                for (int r = 0; r < rg; ++r) {
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        sum[m][r][i] += __ldcg(p + (m*rg + r)*4 + i);
                    }
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

static pq2_mma_config pq2_mma_parse_config(const char * name, pq2_mma_config c) {
    const char * s = getenv(name);
    if (s != nullptr) {
        int v[5] = { c.nwarps, c.nslots, c.rg, c.evict_first, c.pre_sync_issue };
        if (sscanf(s, "%d,%d,%d,%d,%d", &v[0], &v[1], &v[2], &v[3], &v[4]) == 5) {
            c = { v[0], v[1], v[2], v[3] != 0, v[4] != 0 };
        } else {
            GGML_LOG_WARN("%s: %s=%s is not nwarps,nslots,rg,evict_first,pre_sync_issue\n", __func__, name, s);
        }
    }
    return c;
}

// GGML_CUDA_PQ2_MMA_CFG / GGML_CUDA_PQ2_MMA_GATE_CFG="nwarps,nslots,rg,evict_first,pre_sync_issue" override the
// defaults for a sweep. A gated slot holds two boxes, so the gated default keeps the plain one's bytes in flight per SM
// (8 warps x 1 slot x 8,704 B) inside the 99 KB a block may take.
static pq2_mma_config pq2_mma_get_config(const bool gated) {
    static const pq2_mma_config plain = pq2_mma_parse_config("GGML_CUDA_PQ2_MMA_CFG", { 8, 2, 1, true, true });
    static const pq2_mma_config gate  = pq2_mma_parse_config("GGML_CUDA_PQ2_MMA_GATE_CFG", { 8, 1, 1, true, true });
    return gated ? gate : plain;
}

static bool pq2_mma_legacy() {
    static const bool legacy = [] {
        const char * s = getenv("GGML_CUDA_PQ2_MMA_LEGACY");
        return s != nullptr && atoi(s) != 0;
    }();
    return legacy;
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
// buffer per stream (two streams' launches never share it), grown only outside a graph capture (nullptr inside one when
// it is too small, and the matmul is not this kernel's). A graph's first evaluation runs uncaptured, so every matrix
// the graph holds has had its counters before the capture.
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
    const int64_t n = std::max<int64_t>(n_tiles, 1 << 16); // 256 KB: every tile of a 1M-row matrix
    if (b.first != nullptr) {
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaFree(b.first));
    }
    CUDA_CHECK(cudaMalloc(&b.first, n*sizeof(int)));
    CUDA_CHECK(cudaMemset(b.first, 0, n*sizeof(int)));
    b.second = n;
    return b.first;
}

static int64_t pq2_mma_n_tiles(const pq2_mma_config & c, int64_t nrows_x) {
    return (nrows_x + 16*c.rg - 1) / (16*c.rg);
}

bool ggml_cuda_mmvq_pq2_mma_usable(int cc, const void * vx, const void * vgate, int64_t ncols_x, int64_t nrows_x,
                                   int64_t stride_row_x, int64_t ncols_dst, cudaStream_t stream) {
    return !pq2_mma_legacy() && GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_BLACKWELL && cc < GGML_CUDA_CC_DGX_SPARK &&
        ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_BLACKWELL &&
        ncols_dst >= 1 && ncols_dst <= PQ2_MMA_MAX_COLS &&
        ncols_x % 1024 == 0 && stride_row_x*(int64_t) sizeof(block_pq2_0) % 16 == 0 && (uintptr_t) vx % 16 == 0 &&
        (uintptr_t) vgate % 16 == 0 &&
        nrows_x < (int64_t) 1 << 31 && ncols_x*(int64_t) sizeof(block_pq2_0)/QK_PQ2_0 < ((int64_t) 1 << 32) &&
        pq2_mma_counters(stream, pq2_mma_n_tiles(pq2_mma_get_config(vgate != nullptr), nrows_x)) != nullptr;
}

template <int nwarps, int nslots, int rg, int nmat, bool evict_first, bool pre_sync_issue>
static void pq2_mma_launch(const CUtensorMap & tmap, const CUtensorMap & tmap_gate, const block_q8_1 * y,
        const float * x_bias, float * dst, float * ws, int * counters, int nrows, int ncols, int nk, int64_t total_iters,
        int stride_col_y, int stride_col_dst, int nblocks, cudaStream_t stream) {
    constexpr size_t smem = pq2_mma_smem_bytes(nwarps, nslots, rg, nmat);
    static_assert(smem <= 99*1024, "a block takes at most 99 KB of shared memory");
    const ggml_cuda_kernel_launch_params params(dim3(nblocks), dim3(nwarps*32), smem, stream);
    if constexpr (nmat == 1) {
        if (x_bias != nullptr) {
            CUDA_SET_SHARED_MEMORY_LIMIT((mmvq_pq2_mma<nwarps, nslots, rg, 1, evict_first, pre_sync_issue, true>), smem);
            ggml_cuda_kernel_launch(mmvq_pq2_mma<nwarps, nslots, rg, 1, evict_first, pre_sync_issue, true>, params,
                tmap, tmap_gate, y, x_bias, dst, ws, counters, nrows, ncols, nk, total_iters, stride_col_y, stride_col_dst);
            return;
        }
    } else {
        GGML_ASSERT(x_bias == nullptr);
    }
    CUDA_SET_SHARED_MEMORY_LIMIT((mmvq_pq2_mma<nwarps, nslots, rg, nmat, evict_first, pre_sync_issue, false>), smem);
    ggml_cuda_kernel_launch(mmvq_pq2_mma<nwarps, nslots, rg, nmat, evict_first, pre_sync_issue, false>, params,
        tmap, tmap_gate, y, x_bias, dst, ws, counters, nrows, ncols, nk, total_iters, stride_col_y, stride_col_dst);
}

template <int nwarps, int nslots, int rg, int nmat>
static void pq2_mma_launch_flags(const pq2_mma_config & c, const CUtensorMap & tmap, const CUtensorMap & tmap_gate,
        const block_q8_1 * y, const float * x_bias, float * dst, float * ws, int * counters, int nrows, int ncols, int nk,
        int64_t total_iters, int stride_col_y, int stride_col_dst, int nblocks, cudaStream_t stream) {
#define PQ2_MMA_LAUNCH(EF, PRE) pq2_mma_launch<nwarps, nslots, rg, nmat, EF, PRE>(tmap, tmap_gate, y, x_bias, dst, ws,    \
        counters, nrows, ncols, nk, total_iters, stride_col_y, stride_col_dst, nblocks, stream)
    if (c.evict_first) {
        if (c.pre_sync_issue) { PQ2_MMA_LAUNCH(true,  true);  } else { PQ2_MMA_LAUNCH(true,  false); }
    } else {
        if (c.pre_sync_issue) { PQ2_MMA_LAUNCH(false, true);  } else { PQ2_MMA_LAUNCH(false, false); }
    }
#undef PQ2_MMA_LAUNCH
}

// weights as 2-byte elements, [row_bytes/2, nrows] with the row stride; one box = tile_rows rows x 272 bytes
static CUtensorMap pq2_mma_tensor_map(const void * vx, int64_t row_bytes, int64_t nrows_x, int64_t stride_row_x,
                                      int tile_rows) {
    CUtensorMap tmap;
    const cuuint64_t dims[2]      = { (cuuint64_t) (row_bytes / 2), (cuuint64_t) nrows_x };
    const cuuint64_t strides[1]   = { (cuuint64_t) (stride_row_x * (int64_t) sizeof(block_pq2_0)) };
    const cuuint32_t box[2]       = { (cuuint32_t) (PQ2_MMA_ROW_BYTES / 2), (cuuint32_t) tile_rows };
    const cuuint32_t elem_str[2]  = { 1, 1 };
    const CUresult res = pq2_mma_encode_fn()(&tmap, CU_TENSOR_MAP_DATA_TYPE_UINT16, 2, const_cast<void *>(vx), dims,
        strides, box, elem_str, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    GGML_ASSERT(res == CUDA_SUCCESS && "cuTensorMapEncodeTiled for the PQ2_0 weights");
    return tmap;
}

void ggml_cuda_mmvq_pq2_mma(ggml_backend_cuda_context & ctx, const void * vx, const void * vgate, const void * vy,
                            const float * x_bias, float * dst, int64_t ncols_x, int64_t nrows_x, int64_t ncols_dst,
                            int64_t stride_row_x, int64_t stride_col_y, int64_t stride_col_dst, cudaStream_t stream) {
    const bool           gated     = vgate != nullptr;
    const pq2_mma_config c         = pq2_mma_get_config(gated);
    const int            tile_rows = 16 * c.rg;

    const int64_t row_bytes   = ncols_x / QK_PQ2_0 * (int64_t) sizeof(block_pq2_0);
    const int     nk          = (int) (ncols_x / (QK_PQ2_0 * PQ2_MMA_BOX_BLOCKS));
    const int64_t n_tiles     = pq2_mma_n_tiles(c, nrows_x);
    const int64_t total_iters = n_tiles * nk;

    int * counters = pq2_mma_counters(stream, n_tiles);
    GGML_ASSERT(counters != nullptr && "ggml_cuda_mmvq_pq2_mma_usable holds the counters");

    const CUtensorMap tmap      = pq2_mma_tensor_map(vx, row_bytes, nrows_x, stride_row_x, tile_rows);
    const CUtensorMap tmap_gate = gated ? pq2_mma_tensor_map(vgate, row_bytes, nrows_x, stride_row_x, tile_rows) : tmap;

    const int nsm     = ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;
    const int nblocks = (int) std::min<int64_t>(nsm, (total_iters + c.nwarps - 1) / c.nwarps);

    ggml_cuda_pool_alloc<float> ws(ctx.pool(), (size_t) nblocks * c.nwarps * 2 * 32 * 4 * c.rg * (gated ? 2 : 1));

#define PQ2_MMA_CASE(NW, NS, RG, NMAT)                                                                                  \
    if (c.nwarps == (NW) && c.nslots == (NS) && c.rg == (RG) && (gated ? 2 : 1) == (NMAT)) {                            \
        pq2_mma_launch_flags<NW, NS, RG, NMAT>(c, tmap, tmap_gate, (const block_q8_1 *) vy, x_bias, dst, ws.get(),     \
            counters, (int) nrows_x, (int) ncols_dst, nk, total_iters, (int) stride_col_y, (int) stride_col_dst,        \
            nblocks, stream);                                                                                          \
        return;                                                                                                        \
    }
    // (nwarps, nslots, rg) within the 99 KB a block may take. One matrix: 8x2x1 72 KB, 10x2x1 90, 6x3x1 81, 4x2x2 70,
    // 2x2x2 35. Gated (a slot of two boxes): 8x1x1 72 KB, 10x1x1 90, 6x1x1 54, 4x2x1 72.
    PQ2_MMA_CASE(8,  2, 1, 1)
    PQ2_MMA_CASE(10, 2, 1, 1)
    PQ2_MMA_CASE(6,  2, 1, 1)
    PQ2_MMA_CASE(6,  3, 1, 1)
    PQ2_MMA_CASE(4,  2, 1, 1)
    PQ2_MMA_CASE(4,  3, 1, 1)
    PQ2_MMA_CASE(4,  2, 2, 1)
    PQ2_MMA_CASE(2,  2, 2, 1)
    PQ2_MMA_CASE(8,  1, 1, 2)
    PQ2_MMA_CASE(10, 1, 1, 2)
    PQ2_MMA_CASE(6,  1, 1, 2)
    PQ2_MMA_CASE(4,  2, 1, 2)
#undef PQ2_MMA_CASE
    GGML_ABORT("%s: no instance for nwarps %d nslots %d rg %d%s", gated ? "GGML_CUDA_PQ2_MMA_GATE_CFG" : "GGML_CUDA_PQ2_MMA_CFG",
        c.nwarps, c.nslots, c.rg, gated ? " (gated)" : "");
}
