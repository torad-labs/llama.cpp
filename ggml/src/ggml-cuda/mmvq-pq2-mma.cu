#include "mmvq-pq2-mma.cuh"
#include "unary.cuh"

#include <cuda.h>
#include <cudaTypedefs.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>

// Weights: a tile is 16 rows, and each block (one per SM) owns a contiguous run of whole tiles, so nothing is summed
// across blocks: no workspace, no arrival counters, no fixup at the end of a launch. The block's producer warp streams
// its tiles through a ring of shared-memory slots with 2D TMA. A slot is one box of a tile, 16 rows x M*1,024 weights
// (M*272 bytes of each row: the whole row when it fits, else the row in NKB boxes), and with a gate matrix the gate's
// box of the same rows beside it, both under one barrier. An SM's requests walk its part of the matrix in order.
//
// Eight consumer warps split every box along k: warp w takes the box's PQ2_0 blocks [w*M, (w+1)*M), so it always works
// on the same k range of a row, and when a row is one box it loads that range's token fragments once per launch and
// keeps them in registers. At a tile's end the warps' sums meet in shared memory and are added in warp order (the
// result does not depend on timing), and 128 threads write the tile's 16 rows x ncols (+ the bias, or up * silu(gate)).
//
// Tensor cores: mma.m16n8k32 s8, weights in A (16 rows), tokens in B (8 columns; a lane past ncols reads the last
// column, whose products are never written). A PQ2_0 int16 holds 8 weights as 2-bit codes (0 -> -1, 1 -> 0, 2 -> 1,
// 3 -> 2); __byte_perm on it gives the even and the odd four as int8, which go to the fragment's two k halves, and the
// token's 8 int8 are split the same way, so each MMA is exactly one 32-weight chunk's integer dot for 16 rows x 8
// columns. vec_dot_pq2_0_q8_1's d2 * d8 * sumi is regrouped: a block's four chunks sum d8 * sumi, and its d2 scales
// that sum once.
//
// The pointers carry no __restrict__: with PDL a restrict load may compile to ld.global.nc, which the compiler can move
// above the grid dependency wait (upstream #24030), and the tokens and the bias are written by the kernels before this
// one. Only the weights, which no kernel writes, are requested before that wait.
//
// A group launch (mmvq_pq2_mma_group) runs several matrices that read one activation, such as qkv and z, or q, k and v:
// the launch's tiles are the first matrix's, then the second's, and so on, and each tile takes its matrix's map, output
// and rows. The blocks split the whole sequence, so the group streams as one matrix of all the rows.

#define PQ2_MMA_NW        8                                // consumer warps, and one producer warp
#define PQ2_MMA_MAX_COLS  8
#define PQ2_MMA_KB_BYTES  (8 * (int) sizeof(block_pq2_0))  // 1,024 weights of a row: 272 bytes
#define PQ2_MMA_MAX_M     7                                // a box row of at most 7 x 272 bytes
#define PQ2_MMA_SMEM_MAX  (99 * 1024)                      // the shared memory a block may take on sm_120

static_assert(sizeof(block_pq2_0) == 34, "PQ2_0 block layout");
static_assert(PQ2_MMA_KB_BYTES % 16 == 0, "a box row is a whole number of 16-byte TMA units");
static_assert(PQ2_MMA_MAX_M * PQ2_MMA_KB_BYTES / 8 <= 256, "a TMA box is at most 256 8-byte elements wide");

// a group launch's matrices in tile order: matrix g's tiles are [tile_end[g - 1], tile_end[g]), from 0 for g = 0
struct pq2_mma_group {
    CUtensorMap tmap[PQ2_MMA_MAX_GROUP];
    float *     dst[PQ2_MMA_MAX_GROUP];
    int         nrows[PQ2_MMA_MAX_GROUP];
    int         stride_col_dst[PQ2_MMA_MAX_GROUP];
    int         tile_end[PQ2_MMA_MAX_GROUP];
    int         n;
};

static __device__ __forceinline__ int pq2_mma_group_of(const pq2_mma_group * grp, const int tile) {
    int g = 0;
    while (g + 1 < grp->n && tile >= grp->tile_end[g]) {
        ++g;
    }
    return g;
}

static __device__ __forceinline__ int pq2_mma_group_begin(const pq2_mma_group * grp, const int g) {
    return g > 0 ? grp->tile_end[g - 1] : 0;
}

// one matrix's box: 16 rows x m*272 bytes, a multiple of 128 bytes (16*272 = 34*128), so every box starts aligned
static constexpr __host__ __device__ int pq2_mma_box_bytes(int m) {
    return 16 * m * PQ2_MMA_KB_BYTES;
}

// a tile's partial sums, per matrix and consumer warp: 32 lanes x 4 floats
static constexpr __host__ __device__ int pq2_mma_red_bytes(int nmat) {
    return nmat * PQ2_MMA_NW * 32 * 4 * (int) sizeof(float);
}

static constexpr size_t pq2_mma_smem_bytes(int m, int nmat, int nslots) {
    return (size_t) nslots * nmat * pq2_mma_box_bytes(m) + pq2_mma_red_bytes(nmat) + 2 * (size_t) nslots * sizeof(uint64_t);
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

static __device__ __forceinline__ void pq2_mbar_arrive(uint64_t * bar) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(pq2_smem_u32(bar)) : "memory");
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

// The box lands in this block's own shared memory: the kernel is never launched as a cluster. Addressed as
// .shared::cluster, the load compiles on sm_120 to a branch whose other side is a driver syscall for a peer block's
// address (__cuda_syscall_cp_async_bulk_tensor_2d_tile_unicast), and the driver gives every resident thread of a kernel
// that can make a syscall the syscall's stack: 14,512 bytes, which on a 70-SM RTX 5070 Ti took 1,430 MiB of device
// memory from the first launch on. .shared::cta is the TMA load alone. It needs PTX ISA 8.6, which every toolkit that
// targets sm_120 (12.8 and later) emits; an older one builds the cluster form for the Hopper code it can target.
#if __CUDACC_VER_MAJOR__ > 12 || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 8)
#define PQ2_TMA_DST "shared::cta"
#else
#define PQ2_TMA_DST "shared::cluster"
#endif

static __device__ __forceinline__ void pq2_tma_load_2d(void * dst, const CUtensorMap * tmap, int c0, int c1, uint64_t * bar,
                                                       uint64_t policy) {
    asm volatile(
        "cp.async.bulk.tensor.2d." PQ2_TMA_DST ".global.tile.mbarrier::complete_tx::bytes.L2::cache_hint"
        " [%0], [%1, {%2, %3}], [%4], %5;"
        :: "r"(pq2_smem_u32(dst)), "l"((uint64_t) tmap), "r"(c0), "r"(c1), "r"(pq2_smem_u32(bar)), "l"(policy)
        : "memory");
}

// a barrier of the consumer warps alone (the producer warp takes no part; barrier 0 is __syncthreads')
static __device__ __forceinline__ void pq2_consumers_sync(int id) {
    asm volatile("bar.sync %0, %1;" :: "r"(id), "r"(PQ2_MMA_NW * 32) : "memory");
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
#endif // PQ2_MMA_AVAILABLE

// nmat 1: dst = W x (+ x_bias); nmat 2: dst = (W x) * silu(G x), W's map in tmap and G's in tmap_gate. grouped: each
// tile's map, output, rows and output stride come from its matrix in grp, not from tmap, dst, nrows and stride_col_dst.
// M: the box's width in 1,024-weight units, and each consumer warp's share of it in PQ2_0 blocks; nb: PQ2_0 blocks per
// row; nkb: boxes per row (the last one may reach past the row: TMA fills that part with zeros, and it is skipped).
template <int M, int nmat, bool has_bias, bool grouped>
static __device__ __forceinline__ void mmvq_pq2_mma_body(
        const CUtensorMap * tmap, const CUtensorMap * tmap_gate, const pq2_mma_group * grp, const block_q8_1 * y,
        const float * x_bias, float * dst, const int nrows, const int ncols, const int nb, const int n_tiles,
        const int nkb, const int nslots, const int evict_first, const int stride_col_y, const int stride_col_dst) {
    static_assert(nmat == 1 || (nmat == 2 && !has_bias), "a gated product fuses no bias");
    static_assert(!grouped || (nmat == 1 && !has_bias), "a group's matrices fuse nothing");
#ifdef PQ2_MMA_AVAILABLE
    constexpr int box_bytes = pq2_mma_box_bytes(M);
    constexpr int row_bytes = M * PQ2_MMA_KB_BYTES; // a box row
    constexpr int pool      = 0x020100FF;           // byte i of a code's selector: 0 -> 0xFF (-1), 1 -> 0, 2 -> 1, 3 -> 2

    extern __shared__ __align__(128) char smem[];
    char     * ring  = smem;
    float    * red   = (float *) (smem + (size_t) nslots * nmat * box_bytes);
    uint64_t * full  = (uint64_t *) ((char *) red + pq2_mma_red_bytes(nmat));
    uint64_t * empty = full + nslots;

    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;

    // this block's tiles
    const int t_begin = (int) ((int64_t) blockIdx.x       * n_tiles / gridDim.x);
    const int t_end   = (int) ((int64_t) (blockIdx.x + 1) * n_tiles / gridDim.x);
    const int n_boxes = (t_end - t_begin) * nkb;

    ggml_cuda_pdl_lc();

    if (threadIdx.x == 0) {
        for (int s = 0; s < nslots; ++s) {
            pq2_mbar_init(&full[s],  1);
            pq2_mbar_init(&empty[s], PQ2_MMA_NW);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();

    if (warp == PQ2_MMA_NW) {
        // the producer: box i of the block's sequence into slot i % nslots, once the consumers have released it
        if (lane == 0) {
            uint64_t policy;
            if (evict_first) {
                asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(policy));
            } else {
                asm volatile("createpolicy.fractional.L2::evict_normal.b64 %0, 1.0;" : "=l"(policy));
            }
            if constexpr (grouped) {
                for (int g = 0; g < grp->n; ++g) {
                    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t) &grp->tmap[g]) : "memory");
                }
            } else {
                asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t) tmap) : "memory");
                if constexpr (nmat == 2) {
                    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t) tmap_gate) : "memory");
                }
            }
            for (int i = 0; i < n_boxes; ++i) {
                const int s = i % nslots;
                if (i >= nslots) {
                    pq2_mbar_wait(&empty[s], (uint32_t) ((i / nslots - 1) & 1));
                }
                const int tile = t_begin + i / nkb;
                const CUtensorMap * map = tmap;
                int row0 = tile * 16; // the tile's first row in its matrix
                if constexpr (grouped) {
                    const int g = pq2_mma_group_of(grp, tile);
                    map  = &grp->tmap[g];
                    row0 = (tile - pq2_mma_group_begin(grp, g)) * 16;
                }
                const int c0 = (i % nkb) * (row_bytes / 8);  // 8-byte elements
                char * slot = ring + (size_t) s * nmat * box_bytes;
                pq2_mbar_arrive_expect_tx(&full[s], nmat * box_bytes);
                pq2_tma_load_2d(slot, map, c0, row0, &full[s], policy);
                if constexpr (nmat == 2) {
                    pq2_tma_load_2d(slot + box_bytes, tmap_gate, c0, row0, &full[s], policy);
                }
            }
        }
        return;
    }

    const int g = lane >> 2;
    const int t = lane & 3;

    // the columns this lane reads: its B column g and its C columns 2t, 2t+1, past ncols the last one
    const int cg  = min(g,       ncols - 1);
    const int cc0 = min(2*t,     ncols - 1);
    const int cc1 = min(2*t + 1, ncols - 1);

    ggml_cuda_pdl_sync(); // the tokens and the bias are the previous kernels' results, and dst may still be read

    const block_q8_1 * y_g  = y + (int64_t) cg  * stride_col_y;
    const block_q8_1 * y_c0 = y + (int64_t) cc0 * stride_col_y;
    const block_q8_1 * y_c1 = y + (int64_t) cc1 * stride_col_y;

    // the token fragments of this warp's M blocks of box kb (4 chunks each), a chunk past the row clamped to its last
    int   bf[M][4][2];
    float d8[M][4][2];
    const auto load_b = [&](const int kb) {
#pragma unroll
        for (int j = 0; j < M; ++j) {
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const int kq = min(((kb*8 + warp)*M + j)*4 + q, nb*4 - 1);
                const int * qs = (const int *) y_g[kq].qs;
                const int   u  = qs[2*t + 0]; // tokens 8t..8t+3
                const int   v  = qs[2*t + 1]; // tokens 8t+4..8t+7
                bf[j][q][0] = __byte_perm(u, v, 0x6420); // the even ones, against the codes' even weights
                bf[j][q][1] = __byte_perm(u, v, 0x7531); // the odd ones
                d8[j][q][0] = __low2float(y_c0[kq].ds);
                d8[j][q][1] = __low2float(y_c1[kq].ds);
            }
        }
    };
    if (nkb == 1) {
        load_b(0); // the warp's k range is the same in every box
    }

    // a reducing thread's output (threads 0-127): lane j/4's C value j%4
    const int j_out = threadIdx.x;
    const int l_out = j_out >> 2;
    const int r_out = j_out & 3;

    int i = 0; // the block's box sequence
    for (int tile = t_begin; tile < t_end; ++tile) {
        // the tile's matrix: its output, rows and output stride, and the tile's first row in it
        float * dst_t    = dst;
        int     nrows_t  = nrows;
        int     stride_t = stride_col_dst;
        int     row0     = tile*16;
        if constexpr (grouped) {
            const int g = pq2_mma_group_of(grp, tile);
            dst_t    = grp->dst[g];
            nrows_t  = grp->nrows[g];
            stride_t = grp->stride_col_dst[g];
            row0     = (tile - pq2_mma_group_begin(grp, g))*16;
        }
        const int row_out = row0 + (l_out >> 2) + (r_out >= 2 ? 8 : 0);
        const int col_out = 2*(l_out & 3) + (r_out & 1);
        const bool writes = j_out < 128 && row_out < nrows_t && col_out < ncols;
        float bias = 0.0f;
        if constexpr (has_bias) {
            if (writes) {
                bias = x_bias[(int64_t) col_out*stride_t + row_out];
            }
        }

        float acc[nmat][4] = {{0.0f}};
        for (int kb = 0; kb < nkb; ++kb, ++i) {
            if (nkb > 1) {
                load_b(kb); // before the wait, under the TMA's latency
            }
            const int s = i % nslots;
            pq2_mbar_wait(&full[s], (uint32_t) ((i / nslots) & 1));
            const char * sw    = ring + (size_t) s * nmat * box_bytes;
            const int    valid = nb - (kb*8 + warp)*M; // this warp's blocks inside the row

#pragma unroll
            for (int j = 0; j < M; ++j) {
                if (j >= valid) {
                    break;
                }
                const int b = warp*M + j; // the block within the box row
                float blk[nmat][4] = {{0.0f}}; // the block's sum over its 4 chunks of d8 * dot; d2 scales it once
#pragma unroll
                for (int q = 0; q < 4; ++q) {
#pragma unroll
                    for (int m = 0; m < nmat; ++m) {
                        const char * sm  = sw + m*box_bytes;
                        const int    off = b*34 + 2 + 2*(4*q + t); // int16 4q+t of the block: weights 32q+8t..+7
                        const int    qlo = *(const uint16_t *) (sm + g      *row_bytes + off);
                        const int    qhi = *(const uint16_t *) (sm + (g + 8)*row_bytes + off);

                        int c0, c1, c2, c3;
                        pq2_mma_s8(c0, c1, c2, c3,
                            __byte_perm(pool, pool, qlo), __byte_perm(pool, pool, qhi),
                            __byte_perm(pool, pool, qlo >> 2), __byte_perm(pool, pool, qhi >> 2), bf[j][q][0], bf[j][q][1]);

                        blk[m][0] += d8[j][q][0] * pq2_mma_dot(c0);
                        blk[m][1] += d8[j][q][1] * pq2_mma_dot(c1);
                        blk[m][2] += d8[j][q][0] * pq2_mma_dot(c2);
                        blk[m][3] += d8[j][q][1] * pq2_mma_dot(c3);
                    }
                }
#pragma unroll
                for (int m = 0; m < nmat; ++m) {
                    const char * sm    = sw + m*box_bytes;
                    const float  d2_lo = __half2float(*(const half *) (sm + g      *row_bytes + b*34));
                    const float  d2_hi = __half2float(*(const half *) (sm + (g + 8)*row_bytes + b*34));
                    acc[m][0] += d2_lo * blk[m][0];
                    acc[m][1] += d2_lo * blk[m][1];
                    acc[m][2] += d2_hi * blk[m][2];
                    acc[m][3] += d2_hi * blk[m][3];
                }
            }

            __syncwarp(); // every lane has read this slot: release it
            if (lane == 0) {
                pq2_mbar_arrive(&empty[s]);
            }
        }

        // the tile's sum over the consumer warps, in warp order; the reducers read the previous tile's sums before the
        // barrier at the next tile, so one buffer serves every tile
        if (tile > t_begin) {
            pq2_consumers_sync(2);
        }
#pragma unroll
        for (int m = 0; m < nmat; ++m) {
            ((float4 *) red)[(m*PQ2_MMA_NW + warp)*32 + lane] = make_float4(acc[m][0], acc[m][1], acc[m][2], acc[m][3]);
        }
        pq2_consumers_sync(1);
        if (j_out < 128) {
            float v[nmat];
#pragma unroll
            for (int m = 0; m < nmat; ++m) {
                v[m] = 0.0f;
#pragma unroll
                for (int w = 0; w < PQ2_MMA_NW; ++w) {
                    v[m] += red[(m*PQ2_MMA_NW + w)*128 + j_out];
                }
            }
            if (writes) {
                float out = v[0];
                if constexpr (nmat == 2) {
                    out *= ggml_cuda_op_silu_single(v[1]);
                }
                if constexpr (has_bias) {
                    out += bias;
                }
                dst_t[(int64_t) col_out*stride_t + row_out] = out;
            }
        }
    }
#else
    GGML_UNUSED_VARS(tmap, tmap_gate, grp, y, x_bias, dst, nrows, ncols, nb, n_tiles, nkb, nslots, evict_first,
        stride_col_y, stride_col_dst);
    NO_DEVICE_CODE;
#endif // PQ2_MMA_AVAILABLE
}

template <int M, int nmat, bool has_bias>
__launch_bounds__((PQ2_MMA_NW + 1)*32, 1)
static __global__ void mmvq_pq2_mma(
        const __grid_constant__ CUtensorMap tmap, const __grid_constant__ CUtensorMap tmap_gate, const block_q8_1 * y,
        const float * x_bias, float * dst, const int nrows, const int ncols, const int nb, const int n_tiles,
        const int nkb, const int nslots, const int evict_first, const int stride_col_y, const int stride_col_dst) {
    mmvq_pq2_mma_body<M, nmat, has_bias, false>(&tmap, &tmap_gate, nullptr, y, x_bias, dst, nrows, ncols, nb, n_tiles,
        nkb, nslots, evict_first, stride_col_y, stride_col_dst);
}

template <int M>
__launch_bounds__((PQ2_MMA_NW + 1)*32, 1)
static __global__ void mmvq_pq2_mma_group(
        const __grid_constant__ pq2_mma_group grp, const block_q8_1 * y, const int ncols, const int nb, const int n_tiles,
        const int nkb, const int nslots, const int evict_first, const int stride_col_y) {
    mmvq_pq2_mma_body<M, 1, false, true>(nullptr, nullptr, &grp, y, nullptr, nullptr, 0, ncols, nb, n_tiles, nkb, nslots,
        evict_first, stride_col_y, 0);
}

// ---------------------------------------------------------------------------------------------------------------------
// host

// GGML_CUDA_PQ2_MMA_CFG="nslots,evict_first" for a sweep: at most nslots slots in the ring (it takes what fits in the
// shared memory, at least 2), and the weights' L2 policy (1 evict_first, 0 evict_normal). Two slots: four measured
// the same at one column and 3.8 % slower at eight on an RTX 5080 (13.94 against 13.43 ms per step).
struct pq2_mma_config {
    int  nslots      = 2;
    bool evict_first = true;
};

static const pq2_mma_config & pq2_mma_get_config() {
    static const pq2_mma_config c = [] {
        pq2_mma_config c;
        const char * s = getenv("GGML_CUDA_PQ2_MMA_CFG");
        if (s != nullptr) {
            int v[2] = { c.nslots, c.evict_first };
            if (sscanf(s, "%d,%d", &v[0], &v[1]) == 2 && v[0] >= 2) {
                c = { v[0], v[1] != 0 };
            } else {
                GGML_LOG_WARN("%s: GGML_CUDA_PQ2_MMA_CFG=%s is not nslots (>= 2),evict_first\n", __func__, s);
            }
        }
        return c;
    }();
    return c;
}

static bool pq2_mma_legacy() {
    static const bool legacy = [] {
        const char * s = getenv("GGML_CUDA_PQ2_MMA_LEGACY");
        return s != nullptr && atoi(s) != 0;
    }();
    return legacy;
}

// a row's boxes: m (1,024-weight units per box, at most 7) and nkb (boxes per row), the fewest boxes whose slots fit twice
// in the shared memory; nslots, as many as fit up to the configured count
struct pq2_mma_plan {
    int m      = 0;
    int nkb    = 0;
    int nslots = 0;
};

static pq2_mma_plan pq2_mma_make_plan(int64_t ncols_x, int nmat) {
    const int nk = (int) (ncols_x / 1024);
    for (int nkb = 1; nkb <= nk; ++nkb) {
        const int m = (nk + nkb - 1) / nkb;
        if (m > PQ2_MMA_MAX_M) {
            continue;
        }
        const size_t per_slot = (size_t) nmat * pq2_mma_box_bytes(m) + 2 * sizeof(uint64_t);
        const int    fit      = (int) ((PQ2_MMA_SMEM_MAX - pq2_mma_red_bytes(nmat)) / per_slot);
        if (fit >= 2) {
            return { m, nkb, std::min(fit, pq2_mma_get_config().nslots) };
        }
    }
    return {};
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

bool ggml_cuda_mmvq_pq2_mma_usable(int cc, const void * vx, const void * vgate, int64_t ncols_x, int64_t nrows_x,
                                   int64_t stride_row_x, int64_t ncols_dst) {
    return !pq2_mma_legacy() && GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_BLACKWELL && cc < GGML_CUDA_CC_DGX_SPARK &&
        ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_BLACKWELL &&
        ncols_dst >= 1 && ncols_dst <= PQ2_MMA_MAX_COLS &&
        ncols_x % 1024 == 0 && stride_row_x*(int64_t) sizeof(block_pq2_0) % 16 == 0 && (uintptr_t) vx % 16 == 0 &&
        (uintptr_t) vgate % 16 == 0 &&
        nrows_x < (int64_t) 1 << 31 && ncols_x*(int64_t) sizeof(block_pq2_0)/QK_PQ2_0/8 < ((int64_t) 1 << 32) &&
        pq2_mma_make_plan(ncols_x, vgate != nullptr ? 2 : 1).m > 0;
}

// the weights as 8-byte elements, [row_bytes/8, nrows] with the row stride; one box = 16 rows x m*272 bytes
static CUtensorMap pq2_mma_tensor_map(const void * vx, int64_t row_bytes, int64_t nrows_x, int64_t stride_row_x, int m) {
    CUtensorMap tmap;
    const cuuint64_t dims[2]     = { (cuuint64_t) (row_bytes / 8), (cuuint64_t) nrows_x };
    const cuuint64_t strides[1]  = { (cuuint64_t) (stride_row_x * (int64_t) sizeof(block_pq2_0)) };
    const cuuint32_t box[2]      = { (cuuint32_t) (m * PQ2_MMA_KB_BYTES / 8), 16 };
    const cuuint32_t elem_str[2] = { 1, 1 };
    const CUresult res = pq2_mma_encode_fn()(&tmap, CU_TENSOR_MAP_DATA_TYPE_UINT64, 2, const_cast<void *>(vx), dims,
        strides, box, elem_str, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    GGML_ASSERT(res == CUDA_SUCCESS && "cuTensorMapEncodeTiled for the PQ2_0 weights");
    return tmap;
}

template <int M, int nmat, bool has_bias>
static void pq2_mma_launch(const pq2_mma_plan & p, const CUtensorMap & tmap, const CUtensorMap & tmap_gate,
        const block_q8_1 * y, const float * x_bias, float * dst, int nrows, int ncols, int nb, int n_tiles,
        int stride_col_y, int stride_col_dst, cudaStream_t stream) {
    static_assert(pq2_mma_smem_bytes(M, nmat, 2) <= PQ2_MMA_SMEM_MAX || nmat == 2, "two slots of a plain box fit");
    const int nsm     = ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;
    const int nblocks = std::min(nsm, n_tiles);
    const ggml_cuda_kernel_launch_params params(dim3(nblocks), dim3((PQ2_MMA_NW + 1)*32),
        pq2_mma_smem_bytes(M, nmat, p.nslots), stream);
    CUDA_SET_SHARED_MEMORY_LIMIT((mmvq_pq2_mma<M, nmat, has_bias>), PQ2_MMA_SMEM_MAX); // every plan's size, once
    ggml_cuda_kernel_launch(mmvq_pq2_mma<M, nmat, has_bias>, params, tmap, tmap_gate, y, x_bias, dst, nrows, ncols, nb,
        n_tiles, p.nkb, p.nslots, (int) pq2_mma_get_config().evict_first, stride_col_y, stride_col_dst);
}

template <int nmat, bool has_bias>
static void pq2_mma_launch_m(const pq2_mma_plan & p, const CUtensorMap & tmap, const CUtensorMap & tmap_gate,
        const block_q8_1 * y, const float * x_bias, float * dst, int nrows, int ncols, int nb, int n_tiles,
        int stride_col_y, int stride_col_dst, cudaStream_t stream) {
    switch (p.m) {
#define PQ2_MMA_CASE(M) case M: pq2_mma_launch<M, nmat, has_bias>(p, tmap, tmap_gate, y, x_bias, dst, nrows, ncols, nb, \
                                    n_tiles, stride_col_y, stride_col_dst, stream); break;
        PQ2_MMA_CASE(1)
        PQ2_MMA_CASE(2)
        PQ2_MMA_CASE(3)
        PQ2_MMA_CASE(4)
        PQ2_MMA_CASE(5)
        PQ2_MMA_CASE(6)
        PQ2_MMA_CASE(7)
#undef PQ2_MMA_CASE
        default: GGML_ABORT("%s: no instance for m %d", __func__, p.m);
    }
}

void ggml_cuda_mmvq_pq2_mma(const void * vx, const void * vgate, const void * vy, const float * x_bias, float * dst,
                            int64_t ncols_x, int64_t nrows_x, int64_t ncols_dst, int64_t stride_row_x,
                            int64_t stride_col_y, int64_t stride_col_dst, cudaStream_t stream) {
    const int          nmat = vgate != nullptr ? 2 : 1;
    const pq2_mma_plan p    = pq2_mma_make_plan(ncols_x, nmat);
    GGML_ASSERT(p.m > 0 && "ggml_cuda_mmvq_pq2_mma_usable holds a plan");

    const int64_t row_bytes = ncols_x / QK_PQ2_0 * (int64_t) sizeof(block_pq2_0);
    const int     nb        = (int) (ncols_x / QK_PQ2_0);
    const int     n_tiles   = (int) ((nrows_x + 15) / 16);

    const CUtensorMap tmap      = pq2_mma_tensor_map(vx, row_bytes, nrows_x, stride_row_x, p.m);
    const CUtensorMap tmap_gate = nmat == 2 ? pq2_mma_tensor_map(vgate, row_bytes, nrows_x, stride_row_x, p.m) : tmap;

    const block_q8_1 * y = (const block_q8_1 *) vy;
    if (nmat == 2) {
        GGML_ASSERT(x_bias == nullptr);
        pq2_mma_launch_m<2, false>(p, tmap, tmap_gate, y, nullptr, dst, (int) nrows_x, (int) ncols_dst, nb, n_tiles,
            (int) stride_col_y, (int) stride_col_dst, stream);
    } else if (x_bias != nullptr) {
        pq2_mma_launch_m<1, true>(p, tmap, tmap_gate, y, x_bias, dst, (int) nrows_x, (int) ncols_dst, nb, n_tiles,
            (int) stride_col_y, (int) stride_col_dst, stream);
    } else {
        pq2_mma_launch_m<1, false>(p, tmap, tmap_gate, y, nullptr, dst, (int) nrows_x, (int) ncols_dst, nb, n_tiles,
            (int) stride_col_y, (int) stride_col_dst, stream);
    }
}

template <int M>
static void pq2_mma_launch_group(const pq2_mma_plan & p, const pq2_mma_group & grp, const block_q8_1 * y, int ncols,
        int nb, int n_tiles, int stride_col_y, cudaStream_t stream) {
    const int nsm     = ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;
    const int nblocks = std::min(nsm, n_tiles);
    const ggml_cuda_kernel_launch_params params(dim3(nblocks), dim3((PQ2_MMA_NW + 1)*32),
        pq2_mma_smem_bytes(M, 1, p.nslots), stream);
    CUDA_SET_SHARED_MEMORY_LIMIT((mmvq_pq2_mma_group<M>), PQ2_MMA_SMEM_MAX);
    ggml_cuda_kernel_launch(mmvq_pq2_mma_group<M>, params, grp, y, ncols, nb, n_tiles, p.nkb, p.nslots,
        (int) pq2_mma_get_config().evict_first, stride_col_y);
}

void ggml_cuda_mmvq_pq2_mma_group(int n, const void * const * vx, float * const * dst, const int64_t * nrows_x,
                                  const int64_t * stride_row_x, const int64_t * stride_col_dst, const void * vy,
                                  int64_t ncols_x, int64_t ncols_dst, int64_t stride_col_y, cudaStream_t stream) {
    GGML_ASSERT(n >= 2 && n <= PQ2_MMA_MAX_GROUP);
    const pq2_mma_plan p = pq2_mma_make_plan(ncols_x, 1);
    GGML_ASSERT(p.m > 0 && "ggml_cuda_mmvq_pq2_mma_usable holds a plan");

    const int64_t row_bytes = ncols_x / QK_PQ2_0 * (int64_t) sizeof(block_pq2_0);
    const int     nb        = (int) (ncols_x / QK_PQ2_0);

    pq2_mma_group grp{};
    int64_t n_tiles = 0;
    for (int g = 0; g < n; ++g) {
        grp.tmap[g]           = pq2_mma_tensor_map(vx[g], row_bytes, nrows_x[g], stride_row_x[g], p.m);
        grp.dst[g]            = dst[g];
        grp.nrows[g]          = (int) nrows_x[g];
        grp.stride_col_dst[g] = (int) stride_col_dst[g];
        n_tiles              += (nrows_x[g] + 15) / 16;
        grp.tile_end[g]       = (int) n_tiles;
    }
    grp.n = n;
    GGML_ASSERT(n_tiles < ((int64_t) 1 << 31) / 16);

    const block_q8_1 * y = (const block_q8_1 *) vy;
    switch (p.m) {
#define PQ2_MMA_CASE(M) case M: pq2_mma_launch_group<M>(p, grp, y, (int) ncols_dst, nb, (int) n_tiles, (int) stride_col_y, \
                                    stream); break;
        PQ2_MMA_CASE(1)
        PQ2_MMA_CASE(2)
        PQ2_MMA_CASE(3)
        PQ2_MMA_CASE(4)
        PQ2_MMA_CASE(5)
        PQ2_MMA_CASE(6)
        PQ2_MMA_CASE(7)
#undef PQ2_MMA_CASE
        default: GGML_ABORT("%s: no instance for m %d", __func__, p.m);
    }
}
