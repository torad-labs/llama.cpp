#include "mmq-tma-tiles.cuh"

#include "mma.cuh"
#include "mmq-few.cuh"
#include "quantize.cuh"
#include "unary.cuh"
#include "vecdotq.cuh"

#if defined(CUDART_VERSION) && CUDART_VERSION >= 12000 && !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

#include <cuda.h>
#include <cudaTypedefs.h>

#include <algorithm>
#include <atomic>
#include <map>
#include <mutex>
#include <tuple>
#include <type_traits>
#include <utility>

using namespace ggml_cuda_mma;

// A Q3_K row of 4,608 values is 18 blocks of 110 bytes = 1,980 bytes. Four adjacent rows are 7,920 = 9 * 880 bytes,
// so the weight buffer viewed as a rank-2 map of UINT64 chunks (110 elements = 880 bytes a chunk, 16-byte stride)
// has whole 16-row tiles as boxes of 36 chunks, and shared memory receives the tile's rows in order.
static constexpr int MMQ_TMA_ROW_BYTES      = 1980;
static constexpr int MMQ_TMA_UNIT_BYTES     = 110; // one 256-value unit of a Q3_K row
static constexpr int MMQ_TMA_CHUNK          = 880;
static constexpr int MMQ_TMA_TILE_ROWS      = 16;
static constexpr int MMQ_TMA_BOX            = MMQ_TMA_TILE_ROWS*MMQ_TMA_ROW_BYTES/MMQ_TMA_CHUNK; // 36
static constexpr int MMQ_TMA_TILE_BYTES     = MMQ_TMA_BOX*MMQ_TMA_CHUNK;                         // 31,680
static constexpr int MMQ_TMA_TILE_STRIDE    = ((MMQ_TMA_TILE_BYTES + 127)/128)*128;              // 31,744 (128B TMA align)
static constexpr int MMQ_TMA_UNITS          = 18; // 256-value units of the Q3_K row
static constexpr int MMQ_TMA_Q3K_NWARPS     = 9;  // 18 / 9 = 2 units per warp (balanced, >2 warps/SMSP)
static constexpr int MMQ_TMA_Q4K_UNIT_BYTES = 144;
static constexpr int MMQ_TMA_Q5K_UNIT_BYTES = 176;
static constexpr int MMQ_TMA_Q45_MAX_BOX_U64 = 256; // hardware TMA rank-2 inner box limit in UINT64 elements
static constexpr int MMQ_TMA_MAX_TOKENS     = 32;

static_assert(MMQ_TMA_CHUNK % 16 == 0, "a chunk starts on a TMA stride boundary");
static_assert(MMQ_TMA_CHUNK/8 <= 256, "a TMA box is at most 256 8-byte elements wide");
static_assert(MMQ_TMA_BOX <= 256, "a TMA box is at most 256 elements in every dimension");
static_assert(MMQ_TMA_TILE_ROWS % 4 == 0, "whole 4-row chunks");
static_assert(MMQ_TMA_TILE_STRIDE % 128 == 0, "second TMA tile starts 128-byte aligned");
static_assert(MMQ_TMA_Q4K_UNIT_BYTES % 16 == 0, "Q4_K unit is 16-byte aligned");
static_assert(MMQ_TMA_Q5K_UNIT_BYTES % 16 == 0, "Q5_K unit is 16-byte aligned");

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == GGML_CUDA_CC_BLACKWELL
static __device__ __forceinline__ uint32_t mmq_tma_smem(const void * p) {
    return (uint32_t) __cvta_generic_to_shared(p);
}

static __device__ __forceinline__ void mmq_tma_mbar_init(uint64_t * bar) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(mmq_tma_smem(bar)) : "memory");
}

static __device__ __forceinline__ void mmq_tma_mbar_expect(uint64_t * bar, uint32_t bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
        :: "r"(mmq_tma_smem(bar)), "r"(bytes) : "memory");
}

static __device__ __forceinline__ void mmq_tma_mbar_wait(uint64_t * bar, uint32_t parity) {
    const uint32_t addr = mmq_tma_smem(bar);
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

static __device__ __forceinline__ void mmq_tma_load(void * dst, const CUtensorMap * map, const int x, const int y,
                                                    uint64_t * bar, const uint64_t policy) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint "
        "[%0], [%1, {%2, %3}], [%4], %5;"
        :: "r"(mmq_tma_smem(dst)), "l"((uint64_t) map), "r"(x), "r"(y), "r"(mmq_tma_smem(bar)), "l"(policy)
        : "memory");
}

// Weight stream is read once a step: evict_first, matching the fork's other TMA weight loads.
static __device__ __forceinline__ uint64_t mmq_tma_evict_first() {
    uint64_t policy;
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(policy));
    return policy;
}
#endif // __CUDA_ARCH__ == BLACKWELL

// The src1 fragments of a lane for one unit of 256 values, as quantize.cuh lays them out (ggml_cuda_mul_mat_q_few).
struct mmq_tma_y_frag {
    int4 qs[4];
    int4 ds[4];
};

static __device__ __forceinline__ void mmq_tma_load_y(mmq_tma_y_frag & yf, const int4 * y) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        yf.qs[j] = y[j*WARP_SIZE + threadIdx.x];
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        yf.ds[j] = y[4*WARP_SIZE + 4*(threadIdx.x % 4) + j];
    }
}

static __device__ __forceinline__ int2 mmq_tma_get_b(const mmq_tma_y_frag & yf, const int s) {
    const int4 q = yf.qs[s/2];
    return s % 2 == 0 ? make_int2(q.x, q.y) : make_int2(q.z, q.w);
}

// Q3_K quantizes its activations with one scale a sub-block (MMQ_Q8_1_DS_LAYOUT_D4).
static __device__ __forceinline__ float2 mmq_tma_get_d(const mmq_tma_y_frag & yf, const int s) {
    const float * d = (const float *) &yf.ds[s/2];
    return make_float2(d[2*(s % 2) + 0], d[2*(s % 2) + 1]);
}

// Q4_K and Q5_K use MMQ_Q8_1_DS_LAYOUT_DS4 (half2 {d, sum} per sub-block).
static __device__ __forceinline__ float2 mmq_tma_get_d_ds4(const mmq_tma_y_frag & yf, const int s) {
    const half2 * ds = (const half2 *) &yf.ds[s/2];
    return make_float2(__low2float(ds[2*(s % 2) + 0]), __low2float(ds[2*(s % 2) + 1]));
}

static __device__ __forceinline__ float2 mmq_tma_get_sum_ds4(const mmq_tma_y_frag & yf, const int s) {
    const half2 * ds = (const half2 *) &yf.ds[s/2];
    return make_float2(__high2float(ds[2*(s % 2) + 0]), __high2float(ds[2*(s % 2) + 1]));
}

static __device__ __forceinline__ tile<16, 8, int> mmq_tma_mma(const int (&lo)[2], const int (&hi)[2], const int2 b) {
    tile<16, 8, int> A;
    A.x[0] = lo[0];
    A.x[1] = lo[1];
    A.x[2] = hi[0];
    A.x[3] = hi[1];
    tile<8, 8, int> B;
    B.x[0] = b.x;
    B.x[1] = b.y;
    tile<16, 8, int> C;
    mma(C, A, B);
    return C;
}

// C.x[0], C.x[1]: row g, columns 2t and 2t+1. C.x[2], C.x[3]: row g+8.
static __device__ __forceinline__ tile<16, 8, int> mmq_tma_mma_k16(const int (&a)[2], const int b) {
    tile<16, 4, int> A;
    A.x[0] = a[0];
    A.x[1] = a[1];
    tile<8, 4, int> B;
    B.x[0] = b;
    tile<16, 8, int> C;
    mma(C, A, B);
    return C;
}

// One k=16 MMA a half of sub-block s, the int8 scales of the halves are bytes 0 and 1 of sc[r].
static __device__ __forceinline__ void mmq_tma_mma_k16_add(
        const int (&lo)[2], const int (&hi)[2], const int (&sc)[2], const mmq_tma_y_frag & yf, const int s, float (&sum_d)[4]) {
    const int sc_lo[2] = {(int8_t) sc[0], (int8_t) sc[1]};
    const int sc_hi[2] = {(int8_t) (sc[0] >> 8), (int8_t) (sc[1] >> 8)};

    const int2 b = mmq_tma_get_b(yf, s);
    const tile<16, 8, int> C_lo = mmq_tma_mma_k16(lo, b.x);
    const tile<16, 8, int> C_hi = mmq_tma_mma_k16(hi, b.y);

    const float2 d = mmq_tma_get_d(yf, s);
    sum_d[0] += d.x * float(C_lo.x[0]*sc_lo[0] + C_hi.x[0]*sc_hi[0]);
    sum_d[1] += d.y * float(C_lo.x[1]*sc_lo[0] + C_hi.x[1]*sc_hi[0]);
    sum_d[2] += d.x * float(C_lo.x[2]*sc_lo[1] + C_hi.x[2]*sc_hi[1]);
    sum_d[3] += d.y * float(C_lo.x[3]*sc_lo[1] + C_hi.x[3]*sc_hi[1]);
}

static __device__ __forceinline__ void mmq_tma_add_scaled(float (&sum)[4], const float (&d)[2], const float (&sum_d)[4]) {
#pragma unroll
    for (int l = 0; l < 4; ++l) {
        sum[l] += d[l/2]*sum_d[l];
    }
}

// The 6-bit scales 4*ksc .. 4*ksc + 3 of a q3_K block, minus 32, as int8 (ggml_cuda_mmq_load_tiles_q3_K's unpack).
static __device__ __forceinline__ int mmq_tma_unpack_scales_q3_K(const uint8_t * scales, const int ksc) {
    const int sc_low  = (get_int_b2(scales, ksc % 2) >> (4 * (ksc / 2))) & 0x0F0F0F0F;
    const int sc_high = ((get_int_b2(scales, 2) >> (2 * ksc)) << 4) & 0x30303030;
    return __vsub4(sc_low | sc_high, 0x20202020);
}

// One unit of 256 values of the 16 rows of a warp: x[0] and x[1] are the rows g and g+8 of lane (g, t).
static __device__ __forceinline__ void mmq_tma_vec_dot_q3_K(const char * const (&x)[2], const mmq_tma_y_frag & yf, float (&sum)[4]) {
    const int t = threadIdx.x % 4;

    int   hm[2][2];
    int   sc[2][4]; // the 16 scales of a row as int8, minus 32
    float d[2];
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        const block_q3_K * bx = (const block_q3_K *) x[r];
        hm[r][0] = ~get_int_b2(bx->hmask,      t); // a set bit means that 4 is subtracted
        hm[r][1] = ~get_int_b2(bx->hmask + 16, t);
#pragma unroll
        for (int ksc = 0; ksc < 4; ++ksc) {
            sc[r][ksc] = mmq_tma_unpack_scales_q3_K(bx->scales, ksc);
        }
        d[r] = __half2float(bx->d);
    }

    float sum_d[4] = {0.0f};
#pragma unroll
    for (int n = 0; n < 2; ++n) {
        int qs[2][2];
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            const block_q3_K * bx = (const block_q3_K *) x[r];
            qs[r][0] = get_int_b2(bx->qs + 32*n,      t);
            qs[r][1] = get_int_b2(bx->qs + 32*n + 16, t);
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int s = 4*n + j;
            int lo[2];
            int hi[2];
#pragma unroll
            for (int r = 0; r < 2; ++r) {
                lo[r] = __vsub4((qs[r][0] >> (2*j)) & 0x03030303, ((hm[r][0] >> s) & 0x01010101) << 2);
                hi[r] = __vsub4((qs[r][1] >> (2*j)) & 0x03030303, ((hm[r][1] >> s) & 0x01010101) << 2);
            }
            const int sc_s[2] = {sc[0][s/2] >> (16*(s % 2)), sc[1][s/2] >> (16*(s % 2))}; // scales 2s and 2s+1
            mmq_tma_mma_k16_add(lo, hi, sc_s, yf, s, sum_d);
        }
    }
    mmq_tma_add_scaled(sum, d, sum_d);
}

// One 256-value unit of 16 Q4_K or Q5_K rows, matching mmq_few_vec_dot_q45_K<type, 1>.
template <ggml_type type>
static __device__ __forceinline__ void mmq_tma_vec_dot_q45_K(const char * const (&x)[2], const mmq_tma_y_frag & yf, float (&sum)[4]) {
    using block_t = std::conditional_t<type == GGML_TYPE_Q4_K, block_q4_K, block_q5_K>;
    const int t = threadIdx.x % 4;

    int   qs[2][4][2];
    [[maybe_unused]] int qh[2][2];
    int   sc[2][2];
    int   mn[2][2];
    float d[2];
    float dmin[2];
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        const block_t * bx = (const block_t *) x[r];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            qs[r][j][0] = get_int_b4(bx->qs + 32*j,      t);
            qs[r][j][1] = get_int_b4(bx->qs + 32*j + 16, t);
        }
        if constexpr (type == GGML_TYPE_Q5_K) {
            qh[r][0] = get_int_b4(bx->qh,      t);
            qh[r][1] = get_int_b4(bx->qh + 16, t);
        }
        const int4 hdr = *(const int4 *) bx;
        const int scales[3] = {hdr.y, hdr.z, hdr.w};
        sc[r][0] = unpack_scales_q45_K(scales, 0);
        sc[r][1] = unpack_scales_q45_K(scales, 1);
        mn[r][0] = unpack_scales_q45_K(scales, 2);
        mn[r][1] = unpack_scales_q45_K(scales, 3);
        const float2 dm = __half22float2(bx->dm);
        d[r]    = dm.x;
        dmin[r] = dm.y;
    }

    float sum_d[4] = {0.0f};
    float sum_m[4] = {0.0f};
#pragma unroll
    for (int s = 0; s < 8; ++s) {
        const int j     = s/2;
        const int shift = 4*(s % 2);
        int lo[2];
        int hi[2];
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            lo[r] = (qs[r][j][0] >> shift) & 0x0F0F0F0F;
            hi[r] = (qs[r][j][1] >> shift) & 0x0F0F0F0F;
            if constexpr (type == GGML_TYPE_Q5_K) {
                lo[r] |= ((qh[r][0] >> s) & 0x01010101) << 4;
                hi[r] |= ((qh[r][1] >> s) & 0x01010101) << 4;
            }
        }
        const int   ls[2] = {(sc[0][s/4] >> (8*(s % 4))) & 0xFF, (sc[1][s/4] >> (8*(s % 4))) & 0xFF};
        const float m0    = (mn[0][s/4] >> (8*(s % 4))) & 0xFF;
        const float m1    = (mn[1][s/4] >> (8*(s % 4))) & 0xFF;
        const tile<16, 8, int> C = mmq_tma_mma(lo, hi, mmq_tma_get_b(yf, s));
        const float2 dy = mmq_tma_get_d_ds4(yf, s);
        sum_d[0] += dy.x * float(C.x[0]*ls[0]);
        sum_d[1] += dy.y * float(C.x[1]*ls[0]);
        sum_d[2] += dy.x * float(C.x[2]*ls[1]);
        sum_d[3] += dy.y * float(C.x[3]*ls[1]);
        const float2 sm = mmq_tma_get_sum_ds4(yf, s);
        sum_m[0] += m0 * sm.x;
        sum_m[1] += m0 * sm.y;
        sum_m[2] += m1 * sm.x;
        sum_m[3] += m1 * sm.y;
    }
#pragma unroll
    for (int l = 0; l < 4; ++l) {
        sum[l] += d[l/2]*sum_d[l] - dmin[l/2]*sum_m[l];
    }
}

// Persistent 2-slot double-buffered Q3_K TMA kernel (9 balanced warps x 2 units = 18 units).
static __global__ __launch_bounds__(MMQ_TMA_Q3K_NWARPS*WARP_SIZE, 1)
void mul_mat_q_tma_q3_K(const __grid_constant__ CUtensorMap map, const int4 * y, float * dst,
                        const int nrows_x, const int64_t expert_chunks, const int ncols_dst,
                        const int stride_col_dst, const int64_t stride_channel_dst,
                        const int * pairs, const int * npairs_ptr) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == GGML_CUDA_CC_BLACKWELL
    constexpr int NWARPS = MMQ_TMA_Q3K_NWARPS;
    constexpr int REDUCE_FLOATS = (NWARPS - 1)*4*WARP_SIZE;

    extern __shared__ __align__(128) char s_dyn[];
    char *     s_buf0   = s_dyn;
    char *     s_buf1   = s_dyn + MMQ_TMA_TILE_STRIDE;
    uint64_t * bar      = reinterpret_cast<uint64_t *>(s_dyn + 2*MMQ_TMA_TILE_STRIDE);
    float *    buf_sum0 = reinterpret_cast<float *>(bar + 2);
    float *    buf_sum1 = buf_sum0 + REDUCE_FLOATS;

    const int warp = threadIdx.y;
    const int lane = threadIdx.x;
    const int tid  = warp*WARP_SIZE + lane;

    ggml_cuda_pdl_sync();

    const int npairs      = *npairs_ptr;
    const int n_row_tiles = nrows_x / MMQ_TMA_TILE_ROWS;
    const int total_tiles = npairs * n_row_tiles;
    const int t_begin     = (int) ((int64_t(total_tiles) * blockIdx.x) / gridDim.x);
    const int t_end       = (int) ((int64_t(total_tiles) * (blockIdx.x + 1)) / gridDim.x);
    if (t_begin >= t_end) {
        ggml_cuda_pdl_lc();
        return;
    }

    const uint64_t policy = mmq_tma_evict_first();
    if (tid == 0) {
        mmq_tma_mbar_init(&bar[0]);
        mmq_tma_mbar_init(&bar[1]);
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();

    if (tid == 0) {
        const int p0  = t_begin / n_row_tiles;
        const int rt0 = t_begin % n_row_tiles;
        const int e0  = pairs[p0*MMQ_FEW_PAIR_INTS];
        mmq_tma_mbar_expect(&bar[0], MMQ_TMA_TILE_BYTES);
        const int chunk0 = (int) (int64_t(e0)*expert_chunks + int64_t(rt0)*MMQ_TMA_BOX);
        mmq_tma_load(s_buf0, &map, 0, chunk0, &bar[0], policy);
    }
    __syncthreads();

    uint32_t par0 = 0;
    uint32_t par1 = 0;
    int cur = 0;
    int cached_p = -1;
    mmq_tma_y_frag yf0, yf1;
    for (int t = t_begin; t < t_end; ++t) {
        if (tid == 0 && t + 1 < t_end) {
            const int pn  = (t + 1) / n_row_tiles;
            const int rtn = (t + 1) % n_row_tiles;
            const int en  = pairs[pn*MMQ_FEW_PAIR_INTS];
            const int chunk_n = (int) (int64_t(en)*expert_chunks + int64_t(rtn)*MMQ_TMA_BOX);
            uint64_t * bar_nxt = cur == 0 ? &bar[1] : &bar[0];
            char *     s_nxt   = cur == 0 ? s_buf1 : s_buf0;
            mmq_tma_mbar_expect(bar_nxt, MMQ_TMA_TILE_BYTES);
            mmq_tma_load(s_nxt, &map, 0, chunk_n, bar_nxt, policy);
        }

        const int p    = t / n_row_tiles;
        const int rt   = t % n_row_tiles;
        const int row0 = rt*MMQ_TMA_TILE_ROWS;
        const int * pair = pairs + p*MMQ_FEW_PAIR_INTS;
        if (p != cached_p) {
            const int4 * y_pair = y + int64_t(p)*MMQ_TMA_UNITS*MMQ_FEW_Q8_1_INT4;
            mmq_tma_load_y(yf0, y_pair + warp*MMQ_FEW_Q8_1_INT4);
            mmq_tma_load_y(yf1, y_pair + (warp + NWARPS)*MMQ_FEW_Q8_1_INT4);
            cached_p = p;
        }

        if (cur == 0) {
            mmq_tma_mbar_wait(&bar[0], par0);
            par0 ^= 1;
        } else {
            mmq_tma_mbar_wait(&bar[1], par1);
            par1 ^= 1;
        }

        float sum[4] = {0.0f};
        const char * s_cur = cur == 0 ? s_buf0 : s_buf1;
        {
            const char * const xr0[2] = {
                s_cur + (lane/4 + 0)*MMQ_TMA_ROW_BYTES + warp*MMQ_TMA_UNIT_BYTES,
                s_cur + (lane/4 + 8)*MMQ_TMA_ROW_BYTES + warp*MMQ_TMA_UNIT_BYTES,
            };
            mmq_tma_vec_dot_q3_K(xr0, yf0, sum);
            const char * const xr1[2] = {
                s_cur + (lane/4 + 0)*MMQ_TMA_ROW_BYTES + (warp + NWARPS)*MMQ_TMA_UNIT_BYTES,
                s_cur + (lane/4 + 8)*MMQ_TMA_ROW_BYTES + (warp + NWARPS)*MMQ_TMA_UNIT_BYTES,
            };
            mmq_tma_vec_dot_q3_K(xr1, yf1, sum);
        }

        float * buf_sum = cur == 0 ? buf_sum0 : buf_sum1;
        if (warp > 0) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                buf_sum[((warp - 1)*4 + i)*WARP_SIZE + lane] = sum[i];
            }
        }
        __syncthreads();

        if (warp == 0) {
#pragma unroll
            for (int w = 0; w < NWARPS - 1; ++w) {
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    sum[i] += buf_sum[(w*4 + i)*WARP_SIZE + lane];
                }
            }
            const int ncols = pair[1];
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                const int i = row0 + tile<16, 8, int>::get_i(l);
                const int j = tile<16, 8, int>::get_j(l);
                if (i < nrows_x && j < ncols) {
                    const int e = pair[2 + j];
                    dst[int64_t(e / ncols_dst)*stride_channel_dst + int64_t(e % ncols_dst)*stride_col_dst + i] = sum[l];
                }
            }
        }
        cur ^= 1;
    }

    ggml_cuda_pdl_lc();
#else
    GGML_UNUSED_VARS(map, y, dst, nrows_x, expert_chunks, ncols_dst, stride_col_dst, stride_channel_dst, pairs, npairs_ptr);
    NO_DEVICE_CODE;
#endif // __CUDA_ARCH__ == GGML_CUDA_CC_BLACKWELL
}

// Persistent 2-slot alternating gate/up Q3_K + SITU kernel: overlaps every up TMA load with gate compute
// and every next-tile gate TMA load with up compute + SITU writeback. Bit-identical to mul_mat_q_tma_q3_K + SITU.
static __global__ __launch_bounds__(MMQ_TMA_Q3K_NWARPS*WARP_SIZE, 1)
void mul_mat_q_tma_q3_K_gate_up_situ(
        const __grid_constant__ CUtensorMap map, const int4 * y, float * dst,
        const int n_ff, const int64_t expert_chunks, const int64_t half_chunks,
        const int ncols_dst, const int stride_col_dst, const int64_t stride_channel_dst,
        const int * pairs, const int * npairs_ptr, const float beta, const float lb) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == GGML_CUDA_CC_BLACKWELL
    constexpr int NWARPS = MMQ_TMA_Q3K_NWARPS;
    constexpr int REDUCE_FLOATS = (NWARPS - 1)*4*WARP_SIZE;

    extern __shared__ __align__(128) char s_dyn[];
    char *     s_gate   = s_dyn;
    char *     s_up     = s_dyn + MMQ_TMA_TILE_STRIDE;
    uint64_t * bar      = reinterpret_cast<uint64_t *>(s_dyn + 2*MMQ_TMA_TILE_STRIDE);
    float *    buf_gate = reinterpret_cast<float *>(bar + 2);
    float *    buf_up   = buf_gate + REDUCE_FLOATS;

    const int warp = threadIdx.y;
    const int lane = threadIdx.x;
    const int tid  = warp*WARP_SIZE + lane;

    ggml_cuda_pdl_sync();

    const int npairs      = *npairs_ptr;
    const int n_row_tiles = n_ff / MMQ_TMA_TILE_ROWS;
    const int total_tiles = npairs * n_row_tiles;
    const int t_begin     = (int) ((int64_t(total_tiles) * blockIdx.x) / gridDim.x);
    const int t_end       = (int) ((int64_t(total_tiles) * (blockIdx.x + 1)) / gridDim.x);
    if (t_begin >= t_end) {
        ggml_cuda_pdl_lc();
        return;
    }

    const uint64_t policy = mmq_tma_evict_first();
    if (tid == 0) {
        mmq_tma_mbar_init(&bar[0]);
        mmq_tma_mbar_init(&bar[1]);
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();

    if (tid == 0) {
        const int p0  = t_begin / n_row_tiles;
        const int rt0 = t_begin % n_row_tiles;
        const int e0  = pairs[p0*MMQ_FEW_PAIR_INTS];
        mmq_tma_mbar_expect(&bar[0], MMQ_TMA_TILE_BYTES);
        const int chunk_g0 = (int) (int64_t(e0)*expert_chunks + int64_t(rt0)*MMQ_TMA_BOX);
        mmq_tma_load(s_gate, &map, 0, chunk_g0, &bar[0], policy);
    }
    __syncthreads();

    uint32_t p0 = 0;
    uint32_t p1 = 0;
    int cached_p = -1;
    mmq_tma_y_frag yf0, yf1;
    for (int t = t_begin; t < t_end; ++t) {
        const int p    = t / n_row_tiles;
        const int rt   = t % n_row_tiles;
        const int row0 = rt*MMQ_TMA_TILE_ROWS;
        const int * pair = pairs + p*MMQ_FEW_PAIR_INTS;

        // Issue up(t) into s_up while computing gate(t) from s_gate.
        if (tid == 0) {
            const int e = pair[0];
            mmq_tma_mbar_expect(&bar[1], MMQ_TMA_TILE_BYTES);
            const int chunk_u = (int) (int64_t(e)*expert_chunks + half_chunks + int64_t(rt)*MMQ_TMA_BOX);
            mmq_tma_load(s_up, &map, 0, chunk_u, &bar[1], policy);
        }

        if (p != cached_p) {
            const int4 * y_pair = y + int64_t(p)*MMQ_TMA_UNITS*MMQ_FEW_Q8_1_INT4;
            mmq_tma_load_y(yf0, y_pair + warp*MMQ_FEW_Q8_1_INT4);
            mmq_tma_load_y(yf1, y_pair + (warp + NWARPS)*MMQ_FEW_Q8_1_INT4);
            cached_p = p;
        }

        mmq_tma_mbar_wait(&bar[0], p0);
        p0 ^= 1;

        float sum_g[4] = {0.0f};
        {
            const char * const xg0[2] = {
                s_gate + (lane/4 + 0)*MMQ_TMA_ROW_BYTES + warp*MMQ_TMA_UNIT_BYTES,
                s_gate + (lane/4 + 8)*MMQ_TMA_ROW_BYTES + warp*MMQ_TMA_UNIT_BYTES,
            };
            mmq_tma_vec_dot_q3_K(xg0, yf0, sum_g);
            const char * const xg1[2] = {
                s_gate + (lane/4 + 0)*MMQ_TMA_ROW_BYTES + (warp + NWARPS)*MMQ_TMA_UNIT_BYTES,
                s_gate + (lane/4 + 8)*MMQ_TMA_ROW_BYTES + (warp + NWARPS)*MMQ_TMA_UNIT_BYTES,
            };
            mmq_tma_vec_dot_q3_K(xg1, yf1, sum_g);
        }
        // Release s_gate and ensure prior iter's warp-0 reduction finished reading buf_gate/buf_up.
        __syncthreads();

        if (tid == 0 && t + 1 < t_end) {
            const int pn  = (t + 1) / n_row_tiles;
            const int rtn = (t + 1) % n_row_tiles;
            const int en  = pairs[pn*MMQ_FEW_PAIR_INTS];
            mmq_tma_mbar_expect(&bar[0], MMQ_TMA_TILE_BYTES);
            const int chunk_gn = (int) (int64_t(en)*expert_chunks + int64_t(rtn)*MMQ_TMA_BOX);
            mmq_tma_load(s_gate, &map, 0, chunk_gn, &bar[0], policy);
        }

        mmq_tma_mbar_wait(&bar[1], p1);
        p1 ^= 1;

        float sum_u[4] = {0.0f};
        {
            const char * const xu0[2] = {
                s_up + (lane/4 + 0)*MMQ_TMA_ROW_BYTES + warp*MMQ_TMA_UNIT_BYTES,
                s_up + (lane/4 + 8)*MMQ_TMA_ROW_BYTES + warp*MMQ_TMA_UNIT_BYTES,
            };
            mmq_tma_vec_dot_q3_K(xu0, yf0, sum_u);
            const char * const xu1[2] = {
                s_up + (lane/4 + 0)*MMQ_TMA_ROW_BYTES + (warp + NWARPS)*MMQ_TMA_UNIT_BYTES,
                s_up + (lane/4 + 8)*MMQ_TMA_ROW_BYTES + (warp + NWARPS)*MMQ_TMA_UNIT_BYTES,
            };
            mmq_tma_vec_dot_q3_K(xu1, yf1, sum_u);
        }

        if (warp > 0) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                buf_gate[((warp - 1)*4 + i)*WARP_SIZE + lane] = sum_g[i];
                buf_up[((warp - 1)*4 + i)*WARP_SIZE + lane]   = sum_u[i];
            }
        }
        // Release s_up for next iter's up(t+1) TMA and publish buf_gate/buf_up to warp 0.
        __syncthreads();

        if (warp == 0) {
#pragma unroll
            for (int w = 0; w < NWARPS - 1; ++w) {
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    sum_g[i] += buf_gate[(w*4 + i)*WARP_SIZE + lane];
                    sum_u[i] += buf_up[(w*4 + i)*WARP_SIZE + lane];
                }
            }
            const int ncols = pair[1];
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                const int i = row0 + tile<16, 8, int>::get_i(l);
                const int j = tile<16, 8, int>::get_j(l);
                if (i < n_ff && j < ncols) {
                    const int e = pair[2 + j];
                    const float r = ggml_cuda_op_situ_single(sum_g[l], sum_u[l], beta, lb);
                    dst[int64_t(e / ncols_dst)*stride_channel_dst + int64_t(e % ncols_dst)*stride_col_dst + i] = r;
                }
            }
        }
    }

    ggml_cuda_pdl_lc();
#else
    GGML_UNUSED_VARS(map, y, dst, n_ff, expert_chunks, half_chunks, ncols_dst, stride_col_dst, stride_channel_dst,
        pairs, npairs_ptr, beta, lb);
    NO_DEVICE_CODE;
#endif
}

// Persistent 2-slot double-buffered Q4_K / Q5_K TMA kernel for any 16-byte-aligned K = nunits * 256.
template <ggml_type type, int NWARPS, int OCC, bool SINGLE_UNIT>
static __global__ __launch_bounds__(NWARPS*WARP_SIZE, OCC)
void mul_mat_q_tma_q45_K(const __grid_constant__ CUtensorMap map, const int4 * y, float * dst,
                         const int nrows_x, const int64_t expert_rows, const int nunits,
                         const int row_bytes, const int tile_bytes, const int tile_stride,
                         const int ncols_dst, const int stride_col_dst, const int64_t stride_channel_dst,
                         const int * pairs, const int * npairs_ptr) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == GGML_CUDA_CC_BLACKWELL
    constexpr int UNIT_BYTES    = type == GGML_TYPE_Q4_K ? MMQ_TMA_Q4K_UNIT_BYTES : MMQ_TMA_Q5K_UNIT_BYTES;
    constexpr int REDUCE_FLOATS = (NWARPS - 1)*4*WARP_SIZE;

    extern __shared__ __align__(128) char s_dyn[];
    char *     s_buf0   = s_dyn;
    char *     s_buf1   = s_dyn + tile_stride;
    uint64_t * bar      = reinterpret_cast<uint64_t *>(s_dyn + 2*tile_stride);
    float *    buf_sum0 = reinterpret_cast<float *>(bar + 2);
    float *    buf_sum1 = buf_sum0 + REDUCE_FLOATS;

    const int warp = threadIdx.y;
    const int lane = threadIdx.x;
    const int tid  = warp*WARP_SIZE + lane;

    ggml_cuda_pdl_sync();

    const int npairs      = *npairs_ptr;
    const int n_row_tiles = nrows_x / MMQ_TMA_TILE_ROWS;
    const int total_tiles = npairs * n_row_tiles;
    const int t_begin     = (int) ((int64_t(total_tiles) * blockIdx.x) / gridDim.x);
    const int t_end       = (int) ((int64_t(total_tiles) * (blockIdx.x + 1)) / gridDim.x);
    if (t_begin >= t_end) {
        ggml_cuda_pdl_lc();
        return;
    }

    const uint64_t policy = mmq_tma_evict_first();
    if (tid == 0) {
        mmq_tma_mbar_init(&bar[0]);
        mmq_tma_mbar_init(&bar[1]);
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();

    if (tid == 0) {
        const int p0  = t_begin / n_row_tiles;
        const int rt0 = t_begin % n_row_tiles;
        const int e0  = pairs[p0*MMQ_FEW_PAIR_INTS];
        mmq_tma_mbar_expect(&bar[0], (uint32_t) tile_bytes);
        const int y_row0 = (int) (int64_t(e0)*expert_rows + int64_t(rt0)*MMQ_TMA_TILE_ROWS);
        mmq_tma_load(s_buf0, &map, 0, y_row0, &bar[0], policy);
    }
    __syncthreads();

    uint32_t par0 = 0;
    uint32_t par1 = 0;
    int cur = 0;
    int cached_p = -1;
    mmq_tma_y_frag yf;

    for (int t = t_begin; t < t_end; ++t) {
        if (tid == 0 && t + 1 < t_end) {
            const int pn  = (t + 1) / n_row_tiles;
            const int rtn = (t + 1) % n_row_tiles;
            const int en  = pairs[pn*MMQ_FEW_PAIR_INTS];
            const int y_rown = (int) (int64_t(en)*expert_rows + int64_t(rtn)*MMQ_TMA_TILE_ROWS);
            uint64_t * bar_nxt = cur == 0 ? &bar[1] : &bar[0];
            char *     s_nxt   = cur == 0 ? s_buf1 : s_buf0;
            mmq_tma_mbar_expect(bar_nxt, (uint32_t) tile_bytes);
            mmq_tma_load(s_nxt, &map, 0, y_rown, bar_nxt, policy);
        }

        const int p    = t / n_row_tiles;
        const int rt   = t % n_row_tiles;
        const int row0 = rt*MMQ_TMA_TILE_ROWS;
        const int * pair = pairs + p*MMQ_FEW_PAIR_INTS;
        const int4 * y_pair = y + int64_t(p)*nunits*MMQ_FEW_Q8_1_INT4;

        if constexpr (SINGLE_UNIT) {
            if (p != cached_p && warp < nunits) {
                mmq_tma_load_y(yf, y_pair + warp*MMQ_FEW_Q8_1_INT4);
                cached_p = p;
            }
        } else if (warp < nunits) {
            mmq_tma_load_y(yf, y_pair + warp*MMQ_FEW_Q8_1_INT4);
        }

        if (cur == 0) {
            mmq_tma_mbar_wait(&bar[0], par0);
            par0 ^= 1;
        } else {
            mmq_tma_mbar_wait(&bar[1], par1);
            par1 ^= 1;
        }

        float sum[4] = {0.0f};
        const char * s_cur = cur == 0 ? s_buf0 : s_buf1;
        if constexpr (SINGLE_UNIT) {
            if (warp < nunits) {
                const char * const xr[2] = {
                    s_cur + (lane/4 + 0)*row_bytes + warp*UNIT_BYTES,
                    s_cur + (lane/4 + 8)*row_bytes + warp*UNIT_BYTES,
                };
                mmq_tma_vec_dot_q45_K<type>(xr, yf, sum);
            }
        } else {
            for (int iu = warp; iu < nunits; iu += NWARPS) {
                if (iu != warp) {
                    mmq_tma_load_y(yf, y_pair + iu*MMQ_FEW_Q8_1_INT4);
                }
                const char * const xr[2] = {
                    s_cur + (lane/4 + 0)*row_bytes + iu*UNIT_BYTES,
                    s_cur + (lane/4 + 8)*row_bytes + iu*UNIT_BYTES,
                };
                mmq_tma_vec_dot_q45_K<type>(xr, yf, sum);
            }
        }

        float * buf_sum = cur == 0 ? buf_sum0 : buf_sum1;
        if (warp > 0) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                buf_sum[((warp - 1)*4 + i)*WARP_SIZE + lane] = sum[i];
            }
        }
        __syncthreads();

        if (warp == 0) {
#pragma unroll
            for (int w = 0; w < NWARPS - 1; ++w) {
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    sum[i] += buf_sum[(w*4 + i)*WARP_SIZE + lane];
                }
            }
            const int ncols = pair[1];
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                const int i = row0 + tile<16, 8, int>::get_i(l);
                const int j = tile<16, 8, int>::get_j(l);
                if (i < nrows_x && j < ncols) {
                    const int e = pair[2 + j];
                    dst[int64_t(e / ncols_dst)*stride_channel_dst + int64_t(e % ncols_dst)*stride_col_dst + i] = sum[l];
                }
            }
        }
        cur ^= 1;
    }

    ggml_cuda_pdl_lc();
#else
    GGML_UNUSED_VARS(map, y, dst, nrows_x, expert_rows, nunits, row_bytes, tile_bytes, tile_stride,
        ncols_dst, stride_col_dst, stride_channel_dst, pairs, npairs_ptr);
    NO_DEVICE_CODE;
#endif
}

// Map global expert ids to [0, count) or -1.
static __global__ void mmq_tma_local_ids_kernel(
        const int32_t * GGML_CUDA_RESTRICT global_ids, int32_t * GGML_CUDA_RESTRICT local_ids,
        const int n_entries, const int n_used, const int ids_stride,
        const int32_t first, const int32_t count) {
    ggml_cuda_pdl_sync();
    const int entry = (int) (blockIdx.x*blockDim.x + threadIdx.x);
    if (entry >= n_entries) {
        return;
    }
    const int token = entry / n_used;
    const int slot  = entry % n_used;
    const int32_t e = global_ids[token*ids_stride + slot];
    local_ids[entry] = (e >= first && e < first + count) ? (e - first) : -1;
}

// Zero non-owned dst rows after inputs have been staged.
static __global__ void mmq_tma_zero_non_owned_kernel(
        const int32_t * GGML_CUDA_RESTRICT local_ids, float * GGML_CUDA_RESTRICT zero_dst,
        const int n_entries, const int row_elems, const int row_blocks) {
    ggml_cuda_pdl_sync();
    const int entry = (int) (blockIdx.x / row_blocks);
    if (entry >= n_entries || local_ids[entry] >= 0) {
        return;
    }
    const int rb  = (int) (blockIdx.x % row_blocks);
    const int col = rb*blockDim.x + threadIdx.x;
    if (col < row_elems) {
        zero_dst[int64_t(entry)*row_elems + col] = 0.0f;
    }
}

// Deterministic slot-order weighted sum over local expert outputs only (skips non-owned slots without memset).
static __global__ void mmq_tma_weighted_sum_kernel(
        const float * GGML_CUDA_RESTRICT tmp_down, const int32_t * GGML_CUDA_RESTRICT local_ids,
        const float * GGML_CUDA_RESTRICT weights, float * GGML_CUDA_RESTRICT dst,
        const int n_embd, const int n_used, const int dst_col_stride, const int weights_s2) {
    ggml_cuda_pdl_sync();
    const int token = blockIdx.y;
    const int col4  = (blockIdx.x*blockDim.x + threadIdx.x)*4;
    if (col4 >= n_embd) {
        return;
    }
    const int32_t * lids = local_ids + token*n_used;
    const float   * w    = weights   + token*weights_s2;
    const float   * base = tmp_down  + int64_t(token)*n_used*n_embd + col4;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    for (int s = 0; s < n_used; ++s) {
        const float ws = w[s];
        float v0 = 0.0f, v1 = 0.0f, v2 = 0.0f, v3 = 0.0f;
        if (lids[s] >= 0) {
            const float4 v = *reinterpret_cast<const float4 *>(base + int64_t(s)*n_embd);
            v0 = v.x; v1 = v.y; v2 = v.z; v3 = v.w;
        }
        if (s == 0) {
            a0 = __fmul_rn(v0, ws);
            a1 = __fmul_rn(v1, ws);
            a2 = __fmul_rn(v2, ws);
            a3 = __fmul_rn(v3, ws);
        } else {
            a0 = __fadd_rn(a0, __fmul_rn(v0, ws));
            a1 = __fadd_rn(a1, __fmul_rn(v1, ws));
            a2 = __fadd_rn(a2, __fmul_rn(v2, ws));
            a3 = __fadd_rn(a3, __fmul_rn(v3, ws));
        }
    }
    *reinterpret_cast<float4 *>(dst + int64_t(token)*dst_col_stride + col4) = make_float4(a0, a1, a2, a3);
}

// Opt in to >48 KiB dynamic SMEM per kernel and device, safe inside CUDA graph capture.
template <typename KernelFn>
static void mmq_tma_ensure_smem(const int device, KernelFn kernel_fn,
        std::atomic<int> (&max_smem)[GGML_CUDA_MAX_DEVICES], const size_t smem_bytes) {
    if (smem_bytes <= 48*1024) {
        return;
    }
    const int need = (int) smem_bytes;
    int prev = max_smem[device].load(std::memory_order_acquire);
    if (need <= prev) {
        return;
    }
    cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
    CUDA_CHECK(cudaThreadExchangeStreamCaptureMode(&mode));
    CUDA_CHECK(cudaFuncSetAttribute(kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, need));
    CUDA_CHECK(cudaThreadExchangeStreamCaptureMode(&mode));
    while (need > prev && !max_smem[device].compare_exchange_weak(prev, need, std::memory_order_release)) {
    }
}

template <ggml_type type, int NWARPS, int OCC, bool SINGLE_UNIT>
static void launch_mul_mat_q_tma_q45_K(
        const int device, const CUtensorMap & map, const int4 * y, float * dst,
        const int nrows_x, const int64_t expert_rows, const int nunits, const int row_bytes,
        const int ncols_dst, const int stride_col_dst, const int64_t stride_channel_dst,
        const int * pairs, const int * npairs_ptr, const int npairs_max, cudaStream_t stream) {
    const int tile_bytes  = MMQ_TMA_TILE_ROWS * row_bytes;
    const int tile_stride = ((tile_bytes + 127) / 128) * 128;
    const size_t smem_bytes = (size_t) (2*tile_stride) + 2*sizeof(uint64_t) +
        (size_t) (2*(NWARPS - 1)*4*WARP_SIZE)*sizeof(float);
    static std::atomic<int> max_smem[GGML_CUDA_MAX_DEVICES];
    mmq_tma_ensure_smem(device, mul_mat_q_tma_q45_K<type, NWARPS, OCC, SINGLE_UNIT>, max_smem, smem_bytes);

    const int nsm = ggml_cuda_info().devices[device].nsm;
    const int max_tiles = (nrows_x / MMQ_TMA_TILE_ROWS) * npairs_max;
    const int grid_x = std::max(1, std::min(nsm * OCC, max_tiles));
    const dim3 grid((unsigned) grid_x, 1, 1);
    const dim3 block(WARP_SIZE, NWARPS, 1);
    const ggml_cuda_kernel_launch_params params(grid, block, smem_bytes, stream);
    ggml_cuda_kernel_launch(mul_mat_q_tma_q45_K<type, NWARPS, OCC, SINGLE_UNIT>, params,
        map, y, dst, nrows_x, expert_rows, nunits, row_bytes, tile_bytes, tile_stride,
        ncols_dst, stride_col_dst, stride_channel_dst, pairs, npairs_ptr);
}

template <ggml_type type>
static void dispatch_mul_mat_q_tma_q45_K(
        const int device, const CUtensorMap & map, const int4 * y, float * dst,
        const int nrows_x, const int64_t expert_rows, const int nunits, const int row_bytes,
        const int ncols_dst, const int stride_col_dst, const int64_t stride_channel_dst,
        const int * pairs, const int * npairs_ptr, const int npairs_max, cudaStream_t stream) {
    if (nunits == 9) {
        // 9 warps x 1 unit/warp: 0% warp imbalance, y cached in regs across row tiles of each pair.
        launch_mul_mat_q_tma_q45_K<type, 9, 1, true>(device, map, y, dst, nrows_x, expert_rows, nunits, row_bytes,
            ncols_dst, stride_col_dst, stride_channel_dst, pairs, npairs_ptr, npairs_max, stream);
    } else if (nunits == 6) {
        launch_mul_mat_q_tma_q45_K<type, 6, 2, true>(device, map, y, dst, nrows_x, expert_rows, nunits, row_bytes,
            ncols_dst, stride_col_dst, stride_channel_dst, pairs, npairs_ptr, npairs_max, stream);
    } else if (nunits == 4) {
        launch_mul_mat_q_tma_q45_K<type, 4, 2, true>(device, map, y, dst, nrows_x, expert_rows, nunits, row_bytes,
            ncols_dst, stride_col_dst, stride_channel_dst, pairs, npairs_ptr, npairs_max, stream);
    } else {
        GGML_ABORT("unexpected nunits %d for Q4_K/Q5_K TMA", nunits);
    }
}

// The tensor map of one weight buffer, encoded once and kept: the weights never move under it.
static const CUtensorMap * mmq_tma_map_of(const int device, const void * base, const int chunk_bytes,
                                          const int box_rows, const int64_t chunks) {
    static std::mutex mutex;
    static std::map<std::tuple<int, const void *, int, int, int64_t>, CUtensorMap> maps;
    std::lock_guard<std::mutex> lock(mutex);

    const auto key = std::make_tuple(device, base, chunk_bytes, box_rows, chunks);
    const auto it = maps.find(key);
    if (it != maps.end()) {
        return &it->second;
    }

    static const auto encode = [] {
        void * p = nullptr;
        cudaDriverEntryPointQueryResult query;
        const auto err = cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &p, 12000, cudaEnableDefault, &query);
        if (err != cudaSuccess || query != cudaDriverEntryPointSuccess) {
            (void) cudaGetLastError();
            return (PFN_cuTensorMapEncodeTiled_v12000) nullptr;
        }
        return (PFN_cuTensorMapEncodeTiled_v12000) p;
    }();
    if (!encode) {
        return nullptr;
    }

    CUtensorMap map;
    const cuuint64_t dims[]    = { (cuuint64_t) (chunk_bytes/8), (cuuint64_t) chunks };
    const cuuint64_t strides[] = { (cuuint64_t) chunk_bytes };
    const cuuint32_t box[]     = { (cuuint32_t) (chunk_bytes/8), (cuuint32_t) box_rows };
    const cuuint32_t esteps[]  = { 1, 1 };
    if (encode(&map, CU_TENSOR_MAP_DATA_TYPE_UINT64, 2, const_cast<void *>(base), dims, strides, box, esteps,
               CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
               CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) != CUDA_SUCCESS) {
        return nullptr;
    }
    return &maps.emplace(key, map).first->second;
}

static bool mmq_tma_arch_ok(const int device) {
    const int cc = ggml_cuda_info().devices[device].cc;
    return cc == GGML_CUDA_CC_BLACKWELL && ggml_cuda_highest_compiled_arch(cc) == GGML_CUDA_CC_BLACKWELL;
}

static bool mmq_tma_q45_shape_ok(const ggml_tensor * w, int & nunits, int & row_bytes) {
    if (w->type != GGML_TYPE_Q4_K && w->type != GGML_TYPE_Q5_K) {
        return false;
    }
    if (w->ne[0] % QK_K != 0) {
        return false;
    }
    const int u = (int) (w->ne[0] / QK_K);
    if (u != 4 && u != 6 && u != 9) {
        return false;
    }
    const int unit_bytes = w->type == GGML_TYPE_Q4_K ? MMQ_TMA_Q4K_UNIT_BYTES : MMQ_TMA_Q5K_UNIT_BYTES;
    nunits    = u;
    row_bytes = nunits * unit_bytes;
    return row_bytes / 8 <= MMQ_TMA_Q45_MAX_BOX_U64;
}

bool ggml_cuda_mmq_tma_id(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                          const ggml_tensor * ids, ggml_tensor * dst) {
    static const bool enabled = ggml_env_switch("GGML_CUDA_MMQ_TMA");
    if (!enabled || src0 == nullptr || ids == nullptr) {
        return false;
    }
    const bool is_q3k = src0->type == GGML_TYPE_Q3_K && src0->ne[0] == MMQ_TMA_UNITS*QK_K;
    int q45_nunits = 0;
    int q45_row_bytes = 0;
    const bool is_q45 = !is_q3k && mmq_tma_q45_shape_ok(src0, q45_nunits, q45_row_bytes);
    if ((!is_q3k && !is_q45) || src0->ne[1] % MMQ_TMA_TILE_ROWS != 0 || src0->ne[2] < 2 || src0->ne[3] != 1) {
        return false;
    }
    const int chunk_bytes = is_q3k ? MMQ_TMA_CHUNK : q45_row_bytes;
    const int row_bytes   = is_q3k ? MMQ_TMA_ROW_BYTES : q45_row_bytes;
    if (src0->nb[0] != ggml_type_size(src0->type) || src0->nb[1] != (size_t) row_bytes ||
            src0->nb[2] % chunk_bytes != 0 || !ggml_is_contiguous(src0) ||
            (uintptr_t) src0->data % 16 != 0) {
        return false;
    }
    if (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 ||
            src1->nb[0] != sizeof(float) || dst->nb[0] != sizeof(float) ||
            src1->ne[3] != 1 || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst) ||
            (uintptr_t) src1->data % 16 != 0) {
        return false;
    }
    if (ids->type != GGML_TYPE_I32 || ids->nb[0] != sizeof(int32_t) || ids->ne[2] != 1 || ids->ne[3] != 1 ||
            ids->ne[1] < 2 || ids->ne[1] > MMQ_TMA_MAX_TOKENS || src1->ne[2] != ids->ne[1] ||
            ids->ne[0] % src1->ne[1] != 0 ||
            dst->ne[1] != ids->ne[0] || dst->ne[2] != ids->ne[1] || dst->ne[3] != 1) {
        return false;
    }
    if (!mmq_tma_arch_ok(ctx.device)) {
        return false;
    }

    const int box_rows = is_q3k ? MMQ_TMA_BOX : MMQ_TMA_TILE_ROWS;
    const int64_t chunks = (int64_t) src0->ne[2]*src0->nb[2]/chunk_bytes;
    const CUtensorMap * map = mmq_tma_map_of(ctx.device, src0->data, chunk_bytes, box_rows, chunks);
    if (map == nullptr) {
        return false;
    }

    cudaStream_t stream = ctx.stream();

    const int64_t ne10          = src1->ne[0];
    const int64_t n_expert_used = ids->ne[0];
    const int64_t n_tokens      = ids->ne[1];
    const int64_t nentries      = n_tokens*n_expert_used;
    const int64_t nused         = std::min<int64_t>(src0->ne[2], nentries);
    const int npairs_max = (int) std::min<int64_t>(nused + (nentries - nused)/MMQ_FEW_NCOLS,
        src0->ne[2]*((n_tokens + MMQ_FEW_NCOLS - 1)/MMQ_FEW_NCOLS));
    const int nunits = is_q3k ? MMQ_TMA_UNITS : q45_nunits;

    ggml_cuda_pool_alloc<int>  pairs(ctx.pool());
    ggml_cuda_pool_alloc<int4> src1_q(ctx.pool());
    pairs.alloc((size_t) npairs_max*MMQ_FEW_PAIR_INTS + 1);
    int * npairs_ptr = pairs.get() + (size_t) npairs_max*MMQ_FEW_PAIR_INTS;
    src1_q.alloc((size_t) npairs_max*nunits*MMQ_FEW_Q8_1_INT4);
    quantize_mmq_few_id_q8_1_cuda((const float *) src1->data, (const int32_t *) ids->data, src1_q.get(), pairs.get(),
        is_q3k ? MMQ_Q8_1_DS_LAYOUT_D4 : MMQ_Q8_1_DS_LAYOUT_DS4, ne10, src1->nb[1]/sizeof(float), src1->nb[2]/sizeof(float),
        src1->ne[1], src0->ne[2], n_expert_used, n_tokens, ids->nb[1]/sizeof(int32_t), npairs_max, stream, npairs_ptr);

    if (is_q3k) {
        constexpr size_t smem_bytes = (size_t) (2*MMQ_TMA_TILE_STRIDE) + 2*sizeof(uint64_t) +
            (size_t) (2*(MMQ_TMA_Q3K_NWARPS - 1)*4*WARP_SIZE)*sizeof(float);
        static std::atomic<int> max_smem_q3k[GGML_CUDA_MAX_DEVICES];
        mmq_tma_ensure_smem(ctx.device, mul_mat_q_tma_q3_K, max_smem_q3k, smem_bytes);
        const int nsm = ggml_cuda_info().devices[ctx.device].nsm;
        const int max_tiles = (int) (src0->ne[1]/MMQ_TMA_TILE_ROWS) * npairs_max;
        const dim3 grid((unsigned) std::max(1, std::min(nsm, max_tiles)), 1, 1);
        const dim3 block(WARP_SIZE, MMQ_TMA_Q3K_NWARPS, 1);
        const ggml_cuda_kernel_launch_params params(grid, block, smem_bytes, stream);
        ggml_cuda_kernel_launch(mul_mat_q_tma_q3_K, params, *map, (const int4 *) src1_q.get(), (float *) dst->data,
            (int) src0->ne[1], (int64_t) (src0->nb[2]/MMQ_TMA_CHUNK), (int) dst->ne[1],
            (int) (dst->nb[1]/sizeof(float)), (int64_t) (dst->nb[2]/sizeof(float)), pairs.get(), npairs_ptr);
    } else if (src0->type == GGML_TYPE_Q4_K) {
        dispatch_mul_mat_q_tma_q45_K<GGML_TYPE_Q4_K>(ctx.device, *map, (const int4 *) src1_q.get(), (float *) dst->data,
            (int) src0->ne[1], (int64_t) (src0->nb[2]/q45_row_bytes), q45_nunits, q45_row_bytes,
            (int) dst->ne[1], (int) (dst->nb[1]/sizeof(float)), (int64_t) (dst->nb[2]/sizeof(float)),
            pairs.get(), npairs_ptr, npairs_max, stream);
    } else {
        dispatch_mul_mat_q_tma_q45_K<GGML_TYPE_Q5_K>(ctx.device, *map, (const int4 *) src1_q.get(), (float *) dst->data,
            (int) src0->ne[1], (int64_t) (src0->nb[2]/q45_row_bytes), q45_nunits, q45_row_bytes,
            (int) dst->ne[1], (int) (dst->nb[1]/sizeof(float)), (int64_t) (dst->nb[2]/sizeof(float)),
            pairs.get(), npairs_ptr, npairs_max, stream);
    }
    return true;
}

bool ggml_cuda_mmq_tma_gate_up_situ(ggml_backend_cuda_context & ctx, const ggml_tensor * w, const ggml_tensor * x,
                                    const ggml_tensor * ids, ggml_tensor * glu, int32_t first, int32_t count,
                                    float beta, float lb, const int32_t * precomputed_local_ids,
                                    bool zero_non_owned) {
    static const bool enabled = ggml_env_switch("GGML_CUDA_MMQ_TMA");
    if (!enabled || !w || !x || !ids || !glu || w->type != GGML_TYPE_Q3_K) {
        return false;
    }
    const int64_t n_ff = w->ne[1]/2;
    if (w->ne[0] != MMQ_TMA_UNITS*QK_K || n_ff % MMQ_TMA_TILE_ROWS != 0 || (n_ff*MMQ_TMA_ROW_BYTES) % MMQ_TMA_CHUNK != 0 ||
            w->ne[2] != count || w->ne[2] < 2 || w->ne[3] != 1) {
        return false;
    }
    if (w->nb[0] != ggml_type_size(w->type) || w->nb[1] != MMQ_TMA_ROW_BYTES ||
            w->nb[2] % MMQ_TMA_CHUNK != 0 || !ggml_is_contiguous(w) || (uintptr_t) w->data % 16 != 0) {
        return false;
    }
    if (x->type != GGML_TYPE_F32 || glu->type != GGML_TYPE_F32 || !ggml_is_contiguous(x) || !ggml_is_contiguous(glu) ||
            (uintptr_t) x->data % 16 != 0 ||
            ids->type != GGML_TYPE_I32 || ids->nb[0] != sizeof(int32_t) ||
            ids->ne[1] < 2 || ids->ne[1] > MMQ_TMA_MAX_TOKENS || x->ne[2] != ids->ne[1] ||
            ids->ne[0] % x->ne[1] != 0 || glu->ne[0] != n_ff || glu->ne[1] != ids->ne[0] || glu->ne[2] != ids->ne[1]) {
        return false;
    }
    if (!mmq_tma_arch_ok(ctx.device)) {
        return false;
    }

    const int64_t chunks = (int64_t) w->ne[2]*w->nb[2]/MMQ_TMA_CHUNK;
    const CUtensorMap * map = mmq_tma_map_of(ctx.device, w->data, MMQ_TMA_CHUNK, MMQ_TMA_BOX, chunks);
    if (map == nullptr) {
        return false;
    }

    constexpr size_t smem_bytes = (size_t) (2*MMQ_TMA_TILE_STRIDE) + 2*sizeof(uint64_t) +
        (size_t) (2*(MMQ_TMA_Q3K_NWARPS - 1)*4*WARP_SIZE)*sizeof(float);
    static std::atomic<int> max_smem_gu[GGML_CUDA_MAX_DEVICES];
    mmq_tma_ensure_smem(ctx.device, mul_mat_q_tma_q3_K_gate_up_situ, max_smem_gu, smem_bytes);

    cudaStream_t stream = ctx.stream();
    const int64_t n_expert_used = ids->ne[0];
    const int64_t n_tokens      = ids->ne[1];
    const int64_t nentries      = n_tokens*n_expert_used;
    const int64_t ids_stride    = ids->nb[1]/sizeof(int32_t);

    ggml_cuda_pool_alloc<int32_t> local_ids_buf(ctx.pool());
    const int32_t * local_ids = precomputed_local_ids;
    if (local_ids == nullptr) {
        local_ids_buf.alloc((size_t) nentries);
        const ggml_cuda_kernel_launch_params lid_params(
            dim3((unsigned) ((nentries + 255)/256), 1, 1), dim3(256, 1, 1), 0, stream);
        ggml_cuda_kernel_launch(mmq_tma_local_ids_kernel, lid_params,
            (const int32_t *) ids->data, local_ids_buf.get(),
            (int) nentries, (int) n_expert_used, (int) ids_stride, first, count);
        local_ids = local_ids_buf.get();
    }

    const int64_t nused = std::min<int64_t>(count, nentries);
    const int npairs_max = (int) std::min<int64_t>(nused + (nentries - nused)/MMQ_FEW_NCOLS,
        count*((n_tokens + MMQ_FEW_NCOLS - 1)/MMQ_FEW_NCOLS));
    ggml_cuda_pool_alloc<int>  pairs(ctx.pool(), (size_t) npairs_max*MMQ_FEW_PAIR_INTS + 1);
    int * npairs_ptr = pairs.get() + (size_t) npairs_max*MMQ_FEW_PAIR_INTS;
    ggml_cuda_pool_alloc<int4> src1_q(ctx.pool(), (size_t) npairs_max*MMQ_TMA_UNITS*MMQ_FEW_Q8_1_INT4);
    // Quantize x before zeroing non-owned glu rows so glu may alias x.
    quantize_mmq_few_id_q8_1_cuda((const float *) x->data, local_ids, src1_q.get(), pairs.get(),
        MMQ_Q8_1_DS_LAYOUT_D4, x->ne[0], x->nb[1]/sizeof(float), x->nb[2]/sizeof(float),
        x->ne[1], count, n_expert_used, n_tokens, n_expert_used, npairs_max, stream, npairs_ptr);

    if (zero_non_owned) {
        const int row_blocks = (int) ((n_ff + 255)/256);
        const ggml_cuda_kernel_launch_params zero_params(
            dim3((unsigned) (nentries*row_blocks), 1, 1), dim3(256, 1, 1), 0, stream);
        ggml_cuda_kernel_launch(mmq_tma_zero_non_owned_kernel, zero_params,
            local_ids, (float *) glu->data, (int) nentries, (int) n_ff, row_blocks);
    }

    const int nsm = ggml_cuda_info().devices[ctx.device].nsm;
    const int max_tiles = (int) (n_ff/MMQ_TMA_TILE_ROWS) * npairs_max;
    const dim3 grid((unsigned) std::max(1, std::min(nsm, max_tiles)), 1, 1);
    const dim3 block(WARP_SIZE, MMQ_TMA_Q3K_NWARPS, 1);
    const ggml_cuda_kernel_launch_params params(grid, block, smem_bytes, stream);
    ggml_cuda_kernel_launch(mul_mat_q_tma_q3_K_gate_up_situ, params, *map, (const int4 *) src1_q.get(), (float *) glu->data,
        (int) n_ff, (int64_t) (w->nb[2]/MMQ_TMA_CHUNK), (int64_t) (n_ff*MMQ_TMA_ROW_BYTES/MMQ_TMA_CHUNK),
        (int) glu->ne[1], (int) (glu->nb[1]/sizeof(float)), (int64_t) (glu->nb[2]/sizeof(float)),
        pairs.get(), npairs_ptr, beta, lb);
    return true;
}

bool ggml_cuda_mmq_tma_down_weighted_sum(ggml_backend_cuda_context & ctx, const ggml_tensor * w, const ggml_tensor * x,
                                         const ggml_tensor * ids, const ggml_tensor * weights, ggml_tensor * dst,
                                         int32_t first, int32_t count,
                                         const int32_t * precomputed_local_ids) {
    static const bool enabled = ggml_env_switch("GGML_CUDA_MMQ_TMA");
    if (!enabled || !w || !x || !ids || !weights || !dst) {
        return false;
    }
    int nunits = 0;
    int row_bytes = 0;
    if (!mmq_tma_q45_shape_ok(w, nunits, row_bytes)) {
        return false;
    }
    if (w->ne[1] % MMQ_TMA_TILE_ROWS != 0 || w->ne[2] != count || w->ne[2] < 2 || w->ne[3] != 1) {
        return false;
    }
    if (w->nb[0] != ggml_type_size(w->type) || w->nb[1] != (size_t) row_bytes ||
            w->nb[2] % row_bytes != 0 || !ggml_is_contiguous(w) || (uintptr_t) w->data % 16 != 0) {
        return false;
    }
    if (x->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || weights->type != GGML_TYPE_F32 ||
            !ggml_is_contiguous(x) || !ggml_is_contiguous(dst) || (uintptr_t) x->data % 16 != 0 ||
            ids->type != GGML_TYPE_I32 || ids->nb[0] != sizeof(int32_t) ||
            ids->ne[1] < 2 || ids->ne[1] > MMQ_TMA_MAX_TOKENS || x->ne[2] != ids->ne[1] ||
            ids->ne[0] % x->ne[1] != 0 || weights->ne[0] != 1 || weights->nb[1] != sizeof(float) ||
            dst->ne[0] != w->ne[1] || dst->ne[1] != ids->ne[1] ||
            (dst->ne[0] & 3) != 0 || (uintptr_t) dst->data % 16 != 0) {
        return false;
    }
    if (!mmq_tma_arch_ok(ctx.device)) {
        return false;
    }

    const int64_t chunks = (int64_t) w->ne[2]*w->nb[2]/row_bytes;
    const CUtensorMap * map = mmq_tma_map_of(ctx.device, w->data, row_bytes, MMQ_TMA_TILE_ROWS, chunks);
    if (map == nullptr) {
        return false;
    }

    cudaStream_t stream = ctx.stream();
    const int64_t n_embd        = w->ne[1];
    const int64_t n_expert_used = ids->ne[0];
    const int64_t n_tokens      = ids->ne[1];
    const int64_t nentries      = n_tokens*n_expert_used;
    const int64_t ids_stride    = ids->nb[1]/sizeof(int32_t);

    ggml_cuda_pool_alloc<int32_t> local_ids_buf(ctx.pool());
    const int32_t * local_ids = precomputed_local_ids;
    if (local_ids == nullptr) {
        local_ids_buf.alloc((size_t) nentries);
        const ggml_cuda_kernel_launch_params lid_params(
            dim3((unsigned) ((nentries + 255)/256), 1, 1), dim3(256, 1, 1), 0, stream);
        ggml_cuda_kernel_launch(mmq_tma_local_ids_kernel, lid_params,
            (const int32_t *) ids->data, local_ids_buf.get(),
            (int) nentries, (int) n_expert_used, (int) ids_stride, first, count);
        local_ids = local_ids_buf.get();
    }

    const int64_t nused = std::min<int64_t>(count, nentries);
    const int npairs_max = (int) std::min<int64_t>(nused + (nentries - nused)/MMQ_FEW_NCOLS,
        count*((n_tokens + MMQ_FEW_NCOLS - 1)/MMQ_FEW_NCOLS));
    ggml_cuda_pool_alloc<int>   pairs(ctx.pool(), (size_t) npairs_max*MMQ_FEW_PAIR_INTS + 1);
    int * npairs_ptr = pairs.get() + (size_t) npairs_max*MMQ_FEW_PAIR_INTS;
    ggml_cuda_pool_alloc<int4>  src1_q(ctx.pool(), (size_t) npairs_max*nunits*MMQ_FEW_Q8_1_INT4);
    ggml_cuda_pool_alloc<float> tmp_down(ctx.pool(), (size_t) nentries*n_embd);

    quantize_mmq_few_id_q8_1_cuda((const float *) x->data, local_ids, src1_q.get(), pairs.get(),
        MMQ_Q8_1_DS_LAYOUT_DS4, x->ne[0], x->nb[1]/sizeof(float), x->nb[2]/sizeof(float),
        x->ne[1], count, n_expert_used, n_tokens, n_expert_used, npairs_max, stream, npairs_ptr);

    if (w->type == GGML_TYPE_Q4_K) {
        dispatch_mul_mat_q_tma_q45_K<GGML_TYPE_Q4_K>(ctx.device, *map, (const int4 *) src1_q.get(), tmp_down.get(),
            (int) n_embd, (int64_t) (w->nb[2]/row_bytes), nunits, row_bytes,
            (int) n_expert_used, (int) n_embd, (int64_t) (n_expert_used*n_embd),
            pairs.get(), npairs_ptr, npairs_max, stream);
    } else {
        dispatch_mul_mat_q_tma_q45_K<GGML_TYPE_Q5_K>(ctx.device, *map, (const int4 *) src1_q.get(), tmp_down.get(),
            (int) n_embd, (int64_t) (w->nb[2]/row_bytes), nunits, row_bytes,
            (int) n_expert_used, (int) n_embd, (int64_t) (n_expert_used*n_embd),
            pairs.get(), npairs_ptr, npairs_max, stream);
    }

    const dim3 sum_grid((unsigned) ((n_embd/4 + 255)/256), (unsigned) n_tokens, 1);
    const ggml_cuda_kernel_launch_params sum_params(sum_grid, dim3(256, 1, 1), 0, stream);
    ggml_cuda_kernel_launch(mmq_tma_weighted_sum_kernel, sum_params,
        tmp_down.get(), local_ids, (const float *) weights->data, (float *) dst->data,
        (int) n_embd, (int) n_expert_used, (int) (dst->nb[1]/sizeof(float)),
        (int) (weights->nb[2]/sizeof(float)));
    return true;
}

#else

bool ggml_cuda_mmq_tma_id(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                          const ggml_tensor * ids, ggml_tensor * dst) {
    GGML_UNUSED_VARS(ctx, src0, src1, ids, dst);
    return false;
}

bool ggml_cuda_mmq_tma_gate_up_situ(ggml_backend_cuda_context & ctx, const ggml_tensor * w, const ggml_tensor * x,
                                    const ggml_tensor * ids, ggml_tensor * glu, int32_t first, int32_t count,
                                    float beta, float lb, const int32_t * precomputed_local_ids,
                                    bool zero_non_owned) {
    GGML_UNUSED_VARS(ctx, w, x, ids, glu, first, count, beta, lb, precomputed_local_ids, zero_non_owned);
    return false;
}

bool ggml_cuda_mmq_tma_down_weighted_sum(ggml_backend_cuda_context & ctx, const ggml_tensor * w, const ggml_tensor * x,
                                         const ggml_tensor * ids, const ggml_tensor * weights, ggml_tensor * dst,
                                         int32_t first, int32_t count,
                                         const int32_t * precomputed_local_ids) {
    GGML_UNUSED_VARS(ctx, w, x, ids, weights, dst, first, count, precomputed_local_ids);
    return false;
}

#endif
