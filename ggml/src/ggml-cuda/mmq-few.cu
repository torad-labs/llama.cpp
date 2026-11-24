#include "mmq-few.cuh"
#include "cp-async.cuh"
#include "mma.cuh"
#include "mmvq.cuh"
#include "quantize.cuh"
#include "vecdotq.cuh"

#include <type_traits>

// NVIDIA only: the kernel uses the NVIDIA int8 MMA fragment layout and cp.async.
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

#include <cooperative_groups.h>

using namespace ggml_cuda_mma;

// Few-row MMQ: int8 MMA mat-mul for 2-32 src1 columns. A CUDA block has 16 src0 rows (32 for 2 column groups on large matrices).
// Its warps split K in units of 256 values and copy raw src0 rows into their own cp.async ring, src1 is quantized to the MMA fragment layout.
// MUL_MAT_ID: one column group per expert and CUDA block, for short rows the warps split the rows instead of K (64 rows per block).
// When the row tiles do not fill the GPU, a cluster of up to 8 blocks splits K of one tile, and its first block adds the
// others' sums from their shared memory in block order.

static constexpr int MMQ_FEW_MAX_NCG = 4;  // max. number of column groups
static constexpr int MMQ_FEW_NROWS   = 16; // src0 rows of one MMA tile
static constexpr int MMQ_FEW_NWARPS  = 4;
static constexpr int MMQ_FEW_MAX_CLUSTER = 8; // the largest portable cluster

template <ggml_type type>
static constexpr __host__ __device__ int mmq_few_unit_size_of() {
    return QK_K/ggml_cuda_type_traits<type>::qk*ggml_cuda_type_traits<type>::bs;
}

// src0 bytes per 256 values, 0 for the types without a few-row decoder.
static constexpr __host__ __device__ int mmq_few_unit_size(const ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q8_0:   return mmq_few_unit_size_of<GGML_TYPE_Q8_0>();
        case GGML_TYPE_Q3_K:   return mmq_few_unit_size_of<GGML_TYPE_Q3_K>();
        case GGML_TYPE_Q4_K:   return mmq_few_unit_size_of<GGML_TYPE_Q4_K>();
        case GGML_TYPE_Q5_K:   return mmq_few_unit_size_of<GGML_TYPE_Q5_K>();
        case GGML_TYPE_Q6_K:   return mmq_few_unit_size_of<GGML_TYPE_Q6_K>();
        case GGML_TYPE_IQ3_S:  return mmq_few_unit_size_of<GGML_TYPE_IQ3_S>();
        case GGML_TYPE_IQ4_NL: return mmq_few_unit_size_of<GGML_TYPE_IQ4_NL>();
        case GGML_TYPE_IQ4_XS: return mmq_few_unit_size_of<GGML_TYPE_IQ4_XS>();
        default:               return 0;
    }
}

// Q4_K and Q5_K also need the src1 sums for their mins.
static constexpr __host__ __device__ mmq_q8_1_ds_layout mmq_few_ds_layout(const ggml_type type) {
    return type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K ? MMQ_Q8_1_DS_LAYOUT_DS4 : MMQ_Q8_1_DS_LAYOUT_D4;
}

// Units of 256 values per ring stage: 1 for Q6_K, 3-4 column groups (fewer free registers) and 32-row tiles (2 blocks per SM), else 2.
static constexpr __host__ __device__ int mmq_few_nunits_stage(const ggml_type type, const int ncg, const int nmt) {
    return type == GGML_TYPE_Q6_K || ncg > 2 || nmt > 1 ? 1 : 2;
}

// A third ring stage only makes Q6_K faster, the other types get slower on large matrices.
static constexpr __host__ __device__ int mmq_few_nstages(const ggml_type type) {
    return type == GGML_TYPE_Q6_K ? 3 : 2;
}

// The src0 data of a ring stage does not fill whole 16 byte chunks, it is copied from the preceding 16 byte boundary.
static constexpr __host__ __device__ bool mmq_few_unaligned(const ggml_type type, const int nunits_stage) {
    return (nunits_stage*mmq_few_unit_size(type)) % 16 != 0;
}

// for unaligned data the max. over the 8 possible start offsets
static constexpr __host__ __device__ int mmq_few_nchunks(const ggml_type type, const int nunits_stage) {
    const int nbytes = nunits_stage*mmq_few_unit_size(type);
    return mmq_few_unaligned(type, nunits_stage) ? (14 + nbytes + 15)/16 : nbytes/16;
}

// Unaligned rows are padded to an odd number of 16 byte chunks against bank conflicts, for aligned rows the padding costs more than it saves.
static constexpr __host__ __device__ int mmq_few_sram_stride(const ggml_type type, const int nunits_stage) {
    const int nchunks = mmq_few_nchunks(type, nunits_stage);
    return mmq_few_unaligned(type, nunits_stage) && nchunks % 2 == 0 ? 16*(nchunks + 1) : 16*nchunks;
}

static constexpr __host__ __device__ int mmq_few_nbytes_shared(const ggml_type type, const int ncg, const int nmt) {
    return MMQ_FEW_NWARPS*mmq_few_nstages(type)*nmt*MMQ_FEW_NROWS*mmq_few_sram_stride(type, mmq_few_nunits_stage(type, ncg, nmt));
}

// With Q6_K and Q8_0 two CUDA blocks with 32-row tiles would not fit into an SM.
static constexpr __host__ __device__ bool mmq_few_tall_supported(const ggml_type type) {
    return type == GGML_TYPE_Q3_K || type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K ||
           type == GGML_TYPE_IQ3_S || type == GGML_TYPE_IQ4_NL || type == GGML_TYPE_IQ4_XS;
}

// MUL_MAT_ID with fewer than 2 units of 256 values per warp: the warps split the rows, each warp has its own 16 rows.
static constexpr __host__ __device__ bool mmq_few_ksplit(const bool has_ids, const int nunits) {
    return !has_ids || nunits >= 2*MMQ_FEW_NWARPS;
}

#if defined(CP_ASYNC_AVAILABLE)

// The src1 fragments of a lane for one unit of 256 values, see quantize.cuh.
struct mmq_few_y_frag {
    int4 qs[4];
    int4 ds[4];
};

static __device__ __forceinline__ void mmq_few_load_y(mmq_few_y_frag & yf, const int4 * y) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        yf.qs[j] = y[j*WARP_SIZE + threadIdx.x];
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        yf.ds[j] = y[4*WARP_SIZE + 4*(threadIdx.x % 4) + j];
    }
}

static __device__ __forceinline__ int2 mmq_few_get_b(const mmq_few_y_frag & yf, const int s) {
    const int4 q = yf.qs[s/2];
    return s % 2 == 0 ? make_int2(q.x, q.y) : make_int2(q.z, q.w);
}

template <mmq_q8_1_ds_layout ds_layout>
static __device__ __forceinline__ float2 mmq_few_get_d(const mmq_few_y_frag & yf, const int s) {
    if constexpr (ds_layout == MMQ_Q8_1_DS_LAYOUT_D4) {
        const float * d = (const float *) &yf.ds[s/2];
        return make_float2(d[2*(s % 2) + 0], d[2*(s % 2) + 1]);
    } else {
        const half2 * ds = (const half2 *) &yf.ds[s/2];
        return make_float2(__low2float(ds[2*(s % 2) + 0]), __low2float(ds[2*(s % 2) + 1]));
    }
}

static __device__ __forceinline__ float2 mmq_few_get_sum(const mmq_few_y_frag & yf, const int s) {
    const half2 * ds = (const half2 *) &yf.ds[s/2];
    return make_float2(__high2float(ds[2*(s % 2) + 0]), __high2float(ds[2*(s % 2) + 1]));
}

// C.x[0], C.x[1]: row g, columns 2t and 2t+1. C.x[2], C.x[3]: row g+8.
static __device__ __forceinline__ tile<16, 8, int> mmq_few_mma(const int (&lo)[2], const int (&hi)[2], const int2 b) {
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

static __device__ __forceinline__ tile<16, 8, int> mmq_few_mma_k16(const int (&a)[2], const int b) {
    tile<16, 4, int> A;
    A.x[0] = a[0];
    A.x[1] = a[1];
    tile<8, 4, int> B;
    B.x[0] = b;
    tile<16, 8, int> C;
    mma(C, A, B);
    return C;
}

// sum_d += d * (C * ls) of sub-block s for all column groups, ls are the src0 row scales
template <ggml_type type, int ncg, typename scale_t>
static __device__ __forceinline__ void mmq_few_mma_add(
        const int (&lo)[2], const int (&hi)[2], const scale_t (&ls)[2], const mmq_few_y_frag (&yf)[ncg], const int s, float (&sum_d)[ncg][4]) {
#pragma unroll
    for (int c = 0; c < ncg; ++c) {
        const tile<16, 8, int> C = mmq_few_mma(lo, hi, mmq_few_get_b(yf[c], s));

        const float2 d = mmq_few_get_d<mmq_few_ds_layout(type)>(yf[c], s);
        sum_d[c][0] += d.x * float(C.x[0]*ls[0]);
        sum_d[c][1] += d.y * float(C.x[1]*ls[0]);
        sum_d[c][2] += d.x * float(C.x[2]*ls[1]);
        sum_d[c][3] += d.y * float(C.x[3]*ls[1]);
    }
}

// Same with one k=16 MMA per half of sub-block s, the int8 scales of the halves are bytes 0 and 1 of sc[r].
template <ggml_type type, int ncg>
static __device__ __forceinline__ void mmq_few_mma_k16_add(
        const int (&lo)[2], const int (&hi)[2], const int (&sc)[2], const mmq_few_y_frag (&yf)[ncg], const int s, float (&sum_d)[ncg][4]) {
    const int sc_lo[2] = {(int8_t) sc[0], (int8_t) sc[1]};
    const int sc_hi[2] = {(int8_t) (sc[0] >> 8), (int8_t) (sc[1] >> 8)};
#pragma unroll
    for (int c = 0; c < ncg; ++c) {
        const int2 b = mmq_few_get_b(yf[c], s);
        const tile<16, 8, int> C_lo = mmq_few_mma_k16(lo, b.x);
        const tile<16, 8, int> C_hi = mmq_few_mma_k16(hi, b.y);

        const float2 d = mmq_few_get_d<mmq_few_ds_layout(type)>(yf[c], s);
        sum_d[c][0] += d.x * float(C_lo.x[0]*sc_lo[0] + C_hi.x[0]*sc_hi[0]);
        sum_d[c][1] += d.y * float(C_lo.x[1]*sc_lo[0] + C_hi.x[1]*sc_hi[0]);
        sum_d[c][2] += d.x * float(C_lo.x[2]*sc_lo[1] + C_hi.x[2]*sc_hi[1]);
        sum_d[c][3] += d.y * float(C_lo.x[3]*sc_lo[1] + C_hi.x[3]*sc_hi[1]);
    }
}

template <int ncg>
static __device__ __forceinline__ void mmq_few_add_scaled(float (&sum)[ncg][4], const float (&d)[2], const float (&sum_d)[ncg][4]) {
#pragma unroll
    for (int c = 0; c < ncg; ++c) {
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            sum[c][l] += d[l/2]*sum_d[c][l];
        }
    }
}

// vec_dot: one unit of 256 values of the 16 rows of a warp, x[0] and x[1] are the rows g and g+8 of lane (g, t).
// The lane decodes the values [4t, 4t + 4) and [16 + 4t, 16 + 4t + 4) of each sub-block, its A fragments.

template <ggml_type type, int ncg>
static __device__ __forceinline__ void mmq_few_vec_dot_q45_K(const char * const (&x)[2], const mmq_few_y_frag (&yf)[ncg], float (&sum)[ncg][4]) {
    using block_t = std::conditional_t<type == GGML_TYPE_Q4_K, block_q4_K, block_q5_K>;
    const int t = threadIdx.x % 4;

    int   qs[2][4][2];
    [[maybe_unused]] int qh[2][2]; // Q5_K: bit s holds the high bit of sub-block s
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
        const int4 hdr = *(const int4 *) bx; // dm and the 12 scale bytes
        const int scales[3] = {hdr.y, hdr.z, hdr.w};
        sc[r][0] = unpack_scales_q45_K(scales, 0);
        sc[r][1] = unpack_scales_q45_K(scales, 1);
        mn[r][0] = unpack_scales_q45_K(scales, 2);
        mn[r][1] = unpack_scales_q45_K(scales, 3);
        const float2 dm = __half22float2(bx->dm);
        d[r]    = dm.x;
        dmin[r] = dm.y;
    }

    float sum_d[ncg][4] = {{0.0f}};
    float sum_m[ncg][4] = {{0.0f}};
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
        mmq_few_mma_add<type>(lo, hi, ls, yf, s, sum_d);
#pragma unroll
        for (int c = 0; c < ncg; ++c) {
            const float2 sm = mmq_few_get_sum(yf[c], s);
            sum_m[c][0] += m0 * sm.x;
            sum_m[c][1] += m0 * sm.y;
            sum_m[c][2] += m1 * sm.x;
            sum_m[c][3] += m1 * sm.y;
        }
    }
#pragma unroll
    for (int c = 0; c < ncg; ++c) {
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            sum[c][l] += d[l/2]*sum_d[c][l] - dmin[l/2]*sum_m[c][l];
        }
    }
}

// Q6_K and Q3_K have a scale per 16 values: one k=16 MMA per half of a sub-block.
template <int ncg>
static __device__ __forceinline__ void mmq_few_vec_dot_q6_K(const char * const (&x)[2], const mmq_few_y_frag (&yf)[ncg], float (&sum)[ncg][4]) {
    const int t = threadIdx.x % 4;

    float d[2];
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        d[r] = __half2float(((const block_q6_K *) x[r])->d);
    }

    float sum_d[ncg][4] = {{0.0f}};
#pragma unroll
    for (int n = 0; n < 2; ++n) {
        int ql[2][2][2]; // [row][ql offset 0 or 32][lo or hi]
        int qh[2][2];
        int sc[2][2];
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            const block_q6_K * bx = (const block_q6_K *) x[r];
#pragma unroll
            for (int k = 0; k < 2; ++k) {
                ql[r][k][0] = get_int_b2(bx->ql + 64*n + 32*k,      t);
                ql[r][k][1] = get_int_b2(bx->ql + 64*n + 32*k + 16, t);
            }
            qh[r][0] = get_int_b2(bx->qh + 32*n,      t);
            qh[r][1] = get_int_b2(bx->qh + 32*n + 16, t);
            sc[r][0] = get_int_b2(bx->scales + 8*n, 0);
            sc[r][1] = get_int_b2(bx->scales + 8*n, 1);
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int s = 4*n + j;
            int lo[2];
            int hi[2];
#pragma unroll
            for (int r = 0; r < 2; ++r) {
                lo[r] = __vsub4(((ql[r][j % 2][0] >> (4*(j/2))) & 0x0F0F0F0F) | (((qh[r][0] >> (2*j)) & 0x03030303) << 4), 0x20202020);
                hi[r] = __vsub4(((ql[r][j % 2][1] >> (4*(j/2))) & 0x0F0F0F0F) | (((qh[r][1] >> (2*j)) & 0x03030303) << 4), 0x20202020);
            }
            const int sc_s[2] = {sc[0][j/2] >> (16*(j % 2)), sc[1][j/2] >> (16*(j % 2))}; // scales 2s and 2s+1
            mmq_few_mma_k16_add<GGML_TYPE_Q6_K>(lo, hi, sc_s, yf, s, sum_d);
        }
    }
    mmq_few_add_scaled(sum, d, sum_d);
}

template <int ncg>
static __device__ __forceinline__ void mmq_few_vec_dot_q3_K(const char * const (&x)[2], const mmq_few_y_frag (&yf)[ncg], float (&sum)[ncg][4]) {
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
            sc[r][ksc] = unpack_scales_q3_K(bx->scales, ksc);
        }
        d[r] = __half2float(bx->d);
    }

    float sum_d[ncg][4] = {{0.0f}};
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
            mmq_few_mma_k16_add<GGML_TYPE_Q3_K>(lo, hi, sc_s, yf, s, sum_d);
        }
    }
    mmq_few_add_scaled(sum, d, sum_d);
}

template <int ncg>
static __device__ __forceinline__ void mmq_few_vec_dot_iq4_xs(const char * const (&x)[2], const mmq_few_y_frag (&yf)[ncg], float (&sum)[ncg][4]) {
    const int t = threadIdx.x % 4;

    float d[2];
    int   scales_h[2];
    int   scales_l[2];
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        const block_iq4_xs * bx = (const block_iq4_xs *) x[r];
        d[r]        = __half2float(bx->d);
        scales_h[r] = bx->scales_h;
        scales_l[r] = get_int_b4(bx->scales_l, 0);
    }

    float sum_d[ncg][4] = {{0.0f}};
#pragma unroll
    for (int s = 0; s < 8; ++s) {
        int lo[2];
        int hi[2];
        int ls[2];
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            const block_iq4_xs * bx = (const block_iq4_xs *) x[r];
            const int2 v = get_int_from_table_16(get_int_b4(bx->qs + 16*s, t), kvalues_iq4nl);
            lo[r] = v.x;
            hi[r] = v.y;
            ls[r] = (((scales_l[r] >> (4*s)) & 0x0F) | (((scales_h[r] >> (2*s)) & 0x03) << 4)) - 32;
        }
        mmq_few_mma_add<GGML_TYPE_IQ4_XS>(lo, hi, ls, yf, s, sum_d);
    }
    mmq_few_add_scaled(sum, d, sum_d);
}

// Q8_0 and IQ4_NL have a block with its own scale per 32 values.
template <ggml_type type, int ncg>
static __device__ __forceinline__ void mmq_few_vec_dot_q8_0_iq4_nl(const char * const (&x)[2], const mmq_few_y_frag (&yf)[ncg], float (&sum)[ncg][4]) {
    const int t = threadIdx.x % 4;

#pragma unroll
    for (int s = 0; s < 8; ++s) {
        int   lo[2];
        int   hi[2];
        float d[2];
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            if constexpr (type == GGML_TYPE_Q8_0) {
                const block_q8_0 * bx = (const block_q8_0 *) x[r] + s;
                lo[r] = get_int_b2(bx->qs,      t);
                hi[r] = get_int_b2(bx->qs + 16, t);
                d[r]  = __half2float(bx->d);
            } else {
                const block_iq4_nl * bx = (const block_iq4_nl *) x[r] + s;
                const int2 v = get_int_from_table_16(get_int_b2(bx->qs, t), kvalues_iq4nl);
                lo[r] = v.x;
                hi[r] = v.y;
                d[r]  = __half2float(bx->d);
            }
        }
        mmq_few_mma_add<type>(lo, hi, d, yf, s, sum);
    }
}

template <int ncg>
static __device__ __forceinline__ void mmq_few_vec_dot_iq3_s(const char * const (&x)[2], const mmq_few_y_frag (&yf)[ncg], float (&sum)[ncg][4]) {
    const int t = threadIdx.x % 4;

    float d[2];
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        d[r] = __half2float(((const block_iq3_s *) x[r])->d);
    }

    float sum_d[ncg][4] = {{0.0f}};
#pragma unroll
    for (int s = 0; s < 8; ++s) {
        int lo[2];
        int hi[2];
        int ls[2];
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            const block_iq3_s * bx = (const block_iq3_s *) x[r];
            const int qh = bx->qh[s];
            // the values [4k, 4k + 4) of a sub-block come from grid entry k, with the signs in bits 4*(k%2) of byte k/2
            const int k_lo = t;
            const int k_hi = 4 + t;
            const int grid_lo = iq3s_grid[bx->qs[8*s + k_lo] | (((qh >> k_lo) & 1) << 8)];
            const int grid_hi = iq3s_grid[bx->qs[8*s + k_hi] | (((qh >> k_hi) & 1) << 8)];
            const int sign_lo = (bx->signs[4*s + k_lo/2] >> (4*(k_lo % 2))) & 0x0F;
            const int sign_hi = (bx->signs[4*s + k_hi/2] >> (4*(k_hi % 2))) & 0x0F;
            const int mask_lo = __vcmpne4(((sign_lo & 0x03) << 7) | ((sign_lo & 0x0C) << 21), 0x00000000);
            const int mask_hi = __vcmpne4(((sign_hi & 0x03) << 7) | ((sign_hi & 0x0C) << 21), 0x00000000);
            lo[r] = __vsub4(grid_lo ^ mask_lo, mask_lo);
            hi[r] = __vsub4(grid_hi ^ mask_hi, mask_hi);
            ls[r] = 1 + 2*((bx->scales[s/2] >> (4*(s % 2))) & 0x0F);
        }
        mmq_few_mma_add<GGML_TYPE_IQ3_S>(lo, hi, ls, yf, s, sum_d);
    }
    mmq_few_add_scaled(sum, d, sum_d);
}

template <ggml_type type, int ncg>
static __device__ __forceinline__ void mmq_few_vec_dot(const char * const (&x)[2], const mmq_few_y_frag (&yf)[ncg], float (&sum)[ncg][4]) {
    if constexpr (type == GGML_TYPE_Q8_0 || type == GGML_TYPE_IQ4_NL) {
        mmq_few_vec_dot_q8_0_iq4_nl<type, ncg>(x, yf, sum);
    } else if constexpr (type == GGML_TYPE_Q3_K) {
        mmq_few_vec_dot_q3_K<ncg>(x, yf, sum);
    } else if constexpr (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K) {
        mmq_few_vec_dot_q45_K<type, ncg>(x, yf, sum);
    } else if constexpr (type == GGML_TYPE_Q6_K) {
        mmq_few_vec_dot_q6_K<ncg>(x, yf, sum);
    } else if constexpr (type == GGML_TYPE_IQ3_S) {
        mmq_few_vec_dot_iq3_s<ncg>(x, yf, sum);
    } else if constexpr (type == GGML_TYPE_IQ4_XS) {
        mmq_few_vec_dot_iq4_xs<ncg>(x, yf, sum);
    } else {
        static_assert(type == GGML_TYPE_COUNT, "unsupported type");
    }
}

#endif // defined(CP_ASYNC_AVAILABLE)

// has_ids: MUL_MAT_ID, blockIdx.y is a pair of quantize_mmq_few_id_q8_1, its expert is the src0 channel and dst dims 1 and 2 are slot and token.
// blk is the block's blockIdx.x in a launch of one matrix: tile blk/csize, rank blk%csize of a cluster that splits its K.
template <ggml_type type, int ncg, int nmt, bool has_ids>
static __device__ __forceinline__ void mul_mat_q_few_impl(
        const char * x, const int4 * y_ptr, float * dst,
        const int nunits, const int nrows_x, const int ncols_dst, const int64_t stride_row_x, const int stride_col_dst,
        const int channel_ratio, const int64_t stride_channel_x, const int stride_channel_dst,
        const int sample_ratio, const int64_t stride_sample_x, const int stride_sample_dst, const bool x_is_constant, const int * pairs,
        const int blk, const int csize) {
#if defined(CP_ASYNC_AVAILABLE)
    constexpr int nwarps    = MMQ_FEW_NWARPS;
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    static_assert(warp_size == WARP_SIZE, "unsupported warp size");
    static_assert(!has_ids || (ncg == 1 && nmt == 1), "unsupported MUL_MAT_ID config");

    constexpr int  unit_size    = mmq_few_unit_size(type);
    constexpr int  nrows        = nmt*MMQ_FEW_NROWS;
    constexpr int  nunits_stage = mmq_few_nunits_stage(type, ncg, nmt);
    constexpr int  nstages      = mmq_few_nstages(type);
    constexpr bool unaligned    = mmq_few_unaligned(type, nunits_stage);
    constexpr int  nchunks      = mmq_few_nchunks(type, nunits_stage);
    constexpr int  sram_stride  = mmq_few_sram_stride(type, nunits_stage);
    constexpr int  nbytes_stage = nrows*sram_stride;
    constexpr int  ncopies      = nrows*nchunks;
    static_assert(unit_size > 0, "unsupported type");

    extern __shared__ char data_mmq_few[];
    char * ring = data_mmq_few + threadIdx.y*nstages*nbytes_stage;

    // the blocks of a cluster split K of one row tile: warp gw of the cluster takes every (csize*nwarps)-th stage
    const bool ksplit   = mmq_few_ksplit(has_ids, nunits);
    const int  rank     = blk % csize;
    const int  gw       = rank*nwarps + threadIdx.y;
    const int  row0_blk = blk/csize*(ksplit ? nrows : nwarps*nrows);
    const int  row0     = ksplit ? row0_blk : row0_blk + threadIdx.y*nrows;
    const int  iu_begin = ksplit ? nunits_stage*gw : 0;
    const int  iu_step  = ksplit ? nunits_stage*nwarps*csize : nunits_stage;

    const int channel_dst = blockIdx.y;
    const int sample_dst  = blockIdx.z;
    const int nchannels   = gridDim.y;

    int ncols = ncols_dst;
    [[maybe_unused]] const int * pair = nullptr;
    if constexpr (has_ids) {
        ggml_cuda_pdl_sync(); // the pairs come from the preceding kernel
        pair  = pairs + channel_dst*MMQ_FEW_PAIR_INTS;
        ncols = pair[1];
        if (ncols == 0) {
            return;
        }
        x += int64_t(pair[0])*stride_channel_x;
    } else {
        x   += int64_t(sample_dst/sample_ratio)*stride_sample_x + int64_t(channel_dst/channel_ratio)*stride_channel_x;
        dst += int64_t(sample_dst)*stride_sample_dst + int64_t(channel_dst)*stride_channel_dst;
    }
    const int4 * GGML_CUDA_RESTRICT y = y_ptr + (int64_t(sample_dst)*nchannels + channel_dst)*nunits*ncg*MMQ_FEW_Q8_1_INT4;

    // No bounds check for the units, the gate makes nunits a multiple of nunits_stage.
    // The extra bytes of unaligned data are inside the allocation, which ggml aligns and pads to 128 bytes.
    auto load_stage = [&](const int stage, const int iu0) {
        const unsigned int dst_stage = ggml_cuda_cvta_generic_to_shared(ring + stage*nbytes_stage);
#pragma unroll
        for (int c0 = 0; c0 < ncopies; c0 += warp_size) {
            const int c  = c0 + threadIdx.x;
            const int i  = c / nchunks;
            const int ic = c % nchunks;
            if (ncopies % warp_size == 0 || c < ncopies) {
                const int row = min(row0 + i, nrows_x - 1);
                const char * src = x + int64_t(row)*stride_row_x + int64_t(iu0)*unit_size;
                if constexpr (unaligned) {
                    const int off = (uintptr_t) src % 16;
                    if (16*ic < off + nunits_stage*unit_size) {
                        cp_async_cg_16<0>(dst_stage + i*sram_stride + 16*ic, src - off + 16*ic);
                    }
                } else {
                    cp_async_cg_16<0>(dst_stage + i*sram_stride + 16*ic, src + 16*ic);
                }
            }
        }
    };

    // offsets of the rows of this lane from their 16 byte boundary
    int off_row[2*nmt] = {0};
    if constexpr (unaligned) {
#pragma unroll
        for (int r = 0; r < 2*nmt; ++r) {
            const int row = min(row0 + int(threadIdx.x)/4 + 8*r, nrows_x - 1);
            off_row[r] = (uintptr_t) (x + int64_t(row)*stride_row_x) % 16;
        }
    }

    // Only weights, which no preceding kernel writes, are read before the PDL wait.
    if (!x_is_constant) {
        ggml_cuda_pdl_sync();
    }

#if __CUDA_ARCH__ == GGML_CUDA_CC_GB10
    // DGX Spark DRAM serves the short row pieces of the stage copies slowly, so the rows of the block are first prefetched into L2 in address order.
    {
        const int     tid       = threadIdx.y*warp_size + threadIdx.x;
        const int     nrows_blk = min(ksplit ? nrows : nwarps*nrows, nrows_x - row0_blk);
        const int64_t row_bytes = int64_t(nunits)*unit_size;
        const bool    contig    = stride_row_x == row_bytes;
        for (int ir = 0; ir < (contig ? 1 : nrows_blk); ++ir) {
            const uintptr_t p0 = (uintptr_t) (x + int64_t(row0_blk + ir)*stride_row_x);
            const uintptr_t p1 = p0 + (contig ? nrows_blk*row_bytes : row_bytes);
            for (uintptr_t p = (p0 & ~uintptr_t(127)) + 128*tid; p < p1; p += 128*nwarps*warp_size) {
                ggml_cuda_prefetch_l2((const void *) p);
            }
        }
    }
#endif // __CUDA_ARCH__ == GGML_CUDA_CC_GB10

#pragma unroll
    for (int stage = 0; stage < nstages - 1; ++stage) {
        const int iu0 = iu_begin + stage*iu_step;
        if (iu0 < nunits) {
            load_stage(stage, iu0);
        }
        cp_async_commit_group();
    }
    if (x_is_constant) {
        ggml_cuda_pdl_sync();
    }

    auto load_y = [&](mmq_few_y_frag (&yf)[ncg], const int iu) {
#pragma unroll
        for (int c = 0; c < ncg; ++c) {
            mmq_few_load_y(yf[c], y + (int64_t(iu)*ncg + c)*MMQ_FEW_Q8_1_INT4);
        }
    };

    // The next src1 fragments are loaded right after the MMAs that use the current ones, an earlier load is slower.
    mmq_few_y_frag yf[nunits_stage][ncg];
    if (iu_begin < nunits) {
#pragma unroll
        for (int k = 0; k < nunits_stage; ++k) {
            load_y(yf[k], iu_begin + k);
        }
    }

    float sum[nmt][ncg][4] = {{{0.0f}}};

    for (int iter = 0, iu0 = iu_begin; iu0 < nunits; ++iter, iu0 += iu_step) {
        const int iu_load = iu0 + (nstages - 1)*iu_step;
        if (iu_load < nunits) {
            load_stage((iter + nstages - 1) % nstages, iu_load);
        }
        cp_async_commit_group();

        const bool has_next = iu0 + iu_step < nunits;

        cp_async_wait_group<nstages - 1>();
        ggml_cuda_syncwarp();

        const char * xs = ring + (iter % nstages)*nbytes_stage;
        const int off_stage = unaligned ? (iu0*unit_size) % 16 : 0;
#pragma unroll
        for (int k = 0; k < nunits_stage; ++k) {
#pragma unroll
            for (int m = 0; m < nmt; ++m) {
                const char * const xr[2] = {
                    xs + (MMQ_FEW_NROWS*m + threadIdx.x/4 + 0)*sram_stride + (off_row[2*m + 0] + off_stage) % 16 + k*unit_size,
                    xs + (MMQ_FEW_NROWS*m + threadIdx.x/4 + 8)*sram_stride + (off_row[2*m + 1] + off_stage) % 16 + k*unit_size,
                };
                mmq_few_vec_dot<type, ncg>(xr, yf[k], sum[m]);
            }
            if (has_next) {
                load_y(yf[k], iu0 + iu_step + k);
            }
        }

        ggml_cuda_syncwarp(); // the stage is overwritten in the next iteration
    }
    cp_async_wait_all();

    ggml_cuda_pdl_lc();

    // reduce the sums over the warps as one flat array, then over the blocks of a cluster
    if (ksplit) {
        constexpr int nsum = nmt*ncg*4;
        float * sum_flat = &sum[0][0][0];

        float * buf_sum = (float *) data_mmq_few;
        __syncthreads();
        if (threadIdx.y > 0) {
#pragma unroll
            for (int i = 0; i < nsum; ++i) {
                buf_sum[((threadIdx.y - 1)*nsum + i)*warp_size + threadIdx.x] = sum_flat[i];
            }
        }
        __syncthreads();
        if (threadIdx.y == 0) {
#pragma unroll
            for (int w = 0; w < nwarps - 1; ++w) {
#pragma unroll
                for (int i = 0; i < nsum; ++i) {
                    sum_flat[i] += buf_sum[(w*nsum + i)*warp_size + threadIdx.x];
                }
            }
        }
#if __CUDA_ARCH__ >= GGML_CUDA_CC_HOPPER
        if (csize > 1) {
            // each lane overwrites only the sums it has read
            if (threadIdx.y == 0) {
#pragma unroll
                for (int i = 0; i < nsum; ++i) {
                    buf_sum[i*warp_size + threadIdx.x] = sum_flat[i];
                }
            }
            cooperative_groups::cluster_group cluster = cooperative_groups::this_cluster();
            cluster.sync();
            if (rank == 0 && threadIdx.y == 0) {
                for (int r = 1; r < csize; ++r) {
                    const float * buf_r = cluster.map_shared_rank(buf_sum, r);
#pragma unroll
                    for (int i = 0; i < nsum; ++i) {
                        sum_flat[i] += buf_r[i*warp_size + threadIdx.x];
                    }
                }
            }
            cluster.sync(); // the other blocks keep their shared memory until the first has read it
        }
#endif // __CUDA_ARCH__ >= GGML_CUDA_CC_HOPPER
        if (threadIdx.y > 0 || rank > 0) {
            return;
        }
    }

#pragma unroll
    for (int m = 0; m < nmt; ++m) {
#pragma unroll
        for (int c = 0; c < ncg; ++c) {
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                const int i = row0 + MMQ_FEW_NROWS*m + tile<16, 8, int>::get_i(l);
                const int j = c*MMQ_FEW_NCOLS + tile<16, 8, int>::get_j(l);
                if (i < nrows_x && j < ncols) {
                    if constexpr (has_ids) {
                        const int e = pair[2 + j]; // token*n_expert_used + slot
                        dst[int64_t(e / ncols_dst)*stride_channel_dst + int64_t(e % ncols_dst)*stride_col_dst + i] = sum[m][c][l];
                    } else {
                        dst[int64_t(j)*stride_col_dst + i] = sum[m][c][l];
                    }
                }
            }
        }
    }
#else
    GGML_UNUSED_VARS(x, y_ptr, dst, nunits, nrows_x, ncols_dst, stride_row_x, stride_col_dst,
        channel_ratio, stride_channel_x, stride_channel_dst,
        sample_ratio, stride_sample_x, stride_sample_dst, x_is_constant, pairs, blk, csize);
    NO_DEVICE_CODE;
#endif // defined(CP_ASYNC_AVAILABLE)
}

template <ggml_type type, int ncg, int nmt, bool has_ids>
__launch_bounds__(MMQ_FEW_NWARPS*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mul_mat_q_few(
        const char * x, const int4 * y_ptr, float * dst,
        const int nunits, const int nrows_x, const int ncols_dst, const int64_t stride_row_x, const int stride_col_dst,
        const int channel_ratio, const int64_t stride_channel_x, const int stride_channel_dst,
        const int sample_ratio, const int64_t stride_sample_x, const int stride_sample_dst, const bool x_is_constant, const int * pairs,
        const int csize) {
    mul_mat_q_few_impl<type, ncg, nmt, has_ids>(x, y_ptr, dst, nunits, nrows_x, ncols_dst, stride_row_x, stride_col_dst,
        channel_ratio, stride_channel_x, stride_channel_dst, sample_ratio, stride_sample_x, stride_sample_dst, x_is_constant, pairs,
        blockIdx.x, csize);
}

// The matrices of a group launch (ggml_cuda_mul_mat_q_few_group): the blocks of each one follow those of the one before.
struct mmq_few_group {
    const char * x[MMQ_FEW_GROUP_MAX];
    float      * dst[MMQ_FEW_GROUP_MAX];
    int64_t      stride_row_x[MMQ_FEW_GROUP_MAX];
    int          nrows_x[MMQ_FEW_GROUP_MAX];
    int          stride_col_dst[MMQ_FEW_GROUP_MAX];
    int          blk_end[MMQ_FEW_GROUP_MAX]; // one past the last block of each matrix, the grid size past the last matrix
};

// Each block runs the tile of its matrix that the matrix's own launch runs at that blockIdx.x, in clusters of the csize
// each matrix takes alone, so each output is the same bit for bit. Constant indices keep the group in the parameter space.
template <ggml_type type, int ncg, int nmt>
__launch_bounds__(MMQ_FEW_NWARPS*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mul_mat_q_few_group(const mmq_few_group group, const int4 * y_ptr, const int nunits, const int ncols_dst,
        const bool x_is_constant, const int csize) {
    const char * x              = group.x[0];
    float      * dst            = group.dst[0];
    int64_t      stride_row_x   = group.stride_row_x[0];
    int          nrows_x        = group.nrows_x[0];
    int          stride_col_dst = group.stride_col_dst[0];
    int          blk0           = 0;
#pragma unroll
    for (int m = 1; m < MMQ_FEW_GROUP_MAX; ++m) {
        if ((int) blockIdx.x >= group.blk_end[m - 1]) {
            x              = group.x[m];
            dst            = group.dst[m];
            stride_row_x   = group.stride_row_x[m];
            nrows_x        = group.nrows_x[m];
            stride_col_dst = group.stride_col_dst[m];
            blk0           = group.blk_end[m - 1];
        }
    }
    mul_mat_q_few_impl<type, ncg, nmt, false>(x, y_ptr, dst, nunits, nrows_x, ncols_dst, stride_row_x, stride_col_dst,
        1, 0, 0, 1, 0, 0, x_is_constant, nullptr, (int) blockIdx.x - blk0, csize);
}

// Blocks of a cluster that split K of one row tile (nstages_row ring stages along a row, ntiles tiles over all channels).
// A warp's stages run one after the other, each a load and a chain of MMAs that only the warps beside it hide: at 23
// columns the 4 warps of a block took the 18 stages of a 4608 row in 5 steps, and Inkling 2's k (256 rows, 16 tiles)
// took 4.7 us for 0.5 MB once its input was ready (RTX 5080). The cluster grows until its warps take a stage each, while
// all its blocks are resident at once (2 an SM at the kernels' 200-255 registers). Hopper and newer: cluster launches.
static int mmq_few_cluster_size(const int nstages_row, const int64_t ntiles) {
    static const bool legacy = ggml_env_switch("GGML_CUDA_MMQ_FEW_CLUSTER_LEGACY");
    const int device = ggml_cuda_get_device();
    const int cc     = ggml_cuda_info().devices[device].cc;
    if (legacy || !GGML_CUDA_CC_IS_NVIDIA(cc) || ggml_cuda_highest_compiled_arch(cc) < GGML_CUDA_CC_HOPPER ||
            cc == GGML_CUDA_CC_GB10) {
        return 1;
    }
    const int64_t slots = 2*int64_t(ggml_cuda_info().devices[device].nsm);
    int csize = 1;
    while (csize < MMQ_FEW_MAX_CLUSTER && MMQ_FEW_NWARPS*csize < nstages_row && ntiles*(csize + 1) <= slots) {
        ++csize;
    }
    return csize;
}

// A launch in clusters of csize blocks, or a plain one for 1.
template <typename Kernel, typename... Args>
static void mmq_few_launch(Kernel kernel, const ggml_cuda_kernel_launch_params & params, const int csize, Args &&... args) {
    if (csize == 1) {
        ggml_cuda_kernel_launch(kernel, params, std::forward<Args>(args)...);
        return;
    }
    cudaLaunchAttribute attr[2];
    cudaLaunchConfig_t  cfg = {};
    cfg.gridDim          = params.block_nums;
    cfg.blockDim         = params.block_dims;
    cfg.dynamicSmemBytes = params.shmem;
    cfg.stream           = params.stream;
    cfg.attrs            = attr;
    attr[cfg.numAttrs].id               = cudaLaunchAttributeClusterDimension;
    attr[cfg.numAttrs].val.clusterDim.x = csize;
    attr[cfg.numAttrs].val.clusterDim.y = 1;
    attr[cfg.numAttrs].val.clusterDim.z = 1;
    ++cfg.numAttrs;
#if defined(GGML_CUDA_USE_PDL)
    if (ggml_cuda_kernel_can_use_pdl(reinterpret_cast<const void *>(kernel))) {
        attr[cfg.numAttrs].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attr[cfg.numAttrs].val.programmaticStreamSerializationAllowed = 1;
        ++cfg.numAttrs;
    }
#endif // defined(GGML_CUDA_USE_PDL)
    CUDA_CHECK(cudaLaunchKernelEx(&cfg, kernel, std::forward<Args>(args)...));
}

// The largest cluster of this kernel's blocks, with their dynamic shared memory, that one GPC can hold. The count of
// resident blocks above is over the whole GPU, and cudaLaunchKernelEx fails on a cluster that no GPC can place.
// Asked once a device for each kernel: the caller keeps the answers (every few-row kernel has the same pointer type).
template <typename kernel_t>
static int mmq_few_cluster_fit(kernel_t kernel, const dim3 block_dims, const int nbytes_shared, int (&fit)[GGML_CUDA_MAX_DEVICES]) {
    const int device = ggml_cuda_get_device();
    if (fit[device] == 0) {
        int csize = MMQ_FEW_MAX_CLUSTER;
        for (; csize > 1; --csize) {
            cudaLaunchAttribute attr;
            attr.id               = cudaLaunchAttributeClusterDimension;
            attr.val.clusterDim.x = csize;
            attr.val.clusterDim.y = 1;
            attr.val.clusterDim.z = 1;
            cudaLaunchConfig_t cfg = {};
            cfg.gridDim          = dim3(csize, 1, 1);
            cfg.blockDim         = block_dims;
            cfg.dynamicSmemBytes = nbytes_shared;
            cfg.attrs            = &attr;
            cfg.numAttrs         = 1;
            int nclusters = 0;
            CUDA_CHECK(cudaOccupancyMaxActiveClusters(&nclusters, kernel, &cfg));
            if (nclusters > 0) {
                break;
            }
        }
        fit[device] = csize;
    }
    return fit[device];
}

// ggml_is_constant (static in ggml.c) on the tensor that a view points to
static bool mmq_few_is_constant(const ggml_tensor * src0) {
    const ggml_tensor * x_root = src0->view_src ? src0->view_src : src0;
    return x_root->buffer && ggml_backend_buffer_get_usage(x_root->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
        !(x_root->flags & GGML_TENSOR_FLAG_PARAM);
}

// MUL_MAT_ID: grid y is the pairs, the kernel takes the expert from the pair and does not use the channel and sample ratios.
template <ggml_type type, int ncg, int nmt, bool has_ids = false>
static void launch_mul_mat_q_few(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
        const int4 * src1_q, cudaStream_t stream, const int * pairs = nullptr, const int npairs_max = 0) {
    GGML_TENSOR_BINARY_OP_LOCALS;

    const int nunits = ne00 / QK_K;
    GGML_ASSERT(nunits % mmq_few_nunits_stage(type, ncg, nmt) == 0);

    const int warp_size     = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const int nbytes_shared = mmq_few_nbytes_shared(type, ncg, nmt);
    CUDA_SET_SHARED_MEMORY_LIMIT((mul_mat_q_few<type, ncg, nmt, has_ids>), nbytes_shared);

    const bool x_is_constant = mmq_few_is_constant(src0);

    const int     nrows_blk = (mmq_few_ksplit(has_ids, nunits) ? 1 : MMQ_FEW_NWARPS)*nmt*MMQ_FEW_NROWS;
    const int64_t ntiles    = (ne01 + nrows_blk - 1) / nrows_blk;
    const dim3    block_dims(warp_size, MMQ_FEW_NWARPS, 1);
    auto kernel = mul_mat_q_few<type, ncg, nmt, has_ids>;
    // MUL_MAT_ID keeps a block a tile
    int csize = has_ids ? 1 : mmq_few_cluster_size(nunits / mmq_few_nunits_stage(type, ncg, nmt), ntiles*ne12*ne13);
    if (csize > 1) {
        static int fit[GGML_CUDA_MAX_DEVICES] = { 0 }; // 0: not asked yet
        csize = std::min(csize, mmq_few_cluster_fit(kernel, block_dims, nbytes_shared, fit));
    }

    const dim3 block_nums(ntiles*csize, has_ids ? npairs_max : ne12, ne13);
    const ggml_cuda_kernel_launch_params params(block_nums, block_dims, nbytes_shared, stream);
    mmq_few_launch(kernel, params, csize,
        (const char *) src0->data, src1_q, (float *) dst->data,
        nunits, (int) ne01, (int) ne1, (int64_t) nb01, (int) (nb1/sizeof(float)),
        (int) (ne12/ne02), (int64_t) nb02, (int) (nb2/sizeof(float)),
        (int) (ne13/ne03), (int64_t) nb03, (int) (nb3/sizeof(float)), x_is_constant, pairs, csize);
}

static int mmq_few_ncg(const int64_t ncols) {
    return (ncols + MMQ_FEW_NCOLS - 1) / MMQ_FEW_NCOLS;
}

// With 2 column groups, 32-row tiles halve the src1 loads per src0 byte. With 3 column groups they spill registers.
static int mmq_few_nmt(const ggml_type type, const int ncg, const int64_t nrows_x) {
    const int nsm = ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;
    return mmq_few_tall_supported(type) && ncg == 2 && nrows_x >= int64_t(nsm)*2*MMQ_FEW_NROWS ? 2 : 1;
}

template <ggml_type type>
static void launch_mul_mat_q_few_tall(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
        const int4 * src1_q, cudaStream_t stream) {
    if constexpr (mmq_few_tall_supported(type)) {
        if (mmq_few_nmt(type, 2, src0->ne[1]) == 2) {
            launch_mul_mat_q_few<type, 2, 2>(src0, src1, dst, src1_q, stream);
            return;
        }
    }
    launch_mul_mat_q_few<type, 2, 1>(src0, src1, dst, src1_q, stream);
}

template <ggml_type type>
static void launch_mul_mat_q_few_switch_ncg(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
        const int4 * src1_q, const int * pairs, const int npairs_max, const int ncg, cudaStream_t stream) {
    if (pairs) {
        launch_mul_mat_q_few<type, 1, 1, true>(src0, src1, dst, src1_q, stream, pairs, npairs_max);
        return;
    }
    switch (ncg) {
        case 1:
            launch_mul_mat_q_few<type, 1, 1>(src0, src1, dst, src1_q, stream);
            break;
        case 2:
            launch_mul_mat_q_few_tall<type>(src0, src1, dst, src1_q, stream);
            break;
        case 3:
            launch_mul_mat_q_few<type, 3, 1>(src0, src1, dst, src1_q, stream);
            break;
        case 4:
            launch_mul_mat_q_few<type, 4, 1>(src0, src1, dst, src1_q, stream);
            break;
        default:
            GGML_ABORT("fatal error");
    }
}

// K-quants only, the types of the dense weights the groups were made for: each type is one more kernel a tile shape
static constexpr bool mmq_few_group_type(const ggml_type type) {
    return type == GGML_TYPE_Q3_K || type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K || type == GGML_TYPE_Q6_K;
}

// the cluster launch_mul_mat_q_few gives a flat MUL_MAT of src0 alone at this tile shape, before mmq_few_cluster_fit
static int mmq_few_group_cluster_size(const ggml_tensor * src0, const int ncg, const int nmt) {
    const int nunits = (int) (src0->ne[0] / QK_K);
    return mmq_few_cluster_size(nunits / mmq_few_nunits_stage(src0->type, ncg, nmt), (src0->ne[1] + nmt*MMQ_FEW_NROWS - 1) / (nmt*MMQ_FEW_NROWS));
}

template <ggml_type type, int ncg, int nmt>
static void launch_mul_mat_q_few_group(ggml_tensor * const * dsts, const int n, const int4 * src1_q, cudaStream_t stream) {
    const int nunits = dsts[0]->src[0]->ne[0] / QK_K;
    GGML_ASSERT(nunits % mmq_few_nunits_stage(type, ncg, nmt) == 0);

    const int warp_size     = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const int nbytes_shared = mmq_few_nbytes_shared(type, ncg, nmt);
    CUDA_SET_SHARED_MEMORY_LIMIT((mul_mat_q_few_group<type, ncg, nmt>), nbytes_shared);

    // the cluster each matrix takes alone, the same for all of them (ggml_cuda_mmq_few_group_fits)
    const dim3 block_dims(warp_size, MMQ_FEW_NWARPS, 1);
    int csize = mmq_few_group_cluster_size(dsts[0]->src[0], ncg, nmt);
    if (csize > 1) {
        CUDA_SET_SHARED_MEMORY_LIMIT((mul_mat_q_few<type, ncg, nmt, false>), nbytes_shared);
        static int fit_one[GGML_CUDA_MAX_DEVICES]   = { 0 };
        static int fit_group[GGML_CUDA_MAX_DEVICES] = { 0 };
        csize = std::min(csize, mmq_few_cluster_fit(mul_mat_q_few<type, ncg, nmt, false>, block_dims, nbytes_shared, fit_one));
        if (mmq_few_cluster_fit(mul_mat_q_few_group<type, ncg, nmt>, block_dims, nbytes_shared, fit_group) < csize) {
            // a GPC cannot hold the group kernel's clusters of that size: a launch each, the same bit for bit
            for (int m = 0; m < n; ++m) {
                launch_mul_mat_q_few<type, ncg, nmt>(dsts[m]->src[0], dsts[m]->src[1], dsts[m], src1_q, stream);
            }
            return;
        }
    }

    mmq_few_group group = {};
    int  nblocks       = 0;
    bool x_is_constant = true;
    for (int m = 0; m < n; ++m) {
        const ggml_tensor * src0 = dsts[m]->src[0];
        group.x[m]              = (const char *) src0->data;
        group.dst[m]            = (float *) dsts[m]->data;
        group.stride_row_x[m]   = src0->nb[1];
        group.nrows_x[m]        = (int) src0->ne[1];
        group.stride_col_dst[m] = (int) (dsts[m]->nb[1]/sizeof(float));
        nblocks += (int) ((src0->ne[1] + nmt*MMQ_FEW_NROWS - 1) / (nmt*MMQ_FEW_NROWS))*csize;
        group.blk_end[m] = nblocks;
        x_is_constant = x_is_constant && mmq_few_is_constant(src0);
    }
    for (int m = n; m < MMQ_FEW_GROUP_MAX; ++m) {
        group.blk_end[m] = nblocks;
    }

    const dim3 block_nums(nblocks, 1, 1);
    const ggml_cuda_kernel_launch_params params(block_nums, block_dims, nbytes_shared, stream);
    mmq_few_launch(mul_mat_q_few_group<type, ncg, nmt>, params, csize, group, src1_q, nunits, (int) dsts[0]->ne[1], x_is_constant, csize);
}

template <ggml_type type>
static void launch_mul_mat_q_few_group_switch_ncg(ggml_tensor * const * dsts, const int n, const int4 * src1_q, const int ncg,
        const int nmt, cudaStream_t stream) {
    switch (ncg) {
        case 1:
            launch_mul_mat_q_few_group<type, 1, 1>(dsts, n, src1_q, stream);
            break;
        case 2:
            if constexpr (mmq_few_tall_supported(type)) {
                if (nmt == 2) {
                    launch_mul_mat_q_few_group<type, 2, 2>(dsts, n, src1_q, stream);
                    break;
                }
            }
            launch_mul_mat_q_few_group<type, 2, 1>(dsts, n, src1_q, stream);
            break;
        case 3:
            launch_mul_mat_q_few_group<type, 3, 1>(dsts, n, src1_q, stream);
            break;
        default:
            GGML_ABORT("fatal error");
    }
}

bool ggml_cuda_mmq_few_group_fits(const ggml_tensor * first, const ggml_tensor * mm) {
    const ggml_tensor * x0 = first->src[0];
    const ggml_tensor * x  = mm->src[0];
    const ggml_tensor * y  = first->src[1];
    const auto flat = [](const ggml_tensor * t) { return t->ne[2] == 1 && t->ne[3] == 1; };
    if (!mmq_few_group_type(x0->type) || x->type != x0->type || mm->src[1] != y || x->ne[0] != x0->ne[0] ||
            !flat(x) || !flat(y) || !flat(mm) || x->nb[0] != ggml_type_size(x->type) || mm->nb[0] != sizeof(float)) {
        return false;
    }
    // the tile shape and the cluster each would launch alone; 3 column groups at most, as the groups were made for
    const int ncg = mmq_few_ncg(y->ne[1]);
    const int nmt = mmq_few_nmt(x0->type, ncg, x0->ne[1]);
    return ncg <= 3 && mmq_few_nmt(x->type, ncg, x->ne[1]) == nmt &&
        mmq_few_group_cluster_size(x, ncg, nmt) == mmq_few_group_cluster_size(x0, ncg, nmt);
}

void ggml_cuda_mul_mat_q_few_group(ggml_backend_cuda_context & ctx, ggml_tensor * const * dsts, const int n,
        char * src1_q_shared, const bool quantize_src1) {
    const ggml_tensor * src0 = dsts[0]->src[0];
    const ggml_tensor * src1 = dsts[0]->src[1];
    GGML_ASSERT(n >= 1 && n <= MMQ_FEW_GROUP_MAX && src1_q_shared != nullptr);
    GGML_ASSERT(src1->type == GGML_TYPE_F32 && src1->nb[0] == sizeof(float) && src0->ne[0] % QK_K == 0);
    GGML_ASSERT(src1->ne[1] <= MMQ_FEW_MAX_NCG*MMQ_FEW_NCOLS);
    for (int m = 0; m < n; ++m) {
        GGML_ASSERT(ggml_cuda_mmq_few_group_fits(dsts[0], dsts[m]));
    }

    cudaStream_t stream = ctx.stream();
    if (quantize_src1) {
        quantize_mmq_few_q8_1_cuda((const float *) src1->data, (int4 *) src1_q_shared, mmq_few_ds_layout(src0->type), src1->ne[0],
            src1->nb[1]/sizeof(float), src1->nb[2]/sizeof(float), src1->nb[3]/sizeof(float), src1->ne[1], src1->ne[2], src1->ne[3], stream);
    }

    const int4 * src1_q = (const int4 *) src1_q_shared;
    const int    ncg    = mmq_few_ncg(src1->ne[1]);
    const int    nmt    = mmq_few_nmt(src0->type, ncg, src0->ne[1]);
    switch (src0->type) {
        case GGML_TYPE_Q3_K:
            launch_mul_mat_q_few_group_switch_ncg<GGML_TYPE_Q3_K>(dsts, n, src1_q, ncg, nmt, stream);
            break;
        case GGML_TYPE_Q4_K:
            launch_mul_mat_q_few_group_switch_ncg<GGML_TYPE_Q4_K>(dsts, n, src1_q, ncg, nmt, stream);
            break;
        case GGML_TYPE_Q5_K:
            launch_mul_mat_q_few_group_switch_ncg<GGML_TYPE_Q5_K>(dsts, n, src1_q, ncg, nmt, stream);
            break;
        case GGML_TYPE_Q6_K:
            launch_mul_mat_q_few_group_switch_ncg<GGML_TYPE_Q6_K>(dsts, n, src1_q, ncg, nmt, stream);
            break;
        default:
            GGML_ABORT("unsupported type for a few-row MMQ group");
    }
}

size_t ggml_cuda_mmq_few_shared_q8_size(const ggml_tensor * src0, const ggml_tensor * src1) {
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    const int nunits = src0->ne[0] / QK_K;
    const int ncg    = mmq_few_ncg(src1->ne[1]);
    return size_t(src1->ne[2])*src1->ne[3]*nunits*ncg*MMQ_FEW_Q8_1_INT4*sizeof(int4);
}

void ggml_cuda_mul_mat_q_few(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst,
        char * src1_q_shared, const bool quantize_src1) {
    GGML_TENSOR_BINARY_OP_LOCALS;

    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    GGML_ASSERT(nb00 == ggml_type_size(src0->type));
    GGML_ASSERT(nb10 == sizeof(float));
    GGML_ASSERT(nb0  == sizeof(float));
    GGML_ASSERT(ne00 % QK_K == 0);

    cudaStream_t stream = ctx.stream();

    const int nunits = ne00 / QK_K;
    const int ncg    = ids ? 1 : mmq_few_ncg(ne11);

    ggml_cuda_pool_alloc<int>  pairs(ctx.pool());
    ggml_cuda_pool_alloc<int4> src1_q(ctx.pool());
    int npairs_max = 0;
    if (ids) {
        GGML_ASSERT(src1_q_shared == nullptr); // a shared copy holds the MUL_MAT layout, the ids quantize writes pairs
        GGML_ASSERT(ids->type == GGML_TYPE_I32);
        GGML_ASSERT(ids->nb[0] == ggml_type_size(ids->type));
        GGML_ASSERT(ne1 == ids->ne[0] && ne13 == 1 && ne3 == 1);

        const int64_t n_expert_used = ids->ne[0];
        const int64_t n_tokens      = ids->ne[1];
        const int64_t nentries      = n_tokens*n_expert_used;
        const int64_t nused         = std::min(ne02, nentries);
        // an expert with c > 0 entries has ceil(c/MMQ_FEW_NCOLS) pairs
        npairs_max = (int) (nused + (nentries - nused)/MMQ_FEW_NCOLS);

        pairs.alloc(int64_t(npairs_max)*MMQ_FEW_PAIR_INTS);
        src1_q.alloc(int64_t(npairs_max)*nunits*MMQ_FEW_Q8_1_INT4);
        quantize_mmq_few_id_q8_1_cuda((const float *) src1->data, (const int32_t *) ids->data, src1_q.get(), pairs.get(), mmq_few_ds_layout(src0->type),
            ne00, nb11/sizeof(float), nb12/sizeof(float), ne11, ne02, n_expert_used, n_tokens, ids->nb[1]/sizeof(int32_t), npairs_max, stream);
    } else {
        GGML_ASSERT(ne11 <= MMQ_FEW_MAX_NCG*MMQ_FEW_NCOLS);
        GGML_ASSERT(ne12 % ne02 == 0 && ne13 % ne03 == 0);

        if (src1_q_shared != nullptr) {
            if (quantize_src1) {
                quantize_mmq_few_q8_1_cuda((const float *) src1->data, (int4 *) src1_q_shared, mmq_few_ds_layout(src0->type), ne10, nb11/sizeof(float), nb12/sizeof(float), nb13/sizeof(float),
                    ne11, ne12, ne13, stream);
            }
        } else {
            src1_q.alloc(int64_t(ne12)*ne13*nunits*ncg*MMQ_FEW_Q8_1_INT4);
            quantize_mmq_few_q8_1_cuda((const float *) src1->data, src1_q.get(), mmq_few_ds_layout(src0->type), ne10, nb11/sizeof(float), nb12/sizeof(float), nb13/sizeof(float),
                ne11, ne12, ne13, stream);
        }
    }

    const int4 * src1_q_data = src1_q_shared != nullptr ? (const int4 *) src1_q_shared : src1_q.get();

    switch (src0->type) {
        case GGML_TYPE_Q8_0:
            launch_mul_mat_q_few_switch_ncg<GGML_TYPE_Q8_0>(src0, src1, dst, src1_q_data, pairs.get(), npairs_max, ncg, stream);
            break;
        case GGML_TYPE_Q3_K:
            launch_mul_mat_q_few_switch_ncg<GGML_TYPE_Q3_K>(src0, src1, dst, src1_q_data, pairs.get(), npairs_max, ncg, stream);
            break;
        case GGML_TYPE_Q4_K:
            launch_mul_mat_q_few_switch_ncg<GGML_TYPE_Q4_K>(src0, src1, dst, src1_q_data, pairs.get(), npairs_max, ncg, stream);
            break;
        case GGML_TYPE_Q5_K:
            launch_mul_mat_q_few_switch_ncg<GGML_TYPE_Q5_K>(src0, src1, dst, src1_q_data, pairs.get(), npairs_max, ncg, stream);
            break;
        case GGML_TYPE_Q6_K:
            launch_mul_mat_q_few_switch_ncg<GGML_TYPE_Q6_K>(src0, src1, dst, src1_q_data, pairs.get(), npairs_max, ncg, stream);
            break;
        case GGML_TYPE_IQ3_S:
            launch_mul_mat_q_few_switch_ncg<GGML_TYPE_IQ3_S>(src0, src1, dst, src1_q_data, pairs.get(), npairs_max, ncg, stream);
            break;
        case GGML_TYPE_IQ4_NL:
            launch_mul_mat_q_few_switch_ncg<GGML_TYPE_IQ4_NL>(src0, src1, dst, src1_q_data, pairs.get(), npairs_max, ncg, stream);
            break;
        case GGML_TYPE_IQ4_XS:
            launch_mul_mat_q_few_switch_ncg<GGML_TYPE_IQ4_XS>(src0, src1, dst, src1_q_data, pairs.get(), npairs_max, ncg, stream);
            break;
        default:
            GGML_ABORT("unsupported type for few-row MMQ");
    }
}

void ggml_cuda_mul_mat_q_few(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst) {
    ggml_cuda_mul_mat_q_few(ctx, src0, src1, ids, dst, nullptr, false);
}

// Min. number of src1 columns (MUL_MAT_ID: tokens) for the few-row MMQ, 0 if it is not used for this type, GPU and row length.
static int ggml_cuda_mmq_few_min_ncols(const ggml_type type, const int cc, const int64_t ne00, const bool has_ids) {
#ifdef GGML_CUDA_FORCE_CUBLAS
    return 0;
#endif // GGML_CUDA_FORCE_CUBLAS

    // NVIDIA Ampere to Blackwell
    if (!ampere_mma_available(cc) || cc >= GGML_CUDA_CC_RUBIN) {
        return 0;
    }
    // on DGX Spark the kernel needs its L2 prefetch, which is only in the DGX Spark device code
    if (cc == GGML_CUDA_CC_GB10 && ggml_cuda_highest_compiled_arch(cc) != GGML_CUDA_CC_GB10) {
        return 0;
    }
    // with fewer than 6 units of 256 values per row the warps idle, MMVQ and MMQ are better there (8 units: see
    // ggml_cuda_should_use_mmq_few; MUL_MAT_ID splits the rows instead)
    if (mmq_few_unit_size(type) == 0 || (!has_ids && ne00 < 6*QK_K)) {
        return 0;
    }
    if (cc == GGML_CUDA_CC_GB10) { // tuned on DGX Spark GB10, also for MUL_MAT_ID
        return type == GGML_TYPE_IQ4_XS || type == GGML_TYPE_IQ4_NL ? 8 : 2;
    }
    if (has_ids) { // after MMVQ, which is faster up to its max. batch size
        return get_mmvq_mmid_max_batch(type, cc) + 1;
    }
    switch (type) { // all GPUs except DGX Spark, tuned on RTX 5090
        case GGML_TYPE_Q3_K:   return 3;
        case GGML_TYPE_Q5_K:   return 3;
        case GGML_TYPE_Q4_K:   return 4;
        case GGML_TYPE_IQ4_NL: return 4;
        case GGML_TYPE_Q6_K:   return 4;
        case GGML_TYPE_Q8_0:   return 5;
        case GGML_TYPE_IQ3_S:  return 5;
        case GGML_TYPE_IQ4_XS: return 6;
        default:               return 0;
    }
}

bool ggml_cuda_should_use_mmq_few(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, const int cc) {
    // GGML_CUDA_MMQ_FEW_LEGACY=1: these matmuls stay on MMVQ and MMQ, and keep the fusions that go with those, as
    // before upstream v1.0.0 (the fork's switch: upstream has none)
    static const bool legacy = ggml_env_switch("GGML_CUDA_MMQ_FEW_LEGACY");
    if (legacy) {
        return false;
    }
    const int64_t ncols     = ids ? ids->ne[1] : src1->ne[1];
    const int     min_ncols = ggml_cuda_mmq_few_min_ncols(src0->type, cc, src0->ne[0], ids != nullptr);
    // GGML_CUDA_MMQ_FEW_MAX_COLS: most columns a MUL_MAT takes, past them MMQ; MUL_MAT_ID takes 24 tokens.
    // GGML_CUDA_MMQ_FEW_CLUSTER_LEGACY=1: upstream's 24 columns and rows of 8 units or more, and no clusters
    static const bool legacy_cluster = ggml_env_switch("GGML_CUDA_MMQ_FEW_CLUSTER_LEGACY");
    static const int  max_ncols      = legacy_cluster ? std::min<int>(ggml_env_int("GGML_CUDA_MMQ_FEW_MAX_COLS"), 3*MMQ_FEW_NCOLS) :
                                                        (int) ggml_env_int("GGML_CUDA_MMQ_FEW_MAX_COLS");
    if (min_ncols == 0 || ncols < min_ncols || ncols > (ids ? 3*MMQ_FEW_NCOLS : max_ncols)) {
        return false;
    }
    if (!ids) {
        // Past 24 columns MMQ is faster once the tiles outnumber the blocks resident at once (2 an SM). RTX 5080 (84 SMs),
        // Q3_K-Q8_0 at 28-32 columns: 48 tiles (768 x 4608) 6.6-8.6 us against MMQ's 11.0-13.4, 256 (4096 x 14336) and 896
        // (14336 x 4096) 4-15 % slower than MMQ.
        const int64_t ntiles = (src0->ne[1] + MMQ_FEW_NROWS - 1) / MMQ_FEW_NROWS * src1->ne[2]*src1->ne[3];
        const int     nsm    = ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;
        if (ncols > 3*MMQ_FEW_NCOLS && ntiles > 2*int64_t(nsm)) {
            return false;
        }
        // Rows of 6-7 units leave a warp of 4 idle. Q4_K rows of 1536 on the RTX 5080: at 8-24 columns 256-4608 rows take
        // 1.9-7.5 us against MMQ's 5.0-9.0; 8960 rows 6.8-9.3 against 8.9-10.5 at 8-16 columns but 13.3 against 11.9 at 24
        // (past 4 tiles an SM); at 4 columns MMVQ is faster under 1536 rows.
        if (src0->ne[0] < 8*QK_K && (legacy_cluster || ncols < 8 || (ncols > 2*MMQ_FEW_NCOLS && ntiles > 4*int64_t(nsm)))) {
            return false;
        }
    }
    // MUL_MAT_ID: the pairs are built in shared memory
    if (ids && (2*src0->ne[2] + 2)*sizeof(int) > ggml_cuda_info().devices[ggml_cuda_get_device()].smpb) {
        return false;
    }
    // the row must hold a whole number of ring stages of the launched kernel
    const int ncg          = ids ? 1 : mmq_few_ncg(ncols);
    const int nmt          = ids ? 1 : mmq_few_nmt(src0->type, ncg, src0->ne[1]);
    const int nunits_stage = mmq_few_nunits_stage(src0->type, ncg, nmt);
    if (src0->ne[0] % (nunits_stage*QK_K) != 0) {
        return false;
    }
    // unaligned data needs the alignment of the decoder loads
    const int x_alignment = mmq_few_unaligned(src0->type, nunits_stage) ? (ggml_type_size(src0->type) % 4 == 0 ? 4 : 2) : 16;
    return ggml_cuda_is_aligned(src0, x_alignment) && ggml_cuda_is_aligned(src1, 16);
}

#else // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

void ggml_cuda_mul_mat_q_few(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst) {
    GGML_UNUSED_VARS(ctx, src0, src1, ids, dst);
    GGML_ABORT("few-row MMQ is not supported on HIP or MUSA");
}

size_t ggml_cuda_mmq_few_shared_q8_size(const ggml_tensor * src0, const ggml_tensor * src1) {
    GGML_UNUSED_VARS(src0, src1);
    GGML_ABORT("few-row MMQ is not supported on HIP or MUSA");
}

void ggml_cuda_mul_mat_q_few(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst,
        char * src1_q_shared, const bool quantize_src1) {
    GGML_UNUSED_VARS(ctx, src0, src1, ids, dst, src1_q_shared, quantize_src1);
    GGML_ABORT("few-row MMQ is not supported on HIP or MUSA");
}

bool ggml_cuda_should_use_mmq_few(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, const int cc) {
    GGML_UNUSED_VARS(src0, src1, ids, cc);
    return false;
}

bool ggml_cuda_mmq_few_group_fits(const ggml_tensor * first, const ggml_tensor * mm) {
    GGML_UNUSED_VARS(first, mm);
    return false;
}

void ggml_cuda_mul_mat_q_few_group(ggml_backend_cuda_context & ctx, ggml_tensor * const * dsts, const int n,
        char * src1_q_shared, const bool quantize_src1) {
    GGML_UNUSED_VARS(ctx, dsts, n, src1_q_shared, quantize_src1);
    GGML_ABORT("few-row MMQ is not supported on HIP or MUSA");
}

#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
