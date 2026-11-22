#pragma once

#include "common.cuh"

// How the relative tile kernels read a row of a K or V cache into a shared memory tile of f16, so that Q and the products
// stay f16 whatever the cache holds. A row is 128 values. A lane takes the half of a row at part (0 or 1): load reads its
// bytes from global memory, store writes its 64 values as halves (8 int4) to dst. A caller loads K and V before it stores
// either, so all its reads are in flight together.
template<ggml_type type> struct fattn_rel_elem;

#if defined(FP8_AVAILABLE) && !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#define FATTN_REL_ELEM_F8
#endif

// The cache types the tile kernels read besides f16: q8_0, and f8_e4m3 where CUDA has its conversion.
static bool fattn_rel_elem_one_byte(const ggml_type type) {
#ifdef FATTN_REL_ELEM_F8
    return type == GGML_TYPE_Q8_0 || type == GGML_TYPE_F8_E4M3;
#else
    return type == GGML_TYPE_Q8_0;
#endif
}

// The bytes a row's address must be a multiple of: q8_0 rows of 136 bytes are read 8 bytes at a time, the others 16.
static size_t fattn_rel_elem_align(const ggml_type type) {
    return type == GGML_TYPE_Q8_0 ? 8 : 16;
}

template<> struct fattn_rel_elem<GGML_TYPE_F16> {
    struct raw { int4 x[8]; };

    static __device__ __forceinline__ raw load(const char * row, const int part) {
        raw r;
        const int4 * src = (const int4 *) row + 8*part;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            r.x[j] = src[j];
        }
        return r;
    }

    static __device__ __forceinline__ void store(const raw & r, const int part, int4 * dst) {
        GGML_UNUSED(part);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            dst[j] = r.x[j];
        }
    }
};

// q8_0: 4 blocks of an f16 scale d and 32 signed bytes q. Part p is blocks 2p and 2p + 1, bytes 68p to 68p + 68, read as
// the 9 8-byte words from byte 64p. A value is d*q rounded once to f16, as dequantize_block_q8_0 gives it.
template<> struct fattn_rel_elem<GGML_TYPE_Q8_0> {
    struct raw { int2 x[9]; };

    static __device__ __forceinline__ raw load(const char * row, const int part) {
        raw r;
        const int2 * src = (const int2 *) row + 8*part;
#pragma unroll
        for (int j = 0; j < 9; ++j) {
            r.x[j] = src[j];
        }
        return r;
    }

    static __device__ __forceinline__ void store(const raw & r, const int part, int4 * dst) {
        // the 17 words of the two blocks start at word 0 of the load for part 0 and at word 1 for part 1
        int w[17];
#pragma unroll
        for (int l = 0; l < 17; ++l) {
            const int lo = l % 2 == 0 ? r.x[l/2].x : r.x[l/2].y;
            const int hi = l % 2 == 0 ? r.x[l/2].y : r.x[l/2 + 1].x;
            w[l] = part ? hi : lo;
        }
        // Block 0: d in bytes 0-1, q in bytes 2-33. Block 1: d in bytes 34-35, q in bytes 36-67.
        const half d[2] = {__ushort_as_half(w[0] & 0xFFFF), __ushort_as_half(uint32_t(w[8]) >> 16)};
        // A byte q + 128 under the f16 exponent byte 0x64 is 1024 + 128 + q: remove 1152.
        const half2 bias = make_half2(1152.0f, 1152.0f);
#pragma unroll
        for (int b = 0; b < 2; ++b) {
            const half2 d2 = __half2half2(d[b]);
            half2 v[16];
#pragma unroll
            for (int l = 0; l < 8; ++l) {
                const int qs = (b == 0 ? __byte_perm(w[l], w[l + 1], 0x5432) : w[l + 9]) ^ 0x80808080;
                const int v0 = __byte_perm(qs, 0x64646464, 0x4140);
                const int v1 = __byte_perm(qs, 0x64646464, 0x4342);
                v[2*l + 0] = __hmul2(__hsub2(*((const half2 *) &v0), bias), d2);
                v[2*l + 1] = __hmul2(__hsub2(*((const half2 *) &v1), bias), d2);
            }
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                ggml_cuda_memcpy_1<16>(dst + 4*b + j, v + 4*j);
            }
        }
    }
};

#ifdef FATTN_REL_ELEM_F8
// f8_e4m3 without a scale: a byte a value, 128 bytes a row. Every e4m3 value is exact in f16.
template<> struct fattn_rel_elem<GGML_TYPE_F8_E4M3> {
    struct raw { int4 x[4]; };

    static __device__ __forceinline__ raw load(const char * row, const int part) {
        raw r;
        const int4 * src = (const int4 *) row + 4*part;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            r.x[j] = src[j];
        }
        return r;
    }

    // two e4m3 values, the low byte first, as two halves, the low half first
    static __device__ __forceinline__ int pair(const uint32_t x) {
        const __half2_raw h = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t) (x & 0xFFFF), __NV_E4M3);
        return int(uint32_t(h.x) | uint32_t(h.y) << 16);
    }

    static __device__ __forceinline__ void store(const raw & r, const int part, int4 * dst) {
        GGML_UNUSED(part);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const uint32_t w[4] = {uint32_t(r.x[j].x), uint32_t(r.x[j].y), uint32_t(r.x[j].z), uint32_t(r.x[j].w)};
            dst[2*j + 0] = make_int4(pair(w[0]), pair(w[0] >> 16), pair(w[1]), pair(w[1] >> 16));
            dst[2*j + 1] = make_int4(pair(w[2]), pair(w[2] >> 16), pair(w[3]), pair(w[3] >> 16));
        }
    }
};
#endif // FATTN_REL_ELEM_F8
