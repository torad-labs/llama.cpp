#pragma once

#include "common.cuh"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

static constexpr int GGML_CUDA_AR_Q8_BLOCK_SIZE = 32;

struct ggml_cuda_ar_q8_block {
    float  d;
    int8_t qs[GGML_CUDA_AR_Q8_BLOCK_SIZE];
};
static_assert(sizeof(ggml_cuda_ar_q8_block) == 36, "unexpected Q8 wire block size");
static_assert(offsetof(ggml_cuda_ar_q8_block, qs) == sizeof(float), "unexpected Q8 wire layout");

static inline size_t ggml_cuda_ar_q8_nbytes(const int64_t ne) {
    GGML_ASSERT(ne >= 0);
    const uint64_t blocks = (uint64_t) ne / GGML_CUDA_AR_Q8_BLOCK_SIZE +
                            (ne % GGML_CUDA_AR_Q8_BLOCK_SIZE != 0);
    GGML_ASSERT(blocks <= std::numeric_limits<size_t>::max() / sizeof(ggml_cuda_ar_q8_block));
    return (size_t) blocks * sizeof(ggml_cuda_ar_q8_block);
}

// No .ftz: the wire's scale and ordered sum retain IEEE subnormals under -use_fast_math.
static __device__ __forceinline__ float ggml_cuda_ar_q8_div_rn(float a, float b) {
    float r;
    asm("div.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

static __device__ __forceinline__ float ggml_cuda_ar_q8_mul_rn(float a, float b) {
    float r;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

static __device__ __forceinline__ float ggml_cuda_ar_q8_add_rn(float a, float b) {
    float r;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

static __global__ void ggml_cuda_ar_quantize_q8_kernel(
        const float * src, ggml_cuda_ar_q8_block * wire, const int64_t ne, const bool active) {
    const int lane = threadIdx.x % GGML_CUDA_AR_Q8_BLOCK_SIZE;
    const int64_t first = ((int64_t) blockIdx.x * blockDim.x + threadIdx.x) / GGML_CUDA_AR_Q8_BLOCK_SIZE;
    const int64_t step = (int64_t) gridDim.x * blockDim.x / GGML_CUDA_AR_Q8_BLOCK_SIZE;
    const int64_t blocks = ne / GGML_CUDA_AR_Q8_BLOCK_SIZE + (ne % GGML_CUDA_AR_Q8_BLOCK_SIZE != 0);

    for (int64_t b = first; b < blocks; b += step) {
        const int64_t i = b * GGML_CUDA_AR_Q8_BLOCK_SIZE + lane;
        const float x = active && i < ne ? src[i] : 0.0f;
        uint32_t amax_bits = __float_as_uint(x) & 0x7fffffffu;
#pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            amax_bits = max(amax_bits, __shfl_xor_sync(0xffffffffu, amax_bits, offset));
        }
        float d = 0.0f;
        int q = 0;
        if (amax_bits >= 0x7f800000u) {
            // Mark the block invalid without trapping. Padded q lanes stay zero.
            d = __uint_as_float(0x7fc00000u);
            q = i < ne ? 1 : 0;
        } else if (amax_bits != 0) {
            d = ggml_cuda_ar_q8_div_rn(__uint_as_float(amax_bits), 127.0f);
            uint32_t d_bits = __float_as_uint(d);
            uint32_t max_bits = __float_as_uint(ggml_cuda_ar_q8_mul_rn(d, 127.0f));
            // Round a subnormal scale up when rounding it down would clip the block's maximum.
            if (d_bits < 0x00800000u && max_bits < amax_bits) {
                d = __uint_as_float(++d_bits);
                max_bits = __float_as_uint(ggml_cuda_ar_q8_mul_rn(d, 127.0f));
            }
            // At the largest finite inputs, keep the reconstructed maximum finite too.
            if (max_bits == 0x7f800000u) {
                d = __uint_as_float(d_bits - 1);
            }
            q = __float2int_rn(ggml_cuda_ar_q8_div_rn(x, d));
            q = max(-127, min(127, q));
        }
        wire[b].qs[lane] = (int8_t) q;
        if (lane == 0) {
            wire[b].d = d;
        }
    }
}

static __global__ void ggml_cuda_ar_reduce_q8_kernel(
        float * dst, const ggml_cuda_ar_q8_block * own_wire, const char * peers,
        const size_t peer_stride_bytes, const int64_t ne, const int rank, const int n_ranks) {
    const int64_t first = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t step = (int64_t) gridDim.x * blockDim.x;
    for (int64_t i = first; i < ne; i += step) {
        const int64_t b = i / GGML_CUDA_AR_Q8_BLOCK_SIZE;
        const int lane = i % GGML_CUDA_AR_Q8_BLOCK_SIZE;
        float sum = 0.0f;
        for (int peer = 0; peer < n_ranks; ++peer) {
            const int index = peer < rank ? peer : peer - 1;
            const auto * wire = peer == rank ? own_wire :
                reinterpret_cast<const ggml_cuda_ar_q8_block *>(peers + (size_t) index * peer_stride_bytes);
            const float value = ggml_cuda_ar_q8_mul_rn(wire[b].d, (float) wire[b].qs[lane]);
            sum = peer == 0 ? value : ggml_cuda_ar_q8_add_rn(sum, value);
        }
        dst[i] = sum;
    }
}

// Finite blocks: scale is RN(maxabs/127), with the two finite-range corrections above;
// q is nearest-even(RN(x/d)), clamped to [-127,127]. Zero blocks and padded lanes have q=0.
// Any active NaN or infinity marks its block with a quiet-NaN scale and q=1 on valid lanes.
static inline void ggml_cuda_ar_quantize_q8(
        const float * src, ggml_cuda_ar_q8_block * wire, const int64_t ne, const bool active, cudaStream_t stream) {
    GGML_ASSERT(ne > 0 && wire != nullptr && (!active || src != nullptr));
    const int64_t blocks = ne / GGML_CUDA_AR_Q8_BLOCK_SIZE + (ne % GGML_CUDA_AR_Q8_BLOCK_SIZE != 0);
    constexpr int threads = 256;
    const int grid = (int) std::min<int64_t>((blocks + threads / 32 - 1) / (threads / 32), 1024);
    ggml_cuda_ar_quantize_q8_kernel<<<grid, threads, 0, stream>>>(src, wire, ne, active);
    CUDA_CHECK(cudaGetLastError());
}

// Each rank reads its own quantized share too. Dequantize with RN(d*q), take rank 0's
// value, then RN-add ranks 1..n-1. Peers are byte-strided; the stride need not divide by 36.
static inline void ggml_cuda_ar_reduce_q8(
        float * dst, const ggml_cuda_ar_q8_block * own_wire, const char * peers, const size_t peer_stride_bytes,
        const int64_t ne, const int rank, const int n_ranks, cudaStream_t stream) {
    GGML_ASSERT(ne > 0 && dst != nullptr && own_wire != nullptr && peers != nullptr);
    GGML_ASSERT(n_ranks >= 2 && n_ranks <= GGML_CUDA_MAX_DEVICES && rank >= 0 && rank < n_ranks);
    GGML_ASSERT(peer_stride_bytes % alignof(ggml_cuda_ar_q8_block) == 0);
    GGML_ASSERT(ggml_cuda_ar_q8_nbytes(ne) <= peer_stride_bytes);
    constexpr int threads = 256;
    const int grid = (int) std::min<int64_t>((ne + threads - 1) / threads, 1024);
    ggml_cuda_ar_reduce_q8_kernel<<<grid, threads, 0, stream>>>(
        dst, own_wire, peers, peer_stride_bytes, ne, rank, n_ranks);
    CUDA_CHECK(cudaGetLastError());
}
