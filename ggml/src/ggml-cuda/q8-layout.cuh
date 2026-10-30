#pragma once

#include "common.cuh"
#include "convert.cuh"
#include "../ggml-q8-layout.h"

// Metadata belongs to the CUDA buffer, not tensor::extra (used by row split).
// A view's data pointer remains a logical byte address into the same allocation.
struct ggml_cuda_q8_tma_descriptor;
ggml_cuda_q8_layout ggml_cuda_q8_layout_of(const ggml_tensor * tensor, const ggml_cuda_q8_tma_descriptor ** maps = nullptr);
void ggml_cuda_q8_prepare_write(const ggml_tensor * tensor);

#if defined(CUDART_VERSION) && CUDART_VERSION >= 12000 && !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#define GGML_CUDA_Q8_TMA
#include <cuda.h>
struct ggml_cuda_q8_tma_descriptor {
    CUtensorMap codes;
    CUtensorMap scales;
    uint32_t columns;
};
std::shared_ptr<ggml_cuda_q8_tma_descriptor> ggml_cuda_q8_tma_create(void * base, uint32_t columns, uint32_t rows);
#endif

bool ggml_cuda_q8_tma(ggml_backend_cuda_context & ctx, const ggml_tensor * weight, const ggml_tensor * dst,
                      const void * y, uint32_t y_stride, const ggml_cuda_mm_fusion_args_device & fusion, float * out);

static __device__ __forceinline__ size_t ggml_cuda_q8_physical_byte(
        const ggml_cuda_q8_layout & layout, const size_t logical) {
    const size_t row_bytes = (size_t) layout.blocks_per_row.z * sizeof(block_q8_0);
    const size_t byte = layout.offset + logical;
    const size_t row = byte / row_bytes;
    return row * row_bytes + ggml_q8_0_split_layout{layout.blocks_per_row.z}.offset(byte % row_bytes);
}

static __device__ __forceinline__ const int8_t * ggml_cuda_q8_codes(
        const ggml_cuda_q8_layout & layout, const uint32_t block) {
    const uint2 rc = fast_div_modulo((uint32_t) (layout.offset / sizeof(block_q8_0)) + block, layout.blocks_per_row);
    return (const int8_t *) (layout.base + (size_t) rc.x * layout.blocks_per_row.z * sizeof(block_q8_0) + rc.y * QK8_0);
}

static __device__ __forceinline__ half ggml_cuda_q8_scale(
        const ggml_cuda_q8_layout & layout, const uint32_t block) {
    const uint2 rc = fast_div_modulo((uint32_t) (layout.offset / sizeof(block_q8_0)) + block, layout.blocks_per_row);
    const char * row = layout.base + (size_t) rc.x * layout.blocks_per_row.z * sizeof(block_q8_0);
    return ((const half *) (row + layout.blocks_per_row.z * QK8_0))[rc.y];
}

// Copies canonical logical bytes, applying each endpoint's physical layout.
// Endpoints without a layout are ordinary device pointers. No global scratch.
void ggml_cuda_q8_copy_bytes(const void * src, ggml_cuda_q8_layout src_layout,
                           void * dst, ggml_cuda_q8_layout dst_layout, size_t size, cudaStream_t stream);
void ggml_cuda_q8_memset_bytes(ggml_cuda_q8_layout layout, uint8_t value, size_t size, cudaStream_t stream);

// True when the complete supported copy/conversion was submitted.
bool ggml_cuda_q8_copy_tensor(ggml_backend_cuda_context & ctx, const ggml_tensor * src, ggml_tensor * dst);

template <typename T>
static __global__ void ggml_cuda_q8_convert_kernel(ggml_cuda_q8_layout layout, T * dst,
        int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, size_t nb1, size_t nb2, size_t nb3) {
    ggml_cuda_pdl_sync();
    const int64_t n = ne0 * ne1 * ne2 * ne3;
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x) {
        const int64_t c = i % ne0;
        const int64_t r = i / ne0;
        const size_t byte = (r % ne1)*nb1 + (r / ne1 % ne2)*nb2 + (r / (ne1*ne2))*nb3 + (c / QK8_0)*sizeof(block_q8_0);
        const uint32_t block = byte / sizeof(block_q8_0);
        dst[i] = ggml_cuda_cast<T>(float(ggml_cuda_q8_scale(layout, block)) * ggml_cuda_q8_codes(layout, block)[c % QK8_0]);
    }
}

template <typename T>
static void ggml_cuda_q8_convert(const ggml_tensor * src, T * dst, cudaStream_t stream) {
    const auto layout = ggml_cuda_q8_layout_of(src);
    GGML_ASSERT(layout.base != nullptr && layout.offset % sizeof(block_q8_0) == 0);
    if (ggml_nelements(src) == 0) { return; }
    const int blocks = (int) std::min<int64_t>((ggml_nelements(src) + 255) / 256, 65535);
    ggml_cuda_q8_convert_kernel<<<blocks, 256, 0, stream>>>(layout, dst, src->ne[0], src->ne[1], src->ne[2], src->ne[3],
                                                         src->nb[1], src->nb[2], src->nb[3]);
    CUDA_CHECK(cudaGetLastError());
}
