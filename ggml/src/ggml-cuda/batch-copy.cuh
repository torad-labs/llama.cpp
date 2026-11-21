#pragma once

#include "ggml-backend.h"

bool ggml_cuda_get_tensor_n(int device,
        const ggml_tensor * const * tensors, void * const * data, const size_t * offsets,
        const size_t * sizes, size_t n_tensors);

// the bytes of the scratch each card keeps for batched copies
constexpr size_t ggml_cuda_tensor_n_stage_bytes = 4*1024*1024;

// A batch of tensor ranges copied through the scratch: a kernel gathers the ranges and they go to host memory as one copy per run of adjacent host ranges (get), or they come that way and a kernel scatters them (set).
// false: nothing was copied (fewer than a few ranges fit the scratch, the ranges written overlap, or there is no scratch).
bool ggml_cuda_get_tensor_n_staged(int device,
        const ggml_tensor * const * tensors, void * const * data, const size_t * offsets,
        const size_t * sizes, size_t n_tensors);

bool ggml_cuda_set_tensor_n_staged(int device,
        const ggml_tensor * const * tensors, const void * const * data, const size_t * offsets,
        const size_t * sizes, size_t n_tensors);

bool ggml_cuda_set_tensor_n(int device,
        ggml_tensor * const * tensors, const void * const * data, const size_t * offsets,
        const size_t * sizes, size_t n_tensors);
