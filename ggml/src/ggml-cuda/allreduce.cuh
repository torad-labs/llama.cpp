#pragma once

#include "common.cuh"
#include "ggml-backend-impl.h"

#include <cstddef>

// Opaque pipeline context -- owns all pinned buffers, streams, and events.
struct ggml_cuda_ar_pipeline;

// Allocate a pipeline for n_devices GPUs.
// devices[] holds the GPU device IDs in rank order.
// Returns nullptr on allocation failure.
ggml_cuda_ar_pipeline * ggml_cuda_ar_pipeline_init(
    const int * devices, size_t n_devices);

// Release all resources owned by the pipeline.
void ggml_cuda_ar_pipeline_free(ggml_cuda_ar_pipeline * pipeline);

// Execute an in-place AllReduce (sum) across tensors[0..n_devices-1].
// tensors[i] must live on the device managed by backends[i] and be
// contiguous F32, F16, or BF16.
// Preconditions are checked by the CUDA comm dispatcher before calling this.
// Returns true once the reduction work has been enqueued successfully.
bool ggml_cuda_ar_allreduce(
    ggml_cuda_ar_pipeline * pipeline,
    ggml_backend_t        * backends,
    ggml_tensor           ** tensors,
    const ggml_cgraph * const * next = nullptr, bool owned_rows = false);

// In-place exact gather across tensors[0..n_devices-1]: each holds its device's part and zero bits elsewhere, and all end up
// with the bitwise OR of the shares, 32-bit word by word, on the chunked or LL kernel (capturable). tensors[i] must be
// contiguous, 16-byte aligned and a whole number of words.
bool ggml_cuda_ar_allgather(
    ggml_cuda_ar_pipeline * pipeline,
    ggml_backend_t        * backends,
    ggml_tensor           ** tensors);

// Equal contiguous owner ranges in rank order, in 32-bit words. Zero selects the padded gather.
size_t ggml_cuda_ar_equal_gather_count(ggml_tensor ** tensors, size_t n_ranks);

// The AllReduce's sum, delivered only to each rank's own elements [begin[i], begin[i] + len[i]) of tensors[i] (same
// preconditions), with the bits the AllReduce gives them; a rank's other elements keep its share (zeros on an inactive
// rank). period > 0: the ranges repeat every period elements, as when each rank reads a part of every row. It runs on
// the chunked kernel with more than two ranks, and declines (false, nothing queued or changed) on the LL and
// copy-engine routes, two ranks, overlapping or out-of-range owners, or GGML_CUDA_AR_REDUCE_OWNERS=0.
bool ggml_cuda_ar_reduce_owned(
    ggml_cuda_ar_pipeline * pipeline,
    ggml_backend_t        * backends,
    ggml_tensor           ** tensors,
    const int64_t         * begin,
    const int64_t         * len,
    int64_t                 period);

// Whether this reduction would run over the copy engines yet can still be
// captured whole: GGML_CUDA_AR_COPY_ENGINE_CAPTURE=1 opts it in -- its copies,
// add kernels and cross-card guards then run as graph nodes and mapped-counter
// probes, and the two-back pacing between reductions becomes an edge inside the
// step's graph (doc 17 B). GGML_CUDA_AR_COPY_ENGINE_CAPTURE_LEGACY=1 reverts to
// the old refusal, so the route keeps running as live host-sliced segments.
bool ggml_cuda_ar_capturable_copy_engine(
    const ggml_cuda_ar_pipeline * pipeline,
    const ggml_tensor           * t);

// Whether the AllReduce of a tensor like t (same preconditions) can be captured
// into a CUDA graph and replayed: the chunked kernel always, which takes its
// tokens and slots from the device; the copy-engine route only when
// ggml_cuda_ar_capturable_copy_engine admits it.
bool ggml_cuda_ar_capturable(
    const ggml_cuda_ar_pipeline * pipeline,
    const ggml_tensor           * t);


