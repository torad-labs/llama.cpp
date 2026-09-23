#pragma once
#include "common.cuh"
#include "ggml.h"

// fused-kernel recurrent-state output; strides in elements (per-seq stride is always D, set in-kernel)
struct ggml_cuda_gated_delta_net_fused_cache {
    float * data;        // rollback slot 0
    int64_t slot_stride; // between rollback slots (0 when K==1)
};

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// same op, but writes the snapshot(s) into the cache instead of dst (see ggml_cuda_try_gdn_cache_fusion)
void ggml_cuda_op_gated_delta_net_fused_cache(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
                                              ggml_cuda_gated_delta_net_fused_cache cache);

// Shape half of the chunked-prefill predicate (plus the GGML_CUDA_GDN_CHUNKED switch), with no
// dependence on the current device: the buffer type's get_alloc_size sizes the chunked scratch from
// it, and deciding from shape alone can only over-allocate on an ineligible device, never under.
bool ggml_cuda_gdn_chunked_shape_eligible(const ggml_tensor * dst);

// Whether this op takes the chunked prefill path on the current device.
bool ggml_cuda_should_use_chunked_gdn(const ggml_tensor * dst);
