#pragma once
#include "common.cuh"
#include "ggml.h"

// the most snapshot slots whose rows a fused store takes from row ids (ggml_cuda_gated_delta_net_store_rows)
#define GGML_CUDA_GDN_STORE_ROWS_MAX 16

// -sm dp's snapshot stores (build_rs_store_local): sequence s's snapshot of slot i goes to row slot[i][s] of the cache,
// row_stride elements a row. A kernel argument by value; slot[0] == nullptr when the rows are not ids
struct ggml_cuda_gated_delta_net_store_rows {
    const int64_t * slot[GGML_CUDA_GDN_STORE_ROWS_MAX] = {};
    int64_t         row_stride                         = 0;
};

// fused-kernel recurrent-state output; strides in elements (per-seq stride is always D, set in-kernel, unless rows)
struct ggml_cuda_gated_delta_net_fused_cache {
    void *    data        = nullptr; // rollback slot 0: rows of type; with rows, the cache's row 0
    int64_t   slot_stride = 0;       // between rollback slots, in elements (0 when K==1)
    ggml_type type        = GGML_TYPE_F32; // f32, or the -cts cache f16 (the kernel converts as it writes) or q8_0 (it quantizes)
    ggml_cuda_gated_delta_net_store_rows rows; // -sm dp: each sequence's row per slot, in place of slot_stride
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
