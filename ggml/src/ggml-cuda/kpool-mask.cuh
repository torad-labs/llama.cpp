#pragma once

#include "common.cuh"

// GGML_OP_KPOOL_MASK: GLM-5-Next's pooled-indexer masks from a cell's position and pool and a query row's facts
// (ggml_kpool_mask in ggml.h; ggml_compute_forward_kpool_mask on the CPU is the same tests)
void ggml_cuda_op_kpool_mask(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
