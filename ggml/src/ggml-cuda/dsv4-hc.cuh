#include "common.cuh"
#include "ggml.h"

void ggml_cuda_op_dsv4_hc_comb(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_dsv4_hc_pre(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_dsv4_hc_post(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_dsv4_hc_weights(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// The front of a hyper-connection cycle, six nodes in two kernels: the weightless RMS_NORM of the flat streams
// (rms_flat), their MUL_MAT by hc_fn (mm), DSV4_HC_WEIGHTS (weights), DSV4_HC_PRE (pre) on its pre weights' view, and
// the RMS_NORM (rms) and MUL (mul) of the sublayer's norm; it writes weights and mul. Supported at up to 16 tokens, the
// flat streams being pre's streams, contiguous, and hc_fn F32, F16 or BF16.
bool ggml_cuda_dsv4_hc_pre_fused_supported(const ggml_tensor * rms_flat, const ggml_tensor * mm,
        const ggml_tensor * weights, const ggml_tensor * pre, const ggml_tensor * rms, const ggml_tensor * mul);
void ggml_cuda_op_dsv4_hc_pre_fused(ggml_backend_cuda_context & ctx, const ggml_tensor * rms_flat,
        const ggml_tensor * mm, ggml_tensor * weights, const ggml_tensor * pre, const ggml_tensor * rms,
        ggml_tensor * mul);
