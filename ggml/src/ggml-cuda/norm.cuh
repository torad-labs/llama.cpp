#include "common.cuh"

void ggml_cuda_op_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_group_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rms_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// The weightless RMS_NORM dst written to y as BF16 rather than to dst, rounded as convert_unary rounds F32 to BF16: the
// cast cuBLAS makes of an F32 src1 for a BF16 GEMM, of exactly the values ggml_cuda_op_rms_norm writes
void ggml_cuda_op_rms_norm_bf16(ggml_backend_cuda_context & ctx, const ggml_tensor * dst, nv_bfloat16 * y);

void ggml_cuda_op_add_rms_norm_fused(
        ggml_backend_cuda_context & ctx, ggml_tensor * add, ggml_tensor * rms_norm, ggml_tensor * mul);

void ggml_cuda_op_add_rms_norm_scale_fused(
        ggml_backend_cuda_context & ctx, ggml_tensor * add, ggml_tensor * rms_norm, float * row_scale);

void ggml_cuda_op_rms_norm_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * mul_tensor);

void ggml_cuda_op_rms_norm_fused_add(ggml_backend_cuda_context & ctx,
                                     ggml_tensor *               dst,
                                     ggml_tensor *               mul_tensor,
                                     ggml_tensor *               add_tensor);

// silu(glu->src[0]) * (rms_norm * w), the RMS_NORM -> MUL -> GLU (swiglu split, the product as its second operand)
// chain written to the GLU's tensor
void ggml_cuda_op_rms_norm_mul_gate_fused(
        ggml_backend_cuda_context & ctx, ggml_tensor * rms_norm, ggml_tensor * mul_tensor, ggml_tensor * glu);

void ggml_cuda_op_rms_norm_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_l2_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
