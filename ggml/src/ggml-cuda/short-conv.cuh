#include "common.cuh"

// GGML_OP_SHORT_CONV: a depthwise short convolution over its state's window, with its dynamic taps and optional residual add
#if defined(GGML_CUDA_USE_L2_WINDOW)
void ggml_cuda_op_short_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
        const ggml_tensor * residual = nullptr, ggml_tensor * add_dst = nullptr,
        cudaAccessPolicyWindow l2_window = {});
#else
void ggml_cuda_op_short_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
        const ggml_tensor * residual = nullptr, ggml_tensor * add_dst = nullptr);
#endif
