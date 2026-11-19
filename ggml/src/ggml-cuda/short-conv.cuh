#include "common.cuh"

// GGML_OP_SHORT_CONV: a depthwise short convolution over its state's window, with its dynamic taps and residual
void ggml_cuda_op_short_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
