#include "common.cuh"

// GGML_OP_NORM_CLIP: each row scaled, then scaled down to an L2 norm of at most the threshold, in one kernel
void ggml_cuda_op_norm_clip(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// GGML_OP_ATTN_RES_MIX: the softmax-weighted mix of a token's sources over depth, in one kernel a token
void ggml_cuda_op_attn_res_mix(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
