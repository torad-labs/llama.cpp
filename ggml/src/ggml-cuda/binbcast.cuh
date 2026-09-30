#include "common.cuh"

void ggml_cuda_op_repeat(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_add(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_sub(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_mul(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_div(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_repeat_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_fused_add(ggml_backend_cuda_context & ctx, ggml_tensor * dst, int n_fuse);
void ggml_cuda_op_fused_mul(ggml_backend_cuda_context & ctx, ggml_tensor * dst, int n_fuse);

// Routed experts' weighted sum as build_moe_ffn writes it, MUL(experts, weights) then the slots' views added in order
// (ADD(ADD(v0, v1), v2), ...), in one launch: dst[i, t] = ((e[i,0,t]*w[0,t] + e[i,1,t]*w[1,t]) + ...), every product and
// sum rounded as those nodes round them, so bit for bit their result. experts [n_embd, n_used, n_tokens], weights
// [1, n_used, n_tokens], dst [n_embd, n_tokens], all F32 with contiguous rows; dst lies over neither input.
void ggml_cuda_op_moe_weighted_sum(ggml_backend_cuda_context & ctx, const ggml_tensor * experts,
        const ggml_tensor * weights, ggml_tensor * dst);
