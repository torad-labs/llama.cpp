#include "common.cuh"

// Rank-1 LoRA delta folded into one launch (decode, one token):
//   dst[n] = res[n] + scale * dot(a, x) * b[n]
// a: [K, 1] F32, x: [K, 1] F32, b: [1, N] F32, res/dst: [N, 1] F32.
// Returns false (and launches nothing) when a shape or layout is outside that contract.
bool ggml_cuda_op_lora_rank1_fused(ggml_backend_cuda_context & ctx,
                                   const ggml_tensor * a, const ggml_tensor * x, const ggml_tensor * b,
                                   float scale, const ggml_tensor * res, ggml_tensor * dst);
