#include "common.cuh"

// F32 products that cuBLAS runs well below the time their bytes take, where one operand is small:
//   many rows of weights against up to 32 columns (Inkling 2's router, 4,608 -> 384 at 23 tokens): split-K, one launch
//   a short K (up to 32) with the weights shared by every column and head (Inkling 2's relative-bias table)
// ggml_cuda_mul_mat_f32_few runs the product and returns true, or returns false for the dispatch to go on.
// GGML_CUDA_F32_FEW=1 turns it on; the sums run in another order than cuBLAS's.
bool ggml_cuda_mul_mat_f32_few(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
