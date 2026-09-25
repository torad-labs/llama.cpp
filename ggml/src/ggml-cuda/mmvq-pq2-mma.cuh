#pragma once

#include "common.cuh"

// PQ2_0 x q8_1 at 2-8 columns (an MTP verify's rows) on int8 tensor cores, weights streamed through per-warp TMA rings:
// ggml_cuda_mmvq_pq2_mma. It takes the operands mul_mat_vec_q takes (vy is src1 quantized to q8_1, stride_col_y blocks
// per column) and writes the same dst; x_bias, when set, is added per (row, column) as mul_mat_vec_q's fused ADD does.

// Whether the kernel serves this matmul: GeForce Blackwell (sm_120) built for it, 2-8 columns, K a multiple of 1024 (every
// row a whole number of 16-byte TMA units), the weights 16-byte aligned. GGML_CUDA_PQ2_MMA_LEGACY=1 turns it off.
bool ggml_cuda_mmvq_pq2_mma_usable(int cc, const void * vx, int64_t ncols_x, int64_t nrows_x, int64_t stride_row_x,
                                   int64_t ncols_dst);

void ggml_cuda_mmvq_pq2_mma(ggml_backend_cuda_context & ctx, const void * vx, const void * vy, const float * x_bias,
                            float * dst, int64_t ncols_x, int64_t nrows_x, int64_t ncols_dst, int64_t stride_row_x,
                            int64_t stride_col_y, int64_t stride_col_dst, cudaStream_t stream);
