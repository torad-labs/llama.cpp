#pragma once

#include "common.cuh"

// PQ2_0 x q8_1 at 1-8 columns (decode and an MTP verify's rows) on int8 tensor cores, weights streamed through a TMA
// ring per SM: ggml_cuda_mmvq_pq2_mma. It takes the operands mul_mat_vec_q takes (vy is src1 quantized to q8_1,
// stride_col_y blocks per column) and writes the same dst; x_bias, when set, is added per (row, column) as
// mul_mat_vec_q's fused ADD does. With vgate (a second PQ2_0 matrix of vx's shape and strides) it writes a gated FFN's
// up * silu(gate), mul_mat_vec_q's fused SWIGLU, and takes no bias.

// Whether the kernel serves this matmul, and then it always does: GeForce / RTX PRO Blackwell (sm_120, not yet measured
// on GB10's sm_121) built for it, 1-8 columns, K a multiple of 1024 (every row a whole number of 16-byte TMA units), the
// weights 16-byte aligned. It holds no state between launches. GGML_CUDA_PQ2_MMA_LEGACY=1 turns it off.
bool ggml_cuda_mmvq_pq2_mma_usable(int cc, const void * vx, const void * vgate, int64_t ncols_x, int64_t nrows_x,
                                   int64_t stride_row_x, int64_t ncols_dst);

void ggml_cuda_mmvq_pq2_mma(const void * vx, const void * vgate, const void * vy, const float * x_bias, float * dst,
                            int64_t ncols_x, int64_t nrows_x, int64_t ncols_dst, int64_t stride_row_x,
                            int64_t stride_col_y, int64_t stride_col_dst, cudaStream_t stream);

#define PQ2_MMA_MAX_GROUP 4

// n (2 to PQ2_MMA_MAX_GROUP) matrices of ncols_x columns on one activation vy, in one launch: matrix g (vx[g], nrows_x[g]
// rows of stride_row_x[g] blocks) writes dst[g], column c at dst[g] + c*stride_col_dst[g]. Each matrix must be one
// ggml_cuda_mmvq_pq2_mma_usable serves unfused, and each gets that launch's result.
void ggml_cuda_mmvq_pq2_mma_group(int n, const void * const * vx, float * const * dst, const int64_t * nrows_x,
                                  const int64_t * stride_row_x, const int64_t * stride_col_dst, const void * vy,
                                  int64_t ncols_x, int64_t ncols_dst, int64_t stride_col_y, cudaStream_t stream);
