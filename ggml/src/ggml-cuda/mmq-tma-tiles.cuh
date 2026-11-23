#pragma once

// The expert MMQ's weight tiles through TMA (cp.async.bulk.tensor) with mbarriers, on Blackwell sm_120.
//
// A Q3_K row of 4,608 values is 18 blocks of 110 bytes, 1,980 bytes: 4-byte aligned, not a 16-byte multiple, so a
// tile's rows cannot start a TMA copy and stock moves them in 16-byte cp.async copies from the 16-byte boundary
// below each stage (240 bytes moved per 220 used). Four adjacent rows are 7,920 bytes, a 16-byte multiple, and
// every 4-row group of a contiguous tensor starts 16-byte aligned. A rank-2 tensor map over the weight buffer
// as 880-byte chunks (7,920 = 9 * 880, element UINT64, 110 elements a chunk) has a 16-byte stride, and one box of
// 36 chunks (110 x 36 elements) is exactly a 16-row tile: the shared memory it fills is the tile's 16 rows in
// order, because a linear tensor's 880-byte chunks are consecutive. No load-time relayout, no row padding and no
// stride change, so every other reader of the tensor keeps the layout and numerics it has.
//
// The kernel is the 23-row decode shape's path: MUL_MAT_ID, one column group of at most MMQ_FEW_NCOLS entries an
// expert, the four warps splitting K of the expert's 16-row tile, the same int8 MMA fragments, y quantization and
// write-back as ggml_cuda_mul_mat_q_few, so its results are bit-identical to that kernel's.

#include "common.cuh"

// GGML_CUDA_MMQ_TMA=1 opts in; the kernel declines outside Blackwell sm_120 and the few-row MMQ's own gate.
bool ggml_cuda_mmq_tma_id(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                          const ggml_tensor * ids, ggml_tensor * dst);

// Fused Q3_K gate/up + SITU via two TMA boxes per 16-row tile (shared y fragments, evict_first).
bool ggml_cuda_mmq_tma_gate_up_situ(ggml_backend_cuda_context & ctx, const ggml_tensor * w, const ggml_tensor * x,
                                    const ggml_tensor * ids, ggml_tensor * glu, int32_t first, int32_t count,
                                    float beta, float lb, const int32_t * precomputed_local_ids = nullptr,
                                    bool zero_non_owned = true);

// Fused Q4_K down + deterministic slot-order weighted sum via TMA boxes (evict_first).
bool ggml_cuda_mmq_tma_down_weighted_sum(ggml_backend_cuda_context & ctx, const ggml_tensor * w, const ggml_tensor * x,
                                         const ggml_tensor * ids, const ggml_tensor * weights, ggml_tensor * dst,
                                         int32_t first, int32_t count,
                                         const int32_t * precomputed_local_ids = nullptr);
