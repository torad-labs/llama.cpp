#pragma once

#include "common.cuh"

// fwht.cu's own gate for its register kernels; declared here so my launchers can check the hint-marked matmuls their
// chains end at without pulling in that header again inside one translation unit
bool ggml_cuda_op_mul_mat_use_fwht(const struct ggml_tensor * op);

// Schedule-level fusions of Inkling 2 Mini's attention-prep chains (the decode profile's candidates C1, C2, C3, D1
// and E1): the row normalizer, the tau mul and the Hadamard rotation of Q (C1), the K and V epilogues' convolution,
// normalization and rotation (C2/C3), the relative-position clip and bias table (D1) and the conv-state store's
// window joins and cache write (E1), each in one launch where the graph walk recognizes them. Every launcher returns
// false where the shapes do not fit, leaving the chain to its own kernels.
//
//   q tail (C1)    ggml_cuda_op_attn_prep_clip_fwht:  NORM_CLIP -> MUL(tau)? -> reshape -> MUL_MAT(hadamard hint)
//   rel tail (D1)  ggml_cuda_op_attn_prep_rel_table: NORM_CLIP -> MUL(tau)? -> MUL_MAT(rel bias)
//   k/v tails      ggml_cuda_op_attn_prep_kv_tail:   SHORT_CONV -> reshapes -> NORM_CLIP(null)/RMS_NORM ->
//                                                                    reshape -> MUL_MAT(hadamard hint)
//   state store    ggml_cuda_op_attn_prep_state_store: CONCAT(dim 1) chain -> snapshot view/cont -> SET_ROWS

// C1: clip (and the tau mul it feeds) with the Hadamard rotation that consumes it, per head row of n elements.
bool ggml_cuda_op_attn_prep_clip_fwht(ggml_backend_cuda_context & ctx, const ggml_tensor * norm,
        const ggml_tensor * mul, ggml_tensor * mm);

// D1: the relative-position rows' clip and their bias-table matmul in one kernel after the rel mat-vec.
bool ggml_cuda_op_attn_prep_rel_table(ggml_backend_cuda_context & ctx, const ggml_tensor * norm,
        const ggml_tensor * mul, ggml_tensor * table);

// C2/C3: a K or V epilogue from the window's sources to its rotated rows; window is the claimed CONCAT (written by
// this kernel beside its output) when win is given, otherwise an already materialized tensor it only reads.
bool ggml_cuda_op_attn_prep_kv_tail(ggml_backend_cuda_context & ctx, const ggml_tensor * conv, ggml_tensor * mm,
        const ggml_cuda_short_conv_window * win);

// E1: the state store. parts are the layer's window pieces in channel order (base first), window their dim-1 join,
// snapshot the view of it whose columns replace the cache rows, and set_rows the SET_ROWS into the recurrent cache
// this kernel takes over.
struct attn_prep_store_parts {
    const ggml_tensor * part[8];
    int                 m = 0;
};

bool ggml_cuda_op_attn_prep_state_store(ggml_backend_cuda_context & ctx, const attn_prep_store_parts & parts,
        const ggml_tensor * window, const ggml_tensor * snapshot, const ggml_tensor * set_rows);
