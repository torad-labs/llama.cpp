#pragma once

#include "common.cuh"

// Routed experts (MUL_MAT_ID) at 1-MMVQ_MAX_BATCH_SIZE tokens, each SM streaming its tiles of expert rows through a ring
// of shared-memory slots with 1D bulk copies (mmvq-moe.cu). An expert that several tokens route to is read once for all
// of them, and a gate and its GLU fuse at any token count.
//
// The weights: expert e's row r at vx + e*stride_channel_x_bytes + r*row_bytes, rows dense (row_bytes = ncols_x /
// the type's block size * its block bytes). The tokens: q8_1 (quantize_row_q8_1_cuda's MMVQ layout), token t's column
// for expert slot s at y + (s % nchannels_y)*stride_channel_y + t*stride_col_y blocks. ids[s + t*ids_stride] is the
// expert in slot s of token t, and the result for that pair is dst[t*stride_col_dst + s*stride_channel_dst + r].
// A bias (x_bias for vx's product, gate_bias for vgate's) is expert e's row r at e*stride_bias + r.
struct ggml_cuda_mmvq_moe_args {
    ggml_type          type;
    const void *       vx;
    const void *       vgate;     // nullptr: dst = vx's product (+ x_bias); else the GLU of the two products
    const void *       y;
    const int32_t *    ids;
    const float *      x_bias;
    const float *      gate_bias;
    float *            dst;
    ggml_glu_op        glu_op;
    float              glu_limit; // > 0: the gate clamped to [-inf, limit] and vx's product to [-limit, limit] before the GLU
    int64_t            ncols_x;
    int64_t            nrows_x;
    int64_t            stride_channel_x_bytes;
    int64_t            n_used;    // experts a token routes to: ids' columns
    int64_t            ntokens;
    int64_t            ids_stride;
    int64_t            nchannels_y;
    int64_t            stride_col_y;
    int64_t            stride_channel_y;
    int64_t            stride_col_dst;
    int64_t            stride_channel_dst;
    int64_t            stride_bias;
    int *              tile_ctr;  // the stream's tile counter (ggml_cuda_pq2_tile_counters), or nullptr: every tile its
                                  // block's own
};

// Whether ggml_cuda_mmvq_moe takes these weights: an NVIDIA GPU from Hopper on (cp.async.bulk), a type and token count
// it measured faster at (IQ3_XXS; Q8_0 past one token), 16-byte aligned rows, tiles and experts, a ring that fits the
// shared memory, at most MMVQ_MAX_BATCH_SIZE tokens and 64 token/expert pairs. GGML_CUDA_MMVQ_MOE_LEGACY=1: never.
bool ggml_cuda_mmvq_moe_usable(int cc, ggml_type type, const void * vx, const void * vgate, int64_t ncols_x,
                               int64_t nrows_x, int64_t stride_row_x, int64_t stride_channel_x, int64_t n_used,
                               int64_t ntokens);

void ggml_cuda_mmvq_moe(const ggml_cuda_mmvq_moe_args & args, cudaStream_t stream);
