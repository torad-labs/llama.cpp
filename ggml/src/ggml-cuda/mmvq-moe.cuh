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
    // non-null in place of y: the tokens' vectors in f32, slot s's of token t at y_f32 + s*y_f32_s1 + t*y_f32_s2, each
    // slot its own (nchannels_y == n_used), which the launch quantizes into its shared copy exactly as
    // quantize_row_q8_1_cuda writes y (ggml_cuda_mmvq_moe_quantizes_y)
    const float *      y_f32;
    int64_t            y_f32_s1;
    int64_t            y_f32_s2;
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
    int64_t            n_experts; // the experts ids name: src0's channels
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
    bool               ids_ready; // ids whole before the launch starts (an earlier launch here on the stream read them, and
                                  // triggers the next only past its dependency wait): the experts listed and the block's
                                  // first tiles issued before this one's
    unsigned int *     l2_issue_stop; // bumped as the launch's reads start (ggml_cuda_l2_issue_stop), or nullptr
};

// Whether ggml_cuda_mmvq_moe takes these weights: an NVIDIA GPU from Hopper on (cp.async.bulk), a type and token count
// it measured faster at (IQ3_XXS; Q8_0 past one token), 16-byte aligned rows, tiles and experts, a ring that fits the
// shared memory, at most MMVQ_MAX_BATCH_SIZE tokens and 64 token/expert pairs. GGML_CUDA_MMVQ_MOE_LEGACY=1: never.
bool ggml_cuda_mmvq_moe_usable(int cc, ggml_type type, const void * vx, const void * vgate, int64_t ncols_x,
                               int64_t nrows_x, int64_t stride_row_x, int64_t stride_channel_x, int64_t n_used,
                               int64_t ntokens);

// The bytes from y to the end of the last vector a routed launch's pairs read (the args' layout), or 0 when they are not
// a multiple of 16: what the producer copies into shared memory past the ring, with one bulk copy, before any tile. A
// global load that a consumer issues behind the ring's copies waits for them.
int64_t ggml_cuda_mmvq_moe_y_bytes(int64_t ncols_x, int64_t n_used, int64_t ntokens, int64_t nchannels_y,
                                   int64_t stride_col_y, int64_t stride_channel_y);

// Whether a routed launch of these weights (a gate beside them or not) keeps y_bytes of the tokens' vectors in shared
// memory: its plan's tiles and a slot a team fit beside them. Otherwise, and with GGML_CUDA_MMVQ_MOE_Y_GLOBAL=1, the
// consumers read them from global memory.
bool ggml_cuda_mmvq_moe_keeps_y(ggml_type type, int64_t ncols_x, bool gate, int64_t y_bytes);

// Whether a routed launch quantizes its tokens' f32 vectors itself (args.y_f32): each slot's vector its own, 16-byte
// aligned rows, and a plan that keeps their q8_1 in shared memory. GGML_CUDA_MMVQ_MOE_QUANTIZE_LEGACY=1: never.
bool ggml_cuda_mmvq_moe_quantizes_y(ggml_type type, int64_t ncols_x, bool gate, int64_t n_used, int64_t ntokens,
                                    int64_t nchannels_y, const void * y_f32, int64_t y_f32_s1, int64_t y_f32_s2);

void ggml_cuda_mmvq_moe(const ggml_cuda_mmvq_moe_args & args, cudaStream_t stream);
