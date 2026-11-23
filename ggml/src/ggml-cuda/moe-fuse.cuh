#pragma once

#include "common.cuh"

// the fused router kernel tiles n_embd columns by this many floats per step
constexpr int MOE_ROUTER_COL_TILE_C = 128;

// A3: Inkling 2's router product and top-k selection as one launch at decode shapes (build_moe_ffn's SIGMOID gating):
// cur [n_embd, n_tokens] times the f32 expert-router rows gives the logits, whose sigmoid (+ the experts' bias when the
// model carries one) is top-k'd into ids and normalized weights exactly as ggml_cuda_op_topk_moe writes them.
void ggml_cuda_op_moe_router_topk(ggml_backend_cuda_context & ctx,
        const ggml_tensor * cur, const ggml_tensor * router, ggml_tensor * ids, ggml_tensor * weights,
        const ggml_tensor * bias, bool with_norm, bool norm_eps, float clamp_val, float scale_val);

// D2: the attention output tail after build_attn in one launch: the FWHT the v_rot Hadamard-hinted MUL_MAT stands for,
// the SIGLU gate against it, and the q8_1 copy the wo mat-vec reads instead of its own quantize staging. dst_f32 is the
// GLU node's buffer; q8_dst holds nrows(dst_f32) rows of ne0/QK8_1-padded blocks as mmvq_src1_q8_1 lays them out.
void ggml_cuda_op_fwht_siglu_q8(ggml_backend_cuda_context & ctx,
        const ggml_tensor * x, const ggml_tensor * gate, ggml_tensor * dst_f32,
        block_q8_1 * q8_dst, int64_t q8_row_blocks);

enum ggml_cuda_siglu_q8_mode {
    GGML_CUDA_SIGLU_Q8_NONE    = 0,
    GGML_CUDA_SIGLU_Q8_MMVQ    = 1,
    GGML_CUDA_SIGLU_Q8_MMQ_FEW = 2,
};

// Optional 64/128-wide block FWHT + SIGLU + optional mmvq/mmq_few Q8_1 staging; dst_f32 is null when GLU is elided.
void ggml_cuda_op_attn_siglu_fuse(ggml_backend_cuda_context & ctx,
        const ggml_tensor * x, const ggml_tensor * gate, float * dst_f32,
        int rot_width, int ncols, int64_t tokens,
        ggml_cuda_siglu_q8_mode q8_mode, ggml_type wo_type, void * q8_dst);
