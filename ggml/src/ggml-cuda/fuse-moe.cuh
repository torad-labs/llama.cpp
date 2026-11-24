#pragma once

#include "common.cuh"

// Plan 17 section C, group 3: Inkling 2's routed-expert chain at decode rows, each piece one launch a layer.
// The router product keeps f32_cols' summation order (f32-few.cu), the selection topk_moe_cuda's loop, the routes
// mmvq_sort_routes' sort and the q8_1 copies quantize_q8_1's arithmetic, so every output is bit-identical to the
// launches the piece replaces. Off by default behind GGML_CUDA_MOE_FUSE.
//
// expert_cur is the normed expert input [n_embd, n_tokens] (n_tokens rows of n_embd), router the F32 [n_embd, 384] gate_inp and
// bias the F32 [384] selection bias or nullptr. The chain writes the logits (the router product's graph tensor), the
// ids and weights the top-k group would write, the routes sorted by the card's local experts, and the q8_1 copy of cur
// the gate/up product reads. routes holds n_tokens*n_used ints, q8 n_tokens*GGML_PAD(n_embd, MATRIX_ROW_PADDING)/32
// blocks and flag one unsigned that the launch leaves zero.
// cur may be a router-only projection of expert_cur.
// tma_mode skips q8_1 expert_cur quantization and routes sorting; local_ids_out optionally writes [n_used, n_tokens] local ids.
void ggml_cuda_op_moe_router_chain(ggml_backend_cuda_context & ctx, const ggml_tensor * router, const ggml_tensor * cur,
        const ggml_tensor * expert_cur, const ggml_tensor * bias, ggml_tensor * logits, ggml_tensor * ids, ggml_tensor * weights,
        int32_t * routes, block_q8_1 * q8, unsigned int * flag, int32_t first, int32_t count,
        float clamp_val, float scale_val, bool norm_eps, int32_t * local_ids_out = nullptr, bool tma_mode = false);

// The merged gate/up product of the routed experts with the SITU activation in its epilogue: w [ncols_x, 2*n_ff, n_experts]
// quantized, x the F32 gate/up input, ids [n_used, n_tokens] global, glu [n_ff, n_used, n_tokens] F32 the GLU's output.
// q8 is the chain's copy of x, q8_dst the copy of the GLU output made here for the down product. sorted: routes holds the
// sorted (local expert, route index) list the chain wrote; else the product reads ids and maps them itself.
void ggml_cuda_op_moe_gate_up_situ(ggml_backend_cuda_context & ctx, const ggml_tensor * w, const ggml_tensor * x,
        const ggml_tensor * ids, ggml_tensor * glu, const block_q8_1 * q8, block_q8_1 * q8_dst,
        const int32_t * routes, int32_t first, int32_t count, float beta, float lb, bool sorted);

// The down product of the routed experts and the weighted sum of the selected experts' outputs, one launch: w
// [ncols_x, n_embd, n_experts] quantized, q8 the copy of the GLU output, weights [1, n_used, n_tokens] F32, dst
// [n_embd, n_tokens] F32. The eight slots of a token share a block, so the sum runs in its slot order.
void ggml_cuda_op_moe_experts_weighted_sum(ggml_backend_cuda_context & ctx, const ggml_tensor * w, const ggml_tensor * x,
        const ggml_tensor * ids, const ggml_tensor * weights, const block_q8_1 * q8, ggml_tensor * dst,
        int32_t first, int32_t count);

// Inkling prompt MMQ: count routes once, then build tiles for the occupied experts.
bool ggml_cuda_moe_route_gate(ggml_backend_cuda_context & ctx, const ggml_tensor * w, const ggml_tensor * x,
        const ggml_tensor * ids, ggml_tensor * dst);
