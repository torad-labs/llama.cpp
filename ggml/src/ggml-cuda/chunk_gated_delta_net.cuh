#pragma once

#include "common.cuh"
#include "gated_delta_net.cuh"
#include "ggml.h"

// One chunked pass over tokens [0, n_tokens) of every sequence; all pointers already point at the
// first token of sequence 0. Strides are in floats. q and k share their strides.
struct ggml_cuda_gdn_chunked_args {
    const float * q;
    const float * k;
    const float * v;
    const float * g;
    const float * beta;
    const float * raw_dt_bias; // non-null: beta / g are pre-activation (ggml_gated_delta_net_set_raw_gates)
    const float * raw_a;
    const float * state_in;    // [S, S, H, n_seqs], per (seq, head) [v][k]
    float *       state_out;   // final state after n_tokens tokens, same layout as state_in
    float *       out;         // attention output of token 0
    int64_t k_dim, v_dim, H, num_k_heads, n_tokens, n_seqs;
    int64_t sq1, sq2, sq3;     // q/k: head, token, seq
    int64_t sv1, sv2, sv3;     // v
    int64_t sb1, sb2, sb3;     // g/beta
    int64_t so2, so3;          // output: token, seq (head stride is v_dim)
    float   scale;
};

void ggml_cuda_gdn_chunked_launch(ggml_backend_cuda_context & ctx, const ggml_tensor * dst,
                                  const ggml_cuda_gdn_chunked_args & args);

// Chunked-GDN scratch, carved out of the tail of dst's own allocation. Two reasons:
//  1. The graph allocator sizes every tensor through ggml_backend_buft_get_alloc_size, so scratch
//     that lives in dst's buffer is visible to llama_params_fit (--fit). A context-owned cudaMalloc
//     (or a ggml_cuda_pool_alloc) is invisible to that projection, which would under-estimate VRAM
//     by the full scratch size.
//  2. The address is a fixed offset from dst->data, so it is stable across CUDA-graph capture and
//     replay without needing a separate persistent allocation to be pre-sized before capture.
struct ggml_cuda_gdn_chunked_scratch {
    float *   v_corr;
    float *   k_cumdecay;
    float *   g_cum;
    float *   qk;
    uintptr_t end;
};

// Pure function of the tensor graph, so allocation time and execution time cannot disagree.
ggml_cuda_gdn_chunked_scratch ggml_cuda_gdn_get_chunked_scratch(const ggml_tensor * dst);

// ggml_nbytes(dst) plus the scratch above, or just ggml_nbytes(dst) when the shape is ineligible.
size_t ggml_cuda_gdn_get_alloc_size(const ggml_tensor * dst);
