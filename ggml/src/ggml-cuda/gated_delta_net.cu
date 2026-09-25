#include "gated_delta_net.cuh"
#include "chunk_gated_delta_net.cuh"
#include "ggml-cuda/common.cuh"

static __global__ void gdn_precompute_exp(const float * g, float * g_exp, int64_t n) {
    for (int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x; i < n;
         i += (int64_t) blockDim.x*gridDim.x) {
        g_exp[i] = expf(g[i]);
    }
}

// Writes element e of a state destination: f32, or a q8_0 cache row. In q8_0 one block is one warp-wide slice
// of a state column (QK8_0 == warp_size and every column starts on a block), so the block scale is a warp max;
// the formula is cpy's f32 -> q8_0 (quantize_f32_q8_0_block), and the cache holds the bytes the unfused
// gdn -> cpy pair would write.
template <int warp_size, bool STATE_Q8>
static __device__ __forceinline__ void gdn_store_state(void * dst, const int64_t e, const float v) {
    if constexpr (STATE_Q8) {
        // a q8_0 state store is one warp-wide block. The dispatch still instantiates it for 64-lane warps (HIP gfx8/9),
        // where the host declines the fusion (ggml_cuda_try_gdn_cache_fusion), so it compiles there to no device code.
        if constexpr (warp_size != QK8_0) {
            GGML_UNUSED_VARS(dst, e, v);
            NO_DEVICE_CODE;
        } else {
            const float amax = warp_reduce_max<warp_size>(fabsf(v));
            const float d    = amax / ((1 << 7) - 1);
            const float id   = d ? 1.0f/d : 0.0f;
            block_q8_0 * b   = (block_q8_0 *) dst + e / QK8_0;
            b->qs[e % QK8_0] = roundf(v*id);
            if (e % QK8_0 == 0) {
                b->d = d;
            }
        }
    } else {
        ((float *) dst)[e] = v;
    }
}

// Reads element e of a state source: f32, or a q8_0 cache row (the fused gather, ggml_cuda_try_gdn_gather_skip). q8_0
// is dequantized with get_rows' formula (dequantize_q8_0), so the kernel sees the values the skipped GET_ROWS wrote.
static __device__ __forceinline__ float gdn_load_state(const void * src, const int64_t e, const bool q8) {
    if (q8) {
        const block_q8_0 * b = (const block_q8_0 *) src + e / QK8_0;
        const float        d = b->d;
        float              v = b->qs[e % QK8_0];
        v *= d;
        return v;
    }
    return ((const float *) src)[e];
}

// RAW: beta and g arrive pre-activation (ggml_gated_delta_net_set_raw_gates); the kernel applies
// sigmoid(beta) and raw_a[h] * softplus(g + raw_dt_bias[h]) with the unary kernels' formulas.
// G_PRECOMPUTED: g already holds exp(g) (GB10 long-prompt path); only used with RAW == false.
// max_t > 0: n_tokens <= max_t, and every token's k, q, v, beta and g are loaded right after the state, in one memory
// round trip; the runtime loop (max_t == 0) cannot hoist a token's loads above the previous token's stores.
template <int S_v, bool KDA, bool keep_rs_t, bool RAW, bool G_PRECOMPUTED, bool STATE_Q8, int max_t = 0>
__global__ void __launch_bounds__((ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v) * 4, 2)
gated_delta_net_cuda(const float * q,
                                     const float * k,
                                     const float * v,
                                     const float * g,
                                     const float * beta,
                                     const float * raw_dt_bias,
                                     const float * raw_a,
                                     const void *  curr_state,
                                     float *       dst,
                                     void *        state,
                                     int64_t       H,
                                     int64_t       n_tokens,
                                     int64_t       n_seqs,
                                     int64_t       sq1,
                                     int64_t       sq2,
                                     int64_t       sq3,
                                     int64_t       sv1,
                                     int64_t       sv2,
                                     int64_t       sv3,
                                     int64_t       sb1,
                                     int64_t       sb2,
                                     int64_t       sb3,
                                     const uint3   neqk1_magic,
                                     const uint3   rq3_magic,
                                     float         scale,
                                     int64_t       state_slot_stride,
                                     int           K,
                                     int64_t       attn_seq_stride,
                                     const int32_t * s_ids,
                                     int64_t       s_row_stride,
                                     bool          s_q8) {
    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;
    // Each warp owns one or more columns, using warp-level primitives to reduce across rows.
    const int      lane     = threadIdx.x;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == GGML_CUDA_CC_DGX_SPARK
    constexpr int cols_per_warp = S_v == 128 && !KDA ? 4 : 1;
#else
    constexpr int cols_per_warp = 1;
#endif
    const int      col      = (blockIdx.z * blockDim.y + threadIdx.y) * cols_per_warp;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    float *       attn_data        = dst;

    // input state holds s0 only: [S_v, S_v, H, n_seqs] — seq stride is D = H * S_v * S_v; with the fused gather
    // (s_ids) it is the cache itself, this sequence's live state at row s_ids[sequence] (read after the PDL wait).
    // output state layout (per-slot D * n_seqs) — same per-(seq,head) offset as before.
    const int64_t state_out_offset     = (sequence * H + h_idx) * S_v * S_v; // in elements of the destination
    // attention rows of one sequence are attn_seq_stride apart: n_tokens * H * S_v, unless this launch
    // covers only the tail of each sequence (the chunked prefill path hands the last K-1 tokens here)
    attn_data += sequence * attn_seq_stride + h_idx * S_v;

    constexpr int warp_size = ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of warp_size");
    constexpr int rows_per_lane = (S_v + warp_size - 1) / warp_size;
    float         s_shard[cols_per_warp][rows_per_lane];
    // state is stored transposed: M[col][i] = S[i][col], row col is contiguous

    ggml_cuda_pdl_sync();
    const int64_t state_in_offset = (s_ids ? (int64_t) s_ids[sequence] * s_row_stride : (int64_t) sequence * H * S_v * S_v)
                                    + h_idx * S_v * S_v;
#pragma unroll
    for (int c = 0; c < cols_per_warp; ++c) {
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * warp_size + lane;
            s_shard[c][r] = gdn_load_state(curr_state, state_in_offset + (col + c) * S_v + i, s_q8);
        }
    }

    static_assert(max_t == 0 || !KDA, "the token prefetch is for the scalar gate");
    constexpr int n_pre = max_t > 0 ? max_t : 1;
    float k_pre[n_pre][rows_per_lane];
    float q_pre[n_pre][rows_per_lane];
    float v_pre[n_pre][cols_per_warp];
    float beta_pre[n_pre];
    float g_pre[n_pre];
    if constexpr (max_t > 0) {
#pragma unroll
        for (int t = 0; t < max_t; t++) {
            if (t < n_tokens) {
                const int64_t qk_offset = iq3 * sq3 + t * sq2 + iq1 * sq1;
                const int64_t gb_offset = sequence * sb3 + t * sb2 + h_idx * sb1;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    k_pre[t][r] = k[qk_offset + r * warp_size + lane];
                    q_pre[t][r] = q[qk_offset + r * warp_size + lane];
                }
#pragma unroll
                for (int c = 0; c < cols_per_warp; ++c) {
                    v_pre[t][c] = v[sequence * sv3 + t * sv2 + h_idx * sv1 + col + c];
                }
                beta_pre[t] = beta[gb_offset];
                g_pre[t]    = g[gb_offset];
            }
        }
    }

#pragma unroll
    for (int t = 0; t < (max_t > 0 ? max_t : n_tokens); t++) {
        if (max_t > 0 && t >= n_tokens) {
            break;
        }
        const float * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;

        const int64_t gb_offset = sequence * sb3 + t * sb2 + h_idx * sb1;
        const float * beta_t = beta + gb_offset;
        const float * g_t    = g    + gb_offset * (KDA ? S_v : 1);

        float beta_val = max_t > 0 ? beta_pre[t] : *beta_t;
        if constexpr (RAW) {
            beta_val = 1.0f / (1.0f + expf(-beta_val));
        }

        // Cache k and q in registers
        float k_reg[rows_per_lane];
        float q_reg[rows_per_lane];
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * warp_size + lane;
            k_reg[r] = max_t > 0 ? k_pre[t][r] : k_t[i];
            q_reg[r] = max_t > 0 ? q_pre[t][r] : q_t[i];
        }

        if constexpr (!KDA) {
            static_assert(!(RAW && G_PRECOMPUTED), "exp(g) precompute is only defined for activated gates");
            float g0 = max_t > 0 ? g_pre[t] : *g_t;
            if constexpr (RAW) {
                const float x = g0 + raw_dt_bias[h_idx];
                g0 = raw_a[h_idx] * ((x > 20.0f) ? x : logf(1.0f + expf(x)));
            }
            const float g_val = G_PRECOMPUTED ? g0 : expf(g0);

            // Each warp owns one or more columns and reuses the common q/k registers.
#pragma unroll
            for (int c = 0; c < cols_per_warp; ++c) {
                float kv_shard = 0.0f;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    kv_shard += s_shard[c][r] * k_reg[r];
                }
                float kv_col = warp_reduce_sum<warp_size>(kv_shard);

                float delta_col = ((max_t > 0 ? v_pre[t][c] : v_t[col + c]) - g_val * kv_col) * beta_val;

                float attn_partial = 0.0f;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    s_shard[c][r]  = g_val * s_shard[c][r] + k_reg[r] * delta_col;
                    attn_partial += s_shard[c][r] * q_reg[r];
                }

                float attn_col = warp_reduce_sum<warp_size>(attn_partial);

                if (lane == 0) {
                    attn_data[col + c] = attn_col * scale;
                }
            }
        } else {
            // kv[col] = sum_i g[i] * S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                kv_shard += expf(g_t[i]) * s_shard[0][r] * k_reg[r];
            }

            float kv_col = warp_reduce_sum<warp_size>(kv_shard);

            // delta[col] = (v[col] - kv[col]) * beta
            float delta_col = (v_t[col] - kv_col) * beta_val;

            // fused: S[i][col] = g[i] * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                s_shard[0][r]  = expf(g_t[i]) * s_shard[0][r] + k_reg[r] * delta_col;
                attn_partial += s_shard[0][r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<warp_size>(attn_partial);

            if (lane == 0) {
                attn_data[col] = attn_col * scale;
            }
        }

        attn_data += S_v * H;

        if constexpr (keep_rs_t) {
            // snapshot slot mapping: slot 0 = most recent state, slot s = s tokens back.
            // When n_tokens < K only slots 0..n_tokens-1 are written; older slots are caller-owned.
            const int target_slot = (int) n_tokens - 1 - t;
            if (target_slot >= 0 && target_slot < K) {
                const int64_t slot_offset = state_out_offset + target_slot * state_slot_stride;
#pragma unroll
                for (int c = 0; c < cols_per_warp; ++c) {
#pragma unroll
                    for (int r = 0; r < rows_per_lane; r++) {
                        const int i = r * warp_size + lane;
                        gdn_store_state<warp_size, STATE_Q8>(state, slot_offset + (col + c) * S_v + i, s_shard[c][r]);
                    }
                }
            }
        }

    }

    if constexpr (!keep_rs_t) {
#pragma unroll
        for (int c = 0; c < cols_per_warp; ++c) {
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                gdn_store_state<warp_size, STATE_Q8>(state, state_out_offset + (col + c) * S_v + i, s_shard[c][r]);
            }
        }
    }
}

// a decode or an MTP verify (up to 4 tokens) takes the token prefetch; GGML_CUDA_GDN_PREFETCH_LEGACY=1 keeps the loop
#define GGML_CUDA_GDN_PREFETCH_MAX_T 4
static bool ggml_cuda_gdn_prefetch_legacy() {
    static const bool legacy = [] {
        const char * s = getenv("GGML_CUDA_GDN_PREFETCH_LEGACY");
        return s != nullptr && atoi(s) != 0;
    }();
    return legacy;
}

template <bool KDA, bool keep_rs_t, bool RAW, bool G_PRECOMPUTED, bool STATE_Q8>
static void launch_gated_delta_net(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * rb_d, const float * ra_d, const void * s_d,
        float * dst_d, void * state_d,
        int64_t S_v,   int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1,   int64_t sq2, int64_t sq3,
        int64_t sv1,   int64_t sv2, int64_t sv3,
        int64_t sb1,   int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3,
        float scale, int64_t state_slot_stride, int K, int64_t attn_seq_stride,
        const int32_t * s_ids, int64_t s_row_stride, bool s_q8, cudaStream_t stream) {
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const int num_warps = 4;
    const int cols_per_warp = cc == GGML_CUDA_CC_DGX_SPARK && S_v == 128 && !KDA ? 4 : 1;
    dim3      grid_dims(H, n_seqs, (S_v + num_warps * cols_per_warp - 1) / (num_warps * cols_per_warp));
    dim3      block_dims(warp_size <= S_v ? warp_size : S_v, num_warps, 1);

    const uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const uint3 rq3_magic   = init_fastdiv_values(rq3);

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);
    switch (S_v) {
        case 16:
            if constexpr (!STATE_Q8) { // a q8_0 block is 32 wide: a 16-lane warp cannot own one
                ggml_cuda_kernel_launch(gated_delta_net_cuda<16, KDA, keep_rs_t, RAW, G_PRECOMPUTED, false>, launch_params,
                    q_d, k_d, v_d, g_d, b_d, rb_d, ra_d, s_d, dst_d, state_d, H,
                    n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                    sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, attn_seq_stride, s_ids, s_row_stride, s_q8);
                break;
            }
            GGML_ABORT("a q8_0 recurrent state needs S_v >= 32");
        case 32:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<32, KDA, keep_rs_t, RAW, G_PRECOMPUTED, STATE_Q8>, launch_params,
                q_d, k_d, v_d, g_d, b_d, rb_d, ra_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, attn_seq_stride, s_ids, s_row_stride, s_q8);
            break;
        case 64: {
            ggml_cuda_kernel_launch(gated_delta_net_cuda<64, KDA, keep_rs_t, RAW, G_PRECOMPUTED, STATE_Q8>, launch_params,
                q_d, k_d, v_d, g_d, b_d, rb_d, ra_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, attn_seq_stride, s_ids, s_row_stride, s_q8);
            break;
        }
        case 128: {
            if constexpr (!KDA) {
                if (n_tokens <= GGML_CUDA_GDN_PREFETCH_MAX_T && !ggml_cuda_gdn_prefetch_legacy()) {
                    ggml_cuda_kernel_launch(gated_delta_net_cuda<128, KDA, keep_rs_t, RAW, G_PRECOMPUTED, STATE_Q8,
                        GGML_CUDA_GDN_PREFETCH_MAX_T>, launch_params,
                        q_d, k_d, v_d, g_d, b_d, rb_d, ra_d, s_d, dst_d, state_d, H,
                        n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                        sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, attn_seq_stride, s_ids, s_row_stride, s_q8);
                    break;
                }
            }
            ggml_cuda_kernel_launch(gated_delta_net_cuda<128, KDA, keep_rs_t, RAW, G_PRECOMPUTED, STATE_Q8>, launch_params,
                q_d, k_d, v_d, g_d, b_d, rb_d, ra_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, attn_seq_stride, s_ids, s_row_stride, s_q8);
            break;
        }
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

// Chunked prefill (chunk_gated_delta_net.cu) where GGML_CUDA_GDN_CHUNKED is unset: on, since on
// Ternary Bonsai 2 27B its KL against the recurrent kernel (mean 0.00148, same top p 98.41 %) is at
// the bar of the int8 Q.K flash attention already served (0.00150 / 98.3 %).
static constexpr bool gdn_chunked_default = true;

// GGML_CUDA_GDN_CHUNKED=0 keeps every GATED_DELTA_NET on the recurrent kernel; =1 takes the chunked
// prefill path wherever the op is eligible.
static bool ggml_cuda_gdn_chunked_enabled() {
    static const bool enabled = [] {
        const char * s = getenv("GGML_CUDA_GDN_CHUNKED");
        return s == nullptr ? gdn_chunked_default : atoi(s) != 0;
    }();
    return enabled;
}

bool ggml_cuda_gdn_chunked_shape_eligible(const ggml_tensor * dst) {
    if (dst->op != GGML_OP_GATED_DELTA_NET || !ggml_cuda_gdn_chunked_enabled()) {
        return false;
    }

    const ggml_tensor * src_q     = dst->src[0];
    const ggml_tensor * src_k     = dst->src[1];
    const ggml_tensor * src_v     = dst->src[2];
    const ggml_tensor * src_g     = dst->src[3];
    const ggml_tensor * src_beta  = dst->src[4];
    const ggml_tensor * src_state = dst->src[5];

    const int64_t S_v      = src_v->ne[0];
    const int64_t n_tokens = src_v->ne[2];
    const int64_t K        = ggml_get_op_params_i32(dst, 0);
    const bool    kda      = src_g->ne[0] == S_v;

    // - scalar gate (not KDA), not the rows-indexed state read (src[6]), all f32
    // - 128-wide heads, q and k with one head count that divides the v-head count, no broadcast over sequences
    // - q/k/v rows contiguous with any head/token/seq stride (the views qwen35 takes of the conv output are
    //   read in place; q and k share strides), g/beta/state contiguous
    // - n_tokens >= 128; with K > 1 snapshot slots the last K-1 tokens go to the recurrent kernel
    return !kda && dst->src[6] == nullptr
        && dst->type == GGML_TYPE_F32 && src_q->type == GGML_TYPE_F32 && src_k->type == GGML_TYPE_F32
        && src_v->type == GGML_TYPE_F32 && src_g->type == GGML_TYPE_F32 && src_beta->type == GGML_TYPE_F32
        && src_state->type == GGML_TYPE_F32
        && src_q->ne[0] == 128 && S_v == 128
        && src_k->ne[1] == src_q->ne[1] && src_v->ne[1] % src_q->ne[1] == 0
        && src_q->ne[3] == src_v->ne[3]
        && n_tokens >= 128 && K >= 1 && K <= n_tokens
        && ggml_is_contiguous_rows(src_q) && ggml_are_same_stride(src_q, src_k) && ggml_is_contiguous_rows(src_v)
        && ggml_is_contiguous(src_g) && ggml_is_contiguous(src_beta) && ggml_is_contiguous(src_state);
}

bool ggml_cuda_should_use_chunked_gdn(const ggml_tensor * dst) {
    // fp16 WMMA with fp32 accumulation: NVIDIA Ampere+ (sm_80, sm_86, sm_89, sm_90, sm_100, sm_120)
    return ggml_cuda_gdn_chunked_shape_eligible(dst) &&
           ampere_mma_available(ggml_cuda_info().devices[ggml_cuda_get_device()].cc);
}

static void ggml_cuda_op_gated_delta_net_impl(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_cuda_gated_delta_net_fused_cache * cache) {
    ggml_tensor * src_q     = dst->src[0];
    ggml_tensor * src_k     = dst->src[1];
    ggml_tensor * src_v     = dst->src[2];
    ggml_tensor * src_g     = dst->src[3];
    ggml_tensor * src_beta  = dst->src[4];
    ggml_tensor * src_state = dst->src[5];

    GGML_TENSOR_LOCALS(int64_t, neq, src_q, ne);
    GGML_TENSOR_LOCALS(size_t , nbq, src_q, nb);
    GGML_TENSOR_LOCALS(int64_t, nek, src_k, ne);
    GGML_TENSOR_LOCALS(size_t , nbk, src_k, nb);
    GGML_TENSOR_LOCALS(int64_t, nev, src_v, ne);
    GGML_TENSOR_LOCALS(size_t,  nbv, src_v, nb);
    GGML_TENSOR_LOCALS(size_t,  nbb, src_beta, nb);

    const int64_t S_v      = nev0;
    const int64_t H        = nev1;
    const int64_t n_tokens = nev2;
    const int64_t n_seqs   = nev3;

    const bool kda = (src_g->ne[0] == S_v);

    GGML_ASSERT(neq1 == nek1);
    const int64_t neqk1 = neq1;

    const int64_t rq3 = nev3 / neq3;

    const float * q_d = (const float *) src_q->data;
    const float * k_d = (const float *) src_k->data;
    const float * v_d = (const float *) src_v->data;
    const float * g_d = (const float *) src_g->data;
    const float * b_d = (const float *) src_beta->data;

    const float * s_d   = (const float *) src_state->data;
    float *       dst_d = (float *) dst->data;

    // the fused state gather, registered for this node by this context's graph evaluator (ggml_cuda_try_gdn_gather_skip):
    // the state comes from the cache rows, not from src_state (the skipped GET_ROWS's temp, never written)
    const ggml_cuda_gated_delta_net_gather * gather = ctx.gdn_gathers().find(dst);
    const void *    s_in         = gather ? gather->base : (const void *) s_d;
    const int32_t * s_ids        = gather ? gather->ids : nullptr;
    const int64_t   s_row_stride = gather ? gather->row_stride : 0;
    const bool      s_q8         = gather != nullptr && gather->q8_0;

    GGML_ASSERT(ggml_is_contiguous_rows(src_q));
    GGML_ASSERT(ggml_is_contiguous_rows(src_k));
    GGML_ASSERT(ggml_is_contiguous_rows(src_v));
    GGML_ASSERT(ggml_are_same_stride(src_q, src_k));
    GGML_ASSERT(src_g->ne[0] == 1 || kda);
    GGML_ASSERT(ggml_is_contiguous(src_g));
    GGML_ASSERT(ggml_is_contiguous(src_beta));
    GGML_ASSERT(ggml_is_contiguous(src_state));

    // strides in floats (beta strides used for both g and beta offset computation)
    const int64_t sq1 = nbq1 / sizeof(float);
    const int64_t sq2 = nbq2 / sizeof(float);
    const int64_t sq3 = nbq3 / sizeof(float);
    const int64_t sv1 = nbv1 / sizeof(float);
    const int64_t sv2 = nbv2 / sizeof(float);
    const int64_t sv3 = nbv3 / sizeof(float);
    const int64_t sb1 = nbb1 / sizeof(float);
    const int64_t sb2 = nbb2 / sizeof(float);
    const int64_t sb3 = nbb3 / sizeof(float);

    const float scale = 1.0f / sqrtf((float) S_v);

    cudaStream_t stream = ctx.stream();

    // K (snapshot slot count) is an op param; state holds s0 only [S_v, S_v, H, n_seqs].
    const int K = ggml_get_op_params_i32(dst, 0);
    const bool keep_rs = K > 1;

    // recurrent state -> gdn_out tail (after attention scores), or the cache when fusing (f32 or q8_0 rows;
    // strides in elements either way)
    void *  state_d           = dst_d + S_v * H * n_tokens * n_seqs;
    int64_t state_slot_stride = S_v * S_v * H * n_seqs;
    bool    state_q8          = false;
    if (cache != nullptr) {
        state_d           = cache->data;
        state_slot_stride = cache->slot_stride;
        state_q8          = cache->q8_0;
    }

    const bool    raw  = ggml_get_op_params_i32(dst, 1) != 0;
    const float * rb_d = raw ? (const float *) dst->src[7]->data : nullptr;
    const float * ra_d = raw ? (const float *) dst->src[8]->data : nullptr;
    GGML_ASSERT(!(raw && kda)); // raw gates are defined for the scalar gate only

    // Chunked prefill: tokens [0, n_tokens - (K-1)) of every sequence run on the tensor-core chunk
    // pipeline, whose final state is rollback slot K-1 (slot 0 when K == 1); the last K-1 tokens run on
    // the recurrent kernel, which starts from that slot and writes slots K-2..0 as it always does. So
    // every snapshot the recurrent kernel alone would write is written, to the same place (dst tail or
    // the fused cache).
    if (ggml_cuda_should_use_chunked_gdn(dst)) {
        GGML_ASSERT(!state_q8); // the chunk pipeline writes f32 (ggml_cuda_try_gdn_cache_fusion keeps q8_0 off it)
        GGML_ASSERT(!gather);   // and reads a gathered s0 (ggml_cuda_try_gdn_gather_skip keeps the GET_ROWS for it)
        const int64_t n_tail      = K - 1;
        const int64_t n_chunked   = n_tokens - n_tail;
        float *       chunk_state = (float *) state_d + n_tail * state_slot_stride;

        ggml_cuda_gdn_chunked_args args = {};
        args.q           = q_d;
        args.k           = k_d;
        args.v           = v_d;
        args.g           = g_d;
        args.beta        = b_d;
        args.raw_dt_bias = rb_d;
        args.raw_a       = ra_d;
        args.state_in    = s_d;
        args.state_out   = chunk_state;
        args.out         = dst_d;
        args.k_dim       = neq0;
        args.v_dim       = S_v;
        args.H           = H;
        args.num_k_heads = neqk1;
        args.n_tokens    = n_chunked;
        args.n_seqs      = n_seqs;
        args.sq1 = sq1; args.sq2 = sq2; args.sq3 = sq3;
        args.sv1 = sv1; args.sv2 = sv2; args.sv3 = sv3;
        args.sb1 = sb1; args.sb2 = sb2; args.sb3 = sb3;
        args.so2         = S_v * H;
        args.so3         = S_v * H * n_tokens;
        args.scale       = scale;
        ggml_cuda_gdn_chunked_launch(ctx, dst, args);

        if (n_tail > 0) {
            const int64_t t0 = n_chunked;
#define GDN_TAIL_LAUNCH(RAW_)                                                                              \
            launch_gated_delta_net<false, true, RAW_, false, false>(q_d + t0 * sq2, k_d + t0 * sq2, v_d + t0 * sv2, \
                g_d + t0 * sb2, b_d + t0 * sb2, rb_d, ra_d, chunk_state, dst_d + t0 * S_v * H, state_d,        \
                S_v, H, n_tail, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,                                           \
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, S_v * H * n_tokens, nullptr, 0, false, stream)
            if (raw) { GDN_TAIL_LAUNCH(true); } else { GDN_TAIL_LAUNCH(false); }
#undef GDN_TAIL_LAUNCH
        }
        return;
    }

    // GB10 long-prompt path: exp(g) once per (token, head) instead of once per column-warp.
    // Only for activated gates; with raw gates (#165) the activation happens inside the kernel.
    ggml_cuda_pool_alloc<float> g_exp_alloc(ctx.pool());
    bool g_precomputed = false;
    if (!kda && !raw && !state_q8 && S_v == 128 && n_tokens >= 32 &&
            ggml_cuda_info().devices[ggml_cuda_get_device()].cc == GGML_CUDA_CC_DGX_SPARK) {
        const int64_t n_g = ggml_nelements(src_g);
        g_exp_alloc.alloc(n_g);
        const int block = 256;
        const int grid = std::min<int64_t>((n_g + block - 1)/block, 4096);
        gdn_precompute_exp<<<grid, block, 0, stream>>>(g_d, g_exp_alloc.ptr, n_g);
        g_d = g_exp_alloc.ptr;
        g_precomputed = true;
    }

#define GDN_LAUNCH(KDA_, KEEP_, RAW_, PRE_, Q8_)                                                  \
    launch_gated_delta_net<KDA_, KEEP_, RAW_, PRE_, Q8_>(q_d, k_d, v_d, g_d, b_d, rb_d, ra_d, s_in, dst_d, state_d, \
        S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,                                    \
        sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, S_v * H * n_tokens,                \
        s_ids, s_row_stride, s_q8, stream)

    // a q8_0 cache is fused for the scalar gate only (ggml_cuda_try_gdn_cache_fusion)
    GGML_ASSERT(!(state_q8 && kda));
    if (kda) {
        if (keep_rs) { GDN_LAUNCH(true,  true,  false, false, false); } else { GDN_LAUNCH(true,  false, false, false, false); }
    } else if (raw) {
        if (state_q8) {
            if (keep_rs) { GDN_LAUNCH(false, true,  true,  false, true);  } else { GDN_LAUNCH(false, false, true,  false, true);  }
        } else {
            if (keep_rs) { GDN_LAUNCH(false, true,  true,  false, false); } else { GDN_LAUNCH(false, false, true,  false, false); }
        }
    } else {
        if (g_precomputed) {
            if (keep_rs) { GDN_LAUNCH(false, true,  false, true,  false); } else { GDN_LAUNCH(false, false, false, true,  false); }
        } else if (state_q8) {
            if (keep_rs) { GDN_LAUNCH(false, true,  false, false, true);  } else { GDN_LAUNCH(false, false, false, false, true);  }
        } else {
            if (keep_rs) { GDN_LAUNCH(false, true,  false, false, false); } else { GDN_LAUNCH(false, false, false, false, false); }
        }
    }
#undef GDN_LAUNCH
}

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, nullptr);
}

void ggml_cuda_op_gated_delta_net_fused_cache(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_cuda_gated_delta_net_fused_cache cache) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, &cache);
}
