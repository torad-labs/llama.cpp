#include "models.h"
#include "inkling2-attention.h"
#include "ggml-alloc.h"
#include "../llama-memory-hybrid-iswa.h"

#include <algorithm>
#include <array>

// LLAMA_LOOP_DIAG=1, a diagnostic off by default: per-token statistics of the loop's controllers (mean |gain|,
// |forget|, |change| and |input|) and of each later pass's K/V rows against pass 1's and the previous pass's (cosine,
// relative L2), as graph nodes named loop_diag_*, which the context prints as they are computed (llama-context.cpp).
// Off, the graph is unchanged.
static bool loop_diag_on() {
    static const bool on = ggml_env_switch("LLAMA_LOOP_DIAG");
    return on;
}
// the statistics of the last control() call, named by build_layer; each physical layer's pass-1 and latest K and V
static thread_local ggml_tensor * loop_diag_gates = nullptr;
static thread_local std::map<int, std::array<ggml_tensor *, 4>> loop_diag_kv;

void llama_model_inkling2::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_ATTENTION_QK_CLIP,           hparams.f_qk_clip);
    ml.get_key(LLM_KV_ATTN_RES_BLOCK_SIZE,         hparams.attn_res_block_size);
    ml.get_key(LLM_KV_SHORTCONV_L_CACHE,           hparams.n_shortconv_l_cache);

    // relative-position bias, and the log-scaled positions of the full-attention layers
    ml.get_key(LLM_KV_ATTENTION_RELATIVE_LENGTH, hparams.n_relative);
    ml.get_key(LLM_KV_ATTENTION_RELATIVE_EXTENT, hparams.n_relative_extent);
    ml.get_key(LLM_KV_ATTENTION_RELATIVE_CLIP,   hparams.f_relative_clip);
    ml.get_key(LLM_KV_ATTENTION_LOG_SCALE_FLOOR, hparams.n_log_scale_floor);
    ml.get_key(LLM_KV_ATTENTION_LOG_SCALE_ALPHA, hparams.f_log_scale_alpha);

    // the embedding and output tensors are padded past the vocabulary
    ml.get_key(LLM_KV_VOCAB_SIZE,          hparams.n_vocab_padded);
    ml.get_key(LLM_KV_VOCAB_SIZE_UNPADDED, hparams.n_vocab_unpadded);

    // MoE
    ml.get_key_or_arr(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp_arr, hparams.n_layer_all);

    ml.get_key(LLM_KV_ACTIVATION_SITU_BETA,        hparams.situ_beta);
    ml.get_key(LLM_KV_ACTIVATION_SITU_LINEAR_BETA, hparams.situ_linear_beta);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,        hparams.expert_weights_scale, false);

    hparams.f_final_logit_softcapping = 0.0f;
    ml.get_key(LLM_KV_FINAL_LOGIT_SOFTCAPPING, hparams.f_final_logit_softcapping, false);

    // sliding window
    ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW,         hparams.n_swa);
    ml.get_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, hparams.is_swa_impl);

    uint32_t n_pattern = 0;
    ml.get_arr_n(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, n_pattern);

    if (n_pattern != hparams.n_layer() || !hparams.n_relative || !hparams.n_relative_extent || !hparams.n_swa ||
        !hparams.attn_res_block_size || !hparams.n_log_scale_floor || hparams.n_shortconv_l_cache < 2 ||
        hparams.n_shortconv_l_cache > 64 || hparams.n_relative_extent > INT32_MAX || hparams.n_swa > INT32_MAX ||
        !hparams.n_vocab_unpadded || hparams.n_vocab_unpadded > hparams.n_vocab_padded) {
        throw std::runtime_error("inkling2: invalid dimensions, vocabulary, or layer pattern");
    }

    for (float value : {hparams.f_norm_rms_eps, hparams.f_qk_clip, hparams.f_relative_clip, hparams.situ_beta, hparams.situ_linear_beta}) {
        if (!std::isfinite(value) || value <= 0.0f) {
            throw std::runtime_error("inkling2: normalization, clipping and scaling parameters must be finite and positive");
        }
    }

    if (!std::isfinite(hparams.f_log_scale_alpha) || hparams.f_log_scale_alpha < 0.0f) {
        throw std::runtime_error("inkling2: invalid logarithmic scale");
    }

    hparams.swa_type                = LLAMA_SWA_TYPE_STANDARD;
    hparams.expert_router_f32       = true;
    hparams.expert_weights_norm_eps = 1e-20f;

    // no RoPE, the relative-position bias carries the positions
    hparams.rope_pattern.fill(false);

    // The two hidden streams' convolution state is the S state, a row apart from K's and V's, so that -sm tensor can
    // hold it whole on every card while K and V split by heads (llama_meta_device_get_split_state).
    // LLAMA_INKLING2_HIDDEN_CONV_LEGACY=1: all four streams in one row, the hidden ones split by channel as K and V are
    const bool hidden_apart = !ggml_env_switch("LLAMA_INKLING2_HIDDEN_CONV_LEGACY");
    // the convolution state keeps K and V of the widest layer, and the attention output and the FFN output
    uint64_t n_channels_max = 0;

    for (uint32_t il = 0; il < hparams.n_layer(); ++il) {
        if (!hparams.n_head_kv(il) || hparams.n_head(il) % hparams.n_head_kv(il) ||
            hparams.n_embd_head_k(il) != hparams.n_embd_head_v(il) || !hparams.n_expert_used(il) ||
            hparams.n_expert_used(il) > hparams.n_expert || !hparams.n_ff_exp(il)) {
            throw std::runtime_error("inkling2: invalid attention or expert dimensions");
        }

        n_channels_max = std::max<uint64_t>(n_channels_max, 2ull * hparams.n_embd_k_gqa(il) + (hidden_apart ? 0 : 2ull * hparams.n_embd));
    }

    const uint64_t hidden_channels = hidden_apart ? 2ull * hparams.n_embd : 0;
    if (std::max(n_channels_max, hidden_channels) * (hparams.n_shortconv_l_cache - 1) > UINT32_MAX) {
        throw std::runtime_error("inkling2: convolution state is too large");
    }

    hparams.n_sconv_channels = n_channels_max;
    hparams.n_sconv_hidden = hidden_channels;

    type = LLM_TYPE_UNKNOWN;
}

bool llama_model_inkling2::load_tensors(llama_model_loader & ml) {
    if (!llama_model_base::load_tensors(ml)) {
        return false;
    }
    if (!ggml_env_switch("LLAMA_INKLING2_ROUTER_BAKE") || ml.no_alloc) {
        return true;
    }

    baked_routers.resize(n_layer_phys());
    for (int il = 0; il < n_layer_phys(); ++il) {
        const auto * src = layers[il].ffn_gate_inp;
        if (!src || src->type != GGML_TYPE_F32 || hdmd.rot.count(src)) {
            continue;
        }

        const auto buft = src->buffer ? ggml_backend_buffer_get_type(src->buffer) : nullptr;
        const auto dev = buft ? ggml_backend_buft_get_device(buft) : nullptr;
        const bool repacked_cpu = buft && !ggml_backend_buft_is_host(buft) &&
            (!dev || ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU);
        if (!buft || src->extra || repacked_cpu || !ggml_is_contiguous(src)) {
            LLAMA_LOG_WARN("%s: keeping the unbaked router %s (no readable F32 rows)\n", __func__, src->name);
            continue;
        }

        router_bake baked;
        baked.ctx.reset(ggml_init({2*ggml_tensor_overhead(), nullptr, true}));
        if (!baked.ctx) {
            LLAMA_LOG_WARN("%s: keeping the unbaked router %s (no tensor context)\n", __func__, src->name);
            continue;
        }
        baked.weight = ggml_dup_tensor(baked.ctx.get(), src);
        baked.signs = ggml_new_tensor_1d(baked.ctx.get(), GGML_TYPE_F32, src->ne[0]);
        ggml_set_name(baked.weight, src->name);
        ggml_format_name(baked.signs, "blk.%d.ffn_gate_inp.scale", il);
        baked.buffer.reset(ggml_backend_alloc_ctx_tensors_from_buft(baked.ctx.get(), buft));
        if (!baked.buffer) {
            LLAMA_LOG_WARN("%s: keeping the unbaked router %s (no backend buffer)\n", __func__, src->name);
            continue;
        }
        ggml_backend_buffer_set_usage(baked.buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

        // Paired block signs are orthogonal and preserve the product and its accumulation order.
        std::vector<float> signs(src->ne[0]);
        const int64_t width = src->ne[0];
        constexpr int64_t chunk_rows = 64;
        std::vector<float> rows(width*std::min(chunk_rows, src->ne[1]));
        for (int64_t j = 0; j < src->ne[0]; ++j) {
            signs[j] = (j/16) % 2 ? -1.0f : 1.0f;
        }
        ggml_backend_tensor_set(baked.signs, signs.data(), 0, signs.size()*sizeof(float));
        for (int64_t expert = 0; expert < src->ne[1]; expert += chunk_rows) {
            const int64_t count = std::min(chunk_rows, src->ne[1] - expert);
            const size_t bytes = count*src->nb[1];
            ggml_backend_tensor_get(src, rows.data(), expert*src->nb[1], bytes);
            for (int64_t r = 0; r < count; ++r) {
                for (int64_t j = 0; j < width; ++j) {
                    rows[r*width + j] *= signs[j];
                }
            }
            ggml_backend_tensor_set(baked.weight, rows.data(), expert*baked.weight->nb[1], bytes);
        }
        if (const auto it = prec_policy.prec_acc.find(src); it != prec_policy.prec_acc.end()) {
            prec_policy.prec_acc.emplace(baked.weight, it->second);
        }
        if (const auto it = prec_policy.prec_src1.find(src); it != prec_policy.prec_src1.end()) {
            prec_policy.prec_src1.emplace(baked.weight, it->second);
        }
        baked_routers[il] = std::move(baked);
    }
    return true;
}

void llama_model_inkling2::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    if (hparams.n_vocab_unpadded != vocab.n_tokens()) {
        throw std::runtime_error("inkling2: tokenizer size must equal the unpadded vocabulary size");
    }

    const int64_t n_vocab_padded = hparams.n_vocab_padded;
    const int64_t n_rel          = hparams.n_relative;
    const int64_t n_conv         = hparams.n_shortconv_l_cache;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD,      "weight"),    {n_embd, n_vocab_padded}, 0);
    tok_norm = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD_NORM, "weight", 0), {n_embd}, 0);

    // output
    output           = create_tensor(tn(LLM_TENSOR_OUTPUT,           "weight"), {n_embd, n_vocab_padded}, 0);
    output_norm      = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM,      "weight"), {n_embd}, 0);
    output_res_query = create_tensor(tn(LLM_TENSOR_OUTPUT_RES_QUERY, "weight"), {n_embd}, 0);

    for (int i = 0; i < n_layer_phys(); ++i) {
        auto & layer = layers[i];

        const int64_t n_head      = hparams.n_head(i);
        const int64_t n_head_kv   = hparams.n_head_kv(i);
        const int64_t n_embd_head = hparams.n_embd_head_k(i);
        const int64_t n_ff_exp    = hparams.n_ff_exp(i);
        const int64_t n_extent    = hparams.is_swa(i) ? hparams.n_swa : hparams.n_relative_extent;

        // attention
        layer.wq        = create_tensor(tn(LLM_TENSOR_ATTN_Q,    "weight", i), {n_embd, n_embd_head*n_head}, 0);
        layer.wk        = create_tensor(tn(LLM_TENSOR_ATTN_K,    "weight", i), {n_embd, n_embd_head*n_head_kv}, 0);
        layer.wv        = create_tensor(tn(LLM_TENSOR_ATTN_V,    "weight", i), {n_embd, n_embd_head*n_head_kv}, 0);
        layer.wo        = create_tensor(tn(LLM_TENSOR_ATTN_OUT,  "weight", i), {n_embd_head*n_head, n_embd}, 0);
        layer.wqkv_gate = create_tensor(tn(LLM_TENSOR_ATTN_GATE, "weight", i), {n_embd, n_embd_head*n_head}, 0);

        layer.attn_q_clip    = create_tensor(tn(LLM_TENSOR_ATTN_Q_CLIP,    "weight", i), {n_embd_head, n_head}, 0);
        layer.attn_sink_proj = create_tensor(tn(LLM_TENSOR_ATTN_SINK_PROJ, "weight", i), {n_embd, n_head}, 0);

        // relative-position bias
        layer.attn_rel      = create_tensor(tn(LLM_TENSOR_ATTN_REL,      "weight", i), {n_embd, n_rel*n_head}, 0);
        layer.attn_rel_bias = create_tensor(tn(LLM_TENSOR_ATTN_REL_BIAS, "weight", i), {n_rel, n_extent}, 0);
        layer.attn_rel_clip = create_tensor(tn(LLM_TENSOR_ATTN_REL_CLIP, "weight", i), {n_rel, n_head}, 0);

        // short convolutions
        layer.attn_k_conv    = create_tensor(tn(LLM_TENSOR_ATTN_K_CONV,    "weight", i), {n_conv, n_embd_head*n_head_kv}, 0);
        layer.attn_v_conv    = create_tensor(tn(LLM_TENSOR_ATTN_V_CONV,    "weight", i), {n_conv, n_embd_head*n_head_kv}, 0);
        layer.attn_post_conv = create_tensor(tn(LLM_TENSOR_ATTN_POST_CONV, "weight", i), {n_conv, n_embd}, 0);
        layer.ffn_post_conv  = create_tensor(tn(LLM_TENSOR_FFN_POST_CONV,  "weight", i), {n_conv, n_embd}, 0);

        if (hparams.is_swa(i)) {
            layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);
        } else {
            // full-attention layers: separate Q, K and V input norms, and input-dependent taps for the K and V convolutions
            layer.attn_q_inp_norm  = create_tensor(tn(LLM_TENSOR_ATTN_Q_INP_NORM,  "weight", i), {n_embd}, 0);
            layer.attn_k_inp_norm  = create_tensor(tn(LLM_TENSOR_ATTN_K_INP_NORM,  "weight", i), {n_embd}, 0);
            layer.attn_v_inp_norm  = create_tensor(tn(LLM_TENSOR_ATTN_V_INP_NORM,  "weight", i), {n_embd}, 0);
            layer.attn_k_conv_proj = create_tensor(tn(LLM_TENSOR_ATTN_K_CONV_PROJ, "weight", i), {n_embd, n_conv*n_head_kv}, 0);
            layer.attn_v_conv_proj = create_tensor(tn(LLM_TENSOR_ATTN_V_CONV_PROJ, "weight", i), {n_embd, n_conv*n_head_kv}, 0);
        }

        // block attention residuals
        if (i > 0) {
            layer.attn_res_query = create_tensor(tn(LLM_TENSOR_ATTN_RES_QUERY, "weight", i), {n_embd, hparams.is_swa(i) ? 1 : 3}, 0);
        }
        layer.ffn_res_query = create_tensor(tn(LLM_TENSOR_FFN_RES_QUERY, "weight", i), {n_embd}, 0);

        // MoE
        layer.ffn_norm         = create_tensor(tn(LLM_TENSOR_FFN_NORM,         "weight", i), {n_embd}, 0);
        layer.ffn_gate_inp     = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,     "weight", i), {n_embd, n_expert}, 0);
        layer.ffn_exp_probs_b  = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B,  "bias",   i), {n_expert}, TENSOR_NOT_REQUIRED);
        layer.ffn_gate_up_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_UP_EXPS, "weight", i), {n_embd, 2*n_ff_exp, n_expert}, 0);
        layer.ffn_down_exps    = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS,    "weight", i), {n_ff_exp, n_embd, n_expert}, 0);
    }
}

std::unique_ptr<llm_graph_context> llama_model_inkling2::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(params);
}

// the shapes a decode step's mask is classed by, read once (tunables.toml)
static const inkling2_mask_shapes & mask_shapes() {
    static const inkling2_mask_shapes shapes = {
        (uint32_t) ggml_env_int("LLAMA_INKLING2_SHORT_ROWS_CELLS"),
        (uint32_t) ggml_env_int("LLAMA_INKLING2_BOUNDED_ROWS_MIN"),
        (uint32_t) ggml_env_int("LLAMA_INKLING2_BOUNDED_ROWS_MAX"),
        (uint32_t) ggml_env_int("LLAMA_INKLING2_BOUNDED_POOL_MIN"),
        (uint32_t) ggml_env_int("LLAMA_INKLING2_BOUNDED_POOL_SHARE"),
        (uint32_t) ggml_env_int("LLAMA_INKLING2_MEAN_ROWS"),
        (uint32_t) ggml_env_int("LLAMA_INKLING2_MEAN_POOL_MIN"),
        (uint32_t) ggml_env_int("LLAMA_INKLING2_MEAN_POOL_SHARE"),
        (uint32_t) ggml_env_int("LLAMA_INKLING2_DECODE_POOL_READS"),
    };
    return shapes;
}

// without cparams.rel_decode_rows (LLAMA_INKLING2_DECODE_CLASS_LEGACY=1) a decode step of fewer than 8 rows, a drafted step
// with a long sequence and a low mean anywhere but at LLAMA_INKLING2_MEAN_ROWS rows of F16 K/V
// (LLAMA_INKLING2_MEAN_MASK_DECODE) are dense, as before
template<typename Cells>
static inkling2_decode_mask classify_decode(bool decode_rows, uint32_t pool, uint32_t n_rows, bool allow_mean, Cells cells) {
    if (!decode_rows) {
        auto shapes = mask_shapes();
        shapes.rows_min = std::max<uint32_t>(shapes.rows_min, 8);
        return inkling2_classify_decode_mask(pool, n_rows, allow_mean, cells, shapes);
    }
    return inkling2_classify_decode_rows(pool, n_rows, allow_mean, cells, mask_shapes());
}

// the positions the relative bias is looked up by, for the global cache (0) and the window cache (1): with the bias inside
// the attention kernel the position of each cell, from which the kernel derives the distances, and with
// LLAMA_INKLING2_REL_POS_LEGACY=1 the distance of every query to every cell
class llm_graph_input_inkling2_pos : public llm_graph_input_i {
public:
    const llama_kv_cache_context * cache[2];
    ggml_tensor * pos[2];
    bool resident_pos[2] = {false, false};
    ggml_backend_sched_t sched = nullptr;
    bool cells;
    uint32_t extent[2];
    inkling2_decode_mask mask_mode[2];
    bool mean_decode;
    // LLAMA_INKLING2_MEAN_MASK_DECODE: the legacy low-mean shape is low-mean under the new classes too
    bool mean_shape = false;
    // the runs of cells of the global (0) and window (1) caches (LLAMA_INKLING2_KV_RUNS), at most runs_max a row; 0 when off
    ggml_tensor * runs[2] = {};
    uint32_t runs_max[2] = {};
    // a q8_0 or f8_e4m3 cache too, which the runs decode reads with GGML_CUDA_FATTN_REL_DECODE_ONE_BYTE
    bool runs_one_byte = false;
    bool runs_q4 = false;
    // true when any position in the ubatch can reach n_log_scale_floor (or full-cache reserve), so tau != 1
    bool use_tau = true;
    int64_t log_scale_floor = 0;
    // last-layer prefill MoE pruning state
    int32_t n_layer = 0;
    int32_t shortconv_k = 0;
    bool hidden_apart = false;
    bool is_loop = false;
    bool prune_last_moe = false;
    int64_t prune_tail = 0;
    ggml_tensor * pruned_out_ids = nullptr;

    bool can_prune_last_moe(const llm_graph_params & params, int64_t & out_tail, int64_t & out_n_outputs) const {
        static const bool legacy = ggml_env_switch("LLAMA_INKLING2_PRUNE_LAST_MOE_LEGACY");
        if (legacy || is_loop || !hidden_apart || !params.cparams.fused_inkling2 || !params.cparams.fused_inkling2_conv ||
                params.cparams.embeddings || params.cparams.embeddings_layer_inp[n_layer] ||
                params.cparams.pipeline_parallel || params.cparams.n_seq_devices > 1 || !params.ubatch.equal_seqs() ||
                !params.ubatch.output) {
            return false;
        }
        const int64_t nt = params.ubatch.n_seq_tokens;
        const int64_t keep = std::min<int64_t>(nt, int64_t(params.cparams.n_rs_seq) + 1);
        const int64_t tail = (shortconv_k - 1) + keep;
        if (nt < 2*tail) {
            return false;
        }
        int64_t n_out = 0;
        for (uint32_t i = 0; i < params.ubatch.n_tokens; ++i) {
            if (params.ubatch.output[i]) {
                if (int64_t(i % nt) < nt - keep) {
                    return false;
                }
                n_out++;
            }
        }
        out_tail = tail;
        out_n_outputs = n_out;
        return true;
    }

    bool needs_tau(const llm_graph_params & params) const {
        if (log_scale_floor < 1024 || cache[0]->is_full() || !params.ubatch.pos) {
            return true;
        }
        for (uint32_t i = 0; i < params.ubatch.n_tokens; ++i) {
            if (int64_t(params.ubatch.pos[i]) + 1 > log_scale_floor) {
                return true;
            }
        }
        return false;
    }

    // A step of 1 to 8 tip/draft rows per sequence gives global (0) or window (1) layers the runs of its sequences' cells.
    bool classify_runs(const llm_graph_params & params, int i) const {
        const auto & ub = params.ubatch;
        const auto * kv = cache[i];
        const ggml_type type = kv->type_k();
        const bool type_ok = type == kv->type_v() &&
                (type == GGML_TYPE_F16 || (runs_one_byte && (type == GGML_TYPE_Q8_0 || type == GGML_TYPE_F8_E4M3)) ||
                 (runs_q4 && type == GGML_TYPE_Q4_0));
        const uint32_t min_kv = i == 0 ? 8192 : 256;
        const uint32_t per_seq = params.cparams.rel_query_group && ub.equal_seqs() && ub.n_seq_tokens >= 2 && ub.n_seq_tokens <= 8
                ? ub.n_seq_tokens : 1;
        if (runs_max[i] == 0 || !params.cparams.attn_mask_bits || !params.cparams.kv_unified || kv->is_full() || !type_ok ||
                kv->get_n_kv() < min_kv || ub.n_seqs_unq < mask_shapes().rows_min ||
                ub.n_seqs_unq > mask_shapes().rows_max || ub.n_tokens != ub.n_seqs_unq * per_seq) {
            return false;
        }
        for (uint32_t s = 0; s < ub.n_seqs_unq; ++s) {
            const auto seq = ub.seq_id[s*per_seq][0];
            const llama_pos pmax = kv->get_kv()->get_cells(seq).seq_pos_max(seq);
            for (uint32_t g = 0; g < per_seq; ++g) {
                const uint32_t row = s*per_seq + g;
                if (ub.n_seq_id[row] != 1 || ub.seq_id[row][0] != seq) {
                    return false;
                }
                if (g > 0 && ub.pos[row] <= ub.pos[row - 1]) {
                    return false;
                }
            }
            const llama_pos tip = ub.pos[s*per_seq + per_seq - 1];
            const llama_pos first = ub.pos[s*per_seq];
            if (tip < pmax) {
                return false;
            }
            if (i == 1 && (tip != pmax || pmax - first > 7)) {
                return false;
            }
        }
        return kv->get_kv()->kv_runs_fit(ub, runs_max[i]);
    }

    inkling2_decode_mask classify_mask(const llm_graph_params & params, int i) const {
        const auto & ub = params.ubatch;
        const auto * kv = cache[i];
        const bool decode_rows = params.cparams.rel_decode_rows;
        if (!params.cparams.attn_mask_bits || !params.cparams.kv_unified || kv->is_full()) {
            return inkling2_decode_mask::dense;
        }
        const ggml_type kt = kv->type_k();
        const bool short_type_ok = kt == kv->type_v() &&
                (kt == GGML_TYPE_F16 || kt == GGML_TYPE_Q8_0 || kt == GGML_TYPE_Q4_0 || kt == GGML_TYPE_F8_E4M3);
        if (ub.n_tokens <= 32 && kv->get_n_kv() > 256 && short_type_ok) {
            bool short_rows = true;
            for (uint32_t i = 0; i < ub.n_seqs_unq && short_rows; ++i) {
                const auto seq = ub.seq_id_unq[i];
                short_rows = kv->get_kv()->get_cells(seq).seq_n_cells(seq) <= mask_shapes().short_cells;
            }
            if (short_rows) {
                for (uint32_t i = 0; i < ub.n_tokens && short_rows; ++i) {
                    short_rows = ub.n_seq_id[i] == 1;
                }
            }
            if (short_rows) {
                return inkling2_decode_mask::short_rows;
            }
        }
        // Drafted steps group 2..4 rows per sequence (up to 8 when kv_runs handles the step) under cparams.rel_query_group.
        const uint32_t per_seq = ub.equal_seqs() ? ub.n_seq_tokens : 1;
        const bool group_ok = per_seq >= 2 && (per_seq <= 4 || (per_seq <= 8 && classify_runs(params, i)));
        if (group_ok && ub.n_tokens == ub.n_seqs_unq*per_seq && params.cparams.rel_query_group) {
            for (uint32_t row = 0; row < ub.n_tokens; ++row) {
                if (ub.n_seq_id[row] != 1 || ub.seq_id[row][0] != ub.seq_id[row - row % per_seq][0]) {
                    return inkling2_decode_mask::dense;
                }
            }
            return classify_decode(decode_rows, kv->get_n_kv(), ub.n_seqs_unq, mean_decode && decode_rows, [&](uint32_t s) {
                const auto seq = ub.seq_id[s*per_seq][0];
                return kv->get_kv()->get_cells(seq).seq_n_cells(seq);
            });
        }
        if (ub.n_tokens != ub.n_seqs_unq) {
            return inkling2_decode_mask::dense;
        }
        // the low-mean decode kernels read f16 and q8_0, and f8_e4m3 with GGML_CUDA_FATTN_REL_DECODE_ONE_BYTE; the legacy
        // shape is LLAMA_INKLING2_MEAN_ROWS rows of the global cache
        const ggml_type type = kv->type_k();
        const bool mean_type_ok = i == 0 && type == kv->type_v() &&
                (type == GGML_TYPE_F16 || type == GGML_TYPE_Q8_0 || type == GGML_TYPE_F8_E4M3 ||
                 (runs_q4 && type == GGML_TYPE_Q4_0));
        const bool legacy_shape = mean_type_ok && ub.n_tokens == mask_shapes().mean_rows;
        const bool mean_rows_ok = mean_type_ok && ub.n_tokens >= std::max<uint32_t>(mask_shapes().rows_min, 8) &&
                ub.n_tokens <= mask_shapes().rows_max;
        bool allow_mean = mean_decode && (decode_rows || legacy_shape);
        if (allow_mean) {
            for (uint32_t row = 0; row < ub.n_tokens; ++row) {
                if (ub.n_seq_id[row] != 1) {
                    allow_mean = false;
                    break;
                }
            }
        }
        const auto cells = [&](uint32_t row) {
            const auto seq = ub.seq_id[row][0];
            return kv->get_kv()->get_cells(seq).seq_n_cells(seq);
        };
        const auto mode = classify_decode(decode_rows, kv->get_n_kv(), ub.n_tokens, allow_mean, cells);
        // rows that share a long prefix count it once each and can pass LLAMA_INKLING2_DECODE_POOL_READS: keep low-mean
        // across decoding batch sizes under LLAMA_INKLING2_MEAN_MASK_DECODE
        if (mode == inkling2_decode_mask::dense && decode_rows && mean_shape && allow_mean && mean_rows_ok) {
            return classify_decode(false, kv->get_n_kv(), ub.n_tokens, true, cells);
        }
        return mode;
    }

    bool can_reuse(const llm_graph_params & params) override {
        const auto * memory = static_cast<const llama_memory_hybrid_iswa_context *>(params.mctx);
        cache[0] = memory->get_attn()->get_base();
        cache[1] = memory->get_attn()->get_swa();
        sched    = params.sched;
        const int64_t n_stream = params.cparams.kv_unified ? 1 : params.ubatch.n_seqs_unq;
        for (int i = 0; i < 2; ++i) {
            const int64_t want_planes = (i == 0 && cells && cache[0]->has_kv_mixed()) ? 2 : 1;
            if (pos[i]->ne[0] != cache[i]->get_n_kv() || pos[i]->ne[1] != (cells ? n_stream : params.ubatch.n_tokens) ||
                    pos[i]->ne[2] != want_planes ||
                    (cells && mask_mode[i] != classify_mask(params, i))) {
                return false;
            }
            if ((runs[i] != nullptr) != classify_runs(params, i) ||
                    (runs[i] && runs[i]->ne[2] != params.ubatch.n_tokens + 1)) {
                return false;
            }
        }
        int64_t next_tail = 0, next_out = 0;
        const bool next_prune = can_prune_last_moe(params, next_tail, next_out);
        if (next_prune != prune_last_moe || (next_prune && (next_tail != prune_tail ||
                (pruned_out_ids != nullptr) != (next_out > 0) ||
                (pruned_out_ids && pruned_out_ids->ne[0] != next_out)))) {
            return false;
        }
        return use_tau == needs_tau(params);
    }

    void set_input(const llama_ubatch * ubatch) override {
        for (int i = 0; i < 2; ++i) {
            if (!pos[i]->buffer) {
                continue;
            }
            if (resident_pos[i]) {
                cache[i]->get_kv()->sync_resident_cell_pos(pos[i], sched, pos[i]->ne[0]);
            } else if (cells) {
                cache[i]->set_input_cell_pos(pos[i], ubatch);
            } else {
                cache[i]->set_input_relative_pos(pos[i], ubatch, extent[i]);
            }
        }
        for (int i = 0; i < 2; ++i) {
            if (runs[i] && runs[i]->buffer) {
                cache[i]->get_kv()->set_input_kv_runs(runs[i], ubatch, sched);
            }
        }
        if (pruned_out_ids && pruned_out_ids->buffer) {
            const int64_t nt = ubatch->n_seq_tokens;
            std::vector<int32_t> ids;
            ids.reserve(pruned_out_ids->ne[0]);
            for (uint32_t i = 0; i < ubatch->n_tokens; ++i) {
                if (ubatch->output[i]) {
                    const int64_t s = i / nt;
                    const int64_t u = i % nt;
                    ids.push_back(int32_t(s * prune_tail + (u - (nt - prune_tail))));
                }
            }
            GGML_ASSERT((int64_t) ids.size() == pruned_out_ids->ne[0]);
            ggml_backend_tensor_set(pruned_out_ids, ids.data(), 0, ids.size()*sizeof(int32_t));
        }
    }

    void input_lost() override {
        for (int i = 0; i < 2; ++i) {
            if (cache[i] && cache[i]->get_kv()) {
                cache[i]->get_kv()->invalidate_delta_state();
            }
        }
    }
};

ggml_tensor * llama_model_inkling2::graph::clip(ggml_tensor * cur, ggml_tensor * scale, float threshold, int il) {
    if (cparams.fused_inkling2) {
        cur = ggml_norm_clip(ctx0, cur, scale, threshold, 1e-6f);
        res->add_fused_node({LLM_FUSED_OP_INKLING2_LAYER, cur, il});
        return cur;
    }
    if (scale) {
        cur = ggml_mul(ctx0, cur, scale);
    }
    auto * norm = ggml_sqrt(ctx0, ggml_sum_rows(ctx0, ggml_sqr(ctx0, cur)));
    norm = ggml_scale_bias(ctx0, norm, 1.0f, 1e-6f);
    auto * factor = ggml_div(ctx0, ggml_fill(ctx0, norm, threshold), norm);
    return ggml_mul(ctx0, cur, ggml_clamp(ctx0, factor, 0.0f, 1.0f));
}

ggml_tensor * llama_model_inkling2::graph::find_block_norm(ggml_tensor * block) const {
    for (const auto & kv : block_norms) {
        if (kv.first == block) {
            return kv.second;
        }
    }
    return nullptr;
}

ggml_tensor * llama_model_inkling2::graph::mix(const std::vector<ggml_tensor *> & blocks, ggml_tensor * cur, ggml_tensor * query, int il) {
    // -sm data: a mix reads the whole of a token's sources, and the fused op is mirrored only; the unfused ops below
    // run on the device's own rows
    if (cparams.fused_inkling2 && cparams.n_seq_devices <= 1 && blocks.size() + 1 < GGML_MAX_SRC) {
        std::vector<ggml_tensor *> sources;
        sources.reserve(blocks.size() + 1);
        int32_t n_normed = 0;
        bool prefix = true;
        for (auto * block : blocks) {
            ggml_tensor * normed = prefix ? find_block_norm(block) : nullptr;
            if (normed) {
                sources.push_back(normed);
                n_normed++;
            } else {
                prefix = false;
                sources.push_back(block);
            }
        }
        sources.push_back(cur);
        cur = ggml_attn_res_mix(ctx0, query, sources.data(), sources.size(), 1.0f/sqrtf(n_embd), 1e-6f);
        if (n_normed > 0) {
            ggml_attn_res_mix_set_normed(cur, n_normed);
        }
        if (il >= 0) {
            res->add_fused_node({LLM_FUSED_OP_INKLING2_LAYER, cur, il});
        }
        return cur;
    }
    const int64_t n_rows = cur->ne[1];
    auto * states = ggml_reshape_3d(ctx0, cur, n_embd, 1, n_rows);
    for (auto * block : blocks) {
        states = ggml_concat(ctx0, ggml_reshape_3d(ctx0, block, n_embd, 1, n_rows), states, 1);
    }
    states = ggml_rms_norm(ctx0, states, 1e-6f);
    auto * queries = ggml_repeat(ctx0, query, ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_embd, query->ne[1], n_rows));
    auto * scores = ggml_mul_mat(ctx0, states, queries);
    scores = ggml_soft_max_ext(ctx0, scores, nullptr, 1.0f/sqrtf(n_embd), 0.0f);
    return ggml_mul_mat(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, states)), scores);
}

ggml_tensor * llama_model_inkling2::graph::convolve(ggml_tensor * cur, ggml_tensor * weight, ggml_tensor * dynamic,
        ggml_tensor * history, int64_t offset, std::vector<ggml_tensor *> & windows, int il) {
    const int64_t kernel = hparams.n_shortconv_l_cache;
    const int64_t channels = cur->ne[0];
    const int64_t ns = ubatch.n_seqs;
    const int64_t n_cur = cur->ne[1] * cur->ne[2];
    const int64_t nt = n_cur / ns;
    GGML_ASSERT(n_cur % ns == 0);
    auto * past = ggml_view_3d(ctx0, history, kernel - 1, channels, ns, history->nb[1], history->nb[2], offset*history->nb[1]);
    auto * input = cur;
    if (input->ne[1] != nt || input->ne[2] != ns) {
        input = ggml_reshape_3d(ctx0, input, channels, nt, ns);
    }
    auto * x = ggml_concat(ctx0, past, ggml_transpose(ctx0, input), 0);
    // Only the tail needed by build_conv_state_store has to be joined across channel groups.
    const int64_t keep = std::min<int64_t>(nt, int64_t(cparams.n_rs_seq) + 1);
    if (cparams.fused_inkling2_conv && keep < nt) {
        windows.push_back(ggml_view_3d(ctx0, x, kernel - 1 + keep, channels, ns, x->nb[1], x->nb[2], (nt - keep)*x->nb[0]));
    } else {
        windows.push_back(x);
    }
    const float scale = 1.0f/sqrtf((dynamic ? 2 : 1)*kernel + 1);
    if (cparams.fused_inkling2_conv) {
        auto * out = ggml_short_conv(ctx0, x, weight, dynamic, scale);
        res->add_fused_node({LLM_FUSED_OP_INKLING2_CONV, out, il});
        return ggml_reshape_2d(ctx0, out, channels, n_cur);
    }
    auto * out = ggml_ssm_conv(ctx0, x, weight);
    out = ggml_reshape_2d(ctx0, out, channels, n_cur);
    if (dynamic) {
        const int64_t heads = dynamic->ne[0]/kernel;
        // the lags outermost: a lag's taps are one run, at an offset -sm tensor scales as each device's heads are split
        // (an offset inside a head's taps, which the split axis holds, would be scaled with them)
        dynamic = ggml_cont(ctx0, ggml_permute(ctx0, ggml_reshape_3d(ctx0, dynamic, kernel, heads, n_cur), 2, 0, 1, 3));
        for (int64_t lag = 0; lag < kernel; ++lag) {
            auto * tap = ggml_view_3d(ctx0, dynamic, 1, heads, n_cur, dynamic->nb[0], dynamic->nb[1], lag*dynamic->nb[2]);
            auto * window = ggml_view_3d(ctx0, x, nt, channels, ns, x->nb[1], x->nb[2], (kernel - 1 - lag)*x->nb[0]);
            window = ggml_cont(ctx0, ggml_transpose(ctx0, window));
            window = ggml_reshape_3d(ctx0, window, channels/heads, heads, n_cur);
            auto * term = ggml_reshape_2d(ctx0, ggml_mul(ctx0, window, tap), channels, n_cur);
            out = ggml_add(ctx0, out, term);
        }
    }
    auto * residual = ggml_is_contiguous(input) ? ggml_reshape_2d(ctx0, input, channels, n_cur)
                                                : ggml_cont_2d(ctx0, input, channels, n_cur);
    return ggml_scale(ctx0, ggml_add(ctx0, residual, out), scale);
}

static ggml_tensor * gather_channels(ggml_context * ctx, ggml_tensor * x) {
    if (x->ne[1] == 1 || !ggml_env_switch("LLAMA_INKLING2_GATHER_TRANSPOSE")) {
        return ggml_backend_meta_gather(ctx, x);
    }
    // Put each rank's channels in one contiguous block for the gather.
    x = ggml_cont(ctx, ggml_transpose(ctx, x));
    x = ggml_backend_meta_gather(ctx, x);
    return ggml_cont(ctx, ggml_transpose(ctx, x));
}

// -sm tensor: a hidden stream comes out of its reduction whole on every device. Its state is the S state, whole on every
// device with its kernel, so each device convolves every channel after one all-reduce. With
// LLAMA_INKLING2_HIDDEN_CONV_LEGACY the kernel and the state are split by channel (llama_meta_device_get_split_state):
// each device convolves its own channels, which a scatter gives it with the channels outermost so that they are one run
// of the tensor, and the result is gathered, two collectives a stream.
ggml_tensor * llama_model_inkling2::graph::convolve_hidden(ggml_tensor * cur, ggml_tensor * weight, ggml_tensor * history,
        int64_t offset, std::vector<ggml_tensor *> & windows, int il) {
    if (model.split_mode() != LLAMA_SPLIT_MODE_TENSOR || hparams.n_sconv_hidden > 0) {
        return convolve(cur, weight, nullptr, history, offset, windows, il);
    }
    auto ud = model.get_split_state_ud;
    const ggml_backend_meta_split_state split = llama_meta_device_get_split_state(weight, &ud);
    GGML_ASSERT(split.axis == GGML_BACKEND_SPLIT_AXIS_1 && split.n_segments == 1);
    const int64_t channels = cur->ne[0];
    auto * x = ggml_permute(ctx0, ggml_reshape_3d(ctx0, cur, channels, ubatch.n_seq_tokens, ubatch.n_seqs), 2, 0, 1, 3);
    if (!ggml_is_contiguous(x)) {
        x = ggml_cont(ctx0, x);
    }
    x = ggml_backend_meta_scatter(ctx0, x, 2, split.ne, ud.n_devices);
    // CONCAT can read this strided shard directly when it appends the new tokens to the history.
    x = ggml_permute(ctx0, x, 1, 2, 0, 3);
    return gather_channels(ctx0, convolve(x, weight, nullptr, history, offset, windows, il));
}

// mix a cross-pass message into a looped layer input: (1 - forget)*cur + gain*message, both gated on cur and message.
// The message, and a Q, K or V input, come as views (of the four messages, of the depth mix's queries): the fused op
// reads their rows where they are, the unfused ops read copies. A copy is a memcpy node of the CUDA graph, which the
// fused op's PDL launch cannot overlap (RTX 5090, a decode token: ~1.2 us before each of 96 controllers).
ggml_tensor * llama_model_inkling2::graph::control(ggml_tensor * cur, ggml_tensor * message, const loop_layer & loop, int ctrl, int il) {
    ggml_tensor * w_down = loop.ctrl_down[ctrl], * w_gain = loop.ctrl_gain[ctrl], * w_forget = loop.ctrl_forget[ctrl];
    // The fused op reads each weight as it is stored: not split across devices or repacked for the CPU (extra), not
    // folded with a Hadamard transform, no LoRA delta; the diagnostic reads the gates, which the op keeps to itself.
    // The CUDA kernel holds 64 tokens at most; the probe turns the op off where the layer's device lacks it.
    const auto plain = [&](const ggml_tensor * w) {
        const bool typed = w->type == GGML_TYPE_F32 || w->type == GGML_TYPE_F16 || w->type == GGML_TYPE_BF16 || w->type == GGML_TYPE_Q8_0;
        return typed && !w->extra && !(hdmd && hdmd->rot.count(w));
    };
    if (cparams.fused_inkling2_loop_control && model.split_mode() != LLAMA_SPLIT_MODE_TENSOR && cparams.n_seq_devices <= 1 && n_tokens <= 64 &&
            loras->empty() && !loop_diag_on() && plain(w_down) && plain(w_gain) && plain(w_forget) &&
            w_gain->type == w_forget->type) {
        cur = ggml_loop_control(ctx0, cur, message, w_down, w_gain, w_forget, 1e-6f);
        res->add_fused_node({LLM_FUSED_OP_INKLING2_LOOP_CONTROL, cur, il});
        return cur;
    }
    if (cur->op == GGML_OP_VIEW) {
        cur = ggml_cont(ctx0, cur);
    }
    message = ggml_cont(ctx0, message);
    ggml_tensor * gate = ggml_concat(ctx0, ggml_rms_norm(ctx0, cur, 1e-6f), ggml_rms_norm(ctx0, message, 1e-6f), 0);
    gate = build_lora_mm(loop.ctrl_down[ctrl], gate);

    ggml_tensor * gain   = ggml_tanh(ctx0, build_lora_mm(loop.ctrl_gain[ctrl],   gate));
    ggml_tensor * forget = ggml_tanh(ctx0, build_lora_mm(loop.ctrl_forget[ctrl], gate));

    ggml_tensor * out = ggml_add(ctx0, ggml_sub(ctx0, cur, ggml_mul(ctx0, forget, cur)), ggml_mul(ctx0, gain, message));
    if (loop_diag_on()) {
        auto mean_abs = [&](ggml_tensor * t) { return ggml_mean(ctx0, ggml_abs(ctx0, t)); };  // [1, n_tokens]
        auto * change = ggml_sub(ctx0, ggml_mul(ctx0, gain, message), ggml_mul(ctx0, forget, cur));
        loop_diag_gates = ggml_concat(ctx0, ggml_concat(ctx0, mean_abs(gain), mean_abs(forget), 0),
                ggml_concat(ctx0, mean_abs(change), mean_abs(cur), 0), 0);
    }
    return out;
}

ggml_tensor * llama_model_inkling2::graph::build_inputs(const llm_graph_params & params) {
    loop_diag_kv.clear();
    fused_rel    = cparams.flash_attn && cparams.fused_rel_pos;
    direct_sinks = fused_rel && !ggml_env_switch("LLAMA_INKLING2_SHIFT_SINKS");
    // The unfused graph adds the relative bias to a dense mask before attention.
    auto * inp = build_inp_mem_hybrid_iswa(fused_rel);
    attn      = inp->get_attn();
    recr      = inp->get_recr();
    positions = build_inp_pos();
    state_only = cparams.fused_inkling2 && n_outputs == 0 && !cparams.embeddings && !cparams.pipeline_parallel && cparams.n_seq_devices <= 1;
    auto rel = std::make_unique<llm_graph_input_inkling2_pos>();
    rel->cache[0] = attn->mctx->get_base();
    rel->cache[1] = attn->mctx->get_swa();
    rel->sched    = params.sched;
    rel->extent[0] = hparams.n_relative_extent;
    rel->extent[1] = hparams.n_swa;
    rel->cells = fused_rel;
    rel->n_layer = n_layer;
    rel->shortconv_k = hparams.n_shortconv_l_cache;
    rel->hidden_apart = hparams.n_sconv_hidden > 0;
    rel->is_loop = model.loop_own_passes() > 0;
    int64_t prune_out = 0;
    prune_last_moe = rel->prune_last_moe = rel->can_prune_last_moe(params, prune_tail, prune_out);
    rel->prune_tail = prune_tail;
    inp_out_ids = (state_only || prune_last_moe) ? nullptr : build_inp_out_ids();
    if (prune_last_moe && prune_out > 0) {
        pruned_out_ids = rel->pruned_out_ids = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, prune_out);
        ggml_set_input(pruned_out_ids);
    }
    // the low-mean hint takes the decode kernels' sink a query
    rel->mean_decode = direct_sinks && (cparams.rel_decode_rows || ggml_env_switch("LLAMA_INKLING2_MEAN_MASK_DECODE"));
    rel->mean_shape = direct_sinks && ggml_env_switch("LLAMA_INKLING2_MEAN_MASK_DECODE");
    const uint32_t runs_max_global = direct_sinks && ggml_env_switch("LLAMA_INKLING2_KV_RUNS") ? (uint32_t) ggml_env_int("LLAMA_INKLING2_KV_RUNS_MAX") : 0;
    rel->runs_max[0] = runs_max_global;
    rel->runs_max[1] = !ggml_env_switch("LLAMA_INKLING2_SWA_RUNS_LEGACY") ? std::min<uint32_t>(runs_max_global, 64) : 0;
    rel->runs_one_byte = ggml_env_switch("GGML_CUDA_FATTN_REL_DECODE_ONE_BYTE");
    rel->runs_q4 = ggml_env_switch("GGML_CUDA_FATTN_REL_DECODE_Q4");
    const int64_t n_stream = cparams.kv_unified ? 1 : ubatch.n_seqs_unq;
    for (int i = 0; i < 2; ++i) {
        const auto mode = rel->mask_mode[i] = fused_rel ? rel->classify_mask(params, i) : inkling2_decode_mask::dense;
        n_kv_max[i] = mode == inkling2_decode_mask::short_rows ? mask_shapes().short_cells :
                mode == inkling2_decode_mask::bounded_rows ? rel->cache[i]->get_n_kv()/mask_shapes().pool_share : 0;
        low_mean[i] = mode == inkling2_decode_mask::low_mean;
        const bool want_hot_map = i == 0 && rel->cells && rel->cache[0]->has_kv_mixed();
        ggml_tensor * base_pos = rel->cells && n_stream == 1 && cparams.n_seq_devices == 1 && !rel->cache[i]->is_full() ?
                rel->cache[i]->get_kv()->get_resident_cell_pos() : nullptr;
        if (base_pos && (!want_hot_map || base_pos->ne[2] == 2)) {
            const int64_t n_kv_cur = rel->cache[i]->get_n_kv();
            rel_idx[i] = rel->pos[i] = want_hot_map
                ? ggml_view_3d(ctx0, base_pos, n_kv_cur, 1, 2, base_pos->nb[1], base_pos->nb[2], 0)
                : ggml_view_2d(ctx0, base_pos, n_kv_cur, 1, base_pos->nb[1], 0);
            rel->resident_pos[i] = true;
        } else {
            rel_idx[i] = rel->pos[i] = want_hot_map
                ? ggml_new_tensor_3d(ctx0, GGML_TYPE_I32, rel->cache[i]->get_n_kv(), n_stream, 2)
                : ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, rel->cache[i]->get_n_kv(), rel->cells ? n_stream : n_tokens);
            if (cparams.n_seq_devices > 1) {
                // -sm data: the cells read are the device's own sequences', a stream a column, as the mask's streams are
                const auto counts = rel->cells ? ubatch.device_seqs(cparams.n_seq_devices)
                                               : ubatch.device_rows(cparams.n_seq_devices);
                ggml_backend_meta_split_input(rel_idx[i], 1, counts.data(), counts.size());
            }
            ggml_set_input(rel_idx[i]);
        }
    }
    rel->log_scale_floor = hparams.n_log_scale_floor;
    const bool use_tau = rel->use_tau = rel->needs_tau(params);
    for (int i = 0; i < 2; ++i) {
        if (rel->classify_runs(params, i)) {
            const int64_t n_entries = rel->runs_max[i] + 2;
            const int64_t cap_rows_p1 = cparams.rel_query_group ? 257 : 65;
            ggml_tensor * base_runs = n_stream == 1 && cparams.n_seq_devices == 1 && !rel->cache[i]->is_full() ?
                    rel->cache[i]->get_kv()->get_resident_kv_runs(n_entries, n_tokens + 1, cap_rows_p1) : nullptr;
            if (base_runs) {
                kv_runs[i] = rel->runs[i] = ggml_view_3d(ctx0, base_runs, 4, n_entries, n_tokens + 1,
                        4*sizeof(int32_t), 4*n_entries*sizeof(int32_t), 0);
            } else {
                kv_runs[i] = rel->runs[i] = ggml_new_tensor_3d(ctx0, GGML_TYPE_I32, 4, n_entries, n_tokens + 1);
                ggml_set_input(kv_runs[i]);
            }
        }
    }
    res->add_input(std::move(rel));
    // log-scaled positions: 1 + alpha*log(max(1, (pos + 1)/floor)), multiplies q and r of the full-attention layers.
    // Below floor, tau is identically 1.0f; skip building and multiplying it.
    tau = nullptr;
    if (use_tau) {
        tau = ggml_scale_bias(ctx0, ggml_cast(ctx0, positions, GGML_TYPE_F32), 1.0f, 1.0f);
        tau = ggml_log(ctx0, ggml_clamp(ctx0, ggml_scale(ctx0, tau, 1.0f/hparams.n_log_scale_floor), 1.0f, INFINITY));
        tau = ggml_reshape_3d(ctx0, ggml_scale_bias(ctx0, tau, hparams.f_log_scale_alpha, 1.0f), 1, 1, n_tokens);
    }
    auto * cur = build_inp_embd(model.tok_embd);
    // important: do not normalize raw embeddings input (i.e. encoded image embeddings)
    if (ubatch.token) {
        cur = build_norm(cur, model.tok_norm, nullptr, LLM_NORM_RMS, -1);
    }
    return cur;
}

ggml_tensor * llama_model_inkling2::graph::build_layer(int il, int ip, ggml_tensor * cur, std::vector<ggml_tensor *> & blocks,
        ggml_tensor * messages, const loop_layer * loop) {
    GGML_ASSERT(!messages == !loop);
    const auto & l = model.layers[il];
    const bool swa = hparams.is_swa(il);
    const int64_t nh = hparams.n_head(il);
    const int64_t nk = hparams.n_head_kv(il);
    const int64_t hd = hparams.n_embd_head_k(il);
    const int64_t kv = nk*hd;
    // the message of controller ctrl, [n_embd, n_tokens], a view (control() copies it where it needs to)
    auto message = [&](int ctrl) {
        return ggml_view_2d(ctx0, messages, n_embd, n_tokens, messages->nb[2], ctrl*messages->nb[1]);
    };
    // LLAMA_LOOP_DIAG: name and keep the statistics of the control() call just made
    auto diag_gates = [&](int ctrl) {
        if (loop_diag_gates) {
            ggml_format_name(loop_diag_gates, "loop_diag_gates_%d-%d", ctrl, il);
            ggml_set_output(loop_diag_gates);
            ggml_build_forward_expand(gf, loop_diag_gates);
            loop_diag_gates = nullptr;
        }
    };
    auto * state = recr->mctx->get_r_l(il);
    auto * history = build_rs(recr, state, hparams.n_embd_r(), ubatch.n_seqs);
    history = ggml_reshape_3d(ctx0, history, hparams.n_shortconv_l_cache - 1, hparams.n_sconv_channels, ubatch.n_seqs);
    // the hidden streams' state: the S state, or the conv row after K and V (LLAMA_INKLING2_HIDDEN_CONV_LEGACY)
    const bool hidden_apart = hparams.n_sconv_hidden > 0;
    auto * hidden_state = hidden_apart ? recr->mctx->get_s_l(il) : state;
    auto * hidden = history;
    if (hidden_apart) {
        hidden = build_rs(recr, hidden_state, hparams.n_embd_s(), ubatch.n_seqs);
        hidden = ggml_reshape_3d(ctx0, hidden, hparams.n_shortconv_l_cache - 1, hparams.n_sconv_hidden, ubatch.n_seqs);
    }
    const int64_t hidden_offset = hidden_apart ? 0 : 2*kv;
    std::vector<ggml_tensor *> hidden_windows;
    res->t_layer_inp[il] = cur;
    auto * residual = cur;
    ggml_tensor * inputs[3] = {cur, cur, cur};
    if (l.attn_res_query) {
        auto * mixed = mix(blocks, cur, l.attn_res_query, il);
        // the messages make the Q, K and V inputs of a SWA layer differ
        for (int j = 0; j < (swa && !messages ? 1 : 3); ++j) {
            inputs[j] = ggml_view_2d(ctx0, mixed, n_embd, n_tokens, mixed->nb[2], (swa ? 0 : j)*mixed->nb[1]);
        }
    }
    if (messages) {
        for (int j = 0; j < 3; ++j) {
            inputs[j] = control(inputs[j], message(LOOP_CTRL_Q + j), *loop, LOOP_CTRL_Q + j, il);
            diag_gates(LOOP_CTRL_Q + j);
        }
    }
    if (swa && messages) {
        for (int j = 0; j < 3; ++j) {
            inputs[j] = build_norm(inputs[j], l.attn_norm, nullptr, LLM_NORM_RMS, il);
        }
    } else if (swa) {
        inputs[0] = build_norm(inputs[0], l.attn_norm, nullptr, LLM_NORM_RMS, il);
        inputs[1] = inputs[2] = inputs[0];
    } else {
        inputs[0] = build_norm(inputs[0], l.attn_q_inp_norm, nullptr, LLM_NORM_RMS, il);
        inputs[1] = build_norm(inputs[1], l.attn_k_inp_norm, nullptr, LLM_NORM_RMS, il);
        inputs[2] = build_norm(inputs[2], l.attn_v_inp_norm, nullptr, LLM_NORM_RMS, il);
    }
    // the products of the Q input, and K's where it reads the same input, next to each other in the graph: a backend can
    // run them as one launch (the CUDA few-row MMQ group)
    auto * q     = build_lora_mm(l.wq, inputs[0]);
    auto * r     = build_lora_mm(l.attn_rel, inputs[0]);
    auto * sinks = build_lora_mm(l.attn_sink_proj, inputs[0]);
    auto * gate  = build_lora_mm(l.wqkv_gate, inputs[0]);
    auto * k     = build_lora_mm(l.wk, inputs[1]);
    for (auto * t : {q, r, sinks, gate, k}) {
        ggml_build_forward_expand(gf, t);
    }
    // the windows of the state's streams in the order of its channels: K, V, and the two hidden streams when they
    // share the row
    std::vector<ggml_tensor *> windows;
    auto & hwindows = hidden_apart ? hidden_windows : windows;
    k = convolve(k, l.attn_k_conv, swa ? nullptr : build_lora_mm(l.attn_k_conv_proj, inputs[1]), history, 0, windows, il);
    auto * v = convolve(build_lora_mm(l.wv, inputs[2]), l.attn_v_conv,
            swa ? nullptr : build_lora_mm(l.attn_v_conv_proj, inputs[2]), history, kv, windows, il);
    q = clip(ggml_reshape_3d(ctx0, q, hd, nh, n_tokens), l.attn_q_clip, hparams.f_qk_clip, il);
    k = clip(ggml_reshape_3d(ctx0, k, hd, nk, n_tokens), nullptr, hparams.f_qk_clip, il);
    v = ggml_rms_norm(ctx0, ggml_reshape_3d(ctx0, v, hd, nk, n_tokens), hparams.f_norm_rms_eps);
    if (loop_diag_on()) {
        // LLAMA_LOOP_DIAG: a later pass's K and V rows against the same physical layer's pass 1 and previous pass
        auto & seen = loop_diag_kv[ip];
        if (!messages) {
            seen = {k, v, k, v};
        } else if (seen[0]) {
            auto row = [&](ggml_tensor * t) { return ggml_reshape_2d(ctx0, ggml_cont(ctx0, t), kv, n_tokens); };
            auto cosine = [&](ggml_tensor * a, ggml_tensor * b) {
                return ggml_div(ctx0, ggml_sum_rows(ctx0, ggml_mul(ctx0, a, b)), ggml_sqrt(ctx0,
                        ggml_mul(ctx0, ggml_sum_rows(ctx0, ggml_sqr(ctx0, a)), ggml_sum_rows(ctx0, ggml_sqr(ctx0, b)))));
            };
            for (int t = 0; t < 2; ++t) {
                auto * a = row(t ? v : k), * first = row(seen[t]), * prev = row(seen[2 + t]);
                auto * rel = ggml_sqrt(ctx0, ggml_div(ctx0, ggml_sum_rows(ctx0, ggml_sqr(ctx0, ggml_sub(ctx0, a, first))),
                        ggml_sum_rows(ctx0, ggml_sqr(ctx0, first))));
                auto * stats = ggml_concat(ctx0, ggml_concat(ctx0, cosine(a, first), cosine(a, prev), 0), rel, 0);
                ggml_format_name(stats, "loop_diag_kv_%c-%d", t ? 'V' : 'K', il);
                ggml_set_output(stats);
                ggml_build_forward_expand(gf, stats);
            }
            seen[2] = k;
            seen[3] = v;
        }
    }
    r = clip(ggml_reshape_3d(ctx0, r, hparams.n_relative, nh, n_tokens), l.attn_rel_clip, hparams.f_relative_clip, il);
    if (!swa && tau) {
        q = ggml_mul(ctx0, q, tau);
        r = ggml_mul(ctx0, r, tau);
    }
    auto * mask = swa ? attn->self_attn_mask_swa.raw : attn->self_attn_mask.raw;
    const int64_t streams = mask->ne[3];
    const int64_t n_kv = rel_idx[swa]->ne[0];
    auto * bias_weight = direct_sinks ? l.attn_rel_bias : ggml_pad(ctx0, l.attn_rel_bias, 0, 1, 0, 0);
    auto * table = ggml_mul_mat(ctx0, bias_weight, r);
    if (!direct_sinks && cparams.flash_attn) {
        // Shift all logits by the query's sink so flash attention can use a static zero sink.
        table = ggml_sub(ctx0, table, ggml_reshape_3d(ctx0, sinks, 1, nh, n_tokens));
        sinks = ggml_fill(ctx0, ggml_view_1d(ctx0, sinks, nh, 0), 0.0f);
    }
    ggml_tensor * attn_out;
    if (fused_rel) {
        const llm_graph_context::rel_pos rel_bias = {table, positions, rel_idx[swa], direct_sinks ? sinks : nullptr, direct_sinks ? n_kv_max[swa] : 0, low_mean[swa],
                kv_runs[swa], swa};
        attn_out = build_attn(attn, nullptr, nullptr, nullptr, q, k, v, nullptr, direct_sinks ? nullptr : sinks, nullptr, 1.0f/sqrtf(hd), il, nullptr, &rel_bias);
    } else {
        if (!cparams.flash_attn) {
            // build_attn scales the bias along with QK; FA adds the mask after scaling QK.
            table = ggml_scale(ctx0, table, sqrtf(hd));
        }
        table = ggml_cont(ctx0, ggml_permute(ctx0, table, 1, 0, 2, 3));
        auto * bias = ggml_get_rows(ctx0, table, rel_idx[swa]);
        bias = ggml_cont(ctx0, ggml_permute(ctx0, ggml_reshape_4d(ctx0, bias, nh, n_kv, n_tokens/streams, streams), 2, 0, 1, 3));
        ggml_tensor * fa_mask = nullptr;
        if (cparams.flash_attn) {
            fa_mask = ggml_cast(ctx0, ggml_add(ctx0, bias, ggml_cast(ctx0, mask, GGML_TYPE_F32)), GGML_TYPE_F16);
            bias = nullptr;
        } else {
            sinks = ggml_reshape_4d(ctx0, ggml_cont(ctx0, ggml_permute(ctx0,
                    ggml_reshape_3d(ctx0, sinks, nh, n_tokens/streams, streams), 1, 0, 2, 3)), 1, n_tokens/streams, nh, streams);
        }
        ggml_tensor * hot_k = attn->self_hot_k_idxs;
        ggml_tensor * hot_v = attn->self_hot_v_idxs;
        attn->self_hot_k_idxs = nullptr;
        attn->self_hot_v_idxs = nullptr;
        attn_out = build_attn(attn, nullptr, nullptr, nullptr, q, k, v, bias, sinks, nullptr, 1.0f/sqrtf(hd), il, fa_mask);
        attn->self_hot_k_idxs = hot_k;
        attn->self_hot_v_idxs = hot_v;
    }
    if (cparams.fused_inkling2) {
        // the output gate, sigmoid(gate)*attention in one op
        cur = ggml_siglu_split(ctx0, gate, attn_out);
        res->add_fused_node({LLM_FUSED_OP_INKLING2_LAYER, cur, il});
    } else {
        cur = ggml_mul(ctx0, attn_out, ggml_sigmoid(ctx0, gate));
    }
    cur = convolve_hidden(build_lora_mm(l.wo, cur), l.attn_post_conv, hidden, hidden_offset, hwindows, il);
    // the first layer of each block starts a new residual stream, the earlier one joins the mixed blocks
    if (ip % hparams.attn_res_block_size == 0) {
        blocks.push_back(residual);
        static const bool block_norm_cache_legacy = ggml_env_switch("LLAMA_INKLING2_BLOCK_NORM_CACHE_LEGACY");
        if (!block_norm_cache_legacy && cparams.fused_inkling2 && cparams.n_seq_devices <= 1) {
            block_norms.push_back({residual, ggml_rms_norm(ctx0, residual, 1e-6f)});
        }
    } else {
        cur = ggml_add(ctx0, residual, cur);
    }
    residual = cur;
    int64_t ffn_tokens = n_tokens;
    if (il == n_layer - 1 && !messages && prune_last_moe) {
        // Keep only the last prune_tail tokens per sequence for final MoE, conv state, and output.
        const int64_t ns = ubatch.n_seqs;
        const int64_t nt = ubatch.n_seq_tokens;
        auto slice_tail = [&](ggml_tensor * t) -> ggml_tensor * {
            if (ns == 1) {
                return ggml_view_2d(ctx0, t, n_embd, prune_tail, t->nb[1], (nt - prune_tail)*t->nb[1]);
            }
            auto * v = ggml_view_3d(ctx0, t, n_embd, prune_tail, ns, t->nb[1], nt*t->nb[1], (nt - prune_tail)*t->nb[1]);
            return ggml_cont_2d(ctx0, v, n_embd, prune_tail*ns);
        };
        residual = cur = slice_tail(cur);
        for (auto * & b : blocks) {
            ggml_tensor * bn = find_block_norm(b);
            if (bn) {
                ggml_tensor * bns = slice_tail(bn);
                b = bns;
                block_norms.push_back({bns, bns});
            } else {
                b = slice_tail(b);
            }
        }
        ffn_tokens = prune_tail * ns;
    }
    cur = ggml_reshape_2d(ctx0, mix(blocks, cur, l.ffn_res_query, il), n_embd, ffn_tokens);
    if (messages) {
        cur = control(cur, message(LOOP_CTRL_FFN), *loop, LOOP_CTRL_FFN, il);
        diag_gates(LOOP_CTRL_FFN);
    }
    cur = build_norm(cur, l.ffn_norm, nullptr, LLM_NORM_RMS, il);
    // -sm data: the rows are sequence-parallel, each device owning whole sequences, and every device's experts are the
    // experts of its own quarter. The experts read every row, so the rows are gathered before the product and each
    // device's sum comes back to the rows it owns (a reduce to owners: each device reads only its own rows of the
    // others' sums, one collective with the scatter's slice bounds).
    const bool seq_rows = cparams.n_seq_devices > 1;
    auto * moe_in = seq_rows ? ggml_backend_meta_gather(ctx0, cur) : cur;
    ggml_tensor * router_logits = nullptr;
    const auto & baked_routers = static_cast<const llama_model_inkling2 &>(model).baked_routers;
    if (!baked_routers.empty() && baked_routers[ip].weight && (!loras || loras->empty())) {
        const auto & router = baked_routers[ip];
        auto * projected = ggml_mul(ctx0, moe_in, router.signs);
        cb(projected, "ffn_moe_router_basis", il);
        router_logits = build_lora_mm(router.weight, projected);
        if (hparams.expert_router_f32) {
            ggml_prec_set_acc(router_logits, GGML_PREC_F32);
        }
        cb(router_logits, "ffn_moe_logits", il);
    }
    cur = build_moe_ffn(moe_in, l.ffn_gate_inp, nullptr, nullptr, l.ffn_down_exps, l.ffn_exp_probs_b,
            hparams.n_expert, hparams.n_expert_used(il), LLM_FFN_SITU, true, hparams.expert_weights_scale,
            LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID, il, router_logits, l.ffn_gate_up_exps,
            l.ffn_up_exps_s, l.ffn_gate_exps_s, l.ffn_down_exps_s);
    if (seq_rows) {
        const auto counts = ubatch.device_rows(cparams.n_seq_devices);
        cur = ggml_backend_meta_reduce(ctx0, cur);
        cur = ggml_backend_meta_scatter(ctx0, cur, 1, counts.data(), counts.size());
    } else if (model.experts_split_by_expert()) {
        // each card's sum holds only its own experts: summed across the cards as the split rows' sums are
        cur = ggml_backend_meta_reduce(ctx0, cur);
    }
    cur = convolve_hidden(cur, l.ffn_post_conv, hidden, hidden_offset + n_embd, hwindows, il);
    cur = ggml_add(ctx0, residual, cur);
    if (!(state_only && !cparams.embeddings_layer_inp[n_layer] && il == n_layer - 1)) {
        // Expand conv+add before store so short_conv_window and short_conv_add both fuse.
        ggml_build_forward_expand(gf, cur);
    }
    // the layer's new state in one store of each whole row: under -sm tensor the conv row is split by the KV heads in
    // a segment a stream, and each device's windows are its part of every segment in the same order
    auto store = [&](ggml_tensor * dst, const std::vector<ggml_tensor *> & parts) {
        auto * window = parts[0];
        for (size_t i = 1; i < parts.size(); ++i) {
            window = ggml_concat(ctx0, window, parts[i], 1);
        }
        build_conv_state_store(recr, dst, window, hparams.n_shortconv_l_cache - 1, 0);
    };
    store(state, windows);
    if (hidden_apart) {
        store(hidden_state, hidden_windows);
    }
    cb(cur, "l_out", il);
    return cur;
}

void llama_model_inkling2::graph::build_output(ggml_tensor * cur, const std::vector<ggml_tensor *> & blocks, bool candidates_only) {
    const bool tap_output = cparams.embeddings_layer_inp[n_layer];
    if (state_only && !tap_output) {
        // All attention and convolution state writes have been added to the graph.
        return;
    }
    ggml_tensor * out_ids = prune_last_moe ? pruned_out_ids : inp_out_ids;
    // A sparse prefill needs only the selected rows of each depth source for its final mix.
    const bool sparse_output = !tap_output && !prune_last_moe && cparams.fused_inkling2 && !cparams.pipeline_parallel &&
            cparams.n_seq_devices <= 1 && out_ids && n_outputs > 0 && cur->ne[1] >= 256 &&
            n_outputs*int64_t(blocks.size() + 1) < cur->ne[1];
    std::vector<ggml_tensor *> sources(blocks);
    if (sparse_output) {
        for (auto * & block : sources) {
            ggml_tensor * bn = find_block_norm(block);
            if (bn) {
                ggml_tensor * gathered = ggml_get_rows(ctx0, bn, out_ids);
                block = gathered;
                block_norms.push_back({gathered, gathered});
            } else {
                block = ggml_get_rows(ctx0, block, out_ids);
            }
        }
        cur = ggml_get_rows(ctx0, cur, out_ids);
    }
    cur = mix(sources, cur, model.output_res_query, -1);
    cur = ggml_reshape_2d(ctx0, cur, n_embd, cur->ne[2]);
    if (out_ids && !sparse_output && !tap_output) {
        // -sm data: the ids are the device's own rows (llama-graph.cpp, llm_graph_input_out_ids)
        cur = build_output_rows(cur, out_ids);
    }
    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    if (tap_output) {
        res->t_layer_inp[n_layer] = cur;
        ggml_build_forward_expand(gf, cur);
        if (state_only) {
            return;
        }
        if (out_ids) {
            cur = build_output_rows(cur, out_ids);
        }
    }
    res->t_embd = cur;
    // -sm tensor splits the padded vocabulary. The head's rows of the unpadded vocabulary are a view of each device's
    // rows, so each device keeps the logits of its share and the host reads every share where it lies (logits_gather).
    // LLAMA_INKLING2_LOGITS_GATHER_LEGACY=1: the padded logits gathered whole to every device, then sliced
    static const bool logits_gather_legacy = ggml_env_switch("LLAMA_INKLING2_LOGITS_GATHER_LEGACY");
    const bool split_vocab = model.split_mode() == LLAMA_SPLIT_MODE_TENSOR && !logits_gather_legacy;
    cur = split_vocab ?
        build_lora_mm(ggml_view_2d(ctx0, model.output, n_embd, hparams.n_vocab_unpadded, model.output->nb[1], 0), cur) :
        build_lora_mm(model.output, cur);
    ggml_prec_set_acc(cur, GGML_PREC_F32);
    const float cap = hparams.f_final_logit_softcapping;
    if (cap != 0.0f) {
        cur = ggml_scale(ctx0, ggml_tanh(ctx0, ggml_scale(ctx0, cur, 1.0f/cap)), cap);
    }
    if (candidates_only) {
        ggml_tensor * draw = candidates_sample ? build_inp_cand_draw() : nullptr;
        res->t_cand = ggml_top_k_candidates_sample(ctx0, cur, draw, cparams.n_out_candidates, cparams.n_logits_parts,
            hparams.n_vocab_unpadded, candidates_first_nan);
        cb(res->t_cand, "result_candidates", -1);
        ggml_build_forward_expand(gf, res->t_cand);
        return;
    }
    if (model.split_mode() == LLAMA_SPLIT_MODE_TENSOR && !split_vocab) {
        // the slice below cuts the padded vocabulary in no ratio of the split: slice it whole
        cur = gather_channels(ctx0, cur);
    }
    if (cur->ne[0] != hparams.n_vocab_unpadded) {
        cur = ggml_view_2d(ctx0, cur, hparams.n_vocab_unpadded, cur->ne[1], cur->nb[1], 0);
    }
    if (cparams.pipeline_parallel || !ggml_is_contiguous(cur)) {
        cur = ggml_cont(ctx0, cur);
    }
    res->t_logits = cur;
    cb(cur, "result_output", -1);
}

llama_model_inkling2::graph::graph(const llm_graph_params & params) : llm_graph_context(params) {
    auto * cur = build_inputs(params);
    std::vector<ggml_tensor *> blocks;
    for (int il = 0; il < n_layer; ++il) {
        cur = build_layer(il, il, cur, blocks);
    }
    build_output(cur, blocks, params.candidates_only);
}
