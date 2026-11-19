#include "models.h"
#include "inkling2-attention.h"
#include "../llama-memory-hybrid-iswa.h"

#include <algorithm>

void llama_model_inkling2::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key_or_arr(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp_arr, hparams.n_layer_all);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_ATTENTION_RELATIVE_LENGTH, hparams.n_relative);
    ml.get_key(LLM_KV_ATTENTION_RELATIVE_EXTENT, hparams.n_relative_extent);
    ml.get_key(LLM_KV_ATTENTION_QK_CLIP, hparams.f_qk_clip);
    ml.get_key(LLM_KV_ATTENTION_RELATIVE_CLIP, hparams.f_relative_clip);
    ml.get_key(LLM_KV_ATTENTION_LOG_SCALE_FLOOR, hparams.n_log_scale_floor);
    ml.get_key(LLM_KV_ATTENTION_LOG_SCALE_ALPHA, hparams.f_log_scale_alpha);
    ml.get_key(LLM_KV_ATTN_RES_BLOCK_SIZE, hparams.attn_res_block_size);
    ml.get_key(LLM_KV_SHORTCONV_L_CACHE, hparams.n_shortconv_l_cache);
    ml.get_key(LLM_KV_VOCAB_SIZE, hparams.n_vocab_padded);
    ml.get_key(LLM_KV_VOCAB_SIZE_UNPADDED, hparams.n_vocab_unpadded);
    ml.get_key(LLM_KV_ACTIVATION_SITU_BETA, hparams.situ_beta);
    ml.get_key(LLM_KV_ACTIVATION_SITU_LINEAR_BETA, hparams.situ_linear_beta);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE, hparams.expert_weights_scale);
    ml.get_key(LLM_KV_FINAL_LOGIT_SOFTCAPPING, hparams.f_final_logit_softcapping);
    ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW, hparams.n_swa);
    ml.get_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, hparams.is_swa_impl);
    uint32_t n_pattern = 0;
    ml.get_arr_n(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, n_pattern);
    if (n_pattern != hparams.n_layer() || !hparams.n_relative || !hparams.n_relative_extent || !hparams.n_swa ||
        !hparams.attn_res_block_size || !hparams.n_log_scale_floor || hparams.n_shortconv_l_cache < 2 ||
        hparams.n_shortconv_l_cache > 64 || hparams.n_relative_extent > INT32_MAX || hparams.n_swa > INT32_MAX ||
        !hparams.n_vocab_unpadded || hparams.n_vocab_unpadded > hparams.n_vocab_padded) {
        throw std::runtime_error("inkling2: invalid dimensions, vocabulary, or layer pattern");
    }
    for (float value : {hparams.f_norm_rms_eps, hparams.f_qk_clip, hparams.f_relative_clip, hparams.situ_beta,
                       hparams.situ_linear_beta, hparams.expert_weights_scale, hparams.f_final_logit_softcapping}) {
        if (!std::isfinite(value) || value <= 0.0f) {
            throw std::runtime_error("inkling2: normalization, clipping and scaling parameters must be finite and positive");
        }
    }
    if (!std::isfinite(hparams.f_log_scale_alpha) || hparams.f_log_scale_alpha < 0.0f) {
        throw std::runtime_error("inkling2: invalid logarithmic scale");
    }
    hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
    hparams.expert_router_f32 = true;
    hparams.expert_weights_norm_eps = 1e-20f;
    hparams.rope_pattern.fill(false);
    uint64_t max_channels = 0;
    for (uint32_t il = 0; il < hparams.n_layer(); ++il) {
        if (!hparams.n_head_kv(il) || hparams.n_head(il) % hparams.n_head_kv(il) ||
            hparams.n_embd_head_k(il) != hparams.n_embd_head_v(il) || !hparams.n_expert_used(il) ||
            hparams.n_expert_used(il) > hparams.n_expert || !hparams.n_ff_exp(il)) {
            throw std::runtime_error("inkling2: invalid attention or expert dimensions");
        }
        max_channels = std::max<uint64_t>(max_channels, 2ull * hparams.n_embd_k_gqa(il) + 2ull * hparams.n_embd);
    }
    if (max_channels * (hparams.n_shortconv_l_cache - 1) > UINT32_MAX) {
        throw std::runtime_error("inkling2: convolution state is too large");
    }
    hparams.n_sconv_channels = max_channels;
    type = LLM_TYPE_UNKNOWN;
}

void llama_model_inkling2::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;
    if (hparams.n_vocab_unpadded != vocab.n_tokens()) {
        throw std::runtime_error("inkling2: tokenizer size must equal the unpadded vocabulary size");
    }
    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, hparams.n_vocab_padded}, 0);
    tok_norm = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD_NORM, "weight", 0), {n_embd}, 0);
    output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), {n_embd, hparams.n_vocab_padded}, 0);
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output_res_query = create_tensor(tn(LLM_TENSOR_OUTPUT_RES_QUERY, "weight"), {n_embd}, 0);
    for (int i = 0; i < n_layer; ++i) {
        auto & l = layers[i];
        const int64_t nh = hparams.n_head(i);
        const int64_t nk = hparams.n_head_kv(i);
        const int64_t hd = hparams.n_embd_head_k(i);
        const int64_t nr = hparams.n_relative;
        const int64_t kernel = hparams.n_shortconv_l_cache;
        const int64_t extent = hparams.is_swa(i) ? hparams.n_swa : hparams.n_relative_extent;
        l.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q, "weight", i), {n_embd, nh*hd}, 0);
        l.wk = create_tensor(tn(LLM_TENSOR_ATTN_K, "weight", i), {n_embd, nk*hd}, 0);
        l.wv = create_tensor(tn(LLM_TENSOR_ATTN_V, "weight", i), {n_embd, nk*hd}, 0);
        l.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {nh*hd, n_embd}, 0);
        l.wqkv_gate = create_tensor(tn(LLM_TENSOR_ATTN_GATE, "weight", i), {n_embd, nh*hd}, 0);
        l.attn_rel = create_tensor(tn(LLM_TENSOR_ATTN_REL, "weight", i), {n_embd, nh*nr}, 0);
        l.attn_rel_bias = create_tensor(tn(LLM_TENSOR_ATTN_REL_BIAS, "weight", i), {nr, extent}, 0);
        l.attn_q_clip = create_tensor(tn(LLM_TENSOR_ATTN_Q_CLIP, "weight", i), {hd, nh}, 0);
        l.attn_rel_clip = create_tensor(tn(LLM_TENSOR_ATTN_REL_CLIP, "weight", i), {nr, nh}, 0);
        l.attn_sink_proj = create_tensor(tn(LLM_TENSOR_ATTN_SINK_PROJ, "weight", i), {n_embd, nh}, 0);
        l.attn_k_conv = create_tensor(tn(LLM_TENSOR_ATTN_K_CONV, "weight", i), {kernel, nk*hd}, 0);
        l.attn_v_conv = create_tensor(tn(LLM_TENSOR_ATTN_V_CONV, "weight", i), {kernel, nk*hd}, 0);
        l.attn_post_conv = create_tensor(tn(LLM_TENSOR_ATTN_POST_CONV, "weight", i), {kernel, n_embd}, 0);
        l.ffn_post_conv = create_tensor(tn(LLM_TENSOR_FFN_POST_CONV, "weight", i), {kernel, n_embd}, 0);
        if (hparams.is_swa(i)) {
            l.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);
        } else {
            l.attn_q_inp_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_INP_NORM, "weight", i), {n_embd}, 0);
            l.attn_k_inp_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_INP_NORM, "weight", i), {n_embd}, 0);
            l.attn_v_inp_norm = create_tensor(tn(LLM_TENSOR_ATTN_V_INP_NORM, "weight", i), {n_embd}, 0);
            l.attn_k_conv_proj = create_tensor(tn(LLM_TENSOR_ATTN_K_CONV_PROJ, "weight", i), {n_embd, kernel*nk}, 0);
            l.attn_v_conv_proj = create_tensor(tn(LLM_TENSOR_ATTN_V_CONV_PROJ, "weight", i), {n_embd, kernel*nk}, 0);
        }
        if (i > 0) {
            l.attn_res_query = create_tensor(tn(LLM_TENSOR_ATTN_RES_QUERY, "weight", i), {n_embd, hparams.is_swa(i) ? 1 : 3}, 0);
        }
        l.ffn_res_query = create_tensor(tn(LLM_TENSOR_FFN_RES_QUERY, "weight", i), {n_embd}, 0);
        l.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);
        l.ffn_gate_inp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP, "weight", i), {n_embd, n_expert}, 0);
        l.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias", i), {n_expert}, TENSOR_NOT_REQUIRED);
        l.ffn_gate_up_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_UP_EXPS, "weight", i), {n_embd, 2*hparams.n_ff_exp(i), n_expert}, 0);
        l.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {hparams.n_ff_exp(i), n_embd, n_expert}, 0);
    }
}

std::unique_ptr<llm_graph_context> llama_model_inkling2::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(params);
}

// the positions the relative bias is looked up by, for the global cache (0) and the window cache (1): with the bias inside
// the attention kernel the position of each cell, from which the kernel derives the distances, and with
// LLAMA_INKLING2_REL_POS_LEGACY=1 the distance of every query to every cell
class llm_graph_input_inkling2_pos : public llm_graph_input_i {
public:
    const llama_kv_cache_context * cache[2];
    ggml_tensor * pos[2];
    bool cells;
    uint32_t extent[2];
    inkling2_decode_mask mask_mode[2];
    bool mean_decode;

    inkling2_decode_mask classify_mask(const llm_graph_params & params, int i) const {
        const auto & ub = params.ubatch;
        const auto * kv = cache[i];
        if (!params.cparams.attn_mask_bits || !params.cparams.kv_unified || kv->is_full()) {
            return inkling2_decode_mask::dense;
        }
        if (ub.n_tokens <= 32 && kv->get_n_kv() > 256 && kv->type_k() == GGML_TYPE_F16 && kv->type_v() == GGML_TYPE_F16) {
            bool short_rows = true;
            for (uint32_t i = 0; i < ub.n_seqs_unq && short_rows; ++i) {
                const auto seq = ub.seq_id_unq[i];
                short_rows = kv->get_kv()->get_cells(seq).seq_n_cells(seq) <= 32;
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
        if (ub.n_tokens != ub.n_seqs_unq) {
            return inkling2_decode_mask::dense;
        }
        bool allow_mean = mean_decode && i == 0 && ub.n_tokens == 23 && kv->type_k() == GGML_TYPE_F16 && kv->type_v() == GGML_TYPE_F16;
        if (allow_mean) {
            for (uint32_t row = 0; row < ub.n_tokens; ++row) {
                if (ub.n_seq_id[row] != 1) {
                    allow_mean = false;
                    break;
                }
            }
        }
        return inkling2_classify_decode_mask(kv->get_n_kv(), ub.n_tokens, allow_mean, [&](uint32_t row) {
            const auto seq = ub.seq_id[row][0];
            return kv->get_kv()->get_cells(seq).seq_n_cells(seq);
        });
    }

    bool can_reuse(const llm_graph_params & params) override {
        const auto * memory = static_cast<const llama_memory_hybrid_iswa_context *>(params.mctx);
        cache[0] = memory->get_attn()->get_base();
        cache[1] = memory->get_attn()->get_swa();
        const int64_t n_stream = params.cparams.kv_unified ? 1 : params.ubatch.n_seqs_unq;
        for (int i = 0; i < 2; ++i) {
            if (pos[i]->ne[0] != cache[i]->get_n_kv() || pos[i]->ne[1] != (cells ? n_stream : params.ubatch.n_tokens) ||
                    (cells && mask_mode[i] != classify_mask(params, i))) {
                return false;
            }
        }
        return true;
    }

    void set_input(const llama_ubatch * ubatch) override {
        for (int i = 0; i < 2; ++i) {
            if (!pos[i]->buffer) {
                continue;
            }
            if (cells) {
                cache[i]->set_input_cell_pos(pos[i], ubatch);
            } else {
                cache[i]->set_input_relative_pos(pos[i], ubatch, extent[i]);
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

ggml_tensor * llama_model_inkling2::graph::mix(const std::vector<ggml_tensor *> & blocks, ggml_tensor * cur, ggml_tensor * query, int il) {
    if (cparams.fused_inkling2 && blocks.size() + 1 < GGML_MAX_SRC) {
        std::vector<ggml_tensor *> sources(blocks);
        sources.push_back(cur);
        cur = ggml_attn_res_mix(ctx0, query, sources.data(), sources.size(), 1.0f/sqrtf(n_embd), 1e-6f);
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
    const int64_t nt = ubatch.n_seq_tokens;
    const int64_t ns = ubatch.n_seqs;
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
        return ggml_reshape_2d(ctx0, out, channels, n_tokens);
    }
    auto * out = ggml_ssm_conv(ctx0, x, weight);
    out = ggml_reshape_2d(ctx0, out, channels, n_tokens);
    if (dynamic) {
        const int64_t heads = dynamic->ne[0]/kernel;
        // the lags outermost: a lag's taps are one run, at an offset -sm tensor scales as each device's heads are split
        // (an offset inside a head's taps, which the split axis holds, would be scaled with them)
        dynamic = ggml_cont(ctx0, ggml_permute(ctx0, ggml_reshape_3d(ctx0, dynamic, kernel, heads, n_tokens), 2, 0, 1, 3));
        for (int64_t lag = 0; lag < kernel; ++lag) {
            auto * tap = ggml_view_3d(ctx0, dynamic, 1, heads, n_tokens, dynamic->nb[0], dynamic->nb[1], lag*dynamic->nb[2]);
            auto * window = ggml_view_3d(ctx0, x, nt, channels, ns, x->nb[1], x->nb[2], (kernel - 1 - lag)*x->nb[0]);
            window = ggml_cont(ctx0, ggml_transpose(ctx0, window));
            window = ggml_reshape_3d(ctx0, window, channels/heads, heads, n_tokens);
            auto * term = ggml_reshape_2d(ctx0, ggml_mul(ctx0, window, tap), channels, n_tokens);
            out = ggml_add(ctx0, out, term);
        }
    }
    auto * residual = ggml_is_contiguous(input) ? ggml_reshape_2d(ctx0, input, channels, n_tokens)
                                                : ggml_cont_2d(ctx0, input, channels, n_tokens);
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

// -sm tensor: a hidden stream comes out of its reduction whole on every device, while the kernel and the state of its
// convolution are split by channel (llama_meta_device_get_split_state). Each device convolves its own channels, which
// a scatter gives it with the channels outermost so that they are one run of the tensor, and the result is gathered.
ggml_tensor * llama_model_inkling2::graph::convolve_hidden(ggml_tensor * cur, ggml_tensor * weight, ggml_tensor * history,
        int64_t offset, std::vector<ggml_tensor *> & windows, int il) {
    if (model.split_mode() != LLAMA_SPLIT_MODE_TENSOR) {
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

llama_model_inkling2::graph::graph(const llm_graph_params & params) : llm_graph_context(params) {
    const bool fused_rel = cparams.flash_attn && cparams.fused_rel_pos;
    const bool direct_sinks = fused_rel && !ggml_env_switch("LLAMA_INKLING2_SHIFT_SINKS");
    // The unfused graph adds the relative bias to a dense mask before attention.
    auto * inp = build_inp_mem_hybrid_iswa(fused_rel);
    auto * attn = inp->get_attn();
    auto * recr = inp->get_recr();
    auto * positions = build_inp_pos();
    const bool state_only = cparams.fused_inkling2 && n_outputs == 0 && !cparams.embeddings && !cparams.pipeline_parallel && cparams.n_seq_devices <= 1;
    auto * out_ids = state_only ? nullptr : build_inp_out_ids();
    auto rel = std::make_unique<llm_graph_input_inkling2_pos>();
    rel->cache[0] = attn->mctx->get_base();
    rel->cache[1] = attn->mctx->get_swa();
    rel->extent[0] = hparams.n_relative_extent;
    rel->extent[1] = hparams.n_swa;
    rel->cells = fused_rel;
    rel->mean_decode = direct_sinks && ggml_env_switch("LLAMA_INKLING2_MEAN_MASK_DECODE");
    const int64_t n_stream = cparams.kv_unified ? 1 : ubatch.n_seqs_unq;
    ggml_tensor * rel_pos[2];
    int32_t n_kv_max[2];
    bool low_mean[2];
    for (int i = 0; i < 2; ++i) {
        const auto mode = rel->mask_mode[i] = fused_rel ? rel->classify_mask(params, i) : inkling2_decode_mask::dense;
        n_kv_max[i] = mode == inkling2_decode_mask::short_rows ? 32 : mode == inkling2_decode_mask::bounded_rows ? rel->cache[i]->get_n_kv()/8 : 0;
        low_mean[i] = mode == inkling2_decode_mask::low_mean;
        rel_pos[i] = rel->pos[i] = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, rel->cache[i]->get_n_kv(), rel->cells ? n_stream : n_tokens);
        ggml_set_input(rel_pos[i]);
    }
    res->add_input(std::move(rel));
    auto * tau = ggml_scale_bias(ctx0, ggml_cast(ctx0, positions, GGML_TYPE_F32), 1.0f, 1.0f);
    tau = ggml_log(ctx0, ggml_clamp(ctx0, ggml_scale(ctx0, tau, 1.0f/hparams.n_log_scale_floor), 1.0f, INFINITY));
    tau = ggml_reshape_3d(ctx0, ggml_scale_bias(ctx0, tau, hparams.f_log_scale_alpha, 1.0f), 1, 1, n_tokens);
    auto * cur = build_inp_embd(model.tok_embd);
    // important: do not normalize raw embeddings input (i.e. encoded image embeddings)
    if (ubatch.token) {
        cur = build_norm(cur, model.tok_norm, nullptr, LLM_NORM_RMS, -1);
    }
    std::vector<ggml_tensor *> blocks;
    for (int il = 0; il < n_layer; ++il) {
        const auto & l = model.layers[il];
        const bool swa = hparams.is_swa(il);
        const int64_t nh = hparams.n_head(il);
        const int64_t nk = hparams.n_head_kv(il);
        const int64_t hd = hparams.n_embd_head_k(il);
        const int64_t kv = nk*hd;
        auto * state = recr->mctx->get_r_l(il);
        auto * history = build_rs(recr, state, hparams.n_embd_r(), ubatch.n_seqs);
        history = ggml_reshape_3d(ctx0, history, hparams.n_shortconv_l_cache - 1, hparams.n_sconv_channels, ubatch.n_seqs);
        auto * residual = cur;
        ggml_tensor * inputs[3] = {cur, cur, cur};
        if (l.attn_res_query) {
            auto * mixed = mix(blocks, cur, l.attn_res_query, il);
            for (int j = 0; j < (swa ? 1 : 3); ++j) {
                inputs[j] = ggml_view_2d(ctx0, mixed, n_embd, n_tokens, mixed->nb[2], j*mixed->nb[1]);
            }
        }
        if (swa) {
            inputs[0] = build_norm(inputs[0], l.attn_norm, nullptr, LLM_NORM_RMS, il);
            inputs[1] = inputs[2] = inputs[0];
        } else {
            inputs[0] = build_norm(inputs[0], l.attn_q_inp_norm, nullptr, LLM_NORM_RMS, il);
            inputs[1] = build_norm(inputs[1], l.attn_k_inp_norm, nullptr, LLM_NORM_RMS, il);
            inputs[2] = build_norm(inputs[2], l.attn_v_inp_norm, nullptr, LLM_NORM_RMS, il);
        }
        auto * q = build_lora_mm(l.wq, inputs[0]);
        // the windows of the state's streams, K, V and the two hidden streams, in the order of the state's channels
        std::vector<ggml_tensor *> windows;
        auto * k = convolve(build_lora_mm(l.wk, inputs[1]), l.attn_k_conv,
                swa ? nullptr : build_lora_mm(l.attn_k_conv_proj, inputs[1]), history, 0, windows, il);
        auto * v = convolve(build_lora_mm(l.wv, inputs[2]), l.attn_v_conv,
                swa ? nullptr : build_lora_mm(l.attn_v_conv_proj, inputs[2]), history, kv, windows, il);
        auto * r = build_lora_mm(l.attn_rel, inputs[0]);
        q = clip(ggml_reshape_3d(ctx0, q, hd, nh, n_tokens), l.attn_q_clip, hparams.f_qk_clip, il);
        k = clip(ggml_reshape_3d(ctx0, k, hd, nk, n_tokens), nullptr, hparams.f_qk_clip, il);
        v = ggml_rms_norm(ctx0, ggml_reshape_3d(ctx0, v, hd, nk, n_tokens), hparams.f_norm_rms_eps);
        r = clip(ggml_reshape_3d(ctx0, r, hparams.n_relative, nh, n_tokens), l.attn_rel_clip, hparams.f_relative_clip, il);
        if (!swa) {
            q = ggml_mul(ctx0, q, tau);
            r = ggml_mul(ctx0, r, tau);
        }
        auto * mask = swa ? attn->self_attn_mask_swa.raw : attn->self_attn_mask.raw;
        const int64_t streams = mask->ne[3];
        const int64_t n_kv = rel_pos[swa]->ne[0];
        auto * sinks = build_lora_mm(l.attn_sink_proj, inputs[0]);
        auto * bias_weight = direct_sinks ? l.attn_rel_bias : ggml_pad(ctx0, l.attn_rel_bias, 0, 1, 0, 0);
        auto * table = ggml_mul_mat(ctx0, bias_weight, r);
        if (!direct_sinks && cparams.flash_attn) {
            // Shift all logits by the query's sink so flash attention can use a static zero sink.
            table = ggml_sub(ctx0, table, ggml_reshape_3d(ctx0, sinks, 1, nh, n_tokens));
            sinks = ggml_fill(ctx0, ggml_view_1d(ctx0, sinks, nh, 0), 0.0f);
        }
        if (fused_rel) {
            const llm_graph_context::rel_pos rel_bias = {table, positions, rel_pos[swa], direct_sinks ? sinks : nullptr, direct_sinks ? n_kv_max[swa] : 0, low_mean[swa]};
            cur = build_attn(attn, nullptr, nullptr, nullptr, q, k, v, nullptr, direct_sinks ? nullptr : sinks, nullptr, 1.0f/sqrtf(hd), il, nullptr, &rel_bias);
        } else {
            if (!cparams.flash_attn) {
                // build_attn scales the bias along with QK; FA adds the mask after scaling QK.
                table = ggml_scale(ctx0, table, sqrtf(hd));
            }
            table = ggml_cont(ctx0, ggml_permute(ctx0, table, 1, 0, 2, 3));
            auto * bias = ggml_get_rows(ctx0, table, rel_pos[swa]);
            bias = ggml_cont(ctx0, ggml_permute(ctx0, ggml_reshape_4d(ctx0, bias, nh, n_kv, n_tokens/streams, streams), 2, 0, 1, 3));
            ggml_tensor * fa_mask = nullptr;
            if (cparams.flash_attn) {
                fa_mask = ggml_cast(ctx0, ggml_add(ctx0, bias, ggml_cast(ctx0, mask, GGML_TYPE_F32)), GGML_TYPE_F16);
                bias = nullptr;
            } else {
                sinks = ggml_reshape_4d(ctx0, ggml_cont(ctx0, ggml_permute(ctx0,
                        ggml_reshape_3d(ctx0, sinks, nh, n_tokens/streams, streams), 1, 0, 2, 3)), 1, n_tokens/streams, nh, streams);
            }
            cur = build_attn(attn, nullptr, nullptr, nullptr, q, k, v, bias, sinks, nullptr, 1.0f/sqrtf(hd), il, fa_mask);
        }
        auto * gate = build_lora_mm(l.wqkv_gate, inputs[0]);
        if (cparams.fused_inkling2) {
            // the output gate, sigmoid(gate)*attention in one op
            cur = ggml_siglu_split(ctx0, gate, cur);
            res->add_fused_node({LLM_FUSED_OP_INKLING2_LAYER, cur, il});
        } else {
            cur = ggml_mul(ctx0, cur, ggml_sigmoid(ctx0, gate));
        }
        cur = convolve_hidden(build_lora_mm(l.wo, cur), l.attn_post_conv, history, 2*kv, windows, il);
        if (il % hparams.attn_res_block_size == 0) {
            blocks.push_back(residual);
        } else {
            cur = ggml_add(ctx0, residual, cur);
        }
        residual = cur;
        cur = ggml_reshape_2d(ctx0, mix(blocks, cur, l.ffn_res_query, il), n_embd, n_tokens);
        cur = build_norm(cur, l.ffn_norm, nullptr, LLM_NORM_RMS, il);
        cur = build_moe_ffn(cur, l.ffn_gate_inp, nullptr, nullptr, l.ffn_down_exps, l.ffn_exp_probs_b,
                hparams.n_expert, hparams.n_expert_used(il), LLM_FFN_SITU, true, hparams.expert_weights_scale,
                LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID, il, nullptr, l.ffn_gate_up_exps);
        cur = convolve_hidden(cur, l.ffn_post_conv, history, 2*kv + n_embd, windows, il);
        // the layer's new state in one store of its whole row: under -sm tensor the row is split by the KV heads in four
        // segments, one a stream, and each device's windows are its part of every segment in the same order
        auto * window = windows[0];
        for (size_t i = 1; i < windows.size(); ++i) {
            window = ggml_concat(ctx0, window, windows[i], 1);
        }
        build_conv_state_store(recr, state, window, hparams.n_shortconv_l_cache - 1, 0);
        cur = ggml_add(ctx0, residual, cur);
        cb(cur, "l_out", il);
    }
    if (state_only) {
        // All attention and convolution state writes have been added to the graph.
        return;
    }
    // A sparse prefill needs only the selected rows of each depth source for its final mix.
    const bool sparse_output = cparams.fused_inkling2 && !cparams.pipeline_parallel && cparams.n_seq_devices <= 1 &&
            out_ids && n_outputs > 0 && n_tokens >= 256 && n_outputs*int64_t(blocks.size() + 1) < n_tokens;
    if (sparse_output) {
        for (auto * & block : blocks) {
            block = ggml_get_rows(ctx0, block, out_ids);
        }
        cur = ggml_get_rows(ctx0, cur, out_ids);
    }
    cur = mix(blocks, cur, model.output_res_query, -1);
    cur = ggml_reshape_2d(ctx0, cur, n_embd, cur->ne[2]);
    if (out_ids && !sparse_output) {
        cur = ggml_get_rows(ctx0, cur, out_ids);
    }
    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    res->t_embd = cur;
    cur = build_lora_mm(model.output, cur);
    ggml_prec_set_acc(cur, GGML_PREC_F32);
    const float cap = hparams.f_final_logit_softcapping;
    cur = ggml_scale(ctx0, ggml_tanh(ctx0, ggml_scale(ctx0, cur, 1.0f/cap)), cap);
    if (model.split_mode() == LLAMA_SPLIT_MODE_TENSOR) {
        // -sm tensor splits the padded vocabulary, which the slice below cuts in no ratio of the split: slice it whole
        cur = gather_channels(ctx0, cur);
    }
    cur = ggml_view_2d(ctx0, cur, hparams.n_vocab_unpadded, cur->ne[1], cur->nb[1], 0);
    if (cparams.pipeline_parallel || !ggml_is_contiguous(cur)) {
        cur = ggml_cont(ctx0, cur);
    }
    res->t_logits = cur;
    cb(cur, "result_output", -1);
}
