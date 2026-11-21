#include "models.h"

#include <cmath>

#include "llama-impl.h"
#include "llama-kv-cache.h"
#include "llama-kv-cache-iswa.h"

bool llama_model_is_inkling_draft(const llama_model & model) {
    return model.arch == LLM_ARCH_DFLASH && static_cast<const llama_model_dflash &>(model).dspark_markov_rank > 0;
}

void llama_model_dflash::load_arch_hparams(llama_model_loader & ml) {

    ml.get_key(LLM_KV_EMBEDDING_SCALE, hparams.f_embedding_scale, false);
    ml.get_key(LLM_KV_ATTENTION_SCALE, hparams.f_attention_scale, false);
    ml.get_key(LLM_KV_ATTENTION_VALUE_SCALE, hparams.f_attn_value_scale, false);

    hparams.llm_ffn_op = LLM_FFN_SILU;
    std::string hidden_act;
    if (ml.get_key(LLM_KV_HIDDEN_ACT, hidden_act, false)) {
        if (hidden_act == "gelu" || hidden_act == "gelu_pytorch_tanh") {
            hparams.llm_ffn_op = LLM_FFN_GELU;
        } else if (hidden_act == "swigluoai") {
            hparams.llm_ffn_op = LLM_FFN_SWIGLU_OAI_MOE;
        } else if (hidden_act != "silu") {
            throw std::runtime_error("unsupported DFlash hidden activation: " + hidden_act);
        }
    }

    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_LOGIT_SCALE,                 hparams.f_logit_scale, false);
    hparams.f_final_logit_softcapping = 0.0f;
    ml.get_key(LLM_KV_FINAL_LOGIT_SOFTCAPPING,     hparams.f_final_logit_softcapping, false);

    // drafts for M-RoPE targets carry degenerate sections [n_rot/2, 0, 0, 0]
    ml.get_key_or_arr(LLM_KV_ROPE_DIMENSION_SECTIONS, hparams.rope_sections, 4, false);

    ml.get_key(LLM_KV_DFLASH_BLOCK_SIZE,       hparams.dflash_block_size,       false);
    ml.get_key(LLM_KV_DFLASH_CONV_KERNEL_SIZE, hparams.dflash_conv_kernel_size, false);
    ml.get_key(LLM_KV_DFLASH_CONV_GROUP_SIZE,  hparams.dflash_conv_group_size,  false);
    ml.get_key(LLM_KV_DFLASH_SELECTOR_RANK,    hparams.dflash_selector_rank,    false);
    ml.get_key(LLM_KV_DFLASH_SELECTOR_TOP_K,   hparams.dflash_selector_top_k,   false);

    const bool has_markov   = ml.get_tensor_meta("markov_w1.weight")       != nullptr;

    // Reject a declared confidence head when its required Markov head is missing.
    bool kv_confidence_head = false;
    if (ml.get_key(LLM_KV_CONFIDENCE_HEAD, kv_confidence_head, false) && kv_confidence_head && !has_markov) {
        throw std::runtime_error("dflash: metadata declares a confidence head, but markov_w1.weight is missing. "
                                 "The export is incomplete; it would load as plain DFlash and read drafts one row late");
    }

    const bool has_selector = ml.get_tensor_meta("selector_hidden.weight") != nullptr;
    const bool has_lilicorr = ml.get_key(LLM_KV_LILICORR_HIDDEN_SIZE, hparams.lilicorr_hidden_size, false);
    if (has_markov) {
        variant = VARIANT_DSPARK;
    } else if (has_selector) {
        variant = VARIANT_DFLASH2;
    } else if (has_lilicorr) {
        variant = VARIANT_LILICORR;
        load_hparams_lilicorr(ml);
    }

    if (!ml.get_arr(LLM_KV_TARGET_LAYERS, target_layer_ids, false)) {
        throw std::runtime_error("DFlash model requires 'target_layers' in GGUF metadata");
    }

    if (ml.get_key(LLM_KV_DSPARK_MARKOV_RANK, dspark_markov_rank, false)) {
        if (target_layer_ids.empty() || target_layer_ids.size() > LLAMA_MAX_LAYERS + 1 ||
                target_layer_ids.size() * (uint64_t) hparams.n_embd > UINT32_MAX || !dspark_markov_rank || !has_markov) {
            throw std::runtime_error("invalid DFlash target feature dimensions");
        }
    }
    hparams.n_embd_inp_enc_impl = (uint32_t) target_layer_ids.size() * hparams.n_embd;

    // dspark GIDD log-SNR conditioning (drafters trained with the GIDD bundle);
    // absent on every other drafter, so it must default off
    ml.get_key(LLM_KV_LOG_SNR_CONDITIONING, hparams.dspark_log_snr_conditioning, false);
    if (hparams.dspark_log_snr_conditioning) {
        // required once the flag is set: defaulting either bound to 0 collapses
        // (max - min) in the featurizer and fills the input with NaNs instead of
        // failing to load
        ml.get_key(LLM_KV_MIN_LOG_SNR, hparams.dspark_min_log_snr, true);
        ml.get_key(LLM_KV_MAX_LOG_SNR, hparams.dspark_max_log_snr, true);
        if (!std::isfinite(hparams.dspark_min_log_snr) || !std::isfinite(hparams.dspark_max_log_snr)) {
            throw std::runtime_error("dspark log-SNR conditioning: min/max_log_snr must be finite");
        }
        if (!(hparams.dspark_max_log_snr > hparams.dspark_min_log_snr)) {
            throw std::runtime_error("dspark log-SNR conditioning: max_log_snr must be greater than min_log_snr");
        }
    }

    std::string layers;
    const char * sep = "";
    for (const auto id : target_layer_ids) {
        layers += sep;
        layers += std::to_string(id);
        sep = ", ";
    }
    LLAMA_LOG_INFO("%s: DFlash extract_layers = [%s]\n", __func__, layers.c_str());

    if (dspark_markov_rank) {
        ml.get_key(LLM_KV_VOCAB_SIZE, hparams.n_vocab_padded);
        ml.get_key(LLM_KV_VOCAB_SIZE_UNPADDED, hparams.n_vocab_unpadded);
        ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW, hparams.n_swa);
        ml.get_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, hparams.is_swa_impl);
        uint32_t n_pattern = 0;
        ml.get_arr_n(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, n_pattern);
        if (hparams.dflash_block_size != 8 ||
                dspark_markov_rank > hparams.n_embd || !hparams.n_swa || hparams.n_swa >= INT32_MAX ||
                n_pattern != hparams.n_layer() || target_layer_ids.size() < 2 ||
                target_layer_ids.front() < 0 || target_layer_ids.back() > LLAMA_MAX_LAYERS ||
                std::adjacent_find(target_layer_ids.begin(), target_layer_ids.end(), std::greater_equal<int32_t>()) != target_layer_ids.end() ||
                !hparams.n_vocab_unpadded || hparams.n_vocab_unpadded > hparams.n_vocab_padded || hparams.n_vocab_unpadded > (1u << 24) ||
                !std::isfinite(hparams.f_norm_rms_eps) || hparams.f_norm_rms_eps <= 0 ||
                !std::isfinite(hparams.f_final_logit_softcapping) || hparams.f_final_logit_softcapping <= 0) {
            throw std::runtime_error("invalid Inkling2 DSpark metadata");
        }
        for (uint32_t il = 0; il < hparams.n_layer(); ++il) {
            if (!hparams.n_head_kv(il) || hparams.n_head(il) % hparams.n_head_kv(il) ||
                    hparams.n_embd_head_k(il) != hparams.n_embd_head_v(il) || !hparams.n_ff(il) ||
                    hparams.n_rot(il) != hparams.n_embd_head_k(il) || hparams.n_rot(il) % 2) {
                throw std::runtime_error("invalid Inkling2 DSpark attention dimensions");
            }
        }
        // The anchor row includes the anchor and the window before it.
        hparams.n_swa += 1;
        hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
        hparams.rope_freq_base_train_swa = hparams.rope_freq_base_train;
        hparams.rope_freq_scale_train_swa = hparams.rope_freq_scale_train;
        type = LLM_TYPE_UNKNOWN;
        return;
    }

    // DeepSeek-V4 DSpark backbone: stages are full DSV4 blocks, uniform sliding window (the draft KV ring)
    ml.get_key(LLM_KV_HYPER_CONNECTION_COUNT, hparams.dsv4_hc_mult, false);
    if (hparams.dsv4_hc_mult > 0) {
        load_hparams_dsv4(ml);
        backbone = hparams.dsv4_hc_single_pass ? BACKBONE_DSV41 : BACKBONE_DSV4;
    } else if (ml.get_key(LLM_KV_ATTENTION_KV_LORA_RANK, hparams.n_lora_kv, false)) {
        load_hparams_mistral_large4(ml);
        backbone = BACKBONE_MISTRAL_LARGE4;
    } else if (ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW, hparams.n_swa, false) && hparams.n_swa > 0) {
        // optional interleaved sliding-window attention with per-layer pattern array.
        // DFlash has a single rope, so the SWA rope == main rope.
        hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
        ml.get_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, hparams.is_swa_impl);
        hparams.rope_freq_base_train_swa  = hparams.rope_freq_base_train;
        hparams.rope_freq_scale_train_swa = hparams.rope_freq_scale_train;
    }

    ml.get_key_or_arr(LLM_KV_ATTENTION_ROPE_PATTERN, hparams.rope_pattern, hparams.n_layer(), false);

    type = LLM_TYPE_UNKNOWN;
}

void llama_model_dflash::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    if (dspark_markov_rank) {
        const int64_t rank = dspark_markov_rank;
        if (hparams.n_vocab_unpadded != (uint32_t) n_vocab) {
            throw std::runtime_error("Inkling2 DSpark vocabulary size mismatch");
        }
        tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, hparams.n_vocab_padded}, TENSOR_NOT_REQUIRED);
        tok_norm = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD_NORM, "weight", 0), {n_embd}, TENSOR_NOT_REQUIRED);
        output   = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), {n_embd, hparams.n_vocab_padded}, TENSOR_NOT_REQUIRED);
        if ((tok_embd || tok_norm || output) && !(tok_embd && tok_norm && output)) {
            throw std::runtime_error("Inkling2 DSpark needs all three target vocabulary tensors or none");
        }
        output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
        dspark_mask_embd = create_tensor(tn(LLM_TENSOR_DSPARK_MASK_EMBD, "weight"), {n_embd}, 0);
        dspark_fuse = create_tensor(tn(LLM_TENSOR_DSPARK_FUSE, "weight"), {(int64_t) target_layer_ids.size(), n_layer + 1}, 0);
        dspark_norm = create_tensor(tn(LLM_TENSOR_DSPARK_NORM, "weight"), {n_embd}, 0);
        dspark_markov_w1 = create_tensor(tn(LLM_TENSOR_DSPARK_MARKOV_W1, "weight"), {rank, hparams.n_vocab_padded}, 0);
        dspark_joint_proj = create_tensor(tn(LLM_TENSOR_DSPARK_JOINT_PROJ, "weight"), {2*rank + 2*n_embd, 3*rank}, 0);
        dspark_joint_proj_b = create_tensor(tn(LLM_TENSOR_DSPARK_JOINT_PROJ, "bias"), {3*rank}, 0);
        dspark_hidden_out = create_tensor(tn(LLM_TENSOR_DSPARK_HIDDEN_OUT, "weight"), {rank, n_embd}, 0);
        dspark_hidden_out_b = create_tensor(tn(LLM_TENSOR_DSPARK_HIDDEN_OUT, "bias"), {n_embd}, 0);
        inkling_layers.resize(n_layer);
        for (int i = 0; i < n_layer; ++i) {
            auto & l = layers[i];
            auto & ctx = inkling_layers[i];
            const int64_t hd = hparams.n_embd_head_k(i);
            const int64_t q = hd*hparams.n_head(i);
            const int64_t kv = hd*hparams.n_head_kv(i);
            l.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);
            l.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);
            ctx.norm = create_tensor(tn(LLM_TENSOR_DFLARE_NORM, "weight", i), {n_embd}, 0);
            l.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q, "weight", i), {n_embd, q}, 0);
            l.wk = create_tensor(tn(LLM_TENSOR_ATTN_K, "weight", i), {n_embd, kv}, 0);
            l.wv = create_tensor(tn(LLM_TENSOR_ATTN_V, "weight", i), {n_embd, kv}, 0);
            l.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {q, n_embd}, 0);
            ctx.k = create_tensor(tn(LLM_TENSOR_ATTN_K_CTX, "weight", i), {n_embd, kv}, 0);
            ctx.v = create_tensor(tn(LLM_TENSOR_ATTN_V_CTX, "weight", i), {n_embd, kv}, 0);
            l.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {hd}, 0);
            l.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {hd}, 0);
            ctx.k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_CTX_NORM, "weight", i), {hd}, 0);
            l.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd, hparams.n_ff(i)}, 0);
            l.ffn_up = create_tensor(tn(LLM_TENSOR_FFN_UP, "weight", i), {n_embd, hparams.n_ff(i)}, 0);
            l.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {hparams.n_ff(i), n_embd}, 0);
        }
        return;
    }

    const int64_t n_embd_inp = hparams.n_embd_inp_enc();

    tok_embd        = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD,       "weight"), { n_embd, n_vocab }, TENSOR_NOT_REQUIRED);

    // reduced draft vocab (optional): d2t maps draft rows to target token ids
    int64_t n_vocab_draft = n_vocab;
    const struct ggml_tensor * d2t_meta = ml->get_tensor_meta("d2t");
    if (d2t_meta) {
        n_vocab_draft = d2t_meta->ne[0];
        d2t = create_tensor(tn(LLM_TENSOR_D2T), { n_vocab_draft }, 0);
        LLAMA_LOG_INFO("%s: DFlash using d2t mapping (draft_vocab_size = %lld)\n", __func__, (long long) n_vocab_draft);
    }

    switch (variant) {
        case VARIANT_DSPARK:   load_tensors_dspark(n_vocab_draft); break;
        case VARIANT_DFLASH2:  load_tensors_dflash2();             break;
        case VARIANT_LILICORR: load_tensors_lilicorr();            break;
        case VARIANT_DFLASH:                                       break;
    }

    // dspark GIDD log-SNR conditioning (LogSnrEmbed): unlike the markov head, this
    // is built into the decoder graph and changes the draft embedding every forward
    // pass, so when the GGUF says it is on the weights are REQUIRED. A missing
    // tensor here is a broken conversion, not something to degrade past silently.
    if (hparams.dspark_log_snr_conditioning) {
        const int64_t n_freq = 128; // matches LogSnrEmbed.NUM_FREQ_FEATURES
        dspark_log_snr_fc1_w = create_tensor(tn(LLM_TENSOR_DSPARK_LOG_SNR_FC1, "weight"), { n_freq, n_embd }, 0);
        dspark_log_snr_fc1_b = create_tensor(tn(LLM_TENSOR_DSPARK_LOG_SNR_FC1, "bias"),   { n_embd },         0);
        dspark_log_snr_fc2_w = create_tensor(tn(LLM_TENSOR_DSPARK_LOG_SNR_FC2, "weight"), { n_embd, n_embd }, 0);
        dspark_log_snr_fc2_b = create_tensor(tn(LLM_TENSOR_DSPARK_LOG_SNR_FC2, "bias"),   { n_embd },         0);
        LLAMA_LOG_INFO("%s: DFlash with DSpark log-SNR conditioning (min %.3f, max %.3f)\n",
                __func__, (double) hparams.dspark_min_log_snr, (double) hparams.dspark_max_log_snr);
    }

    fc              = create_tensor(tn(LLM_TENSOR_FC,              "weight"), { n_embd_inp, n_embd }, 0);
    fc_s            = create_tensor(tn(LLM_TENSOR_FC,              "scale"),  { 1 }, TENSOR_NOT_REQUIRED);
    fc_in_s         = create_tensor(tn(LLM_TENSOR_FC,              "input_scale"), { 1 }, TENSOR_NOT_REQUIRED);
    output_norm_enc = create_tensor(tn(LLM_TENSOR_ENC_OUTPUT_NORM, "weight"), { n_embd }, 0); // encoder hidden_norm (after fc)
    output_norm     = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM,    "weight"), { n_embd }, 0); // decoder final norm

    // optional: reduced-vocab drafts ship their own lm head, full-vocab drafts can share the target's via ctx_other
    // a draft with its own embeddings + head references no target tensors and can run on devices the target does not use (e.g. -devd with a tensor-split target)
    output   = create_tensor(tn(LLM_TENSOR_OUTPUT,     "weight"), { n_embd, n_vocab_draft }, TENSOR_NOT_REQUIRED);

    switch (backbone) {
        case BACKBONE_BASE:           load_tensors_layers_base();           break;
        case BACKBONE_DSV4:
        case BACKBONE_DSV41:          load_tensors_layers_dsv4();           break;
        case BACKBONE_MISTRAL_LARGE4: load_tensors_layers_mistral_large4(); break;
    }
}

void llama_model_dflash::load_tensors_layers_base() {
    LLAMA_LOAD_LOCALS;

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), { n_embd }, 0);

        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), { n_embd, n_embd_head_k * n_head }, 0);
        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), { n_embd, n_embd_k_gqa }, 0);
        layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), { n_embd, n_embd_v_gqa }, TENSOR_NOT_REQUIRED);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), { n_embd_head_k * n_head, n_embd }, 0);

        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), { n_embd_head_k }, 0);
        layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), { n_embd_head_k }, 0);

        layer.attn_post_norm = create_tensor(tn(LLM_TENSOR_ATTN_POST_NORM, "weight", i), { n_embd }, TENSOR_NOT_REQUIRED);
        layer.ffn_post_norm  = create_tensor(tn(LLM_TENSOR_FFN_POST_NORM,  "weight", i), { n_embd }, TENSOR_NOT_REQUIRED);
        layer.out_scale      = create_tensor(tn(LLM_TENSOR_LAYER_OUT_SCALE, "weight", i), { 1 }, TENSOR_NOT_REQUIRED);
        layer.rope_freqs     = create_tensor(tn(LLM_TENSOR_ROPE_FREQS, "weight", i), { n_embd_head_k/2 }, TENSOR_NOT_REQUIRED | (i > 0 ? TENSOR_DUPLICATED : 0));

        // optional per-head attention sinks (e.g. Nemotron DSpark)
        layer.attn_sinks = create_tensor(tn(LLM_TENSOR_ATTN_SINKS, "weight", i), { n_head }, TENSOR_NOT_REQUIRED);

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), { n_embd }, 0);
        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), { n_embd, n_ff }, 0);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), { n_ff, n_embd }, 0);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), { n_embd, n_ff }, 0);
    }
}

template <>
ggml_tensor * llama_model_dflash::graph<true>::build_inp_embd_enc() const {
    const int64_t n_embd_inp = hparams.n_embd_inp_enc();
    auto inp_target = std::make_unique<llm_graph_input_embd>(n_embd_inp);

    inp_target->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd_inp, n_tokens);
    ggml_set_input(inp_target->embd);

    ggml_tensor * cur = inp_target->embd;
    cb(cur, "inp_embd", -1);

    res->add_input(std::move(inp_target));

    return cur;
}

// DFlash Encoder: processes target model features through feature fusion layer
template <>
llama_model_dflash::graph<true>::graph(const llm_graph_params & params) : llm_graph_context(params) {
    ggml_tensor * cur = build_inp_embd_enc();

    cur = build_lora_mm(model.fc, cur, model.fc_s);
    cb(cur, "fc_out", -1);

    cur = build_norm(cur, model.output_norm_enc, NULL, LLM_NORM_RMS, -1);
    cb(cur, "enc_norm_out", -1);

    ggml_set_output(cur);
    res->t_h_nextn = cur;

    ggml_build_forward_expand(gf, cur);
}

// target features of an injection batch: host embd rows, or rows of the target's layer input device buffer
static ggml_tensor * build_inp_target_features(llm_graph_context & g, llm_graph_input_embd & inp, int64_t n_embd_inp) {
    if (g.ubatch.embd_row) {
        GGML_ASSERT(g.layer_inp_src != nullptr && g.layer_inp_src->ne[0] == n_embd_inp);

        inp.embd_rows = ggml_new_tensor_1d(g.ctx0, GGML_TYPE_I32, g.n_tokens);
        ggml_set_input(inp.embd_rows);

        return ggml_get_rows(g.ctx0, g.layer_inp_src, inp.embd_rows);
    }

    inp.embd = ggml_new_tensor_2d(g.ctx0, GGML_TYPE_F32, n_embd_inp, g.n_tokens);
    ggml_set_input(inp.embd);

    return inp.embd;
}

// injection batch: the target features fused through the encoder
ggml_tensor * llama_model_dflash::build_inject_inp(llm_graph_context & g, const llama_model & model) {
    const int64_t n_embd_inp = g.hparams.n_embd_inp_enc();

    auto inp = std::make_unique<llm_graph_input_embd>(n_embd_inp);

    ggml_tensor * inp_target = build_inp_target_features(g, *inp, n_embd_inp);
    g.cb(inp_target, "inp_target_features", -1);

    g.res->add_input(std::move(inp));

    ggml_tensor * inp_g = g.build_lora_mm(model.fc, inp_target, model.fc_s);
    return g.build_norm(inp_g, model.output_norm_enc, nullptr, LLM_NORM_RMS, -1);
}

// tok_embd from the target model (shared via ctx_other) when the draft has none
ggml_tensor * llama_model_dflash::get_tok_embd(const llm_graph_context & g, const llama_model & model) {
    if (model.tok_embd) {
        return model.tok_embd;
    }
    GGML_ASSERT(g.cparams.ctx_other != nullptr);
    const auto * model_other = llama_get_model(g.cparams.ctx_other);

    GGML_ASSERT(model_other->tok_embd != nullptr && "DFlash draft requires the target model's token embeddings");
    return model_other->tok_embd;
}

// lm_head and its scale from the target model (shared via ctx_other) when the draft has none
std::pair<ggml_tensor *, ggml_tensor *> llama_model_dflash::get_lm_head(const llm_graph_context & g, const llama_model & model) {
    if (model.output) {
        return { model.output, model.output_s };
    }
    GGML_ASSERT(g.cparams.ctx_other != nullptr);
    const auto * model_other = llama_get_model(g.cparams.ctx_other);

    GGML_ASSERT(model_other->output != nullptr && "DFlash draft requires the target model's output projection");
    return { model_other->output, model_other->output_s };
}

// DFlash decoder, dual-mode by batch type:
//   * embd batch  -> fused target features: project + inject K/V into the cache.
//   * token batch -> noise-block diffusion: attend over [committed, MASK...] to generate draft tokens
llama_model_dflash::graph_base::graph_base(const llm_graph_params & params) : llm_graph_context(params) {
    // Shared Bonsai embeddings/head are stored in a rotated basis. The draft's
    // own tensors are unrotated, so only shared target tensor pointers match.
    if (cparams.ctx_other && (!model.tok_embd || !model.output)) {
        const auto * target = llama_get_model(cparams.ctx_other);
        GGML_ASSERT(model.hdmd.rot.empty() && model.hdmd.inv.empty());
        hdmd = target->hdmd.rot.empty() && target->hdmd.inv.empty() ? nullptr : &target->hdmd;
    }

    n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    inp_pos = build_inp_pos();

    // optional iSWA: pick the matching attention input
    use_iswa = hparams.swa_type != LLAMA_SWA_TYPE_NONE;
    if (use_iswa) {
        inp_attn_iswa = build_attn_inp_kv_iswa();
    } else {
        inp_attn = build_attn_inp_kv();
    }

    // drafts for M-RoPE targets use degenerate sections (temporal dim only)
    std::copy(std::begin(hparams.rope_sections), std::begin(hparams.rope_sections) + 4, sections);
}

ggml_tensor * llama_model_dflash::graph_base::build_rope(ggml_tensor * cur, ggml_tensor * pos, int il) {
    if (!hparams.has_rope(il)) {
        return cur;
    }

    return rope_type == GGML_ROPE_TYPE_MROPE
        ? ggml_rope_multi(ctx0, cur, pos, nullptr,
                n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow)
        : ggml_rope_ext(ctx0, cur, pos, model.layers[0].rope_freqs,
                n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow);
}

// KV cache injection
ggml_tensor * llama_model_dflash::graph_base::build_inject() {
    ggml_tensor * inp_g = build_inject_inp(*this, model);
    cb(inp_g, "inp_g_embeddings", -1);

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];

        ggml_tensor * Kcur = build_lora_mm(layer.wk, inp_g, layer.wk_s);
        const bool shared_kv = layer.wv == nullptr;
        ggml_tensor * Vcur = shared_kv ? Kcur : build_lora_mm(layer.wv, inp_g, layer.wv_s);

        Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
        Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

        Kcur = build_norm(Kcur, layer.attn_k_norm, NULL, LLM_NORM_RMS, il);
        if (shared_kv) {
            Vcur = ggml_rms_norm(ctx0, Vcur, hparams.f_norm_rms_eps);
        }
        Kcur = build_rope(Kcur, inp_pos, il);
        cb(Kcur, "Kcur_injected", il);
        cb(Vcur, "Vcur_injected", il);

        if (use_iswa) {
            // route each layer's K/V to its sub-cache: SWA layers -> sliding cache, full -> dense
            const bool    is_swa = hparams.is_swa(il);
            const auto  * kv     = is_swa ? inp_attn_iswa->mctx->get_swa() : inp_attn_iswa->mctx->get_base();
            ggml_tensor * k_idxs = is_swa ? inp_attn_iswa->get_k_idxs_swa() : inp_attn_iswa->get_k_idxs();
            ggml_tensor * v_idxs = is_swa ? inp_attn_iswa->get_v_idxs_swa() : inp_attn_iswa->get_v_idxs();
            // rotate K/V into the cache's rotated space
            ggml_tensor * k_rot  = is_swa ? inp_attn_iswa->self_k_rot_swa[il] : inp_attn_iswa->self_k_rot[il];
            ggml_tensor * v_rot  = is_swa ? inp_attn_iswa->self_v_rot_swa[il] : inp_attn_iswa->self_v_rot[il];
            if (k_rot) {
                Kcur = llama_mul_mat_hadamard(ctx0, Kcur, k_rot);
            }
            if (v_rot) {
                Vcur = llama_mul_mat_hadamard(ctx0, Vcur, v_rot);
            }
            ggml_build_forward_expand(gf, kv->cpy_k(ctx0, Kcur, k_idxs, il));
            ggml_build_forward_expand(gf, kv->cpy_v(ctx0, Vcur, v_idxs, il));
        } else {
            // rotate K/V into the cache's rotated space
            if (inp_attn->self_k_rot[il]) {
                Kcur = llama_mul_mat_hadamard(ctx0, Kcur, inp_attn->self_k_rot[il]);
            }
            if (inp_attn->self_v_rot[il]) {
                Vcur = llama_mul_mat_hadamard(ctx0, Vcur, inp_attn->self_v_rot[il]);
            }
            ggml_build_forward_expand(gf, inp_attn->mctx->cpy_k(ctx0, Kcur, inp_attn->get_k_idxs(), il));
            ggml_build_forward_expand(gf, inp_attn->mctx->cpy_v(ctx0, Vcur, inp_attn->get_v_idxs(), il));
        }
    }

    res->t_embd = inp_g;

    ggml_build_forward_expand(gf, inp_g);

    return inp_g;
}

ggml_tensor * llama_model_dflash::graph_base::build_inp_tokens() {
    ggml_tensor * inp_tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp_tokens);
    res->t_inp_tokens = inp_tokens;

    auto inp = std::make_unique<llm_graph_input_embd>(n_embd);
    inp->tokens = inp_tokens;
    res->add_input(std::move(inp));

    return inp_tokens;
}

ggml_tensor * llama_model_dflash::graph_base::build_layers(ggml_tensor * inp_tokens) {
    const float kq_scale = hparams.f_attention_scale != 0.0f ? hparams.f_attention_scale : 1.0f/sqrtf(float(n_embd_head));

    ggml_tensor * tok_embd = get_tok_embd(*this, model);
    ggml_tensor * inpL = ggml_get_rows(ctx0, tok_embd, inp_tokens);
    if (hdmd) {
        if (const auto it = hdmd->inv.find(tok_embd); it != hdmd->inv.end()) {
            inpL = llama_mul_mat_hadamard(ctx0, inpL, it->second.rot);
            if (it->second.signs) inpL = ggml_mul(ctx0, inpL, it->second.signs);
        }
    }
    if (hparams.f_embedding_scale != 0.0f) {
        inpL = ggml_scale(ctx0, inpL, hparams.f_embedding_scale);
    }
    cb(inpL, "inp_noise_embd", -1);

    if (hparams.dspark_log_snr_conditioning) {
        inpL = build_dspark_log_snr(inpL);
    }

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];

        ggml_tensor * noise_norm = build_norm(inpL, layer.attn_norm, NULL, LLM_NORM_RMS, il);
        cb(noise_norm, "noise_norm", il);

        ggml_tensor * attn_dynamic = nullptr;
        if (layer.dflash_attn_conv_proj) {
            attn_dynamic = build_lora_mm(layer.dflash_attn_conv_proj, noise_norm);
            noise_norm = build_dflash2_conv(*this, noise_norm, attn_dynamic, layer.dflash_attn_conv_base, 0);
            cb(noise_norm, "attn_conv_in", il);
        }

        ggml_tensor * Qcur = build_lora_mm(layer.wq, noise_norm, layer.wq_s);
        ggml_tensor * Kcur = build_lora_mm(layer.wk, noise_norm, layer.wk_s);
        const bool shared_kv = layer.wv == nullptr;
        ggml_tensor * Vcur = shared_kv ? Kcur : build_lora_mm(layer.wv, noise_norm, layer.wv_s);

        Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
        Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
        Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

        Qcur = build_norm(Qcur, layer.attn_q_norm, NULL, LLM_NORM_RMS, il);
        Kcur = build_norm(Kcur, layer.attn_k_norm, NULL, LLM_NORM_RMS, il);
        if (shared_kv) {
            Vcur = ggml_rms_norm(ctx0, Vcur, hparams.f_norm_rms_eps);
        }

        Qcur = build_rope(Qcur, inp_pos, il);
        Kcur = build_rope(Kcur, inp_pos, il);
        cb(Qcur, "Qcur", il);
        cb(Kcur, "Kcur", il);
        cb(Vcur, "Vcur", il);

        // cache-aware, non-causal attention
        ggml_tensor * cur = use_iswa
            ? build_attn(inp_attn_iswa, layer.wo, NULL, layer.wo_s, Qcur, Kcur, Vcur, nullptr, layer.attn_sinks, nullptr, kq_scale, il)
            : build_attn(inp_attn,      layer.wo, NULL, layer.wo_s, Qcur, Kcur, Vcur, nullptr, layer.attn_sinks, nullptr, kq_scale, il);

        if (hparams.f_attn_value_scale != 0.0f) {
            cur = ggml_scale(ctx0, cur, hparams.f_attn_value_scale);
            cb(cur, "attn_out_scaled", il);
        }

        if (attn_dynamic) {
            cur = build_dflash2_conv(*this, cur, attn_dynamic, layer.dflash_attn_conv_base, 1);
            cb(cur, "attn_conv_out", il);
        }

        if (layer.attn_post_norm) {
            cur = build_norm(cur, layer.attn_post_norm, NULL, LLM_NORM_RMS, il);
            cb(cur, "attn_post_norm", il);
        }

        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpL);
        cb(ffn_inp, "ffn_inp", il);

        cur = build_norm(ffn_inp, layer.ffn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        ggml_tensor * ffn_dynamic = nullptr;
        if (layer.dflash_ffn_conv_proj) {
            ffn_dynamic = build_lora_mm(layer.dflash_ffn_conv_proj, cur);
            cur = build_dflash2_conv(*this, cur, ffn_dynamic, layer.dflash_ffn_conv_base, 0);
            cb(cur, "ffn_conv_in", il);
        }

        cur = build_ffn(cur,
                layer.ffn_up,   NULL, layer.ffn_up_s,
                layer.ffn_gate, NULL, layer.ffn_gate_s,
                layer.ffn_down, NULL, layer.ffn_down_s,
                NULL,
                hparams.llm_ffn_op, LLM_FFN_PAR, il);
        cb(cur, "ffn_out", il);

        if (ffn_dynamic) {
            cur = build_dflash2_conv(*this, cur, ffn_dynamic, layer.dflash_ffn_conv_base, 1);
            cb(cur, "ffn_conv_out", il);
        }

        if (layer.ffn_post_norm) {
            cur = build_norm(cur, layer.ffn_post_norm, NULL, LLM_NORM_RMS, il);
            cb(cur, "ffn_post_norm", il);
        }

        cur = ggml_add(ctx0, cur, ffn_inp);
        if (layer.out_scale) {
            cur = ggml_mul(ctx0, cur, layer.out_scale);
        }
        cb(cur, "l_out", il);

        inpL = cur;
    }

    ggml_tensor * cur = build_norm(inpL, model.output_norm, NULL, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);

    res->t_embd = cur;

    return cur;
}

ggml_tensor * llama_model_dflash::graph_base::build_logits(ggml_tensor * cur) {
    const auto [output, output_s] = get_lm_head(*this, model);
    cur = build_lora_mm(output, cur, output_s);

    // DFlash2 feeds these logits to the selector, so they need the target's output
    // transforms; DFlash1 and DSpark read them through the sampler instead
    if (model.dflash_selector_hidden) {
        if (hparams.f_logit_scale != 0.0f) {
            cur = ggml_scale(ctx0, cur, hparams.f_logit_scale);
        }
        if (hparams.f_final_logit_softcapping > 0.0f) {
            cur = ggml_scale(ctx0, cur, 1.0f / hparams.f_final_logit_softcapping);
            cur = ggml_tanh(ctx0, cur);
            cur = ggml_scale(ctx0, cur, hparams.f_final_logit_softcapping);
        }
    }

    // reduced-draft-vocab exports: scatter the draft logits to the target vocabulary via d2t
    if (model.d2t) {
        const int64_t n_draft_vocab = cur->ne[0];
        const int64_t n_outputs     = cur->ne[1];
        const int64_t n_vocab       = (int64_t) model.vocab.n_tokens();

        GGML_ASSERT(model.d2t->type == GGML_TYPE_I64);
        GGML_ASSERT(model.d2t->ne[0] == n_draft_vocab);

        ggml_tensor * logits = ggml_fill(ctx0, ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 1, n_vocab, n_outputs), -INFINITY);
        cur = ggml_set_rows(ctx0, logits,
                ggml_reshape_3d(ctx0, cur,       1,             n_draft_vocab, n_outputs),
                ggml_reshape_3d(ctx0, model.d2t, n_draft_vocab, 1,             1));
        cur = ggml_reshape_2d(ctx0, cur, n_vocab, n_outputs);
    }
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    return cur;
}

llama_model_dflash::graph_dflash::graph_dflash(const llm_graph_params & params) : graph_base(params) {
    ASSERT_EMBD_OR_TOKEN(ubatch);
    if (ubatch.embd || ubatch.embd_row) {
        build_inject();
        return;
    }

    ggml_tensor * inp_tokens = build_inp_tokens();
    ggml_tensor * cur        = build_layers(inp_tokens);

    build_logits(cur);
}

static ggml_tensor * build_inkling2_head(llm_graph_context & g, const llama_model_dflash & model, ggml_tensor * output,
        ggml_tensor * output_s, ggml_tensor * prev, ggml_tensor * & state, ggml_tensor * hidden,
        ggml_tensor * base, ggml_tensor * context) {
    auto * ctx0 = g.ctx0;
    const int64_t rank = model.dspark_markov_rank;
    auto * joint = ggml_concat(ctx0, state, ggml_get_rows(ctx0, model.dspark_markov_w1, prev), 0);
    joint = ggml_concat(ctx0, joint, ggml_concat(ctx0, hidden, context, 0), 0);
    joint = ggml_add(ctx0, g.build_lora_mm(model.dspark_joint_proj, joint), model.dspark_joint_proj_b);
    const auto chunk = [&](int64_t j) {
        return ggml_cont(ctx0, ggml_view_2d(ctx0, joint, rank, joint->ne[1], joint->nb[1], j*rank*joint->nb[0]));
    };
    auto * gate = ggml_sigmoid(ctx0, chunk(0));
    auto * candidate = ggml_tanh(ctx0, chunk(1));
    state = ggml_add(ctx0, candidate, ggml_mul(ctx0, gate, ggml_sub(ctx0, state, candidate)));
    auto * latent = ggml_tanh(ctx0, chunk(2));
    ggml_tensor * bias;
    if (model.dspark_head_out && output == model.output && g.loras->empty()) {
        // head x (W x latent + b) as (head x [W | b]) x [latent; 1], made at load (load_tensors_dspark_head)
        auto * ones = ggml_fill(ctx0, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 1, latent->ne[1]), 1.0f);
        bias = ggml_mul_mat(ctx0, model.dspark_head_out, ggml_concat(ctx0, latent, ones, 0));
    } else {
        auto * correction = ggml_add(ctx0, g.build_lora_mm(model.dspark_hidden_out, latent), model.dspark_hidden_out_b);
        bias = g.build_lora_mm(output, correction, output_s);
    }
    ggml_prec_set_acc(bias, GGML_PREC_F32);
    if (bias->ne[0] != base->ne[0]) {
        bias = ggml_cont(ctx0, ggml_view_2d(ctx0, bias, base->ne[0], bias->ne[1], bias->nb[1], 0));
    }
    auto * logits = ggml_add(ctx0, base, bias);
    const float cap = model.hparams.f_final_logit_softcapping;
    return ggml_scale(ctx0, ggml_tanh(ctx0, ggml_scale(ctx0, logits, 1.0f/cap)), cap);
}

// Draws the token of each column of logits [vocab, nb] from its k largest at the column's inverse temperature: the k in
// token id order, the first whose cumulative probability is above the column's uniform. An inverse temperature of 0 keeps
// first, the column's first maximum. The host draws the same way from the same k candidates (common/speculative.cpp,
// dspark_draw), so the token a step conditions on is the token the host proposes.
static ggml_tensor * build_inkling2_draw(ggml_context * ctx0, ggml_tensor * logits, ggml_tensor * first,
        ggml_tensor * uniform, ggml_tensor * inv_temp, int64_t k) {
    const int64_t vocab = logits->ne[0];
    const int64_t nb = logits->ne[1];
    auto * top = ggml_top_k(ctx0, logits, k);
    auto * order = ggml_argsort(ctx0, ggml_cast(ctx0, top, GGML_TYPE_F32), GGML_SORT_ORDER_ASC);
    auto * ids = ggml_get_rows(ctx0, ggml_reshape_3d(ctx0, top, 1, k, nb), order);
    auto * vals = ggml_get_rows(ctx0, ggml_reshape_3d(ctx0, logits, 1, vocab, nb), ggml_reshape_2d(ctx0, ids, k, nb));
    auto * probs = ggml_soft_max(ctx0, ggml_mul(ctx0, ggml_reshape_2d(ctx0, vals, k, nb), inv_temp));
    auto * cdf = ggml_cumsum(ctx0, probs);
    // the uniform is scaled to the sum the round-off leaves, so the token drawn always has a probability above 0
    auto * total = ggml_cont(ctx0, ggml_view_2d(ctx0, cdf, 1, nb, cdf->nb[1], (k - 1)*cdf->nb[0]));
    auto * above = ggml_step(ctx0, ggml_sub(ctx0, cdf, ggml_mul(ctx0, uniform, total)));
    auto * pick = ggml_scale_bias(ctx0, ggml_clamp(ctx0, ggml_sum_rows(ctx0, above), 1.0f, (float) k), -1.0f, (float) k);
    auto * drawn = ggml_get_rows(ctx0, ggml_cast(ctx0, ids, GGML_TYPE_F32), ggml_cast(ctx0, pick, GGML_TYPE_I32));
    drawn = ggml_reshape_2d(ctx0, drawn, 1, nb);
    auto * first_f = ggml_reshape_2d(ctx0, ggml_cast(ctx0, first, GGML_TYPE_F32), 1, nb);
    auto * token = ggml_add(ctx0, first_f, ggml_mul(ctx0, ggml_step(ctx0, inv_temp), ggml_sub(ctx0, drawn, first_f)));
    return ggml_reshape_1d(ctx0, ggml_cast(ctx0, token, GGML_TYPE_I32), nb);
}

static int64_t inkling2_draft_rows(const llama_ubatch & ubatch, int64_t n_outputs) {
    const int64_t n_seq = ubatch.n_seqs_unq;
    const int64_t block = ubatch.n_tokens/n_seq;
    if (n_outputs == ubatch.n_tokens || n_outputs % n_seq != 0) {
        return block;
    }

    const int64_t prefix = n_outputs/n_seq;
    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        if (bool(ubatch.output[i]) != (i % block < prefix)) {
            return block;
        }
    }
    return prefix;
}

struct llm_graph_input_inkling2_draft : llm_graph_input_i {
    explicit llm_graph_input_inkling2_draft(int64_t rows) : rows(rows) {}

    void set_input(const llama_ubatch *) override {}

    bool can_reuse(const llm_graph_params & params) override {
        return rows == inkling2_draft_rows(params.ubatch, params.n_outputs);
    }

    const int64_t rows;
};

// Inkling2 uses separate context projections and one feature mixture per draft layer.
llama_model_dflash::graph_inkling2::graph_inkling2(const llm_graph_params & params) : llm_graph_context(params) {
    auto * positions = build_inp_pos();
    auto * attn = build_attn_inp_kv_iswa();
    const auto & draft = static_cast<const llama_model_dflash &>(model);
    const int64_t rank = draft.dspark_markov_rank;
    const int64_t ntarget = model.target_layer_ids.size();

    if (ubatch.embd || ubatch.embd_row) {
        auto inp = std::make_unique<llm_graph_input_embd>(hparams.n_embd_inp_enc());
        auto * features = build_inp_target_features(*this, *inp, hparams.n_embd_inp_enc());
        res->add_input(std::move(inp));
        features = ggml_reshape_3d(ctx0, features, n_embd, ntarget, n_tokens);
        features = ggml_cont(ctx0, ggml_permute(ctx0, features, 1, 0, 2, 3));
        const auto mix = [&](int row, ggml_tensor * norm) {
            auto * weights = ggml_soft_max(ctx0, ggml_view_1d(ctx0, draft.dspark_fuse, ntarget, row*draft.dspark_fuse->nb[1]));
            auto * x = ggml_sum_rows(ctx0, ggml_mul(ctx0, features, weights));
            return build_norm(ggml_reshape_2d(ctx0, x, n_embd, n_tokens), norm, nullptr, LLM_NORM_RMS, row == n_layer ? -1 : row);
        };
        for (int il = 0; il < n_layer; ++il) {
            const auto & ctx = draft.inkling_layers[il];
            const int64_t hd = hparams.n_embd_head_k(il);
            const int64_t nk = hparams.n_head_kv(il);
            auto * context = mix(il, ctx.norm);
            cb(context, "dspark_context", il);
            auto * k = ggml_reshape_3d(ctx0, build_lora_mm(ctx.k, context), hd, nk, n_tokens);
            auto * v = ggml_reshape_3d(ctx0, build_lora_mm(ctx.v, context), hd, nk, n_tokens);
            k = build_norm(k, ctx.k_norm, nullptr, LLM_NORM_RMS, il);
            k = build_rope(k, positions, il);
            const bool swa = hparams.is_swa(il);
            const auto * kv = swa ? attn->mctx->get_swa() : attn->mctx->get_base();
            auto * k_rot = swa ? attn->self_k_rot_swa[il] : attn->self_k_rot[il];
            auto * v_rot = swa ? attn->self_v_rot_swa[il] : attn->self_v_rot[il];
            if (k_rot) {
                k = llama_mul_mat_hadamard(ctx0, k, k_rot);
            }
            if (v_rot) {
                v = llama_mul_mat_hadamard(ctx0, v, v_rot);
            }
            ggml_build_forward_expand(gf, kv->cpy_k(ctx0, k, swa ? attn->get_k_idxs_swa() : attn->get_k_idxs(), il));
            ggml_build_forward_expand(gf, kv->cpy_v(ctx0, v, swa ? attn->get_v_idxs_swa() : attn->get_v_idxs(), il));
        }
        res->t_embd = res->t_h_nextn = mix(n_layer, draft.dspark_norm);
        cb(res->t_h_nextn, "dspark_f_rnn", -1);
        ggml_build_forward_expand(gf, res->t_h_nextn);
        return;
    }

    const auto * target = cparams.ctx_other ? llama_get_model(cparams.ctx_other) : nullptr;
    auto * tok_embd = get_tok_embd(*this, model);
    auto * tok_norm = model.tok_norm ? model.tok_norm : target->tok_norm;
    const auto [output, output_s] = get_lm_head(*this, model);
    GGML_ASSERT(tok_norm);
    auto inp = std::make_unique<llm_graph_input_embd_h>(n_embd, n_embd);
    auto * tokens = inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(tokens);
    auto * f_rnn = inp->h = build_inp_embd_state(inp.get(), n_embd);
    res->add_input(std::move(inp));

    const int64_t nb = ubatch.n_seqs_unq;
    const int64_t block = n_tokens/nb;
    auto * ids = ggml_cont_1d(ctx0, ggml_view_2d(ctx0, tokens, 1, nb, block*tokens->nb[0], 0), nb);
    auto * anchors = ggml_get_rows(ctx0, tok_embd, ids);
    const float tok_norm_eps = target ? target->hparams.f_norm_rms_eps : hparams.f_norm_rms_eps;
    anchors = ggml_mul(ctx0, ggml_rms_norm(ctx0, anchors, tok_norm_eps), tok_norm);
    auto * cur = ggml_reshape_3d(ctx0, anchors, n_embd, 1, nb);
    if (block > 1) {
        auto * masks = ggml_repeat_4d(ctx0, ggml_cast(ctx0, model.dspark_mask_embd, GGML_TYPE_F32), n_embd, block - 1, nb, 1);
        cur = ggml_concat(ctx0, cur, masks, 1);
    }
    cur = ggml_reshape_2d(ctx0, cur, n_embd, n_tokens);
    attn->self_attn_mask_swa.block_uniform = true;
    for (int il = 0; il < n_layer; ++il) {
        const auto & l = model.layers[il];
        const int64_t hd = hparams.n_embd_head_k(il);
        auto * residual = cur;
        cur = build_norm(cur, l.attn_norm, nullptr, LLM_NORM_RMS, il);
        auto * q = ggml_reshape_3d(ctx0, build_lora_mm(l.wq, cur), hd, hparams.n_head(il), n_tokens);
        auto * k = ggml_reshape_3d(ctx0, build_lora_mm(l.wk, cur), hd, hparams.n_head_kv(il), n_tokens);
        auto * v = ggml_reshape_3d(ctx0, build_lora_mm(l.wv, cur), hd, hparams.n_head_kv(il), n_tokens);
        q = build_rope(build_norm(q, l.attn_q_norm, nullptr, LLM_NORM_RMS, il), positions, il);
        k = build_rope(build_norm(k, l.attn_k_norm, nullptr, LLM_NORM_RMS, il), positions, il);
        cur = build_attn(attn, l.wo, nullptr, nullptr, q, k, v, nullptr, nullptr, nullptr, 1.0f/sqrtf(hd), il);
        residual = ggml_add(ctx0, residual, cur);
        cur = build_norm(residual, l.ffn_norm, nullptr, LLM_NORM_RMS, il);
        cur = build_ffn(cur, l.ffn_up, nullptr, nullptr, l.ffn_gate, nullptr, nullptr,
                l.ffn_down, nullptr, nullptr, nullptr, LLM_FFN_SILU, LLM_FFN_PAR, il);
        cur = ggml_add(ctx0, residual, cur);
        cb(cur, "dspark_layer", il);
    }
    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    res->t_embd = cur;
    if (n_outputs == 0) {
        return;
    }

    const int64_t head_rows = inkling2_draft_rows(ubatch, n_outputs);
    if (n_outputs < n_tokens && n_outputs % nb == 0) {
        res->add_input(std::make_unique<llm_graph_input_inkling2_draft>(head_rows));
    }
    // A uniform output prefix needs fewer head positions.
    if (head_rows < block) {
        cur = ggml_view_3d(ctx0, cur, n_embd, head_rows, nb, cur->nb[1], block*cur->nb[1], 0);
        cur = ggml_cont_2d(ctx0, cur, n_embd, head_rows*nb);
    }
    auto * base = build_lora_mm(output, cur, output_s);
    ggml_prec_set_acc(base, GGML_PREC_F32);
    auto * state = ggml_fill(ctx0, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, rank, nb), 0.0f);
    const auto column = [&](ggml_tensor * t, int64_t i, int64_t stride) {
        return ggml_cont(ctx0, ggml_view_2d(ctx0, t, t->ne[0], nb, stride*t->nb[1], i*t->nb[1]));
    };
    auto * logits = column(base, 0, head_rows);
    auto * context = column(f_rnn, 0, block);
    ggml_tensor * tie_rank = nullptr;
    const int64_t n_draw = std::min<int64_t>(cparams.n_out_candidates, hparams.n_vocab_unpadded);
    auto * prev = ids;
    for (int64_t i = 1; i < head_rows; ++i) {
        auto * col = build_inkling2_head(*this, draft, output, output_s, prev, state,
                column(cur, i, head_rows), column(base, i, head_rows), context);
        logits = ggml_concat(ctx0, logits, col, 1);
        if (i + 1 == head_rows) {
            break;
        }
        // Select the first maximum, independent of backend argmax tie order.
        const int64_t vocab = hparams.n_vocab_unpadded;
        if (!tie_rank) {
            tie_rank = ggml_scale_bias(ctx0, ggml_arange(ctx0, 0, vocab, 1), -1.0f, vocab);
        }
        auto * valid = ggml_cont(ctx0, ggml_view_2d(ctx0, col, vocab, nb, col->nb[1], 0));
        auto * max_ids = ggml_reshape_2d(ctx0, ggml_argmax(ctx0, valid), 1, nb);
        auto * maximum = ggml_get_rows(ctx0, ggml_reshape_3d(ctx0, valid, 1, vocab, nb), max_ids);
        auto * below = ggml_sub(ctx0, ggml_repeat(ctx0, ggml_reshape_2d(ctx0, maximum, 1, nb), valid), valid);
        auto * is_max = ggml_scale_bias(ctx0, ggml_step(ctx0, below), -1.0f, 1.0f);
        prev = ggml_argmax(ctx0, ggml_mul(ctx0, is_max, tie_rank));
        // the host reads k > 1 candidates a row when it samples the draft: a block row past the anchor carries its
        // step's uniform and the sequence's inverse temperature in the first two values of its state
        if (n_draw > 1) {
            const auto draw_in = [&](int64_t j) {
                return ggml_cont(ctx0, ggml_view_2d(ctx0, f_rnn, 1, nb, block*f_rnn->nb[1], i*f_rnn->nb[1] + j*f_rnn->nb[0]));
            };
            prev = build_inkling2_draw(ctx0, valid, prev, draw_in(0), draw_in(1), n_draw);
        }
    }
    logits = ggml_reshape_3d(ctx0, logits, logits->ne[0], nb, head_rows);
    logits = ggml_cont_2d(ctx0, ggml_permute(ctx0, logits, 0, 2, 1, 3), logits->ne[0], head_rows*nb);
    logits = ggml_view_2d(ctx0, logits, hparams.n_vocab_unpadded, head_rows*nb, logits->nb[1], 0);
    res->t_logits = n_outputs == head_rows*nb ? ggml_cont(ctx0, logits) : ggml_get_rows(ctx0, logits, build_inp_out_ids());
}

std::unique_ptr<llm_graph_context> llama_model_dflash::build_arch_graph(const llm_graph_params & params) const {
    switch (params.gtype) {
        case LLM_GRAPH_TYPE_ENCODER:
            return std::make_unique<graph<true>>(params);
        case LLM_GRAPH_TYPE_DEFAULT:
        case LLM_GRAPH_TYPE_DECODER:
            if (dspark_markov_rank) {
                return std::make_unique<graph_inkling2>(params);
            }
            switch (backbone) {
                case BACKBONE_BASE:           break;
                case BACKBONE_DSV4:           return std::make_unique<graph_dsv4>(params);
                case BACKBONE_DSV41:          return std::make_unique<graph_dsv41>(params);
                case BACKBONE_MISTRAL_LARGE4: return std::make_unique<graph_mistral_large4>(params);
            }
            switch (variant) {
                case VARIANT_DFLASH:   return std::make_unique<graph_dflash>(params);
                case VARIANT_DFLASH2:  return std::make_unique<graph_dflash2>(params);
                case VARIANT_DSPARK:   return std::make_unique<graph_dspark>(params);
                case VARIANT_LILICORR: return std::make_unique<graph_lilicorr>(params);
            }
            GGML_ABORT("invalid DFlash variant");
        default:
            GGML_ABORT("invalid graph type");
    };
}
