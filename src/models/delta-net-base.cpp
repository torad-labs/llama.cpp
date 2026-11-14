#include "models.h"

#include "llama-impl.h"
#include "llama-memory-recurrent.h"

llm_build_delta_net_base::llm_build_delta_net_base(const llm_graph_params & params) : llm_graph_context(params) {}

void llm_build_delta_net_base::gdn_set_raw_gates(ggml_tensor * gdn) const {
    if (gdn_raw_dt_bias) {
        ggml_gated_delta_net_set_raw_gates(gdn, gdn_raw_dt_bias, gdn_raw_a);
    } else {
        ggml_gated_delta_net_set_raw_kda_gates(gdn, gdn_raw_a, gdn_raw_kda_lb);
    }
}

std::pair<ggml_tensor *, ggml_tensor *> llm_build_delta_net_base::build_delta_net(
        ggml_tensor * q,
        ggml_tensor * k,
        ggml_tensor * v,
        ggml_tensor * g,
        ggml_tensor * b,
        ggml_tensor * s) {
    const int64_t S_k      = q->ne[0];
    const int64_t H_k      = q->ne[1];
    const int64_t n_tokens = q->ne[2];
    const int64_t n_seqs   = q->ne[3];

    const int64_t S_v = v->ne[0];
    const int64_t H_v = v->ne[1];

    GGML_ASSERT(S_k == S_v);
    GGML_ASSERT(H_v % H_k == 0);

    GGML_ASSERT(q->ne[0] == S_k && q->ne[1] == H_k && q->ne[2] == n_tokens && q->ne[3] == n_seqs);
    GGML_ASSERT(k->ne[0] == S_k && k->ne[1] == H_k && k->ne[2] == n_tokens && k->ne[3] == n_seqs);
    GGML_ASSERT(v->ne[0] == S_v && v->ne[1] == H_v && v->ne[2] == n_tokens && v->ne[3] == n_seqs);

    GGML_ASSERT(g->ne[0] == 1   || g->ne[0] == S_v);
    GGML_ASSERT(                   g->ne[1] == H_v && g->ne[2] == n_tokens && g->ne[3] == n_seqs);
    GGML_ASSERT(b->ne[0] == 1   && b->ne[1] == H_v && b->ne[2] == n_tokens && b->ne[3] == n_seqs);
    GGML_ASSERT(s->ne[0] == S_v && s->ne[1] == S_v && s->ne[2] == H_v      && s->ne[3] == n_seqs);

    // K=1: output carries the final state only. state s is 4D [S_v, S_v, H_v, n_seqs].
    const bool raw = gdn_raw();

    ggml_tensor * result = ggml_gated_delta_net(ctx0, q, k, v, raw ? gdn_raw_alpha : g, raw ? gdn_raw_beta : b, s, /*K=*/1);
    if (raw) {
        gdn_set_raw_gates(result);
    }
    ggml_tensor * output = ggml_view_4d(ctx0, result,
            S_v, H_v, n_tokens, n_seqs,
            ggml_row_size(result->type, S_v),
            ggml_row_size(result->type, S_v * H_v),
            ggml_row_size(result->type, S_v * H_v * n_tokens), 0);

    ggml_tensor * new_state = ggml_view_4d(ctx0, result,
            S_v, S_v, H_v, n_seqs,
            ggml_row_size(result->type, S_v),
            ggml_row_size(result->type, S_v * S_v),
            ggml_row_size(result->type, S_v * S_v * H_v),
            ggml_row_size(result->type, S_v * H_v * n_tokens * n_seqs));

    return {output, new_state};
}

ggml_tensor * llm_build_delta_net_base::build_conv_state(
        llm_graph_input_rs * inp,
        ggml_tensor *        conv_states_all,
        ggml_tensor *        qkv_mixed,
        int64_t              conv_kernel_size,
        int64_t              conv_channels,
        int                  il) {
    const auto * mctx_cur = inp->mctx;

    const auto kv_head  = mctx_cur->get_head();
    const auto mem_size = mctx_cur->get_size();

    const int64_t n_seqs = ubatch.n_seqs;

    // x first: the graph then reaches the state's gather after all that x depends on (the layer's input, and under
    // -sm tensor the previous layer's all-reduce), so the gather, the CONCAT and the SSM_CONV sit in one of the meta
    // backend's subgraphs, where the CUDA backend folds them into the conv (ggml_cuda_try_ssm_conv_state_update). A
    // gather reached first went before the previous layer's all-reduce, a subgraph apart: GLM-5.3 folded 1 of its 33.
    ggml_build_forward_expand(gf, qkv_mixed);

    ggml_tensor * conv_states = build_rs(inp, conv_states_all, hparams.n_embd_r(), n_seqs);
    cb(conv_states, "conv_states", il);

    conv_states = ggml_reshape_3d(ctx0, conv_states, conv_kernel_size - 1, conv_channels, n_seqs);
    cb(conv_states, "conv_states_reshaped", il);

    qkv_mixed = ggml_transpose(ctx0, qkv_mixed);
    cb(qkv_mixed, "qkv_mixed_transposed", il);

    ggml_tensor * conv_input = ggml_concat(ctx0, conv_states, qkv_mixed, 0);
    cb(conv_input, "conv_input", il);

    const int64_t row_count = (conv_kernel_size - 1) * conv_channels;

    const size_t row_size  = ggml_row_size(conv_states_all->type, row_count);

    if (inp->fixed_rows()) {
        const int64_t tokens = conv_input->ne[0] - conv_states->ne[0];
        const int64_t count = std::min<int64_t>(tokens, 1 + cparams.n_rs_seq);
        for (int64_t snapshot = 0; snapshot < count; ++snapshot) {
            ggml_tensor * window = ggml_view_3d(ctx0, conv_input, conv_kernel_size - 1, conv_channels, n_seqs,
                    conv_input->nb[1], conv_input->nb[2], ggml_row_size(conv_input->type, tokens - snapshot));
            ggml_build_forward_expand(gf, build_rs_store_local(inp, conv_states_all, window, snapshot));
        }
        return conv_input;
    }

    if (cparams.n_rs_seq == 0) {
        const int64_t s_idx  = conv_input->ne[0] - conv_states->ne[0];
        const int64_t s_slot = 0;

        ggml_tensor * conv_state_last =
            ggml_view_3d(ctx0, conv_input,
                    conv_kernel_size - 1, conv_channels, n_seqs,
                    conv_input->nb[1], conv_input->nb[2],
                    ggml_row_size(conv_input->type, s_idx));
        cb(conv_state_last, "conv_state_last", il);

        ggml_tensor * conv_state_update =
            ggml_view_2d(ctx0, conv_states_all,
                    row_count, n_seqs, conv_states_all->nb[1],
                    (s_slot * mem_size + kv_head) * row_size);
        cb(conv_state_update, "conv_state_update", il);

        ggml_build_forward_expand(gf, ggml_cpy(ctx0, conv_state_last, conv_state_update));
    } else {
        // [TAG_RECURRENT_ROLLBACK_SPLITS]
        // this logic assumes that the last (n_rs_seq + 1) tokens of a sequence in a batch are inside
        //   the same ubatch, which `split_equal()` guarantees via its n_keep_tail argument

        const int64_t K = (int64_t) cparams.n_rs_seq + 1;

        // slot s holds the window s tokens before the batch's end. A rollback of r rows reads slot r after a batch of
        // more than r (a verify rolls back its rejected draft rows, never the row the draft grew from), so, as
        // build_recurrent_attn does for the state, only the last min(n_seq_tokens, K) slots are written: a verify of 4
        // rows under --spec-rollback 15 copies 4 windows a layer, not 16
        const int64_t n_seq_tokens = conv_input->ne[0] - conv_states->ne[0];
        const int64_t n_written    = std::min<int64_t>(n_seq_tokens, K);

        for (int64_t s_slot = 0; s_slot < n_written; ++s_slot) {
            const int64_t s_idx = n_seq_tokens - s_slot;

            ggml_tensor * conv_state_last =
                ggml_view_3d(ctx0, conv_input,
                        conv_kernel_size - 1, conv_channels, n_seqs,
                        conv_input->nb[1], conv_input->nb[2],
                        ggml_row_size(conv_input->type, s_idx));

            ggml_tensor * conv_state_update =
                ggml_view_2d(ctx0,
                        conv_states_all, row_count, n_seqs,
                        conv_states_all->nb[1],
                        (s_slot * mem_size + kv_head) * row_size);

            ggml_build_forward_expand(gf, ggml_cpy(ctx0, conv_state_last, conv_state_update));
        }
    }

    return conv_input;
}

ggml_tensor * llm_build_delta_net_base::build_recurrent_attn(
        llm_graph_input_rs * inp,
        ggml_tensor *        ssm_states_all,
        ggml_tensor *        q,
        ggml_tensor *        k,
        ggml_tensor *        v,
        ggml_tensor *        g,
        ggml_tensor *        b,
        ggml_tensor *        s,
        int                  il,
        ggml_tensor *        state_rows) {
    const auto * mctx_cur   = inp->mctx;
    const auto   kv_head    = mctx_cur->get_head();
    const uint32_t mem_size = mctx_cur->get_size();

    // dims from v (always (S_v, H_v, T, B)): in rows mode `s` is the 2D cache
    // view, so its shape no longer carries them
    const int64_t S_v          = v->ne[0];
    const int64_t H_v          = v->ne[1];
    const int64_t n_seqs       = v->ne[3];
    const int64_t n_seq_tokens = q->ne[2];

    const bool keep = cparams.n_rs_seq > 0;

    GGML_ASSERT(state_rows == nullptr || keep); // rows mode is a ring-path optimization

    if (!keep) {
        auto attn_out = build_delta_net(q, k, v, g, b, s);
        ggml_tensor * output    = attn_out.first;
        ggml_tensor * new_state = attn_out.second;
        cb(output, "attn_output", il);
        cb(new_state, "new_state", il);

        if (inp->fixed_rows()) {
            ggml_build_forward_expand(gf, build_rs_store_local(inp, ssm_states_all, new_state, 0));
            return output;
        }

        ggml_build_forward_expand(gf,
                ggml_cpy(ctx0, new_state,
                    ggml_view_2d(ctx0, ssm_states_all, hparams.n_embd_s(), n_seqs, ssm_states_all->nb[1],
                        (size_t) kv_head * ssm_states_all->nb[1])));

        return output;
    }

    const int64_t D = S_v * S_v * H_v;
    const int64_t K = cparams.n_rs_seq + 1;

    const bool raw = gdn_raw();
    ggml_tensor * gg = raw ? gdn_raw_alpha : g;
    ggml_tensor * bb = raw ? gdn_raw_beta  : b;

    ggml_tensor * gdn_out;
    if (state_rows) {
        // rows mode: the fused op reads each seq's live state directly from the
        // 2D cache view at row state_rows[seq] -- no gather
        gdn_out = ggml_gated_delta_net_rows(ctx0, q, k, v, gg, bb, s, state_rows, (int) K);
    } else {
        // state s is 4D [S_v, S_v, H_v, n_seqs]; K snapshot slots are written into the output.
        gdn_out = ggml_gated_delta_net(ctx0, q, k, v, gg, bb, s, K);
    }
    if (raw) {
        gdn_set_raw_gates(gdn_out);
    }

    const int64_t attn_score_elems    = S_v * H_v * n_seq_tokens * n_seqs;
    const int64_t state_size_per_snap = S_v * S_v * H_v * n_seqs;

    ggml_tensor * output = ggml_view_4d(ctx0, gdn_out,
        S_v, H_v, n_seq_tokens, n_seqs,
        ggml_row_size(gdn_out->type, S_v),
        ggml_row_size(gdn_out->type, S_v * H_v),
        ggml_row_size(gdn_out->type, S_v * H_v * n_seq_tokens),
        0);
    cb(output, "attn_output", il);

    const size_t row_size = ggml_row_size(ssm_states_all->type, hparams.n_embd_s()); // a block type's row is not n_embd_s * element_size

    // op writes the last min(n_seq_tokens, K) snapshots; trailing slots are left unwritten
    const int64_t n_written = std::min<int64_t>(n_seq_tokens, K);

    if (inp->fixed_rows()) {
        for (int64_t snapshot = 0; snapshot < n_written; ++snapshot) {
            ggml_tensor * states = ggml_view_2d(ctx0, gdn_out, D, n_seqs,
                    ggml_row_size(gdn_out->type, D),
                    ggml_row_size(gdn_out->type, attn_score_elems + snapshot*state_size_per_snap));
            ggml_build_forward_expand(gf, build_rs_store_local(inp, ssm_states_all, states, snapshot));
        }
        return output;
    }

    if (state_rows) {
        // rows mode: scatter the compact snapshot region [D, n_written*n_seqs]
        // (slot-major, starting right after the attention scores) into the
        // cache rows slot*mem_size + kv_head + seq -- same destinations as the
        // strided cpy below, but expressed as SET_ROWS so the Metal backend
        // can fold the scatter into the fused op's epilogue.
        ggml_tensor * snaps = ggml_view_2d(ctx0, gdn_out,
            D, n_written * n_seqs,
            ggml_row_size(gdn_out->type, D),
            ggml_row_size(gdn_out->type, attn_score_elems));

        ggml_tensor * write_rows = build_rs_write_rows(inp, K, n_seq_tokens, n_seqs);

        ggml_build_forward_expand(gf, ggml_set_rows(ctx0, ssm_states_all, snaps, write_rows));

        return output;
    }

    // write the produced snapshots into the recurrent cache (snapshot slot i -> rollback group i)
    ggml_tensor * src = ggml_view_3d(ctx0, gdn_out,
        D, n_seqs, n_written,
        ggml_row_size(gdn_out->type, D),
        ggml_row_size(gdn_out->type, state_size_per_snap),
        ggml_row_size(gdn_out->type, attn_score_elems));

    ggml_tensor * dst = ggml_view_3d(ctx0, ssm_states_all,
        D, n_seqs, n_written,
        ssm_states_all->nb[1],
        (size_t) mem_size * row_size,
        (size_t) kv_head * row_size);

    ggml_build_forward_expand(gf, ggml_cpy(ctx0, src, dst));

    return output;
}
