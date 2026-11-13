#include "arg.h"
#include "common.h"
#include "llama.h"

#include <algorithm>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static llama_context * make_ctx(const common_params & params, llama_model * model) {
    auto cparams = common_context_params_to_llama(params);
    cparams.n_seq_max = 1;
    cparams.n_rs_seq  = 8;
    cparams.n_batch   = std::max(cparams.n_batch,  (uint32_t) (cparams.n_rs_seq + 1));
    cparams.n_ubatch  = std::max(cparams.n_ubatch, (uint32_t) (cparams.n_rs_seq + 1));
    return llama_init_from_model(model, cparams);
}

static bool decode_tokens(llama_context * ctx, const std::vector<llama_token> & tokens, uint32_t count) {
    llama_batch batch = llama_batch_init(count, 0, 1);
    for (uint32_t pos = 0; pos < count; ++pos) {
        common_batch_add(batch, tokens[pos], pos, { 0 }, pos + 1 == count);
    }
    const bool ok = llama_decode(ctx, batch) == 0;
    llama_batch_free(batch);
    return ok;
}

// tokens[p0, p1) at their positions in one batch, a verify's shape: a logits row for every token
static bool decode_span(llama_context * ctx, const std::vector<llama_token> & tokens, llama_pos p0, llama_pos p1) {
    llama_batch batch = llama_batch_init(p1 - p0, 0, 1);
    for (llama_pos pos = p0; pos < p1; ++pos) {
        common_batch_add(batch, tokens[pos], pos, { 0 }, true);
    }
    const bool ok = llama_decode(ctx, batch) == 0;
    llama_batch_free(batch);
    return ok;
}

static bool decode_one(llama_context * ctx, llama_token tok, llama_pos pos) {
    llama_batch batch = llama_batch_init(1, 0, 1);
    common_batch_add(batch, tok, pos, { 0 }, true);
    const bool ok = llama_decode(ctx, batch) == 0;
    llama_batch_free(batch);
    return ok;
}

// The recurrent state of a sequence as the context serializes it (the snapshot slot a pending rollback selects): its
// conv windows and its state, without the attention cache
static std::vector<float> recurrent_state(llama_context * ctx, llama_seq_id seq_id) {
    std::vector<uint8_t> buf(llama_state_seq_get_size_ext(ctx, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY));
    buf.resize(llama_state_seq_get_data_ext(ctx, buf.data(), buf.size(), seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY));
    std::vector<float> words(buf.size() / sizeof(float));
    std::memcpy(words.data(), buf.data(), words.size() * sizeof(float));
    return words;
}

// How far two contexts' recurrent states of a sequence are apart. The serialized state holds positions and counts
// beside the f32 rows: a word is equal when its bits are, and otherwise as far apart as the two floats (a NaN word, or
// states of different sizes, are infinitely far).
//
// A small model's recurrent branch can sit below the rounding of its residual stream, so its logits stay bitwise equal
// with a wrong conv window or state: a rollback is checked on the state itself, against a context that never advanced.
static float recurrent_state_diff(llama_context * ctx_a, llama_context * ctx_b, llama_seq_id seq_id) {
    const std::vector<float> a = recurrent_state(ctx_a, seq_id);
    const std::vector<float> b = recurrent_state(ctx_b, seq_id);
    float diff = a.size() == b.size() && !b.empty() ? 0.0f : INFINITY;
    for (size_t w = 0; w < b.size() && std::isfinite(diff); ++w) {
        if (std::memcmp(&a[w], &b[w], sizeof(float)) != 0) {
            const float d = std::fabs(a[w] - b[w]);
            diff = std::isnan(d) ? INFINITY : std::max(diff, d);
        }
    }
    return diff;
}

// states reached through batches of different shapes agree to rounding, not bitwise
constexpr float state_eps = 1e-6f;

// Roll back multiple sequences, then replay them in a single batch whose
// per-seq token count exceeds n_ubatch: each seq's replay spans several
// ubatches while its rollback restore is still pending. Compared against a
// reference context that never advanced past the rollback point and decodes
// the identical replay batch. A rollback reaches only into the last batch and
// not to its first row, so each seq's tail batch is one row longer than the
// rollback and the whole tail is refused.
static bool test_multi_seq_split_replay(const common_params & params, llama_model * model, const int n_vocab) {
    constexpr uint32_t  n_seqs     = 2;
    constexpr uint32_t  n_ubatch   = 16;
    constexpr uint32_t  n_prompt   = 19;
    constexpr uint32_t  n_rollback = 3;
    constexpr uint32_t  n_replay   = 40; // > n_ubatch so each seq spans multiple ubatches
    constexpr llama_pos p0         = n_prompt - n_rollback;

    const auto make_ctx_multi = [&]() {
        auto cparams = common_context_params_to_llama(params);
        cparams.n_seq_max  = n_seqs;
        cparams.n_rs_seq   = 8;
        cparams.n_ctx      = 256;
        cparams.n_batch    = 256;
        cparams.n_ubatch   = n_ubatch;
        cparams.kv_unified = false;
        return llama_init_from_model(model, cparams);
    };

    llama_context * ctx_roll = make_ctx_multi();
    llama_context * ctx_ref  = make_ctx_multi();
    if (ctx_roll == nullptr || ctx_ref == nullptr) {
        fprintf(stderr, "%s : failed to init multi-seq contexts\n", __func__);
        return false;
    }

    const auto cleanup = [&]() {
        llama_free(ctx_roll);
        llama_free(ctx_ref);
    };

    if (llama_n_rs_seq(ctx_roll) < n_rollback) {
        fprintf(stderr, "%s : skipping because n_rs_seq is too small\n", __func__);
        cleanup();
        return true;
    }

    const auto tok = [&](uint32_t seq, llama_pos pos) {
        return (llama_token) ((7*(uint32_t) pos + 31*seq + 1) % (uint32_t) n_vocab);
    };

    bool ok = true;

    // both contexts decode the identical [0, p0 - 1) prefill; ctx_roll decodes
    // the tail [p0 - 1, n_prompt), which is then rolled back to p0 so its restore
    // is pending at replay, and ctx_ref only the row the rollback keeps
    for (uint32_t s = 0; s < n_seqs && ok; ++s) {
        llama_batch batch = llama_batch_init(n_prompt, 0, 1);
        for (llama_pos pos = 0; pos < (llama_pos) p0 - 1; ++pos) {
            common_batch_add(batch, tok(s, pos), pos, { (llama_seq_id) s }, false);
        }
        ok = ok && llama_decode(ctx_roll, batch) == 0;
        ok = ok && llama_decode(ctx_ref,  batch) == 0;

        common_batch_clear(batch);
        common_batch_add(batch, tok(s, p0 - 1), p0 - 1, { (llama_seq_id) s }, false);
        ok = ok && llama_decode(ctx_ref, batch) == 0;

        common_batch_clear(batch);
        for (llama_pos pos = p0 - 1; pos < (llama_pos) n_prompt; ++pos) {
            common_batch_add(batch, tok(s, pos), pos, { (llama_seq_id) s }, false);
        }
        ok = ok && llama_decode(ctx_roll, batch) == 0;
        llama_batch_free(batch);

        // the whole tail batch reaches back to the state before it, which no snapshot keeps
        ok = ok && !llama_memory_seq_rm(llama_get_memory(ctx_roll), (llama_seq_id) s, p0 - 1, -1);

        ok = ok && llama_memory_seq_rm(llama_get_memory(ctx_roll), (llama_seq_id) s, p0, -1);

        // a second partial removal while one is pending must be refused
        ok = ok && !llama_memory_seq_rm(llama_get_memory(ctx_roll), (llama_seq_id) s, p0 - 1, -1);
    }
    if (!ok) {
        fprintf(stderr, "%s : multi-seq prefill/rollback failed\n", __func__);
        cleanup();
        return false;
    }

    // each seq rolled back all but the first row of its tail batch: its state must be the one ctx_ref holds at p0
    for (uint32_t s = 0; s < n_seqs; ++s) {
        const float diff = recurrent_state_diff(ctx_roll, ctx_ref, (llama_seq_id) s);
        fprintf(stderr, "%s : seq %u rolled back %u rows, state max diff %g\n", __func__, s, n_rollback, (double) diff);
        if (!(diff <= state_eps)) {
            fprintf(stderr, "%s : seq %u state after the rollback differs from the reference\n", __func__, s);
            cleanup();
            return false;
        }
    }

    llama_batch batch = llama_batch_init(n_seqs*n_replay, 0, 1);
    for (uint32_t s = 0; s < n_seqs; ++s) {
        for (uint32_t i = 0; i < n_replay; ++i) {
            const llama_pos pos = p0 + (llama_pos) i;
            common_batch_add(batch, tok(s, pos), pos, { (llama_seq_id) s }, true);
        }
    }
    ok = llama_decode(ctx_roll, batch) == 0;
    ok = ok && llama_decode(ctx_ref, batch) == 0;
    llama_batch_free(batch);
    if (!ok) {
        fprintf(stderr, "%s : multi-seq replay decode failed\n", __func__);
        cleanup();
        return false;
    }

    // identical ubatch shapes from bit-exact states: a correct implementation
    // matches bitwise, so eps only allows backend scheduling noise
    constexpr float eps = 1e-7f;

    float    diff_max  = 0.0f;
    uint32_t seq_first = 0;
    int32_t  pos_first = -1;
    for (uint32_t i = 0; i < n_seqs*n_replay; ++i) {
        const float * l_roll = llama_get_logits_ith(ctx_roll, i);
        const float * l_ref  = llama_get_logits_ith(ctx_ref,  i);
        if (l_roll == nullptr || l_ref == nullptr) {
            fprintf(stderr, "%s : missing multi-seq logits at index %u\n", __func__, i);
            cleanup();
            return false;
        }
        for (int t = 0; t < n_vocab; ++t) {
            const float diff = std::fabs(l_roll[t] - l_ref[t]);
            if (diff > eps && pos_first < 0) {
                seq_first = i/n_replay;
                pos_first = p0 + (int32_t) (i%n_replay);
            }
            diff_max = std::max(diff_max, diff);
        }
    }

    if (diff_max > eps) {
        fprintf(stderr, "%s : multi-seq split replay logits mismatch (max diff %g, first at seq %u pos %d)\n",
                __func__, (double) diff_max, seq_first, pos_first);
        cleanup();
        return false;
    }

    fprintf(stderr, "%s : multi-seq split replay matched (max diff %g)\n", __func__, (double) diff_max);

    for (uint32_t s = 0; s < n_seqs; ++s) {
        const float diff = recurrent_state_diff(ctx_roll, ctx_ref, (llama_seq_id) s);
        if (!(diff <= state_eps)) {
            fprintf(stderr, "%s : seq %u state after the split replay differs from the reference (max diff %g)\n",
                    __func__, s, (double) diff);
            cleanup();
            return false;
        }
    }

    // seq-1-only decodes must be independent of seq 0's content: diverge seq 0
    // in ctx_ref only, then compare identical seq-1-only continuations bitwise
    constexpr uint32_t n_tail = 4;

    {
        llama_batch batch_tail = llama_batch_init(n_tail, 0, 1);
        for (uint32_t i = 0; i < n_tail; ++i) {
            const llama_pos pos = p0 + (llama_pos) (n_replay + i);
            common_batch_add(batch_tail, tok(0, pos + 7), pos, { 0 }, false);
        }
        ok = llama_decode(ctx_ref, batch_tail) == 0;
        llama_batch_free(batch_tail);
    }

    float diff_tail = 0.0f;
    for (uint32_t i = 0; i < n_tail && ok; ++i) {
        const llama_pos pos = p0 + (llama_pos) (n_replay + i);
        llama_batch batch_one = llama_batch_init(1, 0, 1);
        common_batch_add(batch_one, tok(1, pos), pos, { 1 }, true);
        ok = llama_decode(ctx_roll, batch_one) == 0;
        ok = ok && llama_decode(ctx_ref, batch_one) == 0;
        llama_batch_free(batch_one);
        if (!ok) {
            break;
        }

        const float * l_roll = llama_get_logits_ith(ctx_roll, 0);
        const float * l_ref  = llama_get_logits_ith(ctx_ref,  0);
        ok = l_roll != nullptr && l_ref != nullptr;
        for (int t = 0; ok && t < n_vocab; ++t) {
            diff_tail = std::max(diff_tail, std::fabs(l_roll[t] - l_ref[t]));
        }
    }

    if (!ok || diff_tail > eps) {
        fprintf(stderr, "%s : seq-1-only decode leaked seq 0 state (ok=%d, max diff %g)\n",
                __func__, ok ? 1 : 0, (double) diff_tail);
        cleanup();
        return false;
    }

    fprintf(stderr, "%s : seq-1-only decode independent of seq 0 (max diff %g)\n", __func__, (double) diff_tail);
    cleanup();
    return true;
}

// The speculative shape: a verify batch of fewer rows than n_rs_seq + 1, rolled back by r of them (r < the batch, as a
// verify rolls back its rejected draft rows). Each depth is compared with a reference context that decoded only the
// accepted rows: first the recurrent state itself (a small model's recurrent branch can sit below the rounding of its
// residual stream, so its logits alone would not see a wrong window), then the logits over a correction token and the
// tokens after it. The snapshots a short batch writes must hold every slot such a rollback reads.
static bool test_short_verify_rollback(const common_params & params, llama_model * model, const int n_vocab) {
    constexpr uint32_t n_prompt = 19;
    constexpr uint32_t n_verify = 4;
    constexpr uint32_t n_after  = 3;

    const auto tok = [&](llama_pos pos) {
        return (llama_token) ((7*(uint32_t) pos + 1) % (uint32_t) n_vocab);
    };
    const auto decode_range = [&](llama_context * ctx, llama_pos p0, llama_pos p1) {
        llama_batch batch = llama_batch_init(p1 - p0, 0, 1);
        for (llama_pos pos = p0; pos < p1; ++pos) {
            common_batch_add(batch, tok(pos), pos, { 0 }, true);
        }
        const bool ok = llama_decode(ctx, batch) == 0;
        llama_batch_free(batch);
        return ok;
    };

    for (uint32_t r = 1; r < n_verify; ++r) {
        llama_context * ctx_roll = make_ctx(params, model);
        llama_context * ctx_ref  = make_ctx(params, model);
        const auto cleanup = [&]() {
            llama_free(ctx_roll);
            llama_free(ctx_ref);
        };
        if (ctx_roll == nullptr || ctx_ref == nullptr) {
            fprintf(stderr, "%s : failed to init contexts\n", __func__);
            cleanup();
            return false;
        }
        if (llama_n_rs_seq(ctx_roll) + 1 <= n_verify) {
            fprintf(stderr, "%s : skipping because n_rs_seq + 1 does not exceed the verify batch\n", __func__);
            cleanup();
            return true;
        }

        const llama_pos p_keep = n_prompt + n_verify - r;

        bool ok = decode_range(ctx_roll, 0, n_prompt) && decode_range(ctx_ref, 0, n_prompt);
        ok = ok && decode_range(ctx_roll, n_prompt, n_prompt + n_verify);
        ok = ok && llama_memory_seq_rm(llama_get_memory(ctx_roll), 0, p_keep, -1);
        ok = ok && decode_range(ctx_ref, n_prompt, p_keep);
        if (!ok) {
            fprintf(stderr, "%s : decode or rollback of %u rows failed\n", __func__, r);
            cleanup();
            return false;
        }

        const float state_diff = recurrent_state_diff(ctx_roll, ctx_ref, 0);
        fprintf(stderr, "%s : %u of %u rows rolled back, state max diff %g\n", __func__, r, n_verify, (double) state_diff);
        if (!(state_diff <= state_eps)) {
            fprintf(stderr, "%s : the recurrent state after rolling back %u rows differs from the reference\n", __func__, r);
            cleanup();
            return false;
        }

        float diff_max = 0.0f;
        for (uint32_t i = 0; i < n_after; ++i) {
            // a correction first: the token the rolled-back row did not hold
            const llama_pos   pos = p_keep + (llama_pos) i;
            const llama_token t   = i == 0 ? (llama_token) ((tok(pos) + 3) % n_vocab) : tok(pos);
            if (!decode_one(ctx_roll, t, pos) || !decode_one(ctx_ref, t, pos)) {
                fprintf(stderr, "%s : replay after rolling back %u rows failed at position %d\n", __func__, r, pos);
                cleanup();
                return false;
            }
            const float * l_roll = llama_get_logits_ith(ctx_roll, 0);
            const float * l_ref  = llama_get_logits_ith(ctx_ref,  0);
            for (int token = 0; token < n_vocab; ++token) {
                diff_max = std::max(diff_max, std::fabs(l_roll[token] - l_ref[token]));
            }
        }
        cleanup();

        // the reference decodes the accepted rows as a batch of their own, so the states agree to rounding, not bitwise
        constexpr float eps = 1e-4f;
        fprintf(stderr, "%s : %u of %u rows rolled back, logits max diff %g\n", __func__, r, n_verify, (double) diff_max);
        if (diff_max > eps) {
            fprintf(stderr, "%s : logits after rolling back %u rows differ from the reference\n", __func__, r);
            return false;
        }
    }

    return true;
}

// LLAMA_GRAPH_ANY_SEQS=1: a decode's graph is every set's of as many sequences, and a state the set displaces from its
// cells is moved before the decode. Two contexts, one with the switch, decode the same batches of sets that change: a
// sequence leaves and returns, another set of as many takes over, one starts beside running ones, one is rolled back
// while another has its cell, one is displaced by a later ubatch of the decode it is then rolled back in, and one is
// cleared and starts again. Every row's logits and every sequence's state must be the ones of the context without the
// switch, and the switch must have reused graphs the other context built.
static bool test_changed_sets(const common_params & params, llama_model * model, const int n_vocab) {
#ifdef _WIN32
    GGML_UNUSED(params);
    GGML_UNUSED(model);
    GGML_UNUSED(n_vocab);
    fprintf(stderr, "%s : skipping on windows, which has no setenv\n", __func__);
    return true;
#else
    constexpr uint32_t n_seqs = 4;
    constexpr float    eps    = 1e-7f;

    const auto make_ctx_sets = [&](bool any_seqs) {
        auto cparams = common_context_params_to_llama(params);
        cparams.n_seq_max  = n_seqs;
        cparams.n_rs_seq   = 2;
        cparams.n_ctx      = 256;
        cparams.n_batch    = 256;
        cparams.n_ubatch   = 64;
        cparams.kv_unified = true;
        // the switch is read when a context and its memory are made
        if (any_seqs) {
            setenv("LLAMA_GRAPH_ANY_SEQS", "1", 1);
        } else {
            unsetenv("LLAMA_GRAPH_ANY_SEQS");
        }
        llama_context * ctx = llama_init_from_model(model, cparams);
        unsetenv("LLAMA_GRAPH_ANY_SEQS");
        return ctx;
    };

    llama_context * ctx_ref = make_ctx_sets(false);
    llama_context * ctx_any = make_ctx_sets(true);
    if (ctx_ref == nullptr || ctx_any == nullptr) {
        fprintf(stderr, "%s : failed to init contexts\n", __func__);
        return false;
    }

    const auto cleanup = [&]() {
        llama_free(ctx_ref);
        llama_free(ctx_any);
    };

    if (llama_n_rs_seq(ctx_any) < 2) {
        fprintf(stderr, "%s : skipping because n_rs_seq is too small\n", __func__);
        cleanup();
        return true;
    }

    const auto tok = [&](uint32_t seq, llama_pos pos) {
        return (llama_token) ((7*(uint32_t) pos + 31*seq + 1) % (uint32_t) n_vocab);
    };

    std::vector<llama_pos> pos(n_seqs, 0);

    // one decode in both contexts: counts[s] tokens of each sequence from where it stands, a logits row for each
    const auto step = [&](const char * what, const std::vector<uint32_t> & counts) {
        uint32_t n_rows = 0;
        for (const uint32_t count : counts) {
            n_rows += count;
        }
        llama_batch batch = llama_batch_init(n_rows, 0, 1);
        for (uint32_t s = 0; s < n_seqs; ++s) {
            for (uint32_t i = 0; i < counts[s]; ++i) {
                common_batch_add(batch, tok(s, pos[s] + (llama_pos) i), pos[s] + (llama_pos) i, { (llama_seq_id) s }, true);
            }
            pos[s] += (llama_pos) counts[s];
        }
        const bool decoded = llama_decode(ctx_ref, batch) == 0 && llama_decode(ctx_any, batch) == 0;
        llama_batch_free(batch);
        if (!decoded) {
            fprintf(stderr, "test_changed_sets : %s: decode failed\n", what);
            return false;
        }
        for (uint32_t i = 0; i < n_rows; ++i) {
            const float * l_ref = llama_get_logits_ith(ctx_ref, i);
            const float * l_any = llama_get_logits_ith(ctx_any, i);
            if (l_ref == nullptr || l_any == nullptr) {
                fprintf(stderr, "test_changed_sets : %s: missing logits at row %u\n", what, i);
                return false;
            }
            for (int t = 0; t < n_vocab; ++t) {
                if (!(std::fabs(l_ref[t] - l_any[t]) <= eps)) {
                    fprintf(stderr, "test_changed_sets : %s: logits differ at row %u, token %d (%g != %g)\n",
                            what, i, t, (double) l_ref[t], (double) l_any[t]);
                    return false;
                }
            }
        }
        for (uint32_t s = 0; s < n_seqs; ++s) {
            if (pos[s] == 0) {
                continue;
            }
            const float diff = recurrent_state_diff(ctx_any, ctx_ref, (llama_seq_id) s);
            if (!(diff <= state_eps)) {
                fprintf(stderr, "test_changed_sets : %s: seq %u state differs from the reference by %g\n",
                        what, s, (double) diff);
                return false;
            }
        }
        return true;
    };

    const auto rollback = [&](uint32_t seq, uint32_t n_rows) {
        pos[seq] -= (llama_pos) n_rows;
        return llama_memory_seq_rm(llama_get_memory(ctx_ref), (llama_seq_id) seq, pos[seq], -1) &&
               llama_memory_seq_rm(llama_get_memory(ctx_any), (llama_seq_id) seq, pos[seq], -1);
    };

    const auto clear = [&](uint32_t seq) {
        pos[seq] = 0;
        return llama_memory_seq_rm(llama_get_memory(ctx_ref), (llama_seq_id) seq, -1, -1) &&
               llama_memory_seq_rm(llama_get_memory(ctx_any), (llama_seq_id) seq, -1, -1);
    };

    const auto restore = [&](uint32_t seq, llama_state_seq_flags flags) {
        for (llama_context * ctx : {ctx_ref, ctx_any}) {
            std::vector<std::vector<float>> before;
            for (uint32_t s = 0; s < n_seqs; ++s) {
                before.push_back(recurrent_state(ctx, (llama_seq_id) s));
            }
            std::vector<uint8_t> saved(llama_state_seq_get_size_ext(ctx, (llama_seq_id) seq, flags));
            if (saved.empty() ||
                llama_state_seq_get_data_ext(ctx, saved.data(), saved.size(), (llama_seq_id) seq, flags) != saved.size() ||
                llama_state_seq_set_data_ext(ctx, saved.data(), saved.size(), (llama_seq_id) seq, flags) != saved.size()) {
                fprintf(stderr, "%s : seq %u state round trip failed\n", __func__, seq);
                return false;
            }
            for (uint32_t s = 0; s < n_seqs; ++s) {
                const auto after = recurrent_state(ctx, (llama_seq_id) s);
                if (after.size() != before[s].size() ||
                    std::memcmp(after.data(), before[s].data(), after.size()*sizeof(float)) != 0) {
                    fprintf(stderr, "%s : restoring seq %u changed seq %u state (%s, flags %u)\n",
                            __func__, seq, s, ctx == ctx_any ? "any sequences" : "reference", (unsigned) flags);
                    return false;
                }
            }
        }
        return true;
    };

    const auto copy = [&](uint32_t src, uint32_t dst) {
        std::vector<uint8_t> saved(llama_state_seq_get_size(ctx_any, (llama_seq_id) src));
        if (saved.empty() ||
            llama_state_seq_get_data(ctx_any, saved.data(), saved.size(), (llama_seq_id) src) != saved.size() ||
            llama_state_seq_set_data(ctx_ref, saved.data(), saved.size(), (llama_seq_id) dst) != saved.size()) {
            fprintf(stderr, "%s : failed to restore the copy reference\n", __func__);
            return false;
        }
        llama_memory_seq_cp(llama_get_memory(ctx_any), (llama_seq_id) src, (llama_seq_id) dst, -1, -1);
        pos[dst] = pos[src];
        // Some memories defer copies until the next decode; step checks the resulting logits and states.
        return true;
    };

    const auto rollback_shared = [&](uint32_t src, uint32_t dst, uint32_t n_rows, bool shared_ref) {
        if (!clear(dst)) {
            return false;
        }
        llama_context * shared = shared_ref ? ctx_ref : ctx_any;
        llama_context * independent = shared_ref ? ctx_any : ctx_ref;
        constexpr llama_state_seq_flags flags = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;
        std::vector<uint8_t> saved(llama_state_seq_get_size_ext(shared, (llama_seq_id) src, flags));
        if (saved.empty() ||
            llama_state_seq_get_data_ext(shared, saved.data(), saved.size(), (llama_seq_id) src, flags) != saved.size()) {
            fprintf(stderr, "%s : failed to save the sibling reference\n", __func__);
            return false;
        }
        // Keep identical attention placement; only the reference's recurrent cell is independent.
        for (llama_context * ctx : {ctx_ref, ctx_any}) {
            llama_memory_seq_cp(llama_get_memory(ctx), (llama_seq_id) src, (llama_seq_id) dst, -1, -1);
        }
        pos[dst] = pos[src];
        // Finish deferred copies before constructing the reference; neither sibling decodes.
        std::vector<uint32_t> advance(n_seqs, 0);
        uint32_t other = 0;
        while (other == src || other == dst) { ++other; }
        advance[other] = 1;
        if (!step("finish copied streams while the siblings wait", advance)) {
            return false;
        }
        if (llama_state_seq_set_data_ext(independent, saved.data(), saved.size(), (llama_seq_id) dst, flags) != saved.size()) {
            fprintf(stderr, "%s : failed to restore the independent sibling\n", __func__);
            return false;
        }
        std::vector<std::vector<float>> before;
        for (llama_context * ctx : {ctx_ref, ctx_any}) {
            before.push_back(recurrent_state(ctx, (llama_seq_id) dst));
        }
        if (!rollback(src, n_rows)) {
            fprintf(stderr, "%s : rollback of a copied source was refused\n", __func__);
            return false;
        }
        size_t i = 0;
        for (llama_context * ctx : {ctx_ref, ctx_any}) {
            const llama_pos actual_pos = llama_memory_seq_pos_max(llama_get_memory(ctx), (llama_seq_id) dst);
            const auto after = recurrent_state(ctx, (llama_seq_id) dst);
            if (actual_pos != pos[dst] - 1 || after.size() != before[i].size() ||
                std::memcmp(after.data(), before[i].data(), after.size()*sizeof(float)) != 0) {
                fprintf(stderr, "%s : rolling back seq %u changed copied sibling %u (%s, position %d != %d)\n",
                        __func__, src, dst, ctx == ctx_any ? "any sequences" : "reference",
                        actual_pos, pos[dst] - 1);
                return false;
            }
            ++i;
        }
        const float diff = recurrent_state_diff(shared, independent, (llama_seq_id) src);
        if (!(diff <= state_eps)) {
            fprintf(stderr, "%s : a copied source's rollback differs from an independent source by %g\n",
                    __func__, (double) diff);
            return false;
        }
        return true;
    };

    bool ok = true;
    ok = ok && step("three sequences' prompts",                       {6, 6, 6, 0});
    ok = ok && step("the same set",                                   {1, 1, 1, 0});
    ok = ok && step("the same set again",                             {1, 1, 1, 0});
    ok = ok && step("a middle sequence leaves",                       {1, 0, 1, 0});
    ok = ok && step("the smaller set again",                          {1, 0, 1, 0});
    ok = ok && step("another set of two",                             {0, 1, 1, 0});
    ok = ok && step("a third set of two",                             {1, 1, 0, 0});
    ok = ok && step("one alone",                                      {0, 0, 1, 0});
    ok = ok && step("another alone",                                  {0, 1, 0, 0});
    ok = ok && step("a sequence starts beside three",                 {3, 3, 3, 3});
    ok = ok && rollback(2, 1);
    ok = ok && step("its cell is taken while it is rolled back",      {1, 1, 0, 1});
    ok = ok && step("the rolled back one replays",                    {0, 0, 2, 0});
    ok = ok && step("all four",                                       {1, 1, 1, 1});
    ok = ok && step("two ubatches, the later displacing the earlier", {3, 1, 3, 1});
    ok = ok && rollback(0, 2);
    ok = ok && rollback(2, 1);
    ok = ok && step("the displaced ones replay",                      {2, 0, 1, 0});
    ok = ok && step("another pair",                                   {0, 1, 0, 1});
    ok = ok && clear(1);
    ok = ok && step("a cleared sequence starts again beside another", {0, 3, 0, 3});
    ok = ok && step("all four again",                                 {1, 1, 1, 1});
    // A restore must preserve other sequences' current states and rollback snapshots.
    for (const llama_state_seq_flags flags : {llama_state_seq_flags(0), llama_state_seq_flags(LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY)}) {
        for (int cycle = 0; ok && cycle < 3; ++cycle) {
            ok = ok && step("states before cache exchange", {3, 3, 3, 3});
            ok = ok && rollback(0, 1);
            ok = ok && rollback(1, 1);
            ok = ok && restore(1, flags);
            ok = ok && restore(2, flags);
            ok = ok && rollback(3, 1);
            ok = ok && step("continue after cache exchange", {2, 2, 1, 2});
        }
    }
    if (!ok) {
        fprintf(stderr, "%s : the contexts differ, or a rollback or removal was refused\n", __func__);
        cleanup();
        return false;
    }

    ok = ok && step("states before copying a rolled back sequence", {3, 3, 3, 3});
    ok = ok && rollback(0, 1);
    ok = ok && clear(1);
    ok = ok && copy(0, 1);
    ok = ok && step("continue a copied rollback", {1, 1, 1, 1});
    ok = ok && step("states before reusing a rolled back destination", {3, 3, 3, 3});
    ok = ok && rollback(1, 2);
    ok = ok && clear(1);
    ok = ok && copy(2, 1);
    ok = ok && step("continue after reusing a rolled back destination", {1, 1, 1, 1});
    // Earlier full restores can place attention rows differently. Start this matrix from equal caches.
    for (llama_context * ctx : {ctx_ref, ctx_any}) {
        llama_memory_clear(llama_get_memory(ctx), true);
    }
    std::fill(pos.begin(), pos.end(), 0);
    ok = ok && step("fresh prefixes for the copied-sibling matrix", {6, 6, 6, 6});
    for (const bool shared_ref : {true, false}) {
        for (const uint32_t n_rows : {1u, 2u}) {
            ok = ok && step("states before copying then rolling back the source", {3, 3, 3, 3});
            ok = ok && rollback_shared(0, 1, n_rows, shared_ref);
            ok = ok && step("other sequences advance while both siblings wait", {0, 0, 1, 1});
            ok = ok && step("the copied sibling continues first", {0, 1, 0, 0});
            ok = ok && step("the rolled back source follows", {1, 1, 1, 1});
        }
    }
    if (!ok) {
        cleanup();
        return false;
    }

    const int32_t n_reused_ref = llama_perf_context(ctx_ref).n_reused;
    const int32_t n_reused_any = llama_perf_context(ctx_any).n_reused;
    fprintf(stderr, "%s : graphs reused: %d without the switch, %d with it\n", __func__, n_reused_ref, n_reused_any);
    cleanup();

    // the switch is for the hybrid memory, which these architectures have. Another memory (DeepSeek V4's cache keeps a
    // sequence's rows by its place) is left as it was: there the two contexts only have to agree
    char arch[64] = {};
    llama_model_meta_val_str(model, "general.architecture", arch, sizeof(arch));
    bool hybrid = false;
    for (const char * name : { "glm5next", "qwen35", "qwen35moe", "nemotron_h", "nemotron_h_moe" }) {
        hybrid = hybrid || std::strcmp(arch, name) == 0;
    }
    if (hybrid && n_reused_any <= n_reused_ref) {
        fprintf(stderr, "%s : the switch reused no graph for another set of sequences\n", __func__);
        return false;
    }
    return true;
#endif
}

// LLAMA_PROMPT_SHAPES=1 changes which rows a decode's ubatches hold (a prompt's rows apart from the replies', in a few
// sizes), never what a sequence reads: its logits and its states are those of a context without the switch, with the
// switch alone and beside LLAMA_GRAPH_ANY_SEQS=1, and a rollback after a prompt read that way is as valid. The same
// rows in other ubatches are other sums, so the bounds are not the 1e-7 and 1e-6 of a context that cuts its batches
// the same way: over 136 generated models the logits differed by up to 1.2e-7 and a state, carried over 131 rows cut
// elsewhere, by up to 1.2e-6 (one model; 6.2e-7 on the others). The bounds are 100 times that.
static bool test_prompt_shapes(const common_params & params, llama_model * model, const int n_vocab) {
#ifdef _WIN32
    GGML_UNUSED(params);
    GGML_UNUSED(model);
    GGML_UNUSED(n_vocab);
    fprintf(stderr, "%s : skipping on windows, which has no setenv\n", __func__);
    return true;
#else
    constexpr uint32_t n_seqs = 4;
    constexpr float    eps          = 1e-4f;
    constexpr float    eps_state    = 1e-4f;
    float              max_logits   = 0.0f;
    float              max_state    = 0.0f;

    const auto make_ctx_shapes = [&](bool shapes, bool any_seqs) {
        auto cparams = common_context_params_to_llama(params);
        cparams.n_seq_max  = n_seqs;
        cparams.n_rs_seq   = 2;
        cparams.n_ctx      = 1024;
        cparams.n_batch    = 512;
        cparams.n_ubatch   = 64;
        cparams.kv_unified = true;
        // the switches are read when a context, its memory and its batch allocator are made
        if (shapes) {
            setenv("LLAMA_PROMPT_SHAPES", "1", 1);
        }
        if (any_seqs) {
            setenv("LLAMA_GRAPH_ANY_SEQS", "1", 1);
        }
        llama_context * ctx = llama_init_from_model(model, cparams);
        unsetenv("LLAMA_PROMPT_SHAPES");
        unsetenv("LLAMA_GRAPH_ANY_SEQS");
        return ctx;
    };

    llama_context * ctx_ref = make_ctx_shapes(false, false);
    std::vector<llama_context *> ctxs = { make_ctx_shapes(true, false), make_ctx_shapes(true, true) };
    const auto cleanup = [&]() {
        llama_free(ctx_ref);
        for (llama_context * ctx : ctxs) {
            llama_free(ctx);
        }
    };
    if (ctx_ref == nullptr || ctxs[0] == nullptr || ctxs[1] == nullptr) {
        fprintf(stderr, "%s : failed to init contexts\n", __func__);
        cleanup();
        return false;
    }
    if (llama_n_rs_seq(ctx_ref) < 2) {
        fprintf(stderr, "%s : skipping because n_rs_seq is too small\n", __func__);
        cleanup();
        return true;
    }

    const auto tok = [&](uint32_t seq, llama_pos pos) {
        return (llama_token) ((11*(uint32_t) pos + 37*seq + 5) % (uint32_t) n_vocab);
    };

    std::vector<llama_pos> pos(n_seqs, 0);

    // one decode in every context: counts[s] rows of each sequence from where it stands. A sequence of reply[s] has a
    // logits row for each (a reply and its drafts), another only for its last row (a prompt)
    const auto step = [&](const char * what, const std::vector<uint32_t> & counts, const std::vector<bool> & reply) {
        uint32_t n_rows = 0;
        for (const uint32_t count : counts) {
            n_rows += count;
        }
        llama_batch batch = llama_batch_init(n_rows, 0, 1);
        std::vector<int32_t> out_rows;
        for (uint32_t s = 0; s < n_seqs; ++s) {
            for (uint32_t i = 0; i < counts[s]; ++i) {
                const bool output = reply[s] || i + 1 == counts[s];
                if (output) {
                    out_rows.push_back(batch.n_tokens);
                }
                common_batch_add(batch, tok(s, pos[s] + (llama_pos) i), pos[s] + (llama_pos) i, { (llama_seq_id) s }, output);
            }
            pos[s] += (llama_pos) counts[s];
        }
        bool decoded = llama_decode(ctx_ref, batch) == 0;
        for (llama_context * ctx : ctxs) {
            decoded = decoded && llama_decode(ctx, batch) == 0;
        }
        llama_batch_free(batch);
        if (!decoded) {
            fprintf(stderr, "test_prompt_shapes : %s: decode failed\n", what);
            return false;
        }
        for (size_t c = 0; c < ctxs.size(); ++c) {
            for (const int32_t i : out_rows) {
                const float * l_ref = llama_get_logits_ith(ctx_ref, i);
                const float * l_ctx = llama_get_logits_ith(ctxs[c], i);
                if (l_ref == nullptr || l_ctx == nullptr) {
                    fprintf(stderr, "test_prompt_shapes : %s: context %zu: missing logits at row %d\n", what, c, i);
                    return false;
                }
                for (int t = 0; t < n_vocab; ++t) {
                    max_logits = std::max(max_logits, std::fabs(l_ref[t] - l_ctx[t]));
                    if (!(std::fabs(l_ref[t] - l_ctx[t]) <= eps)) {
                        fprintf(stderr, "test_prompt_shapes : %s: context %zu: logits differ at row %d, token %d (%g != %g)\n",
                                what, c, i, t, (double) l_ref[t], (double) l_ctx[t]);
                        return false;
                    }
                }
            }
            for (uint32_t s = 0; s < n_seqs; ++s) {
                if (pos[s] == 0) {
                    continue;
                }
                const float diff = recurrent_state_diff(ctxs[c], ctx_ref, (llama_seq_id) s);
                max_state = std::max(max_state, diff);
                if (!(diff <= eps_state)) {
                    fprintf(stderr, "test_prompt_shapes : %s: context %zu: seq %u state differs from the reference by %g\n",
                            what, c, s, (double) diff);
                    return false;
                }
            }
        }
        return true;
    };

    const auto rollback = [&](uint32_t seq, uint32_t n_rows) {
        pos[seq] -= (llama_pos) n_rows;
        bool removed = llama_memory_seq_rm(llama_get_memory(ctx_ref), (llama_seq_id) seq, pos[seq], -1);
        if (!removed) {
            fprintf(stderr, "test_prompt_shapes : the reference refused the rollback of seq %u to %d\n", seq, pos[seq]);
        }
        for (size_t c = 0; c < ctxs.size(); ++c) {
            if (!llama_memory_seq_rm(llama_get_memory(ctxs[c]), (llama_seq_id) seq, pos[seq], -1)) {
                fprintf(stderr, "test_prompt_shapes : context %zu refused the rollback of seq %u to %d\n", c, seq, pos[seq]);
                removed = false;
            }
        }
        return removed;
    };

    bool ok = true;
    ok = ok && step("two prompts",                                {70, 9, 0, 0},   {false, false, false, false});
    ok = ok && step("their replies",                              {3, 3, 0, 0},    {true, true, false, false});
    ok = ok && step("a prompt beside two replies",                {3, 3, 45, 0},   {true, true, false, false});
    ok = ok && step("a prompt of one row beside three replies",   {3, 3, 3, 1},    {true, true, true, false});
    ok = ok && step("a long prompt between replies",              {3, 100, 3, 3},  {true, false, true, true});
    ok = ok && step("two prompts beside two replies",             {3, 33, 3, 131}, {true, false, true, false});
    ok = ok && step("three prompts, read together, beside a reply", {40, 3, 50, 70}, {false, true, false, false});
    ok = ok && step("all reply",                                  {3, 3, 3, 3},    {true, true, true, true});
    ok = ok && rollback(1, 2);
    ok = ok && rollback(3, 1);
    ok = ok && step("the rolled back ones replay beside a prompt", {64, 2, 0, 1},  {false, true, false, true});
    ok = ok && step("a prompt as short as its ubatch's kept tail", {3, 3, 3, 3},   {true, false, true, true});
    ok = ok && rollback(1, 2);
    ok = ok && step("its rollback replays",                       {3, 2, 3, 3},    {true, true, true, true});
    if (!ok) {
        fprintf(stderr, "%s : the contexts differ, or a rollback was refused\n", __func__);
    }
    fprintf(stderr, "%s : against a context without the switch, logits max diff %g, state max diff %g\n", __func__,
            (double) max_logits, (double) max_state);
    cleanup();
    return ok;
#endif
}

// LLAMA_GRAPH_PROMPT_KEPT=1 keeps the graph of a prompt's rows in a slot once its shape has come before, and changes
// nothing a sequence reads: the ubatches are those of a context without the switch, so its logits and states are that
// context's to the bounds of two contexts that cut their batches the same way. Here a sequence reads four prompts of
// one ubatch and a tail of 40 rows: without the switch each of the eight is built again on the main scheduler, which
// holds one graph; with it the first prompt's two are built there, the second's in slots, and the later ones' are
// reused where no input of theirs has grown, so that context reuses more graphs (n_slots of 8, reuse expected). With two
// slots every shape takes the slot of the one used least recently, measured before it is taken: the logits and states
// are the reference's all the same, and how many graphs are reused is not checked.
static bool test_prompt_kept(const common_params & params, llama_model * model, const int n_vocab, const char * n_slots,
        const bool reuse_expected) {
#ifdef _WIN32
    GGML_UNUSED(params);
    GGML_UNUSED(model);
    GGML_UNUSED(n_vocab);
    GGML_UNUSED(n_slots);
    GGML_UNUSED(reuse_expected);
    fprintf(stderr, "%s : skipping on windows, which has no setenv\n", __func__);
    return true;
#else
    constexpr uint32_t n_seqs = 2;
    constexpr float    eps    = 1e-7f;

    const auto make_ctx = [&](bool kept) {
        auto cparams = common_context_params_to_llama(params);
        cparams.n_seq_max  = n_seqs;
        cparams.n_rs_seq   = 2;
        cparams.n_ctx      = 1024;
        cparams.n_batch    = 512;
        cparams.n_ubatch   = 64;
        cparams.kv_unified = true;
        // the switch is read when a context is made
        if (kept) {
            setenv("LLAMA_GRAPH_PROMPT_KEPT", "1", 1);
        }
        llama_context * ctx = llama_init_from_model(model, cparams);
        unsetenv("LLAMA_GRAPH_PROMPT_KEPT");
        return ctx;
    };

    // n_slots of 8: slots for the two prompts' shapes and the reply's beside each other, in both contexts
    const bool set_slots = getenv("LLAMA_GRAPH_SLOTS") == nullptr;
    if (set_slots) {
        setenv("LLAMA_GRAPH_SLOTS", n_slots, 1);
    }
    llama_context * ctx_ref  = make_ctx(false);
    llama_context * ctx_kept = make_ctx(true);
    if (set_slots) {
        unsetenv("LLAMA_GRAPH_SLOTS");
    }
    const auto cleanup = [&]() {
        llama_free(ctx_ref);
        llama_free(ctx_kept);
    };
    if (ctx_ref == nullptr || ctx_kept == nullptr) {
        fprintf(stderr, "%s : failed to init contexts\n", __func__);
        cleanup();
        return false;
    }
    if (llama_n_rs_seq(ctx_ref) < 2) {
        fprintf(stderr, "%s : skipping because n_rs_seq is too small\n", __func__);
        cleanup();
        return true;
    }

    std::vector<llama_pos> pos(n_seqs, 0);
    float max_logits = 0.0f;

    // one decode in both contexts: counts[s] rows of each sequence from where it stands, a logits row for its last
    const auto step = [&](const char * what, const std::vector<uint32_t> & counts) {
        uint32_t n_rows = 0;
        for (const uint32_t count : counts) {
            n_rows += count;
        }
        llama_batch batch = llama_batch_init(n_rows, 0, 1);
        std::vector<int32_t> out_rows;
        for (uint32_t s = 0; s < n_seqs; ++s) {
            for (uint32_t i = 0; i < counts[s]; ++i) {
                const bool output = i + 1 == counts[s];
                if (output) {
                    out_rows.push_back(batch.n_tokens);
                }
                const llama_pos p = pos[s] + (llama_pos) i;
                common_batch_add(batch, (llama_token) ((11*(uint32_t) p + 37*s + 5) % (uint32_t) n_vocab), p, { (llama_seq_id) s }, output);
            }
            pos[s] += (llama_pos) counts[s];
        }
        const bool decoded = llama_decode(ctx_ref, batch) == 0 && llama_decode(ctx_kept, batch) == 0;
        llama_batch_free(batch);
        if (!decoded) {
            fprintf(stderr, "test_prompt_kept : %s: decode failed\n", what);
            return false;
        }
        for (const int32_t i : out_rows) {
            const float * l_ref  = llama_get_logits_ith(ctx_ref, i);
            const float * l_kept = llama_get_logits_ith(ctx_kept, i);
            if (l_ref == nullptr || l_kept == nullptr) {
                fprintf(stderr, "test_prompt_kept : %s: missing logits at row %d\n", what, i);
                return false;
            }
            for (int t = 0; t < n_vocab; ++t) {
                max_logits = std::max(max_logits, std::fabs(l_ref[t] - l_kept[t]));
                if (!(std::fabs(l_ref[t] - l_kept[t]) <= eps)) {
                    fprintf(stderr, "test_prompt_kept : %s: logits differ at row %d, token %d (%g != %g)\n",
                            what, i, t, (double) l_ref[t], (double) l_kept[t]);
                    return false;
                }
            }
        }
        for (uint32_t s = 0; s < n_seqs; ++s) {
            if (pos[s] == 0) {
                continue;
            }
            const float diff = recurrent_state_diff(ctx_kept, ctx_ref, (llama_seq_id) s);
            if (!(diff <= state_eps)) {
                fprintf(stderr, "test_prompt_kept : %s: seq %u state differs from the reference by %g\n", what, s, (double) diff);
                return false;
            }
        }
        return true;
    };

    bool ok = true;
    ok = ok && step("a prompt of one ubatch and a tail",       {104, 0});
    ok = ok && step("the same shapes: built in slots",          {104, 0});
    ok = ok && step("a reply row beside another sequence's",    {1, 1});
    ok = ok && step("the same shapes: reused",                  {104, 0});
    ok = ok && step("another sequence's prompt",                {0, 70});
    ok = ok && step("the same shapes after it: reused",         {104, 0});
    if (!ok) {
        fprintf(stderr, "%s : the contexts differ\n", __func__);
        cleanup();
        return false;
    }

    const int32_t n_reused_ref  = llama_perf_context(ctx_ref).n_reused;
    const int32_t n_reused_kept = llama_perf_context(ctx_kept).n_reused;
    fprintf(stderr, "%s : %s slots: logits max diff %g; graphs reused: %d without the switch, %d with it\n", __func__,
            n_slots, (double) max_logits, n_reused_ref, n_reused_kept);
    cleanup();
    // how many of the later prompts' graphs are reused is the model's (an input that grew rejects a kept graph once)
    if (reuse_expected && n_reused_kept <= n_reused_ref) {
        fprintf(stderr, "%s : the switch reused no kept prompt graph\n", __func__);
        return false;
    }
    return true;
#endif
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    params.sampling.seed = 1234;
    params.n_predict = 1;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    ggml_backend_load_all();

    common_init_result_ptr llama_init = common_init_from_params(params);
    llama_model * model = llama_init->model();
    if (model == nullptr) {
        fprintf(stderr, "%s : failed to init model\n", __func__);
        return 1;
    }

    if (!llama_model_is_recurrent(model) && !llama_model_is_hybrid(model)) {
        fprintf(stderr, "%s : skipping for non-recurrent model\n", __func__);
        return 0;
    }

    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int           n_vocab = llama_vocab_n_tokens(vocab);

    llama_context * ctx_src = make_ctx(params, model);
    llama_context * ctx_dst = make_ctx(params, model);
    if (ctx_src == nullptr || ctx_dst == nullptr) {
        fprintf(stderr, "%s : failed to init contexts\n", __func__);
        return 1;
    }

    if (llama_n_rs_seq(ctx_src) == 0) {
        fprintf(stderr, "%s : skipping because n_rs_seq is disabled\n", __func__);
        llama_free(ctx_src);
        llama_free(ctx_dst);
        return 0;
    }

    std::vector<llama_token> tokens;
    if (llama_vocab_type(vocab) == LLAMA_VOCAB_TYPE_NONE) {
        tokens = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    } else {
        tokens = common_tokenize(ctx_src, "The quick brown fox jumps over the lazy dog", true);
    }
    const uint32_t n_rs_seq = llama_n_rs_seq(ctx_src);
    constexpr uint32_t n_rollback = 3;
    if (n_rs_seq < n_rollback) {
        fprintf(stderr, "%s : skipping because n_rs_seq is too small\n", __func__);
        llama_free(ctx_src);
        llama_free(ctx_dst);
        return 0;
    }
    if (tokens.empty()) {
        fprintf(stderr, "%s : not enough prompt tokens\n", __func__);
        return 1;
    }
    tokens.resize(n_rs_seq + 1, tokens.back());

    const uint32_t  n_tokens     = tokens.size();
    const llama_pos rollback_pos = (llama_pos) n_tokens - n_rollback;

    // Decode the full prompt on the source, then roll back three positions.
    // Replaying them crosses DSV4's ratio-4 compressor boundary.
    // Rollback leaves the recurrent memory in a snapshot state (rs_idx != 0).
    if (!decode_tokens(ctx_src, tokens, n_tokens)) {
        fprintf(stderr, "%s : failed to decode prompt\n", __func__);
        return 1;
    }
    if (!llama_memory_seq_rm(llama_get_memory(ctx_src), 0, rollback_pos, -1)) {
        fprintf(stderr, "%s : rollback failed\n", __func__);
        return 1;
    }

    // the truth every rolled-back or restored state is held to: a context that decoded only up to the rollback point
    llama_context * ctx_truth = make_ctx(params, model);
    if (ctx_truth == nullptr || !decode_tokens(ctx_truth, tokens, rollback_pos)) {
        fprintf(stderr, "%s : failed to decode the reference prefix\n", __func__);
        return 1;
    }
    const auto matches = [&](llama_context * ctx, llama_context * truth, const char * what) {
        const float diff = recurrent_state_diff(ctx, truth, 0);
        fprintf(stderr, "%s : %s, state max diff %g\n", __func__, what, (double) diff);
        return diff <= state_eps;
    };
    const auto matches_truth = [&](llama_context * ctx, const char * what) {
        return matches(ctx, ctx_truth, what);
    };
    if (!matches_truth(ctx_src, "rolled back")) {
        return 1;
    }

    // Save the rolled-back state and restore it into a fresh context.
    common_prompt_checkpoint ckpt;
    ckpt.update_tgt(ctx_src, 0, 0);
    ckpt.load_tgt(ctx_dst, 0, 0);
    if (!matches_truth(ctx_dst, "restored")) {
        return 1;
    }

    constexpr float eps = 1e-5f;
    // the full replay's rows, which the dirty context replays against
    std::vector<std::vector<float>> logits_src_replay(n_rollback);
    const auto replay_and_compare = [&](const char * mode, llama_pos p_from) {
        if (!decode_span(ctx_src, tokens, p_from, n_tokens) ||
            !decode_span(ctx_dst, tokens, p_from, n_tokens)) {
            fprintf(stderr, "%s : %s replay failed\n", __func__, mode);
            return false;
        }

        for (llama_pos pos = p_from; pos < (llama_pos) n_tokens; ++pos) {
            const float * logits_src = llama_get_logits_ith(ctx_src, pos - p_from);
            const float * logits_dst = llama_get_logits_ith(ctx_dst, pos - p_from);
            if (logits_src == nullptr || logits_dst == nullptr) {
                fprintf(stderr, "%s : missing %s logits at position %d\n", __func__, mode, pos);
                return false;
            }

            if (p_from == rollback_pos) {
                logits_src_replay[pos - rollback_pos].assign(logits_src, logits_src + n_vocab);
            }
            for (int token = 0; token < n_vocab; ++token) {
                if (std::fabs(logits_src[token] - logits_dst[token]) > eps) {
                    fprintf(stderr, "%s : %s logits mismatch at position %d, token %d (%g != %g)\n",
                            __func__, mode, pos, token, (double) logits_src[token], (double) logits_dst[token]);
                    return false;
                }
            }
        }
        return true;
    };
    if (!replay_and_compare("full", rollback_pos)) {
        return 1;
    }

    // the replay is the last batch: rolling it back whole, or past it, would read a state no snapshot kept
    for (const llama_pos p0 : { rollback_pos, rollback_pos - 1 }) {
        if (llama_memory_seq_rm(llama_get_memory(ctx_src), 0, p0, -1)) {
            fprintf(stderr, "%s : a rollback of %d rows after a batch of %u was not refused\n",
                    __func__, (int) n_tokens - p0, n_rollback);
            return 1;
        }
    }

    // all but the replay's first row
    const llama_pos partial_pos = rollback_pos + 1;
    if (!llama_memory_seq_rm(llama_get_memory(ctx_src), 0, partial_pos, -1) ||
        !llama_memory_seq_rm(llama_get_memory(ctx_dst), 0, partial_pos, -1)) {
        fprintf(stderr, "%s : partial rollback failed\n", __func__);
        return 1;
    }
    // its truth: a context that decoded only up to the replay's first row
    llama_context * ctx_truth_partial = make_ctx(params, model);
    if (ctx_truth_partial == nullptr || !decode_tokens(ctx_truth_partial, tokens, partial_pos)) {
        fprintf(stderr, "%s : failed to decode the partial reference prefix\n", __func__);
        return 1;
    }

    constexpr llama_state_seq_flags partial_flags = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;
    common_prompt_checkpoint ckpt_partial;
    ckpt_partial.update_tgt(ctx_src, 0, partial_flags);
    ckpt_partial.load_tgt(ctx_dst, 0, partial_flags);
    const bool partial_ok = matches(ctx_src, ctx_truth_partial, "rolled back after the replay") &&
                            matches(ctx_dst, ctx_truth_partial, "partially restored");
    llama_free(ctx_truth_partial);
    if (!partial_ok) {
        return 1;
    }

    // a partial restore comes without snapshots too (dsv4 clears its compressed caches only on a full one): the ones
    // ctx_dst's own replay wrote must not be read back
    if (llama_memory_seq_rm(llama_get_memory(ctx_dst), 0, rollback_pos, -1)) {
        fprintf(stderr, "%s : a rollback right after a partial restore was not refused\n", __func__);
        return 1;
    }

    if (!replay_and_compare("partial", partial_pos)) {
        return 1;
    }

    // Repeat the load into a context that already has its own rollback state:
    // groups 1..n_rs_seq hold a different prompt's history, and rs_idx[0] is
    // non-zero at load time. The restore must wipe that state and still match.
    llama_context * ctx_dirty = make_ctx(params, model);
    if (ctx_dirty == nullptr) {
        fprintf(stderr, "%s : failed to init dirty ctx\n", __func__);
        return 1;
    }

    std::vector<llama_token> noise = tokens;
    for (auto & t : noise) {
        t = (t + 1) % n_vocab;
        if (t < 0) {
            t = 0;
        }
    }
    if (!decode_tokens(ctx_dirty, noise, n_tokens)) {
        fprintf(stderr, "%s : dirty prompt decode failed\n", __func__);
        return 1;
    }
    if (!llama_memory_seq_rm(llama_get_memory(ctx_dirty), 0, rollback_pos, -1)) {
        fprintf(stderr, "%s : dirty rollback failed\n", __func__);
        return 1;
    }

    ckpt.load_tgt(ctx_dirty, 0, 0);
    if (!matches_truth(ctx_dirty, "restored over a pending rollback")) {
        return 1;
    }

    // the restored state comes without snapshots: the ones its own noise batch wrote must not be read back
    if (llama_memory_seq_rm(llama_get_memory(ctx_dirty), 0, rollback_pos - 1, -1)) {
        fprintf(stderr, "%s : a rollback right after a restore was not refused\n", __func__);
        return 1;
    }

    if (!decode_span(ctx_dirty, tokens, rollback_pos, n_tokens)) {
        fprintf(stderr, "%s : dirty replay failed\n", __func__);
        return 1;
    }
    for (uint32_t i = 0; i < n_rollback; ++i) {
        const llama_pos pos = rollback_pos + i;
        const float * logits_dirty = llama_get_logits_ith(ctx_dirty, i);
        if (logits_dirty == nullptr) {
            fprintf(stderr, "%s : missing dirty logits at position %d\n", __func__, pos);
            return 1;
        }

        for (int token = 0; token < n_vocab; ++token) {
            if (std::fabs(logits_src_replay[i][token] - logits_dirty[token]) > eps) {
                fprintf(stderr, "%s : dirty-ctx logits mismatch at position %d, token %d (%g != %g)\n",
                        __func__, pos, token, (double) logits_src_replay[i][token], (double) logits_dirty[token]);
                return 1;
            }
        }
    }

    fprintf(stderr, "%s : recurrent rollback checkpoint restored successfully\n", __func__);
    llama_free(ctx_src);
    llama_free(ctx_dst);
    llama_free(ctx_dirty);
    llama_free(ctx_truth);

    if (!test_multi_seq_split_replay(params, model, n_vocab)) {
        return 1;
    }

    if (!test_short_verify_rollback(params, model, n_vocab)) {
        return 1;
    }

    if (!test_changed_sets(params, model, n_vocab)) {
        return 1;
    }
    if (!test_prompt_shapes(params, model, n_vocab)) {
        return 1;
    }
    // A model split by layers over several cards runs its graphs as a pipeline, and such a context does not share its
    // graph buffers, which a kept prompt graph needs: the switch does nothing there and only the outputs are compared.
    // The model's devices: the ones named on the command line, or every card.
    size_t n_devices = 0;
    for (const ggml_backend_dev_t dev : params.devices) {
        n_devices += dev != nullptr ? 1 : 0;
    }
    for (size_t i = 0; params.devices.empty() && i < ggml_backend_dev_count(); ++i) {
        n_devices += ggml_backend_dev_type(ggml_backend_dev_get(i)) == GGML_BACKEND_DEVICE_TYPE_GPU ? 1 : 0;
    }
    const bool kept_possible = n_devices <= 1 || params.split_mode != LLAMA_SPLIT_MODE_LAYER;
    if (!test_prompt_kept(params, model, n_vocab, "8", kept_possible) || !test_prompt_kept(params, model, n_vocab, "2", false)) {
        return 1;
    }

    return 0;
}
