// Inkling 2's optimized paths give the logits of the path each one replaced, which its switch keeps (--inkling-path
// <feature>/<case> on --inkling-path-device, CPU for the host alone). Each context records the nodes it runs: the
// optimized path must run its nodes, on the model's device, and the legacy path none of them. The logits of every
// output row are held to the fixture's tolerance. The features:
//
//   rel    the relative position bias added inside flash attention (ggml_flash_attn_ext_add_rel_pos) against the bias
//          gathered for every cell and added to the mask (LLAMA_INKLING2_REL_POS_LEGACY=1)
//   rel-packed   the same comparison with packed masks requested, including on the legacy path
//   group  with packed masks, a drafted step classed by its sequences, whose rows of a sequence read each cell once
//          together on the decode kernel, against a dense step on the prefill kernel (LLAMA_INKLING2_REL_GROUP_LEGACY=1)
//   decode-rows   with packed masks, decode steps of fewer than 8 rows, a long sequence's among them, reading only
//          their rows' cells against dense steps (LLAMA_INKLING2_DECODE_CLASS_LEGACY=1)
//   fuse   the residual depth mix, the norm clips, the attention output gate and the experts' SITU as one op each
//          against their elementwise graphs (LLAMA_INKLING2_FUSE_LEGACY=1)
//   conv   each short convolution as one op (ggml_short_conv) against ssm_conv and its elementwise taps and residual
//          (LLAMA_INKLING2_CONV_LEGACY=1); the state both leave is carried across the steps of every case
//   kv-q8_0, kv-q4_0   keys and values of every layer in a q8_0 or q4_0 cache against the F16 cache
//   kv-split   the global layers' keys and values in q8_0 and the sliding layers' in q4_0 (type_k_swa, type_v_swa)
//          against the F16 cache
//
// A q4_0 cache cannot meet the fixture's tolerance against F16: its rounding alone moves these logits by an NMSE of
// 3.3e-3 (prompt) to 4.0e-3 (unified), as measured on Nov 17, 2026, where q8_0's moves them by 6.2e-6 to 1.3e-5, the
// square of q4_0's 16 times coarser step. The q4_0 and split caches are held instead to 1e-2 against F16, and on a device
// to 1e-3 against the same cache on the CPU: the two backends' keys and values differ by float rounding, and a value
// near a q4_0 boundary lands a whole step apart, which read 3.5e-5 to 2.6e-4 over seeds 1234, 7, 99 and 4242. A cache
// read with the wrong type, offset or layout is past both bars by orders of magnitude.
//
// The fixture's global table is lifted to the served extent of 1024 so that rows more than 1024 positions past a key
// read the sentinel entry, and its weights spread at 0.1, where the logits follow every earlier token. The cases:
//
//   prompt    1300 tokens of one sequence in ubatches of 512: the last rows reach keys more than 1024 back
//   decode    24 tokens one at a time after a prompt of 1012, crossing distance 1024 to the first keys
//   unified   three sequences in one unified cache, prompts of 40, 56 and 72 tokens, then 8 steps of all three together
//   streams   the same with a cache stream per sequence
//   drafted   eight sequences of 1,030 cells in one unified cache, then 4 steps of a token and 3 drafts a sequence
//   long      one sequence of 6,000 cells and three of 800 in one unified cache, then steps of 1 and 4 rows, and of
//             a token and 2 drafts a sequence
//
// --inkling-window <device> is a check of another kind in this file for its helpers: sessions that share one cache give
// each session the logits it has alone (test_inkling_window).

#include "test-llama-archs.h"

#include "llama-archs-gguf.h"
#include "../ggml/src/ggml-attn-mask.h"

#include <optional>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <map>
#include <numeric>
#include <random>
#include <stdexcept>

namespace {

constexpr double tolerance = 1e-4;
constexpr uint32_t extent = 1024; // the served model's global table
constexpr double q4_0_rounding = 1e-2; // a q4_0 cache against F16 (see above)
constexpr double q4_0_on_host  = 1e-3; // and against itself on the CPU

// the nodes of an optimized path, by kind; a kind is a short name, empty for a node of no kind. The optimized path runs
// every kind and no other, the legacy path none of them. The legacy path is the one the environment switch selects,
// or the one of the context the configuration is given with legacy set.
struct feature {
    const char * legacy_env;
    std::vector<std::string> kinds;
    std::function<std::string(const ggml_tensor *)> kind_of;
    std::function<void(llama_context_params &, bool)> configure = nullptr;
    double vs_legacy = tolerance; // the bar against the legacy path
    double vs_host = 0.0;         // when set, the bar against the same path on the CPU, run on a device
};

// an attention node by the types of the cache it reads
std::string kv_kind(const ggml_tensor * t) {
    if (t->op != GGML_OP_FLASH_ATTN_EXT) {
        return "";
    }
    return std::string("K ") + ggml_type_name(t->src[1]->type) + ", V " + ggml_type_name(t->src[2]->type);
}

// keys and values of the global layers in global, of the sliding layers in swa, against F16
std::function<void(llama_context_params &, bool)> kv_types(ggml_type global, ggml_type swa) {
    return [=](llama_context_params & params, bool legacy) {
        params.type_k = params.type_v = legacy ? GGML_TYPE_F16 : global;
        params.type_k_swa = params.type_v_swa = legacy ? GGML_TYPE_F16 : swa;
    };
}

const std::map<std::string, feature> & features() {
    static const std::map<std::string, feature> all = {
        {"rel", {"LLAMA_INKLING2_REL_POS_LEGACY", {"attention with the bias"}, [](const ggml_tensor * t) {
            return t->op == GGML_OP_FLASH_ATTN_EXT && t->src[6] && t->src[9] ? "attention with the bias" : "";
        }}},
        {"rel-packed", {"LLAMA_INKLING2_REL_POS_LEGACY", {"packed attention with the bias"}, [](const ggml_tensor * t) {
            return t->op == GGML_OP_FLASH_ATTN_EXT && t->src[6] && t->src[3]->type == GGML_TYPE_I16 ? "packed attention with the bias" : "";
        }, [](llama_context_params & params, bool) {
            params.attn_mask_bits = true;
            params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;
        }}},
        {"group", {"LLAMA_INKLING2_REL_GROUP_LEGACY", {"bounded drafted rows"}, [](const ggml_tensor * t) {
            return t->op == GGML_OP_FLASH_ATTN_EXT && t->src[6] && t->src[3]->type == GGML_TYPE_I16 &&
                ggml_flash_attn_ext_get_n_kv_max(t) > 0 && ggml_flash_attn_ext_get_query_group(t) > 1 ?
                "bounded drafted rows" : "";
        }, [](llama_context_params & params, bool) {
            params.attn_mask_bits = true;
            params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;
        }}},
        {"decode-rows", {"LLAMA_INKLING2_DECODE_CLASS_LEGACY", {"decode rows over their own cells"}, [](const ggml_tensor * t) {
            return t->op == GGML_OP_FLASH_ATTN_EXT && t->src[6] && t->src[3]->type == GGML_TYPE_I16 &&
                (ggml_flash_attn_ext_get_n_kv_max(t) > 0 || ggml_flash_attn_ext_has_low_mean_mask(t)) ?
                "decode rows over their own cells" : "";
        }, [](llama_context_params & params, bool) {
            params.attn_mask_bits = true;
            params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;
        }}},
        {"fuse", {"LLAMA_INKLING2_FUSE_LEGACY", {"depth mix", "norm clip", "output gate", "situ"}, [](const ggml_tensor * t) {
            switch (t->op) {
                case GGML_OP_ATTN_RES_MIX: return "depth mix";
                case GGML_OP_NORM_CLIP:    return "norm clip";
                case GGML_OP_GLU:
                    return ggml_get_glu_op(t) == GGML_GLU_OP_SIGLU ? "output gate" :
                           ggml_get_glu_op(t) == GGML_GLU_OP_SITU  ? "situ" : "";
                default: return "";
            }
        }}},
        {"conv", {"LLAMA_INKLING2_CONV_LEGACY", {"static taps", "dynamic taps"}, [](const ggml_tensor * t) {
            return t->op != GGML_OP_SHORT_CONV ? "" : t->src[2] ? "dynamic taps" : "static taps";
        }}},
        {"kv-q8_0", {nullptr, {"K q8_0, V q8_0"}, kv_kind, kv_types(GGML_TYPE_Q8_0, GGML_TYPE_Q8_0)}},
        {"kv-q4_0", {nullptr, {"K q4_0, V q4_0"}, kv_kind, kv_types(GGML_TYPE_Q4_0, GGML_TYPE_Q4_0), q4_0_rounding,
            q4_0_on_host}},
        {"kv-split", {nullptr, {"K q8_0, V q8_0", "K q4_0, V q4_0"}, kv_kind, kv_types(GGML_TYPE_Q8_0, GGML_TYPE_Q4_0),
            q4_0_rounding, q4_0_on_host}},
    };
    return all;
}

struct shape {
    uint32_t n_seq_max = 1;
    bool unified = true;
};

// the nodes a context ran, as the scheduler hands them to the eval callback
struct nodes_seen {
    const feature * f = nullptr;
    bool on_host = false;           // the device is the CPU
    std::map<std::string, int> ran; // by kind
    int off_device = 0;             // of a kind, on another device than the model's
};

bool observe(ggml_tensor * t, bool ask, void * user_data) {
    auto * seen = static_cast<nodes_seen *>(user_data);
    if (ask) {
        const std::string kind = seen->f->kind_of(t);
        if (!kind.empty()) {
            seen->ran[kind]++;
            seen->off_device += !t->buffer || ggml_backend_buffer_is_host(t->buffer) != seen->on_host;
        }
    }
    return false; // nothing is read: the graph runs in one piece
}

llama_context_ptr make_context(llama_model * model, const shape & s, nodes_seen & seen, bool legacy) {
    llama_context_params params = llama_context_default_params();
    params.n_ctx = 2048*s.n_seq_max;
    params.n_batch = 2048;
    params.n_ubatch = 512;
    params.n_seq_max = s.n_seq_max;
    params.kv_unified = s.unified;
    params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    params.n_threads = params.n_threads_batch = 4;
    params.cb_eval = observe;
    params.cb_eval_user_data = &seen;
    if (seen.f->configure) {
        seen.f->configure(params, legacy);
    }
    // the switch is read when the context is created
    if (legacy && seen.f->legacy_env) {
        setenv(seen.f->legacy_env, "1", 1);
    }
    llama_context_ptr ctx(llama_init_from_model(model, params));
    if (seen.f->legacy_env) {
        unsetenv(seen.f->legacy_env);
    }
    if (!ctx) {
        throw std::runtime_error("inkling2: the context was not created");
    }
    return ctx;
}

// Append logits for the requested rows of (token, position, sequence).
void decode(llama_context * ctx, const std::vector<std::array<int32_t, 3>> & rows, std::vector<float> & out,
        const std::optional<std::vector<size_t>> & output_rows = std::nullopt) {
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx)));
    llama_batch batch = llama_batch_init(rows.size(), 0, 1);
    for (size_t i = 0; i < rows.size(); ++i) {
        const auto & [token, pos, seq] = rows[i];
        batch_add(batch, token, pos, {seq}, !output_rows || std::binary_search(output_rows->begin(), output_rows->end(), i));
    }
    const int32_t ret = llama_decode(ctx, batch);
    llama_batch_free(batch);
    if (ret != 0) {
        throw std::runtime_error("inkling2: decode failed with " + std::to_string(ret));
    }
    for (size_t i = 0; i < rows.size(); ++i) {
        if (output_rows && !std::binary_search(output_rows->begin(), output_rows->end(), i)) {
            continue;
        }
        const float * row = llama_get_logits_ith(ctx, i);
        out.insert(out.end(), row, row + n_vocab);
    }
}

std::vector<std::array<int32_t, 3>> span(const std::vector<llama_token> & tokens, llama_pos pos0, llama_seq_id seq) {
    std::vector<std::array<int32_t, 3>> rows;
    for (size_t i = 0; i < tokens.size(); ++i) {
        rows.push_back({tokens[i], pos0 + int32_t(i), seq});
    }
    return rows;
}

using path_case = std::function<void(llama_context *, std::vector<float> &)>;

std::map<std::string, std::pair<shape, path_case>> cases(size_t seed) {
    const auto tokens = [seed](uint32_t n, size_t salt) { return get_tokens(n, 128, seed + salt); };
    const auto together = [=](llama_context * ctx, std::vector<float> & out) {
        const uint32_t lens[3] = {40, 56, 72};
        std::vector<std::array<int32_t, 3>> rows;
        for (int s = 0; s < 3; ++s) {
            const auto r = span(tokens(lens[s], s), 0, s);
            rows.insert(rows.end(), r.begin(), r.end());
        }
        decode(ctx, rows, out);
        for (uint32_t i = 0; i < 8; ++i) {
            rows.clear();
            for (int s = 0; s < 3; ++s) {
                rows.push_back({tokens(8, 10 + s)[i], int32_t(lens[s] + i), s});
            }
            decode(ctx, rows, out);
        }
    };
    std::map<std::string, std::pair<shape, path_case>> ret;
    ret["prompt"] = {{}, [=](llama_context * ctx, std::vector<float> & out) {
        decode(ctx, span(tokens(1300, 0), 0, 0), out);
    }};
    ret["decode"] = {{}, [=](llama_context * ctx, std::vector<float> & out) {
        decode(ctx, span(tokens(1012, 0), 0, 0), out);
        const auto steps = tokens(24, 1);
        for (uint32_t i = 0; i < steps.size(); ++i) {
            decode(ctx, {{steps[i], int32_t(1012 + i), 0}}, out);
        }
    }};
    ret["outputs"] = {{2, true}, [=](llama_context * ctx, std::vector<float> & out) {
        decode(ctx, span(tokens(300, 3), 0, 0), out, std::vector<size_t>{});
        auto rows = span(tokens(160, 0), 300, 0);
        const auto second = span(tokens(160, 1), 0, 1);
        rows.insert(rows.end(), second.begin(), second.end());
        decode(ctx, rows, out, std::vector<size_t>{3, 159, 267, 319});
        decode(ctx, {{11, 460, 0}, {22, 160, 1}}, out);
    }};
    // 8 sequences of 1,030 cells, about an eighth of the pool each, then 4 steps of a token and 3 drafts a sequence:
    // with a packed mask a drafted step is classed by its sequences (bounded rows) on the global layers, and the
    // decode kernel reads a sequence's 4 rows together
    ret["drafted"] = {{8, true}, [=](llama_context * ctx, std::vector<float> & out) {
        for (int s = 0; s < 8; ++s) {
            decode(ctx, span(tokens(1030, 30 + s), 0, s), out, std::vector<size_t>{1029});
        }
        for (int32_t step = 0; step < 4; ++step) {
            std::vector<std::array<int32_t, 3>> rows;
            for (int s = 0; s < 8; ++s) {
                const auto r = span(tokens(4, 40 + 8*step + s), 1030 + 4*step, s);
                rows.insert(rows.end(), r.begin(), r.end());
            }
            decode(ctx, rows, out);
        }
    }};
    // one sequence of 6,000 cells, most of the pool, among three of 800, then steps of fewer than 8 rows: the long
    // sequence alone, a short one alone, all four, and all four with a token and 2 drafts each. Each step reads only
    // its rows' cells, the long sequence's as a low mean and a short one's as a bounded row
    ret["long"] = {{8, true}, [=](llama_context * ctx, std::vector<float> & out) {
        const int32_t lens[4] = {6000, 800, 800, 800};
        for (int s = 0; s < 4; ++s) {
            // in calls of 2,000 rows, which a batch of 2,048 holds
            const auto prompt = span(tokens(lens[s], 50 + s), 0, s);
            for (size_t i0 = 0; i0 < prompt.size(); i0 += 2000) {
                const size_t n = std::min<size_t>(2000, prompt.size() - i0);
                decode(ctx, {prompt.begin() + i0, prompt.begin() + i0 + n}, out, std::vector<size_t>{n - 1});
            }
        }
        int32_t next[4] = {lens[0], lens[1], lens[2], lens[3]};
        const auto step = [&](std::initializer_list<int> seqs, uint32_t per_seq, size_t salt) {
            std::vector<std::array<int32_t, 3>> rows;
            for (const int s : seqs) {
                const auto r = span(tokens(per_seq, salt + s), next[s], s);
                rows.insert(rows.end(), r.begin(), r.end());
                next[s] += per_seq;
            }
            decode(ctx, rows, out);
        };
        for (size_t i = 0; i < 3; ++i) {
            step({0}, 1, 60 + 10*i);
            step({1}, 1, 61 + 10*i);
            step({0, 1, 2, 3}, 1, 62 + 10*i);
        }
        for (size_t i = 0; i < 2; ++i) {
            step({0, 1, 2, 3}, 3, 100 + 10*i);
        }
    }};
    ret["unified"] = {{3, true}, together};
    ret["streams"] = {{3, false}, together};
    ret["churn"] = {{3, true}, [=](llama_context * ctx, std::vector<float> & out) {
        together(ctx, out);
        if (!llama_memory_seq_rm(llama_get_memory(ctx), 1, -1, -1)) {
            throw std::runtime_error("inkling2: could not clear sequence 1");
        }
        decode(ctx, span(tokens(19, 21), 0, 1), out);
        decode(ctx, {{11, 48, 0}, {22, 19, 1}, {33, 80, 2}}, out);
    }};
    return ret;
}

} // namespace

int test_inkling_path(const std::string & path, size_t seed, ggml_backend_dev_t dev) {
    const size_t slash = path.find('/');
    const std::string fname = path.substr(0, slash);
    const std::string name = slash == std::string::npos ? "" : path.substr(slash + 1);
    const auto all = cases(seed);
    const auto it = all.find(name);
    const auto fit = features().find(fname);
    if (it == all.end() || fit == features().end()) {
        fprintf(stderr, "inkling2 path: no feature or case in %s\n", path.c_str());
        return 1;
    }
    const bool on_host = ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU;
    auto gguf = get_gguf_ctx(LLM_ARCH_INKLING2, true);
    gguf_set_val_u32(gguf.get(), llama_arch_kv_name(LLM_ARCH_INKLING2, LLM_KV_ATTENTION_RELATIVE_EXTENT).c_str(), extent);
    gguf_set_val_u32(gguf.get(), llama_arch_kv_name(LLM_ARCH_INKLING2, LLM_KV_CONTEXT_LENGTH).c_str(), 8192);
    const auto load = [&](bool host) {
        return get_model(gguf.get(), nullptr, seed, 0.1f, host ? std::vector<ggml_backend_dev_t>{} :
                std::vector<ggml_backend_dev_t>{dev});
    };
    llama_model_ptr model = load(on_host);
    const feature & f = fit->second;
    std::vector<float> expected, actual, host;
    nodes_seen seen_legacy, seen_new;
    seen_legacy.f = seen_new.f = &fit->second;
    seen_legacy.on_host = seen_new.on_host = on_host;
    {
        auto ctx = make_context(model.get(), it->second.first, seen_legacy, true);
        it->second.second(ctx.get(), expected);
    }
    {
        auto ctx = make_context(model.get(), it->second.first, seen_new, false);
        it->second.second(ctx.get(), actual);
    }
    const bool against_host = f.vs_host > 0.0 && !on_host;
    if (against_host) {
        llama_model_ptr host_model = load(true);
        nodes_seen seen_host;
        seen_host.f = &f;
        seen_host.on_host = true;
        auto ctx = make_context(host_model.get(), it->second.first, seen_host, false);
        it->second.second(ctx.get(), host);
    }
    const auto & kinds = f.kinds;
    const auto is_kind = [&](const std::string & k) { return std::find(kinds.begin(), kinds.end(), k) != kinds.end(); };
    bool ran = seen_new.off_device == 0;
    for (const auto & kind : kinds) {
        const int n = seen_new.ran.count(kind) ? seen_new.ran.at(kind) : 0;
        fprintf(stderr, "inkling2 %s on %s: %d %s nodes ran\n", path.c_str(), ggml_backend_dev_name(dev), n, kind.c_str());
        ran = ran && n > 0 && !seen_legacy.ran.count(kind);
    }
    for (const auto & [kind, n] : seen_new.ran) {
        if (!is_kind(kind)) {
            fprintf(stderr, "inkling2 %s: %d %s nodes ran, none were to\n", path.c_str(), n, kind.c_str());
            ran = false;
        }
    }
    fprintf(stderr, "inkling2 %s: %d of them off the device\n", path.c_str(), seen_new.off_device);
    const bool same = expected.size() == actual.size() && !expected.empty();
    const double d = same ? nmse(expected, actual) : INFINITY;
    fprintf(stderr, "inkling2 %s: %zu logits, NMSE %g (at most %g)\n", path.c_str(), expected.size(), d, f.vs_legacy);
    bool ok = ran && d <= f.vs_legacy;
    if (against_host) {
        const double dh = host.size() == actual.size() ? nmse(host, actual) : INFINITY;
        fprintf(stderr, "inkling2 %s: against the CPU, NMSE %g (at most %g)\n", path.c_str(), dh, f.vs_host);
        ok = ok && dh <= f.vs_host;
    }
    fprintf(stderr, "%s inkling2 %s\n", ok ? "OK" : "FAIL", path.c_str());
    return ok ? 0 : 1;
}

// Sessions that share one cache, where a sliding layer holds a window a session and one micro-batch, give each session
// the logits it has alone. In every round each session adds a run of 8 to 64 rows. The rows go to a cache a run a call
// (one micro-batch), to another in calls of a micro-batch's rows, and to a third a round a call. The equal split cuts a
// call whose sessions bring runs of different lengths into several micro-batches, as it does a draft's verification
// beside other sessions, and the cells of them all are found before the first runs.
int test_inkling_window(size_t seed, ggml_backend_dev_t dev) {
    constexpr uint32_t n_sessions = 6, n_rounds = 6, n_ubatch = 64;
    using rows_t = std::vector<std::array<int32_t, 3>>;
    using logits_t = std::map<std::pair<int32_t, int32_t>, std::vector<float>>; // by session and position

    const bool on_host = ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU;
    auto gguf = get_gguf_ctx(LLM_ARCH_INKLING2, true);
    gguf_set_val_u32(gguf.get(), llama_arch_kv_name(LLM_ARCH_INKLING2, LLM_KV_CONTEXT_LENGTH).c_str(), 8192);
    llama_model_ptr model = get_model(gguf.get(), nullptr, seed, 0.1f, on_host ? std::vector<ggml_backend_dev_t>{} :
            std::vector<ggml_backend_dev_t>{dev});
    const size_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));

    std::mt19937 rng(seed);
    std::vector<std::vector<llama_token>> tokens;
    for (uint32_t s = 0; s < n_sessions; ++s) {
        tokens.push_back(get_tokens(n_rounds*n_ubatch, 128, seed + s));
    }
    std::vector<rows_t> rounds(n_rounds);
    std::vector<int32_t> next(n_sessions, 0);
    for (auto & round : rounds) {
        std::vector<int32_t> order(n_sessions);
        std::iota(order.begin(), order.end(), 0);
        for (uint32_t i = n_sessions - 1; i > 0; --i) {
            std::swap(order[i], order[rng() % (i + 1)]);
        }
        for (const int32_t s : order) {
            for (uint32_t n = 8 + rng() % 57; n > 0; --n, ++next[s]) {
                round.push_back({tokens[s][next[s]], next[s], s});
            }
        }
    }

    const auto context = [&](uint32_t n_seq_max) {
        llama_context_params params = llama_context_default_params();
        params.n_ctx = n_rounds*n_ubatch*n_seq_max;
        params.n_batch = n_sessions*n_ubatch;
        params.n_ubatch = n_ubatch;
        params.n_seq_max = n_seq_max;
        params.kv_unified = true;
        params.swa_full = false; // as served: the sliding layers keep a window a session
        params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        params.n_threads = params.n_threads_batch = 4;
        llama_context_ptr ctx(llama_init_from_model(model.get(), params));
        if (!ctx) {
            throw std::runtime_error("inkling2: the context was not created");
        }
        return ctx;
    };
    // rows in calls of n_call, as sequence 0 when the session has the context alone
    const auto feed = [&](llama_context * ctx, const rows_t & rows, size_t n_call, bool is_alone, logits_t & out) {
        for (size_t i0 = 0; i0 < rows.size(); i0 += n_call) {
            const rows_t call(rows.begin() + i0, rows.begin() + std::min(rows.size(), i0 + n_call));
            rows_t in = call;
            for (auto & row : in) {
                row[2] = is_alone ? 0 : row[2];
            }
            std::vector<float> logits;
            decode(ctx, in, logits);
            for (size_t i = 0; i < call.size(); ++i) {
                out[{call[i][2], call[i][1]}].assign(logits.begin() + i*n_vocab, logits.begin() + (i + 1)*n_vocab);
            }
        }
    };

    // a round's runs, each a session's, in the round's order
    const auto runs_of = [](const rows_t & round) {
        std::vector<rows_t> runs;
        for (const auto & row : round) {
            if (runs.empty() || runs.back().back()[2] != row[2]) {
                runs.emplace_back();
            }
            runs.back().push_back(row);
        }
        return runs;
    };

    logits_t alone, by_run, by_ubatch, by_round;
    {
        std::vector<llama_context_ptr> ctx_alone;
        for (uint32_t s = 0; s < n_sessions; ++s) {
            ctx_alone.push_back(context(1));
        }
        const auto ctx_run = context(n_sessions), ctx_ubatch = context(n_sessions), ctx_round = context(n_sessions);
        for (const auto & round : rounds) {
            for (const auto & run : runs_of(round)) {
                feed(ctx_alone[run[0][2]].get(), run, run.size(), true, alone);
                feed(ctx_run.get(), run, run.size(), false, by_run);
            }
            feed(ctx_ubatch.get(), round, n_ubatch, false, by_ubatch);
            feed(ctx_round.get(), round, round.size(), false, by_round);
        }
    }

    const auto far_rows = [&](const char * name, const logits_t & got) {
        size_t far = 0;
        double worst = 0.0;
        for (const auto & [key, row] : alone) {
            const auto it = got.find(key);
            const double d = it == got.end() ? INFINITY : nmse(row, it->second);
            far += !(d <= tolerance);
            worst = std::max(worst, d);
        }
        fprintf(stderr, "inkling2 window, %s: %zu rows, %zu far from the session alone, worst NMSE %g (at most %g)\n",
                name, alone.size(), far, worst, tolerance);
        return far;
    };
    const size_t far_run = far_rows("a run a call", by_run);
    const size_t far_ubatch = far_rows("a micro-batch's rows a call", by_ubatch);
    const size_t far_round = far_rows("a round a call", by_round);
    const bool ok = !alone.empty() && far_run == 0 && far_ubatch == 0 && far_round == 0;
    fprintf(stderr, "%s inkling2 window on %s\n", ok ? "OK" : "FAIL", ggml_backend_dev_name(dev));
    return ok ? 0 : 1;
}
