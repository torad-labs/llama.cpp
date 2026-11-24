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
//   kv1b-q8_0, kv1b-e4m3   the global layers' keys and values at one byte a value, q8_0 or f8_e4m3 without a scale, the
//          sliding layers' in F16, against the F16 cache, on the served attention of one card of four (12 query heads of
//          128 on 2 KV heads); the decode steps of 23 sessions are low-mean steps on every device (the tests set
//          LLAMA_INKLING2_MEAN_MASK_DECODE, LLAMA_INKLING2_MEAN_POOL_MIN=8192 and GGML_CUDA_FATTN_REL_DECODE_ONE_BYTE)
//   far    the global layers' decode by the runs of cells each row reads (LLAMA_INKLING2_KV_RUNS) against its mask, on
//          the served attention of one card of four; with GGML_CUDA_FATTN_REL_RUNS_SHARED the shared runs are read
//          once for a group of rows
//   far-q8_0, far-e4m3   the same on a q8_0 or f8_e4m3 global cache, the sliding layers' F16, against the mask path on
//          the same caches (the tests set GGML_CUDA_FATTN_REL_DECODE_ONE_BYTE)
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
//   served    23 sequences that share a prompt of 1,800 tokens and add 400 each, then 4 steps of a token a sequence: the
//             pool, padded to 16,384 cells, is 7.4 times a row's cells (6.5 as served), so the steps are low-mean, not
//             bounded, and rows reach keys more than 1024 back, shared and their own
//   restore   the same with one sequence saved, dropped and read back before the steps, as the prompt cache does
//   runs      23 sequences that share a prompt and read their own tokens in chunks taken in turn, one just forked, then
//             steps of a token a sequence, one sequence cleared and read again alone, and more steps
//   long      one sequence of 6,000 cells and three of 800 in one unified cache, then steps of 1 and 4 rows, and of
//             a token and 2 drafts a sequence
//
// --inkling-window <device> is a check of another kind in this file for its helpers: sessions that share one cache give
// each session the logits it has alone (test_inkling_window).
// --inkling-stack <devs> is another: the products of a layer's Q input as one launch (test_inkling_stack).

#include "test-llama-archs.h"

#include "llama-archs-gguf.h"
#include "../ggml/src/ggml-attn-mask.h"
#include "../src/llama-kv-runs.h"
#include "../src/llama-memory-hybrid-iswa.h"
#include "../src/llama-model.h"

#include <chrono>
#include <optional>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
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
    std::function<void(gguf_context *)> fixture = nullptr; // a change to the fixture model
};

// an attention node by the types of the cache it reads
std::string kv_kind(const ggml_tensor * t) {
    if (t->op != GGML_OP_FLASH_ATTN_EXT) {
        return "";
    }
    return std::string("K ") + ggml_type_name(t->src[1]->type) + ", V " + ggml_type_name(t->src[2]->type);
}

// an attention node on a cache that is not F16, and whether it is a low-mean decode step
std::string kv1b_kind(const ggml_tensor * t) {
    if (t->op != GGML_OP_FLASH_ATTN_EXT || t->src[1]->type == GGML_TYPE_F16) {
        return "";
    }
    return kv_kind(t) + (ggml_flash_attn_ext_has_low_mean_mask(t) ? ", low-mean decode" : "");
}

// the served attention of one card of four: 12 query heads of 128 on 2 KV heads
void served_heads(gguf_context * gguf) {
    const auto set = [&](llm_kv key, uint32_t value) {
        gguf_set_val_u32(gguf, llama_arch_kv_name(LLM_ARCH_INKLING2, key).c_str(), value);
    };
    set(LLM_KV_ATTENTION_HEAD_COUNT, 12);
    set(LLM_KV_ATTENTION_HEAD_COUNT_KV, 2);
    set(LLM_KV_ATTENTION_KEY_LENGTH, 128);
    set(LLM_KV_ATTENTION_VALUE_LENGTH, 128);
}

// an attention node given the runs of cells its rows read (ggml_flash_attn_ext_set_kv_runs)
std::string runs_kind(const ggml_tensor * t) {
    return t->op == GGML_OP_FLASH_ATTN_EXT && ggml_flash_attn_ext_get_kv_runs(t) ? "attention by runs" : "";
}

std::string swa_ring_kind(const ggml_tensor * t) {
    return t->op == GGML_OP_FLASH_ATTN_EXT && ggml_flash_attn_ext_get_kv_runs(t) &&
        ggml_flash_attn_ext_is_swa_runs(t) ? "swa ring runs" : "";
}

std::string kv_mixed_kind(const ggml_tensor * t) {
    return t->op == GGML_OP_FLASH_ATTN_EXT && ggml_flash_attn_ext_get_hot_kv(t) ? "hot F16 mixed KV" : "";
}

std::function<void(llama_context_params &, bool)> kv_mixed_on(const char * hot_cells, bool use_runs) {
    return [=](llama_context_params & params, bool legacy) {
        params.attn_mask_bits = true;
        params.type_k_swa = params.type_v_swa = GGML_TYPE_F16;
        if (use_runs) {
            setenv("LLAMA_INKLING2_KV_RUNS", "1", 1);
        } else {
            unsetenv("LLAMA_INKLING2_KV_RUNS");
        }
        if (legacy) {
            params.type_k = params.type_v = GGML_TYPE_F16;
            unsetenv("LLAMA_INKLING2_KV_MIXED");
            unsetenv("LLAMA_INKLING2_KV_MIXED_HOT");
        } else {
            params.type_k = params.type_v = GGML_TYPE_Q4_0;
            setenv("LLAMA_INKLING2_KV_MIXED", "1", 1);
            setenv("LLAMA_INKLING2_KV_MIXED_HOT", hot_cells, 1);
        }
    };
}

void swa_ring_on(llama_context_params & params, bool legacy) {
    params.attn_mask_bits = true;
    params.type_k = params.type_v = GGML_TYPE_F16;
    params.type_k_swa = params.type_v_swa = GGML_TYPE_Q8_0;
    setenv("LLAMA_INKLING2_KV_RUNS", "1", 1);
    if (legacy) {
        setenv("LLAMA_INKLING2_SWA_RUNS_LEGACY", "1", 1);
    } else {
        unsetenv("LLAMA_INKLING2_SWA_RUNS_LEGACY");
    }
}

// the global decode by runs of cells (LLAMA_INKLING2_KV_RUNS, read as each graph is built) with packed masks
void kv_runs_on(llama_context_params & params, bool legacy) {
    params.attn_mask_bits = true;
    if (legacy) {
        unsetenv("LLAMA_INKLING2_KV_RUNS");
    } else {
        setenv("LLAMA_INKLING2_KV_RUNS", "1", 1);
    }
}

// the same on a global cache of one byte a value (the tests set GGML_CUDA_FATTN_REL_DECODE_ONE_BYTE), sliding layers F16,
// against the mask path on the same caches
std::function<void(llama_context_params &, bool)> kv_runs_one_byte(ggml_type global) {
    return [=](llama_context_params & params, bool legacy) {
        kv_runs_on(params, legacy);
        params.type_k = params.type_v = global;
        params.type_k_swa = params.type_v_swa = GGML_TYPE_F16;
    };
}

// keys and values of the global layers in global, of the sliding layers in swa, against F16
std::function<void(llama_context_params &, bool)> kv_types(ggml_type global, ggml_type swa) {
    return [=](llama_context_params & params, bool legacy) {
        params.type_k = params.type_v = legacy ? GGML_TYPE_F16 : global;
        params.type_k_swa = params.type_v_swa = legacy ? GGML_TYPE_F16 : swa;
    };
}

// keys and values of the global layers at one byte a value and of the sliding layers in F16, with packed masks
std::function<void(llama_context_params &, bool)> kv1b_types(ggml_type global) {
    return [=](llama_context_params & params, bool legacy) {
        kv_types(global, GGML_TYPE_F16)(params, legacy);
        params.attn_mask_bits = true;
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
        {"kv1b-q8_0", {nullptr, {"K q8_0, V q8_0", "K q8_0, V q8_0, low-mean decode"}, kv1b_kind,
            kv1b_types(GGML_TYPE_Q8_0), tolerance, 0.0, served_heads}},
        {"kv1b-e4m3", {nullptr, {"K f8_e4m3, V f8_e4m3", "K f8_e4m3, V f8_e4m3, low-mean decode"}, kv1b_kind,
            kv1b_types(GGML_TYPE_F8_E4M3), tolerance, 0.0, served_heads}},
        {"far", {nullptr, {"attention by runs"}, runs_kind, kv_runs_on, tolerance, 0.0, served_heads}},
        {"far-q8_0", {nullptr, {"attention by runs"}, runs_kind, kv_runs_one_byte(GGML_TYPE_Q8_0), tolerance, 0.0, served_heads}},
        {"far-e4m3", {nullptr, {"attention by runs"}, runs_kind, kv_runs_one_byte(GGML_TYPE_F8_E4M3), tolerance, 0.0, served_heads}},
        {"ring", {nullptr, {"swa ring runs"}, swa_ring_kind, swa_ring_on, tolerance, 0.0, served_heads}},
        {"kv-mixed-near", {nullptr, {"hot F16 mixed KV"}, kv_mixed_kind, kv_mixed_on("2048", false), tolerance, tolerance}},
        {"kv-mixed", {nullptr, {"hot F16 mixed KV"}, kv_mixed_kind, kv_mixed_on("1024", true), q4_0_rounding, q4_0_on_host, served_heads}},
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
    const auto served = [=](llama_context * ctx, std::vector<float> & out, bool restore) {
        constexpr int n_seq = 23;
        decode(ctx, span(tokens(1800, 50), 0, 0), out, std::vector<size_t>{1799});
        for (int s = 1; s < n_seq; ++s) {
            llama_memory_seq_cp(llama_get_memory(ctx), 0, s, -1, -1);
        }
        for (int s = 0; s < n_seq; ++s) {
            decode(ctx, span(tokens(400, 60 + s), 1800, s), out, std::vector<size_t>{399});
        }
        if (restore) {
            const size_t size = llama_state_seq_get_size(ctx, 7);
            std::vector<uint8_t> saved(size);
            if (llama_state_seq_get_data(ctx, saved.data(), size, 7) != size ||
                    !llama_memory_seq_rm(llama_get_memory(ctx), 7, -1, -1) ||
                    llama_state_seq_set_data(ctx, saved.data(), size, 7) != size) {
                throw std::runtime_error("inkling2: could not save and restore sequence 7");
            }
        }
        for (int32_t step = 0; step < 4; ++step) {
            std::vector<std::array<int32_t, 3>> rows;
            for (int s = 0; s < n_seq; ++s) {
                rows.push_back({tokens(4, 90 + s)[step], 2200 + step, s});
            }
            decode(ctx, rows, out);
        }
    };
    ret["served"] = {{23, true}, [=](llama_context * ctx, std::vector<float> & out) { served(ctx, out, false); }};
    // 23 sequences that share a prompt of 1,500 tokens (a prefix copy), then read 300 tokens each in chunks of 64 taken in
    // turn, sequence 22 only 40 (its 1,024 near cells reach into the prompt), then 6 steps of a token a sequence (one cell
    // a sequence in turn), sequence 5 cleared and read again alone with 200 tokens of its own, and 3 more steps
    ret["runs"] = {{23, true}, [=](llama_context * ctx, std::vector<float> & out) {
        constexpr int n_seq = 23, prompt = 1500;
        decode(ctx, span(tokens(prompt, 70), 0, 0), out, std::vector<size_t>{prompt - 1});
        for (int s = 1; s < n_seq; ++s) {
            llama_memory_seq_cp(llama_get_memory(ctx), 0, s, -1, -1);
        }
        std::vector<int32_t> next(n_seq, prompt);
        for (int32_t c = 0; c < 300; c += 64) {
            std::vector<std::array<int32_t, 3>> rows;
            for (int s = 0; s < n_seq; ++s) {
                const int32_t n = std::min<int32_t>(64, (s == n_seq - 1 ? 40 : 300) - c);
                if (n > 0) {
                    const auto r = span(tokens(n, 100 + 32*c + s), next[s], s);
                    rows.insert(rows.end(), r.begin(), r.end());
                    next[s] += n;
                }
            }
            decode(ctx, rows, out, std::vector<size_t>{rows.size() - 1});
        }
        const auto steps = [&](int32_t n, size_t salt) {
            for (int32_t i = 0; i < n; ++i) {
                std::vector<std::array<int32_t, 3>> rows;
                for (int s = 0; s < n_seq; ++s) {
                    rows.push_back({tokens(1, salt + n_seq*i + s)[0], next[s]++, s});
                }
                decode(ctx, rows, out);
            }
        };
        steps(6, 2000);
        if (!llama_memory_seq_rm(llama_get_memory(ctx), 5, -1, -1)) {
            throw std::runtime_error("inkling2: could not clear sequence 5");
        }
        decode(ctx, span(tokens(200, 90), 0, 5), out, std::vector<size_t>{199});
        next[5] = 200;
        steps(3, 3000);
    }};
    ret["restore"] = {{23, true}, [=](llama_context * ctx, std::vector<float> & out) { served(ctx, out, true); }};
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
    if (fit->second.fixture) {
        fit->second.fixture(gguf.get());
    }
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

// The products that read a layer's Q input (q, rel, sink and gate, and k on the sliding layers) as one launch of the CUDA
// few-row MMQ group against one launch each (GGML_CUDA_MMQ_FEW_GROUP_LEGACY=1), on one device or under a tensor split
// over several. The group takes K-quant weights whose rows hold 8 blocks of 256 or more at 3 to 24 rows, so the fixture
// is widened to n_embd 2048 in heads of 128, the five weights are Q3_K and attn_v Q4_K as in the served pack, and the
// rows come in ubatches of 23, the served decode's, and a last one of 5. The logits must be the same bit for bit.
int test_inkling_stack(size_t seed, const std::vector<ggml_backend_dev_t> & devs) {
    constexpr uint32_t n_embd = 2048, n_head = 16, n_head_kv = 8, head_dim = 128;
    auto gguf = get_gguf_ctx(LLM_ARCH_INKLING2, true);
    kv_writer ms(LLM_ARCH_INKLING2, gguf.get());
    ms.add_kv(LLM_KV_EMBEDDING_LENGTH,        n_embd);
    ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT,    n_head);
    ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV, n_head_kv);
    ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH,    head_dim);
    ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH,  head_dim);
    const uint32_t n_layer = gguf_get_val_u32(gguf.get(),
            gguf_find_key(gguf.get(), llama_arch_kv_name(LLM_ARCH_INKLING2, LLM_KV_BLOCK_COUNT).c_str()));
    for (uint32_t il = 0; il < n_layer; ++il) {
        for (const char * name : {"attn_q", "attn_k", "attn_rel", "attn_sink_proj", "attn_gate", "attn_v"}) {
            ggml_tensor t;
            memset(&t, 0, sizeof(t));
            t.type = strcmp(name, "attn_v") == 0 ? GGML_TYPE_Q4_K : GGML_TYPE_Q3_K;
            ggml_format_name(&t, "blk.%u.%s.weight", il, name);
            gguf_add_tensor(gguf.get(), &t);
        }
    }
    llama_model_ptr model = get_model(gguf.get(), nullptr, seed, 0.0f, devs,
            devs.size() > 1 ? LLAMA_SPLIT_MODE_TENSOR : LLAMA_SPLIT_MODE_LAYER);

    const auto run = [&](bool legacy, std::vector<float> & out) {
        set_env("GGML_CUDA_MMQ_FEW_GROUP_LEGACY", legacy ? "1" : nullptr); // read at each evaluation
        llama_context_params params = llama_context_default_params();
        params.n_ctx = 256;
        params.n_batch = 256;
        params.n_ubatch = 23;
        params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        params.n_threads = params.n_threads_batch = 4;
        llama_context_ptr ctx(llama_init_from_model(model.get(), params));
        if (!ctx) {
            throw std::runtime_error("inkling2 stack: the context was not created");
        }
        decode(ctx.get(), span(get_tokens(97, 128, seed), 0, 0), out);
        set_env("GGML_CUDA_MMQ_FEW_GROUP_LEGACY", nullptr);
    };
    std::vector<float> expected, actual;
    run(true, expected);
    run(false, actual);

    const bool ok = !expected.empty() && expected.size() == actual.size() &&
        memcmp(expected.data(), actual.data(), expected.size()*sizeof(float)) == 0;
    fprintf(stderr, "inkling2 stack on %zu device(s): %zu logits, %s bit for bit (NMSE %g)\n", devs.size(), expected.size(),
            ok ? "the same" : "not the same", expected.size() == actual.size() ? nmse(expected, actual) : INFINITY);
    fprintf(stderr, "%s inkling2 stack\n", ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}

namespace {

struct delta_obs {
    struct cache_obs {
        int64_t n_kv = 0;
        int64_t pos_planes = 0;
        int64_t mask_words = 0;
        int64_t mask_rows = 0;
        int64_t runs_entries = 0;
        int64_t runs_rows_p1 = 0;
        bool saw_resident_pos = false;
        bool saw_resident_mask = false;
        bool saw_resident_runs = false;
    };

    const llama_kv_cache * kv[2] = {};
    cache_obs obs[2] = {};

    void reset_step() {
        obs[0] = {};
        obs[1] = {};
    }
};

bool observe_delta(ggml_tensor * t, bool ask, void * user_data) {
    if (!ask || t->op != GGML_OP_FLASH_ATTN_EXT) {
        return false;
    }
    auto * d = static_cast<delta_obs *>(user_data);
    for (int i = 0; i < 2; ++i) {
        if (!d->kv[i]) {
            continue;
        }
        const ggml_tensor * pos  = t->src[8];
        const ggml_tensor * mask = t->src[3];
        const ggml_tensor * runs = ggml_flash_attn_ext_get_kv_runs(t);
        const bool is_swa = i == 1;
        // Match global vs SWA layer via runs flag when present, else via resident tensor identity or extent.
        bool match = false;
        if (pos && pos->view_src && pos->view_src == d->kv[i]->get_resident_cell_pos()) {
            match = true;
        } else if (mask && d->kv[i]->is_resident_kq_mask(mask)) {
            match = true;
        } else if (runs && d->kv[i]->is_resident_kv_runs(runs)) {
            match = true;
        } else if (t->src[6] && ((t->src[6]->ne[0] <= 128) == is_swa)) {
            match = true;
        }
        if (!match) {
            continue;
        }
        auto & o = d->obs[i];
        if (pos) {
            o.n_kv = pos->ne[0];
            o.pos_planes = pos->ne[2];
            if (pos->view_src && pos->view_src == d->kv[i]->get_resident_cell_pos()) {
                o.saw_resident_pos = true;
            }
        }
        if (mask && mask->type == GGML_TYPE_I16) {
            o.mask_words = mask->ne[0];
            o.mask_rows  = mask->ne[1];
            if (d->kv[i]->is_resident_kq_mask(mask)) {
                o.saw_resident_mask = true;
            }
        }
        if (runs) {
            o.runs_entries = runs->ne[1];
            o.runs_rows_p1 = runs->ne[2];
            if (d->kv[i]->is_resident_kv_runs(runs)) {
                o.saw_resident_runs = true;
            }
        }
    }
    return false;
}

bool verify_resident_buffers(llama_context * ctx, const delta_obs & d,
        const std::vector<std::array<int32_t, 3>> & last_rows, const char * tag) {
    auto * mem = dynamic_cast<llama_memory_hybrid_iswa *>(llama_get_memory(ctx));
    if (!mem) {
        fprintf(stderr, "FAIL delta %s: missing hybrid_iswa memory\n", tag);
        return false;
    }
    const auto & hparams = static_cast<const llama_model *>(llama_get_model(ctx))->hparams;
    llama_kv_cache * kvs[2] = { mem->get_mem_attn()->get_base(), mem->get_mem_attn()->get_swa() };

    // Build a synthetic ubatch describing last_rows for host ground-truth mask & kv_runs checks.
    const uint32_t nt = (uint32_t) last_rows.size();
    std::vector<llama_token> tok(nt);
    std::vector<llama_pos> pos_vec(nt);
    std::vector<int32_t> n_seq_id(nt, 1);
    std::vector<llama_seq_id> seq_storage(nt);
    std::vector<llama_seq_id *> seq_id_ptrs(nt);
    std::vector<llama_seq_id> seq_unq;
    std::vector<int32_t> seq_idx(LLAMA_MAX_SEQ, -1);
    for (uint32_t i = 0; i < nt; ++i) {
        tok[i] = last_rows[i][0];
        pos_vec[i] = last_rows[i][1];
        seq_storage[i] = last_rows[i][2];
        seq_id_ptrs[i] = &seq_storage[i];
        if (seq_idx[seq_storage[i]] < 0) {
            seq_idx[seq_storage[i]] = (int32_t) seq_unq.size();
            seq_unq.push_back(seq_storage[i]);
        }
    }
    llama_ubatch ub = {};
    ub.n_tokens = nt;
    ub.n_seqs = (uint32_t) seq_unq.size();
    ub.n_seqs_unq = (uint32_t) seq_unq.size();
    ub.token = tok.data();
    ub.pos = pos_vec.data();
    ub.n_seq_id = n_seq_id.data();
    ub.seq_id = seq_id_ptrs.data();
    ub.seq_id_unq = seq_unq.data();
    ub.seq_idx = seq_idx.data();

    for (int i = 0; i < 2; ++i) {
        const auto * kv = kvs[i];
        const auto & o  = d.obs[i];
        const auto & cells = kv->get_cells(0);
        const char * cname = i == 0 ? "global" : "swa";

        // 1. Verify kv_resident_cell_pos (plane 0 + plane 1 when kv_mixed).
        if (o.saw_resident_pos && o.n_kv > 0) {
            ggml_tensor * cp = kv->get_resident_cell_pos();
            std::vector<int32_t> gpu_p0((size_t) o.n_kv);
            ggml_backend_tensor_get(cp, gpu_p0.data(), 0, (size_t) o.n_kv * sizeof(int32_t));
            const auto & host_pos = cells.positions();
            for (int64_t c = 0; c < o.n_kv; ++c) {
                if (gpu_p0[(size_t) c] != host_pos[(size_t) c]) {
                    fprintf(stderr, "FAIL delta %s [%s]: cell_pos p0 mismatch at cell %lld (gpu=%d host=%d)\n",
                            tag, cname, (long long) c, gpu_p0[(size_t) c], host_pos[(size_t) c]);
                    return false;
                }
            }
            if (o.pos_planes == 2 && kv->has_kv_mixed()) {
                std::vector<int32_t> gpu_p1((size_t) o.n_kv);
                ggml_backend_tensor_get(cp, gpu_p1.data(), cp->nb[2], (size_t) o.n_kv * sizeof(int32_t));
                ggml_context_ptr ref_ctx(ggml_init({ ggml_tensor_overhead(), nullptr, true }));
                ggml_tensor * ref_t = ggml_new_tensor_3d(ref_ctx.get(), GGML_TYPE_I32, o.n_kv, 1, 2);
                ggml_backend_buffer_ptr ref_buf(ggml_backend_alloc_ctx_tensors_from_buft(ref_ctx.get(), ggml_backend_cpu_buffer_type()));
                kv->set_input_cell_pos(ref_t, &ub);
                const int32_t * host_p1 = (const int32_t *) ((const char *) ref_t->data + ref_t->nb[2]);
                for (int64_t c = 0; c < o.n_kv; ++c) {
                    if (gpu_p1[(size_t) c] != host_p1[c]) {
                        fprintf(stderr, "FAIL delta %s [%s]: cell_pos p1 (hot_map) mismatch at cell %lld (gpu=%d host=%d)\n",
                                tag, cname, (long long) c, gpu_p1[(size_t) c], host_p1[c]);
                        return false;
                    }
                }
            }
        }

        // 2. Verify kv_resident_kq_mask against host llama_kv_cache_set_input_kq_mask.
        if (o.saw_resident_mask && o.mask_words > 0 && o.mask_rows == (int64_t) nt) {
            ggml_tensor * km = kv->get_resident_kq_mask(o.mask_words * 16, o.mask_rows);
            const size_t nbytes = (size_t) (o.mask_words * o.mask_rows) * sizeof(uint16_t);
            std::vector<uint16_t> gpu_mask((size_t) (o.mask_words * o.mask_rows));
            ggml_backend_tensor_get(km, gpu_mask.data(), 0, nbytes);

            std::vector<uint16_t> host_mask((size_t) (o.mask_words * o.mask_rows), 0);
            ggml_tensor host_t = {};
            host_t.type = GGML_TYPE_I16;
            host_t.ne[0] = o.mask_words;
            host_t.ne[1] = o.mask_rows;
            host_t.ne[2] = 1;
            host_t.ne[3] = 1;
            host_t.nb[0] = sizeof(uint16_t);
            host_t.nb[1] = (size_t) o.mask_words * sizeof(uint16_t);
            host_t.nb[2] = host_t.nb[1] * (size_t) o.mask_rows;
            host_t.nb[3] = host_t.nb[2];
            host_t.data  = host_mask.data();

            std::vector<llama_kv_cells> v_cells_ref(1);
            // llama_kv_cache_set_input_kq_mask only reads cells const; pass via a lightweight copy of cells.
            v_cells_ref[0] = cells;
            std::vector<uint32_t> s2s(LLAMA_MAX_SEQ, 0);
            const uint32_t n_swa_cur = i == 1 ? hparams.n_swa : 0;
            const llama_swa_type swa_cur = i == 1 ? hparams.swa_type : LLAMA_SWA_TYPE_NONE;
            llama_kv_cache_set_input_kq_mask(&host_t, &ub, true, hparams, v_cells_ref, s2s,
                    n_swa_cur, swa_cur, true, false, nullptr, false, false);
            if (memcmp(gpu_mask.data(), host_mask.data(), nbytes) != 0) {
                for (size_t idx = 0; idx < gpu_mask.size(); ++idx) {
                    if (gpu_mask[idx] != host_mask[idx]) {
                        fprintf(stderr, "FAIL delta %s [%s]: kq_mask mismatch at row %zu word %zu (gpu=0x%04x host=0x%04x)\n",
                                tag, cname, idx / (size_t) o.mask_words, idx % (size_t) o.mask_words,
                                gpu_mask[idx], host_mask[idx]);
                        break;
                    }
                }
                return false;
            }
        }

        // 3. Verify kv_resident_kv_runs against full legacy write_kv_runs_host() on the cache's kv_runs.
        if (o.saw_resident_runs && o.runs_entries >= 2 && o.runs_rows_p1 == (int64_t) nt + 1) {
            ggml_tensor * kr = kv->get_resident_kv_runs(o.runs_entries, o.runs_rows_p1);
            const size_t nelems = (size_t) (4 * o.runs_entries * o.runs_rows_p1);
            std::vector<int32_t> gpu_runs(nelems);
            ggml_backend_tensor_get(kr, gpu_runs.data(), 0, nelems * sizeof(int32_t));

            std::vector<int32_t> host_runs(nelems, 0);
            kv->write_kv_runs_host(ub, host_runs.data(), o.runs_entries);
            if (memcmp(gpu_runs.data(), host_runs.data(), nelems * sizeof(int32_t)) != 0) {
                for (size_t idx = 0; idx < nelems; ++idx) {
                    if (gpu_runs[idx] != host_runs[idx]) {
                        fprintf(stderr, "FAIL delta %s [%s]: kv_runs mismatch at elem %zu (slice %zu entry %zu lane %zu: gpu=%d host=%d)\n",
                                tag, cname, idx, idx / (size_t) (4 * o.runs_entries),
                                (idx / 4) % (size_t) o.runs_entries, idx % 4, gpu_runs[idx], host_runs[idx]);
                        break;
                    }
                }
                return false;
            }
        }
    }
    return true;
}

} // namespace

int test_inkling_delta_upload(size_t seed, const std::vector<ggml_backend_dev_t> & devs) {
    // 16 query heads, 8 KV heads (head_dim 128, gqa 2): splits cleanly across 1, 2, and 4 ranks.
    auto make_fixture = [&]() {
        auto gguf = get_gguf_ctx(LLM_ARCH_INKLING2, true);
        kv_writer ms(LLM_ARCH_INKLING2, gguf.get());
        ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT, 16u);
        ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV, 8u);
        ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH, 128u);
        ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH, 128u);
        ms.add_kv(LLM_KV_ATTENTION_RELATIVE_EXTENT, 1024u);
        ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW, 64u);
        return gguf;
    };

    const bool is_gpu = std::none_of(devs.begin(), devs.end(), [](ggml_backend_dev_t d) {
        return ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_CPU;
    });
    auto gguf = make_fixture();
    llama_model_ptr model = get_model(gguf.get(), nullptr, seed, 0.1f, devs,
            devs.size() > 1 ? LLAMA_SPLIT_MODE_TENSOR : LLAMA_SPLIT_MODE_LAYER);

    struct scenario_cfg {
        const char * name;
        bool kv_mixed;
        bool kv_runs;
    };
    std::vector<scenario_cfg> scenarios = {
        { "unified-runs", false, true },
    };
    if (devs.size() == 1 && is_gpu) {
        scenarios.push_back({ "kv-mixed", true, true });
    }

    bool all_ok = true;
    const auto tokens = [seed](uint32_t n, size_t salt) { return get_tokens(n, 128, seed + salt); };

    for (const auto & sc : scenarios) {
        llama_kv_cache::llama_kv_delta_stats leg_steady[2] = {};
        llama_kv_cache::llama_kv_delta_stats opt_steady[2] = {};
        bool saw_res_pos[2] = { false, false };
        bool saw_res_mask[2] = { false, false };
        bool saw_res_runs[2] = { false, false };

        auto run_mode = [&](bool delta_legacy, bool batch_legacy, std::vector<float> & logits_out) -> bool {
            set_env("LLAMA_INKLING2_DELTA_INPUTS_LEGACY",  delta_legacy ? "1" : nullptr);
            set_env("GGML_SCHED_BATCH_INPUTS_LEGACY",      batch_legacy ? "1" : nullptr);
            const bool legacy = delta_legacy;
            set_env("LLAMA_INKLING2_KV_RUNS",             sc.kv_runs ? "1" : nullptr);
            set_env("GGML_CUDA_FATTN_REL_DECODE_ONE_BYTE", sc.kv_runs ? "1" : nullptr);
            set_env("GGML_CUDA_FATTN_REL_DECODE_Q4",      (sc.kv_runs && sc.kv_mixed) ? "1" : nullptr);
            set_env("LLAMA_KV_PAD_SHARED",                "8192");
            set_env("LLAMA_INKLING2_KV_MIXED",            sc.kv_mixed ? "1" : nullptr);
            set_env("LLAMA_INKLING2_KV_MIXED_HOT",        sc.kv_mixed ? "64" : nullptr);
            set_env("LLAMA_GRAPH_SLOTS",                  "8");
            set_env("LLAMA_RS_FIXED_ROWS",                "1");
            set_env("LLAMA_GRAPH_ANY_SEQS",               "1");

            delta_obs obs;
            llama_context_params params = llama_context_default_params();
            params.n_ctx = 16384;
            params.n_batch = 1024;
            params.n_ubatch = 256;
            params.n_seq_max = 12;
            params.n_rs_seq = 4;
            params.kv_unified = true;
            params.attn_mask_bits = true;
            params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
            params.type_k = params.type_v = sc.kv_mixed ? GGML_TYPE_Q4_0 : GGML_TYPE_F16;
            params.type_k_swa = params.type_v_swa = sc.kv_mixed ? GGML_TYPE_F16 : GGML_TYPE_Q8_0;
            params.n_threads = params.n_threads_batch = 4;
            params.cb_eval = observe_delta;
            params.cb_eval_user_data = &obs;

            llama_context_ptr ctx(llama_init_from_model(model.get(), params));
            if (!ctx) {
                fprintf(stderr, "FAIL delta %s: context creation failed\n", sc.name);
                return false;
            }
            auto * mem = dynamic_cast<llama_memory_hybrid_iswa *>(llama_get_memory(ctx.get()));
            llama_kv_cache * kvs[2] = { mem->get_mem_attn()->get_base(), mem->get_mem_attn()->get_swa() };
            obs.kv[0] = kvs[0];
            obs.kv[1] = kvs[1];

            auto step = [&](const std::vector<std::array<int32_t, 3>> & rows, const char * tag,
                    bool check_bufs, const std::optional<std::vector<size_t>> & out_rows = std::nullopt) -> bool {
                obs.reset_step();
                decode(ctx.get(), rows, logits_out, out_rows);
                if (!legacy && is_gpu) {
                    for (int i = 0; i < 2; ++i) {
                        saw_res_pos[i]  |= obs.obs[i].saw_resident_pos;
                        saw_res_mask[i] |= obs.obs[i].saw_resident_mask;
                        saw_res_runs[i] |= obs.obs[i].saw_resident_runs;
                    }
                    if (check_bufs && !verify_resident_buffers(ctx.get(), obs, rows, tag)) {
                        return false;
                    }
                }
                return true;
            };

            // Phase 1: Shared-prefix prefill (> n_ubatch triggers multi-ubatch prepare()) + seq_cp.
            // Fill > 4096 total cells so n_kv pads to 8192 and exercises global + SWA kv_runs too.
            const uint32_t prefix_len = 512;
            if (!step(span(tokens(prefix_len, 1), 0, 0), "prefix", false, std::vector<size_t>{ prefix_len - 1 })) {
                return false;
            }
            const int n_seqs = 10;
            for (int s = 1; s < n_seqs; ++s) {
                llama_memory_seq_cp(mem, 0, s, 0, prefix_len);
            }
            std::vector<int32_t> next_pos(n_seqs, (int32_t) prefix_len);
            for (int s = 0; s < n_seqs; ++s) {
                const uint32_t slen = 360 + 8 * s;
                if (!step(span(tokens(slen, 10 + s), next_pos[s], s), "suffix", false, std::vector<size_t>{ slen - 1 })) {
                    return false;
                }
                next_pos[s] += (int32_t) slen;
            }

            // Phase 2: Warmup + 6 steady-state multi-sequence decode steps (exercises kv_runs + delta pos/mask).
            llama_kv_cache::llama_kv_delta_stats acc_steady[2] = {};
            auto add_stats = [](llama_kv_cache::llama_kv_delta_stats::kind_t & dst,
                    const llama_kv_cache::llama_kv_delta_stats::kind_t & src) {
                dst.calls         += src.calls;
                dst.full_uploads  += src.full_uploads;
                dst.delta_uploads += src.delta_uploads;
                dst.spans         += src.spans;
                dst.bytes         += src.bytes;
            };
            for (int st = 0; st < 8; ++st) {
                std::vector<std::array<int32_t, 3>> rows;
                for (int s = 0; s < n_seqs; ++s) {
                    rows.push_back({ tokens(1, 100 + 16 * st + s)[0], next_pos[s]++, s });
                }
                if (st >= 2) {
                    kvs[0]->reset_delta_stats();
                    kvs[1]->reset_delta_stats();
                }
                obs.reset_step();
                decode(ctx.get(), rows, logits_out);
                if (st >= 2) {
                    for (int i = 0; i < 2; ++i) {
                        const auto s = kvs[i]->get_delta_stats();
                        add_stats(acc_steady[i].cell_pos, s.cell_pos);
                        add_stats(acc_steady[i].kq_mask,  s.kq_mask);
                        add_stats(acc_steady[i].kv_runs,  s.kv_runs);
                    }
                }
                if (!legacy && is_gpu) {
                    for (int i = 0; i < 2; ++i) {
                        saw_res_pos[i]  |= obs.obs[i].saw_resident_pos;
                        saw_res_mask[i] |= obs.obs[i].saw_resident_mask;
                        saw_res_runs[i] |= obs.obs[i].saw_resident_runs;
                    }
                    if (!verify_resident_buffers(ctx.get(), obs, rows, "steady-decode")) {
                        return false;
                    }
                }
            }
            for (int i = 0; i < 2; ++i) {
                (legacy ? leg_steady[i] : opt_steady[i]) = acc_steady[i];
            }

            // Phase 3: Interleaved 1-row then 10-row decode (verifies inactive-row preservation).
            if (!step({{ tokens(1, 300)[0], next_pos[0]++, 0 }}, "single-row", true)) {
                return false;
            }
            {
                std::vector<std::array<int32_t, 3>> rows;
                for (int s = 0; s < n_seqs; ++s) {
                    rows.push_back({ tokens(1, 310 + s)[0], next_pos[s]++, s });
                }
                if (!step(rows, "resume-10-row", true)) {
                    return false;
                }
            }

            // Phase 4: Drafted / non-tip multi-token step (4 seqs x 4 tokens), rollback last 2 on seq 1, then tip decode.
            {
                std::vector<std::array<int32_t, 3>> rows;
                for (int s = 0; s < 4; ++s) {
                    const auto r = span(tokens(4, 340 + s), next_pos[s], s);
                    rows.insert(rows.end(), r.begin(), r.end());
                    next_pos[s] += 4;
                }
                if (!step(rows, "drafted", true)) {
                    return false;
                }
            }
            if (!llama_memory_seq_rm(mem, 1, next_pos[1] - 2, -1)) {
                fprintf(stderr, "FAIL delta %s: draft rollback seq_rm refused\n", sc.name);
                return false;
            }
            next_pos[1] -= 2;
            {
                std::vector<std::array<int32_t, 3>> rows;
                for (int s = 0; s < n_seqs; ++s) {
                    rows.push_back({ tokens(1, 360 + s)[0], next_pos[s]++, s });
                }
                if (!step(rows, "post-draft-tip", true)) {
                    return false;
                }
            }

            // Phase 5: 3-token draft on seq 1 + 1-token rollback + full seq removal & reuse + seq_add (when supported).
            {
                const auto r = span(tokens(3, 390), next_pos[1], 1);
                next_pos[1] += 3;
                if (!step(r, "draft-seq1", true)) {
                    return false;
                }
            }
            if (!llama_memory_seq_rm(mem, 1, next_pos[1] - 1, -1)) {
                fprintf(stderr, "FAIL delta %s: single-step rollback seq_rm refused\n", sc.name);
                return false;
            }
            next_pos[1] -= 1;
            llama_memory_seq_rm(mem, n_seqs - 1, -1, -1);
            next_pos[n_seqs - 1] = 0;
            if (!step(span(tokens(20, 400), 0, n_seqs - 1), "fresh-slot", false, std::vector<size_t>{ 19 })) {
                return false;
            }
            next_pos[n_seqs - 1] = 20;
            if (llama_memory_can_shift(mem)) {
                llama_memory_seq_add(mem, 2, 100, -1, 4);
                next_pos[2] += 4;
            }
            {
                std::vector<std::array<int32_t, 3>> rows;
                for (int s = 0; s < n_seqs; ++s) {
                    rows.push_back({ tokens(1, 420 + s)[0], next_pos[s]++, s });
                }
                if (!step(rows, "post-rm-add", true)) {
                    return false;
                }
            }

            // Phase 6: Per-sequence state save/restore + seq_keep + full state save/clear/restore.
            std::vector<uint8_t> seq_blob(llama_state_seq_get_size(ctx.get(), 0));
            const size_t seq_written = llama_state_seq_get_data(ctx.get(), seq_blob.data(), seq_blob.size(), 0);
            if (seq_written == 0) {
                fprintf(stderr, "FAIL delta %s: seq state save failed\n", sc.name);
                return false;
            }
            llama_memory_seq_keep(mem, 1);
            if (!step({{ tokens(1, 450)[0], next_pos[1]++, 1 }}, "post-seq-keep", true)) {
                return false;
            }
            if (llama_state_seq_set_data(ctx.get(), seq_blob.data(), seq_written, 3) != seq_written) {
                fprintf(stderr, "FAIL delta %s: seq state restore failed\n", sc.name);
                return false;
            }
            next_pos[3] = next_pos[0];
            if (!step({{ tokens(1, 460)[0], next_pos[1]++, 1 }, { tokens(1, 461)[0], next_pos[3]++, 3 }}, "post-seq-restore", true)) {
                return false;
            }

            std::vector<uint8_t> full_blob(llama_state_get_size(ctx.get()));
            const size_t full_written = llama_state_get_data(ctx.get(), full_blob.data(), full_blob.size());
            if (full_written == 0) {
                fprintf(stderr, "FAIL delta %s: full state save failed\n", sc.name);
                return false;
            }
            llama_memory_clear(mem, true);
            if (!step(span(tokens(16, 470), 0, 0), "post-clear-small", true, std::vector<size_t>{ 15 })) {
                return false;
            }
            if (!step({{ tokens(1, 471)[0], 16, 0 }}, "post-clear-decode", true)) {
                return false;
            }
            if (llama_state_set_data(ctx.get(), full_blob.data(), full_written) != full_written) {
                fprintf(stderr, "FAIL delta %s: full state restore failed\n", sc.name);
                return false;
            }
            if (!step({{ tokens(1, 480)[0], next_pos[1]++, 1 }, { tokens(1, 481)[0], next_pos[3]++, 3 }}, "post-full-restore", true)) {
                return false;
            }

            // Phase 7: Simulate input_lost() invalidation and verify clean recovery.
            if (!legacy) {
                kvs[0]->invalidate_delta_state();
                kvs[1]->invalidate_delta_state();
            }
            if (!step({{ tokens(1, 490)[0], next_pos[1]++, 1 }, { tokens(1, 491)[0], next_pos[3]++, 3 }}, "post-invalidate", true)) {
                return false;
            }
            if (!step({{ tokens(1, 492)[0], next_pos[1]++, 1 }, { tokens(1, 493)[0], next_pos[3]++, 3 }}, "post-invalidate-delta", true)) {
                return false;
            }
            return true;
        };

        std::vector<float> expected, leg_batch, actual;
        const bool ok_leg = run_mode(true,  true,  expected);
        const bool ok_lb  = run_mode(true,  false, leg_batch);
        const bool ok_opt = run_mode(false, false, actual);

        const bool exact_batch = ok_leg && ok_lb && !expected.empty() &&
            expected.size() == leg_batch.size() &&
            memcmp(expected.data(), leg_batch.data(), expected.size() * sizeof(float)) == 0;
        const bool bit_exact = ok_leg && ok_opt && !expected.empty() &&
            expected.size() == actual.size() &&
            memcmp(expected.data(), actual.data(), expected.size() * sizeof(float)) == 0;
        const double diff_nmse = expected.size() == actual.size() ? nmse(expected, actual) : INFINITY;
        float max_diff = 0.0f;
        if (ok_leg && ok_opt && expected.size() == actual.size()) {
            for (size_t i = 0; i < expected.size(); ++i) {
                max_diff = std::max(max_diff, std::fabs(expected[i] - actual[i]));
            }
        }
        // Moving mask/cell_pos out of the compute buffer shifts gallocr offsets, which can toggle D1 fusion aliasing by 1 ULP.
        const bool exact_logits = exact_batch && (bit_exact || (ok_leg && ok_opt && !expected.empty() &&
            expected.size() == actual.size() && diff_nmse <= 1e-14 && max_diff <= 1e-6f));

        set_env("LLAMA_INKLING2_DELTA_INPUTS_LEGACY",  nullptr);
        set_env("GGML_SCHED_BATCH_INPUTS_LEGACY",      nullptr);
        set_env("LLAMA_INKLING2_KV_RUNS",              nullptr);
        set_env("GGML_CUDA_FATTN_REL_DECODE_ONE_BYTE", nullptr);
        set_env("GGML_CUDA_FATTN_REL_DECODE_Q4",       nullptr);
        set_env("LLAMA_KV_PAD_SHARED",                 nullptr);
        set_env("LLAMA_INKLING2_KV_MIXED",             nullptr);
        set_env("LLAMA_INKLING2_KV_MIXED_HOT",         nullptr);
        set_env("LLAMA_GRAPH_SLOTS",                   nullptr);
        set_env("LLAMA_RS_FIXED_ROWS",                 nullptr);
        set_env("LLAMA_GRAPH_ANY_SEQS",                nullptr);

        const uint64_t leg_bytes =
            leg_steady[0].cell_pos.bytes + leg_steady[0].kq_mask.bytes + leg_steady[0].kv_runs.bytes +
            leg_steady[1].cell_pos.bytes + leg_steady[1].kq_mask.bytes + leg_steady[1].kv_runs.bytes;
        const uint64_t opt_bytes =
            opt_steady[0].cell_pos.bytes + opt_steady[0].kq_mask.bytes + opt_steady[0].kv_runs.bytes +
            opt_steady[1].cell_pos.bytes + opt_steady[1].kq_mask.bytes + opt_steady[1].kv_runs.bytes;

        const bool resident_used = !is_gpu || (saw_res_pos[0] && saw_res_pos[1] && saw_res_mask[0] && saw_res_mask[1] &&
            (!sc.kv_runs || (saw_res_runs[0] && saw_res_runs[1])));
        const bool bytes_reduced = !is_gpu || (leg_bytes > 0 && opt_bytes > 0 && opt_bytes * 10 < leg_bytes &&
            opt_steady[0].cell_pos.full_uploads == 0 && opt_steady[1].cell_pos.full_uploads == 0 &&
            opt_steady[0].kq_mask.full_uploads  == 0 && opt_steady[1].kq_mask.full_uploads  == 0 &&
            (!sc.kv_runs || (opt_steady[0].kv_runs.full_uploads == 0 && opt_steady[1].kv_runs.full_uploads == 0)));

        fprintf(stderr, "inkling2 delta-upload [%s] on %zu rank(s): %zu logits, exact=%d (bit_exact=%d batch_exact=%d max_abs=%g NMSE %g), "
                "steady_bytes legacy=%llu delta=%llu (%.2fx), resident=%d (pos=%d/%d mask=%d/%d runs=%d/%d), zero_full=%d\n",
                sc.name, devs.size(), expected.size(), exact_logits ? 1 : 0, bit_exact ? 1 : 0, exact_batch ? 1 : 0,
                max_diff, diff_nmse,
                (unsigned long long) leg_bytes, (unsigned long long) opt_bytes,
                opt_bytes > 0 ? (double) leg_bytes / (double) opt_bytes : 0.0,
                resident_used ? 1 : 0,
                saw_res_pos[0] ? 1 : 0, saw_res_pos[1] ? 1 : 0,
                saw_res_mask[0] ? 1 : 0, saw_res_mask[1] ? 1 : 0,
                saw_res_runs[0] ? 1 : 0, saw_res_runs[1] ? 1 : 0,
                bytes_reduced ? 1 : 0);
        fprintf(stderr, "  breakdown (6 steady steps, global+swa): cell_pos %llu->%llu B, kq_mask %llu->%llu B, kv_runs %llu->%llu B\n",
                (unsigned long long) (leg_steady[0].cell_pos.bytes + leg_steady[1].cell_pos.bytes),
                (unsigned long long) (opt_steady[0].cell_pos.bytes + opt_steady[1].cell_pos.bytes),
                (unsigned long long) (leg_steady[0].kq_mask.bytes + leg_steady[1].kq_mask.bytes),
                (unsigned long long) (opt_steady[0].kq_mask.bytes + opt_steady[1].kq_mask.bytes),
                (unsigned long long) (leg_steady[0].kv_runs.bytes + leg_steady[1].kv_runs.bytes),
                (unsigned long long) (opt_steady[0].kv_runs.bytes + opt_steady[1].kv_runs.bytes));

        all_ok = all_ok && exact_logits && resident_used && bytes_reduced;
    }

    fprintf(stderr, "%s inkling2 delta-upload on %zu rank(s)\n", all_ok ? "OK" : "FAIL", devs.size());
    return all_ok ? 0 : 1;
}

int test_inkling_delta_bench(size_t seed, const std::vector<ggml_backend_dev_t> & devs) {
    // 2-layer synthetic Inkling 2 (1 global + 1 SWA) with 8 KV heads (splits cleanly across 1, 2, and 4 ranks).
    auto gguf = get_gguf_ctx(LLM_ARCH_INKLING2, true);
    kv_writer ms(LLM_ARCH_INKLING2, gguf.get());
    ms.add_kv(LLM_KV_BLOCK_COUNT, 2u);
    ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, std::vector<uint32_t>{ 0u, 1u });
    ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT, 16u);
    ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV, 8u);
    ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH, 128u);
    ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH, 128u);
    ms.add_kv(LLM_KV_ATTENTION_RELATIVE_EXTENT, 1024u);
    ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW, 1024u);

    llama_model_ptr model = get_model(gguf.get(), nullptr, seed, 0.1f, devs,
            devs.size() > 1 ? LLAMA_SPLIT_MODE_TENSOR : LLAMA_SPLIT_MODE_LAYER);

    const auto tokens = [seed](uint32_t n, size_t salt) { return get_tokens(n, 128, seed + salt); };

    struct mode_cfg {
        const char * name;
        bool delta_legacy;
        bool batch_legacy;
    };
    const mode_cfg modes[] = {
        { "legacy-full",  true,  true  },
        { "delta-only",   false, true  },
        { "delta+batch",  false, false },
    };

    // 262,144 cells (~1 MB cell_pos, ~754 KB mask) and 1,048,576 cells (~4.2 MB cell_pos, ~3.0 MB mask).
    struct pool_cfg {
        uint32_t n_ctx;
        uint32_t per_seq_fill; // > n_ctx/2 total cells so pad_dense_shared expands n_kv to n_ctx
    };
    const pool_cfg pools[] = {
        { 262144u,  5800u  },
        { 1048576u, 23000u },
    };
    for (const auto & pc : pools) {
        const uint32_t n_ctx = pc.n_ctx;
        fprintf(stderr, "\n=== Inkling 2 delta upload benchmark: pool=%u cells, 23 seqs, %zu rank(s) ===\n",
                n_ctx, devs.size());
        const std::string pad_str = std::to_string(n_ctx);
        for (const auto & m : modes) {
            set_env("LLAMA_INKLING2_DELTA_INPUTS_LEGACY",  m.delta_legacy ? "1" : nullptr);
            set_env("GGML_SCHED_BATCH_INPUTS_LEGACY",      m.batch_legacy ? "1" : nullptr);
            set_env("LLAMA_INKLING2_KV_RUNS",              "1");
            set_env("GGML_CUDA_FATTN_REL_DECODE_ONE_BYTE", "1");
            set_env("LLAMA_KV_PAD_SHARED",                 pad_str.c_str());
            set_env("LLAMA_GRAPH_SLOTS",                   "32");
            set_env("LLAMA_RS_FIXED_ROWS",                 "1");
            set_env("LLAMA_GRAPH_ANY_SEQS",                "1");

            llama_context_params params = llama_context_default_params();
            params.n_ctx = n_ctx;
            params.n_batch = 4096;
            params.n_ubatch = 2048;
            params.n_seq_max = 23;
            params.kv_unified = true;
            params.swa_full = false;
            params.attn_mask_bits = true;
            params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
            params.type_k = params.type_v = GGML_TYPE_F16;
            params.type_k_swa = params.type_v_swa = GGML_TYPE_Q8_0;
            params.n_threads = params.n_threads_batch = 3;

            llama_context_ptr ctx(llama_init_from_model(model.get(), params));
            if (!ctx) {
                fprintf(stderr, "SKIP %s at pool=%u (context alloc failed)\n", m.name, n_ctx);
                continue;
            }
            auto * mem = dynamic_cast<llama_memory_hybrid_iswa *>(llama_get_memory(ctx.get()));
            llama_kv_cache * kv0 = mem->get_mem_attn()->get_base();
            llama_kv_cache * kv1 = mem->get_mem_attn()->get_swa();

            std::vector<float> discard;
            const uint32_t prefix_len = 2048;
            decode(ctx.get(), span(tokens(prefix_len, 1), 0, 0), discard, std::vector<size_t>{});
            for (int s = 1; s < 23; ++s) {
                llama_memory_seq_cp(mem, 0, s, 0, prefix_len);
            }
            std::vector<int32_t> next_pos(23, (int32_t) prefix_len);
            for (int s = 0; s < 23; ++s) {
                uint32_t rem = pc.per_seq_fill;
                while (rem > 0) {
                    const uint32_t chunk = std::min<uint32_t>(rem, params.n_batch);
                    decode(ctx.get(), span(tokens(chunk, 20 + s * 16 + rem), next_pos[s], s), discard, std::vector<size_t>{});
                    next_pos[s] += (int32_t) chunk;
                    rem -= chunk;
                }
            }

            // Warmup 5 decode steps (23 rows each).
            for (int w = 0; w < 5; ++w) {
                std::vector<std::array<int32_t, 3>> rows;
                for (int s = 0; s < 23; ++s) {
                    rows.push_back({ tokens(1, 500 + w * 23 + s)[0], next_pos[s]++, s });
                }
                decode(ctx.get(), rows, discard, std::vector<size_t>{});
            }

            // Measure 30 steady-state 23-seq decode steps.
            kv0->reset_delta_stats();
            kv1->reset_delta_stats();
            const int n_steps = 30;
            std::vector<double> step_us;
            step_us.reserve(n_steps);
            for (int st = 0; st < n_steps; ++st) {
                std::vector<std::array<int32_t, 3>> rows;
                for (int s = 0; s < 23; ++s) {
                    rows.push_back({ tokens(1, 1000 + st * 23 + s)[0], next_pos[s]++, s });
                }
                const auto t0 = std::chrono::steady_clock::now();
                decode(ctx.get(), rows, discard, std::vector<size_t>{});
                const auto t1 = std::chrono::steady_clock::now();
                step_us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
            }
            const auto st0 = kv0->get_delta_stats();
            const auto st1 = kv1->get_delta_stats();
            std::sort(step_us.begin(), step_us.end());
            const double med_us = step_us[n_steps / 2];
            const double mean_us = std::accumulate(step_us.begin(), step_us.end(), 0.0) / n_steps;
            const double b_pos  = (double) (st0.cell_pos.bytes + st1.cell_pos.bytes) / n_steps;
            const double b_mask = (double) (st0.kq_mask.bytes  + st1.kq_mask.bytes)  / n_steps;
            const double b_runs = (double) (st0.kv_runs.bytes  + st1.kv_runs.bytes)  / n_steps;
            const double spans  = (double) (st0.cell_pos.spans + st1.cell_pos.spans +
                                            st0.kq_mask.spans  + st1.kq_mask.spans +
                                            st0.kv_runs.spans  + st1.kv_runs.spans) / n_steps;

            fprintf(stderr, "  decode23 %-12s: med=%7.1f us mean=%7.1f us | bytes/step total=%9.0f (pos=%8.0f mask=%8.0f runs=%7.0f) spans=%.1f full=%llu\n",
                    m.name, med_us, mean_us, b_pos + b_mask + b_runs, b_pos, b_mask, b_runs, spans,
                    (unsigned long long) (st0.cell_pos.full_uploads + st1.cell_pos.full_uploads +
                                          st0.kq_mask.full_uploads  + st1.kq_mask.full_uploads +
                                          st0.kv_runs.full_uploads  + st1.kv_runs.full_uploads));

            // Measure 512-token prefill steps (5 chunks on seq 0).
            kv0->reset_delta_stats();
            kv1->reset_delta_stats();
            const int n_prefill = 5;
            std::vector<double> pf_us;
            for (int p = 0; p < n_prefill; ++p) {
                const auto r = span(tokens(512, 3000 + p), next_pos[0], 0);
                next_pos[0] += 512;
                const auto t0 = std::chrono::steady_clock::now();
                decode(ctx.get(), r, discard, std::vector<size_t>{});
                const auto t1 = std::chrono::steady_clock::now();
                pf_us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
            }
            const auto p0 = kv0->get_delta_stats();
            const auto p1 = kv1->get_delta_stats();
            std::sort(pf_us.begin(), pf_us.end());
            fprintf(stderr, "  prefill512 %-10s: med=%7.1f us | pos_bytes/step=%9.0f mask_bytes/step=%9.0f full_pos=%llu\n",
                    m.name, pf_us[n_prefill / 2],
                    (double) (p0.cell_pos.bytes + p1.cell_pos.bytes) / n_prefill,
                    (double) (p0.kq_mask.bytes  + p1.kq_mask.bytes)  / n_prefill,
                    (unsigned long long) (p0.cell_pos.full_uploads + p1.cell_pos.full_uploads));
        }
    }

    set_env("LLAMA_INKLING2_DELTA_INPUTS_LEGACY",  nullptr);
    set_env("GGML_SCHED_BATCH_INPUTS_LEGACY",      nullptr);
    set_env("LLAMA_INKLING2_KV_RUNS",              nullptr);
    set_env("GGML_CUDA_FATTN_REL_DECODE_ONE_BYTE", nullptr);
    set_env("LLAMA_KV_PAD_SHARED",                 nullptr);
    set_env("LLAMA_GRAPH_SLOTS",                   nullptr);
    set_env("LLAMA_RS_FIXED_ROWS",                 nullptr);
    set_env("LLAMA_GRAPH_ANY_SEQS",                nullptr);
    return 0;
}
