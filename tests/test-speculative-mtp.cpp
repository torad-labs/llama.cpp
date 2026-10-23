// The MTP drafter driven the way the server drives it: target verification batches, process, draft, accept, begin and
// end, on a small GLM-5.3 model with its NextN head.
//
// Goal: with --spec-draft-mtp-adaptive COST each sequence drafts either one token or --spec-draft-n-max tokens a round,
// from the acceptance of its own MTP drafts in the current request. It drafts the full depth while the extra tokens the
// positions past the first are expected to land, relative to a one-token round, exceed (n_max - 1) * COST. Shapes stay
// stable: a sequence's depth holds for windows of 8 of its rounds, the first window of a request is full depth, and a
// one-token window ends in one full-depth probe. Another sequence's rounds, an earlier request on the same slot, and a
// round verified twice (a checkpoint replay) never move it. Without the flag every round drafts n_max, as before.
// The head's kept hidden row stays the target row that a verification accepted, however the batch is laid out.

#include "common.h"
#include "log.h"
#include "llama.h"
#include "llama-cpp.h"
#include "speculative.h"
#include "gguf.h"
#include "ggml-cpp.h"

#include "../src/llama-arch.h"
#include "../src/llama-ext.h"
#include "../src/llama-model-saver.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <functional>
#include <map>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

static constexpr int N_SEQ  = 3;
static constexpr int N_MAX  = 2;
static constexpr int WINDOW = 8;

static void set_tensor_data(struct ggml_tensor * tensor, void * userdata) {
    size_t seed = *(const size_t *) userdata;
    seed ^= std::hash<std::string>()(tensor->name);
    std::mt19937 gen(seed);
    std::normal_distribution<float> dis(0.0f, 1.0e-2f);
    const int64_t ne = ggml_nelements(tensor);
    if (tensor->type == GGML_TYPE_F32) {
        std::vector<float> tmp(ne);
        for (auto & v : tmp) v = dis(gen);
        ggml_backend_tensor_set(tensor, tmp.data(), 0, ggml_nbytes(tensor));
    } else if (tensor->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> tmp(ne);
        for (auto & v : tmp) v = ggml_fp32_to_fp16(dis(gen));
        ggml_backend_tensor_set(tensor, tmp.data(), 0, ggml_nbytes(tensor));
    } else {
        GGML_ABORT("fatal error");
    }
}

// the GLM-5.3 fixture of test-llama-archs: KDA and DSA layers, routed experts, and one NextN block
static gguf_context_ptr glm_gguf() {
    gguf_context_ptr ret(gguf_init_empty());
    llama_model_saver ms(LLM_ARCH_GLM5NEXT, ret.get());
    const uint32_t n_embd = 128, n_head = 4, n_ff = 192, n_layer = 3;
    ms.add_kv(LLM_KV_GENERAL_ARCHITECTURE,      llm_arch_name(LLM_ARCH_GLM5NEXT));
    ms.add_kv(LLM_KV_VOCAB_SIZE,                uint32_t(128));
    ms.add_kv(LLM_KV_CONTEXT_LENGTH,            uint32_t(128));
    ms.add_kv(LLM_KV_EMBEDDING_LENGTH,          n_embd);
    ms.add_kv(LLM_KV_FEATURES_LENGTH,           n_embd);
    ms.add_kv(LLM_KV_BLOCK_COUNT,               n_layer);
    ms.add_kv(LLM_KV_NEXTN_PREDICT_LAYERS,      uint32_t(1));
    ms.add_kv(LLM_KV_LEADING_DENSE_BLOCK_COUNT, uint32_t(1));
    ms.add_kv(LLM_KV_FEED_FORWARD_LENGTH,       n_ff);
    ms.add_kv(LLM_KV_USE_PARALLEL_RESIDUAL,     false);
    ms.add_kv(LLM_KV_LOGIT_SCALE,               1.0f);
    ms.add_kv(LLM_KV_FULL_ATTENTION_INTERVAL,   uint32_t(2));
    ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT,      n_head);
    ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV,   std::vector<uint32_t>({1, 0, 1}));
    ms.add_kv(LLM_KV_ATTENTION_MAX_ALIBI_BIAS,  8.0f);
    ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH,       uint32_t(512));
    ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH,     uint32_t(512));
    ms.add_kv(LLM_KV_ROPE_DIMENSION_COUNT,       uint32_t(0));
    ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH_MLA,   uint32_t(192));
    ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH_MLA, uint32_t(128));
    ms.add_kv(LLM_KV_ATTENTION_CLAMP_KQV,              1.0f);
    ms.add_kv(LLM_KV_ATTENTION_LAYERNORM_EPS,          1e-6f);
    ms.add_kv(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,      1e-5f);
    ms.add_kv(LLM_KV_ATTENTION_GROUPNORM_EPS,          1e-5f);
    ms.add_kv(LLM_KV_ATTENTION_GROUPNORM_GROUPS,       uint32_t(8));
    ms.add_kv(LLM_KV_ATTENTION_Q_LORA_RANK,            uint32_t(512));
    ms.add_kv(LLM_KV_ATTENTION_KV_LORA_RANK,           uint32_t(512));
    ms.add_kv(LLM_KV_ATTENTION_RELATIVE_BUCKETS_COUNT, uint32_t(8));
    ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW,         uint32_t(16));
    ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, uint32_t(2));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT,     uint32_t(1));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH,     uint32_t(64));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_TOP_K,          uint32_t(8));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_BLOCK_SIZE,     uint32_t(4));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_LOCAL_BLOCKS,   uint32_t(1));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_KPOOL,          uint32_t(4));
    ms.add_kv(LLM_KV_ROPE_DIMENSION_SECTIONS,          std::vector<uint32_t>({8, 8, 8, 8}));
    ms.add_kv(LLM_KV_HYPER_CONNECTION_COUNT,               uint32_t(4));
    ms.add_kv(LLM_KV_HYPER_CONNECTION_SINKHORN_ITERATIONS, uint32_t(2));
    ms.add_kv(LLM_KV_HYPER_CONNECTION_EPSILON,             1.0e-6f);
    ms.add_kv(LLM_KV_EXPERT_WEIGHTS_SCALE,                 1.0f);
    ms.add_kv(LLM_KV_EXPERT_WEIGHTS_NORM,                  true);
    ms.add_kv(LLM_KV_TOKENIZER_MODEL,                      "no_vocab");
    ms.add_kv(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,        n_ff);
    ms.add_kv(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, n_ff / 2);
    ms.add_kv(LLM_KV_EXPERT_LATENT_LENGTH,              n_ff);
    ms.add_kv(LLM_KV_INTERLEAVE_MOE_LAYER_STEP,         uint32_t(2));
    ms.add_kv(LLM_KV_EXPERT_COUNT,                      uint32_t(2));
    ms.add_kv(LLM_KV_EXPERT_USED_COUNT,                 uint32_t(1));
    ms.add_kv(LLM_KV_EXPERT_SHARED_COUNT,               uint32_t(1));
    ms.add_kv(LLM_KV_EXPERT_GATING_FUNC,                uint32_t(2));
    ms.add_kv(LLM_KV_EXPERT_GROUP_SCALE,                1.0f);
    ms.add_kv(LLM_KV_EXPERTS_PER_GROUP,                 uint32_t(1));
    ms.add_kv(LLM_KV_SSM_INNER_SIZE,       2*n_embd);
    ms.add_kv(LLM_KV_SSM_CONV_KERNEL,      uint32_t(4));
    ms.add_kv(LLM_KV_SSM_STATE_SIZE,       uint32_t(128));
    ms.add_kv(LLM_KV_SSM_TIME_STEP_RANK,   n_head);
    ms.add_kv(LLM_KV_SSM_GROUP_COUNT,      uint32_t(2));
    ms.add_kv(LLM_KV_KDA_HEAD_DIM,         uint32_t(128));
    ms.add_kv(LLM_KV_KDA_SAFE_GATE,        true);
    ms.add_kv(LLM_KV_KDA_GATE_LOWER_BOUND, -5.0f);
    return ret;
}

static llama_model_ptr glm_model(size_t seed) {
    auto gguf = glm_gguf();
    llama_model_params mparams = llama_model_default_params();
    mparams.progress_callback = [](float, void *) { return true; };
    mparams.load_mtp = true;
    ggml_backend_dev_t no_devs[1] = { nullptr };
    mparams.devices = no_devs;
    llama_model_ptr model(llama_model_init_from_user(gguf.get(), set_tensor_data, &seed, mparams));
    if (!model) {
        throw std::runtime_error("failed to create the GLM model");
    }
    return model;
}

// one server: the target context, the MTP head's context made by the server's own path, and the drafter
struct server_rig {
    common_params params;
    llama_context_ptr tgt;
    common_speculative_init_result_ptr init;
    common_speculative_ptr spec;
    int32_t n_vocab = 0;

    struct seq {
        llama_pos    n_past  = 0;
        llama_token  id_last = 0;
        llama_tokens prompt;
        llama_tokens draft;
    };
    std::vector<seq> seqs = std::vector<seq>(N_SEQ);

    server_rig(llama_model * model, float adaptive) {
        params.n_ctx      = 8192;
        params.n_parallel = N_SEQ;
        params.n_batch    = 512;
        params.n_ubatch   = 512;
        params.kv_unified = true;
        params.cpuparams.n_threads       = 4;
        params.cpuparams_batch.n_threads = 4;
        params.speculative.types = { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
        params.speculative.draft.n_max = N_MAX;
        params.speculative.draft.mtp_adaptive = adaptive;
        const auto limits = common_speculative_get_output_limits(params.n_batch, params.n_parallel, common_speculative_n_max(&params.speculative));
        params.n_outputs_max         = limits.total;
        params.n_outputs_max_per_seq = limits.per_seq;

        tgt.reset(llama_init_from_model(model, common_context_params_to_llama(params)));
        if (!tgt) {
            throw std::runtime_error("failed to create the target context");
        }
        common_params params_dft = common_base_params_to_speculative(params);
        init = common_speculative_init_from_params(params_dft, model, tgt.get());
        if (!init->context()) {
            throw std::runtime_error("failed to create the MTP context");
        }
        params.speculative.draft.ctx_tgt = tgt.get();
        params.speculative.draft.ctx_dft = init->context();
        spec.reset(common_speculative_init(params.speculative, N_SEQ));
        if (!spec) {
            throw std::runtime_error("failed to create the drafter");
        }
        n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    }

    // the target computes a batch of rows, each sequence's rows consecutive; the drafter sees it as the server shows it
    // returns, per sequence, the batch index of its first row
    std::map<llama_seq_id, int> decode(const std::vector<std::pair<llama_seq_id, llama_tokens>> & rows) {
        int n = 0;
        for (const auto & r : rows) n += (int) r.second.size();
        llama_batch batch = llama_batch_init(n, 0, 1);
        std::map<llama_seq_id, int> first;
        for (const auto & r : rows) {
            // as the server asks: every row of a verification, the last row of a prompt
            const size_t n_out = r.second.size() <= 1 + N_MAX ? r.second.size() : 1;
            first[r.first] = batch.n_tokens;
            for (size_t i = 0; i < r.second.size(); ++i) {
                common_batch_add(batch, r.second[i], seqs[r.first].n_past++, { r.first }, i + n_out >= r.second.size());
            }
        }
        const int rc = llama_decode(tgt.get(), batch);
        const bool ok = rc == 0 && common_speculative_process(spec.get(), batch);
        llama_batch_free(batch);
        if (!ok) {
            throw std::runtime_error("failed to decode a target batch");
        }
        return first;
    }

    llama_tokens tokens(std::mt19937 & rng, int n) const {
        std::uniform_int_distribution<int> dis(0, n_vocab - 1);
        llama_tokens out(n);
        for (auto & t : out) t = dis(rng);
        return out;
    }

    // a new request on a slot: the last one released and its cells cleared, the prompt, then generation begins
    void request(llama_seq_id s, std::mt19937 & rng, int n_prompt) {
        common_speculative_end(spec.get(), s);
        llama_memory_seq_rm(llama_get_memory(tgt.get()), s, -1, -1);
        llama_memory_seq_rm(llama_get_memory(init->context()), s, -1, -1);
        common_speculative_set_state(spec.get(), s, {});
        seqs[s].n_past = 0;
        seqs[s].prompt = tokens(rng, n_prompt);
        decode({{ s, seqs[s].prompt }});
        common_speculative_begin(spec.get(), s, seqs[s].prompt);
        seqs[s].id_last = tokens(rng, 1)[0];
    }

    // the sequences given draft in one call; returns each one's draft length
    std::map<llama_seq_id, int> draft(const std::vector<llama_seq_id> & which) {
        for (const llama_seq_id s : which) {
            seqs[s].draft.clear();
            common_speculative_get_draft_params(spec.get(), s) = {
                /* .drafting = */ true,
                /* .n_max    = */ 64,
                /* .n_past   = */ seqs[s].n_past,
                /* .id_last  = */ seqs[s].id_last,
                /* .prompt   = */ &seqs[s].prompt,
                /* .result   = */ &seqs[s].draft,
            };
        }
        common_speculative_draft(spec.get());
        std::map<llama_seq_id, int> len;
        for (const llama_seq_id s : which) {
            len[s] = (int) seqs[s].draft.size();
            // the server drops the head's draft cells before the verification writes them again
            llama_memory_seq_rm(llama_get_memory(init->context()), s, seqs[s].n_past, -1);
        }
        return len;
    }

    // the verification batch of the drafted sequences, in the given row order
    std::map<llama_seq_id, int> verify(const std::vector<llama_seq_id> & order) {
        std::vector<std::pair<llama_seq_id, llama_tokens>> rows;
        for (const llama_seq_id s : order) {
            llama_tokens r = { seqs[s].id_last };
            r.insert(r.end(), seqs[s].draft.begin(), seqs[s].draft.end());
            rows.emplace_back(s, r);
        }
        return decode(rows);
    }

    void accept(llama_seq_id s, int n, std::mt19937 & rng) {
        common_speculative_accept(spec.get(), s, (uint16_t) n);
        seqs[s].id_last = tokens(rng, 1)[0];
    }
};

// which draft positions land: the first with probability p1, the next ones (given the first) with probability p2
struct acceptance {
    float p1 = 1.0f;
    float p2 = 1.0f;
    int operator()(int len, std::mt19937 & rng) const {
        std::uniform_real_distribution<float> u(0.0f, 1.0f);
        int n = 0;
        while (n < len && u(rng) < (n == 0 ? p1 : p2)) {
            ++n;
        }
        return n;
    }
};

static bool fail(const char * what) {
    fprintf(stderr, "FAIL: %s\n", what);
    return false;
}

static std::string str(const std::vector<int> & v) {
    std::string s;
    for (const int x : v) s += std::to_string(x);
    return s;
}

// draft lengths from a window boundary, replayed rounds left out: every window full depth or 7 one-token rounds and a
// full-depth probe, the last window a prefix of either; a request's first window is full depth
static bool stable_shape(const std::vector<int> & len, bool request_start = true) {
    for (size_t w = 0; w*WINDOW < len.size(); ++w) {
        bool full = true, low = w > 0 || !request_start;
        for (size_t i = w*WINDOW; i < std::min(len.size(), (w + 1)*WINDOW); ++i) {
            full = full && len[i] == N_MAX;
            low  = low  && len[i] == (i % WINDOW == WINDOW - 1 ? N_MAX : 1);
        }
        if (!full && !low) {
            return false;
        }
    }
    return true;
}

// fixed depth: every round of every sequence drafts n_max whatever lands
static bool test_baseline(llama_model * model) {
    server_rig rig(model, 0.0f);
    std::mt19937 rng(1);
    for (llama_seq_id s = 0; s < N_SEQ; ++s) rig.request(s, rng, 9 + s);
    const acceptance acc[N_SEQ] = { {0.0f, 0.0f}, {1.0f, 0.0f}, {0.5f, 0.5f} };
    for (int round = 0; round < 4*WINDOW; ++round) {
        const auto len = rig.draft({0, 1, 2});
        rig.verify({2, 0, 1});
        for (llama_seq_id s = 0; s < N_SEQ; ++s) {
            if (len.at(s) != N_MAX) return fail("fixed depth drafted fewer than n_max tokens");
            rig.accept(s, acc[s](len.at(s), rng), rng);
        }
    }
    return true;
}

// one batch, three sequences: the one whose drafts all land keeps the full depth; the one whose drafts never land and
// the one whose second token never lands drop to one token after the first window, probing once a window. A new request
// on a slot starts at full depth again, and leaves the other slots as they were. A sequence that starts to land again
// returns to the full depth within three windows
static bool test_adaptive(llama_model * model) {
    server_rig rig(model, 0.18f);
    std::mt19937 rng(2);
    for (llama_seq_id s = 0; s < N_SEQ; ++s) rig.request(s, rng, 7 + 2*s);
    acceptance acc[N_SEQ] = { {1.0f, 1.0f}, {0.0f, 0.0f}, {1.0f, 0.0f} };
    std::vector<int> len[N_SEQ];
    auto run = [&](int rounds) {
        for (int round = 0; round < rounds; ++round) {
            const auto l = rig.draft({0, 1, 2});
            rig.verify(round % 2 ? std::vector<llama_seq_id>{1, 2, 0} : std::vector<llama_seq_id>{0, 1, 2});
            for (llama_seq_id s = 0; s < N_SEQ; ++s) {
                len[s].push_back(l.at(s));
                rig.accept(s, acc[s](l.at(s), rng), rng);
            }
        }
    };
    run(3*WINDOW);
    const std::string low = "11111112";
    const std::string full(WINDOW, '0' + N_MAX);
    if (str(len[0]) != full + full + full) return fail("a sequence whose drafts all land left the full depth");
    if (str(len[1]) != full + low + low) return fail("a sequence whose drafts never land did not drop to one token");
    if (str(len[2]) != full + low + low) return fail("a sequence whose second token never lands did not drop to one token");

    // a new request on slot 1: full depth from its first round; slots 0 and 2 keep their own depth
    rig.request(1, rng, 5);
    for (auto & l : len) l.clear();
    run(WINDOW);
    if (str(len[1]) != full) return fail("a new request inherited the depth of the slot's previous request");
    if (str(len[0]) != full) return fail("another slot's new request moved a full-depth sequence");
    if (str(len[2]) != low) return fail("another slot's new request moved a one-token sequence");

    // slot 2 starts to land everything: back to the full depth within three windows, and it stays there
    acc[2] = {1.0f, 1.0f};
    for (auto & l : len) l.clear();
    run(6*WINDOW);
    const std::string tail = str(len[2]).substr(3*WINDOW);
    if (tail != full + full + full) return fail("a sequence whose drafts land again did not return to the full depth");
    for (llama_seq_id s = 0; s < N_SEQ; ++s) {
        if (!stable_shape(len[s], false)) return fail("a sequence's depth changed inside a window");
    }
    return true;
}

// a round the target verifies twice (a checkpoint replay) is not an observation: a sequence whose every round is
// replayed keeps the full depth however little lands
static bool test_replay(llama_model * model) {
    server_rig rig(model, 0.18f);
    std::mt19937 rng(3);
    rig.request(0, rng, 11);
    for (int round = 0; round < 4*WINDOW; ++round) {
        const auto len = rig.draft({0});
        if (len.at(0) != N_MAX) return fail("a replayed sequence left the full depth");
        rig.verify({0});
        rig.seqs[0].draft.resize(1);
        rig.verify({0});
        rig.accept(0, 0, rng);
    }
    // the same rounds verified once each are observations: one window, then one token
    std::vector<int> len;
    for (int round = 0; round < 2*WINDOW; ++round) {
        len.push_back(rig.draft({0}).at(0));
        rig.verify({0});
        rig.accept(0, 0, rng);
    }
    if (str(len) != std::string(WINDOW, '0' + N_MAX) + "11111112") return fail("rounds verified once did not move the depth");
    return true;
}

struct history_round {
    std::vector<llama_seq_id> drafting; // the sequences that draft this round, in batch order
    std::vector<bool>         replay;   // per sequence of the round
    std::vector<acceptance>   acc;      // per sequence of the round
};

// generated rounds: acceptance moves in phases, so that depths change; some rounds leave sequences out or replay them
static std::vector<history_round> generate(std::mt19937 & rng, int n_rounds, const std::vector<llama_seq_id> & seqs) {
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    std::vector<acceptance> phase(N_SEQ);
    std::vector<history_round> out;
    for (int round = 0; round < n_rounds; ++round) {
        if (round % 12 == 0) {
            for (auto & a : phase) {
                const float levels[] = { 0.0f, 0.2f, 0.6f, 1.0f };
                a = { levels[rng() % 4], levels[rng() % 4] };
            }
        }
        history_round r;
        for (const llama_seq_id s : seqs) {
            if (u(rng) < 0.85f) {
                r.drafting.push_back(s);
            }
        }
        std::shuffle(r.drafting.begin(), r.drafting.end(), rng);
        for (const llama_seq_id s : r.drafting) {
            r.replay.push_back(u(rng) < 0.1f);
            r.acc.push_back(phase[s]);
        }
        out.push_back(r);
    }
    return out;
}

// plays rounds; each sequence's acceptance draws from its own stream, so that a sequence's own rounds alone decide its
// draws. Returns, per sequence, its draft lengths with replayed rounds left out
static std::vector<std::vector<int>> play(server_rig & rig, const std::vector<history_round> & rounds, std::vector<std::mt19937> & draws) {
    std::vector<std::vector<int>> len(N_SEQ);
    std::mt19937 rng(7);
    for (const auto & r : rounds) {
        if (r.drafting.empty()) {
            continue;
        }
        const auto l = rig.draft(r.drafting);
        rig.verify(r.drafting);
        std::vector<llama_seq_id> replayed;
        for (size_t i = 0; i < r.drafting.size(); ++i) {
            if (r.replay[i]) {
                replayed.push_back(r.drafting[i]);
                rig.seqs[r.drafting[i]].draft.resize(draws[r.drafting[i]]() % (l.at(r.drafting[i]) + 1));
            }
        }
        if (!replayed.empty()) {
            rig.verify(replayed);
        }
        for (size_t i = 0; i < r.drafting.size(); ++i) {
            const llama_seq_id s = r.drafting[i];
            const int n_rows = r.replay[i] ? (int) rig.seqs[s].draft.size() : l.at(s);
            rig.accept(s, r.acc[i](n_rows, draws[s]), rng);
            if (!r.replay[i]) {
                len[s].push_back(l.at(s));
            }
        }
    }
    return len;
}

// over generated histories: shapes stay stable, a sequence's depths follow from its own rounds alone (the same rounds
// alone, or among other sequences' rounds, give the same lengths), and a slot's earlier request never reaches the next
static bool test_histories(llama_model * model) {
    int n_low = 0;
    for (uint32_t seed = 0; seed < 6; ++seed) {
        std::mt19937 rng(100 + seed);
        const auto rounds = generate(rng, 5*WINDOW, {0, 1, 2});

        // all three sequences together
        server_rig together(model, 0.18f);
        std::vector<std::mt19937> draws = { std::mt19937(seed*3 + 0), std::mt19937(seed*3 + 1), std::mt19937(seed*3 + 2) };
        for (llama_seq_id s = 0; s < N_SEQ; ++s) together.request(s, rng, 6 + s);
        const auto len = play(together, rounds, draws);

        for (llama_seq_id s = 0; s < N_SEQ; ++s) {
            n_low += (int) std::count(len[s].begin(), len[s].end(), 1);
            if (!stable_shape(len[s])) {
                fprintf(stderr, "seed %u seq %d: %s\n", seed, s, str(len[s]).c_str());
                return fail("a generated history changed a depth inside a window");
            }
            // the same sequence's rounds alone, after an earlier request of its own on the slot
            std::vector<history_round> own;
            for (const auto & r : rounds) {
                history_round o;
                for (size_t i = 0; i < r.drafting.size(); ++i) {
                    if (r.drafting[i] == s) {
                        o.drafting = { s };
                        o.replay   = { r.replay[i] };
                        o.acc      = { r.acc[i] };
                    }
                }
                own.push_back(o);
            }
            server_rig alone(model, 0.18f);
            std::vector<std::mt19937> earlier = { std::mt19937(99), std::mt19937(98), std::mt19937(97) };
            alone.request(s, rng, 5);
            std::vector<history_round> before = generate(rng, 3*WINDOW, {s});
            for (auto & r : before) {
                for (auto & a : r.acc) a = { 0.0f, 0.0f };
            }
            play(alone, before, earlier);
            alone.request(s, rng, 8);
            std::vector<std::mt19937> draws_alone = { std::mt19937(seed*3 + 0), std::mt19937(seed*3 + 1), std::mt19937(seed*3 + 2) };
            const auto len_alone = play(alone, own, draws_alone);
            if (len_alone[s] != len[s]) {
                fprintf(stderr, "seed %u seq %d: together %s, alone %s\n", seed, s, str(len[s]).c_str(), str(len_alone[s]).c_str());
                return fail("a sequence's depths depend on other sequences or on the slot's earlier request");
            }
        }
    }
    // the histories hold phases where nothing lands: some rounds must have drafted one token
    if (n_low == 0) return fail("no generated history ever drafted one token");
    return true;
}

// the kept row: after a batch, a sequence's state is its last row's position and the target's hidden row there; after
// accept(n) it is the verification's row n. Generated layouts: sequences in any order, prompt-length runs of rows
// longer than any draft, and verification-length runs; with the adaptive depth on and off
static bool test_kept_row(llama_model * model) {
    for (const float adaptive : {0.0f, 0.18f}) {
        server_rig rig(model, adaptive);
        std::mt19937 rng(adaptive > 0.0f ? 11 : 12);
        const int n_embd = llama_model_n_embd_out(llama_get_model(rig.tgt.get()));
        auto check = [&](llama_seq_id s, llama_pos pos, const float * h) {
            std::vector<uint8_t> state;
            if (!common_speculative_get_state(rig.spec.get(), s, state)) return false;
            if (state.size() != sizeof(llama_pos) + (size_t) n_embd*sizeof(float)) return false;
            llama_pos kept = -1;
            std::memcpy(&kept, state.data(), sizeof(llama_pos));
            return kept == pos && std::memcmp(state.data() + sizeof(llama_pos), h, (size_t) n_embd*sizeof(float)) == 0;
        };
        for (llama_seq_id s = 0; s < N_SEQ; ++s) rig.request(s, rng, 3 + s);
        // the head drafts once for every sequence, so that accept reaches it
        rig.draft({0, 1, 2});
        rig.verify({0, 1, 2});
        for (int step = 0; step < 40; ++step) {
            std::vector<llama_seq_id> order = {0, 1, 2};
            std::shuffle(order.begin(), order.end(), rng);
            order.resize(1 + rng() % N_SEQ);
            std::vector<std::pair<llama_seq_id, llama_tokens>> rows;
            for (const llama_seq_id s : order) {
                const int n = rng() % 3 == 0 ? 5 + rng() % 40 : 1 + rng() % (N_MAX + 1);
                rows.emplace_back(s, rig.tokens(rng, n));
            }
            std::map<llama_seq_id, llama_pos> pos0;
            for (const llama_seq_id s : order) pos0[s] = rig.seqs[s].n_past;
            const auto first = rig.decode(rows);
            std::map<llama_seq_id, std::vector<float>> h;
            for (const auto & r : rows) {
                const int n = (int) r.second.size();
                for (int i = 0; i < n; ++i) {
                    const float * row = llama_get_embeddings_nextn_ith(rig.tgt.get(), first.at(r.first) + i);
                    h[r.first].insert(h[r.first].end(), row, row + n_embd);
                }
                if (!check(r.first, pos0[r.first] + n - 1, h[r.first].data() + (size_t) (n - 1)*n_embd)) {
                    return fail("the kept row after a batch is not the sequence's last target row");
                }
            }
            for (const auto & r : rows) {
                if (rng() % 2) continue;
                const int n_rows = (int) r.second.size();
                const int n = rng() % (std::min(n_rows - 1, N_MAX + 1) + 1);
                common_speculative_accept(rig.spec.get(), r.first, (uint16_t) n);
                if (!check(r.first, pos0[r.first] + n, h[r.first].data() + (size_t) n*n_embd)) {
                    return fail("the kept row after accept is not the verification's accepted row");
                }
            }
        }
    }
    return true;
}

int main(void) {
    common_init();
    common_log_set_verbosity_thold(LOG_LEVEL_WARN);
    llama_log_set([](ggml_log_level level, const char * text, void *) {
        if (level >= GGML_LOG_LEVEL_ERROR) fputs(text, stderr);
    }, nullptr);
    llama_backend_init();

    bool ok = true;
    try {
        auto model = glm_model(1234);
        const std::pair<const char *, std::function<bool(llama_model *)>> tests[] = {
            { "baseline",  test_baseline  },
            { "adaptive",  test_adaptive  },
            { "replay",    test_replay    },
            { "histories", test_histories },
            { "kept-row",  test_kept_row  },
        };
        for (const auto & t : tests) {
            const bool pass = t.second(model.get());
            printf("%-10s %s\n", t.first, pass ? "OK" : "FAIL");
            ok = ok && pass;
        }
    } catch (const std::exception & err) {
        fprintf(stderr, "error: %s\n", err.what());
        ok = false;
    }
    llama_backend_free();
    return ok ? 0 : 1;
}
