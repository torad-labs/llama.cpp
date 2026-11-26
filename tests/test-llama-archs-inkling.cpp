// Inkling 2 split across devices gives the results it gives on one (-sm tensor, --inkling-split <case>). The fixture
// model is loaded twice, on the first device alone and over every rank of --inkling-split-device under a tensor split,
// and each case does the same to both contexts: the logits of every row they output, the convolution state of every
// sequence they keep, read directly as the memory serializes it, and the whole of what the sequence keeps (its KV
// cells beside that state) are held to the fixture's tolerance (the bar a device's logits are held to against the
// CPU's). The caches are F32, so that each word is a value (the convolution state takes the K cache's type). They are
// compared as well as the logits because the fixture's logits come mostly through the residual stream, past the
// attention: a split that read the keys' dynamic convolution at the wrong lag still had logits within 3e-6. The cases:
//
//   prefill   one batch of 64 tokens
//   decode    16 tokens one at a time after a prefill
//   chunks    64 tokens in one batch of ubatches of 16: three chunk boundaries the state crosses
//   seq-cp    a prefill copied to a second sequence, then both decoded on in the same batches
//   state     the context's state and a sequence's saved after a prefill, other tokens decoded, both restored (the
//             sequence's into a second sequence) and decoded on
//   rollback  a draft of 4 tokens of which the last 2 are rejected (llama_memory_seq_rm with rollback snapshots), the
//             state compared right after, then decoded on from the kept position
//   graph-churn  23 distinct session pairs through a retained graph, then independent rollbacks on two sessions
//   layer-inp  a prefill and decode steps with the layer inputs a DFlash/DSpark draft reads written to the device buffer
//             (layer_inp_dev): under the split, the copy on each rank holds the rows the one device writes

#include "test-llama-archs.h"

#include "llama-archs-gguf.h"
#include "speculative.h"
#include "../src/llama-context.h"
#include "../src/llama-model.h"
#include "../ggml/src/ggml-backend-impl.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <functional>
#include <map>
#include <stdexcept>

namespace {

constexpr double tolerance = 1e-4;
constexpr uint32_t n_prompt = 32;

struct record {
    int32_t reused = 0;
    std::vector<float> logits;              // every output row, in the order of the batches
    std::vector<std::vector<float>> states; // the serialized convolution state of each sequence, by step
    std::vector<std::vector<float>> memory; // all the sequence keeps, at the same steps: KV cells and the state
    std::vector<std::vector<float>> layer_rows; // the layer input rows of the device buffer, a copy a rank, by step
    std::vector<float> moe;
};

struct context_shape {
    uint32_t n_ubatch  = 64;
    uint32_t n_seq_max = 1;
    uint32_t n_rs_seq  = 0;
};

llama_context_ptr make_context(llama_model * model, const context_shape & shape, record * trace = nullptr) {
    llama_context_params params = llama_context_default_params();
    params.n_ctx = 256*shape.n_seq_max;
    params.n_batch = 256;
    params.n_ubatch = shape.n_ubatch;
    params.n_seq_max = shape.n_seq_max;
    params.n_rs_seq = shape.n_rs_seq;
    params.kv_unified = true;
    params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED; // -sm tensor requires it, and both contexts compute alike
    params.type_k = params.type_v = GGML_TYPE_F32; // the cells compare word by word as the state does
    params.n_threads = params.n_threads_batch = 4;
    if (trace) {
        params.cb_eval_user_data = trace;
        params.cb_eval = [](ggml_tensor * t, bool ask, void * data) {
            if (ask) {
                return strncmp(t->name, "ffn_moe_post_conv", sizeof("ffn_moe_post_conv") - 1) == 0;
            }
            auto & values = static_cast<record *>(data)->moe;
            const size_t pos = values.size();
            values.resize(pos + ggml_nelements(t));
            ggml_backend_tensor_get(t, values.data() + pos, 0, ggml_nbytes(t));
            return true;
        };
    }
    llama_context_ptr ctx(llama_init_from_model(model, params));
    if (!ctx) {
        throw std::runtime_error("inkling2: the context was not created");
    }
    return ctx;
}

// decodes one batch of (token, position, sequence) rows, every row an output, and records its logits
void decode(llama_context * ctx, const std::vector<std::array<int32_t, 3>> & rows, record & out) {
    llama_batch batch = llama_batch_init(rows.size(), 0, 1);
    for (const auto & [token, pos, seq] : rows) {
        batch_add(batch, token, pos, {seq}, true);
    }
    const int32_t ret = llama_decode(ctx, batch);
    llama_batch_free(batch);
    if (ret != 0) {
        throw std::runtime_error("inkling2: decode failed with " + std::to_string(ret));
    }
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx)));
    for (size_t i = 0; i < rows.size(); ++i) {
        const float * row = llama_get_logits_ith(ctx, i);
        out.logits.insert(out.logits.end(), row, row + n_vocab);
    }
}

// tokens at positions pos0.. of seq in one batch
void decode_span(llama_context * ctx, const std::vector<llama_token> & tokens, llama_pos pos0, llama_seq_id seq, record & out) {
    std::vector<std::array<int32_t, 3>> rows;
    for (size_t i = 0; i < tokens.size(); ++i) {
        rows.push_back({tokens[i], pos0 + int32_t(i), seq});
    }
    decode(ctx, rows, out);
}

// tokens at positions pos0.. of seq, one batch each
void decode_steps(llama_context * ctx, const std::vector<llama_token> & tokens, llama_pos pos0, llama_seq_id seq, record & out) {
    for (size_t i = 0; i < tokens.size(); ++i) {
        decode(ctx, {{tokens[i], pos0 + int32_t(i), seq}}, out);
    }
}

std::vector<float> state_words(llama_context * ctx, llama_seq_id seq, llama_state_seq_flags flags) {
    std::vector<uint8_t> bytes(llama_state_seq_get_size_ext(ctx, seq, flags));
    bytes.resize(llama_state_seq_get_data_ext(ctx, bytes.data(), bytes.size(), seq, flags));
    std::vector<float> words((bytes.size() + sizeof(float) - 1)/sizeof(float), 0.0f);
    memcpy(words.data(), bytes.data(), bytes.size());
    return words;
}

// the convolution state of seq, as the memory serializes its recurrent part (counts and positions beside the rows),
// and all that seq keeps, as a sequence's state is saved
void read_state(llama_context * ctx, llama_seq_id seq, record & out) {
    out.states.push_back(state_words(ctx, seq, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY));
    out.memory.push_back(state_words(ctx, seq, 0));
}

// the first n_rows rows of the layer input device buffer as a draft on each of devs reads them: a split context's copy on
// that device, or the buffer of a context on one device
void read_layer_rows(llama_context * ctx, const std::vector<ggml_backend_dev_t> & devs, int64_t n_rows, record & out) {
    ggml_tensor * t = ctx->get_layer_inp_dev();
    if (t == nullptr) {
        throw std::runtime_error("inkling2: the layer inputs have no device buffer");
    }
    llama_synchronize(ctx);
    for (ggml_backend_dev_t dev : devs) {
        ggml_tensor * copy = ggml_backend_buffer_is_meta(t->buffer) ? ggml_backend_meta_mirrored_tensor(t, dev) : t;
        if (copy == nullptr) {
            throw std::runtime_error(std::string("inkling2: no copy of the layer inputs on ") + ggml_backend_dev_name(dev));
        }
        std::vector<float> rows(size_t(t->ne[0])*n_rows);
        ggml_backend_tensor_get(copy, rows.data(), 0, rows.size()*sizeof(float));
        out.layer_rows.push_back(std::move(rows));
    }
}

// NMSE of two serialized states: the words of the counts and positions must be equal bit for bit, the rows within
// the tolerance; a state of another size, or a non-finite value, is infinitely far
double state_nmse(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size() || a.empty()) {
        return INFINITY;
    }
    double err = 0.0, norm = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (memcmp(&a[i], &b[i], sizeof(float)) == 0) {
            norm += std::isfinite(a[i]) ? double(a[i])*a[i] : 0.0;
            continue;
        }
        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) {
            return INFINITY;
        }
        err += double(a[i] - b[i])*(a[i] - b[i]);
        norm += double(a[i])*a[i];
    }
    return norm > 0.0 ? err/norm : INFINITY;
}

float largest(const std::vector<float> & values) {
    float ret = 0.0f;
    for (float v : values) {
        ret = std::isfinite(v) ? std::max(ret, std::fabs(v)) : ret;
    }
    return ret;
}

using inkling_case = std::function<void(llama_context *, record &)>;

struct case_def {
    context_shape shape;
    inkling_case run;
};

std::map<std::string, case_def> cases(size_t seed, const std::vector<ggml_backend_dev_t> & devs) {
    const auto tokens = [seed](uint32_t n, size_t salt) { return get_tokens(n, 128, seed + salt); };
    std::map<std::string, case_def> ret;
    ret["prefill"] = {{}, [=](llama_context * ctx, record & out) {
        decode_span(ctx, tokens(64, 0), 0, 0, out);
        read_state(ctx, 0, out);
    }};
    ret["decode"] = {{}, [=](llama_context * ctx, record & out) {
        decode_span(ctx, tokens(n_prompt, 0), 0, 0, out);
        decode_steps(ctx, tokens(16, 1), n_prompt, 0, out);
        read_state(ctx, 0, out);
    }};
    ret["chunks"] = {{16, 1, 0}, [=](llama_context * ctx, record & out) {
        decode_span(ctx, tokens(64, 0), 0, 0, out);
        read_state(ctx, 0, out);
        decode_steps(ctx, tokens(4, 1), 64, 0, out);
        read_state(ctx, 0, out);
    }};
    ret["seq-cp"] = {{64, 2, 0}, [=](llama_context * ctx, record & out) {
        decode_span(ctx, tokens(n_prompt, 0), 0, 0, out);
        llama_memory_seq_cp(llama_get_memory(ctx), 0, 1, -1, -1);
        read_state(ctx, 1, out);
        const auto a = tokens(8, 1), b = tokens(8, 2);
        for (uint32_t i = 0; i < 8; ++i) {
            decode(ctx, {{a[i], int32_t(n_prompt + i), 0}, {b[i], int32_t(n_prompt + i), 1}}, out);
        }
        read_state(ctx, 0, out);
        read_state(ctx, 1, out);
    }};
    ret["state"] = {{70, 2, 0}, [=](llama_context * ctx, record & out) {
        decode_span(ctx, tokens(n_prompt, 0), 0, 0, out);
        std::vector<uint8_t> whole(llama_state_get_size(ctx));
        std::vector<uint8_t> seq(llama_state_seq_get_size(ctx, 0));
        if (llama_state_get_data(ctx, whole.data(), whole.size()) != whole.size() ||
                llama_state_seq_get_data(ctx, seq.data(), seq.size(), 0) != seq.size()) {
            throw std::runtime_error("inkling2: the state was not saved");
        }
        if (llama_state_staged_rows(ctx) > 0) {
            const size_t psize = llama_state_seq_get_size_ext(ctx, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            std::vector<uint8_t> pref(psize), pstaged(psize);
            int32_t srow = -1;
            if (llama_state_seq_get_data_ext(ctx, pref.data(), psize, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) != psize ||
                    llama_state_seq_get_data_staged(ctx, pstaged.data(), psize, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY, &srow) != psize ||
                    srow < 0) {
                throw std::runtime_error("inkling2: staged state copy did not run");
            }
            while (llama_state_staged_pending(ctx, srow)) {}
            if (pref != pstaged) {
                throw std::runtime_error("inkling2: staged partial state bytes differ from sync partial state");
            }
        }
        record discarded;
        decode_steps(ctx, tokens(8, 1), n_prompt, 0, discarded);
        if (llama_state_set_data(ctx, whole.data(), whole.size()) != whole.size()) {
            throw std::runtime_error("inkling2: the state was not restored");
        }
        read_state(ctx, 0, out);
        const auto b = tokens(8, 2);
        decode_steps(ctx, b, n_prompt, 0, out);
        if (llama_state_seq_set_data(ctx, seq.data(), seq.size(), 1) != seq.size()) {
            throw std::runtime_error("inkling2: the sequence's state was not restored");
        }
        read_state(ctx, 1, out);
        std::vector<uint8_t> ref(llama_state_seq_get_size(ctx, 1)), async(ref.size());
        if (llama_state_seq_get_data(ctx, ref.data(), ref.size(), 1) != ref.size()) {
            throw std::runtime_error("inkling2: the restored sequence's state was not saved");
        }
        decode_steps(ctx, b, n_prompt, 1, out);
        read_state(ctx, 0, out);
        read_state(ctx, 1, out);

        if (llama_state_seq_set_data_async(ctx, seq.data(), seq.size(), 1, LLAMA_STATE_SEQ_FLAGS_NONE) != seq.size()) {
            throw std::runtime_error("inkling2: async restore did not consume the sequence state");
        }
        llama_state_seq_async_wait(ctx, 1);
        if (llama_state_seq_get_data(ctx, async.data(), async.size(), 1) != async.size() || ref != async) {
            throw std::runtime_error("inkling2: async restore differs from sync restore");
        }
    }};
    ret["rollback"] = {{64, 1, 8}, [=](llama_context * ctx, record & out) {
        decode_span(ctx, tokens(n_prompt, 0), 0, 0, out);
        decode_span(ctx, tokens(4, 1), n_prompt, 0, out);
        // the last two of the draft are rejected: the state goes back to the one after its first two
        if (!llama_memory_seq_rm(llama_get_memory(ctx), 0, n_prompt + 2, -1)) {
            throw std::runtime_error("inkling2: the rollback was refused");
        }
        read_state(ctx, 0, out);
        decode_steps(ctx, tokens(4, 2), n_prompt + 2, 0, out);
        read_state(ctx, 0, out);
    }};
    ret["graph-churn"] = {{64, 23, 2}, [=](llama_context * ctx, record & out) {
        constexpr int sessions = 23;
        std::array<llama_pos, sessions> pos{};
        std::vector<std::array<int32_t, 3>> rows;
        for (int seq = 0; seq < sessions; ++seq) {
            rows.push_back({tokens(1, seq)[0], pos[seq]++, seq});
        }
        decode(ctx, rows, out);
        llama_perf_context_reset(ctx);
        for (int i = 0; i < sessions; ++i) {
            rows.clear();
            for (int seq : {i, (i + 11) % sessions}) {
                for (llama_token token : tokens(3, sessions*i + seq)) {
                    rows.push_back({token, pos[seq]++, seq});
                }
            }
            decode(ctx, rows, out);
        }
        out.reused = llama_perf_context(ctx).n_reused;
        for (int seq : {3, 19}) {
            if (!llama_memory_seq_rm(llama_get_memory(ctx), seq, --pos[seq], -1)) {
                throw std::runtime_error("inkling2: the slot rollback was refused");
            }
            decode(ctx, {{tokens(1, 1000 + seq)[0], pos[seq], seq}}, out);
        }
        for (int seq = 0; seq < sessions; ++seq) {
            read_state(ctx, seq, out);
        }
    }};
    ret["layer-inp"] = {{}, [=](llama_context * ctx, record & out) {
        // a middle layer's input and the output tap, as a DSpark draft takes them
        const uint32_t n_layer = llama_model_n_layer(llama_get_model(ctx));
        llama_set_embeddings_layer_inp(ctx, n_layer/2, true);
        llama_set_embeddings_layer_inp(ctx, n_layer, true);
        llama_set_embeddings_layer_inp_dev(ctx, true);
        decode_span(ctx, tokens(n_prompt, 0), 0, 0, out);
        read_layer_rows(ctx, devs, n_prompt, out);
        const auto steps = tokens(4, 1);
        for (uint32_t i = 0; i < steps.size(); ++i) {
            decode(ctx, {{steps[i], int32_t(n_prompt + i), 0}}, out);
            read_layer_rows(ctx, devs, 1, out);
        }
        read_state(ctx, 0, out);
    }};
    return ret;
}

bool compact_greedy(std::size_t seed, const std::vector<ggml_backend_dev_t> & devs) {
    auto gguf = get_gguf_ctx(LLM_ARCH_INKLING2, true);
    auto model = get_model(gguf.get(), nullptr, seed, 0.0f, devs, LLAMA_SPLIT_MODE_TENSOR);
    const int vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
    auto reference = make_context(model.get(), {64, 2, 0});
    auto candidate = make_context(model.get(), {64, 2, 0});
    llama_set_output_candidates(candidate.get(), 3);
    bool ok = true;
    for (int pos = 0; pos < 6; ++pos) {
        const bool compact = pos % 2 == 0;
        ok = llama_set_seq_greedy_candidates(candidate.get(), 0) && ok;
        if (compact) {
            ok = llama_set_seq_greedy_candidates(candidate.get(), 1) && ok;
        } else {
            llama_set_seq_candidates(candidate.get(), 1, false);
        }

        llama_batch batch = llama_batch_init(2, 0, 1);
        batch_add(batch, (13*pos + 5) % vocab, pos, {0}, true);
        batch_add(batch, (19*pos + 7) % vocab, pos, {1}, true);
        const int ref_rc = llama_decode(reference.get(), batch);
        const int cand_rc = llama_decode(candidate.get(), batch);
        llama_batch_free(batch);
        if (ref_rc != 0 || cand_rc != 0) {
            return false;
        }
        for (int row = 0; row < 2; ++row) {
            const float * logits = llama_get_logits_ith(reference.get(), row);
            const auto actual = llama_get_sampling_row_ith(candidate.get(), row);
            // mixed batch: non-greedy row 1 keeps whole logits while greedy row 0 stays on compact candidates
            if (!compact && row == 1) {
                ok = ok && actual.logits && std::equal(logits, logits + vocab, actual.logits);
                continue;
            }
            if (actual.logits || actual.n_candidates != 3 || actual.n_sampled_logits != 3) {
                return false;
            }
            std::vector<llama_token> ids(vocab);
            for (int id = 0; id < vocab; ++id) { ids[id] = id; }
            std::partial_sort(ids.begin(), ids.begin() + 3, ids.end(), [&](llama_token a, llama_token b) {
                return logits[a] > logits[b] || (logits[a] == logits[b] && a < b);
            });
            for (int i = 0; i < 3; ++i) {
                ok = ok && actual.candidates[i] == ids[i] && actual.sampled_logits[i] == logits[ids[i]];
            }
        }
    }
    fprintf(stderr, "%s compact greedy: padded head and mixed output batches\n", ok ? "OK" : "FAIL");
    return ok;
}

} // namespace

int test_inkling_split(const std::string & name, size_t seed, const std::vector<ggml_backend_dev_t> & devs) {
    if (devs.size() < 2) {
        fprintf(stderr, "inkling2 split %s: a tensor split needs two ranks, %zu given\n", name.c_str(), devs.size());
        return 1;
    }
    if (name == "compact-greedy") {
        return compact_greedy(seed, devs) ? 0 : 1;
    }
    const auto all = cases(seed, devs);
    const bool banks = name == "expert-banks";
    const auto it = all.find(banks ? "decode" : name);
    if (it == all.end()) {
        fprintf(stderr, "inkling2 split: no case %s\n", name.c_str());
        return 1;
    }
    auto gguf = get_gguf_ctx(LLM_ARCH_INKLING2, true);
    if (banks) {
        if (devs.size() != 4) {
            throw std::runtime_error("expert banks need four ranks");
        }
        kv_writer write(LLM_ARCH_INKLING2, gguf.get());
        write.add_kv(LLM_KV_EXPERT_COUNT, uint32_t(8));
        write.add_kv(LLM_KV_EXPERT_USED_COUNT, uint32_t(4));
        write.add_kv(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, uint32_t(2304));
        ggml_context_ptr meta(ggml_init({6*ggml_tensor_overhead(), nullptr, true}));
        for (int i = 0; i < 3; ++i) {
            auto * gu = ggml_new_tensor_3d(meta.get(), GGML_TYPE_Q3_K, 512, 4608, 8);
            auto * dn = ggml_new_tensor_3d(meta.get(), i == 0 ? GGML_TYPE_Q5_K : GGML_TYPE_Q4_K, 2304, 512, 8);
            ggml_format_name(gu, "blk.%d.ffn_gate_up_exps.weight", i);
            ggml_format_name(dn, "blk.%d.ffn_down_exps.weight", i);
            gguf_add_tensor(gguf.get(), gu);
            gguf_add_tensor(gguf.get(), dn);
        }
    }
    llama_model_ptr one = get_model(gguf.get(), nullptr, seed, 0.0f, {devs.front()}, LLAMA_SPLIT_MODE_LAYER);
    llama_model_ptr split = get_model(gguf.get(), nullptr, seed, 0.0f, devs, LLAMA_SPLIT_MODE_TENSOR);
    if (banks) {
        const char * layout = getenv("LLAMA_INKLING2_EXPERT_LAYOUT");
        const bool rotated = layout && strcmp(layout, "rotated") == 0;
        for (const auto & l : split->layers) {
            if (bool(l.ffn_down_exps->flags & GGML_TENSOR_FLAG_META_EXPERT_BANKS) != rotated ||
                bool(l.ffn_gate_up_exps->flags & GGML_TENSOR_FLAG_META_EXPERT_BANKS) != rotated) {
                return 1;
            }
        }
    }
    const bool graph_churn = name == "graph-churn";
    if (name == "state") {
        set_env("LLAMA_RS_STAGE_ROWS", "2");
        set_env("LLAMA_STATE_SAVE_ASYNC", "1");
    }
    if (graph_churn) {
        set_env("LLAMA_RS_FIXED_ROWS", "0");
        set_env("LLAMA_GRAPH_ANY_SEQS", "0");
    }
    record expected, actual;
    {
        auto ctx = make_context(one.get(), it->second.shape, banks ? &expected : nullptr);
        it->second.run(ctx.get(), expected);
    }
    {
        if (graph_churn) {
            set_env("LLAMA_RS_FIXED_ROWS", "1");
            set_env("LLAMA_GRAPH_ANY_SEQS", "1");
        }
        auto ctx = make_context(split.get(), it->second.shape, banks ? &actual : nullptr);
        it->second.run(ctx.get(), actual);
    }
    bool ok = expected.logits.size() == actual.logits.size() && !expected.logits.empty() &&
            expected.states.size() == actual.states.size() && !expected.states.empty();
    if (graph_churn) {
        fprintf(stderr, "inkling2 split %s: %d of 23 changed session pairs reused a graph (baseline %d)\n",
                name.c_str(), actual.reused, expected.reused);
        ok = ok && actual.reused == 22 && expected.reused == 0;
    }
    if (banks && ok) {
        const int vocab = llama_vocab_n_tokens(llama_model_get_vocab(one.get()));
        size_t differing = 0;
        for (size_t pos = 0; pos < expected.logits.size(); pos += vocab) {
            const float * a = expected.logits.data() + pos;
            const float * b = actual.logits.data() + pos;
            differing += std::max_element(a, a + vocab) - a != std::max_element(b, b + vocab) - b;
        }
        const double moe_nmse = expected.moe.empty() || expected.moe.size() != actual.moe.size() ?
            INFINITY : nmse(expected.moe, actual.moe);
        fprintf(stderr, "Meta 4-rank expert banks: %zu differing top tokens, MoE NMSE %g\n", differing, moe_nmse);
        ok = differing == 0 && moe_nmse <= 1e-3;
    }
    const double logits_nmse = ok ? nmse(expected.logits, actual.logits) : INFINITY;
    const int vocab = llama_vocab_n_tokens(llama_model_get_vocab(one.get()));
    size_t differ = 0;
    for (size_t offset = 0; ok && offset < expected.logits.size(); offset += vocab) {
        const auto * a = expected.logits.data() + offset;
        const auto * b = actual.logits.data() + offset;
        differ += std::max_element(a, a + vocab) - a != std::max_element(b, b + vocab) - b;
    }
    fprintf(stderr, "inkling2 split %s, %zu ranks: %zu logits, NMSE %g (at most %g)\n", name.c_str(), devs.size(),
            expected.logits.size(), logits_nmse, tolerance);
    if (ok) {
        fprintf(stderr, "inkling2 split %s: %zu differing top tokens\n", name.c_str(), differ);
    }
    ok = ok && logits_nmse <= tolerance;
    for (size_t i = 0; i < expected.states.size() && i < actual.states.size(); ++i) {
        // a state of zeros would agree with any other: the one-rank state must hold the convolution's history
        const double d = state_nmse(expected.states[i], actual.states[i]);
        const bool held = largest(expected.states[i]) > 1e-6f;
        ok = ok && held && d <= tolerance;
        fprintf(stderr, "inkling2 split %s: convolution state %zu, %zu words, largest %g, NMSE %g\n", name.c_str(), i,
                expected.states[i].size(), largest(expected.states[i]), d);
        const double m = state_nmse(expected.memory[i], actual.memory[i]);
        ok = ok && m <= tolerance;
        fprintf(stderr, "inkling2 split %s: sequence memory %zu, %zu words, NMSE %g\n", name.c_str(), i,
                expected.memory[i].size(), m);
    }
    ok = ok && expected.layer_rows.size() == actual.layer_rows.size();
    for (size_t i = 0; i < expected.layer_rows.size() && i < actual.layer_rows.size(); ++i) {
        const double d = nmse(expected.layer_rows[i], actual.layer_rows[i]);
        ok = ok && largest(expected.layer_rows[i]) > 1e-6f && d <= tolerance;
        fprintf(stderr, "inkling2 split %s: layer input rows %zu (rank %zu), %zu values, NMSE %g\n", name.c_str(),
                i/devs.size(), i%devs.size(), expected.layer_rows[i].size(), d);
    }
    if (banks && ok && (split->layers[0].ffn_down_exps->flags & GGML_TENSOR_FLAG_META_EXPERT_BANKS)) {
        // Exchange the first and last K blocks only in down, leaving both gate/up halves unchanged.
        auto * down = split->layers[0].ffn_down_exps;
        std::vector<uint8_t> plane(down->nb[2]);
        for (int64_t e = 0; e < down->ne[2]; ++e) {
            ggml_backend_tensor_get(down, plane.data(), e*plane.size(), plane.size());
            for (int64_t r = 0; r < down->ne[1]; ++r) {
                auto * row = plane.data() + r*down->nb[1];
                std::swap_ranges(row, row + down->nb[0], row + 8*down->nb[0]);
            }
            ggml_backend_tensor_set(down, plane.data(), e*plane.size(), plane.size());
        }
        record wrong;
        auto ctx = make_context(split.get(), it->second.shape, &wrong);
        it->second.run(ctx.get(), wrong);
        const double bad_nmse = expected.moe.size() == wrong.moe.size() ? nmse(expected.moe, wrong.moe) : 0.0;
        fprintf(stderr, "Meta 4-rank mispaired K blocks: MoE NMSE %g\n", bad_nmse);
        ok = std::isfinite(bad_nmse) && bad_nmse > 1e-3;
    }
    fprintf(stderr, "%s inkling2 split %s\n", ok ? "OK" : "FAIL", name.c_str());
    return ok ? 0 : 1;
}

int test_inkling_dspark(const std::string & draft, size_t seed, const std::vector<ggml_backend_dev_t> & devs) {
    if (devs.size() != 4) {
        throw std::runtime_error("Inkling DSpark test requires four ranks");
    }
    set_env("LLAMA_TOK_EMBD_GPU", "1");
    auto gguf = get_gguf_ctx(LLM_ARCH_INKLING2, true);
    auto target = get_model(gguf.get(), nullptr, seed, 0.0f, devs, LLAMA_SPLIT_MODE_TENSOR);
    constexpr int n_seq = 2, depth = 3, rounds = 6;
    struct draft_result {
        llama_tokens ids;
        std::vector<llama_token_data> q;
        std::vector<float> ring;
    };
    const auto run = [&](bool split, bool probabilistic) {
        set_env("LLAMA_DFLASH_DEV", split ? "1" : "0");
        auto ctx_tgt = make_context(target.get(), {256, n_seq, depth});

        common_params params;
        params.n_parallel = n_seq;
        params.n_batch = params.n_ubatch = 256;
        params.kv_unified = true;
        params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        params.cpuparams.n_threads = params.cpuparams_batch.n_threads = 4;
        params.speculative.types = {COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK};
        auto & opts = params.speculative.draft;
        opts.mparams.path = draft;
        opts.devices = split ? devs : std::vector<ggml_backend_dev_t>{devs.front()};
        opts.devices.push_back(nullptr);
        opts.n_gpu_layers = 99;
        opts.n_min = 1;
        opts.n_max = depth;
        opts.probabilistic = probabilistic;
        if (split) {
            opts.split_mode = LLAMA_SPLIT_MODE_TENSOR;
        }
        auto draft_params = common_base_params_to_speculative(params);
        auto init = common_speculative_init_from_params(draft_params, target.get(), ctx_tgt.get());
        if (!init->context()) {
            throw std::runtime_error("Inkling DSpark test could not create the draft context");
        }
        opts.ctx_tgt = ctx_tgt.get();
        opts.ctx_dft = init->context();
        common_speculative_ptr spec(common_speculative_init(params.speculative, n_seq));
        if (!spec || bool(ctx_tgt->get_layer_inp_dev()) != split || llama_supports_embd_row(opts.ctx_dft) != split) {
            throw std::runtime_error("Inkling DSpark test did not select the requested feed path");
        }
        std::vector<draft_result> results;
        std::array<llama_tokens, n_seq> prompt;
        common_batch batch(ctx_tgt.get());
        int pos = 0;
        for (int round = 0; round < rounds; ++round) {
            const int feed = round == 0 ? 16 : depth;
            batch.clear();
            for (int seq = 0; seq < n_seq; ++seq) {
                for (int i = 0; i < feed; ++i) {
                    const llama_token id = 5 + (pos + i + seq * 11) % 112;
                    batch.add(id, pos + i, seq, false);
                    prompt[seq].push_back(id);
                }
            }
            if (llama_process(ctx_tgt.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get()) || !common_speculative_process(spec.get(), batch)) {
                throw std::runtime_error("Inkling DSpark test could not feed target rows");
            }
            pos += feed;
            std::array<draft_result, n_seq> step;
            std::array<std::vector<std::vector<llama_token_data>>, n_seq> q;
            for (int seq = 0; seq < n_seq; ++seq) {
                auto & dp = common_speculative_get_draft_params(spec.get(), seq);
                dp = {};
                dp.drafting = true;
                dp.pos0 = pos;
                dp.id_last = 17 + seq;
                dp.prompt = &prompt[seq];
                dp.result = &step[seq].ids;
                dp.result_q = probabilistic ? &q[seq] : nullptr;
                dp.seed = (uint32_t) seed;
                dp.temp = 0.7f;
            }
            common_speculative_draft(spec.get());
            for (int seq = 0; seq < n_seq; ++seq) {
                if (step[seq].ids.size() != depth || (probabilistic && q[seq].size() != depth)) {
                    throw std::runtime_error("Inkling DSpark test did not draft every row");
                }
                for (size_t i = 0; i < q[seq].size(); ++i) {
                    const auto & row = q[seq][i];
                    const auto sampled = std::find_if(row.begin(), row.end(), [&](const auto & e) { return e.id == step[seq].ids[i]; });
                    if (sampled == row.end() || !(sampled->p > 0.0f)) {
                        throw std::runtime_error("Inkling DSpark drew a token outside its proposal distribution");
                    }
                    float mass = 0.0f;
                    for (const auto & e : row) {
                        mass += e.p;
                    }
                    if (!(std::abs(mass - 1.0f) <= 1e-6f)) {
                        throw std::runtime_error("Inkling DSpark proposal distribution does not sum to one");
                    }
                    step[seq].q.insert(step[seq].q.end(), row.begin(), row.end());
                }
                if (!llama_memory_seq_rm(llama_get_memory(opts.ctx_dft), seq, pos, -1)) {
                    throw std::runtime_error("Inkling DSpark test could not remove draft rows");
                }
                std::vector<uint8_t> state;
                if (!common_speculative_get_state(spec.get(), seq, state) || state.size() <= 9*sizeof(llama_pos)) {
                    throw std::runtime_error("Inkling DSpark test could not save the anchor ring");
                }
                common_speculative_set_state(spec.get(), seq, state);
                step[seq].ring.resize((state.size() - 9*sizeof(llama_pos))/sizeof(float));
                memcpy(step[seq].ring.data(), state.data() + 9*sizeof(llama_pos), step[seq].ring.size()*sizeof(float));
                results.push_back(std::move(step[seq]));
            }
        }
        return results;
    };
    bool ok = true;
    for (const bool sampled : {false, true}) {
        const auto one = run(false, sampled);
        const auto tp4 = run(true, sampled);
        int differences = 0, changed_support = 0;
        float max_q = 0.0f;
        double max_ring = 0.0;
        for (size_t i = 0; i < one.size(); ++i) {
            differences += one[i].ids != tp4[i].ids;
            if (one[i].q.size() != tp4[i].q.size()) {
                throw std::runtime_error("Inkling DSpark proposal support sizes differ");
            }
            const size_t k = one[i].q.size()/depth;
            for (int step = 0; k > 0 && step < depth; ++step) {
                const auto * a = one[i].q.data() + step*k;
                const auto * b = tp4[i].q.data() + step*k;
                float dlogit = 0.0f;
                int changed = 0;
                for (size_t j = 0; j < k; ++j) {
                    const auto * match = std::lower_bound(b, b + k, a[j].id,
                            [](const auto & e, llama_token id) { return e.id < id; });
                    if (match == b + k || match->id != a[j].id) {
                        ++changed;
                    } else {
                        max_q = std::max(max_q, std::abs(a[j].p - match->p));
                        dlogit = std::max(dlogit, std::abs(a[j].logit - match->logit));
                    }
                }
                changed_support += changed;
                if (changed) {
                    fprintf(stderr, "Inkling DSpark round %zu slot %zu proposal %d: %d displaced candidates, max shared logit delta %g\n",
                            i/n_seq, i%n_seq, step + 1, changed, dlogit);
                }
            }
            max_ring = std::max(max_ring, nmse(one[i].ring, tp4[i].ring));
            ok = ok && largest(one[i].ring) > 1e-6f;
        }
        fprintf(stderr, "Inkling DSpark TP4 %s: %d differing blocks, %d differing candidate IDs, max q delta %g, ring NMSE %g\n",
                sampled ? "sampled" : "greedy", differences, changed_support, max_q, max_ring);
        ok = ok && differences == 0 && changed_support == 0 && max_q <= 2e-5f && max_ring <= tolerance;
    }
    return ok ? 0 : 1;
}

// Inkling 2 Loop's run-time pass count (LLAMA_LOOP_PASSES, llama_set_loop_passes, LLAMA_LOOP_PASSES_MAX): a count the
// model's own gives the same logits and state the fixed build gives, a varying model runs the counts its slots hold,
// every other way is refused, and a count change costs the graph its reuse. One device, the fixture of the loop arch
int test_inkling_loop(size_t seed, ggml_backend_dev_t dev) {
    bool ok = true;
    const auto check = [&ok](bool held, const char * what) {
        fprintf(stderr, "%s inkling2 loop %s\n", held ? "OK" : "FAIL", what);
        ok = ok && held;
    };
    const auto refuses = [](const std::function<void()> & f) {
        try {
            f();
            return false;
        } catch (const std::exception &) {
            return true;
        }
    };
    const auto tokens = [&](uint32_t n, size_t salt) { return get_tokens(n, 128, seed + salt); };
    // prefill 32 rows, then 4 steps: two graph shapes a same-count rerun of both reuses; a run starts from empty memory
    const auto run = [&](llama_context * ctx, record & out) {
        llama_memory_clear(llama_get_memory(ctx), true);
        decode_span(ctx, tokens(32, 10), 0, 0, out);
        decode_steps(ctx, tokens(4, 11), 32, 0, out);
    };

    // the varying fixture loaded with slots for three passes, without, and the fixed one; one seed, so their weights
    // agree tensor for tensor and a same-graph run may only differ as the backend is unsteady
    set_env("LLAMA_LOOP_PASSES_MAX", "3");
    llama_model_ptr vary_max = get_model(get_gguf_ctx(LLM_ARCH_INKLING2_LOOP, true, false, 0, true).get(), nullptr, seed, 0.0f, {dev});
    set_env("LLAMA_LOOP_PASSES_MAX", nullptr);
    llama_model_ptr vary     = get_model(get_gguf_ctx(LLM_ARCH_INKLING2_LOOP, true, false, 0, true).get(), nullptr, seed, 0.0f, {dev});
    llama_model_ptr fixed    = get_model(get_gguf_ctx(LLM_ARCH_INKLING2_LOOP, true, false, 0, false).get(), nullptr, seed, 0.0f, {dev});
    llama_model_ptr plain    = get_model(get_gguf_ctx(LLM_ARCH_INKLING2, true).get(), nullptr, seed, 0.0f, {dev});
    const context_shape shape = {64, 1, 0};

    // every count of the varying model runs, and a count's output differs from the next count's
    record r1, r2, r3;
    {
        auto ctx = make_context(vary_max.get(), shape);
        llama_set_loop_passes(ctx.get(), 1);
        run(ctx.get(), r1);
        llama_set_loop_passes(ctx.get(), 2);
        run(ctx.get(), r2);
        llama_set_loop_passes(ctx.get(), 3);
        run(ctx.get(), r3);
    }
    // the fixture's weights are small, so a pass moves the logits by less than the tolerance: a count differs when it moves
    // them past what a rerun of one count moves them (nothing on the CPU, the backend's unsteadiness on a GPU), with a floor
    record r3_again;
    {
        auto ctx = make_context(vary_max.get(), shape);
        llama_set_loop_passes(ctx.get(), 3);
        run(ctx.get(), r3_again);
    }
    const double noise = nmse(r3_again.logits, r3.logits);
    const double apart = std::max(1e-9, 100.0*noise);
    fprintf(stderr, "inkling2 loop counts: %zu logits, NMSE 2 from 1 %g, 3 from 2 %g, a rerun of 3 from 3 %g\n",
            r1.logits.size(), nmse(r2.logits, r1.logits), nmse(r3.logits, r2.logits), noise);
    check(!r1.logits.empty() && nmse(r2.logits, r1.logits) > apart && nmse(r3.logits, r2.logits) > apart,
            "counts 1, 2 and 3 run and differ");

    // a decode reuses the graph of the count before it when the count is the same, and no graph crosses a count change
    {
        auto ctx = make_context(vary_max.get(), shape);
        record discarded;
        llama_set_loop_passes(ctx.get(), 2);
        run(ctx.get(), discarded);
        llama_perf_context_reset(ctx.get());
        run(ctx.get(), discarded);   // same count, same shapes: both rows reused
        const int same = llama_perf_context(ctx.get()).n_reused;
        llama_set_loop_passes(ctx.get(), 1);
        run(ctx.get(), discarded);   // the count changed: rebuilt, nothing reused
        const int changed = llama_perf_context(ctx.get()).n_reused;
        run(ctx.get(), discarded);   // settled at the new count: reused again
        const int settled = llama_perf_context(ctx.get()).n_reused;
        // a run is 5 decodes: the prefill, the first step (a shape of its own) and three steps of that shape. The same
        // count reuses all 5 (the graphs are kept), a changed count rebuilds the prefill and the first step and reuses
        // the other 3, and the settled count reuses all 5 again
        fprintf(stderr, "inkling2 loop reuse: %d reused rerunning a count, then %d on a changed count, then %d settled\n",
                same, changed - same, settled - changed);
        check(same == 5 && changed - same == 3 && settled - changed == 5,
                "graph reuse follows the count (rerun 5, changed count 3, settled 5)");
    }

    // the model's own count is the same logits the fixed build gives, with the count asked twice (LLAMA_LOOP_PASSES=2
    // at the context) and with loop.varying (llama_set_loop_passes(2))
    record ref_fixed, ref_env, ref_vary;
    {
        run(make_context(fixed.get(), shape).get(), ref_fixed);
    }
    {
        set_env("LLAMA_LOOP_PASSES", "2");
        run(make_context(fixed.get(), shape).get(), ref_env);
        set_env("LLAMA_LOOP_PASSES", nullptr);
    }
    {
        auto ctx = make_context(vary.get(), shape);
        llama_set_loop_passes(ctx.get(), 2);
        run(ctx.get(), ref_vary);
    }
    const double env_nmse   = nmse(ref_fixed.logits, ref_env.logits);
    const double vary_nmse  = nmse(ref_fixed.logits, ref_vary.logits);
    check(env_nmse <= tolerance && vary_nmse <= tolerance,
            "the model's own count holds with LLAMA_LOOP_PASSES and on a varying model");

    // reset (0) returns the context to the model's own count
    record d, e, f;
    {
        auto ctx = make_context(vary.get(), shape);
        run(ctx.get(), d);
        llama_set_loop_passes(ctx.get(), 1);
        run(ctx.get(), e);
        llama_set_loop_passes(ctx.get(), 0);
        run(ctx.get(), f);
    }
    check(nmse(d.logits, f.logits) <= tolerance && nmse(d.logits, e.logits) > apart,
            "llama_set_loop_passes(0) resets to the model's own count");

    // a fixed model refuses a count it was not trained for, unless LLAMA_LOOP_PASSES_ANY says run it untrained
    {
        auto ctx = make_context(fixed.get(), shape);
        check(refuses([&] {
            llama_set_loop_passes(ctx.get(), 1);
            record discarded;
            run(ctx.get(), discarded);
        }), "a fixed model refuses one pass");
        check(refuses([&] { record discarded; run(ctx.get(), discarded); }), "a fixed model refuses one pass again");
        // the refusal is the context's to forget: the count allowed since, the same context runs it. A refused build must
        // leave no graph behind for the next decode of the same shape to reuse
        set_env("LLAMA_LOOP_PASSES_ANY", "1");
        record recovered;
        bool ran = true;
        try {
            run(ctx.get(), recovered);
        } catch (const std::exception & err) {
            fprintf(stderr, "inkling2 loop: the refused count did not run once allowed: %s\n", err.what());
            ran = false;
        }
        set_env("LLAMA_LOOP_PASSES_ANY", nullptr);
        const double recovered_nmse = ran ? nmse(recovered.logits, r1.logits) : INFINITY;
        fprintf(stderr, "inkling2 loop: the count refused, then allowed, in one context: NMSE %g from the same count on a fresh one\n",
                recovered_nmse);
        check(ran && recovered_nmse <= tolerance, "a refused count runs in the same context once allowed");
    }
    {
        set_env("LLAMA_LOOP_PASSES_ANY", "1");
        auto ctx = make_context(fixed.get(), shape);
        llama_set_loop_passes(ctx.get(), 1);
        record untrained;
        bool ran = true;
        try {
            run(ctx.get(), untrained);
        } catch (const std::exception & err) {
            fprintf(stderr, "inkling2 loop: LLAMA_LOOP_PASSES_ANY: %s\n", err.what());
            ran = false;
        }
        set_env("LLAMA_LOOP_PASSES_ANY", nullptr);
        // the fixed fixture's weights are the varying fixture's, tensor for tensor: one pass gives r1
        check(ran && nmse(untrained.logits, r1.logits) <= tolerance, "LLAMA_LOOP_PASSES_ANY runs the untrained count");
    }

    // slots are built for the model's own count unless LLAMA_LOOP_PASSES_MAX raises them at load: three passes of the
    // varying fixture refuse without the maximum, run with it, and the maximum leaves the default count's logits alone
    {
        auto ctx = make_context(vary.get(), shape);
        check(refuses([&] {
            llama_set_loop_passes(ctx.get(), 3);
            record discarded;
            run(ctx.get(), discarded);
        }), "three passes refuse without LLAMA_LOOP_PASSES_MAX");
    }
    {
        auto ctx = make_context(vary_max.get(), shape);
        llama_set_loop_passes(ctx.get(), 3);
        record with_max;
        run(ctx.get(), with_max);
        check(nmse(with_max.logits, r3.logits) <= tolerance, "LLAMA_LOOP_PASSES_MAX sizes the slots for three passes");
    }

    // a saved state carries cells laid out for the count it ran under: saving and restoring refuse another count in
    // every surface that moves them, and hold at the model's own (a refusal answers 0, as errors do)
    {
        auto ctx = make_context(vary_max.get(), shape);
        record discarded;
        run(ctx.get(), discarded);   // the model's own count: there are cells to save
        const size_t whole = llama_state_get_size(ctx.get()), seq0 = llama_state_seq_get_size(ctx.get(), 0);
        llama_set_loop_passes(ctx.get(), 1);
        const bool refused = llama_state_get_size(ctx.get()) == 0 && llama_state_seq_get_size(ctx.get(), 0) == 0 &&
                llama_state_seq_get_size_ext(ctx.get(), 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0;
        llama_set_loop_passes(ctx.get(), 0);
        const size_t back = llama_state_get_size(ctx.get());
        fprintf(stderr, "inkling2 loop state sizes: whole %zu, sequence %zu; recounted save refused: %s; reset own: %zu\n",
                whole, seq0, refused ? "yes" : "no", back);
        check(whole > 0 && seq0 > 0 && refused && back == whole,
                "state save/load refuses a recount and holds at the model's own");
    }

    // the draft's layer taps read fixed cache slots: they refuse a count that is not the model's own, and run at it
    {
        auto ctx = make_context(vary_max.get(), shape);
        llama_set_embeddings_layer_inp(ctx.get(), 0, true);
        llama_set_embeddings_layer_inp_dev(ctx.get(), true);
        check(refuses([&] {
            llama_set_loop_passes(ctx.get(), 1);
            record discarded;
            run(ctx.get(), discarded);
        }), "taps refuse a count the model's own");
        bool ran = true;
        try {
            llama_set_loop_passes(ctx.get(), 0);
            record tapped;
            run(ctx.get(), tapped);
        } catch (const std::exception & err) {
            fprintf(stderr, "inkling2 loop: taps at the model's own count: %s\n", err.what());
            ran = false;
        }
        check(ran, "taps run at the model's own count");
    }

    // a model without the loop refuses the setter at any count but the reset
    {
        auto ctx = make_context(plain.get(), shape);
        check(refuses([&] { llama_set_loop_passes(ctx.get(), 1); }) && !refuses([&] { llama_set_loop_passes(ctx.get(), 0); }),
                "a plain inkling2 refuses the setter");
    }

    fprintf(stderr, "%s inkling2 loop\n", ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}
