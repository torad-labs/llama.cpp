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

#include "test-llama-archs.h"

#include "llama-archs-gguf.h"

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
};

struct context_shape {
    uint32_t n_ubatch  = 64;
    uint32_t n_seq_max = 1;
    uint32_t n_rs_seq  = 0;
};

llama_context_ptr make_context(llama_model * model, const context_shape & shape) {
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

std::map<std::string, case_def> cases(size_t seed) {
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
    ret["state"] = {{64, 2, 0}, [=](llama_context * ctx, record & out) {
        decode_span(ctx, tokens(n_prompt, 0), 0, 0, out);
        std::vector<uint8_t> whole(llama_state_get_size(ctx));
        std::vector<uint8_t> seq(llama_state_seq_get_size(ctx, 0));
        if (llama_state_get_data(ctx, whole.data(), whole.size()) != whole.size() ||
                llama_state_seq_get_data(ctx, seq.data(), seq.size(), 0) != seq.size()) {
            throw std::runtime_error("inkling2: the state was not saved");
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
        decode_steps(ctx, b, n_prompt, 1, out);
        read_state(ctx, 0, out);
        read_state(ctx, 1, out);
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
    return ret;
}

} // namespace

int test_inkling_split(const std::string & name, size_t seed, const std::vector<ggml_backend_dev_t> & devs) {
    if (devs.size() < 2) {
        fprintf(stderr, "inkling2 split %s: a tensor split needs two ranks, %zu given\n", name.c_str(), devs.size());
        return 1;
    }
    const auto all = cases(seed);
    const auto it = all.find(name);
    if (it == all.end()) {
        fprintf(stderr, "inkling2 split: no case %s\n", name.c_str());
        return 1;
    }
    auto gguf = get_gguf_ctx(LLM_ARCH_INKLING2, true);
    llama_model_ptr one = get_model(gguf.get(), nullptr, seed, 0.0f, {devs.front()}, LLAMA_SPLIT_MODE_LAYER);
    llama_model_ptr split = get_model(gguf.get(), nullptr, seed, 0.0f, devs, LLAMA_SPLIT_MODE_TENSOR);
    const bool graph_churn = name == "graph-churn";
    if (graph_churn) {
        set_env("LLAMA_RS_FIXED_ROWS", "0");
        set_env("LLAMA_GRAPH_ANY_SEQS", "0");
    }
    record expected, actual;
    {
        auto ctx = make_context(one.get(), it->second.shape);
        it->second.run(ctx.get(), expected);
    }
    {
        if (graph_churn) {
            set_env("LLAMA_RS_FIXED_ROWS", "1");
            set_env("LLAMA_GRAPH_ANY_SEQS", "1");
        }
        auto ctx = make_context(split.get(), it->second.shape);
        it->second.run(ctx.get(), actual);
    }
    bool ok = expected.logits.size() == actual.logits.size() && !expected.logits.empty() &&
            expected.states.size() == actual.states.size() && !expected.states.empty();
    if (graph_churn) {
        fprintf(stderr, "inkling2 split %s: %d of 23 changed session pairs reused a graph (baseline %d)\n",
                name.c_str(), actual.reused, expected.reused);
        ok = ok && actual.reused == 22 && expected.reused == 0;
    }
    const double logits_nmse = ok ? nmse(expected.logits, actual.logits) : INFINITY;
    ok = ok && logits_nmse <= tolerance;
    fprintf(stderr, "inkling2 split %s, %zu ranks: %zu logits, NMSE %g (at most %g)\n", name.c_str(), devs.size(),
            expected.logits.size(), logits_nmse, tolerance);
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
    fprintf(stderr, "%s inkling2 split %s\n", ok ? "OK" : "FAIL", name.c_str());
    return ok ? 0 : 1;
}
