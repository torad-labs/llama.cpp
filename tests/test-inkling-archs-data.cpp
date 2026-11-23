// Inkling 2 under -sm data: sequence-parallel rows with the experts partitioned a device (doc 17 D1's row partition,
// the fork's data-parallel vehicle that GLM-5.3 already uses). Each device owns whole sequences (seq_id % ranks), so
// every per-row op of the layer -- the depth mix, the norms, the convolutions, the relative attention and the KV cache
// -- runs on the device's own rows and reads no other device's, while the experts read every row: the rows are
// gathered before the expert products and each device's share of their sum comes back to it through a reduce to owners
// (ggml_backend_meta_reduce with a row scatter behind it). The fixture is loaded twice, on the first device alone in
// layer mode and over every rank under -sm data, and each case decodes the same batch on both: the logits of every row
// are held to the fixture's tolerance. Cases: a prefill of one sequence a rank, a decode step after it, and a chunked
// prefill whose ubatches end mid-sequence.

#include "test-llama-archs.h"

#include "llama-archs-gguf.h"

#include "llama.h"

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr double tolerance = 1e-4; // the bar test-llama-archs-inkling holds a split's logits to

std::vector<float> decode_rows(llama_context * ctx, const std::vector<int32_t> & tokens,
        const std::vector<int32_t> & pos, const std::vector<int32_t> & seq) {
    GGML_ASSERT(tokens.size() == pos.size() && tokens.size() == seq.size());
    const uint32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx)));

    llama_batch batch = llama_batch_init(tokens.size(), 0, 1);
    for (size_t i = 0; i < tokens.size(); ++i) {
        batch_add(batch, tokens[i], pos[i], {seq[i]}, true);
    }
    const int32_t ret = llama_decode(ctx, batch);
    llama_batch_free(batch);
    if (ret != 0) {
        throw std::runtime_error("inkling2 data: decode failed with " + std::to_string(ret));
    }

    std::vector<float> out;
    out.reserve(tokens.size()*n_vocab);
    for (size_t i = 0; i < tokens.size(); ++i) {
        const float * row = llama_get_logits_ith(ctx, i);
        if (row == nullptr) {
            throw std::runtime_error("inkling2 data: row " + std::to_string(i) + " has no logits");
        }
        out.insert(out.end(), row, row + n_vocab);
    }
    return out;
}

llama_context_ptr make_context(llama_model * model, bool data, uint32_t n_seqs) {
    llama_context_params params = llama_context_default_params();
    // 1,024 rows at four ranks (256 tokens a sequence) in one ubatch: the served prefill step's width
    params.n_ctx           = 1024*n_seqs;
    params.n_batch         = 1024;
    params.n_ubatch        = 1024;
    params.n_seq_max       = n_seqs;
    params.kv_unified      = !data;
    params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED; // -sm data requires it, and both contexts compute alike
    params.n_threads       = params.n_threads_batch = 4;
    llama_context_ptr ctx(llama_init_from_model(model, params));
    if (!ctx) {
        throw std::runtime_error("inkling2 data: the context was not created");
    }
    return ctx;
}

} // namespace

int test_inkling_data(size_t seed, const std::vector<ggml_backend_dev_t> & devs) {
    if (devs.size() < 2) {
        fprintf(stderr, "inkling2 data: skipped, data parallelism needs at least two ranks\n");
        return 0;
    }

    const uint32_t n_seqs   = (uint32_t) devs.size();
    const uint32_t n_tokens = 8; // a sequence's prefill
    bool ok = true;

    auto gguf = get_gguf_ctx(LLM_ARCH_INKLING2, true);
    // the fixture's MoE carries two experts, and -sm data splits whole experts a rank: give it one expert a rank, so
    // four ranks load (llama-model.cpp:450 asserts the expert dim covers the ranks)
    if (devs.size() > 2) {
        gguf_set_val_u32(gguf.get(), llama_arch_kv_name(LLM_ARCH_INKLING2, LLM_KV_EXPERT_COUNT).c_str(),
                (uint32_t) devs.size());
    }

    llama_model_ptr reference = get_model(gguf.get(), nullptr, seed, 0.0f, {devs.front()}, LLAMA_SPLIT_MODE_LAYER);
    llama_model_ptr data      = get_model(gguf.get(), nullptr, seed, 0.0f, devs, LLAMA_SPLIT_MODE_DATA);
    const int vocab = (int) llama_vocab_n_tokens(llama_model_get_vocab(reference.get()));

    llama_context_ptr ref_ctx  = make_context(reference.get(), false, n_seqs);
    llama_context_ptr data_ctx = make_context(data.get(),      true,  n_seqs);

    auto run_case = [&](const char * name, uint32_t pos0, uint32_t n) {
        std::vector<int32_t> tokens, pos, seq;
        const std::vector<llama_token> drawn = get_tokens(n_seqs*n, vocab, seed + pos0);
        for (uint32_t s = 0; s < n_seqs; ++s) {
            for (uint32_t i = 0; i < n; ++i) {
                tokens.push_back(drawn[s*n + i]);
                pos.push_back((int32_t) (pos0 + i));
                seq.push_back((int32_t) s);
            }
        }
        const std::vector<float> expected = decode_rows(ref_ctx.get(),  tokens, pos, seq);
        const std::vector<float> actual   = decode_rows(data_ctx.get(), tokens, pos, seq);
        const double err = nmse(expected, actual);
        fprintf(stderr, "inkling2 data %-8s %u rows over %u ranks: nmse %.9g\n", name, (unsigned) tokens.size(),
                (unsigned) devs.size(), err);
        if (!std::isfinite(err) || err > tolerance) {
            ok = false;
        }
    };

    // the prefill: one sequence a rank, all rows output
    run_case("prefill", 0, n_tokens);
    // the decode step after it
    run_case("decode", n_tokens, 1);
    // another chunk, mid-sequence
    run_case("chunk", n_tokens + 1, 4);
    // the served step's width: 256 tokens a sequence in one ubatch (1,024 rows at four ranks, 512 at two)
    run_case("wide", n_tokens + 5, 256);

    fprintf(stderr, "%s inkling2 data\n", ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}
