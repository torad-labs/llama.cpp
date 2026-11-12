// A recurrent state saved and restored with llama_state_seq_get_data_async and llama_state_seq_set_data_async: its
// rows are copied before the call returns, since the next decode moves a sequence's state to another row or writes the
// row it had, a decode of other sequences included (with LLAMA_GRAPH_ANY_SEQS=1, the second registration of this test,
// a decode gathers its sequences at the first rows).
//
// Three sequences are decoded into one unified cache. Sequence 1's partial state, which is its recurrent state alone,
// is saved with the blocking call and then with the async one: no copy is pending for it when the call returns, and
// the bytes are the blocking ones before anything is waited for. Its full state is saved with the async call and the
// other sequences are decoded, alone and together, before the copy is waited for: the bytes are the blocking ones.
// The sequence is then removed and restored with the async call from the partial state and from the full one, the
// others decoding before the copy is waited for, and saved again: the state is the saved one, and no copy of a
// partial state is pending when its restore returns.

#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static const int n_seq = 3;

// a token that differs by sequence and position, so every state differs
static llama_token token_at(const llama_vocab * vocab, int seq, int pos) {
    return 1 + (seq*997 + pos*31) % (llama_vocab_n_tokens(vocab) - 1);
}

struct decoder {
    llama_context *     ctx;
    const llama_vocab * vocab;
    int                 pos[n_seq] = { 0, 0, 0 };

    // n more tokens of each of the sequences, in one batch
    bool run(const std::vector<int> & seqs, int n) {
        llama_batch batch = llama_batch_init(n*(int) seqs.size(), 0, 1);
        for (int i = 0; i < n; ++i) {
            for (const int s : seqs) {
                batch.token   [batch.n_tokens] = token_at(vocab, s, pos[s] + i);
                batch.pos     [batch.n_tokens] = pos[s] + i;
                batch.n_seq_id[batch.n_tokens] = 1;
                batch.seq_id  [batch.n_tokens][0] = s;
                batch.logits  [batch.n_tokens] = i + 1 == n;
                batch.n_tokens++;
            }
        }
        const bool ok = llama_decode(ctx, batch) == 0;
        llama_batch_free(batch);
        for (const int s : seqs) {
            pos[s] += n;
        }
        return ok;
    }
};

static std::vector<uint8_t> blocking(llama_context * ctx, llama_seq_id seq, llama_state_seq_flags flags) {
    std::vector<uint8_t> buf(llama_state_seq_get_size_ext(ctx, seq, flags));
    buf.resize(llama_state_seq_get_data_ext(ctx, buf.data(), buf.size(), seq, flags));
    return buf;
}

static bool same(const char * what, const char * step, const std::vector<uint8_t> & got, const std::vector<uint8_t> & want) {
    if (!want.empty() && got.size() == want.size() && memcmp(got.data(), want.data(), want.size()) == 0) {
        return true;
    }
    size_t i = 0;
    while (i < got.size() && i < want.size() && got[i] == want[i]) {
        ++i;
    }
    fprintf(stderr, "%s: FAIL, %s: %zu bytes against %zu, the first difference at byte %zu\n", what, step, got.size(), want.size(), i);
    return false;
}

static bool check(llama_model * model, uint32_t n_rs_seq) {
    const llama_state_seq_flags partial = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;
    const llama_state_seq_flags all     = LLAMA_STATE_SEQ_FLAGS_NONE;

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx      = 512;
    cparams.n_batch    = 512;
    cparams.n_ubatch   = 512;
    cparams.n_seq_max  = n_seq;
    cparams.n_rs_seq   = n_rs_seq;
    cparams.kv_unified = true;

    char what[64];
    snprintf(what, sizeof(what), "%u snapshots", n_rs_seq);

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (ctx == nullptr) {
        fprintf(stderr, "%s: failed to create the context\n", what);
        return false;
    }
    llama_memory_t mem = llama_get_memory(ctx);
    decoder dec = { ctx, llama_model_get_vocab(model) };
    bool ok = false;

    std::vector<uint8_t> ref_partial, ref_all, buf;

    if (!dec.run({ 0, 1, 2 }, 12)) {
        fprintf(stderr, "%s: decode failed\n", what);
        goto done;
    }
    ref_partial = blocking(ctx, 1, partial);
    ref_all     = blocking(ctx, 1, all);

    // a partial state is the recurrent one alone: it is in the buffer when the call returns
    buf.assign(ref_partial.size(), 0xa5);
    if (llama_state_seq_get_data_async(ctx, buf.data(), buf.size(), 1, partial) != buf.size() ||
            llama_state_seq_async_pending(ctx, 1)) {
        fprintf(stderr, "%s: FAIL, a copy of the partial state is pending after its save returned\n", what);
        goto done;
    }
    if (!same(what, "the partial state when its async save returned", buf, ref_partial)) {
        goto done;
    }

    // a full state, the other sequences decoded before the copy is waited for
    buf.assign(ref_all.size(), 0xa5);
    if (llama_state_seq_get_data_async(ctx, buf.data(), buf.size(), 1, all) != buf.size()) {
        fprintf(stderr, "%s: FAIL, the async save of the full state returned another size\n", what);
        goto done;
    }
    if (!dec.run({ 2 }, 3) || !dec.run({ 0, 2 }, 3) || !dec.run({ 0 }, 2)) {
        fprintf(stderr, "%s: decode beside the save failed\n", what);
        goto done;
    }
    llama_state_seq_async_wait(ctx, 1);
    if (!same(what, "the full state saved beside decodes of the other sequences", buf, ref_all)) {
        goto done;
    }

    // the sequence removed and restored from the partial state: nothing is pending when the restore returns
    llama_memory_seq_rm(mem, 1, -1, -1);
    if (!dec.run({ 0, 2 }, 1)) {
        fprintf(stderr, "%s: decode after the removal failed\n", what);
        goto done;
    }
    if (llama_state_seq_set_data_async(ctx, ref_all.data(), ref_all.size(), 1, all) != ref_all.size()) {
        fprintf(stderr, "%s: FAIL, the async restore of the full state failed\n", what);
        goto done;
    }
    if (!dec.run({ 2 }, 2) || !dec.run({ 0, 2 }, 2) || !dec.run({ 0 }, 1)) {
        fprintf(stderr, "%s: decode beside the restore failed\n", what);
        goto done;
    }
    if (!same(what, "the full state restored beside decodes of the other sequences", blocking(ctx, 1, all), ref_all)) {
        goto done;
    }
    if (llama_state_seq_set_data_async(ctx, ref_partial.data(), ref_partial.size(), 1, partial) != ref_partial.size() ||
            llama_state_seq_async_pending(ctx, 1)) {
        fprintf(stderr, "%s: FAIL, a copy of the partial state is pending after its restore returned\n", what);
        goto done;
    }
    if (!dec.run({ 2 }, 1) || !dec.run({ 0, 2 }, 1)) {
        fprintf(stderr, "%s: decode after the partial restore failed\n", what);
        goto done;
    }
    if (!same(what, "the partial state restored", blocking(ctx, 1, partial), ref_partial)) {
        goto done;
    }

    fprintf(stderr, "%s: ok, %zu and %zu bytes\n", what, ref_partial.size(), ref_all.size());
    ok = true;

done:
    llama_free(ctx);
    return ok;
}

int main(int argc, char ** argv) {
    const char * path = nullptr;
    for (int i = 1; i + 1 < argc; ++i) {
        if (strcmp(argv[i], "-m") == 0) {
            path = argv[i + 1];
        }
    }
    if (path == nullptr) {
        fprintf(stderr, "usage: %s -m model.gguf\n", argv[0]);
        return 1;
    }

    llama_backend_init();
    ggml_backend_load_all();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 99;
    llama_model * model = llama_model_load_from_file(path, mparams);
    if (model == nullptr) {
        fprintf(stderr, "failed to load %s\n", path);
        return 1;
    }

    int n_fail = 0;
    for (const uint32_t n_rs_seq : { 0u, 2u }) {
        n_fail += check(model, n_rs_seq) ? 0 : 1;
    }

    llama_model_free(model);
    llama_backend_free();

    fprintf(stderr, "%s\n", n_fail == 0 ? "all async copies of a recurrent state match" : "FAILED");
    return n_fail == 0 ? 0 : 1;
}
