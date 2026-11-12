// A sequence's recurrent state copied through a spare row on the devices (LLAMA_RS_STAGE_ROWS and
// llama_state_seq_get_data_staged): the bytes are the ones the blocking copy gives, whatever is decoded before the copy
// from the spare row is waited for.
//
// Three sequences are decoded into one unified cache. Sequence 1's partial state is taken with the blocking call and
// then with the staged one. Before the staged copy is waited for, the other two sequences are decoded alone and
// together, which moves sequence 1's row or writes over the row it had (with LLAMA_GRAPH_ANY_SEQS=1, the second
// registration of this test, a decode gathers its sequences at the first rows), and then sequence 1 itself is decoded
// on: the staged bytes must be the blocking ones taken before all of it. Both spare rows are then taken in one go
// with a third call behind them, a call with row == NULL waits for its own copy, and a full state goes the blocking
// way. Each check runs without and with rollback snapshots (n_rs_seq), where a state's row lies in one of several
// planes, and with them once more on a state a rollback is pending for. A context without spare rows gives the
// blocking copy and no row.

#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static const int n_seq   = 3;
static const int n_stage = 2;

static void set_env(const char * name, const char * value) {
#ifdef _WIN32
    _putenv_s(name, value != nullptr ? value : "");
#else
    if (value != nullptr) {
        setenv(name, value, 1);
    } else {
        unsetenv(name);
    }
#endif
}

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

static const llama_state_seq_flags partial = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;

// the blocking copy of a sequence's partial state
static std::vector<uint8_t> blocking(llama_context * ctx, llama_seq_id seq) {
    std::vector<uint8_t> buf(llama_state_seq_get_size_ext(ctx, seq, partial));
    buf.resize(llama_state_seq_get_data_ext(ctx, buf.data(), buf.size(), seq, partial));
    return buf;
}

// the staged copy of it started: the bytes are in buf once its row is waited for
static bool staged(llama_context * ctx, llama_seq_id seq, std::vector<uint8_t> & buf, int32_t * row) {
    buf.assign(llama_state_seq_get_size_ext(ctx, seq, partial), 0xa5);
    return llama_state_seq_get_data_staged(ctx, buf.data(), buf.size(), seq, partial, row) == buf.size();
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

static bool check(llama_model * model, uint32_t n_rs_seq, bool spare) {
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx      = 512;
    cparams.n_batch    = 512;
    cparams.n_ubatch   = 512;
    cparams.n_seq_max  = n_seq;
    cparams.n_rs_seq   = n_rs_seq;
    cparams.kv_unified = true;

    char what[96];
    snprintf(what, sizeof(what), "%u snapshots, %s", n_rs_seq, spare ? "spare rows" : "no spare rows");

    char rows[16];
    snprintf(rows, sizeof(rows), "%d", n_stage);
    set_env("LLAMA_RS_STAGE_ROWS", spare ? rows : nullptr);
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (ctx == nullptr) {
        fprintf(stderr, "%s: failed to create the context\n", what);
        return false;
    }
    decoder dec = { ctx, llama_model_get_vocab(model) };
    bool ok = false;

    std::vector<uint8_t> ref, later, a, b, c, full, full_ref;
    int32_t row   = -1;
    int32_t row_b = -1;
    int32_t row_c = -1;

    if (llama_state_staged_rows(ctx) != (spare ? (uint32_t) n_stage : 0)) {
        fprintf(stderr, "%s: FAIL, the context has %u spare rows\n", what, llama_state_staged_rows(ctx));
        goto done;
    }
    if (!dec.run({ 0, 1, 2 }, 12)) {
        fprintf(stderr, "%s: decode failed\n", what);
        goto done;
    }
    ref = blocking(ctx, 1);

    // the other sequences decoded before the copy is waited for
    if (!staged(ctx, 1, a, &row) || (row >= 0) != spare) {
        fprintf(stderr, "%s: FAIL, the staged copy took row %d\n", what, row);
        goto done;
    }
    if (!dec.run({ 2 }, 3) || !dec.run({ 0, 2 }, 3) || !dec.run({ 0 }, 2)) {
        fprintf(stderr, "%s: decode beside the copy failed\n", what);
        goto done;
    }
    llama_state_staged_wait(ctx, row);
    if (llama_state_staged_pending(ctx, row)) {
        fprintf(stderr, "%s: FAIL, the copy is pending after the wait for it\n", what);
        goto done;
    }
    if (!same(what, "the state copied beside decodes of the other sequences", a, ref) ||
        !same(what, "the state after decodes of the other sequences", blocking(ctx, 1), ref)) {
        goto done;
    }

    // the sequence itself decoded on before the copy is waited for
    if (!staged(ctx, 1, a, &row) || (row >= 0) != spare) {
        fprintf(stderr, "%s: FAIL, the second staged copy took row %d\n", what, row);
        goto done;
    }
    if (!dec.run({ 1 }, 4) || !dec.run({ 1, 2 }, 2)) {
        fprintf(stderr, "%s: decode of the copied sequence failed\n", what);
        goto done;
    }
    llama_state_staged_wait(ctx, row);
    if (!same(what, "the state copied before its sequence was decoded on", a, ref)) {
        goto done;
    }
    later = blocking(ctx, 1);
    if (later.size() == ref.size() && memcmp(later.data(), ref.data(), ref.size()) == 0) {
        fprintf(stderr, "%s: FAIL, the state did not change when its sequence was decoded on\n", what);
        goto done;
    }

    if (n_rs_seq > 0) {
        // a state a rollback is pending for is read from a snapshot's plane
        if (!dec.run({ 1 }, 4)) {
            fprintf(stderr, "%s: decode before the rollback failed\n", what);
            goto done;
        }
        if (!llama_memory_seq_rm(llama_get_memory(ctx), 1, dec.pos[1] - 1, -1)) {
            fprintf(stderr, "%s: FAIL, the rollback of one token was refused\n", what);
            goto done;
        }
        dec.pos[1] -= 1;
        later = blocking(ctx, 1);
    }

    // every spare row taken, and one call more
    if (!staged(ctx, 1, a, &row) || !staged(ctx, 1, b, &row_b) || !staged(ctx, 1, c, &row_c)) {
        fprintf(stderr, "%s: FAIL, a staged copy returned another size\n", what);
        goto done;
    }
    if (!dec.run({ 0 }, 1)) {
        fprintf(stderr, "%s: decode beside three copies failed\n", what);
        goto done;
    }
    llama_state_staged_wait(ctx, row);
    llama_state_staged_wait(ctx, row_b);
    llama_state_staged_wait(ctx, row_c);
    if (!same(what, "the first of three copies", a, later) || !same(what, "the second of three copies", b, later) ||
        !same(what, "the third of three copies", c, later)) {
        goto done;
    }

    // without a row to give back, the call waits for its copy
    if (!staged(ctx, 1, a, nullptr) || llama_state_seq_async_pending(ctx, -1)) {
        fprintf(stderr, "%s: FAIL, a copy is pending after the call that waits\n", what);
        goto done;
    }
    if (!same(what, "the copy the call waited for", a, later)) {
        goto done;
    }

    // a full state is not staged
    full_ref.resize(llama_state_seq_get_size(ctx, 1));
    full_ref.resize(llama_state_seq_get_data(ctx, full_ref.data(), full_ref.size(), 1));
    full.resize(full_ref.size());
    row = 0;
    if (llama_state_seq_get_data_staged(ctx, full.data(), full.size(), 1, LLAMA_STATE_SEQ_FLAGS_NONE, &row) != full.size() || row != -1) {
        fprintf(stderr, "%s: FAIL, a full state took row %d\n", what, row);
        goto done;
    }
    if (!same(what, "the full state", full, full_ref)) {
        goto done;
    }

    fprintf(stderr, "%s: ok, %zu bytes\n", what, ref.size());
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
        for (const bool spare : { true, false }) {
            n_fail += check(model, n_rs_seq, spare) ? 0 : 1;
        }
    }

    llama_model_free(model);
    llama_backend_free();

    fprintf(stderr, "%s\n", n_fail == 0 ? "all staged copies match" : "FAILED");
    return n_fail == 0 ? 0 : 1;
}
