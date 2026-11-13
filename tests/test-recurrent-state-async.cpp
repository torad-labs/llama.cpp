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
//
// check_dropped: a placement that leaves a state to move and is followed by no move (what a state restore did until
// e7598775e) drops the sequences it was for, the one that kept its row included, and the one it displaced.
// llama_decode answers -4 for a batch that holds one, no state of it and no state of the whole context is saved, a
// copy of one is dropped too, the sequence past the placement keeps its state and decodes, and a dropped sequence
// that is removed starts again.

#include "llama.h"

#include "../src/llama-batch.h"
#include "../src/llama-memory-hybrid-iswa.h"
#include "../src/llama-memory-hybrid.h"
#include "../src/llama-memory-recurrent.h"
#include "../src/llama-model.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static const int n_seq = 3;
static const int n_seq_dropped = 4; // check_dropped's

// a token that differs by sequence and position, so every state differs
static llama_token token_at(const llama_vocab * vocab, int seq, int pos) {
    return 1 + (seq*997 + pos*31) % (llama_vocab_n_tokens(vocab) - 1);
}

struct decoder {
    llama_context *     ctx;
    const llama_vocab * vocab;
    int                 pos[n_seq_dropped] = { 0, 0, 0, 0 };

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

    // what llama_decode answers for one more token of each of the sequences; nothing is counted as decoded
    int32_t probe(const std::vector<int> & seqs) {
        llama_batch batch = llama_batch_init((int) seqs.size(), 0, 1);
        for (const int s : seqs) {
            batch.token   [batch.n_tokens] = token_at(vocab, s, pos[s]);
            batch.pos     [batch.n_tokens] = pos[s];
            batch.n_seq_id[batch.n_tokens] = 1;
            batch.seq_id  [batch.n_tokens][0] = s;
            batch.logits  [batch.n_tokens] = 1;
            batch.n_tokens++;
        }
        const int32_t ret = llama_decode(ctx, batch);
        llama_batch_free(batch);
        return ret;
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

static llama_memory_recurrent * recurrent_of(llama_memory_t mem) {
    if (auto * m = dynamic_cast<llama_memory_recurrent *>(mem)) {
        return m;
    }
    if (auto * m = dynamic_cast<llama_memory_hybrid *>(mem)) {
        return m->get_mem_recr();
    }
    if (auto * m = dynamic_cast<llama_memory_hybrid_iswa *>(mem)) {
        return m->get_mem_recr();
    }
    return nullptr;
}

static bool check_dropped(llama_model * model) {
    const char * what = "an owed move";
    const llama_state_seq_flags partial = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;
    const llama_state_seq_flags all     = LLAMA_STATE_SEQ_FLAGS_NONE;

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx      = 512;
    cparams.n_batch    = 512;
    cparams.n_ubatch   = 512;
    cparams.n_seq_max  = n_seq_dropped;
    cparams.n_rs_seq   = 2;
    cparams.kv_unified = true;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (ctx == nullptr) {
        fprintf(stderr, "%s: failed to create the context\n", what);
        return false;
    }
    llama_memory_t mem = llama_get_memory(ctx);
    llama_memory_recurrent * recr = recurrent_of(mem);
    decoder dec = { ctx, llama_model_get_vocab(model) };
    bool ok = false;

    std::vector<uint8_t> ref_3, buf;
    llama_batch_allocr balloc(model->hparams.n_pos_per_embd());
    llama_seq_id seq_0 = 0;
    llama_seq_id seq_2 = 2;

    if (recr == nullptr || !recr->relocate) {
        // a decode gathers its sequences only with LLAMA_GRAPH_ANY_SEQS=1: elsewhere no placement owes a move
        fprintf(stderr, "%s: no decode moves a state here, nothing to check\n", what);
        ok = true;
        goto done;
    }
    if (!dec.run({ 0, 1, 2, 3 }, 12)) {
        fprintf(stderr, "%s: decode failed\n", what);
        goto done;
    }
    ref_3 = blocking(ctx, 3, partial);

    // Sequences 0 and 2 placed for one more token, as a decode's first step places them, and nothing after. 0 keeps
    // its row, at a position no decode took it to; 2 takes the second row; 1 is displaced to the row 2 had and its
    // move is left owed; 3 lies past the placement's cells
    {
        llama_ubatch ubatch = balloc.ubatch_reserve(1, 2);
        ubatch.pos[0]      = dec.pos[0];
        ubatch.pos[1]      = dec.pos[2];
        ubatch.n_seq_id[0] = 1;
        ubatch.n_seq_id[1] = 1;
        ubatch.seq_id[0]   = &seq_0;
        ubatch.seq_id[1]   = &seq_2;
        if (!recr->find_slot(ubatch) || !recr->relocate_pending) {
            fprintf(stderr, "%s: FAIL, placing sequences 0 and 2 left no state to move\n", what);
            goto done;
        }
    }

    // the decode that finds the move owed refuses the sequence that kept its row, before it asks anything else
    if (dec.probe({ 0 }) != -4) {
        fprintf(stderr, "%s: FAIL, a decode of a sequence placed for a decode that was never made did not answer -4\n", what);
        goto done;
    }
    if (!llama_memory_seq_lost(mem, 0) || !llama_memory_seq_lost(mem, 1) || !llama_memory_seq_lost(mem, 2) ||
            llama_memory_seq_lost(mem, 3) || !llama_memory_seq_lost(mem, -1)) {
        fprintf(stderr, "%s: FAIL, dropped are %d %d %d %d of sequences 0 to 3, expected 1 1 1 0\n", what,
                llama_memory_seq_lost(mem, 0), llama_memory_seq_lost(mem, 1), llama_memory_seq_lost(mem, 2), llama_memory_seq_lost(mem, 3));
        goto done;
    }
    if (dec.probe({ 1 }) != -4 || dec.probe({ 2 }) != -4 || dec.probe({ 2, 3 }) != -4) {
        fprintf(stderr, "%s: FAIL, a batch that holds a dropped sequence did not answer -4\n", what);
        goto done;
    }
    buf.assign(ref_3.size(), 0xa5);
    if (llama_state_seq_get_size_ext(ctx, 0, all) != 0 || llama_state_seq_get_data_ext(ctx, buf.data(), buf.size(), 2, partial) != 0 ||
            llama_state_seq_get_data_async(ctx, buf.data(), buf.size(), 1, partial) != 0) {
        fprintf(stderr, "%s: FAIL, a state of a dropped sequence was saved\n", what);
        goto done;
    }
    // the whole context is not saved either: a load would bring the dropped sequences back without their state
    if (llama_state_get_size(ctx) != 0 || llama_state_get_data(ctx, buf.data(), buf.size()) != 0) {
        fprintf(stderr, "%s: FAIL, the context was saved with dropped sequences in it\n", what);
        goto done;
    }

    // the sequence past the placement's cells was not part of it: its state is the one it had, and it decodes
    if (!same(what, "the state of the sequence that was not displaced", blocking(ctx, 3, partial), ref_3)) {
        goto done;
    }
    if (!dec.run({ 3 }, 2)) {
        fprintf(stderr, "%s: FAIL, the sequence that was not displaced does not decode\n", what);
        goto done;
    }

    // a copy of a dropped sequence is dropped, also into a sequence that was removed
    llama_memory_seq_rm(mem, 1, -1, -1);
    if (llama_memory_seq_lost(mem, 1)) {
        fprintf(stderr, "%s: FAIL, a dropped sequence that was removed is still dropped\n", what);
        goto done;
    }
    llama_memory_seq_cp(mem, 0, 1, -1, -1);
    if (!llama_memory_seq_lost(mem, 1) || dec.probe({ 1 }) != -4) {
        fprintf(stderr, "%s: FAIL, a copy of a dropped sequence is taken as a sequence with a state\n", what);
        goto done;
    }

    // a dropped sequence that is removed starts again from its first token; the others stay dropped until then
    llama_memory_seq_rm(mem, 0, -1, -1);
    dec.pos[0] = 0;
    if (llama_memory_seq_lost(mem, 0) || !llama_memory_seq_lost(mem, 2) || !dec.run({ 0 }, 4) || !dec.run({ 0, 3 }, 2)) {
        fprintf(stderr, "%s: FAIL, a dropped sequence that was removed does not start again\n", what);
        goto done;
    }
    for (const llama_seq_id seq : { 1, 2 }) {
        llama_memory_seq_rm(mem, seq, -1, -1);
        dec.pos[seq] = 0;
    }
    if (llama_memory_seq_lost(mem, -1) || !dec.run({ 1, 2 }, 4) || !dec.run({ 0, 1, 2, 3 }, 2)) {
        fprintf(stderr, "%s: FAIL, the other dropped sequences do not start again\n", what);
        goto done;
    }
    if (blocking(ctx, 0, all).empty() || blocking(ctx, 2, all).empty() || llama_state_get_size(ctx) == 0) {
        fprintf(stderr, "%s: FAIL, a sequence that started again, or the context, has no state to save\n", what);
        goto done;
    }

    fprintf(stderr, "%s: ok, three sequences dropped and refused, a copy of one too, one kept, all started again\n", what);
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
    n_fail += check_dropped(model) ? 0 : 1;

    llama_model_free(model);
    llama_backend_free();

    fprintf(stderr, "%s\n", n_fail == 0 ? "all async copies of a recurrent state match" : "FAILED");
    return n_fail == 0 ? 0 : 1;
}
