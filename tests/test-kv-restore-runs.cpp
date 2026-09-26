// A sequence's state saved, removed and restored into a unified cache where its cells lie between other sequences'
// cells: the restore writes a run of cells that follow each other in the cache as one copy, and the state saved again
// must be the state saved before, byte for byte (the same positions in the same cells, the same K and V rows).
//
// Three sequences are decoded in turns of `turn` tokens, so their cells interleave in runs of `turn`. Sequence 1 is saved
// and removed; with `refill`, sequence 0 then takes the first holes, so the restore fills the holes left and then cells
// past every sequence (the runs change length). Each cache layout the restore writes differently: V transposed (no
// flash attention, f16), V in rows (flash attention, f16), and the served q4_0 rows (flash attention).
// LLAMA_KV_RESTORE_PER_CELL_LEGACY=1 runs the same checks on the per-cell copies.

#include "llama.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <vector>

static const int n_seq = 3;
static const int n_tok = 60; // tokens per sequence

static void add(llama_batch & batch, llama_token token, llama_pos pos, llama_seq_id seq) {
    batch.token   [batch.n_tokens] = token;
    batch.pos     [batch.n_tokens] = pos;
    batch.n_seq_id[batch.n_tokens] = 1;
    batch.seq_id  [batch.n_tokens][0] = seq;
    batch.logits  [batch.n_tokens] = false;
    batch.n_tokens++;
}

// a token that differs by sequence and position, so every cell holds different K and V rows
static llama_token token_at(const llama_vocab * vocab, int seq, int pos) {
    return 1 + (seq*997 + pos*31) % (llama_vocab_n_tokens(vocab) - 1);
}

static bool check(llama_model * model, bool fa, ggml_type type, int turn, bool refill) {
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx           = 512;
    cparams.n_batch         = 512;
    cparams.n_ubatch        = 512;
    cparams.n_seq_max       = n_seq;
    cparams.kv_unified      = true;
    cparams.flash_attn_type = fa ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cparams.type_k          = type;
    cparams.type_v          = type;

    char what[128];
    snprintf(what, sizeof(what), "fa %d, %s, turns of %d%s", fa, ggml_type_name(type), turn, refill ? ", holes refilled" : "");

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (ctx == nullptr) {
        fprintf(stderr, "%s: failed to create the context\n", what);
        return false;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    llama_memory_t mem = llama_get_memory(ctx);
    llama_batch batch = llama_batch_init(n_seq*n_tok, 0, 1);
    bool ok = false;

    // the sequences in turns: 0 0 1 1 2 2 0 0 1 1 2 2 ... for turns of 2, one batch, cells in batch order
    for (int p0 = 0; p0 < n_tok; p0 += turn) {
        for (int s = 0; s < n_seq; ++s) {
            for (int p = p0; p < std::min(n_tok, p0 + turn); ++p) {
                add(batch, token_at(vocab, s, p), p, s);
            }
        }
    }
    batch.logits[batch.n_tokens - 1] = true;

    std::vector<uint8_t> saved, again;
    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "%s: decode failed\n", what);
        goto done;
    }

    saved.resize(llama_state_seq_get_size(ctx, 1));
    if (llama_state_seq_get_data(ctx, saved.data(), saved.size(), 1) != saved.size()) {
        fprintf(stderr, "%s: saving sequence 1 failed\n", what);
        goto done;
    }
    llama_memory_seq_rm(mem, 1, -1, -1);

    if (refill) {
        // sequence 0 takes 2 turns more: its cells go to the first holes sequence 1 left
        batch.n_tokens = 0;
        for (int p = n_tok; p < n_tok + 2*turn; ++p) {
            add(batch, token_at(vocab, 0, p), p, 0);
        }
        batch.logits[batch.n_tokens - 1] = true;
        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "%s: refill decode failed\n", what);
            goto done;
        }
    }

    if (llama_state_seq_set_data(ctx, saved.data(), saved.size(), 1) != saved.size()) {
        fprintf(stderr, "%s: restoring sequence 1 failed\n", what);
        goto done;
    }
    again.resize(llama_state_seq_get_size(ctx, 1));
    if (llama_state_seq_get_data(ctx, again.data(), again.size(), 1) != again.size()) {
        fprintf(stderr, "%s: saving sequence 1 again failed\n", what);
        goto done;
    }
    if (again.size() != saved.size() || memcmp(again.data(), saved.data(), saved.size()) != 0) {
        size_t i = 0;
        while (i < std::min(again.size(), saved.size()) && again[i] == saved[i]) {
            ++i;
        }
        fprintf(stderr, "%s: FAIL, the restored state differs from the saved one (%zu and %zu bytes, first at byte %zu)\n",
                what, saved.size(), again.size(), i);
        goto done;
    }
    fprintf(stderr, "%s: ok, %zu bytes\n", what, saved.size());
    ok = true;

done:
    llama_batch_free(batch);
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

    const struct { bool fa; ggml_type type; } layouts[] = {
        { false, GGML_TYPE_F16  }, // V transposed
        { true,  GGML_TYPE_F16  }, // V in rows
        { true,  GGML_TYPE_Q4_0 }, // the served cache
    };
    int n_fail = 0;
    for (const auto & layout : layouts) {
        for (const int turn : { 1, 5 }) {
            for (const bool refill : { false, true }) {
                n_fail += check(model, layout.fa, layout.type, turn, refill) ? 0 : 1;
            }
        }
    }

    llama_model_free(model);
    llama_backend_free();

    fprintf(stderr, "%s\n", n_fail == 0 ? "all restores match" : "FAILED");
    return n_fail == 0 ? 0 : 1;
}
