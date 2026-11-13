// llama-topk-overlap: reads a prompt as the server reads it, step by step, then with -n N generates up to N tokens, one
// a step, and says two things for every DSA layer of every step (topk-overlap.h). Both are properties of the trained
// weights, so they are read here on the real model.
//
// - How much adjacent rows' indexer selections overlap: the KV a tile of 2, 4, 8 or 16 rows gathers as one union, as a
//   multiple of what one row gathers alone. Query-tiled sparse attention is worth building only where that multiple
//   stays well under the tile's size.
// - How much of the step's selection the steps before it selected: the pools it selects that a hot cache of the
//   layer's last selections would not hold, for a cache of 1, 2, 4, 8 and 16 selections. Row-level tiering (the latents
//   in host memory, the lately selected ones on the card) is worth building only where that part is small.
//
//   llama-topk-overlap -m MODEL -f TEXT [-n N] [the server's flags: -sm, -ngl, -fa, -b, -ub, -c, cache types, sampling]
//
// TEXT is tokenized with its special tokens, so a rendered chat prompt is read as the server reads it. TOPK_TOKENS caps
// the prompt at that many tokens (default: all of TEXT). One line a layer a step goes to stdout, then the summaries
// and the generated text. TOPK_LOG=FILE writes the selections of the generated steps as they are, for questions the
// summaries do not answer: a record a layer a step, of int32s: step, position, layer, rows, select_k, live pools, then
// rows*select_k pool ids. Pool p is the cells of positions [p*r, p*r + r) of the one sequence this reads.

#include "arg.h"
#include "common.h"
#include "log.h"
#include "sampling.h"
#include "llama.h"

#include "ggml.h"
#include "ggml-backend.h"

#include "topk-overlap.h"

#include <algorithm>
#include <clocale>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

static const int64_t TILES[] = {2, 4, 8, 16};
static const int64_t CAPS[]  = {1, 2, 4, 8, 16}; // a hot cache's size, in selections of full_k pools

// what the hot caches of a layer counted: the steps, the pools they selected, and of those the misses at each size
struct lru_count {
    int64_t steps = 0, selected = 0, misses[5] = {0, 0, 0, 0, 0};
};

struct layer_total {
    topk_tile_stats       st[4];
    int64_t               steps = 0;
    std::vector<topk_lru> lru;         // one a size of CAPS
    int64_t               born   = 0;  // the first pool no step has completed yet
    lru_count             prompt;      // what the caches had counted when the prompt ended

    lru_count count() const {
        lru_count c;
        for (size_t i = 0; i < lru.size(); ++i) {
            c.steps     = lru[i].steps;
            c.selected  = lru[i].selected;
            c.misses[i] = lru[i].misses;
        }
        return c;
    }
};

struct overlap_state {
    int                        step    = -1;
    int                        last_il = 1 << 30;
    int64_t                    pos0    = 0;
    int64_t                    r       = 0;  // cells a pool, 0: not a pooled indexer, no hot cache is followed
    int64_t                    full_k  = 0;  // pools a row selects once the cache holds that many
    bool                       generating = false;
    FILE *                     log     = nullptr;
    std::map<int, layer_total> layers;
    std::vector<int32_t>       buf;
    bool                       warned  = false;
    bool                       warned_ids = false;
};

// the live pools the top-k chose among: its input is [the pools' scores, the dump pools], concatenated
static int64_t live_pools(const ggml_tensor * t) {
    const ggml_tensor * k = t->op == GGML_OP_TOP_K ? t : t->src[0];
    if (!k || k->op != GGML_OP_TOP_K) return -1;
    const ggml_tensor * c = k->src[0];
    if (!c || c->op != GGML_OP_CONCAT || !c->src[0]) return -1;
    return c->src[0]->ne[0];
}

static bool on_node(ggml_tensor * t, bool ask, void * user_data) {
    auto * st = (overlap_state *) user_data;
    const int il = topk_selection_layer(t->name);
    if (il < 0) return false;
    if (ask) return true;

    if (il <= st->last_il) st->step++;
    st->last_il = il;

    if (t->type != GGML_TYPE_I32) {
        LOG_ERR("%s is %s, not i32: not read\n", t->name, ggml_type_name(t->type));
        return true;
    }
    const int64_t select_k = t->ne[0];
    const int64_t n_rows   = t->ne[1]*t->ne[2]*t->ne[3];
    int64_t       n_live   = live_pools(t);
    if (n_live < 0) {
        if (!st->warned) LOG_WRN("%s: the top-k's input is not [scores, dump pools]; every id counted as live\n", t->name);
        st->warned = true;
        n_live = INT32_MAX;
    }
    st->buf.resize(ggml_nelements(t));
    ggml_backend_tensor_get(t, st->buf.data(), 0, ggml_nbytes(t));

    std::printf("step %2d pos %6lld rows %5lld layer %2d pools %6lld select %4lld |", st->step, (long long) st->pos0,
            (long long) n_rows, il, (long long) n_live, (long long) select_k);
    layer_total & lt = st->layers[il];
    const bool sparse = n_live > select_k;
    for (int i = 0; i < 4; ++i) {
        const topk_tile_stats s = topk_tile_union(st->buf.data(), select_k, n_rows, n_live, TILES[i]);
        std::printf(" t%lld %6.3f", (long long) TILES[i], s.mean());
        if (sparse) {
            lt.st[i].tiles        += s.tiles;
            lt.st[i].sum_ratio    += s.sum_ratio;
            lt.st[i].rows_skipped += s.rows_skipped;
        }
    }
    if (sparse) lt.steps++;

    if (st->r > 0) {
        if (lt.lru.empty()) {
            for (const int64_t cap : CAPS) lt.lru.emplace_back(cap*st->full_k);
        }
        // the pools whole at the step's end: a row selects among the pools whole before it
        const int64_t whole = (st->pos0 + n_rows)/st->r;
        int64_t top = -1;
        for (const int32_t id : st->buf) {
            if (id < n_live) top = std::max<int64_t>(top, id);
        }
        if (top >= whole && !st->warned_ids) {
            LOG_WRN("%s: pool %lld selected at position %lld, where %lld pools are whole: pool ids are not position "
                    "blocks here, and the hot caches count pools as missed that the step wrote\n", t->name,
                    (long long) top, (long long) st->pos0, (long long) whole);
            st->warned_ids = true;
        }
        std::printf(" | miss");
        for (size_t i = 0; i < lt.lru.size(); ++i) {
            const int64_t before = lt.lru[i].selected;
            lt.lru[i].born(lt.born, whole);
            const int64_t miss = lt.lru[i].step(st->buf.data(), select_k, n_rows, n_live);
            if (miss < 0) {
                std::printf(" x%lld      -", (long long) CAPS[i]);
            } else {
                std::printf(" x%lld %6.3f", (long long) CAPS[i], (double) miss/(double) (lt.lru[i].selected - before));
            }
        }
        lt.born = std::max(lt.born, whole);
    }
    std::printf("%s\n", sparse ? "" : "  (dense: every live pool fits the top-k)");
    std::fflush(stdout);

    if (st->log && st->generating) {
        const int32_t head[6] = {(int32_t) st->step, (int32_t) st->pos0, (int32_t) il, (int32_t) n_rows,
                (int32_t) select_k, (int32_t) std::min<int64_t>(n_live, INT32_MAX)};
        std::fwrite(head, sizeof(head), 1, st->log);
        std::fwrite(st->buf.data(), sizeof(int32_t), st->buf.size(), st->log);
    }
    return true;
}

static int64_t meta_int(const llama_model * model, const std::string & key) {
    char buf[64];
    return llama_model_meta_val_str(model, key.c_str(), buf, sizeof(buf)) > 0 ? std::atoll(buf) : 0;
}

// the misses of the hot caches over the steps counted between two counts, a line a layer and one for all of them
static void print_misses(const overlap_state & st, bool generated) {
    lru_count all;
    for (const auto & [il, lt] : st.layers) {
        const lru_count end = lt.count();
        lru_count c = generated ? end : lt.prompt;
        if (generated) {
            c.steps    -= lt.prompt.steps;
            c.selected -= lt.prompt.selected;
            for (int i = 0; i < 5; ++i) c.misses[i] -= lt.prompt.misses[i];
        }
        if (c.steps == 0) continue;
        std::printf("layer %2d, %lld steps, %.1f pools a step |", il, (long long) c.steps, (double) c.selected/(double) c.steps);
        for (int i = 0; i < 5; ++i) {
            std::printf(" x%lld %5.3f (%.1f)", (long long) CAPS[i], (double) c.misses[i]/(double) c.selected,
                    (double) c.misses[i]/(double) c.steps);
            all.misses[i] += c.misses[i];
        }
        std::printf("\n");
        all.steps    += c.steps;
        all.selected += c.selected;
    }
    if (all.steps == 0) {
        std::printf("no step counted: no row selected %lld live pools\n", (long long) st.full_k);
        return;
    }
    std::printf("all layers, %.1f pools a step |", (double) all.selected/(double) all.steps);
    for (int i = 0; i < 5; ++i) {
        std::printf(" x%lld %5.3f (%.1f)", (long long) CAPS[i], (double) all.misses[i]/(double) all.selected,
                (double) all.misses[i]/(double) all.steps);
    }
    std::printf("\n");
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) return 1;

    llama_backend_init();
    llama_numa_init(params.numa);

    overlap_state st;
    params.cb_eval           = on_node;
    params.cb_eval_user_data = &st;
    params.warmup            = false;

    auto   llama_init = common_init_from_params(params);
    auto * model      = llama_init->model();
    auto * ctx        = llama_init->context();
    if (model == nullptr || ctx == nullptr) {
        LOG_ERR("%s: failed to init\n", __func__);
        return 1;
    }

    const llama_vocab *      vocab  = llama_model_get_vocab(model);
    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, llama_vocab_get_add_bos(vocab), true);
    if (const char * cap = std::getenv("TOPK_TOKENS")) {
        const size_t n = (size_t) std::atoll(cap);
        if (n > 0 && n < tokens.size()) tokens.resize(n);
    }
    if (tokens.empty()) {
        LOG_ERR("%s: no prompt tokens (-f TEXT)\n", __func__);
        return 1;
    }
    const int n_predict = std::max(params.n_predict, 0);
    if (tokens.size() + (size_t) n_predict > llama_n_ctx(ctx)) {
        LOG_ERR("%s: %zu prompt tokens and %d to generate do not fit the context of %u (-c)\n", __func__, tokens.size(),
                n_predict, llama_n_ctx(ctx));
        return 1;
    }

    {
        char arch[64] = "";
        llama_model_meta_val_str(model, "general.architecture", arch, sizeof(arch));
        const int64_t r     = meta_int(model, std::string(arch) + ".attention.indexer.kpool");
        const int64_t top_k = meta_int(model, std::string(arch) + ".attention.indexer.top_k");
        if (r > 0 && top_k >= r) {
            st.r      = r;
            st.full_k = top_k/r;
        }
    }
    if (const char * path = std::getenv("TOPK_LOG")) {
        st.log = std::fopen(path, "wb");
        if (st.log == nullptr) {
            LOG_ERR("%s: cannot write TOPK_LOG=%s\n", __func__, path);
            return 1;
        }
    }

    const size_t n_batch = llama_n_batch(ctx);
    std::printf("tokens %zu, batch %zu, ubatch %u\n", tokens.size(), n_batch, llama_n_ubatch(ctx));
    if (st.r > 0) {
        std::printf("pools of %lld cells, a row selects %lld of them: the hot caches hold 1, 2, 4, 8 and 16 times that\n",
                (long long) st.r, (long long) st.full_k);
    } else {
        std::printf("the model names no pooled indexer: no hot cache is followed\n");
    }

    // one graph a decode: llama_decode splits a batch into -ub graphs, and pos0 labels a whole decode
    const size_t n_step = llama_n_ubatch(ctx);
    for (size_t i = 0; i < tokens.size(); i += n_step) {
        const size_t n = std::min(n_step, tokens.size() - i);
        st.pos0 = (int64_t) i;
        if (llama_decode(ctx, llama_batch_get_one(tokens.data() + i, (int32_t) n))) {
            LOG_ERR("%s: decode failed at token %zu\n", __func__, i);
            return 1;
        }
    }

    for (auto & [il, lt] : st.layers) lt.prompt = lt.count();

    // the tokens after the prompt, one a step, as the sampling flags draw them
    std::string text;
    int         n_gen = 0;
    bool        ended = false;
    if (n_predict > 0) {
        common_sampler * smpl = common_sampler_init(model, params.sampling);
        if (smpl == nullptr) {
            LOG_ERR("%s: failed to init the sampler\n", __func__);
            return 1;
        }
        st.generating = true;
        for (; n_gen < n_predict; ++n_gen) {
            llama_token tok = common_sampler_sample(smpl, ctx, -1);
            common_sampler_accept(smpl, tok, true);
            if (llama_vocab_is_eog(vocab, tok)) {
                ended = true;
                break;
            }
            text += common_token_to_piece(ctx, tok);
            st.pos0 = (int64_t) tokens.size() + n_gen;
            if (llama_decode(ctx, llama_batch_get_one(&tok, 1))) {
                LOG_ERR("%s: decode failed at generated token %d\n", __func__, n_gen);
                return 1;
            }
        }
        common_sampler_free(smpl);
    }
    if (st.log) std::fclose(st.log);

    std::printf("\nsummary over the steps where the cache held more live pools than a row selects: the KV a tile gathers, "
                "as a multiple of one row's\n");
    topk_tile_stats all[4];
    for (const auto & [il, lt] : st.layers) {
        if (lt.steps == 0) continue;
        std::printf("layer %2d, %lld steps |", il, (long long) lt.steps);
        for (int i = 0; i < 4; ++i) {
            std::printf(" t%lld %6.3f", (long long) TILES[i], lt.st[i].mean());
            all[i].tiles     += lt.st[i].tiles;
            all[i].sum_ratio += lt.st[i].sum_ratio;
        }
        std::printf("\n");
    }
    std::printf("all layers |");
    for (int i = 0; i < 4; ++i) std::printf(" t%lld %6.3f", (long long) TILES[i], all[i].mean());
    std::printf("\n");

    if (st.r > 0) {
        std::printf("\nthe pools a step selects that a hot cache of the layer's last selections does not hold, as a part of "
                    "what the step selects, and in pools a step: over the steps whose every row selected %lld live pools\n",
                (long long) st.full_k);
        std::printf("the prompt, %u rows a step at most:\n", llama_n_ubatch(ctx));
        print_misses(st, false);
        if (n_predict > 0) {
            std::printf("the %d generated tokens, one row a step:\n", n_gen);
            print_misses(st, true);
        }
    }
    if (n_predict > 0) {
        std::printf("\ngenerated %d tokens (%s):\n%s\n", n_gen, ended ? "the model ended" : "-n reached", text.c_str());
    }

    llama_perf_context_print(ctx);
    llama_backend_free();
    return 0;
}
