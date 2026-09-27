// The recurrent rollback contract where batches of different shapes round differently.
//
// test-recurrent-state-rollback holds a rolled-back recurrent state to a context that decoded only the kept rows within
// an absolute 1e-6, reading the serialized state as f32 words. That holds where a batch's shape does not change its
// arithmetic (the CPU), and reads nothing meaningful from an f16 or q8_0 state (-cts), the served ones. A GPU's kernels
// differ by batch size (MMVQ up to 8 rows, MMQ above), so the same prefix reached through a 9-row batch rolled back to 6
// and through a 6-row batch differ by rounding amplified through the layers: 0.35 at most in the f32 state of a 27B
// hybrid on a 5080, where the restore itself is right.
//
// Here the state is decoded in its own type (ggml's to_float) and compared in normalized squared error. A rolled-back
// state must be within 4x the rounding spread of its prefix (the same tokens decoded one at a time and three at a time,
// against one batch) of the context that decoded only the kept rows, and further than that from the prefixes one token
// shorter and one token longer: a rollback that lands a token early or late fails, and so does a bound too loose to tell
// them apart. Two shapes: three rows off a 9-row batch, and the speculative verify (a 5-token prompt, then n_max 3's
// 4 rows) rolled back by 1, 2 and 3 rows. A state it cannot read fails it: a model whose memory writes another layout
// (DeepSeek V4's, behind its own magic) is not one it is run on.

#include "arg.h"
#include "common.h"
#include "llama.h"

#include <algorithm>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <vector>

static constexpr double spread_margin = 4.0;
static constexpr double spread_floor  = 1e-8; // a backend whose shapes round alike has no spread: 1e-6 absolute of a
                                              // state near 0.01 in magnitude, as test-recurrent-state-rollback allows

static llama_context * make_ctx(const common_params & params, llama_model * model) {
    auto cparams = common_context_params_to_llama(params);
    cparams.n_seq_max = 1;
    cparams.n_rs_seq  = 8;
    cparams.n_batch   = std::max(cparams.n_batch,  (uint32_t) 9);
    cparams.n_ubatch  = std::max(cparams.n_ubatch, (uint32_t) 9);
    return llama_init_from_model(model, cparams);
}

// tokens[p0, p1) in batches of `step` rows
static bool decode(llama_context * ctx, const std::vector<llama_token> & tokens, llama_pos p0, llama_pos p1, int step) {
    for (llama_pos p = p0; p < p1; p += step) {
        const int n = std::min(step, (int) (p1 - p));
        llama_batch batch = llama_batch_init(n, 0, 1);
        for (int i = 0; i < n; ++i) {
            common_batch_add(batch, tokens[p + i], p + i, { 0 }, true);
        }
        const bool ok = llama_decode(ctx, batch) == 0;
        llama_batch_free(batch);
        if (!ok) {
            return false;
        }
    }
    return true;
}

// The recurrent rows of sequence 0 as floats, whatever their type: the conv windows and states of every layer. The
// context writes a magic and the sequence id (llama_context::state_seq_get_data), then llama_memory_recurrent::state_write
// a cell count, each cell's position and sequence count, the transposition flag and layer count, and per tensor its
// type, its row size and the cells' rows. Empty when the serialized state is not laid out that way, or holds a type ggml
// cannot convert.
static std::vector<float> recurrent_rows(llama_context * ctx) {
    std::vector<uint8_t> buf(llama_state_seq_get_size_ext(ctx, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY));
    buf.resize(llama_state_seq_get_data_ext(ctx, buf.data(), buf.size(), 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY));

    size_t off = 0;
    const auto read = [&](void * dst, size_t n) {
        if (off + n > buf.size()) {
            return false;
        }
        std::memcpy(dst, buf.data() + off, n);
        off += n;
        return true;
    };

    uint32_t     magic  = 0;
    llama_seq_id seq_id = -1;
    if (!read(&magic, sizeof(magic)) || !read(&seq_id, sizeof(seq_id)) || seq_id != 0) {
        return {};
    }
    uint32_t cell_count = 0;
    if (!read(&cell_count, sizeof(cell_count)) || cell_count == 0 || cell_count > buf.size()) {
        return {};
    }
    for (uint32_t c = 0; c < cell_count; ++c) {
        llama_pos pos      = 0;
        uint32_t  n_seq_id = 0;
        if (!read(&pos, sizeof(pos)) || !read(&n_seq_id, sizeof(n_seq_id)) || n_seq_id != 0) {
            return {};
        }
    }
    uint32_t s_trans = 0;
    uint32_t n_layer = 0;
    if (!read(&s_trans, sizeof(s_trans)) || !read(&n_layer, sizeof(n_layer)) || s_trans != 0) {
        return {};
    }

    std::vector<float> rows;
    while (off < buf.size()) {
        int32_t  type     = 0;
        uint64_t row_size = 0;
        if (!read(&type, sizeof(type)) || !read(&row_size, sizeof(row_size)) || type < 0 || type >= GGML_TYPE_COUNT) {
            return {};
        }
        const ggml_type t = (ggml_type) type;
        const size_t    n_bytes = (size_t) row_size * cell_count;
        if (ggml_type_size(t) == 0 || row_size % ggml_type_size(t) != 0 || off + n_bytes > buf.size()) {
            return {};
        }
        const size_t n = n_bytes / ggml_type_size(t) * ggml_blck_size(t);
        const size_t at = rows.size();
        rows.resize(at + n);
        if (t == GGML_TYPE_F32) {
            std::memcpy(rows.data() + at, buf.data() + off, n_bytes);
        } else if (ggml_get_type_traits(t)->to_float != nullptr) {
            ggml_get_type_traits(t)->to_float(buf.data() + off, rows.data() + at, (int64_t) n);
        } else {
            return {};
        }
        off += n_bytes;
    }
    return rows;
}

// normalized squared error of a against the reference b; infinite when they cannot be compared
static double nmse(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size() || b.empty()) {
        return INFINITY;
    }
    double err = 0.0;
    double ref = 0.0;
    for (size_t i = 0; i < b.size(); ++i) {
        const double d = (double) a[i] - (double) b[i];
        err += d * d;
        ref += (double) b[i] * (double) b[i];
    }
    return std::isfinite(err) && ref > 0.0 ? err / ref : INFINITY;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    params.sampling.seed = 1234;
    params.n_predict = 1;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    ggml_backend_load_all();

    common_init_result_ptr llama_init = common_init_from_params(params);
    llama_model * model = llama_init->model();
    if (model == nullptr) {
        fprintf(stderr, "%s : failed to init model\n", __func__);
        return 1;
    }
    if (!llama_model_is_recurrent(model) && !llama_model_is_hybrid(model)) {
        fprintf(stderr, "%s : skipping for non-recurrent model\n", __func__);
        return 0;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::vector<llama_token> tokens;
    {
        llama_context * ctx = make_ctx(params, model);
        if (ctx == nullptr) {
            fprintf(stderr, "%s : failed to init a context\n", __func__);
            return 1;
        }
        if (llama_n_rs_seq(ctx) < 4) {
            fprintf(stderr, "%s : skipping because n_rs_seq is below the rollbacks tested\n", __func__);
            llama_free(ctx);
            return 0;
        }
        if (llama_vocab_type(vocab) == LLAMA_VOCAB_TYPE_NONE) {
            tokens = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
        } else {
            tokens = common_tokenize(ctx, "The quick brown fox jumps over the lazy dog", true);
        }
        llama_free(ctx);
    }
    if (tokens.empty()) {
        fprintf(stderr, "%s : not enough prompt tokens\n", __func__);
        return 1;
    }
    tokens.resize(10, tokens.back());

    // the state of the prefix [0, p) decoded in batches of `step` rows, each once
    std::map<std::pair<llama_pos, int>, std::vector<float>> prefixes;
    const auto prefix = [&](llama_pos p, int step) -> const std::vector<float> & {
        auto it = prefixes.find({ p, step });
        if (it == prefixes.end()) {
            llama_context * ctx = make_ctx(params, model);
            std::vector<float> rows;
            if (ctx != nullptr && decode(ctx, tokens, 0, p, step)) {
                rows = recurrent_rows(ctx);
            }
            llama_free(ctx);
            it = prefixes.emplace(std::make_pair(p, step), std::move(rows)).first;
        }
        return it->second;
    };

    // the state after decoding tokens[0, p_batch) in batches of `steps` rows each, then removing [p_keep, ...)
    const auto rolled_back = [&](const std::vector<int> & steps, llama_pos p_keep, bool & removed) {
        llama_context * ctx = make_ctx(params, model);
        std::vector<float> rows;
        removed = false;
        llama_pos p = 0;
        bool ok = ctx != nullptr;
        for (const int step : steps) {
            ok = ok && decode(ctx, tokens, p, p + step, step);
            p += step;
        }
        if (ok) {
            removed = llama_memory_seq_rm(llama_get_memory(ctx), 0, p_keep, -1);
            rows = recurrent_rows(ctx);
        }
        llama_free(ctx);
        return rows;
    };

    if (prefix(1, 1).empty()) {
        fprintf(stderr, "%s : the serialized state is not the recurrent layout, or holds a type ggml cannot convert: FAILED\n", __func__);
        return 1;
    }

    struct rollback_case {
        const char *     what;
        std::vector<int> steps;  // the batches decoded before the rollback
        llama_pos        p_keep; // the rows kept
    };
    const std::vector<rollback_case> cases = {
        { "3 rows off a 9-row batch",       { 9 },    6 },
        { "1 row off a 4-row verify",       { 5, 4 }, 8 },
        { "2 rows off a 4-row verify",      { 5, 4 }, 7 },
        { "3 rows off a 4-row verify",      { 5, 4 }, 6 },
    };

    bool ok = true;
    for (const auto & c : cases) {
        bool removed = false;
        const std::vector<float> rolled = rolled_back(c.steps, c.p_keep, removed);
        if (!removed) {
            fprintf(stderr, "%s : %s: the rollback was refused\n", __func__, c.what);
            ok = false;
            continue;
        }
        const llama_pos p = c.p_keep;
        const std::vector<float> & truth = prefix(p, p);
        const double spread = std::max(nmse(prefix(p, 1), truth), nmse(prefix(p, 3), truth));
        const double bound  = spread_margin * std::max(spread, spread_floor);
        const double diff   = nmse(rolled, truth);
        const double early  = nmse(rolled, prefix(p - 1, p - 1));
        const double late   = nmse(rolled, prefix(p + 1, p + 1));

        const bool within = diff <= bound;
        const bool apart  = early > bound && late > bound;
        fprintf(stderr, "%s : %s: nmse %.3g against the kept prefix, bound %.3g (spread %.3g); a token early %.3g, late %.3g: %s\n",
                __func__, c.what, diff, bound, spread, early, late,
                within && apart ? "ok" : !within ? "FAILED, beyond the rounding" : "FAILED, the bound cannot tell a token off");
        ok &= within && apart;
    }

    fprintf(stderr, "%s : %s\n", __func__, ok ? "OK" : "FAILED");
    return ok ? 0 : 1;
}
