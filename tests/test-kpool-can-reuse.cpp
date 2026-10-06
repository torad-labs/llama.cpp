// Graph reuse for the GLM-5-Next pooled indexer input.
//
// llm_graph_input_kpool had no can_reuse override, so the base returned false and
// poisoned the whole llama-graph reuse check on every decode step: a fresh cgraph
// per step, a fresh CUDA graph capture per step, zero replays (479/479 reuse
// verdicts false on a 150-token probe, "CUDA Graph id reused" never logged).
//
// CPU-only: tensor shapes plus the pure shape predicate, no model, no GPU.

#include "../src/llama-kv-cache-kpool.h"

#include "ggml.h"

#include <cstdio>
#include <type_traits>

// the override must exist: without it &llm_graph_input_kpool::can_reuse names the
// base member (pointer-to-member-of-base type) and this fails to compile; the base
// returns false unconditionally, which is the 479/479-false serving leg
using kpool_reuse_sig_t = bool (llm_graph_input_kpool::*)(const llm_graph_params &);
static_assert(std::is_same_v<decltype(&llm_graph_input_kpool::can_reuse), kpool_reuse_sig_t>,
        "kpool input does not override can_reuse");

static int n_fail = 0;

#define CHECK(cond, ...) do { if (!(cond)) { ++n_fail; if (n_fail <= 20) { fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } } } while (0)

// representative decode step: kpool=4, n_kv=256 (padded), 1 token, 1 stream,
// top_k=64 (n_select=67), n_ctx far above it so the scoring path is built
static constexpr uint32_t KPOOL = 4;
static constexpr int64_t N_KV = 256;
static constexpr int64_t N_TOKENS = 1;
static constexpr int64_t N_STREAM = 1;
static constexpr int64_t N_TPS = 1;
static constexpr int64_t N_PS = 1;
static constexpr int64_t N_POOLS = N_KV/KPOOL + 2*N_PS; // 66, llama_kpool_n_pools
static constexpr int64_t N_NEW_MAX = N_TPS/KPOOL + N_PS; // 1, decode, not rebuilding
static constexpr int64_t N_DUMP = 64/KPOOL; // 16, llama_kpool_select_k: the top-k's dump pools

struct case_tensors {
    ggml_tensor * k_idxs = nullptr;
    ggml_tensor * pool_cells = nullptr;
    ggml_tensor * pool_bias = nullptr;
    ggml_tensor * pool_dump = nullptr;
    ggml_tensor * pool_bias_f16 = nullptr;
    ggml_tensor * sel_mask = nullptr;
    ggml_tensor * cand_mask = nullptr;
    ggml_tensor * pool_reps = nullptr;
    ggml_tensor * new_pool_cells = nullptr;
    ggml_tensor * new_pool_reps = nullptr;
};

static case_tensors make_tensors(ggml_context * ctx, bool scoring, int64_t n_new_max) {
    case_tensors t;
    t.k_idxs = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, N_TOKENS);
    if (!scoring) {
        return t;
    }
    t.pool_cells     = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, KPOOL*(N_POOLS + N_DUMP), N_STREAM);
    t.pool_bias      = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, N_POOLS, N_TPS, N_STREAM);
    t.pool_dump      = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, N_DUMP, N_TPS, N_STREAM);
    t.pool_bias_f16  = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, N_POOLS, N_TPS, 1, N_STREAM);
    t.sel_mask       = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, N_KV + KPOOL*N_DUMP, N_TPS, 1, N_STREAM);
    t.cand_mask      = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, N_KV, N_TPS, 1, N_STREAM);
    t.pool_reps      = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, N_POOLS, N_STREAM);
    t.new_pool_cells = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, KPOOL*n_new_max, N_STREAM);
    t.new_pool_reps  = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_new_max*N_STREAM);
    return t;
}

static llm_graph_input_kpool make_inp(const case_tensors & t, bool scoring, int64_t n_new_max, bool rebuild) {
    llm_graph_input_kpool inp(nullptr, nullptr, KPOOL);
    inp.k_idxs         = t.k_idxs;
    inp.pool_cells     = t.pool_cells;
    inp.pool_bias      = t.pool_bias;
    inp.pool_dump      = t.pool_dump;
    inp.pool_bias_f16  = t.pool_bias_f16;
    inp.sel_mask       = t.sel_mask;
    inp.cand_mask      = t.cand_mask;
    inp.pool_reps      = t.pool_reps;
    inp.new_pool_cells = t.new_pool_cells;
    inp.new_pool_reps  = t.new_pool_reps;
    inp.n_new_max      = (uint32_t) n_new_max;
    inp.rebuild        = rebuild;
    inp.n_tail         = KPOOL - 1;
    (void) scoring;
    return inp;
}

static llm_graph_input_kpool_dims base_dims() {
    llm_graph_input_kpool_dims d;
    d.n_kv      = N_KV;
    d.n_tokens  = N_TOKENS;
    d.n_stream  = N_STREAM;
    d.n_tps     = N_TPS;
    d.n_ps      = N_PS;
    d.n_pools   = N_POOLS;
    d.n_dump    = N_DUMP;
    d.n_new_max = N_NEW_MAX;
    d.n_tail    = KPOOL - 1;
    d.rebuild   = false;
    d.scoring   = true;
    return d;
}

int main() {
    struct ggml_init_params gparams = { 16*1024*1024, nullptr, true };
    ggml_context * ctx = ggml_init(gparams);

    // identical consecutive decode step reuses
    {
        case_tensors t = make_tensors(ctx, true, N_NEW_MAX);
        llm_graph_input_kpool inp = make_inp(t, true, N_NEW_MAX, false);
        CHECK(llm_graph_input_kpool::shapes_match(base_dims(), inp), "identical step must reuse");
    }

    // rebuild pass (every pool re-emitted after a position mutation) reuses against itself
    {
        const int64_t n_new_rebuild = N_POOLS;
        case_tensors t = make_tensors(ctx, true, n_new_rebuild);
        llm_graph_input_kpool inp = make_inp(t, true, n_new_rebuild, true);
        llm_graph_input_kpool_dims d = base_dims();
        d.rebuild = true;
        d.n_new_max = n_new_rebuild;
        CHECK(llm_graph_input_kpool::shapes_match(d, inp), "rebuild step must reuse against itself");
    }

    // non-scoring graph (small n_ctx) reuses against itself
    {
        case_tensors t = make_tensors(ctx, false, 0);
        llm_graph_input_kpool inp = make_inp(t, false, 0, false);
        llm_graph_input_kpool_dims d = base_dims();
        d.scoring = false;
        CHECK(llm_graph_input_kpool::shapes_match(d, inp), "non-scoring step must reuse");
    }

    // every shape-changing mutation must refuse
    {
        case_tensors t = make_tensors(ctx, true, N_NEW_MAX);
        llm_graph_input_kpool inp = make_inp(t, true, N_NEW_MAX, false);

        llm_graph_input_kpool_dims d = base_dims();
        d.n_tokens = 2;
        CHECK(!llm_graph_input_kpool::shapes_match(d, inp), "n_tokens change must refuse");

        d = base_dims();
        d.n_tps = 2;
        CHECK(!llm_graph_input_kpool::shapes_match(d, inp), "n_tps change must refuse");

        d = base_dims();
        d.n_kv = 512;
        d.n_pools = N_KV*2/KPOOL + 2*N_PS;
        CHECK(!llm_graph_input_kpool::shapes_match(d, inp), "n_kv growth must refuse");

        d = base_dims();
        d.n_stream = 2;
        CHECK(!llm_graph_input_kpool::shapes_match(d, inp), "n_stream change must refuse");

        d = base_dims();
        d.rebuild = true;
        CHECK(!llm_graph_input_kpool::shapes_match(d, inp), "dirty flip must refuse");

        d = base_dims();
        d.n_new_max = N_NEW_MAX + 1;
        CHECK(!llm_graph_input_kpool::shapes_match(d, inp), "n_new_max drift must refuse");

        d = base_dims();
        d.scoring = false;
        CHECK(!llm_graph_input_kpool::shapes_match(d, inp), "scoring flip must refuse");

        d = base_dims();
        d.n_dump = N_DUMP - 1;
        CHECK(!llm_graph_input_kpool::shapes_match(d, inp), "a dump pool count change must refuse");

        // the sparse attention's n_kv_max is sized on the longest tail
        d = base_dims();
        d.n_tail = KPOOL + 8;
        CHECK(!llm_graph_input_kpool::shapes_match(d, inp), "a longer tail must refuse");

        // non-scoring stored graph asked against a scoring step must refuse
        case_tensors ts = make_tensors(ctx, false, 0);
        llm_graph_input_kpool inps = make_inp(ts, false, 0, false);
        CHECK(!llm_graph_input_kpool::shapes_match(base_dims(), inps),
                "non-scoring graph against scoring step must refuse");
    }

    ggml_free(ctx);

    if (n_fail == 0) {
        printf("ok\n");
    }
    return n_fail != 0;
}
