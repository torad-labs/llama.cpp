// Graph reuse for the GLM-5-Next pooled indexer input.
//
// llm_graph_input_kpool had no can_reuse override, so the base returned false and
// poisoned the whole llama-graph reuse check on every decode step: a fresh cgraph
// per step, a fresh CUDA graph capture per step, zero replays (479/479 reuse
// verdicts false on a 150-token probe, "CUDA Graph id reused" never logged).
//
// CPU-only: graph construction and reuse, without model weights or GPU execution.

#include "../src/llama-kv-cache-kpool.h"
#include "../src/llama-memory-hybrid.h"
#include "../src/llama-model.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <stdexcept>
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

static case_tensors make_tensors(ggml_context * ctx, bool scoring, int64_t n_new_max,
        int64_t n_kv = N_KV, int64_t n_pools = N_POOLS, int64_t n_dump = N_DUMP) {
    case_tensors t;
    t.k_idxs = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, N_TOKENS);
    if (!scoring) {
        return t;
    }
    t.pool_cells     = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, KPOOL*(n_pools + n_dump), N_STREAM);
    t.pool_bias      = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_pools, N_TPS, N_STREAM);
    t.pool_dump      = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_dump, N_TPS, N_STREAM);
    t.pool_bias_f16  = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, n_pools, N_TPS, 1, N_STREAM);
    t.sel_mask       = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, n_kv + KPOOL*n_dump, N_TPS, 1, N_STREAM);
    t.cand_mask      = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, n_kv, N_TPS, 1, N_STREAM);
    t.pool_reps      = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_pools, N_STREAM);
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

static void set_host_kv(const char * value) {
#ifdef _WIN32
    _putenv_s("LLAMA_KV_HOST", value ? value : "");
#else
    if (value) { setenv("LLAMA_KV_HOST", value, 1); }
    else { unsetenv("LLAMA_KV_HOST"); }
#endif
}

static void test_host_kv_graph_transition() {
    const char * previous = getenv("LLAMA_KV_HOST");
    const bool had_previous = previous != nullptr;
    const std::string saved = previous ? previous : "";
    set_host_kv("0");
    std::unique_ptr<llama_model> model(llama_model_create(LLM_ARCH_GLM5NEXT, llama_model_default_params()));
    auto & h = model->hparams;
    h.n_layer_all = 1;
    h.n_embd = 512;
    h.n_head_arr.fill(8);
    h.n_head_kv_arr.fill(1);
    h.n_embd_head_k_full = h.n_embd_head_v_full = 512;
    h.n_lora_kv = 512;
    h.indexer_head_size = 128;
    h.indexer_top_k = 2048;
    h.indexer_kpool = 32;
    auto cpu = ggml_backend_cpu_init();

    // The cutoff's row cap is read once a process (LLAMA_KPOOL_DIRECT_ROW_CAP; 2 unless set), so each rule is a run of
    // this test. A row here reads 2,080 cells through indices: under a cap of 4 or more, 512 rows still stage at 8,192
    // cells (8 is the rule as it was: its cutoff there is 16,640); under the default every row count below reads
    // through indices from 8,192 cells on.
    const char * cap_env = getenv("LLAMA_KPOOL_DIRECT_ROW_CAP");
    const bool stage_512_at_8192 = cap_env && atoll(cap_env) >= 4;

    // Host storage must keep the resident attention kernel before and after the direct-index cutoff.
    for (const uint32_t cells : {2560u, 8192u, 32768u}) {
        for (const uint32_t tokens : {512u, 3u, 1u}) {
            llama_memory_hybrid memory(*model, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, false,
                    cells, 256, 0, LLAMA_SWA_TYPE_NONE, GGML_TYPE_F32, GGML_TYPE_F32,
                    1, 1, 0, false, false, [](int32_t) { return true; },
                    [](int32_t) { return false; }, [](int32_t) { return true; }, GGML_TYPE_Q8_0);
            llama_memory_hybrid_context mctx(&memory);
            llama_seq_id seq = 0;
            std::vector<llama_pos> positions(tokens);
            std::vector<int32_t> n_seq_id(tokens, 1);
            std::vector<llama_seq_id *> seq_id(tokens, &seq);
            for (uint32_t i = 0; i < tokens; ++i) { positions[i] = cells - tokens + i; }
            for (int mode = 0; mode < 3; ++mode) {
                // CPU storage lets this test exercise the actual graph builder without requiring mapped GPU memory.
                set_host_kv(mode ? "1" : "0");
                llm_graph_result result(256);
                result.reset();
                llm_graph_params params {};
                params.hparams = h;
                params.cparams.n_ctx = cells;
                params.cparams.n_seq_max = 1;
                params.cparams.n_seq_devices = 1;
                params.cparams.flash_attn = mode != 2;
                params.ubatch.n_tokens = params.ubatch.n_seq_tokens = tokens;
                params.ubatch.n_seqs = params.ubatch.n_seqs_unq = params.ubatch.n_pos = 1;
                params.ubatch.pos = positions.data();
                params.ubatch.n_seq_id = n_seq_id.data();
                params.ubatch.seq_id = seq_id.data();
                params.ubatch.seq_id_unq = &seq;
                params.backend_cpu = cpu;
                params.res = &result;
                llm_graph_context graph(params);
                auto * mask = ggml_new_tensor_4d(graph.ctx0, GGML_TYPE_F16, cells, tokens, 1, 1);
                const bool scoring = llama_kpool_needs_scoring(cells, cells, h.indexer_top_k, h.indexer_kpool);
                CHECK(scoring, "fresh prompt must enter sparse attention");
                auto * input = graph.build_inp_kpool(&mctx, mask, scoring);
                const bool direct = mode == 1 &&
                        (cells == 32768 || (cells == 8192 && (tokens < 512 || !stage_512_at_8192)));
                CHECK(bool(input->tail_cells) == direct,
                        "host storage changed attention selection: cells=%u tokens=%u host=%d flash=%d", cells, tokens, mode != 0, mode != 2);
                if (mode == 2) { continue; }
                llm_graph_input_attn_k attn(h, params.cparams, mctx.get_attn());
                attn.self_k_idxs = ggml_new_tensor_1d(graph.ctx0, GGML_TYPE_I64, tokens);
                attn.self_kq_mask_cnv = mask;
                auto * q = ggml_new_tensor_3d(graph.ctx0, GGML_TYPE_F32, 512, 8, tokens);
                auto * k = ggml_new_tensor_3d(graph.ctx0, GGML_TYPE_F32, 512, 1, tokens);
                auto * top_k = ggml_new_tensor_3d(graph.ctx0, GGML_TYPE_I32, 2048, tokens, 1);
                try {
                    auto * out = graph.build_attn_sparse(&attn, nullptr, nullptr, nullptr, q, k, k, nullptr,
                            nullptr, nullptr, top_k, input->sel_mask, input->cand_mask, input->n_tail, 1.0f, 0, input->tail_cells);
                    ggml_build_forward_expand(graph.gf, out);
                    bool found = false;
                    for (int i = 0; i < ggml_graph_n_nodes(graph.gf); ++i) {
                        auto * node = ggml_graph_node(graph.gf, i);
                        if (node->op == GGML_OP_FLASH_ATTN_EXT || node->op == GGML_OP_FLASH_ATTN_EXT_SPARSE) {
                            found = true;
                            CHECK((node->op == GGML_OP_FLASH_ATTN_EXT_SPARSE) == direct,
                                    "attention kernel differs: cells=%u tokens=%u host=%d", cells, tokens, mode != 0);
                        }
                    }
                    CHECK(found, "missing flash attention node");
                } catch (const std::runtime_error & error) {
                    CHECK(false, "attention graph refused: cells=%u tokens=%u host=%d: %s", cells, tokens, mode != 0, error.what());
                }
            }
            set_host_kv("0");
        }
    }
    ggml_backend_free(cpu);
    set_host_kv(had_previous ? saved.c_str() : nullptr);
}

static void test_cell_capacity(uint32_t top_k) {
    struct env_value {
        const char * name;
        bool present;
        std::string value;
        env_value(const char * name, const char * value) : name(name), present(getenv(name) != nullptr),
                value(present ? getenv(name) : "") { set(value); }
        void set(const char * v) const {
#ifdef _WIN32
            _putenv_s(name, v ? v : "");
#else
            if (v) { setenv(name, v, 1); }
            else { unsetenv(name); }
#endif
        }
        ~env_value() { set(present ? value.c_str() : nullptr); }
    };
    env_value prefill("LLAMA_KV_PAD_PREFILL", "8192");
    env_value shared("LLAMA_KV_PAD_SHARED_LEGACY", "0");
    env_value dense("LLAMA_KV_PAD_DENSE_LEGACY", "0");
    env_value capacity("LLAMA_KV_CELL_CAPACITY_LEGACY", "1");

    std::unique_ptr<llama_model> model(llama_model_create(LLM_ARCH_GLM5NEXT, llama_model_default_params()));
    auto & h = model->hparams;
    h.n_layer_all = 1;
    h.n_embd = 16;
    h.n_head_arr.fill(1);
    h.n_head_kv_arr.fill(1);
    h.n_embd_head_k_full = h.n_embd_head_v_full = 4;
    h.n_lora_kv = 4;
    h.indexer_head_size = 4;
    h.indexer_top_k = top_k;
    h.indexer_kpool = KPOOL;

    constexpr uint32_t size = 131328;
    const std::vector<uint32_t> ends = {
        128, 2048, 2049, 4096, 4097, 8192, 8193, 16384, 16385, 16640, 16641, 24576, 24577,
        32768, 32769, 49152, 49153, 65536, 65537, 81920, 81921, 90112, 131072, 131073, size,
    };
    for (const uint32_t n_seq : {1u, 8u, 23u}) {
        const bool unified = n_seq > 1;
        std::vector<llm_graph_input_kpool_dims> baseline;
        std::vector<uint32_t> pads;
        for (const bool legacy : {true, false}) {
            capacity.set(legacy ? "1" : "0");
            llama_kv_cache kv(*model, h, GGML_TYPE_F32, GGML_TYPE_F32, false, false, unified,
                    size, n_seq, 256, 0, LLAMA_SWA_TYPE_NONE, nullptr, {}, {}, {});
            // The cache keeps the policy it was constructed with.
            capacity.set(legacy ? "0" : "1");
            ggml_context * ctx = ggml_init({32*1024*1024, nullptr, false});
            llama_cparams cp {};
            cp.n_ctx = size;
            cp.n_seq_max = n_seq;
            cp.n_seq_devices = 1;
            cp.kv_unified = unified;
            llama_seq_id seq = 0;
            llama_seq_id * seq_ptr = &seq;
            int32_t n_seq_id = 1;
            llama_pos pos = 0;
            llama_ubatch ub {};
            ub.n_tokens = ub.n_seq_tokens = ub.n_seqs = ub.n_seqs_unq = ub.n_pos = 1;
            ub.pos = &pos;
            ub.n_seq_id = &n_seq_id;
            ub.seq_id = &seq_ptr;
            ub.seq_id_unq = &seq;
            llama_kv_cache::slot_info slot {};
            slot.s0 = slot.s1 = 0;
            slot.strm = {0};
            slot.idxs = {{0}};
            std::unique_ptr<llm_graph_input_kpool> kept;
            uint32_t kept_span = 0;
            for (size_t i = 0; i < ends.size(); ++i) {
                // Sparse physical placement grows the cell span, not the sequence's pool count.
                slot.idxs[0][0] = ends[i] - 1;
                pos = (llama_pos) i;
                llama_kv_cache_context mctx(&kv, {slot}, {ub});
                CHECK(mctx.apply(), "cell capacity context apply");
                const auto d = llm_graph_input_kpool::current_dims(&mctx, ub, cp, h, KPOOL, unified);
                if (legacy) {
                    baseline.push_back(d);
                    pads.push_back(mctx.get_n_pad());
                } else {
                    const auto & old = baseline[i];
                    const uint32_t cutoff = top_k == 8 ? 4096 : 16640;
                    const uint32_t want = unified && old.scoring && old.n_kv >= cutoff ?
                            std::min(size, GGML_PAD((uint32_t) old.n_kv, 65536u)) : (uint32_t) old.n_kv;
                    // under the cutoff's default row cap (2) and under the rule as it was (8), which the gate keeps
                    for (const int64_t cap : {2, 8}) {
                        for (const int64_t rows : {1, 3, 8, 69, 768}) {
                            const int64_t staged = llama_kpool_n_kv_max(KPOOL*old.n_dump, std::max<int64_t>(1, old.n_tail));
                            const int64_t at = std::max<int64_t>(4096, std::min<int64_t>(rows, cap)*staged);
                            CHECK((d.n_kv >= at) == (old.n_kv >= at),
                                    "cell capacity changed dispatch at %u cells, %lld rows, row cap %lld", ends[i],
                                    (long long) rows, (long long) cap);
                        }
                    }
                    CHECK(d.n_kv == want, "cell capacity at %u: got %lld, want %u", ends[i], (long long) d.n_kv, want);
                    CHECK(mctx.get_n_pad() == pads[i], "cell capacity changed pool padding");
                    CHECK(mctx.get_n_kv_pool() == old.n_kv, "cell capacity changed pool span");
                    CHECK(d.n_pools == old.n_pools && d.n_dump == old.n_dump &&
                            d.scoring == old.scoring && d.n_new_max == old.n_new_max && d.n_tail == old.n_tail,
                            "cell capacity changed indexer dimensions at %u", ends[i]);
                }
                CHECK(d.n_kv >= ends[i] && d.n_kv <= size, "cell capacity outside live/allocation bounds");
                if (n_seq == 23) {
                    for (uint32_t rows : {69u, 768u}) {
                        uint32_t pad = 0, pool_span = 0;
                        std::vector<llama_pos> query_pos(rows, pos);
                        std::vector<int32_t> query_n_seq(rows, 1);
                        std::vector<llama_seq_id *> query_seq(rows, &seq);
                        llama_ubatch query = ub;
                        query.n_tokens = query.n_seq_tokens = rows;
                        query.pos = query_pos.data();
                        query.n_seq_id = query_n_seq.data();
                        query.seq_id = query_seq.data();
                        CHECK(kv.get_n_kv(slot, &query, &pad, &pool_span) == d.n_kv,
                                "reply and prompt cell capacities differ");
                        CHECK(pad == mctx.get_n_pad() && pool_span == mctx.get_n_kv_pool(),
                                "reply and prompt pool spans differ");
                    }
                }
                auto * k = mctx.get_k(ctx, 0);
                CHECK(k->ne[2] == d.n_kv && ggml_nbytes(k) <= ggml_nbytes(kv.get_k_storage(0)), "KV view exceeds allocation");
                auto * mask_ctx = ggml_init({ggml_tensor_overhead(), nullptr, true});
                auto * mask = ggml_new_tensor_4d(mask_ctx, GGML_TYPE_F16, d.n_kv, 1, 1, 1);
                auto * mask_buffer = ggml_backend_alloc_ctx_tensors_from_buft(mask_ctx, ggml_backend_cpu_buffer_type());
                mctx.set_input_kq_mask(mask, &ub, true);
                const auto * values = (const ggml_fp16_t *) mask->data;
                size_t live = 0;
                for (uint32_t cell = 0; cell < d.n_kv; ++cell) {
                    const float value = ggml_fp16_to_fp32(values[cell]);
                    if (live <= i && cell == ends[live] - 1) {
                        CHECK(value == 0.0f, "live cell hidden by capacity at %u", cell);
                        ++live;
                    } else {
                        CHECK(value == -INFINITY, "unused cell exposed by capacity at %u", cell);
                    }
                }
                CHECK(live == i + 1, "capacity mask omitted live cells");
                ggml_backend_buffer_free(mask_buffer);
                ggml_free(mask_ctx);
                if (d.scoring) {
                    CHECK(kv.get_kpool_views().serve(kv.get_cells(seq), seq, KPOOL, d.n_kv, d.n_pools),
                            "resident pool view refused valid cell capacity");
                    if (kept && kept_span == d.n_kv) {
                        CHECK(llm_graph_input_kpool::shapes_match(d, *kept), "growth inside capacity rejected kept input");
                    } else {
                        if (kept) CHECK(!llm_graph_input_kpool::shapes_match(d, *kept), "capacity crossing accepted undersized input");
                        const auto tensors = make_tensors(ctx, true, d.n_new_max, d.n_kv, d.n_pools, d.n_dump);
                        kept = std::make_unique<llm_graph_input_kpool>(make_inp(tensors, true, d.n_new_max, d.rebuild));
                        kept_span = d.n_kv;
                    }
                }
            }

            // Duplicate positions can make the tail wider than kpool - 1 and raise the staging cutoff.
            kv.clear(false);
            constexpr uint32_t n_dup = 1030;
            const uint32_t span = top_k == 8 ? 8192 : 24576;
            std::vector<llama_pos> dup_pos(n_dup, 2);
            std::vector<int32_t> dup_n_seq(n_dup, 1);
            std::vector<llama_seq_id *> dup_seq(n_dup, &seq);
            llama_ubatch dup = ub;
            dup.n_tokens = dup.n_seq_tokens = n_dup;
            dup.pos = dup_pos.data();
            dup.n_seq_id = dup_n_seq.data();
            dup.seq_id = dup_seq.data();
            slot.idxs[0].resize(n_dup);
            for (uint32_t i = 0; i < n_dup; ++i) { slot.idxs[0][i] = i; }
            slot.idxs[0].back() = span - 1;
            llama_kv_cache_context dup_ctx(&kv, {slot}, {dup});
            CHECK(dup_ctx.apply(), "duplicate-tail capacity context apply");
            const auto dup_dims = llm_graph_input_kpool::current_dims(&dup_ctx, dup, cp, h, KPOOL, unified);
            CHECK(dup_dims.n_tail == n_dup, "duplicate-tail bound is not the placed cell count");
            CHECK(dup_ctx.get_n_kv_pool() == span && dup_ctx.get_n_kv() == span,
                    "cell capacity crossed duplicate-tail staging cutoff: got %u, want %u", dup_ctx.get_n_kv(), span);
            ggml_free(ctx);
        }
    }
}

int main() {
    test_cell_capacity(8);
    test_cell_capacity(2048);
    test_host_kv_graph_transition();
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
