#pragma once

#include "ggml.h"
#include "llama.h"
#include "llama-graph.h"

#include <cstdint>
#include <functional>
#include <map>
#include <vector>

struct llama_ubatch;
class llama_kv_cache;
class llama_kv_cache_context;
class llama_kv_cells;

// GLM-5-Next indexer pooling. no input may hold a negative index (ggml_set_rows asserts
// i1 >= 0, ggml_get_rows has no sentinel), so unusable entries are clamped and masked.

// n_kv/kpool (exact only while the sequences' cells are disjoint) plus 2 per seq for rebasing
uint32_t llama_kpool_n_pools(uint32_t n_kv, uint32_t kpool, uint32_t n_seqs = 1);

// select_k of Glm5NextTextIndexer.forward: must run over POOLS, a cell cut takes partial pools
uint32_t llama_kpool_select_k(uint32_t n_pools, uint32_t indexer_top_k, uint32_t kpool);

// The pool maps of a decode step are a function of the cells alone, and between two steps a few cells change. A view
// is one sequence's cells in pool terms (per-cell positions, pool table, completeness), kept up to date from the cells
// llama_kv_cells_log names, so a step reads only the cells that changed. A view that cannot follow (the log no longer
// reaches back, two cells share a position, the positions span more pools than a map holds) is rebuilt from the cells,
// or the maps are built from the cells (llama_kpool_set_input with views == nullptr, the reference).
class llama_kpool_views {
public:
    struct view {
        const llama_kv_cells * cells = nullptr;

        uint32_t r     = 0;
        uint32_t cap   = 0;     // pools the tables hold: llama_kpool_n_pools of the whole cache
        bool     built = false; // the tables were built, and follow the cells since
        bool     ok    = false; // the tables can serve; built && !ok: build from the cells

        int64_t  b_lo = 0;      // first pool of the tables: position p is at slot p - b_lo*r, pool b at b - b_lo
        uint32_t n    = 0;      // cells that hold the sequence

        std::vector<int32_t>  pos_at; // per cell: the position if the cell holds the sequence, else -1
        std::vector<int32_t>  pblk;   // per cell: its pool (p/r) if that pool is complete, else -1
        std::vector<int32_t>  slot;   // [cap*r] position -> cell, 0 if none
        std::vector<uint16_t> fill;   // [cap]   cells in the pool
        std::vector<int32_t>  reps;   // [cap]   cell of the last position of a complete pool, else 0

        void rebuild(const llama_kv_cells & c, llama_seq_id seq);

        // false: the cells disagree with the tables (two cells at one position, a position off the tables); rebuild
        bool apply(const llama_kv_cells & c, llama_seq_id seq, const std::vector<uint32_t> & chg);

    private:
        bool insert(uint32_t j, llama_pos p);
        void remove(uint32_t j);

        std::vector<std::pair<uint32_t, int32_t>> upd;
    };

    struct stats {
        uint64_t n_served  = 0; // streams built from a view
        uint64_t n_rebuilt = 0; // views rebuilt from the cells
        uint64_t n_direct  = 0; // streams built from the cells
    };

    // one call of llama_kpool_set_input: every cells object takes its changes again
    void begin() { ++call; }

    // the view of seq after the changes of its cells, nullptr when it cannot serve a map of n_pools over n_kv cells
    const view * serve(const llama_kv_cells & cells, llama_seq_id seq, uint32_t r, int64_t n_kv, int64_t n_pools);

    stats & get_stats() { return st; }

private:
    void sync(const llama_kv_cells & cells, uint32_t r);

    // a sequence past this many takes the maps built from the cells
    static constexpr size_t n_views_max = 8;

    std::map<llama_seq_id, view> views;

    // the call in which a cells object last took its changes
    std::map<const llama_kv_cells *, uint64_t> synced;

    std::vector<uint32_t> chg;

    uint64_t call = 1;
    stats    st;
};

using llama_kpool_cells_fn = std::function<const llama_kv_cells &(llama_seq_id)>;

// the maps of llama_kv_cache_set_input_kpool from a cells accessor. views == nullptr builds them from the cells
// every call; else a sequence alone in its stream is served from its view, byte for byte the same maps
void llama_kpool_set_input(
        const llama_kpool_cells_fn & cells_of,
              llama_kpool_views    * views,
              ggml_tensor    * cell_pool,
              ggml_tensor    * pool_cells,
              ggml_tensor    * bias,
              ggml_tensor    * pool_bias,
              ggml_tensor    * sel_mask,
              ggml_tensor    * cand_mask,
              ggml_tensor    * pool_reps,
              ggml_tensor    * new_pool_cells,
              ggml_tensor    * new_pool_reps,
        const uint32_t       * strm_of,
              int64_t          kv_size,
              bool             rebuild,
        const llama_ubatch   * ubatch,
              uint32_t         kpool);

// `kv` must be the ATTENTION (MLA) cache; the indexer cache shares its slot layout.
// LLAMA_KPOOL_INPUT_LEGACY=1 builds the maps from the cells every call; LLAMA_KPOOL_INPUT_CHECK=1 builds them both
// ways and aborts on a difference.
//   pool_cells  pool member -> cell, 0 if not resident; past pool_bias's n_pools, n_dump dump pools of cells n_kv + d*kpool + t
//   pool_bias   computed, NOT gathered at the last member, which an incomplete pool lacks
//   sel_mask    n_kv columns, then kpool*n_dump dump columns of -inf (n_kv is cand_mask's)
//   cand_mask   bounds top-k spills a partial seq_rm would let escape
// pool_reps / new_pool_cells / new_pool_reps are nullptr when the cache is off, and an entry
// is emitted only for filled == kpool: cell 0 is real, so writing its 0 slot would clobber
void llama_kv_cache_set_input_kpool(
        const llama_kv_cache * kv,
              ggml_tensor    * cell_pool,
              ggml_tensor    * pool_cells,
              ggml_tensor    * bias,
              ggml_tensor    * pool_bias,
              ggml_tensor    * sel_mask,
              ggml_tensor    * cand_mask,
              ggml_tensor    * pool_reps,
              ggml_tensor    * new_pool_cells,
              ggml_tensor    * new_pool_reps,
        // a global row is strm_of[s]*kv_size + cell; only read when pool_reps is set
        const uint32_t       * strm_of,
              int64_t          kv_size,
        // re-emit every complete pool, not only those this ubatch closed; set after a position mutation
              bool             rebuild,
        const llama_ubatch   * ubatch,
              uint32_t         kpool);

// current decode geometry for the reuse check below, in the same units the
// builder sizes the tensors with (llm_graph_context::build_inp_kpool)
struct llm_graph_input_kpool_dims {
    int64_t n_kv      = 0;
    int64_t n_tokens  = 0;
    int64_t n_stream  = 0;
    int64_t n_tps     = 0;
    int64_t n_ps      = 0;
    int64_t n_pools   = 0;
    int64_t n_dump    = 0;
    int64_t n_new_max = 0;
    bool    rebuild   = false;
    bool    scoring   = false;
};

// one map per ubatch; valid only while every indexer layer sees the same candidate set
class llm_graph_input_kpool : public llm_graph_input_i {
public:
    llm_graph_input_kpool(
            const llama_kv_cache_context * mctx_attn,
            const llama_kv_cache_context * mctx_idx,
            uint32_t kpool) : mctx_attn(mctx_attn), mctx_idx(mctx_idx), kpool(kpool) {}

    ~llm_graph_input_kpool() = default;

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override;

    // gather the geometry the builder sizes the tensors with, for the check below
    static llm_graph_input_kpool_dims current_dims(
            const llama_kv_cache_context * mctx_attn,
            const llama_ubatch           & ubatch,
            const llama_cparams          & cparams,
            const llama_hparams          & hparams,
            uint32_t                       kpool);

    // pure shape check, unit-testable without a memory context
    static bool shapes_match(const llm_graph_input_kpool_dims & dims, const llm_graph_input_kpool & inp);

    ggml_tensor * k_idxs     = nullptr;   // I32 [n_tokens]
    ggml_tensor * pool_cells = nullptr;   // I32 [kpool*(n_pools + n_dump), n_stream]
    ggml_tensor * pool_bias  = nullptr;   // F32 [n_pools, n_tps, n_stream]

    // the top-k's dump pools, n_dump = select_k of them after the n_pools real ones: each scores -FLT_MAX, above a dead pool's
    // -inf and under every live score, so a row takes one only where fewer than select_k pools are live, and its cells are its
    // own, past n_kv (sel_mask's dump columns), so the rows of the mask's scatter stay unique
    ggml_tensor * pool_dump     = nullptr; // F32 [n_dump, n_tps, n_stream], every element -FLT_MAX
    ggml_tensor * pool_cells_3d = nullptr; // pool_cells as [kpool, n_pools + n_dump, n_stream]; built once: a view of an input
                                           // made in each layer is a split input, and a host copy, of its own

    // pooled-key cache: a pool's value lives in the row of its LAST member, a pure function of
    // cell content (seq_cp shares it, rebase leaves it alone)
    ggml_tensor * pool_reps      = nullptr; // I32 [n_pools, n_stream]  stream-local rep cell
    ggml_tensor * new_pool_cells = nullptr; // I32 [kpool*n_new_max, n_stream] members to (re)pool
    ggml_tensor * new_pool_reps  = nullptr; // I64 [n_new_max*n_stream] GLOBAL dest row

    // fixed for the decode phase: a shape tracking pools-closed-this-step would flip the graph
    // topology every kpool tokens and force CUDA-graph recapture
    uint32_t n_new_max = 0;

    // set at build time: re-emit every pool after a position mutation
    bool rebuild = false;

    // exact, since pool_bias only holds 0.0f or -INFINITY. nullptr if the fused path is off
    ggml_tensor * pool_bias_f16 = nullptr; // F16 [n_pools, n_tps, 1, n_stream]

    ggml_tensor * sel_mask   = nullptr;   // F16 [n_kv + kpool*n_dump, n_batch, 1, n_stream]; the dump columns -inf
    ggml_tensor * cand_mask  = nullptr;   // F16 [n_kv, n_batch, 1, n_stream]

    const llama_kv_cache_context * mctx_attn;
    const llama_kv_cache_context * mctx_idx;

    const uint32_t kpool;
};
