#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "llama.h"
#include "llama-graph.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <vector>

struct llama_ubatch;
class llama_kv_cache;
class llama_kv_cache_context;
class llama_kv_cells;

// Cache-owned tensors outlive graph allocator resets. Graph inputs retain shared
// ownership when a new selection width replaces an old allocation.
struct llama_kpool_backend_tensor {
    llama_kpool_backend_tensor(ggml_backend_buffer_type_t buft, int64_t count, int64_t slots = 1, uint32_t devices = 1);
    ~llama_kpool_backend_tensor();
    llama_kpool_backend_tensor(const llama_kpool_backend_tensor &) = delete;
    llama_kpool_backend_tensor & operator=(const llama_kpool_backend_tensor &) = delete;
    void begin_upload(ggml_backend_t backend);
    void upload(size_t offset, const int32_t * data, size_t count);
    void end_upload();
    ggml_backend_t upload_backend = nullptr;
    ggml_backend_event_t uploaded = nullptr;
    bool upload_pending = false;
    std::vector<std::vector<int32_t>> staging;
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_tensor * tensor = nullptr;
};

// Transient, per-round MTP selections. A mutation of the owning memory invalidates
// them; these are never serialized with a prompt cache.
class llama_kpool_draft {
public:
    void clear() { ++epoch; rows.clear(); }
    void advance(int32_t next) { if (next <= 0 || next != step + 1) clear(); step = next; }
    bool active() const { return step >= 0; }
    uint64_t generation() const { return epoch; }
    void capture(llama_seq_id seq, llama_pos pos, int64_t n_kv, const std::vector<int32_t> & cells);
    void capture_backend(llama_seq_id seq, llama_pos pos, int64_t n_kv, ggml_tensor * source, int64_t row,
            ggml_backend_t backend = nullptr, uint32_t slots = 1, uint32_t devices = 1);
    // Every row of source at once, row i for seqs[i]. One device of sequences with several slots keeps the rows in one
    // bank of [width, slots], and a run of rows whose sequences are neighbours is one copy: a store and a copy a
    // sequence were 46 a step at 23 sequences. LLAMA_KPOOL_DRAFT_BANK_LEGACY=1: a store and a copy a sequence
    void capture_backend_rows(const llama_seq_id * seqs, const llama_pos * positions, size_t n, int64_t n_kv,
            ggml_tensor * source, ggml_backend_t backend, uint32_t slots, uint32_t devices);
    std::shared_ptr<llama_kpool_backend_tensor> backend_row(llama_seq_id seq) const;
    bool can_reuse(const llama_ubatch & ubatch, int64_t n_kv, int64_t width, uint32_t kpool) const;
    void copy(const llama_ubatch & ubatch, int32_t * dst) const;
private:
    struct row {
        llama_pos pos; int32_t step; int64_t n_kv; std::vector<int32_t> cells;
        std::shared_ptr<llama_kpool_backend_tensor> backend;
        size_t offset = 0; // I32 elements in the bank's canonical layout
        int64_t width() const { return backend ? backend->tensor->ne[0] : (int64_t) cells.size(); }
    };
    std::map<llama_seq_id, std::shared_ptr<llama_kpool_backend_tensor>> stores;
    std::shared_ptr<llama_kpool_backend_tensor> bank;
    int32_t step = -1;
    uint64_t epoch = 0;
    std::map<llama_seq_id, row> rows;
};

// GLM-5-Next indexer pooling. no input may hold a negative index (ggml_set_rows asserts
// i1 >= 0, ggml_get_rows has no sentinel), so unusable entries are clamped and masked.

// n_kv/kpool (exact only while the sequences' cells are disjoint) plus 2 per seq for rebasing
uint32_t llama_kpool_n_pools(uint32_t n_kv, uint32_t kpool, uint32_t n_seqs = 1);

// select_k of Glm5NextTextIndexer.forward: must run over POOLS, a cell cut takes partial pools
uint32_t llama_kpool_select_k(uint32_t n_pools, uint32_t indexer_top_k, uint32_t kpool);

bool llama_kpool_needs_scoring(int64_t n_kv, uint32_t n_ctx, uint32_t indexer_top_k, uint32_t kpool);

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

        // Offsets/counts are I32 elements. The callback is the actual upload boundary.
        std::shared_ptr<llama_kpool_backend_tensor> resident(ggml_backend_buffer_type_t buft) const;
        size_t resident_size() const { return 2*pos_at.size() + slot.size() + 2*cap; }
        size_t emit_patches(const std::function<void(size_t, const int32_t *, size_t)> & write) const;
        void rebuild(const llama_kv_cells & c, llama_seq_id seq);
        // what the device holds of the tables is not known: they go whole on the next emit_patches
        void lose_upload() const { full_upload = true; dirty_cells.clear(); dirty_pools.clear(); }

        // false: the cells disagree with the tables (two cells at one position, a position off the tables); rebuild
        bool apply(const llama_kv_cells & c, llama_seq_id seq, const std::vector<uint32_t> & chg);

    private:
        bool insert(uint32_t j, llama_pos p);
        void remove(uint32_t j);

        std::vector<std::pair<uint32_t, int32_t>> upd;
        mutable std::shared_ptr<llama_kpool_backend_tensor> storage;
        mutable bool full_upload = true;
        mutable std::vector<uint32_t> dirty_cells, dirty_pools;
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
    void reserve_views(size_t count) { n_views_max = std::max(n_views_max, count); }

    // the device cannot make the maps from a bank (ggml_kpool_materialize): a unified pool keeps its packed maps
    bool bank_unsupported = false;
    // LLAMA_KPOOL_SHARED_VIEWS_LEGACY=1, read when the cache is made: a unified pool keeps its packed maps
    const bool packed_legacy = ggml_env_switch("LLAMA_KPOOL_SHARED_VIEWS_LEGACY");
    // LLAMA_KPOOL_PATCH_INPUT_LEGACY=1, read when the cache is made: every change of the bank is a write of its own
    const bool patch_writes_legacy = ggml_env_switch("LLAMA_KPOOL_PATCH_INPUT_LEGACY");
    std::shared_ptr<llama_kpool_backend_tensor> resident_bank(ggml_backend_buffer_type_t buft,
            int64_t count, uint32_t slots, uint32_t devices);
    // a graph that carried the views' changes was not computed: every view goes whole on its next use
    void resend() { for (auto & it : views) { it.second.lose_upload(); } }

private:
    void sync(const llama_kv_cells & cells, uint32_t r);

    // a sequence past this many takes the maps built from the cells
    size_t n_views_max = 8;
    std::shared_ptr<llama_kpool_backend_tensor> bank;

    std::map<llama_seq_id, view> views;

    // the call in which a cells object last took its changes
    std::map<const llama_kv_cells *, uint64_t> synced;

    std::vector<uint32_t> chg;

    uint64_t call = 1;
    stats    st;
};

using llama_kpool_cells_fn = std::function<const llama_kv_cells &(llama_seq_id)>;

// the pools a ubatch's maps hold: its own sequences' runs (every pool from a sequence's lowest position to its highest,
// summed over a stream's sequences), padded to pool_pad, plus 2 per sequence; never more than llama_kpool_n_pools of n_kv,
// and never so few that select_k, and so the dump pools, change
uint32_t llama_kpool_n_pools_ubatch(const llama_kpool_cells_fn & cells_of, const llama_ubatch & ubatch, uint32_t n_kv,
        uint32_t kpool, uint32_t n_stream, uint32_t indexer_top_k, uint32_t pool_pad = 64);

// the pools the graph reserved at load holds: the whole cache's (llama_kpool_n_pools), the worst case, or, under a limit
// of seq_cells positions a sequence (LLAMA_KV_SEQ_CELLS), those of the steps a server's slots make: each sequence of the
// ubatch in a stream of its own, its positions spanning seq_cells at most, with the widest padding a step takes
// (pool_pad_max) on top. That is not a worst case. The packed maps hold every sequence's runs in one stream (cells two
// sequences share after a seq_cp, rows not grouped by sequence, LLAMA_KPOOL_SHARED_VIEWS_LEGACY, a device without the
// bank), and a sequence with a hole in its positions spans more than the cells it holds: such a step's graph can be
// larger than the reserve and is then allocated when it comes. Never more than the whole cache's and never so few that
// select_k changes
uint32_t llama_kpool_n_pools_reserve(uint32_t n_kv, uint32_t kpool, uint32_t n_ps, uint32_t indexer_top_k,
        uint32_t pool_pad_max, uint32_t seq_cells);

// the pools a step's own runs are padded to: 64, or as many pools as the cells a prompt step's cache length is padded to
// (llama_kv_cache::get_n_kv), so its pool count changes when its cache length does and its graph is reused between
uint32_t llama_kpool_pool_pad(uint32_t n_pad_kv, uint32_t kpool);

// the pools a compression makes at once: a rebuild re-emits every pool (n_new_max = n_pools, 131,074 at 524,288 cells), and
// their members and softmax at once took about 1.07 GB of compute buffer at d_idx 128 that no reserve sized; a chunk takes
// about 34 MB (test-kpool-input: 8,388,608 bytes for 4,096 pools at d_idx 32), and a decode's few pools are one chunk
constexpr int64_t LLAMA_KPOOL_COMPRESS_CHUNK = 4096;

// the write of a chunk's pooled keys (src: [d_idx, rows]) at its rep rows (I64 [rows]); and the name of a tensor
using llama_kpool_write_fn = std::function<ggml_tensor *(ggml_tensor * src, ggml_tensor * rows)>;
using llama_kpool_name_fn  = std::function<void(ggml_tensor * t, const char * name)>;

// The pooled keys of the new pools: each pool's r member keys mixed by a softmax over their gates (ape added before it, over
// the slot axis), written at the pool's rep row. kg_rows: [2*d_idx, n_kv, n_stream], each cell's key then gate; new_pool_cells
// (I32 [r*n_new_max, n_stream]) and new_pool_reps (I64 [n_new_max*n_stream]) as llm_graph_input_kpool holds them; ape: [r, d_idx].
// The pools go `chunk` at a time, each chunk's write expanded into gf before the next chunk is made, so the allocator gives the
// next chunk the memory of the last. One chunk is the graph this compression always was
void llama_kpool_compress(
        ggml_context               * ctx,
        ggml_cgraph                * gf,
        ggml_tensor                * kg_rows,
        ggml_tensor                * new_pool_cells,
        ggml_tensor                * new_pool_reps,
        ggml_tensor                * ape,
        int64_t                      r,
        int64_t                      chunk,
        const llama_kpool_write_fn & write,
        const llama_kpool_name_fn  & name);

// A row of the sparse attention's mask is live at its top-k pools' cells and at its tail, the cells of its sequence at
// positions [(q + 1)/r*r, q] (sel_mask's finite columns): at most r - 1 unless the sequence holds two cells at one position.
// The most cells a tail of this ubatch holds, at least r - 1, from the cells as the ubatch left them
int64_t llama_kpool_tail_max(const llama_kpool_cells_fn & cells_of, const llama_ubatch & ubatch, uint32_t kpool);

// the live cells a mask row may hold (ggml_flash_attn_ext_set_n_kv_max): n_top_k top-k cells and an n_tail tail, padded to 32
// cells, a step of the CUDA kernel
int64_t llama_kpool_n_kv_max(int64_t n_top_k, int64_t n_tail);

// A unified pool holds every sequence in one stream, and its maps pack the sequences' pools on one axis: built from every
// cell for each sequence, and every row scored against every sequence's pools. Its ubatch gives each sequence a stream of
// its own instead (one pool map a sequence over the one cache stream, each served from its view) when every token names
// one sequence, the sequences come as equal runs of rows and no cell holds two sequences.
// seqs: each run's sequence, in row order
bool llama_kpool_shared_streams(const llama_kv_cells & cells, const llama_ubatch & ubatch, std::vector<llama_seq_id> & seqs);

// the maps of llama_kv_cache_set_input_kpool from a cells accessor. views == nullptr builds them from the cells
// every call; else a sequence alone in its stream is served from its view, byte for byte the same maps.
// mask_cells and mask_rows (I32, ggml_kpool_mask's cells and rows) instead of sel_mask and cand_mask: the facts the
// device builds both masks from, O(n_kv + n_rows) where the masks are O(n_kv*n_rows); ggml_kpool_mask on them gives
// the masks byte for byte. Returns the most cells a row's tail was granted (sel_mask's finite columns in a row)
int64_t llama_kpool_set_input(
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
              uint32_t         kpool,
              ggml_tensor    * mask_cells = nullptr,
              ggml_tensor    * mask_rows  = nullptr,
              ggml_tensor    * tail_cells = nullptr);

// `kv` must be the ATTENTION (MLA) cache; the indexer cache shares its slot layout.
// LLAMA_KPOOL_INPUT_LEGACY=1 builds the maps from the cells every call; LLAMA_KPOOL_INPUT_CHECK=1 builds them both
// ways and aborts on a difference.
//   pool_cells  pool member -> cell, 0 if not resident; past pool_bias's n_pools, n_dump dump pools of cells n_kv + d*kpool + t
//   pool_bias   computed, NOT gathered at the last member, which an incomplete pool lacks
//   sel_mask    n_kv columns, then kpool*n_dump dump columns of -inf (n_kv is cand_mask's)
//   cand_mask   bounds top-k spills a partial seq_rm would let escape
// pool_reps / new_pool_cells / new_pool_reps are nullptr when the cache is off, and an entry
// is emitted only for filled == kpool: cell 0 is real, so writing its 0 slot would clobber.
// Returns llama_kpool_set_input's
int64_t llama_kv_cache_set_input_kpool(
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
              uint32_t         kpool,
        // the device's mask facts in place of sel_mask and cand_mask (llama_kpool_set_input)
              ggml_tensor    * mask_cells = nullptr,
              ggml_tensor    * mask_rows  = nullptr,
              ggml_tensor    * tail_cells = nullptr);

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
    int64_t n_tail    = 0;
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

    void input_lost() override;

    bool can_reuse(const llm_graph_params & params) override;

    // gather the geometry the builder sizes the tensors with, for the check below; shared: a stream a sequence (shared_streams)
    static llm_graph_input_kpool_dims current_dims(
            const llama_kv_cache_context * mctx_attn,
            const llama_ubatch           & ubatch,
            const llama_cparams          & cparams,
            const llama_hparams          & hparams,
            uint32_t                       kpool,
            bool                           shared = false);

    // The sequences of a unified pool's ubatch as streams of their own (llama_kpool_shared_streams), when each can be
    // served from its view out of the pool's bank. seqs: the streams' sequences. false: the pool's packed maps, or one
    // sequence's own table where the pool holds one sequence at most
    static bool shared_streams(
            const llama_kv_cache_context * mctx_attn,
            const llama_ubatch           & ubatch,
            const llama_cparams          & cparams,
            const llama_hparams          & hparams,
            uint32_t                       kpool,
            std::vector<llama_seq_id>    & seqs);

    // a unified pool of several sequences takes its resident maps from the bank alone: a view's changes go to one table
    static bool shared_pool(const llama_cparams & cparams, const llama_kpool_views & views);

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
    uint32_t n_seq_devices = 1;

    // set at build time: re-emit every pool after a position mutation
    bool rebuild = false;
    // the sequences the last set_input cleared of that mark: marked again if its graph did not run (input_lost)
    std::vector<llama_seq_id> rebuilt_seqs;

    // set at build time: the most cells a row's tail holds (llama_kpool_tail_max), which sizes the sparse attention's
    // n_kv_max; set_input asserts no row is granted more
    int64_t n_tail = 0;

    // exact, since pool_bias only holds 0.0f or -INFINITY. nullptr if the fused path is off
    ggml_tensor * pool_bias_f16 = nullptr; // F16 [n_pools, n_tps, 1, n_stream]

    // The ordinary one-sequence stream uses one stable canonical allocation for
    // every graph slot; only the mutation consumer's changed ranges are uploaded.
    ggml_backend_sched_t sched = nullptr;
    std::shared_ptr<llama_kpool_backend_tensor> resident_table;
    llama_seq_id resident_seq = -1;
    std::vector<llama_seq_id> resident_seqs; // bank users, in device/sequence order
    bool shared = false; // the bank users are a unified pool's sequences, one stream each (llama_kpool_shared_streams)
    ggml_tensor * resident_rows = nullptr; // [6,Q] or [7,Q,streams], see ggml_kpool_materialize
    ggml_tensor * resident_patches = nullptr; // I32 [3,P]: the bank's changes of the decode, see ggml_kpool_patch
    uint64_t resident_patch_count = 0;
    uint64_t resident_upload_bytes = 0;
    uint64_t resident_upload_calls = 0;
    bool reuse_only = false;
    std::vector<std::shared_ptr<llama_kpool_backend_tensor>> draft_rows;
    std::vector<llama_seq_id> draft_seqs;
    ggml_tensor * draft_ids = nullptr; // device-local sequence slots in the selection bank

    llama_kpool_draft * draft = nullptr;
    ggml_tensor * draft_top_k = nullptr; // retained selection in the current batch's sequence order

    ggml_tensor * sel_mask   = nullptr;   // F16 [n_kv + kpool*n_dump, n_batch, 1, n_stream]; the dump columns -inf
    ggml_tensor * cand_mask  = nullptr;   // F16 [n_kv, n_batch, 1, n_stream]

    // the facts ggml_kpool_mask builds both masks from on the device, sel_mask and cand_mask then its results: a cell's
    // position and pool per sequence of a stream, a row's position, sequence, tail and pools seen. nullptr when the host
    // writes the masks (LLAMA_KPOOL_MASK_HOST_LEGACY=1)
    ggml_tensor * mask_cells = nullptr;   // I32 [n_kv, 2, n_ps, n_stream]
    ggml_tensor * mask_rows  = nullptr;   // I32 [4, n_batch, n_stream]
    ggml_tensor * tail_cells = nullptr;  // I32 [n_tail, n_batch, n_stream], physical cells or -1

    const llama_kv_cache_context * mctx_attn;
    const llama_kv_cache_context * mctx_idx;

    const uint32_t kpool;
};
