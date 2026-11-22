#pragma once

// Section 3.5 of doc 18 as a skeleton behind LLAMA_INKLING2_KV_MIXED (off by default): the newest cells a sequence
// writes - about its last LLAMA_INKLING2_KV_MIXED_HOT positions, 1,024 as served - mirror into a small high-precision
// hot pool per global layer while everything older stays in the existing low-bit cold pool.
//
// What this skeleton lands:
//   - the dual-pool allocation plumbing: each cache it applies to holds one llama_kv_mixed_pool whose F16 K/V rows
//     live beside the cold pool's buffers;
//   - store-side routing by recency: the cells every ubatch wrote are mirrored lazily - before the next batch's
//     metadata is applied, so the previous graph's values are final - when the recency predicate says near;
//   - the dense-path reference read merge below: rows materialized from exactly one source per cell, hot where a
//     valid stamp exists and the cell is near, cold otherwise. On a fixture where every visible cell is forced to
//     one side it equals reading that pool alone bit for bit, which test-inkling-kvmix-* asserts.
//
// What remains for the real store (the handoff note in reports/kv-q40-dispatch-pending.md, "section 3.5 skeleton",
// carries the full design): attention consuming the split, and true pre-quantization capture - today the mirror
// reads back the cold pool's own bytes, so hot precision equals cold until a parallel F16 scatter of the pre-store
// k_cur/v_cur values is wired beside cpy_k/cpy_v with its recency index tensor.

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include "ggml.h"
#include "ggml-backend.h"

// one global layer's hot mirror: F16 K/V rows ringed over writes, stamps saying whose value each slot holds
struct llama_kv_mixed_pool {
    llama_kv_mixed_pool() = default;
    ~llama_kv_mixed_pool();

    llama_kv_mixed_pool(const llama_kv_mixed_pool &) = delete;
    llama_kv_mixed_pool & operator=(const llama_kv_mixed_pool &) = delete;

    // allocates the F16 hot rows on `buft` beside the cold pool; false when allocation failed
    bool init(int64_t row_values, uint32_t hot_rows, ggml_backend_buffer_type_t buft);

    int64_t  row_values() const { return n_row_values; }
    uint32_t hot_rows()   const { return n_hot_rows; }

    // store-side routing by recency: for every given written cell the predicate passes, decode that cell's cold row
    // to f32 and encode it into the next F16 ring slot. The cache calls this before applying a new batch, when the
    // previous graph's scatter is final.
    void mirror(ggml_tensor * cold,
            const uint32_t * cells, size_t n_cells,
            const std::function<bool(uint32_t)> & near_cell);

    // dense-path reference read merge: materialize one F32 row per requested cell out of exactly one source - the
    // hot slot whose stamp names the cell while the predicate holds it near, the cold row otherwise. Rows come back
    // in caller memory as `n_cells x row_values` floats.
    void merge_rows(ggml_tensor * cold,
            const uint32_t * cells, size_t n_cells,
            const std::function<bool(uint32_t)> & near_cell,
            float * out) const;

    // drops every stamp and rewinds the ring: restores rebuild the recency picture from what follows
    void reset_stamps();

private:
    static constexpr uint64_t kNoCell = 0; // stamps name live cell ids; 0 marks an empty slot

    struct stamp {
        uint64_t cell = kNoCell;
    };

    int64_t            n_row_values = 0;
    uint32_t           n_hot_rows   = 0;
    uint64_t           cursor       = 0;
    ggml_context *     ctx          = nullptr;
    ggml_backend_buffer_t buf       = nullptr;
    ggml_tensor *      hot          = nullptr; // F16 [row_values, hot_rows]
    std::vector<stamp> stamps;
};

using llama_kv_mixed_pool_ptr = std::unique_ptr<llama_kv_mixed_pool>;

// LLAMA_INKLING2_KV_MIXED, off by default
bool llama_kv_mixed_enabled();

// LLAMA_INKLING2_KV_MIXED_HOT: recency extent a sequence's newest positions span (as served: 1,024)
uint32_t llama_kv_mixed_hot_cells();
