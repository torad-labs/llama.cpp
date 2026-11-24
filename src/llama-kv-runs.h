#pragma once

#include "llama.h"
#include "llama-hparams.h"

#include <cstdint>
#include <vector>

class llama_kv_cells;
struct llama_ubatch;

// len cells of a sequence from cell on, one stride apart, whose positions go up one by one from pos. The cells of a run
// are all shared with another sequence, or none of them is
struct llama_kv_run {
    uint32_t  cell;
    llama_pos pos;
    uint32_t  len;
    uint32_t  stride;
    bool      shared;
};

// The runs of the sequences a decode step reads (ggml_flash_attn_ext_set_kv_runs), kept from one step to the next through
// the cells' change log (llama_kv_cells_log), whose one consumer this is. Cells added at the end of a sequence extend its
// last run or start a new one. A change to a cell a sequence's runs hold, other than which other sequences hold it, and
// a log that lost its list, build the runs again from the sequence's cells
class llama_kv_runs {
public:
    explicit llama_kv_runs(uint32_t n_swa = 0, llama_swa_type swa_type = LLAMA_SWA_TYPE_NONE)
        : n_swa(n_swa), swa_type(swa_type) {}

    // the runs of seq, nullptr when it has more than n_max
    const std::vector<llama_kv_run> * get(const llama_kv_cells & cells, llama_seq_id seq, uint32_t n_max);

    // Write [4, n_entries, n_rows + 1] i32 runs input (up to 8 tip/draft rows per sequence).
    // On global caches, stride-1 runs shared by at least two sequences go to the shared list.
    void write(const llama_kv_cells & cells, const llama_ubatch & ubatch, int32_t * data, int64_t n_entries);

    // Same layout as write(), but only writes active entries [0 .. active_counts[s]) per slice without zeroing tails.
    void write_active(const llama_kv_cells & cells, const llama_ubatch & ubatch, int32_t * data, int64_t n_entries,
            std::vector<int32_t> & active_counts);

private:
    struct seq_runs {
        std::vector<llama_kv_run> runs;
        uint32_t  n_cells  = 0;
        llama_pos pos_end  = -1; // the last position
        size_t    tail     = 0;  // the run that holds pos_end
        uint32_t  cell_max = 0;  // no run holds a cell past it
        bool      built    = false;
    };

    void sync(const llama_kv_cells & cells);
    void build(const llama_kv_cells & cells, llama_seq_id seq, seq_runs & sr) const;

    const uint32_t       n_swa = 0;
    const llama_swa_type swa_type = LLAMA_SWA_TYPE_NONE;
    const llama_kv_cells * synced = nullptr;
    std::vector<seq_runs> seqs;
    std::vector<uint32_t> chg;
};
