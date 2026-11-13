#pragma once

#include "llama-batch.h"
#include "llama-graph.h"
#include "llama-memory.h"

#include <map>
#include <set>
#include <vector>

//
// llama_memory_recurrent
//

// TODO: extract the cache state used for graph computation into llama_memory_recurrent_context_i
//       see the implementation of llama_kv_cache_context_i for an example how to do it
class llama_memory_recurrent : public llama_memory_i {
public:
    llama_memory_recurrent(
            const llama_model & model,
                    ggml_type   type_r,
                    ggml_type   type_s,
                         bool   offload,
                     uint32_t   mem_size,
                     uint32_t   n_seq_max,
                     uint32_t   n_rs_seq,
        const layer_filter_cb & filter);

    ~llama_memory_recurrent() = default;

    //
    // llama_memory_i
    //

    llama_memory_context_ptr init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) override;

    llama_memory_context_ptr init_full() override;

    llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) override;

    void clear(bool data) override;

    bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) override;
    void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id)                                                          override;
    void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) override;

    llama_pos seq_pos_min(llama_seq_id seq_id) const override;
    llama_pos seq_pos_max(llama_seq_id seq_id) const override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    bool prepare(const std::vector<llama_ubatch> & ubatches);

    // find a contiguous slot of memory cells and emplace the ubatch there
    bool find_slot(const llama_ubatch & ubatch, bool allow_relocate = true);

    bool get_can_shift() const override;

    // state write/load

    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) override;

    llama_memory_recurrent * staged() override { return n_stage > 0 ? this : nullptr; }

    // LLAMA_RS_STAGE_ROWS=N: N spare rows of every state tensor on the devices, each a tensor of one row split as the
    // state is. A sequence's state is copied to one on the devices (llama_context::rs_stage), where no decode moves it
    // or writes over it, and from there to host memory by a thread (llama_context::state_seq_get_data_staged): a
    // context checkpoint then holds the caller for the copy on the devices alone. Off (0) unless set; f32 states on
    // one set of sequence devices only, as relocate.
    uint32_t n_stage = 0;
    std::vector<std::vector<ggml_tensor *>> r_stage; // [row][layer], nullptr where r_l is
    std::vector<std::vector<ggml_tensor *>> s_stage;

    // the row of the state tensors that holds the sequence's state, when it has one cell of its own; -1 otherwise
    int64_t stage_src(llama_seq_id seq_id) const;

    // the state tensors and the tensors of spare row `row`, layer by layer, for rs_stage
    void stage_tensors(uint32_t row, std::vector<ggml_tensor *> & states, std::vector<ggml_tensor *> & stages) const;

    // state_write for a sequence stage_src() has a row for, its tensor data read from spare row `row`
    void state_write_staged(llama_io_write_i & io, llama_seq_id seq_id, uint32_t row) const;

    uint32_t head = 0; // the location where the batch will be placed in the cache (see find_slot())
    uint32_t size = 0; // total number of cells, shared across all sequences
    uint32_t used = 0; // used cells (i.e. at least one seq_id)

    // number of recurrent-state snapshots per seq for rollback; tensors are widened to (1 + n_rs_seq) groups
    uint32_t n_rs_seq = 0;

    // per-seq rollback index
    std::vector<uint32_t> rs_idx;

    // per-seq rollback reach: min(n_rs_seq, rows - 1) for the seq's rows in the last ubatch that carried it. A batch
    // of n rows writes the snapshots 0..n-1 (the state s rows before its end): the state before the batch and those of
    // earlier batches are kept in no slot, so seq_rm refuses a rollback past this reach instead of reading one
    std::vector<uint32_t> rs_reach;

    void set_rs_idx(llama_seq_id seq_id, uint32_t idx);
    uint32_t n_devices() const { return n_seq_devices; }
    bool is_fixed_rows() const { return fixed_rows || n_seq_devices > 1; }
    uint32_t state_row(uint32_t cell, uint32_t snapshot, bool local = false) const;

    // computed before each graph build
    uint32_t n = 0;

    // first zero-ed state
    int32_t rs_z = -1;

    // LLAMA_GRAPH_ANY_SEQS=1 (set_relocate): the batch's sequences take the cells from the first on, whichever set
    // they are, and a state another sequence keeps in one of those cells is moved by a pass of its own before the
    // decode (relocate_states, llama_context::rs_relocate). The decode's graph then reads and writes cells 0 to n_seqs - 1
    // and copies no other state, so it is the same graph for every set of that many sequences: as find_slot gathers the
    // set at its lowest cell and has the first decode's graph copy the displaced states, a changed set built two graphs
    // (112, 99 and 128 ms for its first three target decodes of 66 rows on four cards, where a replay is 48; Nov 10, 2026)
    bool relocate = false;
    bool relocate_pending = false; // find_slot left a state to move
    uint32_t owed_seqs = 0;        // the sequences of the placement that left it: its first cells
    // With relocate, unless LLAMA_RS_GATHER_LEGACY=1 (read by set_relocate): a state of the batch that lies in a cell
    // another sequence of the batch is written to is moved before the decode as well. Every sequence of the batch then
    // reads its state from the cell it is written to, or from one the decode writes nothing to, and the decode's graph
    // may read a state as late as it writes the others' (GGML_TENSOR_FLAG_ROWS_DISJOINT, llm_graph_context::build_rs).
    bool relocate_disjoint = false;

    // asked for, before the first batch, by a memory that keeps nothing else by the place of a sequence's cell (the
    // hybrid memory)
    void set_relocate(bool value);

    // moves the states find_slot left to move, and leaves the batch's cells as the decode's graph expects them
    void relocate_states(uint32_t n_seqs);

    // A move find_slot left to relocate_states is owed until it is made. One that was never made (until e7598775e a
    // state restore's placement left one, and the next find_slot cleared the flag without a word) leaves cells that
    // name a row their state is not in, and the placement's own sequences at a position no decode took them to:
    // settle() drops the sequences of both instead of letting them be read. A sequence past the placement's cells
    // keeps its state. It runs where the placement has to be complete: before the next placement, before a restore
    // writes its rows, before a sequence is removed, copied or shifted, and before a state is read out (seq_lost,
    // which llama_context asks before it decodes or saves).
    // A dropped sequence has no cell, an error line says why, and seq_lost() stays true for it until it is removed
    // (seq_rm of all of it, a restore into it, seq_keep of another, clear): no decode and no save takes it before
    // that, a copy of it is dropped as well, and the whole context is not saved while one is left (seq_lost(-1)).
    void settle();
    bool seq_lost(llama_seq_id seq_id) override;

    // zero the rs_z row of every non-f32 S tensor on the host (see llm_graph_context::build_rs)
    void zero_rs_z();

    // TODO: optimize for recurrent state needs
    struct mem_cell {
        llama_pos pos  = -1;
        int32_t   src  = -1; // used to know where states should be copied from
        int32_t   src0 = -1; // like src, but only used when setting the inputs (allowing to copy once)
        int32_t   tail = -1;

        std::set<llama_seq_id> seq_id;

        bool has_seq_id(const llama_seq_id & id) const {
            return seq_id.find(id) != seq_id.end();
        }

        bool is_empty() const {
            return seq_id.empty();
        }

        bool is_same_seq(const mem_cell & other) const {
            return seq_id == other.seq_id;
        }
    };

    std::vector<mem_cell> cells;

    // per layer
    std::vector<ggml_tensor *> r_l;
    std::vector<ggml_tensor *> s_l;

private:
    //const llama_model & model;
    const llama_hparams & hparams;

    const uint32_t n_seq_max = 1;
    const uint32_t n_seq_devices;
    bool fixed_rows = false;
    llama_context * owner_context = nullptr;
    std::vector<uint32_t> zero_cells;
    std::vector<std::vector<uint32_t>> restore_cells;

    // sequences settle() dropped, until each is removed
    std::vector<llama_seq_id> lost;

    // frees cell i and names its sequences in `lost`
    void lose(uint32_t i, const char * why);

    // ggml contexts for the KV cache along with the allocated backend buffers:
    std::vector<std::pair<ggml_context_ptr, ggml_backend_buffer_ptr>> ctxs_bufs;

    size_t total_size() const;

    size_t size_r_bytes() const;
    size_t size_s_bytes() const;

    void state_write_meta(llama_io_write_i & io, const std::vector<std::pair<uint32_t, uint32_t>> & cell_ranges, llama_seq_id seq_id = -1) const;
    void state_write_data(llama_io_write_i & io, const std::vector<std::pair<uint32_t, uint32_t>> & cell_ranges) const;

    bool state_read_meta(llama_io_read_i & io, uint32_t cell_count, llama_seq_id dest_seq_id = -1);
    bool state_read_data(llama_io_read_i & io, uint32_t cell_count);
};

class llama_memory_recurrent_context : public llama_memory_context_i {
public:
    // used for errors
    llama_memory_recurrent_context(llama_memory_status status);

    // used to create a full-cache or update context
    llama_memory_recurrent_context(
            llama_memory_recurrent * mem);

    // used to create a batch processing context from a batch
    llama_memory_recurrent_context(
            llama_memory_recurrent * mem,
            std::vector<llama_ubatch> ubatches);

    virtual ~llama_memory_recurrent_context();

    //
    // llama_memory_context_i
    //

    bool next()  override;
    bool apply() override;

    llama_memory_status  get_status() const override;
    const llama_ubatch & get_ubatch() const override;

    //
    // llama_memory_recurrent_context specific API
    //

    uint32_t get_n_rs() const;
    uint32_t get_head() const;
    int32_t  get_rs_z() const;
    uint32_t get_size() const;
    // whether no sequence of the batch reads its state from a cell another one is written to (relocate_disjoint)
    bool     rows_disjoint() const;

    ggml_tensor * get_r_l(int32_t il) const;
    ggml_tensor * get_s_l(int32_t il) const;

    bool is_fixed_rows() const { return mem->is_fixed_rows(); }
    int32_t s_copy(int i) const;
    uint32_t n_devices() const { return mem->n_devices(); }
    uint32_t s_write(llama_seq_id seq, uint32_t snapshot) const;

private:
    const llama_memory_status status;

    llama_memory_recurrent * mem;

    size_t i_next = 0;

    std::vector<llama_ubatch> ubatches;

    //
    // data needed for building the compute graph for the current ubatch:
    // TODO: extract all the state like `head` and `n` here
    //

    const bool is_full = false;
};
