#pragma once

#include "llama.h"
#include "llama-ext.h"
#include "llama-cparams.h"
#include "llama-graph.h"
#include "llama-adapter.h"
#include "llama-impl.h"
#include "llama-memory.h"

#include "ggml-cpp.h"
#include "ggml-opt.h"

#include <map>
#include <memory>
#include <set>
#include <vector>

struct llama_model;
struct llama_nextn_store;
class llama_batch_allocr;

class llama_io_read_i;
class llama_io_write_i;

// "memory" as in abstract memory for the context
struct llama_memory_i;
struct llama_memory_context_i;

// stores copy of the memory in device buffer. used for fast state save/load
struct llama_memory_buffer {
    int n_tensors = 0;
    size_t total_size = 0;

    ggml_backend_buffer_ptr buf;

    ggml_context_ptr ctx;

    std::vector<ggml_tensor *> org;
    std::vector<ggml_tensor *> cpy;
};

using llama_memory_buffers = std::map<ggml_backend_buffer_type_t, llama_memory_buffer>;

struct llama_context {
    // init scheduler and compute buffers, reserve worst-case graphs
    llama_context(
            const llama_model & model,
                  llama_context_params params);

    ~llama_context();

    // reserve a new backend scheduler (if needed)
    // for example, when:
    //   - changing loras
    //   - changing samplers
    //   - changing attention type
    //   - etc.
    void sched_reserve();

    void synchronize();

    // synchronize(), unless the context has synchronized since its last decode or encode: what an accessor of their
    // outputs waits for
    void synchronize_outputs();

    // every row r of each state tensor takes row rows[r] (llama_memory_recurrent::relocate_states)
    void rs_relocate(const std::vector<ggml_tensor *> & states, const std::vector<int32_t> & rows);

    const llama_model   & get_model()   const;
    const llama_cparams & get_cparams() const;

    ggml_backend_sched_t get_sched() const;

    uint32_t n_ctx()     const;
    uint32_t n_ctx_seq() const;
    uint32_t n_batch()   const;
    uint32_t n_ubatch()  const;
    uint32_t n_seq_max() const;

    uint32_t n_threads()       const;
    uint32_t n_threads_batch() const;

    llama_memory_t get_memory() const;

    // return true if the memory was updated
    bool memory_update(bool optimize);

    enum llama_pooling_type pooling_type() const;

    float * get_logits();
    float * get_logits_ith(int32_t i);

    float * get_embeddings();
    float * get_embeddings_ith(int32_t i);
    float * get_embeddings_seq(llama_seq_id seq_id);

    float * get_embeddings_nextn();
    float * get_embeddings_nextn_ith(int32_t i);

    float * get_embeddings_layer_inp(uint32_t lid);

    // logit lens: n_vocab floats of lens entry k for batch token i, nullptr when the last decode
    // carried no lens (lens off, or a ubatch with more output rows than the lens capacity)
    float * get_lens_ith(int32_t k, int32_t i);
    float * get_lens_channel_ith(int32_t k, int32_t channel, int32_t i);
    int32_t n_lens_channels() const;

    llama_token * get_sampled_tokens() const;
    llama_token   get_sampled_token_ith(int32_t idx);

    float * get_sampled_logits_ith(int32_t idx);
    size_t  get_sampled_logits_count(int32_t idx);

    float * get_sampled_probs_ith(int32_t idx);
    size_t  get_sampled_probs_count(int32_t idx);

    const llama_token * get_sampled_candidates_ith(int32_t idx);
    size_t get_sampled_candidates_count(int32_t idx);

    // a row read as candidates (llama_set_output_candidates): its k largest logits into row, false for a row read whole
    bool get_candidates_ith(int32_t idx, llama_sampling_row & row);

    void attach_threadpool(
            ggml_threadpool_t threadpool,
            ggml_threadpool_t threadpool_batch);

    void detach_threadpool();

    void set_n_threads(int32_t n_threads, int32_t n_threads_batch);

    void set_abort_callback(bool (*abort_callback)(void * data), void * abort_callback_data);

    void set_embeddings (bool value);
    void set_embeddings_nextn(bool value, bool masked);
    void set_embeddings_layer_inp(uint32_t lid, bool enable);
    void set_lens_layers(const int32_t * layers, int32_t n_layers);
    void set_lens_channels(bool enabled);
    void set_nextn_layer_offset(int32_t offset);
    bool mtp_draft_step(int32_t step);
    bool set_mtp_graph_reuse(int32_t n_max);
    void set_causal_attn(bool value);
    void set_warmup(bool value);

    void set_adapters_lora(llama_adapter_lora ** adapters, size_t n_adapters, float * scales);

    bool adapters_lora_are_same(llama_adapter_lora ** adapters, size_t n_adapters, float * scales);

    bool set_adapter_cvec(
            const float * data,
                 size_t   len,
                int32_t   n_embd,
                int32_t   il_start,
                int32_t   il_end);

    // process a single ubatch with a specific graph type
    // if memory_context is provided, it will be applied first to the context's memory
    // ret contains the status of the graph computation
    // returns nullptr only if ret != GGML_STATUS_SUCCESS
    llm_graph_result * process_ubatch(
                const llama_ubatch & ubatch,
                    llm_graph_type   gtype,
            llama_memory_context_i * mctx,
                       ggml_status & ret);

    int encode(const llama_batch & batch_inp);
    int decode(const llama_batch & batch_inp);

    //
    // state save/load
    //

    size_t state_get_size();
    size_t state_get_data(      uint8_t * dst, size_t size);
    size_t state_set_data(const uint8_t * src, size_t size);

    size_t state_seq_get_size(llama_seq_id seq_id, llama_state_seq_flags flags);

    size_t state_seq_get_data(llama_seq_id seq_id,       uint8_t * dst, size_t size, llama_state_seq_flags flags);
    size_t state_seq_set_data(llama_seq_id seq_id, const uint8_t * src, size_t size, llama_state_seq_flags flags);

    bool state_load_file(
            const char * filepath,
           llama_token * tokens_out,
                size_t   n_token_capacity,
                size_t * n_token_count_out);

    bool state_save_file(
            const char * filepath,
     const llama_token * tokens,
                size_t   n_token_count);

    size_t state_seq_load_file(
          llama_seq_id   seq_id,
            const char * filepath,
           llama_token * tokens_out,
                size_t   n_token_capacity,
                size_t * n_token_count_out);

    size_t state_seq_save_file(
          llama_seq_id   seq_id,
            const char * filepath,
     const llama_token * tokens,
                size_t   n_token_count);

    //
    // perf
    //

    llama_perf_context_data perf_get_data() const;
    void perf_reset();

    llama_memory_breakdown memory_breakdown() const;

    //
    // training
    //

    void opt_init(struct llama_model * model, struct llama_opt_params lopt_params);

    // TODO: more flexible combinations of logical/physical batch size and context size
    void opt_epoch(
            ggml_opt_dataset_t      dataset,
            ggml_opt_result_t       result_train,
            ggml_opt_result_t       result_eval,
            int64_t                 idata_split,
            ggml_opt_epoch_callback callback_train,
            ggml_opt_epoch_callback callback_eval);

    void opt_epoch_iter(
            ggml_opt_dataset_t               dataset,
            ggml_opt_result_t                result,
            const std::vector<llama_token> & tokens,
            const std::vector<llama_token> & labels_sparse,
            llama_batch                    & batch,
            ggml_opt_epoch_callback          callback,
            bool                             train,
            int64_t                          idata_in_loop,
            int64_t                          ndata_in_loop,
            int64_t                          t_loop_start);

private:
    //
    // output
    //

    // Make sure enough space is available for outputs.
    // Returns max number of outputs for which space was reserved.
    uint32_t output_reserve(int32_t n_outputs);

    void output_reorder();

    // put the last decode's unmasked nextn rows in batch order (nextn_order), once their reads are done
    void nextn_reorder();

    // map the output row index `i` to batch index
    int64_t output_resolve_row(int32_t i) const;

    // async-copy enabled layer-input tensors (per cparams.output_layer_inp)
    // from backend into host-side embd_layer_inp buffers
    void extract_layer_inputs(const llm_graph_result * res, size_t token_offset, size_t n_tokens);

    //
    // graph
    //

public:
    uint32_t graph_max_nodes(uint32_t n_tokens) const;

    // can reuse the llm_graph_result instance of the context (for example to update a memory module)
    llm_graph_result * get_gf_res_reserve() const;

    // returns the result of ggml_backend_sched_graph_compute_async execution
    // sched_gf: the scheduler that allocated gf (nullptr: the context's own)
    ggml_status graph_compute(ggml_cgraph * gf, bool batched, ggml_backend_sched_t sched_gf = nullptr);

    // reserve a graph with a dummy ubatch of the specified size
    ggml_cgraph * graph_reserve(
        uint32_t n_tokens, uint32_t n_seqs, uint32_t n_outputs, const llama_memory_context_i * mctx, bool split_only = false, size_t * sizes = nullptr);

    bool set_sampler(llama_seq_id seq_id, llama_sampler * sampler);

    void set_output_candidates(int32_t k);
    void set_seq_candidates(llama_seq_id seq_id, bool enable);

    // the MTP store (llama-ext.h): this context is the target, head its MTP head
    bool    nextn_bind(llama_context & head, int32_t n_keep);
    bool    nextn_keep_rows(llama_seq_id seq, const int32_t * rows, int32_t n);
    int32_t nextn_kept_row(llama_seq_id seq, int32_t row) const;
    bool    nextn_kept_get(int32_t ref, float * dst);
    bool    nextn_kept_set(int32_t ref, const float * src);
    int32_t nextn_store_rows() const;
    void    nextn_set_h_rows(const int32_t * src, int32_t n, const float * carry, int32_t n_carry);
    bool    nextn_get_rows(const int32_t * rows, int32_t n, float * dst);
    int64_t nextn_rows_to_host() const { return n_nextn_to_host; }

private:
    // copy the last decode's rows from the MTP store into embd_nextn when they are only there
    void nextn_refill();

    // sched_gf: the scheduler the graph is built for (nullptr: the context's own)
    llm_graph_params graph_params(
                        llm_graph_result * res,
                      const llama_ubatch & ubatch,
            const llama_memory_context_i * mctx,
                          llm_graph_type   gtype,
                    ggml_backend_sched_t   sched_gf = nullptr) const;

    llm_graph_cb graph_get_cb(ggml_backend_sched_t sched_gf) const;

    // the graphs process_ubatch keeps (gf_res_prev and the slots') are rebuilt before their next use
    void graph_results_reset();

    void mtp_index_capture();

    // disable auto fused ops (Flash Attention, Gated Delta Net) whose op lands on a device
    // that differs from the layer it belongs to (usually due to missing backend support)
    void resolve_fused_ops(const llama_memory_context_i * mctx, uint32_t n_seqs);

    // TODO: read/write lora adapters and cvec
    size_t state_write_data(llama_io_write_i & io);
    size_t state_read_data (llama_io_read_i  & io);

    size_t state_seq_write_data(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags);
    size_t state_seq_read_data (llama_io_read_i  & io, llama_seq_id seq_id, llama_state_seq_flags flags);

    //
    // members
    //

    const llama_model & model;

    llama_cparams cparams;

    llama_adapter_cvec_ptr  cvec;
    llama_adapter_loras_ptr loras;

    llama_cross cross; // TODO: tmp for handling cross-attention - need something better probably

    llama_memory_ptr memory;

    struct mtp_index_pending {
        ggml_backend_t backend = nullptr;
        ggml_tensor * tensor = nullptr;
        uint64_t generation = 0;
        int64_t n_kv = 0;
        std::vector<llama_seq_id> seqs;
        std::vector<llama_pos> positions;
    } mtp_index;

    // decode output (2-dimensional array: [n_outputs][n_vocab])
    buffer_view<float> logits = {nullptr, 0};
    bool logits_valid = false; // any raw logits were copied for this batch

    // embeddings output (2-dimensional array: [n_outputs][n_embd])
    // populated only when pooling_type == LLAMA_POOLING_TYPE_NONE
    buffer_view<float> embd = {nullptr, 0};

    // hidden state required by the nextn layers (2-dimensional array: [n_outputs][n_embd])
    // populated only when cparams.embeddings_nextn is enabled and the model graph
    // sets llm_graph_result::t_h_nextn
    buffer_view<float> embd_nextn = {nullptr, 0};

    // the MTP store (llama_set_nextn_source), held by the target and its head: a target batch of min_rows rows or more
    // writes its hidden rows there by batch index instead of to embd_nextn, and the head reads them there
    std::shared_ptr<llama_nextn_store> nextn_store;
    bool nextn_is_target = false;

    llm_graph_nextn_mode nextn_mode = LLM_GRAPH_NEXTN_NONE; // the decode in progress writes or reads the store
    int64_t              nextn_base = 0;                    // the store row of its batch row 0
    std::vector<int32_t> nextn_src;                         // head: the next decode's rows (llama_set_nextn_h_rows)
    bool                 nextn_src_set = false;

    bool    embd_nextn_stale = false; // target: the last decode's rows are in the store, not yet in embd_nextn

    // Unmasked nextn rows are read whole, one ubatch after another, and nextn_order[k] is the batch index of the row
    // read into row k. An equal split interleaves sequences, so a ubatch's rows are not in batch order; nextn_reorder
    // moves them once the reads are done. Reading each run of rows at its own offset instead would fail where a backend
    // reads a tensor asynchronously only whole (the meta backend, under -sm tensor). Empty: nothing to move
    std::vector<int32_t> nextn_order;
    int64_t n_nextn_to_host  = 0;     // hidden rows copied to host memory

    // host buffers for output layer input embeddings, per layer
    // populated when cparams.output_layer_inp[il] is true
    std::vector<buffer_view<float>> embd_layer_inp;

    // logit lens output ([n_lens][n_lens_channels][n_lens_rows][n_vocab], one channel without
    // --lens-channels); rows follow output_ids like logits. lens_valid is false when any ubatch of
    // the last decode skipped it.
    buffer_view<float> lens = {nullptr, 0};
    bool lens_valid  = false;
    bool lens_warned = false; // the model graph records no lens (architecture without lens_record)

    struct sampling_info {
        // !samplers.empty() to check if any samplers are active
        std::map<llama_seq_id, llama_sampler *> samplers;

        buffer_view<float>       logits     = {nullptr, 0};
        buffer_view<llama_token> sampled    = {nullptr, 0};
        buffer_view<float>       probs      = {nullptr, 0};
        buffer_view<llama_token> candidates = {nullptr, 0};

        std::vector<uint32_t> logits_count;
        std::vector<uint32_t> probs_count;
        std::vector<uint32_t> candidates_count;

        // optimization
        std::vector<llama_token> token_ids_full_vocab;
    };

    sampling_info sampling;

    // output rows as their largest logits (llama_set_output_candidates): a decode reads the records
    // ggml_top_k_candidates packed for a row, and the row's first reader cuts them to the k largest
    struct candidates_info {
        enum : uint8_t { ROW_WHOLE, ROW_RECORDS, ROW_DRAFT_RECORDS, ROW_CUT };

        buffer_view<int64_t>     records = {nullptr, 0}; // [n_outputs][n_record]
        buffer_view<float>       logits  = {nullptr, 0}; // [n_outputs][k], the largest first
        buffer_view<llama_token> ids     = {nullptr, 0}; // [n_outputs][k]

        std::vector<uint32_t> count; // candidates of a cut row
        std::vector<uint8_t>  state; // of each output row

        uint32_t k        = 0;
        uint32_t n_record = 0; // records a row: k a device that holds a share of the vocab

        std::vector<uint64_t> keys; // scratch of a cut

        std::set<llama_seq_id> seqs; // sequences whose rows are read as candidates
    };

    candidates_info cand;

    // sequence embeddings output (map of [n_embd] vectors)
    // populated only when pooling_type != LLAMA_POOLING_TYPE_NONE
    std::map<llama_seq_id, std::vector<float>> embd_seq;

    // reuse the batch_allocr to avoid unnecessary memory allocations
    std::unique_ptr<llama_batch_allocr> balloc;

    uint32_t n_outputs = 0; // number of actually-used outputs in the current ubatch or last logical batch

    std::vector<int32_t> output_ids; // map batch token positions to ids of the logits and embd buffers

    struct swap_info {
        uint32_t i0;
        uint32_t i1;
    };

    std::vector<swap_info> output_swaps;

    ggml_backend_sched_ptr sched;

    bool sched_need_reserve = true;

    // the graph node budget (graph_max_nodes) the scheduler was last reserved for
    size_t sched_max_nodes = 0;

    ggml_backend_t backend_cpu = nullptr;
    std::vector<ggml_backend_ptr> backends;

    // training
    ggml_opt_context_t opt_ctx = nullptr;

    ggml_threadpool_t threadpool       = nullptr;
    ggml_threadpool_t threadpool_batch = nullptr;

    ggml_abort_callback abort_callback      = nullptr;
    void *              abort_callback_data = nullptr;

    std::vector<std::pair<ggml_backend_t, ggml_backend_set_n_threads_t>> set_n_threads_fns;

    // pointers and buffer types used for the compute buffer of each backend
    std::vector<ggml_backend_t>             backend_ptrs;
    std::vector<ggml_backend_buffer_type_t> backend_buft;
    std::vector<size_t>                     backend_buf_exp_size; // expected buffer sizes

    llm_graph_result_ptr gf_res_prev;
    llm_graph_result_ptr gf_res_reserve;

    // graphs of up to graph_slot_max_tokens rows kept beside gf_res_prev, each with a scheduler of its own: splitting a
    // graph rewrites its nodes' sources to the scheduler's input copies, so a graph stays computable only while the
    // scheduler that split it holds it. A speculative verify alternates shapes (an n-gram draft of 7 rows, then MTP's
    // 3), and with gf_res_prev alone every change rebuilt the graph: ~5 ms to build, split and allocate it, and ~5 ms
    // more to launch its kernels uncaptured while the backend's CUDA graph warms up again (27B hybrid on a 5080, whose
    // verify round takes ~15 ms)
    struct graph_slot {
        ggml_backend_sched_ptr sched;
        llm_graph_result_ptr   res;
        uint64_t               t_used = 0;
    };
    static constexpr uint32_t graph_slot_max_tokens = 32;
    // 3, or LLAMA_GRAPH_SLOTS: with LLAMA_GRAPH_ANY_SEQS a graph serves every set of as many sequences, and a slot for
    // each count of sequences keeps every decode a reuse (the meta backend then needs GGML_META_CAPTURE_GRAPHS of
    // at least the slots and two more, to keep their captured evaluations)
    const size_t graph_slots_max = [] {
        const char * s = getenv("LLAMA_GRAPH_SLOTS");
        return s != nullptr && atoi(s) >= 1 ? (size_t) atoi(s) : (size_t) 3;
    }();
    // LLAMA_GRAPH_ANY_SEQS=1, read when the context is made (llm_graph_params::any_seqs)
    const bool graph_any_seqs = ggml_env_switch("LLAMA_GRAPH_ANY_SEQS");
    // LLAMA_PROMPT_SHAPES=1, read when the context is made: a ubatch of one sequence and a power of two of rows above
    // graph_slot_max_tokens, up to llama_prompt_shape_max(), keeps its graph in a slot as well. These are the ubatches
    // the batch allocator makes of a prompt's rows with the switch (llama_batch_allocr::split_equal), so a prompt read
    // beside replies is replayed where every prompt of another length built a graph
    const bool     graph_prompt_shapes    = ggml_env_switch("LLAMA_PROMPT_SHAPES");
    const uint32_t graph_prompt_shape_max = llama_prompt_shape_max();
    uint32_t graph_slot_mtp_seq_tokens = 0;
    uint32_t graph_slot_max_rows = graph_slot_max_tokens;
    std::vector<size_t> graph_slot_buf_limit;
    // The context's compute buffers. Each retained graph held a buffer of its own on every device beside the main
    // scheduler's, as large as the largest graph its slot ever held: 404 MiB a card for a 69-row reply graph of
    // GLM-5.3 Flash on four RTX PRO 6000 (-sm tensor, Nov 11, 2026), so 32 slots could hold 12.6 GiB a card, and a
    // long-context pool left no room for more. A context computes one graph at a time and every tensor of a graph is
    // written again at each compute, so the main scheduler and the slots' place their graphs in one buffer a buffer
    // type (ggml_gallocr_set_shared), which the reserve sizes for the largest batch as it did the main scheduler's; a
    // graph that does not fit it gets a buffer of its own. Nothing waits for it: a graph's inputs are host tensors,
    // and what a scheduler writes to a device (its copies of them, the graph's kernels) is queued on the device behind
    // the graph computed before, its output reads and the draft index rows kept from it, as a prompt's passes, each
    // built in the main scheduler's buffer over the one before, already relied on. Not with pipeline parallelism.
    // LLAMA_GRAPH_BUFFER_SHARED_LEGACY=1: a buffer a scheduler
    ggml_gallocr_shared_ptr graph_shared;
    std::vector<graph_slot> graph_slots;
    // The device memory a context leaves free: LLAMA_GRAPH_DEVICE_RESERVE_MIB, default 256; 0: none. It retains one
    // more graph only while every device it computes on has this much free, and before it builds a graph on a device
    // with less, the retained ones used least recently give their buffers back (process_ubatch). The reserve is
    // what the devices make the graphs' executables from: a CUDA device that runs out while it makes one cannot be
    // recovered (ggml_backend_cuda_context::cuda_graph_has_room). A slot also gives its buffers back when an
    // allocation fails; n_graph_slots_no_memory counts both.
    const size_t graph_device_reserve = [] {
        const char * s = getenv("LLAMA_GRAPH_DEVICE_RESERVE_MIB");
        return (size_t) (s != nullptr ? std::max(atoll(s), 0LL) : 256) * 1024 * 1024;
    }();
    bool     graph_devices_low() const;
    uint64_t n_graph_slots_no_memory = 0;

    // the graphs that move recurrent states between cells (rs_relocate), by the number of rows each moves; each is made
    // at its first use and computed again at each later one, with its scheduler
    struct rs_reloc_graph {
        ggml_context_ptr       ctx;
        ggml_backend_sched_ptr sched;
        ggml_cgraph *          gf  = nullptr;
        ggml_tensor *          src = nullptr;
        ggml_tensor *          dst = nullptr;
    };
    std::map<size_t, rs_reloc_graph> rs_reloc;
    uint64_t graph_clock = 0; // process_ubatch calls, the slots' use times

    // the scheduler that allocated the graph process_ubatch last computed: its outputs' backends are looked up there
    ggml_backend_sched_t sched_res = nullptr;

    // one-time Hadamard transform-coverage check on the first built graph
    bool hadamard_verified = false;

    // host buffer for the model output (logits and embeddings)
    ggml_backend_buffer_ptr buf_output;

    // keep copies of the per-sequence memory on the device
    std::map<llama_seq_id, llama_memory_buffers> mem_storage;

    bool has_evaluated_once = false;

    // env: LLAMA_GRAPH_REUSE_DISABLE
    bool graph_reuse_disable = false;

    // perf
    mutable int64_t t_start_us  = 0;
    mutable int64_t t_load_us   = 0;
    mutable int64_t t_p_eval_us = 0;
    mutable int64_t t_eval_us   = 0;

    mutable int64_t t_compute_start_us = 0;
    mutable int64_t n_queued_tokens    = 0;

    // a decode, an encode or a graph of this context has run since it last synchronized (synchronize_outputs)
    bool outputs_pending = false;

    mutable int32_t n_p_eval = 0; // number of tokens in eval calls for the prompt (with batch size > 1)
    mutable int32_t n_eval   = 0; // number of eval calls

    mutable int32_t n_reused = 0; // number of times the previous graph was reused
};
