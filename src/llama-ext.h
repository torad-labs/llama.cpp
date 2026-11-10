#pragma once

// this is a staging header for new llama.cpp API
// breaking changes and C++ are allowed. everything here should be considered WIP
// try as much as possible to not include this header in the rest of the codebase

#include "llama.h"

#include <cstdint>
#include <map>

// Reserve a new compute graph. It is valid until the next call to llama_graph_reserve.
LLAMA_API struct ggml_cgraph * llama_graph_reserve(
        struct llama_context * ctx,
        uint32_t n_tokens,
        uint32_t n_seqs,
        uint32_t n_outputs);

// Restrict the MTP draft head to these token ids (strictly increasing): a draft step then reads their rows of
// the LM head only, and every other token's draft logit is -inf. Call before the MTP context is created.
// Returns false (and logs why) for a model whose MTP graph cannot score a trimmed head.
LLAMA_API bool llama_model_set_draft_vocab(struct llama_model * model, const llama_token * ids, size_t n_ids);

// GLM MTP index reuse: 0 starts a draft round, successive positive steps may reuse its
// pooled selection, and -1 ends it. Unsupported contexts return false. Prompt/catch-up
// decodes must happen outside a round. KV mutations invalidate every retained selection.
LLAMA_API bool llama_mtp_draft_step(struct llama_context * ctx, int32_t step);

// Retain bounded GLM MTP shapes in the existing three extra graph slots: the head's generation/catch-up shapes on its
// context, and the verifications of its drafts on the target's (n_max + 1 rows a sequence). Set during drafter setup.
// Zero restores the default 32-row limit; unsupported contexts return false.
LLAMA_API bool llama_set_mtp_graph_reuse(struct llama_context * ctx, int32_t n_max);

// True while a lazy grammar sampler waits for its trigger: its apply leaves every logit as it is until then.
// False for any other sampler, and for a grammar that constrains (triggered, or not lazy).
LLAMA_API bool llama_sampler_grammar_awaiting_trigger(const struct llama_sampler * smpl);

// Get the default ggml_type for a given ftype.
LLAMA_API ggml_type llama_ftype_get_default_type(llama_ftype ftype);

struct quantize_state_impl;

LLAMA_API quantize_state_impl * llama_quant_init(
        const llama_model * model,
        const llama_model_quantize_params * params);

LLAMA_API void llama_quant_free(quantize_state_impl * qs);

// Descriptor for constructing a mock model for quantization testing.
struct llama_quant_model_desc {
    const char * architecture;
    uint32_t n_embd;
    uint32_t n_ff;
    uint32_t n_layer;
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t n_expert;
    uint32_t n_embd_head_k;
    uint32_t n_embd_head_v;
};

// Create a mock model from a metadata descriptor (for testing).
// The returned model must be freed with llama_model_free().
LLAMA_API llama_model * llama_quant_model_from_metadata(const llama_quant_model_desc * desc);

// Returns true if this tensor should be quantized (based on name, dims, params).
LLAMA_API bool llama_quant_tensor_allows_quantization(
        const quantize_state_impl * qs,
        const ggml_tensor * tensor);

// Compute quantization type assignments for a list of tensors.
// All tensors should be quantizable (use llama_quant_tensor_allows_quantization to filter).
// result_types: caller-allocated array of n_tensors elements, filled with assigned types.
LLAMA_API void llama_quant_compute_types(
        quantize_state_impl * qs,
        llama_ftype ftype,
        ggml_tensor ** tensors,
        ggml_type * result_types,
        size_t n_tensors);

//
// device memory querying
//

// "memory" as in physical memory for a buffer type, in bytes
struct llama_memory_breakdown_data {
    size_t model   = 0; // memory allocated for the model
    size_t context = 0; // memory allocated for the context
    size_t compute = 0; // memory allocated for temporary compute buffers

    size_t total() const {
        return model + context + compute;
    }
};

struct llama_device_memory_data {
    int64_t total;
    int64_t free;
    llama_memory_breakdown_data mb;
};

// TODO: convert to C-style data structure
using llama_memory_breakdown = std::map<ggml_backend_buffer_type_t, llama_memory_breakdown_data>;

LLAMA_API int32_t llama_model_n_expert (const struct llama_model * model);
LLAMA_API int32_t llama_model_n_devices(const struct llama_model * model);

LLAMA_API ggml_backend_dev_t llama_model_get_device(const struct llama_model * model, int i);

LLAMA_API llama_memory_breakdown llama_get_memory_breakdown(const struct llama_context * ctx);

// Set whether the context outputs nextn embeddings or not
// If masked == true,  output the embeddings only for the tokens with batch.logits != 0
// If masked == false, output the embeddings for all tokens in the batch regardless of batch.logits
LLAMA_API void llama_set_embeddings_nextn(struct llama_context * ctx, bool value, bool masked);

// Select which appended NextN block the DECODER_MTP graph runs (offset past
// the trunk: il = n_layer() + offset). Used by the speculative NextN driver to
// chain multiple trained NextN heads. Default 0 (first head).
LLAMA_API void llama_set_nextn_layer_offset(struct llama_context * ctx, int32_t offset);

// mirrors:
// LLAMA_API float * llama_get_embeddings(struct llama_context * ctx);
LLAMA_API float * llama_get_embeddings_nextn(struct llama_context * ctx);

// LLAMA_API float * llama_get_embeddings_ith(struct llama_context * ctx, int32_t i);
LLAMA_API float * llama_get_embeddings_nextn_ith(struct llama_context * ctx, int32_t i);

// Set whether the context outputs the input embeddings of a specific layer
LLAMA_API void llama_set_embeddings_layer_inp(struct llama_context * ctx, uint32_t lid, bool value);

// mirrors:
// LLAMA_API float * llama_get_embeddings(struct llama_context * ctx);
LLAMA_API float * llama_get_embeddings_layer_inp(struct llama_context * ctx, uint32_t lid);

LLAMA_API llama_context * llama_get_ctx_other(struct llama_context * ctx);

// The MTP head reads the target's next-token hidden rows in device memory (GLM5NEXT; common turns it on with
// LLAMA_MTP_DEVICE_H=1). Binding gives the target a store of rows in the memory of its output device (on every card
// under -sm tensor): each target batch of at least LLAMA_MTP_DEVICE_H_MIN_ROWS rows (default 64) writes every row there
// by batch index, and the head gathers its rows there instead of from a host copy. Returns false, and both contexts
// keep the host path, when the pair cannot take it: another model, another arch, a masked target, pooling, or no room
// for the store itself. Once the store is in place, binding reserves both contexts' graphs again with it, and like a
// context's first reserve that throws std::runtime_error when the devices cannot hold them; neither context is usable
// after such a throw, so a caller short of memory fails as it would at context creation.
LLAMA_API bool    llama_set_nextn_source(struct llama_context * ctx_dft, struct llama_context * ctx_tgt, int32_t n_keep = 0);

// Optional sequence-owned rows, separate from the two reusable batch halves. A retained binding stores short batches too.
LLAMA_API bool    llama_nextn_keep_rows(struct llama_context * ctx_tgt, llama_seq_id seq, const int32_t * rows, int32_t n);
LLAMA_API int32_t llama_nextn_kept_row(const struct llama_context * ctx_tgt, llama_seq_id seq, int32_t row);
LLAMA_API bool    llama_nextn_kept_get(struct llama_context * ctx_tgt, int32_t ref, float * dst);
LLAMA_API bool    llama_nextn_kept_set(struct llama_context * ctx_tgt, int32_t ref, const float * src);

// the rows the target's last decode left in the store: its batch's row count, or 0 when it did not write the store
LLAMA_API int32_t llama_get_nextn_store_rows(const struct llama_context * ctx_tgt);

// for the head's next decode, of a batch without embd: per batch row, the target batch row whose hidden row it reads,
// or -(c + 1) for host carry row c, or an owned reference from llama_nextn_kept_row (seq -1 gives zeros).
LLAMA_API void    llama_set_nextn_h_rows(struct llama_context * ctx_dft, const int32_t * src, int32_t n,
                                         const float * carry, int32_t n_carry);

// copies the listed batch rows of the target's last write to dst, n rows of n_embd floats
LLAMA_API bool    llama_get_embeddings_nextn_rows(struct llama_context * ctx_tgt, const int32_t * rows, int32_t n, float * dst);

// the hidden rows the target has copied to host memory since it was created
LLAMA_API int64_t llama_get_nextn_rows_to_host(const struct llama_context * ctx_tgt);

//
// model/context data extraction
//

LLAMA_API int32_t llama_model_dflash_selector_top_k(const struct llama_model * model);

// returns pointer to the target-model layer indices
LLAMA_API const int32_t * llama_model_target_layer_ids  (const struct llama_model * model);
// returns the number of extracted layers from target model
LLAMA_API uint32_t        llama_model_target_layer_ids_n(const struct llama_model * model);
// returns true if the draft model carries a DSpark Markov head. Both lineages declare
// general.architecture = dflash, so this is how to tell them apart.
LLAMA_API bool            llama_model_has_dspark_markov_head(const struct llama_model * model);

// retrieves the whole token embedding matrix in F32 format (n_embd * n_vocab)
// returns total number of elements or 0 on error
// if out is nullptr, returns the number of tokens without writing to out
// caller must allocate enough memory for out before calling
LLAMA_API uint32_t llama_model_get_tok_embd(const struct llama_model * model, float * out);
