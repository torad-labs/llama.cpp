#pragma once

#include "llama.h"

#include <vector>

struct llama_vocab;
struct llama_grammar;

// sampler chain

struct llama_sampler_chain {
    llama_sampler_chain_params params;

    // has .backend_init() been called?
    bool is_init = false;

    uint32_t n_nodes = 0;

    struct info {
        bool is_backend;

        llama_sampler * ptr;
    };

    std::vector<info> samplers;

    // pre-allocated buffer for llama_sampler_sample to avoid repeated allocations
    std::vector<llama_token_data> cur;

    // timing

    mutable int64_t t_sample_us;

    mutable int32_t n_sample;
};

uint32_t llama_sampler_backend_n_nodes(const llama_sampler * sampler);
// whether a backend sampler of the chain writes graph inputs (.backend_set_input, e.g. the dist sampler's uniforms): it
// holds those of the graph it was applied to last, so no other graph with it can be computed without a rebuild
bool llama_sampler_backend_has_inputs(const llama_sampler * sampler);
void llama_sampler_backend_begin(llama_sampler * sampler);
// the context no longer runs the sampler: the chain applies every sampler on the CPU again and the dist sampler draws
// from its own RNG outside any backend transaction (a sequence goes back to the backend with a fresh sampler)
void llama_sampler_backend_release(llama_sampler * sampler);
// the sequence's tokens in the ubatch the sampler's graph is about to run, in batch order, with output[i] set for the
// output rows (tokens may be null: a batch of embeddings). The penalties sampler penalizes each output row for the
// accepted history followed by the sequence's tokens after its first output row of the decode, up to and including the
// row's own: the rows of an MTP verify batch see the drafts before them, as the CPU chain does when it accepts each
// draft before sampling the next row
void llama_sampler_backend_set_batch(
        llama_sampler * sampler, const llama_token * tokens, const int8_t * output, int32_t n_tokens);

struct llama_sampler * llama_sampler_init_dry_testing(
        float   dry_multiplier,
        float   dry_base,
        int32_t dry_allowed_length,
        int32_t dry_penalty_last_n,
        const std::vector<std::vector<llama_token>> & seq_breakers);
