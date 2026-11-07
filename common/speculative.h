#pragma once

#include "llama.h"
#include "common.h"

#include <array>
#include <functional>
#include <vector>

// Pure MTP policy and service-cycle observer. Timestamps are completed first-draft-step boundaries, in any common unit.
struct common_mtp_service {
    static constexpr int history = 5;
    std::array<std::array<double, history>, 2> samples = {};
    std::array<int, 2> n_samples = {};
    std::array<int, 2> next = {};
    std::array<uint64_t, 2> last = {};
    uint64_t serial = 0;
    llama_seq_id owner = -1;
    int chosen = 0, low = 0, full = 0, realized = 0, verified = 0, accepted = -1;
    double start = 0;

    void invalidate();
    void boundary(llama_seq_id seq, int depth, int low_depth, int full_depth, double now);
    void drafted(llama_seq_id seq, int depth);
    void verify(llama_seq_id seq, int rows);
    void accept(llama_seq_id seq, int count, bool other = false);
    float relative_cost(float fallback) const;
};

struct common_mtp_depth {
    static constexpr int32_t window = 8;
    static constexpr float keep = 15.0f / 16.0f;
    static constexpr float band = 0.125f;
    int32_t low = 1, depth = 0, rounds = 0;
    int32_t n_drafted = 0, n_verified = 0; // awaited draft length and target batch count (two means replay)
    // Decayed acceptance: first token, extra full-depth tokens, and extra tokens within low depth.
    float first_sum = 0, first_w = 0, extra_sum = 0, extra_w = 0, low_sum = 0;

    void reset(int32_t n_max, int32_t n_min);
    int32_t n_draft(int32_t n_max, bool timed = false) const;
    void observe(int32_t n_max, float cost, int32_t n_accepted, float service_cut = -1);
};

struct common_speculative;

// comma separated list the provided types
std::string common_speculative_type_name_str(const std::vector<enum common_speculative_type> & types);

// comma separated list of all types
const char * common_speculative_all_types_str();

// parse user provided types
std::vector<enum common_speculative_type> common_speculative_types_from_names(const std::vector<std::string> & names);

// infer the spec types from the GGUF metadata of a draft model; empty if unknown
std::vector<enum common_speculative_type> common_speculative_types_from_gguf(const std::string & path);

// convert string to type
enum common_speculative_type common_speculative_type_from_name(const std::string & name);

// convert type to string
std::string common_speculative_type_to_str(enum common_speculative_type type);

// return the max number of draft tokens based on the speculative parameters
int32_t common_speculative_n_max(const common_params_speculative * spec);

common_params common_base_params_to_speculative(const common_params & params);

struct common_speculative_output_limits {
    int32_t total;
    int32_t per_seq;
};

// return the output limits needed for speculative decoding
common_speculative_output_limits common_speculative_get_output_limits(
        int32_t n_batch, int32_t n_parallel, int32_t n_draft);

// the most tokens a sequence may draft this round: what its context has room for past the next token (n_room), what
// the batch gives each of the n_generating sequences (n_per_seq), and one less than the tokens its request may still
// emit (n_remaining, -1 when it has no limit) unless it verifies with other sequences in a target that keeps each
// sequence's rollback rows in one ubatch, where a shorter draft would split the verification into a pass per sequence
int32_t common_speculative_draft_limit(
        const llama_context * ctx_tgt, int32_t n_generating, int32_t n_room, int32_t n_remaining, int32_t n_per_seq);

struct common_sampler;

// samples a sequence's verification rows (idxs: the row of the token before its draft, then one row per draft token)
// and keeps the prefix of its draft the target agrees with, then the target's own next token, as
// common_sampler_sample_and_accept_n does, but no more than n_remaining tokens, what its request may still emit (-1 when
// it has no limit): draft rows past the output limit roll back as rejected ones
std::vector<llama_token> common_speculative_sample_and_accept(
        common_sampler * smpl, llama_context * ctx_tgt, const std::vector<int> & idxs, const llama_tokens & draft,
        int32_t n_remaining, const std::function<void(size_t, llama_token)> & on_sample = nullptr);

common_speculative * common_speculative_init(common_params_speculative & params, uint32_t n_seq);

void common_speculative_free(common_speculative * spec);

struct common_speculative_draft_params {
    // this flag is used to chain the drafts through all the available implementations
    // after the first successful draft from an implementation, we set it
    //   to false to prevent further drafts for that sequence
    // at the end of the draft() call, all drafting flags will be reset to false
    bool drafting = false;

    // overrides individual configurations (-1 disabled)
    // can be used to constraint the max draft based on the remaining context size
    int32_t n_max = -1;

    llama_pos   n_past;
    llama_token id_last;

    // TODO: remove in the future by keeping track of the prompt from the _begin() call and the consecutive accept calls
    const llama_tokens * prompt;

    // the generated draft from the last _draft() call
    llama_tokens * result;
};

common_speculative_draft_params & common_speculative_get_draft_params(common_speculative * spec, llama_seq_id seq_id);

// optionally call once at the beginning of a new generation
void common_speculative_begin(common_speculative * spec, llama_seq_id seq_id, const llama_tokens & prompt);

// the sequence stopped generating (slot released): a decode-only MTP head stops feeding it until the next begin
void common_speculative_end(common_speculative * spec, llama_seq_id seq_id);

// process the batch and update the internal state of the speculative context
bool common_speculative_process(common_speculative * spec, const llama_batch & batch);

// generate drafts for the sequences specified with `common_speculative_get_draft_params`
void common_speculative_draft(common_speculative * spec);

// informs the speculative context that n_accepted tokens were accepted by the target model
void common_speculative_accept(common_speculative * spec, llama_seq_id, uint16_t n_accepted);

// (optional) get/set internal state
bool common_speculative_get_state(common_speculative * spec, llama_seq_id seq_id, std::vector<uint8_t> & data);
void common_speculative_set_state(common_speculative * spec, llama_seq_id seq_id, const std::vector<uint8_t> & data);

// print statistics about the speculative decoding
void common_speculative_print_stats(const common_speculative * spec);

struct common_speculative_deleter {
    void operator()(common_speculative * s) { common_speculative_free(s); }
};

typedef std::unique_ptr<common_speculative, common_speculative_deleter> common_speculative_ptr;

struct common_speculative_init_result {
    common_speculative_init_result(common_params & params, llama_model * model_tgt, llama_context * ctx_tgt);
    ~common_speculative_init_result();

    llama_model   * model();
    llama_context * context();

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

using common_speculative_init_result_ptr = std::unique_ptr<common_speculative_init_result>;

common_speculative_init_result_ptr common_speculative_init_from_params(common_params & params, llama_model * model_tgt, llama_context * ctx_tgt);
