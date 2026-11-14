#include "sampling.h"

#include "common.h"
#include "fit.h"
#include "log.h"
#include "reasoning-budget.h"

#include "../src/llama-ext.h" // staging API: llama_sampler_grammar_awaiting_trigger

#include "ggml.h"

#include <algorithm>
#include <cctype>
#include <climits>
#include <cmath>
#include <cstring>
#include <functional>
#include <random>
#include <unordered_map>
#include <vector>

#if defined(__SSE2__)
#include <emmintrin.h>
#endif

// the ring buffer works similarly to std::deque, but with a fixed capacity
// TODO: deduplicate with llama-impl.h
template<typename T>
struct ring_buffer {
    ring_buffer(size_t cap) : capacity(cap), data(cap) {}

    T & front() {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        return data[first];
    }

    const T & front() const {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        return data[first];
    }

    T & back() {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        return data[pos];
    }

    const T & back() const {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        return data[pos];
    }

    void push_back(const T & value) {
        if (sz == capacity) {
            // advance the start when buffer is full
            first = (first + 1) % capacity;
        } else {
            sz++;
        }
        data[pos] = value;
        pos = (pos + 1) % capacity;
    }

    T pop_front() {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        T value = data[first];
        first = (first + 1) % capacity;
        sz--;
        return value;
    }

    const T & rat(size_t i) const {
        if (i >= sz) {
            throw std::runtime_error("ring buffer: index out of bounds");
        }
        return data[(first + sz - i - 1) % capacity];
    }

    std::vector<T> to_vector() const {
        std::vector<T> result;
        result.reserve(sz);
        for (size_t i = 0; i < sz; i++) {
            result.push_back(data[(first + i) % capacity]);
        }
        return result;
    }

    void clear() {
        // here only reset the status of the buffer
        sz = 0;
        first = 0;
        pos = 0;
    }

    bool empty() const {
        return sz == 0;
    }

    size_t size() const {
        return sz;
    }

    size_t capacity = 0;
    size_t sz = 0;
    size_t first = 0;
    size_t pos = 0;
    std::vector<T> data;
};

// LLAMA_TOP_K_PREFILTER_LEGACY=1: a chain drawing on the CPU reads every logit of a row, with no top-k taken on the
// backend (the behaviour before common_sampler_backend_top_k), and a backend's candidates stay in the order it gave
static bool top_k_prefilter_legacy() {
    static const bool legacy = ggml_env_switch("LLAMA_TOP_K_PREFILTER_LEGACY");
    return legacy;
}

// LLAMA_SAMPLING_TIES_LEGACY=1, read as src/llama-sampler.cpp reads it: candidates of equal logit are not ordered by id,
// so the heap below may keep any of those tied at its k-th (the behaviour before ties were broken by id)
static bool sampling_ties_legacy() {
    static const bool legacy = ggml_env_switch("LLAMA_SAMPLING_TIES_LEGACY");
    return legacy;
}

// LLAMA_TOP_K_HEAP_LEGACY=1: a row's k largest logits are kept in a heap, a pop and a push for each logit above its
// worst (the behaviour before the buffer below). LLAMA_SAMPLING_TIES_LEGACY=1 keeps the heap too: its order among
// equal logits is the heap's own
static bool top_k_heap_legacy() {
    static const bool legacy = ggml_env_switch("LLAMA_TOP_K_HEAP_LEGACY");
    return legacy;
}

// the largest of n logits, n a multiple of 16
static float top_k_max_logit(const float * x, int32_t n) {
#if defined(__SSE2__)
    __m128 m0 = _mm_loadu_ps(x);
    __m128 m1 = _mm_loadu_ps(x + 4);
    __m128 m2 = _mm_loadu_ps(x + 8);
    __m128 m3 = _mm_loadu_ps(x + 12);
    for (int32_t i = 16; i < n; i += 16) {
        m0 = _mm_max_ps(m0, _mm_loadu_ps(x + i));
        m1 = _mm_max_ps(m1, _mm_loadu_ps(x + i + 4));
        m2 = _mm_max_ps(m2, _mm_loadu_ps(x + i + 8));
        m3 = _mm_max_ps(m3, _mm_loadu_ps(x + i + 12));
    }
    float m[4];
    _mm_storeu_ps(m, _mm_max_ps(_mm_max_ps(m0, m1), _mm_max_ps(m2, m3)));
    return std::max(std::max(m[0], m[1]), std::max(m[2], m[3]));
#else
    float m = x[0];
    for (int32_t i = 1; i < n; ++i) {
        m = std::max(m, x[i]);
    }
    return m;
#endif
}

// The k largest by logit and then id, sorted. The logits above a bar wait in a buffer of 2k; a full buffer keeps its k
// largest and raises the bar to the last of them, so a logit is then passed over only when k earlier ones, their ids
// smaller, are at least as large. The bar starts at the k-th largest of the row's block maxima, the row split in
// blocks n/2k wide: k blocks hold a logit at or above it, so no logit below it is among the k, and about 0.7 of the
// buffer is at or above it when the large logits are spread over the ids. The k kept are the ones a sort of every
// token keeps
static void top_k_candidates_buffered(const float * logits, int32_t n_vocab, int32_t k, std::vector<llama_token_data> & out) {
    const auto before = [](const llama_token_data & a, const llama_token_data & b) {
        return a.logit > b.logit || (a.logit == b.logit && a.id < b.id);
    };

    constexpr int32_t block = 16;

    const size_t keep = (size_t) k;
    const size_t cap  = 2*keep;

    out.resize(cap);

    size_t n = 0;
    float kth = 0.0f;

    const auto cut = [&]() {
        std::nth_element(out.begin(), out.begin() + (keep - 1), out.begin() + n, before);
        n   = keep;
        kth = out[keep - 1].logit;
    };

    const auto consider = [&](int32_t i) {
        if (logits[i] > kth) {
            out[n++] = llama_token_data{i, logits[i], 0.0f};
            if (n == cap) {
                cut();
            }
        }
    };

    // the logits from i on
    const auto scan = [&](int32_t i) {
        for (; i + block <= n_vocab; i += block) {
#if defined(__SSE2__)
            // the compiler leaves the block's test as sixteen scalar compares: four vector ones, and only the logits
            // above the bar are looked at again
            const __m128 bar = _mm_set1_ps(kth);
            const __m128 a   = _mm_cmpgt_ps(_mm_loadu_ps(logits + i),      bar);
            const __m128 b   = _mm_cmpgt_ps(_mm_loadu_ps(logits + i + 4),  bar);
            const __m128 c   = _mm_cmpgt_ps(_mm_loadu_ps(logits + i + 8),  bar);
            const __m128 d   = _mm_cmpgt_ps(_mm_loadu_ps(logits + i + 12), bar);
            if (_mm_movemask_ps(_mm_or_ps(_mm_or_ps(a, b), _mm_or_ps(c, d)))) {
                uint32_t above = (uint32_t) _mm_movemask_ps(a) | (uint32_t) _mm_movemask_ps(b) << 4 |
                    (uint32_t) _mm_movemask_ps(c) << 8 | (uint32_t) _mm_movemask_ps(d) << 12;
                for (; above; above &= above - 1) {
                    consider(i + __builtin_ctz(above));
                }
            }
#else
            bool any = false;
            for (int32_t j = 0; j < block; ++j) {
                any |= logits[i + j] > kth;
            }
            if (any) {
                for (int32_t j = 0; j < block; ++j) {
                    consider(i + j);
                }
            }
#endif
        }
        for (; i < n_vocab; ++i) {
            consider(i);
        }
    };

    const auto finish = [&]() {
        if (n > keep) {
            cut();
        }
        out.resize(n);
        std::sort(out.begin(), out.end(), before);
    };

    const int32_t width = (int32_t) ((size_t) n_vocab/cap)/block*block;

    if (width > 0) {
        std::vector<float> tops((size_t) (n_vocab/width));
        for (size_t b = 0; b < tops.size(); ++b) {
            tops[b] = top_k_max_logit(logits + b*width, width);
        }
        std::nth_element(tops.begin(), tops.begin() + (keep - 1), tops.end(), std::greater<float>());

        const float bar = tops[keep - 1];
        if (bar > -INFINITY) {
            // a logit at the bar passes
            kth = std::nextafter(bar, -INFINITY);
            scan(0);
            // fewer than k only when a logit is not a number: the row is read again from its first logits
            if (n >= keep) {
                finish();
                return;
            }
        }
    }

    // a row too short for blocks: the buffer starts as its first logits, whatever they are
    const int32_t n_first = (int32_t) std::min<size_t>(cap, (size_t) n_vocab);
    for (int32_t i = 0; i < n_first; ++i) {
        out[i] = llama_token_data{i, logits[i], 0.0f};
    }
    n = (size_t) n_first;
    if (n > keep) {
        cut();
        scan(n_first);
    }
    finish();
}

void common_sampler_top_k_candidates(const float * logits, int32_t n_vocab, int32_t k, std::vector<llama_token_data> & out) {
    if (!top_k_heap_legacy() && !sampling_ties_legacy()) {
        top_k_candidates_buffered(logits, n_vocab, k, out);
        return;
    }

    // a heap whose front is the worst kept, by logit and then id as llama's candidate sorts order them: a later token,
    // its id larger than every kept one's, replaces the worst only when its logit is strictly larger
    const auto heap_cmp = [](const llama_token_data & a, const llama_token_data & b) {
        return a.logit > b.logit || (a.logit == b.logit && a.id < b.id && !sampling_ties_legacy());
    };

    out.resize(k);
    for (int32_t i = 0; i < k; ++i) {
        out[i] = llama_token_data{i, logits[i], 0.0f};
    }
    std::make_heap(out.begin(), out.end(), heap_cmp);

    const auto consider = [&](int32_t i) {
        if (logits[i] > out.front().logit) {
            std::pop_heap(out.begin(), out.end(), heap_cmp);
            out.back() = llama_token_data{i, logits[i], 0.0f};
            std::push_heap(out.begin(), out.end(), heap_cmp);
        }
    };

    constexpr int32_t block = 16;

    int32_t i = k;
    for (; i + block <= n_vocab; i += block) {
        const float kth = out.front().logit;
        bool any = false;
        for (int32_t j = 0; j < block; ++j) {
            any |= logits[i + j] > kth;
        }
        if (any) {
            for (int32_t j = 0; j < block; ++j) {
                consider(i + j);
            }
        }
    }
    for (; i < n_vocab; ++i) {
        consider(i);
    }

    std::sort(out.begin(), out.end(), [](const llama_token_data & a, const llama_token_data & b) {
        return a.logit > b.logit || (a.logit == b.logit && a.id < b.id);
    });
}

struct common_sampler {
    common_params_sampling params;

    struct llama_sampler * grmr;
    struct llama_sampler * rbudget;
    struct llama_sampler * chain;

    ring_buffer<llama_token> prev;

    std::vector<llama_token_data> cur;

    llama_token_data_array cur_p;

    // for rejection sampling; independent of the draft, or the target distribution is not preserved
    std::mt19937 rng;

    // the chain's top-k when no sampler before it changes a logit for these params, else 0: the k a backend may take for
    // it (common_sampler_backend_top_k)
    int32_t top_k_first = 0;

    // how many of a row's largest logits the chain reads on the CPU, 0 for every token's: its top-k, and behind penalties
    // that only lower logits, their window too (sampler_top_k_first)
    int32_t n_candidates = 0;

    // common_sampler_sample_coupled's copy of the chain, the tokens a drafter proposed that the copy has accepted since
    // it was made, whether it has drawn since the last of them, and the drafter's candidates. An accept, a reset or a
    // copy into this sampler voids the copy
    struct llama_sampler *        chain_ahead = nullptr;
    bool                          ahead_valid = false;
    bool                          ahead_drawn = false;
    llama_tokens                  ahead       = {};
    std::vector<llama_token_data> cur_ahead   = {};

    void reset() {
        prev.clear();
        ahead_valid = false;

        llama_sampler_reset(chain);
    }

    // top_k > 0: the candidates are the top_k largest logits, sorted, instead of every token's
    void set_logits(struct llama_context * ctx, const llama_sampling_row & row, int32_t top_k = 0) {
        const float *       sampled_probs  = row.sampled_probs;
        const float *       sampled_logits = row.sampled_logits;
        const llama_token * sampled_ids    = row.candidates;

        const llama_model * model = llama_get_model(ctx);
        const llama_vocab * vocab = llama_model_get_vocab(model);

        const int n_vocab = llama_vocab_n_tokens(vocab);

        if (sampled_probs) {
            const uint32_t sampled_probs_count = row.n_sampled_probs;
            cur.resize(sampled_probs_count);
            for (uint32_t i = 0; i < sampled_probs_count; ++i) {
                cur[i] = llama_token_data{sampled_ids[i], sampled_logits[i], sampled_probs[i]};
            }
        } else if (sampled_logits) {
            const uint32_t sampled_logits_count = row.n_sampled_logits;
            cur.resize(sampled_logits_count);
            for (uint32_t i = 0; i < sampled_logits_count; i++) {
                cur[i] = llama_token_data{sampled_ids[i], sampled_logits[i], 0.0f};
            }
            // a backend's top-k gives its candidates in no fixed order among equal logits: sorted by logit, ties by id,
            // as common_sampler_top_k_candidates sorts them, the chain draws from them as from the k the CPU takes (a
            // row the backend did not cut, its device lacking an op, stays as every token's)
            if (!top_k_prefilter_legacy() && sampled_logits_count < (uint32_t) n_vocab) {
                const auto before = [](const llama_token_data & a, const llama_token_data & b) {
                    return a.logit > b.logit || (a.logit == b.logit && a.id < b.id);
                };
                // a context's candidates come sorted (llama_set_output_candidates)
                if (!std::is_sorted(cur.begin(), cur.end(), before)) {
                    std::sort(cur.begin(), cur.end(), before);
                }
                cur_p = { cur.data(), cur.size(), -1, true };
                return;
            }
        } else {
            const auto * logits = row.logits;
            GGML_ASSERT(logits != nullptr);
            // a token the backend sampled is looked up among the candidates, so they stay every token's
            if (top_k > 0 && top_k < n_vocab && row.token == LLAMA_TOKEN_NULL) {
                common_sampler_top_k_candidates(logits, n_vocab, top_k, cur);
                cur_p = { cur.data(), cur.size(), -1, true };
                return;
            }
            cur.resize(n_vocab);
            for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
                cur[token_id] = llama_token_data{token_id, logits[token_id], 0.0f};
            }
        }

        cur_p = { cur.data(), cur.size(), -1, false };
    }

    common_time_meas tm() {
        return common_time_meas(t_total_us, params.no_perf);
    }

    mutable int64_t t_total_us = 0;
};

std::string common_params_sampling::print() const {
    char result[1024];

    snprintf(result, sizeof(result),
            "\trepeat_last_n = %d, repeat_penalty = %.3f, frequency_penalty = %.3f, presence_penalty = %.3f\n"
            "\tdry_multiplier = %.3f, dry_base = %.3f, dry_allowed_length = %d, dry_penalty_last_n = %d\n"
            "\ttop_k = %d, top_p = %.3f, min_p = %.3f, xtc_probability = %.3f, xtc_threshold = %.3f, typical_p = %.3f, top_n_sigma = %.3f, temp = %.3f\n"
            "\tmirostat = %d, mirostat_lr = %.3f, mirostat_ent = %.3f, adaptive_target = %.3f, adaptive_decay = %.3f",
            penalty_last_n, penalty_repeat, penalty_freq, penalty_present,
            dry_multiplier, dry_base, dry_allowed_length, dry_penalty_last_n,
            top_k, top_p, min_p, xtc_probability, xtc_threshold, typ_p, top_n_sigma, temp,
            mirostat, mirostat_eta, mirostat_tau, adaptive_target, adaptive_decay);

    return std::string(result);
}

// LLAMA_SAMPLING_BACKEND_PASSIVE_LEGACY=1: backend sampling is off for any request with a grammar or a reasoning
// budget, passive or not (the behaviour before common_sampler_backend_passive)
static bool backend_passive_legacy() {
    static const bool legacy = ggml_env_switch("LLAMA_SAMPLING_BACKEND_PASSIVE_LEGACY");
    return legacy;
}
// opt-in: every sampler's dist draws by keyed noise, and the MTP drafter proposes with the target's chain
static bool sampler_coupled() {
    static const bool coupled = [] {
        const bool on = ggml_env_switch("LLAMA_SPEC_COUPLED");
        if (on) {
            LOG_INF("sampling: LLAMA_SPEC_COUPLED: the dist sampler draws by keyed noise, which an MTP drafter proposes with\n");
        }
        return on;
    }();
    return coupled;
}

// LLAMA_SAMPLING_TOP_K_FIRST_LEGACY=1: the chain always starts from every token's logit (the behaviour before
// sampler_top_k_first)
static bool top_k_first_legacy() {
    static const bool legacy = ggml_env_switch("LLAMA_SAMPLING_TOP_K_FIRST_LEGACY");
    return legacy;
}

// LLAMA_SAMPLING_TOP_K_PENALTIES_LEGACY=1: penalties before the chain's top-k have it read every token's logit (the
// behaviour before the candidates took in the penalty window)
static bool top_k_penalties_legacy() {
    static const bool legacy = ggml_env_switch("LLAMA_SAMPLING_TOP_K_PENALTIES_LEGACY");
    return legacy;
}

// the most candidates a chain behind penalties is cut to; a longer window (repeat_last_n -1 is the context) reads every
// token's logit
static constexpr int32_t top_k_first_max_candidates = 1024;

// the chain's top-k when it is the first sampler of the chain to change a logit or drop a token (the penalties, DRY and
// top-n-sigma before it off for these params), else 0; up to 128, the k llama_sampler_top_k partial-sorts for.
// Penalties before it that only lower logits (repeat at least 1, frequency and presence at least 0) leave it first in
// effect, and window is then their last_n: they change the logits of the at most last_n tokens in their window and no
// other, so a token outside the k + last_n largest logits (llama's order: by logit, ties by id) has k unpenalized
// tokens above it still, and the k the top-k keeps are among those k + last_n
static int32_t sampler_top_k_first(const common_params_sampling & params, bool has_logit_bias, int32_t & window) {
    window = 0;
    if (top_k_first_legacy() || has_logit_bias || params.mirostat != 0 || params.top_k <= 0 || params.top_k > 128) {
        return 0;
    }
    for (const auto & cnstr : params.samplers) {
        switch (cnstr) {
            case COMMON_SAMPLER_TYPE_TOP_K:
                return params.top_k;
            case COMMON_SAMPLER_TYPE_PENALTIES:
                if (params.penalty_last_n != 0 &&
                        (params.penalty_repeat != 1.0f || params.penalty_freq != 0.0f || params.penalty_present != 0.0f)) {
                    const bool lowers_only = params.penalty_repeat >= 1.0f && params.penalty_freq >= 0.0f &&
                        params.penalty_present >= 0.0f;
                    if (top_k_penalties_legacy() || sampling_ties_legacy() || !lowers_only || params.penalty_last_n < 0 ||
                            params.penalty_last_n > top_k_first_max_candidates - params.top_k) {
                        return 0;
                    }
                    window = params.penalty_last_n;
                }
                break;
            case COMMON_SAMPLER_TYPE_DRY:
                if (params.dry_multiplier != 0.0f && params.dry_base >= 1.0f && params.dry_penalty_last_n != 0) {
                    return 0;
                }
                break;
            case COMMON_SAMPLER_TYPE_TOP_N_SIGMA:
                if (params.top_n_sigma > 0.0f) {
                    return 0;
                }
                break;
            default:
                return 0;
        }
    }
    return 0;
}

struct common_sampler * common_sampler_init(
        const struct llama_model * model,
        struct common_params_sampling & params) {
    if (!std::isfinite(params.penalty_repeat) ||
        params.penalty_repeat <= 0.0f ||
        !std::isfinite(1.0f/params.penalty_repeat)) {
        throw std::invalid_argument("penalty_repeat must be finite and greater than 0");
    }
    if (!std::isfinite(params.penalty_freq)) {
        throw std::invalid_argument("penalty_freq must be finite");
    }
    if (!std::isfinite(params.penalty_present)) {
        throw std::invalid_argument("penalty_present must be finite");
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    llama_sampler_chain_params lparams = llama_sampler_chain_default_params();

    lparams.no_perf = params.no_perf;

    llama_sampler * grmr = nullptr;
    llama_sampler * rbudget = nullptr;
    llama_sampler * chain = llama_sampler_chain_init(lparams);

    std::vector<llama_sampler *> samplers;

    const std::string & grammar_str = common_grammar_value(params.grammar);
    if (grammar_str.compare(0, 11, "%llguidance") == 0) {
#ifdef LLAMA_USE_LLGUIDANCE
        grmr = llama_sampler_init_llg(vocab, "lark", grammar_str.c_str());
#else
        throw std::runtime_error("failed to parse grammar: llguidance is not enabled");
#endif // LLAMA_USE_LLGUIDANCE
    } else {
        std::vector<std::string> trigger_patterns;
        std::vector<llama_token> trigger_tokens;
        for (const auto & trigger : params.grammar_triggers) {
            switch (trigger.type) {
                case COMMON_GRAMMAR_TRIGGER_TYPE_WORD:
                {
                    const auto & word = trigger.value;
                    trigger_patterns.push_back(regex_escape(word));
                    break;
                }
                case COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN:
                {
                    trigger_patterns.push_back(trigger.value);
                    break;
                }
                case COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL:
                {
                    const auto & pattern = trigger.value;
                    std::string anchored = "^$";
                    if (!pattern.empty()) {
                        anchored = (pattern.front() != '^' ? "^" : "")
                            + pattern
                            + (pattern.back() != '$' ? "$" : "");
                    }
                    trigger_patterns.push_back(anchored);
                    break;
                }
                case COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN:
                {
                    const auto token = trigger.token;
                    trigger_tokens.push_back(token);
                    break;
                }
                default:
                    GGML_ASSERT(false && "unknown trigger type");
            }
        }

        std::vector<const char *> trigger_patterns_c;
        trigger_patterns_c.reserve(trigger_patterns.size());
        for (const auto & regex : trigger_patterns) {
            trigger_patterns_c.push_back(regex.c_str());
        }

        if (!grammar_str.empty()) {
             if (params.grammar_lazy) {
                 grmr = llama_sampler_init_grammar_lazy_patterns(vocab, grammar_str.c_str(), "root",
                         trigger_patterns_c.data(), trigger_patterns_c.size(),
                         trigger_tokens.data(), trigger_tokens.size());
             } else {
                 grmr = llama_sampler_init_grammar(vocab, grammar_str.c_str(), "root");
             }
        }
    }
    if (!grmr && !grammar_str.empty()) {
        throw std::runtime_error("failed to parse grammar");
    }

    // Compute prefill tokens from the generation prompt
    std::vector<llama_token> prefill_tokens;
    if (!params.generation_prompt.empty()) {
        GGML_ASSERT(vocab != nullptr);
        auto tokens = common_tokenize(vocab, params.generation_prompt, false, true);
        for (size_t i = 0; i < tokens.size(); i++) {
            std::string piece = common_token_to_piece(vocab, tokens[i], true);
            if (i == 0 && std::isspace(piece[0]) && !std::isspace(params.generation_prompt[0])) {
                // Some tokenizers will add a space before the first special token, need to exclude
                continue;
            }
            LOG_DBG("%s: prefill token: %d = %s\n", __func__, tokens[i], piece.c_str());
            prefill_tokens.push_back(tokens[i]);
        }
    }

    // Feed generation prompt tokens to the grammar sampler so it advances past
    // tokens the template already placed in the prompt.
    // Only applies to output-format and tool-call grammars; user-supplied grammars must not be prefilled.
    if (grmr && !params.grammar_lazy && common_grammar_needs_prefill(params.grammar)) {
        try {
            for (const auto & token : prefill_tokens) {
                llama_sampler_accept(grmr, token);
                LOG_DBG("%s: grammar accepted prefill token (%d)\n", __func__, token);
            }
        } catch (std::exception &e) {
            LOG_ERR("%s: error initializing grammar sampler for grammar:\n%s\n\nGeneration prompt:\n'%s'\n", __func__,
                common_grammar_value(params.grammar).c_str(), params.generation_prompt.c_str());
            throw e;
        }
    }

    // reasoning budget sampler (skip when budget is unlimited unless a lazy grammar is active, which needs rbudget for thinking-block suppression)
    if (!params.reasoning_budget_start.empty() && !params.reasoning_budget_end.empty() && (params.grammar_lazy || params.reasoning_budget_tokens >= 0 || params.reasoning_control)) {
        rbudget = common_reasoning_budget_init(
            vocab,
            {params.reasoning_budget_start},
            params.reasoning_budget_end,
            params.reasoning_budget_forced,
            params.reasoning_budget_tokens < 0 ? INT_MAX : params.reasoning_budget_tokens);

        for (const auto & token : prefill_tokens) {
            llama_sampler_accept(rbudget, token);
            LOG_DBG("%s: reasoning-budget accepted prefill token (%d)\n", __func__, token);
        }
    }

    // logit bias: user biases + model suppress tokens (-INFINITY)
    bool has_logit_bias = false;
    {
        std::vector<llama_logit_bias> merged = params.logit_bias;

        int32_t n_suppress = 0;
        const llama_token * suppress = llama_vocab_get_suppress_tokens(vocab, &n_suppress);
        for (int32_t i = 0; i < n_suppress; ++i) {
            merged.push_back({ suppress[i], -INFINITY });
        }

        if (!merged.empty()) {
            samplers.push_back(llama_sampler_init_logit_bias(llama_vocab_n_tokens(vocab), merged.size(), merged.data()));
            has_logit_bias = true;
        }
    }

    if (params.mirostat == 0) {

        bool use_adaptive_p = false; // see below

        for (const auto & cnstr : params.samplers) {
            switch (cnstr) {
                case COMMON_SAMPLER_TYPE_DRY:
                    {
                        std::vector<const char *> c_breakers;
                        c_breakers.reserve(params.dry_sequence_breakers.size());
                        for (const auto & str : params.dry_sequence_breakers) {
                            c_breakers.push_back(str.c_str());
                        }
                        samplers.push_back(llama_sampler_init_dry(vocab, params.dry_multiplier, params.dry_base, params.dry_allowed_length, params.dry_penalty_last_n, c_breakers.data(), c_breakers.size()));
                    }
                    break;
                case COMMON_SAMPLER_TYPE_TOP_K:
                    samplers.push_back(llama_sampler_init_top_k(params.top_k));
                    break;
                case COMMON_SAMPLER_TYPE_TOP_P:
                    samplers.push_back(llama_sampler_init_top_p(params.top_p, params.min_keep));
                    break;
                case COMMON_SAMPLER_TYPE_TOP_N_SIGMA:
                    samplers.push_back(llama_sampler_init_top_n_sigma(params.top_n_sigma));
                    break;
                case COMMON_SAMPLER_TYPE_MIN_P:
                    samplers.push_back(llama_sampler_init_min_p(params.min_p, params.min_keep));
                    break;
                case COMMON_SAMPLER_TYPE_XTC:
                    samplers.push_back(llama_sampler_init_xtc(params.xtc_probability, params.xtc_threshold, params.min_keep, params.seed));
                    break;
                case COMMON_SAMPLER_TYPE_TYPICAL_P:
                    samplers.push_back(llama_sampler_init_typical(params.typ_p, params.min_keep));
                    break;
                case COMMON_SAMPLER_TYPE_TEMPERATURE:
                    samplers.push_back(llama_sampler_init_temp_ext(params.temp, params.dynatemp_range, params.dynatemp_exponent));
                    break;
                case COMMON_SAMPLER_TYPE_INFILL:
                    samplers.push_back(llama_sampler_init_infill(vocab));
                    break;
                case COMMON_SAMPLER_TYPE_PENALTIES:
                    samplers.push_back(llama_sampler_init_penalties(llama_vocab_n_tokens(vocab), params.penalty_last_n, params.penalty_repeat, params.penalty_freq, params.penalty_present));
                    break;
                case COMMON_SAMPLER_TYPE_ADAPTIVE_P:
                    // the `adaptive-p` sampler is like `dist` and `mirostat` in that it selects
                    // a single token, so we will add `dist` at the end of the chain by default,
                    // unless the user specifically included `adaptive-p`. we set this flag here
                    // so we know to add the sampler at the very end.
                    use_adaptive_p = true;
                    break;
                default:
                    GGML_ASSERT(false && "unknown sampler type");
            }
        }
        if (use_adaptive_p) {
            // only if user explicitly included adaptive-p sampler
            samplers.push_back(llama_sampler_init_adaptive_p(params.adaptive_target, params.adaptive_decay, params.seed));
        } else {
            // Keep distribution sampling when callers request probabilities.
            const bool greedy = params.n_probs == 0 && !params.samplers.empty() &&
                ((params.samplers.back() == COMMON_SAMPLER_TYPE_TEMPERATURE && params.temp == 0.0f && params.dynatemp_range == 0.0f) ||
                 (params.samplers.back() == COMMON_SAMPLER_TYPE_TOP_K && params.top_k == 1));
            samplers.push_back(greedy ? llama_sampler_init_greedy() :
                               params.coupled || sampler_coupled() ? llama_sampler_init_dist_coupled(params.seed)
                                                                   : llama_sampler_init_dist(params.seed));
        }
    } else if (params.mirostat == 1) {
        samplers.push_back(llama_sampler_init_temp(params.temp));
        samplers.push_back(llama_sampler_init_mirostat(llama_vocab_n_tokens(vocab), params.seed, params.mirostat_tau, params.mirostat_eta, 100));
    } else if (params.mirostat == 2) {
        samplers.push_back(llama_sampler_init_temp(params.temp));
        samplers.push_back(llama_sampler_init_mirostat_v2(params.seed, params.mirostat_tau, params.mirostat_eta));
    } else {
        GGML_ASSERT(false && "unknown mirostat version");
    }

    for (auto * smpl : samplers) {
        llama_sampler_chain_add(chain, smpl);
    }

    auto * result = new common_sampler {
        /* .params  = */ params,
        /* .grmr    = */ grmr,
        /* .rbudget = */ rbudget,
        /* .chain   = */ chain,
        /* .prev    = */ ring_buffer<llama_token>(std::max(32, params.n_prev)),
        /* .cur     = */ {},
        /* .cur_p   = */ {},
        // mix it, the chain and the draft are seeded from this one too
        /* .rng     = */ std::mt19937(llama_sampler_get_seed(chain) ^ 0x9e3779b9u),
    };

    int32_t window = 0;
    const int32_t top_k = sampler_top_k_first(params, has_logit_bias, window);
    // a backend's top-k stands in for the chain's own, which a penalty window before it would change: behind one, only
    // the CPU cuts a row, to the top-k and the window
    result->top_k_first  = window == 0 ? top_k : 0;
    result->n_candidates = top_k > 0 ? top_k + window : 0;

    // a lazy grammar waiting for its trigger and a reasoning budget that is not forcing leave every logit as it is,
    // so while both stay so a backend-sampled token is the one the CPU chain would draw (common_sampler_backend_passive);
    // the server takes the slot off the backend when either turns active. A sampler that constrains from the first
    // token (a grammar that is not lazy, llguidance, a budget that starts forcing) keeps the CPU path throughout.
    if (params.backend_sampling && (backend_passive_legacy() ? (grmr || rbudget) : !common_sampler_backend_passive(result))) {
        // a forcing budget holds the grammar off, so it is the budget that constrains even beside a grammar
        const bool forcing = rbudget && common_reasoning_budget_get_state(rbudget) == REASONING_BUDGET_FORCING;
        LOG_WRN("%s: backend sampling is not compatible with %s, disabling\n", __func__, grmr && !forcing ? "this grammar" : "this reasoning budget");

        params.backend_sampling         = false;
        result->params.backend_sampling = false;
    }

    return result;
}

void common_sampler_free(struct common_sampler * gsmpl) {
    if (!gsmpl) {
        return;
    }

    llama_sampler_free(gsmpl->grmr);
    llama_sampler_free(gsmpl->rbudget);
    llama_sampler_free(gsmpl->chain);
    llama_sampler_free(gsmpl->chain_ahead);

    delete gsmpl;
}

static bool grammar_should_apply(const struct common_sampler * gsmpl) {
    if (!gsmpl->grmr) {
        return false;
    }
    if (!gsmpl->rbudget) {
        return true;
    }
    if (gsmpl->params.grammar_lazy) {
        // if grammar is lazy, only apply when reasoning budget is not active
        const auto state = common_reasoning_budget_get_state(gsmpl->rbudget);
        return state == REASONING_BUDGET_IDLE || state == REASONING_BUDGET_DONE;
    }
    return true;
}

bool common_sampler_backend_passive(const struct common_sampler * gsmpl) {
    if (gsmpl->rbudget && common_reasoning_budget_get_state(gsmpl->rbudget) == REASONING_BUDGET_FORCING) {
        return false; // the only state in which the budget rewrites logits
    }
    if (!gsmpl->grmr || !grammar_should_apply(gsmpl)) {
        return true;  // no grammar, or a lazy one held off while reasoning
    }
    return llama_sampler_grammar_awaiting_trigger(gsmpl->grmr);
}

int32_t common_sampler_backend_top_k(const struct common_sampler * gsmpl) {
    return gsmpl && !top_k_prefilter_legacy() ? gsmpl->top_k_first : 0;
}

int32_t common_sampler_n_candidates(const struct common_sampler * gsmpl) {
    return gsmpl && !top_k_prefilter_legacy() ? gsmpl->n_candidates : 0;
}

bool common_output_candidates_legacy() {
    static const bool legacy = ggml_env_switch("LLAMA_OUTPUT_CANDIDATES_LEGACY");
    return legacy;
}

void common_sampler_accept(struct common_sampler * gsmpl, llama_token token, bool is_generated) {
    if (!gsmpl) {
        return;
    }

    const auto tm = gsmpl->tm();

    // grammar_should_apply() checks the reasoning budget state, so calculate this before we accept
    const auto accept_grammar = is_generated && grammar_should_apply(gsmpl);

    if (gsmpl->rbudget && is_generated) {
        llama_sampler_accept(gsmpl->rbudget, token);

        // if done, replay end sequence which may contain a grammar trigger
        const bool is_done = common_reasoning_budget_get_state(gsmpl->rbudget) == REASONING_BUDGET_DONE;
        if (gsmpl->grmr && !accept_grammar && is_done) {
            const llama_tokens * end_seq = common_reasoning_budget_get_end_match(gsmpl->rbudget);
            if (end_seq) {
                for (const llama_token end_token : *end_seq) {
                    llama_sampler_accept(gsmpl->grmr, end_token);
                }
            }
        }
    }

    if (gsmpl->grmr && accept_grammar) {
        llama_sampler_accept(gsmpl->grmr, token);
    }

    llama_sampler_accept(gsmpl->chain, token);

    gsmpl->prev.push_back(token);
    gsmpl->ahead_valid = false;
}

void common_sampler_reset(struct common_sampler * gsmpl) {
    if (!gsmpl) {
        return;
    }

    gsmpl->reset();
}

int32_t common_sampler_n_history(const struct common_sampler * gsmpl) {
    if (!gsmpl) {
        return 0;
    }

    const auto & params = gsmpl->params;

    // accepted without is_generated, a token reaches only the chain and prev. Of the chain, penalties keep the last
    // penalty_last_n tokens and their counts, DRY the last dry_penalty_last_n (both clamped at 0 by their init, and a
    // sampler disabled by its params keeps nothing); dist and adaptive_p act on an accept only after a draw or an apply,
    // which a reset clears; prev keeps its capacity. Every window evicts exactly, so an older token leaves no trace.
    return std::max({ (int32_t) gsmpl->prev.capacity, std::max(params.penalty_last_n, 0), std::max(params.dry_penalty_last_n, 0) });
}

struct common_sampler * common_sampler_clone(common_sampler * gsmpl) {
    auto * result = new common_sampler {
        /* .params  = */ gsmpl->params,
        /* .grmr    = */ llama_sampler_clone(gsmpl->grmr),
        /* .rbudget = */ llama_sampler_clone(gsmpl->rbudget),
        /* .chain   = */ llama_sampler_clone(gsmpl->chain),
        /* .prev    = */ gsmpl->prev,
        /* .cur     = */ gsmpl->cur,
        /* .cur_p   = */ gsmpl->cur_p,
        /* .rng     = */ gsmpl->rng,
    };
    // derived from params at init: a clone reads the rows its original does
    result->top_k_first  = gsmpl->top_k_first;
    result->n_candidates = gsmpl->n_candidates;
    return result;
}

void common_sampler_copy(const common_sampler * src, common_sampler * dst) {
    if (!src || !dst || src == dst) {
        return;
    }

    GGML_ASSERT((src->grmr == nullptr) == (dst->grmr == nullptr));
    GGML_ASSERT((src->rbudget == nullptr) == (dst->rbudget == nullptr));

    llama_sampler_copy(src->grmr,    dst->grmr);
    llama_sampler_copy(src->rbudget, dst->rbudget);
    llama_sampler_copy(src->chain,   dst->chain);

    dst->params       = src->params;
    dst->top_k_first  = src->top_k_first;
    dst->n_candidates = src->n_candidates;
    dst->prev         = src->prev;
    dst->cur          = src->cur;
    dst->cur_p        = src->cur_p;
    dst->cur_p.data   = src->cur_p.data ? dst->cur.data() : nullptr; // re-point to dst's buffer
    dst->rng          = src->rng;
    dst->t_total_us   = src->t_total_us;
    dst->ahead_valid  = false;
}

void common_perf_print(const struct llama_context * ctx, const struct common_sampler * gsmpl) {
    // TODO: measure grammar performance

    const double t_sampling_ms = gsmpl ? 1e-3*gsmpl->t_total_us : 0;

    llama_perf_sampler_data data_smpl;
    llama_perf_context_data data_ctx;

    memset(&data_smpl, 0, sizeof(data_smpl));
    memset(&data_ctx,  0, sizeof(data_ctx));

    if (gsmpl) {
        auto & data = data_smpl;

        data = llama_perf_sampler(gsmpl->chain);

        // note: the sampling time includes the samplers time + extra time spent in common/sampling
        LOG_INF("%s:    sampling time = %10.2f ms\n", __func__, t_sampling_ms);
        LOG_INF("%s:    samplers time = %10.2f ms / %5d tokens\n", __func__, data.t_sample_ms, data.n_sample);
    }

    if (ctx) {
        auto & data = data_ctx;

        data = llama_perf_context(ctx);

        const double t_end_ms = 1e-3 * ggml_time_us();

        const double t_total_ms = t_end_ms - data.t_start_ms;
        const double t_unacc_ms = t_total_ms - (t_sampling_ms + data.t_p_eval_ms + data.t_eval_ms);
        const double t_unacc_pc = 100.0 * t_unacc_ms /  t_total_ms;

        LOG_INF("%s:        load time = %10.2f ms\n", __func__, data.t_load_ms);
        LOG_INF("%s: prompt eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n",
                __func__, data.t_p_eval_ms, data.n_p_eval, data.t_p_eval_ms / data.n_p_eval, 1e3 / data.t_p_eval_ms * data.n_p_eval);
        LOG_INF("%s:        eval time = %10.2f ms / %5d runs   (%8.2f ms per token, %8.2f tokens per second)\n",
                __func__, data.t_eval_ms, data.n_eval, data.t_eval_ms / data.n_eval, 1e3 / data.t_eval_ms * data.n_eval);
        LOG_INF("%s:       total time = %10.2f ms / %5d tokens\n", __func__, (t_end_ms - data.t_start_ms), (data.n_p_eval + data.n_eval));
        LOG_INF("%s: unaccounted time = %10.2f ms / %5.1f %%      (total - sampling - prompt eval - eval) / (total)\n", __func__, t_unacc_ms, t_unacc_pc);
        LOG_INF("%s:    graphs reused = %10d\n", __func__, data.n_reused);
        LOG_INF("%s:   graph reallocs = %10d    / %5d        (same size) / (changed size)\n", __func__, data.n_reallocs_same, data.n_reallocs_diff);

        common_memory_breakdown_print(ctx);
    }
}

struct llama_sampler * common_sampler_get(const struct common_sampler * gsmpl) {
    if (!gsmpl) {
        return nullptr;
    }

    return gsmpl->chain;
}

llama_token common_sampler_sample(struct common_sampler * gsmpl, struct llama_context * ctx, int idx, bool grammar_first) {
    const auto row = llama_get_sampling_row_ith(ctx, idx);

    // start measuring sampling time after the llama_context synchronization in order to not measure any ongoing async operations
    const auto tm = gsmpl->tm();

    llama_token id = LLAMA_TOKEN_NULL;

    auto & grmr  = gsmpl->grmr;
    auto & rbudget = gsmpl->rbudget;
    auto & chain = gsmpl->chain;
    auto & cur_p = gsmpl->cur_p; // initialized by set_logits

    // the chain's top-k picks first unless something before the chain needs every token: a reasoning budget forcing its
    // token or a grammar applied first (a grammar's resample below also starts from every token)
    const bool every_token = (rbudget && common_reasoning_budget_get_state(rbudget) == REASONING_BUDGET_FORCING) ||
        (grammar_first && grammar_should_apply(gsmpl));

    gsmpl->set_logits(ctx, row, every_token ? 0 : gsmpl->n_candidates);

    // Check if a backend sampler has already sampled a token in which case we
    // return that token id directly.
    {
        id = row.token;

        if (id != LLAMA_TOKEN_NULL) {
            LOG_DBG("%s: Backend sampler selected token: '%d'. Will not run any CPU samplers\n", __func__, id);

            // a grammar or reasoning budget that would change these logits must have taken the slot off the backend
            GGML_ASSERT(common_sampler_backend_passive(gsmpl) && "backend-sampled token while the grammar or reasoning budget constrains");

            for (size_t i = 0; i < cur_p.size; ++i) {
                if (cur_p.data[i].id == id) {
                    cur_p.selected = i;
                    break;
                }
            }

            return id;
        }

        LOG_DBG("%s: no backend-sampled token at idx %d: the CPU samplers run\n", __func__, idx);
    }

    // a row a backend cut to its top k holds no other token: as for a sampled token, a grammar or reasoning budget that
    // would change these logits must have taken the slot off the backend
    GGML_ASSERT((common_sampler_backend_passive(gsmpl) || row.sampled_logits == nullptr) &&
            "backend top-k candidates while the grammar or reasoning budget constrains");

    // apply reasoning budget first
    llama_sampler_apply(rbudget, &cur_p);

    if (grammar_first && grammar_should_apply(gsmpl)) {
        llama_sampler_apply(grmr, &cur_p);
    }

    llama_sampler_apply(chain, &cur_p);

    id = cur_p.data[cur_p.selected].id;

    if (grammar_first || !grammar_should_apply(gsmpl)) {
        return id;
    }

    // check if it the sampled token fits the grammar (grammar-based rejection sampling)
    {
        llama_token_data       single_token_data       = { id, 1.0f, 0.0f };
        llama_token_data_array single_token_data_array = { &single_token_data, 1, -1, false };

        llama_sampler_apply(grmr, &single_token_data_array);

        const bool is_valid = single_token_data_array.data[0].logit != -INFINITY;
        if (is_valid) {
            return id;
        }
    }

    // resampling:
    // if the token is not valid, sample again, but first apply the grammar sampler and then the sampling chain
    gsmpl->set_logits(ctx, row);

    llama_sampler_apply(rbudget,  &cur_p);

    if (grammar_should_apply(gsmpl)) {
        llama_sampler_apply(grmr,  &cur_p);
    }

    llama_sampler_apply(chain, &cur_p);

    GGML_ASSERT(cur_p.selected != -1 && "no selected token during sampling - check your sampling configuration");

    id = cur_p.data[cur_p.selected].id;

    return id;
}

llama_token common_sampler_sample_coupled(struct common_sampler * gsmpl, const llama_token_data * cands, size_t n_cands,
        const llama_token * proposed, size_t n_proposed) {
    if (!gsmpl || n_cands == 0 || !llama_sampler_chain_coupled(gsmpl->chain)) {
        return LLAMA_TOKEN_NULL;
    }

    const auto tm = gsmpl->tm();

    // a copy of the chain draws, never the chain: the chain's first draw after an accept is the one foretold, and a draw
    // here would take its place. The tokens proposed before this one are history when the target samples it, so the
    // copy accepts them. It is made again unless what it accepted since the chain last changed is the start of these
    // and it has not drawn after them
    auto & ahead = gsmpl->ahead;
    if (!gsmpl->ahead_valid || ahead.size() > n_proposed || (ahead.size() == n_proposed && gsmpl->ahead_drawn) ||
            !std::equal(ahead.begin(), ahead.end(), proposed)) {
        if (gsmpl->chain_ahead == nullptr) {
            gsmpl->chain_ahead = llama_sampler_clone(gsmpl->chain);
        } else {
            llama_sampler_copy(gsmpl->chain, gsmpl->chain_ahead);
        }
        gsmpl->ahead_valid = true;
        gsmpl->ahead_drawn = false;
        ahead.clear();
    }
    while (ahead.size() < n_proposed) {
        llama_sampler_accept(gsmpl->chain_ahead, proposed[ahead.size()]);
        ahead.push_back(proposed[ahead.size()]);
        gsmpl->ahead_drawn = false;
    }

    auto & cur = gsmpl->cur_ahead;
    cur.assign(cands, cands + n_cands);
    llama_token_data_array cur_p = { cur.data(), cur.size(), -1, false };

    llama_sampler_apply(gsmpl->chain_ahead, &cur_p);
    gsmpl->ahead_drawn = true;

    return cur_p.selected < 0 ? LLAMA_TOKEN_NULL : cur_p.data[cur_p.selected].id;
}

std::vector<llama_token> common_sampler_sample_and_accept_n(struct common_sampler * gsmpl, struct llama_context * ctx, const std::vector<int> & idxs, const llama_tokens & draft, bool grammar_first,
        const std::function<void(size_t, llama_token)> & on_sample) {
    GGML_ASSERT(idxs.size() == draft.size() + 1 && "idxs.size() must be draft.size() + 1");
    const llama_vocab * vocab = llama_model_get_vocab(llama_get_model(ctx));

    std::vector<llama_token> result;
    result.reserve(idxs.size());

    size_t i = 0;
    for (; i < draft.size(); i++) {
        const llama_token id = common_sampler_sample(gsmpl, ctx, idxs[i], grammar_first);

        common_sampler_accept(gsmpl, id, true);

        result.push_back(id);

        if (on_sample) {
            on_sample(i, id);
        }

        // EOG is the terminal sample, even when the draft agrees. Accepting further
        // rows advances sampler/cache state past the response's actual end.
        if (draft[i] != id || llama_vocab_is_eog(vocab, id)) {
            break;
        }

        // the rows after this one were sampled, or cut to their top k, on the backend without the constraint this token
        // just switched on (a lazy grammar's trigger, the reasoning budget forcing): end here with a valid sample as the
        // last token
        if (!common_sampler_backend_passive(gsmpl)) {
            const auto next = llama_get_sampling_row_ith(ctx, idxs[i + 1]);
            if (next.token != LLAMA_TOKEN_NULL || next.sampled_logits != nullptr) {
                return result;
            }
        }
    }

    if (i == draft.size()) {
        const llama_token id = common_sampler_sample(gsmpl, ctx, idxs[i], grammar_first);

        common_sampler_accept(gsmpl, id, true);

        result.push_back(id);

        if (on_sample) {
            on_sample(i, id);
        }
    }

    return result;
}

static float prob_of(const llama_token_data * data, size_t n, llama_token id) {
    for (size_t k = 0; k < n; ++k) {
        if (data[k].id == id) {
            return data[k].p;
        }
    }
    return 0.0f;
}

// Accept a drafted token with probability min(1, p/q), else draw from norm(max(0, p - q)).
// Preserves the target distribution exactly, and accepts more often than matching does when the
// draft samples instead of taking its argmax.
std::vector<llama_token> common_sampler_sample_and_accept_n_rejection(struct common_sampler * gsmpl, struct llama_context * ctx, const std::vector<int> & idxs, const llama_tokens & draft, const std::vector<std::vector<llama_token_data>> & draft_q, bool grammar_first) {
    GGML_ASSERT(idxs.size()    == draft.size() + 1 && "idxs.size() must be draft.size() + 1");
    GGML_ASSERT(draft_q.size() == draft.size() && "draft_q must have one entry per draft token");

    std::vector<llama_token> result;
    result.reserve(idxs.size());

    // draws come from the sampler's own stream, so they stay independent of what was drafted
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);

    std::vector<llama_token_data> residual;

    std::vector<llama_token_data> cand; // candidate array masked by the grammar, if there is one

    size_t i = 0;
    for (; i < draft.size(); i++) {
        // leaves the target distribution in the candidate array
        const llama_token id_tgt = common_sampler_sample(gsmpl, ctx, idxs[i], grammar_first);

        const auto * cur_p = common_sampler_get_candidates(gsmpl, true);
        const auto & q     = draft_q[i];

        const bool masked = !grammar_first && grammar_should_apply(gsmpl);
        if (masked) {
            cand.assign(cur_p->data, cur_p->data + cur_p->size);
            llama_token_data_array arr = { cand.data(), cand.size(), -1, false };
            llama_sampler_apply(gsmpl->grmr, &arr);
        }

        // a candidate the grammar rejects carries no probability, whatever the target thinks
        auto p_raw = [&](size_t k) {
            return masked && cand[k].logit == -INFINITY ? 0.0f : cur_p->data[k].p;
        };

        // masking drops probability mass, so rescale what is left or the residual is over-weighted
        float p_sum = 0.0f;
        if (masked) {
            for (size_t k = 0; k < cur_p->size; ++k) {
                p_sum += p_raw(k);
            }
        }

        const float p_norm = masked && p_sum > 0.0f ? 1.0f/p_sum : 1.0f;

        auto p_of = [&](size_t k) {
            return p_raw(k)*p_norm;
        };

        // q_x is never 0 for a token the draft produced, but guard the divide
        const float q_x = prob_of(q.data(), q.size(), draft[i]);

        float p_x = 0.0f;
        for (size_t k = 0; k < cur_p->size; ++k) {
            if (cur_p->data[k].id == draft[i]) {
                p_x = p_of(k);
                break;
            }
        }

        if (q_x > 0.0f && (p_x >= q_x || uni(gsmpl->rng) < p_x / q_x)) {
            common_sampler_accept(gsmpl, draft[i], true);
            result.push_back(draft[i]);
            continue;
        }

        // rejected: tokens outside q's support keep all of p
        residual.clear();
        float sum = 0.0f;
        for (size_t k = 0; k < cur_p->size; ++k) {
            const float r = p_of(k) - prob_of(q.data(), q.size(), cur_p->data[k].id);
            if (r > 0.0f) {
                residual.push_back({ cur_p->data[k].id, 0.0f, r });
                sum += r;
            }
        }

        llama_token id = id_tgt;
        if (sum > 0.0f) {
            float u = uni(gsmpl->rng) * sum;
            id = residual.back().id;
            for (const auto & e : residual) {
                u -= e.p;
                if (u <= 0.0f) {
                    id = e.id;
                    break;
                }
            }
        }

        common_sampler_accept(gsmpl, id, true);
        result.push_back(id);

        break;
    }

    if (i == draft.size()) {
        const llama_token id = common_sampler_sample(gsmpl, ctx, idxs[i], grammar_first);

        common_sampler_accept(gsmpl, id, true);

        result.push_back(id);
    }

    return result;
}

std::vector<llama_token> common_sampler_sample_and_accept_n(struct common_sampler * gsmpl, struct llama_context * ctx, const llama_tokens & draft, bool grammar_first) {
    std::vector<int> idxs(draft.size() + 1);
    for (size_t i = 0; i < idxs.size(); ++i) {
        idxs[i] = i;
    }

    return common_sampler_sample_and_accept_n(gsmpl, ctx, idxs, draft, grammar_first);
}

uint32_t common_sampler_get_seed(const struct common_sampler * gsmpl) {
    return llama_sampler_get_seed(gsmpl->chain);
}

bool common_sampler_reasoning_budget_force(struct common_sampler * gsmpl) {
    if (!gsmpl) {
        return false;
    }

    return common_reasoning_budget_force(gsmpl->rbudget);
}

// helpers

llama_token_data_array * common_sampler_get_candidates(struct common_sampler * gsmpl, bool do_sort) {
    const auto tm = gsmpl->tm();

    auto * res = &gsmpl->cur_p;

    if (do_sort && !res->sorted) {
        // remember the selected token before sorting
        const llama_token id = res->data[res->selected].id;

        std::sort(res->data, res->data + res->size, [](const llama_token_data & a, const llama_token_data & b) {
            return a.p > b.p;
        });

        // restore the selected token after sorting
        for (size_t i = 0; i < res->size; ++i) {
            if (res->data[i].id == id) {
                res->selected = i;
                break;
            }
        }

        res->sorted = true;
    }

    return res;
}

llama_token common_sampler_last(const struct common_sampler * gsmpl) {
    return gsmpl->prev.rat(0);
}

std::string common_sampler_print(const struct common_sampler * gsmpl) {
    std::string result = "logits ";

    for (int i = 0; i < llama_sampler_chain_n(gsmpl->chain); i++) {
        const auto * smpl = llama_sampler_chain_get(gsmpl->chain, i);
        result += std::string("-> ");
        result += std::string(llama_sampler_name(smpl)) + " ";
    }

    return result;
}

std::string common_sampler_prev_str(common_sampler * gsmpl, llama_context * ctx_main, int n) {
    n = std::min(n, (int) gsmpl->prev.size());

    if (n <= 0) {
        return "";
    }

    std::string result;
    result.reserve(8*n); // 8 is the average length of a token [citation needed], TODO: compute this from the vocab

    for (int i = n - 1; i >= 0; i--) {
        const llama_token id = gsmpl->prev.rat(i);

        GGML_ASSERT(id != LLAMA_TOKEN_NULL && "null token in the sampling history - should not happen");

        result += common_token_to_piece(ctx_main, id);
    }

    return result;
}

char common_sampler_type_to_chr(enum common_sampler_type cnstr) {
    switch (cnstr) {
        case COMMON_SAMPLER_TYPE_DRY:         return 'd';
        case COMMON_SAMPLER_TYPE_TOP_K:       return 'k';
        case COMMON_SAMPLER_TYPE_TYPICAL_P:   return 'y';
        case COMMON_SAMPLER_TYPE_TOP_P:       return 'p';
        case COMMON_SAMPLER_TYPE_TOP_N_SIGMA: return 's';
        case COMMON_SAMPLER_TYPE_MIN_P:       return 'm';
        case COMMON_SAMPLER_TYPE_TEMPERATURE: return 't';
        case COMMON_SAMPLER_TYPE_XTC:         return 'x';
        case COMMON_SAMPLER_TYPE_INFILL:      return 'i';
        case COMMON_SAMPLER_TYPE_PENALTIES:   return 'e';
        case COMMON_SAMPLER_TYPE_ADAPTIVE_P:  return 'a';
        default : return '?';
    }
}

std::string common_sampler_type_to_str(enum common_sampler_type cnstr) {
    switch (cnstr) {
        case COMMON_SAMPLER_TYPE_DRY:         return "dry";
        case COMMON_SAMPLER_TYPE_TOP_K:       return "top_k";
        case COMMON_SAMPLER_TYPE_TYPICAL_P:   return "typ_p";
        case COMMON_SAMPLER_TYPE_TOP_P:       return "top_p";
        case COMMON_SAMPLER_TYPE_TOP_N_SIGMA: return "top_n_sigma";
        case COMMON_SAMPLER_TYPE_MIN_P:       return "min_p";
        case COMMON_SAMPLER_TYPE_TEMPERATURE: return "temperature";
        case COMMON_SAMPLER_TYPE_XTC:         return "xtc";
        case COMMON_SAMPLER_TYPE_INFILL:      return "infill";
        case COMMON_SAMPLER_TYPE_PENALTIES:   return "penalties";
        case COMMON_SAMPLER_TYPE_ADAPTIVE_P:  return "adaptive_p";
        default : return "";
    }
}

std::vector<common_sampler_type> common_sampler_types_from_names(const std::vector<std::string> & names) {
    // sampler names can be written multiple ways; generate aliases from canonical names
    static const auto sampler_name_map = []{
        // canonical sampler name mapping
        std::unordered_map<std::string, common_sampler_type> canonical_name_map {
            { "dry",         COMMON_SAMPLER_TYPE_DRY         },
            { "top_k",       COMMON_SAMPLER_TYPE_TOP_K       },
            { "top_p",       COMMON_SAMPLER_TYPE_TOP_P       },
            { "top_n_sigma", COMMON_SAMPLER_TYPE_TOP_N_SIGMA },
            { "typ_p",       COMMON_SAMPLER_TYPE_TYPICAL_P   },
            { "min_p",       COMMON_SAMPLER_TYPE_MIN_P       },
            { "temperature", COMMON_SAMPLER_TYPE_TEMPERATURE },
            { "xtc",         COMMON_SAMPLER_TYPE_XTC         },
            { "infill",      COMMON_SAMPLER_TYPE_INFILL      },
            { "penalties",   COMMON_SAMPLER_TYPE_PENALTIES   },
            { "adaptive_p",  COMMON_SAMPLER_TYPE_ADAPTIVE_P  }
        };
        std::unordered_map<std::string, common_sampler_type> alias_name_map;
        for (const auto & entry : canonical_name_map) {
            const std::string & canonical = entry.first;
            if (canonical.find('_') == std::string::npos) {
                continue;
            }
            // kebab-case: "top-k", "min-p", etc.
            {
                std::string kebab_case = canonical;
                std::replace(kebab_case.begin(), kebab_case.end(), '_', '-');
                alias_name_map.insert({kebab_case, entry.second});
            }
            // no dash: "topk", "minp", etc.
            {
                std::string no_dash = canonical;
                no_dash.erase(std::remove(no_dash.begin(), no_dash.end(), '_'), no_dash.end());
                alias_name_map.insert({no_dash, entry.second});
            }
        }
        // misc. aliases
        alias_name_map.insert({"nucleus", COMMON_SAMPLER_TYPE_TOP_P});
        alias_name_map.insert({"temp",    COMMON_SAMPLER_TYPE_TEMPERATURE});
        alias_name_map.insert({"typ",     COMMON_SAMPLER_TYPE_TYPICAL_P});
        // include aliases + canonical names in the complete mapping
        alias_name_map.merge(canonical_name_map);
        return alias_name_map;
    }();

    std::vector<common_sampler_type> samplers;
    samplers.reserve(names.size());

    for (const auto & name : names) {
        std::string name_lower = name;
        std::transform(name_lower.begin(), name_lower.end(), name_lower.begin(), ::tolower);
        auto sampler = sampler_name_map.find(name_lower);
        if (sampler != sampler_name_map.end()) {
            samplers.push_back(sampler->second);
            continue;
        }
        LOG_WRN("%s: unable to match sampler by name '%s'\n", __func__, name_lower.c_str());
    }

    return samplers;
}

std::vector<common_sampler_type> common_sampler_types_from_chars(const std::string & chars) {
    std::unordered_map<char, common_sampler_type> sampler_name_map = {
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_DRY),         COMMON_SAMPLER_TYPE_DRY },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_TOP_K),       COMMON_SAMPLER_TYPE_TOP_K },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_TYPICAL_P),   COMMON_SAMPLER_TYPE_TYPICAL_P },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_TOP_P),       COMMON_SAMPLER_TYPE_TOP_P },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_TOP_N_SIGMA), COMMON_SAMPLER_TYPE_TOP_N_SIGMA },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_MIN_P),       COMMON_SAMPLER_TYPE_MIN_P },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_TEMPERATURE), COMMON_SAMPLER_TYPE_TEMPERATURE },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_XTC),         COMMON_SAMPLER_TYPE_XTC },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_INFILL),      COMMON_SAMPLER_TYPE_INFILL },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_PENALTIES),   COMMON_SAMPLER_TYPE_PENALTIES },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_ADAPTIVE_P),  COMMON_SAMPLER_TYPE_ADAPTIVE_P },
    };

    std::vector<common_sampler_type> samplers;
    samplers.reserve(chars.size());

    for (const auto & c : chars) {
        const auto sampler = sampler_name_map.find(c);
        if (sampler != sampler_name_map.end()) {
            samplers.push_back(sampler->second);
        } else {
            LOG_WRN("%s: unable to match sampler by char '%c'\n", __func__, c);
        }
    }

    return samplers;
}

void common_sampling_add_preserved_tokens(common_params_sampling & sampling, const llama_vocab * vocab, const std::vector<std::string> & tokens) {
    GGML_ASSERT(vocab != nullptr);
    for (const auto & t : tokens) {
        auto ids = common_tokenize(vocab, t, false, true);
        if (ids.size() == 1) {
            sampling.preserved_tokens.insert(ids[0]);
        }
    }
}

void common_sampling_add_grammar_triggers(common_params_sampling & sampling, const llama_vocab * vocab, std::vector<common_grammar_trigger> triggers) {
    GGML_ASSERT(vocab != nullptr);
    for (auto & trigger : triggers) {
        if (trigger.type == COMMON_GRAMMAR_TRIGGER_TYPE_WORD) {
            const auto & word = trigger.value;
            auto ids = common_tokenize(vocab, word, false, true);
            if (ids.size() == 1) {
                auto token = ids[0];
                if (std::find(sampling.preserved_tokens.begin(), sampling.preserved_tokens.end(), (llama_token) token) == sampling.preserved_tokens.end()) {
                    throw std::runtime_error("Grammar trigger word should be marked as preserved token: " + word);
                }
                common_grammar_trigger token_trigger;
                token_trigger.type  = COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN;
                token_trigger.value = word;
                token_trigger.token = token;
                sampling.grammar_triggers.push_back(std::move(token_trigger));
            } else {
                sampling.grammar_triggers.push_back({COMMON_GRAMMAR_TRIGGER_TYPE_WORD, word});
            }
        } else {
            sampling.grammar_triggers.push_back(std::move(trigger));
        }
    }
    if (sampling.grammar_lazy && sampling.grammar_triggers.empty()) {
        throw std::runtime_error("Error: no triggers set for lazy grammar!");
    }
}
