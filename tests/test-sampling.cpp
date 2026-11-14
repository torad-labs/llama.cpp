#include "ggml.h"
#include "llama.h"
#include "sampling.h"

#ifdef NDEBUG
#undef NDEBUG
#endif

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <numeric>
#include <random>
#include <string>
#include <vector>

extern struct llama_sampler * llama_sampler_init_dry_testing(float dry_multiplier, float dry_base, int32_t dry_allowed_length, int32_t dry_penalty_last_n, const std::vector<std::vector<llama_token>>& seq_breakers);

static void dump(const llama_token_data_array * cur_p) {
    for (size_t i = 0; i < cur_p->size; i++) {
        printf("%d: %f (%f)\n", cur_p->data[i].id, cur_p->data[i].p, cur_p->data[i].logit);
    }
}

#define DUMP(__cur_p) do { printf("%s:%d (%s)\n", __FILE__, __LINE__, __func__); dump((__cur_p)); printf("-\n"); } while(0)

struct sampler_tester {
    sampler_tester(size_t n_vocab) {
        cur.reserve(n_vocab);
        for (llama_token token_id = 0; token_id < (llama_token)n_vocab; token_id++) {
            const float logit = logf(token_id);
            cur.emplace_back(llama_token_data{token_id, logit, 0.0f});
        }

        cur_p = llama_token_data_array { cur.data(), cur.size(), -1, false };
    }

    sampler_tester(const std::vector<float> & probs, const std::vector<float> & probs_expected) : probs_expected(probs_expected) {
        cur.reserve(probs.size());
        for (llama_token token_id = 0; token_id < (llama_token)probs.size(); token_id++) {
            const float logit = logf(probs[token_id]);
            cur.emplace_back(llama_token_data{token_id, logit, probs[token_id]});
        }

        cur_p = llama_token_data_array { cur.data(), cur.size(), -1, false };
    }

    void apply(llama_sampler * sampler) {
        llama_sampler_apply(sampler, &cur_p);
        llama_sampler_free(sampler);
    }

    void check() {
        GGML_ASSERT(cur_p.size == probs_expected.size());
        for (size_t i = 0; i < cur_p.size; i++) {
            GGML_ASSERT(fabs(cur_p.data[i].p - probs_expected[i]) < 1e-5);
        }
    }

    llama_token_data_array cur_p;

private:
    const std::vector<float> probs_expected;

    std::vector<llama_token_data> cur;
};

static llama_token sample_dist(llama_sampler * sampler, const std::vector<float> & logits) {
    std::vector<llama_token_data> cur;
    for (llama_token token_id = 0; token_id < (llama_token) logits.size(); ++token_id) {
        cur.push_back({ token_id, logits[token_id], 0.0f });
    }

    llama_token_data_array cur_p = { cur.data(), cur.size(), -1, false };
    llama_sampler_apply(sampler, &cur_p);
    GGML_ASSERT(cur_p.selected >= 0);
    GGML_ASSERT((size_t) cur_p.selected < cur_p.size);
    return cur_p.data[cur_p.selected].id;
}

static void test_dist_singleton_rng() {
    llama_sampler * singleton = llama_sampler_init_dist(4242);
    llama_sampler * control   = llama_sampler_init_dist(4242);

    sample_dist(singleton, { 0.0f });
    sample_dist(control,   { 0.0f, 0.0f });

    const std::vector<float> logits(256, 0.0f);
    for (int i = 0; i < 4; ++i) {
        GGML_ASSERT(sample_dist(singleton, logits) == sample_dist(control, logits));
    }

    llama_sampler_free(singleton);
    llama_sampler_free(control);
}

static void test_temp(const std::vector<float> & probs, const std::vector<float> & probs_expected, float temp) {
    sampler_tester tester(probs, probs_expected);

    DUMP(&tester.cur_p);
    tester.apply(llama_sampler_init_temp(temp));
    tester.apply(llama_sampler_init_dist(0));
    DUMP(&tester.cur_p);

    tester.check();
}

static void test_temp_ext(const std::vector<float> & probs, const std::vector<float> & probs_expected, float temp, float delta, float exponent) {
    sampler_tester tester(probs, probs_expected);

    DUMP(&tester.cur_p);
    tester.apply(llama_sampler_init_temp_ext(temp, delta, exponent));
    tester.apply(llama_sampler_init_dist (0));
    DUMP(&tester.cur_p);

    tester.check();
}

static void test_top_k(const std::vector<float> & probs, const std::vector<float> & probs_expected, int k) {
    sampler_tester tester(probs, probs_expected);

    DUMP(&tester.cur_p);
    tester.apply(llama_sampler_init_top_k(k));
    tester.apply(llama_sampler_init_dist (0));
    DUMP(&tester.cur_p);

    tester.check();
}

static void test_top_p(const std::vector<float> & probs, const std::vector<float> & probs_expected, float p) {
    sampler_tester tester(probs, probs_expected);

    DUMP(&tester.cur_p);
    tester.apply(llama_sampler_init_top_p(p, 0));
    tester.apply(llama_sampler_init_dist (0));
    DUMP(&tester.cur_p);

    tester.check();
}

static void test_min_p(const std::vector<float> & probs, const std::vector<float> & probs_expected, float p) {
    sampler_tester tester(probs, probs_expected);

    DUMP(&tester.cur_p);
    tester.apply(llama_sampler_init_min_p(p, 0));
    tester.apply(llama_sampler_init_dist (0));
    DUMP(&tester.cur_p);

    tester.check();
}

static void test_xtc(const std::vector<float> & probs, const std::vector<float> & probs_expected, float p, float t) {
    sampler_tester tester(probs, probs_expected);

    DUMP(&tester.cur_p);
    tester.apply(llama_sampler_init_xtc(p, t, 0, 0));
    DUMP(&tester.cur_p);

    tester.check();
}

static void test_typical(const std::vector<float> & probs, const std::vector<float> & probs_expected, float p) {
    sampler_tester tester(probs, probs_expected);

    DUMP(&tester.cur_p);
    tester.apply(llama_sampler_init_typical(p, 0));
    DUMP(&tester.cur_p);

    tester.check();
}

static void test_penalties(
    const std::vector<float> & probs, const std::vector<llama_token> & last_tokens,
    const std::vector<float> & probs_expected, float repeat_penalty, float alpha_frequency, float alpha_presence
) {
    GGML_ASSERT(probs.size() == probs_expected.size());

    sampler_tester tester(probs, probs_expected);

    auto * sampler = llama_sampler_init_penalties((int32_t) probs.size(), (int32_t) last_tokens.size(), repeat_penalty, alpha_frequency, alpha_presence);

    for (size_t i = 0; i < last_tokens.size(); i++) {
        llama_sampler_accept(sampler, last_tokens[i]);
    }

    DUMP(&tester.cur_p);
    tester.apply(sampler);
    tester.apply(llama_sampler_init_dist(0));
    DUMP(&tester.cur_p);

    tester.check();
}

static void test_dry(
    const std::vector<float> & probs, const std::vector<llama_token> & last_tokens,
    const std::vector<float> & expected_probs, float dry_multiplier, float dry_base,
    int dry_allowed_length, int dry_penalty_last_n,
    const std::vector<std::vector<llama_token>> & seq_breakers
) {
    GGML_ASSERT(probs.size() == expected_probs.size());

    sampler_tester tester(probs, expected_probs);

    auto * sampler = llama_sampler_init_dry_testing(dry_multiplier, dry_base, dry_allowed_length, dry_penalty_last_n, seq_breakers);

    for (size_t i = 0; i < last_tokens.size(); i++) {
        llama_sampler_accept(sampler, last_tokens[i]);
    }

    DUMP(&tester.cur_p);
    tester.apply(sampler);
    tester.apply(llama_sampler_init_dist(0));
    DUMP(&tester.cur_p);
    tester.check();
}

static void test_top_n_sigma(const std::vector<float> & probs, const std::vector<float> & probs_expected, int n) {
    sampler_tester tester(probs, probs_expected);

    DUMP(&tester.cur_p);
    tester.apply(llama_sampler_init_top_n_sigma(n));
    tester.apply(llama_sampler_init_dist (0));
    DUMP(&tester.cur_p);

    tester.check();
}

static void test_sampler_queue(const size_t n_vocab, const std::string & samplers_sequence, const int top_k, const float top_p, const float min_p
) {
    sampler_tester tester(n_vocab);

          llama_token min_token_id = 0;
    const llama_token max_token_id = n_vocab - 1;

    for (auto s : samplers_sequence) {
        switch (s) {
            case 'k': tester.apply(llama_sampler_init_top_k(top_k)); break;
            case 'y': GGML_ABORT("typical test not implemented");
            case 'p': tester.apply(llama_sampler_init_top_p(top_p, 1)); break;
            case 'm': tester.apply(llama_sampler_init_min_p(min_p, 1)); break;
            case 't': GGML_ABORT("temperature test not implemented");
            default : GGML_ABORT("Unknown sampler");
        }

        tester.apply(llama_sampler_init_dist(0));

        auto & cur_p = tester.cur_p;

        const int size = cur_p.size;

        if (s == 'k') {
            const int expected_size = std::min(size, top_k);
            min_token_id = std::max(min_token_id, (llama_token)(n_vocab - top_k));

            GGML_ASSERT(size == expected_size);
            GGML_ASSERT(cur_p.data[0].id == max_token_id);
            GGML_ASSERT(cur_p.data[expected_size-1].id == min_token_id);
        } else if (s == 'p') {
            const int softmax_divisor = n_vocab * (n_vocab-1) / 2 - min_token_id * (min_token_id-1) / 2;
            const int softmax_numerator_target = ceilf(top_p * softmax_divisor);

                min_token_id  = n_vocab;
            int expected_size = 0;
            int cumsum        = 0;
            do { // do-while because always at least one token is sampled
                min_token_id--;
                expected_size++;

                cumsum += min_token_id;
            } while (cumsum < softmax_numerator_target);

            // token 0 has p == 0, need special consideration for cumsum because top_p immediately returns
            if (min_token_id == 1) {
                min_token_id--;
                expected_size += 1;
            }

            GGML_ASSERT(size == expected_size);
            GGML_ASSERT(!cur_p.sorted || cur_p.data[0].id == max_token_id);
            GGML_ASSERT(!cur_p.sorted || cur_p.data[expected_size-1].id == min_token_id);
        } else if (s == 'm') {
            int expected_size = ceilf((1.0f - min_p) * n_vocab);
            expected_size = std::max(expected_size, 1);
            expected_size = std::min(expected_size, size);

            min_token_id = floorf(min_p * n_vocab);
            min_token_id = std::max(min_token_id, 1);
            min_token_id = std::max(min_token_id, (llama_token)(n_vocab - size));
            min_token_id = std::min(min_token_id, (llama_token)(n_vocab - 1));

            GGML_ASSERT(size == expected_size);
            GGML_ASSERT(!cur_p.sorted || cur_p.data[0].id == max_token_id);
            GGML_ASSERT(!cur_p.sorted || cur_p.data[expected_size-1].id == min_token_id);
        } else {
            GGML_ABORT("fatal error");
        }
    }

    printf("Sampler queue %3s OK with n_vocab=%05zu top_k=%5d top_p=%f min_p=%f\n",
           samplers_sequence.c_str(), n_vocab, top_k, top_p, min_p);
}

static void bench(llama_sampler * cnstr, const char * cnstr_name, const std::vector<llama_token_data> & data, int n_iter) {
    std::vector<llama_token_data> cur(data.size());
    std::copy(data.begin(), data.end(), cur.begin());
    llama_token_data_array cur_p = { cur.data(), cur.size(), -1, false };
    llama_sampler_apply(cnstr, &cur_p);
    llama_sampler_reset(cnstr);
    const int64_t t_start = ggml_time_us();
    for (int i = 0; i < n_iter; i++) {
        std::copy(data.begin(), data.end(), cur.begin());
        llama_token_data_array cur_p = { cur.data(), cur.size(), -1, false };
        llama_sampler_apply(cnstr, &cur_p);
        llama_sampler_reset(cnstr);
    }
    const int64_t t_end = ggml_time_us();
    llama_sampler_free(cnstr);
    printf("%-43s: %8.3f us/iter\n", cnstr_name, (t_end - t_start) / (float)n_iter);
}

#define BENCH(__cnstr, __data, __n_iter) bench((__cnstr), #__cnstr, (__data), (__n_iter))

static void test_perf() {
    const int n_vocab = 1 << 17;

    std::vector<llama_token_data> data;

    data.reserve(n_vocab);
    for (int i = 0; i < n_vocab; i++) {
        const float logit = 2.0f*((double)(rand())/RAND_MAX - 0.5);
        data.emplace_back(llama_token_data{i, logit, 0.0f});
    }

    BENCH(llama_sampler_init_top_k  (40),                     data, 32);
    BENCH(llama_sampler_init_top_p  (0.8f, 1),                data, 32);
    BENCH(llama_sampler_init_min_p  (0.2f, 1),                data, 32);
    BENCH(llama_sampler_init_typical(0.5f, 1),                data, 32);
    BENCH(llama_sampler_init_xtc    (1.0f, 0.1f, 1, 1),       data, 32);
}

// a chain whose top-k comes behind penalties, as a GGUF's sampling defaults build one
struct penalized_chain {
    int32_t k;
    int32_t last_n;
    float   repeat;
    float   freq;
    float   present;
    float   top_p;
    float   min_p;
};

// the token the chain draws from a row, then the candidates it kept in its order: from every token's logit
// (n_candidates 0), or from the n_candidates largest as common_sampler reads them
static std::vector<llama_token_data> penalized_draw(const penalized_chain & c, const std::vector<float> & logits,
        const std::vector<llama_token> & history, int32_t n_candidates, uint32_t seed) {
    const int32_t n_vocab = (int32_t) logits.size();

    llama_sampler * chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(chain, llama_sampler_init_penalties(n_vocab, c.last_n, c.repeat, c.freq, c.present));
    llama_sampler_chain_add(chain, llama_sampler_init_top_k(c.k));
    llama_sampler_chain_add(chain, llama_sampler_init_top_p(c.top_p, 1));
    llama_sampler_chain_add(chain, llama_sampler_init_min_p(c.min_p, 1));
    llama_sampler_chain_add(chain, llama_sampler_init_temp(1.0f));
    llama_sampler_chain_add(chain, llama_sampler_init_dist(seed));
    for (const llama_token token : history) {
        llama_sampler_accept(chain, token);
    }

    std::vector<llama_token_data> cur;
    if (n_candidates > 0) {
        common_sampler_top_k_candidates(logits.data(), n_vocab, n_candidates, cur);
    } else {
        for (llama_token id = 0; id < n_vocab; ++id) {
            cur.push_back(llama_token_data{id, logits[id], 0.0f});
        }
    }
    llama_token_data_array cur_p = { cur.data(), cur.size(), -1, n_candidates > 0 };
    llama_sampler_apply(chain, &cur_p);
    llama_sampler_free(chain);

    std::vector<llama_token_data> out = { cur_p.data[cur_p.selected] };
    out.insert(out.end(), cur_p.data, cur_p.data + cur_p.size);
    return out;
}

// the same tokens in the same order with bit-equal logits and probabilities
static bool same_draw(const std::vector<llama_token_data> & a, const std::vector<llama_token_data> & b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i].id != b[i].id || a[i].logit != b[i].logit || a[i].p != b[i].p) {
            return false;
        }
    }
    return true;
}

// The k + last_n largest logits hold the k a top-k behind penalties keeps, a tie cut at rank k + last_n included, and
// one fewer does not: token 9 leads and is the window's one token, the penalty drops it below the three tied at 4, and
// of those the candidates keep ids 2 and 5 (by id), which are the k every token's top-k keeps
static void test_top_k_penalties_candidates_boundary() {
    std::vector<float> logits(12, 1.0f);
    logits[9] = 10.0f;
    logits[2] = logits[5] = logits[8] = 4.0f;
    const penalized_chain c = { 2, 1, 100.0f, 0.0f, 0.0f, 1.0f, 0.0f };
    const std::vector<llama_token> history = { 9 };

    const auto every = penalized_draw(c, logits, history, 0, 7);
    GGML_ASSERT(every.size() == 3 && every[1].id == 2 && every[2].id == 5);
    GGML_ASSERT(same_draw(penalized_draw(c, logits, history, c.k + c.last_n, 7), every));
    GGML_ASSERT(!same_draw(penalized_draw(c, logits, history, c.k + c.last_n - 1, 7), every));
}

// rows on a grid of logits, so equal logits sit at every rank and at every cut, behind windows drawn mostly from the
// largest logits as a model's own recent tokens are, longer than last_n so the window slides: the same draw from the
// k + last_n largest as from every token, for each chain
static void test_top_k_penalties_candidates_random() {
    const penalized_chain chains[] = {
        {  40, 256, 1.1f, 0.0f, 0.0f, 0.95f, 0.05f }, // GLM-5.3-Flash's GGUF sampling, the server's min_p
        {  40,  64, 1.5f, 0.2f, 0.3f, 1.0f,  0.0f  },
        {   5,  16, 1.0f, 0.5f, 0.0f, 0.9f,  0.0f  },
        {   1,   8, 3.0f, 0.0f, 1.0f, 1.0f,  0.0f  },
        { 128, 896, 1.1f, 0.0f, 0.0f, 1.0f,  0.0f  }, // the most candidates common_sampler cuts a row to
    };
    const int32_t n_vocab = 4096;

    std::mt19937 rng(20261001);
    std::normal_distribution<float> normal(0.0f, 3.0f);
    for (const auto & c : chains) {
        for (int trial = 0; trial < 200; ++trial) {
            std::vector<float> logits(n_vocab);
            for (float & logit : logits) {
                logit = std::round(normal(rng) * 2.0f) / 2.0f;
            }
            std::vector<llama_token> by_logit(n_vocab);
            std::iota(by_logit.begin(), by_logit.end(), 0);
            std::stable_sort(by_logit.begin(), by_logit.end(), [&](llama_token a, llama_token b) {
                return logits[a] > logits[b];
            });
            std::uniform_int_distribution<int32_t> rank(0, 2*c.k + c.last_n);
            std::vector<llama_token> history;
            for (int32_t i = 0; i < c.last_n + 7; ++i) {
                history.push_back(by_logit[rank(rng)]);
            }
            const uint32_t seed = rng();
            GGML_ASSERT(same_draw(penalized_draw(c, logits, history, c.k + c.last_n, seed),
                                  penalized_draw(c, logits, history, 0, seed)));
        }
    }
}

// common_sampler_top_k_candidates keeps the k a sort of every token keeps, in its order: rows on a grid of logits, so
// equal logits sit at the cut, at vocabularies that end inside a block of the scan, too short for the bar's blocks
// and long enough for them, and at k from 1 to more than half the row, where the buffer never fills
static void test_top_k_candidates_exact() {
    std::mt19937 rng(20261109);
    std::normal_distribution<float> normal(0.0f, 3.0f);
    for (const int32_t n_vocab : { 12, 33, 600, 4096, 20011, 154880 }) {
        for (const int32_t k : { 1, 5, 40, 296, 1024 }) {
            if (k >= n_vocab) {
                continue;
            }
            for (int trial = 0; trial < 20; ++trial) {
                std::vector<float> logits(n_vocab);
                for (float & logit : logits) {
                    logit = trial % 2 ? normal(rng) : std::round(normal(rng) * 2.0f) / 2.0f;
                }
                // a row whose logits rise with the id passes every logit to the buffer, one that falls passes none
                if (trial % 5 == 3) std::sort(logits.begin(), logits.end());
                if (trial % 5 == 4) std::sort(logits.begin(), logits.end(), std::greater<float>());

                std::vector<llama_token_data> want;
                for (llama_token id = 0; id < n_vocab; ++id) {
                    want.push_back(llama_token_data{id, logits[id], 0.0f});
                }
                std::sort(want.begin(), want.end(), [](const llama_token_data & a, const llama_token_data & b) {
                    return a.logit > b.logit || (a.logit == b.logit && a.id < b.id);
                });
                want.resize(k);

                std::vector<llama_token_data> got;
                common_sampler_top_k_candidates(logits.data(), n_vocab, k, got);
                GGML_ASSERT(same_draw(got, want));
            }
        }
    }
}

// a served chain: a repeat penalty over 256 tokens, top-k 40, top-p 0.95, min-p 0.05, temperature 1 and the draw
static llama_sampler * coupled_chain(int32_t n_vocab, uint32_t seed, bool coupled = true, float xtc_p = -1.0f) {
    llama_sampler * chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(chain, llama_sampler_init_penalties(n_vocab, 256, 1.1f, 0.0f, 0.0f));
    llama_sampler_chain_add(chain, llama_sampler_init_top_k(40));
    llama_sampler_chain_add(chain, llama_sampler_init_top_p(0.95f, 1));
    llama_sampler_chain_add(chain, llama_sampler_init_min_p(0.05f, 1));
    if (xtc_p >= 0.0f) {
        llama_sampler_chain_add(chain, llama_sampler_init_xtc(xtc_p, 0.1f, 1, seed));
    }
    llama_sampler_chain_add(chain, llama_sampler_init_temp(1.0f));
    llama_sampler_chain_add(chain, coupled ? llama_sampler_init_dist_coupled(seed) : llama_sampler_init_dist(seed));
    return chain;
}

static llama_token draw_from(llama_sampler * smpl, const std::vector<float> & logits, std::vector<llama_token_data> & cur) {
    cur.clear();
    for (llama_token id = 0; id < (llama_token) logits.size(); ++id) {
        cur.push_back(llama_token_data{id, logits[id], 0.0f});
    }
    llama_token_data_array cur_p = { cur.data(), cur.size(), -1, false };
    llama_sampler_apply(smpl, &cur_p);
    GGML_ASSERT(cur_p.selected >= 0);
    return cur_p.data[cur_p.selected].id;
}

// llama_sampler_init_dist_coupled: its draws follow the distribution; a copy of the chain draws the same token from the
// same logits and, from logits near them, the chain's token far more often than those logits' top token is it; a clone
// and a copy carry the sample's index; and the chain tells when a copy can stand in for it
static void test_dist_coupled() {
    std::vector<llama_token_data> cur;

    {
        const std::vector<float> logits = { 2.0f, 1.0f, 0.0f, -1.0f, -30.0f };
        double sum = 0.0;
        for (const float l : logits) {
            sum += exp((double) l);
        }
        llama_sampler *  smpl = llama_sampler_init_dist_coupled(1234);
        const int        n    = 200000;
        std::vector<int> count(logits.size(), 0);
        for (int i = 0; i < n; ++i) {
            const llama_token id = draw_from(smpl, logits, cur);
            count[id]++;
            llama_sampler_accept(smpl, id);
        }
        for (size_t t = 0; t < logits.size(); ++t) {
            // one standard deviation of a share is at most 0.0011 at this n
            GGML_ASSERT(fabs((double) count[t] / n - exp((double) logits[t]) / sum) < 0.005);
        }
        llama_sampler_free(smpl);
    }

    // a common offset changes nothing: the noise is not rounded away by logits far from 0
    for (const float offset : { 0.0f, 1e20f, -1e20f }) {
        const std::vector<float> logits(3, offset);
        llama_sampler *  smpl = llama_sampler_init_dist_coupled(1234);
        const int        n    = 30000;
        std::vector<int> count(logits.size(), 0);
        for (int i = 0; i < n; ++i) {
            const llama_token id = draw_from(smpl, logits, cur);
            count[id]++;
            llama_sampler_accept(smpl, id);
        }
        for (size_t t = 0; t < logits.size(); ++t) {
            // one standard deviation of a share is 0.0027 at this n
            GGML_ASSERT(fabs((double) count[t] / n - 1.0 / 3.0) < 0.012);
        }
        llama_sampler_free(smpl);
    }

    // a draw made again before the accept is independent of the first, as a grammar's resample needs: weights 2, 4 and
    // 1 under top-k 2, the token of weight 4 rejected and the draw made again over the other two. The token of weight 1
    // then comes at (4/6)(1/3) = 2/9; with the first draw's noise it would come at 2/7
    {
        const std::vector<float> all   = { logf(2.0f), logf(4.0f), logf(1.0f) };
        const std::vector<float> valid = { logf(2.0f), -INFINITY, logf(1.0f) };
        llama_sampler * chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
        llama_sampler_chain_add(chain, llama_sampler_init_top_k(2));
        llama_sampler_chain_add(chain, llama_sampler_init_dist_coupled(4321));
        GGML_ASSERT(llama_sampler_chain_coupled(chain));
        llama_sampler * ahead = llama_sampler_clone(chain);
        const int n = 100000;
        int last = 0, foretold = 0;
        for (int i = 0; i < n; ++i) {
            const llama_token first = draw_from(chain, all, cur);
            foretold += draw_from(ahead, all, cur) == first;
            const llama_token id = first == 1 ? draw_from(chain, valid, cur) : first;
            GGML_ASSERT(id != 1);
            last += id == 2;
            llama_sampler_accept(chain, id);
            llama_sampler_accept(ahead, id);
        }
        printf("coupled draws made again: the token of weight 1 at %.4f (2/9 = 0.2222, 2/7 = 0.2857), the first draw foretold %d of %d\n",
                (double) last / n, foretold, n);
        // one standard deviation is 0.0013 at this n
        GGML_ASSERT(fabs((double) last / n - 2.0 / 9.0) < 0.006);
        GGML_ASSERT(foretold == n);
        llama_sampler_free(ahead);
        llama_sampler_free(chain);
    }

    const int32_t n_vocab = 64;
    std::mt19937  rng(7);
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    std::vector<float> p(n_vocab), q(n_vocab);

    llama_sampler * target = coupled_chain(n_vocab, 99);
    GGML_ASSERT(llama_sampler_chain_coupled(target));
    {
        // a drafter draws once a token, so each of the two has its own copy
        llama_sampler * drafter      = llama_sampler_clone(target);
        llama_sampler * drafter_near = llama_sampler_clone(target);
        GGML_ASSERT(llama_sampler_chain_coupled(drafter));
        const int n = 20000;
        int same = 0, near = 0, top = 0;
        for (int i = 0; i < n; ++i) {
            for (int t = 0; t < n_vocab; ++t) {
                p[t] = 3.0f * gauss(rng);
                q[t] = p[t] + 0.5f * gauss(rng);
            }
            const llama_token d_same = draw_from(drafter, p, cur);
            const llama_token d_near = draw_from(drafter_near, q, cur);
            const llama_token d_top  = (llama_token) (std::max_element(q.begin(), q.end()) - q.begin());
            const llama_token x      = draw_from(target, p, cur);
            same += d_same == x;
            near += d_near == x;
            top  += d_top == x;
            llama_sampler_accept(target, x);
            llama_sampler_accept(drafter, x);
            llama_sampler_accept(drafter_near, x);
        }
        printf("coupled draws: %d of %d from the same logits, %d from logits 0.5 away, the top token of those %d\n", same, n, near, top);
        GGML_ASSERT(same == n);
        GGML_ASSERT(near > top + n / 20);
        llama_sampler_free(drafter);
        llama_sampler_free(drafter_near);
    }
    {
        llama_sampler * clone = llama_sampler_clone(target);
        llama_sampler * copy  = coupled_chain(n_vocab, 99);
        llama_sampler_copy(target, copy);
        const int64_t t0 = ggml_time_us();
        for (int i = 0; i < 1000; ++i) {
            llama_sampler_copy(target, copy);
        }
        printf("coupled chain copy: %.2f us\n", (ggml_time_us() - t0) / 1000.0);
        for (int i = 0; i < 64; ++i) {
            for (int t = 0; t < n_vocab; ++t) {
                p[t] = 3.0f * gauss(rng);
            }
            const llama_token x = draw_from(target, p, cur);
            GGML_ASSERT(draw_from(clone, p, cur) == x && draw_from(copy, p, cur) == x);
            llama_sampler_accept(target, x);
            llama_sampler_accept(clone, x);
            llama_sampler_accept(copy, x);
        }
        llama_sampler_free(clone);
        llama_sampler_free(copy);
    }
    llama_sampler_free(target);

    // not coupled: the generator's dist sampler, XTC drawing from its own generator, a chain that does not end in the draw
    llama_sampler * plain  = coupled_chain(n_vocab, 99, false);
    llama_sampler * xtc    = coupled_chain(n_vocab, 99, true, 0.5f);
    llama_sampler * xtc0   = coupled_chain(n_vocab, 99, true, 0.0f);
    llama_sampler * behind = coupled_chain(n_vocab, 99);
    llama_sampler_chain_add(behind, llama_sampler_init_top_k(1));
    llama_sampler * bare   = llama_sampler_init_dist_coupled(99);
    GGML_ASSERT(!llama_sampler_chain_coupled(plain) && !llama_sampler_chain_coupled(xtc) && llama_sampler_chain_coupled(xtc0) &&
                !llama_sampler_chain_coupled(behind) && !llama_sampler_chain_coupled(bare) && !llama_sampler_chain_coupled(nullptr));
    llama_sampler_free(plain);
    llama_sampler_free(xtc);
    llama_sampler_free(xtc0);
    llama_sampler_free(behind);
    llama_sampler_free(bare);
}

int main(void) {
    ggml_time_init();

    test_dist_singleton_rng();
    test_dist_coupled();

    test_temp({0.1f, 0.2f, 0.3f, 0.4f}, {0.1f, 0.2f, 0.3f, 0.4f}, 1.0f);
    test_temp({0.1f, 0.2f, 0.3f, 0.4f}, {0.0f, 0.0f, 0.0f, 1.0f}, 0.0f);

    test_temp_ext({0.1f, 0.2f, 0.3f, 0.4f}, {0.1f, 0.2f, 0.3f, 0.4f}, 1.0f, 0.0f, 1.0f);
    test_temp_ext({0.1f, 0.2f, 0.3f, 0.4f}, {0.0f, 0.0f, 0.0f, 1.0f}, 0.0f, 0.0f, 1.0f);

    test_top_k({0.1f, 0.2f, 0.3f, 0.4f}, {1.0f}, 1);
    test_top_k({0.1f, 0.2f, 0.3f, 0.4f}, {0.44444f, 0.33333f, 0.22222f}, 3);
    test_top_k({0.1f, 0.2f, 0.3f, 0.4f}, {0.4f, 0.3f, 0.2f, 0.1f}, 4);
    test_top_k({0.1f, 0.2f, 0.3f, 0.4f}, {0.1f, 0.2f, 0.3f, 0.4f}, 0);

    test_top_p({0.1f, 0.2f, 0.3f, 0.4f}, {1.0f}, 0);
    test_top_p({0.1f, 0.2f, 0.3f, 0.4f}, {0.571429f, 0.428571f}, 0.7f);
    test_top_p({0.1f, 0.2f, 0.3f, 0.4f}, {0.44444f, 0.33333f, 0.22222f}, 0.8f);
    test_top_p({0.1f, 0.2f, 0.3f, 0.4f}, {0.1f, 0.2f, 0.3f, 0.4f}, 1.0f);

    test_min_p({0.1f, 0.2f, 0.3f, 0.4f}, {0.1f/1.0f, 0.2f/1.0f, 0.3f/1.0f, 0.4f/1.0f}, 0.00f);
    test_min_p({0.1f, 0.2f, 0.3f, 0.4f}, {0.1f/1.0f, 0.2f/1.0f, 0.3f/1.0f, 0.4f/1.0f}, 0.24f);
    test_min_p({0.1f, 0.2f, 0.3f, 0.4f}, {0.2f/0.9f, 0.3f/0.9f, 0.4f/0.9f},            0.26f);
    test_min_p({0.1f, 0.2f, 0.3f, 0.4f}, {0.2f/0.9f, 0.3f/0.9f, 0.4f/0.9f},            0.49f);
    test_min_p({0.1f, 0.2f, 0.3f, 0.4f}, {0.3f/0.7f, 0.4f/0.7f},                       0.51f);
    test_min_p({0.1f, 0.2f, 0.3f, 0.4f}, {0.3f/0.7f, 0.4f/0.7f},                       0.74f);
    test_min_p({0.1f, 0.2f, 0.3f, 0.4f}, {0.4f/0.4f},                                  0.76f);
    test_min_p({0.1f, 0.2f, 0.3f, 0.4f}, {0.4f/0.4f},                                  1.00f);
    test_min_p({0.1f, 0.2f, 0.3f, 0.4f}, {0.4f/0.4f},                                  1.05f);

    printf("XTC should:\n");
    test_xtc({0.4f, 0.3f, 0.2f, 0.1f},   {0.1f},                                0.99f, 0.09f);
    test_xtc({0.4f, 0.3f, 0.2f, 0.1f},   {0.2f, 0.1f},                          0.99f, 0.19f);
    test_xtc({0.4f, 0.3f, 0.2f, 0.1f},   {0.3f, 0.2f, 0.1f},                    0.99f, 0.29f);

    printf("XTC should not:\n");
    test_xtc({0.4f, 0.3f, 0.2f, 0.1f},   {0.4f, 0.3f, 0.2f, 0.1f},              0.99f, 0.39f);

    test_typical({0.97f, 0.01f, 0.01f, 0.01f}, {0.97f},            0.5f);
    test_typical({0.4f, 0.2f, 0.2f, 0.2f},     {0.2f, 0.2f, 0.2f}, 0.5f);

    test_penalties({0.2f, 0.2f, 0.2f, 0.2f, 0.2f}, {0}, {0, 0.25f, 0.25f, 0.25f, 0.25f},   50.0f, 0.0f, 0.0f);
    test_penalties({0.2f, 0.2f, 0.2f, 0.2f, 0.2f}, {0, 1, 2}, {0, 0, 0, 0.5f, 0.5f},       50.0f, 0.0f, 0.0f);
    test_penalties({0.2f, 0.2f, 0.2f, 0.2f, 0.2f}, {0, 1, 2, 0, 0}, {0, 0, 0, 0.5f, 0.5f}, 50.0f, 0.0f, 0.0f);

    test_penalties({0.2f, 0.2f, 0.2f, 0.2f, 0.2f}, {0},             {0.000011f, 0.249997f, 0.249997f, 0.249997f, 0.249997f}, 1.0f, 5.0f, 5.0f);
    test_penalties({0.2f, 0.2f, 0.2f, 0.2f, 0.2f}, {0, 1, 2},       {0.000023f, 0.000023f, 0.000023f, 0.499966f, 0.499966f}, 1.0f, 5.0f, 5.0f);
    test_penalties({0.2f, 0.2f, 0.2f, 0.2f, 0.2f}, {0, 1, 2, 0, 0}, {0.000000f, 0.000023f, 0.000023f, 0.499977f, 0.499977f}, 1.0f, 5.0f, 5.0f);


    test_dry({0.25f, 0.25f, 0.25f, 0.25f}, {0, 1}, {0.25f, 0.25f, 0.25f, 0.25f}, 1.0f, 1.1f, 2, 4, {});
    test_dry({0.25f, 0.25f, 0.25f, 0.25f}, {0, 1, 2, 0, 1}, {0.296923f, 0.296923f, 0.109232f, 0.296923f}, 1.0f, 1.1f, 2, 5, {});
    test_dry({0.2f, 0.2f, 0.2f, 0.2f, 0.2f}, {0, 1, 3, 4, 0, 1}, {0.2f, 0.2f, 0.2f, 0.2f, 0.2f}, 1.0f, 1.1f, 2, 6, {{3}});
    test_dry({0.2f, 0.2f, 0.2f, 0.2f, 0.2f}, {0, 1, 2, 0, 1}, {0.241818f, 0.241818f, 0.032727f, 0.241818f, 0.241818f}, 2.0f, 1.1f, 2, 5, {});
    test_dry({0.2f, 0.2f, 0.2f, 0.2f, 0.2f}, {0, 1, 2, 3, 4, 0, 1}, {0.2f, 0.2f, 0.2f, 0.2f, 0.2f}, 1.0f, 1.1f, 4, 7, {});

    test_top_n_sigma({0.1f, 0.2f, 0.3f, 0.4f}, {0.0f, 0.0f, 0.428571f, 0.571429f}, 1.00f);
    test_top_n_sigma({0.1f, 0.2f, 0.3f, 0.4f}, {0.1f, 0.2f, 0.3f, 0.4f}, 0.00f); // top_n_sigma == 0 now represents a no-op rather than greedy decoding as of PR#13345
    test_top_n_sigma({0.1f, 0.2f, 0.3f, 0.4f}, {0.1f, 0.2f, 0.3f, 0.4f}, 3.00f);

    test_sampler_queue(10000, "k", 10000, 1.0f, 1.0f);
    test_sampler_queue(10000, "k",     1, 1.0f, 1.0f);
    test_sampler_queue(10000, "p", 10000, 1.0f, 1.0f);
    test_sampler_queue(10000, "p", 10000, 0.0f, 1.0f);
    test_sampler_queue(10000, "m", 10000, 1.0f, 1.0f);
    test_sampler_queue(10000, "m", 10000, 1.0f, 1e-12);

    test_sampler_queue(10000, "k",   100, 1.0000f, 1.0f);
    test_sampler_queue(10000, "p", 10000, 0.0003f, 1.0f);
    test_sampler_queue(10000, "p", 10000, 0.8000f, 1.0f);
    test_sampler_queue(10000, "m", 10000, 1.0000f, 9997.9f/9999.0f);
    test_sampler_queue(10000, "m", 10000, 1.0000f, 0.1f);

    test_sampler_queue(10000, "kp", 100, 0.8f, 0.1f);
    test_sampler_queue(10000, "km", 100, 0.8f, 0.1f);
    test_sampler_queue(10000, "pk", 100, 0.8f, 0.1f);
    test_sampler_queue(10000, "pm", 100, 0.8f, 0.1f);
    test_sampler_queue(10000, "mk", 100, 0.8f, 0.1f);
    test_sampler_queue(10000, "mp", 100, 0.8f, 9997.9f/9999.0f);
    test_sampler_queue(10000, "mp", 100, 0.8f, 0.1f);

    test_sampler_queue(10000, "kpm", 100, 0.8f, 0.1f);
    test_sampler_queue(10000, "kmp", 100, 0.8f, 0.1f);
    test_sampler_queue(10000, "pkm", 100, 0.8f, 0.1f);
    test_sampler_queue(10000, "pmk", 100, 0.8f, 0.1f);
    test_sampler_queue(10000, "mkp", 100, 0.8f, 0.1f);
    test_sampler_queue(10000, "mpk", 100, 0.8f, 0.1f);

    test_top_k_candidates_exact();
    test_top_k_penalties_candidates_boundary();
    test_top_k_penalties_candidates_random();

    printf("OK\n");

    test_perf();

    return 0;
}
