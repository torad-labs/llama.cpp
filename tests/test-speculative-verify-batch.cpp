// Sequences that verify their drafts side by side keep the tokens each keeps alone.

#include "common.h"
#include "llama.h"
#include "llama-cpp.h"
#include "sampling.h"
#include "speculative.h"

#undef NDEBUG
#include <cassert>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

static const llama_model * g_model   = nullptr;
static const llama_vocab * g_vocab   = nullptr;
static int32_t             g_n_vocab = 0;

// a sequence about to verify its draft: its sampler, its draft and the logits of its rows (the row of the token before
// the draft, then one row per draft token)
struct sequence {
    common_sampler_ptr                      smpl;
    llama_tokens                            draft;
    std::vector<const std::vector<float> *> rows;
    int32_t                                 n_remaining = -1;
};

// the sampling the Inkling 2 launcher serves with: the whole vocabulary at temperature 1
static common_sampler_ptr sampler(uint32_t seed) {
    common_params_sampling params;
    params.seed  = seed;
    params.temp  = 1.0f;
    params.top_k = 0;
    params.top_p = 1.0f;
    params.min_p = 0.0f;
    return common_sampler_ptr(common_sampler_init(g_model, params));
}

static llama_sampling_row row_of(const std::vector<float> & logits) {
    llama_sampling_row row = {};
    row.token  = LLAMA_TOKEN_NULL;
    row.logits = logits.data();
    return row;
}

// the tokens every sequence keeps, verified in one call on n_threads threads
static std::vector<llama_tokens> verify(std::vector<sequence> & seqs, int n_threads) {
    std::vector<common_speculative_verify> jobs(seqs.size());
    for (size_t i = 0; i < seqs.size(); ++i) {
        jobs[i].smpl        = seqs[i].smpl.get();
        jobs[i].draft       = &seqs[i].draft;
        jobs[i].n_remaining = seqs[i].n_remaining;
        for (const auto * logits : seqs[i].rows) {
            jobs[i].rows.push_back(row_of(*logits));
        }
    }
    common_speculative_sample_and_accept_batch(g_vocab, jobs, n_threads);

    std::vector<llama_tokens> kept;
    for (auto & job : jobs) {
        assert(!job.error);
        kept.push_back(std::move(job.accepted));
    }
    return kept;
}

// a row the target draws one token from, whatever its seed
static std::vector<float> only(llama_token token) {
    std::vector<float> logits(g_n_vocab, -1000.0f);
    logits[token] = 0.0f;
    return logits;
}

static void test_what_a_sequence_keeps() {
    const llama_token a = 1000, b = 2000, c = 3000, d = 4000, x = 5000;
    const llama_token end = llama_vocab_eos(g_vocab);
    assert(llama_vocab_is_eog(g_vocab, end));

    const auto row_a = only(a), row_b = only(b), row_c = only(c), row_d = only(d), row_x = only(x), row_end = only(end);

    std::vector<sequence> seqs(5);
    // the target agrees with the whole draft: the draft, then the target's next token
    seqs[0].draft = {a, b, c};
    seqs[0].rows  = {&row_a, &row_b, &row_c, &row_d};
    // the target departs at the second token: its own token ends what is kept
    seqs[1].draft = {a, b, c};
    seqs[1].rows  = {&row_a, &row_x, &row_c, &row_d};
    // an end token is the last one, even where the draft goes on and the target agrees with it
    seqs[2].draft = {a, end, c};
    seqs[2].rows  = {&row_a, &row_end, &row_c, &row_d};
    // a request with two tokens left to emit keeps two
    seqs[3].draft       = {a, b, c};
    seqs[3].rows        = {&row_a, &row_b, &row_c, &row_d};
    seqs[3].n_remaining = 2;
    // no draft: the target's next token
    seqs[4].rows = {&row_a};
    for (size_t i = 0; i < seqs.size(); ++i) {
        seqs[i].smpl = sampler(7 + i);
    }

    const auto kept = verify(seqs, 4);
    assert(kept[0] == llama_tokens({a, b, c, d}));
    assert(kept[1] == llama_tokens({a, x}));
    assert(kept[2] == llama_tokens({a, end}));
    assert(kept[3] == llama_tokens({a, b}));
    assert(kept[4] == llama_tokens({a}));
    // each sampler has accepted what its sequence kept, and nothing after it
    for (size_t i = 0; i < seqs.size(); ++i) {
        assert(common_sampler_last(seqs[i].smpl.get()) == kept[i].back());
    }
}

// rows a drafter guesses right six times in ten: logits of unit noise, and one token holding 0.6 of the probability
struct guessable_rows {
    std::vector<std::vector<float>> logits;
    std::vector<llama_token>        guess;

    guessable_rows(size_t n, std::mt19937 & rng) {
        std::normal_distribution<float>        noise(0.0f, 1.0f);
        std::uniform_int_distribution<int32_t> any_token(0, g_n_vocab - 1);
        for (size_t r = 0; r < n; ++r) {
            std::vector<float> row(g_n_vocab);
            const llama_token  top  = any_token(rng);
            double             rest = 0.0;
            for (int32_t t = 0; t < g_n_vocab; ++t) {
                row[t] = noise(rng);
                rest  += t == top ? 0.0 : std::exp((double) row[t]);
            }
            row[top] = (float) std::log(1.5 * rest);
            logits.push_back(std::move(row));
            guess.push_back(top);
        }
    }
};

// 23 sessions as a server step holds them: drafts of 1 to 7 tokens, each on its own seed, some near their output limit
static std::vector<sequence> sessions(const guessable_rows & pool) {
    std::mt19937                          rng(23);
    std::uniform_int_distribution<size_t> any_row(0, pool.logits.size() - 1);
    std::vector<sequence> seqs(23);
    for (size_t s = 0; s < seqs.size(); ++s) {
        auto & seq = seqs[s];
        seq.smpl = sampler(100 + s);
        const size_t n_draft = 1 + s % 7;
        for (size_t i = 0; i <= n_draft; ++i) {
            const size_t r = any_row(rng);
            seq.rows.push_back(&pool.logits[r]);
            if (i < n_draft) {
                seq.draft.push_back(pool.guess[r]);
            }
        }
        seq.n_remaining = s % 5 == 4 ? 2 + (int32_t) (s % 3) : -1;
    }
    return seqs;
}

// the token each sampler draws next from one more row: equal draws mean the samplers were left in equal states
static llama_tokens next_tokens(std::vector<sequence> & seqs, const std::vector<float> & logits) {
    llama_tokens next;
    for (auto & seq : seqs) {
        next.push_back(common_sampler_sample(seq.smpl.get(), g_vocab, row_of(logits)));
    }
    return next;
}

static void test_side_by_side_keeps_what_each_keeps_alone() {
    std::mt19937         rng(1234);
    const guessable_rows pool(12, rng);

    // each session verified in a call of its own
    std::vector<llama_tokens> alone;
    llama_tokens              alone_next;
    {
        auto seqs = sessions(pool);
        for (auto & seq : seqs) {
            std::vector<sequence> one;
            one.push_back(std::move(seq));
            alone.push_back(verify(one, 1)[0]);
            alone_next.push_back(next_tokens(one, pool.logits[0])[0]);
        }
    }

    // the drafts are worth verifying: some session keeps draft tokens, and some session's draft is cut
    bool kept_draft = false, cut = false;
    {
        const auto seqs = sessions(pool);
        for (size_t s = 0; s < seqs.size(); ++s) {
            kept_draft = kept_draft || alone[s].size() > 1;
            cut        = cut || (seqs[s].n_remaining < 0 && alone[s].size() <= seqs[s].draft.size());
        }
    }
    assert(kept_draft && cut);

    for (const int n_threads : {1, 2, 4, 16, 64}) {
        auto seqs = sessions(pool);
        const auto kept = verify(seqs, n_threads);
        assert(kept == alone);
        assert(next_tokens(seqs, pool.logits[0]) == alone_next);
    }
}

// a draft distribution over the k largest logits of a row at temperature 1, and a token drawn from it
static std::vector<llama_token_data> top_k_q(const std::vector<float> & logits, size_t k) {
    std::vector<llama_token_data> q;
    for (int32_t t = 0; t < g_n_vocab; ++t) {
        q.push_back({ t, logits[t], 0.0f });
    }
    std::partial_sort(q.begin(), q.begin() + k, q.end(),
            [](const llama_token_data & a, const llama_token_data & b) { return a.logit > b.logit; });
    q.resize(k);
    double sum = 0.0;
    for (const auto & c : q) {
        sum += std::exp((double) c.logit - q[0].logit);
    }
    for (auto & c : q) {
        c.p = (float) (std::exp((double) c.logit - q[0].logit)/sum);
    }
    return q;
}

static llama_token draw_from(const std::vector<llama_token_data> & q, std::mt19937 & rng) {
    double u = std::uniform_real_distribution<double>(0.0, 1.0)(rng);
    for (const auto & c : q) {
        u -= c.p;
        if (u < 0.0) {
            return c.id;
        }
    }
    return q.back().id;
}

// sequences whose drafts came with their distribution are verified by rejection, side by side as alone
static void test_rejection_side_by_side_keeps_what_each_keeps_alone() {
    std::mt19937         rng(4321);
    const guessable_rows pool(12, rng);

    const auto make = [&](std::vector<std::vector<std::vector<llama_token_data>>> & qs) {
        std::mt19937 draw(99);
        auto seqs = sessions(pool);
        qs.assign(seqs.size(), {});
        for (size_t s = 0; s < seqs.size(); ++s) {
            for (size_t i = 0; i < seqs[s].draft.size(); ++i) {
                // a draft that has its own idea: the k largest logits of a row it did not see
                qs[s].push_back(top_k_q(*seqs[s].rows[(i + 1) % seqs[s].rows.size()], 16));
                seqs[s].draft[i] = draw_from(qs[s].back(), draw);
            }
        }
        return seqs;
    };
    const auto run = [&](int n_threads, llama_tokens & next) {
        std::vector<std::vector<std::vector<llama_token_data>>> qs;
        auto seqs = make(qs);
        std::vector<common_speculative_verify> jobs(seqs.size());
        for (size_t i = 0; i < seqs.size(); ++i) {
            jobs[i].smpl    = seqs[i].smpl.get();
            jobs[i].draft   = &seqs[i].draft;
            jobs[i].draft_q = &qs[i];
            for (const auto * logits : seqs[i].rows) {
                jobs[i].rows.push_back(row_of(*logits));
            }
        }
        common_speculative_sample_and_accept_batch(g_vocab, jobs, n_threads);
        std::vector<llama_tokens> kept;
        for (auto & job : jobs) {
            assert(!job.error);
            kept.push_back(std::move(job.accepted));
        }
        next = next_tokens(seqs, pool.logits[0]);
        return kept;
    };

    llama_tokens alone_next;
    const auto alone = run(1, alone_next);
    bool kept_draft = false, rejected = false;
    {
        std::vector<std::vector<std::vector<llama_token_data>>> qs;
        const auto seqs = make(qs);
        for (size_t s = 0; s < seqs.size(); ++s) {
            kept_draft = kept_draft || alone[s].size() > 1;
            rejected   = rejected || alone[s].size() <= seqs[s].draft.size();
        }
    }
    assert(kept_draft && rejected);

    for (const int n_threads : {2, 4, 16, 64}) {
        llama_tokens next;
        assert(run(n_threads, next) == alone);
        assert(next == alone_next);
    }
}

// The token a row keeps is a sample of the target's distribution, whatever the draft proposed: a draft over its own top
// k (part of it outside what the target can sample) checked by rejection, against the target's top-k chain. A chi-square
// over the target's support, and the same check catches a draft that proposes from another distribution than its q.
static void test_rejection_keeps_the_target_distribution() {
    const int32_t n_live = 6;
    std::vector<float> target(g_n_vocab, -1000.0f), draft(g_n_vocab, -1000.0f);
    const float target_live[n_live] = { 1.0f, 0.5f, 0.0f, -0.5f, -1.0f, -2.0f };
    const float draft_live [n_live] = { -1.0f, 1.2f, 0.3f, -0.5f, 0.8f, -3.0f };
    for (int32_t t = 0; t < n_live; ++t) {
        target[1000 + 7*t] = target_live[t];
        draft [1000 + 7*t] = draft_live[t];
    }
    draft[50000] = 0.9f; // the target's top-k chain never samples it

    // the chain samples the 5 largest, so the target's 6th live token is out of its support too
    common_params_sampling params;
    params.seed  = 5;
    params.temp  = 1.0f;
    params.top_k = 5;
    params.top_p = 1.0f;
    params.min_p = 0.0f;
    common_sampler_ptr smpl(common_sampler_init(g_model, params));

    double p_sum = 0.0;
    std::vector<double> p(n_live, 0.0);
    for (int32_t t = 0; t < 5; ++t) {
        p[t] = std::exp((double) target_live[t]);
        p_sum += p[t];
    }
    for (auto & x : p) {
        x /= p_sum;
    }

    const auto q      = top_k_q(draft, 4);
    const auto q_skew = top_k_q(target, 4); // what a draft that says q but draws this would propose
    const llama_sampling_row rows[2] = { row_of(target), row_of(target) };

    const auto chi2 = [&](const std::vector<llama_token_data> & proposals, int n_trials) {
        std::mt19937 rng(77);
        std::vector<int> count(n_live, 0);
        int outside = 0;
        for (int n = 0; n < n_trials; ++n) {
            const llama_tokens one = { draw_from(proposals, rng) };
            const auto kept = common_sampler_sample_and_accept_n_rejection(smpl.get(), g_vocab, rows, one, { q });
            const int32_t t = (kept[0] - 1000)/7;
            if (kept[0] < 1000 || (kept[0] - 1000) % 7 != 0 || t >= n_live || p[t] == 0.0) {
                outside++;
                continue;
            }
            count[t]++;
        }
        assert(outside == 0);
        double x2 = 0.0;
        for (int32_t t = 0; t < n_live; ++t) {
            if (p[t] > 0.0) {
                const double e = p[t]*n_trials;
                x2 += (count[t] - e)*(count[t] - e)/e;
            }
        }
        return x2;
    };

    // 4 degrees of freedom: 30 is beyond the 1e-5 tail (25.4)
    const double x2      = chi2(q, 40000);
    const double x2_skew = chi2(q_skew, 40000);
    printf("rejection keeps the target's distribution: chi-square %.2f (4 dof), a draft drawing from another q %.2f\n",
            x2, x2_skew);
    assert(x2 < 30.0);
    assert(x2_skew > 30.0);
}

int main(int argc, char ** argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s <vocab-file>\n", argv[0]);
        return 1;
    }

    llama_backend_init();

    auto mparams = llama_model_default_params();
    mparams.vocab_only = true;
    llama_model_ptr model(llama_model_load_from_file(argv[1], mparams));
    if (!model) {
        fprintf(stderr, "failed to load the vocabulary of '%s'\n", argv[1]);
        return 1;
    }
    g_model   = model.get();
    g_vocab   = llama_model_get_vocab(g_model);
    g_n_vocab = llama_vocab_n_tokens(g_vocab);

    test_what_a_sequence_keeps();
    test_side_by_side_keeps_what_each_keeps_alone();
    test_rejection_side_by_side_keeps_what_each_keeps_alone();
    test_rejection_keeps_the_target_distribution();

    printf("OK speculative verify batch\n");
    return 0;
}
