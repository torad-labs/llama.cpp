#include "testing.h"

#include "llama.h"

#include "../src/llama-batch.h"
#include "../src/llama-memory.h"
#include "../src/llama-vocab.h"

#include <cstdlib>
#include <initializer_list>
#include <map>
#include <string>
#include <utility>
#include <vector>

// mock memory that only provides per-sequence position ranges
struct mock_memory : public llama_memory_i {
    std::map<llama_seq_id, std::pair<llama_pos, llama_pos>> ranges; // seq_id -> [pos_min, pos_max]

    llama_memory_context_ptr init_batch(llama_batch_allocr &, uint32_t, bool) override {  GGML_ASSERT(false && "not implemented"); }
    llama_memory_context_ptr init_full() override {  GGML_ASSERT(false && "not implemented"); }
    llama_memory_context_ptr init_update(llama_context *, bool) override { GGML_ASSERT(false && "not implemented"); }

    bool get_can_shift() const override { GGML_ASSERT(false && "not implemented"); }

    void clear(bool) override { GGML_ASSERT(false && "not implemented"); }

    bool seq_rm  (llama_seq_id, llama_pos, llama_pos) override { GGML_ASSERT(false && "not implemented"); }
    void seq_cp  (llama_seq_id, llama_seq_id, llama_pos, llama_pos) override { GGML_ASSERT(false && "not implemented"); }
    void seq_keep(llama_seq_id) override { GGML_ASSERT(false && "not implemented"); }
    void seq_add (llama_seq_id, llama_pos, llama_pos, llama_pos) override { GGML_ASSERT(false && "not implemented"); }
    void seq_div (llama_seq_id, llama_pos, llama_pos, int) override { GGML_ASSERT(false && "not implemented");  }

    llama_pos seq_pos_min(llama_seq_id seq_id) const override {
        auto it = ranges.find(seq_id);
        return it == ranges.end() ? -1 : it->second.first;
    }

    llama_pos seq_pos_max(llama_seq_id seq_id) const override {
        auto it = ranges.find(seq_id);
        return it == ranges.end() ? -1 : it->second.second;
    }

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override { return {}; }

    void state_write(llama_io_write_i &, llama_seq_id, llama_state_seq_flags) const override { GGML_ASSERT(false && "not implemented"); }
    void state_read (llama_io_read_i &,  llama_seq_id, llama_state_seq_flags) override { GGML_ASSERT(false && "not implemented"); }
};

// builds embedding batches - an empty llama_vocab rejects all token ids, so
// the tests use embeddings everywhere except the token validation tests
struct batch_builder {
    uint32_t n_embd;

    std::vector<float>     embd;
    std::vector<llama_pos> pos;
    std::vector<int32_t>   n_seq_id;
    std::vector<int8_t>    logits;

    std::vector<std::vector<llama_seq_id>> seq;
    std::vector<llama_seq_id *>            seq_ptr;

    batch_builder(uint32_t n_embd = 2) : n_embd(n_embd) {}

    // embd values are 100*i + k so that ubatch contents can be traced back to batch indices
    void add(llama_pos p, std::initializer_list<llama_seq_id> seq_ids, bool output) {
        const int32_t i = (int32_t) seq.size();
        for (uint32_t k = 0; k < n_embd; ++k) {
            embd.push_back(100.0f*i + k);
        }
        pos.push_back(p);
        n_seq_id.push_back((int32_t) seq_ids.size());
        seq.emplace_back(seq_ids);
        logits.push_back(output ? 1 : 0);
    }

    llama_batch make(bool with_pos = true, bool with_seq = true, bool with_logits = true) {
        seq_ptr.clear();
        for (auto & s : seq) {
            seq_ptr.push_back(s.data());
        }
        seq_ptr.push_back(nullptr);

        llama_batch res = {};
        res.n_tokens = (int32_t) seq.size();
        res.embd     = embd.data();
        res.pos      = with_pos    ? pos.data()      : nullptr;
        res.n_seq_id = with_seq    ? n_seq_id.data() : nullptr;
        res.seq_id   = with_seq    ? seq_ptr.data()  : nullptr;
        res.logits   = with_logits ? logits.data()   : nullptr;

        return res;
    }
};

static void test_init(testing & t) {
    llama_vocab vocab;

    t.test("rejects_n_seq_max_too_large", [&](testing & t) {
        batch_builder bb;
        bb.add(0, {0}, true);

        llama_batch_allocr ba(1);
        t.assert_true(!ba.init(bb.make(), vocab, nullptr, bb.n_embd, LLAMA_MAX_SEQ + 1, false));
    });

    t.test("rejects_invalid_token", [&](testing & t) {
        llama_token tok = 0; // empty vocab -> every token id is out of range
        llama_batch batch = llama_batch_get_one(&tok, 1);

        llama_batch_allocr ba(1);
        t.assert_true("token id >= n_tokens", !ba.init(batch, vocab, nullptr, 0, 1, false));

        tok = -1;
        t.assert_true("negative token id", !ba.init(batch, vocab, nullptr, 0, 1, false));
    });

    t.test("rejects_invalid_seq_id", [&](testing & t) {
        llama_batch_allocr ba(1);

        {
            batch_builder bb;
            bb.add(0, {4}, true);
            t.assert_true("seq_id >= n_seq_max", !ba.init(bb.make(), vocab, nullptr, bb.n_embd, 4, false));
        }
        {
            batch_builder bb;
            bb.add(0, {-1}, true);
            t.assert_true("negative seq_id", !ba.init(bb.make(), vocab, nullptr, bb.n_embd, 4, false));
        }
    });

    t.test("autofill_defaults", [&](testing & t) {
        batch_builder bb;
        for (int i = 0; i < 4; ++i) {
            bb.add(0, {0}, false);
        }

        llama_batch_allocr ba(1);
        t.assert_true(ba.init(bb.make(false, false, false), vocab, nullptr, bb.n_embd, 4, false));

        const llama_batch & batch = ba.get_batch();

        t.assert_equal(4u, ba.get_n_tokens());

        for (int i = 0; i < 4; ++i) {
            t.assert_equal("pos defaults to 0..n-1", i, batch.pos[i]);
            t.assert_equal("n_seq_id defaults to 1", 1, batch.n_seq_id[i]);
            t.assert_equal("seq_id defaults to 0",   0, batch.seq_id[i][0]);
        }

        t.assert_equal("only the last token is an output", 1u, ba.get_n_outputs());
        t.assert_equal(0, (int) batch.logits[0]);
        t.assert_equal(1, (int) batch.logits[3]);

        t.assert_equal(0, ba.seq_pos_min(0));
        t.assert_equal(3, ba.seq_pos_max(0));
        t.assert_equal(-1, ba.seq_pos_min(1));
    });

    t.test("output_all", [&](testing & t) {
        batch_builder bb;
        for (int i = 0; i < 4; ++i) {
            bb.add(i, {0}, false);
        }

        llama_batch_allocr ba(1);
        t.assert_true(ba.init(bb.make(true, true, false), vocab, nullptr, bb.n_embd, 4, true));
        t.assert_equal(4u, ba.get_n_outputs());
    });

    t.test("explicit_logits", [&](testing & t) {
        batch_builder bb;
        bb.add(0, {0}, true);
        bb.add(1, {0}, false);
        bb.add(2, {0}, true);

        llama_batch_allocr ba(1);
        t.assert_true(ba.init(bb.make(), vocab, nullptr, bb.n_embd, 4, false));
        t.assert_equal(2u, ba.get_n_outputs());

        llama_ubatch ub = ba.split_simple(10);
        t.assert_equal(3u, ub.n_tokens);
        t.assert_equal(1, (int) ub.output[0]);
        t.assert_equal(0, (int) ub.output[1]);
        t.assert_equal(1, (int) ub.output[2]);

        const auto & out_ids = ba.get_out_ids();
        t.assert_equal((size_t) 2, out_ids.size());
        t.assert_equal(0, out_ids[0]);
        t.assert_equal(2, out_ids[1]);
    });

    t.test("pos_from_memory", [&](testing & t) {
        mock_memory mem;
        mem.ranges[0] = {0, 9};

        batch_builder bb;
        for (int i = 0; i < 3; ++i) {
            bb.add(0, {0}, false);
        }

        llama_batch_allocr ba(1);
        t.assert_true(ba.init(bb.make(false, true, false), vocab, &mem, bb.n_embd, 4, false));

        t.assert_equal("pos continues after memory", 10, ba.seq_pos_min(0));
        t.assert_equal(12, ba.seq_pos_max(0));
    });

    t.test("pos_continuity_with_memory", [&](testing & t) {
        mock_memory mem;
        mem.ranges[0] = {0, 9};

        llama_batch_allocr ba(1);

        {
            batch_builder bb;
            bb.add(10, {0}, false);
            bb.add(11, {0}, true);
            t.assert_true("pos_max + 1 is accepted", ba.init(bb.make(), vocab, &mem, bb.n_embd, 4, false));
        }
        {
            batch_builder bb;
            bb.add(11, {0}, false);
            bb.add(12, {0}, true);
            t.assert_true("gap after memory is rejected", !ba.init(bb.make(), vocab, &mem, bb.n_embd, 4, false));
        }
        {
            batch_builder bb;
            bb.add(9, {0}, false);
            bb.add(10, {0}, true);
            t.assert_true("overlap with memory is rejected", !ba.init(bb.make(), vocab, &mem, bb.n_embd, 4, false));
        }
    });

    t.test("rejects_non_continuous_positions", [&](testing & t) {
        batch_builder bb;
        bb.add(0, {0}, false);
        bb.add(1, {0}, false);
        bb.add(3, {0}, true);

        llama_batch_allocr ba(1);
        t.assert_true(!ba.init(bb.make(), vocab, nullptr, bb.n_embd, 4, false));
    });

    t.test("rejects_decreasing_positions", [&](testing & t) {
        batch_builder bb;
        const llama_pos    pos[7] = {4, 5, 0, 1, 6, 2, 3};
        const llama_seq_id seq[7] = {0, 0, 1, 1, 0, 1, 0};
        for (int i = 0; i < 7; ++i) {
            bb.add(pos[i], {seq[i]}, false);
        }
        // seq 0 sees positions 4,5,6,3 in batch order -> the trailing 3 decreases

        llama_batch_allocr ba(1);
        t.assert_true(!ba.init(bb.make(true, true, false), vocab, nullptr, bb.n_embd, 4, false));
    });

    t.test("allows_equal_positions_in_seq", [&](testing & t) {
        batch_builder bb;
        bb.add(0, {0}, false);
        bb.add(0, {0}, false);
        bb.add(1, {0}, true);

        llama_batch_allocr ba(1);
        t.assert_true(ba.init(bb.make(true, true, false), vocab, nullptr, bb.n_embd, 4, false));
    });


    t.test("rejects_coupled_diverged_seqs", [&](testing & t) {
        batch_builder bb;
        bb.add(6, {0, 1}, true);

        llama_batch_allocr ba(1);

        mock_memory mem;
        mem.ranges[0] = {0, 5};
        mem.ranges[1] = {2, 5}; // same pos_max, different pos_min -> diverged
        t.assert_true(!ba.init(bb.make(), vocab, &mem, bb.n_embd, 4, false));

        mem.ranges[1] = {0, 5};
        t.assert_true(ba.init(bb.make(), vocab, &mem, bb.n_embd, 4, false));
    });
}

static void test_split(testing & t) {
    llama_vocab vocab;

    t.test("split_simple_chunks", [&](testing & t) {
        batch_builder bb;
        for (int i = 0; i < 5; ++i) {
            bb.add(i, {0}, i == 4);
        }

        llama_batch_allocr ba(1);
        t.assert_true(ba.init(bb.make(), vocab, nullptr, bb.n_embd, 4, false));

        llama_ubatch ub = ba.split_simple(2);
        t.assert_equal(2u, ub.n_tokens);
        t.assert_true(!ub.equal_seqs());
        t.assert_equal(1u, ub.n_seqs_unq);
        t.assert_equal(0, ub.seq_id_unq[0]);
        t.assert_equal(0, ub.seq_idx[0]);
        for (int i = 0; i < 2; ++i) {
            t.assert_equal(i, ub.pos[i]);
            t.assert_equal(1, ub.n_seq_id[i]);
            t.assert_equal(0, ub.seq_id[i][0]);
            t.assert_equal(100.0f*i, ub.embd[i*bb.n_embd]);
            t.assert_equal(100.0f*i + 1, ub.embd[i*bb.n_embd + 1]);
        }

        ub = ba.split_simple(2);
        t.assert_equal(2u, ub.n_tokens);
        t.assert_equal(2, ub.pos[0]);
        t.assert_equal(3, ub.pos[1]);

        ub = ba.split_simple(2);
        t.assert_equal(1u, ub.n_tokens);
        t.assert_equal(4, ub.pos[0]);
        t.assert_equal(1, (int) ub.output[0]);

        t.assert_equal(5u, ba.get_n_used());

        ub = ba.split_simple(2);
        t.assert_equal("batch is consumed", 0u, ub.n_tokens);

        const auto & out_ids = ba.get_out_ids();
        t.assert_equal((size_t) 1, out_ids.size());
        t.assert_equal(4, out_ids[0]);
    });

    t.test("split_reset_allows_resplit", [&](testing & t) {
        batch_builder bb;
        for (int i = 0; i < 3; ++i) {
            bb.add(i, {0}, i == 2);
        }

        llama_batch_allocr ba(1);
        t.assert_true(ba.init(bb.make(), vocab, nullptr, bb.n_embd, 4, false));

        while (ba.split_simple(1).n_tokens > 0) {
        }
        t.assert_equal(3u, ba.get_n_used());

        ba.split_reset();
        t.assert_equal(0u, ba.get_n_used());

        llama_ubatch ub = ba.split_simple(10);
        t.assert_equal(3u, ub.n_tokens);
    });

    t.test("split_equal_unequal_lengths", [&](testing & t) {
        batch_builder bb;
        for (int i = 0; i < 4; ++i) {
            bb.add(i, {0}, i == 3);
        }
        for (int i = 0; i < 2; ++i) {
            bb.add(i, {1}, i == 1);
        }

        llama_batch_allocr ba(1);
        t.assert_true(ba.init(bb.make(), vocab, nullptr, bb.n_embd, 4, false));

        llama_ubatch ub = ba.split_equal(8, false, 0);
        t.assert_true(ub.equal_seqs());
        t.assert_equal("both seqs advance by the shorter length", 4u, ub.n_tokens);
        t.assert_equal(2u, ub.n_seq_tokens);
        t.assert_equal(2u, ub.n_seqs);
        t.assert_equal(2u, ub.n_seqs_unq);
        // tokens are grouped per sequence set: [s0 s0 s1 s1]
        t.assert_equal(0, ub.seq_id[0][0]);
        t.assert_equal(0, ub.seq_id[1][0]);
        t.assert_equal(1, ub.seq_id[2][0]);
        t.assert_equal(1, ub.seq_id[3][0]);
        t.assert_equal(0, ub.pos[0]);
        t.assert_equal(1, ub.pos[1]);
        t.assert_equal(0, ub.pos[2]);
        t.assert_equal(1, ub.pos[3]);

        ub = ba.split_equal(8, false, 0);
        t.assert_equal("only seq 0 remains", 2u, ub.n_tokens);
        t.assert_equal(1u, ub.n_seqs);
        t.assert_equal(2, ub.pos[0]);
        t.assert_equal(3, ub.pos[1]);

        ub = ba.split_equal(8, false, 0);
        t.assert_equal(0u, ub.n_tokens);

        t.assert_equal(6u, ba.get_n_used());
    });

    t.test("split_equal_coupled", [&](testing & t) {
        batch_builder bb;
        bb.add(0, {0, 1}, false);
        bb.add(1, {0, 1}, true);

        llama_batch_allocr ba(1);
        t.assert_true(ba.init(bb.make(), vocab, nullptr, bb.n_embd, 4, false));

        llama_ubatch ub = ba.split_equal(4, true, 0);
        t.assert_equal("sequential split rejects coupled seqs", 0u, ub.n_tokens);

        ub = ba.split_equal(4, false, 0);
        t.assert_equal(2u, ub.n_tokens);
        t.assert_equal("one sequence set", 1u, ub.n_seqs);
        t.assert_equal("two unique seq ids", 2u, ub.n_seqs_unq);
        t.assert_equal(2, ub.n_seq_id[0]);
        t.assert_equal(0, ub.seq_idx[0]);
        t.assert_equal(1, ub.seq_idx[1]);
    });

    t.test("split_seq_per_sequence", [&](testing & t) {
        batch_builder bb;
        for (llama_seq_id s = 0; s < 3; ++s) {
            bb.add(0, {s}, false);
            bb.add(1, {s}, true);
        }

        llama_batch_allocr ba(1);
        t.assert_true(ba.init(bb.make(), vocab, nullptr, bb.n_embd, 4, false));

        for (llama_seq_id s = 0; s < 3; ++s) {
            llama_ubatch ub = ba.split_seq(8);
            t.assert_equal(2u, ub.n_tokens);
            t.assert_equal(1u, ub.n_seqs);
            t.assert_equal(s, ub.seq_id[0][0]);
            t.assert_equal(s, ub.seq_id_unq[0]);
        }

        t.assert_equal(0u, ba.split_seq(8).n_tokens);
        t.assert_equal(6u, ba.get_n_used());
    });

    t.test("ubatch_reserve", [&](testing & t) {
        llama_batch_allocr ba(1);

        llama_ubatch ub = ba.ubatch_reserve(3, 2);
        t.assert_equal(6u, ub.n_tokens);
        t.assert_equal(3u, ub.n_seq_tokens);
        t.assert_equal(2u, ub.n_seqs);
        t.assert_equal(2u, ub.n_seqs_unq);
        t.assert_true(ub.equal_seqs());
        t.assert_equal(0, ub.seq_id_unq[0]);
        t.assert_equal(1, ub.seq_id_unq[1]);
        t.assert_true(ub.token != nullptr);
        t.assert_true(ub.embd == nullptr);
    });
}

static void test_keep_tail(testing & t) {
    llama_vocab vocab;

    // batch with n_tokens[s] tokens for each seq s, output on the last token of each seq
    auto make_batch = [](batch_builder & bb, std::initializer_list<int> n_tokens) {
        llama_seq_id s = 0;
        for (int n : n_tokens) {
            for (int i = 0; i < n; ++i) {
                bb.add(i, {s}, i == n - 1);
            }
            ++s;
        }
        return bb.make();
    };

    t.test("noop_when_seqs_complete", [&](testing & t) {
        batch_builder bb;

        llama_batch_allocr ba(1);
        t.assert_true(ba.init(make_batch(bb, {2, 2}), vocab, nullptr, bb.n_embd, 4, false));

        llama_ubatch ub = ba.split_equal(4, false, 2);
        t.assert_equal("both seqs fit whole", 4u, ub.n_tokens);
        t.assert_equal(2u, ub.n_seqs);
        t.assert_equal(2u, ub.n_seq_tokens);

        t.assert_equal(0u, ba.split_equal(4, false, 2).n_tokens);
    });

    t.test("defers_seq_with_short_remainder", [&](testing & t) {
        batch_builder bb;

        llama_batch_allocr ba(1);
        t.assert_true(ba.init(make_batch(bb, {2, 3}), vocab, nullptr, bb.n_embd, 4, false));

        // expansion stops at 2 tokens per seq: seq 0 completes, seq 1 would be left
        // with 1 < n_keep_tail remaining, so it is deferred entirely
        llama_ubatch ub = ba.split_equal(4, true, 2);
        t.assert_equal(2u, ub.n_tokens);
        t.assert_equal(1u, ub.n_seqs);
        t.assert_equal(0, ub.seq_id[0][0]);
        t.assert_equal(2u, ba.get_n_used());

        ub = ba.split_equal(4, true, 2);
        t.assert_equal("deferred seq comes back whole", 3u, ub.n_tokens);
        t.assert_equal(1u, ub.n_seqs);
        t.assert_equal(1, ub.seq_id[0][0]);
        for (int i = 0; i < 3; ++i) {
            t.assert_equal(i, ub.pos[i]);
        }

        t.assert_equal(5u, ba.get_n_used());
        t.assert_equal(0u, ba.split_equal(4, true, 2).n_tokens);
    });

    t.test("completes_first_seq_when_all_violate", [&](testing & t) {
        batch_builder bb;

        llama_batch_allocr ba(1);
        t.assert_true(ba.init(make_batch(bb, {3, 3}), vocab, nullptr, bb.n_embd, 4, false));

        // expansion stops at 2 tokens per seq, leaving both with 1 < n_keep_tail remaining;
        // seq 0 still fits in n_ubatch, so it is extended to completion and emitted alone
        llama_ubatch ub = ba.split_equal(4, false, 2);
        t.assert_equal(3u, ub.n_tokens);
        t.assert_equal(1u, ub.n_seqs);
        t.assert_equal(3u, ub.n_seq_tokens);
        t.assert_equal(0, ub.seq_id[0][0]);
        for (int i = 0; i < 3; ++i) {
            t.assert_equal(i, ub.pos[i]);
        }
        t.assert_equal(3u, ba.get_n_used());

        ub = ba.split_equal(4, false, 2);
        t.assert_equal(3u, ub.n_tokens);
        t.assert_equal(1, ub.seq_id[0][0]);
        t.assert_equal(6u, ba.get_n_used());
    });

    t.test("truncates_to_preserve_tail", [&](testing & t) {
        batch_builder bb;

        llama_batch_allocr ba(1);
        t.assert_true(ba.init(make_batch(bb, {5}), vocab, nullptr, bb.n_embd, 4, false));

        // 4 tokens would leave a remainder of 1, and the seq does not fit in n_ubatch,
        // so the ubatch is truncated until n_keep_tail tokens remain
        llama_ubatch ub = ba.split_equal(4, false, 2);
        t.assert_equal(3u, ub.n_tokens);
        t.assert_equal(1u, ub.n_seqs);
        t.assert_equal(2, ub.pos[2]);
        t.assert_equal(3u, ba.get_n_used());

        ub = ba.split_equal(4, false, 2);
        t.assert_equal("trailing tokens stay in one ubatch", 2u, ub.n_tokens);
        t.assert_equal(3, ub.pos[0]);
        t.assert_equal(4, ub.pos[1]);
        t.assert_equal(1, (int) ub.output[1]);

        t.assert_equal(5u, ba.get_n_used());
    });

    t.test("keeps_full_ubatch_with_sufficient_remainder", [&](testing & t) {
        batch_builder bb;

        llama_batch_allocr ba(1);
        t.assert_true(ba.init(make_batch(bb, {6}), vocab, nullptr, bb.n_embd, 4, false));

        llama_ubatch ub = ba.split_equal(4, false, 2);
        t.assert_equal("remainder >= n_keep_tail, no truncation", 4u, ub.n_tokens);

        ub = ba.split_equal(4, false, 2);
        t.assert_equal(2u, ub.n_tokens);
        t.assert_equal(4, ub.pos[0]);
        t.assert_equal(5, ub.pos[1]);

        t.assert_equal(6u, ba.get_n_used());
    });

    t.test("multi_seq_prefix_kept", [&](testing & t) {
        batch_builder bb;

        llama_batch_allocr ba(1);
        t.assert_true(ba.init(make_batch(bb, {3, 4}), vocab, nullptr, bb.n_embd, 6, false));

        // expansion stops at 3 tokens per seq: seq 0 completes, seq 1 has 1 < n_keep_tail
        // remaining and is deferred even though its tokens were already gathered
        llama_ubatch ub = ba.split_equal(6, true, 2);
        t.assert_equal(3u, ub.n_tokens);
        t.assert_equal(1u, ub.n_seqs);
        t.assert_equal(0, ub.seq_id[0][0]);
        t.assert_equal(3u, ba.get_n_used());

        ub = ba.split_equal(6, true, 2);
        t.assert_equal(4u, ub.n_tokens);
        t.assert_equal(1, ub.seq_id[0][0]);
        t.assert_equal(7u, ba.get_n_used());
    });

    t.test("non_sequential_keeps_every_seq_that_satisfies", [&](testing & t) {
        batch_builder bb;

        llama_batch_allocr ba(1);
        t.assert_true(ba.init(make_batch(bb, {2, 3, 2}), vocab, nullptr, bb.n_embd, 8, false));

        // expansion stops at 2 tokens per seq: seqs 0 and 2 complete, seq 1 would be left 1 < n_keep_tail. With no
        // order of seq ids to preserve, seq 2 is not deferred with it
        llama_ubatch ub = ba.split_equal(8, false, 2);
        t.assert_equal(4u, ub.n_tokens);
        t.assert_equal(2u, ub.n_seqs);
        t.assert_equal(0, ub.seq_id[0][0]);
        t.assert_equal(2, ub.seq_id[2][0]);

        ub = ba.split_equal(8, false, 2);
        t.assert_equal("the deferred seq comes back whole", 3u, ub.n_tokens);
        t.assert_equal(1, ub.seq_id[0][0]);

        t.assert_equal(7u, ba.get_n_used());
        t.assert_equal(0u, ba.split_equal(8, false, 2).n_tokens);
    });

    t.test("completes_seqs_with_the_same_remainder_together", [&](testing & t) {
        batch_builder bb;

        llama_batch_allocr ba(1);
        t.assert_true(ba.init(make_batch(bb, {3, 3, 3}), vocab, nullptr, bb.n_embd, 6, false));

        // expansion stops at 2 tokens per seq, leaving each 1 < n_keep_tail: seq 0 is completed, and seq 1 with it,
        // which ends in the same rows; a third would pass n_ubatch
        llama_ubatch ub = ba.split_equal(6, false, 2);
        t.assert_equal(6u, ub.n_tokens);
        t.assert_equal(2u, ub.n_seqs);
        t.assert_equal(3u, ub.n_seq_tokens);
        t.assert_equal(0, ub.seq_id[0][0]);
        t.assert_equal(1, ub.seq_id[3][0]);

        ub = ba.split_equal(6, false, 2);
        t.assert_equal(3u, ub.n_tokens);
        t.assert_equal(2, ub.seq_id[0][0]);

        t.assert_equal(9u, ba.get_n_used());
    });

    t.test("a_prompt_step_shared_unevenly_takes_a_ubatch_a_length", [&](testing & t) {
        batch_builder bb;

        // a step of 1,024 prompt rows shared among 22 seqs as the server shares it, 47 to the first twelve and 46 to
        // the rest, beside one seq that replies with 3 rows (four RTX PRO 6000, Nov 11, 2026: 14 ubatches)
        llama_batch_allocr ba(1);
        t.assert_true(ba.init(make_batch(bb, {47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47,
                                              46, 46, 46, 46, 3, 46, 46, 46, 46, 46, 46}),
                              vocab, nullptr, bb.n_embd, 32, false));

        llama_ubatch ub = ba.split_equal(2048, false, 3);
        t.assert_equal("every seq's first rows, at the reply's length", 69u, ub.n_tokens);
        t.assert_equal(23u, ub.n_seqs);

        ub = ba.split_equal(2048, false, 3);
        t.assert_equal("the ten seqs that end together", 430u, ub.n_tokens);
        t.assert_equal(10u, ub.n_seqs);
        t.assert_equal(12, ub.seq_id[0][0]);

        ub = ba.split_equal(2048, false, 3);
        t.assert_equal("the twelve one row longer", 528u, ub.n_tokens);
        t.assert_equal(12u, ub.n_seqs);
        t.assert_equal(0, ub.seq_id[0][0]);

        t.assert_equal(1027u, ba.get_n_used());
        t.assert_equal(0u, ba.split_equal(2048, false, 3).n_tokens);
    });
}

static void test_mrope(testing & t) {
    llama_vocab vocab;

    t.test("pos_layout_and_split", [&](testing & t) {
        const uint32_t n_pos = 4;
        const uint32_t n_embd = 2;

        batch_builder bb(n_embd);
        bb.add(10, {0}, false);
        bb.add(11, {0}, true);

        // M-RoPE positions for embeddings are laid out [n_pos][n_tokens]
        std::vector<llama_pos> pos = {
            10, 11, // temporal
             5,  6, // y
             7,  8, // x
             0,  0,
        };

        llama_batch batch = bb.make(false, true, true);
        batch.pos = pos.data();

        llama_batch_allocr ba(n_pos);
        t.assert_true(ba.init(batch, vocab, nullptr, n_embd, 4, false));

        llama_ubatch ub = ba.split_simple(2);
        t.assert_equal(2u, ub.n_tokens);
        t.assert_equal(n_pos, ub.n_pos);
        t.assert_true(ub.is_pos_2d());

        const llama_pos expected[8] = {10, 11, 5, 6, 7, 8, 0, 0};
        for (int i = 0; i < 8; ++i) {
            t.assert_equal(expected[i], ub.pos[i]);
        }
    });

    t.test("pos_jump_allowed", [&](testing & t) {
        const uint32_t n_pos = 4;
        const uint32_t n_embd = 2;

        mock_memory mem;
        mem.ranges[0] = {0, 9};

        llama_batch_allocr ba(n_pos);

        auto try_pos = [&](llama_pos p0) {
            batch_builder bb(n_embd);
            bb.add(p0, {0}, true);

            std::vector<llama_pos> pos = {p0, 1, 1, 0};

            llama_batch batch = bb.make(false, true, true);
            batch.pos = pos.data();

            return ba.init(batch, vocab, &mem, n_embd, 4, false);
        };

        t.assert_true("gap after memory is allowed",     try_pos(15));
        t.assert_true("overlap is allowed for embd",     try_pos(9));
        t.assert_true("pos behind memory is rejected",  !try_pos(8));
    });
}

static void test_device_order(testing & t) {
    llama_vocab vocab;
    const llama_seq_id input[] = {7, 0, 5, 2, 4, 1, 6, 3};
    const llama_seq_id ordered[] = {0, 4, 1, 5, 2, 6, 3, 7};
    const int32_t original[] = {1, 4, 5, 2, 3, 6, 7, 0};
    t.test("device_counts_include_empty_owners", [&](testing & t) {
        const auto capacity = llama_stream_counts(10, 4);
        const int64_t expected_capacity[] = {3, 3, 2, 2};
        for (size_t d = 0; d < 4; ++d) { t.assert_equal(expected_capacity[d], capacity[d]); }
        batch_builder bb;
        bb.add(0, {6}, true);
        bb.add(0, {0}, false);
        bb.add(1, {6}, false);
        bb.add(0, {2}, true);
        llama_batch_allocr ba(1, 4);
        t.assert_true(ba.init(bb.make(), vocab, nullptr, bb.n_embd, 8, false));
        const auto ub = ba.split_simple(4);
        const auto rows = ub.device_rows(4);
        const auto outputs = ub.device_rows(4, true);
        const auto seqs = ub.device_seqs(4);
        const int64_t expected_rows[] = {1, 0, 3, 0};
        const int64_t expected_outputs[] = {0, 0, 2, 0};
        const int64_t expected_seqs[] = {1, 0, 2, 0};
        for (size_t d = 0; d < 4; ++d) {
            t.assert_equal(expected_rows[d], rows[d]);
            t.assert_equal(expected_outputs[d], outputs[d]);
            t.assert_equal(expected_seqs[d], seqs[d]);
        }
    });
    t.test("rows_and_output_mapping", [&](testing & t) {
        batch_builder bb;
        for (int pos = 0; pos < 3; ++pos) {
            for (llama_seq_id seq : input) { bb.add(pos, {seq}, pos != 0); }
        }
        llama_batch_allocr ba(1, 4);
        t.assert_true(ba.init(bb.make(), vocab, nullptr, bb.n_embd, 8, false));
        for (int pos = 0; pos < 3; ++pos) {
            const auto ub = ba.split_equal(8, false, 0);
            t.assert_equal(8u, ub.n_tokens);
            for (int row = 0; row < 8; ++row) {
                const int idx = 8*pos + original[row];
                t.assert_equal(ordered[row], ub.seq_id[row][0]);
                t.assert_equal(ordered[row], ub.seq_id_unq[row]);
                t.assert_equal(row, ub.seq_idx[ordered[row]]);
                t.assert_equal(idx, ub.idx[row]);
                t.assert_equal(pos, ub.pos[row]);
                t.assert_equal(100.0f*idx, ub.embd[row*bb.n_embd]);
                if (pos != 0) { t.assert_equal(idx, ba.get_out_ids()[8*(pos - 1) + row]); }
            }
        }
        t.assert_equal(0u, ba.split_equal(8, false, 0).n_tokens);
    });
    t.test("reserve_uses_the_same_owners", [&](testing & t) {
        llama_batch_allocr ba(1, 4);
        const auto ub = ba.ubatch_reserve(2, 8);
        for (int row = 0; row < 8; ++row) {
            t.assert_equal(ordered[row], ub.seq_id_unq[row]);
            t.assert_equal(row, ub.seq_idx[ordered[row]]);
        }
    });
    t.test("prompt_rows_keep_sequence_order", [&](testing & t) {
        batch_builder bb;
        for (int pos = 0; pos < 2; ++pos) {
            for (llama_seq_id seq : input) { bb.add(pos, {seq}, true); }
        }
        llama_batch_allocr ba(1, 3);
        t.assert_true(ba.init(bb.make(), vocab, nullptr, bb.n_embd, 8, false));
        const auto ub = ba.split_simple(16);
        const llama_seq_id expected[] = {0, 3, 6, 1, 4, 7, 2, 5};
        t.assert_equal(16u, ub.n_tokens);
        for (int row = 0; row < 16; ++row) {
            t.assert_equal(expected[row/2], ub.seq_id[row][0]);
            t.assert_equal(row % 2, ub.pos[row]);
            t.assert_equal(input[ub.idx[row] % 8], ub.seq_id[row][0]);
            t.assert_equal(ub.idx[row], ba.get_out_ids()[row]);
            t.assert_equal(100.0f*ub.idx[row], ub.embd[row*bb.n_embd]);
        }
    });
    t.test("sequential_streams_follow_device_order", [&](testing & t) {
        batch_builder bb;
        for (llama_seq_id seq : input) { bb.add(0, {seq}, true); }
        llama_batch_allocr ba(1, 4);
        t.assert_true(ba.init(bb.make(), vocab, nullptr, bb.n_embd, 8, false));
        for (int batch = 0; batch < 2; ++batch) {
            const auto ub = ba.split_equal(4, true, 0);
            t.assert_equal(4u, ub.n_tokens);
            for (uint32_t row = 0; row < ub.n_tokens && row < 4; ++row) {
                t.assert_equal(ordered[4*batch + row], ub.seq_id[row][0]);
                t.assert_equal(original[4*batch + row], ub.idx[row]);
            }
        }
        t.assert_equal(0u, ba.split_equal(4, true, 0).n_tokens);
    });
    t.test("sequence_count_obeys_ubatch_limit", [&](testing & t) {
        batch_builder bb;
        for (llama_seq_id seq = 0; seq < 8; ++seq) { bb.add(0, {seq}, true); }
        llama_batch_allocr ba(1);
        t.assert_true(ba.init(bb.make(), vocab, nullptr, bb.n_embd, 8, false));
        t.assert_equal(4u, ba.split_equal(4, true, 0).n_tokens);
        t.assert_equal(4u, ba.split_equal(4, true, 0).n_tokens);
        t.assert_equal(0u, ba.split_equal(4, true, 0).n_tokens);
    });
    t.test("partly_occupied_devices_share_a_ubatch", [&](testing & t) {
        batch_builder bb;
        for (llama_seq_id seq = 0; seq < 12; ++seq) { bb.add(0, {seq}, true); }
        llama_batch_allocr ba(1, 4);
        t.assert_true(ba.init(bb.make(), vocab, nullptr, bb.n_embd, 40, false));
        const auto ub = ba.split_equal(40, true, 0);
        t.assert_equal(12u, ub.n_tokens);
        for (auto count : ub.device_seqs(4)) { t.assert_equal(int64_t(3), count); }
    });
    t.test("local_stream_gap_defers_only_that_device", [&](testing & t) {
        batch_builder bb;
        for (llama_seq_id seq : {0, 8, 1, 5, 2, 3}) { bb.add(0, {seq}, true); }
        llama_batch_allocr ba(1, 4);
        t.assert_true(ba.init(bb.make(), vocab, nullptr, bb.n_embd, 40, false));
        const auto ub = ba.split_equal(40, true, 0);
        t.assert_equal(5u, ub.n_tokens);
        const llama_seq_id expected[] = {0, 1, 5, 2, 3};
        for (uint32_t i = 0; i < ub.n_tokens && i < 5; ++i) { t.assert_equal(expected[i], ub.seq_id_unq[i]); }
        const auto next = ba.split_equal(40, true, 0);
        t.assert_equal(1u, next.n_tokens);
        if (next.n_tokens) { t.assert_equal(llama_seq_id(8), next.seq_id_unq[0]); }
    });
}

#ifndef _WIN32
// LLAMA_PROMPT_SHAPES=1 (read when the allocator is made): split_equal gives a sequence's prompt rows a ubatch of their own
static void test_prompt_shapes(testing & t) {
    llama_vocab vocab;

    // n_tokens[s] rows for each seq s: every row an output where reply[s], else only the last one
    auto make_batch = [](batch_builder & bb, std::initializer_list<int> n_tokens, std::initializer_list<bool> reply) {
        llama_seq_id s = 0;
        auto is_reply = reply.begin();
        for (int n : n_tokens) {
            for (int i = 0; i < n; ++i) {
                bb.add(i, {s}, *is_reply || i == n - 1);
            }
            ++s;
            ++is_reply;
        }
        return bb.make();
    };

    t.test("without_the_switch_a_prompt_is_cut_at_the_replies", [&](testing & t) {
        unsetenv("LLAMA_PROMPT_SHAPES");
        batch_builder bb;

        llama_batch_allocr ba(1);
        t.assert_true(ba.init(make_batch(bb, {3, 8, 3}, {true, false, true}), vocab, nullptr, bb.n_embd, 4, false));

        llama_ubatch ub = ba.split_equal(64, false, 0);
        t.assert_equal(9u, ub.n_tokens);
        t.assert_equal(3u, ub.n_seqs);

        ub = ba.split_equal(64, false, 0);
        t.assert_equal(5u, ub.n_tokens);
        t.assert_equal(1u, ub.n_seqs);
    });

    setenv("LLAMA_PROMPT_SHAPES", "1", 1);

    t.test("replies_together_prompt_alone", [&](testing & t) {
        batch_builder bb;

        llama_batch_allocr ba(1);
        t.assert_true(ba.init(make_batch(bb, {3, 8, 3}, {true, false, true}), vocab, nullptr, bb.n_embd, 4, false));

        llama_ubatch ub = ba.split_equal(64, false, 0);
        t.assert_equal("the replies' rows", 6u, ub.n_tokens);
        t.assert_equal(2u, ub.n_seqs);
        t.assert_equal(3u, ub.n_seq_tokens);
        t.assert_equal(0, ub.seq_id[0][0]);
        t.assert_equal(2, ub.seq_id[3][0]);
        for (uint32_t i = 0; i < ub.n_tokens; ++i) {
            t.assert_equal(1, (int) ub.output[i]);
        }

        ub = ba.split_equal(64, false, 0);
        t.assert_equal("the prompt's rows, whole", 8u, ub.n_tokens);
        t.assert_equal(1u, ub.n_seqs);
        t.assert_equal(1, ub.seq_id[0][0]);
        for (int i = 0; i < 8; ++i) {
            t.assert_equal(i, ub.pos[i]);
            t.assert_equal(i == 7 ? 1 : 0, (int) ub.output[i]);
        }

        t.assert_equal(14u, ba.get_n_used());
        t.assert_equal(0u, ba.split_equal(64, false, 0).n_tokens);
    });

    t.test("prompts_first_each_alone", [&](testing & t) {
        batch_builder bb;

        llama_batch_allocr ba(1);
        t.assert_true(ba.init(make_batch(bb, {4, 3, 4, 3}, {false, true, false, true}), vocab, nullptr, bb.n_embd, 4, false));

        const uint32_t    n_tokens[] = {4, 6, 4};
        const uint32_t    n_seqs[]   = {1, 2, 1};
        const llama_seq_id first[]   = {0, 1, 2};
        for (int u = 0; u < 3; ++u) {
            const llama_ubatch ub = ba.split_equal(64, false, 0);
            t.assert_equal(n_tokens[u], ub.n_tokens);
            t.assert_equal(n_seqs[u], ub.n_seqs);
            t.assert_equal(first[u], ub.seq_id[0][0]);
        }

        t.assert_equal(14u, ba.get_n_used());
        t.assert_equal(0u, ba.split_equal(64, false, 0).n_tokens);
    });

    t.test("a_prompt_longer_than_the_ubatch_keeps_its_tail", [&](testing & t) {
        batch_builder bb;

        llama_batch_allocr ba(1);
        t.assert_true(ba.init(make_batch(bb, {3, 5}, {true, false}), vocab, nullptr, bb.n_embd, 4, false));

        llama_ubatch ub = ba.split_equal(4, false, 2);
        t.assert_equal(3u, ub.n_tokens);
        t.assert_equal(0, ub.seq_id[0][0]);

        // what is left over a whole ubatch goes first, so the rows kept together are in the last one
        ub = ba.split_equal(4, false, 2);
        t.assert_equal(1u, ub.n_tokens);
        t.assert_equal(1, ub.seq_id[0][0]);

        ub = ba.split_equal(4, false, 2);
        t.assert_equal(4u, ub.n_tokens);
        t.assert_equal(1, ub.pos[0]);
        t.assert_equal(1, (int) ub.output[3]);

        t.assert_equal(8u, ba.get_n_used());
    });

    t.test("a_long_prompt_goes_in_a_few_sizes", [&](testing & t) {
        batch_builder bb;

        llama_batch_allocr ba(1);
        t.assert_true(ba.init(make_batch(bb, {3, 97, 3, 1624}, {true, false, true, false}), vocab, nullptr, bb.n_embd, 4, false));

        // the replies, then each prompt: of what is left over whole ubatches, what leaves a multiple of 32, then the powers
        // of two up to 256; then the whole ubatches (1,624 = 24 + 64 + 256 + 256 + 1,024)
        const uint32_t     n_tokens[] = {6, 1, 32, 64, 24, 64, 256, 256, 1024};
        const llama_seq_id first[]    = {0, 1, 1,  1,  3,  3,  3,   3,   3};
        llama_pos pos[4] = {0, 0, 0, 0};
        for (int u = 0; u < 9; ++u) {
            const llama_ubatch ub = ba.split_equal(1024, false, 3);
            t.assert_equal(n_tokens[u], ub.n_tokens);
            t.assert_equal(u == 0 ? 2u : 1u, ub.n_seqs);
            t.assert_equal(first[u], ub.seq_id[0][0]);
            if (u > 0) {
                const llama_seq_id seq = first[u];
                for (uint32_t i = 0; i < ub.n_tokens; ++i) {
                    t.assert_equal(pos[seq]++, ub.pos[i]);
                }
            }
        }

        t.assert_equal(1727u, ba.get_n_used());
        t.assert_equal(0u, ba.split_equal(1024, false, 3).n_tokens);
    });

    t.test("more_prompts_than_the_bound_are_read_together", [&](testing & t) {
        batch_builder bb;

        llama_batch_allocr ba(1);
        t.assert_true(ba.init(make_batch(bb, {40, 3, 40, 40}, {false, true, false, false}), vocab, nullptr, bb.n_embd, 4, false));

        // three prompts where two are read alone: together, and apart from the reply
        llama_ubatch ub = ba.split_equal(128, false, 3);
        t.assert_equal(120u, ub.n_tokens);
        t.assert_equal(3u, ub.n_seqs);
        t.assert_equal(40u, ub.n_seq_tokens);
        t.assert_equal(0, ub.seq_id[0][0]);
        t.assert_equal(2, ub.seq_id[40][0]);
        t.assert_equal(3, ub.seq_id[80][0]);

        ub = ba.split_equal(128, false, 3);
        t.assert_equal(3u, ub.n_tokens);
        t.assert_equal(1, ub.seq_id[0][0]);

        t.assert_equal(123u, ba.get_n_used());
        t.assert_equal(0u, ba.split_equal(128, false, 3).n_tokens);
    });

    unsetenv("LLAMA_PROMPT_SHAPES");
}
#endif

int main(int argc, char ** argv) {
    testing t;

    const char * verbose = getenv("LLAMA_TEST_VERBOSE");
    if (verbose) {
        t.verbose = std::string(verbose) == "1";
    }
    if (!t.verbose) {
        llama_log_set([](ggml_log_level, const char *, void *) {}, nullptr);
    }

    if (argc > 1) {
        t.set_filter(argv[1]);
    }

    t.test("init",      test_init);
    t.test("split",     test_split);
    t.test("keep_tail", test_keep_tail);
    t.test("mrope",     test_mrope);
    t.test("device_order", test_device_order);
#ifndef _WIN32
    t.test("prompt_shapes", test_prompt_shapes);
#endif

    return t.summary();
}
