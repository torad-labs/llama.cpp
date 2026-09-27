#include "testing.h"

#include "ngram-mod.h"

#include <vector>

using entry_t = common_ngram_mod::entry_t;

// n-grams of 3 tokens followed by the token they continue with: { t0, t1, t2, next }
static const std::vector<entry_t> a = { 11, 12, 13, 100 };
static const std::vector<entry_t> b = { 21, 22, 23, 200 };

// With one slot every n-gram hashes to it, so each case below is a collision.
static void test_one_slot(testing & t) {
    t.test("a slot another n-gram holds is not found", [](testing & t) {
        common_ngram_mod mod(3, 1);
        mod.add(a.data());
        t.assert_equal("the n-gram that filled it", a[3], mod.get(a.data()));
        t.assert_equal("another n-gram", common_ngram_mod::EMPTY, mod.get(b.data()));
    });

    t.test("a colliding n-gram takes the slot over", [](testing & t) {
        common_ngram_mod mod(3, 1);
        mod.add(a.data());
        mod.add(b.data());
        t.assert_equal("the n-gram written last", b[3], mod.get(b.data()));
        t.assert_equal("the n-gram it replaced", common_ngram_mod::EMPTY, mod.get(a.data()));
        t.assert_equal("slots in use", (size_t) 1, mod.get_used());
    });

    t.test("an n-gram written again keeps its last continuation", [](testing & t) {
        common_ngram_mod mod(3, 1);
        mod.add(a.data());
        const std::vector<entry_t> a2 = { a[0], a[1], a[2], 101 };
        mod.add(a2.data());
        t.assert_equal(a2[3], mod.get(a.data()));
    });

    t.test("reset empties the table", [](testing & t) {
        common_ngram_mod mod(3, 1);
        mod.add(a.data());
        mod.reset();
        t.assert_equal(common_ngram_mod::EMPTY, mod.get(a.data()));
        t.assert_equal((size_t) 0, mod.get_used());
    });
}

// A table sized as the drafter's (4M slots) and filled to its 25% reset threshold with one context's n-grams: an
// n-gram it never saw is found nowhere, where a table that keeps no key finds a quarter of them
static void test_filled(testing & t) {
    t.test("a filled table finds no n-gram it never saw", [](testing & t) {
        constexpr size_t n_slots = 4*1024*1024;
        common_ngram_mod mod(16, n_slots);

        uint64_t x = 1;
        const auto next = [&]() { x = x*6364136223846793005ULL + 1442695040888963407ULL; return (entry_t) ((x >> 33) % 248320); };

        std::vector<entry_t> ctx(n_slots/4 + 17);
        for (auto & tok : ctx) {
            tok = next();
        }
        for (size_t i = 0; i + 17 <= ctx.size(); ++i) {
            mod.add(ctx.data() + i);
        }
        t.assert_true("filled past a fifth", mod.get_used() > n_slots/5);

        size_t n_found = 0;
        std::vector<entry_t> probe(17);
        for (int k = 0; k < 100000; ++k) {
            for (auto & tok : probe) {
                tok = next();
            }
            n_found += mod.get(probe.data()) != common_ngram_mod::EMPTY;
        }
        t.assert_equal("unseen n-grams found", (size_t) 0, n_found);
    });
}

int main() {
    testing t;

    t.test("one_slot", test_one_slot);
    t.test("filled",   test_filled);

    return t.summary();
}
