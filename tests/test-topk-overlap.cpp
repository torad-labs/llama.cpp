// What a tile of adjacent prompt rows gathers from the KV cache when the rows share one gather of the union of their
// indexer selections, as a multiple of what one row gathers alone (examples/topk-overlap). Query-tiled sparse attention
// is worth building only if that multiple stays well under the tile's size on the trained weights.
// And what a step selects that a hot cache of the pools selected lately does not hold (topk_lru), which row-level
// tiering has to bring from host memory.

#include "../examples/topk-overlap/topk-overlap.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <random>
#include <set>
#include <vector>

static int failures = 0;

#define CHECK(cond)                                                                    \
    do {                                                                               \
        if (!(cond)) {                                                                 \
            std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond); \
            failures++;                                                                \
        }                                                                              \
    } while (0)

static bool near(double a, double b) { return std::fabs(a - b) < 1e-9; }

// rows of pool ids, row-major
static std::vector<int32_t> rows(const std::vector<std::vector<int32_t>> & r) {
    std::vector<int32_t> out;
    for (const auto & row : r) out.insert(out.end(), row.begin(), row.end());
    return out;
}

static void test_examples() {
    // every row picks the same pools: a tile gathers what one row does
    {
        std::vector<std::vector<int32_t>> r(8, {0, 1, 2, 3, 4, 5, 6, 7});
        const auto sel = rows(r);
        const auto st = topk_tile_union(sel.data(), 8, 8, 100, 4);
        CHECK(st.tiles == 2);
        CHECK(near(st.mean(), 1.0));
    }
    // no two rows share a pool: a tile gathers its size in rows' worth
    {
        std::vector<std::vector<int32_t>> r;
        for (int i = 0; i < 8; ++i) {
            std::vector<int32_t> row;
            for (int j = 0; j < 8; ++j) row.push_back(8*i + j);
            r.push_back(row);
        }
        const auto sel = rows(r);
        CHECK(near(topk_tile_union(sel.data(), 8, 8, 1000, 4).mean(), 4.0));
        CHECK(near(topk_tile_union(sel.data(), 8, 8, 1000, 8).mean(), 8.0));
    }
    // two rows sharing half their pools gather one and a half rows' worth
    {
        const auto sel = rows({{0, 1, 2, 3, 4, 5, 6, 7}, {4, 5, 6, 7, 8, 9, 10, 11}});
        CHECK(near(topk_tile_union(sel.data(), 8, 2, 100, 2).mean(), 1.5));
    }
    // dump pools, ids at or past n_live, are no KV the tile gathers: rows with four live pools each, the same four
    {
        const auto sel = rows({{0, 1, 2, 3, 100, 101, 102, 103}, {3, 2, 1, 0, 104, 105, 106, 107}});
        const auto st = topk_tile_union(sel.data(), 8, 2, 100, 2);
        CHECK(st.tiles == 1);
        CHECK(near(st.mean(), 1.0));
    }
    // a tile holding a row with no live pool is left out, and its rows are counted as skipped
    {
        const auto sel = rows({{0, 1}, {1, 2}, {100, 101}, {0, 1}});
        const auto st = topk_tile_union(sel.data(), 2, 4, 100, 2);
        CHECK(st.tiles == 1);
        CHECK(st.rows_skipped == 2);
        CHECK(near(st.mean(), 1.5));
    }
    // rows past the last whole tile are left out
    {
        std::vector<std::vector<int32_t>> r(10, {0, 1, 2, 3});
        const auto sel = rows(r);
        const auto st = topk_tile_union(sel.data(), 4, 10, 100, 4);
        CHECK(st.tiles == 2);
        CHECK(st.rows_skipped == 2);
    }
    // a tile of one row gathers exactly that row
    {
        const auto sel = rows({{5, 9, 2}, {1, 7, 3}, {0, 4, 6}});
        const auto st = topk_tile_union(sel.data(), 3, 3, 100, 1);
        CHECK(st.tiles == 3);
        CHECK(near(st.mean(), 1.0));
    }
}

// the tile's gather counted the plain way: a set of every live id in the tile, against the mean live count of its rows
static double model_ratio(const std::vector<int32_t> & sel, int64_t k, int64_t first, int64_t tile, int64_t n_live, bool & counted) {
    std::set<int32_t> uni;
    double live = 0;
    counted = true;
    for (int64_t r = first; r < first + tile; ++r) {
        std::set<int32_t> row;
        for (int64_t j = 0; j < k; ++j) {
            const int32_t id = sel[r*k + j];
            if (id < n_live) row.insert(id);
        }
        if (row.empty()) counted = false;
        live += row.size();
        uni.insert(row.begin(), row.end());
    }
    return counted ? uni.size() / (live / tile) : 0.0;
}

static void test_generated() {
    std::mt19937 rng(20261101);
    for (int iter = 0; iter < 400; ++iter) {
        const int64_t k       = 1 + rng() % 24;
        const int64_t n_live  = 1 + rng() % 64;
        const int64_t n_rows  = 1 + rng() % 70;
        const bool    full    = rng() % 2 == 0 && n_live >= k;
        // each row picks k distinct ids from the live pools and the k dump pools after them, as the top-k over
        // [live scores, dump pools] does; a full run picks live pools only
        std::vector<int32_t> sel;
        for (int64_t r = 0; r < n_rows; ++r) {
            std::vector<int32_t> ids;
            const int64_t span = full ? n_live : n_live + k;
            for (int64_t i = 0; i < span; ++i) ids.push_back((int32_t) i);
            std::shuffle(ids.begin(), ids.end(), rng);
            sel.insert(sel.end(), ids.begin(), ids.begin() + k);
        }
        double prev_mean = 0.0;
        for (int64_t tile : {1, 2, 4, 8, 16}) {
            const auto st = topk_tile_union(sel.data(), k, n_rows, n_live, tile);
            int64_t tiles = 0, skipped = 0;
            double sum = 0.0;
            for (int64_t first = 0; first + tile <= n_rows; first += tile) {
                bool counted;
                const double ratio = model_ratio(sel, k, first, tile, n_live, counted);
                if (!counted) {
                    skipped += tile;
                    continue;
                }
                tiles++;
                sum += ratio;
                // a tile gathers at least one row's worth and at most every row's
                CHECK(ratio >= 1.0 - 1e-9);
                CHECK(ratio <= tile + 1e-9);
            }
            skipped += n_rows % tile;
            CHECK(st.tiles == tiles);
            CHECK(st.rows_skipped == skipped);
            CHECK(std::fabs(st.sum_ratio - sum) < 1e-6);
            // with every row full and the rows filling whole tiles at both sizes, a tile twice as wide gathers at least
            // the mean of its two halves, so the mean multiple never falls as the tile grows
            if (full && n_rows % 16 == 0 && tile > 1) CHECK(st.mean() >= prev_mean - 1e-9);
            prev_mean = st.mean();
        }
    }
}

static void test_lru_examples() {
    // a cache of one selection: the pools of the step before are held, the ones before that are gone
    {
        topk_lru lru(4);
        CHECK(lru.step(rows({{0, 1, 2, 3}}).data(), 4, 1, 100) == 4);
        CHECK(lru.step(rows({{3, 2, 1, 0}}).data(), 4, 1, 100) == 0);
        CHECK(lru.step(rows({{2, 3, 4, 5}}).data(), 4, 1, 100) == 2);
        CHECK(lru.step(rows({{0, 1, 2, 3}}).data(), 4, 1, 100) == 2);
        CHECK(lru.steps == 4 && lru.selected == 16 && lru.misses == 8);
        CHECK(lru.n_held() == 4);
    }
    // a cache of two selections still holds them
    {
        topk_lru lru(8);
        lru.step(rows({{0, 1, 2, 3}}).data(), 4, 1, 100);
        lru.step(rows({{2, 3, 4, 5}}).data(), 4, 1, 100);
        CHECK(lru.step(rows({{0, 1, 2, 3}}).data(), 4, 1, 100) == 0);
        CHECK(lru.n_held() == 6);
    }
    // the pool selected longest ago leaves first, and a pool selected again is selected last
    {
        topk_lru lru(3);
        for (const int32_t id : {0, 1, 2, 0, 3}) lru.step(&id, 1, 1, 100);
        CHECK(lru.held(0) && lru.held(2) && lru.held(3) && !lru.held(1));
        const int32_t one = 1, zero = 0;
        CHECK(lru.step(&one, 1, 1, 100) == 1);
        CHECK(!lru.held(2));
        CHECK(lru.step(&zero, 1, 1, 100) == 0);
    }
    // a pool is born on the card: selecting it is no miss
    {
        topk_lru lru(4);
        lru.born(0, 2);
        CHECK(lru.step(rows({{0, 1, 2, 3}}).data(), 4, 1, 100) == 2);
        // and no more are born than the cache holds: of eight, the last four
        lru.born(4, 12);
        CHECK(lru.n_held() == 4);
        CHECK(lru.step(rows({{8, 9, 10, 11}}).data(), 4, 1, 100) == 0);
    }
    // a row with a dump pool had no choice: the step counts nothing, and the pools it selected are held after it
    {
        topk_lru lru(4);
        CHECK(lru.step(rows({{0, 1, 100, 101}}).data(), 4, 1, 100) == -1);
        CHECK(lru.steps == 0 && lru.selected == 0 && lru.misses == 0);
        CHECK(lru.step(rows({{0, 1, 2, 3}}).data(), 4, 1, 100) == 2);
    }
    // a step of several rows selects their union once, holds all of it while it runs, and keeps the cache's size of it
    {
        topk_lru lru(4);
        CHECK(lru.step(rows({{0, 1, 2, 3}, {2, 3, 4, 5}}).data(), 4, 2, 100) == 6);
        CHECK(lru.selected == 6);
        CHECK(lru.n_held() == 4);
    }
}

// the same cache kept the plain way: the step each held pool was last selected at, the oldest erased while over cap
static void test_lru_generated() {
    std::mt19937 rng(20261113);
    for (int iter = 0; iter < 300; ++iter) {
        const int64_t k      = 1 + rng() % 12;
        const int64_t n_live = k + rng() % 80;
        const int64_t cap    = 1 + rng() % (4*k);
        const int64_t r      = 1 + rng() % 3;

        topk_lru lru(cap);
        std::map<int32_t, int64_t> last; // pool -> the stamp of its last selection
        int64_t stamp = 0, steps = 0, selected = 0, misses = 0, born = 0;
        const auto trim = [&]() {
            while ((int64_t) last.size() > cap) {
                auto old = last.begin();
                for (auto it = last.begin(); it != last.end(); ++it) {
                    if (it->second < old->second) old = it;
                }
                last.erase(old);
            }
        };

        for (int step = 0; step < 60; ++step) {
            const int64_t n_rows = 1 + rng() % 3;
            // some pools are born between steps
            const int64_t born_to = std::min<int64_t>(n_live, born + rng() % (r + 1));
            lru.born(born, born_to);
            for (int64_t id = born; id < born_to; ++id) last[(int32_t) id] = stamp++;
            trim();
            born = born_to;

            // rows drawn near a centre that drifts, so steps repeat part of each other; now and then a dump pool
            const int64_t centre = rng() % n_live;
            std::vector<int32_t> sel;
            for (int64_t row = 0; row < n_rows; ++row) {
                std::set<int32_t> ids;
                while ((int64_t) ids.size() < k) {
                    const bool dump = rng() % 40 == 0;
                    ids.insert(dump ? (int32_t) (n_live + rng() % k) : (int32_t) ((centre + rng() % (2*k)) % n_live));
                }
                sel.insert(sel.end(), ids.begin(), ids.end());
            }

            std::set<int32_t> uni;
            bool full = true;
            for (const int32_t id : sel) {
                if (id < n_live) uni.insert(id); else full = false;
            }
            int64_t miss = 0;
            for (const int32_t id : uni) miss += last.count(id) ? 0 : 1;
            for (const int32_t id : uni) last[id] = stamp++;
            trim();

            const int64_t got = lru.step(sel.data(), k, n_rows, n_live);
            if (full) {
                steps++;
                selected += (int64_t) uni.size();
                misses   += miss;
                CHECK(got == miss);
            } else {
                CHECK(got == -1);
            }
            CHECK(lru.n_held() == (int64_t) last.size());
            for (const auto & [id, at] : last) CHECK(lru.held(id));
        }
        CHECK(lru.steps == steps);
        CHECK(lru.selected == selected);
        CHECK(lru.misses == misses);
    }
}

// the selection node by its exact name, never its reshaped view: ggml names the cell gather's input
// "indexer_top_k_pools-<il> (reshaped)", the same selection as one row, and reading it too doubles every step
static void test_selection_name() {
    CHECK(topk_selection_layer("indexer_top_k_pools-7") == 7);
    CHECK(topk_selection_layer("indexer_top_k_pools-12") == 12);
    CHECK(topk_selection_layer("indexer_top_k_pools-7 (reshaped)") == -1);
    CHECK(topk_selection_layer("indexer_top_k_pools-") == -1);
    CHECK(topk_selection_layer("indexer_top_k_pools") == -1);
    CHECK(topk_selection_layer("other-7") == -1);
}

int main() {
    test_selection_name();
    test_examples();
    test_generated();
    test_lru_examples();
    test_lru_generated();
    if (failures) {
        std::fprintf(stderr, "test-topk-overlap: %d check(s) failed\n", failures);
        return 1;
    }
    std::printf("test-topk-overlap: ok\n");
    return 0;
}
