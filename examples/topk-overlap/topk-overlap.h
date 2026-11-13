#pragma once

// How much adjacent prompt rows' indexer selections overlap. Query-tiled sparse attention gathers, for a tile of
// adjacent rows, the union of the cells each row selected, once, with a mask per row. What that tile loads, as a
// multiple of what one row loads alone, is the tile's union of selected pools over the mean number its rows selected:
// 1 when every row picked the same pools, the tile's size when no two rows share one. The cells of a pool are its own,
// so the multiple is the same counted in pools or in cells.

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

// the layer of a selection node, "indexer_top_k_pools-<il>" and nothing after it, -1 for any other node: the cell
// gather's input is that node reshaped to one row, which ggml_reshape_2d names "indexer_top_k_pools-<il> (reshaped)",
// the same selection seen a second time
inline int topk_selection_layer(const char * name) {
    static const char * prefix = "indexer_top_k_pools-";
    const size_t n = std::strlen(prefix);
    if (std::strncmp(name, prefix, n) != 0) return -1;
    char * end = nullptr;
    const long il = std::strtol(name + n, &end, 10);
    return end != name + n && *end == '\0' ? (int) il : -1;
}

struct topk_tile_stats {
    int64_t tiles        = 0;   // tiles counted
    double  sum_ratio    = 0.0; // the sum over them of the tile's load as a multiple of one row's
    int64_t rows_skipped = 0;   // rows in no counted tile: past the last whole tile, or in a tile with a row of no live pool

    double mean() const { return tiles ? sum_ratio / tiles : 0.0; }
};

// sel holds n_rows rows of select_k pool ids, row r at sel[r*select_k]. An id at or past n_live is a dump pool, which
// the top-k takes only when fewer than select_k pools are live, and which holds no cell the tile gathers. Tiles are rows
// [i*tile, (i+1)*tile).
inline topk_tile_stats topk_tile_union(const int32_t * sel, int64_t select_k, int64_t n_rows, int64_t n_live, int64_t tile) {
    topk_tile_stats st;
    if (tile <= 0) return st;

    std::vector<int32_t> uni;
    std::vector<int32_t> row;
    for (int64_t first = 0; first + tile <= n_rows; first += tile) {
        uni.clear();
        int64_t live    = 0;
        bool    counted = true;
        for (int64_t r = first; r < first + tile; ++r) {
            row.clear();
            for (int64_t j = 0; j < select_k; ++j) {
                const int32_t id = sel[r*select_k + j];
                if (id >= 0 && id < n_live) row.push_back(id);
            }
            std::sort(row.begin(), row.end());
            row.erase(std::unique(row.begin(), row.end()), row.end());
            if (row.empty()) counted = false;
            live += (int64_t) row.size();
            uni.insert(uni.end(), row.begin(), row.end());
        }
        if (!counted) {
            st.rows_skipped += tile;
            continue;
        }
        std::sort(uni.begin(), uni.end());
        const int64_t n_uni = std::unique(uni.begin(), uni.end()) - uni.begin();
        st.tiles++;
        st.sum_ratio += (double) n_uni * (double) tile / (double) live;
    }
    st.rows_skipped += n_rows % tile;
    return st;
}

// How much of what a step selects the card would still hold from the steps before it. Row-level tiering keeps a
// sequence's latents in host memory and, on the card, those of the pools selected lately: cap pools of a layer, the one
// selected longest ago leaving first. A pool a step selects and the card does not hold is a miss: its cells' latents
// come from host memory before the attention reads them. A pool is written on the card, so the step that completes it
// holds it (born). While a step runs the card holds all it selected, whatever cap is: the pools over cap leave after.
class topk_lru {
public:
    int64_t steps    = 0; // steps counted
    int64_t selected = 0; // distinct live pools those steps selected
    int64_t misses   = 0; // of them, not held when the step began

    explicit topk_lru(int64_t cap) : cap(cap) {}

    // the pools [first, last) were written since the last step
    void born(int64_t first, int64_t last) {
        for (int64_t id = first; id < last; ++id) touch((int32_t) id);
        trim();
    }

    // sel, select_k, n_rows and n_live as topk_tile_union takes them: one step's rows. A step is counted when every one
    // of its rows selected select_k live pools; a step with a dump pool in it moves the pools it selected and counts
    // nothing, since a row that takes every live pool has no choice to miss with. Returns the step's misses, -1 for a
    // step not counted.
    int64_t step(const int32_t * sel, int64_t select_k, int64_t n_rows, int64_t n_live) {
        ids.assign(sel, sel + select_k*n_rows);
        std::sort(ids.begin(), ids.end());
        ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
        bool full = !ids.empty() && ids.front() >= 0 && ids.back() < n_live;

        int64_t n_miss = 0, n_sel = 0;
        for (const int32_t id : ids) {
            if (id < 0 || id >= n_live) continue;
            n_sel++;
            if (!held(id)) n_miss++;
        }
        for (const int32_t id : ids) {
            if (id >= 0 && id < n_live) touch(id);
        }
        trim();

        if (!full) return -1;
        steps++;
        selected += n_sel;
        misses   += n_miss;
        return n_miss;
    }

    bool    held(int32_t id) const { return (size_t) id < in.size() && in[id]; }
    int64_t n_held() const { return count; }

private:
    // the pools held, as a list from the one selected last (head) to the one selected longest ago (tail)
    void touch(int32_t id) {
        if ((size_t) id >= in.size()) {
            in.resize((size_t) id + 1, 0);
            prev.resize((size_t) id + 1, -1);
            next.resize((size_t) id + 1, -1);
        }
        if (in[id]) {
            if (head == id) return;
            next[prev[id]] = next[id];
            if (next[id] >= 0) prev[next[id]] = prev[id]; else tail = prev[id];
        } else {
            in[id] = 1;
            count++;
        }
        prev[id] = -1;
        next[id] = head;
        if (head >= 0) prev[head] = id; else tail = id;
        head = id;
    }

    void trim() {
        while (count > cap) {
            const int32_t id = tail;
            tail = prev[id];
            if (tail >= 0) next[tail] = -1; else head = -1;
            in[id] = 0;
            count--;
        }
    }

    int64_t              cap;
    int64_t              count = 0;
    int32_t              head  = -1;
    int32_t              tail  = -1;
    std::vector<uint8_t> in;
    std::vector<int32_t> prev, next, ids;
};
