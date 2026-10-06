// The pooled-indexer inputs built from a sequence's view (llama_kpool_views) against the maps built from the cells.
//
// llama_kpool_set_input with views keeps, for each sequence, its positions, pool table and completeness, and updates them
// from the cells llama_kv_cells_log names; with views == nullptr it builds every map from the cells each call. The two
// must write the same bytes into every input tensor.
//
// Random cell states the way a server makes them (as test-kv-mask.cpp): sequences in one unified pool or one stream each,
// tokens appended, drafts rolled back, conversations ended, prefixes shared, cells removed and kept, saved and restored
// (llama_kv_cache::prepare), positions shifted and divided with a rebuild after, two cells at one position, positions
// sparse enough to span more pools than a map holds, prompts large enough to pass the log's cap, the cells copied over.
// One llama_kpool_views lives through a whole round, so the view follows many small changes; after every few operations
// the inputs of a decode or verify ubatch are built both ways and compared byte for byte.
//
// Two controls: a view whose changes were taken by someone else must be caught by the comparison, and the log of the cells
// is checked against a snapshot of the cells on its own.

#include "../src/llama-batch.h"
#include "../src/llama-kv-cache.h"
#include "../src/llama-kv-cache-kpool.h"
#include "../src/llama-kv-cells.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <random>
#include <set>
#include <vector>

static int n_fail = 0;

#define CHECK(cond, ...) do { if (!(cond)) { ++n_fail; if (n_fail <= 20) { fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } } } while (0)

struct pool {
    std::vector<llama_kv_cells> v_cells;       // one per stream
    std::vector<uint32_t>       seq_to_stream;
    std::vector<uint32_t>       head;          // per stream
    uint32_t                    n_seq;
    std::mt19937 &              rng;

    // prepare() between its apply and its restore: the cells saved and the ubatch placed, the restore still to come
    struct pending_t {
        bool                  on = false;
        llama_seq_id          s  = 0;
        std::vector<uint32_t> idxs;
        llama_kv_cells        saved;
    } pending;

    pool(uint32_t size, uint32_t n_seq, bool unified, std::mt19937 & rng) : n_seq(n_seq), rng(rng) {
        const uint32_t n_stream = unified ? 1 : n_seq;
        v_cells.resize(n_stream);
        for (auto & cells : v_cells) {
            cells.resize(size);
        }
        head.assign(n_stream, 0);
        for (uint32_t s = 0; s < n_seq; ++s) {
            seq_to_stream.push_back(unified ? 0 : s);
        }
    }

    llama_kv_cells & cells_of(llama_seq_id s) { return v_cells[seq_to_stream[s]]; }

    llama_pos p_next(llama_seq_id s) { return cells_of(s).seq_pos_max(s) + 1; }

    // the next empty cell from the stream's head on, wrapping (llama_kv_cache::find_slot, not contiguous)
    int64_t find_empty(llama_seq_id s) {
        auto & cells = cells_of(s);
        uint32_t & h = head[seq_to_stream[s]];
        for (uint32_t n = 0; n < cells.size(); ++n) {
            const uint32_t i = (h + n) % cells.size();
            if (cells.is_empty(i)) {
                h = (i + 1) % cells.size();
                return i;
            }
        }
        return -1;
    }

    // n tokens of the sequence at the positions after its last, as a ubatch is applied; the positions placed
    std::vector<llama_pos> append(llama_seq_id s, uint32_t n) {
        std::vector<llama_pos> placed;
        auto & cells = cells_of(s);
        for (uint32_t k = 0; k < n; ++k) {
            const int64_t i = find_empty(s);
            if (i < 0) {
                break;
            }
            const llama_pos p = p_next(s);
            cells.pos_set(i, p);
            cells.seq_add(i, s);
            placed.push_back(p);
        }
        return placed;
    }

    // drop the sequence's positions from p0 on (a rejected draft: llama_kv_cache::seq_rm)
    void rollback(llama_seq_id s, llama_pos p0, llama_pos p1 = INT32_MAX) {
        auto & cells = cells_of(s);
        uint32_t new_head = cells.size();
        for (uint32_t i = 0; i < cells.size(); ++i) {
            if (cells.pos_in(i, p0, p1) && cells.seq_has(i, s) && cells.seq_rm(i, s)) {
                new_head = std::min(new_head, i);
            }
        }
        uint32_t & h = head[seq_to_stream[s]];
        if (new_head < h) {
            h = new_head;
        }
    }

    // a position of the sequence taken a second time, by another cell
    void duplicate(llama_seq_id s) {
        auto & cells = cells_of(s);
        const llama_pos p = cells.seq_pos_max(s);
        const int64_t   i = find_empty(s);
        if (p >= 0 && i >= 0) {
            cells.pos_set(i, rng() % 2 ? p : (llama_pos) (rng() % (p + 1)));
            cells.seq_add(i, s);
        }
    }

    // true when a position moved: the pooled keys are stale, the next inputs re-emit every pool
    bool random_op() {
        const llama_seq_id s = rng() % n_seq;
        auto & cells = cells_of(s);
        switch (rng() % 16) {
            case 0: case 1: case 2: case 3: // a prompt chunk or a few decoded tokens
                append(s, 1 + rng() % (rng() % 8 == 0 ? 300 : 6));
                break;
            case 4: case 5: case 6: // a verify's rejected draft
                if (p_next(s) > 0) {
                    rollback(s, std::max<llama_pos>(0, p_next(s) - 1 - (llama_pos) (rng() % 4)));
                }
                break;
            case 7: // the conversation ends; the next one starts at 0 in whatever cells are free
                if (rng() % 3 == 0) {
                    rollback(s, 0);
                }
                break;
            case 8: // a range out of the middle
                if (p_next(s) > 4) {
                    const llama_pos p0 = rng() % p_next(s);
                    rollback(s, p0, p0 + 1 + rng() % 12);
                }
                break;
            case 9: { // a second sequence of the same stream takes this one's prefix (a cached prompt: seq_cp)
                const llama_seq_id t = rng() % n_seq;
                if (t != s && seq_to_stream[t] == seq_to_stream[s] && cells.seq_pos_max(t) < 0 && p_next(s) > 1) {
                    const llama_pos p1 = 1 + rng() % p_next(s);
                    for (uint32_t i = 0; i < cells.size(); ++i) {
                        if (cells.pos_in(i, 0, p1) && cells.seq_has(i, s) && !cells.seq_has(i, t)) {
                            cells.seq_add(i, t);
                        }
                    }
                }
            } break;
            case 10: { // one cell removed whatever holds it
                const uint32_t i = rng() % cells.size();
                if (!cells.is_empty(i)) {
                    cells.rm(i);
                }
            } break;
            case 11: { // prepare(): cells saved, a ubatch placed; the cells restored by a later operation
                if (pending.on) {
                    cells_of(pending.s).set(pending.idxs, pending.saved);
                    pending.on = false;
                    break;
                }
                std::vector<uint32_t> idxs;
                const uint32_t n = 1 + rng() % 8;
                for (uint32_t k = 0; k < n; ++k) {
                    idxs.push_back(rng() % cells.size());
                }
                std::sort(idxs.begin(), idxs.end());
                idxs.erase(std::unique(idxs.begin(), idxs.end()), idxs.end());
                pending.saved = cells.cp(idxs);
                for (const uint32_t i : idxs) {
                    if (!cells.is_empty(i)) {
                        cells.rm(i);
                    }
                    cells.pos_set(i, p_next(s) + (llama_pos) (rng() % 4));
                    cells.seq_add(i, s);
                }
                pending.on   = true;
                pending.s    = s;
                pending.idxs = idxs;
            } break;
            case 12: { // positions shifted (a context shift) or divided, applied
                const bool div = rng() % 4 == 0;
                const llama_pos d = (rng() % 2 ? 1 : -1) * (llama_pos) (1 + rng() % 64);
                const llama_pos p0 = p_next(s) > 0 ? rng() % p_next(s) : 0;
                for (uint32_t i = 0; i < cells.size(); ++i) {
                    if (cells.pos_in(i, p0, INT32_MAX) && cells.seq_has(i, s) && cells.seq_count(i) == 1) {
                        if (div) {
                            cells.pos_div(i, 2);
                        } else {
                            cells.pos_add(i, d);
                        }
                    }
                }
                cells.reset_shift();
                return true;
            }
            case 13: // seq_keep of one sequence
                if (rng() % 8 == 0) {
                    for (uint32_t i = 0; i < cells.size(); ++i) {
                        cells.seq_keep(i, s);
                    }
                }
                break;
            case 14: // a position taken twice, or cells sparse enough to span more pools than a map holds
                if (rng() % 2) {
                    duplicate(s);
                } else if (rng() % 4 == 0) {
                    const int64_t i = find_empty(s);
                    if (i >= 0) {
                        cells.pos_set(i, p_next(s) + 1000 + rng() % 100000);
                        cells.seq_add(i, s);
                    }
                } else if (rng() % 2) { // a position with no sequence yet
                    const int64_t i = find_empty(s);
                    if (i >= 0) {
                        cells.pos_set(i, rng() % 512);
                    }
                }
                break;
            case 15: // the cells all cleared, or copied over
                if (rng() % 16 == 0) {
                    if (rng() % 2) {
                        cells.reset();
                    } else {
                        llama_kv_cells copy;
                        copy.resize(cells.size());
                        cells = copy;
                    }
                }
                break;
        }
        return false;
    }
};

// the cells of one round, as a snapshot: the log must name every cell that differs from it
struct snap {
    std::vector<llama_pos>  pos;
    std::vector<uint64_t>   seqs;

    void take(const llama_kv_cells & cells, uint32_t n_seq) {
        pos.assign(cells.size(), -1);
        seqs.assign(cells.size(), 0);
        for (uint32_t i = 0; i < cells.size(); ++i) {
            if (cells.is_empty(i)) {
                continue;
            }
            pos[i] = cells.pos_get(i);
            for (uint32_t s = 0; s < n_seq; ++s) {
                if (cells.seq_has(i, s)) {
                    seqs[i] |= (uint64_t) 1 << s;
                }
            }
        }
    }
};

static void check_log(llama_kv_cells & cells, snap & was, uint32_t n_seq) {
    std::vector<uint32_t> list;
    const bool listed = cells.changes().take(list);

    snap now;
    now.take(cells, n_seq);

    if (listed) {
        const std::set<uint32_t> in_list(list.begin(), list.end());
        for (uint32_t i = 0; i < cells.size(); ++i) {
            if (was.pos[i] != now.pos[i] || was.seqs[i] != now.seqs[i]) {
                CHECK(in_list.count(i), "cell %u changed (position %d -> %d) and is not in the log", i, was.pos[i], now.pos[i]);
            }
        }
    }

    was = std::move(now);
}

// the counts against their definition
static void check_counts(const llama_kv_cells & cells, uint32_t n_seq) {
    for (uint32_t s = 0; s < n_seq; ++s) {
        std::set<llama_pos> distinct;
        uint32_t n = 0;
        for (uint32_t i = 0; i < cells.size(); ++i) {
            if (!cells.is_empty(i) && cells.seq_has(i, s)) {
                distinct.insert(cells.pos_get(i));
                n++;
            }
        }
        CHECK(cells.seq_cell_count(s) == n, "seq %u: %u cells, counted %u", s, cells.seq_cell_count(s), n);
        CHECK(cells.seq_pos_distinct(s) == distinct.size(), "seq %u: %u positions, counted %zu", s, cells.seq_pos_distinct(s), distinct.size());
    }
}

// the input tensors of one build, in host memory
struct maps {
    ggml_context *        ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;

    ggml_tensor * pool_cells     = nullptr;
    ggml_tensor * pool_bias      = nullptr;
    ggml_tensor * sel_mask       = nullptr;
    ggml_tensor * cand_mask      = nullptr;
    ggml_tensor * pool_reps      = nullptr;
    ggml_tensor * new_pool_cells = nullptr;
    ggml_tensor * new_pool_reps  = nullptr;

    // n_dump: the top-k's dump pools past the n_pools real ones, with their columns in sel_mask (llm_graph_input_kpool)
    maps(int64_t n_kv, int64_t n_ns, int64_t n_tps, int64_t n_ps, uint32_t r, bool kcache, bool f16, bool rebuild, int64_t n_dump, int fill) {
        ggml_init_params ip = { 32*ggml_tensor_overhead(), nullptr, true };
        ctx = ggml_init(ip);

        const int64_t n_pools   = llama_kpool_n_pools(n_kv, r, n_ps);
        const int64_t n_new_max = rebuild ? n_pools : n_tps/r + n_ps;
        const ggml_type tm      = f16 ? GGML_TYPE_F16 : GGML_TYPE_F32;

        pool_cells = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, r*(n_pools + n_dump), n_ns);
        pool_bias  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_pools, n_tps, n_ns);
        sel_mask   = ggml_new_tensor_4d(ctx, tm, n_kv + r*n_dump, n_tps, 1, n_ns);
        cand_mask  = ggml_new_tensor_4d(ctx, tm, n_kv, n_tps, 1, n_ns);

        if (kcache) {
            pool_reps      = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_pools, n_ns);
            new_pool_cells = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, r*n_new_max, n_ns);
            new_pool_reps  = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_new_max*n_ns);
        }

        buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_cpu_buffer_type());
        // a byte the build does not write stays different between the two builds
        ggml_backend_buffer_clear(buf, fill);
    }

    ~maps() {
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }

    maps(const maps &) = delete;

    std::vector<ggml_tensor *> all() const {
        return { pool_cells, pool_bias, sel_mask, cand_mask, pool_reps, new_pool_cells, new_pool_reps };
    }
};

// the ubatch: seq_id[i][0] and pos[i] are read, and seq_id_unq and n_seqs_unq
struct ubatch_data {
    std::vector<llama_pos>      pos;
    std::vector<llama_seq_id>   seq;
    std::vector<llama_seq_id *> seq_ptr;
    std::vector<llama_seq_id>   unq;
    llama_ubatch                ub = {};

    void add(llama_seq_id s, llama_pos p) {
        pos.push_back(p);
        seq.push_back(s);
    }

    const llama_ubatch * get() {
        seq_ptr.clear();
        unq.clear();
        for (auto & s : seq) {
            seq_ptr.push_back(&s);
            if (std::find(unq.begin(), unq.end(), s) == unq.end()) {
                unq.push_back(s);
            }
        }
        ub.n_tokens     = pos.size();
        ub.n_seq_tokens = pos.size();
        ub.n_seqs       = unq.size();
        ub.n_seqs_unq   = unq.size();
        ub.n_pos        = 1;
        ub.pos          = pos.data();
        ub.seq_id       = seq_ptr.data();
        ub.seq_id_unq   = unq.data();
        return &ub;
    }
};

static const char * const names[] = { "pool_cells", "pool_bias", "sel_mask", "cand_mask", "pool_reps", "new_pool_cells", "new_pool_reps" };

// every byte of every tensor; the first one that differs is named
static bool same(const maps & a, const maps & b, char * what, size_t n_what) {
    const auto ta = a.all();
    const auto tb = b.all();
    for (size_t t = 0; t < ta.size(); ++t) {
        if (ta[t] == nullptr) {
            continue;
        }
        const size_t n = ggml_nbytes(ta[t]);
        const uint8_t * pa = (const uint8_t *) ta[t]->data;
        const uint8_t * pb = (const uint8_t *) tb[t]->data;
        for (size_t k = 0; k < n; ++k) {
            if (pa[k] != pb[k]) {
                snprintf(what, n_what, "%s byte %zu of %zu: %02x, reference %02x", names[t], k, n, pa[k], pb[k]);
                return false;
            }
        }
    }
    return true;
}

struct config {
    uint32_t r;
    bool     unified;
    bool     kcache;
    bool     f16;
    bool     small_kv = true; // now and then an n_kv below the used cells, which the views must decline
};

enum control {
    CONTROL_NONE,
    CONTROL_STEAL,   // someone else takes the changes the view is waiting for
    CONTROL_CORRUPT, // a byte of the views' output is flipped
};

// the rows the pooled keys are written to: all different, because set_rows writes them from several threads, and a slot
// that names the last cell of a block must be the one that computes that cell's pool, because that cell's pooled key is read
static bool write_rows_safe(const maps & m, const pool & w, int64_t n_ns, uint32_t r, char * what, size_t n_what) {
    if (m.new_pool_reps == nullptr) {
        return true;
    }

    const int64_t  n_slots = m.new_pool_reps->ne[0];
    const int64_t  n_new_max = n_slots/n_ns;
    const int64_t  kv_size = w.v_cells[0].size();
    const int64_t * rows   = (const int64_t *) m.new_pool_reps->data;
    const int32_t * cells  = (const int32_t *) m.new_pool_cells->data;

    // r 1 has no spare cell (every cell is a pool), so an unused slot repeats a pool and writes the same key again
    for (int64_t i = 0; r > 1 && i < n_slots; ++i) {
        for (int64_t j = i + 1; j < n_slots; ++j) {
            if (rows[i] == rows[j]) {
                const llama_kv_cells & c = w.v_cells[0];
                snprintf(what, n_what, "new_pool_reps: row %lld written by slots %lld and %lld of %lld (cell %s, pos %u, n_ns %lld) rows %lld %lld %lld %lld %lld %lld; used %u/%u", (long long) rows[i],
                        (long long) i, (long long) j, (long long) n_slots, c.is_empty(rows[i] % kv_size) ? "empty" : "used", (unsigned) (c.is_empty(rows[i] % kv_size) ? 9999 : c.pos_get(rows[i] % kv_size)), (long long) n_ns,
                        (long long) rows[0], (long long) rows[1], (long long) rows[std::min<int64_t>(2, n_slots - 1)], (long long) rows[std::min<int64_t>(3, n_slots - 1)],
                        (long long) rows[std::min<int64_t>(4, n_slots - 1)], (long long) rows[std::min<int64_t>(5, n_slots - 1)], c.get_used(), c.size());
                return false;
            }
        }
    }

    for (int64_t i = 0; i < n_slots; ++i) {
        const int64_t s    = i/n_new_max;
        const int64_t cell = rows[i] - s*kv_size;

        if (cell < 0 || cell >= kv_size) {
            snprintf(what, n_what, "new_pool_reps[%lld] = %lld: outside stream %lld", (long long) i, (long long) rows[i], (long long) s);
            return false;
        }

        const llama_kv_cells & c = w.v_cells[s];

        if (!c.is_empty(cell) && c.pos_get(cell) % (llama_pos) r == (llama_pos) r - 1 &&
                cells[s*r*n_new_max + (i % n_new_max)*r + r - 1] != (int32_t) cell) {
            snprintf(what, n_what, "new_pool_reps[%lld]: cell %lld is the last of its block but the slot computes another pool", (long long) i, (long long) cell);
            return false;
        }
    }

    return true;
}

// the dump pools name cells n_kv + c, one each, and the real pools' cells are under n_kv, so the slots of a row of the top-k,
// real pools and dump pools together, never name one cell twice (the mask's set_rows writes them from several threads); a
// dump column is masked in every row, and cand_mask has none
static bool dump_safe(const maps & m, int64_t n_kv, uint32_t r, char * what, size_t n_what) {
    const int64_t n_ns    = m.pool_cells->ne[1];
    const int64_t n_pools = m.pool_bias->ne[0];
    const int64_t n_all   = m.pool_cells->ne[0]/r;
    const int64_t n_sel   = m.sel_mask->ne[0];
    const int64_t n_rows  = m.sel_mask->ne[1];

    if (m.cand_mask->ne[0] != n_kv || n_sel != n_kv + r*(n_all - n_pools)) {
        snprintf(what, n_what, "dump: sel_mask %lld columns, cand_mask %lld, for n_kv %lld and %lld dump pools", (long long) n_sel,
                (long long) m.cand_mask->ne[0], (long long) n_kv, (long long) (n_all - n_pools));
        return false;
    }

    for (int64_t s = 0; s < n_ns; ++s) {
        const int32_t * pc = (const int32_t *) m.pool_cells->data + s*r*n_all;

        for (int64_t c = 0; c < r*n_all; ++c) {
            const bool ok = c < r*n_pools ? pc[c] >= 0 && pc[c] < n_kv : pc[c] == n_kv + (c - r*n_pools);
            if (!ok) {
                snprintf(what, n_what, "dump: stream %lld, pool_cells[%lld] = %d (a %s pool; %lld real pools, n_kv %lld)", (long long) s, (long long) c,
                        pc[c], c < r*n_pools ? "real" : "dump", (long long) n_pools, (long long) n_kv);
                return false;
            }
        }

        for (int64_t ii = 0; ii < n_rows; ++ii) {
            for (int64_t j = n_kv; j < n_sel; ++j) {
                const int64_t k = (s*n_rows + ii)*n_sel + j;
                const float   x = m.sel_mask->type == GGML_TYPE_F16 ? ggml_fp16_to_fp32(((const ggml_fp16_t *) m.sel_mask->data)[k]) :
                                                                       ((const float *) m.sel_mask->data)[k];
                if (!(std::isinf(x) && x < 0)) {
                    snprintf(what, n_what, "dump: stream %lld, sel_mask row %lld dump column %lld = %g", (long long) s, (long long) ii, (long long) j, x);
                    return false;
                }
            }
        }
    }

    return true;
}

// the inputs of one ubatch built from the views and from the cells. The ubatch is placed in the cells first, as the cache
// does (apply_ubatch before the inputs are set). rebuild: re-emit every pool, as after a position mutation
static bool build_both(pool & w, llama_kpool_views & views, const config & cfg, bool rebuild, std::mt19937 & rng, control ctl, char * what, size_t n_what) {
    const uint32_t n_ns   = cfg.unified ? 1 : w.n_seq;
    const uint32_t n_tps  = rng() % 2 ? 1 : 2 + rng() % 4;
    const bool     two_ps = cfg.unified && w.n_seq > 1 && rng() % 6 == 0;

    ubatch_data ud;

    if (cfg.unified) {
        // one sequence, or two sharing the ubatch (the maps are then built from the cells)
        for (uint32_t k = 0; k < (two_ps ? 2u : 1u); ++k) {
            const llama_seq_id s = two_ps ? k : (llama_seq_id) (rng() % w.n_seq);
            for (const llama_pos p : w.append(s, two_ps ? 1 + n_tps/2 : n_tps)) {
                ud.add(s, p);
            }
        }
    } else {
        // one stream each, the same count of tokens from each
        std::vector<std::vector<llama_pos>> placed;
        size_t n_min = n_tps;
        for (uint32_t s = 0; s < n_ns; ++s) {
            placed.push_back(w.append(s, n_tps));
            n_min = std::min(n_min, placed.back().size());
        }
        for (uint32_t s = 0; s < n_ns; ++s) {
            // a stream that placed more than another keeps only as many, so the streams hold the same count; the ubatch is
            // the last of its sequence, nothing lies ahead of it
            for (size_t k = 0; k < n_min; ++k) {
                ud.add(s, placed[s][k]);
            }
            if (placed[s].size() > n_min) {
                w.rollback(s, placed[s][n_min]);
            }
        }
    }

    if (ud.pos.empty() || ud.pos.size() % n_ns != 0) {
        return true;
    }

    const llama_ubatch * ub = ud.get();

    if (ub->n_seqs_unq % n_ns != 0 || (n_ns > 1 && ub->n_seqs_unq != n_ns)) {
        return true;
    }

    const int64_t n_tokens = ub->n_tokens;
    const int64_t n_ps     = ub->n_seqs_unq/n_ns;

    // the tokens of a stream must be the stream's own, and one run of positions each (llama-batch.cpp)
    if (n_ns > 1) {
        for (int64_t i = 0; i < n_tokens; ++i) {
            if (ub->seq_id[i][0] != (llama_seq_id) (i/(n_tokens/n_ns))) {
                return true;
            }
        }
    }

    uint32_t used_max = 0;
    for (const auto & cells : w.v_cells) {
        used_max = std::max(used_max, cells.used_max_p1());
    }

    // the cache's n_kv, padded; now and then a bound below the used cells, which the views must decline
    int64_t n_kv = std::min<int64_t>(w.v_cells[0].size(), std::max<int64_t>(32, (used_max + 31)/32*32));
    if (cfg.small_kv && rng() % 24 == 0 && used_max > 8) {
        n_kv = 1 + rng() % (used_max - 1);
    }

    std::vector<uint32_t> strm_of(n_ns);
    for (uint32_t s = 0; s < n_ns; ++s) {
        strm_of[s] = s;
    }

    const llama_kpool_cells_fn cells_of = [&](llama_seq_id s) -> const llama_kv_cells & { return w.cells_of(s); };

    if (ctl == CONTROL_STEAL) {
        std::vector<uint32_t> tmp;
        for (auto & cells : w.v_cells) {
            cells.changes().take(tmp);
        }
    }

    // none, a few, or as many as a top-k over every pool takes
    const int64_t n_pools = llama_kpool_n_pools(n_kv, cfg.r, n_ps);
    const int64_t n_dump  = rng() % 3 == 0 ? 0 : rng() % 2 ? 1 + rng() % 4 : n_pools;

    maps ref(n_kv, n_ns, n_tokens/n_ns, n_ps, cfg.r, cfg.kcache, cfg.f16, rebuild, n_dump, 0xA5);
    maps inc(n_kv, n_ns, n_tokens/n_ns, n_ps, cfg.r, cfg.kcache, cfg.f16, rebuild, n_dump, 0x5A);

    // the views go first: the reference never takes the log
    llama_kpool_set_input(cells_of, &views, nullptr, inc.pool_cells, nullptr, inc.pool_bias, inc.sel_mask, inc.cand_mask,
            inc.pool_reps, inc.new_pool_cells, inc.new_pool_reps, strm_of.data(), w.v_cells[0].size(), rebuild, ub, cfg.r);
    llama_kpool_set_input(cells_of, nullptr, nullptr, ref.pool_cells, nullptr, ref.pool_bias, ref.sel_mask, ref.cand_mask,
            ref.pool_reps, ref.new_pool_cells, ref.new_pool_reps, strm_of.data(), w.v_cells[0].size(), rebuild, ub, cfg.r);

    if (ctl == CONTROL_CORRUPT) {
        ((uint8_t *) inc.sel_mask->data)[0] ^= 0x80;
    }

    return write_rows_safe(ref, w, n_ns, cfg.r, what, n_what) && write_rows_safe(inc, w, n_ns, cfg.r, what, n_what) &&
           dump_safe(ref, n_kv, cfg.r, what, n_what) && same(inc, ref, what, n_what);
}

static void run_round(std::mt19937 & rng, const config & cfg, uint32_t size, uint32_t n_seq, int n_ops, llama_kpool_views::stats & total) {
    pool w(size, n_seq, cfg.unified, rng);
    llama_kpool_views views;

    bool rebuild = false;

    for (int k = 0; k < n_ops; ++k) {
        rebuild |= w.random_op();

        if (rng() % 3 != 0) {
            continue;
        }

        char what[256];
        if (!build_both(w, views, cfg, rebuild, rng, CONTROL_NONE, what, sizeof(what))) {
            CHECK(false, "r %u, %s, %s: %s", cfg.r, cfg.unified ? "unified" : "streams", cfg.f16 ? "f16" : "f32", what);
        }
        rebuild = rng() % 8 == 0;

        for (size_t i = 0; i < w.v_cells.size(); ++i) {
            check_counts(w.v_cells[i], n_seq);
        }
    }

    const auto & st = views.get_stats();
    total.n_served  += st.n_served;
    total.n_rebuilt += st.n_rebuilt;
    total.n_direct  += st.n_direct;
}

// the log on its own: a consumer that takes it after every few operations finds every changed cell in it
static void run_log_round(std::mt19937 & rng, uint32_t size, uint32_t n_seq, bool unified, int n_ops) {
    pool w(size, n_seq, unified, rng);

    std::vector<snap> was(w.v_cells.size());
    for (size_t i = 0; i < w.v_cells.size(); ++i) {
        w.v_cells[i].changes().track(size/16);
        was[i].take(w.v_cells[i], n_seq);
    }

    for (int k = 0; k < n_ops; ++k) {
        w.random_op();

        if (rng() % 3 == 0) {
            for (size_t i = 0; i < w.v_cells.size(); ++i) {
                check_log(w.v_cells[i], was[i], n_seq);
                check_counts(w.v_cells[i], n_seq);
            }
        }
    }
}

// a decode loop with rollbacks, nothing else: the view must follow it without a rebuild. With moves, a hole in the cells
// lies below the newest token, and every few steps the token is moved into it (a defrag): a cell takes the position another
// cell, of a higher index, leaves in the same changes
static void run_steady(std::mt19937 & rng, bool moves) {
    const config cfg = { 4, true, true, true, false };

    pool w(4096, 1, true, rng);
    llama_kpool_views views;
    char what[256];

    for (int k = 0; k < 1200; ++k) {
        if (k % 5 == 4) {
            w.rollback(0, w.p_next(0) - 1 - rng() % 3);
        }

        auto & cells = w.cells_of(0);

        if (moves && k == 200) {
            for (uint32_t i = 0; i < cells.size(); ++i) {
                if (!cells.is_empty(i) && cells.pos_get(i) == 100) {
                    cells.rm(i);
                }
            }
        }

        if (moves && k > 200 && k % 25 == 0) {
            uint32_t hi = 0;
            uint32_t lo = 0;
            while (lo < cells.size() && !cells.is_empty(lo)) {
                lo++;
            }
            for (uint32_t i = 0; i < cells.size(); ++i) {
                if (!cells.is_empty(i) && cells.pos_get(i) == cells.seq_pos_max(0)) {
                    hi = i;
                }
            }
            if (lo < hi) {
                const llama_pos p = cells.pos_get(hi);
                cells.rm(hi);
                cells.pos_set(lo, p);
                cells.seq_add(lo, 0);
            }
        }

        if (!build_both(w, views, cfg, false, rng, CONTROL_NONE, what, sizeof(what))) {
            CHECK(false, "steady, step %d: %s", k, what);
            break;
        }
    }

    const auto & st = views.get_stats();
    printf("steady%s: %llu served, %llu rebuilt, %llu from the cells\n", moves ? " with moves" : "", (unsigned long long) st.n_served,
            (unsigned long long) st.n_rebuilt, (unsigned long long) st.n_direct);
    CHECK(st.n_rebuilt == 1, "steady decode rebuilt the view %llu times, once is the first build", (unsigned long long) st.n_rebuilt);
    CHECK(st.n_direct == 0, "steady decode built %llu streams from the cells", (unsigned long long) st.n_direct);
}

// the comparison fails on a difference; and a view that lost changes which left its count of cells off is rebuilt, not served
static void run_controls(std::mt19937 & rng) {
    const config cfg = { 4, true, true, true, false };

    pool w(512, 1, true, rng);
    llama_kpool_views views;
    char what[256];

    for (int k = 0; k < 40; ++k) {
        CHECK(build_both(w, views, cfg, false, rng, CONTROL_NONE, what, sizeof(what)), "controls, warm-up: %s", what);
    }

    CHECK(!build_both(w, views, cfg, false, rng, CONTROL_CORRUPT, what, sizeof(what)), "a flipped byte passed the comparison");

    const uint64_t n_rebuilt = views.get_stats().n_rebuilt;

    for (int k = 0; k < 20; ++k) {
        CHECK(build_both(w, views, cfg, false, rng, CONTROL_STEAL, what, sizeof(what)), "a view that lost its changes was served: %s", what);
    }

    CHECK(views.get_stats().n_rebuilt > n_rebuilt, "a view whose changes were taken was not rebuilt");
}

// The stale marks of the pooled keys across two sequences, as a server makes them: sequence 0 shifted by a context shift
// (seq_rm of [4, 9), then seq_add of [9, end) by -5), a decode of sequence 1, a decode of sequence 0. The pooled keys are
// kept as the indexer cache keeps them, one row per complete pool written by the new-pool slots; after the last decode
// every complete pool of both sequences must hold the key of its members. `per_seq` false is the cache's former scheme, one
// flag for the cache, which the decode of sequence 1 clears: the count of stale pools it leaves shows the check can fail.
// Returns the count of complete pools whose key is not their members' after the last decode
static int64_t stale_after_shift(std::mt19937 & rng, bool per_seq) {
    const uint32_t r    = 4;
    const uint32_t size = 256;

    pool w(size, 2, true, rng);

    llama_kpool_stale stale;  // the per-sequence marks
    bool              flag = false; // the former flag

    // the indexer cache: rep row -> the members the key in it was made from
    std::map<int64_t, std::vector<int32_t>> key;

    const llama_kpool_cells_fn cells_of = [&](llama_seq_id s) -> const llama_kv_cells & { return w.cells_of(s); };
    const uint32_t strm_of[1] = { 0 };

    auto decode = [&](llama_seq_id s, uint32_t n) {
        ubatch_data ud;
        for (const llama_pos p : w.append(s, n)) {
            ud.add(s, p);
        }
        const llama_ubatch * ub = ud.get();

        const bool rebuild = per_seq ? stale.any(*ub) : flag;

        const int64_t n_kv  = std::max<int64_t>(32, (w.v_cells[0].used_max_p1() + 31)/32*32);
        const int64_t n_tps = ub->n_tokens;

        maps m(n_kv, 1, n_tps, 1, r, true, true, rebuild, 0, 0);

        llama_kpool_set_input(cells_of, nullptr, nullptr, m.pool_cells, nullptr, m.pool_bias, m.sel_mask, m.cand_mask,
                m.pool_reps, m.new_pool_cells, m.new_pool_reps, strm_of, size, rebuild, ub, r);

        // set_rows: every slot writes its row, a real one the key of its members
        const int64_t   n_new_max = m.new_pool_cells->ne[0]/r;
        const int32_t * cells_in  = (const int32_t *) m.new_pool_cells->data;
        const int64_t * rows      = (const int64_t *) m.new_pool_reps->data;

        for (int64_t k = 0; k < n_new_max; ++k) {
            key[rows[k]] = std::vector<int32_t>(cells_in + k*r, cells_in + (k + 1)*r);
        }

        if (per_seq) {
            if (rebuild) {
                stale.clear(*ub);
            }
        } else {
            flag = false;
        }

        return rebuild;
    };

    // the complete pools of a sequence whose row does not hold the key of their members
    auto n_stale = [&](llama_seq_id s) {
        const auto & cells = w.cells_of(s);

        std::map<llama_pos, std::vector<int32_t>> pools;
        for (uint32_t i = 0; i < cells.size(); ++i) {
            if (!cells.is_empty(i) && cells.seq_has(i, s)) {
                auto & m = pools[cells.pos_get(i)/r];
                m.resize(r, -1);
                m[cells.pos_get(i) % r] = (int32_t) i;
            }
        }

        int64_t n = 0;
        for (const auto & [b, members] : pools) {
            if (std::find(members.begin(), members.end(), -1) != members.end()) {
                continue;
            }
            const auto it = key.find(members[r - 1]);
            n += it == key.end() || it->second != members;
        }
        return n;
    };

    decode(0, 16);
    decode(1, 16);
    CHECK(n_stale(0) == 0 && n_stale(1) == 0, "the prefills left %lld and %lld stale pools", (long long) n_stale(0), (long long) n_stale(1));

    // the context shift of sequence 0, as llama_memory_hybrid::seq_rm and seq_add apply it
    w.rollback(0, 4, 9);
    auto & cells = w.cells_of(0);
    for (uint32_t i = 0; i < cells.size(); ++i) {
        if (cells.pos_in(i, 9, INT32_MAX) && cells.seq_has(i, 0)) {
            cells.pos_add(i, -5);
        }
    }
    cells.reset_shift();

    CHECK(llama_kpool_stale::shift_regroups(r, 9, -1, -5), "a shift by -5 over pools of 4 must regroup them");
    stale.mark(0);
    flag = true;

    const bool rebuilt_1 = decode(1, 1);
    const bool rebuilt_0 = decode(0, 1);

    if (per_seq) {
        CHECK(!rebuilt_1, "a decode of sequence 1 rebuilt for sequence 0's shift");
        CHECK(rebuilt_0, "the decode of the shifted sequence 0 did not rebuild");

        // and the marks are spent: the next decode of sequence 0 is a plain one
        ubatch_data ud;
        ud.add(0, w.p_next(0));
        CHECK(!stale.any(*ud.get()), "sequence 0 is still marked after the decode that rebuilt it");
    }

    return n_stale(0) + n_stale(1);
}

// which shifts regroup pools, as llama_memory_hybrid::seq_add asks
static void run_shift_regroups() {
    CHECK(!llama_kpool_stale::shift_regroups(4, 8, -1, -4), "whole pools moved by a whole pool");
    CHECK(!llama_kpool_stale::shift_regroups(4, -1, -1, 8), "the whole sequence moved by two pools");
    CHECK( llama_kpool_stale::shift_regroups(4, 9, -1, -5), "a shift by -5");
    CHECK( llama_kpool_stale::shift_regroups(4, 9, -1, -4), "a range that starts inside a pool");
    CHECK( llama_kpool_stale::shift_regroups(4, 8, 10, -4), "a range that ends inside a pool");
}

int main() {
    std::mt19937 rng(20260930);

    run_shift_regroups();

    const int64_t n_stale_per_seq = stale_after_shift(rng, true);
    const int64_t n_stale_flag    = stale_after_shift(rng, false);

    printf("a context shift of one sequence: %lld stale pools with per-sequence marks, %lld with one flag for the cache\n",
            (long long) n_stale_per_seq, (long long) n_stale_flag);
    CHECK(n_stale_per_seq == 0, "%lld pools kept a stale key after the shifted sequence's own decode", (long long) n_stale_per_seq);
    CHECK(n_stale_flag > 0, "one flag for the cache left no pool stale: the check cannot fail");

    llama_kpool_views::stats total;

    const config configs[] = {
        { 4, true,  true,  true  },
        { 4, true,  true,  false },
        { 2, true,  false, true  },
        { 8, true,  true,  true  },
        { 1, true,  true,  true  },
        { 4, false, true,  true  },
        { 4, false, false, false },
    };

    for (int round = 0; round < 420; ++round) {
        const config & cfg = configs[round % (sizeof(configs)/sizeof(configs[0]))];
        const uint32_t size  = std::vector<uint32_t>{ 64, 256, 1024, 2048 }[rng() % 4];
        const uint32_t n_seq = cfg.unified ? 1 + rng() % 3 : 2 + rng() % 2;
        run_round(rng, cfg, size, n_seq, 200 + rng() % 300, total);
    }

    for (int round = 0; round < 200; ++round) {
        run_log_round(rng, std::vector<uint32_t>{ 64, 256, 1024, 4096 }[rng() % 4], 1 + rng() % 4, rng() % 3 != 0, rng() % 400);
    }

    run_steady(rng, false);
    run_steady(rng, true);
    run_controls(rng);

    printf("random rounds: %llu streams from views, %llu views rebuilt, %llu streams from the cells\n",
            (unsigned long long) total.n_served, (unsigned long long) total.n_rebuilt, (unsigned long long) total.n_direct);

    // the rounds must reach the view path, not only the fallback
    CHECK(total.n_served > 2000, "only %llu streams were served from views", (unsigned long long) total.n_served);
    CHECK(total.n_direct > 100,  "only %llu streams took the maps from the cells: the fallbacks are not reached", (unsigned long long) total.n_direct);

    if (n_fail == 0) {
        printf("ok\n");
    }
    return n_fail != 0;
}
