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
//
// The device build (mask_cells and mask_rows in place of sel_mask and cand_mask) must give, through ggml_kpool_mask, both
// masks byte for byte as the host writes them, from the views and from the cells, with every other tensor and the tail the
// same. And a cell cand_mask lets a row attend to must be one the KQ mask lets it attend to, so the sparse attention can
// leave out its add of the KQ mask (build_attn_sparse) without changing a mask.

#include "../src/llama-batch.h"
#include "../src/llama-hparams.h"
#include "../src/llama-kv-cache.h"
#include "../src/llama-kv-cache-kpool.h"
#include "../src/llama-kv-cells.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "../ggml/src/ggml-backend-impl.h"

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

    // a device build's facts, in place of sel_mask and cand_mask
    ggml_tensor * mask_cells     = nullptr;
    ggml_tensor * mask_rows      = nullptr;
    ggml_tensor * tail_cells     = nullptr;

    // n_dump: the top-k's dump pools past the n_pools real ones, with their columns in sel_mask (llm_graph_input_kpool);
    // n_pools_of > 0 sizes the maps on that many pools instead of the whole cache's; device: the facts, not the masks
    maps(int64_t n_kv, int64_t n_ns, int64_t n_tps, int64_t n_ps, uint32_t r, bool kcache, bool f16, bool rebuild, int64_t n_dump, int fill,
            int64_t n_pools_of = 0, bool device = false) {
        ggml_init_params ip = { 32*ggml_tensor_overhead(), nullptr, true };
        ctx = ggml_init(ip);

        const int64_t n_pools   = n_pools_of > 0 ? n_pools_of : llama_kpool_n_pools(n_kv, r, n_ps);
        const int64_t n_new_max = rebuild ? n_pools : n_tps/r + n_ps;
        const ggml_type tm      = f16 ? GGML_TYPE_F16 : GGML_TYPE_F32;

        pool_cells = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, r*(n_pools + n_dump), n_ns);
        pool_bias  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_pools, n_tps, n_ns);
        if (device) {
            mask_cells = ggml_new_tensor_4d(ctx, GGML_TYPE_I32, n_kv, 2, n_ps, n_ns);
            mask_rows  = ggml_new_tensor_3d(ctx, GGML_TYPE_I32, 4, n_tps, n_ns);
            tail_cells = ggml_new_tensor_3d(ctx, GGML_TYPE_I32, n_kv, n_tps, n_ns);
        } else {
            sel_mask   = ggml_new_tensor_4d(ctx, tm, n_kv + r*n_dump, n_tps, 1, n_ns);
            cand_mask  = ggml_new_tensor_4d(ctx, tm, n_kv, n_tps, 1, n_ns);
        }

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

// both masks as ggml_kpool_mask builds them, on the CPU backend, from the facts of a device build
static void op_masks(const maps & dev, int64_t n_extra, std::vector<ggml_fp16_t> & sel, std::vector<ggml_fp16_t> & cand) {
    static ggml_backend_t cpu = ggml_backend_cpu_init();

    ggml_init_params gp = { 8*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * cg = ggml_init(gp);
    ggml_cgraph  * gf = ggml_new_graph(cg);

    ggml_tensor * t_sel  = ggml_kpool_mask(cg, dev.mask_cells, dev.mask_rows, GGML_KPOOL_MASK_SEL,  n_extra);
    ggml_tensor * t_cand = ggml_kpool_mask(cg, dev.mask_cells, dev.mask_rows, GGML_KPOOL_MASK_CAND, 0);
    ggml_build_forward_expand(gf, t_sel);
    ggml_build_forward_expand(gf, t_cand);

    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
    CHECK(ggml_gallocr_alloc_graph(galloc, gf), "the mask graph did not allocate");
    ggml_backend_graph_compute(cpu, gf);

    sel .resize(ggml_nelements(t_sel));
    cand.resize(ggml_nelements(t_cand));
    ggml_backend_tensor_get(t_sel,  sel .data(), 0, ggml_nbytes(t_sel));
    ggml_backend_tensor_get(t_cand, cand.data(), 0, ggml_nbytes(t_cand));

    ggml_gallocr_free(galloc);
    ggml_free(cg);
}

// a host mask against the op's, as f16 bits: the elements are 0 and -inf, exact in either type
static bool mask_equal(const ggml_tensor * ref, const std::vector<ggml_fp16_t> & got, const char * name, char * what, size_t n_what) {
    const int64_t n = ggml_nelements(ref);

    if ((int64_t) got.size() != n) {
        snprintf(what, n_what, "%s: the op gave %zu elements, the host %lld", name, got.size(), (long long) n);
        return false;
    }

    for (int64_t k = 0; k < n; ++k) {
        const ggml_fp16_t want = ref->type == GGML_TYPE_F16 ? ((const ggml_fp16_t *) ref->data)[k] :
                                                              ggml_fp32_to_fp16(((const float *) ref->data)[k]);
        if (got[k] != want) {
            snprintf(what, n_what, "%s column %lld row %lld of %lld x %lld: %04x from the op, %04x from the host", name,
                    (long long) (k % ref->ne[0]), (long long) (k/ref->ne[0]), (long long) ref->ne[0],
                    (long long) (n/ref->ne[0]), got[k], want);
            return false;
        }
    }

    return true;
}

// every cell cand_mask lets a row attend to is one the KQ mask does: a cell of the row's sequence at or before its position
static bool cand_in_kq(const maps & ref, pool & w, const llama_ubatch * ub, char * what, size_t n_what) {
    const int64_t n_kv  = ref.cand_mask->ne[0];
    const int64_t n_tps = ref.cand_mask->ne[1];
    const int64_t n_ns  = ref.cand_mask->ne[3];

    ggml_init_params ip = { 2*ggml_tensor_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor  * kq  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, n_kv, n_tps, 1, n_ns);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_cpu_buffer_type());

    llama_hparams hparams{};
    llama_kv_cache_set_input_kq_mask(kq, ub, /*causal_attn =*/ true, hparams, w.v_cells, w.seq_to_stream, 0, LLAMA_SWA_TYPE_NONE,
            /*scan_cells =*/ true, /*seq_bits =*/ false);

    bool ok = true;

    for (int64_t k = 0; ok && k < n_kv*n_tps*n_ns; ++k) {
        const float c = ref.cand_mask->type == GGML_TYPE_F16 ? ggml_fp16_to_fp32(((const ggml_fp16_t *) ref.cand_mask->data)[k]) :
                                                               ((const float *) ref.cand_mask->data)[k];
        const float q = ((const float *) kq->data)[k];

        if (c == 0.0f && q != 0.0f) {
            snprintf(what, n_what, "cand_mask lets row %lld attend to cell %lld, the KQ mask does not (%g)", (long long) (k/n_kv),
                    (long long) (k % n_kv), q);
            ok = false;
        }
    }

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);

    return ok;
}

// builds whose masks the op built, and cells among them a row's candidates hold
static int64_t n_device_builds = 0;

// the most finite columns a row of sel_mask holds among its first n_kv: the cells its tail grants
static int64_t sel_tail_max(const maps & m, int64_t n_kv) {
    const int64_t n_sel  = m.sel_mask->ne[0];
    const int64_t n_rows = m.sel_mask->ne[1]*m.sel_mask->ne[3];

    int64_t n_max = 0;

    for (int64_t row = 0; row < n_rows; ++row) {
        int64_t n = 0;
        for (int64_t j = 0; j < n_kv; ++j) {
            const int64_t k = row*n_sel + j;
            const float   x = m.sel_mask->type == GGML_TYPE_F16 ? ggml_fp16_to_fp32(((const ggml_fp16_t *) m.sel_mask->data)[k]) :
                                                                   ((const float *) m.sel_mask->data)[k];
            n += std::isfinite(x);
        }
        n_max = std::max(n_max, n);
    }

    return n_max;
}

// builds whose longest tail held more than r - 1 cells (two cells at one position)
static int64_t n_tail_long = 0;

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
    const int64_t tail_inc = llama_kpool_set_input(cells_of, &views, nullptr, inc.pool_cells, nullptr, inc.pool_bias, inc.sel_mask, inc.cand_mask,
            inc.pool_reps, inc.new_pool_cells, inc.new_pool_reps, strm_of.data(), w.v_cells[0].size(), rebuild, ub, cfg.r);
    const int64_t tail_ref = llama_kpool_set_input(cells_of, nullptr, nullptr, ref.pool_cells, nullptr, ref.pool_bias, ref.sel_mask, ref.cand_mask,
            ref.pool_reps, ref.new_pool_cells, ref.new_pool_reps, strm_of.data(), w.v_cells[0].size(), rebuild, ub, cfg.r);

    // the tail a row is granted, as set_input reports it and as sel_mask holds it, within the bound the graph sizes on
    const int64_t tail_sel = sel_tail_max(ref, n_kv);
    const int64_t tail_max = llama_kpool_tail_max(cells_of, *ub, cfg.r);

    n_tail_long += tail_ref > (int64_t) cfg.r - 1;

    if (tail_inc != tail_ref || tail_ref != tail_sel || tail_ref > tail_max) {
        snprintf(what, n_what, "tail: views report %lld cells, the cells %lld, sel_mask holds %lld, the bound is %lld",
                (long long) tail_inc, (long long) tail_ref, (long long) tail_sel, (long long) tail_max);
        return false;
    }

    if (ctl == CONTROL_CORRUPT) {
        ((uint8_t *) inc.sel_mask->data)[0] ^= 0x80;
    }

    if (!(write_rows_safe(ref, w, n_ns, cfg.r, what, n_what) && write_rows_safe(inc, w, n_ns, cfg.r, what, n_what) &&
          dump_safe(ref, n_kv, cfg.r, what, n_what) && same(inc, ref, what, n_what))) {
        return false;
    }

    if (ctl != CONTROL_NONE) {
        return true;
    }

    if (!cand_in_kq(ref, w, ub, what, n_what)) {
        return false;
    }

    // the device build, from the views and from the cells: the masks through the op, everything else as the host wrote it;
    // and the facts of the two the same bytes, as LLAMA_KPOOL_INPUT_CHECK compares them
    std::vector<uint8_t> facts_views;

    for (const bool from_views : { true, false }) {
        maps dev(n_kv, n_ns, n_tokens/n_ns, n_ps, cfg.r, cfg.kcache, cfg.f16, rebuild, n_dump, from_views ? 0x3C : 0xC3, 0, true);

        const int64_t tail_dev = llama_kpool_set_input(cells_of, from_views ? &views : nullptr, nullptr, dev.pool_cells, nullptr,
                dev.pool_bias, nullptr, nullptr, dev.pool_reps, dev.new_pool_cells, dev.new_pool_reps, strm_of.data(),
                w.v_cells[0].size(), rebuild, ub, cfg.r, dev.mask_cells, dev.mask_rows, dev.tail_cells);

        std::vector<ggml_fp16_t> sel;
        std::vector<ggml_fp16_t> cand;
        op_masks(dev, cfg.r*n_dump, sel, cand);

        char sub[200];
        bool ok = true;

        if (tail_dev != tail_ref) {
            snprintf(sub, sizeof(sub), "tail: %lld cells, the host build %lld", (long long) tail_dev, (long long) tail_ref);
            ok = false;
        } else {
            ok = mask_equal(ref.sel_mask, sel, "sel_mask", sub, sizeof(sub)) && mask_equal(ref.cand_mask, cand, "cand_mask", sub, sizeof(sub)) &&
                 same(dev, ref, sub, sizeof(sub));
        }

        for (int64_t row = 0; row < n_tokens; ++row) {
            std::vector<int32_t> got;
            const int32_t * ids = (const int32_t *) dev.tail_cells->data + row*n_kv;
            bool padded = false;
            for (int64_t j = 0; j < n_kv; ++j) {
                if (ids[j] == -1) { padded = true; continue; }
                CHECK(!padded, "tail has a cell after padding");
                got.push_back(ids[j]);
            }
            std::sort(got.begin(), got.end());
            std::vector<int32_t> want;
            for (int64_t j = 0; j < n_kv; ++j) {
                if (ggml_fp16_to_fp32(sel[row*(n_kv+cfg.r*n_dump)+j]) == 0.0f) { want.push_back((int32_t) j); }
            }
            CHECK(got == want, "physical tail differs from SEL at row %lld", (long long) row);
        }
        std::vector<uint8_t> facts(ggml_nbytes(dev.mask_cells) + ggml_nbytes(dev.mask_rows) + ggml_nbytes(dev.tail_cells));
        memcpy(facts.data(), dev.mask_cells->data, ggml_nbytes(dev.mask_cells));
        memcpy(facts.data() + ggml_nbytes(dev.mask_cells), dev.mask_rows->data, ggml_nbytes(dev.mask_rows));
        memcpy(facts.data() + ggml_nbytes(dev.mask_cells) + ggml_nbytes(dev.mask_rows), dev.tail_cells->data, ggml_nbytes(dev.tail_cells));

        if (ok && !from_views && facts != facts_views) {
            size_t k = 0;
            while (facts[k] == facts_views[k]) {
                ++k;
            }
            snprintf(sub, sizeof(sub), "the facts from the views and from the cells differ at byte %zu of %zu (mask_cells holds %zu)", k,
                    facts.size(), ggml_nbytes(dev.mask_cells));
            ok = false;
        }

        if (!ok) {
            snprintf(what, n_what, "device build from the %s: %s", from_views ? "views" : "cells", sub);
            return false;
        }

        if (from_views) {
            facts_views = std::move(facts);
        }
    }

    n_device_builds++;

    return true;
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

// The new pools' keys compressed `chunk` at a time (llama_kpool_compress) against all at once, as a rebuild re-emits every
// pool: the same bytes in every pooled row, from a compute buffer of a chunk's size. Returns the buffer's bytes
static size_t compress_pools(int64_t chunk, std::vector<float> & pooled) {
    const int64_t d_idx   = 32;
    const int64_t r       = 4;
    const int64_t n_kv    = 16384;
    const int64_t n_pools = n_kv/r;

    ggml_backend_t cpu = ggml_backend_cpu_init();

    ggml_init_params dp = { 8*ggml_tensor_overhead(), nullptr, true };
    ggml_context * cd = ggml_init(dp);

    ggml_tensor * kbuf  = ggml_new_tensor_4d(cd, GGML_TYPE_F32, d_idx, 3, n_kv, 1); // a cell: key, gate, pooled key
    ggml_tensor * cells = ggml_new_tensor_2d(cd, GGML_TYPE_I32, r*n_pools, 1);
    ggml_tensor * reps  = ggml_new_tensor_1d(cd, GGML_TYPE_I64, n_pools);
    ggml_tensor * ape   = ggml_new_tensor_2d(cd, GGML_TYPE_F32, r, d_idx);

    ggml_backend_buffer_t data = ggml_backend_alloc_ctx_tensors(cd, cpu);

    std::mt19937 rng(20261006);
    std::uniform_real_distribution<float> u(-2.0f, 2.0f);

    std::vector<float> kv(ggml_nelements(kbuf));
    for (int64_t j = 0; j < n_kv; ++j) {
        for (int64_t h = 0; h < 3*d_idx; ++h) {
            kv[j*3*d_idx + h] = h < 2*d_idx ? u(rng) : 0.0f;
        }
    }
    ggml_backend_tensor_set(kbuf, kv.data(), 0, ggml_nbytes(kbuf));

    // a pool's members are r cells anywhere in the cache, its key in the row of the last
    std::vector<int32_t> perm(n_kv);
    for (int64_t j = 0; j < n_kv; ++j) {
        perm[j] = (int32_t) j;
    }
    std::shuffle(perm.begin(), perm.end(), rng);
    std::vector<int64_t> rows(n_pools);
    for (int64_t p = 0; p < n_pools; ++p) {
        rows[p] = perm[p*r + r - 1];
    }
    ggml_backend_tensor_set(cells, perm.data(), 0, ggml_nbytes(cells));
    ggml_backend_tensor_set(reps,  rows.data(), 0, ggml_nbytes(reps));

    std::vector<float> a(r*d_idx);
    for (auto & x : a) {
        x = u(rng);
    }
    ggml_backend_tensor_set(ape, a.data(), 0, ggml_nbytes(ape));

    ggml_init_params gp = { 4096*ggml_tensor_overhead() + ggml_graph_overhead_custom(4096, false), nullptr, true };
    ggml_context * cg = ggml_init(gp);
    ggml_cgraph  * gf = ggml_new_graph_custom(cg, 4096, false);

    ggml_tensor * kg_rows = ggml_view_3d(cg, kbuf, 2*d_idx, n_kv, 1, kbuf->nb[2], kbuf->nb[3], 0);

    llama_kpool_compress(cg, gf, kg_rows, cells, reps, ape, r, chunk,
            [&](ggml_tensor * src, ggml_tensor * idx) {
                ggml_tensor * k2  = ggml_reshape_2d(cg, kbuf, 3*d_idx, n_kv);
                ggml_tensor * dst = ggml_view_2d(cg, k2, d_idx, n_kv, k2->nb[1], ggml_row_size(k2->type, 2*d_idx));
                return ggml_set_rows(cg, dst, src, idx);
            },
            [](ggml_tensor * t, const char * name) {
                ggml_set_name(t, name);
            });

    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
    CHECK(ggml_gallocr_alloc_graph(galloc, gf), "the compression graph did not allocate");
    const size_t size = ggml_gallocr_get_buffer_size(galloc, 0);

    ggml_backend_graph_compute(cpu, gf);

    ggml_backend_tensor_get(kbuf, kv.data(), 0, ggml_nbytes(kbuf));
    pooled.assign(n_pools*d_idx, 0.0f);
    for (int64_t p = 0; p < n_pools; ++p) {
        std::copy_n(kv.data() + rows[p]*3*d_idx + 2*d_idx, d_idx, pooled.data() + p*d_idx);
    }

    ggml_gallocr_free(galloc);
    ggml_free(cg);
    ggml_backend_buffer_free(data);
    ggml_free(cd);
    ggml_backend_free(cpu);

    return size;
}

static void run_compress_chunks() {
    std::vector<float> whole;
    std::vector<float> chunked;

    const size_t size_whole   = compress_pools(4096, whole); // every pool at once: the graph before chunks
    const size_t size_chunked = compress_pools(512,  chunked);

    printf("compressing 4,096 pools: %zu bytes of compute buffer at once, %zu in chunks of 512\n", size_whole, size_chunked);

    CHECK(whole.size() == chunked.size() && memcmp(whole.data(), chunked.data(), whole.size()*sizeof(float)) == 0,
            "the pooled keys compressed in chunks differ from all at once");
    CHECK(std::any_of(whole.begin(), whole.end(), [](float x) { return x != 0.0f; }), "no pooled key was written");
    CHECK(size_chunked*4 < size_whole, "chunks of 512 of 4,096 pools took %zu bytes against %zu at once", size_chunked, size_whole);
}

// which shifts regroup pools, as llama_memory_hybrid::seq_add asks
static void run_shift_regroups() {
    CHECK(!llama_kpool_stale::shift_regroups(4, 8, -1, -4), "whole pools moved by a whole pool");
    CHECK(!llama_kpool_stale::shift_regroups(4, -1, -1, 8), "the whole sequence moved by two pools");
    CHECK( llama_kpool_stale::shift_regroups(4, 9, -1, -5), "a shift by -5");
    CHECK( llama_kpool_stale::shift_regroups(4, 9, -1, -4), "a range that starts inside a pool");
    CHECK( llama_kpool_stale::shift_regroups(4, 8, 10, -4), "a range that ends inside a pool");
}

// a sequence holding 11 more cells at each of its last three positions: the row at the last has a tail of 36 cells, and with
// GLM-5-Next's 2,048 top-k cells (512 pools of 4) its live cells pass the n_kv_max an r - 1 tail sizes
static void run_tail_bound(std::mt19937 & rng) {
    const uint32_t r = 4;

    pool w(256, 1, true, rng);
    w.append(0, 63);

    auto & cells = w.cells_of(0);
    for (llama_pos p = 60; p <= 62; ++p) {
        for (int k = 0; k < 11; ++k) {
            const int64_t i = w.find_empty(0);
            cells.pos_set(i, p);
            cells.seq_add(i, 0);
        }
    }

    ubatch_data ud;
    ud.add(0, 62);
    const llama_ubatch * ub = ud.get();

    const llama_kpool_cells_fn cells_of = [&](llama_seq_id s) -> const llama_kv_cells & { return w.cells_of(s); };

    maps m(256, 1, 1, 1, r, false, true, false, 0, 0);
    const int64_t n_tail = llama_kpool_set_input(cells_of, nullptr, nullptr, m.pool_cells, nullptr, m.pool_bias, m.sel_mask, m.cand_mask,
            nullptr, nullptr, nullptr, nullptr, 0, false, ub, r);
    const int64_t n_bound = llama_kpool_tail_max(cells_of, *ub, r);

    const int64_t n_top_k  = 2048;
    const int64_t n_live   = n_top_k + n_tail;
    const int64_t n_kv_old = llama_kpool_n_kv_max(n_top_k, r - 1);
    const int64_t n_kv_new = llama_kpool_n_kv_max(n_top_k, n_bound);

    printf("a tail of %lld cells: %lld live cells a row, n_kv_max %lld for an r - 1 tail, %lld for the longest tail\n",
            (long long) n_tail, (long long) n_live, (long long) n_kv_old, (long long) n_kv_new);

    CHECK(n_tail == 36 && n_bound == 36, "the tail: %lld cells granted, %lld counted, 36 placed", (long long) n_tail, (long long) n_bound);
    CHECK(n_kv_old < n_live, "an r - 1 tail bound holds %lld live cells: the case cannot fail", (long long) n_live);
    CHECK(n_kv_new >= n_live, "n_kv_max %lld under %lld live cells", (long long) n_kv_new, (long long) n_live);
}

static void run_draft_rounds(std::mt19937 & rng) {
    // Selections follow sequence identity when the live batch shrinks/reorders. A
    // new completed pool or any memory mutation must force fresh scoring.
    int first_refuses = 0, last_refuses = 0;
    for (int round = 0; round < 1000; ++round) {
        llama_kpool_draft draft;
        const int r = 2 + rng()%7;
        const int base = (1 + rng()%30)*r + rng()%r;
        const int other = (1 + rng()%30)*r + rng()%r;
        const int width = r*(1 + rng()%8);
        const int n_kv = 512;
        std::vector<int32_t> a(width), b(width);
        for (int j = 0; j < width; ++j) { a[j] = rng()%128; b[j] = 128 + rng()%128; }
        draft.advance(0);
        if (round % 2) {
            ggml_context * ctx = ggml_init({4*ggml_tensor_overhead(), nullptr, true});
            auto * source = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, width, 2);
            auto * buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_cpu_buffer_type());
            ggml_backend_tensor_set(source, a.data(), 0, width*sizeof(int32_t));
            ggml_backend_tensor_set(source, b.data(), width*sizeof(int32_t), width*sizeof(int32_t));
            // Separate captures model two microbatches. Destroy graph storage immediately.
            draft.capture_backend(2, base, n_kv, source, 0);
            draft.capture_backend(7, other, n_kv, source, 1);
            ggml_backend_buffer_free(buf);
            ggml_free(ctx);
        } else {
            draft.capture(2, base, n_kv, a);
            draft.capture(7, other, n_kv, b);
        }
        for (int step = 1; step < r + 2; ++step) {
            draft.advance(step);
            ubatch_data ud;
            std::vector<int> order = {2, 7};
            std::shuffle(order.begin(), order.end(), rng);
            if (rng()%2) order.resize(1);
            std::vector<bool> row_reuses;
            for (const int seq : order) {
                const int captured = seq == 2 ? base : other;
                ud.add(seq, captured + step);
                // The next pool finishes at this token position. No row may
                // replay a selection made before one of its pools completed.
                const int next_completion = ((captured + 1)/r + 1)*r - 1;
                row_reuses.push_back(captured + step < next_completion);
            }
            const auto & ub = *ud.get();
            const bool expected = std::all_of(row_reuses.begin(), row_reuses.end(), [](bool reuse) { return reuse; });
            if (row_reuses.size() == 2) {
                first_refuses += !row_reuses[0] && row_reuses[1];
                last_refuses  += row_reuses[0] && !row_reuses[1];
            }
            CHECK(draft.can_reuse(ub, n_kv, width, r) == expected, "draft pool completion at step %d, r %d", step, r);
            CHECK(!draft.can_reuse(ub, n_kv + 256, width, r), "draft accepted different KV geometry");
            CHECK(!draft.can_reuse(ub, n_kv, width + r, r), "draft accepted different selection width");
            if (expected && draft.can_reuse(ub, n_kv, width, r)) {
                std::vector<int32_t> got(width*order.size());
                draft.copy(ub, got.data());
                for (size_t i = 0; i < order.size(); ++i) {
                    const auto & selected = order[i] == 2 ? a : b;
                    CHECK(std::equal(selected.begin(), selected.end(), got.begin() + i*width),
                            "sequence %d lost its selection", order[i]);
                }
            }
        }
        draft.advance(0);
        ubatch_data ud;
        ud.add(2, base + 1);
        CHECK(!draft.can_reuse(*ud.get(), n_kv, width, r), "selection escaped its round");
        draft.capture(2, base, n_kv, a);
        draft.advance(1);
        draft.clear();
        CHECK(!draft.can_reuse(*ud.get(), n_kv, width, r), "selection survived a memory mutation");
        draft.capture(2, base, n_kv, a);
        draft.advance(-1);
        CHECK(!draft.can_reuse(*ud.get(), n_kv, width, r), "selection reused outside drafting");
    }
    CHECK(first_refuses > 0 && last_refuses > 0, "generated no mixed pool boundaries in both row orders");
}

// Compare compact indices with the old SEL + scatter + CAND route, including dump and duplicate ids.
static void run_direct_indices(std::mt19937 & rng) {
    constexpr int64_t nk = 64, np = 2, nr = 16, ns = 2, nt = 24, nd = 8, bound = 96;
    ggml_init_params gp = { 16*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(gp);
    ggml_backend_t cpu = ggml_backend_cpu_init();
    ggml_tensor * cells = ggml_new_tensor_4d(ctx, GGML_TYPE_I32, nk, 2, np, ns);
    ggml_tensor * rows = ggml_new_tensor_3d(ctx, GGML_TYPE_I32, 4, nr, ns);
    ggml_tensor * top = ggml_new_tensor_3d(ctx, GGML_TYPE_I32, nt, nr, ns);
    ggml_tensor * tail = ggml_new_tensor_3d(ctx, GGML_TYPE_I32, nk, nr, ns);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, cpu);
    ggml_tensor * sel = ggml_kpool_mask(ctx, cells, rows, GGML_KPOOL_MASK_SEL, nd);
    ggml_tensor * cand = ggml_kpool_mask(ctx, cells, rows, GGML_KPOOL_MASK_CAND, 0);
    ggml_tensor * direct = ggml_kpool_indices(ctx, cells, rows, top, tail, nd, bound);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, sel);
    ggml_build_forward_expand(gf, cand);
    ggml_build_forward_expand(gf, direct);
    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
    CHECK(ggml_gallocr_alloc_graph(alloc, gf), "direct index graph allocation");
    std::vector<int32_t> vc(nk*2*np*ns), vr(4*nr*ns), vt(nt*nr*ns), vl(nk*nr*ns);
    std::vector<int32_t> got(bound*nr*ns);
    std::vector<ggml_fp16_t> vs((nk+nd)*nr*ns), vm(nk*nr*ns);
    for (int round = 0; round < 64; ++round) {
        for (int64_t stream = 0; stream < ns; ++stream) {
            for (int64_t ps = 0; ps < np; ++ps) {
                int32_t * pos = vc.data() + (stream*np+ps)*2*nk;
                for (int64_t j = 0; j < nk; ++j) {
                    pos[j] = rng()%5 == 0 ? -1 : int32_t(rng()%32);
                    pos[nk+j] = rng()%6 == 0 ? -1 : int32_t(rng()%8);
                }
            }
        }
        std::fill(vl.begin(), vl.end(), -1);
        for (int64_t row = 0; row < nr*ns; ++row) {
            int32_t * facts = vr.data()+4*row;
            facts[0] = int32_t(rng()%32);
            facts[1] = row%7 == 0 ? -1 : int32_t(rng()%np);
            facts[2] = (facts[0]+1)/4*4;
            facts[3] = int32_t(rng()%8);
            for (int64_t j = 0; j < nt; ++j) {
                vt[row*nt+j] = int32_t(rng()%(nk+nd));
            }
            if (facts[1] >= 0) {
                const int32_t * pos = vc.data()+((row/nr)*np+facts[1])*2*nk;
                int64_t count = 0;
                for (int64_t j = 0; j < nk; ++j) {
                    if (pos[j] >= facts[2] && pos[j] <= facts[0]) { vl[row*nk+count++] = int32_t(j); }
                }
            }
        }
        ggml_backend_tensor_set(cells, vc.data(), 0, ggml_nbytes(cells));
        ggml_backend_tensor_set(rows, vr.data(), 0, ggml_nbytes(rows));
        ggml_backend_tensor_set(top, vt.data(), 0, ggml_nbytes(top));
        ggml_backend_tensor_set(tail, vl.data(), 0, ggml_nbytes(tail));
        CHECK(ggml_backend_graph_compute(cpu, gf) == GGML_STATUS_SUCCESS, "direct index graph compute");
        ggml_backend_tensor_get(sel, vs.data(), 0, ggml_nbytes(sel));
        ggml_backend_tensor_get(cand, vm.data(), 0, ggml_nbytes(cand));
        ggml_backend_tensor_get(direct, got.data(), 0, ggml_nbytes(direct));
        for (int64_t row = 0; row < nr*ns; ++row) {
            std::vector<int32_t> want;
            for (int64_t j = 0; j < nk; ++j) {
                const bool selected = std::find(vt.begin()+row*nt, vt.begin()+(row+1)*nt, j) != vt.begin()+(row+1)*nt;
                if ((selected || ggml_fp16_to_fp32(vs[row*(nk+nd)+j]) == 0.0f) && ggml_fp16_to_fp32(vm[row*nk+j]) == 0.0f) {
                    want.push_back(int32_t(j));
                }
            }
            want.resize(bound, -1);
            CHECK(std::equal(want.begin(), want.end(), got.begin()+row*bound), "direct indices differ at round %d row %lld", round, (long long) row);
        }
    }
    ggml_gallocr_free(alloc);
    ggml_backend_buffer_free(buf);
    ggml_backend_free(cpu);
    ggml_free(ctx);
    printf("direct indices: 2048 rows match SEL + scatter + CAND\n");
}

static ggml_backend_t resident_test_backend() {
    static ggml_backend_t cpu = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(cpu, 1);
    return cpu;
}

static void check_resident_materialization(pool & w, llama_seq_id seq, const llama_kpool_views::view & v,
        const std::vector<int32_t> & uploaded, int step) {
    auto * cpu = resident_test_backend();
    const int64_t capacity = w.cells_of(seq).size();
    const int64_t nk = std::min<int64_t>(capacity, std::max<int64_t>(32, (w.cells_of(seq).used_max_p1()+31)/32*32));
    const int32_t base = v.n ? w.cells_of(seq).seq_pos_min(seq)/v.r : 0;
    const int32_t run = v.n ? w.cells_of(seq).seq_pos_max(seq)/v.r - base + 1 : 0;
    const int64_t nr = 1 + step%3, np = std::min<int64_t>(v.cap, std::max<int64_t>(2, run + step%3)), nd = 2;
    ubatch_data ud;
    for (int64_t i = 0; i < nr; ++i) ud.add(seq, std::max<int64_t>(0, w.cells_of(seq).seq_pos_max(seq) - nr + 1 + i));
    const auto * ub = ud.get();
    maps ref(nk, 1, nr, 1, v.r, true, false, false, nd, 0xa5, np, true);
    const uint32_t stream = 0;
    llama_kpool_set_input([&](llama_seq_id s) -> const llama_kv_cells & { return w.cells_of(s); }, nullptr,
            nullptr, ref.pool_cells, nullptr, ref.pool_bias, nullptr, nullptr,
            ref.pool_reps, ref.new_pool_cells, ref.new_pool_reps, &stream, capacity, false, ub, v.r,
            ref.mask_cells, ref.mask_rows, ref.tail_cells);
    ggml_context * ctx = ggml_init({24*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true});
    auto * table = v.resident(ggml_backend_cpu_buffer_type())->tensor;
    CHECK(memcmp(table->data, uploaded.data(), ggml_nbytes(table)) == 0, "actual backend uploads differ from shadow");
    auto * params = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 6, nr);
    auto * selected = ggml_new_tensor_3d(ctx, GGML_TYPE_I32, 16, nr, 1);
    auto * tail = ggml_new_tensor_3d(ctx, GGML_TYPE_I32, 3, nr, 1);
    auto * buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_cpu_buffer_type());
    auto * p = (int32_t *) params->data;
    for (int64_t i = 0; i < nr; ++i) {
        p[6*i] = ub->pos[i]; p[6*i+1] = base;
        p[6*i+2] = (ub->pos[i]+1)/v.r*v.r;
        p[6*i+3] = std::max<int32_t>(0, p[6*i+2]/v.r - base);
        p[6*i+4] = v.b_lo; p[6*i+5] = run;
        for (int j = 0; j < 16; ++j) ((int32_t *) selected->data)[16*i+j] = j < 2 ? 0 : j == 2 ? nk : (step*13+j*17)%(nk+nd*v.r);
        std::copy_n((int32_t *) ref.tail_cells->data + i*nk, 3, (int32_t *) tail->data + i*3);
    }
    auto * facts = ggml_kpool_materialize(ctx, table, params, capacity, v.r, nk, np, nd, GGML_KPOOL_MATERIALIZE_FACTS);
    auto * cells = ggml_kpool_materialize(ctx, table, params, capacity, v.r, nk, np, nd, GGML_KPOOL_MATERIALIZE_CELLS);
    auto * reps = ggml_kpool_materialize(ctx, table, params, capacity, v.r, nk, np, nd, GGML_KPOOL_MATERIALIZE_REPS);
    auto * bias = ggml_kpool_materialize(ctx, table, params, capacity, v.r, nk, np, nd, GGML_KPOOL_MATERIALIZE_BIAS);
    // Caller visibility is deliberately independent of materialization params.
    // Invalid partitions/early query bounds must not be replaced by params.
    auto * visibility = (int32_t *) ref.mask_rows->data;
    if (step%5 == 0) visibility[1] = -1;
    if (step%7 == 0) visibility[0] = 0;
    if (step%11 == 0) visibility[3] = 0;
    auto * fused = ggml_kpool_indices(ctx, facts, ref.mask_rows, selected, tail, nd*v.r, 32);
    auto * explicit_facts = ggml_dup(ctx, facts);
    auto * explicit_ids = ggml_kpool_indices(ctx, explicit_facts, ref.mask_rows, selected, tail, nd*v.r, 32);
    auto * reference_ids = ggml_kpool_indices(ctx, ref.mask_cells, ref.mask_rows, selected, tail, nd*v.r, 32);
    auto * graph = ggml_new_graph(ctx);
    for (auto * t : {facts, cells, reps, bias, fused, explicit_ids, reference_ids}) {
        ggml_set_output(t);
        ggml_build_forward_expand(graph, t);
    }
    auto * alloc = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
    CHECK(ggml_gallocr_alloc_graph(alloc, graph), "resident materialization allocate");
    CHECK(ggml_backend_graph_compute(cpu, graph) == GGML_STATUS_SUCCESS, "resident materialization compute");
    for (const auto & tensors : {std::make_pair(facts, ref.mask_cells), std::make_pair(cells, ref.pool_cells),
            std::make_pair(reps, ref.pool_reps), std::make_pair(bias, ref.pool_bias),
            std::make_pair(fused, explicit_ids), std::make_pair(fused, reference_ids)}) {
        CHECK(ggml_nbytes(tensors.first) == ggml_nbytes(tensors.second), "resident materialization output shape");
        CHECK(memcmp(tensors.first->data, tensors.second->data, ggml_nbytes(tensors.first)) == 0,
                "resident materialization/reference mismatch step=%d seq=%d op=%d", step, seq, (int) tensors.first->op);
    }
    ggml_gallocr_free(alloc);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
}

// Apply precisely the ranges destined for the backend, retaining poisoned storage
// across changing run extents and A -> B -> A consumers.
static void run_resident_patches(std::mt19937 & rng) {
    pool w(2048, 3, true, rng);
    llama_kpool_views views;
    std::map<llama_seq_id, std::vector<int32_t>> shadow;
    uint64_t calls = 0, bytes = 0;
    for (int step = 0; step < 4000; ++step) {
        w.random_op();
        views.begin();
        for (llama_seq_id seq = 0; seq < 3; ++seq) {
            auto * v = views.serve(w.cells_of(seq), seq, 4, 2048, 514);
            if (!v) continue;
            auto & dst = shadow[seq];
            dst.resize(v->resident_size(), 0x5a5a5a5a);
            const auto storage = v->resident(ggml_backend_cpu_buffer_type());
            storage->begin_upload(resident_test_backend());
            const size_t emitted = v->emit_patches([&](size_t offset, const int32_t * data, size_t count) {
                CHECK(offset + count <= dst.size(), "resident patch outside storage");
                std::copy_n(data, count, dst.begin() + offset);
                storage->upload(offset, data, count);
                ++calls;
                bytes += count*sizeof(int32_t);
            });
            storage->end_upload();
            CHECK(emitted <= dst.size()*sizeof(int32_t), "patch bytes exceed full initialization");
            size_t at = 0;
            for (const auto * values : { &v->pos_at, &v->pblk, &v->slot }) {
                CHECK(std::equal(values->begin(), values->end(), dst.begin() + at), "resident table differs at step %d", step);
                at += values->size();
            }
            CHECK(std::equal(v->fill.begin(), v->fill.end(), dst.begin() + at), "resident fill differs");
            at += v->fill.size();
            CHECK(std::equal(v->reps.begin(), v->reps.end(), dst.begin() + at), "resident reps differ");
            // Independent canonical reference from authoritative cells, including
            // physical cell zero and incomplete pools; no view table reads.
            std::vector<int32_t> reference(dst.size(), 0);
            const auto & cells = w.cells_of(seq);
            std::fill_n(reference.begin(), 2*cells.size(), -1);
            const size_t slots = 2*cells.size();
            const size_t fills = slots + v->cap*v->r;
            const size_t reps = fills + v->cap;
            for (uint32_t j = 0; j < cells.size(); ++j) {
                if (cells.is_empty(j) || !cells.seq_has(j, seq)) continue;
                const int32_t pos = cells.pos_get(j);
                reference[j] = pos;
                reference[slots + pos - v->b_lo*v->r] = j;
                ++reference[fills + pos/v->r - v->b_lo];
            }
            for (uint32_t b = 0; b < v->cap; ++b) {
                if (reference[fills + b] != (int32_t) v->r) continue;
                reference[reps + b] = reference[slots + b*v->r + v->r - 1];
                for (uint32_t k = 0; k < v->r; ++k) {
                    reference[cells.size() + reference[slots + b*v->r + k]] = v->b_lo + b;
                }
            }
            CHECK(reference == dst, "emitted resident bytes differ from cell reference at step %d", step);
            check_resident_materialization(w, seq, *v, dst, step);
            CHECK(v->emit_patches([](size_t, const int32_t *, size_t) { CHECK(false, "clean view emitted bytes"); }) == 0,
                    "clean view byte count");
        }
    }
    pool steady(4096, 1, true, rng);
    llama_kpool_views sv;
    steady.append(0, 1024);
    for (int step = 0; step < 100; ++step) {
        const auto before = steady.p_next(0);
        steady.append(0, 1 + step%3);
        sv.begin();
        const auto * v = sv.serve(steady.cells_of(0), 0, 4, 4096, 1026);
        CHECK(v, "steady resident view unavailable");
        const auto n = v->emit_patches([](size_t, const int32_t *, size_t) {});
        if (step) CHECK(n <= 256, "small append emitted %zu bytes", n);
        steady.rollback(0, before);
        sv.begin();
        v = sv.serve(steady.cells_of(0), 0, 4, 4096, 1026);
        CHECK(v->emit_patches([](size_t, const int32_t *, size_t) {}) <= 256, "rollback emitted full table");
    }
    printf("resident patches: %llu writes, %llu actual callback bytes\n", (unsigned long long) calls, (unsigned long long) bytes);
}

static void run_draft_bank(std::vector<ggml_backend_dev_t> devs) {
    constexpr uint32_t devices = 4, slots = 9;
    GGML_ASSERT(devs.size() == devices);
    auto * meta = ggml_backend_meta_device(devs.data(), devs.size(),
        [](const ggml_tensor *, void *) -> ggml_backend_meta_split_state {
            return {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
        }, nullptr);
    auto * backend = ggml_backend_dev_init(meta, nullptr);
    const auto buft = ggml_backend_get_default_buffer_type(backend);
    static bool (*copy_async)(ggml_backend_t, ggml_backend_t, const ggml_tensor *, ggml_tensor *);
    static size_t copy_calls, direct_copies;
    copy_async = backend->iface.cpy_tensor_async;
    const auto observe_copy = [](ggml_backend_t src_backend, ggml_backend_t dst_backend,
            const ggml_tensor * src, ggml_tensor * dst) {
        ++copy_calls;
        const bool direct = copy_async(src_backend, dst_backend, src, dst);
        direct_copies += direct;
        return direct;
    };
    llama_kpool_draft draft;
    std::shared_ptr<llama_kpool_backend_tensor> previous;
    for (int round = 0; round < 12; ++round) {
        const int width = round < 6 ? 128 : 192;
        std::vector<llama_seq_id> seqs;
        std::vector<int64_t> counts(devices, 0);
        for (uint32_t d = 0; d < devices; ++d) {
            if (round%2 && d == 1) { continue; }
            for (int s = slots - 1; s >= 0; --s) {
                if (uint32_t(s)%devices == d && (s + round)%3 != 0) { seqs.push_back(s); ++counts[d]; }
            }
        }
        auto * ctx = ggml_init({8*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true});
        auto * source = ggml_new_tensor_3d(ctx, GGML_TYPE_I32, width, 1, seqs.size());
        ggml_backend_meta_split_input(source, 2, counts.data(), counts.size());
        auto * input = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
        std::vector<int32_t> expected(width*seqs.size());
        for (size_t i = 0; i < seqs.size(); ++i) {
            for (int k = 0; k < width; ++k) { expected[i*width + k] = 1000*seqs[i] + 100*round + k; }
        }
        ggml_backend_tensor_set(source, expected.data(), 0, ggml_nbytes(source));
        draft.advance(0);
        copy_calls = direct_copies = 0;
        backend->iface.cpy_tensor_async = observe_copy;
        for (size_t i = 0; i < seqs.size(); ++i) {
            draft.capture_backend(seqs[i], 100, 512, source, i, backend, slots, devices);
        }
        backend->iface.cpy_tensor_async = copy_async;
        CHECK(copy_calls == seqs.size(), "captured %zu rows but observed %zu backend copies", seqs.size(), copy_calls);
        CHECK(direct_copies == copy_calls, "%zu production captures fell back to host staging", copy_calls - direct_copies);
        const auto captured = draft.backend_row(seqs.front());
        int64_t slot = seqs.front()/devices;
        for (uint32_t d = 0; d < uint32_t(seqs.front())%devices; ++d) { slot += (slots + devices - 1 - d)/devices; }
        auto * src_view = ggml_view_1d(ctx, source, width, 0);
        auto * dst_view = ggml_view_1d(ctx, captured->tensor, width, slot*width*sizeof(int32_t));
        CHECK(ggml_backend_view_init(src_view) == GGML_STATUS_SUCCESS, "selection source view");
        CHECK(ggml_backend_view_init(dst_view) == GGML_STATUS_SUCCESS, "selection destination view");
        CHECK(backend->iface.cpy_tensor_async(backend, backend, src_view, dst_view), "owner copy fell back to host staging");
        ggml_backend_synchronize(backend);
        CHECK(captured->buffer->iface.cpy_tensor(captured->buffer, src_view, dst_view), "synchronous owner copy fell back to host staging");
        const uint32_t other_owner = (uint32_t(seqs.front()) + 1)%devices;
        int64_t other_slot = 0;
        for (uint32_t d = 0; d < other_owner; ++d) { other_slot += (slots + devices - 1 - d)/devices; }
        auto * wrong_owner = ggml_view_1d(ctx, captured->tensor, width, other_slot*width*sizeof(int32_t));
        CHECK(ggml_backend_view_init(wrong_owner) == GGML_STATUS_SUCCESS, "different-owner view");
        CHECK(!backend->iface.cpy_tensor_async(backend, backend, src_view, wrong_owner), "copy accepted different ownership");
        ggml_backend_synchronize(backend);
        ggml_backend_buffer_free(input);
        ggml_free(ctx);
        auto bank = draft.backend_row(seqs.front());
        CHECK(bank && bank->tensor->ne[1] == slots, "DP selections did not enter the bank");
        if (!bank) { break; }
        if (previous && previous->tensor->ne[0] == width) { CHECK(bank == previous, "unchanged selection bank was reallocated"); }
        CHECK(ggml_backend_buffer_get_size(bank->buffer) < ggml_nbytes(bank->tensor), "selection bank is mirrored");
        previous = bank;
        draft.advance(1);
        ubatch_data ud;
        for (const auto seq : seqs) { ud.add(seq, 101); }
        CHECK(draft.can_reuse(*ud.get(), 512, width, 4), "DP selection could not be reused");
        std::vector<int32_t> copied(expected.size());
        draft.copy(*ud.get(), copied.data());
        CHECK(copied == expected, "DP selection copy used the wrong sequence row");
        ctx = ggml_init({8*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true});
        auto * ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, seqs.size());
        ggml_backend_meta_split_input(ids, 0, counts.data(), counts.size());
        input = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
        std::vector<int32_t> local_ids;
        for (const auto seq : seqs) { local_ids.push_back(seq/devices); }
        ggml_backend_tensor_set(ids, local_ids.data(), 0, ggml_nbytes(ids));
        auto * selected = ggml_backend_meta_get_rows_local(ctx, bank->tensor, ids);
        auto * graph = ggml_new_graph(ctx);
        ggml_set_output(selected);
        ggml_build_forward_expand(graph, selected);
        auto * alloc = ggml_gallocr_new(buft);
        CHECK(ggml_gallocr_alloc_graph(alloc, graph), "selection bank graph allocation");
        CHECK(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "selection bank graph compute");
        ggml_backend_tensor_get(selected, copied.data(), 0, ggml_nbytes(selected));
        CHECK(copied == expected, "DP graph read the wrong sequence selection");
        draft.clear();
        CHECK(!draft.can_reuse(*ud.get(), 512, width, 4), "DP selection survived memory invalidation");
        ggml_gallocr_free(alloc);
        ggml_backend_buffer_free(input);
        ggml_free(ctx);
    }
    ggml_backend_free(backend);
    printf("draft bank: owner-local copies and graph reads, changing batches and widths\n");
}

// unified: the slots are the sequences of one pool, each a stream of its own in the maps (llama_kpool_shared_streams)
static void run_resident_bank(std::mt19937 & rng, uint32_t slots, uint32_t devices, bool unified = false) {
    constexpr int64_t capacity = 256, r = 4, np = capacity/r + 2, nd = 2;
    GGML_ASSERT(!unified || devices == 1);
    auto * cpu = resident_test_backend();
    std::vector<ggml_backend_dev_t> devs(devices, ggml_backend_get_device(cpu));
    auto * meta = ggml_backend_meta_device(devs.data(), devs.size(),
        [](const ggml_tensor *, void *) -> ggml_backend_meta_split_state {
            return {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
        }, nullptr);
    auto * backend = ggml_backend_dev_init(meta, nullptr);
    const auto buft = ggml_backend_get_default_buffer_type(backend);
    pool w(capacity, slots, unified, rng);
    llama_kpool_views views;
    views.reserve_views(slots);
    const auto bank = views.resident_bank(buft, 2*capacity + np*(r + 2), slots, devices);
    for (uint32_t s = 0; s < slots; ++s) { w.append(s, 16 + s%17); }
    for (int step = 0; step < 12; ++step) {
        const llama_seq_id changed = step%slots;
        const auto saved = w.cells_of(changed);
        const auto before = w.p_next(changed);
        w.append(changed, 1 + step%3);
        if (step%4 == 1) { w.rollback(changed, before); }
        if (step%4 == 2) { w.cells_of(changed) = saved; }
        views.begin();
        bank->begin_upload(backend);
        // a unified pool's changes go with the graph (ggml_kpool_patch) on three steps of four, and as writes on the
        // fourth, the first among them, where every view goes whole
        const bool as_patches = unified && step%4 != 0;
        std::vector<int32_t> patches;
        std::vector<const llama_kpool_views::view *> current(slots);
        for (uint32_t s = 0; s < slots; ++s) {
            const auto * v = views.serve(w.cells_of(s), s, r, capacity, np);
            CHECK(v, "resident bank cannot serve slot %u/%u", s, slots);
            if (!v) { continue; }
            current[s] = v;
            int64_t slot = s/devices;
            for (uint32_t d = 0; d < s%devices; ++d) { slot += (slots + devices - 1 - d)/devices; }
            v->emit_patches([&](size_t off, const int32_t * data, size_t count) {
                if (as_patches) {
                    for (size_t i = 0; i < count; ++i) {
                        patches.insert(patches.end(), {(int32_t) slot, (int32_t) (off + i), data[i]});
                    }
                    return;
                }
                bank->upload(slot*bank->tensor->ne[0] + off, data, count);
            });
        }
        bank->end_upload();
        // patches of no change among them and after them, as a graph sized for more than a decode's changes has; a step
        // that takes its rows back changes nothing, and its patches are those alone
        for (size_t i = 0; i < patches.size(); i += 15) { patches.insert(patches.begin() + i, {-1, 0, 0}); }
        patches.insert(patches.end(), {-1, 0, 0, -1, -1, -1});
        // Reverse the sequence order within each device; leave one device idle on alternating rounds.
        std::vector<llama_seq_id> seqs;
        std::vector<int64_t> counts(devices, 0);
        for (uint32_t d = 0; d < devices; ++d) {
            if (devices > 1 && step%2 && d == devices - 1) { continue; }
            for (int64_t s = slots - 1; s >= 0; --s) {
                if (s%devices == d && current[s] && (s + step)%3 == 0) { seqs.push_back(s); ++counts[d]; }
            }
        }
        const int64_t nr = 1 + step%3, ns = seqs.size();
        ubatch_data ud;
        for (const auto seq : seqs) {
            for (int64_t row = 0; row < nr; ++row) { ud.add(seq, w.p_next(seq) - nr + row); }
        }
        const auto * ub = ud.get();
        maps ref(capacity, ns, nr, 1, r, true, false, false, nd, 0xa5, np, true);
        std::vector<uint32_t> streams(ns, 0);
        llama_kpool_set_input([&](llama_seq_id s) -> const llama_kv_cells & { return w.cells_of(s); }, nullptr,
                nullptr, ref.pool_cells, nullptr, ref.pool_bias, nullptr, nullptr, ref.pool_reps,
                ref.new_pool_cells, ref.new_pool_reps, streams.data(), capacity, false, ub, r,
                ref.mask_cells, ref.mask_rows, ref.tail_cells);
        if (unified && ns > 0) {
            std::vector<llama_seq_id> runs;
            CHECK(llama_kpool_shared_streams(w.cells_of(0), *ub, runs) && runs == seqs, "shared streams: step=%d", step);
            // the pool's packed maps of the same ubatch: a row sees the same pools in the same order, with the same cells
            // and rep, the same tail and the same facts of its sequence's cells
            maps packed(capacity, 1, nr*ns, ns, r, true, false, false, nd, 0x5a, 0, true);
            const int64_t np_packed = packed.pool_bias->ne[0];
            llama_kpool_set_input([&](llama_seq_id s) -> const llama_kv_cells & { return w.cells_of(s); }, nullptr,
                    nullptr, packed.pool_cells, nullptr, packed.pool_bias, nullptr, nullptr, packed.pool_reps,
                    packed.new_pool_cells, packed.new_pool_reps, streams.data(), capacity, false, ub, r,
                    packed.mask_cells, packed.mask_rows, packed.tail_cells);
            for (int64_t i = 0; i < nr*ns; ++i) {
                const int64_t s = i/nr;
                const auto * fp = (const int32_t *) packed.mask_rows->data + 4*i;
                const auto * fs = (const int32_t *) ref.mask_rows->data + 4*i;
                CHECK(fp[0] == fs[0] && fp[1] == s && fs[1] == 0 && fp[2] == fs[2] && fp[3] == fs[3],
                        "packed/streams row facts: step=%d row=%d", step, (int) i);
                CHECK(memcmp((const int32_t *) packed.mask_cells->data + s*2*capacity,
                            (const int32_t *) ref.mask_cells->data + s*2*capacity, 2*capacity*sizeof(int32_t)) == 0,
                        "packed/streams cell facts: step=%d row=%d", step, (int) i);
                CHECK(memcmp((const int32_t *) packed.tail_cells->data + i*capacity,
                            (const int32_t *) ref.tail_cells->data + i*capacity, capacity*sizeof(int32_t)) == 0,
                        "packed/streams tail: step=%d row=%d", step, (int) i);
                std::vector<int32_t> seen_packed, seen_streams;
                for (int64_t p = 0; p < np_packed; ++p) {
                    if (((const float *) packed.pool_bias->data)[i*np_packed + p] == 0.0f) {
                        const auto * cells = (const int32_t *) packed.pool_cells->data + p*r;
                        seen_packed.insert(seen_packed.end(), cells, cells + r);
                        seen_packed.push_back(((const int32_t *) packed.pool_reps->data)[p]);
                    }
                }
                for (int64_t p = 0; p < np; ++p) {
                    if (((const float *) ref.pool_bias->data)[i*np + p] == 0.0f) {
                        const auto * cells = (const int32_t *) ref.pool_cells->data + s*r*(np + nd) + p*r;
                        seen_streams.insert(seen_streams.end(), cells, cells + r);
                        seen_streams.push_back(((const int32_t *) ref.pool_reps->data)[s*np + p]);
                    }
                }
                CHECK(seen_packed == seen_streams, "packed/streams pools: step=%d row=%d %zu/%zu", step, (int) i,
                        seen_packed.size(), seen_streams.size());
            }
        }
        auto * ctx = ggml_init({32*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true});
        auto * rows = ggml_new_tensor_3d(ctx, GGML_TYPE_I32, 7, nr, ns);
        auto * visibility = ggml_new_tensor_3d(ctx, GGML_TYPE_I32, 4, nr, ns);
        auto * selected = ggml_new_tensor_3d(ctx, GGML_TYPE_I32, 16, nr, ns);
        auto * tail = ggml_new_tensor_3d(ctx, GGML_TYPE_I32, 3, nr, ns);
        for (auto * t : {rows, visibility, selected, tail}) {
            ggml_backend_meta_split_input(t, 2, counts.data(), counts.size());
        }
        auto * changes = as_patches ? ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 3, patches.size()/3) : nullptr;
        auto * input = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
        if (changes) { ggml_backend_tensor_set(changes, patches.data(), 0, ggml_nbytes(changes)); }
        auto * table = changes ? ggml_kpool_patch(ctx, bank->tensor, changes) : bank->tensor;
        std::vector<int32_t> params(7*nr*ns);
        for (int64_t s = 0; s < ns; ++s) {
            const auto seq = seqs[s];
            const auto * v = current[seq];
            const int32_t base = w.cells_of(seq).seq_pos_min(seq)/r;
            const int32_t run = w.cells_of(seq).seq_pos_max(seq)/r - base + 1;
            for (int64_t row = 0; row < nr; ++row) {
                const int32_t q = ub->pos[s*nr + row], tail = (q + 1)/r*r;
                const int32_t values[] = {q, base, tail, int32_t(tail/r - base), int32_t(v->b_lo), run, int32_t(seq/devices)};
                std::copy_n(values, 7, params.data() + 7*(s*nr + row));
            }
        }
        ggml_backend_tensor_set(rows, params.data(), 0, ggml_nbytes(rows));
        std::vector<int32_t> picks(16*nr*ns), tails(3*nr*ns), expected(32*nr*ns, -1);
        auto * visible = (int32_t *) ref.mask_rows->data;
        for (int64_t i = 0; i < nr*ns; ++i) {
            if ((i + step)%5 == 0) visible[4*i + 1] = -1;
            if ((i + step)%7 == 0) visible[4*i] = 0;
            for (int j = 0; j < 16; ++j) picks[16*i+j] = j < 2 ? 0 : (i*13+j*17)%(capacity+nd*r);
            std::copy_n((int32_t *) ref.tail_cells->data + i*capacity, 3, tails.data() + 3*i);
            std::vector<int32_t> ids;
            const auto * facts = (int32_t *) ref.mask_cells->data + (i/nr)*2*capacity;
            const auto * f = visible + 4*i;
            if (f[1] == 0) {
                for (int j = 0; j < 19; ++j) {
                    const int32_t cell = j < 16 ? picks[16*i+j] : tails[3*i+j-16];
                    if (cell >= 0 && cell < capacity && uint32_t(facts[cell]) <= uint32_t(f[0]) &&
                            (facts[cell] >= f[2] || (j < 16 && uint32_t(facts[capacity+cell]) < uint32_t(f[3])))) {
                        ids.push_back(cell);
                    }
                }
            }
            std::sort(ids.begin(), ids.end());
            ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
            std::copy(ids.begin(), ids.end(), expected.data() + 32*i);
        }
        ggml_backend_tensor_set(visibility, visible, 0, ggml_nbytes(visibility));
        ggml_backend_tensor_set(selected, picks.data(), 0, ggml_nbytes(selected));
        ggml_backend_tensor_set(tail, tails.data(), 0, ggml_nbytes(tail));
        auto * graph = ggml_new_graph(ctx);
        std::vector<std::pair<ggml_tensor *, ggml_tensor *>> outputs;
        for (const auto & pair : {std::make_pair(GGML_KPOOL_MATERIALIZE_FACTS, ref.mask_cells),
                std::make_pair(GGML_KPOOL_MATERIALIZE_CELLS, ref.pool_cells),
                std::make_pair(GGML_KPOOL_MATERIALIZE_REPS, ref.pool_reps),
                std::make_pair(GGML_KPOOL_MATERIALIZE_BIAS, ref.pool_bias)}) {
            auto * out = ggml_kpool_materialize(ctx, table, rows, capacity, r, capacity, np, nd, pair.first);
            ggml_set_output(out);
            ggml_build_forward_expand(graph, out);
            outputs.emplace_back(out, pair.second);
        }
        auto * indices = ggml_kpool_indices(ctx, outputs.front().first, visibility, selected, tail, nd*r, 32);
        ggml_set_output(indices);
        ggml_build_forward_expand(graph, indices);
        auto * alloc = ggml_gallocr_new(buft);
        CHECK(ggml_gallocr_alloc_graph(alloc, graph), "bank graph allocation");
        CHECK(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "bank graph compute");
        for (const auto & pair : outputs) {
            std::vector<uint8_t> got(ggml_nbytes(pair.first));
            ggml_backend_tensor_get(pair.first, got.data(), 0, got.size());
            CHECK(got.size() == ggml_nbytes(pair.second) && memcmp(got.data(), pair.second->data, got.size()) == 0,
                    "bank/reference mismatch: step=%d slots=%u devices=%u mode=%d", step, slots, devices, pair.first->op_params[0]);
        }
        std::vector<int32_t> actual(expected.size());
        ggml_backend_tensor_get(indices, actual.data(), 0, ggml_nbytes(indices));
        CHECK(actual == expected, "bank direct indices mismatch: step=%d slots=%u", step, slots);
        ggml_gallocr_free(alloc);
        ggml_backend_buffer_free(input);
        ggml_free(ctx);
    }
    ggml_backend_free(backend);
}

int main(int argc, char ** argv) {
    if (argc != 1) {
        if (argc != 6 || strcmp(argv[1], "--draft-bank") != 0) {
            fprintf(stderr, "usage: %s [--draft-bank DEVICE0 DEVICE1 DEVICE2 DEVICE3]\n", argv[0]);
            return 1;
        }
        ggml_backend_load_all();
        std::vector<ggml_backend_dev_t> devs;
        for (int i = 2; i < argc; ++i) {
            auto * dev = ggml_backend_dev_by_name(argv[i]);
            if (!dev) {
                fprintf(stderr, "unknown draft-bank device: %s\n", argv[i]);
                return 1;
            }
            devs.push_back(dev);
        }
        run_draft_bank(devs);
        return n_fail != 0;
    }
    std::mt19937 rng(20260930);

    run_resident_patches(rng);
    run_resident_bank(rng, 40, 4);
    run_resident_bank(rng, 41, 4);
    run_resident_bank(rng, 5, 1, true);
    run_resident_bank(rng, 6, 1, true);
    run_direct_indices(rng);
    run_draft_rounds(rng);
    run_draft_bank(std::vector<ggml_backend_dev_t>(4, ggml_backend_get_device(resident_test_backend())));

    run_shift_regroups();
    run_compress_chunks();
    run_tail_bound(rng);

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

    printf("random rounds: %lld builds with a tail over r - 1 cells, each within llama_kpool_tail_max\n", (long long) n_tail_long);
    printf("random rounds: %lld builds whose masks ggml_kpool_mask built from the device facts\n", (long long) n_device_builds);
    CHECK(n_device_builds > 2000, "only %lld device builds: the facts are not tested", (long long) n_device_builds);
    CHECK(n_tail_long > 0, "no build held a tail over r - 1 cells: the bound is not tested");

    if (n_fail == 0) {
        printf("ok\n");
    }
    return n_fail != 0;
}
