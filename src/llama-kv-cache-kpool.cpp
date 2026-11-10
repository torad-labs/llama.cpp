#include "llama-kv-cache-kpool.h"

#include "llama-batch.h"
#include "llama-host-threads.h"
#include "llama-impl.h"
#include "llama-kv-cache.h"
#include "llama-kv-cells.h"
#include "../ggml/src/ggml-backend-impl.h"

#include <algorithm>
#include <cfloat>
#include <cinttypes>
#include <cmath>
#include <cstring>
#include <mutex>
#include <vector>

bool llama_kpool_needs_scoring(int64_t n_kv, uint32_t n_ctx, uint32_t indexer_top_k, uint32_t kpool) {
    GGML_ASSERT(kpool > 0);
    static const bool legacy = ggml_env_switch("LLAMA_GLM_SHORT_DENSE_LEGACY");
    return (legacy ? int64_t(n_ctx) : n_kv) > int64_t(indexer_top_k) + kpool - 1;
}

uint32_t llama_kpool_n_pools(uint32_t n_kv, uint32_t kpool, uint32_t n_seqs) {
    GGML_ASSERT(kpool > 0);
    GGML_ASSERT(n_seqs > 0);

    return n_kv/kpool + 2*n_seqs;
}

uint32_t llama_kpool_pool_pad(uint32_t n_pad_kv, uint32_t kpool) {
    GGML_ASSERT(kpool > 0);

    return std::max<uint32_t>(64, n_pad_kv/kpool);
}

uint32_t llama_kpool_n_pools_ubatch(const llama_kpool_cells_fn & cells_of, const llama_ubatch & ubatch, uint32_t n_kv,
        uint32_t kpool, uint32_t n_stream, uint32_t indexer_top_k, uint32_t pool_pad) {
    GGML_ASSERT(kpool > 0);
    GGML_ASSERT(pool_pad > 0);
    GGML_ASSERT(n_stream > 0 && ubatch.n_seqs_unq > 0 && ubatch.n_seqs_unq % n_stream == 0);

    const uint32_t n_ps  = ubatch.n_seqs_unq/n_stream;
    const uint32_t whole = llama_kpool_n_pools(n_kv, kpool, n_ps);

    // the worst-case graph is reserved on the full-cache context (init_full), which gives no cells: it keeps the whole cache's
    // pools, so the buffers sized at load hold every later graph. Its ubatch names sequences (ubatch_prepare_reserve points
    // every token at one), so the ubatch alone cannot tell the reserve apart
    if (ubatch.n_tokens == 0 || ubatch.seq_id == nullptr || ubatch.seq_id[0] == nullptr || !cells_of) {
        return whole;
    }

    // a stream's maps pack its sequences' runs one after another (llama_kpool_set_input), and every stream's maps are one
    // width: the longest stream's runs. A run is the one llama_kpool_views::serve takes, from the lowest position to the
    // highest, so it is never shorter than the run the cells give below n_kv
    int64_t runs      = 0;
    int64_t in_stream = 0;

    for (uint32_t k = 0; k < ubatch.n_seqs_unq; ++k) {
        const llama_seq_id     s     = ubatch.seq_id_unq[k];
        const llama_kv_cells & cells = cells_of(s);

        const llama_pos p_min = cells.seq_pos_min(s);
        const llama_pos p_max = cells.seq_pos_max(s);

        if (p_min >= 0 && p_max >= p_min) {
            in_stream += p_max/(llama_pos) kpool - p_min/(llama_pos) kpool + 1;
        }

        if ((k + 1) % n_ps == 0) {
            runs      = std::max(runs, in_stream);
            in_stream = 0;
        }
    }

    // padded so a decode changes its graph's shape once every 64 pools (a prompt step once every pool_pad), and never below
    // what select_k takes, so the dump pools and the sparse attention's width stay as they were
    const int64_t at_least = std::min<int64_t>(whole, indexer_top_k/kpool);
    const int64_t own      = GGML_PAD(runs, (int64_t) pool_pad) + 2*(int64_t) n_ps;

    return (uint32_t) std::min<int64_t>(whole, std::max(at_least, own));
}

uint32_t llama_kpool_select_k(uint32_t n_pools, uint32_t indexer_top_k, uint32_t kpool) {
    GGML_ASSERT(kpool > 0);
    GGML_ASSERT(n_pools > 0);
    GGML_ASSERT(indexer_top_k % kpool == 0 && "indexer_top_k must be a whole number of pools");

    return std::min(n_pools, indexer_top_k/kpool);
}

int64_t llama_kpool_tail_max(const llama_kpool_cells_fn & cells_of, const llama_ubatch & ubatch, uint32_t kpool) {
    const llama_pos r = (llama_pos) kpool;

    int64_t n_tail = r - 1;

    for (uint32_t k = 0; k < ubatch.n_seqs_unq; ++k) {
        const llama_seq_id     s     = ubatch.seq_id_unq[k];
        const llama_kv_cells & cells = cells_of(s);

        // one cell a position: a tail spans at most r - 1 positions
        if (cells.seq_cell_count(s) == cells.seq_pos_distinct(s)) {
            continue;
        }

        for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
            // a reserve ubatch (ubatch_reserve) names no sequence for its tokens and places none
            if (ubatch.seq_id[i] == nullptr || ubatch.seq_id[i][0] != s) {
                continue;
            }

            const llama_pos q = ubatch.pos[i];

            n_tail = std::max<int64_t>(n_tail, cells.seq_pos_count(s, (q + 1)/r*r, q + 1, UINT32_MAX));
        }
    }

    return n_tail;
}

int64_t llama_kpool_n_kv_max(int64_t n_top_k, int64_t n_tail) {
    return GGML_PAD(n_top_k + n_tail, 32);
}

bool llama_kpool_shared_streams(const llama_kv_cells & cells, const llama_ubatch & ubatch, std::vector<llama_seq_id> & seqs) {
    seqs.clear();

    const uint32_t n_seqs = ubatch.n_seqs_unq;

    if (n_seqs == 0 || ubatch.n_tokens == 0 || ubatch.n_tokens % n_seqs != 0 || ubatch.seq_id == nullptr) {
        return false;
    }

    const uint32_t n_tps = ubatch.n_tokens/n_seqs;

    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        // a reserve ubatch names no sequence for its tokens
        if (ubatch.seq_id[i] == nullptr || (ubatch.n_seq_id && ubatch.n_seq_id[i] != 1)) {
            seqs.clear();
            return false;
        }

        const llama_seq_id seq = ubatch.seq_id[i][0];

        if (i % n_tps != 0 ? seq != seqs.back() : std::find(seqs.begin(), seqs.end(), seq) != seqs.end()) {
            seqs.clear();
            return false;
        }
        if (i % n_tps == 0) {
            seqs.push_back(seq);
        }
    }

    // a cell that holds two sequences would be in two streams' maps, and its pooled key written by both
    uint64_t held = 0;
    for (llama_seq_id s = 0; s < LLAMA_MAX_SEQ; ++s) {
        held += cells.seq_cell_count(s);
    }
    if (held != cells.get_used()) {
        seqs.clear();
        return false;
    }

    return true;
}

// sel_mask and cand_mask hold only 0.0f and -INFINITY, so f16 is exact here
template <typename T> struct kpool_mask_of;

template <> struct kpool_mask_of<float> {
    static float from(float v) { return v; }
};

template <> struct kpool_mask_of<ggml_fp16_t> {
    static ggml_fp16_t from(float v) { return ggml_fp32_to_fp16(v); }
};

template <typename T>
static void kpool_mask_fill(T * dst, int64_t n) {
    std::fill(dst, dst + n, kpool_mask_of<T>::from(-INFINITY));
}

static void kpool_mask_fill(char * dst, int64_t n, bool f16) {
    if (f16) {
        kpool_mask_fill((ggml_fp16_t *) dst, n);
    } else {
        kpool_mask_fill((float *) dst, n);
    }
}

// pool_of[j] - b_base is the pool of cell j among the run, or negative when its pool is not complete. Returns the cells the
// tail grants
template <typename T>
static int64_t kpool_mask_row(
                T * cur_sel,
                T * cur_cand,
        const llama_pos * pos_at,
        const int32_t   * pool_of,
          int32_t   b_base,
          int64_t   n_kv,
        llama_pos   q,
        llama_pos   tail_start,
          int64_t   bo_vis) {
    const T v_sel  = kpool_mask_of<T>::from(0.0f);
    const T v_mask = kpool_mask_of<T>::from(-INFINITY);

    int64_t n_tail = 0;

    for (int64_t j = 0; j < n_kv; ++j) {
        const bool vis    = (uint32_t) pos_at [j] <= (uint32_t) q;
        const bool pooled = (uint32_t) (pool_of[j] - b_base) < (uint32_t) bo_vis;
        const bool tail   = pos_at[j] >= tail_start;

        cur_sel [j] = vis && tail   ? v_sel : v_mask;
        cur_cand[j] = vis && (pooled || tail) ? v_sel : v_mask;

        n_tail += vis && tail;
    }

    return n_tail;
}

// The device's facts in place of a partition's rows of both masks (ggml_kpool_mask builds them): row 0 of dst a cell's
// position, row 1 its complete pool counted from the run, kpool_mask_row's pos_at and pool_of - b_base. A pool not
// complete is -1 from the views and from the cells alike, so the two write the same bytes (LLAMA_KPOOL_INPUT_CHECK)
static void kpool_facts_cells(int32_t * dst, const llama_pos * pos_at, const int32_t * pool_of, int32_t b_base, int64_t n_kv) {
    for (int64_t j = 0; j < n_kv; ++j) {
        dst[j]        = pos_at[j];
        dst[n_kv + j] = pool_of[j] < 0 ? -1 : pool_of[j] - b_base;
    }
}

// a query row's facts: its position, its partition, the first position of its tail and the pools it sees
static void kpool_facts_row(int32_t * dst, llama_pos q, int64_t ps, llama_pos tail_start, int64_t bo_vis) {
    dst[0] = q;
    dst[1] = (int32_t) ps;
    dst[2] = tail_start;
    dst[3] = (int32_t) bo_vis;
}

// a padded row, or one no partition takes: masked whole
static void kpool_facts_pad(int32_t * dst, int64_t n_rows) {
    for (int64_t ii = 0; ii < n_rows; ++ii) {
        kpool_facts_row(dst + 4*ii, 0, -1, 0, 0);
    }
}

// The positions of a partition's cells from lo on, in order. A row's tail, its cells at positions [tail_start, q], is what
// kpool_mask_row counts over every cell (a visible cell in the tail): here two searches count it, and lo is the lowest
// tail_start of the partition's rows, so the cells below every tail are left out
struct kpool_tails {
    std::vector<llama_pos> pos;
    std::vector<std::pair<llama_pos, int32_t>> cells;

    void init(const llama_pos * pos_at, int64_t n_kv, llama_pos lo, bool physical) {
        pos.clear();
        cells.clear();
        for (int64_t j = 0; j < n_kv; ++j) {
            if (pos_at[j] >= lo) {
                if (physical) { cells.emplace_back(pos_at[j], (int32_t) j); }
                else { pos.push_back(pos_at[j]); }
            }
        }
        if (physical) { std::sort(cells.begin(), cells.end()); }
        else { std::sort(pos.begin(), pos.end()); }
    }

    int64_t write(llama_pos tail_start, llama_pos q, int32_t * dst = nullptr, int64_t capacity = 0) const {
        if (tail_start > q) { return 0; }
        if (!dst) { return std::upper_bound(pos.begin(), pos.end(), q) - std::lower_bound(pos.begin(), pos.end(), tail_start); }
        const auto first = std::lower_bound(cells.begin(), cells.end(), std::make_pair(tail_start, INT32_MIN));
        const auto last = std::upper_bound(cells.begin(), cells.end(), std::make_pair(q, INT32_MAX));
        const int64_t count = last-first;
        if (dst) {
            GGML_ASSERT(count <= capacity);
            for (auto it = first; it != last; ++it) { *dst++ = it->second; }
        }
        return count;
    }
};

// the lowest first position of a tail among the rows of seq in stream s; INT32_MAX when it has none
static llama_pos kpool_tail_lo(const llama_ubatch * ubatch, int64_t s, int64_t n_tps, llama_seq_id seq, int64_t r) {
    llama_pos lo = std::numeric_limits<llama_pos>::max();
    for (int64_t ii = 0; ii < n_tps; ++ii) {
        const int64_t i = s*n_tps + ii;
        if (ubatch->seq_id[i][0] == seq && ubatch->pos[i] >= 0) {
            lo = std::min<llama_pos>(lo, (ubatch->pos[i] + 1)/r*r);
        }
    }
    return lo;
}

// the cell an unused write slot names: empty, or not the last of its block, and not a cell the real slots write, since a pooled key
// is read at the last cell of a block only. One cell per slot keeps the rows of the write different (set_rows writes them from
// several threads). -1 when none is left
struct kpool_spare_cells {
    const llama_kv_cells & cells;
    llama_pos              r;
    const int64_t *        rows; // the rows of the real slots
    int64_t                n_rows;
    int64_t                row0; // the row of cell 0
    std::vector<int64_t>   written;
    bool                   sorted = false;
    uint32_t               next   = 0;

    int32_t take() {
        if (!sorted) {
            sorted = true;
            for (int64_t i = 0; i < n_rows; ++i) {
                written.push_back(rows[i] - row0);
            }
            std::sort(written.begin(), written.end());
        }

        for (; next < cells.size(); ++next) {
            if ((cells.is_empty(next) || cells.pos_get(next) % r != r - 1) && !std::binary_search(written.begin(), written.end(), (int64_t) next)) {
                return (int32_t) next++;
            }
        }
        return -1;
    }
};

// a stream's write slots after its new pools: [n_new, n_new_max), and the members of a complete pool if it has one
struct kpool_stream_unused {
    int64_t         n_new       = 0;
    const int32_t * any_rep_src = nullptr;
};

// where one stream's maps go: the tensors of llama_kpool_set_input, at this stream's first element
struct kpool_stream_out {
    int32_t * pool_cells = nullptr;
    char    * sel_mask   = nullptr;
    char    * cand_mask  = nullptr;
    float   * pool_bias  = nullptr;
    int32_t * pool_reps  = nullptr; // nullptr with the last three when the key cache is off
    int32_t * new_cells  = nullptr;
    int64_t * new_reps   = nullptr;

    // the device's facts in place of sel_mask and cand_mask (both nullptr then): ggml_kpool_mask's cells and rows
    int32_t * mask_cells = nullptr;
    int32_t * mask_rows  = nullptr;
    int32_t * tail_cells = nullptr;
    int64_t n_tail = 0;

    int64_t strm_row  = 0; // the stream's first row of the key cache
    int64_t n_kv      = 0;
    int64_t n_sel     = 0; // a sel_mask row: n_kv and the dump columns
    int64_t n_tps     = 0;
    int64_t n_pools   = 0;
    int64_t n_new_max = 0;
    int64_t r         = 0;

    bool   mask_f16 = false;
    size_t mask_ts  = 0;

    // set: the stream's unused write slots are left to the caller, which names them for every stream at once
    kpool_stream_unused * unused = nullptr;
};

// the fixed row count means unused slots must name a safe destination: a spare cell each (their members stay cell 0)
static void kpool_fill_unused(kpool_spare_cells & spare, int32_t * new_cells, int64_t * new_reps, int64_t n_new,
        int64_t n_new_max, int64_t r, int64_t strm_row, const int32_t * any_rep_src) {
    for (int64_t p = n_new; p < n_new_max; ++p) {
        if (const int32_t cell = spare.take(); cell >= 0) {
            new_reps[p] = strm_row + cell;
        } else if (any_rep_src) {
            std::copy(any_rep_src, any_rep_src + r, new_cells + p*r);
            new_reps[p] = strm_row + any_rep_src[r - 1];
        } else {
            new_reps[p] = strm_row;
        }
    }
}

// Build one sequence's maps from its view; return the largest tail.
// The caller initializes new_cells, padded mask rows, dump pools and sel_mask's dump columns.
static int64_t kpool_stream_from_view(
        const kpool_stream_out         & o,
        const llama_kpool_views::view  & v,
        const llama_kv_cells           & cells,
              llama_seq_id               seq,
        const llama_ubatch             * ubatch,
              int64_t                    s,
              bool                       rebuild) {
    const int64_t r = o.r;

    // the run is every pool from the lowest position to the highest, and starts at the lowest
    int64_t b_base = 0;
    int64_t n_run  = 0;

    if (v.n > 0) {
        b_base = cells.seq_pos_min(seq)/r;
        n_run  = cells.seq_pos_max(seq)/r - b_base + 1;
    }

    const int64_t off = v.n > 0 ? b_base - v.b_lo : 0;

    GGML_ASSERT(n_run <= o.n_pools && off >= 0 && off + n_run <= (int64_t) v.cap);

    const uint16_t * fill = v.fill.data() + off;

    if (o.pool_cells) {
        std::copy_n(v.slot.data() + off*r, n_run*r, o.pool_cells);
        std::fill(o.pool_cells + n_run*r, o.pool_cells + o.n_pools*r, 0);
    }

    int64_t n_new = 0;

    // the fallback of an unused slot when no spare cell is left: recomputing a complete pool is idempotent
    const int32_t * any_rep_src = nullptr;

    if (o.new_reps) {
        if (o.pool_reps) {
            std::copy_n(v.reps.data() + off, n_run, o.pool_reps);
            std::fill(o.pool_reps + n_run, o.pool_reps + o.n_pools, 0);
        }

        for (int64_t p = 0; p < n_run; ++p) {
            if (fill[p] == r) {
                any_rep_src = v.slot.data() + (off + p)*r;
                break;
            }
        }

        auto emit = [&](int64_t p) {
            // bounded while a sequence's ubatch tokens are a contiguous run (llama-batch.cpp enforces);
            // fail loudly, clamping would serve a stale key
            GGML_ASSERT(n_new < o.n_new_max && "k-pool: more pools completed than the fixed bound");

            const int32_t * member = v.slot.data() + (off + p)*r;

            std::copy(member, member + r, o.new_cells + n_new*r);

            o.new_reps[n_new] = o.strm_row + member[r - 1];

            n_new++;
        };

        if (rebuild) {
            for (int64_t p = 0; p < n_run; ++p) {
                if (fill[p] == r) {
                    emit(p);
                }
            }
        } else {
            // the pools this ubatch's tokens are in, in pool order
            std::vector<int64_t> touched;

            for (int64_t ii = 0; ii < o.n_tps; ++ii) {
                const int64_t i = s*o.n_tps + ii;

                if (ubatch->seq_id[i][0] != seq) {
                    continue;
                }

                const int64_t bo = ubatch->pos[i]/r - b_base;

                if (bo >= 0 && bo < n_run) {
                    touched.push_back(bo);
                }
            }

            std::sort(touched.begin(), touched.end());
            touched.erase(std::unique(touched.begin(), touched.end()), touched.end());

            for (const int64_t p : touched) {
                if (fill[p] == r) {
                    emit(p);
                }
            }
        }
    }

    int64_t n_done     = 0;
    int64_t n_tail_max = 0;

    if (o.mask_cells) {
        kpool_facts_cells(o.mask_cells, v.pos_at.data(), v.pblk.data(), (int32_t) b_base, o.n_kv);
    }

    std::mutex merge;

    // a row a token, split across threads: a thread writes its own rows only, from reads the rows share
    llama_host_rows(o.n_tps, (o.n_sel + o.n_kv)*(int64_t) o.mask_ts + o.n_pools*(int64_t) sizeof(float), [&](int64_t i0, int64_t i1, int) {
        int64_t done     = 0;
        int64_t tail_max = 0;

        for (int64_t ii = i0; ii < i1; ++ii) {
            const int64_t i = s*o.n_tps + ii;

            if (ubatch->seq_id[i][0] != seq) {
                continue;
            }

            const llama_pos q = ubatch->pos[i];

            GGML_ASSERT(q >= 0);

            done++;

            const llama_pos tail_start = (q + 1)/r*r;

            // the reference tests visibility at a pool's LAST member, so a straddled pool drops whole
            const int64_t bo_vis = std::max<int64_t>(0, tail_start/r - b_base);

            int64_t n_tail = 0;

            if (o.mask_rows) {
                kpool_facts_row(o.mask_rows + 4*ii, q, 0, tail_start, bo_vis);

                // A valid view has one cell per position; inspect only the unfinished pool.
                const int64_t lo = std::max<int64_t>(tail_start, v.b_lo*r);
                const int64_t hi = std::min<int64_t>(q, (v.b_lo + v.cap)*r - 1);

                for (int64_t p = lo; p <= hi; ++p) {
                    const int32_t cell = v.slot[p - v.b_lo*r];

                    // Cell 0 is also the empty-slot sentinel.
                    if (v.pos_at[cell] != p) {
                        continue;
                    }

                    if (o.tail_cells) {
                        GGML_ASSERT(n_tail < o.n_tail);
                        o.tail_cells[ii*o.n_tail + n_tail] = cell;
                    }
                    n_tail++;
                }
            } else {
                char * cur_sel  = o.sel_mask  + ii*o.n_sel*o.mask_ts;
                char * cur_cand = o.cand_mask + ii*o.n_kv *o.mask_ts;

                n_tail = o.mask_f16 ?
                    kpool_mask_row((ggml_fp16_t *) cur_sel, (ggml_fp16_t *) cur_cand,
                            v.pos_at.data(), v.pblk.data(), (int32_t) b_base, o.n_kv, q, tail_start, bo_vis) :
                    kpool_mask_row((float *) cur_sel, (float *) cur_cand,
                            v.pos_at.data(), v.pblk.data(), (int32_t) b_base, o.n_kv, q, tail_start, bo_vis);
            }

            tail_max = std::max(tail_max, n_tail);

            if (o.pool_bias) {
                float * q_pool_bias = o.pool_bias + ii*o.n_pools;
                const int64_t n_vis = std::min(n_run, bo_vis);
                for (int64_t p = 0; p < n_vis; ++p) {
                    q_pool_bias[p] = fill[p] == r ? 0.0f : -INFINITY;
                }
                std::fill(q_pool_bias + n_vis, q_pool_bias + o.n_pools, -INFINITY);
            }
        }

        std::lock_guard<std::mutex> lock(merge);
        n_done    += done;
        n_tail_max = std::max(n_tail_max, tail_max);
    });

    // exactly one partition per row, or a query reads another sequence's pools
    GGML_ASSERT(n_done == o.n_tps && "every query must belong to a sequence of the ubatch");

    if (o.new_reps && o.unused) {
        o.unused->n_new       = n_new;
        o.unused->any_rep_src = any_rep_src;
    } else if (o.new_reps) {
        kpool_spare_cells spare = { cells, (llama_pos) r, o.new_reps, n_new, o.strm_row, {} };

        kpool_fill_unused(spare, o.new_cells, o.new_reps, n_new, o.n_new_max, r, o.strm_row, any_rep_src);
    }

    return n_tail_max;
}

int64_t llama_kpool_set_input(
        const llama_kpool_cells_fn & cells_of,
              llama_kpool_views    * views,
              ggml_tensor    * cell_pool,
              ggml_tensor    * pool_cells,
              ggml_tensor    * bias,
              ggml_tensor    * pool_bias,
              ggml_tensor    * sel_mask,
              ggml_tensor    * cand_mask,
              ggml_tensor    * pool_reps,
              ggml_tensor    * new_pool_cells,
              ggml_tensor    * new_pool_reps,
        const uint32_t       * strm_of,
              int64_t          kv_size,
              bool             rebuild,
        const llama_ubatch   * ubatch,
              uint32_t         kpool,
              ggml_tensor    * mask_cells,
              ggml_tensor    * mask_rows,
              ggml_tensor    * tail_cells) {
    GGML_ASSERT(cells_of != nullptr);
    GGML_ASSERT(kpool > 0);

    // the masks, or the device's facts for them
    const bool facts = mask_cells != nullptr;

    GGML_ASSERT((mask_rows != nullptr) == facts);
    GGML_ASSERT((sel_mask == nullptr && cand_mask == nullptr) == facts && "sel_mask and cand_mask, or mask_cells and mask_rows");

    GGML_ASSERT(ggml_backend_buffer_is_host(pool_cells->buffer));
    GGML_ASSERT(ggml_backend_buffer_is_host(pool_bias ->buffer));

    GGML_ASSERT(pool_cells->type == GGML_TYPE_I32);
    GGML_ASSERT(pool_bias ->type == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(pool_cells));
    GGML_ASSERT(ggml_is_contiguous(pool_bias));

    if (facts) {
        for (const ggml_tensor * t : { mask_cells, mask_rows }) {
            GGML_ASSERT(ggml_backend_buffer_is_host(t->buffer));
            GGML_ASSERT(t->type == GGML_TYPE_I32);
            GGML_ASSERT(ggml_is_contiguous(t));
        }
    } else {
        GGML_ASSERT(ggml_backend_buffer_is_host(sel_mask  ->buffer));
        GGML_ASSERT(ggml_backend_buffer_is_host(cand_mask ->buffer));

        GGML_ASSERT((sel_mask->type == GGML_TYPE_F16 || sel_mask->type == GGML_TYPE_F32) &&
                "sel_mask must be f16 or f32");
        GGML_ASSERT(cand_mask->type == sel_mask->type && "both masks must have the KQ mask's type");

        GGML_ASSERT(ggml_is_contiguous(sel_mask));
        GGML_ASSERT(ggml_is_contiguous(cand_mask));
    }

    const int64_t n_kv     = facts ? mask_cells->ne[0] : cand_mask->ne[0];
    const int64_t n_ns     = facts ? mask_cells->ne[3] : sel_mask->ne[3];
    const int64_t r        = kpool;
    const int64_t n_tokens = ubatch->n_tokens;

    // [TAG_KPOOL_SEQ_PARTITION] positions are unambiguous only within one sequence, so one
    // pool map per SEQUENCE, not per stream
    GGML_ASSERT(n_ns == 1 || (int64_t) ubatch->n_seqs_unq == n_ns);

    const int64_t n_ps    = (int64_t) ubatch->n_seqs_unq/n_ns;
    const int64_t n_pools = pool_bias->ne[0];
    const int64_t n_dump  = pool_cells->ne[0]/r - n_pools;
    const int64_t n_sel   = facts ? n_kv + r*n_dump : sel_mask->ne[0];

    GGML_ASSERT(n_ps > 0 && (int64_t) ubatch->n_seqs_unq == n_ns*n_ps);
    GGML_ASSERT(pool_cells->ne[0] % r == 0);
    GGML_ASSERT(n_pools >= 2*n_ps);
    GGML_ASSERT(n_dump >= 0 && n_sel == n_kv + r*n_dump && "sel_mask carries kpool dump columns for each dump pool");
    GGML_ASSERT(pool_cells->ne[1] == n_ns);
    if (facts) {
        GGML_ASSERT(mask_cells->ne[1] == 2 && mask_cells->ne[2] == n_ps && "a cell's position and pool, for each sequence of a stream");
        GGML_ASSERT(mask_rows->ne[0] == 4 && mask_rows->ne[2] == n_ns && mask_rows->ne[3] == 1);
    } else {
        GGML_ASSERT(sel_mask->ne[2] == 1);
        GGML_ASSERT(cand_mask->ne[1] == sel_mask->ne[1] && cand_mask->ne[2] == 1 && cand_mask->ne[3] == n_ns);
    }
    GGML_ASSERT(pool_bias->ne[2] == n_ns);
    GGML_ASSERT(n_tokens % n_ns == 0);

    const int64_t n_tps  = n_tokens/n_ns;
    const int64_t n_padq = facts ? mask_rows->ne[1] : sel_mask->ne[1];

    GGML_ASSERT(pool_bias->ne[1] == n_tps);
    GGML_ASSERT(n_padq >= n_tps);
    const int64_t n_tail = tail_cells ? tail_cells->ne[0] : 0;
    if (tail_cells) {
        GGML_ASSERT(facts && tail_cells->type == GGML_TYPE_I32 && ggml_is_contiguous(tail_cells));
        GGML_ASSERT(ggml_backend_buffer_is_host(tail_cells->buffer));
        GGML_ASSERT(tail_cells->ne[1] == n_padq && tail_cells->ne[2] == n_ns && tail_cells->ne[3] == 1);
        std::fill_n((int32_t *) tail_cells->data, ggml_nelements(tail_cells), -1);
    }

    if (cell_pool) {
        GGML_ASSERT(ggml_backend_buffer_is_host(cell_pool->buffer));
        GGML_ASSERT(cell_pool->type == GGML_TYPE_I32);
        GGML_ASSERT(ggml_is_contiguous(cell_pool));
        GGML_ASSERT(cell_pool->ne[0] == n_kv && cell_pool->ne[1] == n_ns);

        // one row per stream, so a shared cell has nowhere to put its second pool
        GGML_ASSERT(n_ps == 1 && "the per-cell pool view needs one sequence per stream");
    }

    if (bias) {
        GGML_ASSERT(ggml_backend_buffer_is_host(bias->buffer));
        GGML_ASSERT(bias->type == GGML_TYPE_F32);
        GGML_ASSERT(ggml_is_contiguous(bias));
        GGML_ASSERT(bias->ne[0] == n_kv && bias->ne[1] == n_tps && bias->ne[2] == n_ns);
    }

    const bool kcache = pool_reps != nullptr;

    GGML_ASSERT((new_pool_cells != nullptr) == kcache);
    GGML_ASSERT((new_pool_reps  != nullptr) == kcache);

    int64_t n_new_max = 0;

    if (kcache) {
        GGML_ASSERT(ggml_backend_buffer_is_host(pool_reps     ->buffer));
        GGML_ASSERT(ggml_backend_buffer_is_host(new_pool_cells->buffer));
        GGML_ASSERT(ggml_backend_buffer_is_host(new_pool_reps ->buffer));

        GGML_ASSERT(pool_reps     ->type == GGML_TYPE_I32);
        GGML_ASSERT(new_pool_cells->type == GGML_TYPE_I32);
        GGML_ASSERT(new_pool_reps ->type == GGML_TYPE_I64);

        GGML_ASSERT(ggml_is_contiguous(pool_reps));
        GGML_ASSERT(ggml_is_contiguous(new_pool_cells));
        GGML_ASSERT(ggml_is_contiguous(new_pool_reps));

        GGML_ASSERT(pool_reps->ne[0] == n_pools && pool_reps->ne[1] == n_ns);
        GGML_ASSERT(new_pool_cells->ne[0] % r == 0 && new_pool_cells->ne[1] == n_ns);

        n_new_max = new_pool_cells->ne[0]/r;

        GGML_ASSERT(new_pool_reps->ne[0] == n_new_max*n_ns);
        GGML_ASSERT(strm_of != nullptr && kv_size > 0);
    }

    int32_t * dst_pool_reps = kcache ? (int32_t *) pool_reps     ->data : nullptr;
    int32_t * dst_new_cells = kcache ? (int32_t *) new_pool_cells->data : nullptr;
    int64_t * dst_new_reps  = kcache ? (int64_t *) new_pool_reps ->data : nullptr;

    int32_t * dst_cell_pool  = cell_pool ? (int32_t *) cell_pool->data : nullptr;
    int32_t * dst_pool_cells = (int32_t *) pool_cells->data;
    float   * dst_bias       = bias ? (float *) bias->data : nullptr;
    float   * dst_pool_bias  = (float   *) pool_bias ->data;
    char    * dst_sel_mask   = facts ? nullptr : (char *) sel_mask ->data;
    char    * dst_cand_mask  = facts ? nullptr : (char *) cand_mask->data;
    int32_t * dst_mask_cells = facts ? (int32_t *) mask_cells->data : nullptr;
    int32_t * dst_mask_rows  = facts ? (int32_t *) mask_rows ->data : nullptr;

    // no mask bytes a row when the device builds the masks
    const bool   mask_f16 = !facts && sel_mask->type == GGML_TYPE_F16;
    const size_t mask_ts  = facts ? 0 : ggml_type_size(sel_mask->type);

    // -1 marks a cell with no usable pool; host side only, never copied into cell_pool
    std::vector<int32_t>   pool_of;
    std::vector<int32_t>   filled;
    std::vector<llama_pos> pos_at;

    std::vector<int64_t> run_off(n_ps);

    int64_t n_tail_max = 0;
    std::vector<int64_t> run_len(n_ps);

    auto seq_of = [&](int64_t s, int64_t ps) {
        return n_ps == 1 ? ubatch->seq_id[s*n_tps][0] : ubatch->seq_id_unq[ps];
    };

    if (views) {
        views->begin();
    }

    for (int64_t s = 0; s < n_ns; ++s) {
        int32_t * cur_pool_cells = dst_pool_cells + s*(r*(n_pools + n_dump));
        char    * cur_sel_mask   = facts ? nullptr : dst_sel_mask  + s*(n_padq*n_sel)*mask_ts;
        char    * cur_cand_mask  = facts ? nullptr : dst_cand_mask + s*(n_padq*n_kv)*mask_ts;
        int32_t * cur_mask_cells = facts ? dst_mask_cells + s*(2*n_kv*n_ps) : nullptr;
        int32_t * cur_mask_rows  = facts ? dst_mask_rows  + s*(4*n_padq)    : nullptr;
        int32_t * cur_tail_cells = tail_cells ? (int32_t *) tail_cells->data + s*n_padq*n_tail : nullptr;
        float   * cur_pool_bias  = dst_pool_bias  + s*(n_tps*n_pools);

        // the dump pools name the dump columns, one cell each, so no two slots of a row name one cell
        for (int64_t c = 0; c < r*n_dump; ++c) {
            cur_pool_cells[r*n_pools + c] = (int32_t) (n_kv + c);
        }

        int32_t * cur_pool_reps = kcache ? dst_pool_reps + s*n_pools        : nullptr;
        int32_t * cur_new_cells = kcache ? dst_new_cells + s*(r*n_new_max)  : nullptr;
        int64_t * cur_new_reps  = kcache ? dst_new_reps  + s*n_new_max      : nullptr;

        int64_t n_new = 0;

        // the fallback of an unused slot when no spare cell is left: recomputing a complete pool is idempotent
        const int32_t * any_rep_src = nullptr;

        if (kcache) {
            std::fill(cur_new_cells, cur_new_cells + r*n_new_max, 0);
        }

        if (facts) {
            // every row padded until its partition writes it; the op masks the dump columns itself
            kpool_facts_pad(cur_mask_rows, n_padq);
        } else {
            // the padded rows whole and a row's dump columns: the rows' first n_kv are kpool_mask_row's
            llama_host_rows(n_padq, (n_sel - n_kv)*(int64_t) mask_ts, [&](int64_t i0, int64_t i1, int) {
                for (int64_t ii = i0; ii < i1; ++ii) {
                    const int64_t j0 = ii < n_tps ? n_kv : 0;

                    kpool_mask_fill(cur_sel_mask + (ii*n_sel + j0)*mask_ts, n_sel - j0, mask_f16);
                }
            });
            kpool_mask_fill(cur_cand_mask + n_tps*n_kv*mask_ts, (n_padq - n_tps)*n_kv, mask_f16);
        }

        // a sequence alone in its stream is served from its view; else (or when the view cannot) from the cells below
        if (views && n_ps == 1 && !cell_pool && !bias) {
            const llama_seq_id seq = seq_of(s, 0);

            if (const auto * v = views->serve(cells_of(seq), seq, r, n_kv, n_pools)) {
                kpool_stream_out o;

                o.pool_cells = cur_pool_cells;
                o.sel_mask   = cur_sel_mask;
                o.cand_mask  = cur_cand_mask;
                o.pool_bias  = cur_pool_bias;
                o.pool_reps  = cur_pool_reps;
                o.new_cells  = cur_new_cells;
                o.new_reps   = cur_new_reps;
                o.mask_cells = cur_mask_cells;
                o.mask_rows  = cur_mask_rows;
                o.tail_cells = cur_tail_cells;
                o.n_tail     = n_tail;
                o.strm_row   = kcache ? (int64_t) strm_of[s]*kv_size : 0;
                o.n_kv       = n_kv;
                o.n_sel      = n_sel;
                o.n_tps      = n_tps;
                o.n_pools    = n_pools;
                o.n_new_max  = n_new_max;
                o.r          = r;
                o.mask_f16   = mask_f16;
                o.mask_ts    = mask_ts;

                n_tail_max = std::max(n_tail_max, kpool_stream_from_view(o, *v, cells_of(seq), seq, ubatch, s, rebuild));

                continue;
            }

            views->get_stats().n_direct++;
        }

        std::fill(cur_pool_cells, cur_pool_cells + r*n_pools, 0);

        llama_host_rows(n_tps, n_pools*(int64_t) sizeof(float), [&](int64_t i0, int64_t i1, int) {
            std::fill(cur_pool_bias + i0*n_pools, cur_pool_bias + i1*n_pools, -INFINITY);
        });

        if (kcache) {
            // a pool with no rep gathers row 0; such a pool is -INFINITY in pool_bias, so discarded
            std::fill(cur_pool_reps, cur_pool_reps + n_pools, 0);
        }

        // the cells named as a rep by the new slots: a byte per cell of the cache, so only on this path, which reads it
        std::vector<uint8_t> emitted(kcache ? kv_size : 0, 0);

        // [TAG_KPOOL_PACK] one packed run per sequence, NOT one full-width table: the indexer
        // scores every slot against every query, which would multiply the score tensor by n_seq_max
        {
            int64_t n_want = 0;

            for (int64_t ps = 0; ps < n_ps; ++ps) {
                const llama_seq_id seq = seq_of(s, ps);
                const auto & cells = cells_of(seq);

                int64_t b_min = 0;
                int64_t b_max = 0;
                bool    found = false;

                for (int64_t j = 0; j < n_kv; ++j) {
                    if (cells.is_empty(j) || !cells.seq_has(j, seq)) {
                        continue;
                    }
                    const int64_t b = cells.pos_get(j)/r;
                    b_min = found ? std::min(b_min, b) : b;
                    b_max = found ? std::max(b_max, b) : b;
                    found = true;
                }

                run_len[ps] = found ? b_max - b_min + 1 : 0;
                n_want += run_len[ps];
            }

            if (n_want > n_pools) {
                int64_t rem = n_pools;

                for (int64_t ps = 0; ps < n_ps; ++ps) {
                    run_len[ps] = std::min(run_len[ps], rem/(n_ps - ps));
                    rem -= run_len[ps];
                }
            }

            int64_t off = 0;
            for (int64_t ps = 0; ps < n_ps; ++ps) {
                run_off[ps] = off;
                off += run_len[ps];
            }

            GGML_ASSERT(off <= n_pools);
        }

        int64_t n_done = 0;

        for (int64_t ps = 0; ps < n_ps; ++ps) {
            const llama_seq_id seq_of_pool = seq_of(s, ps);
            const auto & cells = cells_of(seq_of_pool);

            const int64_t n_run = run_len[ps];

            int32_t * cur_cell_pool   = dst_cell_pool ? dst_cell_pool + s*n_kv : nullptr;
            int32_t * part_pool_cells = cur_pool_cells + run_off[ps]*r;

            pool_of.assign(n_kv, -1);
            filled .assign(n_pools, 0);

            pos_at.resize(n_kv);
            for (int64_t j = 0; j < n_kv; ++j) {
                pos_at[j] = cells.is_empty(j) || !cells.seq_has(j, seq_of_pool) ? -1 : cells.pos_get(j);
            }

            // anchor at the absolute p/kpool (vLLM, SGLang; not HF's valid_keys.argmax(-1)): the only
            // anchor that keeps a pool's identity stable from prefill to the decodes that read it
            int64_t b_base = 0;
            {
                int64_t b_min = 0;
                int64_t b_max = 0;
                bool    found = false;

                for (int64_t j = 0; j < n_kv; ++j) {
                    if (pos_at[j] < 0) {
                        continue;
                    }
                    const int64_t b = pos_at[j]/r;
                    b_min = found ? std::min(b_min, b) : b;
                    b_max = found ? std::max(b_max, b) : b;
                    found = true;
                }

                b_base = std::max(b_min, b_max - (n_run - 1));
            }

            for (int64_t j = 0; j < n_kv; ++j) {
                if (pos_at[j] < 0) {
                    continue;
                }

                const llama_pos p  = pos_at[j];
                const int64_t   bo = p/r - b_base;

                if (bo < 0 || bo >= n_run) {
                    continue;
                }

                pool_of[j] = (int32_t) bo;
                part_pool_cells[bo*r + (p%r)] = (int32_t) j;
                filled[bo]++;
            }

            // pool_valid = grouped_valid_keys.all(-1): the compressor consumes all r keys
            for (int64_t j = 0; j < n_kv; ++j) {
                // != rather than <: two cells claiming one position overwrite each other
                if (pool_of[j] >= 0 && filled[pool_of[j]] != (int32_t) r) {
                    pool_of[j] = -1;
                }
                if (cur_cell_pool) {
                    cur_cell_pool[j] = pool_of[j] < 0 ? 0 : pool_of[j];
                }
            }

            if (kcache) {
                // a complete pool's key lives in the row of its LAST member; a partial pool leaves that
                // slot 0, and cell 0 is a real cell
                for (int64_t p = 0; p < n_run; ++p) {
                    if (filled[p] == (int32_t) r) {
                        cur_pool_reps[run_off[ps] + p] = part_pool_cells[p*r + (r - 1)];

                        if (any_rep_src == nullptr) {
                            any_rep_src = part_pool_cells + p*r;
                        }
                    }
                }

                // touched[] is over the run, so the cost is O(tokens), not O(n_kv)
                std::vector<uint8_t> touched(n_run, 0);

                for (int64_t ii = 0; ii < n_tps; ++ii) {
                    const int64_t i = s*n_tps + ii;

                    if (ubatch->seq_id[i][0] != seq_of_pool) {
                        continue;
                    }

                    const int64_t bo = ubatch->pos[i]/r - b_base;

                    if (bo >= 0 && bo < n_run) {
                        touched[bo] = 1;
                    }
                }

                for (int64_t p = 0; p < n_run; ++p) {
                    if ((!touched[p] && !rebuild) || filled[p] != (int32_t) r) {
                        continue;
                    }

                    // a rep is written once: sequences that share cells, or cells that share a position, give the same rep twice
                    const int32_t rep = part_pool_cells[p*r + (r - 1)];

                    if (emitted[rep]) {
                        continue;
                    }

                    emitted[rep] = 1;

                    // bounded while a sequence's ubatch tokens are a contiguous run (llama-batch.cpp enforces);
                    // fail loudly, clamping would serve a stale key
                    GGML_ASSERT(n_new < n_new_max && "k-pool: more pools completed than the fixed bound");

                    std::copy(part_pool_cells + p*r, part_pool_cells + (p + 1)*r,
                            cur_new_cells + n_new*r);

                    cur_new_reps[n_new] = (int64_t) strm_of[s]*kv_size + rep;

                    n_new++;
                }
            }

            // the device's facts: the partition's cells, and the positions a row's tail is counted from
            kpool_tails tails;

            if (facts) {
                kpool_facts_cells(cur_mask_cells + ps*2*n_kv, pos_at.data(), pool_of.data(), 0, n_kv);
                tails.init(pos_at.data(), n_kv, kpool_tail_lo(ubatch, s, n_tps, seq_of_pool, r), tail_cells != nullptr);
            }

            std::mutex merge;

            // a row a token, split across threads: a thread writes its own rows only, from reads the rows share
            const int64_t row_bytes = (n_sel + n_kv)*(int64_t) mask_ts + (dst_bias ? n_kv : 0)*(int64_t) sizeof(float) +
                n_run*(int64_t) sizeof(float);

            llama_host_rows(n_tps, row_bytes, [&](int64_t i0, int64_t i1, int) {
                int64_t done     = 0;
                int64_t tail_max = 0;

                for (int64_t ii = i0; ii < i1; ++ii) {
                    const int64_t   i = s*n_tps + ii;

                    if (ubatch->seq_id[i][0] != seq_of_pool) {
                        continue;
                    }

                    const llama_pos q = ubatch->pos[i];

                    GGML_ASSERT(q >= 0);

                    done++;

                    const llama_pos tail_start = (q + 1)/r*r;

                    // the reference tests visibility at a pool's LAST member, so a straddled pool drops whole
                    const int64_t bo_vis = std::max<int64_t>(0, tail_start/r - b_base);

                    float * cur_bias = dst_bias ? dst_bias + i*n_kv : nullptr;

                    int64_t n_tail = 0;

                    if (facts) {
                        kpool_facts_row(cur_mask_rows + 4*ii, q, ps, tail_start, bo_vis);

                        n_tail = tails.write(tail_start, q, cur_tail_cells ? cur_tail_cells + ii*tail_cells->ne[0] : nullptr, tail_cells ? tail_cells->ne[0] : 0);
                    } else {
                        char * cur_sel  = cur_sel_mask  + ii*n_sel*mask_ts;
                        char * cur_cand = cur_cand_mask + ii*n_kv *mask_ts;

                        n_tail = mask_f16 ?
                            kpool_mask_row((ggml_fp16_t *) cur_sel, (ggml_fp16_t *) cur_cand,
                                    pos_at.data(), pool_of.data(), 0, n_kv, q, tail_start, bo_vis) :
                            kpool_mask_row((float *) cur_sel, (float *) cur_cand,
                                    pos_at.data(), pool_of.data(), 0, n_kv, q, tail_start, bo_vis);
                    }

                    tail_max = std::max(tail_max, n_tail);

                    if (cur_bias) {
                        for (int64_t j = 0; j < n_kv; ++j) {
                            const bool vis    = (uint32_t) pos_at [j] <= (uint32_t) q;
                            const bool pooled = (uint32_t) pool_of[j] <  (uint32_t) bo_vis;

                            cur_bias[j] = vis && pooled ? 0.0f : -INFINITY;
                        }
                    }

                    float * q_pool_bias = cur_pool_bias + ii*n_pools + run_off[ps];

                    for (int64_t p = 0; p < n_run; ++p) {
                        const bool valid   = filled[p] == (int32_t) r;
                        const bool visible = p < bo_vis;

                        q_pool_bias[p] = valid && visible ? 0.0f : -INFINITY;
                    }
                }

                std::lock_guard<std::mutex> lock(merge);
                n_done    += done;
                n_tail_max = std::max(n_tail_max, tail_max);
            });
        }

        // exactly one partition per row, or a query reads another sequence's pools
        GGML_ASSERT(n_done == n_tps && "every query must belong to a sequence of the ubatch");

        if (kcache) {
            // the fixed row count means unused slots must name a safe destination: a spare cell each (their members stay cell 0)
            kpool_spare_cells spare = { cells_of(seq_of(s, 0)), (llama_pos) r, cur_new_reps, n_new, (int64_t) strm_of[s]*kv_size, {} };

            for (int64_t p = n_new; p < n_new_max; ++p) {
                if (const int32_t cell = spare.take(); cell >= 0) {
                    cur_new_reps[p] = (int64_t) strm_of[s]*kv_size + cell;
                } else if (any_rep_src) {
                    std::copy(any_rep_src, any_rep_src + r, cur_new_cells + p*r);
                    cur_new_reps[p] = (int64_t) strm_of[s]*kv_size + any_rep_src[r - 1];
                } else {
                    cur_new_reps[p] = (int64_t) strm_of[s]*kv_size;
                }
            }
        }
    }

    return n_tail_max;
}

void llama_kpool_compress(
        ggml_context               * ctx,
        ggml_cgraph                * gf,
        ggml_tensor                * kg_rows,
        ggml_tensor                * new_pool_cells,
        ggml_tensor                * new_pool_reps,
        ggml_tensor                * ape,
        int64_t                      r,
        int64_t                      chunk,
        const llama_kpool_write_fn & write,
        const llama_kpool_name_fn  & name) {
    GGML_ASSERT(r > 0 && chunk > 0);
    GGML_ASSERT(kg_rows->ne[0] % 2 == 0);
    GGML_ASSERT(new_pool_cells->ne[0] % r == 0);

    const int64_t d_idx     = kg_rows->ne[0]/2;
    const int64_t n_stream  = new_pool_cells->ne[1];
    const int64_t n_new_max = new_pool_cells->ne[0]/r;

    GGML_ASSERT(kg_rows->ne[2] == n_stream);
    GGML_ASSERT(new_pool_reps->ne[0] == n_new_max*n_stream);
    GGML_ASSERT(ape->ne[0] == r && ape->ne[1] == d_idx);

    for (int64_t p0 = 0; p0 < n_new_max; p0 += chunk) {
        const int64_t n = std::min(chunk, n_new_max - p0);

        // the chunk's members and rep rows, stream by stream; the whole tensors when it is every pool
        ggml_tensor * cells = new_pool_cells;
        ggml_tensor * reps  = new_pool_reps;

        if (n != n_new_max) {
            cells = ggml_view_2d(ctx, new_pool_cells, r*n, n_stream, new_pool_cells->nb[1],
                    ggml_row_size(new_pool_cells->type, r*p0));
            reps  = ggml_view_2d(ctx, new_pool_reps, n, n_stream, ggml_row_size(new_pool_reps->type, n_new_max),
                    ggml_row_size(new_pool_reps->type, p0));

            if (n_stream > 1) {
                cells = ggml_cont(ctx, cells);
                reps  = ggml_cont(ctx, reps);
            }
            reps = ggml_reshape_1d(ctx, reps, n*n_stream);
        }

        ggml_tensor * members = ggml_get_rows(ctx, kg_rows, cells);
        name(members, "indexer_pool_members");

        const size_t nb_mem = members->nb[1];

        ggml_tensor * mem_k = ggml_view_4d(ctx, members, d_idx, r, n, n_stream,
                nb_mem, nb_mem*r, members->nb[2], 0);
        ggml_tensor * mem_g = ggml_view_4d(ctx, members, d_idx, r, n, n_stream,
                nb_mem, nb_mem*r, members->nb[2], d_idx*members->nb[0]);

        // r-way softmaxes over the SLOT axis, so it must be dim 0; ape is added PRE-softmax
        ggml_tensor * keys_t = ggml_cont(ctx, ggml_permute(ctx, mem_k, 1, 0, 2, 3));
        ggml_tensor * gate_t = ggml_cont(ctx, ggml_permute(ctx, mem_g, 1, 0, 2, 3));

        gate_t = ggml_add(ctx, gate_t, ggml_reshape_4d(ctx, ape, r, d_idx, 1, 1));

        // soft_max maps ne2/ne3 to gridDim.y/z (65535 cap); fold into ne1
        ggml_tensor * probs = ggml_soft_max(ctx, ggml_reshape_2d(ctx, gate_t, r, d_idx*n*n_stream));
        probs = ggml_reshape_4d(ctx, probs, r, d_idx, n, n_stream);
        name(probs, "indexer_pool_probs");

        ggml_tensor * pool_new = ggml_sum_rows(ctx, ggml_mul(ctx, keys_t, probs));
        pool_new = ggml_reshape_2d(ctx, pool_new, d_idx, n*n_stream);
        name(pool_new, "indexer_pool_new");

        ggml_build_forward_expand(gf, write(pool_new, reps));
    }
}

void llama_kpool_views::view::rebuild(const llama_kv_cells & c, llama_seq_id seq) {
    built = true;
    ok    = false;
    full_upload = true;
    dirty_cells.clear();
    dirty_pools.clear();

    const uint32_t size = c.size();

    cap = size/r + 2;
    n   = c.seq_cell_count(seq);

    // two cells at one position: a slot holds one cell, so the maps come from the cells
    if (n != c.seq_pos_distinct(seq)) {
        return;
    }

    int64_t b_min = 0;

    if (n > 0) {
        b_min = c.seq_pos_min(seq)/r;

        if (c.seq_pos_max(seq)/r - b_min + 1 > (int64_t) cap) {
            return;
        }
    }

    b_lo = b_min;

    pos_at.assign(size, -1);
    pblk  .assign(size, -1);
    slot  .assign((size_t) cap*r, 0);
    fill  .assign(cap, 0);
    reps  .assign(cap, 0);

    if (n > 0) {
        uint32_t found = 0;

        for (uint32_t j = 0; j < size; ++j) {
            if (c.is_empty(j) || !c.seq_has(j, seq)) {
                continue;
            }

            const llama_pos p = c.pos_get(j);

            pos_at[j]                = p;
            slot[p - b_lo*r]         = (int32_t) j;
            fill[p/r - b_lo]++;

            found++;
        }

        if (found != n) {
            return;
        }

        for (int64_t bi = 0; bi < (int64_t) cap; ++bi) {
            if (fill[bi] == r) {
                for (uint32_t k = 0; k < r; ++k) {
                    pblk[slot[bi*r + k]] = (int32_t) (b_lo + bi);
                }

                reps[bi] = slot[bi*r + r - 1];
            }
        }
    }

    ok = true;
}

bool llama_kpool_views::view::insert(uint32_t j, llama_pos p) {
    const int64_t b  = p/r;
    const int64_t bi = b - b_lo;

    if (bi < 0 || bi >= (int64_t) cap) {
        return false;
    }

    int32_t & at = slot[p - b_lo*r];

    // the slot's cell still holds this position: two cells claim it
    if (pos_at[at] == p) {
        return false;
    }

    if (!full_upload) {
        dirty_cells.push_back(j);
        dirty_pools.push_back((uint32_t) bi);
    }
    at        = (int32_t) j;
    pos_at[j] = p;

    n++;

    if (++fill[bi] == r) {
        for (uint32_t k = 0; k < r; ++k) {
            if (!full_upload) dirty_cells.push_back(slot[bi*r + k]);
            pblk[slot[bi*r + k]] = (int32_t) b;
        }

        reps[bi] = slot[bi*r + r - 1];
    }

    return true;
}

void llama_kpool_views::view::remove(uint32_t j) {
    const llama_pos p  = pos_at[j];
    const int64_t   bi = p/r - b_lo;

    if (!full_upload) {
        dirty_cells.push_back(j);
        dirty_pools.push_back((uint32_t) bi);
    }
    // a complete pool has every slot filled, so each of its cells drops the pool's mark
    if (fill[bi] == r) {
        for (uint32_t k = 0; k < r; ++k) {
            if (!full_upload) dirty_cells.push_back(slot[bi*r + k]);
            pblk[slot[bi*r + k]] = -1;
        }

        reps[bi] = 0;
    }

    slot[p - b_lo*r] = 0;
    fill[bi]--;
    pos_at[j] = -1;

    n--;
}

bool llama_kpool_views::view::apply(const llama_kv_cells & c, llama_seq_id seq, const std::vector<uint32_t> & chg) {
    upd.clear();

    for (const uint32_t j : chg) {
        GGML_ASSERT(j < pos_at.size());

        const llama_pos p = !c.is_empty(j) && c.seq_has(j, seq) ? c.pos_get(j) : -1;

        if (p != pos_at[j]) {
            upd.emplace_back(j, p);
        }
    }

    // every removal first: a cell leaves a position before another takes it, in whatever order they changed
    for (const auto & u : upd) {
        if (pos_at[u.first] >= 0) {
            remove(u.first);
        }
    }

    for (const auto & u : upd) {
        if (u.second >= 0 && !insert(u.first, u.second)) {
            return false;
        }
    }

    // A view can be used only by the legacy path for many evaluations. Bound
    // its pending journal; the next resident consumer then initializes once.
    if (dirty_cells.size() + dirty_pools.size() > pos_at.size()) {
        full_upload = true;
        dirty_cells.clear();
        dirty_pools.clear();
    }
    return true;
}

size_t llama_kpool_views::view::emit_patches(
        const std::function<void(size_t, const int32_t *, size_t)> & write) const {
    GGML_ASSERT(ok);
    size_t bytes = 0;
    auto put = [&](size_t offset, const int32_t * data, size_t count) {
        if (!count) return;
        write(offset, data, count);
        bytes += count*sizeof(int32_t);
    };
    const size_t pool_offset = 2*pos_at.size();
    const size_t fill_offset = pool_offset + slot.size();
    const size_t reps_offset = fill_offset + cap;
    if (full_upload) {
        put(0, pos_at.data(), pos_at.size());
        put(pos_at.size(), pblk.data(), pblk.size());
        put(pool_offset, slot.data(), slot.size());
        const std::vector<int32_t> wide_fill(fill.begin(), fill.end());
        put(fill_offset, wide_fill.data(), wide_fill.size());
        put(reps_offset, reps.data(), reps.size());
    } else {
        auto unique = [](std::vector<uint32_t> & ids) {
            std::sort(ids.begin(), ids.end());
            ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
        };
        unique(dirty_cells);
        unique(dirty_pools);
        // Neighbours go in one write: a decode's cells and the pools they are in come as runs, and each write is a
        // copy of its own to every device. LLAMA_KPOOL_PATCH_RUNS_LEGACY=1: a write a cell and a pool
        static const bool runs_legacy = ggml_env_switch("LLAMA_KPOOL_PATCH_RUNS_LEGACY");
        const auto run_end = [](const std::vector<uint32_t> & ids, size_t a) {
            size_t b = a + 1;
            while (!runs_legacy && b < ids.size() && ids[b] == ids[b - 1] + 1) { ++b; }
            return b;
        };
        for (size_t a = 0, b; a < dirty_cells.size(); a = b) {
            b = run_end(dirty_cells, a);
            const uint32_t j = dirty_cells[a];
            put(j, &pos_at[j], b - a);
            put(pos_at.size() + j, &pblk[j], b - a);
        }
        std::vector<int32_t> wide_fill;
        for (size_t a = 0, b; a < dirty_pools.size(); a = b) {
            b = run_end(dirty_pools, a);
            const uint32_t p = dirty_pools[a];
            put(pool_offset + p*r, &slot[p*r], (b - a)*r);
            wide_fill.assign(fill.begin() + p, fill.begin() + p + (b - a));
            put(fill_offset + p, wide_fill.data(), wide_fill.size());
            put(reps_offset + p, &reps[p], b - a);
        }
    }
    full_upload = false;
    dirty_cells.clear();
    dirty_pools.clear();
    return bytes;
}

void llama_kpool_views::sync(const llama_kv_cells & cells, uint32_t r) {
    uint64_t & at = synced[&cells];

    if (at == call) {
        return;
    }

    at = call;

    // a list past 1/16 of the cells costs more to apply than the views cost to rebuild
    cells.changes().track(std::max<uint32_t>(256, cells.size()/16));

    const bool listed = cells.changes().take(chg);

    std::sort(chg.begin(), chg.end());
    chg.erase(std::unique(chg.begin(), chg.end()), chg.end());

    for (auto & [seq, v] : views) {
        if (v.cells != &cells) {
            continue;
        }

        // no list: the view is stale, and is rebuilt if a stream asks for it
        if (!listed || !v.built || v.r != r) {
            v.built = false;
            continue;
        }

        if (!v.ok || !v.apply(cells, seq, chg)) {
            v.rebuild(cells, seq);
            st.n_rebuilt++;
        }
    }
}

const llama_kpool_views::view * llama_kpool_views::serve(
        const llama_kv_cells & cells, llama_seq_id seq, uint32_t r, int64_t n_kv, int64_t n_pools) {
    sync(cells, r);

    auto it = views.find(seq);

    if (it == views.end()) {
        if (views.size() >= n_views_max) {
            return nullptr;
        }

        it = views.emplace(seq, view()).first;
    }

    view & v = it->second;

    // a view that lost the cells (the list is gone), or disagrees with their count of the sequence
    if (!v.built || v.cells != &cells || v.r != r || (v.ok && v.n != cells.seq_cell_count(seq))) {
        v.cells = &cells;
        v.r     = r;

        v.rebuild(cells, seq);
        st.n_rebuilt++;
    }

    if (!v.ok || (int64_t) cells.used_max_p1() > n_kv) {
        return nullptr;
    }

    if (v.n > 0 && cells.seq_pos_max(seq)/r - cells.seq_pos_min(seq)/r + 1 > n_pools) {
        return nullptr;
    }

    st.n_served++;

    return &v;
}

int64_t llama_kv_cache_set_input_kpool(
        const llama_kv_cache * kv,
              ggml_tensor    * cell_pool,
              ggml_tensor    * pool_cells,
              ggml_tensor    * bias,
              ggml_tensor    * pool_bias,
              ggml_tensor    * sel_mask,
              ggml_tensor    * cand_mask,
              ggml_tensor    * pool_reps,
              ggml_tensor    * new_pool_cells,
              ggml_tensor    * new_pool_reps,
        const uint32_t       * strm_of,
              int64_t          kv_size,
              bool             rebuild,
        const llama_ubatch   * ubatch,
              uint32_t         kpool,
              ggml_tensor    * mask_cells,
              ggml_tensor    * mask_rows,
              ggml_tensor    * tail_cells) {
    GGML_ASSERT(kv != nullptr);

    static const bool legacy = ggml_env_switch("LLAMA_KPOOL_INPUT_LEGACY");
    static const bool check  = ggml_env_switch("LLAMA_KPOOL_INPUT_CHECK");

    const llama_kpool_cells_fn cells_of = [kv](llama_seq_id seq) -> const llama_kv_cells & {
        return kv->get_cells(seq);
    };

    llama_kpool_views * views = legacy ? nullptr : &kv->get_kpool_views();

    const int64_t n_tail = llama_kpool_set_input(cells_of, views,
            cell_pool, pool_cells, bias, pool_bias, sel_mask, cand_mask,
            pool_reps, new_pool_cells, new_pool_reps, strm_of, kv_size, rebuild, ubatch, kpool, mask_cells, mask_rows, tail_cells);

    if (!check || !views) {
        return n_tail;
    }

    // the same maps from the cells; the views must have written the same bytes
    ggml_tensor * outs[] = {
        cell_pool, pool_cells, bias, pool_bias, sel_mask, cand_mask, pool_reps, new_pool_cells, new_pool_reps, mask_cells, mask_rows, tail_cells,
    };

    std::vector<std::vector<uint8_t>> kept;

    for (const ggml_tensor * t : outs) {
        kept.emplace_back(t ? ggml_nbytes(t) : 0);

        if (t) {
            memcpy(kept.back().data(), t->data, ggml_nbytes(t));
        }
    }

    llama_kpool_set_input(cells_of, nullptr,
            cell_pool, pool_cells, bias, pool_bias, sel_mask, cand_mask,
            pool_reps, new_pool_cells, new_pool_reps, strm_of, kv_size, rebuild, ubatch, kpool, mask_cells, mask_rows, tail_cells);

    for (size_t i = 0; i < kept.size(); ++i) {
        if (outs[i] == nullptr) {
            continue;
        }

        const uint8_t * now = (const uint8_t *) outs[i]->data;

        for (size_t k = 0; k < kept[i].size(); ++k) {
            if (kept[i][k] != now[k]) {
                GGML_ABORT("k-pool input: the view maps and the maps from the cells differ in %s at byte %zu",
                        ggml_get_name(outs[i]), k);
            }
        }
    }

    static uint64_t n_checked = 0;

    if (++n_checked % 512 == 0) {
        const auto & st = views->get_stats();

        LLAMA_LOG_WARN("%s: k-pool input checked %" PRIu64 " calls: %" PRIu64 " streams from views, %" PRIu64 " views rebuilt, %" PRIu64 " from the cells\n",
                __func__, n_checked, st.n_served, st.n_rebuilt, st.n_direct);
    }

    return n_tail;
}

void llm_graph_input_kpool::input_lost() {
    const auto * kv = mctx_attn->get_kv();
    if (resident_patches) { kv->get_kpool_views().resend(); }
    // the pools set_input said were re-emitted were not: their sequences are stale again
    for (const llama_seq_id seq : rebuilt_seqs) { kv->set_kpool_dirty(seq); }
    rebuilt_seqs.clear();
}

llama_kpool_backend_tensor::llama_kpool_backend_tensor(ggml_backend_buffer_type_t buft, int64_t count, int64_t slots, uint32_t devices) {
    ctx = ggml_init({ggml_tensor_overhead(), nullptr, true});
    tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, count, slots);
    ggml_set_name(tensor, "kpool_owned_metadata");
    if (devices > 1) {
        std::vector<int64_t> counts(devices);
        for (uint32_t d = 0; d < devices; ++d) { counts[d] = (slots + devices - 1 - d)/devices; }
        ggml_backend_meta_split_input(tensor, 1, counts.data(), counts.size());
    }
    buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
    GGML_ASSERT(buffer);
}

llama_kpool_backend_tensor::~llama_kpool_backend_tensor() {
    if (upload_pending) ggml_backend_event_synchronize(uploaded);
    if (uploaded) ggml_backend_event_free(uploaded);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
}

void llama_kpool_backend_tensor::begin_upload(ggml_backend_t backend) {
    GGML_ASSERT(backend);
    if (upload_pending) {
        ggml_backend_event_synchronize(uploaded);
        upload_pending = false;
    }
    if (upload_backend != backend) {
        if (uploaded) ggml_backend_event_free(uploaded);
        uploaded = ggml_backend_event_new(ggml_backend_get_device(backend));
        upload_backend = backend;
    }
    staging.clear();
}

void llama_kpool_backend_tensor::upload(size_t offset, const int32_t * data, size_t count) {
    staging.emplace_back(data, data + count);
    ggml_backend_tensor_set_async(upload_backend, tensor, staging.back().data(), offset*sizeof(int32_t), count*sizeof(int32_t));
}

void llama_kpool_backend_tensor::end_upload() {
    if (staging.empty()) return;
    if (uploaded) {
        ggml_backend_event_record(uploaded, upload_backend);
        upload_pending = true;
    } else {
        // Backends without events retain an explicit ordered, synchronous fallback.
        ggml_backend_synchronize(upload_backend);
        staging.clear();
    }
}

std::shared_ptr<llama_kpool_backend_tensor> llama_kpool_views::view::resident(ggml_backend_buffer_type_t buft) const {
    if (!storage) {
        storage = std::make_shared<llama_kpool_backend_tensor>(buft, resident_size());
        full_upload = true;
    }
    GGML_ASSERT(ggml_backend_buffer_get_type(storage->buffer) == buft);
    GGML_ASSERT(storage->tensor->ne[0] == (int64_t) resident_size());
    return storage;
}

std::shared_ptr<llama_kpool_backend_tensor> llama_kpool_views::resident_bank(
        ggml_backend_buffer_type_t buft, int64_t count, uint32_t slots, uint32_t devices) {
    if (!bank) { bank = std::make_shared<llama_kpool_backend_tensor>(buft, count, slots, devices); }
    GGML_ASSERT(ggml_backend_buffer_get_type(bank->buffer) == buft);
    GGML_ASSERT(bank->tensor->ne[0] == count && bank->tensor->ne[1] == slots);
    return bank;
}

void llama_kpool_draft::capture_backend(llama_seq_id seq, llama_pos pos, int64_t n_kv,
        ggml_tensor * source, int64_t row_index, ggml_backend_t backend, uint32_t slots, uint32_t devices) {
    if (!active()) return;
    GGML_ASSERT(source->type == GGML_TYPE_I32 && ggml_is_contiguous(source));
    const int64_t width = source->ne[0];
    const auto buft = ggml_backend_buffer_get_type(source->buffer);
    GGML_ASSERT(devices == 1 || ggml_backend_buft_is_meta(buft));
    auto & store = devices > 1 ? bank : stores[seq];
    if (!store || store->tensor->ne[0] != width || ggml_backend_buffer_get_type(store->buffer) != buft) {
        store = std::make_shared<llama_kpool_backend_tensor>(buft, width, devices > 1 ? slots : 1, devices);
    }
    ggml_context * ctx = ggml_init({2*ggml_tensor_overhead(), nullptr, true});
    auto * destination = store->tensor;
    size_t offset = 0;
    if (devices > 1) {
        GGML_ASSERT(seq >= 0 && uint32_t(seq) < slots && store->tensor->ne[1] == slots);
        const uint32_t owner = uint32_t(seq)%devices;
        int64_t bank_slot = seq/devices;
        for (uint32_t d = 0; d < owner; ++d) {
            bank_slot += (slots + devices - 1 - d)/devices;
        }
        offset = bank_slot*width;
        destination = ggml_view_1d(ctx, destination, width, offset*sizeof(int32_t));
        GGML_ASSERT(ggml_backend_view_init(destination) == GGML_STATUS_SUCCESS);
    }
    GGML_ASSERT(row_index >= 0 && row_index < ggml_nrows(source));
    auto * view = ggml_view_1d(ctx, source, width, row_index*width*sizeof(int32_t));
    GGML_ASSERT(ggml_backend_view_init(view) == GGML_STATUS_SUCCESS);
    // Copy before the source graph can reset, ordered on its producing backend.
    if (backend) ggml_backend_tensor_copy_async(backend, backend, view, destination);
    else ggml_backend_tensor_copy(view, destination);
    ggml_free(ctx);
    rows[seq] = {pos, step, n_kv, {}, store, offset};
}

void llama_kpool_draft::capture_backend_rows(const llama_seq_id * seqs, const llama_pos * positions, size_t n, int64_t n_kv,
        ggml_tensor * source, ggml_backend_t backend, uint32_t slots, uint32_t devices) {
    if (devices > 1 || slots == 1 || ggml_env_switch("LLAMA_KPOOL_DRAFT_BANK_LEGACY")) {
        for (size_t i = 0; i < n; ++i) {
            capture_backend(seqs[i], positions[i], n_kv, source, i, backend, slots, devices);
        }
        return;
    }
    if (!active()) return;
    GGML_ASSERT(source->type == GGML_TYPE_I32 && ggml_is_contiguous(source) && (int64_t) n <= ggml_nrows(source));
    const int64_t width = source->ne[0];
    const auto buft = ggml_backend_buffer_get_type(source->buffer);
    if (!bank || bank->tensor->ne[0] != width || bank->tensor->ne[1] != slots || ggml_backend_buffer_get_type(bank->buffer) != buft) {
        bank = std::make_shared<llama_kpool_backend_tensor>(buft, width, slots);
    }
    ggml_context * ctx = ggml_init({2*n*ggml_tensor_overhead(), nullptr, true});
    for (size_t a = 0, b; a < n; a = b) {
        b = a + 1;
        while (b < n && seqs[b] == seqs[b - 1] + 1) { ++b; }
        GGML_ASSERT(seqs[a] >= 0 && uint32_t(seqs[b - 1]) < slots);
        auto * rows_dst = ggml_view_1d(ctx, bank->tensor, width*(b - a), seqs[a]*width*sizeof(int32_t));
        auto * rows_src = ggml_view_1d(ctx, source, width*(b - a), a*width*sizeof(int32_t));
        GGML_ASSERT(ggml_backend_view_init(rows_dst) == GGML_STATUS_SUCCESS);
        GGML_ASSERT(ggml_backend_view_init(rows_src) == GGML_STATUS_SUCCESS);
        // Copy before the source graph can reset, ordered on its producing backend.
        if (backend) ggml_backend_tensor_copy_async(backend, backend, rows_src, rows_dst);
        else ggml_backend_tensor_copy(rows_src, rows_dst);
        for (size_t i = a; i < b; ++i) {
            rows[seqs[i]] = {positions[i], step, n_kv, {}, bank, size_t(seqs[i])*width};
        }
    }
    ggml_free(ctx);
}

std::shared_ptr<llama_kpool_backend_tensor> llama_kpool_draft::backend_row(llama_seq_id seq) const {
    const auto it = rows.find(seq);
    return it == rows.end() ? nullptr : it->second.backend;
}

void llama_kpool_draft::capture(llama_seq_id seq, llama_pos pos, int64_t n_kv, const std::vector<int32_t> & cells) {
    if (active()) rows[seq] = {pos, step, n_kv, cells, nullptr};
}

bool llama_kpool_draft::can_reuse(const llama_ubatch & ubatch, int64_t n_kv, int64_t width, uint32_t kpool) const {
    if (!active() || !kpool || !ubatch.n_tokens || ubatch.n_tokens != ubatch.n_seqs_unq) return false;
    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        if (ubatch.n_seq_id && ubatch.n_seq_id[i] != 1) return false;
        const auto found = rows.find(ubatch.seq_id[i][0]);
        if (found == rows.end()) return false;
        const auto & row = found->second;
        if (row.n_kv != n_kv || row.width() != width || step <= row.step ||
            (int64_t) ubatch.pos[i] != (int64_t) row.pos + step - row.step ||
            (ubatch.pos[i] + 1)/kpool != (row.pos + 1)/kpool) return false;
    }
    return true;
}

void llama_kpool_draft::copy(const llama_ubatch & ubatch, int32_t * dst) const {
    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        const auto & row = rows.at(ubatch.seq_id[i][0]);
        if (row.backend) {
            ggml_backend_tensor_get(row.backend->tensor, dst, row.offset*sizeof(int32_t), row.width()*sizeof(int32_t));
            dst += row.width();
        } else {
            dst = std::copy(row.cells.begin(), row.cells.end(), dst);
        }
    }
}

void llm_graph_input_kpool::set_input(const llama_ubatch * ubatch) {
    rebuilt_seqs.clear();

    // unconditional: the key/gate STORE runs on the dense path too, or cells below n_select
    // would have no indexer state when the first ubatch crosses it
    mctx_idx->set_input_k_idxs(k_idxs, ubatch);

    if (draft_top_k) {
        GGML_ASSERT(draft && draft->can_reuse(*ubatch, mctx_attn->get_n_kv(), draft_top_k->ne[0], kpool));
        if (draft_ids) {
            GGML_ASSERT(ggml_backend_buffer_is_host(draft_ids->buffer));
            auto * ids = (int32_t *) draft_ids->data;
            for (uint32_t i = 0; i < ubatch->n_tokens; ++i) { ids[i] = ubatch->seq_id[i][0]/n_seq_devices; }
        }
        if (draft_rows.empty()) {
            GGML_ASSERT(ggml_backend_buffer_is_host(draft_top_k->buffer));
            draft->copy(*ubatch, (int32_t *) draft_top_k->data);
        }
    }

    if (pool_cells == nullptr) {
        // Raw keys still change. Build every deferred pool when scoring starts.
        for (uint32_t s = 0; s < ubatch->n_seqs_unq; ++s) {
            mctx_attn->get_kv()->set_kpool_dirty(ubatch->seq_id_unq[s]);
        }
        return;
    }

    // A layer may reject direct attention after index generation was accepted. If every layer
    // falls back to a mask, the scheduler does not allocate this unused optional input.
    auto * tail_input = tail_cells && tail_cells->data ? tail_cells : nullptr;

    if (resident_table) {
        auto & views = mctx_attn->get_kv()->get_kpool_views();
        views.begin();
        const bool bank = !resident_seqs.empty();
        GGML_ASSERT(bank ? resident_seqs.size() == ubatch->n_seqs_unq :
                ubatch->n_seqs_unq == 1 && ubatch->seq_id_unq[0] == resident_seq);
        GGML_ASSERT(!shared || (bank && n_seq_devices == 1));
        auto * backend = ggml_backend_sched_get_tensor_backend(sched, resident_table->tensor);
        GGML_ASSERT(backend);
        resident_table->begin_upload(backend);
        // the changes that fit the graph's patches go with them, in the one copy of the input; the rest are written
        int32_t * patch = resident_patches ? (int32_t *) resident_patches->data : nullptr;
        const int64_t n_patch_max = resident_patches ? resident_patches->ne[1] : 0;
        int64_t n_patch = 0;
        const int64_t n_tps = ubatch->n_tokens/ubatch->n_seqs_unq;
        if (tail_input) std::fill_n((int32_t *) tail_input->data, ggml_nelements(tail_input), -1);
        if (!reuse_only) std::fill_n((int32_t *) new_pool_cells->data, ggml_nelements(new_pool_cells), 0);
        // a unified pool's streams write one cache stream: their unused slots are named together, below
        std::vector<kpool_stream_unused> unused(shared && !reuse_only ? ubatch->n_seqs_unq : 0);
        const int64_t shared_row = shared ? (int64_t) mctx_idx->get_strm(0)*mctx_idx->get_kv()->get_size() : 0;
        for (uint32_t s = 0; s < ubatch->n_seqs_unq; ++s) {
            // a unified pool's streams are its rows' runs, whatever the order of the sequences' ids
            const auto seq = shared ? resident_seqs[s] : ubatch->seq_id_unq[s];
            GGML_ASSERT(!bank || resident_seqs[s] == seq);
            const auto & cells = mctx_attn->get_kv()->get_cells(seq);
            const auto * v = views.serve(cells, seq, kpool, mask_cells->ne[0], pool_bias->ne[0]);
            GGML_ASSERT(v);
            int64_t slot = bank ? seq/n_seq_devices : 0;
            int64_t bank_slot = slot;
            for (uint32_t d = 0; bank && d < uint32_t(seq)%n_seq_devices; ++d) {
                bank_slot += (resident_table->tensor->ne[1] + n_seq_devices - 1 - d)/n_seq_devices;
            }
            GGML_ASSERT(bank_slot >= 0 && bank_slot < resident_table->tensor->ne[1]);
            const size_t table_offset = bank_slot*resident_table->tensor->ne[0];
            resident_upload_bytes += v->emit_patches([&](size_t off, const int32_t * data, size_t count) {
                if (n_patch + (int64_t) count <= n_patch_max) {
                    for (size_t i = 0; i < count; ++i, ++n_patch) {
                        patch[3*n_patch + 0] = (int32_t) bank_slot;
                        patch[3*n_patch + 1] = (int32_t) (off + i);
                        patch[3*n_patch + 2] = data[i];
                    }
                    return;
                }
                resident_table->upload(table_offset + off, data, count);
                ++resident_upload_calls;
            });
            const int64_t base = v->n ? cells.seq_pos_min(seq)/kpool : 0;
            const int64_t run = v->n ? cells.seq_pos_max(seq)/kpool - base + 1 : 0;
            auto * params = (int32_t *) resident_rows->data;
            const int width = resident_rows->ne[0];
            for (int64_t row = 0; row < n_tps; ++row) {
                const int64_t i = s*n_tps + row;
                const int32_t tail = (ubatch->pos[i] + 1)/kpool*kpool;
                params[width*i + 0] = ubatch->pos[i];
                params[width*i + 1] = base;
                params[width*i + 2] = tail;
                params[width*i + 3] = std::max<int64_t>(0, tail/kpool - base);
                params[width*i + 4] = v->b_lo;
                params[width*i + 5] = run;
                if (bank) params[width*i + 6] = slot;
            }
            kpool_stream_out o;
            o.mask_rows = (int32_t *) mask_rows->data + s*n_tps*4;
            o.tail_cells = tail_input ? (int32_t *) tail_input->data + s*n_tps*tail_input->ne[0] : nullptr;
            o.n_tail = tail_input ? tail_input->ne[0] : 0;
            o.n_kv = mask_cells->ne[0];
            o.n_pools = pool_bias->ne[0];
            o.n_tps = n_tps;
            o.r = kpool;
            if (!reuse_only) {
                o.new_cells = (int32_t *) new_pool_cells->data + s*new_pool_cells->ne[0];
                o.new_reps = (int64_t *) new_pool_reps->data + s*n_new_max;
                o.n_new_max = n_new_max;
                o.strm_row = shared ? shared_row : (bank ? slot : (int64_t) mctx_idx->get_strm(s))*mctx_idx->get_kv()->get_size();
                o.unused = shared ? &unused[s] : nullptr;
            }
            const auto tail = kpool_stream_from_view(o, *v, cells, seq, ubatch, s, rebuild);
            GGML_ASSERT(tail <= n_tail);
        }
        if (!unused.empty()) {
            // one spare cell a slot over every stream, so the rows of the one write stay different
            auto * new_cells = (int32_t *) new_pool_cells->data;
            auto * new_reps  = (int64_t *) new_pool_reps->data;
            std::vector<int64_t> written;
            for (size_t s = 0; s < unused.size(); ++s) {
                written.insert(written.end(), new_reps + s*n_new_max, new_reps + s*n_new_max + unused[s].n_new);
            }
            kpool_spare_cells spare = { mctx_attn->get_kv()->get_cells(resident_seqs[0]), (llama_pos) kpool,
                written.data(), (int64_t) written.size(), shared_row, {} };
            for (size_t s = 0; s < unused.size(); ++s) {
                kpool_fill_unused(spare, new_cells + s*new_pool_cells->ne[0], new_reps + s*n_new_max, unused[s].n_new,
                        n_new_max, kpool, shared_row, unused[s].any_rep_src);
            }
        }
        resident_patch_count += n_patch;
        for (; n_patch < n_patch_max; ++n_patch) { patch[3*n_patch] = -1; }
        resident_table->end_upload();
        if (rebuild) {
            mctx_attn->get_kv()->clear_kpool_dirty(*ubatch);
            rebuilt_seqs.assign(ubatch->seq_id_unq, ubatch->seq_id_unq + ubatch->n_seqs_unq);
        }
        return;
    }

    std::vector<uint32_t> strm_of;

    if (pool_reps) {
        strm_of.resize(mctx_idx->get_n_stream());

        for (uint32_t s = 0; s < strm_of.size(); ++s) {
            strm_of[s] = n_seq_devices > 1 ? ubatch->seq_id_unq[s]/n_seq_devices : mctx_idx->get_strm(s);
        }
    }

    GGML_ASSERT(ggml_backend_buffer_is_host(pool_dump->buffer) && pool_dump->type == GGML_TYPE_F32 && ggml_is_contiguous(pool_dump));
    std::fill_n((float *) pool_dump->data, ggml_nelements(pool_dump), -FLT_MAX);

    const int64_t n_tail_set = llama_kv_cache_set_input_kpool(
            mctx_attn->get_kv(),
            /* cell_pool */ nullptr, pool_cells, /* bias */ nullptr, pool_bias,
            // the device's facts when it builds the masks; their results are not inputs then
            mask_cells ? nullptr : sel_mask, mask_cells ? nullptr : cand_mask,
            pool_reps, new_pool_cells, new_pool_reps,
            strm_of.empty() ? nullptr : strm_of.data(),
            pool_reps ? (int64_t) mctx_idx->get_kv()->get_size() : 0,
            rebuild,
            ubatch, kpool, mask_cells, mask_rows, tail_input);

    // the sparse attention's n_kv_max counts n_tail tail cells a row: a row granted more would lose cells to its compaction
    GGML_ASSERT(n_tail_set <= n_tail && "k-pool: a row's tail holds more cells than the graph's sparse bound");

    // cleared here, not in build_inp_kpool: a graph built but not evaluated must not clear it. Only this ubatch's
    // sequences were re-emitted, so only theirs
    if (rebuild) {
        mctx_attn->get_kv()->clear_kpool_dirty(*ubatch);
        rebuilt_seqs.assign(ubatch->seq_id_unq, ubatch->seq_id_unq + ubatch->n_seqs_unq);
    }
}
