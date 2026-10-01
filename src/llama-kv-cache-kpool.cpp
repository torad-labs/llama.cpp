#include "llama-kv-cache-kpool.h"

#include "llama-batch.h"
#include "llama-impl.h"
#include "llama-kv-cache.h"
#include "llama-kv-cells.h"

#include <algorithm>
#include <cfloat>
#include <cinttypes>
#include <cmath>
#include <cstring>
#include <vector>

uint32_t llama_kpool_n_pools(uint32_t n_kv, uint32_t kpool, uint32_t n_seqs) {
    GGML_ASSERT(kpool > 0);
    GGML_ASSERT(n_seqs > 0);

    return n_kv/kpool + 2*n_seqs;
}

uint32_t llama_kpool_select_k(uint32_t n_pools, uint32_t indexer_top_k, uint32_t kpool) {
    GGML_ASSERT(kpool > 0);
    GGML_ASSERT(n_pools > 0);
    GGML_ASSERT(indexer_top_k % kpool == 0 && "indexer_top_k must be a whole number of pools");

    return std::min(n_pools, indexer_top_k/kpool);
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

// pool_of[j] - b_base is the pool of cell j among the run, or negative when its pool is not complete
template <typename T>
static void kpool_mask_row(
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

    for (int64_t j = 0; j < n_kv; ++j) {
        const bool vis    = (uint32_t) pos_at [j] <= (uint32_t) q;
        const bool pooled = (uint32_t) (pool_of[j] - b_base) < (uint32_t) bo_vis;
        const bool tail   = pos_at[j] >= tail_start;

        cur_sel [j] = vis && tail   ? v_sel : v_mask;
        cur_cand[j] = vis && (pooled || tail) ? v_sel : v_mask;
    }
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

// where one stream's maps go: the tensors of llama_kpool_set_input, at this stream's first element
struct kpool_stream_out {
    int32_t * pool_cells = nullptr;
    char    * sel_mask   = nullptr;
    char    * cand_mask  = nullptr;
    float   * pool_bias  = nullptr;
    int32_t * pool_reps  = nullptr; // nullptr with the last three when the key cache is off
    int32_t * new_cells  = nullptr;
    int64_t * new_reps   = nullptr;

    int64_t strm_row  = 0; // the stream's first row of the key cache
    int64_t n_kv      = 0;
    int64_t n_sel     = 0; // a sel_mask row: n_kv and the dump columns
    int64_t n_tps     = 0;
    int64_t n_pools   = 0;
    int64_t n_new_max = 0;
    int64_t r         = 0;

    bool   mask_f16 = false;
    size_t mask_ts  = 0;
};

// one sequence alone in its stream, from its view: the bytes the loop over the cells of llama_kpool_set_input writes.
// That loop's caller has zeroed pool_cells, pool_reps, new_cells and the padded mask rows, set pool_bias to -INFINITY, and written
// the dump pools and sel_mask's dump columns
static void kpool_stream_from_view(
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

    std::copy_n(v.slot.data() + off*r, n_run*r, o.pool_cells);

    int64_t n_new = 0;

    // the fallback of an unused slot when no spare cell is left: recomputing a complete pool is idempotent
    const int32_t * any_rep_src = nullptr;

    if (o.pool_reps) {
        std::copy_n(v.reps.data() + off, n_run, o.pool_reps);

        for (int64_t p = 0; p < n_run; ++p) {
            if (fill[p] == r) {
                any_rep_src = o.pool_cells + p*r;
                break;
            }
        }

        auto emit = [&](int64_t p) {
            // bounded while a sequence's ubatch tokens are a contiguous run (llama-batch.cpp enforces);
            // fail loudly, clamping would serve a stale key
            GGML_ASSERT(n_new < o.n_new_max && "k-pool: more pools completed than the fixed bound");

            const int32_t * member = o.pool_cells + p*r;

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

    int64_t n_done = 0;

    for (int64_t ii = 0; ii < o.n_tps; ++ii) {
        const int64_t i = s*o.n_tps + ii;

        if (ubatch->seq_id[i][0] != seq) {
            continue;
        }

        const llama_pos q = ubatch->pos[i];

        GGML_ASSERT(q >= 0);

        n_done++;

        const llama_pos tail_start = (q + 1)/r*r;

        // the reference tests visibility at a pool's LAST member, so a straddled pool drops whole
        const int64_t bo_vis = std::max<int64_t>(0, tail_start/r - b_base);

        char * cur_sel  = o.sel_mask  + ii*o.n_sel*o.mask_ts;
        char * cur_cand = o.cand_mask + ii*o.n_kv *o.mask_ts;

        if (o.mask_f16) {
            kpool_mask_row((ggml_fp16_t *) cur_sel, (ggml_fp16_t *) cur_cand,
                    v.pos_at.data(), v.pblk.data(), (int32_t) b_base, o.n_kv, q, tail_start, bo_vis);
        } else {
            kpool_mask_row((float *) cur_sel, (float *) cur_cand,
                    v.pos_at.data(), v.pblk.data(), (int32_t) b_base, o.n_kv, q, tail_start, bo_vis);
        }

        float * q_pool_bias = o.pool_bias + ii*o.n_pools;

        for (int64_t p = 0, n_vis = std::min(n_run, bo_vis); p < n_vis; ++p) {
            q_pool_bias[p] = fill[p] == r ? 0.0f : -INFINITY;
        }
    }

    // exactly one partition per row, or a query reads another sequence's pools
    GGML_ASSERT(n_done == o.n_tps && "every query must belong to a sequence of the ubatch");

    if (o.pool_reps) {
        // the fixed row count means unused slots must name a safe destination: a spare cell each (their members stay cell 0)
        kpool_spare_cells spare = { cells, (llama_pos) r, o.new_reps, n_new, o.strm_row, {} };

        for (int64_t p = n_new; p < o.n_new_max; ++p) {
            if (const int32_t cell = spare.take(); cell >= 0) {
                o.new_reps[p] = o.strm_row + cell;
            } else if (any_rep_src) {
                std::copy(any_rep_src, any_rep_src + r, o.new_cells + p*r);
                o.new_reps[p] = o.strm_row + any_rep_src[r - 1];
            } else {
                o.new_reps[p] = o.strm_row;
            }
        }
    }
}

void llama_kpool_set_input(
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
              uint32_t         kpool) {
    GGML_ASSERT(cells_of != nullptr);
    GGML_ASSERT(kpool > 0);

    GGML_ASSERT(ggml_backend_buffer_is_host(pool_cells->buffer));
    GGML_ASSERT(ggml_backend_buffer_is_host(pool_bias ->buffer));
    GGML_ASSERT(ggml_backend_buffer_is_host(sel_mask  ->buffer));
    GGML_ASSERT(ggml_backend_buffer_is_host(cand_mask ->buffer));

    GGML_ASSERT(pool_cells->type == GGML_TYPE_I32);
    GGML_ASSERT(pool_bias ->type == GGML_TYPE_F32);
    GGML_ASSERT((sel_mask->type == GGML_TYPE_F16 || sel_mask->type == GGML_TYPE_F32) &&
            "sel_mask must be f16 or f32");
    GGML_ASSERT(cand_mask->type == sel_mask->type && "both masks must have the KQ mask's type");

    GGML_ASSERT(ggml_is_contiguous(pool_cells));
    GGML_ASSERT(ggml_is_contiguous(pool_bias));
    GGML_ASSERT(ggml_is_contiguous(sel_mask));
    GGML_ASSERT(ggml_is_contiguous(cand_mask));

    const int64_t n_kv     = cand_mask->ne[0];
    const int64_t n_sel    = sel_mask->ne[0];
    const int64_t n_ns     = sel_mask->ne[3];
    const int64_t r        = kpool;
    const int64_t n_tokens = ubatch->n_tokens;

    // [TAG_KPOOL_SEQ_PARTITION] positions are unambiguous only within one sequence, so one
    // pool map per SEQUENCE, not per stream
    GGML_ASSERT(n_ns == 1 || (int64_t) ubatch->n_seqs_unq == n_ns);

    const int64_t n_ps    = (int64_t) ubatch->n_seqs_unq/n_ns;
    const int64_t n_pools = pool_bias->ne[0];
    const int64_t n_dump  = pool_cells->ne[0]/r - n_pools;

    GGML_ASSERT(n_ps > 0 && (int64_t) ubatch->n_seqs_unq == n_ns*n_ps);
    GGML_ASSERT(pool_cells->ne[0] % r == 0);
    GGML_ASSERT(n_pools >= 2*n_ps);
    GGML_ASSERT(n_dump >= 0 && n_sel == n_kv + r*n_dump && "sel_mask carries kpool dump columns for each dump pool");
    GGML_ASSERT(pool_cells->ne[1] == n_ns);
    GGML_ASSERT(sel_mask->ne[2] == 1);
    GGML_ASSERT(cand_mask->ne[1] == sel_mask->ne[1] && cand_mask->ne[2] == 1 && cand_mask->ne[3] == n_ns);
    GGML_ASSERT(pool_bias->ne[2] == n_ns);
    GGML_ASSERT(n_tokens % n_ns == 0);

    const int64_t n_tps  = n_tokens/n_ns;
    const int64_t n_padq = sel_mask->ne[1];

    GGML_ASSERT(pool_bias->ne[1] == n_tps);
    GGML_ASSERT(n_padq >= n_tps);

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
    char    * dst_sel_mask   = (char    *) sel_mask  ->data;
    char    * dst_cand_mask  = (char    *) cand_mask ->data;

    const bool   mask_f16 = sel_mask->type == GGML_TYPE_F16;
    const size_t mask_ts  = ggml_type_size(sel_mask->type);

    // -1 marks a cell with no usable pool; host side only, never copied into cell_pool
    std::vector<int32_t>   pool_of;
    std::vector<int32_t>   filled;
    std::vector<llama_pos> pos_at;

    std::vector<int64_t> run_off(n_ps);
    std::vector<int64_t> run_len(n_ps);

    auto seq_of = [&](int64_t s, int64_t ps) {
        return n_ps == 1 ? ubatch->seq_id[s*n_tps][0] : ubatch->seq_id_unq[ps];
    };

    if (views) {
        views->begin();
    }

    for (int64_t s = 0; s < n_ns; ++s) {
        int32_t * cur_pool_cells = dst_pool_cells + s*(r*(n_pools + n_dump));
        char    * cur_sel_mask   = dst_sel_mask   + s*(n_padq*n_sel)*mask_ts;
        char    * cur_cand_mask  = dst_cand_mask  + s*(n_padq*n_kv)*mask_ts;
        float   * cur_pool_bias  = dst_pool_bias  + s*(n_tps*n_pools);

        std::fill(cur_pool_cells, cur_pool_cells + r*n_pools, 0);
        std::fill(cur_pool_bias,  cur_pool_bias  + n_tps*n_pools, -INFINITY);

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

        // the cells named as a rep by the new slots
        std::vector<uint8_t> emitted(kcache ? kv_size : 0, 0);

        if (kcache) {
            // a pool with no rep gathers row 0; such a pool is -INFINITY in pool_bias, so discarded
            std::fill(cur_pool_reps, cur_pool_reps + n_pools, 0);
            std::fill(cur_new_cells, cur_new_cells + r*n_new_max, 0);
        }

        // the padded rows whole and a row's dump columns: the rows' first n_kv are kpool_mask_row's
        for (int64_t ii = 0; ii < n_padq; ++ii) {
            const int64_t j0 = ii < n_tps ? n_kv : 0;

            kpool_mask_fill(cur_sel_mask + (ii*n_sel + j0)*mask_ts, n_sel - j0, mask_f16);
        }
        kpool_mask_fill(cur_cand_mask + n_tps*n_kv*mask_ts, (n_padq - n_tps)*n_kv, mask_f16);

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
                o.strm_row   = kcache ? (int64_t) strm_of[s]*kv_size : 0;
                o.n_kv       = n_kv;
                o.n_sel      = n_sel;
                o.n_tps      = n_tps;
                o.n_pools    = n_pools;
                o.n_new_max  = n_new_max;
                o.r          = r;
                o.mask_f16   = mask_f16;
                o.mask_ts    = mask_ts;

                kpool_stream_from_view(o, *v, cells_of(seq), seq, ubatch, s, rebuild);

                continue;
            }

            views->get_stats().n_direct++;
        }

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

            for (int64_t ii = 0; ii < n_tps; ++ii) {
                const int64_t   i = s*n_tps + ii;

                if (ubatch->seq_id[i][0] != seq_of_pool) {
                    continue;
                }

                const llama_pos q = ubatch->pos[i];

                GGML_ASSERT(q >= 0);

                n_done++;

                const llama_pos tail_start = (q + 1)/r*r;

                // the reference tests visibility at a pool's LAST member, so a straddled pool drops whole
                const int64_t bo_vis = std::max<int64_t>(0, tail_start/r - b_base);

                float * cur_bias = dst_bias ? dst_bias + i*n_kv : nullptr;
                char  * cur_sel  = cur_sel_mask  + ii*n_sel*mask_ts;
                char  * cur_cand = cur_cand_mask + ii*n_kv *mask_ts;

                if (mask_f16) {
                    kpool_mask_row((ggml_fp16_t *) cur_sel, (ggml_fp16_t *) cur_cand,
                            pos_at.data(), pool_of.data(), 0, n_kv, q, tail_start, bo_vis);
                } else {
                    kpool_mask_row((float *) cur_sel, (float *) cur_cand,
                            pos_at.data(), pool_of.data(), 0, n_kv, q, tail_start, bo_vis);
                }

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
}

void llama_kpool_views::view::rebuild(const llama_kv_cells & c, llama_seq_id seq) {
    built = true;
    ok    = false;

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

    at        = (int32_t) j;
    pos_at[j] = p;

    n++;

    if (++fill[bi] == r) {
        for (uint32_t k = 0; k < r; ++k) {
            pblk[slot[bi*r + k]] = (int32_t) b;
        }

        reps[bi] = slot[bi*r + r - 1];
    }

    return true;
}

void llama_kpool_views::view::remove(uint32_t j) {
    const llama_pos p  = pos_at[j];
    const int64_t   bi = p/r - b_lo;

    // a complete pool has every slot filled, so each of its cells drops the pool's mark
    if (fill[bi] == r) {
        for (uint32_t k = 0; k < r; ++k) {
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

    return true;
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

void llama_kv_cache_set_input_kpool(
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
              uint32_t         kpool) {
    GGML_ASSERT(kv != nullptr);

    static const bool legacy = ggml_env_switch("LLAMA_KPOOL_INPUT_LEGACY");
    static const bool check  = ggml_env_switch("LLAMA_KPOOL_INPUT_CHECK");

    const llama_kpool_cells_fn cells_of = [kv](llama_seq_id seq) -> const llama_kv_cells & {
        return kv->get_cells(seq);
    };

    llama_kpool_views * views = legacy ? nullptr : &kv->get_kpool_views();

    llama_kpool_set_input(cells_of, views,
            cell_pool, pool_cells, bias, pool_bias, sel_mask, cand_mask,
            pool_reps, new_pool_cells, new_pool_reps, strm_of, kv_size, rebuild, ubatch, kpool);

    if (!check || !views) {
        return;
    }

    // the same maps from the cells; the views must have written the same bytes
    ggml_tensor * outs[] = {
        cell_pool, pool_cells, bias, pool_bias, sel_mask, cand_mask, pool_reps, new_pool_cells, new_pool_reps,
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
            pool_reps, new_pool_cells, new_pool_reps, strm_of, kv_size, rebuild, ubatch, kpool);

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
}

void llm_graph_input_kpool::set_input(const llama_ubatch * ubatch) {
    // unconditional: the key/gate STORE runs on the dense path too, or cells below n_select
    // would have no indexer state when the first ubatch crosses it
    mctx_idx->set_input_k_idxs(k_idxs, ubatch);

    if (pool_cells == nullptr) {
        return;
    }

    std::vector<uint32_t> strm_of;

    if (pool_reps) {
        strm_of.resize(mctx_idx->get_n_stream());

        for (uint32_t s = 0; s < strm_of.size(); ++s) {
            strm_of[s] = mctx_idx->get_strm(s);
        }
    }

    GGML_ASSERT(ggml_backend_buffer_is_host(pool_dump->buffer) && pool_dump->type == GGML_TYPE_F32 && ggml_is_contiguous(pool_dump));
    std::fill_n((float *) pool_dump->data, ggml_nelements(pool_dump), -FLT_MAX);

    llama_kv_cache_set_input_kpool(
            mctx_attn->get_kv(),
            /* cell_pool */ nullptr, pool_cells, /* bias */ nullptr, pool_bias,
            sel_mask, cand_mask,
            pool_reps, new_pool_cells, new_pool_reps,
            strm_of.empty() ? nullptr : strm_of.data(),
            pool_reps ? (int64_t) mctx_idx->get_kv()->get_size() : 0,
            rebuild,
            ubatch, kpool);

    // cleared here, not in build_inp_kpool: a graph built but not evaluated must not clear it
    if (rebuild) {
        mctx_attn->get_kv()->clear_kpool_dirty();
    }
}
