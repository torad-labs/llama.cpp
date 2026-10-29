// The pools a ubatch's maps hold, sized on its own sequences (llama_kpool_n_pools_ubatch), against the maps sized on the
// whole cache (llama_kpool_n_pools of n_kv).
//
// With --kv-unified a conversation's ubatch was given n_kv/kpool pools, every other conversation's cells counted, and the
// indexer scored all of them, the others at -inf bias. On the Oct 29 lab box (2x RTX PRO 6000, ub 1024) a 16K prompt beside
// 100K cells of another conversation spent 869 ms in the indexer against 73 ms on an empty pool. Sized on the ubatch's own
// runs, a map holds the whole-cache map's leading pools, and the whole-cache map's other pools are the fill of an unused
// pool (bias -inf, cells and rep 0), which the top-k never takes while the dump pools (-FLT_MAX) are there. So:
//   - the ubatch's own maps hold the same bytes as the whole-cache maps' leading part, the rest of which is that fill, from
//     the views and from the cells, for decode, verify and prompt ubatches in pools made as test-kpool-input makes them;
//   - select_k, and so the dump pools, the masks and the sparse attention's bound, do not change;
//   - beside another conversation, the count follows the ubatch's own run, not the pool.

#define main test_kpool_input_main
#include "test-kpool-input.cpp"
#undef main

static int n_compared = 0;
static int n_smaller  = 0; // of which the ubatch's maps held fewer pools than the cache's

// the leading part of `whole` is `own`, and the rest of each row of whole is the fill of an unused pool
static bool same_prefix(const maps & own, const maps & whole, int64_t n_ns, int64_t n_tps, uint32_t r, char * what, size_t n_what) {
    const int64_t n_own   = own.pool_bias->ne[0];
    const int64_t n_whole = whole.pool_bias->ne[0];
    const int64_t n_dump  = own.pool_cells->ne[0]/r - n_own;

    if (n_dump != whole.pool_cells->ne[0]/r - n_whole) {
        snprintf(what, n_what, "dump pools: %lld sized on the ubatch, %lld on the cache", (long long) n_dump,
                (long long) (whole.pool_cells->ne[0]/r - n_whole));
        return false;
    }

    for (int64_t s = 0; s < n_ns; ++s) {
        const int32_t * pc_own   = (const int32_t *) own.pool_cells->data   + s*r*(n_own   + n_dump);
        const int32_t * pc_whole = (const int32_t *) whole.pool_cells->data + s*r*(n_whole + n_dump);

        for (int64_t c = 0; c < r*n_whole; ++c) {
            const int32_t want = c < r*n_own ? pc_own[c] : 0;
            if (pc_whole[c] != want) {
                snprintf(what, n_what, "pool_cells stream %lld slot %lld: %d on the cache, %d on the ubatch", (long long) s,
                        (long long) c, pc_whole[c], want);
                return false;
            }
        }
        for (int64_t c = 0; c < r*n_dump; ++c) {
            if (pc_own[r*n_own + c] != pc_whole[r*n_whole + c]) {
                snprintf(what, n_what, "pool_cells stream %lld dump slot %lld differs", (long long) s, (long long) c);
                return false;
            }
        }

        for (int64_t ii = 0; ii < n_tps; ++ii) {
            const float * b_own   = (const float *) own.pool_bias->data   + (s*n_tps + ii)*n_own;
            const float * b_whole = (const float *) whole.pool_bias->data + (s*n_tps + ii)*n_whole;

            for (int64_t p = 0; p < n_whole; ++p) {
                const bool ok = p < n_own ? memcmp(&b_own[p], &b_whole[p], sizeof(float)) == 0 : std::isinf(b_whole[p]) && b_whole[p] < 0;
                if (!ok) {
                    snprintf(what, n_what, "pool_bias stream %lld row %lld pool %lld: %g on the cache, %g on the ubatch", (long long) s,
                            (long long) ii, (long long) p, b_whole[p], p < n_own ? b_own[p] : -INFINITY);
                    return false;
                }
            }
        }

        if (own.pool_reps) {
            const int32_t * rp_own   = (const int32_t *) own.pool_reps->data   + s*n_own;
            const int32_t * rp_whole = (const int32_t *) whole.pool_reps->data + s*n_whole;

            for (int64_t p = 0; p < n_whole; ++p) {
                if (rp_whole[p] != (p < n_own ? rp_own[p] : 0)) {
                    snprintf(what, n_what, "pool_reps stream %lld pool %lld: %d on the cache, %d on the ubatch", (long long) s,
                            (long long) p, rp_whole[p], p < n_own ? rp_own[p] : 0);
                    return false;
                }
            }
        }
    }

    for (ggml_tensor * const * t : { &own.sel_mask, &own.cand_mask }) {
        ggml_tensor * a = *t;
        ggml_tensor * b = t == &own.sel_mask ? whole.sel_mask : whole.cand_mask;
        if (ggml_nbytes(a) != ggml_nbytes(b) || memcmp(a->data, b->data, ggml_nbytes(a)) != 0) {
            snprintf(what, n_what, "%s differs", t == &own.sel_mask ? "sel_mask" : "cand_mask");
            return false;
        }
    }

    // the new pools' slots: as many as the ubatch can close, or every pool on a rebuild, emitted first and the spare cells
    // after in one order, so a smaller map's slots are the larger's leading ones
    if (own.new_pool_cells) {
        const int64_t m_own   = own.new_pool_cells->ne[0]/r;
        const int64_t m_whole = whole.new_pool_cells->ne[0]/r;
        for (int64_t s = 0; s < n_ns; ++s) {
            for (int64_t p = 0; p < std::min(m_own, m_whole); ++p) {
                const int64_t ro = ((const int64_t *) own.new_pool_reps->data)[s*m_own + p];
                const int64_t rw = ((const int64_t *) whole.new_pool_reps->data)[s*m_whole + p];
                const bool cells_same = memcmp((const int32_t *) own.new_pool_cells->data + (s*m_own + p)*r,
                        (const int32_t *) whole.new_pool_cells->data + (s*m_whole + p)*r, r*sizeof(int32_t)) == 0;
                if (ro != rw || !cells_same) {
                    snprintf(what, n_what, "new pool slot %lld of stream %lld differs (rep %lld on the ubatch, %lld on the cache)",
                            (long long) p, (long long) s, (long long) ro, (long long) rw);
                    return false;
                }
            }
        }
    }

    return true;
}

// a ubatch of n tokens each of one sequence (or two, unified), or of each stream's own; false when the pool has no room.
// Unified, it takes its sequences among the first n_pick
static bool make_ubatch(pool & w, bool unified, uint32_t n, uint32_t n_pick, std::mt19937 & rng, ubatch_data & ud) {
    if (unified) {
        const uint32_t n_two = n_pick > 1 && rng() % 4 == 0 ? std::min<uint32_t>(n_pick, 2 + rng() % 2) : 1;
        for (uint32_t k = 0; k < n_two; ++k) {
            const llama_seq_id s = n_two > 1 ? (llama_seq_id) k : (llama_seq_id) (rng() % n_pick);
            const auto placed = w.append(s, n);
            if (placed.size() != n) {
                return false;
            }
            for (const llama_pos p : placed) {
                ud.add(s, p);
            }
        }
        return true;
    }
    std::vector<std::vector<llama_pos>> placed;
    for (uint32_t s = 0; s < w.n_seq; ++s) {
        placed.push_back(w.append(s, n));
        if (placed.back().size() != n) {
            return false;
        }
    }
    for (uint32_t s = 0; s < w.n_seq; ++s) {
        for (const llama_pos p : placed[s]) {
            ud.add(s, p);
        }
    }
    return true;
}

static constexpr uint32_t TOP_K = 2048; // GLM-5.3 Flash's indexer top_k

static void check_round(std::mt19937 & rng, const config & cfg, uint32_t size, uint32_t n_seq, const char * where0) {
    pool w(size, n_seq, cfg.unified, rng);
    llama_kpool_views views;

    bool rebuild = false;
    const int n_ops = rng() % 300;
    for (int k = 0; k < n_ops; ++k) {
        rebuild |= w.random_op();
    }

    // half the unified rounds: another conversation holds half the pool, and the ubatch is not of it
    const bool neighbour = cfg.unified && n_seq > 1 && rng() % 2 == 0;
    if (neighbour) {
        w.append(n_seq - 1, size/2);
    }

    const uint32_t n_tok = std::vector<uint32_t>{ 1, 3, 64, 1024 }[rng() % 4];

    ubatch_data ud;
    if (!make_ubatch(w, cfg.unified, n_tok, neighbour ? n_seq - 1 : n_seq, rng, ud)) {
        return;
    }
    const llama_ubatch * ub = ud.get();

    const uint32_t n_ns = cfg.unified ? 1 : n_seq;
    const int64_t  n_ps = ub->n_seqs_unq/n_ns;
    const int64_t  n_tps = ub->n_tokens/n_ns;

    uint32_t used_max = 0;
    for (const auto & cells : w.v_cells) {
        used_max = std::max(used_max, cells.used_max_p1());
    }
    const int64_t n_kv = std::min<int64_t>(size, std::max<int64_t>(256, (used_max + 255)/256*256));

    const llama_kpool_cells_fn cells_of = [&](llama_seq_id s) -> const llama_kv_cells & { return w.cells_of(s); };

    const int64_t n_whole = llama_kpool_n_pools(n_kv, cfg.r, n_ps);
    const int64_t n_own   = llama_kpool_n_pools_ubatch(cells_of, *ub, n_kv, cfg.r, n_ns, TOP_K);

    char where[200];
    snprintf(where, sizeof(where), "%s (r %u, %s, %u seqs%s, %lld tokens a stream, %lld pools of %lld)", where0, cfg.r,
            cfg.unified ? "unified" : "streams", n_seq, neighbour ? " beside a neighbour" : "", (long long) n_tps, (long long) n_own,
            (long long) n_whole);

    CHECK(n_own <= n_whole, "%s: more pools than the cache's", where);
    CHECK(llama_kpool_select_k(n_own, TOP_K, cfg.r) == llama_kpool_select_k(n_whole, TOP_K, cfg.r), "%s: select_k changed", where);

    const int64_t n_dump = llama_kpool_select_k(n_whole, TOP_K, cfg.r);

    std::vector<uint32_t> strm_of(n_ns);
    for (uint32_t s = 0; s < n_ns; ++s) {
        strm_of[s] = s;
    }

    // the build takes its pool count from the tensors, as the graph's inputs give it
    const auto make = [&](int64_t n_pools, int fill) {
        return std::make_unique<maps>(n_kv, n_ns, n_tps, n_ps, cfg.r, cfg.kcache, cfg.f16, rebuild, n_dump, fill, n_pools);
    };

    auto whole     = make(n_whole, 0xA5);
    auto own_view  = make(n_own,   0x5A);
    auto own_cells = make(n_own,   0x33);

    const auto build = [&](maps & m, llama_kpool_views * v) {
        return llama_kpool_set_input(cells_of, v, nullptr, m.pool_cells, nullptr, m.pool_bias, m.sel_mask, m.cand_mask,
                m.pool_reps, m.new_pool_cells, m.new_pool_reps, strm_of.data(), w.v_cells[0].size(), rebuild, ub, cfg.r);
    };

    // the view first: the cells' builds never take the log
    const int64_t t_view  = build(*own_view, &views);
    const int64_t t_whole = build(*whole, nullptr);
    const int64_t t_cells = build(*own_cells, nullptr);

    CHECK(t_view == t_whole && t_cells == t_whole, "%s: tails %lld and %lld on the ubatch, %lld on the cache", where,
            (long long) t_view, (long long) t_cells, (long long) t_whole);

    char what[256];
    if (!same_prefix(*own_view, *whole, n_ns, n_tps, cfg.r, what, sizeof(what))) {
        CHECK(false, "%s, from the view: %s", where, what);
    }
    if (!same_prefix(*own_cells, *whole, n_ns, n_tps, cfg.r, what, sizeof(what))) {
        CHECK(false, "%s, from the cells: %s", where, what);
    }

    ++n_compared;
    n_smaller += n_own < n_whole;
}

// beside another conversation: a 1,024-token prompt of a 10K conversation in a unified pool where another holds 100K cells
static void check_neighbour() {
    std::mt19937 rng(1);
    pool w(131072, 2, true, rng);
    w.append(1, 100000);
    w.append(0, 10240);

    ubatch_data ud;
    for (const llama_pos p : w.append(0, 1024)) {
        ud.add(0, p);
    }
    const llama_ubatch * ub = ud.get();

    const llama_kpool_cells_fn cells_of = [&](llama_seq_id s) -> const llama_kv_cells & { return w.cells_of(s); };

    const uint32_t n_kv    = (w.v_cells[0].used_max_p1() + 255)/256*256;
    const uint32_t n_own   = llama_kpool_n_pools_ubatch(cells_of, *ub, n_kv, 4, 1, TOP_K);
    const uint32_t run     = (10240 + 1024 - 1)/4 + 1;

    CHECK(n_own >= run && n_own <= run + 64 + 2, "a 10K conversation beside 100K cells is given %u pools: its run is %u, the cache's %u",
            n_own, run, llama_kpool_n_pools(n_kv, 4, 1));
}

// the worst-case graph is reserved on a ubatch that names no sequence (ubatch_reserve): its maps keep the whole cache's
// pools, so the buffers sized at load hold every later graph
static void check_reserve() {
    std::mt19937 rng(2);
    pool w(131072, 3, true, rng);

    ubatch_data ud;
    for (uint32_t i = 0; i < 1024; ++i) {
        ud.add(0, 0);
    }
    ud.get();
    ud.seq_ptr.assign(ud.pos.size(), nullptr);
    ud.unq = { 0 };
    ud.ub.seq_id     = ud.seq_ptr.data();
    ud.ub.seq_id_unq = ud.unq.data();

    const llama_kpool_cells_fn cells_of = [&](llama_seq_id s) -> const llama_kv_cells & { return w.cells_of(s); };

    const uint32_t n_kv  = 131072;
    const uint32_t n_got = llama_kpool_n_pools_ubatch(cells_of, ud.ub, n_kv, 4, 1, TOP_K);

    CHECK(n_got == llama_kpool_n_pools(n_kv, 4, 1), "the reserve ubatch is given %u pools, the cache's %u", n_got,
            llama_kpool_n_pools(n_kv, 4, 1));
}

int main() {
    std::mt19937 rng(20261029);

    check_neighbour();
    check_reserve();

    for (int round = 0; round < 400; ++round) {
        const config cfg = {
            /*.r        =*/ rng() % 4 == 0 ? 1u : 4u,
            /*.unified  =*/ rng() % 3 != 0,
            /*.kcache   =*/ rng() % 2 == 0,
            /*.f16      =*/ rng() % 4 != 0,
            /*.small_kv =*/ false,
        };
        const uint32_t size  = std::vector<uint32_t>{ 1024, 4096, 16384 }[rng() % 3];
        const uint32_t n_seq = 1 + rng() % 3;

        char where[32];
        snprintf(where, sizeof(where), "round %d", round);
        check_round(rng, cfg, size, n_seq, where);
    }

    printf("test-kpool-pools: %d builds sized on the ubatch compared with the cache's, %d of them smaller, %d failures\n", n_compared,
            n_smaller, n_fail);

    // a run that never sized a map below the cache's compared nothing
    return n_fail == 0 && n_smaller > n_compared/4 ? 0 : 1;
}
