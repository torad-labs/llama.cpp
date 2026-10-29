// The host's row-split input builds (src/llama-host-threads.h) against one thread.
//
// A prompt ubatch writes one row per token, n_kv cells wide, into the pooled indexer's maps (llama_kpool_set_input) and the
// KQ mask (llama_kv_cache_set_input_kq_mask). Split across threads, every byte must be the byte one thread writes, for
// prompt ubatches of up to 1,024 tokens in pools made the way test-kpool-input.cpp makes them: one unified pool or a stream
// per sequence, drafts rolled back, conversations ended, prefixes shared, cells removed, positions shifted. The maps are
// built from a sequence's view and from the cells, f16 and f32, with and without the key cache; the mask additive f16 and
// f32 and bit-packed, causal or not, with a window or not. And the build must really split: a test that only ever ran one
// thread would pass on code that cannot split at all.

#define main test_kpool_input_main
#include "test-kpool-input.cpp"
#undef main

#include "../src/llama-host-threads.h"

static int n_split_maps = 0;
static int n_split_masks = 0;

// a prompt ubatch, placed in the cells first as the cache does: n_tps tokens of one sequence (or of two, sharing a unified
// pool), or n_tps of each stream's own sequence. False when the pool has no room for it
static bool prompt_ubatch(pool & w, bool unified, uint32_t n_tps, std::mt19937 & rng, ubatch_data & ud) {
    if (unified) {
        const bool two = w.n_seq > 1 && rng() % 4 == 0;
        for (uint32_t k = 0; k < (two ? 2u : 1u); ++k) {
            const llama_seq_id s = two ? (llama_seq_id) k : (llama_seq_id) (rng() % w.n_seq);
            const uint32_t     n = two ? n_tps/2 : n_tps;
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
        placed.push_back(w.append(s, n_tps));
        if (placed.back().size() != n_tps) {
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

static int64_t n_kv_of(const pool & w, int64_t pad) {
    uint32_t used_max = 0;
    for (const auto & cells : w.v_cells) {
        used_max = std::max(used_max, cells.used_max_p1());
    }
    return std::min<int64_t>(w.v_cells[0].size(), std::max<int64_t>(pad, (used_max + pad - 1)/pad*pad));
}

// the maps from the view at n_threads, from the cells at one thread, and from the cells at n_threads: the same bytes, and
// both split builds split
static void check_maps(pool & w, llama_kpool_views & views, const config & cfg, const llama_ubatch * ub, bool rebuild, int n_threads,
        std::mt19937 & rng, const char * where) {
    const uint32_t n_ns = cfg.unified ? 1 : w.n_seq;
    const int64_t  n_ps = ub->n_seqs_unq/n_ns;
    const int64_t  n_kv = n_kv_of(w, 32);
    const int64_t  n_tps = ub->n_tokens/n_ns;

    std::vector<uint32_t> strm_of(n_ns);
    for (uint32_t s = 0; s < n_ns; ++s) {
        strm_of[s] = s;
    }

    const llama_kpool_cells_fn cells_of = [&](llama_seq_id s) -> const llama_kv_cells & { return w.cells_of(s); };

    const int64_t n_pools = llama_kpool_n_pools(n_kv, cfg.r, n_ps);
    const int64_t n_dump  = rng() % 3 == 0 ? 0 : rng() % 2 ? 1 + rng() % 4 : std::min<int64_t>(n_pools, 512);

    maps from_view (n_kv, n_ns, n_tps, n_ps, cfg.r, cfg.kcache, cfg.f16, rebuild, n_dump, 0x5A);
    maps one       (n_kv, n_ns, n_tps, n_ps, cfg.r, cfg.kcache, cfg.f16, rebuild, n_dump, 0xA5);
    maps from_cells(n_kv, n_ns, n_tps, n_ps, cfg.r, cfg.kcache, cfg.f16, rebuild, n_dump, 0x33);

    const auto build = [&](maps & m, llama_kpool_views * v, int n) {
        llama_host_threads_set(n);
        llama_kpool_set_input(cells_of, v, nullptr, m.pool_cells, nullptr, m.pool_bias, m.sel_mask, m.cand_mask,
                m.pool_reps, m.new_pool_cells, m.new_pool_reps, strm_of.data(), w.v_cells[0].size(), rebuild, ub, cfg.r);
        return llama_host_threads_last();
    };

    // the view first: the cells' builds never take the log
    const int used_view  = build(from_view,  &views,  n_threads);
    build(one, nullptr, 1);
    const int used_cells = build(from_cells, nullptr, n_threads);
    llama_host_threads_set(0);

    char what[256];
    if (!same(from_view, one, what, sizeof(what))) {
        CHECK(false, "%s: maps from the view at %d threads differ from one thread: %s", where, n_threads, what);
    }
    if (!same(from_cells, one, what, sizeof(what))) {
        CHECK(false, "%s: maps from the cells at %d threads differ from one thread: %s", where, n_threads, what);
    }
    CHECK(used_view == n_threads && used_cells == n_threads, "%s: maps of %lld rows split across %d and %d threads, not %d",
            where, (long long) n_tps, used_view, used_cells, n_threads);

    ++n_split_maps;
}

// the KQ mask at one thread and at n_threads, every type, window and causality
static void check_masks(pool & w, const llama_ubatch * ub, uint32_t n_ns, int n_threads, std::mt19937 & rng, const char * where) {
    const int64_t n_tps = ub->n_tokens/n_ns;

    ggml_init_params ip = { 4*ggml_tensor_overhead(), nullptr, true };

    for (const ggml_type type : { GGML_TYPE_F16, GGML_TYPE_F32, GGML_TYPE_I16 }) {
        const int64_t n_kv = n_kv_of(w, 256);

        for (const llama_swa_type swa_type : { LLAMA_SWA_TYPE_NONE, LLAMA_SWA_TYPE_STANDARD }) {
            for (const bool causal : { true, false }) {
                const uint32_t n_swa = swa_type == LLAMA_SWA_TYPE_NONE ? 0 : 1 + rng() % 300;

                llama_hparams hparams{};

                ggml_context * ctx = ggml_init(ip);
                ggml_tensor * one   = ggml_new_tensor_4d(ctx, type, type == GGML_TYPE_I16 ? n_kv/16 : n_kv, n_tps, 1, n_ns);
                ggml_tensor * split = ggml_new_tensor_4d(ctx, type, type == GGML_TYPE_I16 ? n_kv/16 : n_kv, n_tps, 1, n_ns);
                ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_cpu_buffer_type());
                // a cell neither build writes shows as a difference only if they leave different bytes there
                memset(one->data,   0xa5, ggml_nbytes(one));
                memset(split->data, 0xa5, ggml_nbytes(split));

                llama_host_threads_set(1);
                llama_kv_cache_set_input_kq_mask(one, ub, causal, hparams, w.v_cells, w.seq_to_stream, n_swa, swa_type, false, true);
                llama_host_threads_set(n_threads);
                llama_kv_cache_set_input_kq_mask(split, ub, causal, hparams, w.v_cells, w.seq_to_stream, n_swa, swa_type, false, true);
                const int used = llama_host_threads_last();
                llama_host_threads_set(0);

                CHECK(memcmp(one->data, split->data, ggml_nbytes(one)) == 0, "%s: KQ mask at %d threads differs from one thread (%s, n_kv %lld, %s, %s)",
                        where, n_threads, ggml_type_name(type), (long long) n_kv, causal ? "causal" : "non-causal",
                        swa_type == LLAMA_SWA_TYPE_NONE ? "no window" : "window");
                CHECK(used == n_threads, "%s: a KQ mask of %lld rows split across %d threads, not %d (%s)", where, (long long) n_tps, used,
                        n_threads, ggml_type_name(type));

                ++n_split_masks;

                ggml_backend_buffer_free(buf);
                ggml_free(ctx);
            }
        }
    }
}

int main() {
    std::mt19937 rng(20261029);

    const int threads_of[] = { 2, 3, 8 };

    for (int round = 0; round < 48; ++round) {
        const config cfg = {
            /*.r        =*/ rng() % 4 == 0 ? 1u : 4u,
            /*.unified  =*/ rng() % 3 != 0,
            /*.kcache   =*/ rng() % 2 == 0,
            /*.f16      =*/ rng() % 4 != 0,
            /*.small_kv =*/ false,
        };
        const uint32_t size  = std::vector<uint32_t>{ 4096, 16384, 32768 }[rng() % 3];
        const uint32_t n_seq = 1 + rng() % 3;

        pool w(size, n_seq, cfg.unified, rng);
        llama_kpool_views views;

        bool rebuild = false;

        const int n_ops = rng() % 200;
        for (int k = 0; k < n_ops; ++k) {
            rebuild |= w.random_op();
        }

        // a prompt ubatch: a short one, an odd one, the served size
        const uint32_t n_tps = std::vector<uint32_t>{ 64, 257, 1024 }[rng() % 3];

        ubatch_data ud;
        if (!prompt_ubatch(w, cfg.unified, n_tps, rng, ud)) {
            continue; // the pool is full
        }
        const llama_ubatch * ub = ud.get();

        const uint32_t n_ns = cfg.unified ? 1 : n_seq;
        const int n_threads = threads_of[round % 3];

        char where[160];
        snprintf(where, sizeof(where), "round %d (r %u, %s, %u seqs, %u tokens a stream, %s, %s)", round, cfg.r,
                cfg.unified ? "unified" : "streams", n_seq, ub->n_tokens/n_ns, cfg.f16 ? "f16" : "f32",
                cfg.kcache ? "key cache" : "no key cache");

        check_maps(w, views, cfg, ub, rebuild, n_threads, rng, where);
        check_masks(w, ub, n_ns, n_threads, rng, where);
    }

    printf("test-host-input-threads: %d map builds and %d KQ masks split and compared, %d failures\n", n_split_maps, n_split_masks, n_fail);

    return n_fail == 0 && n_split_maps > 0 && n_split_masks > 0 ? 0 : 1;
}
