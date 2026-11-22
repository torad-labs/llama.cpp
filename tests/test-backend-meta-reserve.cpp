// The compute buffer each rank of a meta backend reserves for a graph, sized from what the rank
// stores and not from the whole merged tensor. A head-split projection like Inkling 2's Q holds
// every rank's slice in one merged node, so sizing without resolved splits gave every rank a slot
// wide enough for all of them together: about four times one rank's need on four ranks. The
// recorded 128-query case of chapter 13 measured a rank at about 2.28x an unsplit reserve once
// attention scratch joined this class (docs/torad/research/inkling-2-small/
// 13-batched-serving-results.md).
//
// Four things are proven here:
// - the probe projection is measured under both sizings over two and four ranks, even and uneven,
//   one mode a process (--legacy sets GGML_META_MERGED_RESERVE_LEGACY=1 before ggml runs);
// - a graph whose chain also reads a weight held outside any meta buffer still measures, that
//   branch left to its own full-width path — and to the other backend's ledger — while the sharded
//   parts shrink; the measure walk never touches a foreign buffer;
// - such a mixed graph also executes over the meta backend and matches a wholly plain twin bit for
//   bit;
// - a matrix of admitted ops on a rows-sharded chain — mul_mat, sqr, add, mul, sub, each one whose
//   split rule admits it — reserves, executes and matches the same way.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

constexpr int64_t E_big    = 1024; // the hidden width feeding the probe projection
constexpr int64_t K_big    = 6144; // its split dim: twelve heads of 128 here
constexpr int64_t R_big    = 128;  // the tokens of the probe ubatch

constexpr int64_t E_small  = 16;
constexpr int64_t K_small  = 64;
constexpr int64_t R_small  = 8;

struct fixture {
    std::vector<ggml_backend_dev_t> parts;
    ggml_backend_dev_t              meta  = nullptr;
    ggml_backend_buffer_type_t      plain = nullptr; // an ordinary buffer type beside the meta one
    ggml_backend_dev_t              cpu   = nullptr;
};

int                  g_n_ranks = 0;
std::vector<double>  g_frac;                    // per-rank fraction of a weight's split axis

// dyadic multiples of an eighth: partial sums stay exact in f32 whatever order they take
float probe_value(int64_t i) {
    return float(((i*7)%13 - 6)/8.0);
}

void fill_probe(float * vals, int64_t n, int64_t seed) {
    for (int64_t i = 0; i < n; ++i) {
        vals[i] = probe_value(seed + i);
    }
}

// the largest share of an extent along the split axis, from this case's fractions
size_t top_share_bytes(int64_t extent, size_t elem_size) {
    double frac = 0;
    for (int j = 0; j < g_n_ranks; ++j) {
        frac = std::max(frac, g_frac[j]);
    }
    return (size_t) ((double) extent*frac)*elem_size;
}

ggml_backend_meta_split_state split(const ggml_tensor * tensor, void *) {
    const char * name = ggml_get_name(tensor);
    if ((strcmp(name, "wq") == 0 || strcmp(name, "wa") == 0) && g_n_ranks > 0) {
        GGML_ASSERT((int) g_frac.size() == g_n_ranks);
        // each weight splits by these fractions of its own extent, whatever that extent is
        ggml_backend_meta_split_state result = {GGML_BACKEND_SPLIT_AXIS_1, {}, {1}, 1};
        const int64_t total = tensor->ne[1];
        for (int j = 0; j < g_n_ranks; ++j) {
            const int64_t share = (int64_t) ((double) total*g_frac[j]);
            GGML_ASSERT(share >= 0 && share <= total);
            result.ne[j] = share;
        }
        return result;
    }
    return {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {}, {1}, 1};
}

fixture make_fixture(int n_ranks, bool uneven) {
    fixture f;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_CPU) { continue; }
        if (!f.cpu) {
            f.cpu = dev;
            continue;
        }
        if ((int) f.parts.size() < n_ranks) {
            f.parts.push_back(dev);
        }
    }
    GGML_ASSERT(f.cpu);
    while ((int) f.parts.size() < n_ranks) {
        f.parts.push_back(f.cpu); // one device splits as many as it can stand
    }
    g_n_ranks = n_ranks;
    g_frac.assign(n_ranks, 1.0/n_ranks);
    if (uneven) {
        // a quarter to three quarters, the split Inkling's expert rows take on two ranks
        GGML_ASSERT(n_ranks == 2);
        g_frac[0] = 0.25;
        g_frac[1] = 0.75;
    }
    f.meta  = ggml_backend_meta_device(f.parts.data(), f.parts.size(), split, nullptr);
    GGML_ASSERT(f.meta);
    f.plain = ggml_backend_dev_buffer_type(f.cpu);
    return f;
}

// weights in their own context so graphs can outlive them; each weight lands in the buffer type
// given for it: the meta one splits by the callback above, an ordinary one stays whole and foreign
struct case_graph {
    struct weight {
        ggml_context *        ctx = nullptr;
        ggml_backend_buffer_t buf = nullptr;
        ggml_tensor *         t   = nullptr;
    };

    ggml_context *      gctx   = nullptr;
    ggml_cgraph *       gf     = nullptr;
    ggml_tensor *       x      = nullptr;
    std::vector<weight> weights;
    ggml_tensor *       result    = nullptr; // the first product
    bool                has_extra = false;

    void add_weight(const char * name, int64_t e, int64_t k, bool with_data,
            ggml_backend_buffer_type_t buft, int64_t seed) {
        const ggml_init_params wp = {2*ggml_tensor_overhead(), nullptr, /*.no_alloc =*/ true};
        weight w;
        w.ctx = ggml_init(wp);
        GGML_ASSERT(w.ctx);
        w.t = ggml_new_tensor_2d(w.ctx, GGML_TYPE_F32, e, k);
        ggml_set_name(w.t, name);
        if (buft) {
            w.buf = ggml_backend_alloc_ctx_tensors_from_buft(w.ctx, buft);
            GGML_ASSERT(w.buf);
        }
        if (w.buf && with_data) {
            std::vector<float> vals(e*k);
            fill_probe(vals.data(), e*k, seed);
            ggml_backend_tensor_set(w.t, vals.data(), 0, sizeof(float)*e*k);
        }
        weights.push_back(std::move(w));
    }

    // builds result = w0 @ x and, beside it, w1 @ x when a second weight exists; the matrix case
    // chains further ops onto the first product through chain_matrix_ops below instead
    void build(int64_t e, int64_t k, int64_t r) {
        GGML_ASSERT(!weights.empty());
        // every chained op adds a tensor object of its own: room for the chain plus slack
        const size_t gsize = (12 + 2*weights.size())*ggml_tensor_overhead() +
                             ggml_graph_overhead_custom(24, false);
        const ggml_init_params gp = {gsize, nullptr, /*.no_alloc =*/ true};
        gctx = ggml_init(gp);
        GGML_ASSERT(gctx);
        x  = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, e, r);
        ggml_set_input(x);
        gf = ggml_new_graph_custom(gctx, 24, false);

        // the weights live in their own contexts and outlive this graph's build
        result = ggml_mul_mat(gctx, weights[0].t, x);
        ggml_set_output(result);
        ggml_build_forward_expand(gf, result);
        if (weights.size() > 1) {
            has_extra     = true;
            ggml_tensor * extra_node = ggml_mul_mat(gctx, weights[1].t, x);
            ggml_set_output(extra_node);
            ggml_build_forward_expand(gf, extra_node);
        }
    }

    // the admitted-op matrix on top of the sharded product: sqr, add, mul and sub, each one whose
    // split rule admits a rows-sharded source, chained so every kernel runs in the graph
    void chain_matrix_ops() {
        GGML_ASSERT(!has_extra); // the matrix chains onto the single sharded product alone
        ggml_tensor * m1 = result;
        ggml_tensor * m2 = ggml_sqr(gctx, m1);
        ggml_tensor * a  = ggml_add(gctx, m2, m1);
        ggml_tensor * s  = ggml_mul(gctx, a, a);
        result = ggml_sub(gctx, s, m1);
        ggml_format_name(result, "matrix_out");
        ggml_set_output(result);
        ggml_build_forward_expand(gf, result);
    }

    void release_buffers() {
        for (auto & w : weights) {
            if (w.buf) { ggml_backend_buffer_free(w.buf); w.buf = nullptr; }
        }
    }

    ~case_graph() {
        release_buffers();
        if (gctx) { ggml_free(gctx); }
        for (auto & w : weights) {
            if (w.ctx) { ggml_free(w.ctx); }
        }
    }
};

bool legacy_env() {
    return getenv("GGML_META_MERGED_RESERVE_LEGACY") != nullptr;
}

// measured sizes indexed by the backends as given: entry i holds what buffer type i reserves. The
// pair is {meta, cpu} like every meta test builds it: the scheduler wants two.
std::vector<size_t> reserve_sizes(const fixture & f, struct ggml_cgraph * graph) {
    ggml_backend_t backends[2] = {
        ggml_backend_dev_init(f.meta, nullptr),
        ggml_backend_dev_init(f.cpu, nullptr),
    };
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, nullptr, 2, 1024, false, true);
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        ggml_backend_sched_set_tensor_backend(sched, ggml_graph_node(graph, i), backends[0]);
    }
    std::vector<size_t> sizes(2, 0);
    ggml_backend_sched_reserve_size(sched, graph, sizes.data());
    const std::vector<size_t> ret = sizes;
    ggml_backend_sched_free(sched);
    ggml_backend_free(backends[1]);
    ggml_backend_free(backends[0]);
    return ret;
}

bool measure_shrink(const fixture & f, int n_ranks, bool uneven, bool legacy_mode) {
    case_graph big;
    // the weight needs no data to be sized: its geometry alone drives every split decision
    big.add_weight("wq", E_big, K_big, /*with_data =*/ false,
            ggml_backend_dev_buffer_type(f.meta), /*seed =*/ 0);
    big.build(E_big, K_big, R_big);

    if (!big.result || !big.x) {
        fprintf(stderr, "the probe graph did not build\n");
        return false;
    }

    const std::vector<size_t> sizes = reserve_sizes(f, big.gf);
    if (sizes.size() < 2 || !sizes[0]) {
        fprintf(stderr, "no meta buffer size reported\n");
        return false;
    }
    const size_t got       = sizes[0];
    const size_t merged    = ggml_nbytes(big.result);
    const size_t shard_top = top_share_bytes(K_big, sizeof(float))*R_big;

    bool ok = true;
    if (legacy_mode) {
        const size_t expect = merged + ggml_nbytes(big.x);
        if (got < expect || got > expect + 2048) {
            fprintf(stderr, "%d-rank%s merged-width reserve %zu outside [%zu, %zu]\n",
                    n_ranks, uneven ? ", uneven shares" : "", got, expect, expect + 2048);
            ok = false;
        }
    } else {
        // the largest share carries the pad its stride needs; an even split meets it exactly and
        // an uneven pair within a row's padding
        const size_t expect = shard_top + ggml_nbytes(big.x);
        if (got < expect || got > expect + 2048) {
            fprintf(stderr, "%d-rank%s per-rank reserve %zu outside [%zu, %zu]\n",
                    n_ranks, uneven ? ", uneven shares" : "", got, expect, expect + 2048);
            ok = false;
        }
    }
    fprintf(stdout, "%s sizing over %d ranks%s of a %lldx%lld projection over %lld rows: meta side reserves %zu bytes\n",
            legacy_mode ? "whole-tensor" : "shard-sized", n_ranks, uneven ? ", uneven shares" : "",
            (long long) E_big, (long long) K_big, (long long) R_big, got);
    return ok;
}

// A graph with one weight in the meta buffer and one in an ordinary buffer beside it. The measure
// walk bails to whole-width for whatever branch reads the foreign weight, and that branch's cost
// lands on the ledger of the backend whose tensors it feeds.
bool measure_foreign(const fixture & f, int n_ranks) {
    case_graph mixed;
    mixed.add_weight("wq", E_big, K_big, /*with_data =*/ false,
            ggml_backend_dev_buffer_type(f.meta), /*seed =*/ 0);
    mixed.add_weight("wq_host", E_big, K_big, /*with_data =*/ false, f.plain, /*seed =*/ 0);
    mixed.build(E_big, K_big, R_big);

    const std::vector<size_t> sizes = reserve_sizes(f, mixed.gf);
    if (sizes.size() < 2) {
        fprintf(stderr, "no buffer sizes reported for the mixed graph\n");
        return false;
    }
    const size_t got_meta   = sizes[0];
    const size_t got_cpu    = sizes[1];
    const size_t merged     = ggml_nbytes(mixed.result); // both products share this extent
    const size_t x_bytes    = ggml_nbytes(mixed.x);
    const size_t shard_top  = top_share_bytes(K_big, sizeof(float))*R_big;

    bool ok = true;
    if (!legacy_env()) {
        const size_t expect = shard_top + x_bytes;
        if (got_meta < expect || got_meta > expect + 2048) {
            fprintf(stderr, "%d-rank mixed-graph meta-side reserve %zu outside [%zu, %zu]\n",
                    n_ranks, got_meta, expect, expect + 2048);
            ok = false;
        }
        // the foreign branch runs whole-width wherever it lands: at least one full product's worth
        if (got_cpu < merged) {
            fprintf(stderr, "%d-rank mixed-graph cpu side holds %zu below a whole product %zu\n",
                    n_ranks, got_cpu, merged);
            ok = false;
        }
    } else {
        const size_t expect = merged + x_bytes;
        if (got_meta < expect || got_meta > expect + 2048) {
            fprintf(stderr, "%d-rank mixed-graph merged-width meta side %zu outside [%zu, %zu]\n",
                    n_ranks, got_meta, expect, expect + 2048);
            ok = false;
        }
        if (got_cpu < merged) {
            fprintf(stderr, "%d-rank mixed-graph merged-width cpu side %zu below a whole product %zu\n",
                    n_ranks, got_cpu, merged);
            ok = false;
        }
    }
    fprintf(stdout, "%s sizing of a graph with a meta weight and a plain-buffered weight over %d ranks: meta %zu, other side %zu bytes\n",
            legacy_env() ? "whole-tensor" : "shard-sized", n_ranks, got_meta, got_cpu);
    return ok;
}

// runs the case on one device and reads every output it declared, in build order; weights must
// carry data already. The scheduler wants two backends, so a plain device rides twice when the case
// is plain and the meta case takes the CPU as its second.
bool compute_and_read(const fixture & f, bool on_meta, case_graph & pg, int64_t e, int64_t r,
        std::vector<float> & out_vals) {
    ggml_backend_t backends[2];
    backends[0] = ggml_backend_dev_init(on_meta ? f.meta : f.cpu, nullptr);
    backends[1] = on_meta ? ggml_backend_dev_init(f.cpu, nullptr) : backends[0];
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, nullptr, 2, 1024, false, true);
    for (int i = 0; i < ggml_graph_n_nodes(pg.gf); ++i) {
        ggml_backend_sched_set_tensor_backend(sched, ggml_graph_node(pg.gf, i), backends[0]);
    }

    bool ok = (bool) ggml_backend_sched_alloc_graph(sched, pg.gf);
    if (ok) {
        std::vector<float> xin(e*r);
        fill_probe(xin.data(), e*r, /*seed =*/ 300000);
        ggml_backend_tensor_set(pg.x, xin.data(), 0, sizeof(float)*e*r);
        ok = ggml_backend_sched_graph_compute(sched, pg.gf) == GGML_STATUS_SUCCESS;
        if (ok) {
            out_vals.clear();
            for (int i = 0; i < ggml_graph_n_nodes(pg.gf); ++i) {
                ggml_tensor * node = ggml_graph_node(pg.gf, i);
                if (!(node->flags & GGML_TENSOR_FLAG_OUTPUT)) { continue; }
                const size_t n_vals = ggml_nbytes(node)/sizeof(float);
                std::vector<float> part(n_vals);
                ggml_backend_tensor_get(node, part.data(), 0, sizeof(float)*n_vals);
                out_vals.insert(out_vals.end(), part.begin(), part.end());
            }
        }
    }
    ggml_backend_sched_free(sched);
    if (backends[1] != backends[0]) {
        ggml_backend_free(backends[1]);
    }
    ggml_backend_free(backends[0]);
    return ok;
}

bool run_case(const fixture & f, int n_ranks, bool uneven, bool with_foreign_weight, bool matrix) {
    case_graph meta_case, ref_case;

    const char * label = matrix ? "the admitted-op matrix"
                                : (with_foreign_weight ? "the probe with a plain-buffered weight beside it"
                                                       : "the split probe");

    meta_case.add_weight(matrix ? "wa" : "wq", E_small, K_small, /*with_data =*/ true,
            ggml_backend_dev_buffer_type(f.meta), /*seed =*/ matrix ? 7000 : 5000);
    if (with_foreign_weight) {
        meta_case.add_weight("wq_host", E_small, K_small, /*with_data =*/ true, f.plain,
                /*seed =*/ 6000);
    }
    meta_case.build(E_small, K_small, R_small);
    if (matrix) {
        meta_case.chain_matrix_ops();
    }

    // the reference computes everything on one plain device: same weights there, no splits
    ref_case.add_weight(matrix ? "wa" : "wq", E_small, K_small, /*with_data =*/ true,
            ggml_backend_dev_buffer_type(f.cpu), /*seed =*/ matrix ? 7000 : 5000);
    if (with_foreign_weight) {
        ref_case.add_weight("wq_host", E_small, K_small, /*with_data =*/ true,
                ggml_backend_dev_buffer_type(f.cpu), /*seed =*/ 6000);
    }
    ref_case.build(E_small, K_small, R_small);
    if (matrix) {
        ref_case.chain_matrix_ops();
    }

    std::vector<float> got, ref;
    bool ok = compute_and_read(f, /*on_meta =*/ true, meta_case, E_small, R_small, got);
    if (ok) { // the foreign branch lands whole-width wherever it goes: same math as the twin's row
        ok = compute_and_read(f, /*on_meta =*/ false, ref_case, E_small, R_small, ref);
    }
    if (!ok || got.size() != ref.size()) {
        fprintf(stderr, "%d-rank%s %s produced %zu values against the reference's %zu\n",
                n_ranks, uneven ? ", uneven shares" : "", label, got.size(), ref.size());
        return false;
    }

    int mismatches = 0;
    for (size_t i = 0; i < got.size() && mismatches < 8; ++i) {
        if (memcmp(&got[i], &ref[i], sizeof(float)) != 0) {
            fprintf(stderr, "result %zu differs on %d ranks%s: %g vs %g\n", i,
                    n_ranks, uneven ? ", uneven shares" : "", (double) got[i], (double) ref[i]);
            ++mismatches;
            ok = false;
        }
    }
    meta_case.release_buffers();
    ref_case.release_buffers();

    fprintf(stdout, "%s: %s over %d ranks%s executed and matched (%zu values)\n",
            ok ? "PASS" : "FAIL", label, n_ranks, uneven ? ", uneven shares" : "", got.size());
    return ok;
}

} // namespace

int main(int argc, char ** argv) {
    const bool legacy = argc > 1 && strcmp(argv[1], "--legacy") == 0;
    if (legacy) {
        setenv("GGML_META_MERGED_RESERVE_LEGACY", "1", /*overwrite =*/ 1);
    }

    bool ok = true;
    for (const int n_ranks : {2, 4}) {
        for (const bool uneven : {false, true}) {
            if (uneven && n_ranks != 2) { continue; } // the uneven pair is Inkling's two-rank shape
            const fixture f = make_fixture(n_ranks, uneven);
            ok = measure_shrink(f, n_ranks, uneven, legacy) && ok;
            ok = measure_foreign(f, n_ranks) && ok;
            if (!legacy) {
                ok = run_case(f, n_ranks, uneven, /*with_foreign_weight =*/ false,
                        /*matrix =*/ false) && ok;
                ok = run_case(f, n_ranks, uneven, /*with_foreign_weight =*/ true,
                        /*matrix =*/ false) && ok;
                ok = run_case(f, n_ranks, uneven, /*with_foreign_weight =*/ false,
                        /*matrix =*/ true) && ok;
            }
        }
    }
    fprintf(stdout, "%s: meta reserve sizing (%s)\n", ok ? "PASS" : "FAIL",
            legacy_env() ? "merged-width" : "per-rank");
    return ok ? 0 : 1;
}
