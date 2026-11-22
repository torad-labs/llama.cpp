// The compute buffer each rank of a meta backend reserves for a graph, sized from what the rank
// stores and not from the whole merged tensor. A head-split projection like Inkling 2's Q holds
// every rank's slice in one merged node, so sizing without resolved splits gave every rank a slot
// wide enough for all of them together: about four times one rank's need on four ranks. The
// recorded 128-query case of chapter 13 measured a rank at about 2.28x an unsplit reserve once
// attention scratch joined this class (docs/torad/research/inkling-2-small/
// 13-batched-serving-results.md).
//
// One graph is measured through ggml_backend_sched_reserve_size under both sizings, one per
// process: --legacy sets GGML_META_MERGED_RESERVE_LEGACY=1 before ggml runs and expects the whole
// tensors; by default it expects the largest single share plus the mirrored input. Then a shape
// small enough to execute runs over the per-rank sizing and against one plain device: with only
// dyadic values every partial sum is exact in f32, so any accumulation order lands bit for bit.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

constexpr int     n_devs   = 4;
constexpr int64_t E_big    = 1024; // the hidden width feeding the probe projection
constexpr int64_t K_big    = 6144; // its split dim: twelve heads of 128 here
constexpr int64_t R_big    = 128;  // the tokens of the probe ubatch

constexpr int64_t E_small  = 16;
constexpr int64_t K_small  = 64;
constexpr int64_t R_small  = 8;

struct fixture {
    std::vector<ggml_backend_dev_t> parts;
    ggml_backend_dev_t              meta = nullptr;
    ggml_backend_dev_t              cpu  = nullptr;
};

ggml_backend_meta_split_state split(const ggml_tensor * tensor, void *) {
    const char * name = ggml_get_name(tensor);
    if (strcmp(name, "wq") == 0) {
        ggml_backend_meta_split_state result = {GGML_BACKEND_SPLIT_AXIS_1, {}, {1}, 1};
        const int64_t share = tensor->ne[1]/n_devs;
        for (int j = 0; j < n_devs; ++j) {
            result.ne[j] = share;
        }
        return result;
    }
    return {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {}, {1}, 1};
}

fixture make_fixture() {
    fixture f;
    for (size_t i = 0; i < ggml_backend_dev_count() && (!f.cpu || (int) f.parts.size() < n_devs); ++i) {
        if (ggml_backend_dev_type(ggml_backend_dev_get(i)) == GGML_BACKEND_DEVICE_TYPE_CPU) {
            if (!f.cpu) { f.cpu = ggml_backend_dev_get(i); continue; }
            f.parts.push_back(ggml_backend_dev_get(i));
        }
    }
    GGML_ASSERT(f.cpu);
    while ((int) f.parts.size() < n_devs) {
        f.parts.push_back(f.cpu); // one device splits as many as it can stand
    }
    f.meta = ggml_backend_meta_device(f.parts.data(), f.parts.size(), split, nullptr);
    GGML_ASSERT(f.meta);
    return f;
}

// dyadic multiples of an eighth: partial sums stay exact in f32 whatever order they take
float probe_value(int64_t i) {
    return float(((i*7)%13 - 6)/8.0);
}

// the probe graph: result = wq @ probe_x, with wq head-split and probe_x mirrored
struct probe_graph {
    ggml_context *        wctx   = nullptr;
    ggml_context *        gctx   = nullptr;
    ggml_backend_buffer_t wbuf   = nullptr;
    ggml_tensor *         x      = nullptr; // the input leaf
    ggml_tensor *         result = nullptr; // the split node
    ggml_cgraph *         gf     = nullptr;

    void build(int64_t e, int64_t k, int64_t r, bool with_data, ggml_backend_buffer_type_t buft) {
        const ggml_init_params wp = {2*ggml_tensor_overhead(), nullptr, /*.no_alloc =*/ true};
        wctx = ggml_init(wp);
        GGML_ASSERT(wctx);
        ggml_tensor * wq = ggml_new_tensor_2d(wctx, GGML_TYPE_F32, e, k);
        ggml_set_name(wq, "wq");
        if (buft) {
            // the weight lives in this device's own buffer type whether or not it carries data:
            // its split decisions apply to every graph that reads it
            wbuf = ggml_backend_alloc_ctx_tensors_from_buft(wctx, buft);
            GGML_ASSERT(wbuf);
            if (with_data) {
                std::vector<float> vals(e*k);
                for (int64_t i = 0; i < e*k; ++i) {
                    vals[i] = probe_value(i);
                }
                ggml_backend_tensor_set(wq, vals.data(), 0, sizeof(float)*e*k);
            }
        }

        const size_t gsize = 6*ggml_tensor_overhead() + ggml_graph_overhead_custom(16, false);
        const ggml_init_params gp = {gsize, nullptr, /*.no_alloc =*/ true};
        gctx = ggml_init(gp);
        GGML_ASSERT(gctx);
        x = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, e, r);
        ggml_set_input(x);
        gf = ggml_new_graph_custom(gctx, 16, false);

        // the weight tensor lives in its own context and outlives this graph's build
        result = ggml_mul_mat(gctx, wq, x);
        ggml_set_output(result);
        ggml_build_forward_expand(gf, result);
    }

    ~probe_graph() {
        if (wbuf) {
            ggml_backend_buffer_free(wbuf);
        }
        if (gctx) {
            ggml_free(gctx);
        }
        if (wctx) {
            ggml_free(wctx);
        }
    }
};

// runs the probe on one device and returns its output values; weights must carry data already.
// The scheduler wants two backends, so a plain device rides twice and the meta case takes the CPU
// as its second.
bool compute_and_read(ggml_backend_dev_t dev, probe_graph & pg, int64_t e, int64_t r,
        std::vector<float> & out_vals) {
    ggml_backend_t backends[2];
    backends[0] = ggml_backend_dev_init(dev, nullptr);
    if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
        backends[1] = backends[0];
    } else {
        ggml_backend_dev_t cpu_dev = nullptr;
        for (size_t i = 0; i < ggml_backend_dev_count() && !cpu_dev; ++i) {
            if (ggml_backend_dev_type(ggml_backend_dev_get(i)) == GGML_BACKEND_DEVICE_TYPE_CPU) {
                cpu_dev = ggml_backend_dev_get(i);
            }
        }
        GGML_ASSERT(cpu_dev);
        backends[1] = ggml_backend_dev_init(cpu_dev, nullptr);
    }
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, nullptr, 2, 1024, false, true);
    for (int i = 0; i < ggml_graph_n_nodes(pg.gf); ++i) {
        ggml_backend_sched_set_tensor_backend(sched, ggml_graph_node(pg.gf, i), backends[0]);
    }

    ggml_tensor * xdev = pg.x;
    ggml_tensor * outn = pg.result;

    bool ok = xdev && outn;
    if (ok) {
        ok = (bool) ggml_backend_sched_alloc_graph(sched, pg.gf);
    }
    if (ok) {
        std::vector<float> xin(e*r);
        for (int64_t i = 0; i < e*r; ++i) {
            xin[i] = probe_value(100000 + i*3);
        }
        ggml_backend_tensor_set(xdev, xin.data(), 0, sizeof(float)*e*r);
        ok = ggml_backend_sched_graph_compute(sched, pg.gf) == GGML_STATUS_SUCCESS;
        if (ok) {
            out_vals.resize(ggml_nbytes(outn)/sizeof(float));
            ggml_backend_tensor_get(outn, out_vals.data(), 0, sizeof(float)*out_vals.size());
        }
    }
    ggml_backend_sched_free(sched);
    if (backends[1] != backends[0]) {
        ggml_backend_free(backends[1]);
    }
    ggml_backend_free(backends[0]);
    return ok;
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

bool measure_shrink(const fixture & f) {
    const bool legacy_mode = getenv("GGML_META_MERGED_RESERVE_LEGACY") != nullptr;

    probe_graph big;
    // the weight needs no data to be sized: its geometry alone drives every split decision, so it
    // lives in the meta device's own buffer type either way and the legacy switch decides which
    // sizing runs
    big.build(E_big, K_big, R_big, /*with_data =*/ false, ggml_backend_dev_buffer_type(f.meta));

    if (!big.result || !big.x) {
        fprintf(stderr, "the probe graph did not build\n");
        return false;
    }
    struct ggml_cgraph * gf = big.gf;
    ggml_tensor      * out  = big.result;
    ggml_tensor      * xin  = big.x;

    const std::vector<size_t> sizes = reserve_sizes(f, gf);
    if (sizes.size() < 2 || !sizes[0]) {
        fprintf(stderr, "no meta buffer size reported\n");
        return false;
    }
    const size_t got = sizes[0];

    bool ok = true;
    if (legacy_mode) {
        // both nodes whole, once alignment is added
        const size_t expect = ggml_nbytes(out) + ggml_nbytes(xin);
        if (got < expect || got > expect + 2048) {
            fprintf(stderr, "merged-width reserve %zu outside [%zu, %zu]\n", got, expect, expect + 2048);
            ok = false;
        }
    } else {
        const size_t expect = ggml_nbytes(out)/n_devs + ggml_nbytes(xin);
        if (got < expect || got > expect + 2048) {
            fprintf(stderr, "per-rank reserve %zu outside [%zu, %zu]\n", got, expect, expect + 2048);
            ok = false;
        }
    }
    fprintf(stdout, "%s sizing of a %lldx%lld projection over %lld rows: reserve %zu bytes (largest share would be %zu plus the input)\n",
            legacy_mode ? "whole-tensor" : "shard-sized", (long long) E_big,
            (long long) K_big, (long long) R_big, got, ggml_nbytes(out)/n_devs);
    return ok;
}

bool run_exact(const fixture & f) {
    ggml_backend_dev_t one_cpu = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count() && !one_cpu; ++i) {
        if (ggml_backend_dev_type(ggml_backend_dev_get(i)) == GGML_BACKEND_DEVICE_TYPE_CPU) {
            one_cpu = ggml_backend_dev_get(i);
        }
    }
    GGML_ASSERT(one_cpu);

    probe_graph meta_case, ref_case;
    meta_case.build(E_small, K_small, R_small, /*with_data =*/ true, ggml_backend_dev_buffer_type(f.meta));
    ref_case.build(E_small, K_small, R_small, /*with_data =*/ true, ggml_backend_dev_buffer_type(one_cpu));

    std::vector<float> got, ref;
    bool ok = compute_and_read(f.meta, meta_case, E_small, R_small, got);
    if (ok) {
        ok = compute_and_read(one_cpu, ref_case, E_small, R_small, ref);
    }
    if (!ok || got.size() != ref.size()) {
        fprintf(stderr, "probe runs did not produce matching outputs\n");
        return false;
    }

    int mismatches = 0;
    for (size_t i = 0; i < got.size() && mismatches < 8; ++i) {
        if (memcmp(&got[i], &ref[i], sizeof(float)) != 0) {
            fprintf(stderr, "result %zu differs: %g vs %g\n", i, (double) got[i], (double) ref[i]);
            ++mismatches;
            ok = false;
        }
    }
    fprintf(stdout, "%s: the split probe of %lld rows matches one plain device bit for bit (%zu values)\n",
            ok ? "PASS" : "FAIL", (long long) R_small, got.size());
    return ok;
}

} // namespace

int main(int argc, char ** argv) {
    const bool legacy = argc > 1 && strcmp(argv[1], "--legacy") == 0;
    if (legacy) {
        setenv("GGML_META_MERGED_RESERVE_LEGACY", "1", /*overwrite =*/ 1);
    }

    const fixture f = make_fixture();
    bool ok = measure_shrink(f);
    if (!legacy) {
        ok = run_exact(f) && ok;
    }
    fprintf(stdout, "%s: meta reserve sizing (%s)\n", ok ? "PASS" : "FAIL",
            getenv("GGML_META_MERGED_RESERVE_LEGACY") ? "merged-width" : "per-rank");
    return ok ? 0 : 1;
}
