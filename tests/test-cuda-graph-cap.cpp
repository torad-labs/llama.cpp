// The CUDA backend keeps a CUDA graph per graph shape (test-cuda-graph-key), and each captured graph's executable holds
// device memory beside the buffers a scheduler sizes. The shapes a context computes follow its traffic, one per graph
// shape and per graph slot for the same shape, so the executables were bounded only by the 10 s eviction: under four
// concurrent requests a 27B hybrid's contexts held 84 MiB of them on a 5080. A context keeps at most
// GGML_CUDA_GRAPH_MAX of them, the least recently used evicted for a new one.
//
// Here the cap is 4, and y = w x is computed for ten row counts of x, three times each: every shape is captured on its
// second compute and replayed on its third, every output matching the CPU backend. The backend's log reports the most
// executables it held at once, which must be the cap: never over it, and reached, since ten shapes were captured. The
// first shape, evicted by then, is computed again and must be captured again and give the right values. A control
// first computes one shape three times: a device that does not capture it does not use CUDA graphs and is skipped.
// Without the eviction all ten are held and it fails.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static constexpr int cap = 4;

struct graph_log {
    int    captured = 0;
    size_t held_max = 0; // "N CUDA graphs held at most", the most the backend reported
};

static void log_callback(ggml_log_level level, const char * text, void * user_data) {
    graph_log * log = (graph_log *) user_data;
    if (strstr(text, "CUDA graph warmup complete")) {
        log->captured++;
    }
    if (const char * held = strstr(text, "CUDA graphs held at most")) {
        // the count is the number right before it: "...: CUDA0: 4 CUDA graphs held at most (cap 4), ..."
        const char * p = held;
        while (p > text && p[-1] == ' ') {
            --p;
        }
        while (p > text && p[-1] >= '0' && p[-1] <= '9') {
            --p;
        }
        log->held_max = std::max(log->held_max, (size_t) strtoull(p, nullptr, 10));
    }
    if (level != GGML_LOG_LEVEL_DEBUG) {
        fputs(text, stderr);
    }
}

static std::vector<float> random_values(std::mt19937 & rng, size_t n) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> v(n);
    for (float & f : v) {
        f = dist(rng);
    }
    return v;
}

// y = w x on the CPU backend
static std::vector<float> cpu_mul_mat(ggml_backend_t cpu, const std::vector<float> & w, const std::vector<float> & x,
        int64_t k, int64_t n, int64_t m) {
    ggml_init_params params = { ggml_tensor_overhead() * 3 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(params);
    ggml_tensor * wt = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, n);
    ggml_tensor * xt = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, m);
    ggml_tensor * yt = ggml_mul_mat(ctx, wt, xt);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, cpu);
    ggml_backend_tensor_set(wt, w.data(), 0, w.size() * sizeof(float));
    ggml_backend_tensor_set(xt, x.data(), 0, x.size() * sizeof(float));
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, yt);
    GGML_ASSERT(ggml_backend_graph_compute(cpu, gf) == GGML_STATUS_SUCCESS);
    std::vector<float> y(n * m);
    ggml_backend_tensor_get(yt, y.data(), 0, y.size() * sizeof(float));
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return y;
}

static double nmse(const std::vector<float> & y, const std::vector<float> & ref) {
    double err = 0.0, sum = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        err += (y[i] - ref[i]) * (double) (y[i] - ref[i]);
        sum += ref[i] * (double) ref[i];
    }
    return err / sum;
}

static constexpr double max_nmse = 5e-4; // test-backend-ops' bound for MUL_MAT

// computes y = w x on one backend for each row count in turn, n_each times in a row, every graph built anew in one
// metadata buffer as a context rebuilds its graph in the same memory; checks each output against the CPU backend.
// Returns false if an output is wrong; sets supported to false if the backend cannot run the graph.
static bool compute_shapes(ggml_backend_dev_t dev, ggml_backend_t cpu, graph_log & log, const std::vector<int64_t> & rows,
        int n_each, bool & supported) {
    const int64_t k = 256;
    const int64_t n = 64;
    std::mt19937 rng(42);
    log = graph_log();
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);

    // the weight lives in its own context and buffer, as a model's weights do
    ggml_init_params wparams = { ggml_tensor_overhead(), nullptr, true };
    ggml_context * wctx = ggml_init(wparams);
    ggml_tensor * w = ggml_new_tensor_2d(wctx, GGML_TYPE_F32, k, n);
    ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors(wctx, backend);
    const std::vector<float> wv = random_values(rng, k * n);
    ggml_backend_tensor_set(w, wv.data(), 0, wv.size() * sizeof(float));

    std::vector<uint8_t> meta(ggml_tensor_overhead() * 2 + ggml_graph_overhead());
    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));

    bool ok = true;
    for (const int64_t m : rows) {
        const int captured_before = log.captured;
        for (int i = 0; i < n_each && ok; ++i) {
            ggml_init_params params = { meta.size(), meta.data(), true };
            ggml_context * ctx = ggml_init(params);
            ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, m);
            ggml_tensor * y = ggml_mul_mat(ctx, w, x);
            ggml_cgraph * gf = ggml_new_graph(ctx);
            ggml_build_forward_expand(gf, y);
            supported = ggml_backend_supports_op(backend, y);
            if (!supported) {
                ggml_free(ctx);
                break;
            }
            GGML_ASSERT(ggml_gallocr_alloc_graph(galloc, gf));

            const std::vector<float> xv = random_values(rng, k * m);
            ggml_backend_tensor_set(x, xv.data(), 0, xv.size() * sizeof(float));
            GGML_ASSERT(ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS);
            ggml_backend_synchronize(backend);

            std::vector<float> yv(n * m);
            ggml_backend_tensor_get(y, yv.data(), 0, yv.size() * sizeof(float));
            const double e = nmse(yv, cpu_mul_mat(cpu, wv, xv, k, n, m));
            if (!(e < max_nmse)) {
                printf("  %lld rows, compute %d: nmse %.2e vs CPU: FAILED\n", (long long) m, i + 1, e);
                ok = false;
            }
            ggml_free(ctx);
        }
        if (!supported) {
            break;
        }
        printf("  %2lld rows x %d: %d captured, outputs %s\n", (long long) m, n_each, log.captured - captured_before,
               ok ? "match the CPU" : "WRONG");
    }

    ggml_gallocr_free(galloc);
    ggml_backend_buffer_free(wbuf);
    ggml_free(wctx);
    ggml_backend_free(backend);
    return ok;
}

int main(void) {
    // the cap is read once, when the first CUDA backend is made
    const std::string cap_env = std::to_string(cap);
#ifdef _WIN32
    _putenv_s("GGML_CUDA_GRAPH_MAX", cap_env.c_str());
#else
    setenv("GGML_CUDA_GRAPH_MAX", cap_env.c_str(), 1);
#endif

    ggml_backend_load_all();

    graph_log log;
    ggml_log_set(log_callback, &log);

    ggml_backend_t cpu = ggml_backend_cpu_init();
    bool ok = true;
    int n_tested = 0;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) {
            continue;
        }
        printf("%s (%s):\n", ggml_backend_dev_name(dev), ggml_backend_dev_description(dev));
        bool supported = true;

        // control: one shape computed three times is captured, or this device does not use CUDA graphs
        printf(" one shape:\n");
        ok &= compute_shapes(dev, cpu, log, { 3 }, 3, supported);
        if (!supported) {
            printf("  MUL_MAT with an f32 weight is not supported here, skipped\n");
            continue;
        }
        if (log.captured == 0) {
            printf("  no graph was captured: this device does not use CUDA graphs, skipped\n");
            continue;
        }

        // ten shapes, then the first again: evicted by the fifth, it is captured anew
        printf(" ten shapes, then the first again, under a cap of %d:\n", cap);
        ok &= compute_shapes(dev, cpu, log, { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1 }, 3, supported);
        const bool all_captured = log.captured == 11;
        const bool at_cap       = log.held_max == (size_t) cap;
        ok &= all_captured && at_cap;
        printf("  captures %d (11 expected), held at most %zu (the cap, %d, expected): %s\n", log.captured, log.held_max,
               cap, all_captured && at_cap ? "ok" : "FAILED");
        n_tested++;
    }
    ggml_backend_free(cpu);
    ggml_log_set(nullptr, nullptr);

    if (n_tested == 0) {
        printf("no GPU device that uses CUDA graphs, skipped\n");
        return 0;
    }
    printf("%s\n", ok ? "OK" : "FAILED");
    return ok ? 0 : 1;
}
