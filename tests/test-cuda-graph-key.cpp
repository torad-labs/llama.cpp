// The CUDA backend keeps a CUDA graph per ggml graph, found by a key computed from the graph. The key was the first
// node's address, and a context that builds graphs of different shapes in turn in the same memory (a speculative draft
// context's catch-up batch, then its one-row draft step) gives them the same first-node address: they shared one CUDA
// graph, each compute found the other shape's properties there, and every compute ran directly, uncaptured.
//
// Two graphs y = w x, x of 3 rows and of 1 row, are built in turn in one metadata buffer and computed alternately.
// Each shape must be captured once and replayed from then on, every output matching the CPU backend on fresh inputs
// (a replay of the other shape's graph would not). The test counts the backend's own debug lines: "CUDA graph warmup
// complete" when a graph is captured, "CUDA graph warmup reset" when a captured graph is found stale. A control first
// computes one shape three times: a device that does not capture it does not use CUDA graphs and is skipped.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

struct graph_log {
    int captured = 0;
    int stale    = 0;
};

static void log_callback(ggml_log_level level, const char * text, void * user_data) {
    graph_log * log = (graph_log *) user_data;
    if (strstr(text, "CUDA graph warmup complete")) {
        log->captured++;
    }
    if (strstr(text, "CUDA graph warmup reset")) {
        log->stale++;
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

// computes y = w x n_computes times on a fresh backend, the rows of x cycling through rows, every graph built anew in
// one metadata buffer as a context rebuilds its graph in the same memory; checks each output against the CPU backend.
// Returns false if an output is wrong; sets supported to false if the backend cannot run the graph.
static bool compute_in_turn(ggml_backend_dev_t dev, ggml_backend_t cpu, graph_log & log,
        const std::vector<int64_t> & rows, int n_computes, bool & supported) {
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

    const void * first_node = nullptr;
    std::vector<bool> armed(rows.size(), false); // per shape: captured, and nothing found stale since
    bool ok = true;
    for (int i = 0; i < n_computes; ++i) {
        const int64_t m = rows[i % rows.size()];
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
        // every shape's graph has the same first-node address, the case the key must tell apart
        GGML_ASSERT(first_node == nullptr || first_node == ggml_graph_node(gf, 0));
        first_node = ggml_graph_node(gf, 0);
        GGML_ASSERT(ggml_gallocr_alloc_graph(galloc, gf));

        const std::vector<float> xv = random_values(rng, k * m);
        ggml_backend_tensor_set(x, xv.data(), 0, xv.size() * sizeof(float));
        const graph_log before = log;
        GGML_ASSERT(ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS);
        ggml_backend_synchronize(backend);

        std::vector<float> yv(n * m);
        ggml_backend_tensor_get(y, yv.data(), 0, yv.size() * sizeof(float));
        const double e = nmse(yv, cpu_mul_mat(cpu, wv, xv, k, n, m));
        const bool pass = e < max_nmse;
        ok &= pass;
        const bool stale    = log.stale > before.stale;
        const bool captured = log.captured > before.captured;
        const char * what = stale ? "stale graph found, ran directly" : captured ? "captured" :
                            armed[i % rows.size()] ? "replayed" : "ran directly";
        armed[i % rows.size()] = captured || (armed[i % rows.size()] && !stale);
        printf("  compute %d, %lld row%s: %-32s nmse %.2e vs CPU: %s\n", i + 1, (long long) m, m == 1 ? " " : "s", what, e,
               pass ? "ok" : "FAILED");
        ggml_free(ctx);
    }

    ggml_gallocr_free(galloc);
    ggml_backend_buffer_free(wbuf);
    ggml_free(wctx);
    ggml_backend_free(backend);
    return ok;
}

int main(void) {
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
        ok &= compute_in_turn(dev, cpu, log, { 3 }, 3, supported);
        if (!supported) {
            printf("  MUL_MAT with an f32 weight is not supported here, skipped\n");
            continue;
        }
        if (log.captured == 0) {
            printf("  no graph was captured: this device does not use CUDA graphs, skipped\n");
            continue;
        }

        printf(" two shapes in turn:\n");
        ok &= compute_in_turn(dev, cpu, log, { 3, 1 }, 8, supported);
        const bool once_each = log.captured == 2 && log.stale == 0;
        ok &= once_each;
        printf("  captures %d, stale %d: %s\n", log.captured, log.stale,
               once_each ? "each shape captured once and replayed after" : "FAILED, expected one capture per shape and none stale");
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
