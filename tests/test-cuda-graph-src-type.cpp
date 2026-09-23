// The CUDA backend replays a captured CUDA graph when ggml_cuda_graph_update_required finds the ggml graph unchanged:
// each node compared whole, each src by its data pointer, ne and nb. The src's type was not compared, and f16 and bf16
// have the same ne and nb. A weight retyped in place from f16 to bf16 after the graph was captured, its bf16 bytes
// uploaded to the same address, was then multiplied by the captured f16 kernel: the output was the f16 reading of the
// bf16 bytes, with no error. After the retype every compute must match the CPU backend on the new type.
//
// The backend runs an unchanged graph directly on the first compute, captures it on the second and replays it from the
// third. The test counts the backend's own debug lines to show where a graph was captured and where one was found stale.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

struct graph_log {
    int captured = 0; // "CUDA graph warmup complete": the graph was captured on this compute
    int stale    = 0; // "CUDA graph warmup reset": a captured graph no longer matched, the compute ran directly
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

static ggml_context * new_ctx(size_t n_tensors) {
    ggml_init_params params = { ggml_tensor_overhead() * n_tensors + ggml_graph_overhead(), nullptr, true };
    return ggml_init(params);
}

// y = w x on the CPU backend, w given as the bytes of a [k, n] tensor of the given type
static std::vector<float> cpu_mul_mat(ggml_backend_t cpu, ggml_type type, const std::vector<uint8_t> & w,
        const std::vector<float> & x, int64_t k, int64_t n, int64_t m) {
    ggml_context * ctx = new_ctx(3);
    ggml_tensor * wt = ggml_new_tensor_2d(ctx, type, k, n);
    ggml_tensor * xt = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, m);
    ggml_tensor * yt = ggml_mul_mat(ctx, wt, xt);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, cpu);
    ggml_backend_tensor_set(wt, w.data(), 0, w.size());
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

static std::vector<float> random_values(std::mt19937 & rng, size_t n) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> v(n);
    for (float & f : v) {
        f = dist(rng);
    }
    return v;
}

// computes gf on backend once; returns what the graph log recorded for that compute
static const char * compute(ggml_backend_t backend, ggml_cgraph * gf, graph_log & log, bool & replay_armed) {
    const graph_log before = log;
    GGML_ASSERT(ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(backend);
    if (log.stale > before.stale) {
        replay_armed = false;
        return "stale graph found, ran directly";
    }
    if (log.captured > before.captured) {
        replay_armed = true;
        return "captured";
    }
    return replay_armed ? "replayed (captured earlier, nothing found stale since)" : "ran directly";
}

static constexpr double max_nmse = 5e-4; // test-backend-ops' bound for MUL_MAT

// y = w x on backend with w f16, three computes; then w retyped in place to bf16 with new bf16 bytes at the same
// address, three more computes, each checked against the CPU on bf16. Sets replayed when a graph was replayed
// before the retype.
static bool test_retype(ggml_backend_t backend, ggml_backend_t cpu, graph_log & log, int64_t m, bool & replayed) {
    const int64_t k = 256;
    const int64_t n = 64;
    std::mt19937 rng(42 + m);

    ggml_context * ctx = new_ctx(3);
    ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, k, n);
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, m);
    ggml_tensor * y = ggml_mul_mat(ctx, w, x);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);

    const std::vector<float> xv = random_values(rng, k * m);
    ggml_backend_tensor_set(x, xv.data(), 0, xv.size() * sizeof(float));

    std::vector<float> yv(n * m);
    bool ok = true;
    bool replay_armed = false;

    std::vector<uint8_t> w_f16(k * n * sizeof(ggml_fp16_t));
    {
        const std::vector<float> wv = random_values(rng, k * n);
        ggml_fp32_to_fp16_row(wv.data(), (ggml_fp16_t *) w_f16.data(), k * n);
    }
    ggml_backend_tensor_set(w, w_f16.data(), 0, w_f16.size());
    const std::vector<float> ref_f16 = cpu_mul_mat(cpu, GGML_TYPE_F16, w_f16, xv, k, n, m);
    for (int i = 1; i <= 3; ++i) {
        const char * what = compute(backend, gf, log, replay_armed);
        ggml_backend_tensor_get(y, yv.data(), 0, yv.size() * sizeof(float));
        const double e = nmse(yv, ref_f16);
        const bool pass = e < max_nmse;
        ok &= pass;
        replayed |= i == 3 && replay_armed;
        printf("  m=%-2lld f16  compute %d: %-55s nmse %.2e vs CPU f16: %s\n", (long long) m, i, what, e, pass ? "ok" : "FAILED");
    }

    // retype in place: the same tensor struct, data pointer, ne and nb, new type and new bytes
    const size_t nb[GGML_MAX_DIMS] = { w->nb[0], w->nb[1], w->nb[2], w->nb[3] };
    w->type = GGML_TYPE_BF16;
    GGML_ASSERT(ggml_type_size(GGML_TYPE_BF16) == ggml_type_size(GGML_TYPE_F16) && ggml_blck_size(GGML_TYPE_BF16) == 1);
    GGML_ASSERT(memcmp(nb, w->nb, sizeof(nb)) == 0);
    std::vector<uint8_t> w_bf16(k * n * sizeof(ggml_bf16_t));
    {
        const std::vector<float> wv = random_values(rng, k * n);
        ggml_fp32_to_bf16_row(wv.data(), (ggml_bf16_t *) w_bf16.data(), k * n);
    }
    ggml_backend_tensor_set(w, w_bf16.data(), 0, w_bf16.size());
    const std::vector<float> ref_bf16  = cpu_mul_mat(cpu, GGML_TYPE_BF16, w_bf16, xv, k, n, m);
    const std::vector<float> ref_stale = cpu_mul_mat(cpu, GGML_TYPE_F16,  w_bf16, xv, k, n, m); // the bf16 bytes read as f16
    for (int i = 4; i <= 6; ++i) {
        const char * what = compute(backend, gf, log, replay_armed);
        ggml_backend_tensor_get(y, yv.data(), 0, yv.size() * sizeof(float));
        const double e     = nmse(yv, ref_bf16);
        const double stale = nmse(yv, ref_stale);
        const bool pass = e < max_nmse;
        ok &= pass;
        printf("  m=%-2lld bf16 compute %d: %-55s nmse %.2e vs CPU bf16: %s", (long long) m, i, what, e, pass ? "ok" : "FAILED");
        if (!pass && stale < max_nmse) {
            printf(" (nmse %.2e vs the bf16 bytes read as f16: the f16 kernel ran on them)", stale);
        }
        printf("\n");
    }

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
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
        ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
        printf("%s (%s):\n", ggml_backend_dev_name(dev), ggml_backend_dev_description(dev));
        bool replayed = false;
        for (int64_t m : { 1, 8 }) {
            ok &= test_retype(backend, cpu, log, m, replayed);
        }
        if (!replayed) {
            printf("  no graph was replayed before the retype: this device does not exercise the stale-replay path\n");
        }
        ggml_backend_free(backend);
        n_tested++;
    }
    ggml_backend_free(cpu);
    ggml_log_set(nullptr, nullptr);

    if (n_tested == 0) {
        printf("no GPU device, skipped\n");
        return 0;
    }
    printf("%s\n", ok ? "OK" : "FAILED");
    return ok ? 0 : 1;
}
