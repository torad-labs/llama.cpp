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

#include <algorithm>
#include <cmath>
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

    // both forms of y = w x must run on this backend
    bool supported = ggml_backend_supports_op(backend, y);
    w->type = GGML_TYPE_BF16;
    supported = supported && ggml_backend_supports_op(backend, y);
    w->type = GGML_TYPE_F16;
    if (!supported) {
        printf("  m=%-2lld: MUL_MAT with an f16 or a bf16 weight is not supported here, skipped\n", (long long) m);
        ggml_free(ctx);
        return true;
    }

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

// Repeated parameter changes must preserve the caller's scalar result through capture and replay.
// The real-Nsight companion check also requires every update's source graph/node IDs to stay valid.
static bool test_parameter_updates(ggml_backend_t backend, graph_log & log) {
    constexpr size_t n = 256;
    std::mt19937 rng(53074);
    const auto input = random_values(rng, n);
    std::vector<float> output(n);
    auto * ctx = new_ctx(2);
    auto * x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    auto * y = ggml_scale(ctx, x, 1.0f);
    auto * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, y);
    auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    GGML_ASSERT(buffer);
    ggml_backend_tensor_set(x, input.data(), 0, n*sizeof(float));
    bool ok = true;
    bool armed = false;
    for (int step = 0; step < 24; ++step) {
        const float scale = float(int(rng()%33) - 16)/8.0f;
        memcpy(y->op_params, &scale, sizeof(scale));
        for (int run = 0; run < 3; ++run) {
            compute(backend, graph, log, armed);
            ggml_backend_tensor_get(y, output.data(), 0, n*sizeof(float));
            for (size_t i = 0; i < n; ++i) {
                ok &= output[i] == input[i]*scale;
            }
        }
    }
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    printf("  generated scalar parameter updates: %s\n", ok ? "ok" : "FAILED");
    return ok;
}

static bool test_moe_replay(ggml_backend_t backend, ggml_backend_t cpu, graph_log & log, bool partitioned = true) {
    constexpr int k = 256, rows = 264, used = 8, total = 288;
    const int local = partitioned ? 72 : total;
    const int first = partitioned ? 216 : 0;
    constexpr int graph_size = 32;
    ggml_backend_t backends[] = { backend, cpu };
    ggml_context * wctx[2];
    ggml_backend_buffer_t wbuf[2];
    ggml_tensor * weights[2][2];
    ggml_gallocr_t alloc[2];
    std::vector<uint8_t> meta[2];
    std::mt19937 rng(53073);
    for (int b = 0; b < 2; ++b) {
        wctx[b] = new_ctx(2);
        for (int w = 0; w < 2; ++w) {
            weights[b][w] = ggml_new_tensor_3d(wctx[b], GGML_TYPE_Q8_0, k, rows, local);
        }
        wbuf[b] = ggml_backend_alloc_ctx_tensors(wctx[b], backends[b]);
        GGML_ASSERT(wbuf[b]);
        ggml_backend_buffer_set_usage(wbuf[b], GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        alloc[b] = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backends[b]));
        meta[b].resize(ggml_tensor_overhead()*16 + ggml_graph_overhead_custom(graph_size, false));
    }
    for (int w = 0; w < 2; ++w) {
        const auto values = random_values(rng, k*rows*local);
        std::vector<uint8_t> quantized(ggml_nbytes(weights[0][w]));
        ggml_quantize_chunk(GGML_TYPE_Q8_0, values.data(), quantized.data(), 0, rows*local, k, nullptr);
        for (int b = 0; b < 2; ++b) {
            ggml_backend_tensor_set(weights[b][w], quantized.data(), 0, quantized.size());
        }
    }

    bool ok = true;
    bool replayed = false;
    const std::vector<int64_t> widths = partitioned ? std::vector<int64_t>{36, 72, 108, 36} : std::vector<int64_t>{23, 69, 23};
    for (int64_t tokens : widths) {
        ggml_context * ctx[2];
        ggml_cgraph * graph[2];
        ggml_tensor * x[2];
        ggml_tensor * ids[2];
        ggml_tensor * outputs[2][3];
        for (int b = 0; b < 2; ++b) {
            ggml_init_params params = { meta[b].size(), meta[b].data(), true };
            ctx[b] = ggml_init(params);
            x[b] = ggml_new_tensor_3d(ctx[b], GGML_TYPE_F32, k, 1, tokens);
            ggml_tensor * ids_base = ggml_new_tensor_2d(ctx[b], GGML_TYPE_I32, used + 3, tokens);
            ids[b] = ggml_view_2d(ctx[b], ids_base, used, tokens, ids_base->nb[1], 0);
            ggml_tensor * gate = ggml_mul_mat_id(ctx[b], weights[b][0], x[b], ids[b]);
            ggml_tensor * up = ggml_mul_mat_id(ctx[b], weights[b][1], x[b], ids[b]);
            if (partitioned) {
                ggml_mul_mat_id_set_expert_range(gate, first, total);
                ggml_mul_mat_id_set_expert_range(up, first, total);
            }
            outputs[b][0] = gate;
            outputs[b][1] = up;
            if (tokens != (partitioned ? 72 : 69)) {
                gate = ggml_clamp(ctx[b], gate, -INFINITY, 7.0f);
                up = ggml_clamp(ctx[b], up, -7.0f, 7.0f);
            }
            outputs[b][2] = ggml_swiglu_split(ctx[b], gate, up);
            graph[b] = ggml_new_graph_custom(ctx[b], graph_size, false);
            ggml_build_forward_expand(graph[b], outputs[b][2]);
            for (auto * out : outputs[b]) { ggml_set_output(out); }
            GGML_ASSERT(ggml_gallocr_alloc_graph(alloc[b], graph[b]));
        }
        bool armed = false;
        for (int run = 0; run < 5; ++run) {
            auto xv = random_values(rng, k*tokens);
            std::vector<int32_t> routes((used + 3)*tokens);
            for (int64_t t = 0; t < tokens; ++t) {
                for (int s = 0; s < used + 3; ++s) {
                    int id = (17*t + 37*s + run) % total;
                    if (run == 1) { id = first + local - 1 - s; }
                    if (run == 2) { id = s; }
                    if (run == 3) { id = s == 0 ? first + local - 1 : s; }
                    if (run == 4) { id = first + (13*t + s) % local; }
                    routes[t*(used + 3) + s] = id;
                }
            }
            for (int b = 0; b < 2; ++b) {
                ggml_backend_tensor_set(x[b], xv.data(), 0, xv.size()*sizeof(float));
                ggml_backend_tensor_set(ids[b]->view_src, routes.data(), 0, routes.size()*sizeof(int32_t));
            }
            compute(backend, graph[0], log, armed);
            replayed |= run >= 2 && armed;
            GGML_ASSERT(ggml_backend_graph_compute(cpu, graph[1]) == GGML_STATUS_SUCCESS);
            for (int o = 0; o < 3; ++o) {
                std::vector<float> actual(rows*used*tokens), expected(actual.size());
                ggml_backend_tensor_get(outputs[0][o], actual.data(), 0, actual.size()*sizeof(float));
                ggml_backend_tensor_get(outputs[1][o], expected.data(), 0, expected.size()*sizeof(float));
                double error = 0.0, norm = 0.0;
                for (size_t j = 0; j < actual.size(); ++j) {
                    const int slot = (j/rows) % used;
                    const int token = j/(rows*used);
                    const int expert = routes[token*(used + 3) + slot];
                    if (!std::isfinite(actual[j]) || (expert < first && actual[j] != 0.0f)) { ok = false; }
                    error += double(actual[j] - expected[j])*(actual[j] - expected[j]);
                    norm += double(expected[j])*expected[j];
                }
                if (error/std::max(norm, 1e-30) > (o == 2 ? 5e-3 : 5e-4)) { ok = false; }
            }
        }
        for (int b = 0; b < 2; ++b) { ggml_free(ctx[b]); }
    }
    for (int b = 0; b < 2; ++b) {
        ggml_gallocr_free(alloc[b]);
        ggml_backend_buffer_free(wbuf[b]);
        ggml_free(wctx[b]);
    }
    printf("  expert pair (%s): changed inputs/routes, tails and rebuilt metadata: %s (%s)\n",
           partitioned ? "partitioned" : "all experts", ok ? "ok" : "FAILED",
           replayed ? "captured replay exercised" : "no CUDA replay");
    return ok;
}

static bool test_dense_q8_replay(ggml_backend_t backend, ggml_backend_t cpu, graph_log & log,
        int64_t k, bool reverse_rows) {
    constexpr int graph_size = 16;
    const int64_t rows[] = { reverse_rows ? 128 : 64, reverse_rows ? 64 : 128, 256 };
    ggml_context * wctx = new_ctx(3);
    ggml_tensor * weights[3];
    std::vector<uint8_t> quantized[3];
    std::mt19937 rng(53123);
    for (int j = 0; j < 3; ++j) {
        weights[j] = ggml_new_tensor_2d(wctx, GGML_TYPE_Q8_0, k, rows[j]);
        const auto values = random_values(rng, k*rows[j]);
        quantized[j].resize(ggml_nbytes(weights[j]));
        ggml_quantize_chunk(GGML_TYPE_Q8_0, values.data(), quantized[j].data(), 0, rows[j], k, nullptr);
    }
    ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors(wctx, backend);
    GGML_ASSERT(wbuf);
    ggml_backend_buffer_set_usage(wbuf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    for (int j = 0; j < 3; ++j) {
        ggml_backend_tensor_set(weights[j], quantized[j].data(), 0, quantized[j].size());
    }

    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    std::vector<uint8_t> meta(ggml_tensor_overhead()*8 + ggml_graph_overhead_custom(graph_size, false));
    bool ok = true;
    bool replayed = false;
    for (int64_t tokens : { 23, 69, 23 }) {
        const int readers = tokens == 69 ? 2 : 3;
        ggml_init_params params = { meta.size(), meta.data(), true };
        ggml_context * ctx = ggml_init(params);
        ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, tokens);
        ggml_set_input(x);
        ggml_tensor * outputs[3];
        ggml_cgraph * graph = ggml_new_graph_custom(ctx, graph_size, false);
        for (int j = 0; j < readers; ++j) {
            outputs[j] = ggml_mul_mat(ctx, weights[j], x);
            ggml_set_output(outputs[j]);
            ggml_build_forward_expand(graph, outputs[j]);
        }
        GGML_ASSERT(ggml_gallocr_alloc_graph(alloc, graph));
        bool armed = false;
        for (int run = 0; run < 5; ++run) {
            const auto xv = random_values(rng, k*tokens);
            ggml_backend_tensor_set(x, xv.data(), 0, xv.size()*sizeof(float));
            const bool was_armed = armed;
            const graph_log before = log;
            compute(backend, graph, log, armed);
            replayed |= was_armed && log.captured == before.captured && log.stale == before.stale;
            for (int j = 0; j < readers; ++j) {
                std::vector<float> actual(rows[j]*tokens);
                ggml_backend_tensor_get(outputs[j], actual.data(), 0, actual.size()*sizeof(float));
                const auto expected = cpu_mul_mat(cpu, GGML_TYPE_Q8_0, quantized[j], xv, k, rows[j], tokens);
                const double error = nmse(actual, expected);
                if (!std::isfinite(error) || error > max_nmse) {
                    printf("  dense Q8 k=%lld tokens=%lld reader=%d run=%d: NMSE %g\n",
                           (long long) k, (long long) tokens, j, run, error);
                    ok = false;
                }
            }
        }
        ggml_free(ctx);
    }
    ggml_gallocr_free(alloc);
    ggml_backend_buffer_free(wbuf);
    ggml_free(wctx);
    printf("  dense Q8 k=%lld rows=%lld,%lld,256: changed inputs and rebuilt pair/triple graphs: %s (%s)\n",
           (long long) k, (long long) rows[0], (long long) rows[1], ok ? "ok" : "FAILED",
           replayed ? "captured replay exercised" : "no CUDA replay");
    return ok;
}

int main(int argc, char ** argv) {
    ggml_backend_load_all();

    graph_log log;
    ggml_log_set(log_callback, &log);

    ggml_backend_t cpu = ggml_backend_cpu_init();
    bool ok = true;
    const bool moe_cpu = argc == 2 && strcmp(argv[1], "--moe-cpu") == 0;
    const bool moe_only = argc == 2 && strcmp(argv[1], "--moe") == 0;
    const bool dense_only = argc == 2 && strcmp(argv[1], "--dense-q8") == 0;
    if (moe_cpu) {
        ok = test_moe_replay(cpu, cpu, log);
        ok &= test_moe_replay(cpu, cpu, log, false);
        ggml_backend_free(cpu);
        ggml_log_set(nullptr, nullptr);
        return ok ? 0 : 1;
    }
    int n_tested = 0;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) {
            continue;
        }
        ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
        printf("%s (%s):\n", ggml_backend_dev_name(dev), ggml_backend_dev_description(dev));
        if (!dense_only) {
            ok &= test_moe_replay(backend, cpu, log);
            ok &= test_moe_replay(backend, cpu, log, false);
        }
        if (!moe_only) {
            for (int64_t k : { 256, 4096 }) {
                ok &= test_dense_q8_replay(backend, cpu, log, k, false);
                ok &= test_dense_q8_replay(backend, cpu, log, k, true);
            }
        }
        bool replayed = false;
        if (!moe_only && !dense_only) {
            ok &= test_parameter_updates(backend, log);
            for (int64_t m : { 1, 8 }) {
                ok &= test_retype(backend, cpu, log, m, replayed);
            }
        }
        if (!moe_only && !dense_only && !replayed) {
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
