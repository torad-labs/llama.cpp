// The meta backend (-sm tensor) captures a cgraph's whole evaluation, every step's subgraph and all-reduce on every
// device, into one graph a device on its second evaluation, and replays those graphs for the evaluations after it, for
// the last 4 cgraphs evaluated (ggml_backend_meta_graph_compute). A replay runs the all-reduces with no host between
// them, so each one takes its token and staging slot from the device (ggml_cuda_ar_kernel).
//
// A stack of feed-forward layers, x = x + down(up(x)), is split over two devices with the up rows and the down columns
// cut unevenly (n_lo on the first device, the rest on the second), so the second device reaches each all-reduce long
// after the first: a reduction that took a stale token or slot sums the second device's share of an earlier step.
// Five graphs, A to E, take x of 1 to 5 columns: A and B are evaluated in turn 6 times each, then C, D and E 3 times
// each (E's capture evicts A's), then A 3 times (captured again), each time with a new x; every output is compared
// with the CPU's. The CUDA backend logs each executable it makes; the test counts them, one a device for each capture
// (A, B, C, D, E, A again), or none under GGML_META_CAPTURE_LEGACY=1. On the first two GPUs, or on the first GPU
// twice; with no GPU the test is skipped.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <vector>

static constexpr int64_t n_embd  = 2048;
static constexpr int64_t n_ff    = 4096;
static constexpr int64_t n_lo    = 32; // up rows and down columns on the first device
static constexpr int     n_layer = 4;

static ggml_backend_meta_split_state split_by_name(const ggml_tensor * tensor, void * /*userdata*/) {
    if (strncmp(tensor->name, "up", 2) == 0 || strncmp(tensor->name, "down", 4) == 0) {
        // up: [n_embd, n_ff], its rows; down: [n_ff, n_embd], its columns
        ggml_backend_meta_split_state split_state = {
            tensor->name[0] == 'u' ? GGML_BACKEND_SPLIT_AXIS_1 : GGML_BACKEND_SPLIT_AXIS_0, {0}, {1}, 1};
        split_state.ne[0] = n_lo;
        split_state.ne[1] = n_ff - n_lo;
        return split_state;
    }
    return {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
}

struct log_counts {
    int executables  = 0; // ggml_backend_cuda_capture_end made one
    int not_captured = 0; // the meta backend refused a capture
};

static void log_cb(ggml_log_level level, const char * text, void * user_data) {
    log_counts * counts = (log_counts *) user_data;
    if (strstr(text, "ggml_backend_cuda_capture_end") != nullptr && strstr(text, " nodes, ") != nullptr) {
        counts->executables++;
    }
    if (strstr(text, "not captured") != nullptr) {
        counts->not_captured++;
    }
    if (level == GGML_LOG_LEVEL_WARN || level == GGML_LOG_LEVEL_ERROR) {
        fputs(text, stderr);
    }
}

// the layers over the weights in ctx_w, on an input of n_tok columns
struct ffn_graph {
    ggml_context * ctx = nullptr;
    ggml_cgraph  * gf  = nullptr;
    ggml_tensor  * x   = nullptr;
    ggml_tensor  * out = nullptr;

    ffn_graph(const std::vector<ggml_tensor *> & up, const std::vector<ggml_tensor *> & down, int64_t n_tok) {
        const ggml_init_params params = {
            /*.mem_size   =*/ (4*n_layer + 2)*ggml_tensor_overhead() + ggml_graph_overhead(),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ctx = ggml_init(params);
        x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_tok);
        ggml_set_input(x);
        ggml_tensor * cur = x;
        for (int il = 0; il < n_layer; il++) {
            ggml_tensor * h = ggml_mul_mat(ctx, up[il], cur);
            ggml_tensor * y = ggml_mul_mat(ctx, down[il], h);
            cur = ggml_add(ctx, y, cur);
        }
        out = cur;
        ggml_set_output(out);
        gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out);
    }

    ~ffn_graph() {
        ggml_free(ctx);
    }
};

// the weights, in a buffer of buft
struct ffn_weights {
    ggml_context *            ctx = nullptr;
    ggml_backend_buffer_t     buf = nullptr;
    std::vector<ggml_tensor *> up;
    std::vector<ggml_tensor *> down;

    ffn_weights(ggml_backend_buffer_type_t buft, const std::vector<std::vector<float>> & data) {
        const ggml_init_params params = {
            /*.mem_size   =*/ 2*n_layer*ggml_tensor_overhead(),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ctx = ggml_init(params);
        for (int il = 0; il < n_layer; il++) {
            up.push_back(ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_ff));
            ggml_format_name(up.back(), "up.%d", il);
            down.push_back(ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_ff, n_embd));
            ggml_format_name(down.back(), "down.%d", il);
        }
        buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
        GGML_ASSERT(buf != nullptr);
        ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        for (int il = 0; il < n_layer; il++) {
            ggml_backend_tensor_set(up[il],   data[2*il + 0].data(), 0, ggml_nbytes(up[il]));
            ggml_backend_tensor_set(down[il], data[2*il + 1].data(), 0, ggml_nbytes(down[il]));
        }
    }

    ~ffn_weights() {
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }
};

static std::vector<float> compute(ggml_backend_sched_t sched, ffn_graph & g, const std::vector<float> & x) {
    ggml_backend_tensor_set(g.x, x.data(), 0, ggml_nbytes(g.x));
    GGML_ASSERT(ggml_backend_sched_graph_compute(sched, g.gf) == GGML_STATUS_SUCCESS);
    std::vector<float> out(ggml_nelements(g.out));
    ggml_backend_tensor_get(g.out, out.data(), 0, ggml_nbytes(g.out));
    return out;
}

static double nmse(const std::vector<float> & ref, const std::vector<float> & v) {
    double err = 0.0;
    double sum = 0.0;
    for (size_t i = 0; i < ref.size(); i++) {
        err += (v[i] - ref[i])*(v[i] - ref[i]);
        sum += ref[i]*ref[i];
    }
    return err/sum;
}

int main() {
    std::vector<ggml_backend_dev_t> gpus;
    for (size_t i = 0; i < ggml_backend_dev_count() && gpus.size() < 2; i++) {
        if (ggml_backend_dev_type(ggml_backend_dev_get(i)) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            gpus.push_back(ggml_backend_dev_get(i));
        }
    }
    if (gpus.empty()) {
        printf("no GPU for the meta backend, skipped\n");
        return 0;
    }
    if (gpus.size() == 1) {
        gpus.push_back(gpus[0]);
    }
    printf("meta backend over %s and %s\n", ggml_backend_dev_name(gpus[0]), ggml_backend_dev_name(gpus[1]));

    log_counts counts;
    ggml_log_set(log_cb, &counts);

    ggml_backend_dev_t meta = ggml_backend_meta_device(gpus.data(), 2, split_by_name, nullptr);

    std::mt19937 rng(42);
    std::vector<std::vector<float>> w_data(2*n_layer);
    for (int il = 0; il < n_layer; il++) {
        for (int k = 0; k < 2; k++) {
            // up is [n_embd, n_ff] and down [n_ff, n_embd]: uniform over +-1/sqrt(the row length)
            const float scale = 1.0f/std::sqrt((float) (k == 0 ? n_embd : n_ff));
            std::uniform_real_distribution<float> dist(-scale, scale);
            w_data[2*il + k].resize(n_embd*n_ff);
            for (float & v : w_data[2*il + k]) {
                v = dist(rng);
            }
        }
    }

    ggml_backend_t cpu     = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    ggml_backend_t backend = ggml_backend_dev_init(meta, nullptr);

    ffn_weights w_meta(ggml_backend_dev_buffer_type(meta), w_data);
    ffn_weights w_cpu(ggml_backend_get_default_buffer_type(cpu), w_data);

    // each graph is allocated once by a scheduler of its own, as llama.cpp's graph slots are
    constexpr int n_graphs = 5;
    std::vector<std::unique_ptr<ffn_graph>> graphs;
    std::vector<std::unique_ptr<ffn_graph>> graphs_cpu;
    std::vector<ggml_backend_sched_t>       scheds;
    std::vector<ggml_backend_sched_t>       scheds_cpu;
    ggml_backend_t backends[2] = {backend, cpu};
    for (int k = 0; k < n_graphs; k++) {
        graphs.push_back(std::make_unique<ffn_graph>(w_meta.up, w_meta.down, k + 1));
        graphs_cpu.push_back(std::make_unique<ffn_graph>(w_cpu.up, w_cpu.down, k + 1));
        scheds.push_back(ggml_backend_sched_new(backends, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE, false, true));
        GGML_ASSERT(ggml_backend_sched_alloc_graph(scheds.back(), graphs.back()->gf));
        scheds_cpu.push_back(ggml_backend_sched_new(&cpu, nullptr, 1, GGML_DEFAULT_GRAPH_SIZE, false, true));
        GGML_ASSERT(ggml_backend_sched_alloc_graph(scheds_cpu.back(), graphs_cpu.back()->gf));
    }

    bool ok = true;
    int  n_eval = 0;
    auto run = [&](int k) {
        std::vector<float> x(ggml_nelements(graphs[k]->x));
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (float & v : x) {
            v = dist(rng);
        }
        const std::vector<float> out = compute(scheds[k], *graphs[k], x);
        const std::vector<float> ref = compute(scheds_cpu[k], *graphs_cpu[k], x);
        const double e = nmse(ref, out);
        const bool ok_eval = std::isfinite(e) && e < 1e-4;
        printf("%s eval %2d, graph %c: NMSE vs CPU %.3e\n", ok_eval ? "ok  " : "FAIL", ++n_eval, 'A' + k, e);
        ok = ok && ok_eval;
    };
    for (int i = 0; i < 6; i++) {
        run(0);
        run(1);
    }
    for (int k = 2; k < n_graphs; k++) {
        for (int i = 0; i < 3; i++) {
            run(k);
        }
    }
    for (int i = 0; i < 3; i++) {
        run(0);
    }

    const bool legacy      = ggml_env_switch("GGML_META_CAPTURE_LEGACY"); // as the backend reads it
    const int  executables = legacy ? 0 : (n_graphs + 1)*2;
    const bool ok_captures = counts.executables == executables && counts.not_captured == 0;
    printf("%s %d executables made (%d expected), %d graphs not captured\n", ok_captures ? "ok  " : "FAIL",
        counts.executables, executables, counts.not_captured);
    ok = ok && ok_captures;

    for (int k = 0; k < n_graphs; k++) {
        ggml_backend_sched_free(scheds_cpu[k]);
        ggml_backend_sched_free(scheds[k]);
    }
    graphs.clear();
    graphs_cpu.clear();
    ggml_backend_free(backend);
    ggml_backend_free(cpu);
    ggml_log_set(nullptr, nullptr);

    printf("%s\n", ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}
