// The meta backend (-sm tensor) reduces a prompt's activations across the two devices with the internal AllReduce. A
// reduction of 1 MB or more on the wire goes over the copy engines (ggml_cuda_ar_allreduce_copy_impl): each device
// copies its share to pinned host memory, copies its peer's share back from there, and adds it. Each device must end
// with the sum of both shares, whichever device reaches the reduction first, and at every reduction of a sequence that
// reuses the same staging, scratch and event slots.
//
// A stack of feed-forward layers, x = x + down(up(x)), is split over two devices with the up rows and the down columns
// cut unevenly, the first device light (32 rows) in even layers and the second in odd layers, so the device that
// reaches each reduction first changes from one reduction to the next. The late device's share is still being copied
// out when its peer's has long arrived. Three graphs take x of 256, 640 and 4,100 columns: 1 MB, 2.5 MB and 16 MB on the
// BF16 wire, and 2 MB, 5 MB and 33.6 MB on the F32 wire, which is past one pass of the 32 MB staging. They are evaluated
// A, B, C, A, B, C with a new x each time, and every output is compared with the CPU's. Three passes: the BF16 wire, the
// F32 wire (GGML_CUDA_AR_BF16_THRESHOLD=0, where each device's share is reduced in place, so a share overwritten before
// it was copied out reaches the peer already summed), and the F32 wire with GGML_CUDA_AR_COPY_ONE_STREAM=1. Each pass
// must initialize the internal AllReduce pipeline. On the first two GPUs, or on the first GPU twice; with no GPU the
// test is skipped.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <thread>
#include <vector>

static constexpr int64_t n_embd  = 2048;
static constexpr int64_t n_ff    = 4096;
static constexpr int64_t n_lo    = 32; // up rows and down columns on the light device
static constexpr int     n_layer = 4;

static constexpr int     n_graphs        = 3;
static constexpr int64_t n_tok[n_graphs] = {256, 640, 4100};
static constexpr int     order[]         = {0, 1, 2, 0, 1, 2};
static constexpr int     n_evals         = sizeof(order)/sizeof(order[0]);

static ggml_backend_meta_split_state split_by_name(const ggml_tensor * tensor, void * /*userdata*/) {
    const bool up = strncmp(tensor->name, "up.", 3) == 0;
    if (up || strncmp(tensor->name, "down.", 5) == 0) {
        const int il = atoi(tensor->name + (up ? 3 : 5));
        // up: [n_embd, n_ff], its rows; down: [n_ff, n_embd], its columns
        ggml_backend_meta_split_state split_state = {
            up ? GGML_BACKEND_SPLIT_AXIS_1 : GGML_BACKEND_SPLIT_AXIS_0, {0}, {1}, 1};
        split_state.ne[0] = il % 2 == 0 ? n_lo : n_ff - n_lo;
        split_state.ne[1] = n_ff - split_state.ne[0];
        return split_state;
    }
    return {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
}

static int n_pipelines = 0; // "initialized AllReduce pipeline" lines logged

static void log_cb(ggml_log_level level, const char * text, void * /*user_data*/) {
    if (strstr(text, "initialized AllReduce pipeline") != nullptr) {
        n_pipelines++;
        fputs(text, stderr);
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
    ggml_context *             ctx = nullptr;
    ggml_backend_buffer_t      buf = nullptr;
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

// the graphs over one set of weights, each allocated once by a scheduler of its own, as llama.cpp's graph slots are
struct ffn_stack {
    ffn_weights                             w;
    std::vector<std::unique_ptr<ffn_graph>> graphs;
    std::vector<ggml_backend_sched_t>       scheds;

    ffn_stack(ggml_backend_t * backends, int n_backends, ggml_backend_buffer_type_t buft,
              const std::vector<std::vector<float>> & w_data) : w(buft, w_data) {
        for (int k = 0; k < n_graphs; k++) {
            graphs.push_back(std::make_unique<ffn_graph>(w.up, w.down, n_tok[k]));
            scheds.push_back(ggml_backend_sched_new(backends, nullptr, n_backends, GGML_DEFAULT_GRAPH_SIZE, false, true));
            GGML_ASSERT(ggml_backend_sched_alloc_graph(scheds.back(), graphs.back()->gf));
        }
    }

    std::vector<float> compute(int k, const std::vector<float> & x) {
        ffn_graph & g = *graphs[k];
        ggml_backend_tensor_set(g.x, x.data(), 0, ggml_nbytes(g.x));
        GGML_ASSERT(ggml_backend_sched_graph_compute(scheds[k], g.gf) == GGML_STATUS_SUCCESS);
        std::vector<float> out(ggml_nelements(g.out));
        ggml_backend_tensor_get(g.out, out.data(), 0, ggml_nbytes(g.out));
        return out;
    }

    ~ffn_stack() {
        for (auto * sched : scheds) {
            ggml_backend_sched_free(sched);
        }
    }
};

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
    if (strncmp(ggml_backend_dev_name(gpus[0]), "CUDA", 4) != 0) {
        printf("%s has no internal AllReduce, skipped\n", ggml_backend_dev_name(gpus[0]));
        return 0;
    }
    printf("meta backend over %s and %s\n", ggml_backend_dev_name(gpus[0]), ggml_backend_dev_name(gpus[1]));

    ggml_log_set(log_cb, nullptr);
    setenv("GGML_CUDA_ALLREDUCE", "internal", 1);
    unsetenv("GGML_CUDA_AR_COPY_THRESHOLD");
    unsetenv("GGML_CUDA_AR_COPY_CHUNK_BYTES");

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

    // the inputs and the CPU's outputs, once for every pass
    std::vector<std::vector<float>> xs(n_evals);
    std::vector<std::vector<float>> refs(n_evals);
    {
        ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        ggml_backend_cpu_set_n_threads(cpu, (int) std::max(1u, std::thread::hardware_concurrency()));
        ffn_stack stack_cpu(&cpu, 1, ggml_backend_get_default_buffer_type(cpu), w_data);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (int e = 0; e < n_evals; e++) {
            xs[e].resize(n_embd*n_tok[order[e]]);
            for (float & v : xs[e]) {
                v = dist(rng);
            }
            refs[e] = stack_cpu.compute(order[e], xs[e]);
        }
        ggml_backend_free(cpu);
    }

    struct pass {
        const char * name;
        const char * bf16_threshold; // GGML_CUDA_AR_BF16_THRESHOLD, nullptr for the default
        bool         one_stream;     // GGML_CUDA_AR_COPY_ONE_STREAM=1
    };
    const pass passes[] = {
        {"BF16 wire",              nullptr, false},
        {"F32 wire",               "0",     false},
        {"F32 wire, one stream",   "0",     true},
    };

    bool ok = true;
    ggml_backend_dev_t meta = ggml_backend_meta_device(gpus.data(), 2, split_by_name, nullptr);
    for (const pass & ps : passes) {
        if (ps.bf16_threshold) {
            setenv("GGML_CUDA_AR_BF16_THRESHOLD", ps.bf16_threshold, 1);
        } else {
            unsetenv("GGML_CUDA_AR_BF16_THRESHOLD");
        }
        if (ps.one_stream) {
            setenv("GGML_CUDA_AR_COPY_ONE_STREAM", "1", 1);
        } else {
            unsetenv("GGML_CUDA_AR_COPY_ONE_STREAM");
        }

        const int n_pipelines_before = n_pipelines;
        ggml_backend_t cpu     = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        ggml_backend_t backend = ggml_backend_dev_init(meta, nullptr);
        {
            ggml_backend_t backends[2] = {backend, cpu};
            ffn_stack stack(backends, 2, ggml_backend_dev_buffer_type(meta), w_data);
            for (int e = 0; e < n_evals; e++) {
                const int k = order[e];
                const std::vector<float> out = stack.compute(k, xs[e]);
                const double err = nmse(refs[e], out);
                const bool ok_eval = std::isfinite(err) && err < 1e-4;
                printf("%s %s, eval %d, %lld columns: NMSE vs CPU %.3e\n", ok_eval ? "ok  " : "FAIL", ps.name, e + 1,
                       (long long) n_tok[k], err);
                ok = ok && ok_eval;
            }
        }
        const bool ok_pipeline = n_pipelines == n_pipelines_before + 1;
        printf("%s %s: %d internal AllReduce pipeline initialized (1 expected)\n", ok_pipeline ? "ok  " : "FAIL",
               ps.name, n_pipelines - n_pipelines_before);
        ok = ok && ok_pipeline;
        ggml_backend_free(backend);
        ggml_backend_free(cpu);
    }
    ggml_log_set(nullptr, nullptr);

    printf("%s\n", ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}
