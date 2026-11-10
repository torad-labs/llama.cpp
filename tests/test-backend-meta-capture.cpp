// The meta backend (-sm tensor) captures a cgraph's whole evaluation, every step's subgraph and all-reduce on every
// device, into one graph a device on its second evaluation, and replays those graphs for the evaluations after it, for
// four graph slots, including their scheduler splits (ggml_backend_meta_graph_compute). A replay runs the all-reduces with no host between
// them, so each one takes its token and staging slot from the device (ggml_cuda_ar_kernel).
//
// A stack of feed-forward layers, x = x + down(up(x)), is split over two devices with the up rows and the down columns
// cut unevenly (n_lo on the first device, the rest on the second), so the second device reaches each all-reduce long
// after the first: a reduction that took a stale token or slot sums the second device's share of an earlier step.
// Five graphs, A to E, take x of 1, 2, 3, 4 and 40 columns: A and B are evaluated in turn 6 times each, then C, D and
// E 3 times each (E's capture evicts A's), then A 3 times (captured again), each time with a new x; every output is
// compared with the CPU's. A to D reduce 4 to 16 KB on the BF16 wire, which the LL kernel takes (card to card where the
// cards can map each other's memory), and E 160 KB, which the chunked kernel takes, so the replays mix the kernels that
// share the device's token. The CUDA backend logs each executable it makes; the test counts them, one a device for
// each capture (A, B, C, D, E, A again), or none under GGML_META_CAPTURE_LEGACY=1. On the first two GPUs, or on the
// first GPU twice; with no GPU the test is skipped.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <vector>

#if __has_include(<nvtx3/nvToolsExt.h>)
#include <nvtx3/nvToolsExt.h>
#define TEST_HAS_NVTX 1
#else
#define TEST_HAS_NVTX 0
#endif

static constexpr int64_t n_embd  = 2048;
static constexpr int64_t n_ff    = 4096;
static constexpr int64_t n_lo    = 32; // up rows and down columns on the first device
static constexpr int     n_layer = 4;

static ggml_backend_meta_split_state split_by_name(const ggml_tensor * tensor, void * userdata) {
    if (strncmp(tensor->name, "up", 2) == 0 || strncmp(tensor->name, "down", 4) == 0) {
        // up: [n_embd, n_ff], its rows; down: [n_ff, n_embd], its columns
        ggml_backend_meta_split_state split_state = {
            tensor->name[0] == 'u' ? GGML_BACKEND_SPLIT_AXIS_1 : GGML_BACKEND_SPLIT_AXIS_0, {0}, {1}, 1};
        const int n = *(const int *) userdata;
        const int layer = atoi(strchr(tensor->name, '.') + 1);
        const int heavy = n == 2 ? 1 : layer % n;
        const int64_t width = tensor->ne[tensor->name[0] == 'u' ? 1 : 0];
        for (int rank = 0; rank < n; ++rank) {
            split_state.ne[rank] = rank == heavy ? width - (n - 1)*n_lo : n_lo;
        }
        return split_state;
    }
    return {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
}

struct log_counts {
    int executables  = 0; // ggml_backend_cuda_capture_end made one
    int evicted_executables = 0;
    int hc_fused = 0;
    int not_captured = 0; // the meta backend refused a capture
    size_t p2p_bytes = 0;
    bool p2p_reported = false;
    bool p2p_unsupported = false;
    bool p2p_failed = false;
    bool capture_details = false;
};

static void log_cb(ggml_log_level level, const char * text, void * user_data) {
    log_counts * counts = (log_counts *) user_data;
    if (strstr(text, "captured LL collective + HC_POST") != nullptr) { counts->hc_fused++; }
    if (strstr(text, "ggml_backend_cuda_capture_end") != nullptr && strstr(text, " nodes, ") != nullptr) {
        counts->executables++;
    }
    if (strstr(text, "layer=meta event=evict") != nullptr) {
        const char * size = strstr(text, "executables=");
        if (size) { counts->evicted_executables += atoi(size + strlen("executables=")); }
    }
    if (strstr(text, "not captured") != nullptr) {
        counts->not_captured++;
    }
    const char * p2p = strstr(text, "LL card to card up to ");
    if (p2p != nullptr) {
        counts->p2p_reported = sscanf(p2p, "LL card to card up to %zu bytes", &counts->p2p_bytes) == 1;
    }
    const bool unsupported = strstr(text, "cannot map each other's memory") != nullptr;
    counts->p2p_unsupported |= unsupported;
    const bool error = level == GGML_LOG_LEVEL_WARN || level == GGML_LOG_LEVEL_ERROR;
    counts->p2p_failed |= error && strstr(text, "ggml_cuda_ar_p2p_init") != nullptr;
    const bool capture_detail = strstr(text, "layer=meta ") || strstr(text, "layer=cuda_outer ") ||
        strstr(text, "ggml_backend_cuda_capture_end");
    if (error || p2p != nullptr || unsupported || (counts->capture_details && capture_detail)) {
        fputs(text, stderr);
    }
}

// the layers over the weights in ctx_w, on an input of n_tok columns
struct ffn_graph {
    ggml_context * ctx = nullptr;
    ggml_cgraph  * gf  = nullptr;
    ggml_tensor  * x   = nullptr;
    ggml_tensor  * out = nullptr;
    ggml_tensor  * residual = nullptr;
    ggml_tensor  * post = nullptr;
    ggml_tensor  * comb = nullptr;

    ffn_graph(const std::vector<ggml_tensor *> & up, const std::vector<ggml_tensor *> & down, int64_t n_tok) {
        const ggml_init_params params = {
            /*.mem_size   =*/ (4*n_layer + 8)*ggml_tensor_overhead() + ggml_graph_overhead(),
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
            if (il == n_layer - 1 && ggml_env_switch("TEST_GLM_HC")) {
                residual = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, 4, n_tok);
                post = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, n_tok);
                comb = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 4, 4, n_tok);
                for (auto * input : {residual, post, comb}) { ggml_set_input(input); }
                cur = ggml_dsv4_hc_post(ctx, y, residual, post, comb);
            } else {
                cur = ggml_add(ctx, y, cur);
            }
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
    for (auto * input : {g.residual, g.post, g.comb}) {
        if (!input) { continue; }
        std::vector<float> data(ggml_nelements(input));
        for (size_t i = 0; i < data.size(); ++i) { data[i] = 0.1f + 0.25f*x[(17*i + 5)%x.size()]; }
        ggml_backend_tensor_set(input, data.data(), 0, ggml_nbytes(input));
    }
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

// A weight-prefetch hint does not authorize the following collective to execute its consumer.
static bool check_collective_hint_scope(const std::vector<ggml_backend_dev_t> & gpus) {
    auto reg = ggml_backend_dev_backend_reg(gpus[0]);
    const auto init = (ggml_backend_comm_init_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_comm_init");
    const auto free_comm = (ggml_backend_comm_free_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_comm_free");
    const auto reduce = (ggml_backend_comm_allreduce_tensor_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_comm_allreduce_tensor");
    const auto hint = (ggml_backend_graph_next_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_graph_next");
    const auto begin = (ggml_backend_capture_begin_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_capture_begin");
    const auto end = (ggml_backend_capture_end_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_capture_end");
    const auto launch = (ggml_backend_capture_launch_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_capture_launch");
    const auto free_exec = (ggml_backend_capture_free_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_capture_free");
    GGML_ASSERT(init && free_comm && reduce && hint && begin && end && launch && free_exec);
    ggml_backend_t backends[2];
    ggml_context * contexts[2];
    ggml_backend_buffer_t buffers[2];
    ggml_tensor * sums[2];
    ggml_tensor * consumers[2];
    ggml_cgraph * prefixes[2];
    ggml_cgraph * hints[2];
    void * execs[2];
    for (int rank = 0; rank < 2; ++rank) {
        backends[rank] = ggml_backend_dev_init(gpus[rank], nullptr);
        contexts[rank] = ggml_init({16*ggml_tensor_overhead() + 2*ggml_graph_overhead(), nullptr, true});
        auto * ctx = contexts[rank];
        auto * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, 1);
        sums[rank] = ggml_scale(ctx, x, 1.0f);
        auto * residual = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, 4, 1);
        auto * post = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 1);
        auto * comb = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 4, 4, 1);
        consumers[rank] = ggml_dsv4_hc_post(ctx, sums[rank], residual, post, comb);
        sums[rank]->flags |= GGML_TENSOR_FLAG_COMPUTE;
        consumers[rank]->flags |= GGML_TENSOR_FLAG_COMPUTE;
        prefixes[rank] = ggml_new_graph(ctx);
        ggml_build_forward_expand(prefixes[rank], sums[rank]);
        hints[rank] = ggml_new_graph(ctx);
        ggml_graph_add_node(hints[rank], consumers[rank]);
        buffers[rank] = ggml_backend_alloc_ctx_tensors(ctx, backends[rank]);
        GGML_ASSERT(buffers[rank]);
        for (auto * t : {x, residual, post, comb, consumers[rank]}) {
            std::vector<float> values(ggml_nelements(t), t == x ? float(rank + 1) : t == consumers[rank] ? -7.0f : 1.0f);
            ggml_backend_tensor_set(t, values.data(), 0, ggml_nbytes(t));
        }
    }
    void * comm = init(backends, 2);
    GGML_ASSERT(comm);
    for (int rank = 0; rank < 2; ++rank) { begin(backends[rank]); }
    for (int rank = 0; rank < 2; ++rank) {
        hint(backends[rank], hints[rank]);
        GGML_ASSERT(ggml_backend_graph_compute_async(backends[rank], prefixes[rank]) == GGML_STATUS_SUCCESS);
    }
    GGML_ASSERT(reduce(comm, sums));
    for (int rank = 0; rank < 2; ++rank) {
        execs[rank] = end(backends[rank]);
        GGML_ASSERT(execs[rank]);
    }
    for (int rank = 0; rank < 2; ++rank) { launch(backends[rank], execs[rank]); }
    bool ok = true;
    for (int rank = 0; rank < 2; ++rank) {
        ggml_backend_synchronize(backends[rank]);
        for (auto * t : {sums[rank], consumers[rank]}) {
            std::vector<float> values(ggml_nelements(t));
            ggml_backend_tensor_get(t, values.data(), 0, ggml_nbytes(t));
            for (float value : values) { ok = ok && value == (t == sums[rank] ? 3.0f : -7.0f); }
        }
        free_exec(backends[rank], execs[rank]);
    }
    free_comm(comm);
    for (int rank = 0; rank < 2; ++rank) {
        ggml_backend_buffer_free(buffers[rank]);
        ggml_free(contexts[rank]);
        ggml_backend_free(backends[rank]);
    }
    printf("%s plain collective leaves an advisory consumer untouched\n", ok ? "ok" : "FAIL");
    return ok;
}

// CPU bridges split each identity chain into multiple meta graphs that must survive together in the capture cache.
static bool check_split_capture(std::vector<ggml_backend_dev_t> gpus, int n_fragments, bool stress = false, bool transition = false) {
    constexpr int width = 128;
    constexpr int n_graphs = 5;
    std::vector<int> fragments(n_graphs, n_fragments);
    if (transition) {
        std::fill(fragments.begin() + 1, fragments.end(), 1);
    }
    int n_devices = (int) gpus.size();
    log_counts counts;
    counts.capture_details = stress;
    ggml_log_set(log_cb, &counts);
    ggml_backend_dev_t meta = ggml_backend_meta_device(gpus.data(), n_devices, split_by_name, (void *) &n_devices);
    ggml_backend_t backend = ggml_backend_dev_init(meta, nullptr);
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    ggml_backend_t backends[] = {backend, cpu};

    ggml_context * weights = ggml_init({2*ggml_tensor_overhead(), nullptr, true});
    ggml_tensor * up = ggml_new_tensor_2d(weights, GGML_TYPE_F32, width, width);
    ggml_tensor * down = ggml_new_tensor_2d(weights, GGML_TYPE_F32, width, width);
    ggml_set_name(up, "up.0");
    ggml_set_name(down, "down.0");
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors_from_buft(weights, ggml_backend_dev_buffer_type(meta));
    GGML_ASSERT(buffer);
    ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    std::vector<float> identity(width*width, 0.0f);
    for (int i = 0; i < width; ++i) { identity[i*width + i] = 1.0f; }
    for (auto * weight : {up, down}) {
        ggml_backend_tensor_set(weight, identity.data(), 0, ggml_nbytes(weight));
    }

    ggml_context * contexts[n_graphs];
    ggml_cgraph * graphs[n_graphs];
    ggml_tensor * inputs[n_graphs];
    ggml_tensor * outputs[n_graphs];
    ggml_backend_sched_t scheds[n_graphs];
    bool ok = true;
    for (int slot = 0; slot < n_graphs; ++slot) {
        // One large graph and small companions exercise capture residency without loading model weights.
        const int depth = stress && slot == 0 ? 4 : 1;
        const int extra_ops = stress && slot == 0 ? 512 : 0;
        const int graph_nodes = (4*depth + extra_ops)*fragments[slot];
        const int graph_size = std::max(GGML_DEFAULT_GRAPH_SIZE, graph_nodes + 16);
        contexts[slot] = ggml_init({(graph_nodes + 1)*ggml_tensor_overhead() + ggml_graph_overhead_custom(graph_size, false), nullptr, true});
        auto * ctx = contexts[slot];
        scheds[slot] = ggml_backend_sched_new(backends, nullptr, 2, graph_size, false, true);
        inputs[slot] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width, stress ? 108 : slot + 1);
        ggml_set_input(inputs[slot]);
        ggml_tensor * cur = inputs[slot];
        for (int fragment = 0; fragment < fragments[slot]; ++fragment) {
            for (int op = 0; op < extra_ops; ++op) {
                cur = ggml_scale(ctx, cur, -1.0f);
                ggml_backend_sched_set_tensor_backend(scheds[slot], cur, backend);
            }
            for (int layer = 0; layer < depth; ++layer) {
                ggml_tensor * residual = cur;
                cur = ggml_mul_mat(ctx, up, cur);
                ggml_backend_sched_set_tensor_backend(scheds[slot], cur, backend);
                cur = ggml_mul_mat(ctx, down, cur);
                ggml_backend_sched_set_tensor_backend(scheds[slot], cur, backend);
                cur = ggml_add(ctx, cur, residual);
                ggml_backend_sched_set_tensor_backend(scheds[slot], cur, backend);
                cur = ggml_scale(ctx, cur, 0.5f);
                const bool bridge = layer + 1 == depth && fragment + 1 < fragments[slot];
                ggml_backend_sched_set_tensor_backend(scheds[slot], cur, bridge ? cpu : backend);
            }
        }
        outputs[slot] = cur;
        ggml_set_output(cur);
        graphs[slot] = ggml_new_graph_custom(ctx, graph_size, false);
        ggml_build_forward_expand(graphs[slot], cur);
        GGML_ASSERT(ggml_backend_sched_alloc_graph(scheds[slot], graphs[slot]));
        const int splits = ggml_backend_sched_get_n_splits(scheds[slot]);
        if (splits != 2*fragments[slot] - 1) {
            fprintf(stderr, "FAIL %d fragments: scheduler made %d splits, expected %d\n",
                fragments[slot], splits, 2*fragments[slot] - 1);
            ok = false;
        }
    }

    const auto memory = [&](const char * phase) {
        if (!stress) { return; }
        size_t free_bytes = 0, total_bytes = 0;
        ggml_backend_dev_memory(gpus[0], &free_bytes, &total_bytes);
        printf("split_memory phase=%s free_bytes=%zu total_bytes=%zu\n", phase, free_bytes, total_bytes);
        fflush(stdout);
    };
    memory("allocated");
    const bool legacy = ggml_env_switch("GGML_META_CAPTURE_LEGACY");
    int n_eval = 0;
    int expected_fragments = 0;
    int last_expected_captures = 0;
    auto run = [&](int slot, int expected_captures, const char * phase) {
        std::vector<float> input(ggml_nelements(inputs[slot]));
        ++n_eval;
        for (size_t i = 0; i < input.size(); ++i) { input[i] = float(int((i + 7*n_eval)%31) - 15)/16.0f; }
        ggml_backend_tensor_set(inputs[slot], input.data(), 0, ggml_nbytes(inputs[slot]));
        char label[96];
        snprintf(label, sizeof(label), "meta-split phase=%s slot=%d", phase, slot);
#if TEST_HAS_NVTX
        if (stress) { nvtxRangePushA(label); }
#endif
        const int64_t start = ggml_time_us();
        const bool computed = ggml_backend_sched_graph_compute(scheds[slot], graphs[slot]) == GGML_STATUS_SUCCESS;
        std::vector<float> output(input.size());
        ggml_backend_tensor_get(outputs[slot], output.data(), 0, ggml_nbytes(outputs[slot]));
        const int64_t elapsed = ggml_time_us() - start;
#if TEST_HAS_NVTX
        if (stress) { nvtxRangePop(); }
#endif
        if (stress) {
            printf("split_eval phase=%s slot=%d host_us=%lld\n", phase, slot, (long long) elapsed);
            fflush(stdout);
        }
        expected_fragments += (expected_captures - last_expected_captures)*fragments[slot];
        last_expected_captures = expected_captures;
        const int expected = legacy ? 0 : expected_fragments*n_devices;
        const bool passed = computed && output == input && counts.executables == expected && counts.not_captured == 0;
        if (!passed) {
            fprintf(stderr, "FAIL %d fragments, slot %d: identity=%d, executables=%d (expected %d), refused=%d\n",
                n_fragments, slot, int(computed && output == input), counts.executables, expected, counts.not_captured);
        }
        ok = ok && passed;
    };
    for (int slot = 0; slot < 4; ++slot) { run(slot, 0, "warmup"); }
    memory("warmup");
    for (int slot = 0; slot < 4; ++slot) { run(slot, slot + 1, "capture"); }
    memory("captured");
    std::mt19937 rng(17);
    for (int i = 0; i < 24; ++i) { run(rng()%4, 4, "replay"); }
    for (int slot = 0; slot < 4; ++slot) { run(slot, 4, "replay"); } // slot 0 is now the oldest
    memory("steady");
    run(4, 4, "evict");
    // Four newer logical graphs must retire the old wide graph, not retain its
    // executables under a historical 4*max_fragments budget.
    if (transition && !legacy && ggml_env_switch("GGML_GRAPH_TRACE") &&
            counts.evicted_executables != fragments[0]*n_devices) {
        fprintf(stderr, "FAIL wide-to-narrow: retired %d executables, expected %d\n",
                counts.evicted_executables, fragments[0]*n_devices);
        ok = false;
    }
    run(4, 5, "capture");
    run(4, 5, "replay");
    run(0, 5, "evict");
    run(0, 6, "capture");
    run(0, 6, "replay");
    memory("recaptured");
    if (stress) {
        for (int cycle = 0; cycle < 15; ++cycle) {
            const int slot = (cycle + 1)%n_graphs;
            run(slot, 6 + cycle, "churn-warmup");
            run(slot, 7 + cycle, "churn-capture");
            run(slot, 7 + cycle, "churn-replay");
            char phase[32];
            snprintf(phase, sizeof(phase), "churn-%d", cycle);
            memory(phase);
        }
    }

    for (int slot = 0; slot < n_graphs; ++slot) {
        ggml_backend_sched_free(scheds[slot]);
        ggml_free(contexts[slot]);
    }
    ggml_backend_free(backend);
    ggml_backend_buffer_free(buffer);
    ggml_free(weights);
    ggml_backend_free(cpu);
    ggml_log_set(nullptr, nullptr);
    printf("%s %d-fragment capture reuse and eviction\n", ok ? "ok" : "FAIL", n_fragments);
    return ok;
}

int main(int argc, char ** argv) {
    const bool transition = argc == 2 && strcmp(argv[1], "--split-transition") == 0;
    const bool stress = argc == 2 && strcmp(argv[1], "--split-stress") == 0;
    const bool split_only = argc == 2 && strcmp(argv[1], "--split-capture") == 0;
    const bool logical = argc == 3 && strcmp(argv[1], "--logical") == 0;
    const bool physical = argc == 3 && strcmp(argv[1], "--devices") == 0;
    const int n_devices = logical || physical ? atoi(argv[2]) : stress ? 4 : 2;
    if ((argc != 1 && !logical && !physical && !split_only && !stress && !transition) || n_devices < 2 || n_devices > 4) {
        fprintf(stderr, "usage: %s [--devices 2|3|4 | --logical 2|3|4 | --split-capture | --split-stress | --split-transition]\n", argv[0]);
        return 1;
    }
    std::vector<ggml_backend_dev_t> gpus;
    for (size_t i = 0; i < ggml_backend_dev_count() && gpus.size() < (size_t) n_devices; i++) {
        if (ggml_backend_dev_type(ggml_backend_dev_get(i)) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            gpus.push_back(ggml_backend_dev_get(i));
        }
    }
    if (gpus.empty()) {
        printf("no GPU for the meta backend, skipped\n");
        return argc == 1 ? 0 : 77;
    }
    if (physical && gpus.size() < (size_t) n_devices) {
        printf("need %d physical GPUs, found %zu, skipped\n", n_devices, gpus.size());
        return 77;
    }
    const size_t n_physical = gpus.size();
    for (size_t rank = gpus.size(); rank < (size_t) n_devices; ++rank) {
        gpus.push_back(gpus[rank % n_physical]);
    }
    printf("meta backend over %d ranks on %zu physical GPUs\n", n_devices, n_physical);
    if (transition) { return check_split_capture(gpus, 11, false, true) ? 0 : 1; }
    if (stress) { return check_split_capture(gpus, 11, true) ? 0 : 1; }
    bool split_ok = true;
    for (int fragments : {5, 11}) { split_ok = check_split_capture(gpus, fragments) && split_ok; }
    if (split_only) { return split_ok ? 0 : 1; }

    const char * ar = getenv("GGML_CUDA_ALLREDUCE");
    if (ggml_env_switch("TEST_GLM_HC") && ggml_env_switch("GGML_CUDA_AR_HC_POST") &&
        ar && strcmp(ar, "internal") == 0 &&
        strncmp(ggml_backend_dev_name(gpus[0]), "CUDA", 4) == 0 && !check_collective_hint_scope(gpus)) { return 1; }

    log_counts counts;
    ggml_log_set(log_cb, &counts);

    ggml_backend_dev_t meta = ggml_backend_meta_device(gpus.data(), n_devices, split_by_name, (void *) &n_devices);

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
    constexpr int     n_graphs        = 6;
    constexpr int64_t n_tok[n_graphs] = {1, 2, 3, 4, 8, 40};
    std::vector<std::unique_ptr<ffn_graph>> graphs;
    std::vector<std::unique_ptr<ffn_graph>> graphs_cpu;
    std::vector<ggml_backend_sched_t>       scheds;
    std::vector<ggml_backend_sched_t>       scheds_cpu;
    ggml_backend_t backends[2] = {backend, cpu};
    for (int k = 0; k < n_graphs; k++) {
        graphs.push_back(std::make_unique<ffn_graph>(w_meta.up, w_meta.down, n_tok[k]));
        graphs_cpu.push_back(std::make_unique<ffn_graph>(w_cpu.up, w_cpu.down, n_tok[k]));
        scheds.push_back(ggml_backend_sched_new(backends, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE, false, true));
        GGML_ASSERT(ggml_backend_sched_alloc_graph(scheds.back(), graphs.back()->gf));
        scheds_cpu.push_back(ggml_backend_sched_new(&cpu, nullptr, 1, GGML_DEFAULT_GRAPH_SIZE, false, true));
        GGML_ASSERT(ggml_backend_sched_alloc_graph(scheds_cpu.back(), graphs_cpu.back()->gf));
    }

    bool ok = split_ok;
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
    const int  executables = legacy ? 0 : (n_graphs + 1)*n_devices;
    const bool ok_captures = counts.executables == executables && counts.not_captured == 0;
    printf("%s %d executables made (%d expected), %d graphs not captured\n", ok_captures ? "ok  " : "FAIL",
        counts.executables, executables, counts.not_captured);
    ok = ok && ok_captures;
    if (ggml_env_switch("TEST_GLM_HC") && ggml_env_switch("GGML_CUDA_AR_HC_POST") && !legacy) {
        const bool ok_hc = n_devices == 2 ? counts.hc_fused > 0 : counts.hc_fused == 0;
        printf("%s %d collective HC consumers captured\n", ok_hc ? "ok" : "FAIL", counts.hc_fused);
        ok = ok && ok_hc;
    }

    if (ar != nullptr && strcmp(ar, "internal") == 0 && strncmp(ggml_backend_dev_name(gpus[0]), "CUDA", 4) == 0) {
        const char * value = getenv("GGML_CUDA_AR_P2P_MAX_BYTES");
        char * end = nullptr;
        const uint64_t parsed = value == nullptr ? 0 : strtoull(value, &end, 10);
        const uint64_t requested = value != nullptr && end != value ? parsed : 128*1024;
        const size_t expected = requested > 1024*1024 ? 1024*1024 : requested;
        const bool unsupported = counts.p2p_unsupported && gpus[0] != gpus[1];
        const bool ok_p2p = counts.p2p_reported && !counts.p2p_failed &&
                           counts.p2p_bytes == (unsupported ? 0 : expected);
        printf("%s card-to-card limit %zu bytes: %s\n", ok_p2p ? "ok  " : "FAIL", counts.p2p_bytes,
               unsupported ? "peer unsupported; peer coverage skipped" :
               counts.p2p_bytes >= 4096 ? "device inboxes exercised" :
               counts.p2p_bytes > 0 ? "below this test's smallest reduction; peer coverage skipped" : "disabled");
        ok = ok && ok_p2p;
    }

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
