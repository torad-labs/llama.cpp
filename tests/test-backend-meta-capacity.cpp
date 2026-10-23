// Alternating prefill and draft graphs must retain valid graph storage when their node and reduction counts grow independently.
// Identity projections split across two backends must return the current input through every generated graph sequence.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstring>
#include <random>
#include <utility>
#include <vector>

static ggml_backend_meta_split_state split(const ggml_tensor * tensor, void *) {
    if (strcmp(tensor->name, "up") == 0 || strcmp(tensor->name, "down") == 0) {
        ggml_backend_meta_split_state result = {tensor->name[0] == 'u' ? GGML_BACKEND_SPLIT_AXIS_1 : GGML_BACKEND_SPLIT_AXIS_0, {0}, {1}, 1};
        result.ne[0] = 4;
        result.ne[1] = 4;
        return result;
    }
    return {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
}

static bool exercise(ggml_backend_dev_t meta, ggml_backend_t cpu, bool nodes_first) {
    ggml_backend_t backend = ggml_backend_dev_init(meta, nullptr);
    const ggml_init_params wp = {2*ggml_tensor_overhead(), nullptr, true};
    ggml_context * wc = ggml_init(wp);
    ggml_tensor * up = ggml_new_tensor_2d(wc, GGML_TYPE_F32, 8, 8);
    ggml_tensor * down = ggml_new_tensor_2d(wc, GGML_TYPE_F32, 8, 8);
    ggml_set_name(up, "up");
    ggml_set_name(down, "down");
    ggml_backend_buffer_t wb = ggml_backend_alloc_ctx_tensors_from_buft(wc, ggml_backend_dev_buffer_type(meta));
    GGML_ASSERT(wb);
    ggml_backend_buffer_set_usage(wb, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    float identity[64] = {};
    for (int i = 0; i < 8; ++i) identity[9*i] = 1.0f;
    ggml_backend_tensor_set(up, identity, 0, sizeof(identity));
    ggml_backend_tensor_set(down, identity, 0, sizeof(identity));

    const std::pair<int, int> many_nodes = {1, 64};
    const std::pair<int, int> many_reductions = {8, 0};
    std::vector<std::pair<int, int>> shapes = nodes_first ?
        std::vector<std::pair<int, int>>{many_nodes, many_reductions, many_nodes} :
        std::vector<std::pair<int, int>>{many_reductions, many_nodes, many_reductions};
    std::mt19937 rng(530);
    for (int i = 0; i < 32; ++i) shapes.push_back({1 + int(rng()%12), int(rng()%80)});
    bool ok = true;
    int step = 0;
    for (const auto & shape : shapes) {
        const size_t max_nodes = 256;
        const ggml_init_params gp = {max_nodes*ggml_tensor_overhead() + ggml_graph_overhead_custom(max_nodes, false), nullptr, true};
        ggml_context * ctx = ggml_init(gp);
        ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 3);
        ggml_set_input(x);
        ggml_tensor * out = x;
        for (int i = 0; i < shape.first; ++i) out = ggml_mul_mat(ctx, down, ggml_mul_mat(ctx, up, out));
        for (int i = 0; i <= shape.second; ++i) out = ggml_scale(ctx, out, 1.0f);
        ggml_set_output(out);
        ggml_cgraph * graph = ggml_new_graph_custom(ctx, max_nodes, false);
        ggml_build_forward_expand(graph, out);
        ggml_backend_t backends[] = {backend, cpu};
        ggml_backend_sched_t sched = ggml_backend_sched_new(backends, nullptr, 2, max_nodes, false, true);
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            ggml_backend_sched_set_tensor_backend(sched, ggml_graph_node(graph, i), backend);
        }
        GGML_ASSERT(ggml_backend_sched_alloc_graph(sched, graph));
        float input[24], actual[24];
        for (int i = 0; i < 24; ++i) input[i] = float(32*step + i) / 4.0f;
        ggml_backend_tensor_set(x, input, 0, sizeof(input));
        printf("capacity: order=%d step=%d reductions=%d tail=%d nodes=%d\n", nodes_first, step, shape.first, shape.second, ggml_graph_n_nodes(graph));
        fflush(stdout);
        GGML_ASSERT(ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS);
        ggml_backend_tensor_get(out, actual, 0, sizeof(actual));
        for (int i = 0; i < 24; ++i) {
            if (actual[i] != input[i]) {
                fprintf(stderr, "FAIL step %d row %d: %g != %g\n", step, i, actual[i], input[i]);
                ok = false;
            }
        }
        ggml_backend_sched_free(sched);
        ggml_free(ctx);
        ++step;
    }
    ggml_backend_free(backend);
    ggml_backend_buffer_free(wb);
    ggml_free(wc);
    return ok;
}

int main(int argc, char ** argv) {
    const bool gpu = argc == 2 && strcmp(argv[1], "--gpu") == 0;
    const auto type = gpu ? GGML_BACKEND_DEVICE_TYPE_GPU : GGML_BACKEND_DEVICE_TYPE_CPU;
    std::vector<ggml_backend_dev_t> devices;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        if (ggml_backend_dev_type(ggml_backend_dev_get(i)) == type) devices.push_back(ggml_backend_dev_get(i));
    }
    GGML_ASSERT(!devices.empty());
    ggml_backend_dev_t parts[] = {devices[0], devices.size() > 1 ? devices[1] : devices[0]};
    ggml_backend_dev_t meta = ggml_backend_meta_device(parts, 2, split, nullptr);
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    const bool nodes = exercise(meta, cpu, true);
    const bool reductions = exercise(meta, cpu, false);
    ggml_backend_free(cpu);
    printf("%s: meta graph capacity\n", nodes && reductions ? "PASS" : "FAIL");
    return nodes && reductions ? 0 : 1;
}
