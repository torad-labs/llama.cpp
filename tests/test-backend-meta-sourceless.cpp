// The meta backend (-sm tensor) derives each node's split from its sources' splits. A node with no sources, as
// ggml_arange, had none to derive it from: its split was unknown, and allocating the graph aborted
// (GGML_ASSERT(ret.axis != GGML_BACKEND_SPLIT_AXIS_UNKNOWN)). GLM-5.3's sparse attention makes one for the dump columns of
// its filler slots (build_attn_sparse), so under -sm tensor every decode aborted. Every device computes the same values
// for such a node: it is mirrored.
//
// y = x + arange(n) is allocated and computed through the meta backend, over the first GPU twice (it splits over two
// devices as two GPUs do), and must hold x[i] + i; with no GPU the test is skipped.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cstdio>
#include <vector>

static constexpr int64_t n = 64;

static ggml_backend_meta_split_state mirrored(const ggml_tensor * /*tensor*/, void * /*userdata*/) {
    return {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
}

int main() {
    ggml_backend_dev_t gpu = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count() && gpu == nullptr; i++) {
        if (ggml_backend_dev_type(ggml_backend_dev_get(i)) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            gpu = ggml_backend_dev_get(i);
        }
    }
    if (gpu == nullptr) {
        printf("no GPU for the meta backend, skipped\n");
        return 0;
    }
    ggml_backend_dev_t devs[2] = {gpu, gpu};
    ggml_backend_dev_t meta = ggml_backend_meta_device(devs, 2, mirrored, nullptr);

    ggml_backend_t cpu     = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    ggml_backend_t backend = ggml_backend_dev_init(meta, nullptr);

    const ggml_init_params params = {
        /*.mem_size   =*/ 4*ggml_tensor_overhead() + ggml_graph_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    ggml_tensor * x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    ggml_set_input(x);
    ggml_tensor * y = ggml_add(ctx, x, ggml_arange(ctx, 0.0f, (float) n, 1.0f));
    ggml_set_output(y);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);

    ggml_backend_t backends[2] = {backend, cpu};
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE, false, true);
    GGML_ASSERT(ggml_backend_sched_alloc_graph(sched, gf));

    std::vector<float> x_data(n);
    for (int64_t i = 0; i < n; i++) {
        x_data[i] = 0.5f + 100.0f*i;
    }
    ggml_backend_tensor_set(x, x_data.data(), 0, ggml_nbytes(x));
    GGML_ASSERT(ggml_backend_sched_graph_compute(sched, gf) == GGML_STATUS_SUCCESS);
    std::vector<float> y_data(n);
    ggml_backend_tensor_get(y, y_data.data(), 0, ggml_nbytes(y));

    bool ok = true;
    for (int64_t i = 0; i < n && ok; i++) {
        if (y_data[i] != x_data[i] + (float) i) {
            printf("FAIL y[%lld] = %g, expected %g\n", (long long) i, y_data[i], x_data[i] + (float) i);
            ok = false;
        }
    }

    ggml_backend_sched_free(sched);
    ggml_free(ctx);
    ggml_backend_free(backend);
    ggml_backend_free(cpu);

    printf("%s\n", ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}
