// A PQ2_0 matmul at 1-8 columns runs on GeForce Blackwell cards on the tensor-core kernel (mmvq-pq2-mma.cu), which
// streams the weights into shared memory with TMA. Its load was addressed to .shared::cluster, which compiles on sm_120
// to a branch with a driver syscall, and the driver then gave every resident thread the syscall's stack, 14,512 bytes:
// 1,430 MiB of device memory on a 70-SM RTX 5070 Ti from the kernel's first launch on, held outside every buffer the
// model and its contexts allocate. The served head's tiers, sized by those buffers, no longer fit their cards. The kernel
// needs nothing but its operands.
//
// On each GPU: the device's free memory once the weights, the activations and the outputs are allocated and set, then
// one matrix and a group of two (one launch over both) computed at 1 and at 8 columns, three times each, then the free
// memory again. What the computes took is the backend's pool for the quantized activations and the captured CUDA graphs,
// a few MiB. The bound is 128 MiB; the syscall's stack is 425 MiB on a GB20x of 20 SMs, the smallest. Another process
// that allocates on the same card in between reads as ours, so a failure prints the amount to check against that.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cstdio>
#include <random>
#include <vector>

static constexpr int64_t k         = 4096;       // four 1,024-weight boxes per row, as the kernel takes them
static constexpr int64_t n         = 256;        // 16 tiles of 16 rows
static constexpr size_t  max_taken = 128u << 20;

static void set_pq2_0(ggml_tensor * w, std::mt19937 & rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> f(ggml_nelements(w));
    for (float & v : f) {
        v = dist(rng);
    }
    std::vector<uint8_t> q(ggml_nbytes(w));
    ggml_quantize_chunk(GGML_TYPE_PQ2_0, f.data(), q.data(), 0, w->ne[1], w->ne[0], nullptr);
    ggml_backend_tensor_set(w, q.data(), 0, q.size());
}

static size_t free_memory(ggml_backend_dev_t dev) {
    size_t free = 0;
    size_t total = 0;
    ggml_backend_dev_memory(dev, &free, &total);
    return free;
}

// returns false when the matmuls took more than max_taken of the device's memory
static bool test_device(ggml_backend_dev_t dev, ggml_backend_t backend) {
    std::mt19937 rng(42);
    ggml_init_params params = { ggml_tensor_overhead() * 16 + 4 * ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(params);

    ggml_tensor * w1 = ggml_new_tensor_2d(ctx, GGML_TYPE_PQ2_0, k, n);
    ggml_tensor * w2 = ggml_new_tensor_2d(ctx, GGML_TYPE_PQ2_0, k, n);
    std::vector<ggml_tensor *> xs;
    std::vector<ggml_cgraph *> graphs;
    for (int64_t m : { 1, 8 }) {
        ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, m);
        xs.push_back(x);

        ggml_cgraph * single = ggml_new_graph(ctx);
        ggml_build_forward_expand(single, ggml_mul_mat(ctx, w1, x));
        graphs.push_back(single);

        // two matrices reading one activation: the backend runs them as one group launch
        ggml_cgraph * group = ggml_new_graph(ctx);
        ggml_build_forward_expand(group, ggml_mul_mat(ctx, w1, x));
        ggml_build_forward_expand(group, ggml_mul_mat(ctx, w2, x));
        graphs.push_back(group);
    }
    for (ggml_cgraph * gf : graphs) {
        if (!ggml_backend_supports_op(backend, ggml_graph_node(gf, -1))) {
            printf("  MUL_MAT with PQ2_0 weights is not supported here, skipped\n");
            ggml_free(ctx);
            return true;
        }
    }

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    set_pq2_0(w1, rng);
    set_pq2_0(w2, rng);
    for (ggml_tensor * x : xs) {
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        std::vector<float> v(ggml_nelements(x));
        for (float & f : v) {
            f = dist(rng);
        }
        ggml_backend_tensor_set(x, v.data(), 0, ggml_nbytes(x));
    }
    ggml_backend_synchronize(backend);

    const size_t before = free_memory(dev);
    for (int rep = 0; rep < 3; ++rep) {
        for (ggml_cgraph * gf : graphs) {
            GGML_ASSERT(ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS);
        }
    }
    ggml_backend_synchronize(backend);
    const size_t after = free_memory(dev);

    const size_t taken = before > after ? before - after : 0;
    const bool   ok    = taken <= max_taken;
    printf("  PQ2_0 matmuls at 1 and 8 columns, single and grouped: %zu MiB of device memory taken (at most %zu): %s\n",
        taken >> 20, max_taken >> 20, ok ? "ok" : "FAILED");

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return ok;
}

int main(void) {
    ggml_backend_load_all();

    bool ok = true;
    int n_tested = 0;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) {
            continue;
        }
        ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
        printf("%s (%s):\n", ggml_backend_dev_name(dev), ggml_backend_dev_description(dev));
        ok &= test_device(dev, backend);
        ggml_backend_free(backend);
        n_tested++;
    }

    if (n_tested == 0) {
        printf("no GPU device, skipped\n");
        return 0;
    }
    printf("%s\n", ok ? "OK" : "FAILED");
    return ok ? 0 : 1;
}
