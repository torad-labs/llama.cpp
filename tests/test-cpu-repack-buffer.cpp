// The CPU extra buffer type CPU_REPACK holds weights in the interleaved layouts its kernels read, chosen per tensor
// when the tensor is allocated. A tensor it has no layout for must be declined at that allocation: Ternary Bonsai 2 27B
// at -ngl 0 put its f32 Hadamard table (prism.hadamard.1024, [1024, 1024]) in CPU_REPACK beside the PQ2_0 weight it
// rotates, and set_tensor then called through the missing layout (SIGSEGV at a null address). A tensor it does repack
// must still allocate, upload and multiply as it does in the CPU buffer type.

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

static ggml_backend_buffer_type_t cpu_repack_buft(ggml_backend_dev_t cpu) {
    auto * reg = ggml_backend_dev_backend_reg(cpu);
    auto get_extra = (ggml_backend_dev_get_extra_bufts_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_dev_get_extra_bufts");
    for (auto * extra = get_extra ? get_extra(cpu) : nullptr; extra && *extra; ++extra) {
        if (strcmp(ggml_backend_buft_name(*extra), "CPU_REPACK") == 0) {
            return *extra;
        }
    }
    return nullptr;
}

static ggml_context * new_ctx(size_t n_tensors) {
    ggml_init_params params = { ggml_tensor_overhead() * n_tensors + ggml_graph_overhead(), nullptr, true };
    return ggml_init(params);
}

// the allocation of a tensor the buffer type has no layout for fails, instead of leaving it for set_tensor
static bool test_declined(ggml_backend_buffer_type_t repack, const char * name, ggml_type type, int64_t ne0, int64_t ne1) {
    ggml_context * ctx = new_ctx(1);
    ggml_tensor * t = ggml_new_tensor_2d(ctx, type, ne0, ne1);
    ggml_set_name(t, name);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, repack);
    const bool ok = buf == nullptr;
    printf("  %-26s %-5s [%lld, %lld]: %s\n", name, ggml_type_name(type), (long long) ne0, (long long) ne1,
            ok ? "declined, ok" : "FAILED: accepted without a repacked layout (its set_tensor would dereference none)");
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return ok;
}

// y = w x with w in buft, on the CPU backend; false when buft declines w
static bool mul_mat(ggml_backend_t backend, ggml_backend_buffer_type_t buft, ggml_type type, int64_t k, int64_t n, int64_t m,
        const std::vector<uint8_t> & wq, const std::vector<float> & x, std::vector<float> & y) {
    ggml_context * ctx_w = new_ctx(1);
    ggml_tensor * w = ggml_new_tensor_2d(ctx_w, type, k, n);
    ggml_backend_buffer_t buf_w = ggml_backend_alloc_ctx_tensors_from_buft(ctx_w, buft);
    if (buf_w == nullptr) {
        ggml_free(ctx_w);
        return false;
    }
    ggml_backend_tensor_set(w, wq.data(), 0, wq.size());

    ggml_context * ctx = new_ctx(2);
    ggml_tensor * xt = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, m);
    ggml_tensor * yt = ggml_mul_mat(ctx, w, xt);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    ggml_backend_tensor_set(xt, x.data(), 0, x.size() * sizeof(float));

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, yt);
    const bool computed = ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS;
    y.resize(n * m);
    ggml_backend_tensor_get(yt, y.data(), 0, y.size() * sizeof(float));

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_buffer_free(buf_w);
    ggml_free(ctx_w);
    return computed;
}

// a weight the CPU repacks: uploaded through CPU_REPACK, it multiplies as it does from the CPU buffer type. Returns
// -1 when this CPU has no layout for the type, 0 on a mismatch, 1 on a match
static int test_repacked(ggml_backend_t backend, ggml_backend_buffer_type_t repack, ggml_type type, int64_t k, int64_t n, int64_t m) {
    std::vector<float> wf(k * n), x(k * m);
    for (size_t i = 0; i < wf.size(); ++i) { wf[i] = 0.1f + 2.0f * cosf(0.7f * i); }
    for (size_t i = 0; i < x.size();  ++i) { x[i]  = sinf(0.3f * i + 1.0f); }
    std::vector<uint8_t> wq(ggml_row_size(type, k) * n);
    ggml_quantize_chunk(type, wf.data(), wq.data(), 0, n, k, nullptr);

    std::vector<float> y_ref, y_rep;
    if (!mul_mat(backend, ggml_backend_cpu_buffer_type(), type, k, n, m, wq, x, y_ref)) {
        printf("  %-5s [%lld, %lld] x %lld: FAILED in the CPU buffer type\n", ggml_type_name(type), (long long) k, (long long) n, (long long) m);
        return 0;
    }
    if (!mul_mat(backend, repack, type, k, n, m, wq, x, y_rep)) {
        printf("  %-5s [%lld, %lld] x %lld: no repacked layout on this CPU, skipped\n", ggml_type_name(type), (long long) k, (long long) n, (long long) m);
        return -1;
    }
    double err = 0.0, ref = 0.0;
    for (size_t i = 0; i < y_ref.size(); ++i) {
        err += (y_rep[i] - y_ref[i]) * (double) (y_rep[i] - y_ref[i]);
        ref += y_ref[i] * (double) y_ref[i];
    }
    const double nmse = err / ref;
    const bool ok = nmse < 1e-6;
    printf("  %-5s [%lld, %lld] x %lld: nmse %.3g vs the CPU buffer type: %s\n", ggml_type_name(type), (long long) k, (long long) n,
            (long long) m, nmse, ok ? "ok" : "FAILED");
    return ok ? 1 : 0;
}

int main(void) {
    ggml_backend_load_all();
    ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    ggml_backend_buffer_type_t repack = cpu ? cpu_repack_buft(cpu) : nullptr;
    if (repack == nullptr) {
        printf("no CPU_REPACK buffer type in this build, skipped\n");
        return 0;
    }
    ggml_backend_t backend = ggml_backend_dev_init(cpu, nullptr);

    bool ok = true;
    printf("tensors CPU_REPACK has no layout for:\n");
    ok &= test_declined(repack, "prism.hadamard.1024",       GGML_TYPE_F32,  1024, 1024);
    ok &= test_declined(repack, "prism.hadamard.signs.5120", GGML_TYPE_F32,  5120,    1);
    ok &= test_declined(repack, "f16 weight",                GGML_TYPE_F16,   512,   64);
    ok &= test_declined(repack, "q4_0, 6 rows",              GGML_TYPE_Q4_0,  512,    6);

    printf("weights CPU_REPACK repacks (one token and a batch of 7):\n");
    int n_repacked = 0;
    for (ggml_type type : { GGML_TYPE_Q4_0, GGML_TYPE_Q8_0, GGML_TYPE_Q4_K, GGML_TYPE_IQ4_NL, GGML_TYPE_PQ2_0 }) {
        for (int64_t m : { 1, 7 }) {
            const int r = test_repacked(backend, repack, type, 512, 64, m);
            ok &= r != 0;
            n_repacked += r == 1;
        }
    }
    if (n_repacked == 0) {
        printf("  (this CPU repacks none of them: the declines above are not checked against an accepted weight)\n");
    }

    ggml_backend_free(backend);
    printf("%s\n", ok ? "OK" : "FAILED");
    return ok ? 0 : 1;
}
