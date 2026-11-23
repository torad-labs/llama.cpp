// mmq-microperf: times and checks GGML_OP_MUL_MAT_ID for quantized MoE experts at a served shape.
//
//   mmq-microperf <type> <m> <k> <tokens> <bcast 0|1> perf [iters] [seed]
//   mmq-microperf <type> <m> <k> <tokens> <bcast 0|1> ref  [seed]
//
// perf: weights are random bytes (timing does not read the values), the op is timed with the CUDA backend.
// ref:  weights are quantized from uniform floats (whole rows at a time, host memory stays flat), the CUDA output
//       is compared against the CPU backend's, and the normalized MSE is printed.
//
// The expert count (384) and the used experts (8) are fixed: Inkling 2 Small's routed experts.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static const int64_t N_MATS = 384;
static const int64_t N_USED = 8;

static ggml_type type_from_name(const char * s) {
    if (strcmp(s, "q3_K") == 0) return GGML_TYPE_Q3_K;
    if (strcmp(s, "q4_K") == 0) return GGML_TYPE_Q4_K;
    if (strcmp(s, "q5_K") == 0) return GGML_TYPE_Q5_K;
    if (strcmp(s, "q6_K") == 0) return GGML_TYPE_Q6_K;
    fprintf(stderr, "unknown type %s\n", s);
    exit(1);
}

static ggml_backend_dev_t find_device(const char * name) {
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (strcmp(ggml_backend_dev_name(dev), name) == 0) {
            return dev;
        }
    }
    fprintf(stderr, "no device %s\n", name);
    exit(1);
}

// Whole rows at a time; the float staging buffer is at most one chunk of about 16M values.
static void fill_quantized(ggml_backend_t backend, ggml_tensor * t, uint64_t seed) {
    const int64_t k     = t->ne[0];
    const size_t  nels  = ggml_nelements(t);
    const size_t  bs    = ggml_blck_size(t->type);
    const size_t  unit  = k % bs == 0 ? k : k * bs;
    const size_t  chunk = std::max<size_t>(1, (size_t(1) << 24) / unit) * unit;
    std::vector<float>   src(std::min<size_t>(nels, chunk));
    std::vector<uint8_t> dst;
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t start = 0; start < nels; start += chunk) {
        const size_t n = std::min(chunk, nels - start);
        for (size_t i = 0; i < n; i++) {
            src[i] = dist(rng);
        }
        dst.resize(ggml_row_size(t->type, n));
        ggml_quantize_chunk(t->type, src.data(), dst.data(), 0, n / bs, bs, nullptr);
        ggml_backend_tensor_set(t, dst.data(), (start / bs) * ggml_type_size(t->type), dst.size());
    }
}

// Random bytes, uploaded in chunks; timing does not read the values.
static void fill_random_bytes(ggml_backend_t backend, ggml_tensor * t, uint64_t seed) {
    const size_t nbytes = ggml_nbytes(t);
    const size_t chunk = size_t(64) << 20;
    std::vector<uint8_t> buf(std::min(nbytes, chunk));
    std::mt19937_64 rng(seed);
    for (size_t off = 0; off < nbytes; off += chunk) {
        const size_t n = std::min(chunk, nbytes - off);
        size_t i = 0;
        for (; i + 8 <= n; i += 8) {
            const uint64_t v = rng();
            memcpy(&buf[i], &v, 8);
        }
        for (; i < n; i++) {
            buf[i] = (uint8_t) (rng() & 0xff);
        }
        ggml_backend_tensor_set(t, buf.data(), off, n);
    }
}

struct cmp_state {
    double  nmse = -1.0;
    int64_t n    = 0;
};

static bool cmp_callback(int index, ggml_tensor * t1, ggml_tensor * t2, void * user_data) {
    GGML_UNUSED(index);
    cmp_state * st = (cmp_state *) user_data;
    std::vector<float> a(ggml_nelements(t1));
    std::vector<float> b(ggml_nelements(t2));
    ggml_backend_tensor_get(t1, a.data(), 0, ggml_nbytes(t1));
    ggml_backend_tensor_get(t2, b.data(), 0, ggml_nbytes(t2));
    double err = 0.0, ref = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        err += double(a[i] - b[i]) * double(a[i] - b[i]);
        ref += double(a[i]) * double(a[i]);
    }
    st->nmse = err / ref;
    st->n    = (int64_t) a.size();
    printf("REF nmse=%.3e n=%lld\n", st->nmse, (long long) st->n);
    return true;
}

static int64_t expected_J(int64_t tokens) {
    // mmq.cu's spread rule for MUL_MAT_ID (GGML_CUDA_MMQ_MOE_TILES_LEGACY unset), without the env override.
    const double  mean   = double(tokens * N_USED) / double(N_MATS);
    const int64_t spread = int64_t(std::ceil(mean + 3.0 * std::sqrt(mean)));
    int64_t ncols = std::min({tokens, int64_t(128), spread});
    const char * env = getenv("GGML_CUDA_MMQ_MOE_NCOLS");
    if (env && atoll(env) > 0) {
        ncols = std::min({tokens, int64_t(128), (int64_t) atoll(env)});
    }
    int64_t J_best = 0;
    int64_t best   = INT64_MAX;
    for (int64_t J = 8; J <= 128; J += 8) {
        const int64_t ntiles = (ncols + J - 1) / J;
        if (ntiles < best) {
            best   = ntiles;
            J_best = J;
        }
    }
    return J_best;
}

int main(int argc, char ** argv) {
    if (argc < 7) {
        fprintf(stderr, "usage: %s <type> <m> <k> <tokens> <bcast> <perf|ref> [iters] [seed]\n", argv[0]);
        return 1;
    }
    const ggml_type type   = type_from_name(argv[1]);
    const int64_t   m      = atoll(argv[2]);
    const int64_t   k      = atoll(argv[3]);
    const int64_t   tokens = atoll(argv[4]);
    const bool      bcast  = atoi(argv[5]) != 0;
    const std::string mode = argv[6];
    const int       iters  = argc > 7 && strcmp(argv[7], "ref") != 0 ? atoi(argv[7]) : 50;
    const uint64_t  seed   = argc > 8 ? strtoull(argv[8], nullptr, 10) : 20261122;

    ggml_backend_load_all();

    ggml_backend_dev_t cuda_dev = find_device("CUDA0");
    ggml_backend_t     cuda     = ggml_backend_dev_init(cuda_dev, nullptr);
    if (!cuda) {
        fprintf(stderr, "failed to init CUDA0\n");
        return 1;
    }

    ggml_init_params params = {
        /* .mem_size   = */ ggml_tensor_overhead() * 32 + size_t(16) << 20,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(params);

    ggml_tensor * as    = ggml_new_tensor_3d(ctx, type, k, m, N_MATS);
    ggml_tensor * ids   = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, N_USED, tokens);
    ggml_tensor * b     = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k, bcast ? 1 : N_USED, tokens);
    ggml_tensor * out   = ggml_mul_mat_id(ctx, as, b, ids);
    ggml_set_name(as, "as");
    ggml_set_name(out, "out");

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, cuda);
    if (!buf) {
        fprintf(stderr, "failed to allocate tensors\n");
        return 1;
    }

    if (mode == "ref") {
        fill_quantized(cuda, as, seed);
    } else {
        fill_random_bytes(cuda, as, seed);
    }

    {
        std::mt19937 rng((uint32_t) seed);
        std::vector<int32_t> roster(N_MATS);
        std::vector<int32_t> idv(N_USED * tokens);
        for (int64_t t = 0; t < tokens; t++) {
            for (int64_t e = 0; e < N_MATS; e++) {
                roster[e] = (int32_t) e;
            }
            for (int64_t u = 0; u < N_USED; u++) {
                const int64_t pick = u + (int64_t) (rng() % (uint32_t) (N_MATS - u));
                std::swap(roster[u], roster[pick]);
                idv[t * N_USED + u] = roster[u];
            }
        }
        ggml_backend_tensor_set(ids, idv.data(), 0, idv.size() * sizeof(int32_t));
    }
    {
        std::mt19937_64 rng(seed + 1);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        std::vector<float> bv(ggml_nelements(b));
        for (auto & x : bv) {
            x = dist(rng);
        }
        ggml_backend_tensor_set(b, bv.data(), 0, bv.size() * sizeof(float));
    }

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    const int64_t J = expected_J(tokens);
    printf("CASE type=%s m=%lld k=%lld tokens=%lld bcast=%d J=%lld\n",
           ggml_type_name(type), (long long) m, (long long) k, (long long) tokens, (int) bcast, (long long) J);

    if (mode == "ref") {
        ggml_backend_dev_t cpu_dev = find_device("CPU");
        ggml_backend_t     cpu     = ggml_backend_dev_init(cpu_dev, nullptr);
        if (!cpu) {
            fprintf(stderr, "failed to init CPU\n");
            return 1;
        }
        {
            // The reference is the slow part; give it the cores.
            ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(cpu_dev);
            auto set_threads = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
            if (set_threads) {
                set_threads(cpu, 8);
            }
        }
        cmp_state st;
        const ggml_tensor * nodes[1] = { out };
        const bool ok = ggml_backend_compare_graph_backend(cuda, cpu, gf, cmp_callback, &st, nodes, 1);
        printf("RESULT type=%s m=%lld k=%lld tokens=%lld bcast=%d J=%lld compare=%s nmse=%.3e\n",
               ggml_type_name(type), (long long) m, (long long) k, (long long) tokens, (int) bcast, (long long) J,
               ok ? "ok" : "FAILED", st.nmse);
        ggml_backend_free(cpu);
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        ggml_backend_free(cuda);
        return ok ? 0 : 1;
    }

    for (int i = 0; i < 5; i++) {
        if (ggml_backend_graph_compute(cuda, gf) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "compute failed\n");
            return 1;
        }
        ggml_backend_synchronize(cuda);
    }

    int64_t best_us = INT64_MAX;
    int64_t sum_us  = 0;
    for (int i = 0; i < iters; i++) {
        const int64_t t0 = ggml_time_us();
        if (ggml_backend_graph_compute(cuda, gf) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "compute failed\n");
            return 1;
        }
        ggml_backend_synchronize(cuda);
        const int64_t dt = ggml_time_us() - t0;
        sum_us += dt;
        best_us = std::min(best_us, dt);
    }
    printf("RESULT type=%s m=%lld k=%lld tokens=%lld bcast=%d J=%lld us_mean=%.2f us_min=%lld iters=%d\n",
           ggml_type_name(type), (long long) m, (long long) k, (long long) tokens, (int) bcast, (long long) J,
           double(sum_us) / iters, (long long) best_us, iters);

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(cuda);
    return 0;
}
