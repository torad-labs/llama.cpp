// Inkling 2's routed experts at one card's 23 decode rows: 384 experts, top 8, hidden 4,608, intermediate 2,304. The
// CUDA backend's few-row MMQ (ggml_cuda_mul_mat_q_few) keeps the expert weights in their stored Q3_K/Q4_K blocks and
// decodes them in the tile that consumes them. This test judges the activation side of that seam: a row of b that a
// token's 8 slots read (b broadcast over the experts) is quantized once and written to each of them, bit for bit as
// the path that quantizes every (token, slot) entry writes it (GGML_CUDA_MMQ_FEW_ID_QUANT_LEGACY=1).
//
//   gate/up: {4,608 x 4,608} of Q3_K, 384 experts, 8 used, 23 tokens, b a token's vector shared by its experts. At 23
//            tokens the few-row MMQ takes this product (GGML_CUDA_MMVQ_KQUANT_MMID_MAX=8) and its quantize writes the
//            new copy.
//   down:    {2,304 x 4,608} of Q4_K, 384 experts, 8 used, 23 tokens, a vector a slot. Its rows do not fill a whole
//            number of ring stages at intermediate 2,304, so the product takes MMQ; it is compared all the same.
//
// Each case runs against the CPU backend at test-backend-ops' bound for MUL_MAT_ID (NMSE 5e-4), and CTest
// compares the output of two processes bit for bit.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <future>
#include <random>
#include <thread>
#include <vector>

static constexpr double max_nmse = 5e-4; // test-backend-ops' bound for MUL_MAT_ID

// value null unsets the variable
static void set_env(const char * name, const char * value) {
#ifdef _WIN32
    _putenv_s(name, value ? value : "");
#else
    if (value) {
        setenv(name, value, 1);
    } else {
        unsetenv(name);
    }
#endif
}

struct seam_case {
    ggml_type type_a;
    int64_t   k;         // hidden or intermediate columns
    int64_t   m;         // rows of an expert
    int64_t   n_mats;    // experts
    int64_t   n_used;    // experts a token routes to
    bool      b_broadcast; // one row of b shared by the slots (gate/up), or a row a slot (down)
    int64_t   n_tokens;
};

struct seam_graph {
    ggml_context *        ctx_in  = nullptr;
    ggml_backend_buffer_t buf     = nullptr;
    ggml_context *        ctx_out = nullptr;
    ggml_cgraph *         gf      = nullptr;
    ggml_gallocr_t        galloc  = nullptr;
    ggml_tensor *         as      = nullptr;
    ggml_tensor *         ids     = nullptr;
    ggml_tensor *         b       = nullptr;
    ggml_tensor *         out     = nullptr;

    ~seam_graph() {
        ggml_gallocr_free(galloc);
        ggml_backend_buffer_free(buf);
        ggml_free(ctx_out);
        ggml_free(ctx_in);
    }
};

static bool build(seam_graph & g, ggml_backend_t backend, const seam_case & c) {
    const ggml_init_params params_in = { ggml_tensor_overhead()*8, nullptr, true };
    g.ctx_in = ggml_init(params_in);
    g.as  = ggml_new_tensor_3d(g.ctx_in, c.type_a, c.k, c.m, c.n_mats);
    g.ids = ggml_new_tensor_2d(g.ctx_in, GGML_TYPE_I32, c.n_used, c.n_tokens);
    g.b   = ggml_new_tensor_3d(g.ctx_in, GGML_TYPE_F32, c.k, c.b_broadcast ? 1 : c.n_used, c.n_tokens);
    g.buf = ggml_backend_alloc_ctx_tensors(g.ctx_in, backend);
    if (g.buf == nullptr) {
        return false;
    }

    const ggml_init_params params_out = { ggml_tensor_overhead()*8 + ggml_graph_overhead(), nullptr, true };
    g.ctx_out = ggml_init(params_out);
    g.out = ggml_mul_mat_id(g.ctx_out, g.as, g.b, g.ids);
    g.gf  = ggml_new_graph(g.ctx_out);
    ggml_build_forward_expand(g.gf, g.out);
    g.galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    return ggml_gallocr_alloc_graph(g.galloc, g.gf);
}

// The same values into both graphs, the weights a whole row of blocks at a time: a tensor of billions of values never
// stands whole as floats.
static void fill(seam_graph & g0, seam_graph & g1, const seam_case & c, uint32_t seed) {
    const int64_t nels = ggml_nelements(g0.as);
    const int64_t blck = ggml_blck_size(c.type_a);
    const int64_t unit = c.k % blck == 0 ? c.k : c.k*blck;
    const int64_t chunk = std::max<int64_t>(1, (int64_t(1) << 24)/unit)*unit;

    std::vector<float>   data(std::min(nels, chunk));
    std::vector<uint8_t> dataq;
    std::mt19937         rng(seed);
    std::uniform_real_distribution<float> uni(-1.0f, 1.0f);

    const int64_t n_threads = std::max(1u, std::min(8u, std::thread::hardware_concurrency()));
    for (int64_t start = 0; start < nels; start += chunk) {
        const int64_t n = std::min(chunk, nels - start);
        for (int64_t i = 0; i < n; ++i) {
            data[i] = uni(rng);
        }
        dataq.resize(ggml_row_size(c.type_a, n));
        const int64_t n_blocks = n/blck;
        const int64_t per_thread = std::max<int64_t>(1, n_blocks/n_threads);
        std::vector<std::future<void>> tasks;
        for (int64_t b0 = 0; b0 < n_blocks; b0 += per_thread) {
            const int64_t b1 = std::min(b0 + per_thread, n_blocks);
            tasks.push_back(std::async(std::launch::async, [&, b0, b1] {
                ggml_quantize_chunk(c.type_a, data.data(), dataq.data(), b0*blck, b1 - b0, blck, nullptr);
            }));
        }
        for (auto & t : tasks) {
            t.get();
        }
        const size_t off = ggml_row_size(c.type_a, start);
        ggml_backend_tensor_set(g0.as, dataq.data(), off, dataq.size());
        ggml_backend_tensor_set(g1.as, dataq.data(), off, dataq.size());
    }

    // the routing of a served step: each token 8 distinct experts of the 384, no skipped slot
    std::mt19937 ri(seed ^ 0x9e3779b9u);
    std::vector<int32_t> ids(size_t(c.n_used*c.n_tokens));
    std::vector<int32_t> perm(size_t(c.n_mats));
    for (int64_t t = 0; t < c.n_tokens; ++t) {
        for (int64_t i = 0; i < c.n_mats; ++i) {
            perm[size_t(i)] = int32_t(i);
        }
        std::shuffle(perm.begin(), perm.end(), ri);
        for (int64_t s = 0; s < c.n_used; ++s) {
            ids[size_t(t*c.n_used + s)] = perm[size_t(s)];
        }
    }
    ggml_backend_tensor_set(g0.ids, ids.data(), 0, ids.size()*sizeof(int32_t));
    ggml_backend_tensor_set(g1.ids, ids.data(), 0, ids.size()*sizeof(int32_t));

    std::vector<float> bv(size_t(ggml_nelements(g0.b)));
    for (float & x : bv) {
        x = uni(rng);
    }
    ggml_backend_tensor_set(g0.b, bv.data(), 0, bv.size()*sizeof(float));
    ggml_backend_tensor_set(g1.b, bv.data(), 0, bv.size()*sizeof(float));
}

static bool compute(ggml_backend_t backend, seam_graph & g) {
    const ggml_status status = ggml_backend_graph_compute(backend, g.gf);
    ggml_backend_synchronize(backend);
    return status == GGML_STATUS_SUCCESS;
}

static std::vector<float> get(const ggml_tensor * t) {
    std::vector<float> v(size_t(ggml_nelements(t)));
    ggml_backend_tensor_get(t, v.data(), 0, ggml_nbytes(t));
    return v;
}

static double nmse(const std::vector<float> & y, const std::vector<float> & ref) {
    double err = 0.0, sum = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        err += (y[i] - ref[i]) * double(y[i] - ref[i]);
        sum += ref[i] * double(ref[i]);
    }
    return sum == 0.0 ? (err == 0.0 ? 0.0 : INFINITY) : err / sum;
}

static bool run_case(ggml_backend_t backend, ggml_backend_t cpu, const seam_case & c, uint32_t seed, FILE * output) {
    printf("  %s, k %4" PRId64 ", m %4" PRId64 ", %" PRId64 " experts, %" PRId64 " used, b %s, %" PRId64 " tokens: ",
           ggml_type_name(c.type_a), c.k, c.m, c.n_mats, c.n_used, c.b_broadcast ? "shared" : "a slot", c.n_tokens);
    fflush(stdout);

    seam_graph gc, gd;
    if (!build(gc, cpu, c) || !build(gd, backend, c)) {
        printf("graph allocation FAILED\n");
        return false;
    }
    if (!ggml_backend_supports_op(backend, gd.out)) {
        printf("CUDA op not supported\n");
        return false;
    }
    fill(gc, gd, c, seed);

    if (!compute(cpu, gc)) {
        printf("CPU backend FAILED\n");
        return false;
    }
    const std::vector<float> y_ref = get(gc.out);

    if (!compute(backend, gd)) {
        printf("CUDA backend FAILED\n");
        return false;
    }
    const std::vector<float> result = get(gd.out);
    const double error = nmse(result, y_ref);
    const bool ok = error <= max_nmse &&
        fwrite(result.data(), sizeof(float), result.size(), output) == result.size();
    printf("NMSE %.2e: %s\n", error, ok ? "OK" : "FAILED");
    return ok;
}

int main(int argc, char ** argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s OUTPUT\n", argv[0]);
        return 1;
    }
    FILE * output = fopen(argv[1], "wb");
    if (!output) {
        perror(argv[1]);
        return 1;
    }

    // the kernels run live, not from a graph captured on the first compute: the two paths must both launch here
    set_env("GGML_CUDA_DISABLE_GRAPHS", "1");
    // the dispatch this test judges: 23 tokens on the few-row MMQ for the K-quant expert products, no earlier path
    set_env("GGML_CUDA_MMVQ_KQUANT_MMID_MAX", "8");
    set_env("GGML_CUDA_MMQ_FEW_MAX_COLS", "32");
    set_env("GGML_CUDA_MMQ_FEW_LEGACY", nullptr);
    set_env("GGML_CUDA_MMQ_TMA", "0");

    ggml_backend_load_all();

    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    GGML_ASSERT(cpu != nullptr);
    ggml_backend_reg_t cpu_reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(cpu));
    auto set_n_threads = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(cpu_reg, "ggml_backend_set_n_threads");
    if (set_n_threads) {
        set_n_threads(cpu, int(std::max(1u, std::min(8u, std::thread::hardware_concurrency()))));
    }

    const std::vector<seam_case> cases = {
        {GGML_TYPE_Q3_K, 4608, 4608, 384, 8, true,  23}, // gate/up: a token's vector shared by its 8 experts
        {GGML_TYPE_Q4_K, 2304, 4608, 384, 8, false, 23}, // down: a vector a slot
    };

    int n_tested = 0;
    bool ok = true;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU ||
                strcmp(ggml_backend_reg_name(ggml_backend_dev_backend_reg(dev)), "CUDA") != 0) {
            continue;
        }
        n_tested++;
        printf("%s (%s):\n", ggml_backend_dev_name(dev), ggml_backend_dev_description(dev));
        ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
        if (backend == nullptr) {
            printf("  backend initialization FAILED\n");
            ok = false;
            continue;
        }
        for (size_t ic = 0; ic < cases.size(); ++ic) {
            ok &= run_case(backend, cpu, cases[ic], 0x5EA1u + uint32_t(ic), output);
        }
        ggml_backend_free(backend);
    }
    ggml_backend_free(cpu);

    ok &= fclose(output) == 0;
    if (n_tested == 0) {
        printf("no CUDA device\n");
        return 1;
    }
    printf("%s\n", ok ? "OK" : "FAILED");
    return ok ? 0 : 1;
}
