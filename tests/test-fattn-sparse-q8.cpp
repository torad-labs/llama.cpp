// GLM-5.3's DSA attention on a q8_0 cache: one 512-wide latent head, the V cache a view of the K cache, and each query's
// n_kv_max cells named by its mask row (ggml_flash_attn_ext_set_n_kv_max). The CUDA backend reads the q8_0 rows a query
// gathers and uses each row for K and for V, with no f16 copy of the cache. For every launch shape GLM serves:
//
//   1. the q8_0 cache's output is the output of the same launch on an f16 cache that holds the q8_0 cache's values, bit
//      for bit: a q8_0 value is its integer times the block's f16 scale, rounded to f16 once, and that f16 value is all
//      an f16 copy of the cache ever held;
//   2. a sparse launch on a card that reads in place reserves no f16 copy: the output tensor's allocation is its own
//      bytes, where a copy of a 524,288-cell cache is 512 MiB a card. The in-place read needs cp.async, an NVIDIA card of
//      compute capability 8.0 or later in a build for sm_80 or later. A launch it does not take (such a card or build
//      below it, a cache under 4096 cells, or under the cells its tokens gather, n_kv_max for each of up to 8) reserves
//      the copy its f16 path writes, so the two never disagree;
//   3. on the f16 cache, V as a view of K gives what V as a separate copy of the same values gives, bit for bit;
//   4. the f16 cache's output matches the CPU backend's (test-backend-ops' bound), so 1 and 3 compare real attention.
//
// The shapes: a decode and an MTP verify (1-3 tokens) on the last cache before the sparse launch takes them, the first
// it takes, and on to 40960;
// prefill ubatches on the 8-head tile and on the 32-head tile (taken from 376 tokens at 32 heads, 188 at 64, on an RTX
// PRO 6000's 188 SMs); 32 and 64 heads (a card's share under -sm tensor, and one card); 1 and 2 sequences. Each row's
// cells are generated: rows near the bound, rows far under it (the -1 slots past a row's count), and a sequence's first
// ubatch (row r on its r + 1 cells at the cache's end). The values are generated from a seed per case.
//
// Devices that are not CUDA are skipped, and with no CUDA device the test passes.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <thread>
#include <vector>

static constexpr int64_t D        = 512;  // GLM-5.3's latent, K and V
static constexpr int64_t n_kv_max = 2080; // GLM-5.3's top-k bound: 2048 + 4 - 1, padded to 32

enum rows_kind {
    ROWS_NEAR_BOUND, // n_kv_max - row % 17 cells
    ROWS_FEW,        // 1, 31, 32, 33, 63, 64, 65 ... then spread under the bound
    ROWS_FIRST,      // row r on its r + 1 cells at the cache's end
};

struct fattn_case {
    int64_t   kv; // cache cells
    int64_t   nb; // tokens a sequence
    int64_t   nh; // query heads on the one latent head
    int64_t   ns; // sequences
    rows_kind rows;
};

static const char * rows_name(rows_kind r) {
    switch (r) {
        case ROWS_NEAR_BOUND: return "near the bound";
        case ROWS_FEW:        return "few cells";
        case ROWS_FIRST:      return "first ubatch";
    }
    return "?";
}

// one f16 mask row per token and sequence: 0 or a small bias at the row's cells, -inf elsewhere
static std::vector<ggml_fp16_t> make_mask(const fattn_case & c, std::mt19937 & rng) {
    const int64_t nrows = c.nb * c.ns;
    std::vector<float> m(size_t(c.kv * nrows), -INFINITY);
    std::vector<int32_t> order(c.kv);
    for (int64_t i = 0; i < c.kv; ++i) {
        order[i] = int32_t(i);
    }
    for (int64_t row = 0; row < nrows; ++row) {
        int64_t count = 0;
        switch (c.rows) {
            case ROWS_NEAR_BOUND: count = n_kv_max - row % 17; break;
            case ROWS_FEW: {
                const int64_t first[] = {1, 31, 32, 33, 63, 64, 65, n_kv_max/2, n_kv_max - 1, n_kv_max};
                const int64_t nfirst  = sizeof(first)/sizeof(first[0]);
                count = row < nfirst ? first[row] : 1 + (row*389) % n_kv_max;
            } break;
            case ROWS_FIRST: count = std::min<int64_t>(row % c.nb + 1, n_kv_max); break;
        }
        count = std::max<int64_t>(1, std::min<int64_t>(count, n_kv_max));

        std::uniform_real_distribution<float> bias(-0.25f, 0.0f);
        if (c.rows == ROWS_FIRST) {
            for (int64_t i = 0; i < count; ++i) {
                m[row*c.kv + c.kv - n_kv_max + i] = bias(rng);
            }
        } else {
            for (int64_t i = 0; i < count; ++i) { // a partial Fisher-Yates: the first count of a permutation
                std::uniform_int_distribution<int64_t> pick(i, c.kv - 1);
                std::swap(order[i], order[pick(rng)]);
                m[row*c.kv + order[i]] = bias(rng);
            }
        }
    }
    std::vector<ggml_fp16_t> m16(m.size());
    ggml_fp32_to_fp16_row(m.data(), m16.data(), int64_t(m.size()));
    return m16;
}

struct fattn_graph {
    ggml_context *        ctx_in  = nullptr;
    ggml_context *        ctx_out = nullptr;
    ggml_backend_buffer_t buf     = nullptr;
    ggml_gallocr_t        galloc  = nullptr;
    ggml_cgraph *         gf      = nullptr;
    ggml_tensor *         out_q8  = nullptr; // q8_0 cache, V a view of K
    ggml_tensor *         out_f16 = nullptr; // f16 cache of the same values, V a view of K
    ggml_tensor *         out_cpy = nullptr; // the same f16 cache, V a separate copy

    ~fattn_graph() {
        ggml_gallocr_free(galloc);
        ggml_backend_buffer_free(buf);
        ggml_free(ctx_out);
        ggml_free(ctx_in);
    }
};

static ggml_tensor * attention(ggml_context * ctx, ggml_tensor * q, ggml_tensor * k, ggml_tensor * v, ggml_tensor * mask) {
    ggml_tensor * out = ggml_flash_attn_ext(ctx, q, k, v, mask, 1.0f/sqrtf(float(D)), 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);
    ggml_flash_attn_ext_set_n_kv_max(out, n_kv_max);
    return out;
}

// builds the three launches on one backend over the same values; with only_f16 the f16 cache's alone (the CPU check)
static void build(fattn_graph & g, ggml_backend_t backend, const fattn_case & c, const std::vector<float> & q,
        const std::vector<uint8_t> & cache_q8, const std::vector<ggml_fp16_t> & cache_f16,
        const std::vector<ggml_fp16_t> & mask, bool only_f16) {
    ggml_init_params params_in = { ggml_tensor_overhead() * 16, nullptr, true };
    g.ctx_in = ggml_init(params_in);

    ggml_tensor * tq     = ggml_new_tensor_4d(g.ctx_in, GGML_TYPE_F32,  D, c.nb, c.nh, c.ns);
    ggml_tensor * tc_q8  = ggml_new_tensor_4d(g.ctx_in, GGML_TYPE_Q8_0, D, c.kv, 1, c.ns); // the caches
    ggml_tensor * tc_f16 = ggml_new_tensor_4d(g.ctx_in, GGML_TYPE_F16,  D, c.kv, 1, c.ns);
    ggml_tensor * tc_cpy = ggml_new_tensor_4d(g.ctx_in, GGML_TYPE_F16,  D, c.kv, 1, c.ns);
    ggml_tensor * tm     = ggml_new_tensor_4d(g.ctx_in, GGML_TYPE_F16,  c.kv, c.nb, 1, c.ns);

    g.buf = ggml_backend_alloc_ctx_tensors(g.ctx_in, backend);
    GGML_ASSERT(g.buf != nullptr);
    ggml_backend_tensor_set(tq,     q.data(),         0, ggml_nbytes(tq));
    ggml_backend_tensor_set(tc_q8,  cache_q8.data(),  0, ggml_nbytes(tc_q8));
    ggml_backend_tensor_set(tc_f16, cache_f16.data(), 0, ggml_nbytes(tc_f16));
    ggml_backend_tensor_set(tc_cpy, cache_f16.data(), 0, ggml_nbytes(tc_cpy));
    ggml_backend_tensor_set(tm,     mask.data(),      0, ggml_nbytes(tm));

    ggml_init_params params_out = { ggml_tensor_overhead() * 16 + ggml_graph_overhead(), nullptr, true };
    g.ctx_out = ggml_init(params_out);
    ggml_context * ctx = g.ctx_out;

    // K a view of the cache, V a view of K, as llama's graph for GLM-5.3 builds them
    auto view = [&](ggml_tensor * t) {
        return ggml_view_4d(ctx, t, D, c.kv, 1, c.ns, t->nb[1], t->nb[2], t->nb[3], 0);
    };
    g.gf = ggml_new_graph(ctx);
    if (!only_f16) {
        ggml_tensor * k8 = view(tc_q8);
        g.out_q8 = attention(ctx, tq, k8, view(k8), tm);
        ggml_build_forward_expand(g.gf, g.out_q8);
    }
    ggml_tensor * k16 = view(tc_f16);
    g.out_f16 = attention(ctx, tq, k16, view(k16), tm);
    ggml_build_forward_expand(g.gf, g.out_f16);
    if (!only_f16) {
        g.out_cpy = attention(ctx, tq, k16, view(tc_cpy), tm);
        ggml_build_forward_expand(g.gf, g.out_cpy);
    }

    g.galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    GGML_ASSERT(ggml_gallocr_alloc_graph(g.galloc, g.gf));
}

static std::vector<float> get(const ggml_tensor * t) {
    std::vector<float> v(ggml_nelements(t));
    ggml_backend_tensor_get(t, v.data(), 0, ggml_nbytes(t));
    return v;
}

static int64_t first_bit_difference(const std::vector<float> & a, const std::vector<float> & b) {
    for (size_t i = 0; i < a.size(); ++i) {
        if (memcmp(&a[i], &b[i], sizeof(float)) != 0) {
            return int64_t(i);
        }
    }
    return -1;
}

static bool all_finite(const std::vector<float> & a) {
    return std::all_of(a.begin(), a.end(), [](float x) { return std::isfinite(x); });
}

static double nmse(const std::vector<float> & y, const std::vector<float> & ref) {
    double err = 0.0, sum = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        err += (y[i] - ref[i]) * double(y[i] - ref[i]);
        sum += ref[i] * double(ref[i]);
    }
    return err / sum;
}

static constexpr double max_nmse = 5e-4; // test-backend-ops' bound for FLASH_ATTN_EXT

// each CUDA device's compute capability, 100 x major + 10 x minor, as the backend reports it when it starts
static std::map<int, int> g_cuda_cc;

static void log_cb(ggml_log_level level, const char * text, void * user_data) {
    int id = 0, major = 0, minor = 0;
    if (sscanf(text, "  Device %d: %*[^,], compute capability %d.%d", &id, &major, &minor) == 3) {
        g_cuda_cc[id] = 100*major + 10*minor;
    }
    (void) level;
    (void) user_data;
    fputs(text, stderr);
}

// the highest arch the CUDA backend was built for that the device runs (its ARCHS feature), or -1 when it says none
static int highest_compiled_arch(ggml_backend_reg_t reg, int cc) {
    auto get_features = (ggml_backend_get_features_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_get_features");
    if (get_features == nullptr) {
        return -1;
    }
    for (ggml_backend_feature * f = get_features(reg); f->name != nullptr; ++f) {
        if (strcmp(f->name, "ARCHS") != 0) {
            continue;
        }
        int best = -1;
        for (const char * s = f->value; *s != '\0'; ) {
            char * end = nullptr;
            const long arch = strtol(s, &end, 10);
            if (end == s) {
                ++s;
                continue;
            }
            if (arch <= cc) {
                best = std::max(best, int(arch));
            }
            s = end;
        }
        return best;
    }
    return -1;
}

// the first cache the sparse launch takes for nb tokens: 4096 cells, and every cell its 8-head tile gathers
static int64_t sparse_from(int64_t nb) {
    const int64_t cells = std::max<int64_t>(4096, std::min<int64_t>(nb, 8) * n_kv_max);
    return (cells + 255) / 256 * 256;
}

static bool run_case(ggml_backend_t backend, ggml_backend_t cpu, const fattn_case & c, bool reads_in_place, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> uni(-1.0f, 1.0f);

    std::vector<float> q(size_t(D * c.nb * c.nh * c.ns));
    for (float & x : q) {
        x = uni(rng);
    }

    // the q8_0 cache from generated values, and an f16 cache that holds its values
    const int64_t nrows = c.kv * c.ns;
    std::vector<float> values(size_t(D * nrows));
    for (float & x : values) {
        x = uni(rng);
    }
    std::vector<uint8_t> cache_q8(ggml_row_size(GGML_TYPE_Q8_0, D) * nrows);
    ggml_quantize_chunk(GGML_TYPE_Q8_0, values.data(), cache_q8.data(), 0, nrows, D, nullptr);
    ggml_get_type_traits(GGML_TYPE_Q8_0)->to_float(cache_q8.data(), values.data(), D * nrows);
    std::vector<ggml_fp16_t> cache_f16(values.size());
    ggml_fp32_to_fp16_row(values.data(), cache_f16.data(), int64_t(values.size()));

    const std::vector<ggml_fp16_t> mask = make_mask(c, rng);

    fattn_graph g;
    build(g, backend, c, q, cache_q8, cache_f16, mask, false);
    if (!ggml_backend_supports_op(backend, g.out_q8) || !ggml_backend_supports_op(backend, g.out_f16)) {
        printf("  kv %6" PRId64 ", %3" PRId64 " tokens, %2" PRId64 " heads, %" PRId64 " seq, %s: not supported, skipped\n",
               c.kv, c.nb, c.nh, c.ns, rows_name(c.rows));
        return true;
    }

    const ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
    const size_t alloc_q8 = ggml_backend_buft_get_alloc_size(buft, g.out_q8);

    GGML_ASSERT(ggml_backend_graph_compute(backend, g.gf) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(backend);
    const std::vector<float> y_q8  = get(g.out_q8);
    const std::vector<float> y_f16 = get(g.out_f16);
    const std::vector<float> y_cpy = get(g.out_cpy);

    fattn_graph g_cpu;
    build(g_cpu, cpu, c, q, cache_q8, cache_f16, mask, true);
    GGML_ASSERT(ggml_backend_graph_compute(cpu, g_cpu.gf) == GGML_STATUS_SUCCESS);
    const double e = nmse(y_f16, get(g_cpu.out_f16));

    const int64_t diff_q8  = first_bit_difference(y_q8, y_f16);
    const int64_t diff_cpy = first_bit_difference(y_cpy, y_f16);
    const bool    sparse   = c.kv >= sparse_from(c.nb);
    const bool    in_place = sparse && reads_in_place;
    const bool    no_copy  = alloc_q8 == ggml_nbytes(g.out_q8);
    const bool    finite   = all_finite(y_f16);
    const bool    ok       = diff_q8 < 0 && diff_cpy < 0 && no_copy == in_place && finite && e < max_nmse;

    printf("  kv %6" PRId64 ", %3" PRId64 " tokens, %2" PRId64 " heads, %" PRId64 " seq, %-14s: ", c.kv, c.nb, c.nh, c.ns,
           rows_name(c.rows));
    if (diff_q8 < 0) {
        printf("q8_0 = f16 bit for bit");
    } else {
        printf("q8_0 differs from f16 at %" PRId64 " (%.9g against %.9g)", diff_q8, y_q8[diff_q8], y_f16[diff_q8]);
    }
    if (in_place) {
        if (no_copy) {
            printf(", sparse, no f16 copy");
        } else {
            printf(", sparse, %zu bytes reserved for %zu of output", alloc_q8, ggml_nbytes(g.out_q8));
        }
    } else {
        const char * path = sparse ? "sparse below sm_80" : "dense";
        if (no_copy) {
            printf(", %s, NO f16 copy reserved for its f16 path", path);
        } else {
            printf(", %s, its f16 copy reserved", path);
        }
    }
    printf(diff_cpy < 0 ? ", V view = V copy" : ", V view differs from V copy");
    printf(", nmse %.2e vs CPU%s: %s\n", e, finite ? "" : ", NOT FINITE", ok ? "ok" : "FAILED");
    return ok;
}

int main() {
    ggml_log_set(log_cb, nullptr);
    ggml_backend_load_all();

    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    GGML_ASSERT(cpu != nullptr);
    ggml_backend_reg_t cpu_reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(cpu));
    auto set_n_threads = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(cpu_reg, "ggml_backend_set_n_threads");
    if (set_n_threads) {
        set_n_threads(cpu, int(std::max(1u, std::thread::hardware_concurrency())));
    }

    std::vector<fattn_case> cases;
    // decodes and verifies: the last cache before the sparse launch takes them, the first it takes, then to 40960 cells
    for (int64_t nb : { 1, 2, 3 }) {
        for (int64_t kv : { sparse_from(nb) - 256, sparse_from(nb), int64_t(8448), int64_t(40960) }) {
            cases.push_back({ kv, nb, 32, 1, ROWS_NEAR_BOUND });
        }
    }
    for (int64_t nb : { 1, 3 }) {
        cases.push_back({ 16640, nb, 64, 2, ROWS_NEAR_BOUND });
        cases.push_back({ 16640, nb, 32, 1, ROWS_FEW });
    }
    // prefill: the 8-head tile, then the 32-head tile with its seams pass
    cases.push_back({ 16640,  64, 32, 1, ROWS_NEAR_BOUND });
    cases.push_back({ 16640,  64, 32, 1, ROWS_FIRST });
    cases.push_back({ 16640, 512, 64, 1, ROWS_NEAR_BOUND });
    cases.push_back({ 16640, 512, 64, 1, ROWS_FEW });
    cases.push_back({ 40960, 384, 32, 2, ROWS_FIRST });

    bool ok = true;
    int n_tested = 0;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU ||
                strcmp(ggml_backend_reg_name(ggml_backend_dev_backend_reg(dev)), "CUDA") != 0) {
            continue;
        }
        n_tested++;
        // CUDA0, CUDA1, ...: the backend's devices in CUDA's order
        const char * name = ggml_backend_dev_name(dev);
        const int    id   = strncmp(name, "CUDA", 4) == 0 ? atoi(name + 4) : -1;
        const auto   cc   = g_cuda_cc.find(id);
        if (cc == g_cuda_cc.end()) {
            printf("%s (%s): the backend reported no compute capability for it: FAILED\n", name, ggml_backend_dev_description(dev));
            ok = false;
            continue;
        }
        const int  arch           = highest_compiled_arch(ggml_backend_dev_backend_reg(dev), cc->second);
        if (arch < 0) {
            printf("%s (%s): the backend reported no arch it was built for that runs here: FAILED\n", name,
                   ggml_backend_dev_description(dev));
            ok = false;
            continue;
        }
        const bool reads_in_place = arch >= 800;
        printf("%s (%s), compute capability %d.%d, built for sm_%d: the sparse launch %s\n", name,
               ggml_backend_dev_description(dev), cc->second / 100, cc->second % 100 / 10, arch / 10,
               reads_in_place ? "reads the cache in place" : "keeps its f16 copy");
        ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
        for (size_t ic = 0; ic < cases.size(); ++ic) {
            ok &= run_case(backend, cpu, cases[ic], reads_in_place, 0x51A7u + uint32_t(ic));
        }
        ggml_backend_free(backend);
    }
    ggml_backend_free(cpu);

    if (n_tested == 0) {
        printf("no CUDA device, skipped\n");
        return 0;
    }
    printf("%s\n", ok ? "OK" : "FAILED");
    return ok ? 0 : 1;
}
