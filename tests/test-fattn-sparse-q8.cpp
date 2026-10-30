// GLM-5.3's DSA attention on a q8_0 or q4_0 cache: one 512-wide latent head, the V cache a view of the K cache, and each
// query's n_kv_max cells named by its mask row (ggml_flash_attn_ext_set_n_kv_max). The CUDA backend reads the quantized
// rows a query gathers and uses each row for K and for V, with no f16 copy of the cache. For both cache types and every
// launch shape GLM serves:
//
//   1. the quantized cache's output is the output of the same launch on an f16 cache that holds the quantized cache's
//      values, bit for bit: a q8_0 or q4_0 value is its integer times the block's f16 scale, rounded to f16 once, and
//      that f16 value is all an f16 copy of the cache ever held;
//   2. a sparse launch reserves no f16 copy: the output tensor's allocation is its own bytes, where a copy of a
//      524,288-cell cache is 512 MiB a card. A launch the sparse kernel does not take (a cache under 4096 cells, or under
//      the cells its tokens gather, n_kv_max for each of up to 8) reserves the copy its f16 path writes, so the two
//      never disagree;
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
#include <cstring>
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
    ggml_tensor *         out_q   = nullptr; // the quantized cache, V a view of K
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
static void build(fattn_graph & g, ggml_backend_t backend, const fattn_case & c, ggml_type type, const std::vector<float> & q,
        const std::vector<uint8_t> & cache_q, const std::vector<ggml_fp16_t> & cache_f16,
        const std::vector<ggml_fp16_t> & mask, bool only_f16) {
    ggml_init_params params_in = { ggml_tensor_overhead() * 16, nullptr, true };
    g.ctx_in = ggml_init(params_in);

    ggml_tensor * tq     = ggml_new_tensor_4d(g.ctx_in, GGML_TYPE_F32,  D, c.nb, c.nh, c.ns);
    ggml_tensor * tc_q   = ggml_new_tensor_4d(g.ctx_in, type,           D, c.kv, 1, c.ns); // the caches
    ggml_tensor * tc_f16 = ggml_new_tensor_4d(g.ctx_in, GGML_TYPE_F16,  D, c.kv, 1, c.ns);
    ggml_tensor * tc_cpy = ggml_new_tensor_4d(g.ctx_in, GGML_TYPE_F16,  D, c.kv, 1, c.ns);
    ggml_tensor * tm     = ggml_new_tensor_4d(g.ctx_in, GGML_TYPE_F16,  c.kv, c.nb, 1, c.ns);

    g.buf = ggml_backend_alloc_ctx_tensors(g.ctx_in, backend);
    GGML_ASSERT(g.buf != nullptr);
    ggml_backend_tensor_set(tq,     q.data(),         0, ggml_nbytes(tq));
    ggml_backend_tensor_set(tc_q,   cache_q.data(),   0, ggml_nbytes(tc_q));
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
        ggml_tensor * kq = view(tc_q);
        g.out_q = attention(ctx, tq, kq, view(kq), tm);
        ggml_build_forward_expand(g.gf, g.out_q);
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

// the first cache the sparse launch takes for nb tokens: 4096 cells, and every cell its 8-head tile gathers
static int64_t sparse_from(int64_t nb) {
    const int64_t cells = std::max<int64_t>(4096, std::min<int64_t>(nb, 8) * n_kv_max);
    return (cells + 255) / 256 * 256;
}

static bool run_case(ggml_backend_t backend, ggml_backend_t cpu, const fattn_case & c, ggml_type type, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> uni(-1.0f, 1.0f);

    std::vector<float> q(size_t(D * c.nb * c.nh * c.ns));
    for (float & x : q) {
        x = uni(rng);
    }

    // the quantized cache from generated values, and an f16 cache that holds its values
    const int64_t nrows = c.kv * c.ns;
    std::vector<float> values(size_t(D * nrows));
    for (float & x : values) {
        x = uni(rng);
    }
    std::vector<uint8_t> cache_q(ggml_row_size(type, D) * nrows);
    ggml_quantize_chunk(type, values.data(), cache_q.data(), 0, nrows, D, nullptr);
    ggml_get_type_traits(type)->to_float(cache_q.data(), values.data(), D * nrows);
    std::vector<ggml_fp16_t> cache_f16(values.size());
    ggml_fp32_to_fp16_row(values.data(), cache_f16.data(), int64_t(values.size()));

    const std::vector<ggml_fp16_t> mask = make_mask(c, rng);

    fattn_graph g;
    build(g, backend, c, type, q, cache_q, cache_f16, mask, false);
    if (!ggml_backend_supports_op(backend, g.out_q) || !ggml_backend_supports_op(backend, g.out_f16)) {
        printf("  %s, kv %6" PRId64 ", %3" PRId64 " tokens, %2" PRId64 " heads, %" PRId64 " seq, %s: not supported, skipped\n",
               ggml_type_name(type), c.kv, c.nb, c.nh, c.ns, rows_name(c.rows));
        return true;
    }

    const ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
    const size_t alloc_q = ggml_backend_buft_get_alloc_size(buft, g.out_q);

    GGML_ASSERT(ggml_backend_graph_compute(backend, g.gf) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(backend);
    const std::vector<float> y_q   = get(g.out_q);
    const std::vector<float> y_f16 = get(g.out_f16);
    const std::vector<float> y_cpy = get(g.out_cpy);

    fattn_graph g_cpu;
    build(g_cpu, cpu, c, type, q, cache_q, cache_f16, mask, true);
    GGML_ASSERT(ggml_backend_graph_compute(cpu, g_cpu.gf) == GGML_STATUS_SUCCESS);
    const double e = nmse(y_f16, get(g_cpu.out_f16));

    const int64_t diff_q   = first_bit_difference(y_q, y_f16);
    const int64_t diff_cpy = first_bit_difference(y_cpy, y_f16);
    const bool    sparse   = c.kv >= sparse_from(c.nb);
    const bool    no_copy  = alloc_q == ggml_nbytes(g.out_q);
    const bool    finite   = all_finite(y_f16);
    const bool    ok       = diff_q < 0 && diff_cpy < 0 && no_copy == sparse && finite && e < max_nmse;
    const char *  name     = ggml_type_name(type);

    printf("  %s, kv %6" PRId64 ", %3" PRId64 " tokens, %2" PRId64 " heads, %" PRId64 " seq, %-14s: ", name, c.kv, c.nb, c.nh,
           c.ns, rows_name(c.rows));
    if (diff_q < 0) {
        printf("%s = f16 bit for bit", name);
    } else {
        printf("%s differs from f16 at %" PRId64 " (%.9g against %.9g)", name, diff_q, y_q[diff_q], y_f16[diff_q]);
    }
    if (sparse) {
        if (no_copy) {
            printf(", sparse, no f16 copy");
        } else {
            printf(", sparse, %zu bytes reserved for %zu of output", alloc_q, ggml_nbytes(g.out_q));
        }
    } else {
        if (no_copy) {
            printf(", dense, NO f16 copy reserved for its f16 path");
        } else {
            printf(", dense, its f16 copy reserved");
        }
    }
    printf(diff_cpy < 0 ? ", V view = V copy" : ", V view differs from V copy");
    printf(", nmse %.2e vs CPU%s: %s\n", e, finite ? "" : ", NOT FINITE", ok ? "ok" : "FAILED");
    return ok;
}

int main() {
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
        printf("%s (%s):\n", ggml_backend_dev_name(dev), ggml_backend_dev_description(dev));
        ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
        for (ggml_type type : { GGML_TYPE_Q8_0, GGML_TYPE_Q4_0 }) {
            for (size_t ic = 0; ic < cases.size(); ++ic) {
                ok &= run_case(backend, cpu, cases[ic], type, 0x51A7u + uint32_t(ic) + (type == GGML_TYPE_Q4_0 ? 0x100u : 0u));
            }
        }
        ggml_backend_free(backend);
        n_tested++;
    }
    ggml_backend_free(cpu);

    if (n_tested == 0) {
        printf("no CUDA device, skipped\n");
        return 0;
    }
    printf("%s\n", ok ? "OK" : "FAILED");
    return ok ? 0 : 1;
}
