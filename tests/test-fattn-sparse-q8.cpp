// GLM-5.3's DSA attention on a q8_0 or q4_0 cache: one 512-wide latent head, the V cache a view of the K cache, and each
// query's n_kv_max cells named by its mask row (ggml_flash_attn_ext_set_n_kv_max). The CUDA backend reads the quantized
// rows a query gathers and uses each row for K and for V, with no f16 copy of the cache. For both cache types and every
// launch shape GLM serves:
//
//   1. the quantized cache's output is the output of the same launch on an f16 cache that holds the quantized cache's
//      values, bit for bit: a q8_0 or q4_0 value is its integer times the block's f16 scale, rounded to f16 once, and
//      that f16 value is all an f16 copy of the cache ever held;
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
// The default short suite skips non-CUDA devices. --direct-longctx requires all 16 cases per CUDA device.
// --require-cuda rejects no-device runs; --cpu-only runs just the small CPU semantic checks.

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
        if (c.kv == 400128 && c.nb <= 3) {
            const int64_t high[] = {399999, 131073, 262145};
            for (int64_t i = 0; i < count; ++i) {
                const int64_t cell = i < 3 ? high[i] : i - 3;
                m[row*c.kv + cell] = bias(rng);
            }
        } else if (c.rows == ROWS_FIRST) {
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

static ggml_tensor * attention(ggml_context * ctx, ggml_tensor * q, ggml_tensor * k, ggml_tensor * v, ggml_tensor * mask, ggml_tensor * indices = nullptr) {
    ggml_tensor * out = indices ? ggml_flash_attn_ext_sparse(ctx, q, k, v, indices, 1.0f/sqrtf(float(D))) :
        ggml_flash_attn_ext(ctx, q, k, v, mask, 1.0f/sqrtf(float(D)), 0.0f, 0.0f);
    ggml_prec_set_acc(out, GGML_PREC_F32);
    ggml_flash_attn_ext_set_n_kv_max(out, n_kv_max);
    return out;
}

// builds the three launches on one backend over the same values; with only_f16 the f16 cache's alone (the CPU check)
static void build(fattn_graph & g, ggml_backend_t backend, const fattn_case & c, ggml_type type, const std::vector<float> & q,
        const std::vector<uint8_t> & cache_q, const std::vector<ggml_fp16_t> & cache_f16,
        const std::vector<ggml_fp16_t> & mask, bool only_f16, bool direct = false) {
    ggml_init_params params_in = { ggml_tensor_overhead() * 16, nullptr, true };
    g.ctx_in = ggml_init(params_in);

    ggml_tensor * tq     = ggml_new_tensor_4d(g.ctx_in, GGML_TYPE_F32,  D, c.nb, c.nh, c.ns);
    ggml_tensor * tc_q   = ggml_new_tensor_4d(g.ctx_in, type,           D, c.kv, 1, c.ns); // the caches
    ggml_tensor * tc_f16 = ggml_new_tensor_4d(g.ctx_in, GGML_TYPE_F16,  D, c.kv, 1, c.ns);
    ggml_tensor * tc_cpy = ggml_new_tensor_4d(g.ctx_in, GGML_TYPE_F16,  D, c.kv, 1, c.ns);
    ggml_tensor * tm     = ggml_new_tensor_4d(g.ctx_in, GGML_TYPE_F16,  c.kv, c.nb, 1, c.ns);
    ggml_tensor * ti = direct ? ggml_new_tensor_4d(g.ctx_in, GGML_TYPE_I32, n_kv_max, c.nb, 1, c.ns) : nullptr;
    ggml_set_name(tm, "mask");
    if (ti) { ggml_set_name(ti, "direct_indices"); }

    g.buf = ggml_backend_alloc_ctx_tensors(g.ctx_in, backend);
    GGML_ASSERT(g.buf != nullptr);
    ggml_backend_tensor_set(tq,     q.data(),         0, ggml_nbytes(tq));
    ggml_backend_tensor_set(tc_q,   cache_q.data(),   0, ggml_nbytes(tc_q));
    ggml_backend_tensor_set(tc_f16, cache_f16.data(), 0, ggml_nbytes(tc_f16));
    ggml_backend_tensor_set(tc_cpy, cache_f16.data(), 0, ggml_nbytes(tc_cpy));
    ggml_backend_tensor_set(tm,     mask.data(),      0, ggml_nbytes(tm));
    if (direct) {
        std::vector<int32_t> ids(size_t(n_kv_max*c.nb*c.ns), -1);
        for (int64_t row = 0; row < c.nb*c.ns; ++row) {
            int64_t count = 0;
            for (int64_t j = 0; j < c.kv; ++j) {
                const float bias = ggml_fp16_to_fp32(mask[row*c.kv+j]);
                if (std::isfinite(bias)) {
                    GGML_ASSERT(bias == 0.0f && count < n_kv_max);
                    ids[row*n_kv_max+count++] = (int32_t) j;
                }
            }
        }
        ggml_backend_tensor_set(ti, ids.data(), 0, ggml_nbytes(ti));
    }

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
        g.out_q = attention(ctx, tq, kq, view(kq), tm, ti);
        ggml_build_forward_expand(g.gf, g.out_q);
    }
    ggml_tensor * k16 = view(tc_f16);
    g.out_f16 = attention(ctx, tq, k16, view(k16), tm, ti);
    ggml_build_forward_expand(g.gf, g.out_f16);
    if (!only_f16) {
        g.out_cpy = attention(ctx, tq, k16, view(tc_cpy), tm, ti);
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
    return sum == 0.0 ? (err == 0.0 ? 0.0 : INFINITY) : err / sum;
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

static bool run_case(ggml_backend_t backend, ggml_backend_t cpu, const fattn_case & c, ggml_type type, bool reads_in_place,
        uint32_t seed, bool direct = false, bool empty_row = true, bool strict = false) {
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

    std::vector<ggml_fp16_t> mask = make_mask(c, rng);
    if (direct) {
        for (auto & x : mask) { if (std::isfinite(ggml_fp16_to_fp32(x))) { x = ggml_fp32_to_fp16(0.0f); } }
        if (empty_row) { std::fill(mask.begin(), mask.begin()+c.kv, ggml_fp32_to_fp16(-INFINITY)); }
    }

    fattn_graph g;
    build(g, backend, c, type, q, cache_q, cache_f16, mask, false, direct);
    if (!ggml_backend_supports_op(backend, g.out_q) || !ggml_backend_supports_op(backend, g.out_f16) ||
            !ggml_backend_supports_op(backend, g.out_cpy)) {
        printf("  %s, kv %6" PRId64 ", %3" PRId64 " tokens, %2" PRId64 " heads, %" PRId64 " seq, %s: not supported, skipped\n",
               ggml_type_name(type), c.kv, c.nb, c.nh, c.ns, rows_name(c.rows));
        return !strict;
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
    const bool    in_place = (direct || sparse) && reads_in_place;
    const bool    no_copy  = alloc_q == ggml_nbytes(g.out_q);
    const bool    finite   = all_finite(y_q) && all_finite(y_f16) && all_finite(y_cpy);
    bool ok = diff_q < 0 && diff_cpy < 0 && no_copy == in_place && finite && e < max_nmse;
    if (direct) {
        auto compute = [&](fattn_graph & graph) {
            GGML_ASSERT(ggml_backend_graph_compute(backend, graph.gf) == GGML_STATUS_SUCCESS);
            ggml_backend_synchronize(backend);
            const auto quant = get(graph.out_q);
            const auto f16 = get(graph.out_f16);
            const auto copy = get(graph.out_cpy);
            ok &= all_finite(quant) && all_finite(f16) && all_finite(copy);
            ok &= first_bit_difference(quant, f16) < 0 && first_bit_difference(copy, f16) < 0;
            return f16;
        };
        auto equivalent = [&](const std::vector<float> & a, const std::vector<float> & b) {
            return sparse ? first_bit_difference(a, b) < 0 : nmse(a, b) < max_nmse;
        };
        ok &= first_bit_difference(y_f16, compute(g)) < 0;
        fattn_graph legacy;
        build(legacy, backend, c, type, q, cache_q, cache_f16, mask, false);
        ok &= equivalent(y_f16, compute(legacy));

        ggml_tensor * ids = ggml_get_tensor(g.ctx_in, "direct_indices");
        std::vector<int32_t> original(size_t(n_kv_max*c.nb*c.ns));
        ggml_backend_tensor_get(ids, original.data(), 0, ggml_nbytes(ids));
        auto changed_ids = original;
        int64_t count = 0;
        while (count < n_kv_max && original[count] >= 0) { ++count; }
        const int64_t slot = count == 0 ? 0 : count - 1;
        int32_t high = int32_t(c.kv - 1);
        while (std::find(original.begin(), original.begin() + count, high) != original.begin() + count) { --high; }
        changed_ids[slot] = high;
        std::sort(changed_ids.begin(), changed_ids.begin() + std::max<int64_t>(count, 1));
        auto changed_mask = mask;
        if (count > 0) { changed_mask[original[slot]] = ggml_fp32_to_fp16(-INFINITY); }
        changed_mask[high] = ggml_fp32_to_fp16(0.0f);
        ggml_backend_tensor_set(ids, changed_ids.data(), 0, ggml_nbytes(ids));
        ggml_tensor * legacy_mask = ggml_get_tensor(legacy.ctx_in, "mask");
        ggml_backend_tensor_set(legacy_mask, changed_mask.data(), 0, ggml_nbytes(legacy_mask));
        const auto changed = compute(g);
        const auto changed_ref = compute(legacy);
        ok &= equivalent(changed, changed_ref);
        ok &= first_bit_difference(y_f16, changed) >= 0;
        if (empty_row) {
            for (int64_t i = 0; i < D*c.nh; ++i) { ok &= y_q[i] == 0.0f; }
        }
        ggml_backend_tensor_set(ids, original.data(), 0, ggml_nbytes(ids));
        ggml_backend_tensor_set(legacy_mask, mask.data(), 0, ggml_nbytes(legacy_mask));
        ok &= first_bit_difference(y_f16, compute(g)) < 0;
        ok &= equivalent(y_f16, compute(legacy));
        printf("  direct indices, %s baseline, changed-input replay + legacy parity + restore: %s\n",
               empty_row ? "empty row" : "nonempty", ok ? "ok" : "FAILED");
    }
    const char *  name     = ggml_type_name(type);

    printf("  %s, kv %6" PRId64 ", %3" PRId64 " tokens, %2" PRId64 " heads, %" PRId64 " seq, %-14s: ", name, c.kv, c.nb, c.nh,
           c.ns, rows_name(c.rows));
    if (diff_q < 0) {
        printf("%s = f16 bit for bit", name);
    } else {
        printf("%s differs from f16 at %" PRId64 " (%.9g against %.9g)", name, diff_q, y_q[diff_q], y_f16[diff_q]);
    }
    if (in_place) {
        if (no_copy) {
            printf(", sparse, no f16 copy");
        } else {
            printf(", sparse, %zu bytes reserved for %zu of output", alloc_q, ggml_nbytes(g.out_q));
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

// The semantic sparse opcode must also run on CPU, including empty rows and multiple sequences.
static bool run_direct_cpu(ggml_backend_t cpu, ggml_type type) {
    constexpr int64_t d = 32, nk = 96, nr = 3, nh = 4, ns = 2, bound = 32;
    ggml_init_params gp = { 16*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(gp);
    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d, nr, nh, ns);
    ggml_tensor * k = ggml_new_tensor_4d(ctx, type, d, nk, 1, ns);
    ggml_tensor * v = ggml_view_tensor(ctx, k);
    ggml_tensor * mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, nk, nr, 1, ns);
    ggml_tensor * ids = ggml_new_tensor_4d(ctx, GGML_TYPE_I32, bound, nr, 1, ns);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, cpu);
    std::vector<float> qv(ggml_nelements(q));
    std::vector<ggml_fp16_t> kv(ggml_nelements(k)), mv(ggml_nelements(mask), ggml_fp32_to_fp16(-INFINITY));
    std::vector<int32_t> iv(ggml_nelements(ids), -1);
    for (size_t i = 0; i < qv.size(); ++i) { qv[i] = std::sin(float(i)*0.13f); }
    for (size_t i = 0; i < kv.size(); ++i) { kv[i] = ggml_fp32_to_fp16(std::cos(float(i)*0.07f)); }
    for (int64_t row = 0; row < nr*ns; ++row) {
        const int64_t count = row == 0 ? 0 : row == 1 ? 1 : bound-row;
        for (int64_t j = 0; j < count; ++j) {
            iv[row*bound+j] = int32_t(j*3);
            mv[row*nk+j*3] = ggml_fp32_to_fp16(0.0f);
        }
    }
    ggml_backend_tensor_set(q, qv.data(), 0, ggml_nbytes(q));
    if (type == GGML_TYPE_F16) {
        ggml_backend_tensor_set(k, kv.data(), 0, ggml_nbytes(k));
    } else {
        std::vector<float> values(kv.size());
        for (size_t i = 0; i < kv.size(); ++i) { values[i] = ggml_fp16_to_fp32(kv[i]); }
        std::vector<uint8_t> quant(ggml_nbytes(k));
        ggml_quantize_chunk(type, values.data(), quant.data(), 0, nk*ns, d, nullptr);
        ggml_backend_tensor_set(k, quant.data(), 0, ggml_nbytes(k));
    }
    ggml_backend_tensor_set(mask, mv.data(), 0, ggml_nbytes(mask));
    ggml_backend_tensor_set(ids, iv.data(), 0, ggml_nbytes(ids));
    ggml_tensor * ref = ggml_flash_attn_ext(ctx, q, k, v, mask, 1.0f/sqrtf(float(d)), 0.0f, 0.0f);
    ggml_tensor * got = ggml_flash_attn_ext_sparse(ctx, q, k, v, ids, 1.0f/sqrtf(float(d)));
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, ref);
    ggml_build_forward_expand(gf, got);
    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(cpu));
    bool ok = ggml_gallocr_alloc_graph(alloc, gf) && ggml_backend_graph_compute(cpu, gf) == GGML_STATUS_SUCCESS;
    auto compare = [&]() {
        const auto a = get(ref), b = get(got);
        ok &= all_finite(a) && all_finite(b) && first_bit_difference(a, b) < 0;
        return b;
    };
    const auto original = compare();
    for (int64_t i = 0; i < d*nh; ++i) { ok &= original[i] == 0.0f; }
    auto changed_ids = iv;
    auto changed_mask = mv;
    changed_ids[0] = nk - 1;
    changed_mask[nk - 1] = ggml_fp32_to_fp16(0.0f);
    changed_ids[bound] = nk - 2;
    changed_mask[nk] = ggml_fp32_to_fp16(-INFINITY);
    changed_mask[2*nk - 2] = ggml_fp32_to_fp16(0.0f);
    ggml_backend_tensor_set(ids, changed_ids.data(), 0, ggml_nbytes(ids));
    ggml_backend_tensor_set(mask, changed_mask.data(), 0, ggml_nbytes(mask));
    ok &= ggml_backend_graph_compute(cpu, gf) == GGML_STATUS_SUCCESS;
    ok &= first_bit_difference(original, compare()) >= 0;
    ggml_backend_tensor_set(ids, iv.data(), 0, ggml_nbytes(ids));
    ggml_backend_tensor_set(mask, mv.data(), 0, ggml_nbytes(mask));
    ok &= ggml_backend_graph_compute(cpu, gf) == GGML_STATUS_SUCCESS;
    ok &= first_bit_difference(original, compare()) < 0;
    ggml_gallocr_free(alloc);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    printf("CPU %s direct attention equals dense zero-bias mask, empty rows included: %s\n", ggml_type_name(type), ok ? "ok" : "FAILED");
    return ok;
}

int main(int argc, char ** argv) {
    bool direct_only = false, direct_longctx = false, require_cuda = false, cpu_only = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--direct-only") == 0) { direct_only = true; }
        else if (strcmp(argv[i], "--direct-longctx") == 0) { direct_longctx = true; }
        else if (strcmp(argv[i], "--require-cuda") == 0) { require_cuda = true; }
        else if (strcmp(argv[i], "--cpu-only") == 0) { cpu_only = true; }
        else {
            fprintf(stderr, "unknown argument: %s\n", argv[i]);
            return 1;
        }
    }
    if ((cpu_only && (require_cuda || direct_longctx || direct_only)) || (direct_only && direct_longctx)) {
        fprintf(stderr, "incompatible test modes\n");
        return 1;
    }
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

    bool ok = nmse({0.0f}, {0.0f}) == 0.0 && std::isinf(nmse({1.0f}, {0.0f}));
    for (ggml_type type : {GGML_TYPE_F16, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0}) { ok &= run_direct_cpu(cpu, type); }
    if (cpu_only) {
        ggml_backend_free(cpu);
        return ok ? 0 : 1;
    }
    int n_tested = 0, longctx_passed = 0;
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
        if (backend == nullptr) {
            printf("%s: backend initialization FAILED\n", name);
            ok = false;
            continue;
        }
        for (ggml_type type : { GGML_TYPE_Q8_0, GGML_TYPE_Q4_0 }) {
            if (direct_longctx) {
                auto run_longctx = [&](int64_t nb, rows_kind rows, bool empty) {
                    const fattn_case c = {400128, nb, 32, 1, rows};
                    const bool passed = run_case(backend, cpu, c, type, reads_in_place,
                        0x400128u + uint32_t(nb) + (type == GGML_TYPE_Q4_0 ? 0x100u : 0u), true, empty, true);
                    longctx_passed += passed ? 1 : 0;
                    ok &= passed;
                };
                for (int64_t nb : {1, 2, 3}) {
                    for (rows_kind rows : {ROWS_NEAR_BOUND, ROWS_FEW}) { run_longctx(nb, rows, false); }
                }
                for (int64_t nb : {1, 3}) { run_longctx(nb, ROWS_FEW, true); }
                continue;
            }
            for (size_t ic = 0; !direct_only && ic < cases.size(); ++ic) {
                ok &= run_case(backend, cpu, cases[ic], type, reads_in_place,
                               0x51A7u + uint32_t(ic) + (type == GGML_TYPE_Q4_0 ? 0x100u : 0u), false, true, require_cuda);
            }
            for (const fattn_case & c : std::vector<fattn_case>{
                    {3840, 3, 32, 1, ROWS_FEW}, {16640, 3, 64, 2, ROWS_FEW},
                    {16640, 64, 32, 1, ROWS_FIRST}, {16640, 512, 64, 1, ROWS_FEW}}) {
                ok &= run_case(backend, cpu, c, type, reads_in_place, 0xD1EC7u, true, true, require_cuda);
            }
        }
        ggml_backend_free(backend);
    }
    ggml_backend_free(cpu);

    if (n_tested == 0) {
        printf("no CUDA device, skipped\n");
        return ok && !require_cuda && !direct_longctx ? 0 : 1;
    }
    if (direct_longctx) {
        ok &= longctx_passed == 16*n_tested;
        printf("%d/%d long-context direct attention cases passed on %d CUDA device%s%s\n",
               longctx_passed, 16*n_tested, n_tested, n_tested == 1 ? "" : "s", ok ? "" : " FAILED");
    }
    printf("%s\n", ok ? "OK" : "FAILED");
    return ok ? 0 : 1;
}
