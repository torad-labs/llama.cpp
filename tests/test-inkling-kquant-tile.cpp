// Focused parity tests for Blackwell K-quant staged MMQ (kquant-staged-kernel) and 23-row TMA MoE decode.
// Default mode runs Q3_K/Q4_K/Q5_K MMQ across Inkling prefill widths, fallback rows, dense Stream-K, and fused GLU,
// checking CPU NMSE < 5e-4 and dumping output bytes for bit-exact staged-vs-legacy comparison.
// --tma-23 mode checks 23-row Q3_K/Q4_K/Q5_K TMA decode across 0/3/8 repeated experts and 1024/1536/4608 shards.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

static float noise_from(uint32_t & state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return (float) (state & 0xffffff) / (float) 0x1000000 - 0.5f;
}

static double compute_nmse(const std::vector<float> & ref, const std::vector<float> & got) {
    double err = 0.0;
    double pwr = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        if (!std::isfinite(got[i])) {
            return 1.0;
        }
        const double d = (double) ref[i] - (double) got[i];
        err += d * d;
        pwr += (double) ref[i] * (double) ref[i];
    }
    return pwr > 0.0 ? err / pwr : (err > 0.0 ? 1.0 : 0.0);
}

// Quantize in bounded host chunks to avoid full f32 weight allocations.
static void quantize_weight_pair(ggml_tensor * w_gpu, ggml_tensor * w_cpu, ggml_type type, int64_t per, uint32_t seed, int n_threads) {
    const int64_t nrow       = ggml_nrows(w_gpu);
    const size_t  row_bytes  = ggml_row_size(type, per);
    const int64_t chunk_rows = std::max<int64_t>(1, (4 << 20) / per);
    std::vector<char> qbuf((size_t) std::min(nrow, chunk_rows) * row_bytes);

    for (int64_t r0 = 0; r0 < nrow; r0 += chunk_rows) {
        const int64_t nr = std::min(chunk_rows, nrow - r0);
        const int nt = std::min<int>(n_threads, (int) nr);
        std::vector<std::thread> workers;
        workers.reserve(nt);
        for (int t = 0; t < nt; ++t) {
            const int64_t a = r0 + (nr * t) / nt;
            const int64_t b = r0 + (nr * (t + 1)) / nt;
            workers.emplace_back([&, a, b]() {
                std::vector<float> src(per);
                for (int64_t r = a; r < b; ++r) {
                    uint32_t st = (seed ^ uint32_t(r * 2654435761u)) | 1u;
                    for (int64_t i = 0; i < per; ++i) {
                        src[i] = noise_from(st);
                    }
                    const size_t written = ggml_quantize_chunk(type, src.data(),
                        qbuf.data() + (size_t) (r - r0) * row_bytes, 0, 1, per, nullptr);
                    GGML_ASSERT(written == row_bytes);
                }
            });
        }
        for (auto & w : workers) {
            w.join();
        }
        const size_t off   = (size_t) r0 * row_bytes;
        const size_t bytes = (size_t) nr * row_bytes;
        ggml_backend_tensor_set(w_gpu, qbuf.data(), off, bytes);
        if (w_cpu) {
            ggml_backend_tensor_set(w_cpu, qbuf.data(), off, bytes);
        }
    }
}

static void fill_f32_pair(ggml_tensor * t_gpu, ggml_tensor * t_cpu, uint32_t seed) {
    const int64_t chunk = 1 << 20;
    std::vector<float> buf(chunk);
    const size_t nels = (size_t) ggml_nelements(t_gpu);
    uint32_t st = seed | 1u;
    for (size_t off = 0; off < nels; off += chunk) {
        const size_t n = std::min<size_t>(chunk, nels - off);
        for (size_t i = 0; i < n; ++i) {
            buf[i] = noise_from(st);
        }
        ggml_backend_tensor_set(t_gpu, buf.data(), off * sizeof(float), n * sizeof(float));
        if (t_cpu) {
            ggml_backend_tensor_set(t_cpu, buf.data(), off * sizeof(float), n * sizeof(float));
        }
    }
}

struct mmid_case {
    ggml_type    type;
    int64_t      k;
    int64_t      m;
    int64_t      n_local;
    int32_t      first;
    int64_t      n_total;
    int64_t      n_used;
    int64_t      n_tok;
    bool         bcast;
    bool         check_cpu;
    const char * label;
};

static bool run_mmid_case(ggml_backend_t cuda, ggml_backend_t cpu, const mmid_case & c,
                          int n_threads, FILE * dump_fp, uint32_t seed) {
    ggml_init_params params  = { 16 * ggml_tensor_overhead(), nullptr, true };
    ggml_init_params gparams = { ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx_g  = ggml_init(params);
    ggml_context * gctx_g = ggml_init(gparams);
    ggml_context * ctx_c  = c.check_cpu ? ggml_init(params)  : nullptr;
    ggml_context * gctx_c = c.check_cpu ? ggml_init(gparams) : nullptr;

    const int64_t b_cols = c.bcast ? 1 : c.n_used;
    ggml_tensor * as_g  = ggml_new_tensor_3d(ctx_g, c.type, c.k, c.m, c.n_local);
    ggml_tensor * ids_g = ggml_new_tensor_2d(ctx_g, GGML_TYPE_I32, c.n_used, c.n_tok);
    ggml_tensor * b_g   = ggml_new_tensor_3d(ctx_g, GGML_TYPE_F32, c.k, b_cols, c.n_tok);
    ggml_tensor * out_g = ggml_mul_mat_id(ctx_g, as_g, b_g, ids_g);
    if (c.n_total > c.n_local) {
        ggml_mul_mat_id_set_expert_range(out_g, c.first, (int32_t) c.n_total);
    }
    ggml_cgraph * gf_g = ggml_new_graph(gctx_g);
    ggml_build_forward_expand(gf_g, out_g);

    ggml_tensor * as_c = nullptr, * ids_c = nullptr, * b_c = nullptr, * out_c = nullptr;
    ggml_cgraph * gf_c = nullptr;
    if (c.check_cpu) {
        as_c  = ggml_new_tensor_3d(ctx_c, c.type, c.k, c.m, c.n_local);
        ids_c = ggml_new_tensor_2d(ctx_c, GGML_TYPE_I32, c.n_used, c.n_tok);
        b_c   = ggml_new_tensor_3d(ctx_c, GGML_TYPE_F32, c.k, b_cols, c.n_tok);
        out_c = ggml_mul_mat_id(ctx_c, as_c, b_c, ids_c);
        if (c.n_total > c.n_local) {
            ggml_mul_mat_id_set_expert_range(out_c, c.first, (int32_t) c.n_total);
        }
        gf_c = ggml_new_graph(gctx_c);
        ggml_build_forward_expand(gf_c, out_c);
    }

    ggml_backend_buffer_t buf_g = ggml_backend_alloc_ctx_tensors(ctx_g, cuda);
    ggml_backend_buffer_t buf_c = c.check_cpu ? ggml_backend_alloc_ctx_tensors(ctx_c, cpu) : nullptr;
    bool ok = buf_g != nullptr && (!c.check_cpu || buf_c != nullptr);

    if (ok) {
        quantize_weight_pair(as_g, as_c, c.type, c.k, seed, n_threads);
        fill_f32_pair(b_g, b_c, seed ^ 0x55aa55aau);

        uint32_t st = (seed ^ 0x13579bdfu) | 1u;
        std::vector<int32_t> ids(c.n_used * c.n_tok);
        for (int64_t t = 0; t < c.n_tok; ++t) {
            for (int64_t u = 0; u < c.n_used; ++u) {
                int32_t cand = 0;
                bool dup = false;
                do {
                    noise_from(st);
                    if (c.n_total > c.n_local && u < 2) {
                        cand = c.first + (int32_t) (st % (uint32_t) c.n_local);
                    } else {
                        cand = (int32_t) (st % (uint32_t) c.n_total);
                    }
                    dup = false;
                    for (int64_t p = 0; p < u; ++p) {
                        if (ids[t * c.n_used + p] == cand) {
                            dup = true;
                            break;
                        }
                    }
                } while (dup);
                ids[t * c.n_used + u] = cand;
            }
        }
        ggml_backend_tensor_set(ids_g, ids.data(), 0, ids.size() * sizeof(int32_t));
        if (ids_c) {
            ggml_backend_tensor_set(ids_c, ids.data(), 0, ids.size() * sizeof(int32_t));
        }

        ok = ggml_backend_graph_compute(cuda, gf_g) == GGML_STATUS_SUCCESS;
        if (ok && c.check_cpu) {
            ok = ggml_backend_graph_compute(cpu, gf_c) == GGML_STATUS_SUCCESS;
        }
        if (ok) {
            std::vector<float> gpu(ggml_nelements(out_g));
            ggml_backend_tensor_get(out_g, gpu.data(), 0, gpu.size() * sizeof(float));
            if (dump_fp) {
                fwrite(gpu.data(), sizeof(float), gpu.size(), dump_fp);
            }
            if (c.check_cpu) {
                std::vector<float> ref(ggml_nelements(out_c));
                ggml_backend_tensor_get(out_c, ref.data(), 0, ref.size() * sizeof(float));
                const double err = compute_nmse(ref, gpu);
                ok = err < 5e-4;
                printf("%-44s nmse %.3e  %s\n", c.label, err, ok ? "ok" : "FAIL");
            } else {
                for (float v : gpu) {
                    if (!std::isfinite(v)) {
                        ok = false;
                        break;
                    }
                }
                printf("%-44s finite %s\n", c.label, ok ? "ok" : "FAIL");
            }
        }
    }

    if (buf_g) { ggml_backend_buffer_free(buf_g); }
    if (buf_c) { ggml_backend_buffer_free(buf_c); }
    ggml_free(gctx_g);
    ggml_free(ctx_g);
    if (gctx_c) { ggml_free(gctx_c); }
    if (ctx_c)  { ggml_free(ctx_c); }
    return ok;
}

static bool run_dense_and_fusion_case(ggml_backend_t cuda, ggml_backend_t cpu, ggml_type type,
                                      int64_t k, int64_t m, int64_t n, bool fused_glu,
                                      int n_threads, FILE * dump_fp, uint32_t seed, const char * label) {
    ggml_init_params params  = { 16 * ggml_tensor_overhead(), nullptr, true };
    ggml_init_params gparams = { ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx_g  = ggml_init(params);
    ggml_context * ctx_c  = ggml_init(params);
    ggml_context * gctx_g = ggml_init(gparams);
    ggml_context * gctx_c = ggml_init(gparams);

    ggml_tensor * w0_g = ggml_new_tensor_2d(ctx_g, type, k, m);
    ggml_tensor * w1_g = fused_glu ? ggml_new_tensor_2d(ctx_g, type, k, m) : nullptr;
    ggml_tensor * x_g  = ggml_new_tensor_2d(ctx_g, GGML_TYPE_F32, k, n);
    ggml_tensor * out_g = fused_glu
        ? ggml_swiglu_split(ctx_g, ggml_mul_mat(ctx_g, w1_g, x_g), ggml_mul_mat(ctx_g, w0_g, x_g))
        : ggml_mul_mat(ctx_g, w0_g, x_g);

    ggml_tensor * w0_c = ggml_new_tensor_2d(ctx_c, type, k, m);
    ggml_tensor * w1_c = fused_glu ? ggml_new_tensor_2d(ctx_c, type, k, m) : nullptr;
    ggml_tensor * x_c  = ggml_new_tensor_2d(ctx_c, GGML_TYPE_F32, k, n);
    ggml_tensor * out_c = fused_glu
        ? ggml_swiglu_split(ctx_c, ggml_mul_mat(ctx_c, w1_c, x_c), ggml_mul_mat(ctx_c, w0_c, x_c))
        : ggml_mul_mat(ctx_c, w0_c, x_c);

    ggml_cgraph * gf_g = ggml_new_graph(gctx_g);
    ggml_cgraph * gf_c = ggml_new_graph(gctx_c);
    ggml_build_forward_expand(gf_g, out_g);
    ggml_build_forward_expand(gf_c, out_c);

    ggml_backend_buffer_t buf_g = ggml_backend_alloc_ctx_tensors(ctx_g, cuda);
    ggml_backend_buffer_t buf_c = ggml_backend_alloc_ctx_tensors(ctx_c, cpu);
    bool ok = buf_g != nullptr && buf_c != nullptr;

    if (ok) {
        quantize_weight_pair(w0_g, w0_c, type, k, seed, n_threads);
        if (fused_glu) {
            quantize_weight_pair(w1_g, w1_c, type, k, seed ^ 0x7777u, n_threads);
        }
        fill_f32_pair(x_g, x_c, seed ^ 0x3333u);

        ok = ggml_backend_graph_compute(cuda, gf_g) == GGML_STATUS_SUCCESS &&
             ggml_backend_graph_compute(cpu,  gf_c) == GGML_STATUS_SUCCESS;
        if (ok) {
            std::vector<float> gpu(ggml_nelements(out_g));
            std::vector<float> ref(ggml_nelements(out_c));
            ggml_backend_tensor_get(out_g, gpu.data(), 0, gpu.size() * sizeof(float));
            ggml_backend_tensor_get(out_c, ref.data(), 0, ref.size() * sizeof(float));
            if (dump_fp) {
                fwrite(gpu.data(), sizeof(float), gpu.size(), dump_fp);
            }
            const double err = compute_nmse(ref, gpu);
            ok = err < 5e-4;
            printf("%-44s nmse %.3e  %s\n", label, err, ok ? "ok" : "FAIL");
        }
    }

    if (buf_g) { ggml_backend_buffer_free(buf_g); }
    if (buf_c) { ggml_backend_buffer_free(buf_c); }
    ggml_free(gctx_g);
    ggml_free(gctx_c);
    ggml_free(ctx_g);
    ggml_free(ctx_c);
    return ok;
}

// 23-row TMA MoE parity: 0, 3, 8 repeated experts and skewed routes across 1024/1536/4608 shards for Q3_K/Q4_K/Q5_K.
static int run_tma_23_suite(ggml_backend_t cuda, ggml_backend_t cpu, int n_threads) {
    struct tma_cfg {
        ggml_type    type;
        int64_t      k;
        int64_t      m;
        int64_t      n_local;
        int32_t      first;
        int64_t      n_total;
        bool         bcast;
        const char * label;
    };
    const tma_cfg cfgs[] = {
        { GGML_TYPE_Q3_K, 4608, 4608,  96,  96, 384, true,  "tma23 q3_K 96/384 m=4608"  },
        { GGML_TYPE_Q4_K, 2304, 4608,  96,  96, 384, false, "tma23 q4_K 96/384 k=2304"  },
        { GGML_TYPE_Q5_K, 2304, 4608,  96,  96, 384, false, "tma23 q5_K 96/384 k=2304"  },
        { GGML_TYPE_Q3_K, 4608, 1024, 384,   0, 384, true,  "tma23 q3_K 384/384 m=1024" },
        { GGML_TYPE_Q3_K, 4608, 1536, 384,   0, 384, true,  "tma23 q3_K 384/384 m=1536" },
        { GGML_TYPE_Q4_K, 1024, 4608, 384,   0, 384, false, "tma23 q4_K 384/384 k=1024" },
        { GGML_TYPE_Q4_K, 1536, 4608, 384,   0, 384, false, "tma23 q4_K 384/384 k=1536" },
        { GGML_TYPE_Q5_K, 1024, 4608, 384,   0, 384, false, "tma23 q5_K 384/384 k=1024" },
        { GGML_TYPE_Q5_K, 1536, 4608, 384,   0, 384, false, "tma23 q5_K 384/384 k=1536" },
    };
    static constexpr int64_t n_tok  = 23;
    static constexpr int64_t n_used = 8;

    int failures = 0;
    for (const tma_cfg & cfg : cfgs) {
        ggml_init_params wparams = { 4 * ggml_tensor_overhead(), nullptr, true };
        ggml_context * wctx_g = ggml_init(wparams);
        ggml_context * wctx_c = ggml_init(wparams);
        ggml_tensor * as_g = ggml_new_tensor_3d(wctx_g, cfg.type, cfg.k, cfg.m, cfg.n_local);
        ggml_tensor * as_c = ggml_new_tensor_3d(wctx_c, cfg.type, cfg.k, cfg.m, cfg.n_local);
        ggml_backend_buffer_t wbuf_g = ggml_backend_alloc_ctx_tensors(wctx_g, cuda);
        ggml_backend_buffer_t wbuf_c = ggml_backend_alloc_ctx_tensors(wctx_c, cpu);
        if (!wbuf_g || !wbuf_c) {
            printf("%-32s FAIL: alloc failed\n", cfg.label);
            if (wbuf_g) { ggml_backend_buffer_free(wbuf_g); }
            if (wbuf_c) { ggml_backend_buffer_free(wbuf_c); }
            ggml_free(wctx_g);
            ggml_free(wctx_c);
            ++failures;
            continue;
        }
        quantize_weight_pair(as_g, as_c, cfg.type, cfg.k, 0x23a00001u + uint32_t(cfg.k + cfg.m), n_threads);

        // Route patterns: 0=zero owned (when partitioned), 1=3 repeated experts, 2=8 repeated experts, 3=mixed.
        const int start_mode = cfg.n_total > cfg.n_local ? 0 : 1;
        double worst_nmse = 0.0;
        bool cfg_ok = true;

        for (int mode = start_mode; mode <= 3; ++mode) {
            ggml_init_params params  = { 12 * ggml_tensor_overhead(), nullptr, true };
            ggml_init_params gparams = { ggml_graph_overhead(), nullptr, true };
            ggml_context * ctx_g  = ggml_init(params);
            ggml_context * ctx_c  = ggml_init(params);
            ggml_context * gctx_g = ggml_init(gparams);
            ggml_context * gctx_c = ggml_init(gparams);

            const int64_t b_cols = cfg.bcast ? 1 : n_used;
            ggml_tensor * ids_g = ggml_new_tensor_2d(ctx_g, GGML_TYPE_I32, n_used, n_tok);
            ggml_tensor * b_g   = ggml_new_tensor_3d(ctx_g, GGML_TYPE_F32, cfg.k, b_cols, n_tok);
            ggml_tensor * out_g = ggml_mul_mat_id(ctx_g, as_g, b_g, ids_g);
            ggml_tensor * ids_c = ggml_new_tensor_2d(ctx_c, GGML_TYPE_I32, n_used, n_tok);
            ggml_tensor * b_c   = ggml_new_tensor_3d(ctx_c, GGML_TYPE_F32, cfg.k, b_cols, n_tok);
            ggml_tensor * out_c = ggml_mul_mat_id(ctx_c, as_c, b_c, ids_c);
            if (cfg.n_total > cfg.n_local) {
                ggml_mul_mat_id_set_expert_range(out_g, cfg.first, (int32_t) cfg.n_total);
                ggml_mul_mat_id_set_expert_range(out_c, cfg.first, (int32_t) cfg.n_total);
            }

            ggml_cgraph * gf_g = ggml_new_graph(gctx_g);
            ggml_cgraph * gf_c = ggml_new_graph(gctx_c);
            ggml_build_forward_expand(gf_g, out_g);
            ggml_build_forward_expand(gf_c, out_c);

            ggml_backend_buffer_t buf_g = ggml_backend_alloc_ctx_tensors(ctx_g, cuda);
            ggml_backend_buffer_t buf_c = ggml_backend_alloc_ctx_tensors(ctx_c, cpu);
            bool ok = buf_g != nullptr && buf_c != nullptr;

            if (ok) {
                std::vector<int32_t> ids(n_used * n_tok);
                uint32_t st = uint32_t(0x900d0001u + mode * 131u) | 1u;
                for (int64_t t = 0; t < n_tok; ++t) {
                    if (mode == 0) {
                        // All 8 slots non-owned (below and above [first, first + n_local)).
                        for (int64_t u = 0; u < n_used; ++u) {
                            ids[t * n_used + u] = u < 4 ? int32_t((t + u) % cfg.first)
                                                        : int32_t(cfg.first + cfg.n_local + ((t + u) % 64));
                        }
                    } else if (mode == 1) {
                        // Exactly 3 shared local experts repeated across all 23 tokens.
                        ids[t * n_used + 0] = cfg.first + 0;
                        ids[t * n_used + 1] = cfg.first + (int32_t) (cfg.n_local / 2);
                        ids[t * n_used + 2] = cfg.first + (int32_t) (cfg.n_local - 1);
                        for (int64_t u = 3; u < n_used; ++u) {
                            if (cfg.n_total > cfg.n_local) {
                                ids[t * n_used + u] = int32_t((t * 5 + u) % cfg.first);
                            } else {
                                ids[t * n_used + u] = int32_t((3 + t * 5 + u) % (cfg.n_local - 1));
                                if (ids[t * n_used + u] == (int32_t) (cfg.n_local / 2)) {
                                    ids[t * n_used + u] += 1;
                                }
                            }
                        }
                    } else if (mode == 2) {
                        // All 8 experts repeated across all 23 tokens (max ncols = 23 per expert).
                        for (int64_t u = 0; u < n_used; ++u) {
                            ids[t * n_used + u] = cfg.first + (int32_t) (u * 11 % cfg.n_local);
                        }
                    } else {
                        for (int64_t u = 0; u < n_used; ++u) {
                            int32_t cand = 0;
                            bool dup = false;
                            do {
                                noise_from(st);
                                cand = (int32_t) (st % (uint32_t) cfg.n_total);
                                dup = false;
                                for (int64_t p = 0; p < u; ++p) {
                                    if (ids[t * n_used + p] == cand) {
                                        dup = true;
                                        break;
                                    }
                                }
                            } while (dup);
                            ids[t * n_used + u] = cand;
                        }
                    }
                }
                ggml_backend_tensor_set(ids_g, ids.data(), 0, ids.size() * sizeof(int32_t));
                ggml_backend_tensor_set(ids_c, ids.data(), 0, ids.size() * sizeof(int32_t));
                fill_f32_pair(b_g, b_c, 0x11223344u + mode);

                std::vector<char> sentinel(ggml_nbytes(out_g), (char) 0xA5);
                ggml_backend_tensor_set(out_g, sentinel.data(), 0, sentinel.size());

                ok = ggml_backend_graph_compute(cuda, gf_g) == GGML_STATUS_SUCCESS &&
                     ggml_backend_graph_compute(cpu,  gf_c) == GGML_STATUS_SUCCESS;
                if (ok) {
                    std::vector<float> gpu(ggml_nelements(out_g));
                    std::vector<float> ref(ggml_nelements(out_c));
                    ggml_backend_tensor_get(out_g, gpu.data(), 0, gpu.size() * sizeof(float));
                    ggml_backend_tensor_get(out_c, ref.data(), 0, ref.size() * sizeof(float));
                    for (int64_t t = 0; t < n_tok && ok; ++t) {
                        for (int64_t u = 0; u < n_used && ok; ++u) {
                            const int32_t e = ids[t * n_used + u];
                            if (e < cfg.first || e >= cfg.first + (int32_t) cfg.n_local) {
                                const float * row = gpu.data() + (t * n_used + u) * cfg.m;
                                for (int64_t col = 0; col < cfg.m; ++col) {
                                    if (row[col] != 0.0f) {
                                        printf("%s mode=%d FAIL: non-owned (%lld,%lld) e=%d val=%g\n",
                                            cfg.label, mode, (long long) t, (long long) u, (int) e, (double) row[col]);
                                        ok = false;
                                        break;
                                    }
                                }
                            }
                        }
                    }
                    const double err = compute_nmse(ref, gpu);
                    worst_nmse = std::max(worst_nmse, err);
                    if (err >= 5e-4) {
                        printf("%s mode=%d FAIL: nmse %.3e\n", cfg.label, mode, err);
                        ok = false;
                    }
                }
            }

            if (buf_g) { ggml_backend_buffer_free(buf_g); }
            if (buf_c) { ggml_backend_buffer_free(buf_c); }
            ggml_free(gctx_g);
            ggml_free(gctx_c);
            ggml_free(ctx_g);
            ggml_free(ctx_c);
            if (!ok) {
                cfg_ok = false;
                break;
            }
        }
        printf("%-32s modes=%d..3 worst nmse %.3e  %s\n", cfg.label, start_mode, worst_nmse, cfg_ok ? "ok" : "FAIL");
        if (!cfg_ok) {
            ++failures;
        }
        ggml_backend_buffer_free(wbuf_g);
        ggml_backend_buffer_free(wbuf_c);
        ggml_free(wctx_g);
        ggml_free(wctx_c);
    }
    return failures;
}

int main(int argc, char ** argv) {
    const bool tma_23 = argc > 1 && strcmp(argv[1], "--tma-23") == 0;
    const char * dump_path = (!tma_23 && argc > 1) ? argv[1] : nullptr;

    ggml_backend_t cuda = ggml_backend_cuda_init(0);
    if (cuda == nullptr) {
        if (dump_path) {
            FILE * fp = fopen(dump_path, "wb");
            if (fp) { fclose(fp); }
        }
        printf("test-inkling-kquant-tile: no CUDA device, skipped\n");
        return 0;
    }

    const int n_threads = std::min<int>(16, std::max<int>(1, (int) std::thread::hardware_concurrency()));
    ggml_backend_t cpu = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(cpu, n_threads);

    if (tma_23) {
        const int failures = run_tma_23_suite(cuda, cpu, n_threads);
        ggml_backend_free(cpu);
        ggml_backend_free(cuda);
        return failures == 0 ? 0 : 1;
    }

    FILE * dump_fp = dump_path ? fopen(dump_path, "wb") : nullptr;
    if (dump_path && !dump_fp) {
        fprintf(stderr, "failed to open dump file: %s\n", dump_path);
        ggml_backend_free(cpu);
        ggml_backend_free(cuda);
        return 1;
    }

    const mmid_case mmid_cases[] = {
        // Served Inkling prefill widths (n_total=384, n_used=8): J=16 (n=512), J=32 (n=1024), J=56 (n=2048 via small local).
        { GGML_TYPE_Q3_K, 4608, 1024, 32, 96, 384, 8,  512, true,  true,  "mmid q3_K m=1024 n=512  (J=16)" },
        { GGML_TYPE_Q3_K, 4608, 1536, 32, 96, 384, 8, 1024, true,  true,  "mmid q3_K m=1536 n=1024 (J=32)" },
        { GGML_TYPE_Q4_K, 2304, 1024, 32, 96, 384, 8,  512, false, true,  "mmid q4_K m=1024 n=512  (J=16)" },
        { GGML_TYPE_Q4_K, 2304, 1536, 32, 96, 384, 8, 1024, false, true,  "mmid q4_K m=1536 n=1024 (J=32)" },
        { GGML_TYPE_Q5_K, 2304, 1024, 32, 96, 384, 8,  512, false, true,  "mmid q5_K m=1024 n=512  (J=16)" },
        { GGML_TYPE_Q5_K, 2304, 1536, 32, 96, 384, 8, 1024, false, true,  "mmid q5_K m=1536 n=1024 (J=32)" },
        // Wider Blackwell J=48 and J=56 tiles (n_total=32, n_used=8 -> mean=32..40).
        { GGML_TYPE_Q3_K, 1024,  512, 32,  0,  32, 8,  160, true,  true,  "mmid q3_K wide (J=56)" },
        { GGML_TYPE_Q4_K, 1024,  512, 32,  0,  32, 8,  160, false, true,  "mmid q4_K wide (J=56)" },
        { GGML_TYPE_Q5_K, 1024,  512, 32,  0,  32, 8,  160, false, true,  "mmid q5_K wide (J=56)" },
        // Fallback rows (nrows_x % 128 != 0).
        { GGML_TYPE_Q3_K, 1024,  192, 16,  0,  16, 4,   64, true,  true,  "mmid q3_K fallback m=192" },
        { GGML_TYPE_Q4_K, 1024,  192, 16,  0,  16, 4,   64, false, true,  "mmid q4_K fallback m=192" },
        { GGML_TYPE_Q5_K, 1024,  192, 16,  0,  16, 4,   64, false, true,  "mmid q5_K fallback m=192" },
    };

    int failures = 0;
    uint32_t seed = 0x51a9ed01u;
    for (const mmid_case & c : mmid_cases) {
        if (!run_mmid_case(cuda, cpu, c, n_threads, dump_fp, seed)) {
            ++failures;
        }
        seed += 17u;
    }

    // Dense Stream-K and fused gate/up/GLU coverage for staged K-quant paths.
    if (!run_dense_and_fusion_case(cuda, cpu, GGML_TYPE_Q3_K, 2304, 512, 32, false, n_threads, dump_fp, seed + 1, "dense q3_K n=32")) { ++failures; }
    if (!run_dense_and_fusion_case(cuda, cpu, GGML_TYPE_Q4_K, 2304, 512, 32, false, n_threads, dump_fp, seed + 2, "dense q4_K n=32")) { ++failures; }
    if (!run_dense_and_fusion_case(cuda, cpu, GGML_TYPE_Q5_K, 2304, 512, 32, false, n_threads, dump_fp, seed + 3, "dense q5_K n=32")) { ++failures; }
    if (!run_dense_and_fusion_case(cuda, cpu, GGML_TYPE_Q4_K, 1024, 512, 64, true,  n_threads, dump_fp, seed + 4, "fused glu q4_K n=64")) { ++failures; }
    if (!run_dense_and_fusion_case(cuda, cpu, GGML_TYPE_Q5_K, 1024, 512, 64, true,  n_threads, dump_fp, seed + 5, "fused glu q5_K n=64")) { ++failures; }

    if (dump_fp) {
        fclose(dump_fp);
    }
    ggml_backend_free(cpu);
    ggml_backend_free(cuda);
    return failures == 0 ? 0 : 1;
}
