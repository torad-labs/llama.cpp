// Focused parity tests for Blackwell K-quant staged MMQ (kquant-staged-kernel) and 23-row TMA MoE decode.
// Default mode runs Q3_K/Q4_K/Q5_K MMQ across Inkling prefill widths, fallback rows, dense Stream-K, and fused GLU,
// checking CPU NMSE < 5e-4 and dumping output bytes for bit-exact staged-vs-legacy comparison.
// --tma-23 mode checks 23-row Q3_K/Q4_K/Q5_K TMA decode across 0/3/8 repeated experts and 1024/1536/4608 shards.
// --rotated mode checks one card's banks of the rotated expert split (gate/up N=1024 and 512, down K=512 and 256) at
// 23 and 512 rows, and the fused TMA gate/up + SITU and down + weighted sum with a strided bonus expert range.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"

#ifdef GGML_USE_CUDA
#include "../ggml/src/ggml-backend-impl.h"
#include "../ggml/src/ggml-cuda/mmq-tma-tiles.cuh"
#endif

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
        // Run Q5_K (58.9 KiB SMEM) before Q4_K (49.7 KiB SMEM) at k=2304 to verify per-kernel SMEM opt-in.
        { GGML_TYPE_Q5_K, 2304, 4608,  96,  96, 384, false, "tma23 q5_K 96/384 k=2304"  },
        { GGML_TYPE_Q4_K, 2304, 4608,  96,  96, 384, false, "tma23 q4_K 96/384 k=2304"  },
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

// One card c of the rotated expert split (LLAMA_INKLING2_EXPERT_LAYOUT=rotated): the base banks hold all 384 experts
// at K [512c, 512c + 512) of the down and N 1024 of gate/up, the bonus banks the 96 experts e % 4 == c at K 256 of the
// down and N 512 of gate/up as local experts (e - c)/4. Each against the CPU on the same bytes.
static constexpr int ROT_TOTAL = 384;
static constexpr int ROT_CARDS = 4;
static constexpr int ROT_CARD  = 1;

static void rot_routes(std::vector<int32_t> & ids, int64_t n_tok, int64_t n_used, uint32_t seed) {
    ids.resize(n_used*n_tok);
    uint32_t st = seed | 1u;
    for (int64_t t = 0; t < n_tok; ++t) {
        for (int64_t u = 0; u < n_used; ++u) {
            int32_t cand = 0;
            bool dup = true;
            while (dup) {
                noise_from(st);
                cand = (int32_t) (st % ROT_TOTAL);
                dup = false;
                for (int64_t p = 0; p < u; ++p) {
                    dup = dup || ids[t*n_used + p] == cand;
                }
            }
            ids[t*n_used + u] = cand;
        }
    }
}

struct rot_case {
    ggml_type    type;
    int64_t      k;
    int64_t      m;
    bool         bonus;
    bool         bcast;
    const char * label;
};

static bool run_rot_case(ggml_backend_t cuda, ggml_backend_t cpu, const rot_case & c, int64_t n_tok, int n_threads,
                         uint32_t seed, int64_t & owned, int iters) {
    constexpr int64_t n_used = 8;
    const int64_t n_local = c.bonus ? ROT_TOTAL/ROT_CARDS : ROT_TOTAL;
    ggml_init_params params  = { 8*ggml_tensor_overhead(), nullptr, true };
    ggml_init_params gparams = { ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx[2]  = { ggml_init(params),  ggml_init(params) };
    ggml_context * gctx[2] = { ggml_init(gparams), ggml_init(gparams) };
    ggml_tensor * as[2], * ids_t[2], * b[2], * out[2];
    ggml_cgraph * gf[2];
    for (int i = 0; i < 2; ++i) {
        as[i]    = ggml_new_tensor_3d(ctx[i], c.type, c.k, c.m, n_local);
        ids_t[i] = ggml_new_tensor_2d(ctx[i], GGML_TYPE_I32, n_used, n_tok);
        b[i]     = ggml_new_tensor_3d(ctx[i], GGML_TYPE_F32, c.k, c.bcast ? 1 : n_used, n_tok);
        out[i]   = ggml_mul_mat_id(ctx[i], as[i], b[i], ids_t[i]);
        if (c.bonus) {
            ggml_mul_mat_id_set_expert_range(out[i], 0, ROT_TOTAL); // ids [0, 96) owned, the rest zero rows
        }
        gf[i] = ggml_new_graph(gctx[i]);
        ggml_build_forward_expand(gf[i], out[i]);
    }
    ggml_backend_buffer_t buf_g = ggml_backend_alloc_ctx_tensors(ctx[0], cuda);
    ggml_backend_buffer_t buf_c = ggml_backend_alloc_ctx_tensors(ctx[1], cpu);
    bool ok = buf_g != nullptr && buf_c != nullptr;
    double err = 1.0;
    if (ok) {
        quantize_weight_pair(as[0], as[1], c.type, c.k, seed, n_threads);
        fill_f32_pair(b[0], b[1], seed ^ 0x2468ace1u);
        std::vector<int32_t> ids;
        rot_routes(ids, n_tok, n_used, seed ^ 0x0badf00du);
        owned = 0;
        for (int32_t & e : ids) {
            const bool own = !c.bonus || e % ROT_CARDS == ROT_CARD;
            owned += own;
            if (c.bonus) {
                e = own ? (e - ROT_CARD)/ROT_CARDS : (int32_t) (n_local + e % (ROT_TOTAL - n_local));
            }
        }
        for (int i = 0; i < 2; ++i) {
            ggml_backend_tensor_set(ids_t[i], ids.data(), 0, ids.size()*sizeof(int32_t));
        }
        std::vector<char> sentinel(ggml_nbytes(out[0]), (char) 0xA5);
        ggml_backend_tensor_set(out[0], sentinel.data(), 0, sentinel.size());
        ok = ggml_backend_graph_compute(cuda, gf[0]) == GGML_STATUS_SUCCESS &&
             ggml_backend_graph_compute(cpu,  gf[1]) == GGML_STATUS_SUCCESS;
        if (ok) {
            std::vector<float> gpu(ggml_nelements(out[0])), ref(ggml_nelements(out[1]));
            ggml_backend_tensor_get(out[0], gpu.data(), 0, gpu.size()*sizeof(float));
            ggml_backend_tensor_get(out[1], ref.data(), 0, ref.size()*sizeof(float));
            err = compute_nmse(ref, gpu);
            ok = err < 5e-4;
        }
    }
    double us = 0.0; // the op's wall time a call: its launches and one synchronize
    if (ok && iters > 0) {
        for (int i = 0; i < 3; ++i) {
            ggml_backend_graph_compute(cuda, gf[0]);
        }
        const int64_t t0 = ggml_time_us();
        for (int i = 0; i < iters; ++i) {
            ggml_backend_graph_compute(cuda, gf[0]);
        }
        us = (double) (ggml_time_us() - t0)/iters;
    }
    printf("%-36s n=%-4lld owned %4lld/%-4lld nmse %.3e  op %8.1f us  %s\n", c.label, (long long) n_tok,
        (long long) owned, (long long) (n_used*n_tok), err, us, ok ? "ok" : "FAIL");
    if (buf_g) { ggml_backend_buffer_free(buf_g); }
    if (buf_c) { ggml_backend_buffer_free(buf_c); }
    for (int i = 0; i < 2; ++i) {
        ggml_free(gctx[i]);
        ggml_free(ctx[i]);
    }
    return ok;
}

#ifdef GGML_USE_CUDA
// The fused TMA entry points on the card's bonus banks (96 local experts, global e = 1 + 4l), called directly with
// global ids and stride 4. 87.5% of the routed slots are other cards' experts, many of them in [1, 97), which a
// contiguous range would take as owned: their gate/up rows must be zero and their down slots must not enter the sum.
// The CPU reference runs on ids remapped here (l or -1), so the expected map does not come from the code under test.
static bool run_rot_fused(ggml_backend_t cuda, ggml_backend_t cpu, int n_threads) {
    const char * tma = getenv("GGML_CUDA_MMQ_TMA");
    if (tma == nullptr || strcmp(tma, "1") != 0) {
        printf("rotated fused bonus: skipped, GGML_CUDA_MMQ_TMA is not 1\n");
        return true;
    }
    constexpr int64_t n_tok = 23, n_used = 8, n_local = ROT_TOTAL/ROT_CARDS, k = 4608, n_ff = 256, n_embd = 4608;
    constexpr float beta = 4.0f, lb = 25.0f;
    ggml_init_params params  = { 16*ggml_tensor_overhead(), nullptr, true };
    ggml_init_params gparams = { ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx_g = ggml_init(params);
    ggml_context * ctx_c = ggml_init(params);
    ggml_context * gctx  = ggml_init(gparams);
    ggml_tensor * gu_g  = ggml_new_tensor_3d(ctx_g, GGML_TYPE_Q3_K, k, 2*n_ff, n_local);
    ggml_tensor * dn_g  = ggml_new_tensor_3d(ctx_g, GGML_TYPE_Q4_K, n_ff, n_embd, n_local);
    ggml_tensor * x_g   = ggml_new_tensor_3d(ctx_g, GGML_TYPE_F32, k, 1, n_tok);
    ggml_tensor * ids_g = ggml_new_tensor_2d(ctx_g, GGML_TYPE_I32, n_used, n_tok);
    ggml_tensor * glu_g = ggml_new_tensor_3d(ctx_g, GGML_TYPE_F32, n_ff, n_used, n_tok);
    ggml_tensor * w_g   = ggml_new_tensor_3d(ctx_g, GGML_TYPE_F32, 1, n_used, n_tok);
    ggml_tensor * out_g = ggml_new_tensor_2d(ctx_g, GGML_TYPE_F32, n_embd, n_tok);

    // CPU: the gate/up product, SITU over its halves, and the down product on the GPU's GLU output
    ggml_tensor * gu_c  = ggml_new_tensor_3d(ctx_c, GGML_TYPE_Q3_K, k, 2*n_ff, n_local);
    ggml_tensor * dn_c  = ggml_new_tensor_3d(ctx_c, GGML_TYPE_Q4_K, n_ff, n_embd, n_local);
    ggml_tensor * x_c   = ggml_new_tensor_3d(ctx_c, GGML_TYPE_F32, k, 1, n_tok);
    ggml_tensor * ids_c = ggml_new_tensor_2d(ctx_c, GGML_TYPE_I32, n_used, n_tok);
    ggml_tensor * h_c   = ggml_new_tensor_3d(ctx_c, GGML_TYPE_F32, n_ff, n_used, n_tok);
    ggml_tensor * pair  = ggml_mul_mat_id(ctx_c, gu_c, x_c, ids_c);
    ggml_tensor * glu_c = ggml_situ_split(ctx_c,
        ggml_view_3d(ctx_c, pair, n_ff, n_used, n_tok, pair->nb[1], pair->nb[2], 0),
        ggml_view_3d(ctx_c, pair, n_ff, n_used, n_tok, pair->nb[1], pair->nb[2], n_ff*sizeof(float)), beta, lb);
    ggml_tensor * dn_out = ggml_mul_mat_id(ctx_c, dn_c, h_c, ids_c);
    ggml_cgraph * gf = ggml_new_graph(gctx);
    ggml_build_forward_expand(gf, glu_c);
    ggml_build_forward_expand(gf, dn_out);

    ggml_backend_buffer_t buf_g = ggml_backend_alloc_ctx_tensors(ctx_g, cuda);
    ggml_backend_buffer_t buf_c = ggml_backend_alloc_ctx_tensors(ctx_c, cpu);
    bool ok = buf_g != nullptr && buf_c != nullptr;
    if (ok) {
        quantize_weight_pair(gu_g, gu_c, GGML_TYPE_Q3_K, k, 0x0b0e5001u, n_threads);
        quantize_weight_pair(dn_g, dn_c, GGML_TYPE_Q4_K, n_ff, 0x0b0e5002u, n_threads);
        fill_f32_pair(x_g, x_c, 0x0b0e5003u);
        // per token 1 slot on an owned expert (two of them shared by every token), the other 7 on other cards
        std::vector<int32_t> ids(n_used*n_tok);
        std::vector<float>   wts(n_used*n_tok);
        uint32_t st = 0x0b0e5004u;
        for (int64_t t = 0; t < n_tok; ++t) {
            for (int64_t u = 0; u < n_used; ++u) {
                noise_from(st);
                int32_t e;
                if (u == t % n_used) {
                    const int32_t l = t < 12 ? (t % 2 ? 95 : 0) : (int32_t) (st % n_local);
                    e = ROT_CARD + ROT_CARDS*l;
                } else {
                    e = (int32_t) (st % ROT_TOTAL);
                    if (e % ROT_CARDS == ROT_CARD) {
                        e = (e + 1) % ROT_TOTAL;
                    }
                }
                ids[t*n_used + u] = e;
                wts[t*n_used + u] = 0.25f + 0.5f*(float) (st % 1000)/1000.0f;
            }
        }
        std::vector<int32_t> lids(ids.size());
        int64_t n_trap = 0; // non-owned ids a contiguous range [1, 97) would take as owned
        for (size_t s = 0; s < ids.size(); ++s) {
            const bool own = ids[s] % ROT_CARDS == ROT_CARD;
            lids[s] = own ? (ids[s] - ROT_CARD)/ROT_CARDS : -1;
            n_trap += !own && ids[s] >= ROT_CARD && ids[s] < ROT_CARD + n_local;
        }
        ggml_backend_tensor_set(ids_g, ids.data(), 0, ids.size()*sizeof(int32_t));
        ggml_backend_tensor_set(ids_c, lids.data(), 0, lids.size()*sizeof(int32_t));
        ggml_backend_tensor_set(w_g, wts.data(), 0, wts.size()*sizeof(float));
        std::vector<char> sentinel(ggml_nbytes(glu_g), (char) 0xA5);
        ggml_backend_tensor_set(glu_g, sentinel.data(), 0, sentinel.size());

        ggml_backend_cuda_context & cctx = *(ggml_backend_cuda_context *) cuda->context;
        ok = ggml_cuda_mmq_tma_gate_up_situ(cctx, gu_g, x_g, ids_g, glu_g, ROT_CARD, (int32_t) n_local, beta, lb,
            nullptr, true, ROT_CARDS);
        ok = ok && ggml_cuda_mmq_tma_down_weighted_sum(cctx, dn_g, glu_g, ids_g, w_g, out_g, ROT_CARD, (int32_t) n_local,
            nullptr, ROT_CARDS);
        ggml_backend_synchronize(cuda);
        if (!ok) {
            printf("rotated fused: TMA entry point declined the bonus bank  FAIL\n");
        } else {
            std::vector<float> glu(ggml_nelements(glu_g)), out(ggml_nelements(out_g));
            ggml_backend_tensor_get(glu_g, glu.data(), 0, glu.size()*sizeof(float));
            ggml_backend_tensor_get(out_g, out.data(), 0, out.size()*sizeof(float));
            ggml_backend_tensor_set(h_c, glu.data(), 0, glu.size()*sizeof(float));
            ok = ggml_backend_graph_compute(cpu, gf) == GGML_STATUS_SUCCESS;
            std::vector<float> glu_ref(ggml_nelements(glu_c)), dn(ggml_nelements(dn_out));
            ggml_backend_tensor_get(glu_c, glu_ref.data(), 0, glu_ref.size()*sizeof(float));
            ggml_backend_tensor_get(dn_out, dn.data(), 0, dn.size()*sizeof(float));
            int64_t n_owned = 0, bad_zero = 0;
            for (int64_t s = 0; s < n_used*n_tok; ++s) {
                const bool own = ids[s] % ROT_CARDS == ROT_CARD;
                n_owned += own;
                for (int64_t j = 0; j < n_ff && !own; ++j) {
                    bad_zero += glu[s*n_ff + j] != 0.0f;
                }
            }
            std::vector<float> out_ref(n_embd*n_tok, 0.0f);
            for (int64_t t = 0; t < n_tok; ++t) {
                for (int64_t u = 0; u < n_used; ++u) {
                    const int64_t s = t*n_used + u;
                    if (ids[s] % ROT_CARDS == ROT_CARD) {
                        for (int64_t j = 0; j < n_embd; ++j) {
                            out_ref[t*n_embd + j] += wts[s]*dn[s*n_embd + j];
                        }
                    }
                }
            }
            const double err_glu = compute_nmse(glu_ref, glu);
            const double err_out = compute_nmse(out_ref, out);

            // control: the same call with a contiguous range must take trap ids as owned, or the check above proves nothing
            int64_t ctl_nonzero = 0;
            ok = ok && n_trap > 0 && ggml_cuda_mmq_tma_gate_up_situ(cctx, gu_g, x_g, ids_g, glu_g, ROT_CARD,
                (int32_t) n_local, beta, lb, nullptr, true, 1);
            ggml_backend_synchronize(cuda);
            ggml_backend_tensor_get(glu_g, glu.data(), 0, glu.size()*sizeof(float));
            for (int64_t s = 0; s < n_used*n_tok; ++s) {
                for (int64_t j = 0; j < n_ff && ids[s] % ROT_CARDS != ROT_CARD; ++j) {
                    ctl_nonzero += glu[s*n_ff + j] != 0.0f;
                }
            }
            ok = ok && bad_zero == 0 && err_glu < 5e-4 && err_out < 5e-4 && ctl_nonzero > 0;
            printf("rotated fused bonus (stride 4): owned %lld/%lld, trap ids %lld, non-owned glu nonzero %lld, "
                "glu nmse %.3e, down+sum nmse %.3e, stride-1 control nonzero %lld  %s\n", (long long) n_owned,
                (long long) (n_used*n_tok), (long long) n_trap, (long long) bad_zero, err_glu, err_out,
                (long long) ctl_nonzero, ok ? "ok" : "FAIL");
        }
    }
    if (buf_g) { ggml_backend_buffer_free(buf_g); }
    if (buf_c) { ggml_backend_buffer_free(buf_c); }
    ggml_free(gctx);
    ggml_free(ctx_g);
    ggml_free(ctx_c);
    return ok;
}
#endif

static int run_rotated_suite(ggml_backend_t cuda, ggml_backend_t cpu, int n_threads, int iters, const char * only) {
    const rot_case cases[] = {
        { GGML_TYPE_Q3_K, 4608, 1024, false, true,  "rot gate_up base q3_K N=1024"  },
        { GGML_TYPE_Q3_K, 4608,  512, true,  true,  "rot gate_up bonus q3_K N=512"  },
        { GGML_TYPE_Q4_K,  512, 4608, false, false, "rot down base q4_K K=512"      },
        { GGML_TYPE_Q5_K,  512, 4608, false, false, "rot down base q5_K K=512"      },
        { GGML_TYPE_Q4_K,  256, 4608, true,  false, "rot down bonus q4_K K=256"     },
        { GGML_TYPE_Q5_K,  256, 4608, true,  false, "rot down bonus q5_K K=256"     },
    };
    int failures = 0;
    uint32_t seed = 0x707a7ed1u;
    for (const rot_case & c : cases) {
        for (int64_t n_tok : { 23, 512 }) {
            int64_t owned = 0;
            seed += 31u;
            char name[96];
            snprintf(name, sizeof(name), "%s n=%lld", c.label, (long long) n_tok);
            if (only == nullptr || strstr(name, only) != nullptr) {
                failures += !run_rot_case(cuda, cpu, c, n_tok, n_threads, seed, owned, iters);
            }
        }
    }
#ifdef GGML_USE_CUDA
    if (only == nullptr || strstr("fused", only) != nullptr) {
        failures += !run_rot_fused(cuda, cpu, n_threads);
    }
#endif
    return failures;
}

int main(int argc, char ** argv) {
    const bool tma_23  = argc > 1 && strcmp(argv[1], "--tma-23") == 0;
    const bool rotated = argc > 1 && strcmp(argv[1], "--rotated") == 0;
    const char * dump_path = (!tma_23 && !rotated && argc > 1) ? argv[1] : nullptr;

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

    if (tma_23 || rotated) {
        // --rotated [iters [filter]]: time each case over iters calls; run only cases whose "label n=rows" holds filter
        const int    iters = rotated && argc > 2 ? atoi(argv[2]) : 0;
        const char * only  = rotated && argc > 3 ? argv[3] : nullptr;
        const int failures = tma_23 ? run_tma_23_suite(cuda, cpu, n_threads) :
            run_rotated_suite(cuda, cpu, n_threads, iters, only);
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
