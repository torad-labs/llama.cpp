// The Blackwell sm_120 MMQ path for Inkling 2's K-quant experts, judged against the host at the served prefill shapes.
//
// mmq-config-blackwell.cuh gave Q3_K/Q4_K/Q5_K rows of their own (64-row tile, 128 threads, three blocks an SM) so the
// prefill MUL_MAT_ID stops falling through to the Ampere table. This binary runs each expert product through the CUDA
// backend and through the CPU backend on the same bytes and compares: the contract is the kernel's own, output equal
// to the host within the quantized NMSE test-backend-ops allows (5e-4), not bit-identity to another GPU config.
//
// Shapes are the served ones: 384 experts, top 8 a token, 1,024 prefill rows, gate/up at k=2304 m=4608 and down at
// k=4608 m=2304, for all three types, plus one 16-row pass to confirm the narrow widths left on the Ampere table still
// match the host. Weights are quantized one row at a time, so no f32 copy of an expert tensor is ever held.
//
// A run with no CUDA device skips (exit 0): the test judges a sm_120 path and means nothing on other cards.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

static constexpr int64_t n_expert = 384;
static constexpr int64_t n_used   = 8;

static float noise_from(uint32_t & state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return (float) (state & 0xffffff) / (float) 0x1000000 - 0.5f;
}

struct kase {
    ggml_type type;
    int64_t   k;      // input width
    int64_t   m;      // an expert's output width
    int64_t   n;      // tokens this pass
    const char * name;
};

struct sample {
    ggml_tensor * as;   // [k, m, n_expert], quantized
    ggml_tensor * ids;  // [n_used, n], i32
    ggml_tensor * b;    // [k, n_used, n], f32
    ggml_tensor * out;
};

static sample build(ggml_context * ctx, const kase & c) {
    sample s;
    s.as  = ggml_new_tensor_3d(ctx, c.type, c.k, c.m, n_expert);
    s.ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, c.n);
    s.b   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, c.k, n_used, c.n);
    s.out = ggml_mul_mat_id(ctx, s.as, s.b, s.ids);
    return s;
}

static void quantize_weight_pair(ggml_tensor * w_gpu, ggml_tensor * w_cpu, ggml_type type, int64_t per, uint32_t seed, int n_threads) {
    const int64_t nrow = ggml_nrows(w_gpu);
    const size_t  row_bytes = ggml_row_size(type, per);
    const int64_t chunk_rows = std::max<int64_t>(1, (8 << 20) / per);
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
        const size_t off = (size_t) r0 * row_bytes;
        const size_t bytes = (size_t) nr * row_bytes;
        ggml_backend_tensor_set(w_gpu, qbuf.data(), off, bytes);
        ggml_backend_tensor_set(w_cpu, qbuf.data(), off, bytes);
    }
}

// Quantize in bounded chunks (~8M floats peak) across threads once and upload the same bytes to both backends.
static void initialize_both(const sample & sg, const sample & sc, const kase & c, uint32_t seed, int n_threads) {
    quantize_weight_pair(sg.as, sc.as, c.type, c.k, seed, n_threads);

    uint32_t st = seed | 1u;
    std::vector<int32_t> ids(n_used * c.n);
    for (auto & id : ids) {
        id = (int32_t) (st % (uint32_t) n_expert);
        noise_from(st);
    }
    ggml_backend_tensor_set(sg.ids, ids.data(), 0, ids.size() * sizeof(int32_t));
    ggml_backend_tensor_set(sc.ids, ids.data(), 0, ids.size() * sizeof(int32_t));

    const int64_t chunk = 1 << 20;
    std::vector<float> values(chunk);
    const size_t total = (size_t) ggml_nelements(sg.b) * sizeof(float);
    for (size_t offset = 0; offset < total; offset += chunk * sizeof(float)) {
        const size_t values_here = std::min<size_t>(chunk, (total - offset) / sizeof(float));
        for (size_t i = 0; i < values_here; ++i) {
            values[i] = noise_from(st);
        }
        ggml_backend_tensor_set(sg.b, values.data(), offset, values_here * sizeof(float));
        ggml_backend_tensor_set(sc.b, values.data(), offset, values_here * sizeof(float));
    }
}

// Served tensor-4 by-expert TMA decode shapes (96 local of 384 experts, n = 2..23):
// Q3_K gate_up (k=4608, m=4608, broadcast src1) and Q4_K down (k=2304, m=4608, per-slot src1),
// including routes outside [first, first + n_local).
static int run_tma_shapes(ggml_backend_t cuda, ggml_backend_t cpu, int n_threads) {
    struct tma_shape {
        ggml_type    type;
        int64_t      k;
        int64_t      m;
        bool         broadcast_b;
        const char * label;
    };
    const tma_shape shapes[] = {
        { GGML_TYPE_Q3_K, 4608, 4608, true,  "tma by-expert gate_up q3_k" },
        { GGML_TYPE_Q4_K, 2304, 4608, false, "tma by-expert down q4_k"    },
        { GGML_TYPE_Q5_K, 2304, 4608, false, "tma by-expert down q5_k"    },
    };
    static constexpr int32_t first   = 96;
    static constexpr int64_t n_local = 96;

    int failures = 0;
    for (const tma_shape & sh : shapes) {
        ggml_init_params wparams = { 4 * ggml_tensor_overhead(), nullptr, true };
        ggml_context * wctx_g = ggml_init(wparams);
        ggml_context * wctx_c = ggml_init(wparams);
        ggml_tensor * as_g = ggml_new_tensor_3d(wctx_g, sh.type, sh.k, sh.m, n_local);
        ggml_tensor * as_c = ggml_new_tensor_3d(wctx_c, sh.type, sh.k, sh.m, n_local);
        ggml_backend_buffer_t wbuf_g = ggml_backend_alloc_ctx_tensors(wctx_g, cuda);
        ggml_backend_buffer_t wbuf_c = ggml_backend_alloc_ctx_tensors(wctx_c, cpu);
        if (!wbuf_g || !wbuf_c) {
            printf("%-40s FAIL: weight alloc failed\n", sh.label);
            if (wbuf_g) { ggml_backend_buffer_free(wbuf_g); }
            if (wbuf_c) { ggml_backend_buffer_free(wbuf_c); }
            ggml_free(wctx_g);
            ggml_free(wctx_c);
            ++failures;
            continue;
        }
        quantize_weight_pair(as_g, as_c, sh.type, sh.k, sh.broadcast_b ? 0x51a7e001u : 0x51a7e002u, n_threads);

        double worst_nmse = 0.0;
        bool shape_ok = true;
        for (int64_t n = 2; n <= 23; ++n) {
            ggml_init_params params  = { 8 * ggml_tensor_overhead(), nullptr, true };
            ggml_init_params gparams = { ggml_graph_overhead(), nullptr, true };
            ggml_context * ctx_g  = ggml_init(params);
            ggml_context * ctx_c  = ggml_init(params);
            ggml_context * gctx_g = ggml_init(gparams);
            ggml_context * gctx_c = ggml_init(gparams);

            const int64_t b_cols = sh.broadcast_b ? 1 : n_used;
            ggml_tensor * ids_g = ggml_new_tensor_2d(ctx_g, GGML_TYPE_I32, n_used, n);
            ggml_tensor * b_g   = ggml_new_tensor_3d(ctx_g, GGML_TYPE_F32, sh.k, b_cols, n);
            ggml_tensor * out_g = ggml_mul_mat_id(ctx_g, as_g, b_g, ids_g);
            ggml_mul_mat_id_set_expert_range(out_g, first, (int32_t) n_expert);

            ggml_tensor * ids_c = ggml_new_tensor_2d(ctx_c, GGML_TYPE_I32, n_used, n);
            ggml_tensor * b_c   = ggml_new_tensor_3d(ctx_c, GGML_TYPE_F32, sh.k, b_cols, n);
            ggml_tensor * out_c = ggml_mul_mat_id(ctx_c, as_c, b_c, ids_c);
            ggml_mul_mat_id_set_expert_range(out_c, first, (int32_t) n_expert);

            ggml_cgraph * gf_g = ggml_new_graph(gctx_g);
            ggml_cgraph * gf_c = ggml_new_graph(gctx_c);
            ggml_build_forward_expand(gf_g, out_g);
            ggml_build_forward_expand(gf_c, out_c);

            ggml_backend_buffer_t buf_g = ggml_backend_alloc_ctx_tensors(ctx_g, cuda);
            ggml_backend_buffer_t buf_c = ggml_backend_alloc_ctx_tensors(ctx_c, cpu);
            bool ok = buf_g != nullptr && buf_c != nullptr;
            if (ok) {
                uint32_t st = uint32_t(0x12345600u + n * 97u) | 1u;
                std::vector<int32_t> ids(n_used * n);
                for (int64_t t = 0; t < n; ++t) {
                    ids[t * n_used + 0] = first + (int32_t) (t % 3);          // shared local expert across tokens
                    ids[t * n_used + 1] = (int32_t) (t % first);              // non-owned below range
                    ids[t * n_used + 2] = first + (int32_t) n_local - 1;      // last local expert
                    ids[t * n_used + 3] = first + (int32_t) n_local + (int32_t) (t % 16); // non-owned above range
                    for (int64_t s = 4; s < n_used; ++s) {
                        int32_t cand = 0;
                        bool dup = false;
                        do {
                            noise_from(st);
                            cand = (int32_t) (st % (uint32_t) n_expert);
                            dup = false;
                            for (int64_t p = 0; p < s; ++p) {
                                if (ids[t * n_used + p] == cand) {
                                    dup = true;
                                    break;
                                }
                            }
                        } while (dup);
                        ids[t * n_used + s] = cand;
                    }
                }
                ggml_backend_tensor_set(ids_g, ids.data(), 0, ids.size() * sizeof(int32_t));
                ggml_backend_tensor_set(ids_c, ids.data(), 0, ids.size() * sizeof(int32_t));

                std::vector<float> b_vals(ggml_nelements(b_g));
                for (float & v : b_vals) {
                    v = noise_from(st);
                }
                ggml_backend_tensor_set(b_g, b_vals.data(), 0, b_vals.size() * sizeof(float));
                ggml_backend_tensor_set(b_c, b_vals.data(), 0, b_vals.size() * sizeof(float));

                std::vector<char> sentinel(ggml_nbytes(out_g), (char) 0xA5);
                ggml_backend_tensor_set(out_g, sentinel.data(), 0, sentinel.size());

                ok = ggml_backend_graph_compute(cuda, gf_g) == GGML_STATUS_SUCCESS &&
                     ggml_backend_graph_compute(cpu,  gf_c) == GGML_STATUS_SUCCESS;
                if (ok) {
                    std::vector<float> gpu(ggml_nelements(out_g));
                    std::vector<float> ref(ggml_nelements(out_c));
                    ggml_backend_tensor_get(out_g, gpu.data(), 0, gpu.size() * sizeof(float));
                    ggml_backend_tensor_get(out_c, ref.data(), 0, ref.size() * sizeof(float));
                    for (int64_t t = 0; t < n && ok; ++t) {
                        for (int64_t s = 0; s < n_used && ok; ++s) {
                            const int32_t e = ids[t * n_used + s];
                            if (e < first || e >= first + (int32_t) n_local) {
                                const float * row = gpu.data() + (t * n_used + s) * sh.m;
                                for (int64_t col = 0; col < sh.m; ++col) {
                                    if (row[col] != 0.0f) {
                                        printf("%s n=%d FAIL: non-owned slot (%d,%d) expert %d not zeroed (%g)\n",
                                            sh.label, (int) n, (int) t, (int) s, (int) e, (double) row[col]);
                                        ok = false;
                                        break;
                                    }
                                }
                            }
                        }
                    }
                    const double err = [&]() {
                        double e = 0.0, p = 0.0;
                        for (size_t i = 0; i < ref.size(); ++i) {
                            const double d = (double) ref[i] - (double) gpu[i];
                            e += d * d;
                            p += (double) ref[i] * (double) ref[i];
                        }
                        return p > 0.0 ? e / p : (e > 0.0 ? 1.0 : 0.0);
                    }();
                    worst_nmse = std::max(worst_nmse, err);
                    if (err >= 5e-4) {
                        printf("%s n=%d FAIL: nmse %.3e\n", sh.label, (int) n, err);
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
                shape_ok = false;
                break;
            }
        }
        printf("%-32s n=2..23 worst nmse %.3e  %s\n", sh.label, worst_nmse, shape_ok ? "ok" : "FAIL");
        if (!shape_ok) {
            ++failures;
        }
        ggml_backend_buffer_free(wbuf_g);
        ggml_backend_buffer_free(wbuf_c);
        ggml_free(wctx_g);
        ggml_free(wctx_c);
    }
    return failures;
}

static double nmse(const std::vector<float> & a, const std::vector<float> & b) {
    double error = 0.0, power = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double d = (double) a[i] - (double) b[i];
        error += d * d;
        power += (double) a[i] * (double) a[i];
    }
    return power > 0.0 ? error / power : (error > 0.0 ? 1.0 : 0.0);
}

int main(int argc, char ** argv) {
    const bool tma_only = argc > 1 && strcmp(argv[1], "--tma") == 0;
    ggml_backend_t cuda = ggml_backend_cuda_init(0);
    if (cuda == nullptr) {
        printf("test-inkling-mmq-kquant: no CUDA device, skipped\n");
        return 0;
    }
    const int n_threads = std::min<int>(16, std::max<int>(1, (int) std::thread::hardware_concurrency()));
    ggml_backend_t cpu = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(cpu, n_threads);

    if (tma_only) {
        const int failures = run_tma_shapes(cuda, cpu, n_threads);
        ggml_backend_free(cpu);
        ggml_backend_free(cuda);
        return failures == 0 ? 0 : 1;
    }

    const kase cases[] = {
        { GGML_TYPE_Q3_K, 2304, 4608, 1024, "gate/up q3_k" },
        { GGML_TYPE_Q5_K, 2304, 4608, 1024, "gate/up q5_k" },
        { GGML_TYPE_Q4_K, 2304, 4608, 1024, "gate/up q4_k" },
        { GGML_TYPE_Q3_K, 4608, 2304, 1024, "down q3_k"    },
        { GGML_TYPE_Q4_K, 4608, 2304, 1024, "down q4_k"    },
        { GGML_TYPE_Q5_K, 4608, 2304, 1024, "down q5_k"    },
        { GGML_TYPE_Q3_K, 2304, 4608,   16, "gate/up q3_k narrow (Ampere widths)" },
    };

    int failures = 0;
    for (const kase & c : cases) {
        ggml_init_params params = { 8 * ggml_tensor_overhead(), nullptr, true };
        ggml_init_params gparams = { ggml_graph_overhead(), nullptr, true };
        ggml_context * ctx_g = ggml_init(params);
        ggml_context * ctx_c = ggml_init(params);
        ggml_context * gctx_g = ggml_init(gparams);
        ggml_context * gctx_c = ggml_init(gparams);
        const sample sg = build(ctx_g, c);
        const sample sc = build(ctx_c, c);
        ggml_cgraph * gf_g = ggml_new_graph(gctx_g);
        ggml_cgraph * gf_c = ggml_new_graph(gctx_c);
        ggml_build_forward_expand(gf_g, sg.out);
        ggml_build_forward_expand(gf_c, sc.out);
        ggml_backend_buffer_t buf_g = ggml_backend_alloc_ctx_tensors(ctx_g, cuda);
        ggml_backend_buffer_t buf_c = ggml_backend_alloc_ctx_tensors(ctx_c, cpu);
        bool ok_gpu = buf_g != nullptr;
        bool ok_cpu = buf_c != nullptr;
        std::vector<float> gpu, ref;
        if (ok_gpu && ok_cpu) {
            initialize_both(sg, sc, c, 0x9e3779b9u, n_threads);
            ok_gpu = ggml_backend_graph_compute(cuda, gf_g) == GGML_STATUS_SUCCESS;
            ok_cpu = ggml_backend_graph_compute(cpu,  gf_c) == GGML_STATUS_SUCCESS;
            if (ok_gpu && ok_cpu) {
                gpu.resize(ggml_nelements(sg.out));
                ref.resize(ggml_nelements(sc.out));
                ggml_backend_tensor_get(sg.out, gpu.data(), 0, gpu.size() * sizeof(float));
                ggml_backend_tensor_get(sc.out, ref.data(), 0, ref.size() * sizeof(float));
            }
        }
        if (buf_g) { ggml_backend_buffer_free(buf_g); }
        if (buf_c) { ggml_backend_buffer_free(buf_c); }
        ggml_free(gctx_g); ggml_free(gctx_c);
        ggml_free(ctx_g);  ggml_free(ctx_c);

        if (!ok_gpu || !ok_cpu || gpu.size() != ref.size()) {
            printf("%-40s FAIL: run failed (gpu=%d cpu=%d)\n", c.name, (int) ok_gpu, (int) ok_cpu);
            ++failures;
            continue;
        }
        const double value = nmse(ref, gpu);
        const bool passed = value < 5e-4;   // test-backend-ops' own bound for a quantized type_a against the host
        printf("%-40s nmse %.3e  %s\n", c.name, value, passed ? "ok" : "FAIL");
        if (!passed) {
            ++failures;
        }
    }
    ggml_backend_free(cpu);
    ggml_backend_free(cuda);
    return failures == 0 ? 0 : 1;
}
