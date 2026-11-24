// Focused test for kquant-j-rule: verifies Blackwell K-quant-scoped MoE J selection and pins GLM IQ3_XXS unchanged.
// Part 1 checks mmq-config-blackwell.cuh entries, SMEM budgets, and spread J selections directly on host.
// Part 2 runs live CUDA MUL_MAT_ID with GGML_CUDA_MMQ_ROUTE_LOG on sm_120 to verify actual dispatched J values.

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "mmq.cuh"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static size_t compute_smem(const ggml_cuda_mmq_config & cfg, const int cc, const bool kquant_staged) {
    const size_t nbs_ids      = cfg.J * sizeof(int);
    const size_t nbs_x        = ggml_cuda_mmq_get_nbytes_shared_x(cfg, cc);
    const size_t nbs_y        = cfg.J * sizeof(block_q8_1_mmq);
    const size_t nbs_y_padded = GGML_PAD(nbs_y, cfg.nthreads * sizeof(int));
    if (mmq_iq3_xxs_staged(cfg.type, cc)) {
        return nbs_ids + nbs_x + 2 * nbs_y_padded + cfg.I * MMQ_IQ3_XXS_STAGED_ROW + 256 * sizeof(uint32_t);
    }
    if (kquant_staged && mmq_is_staged_kquant_type(cfg.type) && blackwell_mma_available(cc)) {
        return nbs_ids + nbs_x + 2 * nbs_y_padded + cfg.I * mmq_kquant_staged_row(cfg.type);
    }
    return nbs_ids + nbs_x + nbs_y_padded;
}

static int select_J_host(const ggml_type type, const int64_t ncols, const bool fallback,
                         const int cc, const size_t smpbo, const bool kquant_staged) {
    int J_best = 0;
    int ntiles_best = INT_MAX;
    for (int J = 8; J <= 128 && ntiles_best > 1; J += 8) {
        const ggml_cuda_mmq_config cfg = ggml_cuda_mmq_get_config(type, J, fallback, cc, GGML_PREC_Q8);
        if (cfg.type == GGML_TYPE_COUNT) {
            continue;
        }
        if (compute_smem(cfg, cc, kquant_staged) > smpbo) {
            continue;
        }
        const int ntiles = (int) ((ncols + cfg.J - 1) / cfg.J);
        if (ntiles < ntiles_best) {
            J_best = J;
            ntiles_best = ntiles;
        }
    }
    return J_best;
}

static int test_host_tables() {
    static constexpr int    cc_bw  = GGML_CUDA_CC_BLACKWELL;
    static constexpr size_t smpbo  = 99 * 1024;
    int failures = 0;

    // 1. Pin GLM IQ3_XXS config table completely unchanged across every J and fallback.
    for (int fb = 0; fb <= 1; ++fb) {
        const bool fallback = fb != 0;
        for (int J = 8; J <= 128; J += 8) {
            const ggml_cuda_mmq_config bw  = ggml_cuda_mmq_get_config(GGML_TYPE_IQ3_XXS, J, fallback, cc_bw, GGML_PREC_Q8);
            const ggml_cuda_mmq_config amp = ggml_cuda_mmq_get_config_ampere(GGML_TYPE_IQ3_XXS, J, fallback);
            if (J == 8 || J == 16) {
                if (bw.type != GGML_TYPE_IQ3_XXS || bw.nthreads != 128 || bw.occupancy != 3 ||
                    bw.I != 64 || bw.J != J || bw.sram_layout != GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0 ||
                    bw.K_vram != MMQ_ITER_K || !bw.stream_k || bw.fallback != fallback) {
                    printf("FAIL: IQ3_XXS Blackwell J=%d fb=%d changed\n", J, fb);
                    ++failures;
                }
            } else {
                if (bw.type != amp.type || bw.nthreads != amp.nthreads || bw.occupancy != amp.occupancy ||
                    bw.I != amp.I || bw.J != amp.J || bw.sram_layout != amp.sram_layout ||
                    bw.K_vram != amp.K_vram || bw.stream_k != amp.stream_k || bw.fallback != amp.fallback) {
                    printf("FAIL: IQ3_XXS Blackwell J=%d fb=%d diverged from Ampere\n", J, fb);
                    ++failures;
                }
            }
            // GB10 config table keeps Ampere config for IQ3_XXS and K-quants.
            const ggml_cuda_mmq_config gb = ggml_cuda_mmq_get_config_gb10(GGML_TYPE_IQ3_XXS, J, fallback);
            if (gb.type != amp.type || gb.I != amp.I || gb.J != amp.J || gb.nthreads != amp.nthreads) {
                printf("FAIL: IQ3_XXS GB10 J=%d fb=%d diverged from Ampere\n", J, fb);
                ++failures;
            }
        }
    }

    // 2. Pin GLM IQ3_XXS MoE spread J selection identical with legacy=false and legacy=true across 1..2048 tokens.
    const int64_t glm_experts[] = { 64, 128, 256, 288, 384 };
    const int64_t glm_tokens[]  = { 1, 8, 23, 64, 128, 256, 512, 1024, 1536, 2048 };
    for (int64_t ne : glm_experts) {
        for (int64_t nt : glm_tokens) {
            const int64_t nc_new = std::min({ nt, int64_t(128), ggml_cuda_mmq_moe_spread_ncols(GGML_TYPE_IQ3_XXS, cc_bw, nt, 8, ne, false) });
            const int64_t nc_leg = std::min({ nt, int64_t(128), ggml_cuda_mmq_moe_spread_ncols(GGML_TYPE_IQ3_XXS, cc_bw, nt, 8, ne, true) });
            const int64_t nc_ref = std::min({ nt, int64_t(128), int64_t(std::ceil(double(nt * 8) / double(ne) + 3.0 * std::sqrt(double(nt * 8) / double(ne)))) });
            if (nc_new != nc_leg || nc_new != nc_ref) {
                printf("FAIL: IQ3_XXS spread changed for ne=%lld nt=%lld (new=%lld leg=%lld ref=%lld)\n",
                    (long long) ne, (long long) nt, (long long) nc_new, (long long) nc_leg, (long long) nc_ref);
                ++failures;
            }
            for (int fb = 0; fb <= 1; ++fb) {
                const int j_new = select_J_host(GGML_TYPE_IQ3_XXS, nc_new, fb != 0, cc_bw, smpbo, true);
                const int j_leg = select_J_host(GGML_TYPE_IQ3_XXS, nc_leg, fb != 0, cc_bw, smpbo, false);
                if (j_new != j_leg) {
                    printf("FAIL: IQ3_XXS J changed for ne=%lld nt=%lld fb=%d (%d vs %d)\n",
                        (long long) ne, (long long) nt, fb, j_new, j_leg);
                    ++failures;
                }
            }
        }
    }

    // 3. Verify Blackwell K-quant J selections at served Inkling shapes (384 experts, top-8) and staged SMEM budgets.
    struct expect_j {
        int64_t n_tok;
        int     j_new;
        int     j_legacy;
    };
    const expect_j expected[] = {
        {   64,  8,  8 },
        {  128,  8,  8 },
        {  256, 16, 16 },
        {  512, 16, 24 },
        { 1024, 32, 40 },
        { 1536, 48, 56 },
        { 2048, 56, 64 },
    };
    const ggml_type ktypes[] = { GGML_TYPE_Q3_K, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K };
    for (ggml_type tp : ktypes) {
        for (int J = 8; J <= 128; J += 8) {
            for (int fb = 0; fb <= 1; ++fb) {
                const ggml_cuda_mmq_config cfg = ggml_cuda_mmq_get_config(tp, J, fb != 0, cc_bw, GGML_PREC_Q8);
                if (cfg.type == GGML_TYPE_COUNT) {
                    continue;
                }
                const size_t smem = compute_smem(cfg, cc_bw, true);
                if (smem > smpbo) {
                    printf("FAIL: %s J=%d fb=%d staged smem=%zu > %zu\n", ggml_type_name(tp), J, fb, smem, smpbo);
                    ++failures;
                }
                if (fb == 0 && J >= 24 && J <= 56) {
                    if (cfg.I != 64 || cfg.nthreads != 128 || cfg.occupancy != 2) {
                        printf("FAIL: %s J=%d Blackwell config unexpected (I=%d nt=%d occ=%d)\n",
                            ggml_type_name(tp), J, cfg.I, cfg.nthreads, cfg.occupancy);
                        ++failures;
                    }
                    if (smem > 50176) {
                        printf("FAIL: %s J=%d staged smem=%zu breaks 2 blocks/SM\n", ggml_type_name(tp), J, smem);
                        ++failures;
                    }
                }
            }
        }
        for (const expect_j & ex : expected) {
            const int64_t nc_new = std::min({ ex.n_tok, int64_t(128), ggml_cuda_mmq_moe_spread_ncols(tp, cc_bw, ex.n_tok, 8, 384, false) });
            const int64_t nc_leg = std::min({ ex.n_tok, int64_t(128), ggml_cuda_mmq_moe_spread_ncols(tp, cc_bw, ex.n_tok, 8, 384, true) });
            const int j_new = select_J_host(tp, nc_new, false, cc_bw, smpbo, true);
            const int j_leg = select_J_host(tp, nc_leg, false, cc_bw, smpbo, false);
            if (j_new != ex.j_new || j_leg != ex.j_legacy) {
                printf("FAIL: %s n=%lld J mismatch: got (%d, leg=%d), expected (%d, leg=%d)\n",
                    ggml_type_name(tp), (long long) ex.n_tok, j_new, j_leg, ex.j_new, ex.j_legacy);
                ++failures;
            }
        }
    }

    printf("host table & IQ3_XXS pin checks: %s\n", failures == 0 ? "ok" : "FAIL");
    return failures;
}

static bool run_live_mmid(ggml_backend_t cuda, ggml_type type, const char * name,
                          int64_t n_local, int64_t n_total, int64_t n_tok) {
    ggml_init_params params  = { 16 * ggml_tensor_overhead(), nullptr, true };
    ggml_init_params gparams = { ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx  = ggml_init(params);
    ggml_context * gctx = ggml_init(gparams);

    ggml_tensor * as  = ggml_new_tensor_3d(ctx, type, 512, 256, n_local);
    ggml_set_name(as, name);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 8, n_tok);
    ggml_tensor * b   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 512, 1, n_tok);
    ggml_tensor * out = ggml_mul_mat_id(ctx, as, b, ids);
    ggml_mul_mat_id_set_expert_range(out, 0, (int32_t) n_total);

    ggml_cgraph * gf = ggml_new_graph(gctx);
    ggml_build_forward_expand(gf, out);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, cuda);
    bool ok = buf != nullptr;
    if (ok) {
        std::vector<char> zeros(ggml_nbytes(as), 0);
        ggml_backend_tensor_set(as, zeros.data(), 0, zeros.size());
        std::vector<float> bzeros(ggml_nelements(b), 0.1f);
        ggml_backend_tensor_set(b, bzeros.data(), 0, bzeros.size() * sizeof(float));
        std::vector<int32_t> idv(8 * n_tok);
        for (int64_t t = 0; t < n_tok; ++t) {
            for (int64_t u = 0; u < 8; ++u) {
                idv[t * 8 + u] = (int32_t) ((t + u * 7) % n_total);
            }
        }
        ggml_backend_tensor_set(ids, idv.data(), 0, idv.size() * sizeof(int32_t));
        ok = ggml_backend_graph_compute(cuda, gf) == GGML_STATUS_SUCCESS;
        ggml_backend_synchronize(cuda);
    }
    if (buf) { ggml_backend_buffer_free(buf); }
    ggml_free(gctx);
    ggml_free(ctx);
    return ok;
}

static int parse_int_field(const std::string & line, const char * key) {
    const std::string pat = std::string("\"") + key + "\":";
    const size_t pos = line.find(pat);
    if (pos == std::string::npos) {
        return -1;
    }
    return std::atoi(line.c_str() + pos + pat.size());
}

static std::string parse_str_field(const std::string & line, const char * key) {
    const std::string pat = std::string("\"") + key + "\":\"";
    const size_t pos = line.find(pat);
    if (pos == std::string::npos) {
        return "";
    }
    const size_t start = pos + pat.size();
    const size_t end   = line.find('"', start);
    return end == std::string::npos ? "" : line.substr(start, end - start);
}

static int test_live_dispatch(const char * log_path) {
    int dev_count = 0;
    if (cudaGetDeviceCount(&dev_count) != cudaSuccess || dev_count <= 0) {
        printf("live sm_120 dispatch check: no CUDA device, skipped\n");
        return 0;
    }
    cudaDeviceProp prop = {};
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess || prop.major * 100 + prop.minor * 10 != GGML_CUDA_CC_BLACKWELL) {
        printf("live sm_120 dispatch check: device cc=%d.%d != 12.0, skipped\n", prop.major, prop.minor);
        return 0;
    }

    const bool legacy = getenv("GGML_CUDA_MMQ_KQUANT_SPREAD_LEGACY") != nullptr &&
                        strcmp(getenv("GGML_CUDA_MMQ_KQUANT_SPREAD_LEGACY"), "0") != 0 &&
                        strcmp(getenv("GGML_CUDA_MMQ_KQUANT_SPREAD_LEGACY"), "") != 0;

    std::remove(log_path);
    setenv("GGML_CUDA_MMQ_ROUTE_LOG", "64", 1);
    setenv("GGML_CUDA_MMQ_ROUTE_LOG_FILE", log_path, 1);

    ggml_backend_t cuda = ggml_backend_cuda_init(0);
    if (!cuda) {
        printf("FAIL: ggml_backend_cuda_init(0) returned null\n");
        return 1;
    }

    struct live_case {
        ggml_type    type;
        const char * name;
        int64_t      n_total;
        int64_t      n_tok;
        int          expect_J;
    };
    const live_case cases[] = {
        { GGML_TYPE_Q3_K,    "ink_q3_512",   384,  512, legacy ? 24 : 16 },
        { GGML_TYPE_Q4_K,    "ink_q4_512",   384,  512, legacy ? 24 : 16 },
        { GGML_TYPE_Q5_K,    "ink_q5_512",   384,  512, legacy ? 24 : 16 },
        { GGML_TYPE_Q3_K,    "ink_q3_1024",  384, 1024, legacy ? 40 : 32 },
        { GGML_TYPE_Q4_K,    "ink_q4_1024",  384, 1024, legacy ? 40 : 32 },
        { GGML_TYPE_Q5_K,    "ink_q5_1024",  384, 1024, legacy ? 40 : 32 },
        { GGML_TYPE_IQ3_XXS, "glm_iq3_512",  288,  512, 32 },
        { GGML_TYPE_IQ3_XXS, "glm_iq3_1024", 288, 1024, 48 },
    };

    int failures = 0;
    for (const live_case & c : cases) {
        if (!run_live_mmid(cuda, c.type, c.name, 32, c.n_total, c.n_tok)) {
            printf("FAIL: live run failed for %s\n", c.name);
            ++failures;
        }
    }
    // Free backend so mmq_route_log flushes records to log_path.
    ggml_backend_free(cuda);

    std::ifstream in(log_path);
    if (!in) {
        printf("FAIL: route log file missing: %s\n", log_path);
        return 1;
    }
    std::vector<std::string> site_names;
    std::vector<int> site_J;
    std::string line;
    while (std::getline(in, line)) {
        if (line.rfind("MMQ_ROUTE_SITE ", 0) == 0) {
            const int sid = parse_int_field(line, "site");
            if (sid >= 0) {
                if ((int) site_names.size() <= sid) {
                    site_names.resize(sid + 1);
                    site_J.resize(sid + 1, -1);
                }
                site_names[sid] = parse_str_field(line, "name");
            }
        } else if (line.rfind("MMQ_ROUTE ", 0) == 0) {
            const int sid = parse_int_field(line, "site");
            const int J   = parse_int_field(line, "j");
            if (sid >= 0 && sid < (int) site_J.size()) {
                site_J[sid] = J;
            }
        }
    }
    std::remove(log_path);

    for (const live_case & c : cases) {
        int got_J = -1;
        for (size_t i = 0; i < site_names.size(); ++i) {
            if (site_names[i] == c.name) {
                got_J = site_J[i];
                break;
            }
        }
        const bool ok = got_J == c.expect_J;
        printf("live %-14s legacy=%d J=%2d (expected %2d)  %s\n",
            c.name, (int) legacy, got_J, c.expect_J, ok ? "ok" : "FAIL");
        if (!ok) {
            ++failures;
        }
    }
    return failures;
}

int main(int argc, char ** argv) {
    const char * log_path = argc > 1 ? argv[1] : "inkling-kquant-jrule-route.jsonl";
    int failures = test_host_tables();
    failures += test_live_dispatch(log_path);
    return failures == 0 ? 0 : 1;
}
