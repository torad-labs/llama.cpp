// Inkling 2's routed-expert chain as fused launches (fuse-moe.cu, mmq-tma-tiles.cu):
// - router product + sigmoid/bias top-k + sorted local routes + q8_1 expert input;
// - merged gate/up product + SITU epilogue (vector or TMA Q3_K);
// - down product + slot-order weighted sum epilogue (vector or TMA Q4_K).
//
// Runs once with GGML_CUDA_MOE_FUSE=0 and once with =1 (both without and with GGML_CUDA_MMQ_TMA=1),
// routing tokens across both local experts [first, first + n_local) and non-owned experts outside that
// range across 1..23 tokens, under both disjoint buffers (checking sentinel elision + non-owned zeroing)
// and ggml_gallocr buffer reuse (checking H1 aliasing safety). Writes outputs to <out.bin> for byte-exact
// comparison between the unfused and fused arms.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <random>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

static constexpr int64_t n_embd    = 4608;
static constexpr int64_t n_ff      = 2304;
static constexpr int64_t n_expert  = 384;
static constexpr int64_t n_local   = 96;  // the card's share of the experts
static constexpr int64_t first     = 32;  // and its first global expert
static constexpr int64_t n_used    = 8;
static constexpr float   situ_beta = 4.0f;
static constexpr float   situ_lb   = 25.0f;

struct model_weights {
    ggml_context *        ctx     = nullptr;
    ggml_backend_buffer_t buf     = nullptr;
    ggml_tensor *         router  = nullptr;
    ggml_tensor *         router_signs = nullptr;
    ggml_tensor *         bias    = nullptr;
    ggml_tensor *         gate_up = nullptr;
    ggml_tensor *         down    = nullptr;
    ggml_tensor *         down_q5 = nullptr;
    std::vector<float>    h_router;
    std::vector<float>    h_bias;
};

struct layer {
    ggml_context * ctx = nullptr;
    ggml_cgraph *  gf  = nullptr;
    ggml_tensor *  cur_in = nullptr;
    ggml_tensor *  cur = nullptr;
    ggml_tensor *  logits = nullptr;
    ggml_tensor *  probs = nullptr;
    ggml_tensor *  ids = nullptr;
    ggml_tensor *  weights = nullptr;
    ggml_tensor *  gate_up = nullptr;
    ggml_tensor *  gate_v = nullptr;
    ggml_tensor *  up_v = nullptr;
    ggml_tensor *  glu = nullptr;
    ggml_tensor *  down = nullptr;
    ggml_tensor *  wmul = nullptr;
    std::vector<ggml_tensor *> views;
    std::vector<ggml_tensor *> adds;
    ggml_tensor *  out = nullptr;
};

static inline float noise(uint32_t & s) {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    return (float) ((int32_t) s)*1.0e-9f;
}

// Quantize in bounded chunks across host threads so peak host RAM stays small.
static void quantize_rows(ggml_tensor * t, unsigned seed, int n_threads) {
    const int64_t per        = t->ne[0];
    const int64_t nrow       = ggml_nrows(t);
    const size_t  row_bytes  = ggml_row_size(t->type, per);
    const int64_t chunk_rows = std::max<int64_t>(1, (8 << 20) / per);
    std::vector<char> qbuf((size_t) std::min(nrow, chunk_rows) * row_bytes);

    for (int64_t r0 = 0; r0 < nrow; r0 += chunk_rows) {
        const int64_t nr = std::min(chunk_rows, nrow - r0);
        const int nt = std::min<int>(n_threads, (int) nr);
        std::vector<std::thread> workers;
        workers.reserve(nt);
        for (int th = 0; th < nt; ++th) {
            const int64_t a = r0 + (nr * th) / nt;
            const int64_t b = r0 + (nr * (th + 1)) / nt;
            workers.emplace_back([&, a, b]() {
                std::vector<float> src(per);
                for (int64_t r = a; r < b; ++r) {
                    uint32_t s = (seed ^ uint32_t(r * 2654435761u)) | 1u;
                    for (int64_t i = 0; i < per; ++i) {
                        src[i] = noise(s);
                    }
                    const size_t written = ggml_quantize_chunk(t->type, src.data(),
                        qbuf.data() + (size_t) (r - r0) * row_bytes, 0, 1, per, nullptr);
                    GGML_ASSERT(written == row_bytes);
                }
            });
        }
        for (auto & w : workers) {
            w.join();
        }
        ggml_backend_tensor_set(t, qbuf.data(), (size_t) r0 * row_bytes, (size_t) nr * row_bytes);
    }
}

// 13 candidate experts: 8 owned in [32, 128) (including boundaries 32 and 127) and 5 non-owned
// outside [32, 128) (0, 31, 128, 220, 383), so every token's top-8 mixes local and remote experts.
static constexpr int32_t k_candidates[13] = {
    0, 31, 32, 33, 47, 64, 95, 110, 126, 127, 128, 220, 383
};

static bool init_weights(model_weights & W, ggml_backend_t dev_backend, int n_threads, bool bake) {
    ggml_init_params params = { 16 * ggml_tensor_overhead(), nullptr, true };
    W.ctx     = ggml_init(params);
    W.router  = ggml_new_tensor_2d(W.ctx, GGML_TYPE_F32, n_embd, n_expert);
    W.bias    = ggml_new_tensor_1d(W.ctx, GGML_TYPE_F32, n_expert);
    W.gate_up = ggml_new_tensor_3d(W.ctx, GGML_TYPE_Q3_K, n_embd, 2*n_ff, n_local);
    W.down    = ggml_new_tensor_3d(W.ctx, GGML_TYPE_Q4_K, n_ff, n_embd, n_local);
    W.down_q5 = ggml_new_tensor_3d(W.ctx, GGML_TYPE_Q5_K, n_ff, n_embd, n_local);
    if (bake) {
        W.router_signs = ggml_new_tensor_1d(W.ctx, GGML_TYPE_F32, n_embd);
    }

    W.buf = ggml_backend_alloc_ctx_tensors(W.ctx, dev_backend);
    if (W.buf == nullptr) {
        return false;
    }

    W.h_router.assign(n_embd*n_expert, -1.0f);
    W.h_bias.assign(n_expert, -1.0f);
    for (int c = 0; c < 13; ++c) {
        const int32_t e = k_candidates[c];
        for (int64_t i = 0; i < n_embd; ++i) {
            W.h_router[e*n_embd + i] = ((i % 13) == c) ? (4.0e-3f + 2.0e-4f * c) : 1.0e-5f * (c + 1);
        }
        W.h_bias[e] = 0.005f * (float) c;
    }

    if (bake) {
        std::vector<float> signs(n_embd), row(n_embd);
        for (int64_t j = 0; j < n_embd; ++j) {
            signs[j] = (j/16) % 2 ? -1.0f : 1.0f;
        }
        ggml_backend_tensor_set(W.router_signs, signs.data(), 0, n_embd*sizeof(float));
        for (int64_t expert = 0; expert < n_expert; ++expert) {
            for (int64_t j = 0; j < n_embd; ++j) {
                row[j] = W.h_router[expert*n_embd + j]*signs[j];
            }
            ggml_backend_tensor_set(W.router, row.data(), expert*n_embd*sizeof(float), n_embd*sizeof(float));
        }
    } else {
        ggml_backend_tensor_set(W.router, W.h_router.data(), 0, W.h_router.size()*sizeof(float));
    }
    ggml_backend_tensor_set(W.bias,   W.h_bias.data(),   0, W.h_bias.size()*sizeof(float));
    quantize_rows(W.gate_up, 99,  n_threads);
    quantize_rows(W.down,    123, n_threads);
    quantize_rows(W.down_q5, 157, n_threads);
    return true;
}

static void fill_cur(int64_t n_tokens, std::vector<float> & cur) {
    cur.resize(n_embd * n_tokens);
    uint32_t s = uint32_t(1234 + n_tokens * 31) | 1u;
    for (int64_t t = 0; t < n_tokens; ++t) {
        for (int64_t i = 0; i < n_embd; ++i) {
            const int band = (int) ((i + t * 3) % 13);
            cur[t*n_embd + i] = 0.35f + 0.05f * band + 0.15f * std::fabs(noise(s));
        }
    }
}

static void cpu_rms_norm(const std::vector<float> & x, int64_t n_tokens, float eps, std::vector<float> & out) {
    out.resize(n_embd * n_tokens);
    for (int64_t t = 0; t < n_tokens; ++t) {
        double sum_sq = 0.0;
        for (int64_t i = 0; i < n_embd; ++i) {
            const double v = x[t*n_embd + i];
            sum_sq += v * v;
        }
        const float scale = 1.0f / std::sqrt((float) (sum_sq / (double) n_embd) + eps);
        for (int64_t i = 0; i < n_embd; ++i) {
            out[t*n_embd + i] = x[t*n_embd + i] * scale;
        }
    }
}

static layer build(const model_weights & W, int64_t n_tokens, bool computed_cur, bool use_q5_down = false,
        int rank = -1, ggml_tensor * gate_bonus = nullptr, ggml_tensor * down_bonus = nullptr) {
    layer L;
    ggml_init_params params = { 512*ggml_tensor_overhead() + 8*ggml_graph_overhead(), nullptr, true };
    L.ctx = ggml_init(params);
    ggml_context * ctx = L.ctx;

    L.cur_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_tokens);
    ggml_set_input(L.cur_in);
    L.cur = computed_cur ? ggml_rms_norm(ctx, L.cur_in, 1e-5f) : L.cur_in;

    auto * router_cur = W.router_signs ? ggml_mul(ctx, L.cur, W.router_signs) : L.cur;
    L.logits = ggml_mul_mat(ctx, W.router, router_cur);
    L.probs  = ggml_sigmoid(ctx, L.logits);
    ggml_tensor * selection = ggml_add(ctx, L.probs, W.bias);
    L.ids = ggml_argsort_top_k(ctx, selection, n_used);
    ggml_tensor * w = ggml_get_rows(ctx, ggml_reshape_3d(ctx, L.probs, 1, n_expert, n_tokens), L.ids);
    w = ggml_reshape_2d(ctx, w, n_used, n_tokens);
    ggml_tensor * sum = ggml_scale_bias(ctx, ggml_sum_rows(ctx, w), 1.0f, 1e-6f);
    w = ggml_reshape_3d(ctx, ggml_div(ctx, w, sum), 1, n_used, n_tokens);
    L.weights = ggml_scale(ctx, w, 8.0f);

    auto * input = ggml_reshape_3d(ctx, L.cur, n_embd, 1, n_tokens);
    L.gate_up = ggml_mul_mat_id(ctx, W.gate_up, input, L.ids);
    const int64_t width = W.gate_up->ne[1]/2;
    L.gate_v  = ggml_view_3d(ctx, L.gate_up, width, L.gate_up->ne[1], L.gate_up->ne[2],
                             L.gate_up->nb[1], L.gate_up->nb[2], 0);
    L.up_v    = ggml_view_3d(ctx, L.gate_up, width, L.gate_up->ne[1], L.gate_up->ne[2],
                             L.gate_up->nb[1], L.gate_up->nb[2], width*L.gate_up->nb[0]);
    L.glu     = ggml_situ_split(ctx, L.gate_v, L.up_v, situ_beta, situ_lb);

    L.down = ggml_mul_mat_id(ctx, use_q5_down ? W.down_q5 : W.down, L.glu, L.ids);
    L.wmul = ggml_mul(ctx, L.down, L.weights);

    if (!gate_bonus) {
        const int start = rank < 0 ? first : rank*n_local;
        ggml_mul_mat_id_set_expert_range(L.gate_up, start, n_expert);
        ggml_mul_mat_id_set_expert_range(L.down, start, n_expert);
    }

    L.gf = ggml_new_graph(ctx);
    if (!computed_cur) {
        // Pass 1 reads glu on host to check non-owned zeroing; pass 2 leaves glu internal so TMA can skip zeroing.
        ggml_set_output(L.glu);
    }
    ggml_build_forward_expand(L.gf, L.weights);
    ggml_build_forward_expand(L.gf, L.cur);
    ggml_build_forward_expand(L.gf, L.wmul);
    for (int64_t i = 0; i < n_used; ++i) {
        ggml_tensor * v = ggml_view_2d(ctx, L.wmul, n_embd, n_tokens, L.wmul->nb[2], i*L.wmul->nb[1]);
        L.views.push_back(v);
        ggml_build_forward_expand(L.gf, v);
    }
    L.out = L.views[0];
    for (int64_t i = 1; i < n_used; ++i) {
        L.out = ggml_add(ctx, L.out, L.views[i]);
        L.adds.push_back(L.out);
        if (!computed_cur) {
            ggml_build_forward_expand(L.gf, L.out);
        }
    }
    if (gate_bonus) {
        ggml_build_forward_expand(L.gf, L.out);
        auto * pair = ggml_mul_mat_id(ctx, gate_bonus, input, L.ids);
        const int64_t w = gate_bonus->ne[1]/2;
        auto * g = ggml_view_3d(ctx, pair, w, n_used, n_tokens, pair->nb[1], pair->nb[2], 0);
        auto * u = ggml_view_3d(ctx, pair, w, n_used, n_tokens, pair->nb[1], pair->nb[2], w*pair->nb[0]);
        auto * d = ggml_mul_mat_id(ctx, down_bonus, ggml_situ_split(ctx, g, u, situ_beta, situ_lb), L.ids);
        ggml_mul_mat_id_set_strided_expert_range(pair, rank, n_expert, 4);
        ggml_mul_mat_id_set_strided_expert_range(d, rank, n_expert, 4);
        auto * weighted = ggml_mul(ctx, d, L.weights);
        ggml_build_forward_expand(L.gf, weighted);
        ggml_tensor * slots[n_used];
        for (int64_t i = 0; i < n_used; ++i) {
            slots[i] = ggml_view_2d(ctx, weighted, n_embd, n_tokens, weighted->nb[2], i*weighted->nb[1]);
            ggml_build_forward_expand(L.gf, slots[i]);
        }
        auto * sum = slots[0];
        for (int64_t i = 1; i < n_used; ++i) {
            sum = ggml_add(ctx, sum, slots[i]);
            ggml_build_forward_expand(L.gf, sum);
        }
        L.out = ggml_add(ctx, L.out, sum);
    }
    if (computed_cur || gate_bonus) {
        ggml_set_output(L.out);
        ggml_build_forward_expand(L.gf, L.out);
    }
    return L;
}

static void fill_tensor(ggml_tensor * t, unsigned char byte) {
    std::vector<char> s(ggml_nbytes(t), (char) byte);
    ggml_backend_tensor_set(t, s.data(), 0, s.size());
}

static bool is_tensor(ggml_tensor * t, unsigned char byte) {
    std::vector<char> s(ggml_nbytes(t));
    ggml_backend_tensor_get(t, s.data(), 0, s.size());
    for (const char c : s) {
        if ((unsigned char) c != byte) {
            return false;
        }
    }
    return true;
}

static void read_row(ggml_tensor * t, std::vector<int32_t> & dst, int64_t row, int64_t n) {
    ggml_backend_tensor_get(t, dst.data() + row*n, (size_t) row*t->nb[1], n*sizeof(int32_t));
}

static void cpu_router(const std::vector<float> & cur, const std::vector<float> & router, const std::vector<float> & bias,
        int64_t n_tokens, std::vector<float> & logits, std::vector<int32_t> & ids, std::vector<float> & weights) {
    logits.assign(n_expert*n_tokens, 0.0f);
    ids.assign(n_used*n_tokens, -1);
    weights.assign(n_used*n_tokens, 0.0f);
    std::vector<double> probs(n_expert);
    for (int64_t t = 0; t < n_tokens; ++t) {
        for (int64_t e = 0; e < n_expert; ++e) {
            double acc = 0.0;
            for (int64_t i = 0; i < n_embd; ++i) {
                acc += (double) cur[t*n_embd + i]*(double) router[e*n_embd + i];
            }
            logits[t*n_expert + e] = (float) acc;
            probs[e] = 1.0/(1.0 + exp(-acc));
        }
        double wsum = 0.0;
        for (int64_t k = 0; k < n_used; ++k) {
            int64_t best = -1;
            double  best_s = -INFINITY;
            for (int64_t e = 0; e < n_expert; ++e) {
                const double s = probs[e] + bias[e];
                if (s > best_s || (s == best_s && e < best)) {
                    best_s = s;
                    best   = e;
                }
            }
            ids[t*n_used + k]     = (int32_t) best;
            weights[t*n_used + k] = (float) probs[best];
            wsum += probs[best];
            probs[best] = -1.0;
        }
        for (int64_t k = 0; k < n_used; ++k) {
            weights[t*n_used + k] = (float) ((double) weights[t*n_used + k]/(wsum + 1e-6)*8.0);
        }
    }
}

static void quantize_q8_1_row(const float * x, int64_t n, std::vector<float> & qd, bool d4_float = false) {
    qd.resize(n);
    for (int64_t b = 0; b < n; b += 32) {
        float amax = 0.0f;
        for (int i = 0; i < 32; ++i) {
            amax = std::max(amax, std::fabs(x[b + i]));
        }
        if (d4_float) {
            const float d_inv = amax == 0.0f ? 0.0f : 127.0f / amax;
            const float d     = amax == 0.0f ? 0.0f : 1.0f / d_inv;
            for (int i = 0; i < 32; ++i) {
                const float q = amax == 0.0f ? 0.0f : std::round(x[b + i] * d_inv);
                qd[b + i] = q * d;
            }
        } else {
            const float d  = amax/127.0f;
            const float dh = ggml_fp16_to_fp32(ggml_fp32_to_fp16(d));
            for (int i = 0; i < 32; ++i) {
                const float q = amax == 0.0f ? 0.0f : std::round(x[b + i]/d);
                qd[b + i] = q*dh;
            }
        }
    }
}

static void row_to_float(ggml_tensor * w, int64_t expert, int64_t row, std::vector<float> & f) {
    const int64_t per = w->ne[0];
    std::vector<char> raw(ggml_row_size(w->type, per));
    ggml_backend_tensor_get(w, raw.data(), (size_t) expert*w->nb[2] + row*w->nb[1], raw.size());
    f.resize(per);
    ggml_get_type_traits(w->type)->to_float(raw.data(), f.data(), per);
}

static double dot(const std::vector<float> & a, const std::vector<float> & b) {
    double acc = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        acc += (double) a[i]*(double) b[i];
    }
    return acc;
}

static float situ(double gate, double up) {
    const double act = situ_beta*tanh(gate/situ_beta)/(1.0 + exp(-gate));
    return (float) (act*(situ_lb > 0.0f ? situ_lb*tanh(up/situ_lb) : up));
}

// Exact DS4 Q4_K/Q5_K dot matching quantize_mmq_few_store<DS4> + mmq_tma_vec_dot_q45_K.
static float dot_q45_k_ds4(ggml_tensor * w, int64_t expert, int64_t row, const float * x) {
    const bool is_q5 = w->type == GGML_TYPE_Q5_K;
    const size_t blk_bytes = is_q5 ? 176 : 144;
    const int64_t per = w->ne[0];
    const int64_t n_blocks = per / 256;
    std::vector<uint8_t> raw((size_t) n_blocks * blk_bytes);
    ggml_backend_tensor_get(w, raw.data(), (size_t) expert*w->nb[2] + row*w->nb[1], raw.size());
    float total = 0.0f;
    for (int64_t b = 0; b < n_blocks; ++b) {
        const uint8_t * blk = raw.data() + (size_t) b * blk_bytes;
        ggml_fp16_t d_h, m_h;
        memcpy(&d_h, blk + 0, sizeof(ggml_fp16_t));
        memcpy(&m_h, blk + 2, sizeof(ggml_fp16_t));
        const float d4 = ggml_fp16_to_fp32(d_h);
        const float m4 = ggml_fp16_to_fp32(m_h);
        const uint8_t * sc_b = blk + 4;
        const uint8_t * qh   = is_q5 ? (blk + 16) : nullptr;
        const uint8_t * qs   = is_q5 ? (blk + 48) : (blk + 16);
        float sum_d = 0.0f;
        float sum_m = 0.0f;
        for (int s = 0; s < 8; ++s) {
            const int sc_s = s < 4 ? (sc_b[s] & 63) : ((sc_b[s + 4] & 0xF) | ((sc_b[s - 4] >> 6) << 4));
            const int mn_s = s < 4 ? (sc_b[s + 4] & 63) : ((sc_b[s + 4] >> 4) | ((sc_b[s] >> 6) << 4));
            const float * xb = x + b * 256 + s * 32;
            float amax = 0.0f;
            for (int i = 0; i < 32; ++i) {
                amax = std::max(amax, std::fabs(xb[i]));
            }
            const float d_inv = amax == 0.0f ? 0.0f : 127.0f / amax;
            const float d_q   = amax == 0.0f ? 0.0f : 1.0f / d_inv;
            int sum_q = 0;
            int c_dot = 0;
            for (int i = 0; i < 32; ++i) {
                const int q_i = amax == 0.0f ? 0 : (int) std::round(xb[i] * d_inv);
                int q_w = (qs[32 * (s / 2) + i] >> (4 * (s % 2))) & 0xF;
                if (is_q5 && ((qh[i] >> s) & 1)) {
                    q_w += 16;
                }
                sum_q += q_i;
                c_dot += q_w * q_i;
            }
            const float ds_x = ggml_fp16_to_fp32(ggml_fp32_to_fp16(d_q));
            const float ds_y = ggml_fp16_to_fp32(ggml_fp32_to_fp16(d_q * (float) sum_q));
            sum_d += ds_x * float(c_dot * sc_s);
            sum_m += float(mn_s) * ds_y;
        }
        total += d4 * sum_d - m4 * sum_m;
    }
    return total;
}

static float cpu_weighted_sum_elem(const model_weights & W, const std::vector<float> & cur,
        const std::vector<int32_t> & ids, const std::vector<float> & weights, int64_t t, int64_t out_row,
        bool tma_d4 = false, bool use_q5_down = false) {
    ggml_tensor * w_down = use_q5_down ? W.down_q5 : W.down;
    std::vector<float> row_gate, row_up, row_qd, glu_s(n_ff), glu_qd, down_row;
    quantize_q8_1_row(cur.data() + t*n_embd, n_embd, row_qd, tma_d4);
    float acc = 0.0f;
    for (int64_t s = 0; s < n_used; ++s) {
        const int64_t e = ids[t*n_used + s] - first;
        float v = 0.0f;
        if (e >= 0 && e < n_local) {
            for (int64_t j = 0; j < n_ff; ++j) {
                row_to_float(W.gate_up, e, j,        row_gate);
                row_to_float(W.gate_up, e, n_ff + j, row_up);
                glu_s[j] = situ(dot(row_gate, row_qd), dot(row_up, row_qd));
            }
            if (tma_d4) {
                v = dot_q45_k_ds4(w_down, e, out_row, glu_s.data());
            } else {
                quantize_q8_1_row(glu_s.data(), n_ff, glu_qd, false);
                row_to_float(w_down, e, out_row, down_row);
                v = (float) dot(glu_qd, down_row);
            }
        }
        const float p = weights[t*n_used + s] * v;
        acc = s == 0 ? p : acc + p;
    }
    return acc;
}

static void bench_weight(ggml_tensor * t) {
    constexpr size_t period = 31;
    const int64_t k = t->ne[0];
    const size_t bytes = t->nb[1];
    std::vector<float> src(period*k);
    uint32_t seed = 8121;
    for (auto & v : src) {
        v = noise(seed)*0.025f;
    }
    std::vector<char> sample(period*bytes);
    ggml_quantize_chunk(t->type, src.data(), sample.data(), 0, period, k, nullptr);
    std::vector<char> chunk(period*64*bytes);
    for (size_t i = 0; i < 64; ++i) {
        memcpy(chunk.data() + i*sample.size(), sample.data(), sample.size());
    }
    for (size_t off = 0; off < ggml_nbytes(t); off += chunk.size()) {
        ggml_backend_tensor_set(t, chunk.data(), off, std::min(chunk.size(), ggml_nbytes(t) - off));
    }
}

static std::vector<int> bench_routes(int rows, bool clustered, std::mt19937 & rng) {
    std::vector<int> ids;
    ids.reserve(rows*n_used);
    std::array<std::vector<int>, 2> pools;
    const int phase = rng() % 2;
    const int quota[4] = {17, 13, 10, 8};
    for (int e = 0; e < n_expert; ++e) {
        const int rank = e/n_local, slot = e%n_local;
        const int common = (3*quota[rank] + 4)/5;
        const bool hot = clustered && (slot < common ||
            (slot >= common + phase*(quota[rank] - common) && slot < quota[rank] + phase*(quota[rank] - common)));
        pools[hot].push_back(e);
    }
    std::bernoulli_distribution seven(0.571217);
    for (int t = 0; t < rows; ++t) {
        const int n_hot = clustered ? 6 + seven(rng) : 0;
        for (int hot : {1, 0}) {
            auto & pool = pools[hot];
            const int n = hot ? n_hot : n_used - n_hot;
            for (int i = 0; i < n; ++i) {
                const int j = std::uniform_int_distribution<int>(i, int(pool.size()) - 1)(rng);
                std::swap(pool[i], pool[j]);
                ids.push_back(pool[i]);
            }
        }
    }
    return ids;
}

static int bench_shards(ggml_backend_t cuda) {
    constexpr int trials = 64;
    struct scenario {
        int rows;
        bool clustered;
        std::vector<std::vector<int>> ids;
        double times[2][4][trials] = {};
    };
    scenario work[] = {{23, false, {}}, {23, true, {}}, {512, false, {}}, {512, true, {}}};
    std::mt19937 rng(294);
    for (auto & s : work) {
        for (int i = 0; i < trials; ++i) {
            s.ids.push_back(bench_routes(s.rows, s.clustered, rng));
        }
    }
    for (int rotated = 0; rotated < 2; ++rotated) {
        ggml_context * ctx = ggml_init({12*ggml_tensor_overhead(), nullptr, true});
        model_weights w;
        w.router = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_expert);
        w.bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_expert);
        const int width = rotated ? 512 : n_ff;
        const int experts = rotated ? n_expert : n_local;
        w.gate_up = ggml_new_tensor_3d(ctx, GGML_TYPE_Q3_K, n_embd, 2*width, experts);
        w.down = ggml_new_tensor_3d(ctx, GGML_TYPE_Q4_K, width, n_embd, experts);
        auto * bg = rotated ? ggml_new_tensor_3d(ctx, GGML_TYPE_Q3_K, n_embd, 512, n_local) : nullptr;
        auto * bd = rotated ? ggml_new_tensor_3d(ctx, GGML_TYPE_Q4_K, 256, n_embd, n_local) : nullptr;
        auto * buf = ggml_backend_alloc_ctx_tensors(ctx, cuda);
        GGML_ASSERT(buf);
        ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        std::vector<float> router(n_expert*n_embd), bias(n_expert);
        for (int e = 0; e < n_expert; ++e) {
            router[e*n_embd + e] = 1.0f;
        }
        ggml_backend_tensor_set(w.router, router.data(), 0, ggml_nbytes(w.router));
        ggml_backend_tensor_set(w.bias, bias.data(), 0, ggml_nbytes(w.bias));
        for (auto * t : {w.gate_up, w.down, bg, bd}) {
            if (t) {
                bench_weight(t);
            }
        }
        // Reuse the synthetic weight arena; each rank runs alone on CUDA0.
        for (auto & s : work) {
            std::vector<float> input;
            fill_cur(s.rows, input);
            for (int rank = 0; rank < 4; ++rank) {
                layer l = build(w, s.rows, true, false, rank, bg, bd);
                auto * galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(cuda));
                GGML_ASSERT(ggml_gallocr_alloc_graph(galloc, l.gf));
                for (int trial = -3; trial < trials; ++trial) {
                    const auto & routes = s.ids[std::max(trial, 0)];
                    for (int r = 0; r < s.rows; ++r) {
                        std::fill_n(input.data() + r*n_embd, n_expert, -2.0f);
                        for (int i = 0; i < n_used; ++i) {
                            input[r*n_embd + routes[r*n_used + i]] = 3.0f - 0.1f*i;
                        }
                    }
                    ggml_backend_tensor_set(l.cur_in, input.data(), 0, ggml_nbytes(l.cur_in));
                    const auto start = std::chrono::steady_clock::now();
                    GGML_ASSERT(ggml_backend_graph_compute(cuda, l.gf) == GGML_STATUS_SUCCESS);
                    const auto stop = std::chrono::steady_clock::now();
                    if (trial >= 0) {
                        s.times[rotated][rank][trial] = std::chrono::duration<double, std::micro>(stop - start).count();
                    }
                }
                ggml_gallocr_free(galloc);
                ggml_free(l.ctx);
            }
        }
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }
    printf("layout rows routes draws mean_max_us mean_rank_us rank0_us rank1_us rank2_us rank3_us\n");
    for (const auto & s : work) {
        for (int rotated = 0; rotated < 2; ++rotated) {
            double max_sum = 0, avg_sum = 0, rank_sum[4] = {};
            for (int i = 0; i < trials; ++i) {
                double max_us = 0;
                for (int r = 0; r < 4; ++r) {
                    const double t = s.times[rotated][r][i];
                    max_us = std::max(max_us, t);
                    rank_sum[r] += t;
                    avg_sum += t/4;
                }
                max_sum += max_us;
            }
            printf("%s %d %s %d %.3f %.3f %.3f %.3f %.3f %.3f\n", rotated ? "rotated" : "contiguous", s.rows,
                s.clustered ? "clustered" : "uniform", trials, max_sum/trials, avg_sum/trials,
                rank_sum[0]/trials, rank_sum[1]/trials, rank_sum[2]/trials, rank_sum[3]/trials);
        }
    }
    return 0;
}

int main(int argc, char ** argv) {
    const bool bench = argc >= 2 && strcmp(argv[1], "--bench-shards") == 0;
    const bool bake = argc == 3 && strcmp(argv[2], "--router-bake") == 0;
    if (argc < 2 || (!bench && argc != 2 && !bake)) {
        printf("usage: %s <out.bin> [--router-bake] | --bench-shards\n", argv[0]);
        return 1;
    }
    ggml_backend_t dev_backend = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        auto * dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU && (!bench || strcmp(ggml_backend_dev_name(dev), "CUDA0") == 0)) {
            dev_backend = ggml_backend_dev_init(dev, nullptr);
            break;
        }
    }
    if (bench) {
        if (!dev_backend) {
            fprintf(stderr, "No CUDA0 backend for shard benchmark\n");
            return 1;
        }
        const int status = bench_shards(dev_backend);
        ggml_backend_free(dev_backend);
        return status;
    }
    if (!dev_backend) {
        if (bake) {
            fprintf(stderr, "CUDA device required for baked router fusion\n");
            return 1;
        }
        FILE * f = fopen(argv[1], "wb");
        if (f) { fclose(f); }
        printf("SKIP: no GPU device\n");
        return 0;
    }

    const bool fused_arm = [] {
        const char * env = getenv("GGML_CUDA_MOE_FUSE");
        return env && *env && strcmp(env, "0") != 0;
    }();
    const bool tma_arm = [] {
        const char * env = getenv("GGML_CUDA_MMQ_TMA");
        return env && *env && strcmp(env, "0") != 0;
    }();

    const int n_threads = std::min<int>(16, std::max<int>(1, (int) std::thread::hardware_concurrency()));
    model_weights W;
    if (!init_weights(W, dev_backend, n_threads, bake)) {
        printf("FAIL: no buffer for model weights\n");
        return 1;
    }

    FILE * f = fopen(argv[1], "wb");
    if (f == nullptr) {
        printf("FAIL: cannot write %s\n", argv[1]);
        return 1;
    }

    int failed = 0;

    // Pass 0: Q5_K down before Q4_K down (mirrors Inkling 2 layers 0-2 -> 3-47) so both fused Q5_K down
    // (including glu_sole_tma_down) and per-kernel >48 KiB SMEM opt-in ordering are exercised.
    for (int sub = 0; sub < 2; ++sub) {
        constexpr int64_t n_tokens = 23;
        const bool computed_cur = sub == 1;
        layer L = build(W, n_tokens, computed_cur, /*use_q5_down=*/true);
        ggml_backend_buffer_t buf = nullptr;
        ggml_gallocr_t galloc = nullptr;
        bool alloc_ok = false;
        if (!computed_cur) {
            buf = ggml_backend_alloc_ctx_tensors(L.ctx, dev_backend);
            alloc_ok = buf != nullptr;
        } else {
            galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(dev_backend));
            alloc_ok = ggml_gallocr_alloc_graph(galloc, L.gf);
        }
        if (!alloc_ok) {
            printf("FAIL: no buffer for Q5_K pass sub=%d\n", sub);
            if (galloc) { ggml_gallocr_free(galloc); }
            if (buf)    { ggml_backend_buffer_free(buf); }
            ggml_free(L.ctx);
            failed++;
            continue;
        }
        std::vector<float> cur_raw, cur_norm;
        fill_cur(n_tokens, cur_raw);
        ggml_backend_tensor_set(L.cur_in, cur_raw.data(), 0, cur_raw.size()*sizeof(float));

        const unsigned char sentinel = 0xA5;
        ggml_tensor * watched[] = { L.probs, L.gate_up, L.down, L.wmul, L.adds[0], L.adds[n_used - 3] };
        if (!computed_cur) {
            for (ggml_tensor * t : watched) {
                fill_tensor(t, sentinel);
            }
            fill_tensor(L.glu, sentinel);
            fill_tensor(L.out, sentinel);
        }

        if (ggml_backend_graph_compute(dev_backend, L.gf) != GGML_STATUS_SUCCESS) {
            printf("FAIL: Q5_K pass sub=%d compute failed\n", sub);
            failed++;
        } else {
            if (!computed_cur) {
                bool elided = true;
                for (ggml_tensor * t : watched) {
                    elided = elided && is_tensor(t, sentinel);
                }
                const bool dispatch_ok = elided == fused_arm;
                printf("q5_K %s%s arm: intermediates %s, expected %s: %s\n",
                    fused_arm ? "fused" : "unfused", tma_arm ? "+tma" : "",
                    elided ? "untouched" : "written", fused_arm ? "untouched" : "written", dispatch_ok ? "ok" : "FAIL");
                failed += !dispatch_ok;
            }
            std::vector<float> out(n_embd*n_tokens);
            ggml_backend_tensor_get(L.out, out.data(), 0, out.size()*sizeof(float));
            fwrite(out.data(), sizeof(float), out.size(), f);

            const std::vector<float> * cur_ref = &cur_raw;
            if (computed_cur) {
                cpu_rms_norm(cur_raw, n_tokens, 1e-5f, cur_norm);
                cur_ref = &cur_norm;
            }
            std::vector<float> c_logits, c_weights;
            std::vector<int32_t> c_ids;
            cpu_router(*cur_ref, W.h_router, W.h_bias, n_tokens, c_logits, c_ids, c_weights);
            const float expect_q5 = cpu_weighted_sum_elem(W, *cur_ref, c_ids, c_weights, 11, 777, tma_arm, /*use_q5_down=*/true);
            const double got_q5   = out[11*n_embd + 777];
            const double rel_q5   = std::fabs(got_q5 - expect_q5) / std::max(1e-6, std::fabs((double) expect_q5));
            const double tol_q5   = tma_arm ? 3e-3 : (computed_cur ? 5e-4 : 5e-5);
            printf("q5_K sub=%d down+sum: out[11][777] %f against %.6f, rel err %.2e: %s\n",
                sub, got_q5, (double) expect_q5, rel_q5, rel_q5 < tol_q5 ? "ok" : "FAIL");
            failed += rel_q5 >= tol_q5;
        }

        if (galloc) { ggml_gallocr_free(galloc); }
        if (buf)    { ggml_backend_buffer_free(buf); }
        ggml_free(L.ctx);
    }

    // Pass 1: 23 tokens with disjoint tensor buffers, checking intermediate sentinel elision,
    // cross-rank non-owned GLU zeroing, router exactness, SITU samples, and down+weighted sum.
    {
        constexpr int64_t n_tokens = 23;
        layer L = build(W, n_tokens, /*computed_cur=*/false);
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(L.ctx, dev_backend);
        if (buf == nullptr) {
            printf("FAIL: no buffer for pass 1 graph\n");
            fclose(f);
            return 1;
        }
        std::vector<float> cur;
        fill_cur(n_tokens, cur);
        ggml_backend_tensor_set(L.cur_in, cur.data(), 0, cur.size()*sizeof(float));

        const unsigned char sentinel = 0xA5;
        ggml_tensor * watched[] = { L.probs, L.gate_up, L.down, L.wmul, L.adds[0], L.adds[n_used - 3] };
        for (ggml_tensor * t : watched) {
            fill_tensor(t, sentinel);
        }
        fill_tensor(L.glu, sentinel);
        fill_tensor(L.out, sentinel);

        if (ggml_backend_graph_compute(dev_backend, L.gf) != GGML_STATUS_SUCCESS) {
            printf("FAIL: pass 1 compute failed\n");
            failed++;
        }

        std::vector<float>   logits(n_expert*n_tokens), weights(n_used*n_tokens), glu(n_ff*n_used*n_tokens), out(n_embd*n_tokens);
        std::vector<int32_t> ids(n_used*n_tokens, -1);
        ggml_backend_tensor_get(L.logits,  logits.data(),  0, logits.size()*sizeof(float));
        ggml_backend_tensor_get(L.weights, weights.data(), 0, weights.size()*sizeof(float));
        ggml_backend_tensor_get(L.glu,     glu.data(),     0, glu.size()*sizeof(float));
        ggml_backend_tensor_get(L.out,     out.data(),     0, out.size()*sizeof(float));
        for (int64_t t = 0; t < n_tokens; ++t) {
            read_row(L.ids, ids, t, n_used);
        }

        bool elided = true;
        for (ggml_tensor * t : watched) {
            elided = elided && is_tensor(t, sentinel);
        }
        const bool dispatch_ok = elided == fused_arm;
        printf("%s%s arm: probs/gate_up/down/mul/adds %s, expected %s: %s\n",
            fused_arm ? "fused" : "unfused", tma_arm ? "+tma" : "",
            elided ? "untouched" : "written", fused_arm ? "untouched" : "written", dispatch_ok ? "ok" : "FAIL");
        failed += !dispatch_ok;

        std::vector<float>   c_logits, c_weights;
        std::vector<int32_t> c_ids;
        cpu_router(cur, W.h_router, W.h_bias, n_tokens, c_logits, c_ids, c_weights);
        int    bad_ids     = 0;
        int    owned_cnt   = 0;
        int    unowned_cnt = 0;
        double max_rel     = 0.0;
        for (int64_t i = 0; i < n_used*n_tokens; ++i) {
            bad_ids += ids[i] != c_ids[i];
            const bool is_owned = c_ids[i] >= first && c_ids[i] < first + n_local;
            owned_cnt   += is_owned;
            unowned_cnt += !is_owned;
            max_rel = std::max(max_rel,
                std::fabs((double) weights[i] - c_weights[i]) / std::max(1e-9, std::fabs((double) c_weights[i])));
        }
        double max_logit = 0.0;
        for (int64_t i = 0; i < n_expert*n_tokens; ++i) {
            max_logit = std::max(max_logit,
                std::fabs((double) logits[i] - c_logits[i]) / std::max(1e-9, std::fabs((double) c_logits[i])));
        }
        const bool chain_ok = bad_ids == 0 && owned_cnt > 0 && unowned_cnt > 0 && max_rel < 1e-5 && max_logit < 1e-4;
        printf("chain: ids diff %d (owned %d, non-owned %d), weights err %.2e, logits err %.2e: %s\n",
            bad_ids, owned_cnt, unowned_cnt, max_rel, max_logit, chain_ok ? "ok" : "FAIL");
        failed += !chain_ok;

        // Verify every non-owned GLU row was zeroed (overwriting 0xA5) and sample owned SITU values.
        bool zero_ok = true;
        for (int64_t t = 0; t < n_tokens && zero_ok; ++t) {
            for (int64_t s = 0; s < n_used && zero_ok; ++s) {
                const int64_t e = ids[t*n_used + s] - first;
                if (e < 0 || e >= n_local) {
                    const float * row_ptr = glu.data() + (t*n_used + s)*n_ff;
                    for (int64_t j = 0; j < n_ff; ++j) {
                        if (row_ptr[j] != 0.0f) {
                            printf("FAIL: non-owned glu(%d,%d) expert %d col %d = %g\n",
                                (int) t, (int) s, (int) (e + first), (int) j, (double) row_ptr[j]);
                            zero_ok = false;
                            break;
                        }
                    }
                }
            }
        }
        std::vector<float> row_gate, row_up, row_qd;
        double max_glu = 0.0;
        int checked_owned = 0;
        for (int64_t t = 0; t < n_tokens && checked_owned < 8; ++t) {
            quantize_q8_1_row(cur.data() + t*n_embd, n_embd, row_qd, tma_arm);
            for (int64_t s = 0; s < n_used && checked_owned < 8; ++s) {
                const int64_t e = ids[t*n_used + s] - first;
                if (e < 0 || e >= n_local) {
                    continue;
                }
                const int64_t row = (checked_owned * 271 + 5) % n_ff;
                row_to_float(W.gate_up, e, row,        row_gate);
                row_to_float(W.gate_up, e, n_ff + row, row_up);
                const float expect = situ(dot(row_gate, row_qd), dot(row_up, row_qd));
                const float got    = glu[(t*n_used + s)*n_ff + row];
                max_glu = std::max(max_glu, std::fabs((double) got - expect) / std::max(1e-6, std::fabs((double) expect)));
                checked_owned++;
            }
        }
        const double glu_tol = 2e-4;
        const bool glu_ok = zero_ok && checked_owned == 8 && max_glu < glu_tol;
        printf("SITU: non-owned zero=%d, %d owned samples max rel err %.2e: %s\n",
            (int) zero_ok, checked_owned, max_glu, glu_ok ? "ok" : "FAIL");
        failed += !glu_ok;

        const float expect_out = cpu_weighted_sum_elem(W, cur, c_ids, c_weights, 11, 777, tma_arm);
        const double got_out   = out[11*n_embd + 777];
        const double rel_out   = std::fabs(got_out - expect_out) / std::max(1e-6, std::fabs((double) expect_out));
        const double out_tol   = tma_arm ? 3e-3 : 3e-5;
        printf("down+sum: out[11][777] %f against %.6f, rel err %.2e: %s\n",
            got_out, (double) expect_out, rel_out, rel_out < out_tol ? "ok" : "FAIL");
        failed += rel_out >= out_tol;

        fwrite(logits.data(),  sizeof(float),   logits.size(),  f);
        fwrite(ids.data(),     sizeof(int32_t), ids.size(),     f);
        fwrite(weights.data(), sizeof(float),   weights.size(), f);
        fwrite(glu.data(),     sizeof(float),   glu.size(),     f);
        fwrite(out.data(),     sizeof(float),   out.size(),     f);

        ggml_backend_buffer_free(buf);
        ggml_free(L.ctx);
    }

    // Pass 2: ggml_gallocr buffer reuse across n_tokens = 1..23 (exercises H1 aliasing of cur/ids/weights/glu
    // and C1 TMA Q3_K gate_up + Q4_K down across every token count 2..23 plus 1-token vector decode).
    {
        double worst_gallocr_rel = 0.0;
        bool gallocr_ok = true;
        for (int64_t n_tokens = 1; n_tokens <= 23; ++n_tokens) {
            layer L = build(W, n_tokens, /*computed_cur=*/true);
            ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(dev_backend));
            if (!ggml_gallocr_alloc_graph(galloc, L.gf)) {
                printf("FAIL: gallocr alloc failed for n_tokens=%d\n", (int) n_tokens);
                gallocr_ok = false;
                ggml_gallocr_free(galloc);
                ggml_free(L.ctx);
                break;
            }
            std::vector<float> cur_raw, cur_norm;
            fill_cur(n_tokens, cur_raw);
            ggml_backend_tensor_set(L.cur_in, cur_raw.data(), 0, cur_raw.size()*sizeof(float));

            if (ggml_backend_graph_compute(dev_backend, L.gf) != GGML_STATUS_SUCCESS) {
                printf("FAIL: gallocr compute failed for n_tokens=%d\n", (int) n_tokens);
                gallocr_ok = false;
                ggml_gallocr_free(galloc);
                ggml_free(L.ctx);
                break;
            }

            std::vector<float> out(n_embd * n_tokens);
            ggml_backend_tensor_get(L.out, out.data(), 0, out.size()*sizeof(float));
            fwrite(out.data(), sizeof(float), out.size(), f);

            for (float v : out) {
                if (!std::isfinite(v)) {
                    printf("FAIL: non-finite output at n_tokens=%d\n", (int) n_tokens);
                    gallocr_ok = false;
                    break;
                }
            }

            if (gallocr_ok && (n_tokens == 1 || n_tokens == 2 || n_tokens == 8 || n_tokens == 16 || n_tokens == 23)) {
                cpu_rms_norm(cur_raw, n_tokens, 1e-5f, cur_norm);
                std::vector<float>   c_logits, c_weights;
                std::vector<int32_t> c_ids;
                cpu_router(cur_norm, W.h_router, W.h_bias, n_tokens, c_logits, c_ids, c_weights);
                const int64_t t_chk   = n_tokens - 1;
                const int64_t row_chk = (n_tokens * 131) % n_embd;
                const float expect = cpu_weighted_sum_elem(W, cur_norm, c_ids, c_weights, t_chk, row_chk,
                    tma_arm && n_tokens >= 2);
                const double got   = out[t_chk*n_embd + row_chk];
                const double rel   = std::fabs(got - expect) / std::max(1e-6, std::fabs((double) expect));
                worst_gallocr_rel = std::max(worst_gallocr_rel, rel);
                const double tol = (tma_arm && n_tokens >= 2) ? 3e-3 : 5e-5;
                if (rel >= tol) {
                    printf("FAIL: gallocr n_tokens=%d out[%d][%d] %f vs %f (rel %.2e)\n",
                        (int) n_tokens, (int) t_chk, (int) row_chk, got, (double) expect, rel);
                    gallocr_ok = false;
                }
            }

            ggml_gallocr_free(galloc);
            ggml_free(L.ctx);
            if (!gallocr_ok) {
                break;
            }
        }
        printf("gallocr n=1..23: worst sampled rel err %.2e: %s\n",
            worst_gallocr_rel, gallocr_ok ? "ok" : "FAIL");
        failed += !gallocr_ok;
    }

    fclose(f);
    ggml_backend_buffer_free(W.buf);
    ggml_free(W.ctx);
    ggml_backend_free(dev_backend);
    printf("%s\n", failed ? "FAILED" : "PASSED");
    return failed ? 1 : 0;
}
