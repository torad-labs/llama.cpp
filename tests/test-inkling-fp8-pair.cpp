// test-inkling-fp8-pair — the e4m3 hi+lo pair algebra and its folds on the sm_120 device
// (ggml/src/ggml-cuda/fattn-rel-fp8-pair.cuh, GGML_CUDA_FATTN_FP8_ACCUM).
//
// The oracles are independent of the code under test:
//   - the host decodes every returned code byte with its own double-precision e4m3 construction, so the
//     device's integer encode/decode are held against plain arithmetic, not a mirror;
//   - the device's RNE code bytes are held against CUDA's own fp8 conversion (__nv_cvt_float_to_fp8,
//     __NV_E4M3, satfinite) at the pair's own scale, NVIDIA's implementation of the same OCP spec;
//   - the folds are held against double sums of the decoded slots, bounded against the sum of |terms|
//     (never against a near-zero sum, where a relative bound would be noise).
//
// Exact cases pin what a f64 bound cannot: a fold of one e4m3-representable value must return that value's
// own two-term encode bit for bit (its block scale is the value's own ceil), and the two-term carrier's
// aggregate error must beat hi alone by at least four - a fold that silently drops lo reds there even when
// its absolute error would still pass.
//
// Runs on a Blackwell device (cc 12.x); skips (77) everywhere else.

#include "ggml-cuda/fattn-rel-fp8-pair.cuh"

#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

static int g_failures = 0;

#define CHECK(cond, ...)                                            \
    do {                                                            \
        if (!(cond)) {                                              \
            if (g_failures < 30) {                                  \
                std::printf("FAIL %s:%d: ", __FILE__, __LINE__);    \
                std::printf(__VA_ARGS__);                           \
                std::printf("\n");                                  \
            }                                                       \
            ++g_failures;                                           \
        }                                                           \
    } while (0)

// ── the host's own e4m3 arithmetic (double precision, no device code involved) ───────────────────────────
static double host_e4m3(uint8_t b) {
    const int sign = (b >> 7) & 1;
    const int e = (b >> 3) & 0xF;
    const int m = b & 0x7;
    double v;
    if (e == 0) {
        v = (double) m * std::pow(2.0, -9.0);   // e4m3 subnormal
    } else {
        v = (1.0 + (double) m / 8.0) * std::pow(2.0, (double) e - 7.0);
    }
    return sign ? -v : v;
}

static double host_pair(uint8_t code, uint8_t sf) {
    return host_e4m3(code) * std::pow(2.0, (double) sf - 127.0);
}

static double host_two(uint32_t slot) {
    const uint8_t hi_code = (uint8_t) (slot & 0xFF);
    const uint8_t lo_code = (uint8_t) ((slot >> 8) & 0xFF);
    const uint8_t hi_sf = (uint8_t) ((slot >> 16) & 0xFF);
    const uint8_t lo_sf = (uint8_t) ((slot >> 24) & 0xFF);
    return host_pair(hi_code, hi_sf) + host_pair(lo_code, lo_sf);
}

// Half an e4m3 step at a code's own binade (e4 == 0 is the subnormal grid, step 2^-9 of the scale).
static double host_step(uint8_t code, uint8_t sf) {
    const int e4 = (code >> 3) & 0xF;
    const int exp = e4 == 0 ? -9 : e4 - 7 - 3;
    return std::pow(2.0, (double) sf - 127.0 + (double) exp);
}

// ── the device kernels ───────────────────────────────────────────────────────────────────────────────────
// k_encode: per value, the pair (packed code | sf << 8), its decode float, the two-term slot, and CUDA's
// own fp8 conversion of the value at the pair's scale.
__global__ void k_encode(const float * in, int n, uint32_t * out_pair, float * out_dec,
                         uint32_t * out_slot, uint8_t * out_cvt) {
#if defined(BLACKWELL_MMA_AVAILABLE)
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    const float v = in[i];
    const fattn_fp8_pair p = fattn_fp8_encode(v);
    out_pair[i] = (uint32_t) p.code | ((uint32_t) p.sf << 8);
    out_dec[i] = fattn_fp8_decode(p);
    out_slot[i] = fattn_fp8_slot_pack(fattn_fp8_encode_two(v));
    // NVIDIA's conversion of v at the pair's own scale: an independent implementation of the same RNE spec.
    const float q = ldexpf(v, 127 - (int) p.sf);
    out_cvt[i] = (uint8_t) __nv_cvt_float_to_fp8(q, __NV_SATFINITE, __NV_E4M3);
#else
    (void) in; (void) n; (void) out_pair; (void) out_dec; (void) out_slot; (void) out_cvt;
#endif
}

// k_block: one warp a block of 32 values; every lane the block-cooperative encode (one shared scale).
__global__ void k_block(const float * in, uint32_t * out_slot) {
#if defined(BLACKWELL_MMA_AVAILABLE)
    const int lane = threadIdx.x;
    const int w = blockIdx.x;
    const fattn_fp8_pair p = fattn_fp8_encode_block(in[32 * w + lane]);
    out_slot[32 * w + lane] = fattn_fp8_slot_pack(fattn_fp8_two{p, fattn_fp8_zero_pair()});
#else
    (void) in; (void) out_slot;
#endif
}

// k_fold32: one warp a group of 32 slots of one element; the block sum exits two-term.
__global__ void k_fold32(const uint32_t * slots, uint32_t * out) {
#if defined(BLACKWELL_MMA_AVAILABLE)
    __shared__ fattn_fp8_fold_stage stage;
    const int lane = threadIdx.x;
    const int w = blockIdx.x;
    const fattn_fp8_two r = fattn_fp8_fold32(slots[32 * w + lane], lane, stage);
    if (lane == 0) {
        out[w] = fattn_fp8_slot_pack(r);
    }
#else
    (void) slots; (void) out;
#endif
}

// k_fold_slots: one warp an element with n slot partials (groups of 32 then the group sums).
__global__ void k_fold_slots(const uint32_t * slots, int n, uint32_t * out) {
#if defined(BLACKWELL_MMA_AVAILABLE)
    __shared__ fattn_fp8_fold_stage stage;
    const int lane = threadIdx.x;
    const int w = blockIdx.x;
    const fattn_fp8_two r = fattn_fp8_fold_slots(slots + (size_t) w * n, n, lane, stage);
    if (lane == 0) {
        out[w] = fattn_fp8_slot_pack(r);
    }
#else
    (void) slots; (void) n; (void) out;
#endif
}

// k_terms: one warp an element; each lane sums N_TERMS one-term values of its own, each under its own scale.
template <int N_TERMS>
__global__ void k_terms(const float * values, uint32_t * out) {
#if defined(BLACKWELL_MMA_AVAILABLE)
    __shared__ fattn_fp8_fold_stage stage;
    const int lane = threadIdx.x;
    const int w = blockIdx.x;
    fattn_fp8_terms terms = {{0u, 0u, 0u, 0u}, {0u, 0u, 0u, 0u}};
#pragma unroll
    for (int t = 0; t < N_TERMS; ++t) {
        const float v = values[((size_t) w * 32 + lane) * N_TERMS + t];
        const fattn_fp8_pair p = fattn_fp8_encode(v);
        terms.code[t >> 2] |= (uint32_t) p.code << ((t & 3) * 8);
        terms.sf[t >> 2] |= (uint32_t) p.sf << ((t & 3) * 8);
    }
    const fattn_fp8_two r = fattn_fp8_sum_terms<N_TERMS>(terms, lane, stage);
    out[(size_t) w * 32 + lane] = fattn_fp8_slot_pack(r);
#else
    (void) values; (void) out;
#endif
}

// k_terms_two: each lane carries N_VALUES of its own as TWO terms a value (hi then lo), the design's case:
// every term is a pair under its own scale, so the fold's only rounding is its one exit encode.
template <int N_VALUES>
__global__ void k_terms_two(const float * values, uint32_t * out) {
#if defined(BLACKWELL_MMA_AVAILABLE)
    __shared__ fattn_fp8_fold_stage stage;
    const int lane = threadIdx.x;
    const int w = blockIdx.x;
    fattn_fp8_terms terms = {{0u, 0u, 0u, 0u}, {0u, 0u, 0u, 0u}};
#pragma unroll
    for (int i = 0; i < N_VALUES; ++i) {
        const float v = values[((size_t) w * 32 + lane) * N_VALUES + i];
        const fattn_fp8_two p = fattn_fp8_encode_two(v);
        const fattn_fp8_pair parts[2] = {p.hi, p.lo};
#pragma unroll
        for (int k = 0; k < 2; ++k) {
            const int t = 2 * i + k;
            terms.code[t >> 2] |= (uint32_t) parts[k].code << ((t & 3) * 8);
            terms.sf[t >> 2] |= (uint32_t) parts[k].sf << ((t & 3) * 8);
        }
    }
    const fattn_fp8_two r = fattn_fp8_sum_terms<2 * N_VALUES>(terms, lane, stage);
    out[(size_t) w * 32 + lane] = fattn_fp8_slot_pack(r);
#else
    (void) values; (void) out;
#endif
}

// ── the host driver ─────────────────────────────────────────────────────────────────────────────────────
static float frand(std::mt19937_64 & rng, double lo_exp, double hi_exp) {
    std::uniform_real_distribution<double> e(lo_exp, hi_exp);
    std::uniform_real_distribution<double> s(-1.0, 1.0);
    return (float) (s(rng) * std::pow(10.0, e(rng)));
}

int main() {
    int dev = 0;
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, dev) != cudaSuccess || prop.major * 100 + prop.minor < 1200) {
        std::printf("test-inkling-fp8-pair: no Blackwell device, skipped\n");
        return 77;
    }
    cudaSetDevice(dev);

    std::mt19937_64 rng(20261122ull);

    // ── the encode: RNE, the residual, and NVIDIA's own conversion ──
    {
        const int n = 4096;
        std::vector<float> in(n);
        for (int i = 0; i < n; ++i) {
            in[i] = frand(rng, -6.0, 3.0);
        }
        // exact corners: powers of two, the e4m3 saturation neighbourhood, and both zeros
        in[0] = 1.0f;   in[1] = -1.0f;  in[2] = 448.0f; in[3] = 449.0f;
        in[4] = 0.0f;   in[5] = -0.0f;  in[6] = 0.5f;   in[7] = -0.5f;
        in[8] = 1e-5f;  in[9] = -1e-5f; in[10] = 2.0f * 448.0f; in[11] = 1.75f;
        uint32_t * d_pair; float * d_dec; uint32_t * d_slot; uint8_t * d_cvt; float * d_in;
        cudaMalloc(&d_pair, n * sizeof(uint32_t)); cudaMalloc(&d_dec, n * sizeof(float));
        cudaMalloc(&d_slot, n * sizeof(uint32_t));  cudaMalloc(&d_cvt, n);
        cudaMalloc(&d_in, n * sizeof(float));
        cudaMemcpy(d_in, in.data(), n * sizeof(float), cudaMemcpyHostToDevice);
        k_encode<<<(n + 127) / 128, 128>>>(d_in, n, d_pair, d_dec, d_slot, d_cvt);
        CHECK(cudaDeviceSynchronize() == cudaSuccess, "k_encode launch: %s", cudaGetErrorString(cudaGetLastError()));
        std::vector<uint32_t> pair(n), slot(n);
        std::vector<float> dec(n);
        std::vector<uint8_t> cvt(n);
        cudaMemcpy(pair.data(), d_pair, n * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(dec.data(), d_dec, n * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(slot.data(), d_slot, n * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(cvt.data(), d_cvt, n, cudaMemcpyDeviceToHost);

        double err_hi_sum = 0.0, err_two_sum = 0.0;
        int live = 0;
        for (int i = 0; i < n; ++i) {
            const uint8_t code = (uint8_t) (pair[i] & 0xFF);
            const uint8_t sf = (uint8_t) ((pair[i] >> 8) & 0xFF);
            const double v = (double) in[i];
            // the device's integer decode, against the host's double construction: bit-exact
            const double hd = host_pair(code, sf);
            CHECK((double) dec[i] == hd, "decode mismatch at %d: code %02x sf %u device %a host %a", i, code, sf,
                  (double) dec[i], hd);
            // the RNE code, against CUDA's own conversion at the pair's scale
            CHECK(cvt[i] == code, "cvt mismatch at %d: v %a device %02x cvt %02x sf %u", i, v, code, cvt[i], sf);
            // hi is RNE: at most half an e4m3 step
            const double err_hi = std::fabs(v - hd);
            const double err_two = std::fabs(v - host_two(slot[i]));
            if (std::fabs(v) > 1e-30) {
                CHECK(err_hi <= host_step(code, sf) * (1.0 + 1e-12),
                      "hi error past half a step at %d: v %a err %a step %a", i, v, err_hi, host_step(code, sf));
                const uint8_t lo_code = (uint8_t) ((slot[i] >> 8) & 0xFF);
                const uint8_t lo_sf = (uint8_t) ((slot[i] >> 24) & 0xFF);
                const double lo_floor = lo_code == 0 ? 0.0 : host_step(lo_code, lo_sf) * (1.0 + 1e-9);
                CHECK(err_two <= lo_floor + 1e-30,
                      "two-term error past half a lo step at %d: v %a err %a lo_floor %a", i, v, err_two, lo_floor);
                err_hi_sum += err_hi;
                err_two_sum += err_two;
                ++live;
            }
        }
        CHECK(live > n / 2, "too few live values in the encode case");
        CHECK(err_two_sum <= err_hi_sum / 4.0,
              "hi+lo carries the residue: err_two %.3e vs err_hi %.3e", err_two_sum, err_hi_sum);
        std::printf("encode: %d values, mean hi err %.3e, mean two-term err %.3e\n", live,
                    err_hi_sum / live, err_two_sum / live);
        cudaFree(d_pair); cudaFree(d_dec); cudaFree(d_slot); cudaFree(d_cvt); cudaFree(d_in);
    }

    // ── the block-cooperative encode: ONE scale a warp, every lane inside its codes ──
    {
        const int warps = 128;
        std::vector<float> in(32 * warps);
        for (int i = 0; i < 32 * warps; ++i) {
            in[i] = frand(rng, -2.0, 1.5);
        }
        uint32_t * d_out; float * d_in;
        cudaMalloc(&d_out, 32 * warps * sizeof(uint32_t));
        cudaMalloc(&d_in, 32 * warps * sizeof(float));
        cudaMemcpy(d_in, in.data(), in.size() * sizeof(float), cudaMemcpyHostToDevice);
        k_block<<<warps, 32>>>(d_in, d_out);
        CHECK(cudaDeviceSynchronize() == cudaSuccess, "k_block launch: %s", cudaGetErrorString(cudaGetLastError()));
        std::vector<uint32_t> out(32 * warps);
        cudaMemcpy(out.data(), d_out, out.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        for (int w = 0; w < warps; ++w) {
            const uint8_t sf = (uint8_t) ((out[32 * w] >> 16) & 0xFF);
            double amax = 0.0;
            for (int l = 0; l < 32; ++l) {
                CHECK(((out[32 * w + l] >> 16) & 0xFF) == sf, "block %d lane %d carries scale %u, block %u",
                      w, l, (out[32 * w + l] >> 16) & 0xFF, sf);
                amax = std::fmax(amax, std::fabs((double) in[32 * w + l]));
            }
            // the ceil rule: amax <= 448 * 2^(sf-127), and below the next step down when sf > 0
            const double cap = 448.0 * std::pow(2.0, (double) sf - 127.0);
            CHECK(amax <= cap * (1.0 + 1e-6), "block %d amax %a over its scale's cap %a", w, amax, cap);
            if (sf > 1) {
                CHECK(amax > 448.0 * std::pow(2.0, (double) sf - 128.0),
                      "block %d scale is not the ceil: amax %a fits %u", w, amax, (unsigned) sf - 1);
            }
            for (int l = 0; l < 32; ++l) {
                const uint8_t code = (uint8_t) (out[32 * w + l] & 0xFF);
                const double v = (double) in[32 * w + l];
                CHECK(std::fabs(v - host_pair(code, sf)) <= host_step(code, sf) * (1.0 + 1e-12),
                      "block %d lane %d: v %a not within half a step at the shared scale", w, l, v);
            }
        }
        cudaFree(d_out); cudaFree(d_in);
    }

    // asks the device for the slot of each of n values (the encoder the fold cases consume)
    auto encode_slots = [&](const std::vector<float> & vals, std::vector<uint32_t> & out) {
        float * d_v; uint32_t * p2; float * d2; uint8_t * cv; uint32_t * d_slot;
        cudaMalloc(&d_v, vals.size() * sizeof(float));
        cudaMalloc(&p2, vals.size() * sizeof(uint32_t));
        cudaMalloc(&d2, vals.size() * sizeof(float));
        cudaMalloc(&cv, vals.size());
        cudaMalloc(&d_slot, vals.size() * sizeof(uint32_t));
        cudaMemcpy(d_v, vals.data(), vals.size() * sizeof(float), cudaMemcpyHostToDevice);
        k_encode<<<(int) (vals.size() + 127) / 128, 128>>>(d_v, (int) vals.size(), p2, d2, d_slot, cv);
        CHECK(cudaDeviceSynchronize() == cudaSuccess, "k_encode (slots) launch");
        out.resize(vals.size());
        cudaMemcpy(out.data(), d_slot, out.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaFree(d_v); cudaFree(p2); cudaFree(d2); cudaFree(cv); cudaFree(d_slot);
    };

    // ── the block sum: exact cases then the f64 bound ──
    {
        const int warps = 96;
        std::vector<uint32_t> slots(32 * warps, 0u);
        std::vector<double> ref(warps, 0.0), l1(warps, 0.0);

        // A (warps 0..31): ONE e4m3-representable lane - the fold returns that lane's own two-term
        // encode bit for bit (its block scale is the value's own ceil, and lo is its exact zero).
        for (int w = 0; w < 32; ++w) {
            const uint8_t code = (uint8_t) ((rng() & 1) << 7) |
                                 (uint8_t) (((rng() % 10) + 1) << 3) | (uint8_t) (rng() & 7);
            const float v = (float) host_e4m3(code);
            ref[w] = (double) v;
            l1[w] = std::fabs((double) v);
            std::vector<float> one(1, v);
            std::vector<uint32_t> s;
            encode_slots(one, s);
            slots[32 * w] = s[0];
        }

        // B (warps 32..63): 32 identical lanes - the sum is 32 times the same pair.
        {
            std::vector<float> vals(32 * 32);
            std::vector<uint32_t> s;
            for (int w = 0; w < 32; ++w) {
                const float v = frand(rng, -2.0, 1.0);
                ref[32 + w] = 32.0 * (double) v;
                l1[32 + w] = 32.0 * std::fabs((double) v);
                for (int l = 0; l < 32; ++l) {
                    vals[32 * w + l] = v;
                }
            }
            encode_slots(vals, s);
            for (size_t i = 0; i < s.size(); ++i) {
                slots[i + 32 * 32] = s[i];
            }
        }

        // C (warps 64..95): mixed magnitudes inside a couple of decades, the f64-bound case.
        {
            std::vector<float> vals(32 * 32);
            std::vector<uint32_t> s;
            for (int w = 0; w < 32; ++w) {
                for (int l = 0; l < 32; ++l) {
                    const float v = frand(rng, -2.0, 0.0);
                    vals[32 * w + l] = v;
                    ref[64 + w] += (double) v;
                    l1[64 + w] += std::fabs((double) v);
                }
            }
            encode_slots(vals, s);
            for (size_t i = 0; i < s.size(); ++i) {
                slots[i + 32 * 64] = s[i];
            }
        }

        uint32_t * d_slots; uint32_t * d_out;
        cudaMalloc(&d_slots, slots.size() * sizeof(uint32_t));
        cudaMalloc(&d_out, warps * sizeof(uint32_t));
        cudaMemcpy(d_slots, slots.data(), slots.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
        k_fold32<<<warps, 32>>>(d_slots, d_out);
        CHECK(cudaDeviceSynchronize() == cudaSuccess, "k_fold32 launch: %s", cudaGetErrorString(cudaGetLastError()));
        std::vector<uint32_t> out(warps);
        cudaMemcpy(out.data(), d_out, out.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaFree(d_slots); cudaFree(d_out);

        int exact_a = 0;
        double worst_b = 0.0, worst_c = 0.0;
        for (int w = 0; w < 32; ++w) {
            if (out[w] == slots[32 * w]) {
                ++exact_a;
            } else {
                CHECK(false, "fold32 single lane: slot %08x, fold %08x (v %a)", slots[32 * w], out[w], ref[w]);
            }
        }
        for (int w = 32; w < 64; ++w) {
            worst_b = std::fmax(worst_b, std::fabs(host_two(out[w]) - ref[w]) / std::fmax(l1[w], 1e-30));
            CHECK(std::fabs(host_two(out[w]) - ref[w]) <= l1[w] * std::pow(2.0, -6.0) + 1e-30,
                  "fold32 x32: got %a want %a", host_two(out[w]), ref[w]);
        }
        for (int w = 64; w < 96; ++w) {
            const double d = std::fabs(host_two(out[w]) - ref[w]);
            worst_c = std::fmax(worst_c, d / std::fmax(l1[w], 1e-30));
            CHECK(d <= l1[w] * std::pow(2.0, -6.0) + 1e-30, "fold32 mixed: got %a want %a", host_two(out[w]), ref[w]);
        }
        std::printf("fold32: %d/32 single-lane exact, x32 worst rel %.3e, mixed worst rel %.3e\n",
                    exact_a, worst_b, worst_c);
    }

    // ── the element fold over 256 partials (groups of 32, then the group sums) ──
    {
        const int warps = 32;
        const int n = 256;
        std::vector<float> vals((size_t) 32 * n);
        std::vector<double> ref(32, 0.0), l1(32, 0.0);
        std::vector<uint32_t> slots;
        for (int w = 0; w < 32; ++w) {
            for (int l = 0; l < n; ++l) {
                const float v = frand(rng, -2.0, 0.0);
                vals[(size_t) w * n + l] = v;
                ref[w] += (double) v;
                l1[w] += std::fabs((double) v);
            }
        }
        encode_slots(vals, slots);
        uint32_t * d_slots; uint32_t * d_out;
        cudaMalloc(&d_slots, slots.size() * sizeof(uint32_t));
        cudaMalloc(&d_out, warps * sizeof(uint32_t));
        cudaMemcpy(d_slots, slots.data(), slots.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
        k_fold_slots<<<warps, 32>>>(d_slots, n, d_out);
        CHECK(cudaDeviceSynchronize() == cudaSuccess, "k_fold_slots launch: %s", cudaGetErrorString(cudaGetLastError()));
        std::vector<uint32_t> out(warps);
        cudaMemcpy(out.data(), d_out, out.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        double worst = 0.0;
        for (int w = 0; w < 32; ++w) {
            const double d = std::fabs(host_two(out[w]) - ref[w]);
            worst = std::fmax(worst, d / std::fmax(l1[w], 1e-30));
            CHECK(d <= l1[w] * std::pow(2.0, -6.0) + 1e-30, "fold_slots 256: got %a want %a", host_two(out[w]), ref[w]);
        }
        std::printf("fold_slots(256): worst rel %.3e\n", worst);
        cudaFree(d_slots); cudaFree(d_out);
    }

    // ── the one-hot per-term fold: each term under its own scale ──
    // Two-term feeds first: every value rides as hi + lo, so the fold's only rounding is its exit encode
    // (the design's quality case: 4 values = 8 terms, 8 values = 16 terms, the capacity full).
    {
        const int value_cases[2] = {4, 8};
        for (int ci = 0; ci < 2; ++ci) {
            const int n_values = value_cases[ci];
            std::vector<float> vals((size_t) 32 * n_values);
            std::vector<double> ref(32, 0.0), l1(32, 0.0);
            for (int lane = 0; lane < 32; ++lane) {
                for (int t = 0; t < n_values; ++t) {
                    const float v = frand(rng, -2.0, 2.0);
                    vals[(size_t) lane * n_values + t] = v;
                    ref[lane] += (double) v;
                    l1[lane] += std::fabs((double) v);
                }
            }
            float * d_v; uint32_t * d_out;
            cudaMalloc(&d_v, vals.size() * sizeof(float));
            cudaMalloc(&d_out, 32 * sizeof(uint32_t));
            cudaMemcpy(d_v, vals.data(), vals.size() * sizeof(float), cudaMemcpyHostToDevice);
            if (n_values == 4) {
                k_terms_two<4><<<1, 32>>>(d_v, d_out);
            } else {
                k_terms_two<8><<<1, 32>>>(d_v, d_out);
            }
            CHECK(cudaDeviceSynchronize() == cudaSuccess, "k_terms_two launch: %s", cudaGetErrorString(cudaGetLastError()));
            std::vector<uint32_t> out(32);
            cudaMemcpy(out.data(), d_out, out.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost);
            double worst = 0.0;
            for (int lane = 0; lane < 32; ++lane) {
                const double d = std::fabs(host_two(out[lane]) - ref[lane]);
                worst = std::fmax(worst, d / std::fmax(l1[lane], 1e-30));
                CHECK(d <= l1[lane] * std::pow(2.0, -6.0) + 1e-30,
                      "sum_terms two-term (%d values) lane %d: got %a want %a", n_values, lane,
                      host_two(out[lane]), ref[lane]);
            }
            std::printf("sum_terms two-term (%d values, %2d terms): worst rel %.3e\n", n_values, 2 * n_values, worst);
            cudaFree(d_v); cudaFree(d_out);
        }
    }
    // One-term feeds: 16 single e4m3 values a lane (the full capacity). The fold itself is exact given its
    // terms, so the measured error is the terms' own 3-bit rounding - documented at 2^-3 of the l1 sum - and
    // this is the case a caller that cannot afford two terms an element accepts.
    {
        const int n_terms = 16;
        std::vector<float> vals((size_t) 32 * n_terms);
        std::vector<double> ref(32, 0.0), l1(32, 0.0);
        for (int lane = 0; lane < 32; ++lane) {
            for (int t = 0; t < n_terms; ++t) {
                const float v = frand(rng, -2.0, 2.0);
                vals[(size_t) lane * n_terms + t] = v;
                ref[lane] += (double) v;
                l1[lane] += std::fabs((double) v);
            }
        }
        float * d_v; uint32_t * d_out;
        cudaMalloc(&d_v, vals.size() * sizeof(float));
        cudaMalloc(&d_out, 32 * sizeof(uint32_t));
        cudaMemcpy(d_v, vals.data(), vals.size() * sizeof(float), cudaMemcpyHostToDevice);
        k_terms<16><<<1, 32>>>(d_v, d_out);
        CHECK(cudaDeviceSynchronize() == cudaSuccess, "k_terms launch: %s", cudaGetErrorString(cudaGetLastError()));
        std::vector<uint32_t> out(32);
        cudaMemcpy(out.data(), d_out, out.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        double worst = 0.0;
        for (int lane = 0; lane < 32; ++lane) {
            const double d = std::fabs(host_two(out[lane]) - ref[lane]);
            worst = std::fmax(worst, d / std::fmax(l1[lane], 1e-30));
            CHECK(d <= l1[lane] * std::pow(2.0, -3.0) + 1e-30,
                  "sum_terms one-term lane %d: got %a want %a", lane, host_two(out[lane]), ref[lane]);
        }
        std::printf("sum_terms one-term (16 terms): worst rel %.3e (the terms' own 3-bit rounding)\n", worst);
        cudaFree(d_v); cudaFree(d_out);
    }

    if (g_failures != 0) {
        std::printf("test-inkling-fp8-pair: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("test-inkling-fp8-pair: all checks passed\n");
    return 0;
}
