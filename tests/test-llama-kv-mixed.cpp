// Section 3.5 skeleton fixture (doc docs/torad/research/inkling-2-small/18-global-kv-off-f16.md section 3.5): the
// dual-pool recency store behind LLAMA_INKLING2_KV_MIXED. A tiny global layer's K rows live in a q4_0 cold pool;
// stores route by recency into an F16 hot pool ringed over writes; reads merge exactly one source per cell.
//
// The judged property: on a fixture where every requested cell is forced to one side - all-hot with valid stamps,
// or all-cold with nothing near - the merged rows equal reading that pool alone bit for bit. Mixed windows serve
// each cell from exactly the source its stamp and recency name, so they read as the two forced fixtures put
// together. Eviction follows too: cells whose slots newer writes reused fall back to the cold row exactly, never
// to stale garbage.
//
// The oracle is the pure-CPU conversion chain itself: dequantize(quantize_q4_0(truth)) -> f32 -> F16 encode ->
// F16 decode equals what mirror wrote and merge returns, because both run ggml's reference quantizers. Runs as
// test-inkling-kvmix-{cpu,cuda0} following the kv1b naming pattern; argv picks which backend holds the pools.

#include "ggml.h"
#include "ggml-backend.h"
#include "../ggml/src/ggml-quants.h"

#include "../src/llama-io.h"
#include "../src/llama-kv-mixed.h"
#include "../src/llama-kv-state-migrate.h"

#include <algorithm>
#include <cmath>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

int failures = 0;

#define CHECK(cond, ...) do { \
    if (!(cond)) { printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); ++failures; } \
} while (0)

// deterministic row content spread across q4_0 boundaries, no zero rows
static void truth_row(uint32_t cell, int64_t n_values, float * out) {
    uint32_t s = 0x9e3779b9u ^ cell;
    auto next = [&s]() {
        s = s*1664525u + 1013904223u;
        return (float)(int32_t)s / 2147483648.0f;
    };
    for (int64_t j = 0; j < n_values; ++j) {
        float v = next()*6.0f;
        // a few values land near nibble boundaries on purpose
        if ((cell + j) % 11 == 0) {
            v = std::nearbyint(v*16.0f)/16.0f - 0.03125f;
        }
        out[j] = v;
    }
}

// what the mirror writes and any hot-side read must return for one cell: the reference chain over the cold bytes,
// run entirely on this host
static void oracle_hot_row(uint32_t cell, int64_t n_values, ggml_fp16_t * out) {
    std::vector<float> row(n_values);
    std::vector<uint8_t> enc(ggml_row_size(GGML_TYPE_Q4_0, n_values));
    truth_row(cell, n_values, row.data());
    quantize_row_q4_0_ref(row.data(), (block_q4_0 *) enc.data(), n_values);

    std::vector<float> dec(n_values);
    dequantize_row_q4_0((const block_q4_0 *) enc.data(), dec.data(), n_values);

    for (int64_t j = 0; j < n_values; ++j) {
        out[j] = ggml_fp32_to_fp16(dec[j]);
    }
}

// what a cold-side read of one cell must return
static void oracle_cold_row(uint32_t cell, int64_t n_values, float * out) {
    std::vector<uint8_t> enc(ggml_row_size(GGML_TYPE_Q4_0, n_values));
    truth_row(cell, n_values, out);
    quantize_row_q4_0_ref(out, (block_q4_0 *) enc.data(), n_values);
    dequantize_row_q4_0((const block_q4_0 *) enc.data(), out, n_values);
}

static ggml_backend_buffer_type_t pick_buft(const char * device_name) {
    const size_t n_devs = ggml_backend_dev_count();
    for (size_t i = 0; i < n_devs; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (std::string(device_name) == ggml_backend_dev_name(dev)) {
            return ggml_backend_dev_buffer_type(dev);
        }
    }
    printf("device '%s' not present\n", device_name);
    exit(77);
}

// one q4_0 cold pool of N cells holding truth rows, as fresh stores would leave it
struct cold_pool_t {
    ggml_context *    ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_tensor *     t   = nullptr;

    void build(ggml_backend_buffer_type_t buft, int64_t n_values, uint32_t n_cells) {
        ggml_init_params params = { 2*ggml_tensor_overhead(), nullptr, true };
        ctx = ggml_init(params);

        t = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, n_values, n_cells);
        buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);

        std::vector<float> row(n_values);
        std::vector<uint8_t> enc(ggml_row_size(GGML_TYPE_Q4_0, n_values));
        for (uint32_t c = 0; c < n_cells; ++c) {
            oracle_cold_row(c, n_values, row.data());
            quantize_row_q4_0_ref(row.data(), (block_q4_0 *) enc.data(), n_values);
            ggml_backend_tensor_set(t, enc.data(), (size_t) c*enc.size(), enc.size());
        }
    }

    ~cold_pool_t() {
        if (buf) {
            ggml_backend_buffer_free(buf);
        }
        if (ctx) {
            ggml_free(ctx);
        }
    }
};

} // namespace

int main(int argc, char ** argv) {
    const char * device_name = argc > 1 ? argv[1] : "CPU";

    llama_kv_mixed_hot_cells(); // knobs read once per process; the arms run under LLAMA_INKLING2_KV_MIXED=1
    CHECK(llama_kv_mixed_enabled(), "the arm must run with LLAMA_INKLING2_KV_MIXED=1");

    constexpr int64_t  n_values = 128;
    constexpr uint32_t n_cells  = 64;

    cold_pool_t cold;
    cold.build(pick_buft(device_name), n_values, n_cells);

    auto always_near = [](uint32_t) { return true; };
    auto never_near  = [](uint32_t) { return false; };

    std::vector<uint32_t> window_a = { 3, 11, 42, 57 };
    std::vector<uint32_t> window_b = { 5, 20, 44, 60 };
    std::vector<float> got(4*n_values);
    std::vector<float> expect(4*n_values);

    // all-cold fixture: nothing near and no stamps consulted - the merge is the cold pool read alone
    {
        llama_kv_mixed_pool pool;
        CHECK(pool.init(n_values, 16, ggml_backend_buffer_get_type(cold.buf)), "all-cold allocation failed");

        const uint32_t cells[4] = { 7, 9, 25, 62 };
        pool.merge_rows(cold.t, cells, 4, never_near, got.data());
        for (int i = 0; i < 4; ++i) {
            oracle_cold_row(cells[i], n_values, expect.data() + (size_t) i*n_values);
            CHECK(memcmp(&got[(size_t) i*n_values], &expect[(size_t) i*n_values], n_values*sizeof(float)) == 0,
                    "all-cold cell %d matches the single-pool reference bit for bit", cells[i]);
        }
    }

    // all-hot fixture: route exactly this window first, then demand it with everything near
    {
        llama_kv_mixed_pool pool;
        CHECK(pool.init(n_values, 8, ggml_backend_buffer_get_type(cold.buf)), "all-hot allocation failed");

        pool.mirror(cold.t, window_a.data(), window_a.size(), always_near);

        pool.merge_rows(cold.t, window_a.data(), window_a.size(), always_near, got.data());

        std::vector<ggml_fp16_t> hot_ref(n_values);
        for (size_t i = 0; i < window_a.size(); ++i) {
            oracle_hot_row(window_a[i], n_values, hot_ref.data());
            for (int64_t j = 0; j < n_values; ++j) {
                CHECK(ggml_fp16_to_fp32(hot_ref[j]) == got[(size_t) i*n_values + j],
                        "all-hot cell %d value %lld matches the hot pool's own write", window_a[i], (long long) j);
            }
        }
    }

    // mixed fixture: stamped near cells serve their F16 slot, unstamped ones read cold - both exact per cell
    {
        llama_kv_mixed_pool pool;
        CHECK(pool.init(n_values, 16, ggml_backend_buffer_get_type(cold.buf)), "mixed allocation failed");

        pool.mirror(cold.t, window_b.data(), window_b.size(), always_near);

        const uint32_t cells[4]  = { 5, 7, 20, 9 }; // 5 and 20 stamped; 7 and 9 untouched
        pool.merge_rows(cold.t, cells, 4,
                [](uint32_t c) { return c == 5 || c == 20; }, got.data());

        for (int i = 0; i < 4; ++i) {
            if (cells[i] == 5 || cells[i] == 20) {
                std::vector<ggml_fp16_t> hot_ref(n_values);
                oracle_hot_row(cells[i], n_values, hot_ref.data());
                for (int64_t j = 0; j < n_values; ++j) {
                    CHECK(ggml_fp16_to_fp32(hot_ref[j]) == got[(size_t) i*n_values + j],
                            "mixed cell %d serves its F16 slot", cells[i]);
                }
            } else {
                oracle_cold_row(cells[i], n_values, expect.data() + (size_t) i*n_values);
                CHECK(memcmp(&got[(size_t) i*n_values], &expect[(size_t) i*n_values], n_values*sizeof(float)) == 0,
                        "mixed cell %d reads cold exactly", cells[i]);
            }
        }
    }

    // eviction: a small ring overwritten by newer writes must fall back to the cold row, never stale F16
    {
        llama_kv_mixed_pool pool;
        CHECK(pool.init(n_values, 2, ggml_backend_buffer_get_type(cold.buf)), "eviction allocation failed");

        pool.mirror(cold.t, window_a.data(), window_a.size(), always_near); // two slots, four writes
        pool.merge_rows(cold.t, window_a.data(), window_a.size(), always_near, got.data());

        for (int i = 0; i < 4; ++i) {
            // only two slots live and both stamp the last two writers; the first two cells have no stamp left
            const bool stamped = i >= 2;
            if (stamped) {
                std::vector<ggml_fp16_t> hot_ref(n_values);
                oracle_hot_row(window_a[i], n_values, hot_ref.data());
                for (int64_t j = 0; j < n_values; ++j) {
                    CHECK(ggml_fp16_to_fp32(hot_ref[j]) == got[(size_t) i*n_values + j],
                            "evicted cell %d keeps its slot's exact values", window_a[i]);
                }
            } else {
                oracle_cold_row(window_a[i], n_values, expect.data() + (size_t) i*n_values);
                CHECK(memcmp(&got[(size_t) i*n_values], &expect[(size_t) i*n_values], n_values*sizeof(float)) == 0,
                        "evicted cell %d falls back to cold exactly", window_a[i]);
            }
        }
    }

    // ---- LLAMA_KV_STATE_MIGRATE units: block layouts, rotation composition, and the converted row reader ----

    // q4_0 handcrafted block in ggml's canonical layout: one F16 scale, then byte j carries value j in its low
    // nibble and value j+16 in its high one (quantize_row_q4_0_ref and the device set-rows quantizer agree). The
    // nibble is biased by 8, so an untouched 0x00 nibble reads as -8*d, here -16.
    {
        const uint8_t blk[18] = { 0x00, 0x40, 0xC3 }; // d=f16(2.0); qs[0]=0xC3: value 0 low 3 -> -10, value 16 high C -> +8
        float vals[32];
        llama_kv_state_row_decode(blk, GGML_TYPE_Q4_0, vals, 32);
        CHECK(vals[0] == -10.0f && vals[16] == 8.0f, "q4_0 byte 0 decodes value 0 low, value 16 high");
        for (int j = 1; j < 32; ++j) {
            if (j != 16) {
                CHECK(vals[j] == -16.0f, "q4_0 zero nibble at index %d reads as its bias floor", j);
            }
        }

        // encode through the helper equals the reference quantizer's bytes on a truth row
        std::vector<float> row(n_values);
        truth_row(9, n_values, row.data());
        std::vector<uint8_t> a(ggml_row_size(GGML_TYPE_Q4_0, n_values));
        std::vector<uint8_t> b(ggml_row_size(GGML_TYPE_Q4_0, n_values));
        llama_kv_state_row_encode(row.data(), GGML_TYPE_Q4_0, a.data(), n_values);
        quantize_row_q4_0_ref(row.data(), (block_q4_0 *) b.data(), n_values);
        CHECK(memcmp(a.data(), b.data(), a.size()) == 0, "helper q4_0 encode equals ggml's reference bytes");

        std::vector<ggml_fp16_t> h(n_values);
        llama_kv_state_row_encode(row.data(), GGML_TYPE_F16, (uint8_t *) h.data(), n_values);
        std::vector<ggml_fp16_t> h2(n_values);
        ggml_fp32_to_fp16_row(row.data(), h2.data(), n_values);
        CHECK(memcmp(h.data(), h2.data(), h.size()*2) == 0, "helper f16 encode equals ggml's row convert");

        std::vector<float> d(n_values);
        llama_kv_state_row_decode((const uint8_t *) h2.data(), GGML_TYPE_F16, d.data(), n_values);
        bool exact = true;
        for (int64_t j = 0; j < n_values; ++j) {
            exact = exact && ggml_fp16_to_fp32(ggml_fp32_to_fp16(d[j])) == d[j];
        }
        CHECK(exact, "f16 decode round-trips through the helper exactly");
    }

    // the hadamard registry matches the engine's own fill pattern and composes to identity through H^T
    {
        const auto & h64 = llama_kv_state_rot_matrix(64);

        bool symmetric = true;
        for (int i = 0; i < 64; ++i) {
            for (int j = 0; j < 64; ++j) {
                symmetric = symmetric && h64[i*64 + j] == h64[j*64 + i];
            }
        }
        CHECK(symmetric, "H is symmetric as the store-time mul_mat assumes");

        std::vector<float> v(64), fwd(64), back(64);
        for (int j = 0; j < 64; ++j) {
            v[j] = (float)(j % 7) - 3.0f;
        }
        llama_kv_state_rot_apply(h64.data(), 64, /*transposed*/ false, v.data(), fwd.data(), 64);
        llama_kv_state_rot_apply(h64.data(), 64, /*transposed*/ true, fwd.data(), back.data(), 64);
        float err = 0.0f;
        for (int j = 0; j < 64; ++j) {
            err = std::max(err, std::fabs(v[j] - back[j]));
        }
        CHECK(err < 1e-5f, "forward then transpose recovers the vector (%g)", (double) err);

        // a rotated q8_0 source into an unrotated f16 target lands where fresh tokens' plain values sit
        const auto & h128 = llama_kv_state_rot_matrix(128);
        std::vector<float> k(128);
        for (int j = 0; j < 128; ++j) {
            k[j] = 4.0f*(float)((j*13) % 17)/17.0f - 2.0f;
        }

        std::vector<uint8_t> enc(ggml_row_size(GGML_TYPE_Q8_0, 128));
        llama_kv_state_row_encode(k.data(), GGML_TYPE_Q8_0, enc.data(), 128);

        std::vector<float> rot(128), un(128);
        llama_kv_state_rot_apply(h128.data(), 128, false, k.data(), rot.data(), 128);   // as stores receive it
        llama_kv_state_rot_apply(h128.data(), 128, true,  rot.data(), un.data(), 128);  // H^T undoes it

        float rel = 0.0f;
        for (int j = 0; j < 128; ++j) {
            rel = std::max(rel, std::fabs(un[j] - k[j]));
        }
        CHECK(rel < 1e-5f, "q8_0-quantized rotated K un-rotates to its plain values (%g)", (double) rel);
    }

    // the converted row reader itself, over a mock stream with shuffled destination indices so consecutive runs
    // split: rows land where the strict reader would place them. Plain-source lands bit-for-bit at its target's
    // own encoding; the rotated source composes back to plain within one float rounding.
    {
        struct mem_io : llama_io_read_i {
            const uint8_t * p = nullptr;

            void read(void * dst, size_t size) override {
                memcpy(dst, p, size);
                p += size;
            }

            void read_tensor(ggml_tensor *, size_t, size_t) override {
                assert(false); // the converted path reads through host staging only
            }

            size_t n_bytes() override { return 0; }
        };

        constexpr int64_t  nv = 128;
        constexpr uint32_t n_c = 7;

        ggml_init_params ip = { 4*ggml_tensor_overhead(), nullptr, true };
        ggml_context * ctx2 = ggml_init(ip);

        // file order i maps to dest cell idxs[i]; the 0,1 pair keeps run grouping honest, everything else splits
        const uint32_t idxs[n_c] = { 5, 2, 4, 0, 1, 6, 3 };

        const auto & H = llama_kv_state_rot_matrix(64);

        for (int src_rotated = 0; src_rotated < 2; ++src_rotated) {
            const ggml_type   src_t     = src_rotated ? GGML_TYPE_Q4_0 : GGML_TYPE_F32;
            const ggml_type   dst_t     = src_rotated ? GGML_TYPE_F16  : GGML_TYPE_Q8_0;
            const size_t      src_row   = ggml_row_size(src_t, nv);
            const size_t      dst_row   = ggml_row_size(dst_t, nv);

            std::vector<uint8_t> stream(n_c*src_row);
            std::vector<float> row(nv), rt(nv);
            for (uint32_t i = 0; i < n_c; ++i) {
                truth_row(i + 40, nv, row.data());
                if (src_rotated) {
                    // what a rotated-on-store cache wrote: the forward rotation then the quantizer
                    llama_kv_state_rot_apply(H.data(), 64, false, row.data(), rt.data(), nv);
                    quantize_row_q4_0_ref(rt.data(), (block_q4_0 *) &stream[(size_t) i*src_row], nv);
                } else {
                    memcpy(&stream[(size_t) i*src_row], row.data(), (size_t) nv*sizeof(float));
                }
            }

            ggml_tensor * dst = ggml_new_tensor_2d(ctx2, dst_t, nv, n_c);
            ggml_backend_buffer_t dbuf = ggml_backend_alloc_ctx_tensors_from_buft(ctx2, pick_buft("CPU"));

            mem_io io;
            io.p = stream.data();

            CHECK(llama_kv_state_read_rows_converted(io, dst,
                        idxs, n_c, src_t, (uint64_t) src_row, nv, dst_t,
                        src_rotated ? H.data() : nullptr, src_rotated ? 64 : 0,
                        /*the target does not rotate*/ nullptr, 0),
                    "converted read succeeds (%s)", src_rotated ? "rotated" : "plain");

            std::vector<uint8_t> got(n_c*dst_row);
            ggml_backend_tensor_get(dst, got.data(), 0, n_c*dst_row);

            float worst = 0.0f;
            bool bit_exact = true;
            std::vector<float> plain(nv), rt2(nv), back(nv);
            for (uint32_t i = 0; i < n_c; ++i) {
                truth_row(i + 40, nv, plain.data());

                if (!src_rotated) {
                    std::vector<uint8_t> want(dst_row);
                    llama_kv_state_row_encode(plain.data(), dst_t, want.data(), nv);
                    const uint8_t * mine = &got[(size_t) idxs[i]*dst_row];
                    bit_exact = bit_exact && memcmp(want.data(), mine, dst_row) == 0;
                    continue;
                }

                // rotated path: the file's own bytes decoded and un-rotated must land near the plain values
                llama_kv_state_row_decode(&stream[(size_t) i*src_row], src_t, rt2.data(), nv);
                llama_kv_state_rot_apply(H.data(), 64, /*transposed*/ true, rt2.data(), back.data(), nv);

                std::vector<ggml_fp16_t> want_f(nv);
                ggml_fp32_to_fp16_row(back.data(), want_f.data(), nv);
                const ggml_fp16_t * mine = (const ggml_fp16_t *) &got[(size_t) idxs[i]*dst_row];
                for (int64_t j = 0; j < nv; ++j) {
                    worst = std::max(worst,
                            std::fabs(ggml_fp16_to_fp32(want_f[j]) - ggml_fp16_to_fp32(mine[j])));
                }
            }

            if (!src_rotated) {
                CHECK(bit_exact, "plain f32 rows re-encode to q8_0 byte for byte at every destination index");
            } else {
                CHECK(worst <= 4.0f/127.0f + 1e-3f,
                        "rotated q4_0 rows un-rotate into f16 within one q4 step (%g)", (double) worst);
            }

            ggml_backend_buffer_free(dbuf);
        }

        ggml_free(ctx2);
    }

    printf("%s: %s\n", device_name, failures ? "FAILED" : "OK");
    return failures ? 1 : 0;
}
