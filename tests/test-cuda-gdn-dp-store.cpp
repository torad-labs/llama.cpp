// Goal: under -sm dp, a KDA layer's recurrent-state snapshots reach the sequence owner's q8_0 cache straight from the
// recurrent kernel, as they do on one device, so a verify pays no f32 snapshot copy and no separate quantize: the cache
// holds the same bytes it would have held through them.
//
// On one device the snapshots are one strided cpy into the cache, which ggml_cuda_try_gdn_cache_fusion folds into the
// kernel. Under -sm dp build_rs_store_local writes each written slot on its own: a view of the slot, a cont of it and a
// set_rows into the cache at each sequence's own row (I64 ids, the owner's cells, in any order). That form was never
// folded, so every KDA layer of a 36-session verify on four RTX PRO 6000 cards made three f32 copies of 4 MiB a session
// and quantized each one again: about 12% of the 108-row target pass.
//
// The graph here is that form: gated_delta_net with KDA gates over n_seqs sequences of n_tokens rows with K snapshot
// slots, then for each written slot view -> cont -> set_rows into a q8_0 cache of K slots of mem_size rows at scattered
// rows. It is computed twice on the device, once as is and once with every cont a graph output, which keeps each cont
// and its set_rows. The two caches (every row, the unwritten ones too) and the two attention outputs must be equal byte
// for byte, and in the first graph each cont's buffer, filled with a sentinel before the compute, must still hold it:
// no snapshot was copied.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

constexpr int64_t S_v      = 128;
constexpr int64_t H        = 2;
constexpr int64_t D        = S_v * S_v * H;
constexpr int64_t mem_size = 8;
constexpr int64_t cells[]  = { 5, 1, 6 }; // each sequence's cell: rows are not the sequences' order nor contiguous

struct shape {
    int64_t n_tokens;
    int64_t n_seqs;
    int64_t K;
    bool    raw;
};

struct inputs {
    std::vector<float>   q, k, v, g, beta, state, a;
    std::vector<uint8_t> cache;
};

struct result {
    std::vector<uint8_t>              cache;
    std::vector<uint8_t>              attn;
    std::vector<std::vector<uint8_t>> conts;
    bool                              ok = false;
};

std::vector<float> uniform(std::mt19937 & rng, size_t n, float lo, float hi) {
    std::uniform_real_distribution<float> u(lo, hi);
    std::vector<float> out(n);
    for (auto & x : out) x = u(rng);
    return out;
}

inputs make_inputs(const shape & s) {
    std::mt19937 rng(1000 + 100*s.n_tokens + 10*s.n_seqs + s.K);
    inputs in;
    const size_t n_qk = S_v * H * s.n_tokens * s.n_seqs;
    in.q     = uniform(rng, n_qk, -1.0f, 1.0f);
    in.k     = uniform(rng, n_qk, -1.0f, 1.0f);
    in.v     = uniform(rng, n_qk, -1.0f, 1.0f);
    in.g     = s.raw ? uniform(rng, n_qk, -1.0f, 1.0f) : uniform(rng, n_qk, -0.5f, -0.01f);
    in.beta  = s.raw ? uniform(rng, H * s.n_tokens * s.n_seqs, -4.0f, 4.0f) : uniform(rng, H * s.n_tokens * s.n_seqs, 0.0f, 1.0f);
    in.state = uniform(rng, D * s.n_seqs, -0.1f, 0.1f);
    in.a     = uniform(rng, H, -8.0f, -0.05f);
    // every cache row starts as valid q8_0 blocks of its own, so a row written in the wrong place shows
    const size_t row = ggml_row_size(GGML_TYPE_Q8_0, D);
    in.cache.resize(row * mem_size * s.K);
    std::uniform_int_distribution<int> byte(-127, 127);
    for (size_t b = 0; b < in.cache.size() / 34; ++b) {
        const ggml_fp16_t d = ggml_fp32_to_fp16(0.001f * (float) (1 + b % 13));
        memcpy(&in.cache[b*34], &d, sizeof(d));
        for (int i = 0; i < 32; ++i) in.cache[b*34 + 2 + i] = (uint8_t) (int8_t) byte(rng);
    }
    return in;
}

result compute(ggml_backend_t backend, const shape & s, const inputs & in, bool conts_as_outputs) {
    result r;
    const int64_t n_written = std::min(s.n_tokens, s.K);

    ggml_init_params params = { ggml_tensor_overhead()*64 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(params);

    ggml_tensor * q     = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S_v, H, s.n_tokens, s.n_seqs);
    ggml_tensor * k     = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S_v, H, s.n_tokens, s.n_seqs);
    ggml_tensor * v     = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S_v, H, s.n_tokens, s.n_seqs);
    ggml_tensor * g     = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S_v, H, s.n_tokens, s.n_seqs);
    ggml_tensor * beta  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, H, s.n_tokens, s.n_seqs);
    ggml_tensor * state = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S_v, S_v, H, s.n_seqs);
    ggml_tensor * a     = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
    ggml_tensor * cache = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, D, mem_size * s.K);
    std::vector<ggml_tensor *> rows(n_written);
    for (auto & t : rows) t = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, s.n_seqs);

    ggml_tensor * qn  = ggml_l2_norm(ctx, q, 1e-6f);
    ggml_tensor * kn  = ggml_l2_norm(ctx, k, 1e-6f);
    ggml_tensor * gdn = ggml_gated_delta_net(ctx, qn, kn, v, g, beta, state, s.K);
    if (s.raw) {
        ggml_gated_delta_net_set_raw_kda_gates(gdn, a, -5.0f);
    }

    // build_recurrent_attn under -sm dp: one store per written slot, slot 0 the newest (build_rs_store_local)
    const size_t attn_bytes = ggml_row_size(GGML_TYPE_F32, S_v * H * s.n_tokens * s.n_seqs);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    std::vector<ggml_tensor *> conts(n_written);
    for (int64_t slot = 0; slot < n_written; ++slot) {
        ggml_tensor * states = ggml_view_2d(ctx, gdn, D, s.n_seqs, ggml_row_size(GGML_TYPE_F32, D),
                attn_bytes + slot * ggml_row_size(GGML_TYPE_F32, D * s.n_seqs));
        conts[slot] = ggml_cont_2d(ctx, states, D, s.n_seqs);
        if (conts_as_outputs) {
            ggml_set_output(conts[slot]);
        }
        ggml_build_forward_expand(gf, ggml_set_rows(ctx, cache, conts[slot], rows[slot]));
    }

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buf == nullptr) {
        ggml_free(ctx);
        return r;
    }
    ggml_backend_tensor_set(q, in.q.data(), 0, ggml_nbytes(q));
    ggml_backend_tensor_set(k, in.k.data(), 0, ggml_nbytes(k));
    ggml_backend_tensor_set(v, in.v.data(), 0, ggml_nbytes(v));
    ggml_backend_tensor_set(g, in.g.data(), 0, ggml_nbytes(g));
    ggml_backend_tensor_set(beta, in.beta.data(), 0, ggml_nbytes(beta));
    ggml_backend_tensor_set(state, in.state.data(), 0, ggml_nbytes(state));
    ggml_backend_tensor_set(a, in.a.data(), 0, ggml_nbytes(a));
    ggml_backend_tensor_set(cache, in.cache.data(), 0, ggml_nbytes(cache));
    for (int64_t slot = 0; slot < n_written; ++slot) {
        std::vector<int64_t> ids(s.n_seqs);
        for (int64_t seq = 0; seq < s.n_seqs; ++seq) ids[seq] = slot * mem_size + cells[seq];
        ggml_backend_tensor_set(rows[slot], ids.data(), 0, ggml_nbytes(rows[slot]));
        const std::vector<uint8_t> sentinel(ggml_nbytes(conts[slot]), 0xA5);
        ggml_backend_tensor_set(conts[slot], sentinel.data(), 0, sentinel.size());
    }

    if (ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS) {
        r.cache.resize(ggml_nbytes(cache));
        ggml_backend_tensor_get(cache, r.cache.data(), 0, r.cache.size());
        r.attn.resize(attn_bytes);
        ggml_backend_tensor_get(gdn, r.attn.data(), 0, attn_bytes);
        for (ggml_tensor * c : conts) {
            r.conts.emplace_back(ggml_nbytes(c));
            ggml_backend_tensor_get(c, r.conts.back().data(), 0, r.conts.back().size());
        }
        r.ok = true;
    }
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return r;
}

bool run(ggml_backend_t backend, const shape & s) {
    const inputs in   = make_inputs(s);
    const result fused = compute(backend, s, in, false);
    const result kept  = compute(backend, s, in, true);
    printf("  n_tokens %lld, n_seqs %lld, K %lld, %s gates: ", (long long) s.n_tokens, (long long) s.n_seqs,
            (long long) s.K, s.raw ? "raw" : "activated");
    if (!fused.ok || !kept.ok) {
        printf("FAILED, a compute failed\n");
        return false;
    }
    if (fused.cache != kept.cache) {
        size_t first = 0;
        while (fused.cache[first] == kept.cache[first]) ++first;
        printf("FAILED, the caches differ from byte %zu (row %zu)\n", first, first / ggml_row_size(GGML_TYPE_Q8_0, D));
        return false;
    }
    if (fused.attn != kept.attn) {
        printf("FAILED, the attention outputs differ\n");
        return false;
    }
    for (size_t slot = 0; slot < fused.conts.size(); ++slot) {
        for (const uint8_t b : fused.conts[slot]) {
            if (b != 0xA5) {
                printf("FAILED, slot %zu's snapshot was copied through f32 before its quantize\n", slot);
                return false;
            }
        }
    }
    printf("OK, the kernel wrote every snapshot into its sequence's row, the same bytes\n");
    return true;
}

} // namespace

int main(void) {
    ggml_backend_load_all();
    bool ok = true;
    int n_tested = 0;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU || strncmp(ggml_backend_dev_name(dev), "CUDA", 4) != 0) {
            continue;
        }
        ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
        printf("%s (%s):\n", ggml_backend_dev_name(dev), ggml_backend_dev_description(dev));
        // an MTP depth-2 verify (3 rows, 3 slots), a one-row decode, a verify shorter than the slots, one sequence an
        // owner (four sessions on four cards), and the activated gates
        for (const shape s : { shape{3, 3, 3, true}, shape{1, 3, 3, true}, shape{2, 3, 3, true}, shape{3, 1, 3, true},
                               shape{3, 3, 3, false} }) {
            ok = run(backend, s) && ok;
        }
        ggml_backend_free(backend);
        n_tested++;
    }
    if (n_tested == 0) {
        printf("no CUDA device, skipped\n");
        return 0;
    }
    printf("%s\n", ok ? "OK" : "FAILED");
    return ok ? 0 : 1;
}
