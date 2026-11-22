// Inkling 2's router chain (sigmoid, selection bias, top 8 of 384, the weights normalized with an epsilon, scaled) as one
// fused launch when ggml-alloc places the weights and ids over the logits. test-backend-ops gives every tensor its own
// memory, so none of its cases reaches that overlap; here the tensors are placed by hand: the ids over the logits, and
// the weights inside the logits of the last rows. At 9 to 32 rows (a 23-session decode) the backend must still fuse, in
// one block that reads every row before it writes any (GGML_CUDA_TOPK_MOE_WIDE_LEGACY=1 keeps the old rule, 8 rows).
// For each row count, on the first CUDA device:
//   1. the ids and the weights equal those of the unfused chain on the CPU;
//   2. the intermediates keep the sentinel they were given if and only if the chain was fused.
// A device-less run skips. Usage: test-inkling-f32-topk [rows ...]

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

static constexpr int64_t n_expert = 384;
static constexpr int     n_used   = 8;

struct chain {
    ggml_context * ctx = nullptr;
    ggml_cgraph *  gf  = nullptr;
    ggml_tensor *  input = nullptr;  // what the router's product would give
    ggml_tensor *  logits = nullptr;  // a node, as the router's product is, so the fusion checks its memory
    ggml_tensor *  bias = nullptr;
    ggml_tensor *  probs = nullptr;
    ggml_tensor *  argsort = nullptr; // the tensor under the ids view
    ggml_tensor *  ids = nullptr;
    ggml_tensor *  weights = nullptr; // the last node
};

static chain build(int64_t n_tokens) {
    chain c;
    ggml_init_params params = { 64*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    c.ctx = ggml_init(params);
    ggml_context * ctx = c.ctx;
    c.input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_expert, n_tokens);
    c.logits = ggml_scale(ctx, c.input, 1.0f);
    c.bias   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_expert);
    c.probs  = ggml_sigmoid(ctx, c.logits);
    ggml_tensor * selection = ggml_add(ctx, c.probs, c.bias);
    c.ids = ggml_argsort_top_k(ctx, selection, n_used);
    c.argsort = c.ids->view_src;
    ggml_tensor * w = ggml_get_rows(ctx, ggml_reshape_3d(ctx, c.probs, 1, n_expert, n_tokens), c.ids);
    w = ggml_reshape_2d(ctx, w, n_used, n_tokens);
    ggml_tensor * sum = ggml_scale_bias(ctx, ggml_sum_rows(ctx, w), 1.0f, 1e-6f);
    w = ggml_reshape_3d(ctx, ggml_div(ctx, w, sum), 1, n_used, n_tokens);
    c.weights = ggml_scale(ctx, w, 8.0f);
    c.gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(c.gf, c.weights);
    return c;
}

static size_t align(size_t n) {
    return (n + 255)/256*256;
}

int main(int argc, char ** argv) {
    std::vector<int64_t> rows;
    for (int i = 1; i < argc; ++i) {
        rows.push_back(atoi(argv[i]));
    }
    if (rows.empty()) {
        rows = { 1, 8, 9, 16, 17, 23, 32, 33 };
    }
    ggml_backend_t dev_backend = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        if (ggml_backend_dev_type(ggml_backend_dev_get(i)) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            dev_backend = ggml_backend_dev_init(ggml_backend_dev_get(i), nullptr);
            break;
        }
    }
    if (!dev_backend) {
        printf("SKIP: no GPU device\n");
        return 0;
    }
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    const char * legacy = getenv("GGML_CUDA_TOPK_MOE_WIDE_LEGACY");
    const bool wide = !(legacy && *legacy && strcmp(legacy, "0") != 0);

    int failed = 0;
    for (const int64_t t : rows) {
        std::mt19937 rng(1234 + t);
        std::normal_distribution<float> dist(0.0f, 2.0f);
        std::vector<float> logits(n_expert*t), bias(n_expert);
        for (auto & v : logits) { v = dist(rng); }
        for (auto & v : bias)   { v = 0.05f*dist(rng); }

        // the reference: the unfused chain on the CPU
        chain ref = build(t);
        ggml_backend_buffer_t ref_buf = ggml_backend_alloc_ctx_tensors(ref.ctx, cpu);
        ggml_backend_tensor_set(ref.input, logits.data(), 0, ggml_nbytes(ref.input));
        ggml_backend_tensor_set(ref.bias, bias.data(), 0, ggml_nbytes(ref.bias));
        ggml_backend_graph_compute(cpu, ref.gf);
        std::vector<float> ref_w(n_used*t);
        std::vector<int32_t> ref_ids(n_used*t);
        ggml_backend_tensor_get(ref.weights, ref_w.data(), 0, ggml_nbytes(ref.weights));
        for (int64_t r = 0; r < t; ++r) {
            ggml_backend_tensor_get(ref.ids, ref_ids.data() + r*n_used, r*ref.ids->nb[1], n_used*sizeof(int32_t));
        }

        // the chain on the device, placed by hand: over the logits, or each tensor on its own
        struct result { std::vector<float> w; std::vector<int32_t> ids; bool fused; };
        const auto run = [&](bool overlap) {
            chain c = build(t);
            const size_t row = n_expert*sizeof(float);
            size_t total = 2*align(row*t) + align(ggml_nbytes(c.bias)) + 4096;
            for (int i = 0; i < ggml_graph_n_nodes(c.gf); ++i) {
                total += ggml_graph_node(c.gf, i)->view_src ? 0 : align(ggml_nbytes(ggml_graph_node(c.gf, i)));
            }
            ggml_backend_buffer_t buf = ggml_backend_alloc_buffer(dev_backend, total);
            char * base = (char *) ggml_backend_buffer_get_base(buf);
            auto place = [&](ggml_tensor * x, size_t at) { ggml_backend_tensor_alloc(buf, x, base + at); };
            place(c.logits, 0);
            size_t off = align(row*t);
            place(c.input, off);
            off += align(row*t);
            place(c.bias, off);
            off += align(ggml_nbytes(c.bias));
            if (overlap) {
                place(c.argsort, 0); // the ids over the logits
                // the weights inside the logits of row t - 3 (a block that reads them later), clear of the ids' first bytes
                place(c.weights, size_t(std::max<int64_t>(t - 3, 0))*row + 64);
            }
            for (int i = 0; i < ggml_graph_n_nodes(c.gf); ++i) {
                ggml_tensor * n = ggml_graph_node(c.gf, i);
                if (n->buffer) {
                    continue;
                }
                if (n->view_src) {
                    ggml_backend_view_init(n);
                } else {
                    place(n, off);
                    off += align(ggml_nbytes(n));
                }
            }
            ggml_backend_tensor_set(c.input, logits.data(), 0, ggml_nbytes(c.input));
            ggml_backend_tensor_set(c.bias, bias.data(), 0, ggml_nbytes(c.bias));
            const std::vector<uint8_t> sentinel(ggml_nbytes(c.probs), 0xFF);
            ggml_backend_tensor_set(c.probs, sentinel.data(), 0, sentinel.size());
            ggml_backend_graph_compute(dev_backend, c.gf);

            result res = { std::vector<float>(n_used*t), std::vector<int32_t>(n_used*t), false };
            ggml_backend_tensor_get(c.weights, res.w.data(), 0, ggml_nbytes(c.weights));
            for (int64_t r = 0; r < t; ++r) {
                ggml_backend_tensor_get(c.ids, res.ids.data() + r*n_used, r*c.ids->nb[1], n_used*sizeof(int32_t));
            }
            std::vector<uint8_t> after(sentinel.size());
            ggml_backend_tensor_get(c.probs, after.data(), 0, after.size());
            res.fused = after == sentinel;
            ggml_backend_buffer_free(buf);
            ggml_free(c.ctx);
            return res;
        };
        const result over = run(true);
        const result apart = run(false);
        const std::vector<float> & w = over.w;
        const std::vector<int32_t> & ids = over.ids;
        const bool fused = over.fused;
        const bool expect = t <= 8 || (wide && t <= 32);

        int bad_ids = 0;
        double max_err = 0.0;
        for (int64_t i = 0; i < n_used*t; ++i) {
            bad_ids += ids[i] != ref_ids[i];
            max_err = std::max(max_err, (double) std::fabs(w[i] - ref_w[i])/std::max(1e-6, (double) std::fabs(ref_w[i])));
        }
        // over the logits or apart, a fused launch gives the same bits
        const bool same = !(fused && apart.fused) || (over.ids == apart.ids &&
            memcmp(over.w.data(), apart.w.data(), over.w.size()*sizeof(float)) == 0);
        const bool ok = bad_ids == 0 && max_err < 1e-4 && fused == expect && same;
        printf("rows %3d: %s, expected %s, ids differing %d, weights max rel err %.2e, %s apart: %s\n", (int) t,
            fused ? "fused" : "unfused", expect ? "fused" : "unfused", bad_ids, max_err,
            !(fused && apart.fused) ? "not compared" : same ? "bits equal" : "BITS DIFFER", ok ? "ok" : "FAIL");
        failed += !ok;

        ggml_backend_buffer_free(ref_buf);
        ggml_free(ref.ctx);
    }
    ggml_backend_free(dev_backend);
    ggml_backend_free(cpu);
    printf("%s\n", failed ? "FAILED" : "PASSED");
    return failed ? 1 : 0;
}
