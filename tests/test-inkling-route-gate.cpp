#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

namespace {
constexpr int hidden = 4608;
constexpr int ff = 2304;
constexpr int experts = 384;
constexpr int slots = 8;
constexpr int local_first = 96;
constexpr int local_count = 96;

struct test_graph {
    ggml_context * ctx = ggml_init({64*ggml_tensor_overhead() + ggml_graph_overhead_custom(64, false), nullptr, true});
    ggml_backend_buffer_t buf = nullptr;

    ~test_graph() {
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }
};

void fill_experts(ggml_tensor * t, int64_t active_k = 0) {
    const int64_t k = active_k ? active_k : t->ne[0];
    const int64_t m = t->ne[1];
    const size_t row_bytes = ggml_row_size(t->type, t->ne[0]);
    std::vector<float> row(k);
    std::vector<unsigned char> data(row_bytes*m);

    for (int expert = 0; expert < t->ne[2]; ++expert) {
        for (int j = 0; j < 4; ++j) {
            for (int64_t i = 0; i < k; ++i) {
                row[i] = float((i*13 + expert*7 + j*5) % 43 - 21)/256;
            }
            ggml_quantize_chunk(t->type, row.data(), data.data() + j*row_bytes, 0, 1, k, nullptr);
        }
        for (int64_t j = 4; j < m; ++j) {
            std::copy_n(data.data() + (j % 4)*row_bytes, row_bytes, data.data() + j*row_bytes);
        }
        ggml_backend_tensor_set(t, data.data(), expert*data.size(), data.size());
    }
}

bool run_step(ggml_backend_t backend, FILE * out, ggml_tensor * gate, ggml_tensor * down, int rows, bool partitioned) {
    test_graph step;
    auto * ctx = step.ctx;
    auto * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hidden, 1, rows);
    auto * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, slots, rows);
    auto * mm = ggml_mul_mat_id(ctx, gate, x, ids);
    auto * g = ggml_view_3d(ctx, mm, ff, slots, rows, mm->nb[1], mm->nb[2], 0);
    auto * u = ggml_view_3d(ctx, mm, ff, slots, rows, mm->nb[1], mm->nb[2], ff*sizeof(float));
    auto * glu = ggml_situ_split(ctx, g, u, 1.0f, 0.0f);
    auto * y = ggml_mul_mat_id(ctx, down, glu, ids);
    if (partitioned) {
        ggml_mul_mat_id_set_expert_range(mm, local_first, experts);
        ggml_mul_mat_id_set_expert_range(y, local_first, experts);
    }
    auto * gf = ggml_new_graph_custom(ctx, 32, false);
    ggml_build_forward_expand(gf, y);
    step.buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!step.buf) {
        return false;
    }

    std::vector<float> activation(ggml_nelements(x));
    for (size_t i = 0; i < activation.size(); ++i) {
        activation[i] = float(int((i*37 + i/hidden*5) % 59) - 29)/64;
    }
    ggml_backend_tensor_set(x, activation.data(), 0, ggml_nbytes(x));
    for (const int active : {8, experts}) {
        std::vector<int32_t> routes(rows*slots);
        std::vector<bool> seen(experts);
        for (int t = 0; t < rows; ++t) {
            for (int s = 0; s < slots; ++s) {
                const bool empty_local = partitioned && active == slots;
                const int expert = empty_local ? local_first + local_count + s :
                    (31 + ((slots*t + s) % active)*37) % experts;
                routes[t*slots + s] = expert;
                seen[expert] = true;
            }
        }
        ggml_backend_tensor_set(ids, routes.data(), 0, ggml_nbytes(ids));
        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
            return false;
        }

        constexpr int reps = 5;
        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < reps; ++i) {
            if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
                return false;
            }
        }
        const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count()/reps;
        const auto occupied = partitioned ? std::count(seen.begin() + local_first, seen.begin() + local_first + local_count, true) :
            std::count(seen.begin(), seen.end(), true);
        printf("%d rows, %td occupied %sexperts, %s down: %.1f us/step\n",
            rows, occupied, partitioned ? "local " : "", ggml_type_name(down->type), us);

        std::vector<float> result(ggml_nelements(y));
        ggml_backend_tensor_get(y, result.data(), 0, ggml_nbytes(y));
        const bool nonzero = std::any_of(result.begin(), result.end(), [](float v) { return v != 0.0f; });
        if (!std::all_of(result.begin(), result.end(), [](float v) { return std::isfinite(v); }) ||
                nonzero != (occupied != 0) ||
                fwrite(result.data(), sizeof(float), result.size(), out) != result.size()) {
            return false;
        }
    }
    return true;
}

bool run(ggml_backend_t backend, FILE * out) {
    test_graph weights;
    auto * gate = ggml_new_tensor_3d(weights.ctx, GGML_TYPE_Q3_K, hidden, 2*ff, experts);
    auto * down = ggml_new_tensor_3d(weights.ctx, GGML_TYPE_Q4_K, ff, hidden, experts);
    auto * local_gate = ggml_view_3d(weights.ctx, gate, hidden, 2*ff, local_count,
        gate->nb[1], gate->nb[2], local_first*gate->nb[2]);
    auto * local_down = ggml_new_tensor_3d(weights.ctx, GGML_TYPE_Q5_K, ff, hidden, local_count);
    ggml_set_name(gate, "blk.0.ffn_gate_up_exps.weight");
    ggml_set_name(down, "blk.0.ffn_down_exps.weight");
    ggml_set_name(local_gate, gate->name);
    ggml_set_name(local_down, down->name);
    weights.buf = ggml_backend_alloc_ctx_tensors(weights.ctx, backend);
    if (!weights.buf) {
        return false;
    }
    ggml_backend_buffer_set_usage(weights.buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    fill_experts(gate);
    fill_experts(down);
    fill_experts(local_down);

    for (const int rows : {23, 128, 512}) {
        if (!run_step(backend, out, gate, down, rows, false)) {
            return false;
        }
    }
    return run_step(backend, out, local_gate, local_down, 128, true);
}
double nmse(const std::vector<float> & actual, const std::vector<float> & expected) {
    double sqerr = 0.0;
    double sqref = 0.0;
    for (size_t i = 0; i < actual.size(); ++i) {
        const double err = double(actual[i]) - expected[i];
        sqerr += err*err;
        sqref += double(expected[i])*expected[i];
    }
    return sqref > 0.0 ? sqerr/sqref : std::numeric_limits<double>::infinity();
}

struct wide_graph {
    test_graph storage;
    ggml_cgraph * graph;
    ggml_tensor * input;
    ggml_tensor * router;
    ggml_tensor * bias;
    ggml_tensor * output;

    wide_graph(ggml_backend_t backend, ggml_tensor * gate, ggml_tensor * down, int rows) {
        auto * ctx = storage.ctx;
        const int64_t n_ff = gate->ne[1]/2;
        const bool partitioned = gate->ne[2] != experts;
        graph = ggml_new_graph_custom(ctx, 64, false);
        input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, rows);
        router = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, experts);
        bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, experts);

        auto * logits = ggml_mul_mat(ctx, router, input);
        auto * probs = ggml_sigmoid(ctx, logits);
        auto * prob_rows = ggml_reshape_3d(ctx, probs, 1, experts, rows);
        ggml_build_forward_expand(graph, prob_rows);
        auto * scores = ggml_add(ctx, probs, bias);
        auto * ids = ggml_argsort_top_k(ctx, scores, slots);
        auto * w = ggml_get_rows(ctx, prob_rows, ids);
        w = ggml_reshape_2d(ctx, w, slots, rows);
        auto * sum = ggml_clamp(ctx, ggml_sum_rows(ctx, w), 6.103515625e-5f, INFINITY);
        w = ggml_reshape_3d(ctx, ggml_div(ctx, w, sum), 1, slots, rows);
        ggml_build_forward_expand(graph, w);

        auto * x = ggml_reshape_3d(ctx, input, hidden, 1, rows);
        auto * mm = ggml_mul_mat_id(ctx, gate, x, ids);
        auto * g = ggml_view_3d(ctx, mm, n_ff, slots, rows, mm->nb[1], mm->nb[2], 0);
        auto * u = ggml_view_3d(ctx, mm, n_ff, slots, rows, mm->nb[1], mm->nb[2], n_ff*sizeof(float));
        auto * glu = ggml_situ_split(ctx, g, u, 1.0f, 0.0f);

        if (n_ff != down->ne[0]) {
            glu = ggml_pad(ctx, glu, down->ne[0] - n_ff, 0, 0, 0);
        }
        auto * d = ggml_mul_mat_id(ctx, down, glu, ids);
        if (partitioned) {
            ggml_mul_mat_id_set_expert_range(mm, local_first, experts);
            ggml_mul_mat_id_set_expert_range(d, local_first, experts);
        }
        auto * weighted = ggml_mul(ctx, d, w);
        ggml_build_forward_expand(graph, weighted);
        ggml_tensor * views[slots];
        for (int s = 0; s < slots; ++s) {
            views[s] = ggml_view_2d(ctx, weighted, hidden, rows, weighted->nb[2], s*weighted->nb[1]);
            ggml_build_forward_expand(graph, views[s]);
        }
        output = views[0];
        for (int s = 1; s < slots; ++s) {
            output = ggml_add(ctx, output, views[s]);
        }
        ggml_build_forward_expand(graph, output);
        storage.buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    }

    void set_inputs(const float * x, const float * w, const float * b) {
        ggml_backend_tensor_set(input, x, 0, ggml_nbytes(input));
        ggml_backend_tensor_set(router, w, 0, ggml_nbytes(router));
        ggml_backend_tensor_set(bias, b, 0, ggml_nbytes(bias));
    }
};

bool run_wide_step(ggml_backend_t backend, FILE * out, FILE * ref, ggml_tensor * gate, ggml_tensor * down,
                   ggml_tensor * reference_down, int rows, bool spread) {
    wide_graph step(backend, gate, down, rows);
    if (!step.storage.buf) {
        return false;
    }
    const int64_t n_ff = down->ne[0];
    const int first = gate->ne[2] != experts ? local_first : 0;

    std::vector<float> activation(hidden*rows);
    std::vector<float> route_weights(hidden*experts);
    std::vector<float> route_bias(experts);
    for (int e = 0; e < experts; ++e) {
        route_bias[e] = e*0.0001f;
    }
    for (int t = 0; t < rows; ++t) {
        activation[t*hidden + t] = 1.0f;
        for (int k = rows; k < hidden; ++k) {
            activation[t*hidden + k] = float((k*37 + t*5) % 59 - 29)/64;
        }
        for (int s = 0; s < slots; ++s) {
            const int e = spread ? (31 + (slots*t + s)*37) % experts : first + s;
            route_weights[e*hidden + t] = 1.0f + s*0.125f;
        }
    }
    step.set_inputs(activation.data(), route_weights.data(), route_bias.data());

    for (int i = 0; i < 3; ++i) {
        if (ggml_backend_graph_compute(backend, step.graph) != GGML_STATUS_SUCCESS) {
            return false;
        }
    }
    constexpr int reps = 20;
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < reps; ++i) {
        if (ggml_backend_graph_compute(backend, step.graph) != GGML_STATUS_SUCCESS) {
            return false;
        }
    }
    const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count()/reps;
    printf("wide rows=%d local=%lld ff=%lld down=%s routes=%s us/step=%.1f",
        rows, (long long) gate->ne[2], (long long) n_ff, ggml_type_name(down->type), spread ? "spread" : "repeat", us);

    auto * y = step.output;
    std::vector<float> result(ggml_nelements(y));
    ggml_backend_tensor_get(y, result.data(), 0, ggml_nbytes(y));
    bool ok = std::all_of(result.begin(), result.end(), [](float v) { return std::isfinite(v); }) &&
        std::any_of(result.begin(), result.end(), [](float v) { return v != 0.0f; });
    if (ref) {
        std::vector<float> expected(result.size());
        if (fread(expected.data(), sizeof(float), expected.size(), ref) != expected.size()) {
            return false;
        }
        const bool exact = memcmp(result.data(), expected.data(), ggml_nbytes(y)) == 0;
        printf(" legacy_nmse=%.8g exact=%s", nmse(result, expected), exact ? "yes" : "no");
        if (rows <= 32) {
            ok &= exact;
        } else {
            // Use the old one-word route packing on balanced chunks of at most 32 rows.
            std::vector<float> reference(result.size());
            const int parts = (rows + 31)/32;
            for (int part = 0; part < parts; ++part) {
                const int begin = part*rows/parts;
                const int end = (part + 1)*rows/parts;
                wide_graph narrow(backend, gate, reference_down, end - begin);
                if (!narrow.storage.buf) {
                    return false;
                }
                narrow.set_inputs(activation.data() + begin*hidden, route_weights.data(), route_bias.data());
                if (ggml_backend_graph_compute(backend, narrow.graph) != GGML_STATUS_SUCCESS) {
                    return false;
                }
                ggml_backend_tensor_get(narrow.output, reference.data() + begin*hidden, 0, ggml_nbytes(narrow.output));
            }
            const double error = nmse(result, reference);
            printf(" ref_nmse=%.8g", error);
            ok &= error < 1e-5;
        }
    }
    printf(" %s\n", ok ? "OK" : "FAIL");
    return ok && fwrite(result.data(), sizeof(float), result.size(), out) == result.size();
}

bool run_wide(ggml_backend_t backend, FILE * out, FILE * ref) {
    bool ok = true;
    for (const int n_ff : {512, 768, ff}) {
        const int count = n_ff == ff ? experts : local_count;
        test_graph weights;
        auto * gate = ggml_new_tensor_3d(weights.ctx, GGML_TYPE_Q3_K, hidden, 2*n_ff, count);
        auto * q4 = ggml_new_tensor_3d(weights.ctx, GGML_TYPE_Q4_K, n_ff, hidden, count);
        auto * q5 = ggml_new_tensor_3d(weights.ctx, GGML_TYPE_Q5_K, n_ff, hidden, count);
        ggml_set_name(gate, "blk.0.ffn_gate_up_exps.weight");
        ggml_set_name(q4, "blk.0.ffn_down_exps.weight");
        ggml_set_name(q5, q4->name);
        weights.buf = ggml_backend_alloc_ctx_tensors(weights.ctx, backend);
        if (!weights.buf) {
            return false;
        }
        ggml_backend_buffer_set_usage(weights.buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        for (auto * tensor : {gate, q4, q5}) {
            fill_experts(tensor);
        }
        for (auto * down : {q4, q5}) {
            test_graph reference_weights;
            auto * reference_down = down;
            if (ref && n_ff == 768) {
                // The old TMA dispatch supports 1024, not 768. Pad with one zero QK_K block.
                reference_down = ggml_new_tensor_3d(reference_weights.ctx, down->type, 1024, hidden, count);
                reference_weights.buf = ggml_backend_alloc_ctx_tensors(reference_weights.ctx, backend);
                if (!reference_weights.buf) {
                    return false;
                }
                fill_experts(reference_down, n_ff);
            }
            for (const int rows : {23, 32, 33, 46, 69, 92, 115, 128}) {
                ok &= run_wide_step(backend, out, ref, gate, down, reference_down, rows, true);
            }
            if (down == q4) {
                // Repeat routes to exercise rank across all four mask words.
                ok &= run_wide_step(backend, out, ref, gate, down, reference_down, 128, false);
            }
        }
    }
    return ok;
}
} // namespace

int main(int argc, char ** argv) {
    const bool wide = argc >= 3 && strcmp(argv[1], "--tma-wide") == 0;
    if (wide ? argc > 4 : argc != 2) {
        fprintf(stderr, "usage: %s OUTPUT | --tma-wide OUTPUT [REFERENCE]\n", argv[0]);
        return 1;
    }
    ggml_backend_load_all();
    auto * backend = ggml_backend_init_by_name("CUDA0", nullptr);
    FILE * out = fopen(argv[wide ? 2 : 1], "wb");
    FILE * ref = argc == 4 ? fopen(argv[3], "rb") : nullptr;
    const bool ok = backend && out && (argc != 4 || ref) &&
        (wide ? run_wide(backend, out, ref) : run(backend, out));
    const bool ref_done = !ref || fgetc(ref) == EOF;
    if (ref) {
        fclose(ref);
    }
    const bool written = out && fclose(out) == 0;
    ggml_backend_free(backend);
    return ok && ref_done && written ? 0 : 1;
}
