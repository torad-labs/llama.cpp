#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

namespace {
constexpr int hidden = 4608;
constexpr int ff = 2304;
constexpr int experts = 384;
constexpr int slots = 8;
constexpr int local_first = 96;
constexpr int local_count = 96;

struct test_graph {
    ggml_context * ctx = ggml_init({32*ggml_tensor_overhead() + ggml_graph_overhead_custom(32, false), nullptr, true});
    ggml_backend_buffer_t buf = nullptr;

    ~test_graph() {
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }
};

void fill_experts(ggml_tensor * t) {
    const int64_t k = t->ne[0];
    const int64_t m = t->ne[1];
    const size_t row_bytes = ggml_row_size(t->type, k);
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
} // namespace

int main(int argc, char ** argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s OUTPUT\n", argv[0]);
        return 1;
    }
    ggml_backend_load_all();
    auto * backend = ggml_backend_init_by_name("CUDA0", nullptr);
    FILE * out = fopen(argv[1], "wb");
    const bool ok = backend && out && run(backend, out);
    const bool written = out && fclose(out) == 0;
    ggml_backend_free(backend);
    return ok && written ? 0 : 1;
}
