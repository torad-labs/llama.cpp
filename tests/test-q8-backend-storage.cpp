// The backend's public byte API must remain canonical after weight repacking.
// Run once per physical GPU as well as on CPU; a missing requested device fails.
#include "ggml.h"
#include "ggml-backend.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

static void check(ggml_tensor * t, const std::vector<uint8_t> & expected) {
    std::vector<uint8_t> actual(expected.size());
    ggml_backend_tensor_get(t, actual.data(), 0, actual.size());
    GGML_ASSERT(actual == expected);
}

// BF16 activations select cuBLAS on CUDA but are not supported by the CPU
// matmul reference. Compare packed weights to the unchanged canonical CUDA
// path instead, with identical bits, geometry and compute precision. The runner
// repeats this whole test with GGML_CUDA_CUBLAS_COMPUTE_TYPE=f16,bf16,f32.
static void check_cublas(ggml_backend_t backend) {
    GGML_ASSERT(strncmp(ggml_backend_name(backend), "CUDA", 4) == 0);
    GGML_ASSERT(getenv("GGML_CUDA_Q8_SPLIT") && atoi(getenv("GGML_CUDA_Q8_SPLIT")) == 1);
    std::mt19937 rng(530);
    int cases = 0;
    for (const int k : {1536, 4096}) {
        for (const int n : {3, 128}) {
            for (const bool sliced : {false, true}) {
                auto * weights = ggml_init({2*ggml_tensor_overhead(), nullptr, true});
                auto * ordinary = ggml_init({2*ggml_tensor_overhead(), nullptr, true});
                auto * ctx = ggml_init({16*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true});
                auto * packed = ggml_new_tensor_2d(weights, GGML_TYPE_Q8_0, k + (sliced ? 256 : 0), 35);
                auto * canonical = ggml_new_tensor_2d(ordinary, GGML_TYPE_Q8_0, packed->ne[0], packed->ne[1]);
                auto * packed_buf = ggml_backend_alloc_ctx_tensors(weights, backend);
                auto * canonical_buf = ggml_backend_alloc_ctx_tensors(ordinary, backend);
                GGML_ASSERT(packed_buf && canonical_buf);
                ggml_backend_buffer_set_usage(packed_buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
                // ANY usage keeps the reference tensor canonical before its first upload.
                ggml_backend_buffer_set_usage(canonical_buf, GGML_BACKEND_BUFFER_USAGE_ANY);
                auto * x = ggml_new_tensor_2d(ctx, GGML_TYPE_BF16, k, n);
                const size_t offset = packed->nb[1] + (sliced ? 34*8 : 0);
                auto * a = ggml_view_2d(ctx, packed, k, 32, packed->nb[1], offset);
                auto * b = ggml_view_2d(ctx, canonical, k, 32, canonical->nb[1], offset);
                auto * ya = ggml_mul_mat(ctx, a, x);
                auto * yb = ggml_mul_mat(ctx, b, x);
                GGML_ASSERT(ggml_backend_supports_op(backend, ya) && ggml_backend_supports_op(backend, yb));
                auto * graph = ggml_new_graph(ctx);
                ggml_build_forward_expand(graph, ya);
                ggml_build_forward_expand(graph, yb);
                auto * compute = ggml_backend_alloc_ctx_tensors(ctx, backend);
                GGML_ASSERT(compute);
                std::vector<uint8_t> codes(ggml_nbytes(packed));
                for (size_t block = 0; block < codes.size(); block += 34) {
                    const ggml_fp16_t d = ggml_fp32_to_fp16(0.003f + float(rng()%100)*0.0001f);
                    memcpy(codes.data() + block, &d, sizeof(d));
                    for (size_t j = 2; j < 34; ++j) { codes[block+j] = uint8_t(rng()%255); }
                }
                std::vector<ggml_bf16_t> input(ggml_nelements(x));
                for (auto & value : input) { value = ggml_fp32_to_bf16(float(int(rng()%65)-32)/16.0f); }
                ggml_backend_tensor_set(packed, codes.data(), 0, codes.size());
                ggml_backend_tensor_set(canonical, codes.data(), 0, codes.size());
                ggml_backend_tensor_set(x, input.data(), 0, input.size()*sizeof(input[0]));
                GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
                std::vector<float> actual(ggml_nelements(ya)), reference(actual.size());
                ggml_backend_tensor_get(ya, actual.data(), 0, ggml_nbytes(ya));
                ggml_backend_tensor_get(yb, reference.data(), 0, ggml_nbytes(yb));
                GGML_ASSERT(memcmp(actual.data(), reference.data(), ggml_nbytes(ya)) == 0);
                GGML_ASSERT(std::all_of(actual.begin(), actual.end(), [](float v) { return std::isfinite(v); }));
                GGML_ASSERT(std::any_of(actual.begin(), actual.end(), [](float v) { return v != 0; }));
                ggml_backend_buffer_free(compute);
                ggml_backend_buffer_free(canonical_buf);
                ggml_backend_buffer_free(packed_buf);
                ggml_free(ctx);
                ggml_free(ordinary);
                ggml_free(weights);
                ++cases;
            }
        }
    }
    GGML_ASSERT(cases == 8);
    printf("8/8 cuBLAS packed/canonical comparisons passed\n");
}

int main(int argc, char ** argv) {
    ggml_backend_load_all();
    const char * name = argc > 1 ? argv[1] : "CPU";
    auto * device = ggml_backend_dev_by_name(name);
    if (!device) { fprintf(stderr, "required device %s missing\n", name); return 1; }
    auto * backend = ggml_backend_dev_init(device, nullptr);
    GGML_ASSERT(backend);
    if (argc > 2 && strcmp(argv[2], "--cublas") == 0) {
        check_cublas(backend);
        ggml_backend_free(backend);
        return 0;
    }
    std::mt19937 rng(530);
    for (int life = 0; life < 3; ++life) {
        for (const int k : {256, 1536, 4096, 8192, 16384}) {
            auto * ctx = ggml_init({4*ggml_tensor_overhead(), nullptr, true});
            auto * src = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, k, 35);
            auto * dst = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, k, 35);
            auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
            GGML_ASSERT(buffer);
            ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            std::vector<uint8_t> expected(ggml_nbytes(src));
            for (auto & v : expected) { v = uint8_t(rng()); }
            ggml_backend_tensor_set_async(backend, src, expected.data(), 0, expected.size());
            ggml_backend_synchronize(backend);
            check(src, expected);
            // Generated partial changes split scales, blocks, row planes and
            // buffer edges; everything outside the requested range is retained.
            for (int step = 0; step < 64; ++step) {
                const size_t off = rng()%expected.size();
                const size_t size = std::min<size_t>(1 + rng()%521, expected.size() - off);
                for (size_t i = off; i < off + size; ++i) { expected[i] = uint8_t(rng()); }
                ggml_backend_tensor_set_async(backend, src, expected.data() + off, off, size);
                ggml_backend_synchronize(backend);
            }
            check(src, expected);
            const size_t row_bytes = src->nb[1], width = 129, pitch = 137, copies = 7;
            std::vector<uint8_t> rows(copies*pitch, 0), readback(rows.size(), 0);
            for (size_t r = 0; r < copies; ++r) {
                for (size_t i = 0; i < width; ++i) { rows[r*pitch + i] = uint8_t(rng()); }
                std::memcpy(expected.data() + 13 + 2*r*row_bytes, rows.data() + r*pitch, width);
            }
            ggml_backend_tensor_set_2d_async(backend, src, rows.data(), 13, width, copies, 2*row_bytes, pitch);
            ggml_backend_tensor_get_2d_async(backend, src, readback.data(), 13, width, copies, 2*row_bytes, pitch);
            ggml_backend_synchronize(backend);
            GGML_ASSERT(rows == readback);
            check(src, expected);
            ggml_backend_tensor_copy_async(backend, backend, src, dst);
            ggml_backend_synchronize(backend);
            check(dst, expected);
            ggml_backend_tensor_memset(dst, 0x9d, row_bytes - 1, row_bytes + 3);
            std::fill(expected.begin() + row_bytes - 1, expected.begin() + 2*row_bytes + 2, uint8_t(0x9d));
            check(dst, expected);
            ggml_backend_tensor_copy(dst, src);
            std::vector<uint8_t> all(expected.size());
            ggml_backend_tensor_get_async(backend, src, all.data(), 0, all.size());
            ggml_backend_synchronize(backend);
            GGML_ASSERT(all == expected);
            ggml_backend_buffer_free(buffer);
            ggml_free(ctx);
        }
    }
    ggml_backend_free(backend);
    printf("%s: Q8 canonical byte API, async/pitched transfer, copy, memset and lifetime reuse passed\n", name);
}
