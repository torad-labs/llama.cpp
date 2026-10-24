// The backend's public byte API must remain canonical after weight repacking.
// Run once per physical GPU as well as on CPU; a missing requested device fails.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
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

static ggml_backend_meta_split_state mirrored(const ggml_tensor *, void *) {
    return {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
}

static void check_meta_cache(ggml_backend_dev_t device) {
    ggml_backend_dev_t devices[] = {device, device};
    auto * backend = ggml_backend_dev_init(ggml_backend_meta_device(devices, 2, mirrored, nullptr), nullptr);
    auto * weights = ggml_init({2*ggml_tensor_overhead(), nullptr, true});
    auto * cache = ggml_new_tensor_2d(weights, GGML_TYPE_Q8_0, 256, 8);
    auto * buffer = ggml_backend_alloc_ctx_tensors(weights, backend);
    GGML_ASSERT(buffer);
    ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
    ggml_backend_buffer_clear(buffer, 0);
    auto * ctx = ggml_init({16*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true});
    for (int generation = 0; generation < 16; ++generation) {
        ggml_reset(ctx);
        auto * view = ggml_view_2d(ctx, cache, 256, 4, cache->nb[1], (generation%2)*4*cache->nb[1]);
        auto * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 256, 2);
        auto * ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 2);
        auto * update = ggml_set_rows(ctx, view, x, ids);
        auto * out = ggml_get_rows(ctx, update, ids);
        GGML_ASSERT(ggml_backend_supports_op(backend, view));
        GGML_ASSERT(ggml_backend_supports_op(backend, update));
        GGML_ASSERT(ggml_backend_supports_op(backend, out));
        auto * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, out);
        auto * compute = ggml_backend_alloc_ctx_tensors(ctx, backend);
        GGML_ASSERT(compute);
        const int32_t indices[] = {0, 3};
        std::vector<float> input(512, 127.0f*(generation + 1)), actual(512);
        ggml_backend_tensor_set(x, input.data(), 0, ggml_nbytes(x));
        ggml_backend_tensor_set(ids, indices, 0, sizeof(indices));
        GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
        ggml_backend_tensor_get(out, actual.data(), 0, ggml_nbytes(out));
        GGML_ASSERT(actual == input);
        ggml_backend_buffer_free(compute);
    }
    ggml_free(ctx);
    ggml_backend_buffer_free(buffer);
    ggml_free(weights);
    ggml_backend_free(backend);
    puts("16/16 meta Q8 cache rebuilds passed");
}

static void check_reinitialization(ggml_backend_t backend) {
    std::mt19937 rng(531);
    for (int life = 0; life < 16; ++life) {
        auto * ctx = ggml_init({8*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true});
        auto * old = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, 512, 16);
        auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        GGML_ASSERT(buffer);
        ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        std::vector<uint8_t> bytes(ggml_nbytes(old), 0x11);
        ggml_backend_tensor_set(old, bytes.data(), 0, bytes.size());
        auto * replacement = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 256);
        const size_t alignment = ggml_backend_buft_get_alignment(ggml_backend_get_default_buffer_type(backend));
        const size_t positions = (ggml_nbytes(old) - ggml_nbytes(replacement))/alignment;
        GGML_ASSERT(positions > 0);
        const size_t offset = alignment*(1 + rng()%positions);
        GGML_ASSERT(ggml_backend_tensor_alloc(buffer, replacement, (char *) old->data + offset) == GGML_STATUS_SUCCESS);
        auto * out = ggml_scale(ctx, replacement, 2.0f);
        auto * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, out);
        auto * compute = ggml_backend_alloc_ctx_tensors(ctx, backend);
        GGML_ASSERT(compute);
        std::vector<float> input(256), actual(256);
        for (int i = 0; i < 256; ++i) { input[i] = float(int(rng()%1024) - 512)/8.0f; }
        ggml_backend_tensor_set(replacement, input.data(), 0, ggml_nbytes(replacement));
        GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
        ggml_backend_tensor_get(out, actual.data(), 0, ggml_nbytes(out));
        for (int i = 0; i < 256; ++i) { GGML_ASSERT(actual[i] == 2.0f*input[i]); }
        ggml_backend_buffer_free(compute);
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
    }
}

static void check_interleaved_transfers(ggml_backend_t backend, ggml_backend_dev_t device) {
    auto * other = ggml_backend_dev_init(device, nullptr);
    auto * ctx = ggml_init({2*ggml_tensor_overhead(), nullptr, true});
    auto * a = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, 8192, 513);
    auto * b = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, 8192, 513);
    auto * ab = ggml_backend_alloc_buffer(backend, ggml_backend_buft_get_alloc_size(ggml_backend_get_default_buffer_type(backend), a));
    auto * bb = ggml_backend_alloc_buffer(other, ggml_backend_buft_get_alloc_size(ggml_backend_get_default_buffer_type(other), b));
    GGML_ASSERT(ab && bb);
    ggml_backend_buffer_set_usage(ab, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_buffer_set_usage(bb, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    GGML_ASSERT(ggml_backend_tensor_alloc(ab, a, ggml_backend_buffer_get_base(ab)) == GGML_STATUS_SUCCESS);
    GGML_ASSERT(ggml_backend_tensor_alloc(bb, b, ggml_backend_buffer_get_base(bb)) == GGML_STATUS_SUCCESS);
    std::vector<uint8_t> input_a(ggml_nbytes(a)), input_b(input_a.size()), actual_a(input_a.size()), actual_b(input_a.size());
    for (int round = 0; round < 8; ++round) {
        for (size_t i = 0; i < input_a.size(); ++i) {
            input_a[i] = uint8_t(i + round);
            input_b[i] = uint8_t(3*i + 19*round);
        }
        ggml_backend_tensor_set_async(backend, a, input_a.data(), 0, input_a.size());
        ggml_backend_tensor_set_async(other, b, input_b.data(), 0, input_b.size());
        ggml_backend_tensor_get_async(backend, a, actual_a.data(), 0, actual_a.size());
        ggml_backend_tensor_get_async(other, b, actual_b.data(), 0, actual_b.size());
        ggml_backend_synchronize(backend);
        ggml_backend_synchronize(other);
        GGML_ASSERT(actual_a == input_a && actual_b == input_b);
    }
    ggml_backend_buffer_free(ab);
    ggml_backend_buffer_free(bb);
    ggml_free(ctx);
    ggml_backend_free(other);
}

static void check_channel_views(ggml_backend_t backend, ggml_backend_dev_t device) {
    auto * cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    auto * wc = ggml_init({2*ggml_tensor_overhead(), nullptr, true});
    auto * weight = ggml_new_tensor_2d(wc, GGML_TYPE_Q8_0, 512, 16);
    auto * wb = ggml_backend_alloc_ctx_tensors(wc, backend);
    GGML_ASSERT(wb);
    ggml_backend_buffer_set_usage(wb, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    std::vector<uint8_t> codes(ggml_nbytes(weight));
    for (int block = 0; block < 256; ++block) {
        const ggml_fp16_t scale = ggml_fp32_to_fp16(1.0f);
        memcpy(codes.data() + block*34, &scale, 2);
        std::fill_n(codes.data() + block*34 + 2, 32, uint8_t(block/16 + 1));
    }
    ggml_backend_tensor_set(weight, codes.data(), 0, codes.size());
    for (const int axis : {2, 3}) for (const bool crossing : {false, true}) {
        auto * ctx = ggml_init({16*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true});
        const size_t stride = crossing ? 9*34 : 2*weight->nb[1];
        auto * view = ggml_view_4d(ctx, weight, 256, 2, axis == 2 ? 2 : 1, axis == 3 ? 2 : 1,
                                 weight->nb[1], axis == 2 ? stride : 2*weight->nb[1],
                                 axis == 3 ? stride : 4*weight->nb[1], 0);
        auto * x = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 256, 3, view->ne[2], view->ne[3]);
        auto * out = ggml_mul_mat(ctx, view, x);
        if (ggml_backend_dev_type(device) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            GGML_ASSERT(ggml_backend_supports_op(backend, out) == !crossing);
        }
        ggml_set_input(x);
        ggml_set_output(out);
        auto * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, out);
        ggml_backend_t backends[] = {backend, cpu};
        auto * sched = ggml_backend_sched_new(backends, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE, false, true);
        GGML_ASSERT(ggml_backend_sched_alloc_graph(sched, graph));
        std::vector<float> input(ggml_nelements(x)), actual(ggml_nelements(out));
        for (int channel = 0; channel < 2; ++channel) for (int token = 0; token < 3; ++token) {
            std::fill_n(input.data() + (channel*3 + token)*256, 256, 127.0f*(token + 1));
        }
        ggml_backend_tensor_set(x, input.data(), 0, ggml_nbytes(x));
        GGML_ASSERT(ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS);
        ggml_backend_tensor_get(out, actual.data(), 0, ggml_nbytes(out));
        for (int channel = 0; channel < 2; ++channel) for (int token = 0; token < 3; ++token) for (int row = 0; row < 2; ++row) {
            int sum = 0;
            for (int block = 0; block < 8; ++block) { sum += (row*16 + channel*stride/34 + block)/16 + 1; }
            GGML_ASSERT(actual[(channel*3 + token)*2 + row] == 32.0f*sum*127.0f*(token + 1));
        }
        ggml_backend_sched_free(sched);
        ggml_free(ctx);
    }
    ggml_backend_buffer_free(wb);
    ggml_free(wc);
    ggml_backend_free(cpu);
    puts("4/4 packed channel views passed");
}

static void check_meta_fallback(ggml_backend_dev_t device) {
    auto * cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    ggml_backend_dev_t devices[] = {device, device};
    auto * meta = ggml_backend_meta_device(devices, 2, mirrored, nullptr);
    auto * backend = ggml_backend_dev_init(meta, nullptr);
    for (const bool crossing : {false, true}) {
        auto * wc = ggml_init({2*ggml_tensor_overhead(), nullptr, true});
        auto * weight = ggml_new_tensor_2d(wc, GGML_TYPE_Q8_0, 256, 8);
        auto * wb = ggml_backend_alloc_ctx_tensors(wc, backend);
        GGML_ASSERT(wb);
        ggml_backend_buffer_set_usage(wb, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        std::vector<uint8_t> codes(ggml_nbytes(weight));
        for (int row = 0; row < 8; ++row) {
            for (int block = 0; block < 8; ++block) {
                const size_t offset = row*weight->nb[1] + block*34;
                const ggml_fp16_t scale = ggml_fp32_to_fp16(1.0f);
                memcpy(codes.data() + offset, &scale, 2);
                std::fill_n(codes.data() + offset + 2, 32, uint8_t(row + 1));
            }
        }
        ggml_backend_tensor_set(weight, codes.data(), 0, codes.size());
        auto * ctx = ggml_init({16*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true});
        auto * x = crossing ? ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 512, 3) :
                              ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 256, 1, 3);
        auto * ids = crossing ? nullptr : ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 1, 3);
        auto * view = crossing ? ggml_view_2d(ctx, weight, 512, 4, 2*weight->nb[1], 0) : weight;
        auto * out = crossing ? ggml_mul_mat(ctx, view, x) : ggml_mul_mat_id(ctx, view, x, ids);
        if (ggml_backend_dev_type(device) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            GGML_ASSERT(!ggml_backend_supports_op(backend, out));
        }
        ggml_set_input(x);
        if (ids) { ggml_set_input(ids); }
        ggml_set_output(out);
        auto * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, out);
        ggml_backend_t backends[] = {backend, cpu};
        auto * sched = ggml_backend_sched_new(backends, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE, false, true);
        GGML_ASSERT(ggml_backend_sched_alloc_graph(sched, graph));
        std::vector<float> input(ggml_nelements(x));
        for (int token = 0; token < 3; ++token) {
            // Multiples of 127 keep the activation scale exact in both CPU and CUDA Q8 paths.
            std::fill_n(input.data() + token*x->ne[0], x->ne[0], 127.0f*(token + 1));
        }
        ggml_backend_tensor_set(x, input.data(), 0, ggml_nbytes(x));
        const int32_t zeros[3] = {};
        if (ids) { ggml_backend_tensor_set(ids, zeros, 0, sizeof(zeros)); }
        GGML_ASSERT(ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS);
        std::vector<float> actual(ggml_nelements(out));
        ggml_backend_tensor_get(out, actual.data(), 0, ggml_nbytes(out));
        for (int token = 0; token < 3; ++token) {
            for (int row = 0; row < out->ne[0]; ++row) {
                const float expected = 256.0f*127.0f*(token + 1)*(crossing ? 4*row + 3 : row + 1);
                if (actual[token*out->ne[0] + row] != expected) {
                    fprintf(stderr, "meta fallback: crossing=%d token=%d row=%d actual=%g expected=%g\n",
                            crossing, token, row, actual[token*out->ne[0] + row], expected);
                }
                GGML_ASSERT(actual[token*out->ne[0] + row] == expected);
            }
        }
        ggml_backend_sched_free(sched);
        ggml_free(ctx);
        ggml_backend_buffer_free(wb);
        ggml_free(wc);
    }
    ggml_backend_free(backend);
    ggml_backend_free(cpu);
    puts("2/2 meta packed reader fallbacks passed");
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
    if (argc > 2 && strcmp(argv[2], "--meta-cache") == 0) {
        check_meta_cache(device);
        return 0;
    }
    if (argc > 2 && strcmp(argv[2], "--meta") == 0) {
        check_meta_fallback(device);
        return 0;
    }
    auto * backend = ggml_backend_dev_init(device, nullptr);
    GGML_ASSERT(backend);
    if (argc > 2 && strcmp(argv[2], "--views") == 0) {
        check_channel_views(backend, device);
        ggml_backend_free(backend);
        return 0;
    }
    if (argc > 2 && strcmp(argv[2], "--cublas") == 0) {
        check_cublas(backend);
        ggml_backend_free(backend);
        return 0;
    }
    std::mt19937 rng(530);
    for (int life = 0; life < 3; ++life) {
        for (const int k : {256, 1536, 4096, 8192, 16384}) {
            auto * ctx = ggml_init({4*ggml_tensor_overhead(), nullptr, true});
            const int n_rows = k == 16384 ? 257 : 35;
            auto * src = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, k, n_rows);
            auto * dst = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, k, n_rows);
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
    check_reinitialization(backend);
    check_interleaved_transfers(backend, device);
    ggml_backend_free(backend);
    printf("%s: Q8 canonical byte API, async/pitched transfer, copy, memset and lifetime reuse passed\n", name);
}
