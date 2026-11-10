// The meta backend (-sm tensor) gives each device its slice of a split tensor: a tensor with the split axis cut to the
// slice, and the strides of the dims outside the split axis scaled with it. The dims outside were those with a greater
// stride than the split axis. A tensor split along an axis of extent 1, as one head split two ways, has the same stride
// on that axis as on the dim after it, so the dim after it kept its full stride: the device with no slice had a tensor
// of 0 bytes that took a whole chunk of every write, and writing the tensor aborted (tensor write out of bounds).
//
// Each tensor below is written, read back and cleared through the meta buffer, split along axis 2 as its name says:
// heads_<a><b> has a heads on the first device and b on the second. The meta backend is over the first GPU twice, which
// splits over two devices as two GPUs do; with no GPU the test is skipped.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#if defined(__unix__) || defined(__APPLE__)
#include <sys/resource.h>
#endif

static constexpr int64_t n_cols = 64;
static constexpr int64_t n_rows = 8;

static std::atomic<int> captured {0};
static unsigned long long expert_hits = 0;
static unsigned long long reductions_to_owners = 0; // the internal AllReduce's, logged when its pipeline is freed
static unsigned long long reductions_to_all = 0;
static ggml_backend_dev_t peer = nullptr;
static std::array<ggml_backend_dev_t, 4> host_devices = {};

static ggml_backend_dev_t second_device(ggml_backend_dev_t first) { return peer ? peer : first; }

static void set_env(const char * name, const char * value) {
#ifdef _WIN32
    _putenv_s(name, value != nullptr ? value : "");
#else
    if (value != nullptr) {
        setenv(name, value, 1);
    } else {
        unsetenv(name);
    }
#endif
}

static void log_cb(ggml_log_level level, const char * text, void *) {
    if (strstr(text, "ggml_backend_cuda_capture_end") && strstr(text, " nodes, ")) { ++captured; }
    int device, first;
    unsigned long long total;
    if (sscanf(text, "dp_experts device=%d first=%d total=%llu", &device, &first, &total) == 3) { expert_hits += total; }
    unsigned long long owners, all;
    if (const char * counts = strstr(text, "ggml_cuda_ar_pipeline_free: ");
            counts && sscanf(counts, "ggml_cuda_ar_pipeline_free: %llu reductions to owners, %llu to every rank", &owners, &all) == 2) {
        reductions_to_owners += owners;
        reductions_to_all += all;
    }
    if (level != GGML_LOG_LEVEL_DEBUG) { fputs(text, stderr); }
}

static ggml_backend_meta_split_state split_by_name(const ggml_tensor * tensor, void * /*userdata*/) {
    GGML_ASSERT(strncmp(tensor->name, "heads_", 6) == 0);
    ggml_backend_meta_split_state split_state = {GGML_BACKEND_SPLIT_AXIS_2, {0}, {1}, 1};
    split_state.ne[0] = tensor->name[6] - '0';
    split_state.ne[1] = tensor->name[7] - '0';
    GGML_ASSERT(split_state.ne[0] + split_state.ne[1] == tensor->ne[2]);
    return split_state;
}

static ggml_backend_meta_split_state split_rows(const ggml_tensor * tensor, void * userdata) {
    if (strcmp(tensor->name, "tokens") == 0) {
        ggml_backend_meta_split_state state = {GGML_BACKEND_SPLIT_AXIS_0, {0}, {1}, 1};
        const auto & rows = *static_cast<const std::array<int64_t, 4> *>(userdata);
        std::copy(rows.begin(), rows.end(), state.ne);
        return state;
    }
    return {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
}

// -sm tensor: the device's callback splits the tensors of a weights buffer by name, and nothing owns rows. q, k and v
// ([d, n_tokens]) are cut in halves along axis 0, as a fused projection's heads are; q_heads ([d, n_head, n_batch]) in
// halves along axis 1; everything else is mirrored, as a latent cache and a mask are.
static ggml_backend_meta_split_state split_columns(const ggml_tensor * tensor, void * /*userdata*/) {
    if (strcmp(tensor->name, "q") == 0 || strcmp(tensor->name, "k") == 0 || strcmp(tensor->name, "v") == 0) {
        return {GGML_BACKEND_SPLIT_AXIS_0, {tensor->ne[0]/2, tensor->ne[0] - tensor->ne[0]/2}, {1}, 1};
    }
    if (strcmp(tensor->name, "q_heads") == 0) {
        return {GGML_BACKEND_SPLIT_AXIS_1, {tensor->ne[1]/2, tensor->ne[1] - tensor->ne[1]/2}, {1}, 1};
    }
    return {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
}

static bool test_permuted_reads(ggml_backend_dev_t device) {
    ggml_backend_dev_t devices[] = {device, second_device(device)};
    ggml_backend_dev_t meta = ggml_backend_meta_device(devices, 2, split_columns, nullptr);
    std::mt19937 rng(1109);
    for (int sample = 0; sample < 12; ++sample) {
        const int64_t columns = 1 + rng()%17, heads = 1 + rng()%7;
        const int64_t tokens = 1 + rng()%9, groups = 1 + rng()%3;
        std::array<int, 4> axes = {0, 1, 2, 3};
        do {
            ggml_context * ctx = ggml_init({4*ggml_tensor_overhead(), nullptr, true});
            ggml_tensor * base = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, columns, heads, tokens, groups);
            ggml_set_name(base, "q_heads");
            ggml_tensor * view = ggml_permute(ctx, base, axes[0], axes[1], axes[2], axes[3]);
            ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_dev_buffer_type(meta));
            GGML_ASSERT(buffer);
            std::vector<uint8_t> data(ggml_nbytes(base));
            for (auto & value : data) { value = uint8_t(rng()); }
            ggml_backend_tensor_set(base, data.data(), 0, data.size());
            bool ok = true;
            for (int range = 0; range < 4; ++range) {
                const size_t offset = range == 0 ? 0 : rng()%data.size();
                const size_t size = range == 0 ? data.size() : 1 + rng()%(data.size() - offset);
                std::vector<uint8_t> actual(size + 2, 0xa5);
                ggml_backend_tensor_get(view, actual.data() + 1, offset, size);
                ok = ok && actual.front() == 0xa5 && actual.back() == 0xa5 &&
                     std::equal(actual.begin() + 1, actual.end() - 1, data.begin() + offset);
            }
            ggml_backend_buffer_free(buffer);
            ggml_free(ctx);
            if (!ok) {
                printf("FAIL permuted split read: sample=%d axes=%d%d%d%d\n",
                        sample, axes[0], axes[1], axes[2], axes[3]);
                return false;
            }
        } while (std::next_permutation(axes.begin(), axes.end()));
    }
    printf("ok   generated permuted split reads preserve byte ranges\n");
    return true;
}

static bool test_rows(ggml_backend_dev_t gpu, bool explicit_layout) {
    std::array<int64_t, 4> rows = {1, 0, 2, 5};
    ggml_backend_dev_t devs[4] = {gpu, second_device(gpu), gpu, second_device(gpu)};
    ggml_backend_dev_t meta = ggml_backend_meta_device(devs, 4, split_rows, &rows);
    ggml_backend_t backend = ggml_backend_dev_init(meta, nullptr);
    ggml_context * weights = ggml_init({4*ggml_tensor_overhead(), nullptr, true});
    ggml_tensor * table = ggml_new_tensor_2d(weights, GGML_TYPE_F32, 32, 16);
    ggml_context * ctx = ggml_init({16*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true});
    ggml_tensor * tokens = ggml_new_tensor_1d(explicit_layout ? ctx : weights, GGML_TYPE_I32, 8);
    ggml_set_name(tokens, "tokens");
    ggml_set_input(tokens);
    if (explicit_layout) { ggml_backend_meta_split_input(tokens, 0, rows.data(), rows.size()); }
    ggml_tensor * projection = ggml_new_tensor_2d(weights, GGML_TYPE_F32, 32, 32);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors_from_buft(weights, ggml_backend_dev_buffer_type(meta));
    GGML_ASSERT(buffer);

    ggml_tensor * embd = ggml_get_rows(ctx, table, tokens);
    ggml_tensor * out = ggml_mul_mat(ctx, projection, embd);
    ggml_tensor * gathered = nullptr;
    if (explicit_layout) {
        gathered = ggml_backend_meta_gather(ctx, out);
        ggml_set_output(gathered);
        const int64_t next_rows[] = {0, 3, 4, 1};
        out = ggml_backend_meta_scatter(ctx, gathered, 1, next_rows, 4);
        out = ggml_mul_mat(ctx, projection, out);
    }
    ggml_set_output(out);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);
    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_dev_buffer_type(meta));
    GGML_ASSERT(ggml_gallocr_alloc_graph(alloc, graph));
    std::vector<float> table_data(32*16), projection_data(32*32, 0.0f), result(32*8);
    for (int i = 0; i < 32*16; ++i) { table_data[i] = i + 1; }
    for (int i = 0; i < 32; ++i) { projection_data[32*i + i] = 2.0f; }
    const int32_t token_data[8] = {15, 2, 9, 0, 7, 1, 13, 4};
    ggml_backend_tensor_set(table, table_data.data(), 0, ggml_nbytes(table));
    ggml_backend_tensor_set(tokens, token_data, 0, ggml_nbytes(tokens));
    ggml_backend_tensor_set(projection, projection_data.data(), 0, ggml_nbytes(projection));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    ggml_backend_tensor_get(out, result.data(), 0, ggml_nbytes(out));
    bool ok = true;
    for (int row = 0; row < 8; ++row) {
        for (int col = 0; col < 32; ++col) {
            ok = ok && result[32*row + col] == (explicit_layout ? 4.0f : 2.0f)*table_data[32*token_data[row] + col];
        }
    }
    if (gathered) {
        ggml_backend_tensor_get(gathered, result.data(), 0, ggml_nbytes(gathered));
        for (int row = 0; row < 8; ++row) {
            for (int col = 0; col < 32; ++col) {
                ok = ok && result[32*row + col] == 2.0f*table_data[32*token_data[row] + col];
            }
        }
    }
    printf("%s four-rank row split with an empty rank%s\n", ok ? "ok  " : "FAIL", explicit_layout ? ", gathered and redistributed" : "");
    ggml_gallocr_free(alloc);
    ggml_free(ctx);
    ggml_backend_buffer_free(buffer);
    ggml_free(weights);
    ggml_backend_free(backend);
    return ok;
}

static bool split_test_stack_limit() {
#if defined(__unix__) || defined(__APPLE__)
    rlimit limit;
    if (getrlimit(RLIMIT_STACK, &limit) != 0) { return false; }
    limit.rlim_cur = std::min(limit.rlim_cur, rlim_t(8*1024*1024));
    if (setrlimit(RLIMIT_STACK, &limit) != 0) { return false; }
#endif
    return true;
}

static bool test_split_cache(ggml_backend_dev_t device, bool cycle = false) {
    ggml_backend_dev_t devices[] = {device, device, device, device};
    ggml_backend_t backend = ggml_backend_dev_init(ggml_backend_meta_device(devices, 4, split_rows, nullptr), nullptr);
    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    constexpr size_t graph_size = 1024;
    ggml_context * ctx = ggml_init({graph_size*ggml_tensor_overhead() + ggml_graph_overhead_custom(graph_size, false), nullptr, true});
    const int64_t layouts[][4] = {{2, 2, 2, 2}, {3, 0, 3, 1}, {0, 0, 3, 0}, {0, 0, 0, 1}};
    bool ok = true;
    for (int depth : {16, 512}) {
        for (const auto & rows : layouts) {
            // Reuse the tensor arena and allocator while the active owners and graph shape change.
            ggml_reset(ctx);
            ggml_cgraph * graph = ggml_new_graph_custom(ctx, graph_size, false);
            const int64_t count = rows[0] + rows[1] + rows[2] + rows[3];
            ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, count);
            ggml_set_input(input);
            ggml_backend_meta_split_input(input, 1, rows, 4);
            ggml_tensor * out = input;
            ggml_tensor * first = nullptr;
            for (int i = 0; i < depth; ++i) {
                out = ggml_scale(ctx, out, 1.0f);
                if (i == 0) { first = out; }
                // Keep graph construction shallow too: only the new node is unvisited.
                ggml_build_forward_expand(graph, out);
            }
            ggml_set_name(first, "split_chain_first");
            ggml_set_name(out, "split_chain_last");
            ggml_set_output(out);
            GGML_ASSERT(ggml_gallocr_alloc_graph(alloc, graph));
            std::vector<float> data(4*count), result(4*count);
            for (size_t i = 0; i < data.size(); ++i) { data[i] = float(i + 1); }
            ggml_backend_tensor_set(input, data.data(), 0, ggml_nbytes(input));
            GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
            ggml_backend_tensor_get(out, result.data(), 0, ggml_nbytes(out));
            ok = result == data && ok;
            if (cycle) {
                GGML_ASSERT(ok);
                first->src[0] = out;
                ggml_set_abort_callback([](const char * message) {
                    const bool detected = strstr(message, "cycle in Meta split") && strstr(message, "split_chain_first") && strstr(message, "split_chain_last");
                    fprintf(stderr, "%s split dependency cycle: %s\n", detected ? "ok  " : "FAIL", message);
                    std::exit(detected ? 0 : 1);
                });
            }
            // A name change invalidates the cached header without changing the valid graph's arithmetic or layout.
            ggml_set_name(out, "split_chain_last_cold");
            ggml_backend_tensor_get(out, result.data(), 0, ggml_nbytes(out));
            ok = !cycle && result == data && ok;
            printf("%s cold split cache depth %d owners %lld/%lld/%lld/%lld\n", ok ? "ok  " : "FAIL", depth,
                (long long) rows[0], (long long) rows[1], (long long) rows[2], (long long) rows[3]);
        }
    }

    ggml_context * leaves = ggml_init({2*ggml_tensor_overhead(), nullptr, true});
    ggml_tensor * a = ggml_new_tensor_2d(leaves, GGML_TYPE_F32, 4, 8);
    ggml_tensor * b = ggml_new_tensor_2d(leaves, GGML_TYPE_F32, 4, 8);
    for (auto * leaf : {a, b}) { ggml_backend_meta_split_input(leaf, 1, layouts[0], 4); }
    ggml_backend_buffer_t leaf_buffer = ggml_backend_alloc_ctx_tensors_from_buft(leaves, ggml_backend_get_default_buffer_type(backend));
    GGML_ASSERT(leaf_buffer);
    ggml_reset(ctx);
    ggml_tensor * sum = ggml_add(ctx, a, b);
    ggml_set_output(sum);
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, graph_size, false);
    ggml_build_forward_expand(graph, sum);
    GGML_ASSERT(ggml_gallocr_alloc_graph(alloc, graph));
    std::vector<float> data(32), result(32);
    for (size_t i = 0; i < data.size(); ++i) { data[i] = float(i + 1); }
    for (auto * leaf : {a, b}) { ggml_backend_tensor_set(leaf, data.data(), 0, ggml_nbytes(leaf)); }
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    ggml_backend_tensor_get(sum, result.data(), 0, ggml_nbytes(sum));
    for (size_t i = 0; i < data.size(); ++i) { ok = result[i] == 2*data[i] && ok; }
    // A fresh root keeps its entry lookup from clearing the cache. The stale later sibling clears it only after the
    // earlier sibling has resolved, forcing the root to resolve that earlier sibling again before it can finish.
    ggml_set_name(b, "changed_late_dependency");
    ggml_context * root_ctx = ggml_init({ggml_tensor_overhead() + ggml_graph_overhead_custom(graph_size, false), nullptr, true});
    ggml_tensor * cold_root = ggml_add(root_ctx, a, b);
    ggml_set_output(cold_root);
    ggml_cgraph * cold_graph = ggml_new_graph_custom(root_ctx, graph_size, false);
    ggml_build_forward_expand(cold_graph, cold_root);
    ggml_backend_buffer_t root_buffer = ggml_backend_alloc_ctx_tensors_from_buft(root_ctx, ggml_backend_get_default_buffer_type(backend));
    GGML_ASSERT(root_buffer);
    ggml_backend_buffer_free(root_buffer);
    cold_root->buffer = nullptr;
    cold_root->data = nullptr;
    GGML_ASSERT(ggml_gallocr_alloc_graph(alloc, cold_graph));
    GGML_ASSERT(ggml_backend_graph_compute(backend, cold_graph) == GGML_STATUS_SUCCESS);
    ggml_backend_tensor_get(cold_root, result.data(), 0, ggml_nbytes(cold_root));
    for (size_t i = 0; i < data.size(); ++i) { ok = result[i] == 2*data[i] && ok; }
    printf("%s split cache invalidation during dependency traversal\n", ok ? "ok  " : "FAIL");
    ggml_free(root_ctx);

    ggml_reset(ctx);
    ggml_tensor * shared = ggml_scale(ctx, a, 2.0f);
    ggml_tensor * left = ggml_scale(ctx, shared, 3.0f);
    ggml_tensor * right = ggml_scale(ctx, shared, 5.0f);
    ggml_tensor * joined = ggml_add(ctx, left, right);
    ggml_set_output(joined);
    graph = ggml_new_graph_custom(ctx, graph_size, false);
    ggml_build_forward_expand(graph, joined);
    GGML_ASSERT(ggml_gallocr_alloc_graph(alloc, graph));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    ggml_backend_tensor_get(joined, result.data(), 0, ggml_nbytes(joined));
    for (size_t i = 0; i < data.size(); ++i) { ok = result[i] == 16*data[i] && ok; }
    ggml_set_name(joined, "cold_shared_descendant");
    ggml_backend_tensor_get(joined, result.data(), 0, ggml_nbytes(joined));
    for (size_t i = 0; i < data.size(); ++i) { ok = result[i] == 16*data[i] && ok; }
    printf("%s split cache shared descendant branches\n", ok ? "ok  " : "FAIL");

    ggml_reset(ctx);
    ggml_tensor * view = ggml_view_2d(ctx, a, 4, 4, 2*a->nb[1], 0);
    GGML_ASSERT(!ggml_is_contiguous(view));
    for (int i = 0; i < 512; ++i) { view = ggml_view_2d(ctx, view, 4, 4, view->nb[1], 0); }
    ggml_tensor * deep_view = ggml_scale(ctx, view, 1.0f);
    ggml_set_output(deep_view);
    graph = ggml_new_graph_custom(ctx, graph_size, false);
    ggml_build_forward_expand(graph, deep_view);
    GGML_ASSERT(ggml_gallocr_alloc_graph(alloc, graph));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    std::vector<float> view_expected(16), view_result(16);
    for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col) { view_expected[row*4 + col] = data[row*8 + col]; }
    }
    ggml_backend_tensor_get(deep_view, view_result.data(), 0, ggml_nbytes(deep_view));
    ok = view_result == view_expected && ok;
    ggml_set_name(deep_view, "cold_nested_view_chain");
    ggml_backend_tensor_get(deep_view, view_result.data(), 0, ggml_nbytes(deep_view));
    ok = view_result == view_expected && ok;
    printf("%s split cache nested view chain depth 512\n", ok ? "ok  " : "FAIL");
    ggml_gallocr_free(alloc);
    ggml_free(ctx);
    ggml_backend_buffer_free(leaf_buffer);
    ggml_free(leaves);
    ggml_backend_free(backend);
    return ok;
}

static bool test_scatter_views(ggml_backend_dev_t device) {
    ggml_backend_dev_t devices[] = {device, second_device(device), device, second_device(device)};
    ggml_backend_t backend = ggml_backend_dev_init(ggml_backend_meta_device(devices, 4, split_rows, nullptr), nullptr);
    bool ok = true;
    for (int kind = 0; kind < 3; ++kind) {
        ggml_context * ctx = ggml_init({16*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true});
        ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 32, 8);
        ggml_set_input(input);
        const int64_t rows[] = {1, 0, 2, 5};
        ggml_tensor * owned = ggml_backend_meta_scatter(ctx, input, 1, rows, 4);
        ggml_tensor * view;
        if (kind == 0) {
            view = ggml_reshape_3d(ctx, owned, 16, 2, 8);
            view = ggml_cont(ctx, ggml_permute(ctx, view, 1, 0, 2, 3));
        } else {
            view = ggml_view_2d(ctx, owned, 32, 5, owned->nb[1], 2*owned->nb[1]);
            if (kind == 2) { view = ggml_view_2d(ctx, view, 32, 2, view->nb[1], view->nb[1]); }
        }
        ggml_tensor * out = ggml_scale(ctx, view, 2.0f);
        ggml_set_output(out);
        ggml_cgraph * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, out);
        ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        GGML_ASSERT(ggml_gallocr_alloc_graph(alloc, graph));
        std::vector<float> data(ggml_nelements(input)), result(ggml_nelements(out));
        for (size_t i = 0; i < data.size(); ++i) { data[i] = i + 1; }
        ggml_backend_tensor_set(input, data.data(), 0, ggml_nbytes(input));
        GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
        ggml_backend_tensor_get(out, result.data(), 0, ggml_nbytes(out));
        bool case_ok = true;
        for (size_t i = 0; i < result.size(); ++i) {
            const size_t index = kind == 0 ? i/32*32 + i%2*16 + i%32/2 : (kind == 1 ? 64 : 96) + i;
            case_ok = case_ok && result[i] == 2.0f*data[index];
        }
        printf("%s scatter view %d\n", case_ok ? "ok  " : "FAIL", kind);
        ok = case_ok && ok;
        ggml_gallocr_free(alloc);
        ggml_free(ctx);
    }
    ggml_backend_free(backend);
    return ok;
}

static bool test_owned_cache_rows(ggml_backend_dev_t device, ggml_type type, bool mapped = false) {
    ggml_backend_dev_t devices[] = {device, second_device(device), device, second_device(device)};
    if (host_devices[0]) { std::copy(host_devices.begin(), host_devices.end(), devices); }
    ggml_backend_t backend = ggml_backend_dev_init(ggml_backend_meta_device(devices, 4, split_rows, nullptr), nullptr);
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    ggml_context * weights = ggml_init({ggml_tensor_overhead(), nullptr, true});
    ggml_tensor * cache = ggml_new_tensor_2d(weights, type, 64, 16);
    const int64_t capacity[] = {2, 4, 3, 7};
    const int64_t active[] = {1, 0, 2, 1};
    ggml_backend_meta_split_input(cache, 1, capacity, 4);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors_from_buft(weights, mapped ?
        ggml_backend_dev_mapped_buffer_type(ggml_backend_get_device(backend)) : ggml_backend_get_default_buffer_type(backend));
    GGML_ASSERT(buffer);
    std::vector<float> original(64*16), updates(64*4);
    for (size_t i = 0; i < original.size(); ++i) { original[i] = 127.0f*(1 + i/32); }
    for (size_t i = 0; i < updates.size(); ++i) { updates[i] = -127.0f*(1 + i/32); }
    std::vector<uint8_t> bytes(ggml_nbytes(cache));
    ggml_quantize_chunk(type, original.data(), bytes.data(), 0, 16, 64, nullptr);
    ggml_backend_tensor_set(cache, bytes.data(), 0, bytes.size());
    // Checkpoint I/O can touch part of one owner or span several owners.
    const size_t first = cache->nb[1] + 3;
    const size_t count = 9*cache->nb[1] - 7;
    std::vector<uint8_t> checkpoint(count);
    ggml_backend_tensor_get(cache, checkpoint.data(), first, count);
    bool checkpoint_ok = std::equal(checkpoint.begin(), checkpoint.end(), bytes.begin() + first);
    ggml_backend_tensor_memset(cache, 0, first, count);
    ggml_backend_tensor_set(cache, checkpoint.data(), first, count);
    ggml_context * ctx = ggml_init({16*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true});
    ggml_tensor * values = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 4);
    ggml_tensor * writes = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, 4);
    ggml_tensor * reads = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4);
    for (ggml_tensor * input : {values, writes, reads}) {
        ggml_set_input(input);
        ggml_backend_meta_split_input(input, input == values ? 1 : 0, active, 4);
    }
    ggml_tensor * stored = ggml_backend_meta_set_rows_local(ctx, cache, values, writes);
    ggml_tensor * out = ggml_backend_meta_get_rows_local(ctx, stored, reads);
    ggml_tensor * full = ggml_cast(ctx, stored, GGML_TYPE_F32);
    ggml_set_output(out);
    ggml_set_output(full);
    bool ok = checkpoint_ok && !ggml_backend_dev_supports_op(ggml_backend_get_device(cpu), stored) &&
              !ggml_backend_dev_supports_op(ggml_backend_get_device(cpu), out);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);
    ggml_build_forward_expand(graph, full);
    ggml_backend_t backends[] = {backend, cpu};
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE, false, true);
    GGML_ASSERT(ggml_backend_sched_alloc_graph(sched, graph));
    const int64_t local_writes[] = {1, 0, 2, 6};
    const int32_t local_reads[] = {1, 0, 2, 6};
    const int32_t global_rows[] = {1, 6, 8, 15};
    ggml_backend_tensor_set(values, updates.data(), 0, ggml_nbytes(values));
    ggml_backend_tensor_set(writes, local_writes, 0, ggml_nbytes(writes));
    ggml_backend_tensor_set(reads, local_reads, 0, ggml_nbytes(reads));
    GGML_ASSERT(ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS);
    std::vector<float> result(64*4), result_full(64*16);
    ggml_backend_tensor_get(out, result.data(), 0, ggml_nbytes(out));
    ggml_backend_tensor_get(full, result_full.data(), 0, ggml_nbytes(full));
    for (size_t row = 0; row < 4; ++row) {
        std::copy(updates.begin() + row*64, updates.begin() + (row + 1)*64, original.begin() + global_rows[row]*64);
    }
    ok = ok && result == updates && result_full == original;
    printf("%s owned %s cache reads/writes%s with unequal capacity and active rows\n", ok ? "ok  " : "FAIL", ggml_type_name(type), mapped ? " (mapped host cache)" : "");
    ggml_backend_sched_free(sched);
    ggml_free(ctx);
    ggml_backend_buffer_free(buffer);
    ggml_free(weights);
    ggml_backend_free(cpu);
    ggml_backend_free(backend);
    return ok;
}

static bool test_host_layout(ggml_backend_dev_t device) {
    ggml_backend_dev_t devices[] = {device, second_device(device), device, second_device(device)};
    auto meta = ggml_backend_dev_init(ggml_backend_meta_device(devices, 4, split_rows, nullptr), nullptr);
    auto cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    bool ok = true;
    for (int kind = 0; kind < 3; ++kind) {
        auto ctx = ggml_init({16*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true});
        const int64_t owners[] = {1, 0, 2, 1}, flat_owners[] = {8, 0, 16, 8};
        auto root = kind == 2 ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 32) : ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 4);
        ggml_backend_meta_split_input(root, kind == 2 ? 0 : 1, kind == 2 ? flat_owners : owners, 4);
        ggml_set_input(root);
        auto view = kind == 2 ? ggml_view_2d(ctx, root, 4, 4, 8*sizeof(float), 2*sizeof(float)) :
            ggml_reshape_4d(ctx, root, 2, 4, 1, 4);
        if (kind == 1) { view = ggml_cast(ctx, ggml_cast(ctx, view, GGML_TYPE_F16), GGML_TYPE_F32); }
        auto x = ggml_dup_tensor(ctx, view);
        ggml_backend_meta_split_input(x, kind == 2 ? 1 : 3, owners, 4);
        ggml_set_input(x);
        auto out = ggml_add(ctx, x, view);
        ggml_set_output(out);
        auto graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, out);
        ggml_backend_t backends[] = {meta, cpu};
        auto sched = ggml_backend_sched_new(backends, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE, false, true);
        ggml_backend_sched_set_tensor_backend(sched, root, cpu);
        ggml_backend_sched_set_tensor_backend(sched, view, cpu);
        ggml_backend_sched_set_tensor_backend(sched, x, meta);
        ggml_backend_sched_set_tensor_backend(sched, out, meta);
        GGML_ASSERT(ggml_backend_sched_alloc_graph(sched, graph));
        std::vector<float> data(32), values(ggml_nelements(x), 100.0f), result(values.size());
        for (size_t i = 0; i < data.size(); ++i) { data[i] = i + 1; }
        ggml_backend_tensor_set(root, data.data(), 0, ggml_nbytes(root));
        ggml_backend_tensor_set(x, values.data(), 0, ggml_nbytes(x));
        GGML_ASSERT(ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS);
        ggml_backend_tensor_get(out, result.data(), 0, ggml_nbytes(out));
        bool case_ok = true;
        for (size_t i = 0; i < result.size(); ++i) {
            case_ok = case_ok && result[i] == 100.0f + data[kind == 2 ? i/4*8 + i%4 + 2 : i];
        }
        printf("%s host-owned input reshape/cast/chunk case %d\n", case_ok ? "ok  " : "FAIL", kind);
        ok = ok && case_ok;
        ggml_backend_sched_free(sched);
        ggml_free(ctx);
    }
    ggml_backend_free(meta);
    ggml_backend_free(cpu);
    return ok;
}

static bool test_stream_view(ggml_backend_dev_t device, ggml_type type, bool single = false, bool mapped = false) {
    ggml_backend_dev_t devices[] = {device, second_device(device), device, second_device(device)};
    if (host_devices[0]) { std::copy(host_devices.begin(), host_devices.end(), devices); }
    auto backend = ggml_backend_dev_init(ggml_backend_meta_device(devices, 4, split_rows, nullptr), nullptr);
    auto ctx = ggml_init({16*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true});
    auto cache_ctx = mapped ? ggml_init({ggml_tensor_overhead(), nullptr, true}) : nullptr;
    auto cache = ggml_new_tensor_3d(mapped ? cache_ctx : ctx, type, 64, 64, 16);
    const int64_t capacity[] = {4, 4, 4, 4}, first[] = {2, 0, 1, 3};
    const int64_t active[] = {single ? 0 : 1, 0, single ? 1 : 2, single ? 0 : 1};
    const int64_t n_streams = single ? 1 : 4;
    ggml_backend_meta_split_input(cache, 2, capacity, 4);
    ggml_set_input(cache);
    ggml_backend_buffer_t cache_buffer = nullptr;
    if (mapped) {
        auto buft = ggml_backend_dev_mapped_buffer_type(ggml_backend_get_device(backend));
        GGML_ASSERT(buft && ggml_backend_dev_supports_buft(ggml_backend_get_device(backend), buft));
        cache_buffer = ggml_backend_alloc_ctx_tensors_from_buft(cache_ctx, buft);
        GGML_ASSERT(cache_buffer);
    }
    auto view = ggml_view_4d(ctx, cache, 32, 2, 32, n_streams, ggml_row_size(type, 32), cache->nb[1], cache->nb[2], 0);
    ggml_backend_meta_set_stream_view(view, active, first, 4);
    auto nested = ggml_view_3d(ctx, view, 32, 32, n_streams, view->nb[2], view->nb[3], view->nb[1]);
    auto out = ggml_cont(ctx, ggml_cast(ctx, nested, GGML_TYPE_F32));
    ggml_set_output(out);
    auto graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);
    auto alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    GGML_ASSERT(ggml_gallocr_alloc_graph(alloc, graph));
    std::vector<float> data(ggml_nelements(cache));
    for (size_t i = 0; i < data.size(); ++i) { data[i] = 127.0f*(1 + i/32); }
    std::vector<uint8_t> bytes(ggml_nbytes(cache));
    ggml_quantize_chunk(type, data.data(), bytes.data(), 0, 64*16, 64, nullptr);
    ggml_backend_tensor_set(cache, bytes.data(), 0, bytes.size());
    bool ok = true;
    if (single) {
        GGML_ASSERT(ggml_is_contiguous(view) && view->nb[3] > ggml_nbytes(view));
        std::vector<uint8_t> transferred(ggml_nbytes(view)), unchanged(bytes.size());
        const auto expected = bytes.data() + 9*cache->nb[2];
        ggml_backend_tensor_set_async(backend, view, expected, 0, transferred.size());
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_get(view, transferred.data(), 0, transferred.size());
        ok = std::equal(transferred.begin(), transferred.end(), expected);
        ggml_backend_tensor_memset(view, 0, 3, transferred.size() - 7);
        ggml_backend_tensor_set_async(backend, view, expected + 3, 3, transferred.size() - 7);
        ggml_backend_tensor_get_async(backend, view, transferred.data(), 0, transferred.size());
        ggml_backend_synchronize(backend);
        ok = ok && std::equal(transferred.begin(), transferred.end(), expected);
        ggml_backend_tensor_get(cache, unchanged.data(), 0, unchanged.size());
        ok = ok && unchanged == bytes;
    }
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    std::vector<float> result(ggml_nelements(out));
    ggml_backend_tensor_get(out, result.data(), 0, ggml_nbytes(out));
    const size_t streams[] = {2, 9, 10, 15};
    std::vector<uint8_t> async_result(ggml_nbytes(out));
    const size_t part = async_result.size()/3 + 7;
    for (size_t offset = 0; offset < async_result.size(); offset += part) {
        ggml_backend_tensor_get_async(backend, out, async_result.data() + offset, offset,
                std::min(part, async_result.size() - offset));
    }
    ggml_backend_synchronize(backend);
    ok = ok && std::memcmp(async_result.data(), result.data(), async_result.size()) == 0;
    for (size_t s = 0; s < (size_t) n_streams; ++s) {
        for (size_t row = 0; row < 32; ++row) {
            for (size_t col = 0; col < 32; ++col) {
                ok = ok && result[s*32*32 + row*32 + col] == data[(single ? 9 : streams[s])*64*64 + row*64 + 32 + col];
            }
        }
    }
    printf("%s %s %s stream views%s with empty owner, unequal starts and nested channel slice\n", ok ? "ok  " : "FAIL", ggml_type_name(type), single ? "single" : "active", mapped ? " (mapped host cache)" : "");
    ggml_gallocr_free(alloc);
    ggml_free(ctx);
    if (cache_buffer) { ggml_backend_buffer_free(cache_buffer); ggml_free(cache_ctx); }
    ggml_backend_free(backend);
    return ok;
}

static bool test_host_attention(ggml_backend_dev_t device, int64_t queries, bool sparse, bool long_context = false, bool masked_sparse = false) {
    ggml_backend_dev_t devices[] = {device, second_device(device), device, second_device(device)};
    if (host_devices[0]) { std::copy(host_devices.begin(), host_devices.end(), devices); }
    auto backend = ggml_backend_dev_init(ggml_backend_meta_device(devices, 4, split_rows, nullptr), nullptr);
    auto cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    const int64_t cells = long_context ? 409600 : masked_sparse ? 8192 : 256;
    const int64_t selected = long_context ? 2080 : 32;
    const int64_t owners[] = {1, 0, 2, 1};
    std::vector<float> result[2];
    int captures[2] = {};
    for (int mapped = 0; mapped < 2; ++mapped) {
        const int captured_before = captured;
        auto weights = ggml_init({ggml_tensor_overhead(), nullptr, true});
        auto cache = ggml_new_tensor_3d(weights, GGML_TYPE_Q8_0, 512, cells, 4);
        ggml_backend_meta_split_input(cache, 2, owners, 4);
        auto buffer = ggml_backend_alloc_ctx_tensors_from_buft(weights, mapped ?
            ggml_backend_dev_mapped_buffer_type(ggml_backend_get_device(backend)) : ggml_backend_get_default_buffer_type(backend));
        GGML_ASSERT(buffer);
        auto ctx = ggml_init({16*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true});
        auto k = ggml_reshape_4d(ctx, cache, 512, cells, 1, 4);
        auto v = ggml_view_4d(ctx, k, 512, cells, 1, 4, k->nb[1], k->nb[2], k->nb[3], 0);
        auto q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 512, queries, 8, 4);
        auto selection = sparse ? ggml_new_tensor_4d(ctx, GGML_TYPE_I32, selected, queries, 1, 4) :
            ggml_new_tensor_4d(ctx, GGML_TYPE_F16, cells, (queries + 31)/32*32, 1, 4);
        for (auto input : {q, selection}) { ggml_set_input(input); ggml_backend_meta_split_input(input, 3, owners, 4); }
        auto out = sparse ? ggml_flash_attn_ext_sparse(ctx, q, k, v, selection, 1.0f/sqrtf(512.0f)) :
            ggml_flash_attn_ext(ctx, q, k, v, selection, 1.0f/sqrtf(512.0f), 0.0f, 0.0f);
        if (masked_sparse) { ggml_flash_attn_ext_set_n_kv_max(out, 2080); }
        ggml_set_output(out);
        auto graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, out);
        ggml_backend_t backends[] = {backend, cpu};
        auto sched = ggml_backend_sched_new(backends, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE, false, true);
        for (auto input : {q, selection}) { ggml_backend_sched_set_tensor_backend(sched, input, backend); }
        GGML_ASSERT(ggml_backend_sched_alloc_graph(sched, graph));
        std::vector<float> cache_data(512*256), query_data(ggml_nelements(q));
        std::vector<uint8_t> cache_bytes(ggml_row_size(GGML_TYPE_Q8_0, 512)*cells);
        const size_t chunk_bytes = ggml_row_size(GGML_TYPE_Q8_0, 512)*256;
        for (size_t i = 0; i < query_data.size(); ++i) { query_data[i] = float(int(i % 31) - 15)/32; }
        ggml_backend_tensor_set(q, query_data.data(), 0, ggml_nbytes(q));
        result[mapped].resize(6*ggml_nelements(out));
        for (int iteration = 0; iteration < 6; ++iteration) {
            for (int seq = 0; seq < 4; ++seq) {
                for (size_t i = 0; i < cache_data.size(); ++i) { cache_data[i] = float(int((i*13 + seq*7 + iteration*11) % 127) - 63)/64; }
                ggml_quantize_chunk(GGML_TYPE_Q8_0, cache_data.data(), cache_bytes.data(), 0, 256, 512, nullptr);
                if (iteration == 0) {
                    for (size_t offset = chunk_bytes; offset < cache_bytes.size(); offset += chunk_bytes) {
                        std::copy_n(cache_bytes.data(), chunk_bytes, cache_bytes.data() + offset);
                    }
                }
                ggml_backend_tensor_set(cache, cache_bytes.data(), seq*cache_bytes.size(), iteration == 0 ? cache_bytes.size() : chunk_bytes);
            }
            if (sparse) {
                std::vector<int32_t> ids(ggml_nelements(selection), -1);
                for (int64_t row = 0; row < queries*4; ++row) {
                    for (int64_t i = 0; i < selected - 3 - iteration; ++i) {
                        ids[row*selected + i] = int32_t(i*(cells/selected) + (row + iteration) % (cells/selected));
                    }
                }
                if (long_context) {
                    const size_t row_bytes = ggml_row_size(GGML_TYPE_Q8_0, 512);
                    std::vector<float> sentinel(512);
                    std::vector<uint8_t> quantized(row_bytes);
                    for (int seq = 0; seq < 4; ++seq) {
                        for (int64_t query = 0; query < queries; ++query) {
                            for (int64_t i = 0; i < selected - 3 - iteration; i += 64) {
                                const int32_t cell = ids[(seq*queries + query)*selected + i];
                                std::seed_seq seed {seq, cell, iteration};
                                std::mt19937 rng(seed);
                                std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
                                for (auto & value : sentinel) { value = distribution(rng); }
                                ggml_quantize_chunk(GGML_TYPE_Q8_0, sentinel.data(), quantized.data(), 0, 1, 512, nullptr);
                                ggml_backend_tensor_set(cache, quantized.data(), seq*cache_bytes.size() + cell*row_bytes, row_bytes);
                            }
                        }
                    }
                }
                ggml_backend_tensor_set(selection, ids.data(), 0, ggml_nbytes(selection));
            } else {
                std::vector<ggml_fp16_t> mask(ggml_nelements(selection));
                for (size_t i = 0; i < mask.size(); ++i) {
                    const bool selected_cell = i % cells < size_t(cells - 7 - iteration) && (!masked_sparse || i % 4 == 0);
                    mask[i] = ggml_fp32_to_fp16(selected_cell ? 0.0f : -INFINITY);
                }
                ggml_backend_tensor_set(selection, mask.data(), 0, ggml_nbytes(selection));
            }
            GGML_ASSERT(ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS);
            ggml_backend_tensor_get(out, result[mapped].data() + iteration*ggml_nelements(out), 0, ggml_nbytes(out));
        }
        captures[mapped] = captured - captured_before;
        ggml_backend_sched_free(sched);
        ggml_free(ctx);
        ggml_backend_buffer_free(buffer);
        ggml_free(weights);
    }
    double error = 0, norm = 0;
    for (size_t i = 0; i < result[0].size(); ++i) {
        const double delta = result[0][i] - result[1][i];
        error += delta*delta;
        norm += double(result[0][i])*result[0][i];
    }
    bool ok = std::isfinite(error) && error <= 1e-9*norm;
    if (ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(device), "ggml_backend_capture_end")) {
        ok = ok && captures[0] == 4 && captures[1] == 4;
    }
    printf("%s mapped Q8 attention q=%lld cells=%lld sparse=%d, changed cache/selection over six evaluations, NMSE %.9g, resident/mapped captures %d/%d\n",
        ok ? "ok  " : "FAIL", (long long) queries, (long long) cells, sparse, error/norm, captures[0], captures[1]);
    ggml_backend_free(backend);
    ggml_backend_free(cpu);
    return ok;
}

static bool test_experts(ggml_backend_dev_t gpu, ggml_type type = GGML_TYPE_F32) {
    const int captured_before = captured;
    const auto hits_before = expert_hits;
    ggml_backend_dev_t devs[] = {gpu, second_device(gpu), gpu, second_device(gpu)};
    ggml_backend_dev_t meta_dev = ggml_backend_meta_device(devs, 4, split_rows, nullptr);
    ggml_backend_t meta = ggml_backend_dev_init(meta_dev, nullptr);
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    // six routes through the experts, unsplit on the CPU or split over the meta backend's four ranks
    const auto run = [&](ggml_backend_t backend, bool partition) {
        std::vector<float> result;
        ggml_context * weights = ggml_init({8*ggml_tensor_overhead(), nullptr, true});
        ggml_tensor * w[3];
        const int64_t counts[] = {2, 2, 2, 2};
        for (auto & tensor : w) {
            tensor = ggml_new_tensor_3d(weights, type, 256, 256, 8);
            if (partition) { ggml_backend_meta_split_input(tensor, 2, counts, 4); }
        }
        ggml_set_name(w[1], "blk.0.ffn_up_exps.weight");
        ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(weights, backend);
        GGML_ASSERT(buffer);
        std::vector<float> data(ggml_nelements(w[0]));
        std::vector<uint8_t> quantized(ggml_nbytes(w[0]));
        std::vector<float> importance(256, 1.0f);
        for (int j = 0; j < 3; ++j) {
            for (size_t i = 0; i < data.size(); ++i) { data[i] = float(int((i*13 + j*7) % 31) - 15)/128; }
            if (type == GGML_TYPE_F32) {
                ggml_backend_tensor_set(w[j], data.data(), 0, ggml_nbytes(w[j]));
            } else {
                ggml_quantize_chunk(type, data.data(), quantized.data(), 0, 256*8, 256, importance.data());
                ggml_backend_tensor_set(w[j], quantized.data(), 0, quantized.size());
            }
        }
        ggml_context * ctx = ggml_init({64*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true});
        ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 256, 5);
        ggml_set_input(x);
        const int64_t rows[] = {1, 0, 3, 1};
        if (partition) { ggml_backend_meta_split_input(x, 1, rows, 4); }
        ggml_tensor * full = partition ? ggml_backend_meta_gather(ctx, x) : x;
        full = ggml_reshape_3d(ctx, full, 256, 1, 5);
        ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 2, 5);
        ggml_set_input(ids);
        ggml_tensor * routing = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, 2, 5);
        ggml_set_input(routing);
        ggml_tensor * gate = ggml_mul_mat_id(ctx, w[0], full, ids);
        ggml_tensor * up = ggml_mul_mat_id(ctx, w[1], full, ids);
        ggml_tensor * out = ggml_mul_mat_id(ctx, w[2], ggml_swiglu_split(ctx, gate, up), ids);
        out = ggml_mul(ctx, out, routing);
        out = ggml_add(ctx, ggml_view_2d(ctx, out, 256, 5, out->nb[2], 0),
                            ggml_view_2d(ctx, out, 256, 5, out->nb[2], out->nb[1]));
        if (partition) { out = ggml_backend_meta_scatter(ctx, ggml_backend_meta_reduce(ctx, out), 1, rows, 4); }
        out = ggml_scale(ctx, out, 0.5f);
        ggml_set_output(out);
        ggml_cgraph * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, out);
        ggml_backend_t backends[] = {backend, cpu};
        ggml_backend_sched_t sched = ggml_backend_sched_new(backends, nullptr, partition ? 2 : 1, GGML_DEFAULT_GRAPH_SIZE, false, true);
        ggml_backend_sched_set_tensor_backend(sched, x, backend);
        ggml_backend_sched_set_tensor_backend(sched, ids, backend);
        ggml_backend_sched_set_tensor_backend(sched, routing, backend);
        GGML_ASSERT(ggml_backend_sched_alloc_graph(sched, graph));
        std::vector<float> input(256*5);
        for (size_t i = 0; i < input.size(); ++i) { input[i] = float(int(i % 17) - 8)/16; }
        // The last rank has no selected experts; rows also mix local and remote expert lanes.
        const int32_t selected[] = {0, 1, 2, 3, 1, 4, 3, 0, 5, 2};
        result.resize(6*256*5);
        for (int iteration = 0; iteration < 6; ++iteration) {
            int32_t routed[10];
            float probabilities[10];
            for (int i = 0; i < 10; ++i) { routed[i] = (selected[i] + 2*iteration) % 8; }
            for (int i = 0; i < 10; ++i) { probabilities[i] = float(1 + (i + iteration) % 3)/4; }
            ggml_backend_tensor_set(x, input.data(), 0, ggml_nbytes(x));
            ggml_backend_tensor_set(ids, routed, 0, ggml_nbytes(ids));
            ggml_backend_tensor_set(routing, probabilities, 0, ggml_nbytes(routing));
            GGML_ASSERT(ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS);
            ggml_backend_tensor_get(out, result.data() + iteration*256*5, 0, ggml_nbytes(out));
        }
        ggml_backend_sched_free(sched);
        ggml_free(ctx);
        ggml_backend_buffer_free(buffer);
        ggml_free(weights);
        return result;
    };
    const std::vector<float> results[2] = {run(cpu, false), run(meta, true)};
    double error = 0, norm = 0;
    for (size_t i = 0; i < results[0].size(); ++i) {
        const double delta = results[1][i] - results[0][i];
        error += delta*delta;
        norm += double(results[0][i])*results[0][i];
    }
    bool ok = std::isfinite(error) && error < (type == GGML_TYPE_F32 ? 1e-6 : 5e-4)*norm;
    if (ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(gpu), "ggml_backend_capture_end")) {
        ok = ok && captured - captured_before == 4;
    }
    printf("%s %s expert partition against unsplit CPU, NMSE %.9g, %d captures\n", ok ? "ok  " : "FAIL", ggml_type_name(type), error/norm, captured - captured_before);
    ggml_backend_free(meta);
    if (strcmp(ggml_backend_reg_name(ggml_backend_dev_backend_reg(gpu)), "CUDA") == 0) {
        const auto hits = expert_hits - hits_before;
        printf("%s expert telemetry: %llu hits in six routes, expected 60\n", hits == 60 ? "ok  " : "FAIL", hits);
        ok = ok && hits == 60;

        // -sm dp reduces the experts' sum before a scatter by rows: each card takes only its own rows of it, and the
        // output is the one the reduction to every card gives, bit for bit (the internal AllReduce's pipeline counts
        // its reductions and logs them when it is freed). The sum goes the chunked kernel's way, as a 36- to 108-row
        // verify's at 4096 wide does; this one, 5 KB, would otherwise take the LL kernel.
        std::vector<float> outputs[2];
        unsigned long long to_owners[2], to_all[2];
        set_env("GGML_CUDA_ALLREDUCE", "internal");
        set_env("GGML_CUDA_AR_LL_MAX_BYTES", "0");
        set_env("GGML_CUDA_AR_P2P_MAX_BYTES", "0");
        for (int all = 0; all < 2; ++all) {
            set_env("GGML_CUDA_AR_REDUCE_OWNERS", all ? "0" : nullptr);
            const auto owners_before = reductions_to_owners, all_before = reductions_to_all;
            ggml_backend_t backend = ggml_backend_dev_init(meta_dev, nullptr);
            outputs[all] = run(backend, true);
            ggml_backend_free(backend);
            to_owners[all] = reductions_to_owners - owners_before;
            to_all[all] = reductions_to_all - all_before;
        }
        set_env("GGML_CUDA_AR_REDUCE_OWNERS", nullptr);
        set_env("GGML_CUDA_AR_P2P_MAX_BYTES", nullptr);
        set_env("GGML_CUDA_AR_LL_MAX_BYTES", nullptr);
        set_env("GGML_CUDA_ALLREDUCE", nullptr);
        const bool same = memcmp(outputs[0].data(), outputs[1].data(), outputs[0].size()*sizeof(float)) == 0;
        const bool owned = to_owners[0] > 0 && to_all[0] == 0 && to_owners[1] == 0 && to_all[1] > 0;
        printf("%s %s experts' sum to its rows' owners: %llu reductions to owners and %llu to every card, against %llu "
               "and %llu with GGML_CUDA_AR_REDUCE_OWNERS=0; outputs %s\n", same && owned ? "ok  " : "FAIL", ggml_type_name(type),
               to_owners[0], to_all[0], to_owners[1], to_all[1], same ? "equal bit for bit" : "differ");
        ok = ok && same && owned;
    }
    ggml_backend_free(cpu);
    return ok;
}

static bool test_owned_ops(ggml_backend_dev_t gpu, int variant, bool single_owner = false) {
    ggml_backend_dev_t devs[] = {gpu, second_device(gpu), gpu, second_device(gpu)};
    ggml_backend_t meta = ggml_backend_dev_init(ggml_backend_meta_device(devs, 4, split_rows, nullptr), nullptr);
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    std::vector<float> results[2];
    const int64_t owners[] = {1, 0, 2, 1};
    const int64_t n_stream = single_owner ? 1 : 4;
    for (int mode = 0; mode < 2; ++mode) {
        ggml_backend_t backend = mode ? meta : cpu;
        ggml_context * ctx = ggml_init({32*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true});
        std::vector<ggml_tensor *> inputs;
        std::vector<ggml_tensor *> outputs;
        auto input = [&](ggml_type type, std::array<int64_t, 4> ne, int axis) {
            if (single_owner && axis >= 0) { ne[axis] = 1; }
            if (single_owner && axis >= 0) {
                const int64_t size = ne[0]*ne[1]*ne[2]*ne[3];
                auto * flat = ggml_new_tensor_1d(ctx, type, size);
                ggml_set_input(flat);
                const int64_t owned[] = {0, 0, size, 0};
                if (mode) { ggml_backend_meta_split_input(flat, 0, owned, 4); }
                inputs.push_back(flat);
                return ggml_reshape_4d(ctx, flat, ne[0], ne[1], ne[2], ne[3]);
            }
            ggml_tensor * t = ggml_new_tensor(ctx, type, 4, ne.data());
            ggml_set_input(t);
            if (mode && axis >= 0) { ggml_backend_meta_split_input(t, axis, owners, 4); }
            inputs.push_back(t);
            return t;
        };
        ggml_tensor * out;
        if (variant == 11) {
            const int64_t first[] = {0, 0, 0, 0};
            const int64_t one[] = {0, 0, 1, 0};
            auto * cache = ggml_new_tensor_3d(ctx, GGML_TYPE_Q8_0, 512, 256, n_stream);
            ggml_set_input(cache);
            inputs.push_back(cache);
            if (mode) { ggml_backend_meta_split_input(cache, 2, single_owner ? one : owners, 4); }
            auto * view = ggml_view_4d(ctx, cache, 512, 1, 256, n_stream, cache->nb[1], cache->nb[1], cache->nb[2], 0);
            if (mode) { ggml_backend_meta_set_stream_view(view, single_owner ? one : owners, first, 4); }
            auto * k = ggml_permute(ctx, view, 0, 2, 1, 3);
            auto * v = ggml_view_4d(ctx, k, k->ne[0], k->ne[1], k->ne[2], k->ne[3], k->nb[1], k->nb[2], k->nb[3], 0);
            auto * q = input(GGML_TYPE_F32, {512, 16, 4, 4}, 3);
            auto * mask = input(GGML_TYPE_F16, {256, 32, 1, 4}, 3);
            out = ggml_flash_attn_ext(ctx, q, k, v, mask, 0.125f, 0.0f, 0.0f);
        } else if (variant == 7) {
            auto * q = input(GGML_TYPE_F32, {128, 2, 4, 1}, 2);
            auto * positions = input(GGML_TYPE_I32, {4, 1, 1, 1}, 0);
            out = ggml_rope(ctx, q, positions, 64, 0);
        } else if (variant == 8) {
            auto * x = input(GGML_TYPE_F32, {6, 128, 4, 1}, 2);
            auto * w = input(GGML_TYPE_F32, {4, 128, 1, 1}, -1);
            out = ggml_ssm_conv(ctx, x, w);
        } else if (variant == 9) {
            auto * x = input(GGML_TYPE_F32, {128, 3, 2, 4}, 3);
            auto * w = input(GGML_TYPE_F32, {128, 64, 2, 1}, -1);
            out = ggml_mul_mat(ctx, w, x);
        } else if (variant == 10) {
            auto * cache = input(GGML_TYPE_F32, {128, 64, 4, 1}, 2);
            auto * indices = input(GGML_TYPE_I32, {8, 4, 1, 1}, 1);
            out = ggml_get_rows(ctx, cache, indices);
        } else if (variant >= 5) {
            const int snapshots = variant == 5 ? 1 : 3;
            ggml_tensor * q = input(GGML_TYPE_F32, {128, 2, 3, 4}, 3);
            ggml_tensor * k = input(GGML_TYPE_F32, {128, 2, 3, 4}, 3);
            ggml_tensor * v = input(GGML_TYPE_F32, {128, 2, 3, 4}, 3);
            ggml_tensor * g = input(GGML_TYPE_F32, {128, 2, 3, 4}, 3);
            ggml_tensor * beta = input(GGML_TYPE_F32, {1, 2, 3, 4}, 3);
            ggml_tensor * state = input(GGML_TYPE_F32, {128, 128, 2, 4}, 3);
            out = ggml_gated_delta_net(ctx, q, k, v, g, beta, state, snapshots);
            outputs.push_back(ggml_scale(ctx, ggml_view_4d(ctx, out, 128, 2, 3, n_stream,
                128*sizeof(float), 256*sizeof(float), 3*256*sizeof(float), 0), 2.0f));
            outputs.push_back(ggml_scale(ctx, ggml_view_3d(ctx, out, 128*128*2, n_stream, snapshots,
                128*128*2*sizeof(float), n_stream*128*128*2*sizeof(float), 3*n_stream*256*sizeof(float)), 2.0f));
        } else if (variant == 3) {
            ggml_tensor * mixes = input(GGML_TYPE_F32, {24, 4, 1, 1}, 1);
            ggml_tensor * scale = input(GGML_TYPE_F32, {3, 1, 1, 1}, -1);
            ggml_tensor * base = input(GGML_TYPE_F32, {24, 1, 1, 1}, -1);
            ggml_tensor * x = input(GGML_TYPE_F32, {64, 4, 4, 1}, 2);
            ggml_tensor * weights = ggml_dsv4_hc_weights(ctx, mixes, scale, base, 1e-6f, 4);
            ggml_tensor * pre = ggml_view_2d(ctx, weights, 4, n_stream, weights->nb[1], 0);
            ggml_tensor * post = ggml_view_2d(ctx, weights, 4, n_stream, weights->nb[1], 4*sizeof(float));
            ggml_tensor * comb = ggml_dsv4_hc_comb(ctx, mixes, scale, base, 1e-6f, 4);
            out = ggml_dsv4_hc_post(ctx, ggml_dsv4_hc_pre(ctx, x, pre), x, post, comb);
        } else if (variant == 4) {
            ggml_tensor * cells = input(GGML_TYPE_I32, {64, 2, 1, 4}, 3);
            ggml_tensor * rows = input(GGML_TYPE_I32, {4, 3, 4, 1}, 2);
            out = ggml_cast(ctx, ggml_kpool_mask(ctx, cells, rows, GGML_KPOOL_MASK_SEL, 0), GGML_TYPE_F32);
        } else {
            const int64_t width = variant == 1 ? 512 : 128;
            const int64_t cells = variant == 1 ? 256 : 64;
            const int64_t heads = variant == 1 ? 8 : variant == 2 ? 32 : 2;
            ggml_tensor * q = input(GGML_TYPE_F32, variant == 2 ? std::array<int64_t, 4>{width, heads, 3, 4} : std::array<int64_t, 4>{width, 3, heads, 4}, 3);
            ggml_tensor * k = input(GGML_TYPE_F16, variant == 2 ? std::array<int64_t, 4>{width, 1, cells, 4} : std::array<int64_t, 4>{width, cells, 1, 4}, 3);
            if (variant == 2) {
                ggml_tensor * w = input(GGML_TYPE_F32, {heads, 3, 1, 4}, 3);
                ggml_tensor * mask = input(GGML_TYPE_F16, {8, 3, 1, 4}, 3);
                ggml_tensor * rows = input(GGML_TYPE_I32, {8, 4, 1, 1}, 1);
                out = ggml_lightning_indexer_rows(ctx, q, k, rows, w, mask);
            } else {
                ggml_tensor * v = input(GGML_TYPE_F16, {width, cells, 1, 4}, 3);
                if (variant == 1) {
                    ggml_tensor * ids = input(GGML_TYPE_I32, {32, 3, 1, 4}, 3);
                    out = ggml_flash_attn_ext_sparse(ctx, q, k, v, ids, 0.125f);
                } else {
                    ggml_tensor * mask = input(GGML_TYPE_F16, {64, 32, 1, 4}, 3);
                    out = ggml_flash_attn_ext(ctx, q, k, v, mask, 0.125f, 0.0f, 0.0f);
                }
            }
        }
        ggml_set_output(out);
        ggml_cgraph * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, out);
        for (auto t : outputs) { ggml_set_output(t); ggml_build_forward_expand(graph, t); }
        ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        GGML_ASSERT(ggml_gallocr_alloc_graph(alloc, graph));
        for (auto t : inputs) {
            std::vector<float> f32(ggml_nelements(t));
            for (size_t i = 0; i < f32.size(); ++i) { f32[i] = float(int(i % 23) - 11)/32; }
            if (t->type == GGML_TYPE_F32) {
                ggml_backend_tensor_set(t, f32.data(), 0, ggml_nbytes(t));
            } else if (t->type == GGML_TYPE_F16) {
                std::vector<ggml_fp16_t> f16(f32.size());
                ggml_fp32_to_fp16_row(f32.data(), f16.data(), f16.size());
                ggml_backend_tensor_set(t, f16.data(), 0, ggml_nbytes(t));
            } else if (ggml_is_quantized(t->type)) {
                std::vector<uint8_t> quantized(ggml_nbytes(t));
                ggml_quantize_chunk(t->type, f32.data(), quantized.data(), 0, ggml_nrows(t), t->ne[0], nullptr);
                ggml_backend_tensor_set(t, quantized.data(), 0, quantized.size());
            } else {
                std::vector<int32_t> ids(f32.size());
                for (size_t i = 0; i < ids.size(); ++i) { ids[i] = variant == 1 ? 2*(i % 32) + (i/32) % 2 : 7*(i % 8) + (i/8) % 8; }
                ggml_backend_tensor_set(t, ids.data(), 0, ggml_nbytes(t));
            }
        }
        GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
        results[mode].resize(ggml_nelements(out));
        ggml_backend_tensor_get(out, results[mode].data(), 0, ggml_nbytes(out));
        for (auto t : outputs) {
            const size_t offset = results[mode].size();
            results[mode].resize(offset + ggml_nelements(t));
            ggml_backend_tensor_get(t, results[mode].data() + offset, 0, ggml_nbytes(t));
        }
        ggml_gallocr_free(alloc);
        ggml_free(ctx);
    }
    double error = 0, norm = 0;
    for (size_t i = 0; i < results[0].size(); ++i) {
        if (results[0][i] == results[1][i] && !std::isfinite(results[0][i])) { continue; }
        const double delta = results[1][i] - results[0][i];
        error += delta*delta;
        norm += double(results[0][i])*results[0][i];
    }
    const bool ok = variant == 4 ? results[0] == results[1] : std::isfinite(error) && error < 1e-6*norm;
    const char * names[] = {"attention", "sparse attention", "indexer", "hyper-connections", "k-pool mask", "KDA final state", "KDA snapshots", "RoPE", "SSM convolution", "batched projection", "batched cache gather", "Q8 aliased attention"};
    printf("%s owned %s%s, NMSE %.9g\n", ok ? "ok  " : "FAIL", names[variant], single_owner ? " (single owner)" : "", norm ? error/norm : error);
    ggml_backend_free(meta);
    ggml_backend_free(cpu);
    return ok;
}

// A graph of rig's two-card profile (-sm tensor), computed by the meta backend over two ranks against the CPU: the
// weights buffer's tensors are split by split_columns, nothing owns rows, and the compute buffer's tensors take their
// splits from them. Variant 0 is glm5next's KDA: the q, k and v projections, split by heads, concatenated for one short
// convolution and read back as three views of the fused tensor; with one token the fused tensor's dims above the split
// axis are 1. Variant 1 is glm5next's DSA: K and V are two permuted views of the one q8_0 latent cache at the same
// offset, as the kv cache hands them to the attention, with the q heads split and the cache and the mask mirrored.
static bool test_tensor_mode(ggml_backend_dev_t gpu, int variant, int64_t n_tokens) {
    ggml_backend_dev_t devs[] = {gpu, second_device(gpu)};
    ggml_backend_t meta = ggml_backend_dev_init(ggml_backend_meta_device(devs, 2, split_columns, nullptr), nullptr);
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    std::vector<float> results[2];
    for (int mode = 0; mode < 2; ++mode) {
        ggml_backend_t backend = mode ? meta : cpu;
        ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
        ggml_context * weights = ggml_init({8*ggml_tensor_overhead(), nullptr, true});
        ggml_context * ctx = ggml_init({32*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true});
        std::vector<ggml_tensor *> inputs;
        auto input = [&](ggml_type type, std::array<int64_t, 4> ne, const char * name) {
            ggml_tensor * t = ggml_new_tensor(weights, type, 4, ne.data());
            ggml_set_name(t, name);
            inputs.push_back(t);
            return t;
        };
        ggml_tensor * out;
        if (variant == 0) {
            const int64_t d = 64;
            ggml_tensor * q = input(GGML_TYPE_F32, {d, n_tokens, 1, 1}, "q");
            ggml_tensor * k = input(GGML_TYPE_F32, {d, n_tokens, 1, 1}, "k");
            ggml_tensor * v = input(GGML_TYPE_F32, {d, n_tokens, 1, 1}, "v");
            ggml_tensor * qkv = ggml_concat(ctx, ggml_concat(ctx, q, k, 0), v, 0);
            ggml_tensor * Qcur = ggml_view_2d(ctx, qkv, d, n_tokens, qkv->nb[1], 0);
            ggml_tensor * Kcur = ggml_view_2d(ctx, qkv, d, n_tokens, qkv->nb[1], d*sizeof(float));
            ggml_tensor * Vcur = ggml_view_2d(ctx, qkv, d, n_tokens, qkv->nb[1], 2*d*sizeof(float));
            out = ggml_add(ctx, ggml_mul(ctx, Qcur, Kcur), Vcur);
        } else {
            ggml_tensor * cache = input(GGML_TYPE_Q8_0, {512, 256, 1, 1}, "cache");
            ggml_tensor * q_in = input(GGML_TYPE_F32, {512, 16, n_tokens, 1}, "q_heads");
            ggml_tensor * mask = input(GGML_TYPE_F16, {256, 64, 1, 1}, "mask");
            ggml_tensor * view_k = ggml_view_4d(ctx, cache, 512, 1, 256, 1, cache->nb[1], cache->nb[1], cache->nb[2], 0);
            ggml_tensor * view_v = ggml_view_4d(ctx, cache, 512, 1, 256, 1, cache->nb[1], cache->nb[1], cache->nb[2], 0);
            ggml_tensor * q = ggml_permute(ctx, q_in, 0, 2, 1, 3);
            ggml_tensor * k = ggml_permute(ctx, view_k, 0, 2, 1, 3);
            ggml_tensor * v = ggml_permute(ctx, view_v, 0, 2, 1, 3);
            out = ggml_flash_attn_ext(ctx, q, k, v, mask, 0.125f, 0.0f, 0.0f);
        }
        ggml_backend_buffer_t weights_buf = ggml_backend_alloc_ctx_tensors_from_buft(weights, buft);
        GGML_ASSERT(weights_buf != nullptr);
        for (auto t : inputs) {
            std::vector<float> f32(ggml_nelements(t));
            for (size_t i = 0; i < f32.size(); ++i) { f32[i] = float(int(i % 23) - 11)/32; }
            if (t->type == GGML_TYPE_F32) {
                ggml_backend_tensor_set(t, f32.data(), 0, ggml_nbytes(t));
            } else if (t->type == GGML_TYPE_F16) {
                std::vector<ggml_fp16_t> f16(f32.size());
                ggml_fp32_to_fp16_row(f32.data(), f16.data(), f16.size());
                ggml_backend_tensor_set(t, f16.data(), 0, ggml_nbytes(t));
            } else {
                std::vector<uint8_t> quantized(ggml_nbytes(t));
                ggml_quantize_chunk(t->type, f32.data(), quantized.data(), 0, ggml_nrows(t), t->ne[0], nullptr);
                ggml_backend_tensor_set(t, quantized.data(), 0, quantized.size());
            }
        }
        ggml_set_output(out);
        ggml_cgraph * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, out);
        ggml_gallocr_t alloc = ggml_gallocr_new(buft);
        GGML_ASSERT(ggml_gallocr_alloc_graph(alloc, graph));
        GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
        results[mode].resize(ggml_nelements(out));
        ggml_backend_tensor_get(out, results[mode].data(), 0, ggml_nbytes(out));
        ggml_gallocr_free(alloc);
        ggml_backend_buffer_free(weights_buf);
        ggml_free(ctx);
        ggml_free(weights);
    }
    double error = 0, norm = 0;
    for (size_t i = 0; i < results[0].size(); ++i) {
        const double delta = results[1][i] - results[0][i];
        error += delta*delta;
        norm += double(results[0][i])*results[0][i];
    }
    const bool ok = std::isfinite(error) && error < 1e-6*norm;
    const char * names[] = {"fused projection views", "aliased q8_0 attention"};
    printf("%s tensor mode %s, %lld tokens, NMSE %.9g\n", ok ? "ok  " : "FAIL", names[variant], (long long) n_tokens, norm ? error/norm : error);
    ggml_backend_free(meta);
    ggml_backend_free(cpu);
    return ok;
}

static bool test_sequence_axis(ggml_backend_dev_t gpu) {
    const int captured_before = captured;
    std::array<int64_t, 4> rows = {1, 0, 2, 5};
    ggml_backend_dev_t devs[4] = {gpu, second_device(gpu), gpu, second_device(gpu)};
    ggml_backend_t backend = ggml_backend_dev_init(ggml_backend_meta_device(devs, 4, split_rows, &rows), nullptr);
    ggml_context * ctx = ggml_init({16*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true});
    ggml_tensor * x = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 32, 2, 2, 8);
    ggml_backend_meta_split_input(x, 3, rows.data(), rows.size());
    ggml_set_input(x);
    ggml_tensor * gathered = ggml_backend_meta_gather(ctx, ggml_scale(ctx, x, 2.0f));
    const int64_t next_rows[] = {0, 0, 0, 8};
    ggml_tensor * out = ggml_scale(ctx, ggml_backend_meta_scatter(ctx, gathered, 3, next_rows, 4), 2.0f);
    ggml_set_output(out);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    ggml_backend_t backends[] = {backend, cpu};
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE, false, true);
    ggml_backend_sched_set_tensor_backend(sched, x, backend);
    GGML_ASSERT(ggml_backend_sched_alloc_graph(sched, graph));
    std::vector<float> data(ggml_nelements(x)), result(data.size());
    bool ok = true;
    for (int iteration = 0; iteration < 6; ++iteration) {
        for (size_t i = 0; i < data.size(); ++i) { data[i] = float(i + iteration); }
        ggml_backend_tensor_set_async(backend, x, data.data(), 0, ggml_nbytes(x));
        GGML_ASSERT(ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS);
        ggml_backend_tensor_get_async(backend, out, result.data(), 0, ggml_nbytes(out));
        ggml_backend_synchronize(backend);
        for (size_t i = 0; i < data.size(); ++i) { ok = ok && result[i] == 4.0f*data[i]; }
    }
    ggml_backend_tensor_memset(x, 0, 0, ggml_nbytes(x));
    ggml_backend_tensor_get(x, result.data(), 0, ggml_nbytes(x));
    ok = ok && result == std::vector<float>(data.size(), 0.0f);
    if (ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(gpu), "ggml_backend_capture_end")) {
        ok = ok && captured - captured_before == 4;
    }
    printf("%s sequence-axis gather/scatter, %d captures and async transfers\n", ok ? "ok  " : "FAIL", captured - captured_before);
    ggml_backend_sched_free(sched);
    ggml_free(ctx);
    ggml_backend_free(backend);
    ggml_backend_free(cpu);
    return ok;
}

static ggml_backend_meta_split_state split_candidate_logits(const ggml_tensor *, void * userdata) {
    ggml_backend_meta_split_state state = {GGML_BACKEND_SPLIT_AXIS_0, {0}, {1}, 1};
    const auto & shards = *static_cast<const std::array<int64_t, 4> *>(userdata);
    std::copy(shards.begin(), shards.end(), state.ne);
    return state;
}

static bool test_candidates(ggml_backend_dev_t device, int vocab, int rows, int k, int split_axis,
                            const std::array<int64_t, 4> & shards, bool benchmark = false) {
    const int captured_before = captured;
    ggml_backend_dev_t devs[] = {device, second_device(device), device, second_device(device)};
    auto layout = shards;
    ggml_backend_t backend = ggml_backend_dev_init(split_axis < 0 ? device :
        ggml_backend_meta_device(devs, 4, split_candidate_logits, &layout), nullptr);
    ggml_context * weights = ggml_init({ggml_tensor_overhead(), nullptr, true});
    ggml_context * ctx = ggml_init({16*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true});
    ggml_tensor * logits = ggml_new_tensor_2d(split_axis == 0 ? weights : ctx, GGML_TYPE_F32, vocab, rows);
    ggml_set_input(logits);
    ggml_backend_buffer_t buffer = nullptr;
    if (split_axis == 0) {
        buffer = ggml_backend_alloc_ctx_tensors(weights, backend);
        GGML_ASSERT(buffer);
        ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    } else if (split_axis == 1) {
        ggml_backend_meta_split_input(logits, split_axis, shards.data(), shards.size());
    }
    const int parts = split_axis == 0 || split_axis < 0 ? 4 : 1;
    ggml_tensor * out = ggml_top_k_candidates(ctx, logits, k, parts);
    ggml_set_output(out);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    ggml_backend_t backends[] = {backend, cpu};
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE, false, true);
    ggml_backend_sched_set_tensor_backend(sched, logits, backend);
    ggml_backend_sched_set_tensor_backend(sched, out, backend);
    GGML_ASSERT(ggml_backend_sched_alloc_graph(sched, graph));
    GGML_ASSERT(ggml_backend_sched_get_tensor_backend(sched, out) == backend);

    std::vector<float> values(size_t(vocab)*rows);
    std::vector<uint64_t> records(size_t(k)*parts*rows);
    std::vector<int> expected(vocab);
    bool ok = true;
    for (int iteration = 0; iteration < 6; ++iteration) {
        std::mt19937 rng(1234 + iteration);
        for (size_t i = 0; i < values.size(); ++i) {
            // Cross-shard ties, all-equal rows, signed zeros, infinities and NaNs.
            const uint32_t mixed[] = {0x7fc00001u, 0xffc00001u, 0x80000000u, 0, 0x7f800000u, 0xff800000u, 0xbf800000u, 0x3f800000u};
            const uint32_t bits = iteration == 4 ? (i % 2 ? 0x80000000u : 0) : mixed[i % 8];
            if (iteration >= 4) { memcpy(&values[i], &bits, sizeof(bits)); }
            else { values[i] = iteration == 3 ? 1.0f : float(int(rng() % 401) - 200); }
        }
        if (!values.empty() && iteration == 2) {
            values[vocab/2] = INFINITY;
            values.back() = -INFINITY;
        }
        ggml_backend_tensor_set_async(backend, logits, values.data(), 0, ggml_nbytes(logits));
        GGML_ASSERT(ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS);
        for (int row = 0; row < rows; ++row) {
            const size_t offset = size_t(row)*k*parts;
            ggml_backend_tensor_get_async(backend, out, records.data() + offset, offset*sizeof(uint64_t),
                                          size_t(k)*parts*sizeof(uint64_t));
        }
        ggml_backend_synchronize(backend);
        for (int row = 0; row < rows; ++row) {
            const float * src = values.data() + size_t(row)*vocab;
            auto better = [&](int a, int b) {
                if (std::isnan(src[a]) != std::isnan(src[b])) { return !std::isnan(src[a]); }
                return src[a] > src[b] || (!(src[a] < src[b]) && a < b);
            };
            for (int i = 0; i < vocab; ++i) { expected[i] = i; }
            std::sort(expected.begin(), expected.end(), better);
            std::vector<int> actual;
            int first = 0;
            for (int part = 0; part < parts; ++part) {
                const int width = split_axis == 0 ? shards[part] : (part == 0 ? vocab : 0);
                std::vector<int> part_ids;
                for (int c = 0; c < k; ++c) {
                    const uint64_t record = records[(size_t(row)*parts + part)*k + c];
                    const int32_t id = int32_t(uint32_t(record));
                    if (id == -1) { continue; }
                    if (id < first || id >= first + width) { ok = false; continue; }
                    uint32_t bits;
                    memcpy(&bits, &src[id], sizeof(bits));
                    ok = ok && uint32_t(record >> 32) == bits;
                    actual.push_back(id);
                    part_ids.push_back(id);
                }
                std::sort(part_ids.begin(), part_ids.end(), better);
                ok = ok && int(part_ids.size()) == std::min(k, width);
                size_t pos = 0;
                for (int id : expected) {
                    if (id >= first && id < first + width && pos < part_ids.size()) {
                        ok = ok && part_ids[pos++] == id;
                    }
                }
                first += width;
            }
            std::sort(actual.begin(), actual.end(), better);
            ok = ok && std::adjacent_find(actual.begin(), actual.end()) == actual.end();
            const int winners = std::min(k, vocab);
            ok = ok && actual.size() >= size_t(winners);
            for (int i = 0; i < winners && i < int(actual.size()); ++i) { ok = ok && actual[i] == expected[i]; }
        }
    }
    const int captures = captured - captured_before;
    if (split_axis >= 0 && ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(device), "ggml_backend_capture_end")) {
        // Meta captures one graph per device, including empty row ranks.
        ok = ok && captures == 4;
    }
    printf("%s candidates vocab=%d rows=%d k=%d axis=%d captures=%d\n", ok ? "ok  " : "FAIL", vocab, rows, k, split_axis, captures);
    if (ok && benchmark) {
        std::mt19937 rng(530);
        for (auto & value : values) { value = float(int(rng() % 100001) - 50000)/1000.0f; }
        ggml_backend_tensor_set(logits, values.data(), 0, ggml_nbytes(logits));
        std::vector<int64_t> times;
        for (int iteration = 0; iteration < 103; ++iteration) {
            const int64_t start = ggml_time_us();
            GGML_ASSERT(ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS);
            ggml_backend_synchronize(backend);
            if (iteration >= 3) { times.push_back(ggml_time_us() - start); }
        }
        std::sort(times.begin(), times.end());
        printf("candidates median: vocab=%d rows=%d k=%d axis=%d %lld us, %zu bytes readback\n",
               vocab, rows, k, split_axis, (long long) times[times.size()/2], ggml_nbytes(out));
    }
    ggml_backend_sched_free(sched);
    ggml_backend_buffer_free(buffer);
    ggml_free(weights);
    ggml_free(ctx);
    ggml_backend_free(backend);
    ggml_backend_free(cpu);
    return ok;
}

static bool test_candidates_all(ggml_backend_dev_t device) {
    bool ok = true;
    for (int k : {1, 10, 40, 296}) {
        ok = test_candidates(device, 17, 3, k, -1, {}) && ok;
        ok = test_candidates(device, 17, 3, k, 0, {1, 0, 2, 14}) && ok;
        ok = test_candidates(device, 17, 3, k, 0, {0, 17, 0, 0}) && ok;
        ok = test_candidates(device, 8193, 3, k, 0, {0, 8190, 2, 1}) && ok;
        ok = test_candidates(device, 257, 8, k, 1, {1, 0, 2, 5}) && ok;
    }
    ok = test_candidates(device, 8193, 3, 296, -1, {}) && ok;
    ok = test_candidates(device, 17001, 3, 5000, -1, {}) && ok;
    for (int rows : {1, 23, 69}) {
        for (int k : {40, 296}) {
            ok = test_candidates(device, 154880, rows, k, -1, {}) && ok;
            ok = test_candidates(device, 154880, rows, k, 0, {30000, 50000, 38720, 36160}) && ok;
        }
    }
    return ok;
}

int main(int argc, char ** argv) {
    // Isolate partition correctness from the optional BF16 collective round trip.
#ifdef _WIN32
    _putenv_s("GGML_CUDA_AR_BF16_THRESHOLD", "0");
    _putenv_s("GGML_CUDA_DP_EXPERT_STATS", "1");
#else
    setenv("GGML_CUDA_AR_BF16_THRESHOLD", "0", 1);
    setenv("GGML_CUDA_DP_EXPERT_STATS", "1", 1);
#endif
    ggml_log_set(log_cb, nullptr);
    ggml_backend_dev_t gpu = nullptr;
    if (argc == 2 && (strcmp(argv[1], "--split-cache-cpu") == 0 || strcmp(argv[1], "--split-cycle-cpu") == 0)) {
        return split_test_stack_limit() && test_split_cache(ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU), strcmp(argv[1], "--split-cycle-cpu") == 0) ? 0 : 1;
    }
    if (argc == 2 && strcmp(argv[1], "--permuted-read-cpu") == 0) {
        return test_permuted_reads(ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU)) ? 0 : 1;
    }
    if (argc == 2 && strcmp(argv[1], "--candidates-cpu") == 0) {
        return test_candidates_all(ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU)) ? 0 : 1;
    }
    if (argc == 2 && strcmp(argv[1], "--host-layout-cpu") == 0) {
        return test_host_layout(ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU)) ? 0 : 1;
    }
    if (argc == 3 && strcmp(argv[1], "--single-owner-cpu") == 0) {
        return test_owned_ops(ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU), atoi(argv[2]), true) ? 0 : 1;
    }
    if (argc == 2 && strcmp(argv[1], "--stream-view-cpu") == 0) {
        gpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        return test_stream_view(gpu, GGML_TYPE_F32) && test_stream_view(gpu, GGML_TYPE_Q8_0) &&
               test_stream_view(gpu, GGML_TYPE_F32, true) && test_stream_view(gpu, GGML_TYPE_Q8_0, true) ? 0 : 1;
    }
    if (argc == 3 && strcmp(argv[1], "--owned-op-cpu") == 0) {
        return test_owned_ops(ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU), atoi(argv[2])) ? 0 : 1;
    }
    if (argc == 4 && strcmp(argv[1], "--tensor-cpu") == 0) {
        return test_tensor_mode(ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU), atoi(argv[2]), atoll(argv[3])) ? 0 : 1;
    }
    if (argc > 1 && strcmp(argv[1], "--cpu") == 0) {
        gpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        bool ok = true;
        for (ggml_type type : {GGML_TYPE_F32, GGML_TYPE_Q8_0, GGML_TYPE_IQ3_XXS}) { ok = test_experts(gpu, type) && ok; }
        for (int variant = 0; variant < 12; ++variant) {
            ok = test_owned_ops(gpu, variant) && ok;
            ok = test_owned_ops(gpu, variant, true) && ok;
        }
        for (int64_t n_tokens : {1, 3}) { ok = test_tensor_mode(gpu, 0, n_tokens) && ok; }
        ok = test_tensor_mode(gpu, 1, 8) && ok;
        ok = test_scatter_views(gpu) && ok;
        ok = test_host_layout(gpu) && ok;
        for (ggml_type type : {GGML_TYPE_F32, GGML_TYPE_Q8_0}) {
            ok = test_owned_cache_rows(gpu, type, true) && test_stream_view(gpu, type, false, true) && test_stream_view(gpu, type, true, true) && ok;
        }
        for (int64_t q : {1, 3}) { ok = test_host_attention(gpu, q, false) && test_host_attention(gpu, q, true) && ok; }
        for (ggml_type type : {GGML_TYPE_F32, GGML_TYPE_Q8_0}) { ok = test_owned_cache_rows(gpu, type) && test_stream_view(gpu, type) && test_stream_view(gpu, type, true) && ok; }
        return ok ? 0 : 1;
    }
    for (size_t i = 0; i < ggml_backend_dev_count() && gpu == nullptr; i++) {
        if (ggml_backend_dev_type(ggml_backend_dev_get(i)) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            gpu = ggml_backend_dev_get(i);
        }
    }
    if (gpu == nullptr) {
        printf("no GPU for the meta backend, skipped\n");
        return argc > 1 && strncmp(argv[1], "--host-kv", 9) == 0 ? 77 : 0;
    }
    if (argc > 1 && (strcmp(argv[1], "--host-kv") == 0 || strcmp(argv[1], "--host-kv-four") == 0)) {
        if (strcmp(argv[1], "--host-kv-four") == 0) {
            size_t count = 0;
            for (size_t i = 0; i < ggml_backend_dev_count() && count < 4; ++i) {
                auto dev = ggml_backend_dev_get(i);
                if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) { host_devices[count++] = dev; }
            }
            if (count != 4) { printf("need four GPUs for mapped host KV qualification\n"); return 77; }
            for (auto dev : host_devices) { printf("host KV device: %s\n", ggml_backend_dev_name(dev)); }
        } else {
            printf("host KV: four logical ranks on one GPU; not four-card qualification\n");
        }
        if (!ggml_backend_dev_mapped_buffer_type(gpu)) { printf("mapped host KV unsupported on this backend\n"); return 77; }
        bool ok = true;
        for (int64_t q : {1, 3, 16}) { ok = test_host_attention(gpu, q, false) && test_host_attention(gpu, q, true, true) && ok; }
        for (int64_t q : {3, 512}) { ok = test_host_attention(gpu, q, false, false, true) && ok; }
        for (ggml_type type : {GGML_TYPE_F32, GGML_TYPE_Q8_0}) {
            ok = test_owned_cache_rows(gpu, type, true) && test_stream_view(gpu, type, false, true) && test_stream_view(gpu, type, true, true) && ok;
        }
        return ok ? 0 : 1;
    }
    if (argc == 2 && strcmp(argv[1], "--permuted-read") == 0) {
        return test_permuted_reads(gpu) ? 0 : 1;
    }
    if (argc == 2 && strcmp(argv[1], "--candidates") == 0) { return test_candidates_all(gpu) ? 0 : 1; }
    if (argc == 2 && strcmp(argv[1], "--candidates-perf") == 0) {
        bool ok = true;
        for (int rows : {1, 23, 69}) {
            for (int k : {40, 296}) {
                ok = test_candidates(gpu, 38720, rows, k, -1, {}, true) && ok;
                ok = test_candidates(gpu, 154880, rows, k, -1, {}, true) && ok;
                ok = test_candidates(gpu, 154880, rows, k, 0, {38720, 38720, 38720, 38720}, true) && ok;
            }
        }
        return ok ? 0 : 1;
    }
    if (argc == 4 && strcmp(argv[1], "--tensor") == 0) {
        return test_tensor_mode(gpu, atoi(argv[2]), atoll(argv[3])) ? 0 : 1;
    }
    if (argc > 1 && strcmp(argv[1], "--two-gpu") == 0) {
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            auto dev = ggml_backend_dev_get(i);
            if (dev != gpu && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) { peer = dev; break; }
        }
        if (!peer) { printf("need two physical GPUs, skipped\n"); return 77; }
        printf("four logical ranks on two physical GPUs; not four-card qualification\n");
    }
    ggml_backend_dev_t devs[2] = {gpu, gpu};
    ggml_backend_dev_t meta = ggml_backend_meta_device(devs, 2, split_by_name, nullptr);

    const char * names[] = {"heads_10", "heads_01", "heads_11"};
    const ggml_init_params params = {
        /*.mem_size   =*/ 3*ggml_tensor_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    std::vector<ggml_tensor *> tensors;
    for (const char * name : names) {
        ggml_tensor * t = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_cols, n_rows, (name[6] - '0') + (name[7] - '0'));
        ggml_set_name(t, name);
        tensors.push_back(t);
    }
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_dev_buffer_type(meta));
    GGML_ASSERT(buf != nullptr);

    bool ok = true;
    for (ggml_tensor * t : tensors) {
        std::vector<float> data(ggml_nelements(t));
        for (size_t i = 0; i < data.size(); i++) {
            data[i] = 1.0f + i;
        }
        ggml_backend_tensor_set(t, data.data(), 0, ggml_nbytes(t));
        std::vector<float> back(data.size());
        ggml_backend_tensor_get(t, back.data(), 0, ggml_nbytes(t));
        bool ok_t = back == data;

        ggml_backend_tensor_memset(t, 0, 0, ggml_nbytes(t));
        ggml_backend_tensor_get(t, back.data(), 0, ggml_nbytes(t));
        ok_t = ok_t && back == std::vector<float>(data.size(), 0.0f);

        printf("%s %s\n", ok_t ? "ok  " : "FAIL", t->name);
        ok = ok && ok_t;
    }

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);

    ok = test_rows(gpu, false) && ok;
    ok = test_rows(gpu, true) && ok;
    ok = test_sequence_axis(gpu) && ok;
    ok = test_candidates_all(gpu) && ok;
    ok = test_scatter_views(gpu) && ok;
    ok = test_host_layout(gpu) && ok;
    if (ggml_backend_dev_mapped_buffer_type(gpu)) {
        for (ggml_type type : {GGML_TYPE_F32, GGML_TYPE_Q8_0}) {
            ok = test_owned_cache_rows(gpu, type, true) && test_stream_view(gpu, type, false, true) && test_stream_view(gpu, type, true, true) && ok;
        }
        for (int64_t q : {1, 3}) { ok = test_host_attention(gpu, q, false) && test_host_attention(gpu, q, true) && ok; }
    }
    for (ggml_type type : {GGML_TYPE_F32, GGML_TYPE_Q8_0}) { ok = test_owned_cache_rows(gpu, type) && test_stream_view(gpu, type) && test_stream_view(gpu, type, true) && ok; }
    for (ggml_type type : {GGML_TYPE_F32, GGML_TYPE_Q8_0, GGML_TYPE_IQ3_XXS}) { ok = test_experts(gpu, type) && ok; }
    for (int variant = 0; variant < 12; ++variant) {
        ok = test_owned_ops(gpu, variant) && ok;
        ok = test_owned_ops(gpu, variant, true) && ok;
    }
    for (int64_t n_tokens : {1, 3}) { ok = test_tensor_mode(gpu, 0, n_tokens) && ok; }
    ok = test_tensor_mode(gpu, 1, 8) && ok;

    printf("%s\n", ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}
