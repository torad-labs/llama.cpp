// The meta backend (-sm tensor) reduces a prompt's activations across the two devices with the internal AllReduce. A
// reduction of 1 MB or more on the wire goes over the copy engines (ggml_cuda_ar_allreduce_copy_impl): each device
// copies its share to pinned host memory, copies its peer's share back from there, and adds it. Each device must end
// with the sum of both shares, whichever device reaches the reduction first, and at every reduction of a sequence that
// reuses the same staging, scratch and event slots.
//
// A stack of feed-forward layers, x = x + down(up(x)), is split over two devices with the up rows and the down columns
// cut unevenly, the first device light (32 rows) in even layers and the second in odd layers, so the device that
// reaches each reduction first changes from one reduction to the next. The late device's share is still being copied
// out when its peer's has long arrived. Three graphs take x of 256, 640 and 4,100 columns: 1 MB, 2.5 MB and 16 MB on the
// BF16 wire, and 2 MB, 5 MB and 33.6 MB on the F32 wire, which is past one pass of the 32 MB staging. They are evaluated
// A, B, C, A, B, C with a new x each time, and every output is compared with the CPU's. Four passes: the BF16 wire and
// the F32 wire (GGML_CUDA_AR_BF16_THRESHOLD=0, where each device's share is reduced in place, so a share overwritten
// before it was copied out reaches the peer already summed), each with the receives beside the sends and then after
// them (GGML_CUDA_AR_COPY_ONE_STREAM=1, the order before). The same kernels add the same bytes in both orders, so each
// output after the sends must equal the one beside them bit for bit: a race that reaches a single chunk shows there,
// though its error can stay under the CPU comparison's bar. Those passes stage through host memory
// (GGML_CUDA_AR_COPY_PEER=0). A third pass on each wire takes the default: where the two cards can map each other's
// memory (the pipeline's LL card-to-card limit is above 0), each copies its share once, straight into a receive buffer
// on its peer, and its pipeline must say its receives are card to card; where they cannot, it stages as before. Its
// outputs must equal the pass beside the sends bit for bit, since the same kernel adds the same bytes. Each pass must
// initialize the internal AllReduce pipeline in its own order. On the first two GPUs, or on the first GPU twice; with
// no GPU the test is skipped.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

static constexpr int64_t n_embd  = 2048;
static constexpr int64_t n_ff    = 4096;
static constexpr int64_t n_lo    = 32; // up rows and down columns on the light device
static constexpr int     n_layer = 4;

static constexpr int     n_graphs        = 3;
static constexpr int64_t n_tok[n_graphs] = {256, 640, 4100};
static constexpr int     order[]         = {0, 1, 2, 0, 1, 2};
static constexpr int     n_evals         = sizeof(order)/sizeof(order[0]);

static ggml_backend_meta_split_state split_by_name(const ggml_tensor * tensor, void * /*userdata*/) {
    const bool up = strncmp(tensor->name, "up.", 3) == 0;
    if (up || strncmp(tensor->name, "down.", 5) == 0) {
        const int il = atoi(tensor->name + (up ? 3 : 5));
        // up: [n_embd, n_ff], its rows; down: [n_ff, n_embd], its columns
        ggml_backend_meta_split_state split_state = {
            up ? GGML_BACKEND_SPLIT_AXIS_1 : GGML_BACKEND_SPLIT_AXIS_0, {0}, {1}, 1};
        split_state.ne[0] = il % 2 == 0 ? n_lo : n_ff - n_lo;
        split_state.ne[1] = n_ff - split_state.ne[0];
        return split_state;
    }
    return {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
}

static int  n_pipelines       = 0;     // "initialized AllReduce pipeline" lines logged
static bool last_one_stream   = false; // the last one's copy-engine receives run after its sends
static bool last_card_to_card = false; // the last one's copy-engine receives come card to card
static bool last_p2p          = false; // the last one's cards map each other's memory: its LL card-to-card limit is above 0
static bool last_gather_peer  = false;
static bool last_owner_slice  = false;
static std::atomic<int> n_captures{0};

static void log_cb(ggml_log_level level, const char * text, void * /*user_data*/) {
    if (strstr(text, "owner-slice-peer") != nullptr) {
        last_owner_slice = true;
    }
    if (strstr(text, "initialized AllReduce pipeline") != nullptr) {
        n_pipelines++;
        last_one_stream   = strstr(text, "receives after the sends") != nullptr;
        last_card_to_card = strstr(text, "receives card to card") != nullptr;
        last_gather_peer = strstr(text, "large gather card to card") != nullptr;
        last_owner_slice = strstr(text, "owner-slice-peer") != nullptr;
        const char * ll   = strstr(text, "LL card to card up to ");
        unsigned long long ll_bytes = 0;
        last_p2p = ll != nullptr && sscanf(ll, "LL card to card up to %llu", &ll_bytes) == 1 && ll_bytes > 0;
        fputs(text, stderr);
    }
    if (strstr(text, "ggml_backend_cuda_capture_end") && strstr(text, " nodes, ")) { ++n_captures; }
    if (level == GGML_LOG_LEVEL_WARN || level == GGML_LOG_LEVEL_ERROR) {
        fputs(text, stderr);
    }
}

// the layers over the weights in ctx_w, on an input of n_tok columns
struct ffn_graph {
    ggml_context * ctx = nullptr;
    ggml_cgraph  * gf  = nullptr;
    ggml_tensor  * x   = nullptr;
    ggml_tensor  * out = nullptr;

    ffn_graph(const std::vector<ggml_tensor *> & up, const std::vector<ggml_tensor *> & down, int64_t n_tok) {
        const ggml_init_params params = {
            /*.mem_size   =*/ (4*n_layer + 2)*ggml_tensor_overhead() + ggml_graph_overhead(),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ctx = ggml_init(params);
        x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_tok);
        ggml_set_input(x);
        ggml_tensor * cur = x;
        for (int il = 0; il < n_layer; il++) {
            ggml_tensor * h = ggml_mul_mat(ctx, up[il], cur);
            ggml_tensor * y = ggml_mul_mat(ctx, down[il], h);
            cur = ggml_add(ctx, y, cur);
        }
        out = cur;
        ggml_set_output(out);
        gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out);
    }

    ~ffn_graph() {
        ggml_free(ctx);
    }
};

// the weights, in a buffer of buft
struct ffn_weights {
    ggml_context *             ctx = nullptr;
    ggml_backend_buffer_t      buf = nullptr;
    std::vector<ggml_tensor *> up;
    std::vector<ggml_tensor *> down;

    ffn_weights(ggml_backend_buffer_type_t buft, const std::vector<std::vector<float>> & data) {
        const ggml_init_params params = {
            /*.mem_size   =*/ 2*n_layer*ggml_tensor_overhead(),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ctx = ggml_init(params);
        for (int il = 0; il < n_layer; il++) {
            up.push_back(ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_ff));
            ggml_format_name(up.back(), "up.%d", il);
            down.push_back(ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_ff, n_embd));
            ggml_format_name(down.back(), "down.%d", il);
        }
        buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
        GGML_ASSERT(buf != nullptr);
        ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        for (int il = 0; il < n_layer; il++) {
            ggml_backend_tensor_set(up[il],   data[2*il + 0].data(), 0, ggml_nbytes(up[il]));
            ggml_backend_tensor_set(down[il], data[2*il + 1].data(), 0, ggml_nbytes(down[il]));
        }
    }

    ~ffn_weights() {
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }
};

// the graphs over one set of weights, each allocated once by a scheduler of its own, as llama.cpp's graph slots are
struct ffn_stack {
    ffn_weights                             w;
    std::vector<std::unique_ptr<ffn_graph>> graphs;
    std::vector<ggml_backend_sched_t>       scheds;

    ffn_stack(ggml_backend_t * backends, int n_backends, ggml_backend_buffer_type_t buft,
              const std::vector<std::vector<float>> & w_data) : w(buft, w_data) {
        for (int k = 0; k < n_graphs; k++) {
            graphs.push_back(std::make_unique<ffn_graph>(w.up, w.down, n_tok[k]));
            scheds.push_back(ggml_backend_sched_new(backends, nullptr, n_backends, GGML_DEFAULT_GRAPH_SIZE, false, true));
            GGML_ASSERT(ggml_backend_sched_alloc_graph(scheds.back(), graphs.back()->gf));
        }
    }

    std::vector<float> compute(int k, const std::vector<float> & x) {
        ffn_graph & g = *graphs[k];
        ggml_backend_tensor_set(g.x, x.data(), 0, ggml_nbytes(g.x));
        GGML_ASSERT(ggml_backend_sched_graph_compute(scheds[k], g.gf) == GGML_STATUS_SUCCESS);
        std::vector<float> out(ggml_nelements(g.out));
        ggml_backend_tensor_get(g.out, out.data(), 0, ggml_nbytes(g.out));
        return out;
    }

    ~ffn_stack() {
        for (auto * sched : scheds) {
            ggml_backend_sched_free(sched);
        }
    }
};

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

static double nmse(const std::vector<float> & ref, const std::vector<float> & v) {
    double err = 0.0;
    double sum = 0.0;
    for (size_t i = 0; i < ref.size(); i++) {
        err += (v[i] - ref[i])*(v[i] - ref[i]);
        sum += ref[i]*ref[i];
    }
    return err/sum;
}

static ggml_backend_meta_split_state mirrored(const ggml_tensor *, void *) {
    return {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
}

// Several gathers in one graph, followed by replay with new data, exercise the shared parity ring across captures.
static bool test_gather_graph(std::vector<ggml_backend_dev_t> devices, int64_t rows) {
    auto * meta = ggml_backend_meta_device(devices.data(), devices.size(), mirrored, nullptr);
    auto * backend = ggml_backend_dev_init(meta, nullptr);
    auto * ctx = ggml_init({64*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true});
    auto * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4096, rows);
    std::vector<int64_t> owned(devices.size(), 0);
    owned.front() = 1;
    owned.back() = rows - 1;
    ggml_backend_meta_split_input(input, 1, owned.data(), owned.size());
    ggml_set_input(input);
    ggml_tensor * out = input;
    for (int step = 0; step < 4; ++step) {
        out = ggml_backend_meta_gather(ctx, ggml_scale(ctx, out, 2.0f));
        if (step != 3) {
            std::rotate(owned.begin(), owned.begin() + 1, owned.end());
            out = ggml_backend_meta_scatter(ctx, out, 1, owned.data(), owned.size());
        }
    }
    ggml_set_output(out);
    auto * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);
    auto * cpu = ggml_backend_cpu_init();
    ggml_backend_t backends[] = {backend, cpu};
    auto * sched = ggml_backend_sched_new(backends, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE, false, true);
    ggml_backend_sched_set_tensor_backend(sched, input, backend);
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        ggml_backend_sched_set_tensor_backend(sched, ggml_graph_node(graph, i), backend);
    }
    GGML_ASSERT(ggml_backend_sched_alloc_graph(sched, graph));
    std::vector<float> values(ggml_nelements(input)), got(values.size());
    bool ok = true;
    const int captures_before = n_captures;
    for (int iteration = 0; iteration < 6; ++iteration) {
        for (size_t j = 0; j < values.size(); ++j) { values[j] = float(int((j + iteration*7) % 1024) - 512); }
        ggml_backend_tensor_set(input, values.data(), 0, ggml_nbytes(input));
        const bool ran = ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS;
        ok = ok && ran;
        if (!ran) { break; }
        ggml_backend_tensor_get(out, got.data(), 0, ggml_nbytes(out));
        for (size_t j = 0; j < got.size(); ++j) { ok = ok && got[j] == 16.0f*values[j]; }
    }
    const bool captured = n_captures > captures_before;
    printf("%s: %zu-rank gather graph, %lld rows, rotating empty ranks, values %s, %d captures\n",
        ok && captured ? "ok" : "FAIL", devices.size(), (long long) rows, ok ? "equal" : "differ", n_captures.load() - captures_before);
    ggml_backend_sched_free(sched);
    ggml_free(ctx);
    ggml_backend_free(backend);
    ggml_backend_free(cpu);
    return ok && captured;
}

// Direct collectives check every rank, including tails and inactive ranks. The graph tests below check delayed producers.
static int test_direct(const std::vector<ggml_backend_dev_t> & devices, bool require_p2p) {
    const int n = (int) devices.size();
    auto reg = ggml_backend_dev_backend_reg(devices[0]);
    auto init = (ggml_backend_comm_init_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_comm_init");
    auto free = (ggml_backend_comm_free_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_comm_free");
    auto sum  = (ggml_backend_comm_allreduce_tensor_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_comm_allreduce_tensor");
    auto gather = (ggml_backend_comm_allgather_tensor_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_comm_allgather_tensor");
    GGML_ASSERT(init && free && sum && gather);
    std::vector<ggml_backend_t> backends(n);
    for (int rank = 0; rank < n; ++rank) {
        backends[rank] = ggml_backend_dev_init(devices[rank], nullptr);
        GGML_ASSERT(backends[rank]);
    }
    struct route {
        const char * name;
        const char * ll;
        const char * p2p;
        const char * copy;
        bool one_stream;
    };
    const route routes[] = {
        {"host LL",      "1048576", "0",       "0", false},
        {"host chunked", "0",       "0",       "0", false},
        {"peer LL",      "0",       "1048576", "0", false},
        {"host copy",    "0",       "0",       "1", false},
        {"host serial",  "0",       "0",       "1", true},
        {"peer copy",    "0",       "1048576", "1", false},
        {"owner-slice",  "0",       "1048576", "0", false},
    };
    bool ok = true;
    for (bool bf16 : {false, true}) {
        for (const auto & route : routes) {
            set_env("GGML_CUDA_AR_BF16_THRESHOLD", bf16 ? "1" : "0");
            set_env("GGML_CUDA_AR_LL_MAX_BYTES", route.ll);
            set_env("GGML_CUDA_AR_P2P_MAX_BYTES", route.p2p);
            set_env("GGML_CUDA_AR_COPY_THRESHOLD", route.copy);
            set_env("GGML_CUDA_AR_COPY_ONE_STREAM", route.one_stream ? "1" : "0");
            set_env("GGML_CUDA_AR_COPY_PEER", "1");
            set_env("GGML_CUDA_AR_GATHER_OWNER_SLICES", strcmp(route.name, "owner-slice") == 0 ? "1" : "0");
            const int before = n_pipelines;
            void * comm = init(backends.data(), backends.size());
            if (!comm || n_pipelines != before + 1) {
                fprintf(stderr, "FAIL: %d-rank internal pipeline did not initialize\n", n);
                if (comm) { free(comm); }
                for (auto backend : backends) { ggml_backend_free(backend); }
                return 1;
            }
            const bool peer_requested = route.p2p[0] != '0';
            const bool peer_active = last_p2p;
            const bool owner_requested = strcmp(route.name, "owner-slice") == 0;
            if (last_one_stream != route.one_stream || (!peer_requested && (last_p2p || last_card_to_card)) ||
                    (peer_requested && require_p2p && (!last_p2p || !last_gather_peer ||
                        (route.copy[0] == '1' && !last_card_to_card) || (owner_requested && !last_owner_slice)))) {
                fprintf(stderr, "FAIL: %d-rank %s did not select the required transport\n", n, route.name);
                free(comm);
                for (auto backend : backends) { ggml_backend_free(backend); }
                return 1;
            }
            void * pair_comm = nullptr;
            if (n > 2) {
                const int before_pair = n_pipelines;
                pair_comm = init(backends.data(), 2);
                if (!pair_comm || n_pipelines != before_pair + 1) {
                    fprintf(stderr, "FAIL: two-rank reference pipeline did not initialize\n");
                    if (pair_comm) { free(pair_comm); }
                    free(comm);
                    for (auto backend : backends) { ggml_backend_free(backend); }
                    return 1;
                }
            }
            std::vector<int64_t> sizes = {1, 3, 17, 4097, 131079, 512*4096 + 3};
            if (route.copy[0] == '1') {
                sizes.push_back(32*1024*1024/(bf16 ? 2 : 4) + 3);
            }
            for (const int64_t ne : sizes) {
                bool case_ok = true;
                std::vector<ggml_context *> contexts(n);
                std::vector<ggml_tensor *> tensors(n);
                std::vector<ggml_backend_buffer_t> buffers(n);
                for (int rank = 0; rank < n; ++rank) {
                    contexts[rank] = ggml_init({ggml_tensor_overhead(), nullptr, true});
                    tensors[rank] = ggml_new_tensor_1d(contexts[rank], GGML_TYPE_F32, ne);
                    buffers[rank] = ggml_backend_alloc_ctx_tensors(contexts[rank], backends[rank]);
                    GGML_ASSERT(buffers[rank]);
                }
                std::vector<float> input(ne), expected(ne), output(ne), first(ne);
                const auto value = [](int rank, int64_t j, int iteration) {
                    if (iteration == 6 && rank >= 2) { return 0.0f; }
                    if (j % 17 == 0) { return rank == 0 ? 1e20f : rank == 1 ? -1e20f : (float) rank; }
                    return (float) ((j*13 + rank*7 + iteration) % 41 - 20) / 7.0f;
                };
                for (int iteration = 0; iteration < (pair_comm ? 7 : 6); ++iteration) {
                    std::fill(expected.begin(), expected.end(), 0.0f);
                    for (int rank = 0; rank < n; ++rank) {
                        const bool active = iteration < 2 || iteration == 6 || (iteration < 5 && rank != iteration % n);
                        tensors[rank]->flags = active ? GGML_TENSOR_FLAG_COMPUTE : 0;
                        for (int64_t j = 0; j < ne; ++j) {
                            float v = value(rank, j, iteration);
                            input[j] = active ? v : 12345.0f;
                            v = active ? v : 0.0f;
                            if (bf16) { v = ggml_bf16_to_fp32(ggml_fp32_to_bf16(v)); }
                            // Keep the reference's logical rank order, without regrouping partial sums.
                            volatile float reduced = rank == 0 ? v : expected[j] + v;
                            expected[j] = reduced;
                        }
                        ggml_backend_tensor_set(tensors[rank], input.data(), 0, input.size()*sizeof(float));
                    }
                    const bool ran = sum(comm, tensors.data());
                    case_ok = case_ok && ran;
                    for (auto backend : backends) { ggml_backend_synchronize(backend); }
                    for (int rank = 0; rank < n && ran; ++rank) {
                        ggml_backend_tensor_get(tensors[rank], output.data(), 0, output.size()*sizeof(float));
                        const bool same = memcmp(output.data(), expected.data(), output.size()*sizeof(float)) == 0;
                        if (!same) { fprintf(stderr, "FAIL: %s %s ne=%lld iteration=%d rank=%d vs ordered sum\n",
                                             route.name, bf16 ? "BF16" : "F32", (long long) ne, iteration, rank); }
                        case_ok = case_ok && same;
                        if (rank == 0) { first = output; }
                        else { case_ok = case_ok && memcmp(first.data(), output.data(), output.size()*sizeof(float)) == 0; }
                    }
                    if (iteration == 6) {
                        // The same two contributions plus zero-valued ranks must retain the two-device result.
                        for (int rank = 0; rank < 2; ++rank) {
                            for (int64_t j = 0; j < ne; ++j) { input[j] = value(rank, j, iteration); }
                            ggml_backend_tensor_set(tensors[rank], input.data(), 0, input.size()*sizeof(float));
                        }
                        const bool pair_ran = sum(pair_comm, tensors.data());
                        case_ok = case_ok && pair_ran;
                        for (int rank = 0; rank < 2; ++rank) { ggml_backend_synchronize(backends[rank]); }
                        for (int rank = 0; rank < 2 && pair_ran; ++rank) {
                            ggml_backend_tensor_get(tensors[rank], output.data(), 0, output.size()*sizeof(float));
                            const bool same = memcmp(first.data(), output.data(), output.size()*sizeof(float)) == 0;
                            if (!same) { fprintf(stderr, "FAIL: %d-rank result differs from two-rank reference, %s, ne=%lld\n",
                                                 n, route.name, (long long) ne); }
                            case_ok = case_ok && same;
                        }
                    }
                }
                if (ne <= 512*4096 + 3) {
                    std::vector<uint32_t> words(ne), want(ne), got(ne);
                    for (int iteration = 0; iteration < 6; ++iteration) {
                        for (int64_t j = 0; j < ne; ++j) {
                            want[j] = iteration >= 4 && j % n == iteration % n ? 0 : 0x7fc00000u | (uint32_t) (j + iteration);
                        }
                        for (int rank = 0; rank < n; ++rank) {
                            const bool active = iteration < 4 || rank != iteration % n;
                            tensors[rank]->flags = active ? GGML_TENSOR_FLAG_COMPUTE : 0;
                            for (int64_t j = 0; j < ne; ++j) { words[j] = !active ? 0xffffffffu : j % n == rank ? want[j] : 0; }
                            ggml_backend_tensor_set(tensors[rank], words.data(), 0, words.size()*sizeof(uint32_t));
                        }
                        const bool ran = gather(comm, tensors.data());
                        case_ok = case_ok && ran;
                        for (auto backend : backends) { ggml_backend_synchronize(backend); }
                        for (int rank = 0; rank < n && ran; ++rank) {
                            ggml_backend_tensor_get(tensors[rank], got.data(), 0, got.size()*sizeof(uint32_t));
                            case_ok = case_ok && got == want;
                        }
                    }
                }
                for (int rank = 0; rank < n; ++rank) {
                    ggml_backend_buffer_free(buffers[rank]);
                    ggml_free(contexts[rank]);
                }
                ok = ok && case_ok;
                printf("%s: %d ranks, %s%s, %s wire, %lld elements%s\n", case_ok ? "ok" : "FAIL", n, route.name,
                       peer_requested && !peer_active ? " (host fallback)" : "", bf16 ? "BF16" : "F32", (long long) ne,
                       pair_comm ? ", including two-rank parity" : "");
            }
            if (pair_comm) { free(pair_comm); }
            free(comm);
            if (!bf16 && (strcmp(route.name, "peer copy") == 0 || strcmp(route.name, "host chunked") == 0 || strcmp(route.name, "owner-slice") == 0)) {
                set_env("GGML_CUDA_GRAPH_MAX", "0");
                ok = test_gather_graph(devices, 512) && ok;
                ok = test_gather_graph(devices, 513) && ok;
            }
        }
    }
    for (auto backend : backends) { ggml_backend_free(backend); }
    return ok ? 0 : 1;
}

// Persistent launch threads keep captured collectives concurrent, including when CUDA launch queues fill.
struct benchmark_launcher {
    std::mutex mutex;
    std::condition_variable work, done;
    std::vector<std::thread> threads;
    const std::vector<ggml_backend_t> & backends;
    ggml_backend_capture_launch_t launch;
    std::vector<void *> execs;
    size_t generation = 0, pending = 0;
    int repeats = 0;
    bool stop = false;

    benchmark_launcher(const std::vector<ggml_backend_t> & backends, ggml_backend_capture_launch_t launch) :
            backends(backends), launch(launch), execs(backends.size()) {
        for (size_t rank = 0; rank < backends.size(); ++rank) {
            threads.emplace_back([this, rank] {
                size_t seen = 0;
                std::unique_lock<std::mutex> lock(mutex);
                for (;;) {
                    work.wait(lock, [&] { return stop || generation != seen; });
                    if (stop) { return; }
                    seen = generation;
                    lock.unlock();
                    for (int i = 0; i < repeats; ++i) { this->launch(this->backends[rank], execs[rank]); }
                    ggml_backend_synchronize(this->backends[rank]);
                    lock.lock();
                    if (--pending == 0) { done.notify_one(); }
                }
            });
        }
    }
    void run(const std::vector<void *> & graphs, int count) {
        std::unique_lock<std::mutex> lock(mutex);
        execs = graphs;
        repeats = count;
        pending = threads.size();
        ++generation;
        work.notify_all();
        done.wait(lock, [&] { return pending == 0; });
    }
    ~benchmark_launcher() {
        { std::lock_guard<std::mutex> lock(mutex); stop = true; work.notify_all(); }
        for (auto & thread : threads) { thread.join(); }
    }
};

// End-to-end replay batch latency includes launch/synchronization. Nsight supplies GPU-only timings separately.
static int benchmark_collectives(const std::vector<ggml_backend_dev_t> & devices, int delay_steps) {
    const auto reg = ggml_backend_dev_backend_reg(devices.front());
    const auto proc = [&](const char * name) { return ggml_backend_reg_get_proc_address(reg, name); };
    const auto init = (ggml_backend_comm_init_t) proc("ggml_backend_comm_init");
    const auto release = (ggml_backend_comm_free_t) proc("ggml_backend_comm_free");
    const auto sum = (ggml_backend_comm_allreduce_tensor_t) proc("ggml_backend_comm_allreduce_tensor");
    const auto gather = (ggml_backend_comm_allgather_tensor_t) proc("ggml_backend_comm_allgather_tensor");
    const auto begin = (ggml_backend_capture_begin_t) proc("ggml_backend_capture_begin");
    const auto end = (ggml_backend_capture_end_t) proc("ggml_backend_capture_end");
    const auto launch = (ggml_backend_capture_launch_t) proc("ggml_backend_capture_launch");
    const auto free_graph = (ggml_backend_capture_free_t) proc("ggml_backend_capture_free");
    const auto can_sum = (ggml_backend_comm_allreduce_capturable_t) proc("ggml_backend_comm_allreduce_capturable");
    const auto can_gather = (ggml_backend_comm_allgather_capturable_t) proc("ggml_backend_comm_allgather_capturable");
    if (!init || !release || !sum || !gather || !begin || !end || !launch || !free_graph || !can_sum || !can_gather) {
        fprintf(stderr, "collective benchmark requires internal CUDA collectives and capture support\n");
        return 77;
    }
    std::vector<ggml_backend_t> backends;
    for (auto device : devices) {
        backends.push_back(ggml_backend_dev_init(device, nullptr));
        GGML_ASSERT(backends.back());
    }
    bool ok = true;
    {
        benchmark_launcher runner(backends, launch);
        for (const char * route : {"peer_ll", "host_ll", "peer_copy", "host_copy", "owner-slice"}) {
            const bool peer = strncmp(route, "peer", 4) == 0 || strcmp(route, "owner-slice") == 0;
            const bool copy = strstr(route, "copy") != nullptr;
            set_env("GGML_CUDA_AR_LL_MAX_BYTES", peer || copy ? "0" : "1048576");
            set_env("GGML_CUDA_AR_P2P_MAX_BYTES", peer ? "1048576" : "0");
            set_env("GGML_CUDA_AR_COPY_THRESHOLD", copy ? "1" : "0");
            set_env("GGML_CUDA_AR_COPY_PEER", peer ? "1" : "0");
            set_env("GGML_CUDA_AR_COPY_ONE_STREAM", "0");
            set_env("GGML_CUDA_AR_GATHER_OWNER_SLICES", strcmp(route, "owner-slice") == 0 ? "1" : "0");
            for (bool bf16 : {false, true}) {
                set_env("GGML_CUDA_AR_BF16_THRESHOLD", bf16 ? "1" : "0");
                void * comm = init(backends.data(), backends.size());
                GGML_ASSERT(comm);
                const bool peer_active = last_p2p;
                if (strcmp(route, "owner-slice") == 0 && !last_owner_slice) {
                    fprintf(stderr, "FAIL: owner-slice-peer protocol was not selected\n");
                    return 1;
                }
                for (int rows : {1, 4, 8, 40, 65, 108, 128, 512, 513}) {
                    for (int operation : {0, 1, 2}) { // reset only, reduce, gather
                        if ((bf16 && operation != 1) || (operation == 2 && rows < (int) devices.size())) { continue; } // gather preserves F32 bits
                        const int64_t ne = int64_t(rows)*4096;
                        std::vector<ggml_context *> contexts;
                        std::vector<ggml_backend_buffer_t> buffers;
                        std::vector<ggml_cgraph *> graphs;
                        std::vector<ggml_tensor *> outputs;
                        for (size_t rank = 0; rank < devices.size(); ++rank) {
                            auto * ctx = ggml_init({(size_t(delay_steps) + 16)*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true});
                            contexts.push_back(ctx);
                            auto * graph = ggml_new_graph(ctx);
                            const int64_t owner_begin = operation == 2 ? rows*rank/devices.size() : 0;
                            const int64_t owner_end = operation == 2 ? rows*(rank + 1)/devices.size() : rows;
                            auto * input = operation == 2 ? ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4096, owner_end - owner_begin) :
                                ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ne);
                            ggml_set_input(input);
                            ggml_set_output(input); // replay always starts from the original contribution
                            ggml_tensor * delay = nullptr;
                            if (rank == 0 && delay_steps > 0) {
                                delay = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 262144);
                                ggml_set_input(delay);
                                ggml_set_output(delay);
                                auto * current = delay;
                                for (int step = 0; step < delay_steps; ++step) { current = ggml_scale(ctx, current, 1.0f); }
                                ggml_build_forward_expand(graph, current);
                            }
                            auto * out = operation == 2 ? ggml_pad_ext(ctx, input,
                                0, 0, (int) owner_begin, (int) (rows - owner_end), 0, 0, 0, 0) :
                                ggml_scale(ctx, input, 1.0f);
                            if (operation == 2) { out->flags |= GGML_TENSOR_FLAG_META; }
                            ggml_set_output(out);
                            ggml_build_forward_expand(graph, out);
                            auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, backends[rank]);
                            GGML_ASSERT(buffer);
                            buffers.push_back(buffer);
                            std::vector<float> values(operation == 2 ? ggml_nelements(input) : ne, float(rank + 1));
                            ggml_backend_tensor_set(input, values.data(), 0, ggml_nbytes(input));
                            if (delay) {
                                values.assign(262144, 1.0f);
                                ggml_backend_tensor_set(delay, values.data(), 0, ggml_nbytes(delay));
                            }
                            graphs.push_back(graph);
                            outputs.push_back(out);
                        }
                        const auto queue = [&] {
                            for (size_t rank = 0; rank < devices.size(); ++rank) {
                                GGML_ASSERT(ggml_backend_graph_compute_async(backends[rank], graphs[rank]) == GGML_STATUS_SUCCESS);
                            }
                            if (operation == 1) { GGML_ASSERT(sum(comm, outputs.data())); }
                            if (operation == 2) {
                                const bool gathered = gather(comm, outputs.data());
                                if (!gathered) {
                                    for (size_t rank = 0; rank < devices.size(); ++rank) {
                                        fprintf(stderr, "gather rejected rank=%zu rows=%d ne=[%lld,%lld] nb=[%zu,%zu] bytes=%zu contiguous=%d data=%p flags=%d\n",
                                            rank, rows, (long long) outputs[rank]->ne[0], (long long) outputs[rank]->ne[1],
                                            outputs[rank]->nb[0], outputs[rank]->nb[1], ggml_nbytes(outputs[rank]),
                                            ggml_is_contiguous(outputs[rank]), outputs[rank]->data, outputs[rank]->flags);
                                    }
                                }
                                GGML_ASSERT(gathered);
                            }
                        };
                        queue(); // allocate transport scratch outside capture
                        for (auto backend : backends) { ggml_backend_synchronize(backend); }
                        const bool captured = operation == 0 || (operation == 1 ? can_sum(comm, outputs.data()) : can_gather(comm, outputs.data()));
                        std::vector<void *> execs;
                        if (captured) {
                            for (auto backend : backends) { begin(backend); }
                            queue();
                            for (auto backend : backends) { execs.push_back(end(backend)); GGML_ASSERT(execs.back()); }
                        }
                        const auto run = [&](int count) {
                            if (captured) {
                                runner.run(execs, count);
                            } else {
                                // The copy-engine reduction synchronizes the host and cannot join a whole-evaluation capture.
                                for (int i = 0; i < count; ++i) { queue(); }
                                for (auto backend : backends) { ggml_backend_synchronize(backend); }
                            }
                        };
                        run(8);
                        constexpr int samples = 20, replays = 16;
                        std::vector<double> us;
                        for (int i = 0; i < samples; ++i) {
                            const auto start = ggml_time_us();
                            run(replays);
                            us.push_back(double(ggml_time_us() - start)/replays);
                        }
                        std::vector<float> actual(ne);
                        bool correct = true;
                        for (size_t rank = 0; rank < devices.size(); ++rank) {
                            ggml_backend_tensor_get(outputs[rank], actual.data(), 0, ggml_nbytes(outputs[rank]));
                            for (int64_t i = 0; i < ne; ++i) {
                                const int64_t row = i/4096;
                                size_t owner = 0;
                                while (owner + 1 < devices.size() && row >= int64_t(rows*(owner + 1))/int64_t(devices.size())) {
                                    ++owner;
                                }
                                const float expected = operation == 1 ? float(devices.size()*(devices.size()+1)/2) :
                                    operation == 2 ? float(owner + 1) : float(rank + 1);
                                correct = correct && actual[i] == expected;
                            }
                        }
                        ok = ok && correct;
                        const auto raw_us = us;
                        std::sort(us.begin(), us.end());
                        printf("{\"ranks\":%zu,\"operation\":\"%s\",\"route_requested\":\"%s\",\"peer_enabled\":%s,\"wire\":\"%s\",\"elements\":%lld,\"owner_delay_steps\":%d,\"delay_elements\":262144,\"whole_eval_capture\":%s,\"iterations_per_sample\":%d,\"samples\":%d,\"wall_us_min\":%.3f,\"wall_us_p50\":%.3f,\"wall_us_p95\":%.3f,\"correct\":%s,\"sample_wall_us\":[",
                            devices.size(), operation == 0 ? "reset_only" : operation == 1 ? "reduce" : "gather", route,
                            peer_active ? "true" : "false", bf16 ? "BF16" : "F32", (long long) ne, delay_steps, captured ? "true" : "false", replays, samples,
                            us.front(), us[samples/2], us[samples*95/100], correct ? "true" : "false");
                        for (size_t i = 0; i < raw_us.size(); ++i) { printf("%s%.3f", i ? "," : "", raw_us[i]); }
                        printf("]}\n");
                        fflush(stdout);
                        for (size_t rank = 0; rank < devices.size(); ++rank) {
                            if (captured) { free_graph(backends[rank], execs[rank]); }
                            ggml_backend_buffer_free(buffers[rank]);
                            ggml_free(contexts[rank]);
                        }
                    }
                }
                release(comm);
            }
        }
    }
    for (auto backend : backends) { ggml_backend_free(backend); }
    return ok ? 0 : 1;
}

int main(int argc, char ** argv) {
    const bool logical_benchmark = (argc == 3 || argc == 5) && strcmp(argv[1], "--bench-logical") == 0;
    const bool benchmark = (argc == 3 || argc == 5) &&
        (strcmp(argv[1], "--bench") == 0 || logical_benchmark);
    const int delay_steps = benchmark && argc == 5 && strcmp(argv[3], "--delay") == 0 ? atoi(argv[4]) : 0;
    const bool logical = logical_benchmark || (argc == 3 && strcmp(argv[1], "--logical") == 0);
    const bool require_p2p = argc == 4 && strcmp(argv[3], "--require-p2p") == 0;
    const bool direct = logical || ((argc == 3 || require_p2p) && strcmp(argv[1], "--direct") == 0);
    const int n_devices = direct || benchmark ? atoi(argv[2]) : 2;
    if ((argc != 1 && !direct && !benchmark) || n_devices < 2 || n_devices > 4 || delay_steps < 0 || delay_steps > 64 ||
            (benchmark && argc == 5 && strcmp(argv[3], "--delay") != 0)) {
        fprintf(stderr, "usage: %s [--direct 2|3|4 [--require-p2p] | --logical 2|3|4 | --bench[-logical] 2|3|4 [--delay 0..64]]\n", argv[0]);
        return 1;
    }
    std::vector<ggml_backend_dev_t> gpus;
    for (size_t i = 0; i < ggml_backend_dev_count() && gpus.size() < (size_t) n_devices; i++) {
        if (ggml_backend_dev_type(ggml_backend_dev_get(i)) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            gpus.push_back(ggml_backend_dev_get(i));
        }
    }
    if (logical && !gpus.empty()) {
        const auto physical = gpus;
        for (size_t rank = gpus.size(); rank < (size_t) n_devices; ++rank) {
            gpus.push_back(physical[rank % physical.size()]);
        }
        printf("%d logical ranks on %zu physical GPUs; not a four-card qualification\n", n_devices, physical.size());
    }
    if ((direct || benchmark) && gpus.size() < (size_t) n_devices) {
        printf("need %d physical GPUs for direct collectives, found %zu, skipped\n", n_devices, gpus.size());
        return 77;
    }
    if (gpus.empty()) {
        printf("no GPU for the meta backend, skipped\n");
        return 0;
    }
    if (gpus.size() == 1) {
        gpus.push_back(gpus[0]);
    }
    if (strncmp(ggml_backend_dev_name(gpus[0]), "CUDA", 4) != 0) {
        printf("%s has no internal AllReduce, skipped\n", ggml_backend_dev_name(gpus[0]));
        return direct || benchmark ? 77 : 0;
    }
    fprintf(benchmark ? stderr : stdout, "meta backend over %s and %s\n", ggml_backend_dev_name(gpus[0]), ggml_backend_dev_name(gpus[1]));

    ggml_log_set(log_cb, nullptr);
    set_env("GGML_CUDA_ALLREDUCE", "internal");
    set_env("GGML_CUDA_AR_COPY_THRESHOLD", nullptr);
    set_env("GGML_CUDA_AR_COPY_CHUNK_BYTES", nullptr);
    if (benchmark) {
        set_env("GGML_CUDA_GRAPH_MEM_STATS", "0");
        return benchmark_collectives(gpus, delay_steps);
    }
    if (direct) {
        return test_direct(gpus, require_p2p);
    }

    std::mt19937 rng(42);
    std::vector<std::vector<float>> w_data(2*n_layer);
    for (int il = 0; il < n_layer; il++) {
        for (int k = 0; k < 2; k++) {
            // up is [n_embd, n_ff] and down [n_ff, n_embd]: uniform over +-1/sqrt(the row length)
            const float scale = 1.0f/std::sqrt((float) (k == 0 ? n_embd : n_ff));
            std::uniform_real_distribution<float> dist(-scale, scale);
            w_data[2*il + k].resize(n_embd*n_ff);
            for (float & v : w_data[2*il + k]) {
                v = dist(rng);
            }
        }
    }

    // the inputs and the CPU's outputs, once for every pass
    std::vector<std::vector<float>> xs(n_evals);
    std::vector<std::vector<float>> refs(n_evals);
    {
        ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        ggml_backend_cpu_set_n_threads(cpu, (int) std::max(1u, std::thread::hardware_concurrency()));
        ffn_stack stack_cpu(&cpu, 1, ggml_backend_get_default_buffer_type(cpu), w_data);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (int e = 0; e < n_evals; e++) {
            xs[e].resize(n_embd*n_tok[order[e]]);
            for (float & v : xs[e]) {
                v = dist(rng);
            }
            refs[e] = stack_cpu.compute(order[e], xs[e]);
        }
        ggml_backend_free(cpu);
    }

    struct pass {
        const char * name;
        const char * bf16_threshold; // GGML_CUDA_AR_BF16_THRESHOLD, nullptr for the default
        bool         one_stream;     // GGML_CUDA_AR_COPY_ONE_STREAM=1
        bool         card_to_card;   // GGML_CUDA_AR_COPY_PEER left to its default; 0 otherwise
    };
    // each wire's passes after the sends and card to card follow its pass beside them, whose outputs they must equal
    const pass passes[] = {
        {"BF16 wire, beside the sends", nullptr, false, false},
        {"BF16 wire, after the sends",  nullptr, true,  false},
        {"BF16 wire, card to card",     nullptr, false, true},
        {"F32 wire, beside the sends",  "0",     false, false},
        {"F32 wire, after the sends",   "0",     true,  false},
        {"F32 wire, card to card",      "0",     false, true},
    };

    bool ok = true;
    std::vector<std::vector<float>> beside(n_evals); // the last pass beside the sends, its outputs
    ggml_backend_dev_t meta = ggml_backend_meta_device(gpus.data(), 2, split_by_name, nullptr);
    for (const pass & ps : passes) {
        set_env("GGML_CUDA_AR_BF16_THRESHOLD", ps.bf16_threshold);
        set_env("GGML_CUDA_AR_COPY_ONE_STREAM", ps.one_stream ? "1" : nullptr);
        set_env("GGML_CUDA_AR_COPY_PEER", ps.card_to_card ? nullptr : "0");

        const int n_pipelines_before = n_pipelines;
        ggml_backend_t cpu     = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        ggml_backend_t backend = ggml_backend_dev_init(meta, nullptr);
        {
            ggml_backend_t backends[2] = {backend, cpu};
            ffn_stack stack(backends, 2, ggml_backend_dev_buffer_type(meta), w_data);
            for (int e = 0; e < n_evals; e++) {
                const int k = order[e];
                const std::vector<float> out = stack.compute(k, xs[e]);
                const double err = nmse(refs[e], out);
                const bool ok_eval = std::isfinite(err) && err < 1e-4;
                printf("%s %s, eval %d, %lld columns: NMSE vs CPU %.3e\n", ok_eval ? "ok  " : "FAIL", ps.name, e + 1,
                       (long long) n_tok[k], err);
                ok = ok && ok_eval;
                if (!ps.one_stream && !ps.card_to_card) {
                    beside[e] = out;
                    continue;
                }
                size_t n_diff = 0;
                double max_diff = 0.0;
                for (size_t i = 0; i < out.size(); i++) {
                    if (memcmp(&out[i], &beside[e][i], sizeof(float)) != 0) {
                        n_diff++;
                        max_diff = std::max(max_diff, (double) std::fabs(out[i] - beside[e][i]));
                    }
                }
                printf("%s %s, eval %d: %zu of %zu values differ from beside the sends (largest by %.3e)\n",
                       n_diff == 0 ? "ok  " : "FAIL", ps.name, e + 1, n_diff, out.size(), max_diff);
                ok = ok && n_diff == 0;
            }
        }
        // card to card wherever the cards map each other's memory and the pass leaves the switch on, and nowhere else
        const bool want_card_to_card = ps.card_to_card && last_p2p;
        const bool ok_pipeline = n_pipelines == n_pipelines_before + 1 && last_one_stream == ps.one_stream &&
            last_card_to_card == want_card_to_card;
        printf("%s %s: %d internal AllReduce pipeline initialized (1 expected), receives %s (%s expected)%s\n",
               ok_pipeline ? "ok  " : "FAIL", ps.name, n_pipelines - n_pipelines_before,
               last_card_to_card ? "card to card" : last_one_stream ? "after the sends" : "beside the sends",
               want_card_to_card ? "card to card" : ps.one_stream ? "after the sends" : "beside the sends",
               ps.card_to_card && !last_p2p ? ", the cards cannot map each other's memory" : "");
        ok = ok && ok_pipeline;
        ggml_backend_free(backend);
        ggml_backend_free(cpu);
    }
    ggml_log_set(nullptr, nullptr);

    printf("%s\n", ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}
