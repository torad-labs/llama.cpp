// GOAL: Marcos serves GLM-5.3 Flash (glm5next) on two GPUs with -sm tensor and MTP drafts, sampled at temp 1.0, top-p 0.95
// and repeat penalty 1.1 over the last 256 tokens. Under -sm tensor the engine refused backend sampling: every step copied
// the full vocab-wide logits rows to the host and sampled on the CPU, the MTP drafter's sampler too. Done means: with a
// model split by -sm tensor, a sequence's sampler chain (penalties, top-k, top-p, min-p, temp, dist, greedy, logit-bias)
// runs in the graph on the devices, and for the same seed and logits it draws exactly the tokens the single-device backend
// sampler draws; the MTP draft sampler offloads too; nothing served changes except where the tokens are drawn.
//
// Under -sm tensor the output head is split by vocab, so the logits are split along axis 0. A ggml_pad along the split
// axis gathers them: the result is the padded tensor on every device, bit for bit what one device's ggml_pad makes. The
// sampler chain then runs on the gathered rows on every device.
//
// This test runs an output head split by vocab (n_lo columns on the first device, the rest on the second) on the meta
// backend, gathers its logits, and applies llama.cpp's sampler chains to each row (one sequence a row). The same chains,
// with the same seeds, run on one GPU on the logits the meta backend made. Each evaluation checks that:
//   - the gathered rows are the logits bit for bit (some logits are -0.0 and some +0.0), and the padding is +0.0
//     (the rows are padded to a multiple of 32 entries, as llama.cpp pads them),
//   - every chain output (the sampled token, its probabilities, candidates and logits) is bit for bit the single GPU's.
// The sampled tokens are accepted into both chains, so the repeat penalty sees a history. Each meta graph is evaluated 4
// times: from its second evaluation on, it must be captured as a graph without a gather is (no "not captured" log).
// The reference is the chains on one GPU, every op pinned there, reading padded rows as llama.cpp builds them on one
// device. Before the cases, it checks that the cumulative sum dist draws from comes out the same bits run after run on a
// GLM-5.3 Flash-wide row: if it did not, no two runs, on one device or two, could draw the same tokens.
// On the first two GPUs, or on the first GPU twice; with no GPU the test is skipped.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "llama.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static constexpr int64_t n_embd = 64;
static constexpr int64_t n_lo   = 384; // vocab entries on the first device
static constexpr int     n_eval = 4;

// the entries llama.cpp's build_sampling pads a row of n_vocab logits with, to keep the rows 128-byte aligned
static int64_t n_pad_of(int64_t n_vocab) {
    return 32 - n_vocab % 32;
}

static ggml_backend_meta_split_state split_by_name(const ggml_tensor * tensor, void * /*userdata*/) {
    // out: [n_embd, n_vocab], its rows; mask: [n_vocab], its entries
    const bool out  = strcmp(tensor->name, "out")  == 0;
    const bool mask = strcmp(tensor->name, "mask") == 0;
    if (out || mask) {
        const int64_t n_vocab = out ? tensor->ne[1] : tensor->ne[0];
        ggml_backend_meta_split_state split_state = {out ? GGML_BACKEND_SPLIT_AXIS_1 : GGML_BACKEND_SPLIT_AXIS_0, {0}, {1}, 1};
        split_state.ne[0] = n_lo;
        split_state.ne[1] = n_vocab - n_lo;
        return split_state;
    }
    return {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
}

struct log_counts {
    int executables  = 0; // ggml_backend_cuda_capture_end made one
    int not_captured = 0;
};

static void log_cb(ggml_log_level level, const char * text, void * user_data) {
    log_counts * counts = (log_counts *) user_data;
    if (strstr(text, "ggml_backend_cuda_capture_end") != nullptr && strstr(text, " nodes, ") != nullptr) {
        counts->executables++;
    }
    if (strstr(text, "not captured") != nullptr) {
        counts->not_captured++;
        fputs(text, stderr);
    }
    if (level == GGML_LOG_LEVEL_WARN || level == GGML_LOG_LEVEL_ERROR) {
        fputs(text, stderr);
    }
}

enum chain_kind {
    CHAIN_GREEDY, // greedy
    CHAIN_SERVED, // penalties (last 256, repeat 1.1), top-p 0.95, temp 1.0, dist
    CHAIN_TOP_K,  // logit-bias, top-k 40, min-p 0.05, temp 0.7, dist
};

static const char * chain_name(chain_kind kind) {
    switch (kind) {
        case CHAIN_GREEDY: return "greedy";
        case CHAIN_SERVED: return "penalties+top_p+temp+dist";
        case CHAIN_TOP_K:  return "logit_bias+top_k+min_p+temp+dist";
    }
    return "?";
}

static llama_sampler * make_chain(chain_kind kind, int32_t n_vocab, uint32_t seed) {
    llama_sampler * chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
    switch (kind) {
        case CHAIN_GREEDY: {
            llama_sampler_chain_add(chain, llama_sampler_init_greedy());
        } break;
        case CHAIN_SERVED: {
            llama_sampler_chain_add(chain, llama_sampler_init_penalties(n_vocab, 256, 1.1f, 0.0f, 0.0f));
            llama_sampler_chain_add(chain, llama_sampler_init_top_p(0.95f, 0));
            llama_sampler_chain_add(chain, llama_sampler_init_temp(1.0f));
            llama_sampler_chain_add(chain, llama_sampler_init_dist(seed));
        } break;
        case CHAIN_TOP_K: {
            const llama_logit_bias bias[2] = {{5, 3.0f}, {17, -INFINITY}};
            llama_sampler_chain_add(chain, llama_sampler_init_logit_bias(n_vocab, 2, bias));
            llama_sampler_chain_add(chain, llama_sampler_init_top_k(40));
            llama_sampler_chain_add(chain, llama_sampler_init_min_p(0.05f, 0));
            llama_sampler_chain_add(chain, llama_sampler_init_temp(0.7f));
            llama_sampler_chain_add(chain, llama_sampler_init_dist(seed));
        } break;
    }
    return chain;
}

// the head's weights, and the mask that makes some logits -0.0 and some +0.0, in a buffer of buft
struct head_weights {
    ggml_context *        ctx  = nullptr;
    ggml_backend_buffer_t buf  = nullptr;
    ggml_tensor *         out  = nullptr;
    ggml_tensor *         mask = nullptr;

    head_weights(ggml_backend_buffer_type_t buft, int64_t n_vocab, std::mt19937 & rng) {
        const ggml_init_params params = {
            /*.mem_size   =*/ 2*ggml_tensor_overhead(),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ctx  = ggml_init(params);
        out  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_vocab);
        ggml_set_name(out, "out");
        mask = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_vocab);
        ggml_set_name(mask, "mask");
        buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
        GGML_ASSERT(buf != nullptr);
        ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

        std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
        std::vector<float> data(ggml_nelements(out));
        for (float & v : data) {
            v = dist(rng);
        }
        ggml_backend_tensor_set(out, data.data(), 0, ggml_nbytes(out));
        std::vector<float> m(n_vocab, 1.0f);
        for (int64_t i = 3; i < n_vocab; i += 7) {
            m[i] = -0.0f; // the logit is a zero with the opposite sign of the head's value
        }
        ggml_backend_tensor_set(mask, m.data(), 0, ggml_nbytes(mask));
    }

    ~head_weights() {
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }
};

// meta: the head on x, its logits gathered by a pad along the vocab, the chains on the gathered rows
// ref:  the chains on the rows of logits given as an input
struct sampling_graph {
    ggml_context * ctx      = nullptr;
    ggml_cgraph  * gf       = nullptr;
    ggml_tensor  * x        = nullptr;
    ggml_tensor  * logits   = nullptr;
    ggml_tensor  * gathered = nullptr;
    std::vector<llama_sampler_data> out;

    sampling_graph(const head_weights * w, int64_t n_vocab, int64_t n_rows, const std::vector<llama_sampler *> & chains) {
        const size_t max_nodes = 8192;
        const ggml_init_params params = {
            /*.mem_size   =*/ max_nodes*ggml_tensor_overhead() + ggml_graph_overhead_custom(max_nodes, false),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ctx = ggml_init(params);
        gf  = ggml_new_graph_custom(ctx, max_nodes, false);

        ggml_tensor * rows = nullptr;
        if (w != nullptr) {
            x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_rows);
            ggml_set_input(x);
            logits = ggml_mul(ctx, ggml_mul_mat(ctx, w->out, x), w->mask);
            ggml_set_output(logits);
            rows = ggml_pad_ext(ctx, logits, 0, (int) n_pad_of(n_vocab), 0, 0, 0, 0, 0, 0);
            // penalties writes into the rows it is given, so the gather is checked on a copy made before the chains run
            gathered = ggml_cont(ctx, rows);
            ggml_set_output(gathered);
            ggml_build_forward_expand(gf, logits);
            ggml_build_forward_expand(gf, gathered);
        } else {
            logits = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_vocab, n_rows);
            ggml_set_input(logits);
            // as llama builds it on one device: the chains read rows padded out of the logits, never the input itself
            rows = ggml_pad_ext(ctx, logits, 0, (int) n_pad_of(n_vocab), 0, 0, 0, 0, 0, 0);
        }

        for (int64_t r = 0; r < n_rows; r++) {
            llama_sampler_data data = {
                /*.logits     =*/ ggml_view_1d(ctx, rows, n_vocab, r*rows->nb[1]),
                /*.probs      =*/ nullptr,
                /*.sampled    =*/ nullptr,
                /*.candidates =*/ nullptr,
            };
            chains[r]->iface->backend_apply(chains[r], ctx, gf, &data);
            for (ggml_tensor * t : {data.logits, data.probs, data.sampled, data.candidates}) {
                if (t != nullptr) {
                    ggml_set_output(t);
                    ggml_build_forward_expand(gf, t);
                }
            }
            out.push_back(data);
        }
    }

    ~sampling_graph() {
        ggml_free(ctx);
    }
};

static std::vector<uint8_t> read_bytes(const ggml_tensor * t) {
    std::vector<uint8_t> v(t == nullptr ? 0 : ggml_nbytes(t));
    if (t != nullptr) {
        ggml_backend_tensor_get(t, v.data(), 0, v.size());
    }
    return v;
}

// whether one output of a chain holds the same bytes on the meta device and the single device; says where it differs
static bool same_output(const char * name, const ggml_tensor * m, const ggml_tensor * f) {
    if ((m == nullptr) != (f == nullptr)) {
        printf("     %s: present on one side only\n", name);
        return false;
    }
    const std::vector<uint8_t> a = read_bytes(m);
    const std::vector<uint8_t> b = read_bytes(f);
    if (a == b) {
        return true;
    }
    if (a.size() != b.size()) {
        printf("     %s: %zu bytes vs %zu\n", name, a.size(), b.size());
        return false;
    }
    const size_t n = a.size()/4;
    size_t n_diff  = 0;
    size_t first   = n;
    for (size_t i = 0; i < n; i++) {
        if (memcmp(&a[4*i], &b[4*i], 4) != 0) {
            n_diff++;
            first = first == n ? i : first;
        }
    }
    float fa;
    float fb;
    int32_t ia;
    int32_t ib;
    memcpy(&fa, &a[4*first], 4);
    memcpy(&fb, &b[4*first], 4);
    memcpy(&ia, &a[4*first], 4);
    memcpy(&ib, &b[4*first], 4);
    printf("     %s %s ne %lld: %zu of %zu differ, first at %zu: %g/%g (as int %d/%d)\n", name, ggml_type_name(m->type),
        (long long) ggml_nelements(m), n_diff, n, first, fa, fb, ia, ib);
    return false;
}

// one shape and one chain kind; returns whether every evaluation matched
static bool run_case(ggml_backend_t meta_backend, ggml_backend_t gpu_backend, ggml_backend_t cpu, ggml_backend_dev_t meta,
        ggml_backend_dev_t gpu, chain_kind kind, int64_t n_vocab, int64_t n_rows, std::mt19937 & rng) {
    head_weights w(ggml_backend_dev_buffer_type(meta), n_vocab, rng);

    std::vector<llama_sampler *> chains_meta;
    std::vector<llama_sampler *> chains_ref;
    for (int64_t r = 0; r < n_rows; r++) {
        chains_meta.push_back(make_chain(kind, n_vocab, 1000 + r));
        chains_ref .push_back(make_chain(kind, n_vocab, 1000 + r));
        GGML_ASSERT(chains_meta[r]->iface->backend_init(chains_meta[r], ggml_backend_dev_buffer_type(meta), 1));
        GGML_ASSERT(chains_ref [r]->iface->backend_init(chains_ref [r], ggml_backend_dev_buffer_type(gpu),  1));
        for (llama_token t = 20; t < 40; t++) {
            llama_sampler_accept(chains_meta[r], t);
            llama_sampler_accept(chains_ref [r], t);
        }
    }

    sampling_graph g_meta(&w,      n_vocab, n_rows, chains_meta);
    sampling_graph g_ref (nullptr, n_vocab, n_rows, chains_ref);

    ggml_backend_t backends_meta[2] = {meta_backend, cpu};
    ggml_backend_t backends_ref [2] = {gpu_backend,  cpu};
    ggml_backend_sched_t sched_meta = ggml_backend_sched_new(backends_meta, nullptr, 2, 8192, false, true);
    ggml_backend_sched_t sched_ref  = ggml_backend_sched_new(backends_ref,  nullptr, 2, 8192, false, true);
    // the scheduler puts a graph input, and the ops reading it, on the CPU: every op is pinned to the device under test,
    // so the reference is the backend sampler on one GPU and the meta graph's chains run on the meta device
    for (int i = 0; i < ggml_graph_n_nodes(g_ref.gf); i++) {
        ggml_backend_sched_set_tensor_backend(sched_ref, ggml_graph_node(g_ref.gf, i), gpu_backend);
    }
    for (int i = 0; i < ggml_graph_n_nodes(g_meta.gf); i++) {
        ggml_backend_sched_set_tensor_backend(sched_meta, ggml_graph_node(g_meta.gf, i), meta_backend);
    }
    GGML_ASSERT(ggml_backend_sched_alloc_graph(sched_meta, g_meta.gf));
    GGML_ASSERT(ggml_backend_sched_alloc_graph(sched_ref,  g_ref.gf));

    bool ok = true;
    for (int e = 0; e < n_eval; e++) {
        std::vector<float> x(n_embd*n_rows);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (float & v : x) {
            v = dist(rng);
        }
        ggml_backend_tensor_set(g_meta.x, x.data(), 0, ggml_nbytes(g_meta.x));
        for (llama_sampler * c : chains_meta) {
            c->iface->backend_set_input(c);
        }
        GGML_ASSERT(ggml_backend_sched_graph_compute(sched_meta, g_meta.gf) == GGML_STATUS_SUCCESS);

        std::vector<float> logits(n_vocab*n_rows);
        ggml_backend_tensor_get(g_meta.logits, logits.data(), 0, ggml_nbytes(g_meta.logits));
        const int64_t n_pad = n_pad_of(n_vocab);
        std::vector<float> gathered((n_vocab + n_pad)*n_rows);
        ggml_backend_tensor_get(g_meta.gathered, gathered.data(), 0, ggml_nbytes(g_meta.gathered));

        int64_t n_neg_zero = 0;
        int64_t n_pos_zero = 0;
        bool ok_gather = true;
        for (int64_t r = 0; r < n_rows; r++) {
            const float * a = logits.data()   + r*n_vocab;
            const float * b = gathered.data() + r*(n_vocab + n_pad);
            const std::vector<float> pad(n_pad, 0.0f);
            ok_gather = ok_gather && memcmp(a, b, n_vocab*sizeof(float)) == 0;
            ok_gather = ok_gather && memcmp(&b[n_vocab], pad.data(), n_pad*sizeof(float)) == 0;
            for (int64_t i = 0; i < n_vocab; i++) {
                n_neg_zero += a[i] == 0.0f &&  std::signbit(a[i]);
                n_pos_zero += a[i] == 0.0f && !std::signbit(a[i]);
            }
        }
        GGML_ASSERT(n_neg_zero > 0 && n_pos_zero > 0);

        ggml_backend_tensor_set(g_ref.logits, logits.data(), 0, ggml_nbytes(g_ref.logits));
        for (llama_sampler * c : chains_ref) {
            c->iface->backend_set_input(c);
        }
        GGML_ASSERT(ggml_backend_sched_graph_compute(sched_ref, g_ref.gf) == GGML_STATUS_SUCCESS);

        bool ok_chain = true;
        std::string tokens;
        for (int64_t r = 0; r < n_rows; r++) {
            const llama_sampler_data & m = g_meta.out[r];
            const llama_sampler_data & f = g_ref.out[r];
            ok_chain = same_output("sampled",    m.sampled,    f.sampled)    && ok_chain;
            ok_chain = same_output("probs",      m.probs,      f.probs)      && ok_chain;
            ok_chain = same_output("logits",     m.logits,     f.logits)     && ok_chain;
            ok_chain = same_output("candidates", m.candidates, f.candidates) && ok_chain;

            GGML_ASSERT(m.sampled != nullptr && f.sampled != nullptr);
            llama_token tok_meta;
            llama_token tok_ref;
            ggml_backend_tensor_get(m.sampled, &tok_meta, 0, sizeof(tok_meta));
            ggml_backend_tensor_get(f.sampled, &tok_ref,  0, sizeof(tok_ref));
            tokens += " " + std::to_string(tok_meta) + (tok_meta == tok_ref ? "" : "/" + std::to_string(tok_ref));
            llama_sampler_accept(chains_meta[r], tok_meta);
            llama_sampler_accept(chains_ref [r], tok_ref);
        }

        const bool ok_eval = ok_gather && ok_chain;
        printf("%s %-34s vocab %6lld rows %lld eval %d: gather %s, chain outputs %s, tokens%s\n", ok_eval ? "ok  " : "FAIL",
            chain_name(kind), (long long) n_vocab, (long long) n_rows, e, ok_gather ? "exact" : "DIFFER",
            ok_chain ? "equal" : "DIFFER", tokens.c_str());
        ok = ok && ok_eval;
    }

    ggml_backend_sched_free(sched_meta);
    ggml_backend_sched_free(sched_ref);
    for (llama_sampler * c : chains_meta) {
        llama_sampler_free(c);
    }
    for (llama_sampler * c : chains_ref) {
        llama_sampler_free(c);
    }
    return ok;
}

// dist draws its token where the cumulative sum of the probabilities crosses a uniform: for one seed and one set of
// logits to draw one token, the sum must come out the same bits every time it is computed
static bool cumsum_reproducible(ggml_backend_t backend, int64_t n, std::mt19937 & rng) {
    const ggml_init_params params = {
        /*.mem_size   =*/ 4*ggml_tensor_overhead() + ggml_graph_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    ggml_tensor * probs = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    ggml_tensor * sums  = ggml_cumsum(ctx, probs);
    ggml_cgraph * gf    = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, sums);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);

    std::vector<float> p(n);
    std::uniform_real_distribution<float> dist(0.0f, 2.0f/n);
    for (float & v : p) {
        v = dist(rng);
    }
    ggml_backend_tensor_set(probs, p.data(), 0, ggml_nbytes(probs));

    const int n_runs = 50;
    std::vector<uint8_t> first;
    int n_differ = 0;
    for (int i = 0; i < n_runs; i++) {
        GGML_ASSERT(ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS);
        const std::vector<uint8_t> got = read_bytes(sums);
        if (i == 0) {
            first = got;
        }
        n_differ += got != first;
    }
    printf("%s cumsum of %lld probabilities, %d runs: %d came out different from the first\n", n_differ == 0 ? "ok  " : "FAIL",
        (long long) n, n_runs, n_differ);

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return n_differ == 0;
}

int main() {
    std::vector<ggml_backend_dev_t> gpus;
    for (size_t i = 0; i < ggml_backend_dev_count() && gpus.size() < 2; i++) {
        if (ggml_backend_dev_type(ggml_backend_dev_get(i)) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            gpus.push_back(ggml_backend_dev_get(i));
        }
    }
    if (gpus.empty()) {
        printf("no GPU for the meta backend, skipped\n");
        return 0;
    }
    if (gpus.size() == 1) {
        gpus.push_back(gpus[0]);
    }
    printf("meta backend over %s and %s\n", ggml_backend_dev_name(gpus[0]), ggml_backend_dev_name(gpus[1]));

    log_counts counts;
    ggml_log_set(log_cb, &counts);
    llama_log_set(log_cb, &counts);

    ggml_backend_dev_t meta = ggml_backend_meta_device(gpus.data(), 2, split_by_name, nullptr);

    ggml_backend_t cpu          = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    ggml_backend_t meta_backend = ggml_backend_dev_init(meta, nullptr);
    ggml_backend_t gpu_backend  = ggml_backend_dev_init(gpus[0], nullptr);

    std::mt19937 rng(42);
    struct shape {
        int64_t n_vocab;
        int64_t n_rows;
    };
    // 154880 is GLM-5.3 Flash's vocab: its 4 rows are 2.4 MB, more than one chunk of an all-reduce
    const shape shapes[] = {{1000, 1}, {1000, 3}, {154880, 4}};
    const chain_kind kinds[] = {CHAIN_GREEDY, CHAIN_SERVED, CHAIN_TOP_K};

    bool ok = cumsum_reproducible(gpu_backend, 154880, rng);
    int n_cases = 0;
    for (const shape & s : shapes) {
        for (chain_kind kind : kinds) {
            ok = run_case(meta_backend, gpu_backend, cpu, meta, gpus[0], kind, s.n_vocab, s.n_rows, rng) && ok;
            n_cases++;
        }
    }

    // as in test-backend-meta-capture: one executable a device for each meta graph, none under GGML_META_CAPTURE_LEGACY=1
    const bool legacy      = ggml_env_switch("GGML_META_CAPTURE_LEGACY");
    const int  executables = legacy ? 0 : n_cases*2;
    const bool ok_captures = counts.executables == executables && counts.not_captured == 0;
    printf("%s %d executables made (%d expected), %d graphs not captured\n", ok_captures ? "ok  " : "FAIL",
        counts.executables, executables, counts.not_captured);
    ok = ok && ok_captures;

    ggml_backend_free(gpu_backend);
    ggml_backend_free(meta_backend);
    ggml_backend_free(cpu);
    ggml_log_set(nullptr, nullptr);
    llama_log_set(nullptr, nullptr);

    printf("%s\n", ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}
