// llama-gram: the full calibration Gram (the sum of x x^T over every token) of each projection's input,
// gathered on the model's own forward.
//
// What it is: llama-imatrix keeps the diagonal of that sum, the squared activation per input column. GPTQ
// chooses a projection's codes on the whole matrix, every off-diagonal included, so this tool keeps all of it.
// The projections that read one input share one Gram: q, k and v read the attention norm's output; the
// attention output projection reads the attention; gate and up read the FFN norm's output; down reads the
// gated product; the head reads the final norm's output. The tool holds every member of a group to the one
// input tensor the group's first member read in that ubatch, and stops if they differ.
//
// How: the scheduler's eval callback stops after every weight matmul, as llama-imatrix's does. The input is
// on the card already; cuBLAS's SYRK forms that ubatch's x x^T in binary32, in its pedantic math mode (never
// TF32), and the lower triangle is added on the host into the group's running sum, also binary32. Each
// sequence is decoded with logits on for every token, so the last layer and the head see every row.
//
// What it writes, into a directory that must not exist yet:
//   <folder>/<group>.hessian  the group's upper triangle row by row (i <= j < n), little-endian binary32
//   gram.json                 written last: the model file, the sequences and tokens, and for every group its
//                             width, token count, file and members, each member with its GGUF type, shape
//                             and the byte range of its data in the model file
// Folders are blk.<i> per layer and output for the head; groups are attn_in, attn_out_in, ffn_in,
// ffn_down_in and head_in.
//
// Input: --sequences names a file with one path per line; each path holds one sequence's token ids as
// consecutive little-endian 4-byte words. Nothing is tokenized here and no BOS is added.

#include "build-info.h"
#include "common.h"
#include "ggml-backend.h"
#include "gguf.h"
#include "llama.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

static_assert(sizeof(float) == 4, "binary32 floats");

namespace fs = std::filesystem;

[[noreturn]] static void gram_die(const char * fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "llama-gram: ");
    vfprintf(stderr, fmt, ap);
    fprintf(stderr, "\n");
    va_end(ap);
    exit(1);
}

#define CUDA_OK(x)                                                                         \
    do {                                                                                   \
        cudaError_t e_ = (x);                                                              \
        if (e_ != cudaSuccess) gram_die("%s: %s", #x, cudaGetErrorString(e_));                  \
    } while (0)
#define CUBLAS_OK(x)                                                                       \
    do {                                                                                   \
        cublasStatus_t s_ = (x);                                                           \
        if (s_ != CUBLAS_STATUS_SUCCESS) gram_die("%s: cuBLAS status %d", #x, (int) s_);       \
    } while (0)

// CUDA0#blk.0.attn_k.weight#0 => blk.0.attn_k.weight (the scheduler's copy names), as llama-imatrix reads them
static std::string weight_name(const char * name) {
    const char * p = strchr(name, '#');
    if (!p) return name;
    p++;
    const char * q = strchr(p, '#');
    return q ? std::string(p, q - p) : std::string(p);
}

struct group_key {
    std::string folder;
    std::string group;
};

// The group a weight's input belongs to, or false for a matmul that is not a projection of the model.
// A weight under blk. of a kind this map does not name stops the run: its input would be left out.
static bool group_of(const std::string & w, group_key & key) {
    if (w == "output.weight" || w == "token_embd.weight") {
        key = { "output", "head_in" };
        return true;
    }
    if (w.rfind("blk.", 0) != 0) return false;
    const size_t dot = w.find('.', 4);
    if (dot == std::string::npos) return false;
    const std::string layer = w.substr(4, dot - 4);
    if (layer.empty() || layer.find_first_not_of("0123456789") != std::string::npos) return false;
    const std::string kind = w.substr(dot + 1);
    static const std::map<std::string, std::string> kinds = {
        { "attn_q.weight", "attn_in" },        { "attn_k.weight", "attn_in" },
        { "attn_v.weight", "attn_in" },        { "attn_qkv.weight", "attn_in" },
        { "attn_output.weight", "attn_out_in" },
        { "ffn_gate.weight", "ffn_in" },       { "ffn_up.weight", "ffn_in" },
        { "ffn_gate_up.weight", "ffn_in" },
        { "ffn_down.weight", "ffn_down_in" },
    };
    auto it = kinds.find(kind);
    if (it == kinds.end()) {
        gram_die("%s is a weight matmul of a kind this tool does not group; its input would be left out", w.c_str());
    }
    key = { "blk." + layer, it->second };
    return true;
}

struct member {
    std::string name;
    int64_t     ne0 = 0, ne1 = 0;
    ggml_type   type = GGML_TYPE_COUNT;
    uint64_t    tokens = 0;
    uint64_t    round = UINT64_MAX;
};

struct group {
    group_key                       key;
    int64_t                         n = 0;
    std::vector<float>              tri;  // upper triangle, row by row
    std::vector<std::string>        members;
    uint64_t                        tokens = 0;
    uint64_t                        round = UINT64_MAX;
    const ggml_tensor *             src1 = nullptr;
    const void *                    data = nullptr;
};

struct collector {
    std::mutex                       mutex;
    std::map<std::string, group>     groups;  // folder/group
    std::map<std::string, member>    members; // weight name
    std::atomic<uint64_t>            round{0};
    int                              device = 0;
    cublasHandle_t                   cublas = nullptr;
    float *                          d_c = nullptr;  // n x n, column-major
    size_t                           d_c_elems = 0;
    float *                          d_a = nullptr;  // a host input copied up, when the input is on the host
    size_t                           d_a_elems = 0;
    float *                          h_c = nullptr;  // pinned copy of d_c
    size_t                           h_c_elems = 0;
    unsigned                         threads = 1;

    void init() {
        CUDA_OK(cudaSetDevice(device));
        CUBLAS_OK(cublasCreate(&cublas));
        // binary32 FMA only: no TF32 or other reduced-precision tensor-core path
        CUBLAS_OK(cublasSetMathMode(cublas, CUBLAS_PEDANTIC_MATH));
        unsigned hc = std::thread::hardware_concurrency();
        threads = std::max(1u, std::min(32u, hc / 2));
    }

    float * device_c(int64_t n) {
        const size_t need = (size_t) n * (size_t) n;
        if (need > d_c_elems) {
            if (d_c) CUDA_OK(cudaFree(d_c));
            CUDA_OK(cudaMalloc(&d_c, need * sizeof(float)));
            d_c_elems = need;
        }
        if (need > h_c_elems) {
            if (h_c) CUDA_OK(cudaFreeHost(h_c));
            CUDA_OK(cudaMallocHost(&h_c, need * sizeof(float)));
            h_c_elems = need;
        }
        return d_c;
    }

    // add this ubatch's x x^T over the rows of src1 (ne0 = n wide) into g's triangle
    void accumulate(group & g, const ggml_tensor * src1) {
        const int64_t n = g.n;
        float * c = device_c(n);
        const bool on_host = src1->buffer == nullptr || ggml_backend_buffer_is_host(src1->buffer);
        const float one = 1.0f;
        bool first = true;
        for (int64_t i3 = 0; i3 < src1->ne[3]; ++i3) {
            for (int64_t i2 = 0; i2 < src1->ne[2]; ++i2) {
                const char * base = (const char *) src1->data + i2 * src1->nb[2] + i3 * src1->nb[3];
                const int64_t rows = src1->ne[1];
                if (rows == 0) continue;
                const float * a = nullptr;
                int lda = 0;
                if (on_host) {
                    const size_t need = (size_t) n * (size_t) rows;
                    if (need > d_a_elems) {
                        if (d_a) CUDA_OK(cudaFree(d_a));
                        CUDA_OK(cudaMalloc(&d_a, need * sizeof(float)));
                        d_a_elems = need;
                    }
                    CUDA_OK(cudaMemcpy2D(d_a, n * sizeof(float), base, src1->nb[1], n * sizeof(float), rows,
                                         cudaMemcpyHostToDevice));
                    a = d_a;
                    lda = (int) n;
                } else {
                    cudaPointerAttributes attr{};
                    CUDA_OK(cudaPointerGetAttributes(&attr, base));
                    if (attr.type != cudaMemoryTypeDevice || attr.device != device) {
                        gram_die("the input of %s/%s is not in device %d's memory", g.key.folder.c_str(),
                            g.key.group.c_str(), device);
                    }
                    a = (const float *) base;
                    lda = (int) (src1->nb[1] / sizeof(float));
                }
                const float beta = first ? 0.0f : 1.0f;
                // C (n x n, lower) = A A^T + beta C, A the n x rows column-major view of the input rows
                CUBLAS_OK(cublasSsyrk(cublas, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, (int) n, (int) rows, &one, a,
                                      lda, &beta, c, (int) n));
                first = false;
            }
        }
        if (first) return;
        CUDA_OK(cudaMemcpy(h_c, c, (size_t) n * (size_t) n * sizeof(float), cudaMemcpyDeviceToHost));
        // column j of the lower triangle, rows j..n, is row j of the upper triangle by symmetry
        float * tri = g.tri.data();
        const float * hc = h_c;
        auto work = [n, tri, hc](int64_t r0, int64_t r1) {
            for (int64_t i = r0; i < r1; ++i) {
                const size_t off = (size_t) i * (size_t) n - (size_t) i * (size_t) (i - 1) / 2;
                const float * col = hc + (size_t) i * (size_t) n + (size_t) i;
                float * row = tri + off;
                const int64_t len = n - i;
                for (int64_t j = 0; j < len; ++j) row[j] += col[j];
            }
        };
        // rows shrink with i: split by equal area, not equal count
        std::vector<int64_t> cuts{ 0 };
        const double total = (double) n * (double) (n + 1) / 2;
        for (unsigned t = 1; t < threads; ++t) {
            const double target = total * t / threads;
            // area of rows [0, r) = r n - r (r - 1) / 2; solve for r
            const double b = 2.0 * n + 1.0;
            const double r = (b - std::sqrt(b * b - 8.0 * target)) / 2.0;
            cuts.push_back(std::clamp<int64_t>((int64_t) r, cuts.back(), n));
        }
        cuts.push_back(n);
        std::vector<std::thread> pool;
        for (size_t t = 0; t + 1 < cuts.size(); ++t) pool.emplace_back(work, cuts[t], cuts[t + 1]);
        for (auto & th : pool) th.join();
    }

    bool callback(ggml_tensor * t, bool ask) {
        if (t->op == GGML_OP_MUL_MAT_ID) {
            if (ask) return true;
            gram_die("%s: an indirect (expert) matmul; this tool gathers dense projections only",
                weight_name(t->src[0]->name).c_str());
        }
        if (t->op != GGML_OP_MUL_MAT) return false;
        const ggml_tensor * src0 = t->src[0];
        const ggml_tensor * src1 = t->src[1];
        const std::string w = weight_name(src0->name);
        group_key key;
        if (!group_of(w, key)) return false;
        if (ask) return true;

        std::lock_guard<std::mutex> lock(mutex);
        if (src1->type != GGML_TYPE_F32) gram_die("%s reads a %s input, not f32", w.c_str(), ggml_type_name(src1->type));
        if (src1->nb[0] != sizeof(float)) gram_die("%s reads an input whose rows are not contiguous", w.c_str());
        const int64_t n = src1->ne[0];
        if (src0->ne[0] != n) gram_die("%s is %" PRId64 " wide and its input %" PRId64, w.c_str(), src0->ne[0], n);
        const uint64_t rows = (uint64_t) (src1->ne[1] * src1->ne[2] * src1->ne[3]);
        const uint64_t r = round.load();

        const std::string gname = key.folder + "/" + key.group;
        auto [git, gnew] = groups.try_emplace(gname);
        group & g = git->second;
        if (gnew) {
            g.key = key;
            g.n = n;
            g.tri.assign((size_t) n * (size_t) (n + 1) / 2, 0.0f);
        } else if (g.n != n) {
            gram_die("%s reads an input %" PRId64 " wide; its group %s is %" PRId64, w.c_str(), n, gname.c_str(), g.n);
        }

        auto [mit, mnew] = members.try_emplace(w);
        member & m = mit->second;
        if (mnew) {
            m.name = w;
            m.ne0 = src0->ne[0];
            m.ne1 = src0->ne[1];
            m.type = src0->type;
            g.members.push_back(w);
        }
        if (m.round == r) gram_die("%s was evaluated twice in one ubatch", w.c_str());
        m.round = r;
        m.tokens += rows;

        if (g.round != r) {
            g.round = r;
            g.src1 = src1;
            g.data = src1->data;
            g.tokens += rows;
            accumulate(g, src1);
        } else if (g.src1 != src1 || g.data != src1->data) {
            gram_die("%s does not read the input the rest of %s read in this ubatch", w.c_str(), gname.c_str());
        }
        return true;
    }
};

static collector g_collector;

static bool eval_callback(ggml_tensor * t, bool ask, void *) {
    return g_collector.callback(t, ask);
}

static std::vector<llama_token> read_tokens(const std::string & path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) gram_die("%s: cannot open", path.c_str());
    const std::streamsize size = f.tellg();
    if (size <= 0 || size % 4 != 0) gram_die("%s: not a whole number of 4-byte token ids", path.c_str());
    std::vector<llama_token> out(size / 4);
    f.seekg(0);
    std::vector<unsigned char> b(size);
    if (!f.read((char *) b.data(), size)) gram_die("%s: short read", path.c_str());
    for (size_t i = 0; i < out.size(); ++i) {
        const uint32_t v = (uint32_t) b[4 * i] | ((uint32_t) b[4 * i + 1] << 8) | ((uint32_t) b[4 * i + 2] << 16) |
                           ((uint32_t) b[4 * i + 3] << 24);
        out[i] = (llama_token) v;
    }
    return out;
}

static std::string json_escape(const std::string & s) {
    std::string o;
    for (unsigned char c : s) {
        switch (c) {
            case '"': o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n"; break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    snprintf(buf, sizeof buf, "\\u%04x", c);
                    o += buf;
                } else {
                    o += (char) c;
                }
        }
    }
    return o;
}

static void usage(const char * argv0) {
    fprintf(stderr,
            "usage: %s -m MODEL.gguf --sequences LIST -o OUT [--ubatch N] [--device N]\n"
            "  LIST holds one path per line, each a sequence of little-endian 4-byte token ids.\n"
            "  OUT must not exist; gram.json is written last.\n",
            argv0);
}

int main(int argc, char ** argv) {
    std::string model_path, list_path, out_path;
    int ubatch = 2048;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                usage(argv[0]);
                gram_die("%s needs a value", a.c_str());
            }
            return argv[++i];
        };
        if (a == "-m" || a == "--model") model_path = next();
        else if (a == "--sequences") list_path = next();
        else if (a == "-o" || a == "--out") out_path = next();
        else if (a == "--ubatch") ubatch = std::atoi(next().c_str());
        else if (a == "--device") g_collector.device = std::atoi(next().c_str());
        else if (a == "-h" || a == "--help") {
            usage(argv[0]);
            return 0;
        } else {
            usage(argv[0]);
            gram_die("unknown argument %s", a.c_str());
        }
    }
    if (model_path.empty() || list_path.empty() || out_path.empty()) {
        usage(argv[0]);
        return 1;
    }
    if (ubatch < 1) gram_die("--ubatch must be positive");
    if (fs::exists(out_path)) gram_die("%s exists: the statistics are written into a new directory", out_path.c_str());

    std::vector<std::string>              paths;
    std::vector<std::vector<llama_token>> seqs;
    {
        std::ifstream f(list_path);
        if (!f) gram_die("%s: cannot open", list_path.c_str());
        std::string line;
        while (std::getline(f, line)) {
            if (line.empty()) continue;
            paths.push_back(line);
            seqs.push_back(read_tokens(line));
        }
    }
    if (seqs.empty()) gram_die("%s names no sequence", list_path.c_str());
    uint64_t total_tokens = 0;
    size_t   longest = 0;
    for (auto & s : seqs) {
        total_tokens += s.size();
        longest = std::max(longest, s.size());
    }

    // the byte range of every tensor in the model file, for the record
    gguf_init_params gp{ /*no_alloc*/ true, /*ctx*/ nullptr };
    gguf_context * gctx = gguf_init_from_file(model_path.c_str(), gp);
    if (!gctx) gram_die("%s: not a GGUF file", model_path.c_str());
    if (gguf_find_key(gctx, "split.count") >= 0) gram_die("%s is one shard of a split model; give a whole file", model_path.c_str());

    g_collector.init();
    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 999;
    mp.main_gpu = g_collector.device;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) gram_die("%s: the model did not load", model_path.c_str());
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    for (size_t s = 0; s < seqs.size(); ++s) {
        for (llama_token t : seqs[s]) {
            if (t < 0 || t >= n_vocab) gram_die("%s: token %d is outside the vocabulary of %d", paths[s].c_str(), t, n_vocab);
        }
    }

    const uint32_t n_seq_max = 64;
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = (uint32_t) std::max<size_t>((size_t) ubatch, longest);
    cp.n_batch = (uint32_t) ubatch;
    cp.n_ubatch = (uint32_t) ubatch;
    cp.n_seq_max = n_seq_max;
    cp.kv_unified = true;
    cp.swa_full = true;
    cp.cb_eval = eval_callback;
    cp.cb_eval_user_data = nullptr;
    cp.no_perf = true;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) gram_die("the context did not initialize");

    common_batch batch(ctx);
    const auto t0 = std::chrono::steady_clock::now();
    uint64_t rounds = 0;
    auto decode = [&]() {
        g_collector.round.store(rounds);
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) gram_die("decode failed at ubatch %" PRIu64, rounds);
        llama_synchronize(ctx);
        rounds++;
    };

    size_t s = 0;
    uint64_t done = 0;
    while (s < seqs.size()) {
        llama_memory_clear(llama_get_memory(ctx), true);
        batch.clear();
        if (seqs[s].size() > (size_t) ubatch) {
            // one long sequence alone, in ubatch-sized pieces over its own cache
            const auto & q = seqs[s];
            for (size_t p = 0; p < q.size(); p += ubatch) {
                batch.clear();
                const size_t e = std::min(q.size(), p + (size_t) ubatch);
                for (size_t k = p; k < e; ++k) batch.add(q[k], (llama_pos) k, 0, true);
                decode();
            }
            done += q.size();
            s++;
        } else {
            // whole sequences packed into one ubatch, each its own sequence id
            size_t used = 0;
            llama_seq_id id = 0;
            while (s < seqs.size() && seqs[s].size() <= (size_t) ubatch && used + seqs[s].size() <= (size_t) ubatch &&
                   (uint32_t) id < n_seq_max) {
                const auto & q = seqs[s];
                for (size_t k = 0; k < q.size(); ++k) batch.add(q[k], (llama_pos) k, id, true);
                used += q.size();
                done += q.size();
                id++;
                s++;
            }
            decode();
        }
        const double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        fprintf(stderr, "llama-gram: %zu/%zu sequences, %" PRIu64 "/%" PRIu64 " tokens, %" PRIu64 " ubatches, %.1f s\n",
                s, seqs.size(), done, total_tokens, rounds, el);
    }

    // every group saw every token, and every member every token
    for (auto & [name, g] : g_collector.groups) {
        if (g.tokens != total_tokens) {
            gram_die("%s saw %" PRIu64 " of the %" PRIu64 " tokens", name.c_str(), g.tokens, total_tokens);
        }
        for (auto & w : g.members) {
            if (g_collector.members[w].tokens != total_tokens) {
                gram_die("%s saw %" PRIu64 " of the %" PRIu64 " tokens", w.c_str(), g_collector.members[w].tokens, total_tokens);
            }
        }
        for (float v : g.tri) {
            if (!std::isfinite(v)) gram_die("%s holds a non-finite entry", name.c_str());
        }
    }
    if (g_collector.groups.empty()) gram_die("no projection was evaluated");

    fs::create_directories(out_path);
    std::string groups_json;
    for (auto & [name, g] : g_collector.groups) {
        const fs::path dir = fs::path(out_path) / g.key.folder;
        fs::create_directories(dir);
        const std::string file = g.key.group + ".hessian";
        {
            std::ofstream f(dir / file, std::ios::binary);
            f.write((const char *) g.tri.data(), (std::streamsize) (g.tri.size() * sizeof(float)));
            if (!f) gram_die("%s: write failed", (dir / file).c_str());
        }
        std::string members_json;
        for (auto & w : g.members) {
            const member & m = g_collector.members[w];
            const int64_t id = gguf_find_tensor(gctx, w.c_str());
            if (id < 0) gram_die("%s is not a tensor of %s", w.c_str(), model_path.c_str());
            const size_t off = gguf_get_data_offset(gctx) + gguf_get_tensor_offset(gctx, id);
            const size_t len = gguf_get_tensor_size(gctx, id);
            char buf[512];
            snprintf(buf, sizeof buf,
                     "{\"name\":\"%s\",\"type\":\"%s\",\"ne\":[%" PRId64 ",%" PRId64 "],\"tokens\":%" PRIu64
                     ",\"offset\":%zu,\"bytes\":%zu}",
                     json_escape(w).c_str(), ggml_type_name(m.type), m.ne0, m.ne1, m.tokens, off, len);
            members_json += (members_json.empty() ? "" : ",") + std::string(buf);
        }
        char buf[512];
        snprintf(buf, sizeof buf,
                 "{\"folder\":\"%s\",\"group\":\"%s\",\"n\":%" PRId64 ",\"tokens\":%" PRIu64
                 ",\"file\":\"%s\",\"bytes\":%zu,\"members\":[",
                 g.key.folder.c_str(), g.key.group.c_str(), g.n, g.tokens, file.c_str(), g.tri.size() * sizeof(float));
        groups_json += (groups_json.empty() ? "" : ",") + std::string(buf) + members_json + "]}";
    }
    std::string seq_json;
    for (auto & p : paths) seq_json += (seq_json.empty() ? "\"" : ",\"") + json_escape(p) + "\"";
    const std::string tmp = (fs::path(out_path) / "gram.json.partial").string();
    {
        std::ofstream f(tmp);
        f << "{\"schema\":\"llama-gram.v1\",\"build\":{\"number\":" << llama_build_number() << ",\"commit\":\""
          << json_escape(llama_commit()) << "\"},\"model\":\"" << json_escape(model_path) << "\",\"ubatch\":" << ubatch
          << ",\"ubatches\":" << rounds << ",\"tokens\":" << total_tokens << ",\"sequences\":[" << seq_json
          << "],\"accumulator\":\"binary32 cuBLAS SSYRK per ubatch, pedantic math, summed in binary32 on the host\""
          << ",\"groups\":[" << groups_json << "]}\n";
        if (!f) gram_die("%s: write failed", tmp.c_str());
    }
    fs::rename(tmp, fs::path(out_path) / "gram.json");

    gguf_free(gctx);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    fprintf(stderr, "llama-gram: %zu groups over %" PRIu64 " tokens in %s\n", g_collector.groups.size(), total_tokens,
            out_path.c_str());
    return 0;
}
