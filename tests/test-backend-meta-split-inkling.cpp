// The tensor split of Inkling 2 (-sm tensor): every weight, cache and convolution state of the architecture has a
// decision in llama_meta_device_get_split_state, held here against what each tensor holds. A model of Inkling's real
// attention shape (48 query and 8 KV heads of 128, n_embd 4608, five sliding layers to a global one, kernel 4) is
// created over the meta backend with no data, with its caches, and every tensor of the model and of its memory is
// looked up by name: a name the table below does not know fails, and so does a tensor split another way than its
// heads are, or mirrored where it should be split. The KV caches and the convolution state are split by KV heads.
//
// The split is over two and four ranks evenly and over two at 3:7. Each layer's heads are those its attn_k is split
// into, and a query head goes with its KV head: a tensor per query head has n_head/n_head_kv of them a KV head.

#include "test-backend-meta-split-inkling.h"

#include "llama-archs-gguf.h"
#include "llama-cpp.h"
#include "../src/llama-kv-cache-iswa.h"
#include "../src/llama-kv-cache.h"
#include "../src/llama-memory-hybrid-iswa.h"
#include "../src/llama-memory-recurrent.h"
#include "../src/llama-model.h"

#include <algorithm>
#include <cstdio>
#include <numeric>
#include <regex>
#include <string>
#include <vector>

namespace {

constexpr uint32_t n_embd    = 4608;
constexpr uint32_t n_head    = 48;
constexpr uint32_t n_head_kv = 8;
constexpr uint32_t head_dim  = 128;
constexpr uint32_t n_layer   = 6;
constexpr uint32_t n_ff_exp  = 2304;

gguf_context_ptr inkling_gguf() {
    gguf_context_ptr gguf = get_gguf_ctx(LLM_ARCH_INKLING2, true);
    kv_writer ms(LLM_ARCH_INKLING2, gguf.get());
    ms.add_kv(LLM_KV_EMBEDDING_LENGTH,           n_embd);
    ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT,       n_head);
    ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV,    n_head_kv);
    ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH,       head_dim);
    ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH,     head_dim);
    ms.add_kv(LLM_KV_BLOCK_COUNT,                n_layer);
    ms.add_kv(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, n_ff_exp);
    ms.add_kv(LLM_KV_SHORTCONV_L_CACHE,          uint32_t(4)); // the fixture's shared tail sets 3
    std::vector<uint8_t> swa(n_layer);
    for (uint32_t il = 0; il < n_layer; ++il) {
        swa[il] = il % 6 != 5;
    }
    gguf_set_arr_data(gguf.get(), llama_arch_kv_name(LLM_ARCH_INKLING2, LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN).c_str(),
            GGUF_TYPE_BOOL, swa.data(), swa.size());
    return gguf;
}

void no_data(ggml_tensor *, void *) {}

// what a tensor must be split into: axis MIRRORED, or an axis and the extent of every segment on every rank
struct expected {
    int axis;
    std::vector<std::vector<int64_t>> segments; // [segment][rank]; empty: only that the ranks' slices cover the axis
};

std::vector<int64_t> times(const std::vector<int64_t> & heads, int64_t per_head) {
    std::vector<int64_t> ret;
    for (int64_t h : heads) {
        ret.push_back(h*per_head);
    }
    return ret;
}

// the decision a tensor of Inkling must have, given the layer's KV heads and query heads on each rank
bool expect(const std::string & name, const llama_hparams & hp, const std::vector<int64_t> & kv, const std::vector<int64_t> & q,
        expected & out) {
    const int64_t kernel = hp.n_shortconv_l_cache;
    const int64_t n_rel  = hp.n_relative;
    static const std::regex mirrored("blk\\.\\d+\\.(attn_norm|attn_(q|k|v)_inp_norm|attn_res_query|ffn_res_query|ffn_norm|"
            "ffn_gate_inp|attn_rel_bias)\\.weight|blk\\.\\d+\\.exp_probs_b\\.bias|"
            "(token_embd|token_embd_norm|output_norm|output_res_query)\\.weight|"
            "blk\\.\\d+\\.[a-z_]+\\.(scale|input_scale|k_scale|v_scale)"); // one value a tensor or an expert
    static const std::regex q_heads ("blk\\.\\d+\\.(attn_q|attn_gate)\\.weight");
    static const std::regex kv_heads("blk\\.\\d+\\.(attn_k|attn_v|attn_k_conv|attn_v_conv)\\.weight|cache_(k|v)_l\\d+");
    static const std::regex per_head("blk\\.\\d+\\.attn_(q_clip|rel_clip|sink_proj)\\.weight");
    const int64_t channels = n_embd/n_head_kv; // of a hidden stream, a KV head's share
    if (std::regex_match(name, mirrored) || std::regex_match(name, std::regex("cache_s_l\\d+"))) {
        out = {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {}};
    } else if (std::regex_match(name, q_heads)) {
        out = {1, {times(q, head_dim)}};
    } else if (name.find("cache_") == 0 && std::regex_match(name, kv_heads)) {
        out = {0, {times(kv, head_dim)}};
    } else if (std::regex_match(name, kv_heads)) {
        out = {1, {times(kv, head_dim)}};
    } else if (std::regex_match(name, std::regex("blk\\.\\d+\\.attn_output\\.weight"))) {
        out = {0, {times(q, head_dim)}};
    } else if (std::regex_match(name, std::regex("blk\\.\\d+\\.attn_rel\\.weight"))) {
        out = {1, {times(q, n_rel)}};
    } else if (std::regex_match(name, per_head)) {
        out = {1, {q}};
    } else if (std::regex_match(name, std::regex("blk\\.\\d+\\.attn_(k|v)_conv_proj\\.weight"))) {
        out = {1, {times(kv, kernel)}};
    } else if (std::regex_match(name, std::regex("blk\\.\\d+\\.(attn|ffn)_post_conv\\.weight"))) {
        out = {1, {times(kv, channels)}};
    } else if (std::regex_match(name, std::regex("cache_r_l\\d+"))) {
        // the state's streams: K, V, the attention's and the feed-forward's hidden stream, kernel - 1 columns each
        out = {0, {times(kv, (kernel - 1)*head_dim), times(kv, (kernel - 1)*head_dim),
                   times(kv, (kernel - 1)*channels), times(kv, (kernel - 1)*channels)}};
    } else if (std::regex_match(name, std::regex("blk\\.\\d+\\.ffn_gate_up_exps\\.weight|output\\.weight"))) {
        out = {1, {}};
    } else if (std::regex_match(name, std::regex("blk\\.\\d+\\.ffn_down_exps\\.weight"))) {
        out = {0, {}};
    } else {
        return false;
    }
    return true;
}

std::string describe(const ggml_backend_meta_split_state & s, size_t n) {
    if (s.axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
        return "mirrored";
    }
    std::string ret = "axis " + std::to_string(int(s.axis)) + " [";
    for (uint32_t seg = 0; seg < s.n_segments; ++seg) {
        for (size_t j = 0; j < n; ++j) {
            ret += (seg || j ? (j ? "," : " | ") : "") + std::to_string(s.ne[seg*n + j]);
        }
    }
    return ret + "]";
}

bool check(const ggml_tensor * t, const llama_meta_device_get_split_state_userdata & ud,
        const std::vector<int64_t> & kv, const std::vector<int64_t> & q, int & n_checked) {
    auto ud_copy = ud;
    const ggml_backend_meta_split_state s = llama_meta_device_get_split_state(t, &ud_copy);
    const size_t n = ud.n_devices;
    expected want;
    bool ok = expect(t->name, ud.model->hparams, kv, q, want);
    if (ok && std::regex_match(t->name, std::regex("cache_s_l\\d+"))) {
        ok = ggml_nelements(t) == 0; // the recurrent memory's S state, which Inkling has none of
    }
    if (ok && want.axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
        ok = s.axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED;
    } else if (ok) {
        ok = s.axis == want.axis;
        int64_t covered = 0;
        for (uint32_t seg = 0; ok && seg < s.n_segments; ++seg) {
            for (size_t j = 0; j < n; ++j) {
                covered += s.nr[seg]*s.ne[seg*n + j];
            }
        }
        ok = ok && covered == t->ne[want.axis];
        if (ok && !want.segments.empty()) {
            ok = s.n_segments == want.segments.size();
            for (uint32_t seg = 0; ok && seg < s.n_segments; ++seg) {
                ok = s.nr[seg] == 1;
                for (size_t j = 0; ok && j < n; ++j) {
                    ok = s.ne[seg*n + j] == want.segments[seg][j];
                }
            }
        }
        // a split tensor must give every rank of a split layer something of it
        for (size_t j = 0; ok && j < n; ++j) {
            int64_t mine = 0;
            for (uint32_t seg = 0; seg < s.n_segments; ++seg) {
                mine += s.ne[seg*n + j];
            }
            ok = mine > 0;
        }
    }
    n_checked++;
    if (!ok) {
        printf("FAIL inkling2 %zu ranks: %s is %s\n", n, t->name, describe(s, n).c_str());
    }
    return ok;
}

bool test_split(ggml_backend_dev_t gpu, size_t n_ranks, const std::vector<float> & tensor_split) {
    auto gguf = inkling_gguf();
    std::vector<ggml_backend_dev_t> devs(n_ranks, gpu);
    devs.push_back(nullptr);
    llama_model_params mparams = llama_model_default_params();
    mparams.devices = devs.data();
    mparams.split_mode = LLAMA_SPLIT_MODE_TENSOR;
    mparams.tensor_split = tensor_split.empty() ? nullptr : tensor_split.data();
    mparams.no_alloc = true;
    llama_model_ptr model(llama_model_init_from_user(gguf.get(), no_data, nullptr, mparams));
    if (!model) {
        printf("FAIL inkling2 %zu ranks: the model was not created\n", n_ranks);
        return false;
    }
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 1024;
    cparams.n_seq_max = 1;
    llama_context_ptr ctx(llama_init_from_model(model.get(), cparams));
    if (!ctx) {
        printf("FAIL inkling2 %zu ranks: the context was not created\n", n_ranks);
        return false;
    }
    const llama_meta_device_get_split_state_userdata & ud = model->get_split_state_ud;
    GGML_ASSERT(ud.n_devices == n_ranks);

    // each layer's KV heads on each rank, as its attn_k is split, and the query heads that go with them
    std::vector<std::vector<int64_t>> kv(n_layer), q(n_layer);
    bool ok = true;
    for (uint32_t il = 0; il < n_layer; ++il) {
        auto ud_copy = ud;
        const ggml_tensor * k  = model->get_tensor(("blk." + std::to_string(il) + ".attn_k.weight").c_str());
        const ggml_tensor * wq = model->get_tensor(("blk." + std::to_string(il) + ".attn_q.weight").c_str());
        const ggml_backend_meta_split_state sk = llama_meta_device_get_split_state(k,  &ud_copy);
        const ggml_backend_meta_split_state sq = llama_meta_device_get_split_state(wq, &ud_copy);
        for (size_t j = 0; j < n_ranks; ++j) {
            kv[il].push_back(sk.ne[j]/head_dim);
            q[il].push_back(kv[il][j]*(n_head/n_head_kv));
            // whole heads, and the query heads of a rank are those of its KV heads
            ok = ok && sk.ne[j] % head_dim == 0 && sq.ne[j] == q[il][j]*head_dim;
            if (tensor_split.empty()) {
                ok = ok && kv[il][j] == n_head_kv/int64_t(n_ranks);
            }
        }
        ok = ok && std::accumulate(kv[il].begin(), kv[il].end(), int64_t(0)) == n_head_kv;
        if (!ok) {
            printf("FAIL inkling2 %zu ranks: layer %u heads are %s and %s\n", n_ranks, il,
                    describe(sk, n_ranks).c_str(), describe(sq, n_ranks).c_str());
            return false;
        }
    }
    const auto layer_of = [](const std::string & name) {
        std::smatch m;
        static const std::regex layer("blk\\.(\\d+)\\..*|cache_[a-z]+_l(\\d+)");
        if (!std::regex_match(name, m, layer)) {
            printf("FAIL inkling2: no layer in the name %s\n", name.c_str());
            exit(1);
        }
        return std::stoul(m[1].matched ? m[1].str() : m[2].str());
    };

    int n_weights = 0;
    for (const auto & [name, t] : model->tensors_by_name) {
        const bool per_layer = name.rfind("blk.", 0) == 0;
        const uint32_t il = per_layer ? layer_of(name) : 0;
        ok = check(t, ud, kv[il], q[il], n_weights) && ok;
    }

    auto * mem = dynamic_cast<llama_memory_hybrid_iswa *>(llama_get_memory(ctx.get()));
    GGML_ASSERT(mem != nullptr);
    int n_caches = 0;
    std::vector<std::string> kinds;
    const auto check_bufs = [&](const llama_memory_bufs & bufs) {
        for (const auto & [buf_ctx, buf] : bufs) {
            for (ggml_tensor * t = ggml_get_first_tensor(buf_ctx.get()); t; t = ggml_get_next_tensor(buf_ctx.get(), t)) {
                if (t->view_src != nullptr) {
                    continue; // a stream's view of a cache tensor, which the meta backend slices as the cache is split
                }
                ok = check(t, ud, kv[layer_of(t->name)], q[layer_of(t->name)], n_caches) && ok;
                kinds.push_back(std::string(t->name).substr(0, 7));
            }
        }
    };
    check_bufs(mem->get_mem_attn()->get_base()->get_bufs());
    check_bufs(mem->get_mem_attn()->get_swa()->get_bufs());
    check_bufs(mem->get_mem_recr()->get_bufs());
    // the global and sliding caches have K and V for every layer of their kind, and the state one tensor a layer
    for (const char * kind : {"cache_k", "cache_v", "cache_r"}) {
        const auto n = std::count(kinds.begin(), kinds.end(), kind);
        if (n < int(n_layer)) {
            printf("FAIL inkling2 %zu ranks: %ld %s tensors for %u layers\n", n_ranks, long(n), kind, n_layer);
            ok = false;
        }
    }

    std::string split = "even";
    if (!tensor_split.empty()) {
        split = std::to_string(int(tensor_split[0])) + ":" + std::to_string(int(tensor_split[1]));
    }
    printf("%s inkling2 tensor split, %zu ranks %s: %d weights and %d caches as their heads are split\n",
            ok ? "ok  " : "FAIL", n_ranks, split.c_str(), n_weights, n_caches);
    return ok;
}

} // namespace

bool test_inkling_split_decisions(ggml_backend_dev_t gpu) {
    llama_backend_init();
    bool ok = test_split(gpu, 2, {});
    ok = test_split(gpu, 4, {}) && ok;
    ok = test_split(gpu, 2, {3.0f, 7.0f}) && ok;
    return ok;
}
