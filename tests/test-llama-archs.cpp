#include "common.h"
#include "log.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"
#include "ggml-cpp.h"
#include "llama.h"
#include "llama-cpp.h"
#include "../src/llama-ext.h"

#include "../src/llama-model.h"
#include "../src/llama-model-saver.h"
#include "../src/llama-memory-hybrid.h"
#include "../ggml/src/ggml-impl.h"

#include <cstdlib>

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <numeric>
#include <random>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

static bool arch_matches(const std::string & filter, llm_arch arch) {
    if (filter.empty()) {
        return true;
    }
    return std::regex_search(llama_arch_name(arch), std::regex(filter));
}

// normalized mean squared error = mse(a, b) / mse(a, 0)
static double nmse(const std::vector<float> & a, const std::vector<float> & b) {
    GGML_ASSERT(a.size() == b.size());
    double mse_a_b = 0.0;
    double mse_a_0 = 0.0;

    for (size_t i = 0; i < a.size(); i++) {
        float a_i = a[i];
        float b_i = b[i];

        mse_a_b += (a_i - b_i) * (a_i - b_i);
        mse_a_0 += a_i * a_i;
    }

    return mse_a_b / mse_a_0;
}

// a row's largest logit magnitude
static float max_abs(const float * row, size_t n) {
    float m = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        m = std::max(m, std::fabs(row[i]));
    }
    return m;
}

// whether every device of a configuration is a CPU: where the width of a batch does not change a token's arithmetic
static bool on_host(const std::vector<ggml_backend_dev_t> & devs) {
    return std::all_of(devs.begin(), devs.end(), [](ggml_backend_dev_t dev) {
        return ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU;
    });
}

// rows of n_vocab logits: the mean over rows of KL(softmax(a) || softmax(b)), the rows whose top token differs, and
// apart from those the rows where the two top tokens are tied: each read scores the other's top token within `tie` of
// its row's largest logit magnitude below its own. 1e-4 is a gap a change of summation order alone can close. `rows`,
// where given, takes the rows whose top token differs
static double mean_kld(const std::vector<float> & a, const std::vector<float> & b, size_t n_vocab, size_t & n_top_differ,
        size_t & n_top_tied, float tie = 1e-4f, std::vector<size_t> * rows = nullptr) {
    GGML_ASSERT(a.size() == b.size() && n_vocab > 0 && a.size() % n_vocab == 0);
    const size_t n_rows = a.size()/n_vocab;
    double sum = 0.0;
    n_top_differ = 0;
    n_top_tied   = 0;
    for (size_t r = 0; r < n_rows; ++r) {
        const float * ra = a.data() + r*n_vocab;
        const float * rb = b.data() + r*n_vocab;
        const size_t top_a = std::max_element(ra, ra + n_vocab) - ra;
        const size_t top_b = std::max_element(rb, rb + n_vocab) - rb;
        if (top_a != top_b) {
            const bool tied = ra[top_a] - ra[top_b] <= tie*max_abs(ra, n_vocab) &&
                              rb[top_b] - rb[top_a] <= tie*max_abs(rb, n_vocab);
            n_top_tied   += tied;
            n_top_differ += !tied;
            if (!tied && rows != nullptr) {
                rows->push_back(r);
            }
        }
        double z_a = 0.0, z_b = 0.0;
        for (size_t i = 0; i < n_vocab; ++i) {
            z_a += std::exp((double) ra[i] - ra[top_a]);
            z_b += std::exp((double) rb[i] - rb[top_b]);
        }
        double kld = 0.0;
        for (size_t i = 0; i < n_vocab; ++i) {
            const double la = (double) ra[i] - ra[top_a] - std::log(z_a);
            const double lb = (double) rb[i] - rb[top_b] - std::log(z_b);
            kld += std::exp(la)*(la - lb);
        }
        sum += kld;
    }
    return sum/n_rows;
}

// What a device is held to where it computes the same logits in two shapes (GLM prefill reuse, shared pool reuse): a
// mean KLD of 1e-6 and an NMSE of 1e-4, the bar this test holds a device's logits to against the CPU's. The rows whose
// top token differs are printed with the seed and the pairing, so that one can be traced, and they fail nothing.
// Two kernels of a card do not agree row by row on this fixture's MTP head, whose logits are 1e-6 to 1e-9 in size by
// the seed (set_tensor_data's weights are N(0, 0.01)). Seed 1846796939 on a card, largest logit 1.4e-9: the pool that
// sequences share and a cache a sequence were 3.5e-3 of a row's largest logit apart in the median row and 1.8e-2 at
// most, 544 of 960 rows over 1e-3, NMSE 2.1e-5, and two rows' top tokens moved. Seed 601529358, largest logit 6.1e-7:
// 7.2e-5 in the median row, NMSE 1.1e-8. The target's logits (4.7e-3) were 4e-6 apart for both. Traced for the first
// seed: the two contexts picked the same pools in every row, from scores bit for bit the same, and in the rows that
// differ neither is further from the CPU's logits than the other. So a top token that differs there is not a cell read
// wrongly, and which kernel parts the two is not known. The KLD sees none of it on logits this small (1e-17 with two
// top tokens moved). Cells read wrongly for more than a few rows are over the NMSE; a few rows are seen where the
// compare is byte for byte, on the host.
static bool glm_device_logits_agree(const char * check, const std::string & pairing, size_t seed,
        const std::vector<float> & a, const std::vector<float> & b, size_t n_vocab, float tie = 1e-4f) {
    size_t n_top_differ = 0;
    size_t n_top_tied   = 0;
    std::vector<size_t> rows;
    const double kld  = mean_kld(a, b, n_vocab, n_top_differ, n_top_tied, tie, &rows);
    const double dist = nmse(a, b);
    const bool   ok   = kld <= 1e-6 && dist <= 1e-4;
    if (!ok || n_top_differ > 0 || n_top_tied > 0) {
        std::string list;
        for (size_t i = 0; i < rows.size() && i < 16; ++i) {
            list += (i > 0 ? " " : "") + std::to_string(rows[i]);
        }
        if (rows.size() > 16) {
            list += " ...";
        }
        char line[512];
        snprintf(line, sizeof(line), "GLM %s: seed %zu: %s: mean KLD %g (at most 1e-6), nmse=%g (at most 1e-4), "
                "%zu rows' top token differs (rows %s), %zu tied rows' swapped", check, seed, pairing.c_str(), kld, dist,
                n_top_differ, list.empty() ? "none" : list.c_str(), n_top_tied);
        if (ok) {
            LOG_WRN("%s\n", line);
        } else {
            LOG_ERR("%s\n", line);
        }
    }
    return ok;
}

struct tensor_data_params {
    size_t seed;
    float  stdev;
};

// fills the values of one tensor, the name carries whatever the property needs
typedef void (*tensor_filler)(std::vector<float> & dst, const char * name, const tensor_data_params & params);

// a pool of normal draws, generated once and reused for every model
static std::vector<float> g_normal_pool;
static const size_t NORMAL_POOL_SIZE = 16 * 1024 * 1024;

// fill the pool once; each tensor copies a slice of it
static void init_normal_pool(const size_t seed, const float stdev) {
    g_normal_pool.resize(NORMAL_POOL_SIZE);
    std::mt19937 gen(seed);
    std::normal_distribution<float> dist(0.0f, stdev);
    for (float & v : g_normal_pool) {
        v = dist(gen);
    }
}

// a normal draw keyed on a name, so that tensors which share a draw ask for the same key
static void fill_normal(std::vector<float> & dst, const char * key, const tensor_data_params &) {
    const size_t n = dst.size();
    const size_t pool_size = g_normal_pool.size();
    const size_t start = std::hash<std::string>{}(key) % pool_size;

    // copy a cyclic slice of the pool, so a tensor larger than the pool still fills
    size_t i = 0;
    size_t pos = start;
    while (i < n) {
        const size_t chunk = std::min(pool_size - pos, n - i);
        std::memcpy(dst.data() + i, g_normal_pool.data() + pos, chunk * sizeof(float));
        i += chunk;
        pos = 0;
    }
}

// Mamba A must be negative (state decay)
static void fill_ssm_a(std::vector<float> & dst, const char * name, const tensor_data_params & params) {
    fill_normal(dst, name, params);
    for (float & v : dst) {
        v = -fabsf(v);
    }
}

// RoPE factors divide the frequencies, a value close to 0 gives angles too large for F32 sin/cos to stay accurate
static void fill_rope_factors(std::vector<float> & dst, const char * name, const tensor_data_params & params) {
    fill_normal(dst, name, params);
    for (float & v : dst) {
        v = 1.0f + fabsf(v);
    }
}

// FFN act scales divide the activation, a value close to 0 overflows the F16 staging of the Metal mul-mat kernel
static void fill_ffn_act(std::vector<float> & dst, const char * name, const tensor_data_params & params) {
    fill_normal(dst, name, params);
    for (float & v : dst) {
        v = copysignf(1.0f + fabsf(v), v);
    }
}

// a norm preserving residual gate holds alpha^2 + beta^2 == 1, and the conversion resolves both from one raw weight
// the alpha and the beta of a block share that weight, so the key drops the suffix and keeps the block
static void fill_resid_gate(std::vector<float> & dst, const char * name, const tensor_data_params & params) {
    const char * suffix = strstr(name, "_resid_");
    const bool is_beta = strstr(suffix, "_resid_beta") != nullptr;
    fill_normal(dst, std::string(name, suffix - name).c_str(), params);
    for (float & v : dst) {
        const float beta = 1.0f / (1.0f + expf(-v));
        v = is_beta ? beta : sqrtf(1.0f - beta*beta);
    }
}

// a random router bias can tie within the noise between the backends, a ramp keeps the top-k selection the same
static void fill_exp_probs_b(std::vector<float> & dst, const char *, const tensor_data_params &) {
    for (size_t i = 0; i < dst.size(); i++) {
        dst[i] = 2.0f*i;
    }
}

// a tensor whose values hold a property of their own, matched on a substring of its name, first match wins
static const struct { const char * match; tensor_filler fill; } TENSOR_FILLERS[] = {
    { "ssm_a",        fill_ssm_a        },
    { "_resid_",      fill_resid_gate   },
    { "exp_probs_b",  fill_exp_probs_b  },
    { "ffn.act",      fill_ffn_act      },
    { "rope_freqs",   fill_rope_factors },
    { "rope_factors", fill_rope_factors },
};

static void set_tensor_data(struct ggml_tensor * tensor, void * userdata) {
    const tensor_data_params & params = *(const tensor_data_params *) userdata;

    tensor_filler fill = fill_normal;
    for (const auto & filler : TENSOR_FILLERS) {
        if (strstr(tensor->name, filler.match) != nullptr) {
            fill = filler.fill;
            break;
        }
    }

    std::vector<float> tmp(ggml_nelements(tensor));
    fill(tmp, tensor->name, params);

    if (tensor->type == GGML_TYPE_F32) {
        ggml_backend_tensor_set(tensor, tmp.data(), 0, ggml_nbytes(tensor));
    } else if (tensor->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> tmp16(tmp.size());
        for (size_t i = 0; i < tmp.size(); i++) {
            tmp16[i] = ggml_fp32_to_fp16(tmp[i]);
        }
        ggml_backend_tensor_set(tensor, tmp16.data(), 0, ggml_nbytes(tensor));
    } else {
        GGML_ABORT("fatal error");
    }
}

static void usage(char ** argv) {
    LOG("Usage: %s [options]\n\n", argv[0]);
    LOG("Options:\n");
    LOG("  -a, --arch <arch|regex>  Run only matching LLM architectures (default: all supported)\n");
    LOG("  -s, --seed <seed>        Set the random seed for tensor initialization and token generation\n");
    LOG("  -d, --stdev <stdev>      Set the standard deviation of the tensor initialization distribution (default: 0.01f)\n");
    LOG("  -o, --out <dir>          Save generated test models to <dir> instead of running backend tests\n");
    LOG("  -v <N>                   Set log verbosity level\n");
    LOG("  -b, --backend <backend>  Run only on the given backend device\n");
    LOG("  --kv-shift               Check K-shift against a decode at the shifted positions (CPU only)\n");
    LOG("  -h, --help               Show this help message\n\n");
    LOG("Examples:\n");
    LOG("  %s\n", argv[0]);
    LOG("  %s -a qwen35moe\n", argv[0]);
    LOG("  %s -a deepseek4 -o tests/test-models/\n", argv[0]);
    LOG("  %s -a cohere2moe -v 5\n", argv[0]);
}

static std::vector<llama_token> get_tokens(const uint32_t n_tokens, const uint32_t n_vocab, const size_t seed){
    std::mt19937 gen(seed);
    std::uniform_int_distribution<> dis(0, n_vocab - 1);
    std::vector<llama_token> ret;
    ret.reserve(n_tokens);
    for (uint32_t i = 0; i < n_tokens; i++) {
        ret.push_back(dis(gen));
    }
    return ret;
}

// MoE archs that are also tested with the experts in host memory
static bool host_experts_test(const llm_arch arch) {
    switch (arch) {
        case LLM_ARCH_DEEPSEEK2:
            return true;
        default:
            return false;
    }
}

// writes arch-resolved GGUF keys into the fixture context, the test-side equivalent of llama_model_saver
struct kv_writer {
    struct gguf_context * gguf_ctx;
    const llm_arch        arch;

    kv_writer(enum llm_arch arch, struct gguf_context * gguf_ctx) : gguf_ctx(gguf_ctx), arch(arch) {}

    void add_kv(enum llm_kv key, uint32_t     value) { gguf_set_val_u32 (gguf_ctx, llama_arch_kv_name(arch, key).c_str(), value); }
    void add_kv(enum llm_kv key, int32_t      value) { gguf_set_val_i32 (gguf_ctx, llama_arch_kv_name(arch, key).c_str(), value); }
    void add_kv(enum llm_kv key, uint64_t     value) { gguf_set_val_u64 (gguf_ctx, llama_arch_kv_name(arch, key).c_str(), value); }
    void add_kv(enum llm_kv key, float        value) { gguf_set_val_f32 (gguf_ctx, llama_arch_kv_name(arch, key).c_str(), value); }
    void add_kv(enum llm_kv key, bool         value) { gguf_set_val_bool(gguf_ctx, llama_arch_kv_name(arch, key).c_str(), value); }
    void add_kv(enum llm_kv key, const char * value) { gguf_set_val_str (gguf_ctx, llama_arch_kv_name(arch, key).c_str(), value); }

    template <typename T>
    void add_kv(enum llm_kv key, const std::vector<T> & value) {
        if (value.empty()) {
            return;
        }
        if (std::is_same<T, uint32_t>::value) {
            gguf_set_arr_data(gguf_ctx, llama_arch_kv_name(arch, key).c_str(), GGUF_TYPE_UINT32, value.data(), value.size());
        } else if (std::is_same<T, int32_t>::value) {
            gguf_set_arr_data(gguf_ctx, llama_arch_kv_name(arch, key).c_str(), GGUF_TYPE_INT32, value.data(), value.size());
        } else if (std::is_same<T, uint64_t>::value) {
            gguf_set_arr_data(gguf_ctx, llama_arch_kv_name(arch, key).c_str(), GGUF_TYPE_UINT64, value.data(), value.size());
        } else if (std::is_same<T, float>::value) {
            gguf_set_arr_data(gguf_ctx, llama_arch_kv_name(arch, key).c_str(), GGUF_TYPE_FLOAT32, value.data(), value.size());
        } else {
            GGML_ABORT("fatal error");
        }
    }

    void add_kv(enum llm_kv key, const std::vector<std::string> & value) {
        std::vector<const char *> tmp(value.size());
        for (size_t i = 0; i < value.size(); ++i) {
            tmp[i] = value[i].c_str();
        }
        gguf_set_arr_str(gguf_ctx, llama_arch_kv_name(arch, key).c_str(), tmp.data(), tmp.size());
    }
};

static gguf_context_ptr get_gguf_ctx(const llm_arch arch, const bool moe, const bool mtp = false, const uint32_t n_layer_min = 0) {
    gguf_context_ptr ret(gguf_init_empty());
    kv_writer ms(arch, ret.get());
    const uint32_t n_ctx = 256;

    uint32_t n_vocab = 128;
    uint32_t n_embd  = 256;
    uint32_t n_head  = 2;
    uint32_t n_ff    = 384;
    uint32_t n_layer = 2;
    if (mtp && arch == LLM_ARCH_QWEN35) {
        n_layer = 3; // keep both recurrent and attention target blocks before NextN
    }
    if (arch == LLM_ARCH_LLAMA4) {
        n_layer = 4; // hparams.n_no_rope_layer_step is hard-coded to 4
    } else if (arch == LLM_ARCH_MAION_CODER) {
        n_head = 4;
        n_layer = 4; // dense and MoE layers, each with local and full attention
    } else if (arch == LLM_ARCH_GEMMA4) {
        // TODO: expand coverage to both GQA and MQA (see https://github.com/ggml-org/llama.cpp/pull/31401#pullrequestreview-5710165130)
        n_head = 4;
        n_layer = 10; // last 5 layers reuse KV, both SWA and full attention
    } else if (arch == LLM_ARCH_GEMMA3N) {
        n_embd = 64;
        n_head = 1;
        n_ff   = 96;
        n_layer = 22; // hparams.n_layer_kv_from_start = 20 is hardcoded
    } else if (arch == LLM_ARCH_LFM3) {
        n_layer = 3;
    } else if (arch == LLM_ARCH_DEEPSEEK4) {
        // head size 64 so that GPU flash attention kernels support the model
        n_embd  = 512;
        n_head  = 8;
        n_ff    = 1024;
        n_layer = 4;
    } else if (arch == LLM_ARCH_DEEPSEEK41) {
        // window-only, Engram, ratio-2 and ratio-1 sources, top-k reuse and a candidate-masked index layer
        n_embd  = 512;
        n_head  = 8;
        n_ff    = 1024;
        n_layer = 6;
    } else if (arch == LLM_ARCH_STEP35 || arch == LLM_ARCH_LAGUNA) {
        n_embd = 160; // exercise per-head tensor split granularity with head size 80
    } else if (arch == LLM_ARCH_QWEN3 || arch == LLM_ARCH_MUSE_GLIMMER || arch == LLM_ARCH_AFMOE || arch == LLM_ARCH_RICURSIVE) {
        n_head = 4;
    } else if (arch == LLM_ARCH_LIMITE) {
        n_head  = 4;
        n_layer = 4; // dense connections on the last two layers
    } else if (arch == LLM_ARCH_DEEPSEEK2
            || arch == LLM_ARCH_DEEPSEEK32
            || arch == LLM_ARCH_GLM_DSA
            || arch == LLM_ARCH_DOTS3NOTE
            || arch == LLM_ARCH_KIMI_LINEAR
            || arch == LLM_ARCH_BAILINGMOE3
            || arch == LLM_ARCH_KIMI_K3
            || arch == LLM_ARCH_KIMI_K31
            || arch == LLM_ARCH_MISTRAL4
            || arch == LLM_ARCH_MISTRAL_LARGE4
            || arch == LLM_ARCH_HY_V4
            || arch == LLM_ARCH_XING4_0) {
        n_embd = 128;
        n_head = 1;
        n_ff   = 192;
        if (arch == LLM_ARCH_MISTRAL_LARGE4) {
            n_layer = 4; // cover NoPE, shared indexer and KV sharing layers
        }
    } else if (arch == LLM_ARCH_GLM5_NEXT) {
        // several heads so that -sm tensor can split them and the MLA flash attention has GQA > 1
        n_embd = 128;
        n_head = 16;
        n_ff   = 192;
    } else if (arch == LLM_ARCH_GLM5NEXT) {
        n_embd = 128;
        n_head = 4; // split two ways, each device gets 2 KDA and 2 DSA heads (2 DSA heads are the fewest with 128-row q_b blocks)
        n_ff   = 192;
        n_layer = mtp ? 3 : 2;
    } else if (arch == LLM_ARCH_NEMOTRON_H || arch == LLM_ARCH_NEMOTRON_H_MOE) {
        n_layer = 3;
    } else if (arch == LLM_ARCH_METIS16) {
        n_layer = 4; // 2 passes over 2 physical layers, one attention and one Mamba-2
    } else if (arch == LLM_ARCH_CHAMELEON) {
        n_vocab = 10240;
    } else if (arch == LLM_ARCH_QWEN3TTS) {
        //n_vocab = 4096; // must be >= the hard-coded codec head size (3072)
        n_vocab = 3072; // TODO: should be 4096, but user code cannot get `n_vocab_out` yet [TAG_LLAMA_N_VOCAB_OUT]
    } else if (arch == LLM_ARCH_HRM_TEXT) {
        n_layer = 6; // 1 layer per stack x 2 h-cycles x (2 l-cycles + 1) cache slots
    }

    uint32_t n_head_kv = n_head;
    if (arch == LLM_ARCH_QWEN3) {
        n_head_kv = 1; // MQA coverage
    } else if (arch == LLM_ARCH_MUSE_GLIMMER || arch == LLM_ARCH_MUSE_SPARK || arch == LLM_ARCH_AFMOE || arch == LLM_ARCH_MAION_CODER ||
            arch == LLM_ARCH_KOLIBRI1 || arch == LLM_ARCH_LIMITE || arch == LLM_ARCH_GEMMA4 || arch == LLM_ARCH_RICURSIVE) {
        n_head_kv = 2; // GQA coverage
    }
    if (arch == LLM_ARCH_BEAM) {
        n_head    = 4;
        n_head_kv = 2;
        n_layer   = 4;
    }
    n_layer = std::max(n_layer, n_layer_min);
    const uint32_t n_embd_head = n_embd / n_head;

    ms.add_kv(LLM_KV_GENERAL_ARCHITECTURE,      llama_arch_name(arch));
    ms.add_kv(LLM_KV_VOCAB_SIZE,                n_vocab);
    ms.add_kv(LLM_KV_CONTEXT_LENGTH,            n_ctx);
    ms.add_kv(LLM_KV_EMBEDDING_LENGTH,          n_embd);
    ms.add_kv(LLM_KV_FEATURES_LENGTH,           n_embd);
    ms.add_kv(LLM_KV_BLOCK_COUNT,               n_layer);
    if (mtp) {
        ms.add_kv(LLM_KV_NEXTN_PREDICT_LAYERS, uint32_t(1));
    }
    ms.add_kv(LLM_KV_LEADING_DENSE_BLOCK_COUNT, uint32_t(1));

    if (arch == LLM_ARCH_K2_HORIZON) {
        ms.add_kv(LLM_KV_ROPE_SCALING_YARN_BETA_FAST, 128.0f);
        ms.add_kv(LLM_KV_ROPE_SCALING_YARN_BETA_SLOW,   4.0f);
    }

    if (arch == LLM_ARCH_NEMOTRON_H || arch == LLM_ARCH_NEMOTRON_H_MOE) {
        std::vector<uint32_t> n_ff_per_layer;
        n_ff_per_layer.reserve(n_layer);
        for (uint32_t il = 0; il < n_layer; il++) {
            n_ff_per_layer.push_back(il <= 1 ? 0 : n_ff);
        }
        ms.add_kv(LLM_KV_FEED_FORWARD_LENGTH, n_ff_per_layer);
    } else {
        ms.add_kv(LLM_KV_FEED_FORWARD_LENGTH, n_ff);
    }

    ms.add_kv(LLM_KV_USE_PARALLEL_RESIDUAL,   false);
    ms.add_kv(LLM_KV_LOGIT_SCALE,             1.0f);
    ms.add_kv(LLM_KV_TIME_MIX_EXTRA_DIM,      uint32_t(64));
    ms.add_kv(LLM_KV_TIME_DECAY_EXTRA_DIM,    uint32_t(128));
    ms.add_kv(LLM_KV_FULL_ATTENTION_INTERVAL, uint32_t(2));

    if (arch == LLM_ARCH_PLAMO2 || arch == LLM_ARCH_JAMBA || arch == LLM_ARCH_NEMOTRON_H || arch == LLM_ARCH_NEMOTRON_H_MOE ||
            arch == LLM_ARCH_GRANITE_HYBRID || arch == LLM_ARCH_LFM2 || arch == LLM_ARCH_LFM2MOE || arch == LLM_ARCH_LFM3 ||
            arch == LLM_ARCH_KIMI_LINEAR || arch == LLM_ARCH_BAILINGMOE3 || arch == LLM_ARCH_KIMI_K3 || arch == LLM_ARCH_KIMI_K31 ||
            arch == LLM_ARCH_GLM5_NEXT
            ) {
        GGML_ASSERT(n_layer >= 2);
        std::vector<uint32_t> n_head_per_layer;
        n_head_per_layer.reserve(n_layer);
        for (uint32_t il = 0; il < n_layer; il++) {
            n_head_per_layer.push_back(il == 1 ? 0 : n_head);
        }
        // GLM5 next KDA heads come from the uniform head count, only head_count_kv is per layer.
        if (arch == LLM_ARCH_GLM5_NEXT) {
            ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT, n_head);
            // the MLA cache holds one latent head
            for (uint32_t & n : n_head_per_layer) {
                n = std::min(n, uint32_t(1));
            }
        } else {
            ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT, n_head_per_layer);
        }
        ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV, n_head_per_layer);
    } else if (arch == LLM_ARCH_METIS16) {
        // the layer types repeat in every pass
        std::vector<uint32_t> n_head_kv_per_layer;
        n_head_kv_per_layer.reserve(n_layer);
        for (uint32_t il = 0; il < n_layer; il++) {
            n_head_kv_per_layer.push_back(il % 2 == 0 ? n_head_kv : 0);
        }
        ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT, n_head);
        ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV, n_head_kv_per_layer);
    } else if (arch == LLM_ARCH_GLM5NEXT) {
        GGML_ASSERT(n_layer >= 2);
        std::vector<uint32_t> n_head_kv_per_layer;
        n_head_kv_per_layer.reserve(n_layer);
        for (uint32_t il = 0; il < n_layer; il++) {
            n_head_kv_per_layer.push_back(il == 1 ? 0 : 1); // KDA, and DSA's one latent KV head as the served file has
        }
        ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT, n_head);
        ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV, n_head_kv_per_layer);
    } else {
        ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT, n_head);
        ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV, (arch == LLM_ARCH_DEEPSEEK4 || arch == LLM_ARCH_DEEPSEEK41) ? uint32_t(1) : n_head_kv);
    }

    ms.add_kv(LLM_KV_ATTENTION_MAX_ALIBI_BIAS, 8.0f);
    if (arch == LLM_ARCH_DEEPSEEK4 || arch == LLM_ARCH_DEEPSEEK41) {
        ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH,   n_embd_head);
        ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH, n_embd_head);
        ms.add_kv(LLM_KV_ROPE_DIMENSION_COUNT,   n_embd_head/2);
    } else if (arch == LLM_ARCH_DEEPSEEK2
            || arch == LLM_ARCH_DEEPSEEK32
            || arch == LLM_ARCH_GLM_DSA
            || arch == LLM_ARCH_DOTS3NOTE
            || arch == LLM_ARCH_KIMI_LINEAR
            || arch == LLM_ARCH_BAILINGMOE3
            || arch == LLM_ARCH_KIMI_K3
            || arch == LLM_ARCH_KIMI_K31
            || arch == LLM_ARCH_GLM5_NEXT
            || arch == LLM_ARCH_HY_V4
            || arch == LLM_ARCH_MISTRAL4
            || arch == LLM_ARCH_MISTRAL_LARGE4
            || arch == LLM_ARCH_XING4_0) {
        // GLM5 next MLA is nope only, the cache row is the compressed latent alone.
        ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH,       arch == LLM_ARCH_GLM5_NEXT ? uint32_t(512) : uint32_t(576));
        ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH,     uint32_t(512));
        ms.add_kv(LLM_KV_ROPE_DIMENSION_COUNT,       arch == LLM_ARCH_GLM5_NEXT ? uint32_t(0) : uint32_t(64));
        ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH_MLA,   uint32_t(192));
        ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH_MLA, uint32_t(128));
        if (arch == LLM_ARCH_DOTS3NOTE) {
            // SWA layers reuse the same MLA geometry as the full layers in this fixture
            ms.add_kv(LLM_KV_ATTENTION_KV_LORA_RANK_SWA,     uint32_t(512));
            ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH_SWA,       uint32_t(576));
            ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH_SWA,     uint32_t(512));
            ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH_MLA_SWA,   uint32_t(192));
            ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH_MLA_SWA, uint32_t(128));
            ms.add_kv(LLM_KV_ROPE_FREQ_BASE_SWA,             10000.0f);
            // indexer on the full-attention layers (inverse of the swa pattern)
            std::vector<uint32_t> indexer_types;
            indexer_types.reserve(n_layer);
            for (uint32_t il = 0; il < n_layer; il++) {
                indexer_types.push_back(il % 2 ? 0 : 1);
            }
            ms.add_kv(LLM_KV_ATTENTION_INDEXER_TYPES, indexer_types);
        }
    } else if (arch == LLM_ARCH_GLM5NEXT) {
        // nope-only MLA: the cache holds the bare latent, so n_rot must be an explicit 0
        ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH,       uint32_t(512));
        ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH,     uint32_t(512));
        ms.add_kv(LLM_KV_ROPE_DIMENSION_COUNT,       uint32_t(0));
        ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH_MLA,   uint32_t(192));
        ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH_MLA, uint32_t(128));
    } else if (arch == LLM_ARCH_MINIMAX_M3) {
        // partial rotary: n_rot must not exceed the indexer key length (64)
        ms.add_kv(LLM_KV_ROPE_DIMENSION_COUNT,       uint32_t(64));
    }
    if (arch == LLM_ARCH_SENSENOVA_U1) {
        ms.add_kv(LLM_KV_ROPE_FREQ_BASE_SPATIAL, 10000.0f);
    }

    if (arch == LLM_ARCH_XING4_0) {
        // Manifold-Constrained Hyper-Connections (MHC) multi-residual-stream blocks
        ms.add_kv(LLM_KV_HYPER_CONNECTION_COUNT,               uint32_t(4));
        ms.add_kv(LLM_KV_HYPER_CONNECTION_SINKHORN_ITERATIONS, uint32_t(20));
        ms.add_kv(LLM_KV_HYPER_CONNECTION_EPSILON,             1e-6f);
    }
    ms.add_kv(LLM_KV_ATTENTION_CLAMP_KQV,              1.0f);
    ms.add_kv(LLM_KV_ATTENTION_LAYERNORM_EPS,          arch == LLM_ARCH_GLM5NEXT ? 1e-6f : 1e-5f);
    ms.add_kv(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,      1e-5f);
    ms.add_kv(LLM_KV_ATTENTION_GROUPNORM_EPS,          1e-5f);
    ms.add_kv(LLM_KV_ATTENTION_GROUPNORM_GROUPS,       uint32_t(8));
    ms.add_kv(LLM_KV_ATTENTION_Q_LORA_RANK,            (arch == LLM_ARCH_DEEPSEEK4 || arch == LLM_ARCH_DEEPSEEK41) ? uint32_t(64) : uint32_t(512));
    ms.add_kv(LLM_KV_ATTENTION_KV_LORA_RANK,           uint32_t(512));
    ms.add_kv(LLM_KV_ATTENTION_RELATIVE_BUCKETS_COUNT, uint32_t(8));
    ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW,         n_ctx/8);

    if (arch == LLM_ARCH_MAION_CODER) {
        ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, std::vector<uint32_t>{1, 1, 0, 0});
    } else if (arch == LLM_ARCH_GEMMA4) {
        ms.add_kv(LLM_KV_EMBEDDING_LENGTH_PER_LAYER,      n_embd/2);
        ms.add_kv(LLM_KV_ATTENTION_SHARED_KV_LAYERS,      uint32_t(5));
        ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH_SWA,        n_embd_head);
        ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH_SWA,      n_embd_head);
        ms.add_kv(LLM_KV_ROPE_FREQ_BASE_SWA,              10000.0f);
        // SWA pattern: every 5th layer is full attention (matches E2B layer_types)
        std::vector<uint32_t> pattern;
        pattern.reserve(n_layer);
        for (uint32_t il = 0; il < n_layer; il++) {
            pattern.push_back((il + 1) % 5 != 0);
        }
        ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, pattern);
    } else if (arch == LLM_ARCH_COHERE2MOE || arch == LLM_ARCH_MIMO2 || arch == LLM_ARCH_STEP35 || arch == LLM_ARCH_SPARK2_5 ||
            arch == LLM_ARCH_MUSE_GLIMMER || arch == LLM_ARCH_MUSE_SPARK || arch == LLM_ARCH_GRANITE_SWA || arch == LLM_ARCH_DOTS3NOTE ||
            arch == LLM_ARCH_KOLIBRI1 || arch == LLM_ARCH_MAPLE || arch == LLM_ARCH_FLINT || arch == LLM_ARCH_BEAM || arch == LLM_ARCH_RICURSIVE) {
        std::vector<uint32_t> pattern;
        pattern.reserve(n_layer);
        for (uint32_t il = 0; il < n_layer; il++) {
            pattern.push_back(il % 2);
        }
        ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, pattern);
        if (arch == LLM_ARCH_FLINT) {
            ms.add_kv(LLM_KV_ATTENTION_ROPE_PATTERN, pattern);
        }
    } else if (arch == LLM_ARCH_LFM3) {
        ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, std::vector<uint32_t>({0, 0, 1}));
        ms.add_kv(LLM_KV_ROPE_FREQ_BASE_SWA,               10000.0f);
    } else {
        ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, uint32_t(2));
    }

    if (arch == LLM_ARCH_MUSE_SPARK) {
        ms.add_kv(LLM_KV_POST_NORM_EPS,           1e-5f);
        ms.add_kv(LLM_KV_ROPE_FREQ_BASE_SWA,      10000.0f);
        ms.add_kv(LLM_KV_FINAL_LOGIT_SOFTCAPPING, 30.0f);
        ms.add_kv(LLM_KV_MOE_LATENT_SIZE,         n_embd/2);
        ms.add_kv(LLM_KV_EXPERT_WEIGHTS_SCALE,    1.0f);
        ms.add_kv(LLM_KV_EXPERT_WEIGHTS_NORM,     true);
    }

    if (arch == LLM_ARCH_KOLIBRI1 || arch == LLM_ARCH_LIMITE) {
        ms.add_kv(LLM_KV_EXPERT_WEIGHTS_NORM, false);
    }

    // MSA requires one indexer head per GQA (KV) head, unlike the DSA archs where the
    // indexer head count is independent of the main attention head count.
    if (arch == LLM_ARCH_QWEN4EXP || arch == LLM_ARCH_GLM5_NEXT) {
        ms.add_kv(LLM_KV_HYPER_CONNECTION_COUNT,    uint32_t(4));
        ms.add_kv(LLM_KV_HYPER_CONNECTION_SINKHORN_ITERATIONS, uint32_t(2));
        ms.add_kv(LLM_KV_HYPER_CONNECTION_EPSILON,  1.0e-6f);
        ms.add_kv(LLM_KV_HYPER_CONNECTION_LOW_RANK, uint32_t(8));
        // without this the QSA layers fall back to dense and go uncovered
        ms.add_kv(LLM_KV_ATTENTION_COMPRESS_RATIOS, std::vector<uint32_t>(n_layer, 4));

        // has_cell_ext() needs ple_n_heads here: the indexer cache serializes no ext without it
        const uint32_t ple_ngram_size      = 3;
        const uint32_t ple_heads_per_ngram = 2;
        const uint32_t ple_n_heads         = (ple_ngram_size - 1)*ple_heads_per_ngram;
        GGML_ASSERT(n_embd % ple_n_heads == 0);
        const uint32_t ple_head_dim = n_embd/ple_n_heads;

        std::vector<uint64_t> ple_head_offsets(ple_n_heads);
        std::vector<uint64_t> ple_head_vocab_sizes(ple_n_heads, n_vocab);
        for (uint32_t h = 0; h < ple_n_heads; h++) {
            ple_head_offsets[h] = uint64_t(h)*n_vocab;
        }

        // the PLE history lives in the recurrent cache, so it must sit on a linear attention layer
        ms.add_kv(LLM_KV_PLE_LAYERS,                  std::vector<uint32_t>({ 0 }));
        ms.add_kv(LLM_KV_PLE_NGRAM_SIZE,              ple_ngram_size);
        ms.add_kv(LLM_KV_PLE_HEADS_PER_NGRAM,         ple_heads_per_ngram);
        ms.add_kv(LLM_KV_PLE_CONV_KERNEL,             uint32_t(4));
        ms.add_kv(LLM_KV_PLE_EOS_TOKEN_ID,            uint32_t(0));
        ms.add_kv(LLM_KV_EMBEDDING_LENGTH_PER_LAYER,  ple_head_dim);
        ms.add_kv(LLM_KV_PLE_LAYER_MULTIPLIERS,       std::vector<uint64_t>({ 1, 3, 5 }));
        ms.add_kv(LLM_KV_PLE_HEAD_OFFSETS,            ple_head_offsets);
        ms.add_kv(LLM_KV_PLE_HEAD_VOCAB_SIZES,        ple_head_vocab_sizes);
    }

    // minimax-m3 keeps one indexer head per GQA head; the rest use a fixed 64 to match the fused
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT,   arch == LLM_ARCH_MINIMAX_M3 ? n_head : uint32_t(64));
    // qwen4exp ropes indexer keys with the main rotary width, so its head can't be < n_rot
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH,
              arch == LLM_ARCH_QWEN4EXP ? n_embd_head : uint32_t(128));

    // note: using a realistic top-k here makes the results unstable and hard to match between CPU and GPU
    //       a large value makes things deterministic since all data is selected by the indexer
    // The fork's GLM-5.3 checks are about the selection itself (a fresh score against a reused one, attention through
    // the indices, the pools of a rollback), so glm5next keeps a top-k its sequences outgrow, as before upstream v1.0.0
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_TOP_K,        arch == LLM_ARCH_GLM5NEXT ? uint32_t(8) : uint32_t(131072));

    ms.add_kv(LLM_KV_ATTENTION_INDEXER_BLOCK_SIZE,   uint32_t(4));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_KPOOL,        uint32_t(4));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_KPOOL_SELECT_TAIL, true);
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_LOCAL_BLOCKS, uint32_t(1));
    // mrope sections count rope pairs; Ling 3.0 VL files carry [t, h, w] sections
    // summing to n_rot / 2 (n_rot is 64 in this fixture)
    if (arch == LLM_ARCH_BAILINGMOE3) {
        ms.add_kv(LLM_KV_ROPE_DIMENSION_SECTIONS, std::vector<uint32_t>({8, 12, 12, 0}));
    } else {
        ms.add_kv(LLM_KV_ROPE_DIMENSION_SECTIONS, std::vector<uint32_t>({n_embd_head/4, n_embd_head/4, n_embd_head/4, n_embd_head/4}));
    }

    if (arch == LLM_ARCH_HY_V4) {
        ms.add_kv(LLM_KV_HYPER_CONNECTION_COUNT,     uint32_t(4));
        ms.add_kv(LLM_KV_HYPER_CONNECTION_EPSILON,   1.0e-6f);
        ms.add_kv(LLM_KV_HYPER_CONNECTION_MAGNITUDE, 2.0f);
        ms.add_kv(LLM_KV_SWIGLU_CLAMP_EXP,           10.0f);
        ms.add_kv(LLM_KV_EXPERT_WEIGHTS_SCALE,       1.0f);
        ms.add_kv(LLM_KV_EXPERT_WEIGHTS_NORM,        true);
        // layer 0 must own an indexer, the odd layers share it
        std::vector<uint32_t> indexer_types;
        indexer_types.reserve(n_layer);
        for (uint32_t il = 0; il < n_layer; il++) {
            indexer_types.push_back(il % 2 ? 0 : 1);
        }
        ms.add_kv(LLM_KV_ATTENTION_INDEXER_TYPES, indexer_types);
    }

    if (arch == LLM_ARCH_MISTRAL_LARGE4) {
        // layer 3 reads the KV cache of layer 2, both are NoPE
        ms.add_kv(LLM_KV_ATTENTION_ROPE_PATTERN,     std::vector<uint32_t>({1, 1, 0, 0}));
        ms.add_kv(LLM_KV_ATTENTION_INDEXER_TYPES,    std::vector<uint32_t>({1, 0, 1, 1}));
        ms.add_kv(LLM_KV_ATTENTION_KV_SOURCE_MAP,    std::vector<uint32_t>({0, 1, 2, 2}));
        ms.add_kv(LLM_KV_FINAL_LOGIT_SOFTCAPPING,    30.0f);
    }

    if (arch == LLM_ARCH_DEEPSEEK4) {
        ms.add_kv(LLM_KV_ATTENTION_OUTPUT_GROUP_COUNT,          uint32_t(8));
        ms.add_kv(LLM_KV_ATTENTION_OUTPUT_LORA_RANK,            uint32_t(32));
        ms.add_kv(LLM_KV_ATTENTION_COMPRESS_RATIOS,             std::vector<uint32_t>({0, 0, 4, 128}));
        ms.add_kv(LLM_KV_ATTENTION_COMPRESS_ROPE_FREQ_BASE,     160000.0f);
        ms.add_kv(LLM_KV_HYPER_CONNECTION_COUNT,                uint32_t(4));
        ms.add_kv(LLM_KV_HYPER_CONNECTION_SINKHORN_ITERATIONS,  uint32_t(2));
        ms.add_kv(LLM_KV_HYPER_CONNECTION_EPSILON,              1.0e-6f);
        ms.add_kv(LLM_KV_HASH_LAYER_COUNT,                      uint32_t(0));
        ms.add_kv(LLM_KV_SWIGLU_CLAMP_EXP,                      10.0f);
        ms.add_kv(LLM_KV_EXPERT_WEIGHTS_SCALE,                  1.0f);
        ms.add_kv(LLM_KV_EXPERT_WEIGHTS_NORM,                   true);
    } else if (arch == LLM_ARCH_GLM5NEXT) {
        ms.add_kv(LLM_KV_HYPER_CONNECTION_COUNT,               uint32_t(4)); // build_hc_pre asserts exactly 4 streams
        ms.add_kv(LLM_KV_HYPER_CONNECTION_SINKHORN_ITERATIONS, uint32_t(2));
        ms.add_kv(LLM_KV_HYPER_CONNECTION_EPSILON,             1.0e-6f);
        // the only arch that pools indexer keys; top_k must be a whole number of pools and the
        // selection width must stay under n_ctx or the sparse path goes unused
        ms.add_kv(LLM_KV_ATTENTION_INDEXER_KPOOL,              uint32_t(4));
        // glm5next reads these unconditionally; the if (moe) block below never sets them
        ms.add_kv(LLM_KV_EXPERT_WEIGHTS_SCALE,                 1.0f);
        ms.add_kv(LLM_KV_EXPERT_WEIGHTS_NORM,                   true);
    }

    if (arch == LLM_ARCH_DEEPSEEK41) {
        ms.add_kv(LLM_KV_ATTENTION_OUTPUT_GROUP_COUNT,          uint32_t(8));
        ms.add_kv(LLM_KV_ATTENTION_OUTPUT_LORA_RANK,            uint32_t(32));
        ms.add_kv(LLM_KV_ATTENTION_COMPRESS_RATIOS,             std::vector<uint32_t>({0, 0, 2, 2, 1, 1}));
        ms.add_kv(LLM_KV_ATTENTION_COMPRESS_ROPE_FREQ_BASE,     160000.0f);
        ms.add_kv(LLM_KV_ATTENTION_KV_SOURCE_LAYERS,            std::vector<uint32_t>({2, 4}));
        ms.add_kv(LLM_KV_ATTENTION_INDEXER_SOURCE_LAYERS,       std::vector<uint32_t>({2, 4, 5}));
        ms.add_kv(LLM_KV_ATTENTION_INDEXER_CANDIDATE_SOURCE_LAYER, int32_t(4));
        ms.add_kv(LLM_KV_ATTENTION_INDEXER_CANDIDATE_TOP_K_BLOCKS, uint32_t(4096));
        ms.add_kv(LLM_KV_ATTENTION_INDEXER_CANDIDATE_BLOCK_SIZE,   uint32_t(4));
        ms.add_kv(LLM_KV_HYPER_CONNECTION_COUNT,                uint32_t(4));
        ms.add_kv(LLM_KV_HYPER_CONNECTION_SINKHORN_ITERATIONS,  uint32_t(2));
        ms.add_kv(LLM_KV_HYPER_CONNECTION_EPSILON,              1.0e-6f);
        ms.add_kv(LLM_KV_HASH_LAYER_COUNT,                      uint32_t(0));
        ms.add_kv(LLM_KV_SWIGLU_CLAMP_EXP,                      10.0f);
        ms.add_kv(LLM_KV_EXPERT_WEIGHTS_SCALE,                  1.0f);
        ms.add_kv(LLM_KV_EXPERT_WEIGHTS_NORM,                   true);
        std::vector<int32_t> token_map(n_vocab);
        for (uint32_t i = 0; i < n_vocab; ++i) {
            token_map[i] = (int32_t) (i % 61);
        }
        ms.add_kv(LLM_KV_PLE_LAYERS,                   std::vector<uint32_t>({1, 3}));
        ms.add_kv(LLM_KV_PLE_NGRAM_SIZE,               uint32_t(3));
        ms.add_kv(LLM_KV_PLE_HEADS_PER_NGRAM,          uint32_t(2));
        ms.add_kv(LLM_KV_EMBEDDING_LENGTH_PER_LAYER,   uint32_t(64));
        ms.add_kv(LLM_KV_PLE_LAYER_MULTIPLIERS,        std::vector<uint64_t>({ 76632096046245ull, 4839876093313ull, 35959672319349ull,
                                                                               67716810739261ull, 51510806800915ull, 30921347202721ull }));
        ms.add_kv(LLM_KV_PLE_HEAD_OFFSETS,             std::vector<uint64_t>({0, 101, 204, 311, 420, 533, 660, 791}));
        ms.add_kv(LLM_KV_PLE_HEAD_VOCAB_SIZES,         std::vector<uint64_t>({101, 103, 107, 109, 113, 127, 131, 137}));
        ms.add_kv(LLM_KV_ENGRAM_PAD_TOKEN_ID,          uint32_t(2));
        ms.add_kv(LLM_KV_ENGRAM_TOKEN_MAP,             token_map);
    }

    if (arch == LLM_ARCH_HRM_TEXT) {
        // 6 cache slots alias 2 physical blocks: 1 low-stack layer + 1 high-stack layer
        ms.add_kv(LLM_KV_HRM_LAYERS_PER_STACK, uint32_t(1));
        ms.add_kv(LLM_KV_HRM_H_CYCLES,         uint32_t(2));
        ms.add_kv(LLM_KV_HRM_L_CYCLES,         uint32_t(2));
    }

    if (arch == LLM_ARCH_MAPLE) {
        ms.add_kv(LLM_KV_SWIGLU_CLAMP_EXP, 7.0f);
    }

    if (arch == LLM_ARCH_RICURSIVE) {
        std::vector<float> swiglu_clamp;
        swiglu_clamp.reserve(n_layer);
        for (uint32_t il = 0; il < n_layer; il++) {
            // the random weight scales keep the FFN activations small, a small limit still clamps so the roundtrip checks the per-layer values
            swiglu_clamp.push_back(il % 3 == 1 ? 0.0f : 1e-3f);
        }
        ms.add_kv(LLM_KV_EMBEDDING_SCALE, 16.0f);
        ms.add_kv(LLM_KV_SWIGLU_CLAMP,    swiglu_clamp);
    }

    if (arch == LLM_ARCH_METIS16) {
        ms.add_kv(LLM_KV_METIS_PASS_COUNT,               uint32_t(2));
        ms.add_kv(LLM_KV_HYPER_CONNECTION_COUNT,         uint32_t(4));
        ms.add_kv(LLM_KV_METIS_MEMORY_LENGTH,            uint32_t(64));
        ms.add_kv(LLM_KV_METIS_ROUTE_LENGTH,             uint32_t(32));
        ms.add_kv(LLM_KV_ATTENTION_PASS_LORA_RANK,       uint32_t(8));
        ms.add_kv(LLM_KV_METIS_CONTINUATION_LOGIT_LIMIT, 4.0f);
        ms.add_kv(LLM_KV_SSM_NORM_EPS,                   1e-5f);
        ms.add_kv(LLM_KV_MOE_LATENT_SIZE,                n_embd/2);
    }

    if (arch == LLM_ARCH_LIMITE) {
        ms.add_kv(LLM_KV_ATTENTION_SCALE,             0.1f);
        ms.add_kv(LLM_KV_ATTENTION_XSA_EPS,           1e-4f);
        ms.add_kv(LLM_KV_FINAL_LOGIT_SIGMOID_CAPPING, std::vector<float>({23.0f, 5.0f, 7.5f}));
        ms.add_kv(LLM_KV_MUDD_FEED_FORWARD_LENGTH,    uint32_t(8));
        ms.add_kv(LLM_KV_MUDD_TAP_COUNT,              uint32_t(2));
        ms.add_kv(LLM_KV_MUDD_TAP_INDICES,            std::vector<int32_t>({-1, -1, -1, -1, 0, 2, 1, 3}));
    }

    // dummy tokenizer: token ids are derived from fixed-size chunks and detokenized as hex ids
    {
        std::vector<std::string> tokenizer_list(n_vocab);
        std::vector<float>       tokenizer_scores(n_vocab, 0.0f);

        ms.add_kv(LLM_KV_TOKENIZER_MODEL,         "test");
        for (uint32_t i = 0; i < n_vocab; i++) {
            tokenizer_list[i] = "tok_" + std::to_string(i);
        }
        ms.add_kv(LLM_KV_TOKENIZER_LIST,   tokenizer_list);
        ms.add_kv(LLM_KV_TOKENIZER_SCORES, tokenizer_scores);
    }

    // ms.add_kv(LLM_KV_DENSE_2_FEAT_OUT,     n_embd);
    // ms.add_kv(LLM_KV_DENSE_3_FEAT_IN,      n_embd);

    if (moe) {
        ms.add_kv(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, n_ff);
        ms.add_kv(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, n_ff / 2);  // distinct from n_ff so a saver key-clobber surfaces on reload
        ms.add_kv(LLM_KV_EXPERT_LATENT_LENGTH,       n_ff);
        ms.add_kv(LLM_KV_INTERLEAVE_MOE_LAYER_STEP,  uint32_t(2));
        // with more experts than a ubatch uses, the copy of the used experts in host memory skips some of them
        ms.add_kv(LLM_KV_EXPERT_COUNT,               uint32_t(host_experts_test(arch) ? 64 : arch == LLM_ARCH_CLOVER_MOE ? 4 : 2));
        ms.add_kv(LLM_KV_EXPERT_USED_COUNT,          uint32_t(2));
        ms.add_kv(LLM_KV_EXPERT_SHARED_COUNT,        uint32_t(1));

        uint32_t gating_func = 2;
        switch (arch) {
            case LLM_ARCH_MELLUM:
                gating_func = 1;
                break;
            case LLM_ARCH_DEEPSEEK4:
            case LLM_ARCH_DEEPSEEK41:
                gating_func = 4;
                break;
            default:
                break;
        }

        ms.add_kv(LLM_KV_EXPERT_GATING_FUNC,         gating_func); // sigmoid : softmax : sqrtsoftplus
        ms.add_kv(LLM_KV_EXPERT_GROUP_SCALE,         1.0f);
        ms.add_kv(LLM_KV_EXPERTS_PER_GROUP,          uint32_t(1));
        if (arch == LLM_ARCH_FLINT) {
            ms.add_kv(LLM_KV_EXPERT_WEIGHTS_NORM, true);
        }
        if (arch == LLM_ARCH_K2_HORIZON) {
            ms.add_kv(LLM_KV_ATTENTION_VALUE_EXPERT_COUNT,      uint32_t(2));
            ms.add_kv(LLM_KV_ATTENTION_VALUE_EXPERT_USED_COUNT, uint32_t(2));
        }
    }

    if (arch == LLM_ARCH_BEAM) {
        ms.add_kv(LLM_KV_EXPERT_COUNT,            uint32_t(8));
        ms.add_kv(LLM_KV_EXPERT_GROUP_COUNT,      uint32_t(4));
        ms.add_kv(LLM_KV_EXPERT_GROUP_USED_COUNT, uint32_t(2));
        ms.add_kv(LLM_KV_EXPERT_WEIGHTS_SCALE,    2.5f);
        ms.add_kv(LLM_KV_EXPERT_WEIGHTS_NORM,     true);
    }

    if (arch == LLM_ARCH_MAION_CODER) {
        ms.add_kv(LLM_KV_MOE_LATENT_SIZE, n_embd / 2);
        ms.add_kv(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, std::vector<uint32_t>{n_ff, n_ff / 2, n_ff, n_ff});
    }

    ms.add_kv(LLM_KV_POSNET_EMBEDDING_LENGTH,   n_embd);
    ms.add_kv(LLM_KV_POSNET_BLOCK_COUNT,        n_layer);
    ms.add_kv(LLM_KV_CONVNEXT_EMBEDDING_LENGTH, n_embd);
    ms.add_kv(LLM_KV_CONVNEXT_BLOCK_COUNT,      n_layer);
    ms.add_kv(LLM_KV_XIELU_ALPHA_N,             1.0f);
    ms.add_kv(LLM_KV_XIELU_ALPHA_P,             1.0f);
    ms.add_kv(LLM_KV_XIELU_BETA,                1.0f);
    ms.add_kv(LLM_KV_XIELU_EPS,                 1.0e-7f);
    ms.add_kv(LLM_KV_SSM_INNER_SIZE,            arch == LLM_ARCH_QWEN3NEXT || arch == LLM_ARCH_QWEN35 || arch == LLM_ARCH_QWEN35MOE || arch == LLM_ARCH_QWEN4EXP || arch == LLM_ARCH_CLOVER_MOE ? 256 : 2*n_embd);
    ms.add_kv(LLM_KV_SSM_CONV_KERNEL,           uint32_t(4));
    ms.add_kv(LLM_KV_SSM_STATE_SIZE,            uint32_t(128));
    ms.add_kv(LLM_KV_SSM_TIME_STEP_RANK,        n_head);
    ms.add_kv(LLM_KV_SSM_GROUP_COUNT,           arch == LLM_ARCH_PLAMO2 ? 0 : uint32_t(2));
    ms.add_kv(LLM_KV_KDA_HEAD_DIM,              uint32_t(128));
    ms.add_kv(LLM_KV_KDA_SAFE_GATE,             true);
    ms.add_kv(LLM_KV_KDA_GATE_LOWER_BOUND,      -5.0f);
    if (arch == LLM_ARCH_BAILINGMOE3) {
        std::vector<float> clamp_exp;
        std::vector<float> clamp_shexp;
        for (uint32_t il = 0; il < n_layer; il++) {
            clamp_exp  .push_back(il % 2 ? 4.0f : 0.0f);
            clamp_shexp.push_back(il % 2 ? 5.0f : 0.0f);
        }
        ms.add_kv(LLM_KV_SWIGLU_CLAMP_EXP,   clamp_exp);
        ms.add_kv(LLM_KV_SWIGLU_CLAMP_SHEXP, clamp_shexp);
    }
    ms.add_kv(LLM_KV_WKV_HEAD_SIZE,               n_embd/n_head);
    ms.add_kv(LLM_KV_SHORTCONV_L_CACHE,           uint32_t(3));
    ms.add_kv(LLM_KV_RESIDUAL_SCALE,              3.5565588200778455f);
    ms.add_kv(LLM_KV_ATTN_RES_BLOCK_SIZE,         uint32_t(12));
    ms.add_kv(LLM_KV_ACTIVATION_SITU_BETA,        4.0f);
    ms.add_kv(LLM_KV_ACTIVATION_SITU_LINEAR_BETA, 25.0f);
    ms.add_kv(LLM_KV_KDA_GATE_LOWER_BOUND,        -5.0f);

    for (uint32_t il = 0; il < n_layer; il++) {
        ggml_tensor t;
        memset(&t, 0, sizeof(ggml_tensor));
        t.type = GGML_TYPE_F16;
        ggml_format_name(&t, "conv%" PRIu32 "d.weight", il);
        gguf_add_tensor(ms.gguf_ctx, &t);
        ggml_format_name(&t, "posnet.%" PRIu32 ".conv1.weight", il);
        gguf_add_tensor(ms.gguf_ctx, &t);
        ggml_format_name(&t, "posnet.%" PRIu32 ".conv2.weight", il);
        gguf_add_tensor(ms.gguf_ctx, &t);
        ggml_format_name(&t, "convnext.%" PRIu32 ".dw.weight", il);
        gguf_add_tensor(ms.gguf_ctx, &t);
    }
    return ret;
}

static bool silent_model_load_progress(float /*progress*/, void * /*user_data*/) {
    return true;
}

static std::string get_tmp_gguf_path() {
    // the suite runs several of this test at once (tests/CMakeLists.txt), each counting from 1: a tag of the process
    // keeps one's file from being written over by another's between its save and its load
    static const unsigned tag = std::random_device{}();
    static size_t id = 0;
    return (std::filesystem::temp_directory_path() /
            ("test-llama-archs-" + std::to_string(tag) + "-" + std::to_string(++id) + ".gguf")).string();
}

static llama_model_ptr get_model(
        struct gguf_context * gguf_ctx, FILE * file, const size_t seed, const float stdev,
        const std::vector<ggml_backend_dev_t> & devs,
        const llama_split_mode split_mode = LLAMA_SPLIT_MODE_LAYER,
        const llama_model_tensor_buft_override * tensor_buft_overrides = nullptr,
        const std::vector<float> & tensor_split = {}, bool mtp = false) {
    GGML_ASSERT((gguf_ctx == nullptr) != (file == nullptr));
    llama_model_params model_params = llama_model_default_params();
    model_params.progress_callback = silent_model_load_progress;
    model_params.load_mtp = mtp;
    std::vector<ggml_backend_dev_t> devs_copy = devs;
    devs_copy.push_back(nullptr);
    model_params.devices = devs_copy.data();
    model_params.split_mode = split_mode;
    model_params.tensor_split = tensor_split.empty() ? nullptr : tensor_split.data();
    model_params.tensor_buft_overrides = tensor_buft_overrides;

    tensor_data_params tensor_params = { seed, stdev };
    llama_model_ptr model(gguf_ctx != nullptr ?
        llama_model_init_from_user(gguf_ctx, set_tensor_data, &tensor_params, model_params) :
        llama_model_load_from_file_ptr(file, model_params));
    if (!model) {
        throw std::runtime_error("failed to create llama model");
    }
    return model;
}

static std::pair<llama_model_ptr, llama_context_ptr> get_model_and_ctx(
        struct gguf_context * gguf_ctx, FILE * file, const size_t seed, const float stdev,
        const std::vector<ggml_backend_dev_t> & devs,
        const llama_split_mode split_mode = LLAMA_SPLIT_MODE_LAYER, bool encode = false,
        const llama_model_tensor_buft_override * tensor_buft_overrides = nullptr, const size_t moe_cache_size = 0,
        const std::vector<float> & tensor_split = {}, bool mtp = false, bool q8_cache = false) {
    llama_model_ptr model = get_model(gguf_ctx, file, seed, stdev, devs, split_mode, tensor_buft_overrides, tensor_split, mtp);

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 0;
    ctx_params.n_threads = 4;
    ctx_params.n_threads_batch = 4;
    if (!encode) {
        ctx_params.n_ubatch = 64;
    }
    if (split_mode == LLAMA_SPLIT_MODE_DATA) {
        ctx_params.kv_unified = false;
        ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        ctx_params.n_seq_max = 8;
        ctx_params.n_ctx = 8*256;
    }
    if (q8_cache) {
        ctx_params.type_k = ctx_params.type_v = ctx_params.type_s = ctx_params.type_idx = GGML_TYPE_Q8_0;
        ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    }
    if (moe_cache_size > 0) {
        // the MoE cache is only used for small ubatches
        ctx_params.moe_cache_size = moe_cache_size;
        ctx_params.n_ubatch = 2;
    }

    llama_context_ptr lctx(llama_init_from_model(model.get(), ctx_params));
    if (!lctx) {
        throw std::runtime_error("failed to create llama context");
    }
    return std::make_pair(std::move(model), std::move(lctx));
}

static std::vector<float> get_logits(
        llama_model * model, llama_context * lctx, const std::vector<llama_token> & tokens, bool encode = false) {
    const uint32_t n_vocab  = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const uint32_t n_ctx    = llama_n_ctx(lctx);
    const uint32_t n_tokens = tokens.size();
    common_batch batch(lctx);
    GGML_ASSERT(n_tokens <= n_ctx);
    for (uint32_t pos = 0; pos < n_tokens; pos++) {
        batch.add(tokens[pos], pos, 0, true);
    }
    if (encode) {
        if (llama_process(lctx, LLAMA_PROCESS_TYPE_ENCODE, batch.get())) {
            throw std::runtime_error("failed to encode batch");
        }
    }
    if (llama_process(lctx, LLAMA_PROCESS_TYPE_DECODE, batch.get())) {
        throw std::runtime_error("failed to decode batch");
    }

    std::vector<float> ret;
    ret.reserve(n_tokens*n_vocab);
    for (uint32_t i = 0; i < n_tokens; i++) {
        const float * logits_ith = llama_get_logits_ith(lctx, i);
        for (uint32_t j = 0; j < n_vocab; j++) {
            ret.push_back(logits_ith[j]);
        }
    }
    return ret;
}

// entries [n/4, n/2) are embd rows, decoded either as token/embd/token chunks or as one mixed batch
// returns the llama_process() error code
static int32_t get_logits_mixed(
        llama_model * model, llama_context * lctx, const std::vector<llama_token> & tokens, const std::vector<float> & embd, bool mixed,
        std::vector<float> & ret) {
    const uint32_t n_vocab  = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const uint32_t n_embd   = llama_model_n_embd_inp(model);
    const uint32_t n_tokens = tokens.size();
    const uint32_t i_embd_0 = n_tokens/4;
    const uint32_t i_embd_1 = n_tokens/2;

    const std::vector<uint32_t> bounds = mixed
        ? std::vector<uint32_t>{0, n_tokens}
        : std::vector<uint32_t>{0, i_embd_0, i_embd_1, n_tokens};

    llama_memory_clear(llama_get_memory(lctx), true);
    llama_batch_ext_ptr batch(llama_batch_ext_init(lctx));

    ret.clear();
    ret.reserve(n_tokens*n_vocab);
    for (size_t c = 0; c + 1 < bounds.size(); c++) {
        llama_batch_ext_clear(batch.get());
        for (uint32_t i = bounds[c]; i < bounds[c + 1]; i++) {
            const bool is_embd = i >= i_embd_0 && i < i_embd_1;
            const int32_t idx = is_embd
                ? llama_batch_ext_add_embd(batch.get(), 0, { embd.data() + (size_t) (i - i_embd_0)*n_embd, 1, n_embd })
                : llama_batch_ext_add_token(batch.get(), 0, tokens[i]);
            GGML_ASSERT(idx >= 0);
            const llama_pos pos[4] = { (llama_pos) i, (llama_pos) i, (llama_pos) i, 0 };
            llama_batch_ext_set_pos(batch.get(), idx, pos);
            llama_batch_ext_set_output_logits(batch.get(), idx, true);
        }
        const int32_t err = llama_process(lctx, LLAMA_PROCESS_TYPE_DECODE, batch.get());
        if (err != 0) {
            return err;
        }
        for (uint32_t i = 0; i < bounds[c + 1] - bounds[c]; i++) {
            const float * logits_ith = llama_get_logits_ith(lctx, i);
            ret.insert(ret.end(), logits_ith, logits_ith + n_vocab);
        }
    }
    return 0;
}

static bool check_causal_attn_toggle(
        llama_model * model, llama_context * lctx, const std::vector<llama_token> & tokens) {
    const uint32_t n_vocab  = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const uint32_t n_past   = tokens.size();
    const uint32_t n_ubatch = llama_n_ubatch(lctx);

    GGML_ASSERT(n_past + n_ubatch/2 + n_ubatch <= llama_n_ctx(lctx));

    llama_set_causal_attn(lctx, false);

    common_batch batch(lctx);

    bool ok = true;
    uint32_t pos = n_past;
    for (const uint32_t n_tokens : { n_ubatch/2, n_ubatch }) {
        batch.clear();
        for (uint32_t i = 0; i < n_tokens; i++) {
            batch.add(tokens[i], pos++, 0, true);
        }

        const int32_t rc = llama_process(lctx, LLAMA_PROCESS_TYPE_DECODE, batch.get());
        if (rc != 0) {
            LOG_ERR("%s: n_tokens=%u: llama_process returned %d\n", __func__, n_tokens, rc);
            ok = false;
            break;
        }

        const float * logits = llama_get_logits_ith(lctx, n_tokens - 1);
        if (logits == nullptr) {
            LOG_ERR("%s: n_tokens=%u: no logits\n", __func__, n_tokens);
            ok = false;
            break;
        }
        for (uint32_t j = 0; j < n_vocab; j++) {
            if (std::isnan(logits[j])) {
                LOG_ERR("%s: n_tokens=%u: nan logit\n", __func__, n_tokens);
                ok = false;
                break;
            }
        }
    }

    return ok;
}

// the GLM checks' form, devices after the seed: tensor values come from the normal pool main fills, so no stdev is passed
static std::pair<llama_model_ptr, llama_context_ptr> get_model_and_ctx(
        struct gguf_context * gguf_ctx, FILE * file, const size_t seed, const std::vector<ggml_backend_dev_t> & devs,
        const llama_split_mode split_mode = LLAMA_SPLIT_MODE_LAYER, bool encode = false, const std::vector<float> & tensor_split = {}, bool mtp = false, bool q8_cache = false) {
    return get_model_and_ctx(gguf_ctx, file, seed, 0.0f, devs, split_mode, encode, nullptr, 0, tensor_split, mtp, q8_cache);
}

// common_batch_add and common_batch_clear as they were before upstream moved common to llama_batch_ext
static void batch_clear(struct llama_batch & batch) {
    batch.n_tokens = 0;
}

static void batch_add(
                 struct llama_batch & batch,
                        llama_token   id,
                          llama_pos   pos,
    const std::vector<llama_seq_id> & seq_ids,
                               bool   logits) {
    GGML_ASSERT(batch.seq_id[batch.n_tokens] && "llama_batch size exceeded");

    batch.token   [batch.n_tokens] = id;
    batch.pos     [batch.n_tokens] = pos;
    batch.n_seq_id[batch.n_tokens] = seq_ids.size();
    for (size_t i = 0; i < seq_ids.size(); ++i) {
        batch.seq_id[batch.n_tokens][i] = seq_ids[i];
    }
    batch.logits  [batch.n_tokens] = logits;

    batch.n_tokens++;
}

static bool test_glm_nextn_store(const size_t seed, const std::vector<ggml_backend_dev_t> & devs,
        llama_split_mode split_mode, const std::vector<float> & tensor_split, bool q8_cache = false);
static bool test_glm_nextn_masked_rows(const size_t seed, const std::vector<ggml_backend_dev_t> & devs,
        llama_split_mode split_mode, const std::vector<float> & tensor_split, bool q8_cache = false);
static bool test_glm_generated_prefixes(const size_t seed, const std::vector<ggml_backend_dev_t> & devs,
        llama_split_mode split_mode, const std::vector<float> & tensor_split, const std::string & diagnostic_dir, bool q8_cache = false);

static void glm_use_hyphenated_metadata(gguf_context * ctx) {
    gguf_context_ptr original(gguf_init_empty());
    gguf_set_kv(original.get(), ctx);
    for (int64_t i = 0; i < gguf_get_n_kv(original.get()); ++i) {
        const std::string key = gguf_get_key(original.get(), i);
        if (key.compare(0, 9, "glm5next.") != 0) { continue; }
        const std::string renamed = "glm5-next." + key.substr(9);
        switch (gguf_get_kv_type(original.get(), i)) {
            case GGUF_TYPE_UINT32: gguf_set_val_u32(ctx, renamed.c_str(), gguf_get_val_u32(original.get(), i)); break;
            case GGUF_TYPE_INT32:  gguf_set_val_i32(ctx, renamed.c_str(), gguf_get_val_i32(original.get(), i)); break;
            case GGUF_TYPE_FLOAT32: gguf_set_val_f32(ctx, renamed.c_str(), gguf_get_val_f32(original.get(), i)); break;
            case GGUF_TYPE_BOOL: gguf_set_val_bool(ctx, renamed.c_str(), gguf_get_val_bool(original.get(), i)); break;
            case GGUF_TYPE_STRING: gguf_set_val_str(ctx, renamed.c_str(), gguf_get_val_str(original.get(), i)); break;
            case GGUF_TYPE_ARRAY:
                if (gguf_get_arr_type(original.get(), i) == GGUF_TYPE_STRING) {
                    std::vector<const char *> values;
                    for (size_t j = 0; j < gguf_get_arr_n(original.get(), i); ++j) {
                        values.push_back(gguf_get_arr_str(original.get(), i, j));
                    }
                    gguf_set_arr_str(ctx, renamed.c_str(), values.data(), values.size());
                } else {
                    gguf_set_arr_data(ctx, renamed.c_str(), gguf_get_arr_type(original.get(), i),
                        gguf_get_arr_data(original.get(), i), gguf_get_arr_n(original.get(), i));
                }
                break;
            default: throw std::runtime_error("unexpected GLM fixture metadata type");
        }
        GGML_ASSERT(gguf_remove_key(ctx, key.c_str()) >= 0);
    }
    gguf_set_val_str(ctx, "general.architecture", "glm5-next");
}

static bool test_glm_arch_alias(const size_t seed) {
    auto metadata = get_gguf_ctx(LLM_ARCH_GLM5NEXT, true);
    auto reference = get_model_and_ctx(metadata.get(), nullptr, seed, {});
    const auto tokens = get_tokens(17, llama_vocab_n_tokens(llama_model_get_vocab(reference.first.get())), seed);
    const auto expected = get_logits(reference.first.get(), reference.second.get(), tokens);
    glm_use_hyphenated_metadata(metadata.get());
    auto aliased = get_model_and_ctx(metadata.get(), nullptr, seed, {});
    if (get_logits(aliased.first.get(), aliased.second.get(), tokens) != expected) { return false; }

    const std::string path = get_tmp_gguf_path();
    llama_model_saver saver(reference.first.get());
    saver.add_kv_from_model();
    saver.add_tensors_from_model();
    glm_use_hyphenated_metadata(saver.gguf_ctx);
    saver.save(path);
    const auto close_file = [](FILE * file) { fclose(file); };
    std::unique_ptr<FILE, decltype(close_file)> file(fopen(path.c_str(), "rb"), close_file);
    if (!file) { throw std::runtime_error("cannot create GLM alias fixture file"); }
    auto loaded = get_model_and_ctx(nullptr, file.get(), seed, {});
    const bool ok = get_logits(loaded.first.get(), loaded.second.get(), tokens) == expected;
    file.reset();
    std::filesystem::remove(path);
    printf("GLM architecture alias: user metadata and file logits %s\n", ok ? "identical" : "differ");
    return ok;
}

// GLM-5.3 under -sm tensor over the given ranks (a device may repeat: one GPU twice splits as two GPUs do), against
// one device under -sm layer: the same bar as the two-rank regression in test_glm_dp
static int test_glm_tensor(const size_t seed, const std::vector<ggml_backend_dev_t> & ranks, bool q8_cache) {
    GGML_ASSERT(!ranks.empty());
    auto gguf = get_gguf_ctx(LLM_ARCH_GLM5NEXT, true);
    kv_writer ms(LLM_ARCH_GLM5NEXT, gguf.get());
    ms.add_kv(LLM_KV_EXPERT_COUNT, uint32_t(8));
    ms.add_kv(LLM_KV_EXPERT_USED_COUNT, uint32_t(4));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT, uint32_t(64));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH, uint32_t(128));
    ms.add_kv(LLM_KV_CONTEXT_LENGTH, uint32_t(256));
    auto reference = get_model_and_ctx(gguf.get(), nullptr, seed, {ranks.front()}, LLAMA_SPLIT_MODE_LAYER, false, {}, false, q8_cache);
    const auto tokens = get_tokens(128, 128, seed);
    const auto expected = get_logits(reference.first.get(), reference.second.get(), tokens);
    auto tensor = get_model_and_ctx(gguf.get(), nullptr, seed, ranks, LLAMA_SPLIT_MODE_TENSOR, false, {}, false, q8_cache);
    if (!tensor.first || !tensor.second) {
        fprintf(stderr, "GLM tensor %zu ranks, %s cache: the model or context did not load\n", ranks.size(), q8_cache ? "q8_0" : "default");
        return 1;
    }
    const auto actual = get_logits(tensor.first.get(), tensor.second.get(), tokens);
    size_t differ, tied;
    const double kl = mean_kld(expected, actual, 128, differ, tied);
    fprintf(stderr, "GLM tensor %zu ranks, %s cache: mean KL %.9g, %zu differing top tokens\n", ranks.size(), q8_cache ? "q8_0" : "default", kl, differ);
    return !std::isfinite(kl) || kl > 1e-6 || differ != 0 ? 1 : 0;
}

static int test_glm_dp(const size_t seed, const std::vector<ggml_backend_dev_t> & devices, bool q8_cache = false, size_t selected_ranks = 0) {
    GGML_ASSERT(!devices.empty());
    const auto ranks_on = [&](size_t count) {
        std::vector<ggml_backend_dev_t> result;
        for (size_t i = 0; i < count; ++i) { result.push_back(devices[i % devices.size()]); }
        return result;
    };
    auto gguf = get_gguf_ctx(LLM_ARCH_GLM5NEXT, true);
    kv_writer ms(LLM_ARCH_GLM5NEXT, gguf.get());
    ms.add_kv(LLM_KV_EXPERT_COUNT, uint32_t(8));
    ms.add_kv(LLM_KV_EXPERT_USED_COUNT, uint32_t(4));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT, uint32_t(64));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH, uint32_t(128));
    ms.add_kv(LLM_KV_CONTEXT_LENGTH, uint32_t(256));
    auto reference = get_model_and_ctx(gguf.get(), nullptr, seed, {devices.front()}, LLAMA_SPLIT_MODE_LAYER, false, {}, false, q8_cache);
    const auto tokens = get_tokens(128, 128, seed);
    const auto expected = get_logits(reference.first.get(), reference.second.get(), tokens);
    {
        auto tensor = get_model_and_ctx(gguf.get(), nullptr, seed, ranks_on(2), LLAMA_SPLIT_MODE_TENSOR, false, {}, false, q8_cache);
        const auto actual = get_logits(tensor.first.get(), tensor.second.get(), tokens);
        size_t differ, tied;
        const double kl = mean_kld(expected, actual, 128, differ, tied);
        fprintf(stderr, "GLM tensor regression: mean KL %.9g, %zu differing top tokens\n", kl, differ);
        if (!std::isfinite(kl) || kl > 1e-6 || differ != 0) { return 1; }
    }
    bool ok = true;
    for (const size_t ranks : {2u, 4u}) {
        if (selected_ranks != 0 && ranks != selected_ranks) { continue; }
        const auto rank_devices = ranks_on(ranks);
        const std::set<ggml_backend_dev_t> physical_devices(rank_devices.begin(), rank_devices.end());
        fprintf(stderr, "GLM dp: %zu logical ranks on %zu devices, %s cache\n", ranks, physical_devices.size(), q8_cache ? "q8_0" : "default");
        auto split = get_model_and_ctx(gguf.get(), nullptr, seed, ranks_on(ranks), LLAMA_SPLIT_MODE_DATA, false, {}, false, q8_cache);
        const auto actual = get_logits(split.first.get(), split.second.get(), tokens);
        size_t differ, tied;
        const double kl = mean_kld(expected, actual, 128, differ, tied);
        fprintf(stderr, "GLM dp %zu: mean KL %.9g, %zu differing top tokens, %zu ties\n", ranks, kl, differ, tied);
        if (!std::isfinite(kl) || kl > 1e-6 || differ != 0) {
            return 1;
        }
        std::vector<std::vector<llama_token>> histories;
        std::vector<std::vector<float>> expected_seq, expected_wide, actual_seq(8);
        // Dense CUDA kernels depend on the local row count. Match the per-device microbatch.
        auto local_params = llama_context_default_params();
        local_params.n_ctx = 256;
        local_params.n_ubatch = 64/ranks;
        local_params.n_threads = local_params.n_threads_batch = 4;
        local_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        if (q8_cache) { local_params.type_k = local_params.type_v = local_params.type_s = local_params.type_idx = GGML_TYPE_Q8_0; }
        llama_context_ptr local_reference(llama_init_from_model(reference.first.get(), local_params));
        if (!local_reference) { return 1; }
        for (llama_seq_id seq = 0; seq < 4; ++seq) {
            histories.push_back(get_tokens(136, 128, seed + seq));
            llama_memory_clear(llama_get_memory(local_reference.get()), true);
            expected_seq.push_back(get_logits(reference.first.get(), local_reference.get(), histories.back()));
            llama_memory_clear(llama_get_memory(reference.second.get()), true);
            expected_wide.push_back(get_logits(reference.first.get(), reference.second.get(), histories.back()));
        }
        llama_memory_clear(llama_get_memory(split.second.get()), true);
        auto batch = llama_batch_init(512, 0, 1);
        histories.resize(8);
        histories[7] = histories[0];
        histories[6] = histories[1];
        const llama_seq_id order[] = {3, 0, 2, 1, 7, 6};
        for (llama_pos begin = 0; begin < 136;) {
            const llama_pos end = begin < 128 ? begin + 64 : begin + 1;
            if (begin == 64) {
                llama_memory_seq_cp(llama_get_memory(split.second.get()), 0, 7, -1, -1);
                std::vector<uint8_t> checkpoint(llama_state_seq_get_size(split.second.get(), 1));
                if (llama_state_seq_get_data(split.second.get(), checkpoint.data(), checkpoint.size(), 1) != checkpoint.size() ||
                        llama_state_seq_set_data(split.second.get(), checkpoint.data(), checkpoint.size(), 6) != checkpoint.size()) {
                    llama_batch_free(batch);
                    return 1;
                }
            }
            if (begin == 128) {
                std::vector<uint8_t> checkpoint(llama_state_get_size(split.second.get()));
                if (llama_state_get_data(split.second.get(), checkpoint.data(), checkpoint.size()) != checkpoint.size() ||
                        llama_state_set_data(split.second.get(), checkpoint.data(), checkpoint.size()) != checkpoint.size()) {
                    llama_batch_free(batch);
                    return 1;
                }
            }
            batch_clear(batch);
            for (llama_pos pos = begin; pos < end; ++pos) {
                for (auto seq : order) {
                    if (begin > 0 || seq < 4) { batch_add(batch, histories[seq][pos], pos, {seq}, true); }
                }
            }
            if (llama_decode(split.second.get(), batch)) { llama_batch_free(batch); return 1; }
            for (int32_t row = 0; row < batch.n_tokens; ++row) {
                auto & logits = actual_seq[batch.seq_id[row][0]];
                const auto * values = llama_get_logits_ith(split.second.get(), row);
                logits.insert(logits.end(), values, values + 128);
            }
            begin = end;
        }
        llama_batch_free(batch);
        const auto compare = [&](size_t seq, size_t parent, size_t begin) {
            const auto suffix = [&](const std::vector<float> & values) {
                return std::vector<float>(values.begin() + begin*128, values.end());
            };
            size_t local_differ, local_tied, wide_differ, wide_tied;
            const double local_kl = mean_kld(suffix(expected_seq[parent]), actual_seq[seq], 128, local_differ, local_tied);
            const double wide_kl = mean_kld(suffix(expected_wide[parent]), actual_seq[seq], 128, wide_differ, wide_tied);
            fprintf(stderr, "GLM dp %zu seq %zu: local/wide KL %.9g/%.9g, top differences %zu/%zu, ties %zu/%zu\n",
                ranks, seq, local_kl, wide_kl, local_differ, wide_differ, local_tied, wide_tied);
            // A second stream on an owner can select the wider dense kernel. Require agreement with one entire
            // unsplit control and lossless KL against both; do not increase the top-token tolerance.
            return std::isfinite(local_kl) && std::isfinite(wide_kl) && local_kl <= 1e-6 && wide_kl <= 1e-6 &&
                (local_differ == 0 || wide_differ == 0);
        };
        for (size_t seq = 0; seq < 4; ++seq) {
            if (!compare(seq, seq, 0)) { return 1; }
        }
        for (const auto seq : {6, 7}) {
            if (!compare(seq, seq == 7 ? 0 : 1, 64)) { return 1; }
        }
        ok = test_glm_nextn_store(seed, ranks_on(ranks), LLAMA_SPLIT_MODE_DATA, {}, q8_cache) && ok;
        ok = test_glm_nextn_masked_rows(seed, ranks_on(ranks), LLAMA_SPLIT_MODE_DATA, {}, q8_cache) && ok;
        ok = test_glm_generated_prefixes(seed, ranks_on(ranks), LLAMA_SPLIT_MODE_DATA, {}, "", q8_cache) && ok;
    }
    return ok ? 0 : 1;
}

static std::vector<float> get_mtp_logits(
        llama_model * model, llama_context * ctx, const std::vector<llama_token> & tokens, const std::vector<float> & hidden,
        llama_pos start = 0, const std::vector<llama_seq_id> & seqs = {},
        const std::vector<llama_pos> & offsets = {}) {
    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    llama_batch batch = llama_batch_init(tokens.size(), 0, 1);
    for (size_t i = 0; i < tokens.size(); ++i) {
        const size_t n_seq = std::max(size_t(1), seqs.size());
        const llama_pos offset = offsets.empty() ? 0 : offsets[i%n_seq];
        batch_add(batch, tokens[i], start + i/n_seq + offset, {seqs.empty() ? 0 : seqs[i%n_seq]}, true);
    }
    batch.embd = const_cast<float *>(hidden.data());
    const int32_t result = llama_decode(ctx, batch);
    batch.embd = nullptr;
    llama_batch_free(batch);
    if (result) {
        throw std::runtime_error("failed to decode MTP batch");
    }
    std::vector<float> logits;
    for (size_t i = 0; i < tokens.size(); ++i) {
        const float * row = llama_get_logits_ith(ctx, i);
        logits.insert(logits.end(), row, row + n_vocab);
    }
    return logits;
}

static bool test_draft_vocab(const llm_arch arch, const size_t seed, const std::vector<ggml_backend_dev_t> & devs,
        llama_split_mode split_mode, const std::vector<float> & tensor_split) {
    std::mt19937 rng(seed);
    for (const int n_hot : {1, 2, 7, 32, 63, 127}) {
        auto gguf = get_gguf_ctx(arch, arch != LLM_ARCH_QWEN35, true);
        auto pair = get_model_and_ctx(gguf.get(), nullptr, seed, devs, split_mode, false, tensor_split, true);
        auto * model = pair.first.get();
        auto * target = pair.second.get();
        const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
        const auto tokens = get_tokens(5, n_vocab, rng());
        const auto target_before = get_logits(model, target, tokens);
        llama_context_params params = llama_context_default_params();
        params.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
        params.n_ctx = 128;
        params.n_batch = params.n_ubatch = 32;
        params.n_threads = params.n_threads_batch = 4;
        std::normal_distribution<float> normal(0.0f, 0.1f);
        std::vector<float> hidden(tokens.size() * llama_model_n_embd(model));
        for (auto & value : hidden) {
            value = normal(rng);
        }
        llama_context_ptr full(llama_init_from_model(model, params));
        if (!full) {
            throw std::runtime_error("failed to create full MTP context");
        }
        const auto expected = get_mtp_logits(model, full.get(), tokens, hidden);
        full.reset();

        for (const std::vector<llama_token> & invalid : {std::vector<llama_token>{}, {-1}, {128}, {3, 3}, {9, 2}}) {
            if (llama_model_set_draft_vocab(model, invalid.data(), invalid.size())) {
                LOG_ERR("%s draft vocabulary accepted invalid IDs\n", llama_arch_name(arch));
                return false;
            }
        }
        std::vector<llama_token> ids(n_vocab);
        std::iota(ids.begin(), ids.end(), 0);
        std::shuffle(ids.begin(), ids.end(), rng);
        ids.resize(n_hot);
        std::sort(ids.begin(), ids.end());
        if (!llama_model_set_draft_vocab(model, ids.data(), ids.size())) {
            LOG_ERR("%s draft vocabulary rejected %d valid IDs\n", llama_arch_name(arch), n_hot);
            return false;
        }
        llama_context_ptr trimmed(llama_init_from_model(model, params));
        if (!trimmed) {
            throw std::runtime_error("failed to create trimmed MTP context");
        }
        const auto actual = get_mtp_logits(model, trimmed.get(), tokens, hidden);
        std::vector<float> selected_expected, selected_actual;
        for (size_t i = 0; i < actual.size(); ++i) {
            if (std::binary_search(ids.begin(), ids.end(), i % n_vocab)) {
                if (!std::isfinite(actual[i])) {
                    return false;
                }
                selected_expected.push_back(expected[i]);
                selected_actual.push_back(actual[i]);
            } else if (actual[i] != -INFINITY) {
                LOG_ERR("%s draft vocabulary scored an excluded ID\n", llama_arch_name(arch));
                return false;
            }
        }
        if (nmse(selected_expected, selected_actual) > 1e-4) {
            LOG_ERR("%s draft vocabulary changed selected logits\n", llama_arch_name(arch));
            return false;
        }
        llama_memory_clear(llama_get_memory(target), true);
        if (get_logits(model, target, tokens) != target_before) {
            LOG_ERR("%s draft vocabulary changed target logits\n", llama_arch_name(arch));
            return false;
        }
    }
    return true;
}

static void set_env(const char * name, const char * value);

// The node shapes the CUDA backend's folds match in the graphs of the two served architectures, read from the nodes a
// one-row decode computes (TORAD.md, "Upstream v1.0.0"). A graph that computes the same values out of other nodes
// passes every other check of this file and runs without the folds, on any device, so the shapes are held here:
// - a recurrent conv state is read by the batch's own GET_ROWS (the conv-state update and the gather fold match it);
// - qwen35's joint q/k norm is one L2_NORM (the conv kernel folds it, and the alpha and beta matmuls behind it), and
//   its Gated DeltaNet state reaches the RESHAPE straight from its GET_ROWS;
// - the GLM architecture's expert limit is two CLAMPs and the split SwiGLU (the gate of the routed-expert kernels).
// qwen35 also reads the logit lens on a context without an MTP head, where the last layer is cropped to the output
// rows before the lens records it: the last layer's lens row is the served row.
static bool test_served_shapes(const llm_arch arch, const size_t seed, const std::vector<ggml_backend_dev_t> & devs,
        llama_split_mode split_mode, const std::vector<float> & tensor_split) {
    struct node_seen {
        std::string name;
        ggml_op     op;
        ggml_op     src0;
        bool        glu_clamp;
    };
    auto gguf = get_gguf_ctx(arch, arch != LLM_ARCH_QWEN35);
    auto pair = get_model_and_ctx(gguf.get(), nullptr, seed, devs, split_mode, false, tensor_split);
    auto * model = pair.first.get();
    pair.second.reset();
    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const int32_t n_layer = llama_model_n_layer(model);
    const auto tokens = get_tokens(6, n_vocab, seed);

    std::vector<node_seen> nodes;
    llama_context_params params = llama_context_default_params();
    params.n_ctx = 64;
    params.n_batch = params.n_ubatch = 16;
    params.n_threads = params.n_threads_batch = 4;
    params.cb_eval = [](ggml_tensor * t, bool ask, void * data) {
        if (ask) {
            static_cast<std::vector<node_seen> *>(data)->push_back({ t->name, t->op,
                    t->src[0] ? t->src[0]->op : GGML_OP_NONE,
                    t->op == GGML_OP_GLU && ggml_get_glu_op(t) == GGML_GLU_OP_SWIGLU_CLAMP });
        }
        return false;
    };
    params.cb_eval_user_data = &nodes;
    llama_context_ptr ctx(llama_init_from_model(model, params));
    if (!ctx) {
        return false;
    }
    const bool lens = arch == LLM_ARCH_QWEN35 && split_mode == LLAMA_SPLIT_MODE_LAYER;
    if (lens) {
        const int32_t lens_layers[2] = { 0, n_layer - 1 };
        llama_set_lens_layers(ctx.get(), lens_layers, 2);
    }

    common_batch batch(ctx.get());
    for (int32_t pos = 0; pos < 5; pos++) {
        batch.add(tokens[pos], pos, 0, pos == 4);
    }
    if (llama_process(ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get())) {
        LOG_ERR("%s: the prompt's decode failed\n", __func__);
        return false;
    }
    if (lens) {
        const float * served = llama_get_logits_ith(ctx.get(), 4);
        const float * first  = llama_get_lens_ith(ctx.get(), 0, 4);
        const float * last   = llama_get_lens_ith(ctx.get(), 1, 4);
        if (!served || !first || !last || memcmp(served, last, n_vocab * sizeof(float)) != 0) {
            LOG_ERR("%s: the last layer's lens row is not the served row\n", __func__);
            return false;
        }
    }

    nodes.clear();
    batch.clear();
    batch.add(tokens[5], 5, 0, true);
    if (llama_process(ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get())) {
        LOG_ERR("%s: the one-row decode failed\n", __func__);
        return false;
    }

    // every node named "<name>-<layer>" is an `op` (whose first source is a `src0`, when one is given), and there is one
    const auto all_are = [&](const std::string & name, ggml_op op, ggml_op src0 = GGML_OP_NONE) {
        const std::string prefix = name + "-";
        int n = 0;
        for (const auto & node : nodes) {
            if (node.name.rfind(prefix, 0) != 0 ||
                    node.name.find_first_not_of("0123456789", prefix.size()) != std::string::npos) {
                continue;
            }
            n++;
            if (node.op != op || (src0 != GGML_OP_NONE && node.src0 != src0)) {
                LOG_ERR("%s: %s is a %s of a %s, and the CUDA folds match a %s%s%s\n", __func__, node.name.c_str(),
                        ggml_op_name(node.op), ggml_op_name(node.src0), ggml_op_name(op),
                        src0 != GGML_OP_NONE ? " of a " : "", src0 != GGML_OP_NONE ? ggml_op_name(src0) : "");
                return false;
            }
        }
        if (n == 0) {
            LOG_ERR("%s: the one-row graph has no node named %s<layer>\n", __func__, prefix.c_str());
        }
        return n > 0;
    };
    if (!all_are("conv_states", GGML_OP_GET_ROWS)) {
        return false;
    }
    if (arch == LLM_ARCH_QWEN35) {
        // LLAMA_GDN_QK_RSQRT=1 builds upstream's norm, which is not folded
        if (!ggml_env_switch("LLAMA_GDN_QK_RSQRT") && !all_are("qk_conv_l2", GGML_OP_L2_NORM)) {
            return false;
        }
        if (!all_are("state_predelta", GGML_OP_RESHAPE, GGML_OP_GET_ROWS)) {
            return false;
        }
    }
    if (arch == LLM_ARCH_GLM5NEXT) {
        if (!all_are("ffn_moe_up_clamped", GGML_OP_CLAMP) || !all_are("ffn_moe_gate_clamped", GGML_OP_CLAMP)) {
            return false;
        }
        for (const auto & node : nodes) {
            if (node.glu_clamp) {
                LOG_ERR("%s: %s is upstream's fused SwiGLU with a clamp; this architecture builds two CLAMPs\n",
                        __func__, node.name.c_str());
                return false;
            }
        }
    }
    return true;
}

static bool test_glm_mtp_index(const size_t seed, const std::vector<ggml_backend_dev_t> & devs,
        llama_split_mode split_mode, const std::vector<float> & tensor_split) {
    auto gguf = get_gguf_ctx(LLM_ARCH_GLM5NEXT, true, true);
    auto pair = get_model_and_ctx(gguf.get(), nullptr, seed, devs, split_mode, false, tensor_split, true);
    auto * model = pair.first.get();
    int counts[2] = {0, 0};
    int & scores = counts[0];
    llama_context_params params = llama_context_default_params();
    params.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
    params.n_ctx = 128;
    params.n_batch = params.n_ubatch = 64;
    params.n_threads = params.n_threads_batch = 4;
    params.cb_eval = [](ggml_tensor * t, bool ask, void * data) {
        if (ask && std::string(t->name).find("indexer_top_k_pools") == 0) ++static_cast<int *>(data)[0];
        if (ask && std::string(t->name).find("indexer_pool_members") == 0) ++static_cast<int *>(data)[1];
        return false;
    };
    params.cb_eval_user_data = counts;
    llama_context_ptr ctx(llama_init_from_model(model, params));
    // the reference builds its pool maps on the host (LLAMA_KPOOL_RESIDENT_LEGACY=1 around its decodes), so they are an
    // input of its graphs: a decode that computes them took the resident tables, and compares them with themselves
    int reference_maps = 0;
    params.cb_eval = [](ggml_tensor * t, bool ask, void * data) {
        if (ask && std::string(t->name) == "kpool_pool_cells") ++*static_cast<int *>(data);
        return false;
    };
    params.cb_eval_user_data = &reference_maps;
    llama_context_ptr reference(llama_init_from_model(model, params));
    if (!ctx || !reference || llama_mtp_draft_step(pair.second.get(), 0)) {
        return false;
    }
    const auto tokens = get_tokens(40, 128, seed);
    const int n_embd = llama_model_n_embd(model);
    std::vector<float> hidden(40*n_embd, 0.1f);
    get_mtp_logits(model, ctx.get(), tokens, hidden);
    const auto reference_logits = [&](const std::vector<llama_token> & input, llama_pos pos) {
        const char * old = std::getenv("LLAMA_KPOOL_RESIDENT_LEGACY");
        const std::string saved = old ? old : "";
        set_env("LLAMA_KPOOL_RESIDENT_LEGACY", "1");
        auto result = get_mtp_logits(model, reference.get(), input, hidden, pos);
        set_env("LLAMA_KPOOL_RESIDENT_LEGACY", old ? saved.c_str() : nullptr);
        if (reference_maps != 0) {
            throw std::runtime_error("the MTP reference made its pool maps on the device: LLAMA_KPOOL_RESIDENT_LEGACY was not read for its decode");
        }
        return result;
    };
    reference_logits(tokens, 0);
    const auto step = [&](int i, int pos, bool fresh) {
        scores = counts[1] = 0;
        if (!llama_mtp_draft_step(ctx.get(), i)) {
            return false;
        }
        if (!fresh && !ggml_env_switch("LLAMA_KPOOL_SELECTION_HOST_LEGACY") &&
                !static_cast<llama_memory_hybrid *>(llama_get_memory(ctx.get()))->get_mtp_index()->backend_row(0)) {
            LOG_ERR("MTP reuse did not retain an owned backend selection\n");
            return false;
        }
        const auto logits = get_mtp_logits(model, ctx.get(), {7}, hidden, pos);
        const auto expected = reference_logits({7}, pos);
        if ((scores > 0) != fresh || !std::all_of(logits.begin(), logits.end(), [](float x) { return std::isfinite(x); })) {
            LOG_ERR("MTP step %d at %d: expected fresh=%d, indexer ran %d times\n", i, pos, fresh, scores);
            return false;
        }
        if (!fresh && !ggml_env_switch("LLAMA_KPOOL_RESIDENT_LEGACY") && counts[1] != 0) {
            LOG_ERR("reused MTP step unnecessarily compressed %d pool tensors\n", counts[1]);
            return false;
        }
        // Reuse must keep writing raw keys. When scoring resumes,
        // the result must agree with a context that scored at every earlier step.
        if (fresh && nmse(expected, logits) > 1e-4) return false;
        return true;
    };
    if (!step(0, 40, true) || !step(1, 41, false) || !step(2, 42, false) ||
        !step(3, 43, true) || !step(4, 44, false) || !step(0, 45, true)) {
        return false;
    }
    if (!llama_memory_seq_rm(llama_get_memory(reference.get()), 0, 45, -1) ||
        !llama_memory_seq_rm(llama_get_memory(ctx.get()), 0, 45, -1) || !step(1, 45, true)) {
        return false;
    }
    const size_t size = llama_state_seq_get_size(ctx.get(), 0);
    std::vector<uint8_t> saved(size);
    if (llama_state_seq_get_data(ctx.get(), saved.data(), saved.size(), 0) != size ||
        llama_state_seq_set_data(ctx.get(), saved.data(), saved.size(), 0) != size || !step(2, 46, true)) {
        return false;
    }
    return step(-1, 47, true);
}

// With a budget covering every live pool, reusing the selection must compute the
// same logits as scoring it afresh. Distinct histories and reordered batch rows
// exercise the graph-to-sequence mapping, in both KV layouts.
static bool test_glm_mtp_index_sequences(const size_t seed, const std::vector<ggml_backend_dev_t> & devs,
        llama_split_mode split_mode, const std::vector<float> & tensor_split) {
    auto gguf = get_gguf_ctx(LLM_ARCH_GLM5NEXT, true, true);
    kv_writer ms(LLM_ARCH_GLM5NEXT, gguf.get());
    // Every live pool fits, but the padded KV span still needs the sparse path.
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_TOP_K, uint32_t(60));
    auto pair = get_model_and_ctx(gguf.get(), nullptr, seed, devs, split_mode, false, tensor_split, true);
    auto * model = pair.first.get();
    const int n_embd = llama_model_n_embd(model);
    for (const bool unified : {false, true}) {
      for (const bool split_capture : {false, true}) {
        int scores = 0;
        llama_context_params params = llama_context_default_params();
        params.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
        params.n_ctx = 256;
        params.n_seq_max = 3;
        params.n_batch = 128;
        params.n_ubatch = split_capture ? 1 : 128;
        params.kv_unified = unified;
        params.n_threads = params.n_threads_batch = 4;
        params.cb_eval = [](ggml_tensor * t, bool ask, void * data) {
            if (ask && std::string(t->name).find("indexer_top_k_pools") == 0) {
                ++*static_cast<int *>(data);
            }
            return false;
        };
        params.cb_eval_user_data = &scores;
        llama_context_ptr ctx(llama_init_from_model(model, params));
        params.cb_eval = nullptr;
        params.cb_eval_user_data = nullptr;
        llama_context_ptr reference(llama_init_from_model(model, params));
        if (!ctx || !reference) return false;
        const auto tokens = get_tokens(81, 128, seed);
        std::mt19937 rng(seed);
        std::normal_distribution<float> normal(0.0f, 0.1f);
        std::vector<float> hidden(tokens.size()*n_embd);
        for (auto & value : hidden) value = normal(rng);
        // Slot zero is idle: neither captured row index is its sequence ID.
        for (llama_seq_id seq : {1, 2}) {
            const auto begin = tokens.begin() + (seq == 1 ? 0 : 40);
            const std::vector<llama_token> history(begin, begin + 39 + seq);
            get_mtp_logits(model, ctx.get(), history, hidden, 0, {seq});
            get_mtp_logits(model, reference.get(), history, hidden, 0, {seq});
        }
        // Unequal histories: sequence 2 closes its pool at step 2, sequence 1 at
        // step 3. Stream-local selections also differ, exposing stream mix-ups.
        int scores_per_ubatch = 0;
        for (int step = 0; step < 5; ++step) {
            scores = 0;
            if (!llama_mtp_draft_step(ctx.get(), step)) return false;
            const std::vector<llama_seq_id> seqs = (bool(step % 2) != split_capture) ?
                std::vector<llama_seq_id>{2, 1} : std::vector<llama_seq_id>{1, 2};
            const std::vector<llama_pos> offsets = {seqs[0] - 1, seqs[1] - 1};
            const auto actual = get_mtp_logits(model, ctx.get(), {7, 19}, hidden, 40 + step, seqs, offsets);
            const auto expected = get_mtp_logits(model, reference.get(), {7, 19}, hidden, 40 + step, seqs, offsets);
            const int scoring_ubatches = step == 0 ? (split_capture ? 2 : 1) :
                step == 2 || step == 3 ? 1 : 0;
            if (step == 0) scores_per_ubatch = scores/scoring_ubatches;
            const int expected_scores = scores_per_ubatch*scoring_ubatches;
            if (scores_per_ubatch == 0 || scores != expected_scores ||
                    !std::all_of(actual.begin(), actual.end(), [](float x) { return std::isfinite(x); })) {
                LOG_ERR("MTP sequence reuse failed: unified=%d split_capture=%d step=%d scores=%d expected=%d nmse=%g\n",
                    unified, split_capture, step, scores, expected_scores, nmse(expected, actual));
                return false;
            }
            const size_t row_size = actual.size()/seqs.size();
            for (size_t row = 0; row < seqs.size(); ++row) {
                const std::vector<float> actual_row(actual.begin() + row*row_size, actual.begin() + (row + 1)*row_size);
                const std::vector<float> expected_row(expected.begin() + row*row_size, expected.begin() + (row + 1)*row_size);
                if (nmse(expected_row, actual_row) > 1e-4) {
                    LOG_ERR("MTP sequence %d logits differ: unified=%d step=%d nmse=%g\n",
                            seqs[row], unified, step, nmse(expected_row, actual_row));
                    return false;
                }
            }
        }
      }
    }
    return true;
}

// Reading a prompt allocates no compute memory past load: the buffers reserved at load hold every ubatch of a read that
// fills the cache, in the main context and in the MTP context, for one sequence and for the served shape (three slots
// in one unified cache). A decode step's graph slot is allocated on its first use, so decode is not measured here. The
// worst-case graph sized the pooled indexer's inputs on the cells of the reserve ubatch's sequences, empty at load, so
// every ubatch of a long read outgrew them and reallocated the buffer: a 45K first read on the Oct 29 lab box (run 12)
// made 44 cudaMallocHost calls (771 ms) and 44 cudaFreeHost calls (496 ms), and ran 5% slower than the pin.
static bool test_glm_reserve_holds_reads(const size_t seed, const std::vector<ggml_backend_dev_t> & devs,
        llama_split_mode split_mode, const std::vector<float> & tensor_split) {
    auto gguf = get_gguf_ctx(LLM_ARCH_GLM5NEXT, true, true);
    auto pair = get_model_and_ctx(gguf.get(), nullptr, seed, devs, split_mode, false, tensor_split, true);
    auto * model = pair.first.get();
    const int n_ctx   = 1024;
    const int n_embd  = llama_model_n_embd(model);
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const auto tokens = get_tokens(n_ctx, n_vocab, seed);
    std::vector<float> hidden((size_t) n_ctx*n_embd, 0.1f);

    const auto compute = [](const llama_context * ctx) {
        std::map<ggml_backend_buffer_type_t, size_t> sizes;
        for (const auto & [buft, mb] : llama_get_memory_breakdown(ctx)) {
            sizes[buft] = mb.compute;
        }
        return sizes;
    };

    for (const bool mtp : {false, true}) {
        for (const bool unified : {false, true}) {
            llama_context_params params = llama_context_default_params();
            if (mtp) params.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
            params.n_ctx      = n_ctx;
            params.n_batch    = n_ctx;
            params.n_ubatch   = 64;
            params.n_seq_max  = unified ? 3 : 1;
            params.kv_unified = unified;
            params.n_threads  = params.n_threads_batch = 4;
            llama_context_ptr ctx(llama_init_from_model(model, params));
            if (!ctx) return false;
            const auto at_load = compute(ctx.get());

            const auto decode = [&](int begin, int end) {
                llama_batch batch = llama_batch_init(end - begin, 0, 1);
                for (int pos = begin; pos < end; ++pos) {
                    batch_add(batch, tokens[pos], pos, {0}, pos == end - 1);
                }
                if (mtp) batch.embd = hidden.data() + (size_t) begin*n_embd;
                const int result = llama_decode(ctx.get(), batch);
                batch.embd = nullptr;
                llama_batch_free(batch);
                if (result) {
                    LOG_ERR("GLM reserve: mtp=%d unified=%d decode of %d:%d failed (%d)\n", mtp, unified, begin, end, result);
                    return false;
                }
                const auto now = compute(ctx.get());
                if (now != at_load) {
                    for (const auto & [buft, size] : now) {
                        LOG_ERR("GLM reserve: mtp=%d unified=%d %s compute buffer %zu bytes at load, %zu after %d:%d\n", mtp,
                                unified, ggml_backend_buft_name(buft), at_load.count(buft) ? at_load.at(buft) : 0, size, begin, end);
                    }
                    return false;
                }
                return true;
            };

            if (!decode(0, n_ctx)) return false;
        }
    }
    return true;
}

// a server with more slots than a graph slot holds rows (graph_slot_max_tokens, 32 rows) must load, which reserves the
// slot graph over every stream of the cache, then answer the two-token probe common_context_can_seq_rm makes at load
// and a step with one row on every sequence
static bool test_glm_many_slots(const size_t seed, const std::vector<ggml_backend_dev_t> & devs,
        llama_split_mode split_mode, const std::vector<float> & tensor_split) {
    auto gguf = get_gguf_ctx(LLM_ARCH_GLM5NEXT, true, true);
    auto pair = get_model_and_ctx(gguf.get(), nullptr, seed, devs, split_mode, false, tensor_split, true);
    auto * model = pair.first.get();
    const int n_seq   = 40;
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const auto tokens = get_tokens(n_seq, n_vocab, seed);

    llama_context_params params = llama_context_default_params();
    params.n_ctx      = 64*n_seq;
    params.n_batch    = 512;
    params.n_ubatch   = 512;
    params.n_seq_max  = n_seq;
    params.kv_unified = false;
    params.n_threads  = params.n_threads_batch = 4;
    llama_context_ptr ctx(llama_init_from_model(model, params));
    if (!ctx) {
        LOG_ERR("GLM many slots: no context for %d sequences\n", n_seq);
        return false;
    }

    const auto finite = [&](int32_t row, const char * what) {
        const float * logits = llama_get_logits_ith(ctx.get(), row);
        if (!logits) {
            LOG_ERR("GLM many slots: %s: no logits for row %d\n", what, row);
            return false;
        }
        for (int v = 0; v < n_vocab; ++v) {
            if (!std::isfinite(logits[v])) {
                LOG_ERR("GLM many slots: %s: row %d logit %d is %f\n", what, row, v, logits[v]);
                return false;
            }
        }
        return true;
    };

    // the probe at load: two tokens on one sequence, the last an output, the other sequences idle
    {
        llama_batch batch = llama_batch_init(2, 0, 1);
        batch_add(batch, tokens[0], 0, {0}, false);
        batch_add(batch, tokens[1], 1, {0}, true);
        const int result = llama_decode(ctx.get(), batch);
        llama_batch_free(batch);
        if (result) {
            LOG_ERR("GLM many slots: the two-token probe failed (%d)\n", result);
            return false;
        }
        if (!finite(-1, "probe")) return false;
    }
    llama_memory_clear(llama_get_memory(ctx.get()), true);

    // one row on every sequence, every row an output: more rows than a graph slot holds
    {
        llama_batch batch = llama_batch_init(n_seq, 0, 1);
        for (int s = 0; s < n_seq; ++s) {
            batch_add(batch, tokens[s], 0, {s}, true);
        }
        const int result = llama_decode(ctx.get(), batch);
        llama_batch_free(batch);
        if (result) {
            LOG_ERR("GLM many slots: the step over %d sequences failed (%d)\n", n_seq, result);
            return false;
        }
        for (int s = 0; s < n_seq; ++s) {
            if (!finite(s, "every sequence")) return false;
        }
    }
    return true;
}

// the MTP head's masked nextn rows (llama_set_embeddings_nextn masked) are the batch's output rows: the drafter reads
// them by output index. A step with fewer outputs than rows must hand back the output rows, not the first rows, on one
// device and across data-parallel ranks alike
static bool test_glm_nextn_masked_rows(const size_t seed, const std::vector<ggml_backend_dev_t> & devs,
        llama_split_mode split_mode, const std::vector<float> & tensor_split, bool q8_cache) {
    auto gguf = get_gguf_ctx(LLM_ARCH_GLM5NEXT, true, true);
    if (split_mode == LLAMA_SPLIT_MODE_DATA) {
        kv_writer ms(LLM_ARCH_GLM5NEXT, gguf.get());
        ms.add_kv(LLM_KV_EXPERT_COUNT, uint32_t(8));
        ms.add_kv(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT, uint32_t(64));
        ms.add_kv(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH, uint32_t(128));
    }
    auto pair = get_model_and_ctx(gguf.get(), nullptr, seed, devs, split_mode, false, tensor_split, true);
    auto * model = pair.first.get();
    const int n_embd  = llama_model_n_embd_out(model);
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const auto tokens = get_tokens(2, n_vocab, seed);

    const auto make = [&](bool masked) {
        llama_context_params params = llama_context_default_params();
        params.ctx_type   = LLAMA_CONTEXT_TYPE_MTP;
        params.n_ctx      = 256;
        params.n_batch    = 64;
        params.n_ubatch   = 64;
        params.n_seq_max  = 2;
        params.kv_unified = split_mode != LLAMA_SPLIT_MODE_DATA;
        if (split_mode == LLAMA_SPLIT_MODE_DATA) { params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED; }
        if (q8_cache) { params.type_k = params.type_v = params.type_s = params.type_idx = GGML_TYPE_Q8_0; }
        params.n_threads  = params.n_threads_batch = 4;
        llama_context_ptr ctx(llama_init_from_model(model, params));
        if (!ctx) throw std::runtime_error("nextn masked rows: failed to create a context");
        llama_set_embeddings_nextn(ctx.get(), true, masked);
        return ctx;
    };
    llama_context_ptr masked = make(true), unmasked = make(false);

    // two rows on one sequence with distinct hidden inputs, only the second an output
    std::vector<float> hidden((size_t) 2*n_embd);
    for (int i = 0; i < 2; ++i) {
        for (int k = 0; k < n_embd; ++k) hidden[(size_t) i*n_embd + k] = 0.05f*(i + 1)*(1.0f + (k % 7));
    }
    const auto rows = [&](llama_context * ctx, const char * what) -> const float * {
        llama_batch batch = llama_batch_init(2, 0, 1);
        batch_add(batch, tokens[0], 0, {0}, false);
        batch_add(batch, tokens[1], 1, {0}, true);
        batch.embd = hidden.data();
        const int rc = llama_decode(ctx, batch);
        batch.embd = nullptr;
        llama_batch_free(batch);
        if (rc) {
            LOG_ERR("GLM nextn masked rows: the %s decode failed (%d)\n", what, rc);
            return nullptr;
        }
        return llama_get_embeddings_nextn(ctx);
    };
    const float * h_masked   = rows(masked.get(),   "masked");
    const float * h_unmasked = rows(unmasked.get(), "unmasked");
    if (!h_masked || !h_unmasked) return false;

    // the one masked row is the output row, the unmasked read's second row
    const std::vector<float> got(h_masked, h_masked + n_embd);
    const std::vector<float> want(h_unmasked + n_embd, h_unmasked + 2*n_embd);
    const std::vector<float> other(h_unmasked, h_unmasked + n_embd);
    if (nmse(other, want) < 1e-6) {
        LOG_ERR("GLM nextn masked rows: the two rows do not differ, the check cannot tell them apart\n");
        return false;
    }
    const double err = nmse(got, want);
    if (!(err < 1e-6)) {
        LOG_ERR("GLM nextn masked rows: the masked row is not the output row (nmse %g against it, %g against the first row)\n",
                err, nmse(got, other));
        return false;
    }
    return true;
}

static bool test_glm_short_dense(const size_t seed, const std::vector<ggml_backend_dev_t> & devs,
        llama_split_mode split_mode, const std::vector<float> & tensor_split) {
    auto gguf = get_gguf_ctx(LLM_ARCH_GLM5NEXT, true, true);
    kv_writer ms(LLM_ARCH_GLM5NEXT, gguf.get());
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_TOP_K, uint32_t(256));
    auto pair = get_model_and_ctx(gguf.get(), nullptr, seed, devs, split_mode, false, tensor_split, true);
    auto * model = pair.first.get();
    const int n_embd = llama_model_n_embd(model);
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const auto tokens = get_tokens(384, n_vocab, seed);
    const auto previous_tokens = get_tokens(384, n_vocab, seed + 1);
    std::mt19937 rng(seed);
    std::normal_distribution<float> normal(0.0f, 0.1f);
    std::vector<float> hidden(tokens.size()*n_embd);
    for (auto & value : hidden) value = normal(rng);

    for (const bool mtp : {false, true}) {
        int scores = 0;
        llama_context_params params = llama_context_default_params();
        if (mtp) params.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
        params.n_ctx = params.n_batch = params.n_ubatch = 512;
        params.n_threads = params.n_threads_batch = 4;
        llama_context_ptr reference(llama_init_from_model(model, params));
        params.cb_eval = [](ggml_tensor * t, bool ask, void * data) {
            if (ask && std::string(t->name).find("indexer_top_k_pools") == 0) {
                ++*static_cast<int *>(data);
            }
            return false;
        };
        params.cb_eval_user_data = &scores;
        llama_context_ptr ctx(llama_init_from_model(model, params));
        llama_context_ptr restored(llama_init_from_model(model, params));
        if (!ctx || !reference || !restored) return false;
        const auto decode = [&](llama_context * context, int begin, int end, llama_seq_id seq_id = 0,
                const std::vector<llama_token> * history = nullptr) {
            const auto & input = history ? *history : tokens;
            llama_batch batch = llama_batch_init(end - begin, 0, 1);
            for (int pos = begin; pos < end; ++pos) {
                batch_add(batch, input[pos], pos, {seq_id}, true);
            }
            if (mtp) batch.embd = hidden.data() + begin*n_embd;
            const int result = llama_decode(context, batch);
            batch.embd = nullptr;
            llama_batch_free(batch);
            if (result) throw std::runtime_error("short dense decode failed");
            std::vector<float> logits;
            for (int pos = begin; pos < end; ++pos) {
                const float * row = llama_get_logits_ith(context, pos - begin);
                logits.insert(logits.end(), row, row + n_vocab);
            }
            return logits;
        };
        const auto expected = decode(reference.get(), 0, tokens.size());
        const auto check = [&](llama_context * context, int begin, int end, llama_seq_id seq_id = 0) {
            scores = 0;
            const auto actual = decode(context, begin, end, seq_id);
            const std::vector<float> wanted(expected.begin() + begin*n_vocab, expected.begin() + end*n_vocab);
            const double error = nmse(wanted, actual);
            if ((scores > 0) != (end > 256) || !(error <= 1e-4)) {
                LOG_ERR("GLM short dense: mtp=%d seq=%d span=%d:%d scores=%d nmse=%g\n",
                        mtp, seq_id, begin, end, scores, error);
                return false;
            }
            return true;
        };
        if (!check(ctx.get(), 0, 128)) return false;
        const size_t size = llama_state_seq_get_size(ctx.get(), 0);
        std::vector<uint8_t> saved(size);
        if (llama_state_seq_get_data(ctx.get(), saved.data(), size, 0) != size) return false;
        for (const auto & span : {std::pair<int, int>{128, 255}, {255, 256}, {256, 257},
                {257, 258}, {258, 259}, {259, 260}, {260, 384}}) {
            if (!check(ctx.get(), span.first, span.second)) return false;
        }
        // Restore the short prefix after its pooled keys were deferred, into both
        // a fresh context and one whose later sparse decode cleared the dirty bit.
        for (auto * context : {restored.get(), ctx.get()}) {
            if (llama_state_seq_set_data(context, saved.data(), size, 0) != size ||
                !check(context, 128, 384)) return false;
        }
        // Copy deferred and built pooled keys, into fresh and reused slots.
        // The child alone continues into sparse attention.
        for (const bool unified : {false, true}) {
            auto copy_params = params;
            copy_params.n_seq_max = 2;
            copy_params.kv_unified = unified;
            copy_params.n_ctx = unified ? 512 : 1024;
            for (const int prefix : {128, 255, 300}) {
              for (const bool reused : {false, true}) {
                llama_context_ptr copied(llama_init_from_model(model, copy_params));
                llama_context_ptr original(llama_init_from_model(model, copy_params));
                if (!copied || !original) return false;
                if (reused) {
                    scores = 0;
                    decode(copied.get(), 0, previous_tokens.size(), 1, &previous_tokens);
                    if (scores == 0 || !llama_memory_seq_rm(llama_get_memory(copied.get()), 1, -1, -1)) return false;
                }
                if (!check(copied.get(), 0, prefix) ||
                        !check(original.get(), 0, prefix)) return false;
                llama_memory_seq_cp(llama_get_memory(copied.get()), 0, 1, -1, -1);
                scores = 0;
                const auto before_copy = decode(original.get(), prefix, tokens.size());
                if (scores == 0) return false;
                scores = 0;
                const auto after_copy = decode(copied.get(), prefix, tokens.size(), 1);
                // Same inputs and batch geometry must preserve the continuation, including selection.
                if (scores == 0 || before_copy != after_copy) {
                    LOG_ERR("GLM prefix copy: mtp=%d unified=%d prefix=%d reused=%d scores=%d nmse=%g\n",
                            mtp, unified, prefix, reused, scores, nmse(before_copy, after_copy));
                    return false;
                }
              }
            }
        }
    }
    return true;
}

// set or clear an environment variable a library reads when a model or context is created
static void set_env(const char * name, const char * value) {
#ifdef _WIN32
    _putenv_s(name, value ? value : "");
#else
    if (value) {
        setenv(name, value, 1);
    } else {
        unsetenv(name);
    }
#endif
}

// The name "glm5-next" loads the fork's glm5next (llm_arch_from_string). While upstream's architecture of that name is
// made, saved and loaded, LLAMA_GLM5_UPSTREAM gives the name to upstream's implementation.
struct glm5_upstream_scope {
    glm5_upstream_scope(const llm_arch arch) : on(arch == LLM_ARCH_GLM5_NEXT) {
        if (on) {
            set_env("LLAMA_GLM5_UPSTREAM", "1");
        }
    }
    ~glm5_upstream_scope() {
        if (on) {
            set_env("LLAMA_GLM5_UPSTREAM", nullptr);
        }
    }
    const bool on;
};

// A prompt read in equal steps reuses its graph. On the served GLM-5.3 Flash (two RTX PRO 6000, -sm tensor, 2,048-row
// steps) every step of a 16K read rebuilt the main and the MTP context's graphs and planned their memory again, 45-60 ms
// of the 555 ms a step took, with both cards idle (the host-gap trace of Nov 1, 2026): the cache length grew by a step's
// rows each step, and the graph's shapes with it. A step of at least 256 rows whose indexer scores has its cache length
// padded to LLAMA_KV_PAD_PREFILL cells and its pool count to as many pools, so a read builds a graph once a step of
// cells. A step that reads densely, and a decode step, keep 256. On the CPU its logits are the 256-padded read's byte for
// byte. On a GPU the attention kernel can depend on the cache length: CUDA's flash attention gathers a row's live cells
// once the cache holds 4,096 cells or more, and sums them in another order than the dense kernel below that
// (ggml_cuda_flash_attn_ext_mma_f16_shall_use_sparse), so a padded step can round differently there. Its logits then meet
// the plan's bar for a lossless change against the 256-padded read: a mean KL divergence of at most 1e-6, and the same top
// token on every row whose two top tokens are not tied (mean_kld). A tie's order is the summation's, so any rounding can
// swap it, and 4,096 rows cannot resolve the plan's 99.99% identical top tokens: a change at exactly that rate leaves a
// swapped row in a third of such reads. Two RTX PRO 6000 at a 3:7 tensor split swapped one row of 4,096 at a mean KL
// divergence of 3.2e-17 (top_k 8, box 58698413, Nov 1, 2026).
static bool test_glm_prefill_reuse(const size_t seed, const std::vector<ggml_backend_dev_t> & devs,
        llama_split_mode split_mode, const std::vector<float> & tensor_split) {
    struct read_result {
        std::vector<float> logits;  // every row of seq 0's read
        int32_t n_reused   = 0;     // graphs reused over the context's reads
        int     n_scored   = 0;     // the read's steps whose indexer scored
        int64_t n_kv_decode = -1;   // the cache length of the one-token decode after the read
    };

    for (const uint32_t top_k : {8u, 256u}) {
        auto gguf = get_gguf_ctx(LLM_ARCH_GLM5NEXT, true, true);
        kv_writer ms(LLM_ARCH_GLM5NEXT, gguf.get());
        ms.add_kv(LLM_KV_ATTENTION_INDEXER_TOP_K, top_k);
        auto pair = get_model_and_ctx(gguf.get(), nullptr, seed, devs, split_mode, false, tensor_split, true);
        auto * model = pair.first.get();
        const int n_embd  = llama_model_n_embd(model);
        const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
        // byte for byte where every device is a CPU
        const bool exact = on_host(devs);

        // top_k 8: every step scores. top_k 256 (the indexer scores past 259 cells): the first step reads densely
        const int n_read   = top_k == 8 ? 4096 : 1024;
        const int n_before = 1000;
        const auto tokens = get_tokens(n_read + 1, n_vocab, seed);
        const auto before = get_tokens(n_before, n_vocab, seed + 1);
        std::mt19937 rng(seed);
        std::normal_distribution<float> normal(0.0f, 0.1f);
        std::vector<float> hidden((size_t) (n_read + 1)*n_embd);
        for (auto & value : hidden) value = normal(rng);

        struct probe {
            bool    decoding = false;
            int     n_scored = 0;
            int64_t n_kv     = -1;
        };

        // seq 0 reads n_read tokens in 256-row steps, after n_before tokens of seq 1 when unified; then one token decodes
        const auto read = [&](const char * pad, bool mtp, bool unified, uint32_t n_ubatch) {
            fprintf(stderr, "GLM prefill reuse case: top_k=%u pad=%s mtp=%d unified=%d ubatch=%u\n", top_k, pad, mtp, unified, n_ubatch);
            set_env("LLAMA_KV_PAD_PREFILL", pad);
            // the prefill padding alone: a pool that sequences share pads to 256 cells a sequence besides
            // (test_glm_shared_pool_reuse)
            set_env("LLAMA_KV_PAD_SHARED_LEGACY", "1");
            probe pr;
            llama_context_params params = llama_context_default_params();
            if (mtp) params.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
            params.n_ctx      = 8192;
            params.n_batch    = n_read;
            params.n_ubatch   = n_ubatch;
            params.n_seq_max  = unified ? 2 : 1;
            params.kv_unified = unified;
            params.n_threads  = params.n_threads_batch = 4;
            params.cb_eval = [](ggml_tensor * t, bool ask, void * data) {
                auto * p = static_cast<probe *>(data);
                if (ask && p->decoding && std::string(t->name).rfind("kpool_cand_mask", 0) == 0) {
                    p->n_kv = t->ne[0];
                }
                if (ask && p->decoding && t->op == GGML_OP_KPOOL_MASK &&
                        ggml_get_op_params_i32(t, 0) == GGML_KPOOL_MASK_INDICES) {
                    p->n_kv = t->src[4] ? ggml_get_op_params_i32(t, 3) : t->src[0]->ne[0];
                }
                if (ask && !p->decoding && std::string(t->name).rfind("indexer_top_k_pools", 0) == 0) {
                    ++p->n_scored;
                }
                return false;
            };
            params.cb_eval_user_data = &pr;
            llama_context_ptr ctx(llama_init_from_model(model, params));
            set_env("LLAMA_KV_PAD_PREFILL", nullptr);
            set_env("LLAMA_KV_PAD_SHARED_LEGACY", nullptr);
            if (!ctx) throw std::runtime_error("prefill reuse: failed to create a context");

            const auto decode = [&](const std::vector<llama_token> & input, int begin, int end, llama_seq_id seq,
                    bool logits) {
                llama_batch batch = llama_batch_init(end - begin, 0, 1);
                for (int pos = begin; pos < end; ++pos) {
                    batch_add(batch, input[pos], pos, {seq}, logits || pos == end - 1);
                }
                if (mtp) batch.embd = hidden.data() + (size_t) begin*n_embd;
                const int result = llama_decode(ctx.get(), batch);
                batch.embd = nullptr;
                llama_batch_free(batch);
                if (result) throw std::runtime_error("prefill reuse: decode failed");
            };

            read_result res;
            if (unified) decode(before, 0, n_before, 1, false);
            const int scored_before = pr.n_scored;
            decode(tokens, 0, n_read, 0, true);
            // the indexer scores once a step in every DSA layer: steps that scored, times those layers
            res.n_scored = pr.n_scored - scored_before;
            for (int i = 0; i < n_read; ++i) {
                const float * row = llama_get_logits_ith(ctx.get(), i);
                res.logits.insert(res.logits.end(), row, row + n_vocab);
            }
            res.n_reused = llama_perf_context(ctx.get()).n_reused;
            pr.decoding = true;
            decode(tokens, n_read, n_read + 1, 0, true);
            res.n_kv_decode = pr.n_kv;
            return res;
        };

        for (const bool mtp : {false, true}) {
            for (const bool unified : {false, true}) {
                const read_result fine    = read("0", mtp, unified, 256);
                const read_result coarse  = read("1024", mtp, unified, 256);
                const read_result chunked = read("0", mtp, unified, 128);
                const read_result wide    = read("0", mtp, unified, 512);

                // Padded to 256, every step's cache is a step longer: nothing is reused. Padded to 1,024, a graph is built
                // when the cache crosses a multiple of 1,024 cells, or seq 0's positions a multiple of 256 pools (seq 1's
                // 1,000 cells put both crossings on the same steps), for a step that reads densely, for the first scoring
                // step after one (it rebuilds the keys the dense step left stale), and for a step of another length.
                //   top_k 8:   seq 0's 16 steps cross at steps 1, 5, 9 and 13: 12 reused. Seq 1's 1,000 tokens (256, 256,
                //              256, 232 rows) build at step 1, reuse 2 and 3, and build again for the shorter step 4: 2
                //   top_k 256: seq 0 reads step 1 densely, rebuilds stale keys at step 2, builds the steady graph at step
                //              3 and reuses it at step 4: 1. Unified, seq 1 does that and ends on its shorter step: 0
                //              reused, and seq 0, with no dense step of its own and 2,048 cells to the end, reuses 3
                // The main context's KDA layers bake the recurrent state's zero row into the graph (build_rs): a sequence's
                // first step starts from it and its second does not, so the second step of each sequence builds too,
                // whatever the padding. The MTP head has no KDA layer, and over a memory that keeps a row a sequence
                // (LLAMA_RS_FIXED_ROWS=1) the main context's graph holds no zero row either: the memory clears a new
                // sequence's row before the decode, so the main context reuses as the MTP head does
                const bool rs_z_baked = !mtp && !ggml_env_switch("LLAMA_RS_FIXED_ROWS");
                const int want_fine   = 0;
                const int want_coarse = !rs_z_baked ?
                    (top_k == 8 ? (unified ? 14 : 12) : (unified ? 3 : 1)) :
                    (top_k == 8 ? (unified ? 12 : 11) : (unified ? 2 : 1));

                const int64_t want_n_kv = GGML_PAD(n_read + 1 + (unified ? n_before : 0), 256);

                bool ok = true;
                if (exact && fine.logits != coarse.logits) {
                    LOG_ERR("GLM prefill reuse: top_k=%u mtp=%d unified=%d: logits differ, nmse=%g\n", top_k, mtp,
                            unified, nmse(fine.logits, coarse.logits));
                    ok = false;
                }
                if (!exact) {
                    // a device: glm_device_logits_agree's bar. A row is a position of the read
                    const std::string pairing = "top_k=" + std::to_string(top_k) + " mtp=" + std::to_string(mtp) +
                        " unified=" + std::to_string(unified) + ": padded to 256 against padded to 1024";
                    if (!glm_device_logits_agree("prefill reuse", pairing, seed, fine.logits, coarse.logits, n_vocab)) {
                        ok = false;
                    }
                }
                if (exact && chunked.logits != wide.logits) {
                    LOG_ERR("GLM prefill chunk invariance: top_k=%u mtp=%d unified=%d: 128-row and 512-row chunks differ, "
                            "nmse=%g\n", top_k, mtp, unified, nmse(chunked.logits, wide.logits));
                    ok = false;
                }
                if (fine.n_reused != want_fine || coarse.n_reused != want_coarse) {
                    LOG_ERR("GLM prefill reuse: top_k=%u mtp=%d unified=%d: reused %d graphs padded to 256 (want %d), "
                            "%d padded to 1024 (want %d)\n", top_k, mtp, unified, fine.n_reused, want_fine,
                            coarse.n_reused, want_coarse);
                    ok = false;
                }
                if (coarse.n_scored != fine.n_scored) {
                    LOG_ERR("GLM prefill reuse: top_k=%u mtp=%d unified=%d: the indexer scored %d times padded to 1024, "
                            "%d padded to 256\n", top_k, mtp, unified, coarse.n_scored, fine.n_scored);
                    ok = false;
                }
                if (fine.n_kv_decode != want_n_kv || coarse.n_kv_decode != want_n_kv) {
                    LOG_ERR("GLM prefill reuse: top_k=%u mtp=%d unified=%d: a decode step read %lld cells padded to "
                            "256 and %lld padded to 1024 (want %lld)\n", top_k, mtp, unified,
                            (long long) fine.n_kv_decode, (long long) coarse.n_kv_decode, (long long) want_n_kv);
                    ok = false;
                }
                if (!ok) return false;
            }
        }
    }
    return true;
}

// Sequences that share a pool decode on a graph that holds. 23 sessions of the served GLM-5.3 Flash, each verifying two
// drafts a step (four RTX PRO 6000, -sm tensor, Nov 9, 2026), grew their pool ~50 cells a step and crossed its 256-cell
// padding every 5 steps, where one sequence crosses it every ~115: each crossing built the target's graph and the MTP
// head's again and planned their memory, in 34 of 64 decodes at 8.7 ms each, the cards 23-30% busy through a 150 ms step.
// A pool that sequences share is padded to 256 cells a sequence, the next power of two, wherever its indexer scores, so
// it crosses as often a step as one sequence does; LLAMA_KV_PAD_SHARED_LEGACY=1 keeps 256. The cells that adds are empty
// and the sparse attention reads none of them: the logits are the 256-padded pool's, byte for byte on the CPU and to
// test_glm_prefill_reuse's bar on a GPU, whose attention kernel can depend on the cache length.
// The pool's maps are one stream a sequence, each from its sequence's view (llama_kpool_shared_streams), where they
// packed every sequence's pools on one axis and were built from every cell; LLAMA_KPOOL_SHARED_VIEWS_LEGACY=1 packs
// them. The logits are those of a cache a sequence (kv_unified off), byte for byte on the CPU. The packed maps' are
// not: 28 of this test's 960 rows differ from a cache a sequence by up to 0.5% of the largest logit and 102 by a
// rounding step (CPU, seed 1234, Nov 10, 2026), so they are held to the GPU's bar, with two top tokens within 1% of
// the row's largest logit counted as tied: over the seeds a row's two top tokens are that close, and the packed maps
// swap them (seeds 14 and 10963435).
// On the CPU the prompt is read 4 positions a decode. The CPU's attention kernel is another one from 64 rows a stream
// (GGML_FA_TILE_Q, the tiled path), whose output is 2e-4 of its size off the vector path's. Read at once, the prompt's
// 128 rows are one stream of the shared pool and 16 a stream of a cache a sequence, so the two took different kernels;
// what that left in the recurrent layer's state moved one row of the first step by a rounding step for about half the
// seeds (7 of 14), and the byte compare failed for those. On a GPU, where the compare is to a bar and the kernels'
// row counts are others, the prompt stays one decode.
// The streams' bank takes a decode's changes as one input its device writes before the maps are made (ggml_kpool_patch),
// where each change was a write of its own; LLAMA_KPOOL_PATCH_INPUT_LEGACY=1 writes them. The logits are the same again.
// LLAMA_KV_SEQ_CELLS=N sizes the graph reserved at load for sequences of N positions, a stream each, where it held the
// whole pool's pools for one sequence. Inside the limit a step is the unlimited pool's: the same logits and reuses. A
// sequence that passes it scores more pools by as many rows as the reserve holds, so its graph is allocated when it
// comes, and the logits are the unlimited pool's again. LLAMA_KV_PAD_PREFILL=256 goes with it here: at its default of
// 8,192 cells the widest padding is this 8,192-cell pool's pools and the reserve stays the pool's.
// Cell capacity must not move a step across the direct-index cutoff. Both capacity modes keep this pool's 2,048-cell
// view below the 4,096-cell floor, with no direct indices and the same logits.
static bool test_glm_shared_pool_reuse(const size_t seed, const std::vector<ggml_backend_dev_t> & devs,
        llama_split_mode split_mode, const std::vector<float> & tensor_split) {
    auto gguf = get_gguf_ctx(LLM_ARCH_GLM5NEXT, true, true);
    kv_writer ms(LLM_ARCH_GLM5NEXT, gguf.get());
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_TOP_K, uint32_t(8)); // the indexer scores from the first step
    auto pair = get_model_and_ctx(gguf.get(), nullptr, seed, devs, split_mode, false, tensor_split, true);
    auto * model = pair.first.get();
    const int n_embd  = llama_model_n_embd(model);
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const bool on_cpu = std::all_of(devs.begin(), devs.end(), [](ggml_backend_dev_t dev) {
        return ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU;
    });

    // 8 sequences read 16 tokens each, then decode 40 steps of 3 rows each (a sampled token and two drafts): 24 cells a
    // step, from 128 cells to 1,088
    const int n_seq    = 8;
    const int n_prompt = 16;
    // the prompt's positions a decode: on the CPU 32 rows, under the 64 of its tiled attention
    const int n_prompt_rows = on_cpu ? 4 : n_prompt;
    const int n_rows   = 3;
    const int n_steps  = 40;
    const int n_pos    = n_prompt + n_steps*n_rows;
    const auto tokens = get_tokens(n_seq*n_pos, n_vocab, seed);
    std::mt19937 rng(seed);
    std::normal_distribution<float> normal(0.0f, 0.1f);
    std::vector<float> hidden((size_t) n_seq*n_prompt*n_embd);
    for (auto & value : hidden) value = normal(rng);

    struct read_result {
        std::vector<float> logits;     // every row of every step
        int32_t            n_reused = 0;
        std::set<int64_t>  n_kv;       // the cache lengths the steps' graphs read
        bool               patched = false; // a step's graph wrote the bank's changes
        bool               direct  = false; // a step's attention read through indices
    };
    struct seen_nodes {
        std::set<int64_t> n_kv;
        bool              patched = false;
        bool              direct  = false;
    };

    const auto read = [&](bool legacy, bool mtp, bool packed = false, bool writes = false, bool unified = true,
            const char * seq_cells = nullptr, bool cells_legacy = false) {
        set_env("LLAMA_KV_PAD_SHARED_LEGACY", legacy ? "1" : nullptr);
        set_env("LLAMA_KV_CELL_CAPACITY_LEGACY", cells_legacy ? "1" : nullptr);
        set_env("LLAMA_KPOOL_SHARED_VIEWS_LEGACY", packed ? "1" : nullptr);
        set_env("LLAMA_KPOOL_PATCH_INPUT_LEGACY", writes ? "1" : nullptr);
        set_env("LLAMA_KV_SEQ_CELLS", seq_cells);
        set_env("LLAMA_KV_PAD_PREFILL", seq_cells ? "256" : nullptr);
        seen_nodes seen;
        std::set<int64_t> & n_kv = seen.n_kv;
        llama_context_params params = llama_context_default_params();
        if (mtp) params.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
        params.n_ctx      = 8192;
        params.n_batch    = 512;
        params.n_ubatch   = 512;
        params.n_seq_max  = n_seq;
        params.kv_unified = unified;
        params.n_threads  = params.n_threads_batch = 4;
        params.cb_eval = [](ggml_tensor * t, bool ask, void * data) {
            const std::string name = t->name;
            if (ask && (name.rfind("kpool_cand_mask", 0) == 0 || name == "kpool_mask_cells")) {
                static_cast<seen_nodes *>(data)->n_kv.insert(t->ne[0]);
            }
            if (ask && name == "kpool_resident_patched") {
                static_cast<seen_nodes *>(data)->patched = true;
            }
            if (ask && name.rfind("kpool_indices", 0) == 0) {
                static_cast<seen_nodes *>(data)->direct = true;
            }
            return false;
        };
        params.cb_eval_user_data = &seen;
        llama_context_ptr ctx(llama_init_from_model(model, params));
        set_env("LLAMA_KV_PAD_SHARED_LEGACY", nullptr);
        set_env("LLAMA_KV_CELL_CAPACITY_LEGACY", nullptr);
        set_env("LLAMA_KPOOL_SHARED_VIEWS_LEGACY", nullptr);
        set_env("LLAMA_KPOOL_PATCH_INPUT_LEGACY", nullptr);
        set_env("LLAMA_KV_SEQ_CELLS", nullptr);
        set_env("LLAMA_KV_PAD_PREFILL", nullptr);
        if (!ctx) throw std::runtime_error("shared pool reuse: failed to create a context");
        if (seq_cells && llama_n_ctx_seq(ctx.get()) != (uint32_t) atoi(seq_cells)) {
            throw std::runtime_error("shared pool reuse: the limit of a sequence's cells was not taken");
        }

        // positions [begin, end) of every sequence, a sequence's rows together
        const auto decode = [&](int begin, int end) {
            llama_batch batch = llama_batch_init(n_seq*(end - begin), 0, 1);
            for (int seq = 0; seq < n_seq; ++seq) {
                for (int pos = begin; pos < end; ++pos) {
                    batch_add(batch, tokens[(size_t) seq*n_pos + pos], pos, {seq}, true);
                }
            }
            if (mtp) batch.embd = hidden.data();
            const int result = llama_decode(ctx.get(), batch);
            batch.embd = nullptr;
            llama_batch_free(batch);
            if (result) throw std::runtime_error("shared pool reuse: decode failed");
        };

        read_result res;
        for (int pos = 0; pos < n_prompt; pos += n_prompt_rows) {
            decode(pos, pos + n_prompt_rows);
        }
        const int32_t n_reused_prompt = llama_perf_context(ctx.get()).n_reused;
        n_kv.clear();
        for (int step = 0; step < n_steps; ++step) {
            decode(n_prompt + step*n_rows, n_prompt + (step + 1)*n_rows);
            for (int i = 0; i < n_seq*n_rows; ++i) {
                const float * row = llama_get_logits_ith(ctx.get(), i);
                res.logits.insert(res.logits.end(), row, row + n_vocab);
            }
        }
        res.n_reused = llama_perf_context(ctx.get()).n_reused - n_reused_prompt;
        res.n_kv     = n_kv;
        res.patched  = seen.patched;
        res.direct   = seen.direct;
        return res;
    };

    // one sequence of the pool reads 2,560 positions, 512 rows a decode: every row's logits. Under a limit of 256 its
    // last decodes score 1,026 pools where the reserve holds 592 (a run of 64, the widest padding's 512, 2 a sequence)
    const int n_long      = 2560;
    const int n_long_rows = 512;
    const auto tokens_long = get_tokens(n_long, n_vocab, seed + 2);
    std::vector<float> hidden_long((size_t) n_long_rows*n_embd);
    for (auto & value : hidden_long) value = normal(rng);

    const auto read_long = [&](bool mtp, const char * seq_cells) {
        set_env("LLAMA_KV_SEQ_CELLS", seq_cells);
        set_env("LLAMA_KV_PAD_PREFILL", "256");
        llama_context_params params = llama_context_default_params();
        if (mtp) params.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
        params.n_ctx      = 8192;
        params.n_batch    = n_long_rows;
        params.n_ubatch   = n_long_rows;
        params.n_seq_max  = n_seq;
        params.kv_unified = true;
        params.n_threads  = params.n_threads_batch = 4;
        llama_context_ptr ctx(llama_init_from_model(model, params));
        set_env("LLAMA_KV_SEQ_CELLS", nullptr);
        set_env("LLAMA_KV_PAD_PREFILL", nullptr);
        if (!ctx) throw std::runtime_error("shared pool reuse: failed to create a context");
        if (llama_n_ctx_seq(ctx.get()) != (seq_cells ? (uint32_t) atoi(seq_cells) : params.n_ctx)) {
            throw std::runtime_error("shared pool reuse: the limit of a sequence's cells was not taken");
        }

        std::vector<float> logits;
        for (int begin = 0; begin < n_long; begin += n_long_rows) {
            llama_batch batch = llama_batch_init(n_long_rows, 0, 1);
            for (int pos = begin; pos < begin + n_long_rows; ++pos) {
                batch_add(batch, tokens_long[pos], pos, {0}, true);
            }
            if (mtp) batch.embd = hidden_long.data();
            const int result = llama_decode(ctx.get(), batch);
            batch.embd = nullptr;
            llama_batch_free(batch);
            if (result) throw std::runtime_error("shared pool reuse: decode of the long sequence failed");
            for (int i = 0; i < n_long_rows; ++i) {
                const float * row = llama_get_logits_ith(ctx.get(), i);
                logits.insert(logits.end(), row, row + n_vocab);
            }
        }
        return logits;
    };

    for (const bool mtp : {false, true}) {
        const read_result legacy = read(true, mtp);
        const read_result shared = read(false, mtp);
        const read_result packed = read(false, mtp, true);
        const read_result writes = read(false, mtp, false, true);
        const read_result alone  = read(false, mtp, false, false, false);
        const read_result limit  = read(false, mtp, false, false, true, "256");
        const read_result cells  = read(false, mtp, false, false, true, nullptr, true);

        const std::vector<float> long_whole = read_long(mtp, nullptr);
        const std::vector<float> long_limit = read_long(mtp, "256");

        // 256 cells a sequence: 2,048, which the 1,088 cells never cross, and their pools never cross a quarter of it. The
        // first step builds its graph (another length than the prompt's decodes); every later step reuses it. The
        // prompt's own reuses are not counted. Padded to 256, the pool crosses at
        // steps 6, 17, 27 and 38 (272, 536, 776 and 1,040 cells), and each builds again. The cells viewed are the
        // pool's 2,048 in both capacity modes, below the direct-index cutoff's 4,096-cell floor
        const int want_shared     = n_steps - 1;
        const int want_legacy_max = want_shared - 4;
        const std::set<int64_t> want_n_kv       = { 2048 };
        const std::set<int64_t> want_n_kv_cells = { 2048 };
        const int64_t want_n_kv_legacy = GGML_PAD(n_seq*n_pos, 256);

        bool ok = true;
        if (shared.n_reused != want_shared || legacy.n_reused > want_legacy_max) {
            LOG_ERR("GLM shared pool reuse: mtp=%d: reused %d graphs padded to 256 a sequence (want %d), %d padded to 256 "
                    "(want at most %d)\n", mtp, shared.n_reused, want_shared, legacy.n_reused, want_legacy_max);
            ok = false;
        }
        if (shared.n_kv != want_n_kv || legacy.n_kv.empty() || *legacy.n_kv.rbegin() != want_n_kv_legacy) {
            LOG_ERR("GLM shared pool reuse: mtp=%d: the steps read %zu cache lengths up to %lld padded to 256 a sequence "
                    "(want 2048 alone), up to %lld padded to 256 (want %lld)\n", mtp, shared.n_kv.size(),
                    shared.n_kv.empty() ? -1LL : (long long) *shared.n_kv.rbegin(),
                    legacy.n_kv.empty() ? -1LL : (long long) *legacy.n_kv.rbegin(), (long long) want_n_kv_legacy);
            ok = false;
        }
        if (cells.n_reused != want_shared || cells.n_kv != want_n_kv_cells || shared.direct || cells.direct) {
            LOG_ERR("GLM shared pool reuse: mtp=%d: with LLAMA_KV_CELL_CAPACITY_LEGACY=1 the steps reused %d graphs (want %d) "
                    "over %zu cache lengths up to %lld (want 2048 alone), default/legacy through indices %d/%d (want 0/0: under 4,096 cells)\n",
                    mtp, cells.n_reused, want_shared, cells.n_kv.size(),
                    cells.n_kv.empty() ? -1LL : (long long) *cells.n_kv.rbegin(), shared.direct, cells.direct);
            ok = false;
        }
        LOG_INF("GLM shared pool reuse: mtp=%d: attention through indices at 2,048 cells: default %d, legacy %d\n",
                mtp, shared.direct, cells.direct);
        if (limit.n_reused != shared.n_reused || limit.n_kv != shared.n_kv) {
            LOG_ERR("GLM shared pool reuse: mtp=%d: under LLAMA_KV_SEQ_CELLS=256 the steps reused %d graphs (want %d) over "
                    "%zu cache lengths (want %zu)\n", mtp, limit.n_reused, shared.n_reused, limit.n_kv.size(), shared.n_kv.size());
            ok = false;
        }
        if (on_cpu ? long_limit != long_whole : !(nmse(long_limit, long_whole) <= 1e-6)) {
            LOG_ERR("GLM shared pool reuse: mtp=%d: a sequence past LLAMA_KV_SEQ_CELLS=256 differs from the unlimited "
                    "pool's, nmse=%g\n", mtp, nmse(long_limit, long_whole));
            ok = false;
        }
        if (!shared.patched || writes.patched || packed.patched) {
            LOG_ERR("GLM shared pool reuse: mtp=%d: the bank's changes went with the graph: %d (want 1), %d with "
                    "LLAMA_KPOOL_PATCH_INPUT_LEGACY=1 (want 0), %d with packed maps (want 0)\n", mtp, shared.patched,
                    writes.patched, packed.patched);
            ok = false;
        }
        for (const auto & other : { std::make_pair("padded to 256", &legacy), std::make_pair("with packed maps", &packed),
                std::make_pair("whose bank is written a change a write", &writes),
                std::make_pair("of a cache a sequence", &alone),
                std::make_pair("under LLAMA_KV_SEQ_CELLS=256", &limit),
                std::make_pair("with LLAMA_KV_CELL_CAPACITY_LEGACY=1", &cells) }) {
            const bool exact = on_cpu && other.second != &packed;
            if (exact && other.second->logits != shared.logits) {
                LOG_ERR("GLM shared pool reuse: mtp=%d: logits differ from the pool %s, nmse=%g\n", mtp, other.first,
                        nmse(other.second->logits, shared.logits));
                ok = false;
            }
            if (!exact && !on_cpu) {
                // a device: glm_device_logits_agree's bar. A row is step*24 + sequence*3 + its row of the step
                const std::string pairing = "mtp=" + std::to_string(mtp) + ": against the pool " + other.first;
                const float tie = other.second == &packed ? 1e-2f : 1e-4f;
                if (!glm_device_logits_agree("shared pool reuse", pairing, seed, other.second->logits, shared.logits,
                            n_vocab, tie)) {
                    ok = false;
                }
            }
            if (!exact && on_cpu) {
                // the packed maps on the CPU
                size_t n_top_differ = 0;
                size_t n_top_tied   = 0;
                const double kld = mean_kld(other.second->logits, shared.logits, n_vocab, n_top_differ, n_top_tied,
                        1e-2f);
                if (!(kld <= 1e-6) || n_top_differ > 0) {
                    LOG_ERR("GLM shared pool reuse: mtp=%d: against the pool %s: mean KLD %g (at most 1e-6), %zu rows' top "
                            "token differs, %zu tied rows' swapped, nmse=%g\n", mtp, other.first, kld, n_top_differ,
                            n_top_tied, nmse(other.second->logits, shared.logits));
                    ok = false;
                }
            }
        }
        if (!ok) return false;
    }
    return true;
}

// The context's schedulers place their graphs in one buffer a buffer type (llama_context::graph_shared), so between two
// computes of a graph the graphs of the other schedulers write over its tensors. Decodes of six shapes in turn, five of
// them retained and one on the main scheduler, give the logits of LLAMA_GRAPH_BUFFER_SHARED_LEGACY=1 (a buffer a
// scheduler) bit for bit, on the target and on the MTP head, and reuse as many graphs. Two arms: one reads every
// decode's logits; the other decodes a round's six shapes back to back, each queued on the devices behind the one
// before with nothing waited for, and reads the last decode's logits, which every sequence's earlier decodes went
// into. The context's compute memory (llama_get_memory_breakdown) stays what its reserve allocated, where a buffer a
// scheduler adds the retained graphs' buffers to it.
static bool test_glm_shared_buffers(const size_t seed, const std::vector<ggml_backend_dev_t> & devs,
        llama_split_mode split_mode, const std::vector<float> & tensor_split) {
    auto gguf = get_gguf_ctx(LLM_ARCH_GLM5NEXT, true, true);
    kv_writer ms(LLM_ARCH_GLM5NEXT, gguf.get());
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_TOP_K, uint32_t(8)); // the indexer scores from the first step
    auto pair = get_model_and_ctx(gguf.get(), nullptr, seed, devs, split_mode, false, tensor_split, true);
    auto * model = pair.first.get();
    const int n_embd  = llama_model_n_embd(model);
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    // the sequences [first, first + n_seqs) decode rows positions each; 48 rows are past a slot's 32
    struct shape { int first, n_seqs, rows; };
    const std::vector<shape> shapes = { {0, 1, 2}, {0, 8, 1}, {0, 4, 3}, {0, 8, 6}, {4, 4, 1}, {0, 8, 3} };
    const int n_seq    = 8;
    const int n_rounds = 5;
    const int n_rows_max = 48;
    int n_pos = 0;
    for (const auto & sh : shapes) n_pos += sh.rows;
    n_pos *= n_rounds;
    const auto tokens = get_tokens(n_seq*n_pos, n_vocab, seed + 4);
    std::mt19937 rng(seed + 4);
    std::normal_distribution<float> normal(0.0f, 0.1f);
    std::vector<float> hidden((size_t) n_rows_max*n_embd);
    for (auto & value : hidden) value = normal(rng);

    struct read_result {
        std::vector<float> logits;  // every row of every decode read
        int32_t            n_reused = 0;
        size_t             compute_reserved = 0; // the context's compute buffers before its first decode
        size_t             compute          = 0; // and after its last
    };
    const auto compute_size = [](const llama_context * ctx) {
        size_t size = 0;
        for (const auto & [buft, data] : llama_get_memory_breakdown(ctx)) {
            size += data.compute;
        }
        return size;
    };
    const auto read = [&](bool legacy, bool mtp, bool queued) {
        set_env("LLAMA_GRAPH_BUFFER_SHARED_LEGACY", legacy ? "1" : nullptr);
        set_env("LLAMA_GRAPH_SLOTS", "6");
        read_result res;
        llama_context_params params = llama_context_default_params();
        if (mtp) params.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
        params.n_ctx      = 8192;
        params.n_batch    = 512;
        params.n_ubatch   = 512;
        params.n_seq_max  = n_seq;
        params.kv_unified = true;
        params.n_threads  = params.n_threads_batch = 4;
        llama_context_ptr ctx(llama_init_from_model(model, params));
        set_env("LLAMA_GRAPH_BUFFER_SHARED_LEGACY", nullptr);
        set_env("LLAMA_GRAPH_SLOTS", nullptr);
        if (!ctx) throw std::runtime_error("shared buffers: failed to create a context");
        res.compute_reserved = compute_size(ctx.get());

        std::vector<int> next(n_seq, 0); // a sequence's next position
        for (int round = 0; round < n_rounds; ++round) {
            for (const auto & sh : shapes) {
                llama_batch batch = llama_batch_init(sh.n_seqs*sh.rows, 0, 1);
                for (int seq = sh.first; seq < sh.first + sh.n_seqs; ++seq) {
                    for (int i = 0; i < sh.rows; ++i, ++next[seq]) {
                        batch_add(batch, tokens[(size_t) seq*n_pos + next[seq]], next[seq], {seq}, true);
                    }
                }
                if (mtp) batch.embd = hidden.data();
                const int result = llama_decode(ctx.get(), batch);
                batch.embd = nullptr;
                llama_batch_free(batch);
                if (result) throw std::runtime_error("shared buffers: decode failed");
                if (queued && &sh != &shapes.back()) {
                    continue; // nothing of this decode is read: the next one is queued behind it
                }
                for (int i = 0; i < sh.n_seqs*sh.rows; ++i) {
                    const float * row = llama_get_logits_ith(ctx.get(), i);
                    res.logits.insert(res.logits.end(), row, row + n_vocab);
                }
            }
        }
        res.n_reused = llama_perf_context(ctx.get()).n_reused;
        res.compute  = compute_size(ctx.get());
        return res;
    };

    bool ok = true;
    for (const bool mtp : {false, true}) {
        const auto shared = read(false, mtp, false);
        const auto legacy = read(true, mtp, false);
        if (shared.logits.size() != legacy.logits.size() ||
                std::memcmp(shared.logits.data(), legacy.logits.data(), shared.logits.size()*sizeof(float)) != 0) {
            LOG_ERR("GLM shared buffers: mtp=%d: the logits differ from LLAMA_GRAPH_BUFFER_SHARED_LEGACY=1's\n", mtp);
            ok = false;
        }
        const auto shared_queued = read(false, mtp, true);
        const auto legacy_queued = read(true, mtp, true);
        if (shared_queued.logits.empty() || shared_queued.logits.size() != legacy_queued.logits.size() ||
                std::memcmp(shared_queued.logits.data(), legacy_queued.logits.data(), shared_queued.logits.size()*sizeof(float)) != 0) {
            LOG_ERR("GLM shared buffers: mtp=%d: with a round's decodes queued back to back the logits differ from "
                    "LLAMA_GRAPH_BUFFER_SHARED_LEGACY=1's\n", mtp);
            ok = false;
        }
        if (shared_queued.n_reused != legacy_queued.n_reused || shared_queued.n_reused != shared.n_reused) {
            LOG_ERR("GLM shared buffers: mtp=%d: %d graphs reused with a round's decodes queued, %d with "
                    "LLAMA_GRAPH_BUFFER_SHARED_LEGACY=1, %d with every decode read (want the same)\n", mtp,
                    shared_queued.n_reused, legacy_queued.n_reused, shared.n_reused);
            ok = false;
        }
        // five retained shapes, each built once
        const int32_t want_reused = (int32_t) (shapes.size() - 1)*(n_rounds - 1);
        if (shared.n_reused != legacy.n_reused || shared.n_reused < want_reused) {
            LOG_ERR("GLM shared buffers: mtp=%d: %d graphs reused, %d with LLAMA_GRAPH_BUFFER_SHARED_LEGACY=1 (want %d or "
                    "more in both)\n", mtp, shared.n_reused, legacy.n_reused, want_reused);
            ok = false;
        }
        // the reserve's buffers hold every graph, where a buffer a scheduler adds the five retained graphs' (pipeline
        // parallelism, which layers split over devices may run with, keeps a buffer a scheduler)
        const bool pipeline = devs.size() > 1 && split_mode == LLAMA_SPLIT_MODE_LAYER;
        for (const auto * res : {&shared, &shared_queued}) {
            if (!pipeline && (res->compute_reserved == 0 || res->compute != res->compute_reserved)) {
                LOG_ERR("GLM shared buffers: mtp=%d: the context's compute buffers are %zu bytes after the decodes, %zu "
                        "as reserved (want the same)\n", mtp, res->compute, res->compute_reserved);
                ok = false;
            }
        }
        if (legacy.compute <= legacy.compute_reserved || legacy.compute_reserved != shared.compute_reserved) {
            LOG_ERR("GLM shared buffers: mtp=%d: with LLAMA_GRAPH_BUFFER_SHARED_LEGACY=1 the context's compute buffers are "
                    "%zu bytes after the decodes, %zu as reserved (want more after, and %zu as reserved)\n", mtp,
                    legacy.compute, legacy.compute_reserved, shared.compute_reserved);
            ok = false;
        }
    }
    return ok;
}

// The token embedding table on the GPUs (LLAMA_TOK_EMBD_GPU=1). The served GLM-5.3 Flash kept its 643 MiB q8_0 table
// in host memory: each 2,048-row prefill step gathered its rows on the CPU and copied the 32 MiB result to each card,
// and the MTP head gathered the same tokens again, 4 x 8 MiB to each card (the host-gap trace of Nov 1, 2026). With the
// switch and every layer on a GPU, the table sits where layer 0 does (on every card under -sm tensor), the gather runs
// there, and the logits of the main and the MTP context are the host gather's, byte for byte. Without a GPU, or with
// the switch off, the table stays in host memory.
static bool test_glm_tok_embd_on_device(const size_t seed, const std::vector<ggml_backend_dev_t> & devs,
        llama_split_mode split_mode, const std::vector<float> & tensor_split) {
    auto gguf = get_gguf_ctx(LLM_ARCH_GLM5NEXT, true, true);

    set_env("LLAMA_TOK_EMBD_GPU", nullptr);
    auto host = get_model_and_ctx(gguf.get(), nullptr, seed, devs, split_mode, false, tensor_split, true);
    set_env("LLAMA_TOK_EMBD_GPU", "1");
    auto device = get_model_and_ctx(gguf.get(), nullptr, seed, devs, split_mode, false, tensor_split, true);
    set_env("LLAMA_TOK_EMBD_GPU", nullptr);

    // a device with memory of its own, as test_backends picks the tensor split's devices
    const bool on_gpu = std::any_of(devs.begin(), devs.end(), [](ggml_backend_dev_t dev) {
        return ggml_backend_dev_buffer_type(dev) != ggml_backend_cpu_buffer_type();
    });
    if (!ggml_backend_buffer_is_host(host.first->tok_embd->buffer)) {
        LOG_ERR("GLM tok_embd: the switch off put the table in device memory (%s)\n",
                ggml_backend_buffer_name(host.first->tok_embd->buffer));
        return false;
    }
    if (ggml_backend_buffer_is_host(device.first->tok_embd->buffer) == on_gpu) {
        LOG_ERR("GLM tok_embd: the switch on with %zu devices left the table in %s\n", devs.size(),
                ggml_backend_buffer_name(device.first->tok_embd->buffer));
        return false;
    }

    auto * model = host.first.get();
    const int n_embd  = llama_model_n_embd(model);
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const auto tokens = get_tokens(100, n_vocab, seed);
    if (get_logits(host.first.get(), host.second.get(), tokens) !=
            get_logits(device.first.get(), device.second.get(), tokens)) {
        LOG_ERR("GLM tok_embd: the main context's logits differ with the table on the device\n");
        return false;
    }

    std::mt19937 rng(seed);
    std::normal_distribution<float> normal(0.0f, 0.1f);
    std::vector<float> hidden(tokens.size()*n_embd);
    for (auto & value : hidden) value = normal(rng);
    llama_context_params params = llama_context_default_params();
    params.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
    params.n_ctx = 256;
    params.n_batch = params.n_ubatch = 128;
    params.n_threads = params.n_threads_batch = 4;
    llama_context_ptr mtp_host(llama_init_from_model(host.first.get(), params));
    llama_context_ptr mtp_device(llama_init_from_model(device.first.get(), params));
    if (!mtp_host || !mtp_device) return false;
    if (get_mtp_logits(host.first.get(), mtp_host.get(), tokens, hidden) !=
            get_mtp_logits(device.first.get(), mtp_device.get(), tokens, hidden)) {
        LOG_ERR("GLM tok_embd: the MTP context's logits differ with the table on the device\n");
        return false;
    }
    return true;
}

// The MTP head reads the target's hidden rows where the target left them (the drafter binds the two contexts when
// LLAMA_MTP_DEVICE_H=1). On the served GLM-5.3 Flash each 2,048-row prefill step copied the target's 32 MiB of hidden
// rows from card 0 to the host, and the head's input back to each card (the host-gap trace of Nov 1, 2026). Bound, a
// target batch of 64 rows or more writes every row into a store in its output device's memory (on every card under
// -sm tensor) by batch index, and the head gathers its rows there. On every device config, against an unbound pair:
// the store holds exactly such a batch's rows, the target's logits and the rows read back through the host are the same
// bytes, the head's logits are those of a head fed the same rows from the host, byte for byte, and the store makes no
// compute buffer grow on its own: since the bind a bound context's buffer has grown no more than its unbound twin's
// since it was made, to a page (the growth is the engine's own after its first reserve, which the store doesn't
// change). The two do not grow on the same steps, as their graphs differ: a bound head whose first short steps of
// the churn took 3 rows grew 3,264 bytes on its first 6-row one, where the unbound head, 65,184 bytes larger since
// two steps before, did not (seeds 5 and 9), so the steps are not compared one by one. A batch under 64 rows keeps
// the host path.
static bool test_glm_nextn_store_run(const size_t seed, const std::vector<ggml_backend_dev_t> & devs,
        llama_split_mode split_mode, const std::vector<float> & tensor_split, bool q8_cache, bool retained) {
    auto gguf = get_gguf_ctx(LLM_ARCH_GLM5NEXT, true, true);
    if (split_mode == LLAMA_SPLIT_MODE_DATA) {
        kv_writer ms(LLM_ARCH_GLM5NEXT, gguf.get());
        ms.add_kv(LLM_KV_EXPERT_COUNT, uint32_t(8));
        ms.add_kv(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT, uint32_t(64));
        ms.add_kv(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH, uint32_t(128));
    }
    auto pair = get_model_and_ctx(gguf.get(), nullptr, seed, devs, split_mode, false, tensor_split, true);
    auto * model = pair.first.get();
    const int n_embd  = llama_model_n_embd_out(model);
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const auto tokens = get_tokens(1024, n_vocab, seed);

    const auto make = [&](bool mtp) {
        llama_context_params params = llama_context_default_params();
        if (mtp) params.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
        params.n_ctx      = 1024;
        params.n_batch    = 512;
        params.n_ubatch   = 64;
        params.n_seq_max  = 2;
        params.kv_unified = split_mode != LLAMA_SPLIT_MODE_DATA;
        if (split_mode == LLAMA_SPLIT_MODE_DATA) { params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED; }
        if (q8_cache) { params.type_k = params.type_v = params.type_s = params.type_idx = GGML_TYPE_Q8_0; }
        params.n_threads  = params.n_threads_batch = 4;
        llama_context_ptr ctx(llama_init_from_model(model, params));
        if (!ctx) throw std::runtime_error("nextn store: failed to create a context");
        // as the MTP drafter asks: every target row, the head's output rows
        llama_set_embeddings_nextn(ctx.get(), true, /*masked*/ mtp);
        return ctx;
    };
    const auto compute = [](const llama_context * ctx) {
        std::map<ggml_backend_buffer_type_t, size_t> sizes;
        for (const auto & [buft, mb] : llama_get_memory_breakdown(ctx)) {
            sizes[buft] = mb.compute;
        }
        return sizes;
    };

    llama_context_ptr tgt_host = make(false), head_host = make(true);
    llama_context_ptr tgt_dev  = make(false), head_dev  = make(true);
    if (!llama_set_nextn_source(head_dev.get(), tgt_dev.get(), retained ? 4 : 0)) {
        LOG_ERR("GLM nextn store: the target and its MTP head would not bind\n");
        return false;
    }
    const auto tgt_at_bind = compute(tgt_dev.get()), head_at_bind = compute(head_dev.get());
    const auto tgt_host_at0 = compute(tgt_host.get()), head_host_at0 = compute(head_host.get());

    llama_pos n_past[2] = { 0, 0 };
    std::vector<float> kept[2] = { std::vector<float>(n_embd, 0.0f), std::vector<float>(n_embd, 0.0f) };
    std::vector<float> retained_rows[2];
    std::mt19937 retain_rng(seed ^ 0x15a3u);
    const auto retained_equal = [&]() {
        for (llama_seq_id s = 0; s < 2; ++s) {
            for (size_t row = 0; row < retained_rows[s].size()/n_embd; ++row) {
                std::vector<float> actual(n_embd);
                if (!llama_nextn_kept_get(tgt_dev.get(), llama_nextn_kept_row(tgt_dev.get(), s, row), actual.data()) ||
                        std::memcmp(actual.data(), retained_rows[s].data() + row*n_embd, n_embd*sizeof(float))) {
                    return false;
                }
            }
        }
        return true;
    };
    int n_tok = 0;

    // one prefill step: each part a sequence's next rows, the last row of each an output
    const auto step = [&](const std::vector<std::pair<llama_seq_id, int>> & parts) {
        int n = 0;
        for (const auto & p : parts) n += p.second;
        llama_batch batch = llama_batch_init(n, 0, 1);
        std::vector<int> first;
        for (const auto & p : parts) {
            first.push_back(batch.n_tokens);
            for (int i = 0; i < p.second; ++i) {
                batch_add(batch, tokens[n_tok++ % tokens.size()], n_past[p.first] + i, {p.first}, i == p.second - 1);
            }
        }
        const auto fail = [&](const char * what) {
            LOG_ERR("GLM nextn store: a %d-row step: %s\n", n, what);
            llama_batch_free(batch);
            return false;
        };
        if (llama_decode(tgt_host.get(), batch) || llama_decode(tgt_dev.get(), batch)) return fail("a target decode failed");
        const bool device = retained || n >= 64;
        if (llama_get_nextn_store_rows(tgt_dev.get()) != (device ? n : 0) || llama_get_nextn_store_rows(tgt_host.get()) != 0) {
            LOG_ERR("GLM nextn store: the store holds %d rows, the unbound target's %d\n",
                    llama_get_nextn_store_rows(tgt_dev.get()), llama_get_nextn_store_rows(tgt_host.get()));
            return fail("the store does not hold exactly the rows of a step of 64 rows or more");
        }
        for (size_t j = 0; j < parts.size(); ++j) {
            const int i = first[j] + parts[j].second - 1;
            if (std::memcmp(llama_get_logits_ith(tgt_host.get(), i), llama_get_logits_ith(tgt_dev.get(), i), n_vocab*sizeof(float))) {
                return fail("the bound target's logits differ");
            }
        }
        // Adjacent, disjoint and repeated rows must stay in caller order across owners.
        std::vector<int32_t> ends;
        for (size_t j = 0; j < parts.size(); ++j) {
            ends.push_back(first[j]);
            if (parts[j].second > 1) { ends.push_back(first[j] + 1); }
            ends.push_back(first[j] + parts[j].second - 1);
        }
        ends.push_back(ends.front());
        std::vector<float> h_ends(ends.size()*n_embd);
        const float * h_host = llama_get_embeddings_nextn(tgt_host.get());
        if (device) {
            const int32_t invalid[] = {0, n};
            std::vector<float> untouched(2*n_embd, -123.0f);
            if (llama_get_embeddings_nextn_row_list(tgt_dev.get(), invalid, 2, untouched.data())) {
                return fail("an out-of-range kept row was accepted");
            }
            llama_synchronize(tgt_dev.get());
            if (!std::all_of(untouched.begin(), untouched.end(), [](float v) { return v == -123.0f; })) {
                return fail("an invalid read queued a partial host write");
            }
            if (!llama_get_embeddings_nextn_row_list(tgt_dev.get(), ends.data(), (int32_t) ends.size(), h_ends.data())) {
                return fail("the kept rows could not be read");
            }
            for (size_t j = 0; j < ends.size(); ++j) {
                if (std::memcmp(h_ends.data() + j*n_embd, h_host + (size_t) ends[j]*n_embd, n_embd*sizeof(float))) {
                    return fail("a kept row read from the store differs from the unbound target's");
                }
            }
        }
        const float * h_dev = llama_get_embeddings_nextn(tgt_dev.get());
        if (h_dev == nullptr || std::memcmp(h_host, h_dev, (size_t) n*n_embd*sizeof(float))) {
            return fail("the bound target's rows read through the host differ from the unbound target's");
        }

        // the head: row k reads the target's row k - 1, a sequence's first row the row kept before it
        std::vector<float> hidden((size_t) n*n_embd);
        std::vector<int32_t> src(n);
        std::vector<float> carry;
        for (size_t j = 0; j < parts.size(); ++j) {
            const llama_seq_id s = parts[j].first;
            for (int i = first[j]; i < first[j] + parts[j].second; ++i) {
                const bool head_row = i == first[j];
                std::memcpy(hidden.data() + (size_t) i*n_embd, head_row ? kept[s].data() : h_host + (size_t) (i - 1)*n_embd,
                            n_embd*sizeof(float));
                src[i] = head_row ? (retained ? llama_nextn_kept_row(tgt_dev.get(), n_past[s] ? s : -1, 0) : -(int32_t) (carry.size()/n_embd + 1)) : i - 1;
                if (head_row) carry.insert(carry.end(), kept[s].begin(), kept[s].end());
            }
        }
        for (int i = 0; i < n; ++i) batch.logits[i] = true;
        batch.embd = hidden.data();
        int rc = llama_decode(head_host.get(), batch);
        if (device) {
            llama_set_nextn_h_rows(head_dev.get(), src.data(), n, carry.data(), retained ? 0 : (int32_t) (carry.size()/n_embd));
            batch.embd = nullptr;
        }
        rc = rc || llama_decode(head_dev.get(), batch);
        batch.embd = nullptr;
        if (rc) return fail("a head decode failed");
        for (int i = 0; i < n; ++i) {
            if (std::memcmp(llama_get_logits_ith(head_host.get(), i), llama_get_logits_ith(head_dev.get(), i), n_vocab*sizeof(float))) {
                LOG_ERR("GLM nextn store: row %d of %d, nmse %g\n", i, n, nmse(
                    std::vector<float>(llama_get_logits_ith(head_host.get(), i), llama_get_logits_ith(head_host.get(), i) + n_vocab),
                    std::vector<float>(llama_get_logits_ith(head_dev.get(), i), llama_get_logits_ith(head_dev.get(), i) + n_vocab)));
                return fail(device ? "the head's logits reading the store differ from the host-fed head's" :
                                     "the head's logits on the host path differ once bound");
            }
        }
        if (!retained_equal()) return fail("a retained row changed across target and head consumption");
        for (size_t j = 0; j < parts.size(); ++j) {
            const llama_seq_id s = parts[j].first;
            const float * last = h_host + (size_t) (first[j] + parts[j].second - 1)*n_embd;
            kept[s].assign(last, last + n_embd);
            if (retained) {
                const int32_t row = first[j] + parts[j].second - 1;
                std::vector<int32_t> rows = {row};
                const size_t count = 1 + retain_rng()%std::min(4, parts[j].second);
                while (rows.size() < count) rows.push_back(first[j] + retain_rng()%parts[j].second);
                retained_rows[s].resize(count*n_embd);
                for (size_t i = 0; i < count; ++i) {
                    std::memcpy(retained_rows[s].data() + i*n_embd, h_host + size_t(rows[i])*n_embd, n_embd*sizeof(float));
                }
                const int64_t before = llama_get_nextn_rows_to_host(tgt_dev.get());
                if (!llama_nextn_keep_rows(tgt_dev.get(), s, rows.data(), rows.size())) return fail("sequence rows could not be retained");
                if (llama_get_nextn_rows_to_host(tgt_dev.get()) != before) return fail("retaining rows copied them to host");
                const int32_t invalid[] = {row, n};
                if (llama_nextn_keep_rows(tgt_dev.get(), s, invalid, 2)) return fail("invalid retained rows were accepted");
            }
            n_past[s] += parts[j].second;
        }
        llama_batch_free(batch);
        // a buffer of a bound context has grown more since the bind than its unbound twin's since it was made, by over
        // a page (the bound target's grew 128 to 384 bytes more over the seeds read, the bound head's less)
        const auto grew_more = [&](const std::map<ggml_backend_buffer_type_t, size_t> & bound_at_bind,
                                   const llama_context * bound,
                                   const std::map<ggml_backend_buffer_type_t, size_t> & unbound_at0,
                                   const llama_context * unbound) {
            const auto size = [](const std::map<ggml_backend_buffer_type_t, size_t> & m, ggml_backend_buffer_type_t b) {
                const auto it = m.find(b);
                return it == m.end() ? (size_t) 0 : it->second;
            };
            const auto grown = [](size_t now, size_t then) { return now > then ? now - then : (size_t) 0; };
            const auto bound_now = compute(bound), unbound_now = compute(unbound);
            for (const auto & [buft, sz] : bound_now) {
                if (grown(sz, size(bound_at_bind, buft)) > grown(size(unbound_now, buft), size(unbound_at0, buft)) + 4096) {
                    return true;
                }
            }
            return false;
        };
        const bool more = grew_more(tgt_at_bind, tgt_dev.get(), tgt_host_at0, tgt_host.get()) ||
                          grew_more(head_at_bind, head_dev.get(), head_host_at0, head_host.get());
        if (more) {
            LOG_ERR("GLM nextn store: after a %d-row step a bound compute buffer has grown more than the unbound one's\n", n);
            const auto show = [](const char * who, const std::map<ggml_backend_buffer_type_t, size_t> & a,
                                 const std::map<ggml_backend_buffer_type_t, size_t> & b) {
                for (const auto & [buft, sz] : b) {
                    const auto it = a.find(buft);
                    LOG_ERR("GLM nextn store:   %s %s: %zu then %zu bytes\n", who, ggml_backend_buft_name(buft),
                            it == a.end() ? (size_t) 0 : it->second, sz);
                }
            };
            show("bound target", tgt_at_bind, compute(tgt_dev.get()));
            show("bound head", head_at_bind, compute(head_dev.get()));
            show("unbound target", tgt_host_at0, compute(tgt_host.get()));
            show("unbound head", head_host_at0, compute(head_host.get()));
            return false;
        }
        return true;
    };

    // two sequences' prompts, the longer first (the equal split interleaves them); one sequence's on the other half of
    // the store; two short steps on the host path; then the first half again
    if (!step({{0, 180}, {1, 120}}) || !step({{1, 100}}) || !step({{0, 3}, {1, 3}}) || !step({{0, 200}})) {
        return false;
    }
    if (!retained) {
        return true;
    }

    // Churn short and paired requests after the store has reached its steady shape. Each retained row has its own
    // sequence slot, so the bound contexts must keep their resource footprint fixed while both owners replace rows.
    std::mt19937 churn(seed ^ 0x5a17u);
    for (int i = 0; i < 24; ++i) {
        std::vector<std::pair<llama_seq_id, int>> parts;
        if (churn() & 1) {
            parts = {{0, 3}, {1, 3}};
        } else {
            parts = {{(llama_seq_id) (churn() & 1), 3}};
        }
        if (!step(parts)) {
            return false;
        }
    }
    return retained_equal();
}

static bool test_glm_nextn_store(const size_t seed, const std::vector<ggml_backend_dev_t> & devs,
        llama_split_mode split_mode, const std::vector<float> & tensor_split, bool q8_cache) {
    return test_glm_nextn_store_run(seed, devs, split_mode, tensor_split, q8_cache, false) &&
           test_glm_nextn_store_run(seed, devs, split_mode, tensor_split, q8_cache, true);
}

// A speculative prefix must have the same recurrent state and continuation as
// a context that never evaluated the rejected suffix. Exercise generated rounds
// at every verification width, including zero/all draft acceptance, IndexPool
// boundaries, cancellation with a pending restore, and reuse of three slots.
struct glm_state_comparison {
    bool equal;
    std::string reason;
};

struct glm_state_tensor {
    size_t offset;
    ggml_type type;
    std::vector<float> values;
};

struct glm_state_view {
    std::vector<std::pair<std::string, std::string>> metadata;
    std::vector<glm_state_tensor> tensors;
    std::string error;
};

static glm_state_view read_glm_state(const std::vector<uint8_t> & bytes) {
    glm_state_view result;
    size_t offset = 0;
    const auto read = [&](auto & value, const std::string & field) {
        if (sizeof(value) > bytes.size()-offset) {
            result.error = "truncated " + field;
            return false;
        }
        memcpy(&value, bytes.data()+offset, sizeof(value));
        offset += sizeof(value);
        result.metadata.emplace_back(field, std::to_string(value));
        return true;
    };
    uint32_t magic, cells, transposed, layers;
    llama_seq_id seq;
    if (!read(magic, "magic") || !read(seq, "sequence") || !read(cells, "cell_count")) return result;
    if (magic != 0xaf143cd8 || seq < 0 || cells == 0 || cells > (bytes.size()-offset)/8) {
        result.error = "invalid recurrent sequence header";
        return result;
    }
    for (uint32_t cell = 0; cell < cells; ++cell) {
        llama_pos pos;
        uint32_t sequences;
        const std::string field = "cell[" + std::to_string(cell) + "]";
        if (!read(pos, field+".position") || !read(sequences, field+".sequence_count")) return result;
        if (pos < 0 || sequences != 0) {
            result.error = "invalid " + field;
            return result;
        }
    }
    if (!read(transposed, "transposed") || !read(layers, "layer_count")) return result;
    if (transposed != 0 || layers == 0) {
        result.error = "unsupported recurrent tensor layout";
        return result;
    }
    while (offset < bytes.size()) {
        const std::string field = "tensor[" + std::to_string(result.tensors.size()) + "]";
        int32_t type;
        uint64_t row_bytes;
        if (!read(type, field+".type") || !read(row_bytes, field+".row_bytes")) return result;
        if (type != GGML_TYPE_F32 && type != GGML_TYPE_F16 && type != GGML_TYPE_BF16 && type != GGML_TYPE_Q8_0) {
            result.error = "unsupported " + field+".type";
            return result;
        }
        const auto t = ggml_type(type);
        const auto * traits = ggml_get_type_traits(t);
        if (row_bytes == 0 || row_bytes%traits->type_size || row_bytes > (bytes.size()-offset)/cells) {
            result.error = "invalid or truncated " + field+".payload";
            return result;
        }
        const size_t count = row_bytes*cells;
        glm_state_tensor tensor {offset, t, std::vector<float>(count/traits->type_size*traits->blck_size)};
        if (t == GGML_TYPE_F32) memcpy(tensor.values.data(), bytes.data()+offset, count);
        else traits->to_float(bytes.data()+offset, tensor.values.data(), tensor.values.size());
        for (size_t i = 0; i < tensor.values.size(); ++i) {
            if (!std::isfinite(tensor.values[i])) {
                result.error = field+" nonfinite element="+std::to_string(i)+" payload_offset="+std::to_string(offset);
                return result;
            }
        }
        offset += count;
        result.tensors.push_back(std::move(tensor));
        if (result.tensors.size() > 2*uint64_t(layers)) {
            result.error = "too many recurrent tensors";
            return result;
        }
    }
    if (result.tensors.empty() || result.tensors.size()%2) result.error = "incomplete recurrent tensor pairs";
    return result;
}

static glm_state_comparison compare_glm_states(const std::vector<uint8_t> & a, const std::vector<uint8_t> & b) {
    const auto actual = read_glm_state(a), reference = read_glm_state(b);
    if (!actual.error.empty()) return {false, "actual: " + actual.error};
    if (!reference.error.empty()) return {false, "reference: " + reference.error};
    if (actual.metadata.size() != reference.metadata.size()) return {false, "metadata field count differs"};
    for (size_t i = 0; i < actual.metadata.size(); ++i) {
        if (actual.metadata[i] != reference.metadata[i]) {
            return {false, "metadata " + actual.metadata[i].first + ": actual=" + actual.metadata[i].second +
                           " reference=" + reference.metadata[i].second};
        }
    }
    std::string failures;
    for (size_t t = 0; t < actual.tensors.size(); ++t) {
        const auto & x = actual.tensors[t];
        const auto & y = reference.tensors[t];
        double error = 0, magnitude = 0, max_abs = 0;
        size_t first = x.values.size();
        for (size_t i = 0; i < x.values.size(); ++i) {
            const double d = double(x.values[i])-double(y.values[i]);
            error += d*d;
            magnitude += double(y.values[i])*double(y.values[i]);
            max_abs = std::max(max_abs, std::fabs(d));
            if (!(std::fabs(x.values[i]-y.values[i]) <= 1e-6f) && first == x.values.size()) first = i;
        }
        if (first != x.values.size()) {
            char line[384];
            snprintf(line, sizeof(line), "tensor[%zu] type=%s payload_offset=%zu first_element=%zu actual=%.9g reference=%.9g max_abs=%.9g nmse=%.9g finite=1; ",
                     t, ggml_type_name(x.type), x.offset, first, x.values[first], y.values[first], max_abs,
                     magnitude > 0 ? error/magnitude : INFINITY);
            failures += line;
        }
    }
    return {failures.empty(), failures};
}

static int test_glm_state_comparison() {
    ggml_context_ptr init(ggml_init({ggml_tensor_overhead(), nullptr, true}));
    int failed = 0, checked = 0;
    const auto check = [&](bool ok, const char * name) {
        ++checked;
        if (!ok && failed++ < 12) fprintf(stderr, "state comparison regression: %s\n", name);
    };
    std::mt19937 rng(1234);
    for (int trial = 0; trial < 64; ++trial) {
        std::vector<uint8_t> a;
        const auto append = [&](const auto value) {
            const auto * bytes = reinterpret_cast<const uint8_t *>(&value);
            a.insert(a.end(), bytes, bytes + sizeof(value));
        };
        append(uint32_t(0xaf143cd8));
        append(llama_seq_id(trial%3));
        append(uint32_t(1));
        append(llama_pos(256 + rng()%1024));
        append(uint32_t(0));
        append(uint32_t(0));
        append(uint32_t(1));
        for (int tensor = 0; tensor < 2; ++tensor) {
            append(int32_t(GGML_TYPE_F32));
            append(uint64_t(32*sizeof(float)));
            for (int i = 0; i < 32; ++i) append(float(int(rng()%1000)-500)/10000.0f);
        }
        check(compare_glm_states(a, a).equal, "identical finite state");
        for (const size_t offset : {size_t(0), size_t(4), size_t(12), size_t(24), size_t(28), size_t(32)}) {
            auto b = a;
            uint32_t value;
            memcpy(&value, b.data()+offset, sizeof(value));
            ++value;
            memcpy(b.data()+offset, &value, sizeof(value));
            check(!compare_glm_states(a, b).equal, "changed metadata");
        }
        auto b = a;
        float value;
        memcpy(&value, b.data()+40, sizeof(value));
        value += 1e-7f;
        memcpy(b.data()+40, &value, sizeof(value));
        check(compare_glm_states(a, b).equal, "numeric difference inside existing tolerance");
        value += 1e-3f;
        memcpy(b.data()+40, &value, sizeof(value));
        check(!compare_glm_states(a, b).equal, "numeric difference outside existing tolerance");
        value = NAN;
        memcpy(b.data()+40, &value, sizeof(value));
        check(!compare_glm_states(b, b).equal, "identical NaN payloads");
        b = a;
        b.resize(b.size()-sizeof(float));
        check(!compare_glm_states(b, b).equal, "identically truncated states");
        for (const auto type : {GGML_TYPE_F16, GGML_TYPE_BF16, GGML_TYPE_Q8_0}) {
            const auto * traits = ggml_get_type_traits(type);
            std::vector<float> values(32);
            for (auto & x : values) x = float(int(rng()%1000)-500)/10000.0f;
            const auto encoded = [&](const std::vector<float> & payload) {
                std::vector<uint8_t> bytes(a.begin(), a.begin()+28);
                for (int tensor = 0; tensor < 2; ++tensor) {
                    const int32_t tag = type;
                    const uint64_t row_bytes = traits->type_size;
                    const size_t start = bytes.size();
                    const size_t row = ggml_row_size(type, payload.size());
                    const uint64_t size = row;
                    bytes.resize(start+sizeof(tag)+sizeof(row_bytes)+row);
                    memcpy(bytes.data()+start, &tag, sizeof(tag));
                    memcpy(bytes.data()+start+sizeof(tag), &size, sizeof(size));
                    traits->from_float_ref(payload.data(), bytes.data()+start+sizeof(tag)+sizeof(size), payload.size());
                }
                return bytes;
            };
            const auto typed = encoded(values);
            check(compare_glm_states(typed, typed).equal, "identical finite typed state");
            values[0] += 0.1f;
            check(!compare_glm_states(typed, encoded(values)).equal, "typed numeric difference outside tolerance");
            auto nonfinite = typed;
            const uint16_t bits = type == GGML_TYPE_BF16 ? 0x7fc0 : 0x7e00;
            memcpy(nonfinite.data()+40, &bits, sizeof(bits)); // first value, or the Q8_0 block scale
            check(!compare_glm_states(nonfinite, nonfinite).equal, "identical typed nonfinite payloads");
            auto truncated = typed;
            truncated.pop_back();
            check(!compare_glm_states(truncated, truncated).equal, "truncated typed payload");
        }
    }
    fprintf(stderr, "state comparison regressions: %d checks, %d failed\n", checked, failed);
    return failed ? 1 : 0;
}

static std::vector<uint8_t> glm_state_bytes(llama_context * ctx, int seq, bool full = false) {
    constexpr auto flags = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;
    const size_t size = full ? llama_state_get_size(ctx) : llama_state_seq_get_size_ext(ctx, seq, flags);
    std::vector<uint8_t> bytes(size);
    const size_t written = full ? llama_state_get_data(ctx, bytes.data(), size) :
        llama_state_seq_get_data_ext(ctx, bytes.data(), size, seq, flags);
    if (!size || written != size) throw std::runtime_error("GLM diagnostic state serialization failed");
    return bytes;
}

// Read the physical plane before seq_rm chooses it. No engine state is modified.
static std::vector<uint8_t> glm_snapshot(llama_context * ctx, int seq, int plane, llama_pos position) {
    llama_synchronize(ctx);
    auto * hybrid = dynamic_cast<llama_memory_hybrid *>(llama_get_memory(ctx));
    if (!hybrid) throw std::runtime_error("GLM diagnostic needs hybrid recurrent memory");
    auto * memory = hybrid->get_mem_recr();
    if (plane < 0 || uint32_t(plane) > memory->rs_reach.at(seq)) {
        throw std::runtime_error("GLM diagnostic snapshot exceeds rollback reach");
    }
    auto bytes = glm_state_bytes(ctx, seq);
    const auto view = read_glm_state(bytes);
    if (!view.error.empty()) throw std::runtime_error("GLM diagnostic snapshot layout: " + view.error);
    uint32_t cells;
    memcpy(&cells, bytes.data()+8, sizeof(cells));
    if (cells != 1) throw std::runtime_error("GLM diagnostic needs one recurrent cell");
    size_t source = memory->size;
    for (size_t i = 0; i < memory->cells.size(); ++i) {
        if (memory->cells[i].has_seq_id(seq)) {
            if (source != memory->size) throw std::runtime_error("GLM diagnostic found multiple recurrent cells");
            source = memory->cells[i].src >= 0 ? size_t(memory->cells[i].src) : i;
        }
    }
    if (source >= memory->size) throw std::runtime_error("GLM diagnostic has no recurrent source cell");
    size_t index = 0;
    for (const auto * layers : {&memory->r_l, &memory->s_l}) {
        for (auto * tensor : *layers) {
            if (!tensor) continue;
            const auto & saved = view.tensors.at(index++);
            const size_t row = ggml_row_size(tensor->type, tensor->ne[0]);
            if (saved.type != tensor->type || row != ggml_row_size(saved.type, saved.values.size()) ||
                    saved.offset + row > bytes.size()) throw std::runtime_error("GLM diagnostic tensor shape differs");
            ggml_backend_tensor_get(tensor, bytes.data()+saved.offset, (size_t(plane)*memory->size + source)*row, row);
        }
    }
    if (index != view.tensors.size()) throw std::runtime_error("GLM diagnostic tensor count differs");
    memcpy(bytes.data()+12, &position, sizeof(position));
    return bytes;
}

static void write_glm_diagnostic(const std::string & directory, const std::string & name, const void * data, size_t size) {
    const auto path = directory + "/" + name;
    FILE * file = fopen(path.c_str(), "wbx");
    if (!file) throw std::runtime_error("cannot create GLM diagnostic " + path);
    const bool written = fwrite(data, 1, size, file) == size;
    const bool closed = fclose(file) == 0;
    if (!written || !closed) throw std::runtime_error("cannot persist GLM diagnostic " + path);
}


struct glm_tensor_capture {
    std::map<std::string, std::vector<uint8_t>> values;

    static bool callback(ggml_tensor * tensor, bool ask, void * user_data) {
        auto * capture = static_cast<glm_tensor_capture *>(user_data);
        const std::string name(tensor->name);
        const bool wanted = name == "kda_qkv-1" || name == "kda_conv-1" || name == "kda_qk_norm-1" ||
                            name == "kda_gate-1" || name == "kda_beta-1";
        if (ask) {
            return wanted;
        }
        if (wanted) {
            auto & bytes = capture->values[name];
            bytes.resize(ggml_nbytes(tensor));
            ggml_backend_tensor_get(tensor, bytes.data(), 0, bytes.size());
        }
        return true;
    }
};

struct glm_round_evidence {
    std::string directory;
    size_t seed;
    bool unified;
    int seq, position, width, keep;
    bool capture = false, prefix_passed = false, correction_passed = false, correction_checked = false, cancelled = false;
    std::string prefix_reason, correction_reason, diagnostic_error;
    std::map<std::string, std::vector<uint8_t>> files;
    std::set<std::string> written;

    void add(const std::string & name, const std::vector<uint8_t> & bytes) {
        if (!directory.empty()) files[name] = bytes;
    }
    void add(const std::string & name, const void * data, size_t size) {
        if (directory.empty()) return;
        const auto * bytes = static_cast<const uint8_t *>(data);
        files[name] = std::vector<uint8_t>(bytes, bytes+size);
    }
    void snapshot(const std::string & name, const std::function<std::vector<uint8_t>()> & read) {
        try { add(name, read()); }
        catch (const std::exception & error) {
            diagnostic_error += name + ": " + error.what() + "; ";
            LOG_ERR("GLM snapshot diagnostic: %s\n", diagnostic_error.c_str());
        }
    }
    std::string report() const {
        std::ostringstream out;
        out << "seed=" << seed << "\nunified=" << unified << "\nsequence=" << seq << "\nposition=" << position
            << "\nwidth=" << width << "\nkeep=" << keep << "\nprefix_passed=" << prefix_passed
            << "\nprefix_reason=" << prefix_reason << "\ncorrection_checked=" << correction_checked
            << "\ncorrection_kind=" << (cancelled ? "cancel-reuse" : "continuation")
            << "\ncontinuation_passed=" << (correction_checked && !cancelled ? (correction_passed ? "1" : "0") : "not-measured")
            << "\nreuse_passed=" << (correction_checked && cancelled ? (correction_passed ? "1" : "0") : "not-measured")
            << "\ncorrection_reason=" << correction_reason << "\ndiagnostic_error=" << diagnostic_error
            << "\nstate_absolute_tolerance=1e-6\nlogit_absolute_tolerance=1e-4\n";
        return out.str();
    }
    void persist(const std::string & phase) {
        if (directory.empty() || !capture) return;
        for (const auto & file : files) {
            if (written.count(file.first)) continue;
            write_glm_diagnostic(directory, file.first, file.second.data(), file.second.size());
            written.insert(file.first);
        }
        const auto text = report();
        write_glm_diagnostic(directory, phase+".txt", text.data(), text.size());
    }
    bool prefix(const std::vector<uint8_t> & actual, const std::vector<uint8_t> & reference) {
        add("actual-prefix.bin", actual); add("reference-prefix.bin", reference);
        const auto comparison = compare_glm_states(actual, reference);
        prefix_passed = comparison.equal; prefix_reason = comparison.reason;
        capture = !directory.empty() && (!prefix_passed || !diagnostic_error.empty() || (!unified && width == 4 && keep == 1));
        if (!prefix_passed) LOG_ERR("GLM generated prefix: unified=%d seq=%d width=%d keep=%d state mismatch: %s\n",
                                   unified, seq, width, keep, prefix_reason.c_str());
        persist("prefix-report");
        return prefix_passed;
    }
    bool correction(const std::vector<std::pair<std::vector<uint8_t>, std::vector<uint8_t>>> & states,
                    const std::vector<float> & actual, const std::vector<float> & reference, bool was_cancelled) {
        correction_checked = true; cancelled = was_cancelled; correction_passed = true;
        add("actual-logits.f32", actual.data(), actual.size()*sizeof(float));
        add("reference-logits.f32", reference.data(), reference.size()*sizeof(float));
        if (actual.size() != reference.size()) {
            correction_passed = false; correction_reason = "logit count differs; ";
        } else {
            for (size_t i = 0; i < actual.size(); ++i) {
                if (!(std::isfinite(actual[i]) && std::isfinite(reference[i]) && std::fabs(actual[i]-reference[i]) <= 1e-4f)) {
                    correction_passed = false;
                    correction_reason += "logit mismatch at token " + std::to_string(i) + "; ";
                    break;
                }
            }
        }
        for (size_t i = 0; i < states.size(); ++i) {
            const auto suffix = "-after-seq" + std::to_string(i) + ".bin";
            add("actual"+suffix, states[i].first); add("reference"+suffix, states[i].second);
            const auto comparison = compare_glm_states(states[i].first, states[i].second);
            if (!comparison.equal) {
                correction_passed = false;
                correction_reason += "sequence " + std::to_string(i) + ": " + comparison.reason + "; ";
            }
        }
        add("actual-after.bin", states.at(seq).first); add("reference-after.bin", states.at(seq).second);
        if (!correction_passed) LOG_ERR("GLM correction: unified=%d seq=%d width=%d keep=%d %s\n",
                                       unified, seq, width, keep, correction_reason.c_str());
        capture = !directory.empty() && (capture || !correction_passed);
        persist("correction-report");
        return correction_passed;
    }
};

static void add_glm_tensor_capture(glm_round_evidence & evidence, const glm_tensor_capture & actual,
        const glm_tensor_capture & reference) {
    std::ostringstream report;
    for (const char * name : {"kda_qkv-1", "kda_conv-1", "kda_qk_norm-1", "kda_gate-1", "kda_beta-1"}) {
        const auto it_actual = actual.values.find(name);
        const auto it_reference = reference.values.find(name);
        if (it_actual == actual.values.end() || it_reference == reference.values.end()) {
            report << name << "=not-captured\n";
            continue;
        }
        evidence.add(std::string("actual-") + name + ".f32", it_actual->second);
        evidence.add(std::string("reference-") + name + ".f32", it_reference->second);
        const auto & batch = it_actual->second;
        const auto & serial = it_reference->second;
        size_t first = 0;
        while (first < serial.size() && first < batch.size() && batch[first] == serial[first]) ++first;
        report << name << ": batch_bytes=" << batch.size() << " serial_bytes=" << serial.size()
               << " first_different_byte=" << (first == serial.size() && batch.size() >= serial.size() ? -1 : (int64_t) first) << "\n";
    }
    const auto text = report.str();
    evidence.add("batch-input-report.txt", text.data(), text.size());
}

static int test_glm_capture_failures() {
    int failed = 0, checked = 0;
    const auto check = [&](bool ok, const char * name) {
        ++checked;
        if (!ok && failed++ < 12) fprintf(stderr, "GLM capture regression: %s\n", name);
    };
    const auto root = std::filesystem::temp_directory_path() / ("glm-capture-"+std::to_string(std::random_device{}()));
    if (!std::filesystem::create_directory(root)) throw std::runtime_error("cannot create capture test directory");
    const auto read = [](const std::filesystem::path & path) {
        std::ifstream file(path, std::ios::binary);
        return std::vector<uint8_t>(std::istreambuf_iterator<char>(file), {});
    };
    std::mt19937 rng(2345);
    for (int trial = 0; trial < 32; ++trial) {
        std::vector<std::pair<std::vector<uint8_t>, std::vector<uint8_t>>> states;
        for (int seq = 0; seq < 3; ++seq) {
            std::vector<uint8_t> raw;
            const auto append = [&](const auto value) {
                const auto * bytes = reinterpret_cast<const uint8_t *>(&value);
                raw.insert(raw.end(), bytes, bytes+sizeof(value));
            };
            append(uint32_t(0xaf143cd8)); append(llama_seq_id(seq)); append(uint32_t(1));
            append(llama_pos(264)); append(uint32_t(0)); append(uint32_t(0)); append(uint32_t(1));
            for (int tensor = 0; tensor < 2; ++tensor) {
                append(int32_t(GGML_TYPE_F32)); append(uint64_t(4)); append(float(rng()%1000)/1000.0f);
            }
            states.emplace_back(raw, raw);
        }
        for (int failure = 0; failure < 4; ++failure) {
            const auto directory = root / (std::to_string(trial)+"-"+std::to_string(failure));
            std::filesystem::create_directory(directory);
            glm_round_evidence evidence {directory.string(), 1234, false, 2, 264, 1, 1};
            auto actual = states[2].first, reference = states[2].second;
            bool threw = false;
            try {
                if (failure == 0) {
                    if (trial%2) actual.pop_back();
                    else { const float nan = NAN; memcpy(actual.data()+40, &nan, sizeof(nan)); }
                    reference = actual;
                    evidence.snapshot("snapshot-selected.bin", [&] {
                        const auto view = read_glm_state(actual);
                        if (!view.error.empty()) throw std::runtime_error(view.error);
                        return actual;
                    });
                    check(!evidence.prefix(actual, reference), "invalid prefix remains failed");
                    // The correction/control step has not run yet: both raw states must already be durable.
                    check(read(directory/"actual-prefix.bin") == actual && read(directory/"reference-prefix.bin") == reference,
                          "invalid raw prefix pair retained before continuation");
                } else if (failure == 1) {
                    check(evidence.prefix(actual, reference), "equal prefix passes");
                    check(!evidence.correction(states, {0, 1}, {0, 2}, false), "bad continuation logits fail");
                    check(!read(directory/"actual-logits.f32").empty() && !read(directory/"correction-report.txt").empty(),
                          "early continuation logit failure captured");
                } else if (failure == 2) {
                    check(evidence.prefix(actual, reference), "equal selected prefix passes");
                    auto other = states;
                    const float changed = 4.0f; memcpy(other[1].first.data()+40, &changed, sizeof(changed));
                    check(!evidence.correction(other, {0, 1}, {0, 1}, false), "other-sequence correction failure fails");
                    check(read(directory/"actual-after-seq1.bin") == other[1].first &&
                          read(directory/"reference-after-seq1.bin") == other[1].second,
                          "other-sequence failing raw pair retained");
                } else {
                    evidence.width = 3;
                    const float changed = 4.0f; memcpy(actual.data()+40, &changed, sizeof(changed));
                    check(!evidence.prefix(actual, reference), "cancelled prefix failure remains failed");
                    check(evidence.correction(states, {0, 1}, {0, 1}, true), "reuse can pass separately");
                    check(evidence.report().find("continuation_passed=not-measured\n") != std::string::npos &&
                          evidence.report().find("reuse_passed=1\n") != std::string::npos,
                          "reuse does not claim continuation of discarded state");
                }
            } catch (const std::exception &) { threw = true; }
            check(!threw, "optional snapshot error must not abort evidence capture");
        }
    }
    std::filesystem::remove_all(root);
    fprintf(stderr, "GLM capture regressions: %d checks, %d failed\n", checked, failed);
    return failed ? 1 : 0;
}

static bool test_glm_generated_prefixes(const size_t seed, const std::vector<ggml_backend_dev_t> & devs,
        llama_split_mode split_mode, const std::vector<float> & tensor_split, const std::string & diagnostic_dir = "", bool q8_cache) {
    auto gguf = get_gguf_ctx(LLM_ARCH_GLM5NEXT, true, true);
    kv_writer ms(LLM_ARCH_GLM5NEXT, gguf.get());
    if (split_mode == LLAMA_SPLIT_MODE_DATA) {
        ms.add_kv(LLM_KV_EXPERT_COUNT, uint32_t(8));
        ms.add_kv(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT, uint32_t(64));
        ms.add_kv(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH, uint32_t(128));
    }
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_TOP_K, uint32_t(4));
    auto pair = get_model_and_ctx(gguf.get(), nullptr, seed, devs, split_mode, false, tensor_split, true);
    auto * model = pair.first.get();
    const bool partition = split_mode == LLAMA_SPLIT_MODE_DATA;
    std::pair<llama_model_ptr, llama_context_ptr> unsplit;
    if (partition) {
        unsplit = get_model_and_ctx(gguf.get(), nullptr, seed, {devs.front()}, LLAMA_SPLIT_MODE_LAYER, false, {}, true);
    }
    const int vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    std::mt19937 rng(seed);
    const bool diagnostics = !diagnostic_dir.empty();
    // The reference decodes only what a round keeps where that is the same arithmetic as decoding the round's whole
    // width and rolling back: on the host. A device computes a token with another kernel in a wider batch, the scores
    // of the indexer's pools round another way, and where two of them are closer than that rounding the token reads
    // another pool: a difference no bound of rounding holds (seed 2742707468 on a card: two pools tied at a score of 0
    // in the one-token batch, one of them at -7.7e-15 in the seven-token batch, the states 5e-3 of their largest value
    // apart). So a device's reference decodes the same width and rolls back as well, as the unsplit reference of a
    // model split by data does: two contexts that did the same must hold the same. The host configuration, which
    // every run of this test has, holds the rollback to the kept prefix.
    const bool same_width = partition || !on_host(devs);
    for (const bool unified : {false, true}) {
        if (split_mode == LLAMA_SPLIT_MODE_DATA && unified) { continue; }
        auto params = llama_context_default_params();
        params.n_ctx = 2048;
        params.n_seq_max = 3;
        params.n_rs_seq = 8;
        params.n_batch = 512;
        params.n_ubatch = 64;
        params.n_threads = params.n_threads_batch = 4;
        params.kv_unified = unified;
        if (split_mode == LLAMA_SPLIT_MODE_DATA) { params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED; }
        if (q8_cache) { params.type_k = params.type_v = params.type_s = params.type_idx = GGML_TYPE_Q8_0; }
        glm_tensor_capture actual_tensors, reference_tensors;
        auto actual_params = params;
        auto reference_params = params;
        if (diagnostics) {
            actual_params.cb_eval = glm_tensor_capture::callback;
            actual_params.cb_eval_user_data = &actual_tensors;
            reference_params.cb_eval = glm_tensor_capture::callback;
            reference_params.cb_eval_user_data = &reference_tensors;
        }
        llama_context_ptr actual(llama_init_from_model(model, actual_params));
        llama_context_ptr reference(llama_init_from_model(partition ? unsplit.first.get() : model, reference_params));
        if (!actual || !reference) return false;
        std::vector<llama_pos> positions(3, 0);
        const auto decode = [&](llama_context * ctx, int seq, int pos, const std::vector<llama_token> & input, int n) {
            llama_batch batch = llama_batch_init(n, 0, 1);
            for (int i = 0; i < n; ++i) batch_add(batch, input[i], pos + i, {seq}, i == n-1);
            const int status = llama_decode(ctx, batch);
            llama_batch_free(batch);
            return status == 0;
        };
        const auto random_tokens = [&](int n) {
            std::vector<llama_token> out(n);
            for (auto & t : out) t = (llama_token)(rng()%vocab);
            return out;
        };
        for (int seq = 0; seq < 3; ++seq) {
            const auto input = random_tokens(253 + seq);
            if (!decode(actual.get(), seq, 0, input, input.size()) ||
                !decode(reference.get(), seq, 0, input, input.size())) return false;
            positions[seq] = input.size();
        }
        for (int width = 1; width <= 8; ++width) {
            for (int keep = 1; keep <= width; ++keep) {
                const int seq = (width + keep)%3;
                const int pos = positions[seq];
                const auto input = random_tokens(width);
                glm_round_evidence evidence {diagnostic_dir, seed, unified, seq, pos, width, keep};
                std::vector<uint8_t> before_full;
                if (diagnostics) {
                    evidence.add("actual-before.bin", glm_state_bytes(actual.get(), seq));
                    evidence.add("reference-before.bin", glm_state_bytes(reference.get(), seq));
                    before_full = glm_state_bytes(reference.get(), seq, true);
                    evidence.add("reference-before-full.bin", before_full);
                    evidence.add("input.i32", input.data(), input.size()*sizeof(llama_token));
                }
                // DP checks the same rollback on an unsplit model, a device on a second context; the CPU test checks
                // the kept-prefix oracle.
                if (diagnostics) {
                    actual_tensors.values.clear();
                    reference_tensors.values.clear();
                }
                if (!decode(actual.get(), seq, pos, input, width) ||
                    !decode(reference.get(), seq, pos, input, same_width ? width : keep)) return false;
                if (diagnostics) {
                    add_glm_tensor_capture(evidence, actual_tensors, reference_tensors);
                }
                if (diagnostics) {
                    evidence.snapshot("snapshot-selected.bin", [&] { return glm_snapshot(actual.get(), seq, width-keep, pos+keep-1); });
                    if (keep < width) evidence.snapshot("snapshot-late.bin", [&] { return glm_snapshot(actual.get(), seq, width-keep-1, pos+keep); });
                }
                if (keep < width && !llama_memory_seq_rm(llama_get_memory(actual.get()), seq, pos+keep, -1)) return false;
                if (same_width && keep < width && !llama_memory_seq_rm(llama_get_memory(reference.get()), seq, pos+keep, -1)) return false;
                const bool prefix_ok = evidence.prefix(glm_state_bytes(actual.get(), seq), glm_state_bytes(reference.get(), seq));
                if (!prefix_ok && !diagnostics) return false;
                positions[seq] += keep;
                const bool cancelled = keep < width && width%3 == 0 && keep == 1;
                if (cancelled) {
                    if (!llama_memory_seq_rm(llama_get_memory(actual.get()), seq, -1, -1) ||
                        !llama_memory_seq_rm(llama_get_memory(reference.get()), seq, -1, -1)) return false;
                    positions[seq] = 0;
                }
                const auto correction = random_tokens(3);
                if (diagnostics) evidence.add("correction.i32", correction.data(), correction.size()*sizeof(llama_token));
                if (!decode(actual.get(), seq, positions[seq], correction, correction.size()) ||
                    !decode(reference.get(), seq, positions[seq], correction, correction.size())) return false;
                positions[seq] += correction.size();
                const float * a = llama_get_logits_ith(actual.get(), correction.size()-1);
                const float * b = llama_get_logits_ith(reference.get(), correction.size()-1);
                if (!a || !b) return false;
                std::vector<std::pair<std::vector<uint8_t>, std::vector<uint8_t>>> states;
                for (int s = 0; s < 3; ++s) states.emplace_back(glm_state_bytes(actual.get(), s), glm_state_bytes(reference.get(), s));
                const bool correction_ok = evidence.correction(states, {a, a+vocab}, {b, b+vocab}, cancelled);
                if (evidence.capture) {
                    // Independent controls start from the same pre-round reference history.
                    llama_context_ptr batched(llama_init_from_model(model, params));
                    llama_context_ptr serial(llama_init_from_model(model, params));
                    for (auto * ctx : {batched.get(), serial.get()}) {
                        if (!ctx || llama_state_set_data(ctx, before_full.data(), before_full.size()) != before_full.size()) {
                            throw std::runtime_error("GLM diagnostic control restore failed");
                        }
                    }
                    if (!decode(batched.get(), seq, pos, input, width)) throw std::runtime_error("GLM diagnostic batch decode failed");
                    evidence.add("control-batched.bin", glm_snapshot(batched.get(), seq, width-keep, pos+keep-1));
                    if (keep == 1) evidence.add("control-early.bin", glm_state_bytes(serial.get(), seq));
                    for (int token = 0; token < std::min(width, keep+1); ++token) {
                        if (!decode(serial.get(), seq, pos+token, {input[token]}, 1)) throw std::runtime_error("GLM diagnostic serial decode failed");
                        if (token+1 == keep-1) evidence.add("control-early.bin", glm_state_bytes(serial.get(), seq));
                        if (token+1 == keep) evidence.add("control-serial.bin", glm_state_bytes(serial.get(), seq));
                        if (token+1 == keep+1) evidence.add("control-late.bin", glm_state_bytes(serial.get(), seq));
                    }
                    evidence.persist("report");
                    return prefix_ok && correction_ok && evidence.diagnostic_error.empty();
                }
                if (!correction_ok) return false;
            }
        }
    }
    return true;
}

static bool moe_mandatory(const llm_arch arch) {
    switch (arch) {
        case LLM_ARCH_FLINT:
        case LLM_ARCH_LLAMA4:
        case LLM_ARCH_MAION_CODER:
        case LLM_ARCH_COHERE2MOE:
        case LLM_ARCH_GROK:
        case LLM_ARCH_QWEN2MOE:
        case LLM_ARCH_QWEN3MOE:
        case LLM_ARCH_QWEN3NEXT:
        case LLM_ARCH_QWEN3VLMOE:
        case LLM_ARCH_QWEN35MOE:
        case LLM_ARCH_CLOVER_MOE:
        case LLM_ARCH_QWEN4EXP:
        case LLM_ARCH_PHIMOE:
        case LLM_ARCH_DBRX:
        case LLM_ARCH_OLMOE:
        case LLM_ARCH_ARCTIC:
        case LLM_ARCH_DEEPSEEK:
        case LLM_ARCH_DEEPSEEK2:
        case LLM_ARCH_DEEPSEEK32:
        case LLM_ARCH_DOTS3NOTE:
        case LLM_ARCH_DEEPSEEK4:
        case LLM_ARCH_DEEPSEEK41:
        case LLM_ARCH_XING4_0:
        case LLM_ARCH_GLM4_MOE:
        case LLM_ARCH_GLM_DSA:
        case LLM_ARCH_GLM5NEXT:
        case LLM_ARCH_EXAONE_MOE:
        case LLM_ARCH_BAILINGMOE:
        case LLM_ARCH_BAILINGMOE2:
        case LLM_ARCH_BAILINGMOE3:
        case LLM_ARCH_DOTS1:
        case LLM_ARCH_AFMOE:
        case LLM_ARCH_BEAM:
        case LLM_ARCH_ERNIE4_5:
        case LLM_ARCH_ERNIE4_5_MOE:
        case LLM_ARCH_HUNYUAN_MOE:
        case LLM_ARCH_HY_V3:
        case LLM_ARCH_HY_V4:
        case LLM_ARCH_OPENAI_MOE:
        case LLM_ARCH_LFM2MOE:
        case LLM_ARCH_SMALLTHINKER:
        case LLM_ARCH_LLADA_MOE:
        case LLM_ARCH_GROVEMOE:
        case LLM_ARCH_MINIMAX_01:
        case LLM_ARCH_MINIMAX_M2:
        case LLM_ARCH_MINIMAX_M3:
        case LLM_ARCH_RND1:
        case LLM_ARCH_PADDLEOCR:
        case LLM_ARCH_MIMO2:
        case LLM_ARCH_KIMI_LINEAR:
        case LLM_ARCH_KIMI_K3:
        case LLM_ARCH_KIMI_K31:
        case LLM_ARCH_GLM5_NEXT:
        case LLM_ARCH_STEP35:
        case LLM_ARCH_MISTRAL4:
        case LLM_ARCH_MISTRAL_LARGE4:
        case LLM_ARCH_MELLUM:
        case LLM_ARCH_LAGUNA:
        case LLM_ARCH_MAPLE:
        case LLM_ARCH_MUSE_SPARK:
        case LLM_ARCH_KOLIBRI1:
        case LLM_ARCH_METIS16:
            return true;
        default:
            return false;
    }
}

static bool moe_implemented(const llm_arch arch) {
    if (moe_mandatory(arch)) {
        return true;
    }
    switch (arch) {
        case LLM_ARCH_LLAMA:
        case LLM_ARCH_REFACT:
        case LLM_ARCH_MINICPM:
        case LLM_ARCH_GRANITE:
        case LLM_ARCH_GRANITE_MOE:
        case LLM_ARCH_MISTRAL3:
        case LLM_ARCH_LLAMA_EMBED:
        case LLM_ARCH_K2_HORIZON:
            return true;
        default:
            return false;
    }
}

static bool arch_supported(const llm_arch arch) {
    if (arch == LLM_ARCH_CLIP || arch == LLM_ARCH_GPTJ || arch == LLM_ARCH_UNKNOWN) {
        return false; // These models don't have usable implementations.
    }
    if (arch == LLM_ARCH_CHAMELEON) {
        return false; // Only half-implemented and to be removed in the future.
    }
    if (arch == LLM_ARCH_WAVTOKENIZER_DEC) {
        return false; // FIXME CUDA backend crashes.
    }
    if (arch == LLM_ARCH_GRANITE_SWITCH) {
        return false; // FIXME adapter fixture
    }
    if (arch == LLM_ARCH_LLAMA_EMBED || arch == LLM_ARCH_GEMMA_EMBEDDING || arch == LLM_ARCH_GEMMA_EMBEDDING2 || arch == LLM_ARCH_T5ENCODER) {
        return false; // FIXME Embedding (?) models produce inconsistent results.
    }
    if (arch == LLM_ARCH_RWKV6 || arch == LLM_ARCH_RWKV6QWEN2 || arch == LLM_ARCH_RWKV7 || arch == LLM_ARCH_ARWKV7) {
        return false; // FIXME RWKV models hang indefinitely.
    }
    if (arch == LLM_ARCH_BERT || arch == LLM_ARCH_MODERN_BERT || arch == LLM_ARCH_NOMIC_BERT || arch == LLM_ARCH_NOMIC_BERT_MOE ||
            arch == LLM_ARCH_NEO_BERT || arch == LLM_ARCH_JINA_BERT_V2 || arch == LLM_ARCH_JINA_BERT_V3 || arch == LLM_ARCH_EUROBERT) {
        return false; // TODO vocab
    }
    if (arch == LLM_ARCH_PLM) {
        return false; // TODO tensor shapes
    }
    if (arch == LLM_ARCH_CLEF || arch == LLM_ARCH_VLLM_DECISION2) {
        return false; // TODO decision head tensors
    }
    if (arch == LLM_ARCH_DEEPSEEK2OCR) {
        return false;
    }
    // FIXME: these hit scheduler/view-backed-output issues with WebGPU on CI.
#ifdef GGML_USE_WEBGPU
    if (arch == LLM_ARCH_DEEPSEEK32 || arch == LLM_ARCH_GLM_DSA || arch == LLM_ARCH_DOTS3NOTE || arch == LLM_ARCH_QWEN4EXP ||
            arch == LLM_ARCH_HY_V4) {
        return false;
    }
#endif // GGML_USE_WEBGPU

    // FIXME: jamba produces incorrect output (~0.55 NMSE vs CPU) on the HIP
    // backend on RDNA3.5 (gfx1151); the SSM kernels need investigation.
#ifdef GGML_USE_HIP
    if (arch == LLM_ARCH_JAMBA) {
        return false;
    }
#endif // GGML_USE_HIP

    return true;
}

static int save_models(const std::string & arch_filter, const size_t seed, const float stdev, const int verbosity, const std::string & dir) {
    struct user_data_t {
        struct {
            ggml_log_callback callback;
            void * user_data;
        } log_old;

        int verbosity;

        user_data_t(int verbosity) : verbosity(verbosity) {
            llama_log_get(&log_old.callback, &log_old.user_data);
        }
    };
    user_data_t ud(verbosity);

    llama_log_set([](ggml_log_level level, const char * text, void * user_data) {
        const user_data_t * ud = (const user_data_t *) user_data;
        int verbosity = common_log_get_verbosity(level);
        if (verbosity <= ud->verbosity) {
            ud->log_old.callback(level, text, ud->log_old.user_data);
        }
    }, &ud);

    for (const llm_arch & arch : llama_arch_all()) {
        if (arch == LLM_ARCH_UNKNOWN) {
            continue;
        }
        if (!arch_matches(arch_filter, arch)) {
            continue;
        }
        const glm5_upstream_scope glm5_upstream(arch);
        if (arch == LLM_ARCH_EAGLE3 || arch == LLM_ARCH_DFLASH || arch == LLM_ARCH_GEMMA4_ASSISTANT) {
            continue;
        }
        for (bool moe : {false, true}) {
            if (moe && !moe_implemented(arch)) {
                continue;
            }
            if (!moe && moe_mandatory(arch)) {
                continue;
            }
            if (!llama_arch_supports_save(arch) || !arch_supported(arch)) {
                LOG_INF("%s: %s model (%s) is unsupported, skipping\n", __func__, llama_arch_name(arch), moe ? "MoE" : "dense");
                continue;
            }
            gguf_context_ptr gguf_ctx = get_gguf_ctx(arch, moe);
            llama_model_ptr model = get_model(gguf_ctx.get(), nullptr, seed, stdev, {});
            const std::string path = dir + "/" + llama_arch_name(arch) + (moe ? "-moe.gguf" : "-dense.gguf");
            LOG_INF("%s: '%s %s' -> %s ...", __func__, llama_arch_name(arch), moe ? "MoE" : "dense", path.c_str());
            llama_model_save_to_file(model.get(), path.c_str());
            const std::uintmax_t size = std::filesystem::file_size(path);
            LOG_CNT(" (%.2f MB)\n", (double) size / (1024.0 * 1024.0));
        }
    }
    llama_log_set(ud.log_old.callback, ud.log_old.user_data);
    return 0;
}

// the CPU and the GPU of an Apple SoC carry the same description, the name tells them apart (CPU, MTL0)
static std::string get_device_label(ggml_backend_dev_t dev) {
    const char * desc = ggml_backend_dev_description(dev);
    const char * name = ggml_backend_dev_name(dev);

    if (strcmp(desc, name) == 0) {
        return desc;
    }

    return std::string(desc) + " (" + name + ")";
}

static int test_backends(const std::string & arch_filter, const size_t seed, const float stdev, const int verbosity, const char * target_backend) {
    struct user_data_t {
        struct {
            ggml_log_callback callback;
            void * user_data;
        } log_old;

        int verbosity;

        user_data_t(int verbosity) : verbosity(verbosity) {
            llama_log_get(&log_old.callback, &log_old.user_data);
        }
    };
    user_data_t ud(verbosity);

    llama_log_set([](ggml_log_level level, const char * text, void * user_data) {
        const user_data_t * ud = (const user_data_t *) user_data;
        int verbosity = common_log_get_verbosity(level);
        if (verbosity <= ud->verbosity) {
            ud->log_old.callback(level, text, ud->log_old.user_data);
        }
    }, &ud);

    const std::vector<llama_token> tokens = get_tokens(128, 128, seed);

    struct device_config {
        std::vector<ggml_backend_dev_t> devs;
        std::string                     label;
        llama_split_mode                split_mode;
        std::vector<float>              tensor_split; // empty: the default, even split
        bool                            host_experts; // keep the experts in host memory, see host_experts_test
        size_t                          moe_cache_size;

        device_config(std::vector<ggml_backend_dev_t> devs, std::string name, llama_split_mode split_mode, bool host_experts = false, size_t moe_cache_size = 0)
            : devs(std::move(devs)), label(std::move(name)), split_mode(split_mode), host_experts(host_experts), moe_cache_size(moe_cache_size) {}
        device_config(std::vector<ggml_backend_dev_t> devs, std::string name, llama_split_mode split_mode, std::vector<float> tensor_split)
            : devs(std::move(devs)), label(std::move(name)), split_mode(split_mode), tensor_split(std::move(tensor_split)), host_experts(false), moe_cache_size(0) {}
    };

    const llama_model_tensor_buft_override host_experts_overrides[] = {
        { LLM_FFN_EXPS_REGEX, ggml_backend_cpu_buffer_type() },
        { nullptr, nullptr },
    };

    std::vector<device_config> dev_configs;
    size_t max_device_label_length = 4;
    {
        std::vector<ggml_backend_dev_t> devices_meta;
        std::vector<ggml_backend_dev_t> devices_dp;
        {
            const size_t device_count = ggml_backend_dev_count();
            for (size_t i = 0; i < device_count; i++) {
                ggml_backend_dev_t dev = ggml_backend_dev_get(i);
                if (target_backend != nullptr && strcmp(target_backend, ggml_backend_dev_name(dev)) != 0) {
                    continue;
                }
                dev_configs.emplace_back(std::vector<ggml_backend_dev_t>{dev}, get_device_label(dev), LLAMA_SPLIT_MODE_LAYER);
                max_device_label_length = std::max(max_device_label_length, dev_configs.back().label.length());

                // cpu-based devices cannot be used in tensor split mode
                if (ggml_backend_dev_buffer_type(dev) != ggml_backend_cpu_buffer_type()) {
                    devices_meta.push_back(dev);
                }
                if (strncmp(ggml_backend_dev_name(dev), "CUDA", 4) == 0) {
                    devices_dp.push_back(dev);
                }
            }
        }
        // one GPU twice splits as two GPUs do, so that a machine with one tests the splits too
        const bool one_gpu_twice = devices_meta.size() == 1;
        if (one_gpu_twice) {
            devices_meta.push_back(devices_meta[0]);
        }

        if (target_backend == nullptr) {
            dev_configs.emplace_back(devices_meta, "Meta", LLAMA_SPLIT_MODE_TENSOR);
        }

        // the ops that use the experts are offloaded to the first device and the scheduler copies the used experts
        if (!devices_meta.empty()) {
            dev_configs.emplace_back(devices_meta, "Host experts", LLAMA_SPLIT_MODE_LAYER, true);
            max_device_label_length = std::max(max_device_label_length, dev_configs.back().label.length());
        }

        // the ops that use the host experts run on a GPU and read the experts from a cache
        // the cache has only a few slots (4 for 288 KiB experts), so the experts are evicted and uploaded again
        if (!devices_meta.empty()) {
            const enum ggml_backend_dev_type type = ggml_backend_dev_type(devices_meta[0]);
            if (type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
                dev_configs.emplace_back(std::vector<ggml_backend_dev_t>{devices_meta[0]}, "MoE cache", LLAMA_SPLIT_MODE_LAYER, true, 1536*1024);
                max_device_label_length = std::max(max_device_label_length, dev_configs.back().label.length());
                // each GPU caches the layers assigned to it. The cache finds a layer's GPU by its device
                // (llama_moe_cache::impl), so one GPU listed twice is one cache for the layers of two backends: not a
                // case it has (deepseek2 came out 1.3e-2 off on the fork's one-GPU-twice list)
                if (devices_meta.size() > 1 && !one_gpu_twice) {
                    dev_configs.emplace_back(devices_meta, "MoE cache, layer split", LLAMA_SPLIT_MODE_LAYER, true, 1536*1024);
                    max_device_label_length = std::max(max_device_label_length, dev_configs.back().label.length());
                }
            }
        }

        // 3:7 puts the boundaries off every power-of-two granularity, so a tensor that splits out of step with the ones
        // it is computed with (a weight and its bias, the heads of q and of the output projection) fails here
        std::vector<float> tensor_split_uneven(llama_max_devices(), 0.0f);
        for (size_t j = 0; j < devices_meta.size(); j++) {
            tensor_split_uneven[j] = j % 2 == 0 ? 3.0f : 7.0f;
        }
        dev_configs.emplace_back(devices_meta, "Meta 3:7", LLAMA_SPLIT_MODE_TENSOR, tensor_split_uneven);
        max_device_label_length = std::max(max_device_label_length, dev_configs.back().label.length());

        if (devices_dp.empty()) {
            if (auto * cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU)) { devices_dp.push_back(cpu); }
        }
        if (!devices_dp.empty()) {
            for (const size_t ranks : {4u, 2u}) {
                std::vector<ggml_backend_dev_t> devs;
                for (size_t i = 0; i < ranks; ++i) { devs.push_back(devices_dp[i % devices_dp.size()]); }
                const std::string label = "DP " + std::to_string(ranks) + (devices_dp.size() < ranks ? " (logical)" : "");
                dev_configs.emplace(dev_configs.begin(), devs, label, LLAMA_SPLIT_MODE_DATA);
                max_device_label_length = std::max(max_device_label_length, label.length());
            }
        }
    }

    size_t max_arch_name_length = 0;
    for (const llm_arch & arch : llama_arch_all()) {
        max_arch_name_length = std::max(max_arch_name_length, strlen(llama_arch_name(arch)));
    }

    const std::string template_header  = std::string("|%" + std::to_string(max_arch_name_length) + "s|%") + std::to_string(max_device_label_length) + "s|%6s|%15s|%9s|%15s|\n";
    const std::string template_row_cfg = std::string("|%" + std::to_string(max_arch_name_length) + "s|%") + std::to_string(max_device_label_length) + "s|%6s|";
    const std::string template_row_res = "%15s %10s|%20s|%15s %10s|\n";

    bool all_ok = true;
    size_t n_tests = 0;
    size_t n_failed = 0;
    common_log_flush(common_log_main());
    LOG(template_header.c_str(), "Model arch.", "Device", "Config", "NMSE vs. CPU", "Roundtrip", "Mixed batch");
    LOG("|");
    for (size_t i = 0; i < max_arch_name_length; i++) {
        LOG("-");
    }
    LOG("|");
    for (size_t i = 0; i < max_device_label_length; i++) {
        LOG("-");
    }
    LOG("|------|---------------|---------|---------------|\n");
    for (const llm_arch & arch : llama_arch_all()) {
        if (arch == LLM_ARCH_UNKNOWN) {
            continue;
        }
        if (!arch_matches(arch_filter, arch)) {
            continue;
        }
        const glm5_upstream_scope glm5_upstream(arch);
        if (arch == LLM_ARCH_EAGLE3 || arch == LLM_ARCH_DFLASH || arch == LLM_ARCH_GEMMA4_ASSISTANT) {
            continue;
        }

        const bool encode = arch == LLM_ARCH_T5 || arch == LLM_ARCH_DREAM || arch == LLM_ARCH_LLADA || arch == LLM_ARCH_LLADA_MOE || arch == LLM_ARCH_RND1;
        for (bool moe : {false, true}) {
            if (moe && !moe_implemented(arch)) {
                continue;
            }
            if (!moe && moe_mandatory(arch)) {
                continue;
            }
            const std::string config_name = moe ? "MoE" : "Dense";
            gguf_context_ptr gguf_ctx = get_gguf_ctx(arch, moe);
            if (arch == LLM_ARCH_BAILINGMOE3) {
                GGML_ASSERT(gguf_remove_key(gguf_ctx.get(), "bailingmoe3.kda.safe_gate") >= 0);
            }
            std::pair<llama_model_ptr, llama_context_ptr> model_and_ctx_cpu;
            std::vector<float> logits_cpu;
            for (device_config & dc : dev_configs) {
                if (dc.split_mode == LLAMA_SPLIT_MODE_DATA && arch != LLM_ARCH_GLM5NEXT) { continue; }
                if (dc.host_experts && (!moe || !host_experts_test(arch))) {
                    continue;
                }
                const llama_model_tensor_buft_override * overrides = dc.host_experts ? host_experts_overrides : nullptr;

                // print test config first; should anything fail during model loading or inference, at least we know which test case caused it
                LOG(template_row_cfg.c_str(), llama_arch_name(arch), dc.label.c_str(), config_name.c_str());
                fflush(stdout);

                if (dc.split_mode == LLAMA_SPLIT_MODE_DATA) {
                    const int plain = test_glm_dp(seed, dc.devs, false, dc.devs.size());
                    const int quantized = test_glm_dp(seed, dc.devs, true, dc.devs.size());
                    all_ok = all_ok && !plain && !quantized;
                    n_tests++;
                    n_failed += plain || quantized;
                    LOG(template_row_res.c_str(), plain || quantized ? "FAIL DP" : "OK DP", "KL <= 1e-6", "SKIP", "SKIP", "");
                    continue;
                }

                std::pair<llama_model_ptr, llama_context_ptr> model_and_ctx_dev;
                std::vector<float> logits_dev;
                std::string status_nmse      = "\033[1;33mSKIP\033[0m";
                std::string status_roundtrip = "\033[1;33mSKIP\033[0m";
                std::string status_mixed     = "\033[1;33mSKIP\033[0m";
                char nmse_str[12] = {0};
                char mixed_str[12] = {0};

                bool skip = !arch_supported(arch) || (dc.split_mode == LLAMA_SPLIT_MODE_TENSOR && dc.devs.empty());
                bool test_executed = false;
                bool test_ok = true;
                if (!skip) {
                    if (logits_cpu.empty()) {
                        model_and_ctx_cpu = get_model_and_ctx(gguf_ctx.get(), nullptr, seed, stdev, {}, LLAMA_SPLIT_MODE_LAYER, encode);
                        logits_cpu = get_logits(model_and_ctx_cpu.first.get(), model_and_ctx_cpu.second.get(), tokens, encode);
                    }
                    if (dc.split_mode != LLAMA_SPLIT_MODE_TENSOR || llama_arch_supports_sm_tensor(arch)) {
                        test_executed = true;
                        model_and_ctx_dev = get_model_and_ctx(gguf_ctx.get(), nullptr, seed, stdev, dc.devs, dc.split_mode, encode, overrides, dc.moe_cache_size, dc.tensor_split);
                        logits_dev = get_logits(model_and_ctx_dev.first.get(), model_and_ctx_dev.second.get(), tokens, encode);
                        const double nmse_val = nmse(logits_cpu, logits_dev);
                        snprintf(nmse_str, sizeof(nmse_str), "(%.2e)", nmse_val);
                        status_nmse = "\033[1;32mOK\033[0m";
                        if (nmse_val > 1e-4) {
                            test_ok = false;
                            status_nmse = "\033[1;31mFAIL\033[0m";
                        }
                        if ((arch == LLM_ARCH_GLM5NEXT || arch == LLM_ARCH_QWEN35) &&
                                !test_draft_vocab(arch, seed, dc.devs, dc.split_mode, dc.tensor_split)) {
                            test_ok = false;
                            status_nmse = "\033[1;31mFAIL MTP vocab\033[0m";
                        }
                        if ((arch == LLM_ARCH_GLM5NEXT || arch == LLM_ARCH_QWEN35) && dc.split_mode != LLAMA_SPLIT_MODE_DATA &&
                                !test_served_shapes(arch, seed, dc.devs, dc.split_mode, dc.tensor_split)) {
                            test_ok = false;
                            status_nmse = "\033[1;31mFAIL served shapes\033[0m";
                        }
                        if (arch == LLM_ARCH_GLM5NEXT && !test_glm_nextn_masked_rows(seed, dc.devs, dc.split_mode, dc.tensor_split)) {
                            test_ok = false;
                            status_nmse = "\033[1;31mFAIL nextn masked rows\033[0m";
                        }
                        if (arch == LLM_ARCH_GLM5NEXT && !test_glm_many_slots(seed, dc.devs, dc.split_mode, dc.tensor_split)) {
                            test_ok = false;
                            status_nmse = "\033[1;31mFAIL many slots\033[0m";
                        }
                        if (arch == LLM_ARCH_GLM5NEXT && !test_glm_generated_prefixes(seed, dc.devs, dc.split_mode, dc.tensor_split)) {
                            throw std::runtime_error("GLM generated speculative prefixes failed");
                        }
                        if (arch == LLM_ARCH_GLM5NEXT && !test_glm_short_dense(seed, dc.devs, dc.split_mode, dc.tensor_split)) {
                            test_ok = false;
                            status_nmse = "\033[1;31mFAIL short dense\033[0m";
                        }
                        if (arch == LLM_ARCH_GLM5NEXT && !test_glm_mtp_index(seed, dc.devs, dc.split_mode, dc.tensor_split)) {
                            test_ok = false;
                            status_nmse = "\033[1;31mFAIL MTP index\033[0m";
                        }
                        if (arch == LLM_ARCH_GLM5NEXT && !test_glm_mtp_index_sequences(seed, dc.devs, dc.split_mode, dc.tensor_split)) {
                            test_ok = false;
                            status_nmse = "\033[1;31mFAIL MTP sequences\033[0m";
                        }
                        if (arch == LLM_ARCH_GLM5NEXT && !test_glm_reserve_holds_reads(seed, dc.devs, dc.split_mode, dc.tensor_split)) {
                            test_ok = false;
                            status_nmse = "\033[1;31mFAIL reserve\033[0m";
                        }
                        if (arch == LLM_ARCH_GLM5NEXT && !test_glm_prefill_reuse(seed, dc.devs, dc.split_mode, dc.tensor_split)) {
                            test_ok = false;
                            status_nmse = "\033[1;31mFAIL prefill reuse\033[0m";
                        }
                        if (arch == LLM_ARCH_GLM5NEXT && !test_glm_shared_pool_reuse(seed, dc.devs, dc.split_mode, dc.tensor_split)) {
                            test_ok = false;
                            status_nmse = "\033[1;31mFAIL shared pool reuse\033[0m";
                        }
                        if (arch == LLM_ARCH_GLM5NEXT && !test_glm_shared_buffers(seed, dc.devs, dc.split_mode, dc.tensor_split)) {
                            test_ok = false;
                            status_nmse = "\033[1;31mFAIL shared buffers\033[0m";
                        }
                        if (arch == LLM_ARCH_GLM5NEXT && !test_glm_tok_embd_on_device(seed, dc.devs, dc.split_mode, dc.tensor_split)) {
                            test_ok = false;
                            status_nmse = "\033[1;31mFAIL tok_embd on device\033[0m";
                        }
                        if (arch == LLM_ARCH_GLM5NEXT && !test_glm_nextn_store(seed, dc.devs, dc.split_mode, dc.tensor_split)) {
                            test_ok = false;
                            status_nmse = "\033[1;31mFAIL nextn store\033[0m";
                        }
                        if (arch == LLM_ARCH_GLM5NEXT && !test_glm_nextn_masked_rows(seed, dc.devs, dc.split_mode, dc.tensor_split)) {
                            test_ok = false;
                            status_nmse = "\033[1;31mFAIL nextn masked rows\033[0m";
                        }
                        if (arch == LLM_ARCH_GLM5NEXT && !test_glm_many_slots(seed, dc.devs, dc.split_mode, dc.tensor_split)) {
                            test_ok = false;
                            status_nmse = "\033[1;31mFAIL many slots\033[0m";
                        }

                        // chunked decode matches a single batch only with causal attention over a memory
                        llama_context * lctx_dev = model_and_ctx_dev.second.get();
                        if (!encode && llama_get_memory(lctx_dev) != nullptr) {
                            std::vector<float> embd_mixed((size_t) llama_model_n_embd_inp(model_and_ctx_dev.first.get())*tokens.size()/4);
                            std::mt19937 gen(seed);
                            std::normal_distribution<float> dis(0.0f, stdev);
                            for (float & v : embd_mixed) {
                                v = dis(gen);
                            }
                            std::vector<float> logits_mixed;
                            std::vector<float> logits_chunks;
                            if (llama_arch_supports_mixed_batch(arch)) {
                                if (get_logits_mixed(model_and_ctx_dev.first.get(), lctx_dev, tokens, embd_mixed, false, logits_chunks) != 0 ||
                                    get_logits_mixed(model_and_ctx_dev.first.get(), lctx_dev, tokens, embd_mixed, true,  logits_mixed)  != 0) {
                                    throw std::runtime_error("failed to decode mixed batch");
                                }
                                const double nmse_mixed = nmse(logits_chunks, logits_mixed);
                                snprintf(mixed_str, sizeof(mixed_str), "(%.2e)", nmse_mixed);
                                status_mixed = "\033[1;32mOK\033[0m";
                                if (nmse_mixed > 1e-4) {
                                    test_ok = false;
                                    status_mixed = "\033[1;31mFAIL\033[0m";
                                }
                            } else {
                                // must be rejected as an invalid batch, mute the expected error log
                                ud.verbosity = LOG_LEVEL_OUTPUT;
                                const int32_t err = get_logits_mixed(model_and_ctx_cpu.first.get(), model_and_ctx_cpu.second.get(), tokens, embd_mixed, true, logits_mixed);
                                ud.verbosity = verbosity;
                                if (err != -1) {
                                    test_ok = false;
                                    status_mixed = "\033[1;31mFAIL\033[0m";
                                }
                            }
                        }

                        // runs after the mixed batch check, as it leaves the context with non-causal attention
                        if (!encode && !check_causal_attn_toggle(model_and_ctx_dev.first.get(), model_and_ctx_dev.second.get(), tokens)) {
                            if (test_ok) {
                                status_nmse = "\033[1;31mFAIL\033[0m (toggle)";
                            }
                            test_ok = false;
                        }
                    }

                    if (llama_arch_supports_save(arch) && dc.split_mode != LLAMA_SPLIT_MODE_TENSOR) {
                        test_executed = true;
                        GGML_ASSERT(model_and_ctx_dev.first && model_and_ctx_dev.second);
                        // FIXME: when adding a tensor to a gguf_context a copy is made, this changes the pointer which the meta backend
                        //     in turn uses to map the tensors to their simple equivalents - this is fundamentally incompatible
                        const std::string path = get_tmp_gguf_path();
                        llama_model_save_to_file(model_and_ctx_dev.first.get(), path.c_str());
                        FILE * file = fopen(path.c_str(), "rb");
                        GGML_ASSERT(file != nullptr);

                        {
                            auto model_and_ctx_roundtrip = get_model_and_ctx(nullptr, file, seed, stdev, dc.devs, dc.split_mode, encode, overrides, dc.moe_cache_size);
                            const std::vector<float> logits_roundtrip = get_logits(
                                model_and_ctx_roundtrip.first.get(), model_and_ctx_roundtrip.second.get(), tokens, encode);
                            status_roundtrip = "\033[1;32mOK\033[0m";
                            GGML_ASSERT(logits_roundtrip.size() == logits_dev.size());
                            for (size_t i = 0; i < logits_roundtrip.size(); i++) {
                                if (logits_roundtrip[i] != logits_dev[i]) {
                                    test_ok = false;
                                    status_roundtrip = "\033[1;31mFAIL\033[0m";
                                    break;
                                }
                            }
                        }

                        fclose(file);
                        std::filesystem::remove(path);
                    }
                }

                if (test_executed) {
                    n_tests++;
                    if (!test_ok) {
                        n_failed++;
                        all_ok = false;
                    }
                }

                // log the results for this test case
                LOG(template_row_res.c_str(), status_nmse.c_str(), nmse_str, status_roundtrip.c_str(), status_mixed.c_str(), mixed_str);
            }
        }
    }

    if (n_tests == 0) {
        LOG("Summary: no tests executed\n");
    } else if (n_failed == 0) {
        LOG("Summary: all %zu test(s) passed\n", n_tests);
    } else {
        LOG("Summary: %zu test(s) executed, %zu failed\n", n_tests, n_failed);
    }

    llama_log_set(ud.log_old.callback, ud.log_old.user_data);
    return all_ok ? 0 : 1;
}

struct kv_shift_move {
    llama_pos p0;
    llama_pos p1;
    llama_pos delta;
};

static void decode_range(llama_context * lctx, const llama_token * tokens, uint32_t n_tokens, llama_pos pos0, llama_seq_id seq_id, bool output) {
    common_batch batch(lctx);
    for (uint32_t i = 0; i < n_tokens; i++) {
        batch.add(tokens[i], pos0 + i, seq_id, output || i == n_tokens - 1);
    }
    if (llama_process(lctx, LLAMA_PROCESS_TYPE_DECODE, batch.get())) {
        throw std::runtime_error("failed to decode batch");
    }
}

// the raw attention scores of each layer are linear in K, unlike the logits they do not depend on how sharp the attention is
struct kq_capture {
    bool active = false;
    std::vector<std::pair<std::string, std::vector<float>>> data;
};

// the scores are compared cell by cell, so the cells of the probe must match: prefix and probe fit in the SWA window,
// otherwise the probe can reuse a masked SWA cell in one run and not in the other
static bool kq_capture_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    kq_capture * cap = (kq_capture *) user_data;
    const char * name = ggml_get_name(t);
    const bool want = cap->active && t->type == GGML_TYPE_F32 && ggml_is_contiguous(t) &&
        (strncmp(name, "kq-", 3) == 0 || strncmp(name, "indexer_kq-", 11) == 0 || strncmp(name, "indexer_score-", 14) == 0);
    if (ask) {
        return want;
    }
    if (want) {
        std::vector<float> buf(ggml_nelements(t));
        ggml_backend_tensor_get(t, buf.data(), 0, ggml_nbytes(t));
        auto it = std::find_if(cap->data.begin(), cap->data.end(), [&](const auto & e) { return e.first == name; });
        if (it == cap->data.end()) {
            cap->data.emplace_back(name, std::move(buf));
        } else {
            it->second = std::move(buf);
        }
    }
    return true;
}

struct kq_diff {
    double      worst = 0.0;
    std::string first_bad; // first tensor above the threshold, later layers can inherit its error through Q
    std::string bad;       // all tensors above the threshold, with their nmse
};

static kq_diff compare_kq(const std::vector<std::pair<std::string, std::vector<float>>> & ref,
                          const std::vector<std::pair<std::string, std::vector<float>>> & cur, double thold) {
    if (ref.size() != cur.size()) {
        throw std::runtime_error("kq tensor count mismatch");
    }
    kq_diff res;
    for (size_t i = 0; i < ref.size(); i++) {
        if (ref[i].first != cur[i].first || ref[i].second.size() != cur[i].second.size()) {
            throw std::runtime_error("kq tensor mismatch: " + ref[i].first);
        }
        double mse_a_b = 0.0;
        double mse_a_0 = 0.0;
        for (size_t j = 0; j < ref[i].second.size(); j++) {
            const double a = ref[i].second[j];
            const double b = cur[i].second[j];
            // masked rows may hold -inf in both
            if (!std::isfinite(a) && a == b) {
                continue;
            }
            mse_a_b += (a - b)*(a - b);
            mse_a_0 += a*a;
        }
        const double val = mse_a_b == 0.0 ? 0.0 : mse_a_0 == 0.0 ? INFINITY : mse_a_b/mse_a_0;
        res.worst = std::max(res.worst, std::isnan(val) ? INFINITY : val);
        if (!(val <= thold)) {
            if (res.first_bad.empty()) {
                res.first_bad = ref[i].first;
            }
            char buf[64];
            snprintf(buf, sizeof(buf), " %s=%.1e", ref[i].first.c_str(), val);
            res.bad += buf;
        }
    }
    return res;
}

// decodes the prefix at pos_prefix, applies each step of moves with its own K-shift and returns the logits of the probe at pos_probe
static std::vector<float> get_logits_kv_shift(
        llama_model * model, llama_context * lctx, kq_capture & cap, const std::vector<llama_token> & prefix, const std::vector<llama_token> & probe,
        llama_pos pos_prefix, const std::vector<std::vector<kv_shift_move>> & steps, llama_pos pos_probe) {
    const uint32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    llama_memory_t mem = llama_get_memory(lctx);

    cap.data.clear();
    llama_memory_clear(mem, true);
    decode_range(lctx, prefix.data(), prefix.size(), pos_prefix, 0, false);

    for (size_t i = 0; i < steps.size(); i++) {
        for (const kv_shift_move & mv : steps[i]) {
            llama_memory_seq_add(mem, 0, mv.p0, mv.p1, mv.delta);
        }
        // there is no public API to apply a pending shift, a decode of a throwaway token in seq 1 does it
        if (i + 1 < steps.size()) {
            decode_range(lctx, probe.data(), 1, 0, 1, false);
            llama_memory_seq_rm(mem, 1, -1, -1);
        }
    }

    cap.active = true;
    decode_range(lctx, probe.data(), probe.size(), pos_probe, 0, true);
    cap.active = false;

    std::vector<float> ret;
    ret.reserve(probe.size()*n_vocab);
    for (uint32_t i = 0; i < probe.size(); i++) {
        const float * logits_ith = llama_get_logits_ith(lctx, i);
        ret.insert(ret.end(), logits_ith, logits_ith + n_vocab);
    }
    return ret;
}

// a K-shift is correct if decoding at [d, d+n) and shifting by -d gives the same probe logits as decoding at [0, n)
static int test_kv_shift(const std::string & arch_filter, const size_t seed, const float stdev, const int verbosity) {
    struct user_data_t {
        struct {
            ggml_log_callback callback;
            void * user_data;
        } log_old;

        int verbosity;

        user_data_t(int verbosity) : verbosity(verbosity) {
            llama_log_get(&log_old.callback, &log_old.user_data);
        }
    };
    user_data_t ud(verbosity);

    llama_log_set([](ggml_log_level level, const char * text, void * user_data) {
        const user_data_t * ud = (const user_data_t *) user_data;
        int verbosity = common_log_get_verbosity(level);
        if (verbosity <= ud->verbosity) {
            ud->log_old.callback(level, text, ud->log_old.user_data);
        }
    }, &ud);

    // n_prefix + n_probe stays below the fixture SWA window (n_ctx/8 = 32)
    const uint32_t  n_prefix = 20;
    const uint32_t  n_probe  = 8;
    const llama_pos d        = 64;
    const llama_pos gap      = 4;

    const std::vector<llama_token> tokens = get_tokens(n_prefix + n_probe, 128, seed);
    const std::vector<llama_token> prefix(tokens.begin(), tokens.begin() + n_prefix);
    const std::vector<llama_token> probe (tokens.begin() + n_prefix, tokens.end());

    const std::vector<std::vector<kv_shift_move>> steps_shift  = { { { d, d + (llama_pos) n_prefix, -d } } };
    const std::vector<std::vector<kv_shift_move>> steps_shift2 = { { { d, d + (llama_pos) n_prefix, -d/2 } }, { { d/2, d/2 + (llama_pos) n_prefix, -d/2 } } };
    // control: the second half of the prefix ends up gap positions too far, which tells if the probe is sensitive to K positions at all
    const std::vector<std::vector<kv_shift_move>> steps_ctrl   = { {
        { d,                           d + (llama_pos) n_prefix/2, -d       },
        { d + (llama_pos) n_prefix/2,  d + (llama_pos) n_prefix,   -d + gap },
    } };

    size_t max_arch_name_length = 0;
    for (const llm_arch & arch : llama_arch_all()) {
        max_arch_name_length = std::max(max_arch_name_length, strlen(llama_arch_name(arch)));
    }
    const std::string template_row = "|%" + std::to_string(max_arch_name_length) + "s|%6s|%10s|%10s|%10s|%10s|%s\n";

    size_t n_tests  = 0;
    size_t n_failed = 0;
    common_log_flush(common_log_main());
    LOG(template_row.c_str(), "Model arch.", "Config", "Shift kq", "Shift2 kq", "Logits", "Control kq", "Status");

    for (const llm_arch & arch : llama_arch_all()) {
        if (arch == LLM_ARCH_UNKNOWN || !arch_matches(arch_filter, arch) || !arch_supported(arch)) {
            continue;
        }
        const glm5_upstream_scope glm5_upstream(arch);
        if (arch == LLM_ARCH_EAGLE3 || arch == LLM_ARCH_DFLASH || arch == LLM_ARCH_GEMMA4_ASSISTANT) {
            continue;
        }
        const bool encode = arch == LLM_ARCH_T5 || arch == LLM_ARCH_DREAM || arch == LLM_ARCH_LLADA || arch == LLM_ARCH_LLADA_MOE || arch == LLM_ARCH_RND1;
        if (encode) {
            continue;
        }
        for (bool moe : {false, true}) {
            if (moe && !moe_implemented(arch)) {
                continue;
            }
            if (!moe && moe_mandatory(arch)) {
                continue;
            }
            const char * config_name = moe ? "MoE" : "Dense";

            // the attention scale depends on the absolute position, so the hidden states of the prefix do too
            const double thold = arch == LLM_ARCH_MISTRAL_LARGE4 ? 1e-4 : 1e-6;

            // at least 4 layers, so that "every 4th layer" patterns (NoPE, full attention) are covered
            // exaone4 is SWA only with 64 layers, smallthinker always ropes with 4
            const uint32_t n_layer_min = arch == LLM_ARCH_EXAONE4 ? 64 : arch == LLM_ARCH_SMALLTHINKER ? 8 : 4;
            gguf_context_ptr gguf_ctx = get_gguf_ctx(arch, moe, false, n_layer_min);
            // different bases, so that a shift using the wrong one is caught
            gguf_set_val_f32(gguf_ctx.get(), llama_arch_kv_name(arch, LLM_KV_ROPE_FREQ_BASE).c_str(),     20000.0f);
            gguf_set_val_f32(gguf_ctx.get(), llama_arch_kv_name(arch, LLM_KV_ROPE_FREQ_BASE_SWA).c_str(), 10000.0f);
            if (arch == LLM_ARCH_BAILINGMOE3) {
                GGML_ASSERT(gguf_remove_key(gguf_ctx.get(), "bailingmoe3.kda.safe_gate") >= 0);
            }

            llama_model_ptr model;
            try {
                model = get_model(gguf_ctx.get(), nullptr, seed, stdev, {});
            } catch (const std::exception & err) {
                LOG(template_row.c_str(), llama_arch_name(arch), config_name, "", "", "", "", "\033[1;31mERROR\033[0m (model load)");
                n_tests++;
                n_failed++;
                continue;
            }
            // absolute position embeddings are not translation invariant, the comparison does not hold
            if (llama_model_rope_type(model.get()) == LLAMA_ROPE_TYPE_NONE) {
                LOG(template_row.c_str(), llama_arch_name(arch), config_name, "", "", "", "", "\033[1;33mSKIP\033[0m (no rope)");
                continue;
            }

            LOG("|%*s|%6s|", (int) max_arch_name_length, llama_arch_name(arch), config_name);
            fflush(stdout);

            llama_context_params ctx_params = llama_context_default_params();
            ctx_params.n_ctx           = 0;
            ctx_params.n_threads       = 4;
            ctx_params.n_threads_batch = 4;
            ctx_params.n_seq_max       = 2;
            ctx_params.kv_unified      = true;
            ctx_params.type_k          = GGML_TYPE_F32;
            ctx_params.type_v          = GGML_TYPE_F32;
            ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
            // differs from the SWA default, so that a shift using the wrong scale is caught
            ctx_params.rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_LINEAR;
            ctx_params.rope_freq_scale   = 0.5f;
            kq_capture cap;
            ctx_params.cb_eval           = kq_capture_cb;
            ctx_params.cb_eval_user_data = &cap;

            ud.verbosity = LOG_LEVEL_OUTPUT;
            llama_context_ptr lctx(llama_init_from_model(model.get(), ctx_params));
            ud.verbosity = verbosity;
            if (!lctx) {
                LOG("%10s|%10s|%10s|%10s|\033[1;33mSKIP\033[0m (context)\n", "", "", "", "");
                continue;
            }
            llama_memory_t mem = llama_get_memory(lctx.get());
            if (mem == nullptr || !llama_memory_can_shift(mem)) {
                LOG("%10s|%10s|%10s|%10s|\033[1;33mSKIP\033[0m (no shift)\n", "", "", "", "");
                continue;
            }

            std::string status;
            char shift_str[16]  = "";
            char shift2_str[16] = "";
            char logits_str[16] = "";
            char ctrl_str[16]   = "";
            n_tests++;
            try {
                const std::vector<float> logits_ref = get_logits_kv_shift(model.get(), lctx.get(), cap, prefix, probe, 0, {}, n_prefix);
                const auto kq_ref = cap.data;
                if (verbosity >= LOG_LEVEL_INFO) {
                    for (const auto & e : kq_ref) {
                        LOG("[%s: %zu] ", e.first.c_str(), e.second.size());
                    }
                }
                const std::vector<float> logits_shift = get_logits_kv_shift(model.get(), lctx.get(), cap, prefix, probe, d, steps_shift, n_prefix);
                const kq_diff diff_shift = compare_kq(kq_ref, cap.data, thold);
                get_logits_kv_shift(model.get(), lctx.get(), cap, prefix, probe, d, steps_shift2, n_prefix);
                const kq_diff diff_shift2 = compare_kq(kq_ref, cap.data, thold);
                const std::vector<float> logits_ctrl = get_logits_kv_shift(model.get(), lctx.get(), cap, prefix, probe, d, steps_ctrl, n_prefix + gap);
                const kq_diff diff_ctrl = compare_kq(kq_ref, cap.data, thold);

                const double nmse_logits = nmse(logits_ref, logits_shift);
                snprintf(shift_str,  sizeof(shift_str),  "%.2e", diff_shift.worst);
                snprintf(shift2_str, sizeof(shift2_str), "%.2e", diff_shift2.worst);
                snprintf(logits_str, sizeof(logits_str), "%.2e", nmse_logits);
                snprintf(ctrl_str,   sizeof(ctrl_str),   "%.2e", diff_ctrl.worst);

                if (kq_ref.empty()) {
                    status = "\033[1;33mSKIP\033[0m (no kq captured)";
                } else if (!diff_shift.first_bad.empty() || !diff_shift2.first_bad.empty()) {
                    n_failed++;
                    status = "\033[1;31mFAIL\033[0m" + (diff_shift.bad.empty() ? " x2:" + diff_shift2.bad : diff_shift.bad);
                } else if (!(diff_ctrl.worst > 10*thold)) {
                    status = "\033[1;33mOK\033[0m (low sensitivity)";
                } else {
                    status = "\033[1;32mOK\033[0m";
                }
            } catch (const std::exception & err) {
                n_failed++;
                status = std::string("\033[1;31mERROR\033[0m (") + err.what() + ")";
            }
            LOG("%10s|%10s|%10s|%10s|%s\n", shift_str, shift2_str, logits_str, ctrl_str, status.c_str());
        }
    }

    if (n_tests == 0) {
        LOG("Summary: no tests executed\n");
    } else if (n_failed == 0) {
        LOG("Summary: all %zu test(s) passed\n", n_tests);
    } else {
        LOG("Summary: %zu test(s) executed, %zu failed\n", n_tests, n_failed);
    }

    llama_log_set(ud.log_old.callback, ud.log_old.user_data);
    return n_failed == 0 ? 0 : 1;
}

int main(int argc, char ** argv) {
    if (argc == 2 && strcmp(argv[1], "--test-state-comparison") == 0) {
        const int comparison = test_glm_state_comparison();
        return test_glm_capture_failures() || comparison;
    }
    // init the logger at max verbosity. filter with a custom callback respecting the user-configure verbosity
    common_log_set_verbosity_thold(LOG_LEVEL_DEBUG);
    common_init();
    llama_backend_init();

    std::random_device rd;

    std::string arch_filter;
    llm_arch arch = LLM_ARCH_UNKNOWN; // the exact architecture -a names, for the GLM checks that run on --arch glm5next
    size_t seed = rd();
    // Upstream v1.0.0's default is 0.1. The fork's checks carry absolute bounds measured at N(0, 0.01) (a rollback's
    // logits within 1e-7, the data split's KL within 1e-6, a prompt shape's logits within 1e-4), and ten times the spread
    // puts the same relative float error past them, so the default stays the fork's. tests/CMakeLists.txt passes
    // upstream's value to upstream's own fixtures
    float stdev = 0.01f;
    std::string out;
    std::string diagnostic_dir, diagnostic_device;
    std::string dp_device, prefill_device, shared_device;
    std::string tensor_device; // --glm-tensor-device CUDA0,CUDA1,CUDA0,CUDA1: one rank per name, repeats allowed
    bool glm_alias_only = false;
    bool glm_kpool_only = false;
    bool glm_nextn_only = false;
    const char * target_backend = nullptr;
    bool kv_shift = false;

    int verbosity = LOG_LEVEL_ERROR;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--glm-nextn-store-only") == 0) {
            glm_nextn_only = true;
            continue;
        }
        if (strcmp(argv[i], "--glm-kpool-only") == 0) {
            glm_kpool_only = true;
            continue;
        }
        if (strcmp(argv[i], "--glm-arch-alias") == 0) {
            glm_alias_only = true;
            continue;
        }
        if (strcmp(argv[i], "--glm-dp-device") == 0 || strcmp(argv[i], "--glm-prefill-reuse-device") == 0) {
            const bool prefill = strcmp(argv[i], "--glm-prefill-reuse-device") == 0;
            if (i + 1 == argc) return 1;
            (prefill ? prefill_device : dp_device) = argv[++i];
            continue;
        }
        if (strcmp(argv[i], "--glm-tensor-device") == 0) {
            if (i + 1 == argc) return 1;
            tensor_device = argv[++i];
            continue;
        }
        if (strcmp(argv[i], "--glm-shared-buffers-device") == 0) { // the tensor split's ranks, as --glm-tensor-device
            if (i + 1 == argc) return 1;
            shared_device = argv[++i];
            continue;
        }
        if (strcmp(argv[i], "--glm-prefix-diagnostics") == 0 || strcmp(argv[i], "--glm-prefix-device") == 0) {
            const bool directory = strcmp(argv[i], "--glm-prefix-diagnostics") == 0;
            if (i+1 == argc) return 1;
            (directory ? diagnostic_dir : diagnostic_device) = argv[++i];
            continue;
        }
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            usage(argv);
            return 0;
        } else if (strcmp(argv[i], "-a") == 0 || strcmp(argv[i], "--arch") == 0) {
            if (i + 1 < argc) {
                const std::string arch_name = argv[++i];
                if (llama_arch_from_string(arch_name) != LLM_ARCH_UNKNOWN) {
                    // exact architecture name
                    arch_filter = "^" + arch_name + "$";
                    arch = llama_arch_from_string(arch_name);
                } else {
                    try {
                        std::regex re(arch_name);
                        arch_filter = arch_name;
                    } catch (const std::regex_error & err) {
                        LOG_ERR("%s: invalid architecture regex: %s (%s)\n", __func__, arch_name.c_str(), err.what());
                        return 1;
                    }
                }
            } else {
                usage(argv);
                return 1;
            }
        } else if (strcmp(argv[i], "-s") == 0 || strcmp(argv[i], "--seed") == 0) {
            if (i + 1 < argc) {
                seed = std::stoull(argv[++i]);
            } else {
                usage(argv);
                return 1;
            }
        } else if (strcmp(argv[i], "-d") == 0 || strcmp(argv[i], "--stdev") == 0) {
            if (i + 1 < argc) {
                stdev = std::stof(argv[++i]);
            } else {
                usage(argv);
                return 1;
            }
        } else if (strcmp(argv[i], "-v") == 0) {
            if (i + 1 < argc) {
                verbosity = std::stoull(argv[++i]);
            } else {
                usage(argv);
                return 1;
            }
        } else if (strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--out") == 0) {
            if (i + 1 < argc) {
                out = argv[++i];
            } else {
                usage(argv);
                return 1;
            }
        } else if (strcmp(argv[i], "-b") == 0 || strcmp(argv[i], "--backend") == 0) {
            if (i + 1 < argc) {
                const char * backend_name = argv[++i];
                ggml_backend_dev_t dev = ggml_backend_dev_by_name(backend_name);
                if (dev == nullptr) {
                    LOG_ERR("%s: unknown backend device: %s\n", __func__, backend_name);
                    return 1;
                }
                target_backend = ggml_backend_dev_name(dev);
            } else {
                usage(argv);
                return 1;
            }
        } else if (strcmp(argv[i], "--kv-shift") == 0) {
            kv_shift = true;
        } else {
            LOG_ERR("%s: unknown argument: %s\n", __func__, argv[i]);
            usage(argv);
            return 1;
        }
    }
    if (stdev <= 0.0f) {
        LOG_ERR("%s: stdev must be > 0\n", __func__);
        return 1;
    }
    LOG_INF("%s: using seed %zu, stdev %f\n", __func__, seed, stdev);

    init_normal_pool(seed, stdev);

    try {
        if (glm_nextn_only) {
            ggml_backend_load_all();
            const auto device = ggml_backend_dev_by_name(dp_device.empty() ? "CPU" : dp_device.c_str());
            if (!device) { throw std::runtime_error("NextN store device is unavailable"); }
            for (bool q8 : {false, true}) {
                for (size_t count : {1u, 2u, 4u}) {
                    const std::vector<ggml_backend_dev_t> ranks(count, device);
                    if (!test_glm_nextn_store(seed, ranks, count == 1 ? LLAMA_SPLIT_MODE_LAYER : LLAMA_SPLIT_MODE_DATA, {}, q8)) { return 1; }
                }
            }
            return 0;
        }
        if (glm_kpool_only) {
            ggml_backend_load_all();
            return test_glm_mtp_index(seed, {}, LLAMA_SPLIT_MODE_LAYER, {}) &&
                   test_glm_mtp_index_sequences(seed, {}, LLAMA_SPLIT_MODE_LAYER, {}) ? 0 : 1;
        }
        if (glm_alias_only || (arch == LLM_ARCH_GLM5NEXT && out.empty())) {
            ggml_backend_load_all();
            if (!test_glm_arch_alias(seed)) { return 1; }
            if (glm_alias_only) { return 0; }
        }
        if (!tensor_device.empty()) {
            ggml_backend_load_all();
            std::vector<ggml_backend_dev_t> ranks;
            for (const auto & name : string_split<std::string>(tensor_device, ',')) {
                const auto device = ggml_backend_dev_by_name(name.c_str());
                if (!device) throw std::runtime_error("GLM tensor device is unavailable: " + name);
                ranks.push_back(device);
            }
            const int plain = test_glm_tensor(seed, ranks, false);
            const int quantized = test_glm_tensor(seed, ranks, true);
            return plain || quantized;
        }
        if (!shared_device.empty()) {
            ggml_backend_load_all();
            std::vector<ggml_backend_dev_t> ranks;
            for (const auto & name : string_split<std::string>(shared_device, ',')) {
                const auto device = ggml_backend_dev_by_name(name.c_str());
                if (!device) throw std::runtime_error("GLM shared buffers device is unavailable: " + name);
                ranks.push_back(device);
            }
            return test_glm_shared_buffers(seed, ranks, ranks.size() == 1 ? LLAMA_SPLIT_MODE_LAYER : LLAMA_SPLIT_MODE_TENSOR, {}) ? 0 : 1;
        }
        if (!dp_device.empty() || !prefill_device.empty()) {
            ggml_backend_load_all();
            std::vector<ggml_backend_dev_t> devices;
            for (const auto & name : string_split<std::string>(prefill_device.empty() ? dp_device : prefill_device, ',')) {
                const auto device = ggml_backend_dev_by_name(name.c_str());
                if (!device) throw std::runtime_error("GLM DP device is unavailable: " + name);
                devices.push_back(device);
            }
            if (!prefill_device.empty()) {
                return test_glm_prefill_reuse(seed, devices, LLAMA_SPLIT_MODE_TENSOR, {}) ? 0 : 1;
            }
            const int plain = test_glm_dp(seed, devices);
            const int quantized = test_glm_dp(seed, devices, true);
            return plain || quantized;
        }
        if (!diagnostic_dir.empty() || !diagnostic_device.empty()) {
            if (arch != LLM_ARCH_GLM5NEXT || diagnostic_dir.empty() || diagnostic_device.empty() || !out.empty()) {
                throw std::runtime_error("GLM prefix diagnostics require --arch glm5next, a fresh directory and an explicit device");
            }
            ggml_backend_load_all();
            const auto device = ggml_backend_dev_by_name(diagnostic_device.c_str());
            if (!device) throw std::runtime_error("GLM diagnostic device is unavailable: " + diagnostic_device);
            if (!std::filesystem::create_directory(diagnostic_dir)) throw std::runtime_error("GLM diagnostic directory already exists");
            return test_glm_generated_prefixes(seed, {device}, LLAMA_SPLIT_MODE_LAYER, {}, diagnostic_dir) ? 0 : 1;
        }
        if (!out.empty()) {
            return save_models(arch_filter, seed, stdev, verbosity, out);
        }
        if (kv_shift) {
            return test_kv_shift(arch_filter, seed, stdev, verbosity);
        }
        return test_backends(arch_filter, seed, stdev, verbosity, target_backend);
    } catch (const std::exception & err) {
        fprintf(stderr, "encountered runtime error: %s\n", err.what());
        return -1;
    }
}
