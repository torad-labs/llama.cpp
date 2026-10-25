#include "common.h"
#include "log.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"
#include "ggml-cpp.h"
#include "llama.h"
#include "llama-cpp.h"

// TODO: replace with #include "llama-ext.h" in the future
#include "../src/llama-arch.h"
#include "../src/llama-ext.h"
#include "../src/llama-model-saver.h"
#include "../src/llama-memory-hybrid.h"

#include <algorithm>
#include <cmath>
#include <cinttypes>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <set>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

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

static void set_tensor_data(struct ggml_tensor * tensor, void * userdata) {
    size_t seed = *(const size_t *) userdata;
    std::hash<std::string> hasher;
    seed ^= hasher(tensor->name);
    std::mt19937 gen(seed);
    std::normal_distribution<float> dis(0.0f, 1.0e-2f);

    const int64_t ne = ggml_nelements(tensor);
    if (tensor->type == GGML_TYPE_F32) {
        std::vector<float> tmp(ne);
        for (int64_t i = 0; i < ne; i++) {
            tmp[i] = dis(gen);
        }
        ggml_backend_tensor_set(tensor, tmp.data(), 0, ggml_nbytes(tensor));
    } else if (tensor->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> tmp(ne);
        for (int64_t i = 0; i < ne; i++) {
            tmp[i] = ggml_fp32_to_fp16(dis(gen));
        }
        ggml_backend_tensor_set(tensor, tmp.data(), 0, ggml_nbytes(tensor));
    } else {
        GGML_ABORT("fatal error");
    }
}

static void usage(char ** argv) {
    printf("Usage: %s [-a/--arch arch] [-s/--seed seed] [-v/--verbose]\n", argv[0]);
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

static gguf_context_ptr get_gguf_ctx(const llm_arch arch, const bool moe, const bool mtp = false) {
    gguf_context_ptr ret(gguf_init_empty());
    llama_model_saver ms(arch, ret.get());
    const uint32_t n_ctx = 128;

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
    } else if (arch == LLM_ARCH_GEMMA4) {
        n_embd = 128;
        n_head = 2;
        n_ff   = 192;
        n_layer = 5; // need at least 5 for swa_pattern (every 5th is full_attention)
    } else if (arch == LLM_ARCH_GEMMA3N) {
        n_embd = 64;
        n_head = 1;
        n_ff   = 96;
        n_layer = 22; // hparams.n_layer_kv_from_start = 20 is hardcoded
    } else if (arch == LLM_ARCH_DEEPSEEK4) {
        // head size 64 so that GPU flash attention kernels support the model
        n_embd  = 512;
        n_head  = 8;
        n_ff    = 1024;
        n_layer = 4;
    } else if (arch == LLM_ARCH_STEP35 || arch == LLM_ARCH_LAGUNA) {
        n_embd = 160; // exercise per-head tensor split granularity with head size 80
    } else if (arch == LLM_ARCH_QWEN3 || arch == LLM_ARCH_MUSE_GLIMMER || arch == LLM_ARCH_AFMOE) {
        n_head = 4;
    } else if (arch == LLM_ARCH_DEEPSEEK2
            || arch == LLM_ARCH_DEEPSEEK32
            || arch == LLM_ARCH_GLM_DSA
            || arch == LLM_ARCH_DOTS3NOTE
            || arch == LLM_ARCH_KIMI_LINEAR
            || arch == LLM_ARCH_BAILINGMOE3
            || arch == LLM_ARCH_KIMI_K3
            || arch == LLM_ARCH_MISTRAL4) {
        n_embd = 128;
        n_head = 1;
        n_ff   = 192;
    } else if (arch == LLM_ARCH_GLM5NEXT) {
        n_embd = 128;
        n_head = 4; // split two ways, each device gets 2 KDA and 2 DSA heads (2 DSA heads are the fewest with 128-row q_b blocks)
        n_ff   = 192;
        n_layer = mtp ? 3 : 2;
    } else if (arch == LLM_ARCH_NEMOTRON_H || arch == LLM_ARCH_NEMOTRON_H_MOE) {
        n_layer = 3;
    } else if (arch == LLM_ARCH_CHAMELEON) {
        n_vocab = 10240;
    } else if (arch == LLM_ARCH_QWEN3TTS) {
        n_vocab = 4096; // must be >= the hard-coded codec head size (3072)
    }

    uint32_t n_head_kv = n_head;
    if (arch == LLM_ARCH_QWEN3) {
        n_head_kv = 1; // MQA coverage
    } else if (arch == LLM_ARCH_MUSE_GLIMMER || arch == LLM_ARCH_AFMOE) {
        n_head_kv = 2; // GQA coverage
    }
    const uint32_t n_embd_head = n_embd / n_head;

    ms.add_kv(LLM_KV_GENERAL_ARCHITECTURE,      llm_arch_name(arch));
    ms.add_kv(LLM_KV_VOCAB_SIZE,                n_vocab);
    ms.add_kv(LLM_KV_CONTEXT_LENGTH,            n_ctx);
    ms.add_kv(LLM_KV_EMBEDDING_LENGTH,          n_embd);
    ms.add_kv(LLM_KV_FEATURES_LENGTH,           n_embd);
    ms.add_kv(LLM_KV_BLOCK_COUNT,               n_layer);
    if (mtp) {
        ms.add_kv(LLM_KV_NEXTN_PREDICT_LAYERS, uint32_t(1));
    }
    ms.add_kv(LLM_KV_LEADING_DENSE_BLOCK_COUNT, uint32_t(1));

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
            arch == LLM_ARCH_GRANITE_HYBRID || arch == LLM_ARCH_LFM2 || arch == LLM_ARCH_LFM2MOE || arch == LLM_ARCH_KIMI_LINEAR ||
            arch == LLM_ARCH_BAILINGMOE3 || arch == LLM_ARCH_KIMI_K3) {
        GGML_ASSERT(n_layer >= 2);
        std::vector<uint32_t> n_head_per_layer;
        n_head_per_layer.reserve(n_layer);
        for (uint32_t il = 0; il < n_layer; il++) {
            n_head_per_layer.push_back(il == 1 ? 0 : n_head);
        }
        ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT, n_head_per_layer);
        ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV, n_head_per_layer);
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
        ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV, arch == LLM_ARCH_DEEPSEEK4 ? uint32_t(1) : n_head_kv);
    }

    ms.add_kv(LLM_KV_ATTENTION_MAX_ALIBI_BIAS, 8.0f);
    if (arch == LLM_ARCH_DEEPSEEK4) {
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
            || arch == LLM_ARCH_MISTRAL4) {
        ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH,       uint32_t(576));
        ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH,     uint32_t(512));
        ms.add_kv(LLM_KV_ROPE_DIMENSION_COUNT,       uint32_t(64));
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
    ms.add_kv(LLM_KV_ATTENTION_CLAMP_KQV,              1.0f);
    ms.add_kv(LLM_KV_ATTENTION_LAYERNORM_EPS,          arch == LLM_ARCH_GLM5NEXT ? 1e-6f : 1e-5f);
    ms.add_kv(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,      1e-5f);
    ms.add_kv(LLM_KV_ATTENTION_GROUPNORM_EPS,          1e-5f);
    ms.add_kv(LLM_KV_ATTENTION_GROUPNORM_GROUPS,       uint32_t(8));
    ms.add_kv(LLM_KV_ATTENTION_Q_LORA_RANK,            arch == LLM_ARCH_DEEPSEEK4 ? uint32_t(64) : uint32_t(512));
    ms.add_kv(LLM_KV_ATTENTION_KV_LORA_RANK,           uint32_t(512));
    ms.add_kv(LLM_KV_ATTENTION_RELATIVE_BUCKETS_COUNT, uint32_t(8));
    ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW,         n_ctx/8);

    if (arch == LLM_ARCH_GEMMA4) {
        ms.add_kv(LLM_KV_EMBEDDING_LENGTH_PER_LAYER,      n_embd/2);
        ms.add_kv(LLM_KV_ATTENTION_SHARED_KV_LAYERS,      uint32_t(0));
        ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH_SWA,        n_embd_head);
        ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH_SWA,      n_embd_head);
        ms.add_kv(LLM_KV_ROPE_FREQ_BASE_SWA,              10000.0f);
        // SWA pattern: every 5th layer is full attention (matches E2B layer_types)
        ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, uint32_t(5));
    } else if (arch == LLM_ARCH_COHERE2MOE || arch == LLM_ARCH_MIMO2 || arch == LLM_ARCH_STEP35 ||
            arch == LLM_ARCH_MUSE_GLIMMER || arch == LLM_ARCH_GRANITE_SWA || arch == LLM_ARCH_DOTS3NOTE) {
        std::vector<uint32_t> pattern;
        pattern.reserve(n_layer);
        for (uint32_t il = 0; il < n_layer; il++) {
            pattern.push_back(il % 2);
        }
        ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, pattern);
    } else {
        ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, uint32_t(2));
    }

    // MSA requires one indexer head per GQA (KV) head, unlike the DSA archs where the
    // indexer head count is independent of the main attention head count.
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT,   arch == LLM_ARCH_MINIMAX_M3 || arch == LLM_ARCH_DEEPSEEK4 ? n_head : uint32_t(1));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH,   uint32_t(64));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_TOP_K,        uint32_t(8));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_BLOCK_SIZE,   uint32_t(4));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_LOCAL_BLOCKS, uint32_t(1));
    ms.add_kv(LLM_KV_ROPE_DIMENSION_SECTIONS, std::vector<uint32_t>({n_embd_head/4, n_embd_head/4, n_embd_head/4, n_embd_head/4}));

    if (arch == LLM_ARCH_DEEPSEEK4) {
        ms.add_kv(LLM_KV_ATTENTION_OUTPUT_GROUP_COUNT,         uint32_t(8));
        ms.add_kv(LLM_KV_ATTENTION_OUTPUT_LORA_RANK,           uint32_t(32));
        ms.add_kv(LLM_KV_ATTENTION_COMPRESS_RATIOS,            std::vector<uint32_t>({0, 0, 4, 128}));
        ms.add_kv(LLM_KV_ATTENTION_COMPRESS_ROPE_FREQ_BASE,    160000.0f);
        ms.add_kv(LLM_KV_HYPER_CONNECTION_COUNT,               uint32_t(4));
        ms.add_kv(LLM_KV_HYPER_CONNECTION_SINKHORN_ITERATIONS, uint32_t(2));
        ms.add_kv(LLM_KV_HYPER_CONNECTION_EPSILON,             1.0e-6f);
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
    ms.add_kv(LLM_KV_TOKENIZER_MODEL,         "no_vocab");
    // ms.add_kv(LLM_KV_DENSE_2_FEAT_OUT,     n_embd);
    // ms.add_kv(LLM_KV_DENSE_3_FEAT_IN,      n_embd);

    if (moe) {
        ms.add_kv(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, n_ff);
        ms.add_kv(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, n_ff / 2);  // distinct from n_ff so a saver key-clobber surfaces on reload
        ms.add_kv(LLM_KV_EXPERT_LATENT_LENGTH,       n_ff);
        ms.add_kv(LLM_KV_INTERLEAVE_MOE_LAYER_STEP,  uint32_t(2));
        ms.add_kv(LLM_KV_EXPERT_COUNT,               uint32_t(2));
        ms.add_kv(LLM_KV_EXPERT_USED_COUNT,          uint32_t(1));
        ms.add_kv(LLM_KV_EXPERT_SHARED_COUNT,        uint32_t(1));
        ms.add_kv(LLM_KV_EXPERT_GATING_FUNC,         arch == LLM_ARCH_DEEPSEEK4 ? uint32_t(4) : uint32_t(2)); // sqrtsoftplus : sigmoid
        ms.add_kv(LLM_KV_EXPERT_GROUP_SCALE,         1.0f);
        ms.add_kv(LLM_KV_EXPERTS_PER_GROUP,          uint32_t(1));
    }

    ms.add_kv(LLM_KV_POSNET_EMBEDDING_LENGTH,   n_embd);
    ms.add_kv(LLM_KV_POSNET_BLOCK_COUNT,        n_layer);
    ms.add_kv(LLM_KV_CONVNEXT_EMBEDDING_LENGTH, n_embd);
    ms.add_kv(LLM_KV_CONVNEXT_BLOCK_COUNT,      n_layer);
    ms.add_kv(LLM_KV_XIELU_ALPHA_N,             1.0f);
    ms.add_kv(LLM_KV_XIELU_ALPHA_P,             1.0f);
    ms.add_kv(LLM_KV_XIELU_BETA,                1.0f);
    ms.add_kv(LLM_KV_XIELU_EPS,                 1.0e-7f);
    ms.add_kv(LLM_KV_SSM_INNER_SIZE,            arch == LLM_ARCH_QWEN3NEXT || arch == LLM_ARCH_QWEN35 || arch == LLM_ARCH_QWEN35MOE ? 256 : 2*n_embd);
    ms.add_kv(LLM_KV_SSM_CONV_KERNEL,           uint32_t(4));
    ms.add_kv(LLM_KV_SSM_STATE_SIZE,            uint32_t(128));
    ms.add_kv(LLM_KV_SSM_TIME_STEP_RANK,        n_head);
    ms.add_kv(LLM_KV_SSM_GROUP_COUNT,           arch == LLM_ARCH_PLAMO2 ? 0 : uint32_t(2));
    ms.add_kv(LLM_KV_KDA_HEAD_DIM,              uint32_t(128));
    ms.add_kv(LLM_KV_KDA_SAFE_GATE,              true);
    ms.add_kv(LLM_KV_KDA_GATE_LOWER_BOUND,       -5.0f);
    if (arch == LLM_ARCH_BAILINGMOE3) {
        ms.add_kv(LLM_KV_SWIGLU_CLAMP_EXP,   std::vector<float>({0.0f, 4.0f}));
        ms.add_kv(LLM_KV_SWIGLU_CLAMP_SHEXP, std::vector<float>({0.0f, 5.0f}));
    }
    ms.add_kv(LLM_KV_WKV_HEAD_SIZE,             n_embd/n_head);
    ms.add_kv(LLM_KV_SHORTCONV_L_CACHE,         uint32_t(3));
    ms.add_kv(LLM_KV_RESIDUAL_SCALE,            3.5565588200778455f);
    ms.add_kv(LLM_KV_ATTN_RES_BLOCK_SIZE,       uint32_t(12));
    ms.add_kv(LLM_KV_ACTIVATION_SITU_BETA,      4.0f);
    ms.add_kv(LLM_KV_ACTIVATION_SITU_LINEAR_BETA, 25.0f);
    ms.add_kv(LLM_KV_KDA_GATE_LOWER_BOUND,      -5.0f);

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

static std::pair<llama_model_ptr, llama_context_ptr> get_model_and_ctx(
        struct gguf_context * gguf_ctx, FILE * file, const size_t seed, const std::vector<ggml_backend_dev_t> & devs,
        const llama_split_mode split_mode = LLAMA_SPLIT_MODE_LAYER, bool encode = false, const std::vector<float> & tensor_split = {}, bool mtp = false) {
    GGML_ASSERT((gguf_ctx == nullptr) != (file == nullptr));
    llama_model_params model_params = llama_model_default_params();
    model_params.progress_callback = silent_model_load_progress;
    model_params.load_mtp = mtp;
    std::vector<ggml_backend_dev_t> devs_copy = devs;
    devs_copy.push_back(nullptr);
    model_params.devices = devs_copy.data();
    model_params.split_mode = split_mode;
    model_params.tensor_split = tensor_split.empty() ? nullptr : tensor_split.data();

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 0;
    ctx_params.n_threads = 4;
    ctx_params.n_threads_batch = 4;
    if (!encode) {
        ctx_params.n_ubatch = 64;
    }

    size_t tmp = seed;
    llama_model_ptr model(gguf_ctx != nullptr ?
        llama_model_init_from_user(gguf_ctx, set_tensor_data, &tmp, model_params) :
        llama_model_load_from_file_ptr(file, model_params));
    if (!model) {
        throw std::runtime_error("failed to create llama model");
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
    llama_batch batch = llama_batch_init(n_ctx, 0, 1);
    GGML_ASSERT(n_tokens <= n_ctx);
    for (uint32_t pos = 0; pos < n_tokens; pos++) {
        common_batch_add(batch, tokens[pos], pos, {0}, true);
    }
    batch.n_tokens = n_tokens;
    if (encode) {
        if (llama_encode(lctx, batch)) {
            llama_batch_free(batch);
            throw std::runtime_error("failed to encode batch");
        }
    }
    if (llama_decode(lctx, batch)) {
        llama_batch_free(batch);
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
    llama_batch_free(batch);
    return ret;
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
        common_batch_add(batch, tokens[i], start + i/n_seq + offset, {seqs.empty() ? 0 : seqs[i%n_seq]}, true);
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
                LOG_ERR("%s draft vocabulary accepted invalid IDs\n", llm_arch_name(arch));
                return false;
            }
        }
        std::vector<llama_token> ids(n_vocab);
        std::iota(ids.begin(), ids.end(), 0);
        std::shuffle(ids.begin(), ids.end(), rng);
        ids.resize(n_hot);
        std::sort(ids.begin(), ids.end());
        if (!llama_model_set_draft_vocab(model, ids.data(), ids.size())) {
            LOG_ERR("%s draft vocabulary rejected %d valid IDs\n", llm_arch_name(arch), n_hot);
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
                LOG_ERR("%s draft vocabulary scored an excluded ID\n", llm_arch_name(arch));
                return false;
            }
        }
        if (nmse(selected_expected, selected_actual) > 1e-4) {
            LOG_ERR("%s draft vocabulary changed selected logits\n", llm_arch_name(arch));
            return false;
        }
        llama_memory_clear(llama_get_memory(target), true);
        if (get_logits(model, target, tokens) != target_before) {
            LOG_ERR("%s draft vocabulary changed target logits\n", llm_arch_name(arch));
            return false;
        }
    }
    return true;
}

static bool test_glm_mtp_index(const size_t seed, const std::vector<ggml_backend_dev_t> & devs,
        llama_split_mode split_mode, const std::vector<float> & tensor_split) {
    auto gguf = get_gguf_ctx(LLM_ARCH_GLM5NEXT, true, true);
    auto pair = get_model_and_ctx(gguf.get(), nullptr, seed, devs, split_mode, false, tensor_split, true);
    auto * model = pair.first.get();
    int scores = 0;
    llama_context_params params = llama_context_default_params();
    params.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
    params.n_ctx = 128;
    params.n_batch = params.n_ubatch = 64;
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
    if (!ctx || !reference || llama_mtp_draft_step(pair.second.get(), 0)) {
        return false;
    }
    const auto tokens = get_tokens(40, 128, seed);
    const int n_embd = llama_model_n_embd(model);
    std::vector<float> hidden(40*n_embd, 0.1f);
    get_mtp_logits(model, ctx.get(), tokens, hidden);
    get_mtp_logits(model, reference.get(), tokens, hidden);
    const auto step = [&](int i, int pos, bool fresh) {
        scores = 0;
        if (!llama_mtp_draft_step(ctx.get(), i)) {
            return false;
        }
        const auto logits = get_mtp_logits(model, ctx.get(), {7}, hidden, pos);
        const auto expected = get_mtp_logits(model, reference.get(), {7}, hidden, pos);
        if ((scores > 0) != fresh || !std::all_of(logits.begin(), logits.end(), [](float x) { return std::isfinite(x); })) {
            LOG_ERR("MTP step %d at %d: expected fresh=%d, indexer ran %d times\n", i, pos, fresh, scores);
            return false;
        }
        // Reuse must keep writing raw and compressed keys. When scoring resumes,
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
    llama_model_saver ms(LLM_ARCH_GLM5NEXT, gguf.get());
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
        params.n_batch = params.n_ubatch = 128;
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
            const int scoring_ubatches = step == 0 ? (!unified && split_capture ? 2 : 1) :
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

static bool test_glm_short_dense(const size_t seed, const std::vector<ggml_backend_dev_t> & devs,
        llama_split_mode split_mode, const std::vector<float> & tensor_split) {
    auto gguf = get_gguf_ctx(LLM_ARCH_GLM5NEXT, true, true);
    llama_model_saver ms(LLM_ARCH_GLM5NEXT, gguf.get());
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
                common_batch_add(batch, input[pos], pos, {seq_id}, true);
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
        llama_split_mode split_mode, const std::vector<float> & tensor_split, const std::string & diagnostic_dir = "") {
    auto gguf = get_gguf_ctx(LLM_ARCH_GLM5NEXT, true, true);
    llama_model_saver ms(LLM_ARCH_GLM5NEXT, gguf.get());
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_TOP_K, uint32_t(4));
    auto pair = get_model_and_ctx(gguf.get(), nullptr, seed, devs, split_mode, false, tensor_split, true);
    auto * model = pair.first.get();
    const int vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    std::mt19937 rng(seed);
    const bool diagnostics = !diagnostic_dir.empty();
    for (const bool unified : {false, true}) {
        auto params = llama_context_default_params();
        params.n_ctx = 2048;
        params.n_seq_max = 3;
        params.n_rs_seq = 8;
        params.n_batch = 512;
        params.n_ubatch = 64;
        params.n_threads = params.n_threads_batch = 4;
        params.kv_unified = unified;
        llama_context_ptr actual(llama_init_from_model(model, params));
        llama_context_ptr reference(llama_init_from_model(model, params));
        if (!actual || !reference) return false;
        std::vector<llama_pos> positions(3, 0);
        const auto decode = [&](llama_context * ctx, int seq, int pos, const std::vector<llama_token> & input, int n) {
            llama_batch batch = llama_batch_init(n, 0, 1);
            for (int i = 0; i < n; ++i) common_batch_add(batch, input[i], pos + i, {seq}, i == n-1);
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
                if (!decode(actual.get(), seq, pos, input, width) ||
                    !decode(reference.get(), seq, pos, input, keep)) return false;
                if (diagnostics) {
                    evidence.snapshot("snapshot-selected.bin", [&] { return glm_snapshot(actual.get(), seq, width-keep, pos+keep-1); });
                    if (keep < width) evidence.snapshot("snapshot-late.bin", [&] { return glm_snapshot(actual.get(), seq, width-keep-1, pos+keep); });
                }
                if (keep < width && !llama_memory_seq_rm(llama_get_memory(actual.get()), seq, pos+keep, -1)) return false;
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
        case LLM_ARCH_LLAMA4:
        case LLM_ARCH_COHERE2MOE:
        case LLM_ARCH_GROK:
        case LLM_ARCH_QWEN2MOE:
        case LLM_ARCH_QWEN3MOE:
        case LLM_ARCH_QWEN3NEXT:
        case LLM_ARCH_QWEN3VLMOE:
        case LLM_ARCH_QWEN35MOE:
        case LLM_ARCH_PHIMOE:
        case LLM_ARCH_DBRX:
        case LLM_ARCH_OLMOE:
        case LLM_ARCH_ARCTIC:
        case LLM_ARCH_DEEPSEEK:
        case LLM_ARCH_DEEPSEEK2:
        case LLM_ARCH_DEEPSEEK32:
        case LLM_ARCH_DOTS3NOTE:
        case LLM_ARCH_DEEPSEEK4:
        case LLM_ARCH_GLM4_MOE:
        case LLM_ARCH_GLM_DSA:
        case LLM_ARCH_GLM5NEXT:
        case LLM_ARCH_EXAONE_MOE:
        case LLM_ARCH_BAILINGMOE:
        case LLM_ARCH_BAILINGMOE2:
        case LLM_ARCH_BAILINGMOE3:
        case LLM_ARCH_DOTS1:
        case LLM_ARCH_AFMOE:
        case LLM_ARCH_ERNIE4_5:
        case LLM_ARCH_ERNIE4_5_MOE:
        case LLM_ARCH_HUNYUAN_MOE:
        case LLM_ARCH_HY_V3:
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
        case LLM_ARCH_STEP35:
        case LLM_ARCH_MISTRAL4:
        case LLM_ARCH_MELLUM:
        case LLM_ARCH_LAGUNA:
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
    if (arch == LLM_ARCH_GEMMA4 || arch == LLM_ARCH_GEMMA4_ASSISTANT) {
        return false; // FIXME @ngxson
    }
    if (arch == LLM_ARCH_GRANITE_SWITCH) {
        return false; // FIXME adapter fixture
    }
    if (arch == LLM_ARCH_LLAMA_EMBED || arch == LLM_ARCH_GEMMA_EMBEDDING || arch == LLM_ARCH_T5ENCODER) {
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
    if (arch == LLM_ARCH_DEEPSEEK2OCR) {
        return false;
    }
    // FIXME: these hit scheduler/view-backed-output issues with WebGPU on CI.
#ifdef GGML_USE_WEBGPU
    if (arch == LLM_ARCH_DEEPSEEK32 || arch == LLM_ARCH_GLM_DSA || arch == LLM_ARCH_DOTS3NOTE) {
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

static int save_models(const llm_arch target_arch, const size_t seed, const ggml_log_level log_level, const std::string & dir) {
    struct user_data_t {
        struct {
            ggml_log_callback callback;
            void * user_data;
        } original_logger;
        ggml_log_level min_level; // prints below this log level go to debug log
    };
    user_data_t ud;
    llama_log_get(&ud.original_logger.callback, &ud.original_logger.user_data);
    ud.min_level = log_level;

    llama_log_set([](ggml_log_level level, const char * text, void * user_data) {
        const user_data_t * ud = (const user_data_t *) user_data;
        const ggml_log_level level_eff = level >= ud->min_level ? level : GGML_LOG_LEVEL_DEBUG;
        ud->original_logger.callback(level_eff, text, ud->original_logger.user_data);
    }, &ud);

    for (const llm_arch & arch : llm_arch_all()) {
        if (arch == LLM_ARCH_UNKNOWN) {
            continue;
        }
        if (target_arch != LLM_ARCH_UNKNOWN && arch != target_arch) {
            continue;
        }
        if (arch == LLM_ARCH_GEMMA4 || arch == LLM_ARCH_GEMMA4_ASSISTANT) {
            continue; // FIXME: ISWA KV cache initialization needs more fixture params
        }
        if (arch == LLM_ARCH_EAGLE3 || arch == LLM_ARCH_DFLASH) {
            continue;
        }
        for (bool moe : {false, true}) {
            if (moe && !moe_implemented(arch)) {
                continue;
            }
            if (!moe && moe_mandatory(arch)) {
                continue;
            }
            if (!llama_model_saver_supports_arch(arch) || !arch_supported(arch)) {
                LOG_INF("%s: %s model (%s) is unsupported, skipping\n", __func__, llm_arch_name(arch), moe ? "MoE" : "dense");
                continue;
            }
            gguf_context_ptr gguf_ctx = get_gguf_ctx(arch, moe);
            auto model_and_ctx = get_model_and_ctx(gguf_ctx.get(), nullptr, seed, {});
            const std::string path = dir + "/" + llm_arch_name(arch) + (moe ? "-moe.gguf" : "-dense.gguf");
            LOG_INF("%s: Saving %s model (%s) to %s...\n", __func__, llm_arch_name(arch), moe ? "MoE" : "dense", path.c_str());
            llama_model_save_to_file(model_and_ctx.first.get(), path.c_str());
        }
    }
    llama_log_set(ud.original_logger.callback, ud.original_logger.user_data);
    return 0;
}

static int test_backends(const llm_arch target_arch, const size_t seed, const ggml_log_level log_level) {
    struct user_data_t {
        struct {
            ggml_log_callback callback;
            void * user_data;
        } original_logger;
        ggml_log_level min_level; // prints below this log level go to debug log
    };
    user_data_t ud;
    llama_log_get(&ud.original_logger.callback, &ud.original_logger.user_data);
    ud.min_level = log_level;

    llama_log_set([](ggml_log_level level, const char * text, void * user_data) {
        const user_data_t * ud = (const user_data_t *) user_data;
        const ggml_log_level level_eff = level >= ud->min_level ? level : GGML_LOG_LEVEL_DEBUG;
        ud->original_logger.callback(level_eff, text, ud->original_logger.user_data);
    }, &ud);

    const std::vector<llama_token> tokens = get_tokens(128, 128, seed);

    struct device_config {
        std::vector<ggml_backend_dev_t> devs;
        std::string                     label;
        llama_split_mode                split_mode;
        std::vector<float>              tensor_split; // empty: the default, even split

        device_config(std::vector<ggml_backend_dev_t> devs, std::string name, llama_split_mode split_mode, std::vector<float> tensor_split = {})
            : devs(std::move(devs)), label(std::move(name)), split_mode(split_mode), tensor_split(std::move(tensor_split)) {}
    };

    std::vector<device_config> dev_configs;
    size_t max_device_label_length = 4;
    {
        std::vector<ggml_backend_dev_t> devices_meta;
        {
            const size_t device_count = ggml_backend_dev_count();
            for (size_t i = 0; i < device_count; i++) {
                ggml_backend_dev_t dev = ggml_backend_dev_get(i);
                dev_configs.emplace_back(std::vector<ggml_backend_dev_t>{dev}, ggml_backend_dev_description(dev), LLAMA_SPLIT_MODE_LAYER);
                max_device_label_length = std::max(max_device_label_length, dev_configs.back().label.length());

                // cpu-based devices cannot be used in tensor split mode
                if (ggml_backend_dev_buffer_type(dev) != ggml_backend_cpu_buffer_type()) {
                    devices_meta.push_back(dev);
                }
            }
        }
        // one GPU twice splits as two GPUs do, so that a machine with one tests the splits too
        if (devices_meta.size() == 1) {
            devices_meta.push_back(devices_meta[0]);
        }

        dev_configs.emplace_back(devices_meta, "Meta", LLAMA_SPLIT_MODE_TENSOR);

        // 3:7 puts the boundaries off every power-of-two granularity, so a tensor that splits out of step with the ones
        // it is computed with (a weight and its bias, the heads of q and of the output projection) fails here
        std::vector<float> tensor_split_uneven(llama_max_devices(), 0.0f);
        for (size_t j = 0; j < devices_meta.size(); j++) {
            tensor_split_uneven[j] = j % 2 == 0 ? 3.0f : 7.0f;
        }
        dev_configs.emplace_back(devices_meta, "Meta 3:7", LLAMA_SPLIT_MODE_TENSOR, tensor_split_uneven);
        max_device_label_length = std::max(max_device_label_length, dev_configs.back().label.length());
    }

    size_t max_arch_name_length = 0;
    for (const llm_arch & arch : llm_arch_all()) {
        max_arch_name_length = std::max(max_arch_name_length, strlen(llm_arch_name(arch)));
    }

    const std::string template_header  = std::string("|%" + std::to_string(max_arch_name_length) + "s|%") + std::to_string(max_device_label_length) + "s|%6s|%15s|%9s|\n";
    const std::string template_row_cfg = std::string("|%" + std::to_string(max_arch_name_length) + "s|%") + std::to_string(max_device_label_length) + "s|%6s|";
    const std::string template_row_res = "%15s %10s|%20s|\n";

    bool all_ok = true;
    common_log_flush(common_log_main());
    printf(template_header.c_str(), "Model arch.", "Device", "Config", "NMSE vs. CPU", "Roundtrip");
    printf("|");
    for (size_t i = 0; i < max_arch_name_length; i++) {
        printf("-");
    }
    printf("|");
    for (size_t i = 0; i < max_device_label_length; i++) {
        printf("-");
    }
    printf("|------|---------------|---------|\n");
    for (const llm_arch & arch : llm_arch_all()) {
        if (arch == LLM_ARCH_UNKNOWN) {
            continue;
        }
        if (target_arch != LLM_ARCH_UNKNOWN && arch != target_arch) {
            continue;
        }
        if (arch == LLM_ARCH_GEMMA4 || arch == LLM_ARCH_GEMMA4_ASSISTANT) {
            continue; // FIXME: ISWA KV cache initialization needs more fixture params
        }
        if (arch == LLM_ARCH_EAGLE3 || arch == LLM_ARCH_DFLASH) {
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
                // print test config first; should anything fail during model loading or inference, at least we know which test case caused it
                printf(template_row_cfg.c_str(),
                    llm_arch_name(arch), dc.label.c_str(), config_name.c_str());
                fflush(stdout);

                std::pair<llama_model_ptr, llama_context_ptr> model_and_ctx_dev;
                std::vector<float> logits_dev;
                std::string status_nmse      = "\033[1;33mSKIP\033[0m";
                std::string status_roundtrip = "\033[1;33mSKIP\033[0m";
                char nmse_str[12] = {0};
                bool skip = !arch_supported(arch) || (dc.split_mode == LLAMA_SPLIT_MODE_TENSOR && dc.devs.empty());
                if (!skip) {
                    if (logits_cpu.empty()) {
                        model_and_ctx_cpu = get_model_and_ctx(gguf_ctx.get(), nullptr, seed, {}, LLAMA_SPLIT_MODE_LAYER, encode);
                        logits_cpu = get_logits(model_and_ctx_cpu.first.get(), model_and_ctx_cpu.second.get(), tokens, encode);
                    }
                    if (dc.split_mode != LLAMA_SPLIT_MODE_TENSOR || llm_arch_supports_sm_tensor(arch)) {
                        model_and_ctx_dev = get_model_and_ctx(gguf_ctx.get(), nullptr, seed, dc.devs, dc.split_mode, encode, dc.tensor_split);
                        logits_dev = get_logits(model_and_ctx_dev.first.get(), model_and_ctx_dev.second.get(), tokens, encode);
                        const double nmse_val = nmse(logits_cpu, logits_dev);
                        snprintf(nmse_str, sizeof(nmse_str), "(%.2e)", nmse_val);
                        status_nmse = "\033[1;32mOK\033[0m";
                        if (nmse_val > 1e-4) {
                            all_ok = false;
                            status_nmse = "\033[1;31mFAIL\033[0m";
                        }
                        if ((arch == LLM_ARCH_GLM5NEXT || arch == LLM_ARCH_QWEN35) &&
                                !test_draft_vocab(arch, seed, dc.devs, dc.split_mode, dc.tensor_split)) {
                            all_ok = false;
                            status_nmse = "\033[1;31mFAIL MTP vocab\033[0m";
                        }
                        if (arch == LLM_ARCH_GLM5NEXT && !test_glm_generated_prefixes(seed, dc.devs, dc.split_mode, dc.tensor_split)) {
                            throw std::runtime_error("GLM generated speculative prefixes failed");
                        }
                        if (arch == LLM_ARCH_GLM5NEXT && !test_glm_short_dense(seed, dc.devs, dc.split_mode, dc.tensor_split)) {
                            all_ok = false;
                            status_nmse = "\033[1;31mFAIL short dense\033[0m";
                        }
                        if (arch == LLM_ARCH_GLM5NEXT && !test_glm_mtp_index(seed, dc.devs, dc.split_mode, dc.tensor_split)) {
                            all_ok = false;
                            status_nmse = "\033[1;31mFAIL MTP index\033[0m";
                        }
                        if (arch == LLM_ARCH_GLM5NEXT && !test_glm_mtp_index_sequences(seed, dc.devs, dc.split_mode, dc.tensor_split)) {
                            all_ok = false;
                            status_nmse = "\033[1;31mFAIL MTP sequences\033[0m";
                        }
                    }

                    FILE * file = tmpfile(); // Can be null on Windows without administrator privileges.
                    // FIXME: when adding a tensor to a gguf_context a copy is made, this changes the pointer which the meta backend
                    //     in turn uses to map the tensors to their simple equivalents - this is fundamentally incompatible
                    if (file != nullptr && llama_model_saver_supports_arch(arch) && dc.split_mode != LLAMA_SPLIT_MODE_TENSOR) {
                        GGML_ASSERT(model_and_ctx_dev.first && model_and_ctx_dev.second);
                        llama_model_saver ms = llama_model_saver(model_and_ctx_dev.first.get());
                        ms.add_kv_from_model();
                        ms.add_tensors_from_model();
                        ms.save(file);
                        rewind(file);

                        auto model_and_ctx_roundtrip = get_model_and_ctx(nullptr, file, seed, dc.devs, dc.split_mode, encode);
                        const std::vector<float> logits_roundtrip = get_logits(
                            model_and_ctx_roundtrip.first.get(), model_and_ctx_roundtrip.second.get(), tokens, encode);
                        status_roundtrip = "\033[1;32mOK\033[0m";
                        GGML_ASSERT(logits_roundtrip.size() == logits_dev.size());
                        for (size_t i = 0; i < logits_roundtrip.size(); i++) {
                            if (logits_roundtrip[i] != logits_dev[i]) {
                                all_ok = false;
                                status_roundtrip = "\033[1;31mFAIL\033[0m";
                                break;
                            }
                        }
                    }
                }

                // log the results for this test case
                printf(template_row_res.c_str(),
                    status_nmse.c_str(), nmse_str, status_roundtrip.c_str());
            }
        }
    }
    llama_log_set(ud.original_logger.callback, ud.original_logger.user_data);
    return all_ok ? 0 : 1;
}

int main(int argc, char ** argv) {
    if (argc == 2 && strcmp(argv[1], "--test-state-comparison") == 0) {
        const int comparison = test_glm_state_comparison();
        return test_glm_capture_failures() || comparison;
    }
    // FIXME these tests are disabled in the CI for macOS-latest-cmake-arm64 because they are segfaulting
    common_init();
    std::random_device rd;

    llm_arch arch = LLM_ARCH_UNKNOWN;
    size_t seed = rd();
    ggml_log_level log_level = GGML_LOG_LEVEL_ERROR;
    std::string out;
    std::string diagnostic_dir, diagnostic_device;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--glm-prefix-diagnostics") == 0 || strcmp(argv[i], "--glm-prefix-device") == 0) {
            const bool directory = strcmp(argv[i], "--glm-prefix-diagnostics") == 0;
            if (i+1 == argc) return 1;
            (directory ? diagnostic_dir : diagnostic_device) = argv[++i];
            continue;
        }
        if (strcmp(argv[i], "-a") == 0 || strcmp(argv[i], "--arch") == 0) {
            if (i + 1 < argc) {
                const std::string arch_name = argv[++i];
                arch = llm_arch_from_string(arch_name);
                if (arch == LLM_ARCH_UNKNOWN) {
                    LOG_ERR("%s: unkown LLM architecture: %s\n", __func__, arch_name.c_str());
                    return 1;
                }
            } else {
                usage(argv);
                return 1;
            }
        }
        if (strcmp(argv[i], "-s") == 0 || strcmp(argv[i], "--seed") == 0) {
            if (i + 1 < argc) {
                seed = std::stoull(argv[++i]);
            } else {
                usage(argv);
                return 1;
            }
        }
        if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--verbose") == 0) {
            log_level = GGML_LOG_LEVEL_INFO;
            continue;
        }
        if (strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--out") == 0) {
            if (i + 1 < argc) {
                out = argv[++i];
            } else {
                usage(argv);
                return 1;
            }
        }
    }
    printf("%s: using seed %zu\n", __func__, seed);

    try {
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
            return save_models(arch, seed, log_level, out);
        }
        return test_backends(arch, seed, log_level);
    } catch (const std::exception & err) {
        fprintf(stderr, "encountered runtime error: %s\n", err.what());
        return -1;
    }
}
