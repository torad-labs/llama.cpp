#include "llama-archs-gguf.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "llama-cpp.h"
#include "llama.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {
void bake_enabled(bool enabled) {
#ifdef _WIN32
    _putenv_s("LLAMA_INKLING2_ROUTER_BAKE", enabled ? "1" : "0");
#else
    setenv("LLAMA_INKLING2_ROUTER_BAKE", enabled ? "1" : "0", 1);
#endif
}

void fill(struct ggml_tensor * t, void *) {
    uint32_t seed = 0x9967u;
    for (const char * p = t->name; *p; ++p) {
        seed = (seed ^ uint8_t(*p))*16777619u;
    }
    std::minstd_rand random(seed);
    std::normal_distribution<float> values(0.0f, 0.1f);
    const bool decay = strstr(t->name, "ssm_a") != nullptr;
    const bool rope = strstr(t->name, "rope_factors") || strstr(t->name, "rope_freqs");

    const int64_t rows = ggml_nrows(t);
    const int64_t k = t->ne[0];
    std::vector<float> data(k);
    std::vector<ggml_fp16_t> half(k);
    for (int64_t i = 0; i < rows; ++i) {
        for (float & x : data) {
            const float raw = values(random);
            x = decay ? -std::abs(raw) : rope ? 1.0f + std::abs(raw) : raw;
        }
        if (t->type == GGML_TYPE_F32) {
            ggml_backend_tensor_set(t, data.data(), i*t->nb[1], data.size()*sizeof(float));
        } else {
            GGML_ASSERT(t->type == GGML_TYPE_F16);
            std::transform(data.begin(), data.end(), half.begin(), ggml_fp32_to_fp16);
            ggml_backend_tensor_set(t, half.data(), i*t->nb[1], half.size()*sizeof(ggml_fp16_t));
        }
    }
}

struct run_result {
    int projections = 0;
    std::vector<int32_t> routes;
    std::vector<float> router_logits;
    std::vector<llama_token> greedy;
};

bool observe(ggml_tensor * t, bool ask, void * user_data) {
    auto & result = *static_cast<run_result *>(user_data);
    const bool projected = strncmp(t->name, "ffn_moe_router_basis", 20) == 0;
    const bool logits = strncmp(t->name, "ffn_moe_logits-", 15) == 0;
    const bool topk = strncmp(t->name, "ffn_moe_topk-", 13) == 0;
    if (ask) {
        return projected || logits || topk;
    }
    if (projected) {
        ++result.projections;
    }
    if (logits) {
        GGML_ASSERT(t->type == GGML_TYPE_F32 && ggml_is_contiguous(t));
        size_t offset = result.router_logits.size();
        result.router_logits.resize(offset + ggml_nelements(t));
        ggml_backend_tensor_get(t, result.router_logits.data() + offset, 0, ggml_nbytes(t));
    }
    if (topk) {
        GGML_ASSERT(t->type == GGML_TYPE_I32 && t->nb[0] == sizeof(int32_t));
        for (int64_t r = 0; r < t->ne[1]; ++r) {
            size_t offset = result.routes.size();
            result.routes.resize(offset + t->ne[0]);
            ggml_backend_tensor_get(t, result.routes.data() + offset, r*t->nb[1], t->ne[0]*sizeof(int32_t));
        }
    }
    return true;
}

bool run(llm_arch arch, ggml_backend_dev_t dev, bool bake, run_result & result) {
    const bool inkling = arch == LLM_ARCH_INKLING2;
    auto gguf = get_gguf_ctx(arch, true);
    if (inkling) {
        kv_writer meta(arch, gguf.get());
        meta.add_kv(LLM_KV_EXPERT_COUNT, uint32_t(384));
        meta.add_kv(LLM_KV_EXPERT_USED_COUNT, uint32_t(8));
    }
    ggml_backend_dev_t devices[] = {dev, nullptr};
    auto mp = llama_model_default_params();
    mp.devices = devices;
    mp.n_gpu_layers = -1;
    mp.progress_callback = [](float, void *) { return true; };

    bake_enabled(bake);
    llama_model_ptr model(llama_model_init_from_user(gguf.get(), fill, nullptr, mp));
    bake_enabled(false);
    if (!model) {
        return false;
    }

    auto cp = llama_context_default_params();
    cp.n_ctx = 256;
    cp.n_batch = cp.n_ubatch = 32;
    cp.n_threads = cp.n_threads_batch = 4;
    cp.cb_eval = observe;
    cp.cb_eval_user_data = &result;
    llama_context_ptr ctx(llama_init_from_model(model.get(), cp));
    if (!ctx) {
        return false;
    }

    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
    std::vector<llama_token> prompt(12);
    for (size_t i = 0; i < prompt.size(); ++i) {
        prompt[i] = (i*31 + 7) % n_vocab;
    }
    const auto start = std::chrono::steady_clock::now();
    if (llama_decode(ctx.get(), llama_batch_get_one(prompt.data(), prompt.size())) != 0) {
        return false;
    }
    for (int i = 0; i < 12; ++i) {
        const float * logits = llama_get_logits_ith(ctx.get(), -1);
        if (!logits || !std::all_of(logits, logits + n_vocab, [](float v) { return std::isfinite(v); })) {
            return false;
        }
        llama_token next = std::max_element(logits, logits + n_vocab) - logits;
        result.greedy.push_back(next);
        if (llama_decode(ctx.get(), llama_batch_get_one(&next, 1)) != 0) {
            return false;
        }
    }
    llama_synchronize(ctx.get());
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    printf("%s %s: %d router projections, %zu routes, %zu greedy tokens, %.1f ms\n",
            inkling ? "Inkling 2" : "GLM 5.3", bake ? "switch on" : "reference",
            result.projections, result.routes.size(), result.greedy.size(), ms);
    return true;
}
} // namespace

int main() {
    ggml_backend_load_all();
    llama_backend_init();
    const auto dev = ggml_backend_dev_by_name("CUDA0");
    if (!dev) {
        fprintf(stderr, "CUDA0 is required\n");
        return 1;
    }

    bool ok = true;
    for (const auto arch : {LLM_ARCH_INKLING2, LLM_ARCH_GLM5NEXT}) {
        run_result ref, baked;
        const bool ran = run(arch, dev, false, ref) && run(arch, dev, true, baked);
        const bool inkling = arch == LLM_ARCH_INKLING2;
        const bool projected = inkling ? baked.projections > 0 : baked.projections == 0;
        const bool same = ran && ref.projections == 0 && projected &&
            !ref.routes.empty() && ref.routes == baked.routes &&
            !ref.router_logits.empty() && ref.router_logits == baked.router_logits &&
            !ref.greedy.empty() && ref.greedy == baked.greedy;
        if (!same) {
            fprintf(stderr, "%s router bake parity failed\n", inkling ? "Inkling 2" : "GLM 5.3");
        }
        ok &= same;
    }
    llama_backend_free();
    return ok ? 0 : 1;
}
