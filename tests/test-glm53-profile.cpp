#include "arg.h"
#include "common.h"
#include "glm53-profile.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    // Every combination must either resolve to a compatible bundle or reject
    // an explicit contradiction. No partial environment application is needed.
    for (const std::string profile : {"canonical", "sm120-balanced", "sm120-wide", "sm120-direct"}) {
        for (const auto & pair : {std::pair<const char *, const char *>{"GGML_CUDA_Q8_TMA", "GGML_CUDA_Q8_SPLIT"},
                                 {"GGML_CUDA_GDN_STATE_PERSIST", "GGML_CUDA_GDN_STATE_ARENA"},
                                 {"GGML_CUDA_HC_CTA_PUBLISH", "GGML_CUDA_HC_FRONT_ONE"}}) {
            const auto off = common_glm53_profile_resolve(profile, {{pair.second, "0"}});
            if (off.at(pair.first) != "0") { return 4; }
            for (const char * prerequisite : {"0", "1"}) {
                for (const char * dependent : {"0", "1"}) {
                    const bool invalid = prerequisite[0] == '0' && dependent[0] == '1';
                    bool rejected = false;
                    try { common_glm53_profile_resolve(profile, {{pair.second, prerequisite}, {pair.first, dependent}}); }
                    catch (const std::invalid_argument &) { rejected = true; }
                    if (invalid != rejected) { return 5; }
                }
            }
        }
        if (common_glm53_profile_resolve(profile, {{"GGML_CUDA_KDA_COLUMNS", "1"}}).at("GGML_CUDA_KDA_FRONT_SHARED") != "0") return 6;
        bool rejected = false;
        try { common_glm53_profile_resolve(profile, {{"GGML_CUDA_KDA_COLUMNS", "1"}, {"GGML_CUDA_KDA_FRONT_SHARED", "1"}}); }
        catch (const std::invalid_argument &) { rejected = true; }
        if (!rejected) return 7;
    }
    const std::string name = argc > 1 ? argv[1] : "sm120-balanced";
    const bool explicit_sampling = argc > 2 && strcmp(argv[2], "--backend-sampling") == 0;
    std::vector<std::string> args {"test-glm53-profile", "--model", "profile-test.gguf", "--glm53-profile", name};
    if (explicit_sampling) { args.emplace_back("--backend-sampling"); }
    std::vector<char *> ptrs;
    for (auto & a : args) { ptrs.push_back(a.data()); }
    common_params params;
    if (!common_params_parse((int)ptrs.size(), ptrs.data(), params, LLAMA_EXAMPLE_COMMON)) {
        fprintf(stderr, "FAIL: complete GLM profile was not accepted\n");
        return 1;
    }
    const auto eq = [](const char * key, const char * value) {
        const char * got = getenv(key);
        if (!got || strcmp(got, value)) {
            fprintf(stderr, "FAIL: %s = %s, expected %s\n", key, got ? got : "(unset)", value);
            return false;
        }
        return true;
    };
    const bool canonical = name == "canonical";
    const bool direct = name == "sm120-direct";
    const bool wide = name == "sm120-wide";
    if (!eq("GGML_CUDA_Q8_SPLIT", canonical ? "0" : "1") ||
        !eq("GGML_CUDA_Q8_TMA", canonical || direct ? "0" : "1") ||
        !eq("GGML_CUDA_Q8_TMA_SLOTS", wide ? "4" : "2") ||
        !eq("GGML_CUDA_IQ3_COMPACT", canonical ? "0" : "1") ||
        !eq("GGML_CUDA_KDA_COLUMNS", canonical ? "1" : wide ? "4" : "2") ||
        !eq("GGML_CUDA_KDA_FRONT_SHARED", canonical || direct ? "0" : "1") ||
        !eq("GGML_CUDA_GDN_STATE_ARENA", canonical ? "0" : "1") ||
        !eq("GGML_CUDA_GDN_STATE_PERSIST", canonical || direct ? "0" : "1") ||
        !eq("GGML_CUDA_HC_CTA_PUBLISH", canonical ? "0" : "1") ||
        !eq("GGML_CUDA_AR_HC_POST", canonical || direct ? "0" : "1") ||
        params.sampling.backend_sampling != (!canonical || explicit_sampling)) { return 2; }
    // Once resolved, changing a profile in-process must not reuse descriptors
    // or captured graphs from its previous storage/ownership configuration.
    args[4] = canonical ? "sm120-balanced" : "canonical";
    ptrs.clear();
    for (auto & a : args) { ptrs.push_back(a.data()); }
    common_params changed;
    if (common_params_parse((int)ptrs.size(), ptrs.data(), changed, LLAMA_EXAMPLE_COMMON)) { return 3; }
    puts("complete profile, target sampling and reload boundary passed");
    return 0;
}
