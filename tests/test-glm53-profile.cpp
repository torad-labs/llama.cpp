#include "arg.h"
#include "common.h"
#include "glm53-profile.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static void set_env(const char * key, const char * value) {
#ifdef _WIN32
    _putenv_s(key, value ? value : "");
#else
    if (value) { setenv(key, value, 1); } else { unsetenv(key); }
#endif
}

static bool parse_server(std::vector<std::string> args, common_params & params) {
    std::vector<char *> ptrs;
    for (auto & arg : args) { ptrs.push_back(arg.data()); }
    return common_params_parse((int) ptrs.size(), ptrs.data(), params, LLAMA_EXAMPLE_SERVER);
}

static int check_router() {
    for (const auto & [key, value] : common_glm53_profile_resolve("canonical")) { set_env(key.c_str(), nullptr); }
    set_env("GGML_CUDA_AR_HC_POST", "0");
    common_params parent;
    if (!parse_server({"test", "--glm53-profile", "canonical"}, parent)) { return 10; }
    common_params remote_parent;
    if (!parse_server({"test", "--hf-repo", "org/remote-preset", "--glm53-profile", "canonical"}, remote_parent)) { return 10; }
    for (const auto & [key, value] : common_glm53_profile_resolve("canonical")) {
        if (key != "GGML_CUDA_AR_HC_POST" && getenv(key.c_str())) {
            fprintf(stderr, "router exported an implicit override: %s\n", key.c_str());
            return 11;
        }
    }
    set_env("GGML_CUDA_Q8_TMA", "1");
    common_params child;
    if (!parse_server({"test", "--model", "profile-test.gguf", "--glm53-profile", "sm120-balanced"}, child)) { return 12; }
    common_glm53_profile_apply(child);
    if (std::string(getenv("GGML_CUDA_Q8_SPLIT")) != "1" || std::string(getenv("GGML_CUDA_KDA_COLUMNS")) != "2" ||
        std::string(getenv("GGML_CUDA_AR_HC_POST")) != "0" || !child.sampling.backend_sampling) { return 13; }
    puts("router leaves child profile and explicit overrides intact");
    return 0;
}

static int check_spellings() {
    for (const std::string word : {"0", "false", "OFF", "No", "", "1", "true", "ON", "Yes", "2", "-1"}) {
        const bool enabled = word == "1" || word == "true" || word == "ON" || word == "Yes" || word == "2" || word == "-1";
        for (const std::string padding : {"", " ", "\t", "\r\n", "\v\f"}) {
            const auto value = padding + word + padding;
            const auto resolved = common_glm53_profile_resolve("sm120-balanced", {{"GGML_CUDA_AR_HC_POST", value}});
            if (resolved.at("GGML_CUDA_AR_HC_POST") != (enabled ? "1" : "0")) { return 14; }
        }
    }
    const auto resolved = common_glm53_profile_resolve("sm120-balanced", {
        {"GGML_CUDA_KDA_COLUMNS", " 2\r"}, {"GGML_CUDA_Q8_TMA_SLOTS", "\t4\r\n"}, {"GGML_CUDA_IQ3_SLOTS", "6 "}});
    if (resolved.at("GGML_CUDA_KDA_COLUMNS") != "2" || resolved.at("GGML_CUDA_Q8_TMA_SLOTS") != "4" ||
        resolved.at("GGML_CUDA_IQ3_SLOTS") != "6") { return 15; }
    bool rejected = false;
    try { common_glm53_profile_resolve("sm120-balanced", {{"GGML_CUDA_Q8_SPLIT", "off"}, {"GGML_CUDA_Q8_TMA", " YES\r"}}); }
    catch (const std::invalid_argument &) { rejected = true; }
    if (!rejected) { return 16; }
    set_env("GGML_CUDA_KDA_COLUMNS", "2\r");
    set_env("GGML_CUDA_AR_HC_POST", "off");
    common_params params;
    if (!parse_server({"test", "--model", "profile-test.gguf", "--glm53-profile", "sm120-balanced"}, params)) { return 17; }
    common_glm53_profile_apply(params);
    if (std::string(getenv("GGML_CUDA_KDA_COLUMNS")) != "2" || std::string(getenv("GGML_CUDA_AR_HC_POST")) != "0") { return 18; }
    puts("profile booleans and discrete values normalize before kernel dispatch");
    return 0;
}

int main(int argc, char ** argv) {
    if (argc > 1 && strcmp(argv[1], "--router") == 0) { return check_router(); }
    if (argc > 1 && strcmp(argv[1], "--spellings") == 0) { return check_spellings(); }
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
