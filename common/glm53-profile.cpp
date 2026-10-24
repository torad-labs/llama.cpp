#include "glm53-profile.h"
#include "common.h"
#include "log.h"
#include <cctype>
#include <cstdlib>
#include <mutex>
#include <stdexcept>

std::map<std::string, std::string> common_glm53_profile_resolve(
        const std::string & name, const std::map<std::string, std::string> & overrides) {
    const bool canonical = name == "canonical";
    const bool wide = name == "sm120-wide";
    const bool direct = name == "sm120-direct";
    if (!canonical && !wide && !direct && name != "sm120-balanced") {
        throw std::invalid_argument("unknown GLM profile: " + name);
    }
    std::map<std::string, std::string> out {
        {"GGML_CUDA_Q8_SPLIT", canonical ? "0" : "1"},
        {"GGML_CUDA_Q8_TMA", canonical || direct ? "0" : "1"},
        {"GGML_CUDA_Q8_TMA_SLOTS", wide ? "4" : "2"},
        {"GGML_CUDA_IQ3_COMPACT", canonical ? "0" : "1"},
        {"GGML_CUDA_IQ3_SLOTS", wide ? "4" : "8"},
        {"GGML_CUDA_KDA_COLUMNS", canonical ? "1" : wide ? "4" : "2"},
        {"GGML_CUDA_KDA_FRONT_SHARED", canonical || direct ? "0" : "1"},
        {"GGML_CUDA_GDN_STATE_ARENA", canonical ? "0" : "1"},
        {"GGML_CUDA_GDN_STATE_PERSIST", canonical || direct ? "0" : "1"},
        {"GGML_CUDA_HC_FRONT_ONE", canonical ? "0" : "1"},
        {"GGML_CUDA_HC_CTA_PUBLISH", canonical ? "0" : "1"},
        {"GGML_CUDA_AR_HC_POST", canonical || direct ? "0" : "1"},
        {"LLAMA_GLM53_TARGET_SAMPLING", canonical ? "0" : "1"},
    };
    for (const auto & [key, value] : overrides) {
        if (!out.count(key)) { throw std::invalid_argument("unsupported GLM profile override: " + key); }
        const auto first = value.find_first_not_of(" \t\r\n\v\f");
        const auto last = value.find_last_not_of(" \t\r\n\v\f");
        std::string normalized = first == std::string::npos ? "" : value.substr(first, last - first + 1);
        const bool discrete = key == "GGML_CUDA_Q8_TMA_SLOTS" || key == "GGML_CUDA_IQ3_SLOTS" || key == "GGML_CUDA_KDA_COLUMNS";
        if (!discrete) {
            for (char & c : normalized) { c = (char) std::tolower((unsigned char) c); }
            if (normalized.empty() || normalized == "false" || normalized == "no" || normalized == "off") {
                normalized = "0";
            } else if (normalized == "true" || normalized == "yes" || normalized == "on") {
                normalized = "1";
            } else {
                char * end = nullptr;
                const long number = std::strtol(normalized.c_str(), &end, 10);
                if (end != normalized.c_str() && end == normalized.c_str() + normalized.size()) { normalized = number == 0 ? "0" : "1"; }
            }
        }
        const bool valid = key == "GGML_CUDA_Q8_TMA_SLOTS" ? normalized == "2" || normalized == "4" :
                           key == "GGML_CUDA_IQ3_SLOTS" ? normalized == "2" || normalized == "4" || normalized == "6" || normalized == "8" :
                           key == "GGML_CUDA_KDA_COLUMNS" ? normalized == "1" || normalized == "2" || normalized == "4" :
                           normalized == "0" || normalized == "1";
        if (!valid) { throw std::invalid_argument("invalid GLM profile value: " + key + "=" + value); }
        out[key] = normalized;
    }
    // Turning a prerequisite off also turns its implicit dependants off. An explicitly contradictory request fails before changing the environment.
    for (const auto & pair : {std::pair<const char *, const char *>{"GGML_CUDA_Q8_TMA", "GGML_CUDA_Q8_SPLIT"},
                             {"GGML_CUDA_GDN_STATE_PERSIST", "GGML_CUDA_GDN_STATE_ARENA"},
                             {"GGML_CUDA_HC_CTA_PUBLISH", "GGML_CUDA_HC_FRONT_ONE"}}) {
        if (out.at(pair.second) == "0") {
            if (overrides.count(pair.first) && out.at(pair.first) == "1") {
                throw std::invalid_argument(std::string(pair.first) + " requires " + pair.second);
            }
            out[pair.first] = "0";
        }
    }
    if (out.at("GGML_CUDA_KDA_COLUMNS") == "1") {
        if (overrides.count("GGML_CUDA_KDA_FRONT_SHARED") && out.at("GGML_CUDA_KDA_FRONT_SHARED") == "1") {
            throw std::invalid_argument("GGML_CUDA_KDA_FRONT_SHARED requires 2 or 4 KDA columns");
        }
        out["GGML_CUDA_KDA_FRONT_SHARED"] = "0";
    }
    return out;
}

void common_glm53_profile_apply(common_params & params) {
    if (params.glm53_profile.empty()) { return; }
    static std::mutex mutex;
    static std::string applied_name;
    static std::map<std::string, std::string> applied;
    const std::lock_guard<std::mutex> lock(mutex);
    if (!applied_name.empty() && applied_name != params.glm53_profile) {
        throw std::invalid_argument("changing GLM profiles requires a new process and model reload");
    }
    auto resolved = common_glm53_profile_resolve(params.glm53_profile);
    std::map<std::string, std::string> overrides;
    for (const auto & [key, value] : resolved) {
        if (const char * env = getenv(key.c_str())) { overrides[key] = env; }
    }
    resolved = common_glm53_profile_resolve(params.glm53_profile, overrides);
    if (const char * warps = getenv("GGML_CUDA_MMVQ_Q8_SM120_WARPS")) {
        if (*warps) { throw std::invalid_argument("GLM lossless profiles preserve the logical Q8 reduction; remove the WARPS override"); }
    }
    if (params.cache_type_s != GGML_TYPE_F32) {
        throw std::invalid_argument("GLM same-pack profiles require F32 recurrent state");
    }
    if (!applied.empty() && applied != resolved) {
        throw std::invalid_argument("changing resolved GLM options requires a new process and model reload");
    }
    if (applied.empty()) {
        for (const auto & [key, value] : resolved) {
#ifdef _WIN32
            const int rc = _putenv_s(key.c_str(), value.c_str());
#else
            const int rc = setenv(key.c_str(), value.c_str(), 1);
#endif
            if (rc != 0) { throw std::runtime_error("failed to apply GLM option " + key); }
            LOG_INF("GLM profile %s: %s=%s\n", params.glm53_profile.c_str(), key.c_str(), value.c_str());
        }
        applied = resolved;
        applied_name = params.glm53_profile;
    }
    params.sampling.backend_sampling = params.sampling.backend_sampling || resolved.at("LLAMA_GLM53_TARGET_SAMPLING") == "1";
}
