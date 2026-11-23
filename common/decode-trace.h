#pragma once

// Opt-in host identity markers. They never read device buffers or synchronize CUDA.
// NVTX ranges identify launches; neither these host spans nor stderr timestamps
// are GPU execution durations. Callers supply metadata, never token text or IDs.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <mutex>
#include <string>
#include <thread>

#include "ggml-ledger.h"

#if __has_include(<nvtx3/nvToolsExt.h>)
#include <nvtx3/nvToolsExt.h>
#define COMMON_DECODE_NVTX_AVAILABLE 1
#else
#define COMMON_DECODE_NVTX_AVAILABLE 0
#endif

namespace common_decode_trace {
inline bool enabled() {
    static const bool on = [] {
        const char * value = std::getenv("LLAMA_DECODE_TRACE");
        return value && std::strcmp(value, "1") == 0;
    }();
    return on;
}

inline void emit(const char * event, const std::string & label) {
    const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    static std::mutex mutex;
    const std::lock_guard<std::mutex> lock(mutex);
    std::fprintf(stderr, "GLM53_TRACE event=%s clock=host-monotonic host_ns=%lld thread=%zx nvtx=%d %s\n",
            event, (long long) ns, std::hash<std::thread::id>{}(std::this_thread::get_id()),
            COMMON_DECODE_NVTX_AVAILABLE, label.c_str());
}

inline void mark(const std::string & label) {
    if (!enabled()) {
        return;
    }
    emit("mark", label);
#if COMMON_DECODE_NVTX_AVAILABLE
    nvtxMarkA(("glm53.v1 " + label).c_str());
#endif
}

// The value after "key=" in a label of "key=value key=value" words, or "" when the label has no such key.
inline std::string label_value(const std::string & label, const char * key) {
    const std::string word = std::string(key) + "=";
    size_t at = label.rfind(word, 0) == 0 ? 0 : label.find(" " + word);
    if (at == std::string::npos) {
        return "";
    }
    at += word.size() + (at == 0 ? 0 : 1);
    return label.substr(at, label.find(' ', at) - at);
}

struct range {
    bool on = enabled();
    std::string label;
    // The ledger (ggml-ledger.h) takes every range as a stage of the server's step, trace or no trace: its time
    // under "stage.<name>" and, where the label says how many, its rows under "stage.<name>.rows".
    int ledger_slot = -1;
    std::chrono::steady_clock::time_point ledger_t0;

    explicit range(const char * text) { open(std::string(text)); }
    explicit range(const std::string & text) { open(text); }
    template <typename MakeLabel>
    explicit range(MakeLabel make_label) {
        if (on || ggml_ledger_on()) {
            open(make_label());
        }
    }
    range(const range &) = delete;
    range & operator=(const range &) = delete;

    void open(const std::string & text) {
        if (ggml_ledger_on()) {
            const std::string stage = label_value(text, "stage");
            if (!stage.empty()) {
                ledger_slot = ggml_ledger_slot_of("stage", stage.c_str());
                const std::string rows = label_value(text, "rows");
                if (!rows.empty()) {
                    ggml_ledger_add(ggml_ledger_slot_of("stage", (stage + ".rows").c_str()), std::atoll(rows.c_str()));
                }
                ledger_t0 = std::chrono::steady_clock::now();
            }
        }
        if (on) {
            begin(text);
        }
    }

    void begin(const std::string & text) {
        label = text;
        emit("begin", label);
#if COMMON_DECODE_NVTX_AVAILABLE
        nvtxRangePushA(("glm53.v1 " + label).c_str());
#endif
    }

    ~range() {
        if (on) {
#if COMMON_DECODE_NVTX_AVAILABLE
            nvtxRangePop();
#endif
            emit("end", label);
        }
        if (ledger_slot >= 0) {
            ggml_ledger_add(ledger_slot, std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - ledger_t0).count());
        }
    }
};
} // namespace common_decode_trace
