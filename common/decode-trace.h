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

struct range {
    bool on = enabled();
    std::string label;

    explicit range(const char * text) { if (on) { begin(text); } }
    explicit range(const std::string & text) { if (on) { begin(text); } }
    template <typename MakeLabel>
    explicit range(MakeLabel make_label) { if (on) { begin(make_label()); } }
    range(const range &) = delete;
    range & operator=(const range &) = delete;

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
    }
};
} // namespace common_decode_trace
