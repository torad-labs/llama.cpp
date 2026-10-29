#include "llama-host-threads.h"

#include <atomic>
#include <cstdlib>

static std::atomic<int> g_host_threads_set  {0};
static std::atomic<int> g_host_threads_last {1};

static int llama_host_threads_default() {
    static const int n = []() {
        if (const char * env = std::getenv("LLAMA_HOST_THREADS")) {
            const int v = std::atoi(env);
            if (v > 0) {
                return v;
            }
        }
        const unsigned hw = std::thread::hardware_concurrency();
        return std::clamp((int) (hw/2), 1, 16);
    }();
    return n;
}

int llama_host_threads(int64_t n_rows, int64_t row_bytes) {
    static constexpr int64_t min_bytes = 4ll << 20;
    // a decode or verify ubatch (a draft-mtp verify is n_max + 1 tokens) stays on one thread however deep the pool, so
    // decode runs as it did
    static constexpr int64_t min_rows = 8;

    const int forced = g_host_threads_set.load(std::memory_order_relaxed);

    int64_t n = forced > 0 ? forced : (n_rows < min_rows || n_rows*row_bytes < min_bytes ? 1 : llama_host_threads_default());

    n = std::max<int64_t>(1, std::min(n, n_rows));

    g_host_threads_last.store((int) n, std::memory_order_relaxed);

    return (int) n;
}

void llama_host_threads_set(int n) {
    g_host_threads_set.store(n > 0 ? n : 0, std::memory_order_relaxed);
}

int llama_host_threads_last() {
    return g_host_threads_last.load(std::memory_order_relaxed);
}
