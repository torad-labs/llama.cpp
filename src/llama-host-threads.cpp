#include "llama-host-threads.h"

#include "ggml.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <mutex>
#include <system_error>
#include <thread>
#include <vector>

static std::atomic<int> g_host_threads_set  {0};
static std::atomic<int> g_host_threads_last {1};

namespace {

thread_local bool g_in_host_pool = false;

// Persistent worker pool for llama_host_rows, avoiding per-step thread spawn/join overhead.
class llama_host_pool {
public:
    ~llama_host_pool() {
        {
            std::lock_guard<std::mutex> lk(mu);
            stop = true;
            ++gen;
        }
        cv_work.notify_all();
        for (auto & w : workers) {
            if (w.joinable()) {
                w.join();
            }
        }
    }

    void run(int n_threads, void (*fn)(int, void *), void * userdata) {
        if (n_threads <= 1 || g_in_host_pool) {
            for (int t = 0; t < n_threads; ++t) {
                fn(t, userdata);
            }
            return;
        }
        struct in_pool_guard {
            ~in_pool_guard() { g_in_host_pool = false; }
        } pool_flag;
        g_in_host_pool = true;
        std::lock_guard<std::mutex> call_lk(call_mu);
        struct work_guard {
            llama_host_pool * self;
            bool dispatched = false;
            ~work_guard() {
                if (dispatched) {
                    std::unique_lock<std::mutex> lk(self->mu);
                    self->cv_done.wait(lk, [this]() { return self->remaining.load(std::memory_order_acquire) == 0; });
                }
            }
        } guard { this };

        int active = 0;
        {
            std::lock_guard<std::mutex> lk(mu);
            const int need = n_threads - 1;
            while ((int) workers.size() < need) {
                const int tid = (int) workers.size() + 1;
                try {
                    workers.emplace_back([this, tid]() { worker_loop(tid); });
                } catch (const std::system_error &) {
                    break;
                }
            }
            active       = std::min(need, (int) workers.size());
            cur_fn       = fn;
            cur_userdata = userdata;
            cur_threads  = active + 1;
            remaining.store(active, std::memory_order_release);
            ++gen;
            guard.dispatched = active > 0;
        }
        if (active > 0) {
            cv_work.notify_all();
        }

        fn(0, userdata);
        for (int t = active + 1; t < n_threads; ++t) {
            fn(t, userdata);
        }
    }

private:
    void worker_loop(int tid) {
        g_in_host_pool = true;
        uint64_t seen = 0;
        while (true) {
            void (*fn)(int, void *) = nullptr;
            void * userdata = nullptr;
            int n_threads = 0;
            {
                std::unique_lock<std::mutex> lk(mu);
                cv_work.wait(lk, [this, seen]() { return stop || gen != seen; });
                if (stop) {
                    return;
                }
                seen      = gen;
                fn        = cur_fn;
                userdata  = cur_userdata;
                n_threads = cur_threads;
            }
            if (tid < n_threads) {
                struct done_guard {
                    llama_host_pool * self;
                    ~done_guard() {
                        if (self->remaining.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                            std::lock_guard<std::mutex> lk(self->mu);
                            self->cv_done.notify_one();
                        }
                    }
                } done { this };
                fn(tid, userdata);
            }
        }
    }

    std::mutex call_mu;
    std::mutex mu;
    std::condition_variable cv_work;
    std::condition_variable cv_done;
    std::vector<std::thread> workers;
    uint64_t gen = 0;
    bool stop = false;
    void (*cur_fn)(int, void *) = nullptr;
    void * cur_userdata = nullptr;
    int cur_threads = 0;
    std::atomic<int> remaining {0};
};

} // namespace

void llama_host_run(int n_threads, void (*fn)(int, void *), void * userdata) {
    static llama_host_pool pool;
    pool.run(n_threads, fn, userdata);
}

static int llama_host_threads_default() {
    static const int n = []() {
        const int v = (int) ggml_env_int_or("LLAMA_HOST_THREADS", 0);
        if (v > 0) {
            return v;
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
