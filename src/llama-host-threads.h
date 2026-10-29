#pragma once

// The host builds some graph inputs a row per token of the ubatch and n_kv cells wide: the KQ mask, and the pooled indexer's
// sel_mask, cand_mask and pool_bias (llama_kpool_set_input). On one thread a 1,024-token prompt ubatch in a 110K-cell pool
// left both GPUs idle about 130 ms per graph on the lab box (2x RTX PRO 6000, Oct 29, 2026), and the gap grows with the
// pool's depth. The rows are independent, so the build splits them across threads: each row is written by one thread, from
// the same reads, so the bytes are those of one thread.

#include <algorithm>
#include <cstdint>
#include <thread>
#include <vector>

// the threads a build of n_rows rows, row_bytes each, splits across: one below 4 MiB of rows, else LLAMA_HOST_THREADS
// (1 keeps one thread) or half the hardware threads, at most 16, and never more than one per row
int llama_host_threads(int64_t n_rows, int64_t row_bytes);

// for tests: n > 0 makes every build that splits use n threads, whatever its size (still at most one per row); 0 restores
// the default
void llama_host_threads_set(int n);

// for tests: the threads the last build that asked split across
int llama_host_threads_last();

// fn(i0, i1, t) writes rows [i0, i1); t, from 0, tells the threads apart for their own counters. The calling thread takes
// the first rows
template <typename F>
void llama_host_rows(int64_t n_rows, int64_t row_bytes, F && fn) {
    const int64_t n_threads = llama_host_threads(n_rows, row_bytes);

    if (n_threads <= 1) {
        fn((int64_t) 0, n_rows, 0);
        return;
    }

    const int64_t chunk = (n_rows + n_threads - 1)/n_threads;

    std::vector<std::thread> workers;
    workers.reserve(n_threads - 1);

    for (int64_t t = 1; t < n_threads; ++t) {
        const int64_t i0 = t*chunk;
        const int64_t i1 = std::min(n_rows, i0 + chunk);

        if (i0 >= i1) {
            break;
        }

        workers.emplace_back([&fn, i0, i1, t]() { fn(i0, i1, (int) t); });
    }

    fn((int64_t) 0, std::min(chunk, n_rows), 0);

    for (auto & w : workers) {
        w.join();
    }
}
