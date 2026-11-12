#pragma once

#include "llama.h"

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <map>
#include <mutex>
#include <thread>
#include <vector>

struct ggml_tensor;

// Copies a restored sequence's tensor rows from host memory on a thread of its own, so the caller's thread decodes other sequences meanwhile.
// The rows go to cells only that sequence holds, and nothing reads them before wait() returns for it.
class llama_state_restore {
public:
    struct write {
        ggml_tensor *   tensor;
        const uint8_t * ptr;
        size_t          size;
        size_t          offset;
    };

    llama_state_restore();
    ~llama_state_restore();

    // A write with no list of device writes (ggml_backend_meta_tensor_ranges) is copied here, on the caller's thread.
    void submit(llama_seq_id seq_id, const std::vector<write> & writes);

    // seq_id < 0: any sequence
    bool pending(llama_seq_id seq_id);
    void wait(llama_seq_id seq_id);

private:
    struct job {
        llama_seq_id       seq_id;
        std::vector<write> writes;
    };

    void run();

    // LLAMA_STATE_RESTORE_CHUNK_KIB: the bytes of one copy call. A copy engine takes copies of one direction in the order they come, so a small copy of another stream waits for one chunk at most. 0: a write is one call
    const size_t chunk;

    std::mutex mutex;
    std::condition_variable cv;
    std::condition_variable cv_done;
    std::deque<job> jobs;
    std::map<llama_seq_id, uint32_t> n_pending;
    bool stop = false;
    std::thread worker;
};
