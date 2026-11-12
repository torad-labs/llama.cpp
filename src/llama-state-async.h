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

// Copies a sequence's tensor rows between host memory and the devices on a thread of its own, so the caller's thread decodes other sequences meanwhile: a restored sequence's rows from a state in host memory, a saved sequence's rows to one.
// The rows are in cells only that sequence holds. Nothing reads the restored ones and nothing writes the saved ones before wait() returns for it.
class llama_state_async {
public:
    // [ptr, ptr + size) of host memory and [offset, offset + size) of a tensor
    struct copy {
        ggml_tensor * tensor;
        uint8_t *     ptr;
        size_t        size;
        size_t        offset;
    };

    llama_state_async();
    ~llama_state_async();

    // to_device: host memory is copied to the tensors (a restore) and is only read. Otherwise the tensors are copied to host memory (a save).
    // A copy with no list of device copies (ggml_backend_meta_tensor_ranges) is made here, on the caller's thread.
    void submit(llama_seq_id seq_id, const std::vector<copy> & copies, bool to_device);

    // seq_id == -1: any copy. A copy from spare rows no sequence holds (llama_context::state_seq_get_data_staged) has a
    // key of its own under -1, staged_key(), so a decode of the sequence does not wait for it.
    bool pending(llama_seq_id seq_id);
    void wait(llama_seq_id seq_id);

    static llama_seq_id staged_key(int32_t row) { return -2 - row; }

private:
    struct job {
        llama_seq_id      seq_id;
        bool              to_device;
        std::vector<copy> copies;
    };

    void run();

    // LLAMA_STATE_ASYNC_CHUNK_KIB: the bytes of one copy call. A copy engine takes copies of one direction in the order they come, so a small copy of another stream waits for one chunk at most. 0: a copy is one call
    const size_t chunk;

    std::mutex mutex;
    std::condition_variable cv;
    std::condition_variable cv_done;
    std::deque<job> jobs;
    std::map<llama_seq_id, uint32_t> n_pending;
    bool stop = false;
    std::thread worker;
};
