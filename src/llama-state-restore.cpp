#include "llama-state-restore.h"

#include "llama-context.h"
#include "llama-impl.h"
#include "llama-io.h"

#include "ggml-backend.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

static size_t llama_state_restore_chunk() {
    const char * value = getenv("LLAMA_STATE_RESTORE_CHUNK_KIB");
    const long long kib = value ? atoll(value) : 1024;
    return kib > 0 ? (size_t) kib * 1024 : 0;
}

llama_state_restore::llama_state_restore() : chunk(llama_state_restore_chunk()), worker([this]() { run(); }) {}

llama_state_restore::~llama_state_restore() {
    {
        std::lock_guard<std::mutex> lock(mutex);
        stop = true;
    }
    cv.notify_all();
    worker.join();
}

void llama_state_restore::submit(llama_seq_id seq_id, const std::vector<write> & writes) {
    struct plan {
        std::vector<write> * out;
        const uint8_t *      ptr;
    };

    job next;
    next.seq_id = seq_id;
    for (const auto & w : writes) {
        plan p = { &next.writes, w.ptr };
        const bool ok = ggml_backend_meta_tensor_ranges(w.tensor, w.offset, w.size,
            [](ggml_tensor * tensor, size_t offset, size_t size, size_t data_offset, void * userdata) {
                auto * p = (plan *) userdata;
                p->out->push_back({ tensor, p->ptr + data_offset, size, offset });
            }, &p);
        if (!ok) {
            ggml_backend_tensor_set(w.tensor, w.ptr, w.offset, w.size);
        }
    }
    if (next.writes.empty()) {
        return;
    }

    {
        std::lock_guard<std::mutex> lock(mutex);
        n_pending[seq_id]++;
        jobs.push_back(std::move(next));
    }
    cv.notify_one();
}

bool llama_state_restore::pending(llama_seq_id seq_id) {
    std::lock_guard<std::mutex> lock(mutex);
    return seq_id < 0 ? !n_pending.empty() : n_pending.count(seq_id) > 0;
}

void llama_state_restore::wait(llama_seq_id seq_id) {
    std::unique_lock<std::mutex> lock(mutex);
    cv_done.wait(lock, [&]() { return seq_id < 0 ? n_pending.empty() : n_pending.count(seq_id) == 0; });
}

void llama_state_restore::run() {
    std::unique_lock<std::mutex> lock(mutex);
    while (true) {
        cv.wait(lock, [this]() { return stop || !jobs.empty(); });
        if (jobs.empty()) {
            return;
        }
        job cur = std::move(jobs.front());
        jobs.pop_front();
        lock.unlock();

        for (const auto & w : cur.writes) {
            for (size_t done = 0; done < w.size;) {
                const size_t n = chunk > 0 ? std::min(chunk, w.size - done) : w.size - done;
                ggml_backend_tensor_set(w.tensor, w.ptr + done, w.offset + done, n);
                done += n;
            }
        }

        lock.lock();
        const auto it = n_pending.find(cur.seq_id);
        if (--it->second == 0) {
            n_pending.erase(it);
        }
        cv_done.notify_all();
    }
}

//
// llama_context
//

// the first bytes of a sequence state: io_magic of llama-context.cpp
static constexpr uint32_t io_magic = 0xaf143cd8;

// A sequence state in host memory, read as llama_io_read_host of llama-context.cpp reads it. The tensor reads are kept as a list and not copied.
class llama_io_read_listed : public llama_io_read_i {
public:
    llama_io_read_listed(const uint8_t * p, size_t len) : ptr(p), buf_size(len) {}

    void read(void * dst, size_t size) override {
        memcpy(dst, take(size), size);
    }

    void read_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        writes.push_back({ tensor, take(size), size, offset });
    }

    size_t n_bytes() override {
        return size_read;
    }

    std::vector<llama_state_restore::write> writes;

private:
    const uint8_t * take(size_t size) {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        const uint8_t * res = ptr;
        ptr       += size;
        size_read += size;
        buf_size  -= size;
        return res;
    }

    const uint8_t * ptr;
    size_t buf_size  = 0;
    size_t size_read = 0;
};

size_t llama_context::state_seq_set_data_async(llama_seq_id seq_id, const uint8_t * src, size_t size, llama_state_seq_flags flags) {
    if (flags & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE) {
        // the tensor data of such a state is on the device: no copy from host memory to take off this thread
        return state_seq_set_data(seq_id, src, size, flags);
    }

    llama_io_read_listed io(src, size);
    bool ok = false;
    try {
        uint32_t magic_read;
        io.read(&magic_read, sizeof(magic_read));
        if (io_magic != magic_read) {
            throw std::runtime_error("wrong sequence state magic");
        }

        llama_seq_id seq_id_read;
        io.read(&seq_id_read, sizeof(seq_id_read));

        // waits for an earlier restore of the sequence, then removes its cells and places them again
        state_seq_read_data(io, seq_id, flags);
        ok = true;
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading state: %s\n", __func__, err.what());
    }

    if (ok) {
        if (!state_restore) {
            state_restore = std::make_unique<llama_state_restore>();
        }
        state_restore->submit(seq_id, io.writes);
        return io.n_bytes();
    }

    // as state_seq_set_data after a failure: what was read before it is copied, on this thread
    for (const auto & w : io.writes) {
        ggml_backend_tensor_set(w.tensor, w.ptr, w.offset, w.size);
    }
    return 0;
}

bool llama_context::state_restore_pending(llama_seq_id seq_id) const {
    return state_restore && state_restore->pending(seq_id);
}

void llama_context::state_restore_wait(llama_seq_id seq_id) const {
    if (state_restore) {
        state_restore->wait(seq_id);
    }
}

void llama_context::state_restore_wait(const llama_batch & batch) const {
    if (!state_restore || !state_restore->pending(-1)) {
        return;
    }
    if (batch.seq_id == nullptr) {
        // llama_batch_get_one: every token is of sequence 0
        state_restore->wait(0);
        return;
    }
    llama_seq_id seq_id_prev = -1;
    for (int32_t i = 0; i < batch.n_tokens; i++) {
        for (int32_t s = 0; s < batch.n_seq_id[i]; s++) {
            const llama_seq_id seq_id = batch.seq_id[i][s];
            if (seq_id != seq_id_prev) {
                state_restore->wait(seq_id);
                seq_id_prev = seq_id;
            }
        }
    }
}

//
// interface implementation
//

size_t llama_state_seq_set_data_async(llama_context * ctx, const uint8_t * src, size_t size, llama_seq_id dest_seq_id, llama_state_seq_flags flags) {
    ctx->synchronize();

    return ctx->state_seq_set_data_async(dest_seq_id, src, size, flags);
}

bool llama_state_seq_restore_pending(const llama_context * ctx, llama_seq_id seq_id) {
    return ctx->state_restore_pending(seq_id);
}

void llama_state_seq_restore_wait(const llama_context * ctx, llama_seq_id seq_id) {
    ctx->state_restore_wait(seq_id);
}
