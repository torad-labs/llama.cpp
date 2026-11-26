#include "llama-state-async.h"

#include "llama-context.h"
#include "llama-impl.h"
#include "llama-io.h"
#include "llama-memory-hybrid-iswa.h"
#include "llama-memory-recurrent.h"

#include "ggml-backend.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

static size_t llama_state_async_chunk() {
    return (size_t) ggml_env_int("LLAMA_STATE_ASYNC_CHUNK_KIB") * 1024;
}

llama_state_async::llama_state_async() : chunk(llama_state_async_chunk()), worker([this]() { run(); }) {}

llama_state_async::~llama_state_async() {
    {
        std::lock_guard<std::mutex> lock(mutex);
        stop = true;
    }
    cv.notify_all();
    worker.join();
}

static constexpr size_t llama_state_read_scratch_bytes = 4*1024*1024;

struct llama_state_copy_plan {
    std::vector<llama_state_async::copy> & out;
    uint8_t * dst;
};

static bool llama_state_plan_split_rows(const llama_state_async::copy & c, std::vector<llama_state_async::copy> & out) {
    static const bool legacy = ggml_env_switch("GGML_TENSOR_N_STAGE_LEGACY");
    if (legacy) { return false; }
    llama_state_copy_plan plan = {out, c.ptr};
    return ggml_backend_meta_read_rows(c.tensor, c.offset, c.size,
            [](ggml_tensor * t, size_t start, size_t width, size_t rows, size_t pitch, size_t stride, size_t host, void * userdata) {
                auto & plan = *static_cast<llama_state_copy_plan *>(userdata);
                plan.out.push_back({t, plan.dst + host, width, start, rows, pitch, stride});
            }, &plan);
}

void llama_state_async::submit(llama_seq_id seq_id, const std::vector<copy> & copies, bool to_device, bool split_rows) {
    job next;
    next.seq_id    = seq_id;
    next.to_device = to_device;
    // copies with no list of device copies: one batched call here, which a split buffer makes as few copies per device
    std::vector<ggml_tensor *> tensors;
    std::vector<size_t>        offsets;
    std::vector<size_t>        sizes;
    std::vector<uint8_t *>     ptrs;
    for (const auto & c : copies) {
        llama_state_copy_plan p = {next.copies, c.ptr};
        const bool ok = ggml_backend_meta_tensor_ranges(c.tensor, c.offset, c.size, /*read =*/ !to_device,
            [](ggml_tensor * tensor, size_t offset, size_t size, size_t data_offset, void * userdata) {
                auto & p = *static_cast<llama_state_copy_plan *>(userdata);
                p.out.push_back({tensor, p.dst + data_offset, size, offset});
            }, &p);
        if (ok) {
            continue;
        }
        if (split_rows && llama_state_plan_split_rows(c, next.copies)) {
            next.packed_split = true;
            continue;
        }
        tensors.push_back(c.tensor);
        offsets.push_back(c.offset);
        sizes.push_back(c.size);
        ptrs.push_back(c.ptr);
    }
    if (!tensors.empty()) {
        uint8_t * base = *std::min_element(ptrs.begin(), ptrs.end());
        std::vector<size_t> data_offsets(ptrs.size());
        for (size_t i = 0; i < ptrs.size(); ++i) {
            data_offsets[i] = ptrs[i] - base;
        }
        if (to_device) {
            ggml_backend_tensor_set_n(tensors.data(), base, offsets.data(), sizes.data(), data_offsets.data(), tensors.size());
        } else {
            ggml_backend_tensor_get_n(tensors.data(), base, offsets.data(), sizes.data(), data_offsets.data(), tensors.size());
        }
    }
    if (next.copies.empty()) {
        return;
    }

    {
        std::lock_guard<std::mutex> lock(mutex);
        n_pending[seq_id]++;
        jobs.push_back(std::move(next));
    }
    cv.notify_one();
}

bool llama_state_async::pending(llama_seq_id seq_id) {
    std::lock_guard<std::mutex> lock(mutex);
    return seq_id == -1 ? !n_pending.empty() : n_pending.count(seq_id) > 0;
}

void llama_state_async::wait(llama_seq_id seq_id) {
    std::unique_lock<std::mutex> lock(mutex);
    cv_done.wait(lock, [&]() { return seq_id == -1 ? n_pending.empty() : n_pending.count(seq_id) == 0; });
}

static void llama_state_read_batch(const llama_state_async::copy * copies, size_t count, std::vector<uint8_t> & scratch) {
    if (count == 1) {
        ggml_backend_tensor_get(copies->tensor, copies->ptr, copies->offset, copies->size);
        return;
    }
    struct group {
        ggml_backend_buffer_type_t type;
        std::vector<const ggml_tensor *> tensors;
        std::vector<size_t> offsets, sizes, host_offsets;
        std::vector<uint8_t *> dests;
        size_t bytes = 0;
    };
    std::vector<group> groups;
    for (size_t i = 0; i < count; ++i) {
        const auto & c = copies[i];
        auto type = ggml_backend_buffer_get_type(c.tensor->view_src ? c.tensor->view_src->buffer : c.tensor->buffer);
        auto it = std::find_if(groups.begin(), groups.end(), [&](const group & g) { return g.type == type; });
        if (it == groups.end()) {
            groups.emplace_back();
            it = groups.end() - 1;
            it->type = type;
        }
        it->tensors.push_back(c.tensor);
        it->offsets.push_back(c.offset);
        it->sizes.push_back(c.size);
        it->host_offsets.push_back(it->bytes);
        it->dests.push_back(c.ptr);
        it->bytes += c.size;
    }
    for (const auto & g : groups) {
        // One contiguous destination per device lets its gather use a single D2H. Rank slices are interleaved on the CPU.
        scratch.resize(g.bytes);
        ggml_backend_tensor_get_n(g.tensors.data(), scratch.data(), g.offsets.data(), g.sizes.data(), g.host_offsets.data(), g.sizes.size());
        for (size_t j = 0; j < g.sizes.size(); ++j) {
            memcpy(g.dests[j], scratch.data() + g.host_offsets[j], g.sizes[j]);
        }
    }
}

template <bool write>
static void llama_state_transfer_split(const std::vector<llama_state_async::copy> & copies, size_t limit, std::vector<uint8_t> & scratch) {
    using copy = llama_state_async::copy;
    struct group {
        ggml_backend_buffer_t buffer;
        std::vector<const copy *> copies;
    };
    std::vector<group> groups;
    for (const auto & c : copies) {
        auto buffer = c.tensor->view_src ? c.tensor->view_src->buffer : c.tensor->buffer;
        auto it = std::find_if(groups.begin(), groups.end(), [&](const group & g) { return g.buffer == buffer; });
        if (it == groups.end()) {
            groups.push_back({ buffer, {} });
            it = groups.end() - 1;
        }
        it->copies.push_back(&c);
    }

    constexpr size_t max_ranges = 16*1024;
    std::vector<ggml_tensor *> tensors;
    std::vector<size_t> offsets, sizes, host_offsets;
    struct piece {
        uint8_t * host;
        size_t start, width, rows, stride;
    };
    std::vector<piece> pieces;
    size_t used = 0;
    const auto flush = [&]() {
        if (tensors.empty()) {
            return;
        }
        if constexpr (write) {
            ggml_backend_tensor_set_n(tensors.data(), scratch.data(), offsets.data(), sizes.data(), host_offsets.data(), tensors.size());
        } else {
            ggml_backend_tensor_get_n(tensors.data(), scratch.data(), offsets.data(), sizes.data(), host_offsets.data(), tensors.size());
            for (const auto & p : pieces) {
                for (size_t j = 0; j < p.rows; ++j) {
                    memcpy(p.host + j*p.stride, scratch.data() + p.start + j*p.width, p.width);
                }
            }
            pieces.clear();
        }
        tensors.clear();
        offsets.clear();
        sizes.clear();
        host_offsets.clear();
        used = 0;
    };
    for (const auto & g : groups) {
        for (const copy * c : g.copies) {
            if (c->size == 0) {
                continue;
            }
            if (c->size > limit) {
                flush();
                for (size_t r = 0; r < c->rows; ++r) {
                    const size_t pos = c->offset + r*c->tensor_stride;
                    uint8_t * host = c->ptr + r*c->data_stride;
                    // A strided copy still transfers whole rows when one exceeds the batch budget.
                    const size_t step = c->rows > 1 ? c->size : limit;
                    for (size_t done = 0; done < c->size; done += step) {
                        const size_t n = std::min(step, c->size - done);
                        if constexpr (write) {
                            ggml_backend_tensor_set(c->tensor, host + done, pos + done, n);
                        } else {
                            ggml_backend_tensor_get(c->tensor, host + done, pos + done, n);
                        }
                    }
                }
                continue;
            }
            const bool contiguous = c->rows == 1 || c->tensor_stride == c->size;
            const auto add = [&](size_t row, size_t n, size_t start) {
                tensors.push_back(c->tensor);
                offsets.push_back(c->offset + row*c->tensor_stride);
                sizes.push_back(n*c->size);
                host_offsets.push_back(start);
            };
            for (size_t r = 0; r < c->rows;) {
                const size_t ranges = max_ranges - tensors.size();
                const size_t n = ranges ? std::min({ c->rows - r, (limit - used)/c->size,
                        contiguous ? c->rows - r : ranges }) : 0;
                if (n == 0) {
                    flush();
                    continue;
                }
                if (scratch.size() < limit) {
                    scratch.resize(limit);
                }
                uint8_t * host = c->ptr + r*c->data_stride;
                if constexpr (write) {
                    for (size_t j = 0; j < n; ++j) {
                        memcpy(scratch.data() + used + j*c->size, host + j*c->data_stride, c->size);
                    }
                } else {
                    pieces.push_back({ host, used, c->size, n, c->data_stride });
                }
                if (contiguous) {
                    add(r, n, used);
                } else {
                    for (size_t j = 0; j < n; ++j) {
                        add(r + j, 1, used + j*c->size);
                    }
                }
                used += n*c->size;
                r += n;
            }
        }
        flush();
    }
}

void llama_state_async::run() {
    std::vector<uint8_t> scratch;
    std::unique_lock<std::mutex> lock(mutex);
    while (true) {
        cv.wait(lock, [this]() { return stop || !jobs.empty(); });
        if (jobs.empty()) {
            return;
        }
        job cur = std::move(jobs.front());
        jobs.pop_front();
        lock.unlock();

        if (cur.packed_split) {
            const size_t limit = chunk ? std::min(chunk, llama_state_read_scratch_bytes) : llama_state_read_scratch_bytes;
            if (cur.to_device) {
                llama_state_transfer_split<true>(cur.copies, limit, scratch);
            } else {
                llama_state_transfer_split<false>(cur.copies, limit, scratch);
            }
        } else {
            // Batch consecutive copies up to the copy budget.
            static const bool batched = !ggml_env_switch("GGML_TENSOR_N_STAGE_LEGACY");
            std::vector<ggml_tensor *> tensors;
            std::vector<size_t> offsets, sizes, data_offsets;
            for (size_t i = 0; i < cur.copies.size();) {
                size_t n_batch = 0;
                size_t bytes = 0;
                uint8_t * base = cur.copies[i].ptr;
                while (batched && i + n_batch < cur.copies.size()) {
                    const auto & c = cur.copies[i + n_batch];
                    if (chunk > 0 && c.size > chunk - bytes) {
                        break;
                    }
                    base = std::min(base, c.ptr);
                    bytes += c.size;
                    n_batch++;
                }
                if (n_batch > 1) {
                    tensors.clear();
                    offsets.clear();
                    sizes.clear();
                    data_offsets.clear();
                    for (size_t j = i; j < i + n_batch; ++j) {
                        const auto & c = cur.copies[j];
                        tensors.push_back(c.tensor);
                        offsets.push_back(c.offset);
                        sizes.push_back(c.size);
                        data_offsets.push_back(c.ptr - base);
                    }
                    if (cur.to_device) {
                        ggml_backend_tensor_set_n(tensors.data(), base, offsets.data(), sizes.data(), data_offsets.data(), n_batch);
                    } else {
                        ggml_backend_tensor_get_n(tensors.data(), base, offsets.data(), sizes.data(), data_offsets.data(), n_batch);
                    }
                    i += n_batch;
                    continue;
                }
                const auto & c = cur.copies[i++];
                for (size_t done = 0; done < c.size;) {
                    const size_t n = chunk > 0 ? std::min(chunk, c.size - done) : c.size - done;
                    if (cur.to_device) {
                        ggml_backend_tensor_set(c.tensor, c.ptr + done, c.offset + done, n);
                    } else {
                        ggml_backend_tensor_get(c.tensor, c.ptr + done, c.offset + done, n);
                    }
                    done += n;
                }
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
        // the copy to the device only reads these bytes
        copies.push_back({ tensor, const_cast<uint8_t *>(take(size)), size, offset });
    }

    void read_tensor_now(ggml_tensor * tensor, size_t offset, size_t size) override {
        ggml_backend_tensor_set(tensor, take(size), offset, size);
    }

    size_t n_bytes() override {
        return size_read;
    }

    std::vector<llama_state_async::copy> copies;

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

// A sequence state written to host memory as llama_io_write_host of llama-context.cpp writes it. The tensor writes are kept as a list and not copied.
class llama_io_write_listed : public llama_io_write_i {
public:
    llama_io_write_listed(uint8_t * p, size_t len) : ptr(p), buf_size(len) {}

    void write(const void * src, size_t size) override {
        memcpy(take(size), src, size);
    }

    void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        copies.push_back({ tensor, take(size), size, offset });
    }

    void write_tensor_now(ggml_tensor * tensor, size_t offset, size_t size) override {
        auto * dst = take(size);
        const llama_state_async::copy c = {tensor, dst, size, offset};
        if (size == 0) { return; }
        if (!llama_state_plan_split_rows(c, now)) {
            ggml_backend_tensor_get(tensor, dst, offset, size);
        }
    }

    void capture_rows() {
        // Other sequences can move these rows. Complete their reads before returning to the caller.
        std::vector<uint8_t> scratch;
        for (size_t i = 0; i < now.size();) {
            if (now[i].rows > 1 || now[i].size > llama_state_read_scratch_bytes) {
                const auto & c = now[i++];
                ggml_backend_tensor_get_2d(c.tensor, c.ptr, c.offset, c.size, c.rows, c.tensor_stride, c.data_stride);
                continue;
            }
            size_t n = 1, bytes = now[i].size;
            while (i+n < now.size() && now[i+n].rows == 1 && now[i+n].size <= llama_state_read_scratch_bytes - bytes) {
                bytes += now[i+n++].size;
            }
            llama_state_read_batch(now.data() + i, n, scratch);
            i += n;
        }
    }

    size_t n_bytes() override {
        return size_written;
    }

    std::vector<llama_state_async::copy> copies;

private:
    uint8_t * take(size_t size) {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        uint8_t * res = ptr;
        ptr          += size;
        size_written += size;
        buf_size     -= size;
        return res;
    }

    std::vector<llama_state_async::copy> now;
    uint8_t * ptr;
    size_t buf_size     = 0;
    size_t size_written = 0;
};

size_t llama_context::state_seq_get_data_async(llama_seq_id seq_id, uint8_t * dst, size_t size, llama_state_seq_flags flags) {
    if (flags & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE) {
        // the tensor data of such a state stays on the device: no copy to host memory to take off this thread
        return state_seq_get_data(seq_id, dst, size, flags);
    }

    llama_io_write_listed io(dst, size);
    state_seq_copies = 0;
    try {
        io.write(&io_magic, sizeof(io_magic));
        io.write(&seq_id,   sizeof(seq_id));

        // waits for an earlier copy of the sequence, then writes where its cells are
        state_seq_write_data(io, seq_id, flags);
        io.capture_rows();
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving state: %s\n", __func__, err.what());
        return 0;
    }

    if (!state_async) {
        state_async = std::make_unique<llama_state_async>();
    }
    state_async->submit(seq_id, io.copies, /*to_device =*/ false, !(flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY));
    state_seq_copies = io.copies.size();
    return io.n_bytes();
}

size_t llama_context::state_seq_set_data_async(llama_seq_id seq_id, const uint8_t * src, size_t size, llama_state_seq_flags flags) {
    if (flags & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE) {
        // the tensor data of such a state is on the device: no copy from host memory to take off this thread
        return state_seq_set_data(seq_id, src, size, flags);
    }

    llama_io_read_listed io(src, size);
    state_seq_copies = 0;
    bool ok = false;
    try {
        uint32_t magic_read;
        io.read(&magic_read, sizeof(magic_read));
        if (io_magic != magic_read) {
            throw std::runtime_error("wrong sequence state magic");
        }

        llama_seq_id seq_id_read;
        io.read(&seq_id_read, sizeof(seq_id_read));

        // waits for an earlier copy of the sequence, then removes its cells and places them again
        state_seq_read_data(io, seq_id, flags);
        ok = true;
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading state: %s\n", __func__, err.what());
    }

    if (ok) {
        if (!state_async) {
            state_async = std::make_unique<llama_state_async>();
        }
        state_async->submit(seq_id, io.copies, /*to_device =*/ true, /*split_rows =*/ true);
        state_seq_copies = io.copies.size();
        return io.n_bytes();
    }

    // as state_seq_set_data after a failure: what was read before it is copied, on this thread
    for (const auto & c : io.copies) {
        ggml_backend_tensor_set(c.tensor, c.ptr, c.offset, c.size);
    }
    return 0;
}

size_t llama_context::state_seq_get_data_staged(llama_seq_id seq_id, uint8_t * dst, size_t size, llama_state_seq_flags flags, int32_t & row) {
    row = -1;

    llama_memory_recurrent * mem = nullptr;
    if (memory && (flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) && !(flags & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE)) {
        mem = memory->staged();
    }
    if (mem != nullptr) {
        // an earlier copy of the sequence may still write the row that is read here
        state_async_wait(seq_id);

        // a sequence the memory dropped has no state to save
        if (memory->seq_lost(seq_id)) {
            LLAMA_LOG_ERROR("%s: the memory dropped the state of sequence %d\n", __func__, seq_id);
            return 0;
        }

        const int64_t src = mem->stage_src(seq_id);
        for (uint32_t k = 0; src >= 0 && k < mem->n_stage; ++k) {
            if (!state_async_pending(llama_state_async::staged_key(k))) {
                row = k;
                break;
            }
        }
        if (row >= 0) {
            auto * hybrid_iswa = dynamic_cast<llama_memory_hybrid_iswa *>(memory.get());
            llama_kv_cache * kv_swa = hybrid_iswa ? hybrid_iswa->get_mem_attn()->get_swa() : nullptr;
            llama_kv_cache::stage_plan swa_plan;
            llama_io_write_listed io(dst, size);
            try {
                if (kv_swa != nullptr) {
                    if (!kv_swa->stage_copy(seq_id, (uint32_t) row,
                            [this](ggml_backend_dev_t dev) { return backend_from_dev(dev); }, swa_plan)) {
                        row = -1;
                        return state_seq_get_data(seq_id, dst, size, flags);
                    }
                }
                std::vector<ggml_tensor *> states;
                std::vector<ggml_tensor *> stages;
                mem->stage_tensors(row, states, stages);
                rs_stage(states, stages, (int32_t) src, row);
                // the thread that copies reads the spare row by calls of its own, which do not wait for this
                // thread's: the copy to the spare row is done before it starts
                synchronize();

                io.write(&io_magic, sizeof(io_magic));
                io.write(&seq_id,   sizeof(seq_id));

                if (kv_swa != nullptr) {
                    kv_swa->state_write_staged(io, seq_id, (uint32_t) row, swa_plan);
                }
                mem->state_write_staged(io, seq_id, row);
                io.capture_rows();
            } catch (const std::exception & err) {
                LLAMA_LOG_ERROR("%s: error saving state: %s\n", __func__, err.what());
                row = -1;
                return 0;
            }

            if (!state_async) {
                state_async = std::make_unique<llama_state_async>();
            }
            state_async->submit(llama_state_async::staged_key(row), io.copies, /*to_device =*/ false, /*split_rows =*/ kv_swa != nullptr);
            state_seq_copies = io.copies.size();
            return io.n_bytes();
        }
    }

    return state_seq_get_data(seq_id, dst, size, flags);
}

uint32_t llama_context::state_staged_rows() const {
    llama_memory_recurrent * mem = memory ? memory->staged() : nullptr;
    return mem != nullptr ? mem->n_stage : 0;
}

bool llama_context::state_async_pending(llama_seq_id seq_id) const {
    return state_async && state_async->pending(seq_id);
}

void llama_context::state_async_wait(llama_seq_id seq_id) const {
    if (state_async) {
        state_async->wait(seq_id);
    }
}

void llama_context::state_async_wait(const llama_batch & batch) const {
    if (!state_async || !state_async->pending(-1)) {
        return;
    }
    if (batch.seq_id == nullptr) {
        // llama_batch_get_one: every token is of sequence 0
        state_async->wait(0);
        return;
    }
    llama_seq_id seq_id_prev = -1;
    for (int32_t i = 0; i < batch.n_tokens; i++) {
        for (int32_t s = 0; s < batch.n_seq_id[i]; s++) {
            const llama_seq_id seq_id = batch.seq_id[i][s];
            if (seq_id != seq_id_prev) {
                state_async->wait(seq_id);
                seq_id_prev = seq_id;
            }
        }
    }
}

//
// interface implementation
//

size_t llama_state_seq_get_data_async(llama_context * ctx, uint8_t * dst, size_t size, llama_seq_id seq_id, llama_state_seq_flags flags) {
    ctx->synchronize();

    return ctx->state_seq_get_data_async(seq_id, dst, size, flags);
}

size_t llama_state_seq_set_data_async(llama_context * ctx, const uint8_t * src, size_t size, llama_seq_id dest_seq_id, llama_state_seq_flags flags) {
    ctx->synchronize();

    return ctx->state_seq_set_data_async(dest_seq_id, src, size, flags);
}

size_t llama_state_seq_get_data_staged(llama_context * ctx, uint8_t * dst, size_t size, llama_seq_id seq_id, llama_state_seq_flags flags, int32_t * row) {
    ctx->synchronize();

    int32_t taken = -1;
    const size_t res = ctx->state_seq_get_data_staged(seq_id, dst, size, flags, taken);
    if (row != nullptr) {
        *row = taken;
    } else if (taken >= 0) {
        ctx->state_async_wait(llama_state_async::staged_key(taken));
    }
    return res;
}

uint32_t llama_state_staged_rows(const llama_context * ctx) {
    return ctx->state_staged_rows();
}

bool llama_state_staged_pending(const llama_context * ctx, int32_t row) {
    return row >= 0 && ctx->state_async_pending(llama_state_async::staged_key(row));
}

void llama_state_staged_wait(const llama_context * ctx, int32_t row) {
    if (row >= 0) {
        ctx->state_async_wait(llama_state_async::staged_key(row));
    }
}

bool llama_state_seq_async_pending(const llama_context * ctx, llama_seq_id seq_id) {
    return ctx->state_async_pending(seq_id);
}

void llama_state_seq_async_wait(const llama_context * ctx, llama_seq_id seq_id) {
    ctx->state_async_wait(seq_id);
}

size_t llama_state_seq_n_copies(const llama_context * ctx) {
    return ctx->state_seq_copies;
}
