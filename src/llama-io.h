#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

struct ggml_tensor;

class llama_io_write_i {
public:
    llama_io_write_i() = default;
    virtual ~llama_io_write_i() = default;

    virtual void write(const void * src, size_t size) = 0;
    virtual void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) = 0;

    // write_tensor for rows that do not stay where they are: a recurrent state's, which the next decode moves or writes
    // over, a decode of other sequences included. A writer that copies its tensors later copies these before it returns
    virtual void write_tensor_now(ggml_tensor * tensor, size_t offset, size_t size) {
        write_tensor(tensor, offset, size);
    }

    // bytes written so far
    virtual size_t n_bytes() = 0;

    void write_string(const std::string & str);
};

class llama_io_read_i {
public:
    llama_io_read_i() = default;
    virtual ~llama_io_read_i() = default;

    virtual void read(void * dst, size_t size) = 0;
    virtual void read_tensor(ggml_tensor * tensor, size_t offset, size_t size) = 0;

    // read_tensor for rows that do not stay where they are, as write_tensor_now
    virtual void read_tensor_now(ggml_tensor * tensor, size_t offset, size_t size) {
        read_tensor(tensor, offset, size);
    }

    // drop tensor data that has been read but not yet applied (e.g. when a restore fails)
    virtual void discard() {}

    // bytes read so far
    virtual size_t n_bytes() = 0;

    void read_string(std::string & str);
};
