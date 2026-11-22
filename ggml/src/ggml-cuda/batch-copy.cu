#include "batch-copy.cuh"
#include "common.cuh"
#include "ggml-cpp.h"

#include <algorithm>
#include <climits>
#include <mutex>
#include <type_traits>
#include <vector>

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
struct ggml_cuda_batch_copy_descriptor {
    const unsigned char * src;
    unsigned char * dst;
    size_t size;
};

static __global__ void ggml_cuda_batch_copy_kernel(const ggml_cuda_batch_copy_descriptor * copies, size_t count) {
    // Blocks along y cooperate on each descriptor.
    const size_t index = blockIdx.x;
    if (index >= count) {
        return;
    }
    const size_t first = (size_t) blockIdx.y * blockDim.x + threadIdx.x;
    const size_t stride = (size_t) gridDim.y * blockDim.x;
    const auto copy = copies[index];
    // Use 16-, 8-, 4-, or 2-byte copies when both pointers meet that alignment; threads copy separate chunks.
    const uintptr_t alignment = (uintptr_t) copy.src | (uintptr_t) copy.dst;
    size_t tail = 0;
    if (alignment % 16 == 0) {
        for (size_t i = first; i < copy.size / 16; i += stride) {
            ggml_cuda_memcpy_1<16>(copy.dst + 16*i, copy.src + 16*i);
        }
        tail = copy.size / 16 * 16;
    } else if (alignment % 8 == 0) {
        for (size_t i = first; i < copy.size / 8; i += stride) {
            ggml_cuda_memcpy_1<8>(copy.dst + 8*i, copy.src + 8*i);
        }
        tail = copy.size / 8 * 8;
    } else if (alignment % 4 == 0) {
        for (size_t i = first; i < copy.size / 4; i += stride) {
            ggml_cuda_memcpy_1<4>(copy.dst + 4*i, copy.src + 4*i);
        }
        tail = copy.size / 4 * 4;
    } else if (alignment % 2 == 0) {
        for (size_t i = first; i < copy.size / 2; i += stride) {
            ggml_cuda_memcpy_1<2>(copy.dst + 2*i, copy.src + 2*i);
        }
        tail = copy.size / 2 * 2;
    }
    // Copy trailing bytes, or the entire range when no wider alignment is available.
    for (size_t i = tail + first; i < copy.size; i += stride) {
        ggml_cuda_memcpy_1<1>(copy.dst + i, copy.src + i);
    }
}

struct ggml_cuda_batch_copy_mapping {
    void * host = nullptr;
    unsigned char * device = nullptr;

    ggml_cuda_batch_copy_mapping(void * host, size_t size) {
        const cudaError_t err = cudaHostRegister(host, size, cudaHostRegisterMapped | cudaHostRegisterPortable);
        if (err != cudaSuccess) {
            (void) cudaGetLastError();
            GGML_LOG_DEBUG("%s: failed to register %.2f MiB of pinned memory: %s\n", __func__,
                           size / 1024.0 / 1024.0, cudaGetErrorString(err));
            return;
        }
        this->host = host;
        if (cudaHostGetDevicePointer((void **) &device, host, 0) != cudaSuccess) {
            (void) cudaGetLastError();
            device = nullptr;
            return;
        }
    }

    ggml_cuda_batch_copy_mapping(const ggml_cuda_batch_copy_mapping &) = delete;
    ggml_cuda_batch_copy_mapping & operator=(const ggml_cuda_batch_copy_mapping &) = delete;

    ~ggml_cuda_batch_copy_mapping() {
        if (host) {
            // Release only the registration acquired by this object.
            CUDA_CHECK(cudaHostUnregister(host));
        }
    }
};

// A scratch on each physical device for batched copies. It is made at the first batch and kept.
struct ggml_cuda_batch_stage {
    static constexpr size_t capacity   = ggml_cuda_tensor_n_stage_bytes;
    static constexpr size_t max_copies = 16*1024;

    std::mutex mutex;
    unsigned char * data = nullptr;
    ggml_cuda_batch_copy_descriptor * copies = nullptr;
    bool failed = false;
};

static ggml_cuda_batch_stage & ggml_cuda_batch_stage_for(int device) {
    static ggml_cuda_batch_stage stages[GGML_CUDA_MAX_DEVICES];
    return stages[ggml_cuda_info().devices[device].physical_device];
}

// write: the host ranges go to the scratch as one copy per run of adjacent ranges, and a kernel scatters them to the tensors.
// Otherwise a kernel gathers the tensor ranges into the scratch, and they go to host memory the same way.
template <bool write>
static bool ggml_cuda_tensor_n_staged(int device,
        const ggml_tensor * const * tensors, std::conditional_t<write, const void *, void *> const * data,
        const size_t * offsets, const size_t * sizes, size_t n_tensors) {
    constexpr size_t min_copies = 4;
    constexpr size_t max_size = ggml_cuda_batch_stage::capacity / 2;

    // the ranges that are written: a batch writes them in no order, so they must not overlap
    std::vector<std::pair<uintptr_t, uintptr_t>> ranges;
    ranges.reserve(n_tensors);
    size_t n_staged = 0;
    bool sorted = true;
    for (size_t i = 0; i < n_tensors; ++i) {
        if (sizes[i] == 0) {
            continue;
        }
        const uintptr_t address = write ? (uintptr_t) tensors[i]->data + offsets[i] : (uintptr_t) data[i];
        GGML_ASSERT(address <= UINTPTR_MAX - sizes[i]);
        sorted = sorted && (ranges.empty() || ranges.back().second <= address);
        ranges.emplace_back(address, address + sizes[i]);
        n_staged += sizes[i] <= max_size;
    }
    if (n_staged < min_copies) {
        return false;
    }
    if (!sorted) {
        std::sort(ranges.begin(), ranges.end());
        for (size_t i = 1; i < ranges.size(); ++i) {
            if (ranges[i].first < ranges[i - 1].second) {
                return false;
            }
        }
    }

    auto & stage = ggml_cuda_batch_stage_for(device);
    std::lock_guard<std::mutex> lock(stage.mutex);
    if (stage.failed) {
        return false;
    }
    if (stage.data == nullptr) {
        void * scratch = nullptr;
        void * copies = nullptr;
        if (cudaMalloc(&scratch, ggml_cuda_batch_stage::capacity) != cudaSuccess ||
                cudaMalloc(&copies, ggml_cuda_batch_stage::max_copies * sizeof(ggml_cuda_batch_copy_descriptor)) != cudaSuccess) {
            (void) cudaGetLastError();
            if (scratch) {
                CUDA_CHECK(cudaFree(scratch));
            }
            stage.failed = true;
            return false;
        }
        stage.data = (unsigned char *) scratch;
        stage.copies = (ggml_cuda_batch_copy_descriptor *) copies;
    }

    // a run of ranges that are adjacent in host memory, in the order they have in the scratch
    struct span {
        std::conditional_t<write, const unsigned char *, unsigned char *> host;
        size_t at;
        size_t size;
    };
    std::vector<ggml_cuda_batch_copy_descriptor> copies;
    std::vector<span> spans;
    size_t used = 0;
    [[maybe_unused]] const unsigned char * last_host = nullptr;
    const size_t nsm = ggml_cuda_info().devices[device].nsm;
    const cudaStream_t stream = cudaStreamPerThread;

    const auto flush = [&]() {
        if (copies.empty()) {
            return;
        }
        // a copy from pageable memory returns once it has read the memory, so the descriptors can be filled again
        CUDA_CHECK(cudaMemcpyAsync(stage.copies, copies.data(), copies.size() * sizeof(copies[0]), cudaMemcpyHostToDevice, stream));
        if constexpr (write) {
            for (const auto & s : spans) {
                CUDA_CHECK(cudaMemcpyAsync(stage.data + s.at, s.host, s.size, cudaMemcpyHostToDevice, stream));
            }
        }
        const size_t blocks_per_copy = std::min<size_t>(UINT16_MAX, std::max<size_t>(1, (nsm + copies.size() - 1) / copies.size()));
        const dim3 blocks((unsigned int) copies.size(), (unsigned int) blocks_per_copy);
        ggml_cuda_batch_copy_kernel<<<blocks, 256, 0, stream>>>(stage.copies, copies.size());
        CUDA_CHECK(cudaGetLastError());
        if constexpr (!write) {
            for (const auto & s : spans) {
                CUDA_CHECK(cudaMemcpyAsync(s.host, stage.data + s.at, s.size, cudaMemcpyDeviceToHost, stream));
            }
        }
        // the scratch is free again, and a copy of another stream can go before the next part
        CUDA_CHECK(cudaStreamSynchronize(stream));
        copies.clear();
        spans.clear();
        used = 0;
    };

    for (size_t i = 0; i < n_tensors; ++i) {
        if (sizes[i] == 0) {
            continue;
        }
        unsigned char * tensor = (unsigned char *) tensors[i]->data + offsets[i];
        if (sizes[i] > max_size) {
            if constexpr (write) {
                CUDA_CHECK(cudaMemcpyAsync(tensor, data[i], sizes[i], cudaMemcpyHostToDevice, stream));
            } else {
                CUDA_CHECK(cudaMemcpyAsync(data[i], tensor, sizes[i], cudaMemcpyDeviceToHost, stream));
            }
            continue;
        }
        if (used + sizes[i] > ggml_cuda_batch_stage::capacity || copies.size() == ggml_cuda_batch_stage::max_copies) {
            flush();
        }
        const auto host = (decltype(span::host)) data[i];
        if constexpr (write) {
            // the bytes of the range before, for another tensor: they are in the scratch (a tensor mirrored on two devices that are one card)
            if (!copies.empty() && host == last_host && sizes[i] == copies.back().size) {
                copies.push_back({copies.back().src, tensor, sizes[i]});
                continue;
            }
            last_host = host;
        }
        if (!spans.empty() && spans.back().host + spans.back().size == host) {
            spans.back().size += sizes[i];
        } else {
            spans.push_back({host, used, sizes[i]});
        }
        if constexpr (write) {
            copies.push_back({stage.data + used, tensor, sizes[i]});
        } else {
            copies.push_back({tensor, stage.data + used, sizes[i]});
        }
        used += sizes[i];
    }
    flush();
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return true;
}
#endif

bool ggml_cuda_get_tensor_n_staged(int device,
        const ggml_tensor * const * tensors, void * const * data, const size_t * offsets,
        const size_t * sizes, size_t n_tensors) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    GGML_UNUSED(device);
    GGML_UNUSED(tensors);
    GGML_UNUSED(data);
    GGML_UNUSED(offsets);
    GGML_UNUSED(sizes);
    GGML_UNUSED(n_tensors);
    return false;
#else
    return ggml_cuda_tensor_n_staged<false>(device, tensors, data, offsets, sizes, n_tensors);
#endif
}

bool ggml_cuda_set_tensor_n_staged(int device,
        const ggml_tensor * const * tensors, const void * const * data, const size_t * offsets,
        const size_t * sizes, size_t n_tensors) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    GGML_UNUSED(device);
    GGML_UNUSED(tensors);
    GGML_UNUSED(data);
    GGML_UNUSED(offsets);
    GGML_UNUSED(sizes);
    GGML_UNUSED(n_tensors);
    return false;
#else
    return ggml_cuda_tensor_n_staged<true>(device, tensors, data, offsets, sizes, n_tensors);
#endif
}

static bool ggml_cuda_copy_tensor_n(int device, cudaMemcpyKind direction,
        const ggml_tensor * const * tensors, const void * const * data, const size_t * offsets,
        const size_t * sizes, size_t n_tensors) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    GGML_UNUSED(device);
    GGML_UNUSED(direction);
    GGML_UNUSED(tensors);
    GGML_UNUSED(data);
    GGML_UNUSED(offsets);
    GGML_UNUSED(sizes);
    GGML_UNUSED(n_tensors);
    return false;
#else
    const bool to_host = direction == cudaMemcpyDeviceToHost;
    // Provisional dispatch values based on experiments.
    constexpr size_t min_batch_reads = 128;
    constexpr size_t max_kernel_read_bytes = 128 * 1024;
    // Collect destination ranges; repeated sources are allowed.
    std::vector<std::pair<uintptr_t, uintptr_t>> ranges;
    ranges.reserve(n_tensors);
    size_t payload_bytes = 0;
    uintptr_t begin = UINTPTR_MAX;
    uintptr_t end = 0;
    // Number of non-empty reads whose size is at most max_kernel_read_bytes.
    size_t n_reads_small = 0;
    for (size_t i = 0; i < n_tensors; ++i) {
        if (sizes[i] == 0) {
            continue;
        }
        const uintptr_t address = (uintptr_t) data[i];
        GGML_ASSERT(address <= UINTPTR_MAX - sizes[i]);
        GGML_ASSERT(payload_bytes <= SIZE_MAX - sizes[i]);
        payload_bytes += sizes[i];
        n_reads_small += sizes[i] <= max_kernel_read_bytes;
        begin = std::min(begin, address);
        end = std::max(end, address + sizes[i]);
        const uintptr_t destination = to_host ? address : (uintptr_t) tensors[i]->data + offsets[i];
        GGML_ASSERT(destination <= UINTPTR_MAX - sizes[i]);
        ranges.emplace_back(destination, destination + sizes[i]);
    }
    if (ranges.empty()) {
        return true;
    }
    // Total number of non-empty reads.
    const size_t n_reads = ranges.size();
    if (n_reads > SIZE_MAX / sizeof(ggml_cuda_batch_copy_descriptor) || n_reads > INT_MAX) {
        return false;
    }
    if (n_reads < 2 || (n_reads < min_batch_reads && n_reads_small == n_reads)) {
        return false;
    }
    std::sort(ranges.begin(), ranges.end());
    for (size_t i = 1; i < ranges.size(); ++i) {
        // Preserve the ordered copy path when destinations overlap.
        if (ranges[i].first < ranges[i - 1].second) {
            return false;
        }
    }
    // Limit gaps in the CPU span without underflow when sources overlap.
    if (end - begin > payload_bytes && end - begin - payload_bytes > payload_bytes) {
        return false;
    }
    // Register and map the CPU span for direct kernel access.
    ggml_cuda_batch_copy_mapping mapping((void *) begin, end - begin);
    if (!mapping.host) {
        return false;
    }
    // Reads selected for the kernel; zero means all reads use DMA.
    size_t n_reads_kernel = mapping.device && n_reads_small >= min_batch_reads ? n_reads_small : 0;
    // Keep descriptor storage alive until the final stream synchronization.
    constexpr size_t max_batch_descriptors = 128 * 1024;
    const size_t bytes = std::min(n_reads_kernel, max_batch_descriptors) * sizeof(ggml_cuda_batch_copy_descriptor);
    ggml_backend_buffer_ptr descriptors(n_reads_kernel ? ggml_backend_buft_alloc_buffer(ggml_backend_cuda_buffer_type(device), bytes) : nullptr);
    if (!descriptors) {
        n_reads_kernel = 0;
    }
    const auto host_type = ggml_backend_cuda_host_buffer_type();
    ggml_backend_buffer_ptr host_descriptors(n_reads_kernel ? ggml_backend_buft_alloc_buffer(host_type, bytes) : nullptr);
    if (!host_descriptors || ggml_backend_buffer_get_type(host_descriptors.get()) != host_type) {
        n_reads_kernel = 0;
    }
    if (n_reads_kernel) {
        auto * copies = static_cast<ggml_cuda_batch_copy_descriptor *>(ggml_backend_buffer_get_base(host_descriptors.get()));
        auto * device_copies = static_cast<ggml_cuda_batch_copy_descriptor *>(ggml_backend_buffer_get_base(descriptors.get()));
        const int threads = 256;
        const size_t nsm = ggml_cuda_info().devices[device].nsm;
        size_t i = 0;
        for (size_t first = 0; first < n_reads_kernel; first += max_batch_descriptors) {
            const size_t n_batch = std::min(n_reads_kernel - first, max_batch_descriptors);
            size_t count = 0;
            for (; i < n_tensors && count < n_batch; ++i) {
                if (sizes[i] == 0 || sizes[i] > max_kernel_read_bytes) {
                    continue;
                }
                auto * host = mapping.device + ((uintptr_t) data[i] - begin);
                auto * tensor = (unsigned char *) tensors[i]->data + offsets[i];
                copies[count++] = {to_host ? tensor : host, to_host ? host : tensor, sizes[i]};
            }
            const size_t blocks_per_copy = std::min<size_t>(UINT16_MAX, std::max<size_t>(1, (nsm + n_batch - 1) / n_batch));
            const dim3 blocks((unsigned int) n_batch, (unsigned int) blocks_per_copy);
            // Stream ordering keeps each kernel ahead of the next descriptor overwrite.
            CUDA_CHECK(cudaMemcpyAsync(device_copies, copies, n_batch * sizeof(copies[0]), cudaMemcpyHostToDevice, cudaStreamPerThread));
            ggml_cuda_batch_copy_kernel<<<blocks, threads, 0, cudaStreamPerThread>>>(device_copies, n_batch);
            CUDA_CHECK(cudaGetLastError());
            if (first + n_batch < n_reads_kernel) {
                // Wait before refilling the pinned host descriptor buffer.
                CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
            }
        }
    }
    // Keep the CPU span pinned for DMA too; overlapping destinations already declined above.
    for (size_t i = 0; i < n_tensors; ++i) {
        if (sizes[i] && (!n_reads_kernel || sizes[i] > max_kernel_read_bytes)) {
            void * host = const_cast<void *>(data[i]);
            void * tensor = (char *) tensors[i]->data + offsets[i];
            CUDA_CHECK(cudaMemcpyAsync(to_host ? host : tensor, to_host ? tensor : host, sizes[i], direction, cudaStreamPerThread));
        }
    }
    // Wait for transfers before releasing the CPU mapping and descriptor buffers.
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
    return true;
#endif
}

bool ggml_cuda_get_tensor_n(int device,
        const ggml_tensor * const * tensors, void * const * data, const size_t * offsets,
        const size_t * sizes, size_t n_tensors) {
    return ggml_cuda_copy_tensor_n(device, cudaMemcpyDeviceToHost, tensors, data, offsets, sizes, n_tensors);
}

bool ggml_cuda_set_tensor_n(int device,
        ggml_tensor * const * tensors, const void * const * data, const size_t * offsets,
        const size_t * sizes, size_t n_tensors) {
    return ggml_cuda_copy_tensor_n(device, cudaMemcpyHostToDevice, tensors, data, offsets, sizes, n_tensors);
}
