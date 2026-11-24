#pragma once

#include "common.cuh"

// The stream a decode graph's paced L2 issuer runs on, beside the graph's own (ggml_cuda_l2_issue); the concurrent-stream
// regions take 1 up.
#define GGML_CUDA_L2_ISSUE_STREAM (GGML_CUDA_MAX_STREAMS - 1)

// Byte ranges of weights to bring into L2 ahead of the kernels that read them, laid end to end in the order they are read.
// Each range a multiple of 16 bytes from a 16-byte aligned start. A hint: no result depends on it.
struct ggml_cuda_l2_ranges {
    static constexpr int max_ranges = 32;

    const void * ptr[max_ranges]   = {};
    int64_t      bytes[max_ranges] = {};
    int          n                 = 0;
    int64_t      total             = 0; // the ranges' bytes together

    // [p, p + nbytes) shrunk to whole 16-byte units inside it, merged into the last range when it continues it; false
    // when the ranges are full
    bool add(const void * p, int64_t nbytes) {
        const uintptr_t b = ((uintptr_t) p + 15) & ~(uintptr_t) 15;
        const uintptr_t e = ((uintptr_t) p + nbytes) & ~(uintptr_t) 15;
        if (e <= b) {
            return true;
        }
        if (n > 0 && (uintptr_t) ptr[n - 1] + bytes[n - 1] == b) {
            bytes[n - 1] += e - b;
        } else if (n < max_ranges) {
            ptr[n]   = (const void *) b;
            bytes[n] = e - b;
            ++n;
        } else {
            return false;
        }
        total += e - b;
        return true;
    }

    // whether [p, p + 1) lies in a range already
    bool covers(const void * p) const {
        for (int r = 0; r < n; ++r) {
            if ((const char *) p >= (const char *) ptr[r] - 15 && (const char *) p < (const char *) ptr[r] + bytes[r]) {
                return true;
            }
        }
        return false;
    }
};

// Requests r into L2 at rate_gbs on stream: a burst of bulk prefetches beyond what DRAM serves at once is dropped (a
// 17.8 MB matrix prefetched at once from 70 SMs of an RTX 5070 Ti lands ~4 MB of it, paced at up to 1,000 GB/s all of it),
// so the kernel spaces its requests out in time. Runs until the last range is requested, total / rate_gbs, or until
// stop (ggml_cuda_l2_issue_stop, or nullptr) changes from its value when the kernel started; nothing waits on it but the
// end of the graph. NVIDIA from sm_90 (cp.async.bulk.prefetch.L2); nothing elsewhere.
void ggml_cuda_l2_issue(const ggml_cuda_l2_ranges & r, double rate_gbs, int nsm, const unsigned int * stop,
                        cudaStream_t stream);

// Routed-expert L2 warming through cp.async.bulk.prefetch.tensor (no SMEM destination, no mbarrier, no evict_last).
// Reads routed expert ids on the device, deduplicates in ascending local expert order, and warms tiles in consumer
// wave order up to dose_mib, paced at rate_gbs and stopped when *stop changes.
struct ggml_cuda_l2_expert_warm_args {
    int                  device        = 0;
    const ggml_tensor *  weight        = nullptr;
    const ggml_tensor *  ids           = nullptr;
    int32_t              first         = 0;
    int32_t              count         = 0;
    bool                 paired_halves = false;
    double               dose_mib      = 24.0;
    double               rate_gbs      = 0.0;
    const unsigned int * stop          = nullptr;
};

// Returns true if the tensor-prefetch warm kernel supports this device, weight layout, and ids shape.
bool ggml_cuda_l2_warm_experts_supported(const ggml_cuda_l2_expert_warm_args & args);

// Returns true if the tensor-prefetch warm kernel was launched on stream; false if the arch or shape is unsupported.
bool ggml_cuda_l2_warm_experts(const ggml_cuda_l2_expert_warm_args & args, cudaStream_t stream);
