#include "prefill-sp.h"

#include "common.cuh"

#include <mutex>
#include <vector>

// One warp a block: lane i handles qs[i], and the warp reduces amax and the sum as quantize_mmq_q8_1 reduces one block
// (fmaxf over the block's values is order-free; the integer sum is associative), so q and d are the exact bits the
// MoE's own quantizer would produce from the same F32 values.
static __global__ void prefill_sp_q8_1_quantize_kernel(
        const float * __restrict__ x, void * __restrict__ vy, const int64_t nblocks) {
    const int64_t tid = (int64_t) blockDim.x*blockIdx.x + threadIdx.x;
    const int64_t ib  = tid / QK8_1;
    const int     iqs = (int) (tid % QK8_1);
    if (ib >= nblocks) {
        return;
    }
    block_q8_1 * y = (block_q8_1 *) vy;

    const float xi = x[ib*QK8_1 + iqs];
    const float amax = warp_reduce_max<QK8_1>(fabsf(xi));
    // quantize_mmq_q8_1's arithmetic: 127/amax, roundf(x*d_inv), d = 1/d_inv, so dequantize then requantize is the
    // identity on (q, d). An all-zero block gives d = 0 (the consumer's d is 0 there too, whatever its q rounds to).
    const float d_inv = amax == 0.0f ? 0.0f : 127.0f / amax;
    const int8_t q = (int8_t) roundf(xi*d_inv);
    const int sum = warp_reduce_sum<QK8_1>((int) q);

    y[ib].qs[iqs] = q;
    if (iqs == 0) {
        const float d = d_inv == 0.0f ? 0.0f : 1.0f / d_inv;
        y[ib].ds = make_half2(d, d*sum);
    }
}

static __global__ void prefill_sp_q8_1_dequantize_kernel(
        const void * __restrict__ vx, float * __restrict__ y, const int64_t nblocks) {
    const int64_t tid = (int64_t) blockDim.x*blockIdx.x + threadIdx.x;
    const int64_t ib  = tid / QK8_1;
    const int     iqs = (int) (tid % QK8_1);
    if (ib >= nblocks) {
        return;
    }
    const block_q8_1 * x = (const block_q8_1 *) vx + ib;
    y[ib*QK8_1 + iqs] = (float) x->qs[iqs] * __half2float(x->ds.x);
}

static constexpr int PREFILL_SP_BLOCK_SIZE = 256;
static_assert(PREFILL_SP_BLOCK_SIZE % QK8_1 == 0, "a warp must not straddle two q8_1 blocks");

void ggml_cuda_prefill_sp_q8_1_quantize(cudaStream_t stream, const float * src, void * dst, const int64_t ne) {
    GGML_ASSERT(ne % QK8_1 == 0);
    const int64_t nblocks = ne / QK8_1;
    if (nblocks == 0) {
        return;
    }
    const int64_t grid = (nblocks*QK8_1 + PREFILL_SP_BLOCK_SIZE - 1) / PREFILL_SP_BLOCK_SIZE;
    prefill_sp_q8_1_quantize_kernel<<<(unsigned) grid, PREFILL_SP_BLOCK_SIZE, 0, stream>>>(src, dst, nblocks);
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_prefill_sp_q8_1_dequantize(cudaStream_t stream, const void * src, float * dst, const int64_t ne) {
    GGML_ASSERT(ne % QK8_1 == 0);
    const int64_t nblocks = ne / QK8_1;
    if (nblocks == 0) {
        return;
    }
    const int64_t grid = (nblocks*QK8_1 + PREFILL_SP_BLOCK_SIZE - 1) / PREFILL_SP_BLOCK_SIZE;
    prefill_sp_q8_1_dequantize_kernel<<<(unsigned) grid, PREFILL_SP_BLOCK_SIZE, 0, stream>>>(src, dst, nblocks);
    CUDA_CHECK(cudaGetLastError());
}

size_t ggml_cuda_prefill_sp_q8_1_bytes(const int64_t ne) {
    GGML_ASSERT(ne % QK8_1 == 0);
    return (size_t) (ne / QK8_1) * sizeof(block_q8_1);
}

namespace {

struct prefill_sp_staging_entry {
    int device = -1;
    int rank = 0;
    size_t bytes = 0;
    void * ptr = nullptr;
};

static std::mutex & prefill_sp_staging_mutex() {
    static std::mutex m;
    return m;
}

struct prefill_sp_staging_store {
    std::vector<prefill_sp_staging_entry> active;
    std::vector<std::pair<int, void *>>   retired; // kept alive so prior captured CUDA graphs never dangle

    ~prefill_sp_staging_store() {
        for (const auto & e : active) {
            if (e.ptr) {
                cudaSetDevice(e.device);
                cudaFree(e.ptr);
            }
        }
        for (const auto & r : retired) {
            if (r.second) {
                cudaSetDevice(r.first);
                cudaFree(r.second);
            }
        }
    }
};

static prefill_sp_staging_store & prefill_sp_staging_cache() {
    static prefill_sp_staging_store store;
    return store;
}

} // namespace

void * ggml_cuda_prefill_sp_q8_1_staging(const int device, const int64_t ne, cudaStream_t stream, const int rank) {
    const size_t bytes = ggml_cuda_prefill_sp_q8_1_bytes(ne);

    std::lock_guard<std::mutex> lock(prefill_sp_staging_mutex());
    auto & store = prefill_sp_staging_cache();
    for (auto & entry : store.active) {
        if (entry.device == device && entry.rank == rank && entry.ptr != nullptr && entry.bytes >= bytes) {
            return entry.ptr;
        }
    }

    // inside a captured step nothing may allocate: the caller keeps the stock F32 gather
    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(stream, &status) != cudaSuccess || status != cudaStreamCaptureStatusNone) {
        return nullptr;
    }

    ggml_cuda_set_device(device);
    for (auto & entry : store.active) {
        if (entry.device != device || entry.rank != rank) {
            continue;
        }
        void * next_ptr = nullptr;
        if (cudaMalloc(&next_ptr, bytes) != cudaSuccess) {
            return nullptr;
        }
        if (entry.ptr != nullptr) {
            store.retired.emplace_back(entry.device, entry.ptr);
        }
        entry.ptr = next_ptr;
        entry.bytes = bytes;
        return entry.ptr;
    }

    prefill_sp_staging_entry entry;
    entry.device = device;
    entry.rank = rank;
    if (cudaMalloc(&entry.ptr, bytes) != cudaSuccess) {
        entry.ptr = nullptr;
        return nullptr;
    }
    entry.bytes = bytes;
    store.active.push_back(entry);
    return entry.ptr;
}
