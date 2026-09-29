#include "l2-issue.cuh"

#include <algorithm>
#include <cmath>

// A request is an 8 KB piece: one uniform-datapath instruction a piece, and inside what the rig measured; pieces go to the
// blocks in turn, each block's one thread requesting its next when the grid's position in the ranges is due at the rate.
#define L2_ISSUE_PIECE 8192

static __device__ __forceinline__ uint64_t l2_issue_now() {
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

static __global__ void l2_issue_paced(const ggml_cuda_l2_ranges r, const float ns_per_byte) {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_HOPPER
    const uint64_t t0   = l2_issue_now();
    const int64_t  step = (int64_t) gridDim.x * L2_ISSUE_PIECE;
    int            k    = 0; // the range holding lo
    int64_t        base = 0; // its first byte's place in the ranges laid end to end
    for (int64_t off = (int64_t) blockIdx.x * L2_ISSUE_PIECE; off < r.total; off += step) {
        const uint64_t due = (uint64_t) ((float) off * ns_per_byte);
        while (l2_issue_now() - t0 < due) {
        }
        const int64_t hi = min(off + (int64_t) L2_ISSUE_PIECE, r.total);
        for (int64_t lo = off; lo < hi;) {
            while (lo >= base + r.bytes[k]) {
                base += r.bytes[k];
                ++k;
            }
            const int64_t e = min(hi, base + r.bytes[k]);
            asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;"
                         :: "l"((uint64_t) ((const char *) r.ptr[k] + (lo - base))), "r"((uint32_t) (e - lo)) : "memory");
            lo = e;
        }
    }
#else
    GGML_UNUSED_VARS(r, ns_per_byte);
#endif
}

void ggml_cuda_l2_issue(const ggml_cuda_l2_ranges & r, const double rate_gbs, const int nsm, cudaStream_t stream) {
    if (r.total <= 0 || rate_gbs <= 0.0) {
        return;
    }
    // a thread requests ~66 GB/s of bulk prefetches at most (a rig on an RTX 5080), so one block for each 40 GB/s of the
    // rate, and no more blocks than pieces
    const int64_t pieces = (r.total + L2_ISSUE_PIECE - 1) / L2_ISSUE_PIECE;
    const int     nb     = (int) std::min<int64_t>({ (int64_t) nsm, pieces, (int64_t) std::ceil(rate_gbs / 40.0) });
    l2_issue_paced<<<std::max(nb, 1), 1, 0, stream>>>(r, (float) (1.0 / rate_gbs));
    CUDA_CHECK(cudaGetLastError());
}
