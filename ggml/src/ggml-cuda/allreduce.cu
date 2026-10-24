#include "allreduce.cuh"

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

#include "convert.cuh"
#include "ggml-impl.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

// ---------------------------------------------------------------------------
// CUDA AllReduce for tensor-parallel inference across two GPUs.
//
// Provides an in-place sum reduction over matching tensors on two CUDA
// devices in the same process.  Used by the tensor-split path alongside
// NCCL; targets setups without NVLink, where data is exchanged between the
// GPUs by staging it through pinned host memory over PCIe.
//
// Two reduction strategies are selected per call by tensor size:
//
//   * Chunked kernel path (small reductions): a single CUDA kernel both
//     stages data through pinned host memory and performs the local sum.
//     Cross-GPU synchronization happens *inside the kernel* (busy-wait on
//     a host-memory flag), which keeps launch overhead low for the
//     latency-sensitive token-generation case.
//
//   * Copy-engine path (large reductions): the transfer is split into
//     D2H + H2D cudaMemcpyAsync chunks driven by the GPU's copy engine,
//     followed by a small device-side add kernel.  Cross-GPU
//     synchronization happens *outside the kernel*, via CUDA events
//     between streams.  This keeps the compute engine free while large
//     transfers are in flight, which matters for prefill-sized tensors.
//     Reductions larger than the per-call inner cap are processed by an
//     outer chunker that issues sequential inner calls.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Cross-GPU signal mechanism
//
// One int per (slot, rank) pair in pinned host memory.  Each AR call writes a
// strictly increasing token (= the chunked kernel's call number, counted on the
// device: ggml_cuda_ar_kernel) into its own arrival int.
// The peer spins until its read of the other's arrival int equals the token
// it expects for this call -- a mismatch means the peer hasn't arrived yet.
// Tokens never repeat over realistic call rates (32-bit int wraps in tens of
// days at thousands of ARs/sec), so arrival ints don't need to be reset
// between calls; we initialize once at pipeline init and let the values
// accumulate.
//
// There is exactly one writer (the owning GPU) and one reader (the peer), so
// we don't need atomics.  A volatile store paired with __threadfence_system()
// provides the release ordering that makes the D2H writes visible system-wide
// before the arrival token is observed.
//
// atomicAdd_system() requires hostNativeAtomicSupported, which is unavailable
// on PCIe-attached consumer GPUs without NVLink, so the volatile path is the
// portable choice.
// ---------------------------------------------------------------------------

static __device__ __forceinline__ void ggml_cuda_ar_signal_set(int * p, int token) {
    *(volatile int *)p = token;
}
static __device__ __forceinline__ int ggml_cuda_ar_signal_get(const int * p) {
    return *(const volatile int *)p;
}

// Byte spacing between adjacent arrival ints.  64 bytes (one cache line)
// ensures each GPU/block's arrival slot lives on its own line, preventing
// false-sharing stalls on the polling GPU.
static constexpr size_t GGML_CUDA_AR_ARRIVAL_STRIDE = 64;

// Number of blocks the chunked kernel launches with.  Each block stripes a
// disjoint slice of the data and synchronizes through its own arrival-token
// slot so multiple SMs can pump PCIe stores in parallel.
static constexpr int GGML_CUDA_AR_KERNEL_BLOCKS = 8;

// A device's value and its peer's: summed in float, each rounded through T_wire first so both devices get the same
// bits; 32-bit words (ggml_cuda_ar_allgather) are ORed, which keeps every bit of a part the peer has zeros for.
template <typename T_dst, typename T_wire>
static __device__ __forceinline__ T_dst ggml_cuda_ar_combine(const T_dst mine, const T_wire other) {
    if constexpr (std::is_same_v<T_dst, uint32_t>) {
        static_assert(std::is_same_v<T_wire, uint32_t>, "words go on the wire as words");
        return mine | other;
    } else {
        const T_wire d_low = ggml_cuda_cast<T_wire>(mine);
        return ggml_cuda_cast<T_dst>(ggml_cuda_cast<float>(d_low) + ggml_cuda_cast<float>(other));
    }
}

// A zero of T_wire: the LL kernel's padding past the last element.
template <typename T_wire>
static __device__ __forceinline__ T_wire ggml_cuda_ar_wire_zero() {
    if constexpr (std::is_same_v<T_wire, uint32_t>) {
        return 0u;
    } else {
        return ggml_cuda_cast<T_wire>(0.0f);
    }
}

// ---------------------------------------------------------------------------
// Chunked kernel AllReduce -- 2 GPUs, supports float, half, and bfloat16.
//
// Both GPUs run this kernel simultaneously on independent streams.  sendbuf
// and recvbuf live in T_dst (the caller's tensor type); host_mine / host_other
// carry data in T_wire (the on-wire type, possibly narrower than T_dst -- e.g.
// T_dst=F32 with T_wire=BF16 halves the bytes pushed across PCIe).  When
// T_dst == T_wire the casts below are no-ops.
//
// Each GPU runs three phases:
//
//   Phase 1 (all threads): cast sendbuf (T_dst) -> T_wire and store as
//                          single-instruction-width vectors into host_mine.
//                          __threadfence_system() commits these writes to host
//                          memory.
//   Phase 2 (thread 0):    write token to arrival_mine; spin until
//                          arrival_other == token.
//   Phase 3 (all threads): read T_wire vectors from host_other, cast
//                          each element to T_dst, and sum with the local
//                          sendbuf value (also rounded through T_wire so that
//                          both GPUs truncate identically -- this guarantees
//                          bit-equivalent results across the two devices).
//
// Multi-block: blocks stripe vectors across (gridDim.x * blockDim.x) global
// threads to keep multiple SMs issuing PCIe stores in parallel.  Each block
// has its own arrival-token slot (offset by blockIdx.x * ARRIVAL_STRIDE);
// thread 0 of each block signals/spins on that slot independently of other
// blocks.  Tail elements (the leftover < ELEMS_PER_VEC at the end) are
// handled only by block 0 to avoid cross-block writes to the same slots.
// ---------------------------------------------------------------------------
//
// The call's token and slot come from the device: counter[0] holds the token of
// this device's last AllReduce (both devices start at 0 and run the same
// sequence of calls, so both reach the same token), every block reads it before
// any block can finish, and the last block to finish publishes the new token
// (counter[1] counts the finished blocks).  The token is monotonic and never
// reset, so a replayed CUDA graph that captured the kernel keeps advancing it.
// The slot is the token's parity: a device reaching call N has seen its peer
// arrive at N-1, so the peer has finished call N-2 and its reads of slot N%2.
template <typename T_dst, typename T_wire>
static __global__ void ggml_cuda_ar_kernel(
        const T_dst  *              sendbuf,
        T_dst        *              recvbuf,
        T_wire       * __restrict__ host_mine,  // slot 0; slot 1 follows at slot_elems
        const T_wire * __restrict__ host_other, // slot 0 of the peer's ring
        size_t                      slot_elems,
        int                         count,
        int *                       arrival,    // the arrival ring
        int                         rank,
        int *                       counter) {

    // Vector unit for the wire type, sized to the arch's widest single-instruction
    // copy (16 B on Volta+).  Each phase-1 iter writes one vector to host memory;
    // each phase-3 iter reads one and produces ELEMS_PER_VEC sums.
    constexpr int ELEMS_PER_VEC = ggml_cuda_get_max_cpy_bytes() / sizeof(T_wire);
    constexpr int ARRIVAL_INTS  = (int)(GGML_CUDA_AR_ARRIVAL_STRIDE / sizeof(int));

    const int tid       = threadIdx.x;
    const int nt        = blockDim.x;
    const int bid       = blockIdx.x;
    const int gtid      = bid * nt + tid;
    const int gnt       = gridDim.x * nt;
    const int count_vec = count / ELEMS_PER_VEC;
    const int tail      = count_vec * ELEMS_PER_VEC;

    __shared__ int token_shared;
    if (tid == 0) {
        token_shared = ggml_cuda_ar_signal_get(counter) + 1;
    }
    __syncthreads();
    const int token = token_shared;
    const int slot  = token & 1;
    host_mine  += slot * slot_elems;
    host_other += slot * slot_elems;
    int       * arrival_mine  = arrival + (size_t) (slot*2 + rank)     * GGML_CUDA_AR_KERNEL_BLOCKS * ARRIVAL_INTS;
    const int * arrival_other = arrival + (size_t) (slot*2 + 1 - rank) * GGML_CUDA_AR_KERNEL_BLOCKS * ARRIVAL_INTS;

    // Phase 1: cast sendbuf (T_dst) -> host_mine (T_wire) and store as vectors.
    {
        for (int i = gtid; i < count_vec; i += gnt) {
            const int off = i * ELEMS_PER_VEC;
            T_wire wire[ELEMS_PER_VEC];
            #pragma unroll
            for (int k = 0; k < ELEMS_PER_VEC; ++k) {
                wire[k] = ggml_cuda_cast<T_wire>(sendbuf[off + k]);
            }
            ggml_cuda_memcpy_1<sizeof(wire)>(&host_mine[off], wire);
        }
        if (bid == 0 && tid < count - tail) {
            host_mine[tail + tid] = ggml_cuda_cast<T_wire>(sendbuf[tail + tid]);
        }
    }

    // Commit this block's host writes before signalling.
    __threadfence_system();
    __syncthreads();

    // Phase 2: thread 0 of each block signals on its own arrival slot, then
    // spins for the matching slot from peer.  Per-block tokens mean blocks
    // proceed independently -- no inter-block barrier needed.
    if (tid == 0) {
        int       * my_slot    = arrival_mine  + bid * ARRIVAL_INTS;
        const int * other_slot = arrival_other + bid * ARRIVAL_INTS;

        ggml_cuda_ar_signal_set(my_slot, token);
        __threadfence_system(); // make our signal visible system-wide

        while (ggml_cuda_ar_signal_get(other_slot) != token) {
#if __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
            __nanosleep(100);
#else
            NO_DEVICE_CODE;
#endif // __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
        }
    }

    __syncthreads();

    // Acquire peer's host_other writes (this block's stripe of them).
    __threadfence_system();

    // Phase 3: read peer's T_wire vector, cast both sides through T_wire for
    // bit-equivalence, sum in T_dst precision, and write back to recvbuf.
    {
        for (int i = gtid; i < count_vec; i += gnt) {
            const int off = i * ELEMS_PER_VEC;
            T_wire wire[ELEMS_PER_VEC];
            ggml_cuda_memcpy_1<sizeof(wire)>(wire, &host_other[off]);
            #pragma unroll
            for (int k = 0; k < ELEMS_PER_VEC; ++k) {
                recvbuf[off + k] = ggml_cuda_ar_combine<T_dst, T_wire>(sendbuf[off + k], wire[k]);
            }
        }
        if (bid == 0 && tid < count - tail) {
            recvbuf[tail + tid] = ggml_cuda_ar_combine<T_dst, T_wire>(sendbuf[tail + tid], host_other[tail + tid]);
        }
    }

    // The last block to finish publishes this call's token for the next call.
    if (tid == 0) {
        __threadfence();
        if (atomicAdd(&counter[1], 1) == (int) gridDim.x - 1) {
            atomicExch(&counter[1], 0);
            ggml_cuda_ar_signal_set(counter, token);
            __threadfence();
        }
    }
}

// ---------------------------------------------------------------------------
// LL (low latency) kernel: small reductions, 2 GPUs.
//
// Each 8 B packet holds 4 B of wire data and the call's token. One st.v2 puts both in host memory and the peer polls the packet
// itself, so there is no __threadfence_system, no arrival int and no second read for the data: one PCIe write and one read
// round trip in place of the chunked kernel's three. The wire carries twice the bytes, so it only pays for small tensors
// (GGML_CUDA_AR_LL_MAX_BYTES). The token comes from the device counter as in ggml_cuda_ar_kernel, and the packets live in their
// own zeroed staging: a stale packet has the token of a call two back, never this one, and a packet is overwritten only after
// the peer finished the call that read it (it arrived at the call between). One block, so the token is published by a plain
// store once every thread has read it. The sum is the chunked kernel's: both sides round through T_wire, so the two devices and
// the two kernels give the same bits.
//
// Where the cards can map each other's memory, reductions up to GGML_CUDA_AR_P2P_MAX_BYTES on the wire run the same kernel on
// an inbox in each card's own memory: a device stores its packets straight into the peer's inbox and polls its own, so the
// call is one PCIe write and no PCIe read (NCCL's P2P transport writes to the receiver the same way). The slot and token
// rules above hold for the inboxes unchanged.
// ---------------------------------------------------------------------------

static constexpr int GGML_CUDA_AR_LL_THREADS = 1024;
static constexpr int GGML_CUDA_AR_LL_UNROLL  = 4;

static __device__ __forceinline__ void ggml_cuda_ar_ll_store(uint2 * p, uint32_t data, uint32_t flag) {
    asm volatile("st.volatile.global.v2.u32 [%0], {%1, %2};" :: "l"(p), "r"(data), "r"(flag) : "memory");
}

static __device__ __forceinline__ uint2 ggml_cuda_ar_ll_load(const uint2 * p) {
    uint2 v;
    asm volatile("ld.volatile.global.v2.u32 {%0, %1}, [%2];" : "=r"(v.x), "=r"(v.y) : "l"(p) : "memory");
    return v;
}

struct ggml_cuda_ar_hc_post {
    const float * residual = nullptr;
    const float * post = nullptr;
    const float * comb = nullptr;
    float * dst = nullptr;
    int width = 0;
    int sr1 = 0, sr2 = 0, sp1 = 0, sc1 = 0, sc2 = 0, sd1 = 0, sd2 = 0;
};

template <typename T_dst, typename T_wire, bool HC_POST = false>
static __global__ void ggml_cuda_ar_ll_kernel(
        const T_dst * sendbuf,
        T_dst       * recvbuf,
        uint2       * ll_mine,       // where this device's packets go, slot 0; slot 1 follows at slot_packets
        const uint2 * ll_other,      // where the peer's packets arrive, slot 0
        int           slot_packets,
        int           count,
        int         * counter,
        ggml_cuda_ar_hc_post hc = {}) {
    constexpr int W = 4 / sizeof(T_wire);

    const int tid = threadIdx.x;
    const int gnt = blockDim.x;
    const int npk = (count + W - 1) / W;

    const int token = ggml_cuda_ar_signal_get(counter) + 1;
    ll_mine  += (size_t) (token & 1) * slot_packets;
    ll_other += (size_t) (token & 1) * slot_packets;

    for (int i = tid; i < npk; i += gnt) {
        T_wire wire[W];
        #pragma unroll
        for (int k = 0; k < W; ++k) {
            const int j = i * W + k;
            wire[k] = j < count ? ggml_cuda_cast<T_wire>(sendbuf[j]) : ggml_cuda_ar_wire_zero<T_wire>();
        }
        uint32_t data;
        memcpy(&data, wire, sizeof(data));
        ggml_cuda_ar_ll_store(ll_mine + i, data, (uint32_t) token);
    }

    // The peer's packets are read LL_UNROLL at a time so their PCIe round trips overlap.
    for (int i0 = tid; i0 < npk; i0 += GGML_CUDA_AR_LL_UNROLL * gnt) {
        uint2 v[GGML_CUDA_AR_LL_UNROLL];
        #pragma unroll
        for (int u = 0; u < GGML_CUDA_AR_LL_UNROLL; ++u) {
            if (i0 + u * gnt < npk) {
                v[u] = ggml_cuda_ar_ll_load(ll_other + i0 + u * gnt);
            }
        }
        #pragma unroll
        for (int u = 0; u < GGML_CUDA_AR_LL_UNROLL; ++u) {
            const int i = i0 + u * gnt;
            if (i < npk) {
                while (v[u].y != (uint32_t) token) {
                    v[u] = ggml_cuda_ar_ll_load(ll_other + i);
                }
                T_wire wire[W];
                memcpy(wire, &v[u].x, sizeof(v[u].x));
                #pragma unroll
                for (int k = 0; k < W; ++k) {
                    const int j = i * W + k;
                    if (j < count) {
                        const T_dst reduced = ggml_cuda_ar_combine<T_dst, T_wire>(sendbuf[j], wire[k]);
                        recvbuf[j] = reduced;
                        if constexpr (HC_POST) {
                            static_assert(std::is_same_v<T_dst, float>, "mHC consumes F32");
                            const int row = j / hc.width;
                            const int col = j % hc.width;
                            float residual[4];
#pragma unroll
                            for (int s = 0; s < 4; ++s) {
                                residual[s] = hc.residual[col + s*hc.sr1 + row*hc.sr2];
                            }
#pragma unroll
                            for (int d = 0; d < 4; ++d) {
                                float sum = __fmul_rn(reduced, hc.post[d + row*hc.sp1]);
#pragma unroll
                                for (int s = 0; s < 4; ++s) {
                                    sum = __fmaf_rn(residual[s], hc.comb[d + s*hc.sc1 + row*hc.sc2], sum);
                                }
                                hc.dst[col + d*hc.sd1 + row*hc.sd2] = sum;
                            }
                        }
                    }
                }
            }
        }
    }

    // Every thread has read the token: publish it for the next call.
    __syncthreads();
    if (tid == 0) {
        ggml_cuda_ar_signal_set(counter, token);
    }
}

// Combined load-convert-add kernel.  The peer's contribution arrives as T_src
// (which may be a lower-precision type than T_dst when the BF16 round-trip is
// active).  For bit-equivalence between the two GPUs, dst is first rounded
// through T_src's precision via ggml_cuda_cast -- peer already truncated its
// own value the same way before sending -- so both sides perform identical
// arithmetic.  When T_dst == T_src the round-trip cast is a no-op.
template <typename T_dst, typename T_src>
static __global__ void ggml_cuda_ar_add_kernel(
        T_dst       * __restrict__ dst,
        const T_src * __restrict__ src,
        int count) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int nt  = gridDim.x * blockDim.x;
    for (int i = tid; i < count; i += nt) {
        const T_src d_low = ggml_cuda_cast<T_src>(dst[i]);
        dst[i] = ggml_cuda_cast<T_dst>(
            ggml_cuda_cast<float>(d_low) + ggml_cuda_cast<float>(src[i]));
    }
}

// ---------------------------------------------------------------------------
// Pipeline structure
// ---------------------------------------------------------------------------

// Number of slots in the event / arrival ring.  Two slots is sufficient:
// lockstep guarantees the two GPUs are at most one AR (or chunk) apart, so
// slot[N%2] is always safe to reuse -- peer has already consumed slot[N%2]
// from AR N-2 by the time we get to AR N.  The chunked kernel takes its slot
// from the device (ggml_cuda_ar_kernel); for the copy-engine path,
// acquire_slot's cudaEventSynchronize on ev.ker for both devices makes that
// consumption explicit before the slot's events are recorded again.
static constexpr int GGML_CUDA_AR_POOL_SIZE = 2;

// Maximum chunk size (bytes per GPU) handled by one chunked kernel launch.
// Larger tensors are reduced by issuing multiple chunked launches.
static constexpr size_t GGML_CUDA_AR_MAX_BYTES = 1024 * 1024; // 1 MB

// Largest wire size the host LL path reduces by default; GGML_CUDA_AR_LL_MAX_BYTES overrides (0 disables this path).
static constexpr size_t GGML_CUDA_AR_LL_MAX_BYTES_DEFAULT = 16 * 1024; // 16 KB

// Largest wire size the LL kernel reduces card to card by default; GGML_CUDA_AR_P2P_MAX_BYTES overrides (0 disables).
static constexpr size_t GGML_CUDA_AR_P2P_MAX_BYTES_DEFAULT = 128 * 1024; // 128 KB

// Copy-engine path: largest tensor accepted on this path; sets host_large /
// dev_tmp allocation size.
static constexpr size_t GGML_CUDA_AR_COPY_MAX_BYTES = 32 * 1024 * 1024; // 32 MB

// AR wire size at which the copy-engine path takes over from the chunked-
// kernel path.  Override via GGML_CUDA_AR_COPY_THRESHOLD.
static constexpr size_t GGML_CUDA_AR_COPY_THRESHOLD_DEFAULT = 1024 * 1024; // 1 MB
// Per-call CE chunk-size heuristic: chunk_bytes = clamp(nbytes / 4, MIN, MAX).
// The /4 keeps ~4 chunks in flight at any moment (good D2H/H2D overlap with
// the peer); the clamps cover the cases where nbytes/4 is too small (per-
// memcpy fixed cost dominates) or too large (chunk-level pipelining stalls).
// Env var GGML_CUDA_AR_COPY_CHUNK_BYTES can override with a fixed value.
static constexpr size_t GGML_CUDA_AR_COPY_CHUNK_BYTES_HEURISTIC_MIN = 512 * 1024;       // 512 KB
static constexpr size_t GGML_CUDA_AR_COPY_CHUNK_BYTES_HEURISTIC_MAX = 2 * 1024 * 1024;  // 2 MB
// Absolute floor that an env-var override is allowed to set; this caps the
// per-slot copy-event array.  256 KB -> up to 128 chunks per 32 MB tensor.
static constexpr size_t GGML_CUDA_AR_COPY_CHUNK_BYTES_MIN = 256 * 1024;
static constexpr int GGML_CUDA_AR_COPY_MAX_CHUNKS =
    static_cast<int>((GGML_CUDA_AR_COPY_MAX_BYTES + GGML_CUDA_AR_COPY_CHUNK_BYTES_MIN - 1) /
                    GGML_CUDA_AR_COPY_CHUNK_BYTES_MIN);

struct ggml_cuda_ar_event_slot {
    cudaEvent_t app = nullptr;  // upstream computation complete
    cudaEvent_t cpy[GGML_CUDA_AR_COPY_MAX_CHUNKS] = {};  // copy-engine D2H chunks complete
    cudaEvent_t h2d = nullptr;  // copy-engine H2Ds complete (handoff AR stream -> compute stream)
    cudaEvent_t ker = nullptr;  // AllReduce kernel complete
};

// Mapped pinned host allocation: cudaHostAlloc + cudaHostGetDevicePointer
// in one place, with the host handle preserved for cudaFreeHost.  Used where
// the CPU never touches the buffer -- only the device reads/writes via the
// mapped device pointer.  Required on systems where cudaDevAttrCanUseHost-
// PointerForRegisteredMem is 0 and the host pointer can't be used as a
// device pointer.
struct ggml_cuda_ar_host_mapping {
    uint8_t * host = nullptr;   // cudaFreeHost handle; also the H-side ptr for cudaMemcpyAsync
    uint8_t * dev  = nullptr;   // device-side pointer for kernels / cudaMemset

    cudaError_t alloc(size_t bytes) {
        cudaError_t rc = cudaHostAlloc(reinterpret_cast<void **>(&host), bytes,
                                       cudaHostAllocPortable | cudaHostAllocMapped);
        if (rc != cudaSuccess) {
            host = nullptr;
            return rc;
        }
        rc = cudaHostGetDevicePointer(reinterpret_cast<void **>(&dev), host, 0);
        if (rc != cudaSuccess) {
            cudaFreeHost(host);
            host = nullptr;
            dev  = nullptr;
        }
        return rc;
    }

    void free() {
        if (host) {
            cudaFreeHost(host);
            host = nullptr;
            dev  = nullptr;
        }
    }
};

// A device's P2P inbox: device memory on that card that the peer stores its LL packets into, 2 slots. cudaMalloc when
// both ranks are one GPU, else a VMM mapping both GPUs may access (ggml_cuda_ar_p2p_alloc).
struct ggml_cuda_ar_p2p_inbox {
    uint2 * dev  = nullptr;
    size_t  size = 0;     // bytes allocated or mapped
    bool    vmm  = false; // mapped with cuMemMap
};

struct ggml_cuda_ar_pipeline {
    int      n_devices;
    int      devices[GGML_CUDA_MAX_DEVICES];
    size_t   buf_bytes;    // bytes per device in host_buf[]
    size_t   copy_bytes;   // bytes per device in host_large[] / dev_tmp[]
    size_t   copy_threshold;
    size_t   copy_chunk_bytes;
    size_t   bf16_threshold; // tensors >= this size (bytes) are reduced via FP32->BF16 round-trip; 0 disables
    size_t   ll_max_bytes;   // wire bytes up to which the host LL path reduces; 0 disables this path
    size_t   p2p_max_bytes;  // wire bytes up to which the LL kernel reduces card to card; 0 when off or unavailable
    uint64_t call_count;

    // Per-device resources.
    ggml_cuda_ar_host_mapping host_buf[GGML_CUDA_MAX_DEVICES];   // pinned staging (chunked kernel)
    ggml_cuda_ar_host_mapping ll_buf[GGML_CUDA_MAX_DEVICES];     // pinned packets (LL kernel), 2 slots
    ggml_cuda_ar_p2p_inbox    p2p[GGML_CUDA_MAX_DEVICES];        // device packets (LL kernel card to card), 2 slots
    ggml_cuda_ar_host_mapping host_large[GGML_CUDA_MAX_DEVICES]; // pinned staging (copy-engine)
    char *                    dev_tmp[GGML_CUDA_MAX_DEVICES];    // device scratch for copy-engine path
    cudaStream_t             streams[GGML_CUDA_MAX_DEVICES];   // non-blocking
    ggml_cuda_ar_event_slot  ev_pool[GGML_CUDA_MAX_DEVICES][GGML_CUDA_AR_POOL_SIZE];

    // Copy-engine: per-device "I finished reading my peer's host_large"
    // event.  Indexed by RECORDER device.  Recorded same-device on streams[i]
    // after stage 2's last H2D from host_large[peer].  Waited cross-device
    // by peer's stage-1 stream before the next AR overwrites host_large[peer].
    cudaEvent_t              host_large_read_done[GGML_CUDA_MAX_DEVICES];
    bool                     host_large_read_done_valid;

    // Copy-engine: per-device "my add_kernel is done with dev_tmp" event.
    // Recorded on the compute stream after each add_kernel; the AR stream
    // waits on it before the next copy_impl's H2D overwrites dev_tmp.  Lets us
    // single-buffer dev_tmp despite add_kernel running on a separate stream.
    cudaEvent_t              dev_tmp_kernel_done[GGML_CUDA_MAX_DEVICES];
    bool                     dev_tmp_kernel_done_valid;

    // Arrival ring: ARRIVAL_STRIDE bytes between adjacent ints, a block of
    // KERNEL_BLOCKS ints per (slot, rank) in that order (ggml_cuda_ar_kernel
    // indexes it).  Mapped pinned memory; CPU never reads/writes -- only the
    // kernel and cudaMemset.
    ggml_cuda_ar_host_mapping arrival;

    // Per-device chunked-kernel counters (device memory, 2 ints): the token of
    // the device's last AllReduce, and the blocks of the running call that have
    // finished (ggml_cuda_ar_kernel).
    int *                    counter[GGML_CUDA_MAX_DEVICES];
};

static uint64_t ggml_cuda_ar_env_u64(const char * name, uint64_t default_value) {
    const char * value = getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return default_value;
    }

    char * end = nullptr;
    const unsigned long long parsed = strtoull(value, &end, 10);
    return end != value ? (uint64_t) parsed : default_value;
}

// The copy-engine path's event slot for its next call, once both devices have
// finished the call two back.
static int ggml_cuda_ar_acquire_slot(ggml_cuda_ar_pipeline * p) {
    const int  slot        = static_cast<int>(p->call_count % GGML_CUDA_AR_POOL_SIZE);
    const bool pool_lapped = p->call_count >= GGML_CUDA_AR_POOL_SIZE;
    p->call_count++;

    if (pool_lapped) {
        for (int i = 0; i < p->n_devices; ++i) {
            ggml_cuda_set_device(p->devices[i]);
            CUDA_CHECK(cudaEventSynchronize(p->ev_pool[i][slot].ker));
        }
    }

    return slot;
}

// Whether the backend's stream is being captured into a CUDA graph.
static bool ggml_cuda_ar_capturing(ggml_backend_t backend) {
    auto * cuda_ctx = static_cast<ggml_backend_cuda_context *>(backend->context);
    ggml_cuda_set_device(cuda_ctx->device);
    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    CUDA_CHECK(cudaStreamIsCapturing(cuda_ctx->stream(), &status));
    return status != cudaStreamCaptureStatusNone;
}

// How a reduction of t goes: F32 at or over bf16_threshold bytes goes on the
// wire as BF16, halving the bytes pushed across PCIe (matches NCCL's
// behaviour), and nbytes on the wire at or over copy_threshold go over the
// copy engines, fewer over the chunked kernel.
struct ggml_cuda_ar_route {
    bool      use_bf16;
    ggml_type kernel_type;
    size_t    nbytes;
    bool      use_copy_engine;
};

static ggml_cuda_ar_route ggml_cuda_ar_route_of(const ggml_cuda_ar_pipeline * p, const ggml_tensor * t) {
    ggml_cuda_ar_route r;
    r.use_bf16 = t->type == GGML_TYPE_F32 && p->bf16_threshold > 0 && ggml_nbytes(t) >= p->bf16_threshold;
    r.kernel_type = r.use_bf16 ? GGML_TYPE_BF16 : t->type;
    r.nbytes = (size_t) ggml_nelements(t) * ggml_type_size(r.kernel_type);
    // No upper bound: copy_outer slices reductions larger than copy_bytes into
    // copy_bytes-sized pieces.
    r.use_copy_engine = p->copy_threshold > 0 && r.nbytes >= p->copy_threshold;
    return r;
}

// Packets one slot of the LL staging holds: a packet carries 4 wire bytes.
static size_t ggml_cuda_ar_ll_slot_packets(const ggml_cuda_ar_pipeline * p) {
    return (p->ll_max_bytes + 3) / 4;
}

// Packets one slot of a P2P inbox holds.
static size_t ggml_cuda_ar_p2p_slot_packets(const ggml_cuda_ar_pipeline * p) {
    return (p->p2p_max_bytes + 3) / 4;
}

// Per-AR copy-engine chunk size: env-var override if set, else heuristic
// (clamp(nbytes/4, HEURISTIC_MIN, HEURISTIC_MAX)).
static size_t ggml_cuda_ar_chunk_bytes(const ggml_cuda_ar_pipeline * p, size_t nbytes) {
    if (p->copy_chunk_bytes > 0) {
        return p->copy_chunk_bytes;
    }
    return std::min(GGML_CUDA_AR_COPY_CHUNK_BYTES_HEURISTIC_MAX,
                    std::max(GGML_CUDA_AR_COPY_CHUNK_BYTES_HEURISTIC_MIN, nbytes / 4));
}

static void ggml_cuda_ar_wait_for_compute(
        ggml_cuda_ar_pipeline * p, ggml_backend_cuda_context * cuda_ctx, int rank, int slot) {
    ggml_cuda_ar_event_slot & ev = p->ev_pool[rank][slot];
    CUDA_CHECK(cudaEventRecord(ev.app, cuda_ctx->stream()));
    CUDA_CHECK(cudaStreamWaitEvent(p->streams[rank], ev.app));
}

// ---------------------------------------------------------------------------
// Init / free
// ---------------------------------------------------------------------------

// Stores a known pattern over all of the peer's inbox, which ggml_cuda_ar_p2p_init reads back.
static __global__ void ggml_cuda_ar_p2p_probe_kernel(uint2 * inbox, int n_packets, uint32_t flag) {
    for (int i = threadIdx.x; i < n_packets; i += blockDim.x) {
        ggml_cuda_ar_ll_store(inbox + i, (uint32_t) i, flag);
    }
}

static void ggml_cuda_ar_p2p_free(ggml_cuda_ar_pipeline * p, int i) {
    ggml_cuda_ar_p2p_inbox & inbox = p->p2p[i];
    if (inbox.dev == nullptr) {
        return;
    }
    ggml_cuda_set_device(p->devices[i]);
    if (inbox.vmm) {
#if defined(GGML_USE_VMM)
        cuMemUnmap(reinterpret_cast<CUdeviceptr>(inbox.dev), inbox.size);
        cuMemAddressFree(reinterpret_cast<CUdeviceptr>(inbox.dev), inbox.size);
#endif // defined(GGML_USE_VMM)
    } else {
        cudaFree(inbox.dev);
    }
    inbox = {};
}

// Allocates device i's inbox, bytes on its card that both devices can store to. Two ranks on one GPU take cudaMalloc;
// two GPUs take a VMM mapping that grants both of them access, which leaves cudaDeviceEnablePeerAccess (every cudaMalloc
// of either card mapped to the other) alone.
static bool ggml_cuda_ar_p2p_alloc(ggml_cuda_ar_pipeline * p, int i, size_t bytes) {
    const int phys      = ggml_cuda_info().devices[p->devices[i]].physical_device;
    const int phys_peer = ggml_cuda_info().devices[p->devices[1 - i]].physical_device;
    ggml_cuda_ar_p2p_inbox & inbox = p->p2p[i];
    ggml_cuda_set_device(p->devices[i]);
    if (phys == phys_peer) {
        void * ptr = nullptr;
        if (cudaMalloc(&ptr, bytes) != cudaSuccess) {
            return false;
        }
        inbox.dev  = static_cast<uint2 *>(ptr);
        inbox.size = bytes;
        return true;
    }
#if defined(GGML_USE_VMM)
    CUmemAllocationProp prop = {};
    prop.type          = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id   = phys;
    size_t granularity = 0;
    if (cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM) != CUDA_SUCCESS ||
            granularity == 0) {
        return false;
    }
    const size_t size = granularity * ((bytes + granularity - 1) / granularity);
    CUmemGenericAllocationHandle handle;
    if (cuMemCreate(&handle, size, &prop, 0) != CUDA_SUCCESS) {
        return false;
    }
    CUdeviceptr ptr = 0;
    if (cuMemAddressReserve(&ptr, size, 0, 0, 0) != CUDA_SUCCESS) {
        cuMemRelease(handle);
        return false;
    }
    const bool mapped = cuMemMap(ptr, size, 0, handle, 0) == CUDA_SUCCESS;
    cuMemRelease(handle); // the mapping keeps the memory
    if (!mapped) {
        cuMemAddressFree(ptr, size);
        return false;
    }
    inbox.dev  = reinterpret_cast<uint2 *>(ptr);
    inbox.size = size;
    inbox.vmm  = true;
    CUmemAccessDesc access[2] = {};
    for (int k = 0; k < 2; ++k) {
        access[k].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        access[k].location.id   = k == 0 ? phys : phys_peer;
        access[k].flags         = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    }
    return cuMemSetAccess(ptr, size, access, 2) == CUDA_SUCCESS;
#else
    return false;
#endif // defined(GGML_USE_VMM)
}

// Sets up both inboxes and checks the link before a reduction relies on it: each device stores a pattern into the
// peer's inbox, the host reads it back, and the inboxes are zeroed so no slot holds a call's token. False when the
// cards cannot map each other's memory or the pattern does not arrive; the host paths then reduce everything.
static bool ggml_cuda_ar_p2p_init(ggml_cuda_ar_pipeline * p) {
    const int phys[2] = {
        ggml_cuda_info().devices[p->devices[0]].physical_device,
        ggml_cuda_info().devices[p->devices[1]].physical_device,
    };
    if (phys[0] != phys[1]) {
        const bool vmm = ggml_cuda_info().devices[p->devices[0]].vmm && ggml_cuda_info().devices[p->devices[1]].vmm;
        int access_01 = 0;
        int access_10 = 0;
        if (!vmm || cudaDeviceCanAccessPeer(&access_01, phys[0], phys[1]) != cudaSuccess ||
                cudaDeviceCanAccessPeer(&access_10, phys[1], phys[0]) != cudaSuccess || !access_01 || !access_10) {
            GGML_LOG_INFO("%s: GPUs %d and %d cannot map each other's memory%s: no card to card reduction\n",
                          __func__, phys[0], phys[1], vmm ? "" : " (no VMM)");
            return false;
        }
    }

    const int    n_packets = (int) (GGML_CUDA_AR_POOL_SIZE * ggml_cuda_ar_p2p_slot_packets(p));
    const size_t bytes     = (size_t) n_packets * sizeof(uint2);
    for (int i = 0; i < 2; ++i) {
        if (!ggml_cuda_ar_p2p_alloc(p, i, bytes)) {
            GGML_LOG_WARN("%s: P2P inbox of %zu bytes failed on device %d: no card to card reduction\n",
                          __func__, bytes, p->devices[i]);
            return false;
        }
    }

    const uint32_t flag[2] = {0xA5A50000u, 0xA5A50001u}; // what device 0 and device 1 store
    for (int i = 0; i < 2; ++i) {
        ggml_cuda_set_device(p->devices[i]);
        ggml_cuda_ar_p2p_probe_kernel<<<1, GGML_CUDA_AR_LL_THREADS, 0, p->streams[i]>>>(p->p2p[1 - i].dev, n_packets, flag[i]);
        if (cudaGetLastError() != cudaSuccess) {
            GGML_LOG_WARN("%s: P2P probe launch failed on device %d: no card to card reduction\n", __func__, p->devices[i]);
            return false;
        }
    }
    for (int i = 0; i < 2; ++i) {
        ggml_cuda_set_device(p->devices[i]);
        if (cudaStreamSynchronize(p->streams[i]) != cudaSuccess) {
            GGML_LOG_WARN("%s: P2P probe failed on device %d: no card to card reduction\n", __func__, p->devices[i]);
            return false;
        }
    }
    std::vector<uint2> seen(n_packets);
    for (int i = 0; i < 2; ++i) {
        ggml_cuda_set_device(p->devices[i]);
        if (cudaMemcpy(seen.data(), p->p2p[i].dev, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
            GGML_LOG_WARN("%s: P2P probe readback failed on device %d: no card to card reduction\n", __func__, p->devices[i]);
            return false;
        }
        for (int k = 0; k < n_packets; ++k) {
            if (seen[k].x != (uint32_t) k || seen[k].y != flag[1 - i]) {
                GGML_LOG_WARN("%s: device %d's stores did not reach device %d's memory (packet %d): "
                              "no card to card reduction\n", __func__, p->devices[1 - i], p->devices[i], k);
                return false;
            }
        }
        if (cudaMemsetAsync(p->p2p[i].dev, 0, bytes, p->streams[i]) != cudaSuccess ||
                cudaStreamSynchronize(p->streams[i]) != cudaSuccess) {
            GGML_LOG_WARN("%s: P2P inbox reset failed on device %d: no card to card reduction\n", __func__, p->devices[i]);
            return false;
        }
    }
    return true;
}

ggml_cuda_ar_pipeline * ggml_cuda_ar_pipeline_init(const int * devices, size_t n_devices) {

    if (n_devices != 2) {
        GGML_LOG_DEBUG("%s: internal AllReduce only supports n_devices=2 (got %zu); "
                       "falling back\n", __func__, n_devices);
        return nullptr;
    }

    // The chunked kernel uses __nanosleep, which is sm70+ (Volta+).
    for (size_t i = 0; i < n_devices; ++i) {
        const int cc = ggml_cuda_info().devices[devices[i]].cc;
        if (cc < GGML_CUDA_CC_VOLTA) {
            GGML_LOG_DEBUG("%s: internal AllReduce requires compute capability >= %d "
                           "(device %d has cc=%d); falling back\n",
                           __func__, GGML_CUDA_CC_VOLTA, devices[i], cc);
            return nullptr;
        }
    }

    // One host thread launches the kernel on device 0, then on device 1 (ggml_cuda_ar_allreduce). Under
    // CUDA_LAUNCH_BLOCKING the first launch waits for the kernel to end, which waits for device 1's, so it never returns.
    const char * launch_blocking = getenv("CUDA_LAUNCH_BLOCKING");
    if (launch_blocking != nullptr && atoi(launch_blocking) != 0) {
        GGML_LOG_WARN("%s: CUDA_LAUNCH_BLOCKING=%s would deadlock the internal AllReduce; "
                      "falling back to the meta backend's butterfly\n", __func__, launch_blocking);
        return nullptr;
    }

    auto * p = new ggml_cuda_ar_pipeline{};
    p->n_devices        = n_devices;
    p->copy_bytes       = GGML_CUDA_AR_COPY_MAX_BYTES;
    p->copy_threshold   = ggml_cuda_ar_env_u64("GGML_CUDA_AR_COPY_THRESHOLD", GGML_CUDA_AR_COPY_THRESHOLD_DEFAULT);
    // 0 = use the per-call heuristic (default).  Non-zero env value forces a
    // fixed chunk size for diagnostics, with a floor at COPY_CHUNK_BYTES_MIN.
    p->copy_chunk_bytes = ggml_cuda_ar_env_u64("GGML_CUDA_AR_COPY_CHUNK_BYTES", 0);
    if (p->copy_chunk_bytes > 0 && p->copy_chunk_bytes < GGML_CUDA_AR_COPY_CHUNK_BYTES_MIN) {
        GGML_LOG_WARN("%s: GGML_CUDA_AR_COPY_CHUNK_BYTES=%zu below minimum %zu; clamping\n",
                      __func__, p->copy_chunk_bytes, GGML_CUDA_AR_COPY_CHUNK_BYTES_MIN);
        p->copy_chunk_bytes = GGML_CUDA_AR_COPY_CHUNK_BYTES_MIN;
    }
    // Default 1: BF16 round-trip is always on for F32 inputs (any non-zero
    // ne).  Set GGML_CUDA_AR_BF16_THRESHOLD=0 to disable, or to a larger
    // byte threshold to opt out for small tensors.
    p->bf16_threshold   = ggml_cuda_ar_env_u64("GGML_CUDA_AR_BF16_THRESHOLD", 1);
    // Wire bytes up to which the host LL path reduces (0 disables this path), capped at one chunk's size.
    p->ll_max_bytes     = std::min(ggml_cuda_ar_env_u64("GGML_CUDA_AR_LL_MAX_BYTES", GGML_CUDA_AR_LL_MAX_BYTES_DEFAULT),
                                   (uint64_t) GGML_CUDA_AR_MAX_BYTES);
    // The same for the LL kernel card to card (ggml_cuda_ar_p2p_init), which takes over from the limit above where it is
    // larger.
    p->p2p_max_bytes    = std::min(ggml_cuda_ar_env_u64("GGML_CUDA_AR_P2P_MAX_BYTES", GGML_CUDA_AR_P2P_MAX_BYTES_DEFAULT),
                                   (uint64_t) GGML_CUDA_AR_MAX_BYTES);
    for (size_t i = 0; i < n_devices; ++i) {
        p->devices[i] = devices[i];
    }

    // Per-device streams and event pools.
    for (size_t i = 0; i < n_devices; ++i) {
        ggml_cuda_set_device(p->devices[i]);

        cudaStream_t stream = nullptr;
        if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) {
            GGML_LOG_ERROR("%s: cudaStreamCreateWithFlags failed for device %d\n",
                           __func__, p->devices[i]);
            ggml_cuda_ar_pipeline_free(p);
            return nullptr;
        }
        p->streams[i] = stream;

        for (int s = 0; s < GGML_CUDA_AR_POOL_SIZE; ++s) {
            bool ok =
                cudaEventCreateWithFlags(&p->ev_pool[i][s].app, cudaEventDisableTiming) == cudaSuccess &&
                cudaEventCreateWithFlags(&p->ev_pool[i][s].h2d, cudaEventDisableTiming) == cudaSuccess &&
                cudaEventCreateWithFlags(&p->ev_pool[i][s].ker, cudaEventDisableTiming) == cudaSuccess;
            for (int c = 0; ok && c < GGML_CUDA_AR_COPY_MAX_CHUNKS; ++c) {
                ok = cudaEventCreateWithFlags(&p->ev_pool[i][s].cpy[c], cudaEventDisableTiming) == cudaSuccess;
            }
            if (!ok) {
                GGML_LOG_ERROR("%s: cudaEventCreate failed for device %d slot %d\n",
                               __func__, p->devices[i], s);
                ggml_cuda_ar_pipeline_free(p);
                return nullptr;
            }
        }

        if (cudaEventCreateWithFlags(&p->host_large_read_done[i], cudaEventDisableTiming) != cudaSuccess) {
            GGML_LOG_ERROR("%s: cudaEventCreate for host_large_read_done failed for device %d\n",
                           __func__, p->devices[i]);
            ggml_cuda_ar_pipeline_free(p);
            return nullptr;
        }
        if (cudaEventCreateWithFlags(&p->dev_tmp_kernel_done[i], cudaEventDisableTiming) != cudaSuccess) {
            GGML_LOG_ERROR("%s: cudaEventCreate for dev_tmp_kernel_done failed for device %d\n",
                           __func__, p->devices[i]);
            ggml_cuda_ar_pipeline_free(p);
            return nullptr;
        }
    }

    // Arrival ring: cache-line padded so each GPU's int is on its own line.
    const size_t arrival_bytes =
        (size_t)GGML_CUDA_AR_POOL_SIZE * n_devices *
        GGML_CUDA_AR_KERNEL_BLOCKS * GGML_CUDA_AR_ARRIVAL_STRIDE;
    if (p->arrival.alloc(arrival_bytes) != cudaSuccess) {
        GGML_LOG_ERROR("%s: alloc for arrival ring failed (%zu bytes)\n",
                       __func__, arrival_bytes);
        ggml_cuda_ar_pipeline_free(p);
        return nullptr;
    }
    ggml_cuda_set_device(p->devices[0]);
    if (cudaMemset(p->arrival.dev, 0, arrival_bytes) != cudaSuccess) {
        GGML_LOG_ERROR("%s: cudaMemset for arrival ring failed (%zu bytes)\n",
                       __func__, arrival_bytes);
        ggml_cuda_ar_pipeline_free(p);
        return nullptr;
    }

    // Chunked-kernel counters, zero on both devices: the first call is token 1.
    for (size_t i = 0; i < n_devices; ++i) {
        ggml_cuda_set_device(p->devices[i]);
        if (cudaMalloc(reinterpret_cast<void **>(&p->counter[i]), 2*sizeof(int)) != cudaSuccess ||
                cudaMemset(p->counter[i], 0, 2*sizeof(int)) != cudaSuccess) {
            GGML_LOG_ERROR("%s: counters failed on device %d\n", __func__, p->devices[i]);
            ggml_cuda_ar_pipeline_free(p);
            return nullptr;
        }
    }

    // Per-device pinned staging buffers -- POOL_SIZE-deep ring so the chunked-
    // kernel can write the next slot's data while the peer is still reading
    // the previous slot's. Indexed by (slot * buf_bytes) at the call site.
    p->buf_bytes = GGML_CUDA_AR_MAX_BYTES;
    const size_t host_buf_total = (size_t) GGML_CUDA_AR_POOL_SIZE * p->buf_bytes;
    for (size_t i = 0; i < n_devices; ++i) {
        if (p->host_buf[i].alloc(host_buf_total) != cudaSuccess) {
            GGML_LOG_ERROR("%s: alloc for staging failed (%zu bytes)\n",
                           __func__, host_buf_total);
            ggml_cuda_ar_pipeline_free(p);
            return nullptr;
        }
    }

    // LL packets: zero, so no slot holds a call's token before the call writes it.
    const size_t ll_total = (size_t) GGML_CUDA_AR_POOL_SIZE * ggml_cuda_ar_ll_slot_packets(p) * sizeof(uint2);
    for (size_t i = 0; ll_total > 0 && i < n_devices; ++i) {
        ggml_cuda_set_device(p->devices[i]);
        if (p->ll_buf[i].alloc(ll_total) != cudaSuccess || cudaMemset(p->ll_buf[i].dev, 0, ll_total) != cudaSuccess) {
            GGML_LOG_ERROR("%s: LL staging failed (%zu bytes) on device %d\n", __func__, ll_total, p->devices[i]);
            ggml_cuda_ar_pipeline_free(p);
            return nullptr;
        }
    }

    if (p->p2p_max_bytes > 0 && !ggml_cuda_ar_p2p_init(p)) {
        for (size_t i = 0; i < n_devices; ++i) {
            ggml_cuda_ar_p2p_free(p, i);
        }
        p->p2p_max_bytes = 0;
    }

    // Copy-engine path: pinned host staging + device scratch, sized for the
    // largest tensor we accept on this path (GGML_CUDA_AR_COPY_MAX_BYTES).
    // dev_tmp is single-buffered; cross-AR safety is enforced by an explicit
    // cross-stream wait in copy_impl on the prior AR's add_kernel-done event.
    for (size_t i = 0; i < n_devices; ++i) {
        ggml_cuda_set_device(p->devices[i]);
        if (p->host_large[i].alloc(p->copy_bytes) != cudaSuccess) {
            GGML_LOG_ERROR("%s: alloc for large staging failed (%zu bytes)\n",
                           __func__, p->copy_bytes);
            ggml_cuda_ar_pipeline_free(p);
            return nullptr;
        }
        if (cudaMalloc(reinterpret_cast<void **>(&p->dev_tmp[i]), p->copy_bytes) != cudaSuccess) {
            GGML_LOG_ERROR("%s: cudaMalloc for copy scratch failed (%zu bytes) on device %d\n",
                           __func__, p->copy_bytes, p->devices[i]);
            ggml_cuda_ar_pipeline_free(p);
            return nullptr;
        }
    }

    GGML_LOG_INFO("%s: initialized AllReduce pipeline: %zu GPUs, "
                  "%zu KB chunked kernel staging + %zu KB LL staging (up to %zu KB on the wire) + %zu MB copy-engine staging per GPU, "
                  "LL card to card up to %zu bytes on the wire\n",
                  __func__, n_devices, p->buf_bytes >> 10, ll_total >> 10, p->ll_max_bytes >> 10, p->copy_bytes >> 20,
                  p->p2p_max_bytes);

    return p;
}

void ggml_cuda_ar_pipeline_free(ggml_cuda_ar_pipeline * p) {
    if (!p) {
        return;
    }

    // Drain all in-flight kernels before tearing down resources.
    for (int i = 0; i < p->n_devices; ++i) {
        if (p->streams[i]) {
            ggml_cuda_set_device(p->devices[i]);
            cudaStreamSynchronize(p->streams[i]);
        }
    }

    for (int i = 0; i < p->n_devices; ++i) {
        p->host_buf[i].free();
        p->ll_buf[i].free();
        ggml_cuda_ar_p2p_free(p, i);
        p->host_large[i].free();
        if (p->dev_tmp[i]) {
            ggml_cuda_set_device(p->devices[i]);
            cudaFree(p->dev_tmp[i]);
        }
        if (p->counter[i]) {
            ggml_cuda_set_device(p->devices[i]);
            cudaFree(p->counter[i]);
        }
        ggml_cuda_set_device(p->devices[i]);
        for (int s = 0; s < GGML_CUDA_AR_POOL_SIZE; ++s) {
            if (p->ev_pool[i][s].app) { cudaEventDestroy(p->ev_pool[i][s].app); }
            for (int c = 0; c < GGML_CUDA_AR_COPY_MAX_CHUNKS; ++c) {
                if (p->ev_pool[i][s].cpy[c]) { cudaEventDestroy(p->ev_pool[i][s].cpy[c]); }
            }
            if (p->ev_pool[i][s].h2d) { cudaEventDestroy(p->ev_pool[i][s].h2d); }
            if (p->ev_pool[i][s].ker) { cudaEventDestroy(p->ev_pool[i][s].ker); }
        }
        if (p->host_large_read_done[i]) {
            ggml_cuda_set_device(p->devices[i]);
            cudaEventDestroy(p->host_large_read_done[i]);
        }
        if (p->dev_tmp_kernel_done[i]) {
            ggml_cuda_set_device(p->devices[i]);
            cudaEventDestroy(p->dev_tmp_kernel_done[i]);
        }
        if (p->streams[i]) {
            ggml_cuda_set_device(p->devices[i]);
            cudaStreamDestroy(p->streams[i]);
        }
    }
    p->arrival.free();
    delete p;
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

// Asymmetric copy_impl: data sent over PCIe in T_src precision (one element of
// nbytes per ne element); accumulated locally into a T_dst buffer.  When
// T_src == T_dst this is the original homogeneous reduction.  When they differ
// (e.g. BF16 wire / F32 accumulator) the add kernel rounds dst through T_src
// for bit-equivalence between GPUs and we skip the otherwise-needed
// post-conversion entirely.
template <typename T_src, typename T_dst>
static bool ggml_cuda_ar_allreduce_copy_impl(
        ggml_cuda_ar_pipeline * p,
        ggml_backend_t        * backends,
        T_src * const           src_buf[GGML_CUDA_MAX_DEVICES],
        T_dst * const           dst_buf[GGML_CUDA_MAX_DEVICES],
        const bool              compute[GGML_CUDA_MAX_DEVICES],
        int64_t                 ne,
        size_t                  nbytes) {
    GGML_ASSERT(p->n_devices == 2);
    GGML_ASSERT(nbytes <= p->copy_bytes);
    GGML_ASSERT(ne <= std::numeric_limits<int>::max());

    const size_t chunk_bytes = ggml_cuda_ar_chunk_bytes(p, nbytes);
    GGML_ASSERT(chunk_bytes > 0);

    const int slot = ggml_cuda_ar_acquire_slot(p);
    const size_t copy_chunks = (nbytes + chunk_bytes - 1) / chunk_bytes;
    GGML_ASSERT(copy_chunks <= GGML_CUDA_AR_COPY_MAX_CHUNKS);

    ggml_backend_cuda_context * cuda_ctx[2] = {};

    // Stage 1: both GPUs copy their local contribution to pinned host memory.
    for (int i = 0; i < 2; ++i) {
        ggml_cuda_set_device(p->devices[i]);
        cuda_ctx[i] = static_cast<ggml_backend_cuda_context *>(backends[i]->context);
        GGML_ASSERT(cuda_ctx[i]->device == p->devices[i]);

        ggml_cuda_ar_wait_for_compute(p, cuda_ctx[i], i, slot);

        // Wait for peer's H2D from our host_large[i] (recorded in the
        // previous AR's stage 2) to complete before we overwrite host_large[i].
        // host_large_read_done[peer] = peer finished reading host_large[i].
        // No-op on the first AR -- no prior record exists.
        if (p->host_large_read_done_valid) {
            const int peer = 1 - i;
            CUDA_CHECK(cudaStreamWaitEvent(p->streams[i], p->host_large_read_done[peer]));
        }

        if (!compute[i]) {
            CUDA_CHECK(cudaMemsetAsync(src_buf[i], 0, nbytes, p->streams[i]));
        }

        for (size_t c = 0; c < copy_chunks; ++c) {
            const size_t offset = c * chunk_bytes;
            const size_t this_bytes = (nbytes - offset) < chunk_bytes ?
                (nbytes - offset) : chunk_bytes;

            CUDA_CHECK(cudaMemcpyAsync(
                p->host_large[i].host + offset, reinterpret_cast<char *>(src_buf[i]) + offset, this_bytes,
                cudaMemcpyDeviceToHost, p->streams[i]));
            CUDA_CHECK(cudaEventRecord(p->ev_pool[i][slot].cpy[c], p->streams[i]));
        }
    }

    // Stage 2: each GPU waits for each peer D2H chunk, pulls that chunk back to
    // local device scratch (dev_tmp), then performs one device-local add over
    // the assembled peer tensor.  The H2Ds run on the AR stream (copy engine)
    // and the add_kernel runs on the caller's compute stream, so the AR stream
    // stays pure-copy and avoids an in-stream copy->compute engine switch every
    // AR.  dev_tmp is single-buffered: the AR stream waits cross-stream on the
    // prior AR's add_kernel-done event before overwriting it.
    for (int i = 0; i < 2; ++i) {
        const int peer = 1 - i;
        ggml_cuda_set_device(p->devices[i]);

        // Wait for the previous AR's add_kernel (on the compute stream) to
        // finish reading dev_tmp before our H2D overwrites it.  No-op on the
        // first copy_impl call.
        if (p->dev_tmp_kernel_done_valid) {
            CUDA_CHECK(cudaStreamWaitEvent(p->streams[i], p->dev_tmp_kernel_done[i]));
        }

        for (size_t c = 0; c < copy_chunks; ++c) {
            const size_t offset = c * chunk_bytes;
            const size_t this_bytes = (nbytes - offset) < chunk_bytes ?
                (nbytes - offset) : chunk_bytes;

            CUDA_CHECK(cudaStreamWaitEvent(p->streams[i], p->ev_pool[peer][slot].cpy[c]));
            CUDA_CHECK(cudaMemcpyAsync(
                p->dev_tmp[i] + offset, p->host_large[peer].host + offset, this_bytes,
                cudaMemcpyHostToDevice, p->streams[i]));
        }

        // Mark our reads of host_large[peer] complete so peer's next AR can
        // safely overwrite it.
        CUDA_CHECK(cudaEventRecord(p->host_large_read_done[i], p->streams[i]));

        // Hand off from AR stream (copy engine) to compute stream: compute
        // stream waits for all H2Ds to finish, then runs the add_kernel.
        CUDA_CHECK(cudaEventRecord(p->ev_pool[i][slot].h2d, p->streams[i]));
        CUDA_CHECK(cudaStreamWaitEvent(cuda_ctx[i]->stream(), p->ev_pool[i][slot].h2d));

        const int block_size = 256;
        int n_blocks = (int) ((ne + block_size - 1) / block_size);
        if (n_blocks > 1024) {
            n_blocks = 1024;
        }
        ggml_cuda_ar_add_kernel<T_dst, T_src><<<n_blocks, block_size, 0, cuda_ctx[i]->stream()>>>(
            dst_buf[i],
            reinterpret_cast<const T_src *>(p->dev_tmp[i]),
            (int) ne);
        CUDA_CHECK(cudaGetLastError());

        // Record dev_tmp-released on the compute stream so the next copy_impl
        // can wait for the kernel to finish before overwriting dev_tmp.  Also
        // record AR-done as ev.ker for acquire_slot's pool-wraparound sync.
        CUDA_CHECK(cudaEventRecord(p->dev_tmp_kernel_done[i], cuda_ctx[i]->stream()));
        CUDA_CHECK(cudaEventRecord(p->ev_pool[i][slot].ker, cuda_ctx[i]->stream()));
    }
    p->host_large_read_done_valid = true;
    p->dev_tmp_kernel_done_valid = true;

    return true;
}

// Outer-level chunker: copy_impl handles up to copy_bytes per call (limited by
// the host_large / dev_tmp allocation size).  When the full AR exceeds that,
// slice the tensor into copy_bytes-sized pieces and call copy_impl repeatedly.
// Each slice goes through its own stage 1 -> stage 2 cycle and acquires its own
// slot, so cross-AR fences and pool wraparound work the same way as for any
// other sequence of small ARs.
template <typename T_src, typename T_dst>
static bool ggml_cuda_ar_allreduce_copy_outer(
        ggml_cuda_ar_pipeline * p,
        ggml_backend_t        * backends,
        T_src * const           src_buf[GGML_CUDA_MAX_DEVICES],
        T_dst * const           dst_buf[GGML_CUDA_MAX_DEVICES],
        const bool              compute[GGML_CUDA_MAX_DEVICES],
        int64_t                 ne) {
    const int64_t outer_max_elems = (int64_t) (p->copy_bytes / sizeof(T_src));
    GGML_ASSERT(outer_max_elems > 0);

    bool ok = true;
    for (int64_t outer_start = 0; outer_start < ne && ok; outer_start += outer_max_elems) {
        const int64_t outer_ne     = std::min(outer_max_elems, ne - outer_start);
        const size_t  outer_nbytes = (size_t) outer_ne * sizeof(T_src);

        T_src * src[GGML_CUDA_MAX_DEVICES] = {};
        T_dst * dst[GGML_CUDA_MAX_DEVICES] = {};
        for (int i = 0; i < p->n_devices; ++i) {
            src[i] = src_buf[i] + outer_start;
            dst[i] = dst_buf[i] + outer_start;
        }
        ok = ggml_cuda_ar_allreduce_copy_impl<T_src, T_dst>(
            p, backends, src, dst, compute, outer_ne, outer_nbytes);
    }
    return ok;
}

// Queues the reduction of ne elements of T_dst at data[i] on each device's compute stream, one LL launch when it is
// small (ggml_cuda_ar_ll_kernel), else one chunked launch a chunk (ggml_cuda_ar_kernel); zero_first[i]: data[i] is
// zeroed first.  Since the reduction is a barrier here, same-stream ordering subsumes any cross-stream event handshake
// that the copy-engine path needs, and skips the cross-stream scheduling overhead that was hurting the small-tensor
// (tg) latency on the AR-stream variant.  The kernels take their tokens and slots from the device, so the host
// neither waits nor records anything here: it can queue a whole graph's reductions ahead of the devices, and a CUDA
// graph that captured them replays them.
template <typename T_dst, typename T_wire>
static void ggml_cuda_ar_launch_kernels(
        ggml_cuda_ar_pipeline * p,
        ggml_backend_t        * backends,
        void * const          * data,
        const bool            * zero_first,
        int64_t                 ne) {
    // host_buf carries T_wire data: max_chunk_elems is what one slot holds
    const size_t max_chunk_elems = p->buf_bytes / sizeof(T_wire);
    // nbytes on the wire is the same on both devices, so both take the same kernel and buffers
    const size_t wire_bytes      = (size_t) ne * sizeof(T_wire);
    const bool   use_p2p         = wire_bytes <= p->p2p_max_bytes;
    const bool   use_ll          = use_p2p || wire_bytes <= p->ll_max_bytes;

    for (int64_t chunk_start = 0; chunk_start < ne; chunk_start += (int64_t) max_chunk_elems) {
        const size_t remaining_elems = (size_t) (ne - chunk_start);
        const size_t chunk_elems     = remaining_elems < max_chunk_elems ? remaining_elems : max_chunk_elems;

        for (int i = 0; i < p->n_devices; ++i) {
            const int peer = 1 - i;  // valid for n == 2 only
            ggml_cuda_set_device(p->devices[i]);
            auto * cuda_ctx = static_cast<ggml_backend_cuda_context *>(backends[i]->context);
            GGML_ASSERT(cuda_ctx->device == p->devices[i]);
            cudaStream_t stream = cuda_ctx->stream();

            T_dst * chunk = static_cast<T_dst *>(data[i]) + chunk_start;
            if (zero_first[i]) {
                CUDA_CHECK(cudaMemsetAsync(chunk, 0, chunk_elems * sizeof(T_dst), stream));
            }

            if (use_p2p) {
                // into the peer's inbox, from this device's own
                ggml_cuda_ar_ll_kernel<T_dst, T_wire><<<dim3(1), dim3(GGML_CUDA_AR_LL_THREADS), 0, stream>>>(
                    chunk,
                    chunk,
                    p->p2p[peer].dev,
                    p->p2p[i].dev,
                    static_cast<int>(ggml_cuda_ar_p2p_slot_packets(p)),
                    static_cast<int>(chunk_elems),
                    p->counter[i]);
            } else if (use_ll) {
                ggml_cuda_ar_ll_kernel<T_dst, T_wire><<<dim3(1), dim3(GGML_CUDA_AR_LL_THREADS), 0, stream>>>(
                    chunk,
                    chunk,
                    reinterpret_cast<uint2 *>(p->ll_buf[i].dev),
                    reinterpret_cast<const uint2 *>(p->ll_buf[peer].dev),
                    static_cast<int>(ggml_cuda_ar_ll_slot_packets(p)),
                    static_cast<int>(chunk_elems),
                    p->counter[i]);
            } else {
                ggml_cuda_ar_kernel<T_dst, T_wire><<<dim3(GGML_CUDA_AR_KERNEL_BLOCKS), dim3(256), 0, stream>>>(
                    chunk,
                    chunk,
                    reinterpret_cast<T_wire *>(p->host_buf[i].dev),
                    reinterpret_cast<const T_wire *>(p->host_buf[peer].dev),
                    p->buf_bytes / sizeof(T_wire),
                    static_cast<int>(chunk_elems),
                    reinterpret_cast<int *>(p->arrival.dev),
                    i,
                    p->counter[i]);
            }
            CUDA_CHECK(cudaGetLastError());
        }
    }
}

// Fuse only inside a joined meta capture. A standalone subgraph executable
// must never remember that a consumer was skipped by a preceding evaluation.
// The LL packet format, polling, parity slots and final counter stay identical.
static bool ggml_cuda_ar_try_hc_post(ggml_cuda_ar_pipeline * p, ggml_backend_t * backends,
                                    ggml_tensor ** tensors, bool bf16, bool copy_engine) {
    static const bool enabled = ggml_env_switch("GGML_CUDA_AR_HC_POST");
    if (!enabled || copy_engine || tensors[0]->type != GGML_TYPE_F32 || tensors[0]->ne[1] > 8 ||
        tensors[0]->ne[2] != 1 || tensors[0]->ne[3] != 1) { return false; }
    const int64_t count = ggml_nelements(tensors[0]);
    const size_t wire_bytes = (size_t) count * (bf16 ? sizeof(nv_bfloat16) : sizeof(float));
    const bool p2p = wire_bytes <= p->p2p_max_bytes;
    if (wire_bytes > p->buf_bytes || (!p2p && wire_bytes > p->ll_max_bytes)) { return false; }
    ggml_cuda_ar_hc_post args[2];
    ggml_tensor * consumer[2] = {};
    for (int rank = 0; rank < 2; ++rank) {
        auto * ctx = static_cast<ggml_backend_cuda_context *>(backends[rank]->context);
        const auto * graph = ctx->ar_graph_next;
        if (!ctx->outer_capture || !graph || ctx->curr_stream_no != 0 ||
            ggml_cuda_info().devices[ctx->device].cc != GGML_CUDA_CC_BLACKWELL ||
            !(tensors[rank]->flags & GGML_TENSOR_FLAG_COMPUTE) || !ggml_is_contiguous(tensors[rank])) { return false; }
        for (int i = 0; i < graph->n_nodes; ++i) {
            auto * n = graph->nodes[i];
            if (n->op == GGML_OP_NONE || n->op == GGML_OP_VIEW || n->op == GGML_OP_RESHAPE ||
                n->op == GGML_OP_PERMUTE || n->op == GGML_OP_TRANSPOSE) { continue; }
            consumer[rank] = n;
            break;
        }
        auto * node = consumer[rank];
        if (!node || !(node->flags & GGML_TENSOR_FLAG_COMPUTE) ||
            node->op != GGML_OP_DSV4_HC_POST || node->src[0] != tensors[rank] ||
            node->ne[0] != tensors[rank]->ne[0] || node->ne[1] != 4 ||
            node->ne[2] != tensors[rank]->ne[1] || node->ne[3] != 1) { return false; }
        const auto overlaps = [](const ggml_tensor * a, const ggml_tensor * b) {
            const uintptr_t ap = (uintptr_t) a->data, bp = (uintptr_t) b->data;
            return ap < bp + ggml_nbytes(b) && bp < ap + ggml_nbytes(a);
        };
        for (auto * t : {node, node->src[1], node->src[2], node->src[3]}) {
            if (!t || t->type != GGML_TYPE_F32 || !t->data || t->nb[0] != sizeof(float) ||
                t->ne[3] != 1 || ggml_nbytes(t) > INT_MAX) { return false; }
            for (int d = 1; d < 4; ++d) { if (t->nb[d] % sizeof(float)) { return false; } }
        }
        for (int s = 0; s < 4; ++s) {
            if (overlaps(node, node->src[s])) { return false; }
        }
        const auto * r = node->src[1];
        const auto * post = node->src[2];
        const auto * comb = node->src[3];
        args[rank] = {(const float *) r->data, (const float *) post->data, (const float *) comb->data,
                      (float *) node->data, (int) node->ne[0],
                      (int) (r->nb[1]/4), (int) (r->nb[2]/4), (int) (post->nb[1]/4),
                      (int) (comb->nb[1]/4), (int) (comb->nb[2]/4), (int) (node->nb[1]/4), (int) (node->nb[2]/4)};
    }
    for (int rank = 0; rank < 2; ++rank) {
        auto * ctx = static_cast<ggml_backend_cuda_context *>(backends[rank]->context);
        ggml_cuda_set_device(ctx->device);
        auto * mine = p2p ? p->p2p[1-rank].dev : reinterpret_cast<uint2 *>(p->ll_buf[rank].dev);
        const auto * other = p2p ? p->p2p[rank].dev : reinterpret_cast<const uint2 *>(p->ll_buf[1-rank].dev);
        const int slots = (int) (p2p ? ggml_cuda_ar_p2p_slot_packets(p) : ggml_cuda_ar_ll_slot_packets(p));
        auto * data = (float *) tensors[rank]->data;
        if (bf16) {
            ggml_cuda_ar_ll_kernel<float, nv_bfloat16, true><<<1, GGML_CUDA_AR_LL_THREADS, 0, ctx->stream()>>>(
                data, data, mine, other, slots, (int) count, p->counter[rank], args[rank]);
        } else {
            ggml_cuda_ar_ll_kernel<float, float, true><<<1, GGML_CUDA_AR_LL_THREADS, 0, ctx->stream()>>>(
                data, data, mine, other, slots, (int) count, p->counter[rank], args[rank]);
        }
        CUDA_CHECK(cudaGetLastError());
        ctx->ar_hc_completed = consumer[rank];
        ctx->ar_hc_graph = ctx->ar_graph_next;
    }
    GGML_LOG_DEBUG("%s: captured LL collective + HC_POST, %lld values, %s wire, %s inbox\n", __func__,
                   (long long) count, bf16 ? "bf16" : "f32", p2p ? "peer" : "host");
    return true;
}

bool ggml_cuda_ar_allreduce(
        ggml_cuda_ar_pipeline * p,
        ggml_backend_t        * backends,
        ggml_tensor           ** tensors) {
    GGML_ASSERT(p != nullptr);

    const int n = p->n_devices;
    GGML_ASSERT(n == 2);

    const ggml_type input_type = tensors[0]->type;
    GGML_ASSERT(input_type == GGML_TYPE_F32 || input_type == GGML_TYPE_F16 || input_type == GGML_TYPE_BF16);

    const int64_t ne = ggml_nelements(tensors[0]);
    GGML_ASSERT(ne > 0);

    // BF16 round-trip (chunked or copy-engine): the pre-conversion zeroes
    // inactive shards so the inner paths see them as already-prepared compute
    // tensors.
    const ggml_cuda_ar_route route = ggml_cuda_ar_route_of(p, tensors[0]);
    const bool      use_bf16        = route.use_bf16;
    const ggml_type kernel_type     = route.kernel_type;
    const size_t    type_size       = ggml_type_size(kernel_type);
    GGML_ASSERT(p->buf_bytes >= type_size);
    const size_t    nbytes          = route.nbytes;
    const bool      use_copy_engine = route.use_copy_engine;

    if (ggml_cuda_ar_try_hc_post(p, backends, tensors, use_bf16, use_copy_engine)) { return true; }

    // The copy-engine path syncs the host on its events, which no CUDA graph
    // can hold (ggml_cuda_ar_capturable).
    GGML_ASSERT(!use_copy_engine || !ggml_cuda_ar_capturing(backends[0]));

    bool compute_flag[GGML_CUDA_MAX_DEVICES] = {};
    for (int i = 0; i < n; ++i) {
        compute_flag[i] = (tensors[i]->flags & GGML_TENSOR_FLAG_COMPUTE) != 0;
    }

    // BF16 inactive-shard zeroing: when use_bf16 is on, the combined kernel
    // (chunked kernel path) and the combined add kernel (copy_engine path)
    // both accumulate into the F32 tensor data directly, so an inactive
    // shard's accumulator must start at zero.
    if (use_bf16) {
        for (int i = 0; i < n; ++i) {
            if (!compute_flag[i]) {
                auto * cuda_ctx = static_cast<ggml_backend_cuda_context *>(backends[i]->context);
                GGML_ASSERT(cuda_ctx->device == p->devices[i]);
                ggml_cuda_set_device(p->devices[i]);
                CUDA_CHECK(cudaMemsetAsync(tensors[i]->data, 0, (size_t) ne * sizeof(float), cuda_ctx->stream()));
            }
        }
    }

    // Pre-convert F32 -> BF16 into bf16_tmp ONLY for the copy_engine + use_bf16
    // path; the chunked kernel path's combined kernel does the conversion
    // inline as it writes to host_buf.
    ggml_cuda_pool_alloc<nv_bfloat16> bf16_tmp[GGML_CUDA_MAX_DEVICES];
    void * copy_src_ptr[GGML_CUDA_MAX_DEVICES] = {};

    if (use_copy_engine && use_bf16) {
        to_bf16_cuda_t to_bf16 = ggml_get_to_bf16_cuda(GGML_TYPE_F32);
        for (int i = 0; i < n; ++i) {
            auto * cuda_ctx = static_cast<ggml_backend_cuda_context *>(backends[i]->context);
            GGML_ASSERT(cuda_ctx->device == p->devices[i]);
            bf16_tmp[i].pool = &cuda_ctx->pool();
            bf16_tmp[i].alloc(ne);
            ggml_cuda_set_device(p->devices[i]);
            if (compute_flag[i]) {
                to_bf16(tensors[i]->data, bf16_tmp[i].get(), ne, cuda_ctx->stream());
                CUDA_CHECK(cudaGetLastError());
            } else {
                CUDA_CHECK(cudaMemsetAsync(bf16_tmp[i].get(), 0, nbytes, cuda_ctx->stream()));
            }
            copy_src_ptr[i] = bf16_tmp[i].get();
        }
    }

    bool ok = true;
    if (use_copy_engine) {
        // After up-front BF16 conversion, the tmp buffers already hold the
        // (possibly zeroed-for-inactive) data, so the inner path can treat
        // every shard as compute.
        bool inner_compute[GGML_CUDA_MAX_DEVICES];
        for (int i = 0; i < n; ++i) {
            inner_compute[i] = use_bf16 ? true : compute_flag[i];
        }

        // Dispatch into copy_impl with explicit src/dst types.  When use_bf16
        // is on, the wire type is BF16 (src = bf16_tmp) and the accumulator
        // is F32 (dst = tensors[i]->data); the combined add kernel rounds dst
        // through BF16 for bit-equivalence and writes F32 directly, so no
        // post-conversion is needed.  Otherwise src == dst (same native type).
        if (use_bf16) {
            GGML_ASSERT(kernel_type == GGML_TYPE_BF16);
            nv_bfloat16 * src[GGML_CUDA_MAX_DEVICES] = {};
            float       * dst[GGML_CUDA_MAX_DEVICES] = {};
            for (int i = 0; i < n; ++i) {
                src[i] = static_cast<nv_bfloat16 *>(copy_src_ptr[i]);
                dst[i] = static_cast<float *>(tensors[i]->data);
            }
            ok = ggml_cuda_ar_allreduce_copy_outer<nv_bfloat16, float>(
                p, backends, src, dst, inner_compute, ne);
        } else {
            switch (kernel_type) {
                case GGML_TYPE_F32: {
                    float * buf[GGML_CUDA_MAX_DEVICES] = {};
                    for (int i = 0; i < n; ++i) {
                        buf[i] = static_cast<float *>(tensors[i]->data);
                    }
                    ok = ggml_cuda_ar_allreduce_copy_outer<float, float>(
                        p, backends, buf, buf, inner_compute, ne);
                    break;
                }
                case GGML_TYPE_BF16: {
                    nv_bfloat16 * buf[GGML_CUDA_MAX_DEVICES] = {};
                    for (int i = 0; i < n; ++i) {
                        buf[i] = static_cast<nv_bfloat16 *>(tensors[i]->data);
                    }
                    ok = ggml_cuda_ar_allreduce_copy_outer<nv_bfloat16, nv_bfloat16>(
                        p, backends, buf, buf, inner_compute, ne);
                    break;
                }
                case GGML_TYPE_F16: {
                    half * buf[GGML_CUDA_MAX_DEVICES] = {};
                    for (int i = 0; i < n; ++i) {
                        buf[i] = static_cast<half *>(tensors[i]->data);
                    }
                    ok = ggml_cuda_ar_allreduce_copy_outer<half, half>(
                        p, backends, buf, buf, inner_compute, ne);
                    break;
                }
                default:
                    GGML_ASSERT(false);
            }
        }
    } else {
        // Match NCCL/meta-backend semantics: inactive shards contribute
        // zeros.  On the BF16 path the F32 tensor data was already
        // zeroed up-front (above), so per-chunk zeroing isn't needed.
        void * data[GGML_CUDA_MAX_DEVICES] = {};
        bool   zero_first[GGML_CUDA_MAX_DEVICES] = {};
        for (int i = 0; i < n; ++i) {
            data[i]       = tensors[i]->data;
            zero_first[i] = !compute_flag[i] && !use_bf16;
        }

        if (use_bf16) {
            GGML_ASSERT(input_type == GGML_TYPE_F32);
            ggml_cuda_ar_launch_kernels<float, nv_bfloat16>(p, backends, data, zero_first, ne);
        } else {
            switch (input_type) {
                case GGML_TYPE_F32:  ggml_cuda_ar_launch_kernels<float,       float>      (p, backends, data, zero_first, ne); break;
                case GGML_TYPE_F16:  ggml_cuda_ar_launch_kernels<half,        half>       (p, backends, data, zero_first, ne); break;
                case GGML_TYPE_BF16: ggml_cuda_ar_launch_kernels<nv_bfloat16, nv_bfloat16>(p, backends, data, zero_first, ne); break;
                default: GGML_ASSERT(false);
            }
        }
    }

    return ok;
}

bool ggml_cuda_ar_allgather(
        ggml_cuda_ar_pipeline * p,
        ggml_backend_t        * backends,
        ggml_tensor           ** tensors) {
    GGML_ASSERT(p != nullptr);
    GGML_ASSERT(p->n_devices == 2);

    const size_t nbytes = ggml_nbytes(tensors[0]);
    GGML_ASSERT(nbytes % sizeof(uint32_t) == 0);

    void * data[GGML_CUDA_MAX_DEVICES] = {};
    bool   zero_first[GGML_CUDA_MAX_DEVICES] = {};
    for (int i = 0; i < p->n_devices; ++i) {
        GGML_ASSERT(ggml_nbytes(tensors[i]) == nbytes);
        data[i]       = tensors[i]->data;
        zero_first[i] = (tensors[i]->flags & GGML_TENSOR_FLAG_COMPUTE) == 0; // a device with no part brings none
    }
    // never the copy engines: the words go through the kernels whatever their size, so a capture holds the gather
    ggml_cuda_ar_launch_kernels<uint32_t, uint32_t>(p, backends, data, zero_first, (int64_t) (nbytes / sizeof(uint32_t)));
    return true;
}

bool ggml_cuda_ar_capturable(const ggml_cuda_ar_pipeline * p, const ggml_tensor * t) {
    GGML_ASSERT(p != nullptr);
    return !ggml_cuda_ar_route_of(p, t).use_copy_engine;
}

#else // defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)

// HIP and MUSA lack the host-mapped pinned-memory APIs (cudaHostAllocPortable
// / cudaHostAllocMapped / cudaHostGetDevicePointer) and __nanosleep that this
// implementation relies on, so the internal AllReduce is a CUDA-only feature.
// The dispatcher in ggml-cuda.cu treats a nullptr pipeline as "init failed"
// and silently falls back to the meta backend's generic AllReduce.
ggml_cuda_ar_pipeline * ggml_cuda_ar_pipeline_init(const int *, size_t) {
    return nullptr;
}
void ggml_cuda_ar_pipeline_free(ggml_cuda_ar_pipeline *) {
}
bool ggml_cuda_ar_allreduce(ggml_cuda_ar_pipeline *, ggml_backend_t *, ggml_tensor **) {
    return false;
}
bool ggml_cuda_ar_allgather(ggml_cuda_ar_pipeline *, ggml_backend_t *, ggml_tensor **) {
    return false;
}

#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
