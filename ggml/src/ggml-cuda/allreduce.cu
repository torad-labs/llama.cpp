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
// CUDA AllReduce for tensor- and expert-parallel inference across CUDA GPUs.
//
// Provides an in-place sum reduction over matching tensors on CUDA
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

template <typename T>
struct ggml_cuda_ar_rank_ptrs {
    T * ptr[GGML_CUDA_MAX_DEVICES] = {};
};

template <typename T_wire>
using ggml_cuda_ar_accumulator = std::conditional_t<std::is_same_v<T_wire, uint32_t>, uint32_t, float>;

template <typename T_wire>
static __device__ __forceinline__ ggml_cuda_ar_accumulator<T_wire> ggml_cuda_ar_accumulate(
        ggml_cuda_ar_accumulator<T_wire> sum, T_wire value, int rank) {
    if constexpr (std::is_same_v<T_wire, uint32_t>) {
        return sum | value;
    } else {
        const float v = ggml_cuda_cast<float>(value);
        return rank == 0 ? v : __fadd_rn(sum, v);
    }
}

// Every receiver adds the original shares in increasing rank order, rounding only at the wire and destination.
template <typename T_dst, typename T_wire, bool peer_push = false>
static __global__ void ggml_cuda_ar_kernel_n(
        T_dst * data, ggml_cuda_ar_rank_ptrs<T_wire> staging, ggml_cuda_ar_rank_ptrs<T_wire> receives, size_t slot_elems, int count,
        int * arrival, int rank, int n_ranks, int * counter) {
    constexpr int V = ggml_cuda_get_max_cpy_bytes() / sizeof(T_wire);
    constexpr int A = GGML_CUDA_AR_ARRIVAL_STRIDE / sizeof(int);
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int gtid = bid * blockDim.x + tid;
    const int gnt = gridDim.x * blockDim.x;
    const int nv = count / V;
    const int tail = nv * V;
    __shared__ int token_shared;
    if (tid == 0) { token_shared = ggml_cuda_ar_signal_get(counter) + 1; }
    __syncthreads();
    const int token = token_shared;
    const size_t offset = (size_t) (token & 1) * slot_elems;
    for (int i = gtid; i < nv; i += gnt) {
        T_wire wire[V];
#pragma unroll
        for (int k = 0; k < V; ++k) { wire[k] = ggml_cuda_cast<T_wire>(data[i*V + k]); }
        if constexpr (peer_push) {
            for (int peer = 0; peer < n_ranks; ++peer) {
                if (peer != rank) { ggml_cuda_memcpy_1<sizeof(wire)>(staging.ptr[peer] + offset + i*V, wire); }
            }
        } else {
            ggml_cuda_memcpy_1<sizeof(wire)>(staging.ptr[rank] + offset + i*V, wire);
        }
    }
    if (bid == 0 && tid < count - tail) {
        const T_wire wire = ggml_cuda_cast<T_wire>(data[tail + tid]);
        if constexpr (peer_push) {
            for (int peer = 0; peer < n_ranks; ++peer) {
                if (peer != rank) { staging.ptr[peer][offset + tail + tid] = wire; }
            }
        } else {
            staging.ptr[rank][offset + tail + tid] = wire;
        }
    }
    __threadfence_system();
    __syncthreads();
    if (tid == 0) {
        int * slots = arrival + (size_t) (token & 1) * n_ranks * GGML_CUDA_AR_KERNEL_BLOCKS * A;
        ggml_cuda_ar_signal_set(slots + (rank*GGML_CUDA_AR_KERNEL_BLOCKS + bid)*A, token);
        __threadfence_system();
        for (int peer = 0; peer < n_ranks; ++peer) {
            if (peer == rank) { continue; }
            while (ggml_cuda_ar_signal_get(slots + (peer*GGML_CUDA_AR_KERNEL_BLOCKS + bid)*A) != token) {
#if __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
                __nanosleep(100);
#else
                NO_DEVICE_CODE;
#endif
            }
        }
    }
    __syncthreads();
    __threadfence_system();
    for (int i = gtid; i < nv; i += gnt) {
        ggml_cuda_ar_accumulator<T_wire> sum[V] = {};
        for (int peer = 0; peer < n_ranks; ++peer) {
            T_wire wire[V];
            if (peer == rank) {
#pragma unroll
                for (int k = 0; k < V; ++k) { wire[k] = ggml_cuda_cast<T_wire>(data[i*V + k]); }
            } else {
                ggml_cuda_memcpy_1<sizeof(wire)>(wire, receives.ptr[peer] + offset + i*V);
            }
#pragma unroll
            for (int k = 0; k < V; ++k) { sum[k] = ggml_cuda_ar_accumulate<T_wire>(sum[k], wire[k], peer); }
        }
#pragma unroll
        for (int k = 0; k < V; ++k) { data[i*V + k] = ggml_cuda_cast<T_dst>(sum[k]); }
    }
    if (bid == 0 && tid < count - tail) {
        ggml_cuda_ar_accumulator<T_wire> sum = {};
        for (int peer = 0; peer < n_ranks; ++peer) {
            const T_wire v = peer == rank ? ggml_cuda_cast<T_wire>(data[tail + tid]) : receives.ptr[peer][offset + tail + tid];
            sum = ggml_cuda_ar_accumulate<T_wire>(sum, v, peer);
        }
        data[tail + tid] = ggml_cuda_cast<T_dst>(sum);
    }
    if (tid == 0) {
        __threadfence();
        if (atomicAdd(counter + 1, 1) == (int) gridDim.x - 1) {
            atomicExch(counter + 1, 0);
            ggml_cuda_ar_signal_set(counter, token);
            __threadfence();
        }
    }
}

// Owner-sliced gather: each rank publishes only its own elements, and each non-owned element is read from its owner.
struct ggml_cuda_ar_owner_layout {
    int64_t begin[GGML_CUDA_MAX_DEVICES] = {};
    int64_t len[GGML_CUDA_MAX_DEVICES] = {};
    bool valid = false;
};

static __device__ int ggml_cuda_ar_owner_of(const ggml_cuda_ar_owner_layout & owner, int index, int n_ranks) {
    for (int peer = 0; peer < n_ranks; ++peer) {
        if (index >= owner.begin[peer] && index < owner.begin[peer] + owner.len[peer]) {
            return peer;
        }
    }
    return -1;
}

// The sender and receiver use the same stripe within each owner, so each arrival flag covers its block's reads.
static __device__ __forceinline__ void ggml_cuda_ar_copy_owner(
        uint32_t * dst, const uint32_t * src, int begin, int end, int gtid, int gnt) {
    const int first = min(end, (begin + 3) & ~3);
    const int last = max(first, end & ~3);
    if (begin + gtid < first) { dst[begin + gtid] = src[begin + gtid]; }
    for (int i = first/4 + gtid; i < last/4; i += gnt) {
        reinterpret_cast<uint4 *>(dst)[i] = reinterpret_cast<const uint4 *>(src)[i];
    }
    if (last + gtid < end) { dst[last + gtid] = src[last + gtid]; }
}

#if CUDART_VERSION >= 12000 && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
#define GGML_CUDA_AR_GRID_CONSTANT __grid_constant__
#else
#define GGML_CUDA_AR_GRID_CONSTANT
#endif

static __global__ void ggml_cuda_ar_owner_gather_kernel_n(
        uint32_t * data, GGML_CUDA_AR_GRID_CONSTANT const ggml_cuda_ar_rank_ptrs<uint32_t> staging,
        GGML_CUDA_AR_GRID_CONSTANT const ggml_cuda_ar_rank_ptrs<uint32_t> receives, size_t slot_elems,
        GGML_CUDA_AR_GRID_CONSTANT const ggml_cuda_ar_owner_layout owner,
        GGML_CUDA_AR_GRID_CONSTANT const ggml_cuda_ar_rank_ptrs<uint2> arrival, int rank, int n_ranks, int * counter) {
    constexpr int A = GGML_CUDA_AR_ARRIVAL_STRIDE / sizeof(uint2);
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int gtid = bid * blockDim.x + tid;
    const int gnt = gridDim.x * blockDim.x;
    __shared__ int token_shared;
    if (tid == 0) { token_shared = ggml_cuda_ar_signal_get(counter) + 1; }
    __syncthreads();
    const int token = token_shared;
    const size_t offset = (size_t) (token & 1) * slot_elems;
    const int own_begin = (int) owner.begin[rank];
    const int own_end = own_begin + (int) owner.len[rank];
#pragma unroll 1
    for (int dst = 0; dst < n_ranks; ++dst) {
        if (dst != rank) {
            ggml_cuda_ar_copy_owner(staging.ptr[dst] + offset, data, own_begin, own_end, gtid, gnt);
        }
    }
    __threadfence_system();
    __syncthreads();
    if (tid == 0) {
        const size_t slot = (size_t) (token & 1) * n_ranks * GGML_CUDA_AR_KERNEL_BLOCKS * A;
        // Publish after the payload stores; each receiver polls its own device memory.
        for (int peer = 0; peer < n_ranks; ++peer) {
            if (peer == rank) { continue; }
            ggml_cuda_ar_ll_store(arrival.ptr[peer] + slot + (rank*GGML_CUDA_AR_KERNEL_BLOCKS + bid)*A, 0, (uint32_t) token);
        }
        __threadfence_system();
        const uint2 * slots = arrival.ptr[rank] + slot;
        for (int peer = 0; peer < n_ranks; ++peer) {
            if (peer == rank) { continue; }
            while (ggml_cuda_ar_ll_load(slots + (peer*GGML_CUDA_AR_KERNEL_BLOCKS + bid)*A).y != (uint32_t) token) {
#if __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
                __nanosleep(100);
#else
                NO_DEVICE_CODE;
#endif
            }
        }
    }
    __syncthreads();
    __threadfence_system();
#pragma unroll 1
    for (int src = 0; src < n_ranks; ++src) {
        if (src != rank) {
            const int begin = (int) owner.begin[src];
            const int end = begin + (int) owner.len[src];
            ggml_cuda_ar_copy_owner(data, receives.ptr[src] + offset, begin, end, gtid, gnt);
        }
    }
    if (tid == 0) {
        __threadfence();
        if (atomicAdd(counter + 1, 1) == (int) gridDim.x - 1) {
            atomicExch(counter + 1, 0);
            ggml_cuda_ar_signal_set(counter, token);
            __threadfence();
        }
    }
}

// Reduce to owners on the chunked kernel's staging (ggml_cuda_ar_kernel_n's): a rank sends only the elements another
// rank owns, each to its owner (sends.ptr[owner], the owner's inbox from this rank, or this rank's host staging), and
// sums only its own, from the same shares (receives.ptr[peer]) in the same increasing rank order, so its elements take
// the AllReduce's bits while its others keep its share. Every block still waits for that block of every peer, so the
// ranks stay in lockstep (a rank that owns nothing would otherwise run ahead into a slot its peers still read).
#undef GGML_CUDA_AR_GRID_CONSTANT

template <typename T_dst, typename T_wire>
static __global__ void ggml_cuda_ar_owner_reduce_kernel_n(
        T_dst * data, ggml_cuda_ar_rank_ptrs<T_wire> sends, ggml_cuda_ar_rank_ptrs<T_wire> receives, size_t slot_elems, int count,
        ggml_cuda_ar_owner_layout owner, int * arrival, int rank, int n_ranks, int * counter) {
    constexpr int V = ggml_cuda_get_max_cpy_bytes() / sizeof(T_wire);
    constexpr int A = GGML_CUDA_AR_ARRIVAL_STRIDE / sizeof(int);
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int gtid = bid * blockDim.x + tid;
    const int gnt = gridDim.x * blockDim.x;
    const int nv = count / V;
    const int tail = nv * V;
    const int own_begin = (int) owner.begin[rank];
    const int own_end = own_begin + (int) owner.len[rank];
    __shared__ int token_shared;
    if (tid == 0) { token_shared = ggml_cuda_ar_signal_get(counter) + 1; }
    __syncthreads();
    const int token = token_shared;
    const size_t offset = (size_t) (token & 1) * slot_elems;
    for (int i = gtid; i < nv; i += gnt) {
        const int first = i*V;
        const int peer = ggml_cuda_ar_owner_of(owner, first, n_ranks);
        if (peer >= 0 && first + V <= owner.begin[peer] + owner.len[peer]) {
            if (peer != rank) {
                T_wire wire[V];
#pragma unroll
                for (int k = 0; k < V; ++k) { wire[k] = ggml_cuda_cast<T_wire>(data[first + k]); }
                ggml_cuda_memcpy_1<sizeof(wire)>(sends.ptr[peer] + offset + first, wire);
            }
            continue;
        }
#pragma unroll
        for (int k = 0; k < V; ++k) {
            const int dst = ggml_cuda_ar_owner_of(owner, first + k, n_ranks);
            if (dst >= 0 && dst != rank) { sends.ptr[dst][offset + first + k] = ggml_cuda_cast<T_wire>(data[first + k]); }
        }
    }
    if (bid == 0 && tid < count - tail) {
        const int dst = ggml_cuda_ar_owner_of(owner, tail + tid, n_ranks);
        if (dst >= 0 && dst != rank) { sends.ptr[dst][offset + tail + tid] = ggml_cuda_cast<T_wire>(data[tail + tid]); }
    }
    __threadfence_system();
    __syncthreads();
    if (tid == 0) {
        int * slots = arrival + (size_t) (token & 1) * n_ranks * GGML_CUDA_AR_KERNEL_BLOCKS * A;
        ggml_cuda_ar_signal_set(slots + (rank*GGML_CUDA_AR_KERNEL_BLOCKS + bid)*A, token);
        __threadfence_system();
        for (int peer = 0; peer < n_ranks; ++peer) {
            if (peer == rank) { continue; }
            while (ggml_cuda_ar_signal_get(slots + (peer*GGML_CUDA_AR_KERNEL_BLOCKS + bid)*A) != token) {
#if __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
                __nanosleep(100);
#else
                NO_DEVICE_CODE;
#endif
            }
        }
    }
    __syncthreads();
    __threadfence_system();
    for (int i = gtid; i < nv; i += gnt) {
        const int first = i*V;
        if (first + V <= own_begin || first >= own_end) { continue; }
        if (first >= own_begin && first + V <= own_end) {
            ggml_cuda_ar_accumulator<T_wire> sum[V] = {};
            for (int peer = 0; peer < n_ranks; ++peer) {
                T_wire wire[V];
                if (peer == rank) {
#pragma unroll
                    for (int k = 0; k < V; ++k) { wire[k] = ggml_cuda_cast<T_wire>(data[first + k]); }
                } else {
                    ggml_cuda_memcpy_1<sizeof(wire)>(wire, receives.ptr[peer] + offset + first);
                }
#pragma unroll
                for (int k = 0; k < V; ++k) { sum[k] = ggml_cuda_ar_accumulate<T_wire>(sum[k], wire[k], peer); }
            }
#pragma unroll
            for (int k = 0; k < V; ++k) { data[first + k] = ggml_cuda_cast<T_dst>(sum[k]); }
            continue;
        }
#pragma unroll
        for (int k = 0; k < V; ++k) {
            const int index = first + k;
            if (index < own_begin || index >= own_end) { continue; }
            ggml_cuda_ar_accumulator<T_wire> sum = {};
            for (int peer = 0; peer < n_ranks; ++peer) {
                const T_wire v = peer == rank ? ggml_cuda_cast<T_wire>(data[index]) : receives.ptr[peer][offset + index];
                sum = ggml_cuda_ar_accumulate<T_wire>(sum, v, peer);
            }
            data[index] = ggml_cuda_cast<T_dst>(sum);
        }
    }
    if (bid == 0 && tid < count - tail && tail + tid >= own_begin && tail + tid < own_end) {
        const int index = tail + tid;
        ggml_cuda_ar_accumulator<T_wire> sum = {};
        for (int peer = 0; peer < n_ranks; ++peer) {
            const T_wire v = peer == rank ? ggml_cuda_cast<T_wire>(data[index]) : receives.ptr[peer][offset + index];
            sum = ggml_cuda_ar_accumulate<T_wire>(sum, v, peer);
        }
        data[index] = ggml_cuda_cast<T_dst>(sum);
    }
    if (tid == 0) {
        __threadfence();
        if (atomicAdd(counter + 1, 1) == (int) gridDim.x - 1) {
            atomicExch(counter + 1, 0);
            ggml_cuda_ar_signal_set(counter, token);
            __threadfence();
        }
    }
}

template <typename T_dst, typename T_wire>
static __global__ void ggml_cuda_ar_ll_kernel_n(
        T_dst * data, ggml_cuda_ar_rank_ptrs<uint2> sends, ggml_cuda_ar_rank_ptrs<const uint2> receives,
        int slot_packets, int count, int rank, int n_ranks, int * counter) {
    constexpr int W = 4 / sizeof(T_wire);
    const int tid = threadIdx.x;
    const int nt = blockDim.x;
    const int npk = (count + W - 1) / W;
    const int token = ggml_cuda_ar_signal_get(counter) + 1;
    const size_t offset = (size_t) (token & 1) * slot_packets;
    for (int i = tid; i < npk; i += nt) {
        T_wire wire[W];
#pragma unroll
        for (int k = 0; k < W; ++k) {
            wire[k] = i*W + k < count ? ggml_cuda_cast<T_wire>(data[i*W + k]) : ggml_cuda_ar_wire_zero<T_wire>();
        }
        uint32_t bits;
        memcpy(&bits, wire, sizeof(bits));
        for (int peer = 0; peer < n_ranks; ++peer) {
            if (sends.ptr[peer]) { ggml_cuda_ar_ll_store(sends.ptr[peer] + offset + i, bits, (uint32_t) token); }
        }
    }
    for (int i0 = tid; i0 < npk; i0 += GGML_CUDA_AR_LL_UNROLL * nt) {
        ggml_cuda_ar_accumulator<T_wire> sum[GGML_CUDA_AR_LL_UNROLL][W] = {};
        for (int peer = 0; peer < n_ranks; ++peer) {
            uint2 packets[GGML_CUDA_AR_LL_UNROLL];
#pragma unroll
            for (int u = 0; u < GGML_CUDA_AR_LL_UNROLL; ++u) {
                if (peer != rank && i0 + u*nt < npk) {
                    packets[u] = ggml_cuda_ar_ll_load(receives.ptr[peer] + offset + i0 + u*nt);
                }
            }
#pragma unroll
            for (int u = 0; u < GGML_CUDA_AR_LL_UNROLL; ++u) {
                const int i = i0 + u*nt;
                if (i >= npk) { continue; }
                T_wire wire[W];
                if (peer == rank) {
#pragma unroll
                    for (int k = 0; k < W; ++k) {
                        wire[k] = i*W + k < count ? ggml_cuda_cast<T_wire>(data[i*W + k]) : ggml_cuda_ar_wire_zero<T_wire>();
                    }
                } else {
                    while (packets[u].y != (uint32_t) token) {
                        packets[u] = ggml_cuda_ar_ll_load(receives.ptr[peer] + offset + i);
                    }
                    memcpy(wire, &packets[u].x, sizeof(wire));
                }
#pragma unroll
                for (int k = 0; k < W; ++k) { sum[u][k] = ggml_cuda_ar_accumulate<T_wire>(sum[u][k], wire[k], peer); }
            }
        }
#pragma unroll
        for (int u = 0; u < GGML_CUDA_AR_LL_UNROLL; ++u) {
#pragma unroll
            for (int k = 0; k < W; ++k) {
                const int j = (i0 + u*nt)*W + k;
                if (j < count) { data[j] = ggml_cuda_cast<T_dst>(sum[u][k]); }
            }
        }
    }
    __syncthreads();
    if (tid == 0) { ggml_cuda_ar_signal_set(counter, token); }
}

template <typename T_dst, typename T_wire>
static __global__ void ggml_cuda_ar_add_kernel_n(
        T_dst * dst, const T_wire * peers, size_t peer_elems, int count, int rank, int n_ranks) {
    for (int i = blockIdx.x*blockDim.x + threadIdx.x; i < count; i += gridDim.x*blockDim.x) {
        ggml_cuda_ar_accumulator<T_wire> sum = {};
        for (int peer = 0; peer < n_ranks; ++peer) {
            const int index = peer < rank ? peer : peer - 1;
            const T_wire v = peer == rank ? ggml_cuda_cast<T_wire>(dst[i]) : peers[index*peer_elems + i];
            sum = ggml_cuda_ar_accumulate<T_wire>(sum, v, peer);
        }
        dst[i] = ggml_cuda_cast<T_dst>(sum);
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
// Peer inboxes can hold a larger gather without splitting its owners across synchronization rounds.
static constexpr size_t GGML_CUDA_AR_PEER_MAX_BYTES = 2 * 1024 * 1024;

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

// Device memory on one card that its peer may also store to: cudaMalloc when both ranks are one GPU, else a VMM mapping
// both GPUs may access (ggml_cuda_ar_peer_alloc). Also holds a plain cudaMalloc that only its own card uses.
struct ggml_cuda_ar_peer_mem {
    void * dev  = nullptr;
    size_t size = 0;     // bytes allocated or mapped
    bool   vmm  = false; // mapped with cuMemMap

    template <typename T>
    T * as() const {
        return static_cast<T *>(dev);
    }
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
    bool     copy_peer;      // copy-engine shares go card to card, into the peer's dev_tmp (ggml_cuda_ar_allreduce_copy_peer)
    bool     gather_peer;    // large captured gathers push into a separate parity ring on each peer
    bool     owner_slices;   // captured gathers write only each rank's own slice to every peer
    bool     reduce_owners;  // a reduce whose result each rank reads only in part sums only that part (ggml_cuda_ar_reduce_owned)
    uint64_t n_reduced_owned; // reductions summed to their owners, and to every rank, logged when the pipeline is freed
    uint64_t n_reduced_all;
    uint64_t n_chunked_peer;  // reductions on the chunked kernel that moved card to card, and through host memory
    uint64_t n_chunked_host;
    uint64_t call_count;

    // Per-device resources.
    ggml_cuda_ar_host_mapping host_buf[GGML_CUDA_MAX_DEVICES];   // pinned staging (chunked kernel)
    ggml_cuda_ar_host_mapping ll_buf[GGML_CUDA_MAX_DEVICES];     // pinned packets (LL kernel), 2 slots
    ggml_cuda_ar_peer_mem     p2p[GGML_CUDA_MAX_DEVICES];        // device packets (LL kernel card to card), 2 slots
    ggml_cuda_ar_peer_mem     gather_buf[GGML_CUDA_MAX_DEVICES]; // one parity ring per sender, local reads after peer writes
    ggml_cuda_ar_host_mapping host_large[GGML_CUDA_MAX_DEVICES]; // pinned staging (copy-engine), none under copy_peer
    // Device scratch for the copy-engine path: the peer's share, which the add_kernel adds. Under copy_peer the peer
    // copies it in, so it is mapped for both cards.
    ggml_cuda_ar_peer_mem     dev_tmp[GGML_CUDA_MAX_DEVICES];
    cudaStream_t             streams[GGML_CUDA_MAX_DEVICES];   // non-blocking
    // Copy-engine receives: the H2Ds of the peer's share run here, beside the D2Hs of the device's own share on
    // streams[], so a device takes in its peer's share while it sends its own. streams[] itself under
    // GGML_CUDA_AR_COPY_ONE_STREAM=1, where every receive waits for the device's last send.
    cudaStream_t             recv_streams[GGML_CUDA_MAX_DEVICES];
    ggml_cuda_ar_event_slot  ev_pool[GGML_CUDA_MAX_DEVICES][GGML_CUDA_AR_POOL_SIZE];

    // Copy-engine: per-device "I finished reading my peer's host_large"
    // event.  Indexed by RECORDER device.  Recorded same-device on recv_streams[i]
    // after stage 2's last H2D from host_large[peer].  Waited cross-device
    // by peer's stage-1 stream before the next AR overwrites host_large[peer].
    cudaEvent_t              host_large_read_done[GGML_CUDA_MAX_DEVICES];
    bool                     host_large_read_done_valid;

    // Copy-engine: per-device "my add_kernel is done with dev_tmp" event.
    // Recorded on the compute stream after each add_kernel; the receive
    // stream waits on it before the next copy_impl's H2D overwrites dev_tmp.  Lets us
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

static size_t ggml_cuda_ar_gather_payload_bytes(const ggml_cuda_ar_pipeline * p) {
    return (size_t) GGML_CUDA_AR_POOL_SIZE * GGML_CUDA_AR_PEER_MAX_BYTES * (p->n_devices - 1);
}

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

static void ggml_cuda_ar_peer_free(ggml_cuda_ar_pipeline * p, int i, ggml_cuda_ar_peer_mem & mem) {
    if (mem.dev == nullptr) {
        return;
    }
    if (mem.vmm) {
#if defined(GGML_USE_VMM)
        // Both cards may still read or store it from a compute stream, which the pipeline's own drain does not cover.
        for (int k = 0; k < p->n_devices; ++k) {
            ggml_cuda_set_device(p->devices[k]);
            cudaDeviceSynchronize();
        }
        ggml_cuda_set_device(p->devices[i]);
        cuMemUnmap(reinterpret_cast<CUdeviceptr>(mem.dev), mem.size);
        cuMemAddressFree(reinterpret_cast<CUdeviceptr>(mem.dev), mem.size);
#endif // defined(GGML_USE_VMM)
    } else {
        ggml_cuda_set_device(p->devices[i]);
        cudaFree(mem.dev);
    }
    mem = {};
}

static void ggml_cuda_ar_p2p_free(ggml_cuda_ar_pipeline * p, int i) {
    ggml_cuda_ar_peer_free(p, i, p->p2p[i]);
}

// Allocates bytes on device i's card that both devices can store to. Two ranks on one GPU take cudaMalloc; two GPUs take
// a VMM mapping that grants both of them access, which leaves cudaDeviceEnablePeerAccess (every cudaMalloc of either
// card mapped to the other) alone. On failure mem may hold a partial mapping, which ggml_cuda_ar_peer_free releases.
static bool ggml_cuda_ar_peer_alloc(ggml_cuda_ar_pipeline * p, int i, size_t bytes, ggml_cuda_ar_peer_mem & mem) {
    const int phys = ggml_cuda_info().devices[p->devices[i]].physical_device;
    std::vector<int> physical;
    for (int rank = 0; rank < p->n_devices; ++rank) {
        const int id = ggml_cuda_info().devices[p->devices[rank]].physical_device;
        if (std::find(physical.begin(), physical.end(), id) == physical.end()) { physical.push_back(id); }
    }
    ggml_cuda_set_device(p->devices[i]);
    if (physical.size() == 1) {
        void * ptr = nullptr;
        if (cudaMalloc(&ptr, bytes) != cudaSuccess) {
            return false;
        }
        mem.dev  = ptr;
        mem.size = bytes;
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
    mem.dev  = reinterpret_cast<void *>(ptr);
    mem.size = size;
    mem.vmm  = true;
    std::vector<CUmemAccessDesc> access(physical.size());
    for (size_t k = 0; k < physical.size(); ++k) {
        access[k].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        access[k].location.id   = physical[k];
        access[k].flags         = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    }
    return cuMemSetAccess(ptr, size, access.data(), access.size()) == CUDA_SUCCESS;
#else
    return false;
#endif // defined(GGML_USE_VMM)
}

// Sets up both inboxes and checks the link before a reduction relies on it: each device stores a pattern into the
// peer's inbox, the host reads it back, and the inboxes are zeroed so no slot holds a call's token. False when the
// cards cannot map each other's memory or the pattern does not arrive; the host paths then reduce everything.
static bool ggml_cuda_ar_p2p_init(ggml_cuda_ar_pipeline * p) {
    for (int i = 0; i < p->n_devices; ++i) {
        for (int peer = 0; peer < p->n_devices; ++peer) {
            const auto & a = ggml_cuda_info().devices[p->devices[i]];
            const auto & b = ggml_cuda_info().devices[p->devices[peer]];
            if (a.physical_device == b.physical_device) { continue; }
            int access = 0;
            if (!a.vmm || !b.vmm || cudaDeviceCanAccessPeer(&access, a.physical_device, b.physical_device) != cudaSuccess || !access) {
                GGML_LOG_INFO("%s: GPUs %d and %d cannot map each other's memory: no card to card reduction\n",
                              __func__, a.physical_device, b.physical_device);
                return false;
            }
        }
    }

    const int    n_packets = (int) (GGML_CUDA_AR_POOL_SIZE * ggml_cuda_ar_p2p_slot_packets(p));
    const size_t bytes     = (size_t) n_packets * sizeof(uint2) * (p->n_devices - 1);
    for (int i = 0; i < p->n_devices; ++i) {
        if (!ggml_cuda_ar_peer_alloc(p, i, bytes, p->p2p[i])) {
            GGML_LOG_WARN("%s: P2P inbox of %zu bytes failed on device %d: no card to card reduction\n",
                          __func__, bytes, p->devices[i]);
            return false;
        }
    }

    for (int i = 0; i < p->n_devices; ++i) {
        ggml_cuda_set_device(p->devices[i]);
        for (int peer = 0; peer < p->n_devices; ++peer) {
            if (peer == i) { continue; }
            const int index = i < peer ? i : i - 1;
            ggml_cuda_ar_p2p_probe_kernel<<<1, GGML_CUDA_AR_LL_THREADS, 0, p->streams[i]>>>(
                p->p2p[peer].as<uint2>() + (size_t) index*n_packets, n_packets, 0xA5A50000u + i);
            if (cudaGetLastError() != cudaSuccess) {
                GGML_LOG_WARN("%s: P2P probe launch failed on device %d: no card to card reduction\n", __func__, p->devices[i]);
                return false;
            }
        }
    }
    for (int i = 0; i < p->n_devices; ++i) {
        ggml_cuda_set_device(p->devices[i]);
        if (cudaStreamSynchronize(p->streams[i]) != cudaSuccess) {
            GGML_LOG_WARN("%s: P2P probe failed on device %d: no card to card reduction\n", __func__, p->devices[i]);
            return false;
        }
    }
    std::vector<uint2> seen((size_t) n_packets * (p->n_devices - 1));
    for (int i = 0; i < p->n_devices; ++i) {
        ggml_cuda_set_device(p->devices[i]);
        if (cudaMemcpy(seen.data(), p->p2p[i].dev, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
            GGML_LOG_WARN("%s: P2P probe readback failed on device %d: no card to card reduction\n", __func__, p->devices[i]);
            return false;
        }
        for (int peer = 0; peer < p->n_devices; ++peer) {
            if (peer == i) { continue; }
            const int index = peer < i ? peer : peer - 1;
            for (int k = 0; k < n_packets; ++k) {
                const auto & packet = seen[(size_t) index*n_packets + k];
                if (packet.x != (uint32_t) k || packet.y != 0xA5A50000u + peer) {
                    GGML_LOG_WARN("%s: device %d's stores did not reach device %d's memory (packet %d): "
                                  "no card to card reduction\n", __func__, p->devices[peer], p->devices[i], k);
                    return false;
                }
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

    if (n_devices < 2 || n_devices > GGML_CUDA_MAX_DEVICES) {
        GGML_LOG_DEBUG("%s: internal AllReduce needs 2..%d devices (got %zu); "
                       "falling back\n", __func__, GGML_CUDA_MAX_DEVICES, n_devices);
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
    // Default 1: a reduce whose only reader is a scatter by rows sums each element only on the rank that reads it.
    p->reduce_owners    = ggml_cuda_ar_env_u64("GGML_CUDA_AR_REDUCE_OWNERS", 1) != 0;
    for (size_t i = 0; i < n_devices; ++i) {
        p->devices[i] = devices[i];
    }

    // Per-device streams and event pools.
    const bool one_stream = ggml_cuda_ar_env_u64("GGML_CUDA_AR_COPY_ONE_STREAM", 0) != 0;
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

        if (one_stream) {
            p->recv_streams[i] = stream;
        } else if (cudaStreamCreateWithFlags(&p->recv_streams[i], cudaStreamNonBlocking) != cudaSuccess) {
            p->recv_streams[i] = nullptr;
            GGML_LOG_ERROR("%s: cudaStreamCreateWithFlags failed for device %d's receives\n",
                           __func__, p->devices[i]);
            ggml_cuda_ar_pipeline_free(p);
            return nullptr;
        }

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

    // Copy-engine path: device scratch sized for the largest piece copy_impl
    // takes (GGML_CUDA_AR_COPY_MAX_BYTES), and pinned host staging unless the
    // shares go card to card.  dev_tmp is single-buffered; cross-AR safety is
    // enforced by an explicit cross-stream wait on the prior AR's
    // add_kernel-done event.
    //
    // Card to card where the probe above found that each card's stores reach
    // the other's memory, unless GGML_CUDA_AR_COPY_PEER=0: each device then
    // copies its share straight into its peer's dev_tmp, which is mapped for
    // both cards.  A mapping that fails leaves the host staging.
    p->copy_peer = p->p2p_max_bytes > 0 && ggml_cuda_ar_env_u64("GGML_CUDA_AR_COPY_PEER", 1) != 0;
    const size_t receive_bytes = p->copy_bytes * (n_devices - 1);
    for (size_t i = 0; p->copy_peer && i < n_devices; ++i) {
        if (!ggml_cuda_ar_peer_alloc(p, (int) i, receive_bytes, p->dev_tmp[i])) {
            GGML_LOG_WARN("%s: copy-engine receive buffer of %zu bytes failed on device %d: "
                          "copy-engine shares stage through host memory\n", __func__, receive_bytes, p->devices[i]);
            for (size_t k = 0; k < n_devices; ++k) {
                ggml_cuda_ar_peer_free(p, (int) k, p->dev_tmp[k]);
            }
            p->copy_peer = false;
        }
    }
    for (size_t i = 0; !p->copy_peer && i < n_devices; ++i) {
        ggml_cuda_set_device(p->devices[i]);
        if (p->host_large[i].alloc(p->copy_bytes) != cudaSuccess) {
            GGML_LOG_ERROR("%s: alloc for large staging failed (%zu bytes)\n",
                           __func__, p->copy_bytes);
            ggml_cuda_ar_pipeline_free(p);
            return nullptr;
        }
        if (cudaMalloc(&p->dev_tmp[i].dev, receive_bytes) != cudaSuccess) {
            p->dev_tmp[i].dev = nullptr;
            GGML_LOG_ERROR("%s: cudaMalloc for copy scratch failed (%zu bytes) on device %d\n",
                           __func__, receive_bytes, p->devices[i]);
            ggml_cuda_ar_pipeline_free(p);
            return nullptr;
        }
        p->dev_tmp[i].size = receive_bytes;
    }

    const size_t gather_bytes = ggml_cuda_ar_gather_payload_bytes(p);
    p->gather_peer = p->copy_peer;
    for (size_t i = 0; p->gather_peer && i < n_devices; ++i) {
        if (!ggml_cuda_ar_peer_alloc(p, (int) i, gather_bytes + arrival_bytes, p->gather_buf[i]) ||
            cudaMemset(p->gather_buf[i].as<char>() + gather_bytes, 0, arrival_bytes) != cudaSuccess) {
            for (size_t k = 0; k < n_devices; ++k) { ggml_cuda_ar_peer_free(p, (int) k, p->gather_buf[k]); }
            p->gather_peer = false;
            GGML_LOG_WARN("%s: gather inbox allocation failed; large gathers stage through host memory\n", __func__);
        }
    }
    p->owner_slices = ggml_cuda_ar_env_u64("GGML_CUDA_AR_GATHER_OWNER_SLICES", 1) != 0 && p->gather_peer;

    GGML_LOG_INFO("%s: initialized AllReduce pipeline: %zu GPUs, "
                  "%zu KB chunked kernel staging + %zu KB LL staging (up to %zu KB on the wire) + %zu MB copy-engine %s per GPU, "
                  "LL card to card up to %zu bytes on the wire, copy-engine receives %s, %zu MB receive scratch per GPU, large gather %s\n",
                  __func__, n_devices, p->buf_bytes >> 10, ll_total >> 10, p->ll_max_bytes >> 10, p->copy_bytes >> 20,
                  p->copy_peer ? "receive buffer" : "staging", p->p2p_max_bytes,
                  p->copy_peer ? "card to card" : one_stream ? "after the sends" : "beside the sends", receive_bytes >> 20,
                  p->gather_peer ? "card to card" : "through host memory");
    if (p->owner_slices) {
        GGML_LOG_INFO("%s: owner-slice-peer gather enabled\n", __func__);
    }

    return p;
}

void ggml_cuda_ar_pipeline_free(ggml_cuda_ar_pipeline * p) {
    if (!p) {
        return;
    }
    GGML_LOG_DEBUG("%s: %llu reductions to owners, %llu to every rank\n", __func__,
                   (unsigned long long) p->n_reduced_owned, (unsigned long long) p->n_reduced_all);
    GGML_LOG_DEBUG("%s: on the chunked kernel %llu card to card, %llu through host memory\n", __func__,
                   (unsigned long long) p->n_chunked_peer, (unsigned long long) p->n_chunked_host);

    // Drain all in-flight kernels and copies before tearing down resources.
    for (int i = 0; i < p->n_devices; ++i) {
        if (p->streams[i]) {
            ggml_cuda_set_device(p->devices[i]);
            cudaStreamSynchronize(p->streams[i]);
        }
        if (p->recv_streams[i] && p->recv_streams[i] != p->streams[i]) {
            ggml_cuda_set_device(p->devices[i]);
            cudaStreamSynchronize(p->recv_streams[i]);
        }
    }

    for (int i = 0; i < p->n_devices; ++i) {
        p->host_buf[i].free();
        p->ll_buf[i].free();
        ggml_cuda_ar_p2p_free(p, i);
        ggml_cuda_ar_peer_free(p, i, p->gather_buf[i]);
        p->host_large[i].free();
        ggml_cuda_ar_peer_free(p, i, p->dev_tmp[i]);
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
        if (p->recv_streams[i] && p->recv_streams[i] != p->streams[i]) {
            ggml_cuda_set_device(p->devices[i]);
            cudaStreamDestroy(p->recv_streams[i]);
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

// The copy-engine path's last step on device i, once the compute stream waits
// for the peer's share in dev_tmp[i]: dst_buf += that share, on the compute
// stream.  Records dev_tmp-released, which the next call waits for before
// dev_tmp[i] is overwritten, and AR-done as ev.ker for acquire_slot's
// pool-wraparound sync.
template <typename T_src, typename T_dst>
static void ggml_cuda_ar_copy_add(
        ggml_cuda_ar_pipeline * p, int i, int slot, cudaStream_t stream, T_dst * dst_buf, int64_t ne) {
    const int block_size = 256;
    int n_blocks = (int) ((ne + block_size - 1) / block_size);
    if (n_blocks > 1024) {
        n_blocks = 1024;
    }
    if (p->n_devices == 2) {
        ggml_cuda_ar_add_kernel<T_dst, T_src><<<n_blocks, block_size, 0, stream>>>(
            dst_buf, p->dev_tmp[i].as<const T_src>(), (int) ne);
    } else {
        ggml_cuda_ar_add_kernel_n<T_dst, T_src><<<n_blocks, block_size, 0, stream>>>(
            dst_buf, p->dev_tmp[i].as<const T_src>(), p->copy_bytes / sizeof(T_src), (int) ne, i, p->n_devices);
    }
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(p->dev_tmp_kernel_done[i], stream));
    CUDA_CHECK(cudaEventRecord(p->ev_pool[i][slot].ker, stream));
}

// The copy-engine reduction card to card (p->copy_peer): each device copies its
// share once, over the link, straight into its peer's dev_tmp, which the
// pipeline mapped for both cards, and adds the share its peer copied into its
// own.  A share crosses the link once and never touches host memory, where the
// host staging takes it across twice.  The same add_kernel adds the same bytes
// as the host-staged path, so each device's result is bit for bit the same.
template <typename T_src, typename T_dst>
static bool ggml_cuda_ar_allreduce_copy_peer(
        ggml_cuda_ar_pipeline * p,
        ggml_backend_t        * backends,
        T_src * const           src_buf[GGML_CUDA_MAX_DEVICES],
        T_dst * const           dst_buf[GGML_CUDA_MAX_DEVICES],
        const bool              compute[GGML_CUDA_MAX_DEVICES],
        int64_t                 ne,
        size_t                  nbytes) {
    const int slot = ggml_cuda_ar_acquire_slot(p);

    ggml_backend_cuda_context * cuda_ctx[2] = {};

    // Stage 1, on each device's AR stream: once its compute has produced the
    // share, and its peer's add_kernel of the previous call is done reading
    // dev_tmp[peer], the share goes into dev_tmp[peer].
    for (int i = 0; i < 2; ++i) {
        const int peer = 1 - i;
        ggml_cuda_set_device(p->devices[i]);
        cuda_ctx[i] = static_cast<ggml_backend_cuda_context *>(backends[i]->context);
        GGML_ASSERT(cuda_ctx[i]->device == p->devices[i]);

        ggml_cuda_ar_wait_for_compute(p, cuda_ctx[i], i, slot);
        if (p->dev_tmp_kernel_done_valid) {
            CUDA_CHECK(cudaStreamWaitEvent(p->streams[i], p->dev_tmp_kernel_done[peer]));
        }
        if (!compute[i]) {
            CUDA_CHECK(cudaMemsetAsync(src_buf[i], 0, nbytes, p->streams[i]));
        }
        CUDA_CHECK(cudaMemcpyAsync(p->dev_tmp[peer].dev, src_buf[i], nbytes, cudaMemcpyDeviceToDevice, p->streams[i]));
        CUDA_CHECK(cudaEventRecord(p->ev_pool[i][slot].cpy[0], p->streams[i]));
    }

    // Stage 2, on each device's compute stream: once the peer's share has
    // arrived, and this device's own copy has read its share (the add writes
    // it in place, and the compute stream reuses a BF16 copy's pool memory once
    // it goes on), add.
    for (int i = 0; i < 2; ++i) {
        const int peer = 1 - i;
        ggml_cuda_set_device(p->devices[i]);
        cudaStream_t stream = cuda_ctx[i]->stream();
        CUDA_CHECK(cudaStreamWaitEvent(stream, p->ev_pool[peer][slot].cpy[0]));
        CUDA_CHECK(cudaStreamWaitEvent(stream, p->ev_pool[i][slot].cpy[0]));
        ggml_cuda_ar_copy_add<T_src, T_dst>(p, i, slot, stream, dst_buf[i], ne);
    }
    p->dev_tmp_kernel_done_valid = true;

    return true;
}

// Asymmetric copy_impl: data sent over PCIe in T_src precision (one element of
// nbytes per ne element); accumulated locally into a T_dst buffer.  When
// T_src == T_dst this is the original homogeneous reduction.  When they differ
// (e.g. BF16 wire / F32 accumulator) the add kernel rounds dst through T_src
// for bit-equivalence between GPUs and we skip the otherwise-needed
// post-conversion entirely.
template <typename T_src, typename T_dst>
static bool ggml_cuda_ar_allreduce_copy_n(
        ggml_cuda_ar_pipeline * p, ggml_backend_t * backends,
        T_src * const src_buf[GGML_CUDA_MAX_DEVICES], T_dst * const dst_buf[GGML_CUDA_MAX_DEVICES],
        const bool compute[GGML_CUDA_MAX_DEVICES], int64_t ne, size_t nbytes) {
    const int slot = ggml_cuda_ar_acquire_slot(p);
    const size_t chunk_bytes = ggml_cuda_ar_chunk_bytes(p, nbytes);
    const size_t chunks = (nbytes + chunk_bytes - 1) / chunk_bytes;
    GGML_ASSERT(chunks > 0 && chunks <= GGML_CUDA_AR_COPY_MAX_CHUNKS);
    ggml_backend_cuda_context * contexts[GGML_CUDA_MAX_DEVICES] = {};

    // Queue every sender's previous-call waits before recording any of this call's completion events.
    for (int i = 0; i < p->n_devices; ++i) {
        ggml_cuda_set_device(p->devices[i]);
        contexts[i] = static_cast<ggml_backend_cuda_context *>(backends[i]->context);
        GGML_ASSERT(contexts[i]->device == p->devices[i]);
        ggml_cuda_ar_wait_for_compute(p, contexts[i], i, slot);
        for (int peer = 0; peer < p->n_devices; ++peer) {
            if (peer == i) { continue; }
            if (p->copy_peer && p->dev_tmp_kernel_done_valid) {
                CUDA_CHECK(cudaStreamWaitEvent(p->streams[i], p->dev_tmp_kernel_done[peer]));
            } else if (!p->copy_peer && p->host_large_read_done_valid) {
                CUDA_CHECK(cudaStreamWaitEvent(p->streams[i], p->host_large_read_done[peer]));
            }
        }
        if (!compute[i]) { CUDA_CHECK(cudaMemsetAsync(src_buf[i], 0, nbytes, p->streams[i])); }
        for (size_t c = 0; c < chunks; ++c) {
            const size_t offset = c*chunk_bytes;
            const size_t bytes = std::min(chunk_bytes, nbytes - offset);
            const char * src = reinterpret_cast<const char *>(src_buf[i]) + offset;
            if (p->copy_peer) {
                for (int peer = 0; peer < p->n_devices; ++peer) {
                    if (peer == i) { continue; }
                    const int index = i < peer ? i : i - 1;
                    CUDA_CHECK(cudaMemcpyAsync(p->dev_tmp[peer].as<char>() + index*p->copy_bytes + offset,
                                               src, bytes, cudaMemcpyDeviceToDevice, p->streams[i]));
                }
            } else {
                CUDA_CHECK(cudaMemcpyAsync(p->host_large[i].host + offset, src, bytes,
                                           cudaMemcpyDeviceToHost, p->streams[i]));
            }
            CUDA_CHECK(cudaEventRecord(p->ev_pool[i][slot].cpy[c], p->streams[i]));
        }
    }
    for (int i = 0; i < p->n_devices; ++i) {
        ggml_cuda_set_device(p->devices[i]);
        const cudaStream_t stream = contexts[i]->stream();
        if (p->copy_peer) {
            for (int peer = 0; peer < p->n_devices; ++peer) {
                // Includes this rank's sends: its in-place source must remain unchanged until every copy has read it.
                CUDA_CHECK(cudaStreamWaitEvent(stream, p->ev_pool[peer][slot].cpy[chunks - 1]));
            }
        } else {
            const cudaStream_t recv = p->recv_streams[i];
            if (p->dev_tmp_kernel_done_valid) {
                CUDA_CHECK(cudaStreamWaitEvent(recv, p->dev_tmp_kernel_done[i]));
            }
            for (size_t c = 0; c < chunks; ++c) {
                const size_t offset = c*chunk_bytes;
                const size_t bytes = std::min(chunk_bytes, nbytes - offset);
                for (int peer = 0; peer < p->n_devices; ++peer) {
                    if (peer == i) { continue; }
                    const int index = peer < i ? peer : peer - 1;
                    CUDA_CHECK(cudaStreamWaitEvent(recv, p->ev_pool[peer][slot].cpy[c]));
                    CUDA_CHECK(cudaMemcpyAsync(p->dev_tmp[i].as<char>() + index*p->copy_bytes + offset,
                                               p->host_large[peer].host + offset, bytes, cudaMemcpyHostToDevice, recv));
                }
            }
            CUDA_CHECK(cudaEventRecord(p->host_large_read_done[i], recv));
            CUDA_CHECK(cudaEventRecord(p->ev_pool[i][slot].h2d, recv));
            CUDA_CHECK(cudaStreamWaitEvent(stream, p->ev_pool[i][slot].h2d));
            CUDA_CHECK(cudaStreamWaitEvent(stream, p->ev_pool[i][slot].cpy[chunks - 1]));
        }
        ggml_cuda_ar_copy_add<T_src, T_dst>(p, i, slot, stream, dst_buf[i], ne);
    }
    p->host_large_read_done_valid = !p->copy_peer;
    p->dev_tmp_kernel_done_valid = true;
    return true;
}

template <typename T_src, typename T_dst>
static bool ggml_cuda_ar_allreduce_copy_impl(
        ggml_cuda_ar_pipeline * p,
        ggml_backend_t        * backends,
        T_src * const           src_buf[GGML_CUDA_MAX_DEVICES],
        T_dst * const           dst_buf[GGML_CUDA_MAX_DEVICES],
        const bool              compute[GGML_CUDA_MAX_DEVICES],
        int64_t                 ne,
        size_t                  nbytes) {
    GGML_ASSERT(nbytes <= p->copy_bytes);
    GGML_ASSERT(ne <= std::numeric_limits<int>::max());

    if (p->n_devices != 2) {
        return ggml_cuda_ar_allreduce_copy_n<T_src, T_dst>(p, backends, src_buf, dst_buf, compute, ne, nbytes);
    }
    if (p->copy_peer) {
        return ggml_cuda_ar_allreduce_copy_peer<T_src, T_dst>(p, backends, src_buf, dst_buf, compute, ne, nbytes);
    }

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
    // the assembled peer tensor.  The H2Ds run on the receive stream, beside
    // this device's own D2Hs on the AR stream, so each chunk of the peer's share
    // comes in as soon as the peer has sent it, while this device is still
    // sending its own: the two directions of the link carry both at once.  The
    // add_kernel runs on the caller's compute stream, so the copy streams stay
    // pure-copy.  dev_tmp is single-buffered: the receive stream waits
    // cross-stream on the prior AR's add_kernel-done event before overwriting
    // it.
    for (int i = 0; i < 2; ++i) {
        const int peer = 1 - i;
        ggml_cuda_set_device(p->devices[i]);
        cudaStream_t recv = p->recv_streams[i];

        // Wait for the previous AR's add_kernel (on the compute stream) to
        // finish reading dev_tmp before our H2D overwrites it.  No-op on the
        // first copy_impl call.
        if (p->dev_tmp_kernel_done_valid) {
            CUDA_CHECK(cudaStreamWaitEvent(recv, p->dev_tmp_kernel_done[i]));
        }

        for (size_t c = 0; c < copy_chunks; ++c) {
            const size_t offset = c * chunk_bytes;
            const size_t this_bytes = (nbytes - offset) < chunk_bytes ?
                (nbytes - offset) : chunk_bytes;

            CUDA_CHECK(cudaStreamWaitEvent(recv, p->ev_pool[peer][slot].cpy[c]));
            CUDA_CHECK(cudaMemcpyAsync(
                p->dev_tmp[i].as<char>() + offset, p->host_large[peer].host + offset, this_bytes,
                cudaMemcpyHostToDevice, recv));
        }

        // Mark our reads of host_large[peer] complete so peer's next AR can
        // safely overwrite it.
        CUDA_CHECK(cudaEventRecord(p->host_large_read_done[i], recv));

        // Hand off from the copy streams to the compute stream: it waits for
        // all H2Ds, and for this device's own last D2H, then runs the
        // add_kernel.  The receives no longer follow the sends on one stream,
        // and a late device takes in its peer's share before it has sent its
        // own: the add_kernel writes the share in place (src_buf == dst_buf),
        // and the compute stream reuses a BF16 copy's pool memory once it goes
        // on, so neither may happen before the last D2H has read it.
        CUDA_CHECK(cudaEventRecord(p->ev_pool[i][slot].h2d, recv));
        CUDA_CHECK(cudaStreamWaitEvent(cuda_ctx[i]->stream(), p->ev_pool[i][slot].h2d));
        CUDA_CHECK(cudaStreamWaitEvent(cuda_ctx[i]->stream(), p->ev_pool[i][slot].cpy[copy_chunks - 1]));

        ggml_cuda_ar_copy_add<T_src, T_dst>(p, i, slot, cuda_ctx[i]->stream(), dst_buf[i], ne);
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

// The part of each rank's elements in the chunk [chunk_start, chunk_start + chunk_elems), counted from the chunk's start.
static ggml_cuda_ar_owner_layout ggml_cuda_ar_chunk_owner(
        const ggml_cuda_ar_owner_layout & owner, int64_t chunk_start, int64_t chunk_elems, int n_ranks) {
    ggml_cuda_ar_owner_layout chunk_owner;
    chunk_owner.valid = true;
    for (int rank = 0; rank < n_ranks; ++rank) {
        const int64_t begin = std::max(owner.begin[rank], chunk_start);
        const int64_t end = std::min(owner.begin[rank] + owner.len[rank], chunk_start + chunk_elems);
        chunk_owner.begin[rank] = end > begin ? begin - chunk_start : 0;
        chunk_owner.len[rank] = end > begin ? end - begin : 0;
    }
    return chunk_owner;
}

// Rank i's chunked staging in the gather inboxes: sends.ptr[peer] is the parity ring rank i pushes into on that peer, and
// receives.ptr[peer] the one that peer pushes into on rank i, each of GGML_CUDA_AR_POOL_SIZE slots of max_chunk_elems.
template <typename T_wire>
static void ggml_cuda_ar_peer_inboxes(ggml_cuda_ar_pipeline * p, int i, size_t max_chunk_elems,
                                      ggml_cuda_ar_rank_ptrs<T_wire> & sends, ggml_cuda_ar_rank_ptrs<T_wire> & receives) {
    for (int peer = 0; peer < p->n_devices; ++peer) {
        if (peer == i) { continue; }
        const int send_index = i < peer ? i : i - 1;
        const int recv_index = peer < i ? peer : peer - 1;
        sends.ptr[peer] = p->gather_buf[peer].as<T_wire>() + send_index*GGML_CUDA_AR_POOL_SIZE*max_chunk_elems;
        receives.ptr[peer] = p->gather_buf[i].as<T_wire>() + recv_index*GGML_CUDA_AR_POOL_SIZE*max_chunk_elems;
    }
}

// Queues the reduction of ne elements of T_dst at data[i] on each device's compute stream, one LL launch when it is
// small (ggml_cuda_ar_ll_kernel), else one chunked launch a chunk (ggml_cuda_ar_kernel); zero_first[i]: data[i] is
// zeroed first. Where the cards map each other's memory (p->gather_peer), the chunked launches push each share into
// its peers' gather inboxes, card to card, instead of staging it through host memory.  Since the reduction is a barrier here, same-stream ordering subsumes any cross-stream event handshake
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
        int64_t                 ne,
        const ggml_cuda_ar_owner_layout * owner = nullptr) {
    // nbytes on the wire is the same on both devices, so both take the same kernel and buffers
    const size_t wire_bytes      = (size_t) ne * sizeof(T_wire);
    const bool   use_p2p         = wire_bytes <= p->p2p_max_bytes;
    const bool   use_ll          = use_p2p || wire_bytes <= p->ll_max_bytes;
    const bool   gather_peer     = p->gather_peer && !use_ll;
    const size_t max_chunk_elems = (gather_peer ? GGML_CUDA_AR_PEER_MAX_BYTES : p->buf_bytes) / sizeof(T_wire);
    ggml_cuda_ar_rank_ptrs<uint2> arrival;
    if constexpr (std::is_same_v<T_dst, uint32_t> && std::is_same_v<T_wire, uint32_t>) {
        if (gather_peer && p->owner_slices) {
            const size_t arrival_offset = ggml_cuda_ar_gather_payload_bytes(p);
            for (int rank = 0; rank < p->n_devices; ++rank) {
                arrival.ptr[rank] = reinterpret_cast<uint2 *>(p->gather_buf[rank].as<char>() + arrival_offset);
            }
        }
    }
    if constexpr (!std::is_same_v<T_dst, uint32_t>) {
        if (!use_ll) { (gather_peer ? p->n_chunked_peer : p->n_chunked_host)++; }
    }

    for (int64_t chunk_start = 0; chunk_start < ne; chunk_start += (int64_t) max_chunk_elems) {
        const size_t remaining_elems = (size_t) (ne - chunk_start);
        const size_t chunk_elems     = remaining_elems < max_chunk_elems ? remaining_elems : max_chunk_elems;
        const ggml_cuda_ar_owner_layout chunk_owner = owner != nullptr && owner->valid ?
            ggml_cuda_ar_chunk_owner(*owner, chunk_start, (int64_t) chunk_elems, p->n_devices) : ggml_cuda_ar_owner_layout{};

        for (int i = 0; i < p->n_devices; ++i) {
            ggml_cuda_set_device(p->devices[i]);
            auto * cuda_ctx = static_cast<ggml_backend_cuda_context *>(backends[i]->context);
            GGML_ASSERT(cuda_ctx->device == p->devices[i]);
            cudaStream_t stream = cuda_ctx->stream();

            T_dst * chunk = static_cast<T_dst *>(data[i]) + chunk_start;
            if (zero_first[i]) {
                CUDA_CHECK(cudaMemsetAsync(chunk, 0, chunk_elems * sizeof(T_dst), stream));
            }

            if (gather_peer) {
                ggml_cuda_ar_rank_ptrs<T_wire> sends, receives;
                ggml_cuda_ar_peer_inboxes<T_wire>(p, i, max_chunk_elems, sends, receives);
                if constexpr (std::is_same_v<T_dst, uint32_t> && std::is_same_v<T_wire, uint32_t>) {
                    if (p->owner_slices && chunk_owner.valid) {
                        ggml_cuda_ar_owner_gather_kernel_n<<<GGML_CUDA_AR_KERNEL_BLOCKS, 256, 0, stream>>>(
                            chunk, sends, receives, max_chunk_elems, chunk_owner, arrival, i, p->n_devices, p->counter[i]);
                        CUDA_CHECK(cudaGetLastError());
                        continue;
                    }
                }
                ggml_cuda_ar_kernel_n<T_dst, T_wire, true><<<GGML_CUDA_AR_KERNEL_BLOCKS, 256, 0, stream>>>(
                    chunk, sends, receives, max_chunk_elems, (int) chunk_elems, reinterpret_cast<int *>(p->arrival.dev),
                    i, p->n_devices, p->counter[i]);
                CUDA_CHECK(cudaGetLastError());
                continue;
            }
            if (p->n_devices != 2) {
                if (use_ll) {
                    ggml_cuda_ar_rank_ptrs<uint2> sends;
                    ggml_cuda_ar_rank_ptrs<const uint2> receives;
                    const size_t packets = use_p2p ? ggml_cuda_ar_p2p_slot_packets(p) : ggml_cuda_ar_ll_slot_packets(p);
                    if (!use_p2p) { sends.ptr[i] = reinterpret_cast<uint2 *>(p->ll_buf[i].dev); }
                    for (int rank = 0; rank < p->n_devices; ++rank) {
                        if (rank == i) { continue; }
                        if (use_p2p) {
                            const int send_index = i < rank ? i : i - 1;
                            const int recv_index = rank < i ? rank : rank - 1;
                            sends.ptr[rank] = p->p2p[rank].as<uint2>() + send_index*GGML_CUDA_AR_POOL_SIZE*packets;
                            receives.ptr[rank] = p->p2p[i].as<const uint2>() + recv_index*GGML_CUDA_AR_POOL_SIZE*packets;
                        } else {
                            receives.ptr[rank] = reinterpret_cast<const uint2 *>(p->ll_buf[rank].dev);
                        }
                    }
                    ggml_cuda_ar_ll_kernel_n<T_dst, T_wire><<<1, GGML_CUDA_AR_LL_THREADS, 0, stream>>>(
                        chunk, sends, receives, (int) packets, (int) chunk_elems, i, p->n_devices, p->counter[i]);
                } else {
                    ggml_cuda_ar_rank_ptrs<T_wire> staging;
                    for (int rank = 0; rank < p->n_devices; ++rank) {
                        staging.ptr[rank] = reinterpret_cast<T_wire *>(p->host_buf[rank].dev);
                    }
                    ggml_cuda_ar_kernel_n<T_dst, T_wire><<<GGML_CUDA_AR_KERNEL_BLOCKS, 256, 0, stream>>>(
                        chunk, staging, staging, max_chunk_elems, (int) chunk_elems, reinterpret_cast<int *>(p->arrival.dev),
                        i, p->n_devices, p->counter[i]);
                }
                CUDA_CHECK(cudaGetLastError());
                continue;
            }
            const int peer = 1 - i;
            if (use_p2p) {
                // into the peer's inbox, from this device's own
                ggml_cuda_ar_ll_kernel<T_dst, T_wire><<<dim3(1), dim3(GGML_CUDA_AR_LL_THREADS), 0, stream>>>(
                    chunk,
                    chunk,
                    p->p2p[peer].as<uint2>(),
                    p->p2p[i].as<uint2>(),
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
                                    ggml_tensor ** tensors, const ggml_cgraph * const * next, bool bf16, bool copy_engine) {
    static const bool enabled = ggml_env_switch("GGML_CUDA_AR_HC_POST");
    if (p->n_devices != 2 || !enabled || !next || copy_engine || tensors[0]->type != GGML_TYPE_F32 || tensors[0]->ne[1] > 8 ||
        tensors[0]->ne[2] != 1 || tensors[0]->ne[3] != 1) { return false; }
    const int64_t count = ggml_nelements(tensors[0]);
    const size_t wire_bytes = (size_t) count * (bf16 ? sizeof(nv_bfloat16) : sizeof(float));
    const bool p2p = wire_bytes <= p->p2p_max_bytes;
    if (wire_bytes > p->buf_bytes || (!p2p && wire_bytes > p->ll_max_bytes)) { return false; }
    ggml_cuda_ar_hc_post args[2];
    ggml_tensor * consumer[2] = {};
    for (int rank = 0; rank < 2; ++rank) {
        auto * ctx = static_cast<ggml_backend_cuda_context *>(backends[rank]->context);
        const auto * graph = next[rank];
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
        auto * mine = p2p ? p->p2p[1-rank].as<uint2>() : reinterpret_cast<uint2 *>(p->ll_buf[rank].dev);
        const auto * other = p2p ? p->p2p[rank].as<const uint2>() : reinterpret_cast<const uint2 *>(p->ll_buf[1-rank].dev);
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
        ctx->ar_hc_graph = next[rank];
    }
    GGML_LOG_DEBUG("%s: captured LL collective + HC_POST, %lld values, %s wire, %s inbox\n", __func__,
                   (long long) count, bf16 ? "bf16" : "f32", p2p ? "peer" : "host");
    return true;
}

bool ggml_cuda_ar_allreduce(
        ggml_cuda_ar_pipeline * p,
        ggml_backend_t        * backends,
        ggml_tensor           ** tensors,
        const ggml_cgraph * const * next) {
    GGML_ASSERT(p != nullptr);

    const int n = p->n_devices;
    GGML_ASSERT(n >= 2 && n <= GGML_CUDA_MAX_DEVICES);

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

    p->n_reduced_all++;
    if (ggml_cuda_ar_try_hc_post(p, backends, tensors, next, use_bf16, use_copy_engine)) { return true; }

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

static ggml_cuda_ar_owner_layout ggml_cuda_ar_owner_layout_of(ggml_tensor ** tensors, int n_ranks) {
    ggml_cuda_ar_owner_layout owner;
    for (int axis = 0; axis < GGML_MAX_DIMS; ++axis) {
        ggml_cuda_ar_owner_layout candidate;
        candidate.valid = true;
        int64_t total = 0;
        const int64_t words_per_count = tensors[0]->nb[axis] / sizeof(uint32_t);
        for (int rank = 0; rank < n_ranks; ++rank) {
            const ggml_tensor * tensor = tensors[rank];
            if (tensor->op != GGML_OP_PAD || (tensor->flags & GGML_TENSOR_FLAG_META) == 0 ||
                    !ggml_is_contiguous(tensor) || tensor->nb[axis] % sizeof(uint32_t) != 0) {
                candidate.valid = false;
                break;
            }
            const int64_t before = ggml_get_op_params_i32(tensor, 2*axis + 0);
            const int64_t after = ggml_get_op_params_i32(tensor, 2*axis + 1);
            const int64_t count = tensor->ne[axis] - before - after;
            if (before < 0 || after < 0 || count < 0 || before != total) {
                candidate.valid = false;
                break;
            }
            candidate.begin[rank] = total*words_per_count;
            candidate.len[rank] = count*words_per_count;
            total += count;
        }
        if (candidate.valid && total == tensors[0]->ne[axis] &&
                total*words_per_count == (int64_t) (ggml_nbytes(tensors[0]) / sizeof(uint32_t))) {
            if (owner.valid) {
                return {}; // ambiguous axes must retain the generic bitwise gather
            }
            owner = candidate;
        }
    }
    return owner;
}

bool ggml_cuda_ar_allgather(
        ggml_cuda_ar_pipeline * p,
        ggml_backend_t        * backends,
        ggml_tensor           ** tensors) {
    GGML_ASSERT(p != nullptr);
    GGML_ASSERT(p->n_devices >= 2 && p->n_devices <= GGML_CUDA_MAX_DEVICES);

    const size_t nbytes = ggml_nbytes(tensors[0]);
    GGML_ASSERT(nbytes % sizeof(uint32_t) == 0);

    void * data[GGML_CUDA_MAX_DEVICES] = {};
    bool   zero_first[GGML_CUDA_MAX_DEVICES] = {};
    for (int i = 0; i < p->n_devices; ++i) {
        GGML_ASSERT(ggml_nbytes(tensors[i]) == nbytes);
        data[i]       = tensors[i]->data;
        zero_first[i] = (tensors[i]->flags & GGML_TENSOR_FLAG_COMPUTE) == 0; // a device with no part brings none
    }
    // Device counters and peer inboxes keep large gathers capturable; unsupported peers retain host staging.
    ggml_cuda_ar_owner_layout owner = ggml_cuda_ar_owner_layout_of(tensors, p->n_devices);
    ggml_cuda_ar_launch_kernels<uint32_t, uint32_t>(p, backends, data, zero_first, (int64_t) (nbytes / sizeof(uint32_t)), &owner);
    return true;
}

// Queues the reduction to owners of ne elements of T_dst at data[i], one chunked launch a chunk on the chunked kernel's
// staging, as ggml_cuda_ar_launch_kernels queues the reduction to every rank: card to card through the gather inboxes
// where the cards map each other's memory, else through host memory; zero_first[i]: data[i] is zeroed first.
template <typename T_dst, typename T_wire>
static void ggml_cuda_ar_launch_owner_reduce(
        ggml_cuda_ar_pipeline * p, ggml_backend_t * backends, void * const * data, const bool * zero_first, int64_t ne,
        const ggml_cuda_ar_owner_layout & owner) {
    const size_t max_chunk_elems = (p->gather_peer ? GGML_CUDA_AR_PEER_MAX_BYTES : p->buf_bytes) / sizeof(T_wire);
    (p->gather_peer ? p->n_chunked_peer : p->n_chunked_host)++;
    for (int64_t chunk_start = 0; chunk_start < ne; chunk_start += (int64_t) max_chunk_elems) {
        const int64_t chunk_elems = std::min(ne - chunk_start, (int64_t) max_chunk_elems);
        const ggml_cuda_ar_owner_layout chunk_owner = ggml_cuda_ar_chunk_owner(owner, chunk_start, chunk_elems, p->n_devices);
        for (int i = 0; i < p->n_devices; ++i) {
            ggml_cuda_set_device(p->devices[i]);
            auto * cuda_ctx = static_cast<ggml_backend_cuda_context *>(backends[i]->context);
            GGML_ASSERT(cuda_ctx->device == p->devices[i]);
            cudaStream_t stream = cuda_ctx->stream();
            T_dst * chunk = static_cast<T_dst *>(data[i]) + chunk_start;
            if (zero_first[i]) {
                CUDA_CHECK(cudaMemsetAsync(chunk, 0, (size_t) chunk_elems * sizeof(T_dst), stream));
            }
            ggml_cuda_ar_rank_ptrs<T_wire> sends, receives;
            if (p->gather_peer) {
                ggml_cuda_ar_peer_inboxes<T_wire>(p, i, max_chunk_elems, sends, receives);
            } else {
                // each rank stages what it sends in its own host buffer, at the element's place, and reads its peers'
                for (int rank = 0; rank < p->n_devices; ++rank) {
                    sends.ptr[rank] = reinterpret_cast<T_wire *>(p->host_buf[i].dev);
                    receives.ptr[rank] = reinterpret_cast<T_wire *>(p->host_buf[rank].dev);
                }
            }
            ggml_cuda_ar_owner_reduce_kernel_n<T_dst, T_wire><<<GGML_CUDA_AR_KERNEL_BLOCKS, 256, 0, stream>>>(
                chunk, sends, receives, max_chunk_elems, (int) chunk_elems, chunk_owner, reinterpret_cast<int *>(p->arrival.dev),
                i, p->n_devices, p->counter[i]);
            CUDA_CHECK(cudaGetLastError());
        }
    }
}

bool ggml_cuda_ar_reduce_owned(
        ggml_cuda_ar_pipeline * p,
        ggml_backend_t        * backends,
        ggml_tensor           ** tensors,
        const int64_t         * begin,
        const int64_t         * len) {
    GGML_ASSERT(p != nullptr);
    const int n = p->n_devices;
    // Two ranks take kernels of their own. The LL kernels pace a rank by the packets it receives, so one that owns
    // nothing would run ahead into a slot its peers still read; and the copy engine's shares go whole. Each of those
    // reduces to every rank.
    if (!p->reduce_owners || n <= 2) {
        return false;
    }
    const ggml_type input_type = tensors[0]->type;
    if (input_type != GGML_TYPE_F32 && input_type != GGML_TYPE_F16 && input_type != GGML_TYPE_BF16) {
        return false;
    }
    const int64_t ne = ggml_nelements(tensors[0]);
    if (ne <= 0) {
        return false;
    }
    const ggml_cuda_ar_route route = ggml_cuda_ar_route_of(p, tensors[0]);
    if (route.use_copy_engine || route.nbytes <= p->p2p_max_bytes || route.nbytes <= p->ll_max_bytes) {
        return false;
    }
    ggml_cuda_ar_owner_layout owner;
    owner.valid = true;
    for (int rank = 0; rank < n; ++rank) {
        if (begin[rank] < 0 || len[rank] < 0 || begin[rank] + len[rank] > ne) {
            return false;
        }
        for (int other = 0; other < rank; ++other) {
            if (len[rank] > 0 && len[other] > 0 &&
                    begin[rank] < begin[other] + len[other] && begin[other] < begin[rank] + len[rank]) {
                return false; // an element two ranks own would be summed twice
            }
        }
        owner.begin[rank] = begin[rank];
        owner.len[rank] = len[rank];
    }
    p->n_reduced_owned++;

    // As in the AllReduce, an inactive rank's share is zeros.
    void * data[GGML_CUDA_MAX_DEVICES] = {};
    bool   zero_first[GGML_CUDA_MAX_DEVICES] = {};
    for (int i = 0; i < n; ++i) {
        data[i]       = tensors[i]->data;
        zero_first[i] = (tensors[i]->flags & GGML_TENSOR_FLAG_COMPUTE) == 0;
    }
    if (route.use_bf16) {
        ggml_cuda_ar_launch_owner_reduce<float, nv_bfloat16>(p, backends, data, zero_first, ne, owner);
    } else {
        switch (input_type) {
            case GGML_TYPE_F32:  ggml_cuda_ar_launch_owner_reduce<float,       float>      (p, backends, data, zero_first, ne, owner); break;
            case GGML_TYPE_F16:  ggml_cuda_ar_launch_owner_reduce<half,        half>       (p, backends, data, zero_first, ne, owner); break;
            case GGML_TYPE_BF16: ggml_cuda_ar_launch_owner_reduce<nv_bfloat16, nv_bfloat16>(p, backends, data, zero_first, ne, owner); break;
            default: GGML_ABORT("unreachable");
        }
    }
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
bool ggml_cuda_ar_allreduce(ggml_cuda_ar_pipeline *, ggml_backend_t *, ggml_tensor **, const ggml_cgraph * const *) {
    return false;
}
bool ggml_cuda_ar_allgather(ggml_cuda_ar_pipeline *, ggml_backend_t *, ggml_tensor **) {
    return false;
}
bool ggml_cuda_ar_reduce_owned(ggml_cuda_ar_pipeline *, ggml_backend_t *, ggml_tensor **, const int64_t *, const int64_t *) {
    return false;
}

#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
