#include "l2-issue.cuh"

#include <algorithm>
#include <cmath>

#if defined(CUDART_VERSION) && CUDART_VERSION >= 12000 && !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#include <cuda.h>
#include <cudaTypedefs.h>
#include <map>
#include <mutex>
#include <tuple>
#endif

// A request is an 8 KB piece: one uniform-datapath instruction a piece, and inside the size that was measured; pieces go to the
// blocks in turn, each block's one thread requesting its next when the grid's position in the ranges is due at the rate.
#define L2_ISSUE_PIECE 8192

static __device__ __forceinline__ uint64_t l2_issue_now() {
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

// A block reads stop every L2_ISSUE_STOP_EVERY pieces and compares what it read at the next read: the load's round trip
// to L2 overlaps the pieces between, and a block stops within 2 * L2_ISSUE_STOP_EVERY pieces of a change (~1.6 us at 40
// GB/s a block)
#define L2_ISSUE_STOP_EVERY 4

static __device__ __forceinline__ unsigned int l2_issue_stop_read(const unsigned int * stop) {
    unsigned int v;
    asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(stop) : "memory");
    return v;
}

static __global__ void l2_issue_paced(const ggml_cuda_l2_ranges r, const float ns_per_byte, const unsigned int * stop) {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_HOPPER
    const uint64_t     t0    = l2_issue_now();
    const int64_t      step  = (int64_t) gridDim.x * L2_ISSUE_PIECE;
    const unsigned int start = stop != nullptr ? l2_issue_stop_read(stop) : 0;
    unsigned int       seen  = start;
    int                k     = 0; // the range holding lo
    int64_t            base  = 0; // its first byte's place in the ranges laid end to end
    for (int64_t off = (int64_t) blockIdx.x * L2_ISSUE_PIECE, n = 0; off < r.total; off += step, ++n) {
        const uint64_t due = (uint64_t) ((float) off * ns_per_byte);
        while (l2_issue_now() - t0 < due) {
        }
        if (stop != nullptr && n % L2_ISSUE_STOP_EVERY == 0) {
            if (seen != start) {
                return; // a launch that keeps DRAM busy has started, and a request now would take DRAM from it
            }
            seen = l2_issue_stop_read(stop);
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
    GGML_UNUSED_VARS(r, ns_per_byte, stop);
#endif
}

void ggml_cuda_l2_issue(const ggml_cuda_l2_ranges & r, const double rate_gbs, const int nsm, const unsigned int * stop,
                        cudaStream_t stream) {
    if (r.total <= 0 || rate_gbs <= 0.0) {
        return;
    }
    // a thread requests ~66 GB/s of bulk prefetches at most (measured on an RTX 5080), so one block for each 40 GB/s of the
    // rate, and no more blocks than pieces
    const int64_t pieces = (r.total + L2_ISSUE_PIECE - 1) / L2_ISSUE_PIECE;
    const int     nb     = (int) std::min<int64_t>({ (int64_t) nsm, pieces, (int64_t) std::ceil(rate_gbs / 40.0) });
    l2_issue_paced<<<std::max(nb, 1), 1, 0, stream>>>(r, (float) (1.0 / rate_gbs), stop);
    CUDA_CHECK(cudaGetLastError());
}

#if defined(CUDART_VERSION) && CUDART_VERSION >= 12000 && !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

static constexpr int L2_WARM_MAX_EXPERTS = 512;
static constexpr int L2_WARM_MASK_WORDS  = L2_WARM_MAX_EXPERTS / 32;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_HOPPER
// Tensor-map L2 prefetch: no SMEM destination, no mbarrier, no evict_last hint.
static __device__ __forceinline__ void l2_warm_tma_prefetch(const CUtensorMap * map, const int x, const int y) {
    asm volatile(
        "cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];"
        :: "l"((uint64_t) map), "r"(x), "r"(y)
        : "memory");
}
#endif

static __global__ __launch_bounds__(WARP_SIZE, 1)
void l2_warm_experts_tma_kernel(
        const __grid_constant__ CUtensorMap map,
        const int32_t * ids, const int nentries, const int n_used, const int ids_stride,
        const int32_t first, const int32_t count,
        const int n_row_tiles, const int64_t expert_chunks, const int64_t half_chunks,
        const int box_rows, const int tile_bytes, const bool paired_halves,
        const int consumer_max_grid, const int max_tiles_cap,
        const float ns_per_byte, const unsigned int * stop) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_HOPPER
    __shared__ uint32_t mask[L2_WARM_MASK_WORDS];
    __shared__ uint16_t experts[L2_WARM_MAX_EXPERTS];

    const int lane = threadIdx.x;
    if (lane < L2_WARM_MASK_WORDS) {
        mask[lane] = 0;
    }
    __syncwarp();

    for (int i = lane; i < nentries; i += WARP_SIZE) {
        const int32_t e  = __ldcg(&ids[(i / n_used) * ids_stride + (i % n_used)]);
        const int32_t le = e - first;
        if (le >= 0 && le < count) {
            atomicOr(&mask[le >> 5], 1u << (le & 31));
        }
    }
    __syncwarp();

    uint32_t w   = lane < L2_WARM_MASK_WORDS ? mask[lane] : 0u;
    int      cnt = __popc(w);
    int      inc = cnt;
#pragma unroll
    for (int off = 1; off < L2_WARM_MASK_WORDS; off <<= 1) {
        const int up = __shfl_up_sync(0xFFFFFFFFu, inc, off);
        if (lane >= off) {
            inc += up;
        }
    }
    int base = inc - cnt;
    while (w != 0u) {
        const int bit = __ffs(w) - 1;
        experts[base++] = (uint16_t) (lane * 32 + bit);
        w &= w - 1u;
    }
    const int n_unique = __shfl_sync(0xFFFFFFFFu, inc, L2_WARM_MASK_WORDS - 1);
    __syncwarp();

    if (lane != 0 || n_unique <= 0 || n_row_tiles <= 0 || max_tiles_cap <= 0) {
        return;
    }

    const int total_tiles = n_unique * n_row_tiles;
    const int cgrid       = min(consumer_max_grid, total_tiles);
    const int max_per_cta = (total_tiles + cgrid - 1) / cgrid;
    const int limit       = max_tiles_cap < total_tiles ? max_tiles_cap : cgrid * max_per_cta;
    const int64_t step_b  = paired_halves ? (int64_t) tile_bytes * 2 : (int64_t) tile_bytes;

    const uint64_t     t0    = l2_issue_now();
    const unsigned int start = stop != nullptr ? l2_issue_stop_read(stop) : 0u;
    unsigned int       seen  = start;

    for (int idx = (int) blockIdx.x, n = 0; idx < limit; idx += (int) gridDim.x, ++n) {
        const int wave = idx / cgrid;
        const int cta  = idx % cgrid;
        const int t_b  = (int) (((int64_t) total_tiles * cta) / cgrid);
        const int t_e  = (int) (((int64_t) total_tiles * (cta + 1)) / cgrid);
        const int t    = t_b + wave;
        if (t >= t_e) {
            continue;
        }
        const uint64_t due = (uint64_t) ((float) ((int64_t) idx * step_b) * ns_per_byte);
        while (l2_issue_now() - t0 < due) {
        }
        if (stop != nullptr && (n % L2_ISSUE_STOP_EVERY) == 0) {
            if (seen != start) {
                return;
            }
            seen = l2_issue_stop_read(stop);
        }
        const int u  = t / n_row_tiles;
        const int rt = t % n_row_tiles;
        const int e  = (int) experts[u];
        const int y0 = (int) ((int64_t) e * expert_chunks + (int64_t) rt * box_rows);
        l2_warm_tma_prefetch(&map, 0, y0);
        if (paired_halves) {
            l2_warm_tma_prefetch(&map, 0, (int) ((int64_t) y0 + half_chunks));
        }
    }
#else
    GGML_UNUSED_VARS(map, ids, nentries, n_used, ids_stride, first, count, n_row_tiles,
        expert_chunks, half_chunks, box_rows, tile_bytes, paired_halves,
        consumer_max_grid, max_tiles_cap, ns_per_byte, stop);
#endif
}

struct l2_warm_tma_geom {
    int     chunk_bytes   = 0;
    int     box_rows      = 0;
    int     tile_rows     = 0;
    int     tile_bytes    = 0;
    int64_t chunks        = 0;
    int64_t expert_chunks = 0;
    int64_t half_chunks   = 0;
    int     n_row_tiles   = 0;
};

static bool l2_warm_tma_geom_of(const ggml_tensor * w, const bool paired_halves, l2_warm_tma_geom & g) {
    if (w == nullptr || w->ne[2] < 2 || w->ne[2] > L2_WARM_MAX_EXPERTS || w->ne[3] != 1 ||
            !ggml_is_contiguous(w) || (uintptr_t) w->data % 16 != 0) {
        return false;
    }
    const int64_t row_bytes = (int64_t) w->nb[1];
    const int64_t eff_rows  = paired_halves ? w->ne[1] / 2 : w->ne[1];
    if (row_bytes <= 0 || eff_rows < 1 || (paired_halves && (w->ne[1] & 1) != 0)) {
        return false;
    }
    if (w->type == GGML_TYPE_Q3_K && row_bytes == 1980 && eff_rows % 16 == 0 && (w->nb[2] % 880) == 0) {
        g.chunk_bytes   = 880;
        g.tile_rows     = 16;
        g.box_rows      = 36;
        g.tile_bytes    = 31680;
        g.chunks        = (int64_t) w->ne[2] * (int64_t) w->nb[2] / g.chunk_bytes;
        g.expert_chunks = (int64_t) w->nb[2] / g.chunk_bytes;
        g.half_chunks   = paired_halves ? (eff_rows * row_bytes) / g.chunk_bytes : 0;
        g.n_row_tiles   = (int) (eff_rows / g.tile_rows);
        return !paired_halves || (eff_rows * row_bytes) % g.chunk_bytes == 0;
    }
    if (row_bytes % 16 == 0 && w->nb[2] % row_bytes == 0) {
        int chunk = (int) row_bytes;
        if (chunk > 2048) {
            chunk = 2048;
            while (chunk >= 16 && (row_bytes % chunk != 0 || (w->nb[2] % chunk) != 0)) {
                chunk -= 16;
            }
            if (chunk < 16) {
                return false;
            }
        }
        const int chunks_per_row = (int) (row_bytes / chunk);
        int tr = 16;
        while (tr > 1 && (eff_rows % tr != 0 || tr * chunks_per_row > 256)) {
            tr >>= 1;
        }
        if (eff_rows % tr != 0 || tr * chunks_per_row > 256) {
            return false;
        }
        g.chunk_bytes   = chunk;
        g.tile_rows     = tr;
        g.box_rows      = tr * chunks_per_row;
        g.tile_bytes    = tr * (int) row_bytes;
        g.chunks        = (int64_t) w->ne[2] * (int64_t) w->nb[2] / g.chunk_bytes;
        g.expert_chunks = (int64_t) w->nb[2] / g.chunk_bytes;
        g.half_chunks   = paired_halves ? (eff_rows * row_bytes) / g.chunk_bytes : 0;
        g.n_row_tiles   = (int) (eff_rows / g.tile_rows);
        return true;
    }
    return false;
}

static const CUtensorMap * l2_warm_tma_map_of(const int device, const void * base, const int chunk_bytes,
                                              const int box_rows, const int64_t chunks) {
    static std::mutex mutex;
    static std::map<std::tuple<int, const void *, int, int, int64_t>, CUtensorMap> maps;
    std::lock_guard<std::mutex> lock(mutex);

    const auto key = std::make_tuple(device, base, chunk_bytes, box_rows, chunks);
    const auto it  = maps.find(key);
    if (it != maps.end()) {
        return &it->second;
    }

    static const auto encode = [] {
        void * p = nullptr;
        cudaDriverEntryPointQueryResult query;
        const auto err = cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &p, 12000, cudaEnableDefault, &query);
        if (err != cudaSuccess || query != cudaDriverEntryPointSuccess) {
            (void) cudaGetLastError();
            return (PFN_cuTensorMapEncodeTiled_v12000) nullptr;
        }
        return (PFN_cuTensorMapEncodeTiled_v12000) p;
    }();
    if (!encode) {
        return nullptr;
    }

    CUtensorMap map;
    const cuuint64_t dims[]    = { (cuuint64_t) (chunk_bytes / 8), (cuuint64_t) chunks };
    const cuuint64_t strides[] = { (cuuint64_t) chunk_bytes };
    const cuuint32_t box[]     = { (cuuint32_t) (chunk_bytes / 8), (cuuint32_t) box_rows };
    const cuuint32_t esteps[]  = { 1, 1 };
    if (encode(&map, CU_TENSOR_MAP_DATA_TYPE_UINT64, 2, const_cast<void *>(base), dims, strides, box, esteps,
               CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
               CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) != CUDA_SUCCESS) {
        return nullptr;
    }
    return &maps.emplace(key, map).first->second;
}

bool ggml_cuda_l2_warm_experts_supported(const ggml_cuda_l2_expert_warm_args & args) {
    const auto & dev = ggml_cuda_info().devices[args.device];
    if (!GGML_CUDA_CC_IS_NVIDIA(dev.cc) || dev.cc < GGML_CUDA_CC_HOPPER ||
            ggml_cuda_highest_compiled_arch(dev.cc) < GGML_CUDA_CC_HOPPER ||
            args.dose_mib <= 0.0 || args.rate_gbs <= 0.0 || args.weight == nullptr || args.ids == nullptr) {
        return false;
    }
    if (args.ids->type != GGML_TYPE_I32 || args.ids->nb[0] != sizeof(int32_t) ||
            args.ids->ne[0] < 1 || args.ids->ne[1] < 1 || args.ids->ne[1] > 32) {
        return false;
    }
    const int32_t count = args.count > 0 ? args.count : (int32_t) args.weight->ne[2];
    if (count < 2 || count > L2_WARM_MAX_EXPERTS || count != (int32_t) args.weight->ne[2]) {
        return false;
    }
    l2_warm_tma_geom g;
    return l2_warm_tma_geom_of(args.weight, args.paired_halves, g);
}

bool ggml_cuda_l2_warm_experts(const ggml_cuda_l2_expert_warm_args & args, cudaStream_t stream) {
    if (!ggml_cuda_l2_warm_experts_supported(args)) {
        return false;
    }
    const auto & dev    = ggml_cuda_info().devices[args.device];
    const int32_t count = args.count > 0 ? args.count : (int32_t) args.weight->ne[2];
    l2_warm_tma_geom g;
    if (!l2_warm_tma_geom_of(args.weight, args.paired_halves, g)) {
        return false;
    }
    const CUtensorMap * map = l2_warm_tma_map_of(args.device, args.weight->data, g.chunk_bytes, g.box_rows, g.chunks);
    if (map == nullptr) {
        return false;
    }

    const int64_t step_bytes    = args.paired_halves ? (int64_t) g.tile_bytes * 2 : (int64_t) g.tile_bytes;
    const int64_t dose_bytes    = (int64_t) (args.dose_mib * 1024.0 * 1024.0);
    const int     max_tiles_cap = (int) std::max<int64_t>(1, dose_bytes / step_bytes);
    const int     n_used        = (int) args.ids->ne[0];
    const int     n_tokens      = (int) args.ids->ne[1];
    const int     nentries      = n_used * n_tokens;
    const int     ids_stride    = (int) (args.ids->nb[1] / sizeof(int32_t));
    const int     nb            = (int) std::min<int64_t>({ (int64_t) dev.nsm, (int64_t) max_tiles_cap,
                                                            (int64_t) std::ceil(args.rate_gbs / 40.0) });

    l2_warm_experts_tma_kernel<<<std::max(nb, 1), WARP_SIZE, 0, stream>>>(
        *map, (const int32_t *) args.ids->data, nentries, n_used, ids_stride, args.first, count,
        g.n_row_tiles, g.expert_chunks, g.half_chunks, g.box_rows, g.tile_bytes, args.paired_halves,
        dev.nsm, max_tiles_cap, (float) (1.0 / args.rate_gbs), args.stop);
    CUDA_CHECK(cudaGetLastError());
    return true;
}

#else

bool ggml_cuda_l2_warm_experts_supported(const ggml_cuda_l2_expert_warm_args & args) {
    GGML_UNUSED(args);
    return false;
}

bool ggml_cuda_l2_warm_experts(const ggml_cuda_l2_expert_warm_args & args, cudaStream_t stream) {
    GGML_UNUSED_VARS(args, stream);
    return false;
}

#endif
