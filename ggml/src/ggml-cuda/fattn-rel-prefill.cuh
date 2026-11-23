#pragma once

#include "common.cuh"
#include "fattn-rel-elem.cuh"
#include "ggml-attn-mask.h"
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#include "cp-async.cuh"
#include "mma.cuh"
#endif

static int fattn_rel_prefill_parts(const ggml_tensor * dst) {
    const auto * q = dst->src[0];
    const auto * k = dst->src[1];
    if (q->ne[1] >= 1024 && k->ne[1] <= 65536) {
        return 1;
    }
    return q->ne[1] <= 32 ? 128 : 8;
}

static constexpr int REL_PREFILL_RANGES = 32;
static constexpr int REL_PREFILL_MAX_HEADS = 64;

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#include <mma.h>

struct fattn_rel_prefill_ranges {
    int first, last, count;
    int3 runs[REL_PREFILL_RANGES]; // First logical position, last logical position, first physical cell.
};

static_assert(sizeof(fattn_rel_prefill_ranges) == (3 + REL_PREFILL_RANGES*3)*sizeof(int32_t), "range scratch layout");

static __device__ unsigned fattn_rel_prefill_union(const uint16_t * mask, size_t stride, int rows, int word) {
    unsigned bits = 0;
    for (int q = 0; q < rows; ++q) {
        bits |= mask[q*stride + word];
    }
    return bits;
}

static __global__ void fattn_rel_prefill_spans(const uint16_t * mask, size_t stride, const int * positions, int nq, int nk, fattn_rel_prefill_ranges * spans) {
    const int query = 16*blockIdx.x, rows = min(16, nq - query);
    mask += query*stride;
    __shared__ int first, last, n_start, n_end, starts[REL_PREFILL_RANGES], ends[REL_PREFILL_RANGES];
    if (threadIdx.x == 0) {
        first = nk;
        last = -1;
        n_start = n_end = 0;
    }
    __syncthreads();
    for (int word = threadIdx.x; word < nk/16; word += blockDim.x) {
        const unsigned bits = fattn_rel_prefill_union(mask, stride, rows, word);
        if (bits == 0) {
            continue;
        }
        const int base = word*16, lo = __ffs(bits) - 1, hi = 31 - __clz(bits);
        atomicMin(&first, base + lo);
        atomicMax(&last, base + hi);
        const unsigned before = word > 0 ? fattn_rel_prefill_union(mask, stride, rows, word - 1) >> 15 : 0;
        const unsigned after = base + 16 < nk ? (fattn_rel_prefill_union(mask, stride, rows, word + 1) & 1) << 15 : 0;
        unsigned start = bits & ~((bits << 1) | before), end = bits & ~((bits >> 1) | after);
        for (int col = lo; col <= hi; ++col) {
            if ((bits & (1u << col)) && !((start >> col) & 1u) && int64_t(positions[base + col - 1]) + 1 != positions[base + col]) {
                start |= 1u << col;
            }
            if ((bits & (1u << col)) && !((end >> col) & 1u) && int64_t(positions[base + col]) + 1 != positions[base + col + 1]) {
                end |= 1u << col;
            }
        }
        while (start) {
            const int at = atomicAdd(&n_start, 1);
            if (at < REL_PREFILL_RANGES) {
                starts[at] = base + __ffs(start) - 1;
            }
            start &= start - 1;
        }
        while (end) {
            const int at = atomicAdd(&n_end, 1);
            if (at < REL_PREFILL_RANGES) {
                ends[at] = base + __ffs(end) - 1;
            }
            end &= end - 1;
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        fattn_rel_prefill_ranges & span = spans[blockIdx.x];
        span.first = first;
        span.last = last;
        span.count = 0;
        if (n_start == 0 || n_start > REL_PREFILL_RANGES || n_start != n_end) {
            return;
        }
        int lo = INT_MAX, hi = INT_MIN;
        for (int i = 0; i < n_start; ++i) {
            int stop = last;
            for (int j = 0; j < n_end; ++j) {
                if (ends[j] >= starts[i]) {
                    stop = min(stop, ends[j]);
                }
            }
            const int3 run = make_int3(positions[starts[i]], positions[stop], starts[i]);
            if (run.x < 0 || run.y < run.x) {
                return;
            }
            for (int j = 0; j < i; ++j) {
                if (!(run.y < span.runs[j].x || span.runs[j].y < run.x)) {
                    return;
                }
            }
            span.runs[i] = run;
            lo = min(lo, run.x);
            hi = max(hi, run.y);
        }
        if (int64_t(hi) - lo + 1 <= nk && hi <= INT_MAX - 4096) {
            for (int i = 1; i < n_start; ++i) {
                const int3 cur = span.runs[i];
                int j = i;
                while (j > 0 && span.runs[j - 1].x > cur.x) {
                    span.runs[j] = span.runs[j - 1];
                    --j;
                }
                span.runs[j] = cur;
            }
            span.first = lo;
            span.last = hi;
            span.count = n_start;
        }
    }
}

template<int dim>
static __global__ void fattn_rel_prefill_tiles(
        const char * q, const char * k, const char * v, const uint16_t * mask, const float * table,
        const int * q_pos, const int * k_pos, const float * sinks, float * partial, const fattn_rel_prefill_ranges * spans,
        float scale, int nq, int nh, int n_dist, int gqa, size_t mask_stride,
        size_t q_row, size_t q_head, size_t k_row, size_t k_head, size_t v_row, size_t v_head,
        const char * hot_kv, size_t hot_row, size_t hot_head, const int32_t * hot_map) {
#if __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
    namespace wmma = nvcuda::wmma;
    using A = wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major>;
    using B_key = wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major>;
    using B_value = wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major>;
    using C = wmma::fragment<wmma::accumulator, 16, 16, 16, float>;

    const int parts = gridDim.z;
    const int lane = threadIdx.x, head = blockIdx.y, query = 16*blockIdx.x;
    const int rows = min(16, nq - query);
    k += (head/gqa)*k_head;
    v += (head/gqa)*v_head;
    if (hot_kv) {
        hot_kv += (head/gqa)*hot_head;
    }
    __shared__ __align__(32) half panel[16][dim], weights[16][16];
    __shared__ __align__(32) float scores[16][16];
    for (int i = lane; i < 16*dim; i += 32) {
        const int row = i/dim, col = i%dim;
        const float value = row < rows ? ((const float *) (q + (query + row)*q_row + head*q_head))[col] : 0.0f;
        panel[row][col] = __float2half(value*scale);
    }
    __syncwarp();
    A queries[dim/16];
    C values[dim/16];
#pragma unroll
    for (int col = 0; col < dim; col += 16) {
        wmma::load_matrix_sync(queries[col/16], &panel[0][col], dim);
        wmma::fill_fragment(values[col/16], 0.0f);
    }
    __syncwarp();
    float maximum = lane < rows ? sinks[int64_t(query + lane)*nh + head] : 0.0f;
    float denominator = blockIdx.z == 0 ? 1.0f : 0.0f;
    const fattn_rel_prefill_ranges & span = spans[blockIdx.x];
    const int first = span.first, last = span.last, count = span.count;
    __shared__ int cells[16];

    // Anchor tiles and split-K ordering to visible cells so a leading allocation offset leaves the arithmetic unchanged.
    for (int logical = first + 16*blockIdx.z; logical <= last; logical += 16*parts) {
        int cell = logical, mapped = lane < 16 && logical + lane <= last ? logical + lane : -1;
        bool pad = logical + 15 > last;
        if (count) {
            mapped = -1;
            cell = -1;
            for (int i = 0; i < count; ++i) {
                const int3 run = span.runs[i];
                if (logical + lane >= run.x && logical + lane <= run.y) {
                    mapped = run.z + (logical + lane - run.x);
                }
                if (logical >= run.x && logical + 15 <= run.y) {
                    cell = run.z + (logical - run.x);
                }
            }
            pad = cell < 0;
        }
        if (hot_map) {
            pad = true;
        }
        if (lane < 16) {
            cells[lane] = mapped;
        }
        __syncwarp();
        if (pad) {
            for (int i = lane; i < 16*dim; i += 32) {
                const int row = cells[i/dim];
                const int32_t hslot = (hot_map && row >= 0) ? hot_map[row] : -1;
                const half * src = hslot >= 0 ? (const half *) (hot_kv + (2*int64_t(hslot) + 0)*hot_row)
                    : (row >= 0 ? (const half *) (k + int64_t(row)*k_row) : nullptr);
                panel[i/dim][i%dim] = src ? src[i%dim] : __float2half(0.0f);
            }
            __syncwarp();
        }
        C dots;
        wmma::fill_fragment(dots, 0.0f);
#pragma unroll
        for (int col = 0; col < dim; col += 16) {
            B_key key;
            wmma::load_matrix_sync(key, pad ? &panel[0][col] : (const half *) (k + cell*k_row) + col, pad ? dim : k_row/sizeof(half));
            wmma::mma_sync(dots, queries[col/16], key, dots);
        }
        wmma::store_matrix_sync(&scores[0][0], dots, 16, wmma::mem_row_major);
        __syncwarp();
        if (lane < 16) {
            const int row = query + lane;
            float next = maximum;
#pragma unroll
            for (int col = 0; col < 16; ++col) {
                const int key = cells[col];
                float score = -INFINITY;
                if (lane < rows && key >= 0 && (mask[row*mask_stride + key/16] & (1u << (key%16)))) {
                    const int64_t dist = int64_t(q_pos[row]) - k_pos[key];
                    score = scores[lane][col] + (dist >= 0 && dist < n_dist ? table[(int64_t(row)*nh + head)*n_dist + dist] : 0.0f);
                }
                scores[lane][col] = score;
                next = fmaxf(next, score);
            }
            const float rescale = maximum == next ? 1.0f : expf(maximum - next);
            denominator *= rescale;
#pragma unroll
            for (int col = 0; col < 16; ++col) {
                const float weight = scores[lane][col] == -INFINITY ? 0.0f : expf(scores[lane][col] - next);
                denominator += weight;
                weights[lane][col] = __float2half(weight);
                scores[lane][col] = rescale;
            }
            maximum = next;
        }
        if (pad) {
            for (int i = lane; i < 16*dim; i += 32) {
                const int row = cells[i/dim];
                const int32_t hslot = (hot_map && row >= 0) ? hot_map[row] : -1;
                const half * src = hslot >= 0 ? (const half *) (hot_kv + (2*int64_t(hslot) + 1)*hot_row)
                    : (row >= 0 ? (const half *) (v + int64_t(row)*v_row) : nullptr);
                panel[i/dim][i%dim] = src ? src[i%dim] : __float2half(0.0f);
            }
        }
        __syncwarp();
        A weight;
        C rescale;
        wmma::load_matrix_sync(weight, &weights[0][0], 16);
        wmma::load_matrix_sync(rescale, &scores[0][0], 16, wmma::mem_row_major);
#pragma unroll
        for (int col = 0; col < dim; col += 16) {
            B_value value;
            wmma::load_matrix_sync(value, pad ? &panel[0][col] : (const half *) (v + cell*v_row) + col, pad ? dim : v_row/sizeof(half));
            C & accum = values[col/16];
#pragma unroll
            for (int i = 0; i < C::num_elements; ++i) {
                accum.x[i] *= rescale.x[i];
            }
            wmma::mma_sync(accum, weight, value, accum);
        }
    }
    if (parts == 1) {
        float * out = partial + (int64_t(query)*nh + head)*dim;
        const float inv_d = lane < rows ? 1.0f/denominator : 0.0f;
#pragma unroll
        for (int col = 0; col < dim; col += 16) {
            wmma::store_matrix_sync(&scores[0][0], values[col/16], 16, wmma::mem_row_major);
            __syncwarp();
#pragma unroll
            for (int i = lane; i < 256; i += 32) {
                const int r = i/16;
                const float inv_r = __shfl_sync(0xffffffff, inv_d, r);
                if (r < rows) {
                    out[int64_t(r)*nh*dim + col + i%16] = scores[r][i%16]*inv_r;
                }
            }
            __syncwarp();
        }
        return;
    }
    partial += ((int64_t(query)*nh + head)*parts + blockIdx.z)*(dim + 2);
    const int stride = nh*parts*(dim + 2);
    if (lane < rows) {
        partial[lane*stride] = maximum;
        partial[lane*stride + 1] = denominator;
    }
#pragma unroll
    for (int col = 0; col < dim; col += 16) {
        wmma::store_matrix_sync(&scores[0][0], values[col/16], 16, wmma::mem_row_major);
        __syncwarp();
#pragma unroll
        for (int i = lane; i < 256; i += 32) {
            if (i/16 < rows) {
                partial[(i/16)*stride + 2 + col + i%16] = scores[i/16][i%16];
            }
        }
        __syncwarp();
    }
#else
    NO_DEVICE_CODE;
#endif
}

template<int dim>
static __global__ void fattn_rel_prefill_merge(const float * partial, float * out, int parts) {
    const int row = blockIdx.x, col = threadIdx.x;
    partial += int64_t(row)*parts*(dim + 2);
    float maximum = -INFINITY;
    for (int p = 0; p < parts; ++p) {
        maximum = fmaxf(maximum, partial[p*(dim + 2)]);
    }
    float sum = 0.0f, value = 0.0f;
    for (int p = 0; p < parts; ++p) {
        const float * src = partial + p*(dim + 2);
        const float scale = src[0] == maximum ? 1.0f : expf(src[0] - maximum);
        sum += src[1]*scale;
        value += src[2 + col]*scale;
    }
    out[int64_t(row)*dim + col] = value/sum;
}
#endif

// GQA-packed relative prefill: one CTA runs all gqa query heads of a KV head (one warp per head), staging K and V tiles
// into padded double-buffered shared memory while computing PTX mma.sync + register exp2f online softmax.
template<int dim, ggml_type kv_type = GGML_TYPE_F16, int max_warps = 8>
__launch_bounds__(max_warps*WARP_SIZE, max_warps <= 6 ? 2 : 1)
static __global__ void fattn_rel_prefill_gqa_packed_tiles(
        const char * q, const char * k, const char * v, const uint16_t * mask, const float * table,
        const int * q_pos, const int * k_pos, const float * sinks, float * out, const fattn_rel_prefill_ranges * spans,
        float scale, int nq, int nh, int n_dist, int gqa, size_t mask_stride,
        size_t q_row, size_t q_head, size_t k_row, size_t k_head, size_t v_row, size_t v_head,
        const char * hot_kv, size_t hot_row, size_t hot_head, const int32_t * hot_map) {
#if defined(AMPERE_MMA_AVAILABLE) && defined(CP_ASYNC_AVAILABLE)
    using namespace ggml_cuda_mma;
    constexpr float LOG2E = 1.4426950408889634f;
    constexpr float LN2   = 0.6931471805599453f;
    constexpr int row_i4     = dim/8 + 1;
    constexpr int chunks_row = dim/8;

    const int parts = gridDim.z;
    const int lane = threadIdx.x, warp = threadIdx.y;
    const int tid = warp*WARP_SIZE + lane, nth = WARP_SIZE*int(blockDim.y);
    const int query = 16*blockIdx.x, kvh = blockIdx.y;
    const int rows = min(16, nq - query);
    const int head = kvh*gqa + warp;
    k += size_t(kvh)*k_head;
    v += size_t(kvh)*v_head;
    if (hot_kv) {
        hot_kv += size_t(kvh)*hot_head;
    }

    const int r0 = lane >> 2, r1 = 8 + (lane >> 2), d0 = 2*(lane & 3);
    const bool valid_q[2] = {r0 < rows, r1 < rows};
    tile<16, 8, half2> Q_A[dim/16];
    tile<16, 16, float> O[dim/16];
    {
        const float q_scale = scale*LOG2E;
        const float * qp0 = (const float *) (q + size_t(query + (valid_q[0] ? r0 : 0))*q_row + size_t(head)*q_head);
        const float * qp1 = (const float *) (q + size_t(query + (valid_q[1] ? r1 : 0))*q_row + size_t(head)*q_head);
#pragma unroll
        for (int c = 0; c < dim/16; ++c) {
            Q_A[c].x[0] = valid_q[0] ? make_half2(qp0[16*c + d0]*q_scale,     qp0[16*c + d0 + 1]*q_scale)     : make_half2(0.0f, 0.0f);
            Q_A[c].x[1] = valid_q[1] ? make_half2(qp1[16*c + d0]*q_scale,     qp1[16*c + d0 + 1]*q_scale)     : make_half2(0.0f, 0.0f);
            Q_A[c].x[2] = valid_q[0] ? make_half2(qp0[16*c + 8 + d0]*q_scale, qp0[16*c + 8 + d0 + 1]*q_scale) : make_half2(0.0f, 0.0f);
            Q_A[c].x[3] = valid_q[1] ? make_half2(qp1[16*c + 8 + d0]*q_scale, qp1[16*c + 8 + d0 + 1]*q_scale) : make_half2(0.0f, 0.0f);
        }
    }

    float maximum[2] = {
        valid_q[0] ? sinks[int64_t(query + r0)*nh + head]*LOG2E : 0.0f,
        valid_q[1] ? sinks[int64_t(query + r1)*nh + head]*LOG2E : 0.0f,
    };
    float denominator[2] = {
        (blockIdx.z == 0 && (lane & 3) == 0) ? 1.0f : 0.0f,
        (blockIdx.z == 0 && (lane & 3) == 0) ? 1.0f : 0.0f,
    };

    const fattn_rel_prefill_ranges & span = spans[blockIdx.x];
    const int first = span.first, last = span.last, count = span.count;

    auto mapped_cell = [&](int logical, int r) {
        const int pos = logical + r;
        if (!count) {
            return pos <= last ? pos : -1;
        }
        for (int i = 0; i < count; ++i) {
            const int3 run = span.runs[i];
            if (pos <= run.y) {
                return pos >= run.x ? run.z + (pos - run.x) : -1;
            }
        }
        return -1;
    };

    __shared__ __align__(16) int cells_sh[3][16];
    __shared__ __align__(16) int kpos_sh[3][16];
    __shared__ int  kpos_max_sh[3];
    __shared__ bool contig16_sh[3];
    __shared__ bool unmasked_sh[3];
    __shared__ int  qp_bounds_sh[2];
    extern __shared__ char smem_dyn[];
    int4 * kv_buf = (int4 *) smem_dyn;

    auto fill_meta = [&](int mslot, int logical) {
        const int cell = lane < 16 ? mapped_cell(logical, lane) : -1;
        const int kpos = cell >= 0 ? k_pos[cell] : INT_MIN;
        if (lane < 16) {
            cells_sh[mslot][lane] = cell;
            kpos_sh[mslot][lane]  = kpos;
        }
        int kp_max = kpos;
#pragma unroll
        for (int offset = 8; offset >= 1; offset >>= 1) {
            kp_max = max(kp_max, __shfl_xor_sync(0xffffffff, kp_max, offset, WARP_SIZE));
        }
        const int key0 = __shfl_sync(0xffffffff, cell, 0);
        const bool c16 = key0 >= 0 && (key0 & 15) == 0 &&
            __all_sync(0xffffffff, lane >= 16 || cell == key0 + lane);
        const bool unm = c16 &&
            __all_sync(0xffffffff, lane >= rows || mask[size_t(query + lane)*mask_stride + (key0 >> 4)] == 0xFFFFu);
        if (lane == 0) {
            kpos_max_sh[mslot] = kp_max;
            contig16_sh[mslot] = c16;
            unmasked_sh[mslot] = unm;
        }
    };

    using pending_t = std::conditional_t<kv_type == GGML_TYPE_F16, int, typename fattn_rel_elem<kv_type>::raw>;

    auto stage_issue = [&](int kv_slot, int mslot, pending_t & pend) {
        int4 * slot_base = kv_buf + size_t(kv_slot)*32*row_i4;
        if constexpr (kv_type == GGML_TYPE_F16) {
            GGML_UNUSED(pend);
            for (int c = tid; c < 32*chunks_row; c += nth) {
                const int panelrow = c/chunks_row, chunk_i4 = c%chunks_row;
                const int r = panelrow & 15;
                const int cell = cells_sh[mslot][r];
                int4 * dst4 = slot_base + panelrow*row_i4 + chunk_i4;
                if (cell >= 0) {
                    const bool is_v = panelrow >= 16;
                    const int32_t hslot = hot_map ? hot_map[cell] : -1;
                    const char * src = hslot >= 0
                        ? hot_kv + size_t(2*int64_t(hslot) + (is_v ? 1 : 0))*hot_row + chunk_i4*sizeof(int4)
                        : (is_v ? v : k) + size_t(cell)*(is_v ? v_row : k_row) + chunk_i4*sizeof(int4);
                    cp_async_cg_16<0>(ggml_cuda_cvta_generic_to_shared(dst4), src);
                } else {
                    *dst4 = make_int4(0, 0, 0, 0);
                }
            }
            cp_async_commit_group();
        } else {
            static_assert(dim == 128, "quantized prefill requires dim == 128");
            if (hot_map) {
                for (int c = tid; c < 32*chunks_row; c += nth) {
                    const int panelrow = c/chunks_row, chunk_i4 = c%chunks_row;
                    const int r = panelrow & 15;
                    const int cell = cells_sh[mslot][r];
                    int4 * dst4 = slot_base + panelrow*row_i4 + chunk_i4;
                    if (cell < 0) {
                        *dst4 = make_int4(0, 0, 0, 0);
                    } else {
                        const int32_t hslot = hot_map[cell];
                        if (hslot >= 0) {
                            const bool is_v = panelrow >= 16;
                            const char * src = hot_kv + size_t(2*int64_t(hslot) + (is_v ? 1 : 0))*hot_row + chunk_i4*sizeof(int4);
                            cp_async_cg_16<0>(ggml_cuda_cvta_generic_to_shared(dst4), src);
                        }
                    }
                }
                cp_async_commit_group();
            }
            if (tid < 64) {
                const int panelrow = tid >> 1, part = tid & 1;
                const int cell = cells_sh[mslot][panelrow & 15];
                if (cell >= 0 && (!hot_map || hot_map[cell] < 0)) {
                    const bool is_v = panelrow >= 16;
                    const char * src = (is_v ? v : k) + size_t(cell)*(is_v ? v_row : k_row);
                    pend = fattn_rel_elem<kv_type>::load(src, part);
                }
            }
        }
    };

    auto stage_commit = [&](int kv_slot, int mslot, pending_t & pend) {
        if constexpr (kv_type == GGML_TYPE_F16) {
            GGML_UNUSED_VARS(kv_slot, mslot, pend);
            cp_async_wait_group<0>();
        } else {
            if (hot_map) {
                cp_async_wait_group<0>();
            }
            if (tid < 64) {
                const int panelrow = tid >> 1, part = tid & 1;
                const int cell = cells_sh[mslot][panelrow & 15];
                int4 * dst4 = kv_buf + size_t(kv_slot)*32*row_i4 + panelrow*row_i4 + 8*part;
                if (cell >= 0) {
                    if (!hot_map || hot_map[cell] < 0) {
                        fattn_rel_elem<kv_type>::store(pend, part, dst4);
                    }
                } else if (!hot_map) {
#pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        dst4[j] = make_int4(0, 0, 0, 0);
                    }
                }
            }
        }
    };

    const int logical0 = first + 16*blockIdx.z;
    const int step = 16*parts;
    if (logical0 <= last) {
        if (warp == 0) {
            const int q_r = lane < rows ? q_pos[query + lane] : 0;
            int qmin = lane < rows ? q_r : INT_MAX;
            int qmax = lane < rows ? q_r : INT_MIN;
#pragma unroll
            for (int offset = 8; offset >= 1; offset >>= 1) {
                qmin = min(qmin, __shfl_xor_sync(0xffffffff, qmin, offset, WARP_SIZE));
                qmax = max(qmax, __shfl_xor_sync(0xffffffff, qmax, offset, WARP_SIZE));
            }
            if (lane == 0) {
                qp_bounds_sh[0] = qmin;
                qp_bounds_sh[1] = qmax;
            }
            fill_meta(0, logical0);
        } else if (warp == 1 && logical0 + step <= last) {
            fill_meta(1, logical0 + step);
        }
        __syncthreads();

        pending_t pend{};
        stage_issue(0, 0, pend);
        stage_commit(0, 0, pend);
        __syncthreads();

        const int qp_min = qp_bounds_sh[0];
        const int qp_max = qp_bounds_sh[1];
        GGML_UNUSED(qp_max);

        int it = 0;
        for (int logical = logical0; logical <= last; logical += step, ++it) {
            const bool has_next = logical + step <= last;
            const int cur = it & 1, nxt = (it + 1) & 1;
            const int m0 = it % 3, m1 = (it + 1) % 3, m2 = (it + 2) % 3;

            if (has_next) {
                stage_issue(nxt, m1, pend);
            }
            if (warp == 0 && last - logical >= 2*step) {
                fill_meta(m2, logical + 2*step);
            }

            const int4 * kt = kv_buf + size_t(cur)*32*row_i4;
            const int4 * vt = kt + 16*row_i4;

            tile<16, 16, float> S;
#pragma unroll
            for (int c = 0; c < dim/16; ++c) {
                tile<16, 8, half2> K_B;
                load_ldmatrix(K_B, (const half2 *) kt + 8*c, 4*row_i4);
                mma(S, Q_A[c], K_B);
            }

            const bool tile_unm = unmasked_sh[m0];
            const int  kp_max   = kpos_max_sh[m0];
            const bool tile_far = kp_max > INT_MIN && int64_t(qp_min) - int64_t(kp_max) >= n_dist;
            const int  sub      = lane & 3;

            if (tile_unm && tile_far) {
                if (!valid_q[0]) {
                    S.x[0] = S.x[1] = S.x[4] = S.x[5] = -INFINITY;
                }
                if (!valid_q[1]) {
                    S.x[2] = S.x[3] = S.x[6] = S.x[7] = -INFINITY;
                }
            } else if (tile_unm) {
                const int qp[2] = {valid_q[0] ? q_pos[query + r0] : 0, valid_q[1] ? q_pos[query + r1] : 0};
                const float * b0 = table + (int64_t(query + (valid_q[0] ? r0 : 0))*nh + head)*n_dist;
                const float * b1 = table + (int64_t(query + (valid_q[1] ? r1 : 0))*nh + head)*n_dist;
#pragma unroll
                for (int k_idx = 0; k_idx < 4; ++k_idx) {
                    const int col = ((k_idx & 2) << 2) + 2*sub + (k_idx & 1);
                    const int kp  = kpos_sh[m0][col];
#pragma unroll
                    for (int p = 0; p < 2; ++p) {
                        const int idx = (k_idx & 2)*2 + 2*p + (k_idx & 1);
                        if (valid_q[p]) {
                            const int64_t dist = int64_t(qp[p]) - kp;
                            if (uint64_t(dist) < uint64_t(n_dist)) {
                                S.x[idx] = fmaf((p ? b1 : b0)[dist], LOG2E, S.x[idx]);
                            }
                        } else {
                            S.x[idx] = -INFINITY;
                        }
                    }
                }
            } else if (tile_far && contig16_sh[m0]) {
                const size_t woff = size_t(cells_sh[m0][0] >> 4);
                const unsigned mw0 = valid_q[0] ? mask[size_t(query + r0)*mask_stride + woff] : 0u;
                const unsigned mw1 = valid_q[1] ? mask[size_t(query + r1)*mask_stride + woff] : 0u;
#pragma unroll
                for (int k_idx = 0; k_idx < 4; ++k_idx) {
                    const int col = ((k_idx & 2) << 2) + 2*sub + (k_idx & 1);
                    const int idx0 = (k_idx & 2)*2 + (k_idx & 1);
                    if (!((mw0 >> col) & 1u)) { S.x[idx0]     = -INFINITY; }
                    if (!((mw1 >> col) & 1u)) { S.x[idx0 + 2] = -INFINITY; }
                }
            } else {
                const int qp[2] = {valid_q[0] ? q_pos[query + r0] : 0, valid_q[1] ? q_pos[query + r1] : 0};
                const float * b0 = table + (int64_t(query + (valid_q[0] ? r0 : 0))*nh + head)*n_dist;
                const float * b1 = table + (int64_t(query + (valid_q[1] ? r1 : 0))*nh + head)*n_dist;
#pragma unroll
                for (int k_idx = 0; k_idx < 4; ++k_idx) {
                    const int col = ((k_idx & 2) << 2) + 2*sub + (k_idx & 1);
                    const int key = cells_sh[m0][col];
                    const int kp  = tile_far ? 0 : kpos_sh[m0][col];
#pragma unroll
                    for (int p = 0; p < 2; ++p) {
                        const int r = p ? r1 : r0;
                        const int idx = (k_idx & 2)*2 + 2*p + (k_idx & 1);
                        const bool vis = valid_q[p] && key >= 0 &&
                            (mask[size_t(query + r)*mask_stride + (key >> 4)] & (1u << (key & 15)));
                        if (!vis) {
                            S.x[idx] = -INFINITY;
                        } else if (!tile_far) {
                            const int64_t dist = int64_t(qp[p]) - kp;
                            if (uint64_t(dist) < uint64_t(n_dist)) {
                                S.x[idx] = fmaf((p ? b1 : b0)[dist], LOG2E, S.x[idx]);
                            }
                        }
                    }
                }
            }

            float rescale[2];
#pragma unroll
            for (int p = 0; p < 2; ++p) {
                const int b = 2*p;
                float m = fmaxf(fmaxf(S.x[b + 0], S.x[b + 1]), fmaxf(S.x[b + 4], S.x[b + 5]));
                m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, 1, WARP_SIZE));
                m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, 2, WARP_SIZE));
                m = fmaxf(maximum[p], m);
                rescale[p] = maximum[p] == m ? 1.0f : exp2f(maximum[p] - m);
                maximum[p] = m;
                S.x[b + 0] = S.x[b + 0] == -INFINITY ? 0.0f : exp2f(S.x[b + 0] - m);
                S.x[b + 1] = S.x[b + 1] == -INFINITY ? 0.0f : exp2f(S.x[b + 1] - m);
                S.x[b + 4] = S.x[b + 4] == -INFINITY ? 0.0f : exp2f(S.x[b + 4] - m);
                S.x[b + 5] = S.x[b + 5] == -INFINITY ? 0.0f : exp2f(S.x[b + 5] - m);
            }

            const tile<16, 8, half2> P_A = get_half2(S);
#pragma unroll
            for (int p = 0; p < 2; ++p) {
                denominator[p] = denominator[p]*rescale[p]
                    + __low2float(P_A.x[p]) + __high2float(P_A.x[p])
                    + __low2float(P_A.x[p + 2]) + __high2float(P_A.x[p + 2]);
            }
            if (__any_sync(0xffffffff, rescale[0] != 1.0f || rescale[1] != 1.0f)) {
#pragma unroll
                for (int c = 0; c < dim/16; ++c) {
#pragma unroll
                    for (int l = 0; l < 8; ++l) {
                        O[c].x[l] *= rescale[(l >> 1) & 1];
                    }
                }
            }
#pragma unroll
            for (int c = 0; c < dim/16; ++c) {
                tile<16, 8, half2> V_B;
                load_ldmatrix_trans(V_B, (const half2 *) vt + 8*c, 4*row_i4);
                mma(O[c], P_A, V_B);
            }

            if (has_next) {
                stage_commit(nxt, m1, pend);
                __syncthreads();
            }
        }
    }

#pragma unroll
    for (int p = 0; p < 2; ++p) {
        denominator[p] += __shfl_xor_sync(0xffffffff, denominator[p], 1, WARP_SIZE);
        denominator[p] += __shfl_xor_sync(0xffffffff, denominator[p], 2, WARP_SIZE);
    }

    if (parts == 1) {
#pragma unroll
        for (int p = 0; p < 2; ++p) {
            const int r = p ? r1 : r0;
            if (r < rows) {
                const float inv_d = 1.0f/denominator[p];
                float * dst_row = out + (int64_t(query + r)*nh + head)*dim;
                const int b = 2*p;
#pragma unroll
                for (int c = 0; c < dim/16; ++c) {
                    *(float2 *) (dst_row + 16*c + d0)     = make_float2(O[c].x[b + 0]*inv_d, O[c].x[b + 1]*inv_d);
                    *(float2 *) (dst_row + 16*c + 8 + d0) = make_float2(O[c].x[b + 4]*inv_d, O[c].x[b + 5]*inv_d);
                }
            }
        }
    } else {
#pragma unroll
        for (int p = 0; p < 2; ++p) {
            const int r = p ? r1 : r0;
            if (r < rows) {
                float * part_row = out + ((int64_t(query + r)*nh + head)*parts + blockIdx.z)*(dim + 2);
                if ((lane & 3) == 0) {
                    part_row[0] = maximum[p]*LN2;
                    part_row[1] = denominator[p];
                }
                const int b = 2*p;
#pragma unroll
                for (int c = 0; c < dim/16; ++c) {
                    *(float2 *) (part_row + 2 + 16*c + d0)     = make_float2(O[c].x[b + 0], O[c].x[b + 1]);
                    *(float2 *) (part_row + 2 + 16*c + 8 + d0) = make_float2(O[c].x[b + 4], O[c].x[b + 5]);
                }
            }
        }
    }
#else
    GGML_UNUSED_VARS(q, k, v, mask, table, q_pos, k_pos, sinks, out, spans, scale, nq, nh, n_dist, gqa, mask_stride,
        q_row, q_head, k_row, k_head, v_row, v_head, hot_kv, hot_row, hot_head, hot_map);
    NO_DEVICE_CODE;
#endif
}

static size_t fattn_rel_prefill_partial_offset(const ggml_tensor * dst, size_t output_bytes) {
    const size_t tiles = (dst->src[0]->ne[1] + 15)/16;
    return GGML_PAD(output_bytes, 128) + GGML_PAD(tiles*(3 + REL_PREFILL_RANGES*3)*sizeof(int32_t), 128);
}

static size_t fattn_rel_prefill_alloc_size(const ggml_tensor * dst) {
    const size_t rows = size_t(dst->ne[1])*dst->ne[2];
    const size_t output_bytes = rows*ggml_row_size(dst->type, dst->ne[0]);
    const int parts = fattn_rel_prefill_parts(dst);
    const size_t spans_end = fattn_rel_prefill_partial_offset(dst, output_bytes);
    return parts == 1 ? spans_end : spans_end + rows*parts*(2 + dst->ne[0])*sizeof(float);
}

// Dynamic shared memory for gqa_packed: two stages of 16 K + 16 V rows, each row padded by one int4.
static size_t fattn_rel_prefill_gqa_packed_smem(int64_t dim, int64_t gqa) {
    GGML_UNUSED(gqa);
    return size_t(2*32)*(dim/8 + 1)*sizeof(int4);
}

static void ggml_cuda_flash_attn_ext_rel_prefill(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    const auto * q = dst->src[0], * k = dst->src[1], * v = dst->src[2], * mask = dst->src[3];
    const int tiles = (q->ne[1] + 15)/16, parts = fattn_rel_prefill_parts(dst);
    char * data = (char *) dst->data;
    fattn_rel_prefill_ranges * spans = (fattn_rel_prefill_ranges *) (data + GGML_PAD(ggml_nbytes(dst), 128));
    float * partial = parts == 1
        ? (float *) dst->data
        : (float *) (data + fattn_rel_prefill_partial_offset(dst, ggml_nbytes(dst)));
    float scale;
    memcpy(&scale, dst->op_params, sizeof(scale));
    fattn_rel_prefill_spans<<<tiles, 128, 0, ctx.stream()>>>((const uint16_t *) mask->data, mask->nb[1]/2, (const int *) dst->src[8]->data, q->ne[1], k->ne[1], spans);

    const ggml_tensor * hot_kv = ggml_flash_attn_ext_get_hot_kv(dst);
    const char * hot_kv_d = hot_kv ? (const char *) hot_kv->data : nullptr;
    const size_t hot_row  = hot_kv ? hot_kv->nb[1] : 0;
    const size_t hot_head = hot_kv ? hot_kv->nb[2] : 0;
    const int32_t * hot_map = (hot_kv && dst->src[8] && dst->src[8]->ne[2] == 2)
        ? (const int32_t *) ((const char *) dst->src[8]->data + dst->src[8]->nb[2]) : nullptr;

    static const bool gqa_packed = ggml_env_switch("GGML_CUDA_FATTN_REL_PACKED_PREFILL");
    if (gqa_packed && q->ne[3] == 1 && q->ne[2] % k->ne[2] == 0 && q->ne[2]/k->ne[2] >= 2 && q->ne[2]/k->ne[2] <= 8) {
        auto launch_packed = [&](auto size, auto ktype) {
            constexpr int dim = decltype(size)::value;
            constexpr ggml_type kv_t = decltype(ktype)::value;
            const int64_t gqa = q->ne[2]/k->ne[2];
            const size_t smem_bytes = fattn_rel_prefill_gqa_packed_smem(dim, gqa);
            auto run_kernel = [&](auto mw_tag) {
                constexpr int mw = decltype(mw_tag)::value;
                auto kernel = fattn_rel_prefill_gqa_packed_tiles<dim, kv_t, mw>;
                if (smem_bytes >= 48*1024) {
                    CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, int(smem_bytes)));
                }
                kernel<<<dim3(tiles, unsigned(k->ne[2]), parts), dim3(WARP_SIZE, unsigned(gqa), 1), int(smem_bytes), ctx.stream()>>>(
                    (const char *) q->data, (const char *) k->data, (const char *) v->data, (const uint16_t *) mask->data,
                    (const float *) dst->src[6]->data, (const int *) dst->src[7]->data, (const int *) dst->src[8]->data,
                    (const float *) dst->src[9]->data, partial, spans, scale, q->ne[1], q->ne[2], dst->src[6]->ne[0],
                    int(gqa), mask->nb[1]/2, q->nb[1], q->nb[2], k->nb[1], k->nb[2], v->nb[1], v->nb[2],
                    hot_kv_d, hot_row, hot_head, hot_map);
            };
            if (gqa <= 6) {
                run_kernel(std::integral_constant<int, 6>{});
            } else {
                run_kernel(std::integral_constant<int, 8>{});
            }
            if (parts > 1) {
                fattn_rel_prefill_merge<dim><<<q->ne[1]*q->ne[2], dim, 0, ctx.stream()>>>(partial, (float *) dst->data, parts);
            }
        };
        if (q->ne[0] == 64) {
            launch_packed(std::integral_constant<int, 64>{}, std::integral_constant<ggml_type, GGML_TYPE_F16>{});
        } else if (k->type == GGML_TYPE_Q8_0) {
            launch_packed(std::integral_constant<int, 128>{}, std::integral_constant<ggml_type, GGML_TYPE_Q8_0>{});
        } else if (k->type == GGML_TYPE_Q4_0) {
            launch_packed(std::integral_constant<int, 128>{}, std::integral_constant<ggml_type, GGML_TYPE_Q4_0>{});
#ifdef FATTN_REL_ELEM_F8
        } else if (k->type == GGML_TYPE_F8_E4M3) {
            launch_packed(std::integral_constant<int, 128>{}, std::integral_constant<ggml_type, GGML_TYPE_F8_E4M3>{});
#endif
        } else {
            launch_packed(std::integral_constant<int, 128>{}, std::integral_constant<ggml_type, GGML_TYPE_F16>{});
        }
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    auto launch = [&](auto size) {
        constexpr int dim = decltype(size)::value;
        fattn_rel_prefill_tiles<dim><<<dim3(tiles, q->ne[2], parts), 32, 0, ctx.stream()>>>(
            (const char *) q->data, (const char *) k->data, (const char *) v->data, (const uint16_t *) mask->data,
            (const float *) dst->src[6]->data, (const int *) dst->src[7]->data, (const int *) dst->src[8]->data,
            (const float *) dst->src[9]->data, partial, spans, scale, q->ne[1], q->ne[2], dst->src[6]->ne[0],
            q->ne[2]/k->ne[2], mask->nb[1]/2, q->nb[1], q->nb[2], k->nb[1], k->nb[2], v->nb[1], v->nb[2],
            hot_kv_d, hot_row, hot_head, hot_map);
        if (parts > 1) {
            fattn_rel_prefill_merge<dim><<<q->ne[1]*q->ne[2], dim, 0, ctx.stream()>>>(partial, (float *) dst->data, parts);
        }
    };
    if (q->ne[0] == 64) {
        launch(std::integral_constant<int, 64>{});
    } else {
        launch(std::integral_constant<int, 128>{});
    }
    CUDA_CHECK(cudaGetLastError());
#else
    GGML_UNUSED_VARS(ctx, dst);
    GGML_ABORT("relative prefill requires CUDA");
#endif
}
