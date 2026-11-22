#pragma once

#include "common.cuh"
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#include "cp-async.cuh"
#endif

static int fattn_rel_prefill_parts(const ggml_tensor * dst) {
    return dst->src[0]->ne[1] <= 32 ? 128 : 8;
}

static constexpr int REL_PREFILL_RANGES = 8;
static constexpr int REL_PREFILL_MAX_HEADS = 12;

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
        if (int64_t(hi) - lo + 1 <= nk && hi <= INT_MAX - 2048) {
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
        size_t q_row, size_t q_head, size_t k_row, size_t k_head, size_t v_row, size_t v_head) {
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
        if (lane < 16) {
            cells[lane] = mapped;
        }
        __syncwarp();
        if (pad) {
            for (int i = lane; i < 16*dim; i += 32) {
                const int row = cells[i/dim];
                panel[i/dim][i%dim] = row >= 0 ? ((const half *) (k + row*k_row))[i%dim] : __float2half(0.0f);
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
                panel[i/dim][i%dim] = row >= 0 ? ((const half *) (v + row*v_row))[i%dim] : __float2half(0.0f);
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

// gqa_packed: the packed successor of fattn_rel_prefill_tiles. One block runs a KV head's whole group of query heads
// as warps, so every K and V tile is staged once into shared memory by cp.async for all of them instead of being
// re-read by each head's block. A warp keeps the unpacked kernel's rows and arithmetic exactly; only where its tiles
// come from changes, so a head's sums stay bit-identical when the KV order is kept.
//
// Shared layout per block: two slots of one tile's K panel then V panel (each 16 x dim halves), then one f16 panel of
// query rows per warp, whose bytes double as the warp's score and weight slices once its fragments are in registers.
template<int dim>
static __global__ void fattn_rel_prefill_gqa_packed_tiles(
        const char * q, const char * k, const char * v, const uint16_t * mask, const float * table,
        const int * q_pos, const int * k_pos, const float * sinks, float * partial, const fattn_rel_prefill_ranges * spans,
        float scale, int nq, int nh, int n_dist, int gqa, size_t mask_stride,
        size_t q_row, size_t q_head, size_t k_row, size_t k_head, size_t v_row, size_t v_head) {
#if defined(CP_ASYNC_AVAILABLE) && __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
    namespace wmma = nvcuda::wmma;
    using A = wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major>;
    using B_key = wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major>;
    using B_value = wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major>;
    using C = wmma::fragment<wmma::accumulator, 16, 16, 16, float>;

    const int parts = gridDim.z;
    const int nth = blockDim.x, lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const int query = 16*blockIdx.x, kvh = blockIdx.y;
    const int rows = min(16, nq - query);
    const int head = kvh*gqa + warp;
    k += size_t(kvh)*k_head;
    v += size_t(kvh)*v_head;

    const fattn_rel_prefill_ranges & span = spans[blockIdx.x];
    const int first = span.first, last = span.last, count = span.count;

    // The physical cell a tile's row r reads (the unpacked kernel's mapping), -1 when it holds nothing.
    auto mapped_cell = [&](int logical, int r) {
        if (!count) {
            return logical + r <= last ? logical + r : -1;
        }
        int cell = -1;
        for (int i = 0; i < count; ++i) {
            if (logical + r >= span.runs[i].x && logical + r <= span.runs[i].y) {
                cell = span.runs[i].z + (logical + r - span.runs[i].x);
            }
        }
        return cell;
    };

    __shared__ __align__(16) int cells_sh[16];
    extern __shared__ char smem_dyn[];
    half * kv_buf = (half *) smem_dyn;
    half * q_base = kv_buf + 64*dim;

    // One warp's own panel of query rows, converted and scaled once as in the unpacked kernel.
    half * qpw = q_base + size_t(warp)*16*dim;
    for (int i = lane; i < 16*dim; i += 32) {
        const int row = i/dim, col = i%dim;
        const float value = row < rows ? ((const float *) (q + (query + row)*q_row + head*q_head))[col] : 0.0f;
        qpw[row*dim + col] = __float2half(value*scale);
    }
    A queries[dim/16];
    C values[dim/16];
#pragma unroll
    for (int col = 0; col < dim; col += 16) {
        wmma::load_matrix_sync(queries[col/16], qpw + col, dim);
        wmma::fill_fragment(values[col/16], 0.0f);
    }

    float maximum = lane < rows ? sinks[int64_t(query + lane)*nh + head] : 0.0f;
    float denominator = blockIdx.z == 0 ? 1.0f : 0.0f;

    // Stage one tile's K and V panels into a slot: mapped rows are copied by cp.async, unmapped ones zeroed,
    // as the unpacked kernel's padded path fills them.
    auto stage_tile = [&](int slot, int logical) {
        half * dst_base = kv_buf + size_t(slot)*32*dim;
        constexpr int chunks_row = dim/8;
        for (int c = threadIdx.x; c < 2*16*chunks_row; c += nth) {
            const int panelrow = c/chunks_row, chunk = (c%chunks_row)*8;
            const int r = panelrow & 15;
            const int cell = mapped_cell(logical, r);
            char * dstp = (char *) (dst_base + panelrow*dim + chunk);
            if (cell >= 0) {
                const bool is_v = panelrow >= 16;
                const char * src = (is_v ? v : k) + size_t(cell)*(is_v ? v_row : k_row) + chunk*sizeof(half);
                cp_async_cg_16<0>(ggml_cuda_cvta_generic_to_shared(dstp), src);
            } else {
                *(int4 *) dstp = make_int4(0, 0, 0, 0);
            }
        }
    };

    if (first + 16*blockIdx.z <= last) {
        stage_tile(0, first + 16*blockIdx.z);
        cp_async_commit_group();
    }

    int it = 0;
    for (int logical = first + 16*blockIdx.z; logical <= last; logical += 16*parts, ++it) {
        const bool has_next = logical + 16*parts <= last;
        if (threadIdx.x < 16) {
            cells_sh[threadIdx.x] = mapped_cell(logical, threadIdx.x);
        }
        if (has_next) {
            stage_tile((it + 1) & 1, logical + 16*parts);
            cp_async_commit_group();
            cp_async_wait_group<1>();
        } else {
            cp_async_wait_group<0>();
        }
        __syncthreads();

        const half * kt = kv_buf + size_t(it & 1)*32*dim;
        const half * vt = kt + 16*dim;
        float * sc = (float *) qpw;
        half * wt = (half *) (sc + 256);

        C dots;
        wmma::fill_fragment(dots, 0.0f);
#pragma unroll
        for (int col = 0; col < dim; col += 16) {
            B_key key;
            wmma::load_matrix_sync(key, kt + col, dim);
            wmma::mma_sync(dots, queries[col/16], key, dots);
        }
        wmma::store_matrix_sync(sc, dots, 16, wmma::mem_row_major);
        __syncwarp();
        if (lane < 16) {
            const int row = query + lane;
            float next = maximum;
#pragma unroll
            for (int col = 0; col < 16; ++col) {
                const int key = cells_sh[col];
                float score = -INFINITY;
                if (lane < rows && key >= 0 && (mask[row*mask_stride + key/16] & (1u << (key%16)))) {
                    const int64_t dist = int64_t(q_pos[row]) - k_pos[key];
                    score = sc[lane*16 + col] + (dist >= 0 && dist < n_dist ? table[(int64_t(row)*nh + head)*n_dist + dist] : 0.0f);
                }
                sc[lane*16 + col] = score;
                next = fmaxf(next, score);
            }
            const float rescale = maximum == next ? 1.0f : expf(maximum - next);
            denominator *= rescale;
#pragma unroll
            for (int col = 0; col < 16; ++col) {
                const float weight = sc[lane*16 + col] == -INFINITY ? 0.0f : expf(sc[lane*16 + col] - next);
                denominator += weight;
                wt[lane*16 + col] = __float2half(weight);
                sc[lane*16 + col] = rescale;
            }
            maximum = next;
        }
        __syncwarp();
        A weight;
        C rescale;
        wmma::load_matrix_sync(weight, wt, 16);
        wmma::load_matrix_sync(rescale, sc, 16, wmma::mem_row_major);
#pragma unroll
        for (int col = 0; col < dim; col += 16) {
            B_value value;
            wmma::load_matrix_sync(value, vt + col, dim);
            C & accum = values[col/16];
#pragma unroll
            for (int i = 0; i < C::num_elements; ++i) {
                accum.x[i] *= rescale.x[i];
            }
            wmma::mma_sync(accum, weight, value, accum);
        }

        __syncthreads();
    }

    partial += ((int64_t(query)*nh + head)*parts + blockIdx.z)*(dim + 2);
    const int stride = nh*parts*(dim + 2);
    if (lane < rows) {
        partial[lane*stride] = maximum;
        partial[lane*stride + 1] = denominator;
    }
#pragma unroll
    for (int col = 0; col < dim; col += 16) {
        float * sc = (float *) qpw;
        wmma::store_matrix_sync(sc, values[col/16], 16, wmma::mem_row_major);
        __syncwarp();
#pragma unroll
        for (int i = lane; i < 256; i += 32) {
            if (i/16 < rows) {
                partial[(i/16)*stride + 2 + col + i%16] = sc[i/16*16 + i%16];
            }
        }
        __syncwarp();
    }
#else
    NO_DEVICE_CODE;
#endif
}

static size_t fattn_rel_prefill_partial_offset(const ggml_tensor * dst, size_t output_bytes) {
    const size_t tiles = (dst->src[0]->ne[1] + 15)/16;
    return GGML_PAD(output_bytes, 128) + GGML_PAD(tiles*(3 + REL_PREFILL_RANGES*3)*sizeof(int32_t), 128);
}

static size_t fattn_rel_prefill_alloc_size(const ggml_tensor * dst) {
    const size_t rows = std::min<int64_t>(dst->ne[1], REL_PREFILL_MAX_HEADS)*dst->ne[2];
    const size_t output_bytes = rows*ggml_row_size(dst->type, dst->ne[0]);
    return fattn_rel_prefill_partial_offset(dst, output_bytes) + rows*fattn_rel_prefill_parts(dst)*(2 + dst->ne[0])*sizeof(float);
}

// The dynamic shared memory the gqa_packed kernel runs on: two slots of a tile's K and V panels, then one f16 panel
// of query rows per warp whose bytes double as its score and weight slices.
static size_t fattn_rel_prefill_gqa_packed_smem(int64_t dim, int64_t gqa) {
    return GGML_PAD((2*32 + 16*gqa)*dim*sizeof(half), 16);
}

static void ggml_cuda_flash_attn_ext_rel_prefill(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    const auto * q = dst->src[0], * k = dst->src[1], * v = dst->src[2], * mask = dst->src[3];
    const int tiles = (q->ne[1] + 15)/16, parts = fattn_rel_prefill_parts(dst);
    char * data = (char *) dst->data;
    fattn_rel_prefill_ranges * spans = (fattn_rel_prefill_ranges *) (data + GGML_PAD(ggml_nbytes(dst), 128));
    float * partial = (float *) (data + fattn_rel_prefill_partial_offset(dst, ggml_nbytes(dst)));
    float scale;
    memcpy(&scale, dst->op_params, sizeof(scale));
    fattn_rel_prefill_spans<<<tiles, 128, 0, ctx.stream()>>>((const uint16_t *) mask->data, mask->nb[1]/2, (const int *) dst->src[8]->data, q->ne[1], k->ne[1], spans);

    static const bool gqa_packed = ggml_env_switch("GGML_CUDA_FATTN_REL_PACKED_PREFILL");
    if (gqa_packed && q->ne[3] == 1 && q->ne[2] % k->ne[2] == 0 && q->ne[2]/k->ne[2] >= 2) {
        auto launch_packed = [&](auto size) {
            constexpr int dim = decltype(size)::value;
            const int64_t gqa = q->ne[2]/k->ne[2];
            const size_t smem_bytes = fattn_rel_prefill_gqa_packed_smem(dim, gqa);
            if (smem_bytes > 49152) {
                CUDA_CHECK(cudaFuncSetAttribute(fattn_rel_prefill_gqa_packed_tiles<dim>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, int(smem_bytes)));
            }
            fattn_rel_prefill_gqa_packed_tiles<dim><<<dim3(tiles, unsigned(k->ne[2]), parts), 32*int(gqa), int(smem_bytes), ctx.stream()>>>(
                (const char *) q->data, (const char *) k->data, (const char *) v->data, (const uint16_t *) mask->data,
                (const float *) dst->src[6]->data, (const int *) dst->src[7]->data, (const int *) dst->src[8]->data,
                (const float *) dst->src[9]->data, partial, spans, scale, q->ne[1], q->ne[2], dst->src[6]->ne[0],
                int(gqa), mask->nb[1]/2, q->nb[1], q->nb[2], k->nb[1], k->nb[2], v->nb[1], v->nb[2]);
            fattn_rel_prefill_merge<dim><<<q->ne[1]*q->ne[2], dim, 0, ctx.stream()>>>(partial, (float *) dst->data, parts);
        };
        if (q->ne[0] == 64) {
            launch_packed(std::integral_constant<int, 64>{});
        } else {
            launch_packed(std::integral_constant<int, 128>{});
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
            q->ne[2]/k->ne[2], mask->nb[1]/2, q->nb[1], q->nb[2], k->nb[1], k->nb[2], v->nb[1], v->nb[2]);
        fattn_rel_prefill_merge<dim><<<q->ne[1]*q->ne[2], dim, 0, ctx.stream()>>>(partial, (float *) dst->data, parts);
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
