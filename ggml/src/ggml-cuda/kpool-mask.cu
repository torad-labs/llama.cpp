#include "kpool-mask.cuh"

// One block row of the grid a query row, its x blocks over the columns, two columns a thread. A row reads its sequence's
// positions and pools, 8 bytes a cell, which every row of the stream shares from L2; it writes 2 bytes a column, the
// same writes the copy of a host mask made, with no read of a mask
template <bool cand>
static __global__ void k_kpool_mask(
        const int32_t * __restrict__ cells,
        const int32_t * __restrict__ rows,
        half2         * __restrict__ dst,
        const int64_t n_kv,
        const int64_t n_ps,
        const int64_t n_pairs,     // columns/2
        const int64_t s_cells_ps,  // elements between two sequences' cells
        const int64_t s_cells_s,   // ... two streams'
        const int64_t s_rows_r,    // elements between two rows' facts
        const int64_t s_rows_s,
        const int64_t s_dst_r,     // pairs between two rows of the mask
        const int64_t s_dst_s) {
    const int64_t ii = blockIdx.y;
    const int64_t s  = blockIdx.z;

    const int32_t * row = rows + ii*s_rows_r + s*s_rows_s;

    const int32_t q      = row[0];
    const int32_t ps     = row[1];
    const int32_t tail   = row[2];
    const int32_t bo_vis = row[3];

    // a padded row is masked whole
    const bool live = ps >= 0 && ps < n_ps;

    const int32_t * pos  = cells + (live ? ps : 0)*s_cells_ps + s*s_cells_s;
    const int32_t * pool = pos + n_kv;

    half2 * out = dst + ii*s_dst_r + s*s_dst_s;

    const half v_keep = __float2half(0.0f);
    const half v_mask = __float2half(-INFINITY);

    for (int64_t k = (int64_t) blockIdx.x*blockDim.x + threadIdx.x; k < n_pairs; k += (int64_t) gridDim.x*blockDim.x) {
        half v[2];

#pragma unroll
        for (int t = 0; t < 2; ++t) {
            const int64_t j = 2*k + t;

            bool keep = false;

            if (live && j < n_kv) {
                const int32_t p = pos[j];

                // kpool_mask_row's tests: a cell not holding the sequence (-1) is never visible
                keep = (uint32_t) p <= (uint32_t) q && (p >= tail || (cand && (uint32_t) pool[j] < (uint32_t) bo_vis));
            }

            v[t] = keep ? v_keep : v_mask;
        }

        out[k] = __halves2half2(v[0], v[1]);
    }
}

// One bounded sort per row; physical cell order matches dense-mask compaction.
static __global__ void k_kpool_indices(
        const int32_t * cells, const int32_t * rows, const int32_t * selected, const int32_t * tail,
        int32_t * dst, int nk, int np, int nr, int ntop, int ntail, int extra, int bound, int nsort,
        int capacity, const int32_t * resident_rows, int resident_width, int64_t table_stride, int n_tables) {
    extern __shared__ int32_t ids[];
    const int row = blockIdx.x;
    const int stream = blockIdx.y;
    const int32_t * f = rows + 4*(stream*nr + row);
    const bool live = f[1] >= 0 && f[1] < np;
    const int32_t * pos = cells;
    if (capacity) {
        resident_rows += int64_t(stream)*nr*resident_width;
        const int slot = resident_width == 7 ? resident_rows[6] : 0;
        if (slot < 0 || slot >= n_tables) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
            __builtin_trap();
#else
            asm("trap;");
#endif
        }
        pos = cells + slot*table_stride;
    } else {
        pos += (int64_t(stream)*np + (live ? f[1] : 0))*2*nk;
    }
    const int32_t * pool = pos + (capacity ? capacity : nk);
    const int32_t base = capacity ? resident_rows[1] : 0;
    for (int j = threadIdx.x; j < nsort; j += blockDim.x) {
        int32_t value = INT32_MAX;
        if (live && j < ntop+ntail) {
            const bool from_top = j < ntop;
            const int32_t cell = from_top ? selected[(int64_t(stream)*nr+row)*ntop+j] : tail[(int64_t(stream)*nr+row)*ntail+j-ntop];
            if (!from_top && cell == -1) {
                // Padding in the compact tail.
            } else if (cell < 0 || cell >= nk + (from_top ? extra : 0)) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
                __builtin_trap();
#else
                asm("trap;");
#endif
            } else if (cell < nk) {
                const bool visible = uint32_t(pos[cell]) <= uint32_t(f[0]);
                const bool in_tail = pos[cell] >= f[2];
                const int32_t block = capacity && pool[cell] >= 0 ? pool[cell] - base : pool[cell];
                const bool pooled = from_top && uint32_t(block) < uint32_t(f[3]);
                if (visible && (in_tail || pooled)) { value = cell; }
            }
        }
        ids[j] = value;
    }
    __syncthreads();
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    // The launch uses 256 threads. Keep each warp's compare/exchange stages in registers;
    // only partners in different warps need shared memory and a block barrier.
    constexpr int warp_width = 32;
    constexpr int n_warps = 256/warp_width;
    for (int base = 0; base < nsort; base += blockDim.x) {
        const int j = base + threadIdx.x;
        int32_t value = j < nsort ? ids[j] : INT32_MAX;
        for (int width = 2; width <= min(nsort, warp_width); width *= 2) {
            for (int stride = width/2; stride > 0; stride /= 2) {
                const int32_t other = __shfl_xor_sync(0xffffffff, value, stride, warp_width);
                const bool take_min = ((j & stride) == 0) == ((j & width) == 0);
                value = take_min ? min(value, other) : max(value, other);
            }
        }
        if (j < nsort) { ids[j] = value; }
    }
    __syncthreads();
    for (int width = 2*warp_width; width <= nsort; width *= 2) {
        for (int stride = width/2; stride >= warp_width; stride /= 2) {
            for (int j = threadIdx.x; j < nsort; j += blockDim.x) {
                const int other = j ^ stride;
                if (other > j && ((ids[j] > ids[other]) == ((j & width) == 0))) {
                    const int32_t tmp = ids[j]; ids[j] = ids[other]; ids[other] = tmp;
                }
            }
            __syncthreads();
        }
        for (int base = 0; base < nsort; base += blockDim.x) {
            const int j = base + threadIdx.x;
            int32_t value = j < nsort ? ids[j] : INT32_MAX;
#pragma unroll
            for (int stride = warp_width/2; stride > 0; stride /= 2) {
                const int32_t other = __shfl_xor_sync(0xffffffff, value, stride, warp_width);
                const bool take_min = ((j & stride) == 0) == ((j & width) == 0);
                value = take_min ? min(value, other) : max(value, other);
            }
            if (j < nsort) { ids[j] = value; }
        }
        __syncthreads();
    }

    // Contiguous chunks preserve sorted order after the exclusive prefix sum. Compare
    // the first entry with its predecessor too, so duplicates spanning chunks vanish.
    const int chunk = (nsort + blockDim.x - 1)/blockDim.x;
    const int begin = threadIdx.x*chunk;
    const int end = min(begin + chunk, nsort);
    int count = 0;
    for (int j = begin; j < end; ++j) {
        count += ids[j] != INT32_MAX && (j == 0 || ids[j] != ids[j-1]);
    }
    const int lane = threadIdx.x % warp_width;
    const int warp = threadIdx.x / warp_width;
    int prefix = count;
#pragma unroll
    for (int offset = 1; offset < warp_width; offset *= 2) {
        const int other = __shfl_up_sync(0xffffffff, prefix, offset, warp_width);
        if (lane >= offset) { prefix += other; }
    }
    __shared__ int warp_offsets[n_warps];
    __shared__ int total;
    if (lane == warp_width-1) { warp_offsets[warp] = prefix; }
    __syncthreads();
    if (warp == 0) {
        const int warp_count = lane < n_warps ? warp_offsets[lane] : 0;
        int warp_prefix = warp_count;
#pragma unroll
        for (int offset = 1; offset < n_warps; offset *= 2) {
            const int other = __shfl_up_sync(0xffffffff, warp_prefix, offset, warp_width);
            if (lane >= offset) { warp_prefix += other; }
        }
        if (lane < n_warps) { warp_offsets[lane] = warp_prefix - warp_count; }
        if (lane == n_warps-1) { total = warp_prefix; }
    }
    __syncthreads();
    if (total > bound) { asm("trap;"); }
    int offset = warp_offsets[warp] + prefix - count;
    int32_t * out = dst + (int64_t(stream)*nr+row)*bound;
    for (int j = begin; j < end; ++j) {
        if (ids[j] != INT32_MAX && (j == 0 || ids[j] != ids[j-1])) {
            out[offset++] = ids[j];
        }
    }
    for (int64_t j = int64_t(total) + threadIdx.x; j < bound; j += blockDim.x) { out[j] = -1; }
#else
    for (int width = 2; width <= nsort; width *= 2) {
        for (int stride = width/2; stride > 0; stride /= 2) {
            for (int j = threadIdx.x; j < nsort; j += blockDim.x) {
                const int other = j ^ stride;
                if (other > j && ((ids[j] > ids[other]) == ((j & width) == 0))) {
                    const int32_t tmp = ids[j]; ids[j] = ids[other]; ids[other] = tmp;
                }
            }
            __syncthreads();
        }
    }
    if (threadIdx.x == 0) {
        int32_t * out = dst + (int64_t(stream)*nr+row)*bound;
        int count = 0;
        for (int j = 0; j < nsort && ids[j] != INT32_MAX; ++j) {
            if (j == 0 || ids[j] != ids[j-1]) {
                if (count >= bound) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
                __builtin_trap();
#else
                asm("trap;");
#endif
                }
                out[count++] = ids[j];
            }
        }
        while (count < bound) { out[count++] = -1; }
    }
#endif
}

static __global__ void k_kpool_materialize(const int32_t * table, const int32_t * rows, int32_t * dst,
        int64_t n, int mode, int64_t capacity, int64_t r, int64_t nk, int64_t np,
        int width, int nr, int64_t table_stride, int n_tables) {
    rows += int64_t(blockIdx.y)*nr*width;
    dst += int64_t(blockIdx.y)*n;
    const int slot = width == 7 ? rows[6] : 0;
    const int64_t cap = capacity/r + 2;
    const int64_t run = rows[5], off = run ? int64_t(rows[1]) - rows[4] : 0;
    if (slot < 0 || slot >= n_tables || run < 0 || run > np || off < 0 || off + run > cap) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
        __builtin_trap();
#else
        asm("trap;");
#endif
    }
    table += slot*table_stride;
    for (int64_t i = int64_t(blockIdx.x)*blockDim.x + threadIdx.x; i < n; i += int64_t(gridDim.x)*blockDim.x) {
        switch (mode) {
            case GGML_KPOOL_MATERIALIZE_FACTS:
                dst[i] = i < nk ? table[i] : table[capacity + i - nk] < 0 ? -1 : table[capacity + i - nk] - rows[1];
                break;
            case GGML_KPOOL_MATERIALIZE_CELLS:
                dst[i] = i >= np*r ? int32_t(nk + i - np*r) : i < run*r ? table[2*capacity + off*r + i] : 0;
                break;
            case GGML_KPOOL_MATERIALIZE_REPS:
                dst[i] = i < run ? table[2*capacity + cap*(r + 1) + off + i] : 0;
                break;
            case GGML_KPOOL_MATERIALIZE_BIAS: {
                const int64_t p = i%np;
                const int32_t * row = rows + width*(i/np);
                ((float *) dst)[i] = p < run && p < row[3] && table[2*capacity + cap*r + off + p] == r ? 0.0f : -INFINITY;
            } break;
        }
    }
}

// ggml_kpool_patch: a thread a patch, each to an element of its own
static __global__ void k_kpool_patch(int32_t * table, const int32_t * patches, int64_t n,
        int64_t table_size, int64_t table_stride, int n_tables) {
    const int64_t i = int64_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (i >= n) { return; }
    const int32_t slot = patches[3*i], element = patches[3*i + 1];
    if (slot < 0) { return; }
    if (slot >= n_tables || element < 0 || element >= table_size) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
        __builtin_trap();
#else
        asm("trap;");
#endif
    }
    table[slot*table_stride + element] = patches[3*i + 2];
}

void ggml_cuda_op_kpool_mask(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    if (ggml_get_op_params_i32(dst, 0) == GGML_KPOOL_MATERIALIZE_PATCH) {
        const auto * table = dst->src[0];
        const auto * patches = dst->src[1];
        GGML_ASSERT(dst->data == table->data);
        const int64_t n = patches->ne[1];
        k_kpool_patch<<<(n + 255)/256, 256, 0, ctx.stream()>>>((int32_t *) table->data, (const int32_t *) patches->data, n,
            table->ne[0], table->nb[1]/sizeof(int32_t), table->ne[1]);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    if (ggml_get_op_params_i32(dst, 0) >= GGML_KPOOL_MATERIALIZE_FACTS) {
        const auto * rows = dst->src[1];
        const int64_t n = ggml_nelements(dst)/rows->ne[2];
        const dim3 grid(std::min<int64_t>((n + 255)/256, 4096), rows->ne[2]);
        k_kpool_materialize<<<grid, 256, 0, ctx.stream()>>>(
            (const int32_t *) dst->src[0]->data, (const int32_t *) dst->src[1]->data, (int32_t *) dst->data, n,
            ggml_get_op_params_i32(dst, 0), ggml_get_op_params_i32(dst, 1), ggml_get_op_params_i32(dst, 2),
            ggml_get_op_params_i32(dst, 3), ggml_get_op_params_i32(dst, 4),
            rows->ne[0], rows->ne[1], dst->src[0]->nb[1]/sizeof(int32_t), dst->src[0]->ne[1]);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    if (ggml_get_op_params_i32(dst, 0) == GGML_KPOOL_MASK_INDICES) {
        const ggml_tensor * cells = dst->src[0];
        const ggml_tensor * rows = dst->src[1];
        const ggml_tensor * selected = dst->src[2];
        const ggml_tensor * tail = dst->src[3];
        const int n = selected->ne[0] + tail->ne[0];
        GGML_ASSERT(n > 0 && n <= GGML_KPOOL_INDICES_MAX && dst->type == GGML_TYPE_I32);
        int nsort = 1;
        while (nsort < n) { nsort *= 2; }
        const dim3 grid(rows->ne[1], rows->ne[2]);
        const int capacity = ggml_get_op_params_i32(dst, 2);
        k_kpool_indices<<<grid, 256, nsort*sizeof(int32_t), ctx.stream()>>>(
            (const int32_t *) cells->data, (const int32_t *) rows->data,
            (const int32_t *) selected->data, (const int32_t *) tail->data, (int32_t *) dst->data,
            capacity ? ggml_get_op_params_i32(dst, 3) : cells->ne[0], capacity ? 1 : cells->ne[2],
            rows->ne[1], selected->ne[0], tail->ne[0], ggml_get_op_params_i32(dst, 1), dst->ne[0], nsort,
            capacity, capacity ? (const int32_t *) dst->src[4]->data : nullptr,
            capacity ? dst->src[4]->ne[0] : 0, cells->nb[1]/sizeof(int32_t), cells->ne[1]);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    const ggml_tensor * cells = dst->src[0];
    const ggml_tensor * rows  = dst->src[1];

    GGML_ASSERT(cells->type == GGML_TYPE_I32 && rows->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_F16);
    GGML_ASSERT(ggml_is_contiguous(cells) && ggml_is_contiguous(rows) && ggml_is_contiguous(dst));
    // two columns a thread, and a row's first column 4-byte aligned
    GGML_ASSERT(dst->ne[0] % 2 == 0 && "kpool_mask on CUDA needs an even number of columns");

    const bool cand = ggml_get_op_params_i32(dst, 0) == GGML_KPOOL_MASK_CAND;

    const int64_t n_kv     = cells->ne[0];
    const int64_t n_ps     = cells->ne[2];
    const int64_t n_pairs  = dst->ne[0]/2;
    const int64_t n_rows   = dst->ne[1];
    const int64_t n_stream = dst->ne[3];

    GGML_ASSERT(n_rows <= 65535 && n_stream <= 65535);

    const int     block  = 256;
    const int64_t n_x    = std::min<int64_t>((n_pairs + block - 1)/block, 64);
    const dim3    grid((unsigned) std::max<int64_t>(n_x, 1), (unsigned) n_rows, (unsigned) n_stream);

    const int32_t * d_cells = (const int32_t *) cells->data;
    const int32_t * d_rows  = (const int32_t *) rows->data;
    half2         * d_dst   = (half2 *) dst->data;

    const int64_t s_cells_ps = cells->nb[2]/sizeof(int32_t);
    const int64_t s_cells_s  = cells->nb[3]/sizeof(int32_t);
    const int64_t s_rows_r   = rows->nb[1]/sizeof(int32_t);
    const int64_t s_rows_s   = rows->nb[2]/sizeof(int32_t);
    const int64_t s_dst_r    = dst->nb[1]/sizeof(half2);
    const int64_t s_dst_s    = dst->nb[3]/sizeof(half2);

    cudaStream_t stream = ctx.stream();

    if (cand) {
        k_kpool_mask<true><<<grid, block, 0, stream>>>(d_cells, d_rows, d_dst, n_kv, n_ps, n_pairs,
                s_cells_ps, s_cells_s, s_rows_r, s_rows_s, s_dst_r, s_dst_s);
    } else {
        k_kpool_mask<false><<<grid, block, 0, stream>>>(d_cells, d_rows, d_dst, n_kv, n_ps, n_pairs,
                s_cells_ps, s_cells_s, s_rows_r, s_rows_s, s_dst_r, s_dst_s);
    }

    CUDA_CHECK(cudaGetLastError());
}
