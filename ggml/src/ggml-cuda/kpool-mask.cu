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

void ggml_cuda_op_kpool_mask(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
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
