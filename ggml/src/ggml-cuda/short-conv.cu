#include "short-conv.cuh"

static __global__ void short_conv_f32(const float * x, const float * w, const float * d, float * y, const int64_t k,
        const int64_t channels, const int64_t n_t, const int64_t n_s, const int64_t sx1, const int64_t sx2,
        const int64_t sd1, const int64_t group, const float scale) {
    const int64_t c = int64_t(blockIdx.x)*blockDim.x + threadIdx.x;
    const int64_t s = blockIdx.y;
    if (c >= channels || s >= n_s) {
        return;
    }
    const float * xc = x + c*sx1 + s*sx2;
    const float * wc = w + c*k;
    for (int64_t t = int64_t(blockIdx.z)*blockDim.y + threadIdx.y; t < n_t; t += gridDim.z*blockDim.y) {
        float acc = xc[k - 1 + t];
        for (int64_t j = 0; j < k; ++j) {
            acc += xc[t + j]*wc[j];
        }
        if (d) {
            const float * dt = d + (t + s*n_t)*sd1 + (c/group)*k;
            for (int64_t lag = 0; lag < k; ++lag) {
                acc += xc[k - 1 - lag + t]*dt[lag];
            }
        }
        y[(s*n_t + t)*channels + c] = scale*acc;
    }
}

template<bool dynamic>
static __global__ void short_conv_4_tiled(const float * x, const float * w, const float * d, float * y,
        const int64_t channels, const int64_t n_t, const int64_t sx1, const int64_t sx2, const int64_t sd1,
        const int64_t group, const float scale) {
    constexpr int tile_size = 32;
    // Read consecutive tokens and transpose the tile to write consecutive channels.
    __shared__ float tile[tile_size][tile_size + 1];
    const int64_t c0 = int64_t(blockIdx.x)*tile_size;
    const int64_t s = blockIdx.y;
    for (int64_t t0 = int64_t(blockIdx.z)*tile_size; t0 < n_t; t0 += int64_t(gridDim.z)*tile_size) {
        const int64_t t = t0 + threadIdx.x;
        for (int i = threadIdx.y; i < tile_size; i += blockDim.y) {
            const int64_t c = c0 + i;
            if (c < channels && t < n_t) {
                const float * xc = x + s*sx2 + c*sx1 + t;
                const float * wc = w + 4*c;
                const float x0 = xc[0], x1 = xc[1], x2 = xc[2], x3 = xc[3];
                float acc = x3;
                acc += x0*wc[0];
                acc += x1*wc[1];
                acc += x2*wc[2];
                acc += x3*wc[3];
                if (dynamic) {
                    const float * dt = d + (s*n_t + t)*sd1 + 4*(c/group);
                    acc += x3*dt[0];
                    acc += x2*dt[1];
                    acc += x1*dt[2];
                    acc += x0*dt[3];
                }
                tile[i][threadIdx.x] = scale*acc;
            }
        }
        __syncthreads();
        const int64_t c = c0 + threadIdx.x;
        for (int i = threadIdx.y; i < tile_size; i += blockDim.y) {
            if (c < channels && t0 + i < n_t) {
                y[(s*n_t + t0 + i)*channels + c] = tile[threadIdx.x][i];
            }
        }
        if (t0 + int64_t(gridDim.z)*tile_size < n_t) {
            __syncthreads();
        }
    }
}

// The SHORT_CONV behind a history CONCAT the evaluator skipped (ggml_cuda_short_conv_window): a thread per
// channel and sequence streams the window's columns from the CONCAT's two sources, writes each into the joined
// window the CONCAT would have made, and from the fourth column on writes the convolution of the last four, in
// short_conv_f32's sum order. Strides are in floats.
template<bool dynamic>
static __global__ void short_conv_window_4_f32(const float * past, const int64_t sp0, const int64_t sp1,
        const int64_t sp2, const float * rows, const int64_t sr0, const int64_t sr1, const int64_t sr2,
        const float * w, const float * d, float * y, float * window, const int64_t channels, const int64_t n_t,
        const int64_t sd1, const int64_t group, const float scale) {
    const int64_t c = int64_t(blockIdx.x)*blockDim.x + threadIdx.x;
    const int64_t s = blockIdx.y;
    if (c >= channels) {
        return;
    }
    const float * pc  = past + c*sp1 + s*sp2;
    const float * rc  = rows + c*sr1 + s*sr2;
    const float * wc  = w + 4*c;
    float *       win = window + (c + s*channels)*(n_t + 3);
    float v0 = 0.0f, v1 = 0.0f, v2 = 0.0f, v3 = 0.0f;
    for (int64_t col = 0; col < n_t + 3; ++col) {
        const float val = col < 3 ? pc[col*sp0] : rc[(col - 3)*sr0];
        win[col] = val;
        v0 = v1; v1 = v2; v2 = v3; v3 = val;
        if (col < 3) {
            continue;
        }
        const int64_t t = col - 3;
        float acc = v3;
        acc += v0*wc[0];
        acc += v1*wc[1];
        acc += v2*wc[2];
        acc += v3*wc[3];
        if (dynamic) {
            const float * dt = d + (t + s*n_t)*sd1 + 4*(c/group);
            acc += v3*dt[0];
            acc += v2*dt[1];
            acc += v1*dt[2];
            acc += v0*dt[3];
        }
        y[(s*n_t + t)*channels + c] = scale*acc;
    }
}

void ggml_cuda_op_short_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * x = dst->src[0];
    const ggml_tensor * w = dst->src[1];
    const ggml_tensor * d = dst->src[2];
    GGML_ASSERT(ggml_is_contiguous(dst));
    const int64_t k = w->ne[0];
    const int64_t channels = dst->ne[0];
    const int64_t n_t = dst->ne[1];
    const int64_t n_s = dst->ne[2];
    const size_t f = sizeof(float);
    // the history CONCAT ggml_cuda_try_short_conv_window matched: src0 (the CONCAT) was never written, the kernel
    // reads its two sources and writes it beside the convolution
    if (const ggml_cuda_short_conv_window * win = ctx.short_conv_windows().find(dst)) {
        const ggml_tensor * past = win->past;
        const ggml_tensor * rows = win->rows;
        const dim3 block(128);
        const dim3 grid((channels + block.x - 1)/block.x, n_s);
        auto kernel = d ? short_conv_window_4_f32<true> : short_conv_window_4_f32<false>;
        kernel<<<grid, block, 0, ctx.stream()>>>((const float *) past->data, past->nb[0]/f, past->nb[1]/f,
            past->nb[2]/f, (const float *) rows->data, rows->nb[0]/f, rows->nb[1]/f, rows->nb[2]/f,
            (const float *) w->data, d ? (const float *) d->data : nullptr, (float *) dst->data,
            (float *) win->window->data, channels, n_t, d ? d->nb[1]/f : 0, d ? channels/(d->ne[0]/4) : 1,
            ggml_get_op_params_f32(dst, 0));
        return;
    }
    const int min_tiled_tokens = !d && channels >= 1024 ? 128 : 512;
    if (k == 4 && n_t >= min_tiled_tokens && ggml_cuda_info().devices[ctx.device].cc == GGML_CUDA_CC_BLACKWELL) {
        const dim3 block(32, 8);
        const dim3 grid((channels + 31)/32, n_s, std::min<int64_t>((n_t + 31)/32, 65535));
        auto kernel = d ? short_conv_4_tiled<true> : short_conv_4_tiled<false>;
        kernel<<<grid, block, 0, ctx.stream()>>>((const float *) x->data, (const float *) w->data,
            d ? (const float *) d->data : nullptr, (float *) dst->data, channels, n_t, x->nb[1]/f, x->nb[2]/f,
            d ? d->nb[1]/f : 0, d ? channels/(d->ne[0]/4) : 1, ggml_get_op_params_f32(dst, 0));
        return;
    }
    const dim3 block = n_t < 8 ? dim3(128) : dim3(32, 4);
    const int64_t token_tiles = n_t < 8 ? 1 : (n_t + block.y - 1)/block.y;
    const dim3 grid((channels + block.x - 1)/block.x, n_s, std::min<int64_t>(token_tiles, 65535));
    short_conv_f32<<<grid, block, 0, ctx.stream()>>>((const float *) x->data, (const float *) w->data,
        d ? (const float *) d->data : nullptr, (float *) dst->data, k, channels, n_t, n_s, x->nb[1]/f, x->nb[2]/f,
        d ? d->nb[1]/f : 0, d ? channels/(d->ne[0]/k) : 1, ggml_get_op_params_f32(dst, 0));
}
