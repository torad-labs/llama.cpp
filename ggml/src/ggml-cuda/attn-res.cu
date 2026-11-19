#include "attn-res.cuh"

// Keep the 16- and 128-channel rows in registers until the norm is known.
template <int width>
static __global__ void norm_clip_f32(const float * x, const float * scale, float * y, const int64_t n,
        const int64_t ne1, const int64_t ne2, const int64_t nr, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t sne1, const int64_t sne2, const int64_t sne3, const int64_t ss1, const int64_t ss2, const int64_t ss3,
        const float threshold, const float eps, const bool dense) {
    const int64_t r = int64_t(blockIdx.x)*blockDim.y + threadIdx.y;
    if (r >= nr) {
        return;
    }
    if (dense) {
        x += r*n;
        if (scale) {
            scale += (r % ne1)*n;
        }
    } else {
        const int64_t i1 = r % ne1;
        const int64_t i2 = (r / ne1) % ne2;
        const int64_t i3 = r / (ne1*ne2);
        x += i1*s01 + i2*s02 + i3*s03;
        if (scale) {
            scale += (i1 % sne1)*ss1 + (i2 % sne2)*ss2 + (i3 % sne3)*ss3;
        }
    }
    y += r*n;
    constexpr int values_per_lane = (width + WARP_SIZE - 1)/WARP_SIZE;
    float values[values_per_lane > 0 ? values_per_lane : 1];
    const int64_t ncols = width ? width : n;
    float sum = 0.0f;
#pragma unroll
    for (int64_t i = threadIdx.x; i < ncols; i += WARP_SIZE) {
        const float v = scale ? x[i]*scale[i] : x[i];
        if (width) {
            values[i/WARP_SIZE] = v;
        } else {
            y[i] = v;
        }
        sum += v*v;
    }
    sum = warp_reduce_sum<width == 16 ? 16 : WARP_SIZE>(sum);
    const float factor = fminf(1.0f, threshold/(sqrtf(sum) + eps));
#pragma unroll
    for (int64_t i = threadIdx.x; i < ncols; i += WARP_SIZE) {
        y[i] = (width ? values[i/WARP_SIZE] : y[i])*factor;
    }
}

void ggml_cuda_op_norm_clip(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * a     = dst->src[0];
    const ggml_tensor * scale = dst->src[1];
    GGML_ASSERT(ggml_is_contiguous(dst));
    const int64_t nr = ggml_nrows(a);
    const bool dense = ggml_is_contiguous(a) && (!scale || (ggml_is_contiguous(scale) && scale->ne[1] == a->ne[1] && ggml_is_matrix(scale)));
    const dim3 block(a->ne[0] == 16 ? 16 : WARP_SIZE, a->ne[0] == 16 ? 16 : 8);
    const dim3 grid((nr + block.y - 1)/block.y);
    const size_t f = sizeof(float);
    const auto kernel = a->ne[0] == 16 ? norm_clip_f32<16> : a->ne[0] == 128 ? norm_clip_f32<128> : norm_clip_f32<0>;
    kernel<<<grid, block, 0, ctx.stream()>>>((const float *) a->data, scale ? (const float *) scale->data : nullptr,
        (float *) dst->data, a->ne[0], a->ne[1], a->ne[2], nr, a->nb[1]/f, a->nb[2]/f, a->nb[3]/f,
        scale ? scale->ne[1] : 1, scale ? scale->ne[2] : 1, scale ? scale->ne[3] : 1,
        scale ? scale->nb[1]/f : 0, scale ? scale->nb[2]/f : 0, scale ? scale->nb[3]/f : 0,
        ggml_get_op_params_f32(dst, 0), ggml_get_op_params_f32(dst, 1), dense);
}

#define ATTN_RES_MAX_SRC   (GGML_MAX_SRC - 1)
#define ATTN_RES_MAX_QUERY 4
#define ATTN_RES_THREADS   256

struct attn_res_sources {
    const float * data[ATTN_RES_MAX_SRC];
    int64_t       stride[ATTN_RES_MAX_SRC]; // floats between tokens
};

// one block a token: the squared sum of each source and its dot with each query, reduced over the block, then the
// softmax over the sources and the sum of the normalized sources it weights
template <int n_query>
static __global__ void attn_res_mix_f32(const attn_res_sources src, const int n_src, const float * query,
        const int64_t query_stride, float * dst, const int64_t n_embd, const float scale, const float eps) {
    constexpr int n_val = 1 + n_query; // the squared sum, then the dots
    constexpr int n_warps = ATTN_RES_THREADS/WARP_SIZE;
    __shared__ float partial[ATTN_RES_MAX_SRC*n_val][n_warps];
    __shared__ float weight[n_query][ATTN_RES_MAX_SRC];
    __shared__ float rms[ATTN_RES_MAX_SRC];

    const int64_t t = blockIdx.x;
    const int warp = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;

    for (int s = 0; s < n_src; ++s) {
        const float * x = src.data[s] + t*src.stride[s];
        float v[n_val] = {};
        for (int64_t i = threadIdx.x; i < n_embd; i += ATTN_RES_THREADS) {
            const float xi = x[i];
            v[0] += xi*xi;
#pragma unroll
            for (int q = 0; q < n_query; ++q) {
                v[1 + q] += xi*query[q*query_stride + i];
            }
        }
#pragma unroll
        for (int j = 0; j < n_val; ++j) {
            const float r = warp_reduce_sum(v[j]);
            if (lane == 0) {
                partial[s*n_val + j][warp] = r;
            }
        }
    }
    __syncthreads();

    if (threadIdx.x < n_query) {
        const int q = threadIdx.x;
        float mx = -INFINITY;
        for (int s = 0; s < n_src; ++s) {
            float sq = 0.0f, dot = 0.0f;
            for (int w = 0; w < n_warps; ++w) {
                sq  += partial[s*n_val][w];
                dot += partial[s*n_val + 1 + q][w];
            }
            const float score = scale*dot*rsqrtf(sq/n_embd + eps);
            weight[q][s] = score;
            if (q == 0) {
                rms[s] = rsqrtf(sq/n_embd + eps);
            }
            mx = fmaxf(mx, score);
        }
        for (int s = 0; s < n_src; ++s) {
            weight[q][s] = expf(weight[q][s] - mx);
        }
    }
    __syncthreads();
    if (threadIdx.x < n_query) {
        // the weight of a source carries its normalization
        const int q = threadIdx.x;
        float sum = 0.0f;
        for (int s = 0; s < n_src; ++s) {
            sum += weight[q][s];
        }
        for (int s = 0; s < n_src; ++s) {
            weight[q][s] *= rms[s]/sum;
        }
    }
    __syncthreads();

    for (int64_t i = threadIdx.x; i < n_embd; i += ATTN_RES_THREADS) {
        float y[n_query] = {};
        for (int s = 0; s < n_src; ++s) {
            const float xi = src.data[s][t*src.stride[s] + i];
#pragma unroll
            for (int q = 0; q < n_query; ++q) {
                y[q] += weight[q][s]*xi;
            }
        }
#pragma unroll
        for (int q = 0; q < n_query; ++q) {
            dst[(t*n_query + q)*n_embd + i] = y[q];
        }
    }
}

static constexpr int ATTN_RES_WIDTH = 4608;
static constexpr int ATTN_RES_CACHED_THREADS = 512;

// Keep the source rows in registers while each query computes its softmax over sources.
template <int n_query>
static __global__ __launch_bounds__(ATTN_RES_CACHED_THREADS) void attn_res_mix_4608(
        const attn_res_sources src, int n_src, const float * query, float * dst, float scale, float eps) {
    constexpr int n_warps = ATTN_RES_CACHED_THREADS/WARP_SIZE;
    constexpr int n_cols = ATTN_RES_WIDTH/ATTN_RES_CACHED_THREADS;
    static_assert(ATTN_RES_WIDTH % ATTN_RES_CACHED_THREADS == 0, "width must fill each block");
    static_assert(ATTN_RES_MAX_SRC <= WARP_SIZE, "each source needs a lane");
    __shared__ float partial[ATTN_RES_MAX_SRC][1 + n_query][n_warps];
    __shared__ float weight[n_query][ATTN_RES_MAX_SRC];
    float saved[ATTN_RES_MAX_SRC][n_cols];

    const int64_t t = blockIdx.x;
    const int warp = threadIdx.x/WARP_SIZE;
    const int lane = threadIdx.x%WARP_SIZE;
#pragma unroll
    for (int s = 0; s < ATTN_RES_MAX_SRC; ++s) {
        if (s >= n_src) {
            break;
        }
        const float * x = src.data[s] + t*src.stride[s];
        float v[1 + n_query] = {};
#pragma unroll
        for (int c = 0; c < n_cols; ++c) {
            const int i = c*ATTN_RES_CACHED_THREADS + threadIdx.x;
            const float xi = x[i];
            saved[s][c] = xi;
            v[0] += xi*xi;
#pragma unroll
            for (int q = 0; q < n_query; ++q) {
                v[1 + q] += xi*query[q*ATTN_RES_WIDTH + i];
            }
        }
#pragma unroll
        for (int j = 0; j < 1 + n_query; ++j) {
            const float sum = warp_reduce_sum(v[j]);
            if (lane == 0) {
                partial[s][j][warp] = sum;
            }
        }
    }
    __syncthreads();

    if (warp < n_query) {
        float sq = 0.0f, dot = 0.0f;
        if (lane < n_src) {
#pragma unroll
            for (int w = 0; w < n_warps; ++w) {
                sq  += partial[lane][0][w];
                dot += partial[lane][1 + warp][w];
            }
        }
        const float rms = rsqrtf(sq*(1.0f/ATTN_RES_WIDTH) + eps);
        const float score = lane < n_src ? scale*dot*rms : -INFINITY;
        const float prob = expf(score - warp_reduce_max(score));
        const float sum = warp_reduce_sum(prob);
        if (lane < n_src) {
            weight[warp][lane] = prob*(rms/sum);
        }
    }
    __syncthreads();

#pragma unroll
    for (int c = 0; c < n_cols; ++c) {
        float y[n_query] = {};
#pragma unroll
        for (int s = 0; s < ATTN_RES_MAX_SRC; ++s) {
            if (s >= n_src) {
                break;
            }
#pragma unroll
            for (int q = 0; q < n_query; ++q) {
                y[q] += weight[q][s]*saved[s][c];
            }
        }
#pragma unroll
        for (int q = 0; q < n_query; ++q) {
            dst[(t*n_query + q)*ATTN_RES_WIDTH + c*ATTN_RES_CACHED_THREADS + threadIdx.x] = y[q];
        }
    }
}

void ggml_cuda_op_attn_res_mix(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * query = dst->src[0];
    GGML_ASSERT(ggml_is_contiguous(dst));
    attn_res_sources src = {};
    int n_src = 0;
    while (1 + n_src < GGML_MAX_SRC && dst->src[1 + n_src]) {
        const ggml_tensor * x = dst->src[1 + n_src];
        src.data[n_src]   = (const float *) x->data;
        src.stride[n_src] = x->nb[1]/sizeof(float);
        ++n_src;
    }
    const int64_t n_embd  = query->ne[0];
    const int     n_query = query->ne[1];
    const float   scale   = ggml_get_op_params_f32(dst, 0);
    const float   eps     = ggml_get_op_params_f32(dst, 1);
    const int64_t qs      = query->nb[1]/sizeof(float);
    const float * q       = (const float *) query->data;
    float *       d       = (float *) dst->data;
    const dim3 grid(dst->ne[2]);
    // Shallow mixes in large batches can reread the rows with less register pressure.
    const bool cache_sources = n_src > 2 || dst->ne[2] <= 512;
    if (cache_sources && n_embd == ATTN_RES_WIDTH && ggml_cuda_info().devices[ctx.device].cc == GGML_CUDA_CC_BLACKWELL) {
        if (n_query == 1) {
            attn_res_mix_4608<1><<<grid, ATTN_RES_CACHED_THREADS, 0, ctx.stream()>>>(src, n_src, q, d, scale, eps);
            return;
        }
        if (n_query == 3) {
            attn_res_mix_4608<3><<<grid, ATTN_RES_CACHED_THREADS, 0, ctx.stream()>>>(src, n_src, q, d, scale, eps);
            return;
        }
    }
    switch (n_query) {
        case 1: attn_res_mix_f32<1><<<grid, ATTN_RES_THREADS, 0, ctx.stream()>>>(src, n_src, q, qs, d, n_embd, scale, eps); break;
        case 2: attn_res_mix_f32<2><<<grid, ATTN_RES_THREADS, 0, ctx.stream()>>>(src, n_src, q, qs, d, n_embd, scale, eps); break;
        case 3: attn_res_mix_f32<3><<<grid, ATTN_RES_THREADS, 0, ctx.stream()>>>(src, n_src, q, qs, d, n_embd, scale, eps); break;
        case 4: attn_res_mix_f32<4><<<grid, ATTN_RES_THREADS, 0, ctx.stream()>>>(src, n_src, q, qs, d, n_embd, scale, eps); break;
        default: GGML_ABORT("attn_res_mix: %d queries", n_query);
    }
}
