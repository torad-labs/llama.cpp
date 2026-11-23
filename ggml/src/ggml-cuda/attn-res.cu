#include "attn-res.cuh"

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#include <cooperative_groups.h>
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

// Keep the 16- and 128-channel rows in registers until the norm is known.
template <int width>
static __global__ void norm_clip_f32(const float * x, const float * scale, float * y, const int64_t n,
        const int64_t ne1, const int64_t ne2, const int64_t nr, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t sne1, const int64_t sne2, const int64_t sne3, const int64_t ss1, const int64_t ss2, const int64_t ss3,
        const float threshold, const float eps, const bool dense) {
    ggml_cuda_pdl_lc();
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
    ggml_cuda_pdl_sync();
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
    const ggml_cuda_kernel_launch_params params = {grid, block, 0, ctx.stream()};
    ggml_cuda_kernel_launch(kernel, params, (const float *) a->data, scale ? (const float *) scale->data : nullptr,
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
static __global__ void attn_res_mix_f32(const attn_res_sources src, const int n_src, const int n_normed, const float * query,
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
        const bool need_sq = s >= n_normed;
        float v[n_val] = {};
        for (int64_t i = threadIdx.x; i < n_embd; i += ATTN_RES_THREADS) {
            const float xi = x[i];
            if (need_sq) {
                v[0] += xi*xi;
            }
#pragma unroll
            for (int q = 0; q < n_query; ++q) {
                v[1 + q] += xi*query[q*query_stride + i];
            }
        }
#pragma unroll
        for (int j = 0; j < n_val; ++j) {
            if (j == 0 && !need_sq) {
                continue;
            }
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
                if (s >= n_normed) {
                    sq += partial[s*n_val][w];
                }
                dot += partial[s*n_val + 1 + q][w];
            }
            const float r = s < n_normed ? 1.0f : rsqrtf(sq/n_embd + eps);
            const float score = scale*dot*r;
            weight[q][s] = score;
            if (q == 0) {
                rms[s] = r;
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
        const attn_res_sources src, int n_src, int n_normed, const float * query, float * dst, float scale, float eps) {
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
        const bool need_sq = s >= n_normed;
        float v[1 + n_query] = {};
#pragma unroll
        for (int c = 0; c < n_cols; ++c) {
            const int i = c*ATTN_RES_CACHED_THREADS + threadIdx.x;
            const float xi = x[i];
            saved[s][c] = xi;
            if (need_sq) {
                v[0] += xi*xi;
            }
#pragma unroll
            for (int q = 0; q < n_query; ++q) {
                v[1 + q] += xi*query[q*ATTN_RES_WIDTH + i];
            }
        }
#pragma unroll
        for (int j = 0; j < 1 + n_query; ++j) {
            if (j == 0 && !need_sq) {
                continue;
            }
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
                if (lane >= n_normed) {
                    sq += partial[lane][0][w];
                }
                dot += partial[lane][1 + warp][w];
            }
        }
        const float rms = lane < n_normed ? 1.0f : rsqrtf(sq*(1.0f/ATTN_RES_WIDTH) + eps);
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

// cudaLaunchKernelEx, which a cluster launch needs, exists where PDL does
#if defined(GGML_CUDA_USE_PDL)
#define ATTN_RES_SPLIT
#endif // defined(GGML_CUDA_USE_PDL)

#define ATTN_RES_SPLIT_BLOCKS     8   // a token's, one cluster: the largest portable one
#define ATTN_RES_SPLIT_THREADS    256 // a block's, at most
#define ATTN_RES_SPLIT_MAX_COLS   4   // a thread's, at most: rows up to 8192
#define ATTN_RES_SPLIT_MAX_TOKENS 32

#ifdef ATTN_RES_SPLIT
// A few tokens (decode, a drafted step) on Hopper and newer. The kernels above give a token one block, which at decode
// is one SM waiting on its loads (RTX 5090, 2304 columns: 3.5 us at 2 sources and 1 query to 15 us at 9 sources and 4
// queries). Here the 8 blocks of a cluster split a token's columns: a thread loads its columns of every source once
// and keeps them in registers, the block sums its squares and dots, every block adds the cluster's 8 sums from their
// shared memory in rank order (so all 8 get the same weights), and mixes its own columns. The query, a weight, is
// requested before the PDL wait. The same in a single pass: 1.7 to 3.7 us.
template <int n_query, int cols>
static __global__ void __launch_bounds__(ATTN_RES_SPLIT_THREADS) attn_res_mix_split(const attn_res_sources src,
        const int n_src, const int n_normed, const float * query, const int64_t query_stride, float * dst, const int n_embd,
        const float scale, const float eps, const bool query_first) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_HOPPER
    constexpr int n_val = 1 + n_query; // the squared sum, then the dots
    __shared__ float partial[ATTN_RES_MAX_SRC*n_val][ATTN_RES_SPLIT_THREADS/WARP_SIZE];
    __shared__ float block_sums[ATTN_RES_MAX_SRC*n_val]; // this block's, which the cluster reads
    __shared__ float sums[ATTN_RES_MAX_SRC*n_val];       // the token's
    __shared__ float weight[n_query][ATTN_RES_MAX_SRC];

    const int64_t t       = blockIdx.y;
    const int     warp    = threadIdx.x/WARP_SIZE;
    const int     lane    = threadIdx.x%WARP_SIZE;
    const int     n_warps = blockDim.x/WARP_SIZE;

    // column c of the thread; a block past the row's end has none, and adds zeros
    int  col[cols];
    bool ok[cols];
#pragma unroll
    for (int c = 0; c < cols; ++c) {
        col[c] = (blockIdx.x*cols + c)*blockDim.x + threadIdx.x;
        ok[c]  = col[c] < n_embd;
    }
    float qv[n_query][cols];
    if (query_first) {
#pragma unroll
        for (int q = 0; q < n_query; ++q) {
#pragma unroll
            for (int c = 0; c < cols; ++c) {
                qv[q][c] = ok[c] ? query[q*query_stride + col[c]] : 0.0f;
            }
        }
    }
    ggml_cuda_pdl_sync();
    if (!query_first) {
#pragma unroll
        for (int q = 0; q < n_query; ++q) {
#pragma unroll
            for (int c = 0; c < cols; ++c) {
                qv[q][c] = ok[c] ? query[q*query_stride + col[c]] : 0.0f;
            }
        }
    }
    // predicated, with no branch between them, so that every load is in flight at once
    float x[ATTN_RES_MAX_SRC][cols];
#pragma unroll
    for (int s = 0; s < ATTN_RES_MAX_SRC; ++s) {
        const float * xs = src.data[s] + t*src.stride[s];
#pragma unroll
        for (int c = 0; c < cols; ++c) {
            x[s][c] = s < n_src && ok[c] ? xs[col[c]] : 0.0f;
        }
    }
#pragma unroll
    for (int s = 0; s < ATTN_RES_MAX_SRC; ++s) {
        if (s >= n_src) {
            break;
        }
        const bool need_sq = s >= n_normed;
        float v[n_val] = {};
#pragma unroll
        for (int c = 0; c < cols; ++c) {
            if (need_sq) {
                v[0] += x[s][c]*x[s][c];
            }
#pragma unroll
            for (int q = 0; q < n_query; ++q) {
                v[1 + q] += x[s][c]*qv[q][c];
            }
        }
#pragma unroll
        for (int j = 0; j < n_val; ++j) {
            if (j == 0 && !need_sq) {
                continue;
            }
            const float r = warp_reduce_sum(v[j]);
            if (lane == 0) {
                partial[s*n_val + j][warp] = r;
            }
        }
    }
    __syncthreads();
    const int n_sums = n_src*n_val;
    if (threadIdx.x < n_sums) {
        const int s_idx = threadIdx.x / n_val;
        const int j_idx = threadIdx.x % n_val;
        float sum = 0.0f;
        if (j_idx != 0 || s_idx >= n_normed) {
            for (int w = 0; w < n_warps; ++w) {
                sum += partial[threadIdx.x][w];
            }
        }
        block_sums[threadIdx.x] = sum;
    }
    cooperative_groups::cluster_group cluster = cooperative_groups::this_cluster();
    cluster.sync();
    if (threadIdx.x < n_sums) {
        const int s_idx = threadIdx.x / n_val;
        const int j_idx = threadIdx.x % n_val;
        float sum = 0.0f;
        if (j_idx != 0 || s_idx >= n_normed) {
#pragma unroll
            for (int r = 0; r < ATTN_RES_SPLIT_BLOCKS; ++r) {
                sum += cluster.map_shared_rank(block_sums, r)[threadIdx.x];
            }
        }
        sums[threadIdx.x] = sum;
    }
    // this block has read the others' sums: they may leave once all 8 arrived (the wait is at the end)
    asm volatile("barrier.cluster.arrive.aligned;" ::: "memory");
    __syncthreads();

    // a warp a query, a lane a source; the weight of a source carries its normalization
    for (int q = warp; q < n_query; q += n_warps) {
        const float sq    = lane < n_src ? sums[lane*n_val]         : 0.0f;
        const float dot   = lane < n_src ? sums[lane*n_val + 1 + q] : 0.0f;
        const float rms   = lane < n_normed ? 1.0f : rsqrtf(sq/n_embd + eps);
        const float score = lane < n_src ? scale*dot*rms : -INFINITY;
        const float prob  = expf(score - warp_reduce_max(score));
        const float sum   = warp_reduce_sum(prob);
        if (lane < n_src) {
            weight[q][lane] = prob*(rms/sum);
        }
    }
    __syncthreads();
    ggml_cuda_pdl_lc();

#pragma unroll
    for (int c = 0; c < cols; ++c) {
        float y[n_query] = {};
#pragma unroll
        for (int s = 0; s < ATTN_RES_MAX_SRC; ++s) {
            if (s >= n_src) {
                break;
            }
#pragma unroll
            for (int q = 0; q < n_query; ++q) {
                y[q] += weight[q][s]*x[s][c];
            }
        }
        if (ok[c]) {
#pragma unroll
            for (int q = 0; q < n_query; ++q) {
                dst[(t*n_query + q)*n_embd + col[c]] = y[q];
            }
        }
    }
    asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
#else
    GGML_UNUSED_VARS(src, n_src, n_normed, query, query_stride, dst, n_embd, scale, eps, query_first);
    NO_DEVICE_CODE;
#endif // defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_HOPPER
}

static bool attn_res_mix_legacy() {
    static const bool legacy = ggml_env_switch("GGML_CUDA_ATTN_RES_MIX_LEGACY");
    return legacy;
}

// a thread's columns: the fewest that keep a block at 256 threads; 0 past 8192 columns
static int attn_res_split_cols(const int64_t n_embd) {
    const int64_t per_cols = ATTN_RES_SPLIT_BLOCKS*ATTN_RES_SPLIT_THREADS;
    const int64_t cols = std::max<int64_t>(1, (n_embd + per_cols - 1)/per_cols);
    return cols <= ATTN_RES_SPLIT_MAX_COLS ? (int) cols : 0;
}

static bool attn_res_split_usable(const int device, const int64_t n_embd, const int n_query, const int64_t n_tokens) {
    const auto & dev = ggml_cuda_info().devices[device];
    const int cc = dev.cc;
    // GB203 (<= 96 SMs) faults on ld.shared::cluster; keep cluster split on GB202 (5090/PRO 6000) and Hopper.
    const bool dsmem_ok = !(cc == GGML_CUDA_CC_BLACKWELL && dev.nsm <= 96);
    return !attn_res_mix_legacy() && dsmem_ok && GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_HOPPER &&
        ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_HOPPER && n_tokens <= ATTN_RES_SPLIT_MAX_TOKENS &&
        n_query >= 1 && n_query <= ATTN_RES_MAX_QUERY && attn_res_split_cols(n_embd) > 0;
}

// the cluster is 8 blocks for every shape, so a CUDA graph updated from another shape keeps the right cluster dims
template <int n_query, int cols>
static void attn_res_mix_split_launch(const attn_res_sources & src, const int n_src, const int n_normed, const float * query,
        const int64_t query_stride, float * dst, const int n_embd, const int n_tokens, const float scale, const float eps,
        const bool query_first, cudaStream_t stream) {
    const auto kernel  = attn_res_mix_split<n_query, cols>;
    const int  threads = GGML_PAD((n_embd + ATTN_RES_SPLIT_BLOCKS*cols - 1)/(ATTN_RES_SPLIT_BLOCKS*cols), WARP_SIZE);
    cudaLaunchAttribute attr[2];
    cudaLaunchConfig_t  cfg = {};
    cfg.gridDim  = dim3(ATTN_RES_SPLIT_BLOCKS, n_tokens);
    cfg.blockDim = dim3(threads);
    cfg.stream   = stream;
    cfg.attrs    = attr;
    attr[cfg.numAttrs].id               = cudaLaunchAttributeClusterDimension;
    attr[cfg.numAttrs].val.clusterDim.x = ATTN_RES_SPLIT_BLOCKS;
    attr[cfg.numAttrs].val.clusterDim.y = 1;
    attr[cfg.numAttrs].val.clusterDim.z = 1;
    ++cfg.numAttrs;
    if (ggml_cuda_kernel_can_use_pdl(reinterpret_cast<const void *>(kernel))) {
        attr[cfg.numAttrs].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attr[cfg.numAttrs].val.programmaticStreamSerializationAllowed = 1;
        ++cfg.numAttrs;
    }
    CUDA_CHECK(cudaLaunchKernelEx(&cfg, kernel, src, n_src, n_normed, query, query_stride, dst, n_embd, scale, eps, query_first));
}

template <int n_query>
static void attn_res_mix_split_cuda(const int cols, const attn_res_sources & src, const int n_src, const int n_normed, const float * query,
        const int64_t query_stride, float * dst, const int n_embd, const int n_tokens, const float scale, const float eps,
        const bool query_first, cudaStream_t stream) {
    switch (cols) {
        case 1: attn_res_mix_split_launch<n_query, 1>(src, n_src, n_normed, query, query_stride, dst, n_embd, n_tokens, scale, eps, query_first, stream); break;
        case 2: attn_res_mix_split_launch<n_query, 2>(src, n_src, n_normed, query, query_stride, dst, n_embd, n_tokens, scale, eps, query_first, stream); break;
        case 3: attn_res_mix_split_launch<n_query, 3>(src, n_src, n_normed, query, query_stride, dst, n_embd, n_tokens, scale, eps, query_first, stream); break;
        case 4: attn_res_mix_split_launch<n_query, 4>(src, n_src, n_normed, query, query_stride, dst, n_embd, n_tokens, scale, eps, query_first, stream); break;
        default: GGML_ABORT("attn_res_mix: %d columns a thread", cols);
    }
}
#endif // ATTN_RES_SPLIT

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
    const int64_t n_embd   = query->ne[0];
    const int     n_query  = query->ne[1];
    const float   scale    = ggml_get_op_params_f32(dst, 0);
    const float   eps      = ggml_get_op_params_f32(dst, 1);
    const int     n_normed = ggml_get_op_params_i32(dst, 2);
    const int64_t qs       = query->nb[1]/sizeof(float);
    const float * q        = (const float *) query->data;
    float *       d        = (float *) dst->data;
#ifdef ATTN_RES_SPLIT
    if (attn_res_split_usable(ctx.device, n_embd, n_query, dst->ne[2])) {
        // whether the query is a weight, which no kernel before this one writes
        const ggml_tensor * qsrc = query;
        while (qsrc->view_src) {
            qsrc = qsrc->view_src;
        }
        const bool query_first = qsrc->op == GGML_OP_NONE;
        const int  cols        = attn_res_split_cols(n_embd);
        const int  n_tokens    = dst->ne[2];
        switch (n_query) {
            case 1: attn_res_mix_split_cuda<1>(cols, src, n_src, n_normed, q, qs, d, n_embd, n_tokens, scale, eps, query_first, ctx.stream()); break;
            case 2: attn_res_mix_split_cuda<2>(cols, src, n_src, n_normed, q, qs, d, n_embd, n_tokens, scale, eps, query_first, ctx.stream()); break;
            case 3: attn_res_mix_split_cuda<3>(cols, src, n_src, n_normed, q, qs, d, n_embd, n_tokens, scale, eps, query_first, ctx.stream()); break;
            case 4: attn_res_mix_split_cuda<4>(cols, src, n_src, n_normed, q, qs, d, n_embd, n_tokens, scale, eps, query_first, ctx.stream()); break;
        }
        return;
    }
#endif // ATTN_RES_SPLIT
    const dim3 grid(dst->ne[2]);
    // Shallow mixes in large batches can reread the rows with less register pressure.
    const bool cache_sources = n_src > 2 || dst->ne[2] <= 512;
    if (cache_sources && n_embd == ATTN_RES_WIDTH && ggml_cuda_info().devices[ctx.device].cc == GGML_CUDA_CC_BLACKWELL) {
        if (n_query == 1) {
            attn_res_mix_4608<1><<<grid, ATTN_RES_CACHED_THREADS, 0, ctx.stream()>>>(src, n_src, n_normed, q, d, scale, eps);
            return;
        }
        if (n_query == 3) {
            attn_res_mix_4608<3><<<grid, ATTN_RES_CACHED_THREADS, 0, ctx.stream()>>>(src, n_src, n_normed, q, d, scale, eps);
            return;
        }
    }
    switch (n_query) {
        case 1: attn_res_mix_f32<1><<<grid, ATTN_RES_THREADS, 0, ctx.stream()>>>(src, n_src, n_normed, q, qs, d, n_embd, scale, eps); break;
        case 2: attn_res_mix_f32<2><<<grid, ATTN_RES_THREADS, 0, ctx.stream()>>>(src, n_src, n_normed, q, qs, d, n_embd, scale, eps); break;
        case 3: attn_res_mix_f32<3><<<grid, ATTN_RES_THREADS, 0, ctx.stream()>>>(src, n_src, n_normed, q, qs, d, n_embd, scale, eps); break;
        case 4: attn_res_mix_f32<4><<<grid, ATTN_RES_THREADS, 0, ctx.stream()>>>(src, n_src, n_normed, q, qs, d, n_embd, scale, eps); break;
        default: GGML_ABORT("attn_res_mix: %d queries", n_query);
    }
}
