#include "common.cuh"
#include "dsv4-hc.cuh"


static constexpr int DSV4_HC = 4;


static __device__ void dsv4_hc_comb_norm_cols(float * comb, float eps) {
    for (int idst = 0; idst < DSV4_HC; ++idst) {
        float sum = eps;
        for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
            sum += comb[idst + DSV4_HC*isrc];
        }

        const float inv_sum = 1.0f / sum;
        for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
            comb[idst + DSV4_HC*isrc] *= inv_sum;
        }
    }
}

static __device__ void dsv4_hc_comb_norm_rows(float * comb, float eps) {
    for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
        float sum = eps;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            sum += comb[idst + DSV4_HC*isrc];
        }

        const float inv_sum = 1.0f / sum;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            comb[idst + DSV4_HC*isrc] *= inv_sum;
        }
    }
}

// one token's comb from its mixes (m points at the token's column): row softmax over dst plus eps, then Sinkhorn
static __device__ void dsv4_hc_comb_token(float * comb, const float * m, int64_t sm0, const float * base, int64_t sb0,
        float scale_comb, float eps, int32_t n_iter) {
    constexpr int comb_offset = 2*DSV4_HC;

    for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
        float max = -INFINITY;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            const int idx = idst + DSV4_HC*isrc;
            const float v = m[(comb_offset + idx)*sm0] * scale_comb + base[(comb_offset + idx)*sb0];
            comb[idx] = v;
            max = fmaxf(max, v);
        }

        float sum = 0.0f;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            const int idx = idst + DSV4_HC*isrc;
            const float v = expf(comb[idx] - max);
            comb[idx] = v;
            sum += v;
        }

        const float inv_sum = 1.0f / sum;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            const int idx = idst + DSV4_HC*isrc;
            comb[idx] = comb[idx] * inv_sum + eps;
        }
    }

    dsv4_hc_comb_norm_cols(comb, eps);
    for (int32_t i = 1; i < n_iter; ++i) {
        dsv4_hc_comb_norm_rows(comb, eps);
        dsv4_hc_comb_norm_cols(comb, eps);
    }
}

static __global__ void dsv4_hc_comb_f32(
        const float * mixes,
        const float * scale,
        const float * base,
        float * dst,
        int64_t n_tokens,
        int64_t sm0,
        int64_t sm1,
        int64_t ss0,
        int64_t sb0,
        int64_t sd0,
        int64_t sd1,
        int64_t sd2,
        float eps,
        int32_t n_iter) {
    ggml_cuda_pdl_lc();
    const int64_t it = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;

    if (it >= n_tokens) {
        return;
    }

    ggml_cuda_pdl_sync();

    float comb[DSV4_HC*DSV4_HC];
    dsv4_hc_comb_token(comb, mixes + it*sm1, sm0, base, sb0, scale[2*ss0], eps, n_iter);

    for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            const int idx = idst + DSV4_HC*isrc;
            dst[idst*sd0 + isrc*sd1 + it*sd2] = comb[idx];
        }
    }
}

// one thread per token: the pre, post and comb weights of ggml_dsv4_hc_weights
static __global__ void dsv4_hc_weights_f32(
        const float * mixes,
        const float * scale,
        const float * base,
        float * dst,
        int64_t n_tokens,
        int64_t sm0,
        int64_t sm1,
        int64_t ss0,
        int64_t sb0,
        int64_t sd0,
        int64_t sd1,
        float eps,
        int32_t n_iter) {
    ggml_cuda_pdl_lc();
    const int64_t it = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;

    if (it >= n_tokens) {
        return;
    }

    ggml_cuda_pdl_sync();

    const float * m = mixes + it*sm1;
    float       * d = dst   + it*sd1;

    const float scale_pre  = scale[0*ss0];
    const float scale_post = scale[1*ss0];
    for (int h = 0; h < DSV4_HC; ++h) {
        const float pre  = m[h*sm0]             * scale_pre  + base[h*sb0];
        const float post = m[(DSV4_HC + h)*sm0] * scale_post + base[(DSV4_HC + h)*sb0];
        d[h*sd0]             = 1.0f/(1.0f + expf(-pre)) + eps;
        d[(DSV4_HC + h)*sd0] = 2.0f/(1.0f + expf(-post));
    }

    float comb[DSV4_HC*DSV4_HC];
    dsv4_hc_comb_token(comb, m, sm0, base, sb0, scale[2*ss0], eps, n_iter);

    for (int idx = 0; idx < DSV4_HC*DSV4_HC; ++idx) {
        d[(2*DSV4_HC + idx)*sd0] = comb[idx];
    }
}

static __global__ void dsv4_hc_pre_f32(
        const float * x,
        const float * weights,
        float * dst,
        int64_t n_embd,
        int64_t hc,
        int64_t n_tokens,
        int64_t sx0,
        int64_t sx1,
        int64_t sx2,
        int64_t sw0,
        int64_t sw1,
        int64_t sd0,
        int64_t sd1) {
    ggml_cuda_pdl_lc();
    const int64_t ir = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t nr = n_embd * n_tokens;

    if (ir >= nr) {
        return;
    }

    ggml_cuda_pdl_sync();

    const int64_t i0 = ir % n_embd;
    const int64_t it = ir / n_embd;

    float sum = x[i0*sx0 + it*sx2] * weights[it*sw1];
    for (int64_t ih = 1; ih < hc; ++ih) {
        const float xv = x[i0*sx0 + ih*sx1 + it*sx2];
        const float wv = weights[ih*sw0 + it*sw1];
        sum += xv * wv;
    }

    dst[i0*sd0 + it*sd1] = sum;
}

static __global__ void dsv4_hc_post_f32(
        const float * x,
        const float * residual,
        const float * post,
        const float * comb,
        float * dst,
        int64_t n_embd,
        int64_t hc,
        int64_t n_tokens,
        int64_t sx0,
        int64_t sx1,
        int64_t sr0,
        int64_t sr1,
        int64_t sr2,
        int64_t sp0,
        int64_t sp1,
        int64_t sc0,
        int64_t sc1,
        int64_t sc2,
        int64_t sd0,
        int64_t sd1,
        int64_t sd2) {
    ggml_cuda_pdl_lc();
    const int64_t ir = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t nr = n_embd * hc * n_tokens;

    if (ir >= nr) {
        return;
    }

    ggml_cuda_pdl_sync();

    const int64_t i0   = ir % n_embd;
    const int64_t idst = (ir / n_embd) % hc;
    const int64_t it   = ir / (n_embd * hc);

    float sum = x[i0*sx0 + it*sx1] * post[idst*sp0 + it*sp1];
    for (int64_t isrc = 0; isrc < hc; ++isrc) {
        sum += residual[i0*sr0 + isrc*sr1 + it*sr2] * comb[idst*sc0 + isrc*sc1 + it*sc2];
    }

    dst[i0*sd0 + idst*sd1 + it*sd2] = sum;
}

// The front of a hyper-connection cycle at a few tokens in two kernels (ggml_cuda_op_dsv4_hc_pre_fused), where it was
// six nodes: the weightless RMS_NORM of the flat streams, their MUL_MAT by hc_fn, DSV4_HC_WEIGHTS, DSV4_HC_PRE, and the
// RMS_NORM and MUL of the sublayer's norm. The mat-vec had one block a row, 24 blocks over 16,384 columns, and each of
// the other nodes a launch of its own.
static constexpr int DSV4_HC_MIX          = (2 + DSV4_HC)*DSV4_HC; // pre, post and comb mixes a token
static constexpr int DSV4_HC_MIX_SLICE    = 256;                   // flat columns a dsv4_hc_mix_partials block reads
static constexpr int DSV4_HC_PRE_NORM_Y   = 8;                     // mixed elements a dsv4_hc_pre_norm_f32 thread holds
static constexpr int DSV4_HC_PRE_NORM_THR = 1024;
static constexpr int64_t DSV4_HC_PRE_FUSED_MAX_TOKENS = 16;      // a decode's token or a draft's verify, not a prefill

static __device__ __forceinline__ void dsv4_hc_load8(const float * p, float * v) {
    const float4 a = ((const float4 *) p)[0];
    const float4 b = ((const float4 *) p)[1];
    v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w;
    v[4] = b.x; v[5] = b.y; v[6] = b.z; v[7] = b.w;
}

static __device__ __forceinline__ void dsv4_hc_load8(const half * p, float * v) {
    const uint4 u = *(const uint4 *) p;
    const half2 * h = (const half2 *) &u;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float2 f = __half22float2(h[j]);
        v[2*j + 0] = f.x;
        v[2*j + 1] = f.y;
    }
}

static __device__ __forceinline__ void dsv4_hc_load8(const nv_bfloat16 * p, float * v) {
    const uint4 u = *(const uint4 *) p;
    const nv_bfloat162 * h = (const nv_bfloat162 *) &u;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float2 f = __bfloat1622float2(h[j]);
        v[2*j + 0] = f.x;
        v[2*j + 1] = f.y;
    }
}

// a block for each slice of DSV4_HC_MIX_SLICE flat columns and each token: each of its 8 warps the dot products of 3 of
// hc_fn's rows with the token's slice, a lane 8 columns, and the slice's sum of squares, into partials, token it's row
// r at [(it*(DSV4_HC_MIX + 1) + r)*n_slices + slice] and its sum of squares as row DSV4_HC_MIX. w_prewait: hc_fn is
// the model's (dsv4_hc_prewait), so it is read before the PDL wait.
template <typename T>
static __global__ void __launch_bounds__(8*WARP_SIZE) dsv4_hc_mix_partials(
        const float * __restrict__ x, const T * __restrict__ w, float * __restrict__ partials,
        const int64_t sx1, const int64_t sw1, const bool w_prewait) {
    constexpr int rows = DSV4_HC_MIX/8;

    const int slice    = blockIdx.x;
    const int n_slices = gridDim.x;
    const int it       = blockIdx.y;
    const int warp     = threadIdx.x / WARP_SIZE;
    const int lane     = threadIdx.x % WARP_SIZE;
    const int64_t c0   = (int64_t) slice*DSV4_HC_MIX_SLICE + 8*lane;

    ggml_cuda_pdl_lc();

    if (!w_prewait) {
        ggml_cuda_pdl_sync();
    }
    float wv[rows][8];
#pragma unroll
    for (int r = 0; r < rows; ++r) {
        dsv4_hc_load8(w + (warp*rows + r)*sw1 + c0, wv[r]);
    }
    if (w_prewait) {
        ggml_cuda_pdl_sync();
    }

    float xv[8];
    dsv4_hc_load8(x + it*sx1 + c0, xv);

    float sumsq = 0.0f;
    float dot[rows] = {};
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        sumsq += xv[j]*xv[j];
#pragma unroll
        for (int r = 0; r < rows; ++r) {
            dot[r] += wv[r][j]*xv[j];
        }
    }

    float * p = partials + (int64_t) it*(DSV4_HC_MIX + 1)*n_slices + slice;
#pragma unroll
    for (int r = 0; r < rows; ++r) {
        dot[r] = warp_reduce_sum(dot[r]);
        if (lane == 0) {
            p[(warp*rows + r)*n_slices] = dot[r];
        }
    }
    if (warp == 0) {
        sumsq = warp_reduce_sum(sumsq);
        if (lane == 0) {
            p[DSV4_HC_MIX*n_slices] = sumsq;
        }
    }
}

// a token's comb from its mixes m (DSV4_HC_MIX, the comb's from 2*DSV4_HC), as dsv4_hc_comb_token makes it, by the
// first 16 lanes of a warp, lane L element L (dst L % DSV4_HC, src L / DSV4_HC): a row's sums over lanes 1 and 2 apart,
// a column's over lanes 4 and 8 apart; every lane of the warp calls it, lanes 16-31 mirroring 0-15
static __device__ float dsv4_hc_comb_lanes(const float * m, const float * base, float scale_comb, float eps,
        int32_t n_iter) {
    const int L = threadIdx.x % 16;

    const float v = m[2*DSV4_HC + L]*scale_comb + base[2*DSV4_HC + L];
    float max = v;
    max = fmaxf(max, __shfl_xor_sync(0xffffffff, max, 1));
    max = fmaxf(max, __shfl_xor_sync(0xffffffff, max, 2));
    const float e = expf(v - max);
    float sum = e + __shfl_xor_sync(0xffffffff, e, 1);
    sum += __shfl_xor_sync(0xffffffff, sum, 2);
    float c = e*(1.0f/sum) + eps;

    auto norm = [&](const int a, const int b) {
        float s = c + __shfl_xor_sync(0xffffffff, c, a);
        s += __shfl_xor_sync(0xffffffff, s, b);
        c *= 1.0f/(s + eps);
    };
    norm(4, 8);
    for (int32_t i = 1; i < n_iter; ++i) {
        norm(1, 2);
        norm(4, 8);
    }
    return c;
}

// a block for each token: the mixes from the partials (the dot products summed, times the streams' inverse RMS, as the
// mat-vec that folds the norm scales them), the weights as dsv4_hc_weights_f32 makes them into weights_out (warp 0,
// its comb by 16 lanes), and meanwhile, in the other warps, the streams mixed by the pre weights; then the mix
// RMS-normalized and multiplied by the norm's weight into dst. base_prewait: base is the model's (dsv4_hc_prewait), so
// it is read before the PDL wait.
static __global__ void __launch_bounds__(DSV4_HC_PRE_NORM_THR) dsv4_hc_pre_norm_f32(
        const float * __restrict__ partials, const int n_slices, const float * __restrict__ x,
        const float * __restrict__ scale, const float * __restrict__ base, const float * __restrict__ norm_w,
        float * __restrict__ weights_out, float * __restrict__ dst, const int64_t n_embd, const int64_t k,
        const int64_t sx1, const int64_t sx2, const int64_t ss0, const int64_t sb0, const int64_t sw0,
        const int64_t sw1, const int64_t sd1, const float eps_flat, const float eps_hc, const int32_t n_iter,
        const float eps_norm, const bool base_prewait) {
    __shared__ float mix[DSV4_HC_MIX + 1];
    __shared__ float base_s[DSV4_HC_MIX];
    __shared__ float sums[DSV4_HC_PRE_NORM_THR/WARP_SIZE];

    const int it   = blockIdx.x;
    const int warp = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;

    ggml_cuda_pdl_lc();

    if (!base_prewait) {
        ggml_cuda_pdl_sync();
    }
    if (threadIdx.x < DSV4_HC_MIX) {
        base_s[threadIdx.x] = base[threadIdx.x*sb0];
    }
    if (base_prewait) {
        ggml_cuda_pdl_sync();
    }

    if (warp <= DSV4_HC_MIX) {
        const float * p = partials + ((int64_t) it*(DSV4_HC_MIX + 1) + warp)*n_slices;
        float s = 0.0f;
        for (int sl = lane; sl < n_slices; sl += WARP_SIZE) {
            s += p[sl];
        }
        s = warp_reduce_sum(s);
        if (lane == 0) {
            mix[warp] = s;
        }
    }
    __syncthreads();

    const float rms_flat = rsqrtf(mix[DSV4_HC_MIX]/k + eps_flat);
    float pre[DSV4_HC];
#pragma unroll
    for (int h = 0; h < DSV4_HC; ++h) {
        pre[h] = 1.0f/(1.0f + expf(-(mix[h]*rms_flat*scale[0] + base_s[h]))) + eps_hc;
    }

    float y[DSV4_HC_PRE_NORM_Y];
    float sumsq = 0.0f;
    if (warp == 0) {
        float m[DSV4_HC_MIX];
#pragma unroll
        for (int r = 0; r < DSV4_HC_MIX; ++r) {
            m[r] = mix[r]*rms_flat;
        }
        const float c = dsv4_hc_comb_lanes(m, base_s, scale[2*ss0], eps_hc, n_iter);
        float * d = weights_out + it*sw1;
        if (lane < DSV4_HC*DSV4_HC) {
            d[(2*DSV4_HC + lane)*sw0] = c;
        } else if (lane < DSV4_HC*DSV4_HC + DSV4_HC) {
            const int h = lane - DSV4_HC*DSV4_HC;
            d[h*sw0]             = pre[h];
            d[(DSV4_HC + h)*sw0] = 2.0f/(1.0f + expf(-(m[DSV4_HC + h]*scale[ss0] + base_s[DSV4_HC + h])));
        }
    } else {
        const float * xt = x + it*sx2;
#pragma unroll
        for (int j = 0; j < DSV4_HC_PRE_NORM_Y; ++j) {
            const int64_t i = threadIdx.x - WARP_SIZE + (int64_t) j*(DSV4_HC_PRE_NORM_THR - WARP_SIZE);
            y[j] = 0.0f;
            if (i < n_embd) {
                float v = xt[i]*pre[0];
#pragma unroll
                for (int h = 1; h < DSV4_HC; ++h) {
                    v += xt[i + h*sx1]*pre[h];
                }
                y[j] = v;
                sumsq += v*v;
            }
        }
    }

    sumsq = block_reduce<block_reduce_method::SUM, DSV4_HC_PRE_NORM_THR>(sumsq, sums);
    const float rms = rsqrtf(sumsq/n_embd + eps_norm);

    if (warp > 0) {
        float * dt = dst + it*sd1;
#pragma unroll
        for (int j = 0; j < DSV4_HC_PRE_NORM_Y; ++j) {
            const int64_t i = threadIdx.x - WARP_SIZE + (int64_t) j*(DSV4_HC_PRE_NORM_THR - WARP_SIZE);
            if (i < n_embd) {
                dt[i] = y[j]*rms*norm_w[i];
            }
        }
    }
}

// a tensor a kernel may read before its PDL wait: in a weights buffer, which no kernel writes (a kernel before it may
// still be running when it starts)
static bool dsv4_hc_prewait(const ggml_tensor * t) {
    return t->buffer != nullptr && ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
}

bool ggml_cuda_dsv4_hc_pre_fused_supported(const ggml_tensor * rms_flat, const ggml_tensor * mm,
        const ggml_tensor * weights, const ggml_tensor * pre, const ggml_tensor * rms, const ggml_tensor * mul) {
    const ggml_tensor * flat   = rms_flat->src[0];
    const ggml_tensor * hc_fn  = mm->src[0];
    const ggml_tensor * x      = pre->src[0];
    const ggml_tensor * pre_w  = pre->src[1];
    const ggml_tensor * norm_w = mul->src[0] == rms ? mul->src[1] : mul->src[0];

    const int64_t n_embd   = x->ne[0];
    const int64_t n_tokens = x->ne[2];
    const int64_t k        = flat->ne[0];

    const bool shapes_ok = x->ne[1] == DSV4_HC && x->ne[3] == 1 && k == DSV4_HC*n_embd && flat->ne[1] == n_tokens &&
        ggml_nrows(flat) == n_tokens && hc_fn->ne[0] == k && hc_fn->ne[1] == DSV4_HC_MIX && ggml_nrows(hc_fn) == DSV4_HC_MIX &&
        mm->ne[0] == DSV4_HC_MIX && mm->ne[1] == n_tokens && weights->ne[1] == n_tokens &&
        rms->ne[0] == n_embd && ggml_nrows(rms) == n_tokens && norm_w->ne[0] == n_embd && ggml_nrows(norm_w) == 1 &&
        mul->ne[0] == n_embd && ggml_nrows(mul) == n_tokens;
    // the flat streams are x itself, contiguous; the pre weights the first DSV4_HC of each token's weights
    const bool layout_ok = flat->data == x->data && ggml_is_contiguous(x) && ggml_is_contiguous(flat) &&
        hc_fn->nb[0] == ggml_type_size(hc_fn->type) && hc_fn->nb[1] % 16 == 0 &&
        pre_w->view_src == weights && pre_w->view_offs == 0 && pre_w->nb[0] == weights->nb[0] &&
        pre_w->nb[1] == weights->nb[1] && ggml_is_contiguous(rms) && ggml_is_contiguous(mul) &&
        ggml_is_contiguous(norm_w) && (uintptr_t) x->data % 16 == 0 && (uintptr_t) hc_fn->data % 16 == 0;
    const bool types_ok = x->type == GGML_TYPE_F32 && flat->type == GGML_TYPE_F32 && mm->type == GGML_TYPE_F32 &&
        (hc_fn->type == GGML_TYPE_F32 || hc_fn->type == GGML_TYPE_F16 || hc_fn->type == GGML_TYPE_BF16) &&
        weights->type == GGML_TYPE_F32 && rms->type == GGML_TYPE_F32 && mul->type == GGML_TYPE_F32 &&
        norm_w->type == GGML_TYPE_F32 && weights->src[1]->type == GGML_TYPE_F32 && weights->src[2]->type == GGML_TYPE_F32;

    return shapes_ok && layout_ok && types_ok && n_tokens <= DSV4_HC_PRE_FUSED_MAX_TOKENS && k % DSV4_HC_MIX_SLICE == 0 &&
        n_embd <= (int64_t) DSV4_HC_PRE_NORM_Y*(DSV4_HC_PRE_NORM_THR - WARP_SIZE);
}

void ggml_cuda_op_dsv4_hc_pre_fused(ggml_backend_cuda_context & ctx, const ggml_tensor * rms_flat,
        const ggml_tensor * mm, ggml_tensor * weights, const ggml_tensor * pre, const ggml_tensor * rms,
        ggml_tensor * mul) {
    const ggml_tensor * hc_fn  = mm->src[0];
    const ggml_tensor * x      = pre->src[0];
    const ggml_tensor * scale  = weights->src[1];
    const ggml_tensor * base   = weights->src[2];
    const ggml_tensor * norm_w = mul->src[0] == rms ? mul->src[1] : mul->src[0];

    const int64_t n_embd   = x->ne[0];
    const int64_t n_tokens = x->ne[2];
    const int64_t k        = DSV4_HC*n_embd;
    const int     n_slices = k / DSV4_HC_MIX_SLICE;

    cudaStream_t stream = ctx.stream();
    ggml_cuda_pool_alloc<float> partials(ctx.pool(), n_tokens*(DSV4_HC_MIX + 1)*n_slices);

    const ggml_cuda_kernel_launch_params partial_params =
        ggml_cuda_kernel_launch_params(dim3(n_slices, n_tokens, 1), dim3(8*WARP_SIZE, 1, 1), 0, stream);
    const int64_t sx1 = x->nb[2] / sizeof(float);
    const int64_t sw1 = hc_fn->nb[1] / ggml_type_size(hc_fn->type);
    const bool w_prewait = dsv4_hc_prewait(hc_fn);
    switch (hc_fn->type) {
        case GGML_TYPE_F32:
            ggml_cuda_kernel_launch(dsv4_hc_mix_partials<float>, partial_params,
                (const float *) x->data, (const float *) hc_fn->data, partials.get(), sx1, sw1, w_prewait);
            break;
        case GGML_TYPE_F16:
            ggml_cuda_kernel_launch(dsv4_hc_mix_partials<half>, partial_params,
                (const float *) x->data, (const half *) hc_fn->data, partials.get(), sx1, sw1, w_prewait);
            break;
        case GGML_TYPE_BF16:
            ggml_cuda_kernel_launch(dsv4_hc_mix_partials<nv_bfloat16>, partial_params,
                (const float *) x->data, (const nv_bfloat16 *) hc_fn->data, partials.get(), sx1, sw1, w_prewait);
            break;
        default:
            GGML_ABORT("unsupported hc_fn type %s", ggml_type_name(hc_fn->type));
    }

    const ggml_cuda_kernel_launch_params norm_params =
        ggml_cuda_kernel_launch_params(dim3(n_tokens, 1, 1), dim3(DSV4_HC_PRE_NORM_THR, 1, 1), 0, stream);
    ggml_cuda_kernel_launch(dsv4_hc_pre_norm_f32, norm_params,
        (const float *) partials.get(), n_slices, (const float *) x->data, (const float *) scale->data,
        (const float *) base->data, (const float *) norm_w->data, (float *) weights->data, (float *) mul->data,
        n_embd, k, x->nb[1] / sizeof(float), x->nb[2] / sizeof(float), scale->nb[0] / sizeof(float),
        base->nb[0] / sizeof(float), weights->nb[0] / sizeof(float), weights->nb[1] / sizeof(float),
        mul->nb[1] / sizeof(float), ggml_get_op_params_f32(rms_flat, 0), ggml_get_op_params_f32(weights, 0),
        ggml_get_op_params_i32(weights, 1), ggml_get_op_params_f32(rms, 0), dsv4_hc_prewait(base));
}

void ggml_cuda_op_dsv4_hc_comb(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * mixes = dst->src[0];
    const ggml_tensor * scale = dst->src[1];
    const ggml_tensor * base  = dst->src[2];

    GGML_ASSERT(mixes->type == GGML_TYPE_F32);
    GGML_ASSERT(scale->type == GGML_TYPE_F32);
    GGML_ASSERT(base->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    constexpr int64_t hc_mix_dim = (2 + DSV4_HC)*DSV4_HC;

    GGML_ASSERT(mixes->ne[0] == hc_mix_dim);
    GGML_ASSERT(dst->ne[0] == DSV4_HC);
    GGML_ASSERT(dst->ne[1] == DSV4_HC);
    GGML_ASSERT(dst->ne[2] == mixes->ne[1]);
    GGML_ASSERT(scale->ne[0] >= 3);
    GGML_ASSERT(base->ne[0] == hc_mix_dim);

    GGML_TENSOR_LOCALS(size_t, nbm, mixes, nb);
    GGML_TENSOR_LOCALS(size_t, nbs, scale, nb);
    GGML_TENSOR_LOCALS(size_t, nbb, base,  nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,   nb);

    const int64_t n_tokens = mixes->ne[1];
    const float eps = ggml_get_op_params_f32(dst, 0);
    const int32_t n_iter = ggml_get_op_params_i32(dst, 1);

    const int block_size = 256;
    const dim3 block_dims(block_size, 1, 1);
    const dim3 grid_dims((n_tokens + block_size - 1) / block_size, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, ctx.stream());

    ggml_cuda_kernel_launch(dsv4_hc_comb_f32, launch_params,
            (const float *) mixes->data, (const float *) scale->data, (const float *) base->data, (float *) dst->data,
            n_tokens,
            nbm0 / sizeof(float), nbm1 / sizeof(float),
            nbs0 / sizeof(float),
            nbb0 / sizeof(float),
            nbd0 / sizeof(float), nbd1 / sizeof(float), nbd2 / sizeof(float),
            eps, n_iter);
}

void ggml_cuda_op_dsv4_hc_pre(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * x       = dst->src[0];
    const ggml_tensor * weights = dst->src[1];

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_LOCALS(size_t, nbx, x,       nb);
    GGML_TENSOR_LOCALS(size_t, nbw, weights, nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,     nb);

    const int64_t n_embd   = x->ne[0];
    const int64_t hc       = x->ne[1];
    const int64_t n_tokens = x->ne[2];

    const int block_size = 256;
    const int64_t nr = n_embd * n_tokens;
    const dim3 block_dims(block_size, 1, 1);
    const dim3 grid_dims((nr + block_size - 1) / block_size, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, ctx.stream());

    ggml_cuda_kernel_launch(dsv4_hc_pre_f32, launch_params,
            (const float *) x->data, (const float *) weights->data, (float *) dst->data,
            n_embd, hc, n_tokens,
            nbx0 / sizeof(float), nbx1 / sizeof(float), nbx2 / sizeof(float),
            nbw0 / sizeof(float), nbw1 / sizeof(float),
            nbd0 / sizeof(float), nbd1 / sizeof(float));
}

void ggml_cuda_op_dsv4_hc_post(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * x        = dst->src[0];
    const ggml_tensor * residual = dst->src[1];
    const ggml_tensor * post     = dst->src[2];
    const ggml_tensor * comb     = dst->src[3];

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(residual->type == GGML_TYPE_F32);
    GGML_ASSERT(post->type == GGML_TYPE_F32);
    GGML_ASSERT(comb->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_LOCALS(size_t, nbx, x,        nb);
    GGML_TENSOR_LOCALS(size_t, nbr, residual, nb);
    GGML_TENSOR_LOCALS(size_t, nbp, post,     nb);
    GGML_TENSOR_LOCALS(size_t, nbc, comb,     nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,      nb);

    const int64_t n_embd   = x->ne[0];
    const int64_t n_tokens = x->ne[1];
    const int64_t hc       = residual->ne[1];

    const int block_size = 256;
    const int64_t nr = n_embd * hc * n_tokens;
    const dim3 block_dims(block_size, 1, 1);
    const dim3 grid_dims((nr + block_size - 1) / block_size, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, ctx.stream());

    ggml_cuda_kernel_launch(dsv4_hc_post_f32, launch_params,
            (const float *) x->data, (const float *) residual->data,
            (const float *) post->data, (const float *) comb->data, (float *) dst->data,
            n_embd, hc, n_tokens,
            nbx0 / sizeof(float), nbx1 / sizeof(float),
            nbr0 / sizeof(float), nbr1 / sizeof(float), nbr2 / sizeof(float),
            nbp0 / sizeof(float), nbp1 / sizeof(float),
            nbc0 / sizeof(float), nbc1 / sizeof(float), nbc2 / sizeof(float),
            nbd0 / sizeof(float), nbd1 / sizeof(float), nbd2 / sizeof(float));
}

void ggml_cuda_op_dsv4_hc_weights(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * mixes = dst->src[0];
    const ggml_tensor * scale = dst->src[1];
    const ggml_tensor * base  = dst->src[2];

    GGML_ASSERT(mixes->type == GGML_TYPE_F32);
    GGML_ASSERT(scale->type == GGML_TYPE_F32);
    GGML_ASSERT(base->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    constexpr int64_t hc_mix_dim = (2 + DSV4_HC)*DSV4_HC;

    GGML_ASSERT(mixes->ne[0] == hc_mix_dim);
    GGML_ASSERT(dst->ne[0] == hc_mix_dim);
    GGML_ASSERT(dst->ne[1] == mixes->ne[1]);
    GGML_ASSERT(scale->ne[0] >= 3);
    GGML_ASSERT(base->ne[0] == hc_mix_dim);

    GGML_TENSOR_LOCALS(size_t, nbm, mixes, nb);
    GGML_TENSOR_LOCALS(size_t, nbs, scale, nb);
    GGML_TENSOR_LOCALS(size_t, nbb, base,  nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,   nb);

    const int64_t n_tokens = mixes->ne[1];
    const float eps = ggml_get_op_params_f32(dst, 0);
    const int32_t n_iter = ggml_get_op_params_i32(dst, 1);

    const int block_size = 256;
    const dim3 block_dims(block_size, 1, 1);
    const dim3 grid_dims((n_tokens + block_size - 1) / block_size, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, ctx.stream());

    ggml_cuda_kernel_launch(dsv4_hc_weights_f32, launch_params,
            (const float *) mixes->data, (const float *) scale->data, (const float *) base->data, (float *) dst->data,
            n_tokens,
            nbm0 / sizeof(float), nbm1 / sizeof(float),
            nbs0 / sizeof(float),
            nbb0 / sizeof(float),
            nbd0 / sizeof(float), nbd1 / sizeof(float),
            eps, n_iter);
}
