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

    // the product, then each stream's term fused in, rounded as written: the compiler may not contract them otherwise,
    // so dsv4_hc_mix_gram's fused post makes the same values
    float sum = __fmul_rn(x[i0*sx0 + it*sx1], post[idst*sp0 + it*sp1]);
    for (int64_t isrc = 0; isrc < hc; ++isrc) {
        sum = __fmaf_rn(residual[i0*sr0 + isrc*sr1 + it*sr2], comb[idst*sc0 + isrc*sc1 + it*sc2], sum);
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
// the model's (dsv4_hc_prewait), so it is read before the PDL wait. No kernel here takes __restrict__ pointers: under
// PDL they let the compiler load the kernel before's output ahead of the wait (GGML_CUDA_RESTRICT, upstream #24030).
template <typename T>
static __global__ void __launch_bounds__(8*WARP_SIZE) dsv4_hc_mix_partials(
        const float * x, const T * w, float * partials,
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

// the comb dsv4_hc_comb_lanes makes, bit for bit, in one thread's registers, c[L] its lane L's: a group's sum there is
// (c0 + c1) + (c2 + c3) in whichever lane holds it (addition commutes), as here. Its 20 iterations were a chain of 78
// dependent shuffles, each step's four sums waiting on one another's lanes; here a step's four groups are independent
// instructions, and m is read at constant indices (the lanes' m[2*DSV4_HC + L] kept it in local memory).
static __device__ __forceinline__ void dsv4_hc_comb_regs(float * c, const float * m, const float * base,
        const float scale_comb, const float eps, const int32_t n_iter) {
#pragma unroll
    for (int s = 0; s < DSV4_HC; ++s) {
        float v[DSV4_HC];
#pragma unroll
        for (int d = 0; d < DSV4_HC; ++d) {
            v[d] = m[2*DSV4_HC + d + DSV4_HC*s]*scale_comb + base[2*DSV4_HC + d + DSV4_HC*s];
        }
        const float max = fmaxf(fmaxf(v[0], v[1]), fmaxf(v[2], v[3]));
        float e[DSV4_HC];
#pragma unroll
        for (int d = 0; d < DSV4_HC; ++d) {
            e[d] = expf(v[d] - max);
        }
        const float sum = (e[0] + e[1]) + (e[2] + e[3]);
#pragma unroll
        for (int d = 0; d < DSV4_HC; ++d) {
            c[d + DSV4_HC*s] = e[d]*(1.0f/sum) + eps;
        }
    }

    // a column: one dst over the four srcs (the lanes' xor 4 and 8); a row: one src over the four dsts (xor 1 and 2)
    auto norm_cols = [&]() {
#pragma unroll
        for (int d = 0; d < DSV4_HC; ++d) {
            const float s = (c[d] + c[d + DSV4_HC]) + (c[d + 2*DSV4_HC] + c[d + 3*DSV4_HC]);
            const float inv = 1.0f/(s + eps);
#pragma unroll
            for (int k = 0; k < DSV4_HC; ++k) {
                c[d + DSV4_HC*k] *= inv;
            }
        }
    };
    auto norm_rows = [&]() {
#pragma unroll
        for (int r = 0; r < DSV4_HC; ++r) {
            const float s = (c[DSV4_HC*r] + c[DSV4_HC*r + 1]) + (c[DSV4_HC*r + 2] + c[DSV4_HC*r + 3]);
            const float inv = 1.0f/(s + eps);
#pragma unroll
            for (int k = 0; k < DSV4_HC; ++k) {
                c[DSV4_HC*r + k] *= inv;
            }
        }
    };
    norm_cols();
    for (int32_t i = 1; i < n_iter; ++i) {
        norm_rows();
        norm_cols();
    }
}

// a block for each token: the mixes from the partials (the dot products summed, times the streams' inverse RMS, as the
// mat-vec that folds the norm scales them), the weights as dsv4_hc_weights_f32 makes them into weights_out (warp 0,
// its comb by 16 lanes), and meanwhile, in the other warps, the streams mixed by the pre weights; then the mix
// RMS-normalized and multiplied by the norm's weight into dst. base_prewait: base is the model's (dsv4_hc_prewait), so
// it is read before the PDL wait. comb_regs: warp 0 makes the comb in each lane's registers (dsv4_hc_comb_regs), the
// same values; otherwise by 16 lanes (dsv4_hc_comb_lanes).
template <bool comb_regs>
static __global__ void __launch_bounds__(DSV4_HC_PRE_NORM_THR) dsv4_hc_pre_norm_f32(
        const float * partials, const int n_slices, const float * x,
        const float * scale, const float * base, const float * norm_w,
        float * weights_out, float * dst, const int64_t n_embd, const int64_t k,
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
        float * d = weights_out + it*sw1;
        if constexpr (comb_regs) {
            // every lane makes the whole comb and picks its element, and a stream's pre and post mix, by constant
            // indices: an index that varies by lane keeps an array in local memory
            float c[DSV4_HC*DSV4_HC];
            dsv4_hc_comb_regs(c, m, base_s, scale[2*ss0], eps_hc, n_iter);
            float c_lane = c[0];
#pragma unroll
            for (int k = 1; k < DSV4_HC*DSV4_HC; ++k) {
                c_lane = lane == k ? c[k] : c_lane;
            }
            const int h     = lane - DSV4_HC*DSV4_HC;
            float     pre_h = pre[0];
            float     m_h   = m[DSV4_HC];
#pragma unroll
            for (int k = 1; k < DSV4_HC; ++k) {
                pre_h = h == k ? pre[k]         : pre_h;
                m_h   = h == k ? m[DSV4_HC + k] : m_h;
            }
            if (lane < DSV4_HC*DSV4_HC) {
                d[(2*DSV4_HC + lane)*sw0] = c_lane;
            } else if (lane < DSV4_HC*DSV4_HC + DSV4_HC) {
                d[h*sw0]             = pre_h;
                d[(DSV4_HC + h)*sw0] = 2.0f/(1.0f + expf(-(m_h*scale[ss0] + base_s[DSV4_HC + h])));
            }
        } else {
            const float c = dsv4_hc_comb_lanes(m, base_s, scale[2*ss0], eps_hc, n_iter);
            if (lane < DSV4_HC*DSV4_HC) {
                d[(2*DSV4_HC + lane)*sw0] = c;
            } else if (lane < DSV4_HC*DSV4_HC + DSV4_HC) {
                const int h = lane - DSV4_HC*DSV4_HC;
                d[h*sw0]             = pre[h];
                d[(DSV4_HC + h)*sw0] = 2.0f/(1.0f + expf(-(m[DSV4_HC + h]*scale[ss0] + base_s[DSV4_HC + h])));
            }
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

// The front in blocks that each write a slice of the normed mix, where dsv4_hc_pre_norm_f32 was one block a token: the
// mix's RMS comes from the streams' Gram matrix G (the mix is sum_h pre[h] x[h], so its sum of squares is pre' G pre),
// which the partials carry beside the mixes, so no block waits for the others' part of the mix; the Sinkhorn runs in
// block 0 alone while the others write their slices.
static constexpr int DSV4_HC_GRAM       = DSV4_HC*(DSV4_HC + 1)/2;   // G's upper triangle, row by row
static constexpr int DSV4_HC_GRAM_ROWS  = DSV4_HC_MIX + DSV4_HC_GRAM; // a slice's partials: the dot products, then G
static constexpr int DSV4_HC_GRAM_SLICE = 64;                         // the columns of each stream a slice holds
static constexpr int DSV4_HC_PRE_GRAM_THR = 256;

// G's entry (a, b), a <= b, in its upper triangle's order
static __device__ __forceinline__ int dsv4_hc_gram_index(const int a, const int b) {
    return a*DSV4_HC - a*(a - 1)/2 + (b - a);
}

// a block for each slice and token: the slice is columns [64 slice, 64 slice + 64) of every stream, lane L 8 of stream
// L / 8's, so a lane's columns in the other streams are on the lanes 8 apart. Each of its 8 warps the dot products of 3
// of hc_fn's rows with the slice, and warps 0-2 G's entries over it: warp p pairs a lane's stream g with stream
// (g + p) % 4 (p = 0 the diagonal, 1 the neighbours, 2 the two opposite pairs, which streams 0 and 1 write). Token it's
// row r at [(it*DSV4_HC_GRAM_ROWS + r)*n_slices + slice]. w_prewait: hc_fn is the model's (dsv4_hc_prewait).
// fuse_post: the streams are the previous sublayer's DSV4_HC_POST, made here as dsv4_hc_post_f32 makes them (the same
// product and fused terms in its order, each rounded as written, so the same values) from its output xo, the streams
// before it (residual) and its post and comb weights, and written to x by warp 0; the post's inputs are contiguous
// along n_embd (dsv4_hc_post_pre_fused_supported).
struct dsv4_hc_post_args {
    const float * xo;       // the sublayer's output [n_embd, n_tokens], token stride sxo1
    const float * residual; // [n_embd, DSV4_HC, n_tokens], strides sr1 and sr2
    const float * post;     // [DSV4_HC, n_tokens], strides sp0 and sp1
    const float * comb;     // [DSV4_HC dst, DSV4_HC src, n_tokens], strides sc0, sc1 and sc2
    int64_t sxo1, sr1, sr2, sp0, sp1, sc0, sc1, sc2;
    int64_t sx_h;           // x's stream stride (its token stride is the kernel's sx1)
};

// dsv4_hc_mix_gram's work, a block's: also the first part of dsv4_hc_front_one's
template <typename T, bool fuse_post>
static __device__ __forceinline__ void dsv4_hc_mix_gram_block(
        float * x, const T * w, float * partials, const int64_t n_embd,
        const int64_t sx1, const int64_t sw1, const bool w_prewait, const dsv4_hc_post_args & pa) {
    constexpr int rows = DSV4_HC_MIX/8;

    const int slice    = blockIdx.x;
    const int n_slices = gridDim.x;
    const int it       = blockIdx.y;
    const int warp     = threadIdx.x / WARP_SIZE;
    const int lane     = threadIdx.x % WARP_SIZE;
    const int g        = lane / 8;
    const int64_t c0   = g*n_embd + (int64_t) slice*DSV4_HC_GRAM_SLICE + 8*(lane % 8);

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
    if constexpr (fuse_post) {
        const int64_t i0 = (int64_t) slice*DSV4_HC_GRAM_SLICE + 8*(lane % 8);
        float xo[8];
        float r[DSV4_HC][8];
        dsv4_hc_load8(pa.xo + it*pa.sxo1 + i0, xo);
#pragma unroll
        for (int s = 0; s < DSV4_HC; ++s) {
            dsv4_hc_load8(pa.residual + it*pa.sr2 + s*pa.sr1 + i0, r[s]);
        }
        const float pw = pa.post[g*pa.sp0 + it*pa.sp1];
        float cw[DSV4_HC];
#pragma unroll
        for (int s = 0; s < DSV4_HC; ++s) {
            cw[s] = pa.comb[g*pa.sc0 + s*pa.sc1 + it*pa.sc2];
        }
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            float v = __fmul_rn(xo[j], pw);
#pragma unroll
            for (int s = 0; s < DSV4_HC; ++s) {
                v = __fmaf_rn(r[s][j], cw[s], v);
            }
            xv[j] = v;
        }
        if (warp == 0) {
            float4 * xn = (float4 *) (x + it*sx1 + g*pa.sx_h + i0);
            xn[0] = make_float4(xv[0], xv[1], xv[2], xv[3]);
            xn[1] = make_float4(xv[4], xv[5], xv[6], xv[7]);
        }
    } else {
        dsv4_hc_load8(x + it*sx1 + c0, xv);
    }

    float dot[rows] = {};
#pragma unroll
    for (int j = 0; j < 8; ++j) {
#pragma unroll
        for (int r = 0; r < rows; ++r) {
            dot[r] += wv[r][j]*xv[j];
        }
    }

    float * p = partials + (int64_t) it*DSV4_HC_GRAM_ROWS*n_slices + slice;
#pragma unroll
    for (int r = 0; r < rows; ++r) {
        dot[r] = warp_reduce_sum(dot[r]);
        if (lane == 0) {
            p[(warp*rows + r)*n_slices] = dot[r];
        }
    }
    if (warp < 3) {
        const int gb  = (g + warp) % DSV4_HC;
        const int src = lane % 8 + 8*gb;
        float s = 0.0f;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            s += xv[j]*__shfl_sync(0xffffffff, xv[j], src);
        }
        s += __shfl_xor_sync(0xffffffff, s, 1);
        s += __shfl_xor_sync(0xffffffff, s, 2);
        s += __shfl_xor_sync(0xffffffff, s, 4);
        if (lane % 8 == 0 && (warp < 2 || g < 2)) {
            p[(DSV4_HC_MIX + dsv4_hc_gram_index(min(g, gb), max(g, gb)))*n_slices] = s;
        }
    }
}

template <typename T, bool fuse_post>
static __global__ void __launch_bounds__(8*WARP_SIZE) dsv4_hc_mix_gram(
        float * x, const T * w, float * partials, const int64_t n_embd,
        const int64_t sx1, const int64_t sw1, const bool w_prewait, const dsv4_hc_post_args pa) {
    dsv4_hc_mix_gram_block<T, fuse_post>(x, w, partials, n_embd, sx1, sw1, w_prewait, pa);
}

// a block for each DSV4_HC_PRE_GRAM_THR elements of a token's mix: every block sums the partials into the mixes and G
// (from L2, a few KB), makes the pre weights, and the mix's RMS as sqrt(pre' G pre / n_embd + eps); then a thread its
// element of the mix, normed, times the norm's weight, into dst. Block 0's warp 0 also makes the post and comb weights
// (the comb in registers, dsv4_hc_comb_regs) into weights_out, while the other blocks write their slices. The streams'
// RMS is G's trace's. base_prewait, norm_prewait: base or the norm's weight is the model's (dsv4_hc_prewait).
// q8 non-null: dst's q8_1 copy as quantize_row_q8_1_cuda writes it (token it's row q8_s1 blocks on), a warp a block
// with quantize_q8_1's arithmetic on the values dst holds, so its bits: n_embd a multiple of MATRIX_ROW_PADDING.
// Its work is in the device functions below, which dsv4_hc_front_one runs too, so the two make the same bits.
struct dsv4_hc_pre_args {
    const float * x;           // the streams [n_embd, DSV4_HC, n_tokens], stream stride sx1, token stride sx2
    const float * scale;       // stride ss0
    const float * base;        // stride sb0
    const float * norm_w;
    float       * weights_out; // stride sw0, token stride sw1
    float       * dst;         // token stride sd1
    block_q8_1  * q8;          // or nullptr; token stride q8_s1 blocks
    int64_t n_embd, k, sx1, sx2, ss0, sb0, sw0, sw1, sd1, q8_s1;
    float   eps_flat, eps_hc, eps_norm;
    int32_t n_iter;
    bool    base_prewait, norm_prewait;
};

// a token's mixes and G from its partials pt, into mix (shared): warp w sums rows w, w + 8, ..., every row's loads issued
// before any row's shuffles, one L2 round trip, not one a row. cg: read past L1 (the launch's own blocks wrote them). The
// caller syncs the block before mix is read.
template <bool cg>
static __device__ __forceinline__ void dsv4_hc_gram_mix(const float * pt, const int n_slices, float * mix) {
    const int warp = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;

    constexpr int n_warps = DSV4_HC_PRE_GRAM_THR/WARP_SIZE;
    constexpr int rows_w  = (DSV4_HC_GRAM_ROWS + n_warps - 1)/n_warps;
    float s[rows_w];
#pragma unroll
    for (int q = 0; q < rows_w; ++q) {
        s[q] = 0.0f;
    }
    for (int sl = lane; sl < n_slices; sl += WARP_SIZE) {
#pragma unroll
        for (int q = 0; q < rows_w; ++q) {
            const int r = warp + q*n_warps;
            if (r < DSV4_HC_GRAM_ROWS) {
                s[q] += cg ? __ldcg(pt + r*n_slices + sl) : pt[r*n_slices + sl];
            }
        }
    }
#pragma unroll
    for (int q = 0; q < rows_w; ++q) {
        const int r = warp + q*n_warps;
        s[q] = warp_reduce_sum(s[q]);
        if (lane == 0 && r < DSV4_HC_GRAM_ROWS) {
            mix[r] = s[q];
        }
    }
}

// element i's streams of token it (i < n_embd); cg: read past L1 (the launch's own blocks wrote them)
template <bool cg>
static __device__ __forceinline__ void dsv4_hc_gram_streams(const dsv4_hc_pre_args & pr, const int it, const int64_t i,
        float * xs) {
    const float * xt = pr.x + it*pr.sx2;
#pragma unroll
    for (int h = 0; h < DSV4_HC; ++h) {
        xs[h] = cg ? __ldcg(xt + i + h*pr.sx1) : xt[i + h*pr.sx1];
    }
}

// the pre weights and the mix's RMS from mix (dsv4_hc_gram_mix's) and base_s, and the streams' RMS (G's trace's)
static __device__ __forceinline__ void dsv4_hc_gram_pre(const float * mix, const float * base_s, const dsv4_hc_pre_args & pr,
        float * pre, float & rms_flat, float & rms) {
    const float * G = mix + DSV4_HC_MIX;
    float trace = 0.0f;
#pragma unroll
    for (int h = 0; h < DSV4_HC; ++h) {
        trace += G[dsv4_hc_gram_index(h, h)];
    }
    rms_flat = rsqrtf(trace/pr.k + pr.eps_flat);
#pragma unroll
    for (int h = 0; h < DSV4_HC; ++h) {
        pre[h] = 1.0f/(1.0f + expf(-(mix[h]*rms_flat*pr.scale[0] + base_s[h]))) + pr.eps_hc;
    }
    // pre' G pre: G symmetric, its off-diagonal entries twice
    float sumsq = 0.0f;
#pragma unroll
    for (int a = 0; a < DSV4_HC; ++a) {
        float row = pre[a]*G[dsv4_hc_gram_index(a, a)];
#pragma unroll
        for (int b = a + 1; b < DSV4_HC; ++b) {
            row += 2.0f*pre[b]*G[dsv4_hc_gram_index(a, b)];
        }
        sumsq += pre[a]*row;
    }
    rms = rsqrtf(sumsq/pr.n_embd + pr.eps_norm);
}

// token it's pre, post and comb weights into weights_out, by one warp: the comb in registers (dsv4_hc_comb_regs), lane
// L < 16 writing its element, lanes 16-19 a pre and a post weight
static __device__ __forceinline__ void dsv4_hc_gram_weights(const float * mix, const float * base_s,
        const dsv4_hc_pre_args & pr, const int it, const float * pre, const float rms_flat) {
    const int lane = threadIdx.x % WARP_SIZE;

    float m[DSV4_HC_MIX];
#pragma unroll
    for (int r = 0; r < DSV4_HC_MIX; ++r) {
        m[r] = mix[r]*rms_flat;
    }
    float c[DSV4_HC*DSV4_HC];
    dsv4_hc_comb_regs(c, m, base_s, pr.scale[2*pr.ss0], pr.eps_hc, pr.n_iter);
    float c_lane = c[0];
#pragma unroll
    for (int j = 1; j < DSV4_HC*DSV4_HC; ++j) {
        c_lane = lane == j ? c[j] : c_lane;
    }
    const int h     = lane - DSV4_HC*DSV4_HC;
    float     pre_h = pre[0];
    float     m_h   = m[DSV4_HC];
#pragma unroll
    for (int j = 1; j < DSV4_HC; ++j) {
        pre_h = h == j ? pre[j]         : pre_h;
        m_h   = h == j ? m[DSV4_HC + j] : m_h;
    }
    float * d = pr.weights_out + it*pr.sw1;
    if (lane < DSV4_HC*DSV4_HC) {
        d[(2*DSV4_HC + lane)*pr.sw0] = c_lane;
    } else if (lane < DSV4_HC*DSV4_HC + DSV4_HC) {
        d[h*pr.sw0]             = pre_h;
        d[(DSV4_HC + h)*pr.sw0] = 2.0f/(1.0f + expf(-(m_h*pr.scale[pr.ss0] + base_s[DSV4_HC + h])));
    }
}

// token it's element i of the mix (i < n_embd) from its streams xs, normed, times the norm's weight nw, into dst, and its
// q8_1 copy when pr.q8: the warp's 32 values are a block (n_embd a multiple of 32: a warp is all in or all out)
static __device__ __forceinline__ void dsv4_hc_gram_out(const dsv4_hc_pre_args & pr, const int it, const int64_t i,
        const float * xs, const float * pre, const float rms, const float nw) {
    const int lane = threadIdx.x % WARP_SIZE;

    float v = xs[0]*pre[0];
#pragma unroll
    for (int h = 1; h < DSV4_HC; ++h) {
        v += xs[h]*pre[h];
    }
    const float xi = v*rms*nw;
    pr.dst[it*pr.sd1 + i] = xi;
    if (pr.q8 != nullptr) {
        float amax = fabsf(xi);
        float sum  = xi;
        amax = warp_reduce_max<QK8_1>(amax);
        sum  = warp_reduce_sum<QK8_1>(sum);
        const float  d = amax / 127.0f;
        const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);
        block_q8_1 * blk = pr.q8 + it*pr.q8_s1 + i/QK8_1;
        blk->qs[lane] = q;
        if (lane == 0) {
            blk->ds = make_half2(d, sum);
        }
    }
}

static __global__ void __launch_bounds__(DSV4_HC_PRE_GRAM_THR) dsv4_hc_pre_gram_f32(
        const float * partials, const int n_slices, const dsv4_hc_pre_args pr) {
    __shared__ float mix[DSV4_HC_GRAM_ROWS];
    __shared__ float base_s[DSV4_HC_MIX];

    const int     it   = blockIdx.y;
    const int     warp = threadIdx.x / WARP_SIZE;
    const int64_t i    = (int64_t) blockIdx.x*DSV4_HC_PRE_GRAM_THR + threadIdx.x;

    ggml_cuda_pdl_lc();

    float nw = 0.0f;
    if (pr.base_prewait && threadIdx.x < DSV4_HC_MIX) {
        base_s[threadIdx.x] = pr.base[threadIdx.x*pr.sb0];
    }
    if (pr.norm_prewait && i < pr.n_embd) {
        nw = pr.norm_w[i];
    }
    ggml_cuda_pdl_sync();
    if (!pr.base_prewait && threadIdx.x < DSV4_HC_MIX) {
        base_s[threadIdx.x] = pr.base[threadIdx.x*pr.sb0];
    }
    if (!pr.norm_prewait && i < pr.n_embd) {
        nw = pr.norm_w[i];
    }
    // the thread's streams, requested with the partials: one round trip for both
    float xs[DSV4_HC] = {};
    if (i < pr.n_embd) {
        dsv4_hc_gram_streams<false>(pr, it, i, xs);
    }
    dsv4_hc_gram_mix<false>(partials + (int64_t) it*DSV4_HC_GRAM_ROWS*n_slices, n_slices, mix);
    __syncthreads();

    float pre[DSV4_HC];
    float rms_flat;
    float rms;
    dsv4_hc_gram_pre(mix, base_s, pr, pre, rms_flat, rms);

    // block 0's warp 0 makes the weights first (the Sinkhorn is the longest chain), then its slice like every warp
    if (blockIdx.x == 0 && warp == 0) {
        dsv4_hc_gram_weights(mix, base_s, pr, it, pre, rms_flat);
    }
    if (i < pr.n_embd) {
        dsv4_hc_gram_out(pr, it, i, xs, pre, rms, nw);
    }
}

// The front in one launch: dsv4_hc_mix_gram's blocks, after which the block that takes its token's last ticket (one a
// token at tickets, ggml_cuda_hc_front_tickets) sets it back to 0 and does dsv4_hc_pre_gram_f32's work for the whole
// token with its device functions, so its bits: the sums from every block's partials, the weights by warp 0, then a
// thread every DSV4_HC_PRE_GRAM_THR-th element, the streams of DSV4_HC_FRONT_ONE_CH of them requested at once. The
// partials and, under fuse_post, the streams are the launch's own blocks' writes, so they are read past L1. Every input
// dsv4_hc_pre_gram_f32 reads is read past the PDL wait mix_gram's part took.
static constexpr int DSV4_HC_FRONT_ONE_CH = 8;

template <typename T, bool fuse_post>
static __global__ void __launch_bounds__(8*WARP_SIZE) dsv4_hc_front_one(
        float * x, const T * w, float * partials, const int64_t sx1, const int64_t sw1, const bool w_prewait,
        const dsv4_hc_post_args pa, const dsv4_hc_pre_args pr, unsigned int * tickets) {
    static_assert(8*WARP_SIZE == DSV4_HC_PRE_GRAM_THR, "the last block does a dsv4_hc_pre_gram_f32 block's work");
    __shared__ float mix[DSV4_HC_GRAM_ROWS];
    __shared__ float base_s[DSV4_HC_MIX];
    __shared__ bool  last;

    dsv4_hc_mix_gram_block<T, fuse_post>(x, w, partials, pr.n_embd, sx1, sw1, w_prewait, pa);

    // every writer's partials (and slice of the streams) seen device-wide before the block's ticket
    const int it       = blockIdx.y;
    const int n_slices = gridDim.x;
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
        const unsigned int t = atomicAdd(tickets + it, 1u);
        last = t == (unsigned int) n_slices - 1;
        if (last) {
            tickets[it] = 0;
        }
    }
    __syncthreads();
    if (!last) {
        return;
    }
    __threadfence();

    const int warp     = threadIdx.x / WARP_SIZE;
    const int n_chunks = (int) ((pr.n_embd + DSV4_HC_PRE_GRAM_THR - 1) / DSV4_HC_PRE_GRAM_THR);
    if (threadIdx.x < DSV4_HC_MIX) {
        base_s[threadIdx.x] = pr.base[threadIdx.x*pr.sb0];
    }
    float xs[DSV4_HC_FRONT_ONE_CH][DSV4_HC];
    float nw[DSV4_HC_FRONT_ONE_CH];
    auto request = [&](const int c0) {
#pragma unroll
        for (int c = 0; c < DSV4_HC_FRONT_ONE_CH; ++c) {
            const int64_t i = (int64_t) (c0 + c)*DSV4_HC_PRE_GRAM_THR + threadIdx.x;
            if (c0 + c < n_chunks && i < pr.n_embd) {
                dsv4_hc_gram_streams<fuse_post>(pr, it, i, xs[c]);
                nw[c] = pr.norm_w[i];
            }
        }
    };
    // the first chunks' streams requested with the partials
    request(0);
    dsv4_hc_gram_mix<true>(partials + (int64_t) it*DSV4_HC_GRAM_ROWS*n_slices, n_slices, mix);
    __syncthreads();

    float pre[DSV4_HC];
    float rms_flat;
    float rms;
    dsv4_hc_gram_pre(mix, base_s, pr, pre, rms_flat, rms);
    if (warp == 0) {
        dsv4_hc_gram_weights(mix, base_s, pr, it, pre, rms_flat);
    }
    for (int c0 = 0; c0 < n_chunks; c0 += DSV4_HC_FRONT_ONE_CH) {
        if (c0 > 0) {
            request(c0);
        }
#pragma unroll
        for (int c = 0; c < DSV4_HC_FRONT_ONE_CH; ++c) {
            const int64_t i = (int64_t) (c0 + c)*DSV4_HC_PRE_GRAM_THR + threadIdx.x;
            if (c0 + c < n_chunks && i < pr.n_embd) {
                dsv4_hc_gram_out(pr, it, i, xs[c], pre, rms, nw[c]);
            }
        }
    }
}

// GGML_CUDA_HC_FRONT_CHECK=1: the one launch's normed mix, q8_1 copy and weights (one) against the two kernels' (two),
// bit for bit; a block a token
static __global__ void dsv4_hc_front_check_bits(const dsv4_hc_pre_args one, const dsv4_hc_pre_args two) {
    const int it = blockIdx.y;
    for (int64_t i = threadIdx.x; i < one.n_embd; i += blockDim.x) {
        const uint32_t a = __float_as_uint(one.dst[it*one.sd1 + i]);
        const uint32_t b = __float_as_uint(two.dst[it*two.sd1 + i]);
        if (a != b) {
            printf("hc front check: token %d element %lld is %08x from the one launch, %08x from the two\n",
                    it, (long long) i, a, b);
            __trap();
        }
        if (one.q8 != nullptr && i % QK8_1 == 0) {
            const block_q8_1 * qa = one.q8 + it*one.q8_s1 + i/QK8_1;
            const block_q8_1 * qb = two.q8 + it*two.q8_s1 + i/QK8_1;
            bool same = *(const uint32_t *) &qa->ds == *(const uint32_t *) &qb->ds;
            for (int j = 0; j < QK8_1; ++j) {
                same = same && qa->qs[j] == qb->qs[j];
            }
            if (!same) {
                printf("hc front check: token %d q8_1 block %lld differs\n", it, (long long) (i/QK8_1));
                __trap();
            }
        }
    }
    if (threadIdx.x < DSV4_HC_MIX) {
        const uint32_t a = __float_as_uint(one.weights_out[it*one.sw1 + threadIdx.x*one.sw0]);
        const uint32_t b = __float_as_uint(two.weights_out[it*two.sw1 + threadIdx.x*two.sw0]);
        if (a != b) {
            printf("hc front check: token %d weight %d is %08x from the one launch, %08x from the two\n",
                    it, (int) threadIdx.x, a, b);
            __trap();
        }
    }
}

// a tensor a kernel may read before its PDL wait: in a weights buffer, which no kernel writes (a kernel before it may
// still be running when it starts)
static bool dsv4_hc_prewait(const ggml_tensor * t) {
    return t->buffer != nullptr && ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
}

// GGML_CUDA_HC_PRE_GRAM_LEGACY=1: the front's second kernel one block a token (dsv4_hc_pre_norm_f32), its partials the
// mixes and the flat sum of squares over slices of one stream
static bool dsv4_hc_pre_gram_legacy() {
    static const bool legacy = ggml_env_switch("GGML_CUDA_HC_PRE_GRAM_LEGACY");
    return legacy;
}

// GGML_CUDA_HC_FRONT_ONE_LEGACY=1: the Gram path's front in its two kernels, not one launch (dsv4_hc_front_one).
// GGML_CUDA_HC_FRONT_CHECK=1: both, the two kernels into scratch, and a trap on any bit of the outputs that differs.
static bool dsv4_hc_front_one_legacy() {
    static const bool legacy = ggml_env_switch("GGML_CUDA_HC_FRONT_ONE_LEGACY");
    return legacy;
}

static bool dsv4_hc_front_check() {
    static const bool check = ggml_env_switch("GGML_CUDA_HC_FRONT_CHECK");
    return check;
}

bool ggml_cuda_dsv4_hc_writes_q8_1(const ggml_tensor * node) {
    if (dsv4_hc_pre_gram_legacy() || node->op != GGML_OP_MUL || node->type != GGML_TYPE_F32 ||
            node->ne[0] % MATRIX_ROW_PADDING != 0) {
        return false;
    }
    const ggml_tensor * rms = node->src[0] != nullptr && node->src[0]->op == GGML_OP_RMS_NORM ? node->src[0] : node->src[1];
    return rms != nullptr && rms->op == GGML_OP_RMS_NORM && rms->src[0] != nullptr &&
        rms->src[0]->op == GGML_OP_DSV4_HC_PRE;
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

    // the one-block path holds a token's mix in its threads
    return shapes_ok && layout_ok && types_ok && n_tokens <= DSV4_HC_PRE_FUSED_MAX_TOKENS && k % DSV4_HC_MIX_SLICE == 0 &&
        (!dsv4_hc_pre_gram_legacy() || n_embd <= (int64_t) DSV4_HC_PRE_NORM_Y*(DSV4_HC_PRE_NORM_THR - WARP_SIZE));
}

bool ggml_cuda_dsv4_hc_post_pre_fused_supported(const ggml_tensor * post, const ggml_tensor * rms_flat,
        const ggml_tensor * mm, const ggml_tensor * weights, const ggml_tensor * pre, const ggml_tensor * rms,
        const ggml_tensor * mul) {
    if (dsv4_hc_pre_gram_legacy() || !ggml_cuda_dsv4_hc_pre_fused_supported(rms_flat, mm, weights, pre, rms, mul)) {
        return false;
    }
    const ggml_tensor * xo       = post->src[0];
    const ggml_tensor * residual = post->src[1];
    const ggml_tensor * post_w   = post->src[2];
    const ggml_tensor * comb_w   = post->src[3];

    const int64_t n_embd   = post->ne[0];
    const int64_t n_tokens = post->ne[2];

    // the front's streams are the post's output; its streams and output are read 16 bytes at a time along n_embd
    const bool shapes_ok = pre->src[0] == post && xo->ne[0] == n_embd && xo->ne[1] == n_tokens && ggml_nrows(xo) == n_tokens &&
        residual->ne[0] == n_embd && residual->ne[1] == DSV4_HC && residual->ne[2] == n_tokens && residual->ne[3] == 1 &&
        post_w->ne[0] == DSV4_HC && post_w->ne[1] == n_tokens && comb_w->ne[0] == DSV4_HC && comb_w->ne[1] == DSV4_HC &&
        comb_w->ne[2] == n_tokens;
    const bool layout_ok = xo->nb[0] == sizeof(float) && xo->nb[1] % 16 == 0 && (uintptr_t) xo->data % 16 == 0 &&
        residual->nb[0] == sizeof(float) && residual->nb[1] % 16 == 0 && residual->nb[2] % 16 == 0 &&
        (uintptr_t) residual->data % 16 == 0 && post->nb[1] % 16 == 0;
    const bool types_ok = post->type == GGML_TYPE_F32 && xo->type == GGML_TYPE_F32 && residual->type == GGML_TYPE_F32 &&
        post_w->type == GGML_TYPE_F32 && comb_w->type == GGML_TYPE_F32;
    // the fused kernels write the post's output and read it back, where the fusion's memory check takes an
    // intermediate as never written: an output the allocator placed over it (it may where nothing reads the streams
    // after the front, as in a model's last sublayer or a test) would be written while other blocks read it. The first
    // kernel's blocks write the streams while others read the post's inputs, so the streams lie over none of those.
    // The second writes the weights and the normed mix only after its PDL wait, once the first has read every input,
    // so those may lie over the post's inputs, as the allocator places them: the post is its inputs' last reader.
    auto overlaps = [](const ggml_tensor * a, const ggml_tensor * b) {
        const char * a0 = (const char *) a->data;
        const char * b0 = (const char *) b->data;
        return a0 < b0 + ggml_nbytes(b) && b0 < a0 + ggml_nbytes(a);
    };
    const bool alias_ok = !overlaps(post, mul) && !overlaps(post, weights) && !overlaps(post, xo) &&
        !overlaps(post, residual) && !overlaps(post, post_w) && !overlaps(post, comb_w);
    return shapes_ok && layout_ok && types_ok && alias_ok;
}

// the front, after the previous sublayer's DSV4_HC_POST (post) when not null: the Gram path's first kernel makes the
// streams and writes them to post's output, which is pre's streams
static void dsv4_hc_front(ggml_backend_cuda_context & ctx, const ggml_tensor * post, const ggml_tensor * rms_flat,
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

    cudaStream_t stream = ctx.stream();
    const int64_t sx1 = x->nb[2] / sizeof(float);
    const int64_t sw1 = hc_fn->nb[1] / ggml_type_size(hc_fn->type);
    const bool w_prewait = dsv4_hc_prewait(hc_fn);

    if (!dsv4_hc_pre_gram_legacy()) {
        const int n_slices = n_embd / DSV4_HC_GRAM_SLICE;
        ggml_cuda_pool_alloc<float> partials(ctx.pool(), n_tokens*DSV4_HC_GRAM_ROWS*n_slices);

        const ggml_cuda_kernel_launch_params gram_params =
            ggml_cuda_kernel_launch_params(dim3(n_slices, n_tokens, 1), dim3(8*WARP_SIZE, 1, 1), 0, stream);
        dsv4_hc_post_args pa = {};
        if (post != nullptr) {
            const ggml_tensor * xo       = post->src[0];
            const ggml_tensor * residual = post->src[1];
            const ggml_tensor * post_w   = post->src[2];
            const ggml_tensor * comb_w   = post->src[3];
            pa.xo       = (const float *) xo->data;
            pa.residual = (const float *) residual->data;
            pa.post     = (const float *) post_w->data;
            pa.comb     = (const float *) comb_w->data;
            pa.sxo1 = xo->nb[1] / sizeof(float);
            pa.sr1  = residual->nb[1] / sizeof(float);
            pa.sr2  = residual->nb[2] / sizeof(float);
            pa.sp0  = post_w->nb[0] / sizeof(float);
            pa.sp1  = post_w->nb[1] / sizeof(float);
            pa.sc0  = comb_w->nb[0] / sizeof(float);
            pa.sc1  = comb_w->nb[1] / sizeof(float);
            pa.sc2  = comb_w->nb[2] / sizeof(float);
            pa.sx_h = x->nb[1] / sizeof(float);
        }
        // the normed mix's q8_1 copy, when the evaluation holds one for its quantized readers
        // (ggml_cuda_mmvq_shared_q8_1::produce): written here, not by a q8_1 launch at its first reader
        ggml_cuda_mmvq_shared_q8_1::entry * q8 = n_embd % MATRIX_ROW_PADDING == 0 ? ctx.mmvq_shared_q8_1.produce(mul) :
            nullptr;

        dsv4_hc_pre_args pr = {};
        pr.x            = (const float *) x->data;
        pr.scale        = (const float *) scale->data;
        pr.base         = (const float *) base->data;
        pr.norm_w       = (const float *) norm_w->data;
        pr.weights_out  = (float *) weights->data;
        pr.dst          = (float *) mul->data;
        pr.q8           = q8 != nullptr ? (block_q8_1 *) q8->q8_1 : nullptr;
        pr.n_embd       = n_embd;
        pr.k            = k;
        pr.sx1          = x->nb[1] / sizeof(float);
        pr.sx2          = x->nb[2] / sizeof(float);
        pr.ss0          = scale->nb[0] / sizeof(float);
        pr.sb0          = base->nb[0] / sizeof(float);
        pr.sw0          = weights->nb[0] / sizeof(float);
        pr.sw1          = weights->nb[1] / sizeof(float);
        pr.sd1          = mul->nb[1] / sizeof(float);
        pr.q8_s1        = n_embd / QK8_1;
        pr.eps_flat     = ggml_get_op_params_f32(rms_flat, 0);
        pr.eps_hc       = ggml_get_op_params_f32(weights, 0);
        pr.eps_norm     = ggml_get_op_params_f32(rms, 0);
        pr.n_iter       = ggml_get_op_params_i32(weights, 1);
        pr.base_prewait = dsv4_hc_prewait(base);
        pr.norm_prewait = dsv4_hc_prewait(norm_w);

        // the two kernels, writing p's outputs
        auto launch_two = [&](const dsv4_hc_pre_args & p) {
            auto launch = [&](auto kernel, const auto * w) {
                ggml_cuda_kernel_launch(kernel, gram_params,
                    (float *) x->data, w, partials.get(), n_embd, sx1, sw1, w_prewait, pa);
            };
            switch (hc_fn->type) {
                case GGML_TYPE_F32:
                    post ? launch(dsv4_hc_mix_gram<float, true>, (const float *) hc_fn->data)
                         : launch(dsv4_hc_mix_gram<float, false>, (const float *) hc_fn->data);
                    break;
                case GGML_TYPE_F16:
                    post ? launch(dsv4_hc_mix_gram<half, true>, (const half *) hc_fn->data)
                         : launch(dsv4_hc_mix_gram<half, false>, (const half *) hc_fn->data);
                    break;
                case GGML_TYPE_BF16:
                    post ? launch(dsv4_hc_mix_gram<nv_bfloat16, true>, (const nv_bfloat16 *) hc_fn->data)
                         : launch(dsv4_hc_mix_gram<nv_bfloat16, false>, (const nv_bfloat16 *) hc_fn->data);
                    break;
                default:
                    GGML_ABORT("unsupported hc_fn type %s", ggml_type_name(hc_fn->type));
            }
            const int n_blocks = (int) ((n_embd + DSV4_HC_PRE_GRAM_THR - 1) / DSV4_HC_PRE_GRAM_THR);
            const ggml_cuda_kernel_launch_params pre_params =
                ggml_cuda_kernel_launch_params(dim3(n_blocks, n_tokens, 1), dim3(DSV4_HC_PRE_GRAM_THR, 1, 1), 0, stream);
            ggml_cuda_kernel_launch(dsv4_hc_pre_gram_f32, pre_params, (const float *) partials.get(), n_slices, p);
        };

        // the tickets exist from the first graph evaluation on; a front takes at most their tokens
        static_assert(DSV4_HC_PRE_FUSED_MAX_TOKENS <= ggml_cuda_hc_front_tickets::n_tokens, "a token a ticket");
        unsigned int * tickets = ctx.hc_front_ticket();
        if (dsv4_hc_front_one_legacy() || tickets == nullptr) {
            launch_two(pr);
            return;
        }
        GGML_ASSERT(n_tokens <= ggml_cuda_hc_front_tickets::n_tokens);

        ggml_cuda_pool_alloc<float>      check_dst(ctx.pool());
        ggml_cuda_pool_alloc<float>      check_weights(ctx.pool());
        ggml_cuda_pool_alloc<block_q8_1> check_q8(ctx.pool());
        dsv4_hc_pre_args two = pr;
        if (dsv4_hc_front_check()) {
            two.dst         = check_dst.alloc(ggml_nbytes(mul) / sizeof(float));
            two.weights_out = check_weights.alloc(ggml_nbytes(weights) / sizeof(float));
            two.q8          = pr.q8 != nullptr ? check_q8.alloc(n_tokens*pr.q8_s1) : nullptr;
            launch_two(two);
        }

        auto launch_one = [&](auto kernel, const auto * w) {
            ggml_cuda_kernel_launch(kernel, gram_params,
                (float *) x->data, w, partials.get(), sx1, sw1, w_prewait, pa, pr, tickets);
        };
        switch (hc_fn->type) {
            case GGML_TYPE_F32:
                post ? launch_one(dsv4_hc_front_one<float, true>, (const float *) hc_fn->data)
                     : launch_one(dsv4_hc_front_one<float, false>, (const float *) hc_fn->data);
                break;
            case GGML_TYPE_F16:
                post ? launch_one(dsv4_hc_front_one<half, true>, (const half *) hc_fn->data)
                     : launch_one(dsv4_hc_front_one<half, false>, (const half *) hc_fn->data);
                break;
            case GGML_TYPE_BF16:
                post ? launch_one(dsv4_hc_front_one<nv_bfloat16, true>, (const nv_bfloat16 *) hc_fn->data)
                     : launch_one(dsv4_hc_front_one<nv_bfloat16, false>, (const nv_bfloat16 *) hc_fn->data);
                break;
            default:
                GGML_ABORT("unsupported hc_fn type %s", ggml_type_name(hc_fn->type));
        }

        if (dsv4_hc_front_check()) {
            dsv4_hc_front_check_bits<<<dim3(1, n_tokens, 1), 256, 0, stream>>>(pr, two);
        }
        return;
    }
    GGML_ASSERT(post == nullptr);

    const int n_slices = k / DSV4_HC_MIX_SLICE;
    ggml_cuda_pool_alloc<float> partials(ctx.pool(), n_tokens*(DSV4_HC_MIX + 1)*n_slices);

    const ggml_cuda_kernel_launch_params partial_params =
        ggml_cuda_kernel_launch_params(dim3(n_slices, n_tokens, 1), dim3(8*WARP_SIZE, 1, 1), 0, stream);
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
    // the comb in registers, bit for bit the 16 lanes'; GGML_CUDA_HC_COMB_LANES_LEGACY=1 makes it by the lanes
    static const bool comb_lanes = ggml_env_switch("GGML_CUDA_HC_COMB_LANES_LEGACY");
    ggml_cuda_kernel_launch(comb_lanes ? dsv4_hc_pre_norm_f32<false> : dsv4_hc_pre_norm_f32<true>, norm_params,
        (const float *) partials.get(), n_slices, (const float *) x->data, (const float *) scale->data,
        (const float *) base->data, (const float *) norm_w->data, (float *) weights->data, (float *) mul->data,
        n_embd, k, x->nb[1] / sizeof(float), x->nb[2] / sizeof(float), scale->nb[0] / sizeof(float),
        base->nb[0] / sizeof(float), weights->nb[0] / sizeof(float), weights->nb[1] / sizeof(float),
        mul->nb[1] / sizeof(float), ggml_get_op_params_f32(rms_flat, 0), ggml_get_op_params_f32(weights, 0),
        ggml_get_op_params_i32(weights, 1), ggml_get_op_params_f32(rms, 0), dsv4_hc_prewait(base));
}

void ggml_cuda_op_dsv4_hc_pre_fused(ggml_backend_cuda_context & ctx, const ggml_tensor * rms_flat,
        const ggml_tensor * mm, ggml_tensor * weights, const ggml_tensor * pre, const ggml_tensor * rms,
        ggml_tensor * mul) {
    dsv4_hc_front(ctx, nullptr, rms_flat, mm, weights, pre, rms, mul);
}

void ggml_cuda_op_dsv4_hc_post_pre_fused(ggml_backend_cuda_context & ctx, ggml_tensor * post,
        const ggml_tensor * rms_flat, const ggml_tensor * mm, ggml_tensor * weights, const ggml_tensor * pre,
        const ggml_tensor * rms, ggml_tensor * mul) {
    dsv4_hc_front(ctx, post, rms_flat, mm, weights, pre, rms, mul);
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
