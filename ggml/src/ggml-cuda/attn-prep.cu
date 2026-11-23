#include "attn-prep.cuh"

// Schedule-level fusions of Inkling 2 Mini's attention-prep chains: see attn-prep.cuh for the map from the decode
// profile's candidates to these kernels. Every kernel waits on its producers with ggml_cuda_pdl_sync before it reads
// anything they wrote, and no copy node sits between a fused kernel and its consumers (the fusion consumes exactly the
// nodes whose launches it replaces).

#define ATTN_PREP_MAX_ROWS 65536

// The Hadamard butterfly of fwht.cu's register kernel, in the same order and sign convention: the low element of a
// pair takes x + y, the high one x - y. A row of N is spread over NT threads, reg[j] holding columns j*NT + tid;
// partners past the row's thread count sit in other registers of the same thread.
template <int N, int NT>
__device__ __forceinline__ void attn_prep_fwht(float * reg) {
    constexpr int NE = N/NT;
    const float scale = 1.0f / sqrtf(float(N));

#pragma unroll
    for (int j = 0; j < NE; ++j) {
        reg[j] *= scale;
    }

#pragma unroll
    for (int h = 1; h < NT && h < N; h *= 2) {
#pragma unroll
        for (int j = 0; j < NE; ++j) {
            const float val  = reg[j];
            const float val2 = ggml_cuda_shfl_xor(val, h, NT);
            reg[j] = (threadIdx.x & h) == 0 ? val + val2 : val2 - val;
        }
    }

#pragma unroll
    for (int h = NT; h < N; h *= 2) {
        const int step = h/NT;
#pragma unroll
        for (int j = 0; j < NE; j += 2*step) {
#pragma unroll
            for (int k = 0; k < step; ++k) {
                const float x = reg[j + k];
                const float y = reg[j + k + step];
                reg[j + k]        = x + y;
                reg[j + k + step] = x - y;
            }
        }
    }
}

// the sum over a row's threads
template <int NT>
__device__ __forceinline__ float attn_prep_row_sum(float v) {
#pragma unroll
    for (int off = NT/2; off > 0; off >>= 1) {
        v += ggml_cuda_shfl_xor(v, off, NT);
    }
    return v;
}

// norm_clip: y * min(1, threshold/(|y| + eps)); rms_norm: y * rsqrtf(mean(y^2) + eps)
__device__ __forceinline__ float attn_prep_factor(const bool clip, const float sum_sq, const float n,
        const float thr, const float eps) {
    return clip ? fminf(1.0f, thr/(sqrtf(sum_sq) + eps)) : rsqrtf(sum_sq/n + eps);
}

// C1: each row of x, one head's channels in one token, is scaled by its head's row of scale, clipped to an L2 norm of
// at most threshold, multiplied by its token's tau and rotated with fwht.cu's butterfly, where the graph had
// NORM_CLIP -> MUL(tau)? -> reshape -> MUL_MAT(hadamard hint).
template <int N, bool do_fwht>
static __global__ void attn_prep_clip_fwht_f32(const float * x, const float * scale, const float * tau,
        const int64_t st2, float * dst, const int ne1, const int64_t rows, const float threshold, const float eps) {
    constexpr int NT = N >= 32 ? 32 : 16;
    constexpr int NE = N/NT;
    ggml_cuda_pdl_sync();

    const int64_t r = int64_t(blockIdx.x)*blockDim.y + threadIdx.y;
    if (r >= rows) {
        return;
    }

    // x is a contiguous [N, ne1, ne2]: its flat row order enumerates the head i1 = r % ne1 fastest
    const float * xr = x + r*N;
    const float * sr = scale ? scale + (r % ne1)*N : nullptr;
    const float   tr = tau ? tau[(r/ne1)*st2] : 1.0f;

    float reg[NE];
    float sum = 0.0f;
#pragma unroll
    for (int j = 0; j < NE; ++j) {
        const int c = j*NT + threadIdx.x;
        const float v = xr[c]*(sr ? sr[c] : 1.0f);
        reg[j] = v;
        sum += v*v;
    }
    sum = attn_prep_row_sum<NT>(sum);

    const float factor = attn_prep_factor(true, sum, N, threshold, eps)*tr;
#pragma unroll
    for (int j = 0; j < NE; ++j) {
        reg[j] *= factor;
    }

    if constexpr (do_fwht) {
        attn_prep_fwht<N, NT>(reg);
    }
    float * dr = dst + r*N;
#pragma unroll
    for (int j = 0; j < NE; ++j) {
        dr[j*NT + threadIdx.x] = reg[j];
    }
}

bool ggml_cuda_op_attn_prep_clip_fwht(ggml_backend_cuda_context & ctx, const ggml_tensor * norm,
        const ggml_tensor * mul, ggml_tensor * mm, bool do_fwht) {
    const ggml_tensor * a     = norm->src[0];
    const ggml_tensor * scale = norm->src[1];

    if (!a || !mm || a->type != GGML_TYPE_F32 || !ggml_is_contiguous(a) || a->nb[0] != sizeof(float) ||
            a->ne[3] != 1 || mm->type != GGML_TYPE_F32 || !ggml_is_contiguous(mm) ||
            (do_fwht && !ggml_cuda_op_mul_mat_use_fwht(mm))) {
        return false;
    }
    const int64_t n    = a->ne[0];
    const int64_t rows = a->ne[1]*a->ne[2];
    if ((n != 16 && n != 64 && n != 128) || rows < 1 || rows > ATTN_PREP_MAX_ROWS ||
            ggml_nelements(mm) != n*rows) {
        return false;
    }
    if (scale && (scale->type != GGML_TYPE_F32 || !ggml_is_contiguous(scale) ||
            scale->ne[0] != a->ne[0] || scale->ne[1] != a->ne[1] || scale->ne[2] != 1 || scale->ne[3] != 1)) {
        return false;
    }

    float threshold = 0.0f;
    float eps       = 0.0f;
    memcpy(&threshold, norm->op_params + 0, sizeof(float));
    memcpy(&eps,       norm->op_params + 1, sizeof(float));

    // the tau mul broadcasts one value over each token: [1, 1, ne2]
    const float * tau_d = nullptr;
    int64_t       st2   = 0;
    if (mul) {
        const ggml_tensor * t = mul->src[0] == norm ? mul->src[1] : mul->src[0];
        if (!t || t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t) || t->ne[0] != 1 || t->ne[1] != 1 ||
                t->ne[3] != 1 || t->ne[2] != a->ne[2]) {
            return false;
        }
        tau_d = (const float *) t->data;
        st2   = t->nb[2]/sizeof(float);
    }

    if (n >= 32 && ggml_cuda_info().devices[ctx.device].warp_size != 32) {
        return false; // the butterfly spreads the row over the lanes of one or two warps
    }

    const dim3 block(n >= 32 ? 32 : 16, n >= 32 ? 4 : 8); // a block stays at 128 threads
    const dim3 grid((unsigned) ((rows + block.y - 1)/block.y));
    const ggml_cuda_kernel_launch_params params(grid, block, 0, ctx.stream());

    auto launch = [&](auto width) {
        constexpr int W = decltype(width)::value;
        auto kernel = do_fwht ? attn_prep_clip_fwht_f32<W, true> : attn_prep_clip_fwht_f32<W, false>;
        ggml_cuda_kernel_launch(kernel, params, (const float *) a->data,
            scale ? (const float *) scale->data : nullptr, tau_d, st2, (float *) mm->data,
            (int) a->ne[1], rows, threshold, eps);
    };
    switch (n) {
        case 16: launch(std::integral_constant<int, 16>{}); break;
        case 64: launch(std::integral_constant<int, 64>{}); break;
        default: launch(std::integral_constant<int, 128>{}); break;
    }
    return true;
}

// D1: the relative-position rows' clip and their bias-table matmul in one launch. Each row of r, a head's n_rel
// positions in one token, is scaled by its head's row of the clip scale, clipped to an L2 norm of at most threshold,
// multiplied by its token's tau, then dotted with every column of the contiguous bias [n_rel, extent]:
// table[e, h, t] = sum_d r[d] * bias[d, e].
static __global__ void attn_prep_rel_table_f32(const float * x, const float * scale, const float * tau,
        const int64_t st2, const float * bias, float * table, const int ne1, const int nrel,
        const int64_t rows, const int extent, const float threshold, const float eps) {
    __shared__ float srow[4][128]; // one clipped row per warp; nrel is at most 128
    ggml_cuda_pdl_sync();

    const int64_t r = int64_t(blockIdx.x)*blockDim.y + threadIdx.y;
    if (r >= rows) {
        return;
    }

    const float * xr = x + r*nrel; // a contiguous [nrel, ne1]
    const float * sr = scale ? scale + (r % ne1)*nrel : nullptr;

    float * sv = srow[threadIdx.y & 3]; // blockDim.y is 4: one shared row per warp
    float sum = 0.0f;
    for (int d = threadIdx.x; d < nrel; d += WARP_SIZE) {
        const float v = xr[d]*(sr ? sr[d] : 1.0f);
        sv[d] = v;
        sum += v*v;
    }
    sum = attn_prep_row_sum<WARP_SIZE>(sum);

    const float factor = attn_prep_factor(true, sum, nrel, threshold, eps)*(tau ? tau[(r/ne1)*st2] : 1.0f);
    __syncwarp();
    for (int d = threadIdx.x; d < nrel; d += WARP_SIZE) {
        sv[d] *= factor;
    }
    __syncwarp();

    // table[e, h, t] with this row's head and token: flat ((t*ne1 + h)*extent + e)
    float * tr_out = table + ((r/ne1)*ne1 + (r % ne1))*extent;
    for (int e = threadIdx.x; e < extent; e += WARP_SIZE) {
        const float * bcol = bias + e*nrel;
        float acc = 0.0f;
#pragma unroll 4
        for (int d = 0; d < nrel; ++d) {
            acc += sv[d]*bcol[d];
        }
        tr_out[e] = acc;
    }
}

bool ggml_cuda_op_attn_prep_rel_table(ggml_backend_cuda_context & ctx, const ggml_tensor * norm,
        const ggml_tensor * mul, ggml_tensor * table) {
    const ggml_tensor * a     = norm->src[0];
    const ggml_tensor * scale = norm->src[1];

    if (!a || !table || a->type != GGML_TYPE_F32 || !ggml_is_contiguous(a) || a->nb[0] != sizeof(float) ||
            a->ne[3] != 1 || table->type != GGML_TYPE_F32 || !ggml_is_contiguous(table)) {
        return false;
    }
    // the matmul reads a real weight: [n_rel, extent], f32, contiguous, its columns the table's rows
    const ggml_tensor * bias = table->src[0];
    if (!bias || !bias->data || bias->type != GGML_TYPE_F32 || !ggml_is_contiguous(bias) ||
            bias->ne[0] != a->ne[0]) {
        return false;
    }
    const int64_t nrel = a->ne[0];
    const int64_t rows = a->ne[1]*a->ne[2];
    if (nrel < 1 || nrel > 128 || rows < 1 || rows > ATTN_PREP_MAX_ROWS ||
            ggml_nelements(table) != bias->ne[1]*rows || table->ne[1] != a->ne[1]) {
        return false;
    }
    if (scale && (scale->type != GGML_TYPE_F32 || !ggml_is_contiguous(scale) ||
            scale->ne[0] != a->ne[0] || scale->ne[1] != a->ne[1] || scale->ne[2] != 1 || scale->ne[3] != 1)) {
        return false;
    }

    float threshold = 0.0f;
    float eps       = 0.0f;
    memcpy(&threshold, norm->op_params + 0, sizeof(float));
    memcpy(&eps,       norm->op_params + 1, sizeof(float));

    const float * tau_d = nullptr;
    int64_t       st2   = 0;
    if (mul) {
        const ggml_tensor * t = mul->src[0] == norm ? mul->src[1] : mul->src[0];
        if (!t || t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t) || t->ne[0] != 1 || t->ne[1] != 1 ||
                t->ne[3] != 1 || t->ne[2] != a->ne[2]) {
            return false;
        }
        tau_d = (const float *) t->data;
        st2   = t->nb[2]/sizeof(float);
    }

    const dim3 block(WARP_SIZE, 4);
    const dim3 grid((unsigned) ((rows + block.y - 1)/block.y));
    const ggml_cuda_kernel_launch_params params(grid, block, 0, ctx.stream());
    ggml_cuda_kernel_launch(attn_prep_rel_table_f32, params, (const float *) a->data,
        scale ? (const float *) scale->data : nullptr, tau_d, st2, (const float *) bias->data,
        (float *) table->data, (int) a->ne[1], (int) nrel, rows, (int) bias->ne[1], threshold, eps);
    return true;
}

// C2/C3: a K or V epilogue. One warp per rotated row (this stream's head fastest, then tokens and sequences): the
// depthwise convolution from the window's sources (the claimed CONCAT's two sides, or a materialized x), then the
// weightless norm clip or rms norm over the head's channels, then fwht.cu's butterfly (1xN or 2x64 inside 128), and
// optional fused SET_ROWS into the F16/Q8_0 KV cache (plus optional F16 hot_kv). The warps whose row is their
// sequence's first token also write the joined window the state store reads, exactly as the skipped CONCAT would have.
enum attn_prep_kv_dst_kind : int {
    ATTN_PREP_KV_DST_NONE = 0,
    ATTN_PREP_KV_DST_F16  = 1,
    ATTN_PREP_KV_DST_Q8_0 = 2,
};

struct attn_prep_kv_args {
    const float * past = nullptr; // [k-1, ch, ns] f32, strides in floats
    int64_t sp0 = 0, sp1 = 0, sp2 = 0;
    const float * rows = nullptr; // [nt, ch, ns]
    int64_t sr0 = 0, sr1 = 0, sr2 = 0;
    const float * xpx = nullptr;  // the whole joined window where it was not claimed by the evaluator
    int64_t xp0 = 0, xp1 = 0, xp2 = 0;
    float * win = nullptr;        // written beside the output where the CONCAT was skipped
    int64_t w1 = 0, w2 = 0;
    const float * wtap = nullptr; // [k, ch]: tap j of channel c at w[c*k + j]
    const float * dyn  = nullptr; // [k*nk, nt*ns] or null: taps h*k .. h*k+k-1 of merged token t'
    int64_t sd1 = 0;
    float * dst = nullptr;
    char *        kv_dst      = nullptr;
    const char *  kv_idxs     = nullptr;
    size_t        kv_nb1      = 0;
    size_t        kv_idx_nb0  = 0;
    char *        hot_dst     = nullptr;
    const char *  hot_idxs    = nullptr;
    size_t        hot_nb1     = 0;
    size_t        hot_idx_nb0 = 0;
    int   nk = 0, nt = 0, ns = 0;
    int   klen = 0;
    bool  split       = false; // read past/rows rather than xpx
    bool  write_win   = false; // fill win as the skipped CONCAT would have
    bool  clip        = false; // a norm clip with threshold rather than an rms norm with eps
    bool  kv_idx_i32  = false;
    bool  hot_idx_i32 = false;
    float conv_scale = 1.0f, threshold = 1.0f, eps = 0.0f;
};

static __device__ __forceinline__ int64_t attn_prep_load_idx(const char * base, const size_t nb0,
        const bool is_i32, const int tp) {
    const char * p = base + size_t(tp)*nb0;
    return is_i32 ? (int64_t) *(const int32_t *) p : *(const int64_t *) p;
}

template <int N, int ROT_W, attn_prep_kv_dst_kind KV_KIND>
static __global__ void attn_prep_kv_tail_f32(const attn_prep_kv_args p) {
    constexpr int NT = N >= 32 ? 32 : 16;
    constexpr int NE = N/NT;
    ggml_cuda_pdl_sync();

    const int64_t rows = int64_t(p.nk)*p.nt*p.ns;
    const int64_t rr   = int64_t(blockIdx.x)*blockDim.y + threadIdx.y;
    if (rr >= rows) {
        return;
    }

    const int h  = rr % p.nk; // r enumerates this stream's heads fastest
    const int tp = rr / p.nk; // then tokens and sequences: t' = t + nt*s
    const int t  = tp % p.nt;
    const int s  = tp / p.nt;
    const int cb = h*N;

    const auto joined = [&](const int64_t q, const int c) -> float {
        if (p.split) {
            return q < p.klen - 1 ? p.past[q*p.sp0 + c*p.sp1 + s*p.sp2]
                                  : p.rows[(q - p.klen + 1)*p.sr0 + c*p.sr1 + s*p.sr2];
        }
        return p.xpx[q*p.xp0 + c*p.xp1 + s*p.xp2];
    };

    float reg[NE];
    float sum = 0.0f;
#pragma unroll
    for (int j = 0; j < NE; ++j) {
        const int64_t c = cb + j*NT + threadIdx.x;
        float acc = 0.0f;
        for (int jt = 0; jt < p.klen; ++jt) {
            acc += joined(t + jt, (int) c)*p.wtap[c*p.klen + jt];
        }
        acc += joined(p.klen - 1 + t, (int) c); // the current value's own weight-one tap
        if (p.dyn) {
            const float * dt = p.dyn + tp*p.sd1 + h*p.klen;
            for (int lag = 0; lag < p.klen; ++lag) {
                acc += joined(p.klen - 1 - lag + t, (int) c)*dt[lag];
            }
        }
        const float v = p.conv_scale*acc;
        reg[j] = v;
        sum += v*v;
    }
    sum = attn_prep_row_sum<NT>(sum);

    const float factor = attn_prep_factor(p.clip, sum, N, p.threshold, p.eps);
#pragma unroll
    for (int j = 0; j < NE; ++j) {
        reg[j] *= factor;
    }

    if constexpr (ROT_W == N && ROT_W > 0) {
        attn_prep_fwht<N, NT>(reg);
    } else if constexpr (ROT_W == 64 && N == 128) {
        attn_prep_fwht<64, NT>(reg + 0);
        attn_prep_fwht<64, NT>(reg + 2);
    } else {
        static_assert(ROT_W == 0, "unsupported Hadamard width");
    }

    if (p.dst) {
        float * dr = p.dst + rr*N;
#pragma unroll
        for (int j = 0; j < NE; ++j) {
            dr[j*NT + threadIdx.x] = reg[j];
        }
    }

    if constexpr (KV_KIND == ATTN_PREP_KV_DST_F16) {
        const int64_t dst_row = attn_prep_load_idx(p.kv_idxs, p.kv_idx_nb0, p.kv_idx_i32, tp);
        if (dst_row >= 0) {
            half * hr = (half *) (p.kv_dst + size_t(dst_row)*p.kv_nb1) + cb;
#pragma unroll
            for (int j = 0; j < NE; ++j) {
                hr[j*NT + threadIdx.x] = __float2half(reg[j]);
            }
        }
        if (p.hot_dst) {
            const int64_t hot_row = attn_prep_load_idx(p.hot_idxs, p.hot_idx_nb0, p.hot_idx_i32, tp);
            if (hot_row >= 0) {
                half * hr = (half *) (p.hot_dst + size_t(hot_row)*p.hot_nb1) + cb;
#pragma unroll
                for (int j = 0; j < NE; ++j) {
                    hr[j*NT + threadIdx.x] = __float2half(reg[j]);
                }
            }
        }
    } else if constexpr (KV_KIND == ATTN_PREP_KV_DST_Q8_0) {
        static_assert(N >= QK8_0 && NT == QK8_0 && N % QK8_0 == 0, "Q8_0 requires 32-lane warp blocks");
        const int64_t dst_row = attn_prep_load_idx(p.kv_idxs, p.kv_idx_nb0, p.kv_idx_i32, tp);
        if (dst_row >= 0) {
            block_q8_0 * br = (block_q8_0 *) (p.kv_dst + size_t(dst_row)*p.kv_nb1) + h*NE;
#pragma unroll
            for (int j = 0; j < NE; ++j) {
                const float amax = warp_reduce_max<QK8_0>(fabsf(reg[j]));
                const float d    = amax / ((1 << 7) - 1);
                const float id   = d ? 1.0f/d : 0.0f;
                br[j].qs[threadIdx.x] = roundf(reg[j]*id);
                if (threadIdx.x == 0) {
                    br[j].d = __float2half(d);
                }
            }
        }
        if (p.hot_dst) {
            const int64_t hot_row = attn_prep_load_idx(p.hot_idxs, p.hot_idx_nb0, p.hot_idx_i32, tp);
            if (hot_row >= 0) {
                half * hr = (half *) (p.hot_dst + size_t(hot_row)*p.hot_nb1) + cb;
#pragma unroll
                for (int j = 0; j < NE; ++j) {
                    hr[j*NT + threadIdx.x] = __float2half(reg[j]);
                }
            }
        }
    }

    // a sequence's window is written by its first token's warps, one per head over that head's channels only: where
    // win does not alias the sources (checked at launch), no thread reads what another writes here
    if (p.write_win && t == 0) {
        const int cols = p.nt + p.klen - 1;
#pragma unroll
        for (int j = 0; j < NE; ++j) {
            const int64_t c = cb + j*NT + threadIdx.x;
            for (int q = 0; q < cols; ++q) {
                p.win[q + c*p.w1 + s*p.w2] = joined(q, (int) c);
            }
        }
    }
}

bool ggml_cuda_op_attn_prep_kv_tail(ggml_backend_cuda_context & ctx, const ggml_tensor * conv,
        const ggml_tensor * norm, ggml_tensor * out, const ggml_cuda_short_conv_window * win,
        bool do_fwht, const ggml_tensor * set_rows, const ggml_tensor * hot_set_rows, bool write_f32_dst) {
    if (!conv || !norm || !out || (!write_f32_dst && !set_rows) || (hot_set_rows && !set_rows)) {
        return false;
    }
    const ggml_tensor * x = conv->src[0];
    const ggml_tensor * w = conv->src[1];
    const ggml_tensor * d = conv->src[2];

    if (!x || !w || x->type != GGML_TYPE_F32 || w->type != GGML_TYPE_F32 || !ggml_is_contiguous(w) ||
            conv->ne[3] != 1 || conv->data == nullptr || w->data == nullptr ||
            out->type != GGML_TYPE_F32 || !ggml_is_contiguous(out) || (write_f32_dst && out->data == nullptr) ||
            (do_fwht && !ggml_cuda_op_mul_mat_use_fwht(out))) {
        return false;
    }
    const int     k         = (int) w->ne[0];
    const int64_t ch        = conv->ne[0];
    const int64_t nt        = conv->ne[1];
    const int64_t ns        = conv->ne[2];
    const int64_t N         = norm->ne[0];
    const int64_t rot_width = do_fwht ? out->src[0]->ne[0] : 0;
    if ((k != 3 && k != 4) || (N != 16 && N != 64 && N != 128) || ch % N != 0 || nt < 1 || ns < 1 ||
            norm->ne[1] != ch/N || ggml_nelements(norm) != ch*nt*ns || ggml_nelements(out) != ch*nt*ns) {
        return false;
    }
    if (do_fwht) {
        const ggml_tensor * rot = out->src[0];
        if (!rot || rot->ne[0] != rot->ne[1] || rot->ne[2] != 1 || rot->ne[3] != 1 || out->ne[0] != rot_width ||
                (rot_width != N && !(N == 128 && rot_width == 64))) {
            return false;
        }
    }
    if (N >= 32 && ggml_cuda_info().devices[ctx.device].warp_size != 32) {
        return false;
    }
    const int64_t nk   = ch/N;
    const int64_t rows = nk*nt*ns;
    if (rows > ATTN_PREP_MAX_ROWS*(int64_t) 8) {
        return false;
    }
    if (d && (d->type != GGML_TYPE_F32 || d->data == nullptr || d->nb[0] != sizeof(float) ||
            d->nb[1] % sizeof(float) != 0 || d->ne[0] != k*nk || d->ne[1] != nt*ns ||
            d->ne[2] != 1 || d->ne[3] != 1)) {
        return false;
    }

    auto check_set_rows = [&](const ggml_tensor * sr, bool hot) -> bool {
        if (!sr || sr->op != GGML_OP_SET_ROWS || sr->data == nullptr ||
                sr->ne[0] != ch || sr->ne[2] != 1 || sr->ne[3] != 1 || !ggml_is_contiguous_rows(sr)) {
            return false;
        }
        if (hot) {
            if (sr->type != GGML_TYPE_F16) {
                return false;
            }
        } else if (sr->type == GGML_TYPE_Q8_0) {
            if (N < QK8_0 || N % QK8_0 != 0) {
                return false;
            }
        } else if (sr->type != GGML_TYPE_F16) {
            return false;
        }
        if (reinterpret_cast<uintptr_t>(sr->data) % alignof(half) != 0 || sr->nb[1] % alignof(half) != 0) {
            return false;
        }
        const ggml_tensor * src0 = sr->src[0];
        const ggml_tensor * idxs = sr->src[1];
        if (!src0 || src0->type != GGML_TYPE_F32 || !ggml_is_contiguous(src0) ||
                src0->ne[0] != ch || src0->ne[1] != nt*ns || src0->ne[2] != 1 || src0->ne[3] != 1) {
            return false;
        }
        const ggml_tensor * base = src0;
        while (base && base != out && (base->op == GGML_OP_VIEW || base->op == GGML_OP_RESHAPE) && base->view_offs == 0) {
            base = base->src[0];
        }
        if (base != out) {
            return false;
        }
        if (!idxs || idxs->data == nullptr || (idxs->type != GGML_TYPE_I64 && idxs->type != GGML_TYPE_I32) ||
                idxs->ne[0] != nt*ns || idxs->ne[1] != 1 || idxs->ne[2] != 1 || idxs->ne[3] != 1) {
            return false;
        }
        const size_t idx_ts = ggml_type_size(idxs->type);
        if (reinterpret_cast<uintptr_t>(idxs->data) % idx_ts != 0 || idxs->nb[0] % idx_ts != 0) {
            return false;
        }
        return true;
    };

    if (set_rows && !check_set_rows(set_rows, /*hot=*/false)) {
        return false;
    }
    if (hot_set_rows && !check_set_rows(hot_set_rows, /*hot=*/true)) {
        return false;
    }

    attn_prep_kv_args p;
    p.wtap = (const float *) w->data;
    p.dyn  = d ? (const float *) d->data : nullptr;
    p.sd1  = d ? d->nb[1]/sizeof(float) : 0;
    p.dst  = write_f32_dst ? (float *) out->data : nullptr;
    p.nk   = (int) nk;
    p.nt   = (int) nt;
    p.ns   = (int) ns;
    p.klen = k;
    p.conv_scale = ggml_get_op_params_f32(conv, 0);
    p.clip = norm->op == GGML_OP_NORM_CLIP;
    if (p.clip) {
        memcpy(&p.threshold, norm->op_params + 0, sizeof(float));
        memcpy(&p.eps,       norm->op_params + 1, sizeof(float));
    } else {
        memcpy(&p.eps, norm->op_params + 0, sizeof(float));
    }

    attn_prep_kv_dst_kind kv_kind = ATTN_PREP_KV_DST_NONE;
    if (set_rows) {
        const ggml_tensor * idxs = set_rows->src[1];
        kv_kind       = set_rows->type == GGML_TYPE_Q8_0 ? ATTN_PREP_KV_DST_Q8_0 : ATTN_PREP_KV_DST_F16;
        p.kv_dst      = (char *) set_rows->data;
        p.kv_idxs     = (const char *) idxs->data;
        p.kv_nb1      = set_rows->nb[1];
        p.kv_idx_nb0  = idxs->nb[0];
        p.kv_idx_i32  = idxs->type == GGML_TYPE_I32;
    }
    if (hot_set_rows) {
        const ggml_tensor * hidxs = hot_set_rows->src[1];
        p.hot_dst     = (char *) hot_set_rows->data;
        p.hot_idxs    = (const char *) hidxs->data;
        p.hot_nb1     = hot_set_rows->nb[1];
        p.hot_idx_nb0 = hidxs->nb[0];
        p.hot_idx_i32 = hidxs->type == GGML_TYPE_I32;
    }

    if (win) {
        // the CONCAT was skipped by the evaluator: this kernel reads its two sides and writes the joined window it
        // would have made, beside the convolution's output as before
        if (!win->past || !win->rows || win->window->type != GGML_TYPE_F32 || !ggml_is_contiguous(win->window)) {
            return false;
        }
        p.past = (const float *) win->past->data;
        p.rows = (const float *) win->rows->data;
        p.sp0  = win->past->nb[0]/sizeof(float);
        p.sp1  = win->past->nb[1]/sizeof(float);
        p.sp2  = win->past->nb[2]/sizeof(float);
        p.sr0  = win->rows->nb[0]/sizeof(float);
        p.sr1  = win->rows->nb[1]/sizeof(float);
        p.sr2  = win->rows->nb[2]/sizeof(float);
        p.win  = (float *) win->window->data;
        p.w1   = win->window->ne[0];
        p.w2   = win->window->ne[0]*win->window->ne[1];
        p.split     = true;
        p.write_win = true;
    } else {
        if (!ggml_is_contiguous(x) || x->data == nullptr) {
            return false;
        }
        p.xpx = (const float *) x->data;
        p.xp0 = x->nb[0]/sizeof(float);
        p.xp1 = x->nb[1]/sizeof(float);
        p.xp2 = x->nb[2]/sizeof(float);
    }

    const dim3 block(N >= 32 ? 32 : 16, N >= 32 ? 4 : 8);
    const dim3 grid((unsigned) ((rows + block.y - 1)/block.y));
    const ggml_cuda_kernel_launch_params params(grid, block, 0, ctx.stream());
    auto launch = [&](auto width, auto rot_w) {
        constexpr int W = decltype(width)::value;
        constexpr int R = decltype(rot_w)::value;
        switch (kv_kind) {
            case ATTN_PREP_KV_DST_NONE:
                ggml_cuda_kernel_launch(attn_prep_kv_tail_f32<W, R, ATTN_PREP_KV_DST_NONE>, params, p);
                break;
            case ATTN_PREP_KV_DST_F16:
                ggml_cuda_kernel_launch(attn_prep_kv_tail_f32<W, R, ATTN_PREP_KV_DST_F16>, params, p);
                break;
            case ATTN_PREP_KV_DST_Q8_0:
                if constexpr (W >= QK8_0) {
                    ggml_cuda_kernel_launch(attn_prep_kv_tail_f32<W, R, ATTN_PREP_KV_DST_Q8_0>, params, p);
                }
                break;
        }
    };
    switch (N) {
        case 16:
            if (rot_width == 16) {
                launch(std::integral_constant<int, 16>{}, std::integral_constant<int, 16>{});
            } else {
                launch(std::integral_constant<int, 16>{}, std::integral_constant<int, 0>{});
            }
            break;
        case 64:
            if (rot_width == 64) {
                launch(std::integral_constant<int, 64>{}, std::integral_constant<int, 64>{});
            } else {
                launch(std::integral_constant<int, 64>{}, std::integral_constant<int, 0>{});
            }
            break;
        default:
            if (rot_width == 128) {
                launch(std::integral_constant<int, 128>{}, std::integral_constant<int, 128>{});
            } else if (rot_width == 64) {
                launch(std::integral_constant<int, 128>{}, std::integral_constant<int, 64>{});
            } else {
                launch(std::integral_constant<int, 128>{}, std::integral_constant<int, 0>{});
            }
            break;
    }
    return true;
}

// E1: the conv-state store. The layer's window parts are written straight into their sequences' cache rows, in the
// state row's own order: element rem of a row is column cp = rem % n_cols of global channel C = rem / n_cols, whose
// part and local channel the cumulative offsets name. (The parts list itself lives in attn-prep.cuh.)
struct attn_prep_store_args {
    const float * p[8];
    int64_t s0[8], s1[8], s2[8]; // each part's strides in floats
    int     cbeg[9];             // the first global channel of each part, one past the last
    int     m = 0;
    int     ncols = 0;
    int     sum_ch = 0;
    int64_t c0 = 0;              // the joined window's column where the copied columns start
    int64_t total = 0;           // n_cols*channels*n_seqs elements
};

template <typename T>
static __global__ void attn_prep_state_store_kernel(const attn_prep_store_args a, const int64_t * ids,
        T * dst, const int64_t row_stride_e) {
    ggml_cuda_pdl_sync();

    const int64_t idx = int64_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (idx >= a.total) {
        return;
    }
    const int64_t per_seq = int64_t(a.ncols)*a.sum_ch;
    const int     s       = idx/per_seq;
    const int64_t rem     = idx % per_seq;
    const int     cp      = rem % a.ncols;
    const int     C       = rem/a.ncols;

    int g = 0;
    while (g < a.m - 1 && C >= a.cbeg[g + 1]) {
        ++g;
    }
    const int cloc = C - a.cbeg[g];
    const float v  = a.p[g][(a.c0 + cp)*a.s0[g] + cloc*a.s1[g] + s*a.s2[g]];

    const int64_t row = ids[s];
    if constexpr (std::is_same_v<T, half>) {
        dst[row*row_stride_e + rem] = __float2half(v);
    } else {
        dst[row*row_stride_e + rem] = v;
    }
}

#if defined(GGML_CUDA_USE_L2_WINDOW)
bool ggml_cuda_op_attn_prep_state_store(ggml_backend_cuda_context & ctx, const attn_prep_store_parts & parts,
        const ggml_tensor * window, const ggml_tensor * snapshot, const ggml_tensor * set_rows,
        cudaAccessPolicyWindow l2_window) {
#else
bool ggml_cuda_op_attn_prep_state_store(ggml_backend_cuda_context & ctx, const attn_prep_store_parts & parts,
        const ggml_tensor * window, const ggml_tensor * snapshot, const ggml_tensor * set_rows) {
#endif
    if (!set_rows || set_rows->op != GGML_OP_SET_ROWS || parts.m < 2 || parts.m > 8 ||
            !window || window->type != GGML_TYPE_F32 || !ggml_is_contiguous(window) ||
            window->ne[3] != 1 || !window->data) {
        return false;
    }
    const ggml_tensor * states = set_rows->src[0];
    const ggml_tensor * ids    = set_rows->src[1];
    const int64_t sum_ch = window->ne[1];
    const int64_t cols   = window->ne[0];
    const int64_t ns     = window->ne[2];
    if (states->type != GGML_TYPE_F32 && states->type != GGML_TYPE_F16) {
        return false; // only whole f32 and f16 rows are written here; quantized states keep the chain
    }
    if (!ids || ids->type != GGML_TYPE_I64 || ids->ne[0] != ns) {
        return false;
    }

    attn_prep_store_args a;
    for (int g = 0; g < parts.m; ++g) {
        const ggml_tensor * part = parts.part[g];
        if (!part || part->type != GGML_TYPE_F32 || part->data == nullptr || part->ne[3] != 1 ||
                part->ne[0] != cols || part->ne[1] > sum_ch || part->ne[2] != ns) {
            return false;
        }
    }
    int64_t cum = 0;
    for (int g = 0; g < parts.m; ++g) {
        const ggml_tensor * part = parts.part[g];
        a.p[g]      = (const float *) part->data;
        a.s0[g]     = part->nb[0]/sizeof(float);
        a.s1[g]     = part->nb[1]/sizeof(float);
        a.s2[g]     = part->nb[2]/sizeof(float);
        a.cbeg[g]   = (int) cum;
        cum        += part->ne[1];
    }
    if (cum != sum_ch) {
        return false;
    }
    a.cbeg[parts.m] = (int) cum;
    a.m      = parts.m;
    a.sum_ch = (int) sum_ch;

    // the snapshot view names the joined window's columns this store copies: one row of channels in
    const int64_t ncols = states->ne[0]/sum_ch;
    if (ncols < 1 || ncols*sum_ch != states->ne[0] || ncols > cols) {
        return false;
    }
    if (!snapshot || snapshot->view_src != window || snapshot->type != GGML_TYPE_F32 ||
            snapshot->ne[0] != ncols || snapshot->ne[1] != sum_ch || snapshot->ne[2] != ns || snapshot->ne[3] != 1 ||
            snapshot->nb[0] != window->nb[0] || snapshot->nb[1] != window->nb[1] || snapshot->nb[2] != window->nb[2] ||
            snapshot->view_offs % window->nb[0] != 0) {
        return false;
    }
    a.ncols = (int) ncols;
    a.c0    = snapshot->view_offs/window->nb[0];
    if (a.c0 + ncols > cols) {
        return false;
    }
    a.total = ncols*sum_ch*ns;
    if (a.total/256 > INT32_MAX) {
        return false;
    }

    const dim3 block(256);
    const dim3 grid((unsigned) ((a.total + block.x - 1)/block.x));
    ggml_cuda_kernel_launch_params params(grid, block, 0, ctx.stream());
#if defined(GGML_CUDA_USE_L2_WINDOW)
    params.l2_window = l2_window;
#endif
    const int64_t * ids_d = (const int64_t *) ids->data;
    if (set_rows->type == GGML_TYPE_F16) {
        ggml_cuda_kernel_launch(attn_prep_state_store_kernel<half>, params, a, ids_d,
            (half *) set_rows->data, set_rows->nb[1]/sizeof(half));
    } else {
        ggml_cuda_kernel_launch(attn_prep_state_store_kernel<float>, params, a, ids_d,
            (float *) set_rows->data, set_rows->nb[1]/sizeof(float));
    }
    return true;
}
