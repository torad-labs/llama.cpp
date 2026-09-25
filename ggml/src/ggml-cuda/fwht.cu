#include "common.cuh"
#include "fwht.cuh"

#include <cstdlib>

template <typename T>
__device__ __forceinline__ float fwht_load(const T value) {
    return value;
}

template <>
__device__ __forceinline__ float fwht_load<half>(const half value) {
    return __half2float(value);
}

template <int N, typename T, bool has_signs>
__launch_bounds__(4*ggml_cuda_get_physical_warp_size(), 1)
__global__ void fwht_cuda(const T * src, float * dst, const int64_t n_rows, const float scale,
                          const float * signs, const int n_blk, const bool pdl_trigger) {
    if (pdl_trigger) {
        ggml_cuda_pdl_lc();
    }
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    const int64_t r = (int64_t) blockIdx.x * blockDim.y + threadIdx.y;

    if (r >= n_rows) {
        return;
    }

    src += r * N;
    dst += r * N;

    static constexpr int el_w = N / warp_size;
    float     reg[el_w];
    const int lane = threadIdx.x;

    ggml_cuda_pdl_sync();
    const float * signs_row = has_signs ? signs + (r % n_blk) * N : nullptr;
#pragma unroll
    for (int i = 0; i < el_w; ++i) {
        reg[i] = fwht_load(src[i * warp_size + lane]) * scale;
        if (has_signs) {
            reg[i] *= signs_row[i * warp_size + lane];
        }
    }

#pragma unroll
    for (int h = 1; h < warp_size; h *= 2) {
#pragma unroll
        for (int j = 0; j < el_w; j++) {
            const float val  = reg[j];
            const float val2 = __shfl_xor_sync(0xFFFFFFFF, val, h, warp_size);

            reg[j] = (lane & h) == 0 ? val + val2 : val2 - val;
        }
    }

#pragma unroll
    for (int h = warp_size; h < N; h *= 2) {
        const int step = h / warp_size;
#pragma unroll
        for (int j = 0; j < el_w; j += 2 * step) {
#pragma unroll
            for (int k = 0; k < step; k++) {
                const float x = reg[j + k];
                const float y = reg[j + k + step];

                reg[j + k]        = x + y;
                reg[j + k + step] = x - y;
            }
        }
    }

#pragma unroll
    for (int i = 0; i < el_w; ++i) {
        dst[i * warp_size + lane] = reg[i];
    }
}

// Large-N path. The register kernel above keeps N/warp_size floats per thread, so it stops being
// viable well before the arithmetic does: N=4096 would need 128 registers per thread and spill,
// which is why the switch below used to end at 2048 and simply decline anything larger (the whole
// op then fell back to CPU). Stage the row in shared memory instead and run all log2(N) butterfly
// stages there. One row per block; every thread handles several butterflies per stage. Slower per
// row than the register path, so it is used only where that path cannot go.
#define FWHT_SMEM_THREADS 256

template <int N, typename T, bool has_signs>
__launch_bounds__(FWHT_SMEM_THREADS, 1)
__global__ void fwht_cuda_smem(const T * src, float * dst, const int64_t n_rows, const float scale,
                               const float * signs, const int n_blk, const bool pdl_trigger) {
    if (pdl_trigger) {
        ggml_cuda_pdl_lc();
    }
    __shared__ float s[N];

    const int64_t r = blockIdx.x;
    if (r >= n_rows) {
        return;
    }

    src += r * N;
    dst += r * N;

    const float * signs_row = has_signs ? signs + (r % n_blk) * N : nullptr;

    ggml_cuda_pdl_sync();
    for (int i = threadIdx.x; i < N; i += FWHT_SMEM_THREADS) {
        float v = fwht_load(src[i]) * scale;
        if (has_signs) {
            v *= signs_row[i];
        }
        s[i] = v;
    }
    __syncthreads();

    // Same butterfly and the same sign convention as the register path: the low element of a pair
    // takes x + y, the high one x - y.
#pragma unroll 1
    for (int h = 1; h < N; h *= 2) {
        for (int idx = threadIdx.x; idx < N / 2; idx += FWHT_SMEM_THREADS) {
            const int j = ((idx / h) * 2 * h) + (idx % h);
            const float x = s[j];
            const float y = s[j + h];
            s[j]     = x + y;
            s[j + h] = x - y;
        }
        __syncthreads();
    }

    for (int i = threadIdx.x; i < N; i += FWHT_SMEM_THREADS) {
        dst[i] = s[i];
    }
}


// Wide rows at small row counts (decode): one row per block instead of per warp.
// The warp kernel serialises every stage on one warp, and the shared-memory kernel above synchronises on each stage.
// Both leave most of the GPU idle at these shapes.
#define FWHT_BLOCK_THREADS 256

// The transform of an N-element row held as reg[i] = element i * NT + tid (NT threads), s: N floats of shared memory.
// The stages run in order of their distance h, so every NT gives the same result.
template <int N, int NT>
static __device__ __forceinline__ void fwht_block_transform(float (&reg)[N / NT], float * s) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int NE        = N / NT;
    static_assert(NE >= 1 && N % NT == 0 && NT % warp_size == 0, "bad FWHT block shape");

    const int tid  = threadIdx.x;
    const int lane = tid % warp_size;

    // stages within a warp: partner differs in the lane bits
#pragma unroll
    for (int h = 1; h < warp_size; h *= 2) {
#pragma unroll
        for (int j = 0; j < NE; j++) {
            const float val  = reg[j];
            const float val2 = __shfl_xor_sync(0xFFFFFFFF, val, h, warp_size);
            reg[j] = (lane & h) == 0 ? val + val2 : val2 - val;
        }
    }

    // stages across warps: partner differs in the thread-index bits above the lane
#pragma unroll
    for (int h = warp_size; h < NT; h *= 2) {
#pragma unroll
        for (int j = 0; j < NE; j++) {
            s[j * NT + tid] = reg[j];
        }
        __syncthreads();
#pragma unroll
        for (int j = 0; j < NE; j++) {
            const float val  = reg[j];
            const float val2 = s[j * NT + (tid ^ h)];
            reg[j] = (tid & h) == 0 ? val + val2 : val2 - val;
        }
        __syncthreads();
    }

    // stages above the block width: partner is another register of the same thread
#pragma unroll
    for (int h = NT; h < N; h *= 2) {
        const int step = h / NT;
#pragma unroll
        for (int j = 0; j < NE; j += 2 * step) {
#pragma unroll
            for (int k = 0; k < step; k++) {
                const float x = reg[j + k];
                const float y = reg[j + k + step];
                reg[j + k]        = x + y;
                reg[j + k + step] = x - y;
            }
        }
    }
}

// Stores a row fwht_block_transform left in reg at dst.
template <int N, int NT>
static __device__ __forceinline__ void fwht_block_store(const float (&reg)[N / NT], float * dst) {
    constexpr int NE = N / NT;
    const int tid = threadIdx.x;

#pragma unroll
    for (int i = 0; i < NE; ++i) {
        dst[i * NT + tid] = reg[i];
    }
}

// A source read through a 4-D view, as a CONT of it would copy it: element g of the contiguous rows is src[i0*s0 + i1*s1
// + i2*s2 + i3*s3], (i0, i1, i2, i3) being g unravelled over the view's ne0, ne1 and ne2 (fastdiv values).
struct fwht_src_view {
    uint3   ne0, ne1, ne2;
    int64_t s0, s1, s2, s3;
};

template <int N, int NT, typename T, bool has_signs, bool has_view = false>
__launch_bounds__(NT, 1)
__global__ void fwht_cuda_block(const T * src, float * dst, const int64_t n_rows, const float scale,
                                const float * signs, const int n_blk, const bool pdl_trigger, const fwht_src_view view) {
    if (pdl_trigger) {
        ggml_cuda_pdl_lc();
    }
    constexpr int NE = N / NT;

    __shared__ float s[N];

    const int64_t r = blockIdx.x;
    if (r >= n_rows) {
        return;
    }

    if constexpr (!has_view) {
        src += r * N;
    }
    dst += r * N;

    const int tid = threadIdx.x;

    ggml_cuda_pdl_sync();
    const float * signs_row = has_signs ? signs + (r % n_blk) * N : nullptr;

    float reg[NE];
#pragma unroll
    for (int i = 0; i < NE; ++i) {
        if constexpr (has_view) {
            const uint32_t g  = (uint32_t) (r * N + i * NT + tid);
            const uint32_t q0 = fastdiv(g,  view.ne0);
            const uint32_t q1 = fastdiv(q0, view.ne1);
            const uint32_t i3 = fastdiv(q1, view.ne2);
            reg[i] = fwht_load(src[(g - q0 * view.ne0.z) * view.s0 + (q0 - q1 * view.ne1.z) * view.s1 +
                                   (q1 - i3 * view.ne2.z) * view.s2 + i3 * view.s3]) * scale;
        } else {
            reg[i] = fwht_load(src[i * NT + tid]) * scale;
        }
        if (has_signs) {
            reg[i] *= signs_row[i * NT + tid];
        }
    }

    fwht_block_transform<N, NT>(reg, s);
    fwht_block_store<N, NT>(reg, dst);
}

// rms_norm with its weight multiply (as rms_norm_f32<1024, true>), the sign flip and the transform in one launch.
// Block (c, t) reduces token t's whole row as rms_norm does, then transforms its chunk c. normed: the multiply's result or nullptr.
template <int N>
__launch_bounds__(1024, 1)
__global__ void rms_norm_fwht_cuda(const float * x, const float * w, const float * signs, float * normed, float * dst,
                                   const int ncols, const float eps, const float scale, const bool pdl_trigger) {
    if (pdl_trigger) {
        ggml_cuda_pdl_lc();
    }
    constexpr int NT = 1024;
    constexpr int NE = N / NT;

    __shared__ float s[N];
    __shared__ float s_sum[32];

    const int     tid = threadIdx.x;
    const int64_t e0  = (int64_t) blockIdx.y * ncols + (int64_t) blockIdx.x * N;
    x += (int64_t) blockIdx.y * ncols;

    float tmp = 0.0f;

    ggml_cuda_pdl_sync();
    for (int col = tid; col < ncols; col += NT) {
        const float xi = x[col];
        tmp += xi * xi;
    }
    tmp = block_reduce<block_reduce_method::SUM, NT>(tmp, s_sum);

    const float mean      = tmp / ncols;
    const float rms_scale = rsqrtf(mean + eps);

    float reg[NE];
#pragma unroll
    for (int i = 0; i < NE; ++i) {
        const int   col = blockIdx.x * N + i * NT + tid;
        const float v   = rms_scale * x[col] * w[col];
        if (normed != nullptr) {
            normed[e0 + i * NT + tid] = v;
        }
        reg[i] = v * scale;
        reg[i] *= signs[col];
    }

    fwht_block_transform<N, NT>(reg, s);
    fwht_block_store<N, NT>(reg, dst + e0);
}

static bool fwht_legacy() {
    static const bool legacy = getenv("GGML_CUDA_FWHT_LEGACY") != nullptr;
    return legacy;
}

// The kernels let their dependents launch as soon as they start (PDL). The next kernels, typically src1's q8_1
// quantization and the matmul after it, wait in ggml_cuda_pdl_sync until this one is done before they read what it
// writes. Until now they launched only after it finished, which put a launch between every Hadamard rotation and its
// matmul and kept the matmul from requesting its weights early. GGML_CUDA_FWHT_PDL_LEGACY=1 restores that.
static bool fwht_pdl_trigger() {
    static const bool pdl_trigger = [] {
        const char * s = getenv("GGML_CUDA_FWHT_PDL_LEGACY");
        return s == nullptr || atoi(s) == 0;
    }();
    return pdl_trigger;
}

template <typename T>
static bool fwht_launch(ggml_backend_cuda_context & ctx, const T * src_d, float * dst_d,
                        const int n, const int64_t rows, const float scale,
                        const float * signs, const int n_blk, const fwht_src_view * view = nullptr) {
    // only fwht_cuda_block reads through a view, and only an F32 source with signs
    GGML_ASSERT(view == nullptr || (std::is_same_v<T, float> && signs != nullptr && n >= 512 && !fwht_legacy()));
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const int rows_per_block = 4;
    const int64_t num_blocks = (rows + rows_per_block - 1) / rows_per_block;
    cudaStream_t stream = ctx.stream();
    dim3 grid_dims(num_blocks, 1, 1);
    dim3 block_dims(warp_size, rows_per_block, 1);
    const ggml_cuda_kernel_launch_params launch_params =
        ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);

    const bool pdl_trigger = fwht_pdl_trigger();

    switch (n) {
#define FWHT_CASE(NN) \
        case NN: \
            if (signs) { \
                ggml_cuda_kernel_launch(fwht_cuda<NN, T, true>,  launch_params, src_d, dst_d, rows, scale, signs, n_blk, pdl_trigger); \
            } else { \
                ggml_cuda_kernel_launch(fwht_cuda<NN, T, false>, launch_params, src_d, dst_d, rows, scale, nullptr, 1, pdl_trigger); \
            } \
            return true;
        FWHT_CASE(64)
        FWHT_CASE(128)
        FWHT_CASE(256)
        default:
            break;
    }
    // From 512 up, one block of FWHT_BLOCK_THREADS per row (fwht_cuda_block).
    // The older kernels were the largest single kernel of a decode step at these widths.
    // GGML_CUDA_FWHT_LEGACY=1 restores them for A/B.
#define FWHT_SMEM_CASE(NN) \
        case NN: { \
            const dim3 g((unsigned) rows, 1, 1), b(FWHT_SMEM_THREADS, 1, 1); \
            const ggml_cuda_kernel_launch_params lp = ggml_cuda_kernel_launch_params(g, b, 0, stream); \
            if (signs) { \
                ggml_cuda_kernel_launch(fwht_cuda_smem<NN, T, true>,  lp, src_d, dst_d, rows, scale, signs, n_blk, pdl_trigger); \
            } else { \
                ggml_cuda_kernel_launch(fwht_cuda_smem<NN, T, false>, lp, src_d, dst_d, rows, scale, nullptr, 1, pdl_trigger); \
            } \
            return true; \
        }
#define FWHT_BLOCK_CASE(NN) \
        case NN: { \
            const dim3 g((unsigned) rows, 1, 1), b(FWHT_BLOCK_THREADS, 1, 1); \
            const ggml_cuda_kernel_launch_params lp = ggml_cuda_kernel_launch_params(g, b, 0, stream); \
            if constexpr (std::is_same_v<T, float>) { \
                if (view) { \
                    ggml_cuda_kernel_launch(fwht_cuda_block<NN, FWHT_BLOCK_THREADS, T, true, true>, lp, src_d, dst_d, rows, scale, signs, n_blk, pdl_trigger, *view); \
                    return true; \
                } \
            } \
            if (signs) { \
                ggml_cuda_kernel_launch(fwht_cuda_block<NN, FWHT_BLOCK_THREADS, T, true>,  lp, src_d, dst_d, rows, scale, signs, n_blk, pdl_trigger, fwht_src_view{}); \
            } else { \
                ggml_cuda_kernel_launch(fwht_cuda_block<NN, FWHT_BLOCK_THREADS, T, false>, lp, src_d, dst_d, rows, scale, nullptr, 1, pdl_trigger, fwht_src_view{}); \
            } \
            return true; \
        }
    if (fwht_legacy()) {
        switch (n) {
            FWHT_CASE(512)
            FWHT_CASE(1024)
            FWHT_CASE(2048)
            FWHT_SMEM_CASE(4096)
            FWHT_SMEM_CASE(8192)
            default:
                return false;
        }
    }
    switch (n) {
        FWHT_BLOCK_CASE(512)
        FWHT_BLOCK_CASE(1024)
        FWHT_BLOCK_CASE(2048)
        FWHT_BLOCK_CASE(4096)
        FWHT_BLOCK_CASE(8192)
#undef FWHT_CASE
#undef FWHT_SMEM_CASE
#undef FWHT_BLOCK_CASE
        default:
            return false;
    }
}

static bool fwht_dispatch(ggml_backend_cuda_context & ctx, const ggml_tensor * src, ggml_tensor * dst,
                          const ggml_tensor * signs_t) {
    GGML_ASSERT(ggml_nelements(src) == ggml_nelements(dst));
    if (!ggml_is_contiguous(src) || !ggml_is_contiguous(dst)) {
        return false;
    }
    const int     n    = dst->ne[0];
    const int64_t rows = ggml_nelements(dst) / n;

    if ((src->type != GGML_TYPE_F32 && src->type != GGML_TYPE_F16) || dst->type != GGML_TYPE_F32) {
        return false;
    }

    const float * signs = nullptr;
    int n_blk = 1;
    if (signs_t) {
        if (signs_t->type != GGML_TYPE_F32 || !ggml_is_contiguous(signs_t) || signs_t->ne[0] % n != 0) {
            return false;
        }
        signs = (const float *) signs_t->data;
        n_blk = signs_t->ne[0] / n;
    }

    float * dst_d = (float *) dst->data;
    const float scale = 1 / sqrtf(n);

    if (src->type == GGML_TYPE_F32) {
        return fwht_launch<float>(ctx, (const float *) src->data, dst_d, n, rows, scale, signs, n_blk);
    }
    return fwht_launch<half>(ctx, (const half *) src->data, dst_d, n, rows, scale, signs, n_blk);
}

bool ggml_cuda_op_fwht(ggml_backend_cuda_context & ctx, const ggml_tensor * src, ggml_tensor * dst) {
    GGML_ASSERT(ggml_are_same_shape(src, dst));
    return fwht_dispatch(ctx, src, dst, nullptr);
}

bool ggml_cuda_op_fwht_signed(ggml_backend_cuda_context & ctx, const ggml_tensor * src,
                              const ggml_tensor * signs, ggml_tensor * dst) {
    return fwht_dispatch(ctx, src, dst, signs);
}

bool ggml_cuda_op_fwht_view(ggml_backend_cuda_context & ctx, const ggml_tensor * src, const ggml_tensor * signs,
                            ggml_tensor * dst) {
    const int     n    = dst->ne[0];
    const int64_t rows = ggml_nelements(dst) / n;
    const size_t  ts   = ggml_type_size(src->type);
    const bool block_kernel = n >= 512 && n <= 8192 && (n & (n - 1)) == 0 && !fwht_legacy();
    if (!block_kernel || src->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || !ggml_is_contiguous(dst) ||
            ggml_nelements(src) != ggml_nelements(dst) || ggml_nelements(dst) > INT_MAX ||
            signs->type != GGML_TYPE_F32 || !ggml_is_contiguous(signs) || signs->ne[0] % n != 0 ||
            src->nb[0] % ts != 0 || src->nb[1] % ts != 0 || src->nb[2] % ts != 0 || src->nb[3] % ts != 0) {
        return false;
    }

    const fwht_src_view view = {
        init_fastdiv_values(src->ne[0]), init_fastdiv_values(src->ne[1]), init_fastdiv_values(src->ne[2]),
        (int64_t) (src->nb[0] / ts), (int64_t) (src->nb[1] / ts), (int64_t) (src->nb[2] / ts), (int64_t) (src->nb[3] / ts),
    };
    return fwht_launch<float>(ctx, (const float *) src->data, (float *) dst->data, n, rows, 1 / sqrtf(n),
                              (const float *) signs->data, signs->ne[0] / n, &view);
}

bool ggml_cuda_rms_norm_fwht_supported(const ggml_tensor * rms_norm, const ggml_tensor * w, const ggml_tensor * normed,
                                       const ggml_tensor * signs, const ggml_tensor * dst) {
    const ggml_tensor * x = rms_norm->src[0];
    const int64_t ncols = x->ne[0];
    const int64_t n     = dst->ne[0];
    // from 1,024 columns rms_norm reduces with 1024 threads, the reduction this kernel repeats
    return !fwht_legacy() && (n == 1024 || n == 2048 || n == 4096) && ggml_nrows(x) <= 8 &&
        x->type == GGML_TYPE_F32 && ggml_is_contiguous(x) && ncols >= 1024 && ncols % n == 0 &&
        w->type == GGML_TYPE_F32 && ggml_is_contiguous(w) && w->ne[0] == ncols && ggml_nrows(w) == 1 &&
        signs->type == GGML_TYPE_F32 && ggml_is_contiguous(signs) && signs->ne[0] == ncols && ggml_nrows(signs) == 1 &&
        dst->type == GGML_TYPE_F32 && ggml_is_contiguous(dst) && ggml_nelements(dst) == ggml_nelements(x) &&
        (normed == nullptr || (normed->type == GGML_TYPE_F32 && ggml_is_contiguous(normed)));
}

void ggml_cuda_op_rms_norm_fwht(ggml_backend_cuda_context & ctx, const ggml_tensor * rms_norm, const ggml_tensor * w,
                                ggml_tensor * normed, const ggml_tensor * signs, ggml_tensor * dst) {
    GGML_ASSERT(ggml_cuda_rms_norm_fwht_supported(rms_norm, w, normed, signs, dst));
    const ggml_tensor * x = rms_norm->src[0];
    const int     ncols = x->ne[0];
    const int     n     = dst->ne[0];
    const int64_t ntok  = ggml_nrows(x);

    float eps;
    memcpy(&eps, rms_norm->op_params, sizeof(float));

    const float * x_d      = (const float *) x->data;
    const float * w_d      = (const float *) w->data;
    const float * signs_d  = (const float *) signs->data;
    float       * normed_d = normed ? (float *) normed->data : nullptr;
    float       * dst_d    = (float *) dst->data;
    const float   scale    = 1 / sqrtf(n);

    const bool pdl_trigger = fwht_pdl_trigger();

    const dim3 grid(ncols / n, ntok, 1), block(1024, 1, 1);
    const ggml_cuda_kernel_launch_params lp = ggml_cuda_kernel_launch_params(grid, block, 0, ctx.stream());
    switch (n) {
        case 1024:
            ggml_cuda_kernel_launch(rms_norm_fwht_cuda<1024>, lp, x_d, w_d, signs_d, normed_d, dst_d, ncols, eps, scale, pdl_trigger);
            break;
        case 2048:
            ggml_cuda_kernel_launch(rms_norm_fwht_cuda<2048>, lp, x_d, w_d, signs_d, normed_d, dst_d, ncols, eps, scale, pdl_trigger);
            break;
        default:
            ggml_cuda_kernel_launch(rms_norm_fwht_cuda<4096>, lp, x_d, w_d, signs_d, normed_d, dst_d, ncols, eps, scale, pdl_trigger);
            break;
    }
}
