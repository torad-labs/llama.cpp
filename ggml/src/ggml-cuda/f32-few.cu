#include "ggml.h"
#include "f32-few.cuh"

#include <algorithm>

// Many rows of weights against few columns, split over K: a block takes `rows` rows and its warps split K, each lane
// of a group of 8 reads a float4 of the rows and of the columns of its column group. The sums of the 8 lanes, then of
// the warps, run in a fixed order, so the result does not depend on timing.
#define F32_COLS_WARPS  12
#define F32_COLS_KLANES 8
#define F32_COLS_MAX    32

template <int rows, int cpl> // rows of a block, columns of a lane (a warp has 4 column groups)
static __global__ __launch_bounds__(F32_COLS_WARPS*WARP_SIZE, 1) void f32_cols(
        const float * a, const float * x, float * dst, const int64_t k, const int64_t nrows, const int ncols,
        const int64_t s01, const int64_t s11, const int64_t ldc) {
    constexpr int ncol = 4*cpl;
    __shared__ float partial[F32_COLS_WARPS][rows][ncol];

    const int warp = threadIdx.x/WARP_SIZE;
    const int lane = threadIdx.x%WARP_SIZE;
    const int kl   = lane%F32_COLS_KLANES;
    const int cg   = lane/F32_COLS_KLANES;
    const int64_t row0 = int64_t(blockIdx.x)*rows;

    // rows and columns past the end read the last one and are not written
    const float * ar[rows];
#pragma unroll
    for (int r = 0; r < rows; ++r) {
        ar[r] = a + min(row0 + r, nrows - 1)*s01;
    }
    const float * xc[cpl];
#pragma unroll
    for (int j = 0; j < cpl; ++j) {
        xc[j] = x + min(int64_t(cg*cpl + j), int64_t(ncols - 1))*s11;
    }

    float acc[rows][cpl] = {};
    constexpr int64_t kstep = F32_COLS_WARPS*F32_COLS_KLANES*4;
#pragma unroll 4
    for (int64_t k0 = (warp*F32_COLS_KLANES + kl)*4; k0 < k; k0 += kstep) {
        float4 av[rows];
#pragma unroll
        for (int r = 0; r < rows; ++r) {
            av[r] = __ldg((const float4 *) (ar[r] + k0));
        }
#pragma unroll
        for (int j = 0; j < cpl; ++j) {
            const float4 xv = __ldg((const float4 *) (xc[j] + k0));
#pragma unroll
            for (int r = 0; r < rows; ++r) {
                acc[r][j] = fmaf(av[r].x, xv.x, acc[r][j]);
                acc[r][j] = fmaf(av[r].y, xv.y, acc[r][j]);
                acc[r][j] = fmaf(av[r].z, xv.z, acc[r][j]);
                acc[r][j] = fmaf(av[r].w, xv.w, acc[r][j]);
            }
        }
    }

#pragma unroll
    for (int r = 0; r < rows; ++r) {
#pragma unroll
        for (int j = 0; j < cpl; ++j) {
#pragma unroll
            for (int offset = F32_COLS_KLANES/2; offset > 0; offset >>= 1) {
                acc[r][j] += ggml_cuda_shfl_xor(acc[r][j], offset, WARP_SIZE);
            }
            if (kl == 0) {
                partial[warp][r][cg*cpl + j] = acc[r][j];
            }
        }
    }
    __syncthreads();

    for (int o = threadIdx.x; o < rows*ncol; o += blockDim.x) {
        const int r = o%rows;
        const int c = o/rows;
        if (row0 + r < nrows && c < ncols) {
            float sum = 0.0f;
#pragma unroll
            for (int w = 0; w < F32_COLS_WARPS; ++w) {
                sum += partial[w][r][c];
            }
            dst[c*ldc + row0 + r] = sum;
        }
    }
}

template <int rows, int cpl>
static void launch_f32_cols(cudaStream_t stream, const float * a, const float * x, float * dst, const int64_t k,
        const int64_t nrows, const int ncols, const int64_t s01, const int64_t s11, const int64_t ldc) {
    const dim3 grid((nrows + rows - 1)/rows);
    f32_cols<rows, cpl><<<grid, F32_COLS_WARPS*WARP_SIZE, 0, stream>>>(a, x, dst, k, nrows, ncols, s01, s11, ldc);
}

template <int rows>
static void launch_f32_cols_rows(cudaStream_t stream, const float * a, const float * x, float * dst, const int64_t k,
        const int64_t nrows, const int ncols, const int64_t s01, const int64_t s11, const int64_t ldc) {
    if (ncols <= 4) {
        launch_f32_cols<rows, 1>(stream, a, x, dst, k, nrows, ncols, s01, s11, ldc);
    } else if (ncols <= 8) {
        launch_f32_cols<rows, 2>(stream, a, x, dst, k, nrows, ncols, s01, s11, ldc);
    } else if (ncols <= 16) {
        launch_f32_cols<rows, 4>(stream, a, x, dst, k, nrows, ncols, s01, s11, ldc);
    } else if (ncols <= 24) {
        launch_f32_cols<rows, 6>(stream, a, x, dst, k, nrows, ncols, s01, s11, ldc);
    } else {
        launch_f32_cols<rows, 8>(stream, a, x, dst, k, nrows, ncols, s01, s11, ldc);
    }
}

// A short K: a thread keeps the K weights of its row in registers and sweeps a tile of columns, which every thread of
// the block reads alike.
#define F32_SHORT_TILE    8
#define F32_SHORT_THREADS 128
#define F32_SHORT_MAX_K4  8

template <int k4>
static __global__ void f32_short_k(const float * w, const float * x, float * dst, const int64_t nrows, const int64_t ncols,
        const int64_t s01, const int64_t s11, const int64_t ldc) {
    const int64_t e = int64_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (e >= nrows) {
        return;
    }
    float4 wv[k4];
#pragma unroll
    for (int j = 0; j < k4; ++j) {
        wv[j] = __ldg((const float4 *) (w + e*s01) + j);
    }
    const int64_t c0 = int64_t(blockIdx.y)*F32_SHORT_TILE;
#pragma unroll
    for (int t = 0; t < F32_SHORT_TILE; ++t) {
        if (c0 + t < ncols) {
            const float4 * xv = (const float4 *) (x + (c0 + t)*s11);
            float sum = 0.0f;
#pragma unroll
            for (int j = 0; j < k4; ++j) {
                const float4 v = __ldg(xv + j);
                sum = fmaf(wv[j].x, v.x, sum);
                sum = fmaf(wv[j].y, v.y, sum);
                sum = fmaf(wv[j].z, v.z, sum);
                sum = fmaf(wv[j].w, v.w, sum);
            }
            dst[(c0 + t)*ldc + e] = sum;
        }
    }
}

template <int k4>
static void launch_f32_short_k(cudaStream_t stream, const float * w, const float * x, float * dst, const int64_t nrows,
        const int64_t ncols, const int64_t s01, const int64_t s11, const int64_t ldc) {
    const dim3 grid((nrows + F32_SHORT_THREADS - 1)/F32_SHORT_THREADS, (ncols + F32_SHORT_TILE - 1)/F32_SHORT_TILE);
    f32_short_k<k4><<<grid, F32_SHORT_THREADS, 0, stream>>>(w, x, dst, nrows, ncols, s01, s11, ldc);
}

static bool aligned16(const void * p) {
    return reinterpret_cast<uintptr_t>(p)%16 == 0;
}

bool ggml_cuda_mul_mat_f32_few(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    static const bool enabled = ggml_env_switch("GGML_CUDA_F32_FEW");
    if (!enabled || src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    GGML_TENSOR_BINARY_OP_LOCALS

    // K contiguous and in float4s, one matrix of weights, dst rows contiguous
    if (ne02 != 1 || ne03 != 1 || ne13 != 1 || nb00 != sizeof(float) || nb10 != sizeof(float) || nb0 != sizeof(float) ||
            ne00 % 4 != 0 || nb01 % 16 != 0 || nb11 % 16 != 0 || !aligned16(src0->data) || !aligned16(src1->data)) {
        return false;
    }
    const int64_t s01 = nb01/sizeof(float);
    const int64_t s11 = nb11/sizeof(float);
    cudaStream_t stream = ctx.stream();

    // the table: [K, rows] weights shared by the columns and batches of src1, which are packed, dst packed alike
    if (ne00 <= 4*F32_SHORT_MAX_K4 && nb11 == ne00*sizeof(float) && nb12 == ne11*nb11 &&
            nb1 == ne0*sizeof(float) && nb2 == ne1*nb1) {
        const float * w = (const float *) src0->data;
        const float * x = (const float *) src1->data;
        float * d = (float *) dst->data;
        const int64_t ncols = ne11*ne12;
        const int64_t ldc = ne0;
        switch (ne00/4) {
            case 1: launch_f32_short_k<1>(stream, w, x, d, ne01, ncols, s01, s11, ldc); break;
            case 2: launch_f32_short_k<2>(stream, w, x, d, ne01, ncols, s01, s11, ldc); break;
            case 3: launch_f32_short_k<3>(stream, w, x, d, ne01, ncols, s01, s11, ldc); break;
            case 4: launch_f32_short_k<4>(stream, w, x, d, ne01, ncols, s01, s11, ldc); break;
            case 5: launch_f32_short_k<5>(stream, w, x, d, ne01, ncols, s01, s11, ldc); break;
            case 6: launch_f32_short_k<6>(stream, w, x, d, ne01, ncols, s01, s11, ldc); break;
            case 7: launch_f32_short_k<7>(stream, w, x, d, ne01, ncols, s01, s11, ldc); break;
            default: launch_f32_short_k<8>(stream, w, x, d, ne01, ncols, s01, s11, ldc); break;
        }
        return true;
    }

    if (ne12 == 1 && ne11 <= F32_COLS_MAX) {
        // the router: weights [K, rows] against up to 32 columns
        const int nsm = ggml_cuda_info().devices[ctx.device].nsm;
        const int64_t ldc = nb1/sizeof(float);
        const int rows = ne01 <= nsm ? 1 : ne01 <= 2*nsm ? 2 : ne01 <= 3*nsm ? 3 : 4;
        const float * a = (const float *) src0->data;
        const float * x = (const float *) src1->data;
        float * d = (float *) dst->data;
        switch (rows) {
            case 1: launch_f32_cols_rows<1>(stream, a, x, d, ne00, ne01, ne11, s01, s11, ldc); break;
            case 2: launch_f32_cols_rows<2>(stream, a, x, d, ne00, ne01, ne11, s01, s11, ldc); break;
            case 3: launch_f32_cols_rows<3>(stream, a, x, d, ne00, ne01, ne11, s01, s11, ldc); break;
            default: launch_f32_cols_rows<4>(stream, a, x, d, ne00, ne01, ne11, s01, s11, ldc); break;
        }
        return true;
    }
    return false;
}
