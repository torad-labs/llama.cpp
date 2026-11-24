// Standalone microbenchmark: a Hadamard rotation as a bf16 tensor-core matmul against the FWHT butterfly.
//
// Rotation at width 4608 is the engine's normalized block Hadamard (block 512, nine blocks), the map the
// signed FWHT in ggml/src/ggml-cuda/fwht.cu computes. The tensor-core arm multiplies bf16(x) by the signed
// block Hadamard with fp32 accumulate; the butterfly arm copies fwht_block_transform<512, 256> verbatim.
// Both arms write f32 or fused q8_1 blocks, quantized as ggml/src/ggml-cuda/quantize.cu does.
//
// The shared arms read the one Sylvester H_512 every block shares (256 KB) and put the block's signs on the
// input, which is the true basis of the rotation; the per-block arms read nine signed copies (2.25 MB). The cuBLAS
// arms run the same product as a tuned bf16 tensor-core GEMM against the shared H_512.
//
// Modes:
//   --device                          device properties as JSON
//   --measure --rows A,B,.. --seed N  timed arms per shape, error figures, checksums
//   --selftest --rows A --seed N      one run per arm, checksums only
//   --replay --rows A,B,.. --seed N   CPU-only reference and bf16 simulation
//
// Metrics and checksums are reduced in double. The host holds one row of floats at a time.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <mma.h>

using namespace nvcuda;

#define CU_CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "cuda error %s at line %d\n", cudaGetErrorString(e_), __LINE__); exit(3); } } while (0)
#define CUBLAS_CHECK(x) do { cublasStatus_t s_ = (x); if (s_ != CUBLAS_STATUS_SUCCESS) { \
    fprintf(stderr, "cublas error %d at line %d\n", (int) s_, __LINE__); exit(3); } } while (0)

constexpr int WIDTH = 4608;              // hidden width
constexpr int BLOCK = 512;               // rotation block (power of two, the engine's block Hadamard)
constexpr int NBLK  = WIDTH / BLOCK;     // 9 blocks
constexpr int QK8   = 32;                // q8_1 block
constexpr int QROW  = WIDTH / QK8;       // 144 q8_1 blocks a row
constexpr int BK    = 64;                // K chunk of the tensor-core arm

struct block_q8_1_s {
    __half2  ds;
    int8_t   qs[QK8];
};
static_assert(sizeof(block_q8_1_s) == 36, "q8_1 block layout");

static __host__ __device__ inline uint32_t rot_hash(uint32_t idx, uint32_t seed) {
    uint32_t h = idx * 747796405u + seed;
    h = ((h >> ((h >> 28) + 4)) ^ h) * 277803737u;
    h = (h >> 22) ^ h;
    return h;
}

static __host__ __device__ inline float rot_input(uint32_t idx, uint32_t seed) {
    const uint32_t h = rot_hash(idx, seed);
    return (float) (h >> 8) * (1.0f / 16777216.0f) * 2.0f - 1.0f;
}

static __host__ __device__ inline int rot_sign(uint32_t idx, uint32_t seed) {
    return (rot_hash(idx ^ 0x9e3779b9u, seed ^ 0x85ebca6bu) & 0x10000u) ? -1 : 1;
}

struct sample_pos { int row, col; };
static std::vector<sample_pos> sample_positions(int rows) {
    return {{0, 0}, {0, 1}, {0, BLOCK - 1}, {0, BLOCK}, {rows / 2, 100},
            {rows - 1, WIDTH - 1}, {std::min(1, rows - 1), 2304}, {rows / 2, WIDTH - 1}};
}

static inline float bf16_round_host(float v) {
    uint32_t u;
    memcpy(&u, &v, 4);
    const uint32_t lsb = (u >> 16) & 1u;
    u += 0x7FFFu + lsb;
    u &= 0xFFFF0000u;
    float r;
    memcpy(&r, &u, 4);
    return r;
}

static void butterfly512(float * v) {
    for (int h = 1; h < BLOCK; h *= 2) {
        for (int base = 0; base < BLOCK; base += 2 * h) {
            for (int j = 0; j < h; ++j) {
                const float x = v[base + j];
                const float y = v[base + j + h];
                v[base + j]     = x + y;
                v[base + j + h] = x - y;
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------
// device kernels

__global__ void fill_input_kernel(float * x, long long n, uint32_t seed) {
    const long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        x[i] = rot_input((uint32_t) i, seed);
    }
}

// The shared basis: H_512 (row-major, int8) followed by the nine blocks' signs, as one stream. Also H_512 in bf16.
__global__ void init_shared_basis_kernel(int8_t * hshared, __nv_bfloat16 * h_bf16, uint32_t seed) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < BLOCK * BLOCK) {
        const int i = idx / BLOCK;
        const int j = idx % BLOCK;
        const int h = (__popc((unsigned) (i & j)) & 1) ? -1 : 1;
        hshared[idx] = (int8_t) h;
        h_bf16[idx] = __float2bfloat16((float) h);
    } else if (idx < BLOCK * BLOCK + NBLK * BLOCK) {
        hshared[idx] = (int8_t) rot_sign((uint32_t) (idx - BLOCK * BLOCK), seed);
    }
}

// The cuBLAS arms' input: each element times its block's sign, rounded to bf16 (exact for the sign).
__global__ void sign_bf16_kernel(const float * __restrict__ x, const float * __restrict__ signs_f,
                                 __nv_bfloat16 * __restrict__ xs, long long n) {
    const long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        xs[i] = __float2bfloat16(x[i] * signs_f[i % WIDTH]);
    }
}

__global__ void init_basis_kernel(int8_t * basis, int8_t * signs, float * signs_f, uint32_t seed) {
    const long long idx = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long) NBLK * BLOCK * BLOCK) {
        return;
    }
    const int b = (int) (idx / (BLOCK * BLOCK));
    const int rem = (int) (idx % (BLOCK * BLOCK));
    const int i = rem / BLOCK;
    const int j = rem % BLOCK;
    const int sg = rot_sign((uint32_t) (b * BLOCK + i), seed);
    const int h  = (__popc((unsigned) (i & j)) & 1) ? -1 : 1;
    basis[idx] = (int8_t) (sg * h);
    if (j == 0) {
        signs[b * BLOCK + i] = (int8_t) sg;
        signs_f[b * BLOCK + i] = (float) sg;
    }
}

// One block per (row, 512-block). The transform body is fwht.cu's fwht_block_transform<512, 256>, copied so the
// baseline is the engine's kernel and not a reimplementation.
template <bool QUANT>
__global__ __launch_bounds__(256) void fwht_kernel(
        const float * __restrict__ src, float * __restrict__ dst, block_q8_1_s * __restrict__ q8,
        const float * __restrict__ signs, int rows, float scale) {
    constexpr int N  = BLOCK;
    constexpr int NT = 256;
    constexpr int NE = N / NT;

    const int64_t r = blockIdx.x;
    const int m = (int) (r / NBLK);
    const int b = (int) (r % NBLK);
    if (m >= rows) {
        return;
    }

    __shared__ float s[N];
    float reg[NE];
    const int tid  = threadIdx.x;
    const int lane = tid % 32;

    const float * x    = src + (int64_t) m * WIDTH + b * N;
    const float * srow = signs + b * N;
#pragma unroll
    for (int i = 0; i < NE; ++i) {
        reg[i] = x[i * NT + tid] * scale;
        reg[i] *= srow[i * NT + tid];
    }

#pragma unroll
    for (int h = 1; h < 32; h *= 2) {
#pragma unroll
        for (int j = 0; j < NE; ++j) {
            const float val  = reg[j];
            const float val2 = __shfl_xor_sync(0xffffffffu, val, h);
            reg[j] = (lane & h) == 0 ? val + val2 : val2 - val;
        }
    }
#pragma unroll
    for (int h = 32; h < NT; h *= 2) {
#pragma unroll
        for (int j = 0; j < NE; ++j) {
            s[j * NT + tid] = reg[j];
        }
        __syncthreads();
#pragma unroll
        for (int j = 0; j < NE; ++j) {
            const float val  = reg[j];
            const float val2 = s[j * NT + (tid ^ h)];
            reg[j] = (tid & h) == 0 ? val + val2 : val2 - val;
        }
        __syncthreads();
    }
#pragma unroll
    for (int h = NT; h < N; h *= 2) {
        const int step = h / NT;
#pragma unroll
        for (int j = 0; j < NE; j += 2 * step) {
#pragma unroll
            for (int k = 0; k < step; ++k) {
                const float a = reg[j + k];
                const float c = reg[j + k + step];
                reg[j + k]        = a + c;
                reg[j + k + step] = a - c;
            }
        }
    }

    if (QUANT) {
        // quantize_q8_1's arithmetic: one warp a 32-value block, d = amax/127, integer sum for the second half.
        const int warp = tid / 32;
#pragma unroll
        for (int i = 0; i < NE; ++i) {
            const float v = reg[i];
            float amax = fabsf(v);
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
            }
            const float d = amax / 127.0f;
            const int8_t qv = amax == 0.0f ? (int8_t) 0 : (int8_t) roundf(v / d);
            int sum = (int) qv;
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                sum += __shfl_xor_sync(0xffffffffu, sum, off);
            }
            block_q8_1_s * out = q8 + (int64_t) m * QROW + b * (BLOCK / QK8) + i * 8 + warp;
            out->qs[lane] = qv;
            if (lane == 0) {
                out->ds = __floats2half2_rn(d, d * (float) sum);
            }
        }
    } else {
#pragma unroll
        for (int i = 0; i < NE; ++i) {
            dst[(int64_t) m * WIDTH + b * N + i * NT + tid] = reg[i];
        }
    }
}

// bf16 is the row's named input format; fp16 is the same arithmetic with 11 mantissa bits instead of 8.
template <bool FP16> struct tc_in;
template <> struct tc_in<false> { using type = __nv_bfloat16; };
template <> struct tc_in<true>  { using type = half; };
template <bool FP16> static __device__ __forceinline__ typename tc_in<FP16>::type tc_cvt(float v) {
    return FP16 ? (typename tc_in<FP16>::type) __float2half_rn(v)
                : (typename tc_in<FP16>::type) __float2bfloat16_rn(v);
}

// Tensor-core arm: BM rows x BN columns of one 512-block, K = 512. FROM_BASIS reads the stored int8 basis;
// otherwise the signed Sylvester entry is generated from the indices.
// SHARED: basis is the shared H_512 and the block's signs multiply the input instead (same products, bit for bit).
template <int BM, int BN, int WARPS, bool FROM_BASIS, bool QUANT, bool FP16 = false, bool SHARED = false>
__global__ __launch_bounds__(WARPS * 32) void tc_rotation_kernel(
        const float * __restrict__ src, const int8_t * __restrict__ basis, const int8_t * __restrict__ signs,
        float * __restrict__ dst, block_q8_1_s * __restrict__ q8, int rows, float scale) {
    static_assert(BN / 16 == WARPS, "one 16-column tile a warp");
    constexpr int NT = WARPS * 32;

    const int n_strip = blockIdx.x;
    const int m_tile  = blockIdx.y;
    const int m0      = m_tile * BM;
    const int n_global = n_strip * BN;
    const int b        = n_global / BLOCK;
    const int n_in_blk = n_global % BLOCK;
    const int tid      = threadIdx.x;
    const int warp     = tid / 32;

    using InT = typename tc_in<FP16>::type;
    __shared__ InT   A_s[BM][BK];
    __shared__ InT   B_s[BK][BN];
    __shared__ float Y_s[BM][BN];

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[BM / 16];
#pragma unroll
    for (int s = 0; s < BM / 16; ++s) {
        wmma::fill_fragment(acc[s], 0.0f);
    }

    for (int kc = 0; kc < BLOCK / BK; ++kc) {
        for (int t = tid; t < BM * BK; t += NT) {
            const int mm = t / BK;
            const int kk = t % BK;
            const int row = m0 + mm;
            float xv = row < rows ? src[(int64_t) row * WIDTH + b * BLOCK + kc * BK + kk] : 0.0f;
            if (SHARED) {
                xv *= (float) signs[b * BLOCK + kc * BK + kk];
            }
            A_s[mm][kk] = tc_cvt<FP16>(xv);
        }
        for (int t = tid; t < BK * BN; t += NT) {
            const int kk = t / BN;
            const int nn = t % BN;
            const int i  = kc * BK + kk;
            const int j  = n_in_blk + nn;
            int8_t v;
            if (FROM_BASIS && SHARED) {
                v = basis[(int64_t) i * BLOCK + j];
            } else if (FROM_BASIS) {
                v = basis[((int64_t) b * BLOCK + i) * BLOCK + j];
            } else {
                const int sg = (int) signs[b * BLOCK + i];
                v = (int8_t) (sg * ((__popc((unsigned) (i & j)) & 1) ? -1 : 1));
            }
            B_s[kk][nn] = tc_cvt<FP16>((float) v);
        }
        __syncthreads();

#pragma unroll
        for (int s = 0; s < BM / 16; ++s) {
#pragma unroll
            for (int ks = 0; ks < BK / 16; ++ks) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, InT, wmma::row_major> fa;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, InT, wmma::row_major> fb;
                wmma::load_matrix_sync(fa, &A_s[s * 16][ks * 16], BK);
                wmma::load_matrix_sync(fb, &B_s[ks * 16][warp * 16], BN);
                wmma::mma_sync(acc[s], fa, fb, acc[s]);
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int s = 0; s < BM / 16; ++s) {
        wmma::store_matrix_sync(&Y_s[s * 16][warp * 16], acc[s], BN, wmma::mem_row_major);
    }
    __syncthreads();

    if (QUANT) {
        constexpr int QC = BN / QK8;
        for (int t = tid; t < BM * QC; t += NT) {
            const int r  = t / QC;
            const int qb = t % QC;
            const int row = m0 + r;
            const int col = qb * QK8;
            float v[QK8];
            float amax = 0.0f;
#pragma unroll
            for (int l = 0; l < QK8; ++l) {
                v[l] = Y_s[r][col + l] * scale;
                amax = fmaxf(amax, fabsf(v[l]));
            }
            const float d = amax / 127.0f;
            int8_t qv[QK8];
            int sum = 0;
#pragma unroll
            for (int l = 0; l < QK8; ++l) {
                qv[l] = d == 0.0f ? (int8_t) 0 : (int8_t) roundf(v[l] / d);
                sum += (int) qv[l];
            }
            if (row < rows) {
                block_q8_1_s * out = q8 + (int64_t) row * QROW + b * (BLOCK / QK8) + n_in_blk / QK8 + qb;
#pragma unroll
                for (int l = 0; l < QK8; ++l) {
                    out->qs[l] = qv[l];
                }
                out->ds = __floats2half2_rn(d, d * (float) sum);
            }
        }
    } else {
        for (int t = tid; t < BM * BN; t += NT) {
            const int r = t / BN;
            const int c = t % BN;
            const int row = m0 + r;
            if (row < rows) {
                dst[(int64_t) row * WIDTH + n_global + c] = Y_s[r][c] * scale;
            }
        }
    }
}

// Floor of any rotation at this shape: read the row once, write it once, the fwht arm's grid.
__global__ __launch_bounds__(128) void copy_kernel(const float * __restrict__ src, float * __restrict__ dst) {
    const int64_t i = (int64_t) blockIdx.x * BLOCK + 4 * threadIdx.x;
    *(float4 *) (dst + i) = *(const float4 *) (src + i);
}

// Floor of a stored-basis rotation's read: every byte of the basis read once with 16-byte loads, reduced in the block,
// one word stored a block.
__global__ __launch_bounds__(256) void basis_stream_kernel(const int4 * __restrict__ basis, long long n16, int * __restrict__ out) {
    __shared__ int warp_acc[8];
    int acc = 0;
    for (long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x; i < n16; i += (long long) gridDim.x * blockDim.x) {
        const int4 v = basis[i];
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    for (int o = 16; o > 0; o >>= 1) {
        acc ^= __shfl_xor_sync(0xffffffff, acc, o);
    }
    if (threadIdx.x % 32 == 0) {
        warp_acc[threadIdx.x / 32] = acc;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        int a = 0;
        for (int w = 0; w < 8; ++w) {
            a ^= warp_acc[w];
        }
        out[blockIdx.x] = a;
    }
}

static unsigned stream_blocks(long long n16) {
    return (unsigned) std::min<long long>(680, (n16 + 255) / 256);
}

__global__ void dequant_kernel(const block_q8_1_s * __restrict__ q8, float * __restrict__ out, long long n) {
    const long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    const int row = (int) (i / WIDTH);
    const int col = (int) (i % WIDTH);
    const block_q8_1_s * blk = q8 + (int64_t) row * QROW + col / QK8;
    out[i] = (float) blk->qs[col % QK8] * __half2float(blk->ds.x);
}

// sums[0..4]: sum_a, sumsq_a, sum_b, sumsq_b, sumsq_diff. sums[5..7]: maxabs_a, maxabs_b, maxabs_diff.
__global__ void metric_kernel(const float * __restrict__ a, const float * __restrict__ b, long long n,
                              double * __restrict__ sums) {
    double sa = 0.0, qa = 0.0, sb = 0.0, qb = 0.0, qd = 0.0;
    double ma = 0.0, mb = 0.0, md = 0.0;
    const long long stride = (long long) gridDim.x * blockDim.x;
    for (long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        const double x = a[i];
        const double y = b[i];
        const double d = x - y;
        sa += x;
        qa += x * x;
        sb += y;
        qb += y * y;
        qd += d * d;
        ma = fmax(ma, fabs(x));
        mb = fmax(mb, fabs(y));
        md = fmax(md, fabs(d));
    }
    __shared__ double sh[8][256];
    const int t = threadIdx.x;
    sh[0][t] = sa; sh[1][t] = qa; sh[2][t] = sb; sh[3][t] = qb; sh[4][t] = qd;
    sh[5][t] = ma; sh[6][t] = mb; sh[7][t] = md;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (t < s) {
            for (int k = 0; k < 5; ++k) {
                sh[k][t] += sh[k][t + s];
            }
            for (int k = 5; k < 8; ++k) {
                sh[k][t] = fmax(sh[k][t], sh[k][t + s]);
            }
        }
        __syncthreads();
    }
    if (t == 0) {
        for (int k = 0; k < 5; ++k) {
            atomicAdd(&sums[k], sh[k][0]);
        }
        for (int k = 5; k < 8; ++k) {
            atomicMax((unsigned long long *) &sums[k], __double_as_longlong(sh[k][0]));
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------
// host helpers

struct cmp_result {
    double sum_a = 0, sumsq_a = 0, sum_b = 0, sumsq_b = 0, sumsq_d = 0;
    double maxabs_a = 0, maxabs_b = 0, maxabs_d = 0;
};

static cmp_result compare_dev(const float * a, const float * b, long long n, double * scratch) {
    CU_CHECK(cudaMemset(scratch, 0, 8 * sizeof(double)));
    metric_kernel<<<256, 256>>>(a, b, n, scratch);
    CU_CHECK(cudaGetLastError());
    double h[8];
    CU_CHECK(cudaMemcpy(h, scratch, sizeof(h), cudaMemcpyDeviceToHost));
    cmp_result r;
    r.sum_a = h[0]; r.sumsq_a = h[1]; r.sum_b = h[2]; r.sumsq_b = h[3]; r.sumsq_d = h[4];
    r.maxabs_a = h[5]; r.maxabs_b = h[6]; r.maxabs_d = h[7];
    return r;
}

static double rel_l2(const cmp_result & c) {
    return c.sumsq_b > 0.0 ? std::sqrt(c.sumsq_d / c.sumsq_b) : 0.0;
}

static float sample_at(const float * d, int row, int col) {
    float v = 0.0f;
    CU_CHECK(cudaMemcpy(&v, d + (int64_t) row * WIDTH + col, sizeof(float), cudaMemcpyDeviceToHost));
    return v;
}

struct time_stats {
    std::vector<double> samples_us;
    double median_us = 0, min_us = 0, p10_us = 0, p90_us = 0;
};

static void finish_stats(time_stats & t) {
    std::vector<double> v = t.samples_us;
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    t.min_us = v.front();
    t.median_us = (n % 2) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
    t.p10_us = v[(size_t) std::floor(0.10 * (double) (n - 1))];
    t.p90_us = v[(size_t) std::floor(0.90 * (double) (n - 1))];
}

struct device_info {
    char name[256] = {0};
    int  cc_major = 0, cc_minor = 0, sm_count = 0, l2_bytes = 0;
    int  mem_clock_khz = 0, mem_bus_bits = 0, clock_khz = 0;
    char driver[64] = {0}, runtime[64] = {0};
};

static device_info get_device_info() {
    device_info di;
    cudaDeviceProp p{};
    CU_CHECK(cudaGetDeviceProperties(&p, 0));
    snprintf(di.name, sizeof(di.name), "%s", p.name);
    di.cc_major = p.major;
    di.cc_minor = p.minor;
    di.sm_count = p.multiProcessorCount;
    di.l2_bytes = (int) p.l2CacheSize;
    di.mem_bus_bits = p.memoryBusWidth;
    int v = 0;
    CU_CHECK(cudaDeviceGetAttribute(&v, cudaDevAttrMemoryClockRate, 0));
    di.mem_clock_khz = v;
    CU_CHECK(cudaDeviceGetAttribute(&v, cudaDevAttrClockRate, 0));
    di.clock_khz = v;
    CU_CHECK(cudaDriverGetVersion(&v));
    snprintf(di.driver, sizeof(di.driver), "%d.%d", v / 1000, (v % 1000) / 10);
    CU_CHECK(cudaRuntimeGetVersion(&v));
    snprintf(di.runtime, sizeof(di.runtime), "%d.%d", v / 1000, (v % 1000) / 10);
    return di;
}

// ---------------------------------------------------------------------------------------------------------------
// measurement

struct arm {
    std::string name;
    std::function<void()> launch;
    cudaGraphExec_t exec = nullptr;
    cudaGraph_t     graph = nullptr;
    time_stats      st;
};

static constexpr int REPS_PER_GRAPH = 50;
static constexpr int ROUNDS = 40;

static void time_arms(std::vector<arm> & arms, cudaStream_t stream, cudaEvent_t e0, cudaEvent_t e1) {
    for (arm & a : arms) {
        CU_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        for (int i = 0; i < REPS_PER_GRAPH; ++i) {
            a.launch();
        }
        CU_CHECK(cudaStreamEndCapture(stream, &a.graph));
        CU_CHECK(cudaGraphInstantiate(&a.exec, a.graph, 0));
    }
    for (arm & a : arms) {
        for (int i = 0; i < 10; ++i) {
            CU_CHECK(cudaGraphLaunch(a.exec, stream));
        }
    }
    CU_CHECK(cudaStreamSynchronize(stream));

    std::mt19937 rng(0x5eed1234u);
    std::vector<int> order(arms.size());
    for (size_t i = 0; i < order.size(); ++i) {
        order[i] = (int) i;
    }
    for (int r = 0; r < ROUNDS; ++r) {
        std::shuffle(order.begin(), order.end(), rng);
        for (int idx : order) {
            arm & a = arms[idx];
            CU_CHECK(cudaEventRecord(e0, stream));
            CU_CHECK(cudaGraphLaunch(a.exec, stream));
            CU_CHECK(cudaEventRecord(e1, stream));
            CU_CHECK(cudaEventSynchronize(e1));
            float ms = 0.0f;
            CU_CHECK(cudaEventElapsedTime(&ms, e0, e1));
            a.st.samples_us.push_back((double) ms * 1000.0 / REPS_PER_GRAPH);
        }
    }
    for (arm & a : arms) {
        finish_stats(a.st);
        CU_CHECK(cudaGraphExecDestroy(a.exec));
        CU_CHECK(cudaGraphDestroy(a.graph));
    }
}

static const char * CONFIG_NAMES[3] = {"bm16_bn64_w4", "bm16_bn128_w8", "bm32_bn128_w8"};

template <int BM, int BN, int WARPS, bool FROM_BASIS, bool QUANT, bool FP16 = false, bool SHARED = false>
static void launch_tc(const float * x, const int8_t * basis, const int8_t * signs, float * dst,
                      block_q8_1_s * q8, int rows, float scale, cudaStream_t stream) {
    const dim3 grid(WIDTH / BN, (rows + BM - 1) / BM, 1);
    tc_rotation_kernel<BM, BN, WARPS, FROM_BASIS, QUANT, FP16, SHARED><<<grid, WARPS * 32, 0, stream>>>(
        x, basis, signs, dst, q8, rows, scale);
    CU_CHECK(cudaGetLastError());
}

static constexpr size_t CUBLAS_WS_BYTES = 32u << 20;
static cublasHandle_t g_cublas = nullptr;

// Y_b = (s_b * X_b) H_512 * scale for the nine blocks, row-major X and Y: column-major C = H^T X^T with H symmetric.
static void launch_cublas(const __nv_bfloat16 * xs, const __nv_bfloat16 * h, float * y, int rows, float scale) {
    const float beta = 0.0f;
    CUBLAS_CHECK(cublasGemmStridedBatchedEx(g_cublas, CUBLAS_OP_N, CUBLAS_OP_N, BLOCK, rows, BLOCK, &scale,
        h, CUDA_R_16BF, BLOCK, 0, xs, CUDA_R_16BF, WIDTH, BLOCK, &beta, y, CUDA_R_32F, WIDTH, BLOCK, NBLK,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
}

struct buffers {
    float * x = nullptr;
    float * y_ref = nullptr;
    float * y_tc = nullptr;
    float * y_gen = nullptr;
    float * deq_ref = nullptr;
    float * deq_tc = nullptr;
    float * deq_gen = nullptr;
    float * y_h = nullptr;
    float * deq_h = nullptr;
    block_q8_1_s * q8_ref = nullptr;
    block_q8_1_s * q8_tc = nullptr;
    block_q8_1_s * q8_gen = nullptr;
    block_q8_1_s * q8_h = nullptr;
    int8_t * basis = nullptr;
    int8_t * hshared = nullptr;          // H_512 then the nine blocks' signs
    __nv_bfloat16 * h_bf16 = nullptr;    // H_512 for cuBLAS
    __nv_bfloat16 * xs_bf16 = nullptr;   // signed bf16 input for cuBLAS
    float * y_sh = nullptr;
    float * y_cub = nullptr;
    float * scratch_out = nullptr;
    void  * cublas_ws = nullptr;
    int8_t * signs = nullptr;
    float  * signs_f = nullptr;
    double * scratch = nullptr;
    int max_rows = 0;
};

static void alloc_buffers(buffers & b, int max_rows) {
    b.max_rows = max_rows;
    const size_t n = (size_t) max_rows * WIDTH;
    const size_t nq = (size_t) max_rows * QROW;
    CU_CHECK(cudaMalloc(&b.x, n * sizeof(float)));
    CU_CHECK(cudaMalloc(&b.y_ref, n * sizeof(float)));
    CU_CHECK(cudaMalloc(&b.y_tc, n * sizeof(float)));
    CU_CHECK(cudaMalloc(&b.y_gen, n * sizeof(float)));
    CU_CHECK(cudaMalloc(&b.deq_ref, n * sizeof(float)));
    CU_CHECK(cudaMalloc(&b.deq_tc, n * sizeof(float)));
    CU_CHECK(cudaMalloc(&b.deq_gen, n * sizeof(float)));
    CU_CHECK(cudaMalloc(&b.y_h, n * sizeof(float)));
    CU_CHECK(cudaMalloc(&b.deq_h, n * sizeof(float)));
    CU_CHECK(cudaMalloc(&b.q8_ref, nq * sizeof(block_q8_1_s)));
    CU_CHECK(cudaMalloc(&b.q8_tc, nq * sizeof(block_q8_1_s)));
    CU_CHECK(cudaMalloc(&b.q8_gen, nq * sizeof(block_q8_1_s)));
    CU_CHECK(cudaMalloc(&b.q8_h, nq * sizeof(block_q8_1_s)));
    CU_CHECK(cudaMalloc(&b.basis, (size_t) NBLK * BLOCK * BLOCK));
    CU_CHECK(cudaMalloc(&b.hshared, (size_t) BLOCK * BLOCK + NBLK * BLOCK));
    CU_CHECK(cudaMalloc(&b.h_bf16, (size_t) BLOCK * BLOCK * sizeof(__nv_bfloat16)));
    CU_CHECK(cudaMalloc(&b.xs_bf16, n * sizeof(__nv_bfloat16)));
    CU_CHECK(cudaMalloc(&b.y_sh, n * sizeof(float)));
    CU_CHECK(cudaMalloc(&b.y_cub, n * sizeof(float)));
    CU_CHECK(cudaMalloc(&b.scratch_out, 1024 * sizeof(int)));
    CU_CHECK(cudaMalloc(&b.cublas_ws, CUBLAS_WS_BYTES));
    CU_CHECK(cudaMalloc(&b.signs, (size_t) NBLK * BLOCK));
    CU_CHECK(cudaMalloc(&b.signs_f, (size_t) NBLK * BLOCK * sizeof(float)));
    CU_CHECK(cudaMalloc(&b.scratch, 8 * sizeof(double)));
}

static void free_buffers(buffers & b) {
    cudaFree(b.x); cudaFree(b.y_ref); cudaFree(b.y_tc); cudaFree(b.y_gen);
    cudaFree(b.deq_ref); cudaFree(b.deq_tc); cudaFree(b.deq_gen);
    cudaFree(b.y_h); cudaFree(b.deq_h);
    cudaFree(b.q8_ref); cudaFree(b.q8_tc); cudaFree(b.q8_gen); cudaFree(b.q8_h);
    cudaFree(b.basis); cudaFree(b.signs); cudaFree(b.signs_f); cudaFree(b.scratch);
    cudaFree(b.hshared); cudaFree(b.h_bf16); cudaFree(b.xs_bf16); cudaFree(b.y_sh); cudaFree(b.y_cub);
    cudaFree(b.scratch_out); cudaFree(b.cublas_ws);
}

static void init_data(buffers & b, int max_rows, uint32_t seed, cudaStream_t stream) {
    const long long n = (long long) max_rows * WIDTH;
    fill_input_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, stream>>>(b.x, n, seed);
    const long long nb = (long long) NBLK * BLOCK * BLOCK;
    init_basis_kernel<<<(unsigned) ((nb + 255) / 256), 256, 0, stream>>>(b.basis, b.signs, b.signs_f, seed);
    const int ns = BLOCK * BLOCK + NBLK * BLOCK;
    init_shared_basis_kernel<<<(ns + 255) / 256, 256, 0, stream>>>(b.hshared, b.h_bf16, seed);
    sign_bf16_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, stream>>>(b.x, b.signs_f, b.xs_bf16, n);
    CU_CHECK(cudaGetLastError());
    CU_CHECK(cudaStreamSynchronize(stream));
}

static void build_arms(std::vector<arm> & arms, buffers & b, int rows, float scale, cudaStream_t stream) {
    const dim3 grid_fwht((unsigned) rows * NBLK, 1, 1);
    arms.push_back({"fwht_f32", [&b, rows, scale, grid_fwht, stream]() {
        fwht_kernel<false><<<grid_fwht, 256, 0, stream>>>(b.x, b.y_ref, nullptr, b.signs_f, rows, scale); }});
    arms.push_back({"fwht_q8", [&b, rows, scale, grid_fwht, stream]() {
        fwht_kernel<true><<<grid_fwht, 256, 0, stream>>>(b.x, nullptr, b.q8_ref, b.signs_f, rows, scale); }});
    arms.push_back({"copy_f32", [&b, grid_fwht, stream]() {
        copy_kernel<<<grid_fwht, 128, 0, stream>>>(b.x, b.y_gen); }});
    arms.push_back({"basis_stream", [&b, stream]() {
        const long long n16 = (long long) NBLK * BLOCK * BLOCK / 16;
        basis_stream_kernel<<<stream_blocks(n16), 256, 0, stream>>>((const int4 *) b.basis, n16, (int *) b.scratch_out); }});
    arms.push_back({"basis_stream_shared", [&b, stream]() {
        const long long n16 = ((long long) BLOCK * BLOCK + NBLK * BLOCK) / 16;
        basis_stream_kernel<<<stream_blocks(n16), 256, 0, stream>>>((const int4 *) b.hshared, n16, (int *) b.scratch_out); }});
    arms.push_back({"cublas_gemm", [&b, rows, scale]() { launch_cublas(b.xs_bf16, b.h_bf16, b.y_cub, rows, scale); }});
    arms.push_back({"cublas_full", [&b, rows, scale, stream]() {
        const long long n = (long long) rows * WIDTH;
        sign_bf16_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, stream>>>(b.x, b.signs_f, b.xs_bf16, n);
        launch_cublas(b.xs_bf16, b.h_bf16, b.y_cub, rows, scale); }});
    arms.push_back({std::string("tc_shared_f32/") + CONFIG_NAMES[0], [&b, rows, scale, stream]() {
        launch_tc<16, 64, 4, true, false, false, true>(b.x, b.hshared, b.signs, b.y_sh, nullptr, rows, scale, stream); }});
    arms.push_back({std::string("tc_shared_q8/") + CONFIG_NAMES[0], [&b, rows, scale, stream]() {
        launch_tc<16, 64, 4, true, true, false, true>(b.x, b.hshared, b.signs, nullptr, b.q8_tc, rows, scale, stream); }});
    arms.push_back({std::string("tc_shared_f32/") + CONFIG_NAMES[1], [&b, rows, scale, stream]() {
        launch_tc<16, 128, 8, true, false, false, true>(b.x, b.hshared, b.signs, b.y_sh, nullptr, rows, scale, stream); }});
    arms.push_back({std::string("tc_shared_q8/") + CONFIG_NAMES[1], [&b, rows, scale, stream]() {
        launch_tc<16, 128, 8, true, true, false, true>(b.x, b.hshared, b.signs, nullptr, b.q8_tc, rows, scale, stream); }});
    arms.push_back({std::string("tc_shared_f32/") + CONFIG_NAMES[2], [&b, rows, scale, stream]() {
        launch_tc<32, 128, 8, true, false, false, true>(b.x, b.hshared, b.signs, b.y_sh, nullptr, rows, scale, stream); }});
    arms.push_back({std::string("tc_shared_q8/") + CONFIG_NAMES[2], [&b, rows, scale, stream]() {
        launch_tc<32, 128, 8, true, true, false, true>(b.x, b.hshared, b.signs, nullptr, b.q8_tc, rows, scale, stream); }});

    arms.push_back({std::string("tc_basis_f32/") + CONFIG_NAMES[0], [&b, rows, scale, stream]() {
        launch_tc<16, 64, 4, true, false>(b.x, b.basis, b.signs, b.y_tc, nullptr, rows, scale, stream); }});
    arms.push_back({std::string("tc_gen_f32/") + CONFIG_NAMES[0], [&b, rows, scale, stream]() {
        launch_tc<16, 64, 4, false, false>(b.x, b.basis, b.signs, b.y_gen, nullptr, rows, scale, stream); }});
    arms.push_back({std::string("tc_basis_q8/") + CONFIG_NAMES[0], [&b, rows, scale, stream]() {
        launch_tc<16, 64, 4, true, true>(b.x, b.basis, b.signs, nullptr, b.q8_tc, rows, scale, stream); }});
    arms.push_back({std::string("tc_gen_q8/") + CONFIG_NAMES[0], [&b, rows, scale, stream]() {
        launch_tc<16, 64, 4, false, true>(b.x, b.basis, b.signs, nullptr, b.q8_gen, rows, scale, stream); }});

    arms.push_back({std::string("tc_basis_f32/") + CONFIG_NAMES[1], [&b, rows, scale, stream]() {
        launch_tc<16, 128, 8, true, false>(b.x, b.basis, b.signs, b.y_tc, nullptr, rows, scale, stream); }});
    arms.push_back({std::string("tc_gen_f32/") + CONFIG_NAMES[1], [&b, rows, scale, stream]() {
        launch_tc<16, 128, 8, false, false>(b.x, b.basis, b.signs, b.y_gen, nullptr, rows, scale, stream); }});
    arms.push_back({std::string("tc_basis_q8/") + CONFIG_NAMES[1], [&b, rows, scale, stream]() {
        launch_tc<16, 128, 8, true, true>(b.x, b.basis, b.signs, nullptr, b.q8_tc, rows, scale, stream); }});
    arms.push_back({std::string("tc_gen_q8/") + CONFIG_NAMES[1], [&b, rows, scale, stream]() {
        launch_tc<16, 128, 8, false, true>(b.x, b.basis, b.signs, nullptr, b.q8_gen, rows, scale, stream); }});

    arms.push_back({std::string("tc_basis_f32/") + CONFIG_NAMES[2], [&b, rows, scale, stream]() {
        launch_tc<32, 128, 8, true, false>(b.x, b.basis, b.signs, b.y_tc, nullptr, rows, scale, stream); }});
    arms.push_back({std::string("tc_gen_f32/") + CONFIG_NAMES[2], [&b, rows, scale, stream]() {
        launch_tc<32, 128, 8, false, false>(b.x, b.basis, b.signs, b.y_gen, nullptr, rows, scale, stream); }});
    arms.push_back({std::string("tc_basis_q8/") + CONFIG_NAMES[2], [&b, rows, scale, stream]() {
        launch_tc<32, 128, 8, true, true>(b.x, b.basis, b.signs, nullptr, b.q8_tc, rows, scale, stream); }});
    arms.push_back({std::string("tc_gen_q8/") + CONFIG_NAMES[2], [&b, rows, scale, stream]() {
        launch_tc<32, 128, 8, false, true>(b.x, b.basis, b.signs, nullptr, b.q8_gen, rows, scale, stream); }});

    // fp16 input, the 11-mantissa-bit mitigation, at the middle config
    arms.push_back({std::string("tc_basis_f32_fp16/") + CONFIG_NAMES[1], [&b, rows, scale, stream]() {
        launch_tc<16, 128, 8, true, false, true>(b.x, b.basis, b.signs, b.y_h, nullptr, rows, scale, stream); }});
    arms.push_back({std::string("tc_gen_f32_fp16/") + CONFIG_NAMES[1], [&b, rows, scale, stream]() {
        launch_tc<16, 128, 8, false, false, true>(b.x, b.basis, b.signs, b.y_h, nullptr, rows, scale, stream); }});
    arms.push_back({std::string("tc_basis_q8_fp16/") + CONFIG_NAMES[1], [&b, rows, scale, stream]() {
        launch_tc<16, 128, 8, true, true, true>(b.x, b.basis, b.signs, nullptr, b.q8_h, rows, scale, stream); }});
}

struct run_results {
    cmp_result tc_vs_ref;
    cmp_result gen_vs_tc;
    cmp_result y_ref_checksum;
    cmp_result y_tc_checksum;
    cmp_result y_gen_checksum;
    cmp_result y_h_checksum;
    cmp_result q8_ref_err;
    cmp_result q8_tc_err;
    cmp_result q8_tc_vs_ref;
    cmp_result tc16_vs_ref;
    cmp_result q8_h_err;
    cmp_result q8_h_vs_ref;
    cmp_result shared_vs_tc;
    cmp_result cub_vs_ref;
    cmp_result q8_tc_vs_f32; // quantized tensor-core result against the f32 rotation: the error the matmul sees
    cmp_result q8_h_vs_f32;
    std::vector<float> samples_ref, samples_tc;
};

static run_results compute_errors(buffers & b, int rows, float scale, cudaStream_t stream) {
    const long long n = (long long) rows * WIDTH;
    const dim3 grid_fwht((unsigned) rows * NBLK, 1, 1);

    fwht_kernel<false><<<grid_fwht, 256, 0, stream>>>(b.x, b.y_ref, nullptr, b.signs_f, rows, scale);
    fwht_kernel<true><<<grid_fwht, 256, 0, stream>>>(b.x, nullptr, b.q8_ref, b.signs_f, rows, scale);
    launch_tc<16, 128, 8, true, false>(b.x, b.basis, b.signs, b.y_tc, nullptr, rows, scale, stream);
    launch_tc<16, 128, 8, false, false>(b.x, b.basis, b.signs, b.y_gen, nullptr, rows, scale, stream);
    launch_tc<16, 128, 8, true, true>(b.x, b.basis, b.signs, nullptr, b.q8_tc, rows, scale, stream);
    launch_tc<16, 128, 8, true, false, true>(b.x, b.basis, b.signs, b.y_h, nullptr, rows, scale, stream);
    launch_tc<16, 128, 8, true, true, true>(b.x, b.basis, b.signs, nullptr, b.q8_h, rows, scale, stream);
    launch_tc<16, 128, 8, true, false, false, true>(b.x, b.hshared, b.signs, b.y_sh, nullptr, rows, scale, stream);
    launch_cublas(b.xs_bf16, b.h_bf16, b.y_cub, rows, scale);
    CU_CHECK(cudaStreamSynchronize(stream));

    run_results r;
    r.tc_vs_ref = compare_dev(b.y_tc, b.y_ref, n, b.scratch);
    r.tc16_vs_ref = compare_dev(b.y_h, b.y_ref, n, b.scratch);
    r.gen_vs_tc = compare_dev(b.y_gen, b.y_tc, n, b.scratch);
    r.shared_vs_tc = compare_dev(b.y_sh, b.y_tc, n, b.scratch);
    r.cub_vs_ref = compare_dev(b.y_cub, b.y_ref, n, b.scratch);
    r.y_ref_checksum = compare_dev(b.y_ref, b.y_ref, n, b.scratch);
    r.y_tc_checksum = compare_dev(b.y_tc, b.y_tc, n, b.scratch);
    r.y_gen_checksum = compare_dev(b.y_gen, b.y_gen, n, b.scratch);
    r.y_h_checksum = compare_dev(b.y_h, b.y_h, n, b.scratch);

    dequant_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, stream>>>(b.q8_ref, b.deq_ref, n);
    dequant_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, stream>>>(b.q8_tc, b.deq_tc, n);
    dequant_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, stream>>>(b.q8_h, b.deq_h, n);
    CU_CHECK(cudaStreamSynchronize(stream));
    r.q8_ref_err = compare_dev(b.deq_ref, b.y_ref, n, b.scratch);
    r.q8_tc_err = compare_dev(b.deq_tc, b.y_tc, n, b.scratch);
    r.q8_tc_vs_ref = compare_dev(b.deq_tc, b.deq_ref, n, b.scratch);
    r.q8_h_err = compare_dev(b.deq_h, b.y_h, n, b.scratch);
    r.q8_h_vs_ref = compare_dev(b.deq_h, b.deq_ref, n, b.scratch);
    r.q8_tc_vs_f32 = compare_dev(b.deq_tc, b.y_ref, n, b.scratch);
    r.q8_h_vs_f32 = compare_dev(b.deq_h, b.y_ref, n, b.scratch);

    for (const sample_pos & p : sample_positions(rows)) {
        r.samples_ref.push_back(sample_at(b.y_ref, p.row, p.col));
        r.samples_tc.push_back(sample_at(b.y_tc, p.row, p.col));
    }
    return r;
}

// ---------------------------------------------------------------------------------------------------------------
// CPU reference and bf16 simulation

struct replay_shape_results {
    double ref_sum = 0, ref_sumsq = 0, ref_maxabs = 0;
    double sim_sum = 0, sim_sumsq = 0, sim_maxabs = 0;
    double sim16_sum = 0, sim16_sumsq = 0, sim16_maxabs = 0;
    double rel_l2 = 0, max_abs_diff = 0;
    double q8_ref_rel = 0, q8_sim_rel = 0, q8_diff_rel = 0, q8_ratio = 0;
    double rel_l2_fp16 = 0, q8_fp16_rel = 0, q8_fp16_diff_rel = 0, q8_fp16_ratio = 0;
    double q8_sim_vs_f32 = 0, q8_fp16_vs_f32 = 0;
    std::vector<float> ref_samples, sim_samples;
};

static replay_shape_results replay_shape(int rows, uint32_t seed) {
    replay_shape_results r;
    const std::vector<sample_pos> pos = sample_positions(rows);
    r.ref_samples.assign(pos.size(), 0.0f);
    r.sim_samples.assign(pos.size(), 0.0f);

    std::vector<float> x(WIDTH), yr(WIDTH), ys(WIDTH), yh(WIDTH);
    std::vector<block_q8_1_s> qr(QROW / NBLK), qs(QROW / NBLK), qh(QROW / NBLK);
    std::vector<float> dr(WIDTH), ds(WIDTH), dh(WIDTH);
    const float scale = 1.0f / std::sqrt((float) BLOCK);

    double dsum = 0.0, qdsum = 0.0, refq = 0.0, simq = 0.0, drefq = 0.0;
    double dsum16 = 0.0, qdsum16 = 0.0, fp16q = 0.0, simt = 0.0, fp16t = 0.0;
    double maxd = 0.0, maxr = 0.0, maxs = 0.0;

    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < WIDTH; ++col) {
            x[col] = rot_input((uint32_t) (row * WIDTH + col), seed);
        }
        for (int b = 0; b < NBLK; ++b) {
            for (int i = 0; i < BLOCK; ++i) {
                const float xi = x[b * BLOCK + i];
                const float sg = (float) rot_sign((uint32_t) (b * BLOCK + i), seed);
                yr[i] = (xi * scale) * sg;
                ys[i] = bf16_round_host(xi) * sg;
                yh[i] = __half2float(__float2half_rn(xi)) * sg;
            }
            butterfly512(yr.data());
            butterfly512(ys.data());
            butterfly512(yh.data());
            for (int i = 0; i < BLOCK; ++i) {
                const int col = b * BLOCK + i;
                ys[i] *= scale;
                yh[i] *= scale;
                const double a = yr[i], c = ys[i], d = c - a;
                r.ref_sum += a; r.ref_sumsq += a * a; maxr = fmax(maxr, fabs(a));
                r.sim_sum += c; r.sim_sumsq += c * c; maxs = fmax(maxs, fabs(c));
                dsum += d * d; maxd = fmax(maxd, fabs(d));
                const double h = yh[i];
                r.sim16_sum += h; r.sim16_sumsq += h * h; r.sim16_maxabs = fmax(r.sim16_maxabs, fabs(h));
                const double d16 = h - a;
                dsum16 += d16 * d16;
                for (size_t p = 0; p < pos.size(); ++p) {
                    if (pos[p].row == row && pos[p].col == col) {
                        r.ref_samples[p] = yr[i];
                        r.sim_samples[p] = ys[i];
                    }
                }
            }
            for (int qb = 0; qb < QROW / NBLK; ++qb) {
                float amax_r = 0.0f, amax_s = 0.0f;
                for (int l = 0; l < QK8; ++l) {
                    amax_r = fmaxf(amax_r, fabsf(yr[qb * QK8 + l]));
                    amax_s = fmaxf(amax_s, fabsf(ys[qb * QK8 + l]));
                }
                const float dr_ = amax_r / 127.0f;
                const float ds_ = amax_s / 127.0f;
                int sum_r = 0, sum_s = 0;
                for (int l = 0; l < QK8; ++l) {
                    qr[qb].qs[l] = dr_ == 0.0f ? (int8_t) 0 : (int8_t) roundf(yr[qb * QK8 + l] / dr_);
                    qs[qb].qs[l] = ds_ == 0.0f ? (int8_t) 0 : (int8_t) roundf(ys[qb * QK8 + l] / ds_);
                    sum_r += (int) qr[qb].qs[l];
                    sum_s += (int) qs[qb].qs[l];
                }
                qr[qb].ds = __floats2half2_rn(dr_, dr_ * (float) sum_r);
                qs[qb].ds = __floats2half2_rn(ds_, ds_ * (float) sum_s);

                float amax_h = 0.0f;
                for (int l = 0; l < QK8; ++l) {
                    amax_h = fmaxf(amax_h, fabsf(yh[qb * QK8 + l]));
                }
                const float dh_ = amax_h / 127.0f;
                int sum_h = 0;
                for (int l = 0; l < QK8; ++l) {
                    qh[qb].qs[l] = dh_ == 0.0f ? (int8_t) 0 : (int8_t) roundf(yh[qb * QK8 + l] / dh_);
                    sum_h += (int) qh[qb].qs[l];
                }
                qh[qb].ds = __floats2half2_rn(dh_, dh_ * (float) sum_h);
            }
            for (int col = 0; col < BLOCK; ++col) {
                const int qb = col / QK8;
                dr[col] = (float) qr[qb].qs[col % QK8] * __half2float(qr[qb].ds.x);
                ds[col] = (float) qs[qb].qs[col % QK8] * __half2float(qs[qb].ds.x);
                dh[col] = (float) qh[qb].qs[col % QK8] * __half2float(qh[qb].ds.x);
                const double a = (double) dr[col] - (double) yr[col];
                const double c = (double) ds[col] - (double) ys[col];
                const double e = (double) ds[col] - (double) dr[col];
                const double f = (double) dh[col] - (double) yh[col];
                const double g = (double) dh[col] - (double) dr[col];
                refq += a * a;
                simq += c * c;
                qdsum += e * e;
                fp16q += f * f;
                qdsum16 += g * g;
                drefq += (double) dr[col] * (double) dr[col];
                const double t   = (double) ds[col] - (double) yr[col];
                const double t16 = (double) dh[col] - (double) yr[col];
                simt  += t * t;
                fp16t += t16 * t16;
            }
        }
    }

    r.ref_maxabs = maxr;
    r.sim_maxabs = maxs;
    r.max_abs_diff = maxd;
    r.rel_l2 = r.ref_sumsq > 0.0 ? std::sqrt(dsum / r.ref_sumsq) : 0.0;
    r.q8_ref_rel = r.ref_sumsq > 0.0 ? std::sqrt(refq / r.ref_sumsq) : 0.0;
    r.q8_sim_rel = r.sim_sumsq > 0.0 ? std::sqrt(simq / r.sim_sumsq) : 0.0;
    r.q8_diff_rel = drefq > 0.0 ? std::sqrt(qdsum / drefq) : 0.0;
    r.q8_ratio = r.q8_ref_rel > 0.0 ? r.q8_diff_rel / r.q8_ref_rel : 0.0;
    r.rel_l2_fp16 = r.ref_sumsq > 0.0 ? std::sqrt(dsum16 / r.ref_sumsq) : 0.0;
    r.q8_fp16_rel = r.sim16_sumsq > 0.0 ? std::sqrt(fp16q / r.sim16_sumsq) : 0.0;
    r.q8_fp16_diff_rel = drefq > 0.0 ? std::sqrt(qdsum16 / drefq) : 0.0;
    r.q8_fp16_ratio = r.q8_ref_rel > 0.0 ? r.q8_fp16_diff_rel / r.q8_ref_rel : 0.0;
    r.q8_sim_vs_f32 = r.ref_sumsq > 0.0 ? std::sqrt(simt / r.ref_sumsq) : 0.0;
    r.q8_fp16_vs_f32 = r.ref_sumsq > 0.0 ? std::sqrt(fp16t / r.ref_sumsq) : 0.0;
    return r;
}

// ---------------------------------------------------------------------------------------------------------------
// CLI

static std::vector<int> parse_rows(const char * s) {
    std::vector<int> rows;
    const char * p = s;
    while (*p) {
        char * end = nullptr;
        const long v = strtol(p, &end, 10);
        if (end == p || v <= 0) {
            fprintf(stderr, "bad --rows\n");
            exit(64);
        }
        rows.push_back((int) v);
        p = end;
        if (*p == ',') {
            ++p;
        }
    }
    return rows;
}

static uint32_t parse_seed(const char * s) {
    return (uint32_t) strtoul(s, nullptr, 0);
}

static void print_device_json(const device_info & di) {
    printf("{\"name\":\"%s\",\"cc\":\"%d.%d\",\"sm_count\":%d,\"l2_bytes\":%d,"
           "\"mem_clock_khz\":%d,\"mem_bus_bits\":%d,\"clock_khz\":%d,"
           "\"driver\":\"%s\",\"runtime\":\"%s\"}",
           di.name, di.cc_major, di.cc_minor, di.sm_count, di.l2_bytes,
           di.mem_clock_khz, di.mem_bus_bits, di.clock_khz, di.driver, di.runtime);
}

static void print_time_stats(const time_stats & t) {
    printf("{\"samples_us\":[");
    for (size_t i = 0; i < t.samples_us.size(); ++i) {
        printf("%s%.9g", i ? "," : "", t.samples_us[i]);
    }
    printf("],\"median_us\":%.9g,\"min_us\":%.9g,\"p10_us\":%.9g,\"p90_us\":%.9g}",
           t.median_us, t.min_us, t.p10_us, t.p90_us);
}

static void print_samples(const float * v, size_t n) {
    printf("[");
    for (size_t i = 0; i < n; ++i) {
        printf("%s%.9g", i ? "," : "", v[i]);
    }
    printf("]");
}

static void print_shape_metrics(const run_results & rr) {
    printf("\"errors\":{");
    printf("\"tc_vs_f32\":{\"rel_l2\":%.17g,\"max_abs\":%.17g,\"ref_maxabs\":%.17g},",
           rel_l2(rr.tc_vs_ref), rr.tc_vs_ref.maxabs_d, rr.tc_vs_ref.maxabs_b);
    printf("\"tc16_vs_f32\":{\"rel_l2\":%.17g,\"max_abs\":%.17g},",
           rel_l2(rr.tc16_vs_ref), rr.tc16_vs_ref.maxabs_d);
    printf("\"gen_vs_tc_max_abs\":%.17g,", rr.gen_vs_tc.maxabs_d);
    printf("\"shared_vs_tc_max_abs\":%.17g,", rr.shared_vs_tc.maxabs_d);
    printf("\"cublas_vs_f32\":{\"rel_l2\":%.17g,\"max_abs\":%.17g},", rel_l2(rr.cub_vs_ref), rr.cub_vs_ref.maxabs_d);
    printf("\"q8\":{\"ref_rel\":%.17g,\"tc_rel\":%.17g,\"tc_vs_ref\":%.17g,\"ratio\":%.17g,\"tc_vs_f32\":%.17g},",
           rel_l2(rr.q8_ref_err), rel_l2(rr.q8_tc_err), rel_l2(rr.q8_tc_vs_ref),
           rel_l2(rr.q8_ref_err) > 0 ? rel_l2(rr.q8_tc_vs_ref) / rel_l2(rr.q8_ref_err) : 0.0,
           rel_l2(rr.q8_tc_vs_f32));
    printf("\"q8_fp16\":{\"tc_rel\":%.17g,\"tc_vs_ref\":%.17g,\"ratio\":%.17g,\"tc_vs_f32\":%.17g},",
           rel_l2(rr.q8_h_err), rel_l2(rr.q8_h_vs_ref),
           rel_l2(rr.q8_ref_err) > 0 ? rel_l2(rr.q8_h_vs_ref) / rel_l2(rr.q8_ref_err) : 0.0,
           rel_l2(rr.q8_h_vs_f32));
    printf("\"checksums\":{");
    printf("\"y_ref\":{\"sum\":%.17g,\"sumsq\":%.17g,\"maxabs\":%.17g},",
           rr.y_ref_checksum.sum_a, rr.y_ref_checksum.sumsq_a, rr.y_ref_checksum.maxabs_a);
    printf("\"y_tc\":{\"sum\":%.17g,\"sumsq\":%.17g,\"maxabs\":%.17g},",
           rr.y_tc_checksum.sum_a, rr.y_tc_checksum.sumsq_a, rr.y_tc_checksum.maxabs_a);
    printf("\"y_gen\":{\"sum\":%.17g,\"sumsq\":%.17g,\"maxabs\":%.17g},",
           rr.y_gen_checksum.sum_a, rr.y_gen_checksum.sumsq_a, rr.y_gen_checksum.maxabs_a);
    printf("\"y_h\":{\"sum\":%.17g,\"sumsq\":%.17g,\"maxabs\":%.17g},",
           rr.y_h_checksum.sum_a, rr.y_h_checksum.sumsq_a, rr.y_h_checksum.maxabs_a);
    printf("\"q8_ref\":{\"sum\":%.17g,\"sumsq\":%.17g,\"maxabs\":%.17g}},",
           rr.q8_ref_err.sum_a, rr.q8_ref_err.sumsq_a, rr.q8_ref_err.maxabs_a);
    printf("\"samples\":{\"y_ref\":");
    print_samples(rr.samples_ref.data(), rr.samples_ref.size());
    printf(",\"y_tc\":");
    print_samples(rr.samples_tc.data(), rr.samples_tc.size());
    printf("}}");
}

int main(int argc, char ** argv) {
    const char * mode = nullptr;
    std::vector<int> rows;
    uint32_t seed = 0x5eed1234u;

    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--device")) {
            mode = "device";
        } else if (!strcmp(argv[i], "--measure")) {
            mode = "measure";
        } else if (!strcmp(argv[i], "--selftest")) {
            mode = "selftest";
        } else if (!strcmp(argv[i], "--replay")) {
            mode = "replay";
        } else if (!strcmp(argv[i], "--rows") && i + 1 < argc) {
            rows = parse_rows(argv[++i]);
        } else if (!strcmp(argv[i], "--seed") && i + 1 < argc) {
            seed = parse_seed(argv[++i]);
        } else {
            fprintf(stderr, "unknown argument %s\n", argv[i]);
            return 64;
        }
    }
    if (!mode) {
        fprintf(stderr, "one of --device, --measure, --selftest, --replay is required\n");
        return 64;
    }

    if (!strcmp(mode, "replay")) {
        if (rows.empty()) {
            rows = {23};
        }
        printf("{\"seed\":%u,\"shapes\":[", seed);
        for (size_t si = 0; si < rows.size(); ++si) {
            const replay_shape_results r = replay_shape(rows[si], seed);
            printf("%s{\"rows\":%d,", si ? "," : "", rows[si]);
            printf("\"ref\":{\"sum\":%.17g,\"sumsq\":%.17g,\"maxabs\":%.17g,\"samples\":",
                   r.ref_sum, r.ref_sumsq, r.ref_maxabs);
            print_samples(r.ref_samples.data(), r.ref_samples.size());
            printf("},\"sim\":{\"sum\":%.17g,\"sumsq\":%.17g,\"maxabs\":%.17g,\"samples\":",
                   r.sim_sum, r.sim_sumsq, r.sim_maxabs);
            print_samples(r.sim_samples.data(), r.sim_samples.size());
            printf("},\"sim16\":{\"sum\":%.17g,\"sumsq\":%.17g,\"maxabs\":%.17g},",
                   r.sim16_sum, r.sim16_sumsq, r.sim16_maxabs);
            printf("\"tc_vs_ref\":{\"rel_l2\":%.17g,\"max_abs\":%.17g},", r.rel_l2, r.max_abs_diff);
            printf("\"q8\":{\"ref_rel\":%.17g,\"sim_rel\":%.17g,\"diff_rel\":%.17g,\"ratio\":%.17g,\"sim_vs_f32\":%.17g},",
                   r.q8_ref_rel, r.q8_sim_rel, r.q8_diff_rel, r.q8_ratio, r.q8_sim_vs_f32);
            printf("\"fp16\":{\"rel_l2\":%.17g,\"q8_rel\":%.17g,\"diff_rel\":%.17g,\"ratio\":%.17g,\"vs_f32\":%.17g}}",
                   r.rel_l2_fp16, r.q8_fp16_rel, r.q8_fp16_diff_rel, r.q8_fp16_ratio, r.q8_fp16_vs_f32);
        }
        printf("]}\n");
        return 0;
    }

    CU_CHECK(cudaSetDevice(0));
    const device_info di = get_device_info();
    if (!strcmp(mode, "device")) {
        print_device_json(di);
        printf("\n");
        return 0;
    }

    if (rows.empty()) {
        rows = {23, 128, 512, 2048};
    }
    const int max_rows = *std::max_element(rows.begin(), rows.end());
    const float scale = 1.0f / std::sqrt((float) BLOCK);

    cudaStream_t stream;
    CU_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    cudaEvent_t e0, e1;
    CU_CHECK(cudaEventCreate(&e0));
    CU_CHECK(cudaEventCreate(&e1));

    buffers b;
    alloc_buffers(b, max_rows);
    init_data(b, max_rows, seed, stream);
    CUBLAS_CHECK(cublasCreate(&g_cublas));
    CUBLAS_CHECK(cublasSetStream(g_cublas, stream));
    // a workspace of its own, so a captured GEMM never allocates
    CUBLAS_CHECK(cublasSetWorkspace(g_cublas, b.cublas_ws, CUBLAS_WS_BYTES));

    printf("{\"seed\":%u,\"device\":", seed);
    print_device_json(di);
    printf(",\"shapes\":[");

    for (size_t si = 0; si < rows.size(); ++si) {
        const int nrows = rows[si];
        if (!strcmp(mode, "measure")) {
            std::vector<arm> arms;
            build_arms(arms, b, nrows, scale, stream);
            time_arms(arms, stream, e0, e1);

            const run_results rr = compute_errors(b, nrows, scale, stream);

            printf("%s{\"rows\":%d,\"arms\":{", si ? "," : "", nrows);
            bool first = true;
            for (const arm & a : arms) {
                printf("%s\"%s\":", first ? "" : ",", a.name.c_str());
                print_time_stats(a.st);
                first = false;
            }
            printf("},");
            print_shape_metrics(rr);
            printf("}");
        } else {
            const run_results rr = compute_errors(b, nrows, scale, stream);
            printf("%s{\"rows\":%d,", si ? "," : "", nrows);
            print_shape_metrics(rr);
            printf("}");
        }
    }
    printf("]}\n");

    CUBLAS_CHECK(cublasDestroy(g_cublas));
    free_buffers(b);
    CU_CHECK(cudaEventDestroy(e0));
    CU_CHECK(cudaEventDestroy(e1));
    CU_CHECK(cudaStreamDestroy(stream));
    return 0;
}
