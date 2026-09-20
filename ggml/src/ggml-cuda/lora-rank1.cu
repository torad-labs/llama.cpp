#include "lora-rank1.cuh"

// A rank-1 adapter (the abliteration LoRAs are rank 1) costs four launches per adapted weight
// through build_lora_mm: a K-wide dot (cuBLAS gemm k=1), an N-wide outer product with the scalar,
// a scale, and the add into the base projection. At decode each is ~1-3 us of latency and the
// model adapts 128 weights, so the four launches are ~0.75 ms of a ~13 ms token. This kernel does
// all four in one launch: every block recomputes the dot (a and x are L2-resident after the first
// block reads them, and K is at most a few tens of thousands), so no grid-wide sync is needed.
#define LORA_RANK1_BLOCK 256

static __global__ void lora_rank1_fused_f32(
        const float * __restrict__ a, const float * __restrict__ x, const float * __restrict__ b,
        const float * __restrict__ res, float * __restrict__ dst,
        const int K, const int N, const float scale) {
    ggml_cuda_pdl_lc();
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    __shared__ float red[LORA_RANK1_BLOCK / warp_size];

    ggml_cuda_pdl_sync();
    float acc = 0.0f;
    for (int k = threadIdx.x; k < K; k += LORA_RANK1_BLOCK) {
        acc += a[k] * x[k];
    }
    acc = warp_reduce_sum(acc);
    if (threadIdx.x % warp_size == 0) {
        red[threadIdx.x / warp_size] = acc;
    }
    __syncthreads();
    float s = 0.0f;
#pragma unroll
    for (int w = 0; w < LORA_RANK1_BLOCK / warp_size; ++w) {
        s += red[w];
    }
    s *= scale;

    const int n = blockIdx.x * LORA_RANK1_BLOCK + threadIdx.x;
    if (n < N) {
        dst[n] = res[n] + s * b[n];
    }
}

bool ggml_cuda_op_lora_rank1_fused(ggml_backend_cuda_context & ctx,
                                   const ggml_tensor * a, const ggml_tensor * x, const ggml_tensor * b,
                                   float scale, const ggml_tensor * res, ggml_tensor * dst) {
    const int64_t K = a->ne[0];
    const int64_t N = b->ne[1];

    if (a->type != GGML_TYPE_F32 || x->type != GGML_TYPE_F32 || b->type != GGML_TYPE_F32 ||
        res->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    // rank 1, one token: a [K,1], x [K,1], b [1,N], res/dst [N,1]
    if (a->ne[1] != 1 || ggml_nrows(a) != 1 || ggml_nrows(x) != 1 || x->ne[0] != K ||
        b->ne[0] != 1 || b->ne[2] != 1 || b->ne[3] != 1 ||
        res->ne[0] != N || ggml_nrows(res) != 1 || !ggml_are_same_shape(res, dst)) {
        return false;
    }
    if (!ggml_is_contiguous(a) || !ggml_is_contiguous(x) || !ggml_is_contiguous(b) ||
        !ggml_is_contiguous(res) || !ggml_is_contiguous(dst)) {
        return false;
    }
    if (K > INT32_MAX || N > INT32_MAX) {
        return false;
    }

    const dim3 grid((N + LORA_RANK1_BLOCK - 1) / LORA_RANK1_BLOCK, 1, 1);
    const dim3 block(LORA_RANK1_BLOCK, 1, 1);
    const ggml_cuda_kernel_launch_params lp = ggml_cuda_kernel_launch_params(grid, block, 0, ctx.stream());
    ggml_cuda_kernel_launch(lora_rank1_fused_f32, lp,
        (const float *) a->data, (const float *) x->data, (const float *) b->data,
        (const float *) res->data, (float *) dst->data, (int) K, (int) N, scale);
    return true;
}
