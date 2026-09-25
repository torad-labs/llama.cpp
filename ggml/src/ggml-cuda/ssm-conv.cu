#include "common.cuh"
#include "ssm-conv.cuh"
#include "unary.cuh"

template <bool apply_silu, size_t split_d_inner, size_t d_conv>
static __global__ void ssm_conv_f32(const float * src0_ptr, const float * src1_ptr,
                                    const float * bias_ptr,
                                    const int src0_nb0, const int src0_nb1, const int src0_nb2, const int src1_nb1,
                                    float * dst_ptr, const int dst_nb0, const int dst_nb1, const int dst_nb2,
                                    const int64_t n_t) {
    ggml_cuda_pdl_lc();
    const float * GGML_CUDA_RESTRICT src0 = src0_ptr;
    const float * GGML_CUDA_RESTRICT src1 = src1_ptr;
    const float * GGML_CUDA_RESTRICT bias = bias_ptr;
    float       * GGML_CUDA_RESTRICT dst  = dst_ptr;
    GGML_UNUSED(src0_nb0);
    const int tid  = threadIdx.x;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;

    const float * x_block = (const float *) ((const char *) src0 + bidx * src0_nb2 + bidy * split_d_inner * src0_nb1);
    const float * w_block = (const float *) ((const char *) src1 + bidy * split_d_inner * src1_nb1);
    float *       y_block = (float *) ((char *) dst + bidx * dst_nb2 + bidy * split_d_inner * dst_nb0);

    const int stride_x = src0_nb1 / sizeof(float);
    const int stride_w = src1_nb1 / sizeof(float);
    const int stride_y = dst_nb1 / sizeof(float);

    float x[d_conv] = { 0.0f };
    float w[d_conv] = { 0.0f };

    ggml_cuda_pdl_sync();
#pragma unroll
    for (size_t j = 0; j < d_conv; j++) {
        w[j] = w_block[tid * stride_w + j];
    }

    float b = bias != nullptr ? bias[bidy * split_d_inner + tid] : 0.0f;

    for (int64_t i = 0; i < n_t; i++) {
        float sumf = 0.0f;

        if (i == 0) {
            for (size_t j = 0; j < d_conv; j++) {
                x[j] = x_block[tid * stride_x + j];
            }
        } else {
            x[(i - 1) % d_conv] = x_block[tid * stride_x + i + d_conv - 1];
        }

#pragma unroll
        for (size_t j = 0; j < d_conv; j++) {
            sumf += x[(i + j) % d_conv] * w[j];
        }
        sumf += b;
        y_block[i * stride_y + tid] = apply_silu ? ggml_cuda_op_silu_single(sumf) : sumf;
    }
}

template <bool apply_silu, size_t split_d_inner, size_t d_conv, int64_t split_n_t>
static __global__ void ssm_conv_long_token_f32(const float * __restrict__ src0, const float * __restrict__ src1,
                                               const float * __restrict__ bias,
                                               const int src0_nb0, const int src0_nb1, const int src0_nb2,
                                               const int src1_nb1, float * __restrict__ dst, const int dst_nb0,
                                               const int dst_nb1, const int dst_nb2, const int64_t n_t) {
    const int tid  = threadIdx.x;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;
    const int bidz = blockIdx.z;

    const float * x_block = (const float *) ((const char *) src0 + bidx * src0_nb2 + bidy * split_d_inner * src0_nb1 +
                                             bidz * split_n_t * src0_nb0);
    const float * w_block = (const float *) ((const char *) src1 + bidy * split_d_inner * src1_nb1);
    float *       y_block =
        (float *) ((char *) dst + bidx * dst_nb2 + bidz * split_n_t * dst_nb1 + bidy * split_d_inner * dst_nb0);

    const int stride_x = src0_nb1 / sizeof(float);
    const int stride_w = src1_nb1 / sizeof(float);
    const int stride_y = dst_nb1 / sizeof(float);

    const int64_t local_n_t = min(split_n_t, n_t - bidz * split_n_t);
    const int     n_cols    = d_conv - 1 + split_n_t;

    extern __shared__ float smem[];

    constexpr int load_cols   = d_conv - 1 + split_n_t;
    constexpr int total_elems = split_d_inner * load_cols;
    int row = tid / load_cols;
    int col = tid % load_cols;
#pragma unroll
    for (int idx = 0; idx < total_elems; idx += split_d_inner) {
        if (row < (int)split_d_inner) {
            smem[row * n_cols + col] = x_block[row * stride_x + col];
        }

        col += split_d_inner;
        row += col / load_cols;
        col  = col % load_cols;
        if (idx >= total_elems - tid - split_d_inner) {
            break;
        }
    }
    __syncthreads();

    // Load weights into registers (done once, small)
    float w[d_conv] = { 0.0f };
#pragma unroll
    for (size_t j = 0; j < d_conv; j++) {
        w[j] = w_block[tid * stride_w + j];
    }

    float b = bias != nullptr ? bias[bidy * split_d_inner + tid] : 0.0f;

    // Compute from shared memory
    for (int64_t i = 0; i < local_n_t; i++) {
        float sumf = 0.0f;
#pragma unroll
        for (size_t j = 0; j < d_conv; j++) {
            sumf += smem[tid * n_cols + i + j] * w[j];
        }
        sumf += b;
        y_block[i * stride_y + tid] = apply_silu ? ggml_cuda_op_silu_single(sumf) : sumf;
    }
}

// The fused conv-state update (ggml_cuda_ssm_conv_state_update), one sequence: a thread per channel reads its
// d_conv - 1 state columns out of the cache row ids[0] and its n_t new inputs out of the projection, writes its n_t
// outputs, then its window of every snapshot. All its reads come before any of its writes, and a thread touches only
// its own channel in every row and every column, so a snapshot that overwrites the row being read, and an output the
// allocator placed exactly on the inputs (the matcher declines any other overlap), read the old values. A block is one
// head of GGML_CUDA_SSM_CONV_UPDATE_THREADS channels, so the blocks of the leading u.l2_heads heads also write the L2_NORM
// of their outputs (u.l2_dst) with l2_norm_f32's arithmetic: one warp sums the squares lane by lane in its column order,
// then the same warp reduction, rsqrtf and product, so the values match the L2_NORM's bit for bit.
template <bool apply_silu, int d_conv, int max_n_t>
static __global__ void ssm_conv_state_update_f32(const ggml_cuda_ssm_conv_state_update u, const float * w,
                                                 const int w_stride, float * dst, const int dst_stride, const int n_t) {
    ggml_cuda_pdl_lc();
    const int c = blockIdx.x * blockDim.x + threadIdx.x;

    float x[d_conv - 1 + max_n_t];
    float wc[d_conv];

    ggml_cuda_pdl_sync();
    const float * state = u.cache + (int64_t) u.ids[0] * u.row_stride + (int64_t) c * (d_conv - 1);
#pragma unroll
    for (int j = 0; j < d_conv - 1; ++j) {
        x[j] = state[j];
    }
#pragma unroll
    for (int t = 0; t < max_n_t; ++t) {
        if (t < n_t) {
            x[d_conv - 1 + t] = u.x[t * u.x_stride + c];
        }
    }
#pragma unroll
    for (int j = 0; j < d_conv; ++j) {
        wc[j] = w[c * w_stride + j];
    }

    // the same sum as ssm_conv_f32, including its zero bias (it turns a -0.0f sum into +0.0f)
    const float b = 0.0f;
    float       y[max_n_t];
#pragma unroll
    for (int t = 0; t < max_n_t; ++t) {
        if (t < n_t) {
            float sumf = 0.0f;
#pragma unroll
            for (int j = 0; j < d_conv; ++j) {
                sumf += x[t + j] * wc[j];
            }
            sumf += b;
            y[t] = apply_silu ? ggml_cuda_op_silu_single(sumf) : sumf;
            dst[t * dst_stride + c] = y[t];
        }
    }

    if ((int) blockIdx.x < u.l2_heads) { // uniform across the block
        constexpr int width = GGML_CUDA_SSM_CONV_UPDATE_THREADS;
        __shared__ float row[width];
        __shared__ float row_scale;
#pragma unroll
        for (int t = 0; t < max_n_t; ++t) {
            if (t < n_t) {
                row[threadIdx.x] = y[t];
                __syncthreads();
                if (threadIdx.x < WARP_SIZE) {
                    float tmp = 0.0f;
                    for (int col = threadIdx.x; col < width; col += WARP_SIZE) {
                        const float xi = row[col];
                        tmp += xi * xi;
                    }
                    tmp = warp_reduce_sum(tmp);
                    if (threadIdx.x == 0) {
                        row_scale = rsqrtf(fmaxf(tmp, u.l2_eps * u.l2_eps));
                    }
                }
                __syncthreads();
                u.l2_dst[((int64_t) t * u.l2_heads + blockIdx.x) * width + threadIdx.x] = row_scale * y[t];
                __syncthreads();
            }
        }
    }

    for (int k = 0; k < u.n_snapshots; ++k) {
        float *   out = u.snapshot_dst[k] + (int64_t) c * (d_conv - 1);
        const int col = u.snapshot_col[k];
        // a compile-time index per candidate column keeps x in registers
#pragma unroll
        for (int c0 = 0; c0 <= max_n_t; ++c0) {
            if (c0 == col) {
#pragma unroll
                for (int j = 0; j < d_conv - 1; ++j) {
                    out[j] = x[c0 + j];
                }
            }
        }
    }
}

template <bool apply_silu>
static void ssm_conv_f32_cuda(const float * src0, const float * src1, const float * bias, const int src0_nb0, const int src0_nb1,
                              const int src0_nb2, const int src1_nb1, float * dst, const int dst_nb0, const int dst_nb1,
                              const int dst_nb2, const int64_t nc, const int64_t nr, const int64_t n_t,
                              const int64_t n_s, cudaStream_t stream) {
    const int threads = 128;
    GGML_ASSERT(nr % threads == 0);

    auto launch_kernel = [&](auto NC) {
        constexpr int kNC = decltype(NC)::value;
        if (n_t <= 32) {
            const dim3 blocks(n_s, (nr + threads - 1) / threads, 1);
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(blocks, threads, 0, stream);
            ggml_cuda_kernel_launch(ssm_conv_f32<apply_silu, threads, kNC>, launch_params, src0, src1, bias, src0_nb0, src0_nb1,
                                                                        src0_nb2, src1_nb1, dst, dst_nb0, dst_nb1, dst_nb2, n_t);
        } else {
            const int64_t split_n_t = 32;
            dim3          blocks(n_s, (nr + threads - 1) / threads, (n_t + split_n_t - 1) / split_n_t);
            const size_t  smem_size = threads * (kNC - 1 + split_n_t) * sizeof(float);
            ssm_conv_long_token_f32<apply_silu, threads, kNC, split_n_t><<<blocks, threads, smem_size, stream>>>(
                src0, src1, bias, src0_nb0, src0_nb1, src0_nb2, src1_nb1, dst, dst_nb0, dst_nb1, dst_nb2, n_t);
        }
    };

    switch (nc) {
        case 3:  launch_kernel(std::integral_constant<int, 3 >{}); break;
        case 4:  launch_kernel(std::integral_constant<int, 4 >{}); break;
        case 5:  launch_kernel(std::integral_constant<int, 5 >{}); break;
        case 9:  launch_kernel(std::integral_constant<int, 9 >{}); break;
        case 15: launch_kernel(std::integral_constant<int, 15>{}); break;
        default: GGML_ABORT("Only support kernel sizes 3, 4, 5, 9, 15 right now.");
    }
}

void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * bias_add_node, ggml_tensor * silu_dst) {
    const struct ggml_tensor * src0 = dst->src[0];  // conv_x
    const struct ggml_tensor * src1 = dst->src[1];  // conv1d.weight
    const bool fuse_bias = bias_add_node != nullptr;
    const bool fuse_silu = silu_dst != nullptr;

    // bias always comes with silu.
    GGML_ASSERT(!fuse_bias || fuse_silu);

    // The bias (when fused) is the non-conv operand of the ADD node.
    const struct ggml_tensor * bias = fuse_bias ? (bias_add_node->src[0] == dst ? bias_add_node->src[1] : bias_add_node->src[0]) : nullptr;

    // When fusing, write to silu_dst (the node downstream references).
    const struct ggml_tensor * out = fuse_silu ? silu_dst : dst;

    const int64_t nc  = src1->ne[0];                // d_conv
    const int64_t nr  = src0->ne[1];                // d_inner
    const int64_t n_t = out->ne[1];                 // tokens per sequence
    const int64_t n_s = out->ne[2];                 // number of sequences in the batch

    GGML_ASSERT(out->ne[0] == nr);
    GGML_ASSERT(src0->nb[0] == sizeof(float));
    GGML_ASSERT(src1->nb[0] == sizeof(float));
    GGML_ASSERT(src0->nb[1] == src0->ne[0] * sizeof(float));

    const float * src0_d = (const float *) src0->data;
    const float * src1_d = (const float *) src1->data;
    const float * bias_d = fuse_bias ? (const float *) bias->data : nullptr;
    float *       dst_d  = (float *) out->data;
    cudaStream_t  stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(out->type == GGML_TYPE_F32);
    if (fuse_bias) {
        GGML_ASSERT(bias->type == GGML_TYPE_F32);
        GGML_ASSERT(ggml_is_contiguous(bias));
        GGML_ASSERT(ggml_nelements(bias) == nr);
    }

    // the chain ggml_cuda_try_ssm_conv_state_update matched: src0 (the CONCAT) was never written, the kernel builds it
    if (const ggml_cuda_ssm_conv_state_update * u = ctx.ssm_conv_updates().find(dst)) {
        constexpr int threads = GGML_CUDA_SSM_CONV_UPDATE_THREADS;
        GGML_ASSERT(!fuse_bias && nc == GGML_CUDA_SSM_CONV_UPDATE_D_CONV && n_s == 1 && nr % threads == 0 &&
                    n_t <= GGML_CUDA_SSM_CONV_UPDATE_MAX_N_T);
        const ggml_cuda_kernel_launch_params launch_params(dim3(nr / threads), dim3(threads), 0, stream);
        const int w_stride   = src1->nb[1] / sizeof(float);
        const int dst_stride = out->nb[1] / sizeof(float);
        if (fuse_silu) {
            // the L2_NORM normalizes the SILU's output: run it here only with the SILU fused, and only then skip it
            if (const ggml_tensor * l2 = ctx.ssm_conv_updates().l2_norm_of(dst)) {
                ctx.ssm_conv_updates().skipped.insert(l2);
            }
            ggml_cuda_kernel_launch(ssm_conv_state_update_f32<true, GGML_CUDA_SSM_CONV_UPDATE_D_CONV, GGML_CUDA_SSM_CONV_UPDATE_MAX_N_T>,
                                    launch_params, *u, src1_d, w_stride, dst_d, dst_stride, (int) n_t);
        } else {
            ggml_cuda_ssm_conv_state_update raw = *u;
            raw.l2_heads = 0;
            ggml_cuda_kernel_launch(ssm_conv_state_update_f32<false, GGML_CUDA_SSM_CONV_UPDATE_D_CONV, GGML_CUDA_SSM_CONV_UPDATE_MAX_N_T>,
                                    launch_params, raw, src1_d, w_stride, dst_d, dst_stride, (int) n_t);
        }
        return;
    }

    if (fuse_silu) {
        ssm_conv_f32_cuda<true>(src0_d, src1_d, bias_d, src0->nb[0], src0->nb[1], src0->nb[2], src1->nb[1], dst_d, out->nb[0], out->nb[1],
                          out->nb[2], nc, nr, n_t, n_s, stream);
    } else {
        ssm_conv_f32_cuda<false>(src0_d, src1_d, bias_d, src0->nb[0], src0->nb[1], src0->nb[2], src1->nb[1], dst_d, out->nb[0], out->nb[1],
                          out->nb[2], nc, nr, n_t, n_s, stream);
    }
}
