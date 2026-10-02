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

// Qwen3.5's alpha/beta pair folded into the conv-state update (u.ab_rows > 0, ggml_cuda_try_ssm_conv_ab): the block takes
// rows blockIdx.x, blockIdx.x + gridDim.x, ... of the two matrices laid end to end, at most
// GGML_CUDA_SSM_CONV_AB_MAX_ROWS, and computes each as mul_mat_vec_f does at its 256 threads, a thread here being two of
// its threads (tid and tid + 128): each one's column pairs strided by 256 in order, one warp reduction for each, then
// warp 0 over the eight partials and 24 zeros, so each value is the pair launch's bit for bit. The results wait in
// ab_res for the writes after the dependency wait. Everything here is read before that wait: the weights, which no
// kernel writes, and the activation, once its writer's release reaches the block (ssm_conv_ab_acquire), through L2
// (ld.global.cg), where that kernel's writes are, not through this SM's L1.
// The fold's handoff (ggml_cuda_ssm_conv_ab_slots): thread 0 acquires the slot until every one of the activation's
// writer's blocks has released to it, the barrier orders the block's reads after that, and thread 0 takes a ticket; the
// block with the grid's last sets the slot back to 0, after every block's acquire. The wait costs one load: the group
// launch writing this conv's inputs, which comes after the writer, lets the kernels after it launch only after its own
// wait (ggml_cuda_ssm_conv_ab_enabled), when the writer has ended. A slot that never fills would be a planning fault:
// the block traps.
static __device__ __forceinline__ void ssm_conv_ab_acquire(const ggml_cuda_ssm_conv_state_update & u) {
    if (threadIdx.x == 0) {
        for (int i = 0; (ggml_cuda_ld_acquire(u.ab_slot) & 0xffffu) < (unsigned int) u.ab_writers; ++i) {
            if (i == 1 << 22) {
                __trap();
            }
        }
        if (atomicAdd(u.ab_slot, 1u << 16) >> 16 == gridDim.x - 1) {
            atomicExch(u.ab_slot, 0u);
        }
    }
    __syncthreads();
}

template <int max_n_t>
static __device__ __forceinline__ int ssm_conv_ab_rows(const ggml_cuda_ssm_conv_state_update & u, const int n_t,
                                                       float (*ab_res)[max_n_t]) {
    constexpr int block = 256; // mul_mat_vec_f's block for the pair's rows (launch_mul_mat_vec_f_cuda: 5,120 columns)
    constexpr int half  = GGML_CUDA_SSM_CONV_UPDATE_THREADS;
    static_assert(2*half == block, "a thread here is two of mul_mat_vec_f's");
    __shared__ float part[block/WARP_SIZE][max_n_t];

    const int warp   = threadIdx.x / WARP_SIZE;
    const int lane   = threadIdx.x % WARP_SIZE;
    const int ncols2 = u.ab_ncols / 2;
    const int rows   = 2*u.ab_rows;

    // the block's rows into L2 first: their loads below then wait on L2, not DRAM
    if (threadIdx.x == 0) {
        for (int r = blockIdx.x; r < rows; r += gridDim.x) {
            const nv_bfloat16 * x = u.ab_w + (r / u.ab_rows)*u.ab_s02 + (int64_t) (r % u.ab_rows)*u.ab_stride_row;
            ggml_cuda_prefetch_l2(x, (int64_t) u.ab_ncols*sizeof(nv_bfloat16));
        }
    }
    ssm_conv_ab_acquire(u);

    int n = 0;
    for (int r = blockIdx.x; r < rows; r += gridDim.x, ++n) {
        const nv_bfloat162 * x2 = (const nv_bfloat162 *) (u.ab_w + (r / u.ab_rows)*u.ab_s02 +
                                                          (int64_t) (r % u.ab_rows)*u.ab_stride_row);
        const float2 * y2 = (const float2 *) u.ab_y;

        float lo[max_n_t]; // mul_mat_vec_f's thread tid
        float hi[max_n_t]; // and its thread tid + 128
#pragma unroll
        for (int j = 0; j < max_n_t; ++j) {
            lo[j] = 0.0f;
            hi[j] = 0.0f;
        }
#pragma unroll 5
        for (int col2 = threadIdx.x; col2 < ncols2; col2 += block) {
            const nv_bfloat162 tmpx = x2[col2];
#pragma unroll
            for (int j = 0; j < max_n_t; ++j) {
                if (j < n_t) {
                    const float2 tmpy = __ldcg(y2 + j*(u.ab_stride_y/2) + col2);
                    ggml_cuda_mad(lo[j], tmpx.x, tmpy.x);
                    ggml_cuda_mad(lo[j], tmpx.y, tmpy.y);
                }
            }
            if (col2 + half < ncols2) {
                const nv_bfloat162 tmpx_hi = x2[col2 + half];
#pragma unroll
                for (int j = 0; j < max_n_t; ++j) {
                    if (j < n_t) {
                        const float2 tmpy = __ldcg(y2 + j*(u.ab_stride_y/2) + col2 + half);
                        ggml_cuda_mad(hi[j], tmpx_hi.x, tmpy.x);
                        ggml_cuda_mad(hi[j], tmpx_hi.y, tmpy.y);
                    }
                }
            }
        }
#pragma unroll
        for (int j = 0; j < max_n_t; ++j) {
            if (j < n_t) {
                lo[j] = warp_reduce_sum<WARP_SIZE>(lo[j]);
                hi[j] = warp_reduce_sum<WARP_SIZE>(hi[j]);
                if (lane == 0) {
                    part[warp][j]                  = lo[j];
                    part[warp + half/WARP_SIZE][j] = hi[j];
                }
            }
        }
        __syncthreads();
        if (warp == 0) {
#pragma unroll
            for (int j = 0; j < max_n_t; ++j) {
                if (j < n_t) {
                    float v = lane < block/WARP_SIZE ? part[lane][j] : 0.0f;
                    v = warp_reduce_sum<WARP_SIZE>(v);
                    if (lane == 0) {
                        ab_res[n][j] = v;
                    }
                }
            }
        }
        __syncthreads();
    }
    return n;
}

// The weights of channel c: its row of the SSM_CONV's src1, or of the one of the three tensors src1 is a concatenation of
// that holds it (ggml_cuda_ssm_conv_state_update::w_seg)
static __device__ __forceinline__ const float * ssm_conv_w_row(const ggml_cuda_ssm_conv_state_update & u, const float * w,
                                                              const int w_stride, const int c) {
    if (u.w_seg_channels == 0) {
        return w + c * w_stride;
    }
    const int s = c / u.w_seg_channels;
    return u.w_seg[s] + (c - s * u.w_seg_channels) * w_stride;
}

// The fused conv-state update (ggml_cuda_ssm_conv_state_update), one sequence: a thread per channel reads its
// d_conv - 1 state columns out of the cache row ids[0] and its n_t new inputs out of the projection, writes its n_t
// outputs, then its window of every snapshot. All its reads come before any of its writes, and a thread touches only
// its own channel in every row and every column, so a snapshot that overwrites the row being read, and an output the
// allocator placed exactly on the inputs (the matcher declines any other overlap), read the old values. A block is one
// head of GGML_CUDA_SSM_CONV_UPDATE_THREADS channels, so the blocks of the leading u.l2_heads heads also write the L2_NORM
// of their outputs (u.l2_dst) with l2_norm_f32's arithmetic: one warp sums the squares lane by lane in its column order,
// then the same warp reduction, rsqrtf and product, so the values match the L2_NORM's bit for bit.
// At most 56 registers a thread: a block of it lands beside a PQ2_0 group launch's block at 152 registers (a sub-partition
// keeps 1,792 of its 16,384, one warp of up to 56, mmvq-pq2-mma.cu), so it starts under the qkv matmul and its work
// before the dependency wait (the weights, the alpha/beta fold) runs there, off the chain
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA) || CUDART_VERSION < 12040
#define SSM_CONV_UPDATE_LAUNCH_BOUNDS __launch_bounds__(GGML_CUDA_SSM_CONV_UPDATE_THREADS)
#else
#define SSM_CONV_UPDATE_LAUNCH_BOUNDS __maxnreg__(56)
#endif

template <bool apply_silu, int d_conv, int max_n_t>
SSM_CONV_UPDATE_LAUNCH_BOUNDS
static __global__ void ssm_conv_state_update_f32(const ggml_cuda_ssm_conv_state_update u, const float * w,
                                                 const int w_stride, float * dst, const int dst_stride, const int n_t,
                                                 const bool prewait, const bool state_prefetch) {
    ggml_cuda_pdl_lc();
    const int c = blockIdx.x * blockDim.x + threadIdx.x;

    // state_prefetch: the block's columns of the cache row into L2 before the wait, so their DRAM latency is paid under
    // the matmul before this kernel (a hint: no result depends on it, ggml_cuda_prefetch_l2)
    if (state_prefetch && threadIdx.x == 0) {
        ggml_cuda_prefetch_l2(u.cache + (int64_t) u.ids[0] * u.row_stride + (int64_t) blockIdx.x * blockDim.x * (d_conv - 1),
                              (int64_t) blockDim.x * (d_conv - 1) * sizeof(float));
    }

    __shared__ float ab_res[GGML_CUDA_SSM_CONV_AB_MAX_ROWS][max_n_t];
    int              n_ab = 0;
    if (u.ab_rows > 0) {
        n_ab = ssm_conv_ab_rows<max_n_t>(u, n_t, ab_res);
    } else if (u.ab_slot != nullptr) {
        ssm_conv_ab_acquire(u); // released to, but not folded: the slot still goes back to 0
    }

    float x[d_conv - 1 + max_n_t];
    float wc[d_conv];
    const float * w_row = ssm_conv_w_row(u, w, w_stride, c);

    // prewait: w is a model weight, which no kernel writes, so it is requested before the dependency wait and arrives
    // while the qkv matmul before this kernel streams (its blocks leave this one room beside them: PQ2_MMA_LAUNCH_BOUNDS)
    if (prewait) {
#pragma unroll
        for (int j = 0; j < d_conv; ++j) {
            wc[j] = w_row[j];
        }
    }
    ggml_cuda_pdl_sync();
    // the pair's results: their memory is free now that every kernel before this one has completed
    if ((int) threadIdx.x < n_t) {
        for (int k = 0; k < n_ab; ++k) {
            const int r = blockIdx.x + k*gridDim.x;
            u.ab_dst[(r / u.ab_rows)*u.ab_s2 + threadIdx.x*u.ab_stride_dst + r % u.ab_rows] = ab_res[k][threadIdx.x];
        }
    }
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
    if (!prewait) {
#pragma unroll
        for (int j = 0; j < d_conv; ++j) {
            wc[j] = w_row[j];
        }
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
        // GGML_CUDA_SSM_CONV_PREWAIT_LEGACY=1: the weights after the dependency wait, as everything else
        static const bool prewait_legacy = ggml_env_switch("GGML_CUDA_SSM_CONV_PREWAIT_LEGACY");
        const bool prewait = !prewait_legacy && (u->w_seg_channels != 0 || (src1->buffer != nullptr &&
                             ggml_backend_buffer_get_usage(src1->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS));
        // GGML_CUDA_SSM_CONV_STATE_PREFETCH_LEGACY=1: no L2 prefetch of the cache row before the wait
        static const bool state_prefetch = !ggml_env_switch("GGML_CUDA_SSM_CONV_STATE_PREFETCH_LEGACY");
        // the alpha/beta pair folds in only if its activation's writer released to the plan's slot and the PQ2_0 group
        // launch writing this conv's inputs came after it, set up for the fold (ggml_cuda_try_ssm_conv_ab); then the
        // pair's own nodes, which come later, are skipped. A slot released to is acquired either way, which sets it
        // back to 0
        ggml_cuda_ssm_conv_state_update     uab = *u;
        const ggml_cuda_ssm_conv_ab_plan * ab  = ctx.ssm_conv_updates().ab_plan_of(dst);
        if (ab != nullptr && ab->writers > 0) {
            uab.ab_slot    = ctx.ssm_conv_ab_slots.ptr + ab->slot;
            uab.ab_writers = ab->writers;
        }
        if (uab.ab_slot != nullptr && ab->fed) {
            ctx.ssm_conv_updates().skipped.insert(ab->pair, ab->pair + 2);
        } else {
            uab.ab_rows = 0;
        }
        if (fuse_silu) {
            // the L2_NORM normalizes the SILU's output: run it here only with the SILU fused, and only then skip it
            if (const ggml_tensor * l2 = ctx.ssm_conv_updates().l2_norm_of(dst)) {
                ctx.ssm_conv_updates().skipped.insert(l2);
            }
            // up to 4 tokens (decode, an MTP verify of up to 3 drafts) with half the registers of 8
            if (n_t <= 4) {
                ggml_cuda_kernel_launch(ssm_conv_state_update_f32<true, GGML_CUDA_SSM_CONV_UPDATE_D_CONV, 4>,
                                        launch_params, uab, src1_d, w_stride, dst_d, dst_stride, (int) n_t, prewait,
                                        state_prefetch);
            } else {
                ggml_cuda_kernel_launch(ssm_conv_state_update_f32<true, GGML_CUDA_SSM_CONV_UPDATE_D_CONV, GGML_CUDA_SSM_CONV_UPDATE_MAX_N_T>,
                                        launch_params, uab, src1_d, w_stride, dst_d, dst_stride, (int) n_t, prewait,
                                        state_prefetch);
            }
        } else {
            ggml_cuda_ssm_conv_state_update raw = uab;
            raw.l2_heads = 0;
            if (n_t <= 4) {
                ggml_cuda_kernel_launch(ssm_conv_state_update_f32<false, GGML_CUDA_SSM_CONV_UPDATE_D_CONV, 4>,
                                        launch_params, raw, src1_d, w_stride, dst_d, dst_stride, (int) n_t, prewait,
                                        state_prefetch);
            } else {
                ggml_cuda_kernel_launch(ssm_conv_state_update_f32<false, GGML_CUDA_SSM_CONV_UPDATE_D_CONV, GGML_CUDA_SSM_CONV_UPDATE_MAX_N_T>,
                                        launch_params, raw, src1_d, w_stride, dst_d, dst_stride, (int) n_t, prewait,
                                        state_prefetch);
            }
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
