#include "ggml.h"
#include "common.cuh"
#include "unary.cuh"
#include "mmvf.cuh"
#include "convert.cuh"

template <typename T, typename type_acc, int ncols_dst, int block_size, bool has_fusion = false, bool is_multi_token_id = false>
static __global__ void mul_mat_vec_f(
        const T * x_ptr, const float * y_ptr, const int32_t * ids_ptr, const ggml_cuda_mm_fusion_args_device fusion, float * dst_ptr,
        const int ncols2, const uint3 nchannels_y, const int stride_row, const int stride_col_y2, const int stride_col_dst,
        const uint3 channel_ratio, const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const uint3 sample_ratio, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,
        const int ids_stride, const bool lc_early) {
    // lc_early: the next kernel may launch once every block has started, so it lands while this one reads and its
    // requests before its own wait (the Gated DeltaNet's state, after qwen35's alpha/beta pair) run under this one's;
    // it still waits for this grid to complete, so no result depends on where the trigger sits
    if (lc_early) {
        ggml_cuda_pdl_lc();
    }
    const T       * GGML_CUDA_RESTRICT x   = x_ptr;
    const float   * GGML_CUDA_RESTRICT y   = y_ptr;
    const int32_t * GGML_CUDA_RESTRICT ids = ids_ptr;
    float         * GGML_CUDA_RESTRICT dst = dst_ptr;
    const int row         = blockIdx.x;
    // for MUL_MAT_ID - blockIdx.y = n_expert_used, blockIdx.z = ncols_dst (tokens)
    const int channel_dst = blockIdx.y;
    const int tid         = threadIdx.x;

    int token_idx;
    int channel_x;
    int channel_y;
    int sample_dst;

    ggml_cuda_pdl_sync();
    if constexpr (is_multi_token_id) {
        // Multi-token MUL_MAT_ID path, adding these in the normal path causes a perf regression for n_tokens=1 case
        token_idx  = blockIdx.z;
        channel_x  = ids[channel_dst + token_idx * ids_stride];
        channel_y  = fastmodulo(channel_dst, nchannels_y);
        sample_dst = 0;
    } else {
        token_idx  = ids ? blockIdx.z                                          : 0;
        channel_x  = ids ? ids[blockIdx.y + token_idx * ids_stride]            : fastdiv((uint32_t) channel_dst, channel_ratio);
        channel_y  = ids ? fastmodulo(blockIdx.y, nchannels_y)                 : channel_dst;
        sample_dst = ids ? 0                                                   : blockIdx.z;
    }

    const int sample_x    = fastdiv((uint32_t) sample_dst, sample_ratio);
    const int sample_y    = sample_dst;

    constexpr int warp_size   = ggml_cuda_get_physical_warp_size();

    x   += int64_t(sample_x)  *stride_sample_x   + channel_x  *stride_channel_x   + row*stride_row;
    y   += int64_t(sample_y)  *stride_sample_y   + channel_y  *stride_channel_y;
    dst += int64_t(sample_dst)*stride_sample_dst + channel_dst*stride_channel_dst;
    if constexpr (is_multi_token_id) {
        y   += token_idx*stride_col_y2*2;
        dst += token_idx*stride_col_dst;
    }

    bool use_gate = false;
    bool use_bias = false;
    bool use_gate_bias = false;
    bool use_norm = false;
    ggml_glu_op glu_op = ggml_glu_op::GGML_GLU_OP_SWIGLU;
    const T * gate_x = nullptr;
    const float * x_bias = nullptr;
    const float * gate_bias = nullptr;

    if constexpr (has_fusion) {
        use_gate = fusion.gate != nullptr;
        use_bias = fusion.x_bias != nullptr;
        use_gate_bias = fusion.gate_bias != nullptr;
        use_norm = fusion.rms_norm;
        glu_op = fusion.glu_op;

        if (use_gate) {
            gate_x = static_cast<const T *>(fusion.gate);
        }
        if (use_bias) {
            x_bias = static_cast<const float *>(fusion.x_bias);
        }
        if (use_gate_bias) {
            gate_bias = static_cast<const float *>(fusion.gate_bias);
            use_gate_bias = use_gate;
        } else {
            use_gate_bias = false;
        }
    }

    if (use_gate) {
        gate_x += int64_t(sample_x)  *stride_sample_x   + channel_x  *stride_channel_x   + row*stride_row;
    }

    if constexpr (has_fusion) {
        const int channel_bias = ids ? channel_x : channel_dst;
        if (use_bias) {
            x_bias += int64_t(sample_dst)*stride_sample_dst + channel_bias*stride_channel_dst;
        }
        if (use_gate_bias) {
            gate_bias += int64_t(sample_dst)*stride_sample_dst + channel_bias*stride_channel_dst;
        }
    }

    const float2 * y2 = (const float2 *) y;

    extern __shared__ char data_mmv[];
    float * buf_iw = (float *) data_mmv;
    [[maybe_unused]] float * buf_iw_gate = nullptr;
    [[maybe_unused]] float * buf_iw_sq   = nullptr;
    if constexpr (has_fusion) {
        buf_iw_gate = (float *) (data_mmv + warp_size*sizeof(float));
        buf_iw_sq   = (float *) (data_mmv + 2*warp_size*sizeof(float));
    }

    if (block_size > warp_size) {
        if (tid < warp_size) {
            buf_iw[tid] = 0.0f;
            if constexpr (has_fusion) {
                if (use_gate) {
                    buf_iw_gate[tid] = 0.0f;
                }
                if (use_norm) {
                    buf_iw_sq[tid] = 0.0f;
                }
            }
        }
        __syncthreads();
    }

    float sumf[ncols_dst] = {0.0f};
    float sumf_gate[ncols_dst];
    float sumsq[ncols_dst]; // with use_norm: the sum of squares of src1's column, for the RMS norm's scale
    if constexpr (has_fusion) {
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
            sumf_gate[j] = 0.0f;
            sumsq[j]     = 0.0f;
        }
    }

    if constexpr (std::is_same_v<T, float>) {
        const float2 * x2 = (const float2 *) x;
        [[maybe_unused]] const float2 * gate_x2 = nullptr;
        if constexpr (has_fusion) {
            if (use_gate) {
                gate_x2 = (const float2 *) gate_x;
            }
        }

        for (int col2 = tid; col2 < ncols2; col2 += block_size) {
            const float2 tmpx = x2[col2];
            float2 tmpx_gate = make_float2(0.0f, 0.0f);
            if constexpr (has_fusion) {
                if (use_gate) {
                    tmpx_gate = gate_x2[col2];
                }
            }

#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
                const float2 tmpy = y2[j*stride_col_y2 + col2];
                if constexpr (has_fusion) {
                    if (use_norm) {
                        ggml_cuda_mad(sumsq[j], tmpy.x, tmpy.x);
                        ggml_cuda_mad(sumsq[j], tmpy.y, tmpy.y);
                    }
                }
                ggml_cuda_mad(sumf[j], tmpx.x, tmpy.x);
                ggml_cuda_mad(sumf[j], tmpx.y, tmpy.y);

                if constexpr (has_fusion) {
                    if (use_gate) {
                        ggml_cuda_mad(sumf_gate[j], tmpx_gate.x, tmpy.x);
                        ggml_cuda_mad(sumf_gate[j], tmpx_gate.y, tmpy.y);
                    }
                }
            }
        }
    } else if constexpr (std::is_same_v<T, half>) {
        const half2 * x2 = (const half2 *) x;
        [[maybe_unused]] const half2 * gate_x2 = nullptr;
        if constexpr (has_fusion) {
            if (use_gate) {
                gate_x2 = (const half2 *) gate_x;
            }
        }

        if (std::is_same_v<type_acc, float>) {
            for (int col2 = tid; col2 < ncols2; col2 += block_size) {
                const float2 tmpx = __half22float2(x2[col2]);
                float2 tmpx_gate = make_float2(0.0f, 0.0f);
                if constexpr (has_fusion) {
                    if (use_gate) {
                        tmpx_gate = __half22float2(gate_x2[col2]);
                    }
                }
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    const float2 tmpy = y2[j*stride_col_y2 + col2];
                    if constexpr (has_fusion) {
                        if (use_norm) {
                            ggml_cuda_mad(sumsq[j], tmpy.x, tmpy.x);
                            ggml_cuda_mad(sumsq[j], tmpy.y, tmpy.y);
                        }
                    }
                    ggml_cuda_mad(sumf[j], tmpx.x, tmpy.x);
                    ggml_cuda_mad(sumf[j], tmpx.y, tmpy.y);

                    if constexpr (has_fusion) {
                        if (use_gate) {
                            ggml_cuda_mad(sumf_gate[j], tmpx_gate.x, tmpy.x);
                            ggml_cuda_mad(sumf_gate[j], tmpx_gate.y, tmpy.y);
                        }
                    }
                }
            }
        } else {
#ifdef FP16_AVAILABLE
            half2 sumh2[ncols_dst] = {{0.0f, 0.0f}};
            half2 sumh2_gate[ncols_dst] = {{0.0f, 0.0f}};

            for (int col2 = tid; col2 < ncols2; col2 += block_size) {
                const half2 tmpx = x2[col2];
                half2 tmpx_gate = make_half2(0.0f, 0.0f);
                if constexpr (has_fusion) {
                    if (use_gate) {
                        tmpx_gate = gate_x2[col2];
                    }
                }
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    const float2 tmpy = y2[j*stride_col_y2 + col2];
                    if constexpr (has_fusion) {
                        if (use_norm) {
                            ggml_cuda_mad(sumsq[j], tmpy.x, tmpy.x);
                            ggml_cuda_mad(sumsq[j], tmpy.y, tmpy.y);
                        }
                    }
                    sumh2[j] += tmpx * make_half2(tmpy.x, tmpy.y);

                    if constexpr (has_fusion) {
                        if (use_gate) {
                            sumh2_gate[j] += tmpx_gate * make_half2(tmpy.x, tmpy.y);
                        }
                    }
                }
            }

#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
                sumf[j] = __low2float(sumh2[j]) + __high2float(sumh2[j]);
            }

            if constexpr (has_fusion) {
                if (use_gate) {
#pragma unroll
                    for (int j = 0; j < ncols_dst; ++j) {
                        sumf_gate[j] = __low2float(sumh2_gate[j]) + __high2float(sumh2_gate[j]);
                    }
                }
            }
#else
            NO_DEVICE_CODE;
#endif // FP16_AVAILABLE
        }
    } else if constexpr (std::is_same_v<T, nv_bfloat16>) {
//TODO: add support for ggml_cuda_mad for hip_bfloat162
#if defined(GGML_USE_HIP)
        const int * x2 = (const int *) x;
        const int * gate_x2 = nullptr;
        if constexpr (has_fusion) {
            if (use_gate) {
                gate_x2 = (const int *) gate_x;
            }
        }
        for (int col2 = tid; col2 < ncols2; col2 += block_size) {
            const int tmpx = x2[col2];
            int tmpx_gate = 0;
            if constexpr (has_fusion) {
                if (use_gate) {
                    tmpx_gate = gate_x2[col2];
                }
            }
#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
                const float2 tmpy = y2[j*stride_col_y2 + col2];
                if constexpr (has_fusion) {
                    if (use_norm) {
                        ggml_cuda_mad(sumsq[j], tmpy.x, tmpy.x);
                        ggml_cuda_mad(sumsq[j], tmpy.y, tmpy.y);
                    }
                }
                const float tmpx0 = ggml_cuda_cast<float>(reinterpret_cast<const nv_bfloat16 *>(&tmpx)[0]);
                const float tmpx1 = ggml_cuda_cast<float>(reinterpret_cast<const nv_bfloat16 *>(&tmpx)[1]);
                ggml_cuda_mad(sumf[j], tmpx0, tmpy.x);
                ggml_cuda_mad(sumf[j], tmpx1, tmpy.y);

                if constexpr (has_fusion) {
                    if (use_gate) {
                        const float tmpx0_gate = ggml_cuda_cast<float>(reinterpret_cast<const nv_bfloat16 *>(&tmpx_gate)[0]);
                        const float tmpx1_gate = ggml_cuda_cast<float>(reinterpret_cast<const nv_bfloat16 *>(&tmpx_gate)[1]);
                        ggml_cuda_mad(sumf_gate[j], tmpx0_gate, tmpy.x);
                        ggml_cuda_mad(sumf_gate[j], tmpx1_gate, tmpy.y);
                    }
                }
            }
        }
#else
        const nv_bfloat162 * x2 = (const nv_bfloat162 *) x;
        [[maybe_unused]] const nv_bfloat162 * gate_x2 = nullptr;
        if constexpr (has_fusion) {
            if (use_gate) {
                gate_x2 = (const nv_bfloat162 *) gate_x;
            }
        }
        for (int col2 = tid; col2 < ncols2; col2 += block_size) {
            const nv_bfloat162 tmpx = x2[col2];
            [[maybe_unused]] nv_bfloat162 tmpx_gate;
            if constexpr (has_fusion) {
                if (use_gate) {
                    tmpx_gate = gate_x2[col2];
                }
            }
#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
                const float2 tmpy = y2[j*stride_col_y2 + col2];
                if constexpr (has_fusion) {
                    if (use_norm) {
                        ggml_cuda_mad(sumsq[j], tmpy.x, tmpy.x);
                        ggml_cuda_mad(sumsq[j], tmpy.y, tmpy.y);
                    }
                }
                ggml_cuda_mad(sumf[j], tmpx.x, tmpy.x);
                ggml_cuda_mad(sumf[j], tmpx.y, tmpy.y);

                if constexpr (has_fusion) {
                    if (use_gate) {
                        ggml_cuda_mad(sumf_gate[j], tmpx_gate.x, tmpy.x);
                        ggml_cuda_mad(sumf_gate[j], tmpx_gate.y, tmpy.y);
                    }
                }
            }
        }
#endif
    } else {
        static_assert(std::is_same_v<T, void>, "unsupported type");
    }

    if (!lc_early) {
        ggml_cuda_pdl_lc();
    }
#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
        sumf[j] = warp_reduce_sum<warp_size>(sumf[j]);

        if constexpr (has_fusion) {
            if (use_gate) {
                sumf_gate[j] = warp_reduce_sum<warp_size>(sumf_gate[j]);
            }
            if (use_norm) {
                sumsq[j] = warp_reduce_sum<warp_size>(sumsq[j]);
            }
        }

        if (block_size > warp_size) {
            buf_iw[tid/warp_size] = sumf[j];
            if constexpr (has_fusion) {
                if (use_gate) {
                    buf_iw_gate[tid/warp_size] = sumf_gate[j];
                }
                if (use_norm) {
                    buf_iw_sq[tid/warp_size] = sumsq[j];
                }
            }
            __syncthreads();
            if (tid < warp_size) {
                sumf[j] = buf_iw[tid];
                sumf[j] = warp_reduce_sum<warp_size>(sumf[j]);
                if constexpr (has_fusion) {
                    if (use_gate) {
                        sumf_gate[j] = buf_iw_gate[tid];
                        sumf_gate[j] = warp_reduce_sum<warp_size>(sumf_gate[j]);
                    }
                    if (use_norm) {
                        sumsq[j] = buf_iw_sq[tid];
                        sumsq[j] = warp_reduce_sum<warp_size>(sumsq[j]);
                    }
                }
            }

            if (j < ncols_dst) {
                __syncthreads();
            }
        }
    }

    if (tid >= ncols_dst) {
        return;
    }

    float value = sumf[tid];

    // the RMS norm's scale is one per column of src1, so it scales the dot products: W(s y) = s W y
    [[maybe_unused]] float norm_scale = 1.0f;
    if constexpr (has_fusion) {
        if (use_norm) {
            norm_scale = rsqrtf(sumsq[tid]/(2*ncols2) + fusion.rms_norm_eps);
            value *= norm_scale;
        }
    }

    if constexpr (has_fusion) {
        if (use_bias) {
            value += x_bias[tid*stride_col_dst + row];
        }

        if (use_gate) {
            float gate_value = sumf_gate[tid] * norm_scale;
            if (use_gate_bias) {
                gate_value += gate_bias[tid*stride_col_dst + row];
            }
            switch (glu_op) {
                case GGML_GLU_OP_SWIGLU:
                    value *= ggml_cuda_op_silu_single(gate_value);
                    break;
                case GGML_GLU_OP_GEGLU:
                    value *= ggml_cuda_op_gelu_single(gate_value);
                    break;
                case GGML_GLU_OP_SWIGLU_OAI: {
                    value = ggml_cuda_op_swiglu_oai_single(gate_value, value);
                    break;
                }
                default:
                    break;
            }
        }
    }

    dst[tid*stride_col_dst + row] = value;

    if constexpr (!has_fusion) {
        GGML_UNUSED_VARS(use_gate, use_bias, use_gate_bias, use_norm, glu_op, gate_x, x_bias, gate_bias, sumf_gate, sumsq);
    }
}

// The small-shape path. The kernel above gives a row one block and reads it a pair of elements an iteration (4 bytes of
// f16 or bf16), each iteration's load awaited before the next is requested: at the Gated DeltaNet's projections (K 128
// over 4096 rows, K 4096 over 128 rows) that is one DRAM round trip after another, or thousands of two-warp blocks, at
// 144 to 446 GB/s of an RTX PRO 6000's 1,598 (GLM-5.3, a token's ssm_f_b, ssm_g_b, ssm_f_a, ssm_g_a and ssm_beta 564 us
// against a ~100 us floor). Here a row is tpr threads, each requesting its vpt 16-byte chunks of the row at once (before
// the PDL wait when x is the model's weights, which no kernel writes), MMVF_VEC_BLOCK/tpr rows a block, and a row's sum
// is taken over its lanes, and over its warps past one. The dot products accumulate in f32 whatever the type.
static constexpr int MMVF_VEC_BLOCK    = 256;
static constexpr int MMVF_VEC_MAX_VPT  = 4;
static constexpr int MMVF_VEC_MAX_COLS = 4;

template <typename T>
static __device__ __forceinline__ void mmvf_vec_unpack(const uint4 & u, float * v) {
    if constexpr (std::is_same_v<T, float>) {
        v[0] = __uint_as_float(u.x); v[1] = __uint_as_float(u.y); v[2] = __uint_as_float(u.z); v[3] = __uint_as_float(u.w);
    } else if constexpr (std::is_same_v<T, half>) {
        const half2 * h = (const half2 *) &u;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const float2 f = __half22float2(h[k]);
            v[2*k + 0] = f.x;
            v[2*k + 1] = f.y;
        }
    } else {
        static_assert(std::is_same_v<T, nv_bfloat16>, "unsupported type");
        const nv_bfloat162 * h = (const nv_bfloat162 *) &u;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            v[2*k + 0] = __low2float(h[k]);
            v[2*k + 1] = __high2float(h[k]);
        }
    }
}

template <typename T, int ncols_dst, int vpt>
static __global__ void __launch_bounds__(MMVF_VEC_BLOCK) mul_mat_vec_f_vec(
        const T * x_ptr, const float * y_ptr, const float * bias_ptr, float * dst_ptr,
        const int nrows, const int nchunks, const int tpr, const int64_t stride_row, const int64_t stride_col_y,
        const int64_t stride_col_dst, const uint3 channel_ratio, const int64_t stride_channel_x,
        const int64_t stride_channel_y, const int64_t stride_channel_dst, const uint3 sample_ratio,
        const int64_t stride_sample_x, const int64_t stride_sample_y, const int64_t stride_sample_dst,
        const bool x_prewait) {
    constexpr int E         = 16 / sizeof(T); // elements a chunk
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    // the next launch may start once every block has started: it still waits for this grid to complete
    ggml_cuda_pdl_lc();
    // no __restrict__ under PDL (GGML_CUDA_RESTRICT): y is the kernel before's output, read after the wait
    const T     * GGML_CUDA_RESTRICT x    = x_ptr;
    const float * GGML_CUDA_RESTRICT y    = y_ptr;
    const float * GGML_CUDA_RESTRICT bias = bias_ptr;
    float       * GGML_CUDA_RESTRICT dst  = dst_ptr;

    const int  t      = threadIdx.x % tpr;
    const int  row    = blockIdx.x*(MMVF_VEC_BLOCK/tpr) + threadIdx.x/tpr;
    const bool row_ok = row < nrows;

    const int channel_dst = blockIdx.y;
    const int sample_dst  = blockIdx.z;
    const int channel_x   = fastdiv((uint32_t) channel_dst, channel_ratio);
    const int sample_x    = fastdiv((uint32_t) sample_dst,  sample_ratio);

    const uint4 * xr = (const uint4 *) (x + sample_x*stride_sample_x + channel_x*stride_channel_x + int64_t(row)*stride_row);
    y   += sample_dst*stride_sample_y   + channel_dst*stride_channel_y;
    dst += sample_dst*stride_sample_dst + channel_dst*stride_channel_dst;
    if (bias) {
        bias += sample_dst*stride_sample_dst + channel_dst*stride_channel_dst;
    }

    uint4 xv[vpt];
    const auto load_x = [&]() {
#pragma unroll
        for (int v = 0; v < vpt; ++v) {
            const int c = t + v*tpr;
            xv[v] = row_ok && c < nchunks ? xr[c] : make_uint4(0, 0, 0, 0);
        }
    };
    if (x_prewait) {
        load_x();
    }
    ggml_cuda_pdl_sync();
    if (!x_prewait) {
        load_x();
    }

    float sum[ncols_dst] = {0.0f};
#pragma unroll
    for (int v = 0; v < vpt; ++v) {
        const int c = t + v*tpr;
        if (c < nchunks) {
            float xf[E];
            mmvf_vec_unpack<T>(xv[v], xf);
#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
                const float4 * yc = (const float4 *) (y + j*stride_col_y + int64_t(c)*E);
#pragma unroll
                for (int q = 0; q < E/4; ++q) {
                    const float4 yq = yc[q];
                    sum[j] += xf[4*q + 0]*yq.x;
                    sum[j] += xf[4*q + 1]*yq.y;
                    sum[j] += xf[4*q + 2]*yq.z;
                    sum[j] += xf[4*q + 3]*yq.w;
                }
            }
        }
    }

    // a row's lanes are tpr aligned lanes of a warp (all of it past warp_size): an xor under tpr stays inside the row
    const int lanes = tpr < warp_size ? tpr : warp_size;
#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
        for (int offset = lanes/2; offset > 0; offset >>= 1) {
            sum[j] += __shfl_xor_sync(0xffffffff, sum[j], offset, warp_size);
        }
    }
    if (tpr > warp_size) {
        __shared__ float buf[ncols_dst][MMVF_VEC_BLOCK/warp_size];
        const int warp = threadIdx.x / warp_size;
        if (threadIdx.x % warp_size == 0) {
#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
                buf[j][warp] = sum[j];
            }
        }
        __syncthreads();
        if (t == 0) {
#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
                float s = 0.0f;
                for (int w = 0; w < tpr/warp_size; ++w) {
                    s += buf[j][warp + w];
                }
                sum[j] = s;
            }
        }
    }

    if (t != 0 || !row_ok) {
        return;
    }
#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
        dst[j*stride_col_dst + row] = bias ? sum[j] + bias[j*stride_col_dst + row] : sum[j];
    }
}

// the small-shape path's plan for rows of nchunks 16-byte chunks: tpr threads a row (a power of two up to
// MMVF_VEC_BLOCK) and vpt chunks a thread; false where a row would need more than MMVF_VEC_MAX_VPT a thread
static bool mmvf_vec_plan(const int64_t nchunks, int & tpr, int & vpt) {
    tpr = 1;
    while (tpr < nchunks && tpr < MMVF_VEC_BLOCK) {
        tpr *= 2;
    }
    vpt = (int) ((nchunks + tpr - 1) / tpr);
    return nchunks >= 1 && vpt <= MMVF_VEC_MAX_VPT;
}

// launches the small-shape path where it takes the shape (no MUL_MAT_ID, no fusion but a bias, at most
// MMVF_VEC_MAX_COLS columns, every row and column 16-byte aligned, a row at most MMVF_VEC_MAX_VPT chunks a thread);
// false leaves the product to the kernel above. GGML_CUDA_MMVF_VEC_LEGACY=1 turns it off.
template <typename T, int ncols_dst>
static bool mul_mat_vec_f_vec_launch(
        const T * x, const float * y, const ggml_cuda_mm_fusion_args_device & fusion, float * dst,
        const int64_t ncols, const int64_t nrows, const int64_t stride_row, const int64_t stride_col_y,
        const int64_t stride_col_dst, const int64_t nchannels_dst, const uint3 channel_ratio, const int64_t stride_channel_x,
        const int64_t stride_channel_y, const int64_t stride_channel_dst, const int64_t nsamples_dst,
        const uint3 sample_ratio, const int64_t stride_sample_x, const int64_t stride_sample_y,
        const int64_t stride_sample_dst, const bool x_prewait, cudaStream_t stream) {
    static const bool legacy = ggml_env_switch("GGML_CUDA_MMVF_VEC_LEGACY");
    constexpr int64_t E = 16 / sizeof(T);

    if constexpr (ncols_dst > MMVF_VEC_MAX_COLS) {
        return false;
    } else {
        const bool fusion_ok = fusion.gate == nullptr && fusion.gate_bias == nullptr && !fusion.rms_norm;
        const bool aligned   = (uintptr_t) x % 16 == 0 && (uintptr_t) y % 16 == 0 && ncols % E == 0 &&
            stride_row % E == 0 && stride_channel_x % E == 0 && stride_sample_x % E == 0 &&
            stride_col_y % 4 == 0 && stride_channel_y % 4 == 0 && stride_sample_y % 4 == 0;
        int tpr = 0;
        int vpt = 0;
        if (legacy || !fusion_ok || !aligned || nrows > INT_MAX || !mmvf_vec_plan(ncols / E, tpr, vpt)) {
            return false;
        }

        const int  rows_per_block = MMVF_VEC_BLOCK / tpr;
        const dim3 block_nums((nrows + rows_per_block - 1) / rows_per_block, nchannels_dst, nsamples_dst);
        const ggml_cuda_kernel_launch_params params(block_nums, dim3(MMVF_VEC_BLOCK, 1, 1), 0, stream);
        const float * bias = (const float *) fusion.x_bias;
#define MMVF_VEC_LAUNCH(VPT) \
        ggml_cuda_kernel_launch(mul_mat_vec_f_vec<T, ncols_dst, VPT>, params, x, y, bias, dst, (int) nrows, (int) (ncols / E), \
            tpr, stride_row, stride_col_y, stride_col_dst, channel_ratio, stride_channel_x, stride_channel_y, \
            stride_channel_dst, sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, x_prewait)
        switch (vpt) {
            case 1: MMVF_VEC_LAUNCH(1); break;
            case 2: MMVF_VEC_LAUNCH(2); break;
            case 3: MMVF_VEC_LAUNCH(3); break;
            case 4: MMVF_VEC_LAUNCH(4); break;
            default: GGML_ABORT("fatal error");
        }
#undef MMVF_VEC_LAUNCH
        return true;
    }
}

template<typename T, typename type_acc, int ncols_dst, int block_size, bool is_multi_token_id = false>
static void mul_mat_vec_f_switch_fusion(
        const T * x, const float * y, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const int64_t ncols, const uint3 nchannels_y,
        const int64_t stride_row, const int64_t stride_col_y, const int64_t stride_col_dst,
        const uint3 channel_ratio, const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const uint3 sample_ratio, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,
        const dim3 & block_dims, const dim3 & block_nums, const int nbytes_shared, const int ids_stride, const cudaStream_t stream) {

    const ggml_cuda_kernel_launch_params launch_params = {block_nums, block_dims, nbytes_shared, stream};

    // the kernel triggers the next launch at its start; GGML_CUDA_MMVF_TRIGGER_LEGACY=1 after its dot products
    static const bool lc_legacy = ggml_env_switch("GGML_CUDA_MMVF_TRIGGER_LEGACY");

    const bool has_fusion = fusion.gate != nullptr || fusion.x_bias != nullptr || fusion.gate_bias != nullptr || fusion.rms_norm;
    if constexpr (ncols_dst == 1) {
        if (has_fusion) {
            ggml_cuda_kernel_launch(mul_mat_vec_f<T, type_acc, ncols_dst, block_size, true, is_multi_token_id>, launch_params,
                x, y, ids, fusion, dst, ncols, nchannels_y, stride_row, stride_col_y, stride_col_dst,
                channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, !lc_legacy);
            return;
       }
    }

    GGML_ASSERT(!has_fusion && "fusion only supported for ncols_dst=1");

    ggml_cuda_kernel_launch(mul_mat_vec_f<T, type_acc, ncols_dst, block_size, false, is_multi_token_id>, launch_params,
        x, y, ids, fusion, dst, ncols, nchannels_y, stride_row, stride_col_y, stride_col_dst,
        channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
        sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, !lc_legacy);

}

template <typename T, typename type_acc, int ncols_dst, bool is_multi_token_id = false>
void launch_mul_mat_vec_f_cuda(
        const T * x, const float * y, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const int64_t ncols, const int64_t nrows,
        const int64_t stride_row, const int64_t stride_col_y, const int64_t stride_col_dst,
        const int64_t nchannels_x, const int64_t nchannels_y, const int64_t nchannels_dst,
        const int64_t stride_channel_x, const int64_t stride_channel_y, const int64_t stride_channel_dst, const int64_t nsamples_x,
        const int64_t nsamples_dst, const int64_t stride_sample_x, const int64_t stride_sample_y, const int64_t stride_sample_dst,
        const int64_t nsamples_or_ntokens, const int64_t ids_stride, cudaStream_t stream, const bool x_prewait = false) {
    GGML_ASSERT(ncols        % 2 == 0);
    GGML_ASSERT(stride_row   % 2 == 0);
    GGML_ASSERT(stride_col_y % 2 == 0);
    GGML_ASSERT(ids || nchannels_dst % nchannels_x == 0);
    GGML_ASSERT(       nsamples_dst  % nsamples_x  == 0);
    const uint3 nchannels_y_fd   = ids ? init_fastdiv_values(nchannels_y) : make_uint3(0, 0, 0);
    const uint3 channel_ratio_fd = ids ? make_uint3(0, 0, 0) : init_fastdiv_values(nchannels_dst / nchannels_x);
    const uint3 sample_ratio_fd  = init_fastdiv_values(nsamples_dst  / nsamples_x);

    if constexpr (!is_multi_token_id) {
        if (!ids && mul_mat_vec_f_vec_launch<T, ncols_dst>(x, y, fusion, dst, ncols, nrows, stride_row, stride_col_y,
                stride_col_dst, nchannels_dst, channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                nsamples_dst, sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst, x_prewait, stream)) {
            return;
        }
    }

    const int device = ggml_cuda_get_device();
    const int warp_size = ggml_cuda_info().devices[device].warp_size;

    int64_t block_size_best = warp_size;
    int64_t niter_best      = (ncols + 2*warp_size - 1) / (2*warp_size);
    int64_t max_block_size  = 256;
    if(ggml_cuda_info().devices[device].cc > GGML_CUDA_CC_OFFSET_AMD && ggml_cuda_info().devices[device].cc < GGML_CUDA_CC_RDNA1) {
        max_block_size = 128;
    }
    for (int64_t block_size = 2*warp_size; block_size <= max_block_size; block_size += warp_size) {
        const int64_t niter = (ncols + 2*block_size - 1) / (2*block_size);
        if (niter < niter_best) {
            niter_best      = niter;
            block_size_best = block_size;
        }
    }

    const bool has_fusion = fusion.gate != nullptr || fusion.x_bias != nullptr || fusion.gate_bias != nullptr || fusion.rms_norm;

    const int nbytes_shared = warp_size*sizeof(float) + (has_fusion ? 2*warp_size*sizeof(float) : 0);
    const dim3 block_nums(nrows, nchannels_dst, nsamples_or_ntokens);
    const dim3 block_dims(block_size_best, 1, 1);
    switch (block_size_best) {
        case   32: {
            mul_mat_vec_f_switch_fusion<T, type_acc, ncols_dst, 32, is_multi_token_id>
                (x, y, ids, fusion, dst, ncols/2, nchannels_y_fd, stride_row, stride_col_y/2, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst, block_dims, block_nums, nbytes_shared, ids_stride, stream);
        } break;
        case   64: {
            mul_mat_vec_f_switch_fusion<T, type_acc, ncols_dst, 64, is_multi_token_id>
                (x, y, ids, fusion, dst, ncols/2, nchannels_y_fd, stride_row, stride_col_y/2, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst, block_dims, block_nums, nbytes_shared, ids_stride, stream);
        } break;
        case   96: {
            mul_mat_vec_f_switch_fusion<T, type_acc, ncols_dst, 96, is_multi_token_id>
                (x, y, ids, fusion, dst, ncols/2, nchannels_y_fd, stride_row, stride_col_y/2, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst, block_dims, block_nums, nbytes_shared, ids_stride, stream);
        } break;
        case  128: {
            mul_mat_vec_f_switch_fusion<T, type_acc, ncols_dst, 128, is_multi_token_id>
                (x, y, ids, fusion, dst, ncols/2, nchannels_y_fd, stride_row, stride_col_y/2, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst, block_dims, block_nums, nbytes_shared, ids_stride, stream);
        } break;
        case  160: {
            mul_mat_vec_f_switch_fusion<T, type_acc, ncols_dst, 160, is_multi_token_id>
                (x, y, ids, fusion, dst, ncols/2, nchannels_y_fd, stride_row, stride_col_y/2, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst, block_dims, block_nums, nbytes_shared, ids_stride, stream);
        } break;
        case  192: {
            mul_mat_vec_f_switch_fusion<T, type_acc, ncols_dst, 192, is_multi_token_id>
                (x, y, ids, fusion, dst, ncols/2, nchannels_y_fd, stride_row, stride_col_y/2, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst, block_dims, block_nums, nbytes_shared, ids_stride, stream);
        } break;
        case  224: {
            mul_mat_vec_f_switch_fusion<T, type_acc, ncols_dst, 224, is_multi_token_id>
                (x, y, ids, fusion, dst, ncols/2, nchannels_y_fd, stride_row, stride_col_y/2, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst, block_dims, block_nums, nbytes_shared, ids_stride, stream);
        } break;
        case  256: {
            mul_mat_vec_f_switch_fusion<T, type_acc, ncols_dst, 256, is_multi_token_id>
                (x, y, ids, fusion, dst, ncols/2, nchannels_y_fd, stride_row, stride_col_y/2, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst, block_dims, block_nums, nbytes_shared, ids_stride, stream);
        } break;
        default: {
            GGML_ABORT("fatal error");
        } break;
    }
}

template <typename T, typename type_acc>
static void mul_mat_vec_f_cuda_switch_ncols_dst(
        const T * x, const float * y, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const int64_t ncols, const int64_t nrows, const int64_t ncols_dst,
        const int64_t stride_row, const int64_t stride_col_y, const int64_t stride_col_dst,
        const int64_t nchannels_x, const int64_t nchannels_y, const int64_t nchannels_dst,
        const int64_t stride_channel_x, const int64_t stride_channel_y, const int64_t stride_channel_dst, const int64_t nsamples_x,
        const int64_t nsamples_dst, const int64_t stride_sample_x, const int64_t stride_sample_y, const int64_t stride_sample_dst,
        const int64_t ids_stride, cudaStream_t stream, const bool x_prewait = false) {

    const bool has_ids = ids != nullptr;

    if (has_ids && ncols_dst > 1) {
        // Multi-token MUL_MAT_ID path only - single-token goes through regular path below
        constexpr int c_ncols_dst = 1;
        launch_mul_mat_vec_f_cuda<T, type_acc, c_ncols_dst, true>
            (x, y, ids, fusion, dst, ncols, nrows, stride_row, stride_col_y, stride_col_dst,
             nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
             stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
             ncols_dst, ids_stride, stream);
        return;
    }

    if (has_ids) {
        // Single-token MUL_MAT_ID path
        constexpr int c_ncols_dst = 1;
        launch_mul_mat_vec_f_cuda<T, type_acc, c_ncols_dst>
            (x, y, ids, fusion, dst, ncols, nrows, stride_row, stride_col_y, stride_col_dst,
             nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
             stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
             ncols_dst, ids_stride, stream);
        return;
    }

    switch (ncols_dst) {
        case 1:
            launch_mul_mat_vec_f_cuda<T, type_acc, 1>
                (x, y, ids, fusion, dst, ncols, nrows, stride_row, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
                 stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                 nsamples_dst, ids_stride, stream, x_prewait);
            break;
        case 2:
            launch_mul_mat_vec_f_cuda<T, type_acc, 2>
                (x, y, ids, fusion, dst, ncols, nrows, stride_row, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
                 stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                 nsamples_dst, ids_stride, stream, x_prewait);
            break;
        case 3:
            launch_mul_mat_vec_f_cuda<T, type_acc, 3>
                (x, y, ids, fusion, dst, ncols, nrows, stride_row, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
                 stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                 nsamples_dst, ids_stride, stream, x_prewait);
            break;
        case 4:
            launch_mul_mat_vec_f_cuda<T, type_acc, 4>
                (x, y, ids, fusion, dst, ncols, nrows, stride_row, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
                 stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                 nsamples_dst, ids_stride, stream, x_prewait);
            break;
        case 5:
            launch_mul_mat_vec_f_cuda<T, type_acc, 5>
                (x, y, ids, fusion, dst, ncols, nrows, stride_row, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
                 stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                 nsamples_dst, ids_stride, stream, x_prewait);
            break;
        case 6:
            launch_mul_mat_vec_f_cuda<T, type_acc, 6>
                (x, y, ids, fusion, dst, ncols, nrows, stride_row, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
                 stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                 nsamples_dst, ids_stride, stream, x_prewait);
            break;
        case 7:
            launch_mul_mat_vec_f_cuda<T, type_acc, 7>
                (x, y, ids, fusion, dst, ncols, nrows, stride_row, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
                 stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                 nsamples_dst, ids_stride, stream, x_prewait);
            break;
        case 8:
            launch_mul_mat_vec_f_cuda<T, type_acc, 8>
                (x, y, ids, fusion, dst, ncols, nrows, stride_row, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
                 stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                 nsamples_dst, ids_stride, stream, x_prewait);
            break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

template<typename T>
static void mul_mat_vec_f_cuda(
        const T * x, const float * y, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const int64_t ncols, const int64_t nrows, const int64_t ncols_dst,
        const int64_t stride_row, const int64_t stride_col_y, const int stride_col_dst,
        const int64_t nchannels_x, const int64_t nchannels_y, const int64_t nchannels_dst,
        const int64_t stride_channel_x, const int64_t stride_channel_y, const int64_t stride_channel_dst, const int64_t nsamples_x,
        const int64_t nsamples_dst, const int64_t stride_sample_x, const int64_t stride_sample_y, const int64_t stride_sample_dst,
        const int64_t ids_stride, enum ggml_prec prec, cudaStream_t stream, const bool x_prewait = false) {

    if constexpr(std::is_same_v<T, half>) {
        if (prec == GGML_PREC_DEFAULT) {
            mul_mat_vec_f_cuda_switch_ncols_dst<T, half>
                (x, y, ids, fusion, dst, ncols, nrows, ncols_dst, stride_row, stride_col_y, stride_col_dst,
                nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
                stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream,
                x_prewait);
            return;
        }
    }
    mul_mat_vec_f_cuda_switch_ncols_dst<T, float>
        (x, y, ids, fusion, dst, ncols, nrows, ncols_dst, stride_row, stride_col_y, stride_col_dst,
        nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
        stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream,
        x_prewait);
}

// weights a kernel may read before its PDL wait: in a weights buffer, which no kernel writes
static bool mmvf_x_prewait(const ggml_tensor * t) {
    return t->buffer != nullptr && ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
}

void ggml_cuda_mul_mat_vec_f(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst,
    const ggml_cuda_mm_fusion_args_host * fusion) {
    GGML_ASSERT(        src1->type == GGML_TYPE_F32);
    GGML_ASSERT(!ids ||  ids->type == GGML_TYPE_I32);
    GGML_ASSERT(         dst->type == GGML_TYPE_F32);

    GGML_TENSOR_BINARY_OP_LOCALS;

    const size_t ts_src0 = ggml_type_size(src0->type);
    const size_t ts_src1 = ggml_type_size(src1->type);
    const size_t ts_dst  = ggml_type_size(dst->type);

    GGML_ASSERT(!ids || ne12 <= MMVF_MAX_BATCH_SIZE);
    GGML_ASSERT(ne13 == ne3);

    GGML_ASSERT(        nb00       == ts_src0);
    GGML_ASSERT(        nb10       == ts_src1);
    GGML_ASSERT(!ids || ids->nb[0] == ggml_type_size(ids->type));
    GGML_ASSERT(        nb0        == ts_dst);

    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const enum ggml_prec prec = fast_fp16_available(cc) ? ggml_prec(dst->op_params[0]) : GGML_PREC_F32;

    const float   * src1_d =       (const float   *) src1->data;
    const int32_t *  ids_d = ids ? (const int32_t *)  ids->data : nullptr;
    float         *  dst_d =       (float         *)  dst->data;

    ggml_cuda_mm_fusion_args_device fusion_local{};

    if (fusion) {
        GGML_ASSERT( !ids || dst->ne[2] == 1);
        GGML_ASSERT(  ids || dst->ne[1] == 1);
        if (fusion->x_bias) {
            GGML_ASSERT(fusion->x_bias->type == GGML_TYPE_F32);
            GGML_ASSERT(fusion->x_bias->ne[0] == dst->ne[0]);
            GGML_ASSERT(!ids || fusion->x_bias->ne[1] == src0->ne[2]);
            fusion_local.x_bias = fusion->x_bias->data;
        }
        if (fusion->gate) {
            GGML_ASSERT(fusion->gate->type == src0->type && ggml_are_same_stride(fusion->gate, src0));
            fusion_local.gate = fusion->gate->data;
        }
        if (fusion->gate_bias) {
            GGML_ASSERT(fusion->gate_bias->type == GGML_TYPE_F32);
            GGML_ASSERT(fusion->gate_bias->ne[0] == dst->ne[0]);
            GGML_ASSERT(!ids || fusion->gate_bias->ne[1] == src0->ne[2]);
            fusion_local.gate_bias = fusion->gate_bias->data;
        }
        if (fusion->rms_norm) {
            GGML_ASSERT(fusion->rms_norm->op == GGML_OP_RMS_NORM && fusion->rms_norm->src[0] == src1);
            fusion_local.rms_norm     = true;
            fusion_local.rms_norm_eps = ggml_get_op_params_f32(fusion->rms_norm, 0);
        }
        fusion_local.glu_op = fusion->glu_op;
    }

    const int64_t s01 = src0->nb[1] / ts_src0;
    const int64_t s11 = src1->nb[1] / ts_src1;
    const int64_t s1  =  dst->nb[1] / ts_dst;
    const int64_t s02 = src0->nb[2] / ts_src0;
    const int64_t s12 = src1->nb[2] / ts_src1;
    const int64_t s2  =  dst->nb[2] / ts_dst;
    const int64_t s03 = src0->nb[3] / ts_src0;
    const int64_t s13 = src1->nb[3] / ts_src1;
    const int64_t s3  =  dst->nb[3] / ts_dst;

    // For MUL_MAT_ID the memory layout is different than for MUL_MAT:
    const int64_t ncols_dst          = ids ? ne2  : ne1;
    const int64_t nchannels_y        = ids ? ne11 : ne12;
    const int64_t nchannels_dst      = ids ? ne1  : ne2;
    const int64_t stride_col_dst     = ids ? s2   : s1;
    const int64_t stride_col_y       = ids ? s12  : s11;
    const int64_t stride_channel_dst = ids ? s1   : s2;
    const int64_t stride_channel_y   = ids ? s11  : s12;

    const int64_t ids_stride = ids ? ids->nb[1] / ggml_type_size(ids->type) : 0;

    switch (src0->type) {
        case GGML_TYPE_F32: {
            const float * src0_d = (const float *) src0->data;
            mul_mat_vec_f_cuda(src0_d, src1_d, ids_d, fusion_local, dst_d, ne00, ne01, ncols_dst, s01, stride_col_y, stride_col_dst,
                ne02, nchannels_y, nchannels_dst, s02, stride_channel_y, stride_channel_dst,
                ne03,              ne3,           s03, s13,              s3,                 ids_stride, prec, ctx.stream(),
                mmvf_x_prewait(src0));
        } break;
        case GGML_TYPE_F16: {
            const half * src0_d = (const half *) src0->data;
            mul_mat_vec_f_cuda(src0_d, src1_d, ids_d, fusion_local, dst_d, ne00, ne01, ncols_dst, s01, stride_col_y, stride_col_dst,
                ne02, nchannels_y, nchannels_dst, s02, stride_channel_y, stride_channel_dst,
                ne03,              ne3,           s03, s13,              s3,                 ids_stride, prec, ctx.stream(),
                mmvf_x_prewait(src0));
        } break;
        case GGML_TYPE_BF16: {
            const nv_bfloat16 * src0_d = (const nv_bfloat16 *) src0->data;
            mul_mat_vec_f_cuda(src0_d, src1_d, ids_d, fusion_local, dst_d, ne00, ne01, ncols_dst, s01, stride_col_y, stride_col_dst,
                ne02, nchannels_y, nchannels_dst, s02, stride_channel_y, stride_channel_dst,
                ne03,              ne3,           s03, s13,              s3,                 ids_stride, prec, ctx.stream(),
                mmvf_x_prewait(src0));
        } break;
        default:
            GGML_ABORT("unsupported type: %s", ggml_type_name(src0->type));
    }
}

void ggml_cuda_mul_mat_vec_f_pair(ggml_backend_cuda_context & ctx, const ggml_tensor * src0_a, const ggml_tensor * src0_b,
    const ggml_tensor * src1, ggml_tensor * dst_a, ggml_tensor * dst_b) {
    GGML_ASSERT(ggml_cuda_mmvf_pair_supports(src0_a, src0_b, src1, dst_a, dst_b));

    const size_t ts_src0 = ggml_type_size(src0_a->type);

    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const enum ggml_prec prec = fast_fp16_available(cc) ? ggml_prec(dst_a->op_params[0]) : GGML_PREC_F32;

    const float * src1_d = (const float *) src1->data;
    float       *  dst_d = (float       *) dst_a->data;

    const int64_t ne00 = src0_a->ne[0];
    const int64_t ne01 = src0_a->ne[1];
    const int64_t ne11 = src1->ne[1];
    const int64_t s01  = src0_a->nb[1] / ts_src0;
    const int64_t s11  = src1->nb[1] / sizeof(float);
    const int64_t s1   = dst_a->nb[1] / sizeof(float);
    // channel 1: the second weights and output, at their offsets from the first; both channels read the same src1
    const int64_t s02  = ((const char *) src0_b->data - (const char *) src0_a->data) / (int64_t) ts_src0;
    const int64_t s2   = ((const char *)  dst_b->data - (const char *)  dst_a->data) / (int64_t) sizeof(float);

    const ggml_cuda_mm_fusion_args_device fusion{};
    switch (src0_a->type) {
        case GGML_TYPE_F32: {
            const float * src0_d = (const float *) src0_a->data;
            mul_mat_vec_f_cuda(src0_d, src1_d, nullptr, fusion, dst_d, ne00, ne01, ne11, s01, s11, s1,
                2, 2, 2, s02, 0, s2, 1, 1, 0, 0, 0, 0, prec, ctx.stream(), mmvf_x_prewait(src0_a) && mmvf_x_prewait(src0_b));
        } break;
        case GGML_TYPE_F16: {
            const half * src0_d = (const half *) src0_a->data;
            mul_mat_vec_f_cuda(src0_d, src1_d, nullptr, fusion, dst_d, ne00, ne01, ne11, s01, s11, s1,
                2, 2, 2, s02, 0, s2, 1, 1, 0, 0, 0, 0, prec, ctx.stream(), mmvf_x_prewait(src0_a) && mmvf_x_prewait(src0_b));
        } break;
        case GGML_TYPE_BF16: {
            const nv_bfloat16 * src0_d = (const nv_bfloat16 *) src0_a->data;
            mul_mat_vec_f_cuda(src0_d, src1_d, nullptr, fusion, dst_d, ne00, ne01, ne11, s01, s11, s1,
                2, 2, 2, s02, 0, s2, 1, 1, 0, 0, 0, 0, prec, ctx.stream(), mmvf_x_prewait(src0_a) && mmvf_x_prewait(src0_b));
        } break;
        default:
            GGML_ABORT("unsupported type: %s", ggml_type_name(src0_a->type));
    }
}

void ggml_cuda_op_mul_mat_vec_f(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream) {

    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    const int64_t ne00 = src0->ne[0];
    const int64_t ne10 = src1->ne[0];
    const int64_t ne0  =  dst->ne[0];
    const int64_t row_diff = row_high - row_low;

    const int id = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[id].cc;
    const enum ggml_prec prec = fast_fp16_available(cc) ? ggml_prec(dst->op_params[0]) : GGML_PREC_F32;

    // ggml_cuda_op provides single, contiguous matrices
    const int64_t stride_row         = ne00;
    const int64_t stride_col_y       = ne10;
    const int64_t stride_col_dst     = id == ctx.device ? ne0 : row_diff; // main device has larger memory buffer
    const int64_t nchannels_x        = 1;
    const int64_t nchannels_y        = 1;
    const int64_t nchannels_dst      = 1;
    const int64_t stride_channel_x   = 0;
    const int64_t stride_channel_y   = 0;
    const int64_t stride_channel_dst = 0;
    const int64_t nsamples_x         = 1;
    const int64_t nsamples_dst       = 1;
    const int64_t stride_sample_x    = 0;
    const int64_t stride_sample_y    = 0;
    const int64_t stride_sample_dst  = 0;

    ggml_cuda_mm_fusion_args_device empty{};
    switch (src0->type) {
        case GGML_TYPE_F32: {
            const float * src0_d = (const float *) src0_dd_i;
            mul_mat_vec_f_cuda(src0_d, src1_ddf_i, nullptr, empty, dst_dd_i, ne00, row_diff, src1_ncols, stride_row, stride_col_y, stride_col_dst,
                nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, 0, prec, stream);
        } break;
        case GGML_TYPE_F16: {
            const half * src0_d = (const half *) src0_dd_i;
            mul_mat_vec_f_cuda(src0_d, src1_ddf_i, nullptr, empty, dst_dd_i, ne00, row_diff, src1_ncols, stride_row, stride_col_y, stride_col_dst,
                nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, 0, prec, stream);
        } break;
        case GGML_TYPE_BF16: {
            const nv_bfloat16 * src0_d = (const nv_bfloat16 *) src0_dd_i;
            mul_mat_vec_f_cuda(src0_d, src1_ddf_i, nullptr, empty, dst_dd_i, ne00, row_diff, src1_ncols, stride_row, stride_col_y, stride_col_dst,
                nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, 0, prec, stream);
        } break;
        default:
            GGML_ABORT("unsupported type: %s", ggml_type_name(src0->type));
    }

    GGML_UNUSED_VARS(ctx, src1, dst, src1_ddq_i, src1_ncols, src1_padded_row_size);
}

// The kernel reads every row of an operand in pairs (half2, nv_bfloat162 or float2), so the row length must be even,
// the rows contiguous, and the data address and every stride a multiple of the pair size. An odd stride trips the
// launcher's asserts; a data address offset by an odd number of elements faults with a misaligned address.
static bool ggml_cuda_mmvf_can_read(const ggml_tensor * t) {
    if (t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_F16 && t->type != GGML_TYPE_BF16) {
        return false;
    }

    const size_t ts = ggml_type_size(t->type);
    return t->ne[0] % 2 == 0 && t->nb[0] == ts && ggml_cuda_is_aligned(t, 2*ts);
}

bool ggml_cuda_mmvf_supports(const ggml_tensor * src0, const ggml_tensor * src1) {
    return src1->type == GGML_TYPE_F32 && ggml_cuda_mmvf_can_read(src0) && ggml_cuda_mmvf_can_read(src1);
}

bool ggml_cuda_mmvf_pair_supports(const ggml_tensor * src0_a, const ggml_tensor * src0_b, const ggml_tensor * src1,
        const ggml_tensor * dst_a, const ggml_tensor * dst_b) {
    // b's offset from a in elements, as the kernel's int channel stride
    const auto channel_stride_ok = [](const ggml_tensor * a, const ggml_tensor * b) {
        const int64_t ts   = ggml_type_size(a->type);
        const int64_t offs = (const char *) b->data - (const char *) a->data;
        return a->buffer != nullptr && a->buffer == b->buffer && offs % ts == 0 && offs/ts >= INT_MIN && offs/ts <= INT_MAX;
    };
    return src0_a->ne[2] == 1 && src0_a->ne[3] == 1 && src1->ne[2] == 1 && src1->ne[3] == 1 && src1->ne[1] <= MMVF_MAX_BATCH_SIZE
        && src0_a->type == src0_b->type && ggml_are_same_shape(src0_a, src0_b) && ggml_are_same_stride(src0_a, src0_b)
        && dst_a->type == GGML_TYPE_F32 && dst_b->type == GGML_TYPE_F32 && dst_a->nb[0] == sizeof(float)
        && ggml_are_same_shape(dst_a, dst_b) && ggml_are_same_stride(dst_a, dst_b) && dst_a->op_params[0] == dst_b->op_params[0]
        && ggml_cuda_mmvf_supports(src0_a, src1) && ggml_cuda_mmvf_supports(src0_b, src1)
        && channel_stride_ok(src0_a, src0_b) && channel_stride_ok(dst_a, dst_b);
}

bool ggml_cuda_should_use_mmvf(const ggml_tensor * src0, const ggml_tensor * src1, int cc, int64_t ne11) {
    if (!ggml_cuda_mmvf_supports(src0, src1)) {
        return false;
    }

    const enum ggml_type type    = src0->type;
    const int64_t *      src0_ne = src0->ne;

    switch (type) {
        case GGML_TYPE_F32:
            if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
                if (ampere_mma_available(cc)) {
                    return ne11 <= 3;
                }
                if (cc >= GGML_CUDA_CC_TURING) {
                    return ne11 <= 4;
                }
                return ne11 <= 3;
            } else if (GGML_CUDA_CC_IS_AMD(cc)) {
                if (fp32_mma_hardware_available(cc)) {
                    return ne11 <= 3;
                }
                return ne11 <= 8;
            }
            return ne11 <= 8;
        case GGML_TYPE_F16:
            if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
                const bool src0_small = (src0_ne[1] <= 512 || src0_ne[2]*src0_ne[3] == 1);
                if (ampere_mma_available(cc)) {
                    return src0_small && ne11 == 1;
                }
                if (cc >= GGML_CUDA_CC_ADA_LOVELACE) {
                    return src0_small && ne11 <= 4;
                }
                if (fp16_mma_hardware_available(cc)) {
                    return src0_small && ne11 <= 3;
                }
                return ne11 <= 8;
            } else if (GGML_CUDA_CC_IS_AMD(cc)) {
                if (fp16_mma_hardware_available(cc)) {
                    if (GGML_CUDA_CC_IS_RDNA3(cc)) {
                        return ne11 <= 3;
                    }
                    if (GGML_CUDA_CC_IS_RDNA4(cc)) {
                        return ne11 <= 5;
                    }
                    return ne11 <= 2;
                }
                return ne11 <= 8;
            }
            return ne11 <= 8;
        case GGML_TYPE_BF16:
            if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
                const bool src0_small = (src0_ne[1] <= 512 || src0_ne[2]*src0_ne[3] == 1);
                if (ampere_mma_available(cc)) {
                    return src0_small && ne11 == 1;
                }
                if (cc >= GGML_CUDA_CC_ADA_LOVELACE) {
                    return src0_small && ne11 <= 4;
                }
                if (bf16_mma_hardware_available(cc)) {
                    return src0_small && ne11 <= 3;
                }
                return ne11 <= 8;
            } else if (GGML_CUDA_CC_IS_AMD(cc)) {
                if (bf16_mma_hardware_available(cc)) {
                    return ne11 <= 3;
                }
                return ne11 <= 8;
            }
            return ne11 <= 8;
        default:
            return false;
    }
}
