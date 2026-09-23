//
// Chunked Gated Delta Net prefill: fwdsub intra -> Q@K^T precompute -> tensor-core state+output.
// Tensor-core GEMMs use fp16 operands with fp32 accumulation; gating/state/accum stay fp32.
//
// Port of ggml-org/llama.cpp#26001 by @BLSharda (three-stage pipeline after FLA/vLLM). Fork changes:
//  - NVIDIA nvcuda::wmma only (the PR's second ggml_cuda_mma path for HIP/MUSA is dropped).
//  - q/k/v/g/beta and the output are addressed through explicit strides, so the q/k views into the
//    joint l2_norm output and the v view into the conv output (qwen35.cpp) are read in place.
//  - raw gates (ggml_gated_delta_net_set_raw_gates): stage 1 applies sigmoid(beta) and
//    a[h] * softplus(g + dt_bias[h]) with the recurrent kernel's formulas.
//  - the chunk pipeline may run on a leading token range and write its final state to any slot,
//    so the recurrent kernel can produce the rollback snapshots of the last tokens (see
//    ggml_cuda_op_gated_delta_net_impl).
//
#include "chunk_gated_delta_net.cuh"

#include <cmath>

// Tensor-core kernels need fp16 WMMA with fp32 accumulation; the bodies are stubs elsewhere.
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#include <mma.h>
#  if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
#    define GDN_TC_AVAILABLE
#  endif
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

#if defined(GDN_TC_AVAILABLE)
// Single warp computes C[0:16][c_col:c_col+16] = A[16 x BK] @ B[16 x BK]^T (fp16 inputs, fp32 accum),
// output row-major
template <int BK>
__device__ __forceinline__ void cgdr_gemm_abt_16(const __half * s_a, const __half * s_b, float * s_c,
                                                 const int ldc, const int c_col) {
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> acc;
    nvcuda::wmma::fill_fragment(acc, 0.f);
#pragma unroll
    for (int kt = 0; kt < BK / 16; kt++) {
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __half, nvcuda::wmma::row_major> fa;
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __half, nvcuda::wmma::col_major> fb;
        nvcuda::wmma::load_matrix_sync(fa, s_a + kt * 16, BK);
        nvcuda::wmma::load_matrix_sync(fb, s_b + kt * 16, BK);
        nvcuda::wmma::mma_sync(acc, fa, fb, acc);
    }
    nvcuda::wmma::store_matrix_sync(s_c + c_col, acc, ldc, nvcuda::wmma::mem_row_major);
}

// One warp, v-tile m_off: delta[v][0:BK] = sum_t Vnew[t][v] * Kch[t][k], stored v-major into s_hdelta[v*BK+k].
template <int BK, int BV>
__device__ __forceinline__ void cgdr_gemm_ktv(const __half * s_vnew, const __half * s_kch, float * s_hdelta,
                                              const int m_off) {
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __half, nvcuda::wmma::col_major> a_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __half, nvcuda::wmma::row_major> b_frag;
    nvcuda::wmma::load_matrix_sync(a_frag, s_vnew + m_off, BV);
#pragma unroll
    for (int nk = 0; nk < BK; nk += 16) {
        nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> acc;
        nvcuda::wmma::fill_fragment(acc, 0.f);
        nvcuda::wmma::load_matrix_sync(b_frag, s_kch + nk, BK);
        nvcuda::wmma::mma_sync(acc, a_frag, b_frag, acc);
        nvcuda::wmma::store_matrix_sync(s_hdelta + m_off * BK + nk, acc, BK, nvcuda::wmma::mem_row_major);
    }
}

// One warp, v-tile n_off: O[0:16][v] = sum_t' qk[t][t'] * Vnew[t'][v], stored row-major into s_hdelta[t*BV+v].
template <int BV>
__device__ __forceinline__ void cgdr_gemm_qkv(const __half * s_qk, const __half * s_vnew, float * s_hdelta,
                                              const int n_off) {
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> acc;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __half, nvcuda::wmma::row_major> a_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __half, nvcuda::wmma::row_major> b_frag;
    nvcuda::wmma::fill_fragment(acc, 0.f);
    nvcuda::wmma::load_matrix_sync(a_frag, s_qk, 16);
    nvcuda::wmma::load_matrix_sync(b_frag, s_vnew + n_off, BV);
    nvcuda::wmma::mma_sync(acc, a_frag, b_frag, acc);
    nvcuda::wmma::store_matrix_sync(s_hdelta + n_off, acc, BV, nvcuda::wmma::mem_row_major);
}

// Convert fp32 to fp16 for WMMA operands. Inputs stay small (unit-length q/k, beta in [0,1],
// decay <= 1), so fp16 is safe. The assert is debug-only (Release compiles with -DNDEBUG), so it
// costs nothing in these hot loops. It fires on NaN too, since fabsf(NaN) <= 65504 is false.
__device__ __forceinline__ __half cgdr_to_fp16(const float v) {
    assert(fabsf(v) <= 65504.0f && "chunked GDN: value outside fp16 range (recurrent state blow-up?)");
    return __float2half(v);
}
#endif // defined(GDN_TC_AVAILABLE)

// Intra-chunk forward substitution. Builds the strictly-lower coupling matrix
//   L[t][s] = beta[t] * exp(g_cum[t]-g_cum[s]) * (k[t].k[s])   (s < t, else 0)
// and solves (I + L) x = b twice: b = beta*exp(g_cum)*k -> k_cumdecay, b = beta*v -> v_corr.
// Also emits g_cum. Grid (B*H, num_chunks); 128 threads. Inputs are read through their strides.
// RAW: beta and g arrive pre-activation; sigmoid(beta) and a[h] * softplus(g + dt_bias[h]) are
// applied here with the recurrent kernel's formulas (only this stage reads beta and g).
template <int CS, int BK, bool RAW>
__launch_bounds__(128, 4) __global__ void cgdr_fwdsub_intra_kernel(
    const float * __restrict__ k_in,
    const float * __restrict__ v_in,
    const float * __restrict__ beta,
    const float * __restrict__ g_in,
    const float * __restrict__ raw_dt_bias,  // [H], RAW only
    const float * __restrict__ raw_a,        // [H], RAW only
    float * __restrict__ v_corr,             // (B, H, C, CS, V) output
    float * __restrict__ k_cumdecay,         // (B, H, C, CS, K) output
    float * __restrict__ g_cum_out,          // (B, H, C, CS)    output
    const int       seq_len,                 // tokens this pass covers
    const int       H,
    const int       num_chunks,
    const int       k_dim,
    const int       v_dim,
    const int       num_k_heads,             // q/k head count (H is the v-head count; GQA when smaller)
    const long long sk1, const long long sk2, const long long sk3,  // k strides: head, token, seq
    const long long sv1, const long long sv2, const long long sv3,  // v strides
    const long long sb1, const long long sb2, const long long sb3)  // g/beta strides
{
    static_assert(BK == 128, "cgdr_fwdsub_intra_kernel: BK=128 only");

    // SMEM: s_k[CS][BK+1] fp32 (pad sk=BK+1), s_l[CS][CS] fp32, s_gcum[CS], s_beta[CS].
    constexpr int           sk = BK + 1;
    extern __shared__ float smem[];
    float *                 s_k    = smem;
    float *                 s_l    = s_k + CS * sk;
    float *                 s_gcum = s_l + CS * CS;
    float *                 s_beta = s_gcum + CS;

    const int tid       = threadIdx.x;
    const int pid_bh    = blockIdx.x;
    const int pid_chunk = blockIdx.y;
    const int b         = pid_bh / H;
    const int h         = pid_bh % H;       // v-head
    const int h_k       = h % num_k_heads;  // GQA: v-head -> shared k-head (identity when num_k_heads==H)
    const int t_off     = pid_chunk * CS;

    const float * k_chunk    = k_in + b * sk3 + t_off * sk2 + h_k * sk1;
    const float * v_chunk    = v_in + b * sv3 + t_off * sv2 + h * sv1;
    const float * beta_chunk = beta + b * sb3 + t_off * sb2 + h * sb1;
    const float * g_chunk    = g_in + b * sb3 + t_off * sb2 + h * sb1;

    const long long out_k_base = (long long) pid_bh * num_chunks * CS * k_dim + (long long) pid_chunk * CS * k_dim;
    const long long out_v_base = (long long) pid_bh * num_chunks * CS * v_dim + (long long) pid_chunk * CS * v_dim;
    const long long out_g_base = (long long) pid_bh * num_chunks * CS + (long long) pid_chunk * CS;

    const int valid_cs = min(CS, seq_len - t_off);

    // Step 0: load k, beta, g into SMEM
    for (int i = tid; i < CS * BK; i += 128) {
        int t = i / BK, k = i % BK;
        s_k[t * sk + k] = (t < valid_cs) ? k_chunk[t * sk2 + k] : 0.f;
    }
    for (int i = tid; i < CS; i += 128) {
        float beta_val = 0.f;
        float g_val    = 0.f;
        if (i < valid_cs) {
            beta_val = beta_chunk[i * sb2];
            g_val    = g_chunk[i * sb2];
            if constexpr (RAW) {
                beta_val = 1.0f / (1.0f + expf(-beta_val));
                const float x = g_val + raw_dt_bias[h];
                g_val = raw_a[h] * ((x > 20.0f) ? x : logf(1.0f + expf(x)));
            }
        }
        s_beta[i] = beta_val;
        s_gcum[i] = g_val;
    }
    __syncthreads();

    // Step 1: g_cum prefix sum (serial, thread 0)
    if (tid == 0) {
        float acc = 0.f;
        for (int t = 0; t < CS; t++) {
            acc += s_gcum[t];
            s_gcum[t] = acc;
        }
    }
    __syncthreads();

    // Step 2: build L coupling matrix (exact FP32 scalar dot products)
    for (int idx = tid; idx < CS * CS; idx += 128) {
        const int t = idx / CS, s = idx % CS;
        if (s < t) {
            float kkt = 0.f;
            for (int k = 0; k < BK; k++) {
                kkt += s_k[t * sk + k] * s_k[s * sk + k];
            }
            s_l[idx] = s_beta[t] * __expf(s_gcum[t] - s_gcum[s]) * kkt;
        } else {
            s_l[idx] = 0.f;
        }
    }
    __syncthreads();

    // Step 3: write g_cum_out (k/Q are not staged -- later kernels read them raw)
    for (int i = tid; i < CS; i += 128) {
        g_cum_out[out_g_base + i] = s_gcum[i];
    }

    // Step 4: k_cumdecay via forward substitution
    {
        const int k_col = tid;
        if (k_col < k_dim) {
            float xreg[CS];
            for (int t = 0; t < CS; t++) {
                float xt = s_beta[t] * __expf(s_gcum[t]) * s_k[t * sk + k_col];
                for (int s = 0; s < t; s++) {
                    xt -= s_l[t * CS + s] * xreg[s];
                }
                xreg[t] = xt;
            }
            for (int t = 0; t < CS; t++) {
                k_cumdecay[out_k_base + (long long) t * k_dim + k_col] = xreg[t];
            }
        }
    }
    __syncthreads();

    // Step 5: v_corr via forward substitution (reuses s_k for v tile staging)
    const int num_vt = (v_dim + BK - 1) / BK;
    for (int vt = 0; vt < num_vt; vt++) {
        const int v_off  = vt * BK;
        const int v_cols = min(BK, v_dim - v_off);

        for (int i = tid; i < CS * BK; i += 128) {
            const int t = i / BK, v = i % BK;
            float     val = 0.f;
            if (t < valid_cs && v < v_cols) {
                val = v_chunk[t * sv2 + v_off + v] * s_beta[t];
            }
            s_k[t * sk + v] = val;
        }
        __syncthreads();

        if (tid < v_cols) {
            float     xreg[CS];
            const int v_col = tid;
            for (int t = 0; t < CS; t++) {
                float xt = s_k[t * sk + v_col];
                for (int s = 0; s < t; s++) {
                    xt -= s_l[t * CS + s] * xreg[s];
                }
                xreg[t] = xt;
            }
            for (int t = 0; t < CS; t++) {
                v_corr[out_v_base + (long long) t * v_dim + (v_off + v_col)] = xreg[t];
            }
        }
        __syncthreads();
    }
}

// Masked Q@K^T on tensor cores (fp16 WMMA, one warp per block):
//   qk_buf[i,j] = (Q_ch . K_ch[j]) * exp(g_cum[i] - g_cum[j])   for j <= i, else 0.
// Grid (B*H, num_chunks); 32 threads. Requires CS==16, BK%16==0.
template <int CS, int BK>
__launch_bounds__(WARP_SIZE, 8) __global__ void cgdr_precompute_qk_wmma_kernel(
                                                                        const float * __restrict__ q_raw,
                                                                        const float * __restrict__ k_raw,
                                                                        const float * __restrict__ g_cum,
                                                                        float * __restrict__ qk_buf,
                                                                        const int   num_chunks,
                                                                        const float scale,
                                                                        const int   H,
                                                                        const int   num_k_heads,
                                                                        const int   seq_len,
                                                                        const long long sq1,
                                                                        const long long sq2,
                                                                        const long long sq3) {
#if defined(GDN_TC_AVAILABLE)
    static_assert(CS == 16, "preqk requires CS=16");
    static_assert(BK % 16 == 0, "BK must be a multiple of 16");

    // SMEM: s_q[CS*BK] fp16, s_k[CS*BK] fp16, s_gcum[CS] fp32, s_acc[CS*CS] fp32.
    extern __shared__ char smem_preqk[];
    auto *                 s_q    = reinterpret_cast<__half *>(smem_preqk);
    auto *                 s_k    = reinterpret_cast<__half *>(smem_preqk + CS * BK * sizeof(__half));
    auto *                 s_gcum = reinterpret_cast<float *>(smem_preqk + 2 * CS * BK * sizeof(__half));
    auto *                 s_acc  = s_gcum + CS;

    const int bh  = blockIdx.x;
    const int c   = blockIdx.y;
    const int tid = threadIdx.x;  // 0..31

    const int b_idx = bh / H;
    const int h_idx = bh % H;              // v-head
    const int h_k   = h_idx % num_k_heads; // GQA: v-head -> shared k-head
    const int t_off = c * CS;

    // q and k share strides (asserted by the caller); only their base pointers differ.
    const float * q_chunk = q_raw + b_idx * sq3 + t_off * sq2 + h_k * sq1;
    const float * k_chunk = k_raw + b_idx * sq3 + t_off * sq2 + h_k * sq1;

    if (tid < CS) {
        s_gcum[tid] = g_cum[(bh * num_chunks + c) * CS + tid];
    }

    // Load Q and k (float->fp16). The last chunk may be partial when seq_len is not a multiple of
    // CS -- zero-fill rows past valid_cs to avoid out-of-bounds reads.
    // Q*scale and k are small (unit-length vectors), so fp16 is safe.
    const int valid_cs = min(CS, seq_len - t_off);
    for (int i = tid; i < CS * BK; i += blockDim.x) {
        const int   row = i / BK, col = i % BK;
        const float qv = (row < valid_cs) ? q_chunk[row * sq2 + col] : 0.f;
        const float kv = (row < valid_cs) ? k_chunk[row * sq2 + col] : 0.f;
        s_q[i]         = __float2half(qv * scale);
        s_k[i]         = __float2half(kv);
    }
    __syncthreads();

    // acc = Q @ k^T -> s_acc[CS][CS] (A=Q, B=k, both row-major; see cgdr_gemm_abt_16).
    cgdr_gemm_abt_16<BK>(s_q, s_k, s_acc, CS, 0);
    __syncthreads();

    // Causal mask + cumulative-decay scaling, then write qk_buf.
    float * o_base = qk_buf + (long long) (bh * num_chunks + c) * CS * CS;
    for (int flat = tid; flat < CS * CS; flat += blockDim.x) {
        const int row = flat / CS;
        const int col = flat % CS;
        o_base[flat]  = (col <= row) ? s_acc[flat] * __expf(s_gcum[row] - s_gcum[col]) : 0.f;
    }
#else
    GGML_UNUSED_VARS(q_raw, k_raw, g_cum, qk_buf, num_chunks, scale, H, num_k_heads, seq_len, sq1, sq2, sq3);
    NO_DEVICE_CODE;
#endif // defined(GDN_TC_AVAILABLE)
}

// Recurrent state update + fused output. Grid tiles v by BV; H stays in fp32 h_regs across chunks
// with a per-chunk fp16 copy (s_hfp16) for the WMMA B-operand. Matmuls are m16n16k16 WMMA (fp32
// accum). H SMEM is v-major [BV][BK], matching GGML [bh][v][k]. Grid (B*H, v_dim/BV); NT threads.
template <int CS, int BK, int BV, int NT, int OCC>
__launch_bounds__(NT, OCC) __global__ void cgdr_state_wmma_kernel(
    const float * __restrict__ v_corr,
    const float * __restrict__ k_cumdecay,
    const float * __restrict__ k_raw,   // raw k input, strides sq1/sq2/sq3
    const float * __restrict__ q_raw,   // raw Q input, same strides as k
    const float * __restrict__ g_cum_in,
    const float * __restrict__ qk_buf,  // [B*H, num_chunks, CS, CS] -- fused output input
    float * __restrict__ output,        // [B, T, H, v_dim] GGML layout -- direct output write
    const float * __restrict__ init_state,
    float * __restrict__ final_state,
    const float scale,  // Q scale = 1/sqrt(k_dim)
    const int   num_chunks,
    const int   H,
    const int   num_k_heads,
    const int   v_dim,
    const int   seq_len,
    const long long sq1, const long long sq2, const long long sq3,  // q/k strides: head, token, seq
    const long long so2, const long long so3) {                     // output strides: token, seq
#if defined(GDN_TC_AVAILABLE)
    static_assert(BK == 128, "fp16 state kernel requires BK=128");
    static_assert(CS == 16, "fp16 state kernel requires CS=16");
    static_assert(BV % 16 == 0, "BV must be a multiple of 16");
    static_assert(NT % 32 == 0, "NT must be a multiple of warp size");
    static_assert((CS * BV) % NT == 0, "CS*BV must be divisible by NT");
    static_assert((BK * BV) % NT == 0, "BK*BV must be divisible by NT");
    static_assert(NT / 32 >= BV / 16, "need at least BV/16 warps for WMMA n-tiles");

    // SMEM: s_hfp16[BK*BV] fp16, s_kbuf[CS*BK] fp16 (also s_kch/s_qkb),
    //        s_result[CS*BV] fp32 (also s_vnew fp16), s_gcum[CS] fp32, s_hdelta[BK*BV] fp32.
    constexpr int h_bytes    = BK * BV * (int) sizeof(__half);
    constexpr int kbuf_bytes = CS * BK * (int) sizeof(__half);
    constexpr int res_bytes  = CS * BV * (int) sizeof(float);
    constexpr int gcum_bytes = CS * (int) sizeof(float);

    extern __shared__ char smem_st[];
    __half * s_hfp16  = reinterpret_cast<__half *>(smem_st);
    __half * s_kbuf   = reinterpret_cast<__half *>(smem_st + h_bytes);
    float *  s_result = reinterpret_cast<float *>(smem_st + h_bytes + kbuf_bytes);
    float *  s_gcum   = reinterpret_cast<float *>(smem_st + h_bytes + kbuf_bytes + res_bytes);
    float *  s_hdelta = reinterpret_cast<float *>(smem_st + h_bytes + kbuf_bytes + res_bytes + gcum_bytes);
    __half * s_kch    = s_kbuf;                                // aliases s_kbuf
    __half * s_vnew   = reinterpret_cast<__half *>(s_result);  // aliases s_result

    const int     pid_bh  = blockIdx.x;
    const int     tile_v  = blockIdx.y;
    const int     v_off   = tile_v * BV;

    const int     tid     = threadIdx.x;  // flat 1D block
    const int     warp_id = tid / 32;
    constexpr int ept     = (CS * BV) / NT;
    constexpr int ept_h   = (BK * BV) / NT;
    constexpr int n_tiles = BV / 16;

    const long long bh_off = pid_bh;
    const long long off_k  = bh_off * (long long) num_chunks * CS * BK;
    const long long off_v  = bh_off * (long long) num_chunks * CS * v_dim;

    const float * vcorr_base = v_corr + off_v;
    const float * kcd_base   = k_cumdecay + off_k;
    const float * gcum_base  = g_cum_in + bh_off * (long long) num_chunks * CS;

    const int       b_idx   = pid_bh / H;
    const int       h_idx   = pid_bh % H;            // v-head
    const int       h_k     = h_idx % num_k_heads;   // GQA: v-head -> shared k-head
    float *       out_bh  = output + b_idx * so3 + (long long) h_idx * v_dim + v_off;
    const float * kraw_bh = k_raw + b_idx * sq3 + h_k * sq1;
    const float * qraw_bh = q_raw + b_idx * sq3 + h_k * sq1;

    // FP32 H state in thread registers -- persistent across all chunks (no per-chunk fp16 rounding).
    float h_regs[ept_h];

    // Initialize h_regs from init_state (GDN always has an input recurrent state s0), then prime
    // s_hfp16 for the first chunk's WMMA.
    {
        const long long src_base = bh_off * (long long) v_dim * BK;
        for (int j = 0; j < ept_h; j++) {
            const int idx = tid + j * NT;
            h_regs[j]     = init_state[src_base + (idx / BK + v_off) * BK + (idx % BK)];
            s_hfp16[idx]  = cgdr_to_fp16(h_regs[j]);
        }
    }
    __syncthreads();

    for (int ci = 0; ci < num_chunks; ci++) {
        const float * vcorr_ptr  = vcorr_base + (long long) ci * CS * v_dim;
        const float * kcd_ptr    = kcd_base + (long long) ci * CS * BK;
        const float * gcum_ptr   = gcum_base + (long long) ci * CS;
        const float * kraw_chunk = kraw_bh + (long long) ci * CS * sq2;
        const float * qraw_chunk = qraw_bh + (long long) ci * CS * sq2;
        const int     valid_cs   = min(CS, seq_len - ci * CS);  // < CS on the last chunk if seq_len % CS != 0
        float         vnew_regs[ept];
        float         oi_regs[ept];

        // Step 1a: G_cum -> s_gcum
        for (int i = tid; i < CS; i += NT) {
            s_gcum[i] = gcum_ptr[i];
        }

        // Step 2a: load k_cumdecay into s_kbuf as fp16 (values are small, so fp16 is safe)
        for (int i = tid; i < CS * BK; i += NT) {
            s_kbuf[i] = cgdr_to_fp16(kcd_ptr[i]);
        }

        __syncthreads();

        // Step 2b: s_result[CSxBV] = k_cumdecay @ H  (v_new = u - w*h).
        if (warp_id < n_tiles) {
            const int n_off = warp_id * 16;
            cgdr_gemm_abt_16<BK>(s_kbuf, s_hfp16 + n_off * BK, s_result, BV, n_off);
        }

        __syncthreads();

        // V_new = v_corr - s_result; kept in registers.
        for (int j = 0; j < ept; j++) {
            const int idx   = tid + j * NT;
            const int t_idx = idx / BV;
            const int v_loc = idx % BV;
            const int g_idx = t_idx * v_dim + v_off + v_loc;
            vnew_regs[j]    = vcorr_ptr[g_idx] - s_result[t_idx * BV + v_loc];
        }

        // Step 2.5: q_raw -> s_kbuf, then WMMA for O_inter (o_inter = (q*scale) @ h)
        __syncthreads();

        for (int i = tid; i < CS * BK; i += NT) {
            const int   t = i / BK, k = i % BK;
            const float qv = (t < valid_cs) ? qraw_chunk[t * sq2 + k] : 0.f;
            s_kbuf[i] = cgdr_to_fp16(qv * scale);
        }

        __syncthreads();

        // Step 2.5b: s_result = (q*scale) @ H  (o_inter), same layout as step 2b.
        if (warp_id < n_tiles) {
            const int n_off = warp_id * 16;
            cgdr_gemm_abt_16<BK>(s_kbuf, s_hfp16 + n_off * BK, s_result, BV, n_off);
        }

        __syncthreads();

        // O_inter = s_result x exp(g_cum[t]); kept in registers.
        for (int j = 0; j < ept; j++) {
            const int idx   = tid + j * NT;
            const int t_idx = idx / BV;
            const int v_loc = idx % BV;
            // g_cum is a prefix sum of non-positive log-gates, so it is <= 0 and exp(g_cum) is in
            // (0, 1]; overflow is structurally impossible here. Any Inf/NaN therefore means the
            // gates themselves are corrupt, and is left to propagate rather than be masked.
            oi_regs[j] = s_result[t_idx * BV + v_loc] * __expf(s_gcum[t_idx]);
        }

        // Step 3: k_raw -> s_kch fp16 (row-major [CS][BK], decay-scaled by exp(g_last-g_cum[t]))
        __syncthreads();

        const float g_last = s_gcum[CS - 1];
        for (int i = tid; i < BK * CS; i += NT) {
            const int   k     = i % BK;
            const int   t     = i / BK;
            const float kv    = (t < valid_cs) ? kraw_chunk[t * sq2 + k] : 0.f;
            s_kch[t * BK + k] = cgdr_to_fp16(kv * __expf(g_last - s_gcum[t]));
        }

        // Step 4: vnew_regs -> s_vnew 16-bit (no global read)
        for (int j = 0; j < ept; j++) {
            s_vnew[tid + j * NT] = cgdr_to_fp16(vnew_regs[j]);
        }

        __syncthreads();

        // Step 5: delta[v][k] = sum_t Vnew[t][v] * Kch[t][k], stored v-major
        if (warp_id < n_tiles) {
            cgdr_gemm_ktv<BK, BV>(s_vnew, s_kch, s_hdelta, warp_id * 16);
        }
        __syncthreads();

        // H = exp(g_last)*H + delta  (fp32 accumulation preserved in h_regs)
        const float exp_g = __expf(g_last);
        for (int j = 0; j < ept_h; j++) {
            h_regs[j] = exp_g * h_regs[j] + s_hdelta[tid + j * NT];
        }
        __syncthreads();  // all reads of s_hdelta done before output WMMA overwrites it

        // Refresh s_hfp16 from fp32 h_regs for the next chunk's B-matrix (fp16, operands in range).
        for (int j = 0; j < ept_h; j++) {
            s_hfp16[tid + j * NT] = cgdr_to_fp16(h_regs[j]);
        }

        // output (fp16 WMMA): O[t][v] = O_inter + sum_t' qk[t][t'] * Vnew[t'][v]
        //   qk loaded fp32 from global, downcast to fp16.
        {
            __half *      s_qkb   = reinterpret_cast<__half *>(s_kbuf);  // reuse kbuf: [CS][CS] fp16
            const float * qk_base = qk_buf + (bh_off * (long long) num_chunks + ci) * CS * CS;
            for (int i = tid; i < CS * CS; i += NT) {
                s_qkb[i] = cgdr_to_fp16(qk_base[i]);
            }
            __syncthreads();

            // O_intra[t][v] = sum_t' qk[t][t'] * Vnew[t'][v]  (see cgdr_gemm_qkv).
            if (warp_id < n_tiles) {
                cgdr_gemm_qkv<BV>(s_qkb, s_vnew, s_hdelta, warp_id * 16);
            }
            __syncthreads();

            // O = O_intra (s_hdelta[CS][BV] row-major) + O_inter (oi_regs). Skip padding tokens on
            // the last (partial) chunk -- their output row is past seq_len (out-of-bounds write).
            float * out_chunk = out_bh + (long long) ci * CS * so2;
#pragma unroll
            for (int j = 0; j < ept; j++) {
                const int idx = tid + j * NT;
                const int t_p = idx / BV;
                const int v_p = idx % BV;
                if (t_p < valid_cs) {
                    out_chunk[t_p * so2 + v_p] = s_hdelta[t_p * BV + v_p] + oi_regs[j];
                }
            }
        }

        __syncthreads();  // ensure s_hfp16 refresh + output done before next chunk

    }  // end chunk loop

    // Write final H state to GGML v-major [bh][v][k] (fp32 from h_regs, no conversion loss).
    {
        float * dst = final_state + bh_off * (long long) v_dim * BK;
        for (int j = 0; j < ept_h; j++) {
            const int idx = tid + j * NT;
            dst[(idx / BK + v_off) * BK + (idx % BK)] = h_regs[j];
        }
    }
#else
    // Dispatched on Ampere+ only; body compiled out on older arches.
    GGML_UNUSED_VARS(v_corr, k_cumdecay, k_raw, q_raw, g_cum_in, qk_buf, output, init_state, final_state,
                     scale, num_chunks, H, num_k_heads, v_dim, seq_len, sq1, sq2, sq3, so2, so3);
    NO_DEVICE_CODE;
#endif // defined(GDN_TC_AVAILABLE)
}

// Dynamic SMEM bytes per kernel launch (<<<>>> third arg).
static constexpr size_t cgdr_smem_fwdsub_intra(const int CS, const int BK) {
    return ((size_t) CS * (BK + 1) + (size_t) CS * CS + 2 * (size_t) CS) * sizeof(float);
}

static constexpr size_t cgdr_smem_preqk_wmma(const int CS, const int BK) {
    // SMEM: s_q[CS*BK] fp16, s_k[CS*BK] fp16, s_gcum[CS] fp32, s_acc[CS*CS] fp32.
    return (size_t) 2 * CS * BK * sizeof(__half) + (size_t) (CS + CS * CS) * sizeof(float);
}

static constexpr size_t cgdr_smem_state_wmma(const int CS, const int BK, const int BV) {
    const size_t s_h      = (size_t) BK * BV * sizeof(__half);
    const size_t s_kbuf   = (size_t) CS * BK * sizeof(__half);
    const size_t s_res    = (size_t) CS * BV * sizeof(float);
    const size_t s_gcum   = (size_t) CS * sizeof(float);
    const size_t s_hdelta = (size_t) BK * BV * sizeof(float);
    return s_h + s_kbuf + s_res + s_gcum + s_hdelta;
}

ggml_cuda_gdn_chunked_scratch ggml_cuda_gdn_get_chunked_scratch(const ggml_tensor * dst) {
    GGML_ASSERT(dst->op == GGML_OP_GATED_DELTA_NET);

    const ggml_tensor * src_q = dst->src[0];
    const ggml_tensor * src_v = dst->src[2];

    const int64_t v_dim = src_v->ne[0];
    const int64_t h     = src_v->ne[1];
    const int64_t t     = src_v->ne[2];
    const int64_t b     = src_v->ne[3];
    const int64_t k_dim = src_q->ne[0];

    constexpr int64_t CS = 16;
    // ceil(T/CS); the last chunk may be partial and the kernels guard the padding tokens. Sized for
    // all T tokens; the chunked pass may cover fewer (the last K-1 go to the recurrent kernel).
    const int64_t num_chunks = (t + CS - 1) / CS;
    const int64_t bhcs       = b * h * num_chunks * CS;

    ggml_cuda_gdn_chunked_scratch scratch = {};
    scratch.end = (uintptr_t) dst->data + ggml_nbytes(dst);

    // Every CUDA buffer is 128-aligned (ggml_backend_cuda_buffer_type_get_alignment), so each padded
    // offset is identical whether dst->data is the real pointer or the null it still is at
    // allocation time. That is what lets ggml_cuda_gdn_get_alloc_size call this before allocation.
    auto carve = [&scratch](const int64_t n_floats) {
        scratch.end   = GGML_PAD(scratch.end, 128);
        float * const p = (float *) scratch.end;
        scratch.end  += (uintptr_t) n_floats * sizeof(float);
        return p;
    };

    // Sized exactly: these are written only by scalar, exactly-bounded stores in the fwdsub/preqk
    // kernels (the WMMA store_matrix_sync writes target shared memory, not these).
    scratch.v_corr     = carve(bhcs * v_dim);
    scratch.k_cumdecay = carve(bhcs * k_dim);
    scratch.g_cum      = carve(bhcs);
    scratch.qk         = carve(bhcs * CS);

    return scratch;
}

size_t ggml_cuda_gdn_get_alloc_size(const ggml_tensor * dst) {
    if (!ggml_cuda_gdn_chunked_shape_eligible(dst)) {
        return ggml_nbytes(dst);
    }
    return ggml_cuda_gdn_get_chunked_scratch(dst).end - (uintptr_t) dst->data;
}

void ggml_cuda_gdn_chunked_launch(ggml_backend_cuda_context & ctx, const ggml_tensor * dst,
                                  const ggml_cuda_gdn_chunked_args & a) {
    constexpr int CS = 16;
    constexpr int BK = 128;

    GGML_ASSERT(a.k_dim == BK && a.v_dim == BK && "chunked GDN requires k_dim == v_dim == 128");
    GGML_ASSERT(a.H % a.num_k_heads == 0 && "chunked GDN: v-head count must be a multiple of k-head count");
    GGML_ASSERT(a.n_tokens > 0);

    cudaStream_t stream = ctx.stream();

    // num_chunks = ceil(T/CS): the last chunk may be partial; the kernels guard the padding tokens.
    const int num_chunks = (int) ((a.n_tokens + CS - 1) / CS);
    const int B          = (int) a.n_seqs;
    const int H          = (int) a.H;
    const int T          = (int) a.n_tokens;

    // Scratch lives in the tail of dst's own allocation; see ggml_cuda_gdn_get_chunked_scratch.
    const ggml_cuda_gdn_chunked_scratch scratch = ggml_cuda_gdn_get_chunked_scratch(dst);

    // Stage 1 -- intra pass: exact FP32 forward substitution -> v_corr, k_cumdecay, g_cum.
    {
        constexpr size_t fs_smem = cgdr_smem_fwdsub_intra(CS, BK);
        const dim3       intra_grid(B * H, num_chunks, 1);
        if (a.raw_dt_bias != nullptr) {
            cgdr_fwdsub_intra_kernel<CS, BK, true><<<intra_grid, 128, fs_smem, stream>>>(
                a.k, a.v, a.beta, a.g, a.raw_dt_bias, a.raw_a, scratch.v_corr, scratch.k_cumdecay, scratch.g_cum,
                T, H, num_chunks, (int) a.k_dim, (int) a.v_dim, (int) a.num_k_heads,
                a.sq1, a.sq2, a.sq3, a.sv1, a.sv2, a.sv3, a.sb1, a.sb2, a.sb3);
        } else {
            cgdr_fwdsub_intra_kernel<CS, BK, false><<<intra_grid, 128, fs_smem, stream>>>(
                a.k, a.v, a.beta, a.g, nullptr, nullptr, scratch.v_corr, scratch.k_cumdecay, scratch.g_cum,
                T, H, num_chunks, (int) a.k_dim, (int) a.v_dim, (int) a.num_k_heads,
                a.sq1, a.sq2, a.sq3, a.sv1, a.sv2, a.sv3, a.sb1, a.sb2, a.sb3);
        }
    }
    CUDA_CHECK(cudaGetLastError());

    // Stage 2 -- preqk pass: masked Q@K^T (one warp per block).
    {
        constexpr size_t qk_smem = cgdr_smem_preqk_wmma(CS, BK);
        const dim3       qk_grid(B * H, num_chunks, 1);
        cgdr_precompute_qk_wmma_kernel<CS, BK><<<qk_grid, WARP_SIZE, qk_smem, stream>>>(
            a.q, a.k, scratch.g_cum, scratch.qk, num_chunks, a.scale, H, (int) a.num_k_heads, T,
            a.sq1, a.sq2, a.sq3);
    }
    CUDA_CHECK(cudaGetLastError());

    // Stage 3 -- state+output pass: WMMA tensor cores, fixed tile (BV=32/NT=256/OCC=4). ~30 KB
    // dynamic SMEM, under the 48 KB default, so no cudaFuncAttribute opt-in needed.
    {
        constexpr int    BV = 32, NT = 256, OCC = 4;
        constexpr size_t st_smem = cgdr_smem_state_wmma(CS, BK, BV);
        const dim3       state_grid(B * H, (int) a.v_dim / BV, 1);
        cgdr_state_wmma_kernel<CS, BK, BV, NT, OCC><<<state_grid, NT, st_smem, stream>>>(
            scratch.v_corr, scratch.k_cumdecay, a.k, a.q, scratch.g_cum, scratch.qk, a.out,
            a.state_in, a.state_out, a.scale, num_chunks, H, (int) a.num_k_heads, (int) a.v_dim, T,
            a.sq1, a.sq2, a.sq3, a.so2, a.so3);
    }
    CUDA_CHECK(cudaGetLastError());
}
