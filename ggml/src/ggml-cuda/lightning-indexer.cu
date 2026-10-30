#include "common.cuh"
#include "lightning-indexer.cuh"
#include "fattn-common.cuh"
#include "convert.cuh"
#include "mma.cuh"
#include "vecdotq.cuh"

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#if defined(TURING_MMA_AVAILABLE)

typedef union {
    int2 i2;
    half2 h2[2];
} half4;

// TODO add support for AMD cards via rocWMMA
#include <mma.h>
namespace wmma = nvcuda::wmma;

template <int WARPS_PER_BLOCK, int K_VECS_PER_BLOCK, int64_t N_EMBD, int64_t N_HEAD, ggml_type TYPE_K>
static __global__ void lightning_indexer_kernel_wmma(
        const float * Q, const char * K, const float * W, const half * M, float * dst,
        int64_t n_stream, int64_t n_batch, int64_t n_kv,
        size_t nb1, size_t nb2, size_t nb3,
        size_t nbq1, size_t nbq2, size_t nbq3,
        size_t nbk1, size_t nbk2, size_t nbk3,
        size_t nbw1, size_t nbw2, size_t nbw3,
        size_t nbm1, size_t nbm2, size_t nbm3,
        int64_t nem3
    ) {

    constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * WARP_SIZE;
    constexpr int HEADS_PER_INNER_LOOP = 8;
    constexpr int K_EMBD_PER_INNER_LOOP = 16;
    constexpr int N_EMBD_PADDED = N_EMBD + 8;

    const int i_batch  = blockIdx.y;
    const int i_stream = blockIdx.z;
    const int i_warp   = threadIdx.y;
    const int i_lane   = threadIdx.x;
    const int tid      = i_warp * WARP_SIZE + i_lane;

    // each block processes K_VECS_PER_BLOCK K vectors
    const int start_kv = blockIdx.x * K_VECS_PER_BLOCK;

    const char  * q_base = (const char  *)                 Q + i_batch*nbq2 + i_stream*nbq3;
    const float * w_base = (const float *) ((const char *) W + i_batch*nbw1 + i_stream*nbw3);

    // phase 1 - load weights and first Q tile to shared memory

    __shared__ float w_shared[N_HEAD];
    __shared__ int2  q_shared_h[HEADS_PER_INNER_LOOP][N_EMBD_PADDED / 4];

    if (tid < N_HEAD) {
        w_shared[tid] = w_base[tid];
    }

    // total number of half4 elements in HEADS_PER_INNER_LOOP x N_EMBD Q tile
    constexpr int N_Q_TILE = HEADS_PER_INNER_LOOP * (N_EMBD / 4);
    // number of registers needed in each thread to store Q tile in thread block
    constexpr int N_Q_NEXT = (N_Q_TILE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

#pragma unroll
    for (int i_q = tid; i_q < N_Q_TILE; i_q += THREADS_PER_BLOCK) {
        const int i_head = i_q / (N_EMBD / 4);
        const int i_embd = i_q % (N_EMBD / 4);
        const float4 q = *(const float4 *) (q_base + i_head*nbq1 + i_embd*sizeof(float4));
        half4 q_packed;
        q_packed.h2[0] = __float22half2_rn(make_float2(q.x, q.y));
        q_packed.h2[1] = __float22half2_rn(make_float2(q.z, q.w));
        q_shared_h[i_head][i_embd] = q_packed.i2;
    }

    // phase 2 - load (and dequantize if needed) K to shared mem

    __shared__ half2 k_shared_h[K_VECS_PER_BLOCK][N_EMBD_PADDED / 4][2];

    constexpr int n_k = K_VECS_PER_BLOCK * (N_EMBD / 4);

    if constexpr (TYPE_K == GGML_TYPE_F16) {
#pragma unroll
        for (int i_k = tid; i_k < n_k; i_k += THREADS_PER_BLOCK) {
            const int i_k_vec = i_k / (N_EMBD / 4);
            const int i_embd = i_k % (N_EMBD / 4);
            const int i_kv = start_kv + i_k_vec;
            if (i_kv < n_kv) {
                const int2 * k_base = (const int2 *) ((const char *) K + i_kv*nbk2 + i_stream*nbk3);
                *(int2*) &k_shared_h[i_k_vec][i_embd] = k_base[i_embd];
            } else {
                *(int2*) &k_shared_h[i_k_vec][i_embd] = make_int2(0, 0);
            }
        }
    } else {
        constexpr dequantize_V_t dequantize_k = get_dequantize_V<TYPE_K, half, 4>();
#pragma unroll
        for (int i_k = tid; i_k < n_k; i_k += THREADS_PER_BLOCK) {
            const int i_k_vec = i_k / (N_EMBD / 4);
            const int i_embd = i_k % (N_EMBD / 4);
            const int i_kv = start_kv + i_k_vec;
            if (i_kv < n_kv) {
                const void * k_base = (const void *) ((const char *) K + i_kv*nbk2 + i_stream*nbk3);
                dequantize_k(k_base, &k_shared_h[i_k_vec][i_embd][0], i_embd * 4);
            } else {
                *(int2*) &k_shared_h[i_k_vec][i_embd] = make_int2(0, 0);
            }
        }
    }

    __syncthreads();

    // phase 3 - calculate lightning indexer scores

    __shared__ float qk_shared[WARPS_PER_BLOCK][HEADS_PER_INNER_LOOP][K_VECS_PER_BLOCK];

    // load K fragment
    wmma::fragment<wmma::matrix_b, HEADS_PER_INNER_LOOP, K_VECS_PER_BLOCK, K_EMBD_PER_INNER_LOOP, half, wmma::col_major> frag_k;
    wmma::load_matrix_sync(frag_k, (half*) &k_shared_h[0][i_warp * K_EMBD_PER_INNER_LOOP / 4], N_EMBD_PADDED);

    float score_k = 0.0f;

    for (int i_head_0 = 0; i_head_0 < N_HEAD; i_head_0 += HEADS_PER_INNER_LOOP) {
        const int i_head_next = i_head_0 + HEADS_PER_INNER_LOOP;

        // we don't use accumulator for anything, fill it with zeros
        wmma::fragment<wmma::accumulator, HEADS_PER_INNER_LOOP, K_VECS_PER_BLOCK, K_EMBD_PER_INNER_LOOP, float> frag_acc;
        wmma::fill_fragment(frag_acc, 0.0f);

        // load Q fragment
        wmma::fragment<wmma::matrix_a, HEADS_PER_INNER_LOOP, K_VECS_PER_BLOCK, K_EMBD_PER_INNER_LOOP, half, wmma::row_major> frag_q;
        wmma::load_matrix_sync(frag_q, (half*) &q_shared_h[0][i_warp * K_EMBD_PER_INNER_LOOP / 4], N_EMBD_PADDED);

        // preload next Q tile to registers during matrix multiplication
        float4 q_next[N_Q_NEXT];

        if (i_head_next < N_HEAD) {
#pragma unroll
            for (int i_q = tid, i_q_next = 0; i_q < N_Q_TILE; i_q += THREADS_PER_BLOCK) {
                const int i_head = i_head_next + i_q / (N_EMBD / 4);
                const int i_embd =               i_q % (N_EMBD / 4);
                q_next[i_q_next++] = *(const float4 *) (q_base + i_head*nbq1 + i_embd*sizeof(float4));
            }
        }

        // perform matrix multiplication
        wmma::mma_sync(frag_acc, frag_q, frag_k, frag_acc);
        wmma::store_matrix_sync((float*) &qk_shared[i_warp][0][0], frag_acc, K_VECS_PER_BLOCK, wmma::mem_row_major);

        // make sure all threads finished using q_shared_h so we can store next tile
        __syncthreads();

        // write preloaded Q tile to shared memory
        if (i_head_next < N_HEAD) {
#pragma unroll
            for (int i_q = tid, i_q_next = 0; i_q < N_Q_TILE; i_q += THREADS_PER_BLOCK) {
                const int i_head = i_q / (N_EMBD / 4);
                const int i_embd = i_q % (N_EMBD / 4);
                half4 q_packed;
                q_packed.h2[0] = __float22half2_rn(make_float2(q_next[i_q_next].x, q_next[i_q_next].y));
                q_packed.h2[1] = __float22half2_rn(make_float2(q_next[i_q_next].z, q_next[i_q_next].w));
                q_shared_h[i_head][i_embd] = q_packed.i2;
                ++i_q_next;
            }
        }

        // accumulate QK multiplication results from all block warps
        // (there are 256 threads in block and 256 matmul outputs)
        // TODO it will break if WARP_SIZE is not 32
        const int h = tid / K_VECS_PER_BLOCK;
        const int k = tid % K_VECS_PER_BLOCK;
        const float w_val = w_shared[i_head_0 + h];

        float sum = 0.0f;
#pragma unroll
        for (int w = 0; w < WARPS_PER_BLOCK; ++w) {
            sum += qk_shared[w][h][k];
        }

        // ReLU, weight
        sum = sum > 0.0f ? sum : 0.0f;
        sum *= w_val;

        // wait until qk_shared[0] is no longer used
        __syncthreads();

        // reuse qk_shared[0] for storing partial results
        qk_shared[0][h][k] = sum;

        // wait until all threads write their results
        __syncthreads();

        // accumulate result over heads
        if (tid < K_VECS_PER_BLOCK) {
#pragma unroll
            for (int i_head = 0; i_head < HEADS_PER_INNER_LOOP; ++i_head) {
                score_k += qk_shared[0][i_head][tid];
            }
        }

        // make sure all threads finished using qk_shared
        __syncthreads();
    }

    // phase 4 - store output to VRAM

    if (tid < K_VECS_PER_BLOCK) {
        const int i_kv = start_kv + tid;
        if (i_kv < n_kv) {
            const half * m_base = (const half *) ((const char *) M + i_batch*nbm1 + (i_stream%nem3)*nbm3);
            float * dst_base = (float *) ((char *) dst + i_batch*nb1 + i_stream*nb3);
            dst_base[i_kv] = score_k + __half2float(m_base[i_kv]);
        }
    }
}

#else // defined(TURING_MMA_AVAILABLE)

template <int WARPS_PER_BLOCK, int K_VECS_PER_BLOCK, int64_t N_EMBD, int64_t N_HEAD, ggml_type TYPE_K>
static __global__ void lightning_indexer_kernel_wmma(
        const float * Q, const char * K, const float * W, const half * M, float * dst,
        int64_t n_stream, int64_t n_batch, int64_t n_kv,
        size_t nb1, size_t nb2, size_t nb3,
        size_t nbq1, size_t nbq2, size_t nbq3,
        size_t nbk1, size_t nbk2, size_t nbk3,
        size_t nbw1, size_t nbw2, size_t nbw3,
        size_t nbm1, size_t nbm2, size_t nbm3,
        int64_t nem3
    ) {
    GGML_UNUSED_VARS(Q, K, W, M, dst,
        n_stream, n_batch, n_kv,
        nb1, nb2, nb3,
        nbq1, nbq2, nbq3,
        nbk1, nbk2, nbk3,
        nbw1, nbw2, nbw3,
        nem3);
    NO_DEVICE_CODE;
}

#endif // defined(TURING_MMA_AVAILABLE)
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

// TODO there is one ugly assumption used in this kernel - that WARP_SIZE is equal to 32
// thanks to that one warp operating on float4 processes whole indexer K/Q vectors
// 32 * 4 = 128 (N_EMBD)

template <int WARPS_PER_BLOCK, int K_VECS_PER_BLOCK, int64_t N_EMBD, int64_t N_HEAD, ggml_type TYPE_K>
static __global__ void lightning_indexer_kernel_vec(
        const float * Q, const char * K, const float * W, const half * M, float * dst,
        int64_t n_stream, int64_t n_batch, int64_t n_kv,
        size_t nb1, size_t nb2, size_t nb3,
        size_t nbq1, size_t nbq2, size_t nbq3,
        size_t nbk1, size_t nbk2, size_t nbk3,
        size_t nbw1, size_t nbw2, size_t nbw3,
        size_t nbm1, size_t nbm2, size_t nbm3,
        int64_t nem3,
        const int32_t * rows, int64_t s_rows // ggml_lightning_indexer_rows: key i of stream s at K's row rows[s*s_rows + i]
    ) {

    constexpr int K_VECS_PER_WARP = K_VECS_PER_BLOCK / WARPS_PER_BLOCK;
    constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * WARP_SIZE;

    const int i_batch  = blockIdx.y;
    const int i_stream = blockIdx.z;
    const int i_warp   = threadIdx.y;
    const int i_lane   = threadIdx.x;
    const int tid      = i_warp * WARP_SIZE + i_lane;

    // each warp processes K_VECS_PER_WARP K vectors
    const int start_kv_block = blockIdx.x * K_VECS_PER_BLOCK;
    const int start_kv = start_kv_block + i_warp * K_VECS_PER_WARP;

    const char  * q_base = (const char  *)                 Q + i_batch*nbq2 + i_stream*nbq3;
    const float * w_base = (const float *) ((const char *) W + i_batch*nbw1 + i_stream*nbw3);
    const int32_t * rows_s = rows ? rows + i_stream*s_rows : nullptr;

    // phase 1 - load (and dequantize if needed) K to registers

    float4 k_reg_f[K_VECS_PER_WARP];

    if constexpr (TYPE_K == GGML_TYPE_F32) {
        // direct copy of float4
#pragma unroll
        for (int k = 0; k < K_VECS_PER_WARP; ++k) {
            int i_kv = start_kv + k;
            if (i_kv < n_kv) {
                const int64_t i_row = rows_s ? rows_s[i_kv] : i_kv;
                const float4 * k_base = (const float4 *) ((const char *) K + i_row*nbk2 + i_stream*nbk3);
                k_reg_f[k] = k_base[i_lane];
            } else {
                k_reg_f[k] = make_float4(0, 0, 0, 0);
            }
        }
    } else {
        // dequantize remaining types to float
        constexpr dequantize_V_t dequantize_k = get_dequantize_V<TYPE_K, float, 4>();
#pragma unroll
        for (int k = 0; k < K_VECS_PER_WARP; ++k) {
            int i_kv = start_kv + k;
            if (i_kv < n_kv) {
                const int64_t i_row = rows_s ? rows_s[i_kv] : i_kv;
                const void * k_base = (const void *) ((const char *) K + i_row*nbk2 + i_stream*nbk3);
                dequantize_k(k_base, &k_reg_f[k], i_lane * 4);
            } else {
                k_reg_f[k] = make_float4(0, 0, 0, 0);
            }
        }
    }

    float score_k[K_VECS_PER_WARP] = { 0.0f };

    // load weights and Q only for N_HEAD_INNER heads at once to reduce shared memory usage
    constexpr int N_HEAD_INNER = N_HEAD / 4;

    for (int i_head_0 = 0; i_head_0 < N_HEAD; i_head_0 += N_HEAD_INNER) {
        // phase 2 - load weights and Q to shared memory

        __shared__ float  w_shared[N_HEAD_INNER];
        __shared__ float4 q_shared_f[N_HEAD_INNER][N_EMBD / 4];

        if (tid < N_HEAD_INNER) {
            w_shared[tid] = w_base[i_head_0 + tid];
        }

        constexpr int n_q = N_HEAD_INNER * (N_EMBD / 4);
#pragma unroll
        for (int i_q = tid; i_q < n_q; i_q += THREADS_PER_BLOCK) {
            const int i_head_inner = i_q / (N_EMBD / 4);
            const int i_head = i_head_0 + i_head_inner;
            const int i_embd = i_q % (N_EMBD / 4);
            q_shared_f[i_head_inner][i_embd] = *(const float4 *) (q_base + i_head*nbq1 + i_embd*sizeof(float4));
        }

        __syncthreads();

        // phase 3 - calculate lightning indexer scores

        for (int i_head_inner = 0; i_head_inner < N_HEAD_INNER; ++i_head_inner) {
            const float w_val = w_shared[i_head_inner];
            float qk[K_VECS_PER_WARP] = { 0.0f };

            // dot product of floats
            const float4 q_vec = q_shared_f[i_head_inner][i_lane];

#pragma unroll
            for (int k = 0; k < K_VECS_PER_WARP; ++k) {
                ggml_cuda_mad(qk[k], q_vec.x, k_reg_f[k].x);
                ggml_cuda_mad(qk[k], q_vec.y, k_reg_f[k].y);
                ggml_cuda_mad(qk[k], q_vec.z, k_reg_f[k].z);
                ggml_cuda_mad(qk[k], q_vec.w, k_reg_f[k].w);
            }

#pragma unroll
            for (int k = 0; k < K_VECS_PER_WARP; ++k) {
                float sum = warp_reduce_sum(qk[k]);

                // ReLU, weight
                if (i_lane == 0) {
                    sum = (sum > 0.0f) ? sum : 0.0f;
                    score_k[k] += sum * w_val;
                }
            }
        }

        __syncthreads();
    }

    // phase 4 - store outputs to shared memory

    __shared__ float dst_shared[K_VECS_PER_BLOCK];

    if (i_lane == 0) {
#pragma unroll
        for (int k = 0; k < K_VECS_PER_WARP; ++k) {
            dst_shared[i_warp * K_VECS_PER_WARP + k] = score_k[k];
        }
    }

    __syncthreads();

    // phase 5 - write from shared memory to VRAM in coalesced manner

    if (tid < K_VECS_PER_BLOCK) {
        int i_kv = start_kv_block + tid;
        if (i_kv < n_kv) {
            const half * m_base = (const half *) ((const char *) M + i_batch*nbm1 + (i_stream%nem3)*nbm3);
            float * dst_base = (float *) ((char *) dst + i_batch*nb1 + i_stream*nb3);
            dst_base[i_kv] = dst_shared[tid] + __half2float(m_base[i_kv]);
        }
    }
}

// The vector kernel's scores bit for bit, a key to a quad of lanes: lane s of a quad holds the dims lanes 4j + s
// (j = 0..7) hold in the vector kernel, 16j + 4s to 16j + 4s + 3, and adds its 8 products in registers in the order
// warp_reduce_sum's xor 16, 8 and 4 add them (the bits of j), then xor 2 and 1 across the quad: 2 shuffles a key and head
// where the vector kernel takes 5 and serves one key with them. A block holds all heads' q at once.
template <int WARPS_PER_BLOCK, int KEYS_PER_QUAD, int64_t N_EMBD, int64_t N_HEAD, ggml_type TYPE_K>
static __global__ void lightning_indexer_kernel_quad(
        const float * Q, const char * K, const float * W, const half * M, float * dst,
        int64_t n_stream, int64_t n_batch, int64_t n_kv,
        size_t nb1, size_t nb2, size_t nb3,
        size_t nbq1, size_t nbq2, size_t nbq3,
        size_t nbk1, size_t nbk2, size_t nbk3,
        size_t nbw1, size_t nbw2, size_t nbw3,
        size_t nbm1, size_t nbm2, size_t nbm3,
        int64_t nem3,
        const int32_t * rows, int64_t s_rows
    ) {
    static_assert(N_EMBD == 4*WARP_SIZE, "a lane of the vector kernel holds 4 dims");

    constexpr int N_J               = N_EMBD / 16; // float4s a lane holds of a key
    constexpr int QUADS_PER_WARP    = WARP_SIZE / 4;
    constexpr int KEYS_PER_WARP     = QUADS_PER_WARP * KEYS_PER_QUAD;
    constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * WARP_SIZE;
    static_assert(THREADS_PER_BLOCK >= N_HEAD, "a thread loads a head's weight");

    const int i_batch  = blockIdx.y;
    const int i_stream = blockIdx.z;
    const int i_warp   = threadIdx.y;
    const int i_lane   = threadIdx.x;
    const int tid      = i_warp * WARP_SIZE + i_lane;
    const int i_quad   = i_lane / 4;
    const int s        = i_lane % 4;

    // key c of a quad: consecutive quads hold consecutive keys, so a warp writes its scores contiguously
    const int start_kv = (blockIdx.x * WARPS_PER_BLOCK + i_warp) * KEYS_PER_WARP + i_quad;

    const char  * q_base = (const char  *)                 Q + i_batch*nbq2 + i_stream*nbq3;
    const float * w_base = (const float *) ((const char *) W + i_batch*nbw1 + i_stream*nbw3);
    const int32_t * rows_s = rows ? rows + i_stream*s_rows : nullptr;

    float4 k_reg_f[KEYS_PER_QUAD][N_J];

#pragma unroll
    for (int c = 0; c < KEYS_PER_QUAD; ++c) {
        const int i_kv = start_kv + c*QUADS_PER_WARP;
        if (i_kv < n_kv) {
            const int64_t i_row = rows_s ? rows_s[i_kv] : i_kv;
            const char * k_base = K + i_row*nbk2 + i_stream*nbk3;
            if constexpr (TYPE_K == GGML_TYPE_F32) {
#pragma unroll
                for (int j = 0; j < N_J; ++j) {
                    k_reg_f[c][j] = ((const float4 *) k_base)[4*j + s];
                }
            } else {
                constexpr dequantize_V_t dequantize_k = get_dequantize_V<TYPE_K, float, 4>();
#pragma unroll
                for (int j = 0; j < N_J; ++j) {
                    dequantize_k(k_base, &k_reg_f[c][j], 16*j + 4*s);
                }
            }
        } else {
#pragma unroll
            for (int j = 0; j < N_J; ++j) {
                k_reg_f[c][j] = make_float4(0, 0, 0, 0);
            }
        }
    }

    __shared__ float  w_shared[N_HEAD];
    __shared__ float4 q_shared_f[N_HEAD][N_EMBD / 4];

    if (tid < N_HEAD) {
        w_shared[tid] = w_base[tid];
    }
#pragma unroll
    for (int i_q = tid; i_q < N_HEAD * (N_EMBD / 4); i_q += THREADS_PER_BLOCK) {
        const int i_head = i_q / (N_EMBD / 4);
        const int i_embd = i_q % (N_EMBD / 4);
        q_shared_f[i_head][i_embd] = *(const float4 *) (q_base + i_head*nbq1 + i_embd*sizeof(float4));
    }

    __syncthreads();

    float score_k[KEYS_PER_QUAD] = { 0.0f };

    for (int i_head = 0; i_head < N_HEAD; ++i_head) {
        const float w_val = w_shared[i_head];
        float qk[KEYS_PER_QUAD][N_J];

#pragma unroll
        for (int j = 0; j < N_J; ++j) {
            const float4 q_vec = q_shared_f[i_head][4*j + s];
#pragma unroll
            for (int c = 0; c < KEYS_PER_QUAD; ++c) {
                qk[c][j] = 0.0f;
                ggml_cuda_mad(qk[c][j], q_vec.x, k_reg_f[c][j].x);
                ggml_cuda_mad(qk[c][j], q_vec.y, k_reg_f[c][j].y);
                ggml_cuda_mad(qk[c][j], q_vec.z, k_reg_f[c][j].z);
                ggml_cuda_mad(qk[c][j], q_vec.w, k_reg_f[c][j].w);
            }
        }

#pragma unroll
        for (int c = 0; c < KEYS_PER_QUAD; ++c) {
            static_assert(N_J == 8, "the in-register levels are xor 16, 8 and 4");
            const float x16_0 = qk[c][0] + qk[c][4];
            const float x16_1 = qk[c][1] + qk[c][5];
            const float x16_2 = qk[c][2] + qk[c][6];
            const float x16_3 = qk[c][3] + qk[c][7];
            const float x8_0  = x16_0 + x16_2;
            const float x8_1  = x16_1 + x16_3;
            float sum = x8_0 + x8_1;
            sum += __shfl_xor_sync(0xffffffff, sum, 2, WARP_SIZE);
            sum += __shfl_xor_sync(0xffffffff, sum, 1, WARP_SIZE);

            // ReLU, weight
            sum = (sum > 0.0f) ? sum : 0.0f;
            score_k[c] += sum * w_val;
        }
    }

    if (s == 0) {
        const half * m_base = (const half *) ((const char *) M + i_batch*nbm1 + (i_stream%nem3)*nbm3);
        float * dst_base = (float *) ((char *) dst + i_batch*nb1 + i_stream*nb3);
#pragma unroll
        for (int c = 0; c < KEYS_PER_QUAD; ++c) {
            const int i_kv = start_kv + c*QUADS_PER_WARP;
            if (i_kv < n_kv) {
                dst_base[i_kv] = score_k[c] + __half2float(m_base[i_kv]);
            }
        }
    }
}

// One level of an octet's cross-lane tree over the N keys each lane holds: the lanes with bit m keep the upper half of
// the keys and send the lower half, their partners the reverse, and each adds its partner's value of a key it keeps to
// its own, the sum the butterfly's level makes for that key. One shuffle a kept key where the butterfly takes one a key.
template <int N>
static __device__ __forceinline__ void lightning_indexer_oct_level(float * part, const int m, const bool hi) {
#pragma unroll
    for (int i = 0; i < N/2; ++i) {
        const float send = hi ? part[i]       : part[i + N/2];
        const float keep = hi ? part[i + N/2] : part[i];
        part[i] = keep + __shfl_xor_sync(0xffffffff, send, m, WARP_SIZE);
    }
}

// The vector kernel's scores bit for bit, a key to an octet of lanes: lane t of an octet holds the dims lanes t + 8j
// (j = 0..3) hold in the vector kernel, 32j + 4t to 32j + 4t + 3, and adds its 4 products in registers in the order
// warp_reduce_sum's xor 16 and 8 add them (the bits of j), then xor 4, 2 and 1 across the octet, each level handing half
// of the keys a lane holds to its partner (lightning_indexer_oct_level) until a lane holds one. Every q value a lane
// reads from shared memory serves the KEYS_PER_OCT keys it holds, twice the quad kernel's 2 at the same registers: that
// kernel is bound by q's delivery from shared memory (L1 at 90 %, DRAM at 2 %, GLM-5.3's pooled keys at 64K cells).
template <int WARPS_PER_BLOCK, int KEYS_PER_OCT, int64_t N_EMBD, int64_t N_HEAD, ggml_type TYPE_K>
static __global__ void lightning_indexer_kernel_oct(
        const float * Q, const char * K, const float * W, const half * M, float * dst,
        int64_t n_stream, int64_t n_batch, int64_t n_kv,
        size_t nb1, size_t nb2, size_t nb3,
        size_t nbq1, size_t nbq2, size_t nbq3,
        size_t nbk1, size_t nbk2, size_t nbk3,
        size_t nbw1, size_t nbw2, size_t nbw3,
        size_t nbm1, size_t nbm2, size_t nbm3,
        int64_t nem3,
        const int32_t * rows, int64_t s_rows
    ) {
    static_assert(N_EMBD == 4*WARP_SIZE, "a lane of the vector kernel holds 4 dims");
    static_assert(KEYS_PER_OCT == 1 || KEYS_PER_OCT == 2 || KEYS_PER_OCT == 4 || KEYS_PER_OCT == 8, "halved down to one at 3 levels");

    constexpr int N_J               = N_EMBD / 32; // float4s a lane holds of a key
    constexpr int OCTS_PER_WARP     = WARP_SIZE / 8;
    constexpr int KEYS_PER_WARP     = OCTS_PER_WARP * KEYS_PER_OCT;
    constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * WARP_SIZE;
    static_assert(THREADS_PER_BLOCK >= N_HEAD, "a thread loads a head's weight");

    const int i_batch  = blockIdx.y;
    const int i_stream = blockIdx.z;
    const int i_warp   = threadIdx.y;
    const int i_lane   = threadIdx.x;
    const int tid      = i_warp * WARP_SIZE + i_lane;
    const int i_oct    = i_lane / 8;
    const int t        = i_lane % 8;

    // key c of an octet: consecutive octets hold consecutive keys, so a warp writes its scores contiguously
    const int start_kv = (blockIdx.x * WARPS_PER_BLOCK + i_warp) * KEYS_PER_WARP + i_oct;

    const char  * q_base = (const char  *)                 Q + i_batch*nbq2 + i_stream*nbq3;
    const float * w_base = (const float *) ((const char *) W + i_batch*nbw1 + i_stream*nbw3);
    const int32_t * rows_s = rows ? rows + i_stream*s_rows : nullptr;

    float4 k_reg_f[KEYS_PER_OCT][N_J];

#pragma unroll
    for (int c = 0; c < KEYS_PER_OCT; ++c) {
        const int i_kv = start_kv + c*OCTS_PER_WARP;
        if (i_kv < n_kv) {
            const int64_t i_row = rows_s ? rows_s[i_kv] : i_kv;
            const char * k_base = K + i_row*nbk2 + i_stream*nbk3;
            if constexpr (TYPE_K == GGML_TYPE_F32) {
#pragma unroll
                for (int j = 0; j < N_J; ++j) {
                    k_reg_f[c][j] = ((const float4 *) k_base)[8*j + t];
                }
            } else {
                constexpr dequantize_V_t dequantize_k = get_dequantize_V<TYPE_K, float, 4>();
#pragma unroll
                for (int j = 0; j < N_J; ++j) {
                    dequantize_k(k_base, &k_reg_f[c][j], 32*j + 4*t);
                }
            }
        } else {
#pragma unroll
            for (int j = 0; j < N_J; ++j) {
                k_reg_f[c][j] = make_float4(0, 0, 0, 0);
            }
        }
    }

    __shared__ float  w_shared[N_HEAD];
    __shared__ float4 q_shared_f[N_HEAD][N_EMBD / 4];

    if (tid < N_HEAD) {
        w_shared[tid] = w_base[tid];
    }
#pragma unroll
    for (int i_q = tid; i_q < N_HEAD * (N_EMBD / 4); i_q += THREADS_PER_BLOCK) {
        const int i_head = i_q / (N_EMBD / 4);
        const int i_embd = i_q % (N_EMBD / 4);
        q_shared_f[i_head][i_embd] = *(const float4 *) (q_base + i_head*nbq1 + i_embd*sizeof(float4));
    }

    __syncthreads();

    float score = 0.0f; // of the one key the cross-lane levels leave this lane

#pragma unroll 2
    for (int i_head = 0; i_head < N_HEAD; ++i_head) {
        const float w_val = w_shared[i_head];
        float qk[KEYS_PER_OCT][N_J];

#pragma unroll
        for (int j = 0; j < N_J; ++j) {
            const float4 q_vec = q_shared_f[i_head][8*j + t];
#pragma unroll
            for (int c = 0; c < KEYS_PER_OCT; ++c) {
                qk[c][j] = 0.0f;
                ggml_cuda_mad(qk[c][j], q_vec.x, k_reg_f[c][j].x);
                ggml_cuda_mad(qk[c][j], q_vec.y, k_reg_f[c][j].y);
                ggml_cuda_mad(qk[c][j], q_vec.z, k_reg_f[c][j].z);
                ggml_cuda_mad(qk[c][j], q_vec.w, k_reg_f[c][j].w);
            }
        }

        float part[KEYS_PER_OCT];
#pragma unroll
        for (int c = 0; c < KEYS_PER_OCT; ++c) {
            static_assert(N_J == 4, "the in-register levels are xor 16 and 8");
            const float x16_0 = qk[c][0] + qk[c][2];
            const float x16_1 = qk[c][1] + qk[c][3];
            part[c] = x16_0 + x16_1;
        }

        // xor 4, 2 and 1 across the octet, halving the keys a lane holds at each level while it holds more than one
        if constexpr (KEYS_PER_OCT >= 2) {
            lightning_indexer_oct_level<KEYS_PER_OCT>(part, 4, t & 4);
        } else {
            part[0] += __shfl_xor_sync(0xffffffff, part[0], 4, WARP_SIZE);
        }
        if constexpr (KEYS_PER_OCT >= 4) {
            lightning_indexer_oct_level<KEYS_PER_OCT/2>(part, 2, t & 2);
        } else {
            part[0] += __shfl_xor_sync(0xffffffff, part[0], 2, WARP_SIZE);
        }
        if constexpr (KEYS_PER_OCT >= 8) {
            lightning_indexer_oct_level<KEYS_PER_OCT/4>(part, 1, t & 1);
        } else {
            part[0] += __shfl_xor_sync(0xffffffff, part[0], 1, WARP_SIZE);
        }

        // ReLU, weight
        const float sum = (part[0] > 0.0f) ? part[0] : 0.0f;
        score += sum * w_val;
    }

    // the key the levels left this lane, and one lane of each that holds it
    const int c = (KEYS_PER_OCT >= 2 && (t & 4) ? KEYS_PER_OCT/2 : 0) + (KEYS_PER_OCT >= 4 && (t & 2) ? KEYS_PER_OCT/4 : 0) +
        (KEYS_PER_OCT >= 8 && (t & 1) ? KEYS_PER_OCT/8 : 0);
    constexpr int same_key = 8/KEYS_PER_OCT - 1; // the low bits of t whose lanes hold the same key
    const int i_kv = start_kv + c*OCTS_PER_WARP;
    if ((t & same_key) == 0 && i_kv < n_kv) {
        const half * m_base = (const half *) ((const char *) M + i_batch*nbm1 + (i_stream%nem3)*nbm3);
        float * dst_base = (float *) ((char *) dst + i_batch*nb1 + i_stream*nb3);
        dst_base[i_kv] = score + __half2float(m_base[i_kv]);
    }
}

// ggml_lightning_indexer_set_q8_query: q rounded to 8-bit blocks of 32 as the CPU rounds it, d = max |x| / 127 and
// roundf(x / d); __fdiv_rn keeps both divisions IEEE under -use_fast_math, so the integers are the CPU's
static __device__ __forceinline__ float lightning_indexer_q8_scale(const float amax) {
    return __fdiv_rn(amax, 127.0f);
}

static __device__ __forceinline__ int lightning_indexer_q8_round(const float x, const float d) {
    return d != 0.0f ? (int) roundf(__fdiv_rn(x, d)) : 0;
}

// the q8 query where the int8 kernel does not run: q rounded into dst at q's own offsets, for the vector kernels to
// score (the CPU's arithmetic: d times the rounded integer, in f32). A warp a row of 128, 4 values a lane, so a block
// of 32 is 8 lanes.
template <int N_EMBD>
static __global__ void lightning_indexer_round_q8(
        const char * Q, char * dst, int64_t n_head, int64_t n_batch, int64_t n_q_rows, size_t nbq1, size_t nbq2, size_t nbq3) {
    static_assert(N_EMBD == 4*WARP_SIZE, "a lane rounds 4 values");
    const int64_t i_row = (int64_t) blockIdx.x*blockDim.y + threadIdx.y;
    if (i_row >= n_q_rows) {
        return;
    }
    const int64_t i_head  = i_row % n_head;
    const int64_t i_batch = (i_row / n_head) % n_batch;
    const int64_t i_strm  = i_row / (n_head*n_batch);
    const size_t  off     = i_head*nbq1 + i_batch*nbq2 + i_strm*nbq3;

    const float4 x = ((const float4 *) (Q + off))[threadIdx.x];
    float amax = fmaxf(fmaxf(fabsf(x.x), fabsf(x.y)), fmaxf(fabsf(x.z), fabsf(x.w)));
#pragma unroll
    for (int m = 1; m < 8; m *= 2) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, m, WARP_SIZE));
    }
    const float d = lightning_indexer_q8_scale(amax);
    float4 y;
    y.x = d * (float) lightning_indexer_q8_round(x.x, d);
    y.y = d * (float) lightning_indexer_q8_round(x.y, d);
    y.z = d * (float) lightning_indexer_q8_round(x.z, d);
    y.w = d * (float) lightning_indexer_q8_round(x.w, d);
    ((float4 *) (dst + off))[threadIdx.x] = y;
}

static __device__ __forceinline__ int lightning_indexer_q8_pack(const float4 x, const float d) {
    char4 c;
    c.x = (signed char) lightning_indexer_q8_round(x.x, d);
    c.y = (signed char) lightning_indexer_q8_round(x.y, d);
    c.z = (signed char) lightning_indexer_q8_round(x.z, d);
    c.w = (signed char) lightning_indexer_q8_round(x.w, d);
    return *((const int *) &c);
}

static constexpr int LIGHTNING_INDEXER_Q8_WARPS = 8;  // tokens a block, a warp each
static constexpr int LIGHTNING_INDEXER_Q8_POOLS = 64; // keys a tile of shared memory
// a warp scores one token, so a decode or a verify leaves most of the kernel's warps idle and the rounded buffer is faster
static constexpr int LIGHTNING_INDEXER_Q8_MIN_TOKENS = 8;
static constexpr int LIGHTNING_INDEXER_Q8_KSTR  = 36; // ints a key's row in shared memory: 32 and 4 of padding, so a
                                                      // B fragment's 8 keys x 4 ints fall on 32 banks

// The q8 query on int8 tensor cores, for a Q8_0 k of 128 values a key and 32 heads (GLM-5.3). A warp holds one token:
// its q rounded once into the A fragments of m16n8k32 (2 tiles of 16 heads x 4 blocks of 32, with each row's 4 block
// scales), and the block's warps share each tile of 64 keys in shared memory, B fragments of 8 keys. A key's dot with a
// head is the sum over its 4 blocks of the int32 block dot times q's and k's block scales, in f32; then ReLU, the head's
// weight, and the sum over the 32 heads (the lane's 4 rows, then across the 8 lanes of a column), and the mask.
template <int N_WARPS>
static __global__ void __launch_bounds__(N_WARPS*WARP_SIZE, 2) lightning_indexer_kernel_q8_mma(
        const float * Q, const char * K, const float * W, const half * M, float * dst,
        int64_t n_batch, int64_t n_kv, int64_t kv_per_block,
        size_t nb1, size_t nb3,
        size_t nbq1, size_t nbq2, size_t nbq3,
        size_t nbk2, size_t nbk3,
        size_t nbw1, size_t nbw3,
        size_t nbm1, size_t nbm3,
        int64_t nem3,
        const int32_t * rows, int64_t s_rows) {
#if defined(TURING_MMA_AVAILABLE) && !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    using namespace ggml_cuda_mma;
    constexpr int N_EMBD = 128;
    constexpr int N_BLK  = N_EMBD / QK8_0; // 4 blocks of 32 a key
    constexpr int N_HT   = 2;              // tiles of 16 heads
    constexpr int P      = LIGHTNING_INDEXER_Q8_POOLS;
    constexpr int KSTR   = LIGHTNING_INDEXER_Q8_KSTR;
    constexpr int N_THREADS = N_WARPS*WARP_SIZE;

    const int i_warp   = threadIdx.y;
    const int lane     = threadIdx.x;
    const int tid      = i_warp*WARP_SIZE + lane;
    const int g        = lane / 4; // the fragments' row (head) or column (key) group
    const int tig      = lane % 4;
    const int i_stream = blockIdx.z;
    const int64_t i_batch = (int64_t) blockIdx.x*N_WARPS + i_warp;
    const bool active  = i_batch < n_batch;

    const int64_t kv0 = (int64_t) blockIdx.y*kv_per_block;
    const int64_t kv1 = min(kv0 + kv_per_block, n_kv);

    // a0..a3 of head tile ht, block b: rows (heads) g and g + 8 at the block's values 4 tig..4 tig + 3 (a0, a1) and
    // 16 + 4 tig.. (a2, a3); dq the two rows' block scales; wv their weights
    tile<16, 8, int> A[N_HT][N_BLK];
    float dq[N_HT][N_BLK][2];
    float wv[N_HT][2];
    if (active) {
        const char  * q_base = (const char  *) Q + i_batch*nbq2 + i_stream*nbq3;
        const float * w_base = (const float *) ((const char *) W + i_batch*nbw1 + i_stream*nbw3);
#pragma unroll
        for (int ht = 0; ht < N_HT; ++ht) {
#pragma unroll
            for (int r = 0; r < 2; ++r) {
                const int i_head = 16*ht + 8*r + g;
                const float4 * qh = (const float4 *) (q_base + i_head*nbq1);
                wv[ht][r] = w_base[i_head];
#pragma unroll
                for (int b = 0; b < N_BLK; ++b) {
                    const float4 lo = qh[(QK8_0*b +      4*tig)/4];
                    const float4 hi = qh[(QK8_0*b + 16 + 4*tig)/4];
                    float amax = fmaxf(fmaxf(fmaxf(fabsf(lo.x), fabsf(lo.y)), fmaxf(fabsf(lo.z), fabsf(lo.w))),
                                       fmaxf(fmaxf(fabsf(hi.x), fabsf(hi.y)), fmaxf(fabsf(hi.z), fabsf(hi.w))));
                    amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 1, WARP_SIZE));
                    amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 2, WARP_SIZE));
                    const float d = lightning_indexer_q8_scale(amax);
                    A[ht][b].x[r]     = lightning_indexer_q8_pack(lo, d);
                    A[ht][b].x[2 + r] = lightning_indexer_q8_pack(hi, d);
                    dq[ht][b][r] = d;
                }
            }
        }
    }

    __shared__ int   k_s[P*KSTR];
    __shared__ float dk_s[P*N_BLK];

    const int32_t * rows_s = rows ? rows + i_stream*s_rows : nullptr;
    const half  * m_base   = (const half *) ((const char *) M + i_batch*nbm1 + (i_stream%nem3)*nbm3);
    float       * dst_base = (float *) ((char *) dst + i_batch*nb1 + i_stream*nb3);

    for (int64_t kv_tile = kv0; kv_tile < kv1; kv_tile += P) {
        __syncthreads(); // the previous tile is read

        // a thread a block of a key: its scale, and its 32 values as 8 ints (2-byte aligned in a block_q8_0)
        for (int u = tid; u < P*N_BLK; u += N_THREADS) {
            const int p = u / N_BLK;
            const int b = u % N_BLK;
            const int64_t i_kv = kv_tile + p;
            int * kp = k_s + p*KSTR + 8*b;
            if (i_kv < kv1) {
                const int64_t i_row = rows_s ? rows_s[i_kv] : i_kv;
                const block_q8_0 * kb = (const block_q8_0 *) (K + i_row*nbk2 + i_stream*nbk3) + b;
                dk_s[p*N_BLK + b] = __half2float(kb->d);
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    kp[j] = get_int_b2(kb->qs, j);
                }
            } else {
                dk_s[p*N_BLK + b] = 0.0f;
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    kp[j] = 0;
                }
            }
        }

        __syncthreads();

        if (!active) {
            continue;
        }

#pragma unroll 2
        for (int nt = 0; nt < P/8; ++nt) {
            // D of a head tile: rows g and g + 8, columns (keys) 2 tig and 2 tig + 1 of the 8
            float acc[N_HT][4] = {{0.0f}};
#pragma unroll
            for (int b = 0; b < N_BLK; ++b) {
                tile<8, 8, int> B; // b0, b1: key g at the block's values 4 tig.. and 16 + 4 tig..
                B.x[0] = k_s[(8*nt + g)*KSTR + 8*b +     tig];
                B.x[1] = k_s[(8*nt + g)*KSTR + 8*b + 4 + tig];
                const float dk0 = dk_s[(8*nt + 2*tig + 0)*N_BLK + b];
                const float dk1 = dk_s[(8*nt + 2*tig + 1)*N_BLK + b];
#pragma unroll
                for (int ht = 0; ht < N_HT; ++ht) {
                    tile<16, 8, int> D;
                    mma(D, A[ht][b], B);
                    acc[ht][0] += (dq[ht][b][0]*dk0) * (float) D.x[0];
                    acc[ht][1] += (dq[ht][b][0]*dk1) * (float) D.x[1];
                    acc[ht][2] += (dq[ht][b][1]*dk0) * (float) D.x[2];
                    acc[ht][3] += (dq[ht][b][1]*dk1) * (float) D.x[3];
                }
            }

            // ReLU, weight, and the sum over heads: the lane's 4 rows of each column, then the 8 lanes that share it
            float s0 = 0.0f;
            float s1 = 0.0f;
#pragma unroll
            for (int ht = 0; ht < N_HT; ++ht) {
                s0 += fmaxf(acc[ht][0], 0.0f)*wv[ht][0] + fmaxf(acc[ht][2], 0.0f)*wv[ht][1];
                s1 += fmaxf(acc[ht][1], 0.0f)*wv[ht][0] + fmaxf(acc[ht][3], 0.0f)*wv[ht][1];
            }
#pragma unroll
            for (int m = 4; m < WARP_SIZE; m *= 2) {
                s0 += __shfl_xor_sync(0xffffffff, s0, m, WARP_SIZE);
                s1 += __shfl_xor_sync(0xffffffff, s1, m, WARP_SIZE);
            }

            const int64_t i_kv = kv_tile + 8*nt + 2*tig;
            if (g == 0 && i_kv < kv1) {
                dst_base[i_kv] = s0 + __half2float(m_base[i_kv]);
            }
            if (g == 0 && i_kv + 1 < kv1) {
                dst_base[i_kv + 1] = s1 + __half2float(m_base[i_kv + 1]);
            }
        }
    }
#else
    GGML_UNUSED_VARS(Q, K, W, M, dst, n_batch, n_kv, kv_per_block, nb1, nb3, nbq1, nbq2, nbq3, nbk2, nbk3,
        nbw1, nbw3, nbm1, nbm3, nem3, rows, s_rows);
    NO_DEVICE_CODE;
#endif // defined(TURING_MMA_AVAILABLE) && !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
}

// GGML_CUDA_LIGHTNING_INDEXER_CHECK=1: the vector kernel runs beside the octet (or quad) kernel and a differing bit traps
static __global__ void lightning_indexer_check(const float * quad, const float * vec, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n && __float_as_uint(quad[i]) != __float_as_uint(vec[i])) {
        printf("lightning indexer check: score %lld is %08x from the octet (or quad) kernel, %08x from the vector kernel\n",
                (long long) i, __float_as_uint(quad[i]), __float_as_uint(vec[i]));
        __trap();
    }
}

#define LIGHTNING_INDEXER_CASE(lightning_indexer_kernel, n_embd, n_head, K, type_K)         \
    if (K->type == (type_K)) {                                                              \
        lightning_indexer_kernel<WARPS_PER_BLOCK, K_VECS_PER_BLOCK, n_embd, n_head, type_K> \
            <<<grid, block, 0, ctx.stream()>>>(                                             \
            q_d, k_d, w_d, m_d, dst_d,                                                      \
            n_stream, n_batch, n_kv,                                                        \
            nb1, nb2, nb3,                                                                  \
            nbq1, nbq2, nbq3,                                                               \
            nbk1, nbk2, nbk3,                                                               \
            nbw1, nbw2, nbw3,                                                               \
            nbm1, nbm2, nbm3,                                                               \
            nem3                                                                            \
        );                                                                                  \
    } else

// the quad kernel's launch: 64 keys a block, the vector kernel's grid
static constexpr int LIGHTNING_INDEXER_QUAD_WARPS = 4;
static constexpr int LIGHTNING_INDEXER_QUAD_KEYS  = 2;

// the octet kernel's: 4 warps of 4 octets, LIGHTNING_INDEXER_OCT_KEYS keys an octet
static constexpr int LIGHTNING_INDEXER_OCT_WARPS = 4;
static constexpr int LIGHTNING_INDEXER_OCT_KEYS  = 4;
static constexpr int LIGHTNING_INDEXER_OCT_KEYS_PER_BLOCK = LIGHTNING_INDEXER_OCT_WARPS * (WARP_SIZE/8) * LIGHTNING_INDEXER_OCT_KEYS;

// the octet (or quad) kernel into dst, the vector kernel into vec_d (dst, or the check's buffer), whichever of them run
#define LIGHTNING_INDEXER_VEC_CASE(n_embd, n_head, K, type_K)                                   \
    if (K->type == (type_K)) {                                                                   \
        if (!vec_legacy && !oct_legacy) {                                                        \
            lightning_indexer_kernel_oct<LIGHTNING_INDEXER_OCT_WARPS, LIGHTNING_INDEXER_OCT_KEYS,   \
                n_embd, n_head, type_K>                                                          \
                <<<grid_oct, block_oct, 0, ctx.stream()>>>(                                      \
                q_d, k_d, w_d, m_d, dst_d,                                                       \
                n_stream, n_batch, n_kv,                                                         \
                nb1, nb2, nb3,                                                                   \
                nbq1, nbq2, nbq3,                                                                \
                nbk1, nbk2, nbk3,                                                                \
                nbw1, nbw2, nbw3,                                                                \
                nbm1, nbm2, nbm3,                                                                \
                nem3,                                                                            \
                rows_d, s_rows                                                                   \
            );                                                                                   \
        } else if (!vec_legacy) {                                                                \
            lightning_indexer_kernel_quad<LIGHTNING_INDEXER_QUAD_WARPS, LIGHTNING_INDEXER_QUAD_KEYS, \
                n_embd, n_head, type_K>                                                          \
                <<<grid, block_quad, 0, ctx.stream()>>>(                                         \
                q_d, k_d, w_d, m_d, dst_d,                                                       \
                n_stream, n_batch, n_kv,                                                         \
                nb1, nb2, nb3,                                                                   \
                nbq1, nbq2, nbq3,                                                                \
                nbk1, nbk2, nbk3,                                                                \
                nbw1, nbw2, nbw3,                                                                \
                nbm1, nbm2, nbm3,                                                                \
                nem3,                                                                            \
                rows_d, s_rows                                                                   \
            );                                                                                   \
        }                                                                                        \
        if (vec_d != nullptr) {                                                                  \
            lightning_indexer_kernel_vec<WARPS_PER_BLOCK, K_VECS_PER_BLOCK, n_embd, n_head, type_K> \
                <<<grid, block, 0, ctx.stream()>>>(                                              \
                q_d, k_d, w_d, m_d, vec_d,                                                       \
                n_stream, n_batch, n_kv,                                                         \
                nb1, nb2, nb3,                                                                   \
                nbq1, nbq2, nbq3,                                                                \
                nbk1, nbk2, nbk3,                                                                \
                nbw1, nbw2, nbw3,                                                                \
                nbm1, nbm2, nbm3,                                                                \
                nem3,                                                                            \
                rows_d, s_rows                                                                   \
            );                                                                                   \
        }                                                                                        \
    } else

void ggml_cuda_lightning_indexer(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];
    const ggml_tensor * w = dst->src[2]; // weights
    const ggml_tensor * m = dst->src[3]; // mask
    const ggml_tensor * r = dst->src[4]; // ggml_lightning_indexer_rows, or none

    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(  q->type == GGML_TYPE_F32);
    GGML_ASSERT(  w->type == GGML_TYPE_F32);
    GGML_ASSERT(  m->type == GGML_TYPE_F16);

    GGML_TENSOR_LOCALS(int64_t, neq,  q, ne)
    GGML_TENSOR_LOCALS(size_t,  nbq,  q, nb)
    GGML_TENSOR_LOCALS(int64_t, nek,  k, ne)
    GGML_TENSOR_LOCALS(size_t,  nbk,  k, nb)
    GGML_TENSOR_LOCALS(int64_t, new,  w, ne)
    GGML_TENSOR_LOCALS(size_t,  nbw,  w, nb)
    GGML_TENSOR_LOCALS(int64_t, nem,  m, ne)
    GGML_TENSOR_LOCALS(size_t,  nbm,  m, nb)
    GGML_TENSOR_LOCALS(int64_t, ne, dst, ne)
    GGML_TENSOR_LOCALS(size_t,  nb, dst, nb)

    // the operands the scheduler assigned here at supports_op time must still be readable (data addresses included)
    GGML_ASSERT(ggml_cuda_lightning_indexer_supported(ctx.device, dst));

    // input tensor rows must be contiguous
    GGML_ASSERT(nbq0 == ggml_type_size(q->type));
    GGML_ASSERT(nbk0 == ggml_type_size(k->type));
    GGML_ASSERT(nbw0 == ggml_type_size(w->type));
    GGML_ASSERT(nbm0 == ggml_type_size(m->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == sizeof(float));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    const int n_embd   = q->ne[0];
    const int n_head   = q->ne[1];
    const int n_batch  = q->ne[2];
    const int n_stream = q->ne[3];
    const int n_kv     = dst->ne[0];

    const float *   q_d = (const float *)   q->data;
    const char  *   k_d = (const char  *)   k->data;
    const float *   w_d = (const float *)   w->data;
    const half  *   m_d = (const half  *)   m->data;
    float       * dst_d = (      float *) dst->data;

    // rows: the vector kernel, which scores in f32 whatever k's type (the wmma kernel rounds q to f16)
    const int32_t * rows_d = r ? (const int32_t *) r->data : nullptr;
    const int64_t   s_rows = r ? r->nb[1] / sizeof(int32_t) : 0;

    // the vector kernel's cases run the octet kernel; GGML_CUDA_LIGHTNING_INDEXER_OCT_LEGACY=1 the quad kernel,
    // GGML_CUDA_LIGHTNING_INDEXER_VEC_LEGACY=1 the vector kernel, and GGML_CUDA_LIGHTNING_INDEXER_CHECK=1, a debug
    // instrument, the vector kernel too (into a buffer the check compares)
    static const bool vec_legacy = ggml_env_switch("GGML_CUDA_LIGHTNING_INDEXER_VEC_LEGACY");
    static const bool oct_legacy = ggml_env_switch("GGML_CUDA_LIGHTNING_INDEXER_OCT_LEGACY");
    static const bool vec_check  = !vec_legacy && ggml_env_switch("GGML_CUDA_LIGHTNING_INDEXER_CHECK");
    ggml_cuda_pool_alloc<float> vec_check_buf(ctx.pool());
    float * vec_d = vec_legacy ? dst_d : nullptr;
    const dim3 block_quad(WARP_SIZE, LIGHTNING_INDEXER_QUAD_WARPS);
    const dim3 block_oct(WARP_SIZE, LIGHTNING_INDEXER_OCT_WARPS);
    const dim3 grid_oct((n_kv + LIGHTNING_INDEXER_OCT_KEYS_PER_BLOCK - 1) / LIGHTNING_INDEXER_OCT_KEYS_PER_BLOCK, n_batch, n_stream);

    const int device = ggml_cuda_get_device();
    const int cc     = ggml_cuda_info().devices[device].cc;

    // ggml_lightning_indexer_set_q8_query: the int8 kernel where it runs (NVIDIA from Turing, 32 heads) on 8 tokens or
    // more, elsewhere q rounded into a buffer the kernels below score, the same scores to f32 rounding.
    // GGML_CUDA_LIGHTNING_INDEXER_Q8_MMA_LEGACY=1 takes the buffer everywhere.
    const bool q8_query = ggml_get_op_params_i32(dst, 0) == 1;
    ggml_cuda_pool_alloc<char> q_round_buf(ctx.pool());
    if (q8_query) {
        GGML_ASSERT(k->type == GGML_TYPE_Q8_0 && rows_d != nullptr && n_embd == 128);
        static const bool q8_mma_legacy = ggml_env_switch("GGML_CUDA_LIGHTNING_INDEXER_Q8_MMA_LEGACY");
        if (!q8_mma_legacy && n_head == 32 && n_batch >= LIGHTNING_INDEXER_Q8_MIN_TOKENS && GGML_CUDA_CC_IS_NVIDIA(cc) &&
                turing_mma_available(cc)) {
            constexpr int N_WARPS = LIGHTNING_INDEXER_Q8_WARPS;
            constexpr int P       = LIGHTNING_INDEXER_Q8_POOLS;
            const int64_t n_tiles = (n_kv + P - 1) / P;
            const int64_t n_bx    = (n_batch + N_WARPS - 1) / N_WARPS;
            const int64_t nsm     = ggml_cuda_info().devices[device].nsm;
            // the keys split so the grid is about 4 blocks an SM
            const int64_t n_by = std::max<int64_t>(1, std::min<int64_t>(n_tiles, (4*nsm + n_bx*n_stream - 1) / (n_bx*n_stream)));
            const int64_t kv_per_block = (n_tiles + n_by - 1) / n_by * P;
            const dim3 grid(n_bx, (n_kv + kv_per_block - 1) / kv_per_block, n_stream);
            const dim3 block(WARP_SIZE, N_WARPS);
            lightning_indexer_kernel_q8_mma<N_WARPS><<<grid, block, 0, ctx.stream()>>>(
                q_d, k_d, w_d, m_d, dst_d,
                n_batch, n_kv, kv_per_block,
                nb1, nb3,
                nbq1, nbq2, nbq3,
                nbk2, nbk3,
                nbw1, nbw3,
                nbm1, nbm3,
                nem3,
                rows_d, s_rows);
            return;
        }
        constexpr int ROUND_ROWS = 4; // warps a block
        const int64_t n_q_rows = (int64_t) n_head*n_batch*n_stream;
        q_round_buf.alloc(ggml_nbytes(q));
        lightning_indexer_round_q8<128><<<(n_q_rows + ROUND_ROWS - 1) / ROUND_ROWS, dim3(WARP_SIZE, ROUND_ROWS), 0, ctx.stream()>>>(
            (const char *) q_d, q_round_buf.get(), n_head, n_batch, n_q_rows, nbq1, nbq2, nbq3);
        q_d = (const float *) q_round_buf.get();
    }

    if (n_embd == 128 && n_head == 64) {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
        if (GGML_CUDA_CC_IS_NVIDIA(cc) && turing_mma_available(cc) && k->type != GGML_TYPE_F32 && k->type != GGML_TYPE_BF16 &&
                rows_d == nullptr) {
            // use wmma kernel
            constexpr int K_VECS_PER_BLOCK = 32;
            constexpr int WARPS_PER_BLOCK = 8;

            dim3 block(32, WARPS_PER_BLOCK);
            int num_kv_blocks = (n_kv + (K_VECS_PER_BLOCK) - 1) / (K_VECS_PER_BLOCK);
            dim3 grid(num_kv_blocks, n_batch, n_stream);

            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 64, k, GGML_TYPE_F16)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 64, k, GGML_TYPE_Q4_0)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 64, k, GGML_TYPE_Q4_1)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 64, k, GGML_TYPE_Q5_0)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 64, k, GGML_TYPE_Q5_1)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 64, k, GGML_TYPE_Q8_0)
            GGML_ABORT("fatal error");
        } else {
#else // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
        {
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
            // use vector kernel
            constexpr int K_VECS_PER_WARP = 8;
            constexpr int WARPS_PER_BLOCK = 8;
            constexpr int K_VECS_PER_BLOCK = K_VECS_PER_WARP * WARPS_PER_BLOCK;

            dim3 block(32, WARPS_PER_BLOCK);
            int num_kv_blocks = (n_kv + (K_VECS_PER_BLOCK) - 1) / (K_VECS_PER_BLOCK);
            dim3 grid(num_kv_blocks, n_batch, n_stream);

            static_assert(LIGHTNING_INDEXER_QUAD_WARPS * (WARP_SIZE/4) * LIGHTNING_INDEXER_QUAD_KEYS == K_VECS_PER_BLOCK,
                    "the quad kernel takes the vector kernel's grid");
            if (vec_check) {
                vec_d = vec_check_buf.alloc(ggml_nelements(dst));
            }

            LIGHTNING_INDEXER_VEC_CASE(128, 64, k, GGML_TYPE_F16)
            LIGHTNING_INDEXER_VEC_CASE(128, 64, k, GGML_TYPE_Q4_0)
            LIGHTNING_INDEXER_VEC_CASE(128, 64, k, GGML_TYPE_Q4_1)
            LIGHTNING_INDEXER_VEC_CASE(128, 64, k, GGML_TYPE_Q5_0)
            LIGHTNING_INDEXER_VEC_CASE(128, 64, k, GGML_TYPE_Q5_1)
            LIGHTNING_INDEXER_VEC_CASE(128, 64, k, GGML_TYPE_Q8_0)
            LIGHTNING_INDEXER_VEC_CASE(128, 64, k, GGML_TYPE_BF16)
            LIGHTNING_INDEXER_VEC_CASE(128, 64, k, GGML_TYPE_F32)
            GGML_ABORT("fatal error");
        }
    } else if (n_embd == 128 && n_head == 32) {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
        if (GGML_CUDA_CC_IS_NVIDIA(cc) && turing_mma_available(cc) && k->type != GGML_TYPE_F32 && k->type != GGML_TYPE_BF16 &&
                rows_d == nullptr) {
            // use wmma kernel
            constexpr int K_VECS_PER_BLOCK = 32;
            constexpr int WARPS_PER_BLOCK = 8;

            dim3 block(32, WARPS_PER_BLOCK);
            int num_kv_blocks = (n_kv + (K_VECS_PER_BLOCK) - 1) / (K_VECS_PER_BLOCK);
            dim3 grid(num_kv_blocks, n_batch, n_stream);

            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 32, k, GGML_TYPE_F16)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 32, k, GGML_TYPE_Q4_0)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 32, k, GGML_TYPE_Q4_1)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 32, k, GGML_TYPE_Q5_0)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 32, k, GGML_TYPE_Q5_1)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 32, k, GGML_TYPE_Q8_0)
            GGML_ABORT("fatal error");
        } else {
#else // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
        {
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
            // use vector kernel
            constexpr int K_VECS_PER_WARP = 8;
            constexpr int WARPS_PER_BLOCK = 8;
            constexpr int K_VECS_PER_BLOCK = K_VECS_PER_WARP * WARPS_PER_BLOCK;

            dim3 block(32, WARPS_PER_BLOCK);
            int num_kv_blocks = (n_kv + (K_VECS_PER_BLOCK) - 1) / (K_VECS_PER_BLOCK);
            dim3 grid(num_kv_blocks, n_batch, n_stream);

            static_assert(LIGHTNING_INDEXER_QUAD_WARPS * (WARP_SIZE/4) * LIGHTNING_INDEXER_QUAD_KEYS == K_VECS_PER_BLOCK,
                    "the quad kernel takes the vector kernel's grid");
            if (vec_check) {
                vec_d = vec_check_buf.alloc(ggml_nelements(dst));
            }

            LIGHTNING_INDEXER_VEC_CASE(128, 32, k, GGML_TYPE_F16)
            LIGHTNING_INDEXER_VEC_CASE(128, 32, k, GGML_TYPE_Q4_0)
            LIGHTNING_INDEXER_VEC_CASE(128, 32, k, GGML_TYPE_Q4_1)
            LIGHTNING_INDEXER_VEC_CASE(128, 32, k, GGML_TYPE_Q5_0)
            LIGHTNING_INDEXER_VEC_CASE(128, 32, k, GGML_TYPE_Q5_1)
            LIGHTNING_INDEXER_VEC_CASE(128, 32, k, GGML_TYPE_Q8_0)
            LIGHTNING_INDEXER_VEC_CASE(128, 32, k, GGML_TYPE_BF16)
            LIGHTNING_INDEXER_VEC_CASE(128, 32, k, GGML_TYPE_F32)
            GGML_ABORT("fatal error");
        }
    } else {
        GGML_ABORT("fatal error");
    }

    if (vec_check && vec_d != nullptr) {
        GGML_ASSERT(ggml_is_contiguous(dst));
        const int64_t n = ggml_nelements(dst);
        lightning_indexer_check<<<(n + 255) / 256, 256, 0, ctx.stream()>>>(dst_d, vec_d, n);
    }
}

bool ggml_cuda_lightning_indexer_supported(int device, const ggml_tensor * dst) {
    GGML_UNUSED(device);

    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];
    const ggml_tensor * w = dst->src[2]; // weights
    const ggml_tensor * m = dst->src[3]; // mask

    GGML_TENSOR_LOCALS(int64_t, neq,  q, ne)
    GGML_TENSOR_LOCALS(size_t,  nbq,  q, nb)
    GGML_TENSOR_LOCALS(int64_t, nek,  k, ne)
    GGML_TENSOR_LOCALS(size_t,  nbk,  k, nb)
    GGML_TENSOR_LOCALS(int64_t, new,  w, ne)
    GGML_TENSOR_LOCALS(size_t,  nbw,  w, nb)
    GGML_TENSOR_LOCALS(int64_t, nem,  m, ne)
    GGML_TENSOR_LOCALS(size_t,  nbm,  m, nb)
    GGML_TENSOR_LOCALS(int64_t, ne, dst, ne)
    GGML_TENSOR_LOCALS(size_t,  nb, dst, nb)

    if (neq0 != 128) {
        return false;
    }

    if (neq1 != 64 && neq1 != 32) {
        return false;
    }

    const ggml_tensor * r = dst->src[4];
    if (r != nullptr && (r->type != GGML_TYPE_I32 || !ggml_is_contiguous(r))) {
        return false;
    }

    // alignment checks: the kernels read q as float4 and a float k as int2 or float4, so the data addresses must be
    // multiples of 16 bytes as well as the strides (a view offset by an odd number of elements would fault with a
    // misaligned address). This runs in supports_op, before allocation, and again at dispatch.
    for (const ggml_tensor * t : {q, k}) {
        if (ggml_is_quantized(t->type)) {
            continue;
        }
        if (!ggml_cuda_data_is_aligned(t, 16)) {
            return false;
        }
        for (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
            if (t->nb[i] % 16 != 0) {
                return false;
            }
        }
    }

    switch(k->type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_BF16:
        case GGML_TYPE_F16:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q4_0:
            return true;
        default:
            return false;
    }
}
