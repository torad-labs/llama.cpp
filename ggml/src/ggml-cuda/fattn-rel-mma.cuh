#pragma once

#include "common.cuh"
#include "mma.cuh"
#include "fattn-common.cuh"
#include "fattn-rel-decode.cuh"

// A tile holds 16 visible KV rows. A scan step reads 4 mask words a lane, 4096 cells a warp.
constexpr int fattn_rel_mma_cells = 16;
constexpr int fattn_rel_mma_words = 4;

// GGML_CUDA_FATTN_REL_DECODE_STREAMS: most warps the cells of one query and KV head are split across.
static int fattn_rel_mma_streams(const ggml_tensor * dst) {
    static const int streams_max = (int) ggml_env_int("GGML_CUDA_FATTN_REL_DECODE_STREAMS");
    const int64_t step_cells = 32*32*fattn_rel_mma_words;
    return std::max<int64_t>(1, std::min<int64_t>(streams_max, dst->src[1]->ne[1]/step_cells));
}

// F16 K/V shared by 2 to 8 query heads a KV head. The one-cell kernel keeps Q8_0 and the other ratios.
static bool fattn_rel_mma_shapes(const ggml_tensor * dst) {
    static const bool legacy = ggml_env_switch("GGML_CUDA_FATTN_REL_DECODE_LEGACY");
    const auto * q = dst->src[0];
    const auto * k = dst->src[1];
    const auto * v = dst->src[2];
    const int64_t gqa = q->ne[2]/k->ne[2];
    return !legacy && ampere_mma_available(GGML_CUDA_CC_BLACKWELL) && k->type == GGML_TYPE_F16 &&
        q->ne[2] == gqa*k->ne[2] && gqa >= 2 && gqa <= 8 &&
        k->nb[1]%16 == 0 && k->nb[2]%16 == 0 && v->nb[1]%16 == 0 && v->nb[2]%16 == 0;
}

// The copies that stage a tile read K and V 16 bytes at a time.
static bool fattn_rel_mma_aligned(const ggml_tensor * dst) {
    return (uintptr_t) dst->src[1]->data%16 == 0 && (uintptr_t) dst->src[2]->data%16 == 0;
}

// Each stream stores what a part of the one-cell kernel stores, so one merge reads both.
static size_t fattn_rel_mma_alloc_size(const ggml_tensor * dst) {
    return fattn_rel_mma_shapes(dst) ?
        GGML_PAD(ggml_nbytes(dst), 128) + size_t(dst->ne[1])*dst->ne[2]*fattn_rel_mma_streams(dst)*130*sizeof(float) : 0;
}

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
// One warp gathers 16 visible KV rows into a tile and multiplies it by all the query heads of its KV head on the tensor cores.
// Each row is read once a KV head. Masked cells are never read, so they cannot change how the visible ones are rounded.
static __global__ void fattn_rel_mma_rows(
        const char * q, const char * k, const char * v, const uint32_t * mask, const float * table,
        const int32_t * q_pos, const int32_t * kv_pos, const float * shift, float * partial, float scale, int n_kv, int n_dist, int gqa,
        size_t q_row, size_t q_head, size_t k_row, size_t k_head, size_t v_row, size_t v_head) {
#ifdef AMPERE_MMA_AVAILABLE
    using namespace ggml_cuda_mma;
    constexpr int cells = fattn_rel_mma_cells, words = fattn_rel_mma_words;
    // A staged row is 128 halves and 16 bytes of padding, which spread the rows ldmatrix reads together over the banks.
    constexpr int row_i4 = 17;
    const int lane = threadIdx.x, stream = blockIdx.x, n_streams = gridDim.x;
    const int kv_head = blockIdx.y, query = blockIdx.z;
    const int head0 = kv_head*gqa, n_heads = gridDim.y*gqa;
    const int64_t row0 = int64_t(query)*n_heads + head0;
    __shared__ int4 staged_k[cells*row_i4], staged_v[cells*row_i4];
    __shared__ int list[32*32 + cells];
    k += kv_head*k_head;
    v += kv_head*v_head;

    // Q is the B operand: 8 heads by 16 values a tile, zero for the heads this KV head does not have.
    tile<8, 8, half2> Q_B[8];
    {
        const int j = lane/4;
        const float * q_ptr = (const float *) (q + query*q_row + (head0 + (j < gqa ? j : 0))*q_head);
#pragma unroll
        for (int c = 0; c < 8; ++c) {
#pragma unroll
            for (int l = 0; l < 2; ++l) {
                const int d = 16*c + 2*(4*l + lane%4);
                Q_B[c].x[l] = j < gqa ? make_half2(q_ptr[d]*scale, q_ptr[d + 1]*scale) : make_half2(0.0f, 0.0f);
            }
        }
    }

    // As an MMA result is laid out, a lane holds heads j0 and j0 + 1 at two cells of the tile, and two values of each chunk.
    const int j0 = 2*(lane%4);
    bool valid[2];
    float sink[2], maximum[2], denominator[2];
    const float * bias_row[2];
#pragma unroll
    for (int p = 0; p < 2; ++p) {
        valid[p] = j0 + p < gqa;
        const int64_t row = row0 + (valid[p] ? j0 + p : 0);
        sink[p] = shift ? shift[row] : 0.0f;
        bias_row[p] = table + row*n_dist;
        maximum[p] = -FLT_MAX;
        denominator[p] = 0.0f;
    }
    tile<16, 8, float> O[8];
    const int pos = q_pos[query];

    const int n_words = n_kv/32;
    const int step = n_streams*32*words;
    mask += int64_t(query)*n_words;
    int fill = 0;
    for (int base = stream*32*words; base < n_words; base += step) {
        const bool last = base + step >= n_words;
        uint32_t live[words], any = 0;
#pragma unroll
        for (int i = 0; i < words; ++i) {
            const int word = base + 32*i + lane;
            live[i] = word < n_words ? mask[word] : 0;
            any |= live[i];
        }
        if (!__any_sync(0xffffffff, any != 0) && !(last && fill > 0)) {
            continue;
        }
        // Not unrolled: the tile below is compiled once.
#pragma unroll 1
        for (int i = 0; i < words; ++i) {
            const uint32_t cur = live[0];
#pragma unroll
            for (int w = 0; w + 1 < words; ++w) {
                live[w] = live[w + 1];
            }
            if (__any_sync(0xffffffff, cur != 0)) {
                // The visible cells of the 32 words join the list in ascending order.
                const int count = __popc(cur);
                int end = count;
#pragma unroll
                for (int offset = 1; offset < WARP_SIZE; offset *= 2) {
                    const int below = __shfl_up_sync(0xffffffff, end, offset, WARP_SIZE);
                    end += lane >= offset ? below : 0;
                }
                const int total = __shfl_sync(0xffffffff, end, WARP_SIZE - 1, WARP_SIZE);
                int at = fill + end - count;
                for (uint32_t bits = cur; bits; bits &= bits - 1) {
                    list[at++] = 32*(base + 32*i + lane) + __ffs(bits) - 1;
                }
                __syncwarp();
                fill += total;
            }

            // The last words of the stream also send a tile that is not full.
            const bool flush = last && i == words - 1;
            int done = 0;
            while (fill - done >= cells || (flush && fill > done)) {
                const int n = min(cells, fill - done);
                const int * tile_cells = list + done;
                // A slot past n repeats the first cell, and its weight is 0.
                const int slot = lane%cells, half_row = lane/cells;
                const int slot_a = lane/4, slot_b = 8 + lane/4;
                const int cell = tile_cells[slot < n ? slot : 0];
                const int cell_a = tile_cells[slot_a < n ? slot_a : 0];
                const int cell_b = tile_cells[slot_b < n ? slot_b : 0];
                const int4 * k_src = (const int4 *) (k + int64_t(cell)*k_row) + 8*half_row;
                const int4 * v_src = (const int4 *) (v + int64_t(cell)*v_row) + 8*half_row;
                int4 * k_dst = staged_k + slot*row_i4 + 8*half_row;
                int4 * v_dst = staged_v + slot*row_i4 + 8*half_row;
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    k_dst[j] = k_src[j];
                    v_dst[j] = v_src[j];
                }

                float bias[4];
                bool use[4];
#pragma unroll
                for (int l = 0; l < 4; ++l) {
                    const int p = l%2;
                    const int64_t distance = int64_t(pos) - kv_pos[l < 2 ? cell_a : cell_b];
                    const float b = distance >= 0 && distance < n_dist ? bias_row[p][distance] : 0.0f;
                    bias[l] = shift ? __half2float(__float2half(b - sink[p])) : b;
                    use[l] = valid[p] && (l < 2 ? slot_a : slot_b) < n;
                }
                __syncwarp();

                // S.x[l] is the logit of cell slot (l/2)*8 + lane/4 for head j0 + l%2.
                tile<16, 8, float> S;
#pragma unroll
                for (int c = 0; c < 8; ++c) {
                    tile<16, 8, half2> K_A;
                    load_ldmatrix(K_A, (const half2 *) staged_k + 8*c, 4*row_i4);
                    mma(S, K_A, Q_B[c]);
                }
#pragma unroll
                for (int l = 0; l < 4; ++l) {
                    S.x[l] = use[l] ? S.x[l] + bias[l] : -FLT_MAX;
                }
                float rescale[2];
#pragma unroll
                for (int p = 0; p < 2; ++p) {
                    float m = fmaxf(maximum[p], fmaxf(S.x[p], S.x[p + 2]));
#pragma unroll
                    for (int offset = 16; offset >= 4; offset >>= 1) {
                        m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, offset, WARP_SIZE));
                    }
                    rescale[p] = expf(maximum[p] - m);
                    maximum[p] = m;
                }
#pragma unroll
                for (int l = 0; l < 4; ++l) {
                    S.x[l] = use[l] ? expf(S.x[l] - maximum[l%2]) : 0.0f;
                }

                // The value product takes the weights as halves, and the denominator sums the same halves.
                const tile<16, 4, half2> W = get_half2(S);
                denominator[0] = denominator[0]*rescale[0] + __low2float(W.x[0]) + __low2float(W.x[1]);
                denominator[1] = denominator[1]*rescale[1] + __high2float(W.x[0]) + __high2float(W.x[1]);
                if (__any_sync(0xffffffff, rescale[0] != 1.0f || rescale[1] != 1.0f)) {
#pragma unroll
                    for (int c = 0; c < 8; ++c) {
#pragma unroll
                        for (int l = 0; l < 4; ++l) {
                            O[c].x[l] *= rescale[l%2];
                        }
                    }
                }
                const tile<8, 8, half2> W_B = get_transposed(W);
#pragma unroll
                for (int c = 0; c < 8; ++c) {
                    tile<16, 8, half2> V_A;
                    load_ldmatrix_trans(V_A, (const half2 *) staged_v + 8*c, 4*row_i4);
                    mma(O[c], V_A, W_B);
                }
                __syncwarp();
                done += n;
            }

            // Fewer cells than a tile wait at the head of the list for the next words.
            if (done > 0 && fill > done) {
                const int rest = fill - done;
                const int moved = lane < rest ? list[done + lane] : 0;
                __syncwarp();
                if (lane < rest) {
                    list[lane] = moved;
                }
                __syncwarp();
            }
            fill -= done;
        }
    }

#pragma unroll
    for (int p = 0; p < 2; ++p) {
#pragma unroll
        for (int offset = 16; offset >= 4; offset >>= 1) {
            denominator[p] += __shfl_xor_sync(0xffffffff, denominator[p], offset, WARP_SIZE);
        }
    }
    // A stream with no visible cell stores weight 0, which the merge adds as nothing.
#pragma unroll
    for (int p = 0; p < 2; ++p) {
        if (!valid[p]) {
            continue;
        }
        float * out = partial + ((row0 + j0 + p)*n_streams + stream)*130;
        if (lane < 4) {
            out[0] = maximum[p];
            out[1] = denominator[p];
        }
#pragma unroll
        for (int c = 0; c < 8; ++c) {
            out[2 + 16*c + lane/4] = O[c].x[p];
            out[2 + 16*c + 8 + lane/4] = O[c].x[p + 2];
        }
    }
#else
    GGML_UNUSED_VARS(q, k, v, mask, table, q_pos, kv_pos, shift, partial, scale, n_kv, n_dist, gqa,
        q_row, q_head, k_row, k_head, v_row, v_head);
    NO_DEVICE_CODE;
#endif // AMPERE_MMA_AVAILABLE
}
#endif

static void ggml_cuda_fattn_rel_mma(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    const auto * q = dst->src[0];
    const auto * k = dst->src[1];
    const auto * v = dst->src[2];
    const int n_streams = fattn_rel_mma_streams(dst);
    float * partial = (float *) ((char *) dst->data + GGML_PAD(ggml_nbytes(dst), 128));
    float scale;
    memcpy(&scale, dst->op_params, sizeof(scale));
    const auto * sinks = (const float *) dst->src[9]->data;
    const bool shift = ggml_cuda_fattn_shift_query_sinks();
    fattn_rel_mma_rows<<<dim3(n_streams, k->ne[2], q->ne[1]), WARP_SIZE, 0, ctx.stream()>>>(
            (const char *) q->data, (const char *) k->data, (const char *) v->data,
            (const uint32_t *) dst->src[3]->data, (const float *) dst->src[6]->data,
            (const int32_t *) dst->src[7]->data, (const int32_t *) dst->src[8]->data, shift ? sinks : nullptr, partial,
            scale, k->ne[1], dst->src[6]->ne[0], q->ne[2]/k->ne[2], q->nb[1], q->nb[2], k->nb[1], k->nb[2], v->nb[1], v->nb[2]);
    fattn_rel_decode_merge<<<dim3(q->ne[2], q->ne[1]), 128, 0, ctx.stream()>>>(
            partial, shift ? nullptr : sinks, (float *) dst->data, n_streams);
    CUDA_CHECK(cudaGetLastError());
#else
    GGML_UNUSED_VARS(ctx, dst);
    GGML_ABORT("relative mask decode requires CUDA");
#endif
}
