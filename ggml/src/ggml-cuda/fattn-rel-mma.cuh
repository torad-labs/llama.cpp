#pragma once

#include "common.cuh"
#include "mma.cuh"
#include "fattn-common.cuh"
#include "fattn-rel-decode.cuh"

// A tile holds 16 visible KV rows. A scan step reads 4 mask words a lane, 4096 cells a warp, or 2 a row of a query group.
constexpr int fattn_rel_mma_cells = 16;
constexpr int fattn_rel_mma_words = 4;
// A warp's columns are its rows times the query heads of its KV head, in tiles of 8: at most 3 tiles.
constexpr int fattn_rel_mma_cols = 24;

// GGML_CUDA_FATTN_REL_DECODE_STREAMS, GGML_CUDA_FATTN_REL_DECODE_MEAN_STREAMS: most warps the cells of one query and KV head
// are split across, where 8 rows or more are each bounded, and in a low-mean step, where one row may read the whole
// pool, or a step of fewer rows, whose warps would not fill the card at the bounded count. Every stream keeps a mask
// word a lane.
static int fattn_rel_mma_streams(const ggml_tensor * dst) {
    static const int streams_rows = (int) ggml_env_int("GGML_CUDA_FATTN_REL_DECODE_STREAMS");
    static const int streams_mean = (int) ggml_env_int("GGML_CUDA_FATTN_REL_DECODE_MEAN_STREAMS");
    const int64_t n_kv = dst->src[1]->ne[1], n_live = ggml_flash_attn_ext_get_n_kv_max(dst);
    const bool bounded = n_live > 0 && n_live*8 <= n_kv && dst->src[0]->ne[1] >= 8;
    return std::max<int64_t>(1, std::min<int64_t>(bounded ? streams_rows : streams_mean, n_kv/(32*WARP_SIZE)));
}

// GGML_CUDA_FATTN_REL_DECODE_RUN_WORDS: the mask words a stream reads side by side before the next stream's begin.
static int fattn_rel_mma_run_log2() {
    static const int run_words = (int) ggml_env_int("GGML_CUDA_FATTN_REL_DECODE_RUN_WORDS");
    int run_log2 = 0;
    while ((2 << run_log2) <= run_words) {
        ++run_log2;
    }
    return run_log2;
}

// GGML_CUDA_FATTN_REL_MMA_WIDE_LEGACY: F16 K/V and one row a warp only, as before q8_0 and query groups came here.
static bool fattn_rel_mma_wide() {
    static const bool legacy = ggml_env_switch("GGML_CUDA_FATTN_REL_MMA_WIDE_LEGACY");
    return !legacy;
}

// F16 or Q8_0 K/V shared by 2 to 8 query heads a KV head, and the rows of a query group while they fit 24 columns.
// The one-cell kernel keeps the other shapes.
static bool fattn_rel_mma_shapes(const int device, const ggml_tensor * dst) {
    static const bool legacy = ggml_env_switch("GGML_CUDA_FATTN_REL_DECODE_LEGACY");
    const auto * q = dst->src[0];
    const auto * k = dst->src[1];
    const auto * v = dst->src[2];
    const int64_t gqa = q->ne[2]/k->ne[2];
    const int group = fattn_rel_decode_group(dst);
    const bool q8 = k->type == GGML_TYPE_Q8_0 && v->type == GGML_TYPE_Q8_0 && fattn_rel_mma_wide();
    const bool f16 = k->type == GGML_TYPE_F16 && v->type == GGML_TYPE_F16;
    // A stage reads F16 rows 16 bytes at a time and Q8_0 rows (136 bytes) 4 bytes at a time.
    const size_t align = q8 ? 4 : 16;
    return !legacy && ampere_mma_available(ggml_cuda_info().devices[device].cc) && (f16 || q8) &&
        q->ne[2] == gqa*k->ne[2] && gqa >= 2 && gqa <= 8 &&
        (group == 1 || (fattn_rel_mma_wide() && group*gqa <= fattn_rel_mma_cols)) &&
        k->nb[1]%align == 0 && k->nb[2]%align == 0 && v->nb[1]%align == 0 && v->nb[2]%align == 0;
}

static bool fattn_rel_mma_aligned(const ggml_tensor * dst) {
    const uintptr_t align = dst->src[1]->type == GGML_TYPE_Q8_0 ? 4 : 16;
    return (uintptr_t) dst->src[1]->data%align == 0 && (uintptr_t) dst->src[2]->data%align == 0;
}

// Each stream stores what a part of the one-cell kernel stores, so one merge reads both.
static size_t fattn_rel_mma_alloc_size(const int device, const ggml_tensor * dst) {
    return fattn_rel_mma_shapes(device, dst) ?
        GGML_PAD(ggml_nbytes(dst), 128) + size_t(dst->ne[1])*dst->ne[2]*fattn_rel_mma_streams(dst)*130*sizeof(float) : 0;
}

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
// The mask word a lane reads at place own of its stream. The streams take the runs of words in turn, so the cells one
// sequence holds side by side are spread over all of them.
static __device__ __forceinline__ int fattn_rel_mma_word(int own, int lane, int stream, int n_streams, int run_log2) {
    return (((own + (lane >> run_log2))*n_streams + stream) << run_log2) + (lane & ((1 << run_log2) - 1));
}

// 8 values of a q8_0 block, 2 words of quants, as halves.
static __device__ __forceinline__ int4 fattn_rel_mma_q8_halves(const uint32_t q0, const uint32_t q1, const float d) {
    half2 h[4];
#pragma unroll
    for (int e = 0; e < 4; ++e) {
        const uint32_t w = e < 2 ? q0 : q1;
        const int s = 16*(e%2);
        h[e] = __floats2half2_rn(d*float(int8_t(w >> s)), d*float(int8_t(w >> (s + 8))));
    }
    int4 r;
    memcpy(&r, h, sizeof(r));
    return r;
}

// A lane stages half of a K row and half of its V row, 64 values each, as halves. Q8_0 is read in place: half a row is
// 2 blocks, 68 bytes, 17 words: the first scale, its quants 2 bytes into a word, the second scale and its quants.
template <ggml_type type>
static __device__ __forceinline__ void fattn_rel_mma_stage(int4 * k_dst, int4 * v_dst, const char * k_src, const char * v_src, int half_row) {
    if constexpr (type == GGML_TYPE_F16) {
        const int4 * k4 = (const int4 *) k_src + 8*half_row;
        const int4 * v4 = (const int4 *) v_src + 8*half_row;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            k_dst[j] = k4[j];
            v_dst[j] = v4[j];
        }
    } else {
        static_assert(type == GGML_TYPE_Q8_0, "bad type");
        const uint32_t * k32 = (const uint32_t *) (k_src + 68*half_row);
        const uint32_t * v32 = (const uint32_t *) (v_src + 68*half_row);
        uint32_t wk[17], wv[17];
#pragma unroll
        for (int j = 0; j < 17; ++j) {
            wk[j] = k32[j];
            wv[j] = v32[j];
        }
#pragma unroll
        for (int kv = 0; kv < 2; ++kv) {
            const uint32_t * w = kv == 0 ? wk : wv;
            int4 * dst = kv == 0 ? k_dst : v_dst;
            const float d0 = __half2float(__ushort_as_half(uint16_t(w[0])));
            const float d1 = __half2float(__ushort_as_half(uint16_t(w[8] >> 16)));
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                dst[j] = fattn_rel_mma_q8_halves(__byte_perm(w[2*j], w[2*j + 1], 0x5432),
                    __byte_perm(w[2*j + 1], w[2*j + 2], 0x5432), d0);
                dst[4 + j] = fattn_rel_mma_q8_halves(w[9 + 2*j], w[10 + 2*j], d1);
            }
        }
    }
}

// One warp gathers 16 visible KV rows into a tile and multiplies it by all the query heads of its KV head on the tensor cores.
// Each row is read once a KV head. Masked cells are never read, so they cannot change how the visible ones are rounded.
// A query group's rows (a token and its drafts) share the tile: a column is one row's head and takes only the cells its
// row's mask holds, in ascending order. Tiles are cut on the group's cells. ncols: tiles of 8 columns, max_group: most rows.
template <ggml_type type, int ncols, int max_group>
static __global__ void fattn_rel_mma_rows(
        const char * q, const char * k, const char * v, const uint32_t * mask, const float * table,
        const int32_t * q_pos, const int32_t * kv_pos, const float * shift, float * partial, float scale, int n_kv, int n_dist, int gqa,
        int group, int run_log2, size_t q_row, size_t q_head, size_t k_row, size_t k_head, size_t v_row, size_t v_head) {
#ifdef AMPERE_MMA_AVAILABLE
    using namespace ggml_cuda_mma;
    constexpr int cells = fattn_rel_mma_cells, words = max_group > 1 ? fattn_rel_mma_words/2 : fattn_rel_mma_words;
    // A staged row is 128 halves and 16 bytes of padding, which spread the rows ldmatrix reads together over the banks.
    constexpr int row_i4 = 17;
    const int lane = threadIdx.x, stream = blockIdx.x, n_streams = gridDim.x;
    const int kv_head = blockIdx.y, query0 = blockIdx.z*group;
    const int head0 = kv_head*gqa, n_heads = gridDim.y*gqa, n_cols = group*gqa;
    __shared__ int4 staged_k[cells*row_i4], staged_v[cells*row_i4];
    __shared__ int list[32*32 + cells];
    // bit g: row g of the group holds the listed cell
    __shared__ uint8_t held[32*32 + cells];
    k += kv_head*k_head;
    v += kv_head*v_head;

    // Q is the B operand: 8 columns by 16 values a tile, zero past the group's columns.
    tile<8, 8, half2> Q_B[ncols][8];
#pragma unroll
    for (int t = 0; t < ncols; ++t) {
        const int col = 8*t + lane/4;
        const bool ok = col < n_cols;
        const float * q_ptr = (const float *) (q + (query0 + (ok ? col/gqa : 0))*q_row + (head0 + (ok ? col%gqa : 0))*q_head);
#pragma unroll
        for (int c = 0; c < 8; ++c) {
#pragma unroll
            for (int l = 0; l < 2; ++l) {
                const int d = 16*c + 2*(4*l + lane%4);
                Q_B[t][c].x[l] = ok ? make_half2(q_ptr[d]*scale, q_ptr[d + 1]*scale) : make_half2(0.0f, 0.0f);
            }
        }
    }

    // As an MMA result is laid out, a lane holds columns 8t + j0 and 8t + j0 + 1 at two cells of the tile, and two values of each chunk.
    const int j0 = 2*(lane%4);
    bool valid[ncols][2];
    int member[ncols][2], pos[ncols][2];
    int row[ncols][2];
    float sink[ncols][2], maximum[ncols][2], denominator[ncols][2];
#pragma unroll
    for (int t = 0; t < ncols; ++t) {
#pragma unroll
        for (int p = 0; p < 2; ++p) {
            const int col = 8*t + j0 + p;
            valid[t][p] = col < n_cols;
            member[t][p] = valid[t][p] ? col/gqa : 0;
            row[t][p] = (query0 + member[t][p])*n_heads + head0 + (valid[t][p] ? col%gqa : 0);
            pos[t][p] = q_pos[query0 + member[t][p]];
            sink[t][p] = shift ? shift[row[t][p]] : 0.0f;
            maximum[t][p] = -FLT_MAX;
            denominator[t][p] = 0.0f;
        }
    }
    tile<16, 8, float> O[ncols][8];

    const int n_words = n_kv/32;
    // A load reads runs_load runs of the stream's n_own, each run 2^run_log2 words.
    const int runs_load = WARP_SIZE >> run_log2;
    const int n_runs = (n_words + (1 << run_log2) - 1) >> run_log2;
    const int n_own = (n_runs - stream + n_streams - 1)/n_streams;
    mask += int64_t(query0)*n_words;
    int fill = 0;
    for (int base = 0; base < n_own; base += runs_load*words) {
        const bool last = base + runs_load*words >= n_own;
        uint32_t live[max_group][words], any = 0;
#pragma unroll
        for (int i = 0; i < words; ++i) {
            const int word = fattn_rel_mma_word(base + runs_load*i, lane, stream, n_streams, run_log2);
#pragma unroll
            for (int g = 0; g < max_group; ++g) {
                live[g][i] = g < group && word < n_words ? mask[g*n_words + word] : 0;
                any |= live[g][i];
            }
        }
        if (!__any_sync(0xffffffff, any != 0) && !(last && fill > 0)) {
            continue;
        }
        // Not unrolled: the tile below is compiled once.
#pragma unroll 1
        for (int i = 0; i < words; ++i) {
            uint32_t own[max_group], cur = 0;
#pragma unroll
            for (int g = 0; g < max_group; ++g) {
                own[g] = live[g][0];
                cur |= own[g];
#pragma unroll
                for (int w = 0; w + 1 < words; ++w) {
                    live[g][w] = live[g][w + 1];
                }
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
                const int word = fattn_rel_mma_word(base + runs_load*i, lane, stream, n_streams, run_log2);
                for (uint32_t bits = cur; bits; bits &= bits - 1) {
                    const int bit = __ffs(bits) - 1;
                    list[at] = 32*word + bit;
                    if constexpr (max_group > 1) {
                        int rows = 0;
#pragma unroll
                        for (int g = 0; g < max_group; ++g) {
                            rows |= (own[g] >> bit & 1) << g;
                        }
                        held[at] = rows;
                    }
                    ++at;
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
                // A slot past n repeats the first cell, and no row holds it.
                const int slot = lane%cells, half_row = lane/cells;
                const int slot_a = lane/4, slot_b = 8 + lane/4;
                const int cell = tile_cells[slot < n ? slot : 0];
                const int cell_a = tile_cells[slot_a < n ? slot_a : 0];
                const int cell_b = tile_cells[slot_b < n ? slot_b : 0];
                const int held_a = slot_a >= n ? 0 : max_group > 1 ? held[done + slot_a] : 1;
                const int held_b = slot_b >= n ? 0 : max_group > 1 ? held[done + slot_b] : 1;
                fattn_rel_mma_stage<type>(staged_k + slot*row_i4 + 8*half_row, staged_v + slot*row_i4 + 8*half_row,
                    k + int64_t(cell)*k_row, v + int64_t(cell)*v_row, half_row);

                const int cell_pos[2] = {kv_pos[cell_a], kv_pos[cell_b]};
                float bias[ncols][4];
                bool use[ncols][4];
#pragma unroll
                for (int t = 0; t < ncols; ++t) {
#pragma unroll
                    for (int l = 0; l < 4; ++l) {
                        const int p = l%2;
                        const int64_t distance = int64_t(pos[t][p]) - cell_pos[l/2];
                        const float b = distance >= 0 && distance < n_dist ? table[int64_t(row[t][p])*n_dist + distance] : 0.0f;
                        bias[t][l] = shift ? __half2float(__float2half(b - sink[t][p])) : b;
                        use[t][l] = valid[t][p] && ((l < 2 ? held_a : held_b) >> member[t][p] & 1);
                    }
                }
                __syncwarp();

                // S[t].x[l] is the logit of cell slot (l/2)*8 + lane/4 for column 8t + j0 + l%2.
                tile<16, 8, float> S[ncols];
#pragma unroll
                for (int c = 0; c < 8; ++c) {
                    tile<16, 8, half2> K_A;
                    load_ldmatrix(K_A, (const half2 *) staged_k + 8*c, 4*row_i4);
#pragma unroll
                    for (int t = 0; t < ncols; ++t) {
                        mma(S[t], K_A, Q_B[t][c]);
                    }
                }
                tile<8, 8, half2> W_B[ncols];
#pragma unroll
                for (int t = 0; t < ncols; ++t) {
#pragma unroll
                    for (int l = 0; l < 4; ++l) {
                        S[t].x[l] = use[t][l] ? S[t].x[l] + bias[t][l] : -FLT_MAX;
                    }
                    float rescale[2];
#pragma unroll
                    for (int p = 0; p < 2; ++p) {
                        float m = fmaxf(maximum[t][p], fmaxf(S[t].x[p], S[t].x[p + 2]));
#pragma unroll
                        for (int offset = 16; offset >= 4; offset >>= 1) {
                            m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, offset, WARP_SIZE));
                        }
                        rescale[p] = expf(maximum[t][p] - m);
                        maximum[t][p] = m;
                    }
#pragma unroll
                    for (int l = 0; l < 4; ++l) {
                        S[t].x[l] = use[t][l] ? expf(S[t].x[l] - maximum[t][l%2]) : 0.0f;
                    }

                    // The value product takes the weights as halves, and the denominator sums the same halves.
                    const tile<16, 4, half2> W = get_half2(S[t]);
                    denominator[t][0] = denominator[t][0]*rescale[0] + __low2float(W.x[0]) + __low2float(W.x[1]);
                    denominator[t][1] = denominator[t][1]*rescale[1] + __high2float(W.x[0]) + __high2float(W.x[1]);
                    if (__any_sync(0xffffffff, rescale[0] != 1.0f || rescale[1] != 1.0f)) {
#pragma unroll
                        for (int c = 0; c < 8; ++c) {
#pragma unroll
                            for (int l = 0; l < 4; ++l) {
                                O[t][c].x[l] *= rescale[l%2];
                            }
                        }
                    }
                    W_B[t] = get_transposed(W);
                }
#pragma unroll
                for (int c = 0; c < 8; ++c) {
                    tile<16, 8, half2> V_A;
                    load_ldmatrix_trans(V_A, (const half2 *) staged_v + 8*c, 4*row_i4);
#pragma unroll
                    for (int t = 0; t < ncols; ++t) {
                        mma(O[t][c], V_A, W_B[t]);
                    }
                }
                __syncwarp();
                done += n;
            }

            // Fewer cells than a tile wait at the head of the list for the next words.
            if (done > 0 && fill > done) {
                const int rest = fill - done;
                const int moved = lane < rest ? list[done + lane] : 0;
                const int moved_held = max_group > 1 && lane < rest ? held[done + lane] : 0;
                __syncwarp();
                if (lane < rest) {
                    list[lane] = moved;
                    if constexpr (max_group > 1) {
                        held[lane] = moved_held;
                    }
                }
                __syncwarp();
            }
            fill -= done;
        }
    }

#pragma unroll
    for (int t = 0; t < ncols; ++t) {
#pragma unroll
        for (int p = 0; p < 2; ++p) {
#pragma unroll
            for (int offset = 16; offset >= 4; offset >>= 1) {
                denominator[t][p] += __shfl_xor_sync(0xffffffff, denominator[t][p], offset, WARP_SIZE);
            }
        }
    }
    // A stream with no visible cell stores weight 0, which the merge adds as nothing.
#pragma unroll
    for (int t = 0; t < ncols; ++t) {
#pragma unroll
        for (int p = 0; p < 2; ++p) {
            if (!valid[t][p]) {
                continue;
            }
            float * out = partial + (int64_t(row[t][p])*n_streams + stream)*130;
            if (lane < 4) {
                out[0] = maximum[t][p];
                out[1] = denominator[t][p];
            }
#pragma unroll
            for (int c = 0; c < 8; ++c) {
                out[2 + 16*c + lane/4] = O[t][c].x[p];
                out[2 + 16*c + 8 + lane/4] = O[t][c].x[p + 2];
            }
        }
    }
#else
    GGML_UNUSED_VARS(q, k, v, mask, table, q_pos, kv_pos, shift, partial, scale, n_kv, n_dist, gqa, group, run_log2,
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
    const int gqa = q->ne[2]/k->ne[2];
    const int group = fattn_rel_decode_group(dst);
    const bool q8 = k->type == GGML_TYPE_Q8_0;
    using kernel_t = decltype(&fattn_rel_mma_rows<GGML_TYPE_F16, 1, 1>);
    kernel_t kernel;
    switch (group == 1 ? 0 : (group*gqa + 7)/8) {
        case 0:  kernel = q8 ? fattn_rel_mma_rows<GGML_TYPE_Q8_0, 1, 1> : fattn_rel_mma_rows<GGML_TYPE_F16, 1, 1>; break;
        case 1:  kernel = q8 ? fattn_rel_mma_rows<GGML_TYPE_Q8_0, 1, 4> : fattn_rel_mma_rows<GGML_TYPE_F16, 1, 4>; break;
        case 2:  kernel = q8 ? fattn_rel_mma_rows<GGML_TYPE_Q8_0, 2, 4> : fattn_rel_mma_rows<GGML_TYPE_F16, 2, 4>; break;
        default: kernel = q8 ? fattn_rel_mma_rows<GGML_TYPE_Q8_0, 3, 4> : fattn_rel_mma_rows<GGML_TYPE_F16, 3, 4>; break;
    }
    kernel<<<dim3(n_streams, k->ne[2], q->ne[1]/group), WARP_SIZE, 0, ctx.stream()>>>(
            (const char *) q->data, (const char *) k->data, (const char *) v->data,
            (const uint32_t *) dst->src[3]->data, (const float *) dst->src[6]->data,
            (const int32_t *) dst->src[7]->data, (const int32_t *) dst->src[8]->data, shift ? sinks : nullptr, partial,
            scale, k->ne[1], dst->src[6]->ne[0], gqa, group, fattn_rel_mma_run_log2(),
            q->nb[1], q->nb[2], k->nb[1], k->nb[2], v->nb[1], v->nb[2]);
    fattn_rel_decode_merge<<<dim3(q->ne[2], q->ne[1]), 128, 0, ctx.stream()>>>(
            partial, shift ? nullptr : sinks, (float *) dst->data, n_streams);
    CUDA_CHECK(cudaGetLastError());
#else
    GGML_UNUSED_VARS(ctx, dst);
    GGML_ABORT("relative mask decode requires CUDA");
#endif
}
