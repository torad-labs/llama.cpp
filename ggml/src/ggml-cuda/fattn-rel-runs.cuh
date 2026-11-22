#pragma once

#include "common.cuh"
#include "cp-async.cuh"
#include "mma.cuh"
#include "fattn-common.cuh"
#include "fattn-rel-decode.cuh"
#include "fattn-rel-elem.cuh"
#include "ggml-attn-mask.h"

// Relative decode over the runs of cells the model gives (ggml_flash_attn_ext_set_kv_runs) in place of a mask scan. A
// warp copies the K and V rows of 16 cells into a tile with cp.async while it multiplies the tile before on the tensor
// cores, as fattn_rel_mma_rows does (a q8_0 or f8_e4m3 cache is loaded into registers instead and converted to f16 as
// it is staged, fattn_rel_elem). A cell 1024 positions or more before its query has no bias: a tile with no near
// cell reads no position and no table. With GGML_CUDA_FATTN_REL_RUNS_SHARED a block reads each tile of the shared list
// once for a group of rows (fattn_rel_runs_shared) and each row's warps read only its other runs.

constexpr int fattn_rel_runs_cells  = 16; // cells a tile
constexpr int fattn_rel_runs_stages = 2;  // tiles a block holds: one is read while the next one is copied
constexpr int fattn_rel_runs_row_i4 = 17; // a staged row: 128 halves and 16 bytes of padding against bank conflicts

// The cache types the tiles are read from: f16, copied as it is, and with GGML_CUDA_FATTN_REL_DECODE_ONE_BYTE q8_0 and
// f8_e4m3, converted to f16 as a tile is staged (fattn_rel_elem), as the mask kernels read them.
static bool fattn_rel_runs_type(const ggml_type type) {
    static const bool one_byte = ggml_env_switch("GGML_CUDA_FATTN_REL_DECODE_ONE_BYTE");
    return type == GGML_TYPE_F16 || (one_byte && fattn_rel_elem_one_byte(type));
}

static bool fattn_rel_runs_hinted(const ggml_tensor * dst) {
    return ggml_flash_attn_ext_get_kv_runs(dst) != nullptr;
}

// GGML_CUDA_FATTN_REL_RUNS_LEGACY=1: the runs are not read, the mask kernels take the step.
static bool fattn_rel_runs_takes(const int device, const ggml_tensor * dst) {
    static const bool legacy = ggml_env_switch("GGML_CUDA_FATTN_REL_RUNS_LEGACY");
    const ggml_tensor * runs = ggml_flash_attn_ext_get_kv_runs(dst);
    if (legacy || !runs || !ampere_mma_available(ggml_cuda_info().devices[device].cc)) {
        return false;
    }
    const auto * q = dst->src[0];
    const auto * k = dst->src[1];
    const auto * v = dst->src[2];
    const int64_t gqa = q->ne[2]/k->ne[2];
    const size_t align = fattn_rel_elem_align(k->type);
    return fattn_rel_runs_type(k->type) && v->type == k->type && q->type == GGML_TYPE_F32 && q->nb[0] == sizeof(float) &&
        q->ne[0] == 128 && k->ne[0] == 128 && v->ne[0] == 128 && q->ne[3] == 1 && k->ne[3] == 1 &&
        q->ne[2] == gqa*k->ne[2] && gqa >= 2 && gqa <= 8 && q->ne[1] <= 32 && fattn_rel_decode_group(dst) == 1 &&
        runs->ne[1] >= 2 && runs->ne[2] == q->ne[1] + 1 && k->ne[1] <= INT32_MAX &&
        k->nb[1]%align == 0 && k->nb[2]%align == 0 && v->nb[1]%align == 0 && v->nb[2]%align == 0;
}

// f16 and f8_e4m3 rows are read 16 bytes at a time, q8_0 rows 8.
static bool fattn_rel_runs_aligned(const ggml_tensor * dst) {
    const size_t align = fattn_rel_elem_align(dst->src[1]->type);
    return (uintptr_t) dst->src[1]->data%align == 0 && (uintptr_t) dst->src[2]->data%align == 0;
}

// GGML_CUDA_FATTN_REL_RUNS_PARTS: warps a row's cells are split across for each KV head, never more than the pool has
// cells/1024. With the shared pass, they are the cells the shared list does not hold.
static int fattn_rel_runs_parts(const ggml_tensor * dst) {
    static const int parts = (int) ggml_env_int("GGML_CUDA_FATTN_REL_RUNS_PARTS");
    return (int) std::max<int64_t>(1, std::min<int64_t>(parts, dst->src[1]->ne[1]/1024));
}

// GGML_CUDA_FATTN_REL_RUNS_SHARED: the shared list is read once for a group of rows, in GGML_CUDA_FATTN_REL_RUNS_SHARED_PARTS
// parts for each KV head and group. 0 parts: no shared pass.
static int fattn_rel_runs_shared_parts(const ggml_tensor * dst) {
    static const bool shared = ggml_env_switch("GGML_CUDA_FATTN_REL_RUNS_SHARED");
    static const int parts = (int) ggml_env_int("GGML_CUDA_FATTN_REL_RUNS_SHARED_PARTS");
    return shared ? (int) std::max<int64_t>(1, std::min<int64_t>(parts, dst->src[1]->ne[1]/1024)) : 0;
}

// GGML_CUDA_FATTN_REL_RUNS_GROUP: the rows a block of the shared pass reads a tile for, a warp for each 8 of their heads.
// A block has 16 warps at most.
static int fattn_rel_runs_group(const ggml_tensor * dst) {
    static const int group = (int) ggml_env_int("GGML_CUDA_FATTN_REL_RUNS_GROUP");
    const int gqa = (int) (dst->src[0]->ne[2]/dst->src[1]->ne[2]);
    return std::max(1, std::min<int>({group, (int) dst->src[0]->ne[1], 128/gqa}));
}

// Each part stores what a part of the one-cell kernel stores, so one merge reads them all.
static size_t fattn_rel_runs_alloc_size(const int device, const ggml_tensor * dst) {
    if (!fattn_rel_runs_takes(device, dst)) {
        return 0;
    }
    const int n_slots = fattn_rel_runs_parts(dst) + fattn_rel_runs_shared_parts(dst);
    return GGML_PAD(ggml_nbytes(dst), 128) + size_t(dst->ne[1])*dst->ne[2]*n_slots*130*sizeof(float);
}

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
// The run of a list that holds a cell of the list: the last run that starts at it or before. It only moves on.
struct fattn_rel_runs_cursor {
    const int4 * list; // entry 0 is the head of the list, entry 1 its first run
    int  i;            // the run
    int4 run;          // (first cell, its position, start in the list, stride or rows)
    int  end;          // where run i + 1 starts

    __device__ __forceinline__ void seek(const int at, const int n_runs) {
        int lo = 1, hi = n_runs;
        while (lo < hi) {
            const int mid = (lo + hi + 1)/2;
            if (list[mid].z <= at) {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        i   = lo;
        run = list[i];
        end = list[i + 1].z;
    }

    __device__ __forceinline__ void move(const int at) {
        while (at >= end) {
            ++i;
            run = list[i];
            end = list[i + 1].z;
        }
    }
};

// A lane's half of a K row and of a V row on their way to a stage. f16 rows go by cp.async and leave nothing here. Rows of
// another type are loaded into registers, and fattn_rel_runs_finish writes them to the stage as f16 once the warp has
// done the work it does while they load.
template <ggml_type type>
struct fattn_rel_runs_pending {
    typename fattn_rel_elem<type>::raw k, v;
    int4 * k_dst = nullptr;
    int4 * v_dst = nullptr;
};

template <ggml_type type>
static __device__ __forceinline__ void fattn_rel_runs_finish(fattn_rel_runs_pending<type> & pending) {
    if constexpr (type != GGML_TYPE_F16) {
        if (pending.k_dst) {
            const int half_row = threadIdx.x/fattn_rel_runs_cells;
            fattn_rel_elem<type>::store(pending.k, half_row, pending.k_dst);
            fattn_rel_elem<type>::store(pending.v, half_row, pending.v_dst);
            pending.k_dst = nullptr;
        }
    } else {
        GGML_UNUSED(pending);
    }
}

// The lanes of a warp copy the K and V rows of the 16 cells of the list from at into a stage, half a row a lane, with the
// position of each cell, INT_MIN past end (such a slot copies a cell of the run again, and has no weight). In the shared
// list the last field of a run is the sequences whose rows read it, which the stage keeps too.
template <ggml_type type, bool shared_list>
static __device__ __forceinline__ void fattn_rel_runs_stage(
        fattn_rel_runs_cursor & cur, const int at, const int end, const char * k, const char * v,
        const size_t k_row, const size_t v_row, int4 * staged_k, int4 * staged_v, int * staged_pos, uint32_t * staged_rows,
        fattn_rel_runs_pending<type> & pending) {
    constexpr int row_i4 = fattn_rel_runs_row_i4;
    const int lane = threadIdx.x;
    const int slot = lane%fattn_rel_runs_cells, half_row = lane/fattn_rel_runs_cells;
    const int at_slot = at + slot;
    if (at_slot < end) {
        cur.move(at_slot);
    }
    int64_t cell = cur.run.x;
    int pos = INT_MIN;
    if (at_slot < end) {
        const int offset = at_slot - cur.run.z;
        cell += shared_list ? int64_t(offset) : int64_t(offset)*(cur.run.w & 0x7fffffff);
        pos = cur.run.y + offset;
    }
    const char * k_src = k + cell*k_row;
    const char * v_src = v + cell*v_row;
    int4 * k_dst = staged_k + slot*row_i4 + 8*half_row;
    int4 * v_dst = staged_v + slot*row_i4 + 8*half_row;
    if constexpr (type == GGML_TYPE_F16) {
        const unsigned int k_dst_s = ggml_cuda_cvta_generic_to_shared(k_dst);
        const unsigned int v_dst_s = ggml_cuda_cvta_generic_to_shared(v_dst);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            cp_async_cg_16<0>(k_dst_s + 16*j, k_src + 128*half_row + 16*j);
            cp_async_cg_16<0>(v_dst_s + 16*j, v_src + 128*half_row + 16*j);
        }
        GGML_UNUSED(pending);
    } else {
        pending.k = fattn_rel_elem<type>::load(k_src, half_row);
        pending.v = fattn_rel_elem<type>::load(v_src, half_row);
        pending.k_dst = k_dst;
        pending.v_dst = v_dst;
    }
    if (half_row == 0) {
        staged_pos[slot] = pos;
        if constexpr (shared_list) {
            staged_rows[slot] = at_slot < end ? uint32_t(cur.run.w) : 0;
        }
    }
}

// The logits of a staged tile against the Q heads in B, then one online softmax step into O. Lane holds heads j0, j0 + 1
// of cells lane/4 and 8 + lane/4. A head takes the bias of a cell only in a tile that has a near cell for some head of the
// warp; a far cell adds no bias, or the rounded sink shift where the sinks are shifted, as the mask kernels add it.
static __device__ __forceinline__ void fattn_rel_runs_tile(
        const ggml_cuda_mma::tile<8, 8, half2> * Q_B, const int4 * staged_k, const int4 * staged_v, const int * staged_pos,
        const int n, const int * pos_q, const bool * use_head, const uint32_t * row_bits, const uint32_t * staged_rows,
        const float * const * bias_row, const float * sink, const float * far, const bool shift, const int n_dist,
        float * maximum, float * denominator, ggml_cuda_mma::tile<16, 8, float> * O) {
#ifdef AMPERE_MMA_AVAILABLE
    using namespace ggml_cuda_mma;
    constexpr int row_i4 = fattn_rel_runs_row_i4;
    const int lane = threadIdx.x;
    const int slots[2] = {lane/4, 8 + lane/4};
    int distance[4];
    bool use[4], near = false;
#pragma unroll
    for (int l = 0; l < 4; ++l) {
        const int p = l%2, slot = slots[l/2];
        const bool live = slot < n && use_head[p] && (!staged_rows || (staged_rows[slot] & row_bits[p]));
        distance[l] = live ? pos_q[p] - staged_pos[slot] : INT_MAX;
        use[l] = live;
        near = near || distance[l] < n_dist;
    }
    near = __any_sync(0xffffffff, near);
    float bias[4];
#pragma unroll
    for (int l = 0; l < 4; ++l) {
        const int p = l%2;
        if (near) {
            const float b = distance[l] >= 0 && distance[l] < n_dist ? bias_row[p][distance[l]] : 0.0f;
            bias[l] = shift ? __half2float(__float2half(b - sink[p])) : b;
        } else {
            bias[l] = far[p];
        }
    }

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
#else
    GGML_UNUSED_VARS(Q_B, staged_k, staged_v, staged_pos, n, pos_q, use_head, row_bits, staged_rows, bias_row, sink, far,
        shift, n_dist, maximum, denominator, O);
    NO_DEVICE_CODE;
#endif // AMPERE_MMA_AVAILABLE
}

// The denominators summed over the lanes of a column, and the partial of each column a lane holds stored.
static __device__ __forceinline__ void fattn_rel_runs_store(
        const float * maximum, float * denominator, const ggml_cuda_mma::tile<16, 8, float> * O, const bool * valid,
        const int64_t * row, float * partial, const int n_slots, const int slot) {
    const int lane = threadIdx.x;
#pragma unroll
    for (int p = 0; p < 2; ++p) {
#pragma unroll
        for (int offset = 16; offset >= 4; offset >>= 1) {
            denominator[p] += __shfl_xor_sync(0xffffffff, denominator[p], offset, WARP_SIZE);
        }
    }
    // A part with no cell stores weight 0, which the merge adds as nothing.
#pragma unroll
    for (int p = 0; p < 2; ++p) {
        if (!valid[p]) {
            continue;
        }
        float * out = partial + (row[p]*n_slots + slot)*130;
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
}

// One warp a part of a row's runs and a KV head, multiplying all the query heads of the KV head at once. After the shared
// pass it starts at the row's first run the shared list does not hold.
template <ggml_type type, bool query_first>
static __global__ void fattn_rel_runs_rows(
        const char * __restrict__ q, const char * __restrict__ k, const char * __restrict__ v, const int4 * __restrict__ runs,
        const float * __restrict__ table, const int32_t * __restrict__ q_pos, const float * __restrict__ shift,
        float * __restrict__ partial, const float scale, const int n_entries, const int n_dist, const int gqa,
        const int slot0, const int n_slots, const bool after_shared,
        const size_t q_row, const size_t q_head, const size_t k_row, const size_t k_head, const size_t v_row, const size_t v_head) {
#if defined(AMPERE_MMA_AVAILABLE) && defined(CP_ASYNC_AVAILABLE)
    using namespace ggml_cuda_mma;
    constexpr int cells = fattn_rel_runs_cells, stages = fattn_rel_runs_stages, row_i4 = fattn_rel_runs_row_i4;
    const int lane = threadIdx.x;
    const int part = query_first ? blockIdx.z : blockIdx.x;
    const int n_parts = query_first ? gridDim.z : gridDim.x;
    const int kv_head = blockIdx.y, query = query_first ? blockIdx.x : blockIdx.z;
    const int head0 = kv_head*gqa, n_heads = gridDim.y*gqa;
    const int64_t row0 = int64_t(query)*n_heads + head0;
    __shared__ int4 staged_k[stages][cells*row_i4], staged_v[stages][cells*row_i4];
    __shared__ int staged_pos[stages][cells];
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

    const int j0 = 2*(lane%4);
    bool valid[2];
    int pos_q[2];
    int64_t row[2];
    float sink[2], far[2], maximum[2], denominator[2];
    const float * bias_row[2];
#pragma unroll
    for (int p = 0; p < 2; ++p) {
        valid[p] = j0 + p < gqa;
        row[p] = row0 + (valid[p] ? j0 + p : 0);
        sink[p] = shift ? shift[row[p]] : 0.0f;
        far[p] = shift ? __half2float(__float2half(0.0f - sink[p])) : 0.0f;
        bias_row[p] = table + row[p]*n_dist;
        pos_q[p] = q_pos[query];
        maximum[p] = -FLT_MAX;
        denominator[p] = 0.0f;
    }
    tile<16, 8, float> O[8];

    fattn_rel_runs_cursor cur;
    cur.list = runs + int64_t(query)*n_entries;
    const int4 head = cur.list[0];
    const int first = after_shared ? head.z : 0;
    const int n_tiles = (head.y - first + cells - 1)/cells;
    const int per_part = (n_tiles + n_parts - 1)/n_parts;
    const int begin = first + part*per_part*cells;
    const int end = min(head.y, begin + per_part*cells);

    if (begin < end) {
        fattn_rel_runs_pending<type> pending;
        cur.seek(begin, head.x);
#pragma unroll
        for (int t = 0; t + 1 < stages; ++t) {
            if (begin + t*cells < end) {
                fattn_rel_runs_stage<type, false>(cur, begin + t*cells, end, k, v, k_row, v_row, staged_k[t], staged_v[t],
                        staged_pos[t], nullptr, pending);
                fattn_rel_runs_finish(pending);
            }
            cp_async_commit_group();
        }
        for (int at = begin, s = 0; at < end; at += cells, s = (s + 1)%stages) {
            cp_async_wait_group<stages - 2>();
            __syncwarp();
            // the next tile goes to the stage the warp read last
            if (at + (stages - 1)*cells < end) {
                const int t = (s + stages - 1)%stages;
                fattn_rel_runs_stage<type, false>(cur, at + (stages - 1)*cells, end, k, v, k_row, v_row,
                        staged_k[t], staged_v[t], staged_pos[t], nullptr, pending);
            }
            cp_async_commit_group();
            fattn_rel_runs_tile(Q_B, staged_k[s], staged_v[s], staged_pos[s], min(cells, end - at), pos_q, valid, nullptr,
                    nullptr, bias_row, sink, far, shift != nullptr, n_dist, maximum, denominator, O);
            fattn_rel_runs_finish(pending);
        }
    }
    fattn_rel_runs_store(maximum, denominator, O, valid, row, partial, n_slots, slot0 + part);
#else
    GGML_UNUSED_VARS(q, k, v, runs, table, q_pos, shift, partial, scale, n_entries, n_dist, gqa, slot0, n_slots,
        after_shared, q_row, q_head, k_row, k_head, v_row, v_head);
    NO_DEVICE_CODE;
#endif // defined(AMPERE_MMA_AVAILABLE) && defined(CP_ASYNC_AVAILABLE)
}

// The shared list, a part a block and KV head: warp 0 copies each tile once, and warp w multiplies it by the heads of
// columns 8w to 8w + 7 of the block's group of rows (row-major, gqa heads a row), each column a (row, head) of its own.
template <ggml_type type, bool query_first>
__launch_bounds__(16*WARP_SIZE, 1)
static __global__ void fattn_rel_runs_shared(
        const char * __restrict__ q, const char * __restrict__ k, const char * __restrict__ v, const int4 * __restrict__ runs,
        const float * __restrict__ table, const int32_t * __restrict__ q_pos, const float * __restrict__ shift,
        float * __restrict__ partial, const float scale, const int n_entries, const int n_dist, const int gqa,
        const int n_rows, const int group, const int n_slots,
        const size_t q_row, const size_t q_head, const size_t k_row, const size_t k_head, const size_t v_row, const size_t v_head) {
#if defined(AMPERE_MMA_AVAILABLE) && defined(CP_ASYNC_AVAILABLE)
    using namespace ggml_cuda_mma;
    constexpr int cells = fattn_rel_runs_cells, stages = fattn_rel_runs_stages, row_i4 = fattn_rel_runs_row_i4;
    const int lane = threadIdx.x, warp = threadIdx.y;
    const int part = query_first ? blockIdx.z : blockIdx.x;
    const int n_parts = query_first ? gridDim.z : gridDim.x;
    const int kv_head = blockIdx.y, row_first = (query_first ? blockIdx.x : blockIdx.z)*group;
    const int n_heads = gridDim.y*gqa;
    __shared__ int4 staged_k[stages][cells*row_i4], staged_v[stages][cells*row_i4];
    __shared__ int staged_pos[stages][cells];
    __shared__ uint32_t staged_rows[stages][cells];
    k += kv_head*k_head;
    v += kv_head*v_head;

    // the row and the Q head of column c of the warp, or -1 for a column past the group
    const auto column = [&](const int c, int & head) {
        const int g = (8*warp + c)/gqa;
        head = kv_head*gqa + (8*warp + c)%gqa;
        return g < group && row_first + g < n_rows ? row_first + g : -1;
    };

    tile<8, 8, half2> Q_B[8];
    {
        int h;
        const int r = column(lane/4, h);
        const float * q_ptr = (const float *) (q + (r >= 0 ? r : 0)*q_row + h*q_head);
#pragma unroll
        for (int c = 0; c < 8; ++c) {
#pragma unroll
            for (int l = 0; l < 2; ++l) {
                const int d = 16*c + 2*(4*l + lane%4);
                Q_B[c].x[l] = r >= 0 ? make_half2(q_ptr[d]*scale, q_ptr[d + 1]*scale) : make_half2(0.0f, 0.0f);
            }
        }
    }

    const int j0 = 2*(lane%4);
    bool valid[2];
    int pos_q[2];
    uint32_t row_bits[2];
    int64_t row[2];
    float sink[2], far[2], maximum[2], denominator[2];
    const float * bias_row[2];
#pragma unroll
    for (int p = 0; p < 2; ++p) {
        int h;
        const int r = column(j0 + p, h);
        valid[p] = r >= 0;
        row[p] = int64_t(valid[p] ? r : 0)*n_heads + h;
        // a shared run names the sequences whose rows read it: a token and its drafts are rows of one sequence
        row_bits[p] = valid[p] ? 1u << (runs[int64_t(r)*n_entries].w & 31) : 0;
        sink[p] = shift ? shift[row[p]] : 0.0f;
        far[p] = shift ? __half2float(__float2half(0.0f - sink[p])) : 0.0f;
        bias_row[p] = table + row[p]*n_dist;
        pos_q[p] = q_pos[valid[p] ? r : 0];
        maximum[p] = -FLT_MAX;
        denominator[p] = 0.0f;
    }
    tile<16, 8, float> O[8];

    fattn_rel_runs_cursor cur;
    cur.list = runs + int64_t(n_rows)*n_entries;
    const int4 head = cur.list[0];
    const int n_tiles = (head.y + cells - 1)/cells;
    const int per_part = (n_tiles + n_parts - 1)/n_parts;
    const int begin = part*per_part*cells;
    const int end = min(head.y, begin + per_part*cells);

    if (begin < end) {
        fattn_rel_runs_pending<type> pending;
        if (warp == 0) {
            cur.seek(begin, head.x);
#pragma unroll
            for (int t = 0; t + 1 < stages; ++t) {
                if (begin + t*cells < end) {
                    fattn_rel_runs_stage<type, true>(cur, begin + t*cells, end, k, v, k_row, v_row, staged_k[t], staged_v[t],
                            staged_pos[t], staged_rows[t], pending);
                    fattn_rel_runs_finish(pending);
                }
                cp_async_commit_group();
            }
        }
        for (int at = begin, s = 0; at < end; at += cells, s = (s + 1)%stages) {
            if (warp == 0) {
                cp_async_wait_group<stages - 2>();
            }
            // the tile is in place for every warp, and every warp is done with the stage the next copy takes
            __syncthreads();
            if (warp == 0) {
                if (at + (stages - 1)*cells < end) {
                    const int t = (s + stages - 1)%stages;
                    fattn_rel_runs_stage<type, true>(cur, at + (stages - 1)*cells, end, k, v, k_row, v_row,
                            staged_k[t], staged_v[t], staged_pos[t], staged_rows[t], pending);
                }
                cp_async_commit_group();
            }
            fattn_rel_runs_tile(Q_B, staged_k[s], staged_v[s], staged_pos[s], min(cells, end - at), pos_q, valid, row_bits,
                    staged_rows[s], bias_row, sink, far, shift != nullptr, n_dist, maximum, denominator, O);
            // the next tile's rows, loaded while this one was read, go to its stage, which no warp reads until the barrier
            if (warp == 0) {
                fattn_rel_runs_finish(pending);
            }
        }
    }
    fattn_rel_runs_store(maximum, denominator, O, valid, row, partial, n_slots, part);
#else
    GGML_UNUSED_VARS(q, k, v, runs, table, q_pos, shift, partial, scale, n_entries, n_dist, gqa, n_rows, group, n_slots,
        q_row, q_head, k_row, k_head, v_row, v_head);
    NO_DEVICE_CODE;
#endif // defined(AMPERE_MMA_AVAILABLE) && defined(CP_ASYNC_AVAILABLE)
}
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
template <ggml_type type>
static void ggml_cuda_fattn_rel_runs_type(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const auto * q = dst->src[0];
    const auto * k = dst->src[1];
    const auto * v = dst->src[2];
    const auto * runs = ggml_flash_attn_ext_get_kv_runs(dst);
    const int parts = fattn_rel_runs_parts(dst), shared_parts = fattn_rel_runs_shared_parts(dst);
    const int n_slots = shared_parts + parts;
    float * partial = (float *) ((char *) dst->data + GGML_PAD(ggml_nbytes(dst), 128));
    float scale;
    memcpy(&scale, dst->op_params, sizeof(scale));
    const auto * sinks = (const float *) dst->src[9]->data;
    const bool shift = ggml_cuda_fattn_shift_query_sinks();
    // the blocks of one cell range of all rows run side by side, as GGML_CUDA_FATTN_REL_QUERY_FIRST orders the mask kernel's
    static const bool query_first = ggml_env_switch("GGML_CUDA_FATTN_REL_QUERY_FIRST");
    const int gqa = (int) (q->ne[2]/k->ne[2]), n_rows = (int) q->ne[1], n_kv_heads = (int) k->ne[2];
    const int n_entries = (int) runs->ne[1], n_dist = (int) dst->src[6]->ne[0];
    const int4 * runs_d = (const int4 *) runs->data;
    const float * table = (const float *) dst->src[6]->data;
    const int32_t * q_pos = (const int32_t *) dst->src[7]->data;
    const float * shift_d = shift ? sinks : nullptr;
    if (shared_parts > 0) {
        const int group = fattn_rel_runs_group(dst);
        const int groups = (n_rows + group - 1)/group;
        const dim3 block(WARP_SIZE, (group*gqa + 7)/8);
        const auto kernel = query_first ? fattn_rel_runs_shared<type, true> : fattn_rel_runs_shared<type, false>;
        kernel<<<query_first ? dim3(groups, n_kv_heads, shared_parts) : dim3(shared_parts, n_kv_heads, groups), block, 0, ctx.stream()>>>(
                (const char *) q->data, (const char *) k->data, (const char *) v->data, runs_d, table, q_pos, shift_d, partial,
                scale, n_entries, n_dist, gqa, n_rows, group, n_slots, q->nb[1], q->nb[2], k->nb[1], k->nb[2], v->nb[1], v->nb[2]);
    }
    const auto kernel = query_first ? fattn_rel_runs_rows<type, true> : fattn_rel_runs_rows<type, false>;
    kernel<<<query_first ? dim3(n_rows, n_kv_heads, parts) : dim3(parts, n_kv_heads, n_rows), WARP_SIZE, 0, ctx.stream()>>>(
            (const char *) q->data, (const char *) k->data, (const char *) v->data, runs_d, table, q_pos, shift_d, partial,
            scale, n_entries, n_dist, gqa, shared_parts, n_slots, shared_parts > 0,
            q->nb[1], q->nb[2], k->nb[1], k->nb[2], v->nb[1], v->nb[2]);
    fattn_rel_decode_merge<<<dim3(q->ne[2], q->ne[1]), 128, 0, ctx.stream()>>>(
            partial, shift ? nullptr : sinks, (float *) dst->data, n_slots);
    CUDA_CHECK(cudaGetLastError());
}
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

static void ggml_cuda_fattn_rel_runs(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    switch (dst->src[1]->type) {
        case GGML_TYPE_Q8_0:
            ggml_cuda_fattn_rel_runs_type<GGML_TYPE_Q8_0>(ctx, dst);
            break;
#ifdef FATTN_REL_ELEM_F8
        case GGML_TYPE_F8_E4M3:
            ggml_cuda_fattn_rel_runs_type<GGML_TYPE_F8_E4M3>(ctx, dst);
            break;
#endif // FATTN_REL_ELEM_F8
        default:
            ggml_cuda_fattn_rel_runs_type<GGML_TYPE_F16>(ctx, dst);
            break;
    }
#else
    GGML_UNUSED_VARS(ctx, dst);
    GGML_ABORT("relative runs decode requires CUDA");
#endif
}
