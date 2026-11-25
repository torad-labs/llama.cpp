#pragma once

#include "common.cuh"
#include "cp-async.cuh"
#include "mma.cuh"
#include "fattn-common.cuh"
#include "fattn-rel-decode.cuh"
#include "fattn-rel-elem.cuh"
#include "ggml-attn-mask.h"

#include <cstdlib>
#include <map>
#include <mutex>
#include <tuple>

// Relative decode by cell runs instead of a mask scan. Private warps stage K/V panels of their own tiles;
// shared and drafted rows reuse a staged tile across warps. Only tiles with near cells look up relative bias.

constexpr int fattn_rel_runs_cells  = 16; // cells a tile
constexpr int fattn_rel_runs_stages = 2;  // tiles a block holds: one is read while the next one is copied
constexpr int fattn_rel_runs_row_i4 = 17; // a staged row: 128 halves and 16 bytes of padding against bank conflicts

constexpr int fattn_rel_runs_private_warps  = 4;
constexpr int fattn_rel_runs_private_stages = 3;
constexpr int fattn_rel_runs_panel_i4 = 5; // 32 halves and 16 bytes of padding

// The cache types the tiles are read from: f16, copied as it is, and with GGML_CUDA_FATTN_REL_DECODE_ONE_BYTE q8_0 and
// f8_e4m3, and q4_0 under its own switch (fattn_rel_elem_q4), converted to f16 as a tile is staged (fattn_rel_elem), as
// the mask kernels read them.
static bool fattn_rel_runs_type(const ggml_type type) {
    static const bool one_byte = ggml_env_switch("GGML_CUDA_FATTN_REL_DECODE_ONE_BYTE");
    return type == GGML_TYPE_F16 || (one_byte && fattn_rel_elem_one_byte(type)) || fattn_rel_elem_q4(type);
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
    const int q_group = fattn_rel_decode_group(dst);
    const size_t align = fattn_rel_elem_align(k->type);
    const ggml_tensor * hot_kv = ggml_flash_attn_ext_get_hot_kv(dst);
    const bool hot_ok = !hot_kv || (hot_kv->nb[1]%16 == 0 && hot_kv->nb[2]%16 == 0);
    return fattn_rel_runs_type(k->type) && v->type == k->type && q->type == GGML_TYPE_F32 && q->nb[0] == sizeof(float) && hot_ok &&
        q->ne[0] == 128 && k->ne[0] == 128 && v->ne[0] == 128 && q->ne[3] == 1 && k->ne[3] == 1 &&
        q->ne[2] == gqa*k->ne[2] && gqa >= 2 && gqa <= 8 && q->ne[1] <= 32*q_group &&
        runs->ne[1] >= 2 && runs->ne[2] == q->ne[1] + 1 && k->ne[1] <= INT32_MAX &&
        k->nb[1]%align == 0 && k->nb[2]%align == 0 && v->nb[1]%align == 0 && v->nb[2]%align == 0;
}

// f16 and f8_e4m3 rows are read 16 bytes at a time, q8_0 rows 8.
static bool fattn_rel_runs_aligned(const ggml_tensor * dst) {
    const size_t align = fattn_rel_elem_align(dst->src[1]->type);
    const ggml_tensor * hot_kv = ggml_flash_attn_ext_get_hot_kv(dst);
    return (uintptr_t) dst->src[1]->data%align == 0 && (uintptr_t) dst->src[2]->data%align == 0 &&
        (!hot_kv || (uintptr_t) hot_kv->data%16 == 0);
}

static bool fattn_rel_runs_env_set(const char * name) {
    const char * s = getenv(name);
    if (!s) {
        return false;
    }
    while (*s == ' ' || *s == '\t' || *s == '\r' || *s == '\n') {
        ++s;
    }
    return *s != '\0';
}

// GGML_CUDA_FATTN_REL_RUNS_GROUP: rows a block of the shared pass reads a tile for, up to 18 warps (144 columns).
static int fattn_rel_runs_group(const ggml_tensor * dst) {
    static const int env_group = (int) ggml_env_int("GGML_CUDA_FATTN_REL_RUNS_GROUP");
    static const bool has_env  = fattn_rel_runs_env_set("GGML_CUDA_FATTN_REL_RUNS_GROUP");
    const int gqa     = (int) (dst->src[0]->ne[2]/dst->src[1]->ne[2]);
    const int n_rows  = (int) dst->src[0]->ne[1];
    const int q_group = fattn_rel_decode_group(dst);
    const int max_rows = 144/std::max(1, gqa);
    if (has_env) {
        return std::max(1, std::min({env_group, n_rows, max_rows}));
    }
    if (gqa == 6 && n_rows <= 24) {
        return std::max(1, std::min(n_rows, max_rows));
    }
    if (q_group > 1) {
        const int seqs_per_cta = std::max(1, max_rows/q_group);
        return std::max(1, std::min(n_rows, seqs_per_cta*q_group));
    }
    return std::max(1, std::min({env_group, n_rows, max_rows}));
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

struct fattn_rel_runs_panel_cells {
    int row[fattn_rel_runs_cells]; // cold cell index, or bitwise complement of hot slot
    int pos[fattn_rel_runs_cells];
};

static __device__ __forceinline__ void fattn_rel_runs_panel_resolve(
        fattn_rel_runs_cursor & cur, const int at, const int end, const int32_t * hot_map, fattn_rel_runs_panel_cells & meta) {
    const int lane = threadIdx.x;
    if (lane < fattn_rel_runs_cells) {
        int64_t cell = cur.run.x;
        int pos = INT_MIN;
        if (lane < end - at) {
            cur.move(at + lane);
            const int offset = at + lane - cur.run.z;
            cell = cur.run.x + int64_t(offset)*(cur.run.w & 0x7fffffff);
            pos = cur.run.y + offset;
        }
        const int slot = hot_map ? hot_map[cell] : -1;
        meta.row[lane] = slot >= 0 ? ~slot : int(cell);
        meta.pos[lane] = pos;
    }
    __syncwarp();
}

template <ggml_type type>
static __device__ __forceinline__ void fattn_rel_runs_panel_copy(
        int4 * dst, const fattn_rel_runs_panel_cells & meta, const int b, const bool is_v,
        const char * src, const size_t stride, const char * hot_kv, const size_t hot_row) {
    if constexpr (type == GGML_TYPE_F8_E4M3) {
        if (!hot_kv) {
            const int cell = threadIdx.x/2, j = threadIdx.x%2;
            const unsigned int addr = ggml_cuda_cvta_generic_to_shared(dst + cell*fattn_rel_runs_panel_i4);
            cp_async_cg_16<0>(addr + 64 - 16*j, src + int64_t(meta.row[cell])*stride + 32*b + 16*j);
            return;
        }
    }
    constexpr int vectors = type == GGML_TYPE_Q8_0 ? 5 : 4;
#pragma unroll
    for (int i = 0; i < (vectors*fattn_rel_runs_cells + WARP_SIZE - 1)/WARP_SIZE; ++i) {
        const int item = threadIdx.x + i*WARP_SIZE;
        if (item >= vectors*fattn_rel_runs_cells) {
            continue;
        }
        const int cell = item/vectors, j = item%vectors;
        const int row = meta.row[cell];
        const char * ptr = row >= 0 ? src + int64_t(row)*stride : hot_kv + (2*int64_t(~row) + is_v)*hot_row;
        const unsigned int addr = ggml_cuda_cvta_generic_to_shared(dst + cell*fattn_rel_runs_panel_i4);
        if (type == GGML_TYPE_F16 || row < 0) {
            if (j < 4) {
                cp_async_cg_16<0>(addr + 16*j, ptr + 64*b + 16*j);
            }
        } else if constexpr (type == GGML_TYPE_Q8_0) {
            cp_async_ca_8(addr + 40 + 8*j, ptr + 32*b + 8*j);
        } else if constexpr (type == GGML_TYPE_Q4_0) {
            if (j < 3) {
                cp_async_ca_8(addr + 56 + 8*j, ptr + 16*b + 8*j);
            }
        } else if (j < 2) {
            // Expand each FP8 half over only its own raw bytes.
            cp_async_cg_16<0>(addr + 64 - 16*j, ptr + 32*b + 16*j);
        }
    }
}

template <ggml_type type>
static __device__ __forceinline__ void fattn_rel_runs_panel_expand(
        int4 * panel, const fattn_rel_runs_panel_cells & meta, const int b, const char * packed = nullptr) {
    if constexpr (type == GGML_TYPE_F16) {
        GGML_UNUSED_VARS(panel, meta, b);
    } else {
        const int cell = threadIdx.x/2, half_row = threadIdx.x%2;
        int4 * dst = panel + cell*fattn_rel_runs_panel_i4;
        const bool cold = meta.row[cell] >= 0;
        int4 lo = {}, hi = {};
        if (cold) {
            if constexpr (type == GGML_TYPE_F8_E4M3) {
                const int4 r = packed ? ((const int4 *) packed)[cell*8 + 2*b + half_row] : dst[4 - half_row];
                lo = make_int4(fattn_rel_elem<type>::pair(r.x), fattn_rel_elem<type>::pair(uint32_t(r.x) >> 16),
                               fattn_rel_elem<type>::pair(r.y), fattn_rel_elem<type>::pair(uint32_t(r.y) >> 16));
                hi = make_int4(fattn_rel_elem<type>::pair(r.z), fattn_rel_elem<type>::pair(uint32_t(r.z) >> 16),
                               fattn_rel_elem<type>::pair(r.w), fattn_rel_elem<type>::pair(uint32_t(r.w) >> 16));
            } else {
                constexpr bool q8 = type == GGML_TYPE_Q8_0;
                const char * bytes = (const char *) dst + (q8 ? 40 : 56);
                if constexpr (q8) {
                    if (packed) {
                        bytes = packed + 136*cell + 32*b;
                    }
                }
                const uint32_t * raw = (const uint32_t *) bytes;
                // The aligned copy starts 2*b bytes before the half scale.
                const half2 d = __half2half2(__ushort_as_half(uint16_t(raw[b/2] >> (16*(b%2)))));
                const int off = 2*b + 2 + (q8 ? 16*half_row : 0);
                const half2 bias = q8 ? make_half2(1152.0f, 1152.0f) : make_half2(1032.0f, 1032.0f);
                half2 values[8];
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int w = off/4 + j;
                    uint32_t qs = off%4 ? __byte_perm(raw[w], raw[w + 1], 0x5432) : raw[w];
                    qs = q8 ? qs ^ 0x80808080u : (qs >> (4*half_row)) & 0x0f0f0f0fu;
                    const uint32_t x = __byte_perm(qs, 0x64646464u, 0x4140);
                    const uint32_t y = __byte_perm(qs, 0x64646464u, 0x4342);
                    values[2*j + 0] = __hmul2(__hsub2(*((const half2 *) &x), bias), d);
                    values[2*j + 1] = __hmul2(__hsub2(*((const half2 *) &y), bias), d);
                }
                ggml_cuda_memcpy_1<16>(&lo, values);
                ggml_cuda_memcpy_1<16>(&hi, values + 4);
            }
        }
        if constexpr (type == GGML_TYPE_Q4_0 || type == GGML_TYPE_Q8_0) {
            // Both halves must load overlapping raw bytes before writing.
            if (!packed) {
                __syncwarp();
            }
        }
        if (cold) {
            dst[2*half_row + 0] = lo;
            dst[2*half_row + 1] = hi;
        }
        __syncwarp();
    }
}

// Cooperative cp.async staging across all warps of the block (require blockDim.y >= 2 so nthreads >= 64).
template <ggml_type type, bool shared_list>
static __device__ __forceinline__ void fattn_rel_runs_stage_coop(
        fattn_rel_runs_cursor & cur, const int at, const int end, const char * k, const char * v,
        const size_t k_row, const size_t v_row, int4 * staged_k, int4 * staged_v, int * staged_pos, uint32_t * staged_rows,
        uint8_t * staged_hot, const char * hot_kv, const size_t hot_row, const int32_t * hot_map) {
    constexpr int cells = fattn_rel_runs_cells, row_i4 = fattn_rel_runs_row_i4;
    constexpr int max_per_row = type == GGML_TYPE_Q8_0 ? 17 : 16;
    const int tid = threadIdx.y*WARP_SIZE + threadIdx.x;
    const int nthreads = blockDim.y*WARP_SIZE;
    if constexpr (type == GGML_TYPE_F16) {
        GGML_UNUSED(staged_hot);
    }
    for (int c = tid; c < 2*max_per_row*cells; c += nthreads) {
        const int slot = c / (2*max_per_row);
        const int rem  = c - slot*(2*max_per_row);
        const int is_v = rem >= max_per_row ? 1 : 0;
        const int j    = rem - is_v*max_per_row;
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
        const int32_t hslot = hot_map ? hot_map[cell] : -1;
        const char * src = hslot >= 0
            ? hot_kv + (2*int64_t(hslot) + is_v)*hot_row
            : (is_v ? v + cell*v_row : k + cell*k_row);
        const unsigned int row_s = ggml_cuda_cvta_generic_to_shared((is_v ? staged_v : staged_k) + slot*row_i4);
        if (type == GGML_TYPE_F16 || hslot >= 0) {
            if (j < 16) {
                cp_async_cg_16<0>(row_s + 16*j, src + 16*j);
            }
        } else if constexpr (type == GGML_TYPE_Q8_0) {
            cp_async_ca_8(row_s + 120 + 8*j, src + 8*j);
        } else if constexpr (type == GGML_TYPE_Q4_0) {
            if (j < 9) {
                cp_async_ca_8(row_s + 184 + 8*j, src + 8*j);
            }
        } else {
            if (j < 8) {
                cp_async_cg_16<0>(row_s + 128 + 16*j, src + 16*j);
            }
        }
        if (rem == 0) {
            staged_pos[slot] = pos;
            if constexpr (type != GGML_TYPE_F16) {
                staged_hot[slot] = uint8_t(hslot >= 0);
            }
            if constexpr (shared_list) {
                staged_rows[slot] = at_slot < end ? uint32_t(cur.run.w) : 0;
            }
        }
    }
}

template <ggml_type type>
static __device__ __forceinline__ void fattn_rel_runs_dequant_coop(
        int4 * staged_k, int4 * staged_v, const uint8_t * staged_hot) {
    if constexpr (type != GGML_TYPE_F16) {
        if (threadIdx.y < 2) {
            const int lane = threadIdx.x;
            const int slot = lane >> 1;
            const int half_row = lane & 1;
            int4 * row_ptr = (threadIdx.y ? staged_v : staged_k) + slot*fattn_rel_runs_row_i4;
            const bool is_cold = !staged_hot[slot];
            typename fattn_rel_elem<type>::raw r = {};
            if (is_cold) {
                if constexpr (type == GGML_TYPE_Q8_0) {
                    const int2 * smem = (const int2 *) ((const char *) row_ptr + 120) + 8*half_row;
#pragma unroll
                    for (int l = 0; l < 9; ++l) {
                        r.x[l] = smem[l];
                    }
                } else if constexpr (type == GGML_TYPE_Q4_0) {
                    const int2 * smem = (const int2 *) ((const char *) row_ptr + 184) + 4*half_row;
#pragma unroll
                    for (int l = 0; l < 5; ++l) {
                        r.x[l] = smem[l];
                    }
                } else {
                    const int4 * smem = (const int4 *) ((const char *) row_ptr + 128) + 4*half_row;
#pragma unroll
                    for (int l = 0; l < 4; ++l) {
                        r.x[l] = smem[l];
                    }
                }
            }
            __syncwarp();
            if (is_cold) {
                fattn_rel_elem<type>::store(r, half_row, row_ptr + 8*half_row);
            }
        }
    } else {
        GGML_UNUSED_VARS(staged_k, staged_v, staged_hot);
    }
}

#ifdef AMPERE_MMA_AVAILABLE
// Lane holds two Q heads of cells lane/4 and 8 + lane/4. Only a tile with a near cell reads the bias table.
template <bool use_log2, bool swa_mask>
static __device__ __forceinline__ ggml_cuda_mma::tile<8, 8, half2> fattn_rel_runs_weights(
        ggml_cuda_mma::tile<16, 8, float> S, const int * staged_pos, const int n, const int * pos_q, const int pos_q_min,
        const bool * use_head, const uint32_t * row_bits, const uint32_t * staged_rows, const float * const * bias_row,
        const float * sink, const float * far, const bool shift, const int n_dist,
        float * maximum, float * denominator, ggml_cuda_mma::tile<16, 8, float> * O) {
    using namespace ggml_cuda_mma;
    constexpr float LOG2E = 1.4426950408889634f;
    const int lane = threadIdx.x;
    const int slots[2] = {lane/4, 8 + lane/4};
    bool use[4];
    float bias[4];

    const int p_slot = lane < n ? staged_pos[lane] : INT_MIN;
    const int tile_max_pos = __reduce_max_sync(0xffffffff, p_slot);
    if (!swa_mask && (pos_q_min == INT_MAX || int64_t(pos_q_min) - int64_t(tile_max_pos) >= n_dist)) {
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const int p = l%2, slot = slots[l/2];
            use[l]  = slot < n && use_head[p] && (!staged_rows || (staged_rows[slot] & row_bits[p]));
            bias[l] = far[p];
        }
    } else {
        int distance[4];
        bool near = false;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const int p = l%2, slot = slots[l/2];
            const bool base_live = slot < n && use_head[p] && (!staged_rows || (staged_rows[slot] & row_bits[p]));
            const int dist = base_live ? pos_q[p] - staged_pos[slot] : INT_MAX;
            const bool live = base_live && dist >= 0 && (!swa_mask || dist < n_dist);
            distance[l] = live ? dist : INT_MAX;
            use[l] = live;
            near = near || distance[l] < n_dist;
        }
        near = __any_sync(0xffffffff, near);
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const int p = l%2;
            if (near) {
                const float b = distance[l] >= 0 && distance[l] < n_dist ? bias_row[p][distance[l]] : 0.0f;
                const float b_eff = shift ? __half2float(__float2half(b - sink[p])) : b;
                bias[l] = use_log2 ? b_eff*LOG2E : b_eff;
            } else {
                bias[l] = far[p];
            }
        }
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
        rescale[p] = use_log2 ? exp2f(maximum[p] - m) : expf(maximum[p] - m);
        maximum[p] = m;
    }
#pragma unroll
    for (int l = 0; l < 4; ++l) {
        S.x[l] = use[l] ? (use_log2 ? exp2f(S.x[l] - maximum[l%2]) : expf(S.x[l] - maximum[l%2])) : 0.0f;
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
    return get_transposed(W);
}
#endif // AMPERE_MMA_AVAILABLE

template <bool use_log2, bool swa_mask = false>
static __device__ __forceinline__ void fattn_rel_runs_tile(
        const ggml_cuda_mma::tile<8, 8, half2> * Q_B, const int4 * staged_k, const int4 * staged_v, const int * staged_pos,
        const int n, const int * pos_q, const int pos_q_min, const bool * use_head, const uint32_t * row_bits,
        const uint32_t * staged_rows, const float * const * bias_row, const float * sink, const float * far,
        const bool shift, const int n_dist, float * maximum, float * denominator, ggml_cuda_mma::tile<16, 8, float> * O) {
#ifdef AMPERE_MMA_AVAILABLE
    using namespace ggml_cuda_mma;
    constexpr int row_i4 = fattn_rel_runs_row_i4;
    tile<16, 8, float> S;
#pragma unroll
    for (int c = 0; c < 8; ++c) {
        tile<16, 8, half2> K_A;
        load_ldmatrix(K_A, (const half2 *) staged_k + 8*c, 4*row_i4);
        mma(S, K_A, Q_B[c]);
    }
    const tile<8, 8, half2> W_B = fattn_rel_runs_weights<use_log2, swa_mask>(S, staged_pos, n, pos_q, pos_q_min,
            use_head, row_bits, staged_rows, bias_row, sink, far, shift, n_dist, maximum, denominator, O);
#pragma unroll
    for (int c = 0; c < 8; ++c) {
        tile<16, 8, half2> V_A;
        load_ldmatrix_trans(V_A, (const half2 *) staged_v + 8*c, 4*row_i4);
        mma(O[c], V_A, W_B);
    }
#else
    GGML_UNUSED_VARS(Q_B, staged_k, staged_v, staged_pos, n, pos_q, pos_q_min, use_head, row_bits, staged_rows, bias_row,
        sink, far, shift, n_dist, maximum, denominator, O);
    NO_DEVICE_CODE;
#endif // AMPERE_MMA_AVAILABLE
}

// The denominators summed over the lanes of a column, and the partial of each column a lane holds stored.
template <bool use_log2>
static __device__ __forceinline__ void fattn_rel_runs_store(
        const float * maximum, float * denominator, const ggml_cuda_mma::tile<16, 8, float> * O, const bool * valid,
        const int64_t * row, float * partial, const int n_slots, const int slot, const bool fp8) {
    constexpr float LN2 = 0.6931471805599453f;
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
            out[0] = (use_log2 && maximum[p] > -1e30f) ? maximum[p]*LN2 : maximum[p];
            out[1] = fattn_rel_partial_word(denominator[p], fp8);
        }
#pragma unroll
        for (int c = 0; c < 8; ++c) {
            out[2 + 16*c + lane/4] = fattn_rel_partial_word(O[c].x[p], fp8);
            out[2 + 16*c + 8 + lane/4] = fattn_rel_partial_word(O[c].x[p + 2], fp8);
        }
    }
}

// Direct single-pass store: fold query sink in registers and write normalized output straight to dst (no partial or merge).
template <bool use_log2>
static __device__ __forceinline__ void fattn_rel_runs_store_direct(
        const float * maximum, float * denominator, const ggml_cuda_mma::tile<16, 8, float> * O, const bool * valid,
        const int64_t * row, const float * sink, const bool shift, float * dst) {
    constexpr float LN2 = 0.6931471805599453f;
    const int lane = threadIdx.x;
#pragma unroll
    for (int p = 0; p < 2; ++p) {
#pragma unroll
        for (int offset = 16; offset >= 4; offset >>= 1) {
            denominator[p] += __shfl_xor_sync(0xffffffff, denominator[p], offset, WARP_SIZE);
        }
    }
#pragma unroll
    for (int p = 0; p < 2; ++p) {
        if (!valid[p]) {
            continue;
        }
        const float max_e = (use_log2 && maximum[p] > -1e30f) ? maximum[p]*LN2 : maximum[p];
        const float sink_e = shift ? 0.0f : sink[p];
        const float m_all = fmaxf(sink_e, max_e);
        const float w_sink = expf(sink_e - m_all);
        const float w_part = max_e > -1e30f ? expf(max_e - m_all) : 0.0f;
        const float inv_d = w_part/(w_sink + w_part*denominator[p]);
        float * out = dst + row[p]*128;
#pragma unroll
        for (int c = 0; c < 8; ++c) {
            out[16*c + lane/4] = O[c].x[p]*inv_d;
            out[16*c + 8 + lane/4] = O[c].x[p + 2]*inv_d;
        }
    }
}

// Split a part's private tiles across warps, then store one partial for the CTA.
template <ggml_type type, bool query_first, bool use_log2, bool swa_mask, bool direct>
__launch_bounds__(fattn_rel_runs_private_warps*WARP_SIZE, 4)
static __global__ void fattn_rel_runs_rows(
        const char * __restrict__ q, const char * __restrict__ k, const char * __restrict__ v, const int4 * __restrict__ runs,
        const float * __restrict__ table, const int32_t * __restrict__ q_pos, const float * __restrict__ sinks, const bool shift,
        float * __restrict__ out, const float scale, const int n_entries, const int table_stride, const int n_dist, const int gqa,
        const int slot0, const int n_slots, const bool after_shared,
        const size_t q_row, const size_t q_head, const size_t k_row, const size_t k_head, const size_t v_row, const size_t v_head,
        const bool fp8, const char * __restrict__ hot_kv, const size_t hot_row, const size_t hot_head, const int32_t * __restrict__ hot_map) {
#if defined(AMPERE_MMA_AVAILABLE) && defined(CP_ASYNC_AVAILABLE)
    using namespace ggml_cuda_mma;
    constexpr float LOG2E = 1.4426950408889634f;
    constexpr float LN2 = 0.6931471805599453f;
    const float q_scale = use_log2 ? scale*LOG2E : scale;
    constexpr int cells = fattn_rel_runs_cells, stages = fattn_rel_runs_private_stages;
    constexpr int warps = fattn_rel_runs_private_warps, row_i4 = fattn_rel_runs_panel_i4;
    const int lane = threadIdx.x, warp = threadIdx.y;
    const int part = query_first ? blockIdx.z : blockIdx.x;
    const int n_parts = query_first ? gridDim.z : gridDim.x;
    const int kv_head = blockIdx.y, query = query_first ? blockIdx.x : blockIdx.z;
    const int head0 = kv_head*gqa, n_heads = gridDim.y*gqa;
    const int64_t row0 = int64_t(query)*n_heads + head0;
    constexpr bool one_byte = type == GGML_TYPE_F8_E4M3 || type == GGML_TYPE_Q8_0;
    constexpr int raw_words = type == GGML_TYPE_Q8_0 ? 17 : 16;
    __shared__ union {
        struct {
            int4 panel[warps][stages][cells*row_i4];
            fattn_rel_runs_panel_cells meta[warps][2];
        } stage;
        struct {
            float o[warps][8][128];
            float m[warps][8];
            float d[warps][8];
        } fold;
        struct {
            alignas(16) int2 raw[one_byte ? warps : 1][2][cells*raw_words];
            int4 panel[warps][cells*row_i4];
            fattn_rel_runs_panel_cells meta[warps][2];
        } packed;
    } scratch;
    k += kv_head*k_head;
    v += kv_head*v_head;
    if (hot_kv) {
        hot_kv += kv_head*hot_head;
    }

    tile<8, 8, half2> Q_B[8];
    {
        const int j = lane/4;
        const float * q_ptr = (const float *) (q + query*q_row + (head0 + (j < gqa ? j : 0))*q_head);
#pragma unroll
        for (int c = 0; c < 8; ++c) {
#pragma unroll
            for (int l = 0; l < 2; ++l) {
                const int d = 16*c + 2*(4*l + lane%4);
                Q_B[c].x[l] = j < gqa ? make_half2(q_ptr[d]*q_scale, q_ptr[d + 1]*q_scale) : make_half2(0.0f, 0.0f);
            }
        }
    }

    const int j0 = 2*(lane%4);
    const int q_pos_row = q_pos[query];
    bool valid[2];
    int pos_q[2];
    int64_t row[2];
    float sink[2], far[2], maximum[2], denominator[2];
    const float * bias_row[2];
#pragma unroll
    for (int p = 0; p < 2; ++p) {
        valid[p] = j0 + p < gqa;
        row[p] = row0 + (valid[p] ? j0 + p : 0);
        sink[p] = sinks[row[p]];
        const float far_raw = shift ? __half2float(__float2half(0.0f - sink[p])) : 0.0f;
        far[p] = use_log2 ? far_raw*LOG2E : far_raw;
        bias_row[p] = table + row[p]*table_stride;
        pos_q[p] = q_pos_row;
        maximum[p] = -FLT_MAX;
        denominator[p] = 0.0f;
    }
    tile<16, 8, float> O[8];

    fattn_rel_runs_cursor cur;
    cur.list = runs + int64_t(query)*n_entries;
    const int4 head = cur.list[0];
    const int first = after_shared ? head.z : 0;
    const int n_tiles = (int64_t(head.y) - first + cells - 1)/cells;
    const int per_part = (n_tiles + n_parts - 1)/n_parts;
    const int part_begin = min(n_tiles, part*per_part);
    const int part_end = min(n_tiles, part_begin + per_part);
    const int per_warp = (part_end - part_begin + warps - 1)/warps;
    const int begin = part_begin + warp*per_warp;
    const int end = min(part_end, begin + per_warp);

    // Copy cold operands as a group; only expand one 32-value panel at a time.
    if (one_byte && !hot_kv && begin < end) {
        auto & raw = scratch.packed.raw[warp];
        auto & panel = scratch.packed.panel[warp];
        auto & meta = scratch.packed.meta[warp];
        const int stop = end == n_tiles ? head.y : first + end*cells;
        const auto copy_operand = [&](int2 * dst, const fattn_rel_runs_panel_cells & m, const char * src, size_t stride) {
            if constexpr (type == GGML_TYPE_Q8_0) {
                // Copy the 128-byte body in two-row stripes; the last word follows separately.
#pragma unroll
                for (int j = 0; j < cells/2; ++j) {
                    const int cell = 2*j + lane/16, seg = lane%16;
                    cp_async_ca_8(ggml_cuda_cvta_generic_to_shared(dst + 17*cell + seg),
                            src + int64_t(m.row[cell])*stride + 8*seg);
                }
                if (lane < cells) {
                    cp_async_ca_8(ggml_cuda_cvta_generic_to_shared(dst + 17*lane + 16),
                            src + int64_t(m.row[lane])*stride + 128);
                }
            } else {
#pragma unroll
                for (int j = 0; j < cells*8/WARP_SIZE; ++j) {
                    const int i = j*WARP_SIZE + lane;
                    cp_async_cg_16<0>(ggml_cuda_cvta_generic_to_shared((int4 *) dst + i),
                            src + int64_t(m.row[i/8])*stride + 16*(i%8));
                }
            }
        };
        if (lane < cells) {
            cur.seek(first + begin*cells, head.x);
        }
        fattn_rel_runs_panel_resolve(cur, first + begin*cells, stop, nullptr, meta[0]);
        copy_operand(raw[0], meta[0], k, k_row);
        cp_async_commit_group();
        int current = 0;
        for (int i = begin; i < end; ++i) {
            cp_async_wait_group<0>();
            __syncwarp();
            copy_operand(raw[1], meta[current], v, v_row);
            cp_async_commit_group();
            tile<16, 8, float> S;
#pragma unroll
            for (int b = 0; b < 4; ++b) {
                fattn_rel_runs_panel_expand<type>(panel, meta[current], b, (const char *) raw[0]);
#pragma unroll
                for (int c = 0; c < 2; ++c) {
                    tile<16, 8, half2> K_A;
                    load_ldmatrix(K_A, (const half2 *) panel + 8*c, 4*row_i4);
                    mma(S, K_A, Q_B[2*b + c]);
                }
                __syncwarp();
            }
            const int at = first + i*cells;
            if (i + 1 < end) {
                fattn_rel_runs_panel_resolve(cur, at + cells, stop, nullptr, meta[current ^ 1]);
                copy_operand(raw[0], meta[current ^ 1], k, k_row);
            }
            cp_async_commit_group();
            const auto W_B = fattn_rel_runs_weights<use_log2, swa_mask>(S, meta[current].pos, min(cells, stop - at),
                    pos_q, q_pos_row, valid, nullptr, nullptr, bias_row, sink, far, shift, n_dist, maximum, denominator, O);
            cp_async_wait_group<1>();
            __syncwarp();
#pragma unroll
            for (int b = 0; b < 4; ++b) {
                fattn_rel_runs_panel_expand<type>(panel, meta[current], b, (const char *) raw[1]);
#pragma unroll
                for (int c = 0; c < 2; ++c) {
                    tile<16, 8, half2> V_A;
                    load_ldmatrix_trans(V_A, (const half2 *) panel + 8*c, 4*row_i4);
                    mma(O[2*b + c], V_A, W_B);
                }
                __syncwarp();
            }
            current ^= 1;
        }
        cp_async_wait_all();
    } else if (begin < end) {
        auto & panel = scratch.stage.panel[warp];
        auto & meta = scratch.stage.meta[warp];
        const int stop = end == n_tiles ? head.y : first + end*cells;
        if (lane < cells) {
            cur.seek(first + begin*cells, head.x);
        }
        fattn_rel_runs_panel_resolve(cur, first + begin*cells, stop, hot_map, meta[0]);

        int stage = 0, current = 0;
        constexpr int ahead = stages - 1;
#pragma unroll
        for (int b = 0; b < ahead; ++b) {
            fattn_rel_runs_panel_copy<type>(panel[b], meta[0], b, false, k, k_row, hot_kv, hot_row);
            cp_async_commit_group();
        }

        for (int i = begin; i < end; ++i) {
            tile<16, 8, float> S;
#pragma unroll
            for (int b = 0; b < 4; ++b) {
                cp_async_wait_group<stages - 2>();
                __syncwarp();
                const int next = (stage + ahead)%stages;
                const bool is_v = b + ahead >= 4;
                fattn_rel_runs_panel_copy<type>(panel[next], meta[current], (b + ahead)%4, is_v,
                        is_v ? v : k, is_v ? v_row : k_row, hot_kv, hot_row);
                cp_async_commit_group();

                fattn_rel_runs_panel_expand<type>(panel[stage], meta[current], b);
#pragma unroll
                for (int c = 0; c < 2; ++c) {
                    tile<16, 8, half2> K_A;
                    load_ldmatrix(K_A, (const half2 *) panel[stage] + 8*c, 4*row_i4);
                    mma(S, K_A, Q_B[2*b + c]);
                }
                stage = (stage + 1)%stages;
            }
            const int at = first + i*cells;
            const auto W_B = fattn_rel_runs_weights<use_log2, swa_mask>(S, meta[current].pos, min(cells, stop - at),
                    pos_q, q_pos_row, valid, nullptr, nullptr, bias_row, sink, far, shift, n_dist, maximum, denominator, O);
            const bool more = i + 1 < end;
            if (more) {
                fattn_rel_runs_panel_resolve(cur, at + cells, stop, hot_map, meta[current ^ 1]);
            }
#pragma unroll
            for (int b = 0; b < 4; ++b) {
                cp_async_wait_group<stages - 2>();
                __syncwarp();
                const int next = (stage + ahead)%stages;
                if (b + ahead < 4) {
                    fattn_rel_runs_panel_copy<type>(panel[next], meta[current], b + ahead, true, v, v_row, hot_kv, hot_row);
                } else if (more) {
                    fattn_rel_runs_panel_copy<type>(panel[next], meta[current ^ 1], b + ahead - 4, false, k, k_row, hot_kv, hot_row);
                }
                cp_async_commit_group();

                fattn_rel_runs_panel_expand<type>(panel[stage], meta[current], b);
#pragma unroll
                for (int c = 0; c < 2; ++c) {
                    tile<16, 8, half2> V_A;
                    load_ldmatrix_trans(V_A, (const half2 *) panel[stage] + 8*c, 4*row_i4);
                    mma(O[2*b + c], V_A, W_B);
                }
                stage = (stage + 1)%stages;
            }
            current ^= 1;
        }
        cp_async_wait_all();
    }
#pragma unroll
    for (int p = 0; p < 2; ++p) {
#pragma unroll
        for (int offset = 16; offset >= 4; offset >>= 1) {
            denominator[p] += __shfl_xor_sync(0xffffffff, denominator[p], offset, WARP_SIZE);
        }
    }

    // The staging storage is dead. Keep the fold in the tile's log base until writing the partial.
    __syncthreads();
#pragma unroll
    for (int p = 0; p < 2; ++p) {
        const int h = j0 + p;
        if (lane < 4) {
            scratch.fold.m[warp][h] = maximum[p];
            scratch.fold.d[warp][h] = denominator[p];
        }
#pragma unroll
        for (int c = 0; c < 8; ++c) {
            scratch.fold.o[warp][h][16*c + lane/4] = O[c].x[p];
            scratch.fold.o[warp][h][16*c + 8 + lane/4] = O[c].x[p + 2];
        }
    }
    __syncthreads();

    if (warp == 0 && lane < gqa) {
        float m = -FLT_MAX;
#pragma unroll
        for (int w = 0; w < warps; ++w) {
            m = fmaxf(m, scratch.fold.m[w][lane]);
        }
        float d = 0.0f;
#pragma unroll
        for (int w = 0; w < warps; ++w) {
            const float dw = scratch.fold.d[w][lane];
            const float md = scratch.fold.m[w][lane] - m;
            const float weight = dw > 0.0f ? (use_log2 ? exp2f(md) : expf(md)) : 0.0f;
            d += weight*dw;
            scratch.fold.m[w][lane] = weight;
        }
        const float max_e = use_log2 && m > -1e30f ? m*LN2 : m;
        if constexpr (direct) {
            const float sink_e = shift ? 0.0f : sinks[row0 + lane];
            const float m_all = fmaxf(max_e, sink_e);
            const float w = d > 0.0f ? expf(max_e - m_all) : 0.0f;
            scratch.fold.d[0][lane] = w/(expf(sink_e - m_all) + w*d);
        } else {
            float * dst = out + ((row0 + lane)*n_slots + slot0 + part)*130;
            dst[0] = max_e;
            dst[1] = fattn_rel_partial_word(d, fp8);
        }
    }
    __syncthreads();

    const int dim = warp*WARP_SIZE + lane;
    for (int h = 0; h < gqa; ++h) {
        float acc = 0.0f;
#pragma unroll
        for (int w = 0; w < warps; ++w) {
            acc += scratch.fold.m[w][h]*scratch.fold.o[w][h][dim];
        }
        if constexpr (direct) {
            out[(row0 + h)*128 + dim] = acc*scratch.fold.d[0][h];
        } else {
            out[((row0 + h)*n_slots + slot0 + part)*130 + 2 + dim] = fattn_rel_partial_word(acc, fp8);
        }
    }
#else
    GGML_UNUSED_VARS(q, k, v, runs, table, q_pos, sinks, shift, out, scale, n_entries, table_stride, n_dist, gqa, slot0, n_slots,
        after_shared, q_row, q_head, k_row, k_head, v_row, v_head, fp8, hot_kv, hot_row, hot_head, hot_map);
    NO_DEVICE_CODE;
#endif // defined(AMPERE_MMA_AVAILABLE) && defined(CP_ASYNC_AVAILABLE)
}

// 4-warp intra-CTA split-K direct SWA kernel: 1 CTA per (row, kv_head), each warp takes 1/4 of the SWA tiles,
// reduces (maximum, denominator, O) across the 4 warps in SMEM, folds sink, and writes normalized output directly to dst.
template <ggml_type type, bool use_log2>
__launch_bounds__(4*WARP_SIZE, 2)
static __global__ void fattn_rel_runs_swa_direct(
        const char * __restrict__ q, const char * __restrict__ k, const char * __restrict__ v, const int4 * __restrict__ runs,
        const float * __restrict__ table, const int32_t * __restrict__ q_pos, const float * __restrict__ sinks, const bool shift,
        float * __restrict__ out, const float scale, const int n_entries, const int table_stride, const int n_dist, const int gqa,
        const size_t q_row, const size_t q_head, const size_t k_row, const size_t k_head, const size_t v_row, const size_t v_head,
        const char * __restrict__ hot_kv, const size_t hot_row, const size_t hot_head, const int32_t * __restrict__ hot_map) {
#if defined(AMPERE_MMA_AVAILABLE) && defined(CP_ASYNC_AVAILABLE)
    using namespace ggml_cuda_mma;
    constexpr float LOG2E = 1.4426950408889634f;
    constexpr float LN2   = 0.6931471805599453f;
    const float q_scale = use_log2 ? scale*LOG2E : scale;
    constexpr int cells = fattn_rel_runs_cells, row_i4 = fattn_rel_runs_row_i4;
    const int lane = threadIdx.x, warp = threadIdx.y;
    const int query = blockIdx.x, kv_head = blockIdx.y;
    const int head0 = kv_head*gqa, n_heads = gridDim.y*gqa;
    const int64_t row0 = int64_t(query)*n_heads + head0;
    __shared__ int4 staged_k[4][cells*row_i4], staged_v[4][cells*row_i4];
    __shared__ int staged_pos[4][cells];
    k += kv_head*k_head;
    v += kv_head*v_head;
    if (hot_kv) {
        hot_kv += kv_head*hot_head;
    }

    tile<8, 8, half2> Q_B[8];
    {
        const int j = lane/4;
        const float * q_ptr = (const float *) (q + query*q_row + (head0 + (j < gqa ? j : 0))*q_head);
#pragma unroll
        for (int c = 0; c < 8; ++c) {
#pragma unroll
            for (int l = 0; l < 2; ++l) {
                const int d = 16*c + 2*(4*l + lane%4);
                Q_B[c].x[l] = j < gqa ? make_half2(q_ptr[d]*q_scale, q_ptr[d + 1]*q_scale) : make_half2(0.0f, 0.0f);
            }
        }
    }

    const int j0 = 2*(lane%4);
    const int q_pos_row = q_pos[query];
    bool valid[2];
    int pos_q[2];
    int64_t row[2];
    float sink[2], far[2], maximum[2], denominator[2];
    const float * bias_row[2];
#pragma unroll
    for (int p = 0; p < 2; ++p) {
        valid[p] = j0 + p < gqa;
        row[p] = row0 + (valid[p] ? j0 + p : 0);
        sink[p] = sinks[row[p]];
        const float far_raw = shift ? __half2float(__float2half(0.0f - sink[p])) : 0.0f;
        far[p] = use_log2 ? far_raw*LOG2E : far_raw;
        bias_row[p] = table + row[p]*table_stride;
        pos_q[p] = q_pos_row;
        maximum[p] = -FLT_MAX;
        denominator[p] = 0.0f;
    }
    tile<16, 8, float> O[8];

    fattn_rel_runs_cursor cur;
    cur.list = runs + int64_t(query)*n_entries;
    const int4 head = cur.list[0];
    const int n_tiles = (head.y + cells - 1)/cells;
    const int per_warp = (n_tiles + 3)/4;
    const int begin = warp*per_warp*cells;
    const int end = min(head.y, begin + per_warp*cells);

    if (begin < end) {
        cur.seek(begin, head.x);
        const int slot = lane%cells, half_row = lane/cells;
        auto load_tile_regs = [&](const int tile_at,
                                  typename fattn_rel_elem<type>::raw & kr, typename fattn_rel_elem<type>::raw & vr,
                                  fattn_rel_elem<GGML_TYPE_F16>::raw & kh, fattn_rel_elem<GGML_TYPE_F16>::raw & vh,
                                  bool & is_hot, int & pos_out) {
            const int at_slot = tile_at + slot;
            if (at_slot < end) {
                cur.move(at_slot);
            }
            int64_t cell = cur.run.x;
            pos_out = INT_MIN;
            if (at_slot < end) {
                const int offset = at_slot - cur.run.z;
                cell += int64_t(offset)*(cur.run.w & 0x7fffffff);
                pos_out = cur.run.y + offset;
            }
            const int32_t hslot = hot_map ? hot_map[cell] : -1;
            is_hot = hslot >= 0;
            if constexpr (type == GGML_TYPE_F16) {
                GGML_UNUSED_VARS(kr, vr);
                const char * ks = is_hot ? hot_kv + (2*int64_t(hslot) + 0)*hot_row : k + cell*k_row;
                const char * vs = is_hot ? hot_kv + (2*int64_t(hslot) + 1)*hot_row : v + cell*v_row;
                kh = fattn_rel_elem<GGML_TYPE_F16>::load(ks, half_row);
                vh = fattn_rel_elem<GGML_TYPE_F16>::load(vs, half_row);
            } else if (is_hot) {
                kh = fattn_rel_elem<GGML_TYPE_F16>::load(hot_kv + (2*int64_t(hslot) + 0)*hot_row, half_row);
                vh = fattn_rel_elem<GGML_TYPE_F16>::load(hot_kv + (2*int64_t(hslot) + 1)*hot_row, half_row);
            } else {
                kr = fattn_rel_elem<type>::load(k + cell*k_row, half_row);
                vr = fattn_rel_elem<type>::load(v + cell*v_row, half_row);
            }
        };
        auto store_tile_smem = [&](const typename fattn_rel_elem<type>::raw & kr, const typename fattn_rel_elem<type>::raw & vr,
                                   const fattn_rel_elem<GGML_TYPE_F16>::raw & kh, const fattn_rel_elem<GGML_TYPE_F16>::raw & vh,
                                   const bool is_hot, const int pos_in) {
            int4 * kd = staged_k[warp] + slot*row_i4 + 8*half_row;
            int4 * vd = staged_v[warp] + slot*row_i4 + 8*half_row;
            if constexpr (type == GGML_TYPE_F16) {
                GGML_UNUSED_VARS(kr, vr, is_hot);
                fattn_rel_elem<GGML_TYPE_F16>::store(kh, half_row, kd);
                fattn_rel_elem<GGML_TYPE_F16>::store(vh, half_row, vd);
            } else if (is_hot) {
                fattn_rel_elem<GGML_TYPE_F16>::store(kh, half_row, kd);
                fattn_rel_elem<GGML_TYPE_F16>::store(vh, half_row, vd);
            } else {
                fattn_rel_elem<type>::store(kr, half_row, kd);
                fattn_rel_elem<type>::store(vr, half_row, vd);
            }
            if (half_row == 0) {
                staged_pos[warp][slot] = pos_in;
            }
        };

        typename fattn_rel_elem<type>::raw kr = {}, vr = {};
        fattn_rel_elem<GGML_TYPE_F16>::raw kh = {}, vh = {};
        bool is_hot = false;
        int pos_cur = INT_MIN;
        load_tile_regs(begin, kr, vr, kh, vh, is_hot, pos_cur);

        for (int at = begin; at < end; at += cells) {
            __syncwarp();
            store_tile_smem(kr, vr, kh, vh, is_hot, pos_cur);
            __syncwarp();
            if (at + cells < end) {
                load_tile_regs(at + cells, kr, vr, kh, vh, is_hot, pos_cur);
            }
            fattn_rel_runs_tile<use_log2, true>(Q_B, staged_k[warp], staged_v[warp], staged_pos[warp], min(cells, end - at),
                    pos_q, q_pos_row, valid, nullptr, nullptr, bias_row, sink, far, shift, n_dist, maximum, denominator, O);
        }
    }

#pragma unroll
    for (int p = 0; p < 2; ++p) {
#pragma unroll
        for (int offset = 16; offset >= 4; offset >>= 1) {
            denominator[p] += __shfl_xor_sync(0xffffffff, denominator[p], offset, WARP_SIZE);
        }
    }

    __syncthreads();
    float * red_o = (float *) staged_k;
    float * red_m = (float *) staged_v;
    float * red_d = red_m + 32;
#pragma unroll
    for (int p = 0; p < 2; ++p) {
        const int h = j0 + p;
        const float max_e = (use_log2 && maximum[p] > -1e30f) ? maximum[p]*LN2 : maximum[p];
        if (lane < 4) {
            red_m[warp*8 + h] = max_e;
            red_d[warp*8 + h] = denominator[p];
        }
        float * dst_o = red_o + (warp*8 + h)*128;
#pragma unroll
        for (int c = 0; c < 8; ++c) {
            dst_o[16*c + lane/4]     = O[c].x[p];
            dst_o[16*c + 8 + lane/4] = O[c].x[p + 2];
        }
    }
    __syncthreads();

    const int tid = warp*WARP_SIZE + lane;
    for (int h = 0; h < gqa; ++h) {
        const int64_t out_row = row0 + h;
        const float sink_e = shift ? 0.0f : sinks[out_row];
        float m_all = sink_e;
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            m_all = fmaxf(m_all, red_m[w*8 + h]);
        }
        float denom_all = expf(sink_e - m_all);
        float sum_all = 0.0f;
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            const float mw = red_m[w*8 + h];
            const float wt = mw > -1e30f ? expf(mw - m_all) : 0.0f;
            denom_all += wt*red_d[w*8 + h];
            sum_all   += wt*red_o[(w*8 + h)*128 + tid];
        }
        out[out_row*128 + tid] = denom_all > 0.0f ? sum_all/denom_all : 0.0f;
    }
#else
    GGML_UNUSED_VARS(q, k, v, runs, table, q_pos, sinks, shift, out, scale, n_entries, table_stride, n_dist, gqa,
        q_row, q_head, k_row, k_head, v_row, v_head, hot_kv, hot_row, hot_head, hot_map);
    NO_DEVICE_CODE;
#endif // defined(AMPERE_MMA_AVAILABLE) && defined(CP_ASYNC_AVAILABLE)
}

// Draft verify (q_group = 2..8): all warps cooperatively stage each private K/V tile once for all q_group draft rows.
template <ggml_type type, bool query_first, bool use_log2, bool swa_mask, bool direct>
__launch_bounds__(6*WARP_SIZE, 2)
static __global__ void fattn_rel_runs_rows_grouped(
        const char * __restrict__ q, const char * __restrict__ k, const char * __restrict__ v, const int4 * __restrict__ runs,
        const float * __restrict__ table, const int32_t * __restrict__ q_pos, const float * __restrict__ sinks, const bool shift,
        float * __restrict__ out, const float scale, const int n_entries, const int table_stride, const int n_dist, const int gqa,
        const int q_group, const int slot0, const int n_slots, const bool after_shared,
        const size_t q_row, const size_t q_head, const size_t k_row, const size_t k_head, const size_t v_row, const size_t v_head,
        const bool fp8, const char * __restrict__ hot_kv, const size_t hot_row, const size_t hot_head, const int32_t * __restrict__ hot_map) {
#if defined(AMPERE_MMA_AVAILABLE) && defined(CP_ASYNC_AVAILABLE)
    using namespace ggml_cuda_mma;
    constexpr float LOG2E = 1.4426950408889634f;
    const float q_scale = use_log2 ? scale*LOG2E : scale;
    constexpr int cells = fattn_rel_runs_cells, stages = fattn_rel_runs_stages, row_i4 = fattn_rel_runs_row_i4;
    const int lane = threadIdx.x, warp = threadIdx.y;
    const int part = query_first ? blockIdx.z : blockIdx.x;
    const int n_parts = query_first ? gridDim.z : gridDim.x;
    const int kv_head = blockIdx.y, seq = query_first ? blockIdx.x : blockIdx.z;
    const int n_heads = gridDim.y*gqa;
    const int query0 = seq*q_group;
    __shared__ int4 staged_k[stages][cells*row_i4], staged_v[stages][cells*row_i4];
    __shared__ int staged_pos[stages][cells];
    __shared__ uint8_t staged_hot[stages][cells];
    k += kv_head*k_head;
    v += kv_head*v_head;
    if (hot_kv) {
        hot_kv += kv_head*hot_head;
    }

    const auto column = [&](const int c, int & head) {
        const int g = (8*warp + c)/gqa;
        head = kv_head*gqa + (8*warp + c)%gqa;
        return g < q_group ? query0 + g : -1;
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
                Q_B[c].x[l] = r >= 0 ? make_half2(q_ptr[d]*q_scale, q_ptr[d + 1]*q_scale) : make_half2(0.0f, 0.0f);
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
        int h;
        const int r = column(j0 + p, h);
        valid[p] = r >= 0;
        row[p] = int64_t(valid[p] ? r : 0)*n_heads + h;
        sink[p] = sinks[row[p]];
        const float far_raw = shift ? __half2float(__float2half(0.0f - sink[p])) : 0.0f;
        far[p] = use_log2 ? far_raw*LOG2E : far_raw;
        bias_row[p] = table + row[p]*table_stride;
        pos_q[p] = q_pos[valid[p] ? r : 0];
        maximum[p] = -FLT_MAX;
        denominator[p] = 0.0f;
    }
    int pos_q_min = min(valid[0] ? pos_q[0] : INT_MAX, valid[1] ? pos_q[1] : INT_MAX);
#pragma unroll
    for (int offset = 16; offset >= 1; offset >>= 1) {
        pos_q_min = min(pos_q_min, __shfl_xor_sync(0xffffffff, pos_q_min, offset, WARP_SIZE));
    }
    tile<16, 8, float> O[8];

    fattn_rel_runs_cursor cur;
    cur.list = runs + int64_t(query0)*n_entries;
    const int4 head = cur.list[0];
    const int first = after_shared ? head.z : 0;
    const int n_tiles = (head.y - first + cells - 1)/cells;
    const int per_part = (n_tiles + n_parts - 1)/n_parts;
    const int begin = first + part*per_part*cells;
    const int end = min(head.y, begin + per_part*cells);

    if (begin < end) {
        constexpr int active_threads = 2*(type == GGML_TYPE_Q8_0 ? 17 : 16)*cells;
        const int tid = warp*WARP_SIZE + lane;
        if (tid < active_threads) {
            cur.seek(begin, head.x);
        }
#pragma unroll
        for (int t = 0; t + 1 < stages; ++t) {
            if (begin + t*cells < end) {
                fattn_rel_runs_stage_coop<type, false>(cur, begin + t*cells, end, k, v, k_row, v_row, staged_k[t], staged_v[t],
                        staged_pos[t], nullptr, staged_hot[t], hot_kv, hot_row, hot_map);
            }
            cp_async_commit_group();
        }
        for (int at = begin, s = 0; at < end; at += cells, s = (s + 1)%stages) {
            cp_async_wait_group<stages - 2>();
            __syncthreads();
            if constexpr (type != GGML_TYPE_F16) {
                fattn_rel_runs_dequant_coop<type>(staged_k[s], staged_v[s], staged_hot[s]);
                __syncthreads();
            }
            if (at + (stages - 1)*cells < end) {
                const int t = (s + stages - 1)%stages;
                fattn_rel_runs_stage_coop<type, false>(cur, at + (stages - 1)*cells, end, k, v, k_row, v_row,
                        staged_k[t], staged_v[t], staged_pos[t], nullptr, staged_hot[t], hot_kv, hot_row, hot_map);
            }
            cp_async_commit_group();
            fattn_rel_runs_tile<use_log2, swa_mask>(Q_B, staged_k[s], staged_v[s], staged_pos[s], min(cells, end - at), pos_q, pos_q_min, valid, nullptr,
                    nullptr, bias_row, sink, far, shift, n_dist, maximum, denominator, O);
        }
    }
    if constexpr (direct) {
        fattn_rel_runs_store_direct<use_log2>(maximum, denominator, O, valid, row, sink, shift, out);
    } else {
        fattn_rel_runs_store<use_log2>(maximum, denominator, O, valid, row, out, n_slots, slot0 + part, fp8);
    }
#else
    GGML_UNUSED_VARS(q, k, v, runs, table, q_pos, sinks, shift, out, scale, n_entries, table_stride, n_dist, gqa, q_group, slot0, n_slots,
        after_shared, q_row, q_head, k_row, k_head, v_row, v_head, fp8, hot_kv, hot_row, hot_head, hot_map);
    NO_DEVICE_CODE;
#endif // defined(AMPERE_MMA_AVAILABLE) && defined(CP_ASYNC_AVAILABLE)
}

// The shared list, a part a block and KV head: all warps cooperatively stage each tile once, and warp w multiplies it by
// columns 8w to 8w + 7 of the block's group of rows (up to 18 warps = 144 columns -> 24 rows at GQA 6).
template <ggml_type type, bool query_first, bool use_log2>
__launch_bounds__(18*WARP_SIZE, 1)
static __global__ void fattn_rel_runs_shared(
        const char * __restrict__ q, const char * __restrict__ k, const char * __restrict__ v, const int4 * __restrict__ runs,
        const float * __restrict__ table, const int32_t * __restrict__ q_pos, const float * __restrict__ shift,
        float * __restrict__ partial, const float scale, const int n_entries, const int table_stride, const int n_dist, const int gqa,
        const int n_rows, const int group, const int n_slots,
        const size_t q_row, const size_t q_head, const size_t k_row, const size_t k_head, const size_t v_row, const size_t v_head,
        const bool fp8, const char * __restrict__ hot_kv, const size_t hot_row, const size_t hot_head, const int32_t * __restrict__ hot_map) {
#if defined(AMPERE_MMA_AVAILABLE) && defined(CP_ASYNC_AVAILABLE)
    using namespace ggml_cuda_mma;
    constexpr float LOG2E = 1.4426950408889634f;
    const float q_scale = use_log2 ? scale*LOG2E : scale;
    constexpr int cells = fattn_rel_runs_cells, stages = fattn_rel_runs_stages, row_i4 = fattn_rel_runs_row_i4;
    const int lane = threadIdx.x, warp = threadIdx.y;
    const int part = query_first ? blockIdx.z : blockIdx.x;
    const int n_parts = query_first ? gridDim.z : gridDim.x;
    const int kv_head = blockIdx.y, row_first = (query_first ? blockIdx.x : blockIdx.z)*group;
    const int n_heads = gridDim.y*gqa;
    __shared__ int4 staged_k[stages][cells*row_i4], staged_v[stages][cells*row_i4];
    __shared__ int staged_pos[stages][cells];
    __shared__ uint32_t staged_rows[stages][cells];
    __shared__ uint8_t staged_hot[stages][cells];
    __shared__ int col_row_idx[144];
    __shared__ int col_pos_q[144];
    __shared__ uint32_t col_row_bits[144];
    __shared__ float col_sink[144];
    __shared__ float col_far[144];
    __shared__ int warp_pos_q_min[18];
    k += kv_head*k_head;
    v += kv_head*v_head;
    if (hot_kv) {
        hot_kv += kv_head*hot_head;
    }

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
                Q_B[c].x[l] = r >= 0 ? make_half2(q_ptr[d]*q_scale, q_ptr[d + 1]*q_scale) : make_half2(0.0f, 0.0f);
            }
        }
    }

    const int j0 = 2*(lane%4);
    float maximum[2] = {-FLT_MAX, -FLT_MAX};
    float denominator[2] = {0.0f, 0.0f};
    {
        int pq[2];
        bool val[2];
#pragma unroll
        for (int p = 0; p < 2; ++p) {
            int h;
            const int r = column(j0 + p, h);
            val[p] = r >= 0;
            const int r_idx = val[p] ? r*n_heads + h : -1;
            const float sk = (shift && val[p]) ? shift[r_idx] : 0.0f;
            const float far_raw = shift ? __half2float(__float2half(0.0f - sk)) : 0.0f;
            pq[p] = q_pos[val[p] ? r : 0];
            if (lane < 4) {
                const int col = 8*warp + j0 + p;
                col_row_idx[col]  = r_idx;
                col_pos_q[col]    = pq[p];
                col_row_bits[col] = val[p] ? 1u << (runs[int64_t(r)*n_entries].w & 31) : 0;
                col_sink[col]     = sk;
                col_far[col]      = use_log2 ? far_raw*LOG2E : far_raw;
            }
        }
        int pq_min = min(val[0] ? pq[0] : INT_MAX, val[1] ? pq[1] : INT_MAX);
#pragma unroll
        for (int offset = 16; offset >= 1; offset >>= 1) {
            pq_min = min(pq_min, __shfl_xor_sync(0xffffffff, pq_min, offset, WARP_SIZE));
        }
        if (lane == 0) {
            warp_pos_q_min[warp] = pq_min;
        }
    }
    __syncwarp();
    tile<16, 8, float> O[8];

    fattn_rel_runs_cursor cur;
    cur.list = runs + int64_t(n_rows)*n_entries;
    const int4 head = cur.list[0];
    const int n_tiles = (head.y + cells - 1)/cells;
    const int per_part = (n_tiles + n_parts - 1)/n_parts;
    const int begin = part*per_part*cells;
    const int end = min(head.y, begin + per_part*cells);

    if (begin < end) {
        constexpr int active_threads = 2*(type == GGML_TYPE_Q8_0 ? 17 : 16)*cells;
        const int tid = warp*WARP_SIZE + lane;
        if (tid < active_threads) {
            cur.seek(begin, head.x);
        }
#pragma unroll
        for (int t = 0; t + 1 < stages; ++t) {
            if (begin + t*cells < end) {
                fattn_rel_runs_stage_coop<type, true>(cur, begin + t*cells, end, k, v, k_row, v_row, staged_k[t], staged_v[t],
                        staged_pos[t], staged_rows[t], staged_hot[t], hot_kv, hot_row, hot_map);
            }
            cp_async_commit_group();
        }
        for (int at = begin, s = 0; at < end; at += cells, s = (s + 1)%stages) {
            cp_async_wait_group<stages - 2>();
            __syncthreads();
            if constexpr (type != GGML_TYPE_F16) {
                fattn_rel_runs_dequant_coop<type>(staged_k[s], staged_v[s], staged_hot[s]);
                __syncthreads();
            }
            if (at + (stages - 1)*cells < end) {
                const int t = (s + stages - 1)%stages;
                fattn_rel_runs_stage_coop<type, true>(cur, at + (stages - 1)*cells, end, k, v, k_row, v_row,
                        staged_k[t], staged_v[t], staged_pos[t], staged_rows[t], staged_hot[t], hot_kv, hot_row, hot_map);
            }
            cp_async_commit_group();
            bool valid[2];
            int pos_q[2];
            uint32_t row_bits[2];
            float sink[2], far[2];
            const float * bias_row[2];
#pragma unroll
            for (int p = 0; p < 2; ++p) {
                const int col = 8*warp + j0 + p;
                const int r_idx = col_row_idx[col];
                valid[p]    = r_idx >= 0;
                pos_q[p]    = col_pos_q[col];
                row_bits[p] = col_row_bits[col];
                sink[p]     = col_sink[col];
                far[p]      = col_far[col];
                bias_row[p] = table + int64_t(valid[p] ? r_idx : 0)*table_stride;
            }
            fattn_rel_runs_tile<use_log2>(Q_B, staged_k[s], staged_v[s], staged_pos[s], min(cells, end - at), pos_q, warp_pos_q_min[warp], valid, row_bits,
                    staged_rows[s], bias_row, sink, far, shift != nullptr, n_dist, maximum, denominator, O);
        }
    }
    bool valid_out[2];
    int64_t row_out[2];
#pragma unroll
    for (int p = 0; p < 2; ++p) {
        const int r_idx = col_row_idx[8*warp + j0 + p];
        valid_out[p] = r_idx >= 0;
        row_out[p]   = valid_out[p] ? r_idx : 0;
    }
    fattn_rel_runs_store<use_log2>(maximum, denominator, O, valid_out, row_out, partial, n_slots, part, fp8);
#else
    GGML_UNUSED_VARS(q, k, v, runs, table, q_pos, shift, partial, scale, n_entries, table_stride, n_dist, gqa, n_rows, group, n_slots,
        q_row, q_head, k_row, k_head, v_row, v_head, fp8, hot_kv, hot_row, hot_head, hot_map);
    NO_DEVICE_CODE;
#endif // defined(AMPERE_MMA_AVAILABLE) && defined(CP_ASYNC_AVAILABLE)
}

static int fattn_rel_runs_occupancy_blocks(const int device, const void * kernel, const int block_threads) {
    static std::mutex mutex;
    static std::map<std::tuple<int, const void *, int>, int> cache;
    std::lock_guard<std::mutex> lock(mutex);
    const auto key = std::make_tuple(device, kernel, block_threads);
    auto it = cache.find(key);
    if (it == cache.end()) {
        int current = 0, n = 0;
        CUDA_CHECK(cudaGetDevice(&current));
        CUDA_CHECK(cudaSetDevice(device));
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, kernel, block_threads, 0));
        CUDA_CHECK(cudaSetDevice(current));
        it = cache.emplace(key, std::max(1, n)).first;
    }
    return it->second;
}
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

// GGML_CUDA_FATTN_REL_RUNS_PARTS: CTAs per row and KV head, capped to SM occupancy unless overridden.
static int fattn_rel_runs_parts(const int device, const ggml_tensor * dst) {
    const int q_group = fattn_rel_decode_group(dst);
    if (ggml_flash_attn_ext_is_swa_runs(dst)) {
        if (q_group == 1) {
            return 1;
        }
        const int64_t n_seqs = dst->src[0]->ne[1]/q_group;
        const int64_t n_units = n_seqs*dst->src[1]->ne[2];
        if (n_units >= 64 || dst->src[1]->ne[1] <= 4096) {
            return 1;
        }
        return (int) std::max<int64_t>(1, std::min<int64_t>(4, dst->src[1]->ne[1]/256));
    }
    static const int env_parts = (int) ggml_env_int("GGML_CUDA_FATTN_REL_RUNS_PARTS");
    static const bool has_env  = fattn_rel_runs_env_set("GGML_CUDA_FATTN_REL_RUNS_PARTS");
    const int64_t cap = dst->src[1]->ne[1] <= 32768 ? 8 : dst->src[1]->ne[1]/1024;
    int64_t limit = std::min<int64_t>(env_parts, std::min<int64_t>(cap, std::max<int64_t>(1, dst->src[1]->ne[1]/256)));
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    if (!has_env) {
        const auto * q = dst->src[0];
        const auto * k = dst->src[1];
        static const bool query_first = ggml_env_switch("GGML_CUDA_FATTN_REL_QUERY_FIRST");
        static const bool log2_legacy = ggml_env_switch("GGML_CUDA_FATTN_REL_LOG2_LEGACY");
        const bool use_log2 = !log2_legacy;
        const int gqa = (int) (q->ne[2]/k->ne[2]);
        const int block_threads = WARP_SIZE*(q_group > 1 ? std::max(2, (q_group*gqa + 7)/8) : fattn_rel_runs_private_warps);
        auto pick_kernel = [&](auto type_tag) -> const void * {
            constexpr ggml_type T = decltype(type_tag)::value;
            if (q_group > 1) {
                return use_log2
                    ? (query_first ? (const void *) fattn_rel_runs_rows_grouped<T, true,  true,  false, false>
                                   : (const void *) fattn_rel_runs_rows_grouped<T, false, true,  false, false>)
                    : (query_first ? (const void *) fattn_rel_runs_rows_grouped<T, true,  false, false, false>
                                   : (const void *) fattn_rel_runs_rows_grouped<T, false, false, false, false>);
            }
            return use_log2
                ? (query_first ? (const void *) fattn_rel_runs_rows<T, true,  true,  false, false>
                               : (const void *) fattn_rel_runs_rows<T, false, true,  false, false>)
                : (query_first ? (const void *) fattn_rel_runs_rows<T, true,  false, false, false>
                               : (const void *) fattn_rel_runs_rows<T, false, false, false, false>);
        };
        const void * kernel = nullptr;
        switch (k->type) {
            case GGML_TYPE_Q4_0:    kernel = pick_kernel(std::integral_constant<ggml_type, GGML_TYPE_Q4_0>{});    break;
            case GGML_TYPE_Q8_0:    kernel = pick_kernel(std::integral_constant<ggml_type, GGML_TYPE_Q8_0>{});    break;
#ifdef FATTN_REL_ELEM_F8
            case GGML_TYPE_F8_E4M3: kernel = pick_kernel(std::integral_constant<ggml_type, GGML_TYPE_F8_E4M3>{}); break;
#endif
            default:                kernel = pick_kernel(std::integral_constant<ggml_type, GGML_TYPE_F16>{});     break;
        }
        const int64_t nsm = ggml_cuda_info().devices[device].nsm;
        const int64_t resident = int64_t(fattn_rel_runs_occupancy_blocks(device, kernel, block_threads))*nsm;
        const int64_t n_units = std::max<int64_t>(1, (q->ne[1]/q_group)*k->ne[2]);
        const int64_t occ_cap = std::max<int64_t>(1, resident/n_units);
        limit = std::min<int64_t>(limit, occ_cap);
    }
#else
    GGML_UNUSED_VARS(device, has_env);
#endif
    return (int) std::max<int64_t>(1, limit);
}

// GGML_CUDA_FATTN_REL_RUNS_SHARED_PARTS: blocks per (KV head, row group) in the shared pass, capped to SM occupancy unless overridden.
static int fattn_rel_runs_shared_parts(const int device, const ggml_tensor * dst) {
    if (ggml_flash_attn_ext_is_swa_runs(dst)) {
        return 0;
    }
    static const bool shared   = ggml_env_switch("GGML_CUDA_FATTN_REL_RUNS_SHARED");
    static const int env_parts = (int) ggml_env_int("GGML_CUDA_FATTN_REL_RUNS_SHARED_PARTS");
    static const bool has_env  = fattn_rel_runs_env_set("GGML_CUDA_FATTN_REL_RUNS_SHARED_PARTS");
    if (!shared) {
        return 0;
    }
    const int64_t cap = dst->src[1]->ne[1] <= 32768 ? 8 : dst->src[1]->ne[1]/1024;
    int64_t limit = std::min<int64_t>(env_parts, std::min<int64_t>(cap, std::max<int64_t>(1, dst->src[1]->ne[1]/256)));
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    if (!has_env) {
        const auto * q = dst->src[0];
        const auto * k = dst->src[1];
        static const bool query_first = ggml_env_switch("GGML_CUDA_FATTN_REL_QUERY_FIRST");
        static const bool log2_legacy = ggml_env_switch("GGML_CUDA_FATTN_REL_LOG2_LEGACY");
        const bool use_log2 = !log2_legacy;
        const int gqa = (int) (q->ne[2]/k->ne[2]);
        const int group = fattn_rel_runs_group(dst);
        const int groups = (int) ((q->ne[1] + group - 1)/group);
        const int block_threads = WARP_SIZE*std::max(2, (group*gqa + 7)/8);
        auto pick_kernel = [&](auto type_tag) -> const void * {
            constexpr ggml_type T = decltype(type_tag)::value;
            return use_log2
                ? (query_first ? (const void *) fattn_rel_runs_shared<T, true,  true>
                               : (const void *) fattn_rel_runs_shared<T, false, true>)
                : (query_first ? (const void *) fattn_rel_runs_shared<T, true,  false>
                               : (const void *) fattn_rel_runs_shared<T, false, false>);
        };
        const void * kernel = nullptr;
        switch (k->type) {
            case GGML_TYPE_Q4_0:    kernel = pick_kernel(std::integral_constant<ggml_type, GGML_TYPE_Q4_0>{});    break;
            case GGML_TYPE_Q8_0:    kernel = pick_kernel(std::integral_constant<ggml_type, GGML_TYPE_Q8_0>{});    break;
#ifdef FATTN_REL_ELEM_F8
            case GGML_TYPE_F8_E4M3: kernel = pick_kernel(std::integral_constant<ggml_type, GGML_TYPE_F8_E4M3>{}); break;
#endif
            default:                kernel = pick_kernel(std::integral_constant<ggml_type, GGML_TYPE_F16>{});     break;
        }
        const int64_t nsm = ggml_cuda_info().devices[device].nsm;
        const int64_t resident = int64_t(fattn_rel_runs_occupancy_blocks(device, kernel, block_threads))*nsm;
        const int64_t n_units = std::max<int64_t>(1, int64_t(groups)*k->ne[2]);
        const int64_t occ_cap = std::max<int64_t>(1, resident/n_units);
        limit = std::min<int64_t>(limit, occ_cap);
    }
#else
    GGML_UNUSED_VARS(device, has_env);
#endif
    return (int) std::max<int64_t>(1, limit);
}

static bool fattn_rel_runs_direct(const int device, const ggml_tensor * dst) {
    return ggml_flash_attn_ext_is_swa_runs(dst) &&
           fattn_rel_runs_parts(device, dst) == 1 &&
           fattn_rel_runs_shared_parts(device, dst) == 0;
}

// Each part stores what a part of the one-cell kernel stores, so one merge reads them all.
static size_t fattn_rel_runs_alloc_size(const int device, const ggml_tensor * dst) {
    if (!fattn_rel_runs_takes(device, dst)) {
        return 0;
    }
    if (fattn_rel_runs_direct(device, dst)) {
        return ggml_nbytes(dst);
    }
    const int n_slots = fattn_rel_runs_parts(device, dst) + fattn_rel_runs_shared_parts(device, dst);
    return GGML_PAD(ggml_nbytes(dst), 128) + size_t(dst->ne[1])*dst->ne[2]*n_slots*130*sizeof(float);
}

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
template <ggml_type type>
static void ggml_cuda_fattn_rel_runs_type(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const auto * q = dst->src[0];
    const auto * k = dst->src[1];
    const auto * v = dst->src[2];
    const auto * runs = ggml_flash_attn_ext_get_kv_runs(dst);
    const int parts = fattn_rel_runs_parts(ctx.device, dst), shared_parts = fattn_rel_runs_shared_parts(ctx.device, dst);
    const int q_group = fattn_rel_decode_group(dst);
    const bool is_swa = ggml_flash_attn_ext_is_swa_runs(dst);
    const bool direct = fattn_rel_runs_direct(ctx.device, dst);
    const int n_slots = shared_parts + parts;
    float * partial = (float *) ((char *) dst->data + GGML_PAD(ggml_nbytes(dst), 128));
    float scale;
    memcpy(&scale, dst->op_params, sizeof(scale));
    const auto * sinks = (const float *) dst->src[9]->data;
    const bool shift = ggml_cuda_fattn_shift_query_sinks();
    // the blocks of one cell range of all rows run side by side, as GGML_CUDA_FATTN_REL_QUERY_FIRST orders the mask kernel's
    static const bool query_first = ggml_env_switch("GGML_CUDA_FATTN_REL_QUERY_FIRST");
    static const bool log2_legacy = ggml_env_switch("GGML_CUDA_FATTN_REL_LOG2_LEGACY");
    const bool use_log2 = !log2_legacy;
    const int gqa = (int) (q->ne[2]/k->ne[2]), n_rows = (int) q->ne[1], n_kv_heads = (int) k->ne[2];
    const int n_entries = (int) runs->ne[1], table_stride = (int) dst->src[6]->ne[0];
    // Padded --reference-sinks tables carry a sentinel column at table->ne[0] - 1 (odd extent 1025 vs 1024).
    const int swa_window = (!dst->src[9] || (table_stride & 1)) ? table_stride - 1 : table_stride;
    const int n_dist = is_swa ? swa_window : (!dst->src[9] ? table_stride - 1 : table_stride);
    const int4 * runs_d = (const int4 *) runs->data;
    const float * table = (const float *) dst->src[6]->data;
    const int32_t * q_pos = (const int32_t *) dst->src[7]->data;
    const float * shift_d = shift ? sinks : nullptr;
    const ggml_tensor * hot_kv = ggml_flash_attn_ext_get_hot_kv(dst);
    const char * hot_kv_d = hot_kv ? (const char *) hot_kv->data : nullptr;
    const size_t hot_row  = hot_kv ? hot_kv->nb[1] : 0;
    const size_t hot_head = hot_kv ? hot_kv->nb[2] : 0;
    const int32_t * hot_map = (hot_kv && dst->src[8] && dst->src[8]->ne[2] == 2)
        ? (const int32_t *) ((const char *) dst->src[8]->data + dst->src[8]->nb[2]) : nullptr;
    const bool fp8 = fattn_rel_fp8_partials(ctx, n_slots);
    if (shared_parts > 0) {
        const int group = fattn_rel_runs_group(dst);
        const int groups = (n_rows + group - 1)/group;
        const dim3 block(WARP_SIZE, std::max(2, (group*gqa + 7)/8));
        const auto kernel = use_log2
            ? (query_first ? fattn_rel_runs_shared<type, true, true>  : fattn_rel_runs_shared<type, false, true>)
            : (query_first ? fattn_rel_runs_shared<type, true, false> : fattn_rel_runs_shared<type, false, false>);
        kernel<<<query_first ? dim3(groups, n_kv_heads, shared_parts) : dim3(shared_parts, n_kv_heads, groups), block, 0, ctx.stream()>>>(
                (const char *) q->data, (const char *) k->data, (const char *) v->data, runs_d, table, q_pos, shift_d, partial,
                scale, n_entries, table_stride, n_dist, gqa, n_rows, group, n_slots, q->nb[1], q->nb[2], k->nb[1], k->nb[2], v->nb[1], v->nb[2], fp8,
                hot_kv_d, hot_row, hot_head, hot_map);
    }
    float * row_out = direct ? (float *) dst->data : partial;
    if (is_swa && direct && q_group == 1) {
        const auto kernel = use_log2 ? fattn_rel_runs_swa_direct<type, true> : fattn_rel_runs_swa_direct<type, false>;
        kernel<<<dim3(n_rows, n_kv_heads), dim3(WARP_SIZE, 4), 0, ctx.stream()>>>(
                (const char *) q->data, (const char *) k->data, (const char *) v->data, runs_d, table, q_pos, sinks, shift, row_out,
                scale, n_entries, table_stride, swa_window, gqa,
                q->nb[1], q->nb[2], k->nb[1], k->nb[2], v->nb[1], v->nb[2],
                hot_kv_d, hot_row, hot_head, hot_map);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    auto launch_rows = [&](auto swa_tag, auto direct_tag) {
        constexpr bool swa_c = decltype(swa_tag)::value;
        constexpr bool dir_c = decltype(direct_tag)::value;
        if (q_group > 1) {
            const int n_seqs = n_rows/q_group;
            const dim3 block(WARP_SIZE, std::max(2, (q_group*gqa + 7)/8));
            const auto kernel = use_log2
                ? (query_first ? fattn_rel_runs_rows_grouped<type, true,  true,  swa_c, dir_c>
                               : fattn_rel_runs_rows_grouped<type, false, true,  swa_c, dir_c>)
                : (query_first ? fattn_rel_runs_rows_grouped<type, true,  false, swa_c, dir_c>
                               : fattn_rel_runs_rows_grouped<type, false, false, swa_c, dir_c>);
            kernel<<<query_first ? dim3(n_seqs, n_kv_heads, parts) : dim3(parts, n_kv_heads, n_seqs), block, 0, ctx.stream()>>>(
                    (const char *) q->data, (const char *) k->data, (const char *) v->data, runs_d, table, q_pos, sinks, shift, row_out,
                    scale, n_entries, table_stride, n_dist, gqa, q_group, shared_parts, n_slots, shared_parts > 0,
                    q->nb[1], q->nb[2], k->nb[1], k->nb[2], v->nb[1], v->nb[2], fp8,
                    hot_kv_d, hot_row, hot_head, hot_map);
        } else {
            const auto kernel = use_log2
                ? (query_first ? fattn_rel_runs_rows<type, true,  true,  swa_c, dir_c>
                               : fattn_rel_runs_rows<type, false, true,  swa_c, dir_c>)
                : (query_first ? fattn_rel_runs_rows<type, true,  false, swa_c, dir_c>
                               : fattn_rel_runs_rows<type, false, false, swa_c, dir_c>);
            const dim3 block(WARP_SIZE, fattn_rel_runs_private_warps);
            kernel<<<query_first ? dim3(n_rows, n_kv_heads, parts) : dim3(parts, n_kv_heads, n_rows), block, 0, ctx.stream()>>>(
                    (const char *) q->data, (const char *) k->data, (const char *) v->data, runs_d, table, q_pos, sinks, shift, row_out,
                    scale, n_entries, table_stride, n_dist, gqa, shared_parts, n_slots, shared_parts > 0,
                    q->nb[1], q->nb[2], k->nb[1], k->nb[2], v->nb[1], v->nb[2], fp8,
                    hot_kv_d, hot_row, hot_head, hot_map);
        }
    };
    if (is_swa && direct) {
        launch_rows(std::true_type{}, std::true_type{});
    } else if (is_swa) {
        launch_rows(std::true_type{}, std::false_type{});
    } else {
        launch_rows(std::false_type{}, std::false_type{});
    }
    if (!direct) {
        fattn_rel_launch_merge(ctx, dst, partial, n_slots, shift ? nullptr : sinks, (float *) dst->data);
    }
    CUDA_CHECK(cudaGetLastError());
}
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

static void ggml_cuda_fattn_rel_runs(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    switch (dst->src[1]->type) {
        case GGML_TYPE_Q4_0:
            ggml_cuda_fattn_rel_runs_type<GGML_TYPE_Q4_0>(ctx, dst);
            break;
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
