#pragma once

#include "common.cuh"
#include "ggml-attn-mask.h"
#include "fattn-rel-fp8-pair.cuh"

// The partial word of one value: the two-term pair slot under GGML_CUDA_FATTN_FP8_ACCUM on a Blackwell device
// (4 bytes, the f32's own footprint), else the f32 itself. The merge reads the planes with the same gate, so
// producer and consumer always agree.
static __device__ __forceinline__ float fattn_rel_partial_word(float value, bool fp8) {
#if defined(BLACKWELL_MMA_AVAILABLE)
    if (fp8) {
        return __uint_as_float(fattn_fp8_slot_pack(fattn_fp8_encode_two(value)));
    }
#else
    GGML_UNUSED(fp8);
#endif
    return value;
}

constexpr int fattn_rel_rows_parts_max   = 1024;
constexpr int fattn_rel_rows_merge_lanes = 4;

// The rows a block reads each cell once for: the hint's rows a sequence when the queries divide into them, else 1
static int fattn_rel_decode_group(const ggml_tensor * dst) {
    const int hint = ggml_flash_attn_ext_get_query_group(dst);
    const int64_t gqa = dst->src[0]->ne[2]/dst->src[1]->ne[2];
    static const bool runs_legacy = ggml_env_switch("GGML_CUDA_FATTN_REL_RUNS_LEGACY");
    const bool runs = !runs_legacy && ggml_flash_attn_ext_get_kv_runs(dst) != nullptr;
    const int max_group = runs ? std::min<int>(8, int(48/std::max<int64_t>(1, gqa))) : 4;
    return hint >= 2 && hint <= max_group && dst->src[0]->ne[1] % hint == 0 ? hint : 1;
}

static int fattn_rel_decode_parts(const ggml_tensor * dst) {
    return std::min<int64_t>(64, std::max<int64_t>(1, dst->src[1]->ne[1]/1024));
}

// Each part stores its logit maximum, the softmax denominator and 128 weighted values.
static size_t fattn_rel_decode_alloc_size(const ggml_tensor * dst) {
    return GGML_PAD(ggml_nbytes(dst), 128) + size_t(dst->ne[1])*dst->ne[2]*fattn_rel_decode_parts(dst)*130*sizeof(float);
}

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
template <ggml_type type>
static __device__ __forceinline__ float fattn_rel_decode_value(const char * row, int col) {
    if constexpr (type == GGML_TYPE_F16) {
        return __half2float(((const half *) row)[col]);
    } else if constexpr (type == GGML_TYPE_F8_E4M3) {
        return ggml_cuda_f8_e4m3_to_fp32(((const uint8_t *) row)[col]);
    } else if constexpr (type == GGML_TYPE_Q4_0) {
        // a block keeps its scale and 16 bytes qs holding a nibble a value: within one block values 0-15 are the low
        // nibbles of qs, one per byte, values 16-31 its high nibbles, as dequantize_row_q4_0 reads them; the column
        // takes its own block first (col % QK4_0), like the q8_0 arm below.
        const block_q4_0 & block = ((const block_q4_0 *) row)[col/QK4_0];
        const int c = col % QK4_0;
        const int q = c < QK4_0/2 ? (block.qs[c] & 0x0F) : (block.qs[c - QK4_0/2] >> 4);
        return __half2float(block.d)*float(q - 8);
    } else {
        const block_q8_0 & block = ((const block_q8_0 *) row)[col/QK8_0];
        return __half2float(block.d)*float(block.qs[col%QK8_0]);
    }
}

// One warp sums visible rows in order. Masked cells must not affect the rounding of another sequence's attention.
template<int dim, typename acc, ggml_type type = GGML_TYPE_F16>
static __global__ void fattn_rel_short_rows(
        const char * q, const char * k, const char * v, const half * mask, const float * table,
        const int32_t * q_pos, const int32_t * kv_pos, const float * sinks, float * out,
        float scale, int n_kv, int n_dist, int gqa, bool packed, size_t mask_stride,
        size_t q_row, size_t q_head, size_t k_row, size_t k_head, size_t v_row, size_t v_head,
        const char * hot_kv, size_t hot_row, size_t hot_head, const int32_t * hot_map) {
    const int lane = threadIdx.x, head = blockIdx.x, query = blockIdx.y;
    const int64_t row = int64_t(query)*gridDim.x + head;
    const float * qr = (const float *) (q + query*q_row + head*q_head);
    k += (head/gqa)*k_head;
    v += (head/gqa)*v_head;
    if (hot_kv) {
        hot_kv += (head/gqa)*hot_head;
    }
    mask += query*mask_stride;
    table += int64_t(row)*n_dist;
    acc vals[dim/32] = {};
    float query_vals[dim/32];
#pragma unroll
    for (int i = 0; i < dim/32; ++i) {
        query_vals[i] = qr[32*i + lane]*scale;
    }

    // The sink has no value vector and supplies one term to the denominator.
    acc maximum = sinks[row], denominator = 1;
    for (int base = 0; base < n_kv; base += 32) {
        if (packed && n_kv > 256 && base%1024 == 0) {
            const int word = base/32 + lane;
            const uint16_t * bits = (const uint16_t *) mask;
            const bool present = word < n_kv/32 && (bits[2*word] || bits[2*word + 1]);
            if (!__any_sync(0xffffffff, present)) {
                base += 1024 - 32;
                continue;
            }
        }
        const float mask_value = fattn_mask_value(mask, packed, 0, base + lane);
        uint32_t live = __ballot_sync(0xffffffff, mask_value != -INFINITY);
        while (live) {
            const int bit = __ffs(live) - 1;
            live &= live - 1;
            const int cell = base + bit;
            const int32_t hslot = hot_map ? hot_map[cell] : -1;
            const char * key   = hslot >= 0 ? hot_kv + (2*int64_t(hslot) + 0)*hot_row : k + cell*k_row;
            const char * value = hslot >= 0 ? hot_kv + (2*int64_t(hslot) + 1)*hot_row : v + cell*v_row;
            float dot = 0.0f;
#pragma unroll
            for (int i = 0; i < dim/32; ++i) {
                const float kv = hslot >= 0 ? fattn_rel_decode_value<GGML_TYPE_F16>(key, 32*i + lane)
                                            : fattn_rel_decode_value<type>(key, 32*i + lane);
                dot += query_vals[i]*kv;
            }
            const int64_t dist = int64_t(q_pos[query]) - kv_pos[cell];
            const float bias = dist >= 0 && dist < n_dist ? table[dist] : 0.0f;
            dot = warp_reduce_sum<32>(dot) + bias + __shfl_sync(0xffffffff, mask_value, bit);
            const acc next = max(maximum, acc(dot));
            const acc old_weight = exp(maximum - next);
            const acc weight = exp(acc(dot) - next);
            denominator = denominator*old_weight + weight;
#pragma unroll
            for (int i = 0; i < dim/32; ++i) {
                const float vv = hslot >= 0 ? fattn_rel_decode_value<GGML_TYPE_F16>(value, 32*i + lane)
                                            : fattn_rel_decode_value<type>(value, 32*i + lane);
                vals[i] = vals[i]*old_weight + weight*vv;
            }
            maximum = next;
        }
    }

#pragma unroll
    for (int i = 0; i < dim/32; ++i) {
        out[int64_t(row)*dim + 32*i + lane] = vals[i]/denominator;
    }
}

// A block reads each visible KV row once for `group` consecutive query rows, which see nearly the same cells when
// they are one sequence's rows (a token and its drafts): each row still takes only the cells its own mask holds.
template <ggml_type type, int heads, int group>
static __global__ void fattn_rel_decode_rows(
        const char * q, const char * k, const char * v, const uint32_t * mask, const float * table,
        const int32_t * q_pos, const int32_t * kv_pos, const float * shift, float * partial, float scale, int n_kv, int n_dist, int gqa,
        size_t q_row, size_t q_head, size_t k_row, size_t k_head, size_t v_row, size_t v_head,
        const bool fp8, const char * hot_kv, size_t hot_row, size_t hot_head, const int32_t * hot_map) {
    constexpr int warps = 4;
    const int lane = threadIdx.x, warp = threadIdx.y;
    const int head0 = blockIdx.y*heads, query0 = blockIdx.z*group, kv_head = head0/gqa;
    const int n_heads = gridDim.y*heads;
    float query_value[group][heads][4], sum[group][heads][4] = {};
    float maximum[group][heads], denominator[group][heads] = {}, sink_shift[group][heads];
    int64_t q_position[group];

#pragma unroll
    for (int g = 0; g < group; ++g) {
        const int64_t row0 = int64_t(query0 + g)*n_heads + head0;
        q_position[g] = q_pos[query0 + g];
#pragma unroll
        for (int h = 0; h < heads; ++h) {
            const float * q_ptr = (const float *) (q + (query0 + g)*q_row + (head0 + h)*q_head);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                query_value[g][h][i] = q_ptr[32*i + lane]*scale;
            }
            maximum[g][h] = -FLT_MAX;
            sink_shift[g][h] = shift ? shift[row0 + h] : 0.0f;
        }
    }
    const float * tables = table + (int64_t(query0)*n_heads + head0)*n_dist;
    const int64_t table_query = int64_t(n_heads)*n_dist;
    mask += int64_t(query0)*(n_kv/32);

    // A warp shares each visible KV row across query heads in the same GQA group, and across the group's queries.
    for (int word = blockIdx.x*warps + warp; word < n_kv/32; word += gridDim.x*warps) {
        uint32_t own[group], live = 0;
#pragma unroll
        for (int g = 0; g < group; ++g) {
            own[g] = mask[g*(n_kv/32) + word];
            live |= own[g];
        }
        while (live) {
            const int bit = __ffs(live) - 1;
            const int cell = 32*word + bit;
            live &= live - 1;
            const int32_t hslot = hot_map ? hot_map[cell] : -1;
            const char * key   = hslot >= 0 ? hot_kv + (2*int64_t(hslot) + 0)*hot_row + kv_head*hot_head : k + cell*k_row + kv_head*k_head;
            const char * value = hslot >= 0 ? hot_kv + (2*int64_t(hslot) + 1)*hot_row + kv_head*hot_head : v + cell*v_row + kv_head*v_head;
            float key_value[4], val[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                key_value[i] = hslot >= 0 ? fattn_rel_decode_value<GGML_TYPE_F16>(key,   32*i + lane)
                                          : fattn_rel_decode_value<type>(key,   32*i + lane);
                val[i]       = hslot >= 0 ? fattn_rel_decode_value<GGML_TYPE_F16>(value, 32*i + lane)
                                          : fattn_rel_decode_value<type>(value, 32*i + lane);
            }
            const int64_t cell_position = kv_pos[cell];
#pragma unroll
            for (int g = 0; g < group; ++g) {
                if (!(own[g] >> bit & 1)) {
                    continue;
                }
                const int64_t distance = q_position[g] - cell_position;
#pragma unroll
                for (int h = 0; h < heads; ++h) {
                    float dot = 0.0f;
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        dot += query_value[g][h][i]*key_value[i];
                    }
                    dot = warp_reduce_sum<32>(dot);
                    const float bias = distance >= 0 && distance < n_dist ? tables[g*table_query + h*n_dist + distance] : 0.0f;
                    dot += shift ? __half2float(__float2half(bias - sink_shift[g][h])) : bias;
                    const float next = fmaxf(maximum[g][h], dot);
                    const float old_weight = expf(maximum[g][h] - next);
                    const float weight = expf(dot - next);
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        sum[g][h][i] = sum[g][h][i]*old_weight + weight*val[i];
                    }
                    denominator[g][h] = denominator[g][h]*old_weight + weight;
                    maximum[g][h] = next;
                }
            }
        }
    }

    __shared__ float maxima[group*heads][warps], denominators[group*heads][warps], sums[group*heads][warps][128];
#pragma unroll
    for (int g = 0; g < group; ++g) {
#pragma unroll
        for (int h = 0; h < heads; ++h) {
            if (lane == 0) {
                maxima[g*heads + h][warp] = maximum[g][h];
                denominators[g*heads + h][warp] = denominator[g][h];
            }
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                sums[g*heads + h][warp][32*i + lane] = sum[g][h][i];
            }
        }
    }
    __syncthreads();

    // one warp merges each of the group's query heads
    for (int gh = warp; gh < group*heads; gh += warps) {
        const int g = gh/heads, h = gh % heads;
        const int64_t row = int64_t(query0 + g)*n_heads + head0 + h;
        float * out = partial + (row*gridDim.x + blockIdx.x)*130;
        float max_all = -FLT_MAX, denom_all = 0.0f, sum_all[4] = {};
#pragma unroll
        for (int w = 0; w < warps; ++w) {
            max_all = fmaxf(max_all, maxima[gh][w]);
        }
#pragma unroll
        for (int w = 0; w < warps; ++w) {
            const float weight = expf(maxima[gh][w] - max_all);
            denom_all += weight*denominators[gh][w];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                sum_all[i] += weight*sums[gh][w][32*i + lane];
            }
        }
        if (lane == 0) {
            out[0] = max_all;
            out[1] = fattn_rel_partial_word(denom_all, fp8);
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            out[2 + 32*i + lane] = fattn_rel_partial_word(sum_all[i], fp8);
        }
    }
}

static __global__ void fattn_rel_decode_merge(const float * partial, const float * sinks, float * out, int n_parts) {
    const int row = blockIdx.y*gridDim.x + blockIdx.x;
    partial += int64_t(row)*n_parts*130;
    float maximum = sinks ? sinks[row] : 0.0f, denominator = 1.0f, sum = 0.0f;
    for (int i = 0; i < n_parts; ++i) {
        const float * part = partial + i*130;
        const float next = fmaxf(maximum, part[0]);
        const float old_weight = expf(maximum - next);
        const float weight = expf(part[0] - next);
        denominator = denominator*old_weight + weight*part[1];
        sum = sum*old_weight + weight*part[2 + threadIdx.x];
        maximum = next;
    }
    out[int64_t(row)*128 + threadIdx.x] = sum/denominator;
}

// Parallel reduction across up to 1024 parts using 4 lanes x 128 threads per (row, head).
static __global__ void __launch_bounds__(fattn_rel_rows_merge_lanes*128) fattn_rel_rows_merge(
        const float * __restrict__ partial, const float * __restrict__ sinks, float * __restrict__ out, const int n_parts) {
    constexpr int lanes = fattn_rel_rows_merge_lanes, nthreads = lanes*128, nwarps = nthreads/WARP_SIZE;
    const int d = threadIdx.x, lane = threadIdx.y, tid = lane*128 + d;
    const int row = blockIdx.y*gridDim.x + blockIdx.x;
    partial += int64_t(row)*n_parts*130;
    __shared__ float weight[fattn_rel_rows_parts_max];
    __shared__ float reduce[nwarps];
    __shared__ float sums[lanes][128];

    const float sink = sinks ? sinks[row] : 0.0f;
    float maximum = sink;
    for (int i = tid; i < n_parts; i += nthreads) {
        maximum = fmaxf(maximum, partial[i*130]);
    }
    maximum = warp_reduce_max(maximum);
    if (tid % WARP_SIZE == 0) {
        reduce[tid/WARP_SIZE] = maximum;
    }
    __syncthreads();
    maximum = reduce[0];
#pragma unroll
    for (int w = 1; w < nwarps; ++w) {
        maximum = fmaxf(maximum, reduce[w]);
    }
    __syncthreads();

    float denominator = 0.0f;
    for (int i = tid; i < n_parts; i += nthreads) {
        const float w = expf(partial[i*130] - maximum);
        weight[i] = w;
        denominator += w*partial[i*130 + 1];
    }
    denominator = warp_reduce_sum(denominator);
    if (tid % WARP_SIZE == 0) {
        reduce[tid/WARP_SIZE] = denominator;
    }
    __syncthreads();

    float sum = 0.0f;
#pragma unroll 4
    for (int i = lane; i < n_parts; i += lanes) {
        sum += weight[i]*partial[i*130 + 2 + d];
    }
    sums[lane][d] = sum;
    __syncthreads();
    if (lane == 0) {
        denominator = expf(sink - maximum);
#pragma unroll
        for (int w = 0; w < nwarps; ++w) {
            denominator += reduce[w];
        }
        sum = sums[0][d];
#pragma unroll
        for (int l = 1; l < lanes; ++l) {
            sum += sums[l][d];
        }
        out[int64_t(row)*128 + d] = sum/denominator;
    }
}

// GGML_CUDA_FATTN_FP8_ACCUM: the partial planes hold two-term pair slots - word 0 the logit maximum as f32,
// word 1 the denominator's slot, words 2.. the 128 value slots (fattn_fp8_slot_pack; same stride) - and this
// merge folds them on the block-scaled MMA. Each stream's softmax weight is applied as one product before the
// fold (a product is not an accumulation), and the numerator and denominator fold under the same weights.
// One warp folds one element at a time; four warps cover the row's 128. The sink keeps the f32 merge's
// semantics: it seeds the maximum and contributes exp(sink - max) to the denominator, the 1.0 the f32 merge
// starts from.
static __global__ void fattn_rel_decode_merge_fp8(const uint32_t * partial, const float * sinks, float * out, int n_parts) {
#if defined(BLACKWELL_MMA_AVAILABLE)
    static_assert(FATTN_FP8_MAX_PARTS == FATTN_FP8_GROUPS_MAX * FATTN_FP8_GROUP, "fp8 merge max parts mismatch");
    __shared__ fattn_fp8_fold_stage stage[4];
    __shared__ float max_all_sh, denom_sh;
    __shared__ float weight_sh[FATTN_FP8_GROUPS_MAX * FATTN_FP8_GROUP];
    const int row = blockIdx.y*gridDim.x + blockIdx.x;
    const uint32_t * base = partial + int64_t(row)*n_parts*130;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const float seed = sinks ? sinks[row] : 0.0f;
    if (threadIdx.x == 0) {
        float m = seed;
        for (int i = 0; i < n_parts; ++i) {
            m = fmaxf(m, __uint_as_float(base[i*130]));
        }
        max_all_sh = m;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < n_parts; i += blockDim.x) {
        weight_sh[i] = expf(__uint_as_float(base[i*130]) - max_all_sh);
    }
    __syncthreads();
    const uint32_t zero = fattn_fp8_slot_pack(fattn_fp8_zero_two());
    if (warp == 0) {
        const fattn_fp8_two d = fattn_fp8_fold_groups(n_parts, lane, stage[0], [&](int g) {
            const int i = g*FATTN_FP8_GROUP + lane;
            return i < n_parts ? fattn_fp8_scale_slot(base[i*130 + 1], weight_sh[i]) : zero;
        });
        if (lane == 0) {
            denom_sh = expf(seed - max_all_sh) + fattn_fp8_decode(d.hi) + fattn_fp8_decode(d.lo);
        }
    }
    __syncthreads();
    for (int e = 0; e < 32; ++e) {
        const int element = warp*32 + e;
        const fattn_fp8_two o = fattn_fp8_fold_groups(n_parts, lane, stage[warp], [&](int g) {
            const int i = g*FATTN_FP8_GROUP + lane;
            return i < n_parts ? fattn_fp8_scale_slot(base[i*130 + 2 + element], weight_sh[i]) : zero;
        });
        if (lane == 0) {
            out[int64_t(row)*128 + element] = (fattn_fp8_decode(o.hi) + fattn_fp8_decode(o.lo)) / denom_sh;
        }
    }
#else
    GGML_UNUSED_VARS(partial, sinks, out, n_parts);
    NO_DEVICE_CODE;
#endif
}

// The gate the writer kernels and the merge share: GGML_CUDA_FATTN_FP8_ACCUM on a Blackwell device compiles
// the pair path when n_parts fits weight_sh (256); otherwise the planes stay f32.
static bool fattn_rel_fp8_partials(const ggml_backend_cuda_context & ctx, int n_parts) {
    return n_parts <= FATTN_FP8_MAX_PARTS &&
           fattn_fp8_accum() && blackwell_mma_available(ggml_cuda_info().devices[ctx.device].cc);
}

// The merge the gate asks for, at the grid every rel decode launcher already used.
static void fattn_rel_launch_merge(ggml_backend_cuda_context & ctx, const ggml_tensor * dst, float * partial,
                                   int n_parts, const float * sinks, float * out) {
    const dim3 grid(dst->src[0]->ne[2], dst->src[0]->ne[1]);
    const bool fp8 = fattn_rel_fp8_partials(ctx, n_parts);
    if (fp8) {
        fattn_rel_decode_merge_fp8<<<grid, 128, 0, ctx.stream()>>>((const uint32_t *) partial, sinks, out, n_parts);
    } else if (n_parts >= 8 && n_parts <= fattn_rel_rows_parts_max) {
        fattn_rel_rows_merge<<<grid, dim3(128, fattn_rel_rows_merge_lanes), 0, ctx.stream()>>>(partial, sinks, out, n_parts);
    } else {
        fattn_rel_decode_merge<<<grid, 128, 0, ctx.stream()>>>(partial, sinks, out, n_parts);
    }
}
#endif

static void ggml_cuda_fattn_rel_decode(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    const auto * q = dst->src[0];
    const auto * k = dst->src[1];
    const auto * v = dst->src[2];
    const int n_parts = fattn_rel_decode_parts(dst);
    float * partial = (float *) ((char *) dst->data + GGML_PAD(ggml_nbytes(dst), 128));
    float scale;
    memcpy(&scale, dst->op_params, sizeof(scale));
    const int gqa = q->ne[2]/k->ne[2];
    const bool group_heads = gqa == 6 && k->ne[1] >= 65536;
    const auto * sinks = (const float *) dst->src[9]->data;
    const bool shift = ggml_cuda_fattn_shift_query_sinks();
    const int group_raw = fattn_rel_decode_group(dst);
    const int group = group_raw <= 4 ? group_raw : 1;
    using kernel_t = decltype(&fattn_rel_decode_rows<GGML_TYPE_F16, 1, 1>);
    const auto pick_group = [&](auto type, auto heads) -> kernel_t {
        constexpr ggml_type T = decltype(type)::value;
        constexpr int H = decltype(heads)::value;
        switch (group) {
            case 2:  return fattn_rel_decode_rows<T, H, 2>;
            case 3:  return fattn_rel_decode_rows<T, H, 3>;
            case 4:  return fattn_rel_decode_rows<T, H, 4>;
            default: return fattn_rel_decode_rows<T, H, 1>;
        }
    };
    const auto pick = [&](auto heads) -> kernel_t {
        switch (k->type) {
            case GGML_TYPE_Q4_0:    return pick_group(std::integral_constant<ggml_type, GGML_TYPE_Q4_0>{}, heads);
            case GGML_TYPE_Q8_0:    return pick_group(std::integral_constant<ggml_type, GGML_TYPE_Q8_0>{}, heads);
            case GGML_TYPE_F8_E4M3: return pick_group(std::integral_constant<ggml_type, GGML_TYPE_F8_E4M3>{}, heads);
            default:                return pick_group(std::integral_constant<ggml_type, GGML_TYPE_F16>{}, heads);
        }
    };
    const kernel_t kernel = group_heads ? pick(std::integral_constant<int, 3>{}) : pick(std::integral_constant<int, 1>{});
    const bool fp8 = fattn_rel_fp8_partials(ctx, n_parts);
    const ggml_tensor * hot_kv = ggml_flash_attn_ext_get_hot_kv(dst);
    const int32_t * hot_map = (hot_kv && dst->src[8]->ne[2] == 2)
        ? (const int32_t *) ((const char *) dst->src[8]->data + dst->src[8]->nb[2]) : nullptr;
    const dim3 blocks(n_parts, q->ne[2]/(group_heads ? 3 : 1), q->ne[1]/group);
    kernel<<<blocks, dim3(32, 4), 0, ctx.stream()>>>(
            (const char *) q->data, (const char *) k->data, (const char *) v->data,
            (const uint32_t *) dst->src[3]->data, (const float *) dst->src[6]->data,
            (const int32_t *) dst->src[7]->data, (const int32_t *) dst->src[8]->data, shift ? sinks : nullptr, partial,
            scale, k->ne[1], dst->src[6]->ne[0], gqa, q->nb[1], q->nb[2], k->nb[1], k->nb[2], v->nb[1], v->nb[2], fp8,
            hot_kv ? (const char *) hot_kv->data : nullptr, hot_kv ? hot_kv->nb[1] : 0, hot_kv ? hot_kv->nb[2] : 0, hot_map);
    fattn_rel_launch_merge(ctx, dst, partial, n_parts, shift ? nullptr : sinks, (float *) dst->data);
    CUDA_CHECK(cudaGetLastError());
#else
    GGML_UNUSED_VARS(ctx, dst);
    GGML_ABORT("relative mask decode requires CUDA");
#endif
}

static void ggml_cuda_fattn_rel_short(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    const auto * q = dst->src[0];
    const auto * k = dst->src[1];
    const auto * v = dst->src[2];
    const auto * mask = dst->src[3];
    float scale;
    memcpy(&scale, dst->op_params, sizeof(scale));
    static const bool reference = ggml_env_switch("GGML_CUDA_FATTN_REL_ORDERED_REF");
    using short_kernel_t = decltype(&fattn_rel_short_rows<128, float, GGML_TYPE_F16>);
    auto pick_dim = [&](auto type_tag) -> short_kernel_t {
        constexpr ggml_type T = decltype(type_tag)::value;
        if (reference) {
            return q->ne[0] == 64 ? fattn_rel_short_rows<64, double, T> : fattn_rel_short_rows<128, double, T>;
        }
        return q->ne[0] == 64 ? fattn_rel_short_rows<64, float, T> : fattn_rel_short_rows<128, float, T>;
    };
    short_kernel_t kernel;
    switch (k->type) {
        case GGML_TYPE_Q4_0:    kernel = pick_dim(std::integral_constant<ggml_type, GGML_TYPE_Q4_0>{});    break;
        case GGML_TYPE_Q8_0:    kernel = pick_dim(std::integral_constant<ggml_type, GGML_TYPE_Q8_0>{});    break;
        case GGML_TYPE_F8_E4M3: kernel = pick_dim(std::integral_constant<ggml_type, GGML_TYPE_F8_E4M3>{}); break;
        default:                kernel = pick_dim(std::integral_constant<ggml_type, GGML_TYPE_F16>{});     break;
    }
    const ggml_tensor * hot_kv = ggml_flash_attn_ext_get_hot_kv(dst);
    const int32_t * hot_map = (hot_kv && dst->src[8]->ne[2] == 2)
        ? (const int32_t *) ((const char *) dst->src[8]->data + dst->src[8]->nb[2]) : nullptr;
    kernel<<<dim3(q->ne[2], q->ne[1]), 32, 0, ctx.stream()>>>(
            (const char *) q->data, (const char *) k->data, (const char *) v->data,
            (const half *) mask->data, (const float *) dst->src[6]->data,
            (const int32_t *) dst->src[7]->data, (const int32_t *) dst->src[8]->data,
            (const float *) dst->src[9]->data, (float *) dst->data, scale, k->ne[1], dst->src[6]->ne[0],
            q->ne[2]/k->ne[2], mask->type == GGML_TYPE_I16, mask->nb[1]/sizeof(half),
            q->nb[1], q->nb[2], k->nb[1], k->nb[2], v->nb[1], v->nb[2],
            hot_kv ? (const char *) hot_kv->data : nullptr, hot_kv ? hot_kv->nb[1] : 0, hot_kv ? hot_kv->nb[2] : 0, hot_map);
    CUDA_CHECK(cudaGetLastError());
#else
    GGML_UNUSED_VARS(ctx, dst);
    GGML_ABORT("relative attention requires CUDA");
#endif
}
