#pragma once

#include "common.cuh"
#include "ggml-attn-mask.h"

// The rows a block reads each cell once for: the hint's rows a sequence when the queries divide into them, else 1
static int fattn_rel_decode_group(const ggml_tensor * dst) {
    const int hint = ggml_flash_attn_ext_get_query_group(dst);
    return hint >= 2 && hint <= 4 && dst->src[0]->ne[1] % hint == 0 ? hint : 1;
}

static int fattn_rel_decode_parts(const ggml_tensor * dst) {
    return std::min<int64_t>(64, std::max<int64_t>(1, dst->src[1]->ne[1]/1024));
}

// Each part stores its logit maximum, the softmax denominator and 128 weighted values.
static size_t fattn_rel_decode_alloc_size(const ggml_tensor * dst) {
    return GGML_PAD(ggml_nbytes(dst), 128) + size_t(dst->ne[1])*dst->ne[2]*fattn_rel_decode_parts(dst)*130*sizeof(float);
}

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
// One warp sums visible rows in order. Masked cells must not affect the rounding of another sequence's attention.
template<int dim, typename acc>
static __global__ void fattn_rel_short_rows(
        const char * q, const char * k, const char * v, const half * mask, const float * table,
        const int32_t * q_pos, const int32_t * kv_pos, const float * sinks, float * out,
        float scale, int n_kv, int n_dist, int gqa, bool packed, size_t mask_stride,
        size_t q_row, size_t q_head, size_t k_row, size_t k_head, size_t v_row, size_t v_head) {
    const int lane = threadIdx.x, head = blockIdx.x, query = blockIdx.y;
    const int64_t row = int64_t(query)*gridDim.x + head;
    const float * qr = (const float *) (q + query*q_row + head*q_head);
    k += (head/gqa)*k_head;
    v += (head/gqa)*v_head;
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
            const half * key = (const half *) (k + cell*k_row);
            const half * value = (const half *) (v + cell*v_row);
            float dot = 0.0f;
#pragma unroll
            for (int i = 0; i < dim/32; ++i) {
                dot += query_vals[i]*__half2float(key[32*i + lane]);
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
                vals[i] = vals[i]*old_weight + weight*__half2float(value[32*i + lane]);
            }
            maximum = next;
        }
    }

#pragma unroll
    for (int i = 0; i < dim/32; ++i) {
        out[int64_t(row)*dim + 32*i + lane] = vals[i]/denominator;
    }
}

template <ggml_type type>
static __device__ __forceinline__ float fattn_rel_decode_value(const char * row, int col) {
    if constexpr (type == GGML_TYPE_F16) {
        return __half2float(((const half *) row)[col]);
    } else if constexpr (type == GGML_TYPE_F8_E4M3) {
        return ggml_cuda_f8_e4m3_to_fp32(((const uint8_t *) row)[col]);
    } else if constexpr (type == GGML_TYPE_Q4_0) {
        // a block keeps its scale and 16 bytes qs holding a nibble a value: values 0-15 of the block are the low nibbles
        // of qs, one per byte, values 16-31 its high nibbles, as dequantize_row_q4_0 reads them.
        const block_q4_0 & block = ((const block_q4_0 *) row)[col/QK4_0];
        const int q = col < QK4_0/2 ? (block.qs[col] & 0x0F) : (block.qs[col - QK4_0/2] >> 4);
        return __half2float(block.d)*float(q - 8);
    } else {
        const block_q8_0 & block = ((const block_q8_0 *) row)[col/QK8_0];
        return __half2float(block.d)*float(block.qs[col%QK8_0]);
    }
}

// A block reads each visible KV row once for `group` consecutive query rows, which see nearly the same cells when
// they are one sequence's rows (a token and its drafts): each row still takes only the cells its own mask holds.
template <ggml_type type, int heads, int group>
static __global__ void fattn_rel_decode_rows(
        const char * q, const char * k, const char * v, const uint32_t * mask, const float * table,
        const int32_t * q_pos, const int32_t * kv_pos, const float * shift, float * partial, float scale, int n_kv, int n_dist, int gqa,
        size_t q_row, size_t q_head, size_t k_row, size_t k_head, size_t v_row, size_t v_head) {
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
            const char * key = k + cell*k_row + kv_head*k_head;
            const char * value = v + cell*v_row + kv_head*v_head;
            float key_value[4], val[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                key_value[i] = fattn_rel_decode_value<type>(key, 32*i + lane);
                val[i] = fattn_rel_decode_value<type>(value, 32*i + lane);
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
            out[1] = denom_all;
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            out[2 + 32*i + lane] = sum_all[i];
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
    const int group = fattn_rel_decode_group(dst);
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
    const dim3 blocks(n_parts, q->ne[2]/(group_heads ? 3 : 1), q->ne[1]/group);
    kernel<<<blocks, dim3(32, 4), 0, ctx.stream()>>>(
            (const char *) q->data, (const char *) k->data, (const char *) v->data,
            (const uint32_t *) dst->src[3]->data, (const float *) dst->src[6]->data,
            (const int32_t *) dst->src[7]->data, (const int32_t *) dst->src[8]->data, shift ? sinks : nullptr, partial,
            scale, k->ne[1], dst->src[6]->ne[0], gqa, q->nb[1], q->nb[2], k->nb[1], k->nb[2], v->nb[1], v->nb[2]);
    fattn_rel_decode_merge<<<dim3(q->ne[2], q->ne[1]), 128, 0, ctx.stream()>>>(
            partial, shift ? nullptr : sinks, (float *) dst->data, n_parts);
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
    auto kernel = q->ne[0] == 64 ? fattn_rel_short_rows<64, float> : fattn_rel_short_rows<128, float>;
    if (reference) {
        kernel = q->ne[0] == 64 ? fattn_rel_short_rows<64, double> : fattn_rel_short_rows<128, double>;
    }
    kernel<<<dim3(q->ne[2], q->ne[1]), 32, 0, ctx.stream()>>>(
            (const char *) q->data, (const char *) k->data, (const char *) v->data,
            (const half *) mask->data, (const float *) dst->src[6]->data,
            (const int32_t *) dst->src[7]->data, (const int32_t *) dst->src[8]->data,
            (const float *) dst->src[9]->data, (float *) dst->data, scale, k->ne[1], dst->src[6]->ne[0],
            q->ne[2]/k->ne[2], mask->type == GGML_TYPE_I16, mask->nb[1]/sizeof(half),
            q->nb[1], q->nb[2], k->nb[1], k->nb[2], v->nb[1], v->nb[2]);
    CUDA_CHECK(cudaGetLastError());
#else
    GGML_UNUSED_VARS(ctx, dst);
    GGML_ABORT("relative attention requires CUDA");
#endif
}
