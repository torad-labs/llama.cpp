#pragma once

#include "common.cuh"
#include "fattn-common.cuh"
#include "fattn-rel-decode.cuh"
#include "fattn-rel-mma.cuh"
#include "ggml-attn-mask.h"

#include <map>
#include <mutex>

// Inkling 2 Mini's decode: 1 to 8 rows of a unified pool (a step's sessions, a row each) on a q4_0 or q8_0 cache of head
// size 128, under a relative position bias with a sink a query and a bit-packed mask. The dense kernels read the whole
// pool once a query head (a tile is one Q head, so a KV head's cells three times at GQA 3, for every row's cells), and at
// a short context launch a few blocks. Here the tile kernel fattn_rel_mma_rows reads each row's own cells: a warp a
// (stream, KV head, row) gathers the cells its row's mask words show in its stream and multiplies each tile by the KV
// head's query heads at once, and the streams are as many as the card holds warps of the kernel, so a step reads only
// what its rows see, once a KV head and row, and fills the card at any context. The streams' parts are merged in parallel.
// A step whose rows share a prefix reads the prefix once a row; one query group for all rows (a column a row's head)
// reads it once, but costs twice the tile work where the rows share nothing (the four sessions apart, every sliding layer).
// GGML_CUDA_FATTN_REL_ROWS_LEGACY=1: the dense kernels, or the bounded and low-mean ones the model's hints pick, as before.

constexpr int fattn_rel_rows_max       = 8;    // rows a step
constexpr int fattn_rel_rows_parts_max = 1024; // streams, which the merge holds a weight each of

// The shapes the path takes. GQA 6 on q8_0 is Inkling 2 Small's, which keeps its own classes and kernels. f16 caches
// keep theirs too: this path exists to read a quantized pool's cells.
static bool fattn_rel_rows_takes(const int device, const ggml_tensor * dst) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    GGML_UNUSED_VARS(device, dst);
    return false;
#else
    static const bool legacy = ggml_env_switch("GGML_CUDA_FATTN_REL_ROWS_LEGACY");
    const auto * q = dst->src[0];
    const auto * k = dst->src[1];
    const auto * v = dst->src[2];
    const auto * mask = dst->src[3];
    const auto * table = dst->src[6];
    if (legacy || dst->op != GGML_OP_FLASH_ATTN_EXT || !mask || !table || !dst->src[9] || dst->src[5]) {
        return false;
    }
    const int cc = ggml_cuda_info().devices[device].cc;
    if (!GGML_CUDA_CC_IS_NVIDIA(cc) || !ampere_mma_available(cc)) {
        return false;
    }
    if (k->type != v->type || (k->type != GGML_TYPE_Q4_0 && k->type != GGML_TYPE_Q8_0)) {
        return false;
    }
    if (q->ne[0] != 128 || k->ne[0] != 128 || v->ne[0] != 128 || q->type != GGML_TYPE_F32 || q->nb[0] != sizeof(float) ||
            k->nb[0] != ggml_type_size(k->type) || v->nb[0] != k->nb[0] || q->ne[3] != 1 || k->ne[3] != 1 || v->ne[3] != 1 ||
            v->ne[1] != k->ne[1] || v->ne[2] != k->ne[2] || q->ne[2] % k->ne[2] != 0) {
        return false;
    }
    const int64_t gqa = q->ne[2]/k->ne[2];
    if (gqa < 2 || gqa > 8 || (k->type == GGML_TYPE_Q8_0 && gqa == 6) || q->ne[1] < 1 || q->ne[1] > fattn_rel_rows_max) {
        return false;
    }
    if (mask->type != GGML_TYPE_I16 || mask->ne[0]*16 != k->ne[1] || mask->ne[1] != q->ne[1] || mask->ne[2] != 1 ||
            mask->ne[3] != 1 || !ggml_is_contiguous(mask) || k->ne[1] % 32 != 0 || k->ne[1] > INT32_MAX) {
        return false;
    }
    // the bias table a row and head, the sinks and the positions as the kernel indexes them
    if (!ggml_is_contiguous(table) || table->ne[1] != q->ne[2] || table->ne[2] != q->ne[1] || table->ne[3] != 1 ||
            !ggml_is_contiguous(dst->src[9]) || ggml_nelements(dst->src[9]) != q->ne[1]*q->ne[2] ||
            dst->src[7]->ne[0] != q->ne[1] || dst->src[8]->ne[0] != k->ne[1]) {
        return false;
    }
    float params[3];
    memcpy(params, dst->op_params, sizeof(params));
    if (params[1] != 0.0f || params[2] != 0.0f) {
        return false;
    }
    // q4_0 rows (72 bytes) are staged through fattn_rel_elem 8 bytes at a time, q8_0 rows (136 bytes) by the kernel's
    // own stage 4 bytes at a time: strides and view offsets alone, since the alloc size asks before K has an address (a
    // buffer's base is aligned past them)
    const size_t align = k->type == GGML_TYPE_Q8_0 ? 4 : 8;
    for (const ggml_tensor * t : {k, v}) {
        if ((t->view_src ? t->view_offs : 0) % align != 0 || t->nb[1] % align != 0 || t->nb[2] % align != 0) {
            return false;
        }
    }
    return true;
#endif // defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
}

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
using fattn_rel_rows_kernel_t = decltype(&fattn_rel_mma_rows<false, GGML_TYPE_F16, true, 1, 1>);

// One row a block, one tile of 8 columns: the KV head's query heads. q4_0 has no other staging than fattn_rel_elem's;
// q8_0 stages through the kernel's own conversion in the float domain, which measures ahead of the f16-domain one
// (55 vs 73 us at four rows over 80K cells).
static fattn_rel_rows_kernel_t fattn_rel_rows_kernel(const ggml_tensor * dst) {
    if (dst->src[1]->type == GGML_TYPE_Q4_0) {
        return fattn_rel_mma_rows<false, GGML_TYPE_Q4_0, true, 1, 1>;
    }
    GGML_ASSERT(dst->src[1]->type == GGML_TYPE_Q8_0);
    return fattn_rel_mma_rows<false, GGML_TYPE_Q8_0, false, 1, 1>;
}
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

// The streams of a row: the warps of the kernel the card holds at once, over the KV heads and rows, each taking the same
// count of the pool's mask words (32 cells each) in turn, so that no stream holds a word more than another: a warp's time
// is its stream's, and one stream of twice the words doubles the step. Shapes and the device alone: the alloc size asks
// before the launch.
static int fattn_rel_rows_streams(const int device, const ggml_tensor * dst) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    GGML_UNUSED_VARS(device, dst);
    return 1;
#else
    static std::mutex mutex;
    static std::map<std::pair<int, const void *>, int> resident; // the blocks of a kernel an SM holds, a device
    const fattn_rel_rows_kernel_t kernel = fattn_rel_rows_kernel(dst);
    int blocks_per_sm;
    {
        std::lock_guard<std::mutex> lock(mutex);
        auto it = resident.find({device, (const void *) kernel});
        if (it == resident.end()) {
            int current = 0, n = 0;
            CUDA_CHECK(cudaGetDevice(&current));
            CUDA_CHECK(cudaSetDevice(device));
            CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, kernel, WARP_SIZE, 0));
            CUDA_CHECK(cudaSetDevice(current));
            it = resident.emplace(std::make_pair(device, (const void *) kernel), std::max(1, n)).first;
        }
        blocks_per_sm = it->second;
    }
    const int64_t warps = int64_t(blocks_per_sm)*ggml_cuda_info().devices[device].nsm;
    const int64_t n_words = dst->src[1]->ne[1]/32;
    const int64_t target = std::max<int64_t>(1, std::min<int64_t>(warps/(dst->src[1]->ne[2]*dst->src[0]->ne[1]), fattn_rel_rows_parts_max));
    const int64_t per_stream = (n_words + target - 1)/target;
    return (int) ((n_words + per_stream - 1)/per_stream);
#endif // defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
}

// A part a stream and (row, query head): its logit maximum, its denominator and 128 weighted values.
static size_t fattn_rel_rows_alloc_size(const int device, const ggml_tensor * dst) {
    return GGML_PAD(ggml_nbytes(dst), 128) + size_t(dst->ne[1])*dst->ne[2]*fattn_rel_rows_streams(device, dst)*130*sizeof(float);
}

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
// The parts of a (row, query head) merged by a block of fattn_rel_rows_merge_lanes x 128 threads: the largest part
// maximum (the sink's logit among them: the sink is a part of weight 1 and no value), each part's weight, then lane l of
// value d sums the parts l, l + lanes, ... and the lanes' sums are added in their order. fattn_rel_decode_merge folds the
// parts one after another, a memory round trip each.
constexpr int fattn_rel_rows_merge_lanes = 4;

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
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

static void ggml_cuda_fattn_rel_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    const auto * q = dst->src[0];
    const auto * k = dst->src[1];
    const auto * v = dst->src[2];
    const size_t align = k->type == GGML_TYPE_Q4_0 ? 8 : 4;
    GGML_ASSERT((uintptr_t) k->data % align == 0 && (uintptr_t) v->data % align == 0);
    const int n_streams = fattn_rel_rows_streams(ctx.device, dst);
    float * partial = (float *) ((char *) dst->data + GGML_PAD(ggml_nbytes(dst), 128));
    float scale;
    memcpy(&scale, dst->op_params, sizeof(scale));
    const auto * sinks = (const float *) dst->src[9]->data;
    const bool shift = ggml_cuda_fattn_shift_query_sinks();
    fattn_rel_rows_kernel(dst)<<<dim3(n_streams, k->ne[2], q->ne[1]), WARP_SIZE, 0, ctx.stream()>>>(
            (const char *) q->data, (const char *) k->data, (const char *) v->data,
            (const uint32_t *) dst->src[3]->data, (const float *) dst->src[6]->data,
            (const int32_t *) dst->src[7]->data, (const int32_t *) dst->src[8]->data, shift ? sinks : nullptr, partial,
            scale, k->ne[1], dst->src[6]->ne[0], q->ne[2]/k->ne[2], 1, 0,
            q->nb[1], q->nb[2], k->nb[1], k->nb[2], v->nb[1], v->nb[2]);
    fattn_rel_rows_merge<<<dim3(q->ne[2], q->ne[1]), dim3(128, fattn_rel_rows_merge_lanes), 0, ctx.stream()>>>(
            partial, shift ? nullptr : sinks, (float *) dst->data, n_streams);
    CUDA_CHECK(cudaGetLastError());
#else
    GGML_UNUSED_VARS(ctx, dst);
    GGML_ABORT("relative decode rows require CUDA");
#endif
}
