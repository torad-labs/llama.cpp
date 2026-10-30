#include "q8-layout.cuh"
#include "vecdotq.cuh"

// Four independent output rows share a producer. The consumer warps retain the
// canonical MMVQ K ownership and fold: four at 1-4 tokens, two at 5-8 tokens.
// Each slot is exactly 4*(1024 code bytes + 64 scale bytes), with no swizzle.
// Both tensor maps belong to the weight buffer and are encoded during loading.
#ifdef GGML_CUDA_Q8_TMA
#include <cudaTypedefs.h>
std::shared_ptr<ggml_cuda_q8_tma_descriptor> ggml_cuda_q8_tma_create(void * base, uint32_t columns, uint32_t rows) {
    if (columns % 1024 || columns > 16384 || !rows || (uintptr_t) base % 16) { return {}; }
    static const auto encode = [] {
        void * p = nullptr;
        cudaDriverEntryPointQueryResult query;
        const auto err = cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &p, 12000, cudaEnableDefault, &query);
        if (err != cudaSuccess || query != cudaDriverEntryPointSuccess) {
            (void) cudaGetLastError();
            return (PFN_cuTensorMapEncodeTiled_v12000) nullptr;
        }
        return (PFN_cuTensorMapEncodeTiled_v12000) p;
    }();
    if (!encode) { return {}; }
    auto maps = std::make_shared<ggml_cuda_q8_tma_descriptor>();
    const cuuint64_t stride[] = { (cuuint64_t) columns / QK8_0 * sizeof(block_q8_0) };
    const cuuint32_t steps[] = {1, 1};
    const auto make = [&](CUtensorMap & map, char * ptr, uint32_t width, uint32_t tile) {
        const cuuint64_t dimensions[] = {width/8, rows};
        const cuuint32_t box[] = {tile/8, 4};
        return encode(&map, CU_TENSOR_MAP_DATA_TYPE_UINT64, 2, ptr, dimensions, stride, box, steps,
                      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
                      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS;
    };
    if (!make(maps->codes, (char *) base, columns, 1024) ||
        !make(maps->scales, (char *) base + columns, columns/16, 64)) { return {}; }
    maps->columns = columns;
    return maps;
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == GGML_CUDA_CC_BLACKWELL
static __device__ __forceinline__ uint32_t q8_tma_shared(const void * p) {
    return (uint32_t) __cvta_generic_to_shared(p);
}
static __device__ __forceinline__ void q8_tma_wait(uint64_t * barrier, uint32_t phase) {
    uint32_t ready;
    do {
        asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2; selp.u32 %0, 1, 0, p; }"
                     : "=r"(ready) : "r"(q8_tma_shared(barrier)), "r"(phase) : "memory");
    } while (!ready);
}
static __device__ __forceinline__ void q8_tma_load(void * dst, const CUtensorMap * map, int k, int row,
                                                 uint64_t * barrier, uint64_t policy) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint "
                 "[%0], [%1, {%2, %3}], [%4], %5;"
                 :: "r"(q8_tma_shared(dst)), "l"((uint64_t) map), "r"(k), "r"(row),
                    "r"(q8_tma_shared(barrier)), "l"(policy) : "memory");
}
#endif

template<int NC, int SLOTS>
__launch_bounds__((NC <= 4 ? 5 : 3)*32, 1)
static __global__ void q8_tma_mmvq(const __grid_constant__ CUtensorMap codes,
                                  const __grid_constant__ CUtensorMap scales,
                                  const block_q8_1 * y, const float * bias, float * dst,
                                  int columns, int rows, int row_offset, int sy, int sd) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == GGML_CUDA_CC_BLACKWELL
    constexpr int NW = NC <= 4 ? 4 : 2;
    constexpr int R = 4, KB = 32, SLOT_BYTES = R*(1024 + 64);
    constexpr int VDR = VDR_Q8_0_Q8_1_MMVQ;
    constexpr int LPB = QI8_0 / VDR, BPI = NW*32/LPB;
    __shared__ __align__(128) char ring[SLOTS][SLOT_BYTES];
    __shared__ uint64_t full[SLOTS], empty[SLOTS];
    __shared__ float partial[NW-1][NC][R][32];
    const int lane = threadIdx.x, warp = threadIdx.y, tid = warp*32 + lane;
    const int row = blockIdx.x*R, tiles = columns/1024;
    if (tid == 0) {
        for (int s = 0; s < SLOTS; ++s) {
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(q8_tma_shared(&full[s])) : "memory");
            asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(q8_tma_shared(&empty[s])), "r"(NW) : "memory");
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();
    ggml_cuda_pdl_sync();
    if (warp == NW) {
        if (lane == 0) {
            uint64_t policy;
            asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(policy));
            for (int t = 0; t < tiles; ++t) {
                const int slot = t % SLOTS;
                if (t >= SLOTS) { q8_tma_wait(&empty[slot], ((t/SLOTS)-1)&1); }
                asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                             :: "r"(q8_tma_shared(&full[slot])), "r"(SLOT_BYTES) : "memory");
                q8_tma_load(ring[slot], &codes, t*128, row + row_offset, &full[slot], policy);
                q8_tma_load(ring[slot] + R*1024, &scales, t*8, row + row_offset, &full[slot], policy);
            }
        }
        return;
    }
    float sums[NC][R] = {};
    const int qs = VDR*(tid % LPB);
    for (int t = 0; t < tiles; ++t) {
        const int slot = t % SLOTS;
        q8_tma_wait(&full[slot], (t/SLOTS)&1);
#pragma unroll
        for (int b = tid/LPB; b < KB; b += BPI) {
            int xq[R][VDR];
            half xd[R];
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const int8_t * q = (const int8_t *) ring[slot] + r*1024 + b*QK8_0;
#pragma unroll
                for (int v = 0; v < VDR; ++v) { xq[r][v] = get_int_b4(q, qs + v); }
                xd[r] = ((const half *) (ring[slot] + R*1024))[r*KB + b];
            }
#pragma unroll
            for (int c = 0; c < NC; ++c) {
                int yq[VDR]; half yd;
                load_q8_1_mmvq_q8_0(y + c*sy + t*KB + b, qs, yq, yd);
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    sums[c][r] += vec_dot_q8_0_q8_1_impl<float, VDR>(xq[r], yq, xd[r], yd);
                }
            }
        }
        __syncwarp();
        if (lane == 0) {
            asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(q8_tma_shared(&empty[slot])) : "memory");
        }
    }
    if (warp > 0) {
#pragma unroll
        for (int c = 0; c < NC; ++c) {
#pragma unroll
            for (int r = 0; r < R; ++r) { partial[warp-1][c][r][lane] = sums[c][r]; }
        }
    }
    asm volatile("bar.sync 1, %0;" :: "n"(NW*32) : "memory");
    if (warp != 0) { return; }
#pragma unroll
    for (int c = 0; c < NC; ++c) {
#pragma unroll
        for (int r = 0; r < R; ++r) {
#pragma unroll
            for (int w = 0; w < NW-1; ++w) { sums[c][r] += partial[w][c][r][lane]; }
            sums[c][r] = warp_reduce_sum<32>(sums[c][r]);
            if (lane == r && row + r < rows) {
                float value = sums[c][r];
                if (bias) { value += bias[c*sd + row + r]; }
                dst[c*sd + row + r] = value;
            }
        }
    }
#else
    GGML_UNUSED(codes); GGML_UNUSED(scales); GGML_UNUSED(y); GGML_UNUSED(bias); GGML_UNUSED(dst);
    GGML_UNUSED(columns); GGML_UNUSED(rows); GGML_UNUSED(row_offset); GGML_UNUSED(sy); GGML_UNUSED(sd);
#endif
}
#endif

bool ggml_cuda_q8_tma(ggml_backend_cuda_context & ctx, const ggml_tensor * weight, const ggml_tensor * dst,
                      const void * y, uint32_t y_stride, const ggml_cuda_mm_fusion_args_device & fusion, float * out) {
#ifdef GGML_CUDA_Q8_TMA
    static const bool enabled = ggml_env_switch("GGML_CUDA_Q8_TMA");
    if (!enabled || ggml_cuda_info().devices[ctx.device].cc != GGML_CUDA_CC_BLACKWELL ||
            ggml_cuda_highest_compiled_arch(ggml_cuda_info().devices[ctx.device].cc) != GGML_CUDA_CC_BLACKWELL ||
        fusion.gate || fusion.gate_bias || fusion.x_scale || fusion.gate_scale ||
        weight->type != GGML_TYPE_Q8_0 || weight->ne[2] != 1 || weight->ne[3] != 1 ||
        dst->ne[1] < 1 || dst->ne[1] > 8 || dst->ne[2] != 1 || dst->ne[3] != 1 || dst->type != GGML_TYPE_F32) { return false; }
    const ggml_cuda_q8_tma_descriptor * maps = nullptr;
    const auto layout = ggml_cuda_q8_layout_of(weight, &maps);
    const size_t row_bytes = layout.blocks_per_row.z * sizeof(block_q8_0);
    if (!maps || weight->ne[0] != maps->columns || weight->nb[1] != row_bytes || layout.offset % row_bytes) { return false; }
    static const int slots = [] {
        const char * value = getenv("GGML_CUDA_Q8_TMA_SLOTS");
        if (!value) { return 2; }
        if (strcmp(value, "2") == 0) { return 2; }
        if (strcmp(value, "4") == 0) { return 4; }
        GGML_ABORT("GGML_CUDA_Q8_TMA_SLOTS must be 2 or 4");
    }();
    const auto launch = [&](auto nc, auto ns) {
        constexpr int NC = decltype(nc)::value, NS = decltype(ns)::value;
        const ggml_cuda_kernel_launch_params params(dim3((weight->ne[1]+3)/4), dim3(32, (NC <= 4 ? 4 : 2)+1), 0, ctx.stream());
        ggml_cuda_kernel_launch(q8_tma_mmvq<NC, NS>, params, maps->codes, maps->scales, (const block_q8_1 *) y,
            (const float *) fusion.x_bias, out, (int) weight->ne[0], (int) weight->ne[1], (int) (layout.offset / row_bytes),
            (int) y_stride, (int) (dst->nb[1]/sizeof(float)));
    };
    const auto dispatch = [&](auto nc) {
        if (slots == 4) { launch(nc, std::integral_constant<int, 4>{}); }
        else            { launch(nc, std::integral_constant<int, 2>{}); }
    };
#define Q8_TMA_CASE(N) case N: dispatch(std::integral_constant<int, N>{}); break
    switch (dst->ne[1]) {
        Q8_TMA_CASE(1); Q8_TMA_CASE(2); Q8_TMA_CASE(3); Q8_TMA_CASE(4);
        Q8_TMA_CASE(5); Q8_TMA_CASE(6); Q8_TMA_CASE(7); Q8_TMA_CASE(8);
    }
#undef Q8_TMA_CASE
    return true;
#else
    GGML_UNUSED(ctx); GGML_UNUSED(weight); GGML_UNUSED(dst); GGML_UNUSED(y);
    GGML_UNUSED(y_stride); GGML_UNUSED(fusion); GGML_UNUSED(out);
    return false;
#endif
}
