#include "q8-layout.cuh"
#include "mmvq.cuh"
#include "mmvq-moe.cuh"
#include "mmvq-pq2-mma.cuh"
#include "quantize.cuh"
#include "unary.cuh"
#include "vecdotq.cuh"

#include <cstdint>
#include <type_traits>

enum mmvq_parameter_table_id {
    MMVQ_PARAMETERS_GENERIC = 0,
    MMVQ_PARAMETERS_TURING,
    MMVQ_PARAMETERS_GCN,
    MMVQ_PARAMETERS_RDNA2,
    MMVQ_PARAMETERS_RDNA3_0,
    MMVQ_PARAMETERS_RDNA4,
    MMVQ_PARAMETERS_GB10
};

static constexpr __device__ mmvq_parameter_table_id get_device_table_id() {
#if defined(RDNA4)
    return MMVQ_PARAMETERS_RDNA4;
#elif defined(RDNA3_0)
    return MMVQ_PARAMETERS_RDNA3_0;
#elif defined(RDNA2) || defined(RDNA3_5)
    return MMVQ_PARAMETERS_RDNA2;
#elif defined(GCN) || defined(CDNA)
    return MMVQ_PARAMETERS_GCN;
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_TURING && __CUDA_ARCH__ < GGML_CUDA_CC_AMPERE
    return MMVQ_PARAMETERS_TURING;
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ == GGML_CUDA_CC_DGX_SPARK
    return MMVQ_PARAMETERS_GB10;
#else
    return MMVQ_PARAMETERS_GENERIC;
#endif
}

static __host__ mmvq_parameter_table_id get_device_table_id(int cc) {
    if (GGML_CUDA_CC_IS_RDNA4(cc)) {
        return MMVQ_PARAMETERS_RDNA4;
    }
    if (GGML_CUDA_CC_IS_RDNA3_0(cc)) {
        return MMVQ_PARAMETERS_RDNA3_0;
    }
    if (GGML_CUDA_CC_IS_RDNA2(cc) || GGML_CUDA_CC_IS_RDNA3_5(cc)) {
        return MMVQ_PARAMETERS_RDNA2;
    }
    if (GGML_CUDA_CC_IS_GCN(cc) || GGML_CUDA_CC_IS_CDNA(cc)) {
        return MMVQ_PARAMETERS_GCN;
    }
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_TURING && ggml_cuda_highest_compiled_arch(cc) < GGML_CUDA_CC_AMPERE) {
        return MMVQ_PARAMETERS_TURING;
    }
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) == GGML_CUDA_CC_DGX_SPARK) {
        return MMVQ_PARAMETERS_GB10;
    }
    return MMVQ_PARAMETERS_GENERIC;
}

// Per-architecture maximum batch size for which MMVQ should be used for MUL_MAT_ID.
// Returns a value <= MMVQ_MAX_BATCH_SIZE. Default is MMVQ_MAX_BATCH_SIZE.
// Check https://github.com/ggml-org/llama.cpp/pull/20905#issuecomment-4145835627 for details

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_pascal_older(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ1_S:   return 6;
        case GGML_TYPE_IQ1_M:   return 6;
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 5;
        case GGML_TYPE_IQ2_XXS: return 5;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_IQ4_NL:  return 6;
        case GGML_TYPE_IQ4_XS:  return 5;
        case GGML_TYPE_MXFP4:   return 4;
        case GGML_TYPE_NVFP4:   return 4;
        case GGML_TYPE_Q2_K:    return 4;
        case GGML_TYPE_Q3_K:    return 4;
        case GGML_TYPE_Q4_0:    return 6;
        case GGML_TYPE_Q4_1:    return 6;
        case GGML_TYPE_Q4_K:    return 5;
        case GGML_TYPE_Q5_0:    return 6;
        case GGML_TYPE_Q5_1:    return 6;
        case GGML_TYPE_Q5_K:    return 5;
        case GGML_TYPE_Q6_K:    return 4;
        case GGML_TYPE_Q8_0:    return 4;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_turing_plus(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ2_S:   return 7;
        case GGML_TYPE_IQ3_S:   return 6;
        case GGML_TYPE_IQ3_XXS: return 7;
        case GGML_TYPE_MXFP4:   return 7;
        case GGML_TYPE_NVFP4:   return 8;
        case GGML_TYPE_Q2_K:    return 7;
        case GGML_TYPE_Q3_K:    return 5;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_gcn(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ1_S:   return 5;
        case GGML_TYPE_IQ1_M:   return 5;
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 4;
        case GGML_TYPE_IQ2_XXS: return 4;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_IQ4_NL:  return 6;
        case GGML_TYPE_IQ4_XS:  return 4;
        case GGML_TYPE_Q2_K:    return 4;
        case GGML_TYPE_Q3_K:    return 4;
        case GGML_TYPE_Q4_0:    return 5;
        case GGML_TYPE_Q4_1:    return 5;
        case GGML_TYPE_Q4_K:    return 4;
        case GGML_TYPE_Q5_K:    return 4;
        case GGML_TYPE_Q6_K:    return 4;
        case GGML_TYPE_Q8_0:    return 4;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_cdna(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ2_S:   return 5;
        case GGML_TYPE_IQ2_XS:  return 5;
        case GGML_TYPE_IQ2_XXS: return 5;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 5;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_rdna1_rdna2(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 4;
        case GGML_TYPE_IQ2_XXS: return 4;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_Q2_K:    return 7;
        case GGML_TYPE_Q3_K:    return 4;
        case GGML_TYPE_Q4_K:    return 5;
        case GGML_TYPE_Q5_K:    return 6;
        case GGML_TYPE_Q6_K:    return 5;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_rdna3(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ1_S:   return 6;
        case GGML_TYPE_IQ1_M:   return 6;
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 4;
        case GGML_TYPE_IQ2_XXS: return 4;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_IQ4_NL:  return 6;
        case GGML_TYPE_IQ4_XS:  return 6;
        case GGML_TYPE_Q4_K:    return 4;
        case GGML_TYPE_Q5_K:    return 4;
        case GGML_TYPE_Q6_K:    return 4;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_rdna4(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ1_S:   return 7;
        case GGML_TYPE_IQ1_M:   return 7;
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 4;
        case GGML_TYPE_IQ2_XXS: return 4;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_IQ4_NL:  return 7;
        case GGML_TYPE_IQ4_XS:  return 5;
        case GGML_TYPE_MXFP4:   return 5;
        case GGML_TYPE_NVFP4:   return 5;
        case GGML_TYPE_Q3_K:    return 4;
        case GGML_TYPE_Q4_0:    return 7;
        case GGML_TYPE_Q4_1:    return 7;
        case GGML_TYPE_Q4_K:    return 4;
        case GGML_TYPE_Q5_0:    return 7;
        case GGML_TYPE_Q5_1:    return 7;
        case GGML_TYPE_Q5_K:    return 5;
        case GGML_TYPE_Q6_K:    return 5;
        case GGML_TYPE_Q8_0:    return 7;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

// Ada and newer: past the routed-expert ring's 8 tokens (mmvq-moe.cu), mul_mat_vec_q_moe is a warp a token and, for IQ3_XXS,
// 25 % faster than MMQ at 9-12 tokens and 19 % at 16 (RTX PRO 4500, E288; TORAD.md). Other types are not measured: 8.
// GGML_CUDA_MMVQ_MMID_MAX=N (8-16, read once) sets every type's ceiling to N; 8 is the ceiling before this was added.
static int get_mmvq_mmid_max_batch_ada_plus(ggml_type type) {
    static const int forced = [] {
        const char * s = getenv("GGML_CUDA_MMVQ_MMID_MAX");
        return s ? std::min(MMVQ_MMID_CEILING, std::max(MMVQ_MAX_BATCH_SIZE, atoi(s))) : 0;
    }();
    return forced != 0 ? forced : type == GGML_TYPE_IQ3_XXS ? MMVQ_MMID_CEILING : MMVQ_MAX_BATCH_SIZE;
}

// Host function: returns the max batch size for the current arch+type at runtime.
int get_mmvq_mmid_max_batch(ggml_type type, int cc) {
    // NVIDIA: Volta, Ada Lovelace, and Blackwell always use MMVQ for MUL_MAT_ID.
    if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
        if (cc == GGML_CUDA_CC_VOLTA || cc >= GGML_CUDA_CC_ADA_LOVELACE) {
            return get_mmvq_mmid_max_batch_ada_plus(type);
        }
        if (cc >= GGML_CUDA_CC_TURING) {
            return get_mmvq_mmid_max_batch_turing_plus(type);
        }
        return get_mmvq_mmid_max_batch_pascal_older(type);
    }

    // AMD
    if (GGML_CUDA_CC_IS_AMD(cc)) {
        if (GGML_CUDA_CC_IS_RDNA4(cc)) {
            return get_mmvq_mmid_max_batch_rdna4(type);
        }
        if (GGML_CUDA_CC_IS_RDNA3(cc)) {
            return get_mmvq_mmid_max_batch_rdna3(type);
        }
        if (GGML_CUDA_CC_IS_RDNA1(cc) || GGML_CUDA_CC_IS_RDNA2(cc)) {
            return get_mmvq_mmid_max_batch_rdna1_rdna2(type);
        }
        if (GGML_CUDA_CC_IS_CDNA(cc)) {
            return get_mmvq_mmid_max_batch_cdna(type);
        }
        if (GGML_CUDA_CC_IS_GCN(cc)) {
            return get_mmvq_mmid_max_batch_gcn(type);
        }
    }
    return MMVQ_MAX_BATCH_SIZE;
}

bool ggml_cuda_should_use_mmvq(enum ggml_type type, int cc, int64_t ne11) {
    if (!ggml_is_quantized(type)) {
        return false;
    }
#if !defined(GGML_USE_HIP)
    if (type == GGML_TYPE_PTQ1_0 && GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_TURING) {
        return ne11 <= 7;
    }
#endif
    // k-quants cost more to decode and mvq redoes that per column, so MMQ wins sooner.
    // Only list quant-types MMQ supports, others would fall back to cuBLAS.
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && cc == GGML_CUDA_CC_ADA_LOVELACE) {
        switch (type) { // tuned on RTX 4090
            case GGML_TYPE_Q2_K:
                return ne11 <= 4;
            case GGML_TYPE_Q3_K:
                return ne11 <= 6;
            case GGML_TYPE_Q4_K:
            case GGML_TYPE_Q5_K:
                return ne11 <= 7;
            default:
                return ne11 <= MMVQ_MAX_BATCH_SIZE;
        }
    }
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && cc == GGML_CUDA_CC_BLACKWELL) {
        switch (type) { // tuned on RTX 5090
            case GGML_TYPE_Q2_K:
            case GGML_TYPE_Q3_K:
            case GGML_TYPE_Q4_K:
            case GGML_TYPE_Q5_K:
                return ne11 <= 5;
            case GGML_TYPE_Q6_K:
                return ne11 <= 7;
            default:
                return ne11 <= MMVQ_MAX_BATCH_SIZE;
        }
    }
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && cc == GGML_CUDA_CC_DGX_SPARK) {
        switch (type) { // tuned on DGX Spark GB10
            case GGML_TYPE_Q2_K:
                return ne11 <= 6;
            default:
                return ne11 <= MMVQ_MAX_BATCH_SIZE;
        }
    }
    if (GGML_CUDA_CC_IS_CDNA(cc)) {
        if (GGML_CUDA_CC_IS_CDNA1(cc)) {
            switch (type) {
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                    return ne11 <= 7;
                case GGML_TYPE_Q5_1:
                    return ne11 <= 7;
                case GGML_TYPE_Q8_0:
                    return ne11 <= 6;
                case GGML_TYPE_Q2_K:
                    return ne11 <= 4;
                case GGML_TYPE_Q3_K:
                    return ne11 <= 3;
                case GGML_TYPE_Q4_K:
                    return ne11 <= 2;
                case GGML_TYPE_Q5_K:
                    return ne11 <= 3;
                case GGML_TYPE_Q6_K:
                    return ne11 <= 4;
                case GGML_TYPE_IQ1_S:
                    return ne11 <= 5;
                case GGML_TYPE_IQ2_XXS:
                case GGML_TYPE_IQ3_S:
                case GGML_TYPE_IQ4_XS:
                    return ne11 <= 6;
                default:
                    return ne11 <= MMVQ_MAX_BATCH_SIZE;
            }
        }
        switch (type) { // tuned for CDNA2
            case GGML_TYPE_Q2_K:
                return ne11 <= 5;
            case GGML_TYPE_Q3_K:
            case GGML_TYPE_Q4_K:
            case GGML_TYPE_Q5_K:
                return ne11 <= 3;
            case GGML_TYPE_Q6_K:
                return ne11 <= 5;
            default:
                return ne11 <= MMVQ_MAX_BATCH_SIZE;
        }
    }
    return ne11 <= MMVQ_MAX_BATCH_SIZE;
}

// Device constexpr: returns the max batch size for the current arch+type at compile time.
template <ggml_type type>
static constexpr __device__ int get_mmvq_mmid_max_batch_for_device() {
#if defined(RDNA4)
    return get_mmvq_mmid_max_batch_rdna4(type);
#elif defined(RDNA3)
    return get_mmvq_mmid_max_batch_rdna3(type);
#elif defined(RDNA2) || defined(RDNA1)
    return get_mmvq_mmid_max_batch_rdna1_rdna2(type);
#elif defined(CDNA)
    return get_mmvq_mmid_max_batch_cdna(type);
#elif defined(GCN)
    return get_mmvq_mmid_max_batch_gcn(type);
#elif defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == GGML_CUDA_CC_VOLTA || __CUDA_ARCH__ >= GGML_CUDA_CC_ADA_LOVELACE)
    return MMVQ_MMID_CEILING;
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_TURING
    return get_mmvq_mmid_max_batch_turing_plus(type);
#else
    return get_mmvq_mmid_max_batch_pascal_older(type);
#endif
}

static constexpr __host__ __device__ int calc_nwarps(ggml_type type, int ncols_dst, mmvq_parameter_table_id table_id, bool small_k = false, bool halve_iters = false) {
    if (table_id == MMVQ_PARAMETERS_GENERIC) {
        switch (ncols_dst) {
            case 1:
            case 2:
            case 3:
            case 4:
                return 4;
            case 5:
            case 6:
            case 7:
            case 8:
                return 2;
            default:
                return 1;
        }
    } else if (table_id == MMVQ_PARAMETERS_GCN) {
        switch (ncols_dst) {
            case 1:
            case 2:
            case 3:
            case 4:
                return 2;
            case 5:
            case 6:
            case 7:
            case 8:
            default:
                return 1;
        }
    }
    if (table_id == MMVQ_PARAMETERS_RDNA4) {
        // nwarps=8 benefits types with simple vec_dot on RDNA4 (ncols_dst=1).
        // Types with complex vec_dot (Q3_K, IQ2_*, IQ3_*) regress due to register
        // pressure and lookup table contention at higher thread counts.
        if (ncols_dst == 1) {
            switch (type) {
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                case GGML_TYPE_Q5_0:
                case GGML_TYPE_Q5_1:
                case GGML_TYPE_Q8_0:
                case GGML_TYPE_Q2_K:
                case GGML_TYPE_Q4_K:
                case GGML_TYPE_Q5_K:
                case GGML_TYPE_Q6_K:
                case GGML_TYPE_IQ4_NL:
                case GGML_TYPE_IQ4_XS:
                    return 8;
                default:
                    return 1;
            }
        }
        return 1;
    }
    if (table_id == MMVQ_PARAMETERS_RDNA3_0) {
        // RDNA3 (W7900): stricter whitelist than RDNA4.
        // Q2_K / Q5_K / IQ4_XS regress in full quant sweeps.
        if (ncols_dst == 1) {
            switch (type) {
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                case GGML_TYPE_Q5_0:
                case GGML_TYPE_Q5_1:
                case GGML_TYPE_Q8_0:
                    return 8;
                case GGML_TYPE_Q6_K:
                    return 2;
                case GGML_TYPE_IQ4_NL:
                    return 8;
                default:
                    return 1;
            }
        }
        return 1;
    }
    if (table_id == MMVQ_PARAMETERS_TURING) {
        if (ncols_dst == 1) {
            switch (type) {
                case GGML_TYPE_Q2_K:
                case GGML_TYPE_Q3_K:
                case GGML_TYPE_Q4_K:
                case GGML_TYPE_Q5_K:
                case GGML_TYPE_Q6_K:
                    return 2;
                default:
                    return 4;
            }
        }
        switch (ncols_dst) {
            case 2:
            case 3:
            case 4:
                return 4;
            case 5:
            case 6:
            case 7:
            case 8:
                return 2;
            default:
                return 1;
        }
    }
    if (table_id == MMVQ_PARAMETERS_GB10) {
        const int generic = calc_nwarps(type, ncols_dst, MMVQ_PARAMETERS_GENERIC);
        // Only worth a wider block when it materially reduces K-loop trips (observation).
        if (ncols_dst == 1 && !small_k && halve_iters) {
            switch (type) {
                case GGML_TYPE_Q1_0:
                    return generic;
                case GGML_TYPE_PQ2_0:
                    return generic + generic / 2;
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                case GGML_TYPE_Q5_0:
                case GGML_TYPE_Q5_1:
                case GGML_TYPE_Q8_0:
                case GGML_TYPE_Q4_K:
                case GGML_TYPE_Q5_K:
                case GGML_TYPE_Q6_K:
                case GGML_TYPE_IQ4_NL:
                    return 2 * generic;
                default:
                    break;
            }
        }
        return generic;
    }
    return 1;
}

static constexpr __host__ __device__ int calc_rows_per_block(int ncols_dst, int table_id, bool small_k = false, int nwarps = 1) {
    if (table_id == MMVQ_PARAMETERS_GENERIC || table_id == MMVQ_PARAMETERS_GCN || table_id == MMVQ_PARAMETERS_TURING || table_id == MMVQ_PARAMETERS_GB10) {
        switch (ncols_dst) {
            case 1:
                return small_k ? nwarps : 1;
            case 2:
            case 3:
            case 4:
            case 5:
            case 6:
            case 7:
            case 8:
                return 2;
            default:
                return 1;
        }
    }
    return 1;
}

// A launch shape for mul_mat_vec_q given in full, not by the table: warps, rows per block, and K steps whose loads go out
// before their dot products (more than one for Q8_0 only). 0 is the table's shape.
static constexpr int mmvq_tune(const int nwarps, const int rows_per_block, const int k_unroll) {
    return nwarps | rows_per_block << 4 | k_unroll << 8;
}

struct mmvq_q8_row {
    const int8_t * codes = nullptr;
    const half * scales = nullptr;
};

static __device__ __forceinline__ mmvq_q8_row mmvq_q8_row_of(const ggml_cuda_q8_layout & layout, uint32_t block) {
    if (!layout.base) { return {}; }
    const uint2 rc = fast_div_modulo((uint32_t) (layout.offset/sizeof(block_q8_0)) + block, layout.blocks_per_row);
    const char * row = layout.base + (size_t) rc.x*layout.blocks_per_row.z*sizeof(block_q8_0);
    return {(const int8_t *) row + rc.y*QK8_0, (const half *) (row + layout.blocks_per_row.z*QK8_0) + rc.y};
}

template<bool packed>
static __device__ __forceinline__ void mmvq_load_q8(
        const void * vx, const mmvq_q8_row & row, const int k, const int block, const int qs, int * codes, half & scale) {
    if (!packed || row.codes == nullptr) {
        load_q8_0_mmvq(vx, block, qs, codes, scale);
        return;
    }
    const int8_t * q = row.codes + k*QK8_0;
#pragma unroll
    for (int i = 0; i < VDR_Q8_0_Q8_1_MMVQ; ++i) {
        codes[i] = get_int_b4(q, qs + i);
    }
    scale = row.scales[k];
}

template <ggml_type type, int ncols_dst, bool has_fusion, bool has_gate, bool small_k = false, bool halve_iters = false, int tune = 0, bool packed = false>
__launch_bounds__((tune ? tune & 15 : calc_nwarps(type, ncols_dst, get_device_table_id(), small_k, halve_iters))*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mul_mat_vec_q(
        const void * vx_ptr, const void * vy_ptr, const int32_t * ids_ptr, const ggml_cuda_mm_fusion_args_device fusion, float * dst_ptr,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t stride_row_x, const uint32_t stride_col_y,
        const uint32_t stride_col_dst, const uint3 channel_ratio, const uint32_t stride_channel_x,
        const uint32_t stride_channel_y, const uint32_t stride_channel_dst, const uint3 sample_ratio,
        const uint32_t stride_sample_x, const uint32_t stride_sample_y, const uint32_t stride_sample_dst,
        const uint32_t ids_stride) {
    const void    * GGML_CUDA_RESTRICT vx  = vx_ptr;
    const void    * GGML_CUDA_RESTRICT vy  = vy_ptr;
    const int32_t * GGML_CUDA_RESTRICT ids = ids_ptr;
    float         * GGML_CUDA_RESTRICT dst = dst_ptr;

    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = get_vdr_mmvq(type);
    constexpr mmvq_parameter_table_id table_id = get_device_table_id();
    constexpr int nwarps = tune ? tune & 15 : calc_nwarps(type, ncols_dst, table_id, small_k, halve_iters);
    constexpr int rows_per_cuda_block = tune ? tune >> 4 & 15 : calc_rows_per_block(ncols_dst, table_id, small_k, nwarps);
    constexpr int k_unroll = tune ? tune >> 8 : 1;
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    static_assert(k_unroll >= 1 && (k_unroll == 1 || type == GGML_TYPE_Q8_0), "only Q8_0 issues its loads apart");

    constexpr vec_dot_q_cuda_t vec_dot_q_cuda = get_vec_dot_q_cuda(type);

    const     int tid = warp_size*threadIdx.y + threadIdx.x;
    const     int row0 = rows_per_cuda_block*blockIdx.x;
    const     int blocks_per_row_x = ncols_x / qk;
    constexpr int blocks_per_iter = vdr * nwarps*warp_size / qi;

    const uint32_t channel_dst = blockIdx.y;

    uint32_t channel_x;
    uint32_t channel_y;
    uint32_t sample_dst;

    ggml_cuda_pdl_sync();
    channel_x  = ncols_dst == 1 && ids ? ids[channel_dst]                     : fastdiv(channel_dst, channel_ratio);
    channel_y  = ncols_dst == 1 && ids ? fastmodulo(channel_dst, nchannels_y) : channel_dst;
    sample_dst = blockIdx.z;

    const uint32_t sample_x    = fastdiv(sample_dst, sample_ratio);
    const uint32_t sample_y    = sample_dst;

    constexpr bool use_gate = has_gate;
    bool use_bias = false;
    bool use_gate_bias = false;
    bool use_scale = false;
    bool use_gate_scale = false;
    [[maybe_unused]] const void * vgate = nullptr;
    const float * x_bias = nullptr;
    const float * gate_bias = nullptr;
    const float * x_scale = nullptr;
    const float * gate_scale = nullptr;
    ggml_glu_op active_glu;
    float glu_limit = 0.0f;

    if constexpr (has_fusion) {
        use_bias      = fusion.x_bias    != nullptr;
        use_gate_bias = fusion.gate_bias != nullptr && use_gate;
        vgate         = fusion.gate;
        x_bias        = (const float *) fusion.x_bias;
        gate_bias     = (const float *) fusion.gate_bias;
        active_glu    = fusion.glu_op;
        glu_limit     = fusion.glu_limit;
        if constexpr (type == GGML_TYPE_NVFP4) {
            use_scale      = fusion.x_scale    != nullptr;
            use_gate_scale = fusion.gate_scale != nullptr && use_gate;
            x_scale        = (const float *) fusion.x_scale;
            gate_scale     = (const float *) fusion.gate_scale;
        }
    }


    [[maybe_unused]] float x_biases[ncols_dst]    = { 0.0f };
    [[maybe_unused]] float gate_biases[ncols_dst] = { 0.0f };
    [[maybe_unused]] float x_scales = 1.0f;
    [[maybe_unused]] float gate_scales = 1.0f;
    if constexpr (has_fusion) {
        // 1. Hide latency by prefetching bias, gates and scales here
        // 2. load only on threads that won't die after partial sum calculation
        const uint32_t channel_bias = ids ? channel_x : channel_dst;
        if (threadIdx.x < rows_per_cuda_block && threadIdx.y == 0 &&
            (rows_per_cuda_block == 1 || uint32_t(row0 + threadIdx.x) < stride_col_dst)) {
            if (use_bias) {
                x_bias = x_bias + sample_dst * stride_sample_dst + channel_bias * stride_channel_dst + row0;
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    x_biases[j] = x_bias[j * stride_col_dst + threadIdx.x];
                }
            }
            if (use_gate_bias) {
                gate_bias = gate_bias + sample_dst * stride_sample_dst + channel_bias * stride_channel_dst + row0;
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    gate_biases[j] = gate_bias[j * stride_col_dst + threadIdx.x];
                }
            }
            if constexpr (type == GGML_TYPE_NVFP4) {
                if (use_scale) {
                    x_scales = x_scale[ids ? channel_x : 0];
                }
                if (use_gate_scale) {
                    gate_scales = gate_scale[ids ? channel_x : 0];
                }
            }
        }
    }

    // partial sum for each thread
    float tmp[ncols_dst][rows_per_cuda_block] = {{0.0f}};
    float tmp_gate[ncols_dst][rows_per_cuda_block] = {{0.0f}};

    const block_q8_1 * y = ((const block_q8_1 *) vy) + sample_y*stride_sample_y + channel_y*stride_channel_y;
    const int kbx_offset = sample_x*stride_sample_x + channel_x*stride_channel_x + row0*stride_row_x;
    // Row ownership is invariant across K and verification tokens. Resolve the
    // split plane once per output row instead of dividing every block address.
    mmvq_q8_row q8_rows[rows_per_cuda_block], q8_gates[rows_per_cuda_block];
    if constexpr (packed) {
#pragma unroll
        for (int i = 0; i < rows_per_cuda_block; ++i) {
            q8_rows[i] = mmvq_q8_row_of(fusion.q8_layout, kbx_offset + i*stride_row_x);
            if constexpr (has_fusion && has_gate) {
                q8_gates[i] = mmvq_q8_row_of(fusion.gate_q8_layout, kbx_offset + i*stride_row_x);
            }
        }
    }

    if constexpr ((type == GGML_TYPE_Q1_0 || type == GGML_TYPE_Q2_0 || type == GGML_TYPE_PQ2_0) &&
                  table_id == MMVQ_PARAMETERS_GB10) {
        using block_t = std::conditional_t<type == GGML_TYPE_Q1_0, block_q1_0,
            std::conditional_t<type == GGML_TYPE_Q2_0, block_q2_0, block_pq2_0>>;
        // These packed AoS formats are scoreboard-latency bound on GB10. Prefetch one
        // K iteration ahead; only the first lane that consumes a quant block issues it.
#pragma unroll (type == GGML_TYPE_Q2_0 ? 2 : 1)
        for (int kbx = tid / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
            const int kby = kbx * (qk/QK8_1);
            const int kqs = vdr * (tid % (qi/vdr));
            const int kbx_prefetch = kbx + blocks_per_iter;
            if (kbx_prefetch < blocks_per_row_x && tid % (qi/vdr) == 0) {
#pragma unroll
                for (int i = 0; i < rows_per_cuda_block; ++i) {
#if defined(__CUDA_ARCH__) && !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
                    const block_t * prefetch_ptr = (const block_t *) vx +
                        kbx_offset + i*stride_row_x + kbx_prefetch;
                    asm volatile("prefetch.global.L2 [%0];" :: "l"(__cvta_generic_to_global(prefetch_ptr)));
                    if constexpr (has_fusion && has_gate && type != GGML_TYPE_PQ2_0) {
                        const block_t * gate_prefetch_ptr = (const block_t *) vgate +
                            kbx_offset + i*stride_row_x + kbx_prefetch;
                        asm volatile("prefetch.global.L2 [%0];" :: "l"(__cvta_generic_to_global(gate_prefetch_ptr)));
                    }
#endif
                }
            }
#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
                for (int i = 0; i < rows_per_cuda_block; ++i) {
                    tmp[j][i] += vec_dot_q_cuda(
                        vx, &y[j*stride_col_y + kby], kbx_offset + i*stride_row_x + kbx, kqs);
                    if constexpr (has_fusion) {
                        if constexpr (has_gate) {
                            tmp_gate[j][i] += vec_dot_q_cuda(
                                vgate, &y[j*stride_col_y + kby], kbx_offset + i*stride_row_x + kbx, kqs);
                        }
                    }
                }
            }
        }
    } else if constexpr (k_unroll > 1) {
        // The loads of k_unroll K steps go out before their dot products, one round trip for k_unroll steps. Each sum
        // still takes its terms in K order, so the results are bit for bit the loop's below.
        const int kqs = vdr * (tid % (qi/vdr));
        for (int kbx0 = tid / (qi/vdr); kbx0 < blocks_per_row_x; kbx0 += k_unroll*blocks_per_iter) {
            int  xq[k_unroll][rows_per_cuda_block][vdr];
            half xd[k_unroll][rows_per_cuda_block];
            [[maybe_unused]] int  gq[k_unroll][rows_per_cuda_block][vdr];
            [[maybe_unused]] half gd[k_unroll][rows_per_cuda_block];
            int  yq[k_unroll][ncols_dst][vdr];
            half yd[k_unroll][ncols_dst];
#pragma unroll
            for (int u = 0; u < k_unroll; ++u) {
                // a step past the row loads its last block again and adds nothing: no branch between the loads
                const int kbx = min(kbx0 + u*blocks_per_iter, blocks_per_row_x - 1);
#pragma unroll
                for (int i = 0; i < rows_per_cuda_block; ++i) {
                    mmvq_load_q8<packed>(vx, q8_rows[i], kbx, kbx_offset + i*stride_row_x + kbx, kqs, xq[u][i], xd[u][i]);
                    if constexpr (has_fusion && has_gate) {
                        mmvq_load_q8<packed>(vgate, q8_gates[i], kbx, kbx_offset + i*stride_row_x + kbx, kqs, gq[u][i], gd[u][i]);
                    }
                }
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    load_q8_1_mmvq_q8_0(&y[j*stride_col_y + kbx*(qk/QK8_1)], kqs, yq[u][j], yd[u][j]);
                }
            }
#pragma unroll
            for (int u = 0; u < k_unroll; ++u) {
                // a select, not a branch: with a branch the compiler sinks a step's loads into its math, a round trip each
                const bool in_row = kbx0 + u*blocks_per_iter < blocks_per_row_x;
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
                    for (int i = 0; i < rows_per_cuda_block; ++i) {
                        const float dot = vec_dot_q8_0_q8_1_impl<float, vdr>(xq[u][i], yq[u][j], xd[u][i], yd[u][j]);
                        tmp[j][i] = in_row ? tmp[j][i] + dot : tmp[j][i];
                        if constexpr (has_fusion && has_gate) {
                            const float gdot = vec_dot_q8_0_q8_1_impl<float, vdr>(gq[u][i], yq[u][j], gd[u][i], yd[u][j]);
                            tmp_gate[j][i] = in_row ? tmp_gate[j][i] + gdot : tmp_gate[j][i];
                        }
                    }
                }
            }
        }
    } else {
        for (int kbx = tid / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
            const int kby = kbx * (qk/QK8_1); // y block index that aligns with kbx

            // x block quant index when casting the quants to int
            const int kqs = vdr * (tid % (qi/vdr));

#if !defined(GGML_USE_HIP)
        if constexpr (type == GGML_TYPE_PTQ1_0 && ncols_dst > 1 && ncols_dst <= 3) {
#    pragma unroll
            for (int i = 0; i < rows_per_cuda_block; ++i) {
                float dots[ncols_dst];
                vec_dot_ptq1_0_q8_1_multi<ncols_dst>(vx, &y[kby], kbx_offset + i * stride_row_x + kbx, kqs,
                                                     stride_col_y, dots);
#    pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    tmp[j][i] += dots[j];
                }

                if constexpr (has_fusion) {
                    if constexpr (has_gate) {
                        vec_dot_ptq1_0_q8_1_multi<ncols_dst>(vgate, &y[kby], kbx_offset + i * stride_row_x + kbx, kqs,
                                                             stride_col_y, dots);
#    pragma unroll
                        for (int j = 0; j < ncols_dst; ++j) {
                            tmp_gate[j][i] += dots[j];
                        }
                    }
                }
            }
        } else
#endif
        {
#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
                for (int i = 0; i < rows_per_cuda_block; ++i) {
                    if constexpr (type == GGML_TYPE_Q8_0) {
                        int xq[vdr], yq[vdr];
                        half xd, yd;
                        mmvq_load_q8<packed>(vx, q8_rows[i], kbx, kbx_offset + i*stride_row_x + kbx, kqs, xq, xd);
                        load_q8_1_mmvq_q8_0(&y[j*stride_col_y + kby], kqs, yq, yd);
                        tmp[j][i] += vec_dot_q8_0_q8_1_impl<float, vdr>(xq, yq, xd, yd);
                        if constexpr (has_fusion && has_gate) {
                            mmvq_load_q8<packed>(vgate, q8_gates[i], kbx, kbx_offset + i*stride_row_x + kbx, kqs, xq, xd);
                            tmp_gate[j][i] += vec_dot_q8_0_q8_1_impl<float, vdr>(xq, yq, xd, yd);
                        }
                    } else {
                        tmp[j][i] += vec_dot_q_cuda(vx, &y[j*stride_col_y + kby], kbx_offset + i*stride_row_x + kbx, kqs);
                        if constexpr (has_fusion && has_gate) {
                            tmp_gate[j][i] += vec_dot_q_cuda(vgate, &y[j*stride_col_y + kby], kbx_offset + i*stride_row_x + kbx, kqs);
                        }
                    }
                }
            }
        }
        }
    }

    __shared__ float tmp_shared[nwarps - 1 > 0 ? nwarps - 1 : 1][ncols_dst][rows_per_cuda_block][warp_size];
    [[maybe_unused]] __shared__ float tmp_shared_gate[(has_fusion && (nwarps - 1 > 0)) ? nwarps - 1 : 1][ncols_dst]
                                                     [rows_per_cuda_block][warp_size];

    if (threadIdx.y > 0) {
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
            for (int i = 0; i < rows_per_cuda_block; ++i) {
                tmp_shared[threadIdx.y-1][j][i][threadIdx.x] = tmp[j][i];
                if constexpr (has_fusion) {
                    if constexpr (has_gate) {
                        tmp_shared_gate[threadIdx.y-1][j][i][threadIdx.x] = tmp_gate[j][i];
                    }
                }
            }
        }
    }
    __syncthreads();
    if (threadIdx.y > 0) {
        return;
    }

    dst += sample_dst*stride_sample_dst + channel_dst*stride_channel_dst + row0;

    // sum up partial sums and write back result
#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
        for (int i = 0; i < rows_per_cuda_block; ++i) {
#pragma unroll
            for (int l = 0; l < nwarps-1; ++l) {
                tmp[j][i] += tmp_shared[l][j][i][threadIdx.x];
                if constexpr (has_fusion) {
                    if constexpr (has_gate) {
                        tmp_gate[j][i] += tmp_shared_gate[l][j][i][threadIdx.x];
                    }
                }
            }
            tmp[j][i] = warp_reduce_sum<warp_size>(tmp[j][i]);
            if constexpr (has_fusion) {
                if constexpr (has_gate) {
                    tmp_gate[j][i] = warp_reduce_sum<warp_size>(tmp_gate[j][i]);
                }
            }

            if (threadIdx.x == i && (rows_per_cuda_block == 1 || uint32_t(row0 + i) < stride_col_dst)) {
                float result = tmp[j][i];
                if constexpr (has_fusion) {
                    if constexpr (type == GGML_TYPE_NVFP4) {
                        result *= x_scales;
                    }
                    result += x_biases[j];
                    if constexpr (has_gate) {
                        float gate_value = tmp_gate[j][i];
                        if constexpr (type == GGML_TYPE_NVFP4) {
                            gate_value *= gate_scales;
                        }
                        gate_value += gate_biases[j];
                        if (glu_limit > 0.0f) {
                            gate_value = fminf(gate_value, glu_limit);
                            result     = fminf(fmaxf(result, -glu_limit), glu_limit);
                        }
                        switch (active_glu) {
                            case GGML_GLU_OP_SWIGLU:
                                result *= ggml_cuda_op_silu_single(gate_value);
                                break;
                            case GGML_GLU_OP_GEGLU:
                                result *= ggml_cuda_op_gelu_single(gate_value);
                                break;
                            case GGML_GLU_OP_SWIGLU_OAI:
                                result = ggml_cuda_op_swiglu_oai_single(gate_value, result);
                                break;
                            default:
                                result = result * gate_value;
                                break;
                        }
                    }
                }
                dst[j*stride_col_dst + i] = result;
            }
        }
    }

    if constexpr (!has_fusion) {
        GGML_UNUSED_VARS(use_gate, use_bias, use_gate_bias, use_scale, use_gate_scale, active_glu, glu_limit, gate_bias, x_bias, x_scale, gate_scale, tmp_gate);
    }
    if constexpr (type != GGML_TYPE_NVFP4) {
        GGML_UNUSED_VARS(use_scale, use_gate_scale, x_scale, gate_scale, x_scales, gate_scales);
    }
}

// Dedicated MoE multi-token kernel.
// Grid: (ceil(nrows_x / c_rows_per_block), nchannels_dst)
// Block: (warp_size, ncols_dst) - each warp handles one token independently.
// No shared memory reduction needed since each warp works alone.
template <ggml_type type, int c_rows_per_block>
__launch_bounds__(get_mmvq_mmid_max_batch_for_device<type>()*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mul_mat_vec_q_moe(
        const void * vx_ptr, const void * vy_ptr, const int32_t * ids_ptr,
        float * dst_ptr,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t nrows_x,
        const uint32_t stride_row_x, const uint32_t stride_col_y, const uint32_t stride_col_dst,
        const uint32_t stride_channel_x, const uint32_t stride_channel_y, const uint32_t stride_channel_dst,
        const uint32_t ncols_dst, const uint32_t ids_stride) {
    const void    * GGML_CUDA_RESTRICT vx  = vx_ptr;
    const void    * GGML_CUDA_RESTRICT vy  = vy_ptr;
    const int32_t * GGML_CUDA_RESTRICT ids = ids_ptr;
    float         * GGML_CUDA_RESTRICT dst = dst_ptr;

    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = get_vdr_mmvq(type);
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    constexpr vec_dot_q_cuda_t vec_dot_q_cuda = get_vec_dot_q_cuda(type);

    const uint32_t token_idx   = threadIdx.y;
    const int      row0        = c_rows_per_block*blockIdx.x;
    const int      blocks_per_row_x = ncols_x / qk;
    constexpr int  blocks_per_iter  = vdr * warp_size / qi;

    const uint32_t channel_dst = blockIdx.y;

    if (token_idx >= ncols_dst) {
        return;
    }

    ggml_cuda_pdl_sync();
    const uint32_t channel_x = ids[channel_dst + token_idx * ids_stride];
    const uint32_t channel_y = fastmodulo(channel_dst, nchannels_y);

    const block_q8_1 * y = ((const block_q8_1 *) vy) + channel_y*stride_channel_y + token_idx*stride_col_y;
    const int kbx_offset  = channel_x*stride_channel_x + row0*stride_row_x;

    // partial sum for each thread
    float tmp[c_rows_per_block] = {0.0f};

    for (int kbx = threadIdx.x / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1);
        const int kqs = vdr * (threadIdx.x % (qi/vdr));

#pragma unroll
        for (int i = 0; i < c_rows_per_block; ++i) {
            tmp[i] += vec_dot_q_cuda(vx, &y[kby], kbx_offset + i*stride_row_x + kbx, kqs);
        }
    }

    ggml_cuda_pdl_lc();

    // Warp-level reduction only - no shared memory needed
#pragma unroll
    for (int i = 0; i < c_rows_per_block; ++i) {
        tmp[i] = warp_reduce_sum<warp_size>(tmp[i]);
    }

    // Write results
    if (threadIdx.x < c_rows_per_block && (c_rows_per_block == 1 || uint32_t(row0 + threadIdx.x) < nrows_x)) {
        dst[channel_dst*stride_channel_dst + token_idx*stride_col_dst + row0 + threadIdx.x] = tmp[threadIdx.x];
    }
}

template<ggml_type type>
static std::pair<dim3, dim3> calc_launch_params(
        const int ncols_dst, const int nrows_x, const int nchannels_dst, const int nsamples_or_ntokens,
        const int warp_size, const mmvq_parameter_table_id table_id, const bool small_k = false, const bool halve_iters = false) {
    const int nwarps = calc_nwarps(type, ncols_dst, table_id, small_k, halve_iters);
    const int rpb = calc_rows_per_block(ncols_dst, table_id, small_k, nwarps);
    const int64_t nblocks = (nrows_x + rpb - 1) / rpb;
    const dim3 block_nums(nblocks, nchannels_dst, nsamples_or_ntokens);
    const dim3 block_dims(warp_size, nwarps, 1);
    return {block_nums, block_dims};
}

template<ggml_type type, int c_ncols_dst, bool small_k = false, bool halve_iters = false, int tune = 0, bool packed = false>
static void mul_mat_vec_q_switch_fusion(
        const void * vx, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t stride_row_x, const uint32_t stride_col_y,
        const uint32_t stride_col_dst, const uint3 channel_ratio, const uint32_t stride_channel_x,
        const uint32_t stride_channel_y, const uint32_t stride_channel_dst, const uint3 sample_ratio,
        const uint32_t stride_sample_x, const uint32_t stride_sample_y, const uint32_t stride_sample_dst,
        const dim3 & block_nums, const dim3 & block_dims, const int nbytes_shared,
        const uint32_t ids_stride, cudaStream_t stream) {

    if constexpr (type == GGML_TYPE_Q8_0 && !packed) {
        if (fusion.q8_layout.base || fusion.gate_q8_layout.base) {
            return mul_mat_vec_q_switch_fusion<type, c_ncols_dst, small_k, halve_iters, tune, true>(
                vx, vy, ids, fusion, dst, ncols_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst,
                channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst,
                block_nums, block_dims, nbytes_shared, ids_stride, stream);
        }
    }

    const bool has_fusion = fusion.gate != nullptr || fusion.x_bias != nullptr || fusion.gate_bias != nullptr ||
                            fusion.x_scale != nullptr || fusion.gate_scale != nullptr;
    if constexpr (c_ncols_dst <= MMVQ_MAX_FUSED_NCOLS) {
        if (has_fusion) {
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, nbytes_shared, stream);
            if constexpr (c_ncols_dst == 1) {
                if (fusion.gate != nullptr) {
                    ggml_cuda_kernel_launch(mul_mat_vec_q<type, c_ncols_dst, true, true, small_k, halve_iters, tune, packed>, launch_params,
                         vx, vy, ids, fusion, dst, ncols_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst,
                         channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                         sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride);
                    return;
                }
            }
            GGML_ASSERT(fusion.gate == nullptr && "a gate and GLU fuse at one column only");
            ggml_cuda_kernel_launch(mul_mat_vec_q<type, c_ncols_dst, true, false, small_k, halve_iters, tune, packed>, launch_params,
                 vx, vy, ids, fusion, dst, ncols_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride);
            return;
        }
    }

    GGML_ASSERT(!has_fusion && "fusion only supported for ncols_dst <= MMVQ_MAX_FUSED_NCOLS");

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, nbytes_shared, stream);
    ggml_cuda_kernel_launch(mul_mat_vec_q<type, c_ncols_dst, false, false, small_k, halve_iters, tune, packed>, launch_params,
        vx, vy, ids, fusion, dst, ncols_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst,
        channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
        sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride);
}

template <ggml_type type>
static void mul_mat_vec_q_moe_launch(
        const void * vx, const void * vy, const int32_t * ids, float * dst,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t nrows_x,
        const uint32_t stride_row_x, const uint32_t stride_col_y, const uint32_t stride_col_dst,
        const uint32_t stride_channel_x, const uint32_t stride_channel_y, const uint32_t stride_channel_dst,
        const uint32_t ncols_dst, const uint32_t ids_stride,
        const int warp_size, const int nchannels_dst, cudaStream_t stream) {

    constexpr int rows_per_block = 2; // 2 gives best perf based on tuning
    const int64_t nblocks_rows = (nrows_x + rows_per_block - 1) / rows_per_block;
    const dim3 block_nums(nblocks_rows, nchannels_dst);
    const dim3 block_dims(warp_size, ncols_dst);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, 0, stream);

    ggml_cuda_kernel_launch(mul_mat_vec_q_moe<type, rows_per_block>, launch_params,
        vx, vy, ids, dst, ncols_x, nchannels_y, nrows_x,
        stride_row_x, stride_col_y, stride_col_dst,
        stride_channel_x, stride_channel_y, stride_channel_dst,
        ncols_dst, ids_stride);
}

// SM120 Q8_0 without ids at 1, 2, 3 and 6 columns: GLM-5.3's decode, MTP draft and verify. Two shapes leave the table's,
// both bit for bit (but for the sign of a zero):
// - K under the table's block width: only the warps a K block falls to, at 4 rows a block (MLA k_b and v_b heads).
// - a grid the GPU holds at once and two or more K steps a thread: its loads of 2 or 4 steps go out before their dot
//   products, one round trip where the table's waits for each (the shared expert, MLA q_a and kv_a). The unrolled kernel
//   holds more registers, so only where its grid still fits.
// GGML_CUDA_MMVQ_Q8_SM120_LEGACY=1: the table's launch at every shape, whatever the switches below say.
// GGML_CUDA_MMVQ_Q8_SM120_UNROLL=1|2|4: that many K steps at every shape with two or more, fitting or not (1: none).
// GGML_CUDA_MMVQ_Q8_SM120_WARPS=1|2|4|8: that many warps at the table's rows, no unrolling; not bit for bit.
// GGML_CUDA_MMVQ_Q8_SM120_CHECK=1: the table's launch beside each, and a kernel traps on the first result that differs.
struct mmvq_q8_sm120_env {
    bool legacy;
    bool check;
    int  unroll; // 0: as the grid fits
    int  warps;  // 0: as above
};

static const mmvq_q8_sm120_env & mmvq_q8_sm120_env_get() {
    static const mmvq_q8_sm120_env env = [] {
        const auto digit = [](const char * name, std::initializer_list<int> allowed) {
            const char * value = getenv(name);
            if (value == nullptr || value[0] == '\0' || (value[0] == '0' && value[1] == '\0')) {
                return 0;
            }
            for (const int n : allowed) {
                if (value[0] == '0' + n && value[1] == '\0') {
                    return n;
                }
            }
            GGML_ABORT("%s=%s is none of the values it takes", name, value);
        };
        mmvq_q8_sm120_env e;
        e.legacy = ggml_env_switch("GGML_CUDA_MMVQ_Q8_SM120_LEGACY");
        e.check  = ggml_env_switch("GGML_CUDA_MMVQ_Q8_SM120_CHECK");
        e.unroll = digit("GGML_CUDA_MMVQ_Q8_SM120_UNROLL", {1, 2, 4});
        e.warps  = digit("GGML_CUDA_MMVQ_Q8_SM120_WARPS",  {1, 2, 4, 8});
        return e;
    }();
    return env;
}

static bool mmvq_q8_sm120_checked(const ggml_type type, const int cc, const bool has_ids) {
    const mmvq_q8_sm120_env & env = mmvq_q8_sm120_env_get();
    return env.check && !env.legacy && type == GGML_TYPE_Q8_0 && cc == GGML_CUDA_CC_BLACKWELL && !has_ids;
}

static __global__ void mmvq_q8_sm120_compare(
        const float * tuned, const float * ref, const int nrows, const int ncols,
        const int64_t stride_col, const int64_t stride_channel, const int64_t stride_sample) {
    const int row = blockIdx.x*blockDim.x + threadIdx.x;
    if (row >= nrows) {
        return;
    }
    const int     col     = blockIdx.y % ncols;
    const int     channel = blockIdx.y / ncols;
    const int64_t i       = blockIdx.z*stride_sample + channel*stride_channel + col*stride_col + row;
    if (__float_as_uint(tuned[i]) != __float_as_uint(ref[i]) && !(tuned[i] == 0.0f && ref[i] == 0.0f)) {
        printf("mul_mat_vec_q Q8_0 SM120 check: row %d column %d channel %d sample %d is %08x, %08x from the table's launch\n",
                row, col, channel, (int) blockIdx.z, __float_as_uint(tuned[i]), __float_as_uint(ref[i]));
        __trap();
    }
}

static void mmvq_q8_sm120_compare_launch(
        const float * tuned, const float * ref, const int nrows, const int ncols, const int nchannels, const int nsamples,
        const int64_t stride_col, const int64_t stride_channel, const int64_t stride_sample, cudaStream_t stream) {
    GGML_ASSERT((int64_t) ncols*nchannels <= 65535 && nsamples <= 65535);
    const dim3 grid((nrows + 255)/256, ncols*nchannels, nsamples);
    mmvq_q8_sm120_compare<<<grid, 256, 0, stream>>>(tuned, ref, nrows, ncols, stride_col, stride_channel, stride_sample);
    CUDA_CHECK(cudaGetLastError());
}

// blocks an SM holds at once of the kernel mul_mat_vec_q_switch_fusion launches for these fusion args
template <int ncols_dst, int tune, bool packed = false>
static int mmvq_q8_sm120_blocks_per_sm(const ggml_cuda_mm_fusion_args_device & fusion) {
    if constexpr (!packed) {
        if (fusion.q8_layout.base || fusion.gate_q8_layout.base) {
            return mmvq_q8_sm120_blocks_per_sm<ncols_dst, tune, true>(fusion);
        }
    }
    const auto query = [](const auto kernel) {
        int n = 0;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, kernel, (tune & 15)*WARP_SIZE, 0));
        return n;
    };
    const bool has_fusion = fusion.gate != nullptr || fusion.x_bias != nullptr || fusion.gate_bias != nullptr ||
                            fusion.x_scale != nullptr || fusion.gate_scale != nullptr;
    if constexpr (ncols_dst <= MMVQ_MAX_FUSED_NCOLS) {
        if constexpr (ncols_dst == 1) {
            if (fusion.gate != nullptr) {
                static const int gated = query(mul_mat_vec_q<GGML_TYPE_Q8_0, ncols_dst, true, true, false, false, tune, packed>);
                return gated;
            }
        }
        if (has_fusion) {
            static const int fused = query(mul_mat_vec_q<GGML_TYPE_Q8_0, ncols_dst, true, false, false, false, tune, packed>);
            return fused;
        }
    }
    static const int plain = query(mul_mat_vec_q<GGML_TYPE_Q8_0, ncols_dst, false, false, false, false, tune, packed>);
    return plain;
}

template <ggml_type type>
static void mul_mat_vec_q_switch_ncols_dst(
        const void * vx, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const int ncols_x, const int nrows_x, const int ncols_dst,
        const int stride_row_x, const int stride_col_y, const int stride_col_dst,
        const int nchannels_x, const int nchannels_y, const int nchannels_dst,
        const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const int nsamples_x, const int nsamples_dst, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,
        const int ids_stride, cudaStream_t stream, const bool q8_sm120_legacy = false) {

    GGML_ASSERT(ncols_x % ggml_blck_size(type) == 0);
    GGML_ASSERT(ncols_dst <= (ids ? MMVQ_MMID_CEILING : MMVQ_MAX_BATCH_SIZE));

    const uint3 nchannels_y_fd   = ids ? init_fastdiv_values(nchannels_y) : make_uint3(0, 0, 0);
    const uint3 channel_ratio_fd = ids ? make_uint3(0, 0, 0)              : init_fastdiv_values(nchannels_dst / nchannels_x);
    const uint3 sample_ratio_fd  = init_fastdiv_values(nsamples_dst  / nsamples_x);

    const int device = ggml_cuda_get_device();
    const int                     cc        = ggml_cuda_info().devices[device].cc;
    const int warp_size = ggml_cuda_info().devices[device].warp_size;
    const mmvq_parameter_table_id table_id  = get_device_table_id(cc);

    const bool has_ids = ids != nullptr;

    // SM120 Q8_0 shapes (mmvq_q8_sm120_env); a shape this does not take goes on to the table's launch below.
    if constexpr (type == GGML_TYPE_Q8_0) {
        const mmvq_q8_sm120_env & env = mmvq_q8_sm120_env_get();
        if (cc == GGML_CUDA_CC_BLACKWELL && !has_ids && !env.legacy && !q8_sm120_legacy &&
                (ncols_dst == 1 || ncols_dst == 2 || ncols_dst == 3 || ncols_dst == 6)) {
            const auto launch = [&](auto cols, auto tune_tag) {
                constexpr int nc  = decltype(cols)::value;
                constexpr int tn  = decltype(tune_tag)::value;
                constexpr int nw  = tn & 15;
                constexpr int rpb = tn >> 4 & 15;
                const dim3 grid((nrows_x + rpb - 1)/rpb, nchannels_dst, nsamples_dst);
                const dim3 block(warp_size, nw, 1);
                mul_mat_vec_q_switch_fusion<type, nc, false, false, tn>(
                    vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                    channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                    sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                    grid, block, 0, ids_stride, stream);
            };
            const auto dispatch = [&](auto cols) {
                constexpr int nc        = decltype(cols)::value;
                constexpr int nw_table  = calc_nwarps(type, nc, MMVQ_PARAMETERS_GENERIC);
                constexpr int rpb_table = calc_rows_per_block(nc, MMVQ_PARAMETERS_GENERIC);
                constexpr int bpi_warp  = get_vdr_mmvq(type) * WARP_SIZE / ggml_cuda_type_traits<type>::qi;
                using std::integral_constant;

                switch (env.warps) {
                    case 0: break;
                    case 1: launch(cols, integral_constant<int, mmvq_tune(1, rpb_table, 1)>{}); return true;
                    case 2: launch(cols, integral_constant<int, mmvq_tune(2, rpb_table, 1)>{}); return true;
                    case 4: launch(cols, integral_constant<int, mmvq_tune(4, rpb_table, 1)>{}); return true;
                    case 8: launch(cols, integral_constant<int, mmvq_tune(8, rpb_table, 1)>{}); return true;
                }

                // the table's warps past the first ceil(K blocks / bpi_warp) get no K block: only theirs, as they would add
                // zeros; at one column the table already takes 4 rows a block here (small_k)
                const int blocks_per_row = ncols_x / ggml_cuda_type_traits<type>::qk;
                const int nw_used        = (blocks_per_row + bpi_warp - 1) / bpi_warp;
                if (nw_used <= nw_table/2 && (nc == 1 || nrows_x % 4 == 0)) {
                    if (nw_used <= 1) {
                        launch(cols, integral_constant<int, mmvq_tune(1, 4, 1)>{});
                    } else if constexpr (nw_table >= 4) {
                        launch(cols, integral_constant<int, mmvq_tune(2, 4, 1)>{});
                    }
                    return true;
                }

                const int     steps   = (blocks_per_row + nw_table*bpi_warp - 1) / (nw_table*bpi_warp);
                const int64_t nblocks = (int64_t) ((nrows_x + rpb_table - 1) / rpb_table) * nchannels_dst * nsamples_dst;
                const int64_t nsm     = ggml_cuda_info().devices[device].nsm;
                int unroll = steps >= 2 ? env.unroll : 1;
                if (unroll == 0) {
                    if (steps >= 3 && nblocks <= mmvq_q8_sm120_blocks_per_sm<nc, mmvq_tune(nw_table, rpb_table, 4)>(fusion)*nsm) {
                        unroll = 4;
                    } else if (nblocks <= mmvq_q8_sm120_blocks_per_sm<nc, mmvq_tune(nw_table, rpb_table, 2)>(fusion)*nsm) {
                        unroll = 2;
                    } else {
                        unroll = 1;
                    }
                }
                switch (unroll) {
                    case 2: launch(cols, integral_constant<int, mmvq_tune(nw_table, rpb_table, 2)>{}); return true;
                    case 4: launch(cols, integral_constant<int, mmvq_tune(nw_table, rpb_table, 4)>{}); return true;
                    default: return false;
                }
            };
            bool launched = false;
            switch (ncols_dst) {
                case 1: launched = dispatch(std::integral_constant<int, 1>{}); break;
                case 2: launched = dispatch(std::integral_constant<int, 2>{}); break;
                case 3: launched = dispatch(std::integral_constant<int, 3>{}); break;
                case 6: launched = dispatch(std::integral_constant<int, 6>{}); break;
            }
            if (launched) {
                return;
            }
        }
    }

    // How the K loop divides up at the baseline block width, both decisions below use these.
    constexpr int qk                    = ggml_cuda_type_traits<type>::qk;
    constexpr int qi                    = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr                   = get_vdr_mmvq(type);
    const int     blocks_per_row_x      = ncols_x / qk;
    const int     blocks_per_iter_1warp = vdr * warp_size / qi;

    const auto should_use_small_k = [&](int c_ncols_dst) {
        // When K is small, increase rows_per_block to match nwarps so each warp has more work to do
        // Trigger when the full thread block covers all K blocks in a single loop iteration and few threads remain idle.
        const int  nwarps = calc_nwarps(type, c_ncols_dst, table_id);
        bool       use    = nwarps > 1 && blocks_per_row_x < nwarps * blocks_per_iter_1warp;
        // on CDNA the small_k geometry (rows_per_block = nwarps) is several times slower than the default split-K geometry for these types, and with wave64 it triggers for every K < 4096 projection
        if (GGML_CUDA_CC_IS_CDNA(cc) && (type == GGML_TYPE_Q1_0 || type == GGML_TYPE_Q2_0 || type == GGML_TYPE_PQ2_0)) {
            use = false;
        }

        constexpr std::array<ggml_type, 2> iq_slow_turing = {
            GGML_TYPE_IQ3_XXS,
            GGML_TYPE_IQ3_S,
        };
        constexpr std::array<ggml_type, 8> iq_slow_other = {
            GGML_TYPE_IQ1_S, GGML_TYPE_IQ1_M,   GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS,
            GGML_TYPE_IQ2_S, GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ3_S,   GGML_TYPE_IQ4_XS,
        };
        constexpr std::array<ggml_type, 3> slow_pascal = {
            GGML_TYPE_IQ3_S,
            GGML_TYPE_Q2_K,
            GGML_TYPE_Q3_K,
        };

        const bool is_nvidia_turing_plus  = GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_TURING;
        const bool is_nvidia_pascal_older = GGML_CUDA_CC_IS_NVIDIA(cc) && cc < GGML_CUDA_CC_VOLTA;

        if (is_nvidia_turing_plus) {
            if (ncols_dst == 1 &&
                    std::find(iq_slow_turing.begin(), iq_slow_turing.end(), type) != iq_slow_turing.end()) {
                use = false;
            }
        } else if ((ncols_dst == 1 && std::find(iq_slow_other.begin(), iq_slow_other.end(), type) != iq_slow_other.end()) ||
                (is_nvidia_pascal_older && std::find(slow_pascal.begin(), slow_pascal.end(), type) != slow_pascal.end()) ||
                GGML_CUDA_CC_IS_RDNA(cc)) {
            use = false;
        }

        return use;
    };

    // Whether doubling nwarps pays off on the ncols_dst == 1 path, where K sets the K loop trip count.
    const auto should_halve_iters = [&] {
        if (table_id != MMVQ_PARAMETERS_GB10) {
            return false;
        }

        if (type == GGML_TYPE_PQ2_0 && ncols_x == 6144 && nrows_x == 2048) {
            return false;
        }

        // Expert rows are gathered per token, so a wider block adds reduction work without reuse.
        if (has_ids) {
            return false;
        }

        const int blocks_per_iter = calc_nwarps(type, 1, table_id) * blocks_per_iter_1warp;
        const int iters           = (blocks_per_row_x + blocks_per_iter - 1) /  blocks_per_iter;
        const int iters_wide      = (blocks_per_row_x + blocks_per_iter * 2 - 1) / (blocks_per_iter * 2);

        // An odd trip count leaves half the wider block idle for its last iteration, that tail is
        // only affordable once the loop is long enough to dilute it to an eighth of the work (observation).
        const int idle = iters_wide * 2 - iters;

        return idle * 8 <= iters_wide * 2;
    };

    if (has_ids && ncols_dst > 1) {
        // Multi-token MUL_MAT_ID path - dedicated MoE kernel
        mul_mat_vec_q_moe_launch<type>(
            vx, vy, ids, dst, ncols_x, nchannels_y_fd, nrows_x,
            stride_row_x, stride_col_y, stride_col_dst,
            stride_channel_x, stride_channel_y, stride_channel_dst,
            ncols_dst, ids_stride, warp_size, nchannels_dst, stream);
        return;
    }

    switch (ncols_dst) {
        case 1: {
            // static, else MSVC lambda capture breaks the constexpr uses below
            static constexpr int c_ncols_dst = 1;

            // Tag types keep the flags compile-time, so __launch_bounds__ matches what is launched.
            const auto launch = [&](auto small_k_tag, auto halve_iters_tag) {
                constexpr bool c_small_k = decltype(small_k_tag)::value;
                // Types the table does not promote would compile a second, identical kernel.
                constexpr bool c_promoted =
                    calc_nwarps(type, c_ncols_dst, MMVQ_PARAMETERS_GB10, false, true) !=
                    calc_nwarps(type, c_ncols_dst, MMVQ_PARAMETERS_GB10, false, false);

                constexpr bool c_halve_iters = decltype(halve_iters_tag)::value && c_promoted;

                std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst,
                                                                      nsamples_dst, warp_size, table_id, c_small_k, c_halve_iters);
                mul_mat_vec_q_switch_fusion<type, c_ncols_dst, c_small_k, c_halve_iters>(
                    vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                    channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst, sample_ratio_fd,
                    stride_sample_x, stride_sample_y, stride_sample_dst, dims.first, dims.second, 0, ids_stride,
                    stream);
            };

            if (should_use_small_k(c_ncols_dst)) {
                launch(std::true_type{},  std::false_type{});
            } else if (should_halve_iters()) {
                launch(std::false_type{}, std::true_type{});
            } else {
                launch(std::false_type{}, std::false_type{});
            }
        } break;
        case 2: {
            constexpr int c_ncols_dst = 2;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream);
        } break;
        case 3: {
            constexpr int c_ncols_dst = 3;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream);
        } break;
        case 4: {
            constexpr int c_ncols_dst = 4;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream);
        } break;
        case 5: {
            constexpr int c_ncols_dst = 5;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream);
        } break;
        case 6: {
            constexpr int c_ncols_dst = 6;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream);
        } break;
        case 7: {
            constexpr int c_ncols_dst = 7;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream);
        } break;
        case 8: {
            constexpr int c_ncols_dst = 8;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream);
        } break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}
static void mul_mat_vec_q_switch_type(
        const void * vx, const ggml_type type_x, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const int ncols_x, const int nrows_x, const int ncols_dst,
        const int stride_row_x, const int stride_col_y, const int stride_col_dst,
        const int nchannels_x, const int nchannels_y, const int nchannels_dst,
        const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const int nsamples_x, const int nsamples_dst, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,
        const int ids_stride, cudaStream_t stream, const bool q8_sm120_legacy = false) {
    switch (type_x) {
        case GGML_TYPE_Q1_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q1_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_PQ2_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_PQ2_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_PTQ1_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_PTQ1_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q2_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q2_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q4_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q4_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q4_1:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q4_1>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q5_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q5_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q5_1:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q5_1>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q8_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q8_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream,
                 q8_sm120_legacy);
            break;
        case GGML_TYPE_MXFP4:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_MXFP4>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_NVFP4:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_NVFP4>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q2_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q2_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q3_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q3_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q4_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q4_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q5_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q5_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q6_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q6_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ2_XXS:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ2_XXS>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ2_XS:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ2_XS>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ2_S:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ2_S>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ3_XXS:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ3_XXS>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ1_S:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ1_S>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ1_M:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ1_M>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ4_NL:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ4_NL>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ4_XS:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ4_XS>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ3_S:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ3_S>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

// src1 as q8_1, rows of its columns padded to MATRIX_ROW_PADDING: an input other MUL_MATs of the evaluation read too is
// quantized once, into the copy the evaluation holds for them (ggml_cuda_mmvq_shared_q8_1), any other into own
static char * mmvq_src1_q8_1(ggml_backend_cuda_context & ctx, const ggml_tensor * src1, const ggml_type type_x,
                             ggml_cuda_pool_alloc<char> & own) {
    const int64_t ne10_padded = GGML_PAD(src1->ne[0], MATRIX_ROW_PADDING);
    ggml_cuda_mmvq_shared_q8_1::entry * shared = ctx.mmvq_shared_q8_1.find(src1);
    char * q8_1 = shared != nullptr ? shared->q8_1 : own.alloc(ggml_nrows(src1) * ne10_padded * sizeof(block_q8_1)/QK8_1);
    if (shared == nullptr || !shared->quantized) {
        const size_t ts = ggml_type_size(src1->type);
        quantize_row_q8_1_cuda((const float *) src1->data, nullptr, q8_1, type_x, src1->ne[0],
            src1->nb[1] / ts, src1->nb[2] / ts, src1->nb[3] / ts, ne10_padded, src1->ne[1], src1->ne[2], src1->ne[3],
            ctx.stream());
        if (shared != nullptr) {
            shared->quantized = true;
        }
    }
    return q8_1;
}

// GGML_CUDA_MMVQ_MOE_QUANTIZE_CHECK=1: a routed ring's results from the vectors it quantized beside those from the
// quantize launch's
static __global__ void mmvq_moe_quantize_check(const float * own, const float * launch, const int64_t n) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n && __float_as_uint(own[i]) != __float_as_uint(launch[i])) {
        printf("mmvq_moe quantize check: result %lld is %08x from the ring's own q8_1, %08x from the quantize launch's\n",
                (long long) i, __float_as_uint(own[i]), __float_as_uint(launch[i]));
        __trap();
    }
}

bool ggml_cuda_mul_mat_vec_q_moe_gate_fuses(const ggml_tensor * mm, const ggml_tensor * gate, const int cc) {
    const ggml_tensor * src0 = mm->src[0];
    const ggml_tensor * src1 = mm->src[1];
    if (mm->op != GGML_OP_MUL_MAT_ID || src0->ne[3] != 1 || src1->ne[3] != 1 || gate->type != src0->type ||
            !ggml_are_same_shape(gate, src0) || !ggml_are_same_stride(gate, src0)) {
        return false;
    }
    // the pairs as ggml_cuda_mul_mat_vec_q reads a MUL_MAT_ID's, and src1's q8_1 as mmvq_src1_q8_1 lays it out
    const int64_t ts      = ggml_type_size(src0->type);
    const int64_t s11     = GGML_PAD(src1->ne[0], MATRIX_ROW_PADDING) / QK8_1;
    const int64_t y_bytes = ggml_cuda_mmvq_moe_y_bytes(src0->ne[0], mm->ne[1], mm->ne[2], src1->ne[1], src1->ne[1]*s11, s11);
    return ggml_cuda_mmvq_moe_usable(cc, src0->type, src0->data, gate->data, src0->ne[0], src0->ne[1], src0->nb[1]/ts,
            src0->nb[2]/ts, mm->ne[1], mm->ne[2]) &&
        ggml_cuda_mmvq_moe_keeps_y(src0->type, src0->ne[0], /*gate =*/ true, y_bytes);
}

void ggml_cuda_mul_mat_vec_q(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst,
        const ggml_cuda_mm_fusion_args_host * fusion) {
    GGML_ASSERT(        src1->type == GGML_TYPE_F32);
    GGML_ASSERT(        dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(!ids || ids->type  == GGML_TYPE_I32); // Optional, used for batched GGML_MUL_MAT_ID.

    GGML_TENSOR_BINARY_OP_LOCALS;

    cudaStream_t stream = ctx.stream();

    const size_t ts_src0 = ggml_type_size(src0->type);
    const size_t ts_src1 = ggml_type_size(src1->type);
    const size_t ts_dst  = ggml_type_size(dst->type);

    GGML_ASSERT(        nb00       == ts_src0);
    GGML_ASSERT(        nb10       == ts_src1);
    GGML_ASSERT(        nb0        == ts_dst);
    GGML_ASSERT(!ids || ids->nb[0] == ggml_type_size(ids->type));

    GGML_ASSERT(!ids || ne12 <= MMVQ_MMID_CEILING);

    const int32_t *  ids_d = ids ? (const int32_t *)  ids->data : nullptr;
    float         *  dst_d =       (float         *)  dst->data;

    // PQ2_0 at 1-8 columns (decode and an MTP verify) on int8 tensor cores fed by a TMA ring per SM (mmvq-pq2-mma.cu),
    // which also takes a gated FFN (SWIGLU, no biases) and a residual add at columns past MMVQ_MAX_FUSED_NCOLS
    const bool pq2_mma = src0->type == GGML_TYPE_PQ2_0 && !ids && ne02 == 1 && ne03 == 1 && ne12 == 1 && ne13 == 1 &&
        (fusion == nullptr || (fusion->gate_bias == nullptr && fusion->x_scale == nullptr && fusion->gate_scale == nullptr &&
            fusion->glu_limit == 0.0f &&
            (fusion->gate == nullptr || (fusion->x_bias == nullptr && fusion->glu_op == GGML_GLU_OP_SWIGLU)))) &&
        ggml_cuda_mmvq_pq2_mma_usable(ggml_cuda_info().devices[ctx.device].cc, src0->data,
            fusion && fusion->gate ? fusion->gate->data : nullptr, ne00, ne01, nb01 / ts_src0, ne1);

    ggml_cuda_mm_fusion_args_device fusion_local{};
    fusion_local.q8_layout = ggml_cuda_q8_layout_of(src0);
    if (fusion && fusion->gate) { fusion_local.gate_q8_layout = ggml_cuda_q8_layout_of(fusion->gate); }
    GGML_ASSERT(!fusion_local.q8_layout.base || !ids);

    if (fusion) {
        GGML_ASSERT(  ids || dst->ne[1] <= MMVQ_MAX_FUSED_NCOLS || pq2_mma);
        // Scale fusion is only allowed for NVFP4 currently as the cost of checking this at run-time in the prologue is
        // non-negligible for some models such as gpt-oss-20b
        GGML_ASSERT((fusion->x_scale == nullptr && fusion->gate_scale == nullptr) || src0->type == GGML_TYPE_NVFP4);

        if (fusion->x_bias) {
            GGML_ASSERT(fusion->x_bias->type == GGML_TYPE_F32);
            GGML_ASSERT(fusion->x_bias->ne[0] == dst->ne[0]);
            GGML_ASSERT(!ids || fusion->x_bias->ne[1] == src0->ne[2]);
            GGML_ASSERT(ids || ggml_cuda_mmvq_fusion_operand_ok(fusion->x_bias, dst));
            fusion_local.x_bias = fusion->x_bias->data;
        }
        if (fusion->gate) {
            GGML_ASSERT(fusion->gate->type == src0->type && ggml_are_same_stride(fusion->gate, src0));
            fusion_local.gate = fusion->gate->data;
        }
        if (fusion->gate_bias) {
            GGML_ASSERT(fusion->gate_bias->type == GGML_TYPE_F32);
            GGML_ASSERT(fusion->gate_bias->ne[0] == dst->ne[0]);
            GGML_ASSERT(!ids || fusion->gate_bias->ne[1] == src0->ne[2]);
            GGML_ASSERT(ids || ggml_cuda_mmvq_fusion_operand_ok(fusion->gate_bias, dst));
            fusion_local.gate_bias = fusion->gate_bias->data;
        }
        if (fusion->x_scale) {
            GGML_ASSERT(fusion->x_scale->type == GGML_TYPE_F32);
            GGML_ASSERT(ggml_is_contiguous(fusion->x_scale));
            GGML_ASSERT(ggml_nelements(fusion->x_scale) == (ids ? src0->ne[2] : 1));
            fusion_local.x_scale = fusion->x_scale->data;
        }
        if (fusion->gate_scale) {
            GGML_ASSERT(fusion->gate_scale->type == GGML_TYPE_F32);
            GGML_ASSERT(ggml_is_contiguous(fusion->gate_scale));
            GGML_ASSERT(ggml_nelements(fusion->gate_scale) == (ids ? src0->ne[2] : 1));
            fusion_local.gate_scale = fusion->gate_scale->data;
        }
        fusion_local.glu_op    = fusion->glu_op;
        fusion_local.glu_limit = fusion->glu_limit;
    }

    // If src0 is a temporary compute buffer, clear any potential padding.
    if (ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        const size_t size_data  = ggml_nbytes(src0);
        const size_t size_alloc = ggml_backend_buffer_get_alloc_size(src0->buffer, src0);
        if (size_alloc > size_data) {
            GGML_ASSERT(ggml_is_contiguously_allocated(src0));
            GGML_ASSERT(!src0->view_src);
            CUDA_CHECK(cudaMemsetAsync((char *) src0->data + size_data, 0, size_alloc - size_data, stream));
        }
    }

    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);

    // routed experts at 1-8 tokens, each SM streaming its tiles of expert rows through a ring of bulk copies, and each
    // distinct expert read once (mmvq-moe.cu)
    const bool moe_ring = ids && ne03 == 1 && ne13 == 1 &&
        (fusion == nullptr || (fusion->x_scale == nullptr && fusion->gate_scale == nullptr)) &&
        ggml_cuda_mmvq_moe_usable(ggml_cuda_info().devices[ctx.device].cc, src0->type, src0->data, fusion_local.gate, ne00,
            ne01, nb01 / ts_src0, nb02 / ts_src0, ne1, ne2);
    // a ring whose tokens' vectors are each its own (a down projection: ne11 == ne1) and read by no other MUL_MAT
    // quantizes them itself from f32, into the shared memory it would copy them to, bit for bit as the quantize launch:
    // no launch between the gate/up and the down (GLM-5.3's proxy under -sm tensor at 32K cached tokens, RTX 5070 Ti:
    // 3.1 us of the chain after each gate/up, 43 a token). GGML_CUDA_MMVQ_MOE_QUANTIZE_LEGACY=1: the launch;
    // GGML_CUDA_MMVQ_MOE_QUANTIZE_CHECK=1: the ring runs from both, and a kernel traps on the first result whose bits differ
    const bool ring_quantizes = moe_ring && ctx.mmvq_shared_q8_1.find(src1) == nullptr &&
        ggml_cuda_mmvq_moe_quantizes_y(src0->type, ne00, fusion_local.gate != nullptr, ne1, ne2, ne11, src1->data,
            nb11 / ts_src1, nb12 / ts_src1);
    static const bool ring_quantize_check = ggml_env_switch("GGML_CUDA_MMVQ_MOE_QUANTIZE_CHECK");

    ggml_cuda_pool_alloc<char> src1_q8_1_own(ctx.pool());
    char * src1_q8_1 = ring_quantizes && !ring_quantize_check ? nullptr :
        mmvq_src1_q8_1(ctx, src1, src0->type, src1_q8_1_own);

    const int64_t s01 = src0->nb[1] / ts_src0;
    const int64_t s11 = ne10_padded / QK8_1;
    const int64_t s1  =  dst->nb[1] / ts_dst;
    const int64_t s02 = src0->nb[2] / ts_src0;
    const int64_t s2  =  dst->nb[2] / ts_dst;
    const int64_t s03 = src0->nb[3] / ts_src0;
    const int64_t s3  =  dst->nb[3] / ts_dst;

    const int64_t s12 = ne11*s11;
    const int64_t s13 = ne12*s12;

    // For MUL_MAT_ID the memory layout is different than for MUL_MAT:
    const int64_t ncols_dst          = ids ? ne2  : ne1;
    const int64_t nchannels_y        = ids ? ne11 : ne12;
    const int64_t nchannels_dst      = ids ? ne1  : ne2;
    const int64_t stride_col_dst     = ids ? s2   : s1;
    const int64_t stride_col_y       = ids ? s12  : s11;
    const int64_t stride_channel_dst = ids ? s1   : s2;
    const int64_t stride_channel_y   = ids ? s11  : s12;

    const int64_t ids_stride = ids ? ids->nb[1] / ggml_type_size(ids->type) : 0;

    if (moe_ring) {
        ggml_cuda_mmvq_moe_args args{};
        args.type                   = src0->type;
        args.vx                     = src0->data;
        args.vgate                  = fusion_local.gate;
        args.y                      = src1_q8_1;
        args.ids                    = ids_d;
        args.x_bias                 = (const float *) fusion_local.x_bias;
        args.gate_bias              = (const float *) fusion_local.gate_bias;
        args.dst                    = dst_d;
        args.glu_op                 = fusion_local.glu_op;
        args.glu_limit              = fusion_local.glu_limit;
        args.ncols_x                = ne00;
        args.nrows_x                = ne01;
        args.stride_channel_x_bytes = s02 * (int64_t) ts_src0;
        args.n_used                 = ne1;
        args.n_experts              = ne02;
        args.ntokens                = ne2;
        args.ids_stride             = ids_stride;
        args.nchannels_y            = nchannels_y;
        args.stride_col_y           = stride_col_y;
        args.stride_channel_y       = stride_channel_y;
        args.stride_col_dst         = stride_col_dst;
        args.stride_channel_dst     = stride_channel_dst;
        args.stride_bias            = stride_channel_dst; // as mul_mat_vec_q reads an expert's bias
        args.tile_ctr               = ctx.pq2_tile_counter();
        // the ids an earlier ring launch on this stream read in this evaluation (a down projection's, the gate/up's
        // before it) are whole before this launch starts, so it lists its experts and issues its first tiles before its
        // dependency wait (mmvq-moe.cu). GGML_CUDA_MMVQ_MOE_IDS_EARLY_LEGACY=1: past it
        static const bool ids_early_legacy = ggml_env_switch("GGML_CUDA_MMVQ_MOE_IDS_EARLY_LEGACY");
        const ggml_tensor *& ids_before = ctx.mmvq_moe_ids[ctx.curr_stream_no];
        args.ids_ready = !ids_early_legacy && ids_before == ids;
        ids_before     = ids;
        args.l2_issue_stop = ctx.l2_issue_stop.ptr;
        args.route_stats = ctx.route_stats.ptr;
        if (!ring_quantizes) {
            ggml_cuda_mmvq_moe(args, stream);
            return;
        }
        ggml_cuda_pool_alloc<float> check_dst(ctx.pool());
        if (ring_quantize_check) {
            GGML_ASSERT(ggml_is_contiguous(dst));
            args.dst = check_dst.alloc(ggml_nelements(dst));
            ggml_cuda_mmvq_moe(args, stream); // from the quantize launch's vectors
            args.dst = dst_d;
        }
        args.y        = nullptr;
        args.y_f32    = (const float *) src1->data;
        args.y_f32_s1 = nb11 / ts_src1;
        args.y_f32_s2 = nb12 / ts_src1;
        ggml_cuda_mmvq_moe(args, stream);
        if (ring_quantize_check) {
            const int64_t n = ggml_nelements(dst);
            mmvq_moe_quantize_check<<<(n + 255) / 256, 256, 0, stream>>>(dst_d, check_dst.get(), n);
        }
        return;
    }
    // past one token only the ring above fuses a routed gate (ggml_cuda_mul_mat_vec_q_moe_gate_fuses)
    GGML_ASSERT(!fusion || !ids || dst->ne[2] == 1);

    if (pq2_mma) {
        ggml_cuda_mmvq_pq2_mma(src0->data, fusion_local.gate, src1_q8_1, (const float *) fusion_local.x_bias,
            dst_d, ne00, ne01, ne1, s01, s11, s1, ctx.pq2_next, stream);
        return;
    }

    const auto run = [&](float * out, const bool q8_sm120_legacy) {
        if (!q8_sm120_legacy && !mmvq_q8_sm120_env_get().legacy && !ids &&
            ggml_cuda_q8_tma(ctx, src0, dst, src1_q8_1, stride_col_y, fusion_local, out)) {
            return;
        }
        mul_mat_vec_q_switch_type(
            src0->data, src0->type, src1_q8_1, ids_d, fusion_local, out, ne00,
            ne01,              ncols_dst,     s01, stride_col_y,     stride_col_dst,
            ne02, nchannels_y, nchannels_dst, s02, stride_channel_y, stride_channel_dst,
            ne03,              ne3,           s03, s13,              s3,               ids_stride, stream, q8_sm120_legacy);
    };
    if (!mmvq_q8_sm120_checked(src0->type, ggml_cuda_info().devices[ctx.device].cc, ids != nullptr)) {
        run(dst_d, false);
        return;
    }
    // the table's launch first, so a bias read from dst is read before either writes it
    ggml_cuda_pool_alloc<float> ref(ctx.pool(), (ne3 - 1)*s3 + (nchannels_dst - 1)*stride_channel_dst +
        (ncols_dst - 1)*stride_col_dst + ne01);
    run(ref.get(), true);
    run(dst_d, false);
    mmvq_q8_sm120_compare_launch(dst_d, ref.get(), ne01, ncols_dst, nchannels_dst, ne3, stride_col_dst, stride_channel_dst,
        s3, stream);
}

void ggml_cuda_mul_mat_vec_q_pq2_group(ggml_backend_cuda_context & ctx, ggml_tensor * const * dsts, const int n,
                                       const bool feeds_fold) {
    GGML_ASSERT(n >= 2 && n <= PQ2_MMA_MAX_GROUP);
    const ggml_tensor * src1 = dsts[0]->src[1];
    GGML_ASSERT(src1->type == GGML_TYPE_F32 && src1->nb[0] == sizeof(float) && src1->ne[2] == 1 && src1->ne[3] == 1);

    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];

    const void * vx[PQ2_MMA_MAX_GROUP];
    float *      dst[PQ2_MMA_MAX_GROUP];
    int64_t      nrows[PQ2_MMA_MAX_GROUP];
    int64_t      stride_row[PQ2_MMA_MAX_GROUP];
    int64_t      stride_col_dst[PQ2_MMA_MAX_GROUP];
    for (int g = 0; g < n; ++g) {
        const ggml_tensor * src0 = dsts[g]->src[0];
        GGML_ASSERT(dsts[g]->src[1] == src1 && src0->type == GGML_TYPE_PQ2_0 && src0->ne[0] == ne10);
        GGML_ASSERT(dsts[g]->type == GGML_TYPE_F32 && dsts[g]->nb[0] == sizeof(float) && dsts[g]->ne[1] == ne11);
        vx[g]             = src0->data;
        dst[g]            = (float *) dsts[g]->data;
        nrows[g]          = src0->ne[1];
        stride_row[g]     = src0->nb[1] / ggml_type_size(src0->type);
        stride_col_dst[g] = dsts[g]->nb[1] / sizeof(float);
    }

    cudaStream_t stream = ctx.stream();

    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);
    ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(), ne11*ne10_padded * sizeof(block_q8_1)/QK8_1);
    {
        const int64_t s11 = src1->nb[1] / sizeof(float);
        const int64_t s12 = src1->nb[2] / sizeof(float);
        const int64_t s13 = src1->nb[3] / sizeof(float);
        quantize_row_q8_1_cuda((const float *) src1->data, nullptr, src1_q8_1.get(), GGML_TYPE_PQ2_0, ne10, s11, s12, s13,
            ne10_padded, ne11, 1, 1, stream);
    }

    ggml_cuda_mmvq_pq2_mma_group(n, vx, dst, nrows, stride_row, stride_col_dst, src1_q8_1.get(), ne10, ne11,
        ne10_padded / QK8_1, ctx.pq2_next, feeds_fold, stream);
}

bool ggml_cuda_mul_mat_vec_q_concat_supported(const ggml_tensor * const * w, const int n, const ggml_tensor * src1,
                                              const ggml_tensor * dst, const int cc) {
    const ggml_tensor * w0 = w[0];
    const size_t        ts = ggml_type_size(w0->type);
    const auto q8_layout = ggml_cuda_q8_layout_of(w0);
    // PQ2_0 runs on its tensor-core kernel (ggml_cuda_mul_mat_vec_q_pq2_group groups it)
    if (n < 2 || !ggml_is_quantized(w0->type) || w0->type == GGML_TYPE_PQ2_0 || w0->ne[2] != 1 || w0->ne[3] != 1 ||
            w0->nb[1] != ggml_row_size(w0->type, w0->ne[0])) {
        return false;
    }
    // the weights' stride, the channel stride of the launch
    const ptrdiff_t stride = (const char *) w[1]->data - (const char *) w0->data;
    for (int g = 0; g < n; ++g) {
        const auto other = ggml_cuda_q8_layout_of(w[g]);
        if ((q8_layout.base != nullptr) != (other.base != nullptr)) { return false; }
        if (q8_layout.base && (other.blocks_per_row.z != q8_layout.blocks_per_row.z ||
                stride % (q8_layout.blocks_per_row.z * sizeof(block_q8_0)) != 0 ||
                other.offset % (q8_layout.blocks_per_row.z * sizeof(block_q8_0)) != q8_layout.offset %
                    (q8_layout.blocks_per_row.z * sizeof(block_q8_0)))) { return false; }
        if (w[g]->type != w0->type || !ggml_are_same_shape(w[g], w0) || !ggml_are_same_stride(w[g], w0) ||
                ggml_backend_buffer_get_usage(w[g]->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE ||
                (const char *) w[g]->data - (const char *) w0->data != g*stride) {
            return false;
        }
    }
    const int64_t ne01 = w0->ne[1];
    const int64_t ne11 = src1->ne[1];
    // the kernel's block offsets are 32-bit; the last row of the last weight bounds them
    if (stride <= 0 || stride % ts != 0 || (n - 1)*(stride/ts) + ne01*(w0->nb[1]/ts) > INT_MAX) {
        return false;
    }
    // a block's rows are bounded by the column stride, here all n weights' rows: each block must lie inside one weight's,
    // and a block has at most 8 rows (calc_rows_per_block, at most calc_nwarps)
    if (ne01 % 8 != 0) {
        return false;
    }
    return src1->type == GGML_TYPE_F32 && src1->nb[0] == sizeof(float) && src1->ne[0] == w0->ne[0] && src1->ne[2] == 1 &&
        src1->ne[3] == 1 && ne11 <= MMVQ_MAX_BATCH_SIZE && ggml_cuda_should_use_mmvq(w0->type, cc, ne11) &&
        dst->type == GGML_TYPE_F32 && ggml_is_contiguous(dst) && dst->ne[0] == n*ne01 && dst->ne[1] == ne11 &&
        dst->ne[2] == 1 && dst->ne[3] == 1;
}

void ggml_cuda_mul_mat_vec_q_concat(ggml_backend_cuda_context & ctx, const ggml_tensor * const * w, const int n,
                                    const ggml_tensor * src1, ggml_tensor * dst) {
    const ggml_tensor * w0 = w[0];
    const size_t        ts = ggml_type_size(w0->type);

    ggml_cuda_pool_alloc<char> src1_q8_1_own(ctx.pool());
    const char * src1_q8_1 = mmvq_src1_q8_1(ctx, src1, w0->type, src1_q8_1_own);

    const int64_t ne00 = w0->ne[0];
    const int64_t ne01 = w0->ne[1];
    const int64_t ne11 = src1->ne[1];
    const int64_t s01  = w0->nb[1] / ts;
    const int64_t s0w  = ((const char *) w[1]->data - (const char *) w0->data) / (int64_t) ts;
    const int64_t s11  = GGML_PAD(src1->ne[0], MATRIX_ROW_PADDING) / QK8_1;

    // channel g: weight g, src1 (a channel stride of 0), and dst's rows [g*ne01, (g + 1)*ne01), n*ne01 to a column
    ggml_cuda_mm_fusion_args_device fusion{};
    fusion.q8_layout = ggml_cuda_q8_layout_of(w0);
    const auto run = [&](float * out, const bool q8_sm120_legacy) {
        mul_mat_vec_q_switch_type(
            w0->data, w0->type, src1_q8_1, /*ids =*/ nullptr, fusion, out, ne00,
            ne01, ne11, s01, s11, n*ne01,
            n, n, n, s0w, 0, ne01,
            1, 1, 0, 0, 0, /*ids_stride =*/ 0, ctx.stream(), q8_sm120_legacy);
    };
    if (!mmvq_q8_sm120_checked(w0->type, ggml_cuda_info().devices[ctx.device].cc, false)) {
        run((float *) dst->data, false);
        return;
    }
    ggml_cuda_pool_alloc<float> ref(ctx.pool(), n*ne01*ne11);
    run(ref.get(), true);
    run((float *) dst->data, false);
    mmvq_q8_sm120_compare_launch((const float *) dst->data, ref.get(), ne01, ne11, n, 1, n*ne01, ne01, 0, ctx.stream());
}

void ggml_cuda_op_mul_mat_vec_q(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream) {

    const int64_t ne00 = src0->ne[0];
    const int64_t row_diff = row_high - row_low;

    const int64_t ne10 = src1->ne[0];
    GGML_ASSERT(ne10 % QK8_1 == 0);

    const int64_t ne0 = dst->ne[0];

    int id = ggml_cuda_get_device();

    // the main device has a larger memory buffer to hold the results from all GPUs
    // nrows_dst == nrows of the matrix that the kernel writes into
    const int64_t nrows_dst = id == ctx.device ? ne0 : row_diff;

    const int stride_row_x = ne00 / ggml_blck_size(src0->type);
    const int stride_col_y = src1_padded_row_size / QK8_1;

    ggml_cuda_mm_fusion_args_device fusion_local{};
    mul_mat_vec_q_switch_type(
        src0_dd_i, src0->type, src1_ddq_i, nullptr, fusion_local, dst_dd_i, ne00, row_diff, src1_ncols, stride_row_x, stride_col_y, nrows_dst,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, stream);

    GGML_UNUSED_VARS(src1, dst, src1_ddf_i, src1_ncols, src1_padded_row_size);
}
