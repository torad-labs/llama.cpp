static constexpr __host__ __device__ ggml_cuda_mmq_config ggml_cuda_mmq_get_config_blackwell(ggml_type type, int J, bool fallback) {
    CASE(GGML_TYPE_Q8_0, 128, 3, 64,  8, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, true, true);
    CASE(GGML_TYPE_Q8_0, 128, 3, 64,  8, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, true, false);

    // Smaller staged expert tiles target three blocks per SM with the same work per thread.
    CASE(GGML_TYPE_IQ3_XXS, 128, 3, 64,  8, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, true, true);
    CASE(GGML_TYPE_IQ3_XXS, 128, 3, 64, 16, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, true, true);
    CASE(GGML_TYPE_IQ3_XXS, 128, 3, 64,  8, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, true, false);
    CASE(GGML_TYPE_IQ3_XXS, 128, 3, 64, 16, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, true, false);

    // A routed FFN's K-quant experts read every weight once, so the op is bound by that stream, and one 128-row
    // tile of 256 threads a block (254 registers, one block an SM) leaves the stream idle while the tile computes.
    // These are the Q8_0/IQ3_XXS Blackwell entries' shape - 64-row tiles of 128 threads, three blocks an SM at the
    // same work a thread - at the widths the spread rule picks for a 400- to 1,850-row ubatch. The widths outside
    // them keep the Ampere entries: a 64-row tile loses at eight columns (twice the tiles, the y reads doubling)
    // and past 64 columns (its smaller smem reuse stops paying).
    CASE(GGML_TYPE_Q3_K, 128, 3, 64,  24, GGML_CUDA_MMQ_SRAM_LAYOUT_Q3_K, MMQ_ITER_K, true, false);
    CASE(GGML_TYPE_Q3_K, 128, 3, 64,  32, GGML_CUDA_MMQ_SRAM_LAYOUT_Q3_K, MMQ_ITER_K, true, false);
    CASE(GGML_TYPE_Q3_K, 128, 3, 64,  40, GGML_CUDA_MMQ_SRAM_LAYOUT_Q3_K, MMQ_ITER_K, true, false);
    CASE(GGML_TYPE_Q3_K, 128, 3, 64,  48, GGML_CUDA_MMQ_SRAM_LAYOUT_Q3_K, MMQ_ITER_K, true, false);
    CASE(GGML_TYPE_Q3_K, 128, 3, 64,  56, GGML_CUDA_MMQ_SRAM_LAYOUT_Q3_K, MMQ_ITER_K, true, false);

    CASE(GGML_TYPE_Q4_K, 128, 3, 64,  24, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, true, false);
    CASE(GGML_TYPE_Q4_K, 128, 3, 64,  32, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, true, false);
    CASE(GGML_TYPE_Q4_K, 128, 3, 64,  40, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, true, false);
    CASE(GGML_TYPE_Q4_K, 128, 3, 64,  48, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, true, false);
    CASE(GGML_TYPE_Q4_K, 128, 3, 64,  56, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, true, false);

    CASE(GGML_TYPE_Q5_K, 128, 3, 64,  24, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, true, false);
    CASE(GGML_TYPE_Q5_K, 128, 3, 64,  32, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, true, false);
    CASE(GGML_TYPE_Q5_K, 128, 3, 64,  40, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, true, false);
    CASE(GGML_TYPE_Q5_K, 128, 3, 64,  48, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, true, false);
    CASE(GGML_TYPE_Q5_K, 128, 3, 64,  56, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, true, false);

    CASE(GGML_TYPE_MXFP4, 256, 1, 128,   8, GGML_CUDA_MMQ_SRAM_LAYOUT_FP4, MMQ_ITER_K_FP4, true, true);
    CASE(GGML_TYPE_MXFP4, 256, 1, 128,  16, GGML_CUDA_MMQ_SRAM_LAYOUT_FP4, MMQ_ITER_K_FP4, true, true);
    CASE(GGML_TYPE_MXFP4, 256, 1, 128,  32, GGML_CUDA_MMQ_SRAM_LAYOUT_FP4, MMQ_ITER_K_FP4, true, true);
    CASE(GGML_TYPE_MXFP4, 256, 1, 128,  64, GGML_CUDA_MMQ_SRAM_LAYOUT_FP4, MMQ_ITER_K_FP4, true, true);
    CASE(GGML_TYPE_MXFP4, 256, 1, 128, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_FP4, MMQ_ITER_K_FP4, true, true);
    CASE(GGML_TYPE_MXFP4, 256, 1, 128,   8, GGML_CUDA_MMQ_SRAM_LAYOUT_FP4, MMQ_ITER_K_FP4, true, false);
    CASE(GGML_TYPE_MXFP4, 256, 1, 128,  16, GGML_CUDA_MMQ_SRAM_LAYOUT_FP4, MMQ_ITER_K_FP4, true, false);
    CASE(GGML_TYPE_MXFP4, 256, 1, 128,  24, GGML_CUDA_MMQ_SRAM_LAYOUT_FP4, MMQ_ITER_K_FP4, true, false);
    CASE(GGML_TYPE_MXFP4, 256, 1, 128,  32, GGML_CUDA_MMQ_SRAM_LAYOUT_FP4, MMQ_ITER_K_FP4, true, false);
    CASE(GGML_TYPE_MXFP4, 256, 1, 128,  40, GGML_CUDA_MMQ_SRAM_LAYOUT_FP4, MMQ_ITER_K_FP4, true, false);
    CASE(GGML_TYPE_MXFP4, 256, 1, 128,  48, GGML_CUDA_MMQ_SRAM_LAYOUT_FP4, MMQ_ITER_K_FP4, true, false);
    CASE(GGML_TYPE_MXFP4, 256, 1, 128,  64, GGML_CUDA_MMQ_SRAM_LAYOUT_FP4, MMQ_ITER_K_FP4, true, false);
    CASE(GGML_TYPE_MXFP4, 256, 1, 128,  80, GGML_CUDA_MMQ_SRAM_LAYOUT_FP4, MMQ_ITER_K_FP4, true, false);
    CASE(GGML_TYPE_MXFP4, 256, 1, 128,  96, GGML_CUDA_MMQ_SRAM_LAYOUT_FP4, MMQ_ITER_K_FP4, true, false);
    CASE(GGML_TYPE_MXFP4, 256, 1, 128, 112, GGML_CUDA_MMQ_SRAM_LAYOUT_FP4, MMQ_ITER_K_FP4, true, false);
    CASE(GGML_TYPE_MXFP4, 256, 1, 128, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_FP4, MMQ_ITER_K_FP4, true, false);

    CASE(GGML_TYPE_NVFP4, 256, 1, 128,   8, GGML_CUDA_MMQ_SRAM_LAYOUT_FP4, MMQ_ITER_K_FP4, true, true);
    CASE(GGML_TYPE_NVFP4, 256, 1, 128,  16, GGML_CUDA_MMQ_SRAM_LAYOUT_FP4, MMQ_ITER_K_FP4, true, true);
    CASE(GGML_TYPE_NVFP4, 256, 1, 128,  32, GGML_CUDA_MMQ_SRAM_LAYOUT_FP4, MMQ_ITER_K_FP4, true, true);
    CASE(GGML_TYPE_NVFP4, 256, 1, 128,  64, GGML_CUDA_MMQ_SRAM_LAYOUT_FP4, MMQ_ITER_K_FP4, true, true);
    CASE(GGML_TYPE_NVFP4, 256, 1, 128, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_FP4, MMQ_ITER_K_FP4, true, true);
    CASE(GGML_TYPE_NVFP4, 256, 1, 128,   8, GGML_CUDA_MMQ_SRAM_LAYOUT_FP4, MMQ_ITER_K_FP4, true, false);
    CASE(GGML_TYPE_NVFP4, 256, 1, 128,  16, GGML_CUDA_MMQ_SRAM_LAYOUT_FP4, MMQ_ITER_K_FP4, true, false);
    CASE(GGML_TYPE_NVFP4, 256, 1, 128,  24, GGML_CUDA_MMQ_SRAM_LAYOUT_FP4, MMQ_ITER_K_FP4, true, false);
    CASE(GGML_TYPE_NVFP4, 256, 1, 128,  32, GGML_CUDA_MMQ_SRAM_LAYOUT_FP4, MMQ_ITER_K_FP4, true, false);
    CASE(GGML_TYPE_NVFP4, 256, 1, 128,  40, GGML_CUDA_MMQ_SRAM_LAYOUT_FP4, MMQ_ITER_K_FP4, true, false);
    CASE(GGML_TYPE_NVFP4, 256, 1, 128,  48, GGML_CUDA_MMQ_SRAM_LAYOUT_FP4, MMQ_ITER_K_FP4, true, false);
    CASE(GGML_TYPE_NVFP4, 256, 1, 128,  64, GGML_CUDA_MMQ_SRAM_LAYOUT_FP4, MMQ_ITER_K_FP4, true, false);
    CASE(GGML_TYPE_NVFP4, 256, 1, 128,  80, GGML_CUDA_MMQ_SRAM_LAYOUT_FP4, MMQ_ITER_K_FP4, true, false);
    CASE(GGML_TYPE_NVFP4, 256, 1, 128,  96, GGML_CUDA_MMQ_SRAM_LAYOUT_FP4, MMQ_ITER_K_FP4, true, false);
    CASE(GGML_TYPE_NVFP4, 256, 1, 128, 112, GGML_CUDA_MMQ_SRAM_LAYOUT_FP4, MMQ_ITER_K_FP4, true, false);
    CASE(GGML_TYPE_NVFP4, 256, 1, 128, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_FP4, MMQ_ITER_K_FP4, true, false);

    return ggml_cuda_mmq_get_config_ampere(type, J, fallback);
}

static constexpr __host__ __device__ ggml_cuda_mmq_config ggml_cuda_mmq_get_config_gb10(ggml_type type, int J, bool fallback) {
    if (type == GGML_TYPE_IQ3_XXS || type == GGML_TYPE_Q8_0 ||
            type == GGML_TYPE_Q3_K || type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K) {
        return ggml_cuda_mmq_get_config_ampere(type, J, fallback);
    }
    return ggml_cuda_mmq_get_config_blackwell(type, J, fallback);
}
