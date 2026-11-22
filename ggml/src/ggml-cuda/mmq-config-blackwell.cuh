static constexpr __host__ __device__ ggml_cuda_mmq_config ggml_cuda_mmq_get_config_blackwell(ggml_type type, int J, bool fallback) {
    // IQ3_S and IQ4_NL fell through to the Ampere table: 256 threads, 1 block an SM. Measured on an RTX 5090 at Inkling 2
    // Mini's shapes (2304-wide experts, 128 of them; prefill batches of 512 tokens), 128 threads, 2 blocks an SM and 128
    // rows a tile is +3.9% (geo.py, three runs within 0.2%); 3 blocks +3.5%, 4 blocks +1.1%, every 64-row tile slower.
    CASE(GGML_TYPE_IQ3_S, 128, 2, 128, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, true, true);
    CASE(GGML_TYPE_IQ3_S, 128, 2, 128, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, true, false);
    CASE(GGML_TYPE_IQ3_S, 128, 3, 128, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, true, false);
    CASE(GGML_TYPE_IQ3_S, 128, 4, 128, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, true, false);
    CASE(GGML_TYPE_IQ4_NL, 128, 2, 128, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, true, true);
    CASE(GGML_TYPE_IQ4_NL, 128, 2, 128, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, true, false);
    CASE(GGML_TYPE_IQ4_NL, 128, 3, 128, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, true, false);
    CASE(GGML_TYPE_IQ4_NL, 128, 4, 128, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, true, false);

    CASE(GGML_TYPE_Q8_0, 128, 3, 64,  8, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, true, true);
    CASE(GGML_TYPE_Q8_0, 128, 3, 64,  8, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, true, false);

    // Smaller staged expert tiles target three blocks per SM with the same work per thread.
    CASE(GGML_TYPE_IQ3_XXS, 128, 3, 64,  8, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, true, true);
    CASE(GGML_TYPE_IQ3_XXS, 128, 3, 64, 16, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, true, true);
    CASE(GGML_TYPE_IQ3_XXS, 128, 3, 64,  8, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, true, false);
    CASE(GGML_TYPE_IQ3_XXS, 128, 3, 64, 16, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, true, false);

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
    if (type == GGML_TYPE_IQ3_XXS || type == GGML_TYPE_Q8_0) {
        return ggml_cuda_mmq_get_config_ampere(type, J, fallback);
    }
    return ggml_cuda_mmq_get_config_blackwell(type, J, fallback);
}
