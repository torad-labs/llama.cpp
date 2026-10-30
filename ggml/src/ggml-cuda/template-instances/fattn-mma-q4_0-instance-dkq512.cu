// GLM-5.3's sparse attention on a q4_0 latent cache, read without an f16 copy, see flash_attn_ext_deq_load_tile.

#include "../fattn-mma-f16.cuh"

DECL_FATTN_MMA_RAW_CASE(512, 512,  1, 8, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0);
