// q8_0 K/V of head size 128 read by the MMA kernel without an f16 copy (the DSpark draft's attention), see
// flash_attn_ext_raw_load.

#include "../fattn-mma-f16.cuh"

DECL_FATTN_MMA_RAW_CASE(128, 128,  2, 8, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0);
DECL_FATTN_MMA_RAW_CASE(128, 128,  4, 8, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0);
DECL_FATTN_MMA_RAW_CASE(128, 128,  8, 8, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0);
DECL_FATTN_MMA_RAW_CASE(128, 128, 32, 2, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0);
