// q4_0 K/V of head size 128 read one Q head a tile without an f16 copy (Inkling 2's relative position bias), see
// flash_attn_ext_deq_dense_load_tile.

#include "../fattn-mma-f16.cuh"

DECL_FATTN_MMA_RAW_CASE(128, 128,  8, 1, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0);
DECL_FATTN_MMA_RAW_CASE(128, 128, 16, 1, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0);
DECL_FATTN_MMA_RAW_CASE(128, 128, 32, 1, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0);
DECL_FATTN_MMA_RAW_CASE(128, 128, 64, 1, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0);
