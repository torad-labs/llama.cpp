// q4_0 K/V read by the MMA kernel without an f16 copy, see flash_attn_ext_q4_0_load_raw.

#include "../fattn-mma-f16.cuh"

DECL_FATTN_MMA_Q4_0_CASE(256, 256, 2, 8);
DECL_FATTN_MMA_Q4_0_CASE(256, 256, 4, 8);
DECL_FATTN_MMA_Q4_0_CASE(256, 256, 8, 8);
DECL_FATTN_MMA_Q4_0_CASE(256, 256, 32, 2);
