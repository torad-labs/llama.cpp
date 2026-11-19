#pragma once

#include "ggml.h"

// A performance hint for relative attention. Every mask bit is still applied. It places no bound on an individual row.
static inline void ggml_flash_attn_ext_set_low_mean_mask(struct ggml_tensor * op, bool low_mean) {
    GGML_ASSERT(op->op == GGML_OP_FLASH_ATTN_EXT);
    op->op_params[6] = low_mean;
}

static inline bool ggml_flash_attn_ext_has_low_mean_mask(const struct ggml_tensor * op) {
    GGML_ASSERT(op->op == GGML_OP_FLASH_ATTN_EXT);
    return op->op_params[6] != 0;
}
