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

// A performance hint for relative attention: the queries come in runs of n rows of one sequence (a token and its
// drafts), which see nearly the same cells, so a backend may read each cell once for a run. Every row keeps its own mask.
static inline void ggml_flash_attn_ext_set_query_group(struct ggml_tensor * op, int n) {
    GGML_ASSERT(op->op == GGML_OP_FLASH_ATTN_EXT && n >= 0);
    op->op_params[7] = n;
}

static inline int ggml_flash_attn_ext_get_query_group(const struct ggml_tensor * op) {
    GGML_ASSERT(op->op == GGML_OP_FLASH_ATTN_EXT);
    return op->op_params[7];
}

// A performance hint for a relative decode with query sinks: the cells each query row sees, as runs of cells whose
// positions go up one by one. The mask and kv_pos still define the result, so a backend may ignore it.
//   runs [4, n_entries, n_rows + 1] i32, an entry is 4 ints. Row r: entry 0 is (n_runs, n_cells, n_shared, the row's
//   sequence among the step's 32 at most); entry 1 + i is run i (first cell, its position, cells of the row before it,
//   cell stride | 1 << 31 when the shared list holds the run); entry 1 + n_runs is (0, 0, n_cells, 0). The n_shared
//   cells of the runs the shared list holds come first. The last row is the shared list, runs of stride 1 that two rows
//   or more read whole: entry 0 is (n_runs, n_cells, 0, 0), a run is (first cell, its position, cells of the list before
//   it, a bit for each sequence whose rows read it), and the entry after the last run is (0, 0, n_cells, 0).
// Kept in src[4], which the query sinks (ggml_flash_attn_ext_add_rel_pos_sinks) leave free.
static inline void ggml_flash_attn_ext_set_kv_runs(struct ggml_tensor * op, struct ggml_tensor * runs) {
    GGML_ASSERT(op->op == GGML_OP_FLASH_ATTN_EXT && op->src[9] != NULL && op->src[4] == NULL);
    GGML_ASSERT(runs->type == GGML_TYPE_I32 && ggml_is_contiguous(runs) && runs->ne[0] == 4 && runs->ne[1] >= 2);
    GGML_ASSERT(runs->ne[2] == op->src[0]->ne[1] + 1 && runs->ne[3] == 1);
    op->src[4] = runs;
}

static inline const struct ggml_tensor * ggml_flash_attn_ext_get_kv_runs(const struct ggml_tensor * op) {
    return op->op == GGML_OP_FLASH_ATTN_EXT && op->src[9] != NULL ? op->src[4] : NULL;
}
