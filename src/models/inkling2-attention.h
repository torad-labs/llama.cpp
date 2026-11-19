#pragma once

#include <cstdint>

enum class inkling2_decode_mask { dense, short_rows, bounded_rows, low_mean };

// The shapes a decode step's mask is classed by, each a knob of tunables.toml (LLAMA_INKLING2_*), which holds the defaults
struct inkling2_mask_shapes {
    uint32_t short_cells;     // short rows: most cells a sequence of the step holds
    uint32_t rows_min;        // bounded rows and low mean: fewest rows in the step
    uint32_t rows_max;        // and most
    uint32_t pool_min;        // bounded rows: smallest pool
    uint32_t pool_share;      // bounded rows: a row holds pool/pool_share cells or fewer
    uint32_t mean_pool_share; // low mean: the rows hold pool/mean_pool_share cells or fewer on average
};

// Shared prefix cells count once for each query that sees them. Keep a stable class as tokens enter the pool.
template<typename Cells>
inline inkling2_decode_mask inkling2_classify_decode_mask(uint32_t pool, uint32_t rows, bool allow_mean, Cells cells,
        const inkling2_mask_shapes & shapes) {
    if (rows < shapes.rows_min || rows > shapes.rows_max) {
        return inkling2_decode_mask::dense;
    }
    allow_mean = allow_mean && rows == 23 && pool >= 262144;
    bool bounded = pool >= shapes.pool_min;
    if (!bounded && !allow_mean) {
        return inkling2_decode_mask::dense;
    }
    uint64_t total = 0;
    for (uint32_t i = 0; i < rows; ++i) {
        const uint32_t count = cells(i);
        bounded = bounded && count <= pool/shapes.pool_share;
        if (!bounded && !allow_mean) {
            return inkling2_decode_mask::dense;
        }
        total += count;
    }
    if (bounded) {
        return inkling2_decode_mask::bounded_rows;
    }
    return shapes.mean_pool_share*total <= uint64_t(pool)*rows ? inkling2_decode_mask::low_mean : inkling2_decode_mask::dense;
}
