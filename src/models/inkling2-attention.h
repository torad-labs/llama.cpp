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
    uint32_t mean_rows;       // low mean (LLAMA_INKLING2_DECODE_CLASS_LEGACY): the one row count it takes
    uint32_t mean_pool_min;   // low mean (legacy): smallest pool
    uint32_t mean_pool_share; // low mean (legacy): the rows hold pool/mean_pool_share cells or fewer on average
    uint32_t pool_reads;      // low mean: the rows hold pool_reads pools of cells or fewer together
};

// Shared prefix cells count once for each query that sees them. Keep a stable class as tokens enter the pool.
// LLAMA_INKLING2_DECODE_CLASS_LEGACY: 8 rows or more, and a low mean only at mean_rows rows in a pool of mean_pool_min.
template<typename Cells>
inline inkling2_decode_mask inkling2_classify_decode_mask(uint32_t pool, uint32_t rows, bool allow_mean, Cells cells,
        const inkling2_mask_shapes & shapes) {
    if (rows < shapes.rows_min || rows > shapes.rows_max) {
        return inkling2_decode_mask::dense;
    }
    allow_mean = allow_mean && rows == shapes.mean_rows && pool >= shapes.mean_pool_min;
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

// A decode step reads only the cells its rows see, whatever share of the pool one row holds: the decode kernels read a
// visible cell once a KV head and a row, and the dense kernel the whole pool once a query head, so the rows' cells
// together decide. Bounded rows (each a pool/pool_share or less) keep their own class; other rows that hold pool_reads
// pools or fewer together are low mean, and past that a step is dense. Without allow_mean (rows of several sequences,
// whose cells are not counted) only bounded rows leave the dense class.
template<typename Cells>
inline inkling2_decode_mask inkling2_classify_decode_rows(uint32_t pool, uint32_t rows, bool allow_mean, Cells cells,
        const inkling2_mask_shapes & shapes) {
    if (rows < shapes.rows_min || rows > shapes.rows_max || pool < shapes.pool_min) {
        return inkling2_decode_mask::dense;
    }
    bool bounded = true;
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
    return total <= uint64_t(pool)*shapes.pool_reads ? inkling2_decode_mask::low_mean : inkling2_decode_mask::dense;
}
