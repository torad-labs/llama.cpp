#pragma once

#include <cstdint>

enum class inkling2_decode_mask { dense, short_rows, bounded_rows, low_mean };

// Shared prefix cells count once for each query that sees them. Keep a stable class as tokens enter the pool.
template<typename Cells>
inline inkling2_decode_mask inkling2_classify_decode_mask(uint32_t pool, uint32_t rows, bool allow_mean, Cells cells) {
    if (pool < 8192 || rows < 8 || rows > 32) {
        return inkling2_decode_mask::dense;
    }
    allow_mean = allow_mean && rows == 23 && pool >= 262144;
    bool bounded = true;
    uint64_t total = 0;
    for (uint32_t i = 0; i < rows; ++i) {
        const uint32_t count = cells(i);
        bounded = bounded && count <= pool/8;
        if (!bounded && !allow_mean) {
            return inkling2_decode_mask::dense;
        }
        total += count;
    }
    if (bounded) {
        return inkling2_decode_mask::bounded_rows;
    }
    return 6*total <= uint64_t(pool)*rows ? inkling2_decode_mask::low_mean : inkling2_decode_mask::dense;
}
