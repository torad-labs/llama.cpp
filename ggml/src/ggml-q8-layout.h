#pragma once

#include <cstddef>
#include "ggml.h"

#if defined(__CUDACC__)
#define GGML_Q8_LAYOUT_HD __host__ __device__
#else
#define GGML_Q8_LAYOUT_HD
#endif

// Physical storage only: the logical tensor still contains 34-byte Q8_0 blocks.
// Codes precede the scale plane; neither values nor allocation size change.
struct ggml_q8_0_split_layout {
    size_t blocks;

    GGML_Q8_LAYOUT_HD constexpr size_t size() const { return blocks * 34; }
    GGML_Q8_LAYOUT_HD constexpr size_t scale_offset() const { return blocks * 32; }
    GGML_Q8_LAYOUT_HD constexpr size_t offset(size_t logical) const {
        const size_t block = logical / 34;
        const size_t byte = logical % 34;
        return byte < 2 ? scale_offset() + block * 2 + byte : block * 32 + byte - 2;
    }
};

#undef GGML_Q8_LAYOUT_HD

// Device copy kernels may parallelize independent bytes, never an in-place
// reordering. The scheduler can transfer a refused copy through canonical host
// storage. Keep this predicate shared with the host aliasing property tests.
static inline bool ggml_q8_copy_needs_staging(const ggml_tensor * src, const ggml_tensor * dst) {
    if (!src->data || !dst->data || ggml_is_empty(src) || ggml_is_empty(dst)) { return false; }
    bool identity = src->type == dst->type && src->data == dst->data;
    for (int d = 0; d < 4; ++d) { identity &= src->ne[d] == dst->ne[d] && src->nb[d] == dst->nb[d]; }
    if (identity) { return false; }
    const uintptr_t s = (uintptr_t) src->data, d = (uintptr_t) dst->data;
    return s <= d ? d - s < ggml_nbytes(src) : s - d < ggml_nbytes(dst);
}
