#pragma once
#include "ggml.h"
#include <cstring>
#include <cstdint>

// Private metadata on a NONE tensor, copied by the meta backend to each rank.
// Views keep their own op params; the root owns this tag for its whole lifetime.
struct ggml_recurrent_arena_info {
    uint32_t magic;
    uint32_t current_rows;
    uint32_t layers;
};
static constexpr uint32_t GGML_RECURRENT_ARENA_MAGIC = 0x52415253;
static inline void ggml_recurrent_arena_mark(ggml_tensor * t, uint32_t rows, uint32_t layers) {
    GGML_ASSERT(t->op == GGML_OP_NONE && !t->view_src && rows > 0 && rows <= t->ne[1] && layers > 0);
    const ggml_recurrent_arena_info info {GGML_RECURRENT_ARENA_MAGIC, rows, layers};
    static_assert(sizeof(info) <= GGML_MAX_OP_PARAMS, "arena metadata must fit");
    std::memcpy(t->op_params, &info, sizeof(info));
}
static inline ggml_recurrent_arena_info ggml_recurrent_arena_of(const ggml_tensor * t) {
    while (t->view_src) { t = t->view_src; }
    ggml_recurrent_arena_info info {};
    if (t->op != GGML_OP_NONE || t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t)) { return info; }
    std::memcpy(&info, t->op_params, sizeof(info));
    if (info.magic != GGML_RECURRENT_ARENA_MAGIC || !info.current_rows || !info.layers || info.current_rows > t->ne[1]) {
        return {};
    }
    return info;
}

struct ggml_recurrent_arena_window {
    void * base;
    size_t bytes;
    size_t active_bytes;
};
static inline ggml_recurrent_arena_window ggml_recurrent_arena_window_of(const ggml_tensor * state, size_t max_window) {
    const auto info = ggml_recurrent_arena_of(state);
    if (!info.magic || !state->data) { return {}; }
    const ggml_tensor * root = state;
    while (root->view_src) { root = root->view_src; }
    if (!root->data || root->nb[1] > SIZE_MAX/info.current_rows) { return {}; }
    const size_t bytes = root->nb[1]*info.current_rows;
    if (!bytes || bytes > max_window || bytes > SIZE_MAX/info.layers) { return {}; }
    const uintptr_t base = (uintptr_t) root->data, view = (uintptr_t) state->data;
    // A root can also contain rollback planes. The window covers only current
    // rows; a view beginning in a snapshot plane gets no persisting hint.
    if (view < base || view - base >= bytes) { return {}; }
    return {root->data, bytes, bytes*info.layers};
}
