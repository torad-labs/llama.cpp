#include "llama-kv-mixed.h"

#include "../ggml/src/ggml-quants.h"

#include <cassert>
#include <cstdlib>
#include <cstring>

bool llama_kv_mixed_enabled() {
    static const bool enabled = ggml_env_switch("LLAMA_INKLING2_KV_MIXED");
    return enabled;
}

uint32_t llama_kv_mixed_hot_cells() {
    static const uint32_t cells = [] {
        const char * env = getenv("LLAMA_INKLING2_KV_MIXED_HOT");
        if (!env || !*env) {
            return 1024u;
        }
        const long v = atol(env);
        return v > 0 ? (uint32_t) v : 1024u;
    }();
    return cells;
}

llama_kv_mixed_pool::~llama_kv_mixed_pool() {
    if (buf) {
        ggml_backend_buffer_free(buf);
    }
    if (ctx) {
        ggml_free(ctx);
    }
}

bool llama_kv_mixed_pool::init(int64_t row_values, uint32_t hot_rows, ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(row_values > 0 && hot_rows > 0);

    ggml_init_params params = {
        /* .mem_size   = */ 1*ggml_tensor_overhead(),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };

    ctx = ggml_init(params);
    if (!ctx) {
        return false;
    }

    n_row_values = row_values;
    n_hot_rows   = hot_rows;

    // one F16 pool serves K and V of a layer: the cache owns two pools where both need mirroring
    hot = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, row_values, hot_rows);
    buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
    if (!buf) {
        return false;
    }

    ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    stamps.assign(hot_rows, {});

    return true;
}

// store-side routing by recency ...
void llama_kv_mixed_pool::mirror(ggml_tensor * cold,
        const uint32_t * cells, size_t n_cells,
        const std::function<bool(uint32_t)> & near_cell) {
    if (!hot || !cold || n_cells == 0) {
        return;
    }

    assert((int64_t) cold->ne[0] == n_row_values);

    const ggml_type cold_type = cold->type;
    if (cold_type != GGML_TYPE_F32 && cold_type != GGML_TYPE_F16 && cold_type != GGML_TYPE_BF16 &&
            cold_type != GGML_TYPE_Q4_0 && cold_type != GGML_TYPE_Q8_0) {
        // the skeleton mirrors the types the plan names; other caches keep their single pool
        return;
    }

    const size_t cold_row = ggml_row_size(cold_type, n_row_values);

    // rows go through host staging: one cell at a time keeps peak memory near two rows
    std::vector<uint8_t> src(cold_row);
    std::vector<float>   f32(n_row_values);
    std::vector<ggml_fp16_t> dst(n_row_values);

    for (size_t i = 0; i < n_cells; ++i) {
        const uint32_t cell = cells[i];

        if (!near_cell(cell)) {
            continue;
        }

        ggml_backend_tensor_get(cold, src.data(), (size_t) cell*cold_row, cold_row);

        switch (cold_type) {
            case GGML_TYPE_F32:
                for (int64_t j = 0; j < n_row_values; ++j) {
                    f32[j] = ((const float *) src.data())[j];
                }
                break;
            case GGML_TYPE_F16:
                for (int64_t j = 0; j < n_row_values; ++j) {
                    f32[j] = ggml_fp16_to_fp32(((const ggml_fp16_t *) src.data())[j]);
                }
                break;
            case GGML_TYPE_BF16:
                for (int64_t j = 0; j < n_row_values; ++j) {
                    f32[j] = ggml_bf16_to_fp32(((const ggml_bf16_t *) src.data())[j]);
                }
                break;
            case GGML_TYPE_Q4_0:
                dequantize_row_q4_0((const block_q4_0 *) src.data(), f32.data(), n_row_values);
                break;
            case GGML_TYPE_Q8_0:
                dequantize_row_q8_0((const block_q8_0 *) src.data(), f32.data(), n_row_values);
                break;
            default:
                assert(false);
        }

        for (int64_t j = 0; j < n_row_values; ++j) {
            dst[j] = ggml_fp32_to_fp16(f32[j]);
        }

        const uint32_t slot = (uint32_t) (cursor++ % n_hot_rows);

        ggml_backend_tensor_set(hot, dst.data(), (size_t) slot*n_row_values*sizeof(ggml_fp16_t),
                (size_t) n_row_values*sizeof(ggml_fp16_t));

        stamps[slot].cell = cell;
    }
}

void llama_kv_mixed_pool::reset_stamps() {
    cursor = 0;
    stamps.assign(n_hot_rows, {});
}

void llama_kv_mixed_pool::merge_rows(ggml_tensor * cold,
        const uint32_t * cells, size_t n_cells,
        const std::function<bool(uint32_t)> & near_cell,
        float * out) const {
    if (!hot || !cold) {
        return;
    }

    const ggml_type cold_type = cold->type;

    // hot rows read whole: one device get covers the pool once per merge batch
    const size_t hot_bytes = (size_t) n_hot_rows*n_row_values*sizeof(ggml_fp16_t);
    std::vector<uint8_t> hbuf(n_cells > 0 ? hot_bytes : 1);
    if (n_cells > 0) {
        ggml_backend_tensor_get(hot, hbuf.data(), 0, hot_bytes);
    }

    for (size_t i = 0; i < n_cells; ++i) {
        const uint32_t cell   = cells[i];
        float *       row_out = out + (size_t) i*n_row_values;

        // exactly one source decides each cell: a live stamp whose owner is still near serves F16, else the cold row
        int64_t slot = -1;
        if (near_cell(cell)) {
            for (uint32_t s = 0; s < n_hot_rows; ++s) {
                if (stamps[s].cell == cell) {
                    slot = s;
                    break;
                }
            }
        }

        if (slot >= 0) {
            const ggml_fp16_t * row = (const ggml_fp16_t *) (hbuf.data() + (size_t) slot*n_row_values*sizeof(ggml_fp16_t));
            for (int64_t j = 0; j < n_row_values; ++j) {
                row_out[j] = ggml_fp16_to_fp32(row[j]);
            }
            continue;
        }

        const size_t cold_row = ggml_row_size(cold_type, n_row_values);
        std::vector<uint8_t> src(cold_row);
        ggml_backend_tensor_get(cold, src.data(), (size_t) cell*cold_row, cold_row);

        switch (cold_type) {
            case GGML_TYPE_F32:
                memcpy(row_out, src.data(), n_row_values*sizeof(float));
                break;
            case GGML_TYPE_F16:
                for (int64_t j = 0; j < n_row_values; ++j) {
                    row_out[j] = ggml_fp16_to_fp32(((const ggml_fp16_t *) src.data())[j]);
                }
                break;
            case GGML_TYPE_BF16:
                for (int64_t j = 0; j < n_row_values; ++j) {
                    row_out[j] = ggml_bf16_to_fp32(((const ggml_bf16_t *) src.data())[j]);
                }
                break;
            case GGML_TYPE_Q4_0:
                dequantize_row_q4_0((const block_q4_0 *) src.data(), row_out, n_row_values);
                break;
            case GGML_TYPE_Q8_0:
                dequantize_row_q8_0((const block_q8_0 *) src.data(), row_out, n_row_values);
                break;
            default:
                assert(false);
        }
    }
}
