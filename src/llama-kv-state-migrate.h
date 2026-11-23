#pragma once

// the migrator's row conversion and rotation helpers, shared by llama-kv-cache.cpp's state reader and
// tests/test-llama-kv-mixed.cpp's migrate unit: little-endian layouts exactly as scripts/inkling2/kv_stats.py reads
// them, q4_0 byte j holding values 2j low / 2j+1 high under an F16 scale pair, q8_0 a per-block F16 scale plus int8.

#include "ggml.h"
#include "ggml-backend.h"
#include "llama-io.h"

#include "../ggml/src/ggml-quants.h"

#include <cassert>
#include <cmath>
#include <cstring>
#include <map>
#include <vector>

// Slot-state migration across cache-type changes: a state saved under one KV configuration loads into another behind
// LLAMA_KV_STATE_MIGRATE (off by default), so flipping the profile's cache default does not strand saved snapshots and
// saves (doc 18 section 3.3). The file's per-layer records carry their own type id and row bytes in the little-endian
// layout scripts/inkling2/kv_stats.py documents; rows decode to f32, un-rotate with H^T when the source stored rotated
// values, rotate with the target's H where the target config rotates its quantized stores (exactly what fresh tokens
// receive before set_rows), and re-encode. Layer counts and V transposition still have to match, and without the
// switch every check below keeps today's error behavior untouched.

inline bool llama_kv_state_migrate_enabled() {
    static const bool enabled = ggml_env_switch("LLAMA_KV_STATE_MIGRATE");
    return enabled;
}

// can the migrator both read this row type out of the file and write it back as the live cache's
inline bool llama_kv_state_type_supported(int32_t t) {
    return t == GGML_TYPE_F32 || t == GGML_TYPE_F16 || t == GGML_TYPE_BF16 ||
           t == GGML_TYPE_Q4_0 || t == GGML_TYPE_Q8_0;
}

// the orthonormal Walsh-Hadamard matrix of one rotation size, kept for the process lifetime: the cache precomputes
// host copies for the sizes its own stores rotate with, but a state saved while the rotation was on restoring into
// a configuration without it (a flipped LLAMA_ATTN_ROT_DISABLE, say) needs the source side too. Same fill pattern
// ggml_gen_hadamard uses, so the values match the engine's matrices bit for bit.
inline const std::vector<float> & llama_kv_state_rot_matrix(uint32_t n) {
    struct store_t {
        std::map<uint32_t, std::vector<float>> m;

        const std::vector<float> & get(uint32_t n) {
            auto it = m.find(n);
            if (it != m.end()) {
                return it->second;
            }

            std::vector<float> h((size_t) n*n, 0.0f);

            h[0] = 1.0f/sqrtf((float) n);

            for (uint32_t s = 1; s < n; s *= 2) {
                for (uint32_t i = 0; i < s; ++i) {
                    for (uint32_t j = 0; j < s; ++j) {
                        const float val = h[i*(size_t) n + j];

                        h[(i + s)*(size_t) n + (j    )] =  val;
                        h[ i     *(size_t) n + (j + s)] =  val;
                        h[(i + s)*(size_t) n + (j + s)] = -val;
                    }
                }
            }

            it = m.emplace(n, std::move(h)).first;
            return it->second;
        }
    };

    static store_t store;
    return store.get(n);
}

// one cell-row of the state file decoded to f32; block layouts are ggml's own, so the bytes go straight in
inline void llama_kv_state_row_decode(const uint8_t * src, ggml_type type, float * dst, int64_t n_values) {
    switch (type) {
        case GGML_TYPE_F32:
            memcpy(dst, src, n_values*sizeof(float));
            break;
        case GGML_TYPE_F16:
            ggml_fp16_to_fp32_row((const ggml_fp16_t *) src, dst, n_values);
            break;
        case GGML_TYPE_BF16:
            ggml_bf16_to_fp32_row((const ggml_bf16_t *) src, dst, n_values);
            break;
        case GGML_TYPE_Q4_0:
            dequantize_row_q4_0((const block_q4_0 *) src, dst, n_values);
            break;
        case GGML_TYPE_Q8_0:
            dequantize_row_q8_0((const block_q8_0 *) src, dst, n_values);
            break;
        default:
            assert(false);
    }
}

// f32 values of one cell-row encoded into a row as the live cache stores them fresh: the reference quantizers the
// CPU backend and the device set-rows kernels both follow (byte j holds values 2j low, 2j+1 high for Q4_0)
inline void llama_kv_state_row_encode(const float * x, ggml_type type, uint8_t * dst, int64_t n_values) {
    switch (type) {
        case GGML_TYPE_F32:
            memcpy(dst, x, n_values*sizeof(float));
            break;
        case GGML_TYPE_F16:
            ggml_fp32_to_fp16_row(x, (ggml_fp16_t *) dst, n_values);
            break;
        case GGML_TYPE_BF16:
            ggml_fp32_to_bf16_row(x, (ggml_bf16_t *) dst, n_values);
            break;
        case GGML_TYPE_Q4_0:
            quantize_row_q4_0_ref(x, (block_q4_0 *) dst, n_values);
            break;
        case GGML_TYPE_Q8_0:
            quantize_row_q8_0_ref(x, (block_q8_0 *) dst, n_values);
            break;
        default:
            assert(false);
    }
}

// orthonormal Walsh-Hadamard on one row's consecutive groups of `n` values: the forward direction multiplies each
// group by H as the graph's mul_mat against the cache's rot tensor does before a quantized store, and the transpose
// undoes it. Applied in f32 on whole rows; callers pass row lengths that are multiples of n whenever they rotate.
inline void llama_kv_state_rot_apply(const float * h, int n, bool transposed, const float * src, float * dst, int64_t n_values) {
    for (int64_t off = 0; off + n <= n_values; off += n) {
        for (int r = 0; r < n; ++r) {
            float acc = 0.0f;
            if (!transposed) {
                const float * row = h + (int64_t) r*n;
                for (int c = 0; c < n; ++c) {
                    acc += row[c]*src[off + c];
                }
            } else {
                for (int c = 0; c < n; ++c) {
                    acc += h[(int64_t) c*n + r]*src[off + c];
                }
            }
            dst[off + r] = acc;
        }
    }
}

// Reads cell_count consecutive file rows whose destination cell indices are idxs (aligned as state_read_meta laid
// them out), converts each through decode - un-rotate/rotate - encode, and writes it into dst where the strict
// reader would have placed it: same offsets, grouped per maximal run of consecutive destinations. The f32 buffer is
// chunked so peak host memory stays bounded on whole-cache restores.
inline bool llama_kv_state_read_rows_converted(
        llama_io_read_i & io,
          ggml_tensor   * dst,
     const uint32_t     * idxs,
             size_t       cell_count,
             ggml_type    src_type,
             uint64_t     src_row_bytes,
             int64_t      row_values,
             ggml_type    dst_type,
        const float     * rot_src, // source stored rotated values when set, with the matrix to undo
             int          n_rot_src,
        const float     * rot_dst, // target rotates its stores when set, with the matrix to apply
             int          n_rot_dst) {
    const size_t dst_row = ggml_row_size(dst_type, row_values);

    // keep the resident f32 staging near 256 MiB on a whole-pool restore
    const size_t chunk_rows_max = std::max<size_t>(1, (size_t) (64*1024*1024/sizeof(float)/row_values));

    std::vector<uint8_t> buf_src;
    std::vector<float>   buf_a;
    std::vector<float>   buf_b;
    std::vector<uint8_t> buf_dst;

    size_t i0 = 0;
    while (i0 < cell_count) {
        size_t i1 = i0 + 1;
        while (i1 < cell_count && idxs[i1] == idxs[i1 - 1] + 1) {
            ++i1;
        }

        for (size_t r0 = i0; r0 < i1; ) {
            const size_t rows = std::min(i1 - r0, chunk_rows_max);

            const size_t src_span = rows*(size_t) src_row_bytes;
            buf_src.resize(src_span);
            io.read(buf_src.data(), src_span);

            buf_a.resize(rows*(size_t) row_values);
            for (size_t r = 0; r < rows; ++r) {
                llama_kv_state_row_decode(buf_src.data() + r*(size_t) src_row_bytes, src_type,
                        buf_a.data() + r*(size_t) row_values, row_values);
            }

            const float * cur = buf_a.data();
            if (rot_src) {
                buf_b.assign(rows*(size_t) row_values, 0.0f);
                for (size_t r = 0; r < rows; ++r) {
                    llama_kv_state_rot_apply(rot_src, n_rot_src, /*transposed*/ true,
                            buf_a.data() + r*(size_t) row_values,
                            buf_b.data() + r*(size_t) row_values,
                            row_values);
                }
                cur = buf_b.data();

                // the rotation may feed the target's straight after: fold it into one more pass on top
                if (rot_dst) {
                    std::vector<float> tmp(rows*(size_t) row_values, 0.0f);
                    for (size_t r = 0; r < rows; ++r) {
                        llama_kv_state_rot_apply(rot_dst, n_rot_dst, /*forward*/ false,
                                buf_b.data() + r*(size_t) row_values,
                                tmp.data()     + r*(size_t) row_values,
                                row_values);
                    }
                    buf_b.swap(tmp);
                }
            } else if (rot_dst) {
                buf_b.assign(rows*(size_t) row_values, 0.0f);
                for (size_t r = 0; r < rows; ++r) {
                    llama_kv_state_rot_apply(rot_dst, n_rot_dst, /*forward*/ false,
                            buf_a.data() + r*(size_t) row_values,
                            buf_b.data() + r*(size_t) row_values,
                            row_values);
                }
                cur = buf_b.data();
            }

            buf_dst.resize(rows*dst_row);
            for (size_t r = 0; r < rows; ++r) {
                llama_kv_state_row_encode(cur + r*(size_t) row_values, dst_type,
                        buf_dst.data() + r*dst_row, row_values);
            }

            ggml_backend_tensor_set(dst, buf_dst.data(), (size_t) idxs[r0]*dst_row, rows*dst_row);

            r0 += rows;
        }

        i0 = i1;
    }

    return true;
}

