#include "llama-kv-cache.h"

#include "gguf.h"

#include "../ggml/src/ggml-quants.h"

#include "llama-host-threads.h"
#include "llama-impl.h"
#include "llama-io.h"
#include "llama-kv-cache-kpool.h"
#include "llama-kv-runs.h"
#include "llama-model.h"
#include "llama-context.h"

#include <algorithm>
#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <unordered_map>

static bool ggml_is_power_of_2(int n) {
    return (n & (n - 1)) == 0;
}

// orthonormal Walsh-Hadamard rotation matrix
// note: res^2 == I
static void ggml_gen_hadamard(ggml_tensor * tensor) {
    assert(tensor->type == GGML_TYPE_F32);

    const int n = tensor->ne[0];

    assert(ggml_is_power_of_2(n));
    assert(tensor->ne[1] == n);
    assert(tensor->ne[2] == 1);
    assert(tensor->ne[3] == 1);

    std::vector<float> data_f32;

    float * data = (float *) tensor->data;

    if (tensor->type != GGML_TYPE_F32) {
        data_f32.resize(n*n);
        data = data_f32.data();
    }

    data[0*n + 0] = 1.0 / sqrtf(n);

    for (int s = 1; s < n; s *= 2) {
        for (int i = 0; i < s; i++) {
            for (int j = 0; j < s; j++) {
                const float val = data[i*n + j];

                data[(i + s)*n + (j    )] =  val;
                data[(i    )*n + (j + s)] =  val;
                data[(i + s)*n + (j + s)] = -val;
            }
        }
    }

    if (tensor->type != GGML_TYPE_F32) {
        ggml_quantize_chunk(tensor->type, data, tensor->data, 0, 1, n*n, nullptr);
    }
}

//
// llama_kv_cache
//

llama_kv_cache::llama_kv_cache(
        const llama_model & model,
                     bool   no_alloc,
      const llama_hparams & hparams,
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   v_trans,
                     bool   offload,
                     bool   unified,
                 uint32_t   kv_size,
                 uint32_t   n_seq_max,
                 uint32_t   n_swa,
           llama_swa_type   swa_type,
           llama_memory_t   mem_other,
    const layer_filter_cb & filter,
    const  layer_reuse_cb & reuse,
    const  layer_share_cb & share,
               const char * name_tag,
                     bool   is_indexer,
                     bool   host_mapped) :
    model(model), hparams(hparams), v_trans(v_trans),
    delta_inputs_legacy(ggml_env_switch("LLAMA_INKLING2_DELTA_INPUTS_LEGACY")),
    batch_delta_spans(!ggml_env_switch("GGML_SCHED_BATCH_INPUTS_LEGACY")),
    n_seq_max(n_seq_max), n_stream(unified ? 1 : n_seq_max), n_swa(n_swa), is_indexer(is_indexer),
    attn_rot_bufs(no_alloc), swa_type(swa_type), ctxs_bufs(no_alloc),
    other(static_cast<llama_kv_cache *>(mem_other)),
    v_cells_impl(other ? other->v_cells_impl : std::make_shared<llama_kv_cells_vec>()),
    v_cells(*v_cells_impl) {

    if (host_mapped && (!offload || unified || v_trans || !hparams.is_mla() ||
                       hparams.indexer_kpool == 0 || hparams.n_lora_kv != 512 || type_k != GGML_TYPE_Q8_0)) {
        throw std::runtime_error("host KV requires non-unified GLM pooled MLA with GPU attention and Q8_0 latent caches");
    }

    // shared cells view the source cache's K/V tensors, so the cell count
    // follows the source allocation: a fitted target can be smaller than the
    // draft default and oversized views would overflow the source tensors
    if (other) {
        const uint32_t size_other = other->get_size();
        if (kv_size != size_other) {
            LLAMA_LOG_WARN("%s: kv_size = %u overridden to %u to match the shared source cache\n", __func__, kv_size, size_other);
            kv_size = size_other;
        }
    }

    // A prompt read in equal steps grows the cache by a step's rows each step, and with it every graph shape that n_kv sets:
    // padded to 256, each 2,048-row step of GLM-5.3 Flash rebuilt its graphs and planned their memory again, 45-60 ms of a
    // 555 ms step with both cards idle (two RTX PRO 6000, -sm tensor, Nov 1, 2026). Read per cache, so a context takes the
    // value set when it is made
    if (hparams.indexer_kpool > 0) {
        const char * pad = getenv("LLAMA_KV_PAD_PREFILL");
        const long   val = pad ? strtol(pad, nullptr, 10) : 8192;
        const uint32_t n_pad_cur = 256u;

        n_pad_prefill = val > 0 ? GGML_PAD((uint32_t) std::min<long>(val, kv_size), n_pad_cur) : 0;
    }

    // A pooled cache's views are made with it, so their switches (LLAMA_KPOOL_SHARED_VIEWS_LEGACY,
    // LLAMA_KPOOL_PATCH_INPUT_LEGACY) are read here: made on their first use, they were read in the first decode, after
    // a caller that set a switch for one context had put it back
    if (hparams.indexer_kpool > 0) {
        get_kpool_views();
    }

    // A pool that sequences share grows by every sequence's rows each step, and a graph holds only while its padded length
    // does. One sequence drafting two tokens crosses 256 cells every ~115 steps; 23 of them in one pool cross it every 5
    // (~50 cells a step), and each crossing builds every decode shape again and plans its memory: the target's, the MTP
    // head's two proposals and its catch-up. On four RTX PRO 6000 (-sm tensor, 23 sessions, Nov 9, 2026) a graph was
    // planned in 34 of 64 decodes, 8.7 ms each, and a session's target graph was reused in 41 of its ~61 steps; on one
    // card the crossing step and the one after it took 76 ms more than a 31 ms step. Padded to 256 cells a sequence, the
    // pool crosses as often a step as one sequence does; the next power of two, which GGML_PAD and the pool padding
    // derived from it (llama_kpool_pool_pad) take. LLAMA_KV_PAD_SHARED_LEGACY=1: 256, whatever the pool holds
    if ((hparams.indexer_kpool > 0 || model.arch == LLM_ARCH_INKLING2 || model.arch == LLM_ARCH_DFLASH) && n_stream < n_seq_max) {
        const char * legacy = getenv("LLAMA_KV_PAD_SHARED_LEGACY");

        if (!legacy || atoi(legacy) != 1) {
            const uint64_t a_sequence = 256u;
            uint64_t pad = a_sequence;
            while (pad < std::min<uint64_t>(a_sequence*n_seq_max, kv_size)) {
                pad *= 2;
            }
            // A session's graphs are sized on its own pools, padded to this many cells (llama_kpool_n_pools_ubatch), so
            // at 8,192 cells (23 sequences) every graph a session uses is built again each 8,192 tokens it grows, and
            // sessions 8,192 tokens apart share none. LLAMA_KV_PAD_SHARED=N: at least N cells (the next power of two):
            // the steps are that much rarer and the sessions inside one share their graphs, and the indexer scores the
            // empty pools the padding adds
            const char * wide = getenv("LLAMA_KV_PAD_SHARED");
            if (wide && atol(wide) > 0) {
                const uint64_t pad_default = pad;
                while (pad < std::min<uint64_t>((uint64_t) atol(wide), kv_size)) {
                    pad *= 2;
                }
                pad_shared_wide = pad > pad_default;
            }
            n_pad_shared = (uint32_t) pad;
        }
    }

    // Below the cells its indexer selects a pool reads densely and kept 256, so a pool that sequences share crossed it
    // every few steps until it scored: on four RTX PRO 6000 (-sm tensor, 23 sessions of ~48 prompt tokens, Nov 10, 2026)
    // two crossings in a session's first 18 steps, each a step that ran its graphs uncaptured (+47 ms) and one that
    // captured them (+70 ms) on a 59 ms step. LLAMA_KV_PAD_DENSE_LEGACY=1: 256 while the pool reads densely
    if (n_pad_shared > 0) {
        const char * legacy = getenv("LLAMA_KV_PAD_DENSE_LEGACY");

        pad_dense_shared = !legacy || atoi(legacy) != 1;
        pad_dense_inkling = pad_dense_shared && hparams.indexer_kpool == 0 &&
                (model.arch == LLM_ARCH_INKLING2 || llama_model_is_inkling_draft(model));
        if (!ggml_env_switch("LLAMA_KV_CELL_CAPACITY_LEGACY")) {
            n_pad_cells = std::max(n_pad_shared, 65536u);
        }
    }

    const uint32_t n_layer = hparams.n_layer_all;

    // define a comparator for the buft -> ctx map to ensure that the order is well-defined:
    struct ggml_backend_buft_comparator {
        bool operator()(const ggml_backend_buffer_type_t & lhs, const ggml_backend_buffer_type_t & rhs) const {
            return strcmp(ggml_backend_buft_name(lhs), ggml_backend_buft_name(rhs)) < 0;
        }
    };
    std::map<ggml_backend_buffer_type_t, ggml_context_ptr, ggml_backend_buft_comparator> ctx_map;

    if (n_swa > 0 && !v_trans && n_stream == 1 && !other && model.n_seq_devices() == 1 && !host_mapped) {
        n_stage = (uint32_t) ggml_env_int("LLAMA_RS_STAGE_ROWS");
    }

    const bool use_kv_mixed = !no_alloc && !other && !is_indexer && !v_trans && type_k == type_v &&
            model.n_seq_devices() == 1 && swa_type == LLAMA_SWA_TYPE_NONE &&
            model.arch == LLM_ARCH_INKLING2 && llama_kv_mixed_enabled();
    if (use_kv_mixed) {
        kv_mixed_extent   = llama_kv_mixed_hot_cells();
        kv_mixed_hot_rows = std::max<uint32_t>(1u, (uint32_t) std::min<uint64_t>((uint64_t) kv_mixed_extent*(unified ? n_seq_max : 1u), kv_size));
    }

    // create a context for each buffer type
    auto ctx_for_buft = [&](ggml_backend_buffer_type_t buft) -> ggml_context * {
        auto it = ctx_map.find(buft);
        if (it == ctx_map.end()) {
            ggml_init_params params = {
                /*.mem_size   =*/ size_t((2u*(1 + n_stream) + 2u*n_stage + (use_kv_mixed ? 1u : 0u))*n_layer*ggml_tensor_overhead()),
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };

            ggml_context * ctx = ggml_init(params);
            if (!ctx) {
                return nullptr;
            }

            ctx_map.emplace(buft, ctx);

            return ctx;
        }

        return it->second.get();
    };

    GGML_ASSERT(n_stream == 1 || n_stream == n_seq_max);

    v_heads.resize(n_stream);
    for (uint32_t s = 0; s < n_stream; ++s) {
        v_heads[s] = 0;
    }

    v_cells.resize(n_stream);
    for (uint32_t s = 0; s < n_stream; ++s) {
        v_cells[s].resize(kv_size);
    }

    // by default, all sequence ids are mapped to the 0th stream
    seq_to_stream.resize(LLAMA_MAX_SEQ, 0);

    if (n_stream > 1) {
        seq_to_stream.resize(n_stream, 0);
        for (uint32_t s = 0; s < n_stream; ++s) {
            seq_to_stream[s] = llama_seq_to_stream(s, n_seq_max, model.n_seq_devices());
        }
    }

    // [TAG_V_CACHE_VARIABLE]
    if (v_trans && hparams.is_n_embd_v_gqa_variable()) {
        LLAMA_LOG_WARN("%s: the V embeddings have different sizes across layers and FA is not enabled - padding V cache to %d\n",
                __func__, hparams.n_embd_v_gqa_max());
    }

    const bool is_mla = hparams.is_mla();

    for (uint32_t il = 0; il < n_layer; il++) {
        if (!hparams.has_kv(il)) {
            LLAMA_LOG_DEBUG("%s: layer %3d: does not have KV cache\n", __func__, il);
            continue;
        }

        if (filter && !filter(il)) {
            LLAMA_LOG_DEBUG("%s: layer %3d: filtered\n", __func__, il);
            continue;
        }

        if (share && other) {
            const int32_t il_share = share(il);

            if (il_share >= 0) {
                const auto & layer_share = other->layers[other->map_layer_ids[il_share]];

                LLAMA_LOG_WARN("%s: layer %3d: sharing with layer %d. k = %p, v = %p\n", __func__, il, il_share,
                        layer_share.k->data, layer_share.v->data);

                map_layer_ids[il] = layers.size();

                layers.push_back(layer_share);
                layers.back().il = il;

                continue;
            }
        }

        if (n_embd_head_k_all == 0) {
            n_embd_head_k_all = (int32_t) hparams.n_embd_head_k(il);
        } else if (n_embd_head_k_all > 0 && n_embd_head_k_all != (int32_t) hparams.n_embd_head_k(il)) {
            n_embd_head_k_all = -1;
        }

        if (!is_mla) {
            if (n_embd_head_v_all == 0) {
                n_embd_head_v_all = (int32_t) hparams.n_embd_head_v(il);
            } else if (n_embd_head_v_all > 0 && n_embd_head_v_all != (int32_t) hparams.n_embd_head_v(il)) {
                n_embd_head_v_all = -1;
            }
        }

        // [TAG_V_CACHE_VARIABLE]
        const uint32_t n_embd_k_gqa =            hparams.n_embd_k_gqa(il);
        const uint32_t n_embd_v_gqa = !v_trans ? hparams.n_embd_v_gqa(il) : hparams.n_embd_v_gqa_max();

        const char * dev_name = "CPU";

        ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();

        if (offload) {
            auto * dev = model.dev_layer(il);
            buft = host_mapped ? ggml_backend_dev_mapped_buffer_type(dev) : ggml_backend_dev_buffer_type(dev);
            if (!buft) { throw std::runtime_error("backend does not support mapped host KV storage"); }

            dev_name = host_mapped ? ggml_backend_buft_name(buft) : ggml_backend_dev_name(dev);
        }

        LLAMA_LOG_DEBUG("%s: layer %3d: dev = %s\n", __func__, il, dev_name);

        ggml_context * ctx = ctx_for_buft(buft);
        if (!ctx) {
            throw std::runtime_error("failed to create ggml context for kv cache");
        }

        const bool has_k = true;
        const bool has_v = !is_mla;

        ggml_tensor * k = has_k ? ggml_new_tensor_3d(ctx, type_k, n_embd_k_gqa, kv_size, n_stream) : nullptr;
        ggml_tensor * v = has_v ? ggml_new_tensor_3d(ctx, type_v, n_embd_v_gqa, kv_size, n_stream) : nullptr;

        if (model.n_seq_devices() > 1) {
            GGML_ASSERT(!unified && offload && !v_trans);
            const auto counts = llama_stream_counts(n_seq_max, model.n_seq_devices());
            if (k) { ggml_backend_meta_split_input(k, 2, counts.data(), counts.size()); }
            if (v) { ggml_backend_meta_split_input(v, 2, counts.data(), counts.size()); }
        }

        has_k && ggml_format_name(k, "cache_%sk_l%d", name_tag, il);
        has_v && ggml_format_name(v, "cache_%sv_l%d", name_tag, il);

        std::vector<ggml_tensor *> k_stream;
        std::vector<ggml_tensor *> v_stream;

        for (uint32_t s = 0; s < n_stream; ++s) {
            k_stream.push_back(has_k ? ggml_view_2d(ctx, k, n_embd_k_gqa, kv_size, k->nb[1], s*k->nb[2]) : nullptr);
            v_stream.push_back(has_v ? ggml_view_2d(ctx, v, n_embd_v_gqa, kv_size, v->nb[1], s*v->nb[2]) : nullptr);
        }

        map_layer_ids[il] = layers.size();

        ggml_tensor * hot_kv = nullptr;
        if (use_kv_mixed && has_k && has_v && n_embd_k_gqa == n_embd_v_gqa) {
            hot_kv = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, n_embd_k_gqa, 2*(kv_mixed_hot_rows + 1), n_stream);
            ggml_format_name(hot_kv, "cache_%shot_kv_l%d", name_tag, il);
        }

        layers.push_back({ il, k, v, model.layers[il].k_cache_s, model.layers[il].v_cache_s, k_stream, v_stream, nullptr, nullptr, hot_kv });

        if (n_stage > 0) {
            if (k_stage.empty()) {
                k_stage.resize(n_stage);
                v_stage.resize(n_stage);
            }
            for (uint32_t r = 0; r < n_stage; ++r) {
                // Keep cache tensor name so meta backend assigns the same axis-0 split as layer.k / layer.v.
                ggml_tensor * ks = has_k ? ggml_new_tensor_2d(ctx, type_k, n_embd_k_gqa, n_swa) : nullptr;
                ggml_tensor * vs = has_v ? ggml_new_tensor_2d(ctx, type_v, n_embd_v_gqa, n_swa) : nullptr;
                if (ks) { ggml_format_name(ks, "cache_%sk_l%d", name_tag, il); }
                if (vs) { ggml_format_name(vs, "cache_%sv_l%d", name_tag, il); }
                k_stage[r].push_back(ks);
                v_stage[r].push_back(vs);
            }
        }
    }

    if (layers.empty()) {
        n_stage = 0;
        k_stage.clear();
        v_stage.clear();
    }

    if (reuse) {
        LLAMA_LOG_DEBUG("%s: reusing layers:\n", __func__);

        for (uint32_t il = 0; il < n_layer; il++) {
            const int32_t il_reuse = reuse(il);

            if (il_reuse < 0) {
                LLAMA_LOG_DEBUG("%s: - layer %3d: no reuse\n", __func__, il);
                continue;
            }

            if (filter && !filter(il)) {
                LLAMA_LOG_DEBUG("%s: - layer %3d: filtered\n", __func__, il);
                continue;
            }

            GGML_ASSERT(map_layer_ids.find(il_reuse) != map_layer_ids.end());

            map_layer_ids[il] = map_layer_ids[il_reuse];

            LLAMA_LOG_DEBUG("%s: - layer %3d: reuse layer %d, is_swa = %d\n", __func__, il, il_reuse, hparams.is_swa(il));
        }
    }

    // allocate the tensors and initialize the buffers to avoid NaNs in the padding
    for (auto & [buft, ctx] : ctx_map) {
        ggml_backend_buffer_t buf = ctxs_bufs.emplace(std::move(ctx), buft, "kv cache");

        LLAMA_LOG_INFO("%s: %10s KV buffer size = %8.2f MiB\n", __func__, ggml_backend_buffer_name(buf), ggml_backend_buffer_get_size(buf)/1024.0/1024.0);
    }

    {
        const size_t memory_size_k = size_k_bytes();
        const size_t memory_size_v = size_v_bytes();

        LLAMA_LOG_INFO("%s: size = %7.2f MiB (%6u cells, %3d layers, %2u/%u seqs), K (%s): %7.2f MiB, V (%s): %7.2f MiB\n", __func__,
                (float)(memory_size_k + memory_size_v) / (1024.0f * 1024.0f), kv_size, (int) layers.size(), n_seq_max, n_stream,
                ggml_type_name(type_k), (float)memory_size_k / (1024.0f * 1024.0f),
                ggml_type_name(type_v), (float)memory_size_v / (1024.0f * 1024.0f));

        if (n_stage > 0 && !k_stage.empty()) {
            size_t stage_bytes = 0;
            for (const ggml_tensor * t : k_stage[0]) { stage_bytes += t ? ggml_nbytes(t) : 0; }
            for (const ggml_tensor * t : v_stage[0]) { stage_bytes += t ? ggml_nbytes(t) : 0; }
            LLAMA_LOG_INFO("%s: %u spare SWA windows for staged checkpoints (LLAMA_RS_STAGE_ROWS), %.3f MiB each\n",
                    __func__, n_stage, (float) stage_bytes / (1024.0f * 1024.0f));
        }
    }

    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        n_embd_head_k_all = other->n_embd_head_k_all;
        n_embd_head_v_all = other->n_embd_head_v_all;

        n_rot_k = other->n_rot_k;
        n_rot_v = other->n_rot_v;
    } else {
        // two knobs turn the same thing off: the environment variable, for the whole process, and
        //   the model setting, for the contexts of one model - see llama_model_set_attn_rot
        const bool attn_rot_disable = !model.attn_rot || ggml_env_switch("LLAMA_ATTN_ROT_DISABLE");
        if (attn_rot_disable) {
            LLAMA_LOG_INFO("%s: attention rotation force disabled (%s)\n", __func__,
                    model.attn_rot ? "LLAMA_ATTN_ROT_DISABLE" : "llama_model_set_attn_rot");
        }

        bool attn_rot_k =
            !attn_rot_disable &&
            n_embd_head_k_all > 0 &&
            ggml_is_quantized(type_k) &&
            hparams.n_embd_head_k() % 64 == 0;

        // always create Hadamard rotation tensors for lightning indexer keys
        if (model.lid_k_hdmd() && hparams.n_embd_head_k_full == hparams.indexer_head_size) {
            attn_rot_k = true;
        }

        const bool attn_rot_v =
            !attn_rot_disable &&
            n_embd_head_v_all > 0 &&
            ggml_is_quantized(type_v) &&
            hparams.n_embd_head_v() % 64 == 0;

        if (attn_rot_k) {
            n_rot_k = 64;

            // TODO: investigate if using the smallest rotation matrix is beneficial also for K (similar as for V)
            // ref: https://github.com/ggml-org/llama.cpp/pull/21038#issuecomment-4141323088
            while (n_embd_head_k_all % (2*n_rot_k) == 0) {
                n_rot_k *= 2;
            }
        }
        if (attn_rot_v) {
            n_rot_v = 64;
            // using smaller rotation matrices for V seems beneficial
            // ref: https://github.com/ggml-org/llama.cpp/pull/21038#issuecomment-4146397570
            //while (hparams.n_embd_head_v() % (2*n_rot_v) == 0) {
            //    n_rot_v *= 2;
            //}
        }
    }

    LLAMA_LOG_INFO("%s: n_rot_k = %u, n_embd_head_k_all = %d\n", __func__, n_rot_k, n_embd_head_k_all);
    LLAMA_LOG_INFO("%s: n_rot_v = %u, n_embd_head_k_all = %d\n", __func__, n_rot_v, n_embd_head_v_all);

    // pre-compute the haramard matrices and keep them in host memory
    if (n_rot_k || n_rot_v) {
        for (int64_t n = 64; n <= std::max(n_embd_head_k_all, n_embd_head_v_all); n *= 2) {
            attn_rot_hadamard[n] = std::vector<float>(n*n);

            ggml_init_params params = {
                /* .mem_size   = */ 1*ggml_tensor_overhead(),
                /* .mem_buffer = */ nullptr,
                /* .no_alloc   = */ true,
            };

            ggml_context_ptr ctx { ggml_init(params) };

            ggml_tensor * tmp = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, n, n);
            tmp->data = attn_rot_hadamard[n].data();

            ggml_gen_hadamard(tmp);
        }
    }

    init_rotation(name_tag);

    if (use_kv_mixed) {
        bool all_ok = !layers.empty();
        for (const auto & layer : layers) {
            if (!layer.hot_kv) {
                all_ok = false;
                break;
            }
        }
        if (all_ok) {
            kv_mixed_slot_to_cell.assign(n_stream, std::vector<uint32_t>(kv_mixed_hot_rows, UINT32_MAX));
            kv_mixed_cell_to_slot.assign(n_stream, std::vector<int32_t>(kv_size, -1));
            kv_mixed_cursor.assign(n_stream, 0);
            LLAMA_LOG_INFO("%s: mixed recency hot F16 pool allocated (%u streams, extent %u, rows/stream %u)\n",
                    __func__, n_stream, kv_mixed_extent, kv_mixed_hot_rows);
        } else {
            kv_mixed_extent   = 0;
            kv_mixed_hot_rows = 0;
        }
    }

    const char * LLAMA_KV_CACHE_DEBUG = getenv("LLAMA_KV_CACHE_DEBUG");
    debug = LLAMA_KV_CACHE_DEBUG ? atoi(LLAMA_KV_CACHE_DEBUG) : 0;
}

void llama_kv_cache::init_rotation(const char * name_tag) {
    if (!(n_rot_k || n_rot_v) || layers.empty()) {
        return;
    }

    std::map<ggml_backend_buffer_type_t, ggml_context_ptr> ctx_map;
    for (auto & layer : layers) {
        for (bool is_k : { true, false }) {
            const uint32_t n_rot = is_k ? n_rot_k : n_rot_v;
            auto * cache = is_k ? layer.k : layer.v;
            if (!n_rot || !cache) {
                continue;
            }

            const auto buft = ggml_backend_buffer_get_type(cache->buffer);
            auto & ctx = ctx_map[buft];
            if (!ctx) {
                ggml_init_params params = { 2*layers.size()*ggml_tensor_overhead(), nullptr, true };
                ctx.reset(ggml_init(params));
                if (!ctx) {
                    throw std::runtime_error("failed to create ggml context for rotation matrices");
                }
            }

            auto * rot = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, n_rot, n_rot);
            ggml_format_name(rot, "cache_%s%s_rot_l%d", name_tag, is_k ? "k" : "v", layer.il);
            (is_k ? layer.k_rot : layer.v_rot) = rot;
        }
    }

    for (auto & [buft, ctx] : ctx_map) {
        auto * buf = attn_rot_bufs.emplace(std::move(ctx), buft, "rotation matrices");
        ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        LLAMA_LOG_INFO("%s: %10s rotation buffer size = %8.2f MiB\n", __func__,
                ggml_backend_buffer_name(buf), ggml_backend_buffer_get_size(buf)/1024.0/1024.0);
    }

    if (!attn_rot_bufs.no_alloc) {
        for (const auto & layer : layers) {
            for (auto * t : { layer.k_rot, layer.v_rot }) {
                if (t) {
                    ggml_backend_tensor_set(t, attn_rot_hadamard.at(t->ne[0]).data(), 0, ggml_nbytes(t));
                }
            }
        }
    }
}

void llama_kv_cache::clear(bool data) {
    n_kv_bucket = 0;
    for (uint32_t s = 0; s < n_stream; ++s) {
        v_cells[s].reset();
        v_heads[s] = 0;
    }

    // whatever the pools stamp now names dropped cells
    kv_mixed_reset();

    if (data) {
        for (auto & [_, buf] : ctxs_bufs) {
            ggml_backend_buffer_clear(buf.get(), 0);
        }
    }
}

bool llama_kv_cache::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return true;
    }

    // TODO: fix incosistent handling of `seq_id < 0` and `seq_id == -1` in the codebase [TAG_LLAMA_SEQ_ID_NEG]
    GGML_ASSERT(seq_id == -1 || (seq_id >= 0 && (size_t) seq_id < seq_to_stream.size()));

    if (p0 < 0) {
        p0 = 0;
    }

    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    if (seq_id >= 0) {
        const uint32_t strm = seq_to_stream[seq_id];
        auto & cells = v_cells[strm];
        auto & head  = v_heads[strm];

        if (has_kv_mixed() && strm == 0 && !kv_mixed_dirty_all) {
            const auto & sp = cells.seq_pos_get(seq_id);
            for (auto it = sp.lower_bound({ p0, 0 }); it != sp.end() && it->first < p1; ++it) {
                if (kv_mixed_cell_to_slot[0][it->second] >= 0) {
                    kv_mixed_mark_dirty(0, it->second);
                }
            }
        }

        // the sequence's position index finds the cells, so a speculative rollback's few cells cost a lookup each, not a
        // walk of the cache (this replaces the fork's walk from the head, 6ea8df2 and 3db1d577d)
        const uint32_t new_head = cells.seq_rm(seq_id, p0, p1);

        // If we freed up a slot, set head to it so searching can start there.
        if (new_head != cells.size() && new_head < head) {
            head = new_head;
        }
    } else {
        if (has_kv_mixed()) {
            kv_mixed_dirty.clear();
            kv_mixed_dirty_all = true;
        }
        // match any sequence
        for (uint32_t s = 0; s < n_stream; ++s) {
            auto & cells = v_cells[s];
            auto & head  = v_heads[s];

            uint32_t new_head = cells.size();

            for (uint32_t i = 0; i < cells.size(); ++i) {
                if (!cells.pos_in(i, p0, p1)) {
                    continue;
                }

                cells.rm(i);

                if (new_head == cells.size()) {
                    new_head = i;
                }
            }

            // If we freed up a slot, set head to it so searching can start there.
            if (new_head != cells.size() && new_head < head) {
                head = new_head;
            }
        }
    }

    return true;
}

void llama_kv_cache::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return;
    }

    GGML_ASSERT(seq_id_src >= 0 && (size_t) seq_id_src < seq_to_stream.size());
    GGML_ASSERT(seq_id_dst >= 0 && (size_t) seq_id_dst < seq_to_stream.size());

    // A dense prefix can have raw keys but no pooled keys yet. Whether cells
    // are shared or copied, its child must rebuild those keys before scoring.
    kpool_stale.inherit(seq_id_src, seq_id_dst);

    const auto s0 = seq_to_stream[seq_id_src];
    const auto s1 = seq_to_stream[seq_id_dst];

    if (has_kv_mixed() && (s0 == 0 || s1 == 0)) {
        kv_mixed_dirty.clear();
        kv_mixed_dirty_all = true;
    }

    if (s0 == s1) {
        // since both sequences are in the same stream, no data copy is necessary
        // we just have to update the cells meta data

        auto & cells = v_cells[s0];

        if (seq_id_src == seq_id_dst) {
            return;
        }

        if (p0 < 0) {
            p0 = 0;
        }

        if (p1 < 0) {
            p1 = std::numeric_limits<llama_pos>::max();
        }

        for (uint32_t i = 0; i < cells.size(); ++i) {
            if (!cells.pos_in(i, p0, p1)) {
                continue;
            }

            if (cells.seq_has(i, seq_id_src)) {
                cells.seq_add(i, seq_id_dst);
            }
        }

        return;
    }

    // cross-stream sequence copies require to copy the actual buffer data

    bool is_full = true;

    if (p0 > 0 && p0 + 1 < (int) get_size()) {
        is_full = false;
    }

    if (p1 > 0 && p1 + 1 < (int) get_size()) {
        is_full = false;
    }

    GGML_ASSERT(is_full && "seq_cp() is only supported for full KV buffers");

    // enqueue the copy operation - the buffer copy will be performed during the next update
    sc_info.ssrc.push_back(s0);
    sc_info.sdst.push_back(s1);

    v_cells[s1].reset();
    for (uint32_t i = 0; i < v_cells[s0].size(); ++i) {
        if (v_cells[s0].seq_has(i, seq_id_src)) {
            llama_pos pos   = v_cells[s0].pos_get(i);
            llama_pos shift = v_cells[s0].get_shift(i);

            llama_kv_cell_ext ext = v_cells[s0].ext_get(i);

            if (shift != 0) {
                pos -= shift;
                assert(pos >= 0);
            }

            v_cells[s1].pos_set(i, pos);
            v_cells[s1].seq_add(i, seq_id_dst);

            if (shift != 0) {
                v_cells[s1].pos_add(i, shift);
            }

            v_cells[s1].ext_set(i, ext);
        }
    }

    v_heads[s1] = v_heads[s0];

    //for (uint32_t s = 0; s < n_stream; ++s) {
    //    LLAMA_LOG_WARN("%s: seq %d: min = %d, max = %d\n", __func__, s, v_cells[s].seq_pos_min(s), v_cells[s].seq_pos_max(s));
    //}
}

void llama_kv_cache::seq_keep(llama_seq_id seq_id) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return;
    }

    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());

    if (has_kv_mixed() && seq_to_stream[seq_id] == 0) {
        kv_mixed_dirty.clear();
        kv_mixed_dirty_all = true;
    }

    auto & cells = v_cells[seq_to_stream[seq_id]];
    auto & head  = v_heads[seq_to_stream[seq_id]];

    uint32_t new_head = cells.size();

    for (uint32_t i = 0; i < cells.size(); ++i) {
        if (cells.seq_keep(i, seq_id)) {
            if (new_head == cells.size()) {
                new_head = i;
            }
        }
    }

    // If we freed up a slot, set head to it so searching can start there.
    if (new_head != cells.size() && new_head < head) {
        head = new_head;
    }
}

void llama_kv_cache::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return;
    }

    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());
    GGML_ASSERT(hparams.n_pos_per_embd() == 1 && "seq_add() is only supported for n_pos_per_embd() == 1");

    auto & cells = v_cells[seq_to_stream[seq_id]];
    auto & head  = v_heads[seq_to_stream[seq_id]];

    if (shift == 0) {
        return;
    }

    if (has_kv_mixed() && seq_to_stream[seq_id] == 0) {
        kv_mixed_dirty.clear();
        kv_mixed_dirty_all = true;
    }

    uint32_t new_head = cells.size();

    if (p0 < 0) {
        p0 = 0;
    }

    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    // If there is no range then return early to avoid looping over all cells.
    if (p0 == p1) {
        return;
    }

    for (uint32_t i = 0; i < cells.size(); ++i) {
        if (!cells.pos_in(i, p0, p1)) {
            continue;
        }

        if (cells.seq_has(i, seq_id)) {
            if (cells.pos_add(i, shift)) {
                if (new_head == cells.size()) {
                    new_head = i;
                }
            }
        }
    }

    // If we freed up a slot, set head to it so searching can start there.
    // Otherwise we just start the next search from the beginning.
    head = new_head != cells.size() ? new_head : 0;
}

void llama_kv_cache::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return;
    }

    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());
    GGML_ASSERT(hparams.n_pos_per_embd() == 1 && "seq_div() is only supported for n_pos_per_embd() == 1");

    auto & cells = v_cells[seq_to_stream[seq_id]];

    if (d == 1) {
        return;
    }

    if (has_kv_mixed() && seq_to_stream[seq_id] == 0) {
        kv_mixed_dirty.clear();
        kv_mixed_dirty_all = true;
    }

    if (p0 < 0) {
        p0 = 0;
    }

    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    // If there is no range then return early to avoid looping over the cache.
    if (p0 == p1) {
        return;
    }

    for (uint32_t i = 0; i < cells.size(); ++i) {
        if (!cells.pos_in(i, p0, p1)) {
            continue;
        }

        if (cells.seq_has(i, seq_id)) {
            cells.pos_div(i, d);
        }
    }
}

llama_pos llama_kv_cache::seq_pos_min(llama_seq_id seq_id) const {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return other->seq_pos_min(seq_id);
    }

    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());

    const auto & cells = v_cells[seq_to_stream[seq_id]];

    return cells.seq_pos_min(seq_id);
}

llama_pos llama_kv_cache::seq_pos_max(llama_seq_id seq_id) const {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return other->seq_pos_max(seq_id);
    }

    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());

    const auto & cells = v_cells[seq_to_stream[seq_id]];

    return cells.seq_pos_max(seq_id);
}

std::map<ggml_backend_buffer_type_t, size_t> llama_kv_cache::memory_breakdown() const {
    auto ret = ctxs_bufs.memory_breakdown();

    for (const auto & [buft, size] : attn_rot_bufs.memory_breakdown()) {
        ret[buft] += size;
    }

    return ret;
}

llama_memory_context_ptr llama_kv_cache::init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) {
    do {
        balloc.split_reset();

        std::vector<llama_ubatch> ubatches;
        while (true) {
            auto ubatch = n_stream == 1 ? balloc.split_simple(n_ubatch) : embd_all ? balloc.split_seq(n_ubatch) : balloc.split_equal(n_ubatch, true, 0);

            if (ubatch.n_tokens == 0) {
                break;
            }

            ubatches.push_back(std::move(ubatch)); // NOLINT
        }

        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            // failed to find a suitable split
            break;
        }

        auto sinfos = prepare(ubatches);
        if (sinfos.empty()) {
            break;
        }

        return std::make_unique<llama_kv_cache_context>(
                this, std::move(sinfos), std::move(ubatches));
    } while (false);

    return std::make_unique<llama_kv_cache_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
}

llama_memory_context_ptr llama_kv_cache::init_full() {
    return std::make_unique<llama_kv_cache_context>(this);
}

llama_memory_context_ptr llama_kv_cache::init_update(llama_context * lctx, bool optimize) {
    GGML_UNUSED(optimize);

    bool do_shift = get_has_shift();

    return std::make_unique<llama_kv_cache_context>(this, lctx, do_shift, std::move(sc_info));
}

llama_kv_cache::slot_info_vec_t llama_kv_cache::prepare(const std::vector<llama_ubatch> & ubatches) {
    llama_kv_cache::slot_info_vec_t res;

    // Dry-run placement restores every cell before returning, so keep cell logs clean.
    struct log_pause_all {
        llama_kv_cells_vec & cells;
        explicit log_pause_all(llama_kv_cells_vec & v) : cells(v) {
            for (auto & c : cells) {
                c.pause_logging_push();
            }
        }
        ~log_pause_all() {
            for (auto & c : cells) {
                c.pause_logging_pop();
            }
        }
    } pause_logs(v_cells);

    struct state_t {
        slot_info sinfo; // slot info for the ubatch

        std::vector<uint32_t> v_heads_old; // old positions of the heads, before placing the ubatch

        std::vector<llama_kv_cells> v_cells; // copy of the old cells, before placing the ubatch
    };

    // remember the old state of the cells so we can restore it in the end
    std::vector<state_t> states;

    bool success = true;

    for (const auto & ubatch : ubatches) {
        // only find a suitable slot for the ubatch. don't modify the cells yet
        const auto sinfo_new = find_slot(ubatch, false);
        if (sinfo_new.empty()) {
            success = false;
            break;
        }

        // remember the position that we found
        res.push_back(sinfo_new);

        // store the old state of the cells in the recovery stack
        {
            state_t state = { sinfo_new, v_heads, {} };

            for (uint32_t s = 0; s < sinfo_new.n_stream(); ++s) {
                auto & cells = v_cells[sinfo_new.strm[s]];

                state.v_cells.push_back(cells.cp(sinfo_new.idxs[s]));
            }

            states.push_back(std::move(state));
        }

        // now emplace the ubatch
        // without the purge: it removes cells outside the slot, which the restore below would not bring back,
        // and the earlier ubatches of this call still read them when they run
        apply_ubatch(sinfo_new, ubatch, false);
    }

    GGML_ASSERT(!states.empty() || !success);

    // iterate backwards and restore the cells to their original state
    for (auto it = states.rbegin(); it != states.rend(); ++it) {
        const auto & sinfo = it->sinfo;

        for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
            auto & cells = v_cells[sinfo.strm[s]];
            auto & head  = v_heads[sinfo.strm[s]];

            cells.set(sinfo.idxs[s], it->v_cells[s]);
            head = it->v_heads_old[sinfo.strm[s]];
        }
    }

    if (!success) {
        return {};
    }

    return res;
}

bool llama_kv_cache::update(llama_context * lctx, bool do_shift, const stream_copy_info & sc_info) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return true;
    }

    bool updated = false;

    auto * sched = lctx->get_sched();

    if (!sc_info.empty()) {
        assert(n_stream > 1 && "stream copy should never happen with a single stream");

        llama_synchronize(lctx);

        const size_t n_copy = sc_info.ssrc.size();

        for (size_t i = 0; i < n_copy; ++i) {
            const auto ssrc = sc_info.ssrc[i];
            const auto sdst = sc_info.sdst[i];

            assert(ssrc < n_stream);
            assert(sdst < n_stream);

            LLAMA_LOG_DEBUG("%s: copying KV buffer: stream %d to stream %d\n", __func__, ssrc, sdst);

            assert(ssrc != sdst);

            for (uint32_t il = 0; il < layers.size(); ++il) {
                const auto & layer = layers[il];

                ggml_backend_tensor_copy(layer.k_stream[ssrc], layer.k_stream[sdst]);

                if (layer.v_stream[ssrc]) {
                    ggml_backend_tensor_copy(layer.v_stream[ssrc], layer.v_stream[sdst]);
                }
            }
            kv_mixed_reset_stream(sdst);
        }
    }

    if (do_shift) {
        if (!get_can_shift()) {
            GGML_ABORT("The current KV cache / model configuration does not support K-shift");
        }

        LLAMA_LOG_DEBUG("%s: applying K-shift\n", __func__);

        // apply K-shift if needed
        if (hparams.rope_type != LLAMA_ROPE_TYPE_NONE &&
                std::any_of(layers.begin(), layers.end(), [&](const auto & layer) {
                    return hparams.has_rope(layer.il);
                })) {
            ggml_backend_sched_reset(sched);

            auto * res = lctx->get_gf_res_reserve();

            res->reset();

            auto * gf = build_graph_shift(res, lctx);
            if (!ggml_backend_sched_alloc_graph(sched, gf)) {
                LLAMA_LOG_ERROR("%s: failed to allocate compute graph for K-shift\n", __func__);
                return updated;
            }

            res->set_inputs(nullptr);

            if (lctx->graph_compute(gf, false) != GGML_STATUS_SUCCESS) {
                LLAMA_LOG_ERROR("%s: failed to compute K-shift\n", __func__);
                return updated;
            }

            updated = true;
        }

        for (uint32_t s = 0; s < n_stream; ++s) {
            auto & cells = v_cells[s];

            cells.reset_shift();
        }
    }

    return updated;
}

llama_kv_cache::slot_info llama_kv_cache::find_slot(const llama_ubatch & ubatch, bool cont) const {

    if (debug > 0) {
        for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
            const auto seq_id = ubatch.seq_id_unq[s];
            const auto stream_id = seq_to_stream[seq_id];
            const auto & cells = v_cells[stream_id];
            const uint32_t head_cur = v_heads[stream_id];

            LLAMA_LOG_DEBUG("%s: stream[%d], n = %5d, used = %5d, head = %5d, size = %5d, n_swa = %5d\n",
                    __func__, stream_id, cells.used_max_p1(), cells.get_used(), head_cur, get_size(), n_swa);

            if ((debug == 2 && n_swa > 0) || debug > 2) {
                std::string ss;
                for (uint32_t i = 0; i < cells.size(); ++i) {
                    if (cells.is_empty(i)) {
                        ss += '.';
                    } else {
                        assert(cells.seq_count(i) >= 1);

                        if (cells.seq_count(i) == 1) {
                            ss += std::to_string(cells.seq_get(i));
                        } else {
                            ss += 'M';
                        }
                    }
                    if (i%256 == 255) {
                        ss += " *";
                        ss += '\n';
                    }
                }
                LLAMA_LOG_DEBUG("\n%s\n", ss.c_str());
            }

            if ((debug == 2 && n_swa > 0) || debug > 2) {
                std::string ss;
                for (uint32_t i = 0; i < cells.size(); ++i) {
                    std::string cur;
                    if (cells.is_empty(i)) {
                        cur = '.';
                    } else {
                        cur = std::to_string(cells.pos_get(i));
                    }
                    const int n = cur.size();
                    for (int j = 0; j < 5 - n; ++j) {
                        cur += ' ';
                    }
                    ss += cur;
                    if (i%256 == 255) {
                        ss += " *";
                    }
                    if (i%64 == 63) {
                        ss += '\n';
                    }
                }
                LLAMA_LOG_DEBUG("\n%s\n", ss.c_str());
            }

            for (int s = 0; s < LLAMA_MAX_SEQ; ++s) {
                if (cells.seq_pos_min(s) < 0) {
                    continue;
                }

                LLAMA_LOG_DEBUG("%s: stream[%d] min[%d] = %5d, max[%d] = %5d\n", __func__, stream_id, s, cells.seq_pos_min(s), s, cells.seq_pos_max(s));
            }
        }
    }

    uint32_t n_tokens = ubatch.n_tokens;
    uint32_t n_seqs   = 1;

    if (n_stream > 1) {
        GGML_ASSERT(n_tokens % ubatch.n_seqs_unq == 0);

        n_seqs   = ubatch.n_seqs_unq;
        n_tokens = n_tokens / n_seqs;
    }

    slot_info res = {
        /*.s0   =*/ LLAMA_MAX_SEQ,
        /*.s1   =*/ 0,
        /*.strm =*/ { },
        /*.idxs =*/ { },
    };

    res.resize(n_seqs);

    for (uint32_t s = 0; s < n_seqs; ++s) {
        const auto seq_id = ubatch.seq_id_unq[s];

        if (n_stream > 1) {
            GGML_ASSERT(ubatch.n_seq_id[s*n_tokens]    == 1);
            GGML_ASSERT(ubatch.seq_id  [s*n_tokens][0] == seq_id);
        }

        res.s0 = std::min<uint32_t>(res.s0, seq_to_stream[seq_id]);
        res.s1 = std::max<uint32_t>(res.s1, seq_to_stream[seq_id]);

        res.strm[s] = seq_to_stream[seq_id];
        res.idxs[s].reserve(n_tokens);

        const auto & cells = v_cells[seq_to_stream[seq_id]];

        uint32_t head_cur = v_heads[seq_to_stream[seq_id]];

        // if we have enough unused cells before the current head ->
        //   better to start searching from the beginning of the cache, hoping to fill it
        if (head_cur > cells.get_used() + 2*n_tokens) {
            head_cur = 0;
        }

        if (n_tokens > cells.size()) {
            LLAMA_LOG_ERROR("%s: n_tokens = %d > size = %u\n", __func__, n_tokens, cells.size());
            return { };
        }

        uint32_t n_tested = 0;

        // for continuous slots, we test that all tokens in the ubatch fit, starting from the current head
        // for non-continuous slots, we test the tokens one by one
        const uint32_t n_test = cont ? n_tokens : 1;

        while (true) {
            if (head_cur + n_test > cells.size()) {
                n_tested += cells.size() - head_cur;
                head_cur = 0;
                continue;
            }

            for (uint32_t i = 0; i < n_test; i++) {
                const auto idx = head_cur;

                head_cur++;
                n_tested++;

                bool can_use = cells.is_empty(idx);
                if (!can_use && swa_type != LLAMA_SWA_TYPE_NONE) {
                    // A shared row can be reused once it is outside every owner's window.
                    const llama_pos pos = cells.pos_get(idx);
                    can_use = true;
                    for (int owner = 0, remaining = cells.seq_count(idx); remaining > 0; ++owner) {
                        if (!cells.seq_has(idx, owner)) {
                            continue;
                        }
                        --remaining;
                        if (!llama_hparams::is_masked_swa(n_swa, swa_type, pos, cells.seq_pos_max(owner) + 1)) {
                            can_use = false;
                            break;
                        }
                    }
                }

                if (can_use) {
                    res.idxs[s].push_back(idx);
                } else {
                    if (cont) {
                        break;
                    }
                }
            }

            if (res.idxs[s].size() == n_tokens) {
                break;
            }

            if (cont) {
                res.idxs[s].clear();
            }

            if (n_tested >= cells.size()) {
                //LLAMA_LOG_ERROR("%s: failed to find a slot for %d tokens\n", __func__, n_tokens);
                return { };
            }
        }

        // we didn't find a suitable slot - return empty result
        if (res.idxs[s].size() < n_tokens) {
            return { };
        }
    }

    assert(res.s1 >= res.s0);

    return res;
}

bool llama_kv_cache::kv_mixed_is_near(uint32_t strm, uint32_t cell) const {
    const auto & cells = v_cells[strm];
    if (cells.is_empty(cell)) {
        return false;
    }
    const llama_pos p = cells.pos_get(cell);
    for (int seq = 0, remaining = cells.seq_count(cell); remaining > 0; ++seq) {
        if (!cells.seq_has(cell, seq)) {
            continue;
        }
        --remaining;
        const llama_pos tip = cells.seq_pos_max(seq);
        if (tip >= p && tip - p < (llama_pos) kv_mixed_extent) {
            return true;
        }
    }
    return false;
}

void llama_kv_cache::kv_mixed_mark_dirty(uint32_t strm, uint32_t cell) const {
    if (strm != 0 || kv_mixed_dirty_all) {
        return;
    }
    if (kv_mixed_dirty.size() >= 4096) {
        kv_mixed_dirty.clear();
        kv_mixed_dirty_all = true;
        return;
    }
    kv_mixed_dirty.push_back(cell);
}

void llama_kv_cache::kv_mixed_unlink(uint32_t strm, const uint32_t * idxs, size_t n) {
    if (!has_kv_mixed() || strm >= kv_mixed_cell_to_slot.size()) {
        return;
    }
    for (size_t i = 0; i < n; ++i) {
        const uint32_t idx = idxs[i];
        const int32_t old_slot = kv_mixed_cell_to_slot[strm][idx];
        if (old_slot >= 0) {
            kv_mixed_slot_to_cell[strm][old_slot] = UINT32_MAX;
            kv_mixed_cell_to_slot[strm][idx]      = -1;
            kv_mixed_mark_dirty(strm, idx);
        }
    }
}

void llama_kv_cache::kv_mixed_reset_stream(uint32_t strm) {
    if (!has_kv_mixed() || strm >= kv_mixed_slot_to_cell.size()) {
        return;
    }
    std::fill(kv_mixed_slot_to_cell[strm].begin(), kv_mixed_slot_to_cell[strm].end(), UINT32_MAX);
    std::fill(kv_mixed_cell_to_slot[strm].begin(), kv_mixed_cell_to_slot[strm].end(), -1);
    kv_mixed_cursor[strm] = 0;
    if (strm == 0) {
        kv_mixed_dirty.clear();
        kv_mixed_dirty_all = true;
    }
}

void llama_kv_cache::kv_mixed_reset() {
    if (!has_kv_mixed()) {
        return;
    }
    for (uint32_t s = 0; s < n_stream; ++s) {
        kv_mixed_reset_stream(s);
    }
    kv_mixed_hot_k_idxs.clear();
    kv_mixed_hot_v_idxs.clear();
}

void llama_kv_cache::kv_mixed_apply(const slot_info & sinfo, const llama_ubatch & ubatch) {
    if (!has_kv_mixed()) {
        return;
    }
    const uint32_t n_tokens = ubatch.n_tokens;
    const uint32_t kv_size  = get_size();
    GGML_ASSERT(n_tokens == sinfo.n_stream()*sinfo.size());

    kv_mixed_hot_k_idxs.resize(n_tokens);
    kv_mixed_hot_v_idxs.resize(n_tokens);

    // Unlink any old hot stamps for overwritten cells and default every token to the unread trash slot.
    for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
        const uint32_t strm = sinfo.strm[s];
        const int64_t row_offs = (int64_t) strm * 2 * (int64_t) (kv_mixed_hot_rows + 1);
        const int64_t trash_k  = row_offs + 2 * (int64_t) kv_mixed_hot_rows;
        const int64_t trash_v  = trash_k + 1;

        kv_mixed_unlink(strm, sinfo.idxs[s].data(), sinfo.size());
        for (uint32_t ii = 0; ii < sinfo.size(); ++ii) {
            const uint32_t i = s*sinfo.size() + ii;
            kv_mixed_hot_k_idxs[i] = trash_k;
            kv_mixed_hot_v_idxs[i] = trash_v;
        }
    }

    // Assign unique live slots newest-first so large prefills keep the newest tail without write races.
    std::vector<uint32_t> assigned(n_stream, 0);
    for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
        const uint32_t strm = sinfo.strm[s];
        const int64_t row_offs = (int64_t) strm * 2 * (int64_t) (kv_mixed_hot_rows + 1);

        for (uint32_t ii = sinfo.size(); ii > 0; --ii) {
            const uint32_t i   = s*sinfo.size() + (ii - 1);
            const uint32_t idx = sinfo.idxs[s][ii - 1];
            if (assigned[strm] >= kv_mixed_hot_rows || kv_mixed_cell_to_slot[strm][idx] >= 0 || !kv_mixed_is_near(strm, idx)) {
                continue;
            }
            const uint32_t slot = (uint32_t) (kv_mixed_cursor[strm]++ % kv_mixed_hot_rows);
            const uint32_t old_cell = kv_mixed_slot_to_cell[strm][slot];
            if (old_cell != UINT32_MAX && old_cell < kv_size && kv_mixed_cell_to_slot[strm][old_cell] == (int32_t) slot) {
                kv_mixed_cell_to_slot[strm][old_cell] = -1;
                kv_mixed_mark_dirty(strm, old_cell);
            }
            kv_mixed_slot_to_cell[strm][slot] = idx;
            kv_mixed_cell_to_slot[strm][idx]  = (int32_t) slot;
            kv_mixed_mark_dirty(strm, idx);
            kv_mixed_hot_k_idxs[i] = row_offs + 2 * (int64_t) slot;
            kv_mixed_hot_v_idxs[i] = row_offs + 2 * (int64_t) slot + 1;
            assigned[strm]++;
        }
    }
}

void llama_kv_cache::apply_ubatch(const slot_info & sinfo, const llama_ubatch & ubatch, bool purge) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return;
    }

    // keep track of the max sequence position that we would overwrite with this ubatch
    // for non-SWA cache, this would be always empty
    llama_seq_id seq_pos_max_rm[LLAMA_MAX_SEQ];
    for (uint32_t s = 0; s < LLAMA_MAX_SEQ; ++s) {
        seq_pos_max_rm[s] = -1;
    }

    assert(ubatch.n_tokens == sinfo.n_stream()*sinfo.size());

    for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
        for (uint32_t ii = 0; ii < sinfo.size(); ++ii) {
            const uint32_t i = s*sinfo.size() + ii;

            auto & cells = v_cells[sinfo.strm[s]];

            const auto idx = sinfo.idxs[s][ii];

            if (!cells.is_empty(idx)) {
                const llama_pos pos = cells.pos_get(idx);
                for (int owner = 0, remaining = cells.seq_count(idx); remaining > 0; ++owner) {
                    if (cells.seq_has(idx, owner)) {
                        --remaining;
                        seq_pos_max_rm[owner] = std::max(seq_pos_max_rm[owner], pos);
                    }
                }

                cells.rm(idx);
            }

            cells.pos_set(idx, ubatch.pos[i]);

            if (ubatch.is_pos_2d() || ubatch.token || hparams.ple_n_heads > 0) {
                llama_kv_cell_ext ext;

                if (ubatch.is_pos_2d()) {
                    ext.x = ubatch.pos[i + ubatch.n_tokens*2];
                    ext.y = ubatch.pos[i + ubatch.n_tokens];
                }

                const bool is_embd = !ubatch.token || (ubatch.is_mixed() && ubatch.type[i]);

                if (!is_embd) {
                    ext.tok = ubatch.token[i];
                } else if (hparams.ple_n_heads > 0) {
                    // embd batch (multimodal input) has no token ids, need to pad it with the correct ID for PLE layers
                    // TODO @ngxson : check if we can do the same as gemma 3n / gemma 4
                    ext.tok = hparams.ple_embd_token;
                }

                cells.ext_set(idx, ext);
            }

            for (int32_t s = 0; s < ubatch.n_seq_id[i]; s++) {
                cells.seq_add(idx, ubatch.seq_id[i][s]);
            }
        }
    }

    // note: we want to preserve the invariant that all positions between [pos_min, pos_max] for each sequence
    //       will be present in the cache. so we have to purge any position which is less than those we would overwrite
    //       ref: https://github.com/ggml-org/llama.cpp/pull/13746#issuecomment-2916057092
    for (uint32_t s = 0; purge && s < LLAMA_MAX_SEQ; ++s) {
        if (seq_pos_max_rm[s] == -1) {
            continue;
        }

        GGML_ASSERT(s < seq_to_stream.size());

        auto & cells = v_cells[seq_to_stream[s]];

        if (cells.seq_pos_min(s) <= seq_pos_max_rm[s]) {
            LLAMA_LOG_DEBUG("%s: purging positions [%d, %d] of sequence %d from KV cache\n",
                    __func__, cells.seq_pos_min(s), seq_pos_max_rm[s], s);

            seq_rm(s, cells.seq_pos_min(s), seq_pos_max_rm[s] + 1);
        }
    }

    // move the head at the end of the slot
    for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
        auto & head = v_heads[sinfo.strm[s]];

        head = sinfo.idxs[s].back() + 1;
    }
}

bool llama_kv_cache::get_can_shift() const {
    // Step35 uses per-layer RoPE dims; K-shift assumes a single global n_rot.
    if (model.arch == LLM_ARCH_STEP35) {
        return false;
    }
    if (hparams.n_pos_per_embd() > 1) {
        return false;
    }
    return true;
}

uint32_t llama_kv_cache::get_size() const {
    const auto & cells = v_cells[seq_to_stream[0]];

    return cells.size();
}

uint32_t llama_kv_cache::get_n_seq_max() const {
    return n_seq_max;
}

uint32_t llama_kv_cache::get_n_stream() const {
    return n_stream;
}

bool llama_kv_cache::get_has_shift() const {
    bool result = false;

    for (uint32_t s = 0; s < n_stream; ++s) {
        result |= v_cells[s].get_has_shift();
    }

    return result;
}

bool llama_kpool_stale::shift_regroups(uint32_t kpool, llama_pos p0, llama_pos p1, llama_pos shift) {
    GGML_ASSERT(kpool > 0);

    const llama_pos r = (llama_pos) kpool;

    const bool whole_pools = shift % r == 0 &&
                             (p0 <= 0 || p0 % r == 0) &&
                             (p1 <  0 || p1 % r == 0);

    return !whole_pools;
}

void llama_kpool_stale::mark(llama_seq_id seq_id) {
    if (seq_id < 0) {
        seqs.set();
        return;
    }

    GGML_ASSERT(seq_id < LLAMA_MAX_SEQ);
    seqs.set(seq_id);
}

void llama_kpool_stale::inherit(llama_seq_id seq_id_src, llama_seq_id seq_id_dst) {
    if (seqs.test(seq_id_src)) {
        seqs.set(seq_id_dst);
    }
}

bool llama_kpool_stale::any(const llama_ubatch & ubatch) const {
    for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
        if (seqs.test(ubatch.seq_id_unq[s])) {
            return true;
        }
    }

    return false;
}

void llama_kpool_stale::clear(const llama_ubatch & ubatch) {
    for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
        seqs.reset(ubatch.seq_id_unq[s]);
    }
}

void llama_kv_cache::set_kpool_dirty(llama_seq_id seq_id) const {
    kpool_stale.mark(seq_id);
}

bool llama_kv_cache::get_kpool_dirty(const llama_ubatch & ubatch) const {
    return kpool_stale.any(ubatch);
}

void llama_kv_cache::clear_kpool_dirty(const llama_ubatch & ubatch) const {
    kpool_stale.clear(ubatch);
}

llama_kpool_views & llama_kv_cache::get_kpool_views() const {
    if (!kpool_views) {
        kpool_views = std::make_shared<llama_kpool_views>();
    }

    return *kpool_views;
}

ggml_type llama_kv_cache::type_k() const {
    return layers[0].k->type;
}

ggml_type llama_kv_cache::type_v() const {
    return layers[0].v->type;
}

std::vector<uint32_t> llama_kv_cache::get_layer_ids() const {
    std::vector<uint32_t> res;
    res.reserve(layers.size());

    for (const auto & layer : layers) {
        res.push_back(layer.il);
    }

    return res;
}

ggml_tensor * llama_kv_cache::get_k_storage(int32_t il) const {
    const int32_t ikv = map_layer_ids.at(il);

    return layers[ikv].k;
}

const llama_kv_cells & llama_kv_cache::get_cells(llama_seq_id seq_id) const {
    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());

    return v_cells[seq_to_stream[seq_id]];
}

uint32_t llama_kv_cache::get_stream(llama_seq_id seq_id) const {
    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());

    return seq_to_stream[seq_id];
}

uint32_t llama_kv_cache::get_n_pad_max() const {
    return std::max({ 256u, n_pad_prefill, n_pad_shared });
}

bool llama_kv_cache::get_pad_shared_wide() const {
    return pad_shared_wide;
}

uint32_t llama_kv_cache::get_n_kv(const slot_info & sinfo, const llama_ubatch * ubatch, uint32_t * n_pad_used,
        uint32_t * n_kv_pool) const {
    const auto padded = [&](uint32_t pad) {
        uint32_t result = 0;

        for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
            const auto & cells = v_cells[sinfo.strm[s]];

            result = std::max(std::min(cells.size(), std::max(pad, GGML_PAD(cells.used_max_p1(), pad))), result);
        }

        return result;
    };

    // pad the n_kv value so that the graph remains constant across batches and can be reused
    // note: this also helps some backends with performance (f.ex https://github.com/ggml-org/llama.cpp/pull/16812#issuecomment-3455112220)
    uint32_t pad    = 256u;
    uint32_t result = padded(pad);

    // A prompt step of at least 256 rows whose pooled indexer scores at 256 (llama_kpool_needs_scoring) is padded to
    // n_pad_prefill: it scores there too, so whether it scores is unchanged, and its sparse attention reads only the cells
    // its top-k picks, so the empty cells it adds are never read. A step of any length in a pool that sequences share is
    // padded to at least n_pad_shared on the same terms. One sequence's decode step, and a step that reads densely in a
    // pool of its own, keep 256
    // the most cells a pool reads densely; a cache with no indexer reads densely at every length
    const uint32_t n_dense = hparams.indexer_kpool > 0 ? hparams.indexer_top_k + hparams.indexer_kpool - 1 : UINT32_MAX;
    const bool scores = n_pad_prefill + n_pad_shared > 0 && result > n_dense;
    const uint32_t pad_scoring = std::max(ubatch && ubatch->n_tokens >= 256 ? n_pad_prefill : 0, n_pad_shared);
    if (scores && pad_scoring > pad) {
        pad    = pad_scoring;
        result = padded(pad);
    }

    if (pad_dense_inkling && sinfo.n_stream() > 0) {
        const uint32_t used = v_cells[sinfo.strm[0]].used_max_p1();
        const uint32_t capacity = get_size();
        if (keep_span) {
            // Hold the read span until it fills or the pool's high-water falls to a quarter.
            if (used > n_kv_bucket || 4ull*used <= n_kv_bucket) {
                uint64_t span = 512;
                while (span < used) {
                    span *= 2;
                }
                n_kv_bucket = std::min<uint64_t>(span, capacity);
            }
        } else {
            const auto bucket = [&](uint64_t span) {
                uint64_t step = 512;
                while (step < span && step < n_pad_shared) {
                    step *= 2;
                }
                if (span > 4ull*n_pad_shared) {
                    step = 2ull*n_pad_shared;
                }
                return std::min<uint64_t>(capacity, std::max(step, GGML_PAD(span, step)));
            };

            // Grow at the bucket edge, but require a drop past the edge and the deadband before shrinking.
            const uint32_t next = bucket(used);
            if (next >= n_kv_bucket || bucket(uint64_t(used) + std::min(used/4, n_pad_shared/4)) < n_kv_bucket) {
                n_kv_bucket = next;
            }
        }
        result = n_kv_bucket;
    } else if (!scores && pad_dense_shared) {
        // The next power of two that holds the cells and still reads densely.
        while (pad < result && 2*pad <= n_pad_shared && padded(2*pad) <= n_dense) {
            pad *= 2;
        }
        result = padded(pad);
    }

    if (!pad_dense_inkling && model.arch == LLM_ARCH_INKLING2 && n_swa == 0 && n_stream == 1 && !v_trans && ubatch &&
            ubatch->n_tokens >= 256 && result >= 32768 &&
            !layers.empty() && layers[0].k && layers[0].v && type_k() == type_v() &&
            (type_k() == GGML_TYPE_F16 || type_k() == GGML_TYPE_Q8_0 || type_k() == GGML_TYPE_Q4_0 || type_k() == GGML_TYPE_F8_E4M3) &&
            ggml_env_switch("LLAMA_INKLING2_BUCKET_PREFILL_KV")) {
        // An explicitly configured long prefill keeps its fixed bucket width.
        pad = (uint32_t) ggml_env_int("LLAMA_INKLING2_BUCKET_CELLS");
        result = padded(pad);
    }

    if (n_pad_used) {
        *n_pad_used = pad;
    }
    if (n_kv_pool) {
        *n_kv_pool = result;
    }

    // Coarsen only above every row count's staging cutoff (build_inp_kpool). Duplicate positions can widen the tail.
    if (scores && n_pad_cells > pad && ubatch) {
        const int64_t tail = std::max<int64_t>(1, llama_kpool_tail_max(
                [this](llama_seq_id seq) -> const llama_kv_cells & { return get_cells(seq); }, *ubatch, hparams.indexer_kpool));
        const int64_t cutoff = std::max<int64_t>(4096, 8*llama_kpool_n_kv_max(hparams.indexer_top_k, tail));
        if (result >= cutoff) {
            return padded(n_pad_cells);
        }
    }
    return result;
}

uint32_t llama_kv_cache::get_n_devices() const {
    return model.n_seq_devices();
}

static ggml_tensor * split_cache_view(ggml_tensor * view, const llama_kv_cache::slot_info & sinfo, uint32_t n_devices) {
    if (n_devices == 1) { return view; }
    const auto capacity = llama_stream_counts(view->src[0]->ne[2], n_devices);
    std::vector<int64_t> counts(n_devices, 0), first(n_devices, 0);
    uint32_t owner = 0;
    int64_t offset = 0;
    for (const auto stream : sinfo.strm) {
        while (owner < n_devices && stream >= offset + capacity[owner]) { offset += capacity[owner++]; }
        GGML_ASSERT(owner < n_devices);
        if (counts[owner] == 0) { first[owner] = stream - offset; }
        GGML_ASSERT(stream - offset == first[owner] + counts[owner]);
        ++counts[owner];
    }
    ggml_backend_meta_set_stream_view(view, counts.data(), first.data(), n_devices);
    return view;
}

uint32_t llama_kv_cache::get_n_kv_slices(const llama_ubatch & ubatch) {
    if (ubatch.n_seqs_unq == 1) {
        return 1;
    }

    // each sequence set of an equal split is a single sequence
    if (ubatch.equal_seqs() && ubatch.n_seqs == ubatch.n_seqs_unq) {
        return ubatch.n_seqs;
    }

    // runs of equal length with a single sequence each, f.ex. one token per sequence or the drafts of several slots
    // there are n_seqs_unq runs, so each run has a different sequence
    if (ubatch.n_tokens % ubatch.n_seqs_unq == 0) {
        const uint32_t n_tps = ubatch.n_tokens/ubatch.n_seqs_unq;

        bool is_runs = true;
        for (uint32_t i = 0; i < ubatch.n_tokens && is_runs; ++i) {
            is_runs = ubatch.n_seq_id[i] == 1 && ubatch.seq_id[i][0] == ubatch.seq_id[i - i%n_tps][0];
        }

        if (is_runs) {
            return ubatch.n_seqs_unq;
        }
    }

    // cannot split into per-sequence slices - a single slice over the cells of all the sequences of the ubatch
    return 1;
}

// per slice: the sequences whose cells form the list of the slice and the min position of the cells that are not masked by the SWA
static std::vector<llama_kv_rows_slice> get_kv_rows_slices(const llama_ubatch & ubatch, uint32_t n_slices, uint32_t n_swa, llama_swa_type swa_type) {
    std::vector<llama_kv_rows_slice> res(n_slices);

    if (n_slices == 1) {
        res[0].seq_ids.assign(ubatch.seq_id_unq, ubatch.seq_id_unq + ubatch.n_seqs_unq);
    } else {
        const uint32_t n_tps = ubatch.n_tokens/n_slices;

        for (uint32_t s = 0; s < n_slices; ++s) {
            res[s].seq_ids = { ubatch.seq_id[s*n_tps][0] };
        }
    }

    // the cells below these positions are masked by the SWA for all tokens of the slice
    // only for SWA types that mask more cells as the token position grows
    if (n_swa == 0 || (swa_type != LLAMA_SWA_TYPE_STANDARD && swa_type != LLAMA_SWA_TYPE_CHUNKED)) {
        return res;
    }

    const uint32_t n_tps = ubatch.n_tokens/n_slices;

    for (uint32_t s = 0; s < n_slices; ++s) {
        llama_pos p = ubatch.pos[s*n_tps];
        for (uint32_t i = 1; i < n_tps; ++i) {
            p = std::min(p, ubatch.pos[s*n_tps + i]);
        }

        res[s].p_min = swa_type == LLAMA_SWA_TYPE_STANDARD ? p - (llama_pos) n_swa + 1 : (p/(llama_pos) n_swa)*(llama_pos) n_swa;
    }

    return res;
}

uint32_t llama_kv_cache::get_n_kv_rows(const llama_ubatch & ubatch) const {
    if (n_stream != 1) {
        return 0;
    }

    const uint32_t n_slices = get_n_kv_slices(ubatch);

    const auto & cells = v_cells[0];

    const uint32_t n_pad_cur = 256;

    uint32_t n_max = 0;

    const auto slices = get_kv_rows_slices(ubatch, n_slices, n_swa, swa_type);

    for (uint32_t s = 0; s < n_slices; ++s) {
        // a cell shared by several sequences of the slice is counted once per sequence, an upper bound is enough
        uint32_t n = 0;
        for (const llama_seq_id seq_id : slices[s].seq_ids) {
            n += slices[s].p_min < 0 ? cells.seq_n_cells(seq_id) : cells.seq_n_cells(seq_id, slices[s].p_min);
        }

        n_max = std::max(n_max, std::min(n, cells.used_max_p1()));
    }

    // pad like n_kv so that the graph can be reused
    // never 0, it means that kv_rows are not used (f.ex. the sequences have no cells yet in a cache that shares the cells of another one)
    return std::min(std::max(n_pad_cur, GGML_PAD(n_max, n_pad_cur)), get_size());
}

ggml_tensor * llama_kv_cache::get_k(ggml_context * ctx, int32_t il, uint32_t n_kv, const slot_info & sinfo) const {
    const int32_t ikv = map_layer_ids.at(il);

    auto * k = layers[ikv].k;

    const uint64_t kv_size      = get_size();
    const uint64_t n_embd_k_gqa = k->ne[0];

    assert(n_embd_k_gqa == hparams.n_embd_k_gqa(il));

    const bool dp = model.n_seq_devices() > 1;
    const uint32_t ns = dp ? sinfo.n_stream() : sinfo.s1 - sinfo.s0 + 1;

    return split_cache_view(ggml_view_4d(ctx, k,
            hparams.n_embd_head_k(il), hparams.n_head_kv(il), n_kv, ns,
            ggml_row_size(k->type, hparams.n_embd_head_k(il)),
            ggml_row_size(k->type, n_embd_k_gqa),
            ggml_row_size(k->type, n_embd_k_gqa*kv_size),
            ggml_row_size(k->type, n_embd_k_gqa*kv_size)*(dp ? 0 : sinfo.s0)), sinfo, model.n_seq_devices());
}

ggml_tensor * llama_kv_cache::get_v(ggml_context * ctx, int32_t il, uint32_t n_kv, const slot_info & sinfo) const {
    const int32_t ikv = map_layer_ids.at(il);

    auto * v = layers[ikv].v;

    const uint64_t kv_size      = get_size();
    const uint64_t n_embd_v_gqa = v->ne[0];

    // [TAG_V_CACHE_VARIABLE]
    assert(n_embd_v_gqa >= hparams.n_embd_v_gqa(il));

    const bool dp = model.n_seq_devices() > 1;
    const uint32_t ns = dp ? sinfo.n_stream() : sinfo.s1 - sinfo.s0 + 1;

    if (!v_trans) {
        // note: v->nb[1] <= v->nb[2]
        return split_cache_view(ggml_view_4d(ctx, v,
                hparams.n_embd_head_v(il), hparams.n_head_kv(il), n_kv, ns,
                ggml_row_size(v->type, hparams.n_embd_head_v(il)),          // v->nb[1]
                ggml_row_size(v->type, n_embd_v_gqa),                   // v->nb[2]
                ggml_row_size(v->type, n_embd_v_gqa*kv_size),           // v->nb[3]
                ggml_row_size(v->type, n_embd_v_gqa*kv_size)*(dp ? 0 : sinfo.s0)), sinfo, model.n_seq_devices());
    }

    // note: v->nb[1] > v->nb[2]
    return ggml_view_4d(ctx, v,
            n_kv, hparams.n_head_kv(il), hparams.n_embd_head_v(il), ns,
            ggml_row_size(v->type, kv_size*hparams.n_embd_head_v(il)),  // v->nb[1]
            ggml_row_size(v->type, kv_size),                        // v->nb[2]
            ggml_row_size(v->type, kv_size*n_embd_v_gqa),           // v->nb[3]
            ggml_row_size(v->type, kv_size*n_embd_v_gqa)*sinfo.s0);
}

static bool is_type_scaled(ggml_type type) {
    return type == GGML_TYPE_F8_E4M3;
}

ggml_tensor * llama_kv_cache::get_hot_kv(ggml_context * ctx, int32_t il, const slot_info & sinfo) const {
    const int32_t ikv = map_layer_ids.at(il);
    auto * hot = layers[ikv].hot_kv;
    GGML_ASSERT(hot != nullptr);

    const uint64_t n_rows       = 2*(uint64_t) (kv_mixed_hot_rows + 1);
    const uint64_t n_embd_k_gqa = hot->ne[0];
    const uint32_t ns           = sinfo.s1 - sinfo.s0 + 1;

    return ggml_view_4d(ctx, hot,
            hparams.n_embd_head_k(il), hparams.n_head_kv(il), n_rows, ns,
            ggml_row_size(GGML_TYPE_F16, hparams.n_embd_head_k(il)),
            ggml_row_size(GGML_TYPE_F16, n_embd_k_gqa),
            ggml_row_size(GGML_TYPE_F16, n_embd_k_gqa*n_rows),
            ggml_row_size(GGML_TYPE_F16, n_embd_k_gqa*n_rows)*sinfo.s0);
}

ggml_tensor * llama_kv_cache::cpy_k_hot(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * hot_k_idxs, int32_t il) const {
    const int32_t ikv = map_layer_ids.at(il);
    ggml_tensor * hot = layers[ikv].hot_kv;
    GGML_ASSERT(hot != nullptr);

    if (!k_bar.empty() && k_bar[ikv] != nullptr) {
        ggml_tensor * bias = k_bar[ikv];
        if (bias->type != k_cur->type) {
            bias = ggml_cast(ctx, bias, k_cur->type);
        }
        k_cur = ggml_sub(ctx, k_cur, bias);
    }

    const int64_t n_embd_head = k_cur->ne[0];
    const int64_t n_head      = k_cur->ne[1];
    const int64_t n_tokens    = k_cur->ne[2];
    const int64_t n_embd_gqa  = n_embd_head*n_head;

    GGML_ASSERT(ggml_row_size(k_cur->type, n_embd_head) == k_cur->nb[1]);
    k_cur = ggml_view_2d(ctx, k_cur, n_embd_gqa, n_tokens, k_cur->nb[2], 0);

    if (hot->ne[2] > 1) {
        hot = ggml_reshape_2d(ctx, hot, n_embd_gqa, hot->ne[1]*hot->ne[2]);
    }
    return ggml_set_rows(ctx, hot, k_cur, hot_k_idxs);
}

ggml_tensor * llama_kv_cache::cpy_v_hot(ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * hot_v_idxs, int32_t il) const {
    const int32_t ikv = map_layer_ids.at(il);
    ggml_tensor * hot = layers[ikv].hot_kv;
    GGML_ASSERT(hot != nullptr);

    const int64_t n_embd_head = v_cur->ne[0];
    const int64_t n_head      = v_cur->ne[1];
    const int64_t n_tokens    = v_cur->ne[2];
    const int64_t n_embd_gqa  = n_embd_head*n_head;

    GGML_ASSERT(ggml_row_size(v_cur->type, n_embd_head) == v_cur->nb[1]);
    v_cur = ggml_view_2d(ctx, v_cur, n_embd_gqa, n_tokens, v_cur->nb[2], 0);

    if (hot->ne[2] > 1) {
        hot = ggml_reshape_2d(ctx, hot, n_embd_gqa, hot->ne[1]*hot->ne[2]);
    }
    return ggml_set_rows(ctx, hot, v_cur, hot_v_idxs);
}

ggml_tensor * llama_kv_cache::build_input_hot_idxs(ggml_context * ctx, const llama_ubatch & ubatch, const char * name) const {
    ggml_tensor * idxs = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, ubatch.n_tokens);
    ggml_set_input(idxs);
    ggml_set_name(idxs, name);
    return idxs;
}

void llama_kv_cache::set_input_hot_idxs(ggml_tensor * k_dst, ggml_tensor * v_dst, const llama_ubatch * ubatch) const {
    const uint32_t n_tokens = ubatch->n_tokens;
    if (kv_mixed_hot_k_idxs.size() == n_tokens && kv_mixed_hot_v_idxs.size() == n_tokens) {
        ggml_backend_tensor_set(k_dst, kv_mixed_hot_k_idxs.data(), 0, n_tokens*sizeof(int64_t));
        ggml_backend_tensor_set(v_dst, kv_mixed_hot_v_idxs.data(), 0, n_tokens*sizeof(int64_t));
        return;
    }
    std::vector<int64_t> tk(n_tokens, 2*(int64_t) kv_mixed_hot_rows);
    std::vector<int64_t> tv(n_tokens, 2*(int64_t) kv_mixed_hot_rows + 1);
    ggml_backend_tensor_set(k_dst, tk.data(), 0, n_tokens*sizeof(int64_t));
    ggml_backend_tensor_set(v_dst, tv.data(), 0, n_tokens*sizeof(int64_t));
}

ggml_tensor * llama_kv_cache::get_k_scale(int32_t il) const {
    if (!is_type_scaled(type_k())) {
        return nullptr;
    }

    const int32_t ikv = map_layer_ids.at(il);

    return layers[ikv].k_scale;
}

ggml_tensor * llama_kv_cache::get_v_scale(int32_t il) const {
    if (!is_type_scaled(type_v())) {
        return nullptr;
    }

    const int32_t ikv = map_layer_ids.at(il);

    return layers[ikv].v_scale;
}

ggml_tensor * llama_kv_cache::cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il, const slot_info & sinfo) const {
    GGML_UNUSED(sinfo);

    const int32_t ikv = map_layer_ids.at(il);

    ggml_tensor * k = layers[ikv].k;

    // optional per-(head,channel) mean-centering: subtract a fixed bias from the K vector
    // before it is written into the cache. this is exactly softmax-invariant (the same
    // constant is added to every logit of a query's row, which softmax does not see), so
    // nothing else in attention needs to change. see load_kv_mean_center().
    if (!k_bar.empty() && k_bar[ikv] != nullptr) {
        ggml_tensor * bias = k_bar[ikv];
        if (bias->type != k_cur->type) {
            bias = ggml_cast(ctx, bias, k_cur->type);
        }
        k_cur = ggml_sub(ctx, k_cur, bias);
    }

    const int64_t n_embd_head = k_cur->ne[0];
    const int64_t n_head      = k_cur->ne[1];
    const int64_t n_tokens    = k_cur->ne[2];

    const int64_t n_embd_gqa = n_embd_head*n_head;

    // we can merge dims 0 and 1
    // TODO: add ggml helper function for this?
    GGML_ASSERT(ggml_row_size(k_cur->type, n_embd_head) == k_cur->nb[1]);

    k_cur = ggml_view_2d(ctx, k_cur, n_embd_gqa, n_tokens, k_cur->nb[2], 0);

    const int64_t n_stream = k->ne[2];

    if (n_stream > 1) {
        const int64_t kv_size = get_size();

        assert(n_embd_gqa == k->ne[0]);
        assert(kv_size    == k->ne[1]);

        // merge the buffer across all streams because the idxs are global
        k = ggml_reshape_2d(ctx, k, n_embd_gqa, kv_size*n_stream);
    }

    // store the current K values into the cache
    return model.n_seq_devices() > 1 ? ggml_backend_meta_set_rows_local(ctx, k, k_cur, k_idxs)
                                     : ggml_set_rows(ctx, k, k_cur, k_idxs);
}

ggml_tensor * llama_kv_cache::cpy_k_part(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il,
        int64_t n_embd, int64_t i_off) const {
    const int32_t ikv = map_layer_ids.at(il);

    ggml_tensor * k = layers[ikv].k;

    const int64_t n_embd_gqa = k->ne[0];
    const int64_t kv_size    = get_size();
    const int64_t n_stream   = k->ne[2];

    GGML_ASSERT(i_off >= 0 && i_off + n_embd <= n_embd_gqa);
    GGML_ASSERT(k_cur->ne[0] == n_embd);

    // merge the streams: k_idxs are global, exactly as in cpy_k
    ggml_tensor * k2 = ggml_reshape_2d(ctx, k, n_embd_gqa, kv_size*n_stream);

    // a row-slice view of every cell. ggml_set_rows needs contiguous rows in the DEST,
    // which ggml_is_contiguous_rows() grants for a view whose ne[0] slice is contiguous.
    ggml_tensor * dst = ggml_view_2d(ctx, k2, n_embd, kv_size*n_stream,
            k2->nb[1], ggml_row_size(k2->type, i_off));

    return model.n_seq_devices() > 1 ? ggml_backend_meta_set_rows_local(ctx, dst, k_cur, k_idxs)
                                     : ggml_set_rows(ctx, dst, k_cur, k_idxs);
}

ggml_tensor * llama_kv_cache::cpy_v(ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * v_idxs, int32_t il, const slot_info & sinfo) const {
    GGML_UNUSED(sinfo);

    const int32_t ikv = map_layer_ids.at(il);

    auto * v = layers[ikv].v;

    const int64_t n_embd_head = v_cur->ne[0];
    const int64_t n_head      = v_cur->ne[1];
    const int64_t n_tokens    = v_cur->ne[2];

    const int64_t n_embd_gqa = n_embd_head*n_head;

    // we can merge dims 0 and 1
    GGML_ASSERT(ggml_row_size(v_cur->type, n_embd_head) == v_cur->nb[1]);

    const int64_t n_stream = v->ne[2];

    // take this branch when FA is enabled (the V cache is not transposed)
    if (!v_trans) {
        v_cur = ggml_view_2d(ctx, v_cur, n_embd_gqa, n_tokens, v_cur->nb[2], 0);

        if (n_stream > 1) {
            const int64_t kv_size = get_size();

            assert(n_embd_gqa == v->ne[0]);
            assert(kv_size    == v->ne[1]);

            // merge the buffer across all streams because the idxs are global
            v = ggml_reshape_2d(ctx, v, n_embd_gqa, kv_size*n_stream);
        }

        return model.n_seq_devices() > 1 ? ggml_backend_meta_set_rows_local(ctx, v, v_cur, v_idxs)
                                         : ggml_set_rows(ctx, v, v_cur, v_idxs);
    }

    if (ggml_row_size(v_cur->type, n_embd_gqa) == v_cur->nb[2]) {
        // we can merge dims 0, 1 and 2
        v_cur = ggml_reshape_2d(ctx, v_cur, n_embd_gqa, n_tokens);
    } else {
        // otherwise -> make a copy to get contiguous data
        v_cur = ggml_cont_2d   (ctx, v_cur, n_embd_gqa, n_tokens);
    }

    // [TAG_V_CACHE_VARIABLE]
    if (n_embd_gqa < v->ne[0]) {
        v_cur = ggml_pad(ctx, v_cur, v->ne[0] - n_embd_gqa, 0, 0, 0);
    }

    // in this branch the v_idxs are constructed in such a way that each row is a single head element
    ggml_tensor * v_view = ggml_reshape_2d(ctx, v, 1, ggml_nelements(v));

    v_cur = ggml_reshape_2d(ctx, v_cur, 1, ggml_nelements(v_cur));

    return ggml_set_rows(ctx, v_view, v_cur, v_idxs);
}

bool llama_kv_cache::load_kv_mean_center(const char * path, bool require_q4_0) {
    GGML_ASSERT(path != nullptr);
    GGML_ASSERT(k_bar.empty() && "K-cache mean-centering already loaded");

    ggml_context * ctx_data = nullptr;

    struct gguf_init_params gguf_params = {
        /*.no_alloc =*/ false,
        /*.ctx      =*/ &ctx_data,
    };

    struct gguf_context * ctx_gguf = gguf_init_from_file(path, gguf_params);
    if (!ctx_gguf) {
        LLAMA_LOG_ERROR("%s: failed to load K-cache mean-centering bias file from %s\n", __func__, path);
        return false;
    }

    // validate the cache type first, so an unsupported cache type gets its actionable error
    // even when the bias file's basis would also mismatch
    if (require_q4_0) {
        for (const auto & layer : layers) {
            if (layer.k->type != GGML_TYPE_Q4_0) {
                LLAMA_LOG_ERROR("%s: K-cache mean-centering requires K cache type %s, but layer %d has type %s\n",
                        __func__, ggml_type_name(GGML_TYPE_Q4_0), layer.il, ggml_type_name(layer.k->type));
                gguf_free(ctx_gguf);
                ggml_free(ctx_data);
                return false;
            }
        }
    }

    // the bias is only valid in the basis it was measured in: a bias calibrated with the
    // Hadamard K-cache rotation active must be applied with the rotation active, and vice
    // versa. a mismatched basis measurably degrades quantization quality instead of
    // improving it (see tools/kv-mean-center/README.md), so refuse it outright.
    {
        const int64_t idx_k_rot = gguf_find_key(ctx_gguf, "kv_mean_center.k_rot");
        if (idx_k_rot >= 0) {
            if (gguf_get_kv_type(ctx_gguf, idx_k_rot) != GGUF_TYPE_BOOL) {
                LLAMA_LOG_ERROR("%s: bias file %s has a non-boolean kv_mean_center.k_rot key - malformed file\n",
                        __func__, path);
                gguf_free(ctx_gguf);
                ggml_free(ctx_data);
                return false;
            }
            const bool bias_k_rot = gguf_get_val_bool(ctx_gguf, idx_k_rot);
            if (bias_k_rot != (n_rot_k != 0)) {
                LLAMA_LOG_ERROR("%s: bias file %s was calibrated with the K-cache rotation %s, but it is %s "
                        "for this context - recalibrate with matching cache settings "
                        "(or set LLAMA_ATTN_ROT_DISABLE=1 consistently in both)\n",
                        __func__, path,
                        bias_k_rot  ? "active" : "inactive",
                        n_rot_k != 0 ? "active" : "inactive");
                gguf_free(ctx_gguf);
                ggml_free(ctx_data);
                return false;
            }
        } else {
            LLAMA_LOG_WARN("%s: bias file %s does not record its calibration basis (kv_mean_center.k_rot); "
                    "K-cache rotation is %s for this context - a basis mismatch degrades quality\n",
                    __func__, path, n_rot_k != 0 ? "active" : "inactive");
        }
    }

    k_bar.resize(layers.size(), nullptr);

    // one ggml context (+ backend buffer) per unique buffer type, so each bias tensor ends up
    // on the same device as the K cache tensor it is subtracted against (see cpy_k())
    std::map<ggml_backend_buffer_type_t, ggml_context *> ctx_map;

    auto ctx_for_buft = [&](ggml_backend_buffer_type_t buft) -> ggml_context * {
        auto it = ctx_map.find(buft);
        if (it != ctx_map.end()) {
            return it->second;
        }

        ggml_init_params params = {
            /*.mem_size   =*/ layers.size()*ggml_tensor_overhead(),
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ true,
        };

        ggml_context * res = ggml_init(params);
        if (res) {
            ctx_map.emplace(buft, res);
            k_bar_ctxs.emplace_back(res);
        }

        return res;
    };

    bool ok = true;
    size_t n_centered = 0;

    for (size_t ikv = 0; ikv < layers.size() && ok; ++ikv) {
        const int32_t il = layers[ikv].il;

        const std::string name = "kv_bar.blk." + std::to_string(il) + ".k";

        ggml_tensor * src = ggml_get_tensor(ctx_data, name.c_str());
        if (!src) {
            // no bias provided for this layer - leave it uncentered
            continue;
        }

        if (src->type != GGML_TYPE_F32) {
            LLAMA_LOG_ERROR("%s: bias tensor %s must be F32 (got %s)\n",
                    __func__, name.c_str(), ggml_type_name(src->type));
            ok = false;
            break;
        }

        const int64_t n_embd_head = hparams.n_embd_head_k(il);
        const int64_t n_head_kv   = hparams.n_head_kv(il);

        if (ggml_nelements(src) != n_embd_head*n_head_kv) {
            LLAMA_LOG_ERROR("%s: bias tensor %s has %" PRId64 " elements, expected %" PRId64 " (n_embd_head_k * n_head_kv)\n",
                    __func__, name.c_str(), ggml_nelements(src), n_embd_head*n_head_kv);
            ok = false;
            break;
        }

        ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(model.dev_layer(il));

        ggml_context * ctx = ctx_for_buft(buft);
        if (!ctx) {
            LLAMA_LOG_ERROR("%s: failed to allocate context for K-cache mean-centering bias (layer %d)\n", __func__, il);
            ok = false;
            break;
        }

        ggml_tensor * bias = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd_head, n_head_kv);
        ggml_format_name(bias, "kv_bar_l%d", il);

        k_bar[ikv] = bias;
        n_centered++;
    }

    if (ok) {
        for (auto & [buft, ctx] : ctx_map) {
            ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
            if (!buf) {
                LLAMA_LOG_ERROR("%s: failed to allocate buffer for K-cache mean-centering bias\n", __func__);
                ok = false;
                break;
            }
            k_bar_bufs.emplace_back(buf);
        }
    }

    if (ok) {
        for (size_t ikv = 0; ikv < layers.size(); ++ikv) {
            if (!k_bar[ikv]) {
                continue;
            }

            const int32_t il = layers[ikv].il;
            const std::string name = "kv_bar.blk." + std::to_string(il) + ".k";

            ggml_tensor * src = ggml_get_tensor(ctx_data, name.c_str());
            ggml_backend_tensor_set(k_bar[ikv], src->data, 0, ggml_nbytes(k_bar[ikv]));
        }
    }

    gguf_free(ctx_gguf);
    ggml_free(ctx_data);

    if (!ok) {
        // roll back so cpy_k() never observes a half-initialized k_bar
        k_bar.clear();
        k_bar_bufs.clear();
        k_bar_ctxs.clear();

        return false;
    }

    LLAMA_LOG_INFO("%s: loaded K-cache mean-centering bias for %zu / %zu layer(s) from %s\n",
            __func__, n_centered, layers.size(), path);

    return true;
}

ggml_tensor * llama_kv_cache::build_input_k_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const {
    const uint32_t n_tokens = ubatch.n_tokens;

    ggml_tensor * k_idxs = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_tokens);

    if (model.n_seq_devices() > 1) {
        const auto counts = ubatch.device_rows(model.n_seq_devices());
        ggml_backend_meta_split_input(k_idxs, 0, counts.data(), counts.size());
    }

    ggml_set_input(k_idxs);
    ggml_set_name(k_idxs, "attn_inp_k_idxs");

    return k_idxs;
}

ggml_tensor * llama_kv_cache::build_input_v_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const {
    const uint32_t n_tokens = ubatch.n_tokens;

    ggml_tensor * v_idxs;

    if (!v_trans) {
        v_idxs = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_tokens);
    } else {
        v_idxs = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_tokens*hparams.n_embd_v_gqa_max());
    }

    ggml_set_input(v_idxs);
    ggml_set_name(v_idxs, "attn_inp_v_idxs");

    if (model.n_seq_devices() > 1) {
        GGML_ASSERT(!v_trans);
        const auto counts = ubatch.device_rows(model.n_seq_devices());
        ggml_backend_meta_split_input(v_idxs, 0, counts.data(), counts.size());
    }

    return v_idxs;
}

std::vector<ggml_tensor *> llama_kv_cache::get_k_rot() const {
    std::vector<ggml_tensor *> res(hparams.n_layer_all, nullptr);
    for (const auto & [il, idx] : map_layer_ids) {
        res[il] = layers[idx].k_rot;
    }
    return res;
}

std::vector<ggml_tensor *> llama_kv_cache::get_v_rot() const {
    std::vector<ggml_tensor *> res(hparams.n_layer_all, nullptr);
    for (const auto & [il, idx] : map_layer_ids) {
        res[il] = layers[idx].v_rot;
    }
    return res;
}

void llama_kv_cache::set_input_k_idxs(ggml_tensor * dst, const llama_ubatch * ubatch, const slot_info & sinfo) const {
    const uint32_t n_tokens = ubatch->n_tokens;
    GGML_ASSERT(n_tokens == (int64_t) sinfo.size()*sinfo.n_stream());

    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));
    int64_t * data = (int64_t *) dst->data;

    for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
        const llama_seq_id seq = ubatch->seq_id[s*sinfo.size()][0];
        GGML_ASSERT(int32_t(seq_to_stream[seq]) == sinfo.strm[s]);
        const int64_t stream = model.n_seq_devices() > 1 ? seq/model.n_seq_devices() : sinfo.strm[s];
        const int64_t offs = stream*get_size();

        for (uint32_t i = 0; i < sinfo.size(); ++i) {
            data[s*sinfo.size() + i] = offs + sinfo.idxs[s][i];
        }
    }
}

void llama_kv_cache::set_input_v_idxs(ggml_tensor * dst, const llama_ubatch * ubatch, const slot_info & sinfo) const {
    const uint32_t n_tokens = ubatch->n_tokens;
    GGML_ASSERT(n_tokens == (int64_t) sinfo.size()*sinfo.n_stream());

    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));
    int64_t * data = (int64_t *) dst->data;

    if (!v_trans) {
        for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
            const llama_seq_id seq = ubatch->seq_id[s*sinfo.size()][0];
            GGML_ASSERT(int32_t(seq_to_stream[seq]) == sinfo.strm[s]);
            const int64_t stream = model.n_seq_devices() > 1 ? seq/model.n_seq_devices() : sinfo.strm[s];
            const int64_t offs = stream*get_size();

            for (uint32_t i = 0; i < sinfo.size(); ++i) {
                data[s*sinfo.size() + i] = offs + sinfo.idxs[s][i];
            }
        }
    } else {
        // note: the V cache is transposed when not using flash attention
        const int64_t kv_size = get_size();

        const int64_t n_embd_v_gqa = hparams.n_embd_v_gqa_max();

        for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
            const int64_t offs = sinfo.strm[s]*kv_size*n_embd_v_gqa;

            for (uint32_t i = 0; i < sinfo.size(); ++i) {
                for (uint32_t j = 0; j < n_embd_v_gqa; ++j) {
                    data[s*sinfo.size()*n_embd_v_gqa + i*n_embd_v_gqa + j] = offs + j*kv_size + sinfo.idxs[s][i];
                }
            }
        }
    }
}

void llama_kv_cache::set_input_k_shift(ggml_tensor * dst) const {
    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));

    int32_t * data = (int32_t *) dst->data;

    for (uint32_t s = 0; s < n_stream; ++s) {
        const auto & cells = v_cells[s];

        for (uint32_t i = 0; i < cells.size(); ++i) {
            data[s*cells.size() + i] = cells.is_empty(i) ? 0 : cells.get_shift(i);
        }
    }
}

struct args_set_input_attn_mask {
    const llama_hparams & hparams;
    const llama_ubatch  * ubatch;

    const std::vector<llama_kv_cells> & v_cells;
    const std::vector<uint32_t>       & seq_to_stream;

    uint32_t       n_swa;
    llama_swa_type swa_type;

    int64_t n_kv;
    int64_t n_stream;
    int64_t n_tps;

    bool scan_cells; // LLAMA_KQ_MASK_SCAN_LEGACY=1: every row's first scan tests cell by cell, not 16 cells at a time
    bool seq_bits;   // off with LLAMA_KQ_MASK_SEQ_BITS_LEGACY=1: a group of 64 cells before the batch is scanned, not taken
                     // whole from the sequence's bits (llama_kv_cells::seq_cells64)

    // cell of each mask column per stream, nullptr if the columns are the cells
    const int32_t * rows;

    // every token of a sequence reuses the mask row of the first one (dspark draft blocks)
    bool block_uniform;
};

// How a mask cell is stored: an additive f16/f32 value, or one bit of a 16-bit word (set = attend; GGML_TYPE_I16, 16 cells per element).
// Cell indices are row*n_kv + cell in both layouts; a packed row is n_kv/16 words, so rows start on a word boundary.
// keep_n: the n (<= 16) cells from idx, bit k of live set = cell idx + k attended.
template<typename T>
struct llama_kq_mask_cells {
    T * data;
    const T v_keep = llama_cast<T>(0.0f);
    const T v_drop = llama_cast<T>(-INFINITY);

    void keep(uint64_t idx) const { data[idx] = v_keep; }
    void drop(uint64_t idx) const { data[idx] = v_drop; }
    void set (uint64_t idx, float v) const { data[idx] = llama_cast<T>(v); }
    void keep_n(uint64_t idx, uint32_t live, uint32_t n) const {
        for (uint32_t k = 0; k < n; ++k) {
            data[idx + k] = (live >> k) & 1 ? v_keep : v_drop;
        }
    }
    void copy_row(uint64_t dst, uint64_t src, uint64_t n_kv) const { std::copy(data + src, data + src + n_kv, data + dst); }
};

struct llama_kq_mask_bits {
    uint16_t * data;

    void keep(uint64_t idx) const { data[idx >> 4] |=  (uint16_t) (1u << (idx & 15)); }
    void drop(uint64_t idx) const { data[idx >> 4] &= ~(uint16_t) (1u << (idx & 15)); }
    void set (uint64_t, float)  const { GGML_ABORT("a bit-packed KQ mask holds no ALiBi values"); }
    void keep_n(uint64_t idx, uint32_t live, uint32_t n) const {
        GGML_ASSERT(idx % 16 == 0 && n == 16); // whole words: n_kv is a multiple of 16
        data[idx >> 4] = (uint16_t) live;
    }
    void copy_row(uint64_t dst, uint64_t src, uint64_t n_kv) const { std::copy(data + src/16, data + (src + n_kv)/16, data + dst/16); }
};

template<typename S, bool causal, bool swa, bool is_2d, bool alibi>
static void set_input_attn_mask_impl(const args_set_input_attn_mask & args, const S & st) {
  //const auto & hparams = args.hparams;
    const auto & ubatch  = args.ubatch;

    const auto & v_cells       = args.v_cells;
    const auto & seq_to_stream = args.seq_to_stream;

    const uint32_t       n_swa    = args.n_swa;
    const llama_swa_type swa_type = args.swa_type;

    const int64_t n_kv     = args.n_kv;
    const int64_t n_stream = args.n_stream;
    const int64_t n_tps    = args.n_tps;

    const int32_t * rows_all = args.rows;

    // the min and max position in the batch for each sequence
    llama_pos seq_pos_min[LLAMA_MAX_SEQ];
    llama_pos seq_pos_max[LLAMA_MAX_SEQ];
    std::fill(seq_pos_min, seq_pos_min + LLAMA_MAX_SEQ, INT32_MAX);
    std::fill(seq_pos_max, seq_pos_max + LLAMA_MAX_SEQ, -1);

    // first token of each sequence, used when the mask rows are shared within a sequence
    uint32_t seq_first[LLAMA_MAX_SEQ];
    std::fill(seq_first, seq_first + LLAMA_MAX_SEQ, ubatch->n_tokens);

    for (uint32_t i = 0; i < ubatch->n_tokens; ++i) {
        const llama_seq_id seq_id = ubatch->seq_id[i][0];

        seq_pos_min[seq_id] = std::min(seq_pos_min[seq_id], ubatch->pos[i]);
        seq_pos_max[seq_id] = std::max(seq_pos_max[seq_id], ubatch->pos[i]);
        seq_first[seq_id]   = std::min(seq_first[seq_id],   i);
    }

    for (uint32_t s = 0; s < n_stream; ++s) {
        // bookkeeping of the KQ mask cells that could change for other tokens of the same sequence
        std::unordered_map<llama_seq_id, uint32_t>              seq_srct;
        std::unordered_map<llama_seq_id, std::vector<uint32_t>> seq_idxs;

        const int32_t * rows = rows_all ? rows_all + s*n_kv : nullptr;

        // a row after its sequence's first copies that row and redoes only the cells near the batch's positions, reading the
        // first row and its bookkeeping and never writing them: those rows wait for every first row, then split across
        // threads (llama-host-threads.h)
        std::vector<uint32_t> later;

        const auto row = [&](uint32_t ii, bool is_later) {
            const uint32_t i = s*n_tps + ii;

            const llama_seq_id seq_id = ubatch->seq_id[i][0];

            if (!alibi && !is_later && seq_srct.find(seq_id) != seq_srct.end()) {
                later.push_back(ii);
                return;
            }

            const auto & cells = v_cells.at(seq_to_stream[seq_id]);

            // shared mask rows: every query of the sequence uses the row of its first token
            const uint32_t iq = args.block_uniform ? seq_first[seq_id] : i;

                  llama_pos p0 = -1;
            const llama_pos p1 = ubatch->pos[iq];

            // for M-RoPE
            const llama_pos p1_x = is_2d ? ubatch->pos[iq + ubatch->n_tokens*2] : 0;
            const llama_pos p1_y = is_2d ? ubatch->pos[iq + ubatch->n_tokens]   : 0;

            const uint64_t idst = n_kv*i;

            // for tokens of the same sequence, the mask is mostly the same, so we can reuse it
            // the only cells that could change are the ones that are with similar positions as the
            //   ones in the batch (i.e. due to causal masking, SWA, etc.)
            // keep track of those cells and shortcut the loop to save time
            // note: this optimization is not compatible with Alibi position encoding
            // ref:  https://github.com/ggml-org/llama.cpp/pull/18842
            bool prev = false;

            auto & idxs = is_later ? seq_idxs.find(seq_id)->second : seq_idxs[seq_id];

            if (!alibi) {
                if (is_later) {
                    const uint32_t srct = seq_srct.find(seq_id)->second;

                    const uint64_t idst_prev = n_kv*srct;

                    st.copy_row(idst, idst_prev, n_kv);

                    prev = true;
                } else {
                    idxs.clear();
                    idxs.reserve(ubatch->n_tokens + n_swa + 32);

                    seq_srct[seq_id] = i;
                }
            }

            // true if the token does not see cell j, a cell of its sequence at position pj
            const auto masked = [&cells, &args, &seq_pos_min, seq_id, p1, p1_x, p1_y, n_swa, swa_type](uint32_t j, llama_pos pj) {
                if (causal) {
                    // mask future tokens
                    if (pj > p1) {
                        return true;
                    }

                    // M-RoPE causal mask
                    if (is_2d && pj == p1 && cells.ext_get(j).is_2d_gt(p1_x, p1_y)) {
                        return true;
                    }
                }

                // apply SWA if any
                // see llama_non_causal_type
                const bool in_span = !causal && args.hparams.non_causal_type == LLAMA_NON_CAUSAL_TYPE_SWA_FULL && pj >= seq_pos_min[seq_id];
                return swa && !in_span && llama_hparams::is_masked_swa(n_swa, swa_type, pj, p1);
            };

            // the first scan of a sequence, 16 cells at a time: a cell past the sequence's last position in the batch is masked
            // for all its tokens (causal) and needs no bookkeeping, so a group with no cell before that position is masked on
            // the positions alone, before any sequence bitset is read (with --kv-unified, the cells of other sequences)
            if (!alibi && !prev && !args.scan_cells && !rows) {
                const llama_pos p_end = causal ? seq_pos_max[seq_id] + 1 : std::numeric_limits<llama_pos>::max();
                const llama_pos p_idx = seq_pos_min[seq_id] - (llama_pos) (n_swa + 32);

                // bit k set: the token sees cell j0 + k; n is a constant for the full groups. The cells of a sequence come in
                // runs, so the positions alone are checked first only after a group with no cell seen.
                bool check = true;
                const auto scan = [&cells, &idxs, &check, masked, p_end, p_idx, seq_id](uint32_t j0, auto n) {
                    bool any = !check;
                    for (uint32_t k = 0; check && k < n; ++k) {
                        any |= cells.pos_in(j0 + k, 0, p_end);
                    }

                    uint32_t live = 0;
                    for (uint32_t k = 0; any && k < n; ++k) {
                        const uint32_t j = j0 + k;

                        if (!cells.pos_in(j, 0, p_end) || !cells.seq_has(j, seq_id)) {
                            continue;
                        }

                        const llama_pos pj = cells.pos_get(j);

                        if (pj >= p_idx) {
                            idxs.push_back(j);
                        }

                        live |= (uint32_t) !masked(j, pj) << k;
                    }

                    check = live == 0;

                    return live;
                };

                uint32_t j0 = 0;

                // a group of 64 cells with no position from p_idx on holds no cell the batch's positions can mask and none
                // for the bookkeeping: without a window the sequence's bits are the row there, read in one word instead of
                // 64 cells' positions and sequence sets. The groups with a later position (the batch's own cells, the cells
                // a rejected draft freed and the batch took) are scanned 16 cells at a time
                if (args.seq_bits) {
                    for (; j0 + 64 <= n_kv; j0 += 64) {
                        const uint64_t bits = cells.seq_cells64(seq_id, j0/64);

                        if (bits == 0 || (!swa && cells.pos_max64(j0/64) < p_idx)) {
                            for (uint32_t k = 0; k < 64; k += 16) {
                                st.keep_n(idst + j0 + k, (uint32_t) (bits >> k) & 0xffff, 16);
                            }
                            continue;
                        }

                        for (uint32_t k = 0; k < 64; k += 16) {
                            st.keep_n(idst + j0 + k, scan(j0 + k, std::integral_constant<uint32_t, 16>{}), 16);
                        }
                    }
                }

                for (; j0 + 16 <= n_kv; j0 += 16) {
                    st.keep_n(idst + j0, scan(j0, std::integral_constant<uint32_t, 16>{}), 16);
                }
                if (j0 < n_kv) {
                    st.keep_n(idst + j0, scan(j0, (uint32_t) (n_kv - j0)), n_kv - j0);
                }

                return;
            }

            for (uint32_t jj = 0; jj < n_kv; ++jj) {
                // column of the mask
                uint32_t c = jj;

                // we have an exiting mask for this sequence -> update just seq_idxs
                if (!alibi) {
                    if (prev) {
                        if (jj >= idxs.size()) {
                            break;
                        }

                        c = idxs[jj];
                    }
                }

                // cell of the column, negative for padding
                const int32_t j = rows ? rows[c] : (int32_t) c;

                if (j < 0 || cells.is_empty(j)) {
                    goto skip;
                }

                // mask the token if not the same sequence
                if (!cells.seq_has(j, seq_id)) {
                    goto skip;
                }

                p0 = cells.pos_get(j);

                if (!alibi) {
                    if (!prev) {
                        // record all cells for which: p0 >= seq_pos_min[seq_id] - n_swa - 32
                        if (p0 + (int32_t) (n_swa + 32) >= seq_pos_min[seq_id]) {
                            idxs.push_back(c);
                        }
                    }
                }

                if (masked(j, p0)) {
                    goto skip;
                }

                if (alibi) {
                    st.set(idst + c, static_cast<float>(-std::abs(p0 - p1)));
                } else {
                    st.keep(idst + c);
                }

                continue;
skip:
                st.drop(idst + c);
            }
        };

        for (uint32_t ii = 0; ii < n_tps; ++ii) {
            row(ii, false);
        }

        llama_host_rows((int64_t) later.size(), n_kv*(int64_t) sizeof(ggml_fp16_t), [&](int64_t k0, int64_t k1, int) {
            for (int64_t k = k0; k < k1; ++k) {
                row(later[k], true);
            }
        });
    }
}

template<typename S, bool causal, bool swa, bool is_2d>
static void set_input_attn_mask_impl(const args_set_input_attn_mask & args, const S & st) {
    const bool alibi = args.hparams.use_alibi;
    if (alibi) {
        set_input_attn_mask_impl<S, causal, swa, is_2d, true> (args, st);
    } else {
        set_input_attn_mask_impl<S, causal, swa, is_2d, false>(args, st);
    }
}

template<typename S, bool causal, bool swa>
static void set_input_attn_mask_impl(const args_set_input_attn_mask & args, const S & st) {
    const bool is_2d = args.ubatch->is_pos_2d();
    if (is_2d) {
        set_input_attn_mask_impl<S, causal, swa, true> (args, st);
    } else {
        set_input_attn_mask_impl<S, causal, swa, false>(args, st);
    }
}

template<typename S, bool causal>
static void set_input_attn_mask_impl(const args_set_input_attn_mask & args, const S & st) {
    const bool swa = args.swa_type != LLAMA_SWA_TYPE_NONE;
    if (swa) {
        set_input_attn_mask_impl<S, causal, true> (args, st);
    } else {
        set_input_attn_mask_impl<S, causal, false>(args, st);
    }
}

template<typename S>
static void set_input_attn_mask_impl(const args_set_input_attn_mask & args, const S & st, bool causal_attn) {
    if (causal_attn) {
        set_input_attn_mask_impl<S, true> (args, st);
    } else {
        set_input_attn_mask_impl<S, false>(args, st);
    }
}

// The rows of a prompt chunk in a bit-packed mask: a row of a sequence sees exactly the sequence's cells within a position
// window, so the row is the sequence's bits from the window's lower bound on with the cells above its position cleared, and
// every row after it is the row before it with the cells the window's edges passed set or cleared, instead of a scan of
// the row's n_kv cells. The caller admits only the layouts whose scans see the same window: no cell rows, no ALiBi, no 2D
// positions, no per-block uniform rows, and no window or the standard causal window (llama_kv_cache_set_input_kq_mask)
struct llama_kq_mask_rows {
    llama_seq_id seq;

    const llama_kv_cells * cells;

    // (position, mask row of the ubatch) of the sequence's tokens, ascending
    std::vector<std::pair<llama_pos, uint32_t>> rows;
};

// rows [k0, k1) of one sequence: the first from the sequence's bits, each later one from the row before it
static void set_input_attn_mask_rows_block(
        uint16_t * data, int64_t n_words, const llama_kq_mask_rows & of, size_t k0, size_t k1,
        bool causal, bool window, uint32_t n_swa) {
    const auto & rows = of.rows;
    const auto & sp   = of.cells->seq_pos_get(of.seq);

    const llama_pos p_first = rows[k0].first;

    uint16_t * row = data + (int64_t) rows[k0].second*n_words;

    // the first row of the block: the sequence's bits from the window's lower bound on
    const int64_t p_min = window ? std::max<int64_t>(0, (int64_t) p_first + 1 - n_swa) : 0;

    of.cells->copy_seq_mask(of.seq, row, (size_t) n_words, (llama_pos) p_min, window ? n_swa : 0);

    // the cells above this row's position: cleared now, and the rows after it set back the ones their positions pass
    if (causal) {
        for (auto it = sp.upper_bound({ p_first, std::numeric_limits<uint32_t>::max() }); it != sp.end(); ++it) {
            const uint32_t cell = it->second;
            if (cell/16 < (uint32_t) n_words) {
                row[cell/16] &= (uint16_t) ~(1u << (cell%16));
            }
        }
    }

    for (size_t k = k0 + 1; k < k1; ++k) {
        const llama_pos p0 = rows[k - 1].first;
        const llama_pos p1 = rows[k    ].first;

        uint16_t       * cur  = data + (int64_t) rows[k    ].second*n_words;
        const uint16_t * prev = data + (int64_t) rows[k - 1].second*n_words;

        std::memcpy(cur, prev, (size_t) n_words*sizeof(uint16_t));

        // the cells the window's upper edge passed are visible to this row
        if (causal && p1 != p0) {
            for (auto it = sp.upper_bound({ p0, std::numeric_limits<uint32_t>::max() }); it != sp.end() && it->first <= p1; ++it) {
                const uint32_t cell = it->second;
                if (cell/16 < (uint32_t) n_words) {
                    cur[cell/16] |= (uint16_t) (1u << (cell%16));
                }
            }
        }

        // the cells the window's lower edge passed are not
        if (window) {
            const int64_t m0 = std::max<int64_t>(0, (int64_t) p0 + 1 - n_swa);
            const int64_t m1 = std::max<int64_t>(0, (int64_t) p1 + 1 - n_swa);

            if (m1 > m0) {
                for (auto it = sp.lower_bound({ (llama_pos) m0, 0 }); it != sp.end() && it->first < m1; ++it) {
                    const uint32_t cell = it->second;
                    if (cell/16 < (uint32_t) n_words) {
                        cur[cell/16] &= (uint16_t) ~(1u << (cell%16));
                    }
                }
            }
        }
    }
}

// every row of the ubatch from the sequences' bits (see llama_kq_mask_rows): one job a block of rows of one sequence, so a
// single sequence's prompt also splits across the host threads (llama_host_rows)
static void set_input_attn_mask_rows(
        uint16_t * data, const llama_ubatch * ubatch,
        const std::vector<llama_kv_cells> & v_cells, const std::vector<uint32_t> & seq_to_stream,
        int64_t n_words, int64_t n_stream, int64_t n_tps, bool causal, bool window, uint32_t n_swa) {
    // the rows a job: enough for one sequence's prompt to split across the host threads, few enough that a block's first
    // row (from the bits, the cells above it cleared) is rebuilt only once a handful of rows
    static constexpr size_t rows_per_job = 8;

    struct job {
        size_t group;
        size_t k0;
        size_t k1;
    };

    std::vector<llama_kq_mask_rows> groups;
    std::vector<job> jobs;

    groups.reserve(ubatch->n_tokens);
    jobs.reserve(ubatch->n_tokens/rows_per_job + 1);

    for (int64_t s = 0; s < n_stream; ++s) {
        const size_t g0 = groups.size();

        // the group of each sequence's rows in this stream
        int32_t group_of[LLAMA_MAX_SEQ];
        std::fill(group_of, group_of + LLAMA_MAX_SEQ, -1);

        for (int64_t ii = 0; ii < n_tps; ++ii) {
            const uint32_t i = (uint32_t) (s*n_tps + ii);

            const llama_seq_id seq = ubatch->seq_id[i][0];

            GGML_ASSERT(seq >= 0 && seq < LLAMA_MAX_SEQ);

            if (group_of[seq] < 0) {
                group_of[seq] = (int32_t) groups.size();
                groups.push_back({ seq, &v_cells.at(seq_to_stream.at(seq)), {} });
            }

            groups[group_of[seq]].rows.emplace_back(ubatch->pos[i], i);
        }

        for (size_t g = g0; g < groups.size(); ++g) {
            auto & rows = groups[g].rows;
            std::sort(rows.begin(), rows.end());

            // at most 32 jobs (and fewer when the sequence keeps many cells above the ubatch, which every job's first
            // row clears): a block's first row is built from the bits, so the cells above it cost once a block
            const auto & sp = groups[g].cells->seq_pos_get(groups[g].seq);

            const int64_t n_future = (int64_t) std::distance(
                    sp.upper_bound({ rows.back().first, std::numeric_limits<uint32_t>::max() }), sp.end());

            size_t n_jobs = std::max<size_t>(1, std::min<size_t>((rows.size() + rows_per_job - 1)/rows_per_job, 32));
            if (n_future > 0) {
                n_jobs = std::min<size_t>(n_jobs, std::max<size_t>(1, (size_t) ((int64_t) rows.size()*8/n_future)));
            }

            const size_t per_job = (rows.size() + n_jobs - 1)/n_jobs;

            for (size_t k0 = 0; k0 < rows.size(); k0 += per_job) {
                jobs.push_back({ g, k0, std::min(rows.size(), k0 + per_job) });
            }
        }
    }

    if (jobs.empty()) {
        return;
    }

    const int64_t row_bytes = std::max<int64_t>(1, (int64_t) ubatch->n_tokens*n_words*(int64_t) sizeof(uint16_t)/(int64_t) jobs.size());

    llama_host_rows((int64_t) jobs.size(), row_bytes, [&](int64_t b0, int64_t b1, int) {
        for (int64_t b = b0; b < b1; ++b) {
            const auto & j = jobs[b];
            set_input_attn_mask_rows_block(data, n_words, groups[j.group], j.k0, j.k1, causal, window, n_swa);
        }
    });
}

bool llama_kv_cache_set_input_kq_mask(
        ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn, const llama_hparams & hparams,
        const std::vector<llama_kv_cells> & v_cells, const std::vector<uint32_t> & seq_to_stream,
        uint32_t n_swa, llama_swa_type swa_type, bool scan_cells, bool seq_bits,
        const int32_t * rows, bool block_uniform, bool chunk_rows) {
    const uint32_t n_tokens = ubatch->n_tokens;

    const int64_t n_kv     = dst->type == GGML_TYPE_I16 ? 16*dst->ne[0] : dst->ne[0]; // a bit-packed mask holds 16 cells per element
    const int64_t n_stream = dst->ne[3]; // num streams in the current ubatch

    GGML_ASSERT(n_tokens%n_stream == 0);

    // n_tps == n_tokens_per_stream
    const int64_t n_tps = n_tokens/n_stream;

    // see llama_non_causal_type
    // only the SWA cache (or the SWA layers of a single cache) become non-causal
    if (!causal_attn && hparams.non_causal_type == LLAMA_NON_CAUSAL_TYPE_SWA_ONLY) {
        causal_attn = swa_type == LLAMA_SWA_TYPE_NONE;
    }
    if (!causal_attn && hparams.non_causal_type == LLAMA_NON_CAUSAL_TYPE_DENSE_ONLY) {
        causal_attn = swa_type != LLAMA_SWA_TYPE_NONE;
    }

    // At the tip of a sequence only the sliding window can hide any of its cells.
    const bool standard_window = causal_attn && swa_type == LLAMA_SWA_TYPE_STANDARD &&
            n_swa > 0 && n_swa <= INT32_MAX && !block_uniform;
    if (dst->type == GGML_TYPE_I16 && seq_bits && !scan_cells && !rows && !hparams.use_alibi &&
            (swa_type == LLAMA_SWA_TYPE_NONE || standard_window) && !ubatch->is_pos_2d()) {
        bool complete = true;
        for (uint32_t i = 0; causal_attn && i < n_tokens; ++i) {
            const auto seq = ubatch->seq_id[i][0];
            if (ubatch->pos[i] < v_cells.at(seq_to_stream.at(seq)).seq_pos_max(seq)) {
                complete = false;
                break;
            }
        }
        if (complete) {
            auto * data = static_cast<uint16_t *>(dst->data);
            const size_t words = dst->ne[0];
            const auto copy_rows = [&](int64_t first, int64_t end, int) {
                for (int64_t i = first; i < end; ++i) {
                    const auto seq = ubatch->seq_id[i][0];
                    const auto & cells = v_cells.at(seq_to_stream.at(seq));
                    const llama_pos p_min = standard_window ? std::max<int64_t>(0, int64_t(ubatch->pos[i]) + 1 - n_swa) : 0;
                    cells.copy_seq_mask(seq, data + i*words, words, p_min, n_swa);
                }
            };
            if (causal_attn) {
                // Each row is at its sequence tip. Copy the decode rows without starting a team of threads.
                copy_rows(0, n_tokens, 0);
            } else {
                llama_host_rows(n_tokens, words*sizeof(uint16_t), copy_rows);
            }
            return false;
        }
    }

    // A prompt chunk, its rows anywhere in their sequences: the tip path's layouts, whose rows are the sequences' bits
    // within a position window (see llama_kq_mask_rows), built a row from the row before it instead of cell by cell
    if (chunk_rows && dst->type == GGML_TYPE_I16 && seq_bits && !scan_cells && !rows && !hparams.use_alibi &&
            (swa_type == LLAMA_SWA_TYPE_NONE || standard_window) && !ubatch->is_pos_2d() && !block_uniform) {
        set_input_attn_mask_rows(static_cast<uint16_t *>(dst->data), ubatch, v_cells, seq_to_stream,
                dst->ne[0], n_stream, n_tps, causal_attn, standard_window, n_swa);
        return true;
    }

    const args_set_input_attn_mask args = {
        /*.hparams          =*/ hparams,
        /*.ubatch           =*/ ubatch,
        /*.v_cells          =*/ v_cells,
        /*.seq_to_stream    =*/ seq_to_stream,
        /*.n_swa            =*/ n_swa,
        /*.swa_type         =*/ swa_type,
        /*.n_kv             =*/ n_kv,
        /*.n_stream         =*/ n_stream,
        /*.n_tps            =*/ n_tps,
        /*.scan_cells       =*/ scan_cells,
        /*.seq_bits         =*/ seq_bits,
        /*.rows             =*/ rows,
        /*.block_uniform    =*/ block_uniform,
    };

    switch (dst->type) {
        case GGML_TYPE_F16: set_input_attn_mask_impl(args, llama_kq_mask_cells<ggml_fp16_t>{(ggml_fp16_t *) dst->data}, causal_attn); break;
        case GGML_TYPE_F32: set_input_attn_mask_impl(args, llama_kq_mask_cells<float>      {(float       *) dst->data}, causal_attn); break;
        case GGML_TYPE_I16: set_input_attn_mask_impl(args, llama_kq_mask_bits              {(uint16_t    *) dst->data}, causal_attn); break;
        default: GGML_ABORT("unsupported KQ mask type %s", ggml_type_name(dst->type));
    }

    return false;
}

static constexpr int64_t LLAMA_KV_RESIDENT_KQ_MASK_MAX_ROWS = 32;
static constexpr int64_t LLAMA_KV_RESIDENT_KV_RUNS_MAX_CAP  = 512 * 1024;

struct llama_kv_delta_buf {
    struct slot_t {
        ggml_backend_event_t              uploaded       = nullptr;
        bool                              upload_pending = false;
        ggml_backend_buffer_t             stage_buf      = nullptr;
        uint8_t                         * stage_ptr      = nullptr;
        size_t                            stage_cap      = 0;
        std::vector<uint8_t>              stage_vec;
        std::vector<std::vector<uint8_t>> spill;
    };

    ggml_context          * ctx            = nullptr;
    ggml_backend_buffer_t   buffer         = nullptr;
    ggml_tensor           * tensor         = nullptr;
    ggml_backend_t          upload_backend = nullptr;
    slot_t                  slots[2]       = {};
    int                     cur_slot       = 0;
    size_t                  stage_used     = 0;
    bool                    batch_spans    = true;

    std::vector<ggml_tensor *> pending_tensors;
    std::vector<const void *>  pending_data;
    std::vector<size_t>        pending_offsets;
    std::vector<size_t>        pending_sizes;

    void release_slots() {
        for (auto & sl : slots) {
            if (sl.upload_pending && sl.uploaded) {
                ggml_backend_event_synchronize(sl.uploaded);
                sl.upload_pending = false;
            }
            if (sl.uploaded) {
                ggml_backend_event_free(sl.uploaded);
                sl.uploaded = nullptr;
            }
            if (sl.stage_buf) {
                ggml_backend_buffer_free(sl.stage_buf);
                sl.stage_buf = nullptr;
            }
            sl.stage_ptr = nullptr;
            sl.stage_cap = 0;
            sl.stage_vec.clear();
            sl.spill.clear();
        }
        cur_slot = 0;
        stage_used = 0;
    }

    ~llama_kv_delta_buf() {
        release_slots();
        if (buffer) {
            ggml_backend_buffer_free(buffer);
        }
        if (ctx) {
            ggml_free(ctx);
        }
    }

    void begin_upload(ggml_backend_t backend, size_t need_bytes = 0) {
        GGML_ASSERT(backend != nullptr);
        if (upload_backend != backend) {
            release_slots();
            upload_backend = backend;
            ggml_backend_dev_t dev = ggml_backend_get_device(backend);
            if (dev) {
                for (auto & sl : slots) {
                    sl.uploaded = ggml_backend_event_new(dev);
                }
            }
        }
        auto & sl = slots[cur_slot];
        if (sl.upload_pending && sl.uploaded) {
            ggml_backend_event_synchronize(sl.uploaded);
            sl.upload_pending = false;
        }
        if (need_bytes > sl.stage_cap) {
            const size_t new_cap = std::max<size_t>(need_bytes, std::max<size_t>(sl.stage_cap * 2, 65536));
            if (sl.stage_buf) {
                ggml_backend_buffer_free(sl.stage_buf);
                sl.stage_buf = nullptr;
                sl.stage_ptr = nullptr;
                sl.stage_cap = 0;
            }
            ggml_backend_dev_t dev = ggml_backend_get_device(backend);
            ggml_backend_buffer_type_t host_buft = dev ? ggml_backend_dev_host_buffer_type(dev) : nullptr;
            if (host_buft) {
                sl.stage_buf = ggml_backend_buft_alloc_buffer(host_buft, new_cap);
                if (sl.stage_buf) {
                    sl.stage_ptr = (uint8_t *) ggml_backend_buffer_get_base(sl.stage_buf);
                    sl.stage_cap = new_cap;
                }
            }
            if (!sl.stage_ptr) {
                sl.stage_vec.resize(new_cap);
                sl.stage_ptr = sl.stage_vec.data();
                sl.stage_cap = new_cap;
            }
        }
        stage_used = 0;
        sl.spill.clear();
        pending_tensors.clear();
        pending_data.clear();
        pending_offsets.clear();
        pending_sizes.clear();
    }

    void upload_to(ggml_tensor * dst, size_t byte_offset, const void * src, size_t nbytes) {
        if (nbytes == 0) {
            return;
        }
        auto & sl = slots[cur_slot];
        const size_t aligned = (stage_used + 15) & ~size_t(15);
        uint8_t * dst_stage = nullptr;
        if (sl.stage_ptr && aligned + nbytes <= sl.stage_cap) {
            dst_stage = sl.stage_ptr + aligned;
            stage_used = aligned + nbytes;
        } else {
            sl.spill.emplace_back(nbytes);
            dst_stage = sl.spill.back().data();
        }
        std::memcpy(dst_stage, src, nbytes);
        pending_tensors.push_back(dst);
        pending_data.push_back(dst_stage);
        pending_offsets.push_back(byte_offset);
        pending_sizes.push_back(nbytes);
    }

    void end_upload() {
        const size_t n = pending_tensors.size();
        if (n == 0) {
            return;
        }
        if (batch_spans && n >= 2) {
            ggml_backend_tensor_set_n_async(upload_backend, pending_tensors.data(), pending_data.data(),
                    pending_offsets.data(), pending_sizes.data(), n);
        } else {
            for (size_t i = 0; i < n; ++i) {
                ggml_backend_tensor_set_async(upload_backend, pending_tensors[i], pending_data[i],
                        pending_offsets[i], pending_sizes[i]);
            }
        }
        pending_tensors.clear();
        pending_data.clear();
        pending_offsets.clear();
        pending_sizes.clear();

        auto & sl = slots[cur_slot];
        if (sl.uploaded) {
            ggml_backend_event_record(sl.uploaded, upload_backend);
            sl.upload_pending = true;
            cur_slot ^= 1;
        } else {
            ggml_backend_synchronize(upload_backend);
            stage_used = 0;
            sl.spill.clear();
        }
    }
};

struct llama_kv_delta_state {
    llama_kv_delta_buf    cell_pos;
    bool                  cell_pos_synced = false;
    size_t                cell_pos_hi     = 0;
    std::vector<uint32_t> pos_dirty;

    bool                  hot_map_synced  = false;
    size_t                hot_map_hi      = 0;
    std::vector<int32_t>  hot_map_shadow;
    std::vector<uint32_t> hot_active;
    std::vector<uint32_t> hot_cand;
    std::vector<uint32_t> hot_plane_dirty;
    llama_pos             hot_seq_tip[LLAMA_MAX_SEQ] = {};

    struct kq_row_state {
        llama_seq_id seq   = -1;
        llama_pos    p_min = 0;
        bool         tip   = false;
        bool         valid = false;
    };
    llama_kv_delta_buf    kq_mask;
    size_t                kq_mask_n_words = 0;
    std::vector<uint16_t> kq_mask_shadow;
    kq_row_state          kq_rows[LLAMA_KV_RESIDENT_KQ_MASK_MAX_ROWS];
    std::vector<uint32_t> mask_dirty;
    std::vector<size_t>   dirty_words;

    llama_kv_delta_buf    kv_runs;
    int64_t               kv_runs_cap_entries = 0;
    int64_t               kv_runs_n_entries   = 0;
    std::vector<int32_t>  kv_runs_shadow;
    std::vector<int32_t>  kv_runs_tmp;
    std::vector<int32_t>  kv_runs_active;
    std::vector<int32_t>  kv_runs_slice_len;
    std::vector<uint8_t>  kv_runs_slice_valid;

    llama_kv_delta_buf    kv_runs_stage;
    std::vector<int32_t>  kv_runs_host;
};

ggml_backend_buffer_type_t llama_kv_cache::get_resident_buft() const {
    if (n_stream != 1 || other != nullptr || layers.empty() || delta_inputs_legacy) {
        return nullptr;
    }
    ggml_backend_buffer_type_t buft = nullptr;
    for (const auto & layer : layers) {
        if (!layer.k || !layer.k->buffer) {
            return nullptr;
        }
        ggml_backend_buffer_type_t b = ggml_backend_buffer_get_type(layer.k->buffer);
        if (ggml_backend_buft_is_host(b)) {
            return nullptr;
        }
        if (buft == nullptr) {
            buft = b;
        } else if (buft != b) {
            return nullptr;
        }
    }
    return buft;
}

void llama_kv_cache::invalidate_delta_state() const {
    if (!delta_state) {
        return;
    }
    const size_t n_cells = v_cells.empty() ? 0 : v_cells[0].size();
    delta_state->cell_pos_synced = false;
    delta_state->cell_pos_hi     = n_cells;
    delta_state->hot_map_synced  = false;
    delta_state->hot_map_hi      = n_cells;
    delta_state->kq_mask_n_words = 0;
    for (auto & rs : delta_state->kq_rows) {
        rs.valid = false;
    }
    std::fill(delta_state->kv_runs_slice_valid.begin(), delta_state->kv_runs_slice_valid.end(), uint8_t(0));
}

ggml_tensor * llama_kv_cache::get_resident_cell_pos() const {
    ggml_backend_buffer_type_t buft = get_resident_buft();
    if (!buft) {
        return nullptr;
    }
    if (!delta_state) {
        delta_state = std::make_shared<llama_kv_delta_state>();
    }
    auto & cp = delta_state->cell_pos;
    if (!cp.tensor) {
        const auto & cells = v_cells[0];
        cp.ctx = ggml_init({ ggml_tensor_overhead(), nullptr, true });
        if (!cp.ctx) {
            return nullptr;
        }
        cp.tensor = has_kv_mixed()
            ? ggml_new_tensor_3d(cp.ctx, GGML_TYPE_I32, cells.size(), 1, 2)
            : ggml_new_tensor_2d(cp.ctx, GGML_TYPE_I32, cells.size(), 1);
        ggml_set_name(cp.tensor, "kv_resident_cell_pos");
        cp.buffer = ggml_backend_alloc_ctx_tensors_from_buft(cp.ctx, buft);
        if (!cp.buffer) {
            ggml_free(cp.ctx);
            cp.ctx    = nullptr;
            cp.tensor = nullptr;
            return nullptr;
        }
        ggml_backend_buffer_clear(cp.buffer, 0xFF);
        cp.batch_spans = batch_delta_spans;
        cells.pos_changes().track(std::min<uint32_t>(cells.size(), std::max<uint32_t>(4096, cells.size()/32)));
        delta_state->cell_pos_synced = false;
        delta_state->cell_pos_hi     = 0;
        delta_state->hot_map_synced  = false;
        delta_state->hot_map_hi      = 0;
        std::fill(std::begin(delta_state->hot_seq_tip), std::end(delta_state->hot_seq_tip), -1);
    }
    return cp.tensor;
}

void llama_kv_cache::sync_resident_cell_pos(ggml_tensor * dst, ggml_backend_sched_t sched, int64_t n_kv) const {
    GGML_ASSERT(delta_state && delta_state->cell_pos.tensor);
    const auto & cells = v_cells[0];
    const auto & pos   = cells.positions();
    GGML_ASSERT(n_kv <= (int64_t) cells.size());

    delta_stats.cell_pos.calls++;

    cells.pos_changes().track(std::min<uint32_t>(cells.size(), std::max<uint32_t>(4096, cells.size()/32)));
    const bool p0_listed = cells.pos_changes().take(delta_state->pos_dirty) && delta_state->cell_pos_synced;

    ggml_backend_t backend = sched ? ggml_backend_sched_get_tensor_backend(sched, dst) : nullptr;
    if (!backend && sched) {
        backend = ggml_backend_sched_get_tensor_backend(sched, delta_state->cell_pos.tensor);
    }

    auto & cp = delta_state->cell_pos;
    const bool want_hot = dst && dst->ne[2] == 2 && has_kv_mixed();
    if (!want_hot && delta_state->hot_map_synced) {
        delta_state->hot_map_synced = false;
    }

    auto build_ranges = [](const std::vector<uint32_t> & d, std::vector<std::pair<size_t, size_t>> & out,
            size_t & raw_bytes, size_t & stage_bytes) {
        for (size_t idx = 0; idx < d.size(); ) {
            const size_t lo = d[idx];
            size_t hi = lo + 1;
            ++idx;
            while (idx < d.size() && d[idx] <= hi + 16) {
                hi = d[idx] + 1;
                ++idx;
            }
            const size_t nb = (hi - lo) * sizeof(int32_t);
            out.emplace_back(lo, hi);
            raw_bytes   += nb;
            stage_bytes += (nb + 15) & ~size_t(15);
        }
    };

    std::vector<std::pair<size_t, size_t>> p0_ranges, p1_ranges;
    size_t raw_bytes = 0;
    size_t stage_bytes = 0;
    bool did_full = false;

    // Plane 0: cell positions across [0 .. cells.size()), never colliding with plane 1 at cp.tensor->nb[2].
    if (!p0_listed) {
        const size_t hi = std::min<size_t>(cells.size(),
                std::max<size_t>({ (size_t) n_kv, (size_t) cells.used_max_p1(), delta_state->cell_pos_hi }));
        delta_state->cell_pos_hi     = cells.used_max_p1();
        delta_state->cell_pos_synced = true;
        if (hi > 0) {
            const size_t nb = hi * sizeof(llama_pos);
            p0_ranges.emplace_back(0, hi);
            raw_bytes   += nb;
            stage_bytes += (nb + 15) & ~size_t(15);
            did_full = true;
        }
    } else {
        auto & dirty = delta_state->pos_dirty;
        if (!dirty.empty()) {
            std::sort(dirty.begin(), dirty.end());
            dirty.erase(std::unique(dirty.begin(), dirty.end()), dirty.end());
            delta_state->cell_pos_hi = std::max<size_t>({ delta_state->cell_pos_hi, (size_t) cells.used_max_p1(), (size_t) dirty.back() + 1 });
            build_ranges(dirty, p0_ranges, raw_bytes, stage_bytes);
        }
    }

    // Plane 1 (kv_mixed hot_map): fixed offset at cp.tensor->nb[2], independent of n_kv.
    const size_t plane1_off = want_hot ? cp.tensor->nb[2] : 0;
    if (want_hot) {
        if (delta_state->hot_map_shadow.size() != cells.size()) {
            delta_state->hot_map_shadow.assign(cells.size(), -1);
            delta_state->hot_active.clear();
            delta_state->hot_map_synced = false;
            delta_state->hot_map_hi     = cells.size();
        }
        const auto & s2c = kv_mixed_slot_to_cell[0];
        const auto & c2s = kv_mixed_cell_to_slot[0];
        auto want_slot = [&](uint32_t c) -> int32_t {
            if (c >= cells.size()) {
                return -1;
            }
            const int32_t slot = c2s[c];
            return (slot >= 0 && (uint32_t) slot < kv_mixed_hot_rows && s2c[slot] == c && kv_mixed_is_near(0, c))
                ? slot : -1;
        };

        if (!p0_listed || !delta_state->hot_map_synced) {
            for (const uint32_t c : delta_state->hot_active) {
                delta_state->hot_map_shadow[c] = -1;
            }
            delta_state->hot_active.clear();
            size_t max_hot_p1 = 0;
            for (uint32_t slot = 0; slot < kv_mixed_hot_rows; ++slot) {
                const uint32_t c = s2c[slot];
                if (c < cells.size() && c2s[c] == (int32_t) slot && kv_mixed_is_near(0, c)) {
                    delta_state->hot_map_shadow[c] = (int32_t) slot;
                    delta_state->hot_active.push_back(c);
                    max_hot_p1 = std::max<size_t>(max_hot_p1, (size_t) c + 1);
                }
            }
            for (uint32_t s = 0; s < LLAMA_MAX_SEQ; ++s) {
                delta_state->hot_seq_tip[s] = cells.seq_pos_max(s);
            }
            kv_mixed_dirty.clear();
            kv_mixed_dirty_all = false;
            const size_t hi_hot = std::min<size_t>(cells.size(),
                    std::max<size_t>({ (size_t) n_kv, max_hot_p1, delta_state->hot_map_hi }));
            delta_state->hot_map_hi     = max_hot_p1;
            delta_state->hot_map_synced = true;
            if (hi_hot > 0) {
                const size_t nb = hi_hot * sizeof(int32_t);
                p1_ranges.emplace_back(0, hi_hot);
                raw_bytes   += nb;
                stage_bytes += (nb + 15) & ~size_t(15);
                did_full = true;
            }
        } else {
            auto & cand = delta_state->hot_cand;
            cand.clear();
            if (kv_mixed_dirty_all) {
                cand.insert(cand.end(), delta_state->hot_active.begin(), delta_state->hot_active.end());
                for (uint32_t slot = 0; slot < kv_mixed_hot_rows; ++slot) {
                    if (s2c[slot] < cells.size()) {
                        cand.push_back(s2c[slot]);
                    }
                }
            } else {
                cand.swap(kv_mixed_dirty);
                const llama_pos ext = (llama_pos) kv_mixed_extent;
                bool seq_emptied = false;
                for (uint32_t s = 0; s < LLAMA_MAX_SEQ; ++s) {
                    const llama_pos old_tip = delta_state->hot_seq_tip[s];
                    const llama_pos new_tip = cells.seq_pos_max(s);
                    if (new_tip == old_tip) {
                        continue;
                    }
                    if (new_tip < 0) {
                        seq_emptied = true;
                        continue;
                    }
                    const llama_pos lo = old_tip < 0 ? std::max<llama_pos>(0, new_tip - ext + 1)
                        : std::max<llama_pos>(0, std::min(old_tip, new_tip) - ext + 1);
                    const llama_pos hi = old_tip < 0 ? new_tip + 1 : std::max(old_tip, new_tip) - ext + 1;
                    if (hi > lo) {
                        const auto & sp = cells.seq_pos_get(s);
                        for (auto it = sp.lower_bound({ lo, 0 }); it != sp.end() && it->first < hi; ++it) {
                            const uint32_t c = it->second;
                            if (c < cells.size() && (c2s[c] >= 0 || delta_state->hot_map_shadow[c] >= 0)) {
                                cand.push_back(c);
                            }
                        }
                    }
                }
                if (seq_emptied) {
                    cand.insert(cand.end(), delta_state->hot_active.begin(), delta_state->hot_active.end());
                }
            }
            for (uint32_t s = 0; s < LLAMA_MAX_SEQ; ++s) {
                delta_state->hot_seq_tip[s] = cells.seq_pos_max(s);
            }
            kv_mixed_dirty.clear();
            kv_mixed_dirty_all = false;

            auto & p1_dirty = delta_state->hot_plane_dirty;
            p1_dirty.clear();
            if (!cand.empty()) {
                std::sort(cand.begin(), cand.end());
                cand.erase(std::unique(cand.begin(), cand.end()), cand.end());
                for (const uint32_t c : cand) {
                    if (c >= cells.size()) {
                        continue;
                    }
                    const int32_t old_v = delta_state->hot_map_shadow[c];
                    const int32_t new_v = want_slot(c);
                    if (new_v != old_v) {
                        delta_state->hot_map_shadow[c] = new_v;
                        if (old_v < 0 && new_v >= 0) {
                            delta_state->hot_active.push_back(c);
                        }
                        p1_dirty.push_back(c);
                    }
                }
                if (delta_state->hot_active.size() > 2 * (size_t) kv_mixed_hot_rows) {
                    auto & ha = delta_state->hot_active;
                    ha.erase(std::remove_if(ha.begin(), ha.end(), [&](uint32_t c) {
                        return delta_state->hot_map_shadow[c] < 0;
                    }), ha.end());
                }
            }
            if (!p1_dirty.empty()) {
                delta_state->hot_map_hi = std::max<size_t>(delta_state->hot_map_hi, (size_t) p1_dirty.back() + 1);
                build_ranges(p1_dirty, p1_ranges, raw_bytes, stage_bytes);
            }
        }
    }

    const size_t n_spans = p0_ranges.size() + p1_ranges.size();
    if (n_spans == 0) {
        return;
    }

    if (did_full) {
        delta_stats.cell_pos.full_uploads++;
    } else {
        delta_stats.cell_pos.delta_uploads++;
    }
    delta_stats.cell_pos.spans += n_spans;
    delta_stats.cell_pos.bytes += raw_bytes;

    if (backend) {
        cp.begin_upload(backend, stage_bytes);
        for (const auto & [lo, hi] : p0_ranges) {
            cp.upload_to(cp.tensor, lo * sizeof(llama_pos), pos.data() + lo, (hi - lo) * sizeof(llama_pos));
        }
        for (const auto & [lo, hi] : p1_ranges) {
            cp.upload_to(cp.tensor, plane1_off + lo * sizeof(int32_t), delta_state->hot_map_shadow.data() + lo, (hi - lo) * sizeof(int32_t));
        }
        cp.end_upload();
    } else {
        for (const auto & [lo, hi] : p0_ranges) {
            ggml_backend_tensor_set(cp.tensor, pos.data() + lo, lo * sizeof(llama_pos), (hi - lo) * sizeof(llama_pos));
        }
        for (const auto & [lo, hi] : p1_ranges) {
            ggml_backend_tensor_set(cp.tensor, delta_state->hot_map_shadow.data() + lo, plane1_off + lo * sizeof(int32_t), (hi - lo) * sizeof(int32_t));
        }
    }
}

ggml_tensor * llama_kv_cache::get_resident_kq_mask(int64_t n_kv, int64_t n_tokens) const {
    if (n_tokens <= 0 || n_tokens > LLAMA_KV_RESIDENT_KQ_MASK_MAX_ROWS || n_kv <= 0 || n_kv % 16 != 0) {
        return nullptr;
    }
    ggml_backend_buffer_type_t buft = get_resident_buft();
    if (!buft) {
        return nullptr;
    }
    if (!delta_state) {
        delta_state = std::make_shared<llama_kv_delta_state>();
    }
    auto & km = delta_state->kq_mask;
    if (!km.tensor) {
        const auto & cells = v_cells[0];
        const int64_t max_words = (cells.size() + 15) / 16;
        km.ctx = ggml_init({ ggml_tensor_overhead(), nullptr, true });
        if (!km.ctx) {
            return nullptr;
        }
        km.tensor = ggml_new_tensor_1d(km.ctx, GGML_TYPE_I16, max_words * LLAMA_KV_RESIDENT_KQ_MASK_MAX_ROWS);
        ggml_set_name(km.tensor, "kv_resident_kq_mask");
        km.buffer = ggml_backend_alloc_ctx_tensors_from_buft(km.ctx, buft);
        if (!km.buffer) {
            ggml_free(km.ctx);
            km.ctx    = nullptr;
            km.tensor = nullptr;
            return nullptr;
        }
        ggml_backend_buffer_clear(km.buffer, 0);
        km.batch_spans = batch_delta_spans;
        cells.mask_changes().track(std::min<uint32_t>(cells.size(), std::max<uint32_t>(4096, cells.size()/32)));
        delta_state->kq_mask_n_words = 0;
        for (auto & rs : delta_state->kq_rows) {
            rs = {};
        }
    }
    return km.tensor;
}

bool llama_kv_cache::is_resident_kq_mask(const ggml_tensor * dst) const {
    return dst && delta_state && delta_state->kq_mask.tensor && dst->view_src == delta_state->kq_mask.tensor;
}

void llama_kv_cache::sync_resident_kq_mask(ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn, bool block_uniform, ggml_backend_sched_t sched) const {
    GGML_ASSERT(delta_state && delta_state->kq_mask.tensor && dst->type == GGML_TYPE_I16 && dst->ne[3] == 1);
    const uint32_t n_tokens = ubatch->n_tokens;
    GGML_ASSERT(n_tokens > 0 && n_tokens <= LLAMA_KV_RESIDENT_KQ_MASK_MAX_ROWS);

    delta_stats.kq_mask.calls++;

    const size_t n_words = (size_t) dst->ne[0];
    const auto & cells   = v_cells[0];

    if (!causal_attn && hparams.non_causal_type == LLAMA_NON_CAUSAL_TYPE_SWA_ONLY) {
        causal_attn = swa_type == LLAMA_SWA_TYPE_NONE;
    }
    if (!causal_attn && hparams.non_causal_type == LLAMA_NON_CAUSAL_TYPE_DENSE_ONLY) {
        causal_attn = swa_type != LLAMA_SWA_TYPE_NONE;
    }

    ggml_backend_t backend = sched ? ggml_backend_sched_get_tensor_backend(sched, dst) : nullptr;
    if (!backend && sched) {
        backend = ggml_backend_sched_get_tensor_backend(sched, delta_state->kq_mask.tensor);
    }

    auto & km     = delta_state->kq_mask;
    auto & shadow = delta_state->kq_mask_shadow;

    cells.mask_changes().track(std::min<uint32_t>(cells.size(), std::max<uint32_t>(4096, cells.size()/32)));
    const bool listed = cells.mask_changes().take(delta_state->mask_dirty);

    const bool standard_window = causal_attn && swa_type == LLAMA_SWA_TYPE_STANDARD &&
            n_swa > 0 && n_swa <= INT32_MAX && !block_uniform;
    bool tip_decode = causal_attn && !hparams.use_alibi && !ubatch->is_pos_2d() && !block_uniform &&
            (swa_type == LLAMA_SWA_TYPE_NONE || standard_window);
    for (uint32_t i = 0; tip_decode && i < n_tokens; ++i) {
        if (ubatch->n_seq_id[i] != 1) {
            tip_decode = false;
            break;
        }
        const auto seq = ubatch->seq_id[i][0];
        if (ubatch->pos[i] < cells.seq_pos_max(seq)) {
            tip_decode = false;
            break;
        }
    }

    if (!tip_decode) {
        // non-tip step (drafted or small prompt): build into host shadow and upload asynchronously
        shadow.resize(n_words * LLAMA_KV_RESIDENT_KQ_MASK_MAX_ROWS);
        delta_state->kq_mask_n_words = n_words;
        for (auto & rs : delta_state->kq_rows) {
            rs.valid = false;
        }
        ggml_tensor host_view = *dst;
        host_view.data        = shadow.data();
        host_view.view_src    = nullptr;
        static const bool scan_cells = ggml_env_switch("LLAMA_KQ_MASK_SCAN_LEGACY");
        static const bool seq_bits   = !ggml_env_switch("LLAMA_KQ_MASK_SEQ_BITS_LEGACY");
        llama_kv_cache_set_input_kq_mask(&host_view, ubatch, causal_attn, hparams, v_cells, seq_to_stream, n_swa, swa_type,
                scan_cells, seq_bits, nullptr, block_uniform);
        const size_t nbytes = n_tokens * n_words * sizeof(uint16_t);
        delta_stats.kq_mask.full_uploads++;
        delta_stats.kq_mask.spans++;
        delta_stats.kq_mask.bytes += nbytes;
        if (backend) {
            km.begin_upload(backend, nbytes);
            km.upload_to(km.tensor, 0, shadow.data(), nbytes);
            km.end_upload();
        } else {
            ggml_backend_tensor_set(km.tensor, shadow.data(), 0, nbytes);
        }
        return;
    }

    if (delta_state->kq_mask_n_words != n_words || !listed) {
        shadow.resize(n_words * LLAMA_KV_RESIDENT_KQ_MASK_MAX_ROWS);
        delta_state->kq_mask_n_words = n_words;
        for (auto & rs : delta_state->kq_rows) {
            rs.valid = false;
        }
    }

    auto & dirty = delta_state->mask_dirty;
    if (listed && !dirty.empty()) {
        std::sort(dirty.begin(), dirty.end());
        dirty.erase(std::unique(dirty.begin(), dirty.end()), dirty.end());
        // Preserve cached inactive rows unless a dirty cell actually flips one of their bits.
        for (uint32_t k = n_tokens; k < LLAMA_KV_RESIDENT_KQ_MASK_MAX_ROWS; ++k) {
            auto & rs = delta_state->kq_rows[k];
            if (!rs.valid) {
                continue;
            }
            const uint16_t * row_ptr = shadow.data() + k * n_words;
            for (const uint32_t cell : dirty) {
                const size_t w = cell / 16;
                if (w >= n_words) {
                    continue;
                }
                const uint16_t mask_bit = uint16_t(1u << (cell % 16));
                const bool was_set = (row_ptr[w] & mask_bit) != 0;
                const bool live = !cells.is_empty(cell) && cells.seq_has(cell, rs.seq) && cells.pos_get(cell) >= rs.p_min;
                if (live != was_set) {
                    rs.valid = false;
                    break;
                }
            }
        }
    }

    struct upload_span {
        size_t byte_off;
        const uint16_t * ptr;
        size_t nbytes;
    };
    std::vector<upload_span> spans;
    size_t raw_bytes   = 0;
    size_t total_bytes = 0;
    bool did_full = false;

    for (uint32_t r = 0; r < n_tokens; ++r) {
        const llama_seq_id seq = ubatch->seq_id[r][0];
        const llama_pos p_min  = standard_window ? std::max<int64_t>(0, int64_t(ubatch->pos[r]) + 1 - n_swa) : 0;
        uint16_t * row_ptr     = shadow.data() + r * n_words;
        auto & rs              = delta_state->kq_rows[r];

        if (rs.valid && rs.tip && rs.seq == seq && rs.p_min <= p_min) {
            auto & dw = delta_state->dirty_words;
            dw.clear();
            for (const uint32_t cell : dirty) {
                const size_t w = cell / 16;
                if (w >= n_words) {
                    continue;
                }
                const uint16_t mask_bit = uint16_t(1u << (cell % 16));
                const bool live = !cells.is_empty(cell) && cells.seq_has(cell, seq) && cells.pos_get(cell) >= p_min;
                const uint16_t old_v = row_ptr[w];
                const uint16_t new_v = live ? uint16_t(old_v | mask_bit) : uint16_t(old_v & ~mask_bit);
                if (new_v != old_v) {
                    row_ptr[w] = new_v;
                    dw.push_back(w);
                }
            }
            if (standard_window && p_min > rs.p_min) {
                const auto & sp = cells.seq_pos_get(seq);
                for (auto it = sp.lower_bound({ rs.p_min, 0 }); it != sp.end() && it->first < p_min; ++it) {
                    const uint32_t cell = it->second;
                    const size_t w = cell / 16;
                    if (w >= n_words) {
                        continue;
                    }
                    const uint16_t mask_bit = uint16_t(1u << (cell % 16));
                    if (row_ptr[w] & mask_bit) {
                        row_ptr[w] &= uint16_t(~mask_bit);
                        dw.push_back(w);
                    }
                }
            }
            rs.p_min = p_min;
            if (!dw.empty()) {
                std::sort(dw.begin(), dw.end());
                dw.erase(std::unique(dw.begin(), dw.end()), dw.end());
                for (size_t idx = 0; idx < dw.size(); ) {
                    const size_t w0 = dw[idx];
                    size_t w1 = w0 + 1;
                    ++idx;
                    while (idx < dw.size() && dw[idx] <= w1 + 8) {
                        w1 = dw[idx] + 1;
                        ++idx;
                    }
                    const size_t nb = (w1 - w0) * sizeof(uint16_t);
                    spans.push_back({ (r * n_words + w0) * sizeof(uint16_t), row_ptr + w0, nb });
                    raw_bytes   += nb;
                    total_bytes += (nb + 15) & ~size_t(15);
                }
            }
        } else {
            cells.copy_seq_mask(seq, row_ptr, n_words, p_min, n_swa);
            rs = { seq, p_min, true, true };
            const size_t nb = n_words * sizeof(uint16_t);
            spans.push_back({ r * n_words * sizeof(uint16_t), row_ptr, nb });
            raw_bytes   += nb;
            total_bytes += (nb + 15) & ~size_t(15);
            did_full = true;
        }
    }

    if (spans.empty()) {
        return;
    }
    if (did_full) {
        delta_stats.kq_mask.full_uploads++;
    } else {
        delta_stats.kq_mask.delta_uploads++;
    }
    delta_stats.kq_mask.spans += spans.size();
    delta_stats.kq_mask.bytes += raw_bytes;

    if (backend) {
        km.begin_upload(backend, total_bytes);
        for (const auto & sp : spans) {
            km.upload_to(km.tensor, sp.byte_off, sp.ptr, sp.nbytes);
        }
        km.end_upload();
    } else {
        for (const auto & sp : spans) {
            ggml_backend_tensor_set(km.tensor, sp.ptr, sp.byte_off, sp.nbytes);
        }
    }
}

ggml_tensor * llama_kv_cache::get_resident_kv_runs(int64_t n_entries, int64_t n_rows_p1, int64_t cap_rows_p1) const {
    if (n_entries < 2 || n_rows_p1 < 2) {
        return nullptr;
    }
    ggml_backend_buffer_type_t buft = get_resident_buft();
    if (!buft) {
        return nullptr;
    }
    const int64_t need = n_entries * n_rows_p1;
    if (need > LLAMA_KV_RESIDENT_KV_RUNS_MAX_CAP) {
        return nullptr;
    }
    if (!delta_state) {
        delta_state = std::make_shared<llama_kv_delta_state>();
    }
    auto & kr = delta_state->kv_runs;
    if (!kr.tensor) {
        const int64_t min_rows = std::max<int64_t>(cap_rows_p1, 65);
        const int64_t cap = std::min<int64_t>(LLAMA_KV_RESIDENT_KV_RUNS_MAX_CAP,
                std::max<int64_t>(need, n_entries * min_rows));
        kr.ctx = ggml_init({ ggml_tensor_overhead(), nullptr, true });
        if (!kr.ctx) {
            return nullptr;
        }
        kr.tensor = ggml_new_tensor_1d(kr.ctx, GGML_TYPE_I32, 4 * cap);
        ggml_set_name(kr.tensor, "kv_resident_kv_runs");
        kr.buffer = ggml_backend_alloc_ctx_tensors_from_buft(kr.ctx, buft);
        if (!kr.buffer) {
            ggml_free(kr.ctx);
            kr.ctx    = nullptr;
            kr.tensor = nullptr;
            return nullptr;
        }
        ggml_backend_buffer_clear(kr.buffer, 0);
        kr.batch_spans = batch_delta_spans;
        delta_state->kv_runs_cap_entries = cap;
        delta_state->kv_runs_n_entries   = n_entries;
        delta_state->kv_runs_shadow.assign((size_t) (4 * cap), 0);
        const size_t max_slices = (size_t) (cap / n_entries);
        delta_state->kv_runs_slice_len.assign(max_slices, 0);
        delta_state->kv_runs_slice_valid.assign(max_slices, uint8_t(1));
    }
    if (need > delta_state->kv_runs_cap_entries) {
        return nullptr;
    }
    return kr.tensor;
}

bool llama_kv_cache::is_resident_kv_runs(const ggml_tensor * dst) const {
    return dst && delta_state && delta_state->kv_runs.tensor && dst->view_src == delta_state->kv_runs.tensor;
}

void llama_kv_cache::sync_resident_kv_runs(ggml_tensor * dst, const llama_ubatch * ubatch, ggml_backend_sched_t sched) const {
    GGML_ASSERT(delta_state && delta_state->kv_runs.tensor && kv_runs);
    const int64_t n_entries = dst->ne[1];
    const int64_t n_rows_p1 = dst->ne[2];
    GGML_ASSERT(n_entries * n_rows_p1 <= delta_state->kv_runs_cap_entries);

    delta_stats.kv_runs.calls++;

    if (delta_state->kv_runs_n_entries != n_entries) {
        delta_state->kv_runs_n_entries = n_entries;
        const size_t max_slices = (size_t) (delta_state->kv_runs_cap_entries / n_entries);
        delta_state->kv_runs_slice_len.assign(max_slices, 0);
        delta_state->kv_runs_slice_valid.assign(max_slices, uint8_t(0));
    }

    auto & tmp    = delta_state->kv_runs_tmp;
    auto & active = delta_state->kv_runs_active;
    auto & shadow = delta_state->kv_runs_shadow;
    tmp.resize((size_t) (4 * n_entries * n_rows_p1));
    kv_runs->write_active(v_cells[0], *ubatch, tmp.data(), n_entries, active);

    struct upload_span {
        size_t          byte_off;
        const int32_t * ptr;
        size_t          nbytes;
    };
    std::vector<upload_span> spans;
    size_t raw_bytes   = 0;
    size_t stage_bytes = 0;
    bool did_full = false;
    static const int32_t zero4[4] = { 0, 0, 0, 0 };

    for (int64_t s = 0; s < n_rows_p1; ++s) {
        const int32_t cur_len = active[(size_t) s];
        int32_t * sh = shadow.data() + 4 * n_entries * s;
        const int32_t * tm = tmp.data() + 4 * n_entries * s;

        if (!delta_state->kv_runs_slice_valid[(size_t) s]) {
            std::fill_n(sh, (size_t) (4 * n_entries), 0);
            std::memcpy(sh, tm, (size_t) cur_len * 4 * sizeof(int32_t));
            delta_state->kv_runs_slice_valid[(size_t) s] = 1;
            delta_state->kv_runs_slice_len[(size_t) s]   = cur_len;
            const size_t nb = (size_t) n_entries * 4 * sizeof(int32_t);
            spans.push_back({ (size_t) (s * n_entries) * 16, sh, nb });
            raw_bytes   += nb;
            stage_bytes += nb;
            did_full = true;
            continue;
        }

        const int32_t old_len  = delta_state->kv_runs_slice_len[(size_t) s];
        const int32_t scan_len = std::max(cur_len, old_len);
        delta_state->kv_runs_slice_len[(size_t) s] = cur_len;

        int32_t span_lo = -1;
        int32_t span_hi = -1;
        auto flush_span = [&]() {
            if (span_lo < 0) {
                return;
            }
            const size_t nb = (size_t) (span_hi - span_lo) * 16;
            spans.push_back({ (size_t) (s * n_entries + span_lo) * 16, sh + 4 * span_lo, nb });
            raw_bytes   += nb;
            stage_bytes += nb;
            span_lo = -1;
            span_hi = -1;
        };

        for (int32_t e = 0; e < scan_len; ++e) {
            const int32_t * want = e < cur_len ? (tm + 4 * e) : zero4;
            int32_t * cur = sh + 4 * e;
            if (cur[0] != want[0] || cur[1] != want[1] || cur[2] != want[2] || cur[3] != want[3]) {
                cur[0] = want[0];
                cur[1] = want[1];
                cur[2] = want[2];
                cur[3] = want[3];
                if (span_lo < 0) {
                    span_lo = e;
                    span_hi = e + 1;
                } else if (e <= span_hi + 2) {
                    span_hi = e + 1;
                } else {
                    flush_span();
                    span_lo = e;
                    span_hi = e + 1;
                }
            }
        }
        flush_span();
    }

    if (spans.empty()) {
        return;
    }
    if (did_full) {
        delta_stats.kv_runs.full_uploads++;
    } else {
        delta_stats.kv_runs.delta_uploads++;
    }
    delta_stats.kv_runs.spans += spans.size();
    delta_stats.kv_runs.bytes += raw_bytes;

    auto & kr = delta_state->kv_runs;
    ggml_backend_t backend = sched ? ggml_backend_sched_get_tensor_backend(sched, dst) : nullptr;
    if (!backend && sched) {
        backend = ggml_backend_sched_get_tensor_backend(sched, kr.tensor);
    }
    if (backend) {
        kr.begin_upload(backend, stage_bytes);
        for (const auto & sp : spans) {
            kr.upload_to(kr.tensor, sp.byte_off, sp.ptr, sp.nbytes);
        }
        kr.end_upload();
    } else {
        for (const auto & sp : spans) {
            ggml_backend_tensor_set(kr.tensor, sp.ptr, sp.byte_off, sp.nbytes);
        }
    }
}

void llama_kv_cache::set_input_attn_mask(const llm_attn_mask & attn_mask, const llama_ubatch * ubatch, bool causal_attn, ggml_backend_sched_t sched) const {
    // fill the cell rows of each slice
    if (attn_mask.rows && attn_mask.rows->buffer) {
        GGML_ASSERT(ggml_backend_buffer_is_host(attn_mask.rows->buffer));
        GGML_ASSERT(n_stream == 1);

        const int64_t n_rows   = attn_mask.rows->ne[0];
        const int64_t n_slices = attn_mask.rows->ne[1];

        const auto & cells = v_cells[0];
        int32_t * data = (int32_t *) attn_mask.rows->data;

        cells.get_seq_cells(get_kv_rows_slices(*ubatch, n_slices, n_swa, swa_type), data, n_rows);
    }

    // the mask is left unallocated when the graph only stores K/V without attending
    if (attn_mask.raw && attn_mask.raw->buffer) {
        ggml_tensor * dst = attn_mask.raw;
        const int32_t * rows = attn_mask.rows && attn_mask.rows->buffer ? (const int32_t *) attn_mask.rows->data : nullptr;

        if (!rows && is_resident_kq_mask(dst)) {
            sync_resident_kq_mask(dst, ubatch, causal_attn, attn_mask.block_uniform, sched);
            return;
        }

        GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));

        if (rows) {
            const int64_t n_kv = dst->type == GGML_TYPE_I16 ? 16*dst->ne[0] : dst->ne[0]; // a bit-packed mask holds 16 cells per element
            GGML_ASSERT(attn_mask.rows->ne[0] == n_kv && attn_mask.rows->ne[1] == dst->ne[3]);
        }

        //const int64_t t_start = ggml_time_us();

        static const bool scan_cells = ggml_env_switch("LLAMA_KQ_MASK_SCAN_LEGACY");

        static const bool seq_bits = !ggml_env_switch("LLAMA_KQ_MASK_SEQ_BITS_LEGACY");

        llama_kv_cache_set_input_kq_mask(dst, ubatch, causal_attn, hparams, v_cells, seq_to_stream, n_swa, swa_type, scan_cells, seq_bits,
                rows, attn_mask.block_uniform);

        delta_stats.kq_mask.calls++;
        delta_stats.kq_mask.full_uploads++;
        delta_stats.kq_mask.spans++;
        delta_stats.kq_mask.bytes += ggml_nbytes(dst);

        //const int64_t t_end = ggml_time_us();

        //LLAMA_LOG_ERROR("%s: kq mask time: %0.3f ms\n", __func__, (t_end - t_start)/1000.0);
    }
}

void llama_kv_cache::set_input_pos_bucket(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    const int64_t n_tokens = ubatch->n_tokens;

    GGML_ASSERT(n_stream == 1 && "TODO: support multiple streams");
    const auto & cells = v_cells[0];

    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));
    GGML_ASSERT(!ubatch->equal_seqs()); // TODO: use ubatch->n_seqs instead of failing

    int32_t * data = (int32_t *) dst->data;

    const int32_t n_kv = dst->ne[0];

    for (int h = 0; h < 1; ++h) {
        for (int i = 0; i < n_tokens; ++i) {
            for (int j = 0; j < n_kv; ++j) {
                // the position when the cells is empty is irrelevant - it will be masked out later in the attention
                const llama_pos p0 = cells.is_empty(j) ? -1 : cells.pos_get(j);

                data[h*(n_kv*n_tokens) + i*n_kv + j] = llama_relative_position_bucket(p0, ubatch->pos[i], hparams.n_rel_attn_bkts, false);
            }
        }
    }
}

bool llama_kv_cache::has_cell_ext() const {
    // M-RoPE needs the 2D position, the PLE / Engram n-gram hashes need the token id
    return hparams.n_pos_per_embd() > 1 || hparams.ple_n_heads > 0;
}

void llama_kv_cache::set_input_relative_pos(ggml_tensor * dst, const llama_ubatch * ubatch, uint32_t extent) const {
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    const int64_t n_kv = dst->ne[0];
    std::vector<int32_t> data(n_kv * ubatch->n_tokens);
    for (uint32_t i = 0; i < ubatch->n_tokens; ++i) {
        const auto & cells = v_cells.at(seq_to_stream.at(ubatch->seq_id[i][0]));
        for (int64_t j = 0; j < n_kv; ++j) {
            const int64_t distance = cells.is_empty(j) ? -1 : int64_t(ubatch->pos[i]) - cells.pos_get(j);
            data[i*n_kv + j] = distance >= 0 && distance < extent ? distance : extent;
        }
    }
    ggml_backend_tensor_set(dst, data.data(), 0, ggml_nbytes(dst));
}

void llama_kv_cache::set_input_cell_pos(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    GGML_ASSERT(dst->type == GGML_TYPE_I32 && ubatch->n_tokens % dst->ne[1] == 0);
    const int64_t n_kv  = dst->ne[0];
    const int64_t n_tps = ubatch->n_tokens/dst->ne[1];
    std::vector<int32_t> hot_map;
    if (dst->ne[2] == 2) {
        hot_map.resize(n_kv);
    }
    for (int64_t s = 0; s < dst->ne[1]; ++s) {
        // a stream of the ubatch is the stream of its first token's sequence, as in the KQ mask
        const uint32_t strm = seq_to_stream.at(ubatch->seq_id[s*n_tps][0]);
        const auto & cells = v_cells.at(strm);
        GGML_ASSERT(n_kv <= cells.size());
        ggml_backend_tensor_set(dst, cells.positions().data(), s*dst->nb[1], n_kv*sizeof(llama_pos));
        delta_stats.cell_pos.spans++;
        delta_stats.cell_pos.bytes += (size_t) n_kv * sizeof(llama_pos);
        if (dst->ne[2] == 2) {
            std::fill(hot_map.begin(), hot_map.end(), -1);
            if (has_kv_mixed() && strm < kv_mixed_slot_to_cell.size()) {
                const auto & s2c = kv_mixed_slot_to_cell[strm];
                const auto & c2s = kv_mixed_cell_to_slot[strm];
                for (uint32_t slot = 0; slot < kv_mixed_hot_rows; ++slot) {
                    const uint32_t cell = s2c[slot];
                    if (cell < (uint32_t) n_kv && c2s[cell] == (int32_t) slot && kv_mixed_is_near(strm, cell)) {
                        hot_map[cell] = (int32_t) slot;
                    }
                }
            }
            ggml_backend_tensor_set(dst, hot_map.data(), s*dst->nb[1] + dst->nb[2], n_kv*sizeof(int32_t));
            delta_stats.cell_pos.spans++;
            delta_stats.cell_pos.bytes += (size_t) n_kv * sizeof(int32_t);
        }
    }
    delta_stats.cell_pos.calls++;
    delta_stats.cell_pos.full_uploads++;
}

bool llama_kv_cache::kv_runs_fit(const llama_ubatch & ubatch, uint32_t n_max) const {
    // the runs follow the cells through their change log: a cache that reads another's cells, or the pooled keys that
    // take the log too, have none
    if (n_stream != 1 || other || hparams.indexer_kpool > 0) {
        return false;
    }
    if (!kv_runs) {
        kv_runs = std::make_shared<llama_kv_runs>(n_swa, swa_type);
    }
    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        if (!kv_runs->get(v_cells[0], ubatch.seq_id[i][0], n_max)) {
            return false;
        }
    }
    return true;
}

void llama_kv_cache::set_input_kv_runs(ggml_tensor * dst, const llama_ubatch * ubatch, ggml_backend_sched_t sched) const {
    GGML_ASSERT(kv_runs && dst->type == GGML_TYPE_I32 && dst->ne[0] == 4 && dst->ne[2] == ubatch->n_tokens + 1);
    if (is_resident_kv_runs(dst)) {
        sync_resident_kv_runs(dst, ubatch, sched);
        return;
    }
    const size_t nbytes = ggml_nbytes(dst);
    delta_stats.kv_runs.calls++;
    delta_stats.kv_runs.full_uploads++;
    delta_stats.kv_runs.spans++;
    delta_stats.kv_runs.bytes += nbytes;
    if (!delta_state) {
        delta_state = std::make_shared<llama_kv_delta_state>();
    }
    auto & data = delta_state->kv_runs_host;
    data.resize(ggml_nelements(dst));
    kv_runs->write(v_cells[0], *ubatch, data.data(), dst->ne[1]);
    ggml_backend_t backend = sched ? ggml_backend_sched_get_tensor_backend(sched, dst) : nullptr;
    if (backend && dst->buffer && !ggml_backend_buffer_is_host(dst->buffer)) {
        auto & st = delta_state->kv_runs_stage;
        st.batch_spans = batch_delta_spans;
        st.begin_upload(backend, nbytes);
        st.upload_to(dst, 0, data.data(), nbytes);
        st.end_upload();
    } else {
        ggml_backend_tensor_set(dst, data.data(), 0, nbytes);
    }
}

void llama_kv_cache::write_kv_runs_host(const llama_ubatch & ubatch, int32_t * data, int64_t n_entries) const {
    GGML_ASSERT(kv_runs);
    kv_runs->write(v_cells[0], ubatch, data, n_entries);
}

void llama_kv_cache::get_prev_tokens(const llama_ubatch & ubatch, uint32_t n, std::vector<llama_token> & res) const {
    const uint32_t n_tokens = ubatch.n_tokens;

    res.clear();
    res.resize(n_tokens*n, LLAMA_TOKEN_NULL);

    if (n == 0) {
        return;
    }

    // note: apply_ubatch() has already stored the current ubatch, so the cells cover the tokens
    //       of this very ubatch as well, which is what we want
    // the nearest cell at or before a position also resolves M-RoPE gaps, where multiple tokens
    // share the same temporal pos

    // an embd (multimodal) ubatch can repeat one position for a whole image, so positions
    // do not encode the token order; resolve its predecessors by ubatch order instead
    // same for a mixed ubatch
    const bool by_order = !ubatch.token || ubatch.is_mixed();

    std::vector<uint32_t> ord; // index among the ubatch tokens of the same seq
    std::unordered_map<llama_seq_id, std::vector<uint32_t>> seq_idx;

    if (by_order) {
        ord.resize(n_tokens);
        for (uint32_t i = 0; i < n_tokens; ++i) {
            auto & v = seq_idx[ubatch.seq_id[i][0]];
            ord[i] = v.size();
            v.push_back(i);
        }
    }

    for (uint32_t i = 0; i < n_tokens; ++i) {
        // TODO: a token that belongs to more than one sequence has an ambiguous history.
        //       the n-gram architectures have to reject such batches
        const llama_seq_id seq_id = ubatch.seq_id[i][0];

        for (uint32_t j = 0; j < n; ++j) {
            const llama_pos d = (llama_pos) (n - j);

            llama_pos p;
            if (by_order) {
                const auto & v = seq_idx[seq_id];
                const int64_t k = (int64_t) ord[i] - d;
                // k >= 0: an earlier token of this very ubatch; k < 0: before the chunk
                p = k >= 0 ? ubatch.pos[v[k]] : ubatch.pos[v[0]] + (llama_pos) k;
            } else {
                p = ubatch.pos[i] - d;
            }

            if (p < 0) {
                continue;
            }

            res[i*n + j] = v_cells[seq_to_stream[seq_id]].seq_pos_tok_le(seq_id, p);
        }
    }
}

size_t llama_kv_cache::total_size() const {
    size_t size = 0;

    for (const auto & [_, buf] : ctxs_bufs) {
        size += ggml_backend_buffer_get_size(buf.get());
    }

    return size;
}

size_t llama_kv_cache::size_k_bytes() const {
    size_t size_k_bytes = 0;

    for (const auto & layer : layers) {
        size_k_bytes += ggml_nbytes(layer.k);
    }

    return size_k_bytes;
}

size_t llama_kv_cache::size_v_bytes() const {
    size_t size_v_bytes = 0;

    for (const auto & layer : layers) {
        size_v_bytes += layer.v ? ggml_nbytes(layer.v) : 0;
    }

    return size_v_bytes;
}

ggml_tensor * llama_kv_cache::build_rope_shift(
        const llama_cparams & cparams,
               ggml_context * ctx,
                ggml_tensor * k,
                ggml_tensor * shift,
                ggml_tensor * rot,
                   uint32_t   il) const {
    const llama_rope_params rp = is_indexer ? model.get_rope_params_indexer(cparams, il) : model.get_rope_params(cparams, il);

    const auto rope_type = rp.type == LLAMA_ROPE_TYPE_MROPE || rp.type == LLAMA_ROPE_TYPE_IMROPE
                                // @ngxson : this is a workaround
                                // for M-RoPE, we want to rotate the whole vector when doing KV shift
                                // a normal RoPE should work, we just need to use the correct ordering
                                // ref: https://github.com/ggml-org/llama.cpp/pull/13870
                                ? LLAMA_ROPE_TYPE_NEOX
                                : rp.type;

    const int64_t n_head_kv     = hparams.n_head_kv(il);
    const int64_t n_embd_k_gqa  = hparams.n_embd_k_gqa(il);
    const int64_t n_embd_head_k = hparams.n_embd_head_k(il);
    const int64_t n_cells       = ggml_nelements(k)/n_embd_k_gqa;

    if (!ggml_is_quantized(k->type) && k->type != GGML_TYPE_F8_E4M3 && !rot && !rp.k_norm) {
        // we rotate only the n_rot dimensions
        ggml_tensor * k_rope =
            ggml_view_3d(ctx, k,
                rp.n_rot, n_head_kv, n_cells,
                ggml_row_size(k->type, n_embd_head_k),
                ggml_row_size(k->type, n_embd_k_gqa),
                ggml_row_size(k->type, rp.n_offs));

        return ggml_rope_ext_inplace(ctx, k_rope,
                shift, rp.factors, rp.n_rot, rope_type, rp.n_ctx_orig, rp.freq_base, rp.freq_scale,
                rp.ext_factor, rp.attn_factor, rp.beta_fast, rp.beta_slow);
    }

    // convert to f32 -> RoPE -> convert back, on whole rows as a head is not always a multiple of the block size
    ggml_tensor * tmp = ggml_reshape_3d(ctx, ggml_cast(ctx, k, GGML_TYPE_F32), n_embd_head_k, n_head_kv, n_cells);

    // rotate back
    if (rot) {
        tmp = llama_mul_mat_hadamard(ctx, tmp, rot);
    }

    // the RMS norm commutes with RoPE, but the weight does not
    if (rp.k_norm) {
        tmp = ggml_div(ctx, tmp, rp.k_norm);
    }

    tmp = ggml_rope_ext(ctx, tmp,
            shift, rp.factors, rp.n_rot, rope_type, rp.n_ctx_orig, rp.freq_base, rp.freq_scale,
            rp.ext_factor, rp.attn_factor, rp.beta_fast, rp.beta_slow);
    if (rp.n_offs > 0) {
        tmp = ggml_rope_set_offset(tmp, rp.n_offs);
    }

    if (rp.k_norm) {
        tmp = ggml_mul(ctx, tmp, rp.k_norm);
    }

    // rotate fwd
    if (rot) {
        tmp = llama_mul_mat_hadamard(ctx, tmp, rot);
    }

    return ggml_cpy(ctx, tmp, k);
}

class llm_graph_input_k_shift : public llm_graph_input_i {
public:
    llm_graph_input_k_shift(const llama_kv_cache * kv_self) : kv_self(kv_self) {}
    virtual ~llm_graph_input_k_shift() = default;

    void set_input(const llama_ubatch * ubatch) override;

    ggml_tensor * k_shift; // I32 [kv_size*n_stream]

    const llama_kv_cache * kv_self;
};

void llm_graph_input_k_shift::set_input(const llama_ubatch * ubatch) {
    GGML_UNUSED(ubatch);

    if (k_shift && k_shift->buffer) {
        kv_self->set_input_k_shift(k_shift);
    }
}

ggml_cgraph * llama_kv_cache::build_graph_shift(llm_graph_result * res, llama_context * lctx) const {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    GGML_ASSERT(!other);

    auto * ctx = res->get_ctx();
    auto * gf  = res->get_gf();

    auto inp = std::make_unique<llm_graph_input_k_shift>(this);

    inp->k_shift = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, (int64_t) get_size()*n_stream);
    ggml_set_input(inp->k_shift);

    const auto & cparams = lctx->get_cparams();

    for (const auto & layer : layers) {
        const uint32_t il = layer.il;

        if (!hparams.has_rope(il)) {
            continue;
        }

        ggml_tensor * cur = build_rope_shift(cparams, ctx, layer.k, inp->k_shift, layer.k_rot, il);

        ggml_build_forward_expand(gf, cur);
    }

    res->add_input(std::move(inp));

    return gf;
}

// A partial state (LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY, what a context checkpoint stores) is what trimming a sequence's
// cells cannot bring back: a sliding window's cells, reused as they fall out of the window. A cache without a window is
// restored by seq_rm, as the hybrid and iswa memories restore their full-attention parts, so its partial state is empty.
// LLAMA_KV_PARTIAL_STATE_LEGACY=1 writes and reads the whole sequence under the flag, as before.
static bool llama_kv_cache_partial_state_empty(llama_state_seq_flags flags, llama_swa_type swa_type) {
    static const bool legacy = ggml_env_switch("LLAMA_KV_PARTIAL_STATE_LEGACY");
    return !legacy && (flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) && swa_type == LLAMA_SWA_TYPE_NONE;
}

void llama_kv_cache::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return;
    }

    if (llama_kv_cache_partial_state_empty(flags, swa_type)) {
        return;
    }

    io.write(&n_stream, sizeof(n_stream));

    for (uint32_t s = 0; s < n_stream; ++s) {
        cell_ranges_t cr { s, {} };

        uint32_t cell_count = 0;

        const auto & cells = v_cells[s];

        // Count the number of cells with the specified seq_id
        // Find all the ranges of cells with this seq id (or all, when -1)
        uint32_t cell_range_begin = cells.size();

        for (uint32_t i = 0; i < cells.size(); ++i) {
            bool add_cell = true;

            add_cell = add_cell && !cells.is_empty(i);
            add_cell = add_cell && (seq_id == -1 || cells.seq_has(i, seq_id));

            // check the cell is not SWA-masked
            if (add_cell && seq_id != -1) {
                const bool is_masked = llama_hparams::is_masked_swa(n_swa, swa_type, cells.pos_get(i), cells.seq_pos_max(seq_id));

                add_cell = !is_masked;
            }

            if (add_cell) {
                ++cell_count;
                if (cell_range_begin == cells.size()) {
                    cell_range_begin = i;
                }
            } else {
                if (cell_range_begin != cells.size()) {
                    cr.data.emplace_back(cell_range_begin, i);
                    cell_range_begin = cells.size();
                }
            }
        }

        if (cell_range_begin != cells.size()) {
            cr.data.emplace_back(cell_range_begin, cells.size());
        }

        // DEBUG CHECK: Sum of cell counts in ranges should equal the total cell count
        uint32_t cell_count_check = 0;
        for (const auto & range : cr.data) {
            cell_count_check += range.second - range.first;
        }
        GGML_ASSERT(cell_count == cell_count_check);

        io.write(&cell_count, sizeof(cell_count));

        // skip empty streams
        if (cell_count == 0) {
            continue;
        }

        state_write_meta(io, cr, seq_id);
        state_write_data(io, cr);
    }
}

bool llama_kv_cache::stage_copy(
        llama_seq_id seq_id,
        uint32_t row,
        const std::function<ggml_backend_t(ggml_backend_dev_t)> & get_backend,
        stage_plan & plan) const {
    plan = {};
    if (other || n_stage == 0 || row >= n_stage || v_trans || n_stream != 1 || seq_id < 0 ||
            k_stage.size() <= row || k_stage[row].size() != layers.size()) {
        return false;
    }

    const auto & cells = v_cells[0];
    uint32_t cell_count = 0;
    uint32_t cell_range_begin = cells.size();
    for (uint32_t i = 0; i < cells.size(); ++i) {
        bool add_cell = !cells.is_empty(i) && cells.seq_has(i, seq_id);
        if (add_cell) {
            add_cell = !llama_hparams::is_masked_swa(n_swa, swa_type, cells.pos_get(i), cells.seq_pos_max(seq_id));
        }
        if (add_cell) {
            ++cell_count;
            if (cell_range_begin == cells.size()) {
                cell_range_begin = i;
            }
        } else if (cell_range_begin != cells.size()) {
            plan.ranges.emplace_back(cell_range_begin, i);
            cell_range_begin = cells.size();
        }
    }
    if (cell_range_begin != cells.size()) {
        plan.ranges.emplace_back(cell_range_begin, cells.size());
    }
    if (cell_count > n_swa) {
        return false;
    }
    plan.strm = 0;
    plan.cell_count = cell_count;

    if (cell_count == 0) {
        return true;
    }

    for (size_t ikv = 0; ikv < layers.size(); ++ikv) {
        const auto & layer = layers[ikv];
        auto copy_tensor_ranges = [&](const ggml_tensor * src, ggml_tensor * dst) -> bool {
            if (!src || !dst) {
                return src == dst;
            }
            ggml_backend_dev_t dev = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(src->buffer));
            ggml_backend_t backend = get_backend && dev ? get_backend(dev) : nullptr;
            int64_t dst_row = 0;
            for (const auto & range : plan.ranges) {
                const int64_t len = (int64_t) (range.second - range.first);
                if (!ggml_backend_meta_copy_rows_async(backend, src, dst, (int64_t) range.first, dst_row, len)) {
                    return false;
                }
                dst_row += len;
            }
            return true;
        };
        if (!copy_tensor_ranges(layer.k, k_stage[row][ikv]) ||
            !copy_tensor_ranges(layer.v, v_stage[row][ikv])) {
            return false;
        }
    }
    return true;
}

void llama_kv_cache::state_write_staged(
        llama_io_write_i & io,
        llama_seq_id seq_id,
        uint32_t row,
        const stage_plan & plan) const {
    GGML_ASSERT(!other && !v_trans && n_stream == 1 && row < n_stage);

    io.write(&n_stream, sizeof(n_stream));
    io.write(&plan.cell_count, sizeof(plan.cell_count));
    if (plan.cell_count == 0) {
        return;
    }

    cell_ranges_t cr { plan.strm, plan.ranges };
    state_write_meta(io, cr, seq_id);

    const uint32_t v_trans_u = 0;
    const uint32_t n_layer   = layers.size();
    io.write(&v_trans_u, sizeof(v_trans_u));
    io.write(&n_layer,   sizeof(n_layer));
    io.write(&n_rot_k,   sizeof(n_rot_k));
    io.write(&n_rot_v,   sizeof(n_rot_v));

    for (size_t ikv = 0; ikv < layers.size(); ++ikv) {
        const auto & layer = layers[ikv];
        auto * ks = k_stage[row][ikv];
        const int32_t  k_type_i   = (int32_t) ks->type;
        const uint64_t k_size_row = ggml_row_size(ks->type, hparams.n_embd_k_gqa(layer.il));
        io.write(&k_type_i,   sizeof(k_type_i));
        io.write(&k_size_row, sizeof(k_size_row));
        io.write_tensor(ks, 0, (size_t) plan.cell_count * k_size_row);
    }

    for (size_t ikv = 0; ikv < layers.size(); ++ikv) {
        const auto & layer = layers[ikv];
        auto * vs = v_stage[row][ikv];
        if (!vs) {
            continue;
        }
        const int32_t  v_type_i   = (int32_t) vs->type;
        const uint64_t v_size_row = ggml_row_size(vs->type, hparams.n_embd_v_gqa(layer.il));
        io.write(&v_type_i,   sizeof(v_type_i));
        io.write(&v_size_row, sizeof(v_size_row));
        io.write_tensor(vs, 0, (size_t) plan.cell_count * v_size_row);
    }
}

void llama_kv_cache::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    state_read_sinfo(io, seq_id, flags, nullptr, nullptr);
}

void llama_kv_cache::state_read_sinfo(
        llama_io_read_i & io,
           llama_seq_id   seq_id,
  llama_state_seq_flags   flags,
      slot_info_vec_t *   sinfos_out,
const slot_info_vec_t *   sinfos_in) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return;
    }

    if (llama_kv_cache_partial_state_empty(flags, swa_type)) {
        return;
    }

    if (seq_id < 0) {
        kv_mixed_reset();
    }

    // TODO: fix incosistent handling of `seq_id < 0` and `seq_id == -1` in the codebase [TAG_LLAMA_SEQ_ID_NEG]
    GGML_ASSERT(seq_id == -1 || (seq_id >= 0 && (size_t) seq_id < seq_to_stream.size()));

    if (sinfos_out) {
        sinfos_out->assign(n_stream, slot_info{});
    }

    if (sinfos_in && sinfos_in->size() != n_stream) {
        throw std::runtime_error("failed to restore kv cache: mirrored slot layout has the wrong stream count");
    }

    uint32_t n_stream_cur;
    io.read(&n_stream_cur, sizeof(n_stream_cur));
    if (n_stream_cur != n_stream) {
        throw std::runtime_error("n_stream mismatch");
    }

    // a whole-context restore replaces every stream, so the cache is emptied once here
    // clear() resets all streams at once, so doing it per stream below would keep only the last one
    if (seq_id == -1) {
        clear(true);
    }

    for (uint32_t s = 0; s < n_stream; ++s) {
        uint32_t cell_count;
        io.read(&cell_count, sizeof(cell_count));

        if (cell_count == 0) {
            // a mirrored cache must be empty here as well, or the two no longer agree cell for cell
            if (sinfos_in && !(*sinfos_in)[s].empty()) {
                throw std::runtime_error("failed to restore kv cache: mirrored cache holds cells this one does not");
            }
            continue;
        }

        const uint32_t strm = seq_id == -1 ? s : seq_to_stream[seq_id];

        slot_info sinfo;

        bool res = true;
        res = res && state_read_meta(io, strm, cell_count, sinfo, seq_id, sinfos_in ? &(*sinfos_in)[s] : nullptr);
        if (seq_id >= 0 && !sinfo.empty() && !sinfo.idxs[0].empty()) {
            kv_mixed_unlink(strm, sinfo.idxs[0].data(), sinfo.idxs[0].size());
        }

        try {
            res = res && state_read_data(io, strm, cell_count, sinfo);
        } catch (const std::exception & err) {
            LLAMA_LOG_ERROR("%s: %s\n", __func__, err.what());
            res = false;
        } catch (...) {
            res = false;
        }

        if (!res) {
            state_clear(seq_id, strm, sinfo);
            throw std::runtime_error("failed to restore kv cache");
        }

        if (sinfos_out) {
            (*sinfos_out)[s] = sinfo;
        }
    }
}

void llama_kv_cache::state_write_meta(llama_io_write_i & io, const cell_ranges_t & cr, llama_seq_id seq_id) const {
    const auto & cells = v_cells[cr.strm];

    for (const auto & range : cr.data) {
        for (uint32_t i = range.first; i < range.second; ++i) {
            std::vector<llama_seq_id> seq_ids;

            for (llama_seq_id cur = 0; cur < (int) n_seq_max; ++cur) {
                if (cur == seq_id || seq_id == -1) {
                    if (cells.seq_has(i, cur)) {
                        seq_ids.push_back(cur);
                    }
                }
            }

            const llama_pos pos     = cells.pos_get(i);
            const uint32_t n_seq_id = seq_ids.size();

            io.write(&pos,      sizeof(pos));
            io.write(&n_seq_id, sizeof(n_seq_id));

            if (has_cell_ext()) {
                const llama_kv_cell_ext ext = cells.ext_get(i);
                io.write(&ext, sizeof(ext));
            }

            for (const auto & seq_id : seq_ids) {
                io.write(&seq_id, sizeof(seq_id));
            }
        }
    }
}

void llama_kv_cache::state_write_data(llama_io_write_i & io, const cell_ranges_t & cr) const {
    const auto & cells = v_cells[cr.strm];

    const uint32_t v_trans = this->v_trans ? 1 : 0;
    const uint32_t n_layer = layers.size();

    io.write(&v_trans, sizeof(v_trans));
    io.write(&n_layer, sizeof(n_layer));
    io.write(&n_rot_k, sizeof(n_rot_k));
    io.write(&n_rot_v, sizeof(n_rot_v));

    // Iterate and write all the keys first, each row is a cell
    // Get whole range at a time
    for (const auto & layer : layers) {
        const uint32_t il = layer.il;

        const uint32_t n_embd_k_gqa = hparams.n_embd_k_gqa(il);

        auto * k = layer.k_stream[cr.strm];

        // Write key type
        const int32_t k_type_i = (int32_t) k->type;
        io.write(&k_type_i, sizeof(k_type_i));

        // Write row size of key
        const uint64_t k_size_row = ggml_row_size(k->type, n_embd_k_gqa);
        io.write(&k_size_row, sizeof(k_size_row));

        // Read each range of cells of k_size length and write out
        for (const auto & range : cr.data) {
            const size_t range_size = range.second - range.first;
            const size_t buf_size = range_size * k_size_row;
            io.write_tensor(k, range.first * k_size_row, buf_size);
        }
    }

    if (!v_trans) {
        for (const auto & layer : layers) {
            const uint32_t il = layer.il;

            const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(il);

            auto * v = layer.v_stream[cr.strm];
            if (!v) {
                continue;
            }

            // Write value type
            const int32_t v_type_i = (int32_t) v->type;
            io.write(&v_type_i, sizeof(v_type_i));

            // Write row size of value
            const uint64_t v_size_row = ggml_row_size(v->type, n_embd_v_gqa);
            io.write(&v_size_row, sizeof(v_size_row));

            // Read each range of cells of v_size length and write out
            for (const auto & range : cr.data) {
                const size_t range_size = range.second - range.first;
                const size_t buf_size = range_size * v_size_row;
                io.write_tensor(v, range.first * v_size_row, buf_size);
            }
        }
    } else {
        // When v is transposed, we also need the element size and get the element ranges from each row
        const uint32_t kv_size = cells.size();

        for (const auto & layer : layers) {
            const uint32_t il = layer.il;

            const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(il);

            auto * v = layer.v_stream[cr.strm];
            if (!v) {
                continue;
            }

            // Write value type
            const int32_t v_type_i = (int32_t) v->type;
            io.write(&v_type_i, sizeof(v_type_i));

            // Write element size
            const uint32_t v_size_el = ggml_type_size(v->type);
            io.write(&v_size_el, sizeof(v_size_el));

            // Write GQA embedding size
            io.write(&n_embd_v_gqa, sizeof(n_embd_v_gqa));

            // For each row, we get the element values of each cell
            for (uint32_t j = 0; j < n_embd_v_gqa; ++j) {
                // Read each range of cells of v_size_el length and write out
                for (const auto & range : cr.data) {
                    const size_t range_size = range.second - range.first;
                    // in 64 bits: n_embd_v_gqa * kv_size can reach 2^32
                    const size_t src_offset = (range.first + (size_t) j * kv_size) * v_size_el;
                    const size_t buf_size = range_size * v_size_el;
                    io.write_tensor(v, src_offset, buf_size);
                }
            }
        }
    }
}

bool llama_kv_cache::state_read_meta(llama_io_read_i & io, uint32_t strm, uint32_t cell_count, slot_info & sinfo, llama_seq_id dest_seq_id, const slot_info * sinfo_in) {
    auto & cells = v_cells[strm];
    auto & head  = v_heads[strm];

    if (dest_seq_id != -1) {
        // single sequence
        if (cell_count > cells.size()) {
            LLAMA_LOG_ERROR("%s: not enough cells in kv cache\n", __func__);
            return false;
        }

        seq_rm(dest_seq_id, -1, -1);

        llama_batch_allocr balloc(hparams.n_pos_per_embd());

        llama_ubatch ubatch = balloc.ubatch_reserve(cell_count, 1);

        ubatch.seq_id_unq[0] = dest_seq_id;

        // the ext as it was saved, to put back after apply_ubatch()
        std::vector<llama_kv_cell_ext> exts;
        if (has_cell_ext()) {
            exts.resize(cell_count);
        }

        for (uint32_t i = 0; i < cell_count; ++i) {
            llama_pos pos;
            uint32_t n_seq_id;

            io.read(&pos,      sizeof(pos));
            io.read(&n_seq_id, sizeof(n_seq_id));

            if (n_seq_id != 1) {
                LLAMA_LOG_ERROR("%s: invalid seq_id-agnostic kv cell\n", __func__);
                return false;
            }

            if (has_cell_ext()) {
                llama_kv_cell_ext ext;
                io.read(&ext, sizeof(ext));

                if (hparams.n_pos_per_embd() > 1) {
                    ubatch.pos[i + ubatch.n_tokens]   = ext.y;
                    ubatch.pos[i + ubatch.n_tokens*2] = ext.x;
                }

                // apply_ubatch() below restores ext.tok from the ubatch tokens
                ubatch.token[i] = ext.tok;

                exts[i] = ext;
            }

            // read the sequence id, but directly discard it - we will use dest_seq_id instead
            {
                llama_seq_id seq_id;
                io.read(&seq_id, sizeof(seq_id));
            }

            ubatch.pos[i]      = pos;
            ubatch.n_seq_id[i] = n_seq_id;
            ubatch.seq_id[i]   = &dest_seq_id;
        }

        if (sinfo_in) {
            // this cache mirrors another one, so it takes that cache's layout instead of searching for its own cells
            if (sinfo_in->empty() || sinfo_in->n_stream() != 1 || sinfo_in->idxs[0].size() != cell_count) {
                LLAMA_LOG_ERROR("%s: mirrored slot layout holds %d cells, this cache restores %d\n", __func__,
                        sinfo_in->empty() ? 0 : (int) sinfo_in->idxs[0].size(), cell_count);
                return false;
            }

            sinfo = *sinfo_in;

            // the layout is cell indices, so it means the same in both caches only while their streams line up
            sinfo.s0 = strm;
            sinfo.s1 = strm;
            sinfo.strm[0] = strm;

            // seq_rm above freed exactly the cells this sequence held
            // anything else in the way is a cache that had already drifted, which this restore must not hide
            for (uint32_t i = 0; i < cell_count; ++i) {
                const uint32_t idx = sinfo.idxs[0][i];

                if (idx >= cells.size() || !cells.is_empty(idx)) {
                    LLAMA_LOG_ERROR("%s: cell %u of the mirrored slot layout is not free\n", __func__, idx);
                    return false;
                }
            }
        } else {
            sinfo = find_slot(ubatch, false);
            if (sinfo.empty()) {
                LLAMA_LOG_ERROR("%s: failed to find %d available cells in kv cache\n", __func__,  cell_count);
                return false;
            }
        }

        // note: apply_ubatch() rebuilds llama_kv_cell_ext from the ubatch
        //       only ext.tok and the M-RoPE 2D position round-trip through it
        //       see: https://github.com/ggml-org/llama.cpp/pull/16825#issuecomment-3460868350
        apply_ubatch(sinfo, ubatch);

        // apply_ubatch() takes the 2D position from the ubatch, and that ubatch is built with this
        // cache's own n_pos_per_embd. a cache that does not use M-RoPE itself but mirrors one that
        // does (the qwen4exp QSA indexer) would drop x and y. put the saved ext back instead, which
        // is what the whole-context path below already does.
        for (uint32_t i = 0; i < (uint32_t) exts.size(); ++i) {
            cells.ext_set(sinfo.idxs[0][i], exts[i]);
        }

        LLAMA_LOG_DEBUG("%s: cell_count = %d, dest_seq_id = %d\n", __func__, cell_count, dest_seq_id);

        // DEBUG CHECK: verify that all cells were allocated and have correct seq_id and pos values
        GGML_ASSERT(sinfo.n_stream() == 1);
        GGML_ASSERT(sinfo.idxs[0].size() == cell_count);
        for (uint32_t i = 0; i < cell_count; ++i) {
            const uint32_t idx = sinfo.idxs[0][i];
            GGML_ASSERT(cells.pos_get(idx) == ubatch.pos[i]);
            GGML_ASSERT(cells.seq_has(idx, dest_seq_id));
        }
    } else {
        // whole KV cache restore

        if (cell_count > cells.size()) {
            LLAMA_LOG_ERROR("%s: not enough cells in kv cache\n", __func__);
            return false;
        }

        // the cells go in from 0, so a mirrored cache lands on the same ones as long as it restores the same count. the layout itself carries no more information here
        if (sinfo_in && (sinfo_in->empty() || sinfo_in->n_stream() != 1 || sinfo_in->idxs[0].size() != cell_count)) {
            LLAMA_LOG_ERROR("%s: mirrored slot layout holds %d cells, this cache restores %d\n", __func__,
                    sinfo_in->empty() ? 0 : (int) sinfo_in->idxs[0].size(), cell_count);
            return false;
        }

        for (uint32_t i = 0; i < cell_count; ++i) {
            llama_pos pos;
            uint32_t  n_seq_id;

            io.read(&pos,      sizeof(pos));
            io.read(&n_seq_id, sizeof(n_seq_id));

            cells.pos_set(i, pos);

            if (has_cell_ext()) {
                llama_kv_cell_ext ext;
                io.read(&ext, sizeof(ext));
                cells.ext_set(i, ext);
            }

            for (uint32_t j = 0; j < n_seq_id; ++j) {
                llama_seq_id seq_id;
                io.read(&seq_id, sizeof(seq_id));

                if (seq_id < 0 || (uint32_t) seq_id >= n_seq_max) {
                    LLAMA_LOG_ERROR("%s: invalid seq_id, %d is out of range [0, %u)\n", __func__, seq_id, n_seq_max);
                    return false;
                }
                if (seq_to_stream[seq_id] != strm) {
                    LLAMA_LOG_ERROR("%s: sequence %d belongs to a different cache stream\n", __func__, seq_id);
                    return false;
                }

                cells.seq_add(i, seq_id);
            }
        }

        // Create contiguous slot_info for whole cache restore
        sinfo.s0 = strm;
        sinfo.s1 = strm;
        sinfo.resize(1);
        sinfo.strm[0] = strm;
        sinfo.idxs[0].resize(cell_count);
        for (uint32_t i = 0; i < cell_count; ++i) {
            sinfo.idxs[0][i] = i;
        }

        head = 0;
    }

    return true;
}

#include "llama-kv-state-migrate.h"
bool llama_kv_cache::state_read_data(llama_io_read_i & io, uint32_t strm, uint32_t cell_count, const slot_info & sinfo) {
    // with the knob on a state saved under another cache configuration migrates into this one (see the helpers
    // above); unset, every mismatch below fails exactly as it always has
    const bool migrate = llama_kv_state_migrate_enabled();
    auto & cells = v_cells[strm];

    // batch the scatter reads per contiguous run of destination indices
    // from inclusive, to exclusive - same convention as cell_ranges_t
    // contiguous cells yield a single run covering the whole block
    // LLAMA_KV_RESTORE_PER_CELL_LEGACY=1: scattered cells are read one cell at a time, as before
    static const bool per_cell = ggml_env_switch("LLAMA_KV_RESTORE_PER_CELL_LEGACY");
    struct cell_run { uint32_t from; uint32_t to; };
    std::vector<cell_run> runs;
    if (cell_count > 0) {
        const auto & idxs = sinfo.idxs[0];
        const bool runs_ok = !per_cell || sinfo.is_contiguous();
        uint32_t i0 = 0;
        while (i0 < cell_count) {
            uint32_t i1 = i0 + 1;
            while (runs_ok && i1 < cell_count && idxs[i1] == idxs[i1 - 1] + 1) {
                ++i1;
            }
            runs.push_back({idxs[i0], idxs[i1 - 1] + 1});
            i0 = i1;
        }
    }

    uint32_t v_trans;
    uint32_t n_layer;
    uint32_t n_rot_k_ref;
    uint32_t n_rot_v_ref;

    io.read(&v_trans, sizeof(v_trans));
    io.read(&n_layer, sizeof(n_layer));
    io.read(&n_rot_k_ref, sizeof(n_rot_k_ref));
    io.read(&n_rot_v_ref, sizeof(n_rot_v_ref));

    if (n_layer != layers.size()) {
        LLAMA_LOG_ERROR("%s: mismatched layer count (%u instead of %u)\n", __func__, n_layer, (uint32_t) layers.size());
        return false;
    }

    if (cell_count > cells.size()) {
        LLAMA_LOG_ERROR("%s: not enough cells in kv cache to restore state (%u > %u)\n", __func__, cell_count, cells.size());
        return false;
    }

    if (this->v_trans != (bool) v_trans) {
        LLAMA_LOG_ERROR("%s: incompatible V transposition%s\n", __func__,
                migrate ? " (state migration keeps it; only cache types and rotations convert)" : "");
        return false;
    }

    // the rotation sides a converted read works between: what stored the file's values and what this cache would
    // store fresh. Sizes resolve to matrices lazily, so a size the engine never precomputed still converts.
    struct rot_side { const float * m = nullptr; int n = 0; };
    auto rot_of = [&](uint32_t n_rot, bool & ok) -> rot_side {
        if (!n_rot) {
            return {};
        }
        if (!ggml_is_power_of_2((int) n_rot) || n_rot > 4096u) {
            ok = false;
            return {};
        }
        const std::vector<float> & h = llama_kv_state_rot_matrix(n_rot);
        ok = true;
        return { h.data(), (int) n_rot };
    };

    bool rots_ok = true;
    const rot_side rot_src_k = rot_of(n_rot_k_ref, rots_ok);
    const rot_side rot_dst_k = rot_of(n_rot_k,     rots_ok);
    const rot_side rot_src_v = rot_of(n_rot_v_ref, rots_ok);
    const rot_side rot_dst_v = rot_of(n_rot_v,     rots_ok);

    if (rots_ok && migrate && (n_rot_k_ref != n_rot_k || n_rot_v_ref != n_rot_v)) {
        LLAMA_LOG_INFO("%s: migrating state rotation (%u/%u stored into %u/%u)\n", __func__,
                n_rot_k_ref, n_rot_v_ref, n_rot_k, n_rot_v);
    }

    if (!migrate || !rots_ok) {
        if (n_rot_k_ref != n_rot_k) {
            LLAMA_LOG_ERROR("%s: incompatible key rotation (%u instead of %u)%s\n", __func__, n_rot_k_ref, n_rot_k,
                    migrate && !rots_ok ? " (state migration could not resolve a rotation matrix)" : "");
            return false;
        }

        if (n_rot_v_ref != n_rot_v) {
            LLAMA_LOG_ERROR("%s: incompatible value rotation (%u instead of %u)%s\n", __func__, n_rot_v_ref, n_rot_v,
                    migrate && !rots_ok ? " (state migration could not resolve a rotation matrix)" : "");
            return false;
        }
    }

    // For each layer, read the keys for each cell, one row is one cell, read as one contiguous block
    for (const auto & layer : layers) {
        const uint32_t il = layer.il;

        const uint32_t n_embd_k_gqa = hparams.n_embd_k_gqa(il);

        auto * k = layer.k_stream[strm];

        // Read type of key
        int32_t k_type_i_ref;
        io.read(&k_type_i_ref, sizeof(k_type_i_ref));
        const int32_t k_type_i = (int32_t) k->type;

        // Read row size of key
        uint64_t k_size_row_ref;
        io.read(&k_size_row_ref, sizeof(k_size_row_ref));
        const size_t k_size_row = ggml_row_size(k->type, n_embd_k_gqa);

        // the record's own type and row bytes must describe rows this migrator can interpret
        auto row_bytes_ok = [](int32_t t, int64_t values, uint64_t ref) -> bool {
            if (!llama_kv_state_type_supported(t)) {
                return false;
            }
            const ggml_type type = (ggml_type) t;
            if ((type == GGML_TYPE_Q4_0 || type == GGML_TYPE_Q8_0) && values % QK8_0 != 0) {
                return false;
            }
            return ref == (uint64_t) ggml_row_size(type, values);
        };

        // an identical configuration reads exactly as before: the migrator's own supported types have no
        // bearing here, a same-type restore never encodes anything
        if (!migrate || (k_type_i == k_type_i_ref && n_rot_k_ref == n_rot_k)) {
            if (k_type_i != k_type_i_ref) {
                LLAMA_LOG_ERROR("%s: mismatched key type (%d != %d, layer %d)\n", __func__, k_type_i, k_type_i_ref, il);
                return false;
            }

            if (k_size_row != k_size_row_ref) {
                LLAMA_LOG_ERROR("%s: mismatched key row size (%zu != %zu, layer %d)\n", __func__, k_size_row, (size_t) k_size_row_ref, il);
                return false;
            }

            for (const auto & r : runs) {
                io.read_tensor(k, (size_t) r.from * k_size_row, (size_t) (r.to - r.from) * k_size_row);
            }
            continue;
        }

        // migrate the record's rows into this cache's type and domain of stored values
        if (!llama_kv_state_type_supported(k_type_i) || !row_bytes_ok(k_type_i_ref, n_embd_k_gqa, k_size_row_ref)) {
            LLAMA_LOG_ERROR("%s: cannot migrate key to a %s cache from a %s row of %llu bytes (layer %d)%s\n",
                    __func__, ggml_type_name(k->type), ggml_type_name((ggml_type) k_type_i_ref),
                    (unsigned long long) k_size_row_ref, il,
                    llama_kv_state_migrate_enabled() ? "" : " (LLAMA_KV_STATE_MIGRATE off)");
            return false;
        }

        const bool same_rot = (rot_src_k.n == rot_dst_k.n) && (!rot_src_k.m || rot_src_k.m == rot_dst_k.m);

        llama_kv_state_read_rows_converted(io, k, sinfo.idxs[0].data(), cell_count,
                (ggml_type) k_type_i_ref, k_size_row_ref, (int64_t) n_embd_k_gqa, k->type,
                same_rot ? nullptr : rot_src_k.m, same_rot ? 0 : rot_src_k.n,
                same_rot ? nullptr : rot_dst_k.m, same_rot ? 0 : rot_dst_k.n);
    }

    if (!this->v_trans) {
        for (const auto & layer : layers) {
            const uint32_t il = layer.il;

            const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(il);

            auto * v = layer.v_stream[strm];
            if (!v) {
                continue;
            }

            // Read type of value
            int32_t v_type_i_ref;
            io.read(&v_type_i_ref, sizeof(v_type_i_ref));
            const int32_t v_type_i = (int32_t) v->type;

            // Read row size of value
            uint64_t v_size_row_ref;
            io.read(&v_size_row_ref, sizeof(v_size_row_ref));
            const size_t v_size_row = ggml_row_size(v->type, n_embd_v_gqa);

            if (!migrate || (v_type_i == v_type_i_ref && n_rot_v_ref == n_rot_v)) {
                if (v_type_i != v_type_i_ref) {
                    LLAMA_LOG_ERROR("%s: mismatched value type (%d != %d, layer %d)\n", __func__, v_type_i, v_type_i_ref, il);
                    return false;
                }

                if (v_size_row != v_size_row_ref) {
                    LLAMA_LOG_ERROR("%s: mismatched value row size (%zu != %zu, layer %d)\n", __func__, v_size_row, (size_t) v_size_row_ref, il);
                    return false;
                }

                for (const auto & r : runs) {
                    io.read_tensor(v, (size_t) r.from * v_size_row, (size_t) (r.to - r.from) * v_size_row);
                }
                continue;
            }

            // migrate the record's rows into this cache's type and domain of stored values
            if (!llama_kv_state_type_supported(v_type_i) ||
                    !(v_size_row_ref == (uint64_t) ggml_row_size((ggml_type) v_type_i_ref, n_embd_v_gqa)) ||
                    ((v_type_i_ref == GGML_TYPE_Q4_0 || v_type_i_ref == GGML_TYPE_Q8_0) && n_embd_v_gqa % QK8_0 != 0)) {
                LLAMA_LOG_ERROR("%s: cannot migrate value to a %s cache from a %s row of %llu bytes (layer %d)%s\n",
                        __func__, ggml_type_name(v->type), ggml_type_name((ggml_type) v_type_i_ref),
                        (unsigned long long) v_size_row_ref, il,
                        llama_kv_state_migrate_enabled() ? "" : " (LLAMA_KV_STATE_MIGRATE off)");
                return false;
            }

            const bool same_rot = (rot_src_v.n == rot_dst_v.n) && (!rot_src_v.m || rot_src_v.m == rot_dst_v.m);

            llama_kv_state_read_rows_converted(io, v, sinfo.idxs[0].data(), cell_count,
                    (ggml_type) v_type_i_ref, v_size_row_ref, (int64_t) n_embd_v_gqa, v->type,
                    same_rot ? nullptr : rot_src_v.m, same_rot ? 0 : rot_src_v.n,
                    same_rot ? nullptr : rot_dst_v.m, same_rot ? 0 : rot_dst_v.n);
        }
    } else {
        // For each layer, read the values for each cell (transposed)
        for (const auto & layer : layers) {
            const uint32_t il = layer.il;

            const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(il);

            auto * v = layer.v_stream[strm];
            if (!v) {
                continue;
            }

            // Read type of value
            int32_t v_type_i_ref;
            io.read(&v_type_i_ref, sizeof(v_type_i_ref));
            const int32_t v_type_i = (int32_t) v->type;
            if (v_type_i != v_type_i_ref) {
                // state migration converts plain rows only; transposed V records keep today's refusal
                LLAMA_LOG_ERROR("%s: mismatched value type (%d != %d, layer %d)%s\n", __func__, v_type_i, v_type_i_ref, il,
                        migrate ? " (state migration does not convert transposed V records)" : "");
                return false;
            }

            // Read element size of value
            uint32_t v_size_el_ref;
            io.read(&v_size_el_ref, sizeof(v_size_el_ref));

            const size_t v_size_el = ggml_type_size(v->type);
            if (v_size_el != v_size_el_ref) {
                LLAMA_LOG_ERROR("%s: mismatched value element size (%zu != %zu, layer %d)\n", __func__, v_size_el, (size_t) v_size_el_ref, il);
                return false;
            }

            // Read GQA embedding size
            uint32_t n_embd_v_gqa_ref;
            io.read(&n_embd_v_gqa_ref, sizeof(n_embd_v_gqa_ref));
            if (n_embd_v_gqa != n_embd_v_gqa_ref) {
                LLAMA_LOG_ERROR("%s: mismatched GQA embedding size (%u != %u, layer %d)\n", __func__, n_embd_v_gqa, n_embd_v_gqa_ref, il);
                return false;
            }

            // the offsets in 64 bits: n_embd_v_gqa * kv_size can reach 2^32
            for (uint32_t j = 0; j < n_embd_v_gqa; ++j) {
                for (const auto & r : runs) {
                    const size_t dst_offset = ((size_t) r.from + (size_t) j * cells.size()) * v_size_el;
                    io.read_tensor(v, dst_offset, (size_t) (r.to - r.from) * v_size_el);
                }
            }
        }
    }

    return true;
}

void llama_kv_cache::state_clear(llama_seq_id seq_id) {
    if (seq_id == -1) {
        clear(true);
        return;
    }

    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());

    const uint32_t strm = seq_to_stream[seq_id];

    const auto & cells = v_cells[strm];

    slot_info sinfo;
    sinfo.s0 = strm;
    sinfo.s1 = strm;
    sinfo.resize(1);
    sinfo.strm[0] = strm;

    // a cell that another sequence still uses keeps its data
    for (uint32_t i = 0; i < cells.size(); ++i) {
        if (cells.seq_has(i, seq_id) && cells.seq_count(i) == 1) {
            sinfo.idxs[0].push_back(i);
        }
    }

    state_clear(seq_id, strm, sinfo);
}

// the cleared ranges mirror the write pattern of state_read_data() - keep both in sync
void llama_kv_cache::state_clear(llama_seq_id seq_id, uint32_t strm, const slot_info & sinfo) {
    if (seq_id == -1) {
        clear(true);
        return;
    }

    seq_rm(seq_id, -1, -1);

    // zero the K/V data of the failed restore attempt - the attention can still read the data of free cells
    if (sinfo.empty() || sinfo.size() == 0) {
        return;
    }

    const auto & cells = v_cells[strm];

    const uint32_t cell_count = sinfo.size();

    const bool is_contiguous = sinfo.is_contiguous();

    for (const auto & layer : layers) {
        const uint32_t il = layer.il;

        const uint32_t n_embd_k_gqa = hparams.n_embd_k_gqa(il);

        auto * k = layer.k_stream[strm];

        const size_t k_size_row = ggml_row_size(k->type, n_embd_k_gqa);

        if (is_contiguous) {
            llama_clear_tensor_data(k, sinfo.head() * k_size_row, cell_count * k_size_row);
        } else {
            for (uint32_t i = 0; i < cell_count; ++i) {
                llama_clear_tensor_data(k, sinfo.idxs[0][i] * k_size_row, k_size_row);
            }
        }
    }

    for (const auto & layer : layers) {
        const uint32_t il = layer.il;

        const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(il);

        auto * v = layer.v_stream[strm];
        if (!v) {
            continue;
        }

        if (!v_trans) {
            const size_t v_size_row = ggml_row_size(v->type, n_embd_v_gqa);

            if (is_contiguous) {
                llama_clear_tensor_data(v, sinfo.head() * v_size_row, cell_count * v_size_row);
            } else {
                for (uint32_t i = 0; i < cell_count; ++i) {
                    llama_clear_tensor_data(v, sinfo.idxs[0][i] * v_size_row, v_size_row);
                }
            }
        } else {
            const size_t v_size_el = ggml_type_size(v->type);

            if (is_contiguous) {
                const uint32_t h = sinfo.head();

                for (uint32_t j = 0; j < n_embd_v_gqa; ++j) {
                    llama_clear_tensor_data(v, (h + (size_t) j * cells.size()) * v_size_el, cell_count * v_size_el);
                }
            } else {
                for (uint32_t j = 0; j < n_embd_v_gqa; ++j) {
                    for (uint32_t i = 0; i < cell_count; ++i) {
                        llama_clear_tensor_data(v, (sinfo.idxs[0][i] + (size_t) j * cells.size()) * v_size_el, v_size_el);
                    }
                }
            }
        }
    }
}

//
// llama_kv_cache_context
//

llama_kv_cache_context::llama_kv_cache_context(llama_memory_status status) : status(status) {}

llama_kv_cache_context::llama_kv_cache_context(
        llama_kv_cache * kv) : status(LLAMA_MEMORY_STATUS_SUCCESS), kv(kv), full(true) {
    n_kv = n_kv_pool = kv->get_size();

    // worst case for graph reservation, only a unified cache splits into per-sequence lists
    n_kv_rows = kv->get_n_stream() == 1 ? n_kv : 0;

    const uint32_t n_stream = kv->get_n_stream();

    // create a dummy slot info - the actual data is irrelevant. we just need to build the graph
    sinfos.resize(1);
    sinfos[0].s0 = 0;
    sinfos[0].s1 = n_stream - 1;
    sinfos[0].idxs.resize(n_stream);
    for (uint32_t s = 0; s < n_stream; ++s) {
        sinfos[0].strm.push_back(s);
        sinfos[0].idxs[s].resize(1, 0);
    }
}

llama_kv_cache_context::llama_kv_cache_context(
        llama_kv_cache * kv,
        llama_context * lctx,
        bool do_shift,
        stream_copy_info sc_info) : status(LLAMA_MEMORY_STATUS_SUCCESS), kv(kv), lctx(lctx), do_shift(do_shift), sc_info(std::move(sc_info)) {
    if (!do_shift && this->sc_info.empty()) {
        status = LLAMA_MEMORY_STATUS_NO_UPDATE;
    }
}

llama_kv_cache_context::llama_kv_cache_context(
        llama_kv_cache * kv,
        llama_kv_cache::slot_info_vec_t sinfos,
        std::vector<llama_ubatch> ubatches) : status(LLAMA_MEMORY_STATUS_SUCCESS), kv(kv), sinfos(std::move(sinfos)), ubatches(std::move(ubatches)) {
}

llama_kv_cache_context::~llama_kv_cache_context() = default;

bool llama_kv_cache_context::next() {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    if (++i_cur >= ubatches.size()) {
        return false;
    }

    return true;
}

bool llama_kv_cache_context::apply() {
    assert(!llama_memory_status_is_fail(status));

    // no ubatches -> this is a KV cache update
    if (ubatches.empty()) {
        kv->update(lctx, do_shift, sc_info);

        return true;
    }

    kv->apply_ubatch(sinfos[i_cur], ubatches[i_cur]);
    kv->kv_mixed_apply(sinfos[i_cur], ubatches[i_cur]);
    n_kv      = kv->get_n_kv(sinfos[i_cur], &ubatches[i_cur], &n_pad_kv, &n_kv_pool);
    n_kv_rows = kv->get_n_kv_rows(ubatches[i_cur]);

    return true;
}

llama_memory_status llama_kv_cache_context::get_status() const {
    return status;
}

const llama_ubatch & llama_kv_cache_context::get_ubatch() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    return ubatches[i_cur];
}

uint32_t llama_kv_cache_context::get_n_kv() const {
    return n_kv;
}

uint32_t llama_kv_cache_context::get_n_kv_pool() const {
    return n_kv_pool;
}

uint32_t llama_kv_cache_context::get_n_pad() const {
    return n_pad_kv;
}

uint32_t llama_kv_cache_context::get_n_kv_rows() const {
    return n_kv_rows;
}

ggml_type llama_kv_cache_context::type_k() const {
    return kv->type_k();
}

ggml_type llama_kv_cache_context::type_v() const {
    return kv->type_v();
}

uint32_t llama_kv_cache_context::get_n_stream() const {
    return kv->get_n_devices() > 1 ? sinfos[i_cur].n_stream() : sinfos[i_cur].s1 - sinfos[i_cur].s0 + 1;
}

uint32_t llama_kv_cache_context::get_strm(uint32_t s) const {
    const auto & sinfo = sinfos[i_cur];

    GGML_ASSERT(s < sinfo.strm.size());

    return sinfo.strm[s];
}

const llama_kv_cache * llama_kv_cache_context::get_kv() const {
    return kv;
}

bool llama_kv_cache_context::is_full() const {
    return full;
}

ggml_tensor * llama_kv_cache_context::get_k(ggml_context * ctx, int32_t il) const {
    return kv->get_k(ctx, il, n_kv, sinfos[i_cur]);
}

ggml_tensor * llama_kv_cache_context::get_v(ggml_context * ctx, int32_t il) const {
    return kv->get_v(ctx, il, n_kv, sinfos[i_cur]);
}

bool llama_kv_cache_context::has_kv_mixed() const {
    return kv->has_kv_mixed();
}

uint32_t llama_kv_cache_context::get_hot_extent() const {
    return kv->get_hot_extent();
}

ggml_tensor * llama_kv_cache_context::get_hot_kv(ggml_context * ctx, int32_t il) const {
    return kv->get_hot_kv(ctx, il, sinfos[i_cur]);
}

ggml_tensor * llama_kv_cache_context::cpy_k_hot(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * hot_k_idxs, int32_t il) const {
    return kv->cpy_k_hot(ctx, k_cur, hot_k_idxs, il);
}

ggml_tensor * llama_kv_cache_context::cpy_v_hot(ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * hot_v_idxs, int32_t il) const {
    return kv->cpy_v_hot(ctx, v_cur, hot_v_idxs, il);
}

ggml_tensor * llama_kv_cache_context::build_input_hot_idxs(ggml_context * ctx, const llama_ubatch & ubatch, const char * name) const {
    return kv->build_input_hot_idxs(ctx, ubatch, name);
}

void llama_kv_cache_context::set_input_hot_idxs(ggml_tensor * k_dst, ggml_tensor * v_dst, const llama_ubatch * ubatch) const {
    kv->set_input_hot_idxs(k_dst, v_dst, ubatch);
}

ggml_tensor * llama_kv_cache_context::get_k_scale(int32_t il) const {
    return kv->get_k_scale(il);
}

ggml_tensor * llama_kv_cache_context::get_v_scale(int32_t il) const {
    return kv->get_v_scale(il);
}

ggml_tensor * llama_kv_cache_context::cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il) const {
    return kv->cpy_k(ctx, k_cur, k_idxs, il, sinfos[i_cur]);
}

ggml_tensor * llama_kv_cache_context::cpy_k_part(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il,
        int64_t n_embd, int64_t i_off) const {
    return kv->cpy_k_part(ctx, k_cur, k_idxs, il, n_embd, i_off);
}

ggml_tensor * llama_kv_cache_context::cpy_v(ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * v_idxs, int32_t il) const {
    return kv->cpy_v(ctx, v_cur, v_idxs, il, sinfos[i_cur]);
}

ggml_tensor * llama_kv_cache_context::build_input_k_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const {
    return kv->build_input_k_idxs(ctx, ubatch);
}

ggml_tensor * llama_kv_cache_context::build_input_v_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const {
    return kv->build_input_v_idxs(ctx, ubatch);
}

std::vector<ggml_tensor *> llama_kv_cache_context::get_k_rot() const {
    return kv->get_k_rot();
}

std::vector<ggml_tensor *> llama_kv_cache_context::get_v_rot() const {
    return kv->get_v_rot();
}

void llama_kv_cache_context::set_input_k_shift(ggml_tensor * dst) const {
    kv->set_input_k_shift(dst);
}

void llama_kv_cache_context::set_input_k_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    kv->set_input_k_idxs(dst, ubatch, sinfos[i_cur]);
}

void llama_kv_cache_context::set_input_v_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    kv->set_input_v_idxs(dst, ubatch, sinfos[i_cur]);
}

void llama_kv_cache_context::set_input_attn_mask(const llm_attn_mask & attn_mask, const llama_ubatch * ubatch, bool causal_attn, ggml_backend_sched_t sched) const {
    kv->set_input_attn_mask(attn_mask, ubatch, causal_attn, sched);
}

void llama_kv_cache_context::set_input_pos_bucket(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    kv->set_input_pos_bucket(dst, ubatch);
}

void llama_kv_cache_context::set_input_relative_pos(ggml_tensor * dst, const llama_ubatch * ubatch, uint32_t extent) const {
    kv->set_input_relative_pos(dst, ubatch, extent);
}

void llama_kv_cache_context::set_input_cell_pos(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    kv->set_input_cell_pos(dst, ubatch);
}

void llama_kv_cache_context::get_prev_tokens(const llama_ubatch & ubatch, uint32_t n, std::vector<llama_token> & res) const {
    kv->get_prev_tokens(ubatch, n, res);
}
