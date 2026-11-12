#include "llama-memory-recurrent.h"
#include "../ggml/src/ggml-recurrent-arena.h"

#include "ggml-backend.h"
#include "llama-impl.h"
#include "llama-io.h"
#include "llama-batch.h"
#include "llama-context.h"
#include "llama-model.h"

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <stdexcept>

//
// llama_memory_recurrent
//

llama_memory_recurrent::llama_memory_recurrent(
        const llama_model & model,
                ggml_type   type_r,
                ggml_type   type_s,
                     bool   offload,
                 uint32_t   mem_size,
                 uint32_t   n_seq_max,
                 uint32_t   n_rs_seq,
    const layer_filter_cb & filter) : hparams(model.hparams), n_seq_max(n_seq_max), n_seq_devices(model.n_seq_devices()) {
    const int32_t n_layer = hparams.n_layer();

    if (n_seq_devices > 1) {
        GGML_ASSERT(offload);
        mem_size = n_seq_max;
    }

    head = 0;
    size = mem_size;
    used = 0;

    this->n_rs_seq = n_rs_seq;
    rs_idx.assign(n_seq_max, 0);
    rs_reach.assign(n_seq_max, 0);

    cells.clear();
    cells.resize(mem_size);

    // spare rows (n_stage in the header): rs_stage moves a row through an f32 tensor, as rs_relocate does
    if (const char * value = getenv("LLAMA_RS_STAGE_ROWS"); value != nullptr && n_seq_devices == 1 &&
            type_r == GGML_TYPE_F32 && type_s == GGML_TYPE_F32) {
        n_stage = (uint32_t) std::clamp(atoi(value), 0, 64);
    }
    r_stage.assign(n_stage, std::vector<ggml_tensor *>(n_layer, nullptr));
    s_stage.assign(n_stage, std::vector<ggml_tensor *>(n_layer, nullptr));

    // Independent layer arenas keep every state's current rows and rollback
    // planes reachable by an access-policy window. They retain the canonical
    // row layout used by tensor splitting, gather, restore and serialization.
    const bool state_arena = model.arch == LLM_ARCH_GLM5NEXT && type_s == GGML_TYPE_F32 &&
                             ggml_env_switch("GGML_CUDA_GDN_STATE_ARENA");
    using arena_key = std::pair<ggml_backend_buffer_type_t, int>;
    // define a comparator for the buft -> ctx map to ensure that the order is well-defined:
    struct ggml_backend_buft_comparator {
        bool operator()(const arena_key & lhs, const arena_key & rhs) const {
            const int order = strcmp(ggml_backend_buft_name(lhs.first), ggml_backend_buft_name(rhs.first));
            return order != 0 ? order < 0 : lhs.second < rhs.second;
        }
    };
    std::map<arena_key, ggml_context_ptr, ggml_backend_buft_comparator> ctx_map;

    // create a context for each buffer type
    auto ctx_for_buft = [&](ggml_backend_buffer_type_t buft, int layer = -1) -> ggml_context * {
        const arena_key key { buft, layer };
        auto it = ctx_map.find(key);
        if (it == ctx_map.end()) {
            ggml_init_params params = {
                /*.mem_size   =*/ size_t((2u + 2u*n_stage)*n_layer*ggml_tensor_overhead()),
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };

            ggml_context * ctx = ggml_init(params);
            if (!ctx) {
                return nullptr;
            }

            ctx_map.emplace(key, ctx);

            return ctx;
        }

        return it->second.get();
    };

    r_l.resize(n_layer);
    s_l.resize(n_layer);

    for (int i = 0; i < n_layer; i++) {
        if (filter && !filter(i)) {
            LLAMA_LOG_DEBUG("%s: layer %3d: skipped\n", __func__, i);
            continue;
        }

        const char * dev_name = "CPU";

        ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();

        if (offload) {
            auto * dev = model.dev_layer(i);
            buft = ggml_backend_dev_buffer_type(dev);

            dev_name = ggml_backend_dev_name(dev);
        }

        LLAMA_LOG_DEBUG("%s, layer %3d: dev = %s\n", __func__, i, dev_name);

        ggml_context * ctx = ctx_for_buft(buft);
        if (!ctx) {
            throw std::runtime_error("failed to create ggml context for rs cache");
        }

        const uint32_t n_rows = mem_size * (1 + n_rs_seq);
        ggml_tensor * r = ggml_new_tensor_2d(ctx, type_r, hparams.n_embd_r(), n_rows);
        ggml_context * ctx_s = state_arena ? ctx_for_buft(buft, i) : ctx;
        if (!ctx_s) {
            throw std::runtime_error("failed to create recurrent state arena");
        }
        ggml_tensor * s = ggml_new_tensor_2d(ctx_s, type_s, hparams.n_embd_s(), n_rows);
        ggml_format_name(r, "cache_r_l%d", i);
        ggml_format_name(s, "cache_s_l%d", i);
        if (n_seq_devices > 1) {
            auto counts = llama_stream_counts(n_seq_max, n_seq_devices);
            for (auto & count : counts) { count *= 1 + n_rs_seq; }
            ggml_backend_meta_split_input(r, 1, counts.data(), counts.size());
            ggml_backend_meta_split_input(s, 1, counts.data(), counts.size());
        }
        r_l[i] = r;
        s_l[i] = s;

        for (uint32_t k = 0; k < n_stage; ++k) {
            // named as the state it holds a row of: a tensor is split over the devices by its name
            // (llama_meta_device_get_split_state), and a spare row has to lie on them as the state's rows do
            r_stage[k][i] = ggml_new_tensor_2d(ctx, type_r, hparams.n_embd_r(), 1);
            s_stage[k][i] = ggml_new_tensor_2d(ctx, type_s, hparams.n_embd_s(), 1);
            ggml_format_name(r_stage[k][i], "cache_r_l%d", i);
            ggml_format_name(s_stage[k][i], "cache_s_l%d", i);
        }
    }

    // a memory none of whose layers is recurrent (an MTP head's) has no row to copy
    if (std::all_of(r_l.begin(), r_l.end(), [](const ggml_tensor * tensor) { return tensor == nullptr; })) {
        n_stage = 0;
        r_stage.clear();
        s_stage.clear();
    }

    if (state_arena) {
        std::map<ggml_backend_buffer_type_t, uint32_t> layers;
        for (const auto & item : ctx_map) {
            if (item.first.second >= 0) { ++layers[item.first.first]; }
        }
        for (const auto & item : ctx_map) {
            if (item.first.second >= 0) {
                ggml_recurrent_arena_mark(s_l[item.first.second], mem_size, layers.at(item.first.first));
            }
        }
    }

    // allocate tensors and initialize the buffers to avoid NaNs in the padding
    for (auto & [key, ctx] : ctx_map) {
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), key.first);
        if (!buf) {
            throw std::runtime_error("failed to allocate buffer for rs cache");
        }
        ggml_backend_buffer_clear(buf, 0);
        LLAMA_LOG_INFO("%s: %10s RS buffer size = %8.2f MiB\n", __func__, ggml_backend_buffer_name(buf), ggml_backend_buffer_get_size(buf)/1024.0/1024.0);
        ctxs_bufs.emplace_back(std::move(ctx), buf);
    }

    {
        const size_t memory_size_r = size_r_bytes();
        const size_t memory_size_s = size_s_bytes();

        LLAMA_LOG_INFO("%s: size = %7.2f MiB (%6u cells, %3d layers, %2u seqs %2u rs_seq), R (%s): %7.2f MiB, S (%s): %7.2f MiB\n", __func__,
                (float)(memory_size_r + memory_size_s) / (1024.0f * 1024.0f), mem_size, n_layer, n_seq_max, n_rs_seq,
                ggml_type_name(type_r), (float)memory_size_r / (1024.0f * 1024.0f),
                ggml_type_name(type_s), (float)memory_size_s / (1024.0f * 1024.0f));
    }

    if (n_stage > 0) {
        size_t row_bytes = 0;
        for (const auto & layers : {r_stage[0], s_stage[0]}) {
            for (const ggml_tensor * tensor : layers) {
                row_bytes += tensor != nullptr ? ggml_nbytes(tensor) : 0;
            }
        }
        LLAMA_LOG_INFO("%s: %u spare rows for staged state copies (LLAMA_RS_STAGE_ROWS), %.3f MiB each\n", __func__,
                n_stage, (float) row_bytes / (1024.0f * 1024.0f));
    }
}

// the graph zeroes the rs_z row of an f32 state in place (llm_graph_context::build_rs); any other
// state type (f16, bf16, a block type) has no in-graph scale op, so its row is zeroed here, before the
// graph runs. All-zero bytes are an exact zero for f16, bf16 and every block type (scale 0, quants 0).
void llama_memory_recurrent::zero_rs_z() {
    if (relocate) {
        // every state type's row is zeroed here: the decode's graph takes no row to zero (get_rs_z), so it is the same
        // graph with a new sequence in the batch as without one. No state is read from the row (find_slot)
        if (rs_z < 0) {
            return;
        }
        if (owner_context) { llama_synchronize(owner_context); }
        for (const auto & layers : {r_l, s_l}) {
            for (ggml_tensor * tensor : layers) {
                if (!tensor) { continue; }
                ggml_backend_tensor_memset(tensor, 0, (size_t) rs_z*tensor->nb[1], tensor->nb[1]);
            }
        }
        return;
    }
    if (n_seq_devices > 1) {
        if (!zero_cells.empty() && owner_context) { llama_synchronize(owner_context); }
        for (const auto & layers : {r_l, s_l}) {
            for (ggml_tensor * tensor : layers) {
                if (!tensor) { continue; }
                for (const uint32_t cell : zero_cells) {
                    ggml_backend_tensor_memset(tensor, 0, state_row(cell, 0)*tensor->nb[1], tensor->nb[1]);
                }
            }
        }
        zero_cells.clear();
        return;
    }
    if (rs_z < 0) {
        return;
    }
    for (ggml_tensor * s : s_l) {
        if (s == nullptr || s->type == GGML_TYPE_F32) {
            continue;
        }
        ggml_backend_tensor_memset(s, 0, (size_t) rs_z * s->nb[1], s->nb[1]);
    }
}

void llama_memory_recurrent::clear(bool data) {
    for (int32_t i = 0; i < (int32_t) size; ++i) {
        cells[i].pos = -1;
        cells[i].seq_id.clear();
        cells[i].src = -1;
        cells[i].tail = -1;
    }

    head = 0;
    used = 0;

    if (data) {
        for (auto & [_, buf] : ctxs_bufs) {
            ggml_backend_buffer_clear(buf.get(), 0);
        }
    }

    std::fill(rs_idx.begin(), rs_idx.end(), 0);
    std::fill(rs_reach.begin(), rs_reach.end(), 0);
    zero_cells.clear();
    restore_cells.clear();
}

bool llama_memory_recurrent::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    uint32_t new_head = size;

    if (p0 < 0) {
        p0 = 0;
    }

    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    if ((uint32_t) seq_id >= this->n_seq_max) {
        LLAMA_LOG_ERROR("%s: invalid seq_id (%d) - larger than n_seq_max (%d)\n", __func__, seq_id, this->n_seq_max);
        return false;
    }

    const bool rm_all = p0 == 0 && p1 == std::numeric_limits<llama_pos>::max();
    if (rm_all) {
        set_rs_idx(seq_id, 0);
    }

    // models like Mamba or RWKV can't have a state partially erased at the end
    // of the sequence because their state isn't preserved for previous tokens
    if (seq_id >= (int64_t) size) {
        // could be fatal
        return false;
    }
    // a cache with no resident recurrent layers holds no state that could be partially
    // erased, so the restriction does not apply to it. this is the glm5next MTP draft
    // context, which runs only the NextN block and filters every KDA layer out.
    const bool has_state = std::any_of(s_l.begin(), s_l.end(),
            [](const ggml_tensor * t) { return t != nullptr; });

    if (0 <= seq_id) {
        int32_t & tail_id = cells[seq_id].tail;
        if (tail_id >= 0) {
            auto & cell = cells[tail_id];

            // partial rollback via per-token snapshot index (bounded by the snapshots the last batch wrote)
            if (0 < p0 && p0 <= cell.pos && p1 > cell.pos) {
                const llama_pos rollback = cell.pos - (p0 - 1);
                // pending rollback is single-use
                const bool pending = rs_idx[seq_id] != 0;
                if (!pending && rollback >= 1 && rollback <= (llama_pos) rs_reach[seq_id]) {
                    set_rs_idx(seq_id, (uint32_t) rollback);
                    cell.pos = p0 - 1;
                    return true;
                }
                if (has_state) {
                    return false;
                }
            }
            // invalidate tails which will be cleared
            if (p0 <= cell.pos && cell.pos < p1) {
                tail_id = -1;
            }
        }
    } else if (seq_id < 0) {
        // seq_id is negative, then the range should include everything or nothing
        if (p0 != p1 && (p0 != 0 || p1 != std::numeric_limits<llama_pos>::max())) {
            //printf("[DEBUG] inside `llama_memory_recurrent::seq_rm`: `seq_id` is negative, so returning false\n");
            return false;
        }
    }

    for (uint32_t i = 0; i < size; ++i) {
        if (cells[i].pos >= p0 && cells[i].pos < p1) {
            if (seq_id < 0) {
                cells[i].seq_id.clear();
            } else if (cells[i].has_seq_id(seq_id)) {
                cells[i].seq_id.erase(seq_id);
            } else {
                continue;
            }
            if (cells[i].is_empty()) {
                // keep count of the number of used cells
                if (cells[i].pos >= 0) {
                    used--;
                }
                cells[i].pos = -1;
                cells[i].src = -1;
                if (new_head == size) {
                    new_head = i;
                }
            }
        }
    }

    // If we freed up a slot, set head to it so searching can start there.
    if (new_head != size && new_head < head) {
        head = new_head;
    }

    return true;
}

void llama_memory_recurrent::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    if (seq_id_src == seq_id_dst) {
        return;
    }

    if (p0 < 0) {
        p0 = 0;
    }

    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    if (n_seq_devices > 1) {
        GGML_ASSERT(seq_id_src >= 0 && uint32_t(seq_id_src) < n_seq_max);
        GGML_ASSERT(seq_id_dst >= 0 && uint32_t(seq_id_dst) < n_seq_max);
        const int32_t source = cells[seq_id_src].tail;
        if (source < 0 || cells[source].pos < p0 || cells[source].pos >= p1) { return; }
        const llama_pos pos = cells[source].pos;
        const uint32_t source_row = state_row(source, rs_idx[seq_id_src]);
        const uint32_t target = llama_seq_to_stream(seq_id_dst, n_seq_max, n_seq_devices);
        if (owner_context) { llama_synchronize(owner_context); }
        seq_rm(seq_id_dst, -1, -1);
        for (const auto & layers : {r_l, s_l}) {
            for (ggml_tensor * tensor : layers) {
                if (!tensor) { continue; }
                std::vector<uint8_t> bytes(tensor->nb[1]);
                ggml_backend_tensor_get(tensor, bytes.data(), source_row*tensor->nb[1], bytes.size());
                ggml_backend_tensor_set(tensor, bytes.data(), state_row(target, 0)*tensor->nb[1], bytes.size());
            }
        }
        auto & cell = cells[target];
        cell.pos = pos;
        cell.src = target;
        cell.seq_id.insert(seq_id_dst);
        cells[seq_id_dst].tail = target;
        rs_idx[seq_id_dst] = rs_reach[seq_id_dst] = 0;
        ++used;
        return;
    }

    if ((uint32_t) seq_id_dst < size && (uint32_t) seq_id_src < size) {
        auto & tail_src = cells[seq_id_src];
        auto & tail_dst = cells[seq_id_dst];
        if (tail_dst.tail >= 0) {
            // clear destination seq_id if it wasn't empty
            auto & cell_dst = cells[tail_dst.tail];

            cell_dst.seq_id.erase(seq_id_dst);
            tail_dst.tail = -1;
            if (cell_dst.seq_id.empty()) {
                cell_dst.pos = -1;
                cell_dst.src = -1;
                used -= 1;
            }
        }
        if (tail_src.tail >= 0) {
            auto & cell_src = cells[tail_src.tail];

            cell_src.seq_id.insert(seq_id_dst);
            tail_dst.tail = tail_src.tail;
        }
        // the destination rolls back only into a batch of its own: the source's snapshots count from the batch end,
        // and a rollback the source has pending already moved the cell back
        rs_reach[seq_id_dst] = 0;
    }
}

void llama_memory_recurrent::seq_keep(llama_seq_id seq_id) {
    uint32_t new_head = size;

    for (uint32_t i = 0; i < size; ++i) {
        if ((llama_seq_id) i != seq_id) {
            cells[i].tail = -1;
        }

        if (!cells[i].has_seq_id(seq_id)) {
            if (cells[i].pos >= 0) {
                used--;
            }

            cells[i].pos = -1;
            cells[i].src = -1;
            cells[i].seq_id.clear();

            if (new_head == size){
                new_head = i;
            }
        } else {
            cells[i].seq_id.clear();
            cells[i].seq_id.insert(seq_id);
        }
    }

    // If we freed up a slot, set head to it so searching can start there.
    if (new_head != size && new_head < head) {
        head = new_head;
    }
}

void llama_memory_recurrent::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    if (shift == 0) {
        return;
    }

    if (p0 < 0) {
        p0 = 0;
    }

    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    // If there is no range then return early to avoid looping over the
    if (p0 == p1) {
        return;
    }

    // for Mamba-like or RWKV models, only the pos needs to be shifted
    if (0 <= seq_id && seq_id < (int64_t) size) {
        const int32_t tail_id = cells[seq_id].tail;
        if (tail_id >= 0) {
            auto & cell = cells[tail_id];
            if (cell.has_seq_id(seq_id) && p0 <= cell.pos && cell.pos < p1) {
                cell.pos += shift;
            }
        }
    }
}

void llama_memory_recurrent::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    if (d == 1) {
        return;
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

    // for Mamba-like or RWKV models, only the pos needs to be changed
    if (0 <= seq_id && seq_id < (int64_t) size) {
        const int32_t tail_id = cells[seq_id].tail;
        if (tail_id >= 0) {
            auto & cell = cells[tail_id];
            if (cell.has_seq_id(seq_id) && p0 <= cell.pos && cell.pos < p1) {
                cell.pos /= d;
            }
        }
    }
}

llama_pos llama_memory_recurrent::seq_pos_min(llama_seq_id seq_id) const {
    llama_pos result = std::numeric_limits<llama_pos>::max();

    for (uint32_t i = 0; i < size; ++i) {
        if (cells[i].has_seq_id(seq_id)) {
            result = std::min(result, cells[i].pos);
        }
    }

    if (result == std::numeric_limits<llama_pos>::max()) {
        result = -1;
    }

    return result;
}

llama_pos llama_memory_recurrent::seq_pos_max(llama_seq_id seq_id) const {
    llama_pos result = -1;

    for (uint32_t i = 0; i < size; ++i) {
        if (cells[i].has_seq_id(seq_id)) {
            result = std::max(result, cells[i].pos);
        }
    }

    return result;
}

void llama_memory_recurrent::set_rs_idx(llama_seq_id seq_id, uint32_t idx) {
    if (seq_id < 0) {
        std::fill(rs_idx.begin(), rs_idx.end(), 0);
        return;
    }

    assert(n_seq_max == rs_idx.size());

    GGML_ASSERT((uint32_t) seq_id < n_seq_max);
    GGML_ASSERT(idx <= n_rs_seq);

    rs_idx[seq_id] = idx;
}

uint32_t llama_memory_recurrent::state_row(uint32_t cell, uint32_t snapshot, bool local) const {
    GGML_ASSERT(cell < size && snapshot <= n_rs_seq);
    if (n_seq_devices == 1) { return snapshot*size + cell; }
    uint32_t first = 0;
    for (const int64_t count : llama_stream_counts(n_seq_max, n_seq_devices)) {
        if (cell < first + count) {
            return (local ? 0 : first*(1 + n_rs_seq)) + snapshot*count + cell - first;
        }
        first += count;
    }
    GGML_ABORT("recurrent cell has no owning device");
}

std::map<ggml_backend_buffer_type_t, size_t> llama_memory_recurrent::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> ret;
    for (const auto & [_, buf] : ctxs_bufs) {
        ret[ggml_backend_buffer_get_type(buf.get())] += ggml_backend_buffer_get_size(buf.get());
    }
    return ret;
}

llama_memory_context_ptr llama_memory_recurrent::init_batch(llama_batch_allocr & balloc, uint32_t n_ubatch, bool embd_all) {
    do {
        balloc.split_reset();

        std::vector<llama_ubatch> ubatches;
        while (true) {
            llama_ubatch ubatch;

            if (embd_all) {
                // if all tokens are output, split by sequence
                ubatch = balloc.split_seq(n_ubatch);
            } else {
                // TODO: non-sequential equal split can be done if using unified KV cache
                //       for simplicity, we always use sequential equal split for now
                // [TAG_RECURRENT_ROLLBACK_SPLITS]
                // the trailing (1 + n_rs_seq) tokens of each seq must stay in the same ubatch
                //   so that the rollback snapshots remain valid
                ubatch = balloc.split_equal(n_ubatch, true, n_rs_seq > 0 ? n_rs_seq + 1 : 0);
            }

            if (ubatch.n_tokens == 0) {
                break;
            }

            ubatches.push_back(std::move(ubatch)); // NOLINT
        }

        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            // failed to find a suitable split
            break;
        }

        if (!prepare(ubatches)) {
            break;
        }

        return std::make_unique<llama_memory_recurrent_context>(this, std::move(ubatches));
    } while (false);

    return std::make_unique<llama_memory_recurrent_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
}

llama_memory_context_ptr llama_memory_recurrent::init_full() {
    return std::make_unique<llama_memory_recurrent_context>(this);
}

llama_memory_context_ptr llama_memory_recurrent::init_update(llama_context * lctx, bool optimize) {
    owner_context = lctx;
    GGML_UNUSED(optimize);

    return std::make_unique<llama_memory_recurrent_context>(LLAMA_MEMORY_STATUS_NO_UPDATE);
}

bool llama_memory_recurrent::prepare(const std::vector<llama_ubatch> & ubatches) {
    // simply remember the full state because it is very small for this type of cache
    // TODO: optimize
    auto org_cells = cells;
    auto org_used = used;
    auto org_head = head;

    bool success = true;

    for (const auto & ubatch : ubatches) {
        if (!find_slot(ubatch)) {
            success = false;
            break;
        }
    }

    // restore the original state
    cells = std::move(org_cells);
    used = org_used;
    head = org_head;

    return success;
}

bool llama_memory_recurrent::find_slot(const llama_ubatch & ubatch) {
    const uint32_t n_seq_tokens = ubatch.n_seq_tokens;
    const uint32_t n_seqs       = ubatch.n_seqs;

    relocate_pending = false;

    if (n_seq_devices > 1) {
        GGML_ASSERT(ubatch.equal_seqs() && n_seqs > 0);
        zero_cells.clear();
        head = llama_seq_to_stream(ubatch.seq_id[0][0], n_seq_max, n_seq_devices);
        n = n_seqs;
        rs_z = -1;
        for (uint32_t s = 0; s < n_seqs; ++s) {
            const uint32_t i = s*n_seq_tokens;
            GGML_ASSERT(ubatch.n_seq_id[i] == 1);
            const llama_seq_id seq = ubatch.seq_id[i][0];
            const uint32_t slot = llama_seq_to_stream(seq, n_seq_max, n_seq_devices);
            auto & cell = cells[slot];
            if (cells[seq].tail < 0) {
                GGML_ASSERT(cell.is_empty());
                zero_cells.push_back(slot);
                ++used;
            } else {
                GGML_ASSERT(cells[seq].tail == (int32_t) slot && cell.has_seq_id(seq));
            }
            cell.pos = ubatch.pos[i + n_seq_tokens - 1];
            cell.src0 = cell.src = slot;
            cell.seq_id.insert(seq);
            cells[seq].tail = slot;
        }
        return true;
    }

    // if we have enough unused cells before the current head ->
    //   better to start searching from the beginning of the cache, hoping to fill it
    if (head > used + 2*n_seqs) {
        head = 0;
    }

    // For recurrent state architectures (like Mamba or RWKV),
    // each cache cell can store the state for a whole sequence.
    // A slot should be always be contiguous.

    // can only process batches with an equal number of new tokens in each sequence
    GGML_ASSERT(ubatch.equal_seqs());

    int32_t min = size - 1;
    int32_t max = 0;

    // everything should fit if all seq_ids are smaller than the max
    for (uint32_t s = 0; s < n_seqs; ++s) {
        const uint32_t i = s*n_seq_tokens; // first token of sequence set s
        const uint32_t n_seq_id = ubatch.n_seq_id[i];

        for (uint32_t j = 0; j < n_seq_id; ++j) {
            const llama_seq_id seq_id = ubatch.seq_id[i][j];

            if (seq_id < 0 || (uint32_t) seq_id >= size) {
                // too big seq_id
                // TODO: would it be possible to resize the cache instead?
                LLAMA_LOG_ERROR("%s: seq_id=%d >= n_seq_max=%u Try using a bigger --parallel value\n", __func__, seq_id, n_seq_max);
                return false;
            }
            if (j > 0) {
                auto & seq = cells[seq_id];
                if (seq.tail >= 0) {
                    auto & cell = cells[seq.tail];
                    // clear cells from seq_ids that become shared
                    // (should not normally happen, but let's handle it anyway)
                    cell.seq_id.erase(seq_id);
                    seq.tail = -1;
                    if (cell.seq_id.empty()) {
                        cell.pos = -1;
                        cell.src = -1;
                        used -= 1;
                    }
                }
            }
        }
    }

#ifndef NDEBUG
    {
        std::vector<int32_t> tails_verif;
        tails_verif.assign(size, -1);
        for (uint32_t i = 0; i < size; ++i) {
            auto & cell = cells[i];
            for (llama_seq_id seq_id : cell.seq_id) {
                if (tails_verif[seq_id] != -1) {
                    LLAMA_LOG_ERROR("%s: duplicate tail for seq_id %d in cell %d and %d\n", __func__, seq_id, i, tails_verif[seq_id]);
                }
                tails_verif[seq_id] = i;
            }
        }
        for (uint32_t i = 0; i < size; ++i) {
            if (tails_verif[i] != cells[i].tail) {
                LLAMA_LOG_ERROR("%s: wrong tail for seq_id %d, (%d instead of %d)\n", __func__, i, cells[i].tail, tails_verif[i]);
            }
        }
    }
#endif

    // find next empty cell
    uint32_t next_empty_cell = head;

    for (uint32_t i = 0; i < size; ++i) {
        if (next_empty_cell >= size) { next_empty_cell -= size; }
        auto & cell = cells[next_empty_cell];
        if (cell.is_empty()) { break; }
        next_empty_cell += 1;
    }

    // find usable cell range
    for (uint32_t s = 0; s < n_seqs; ++s) {
        const uint32_t i = s*n_seq_tokens;
        const llama_seq_id seq_id = ubatch.seq_id[i][0];
        auto & seq_meta = cells[seq_id];
        bool has_cell = false;
        if (seq_meta.tail >= 0) {
            auto & cell = cells[seq_meta.tail];
            GGML_ASSERT(cell.has_seq_id(seq_id));
            // does this seq_id "own" the cell?
            if (cell.seq_id.size() == 1) { has_cell = true; }
        }
        if (!has_cell) {
            auto & empty_cell = cells[next_empty_cell];
            GGML_ASSERT(empty_cell.is_empty());
            // copy old tail into the empty cell
            if (seq_meta.tail >= 0) {
                auto & orig_cell = cells[seq_meta.tail];
                empty_cell.pos = orig_cell.pos;
                empty_cell.src = orig_cell.src;
                orig_cell.seq_id.erase(seq_id);
                empty_cell.seq_id.insert(seq_id); // will be overwritten
                GGML_ASSERT(!orig_cell.is_empty()); // has at least one remaining seq_id
            }
            seq_meta.tail = next_empty_cell;
            // find next empty cell
            if (s + 1 < n_seqs) {
                for (uint32_t j = 0; j < size; ++j) {
                    next_empty_cell += 1;
                    if (next_empty_cell >= size) { next_empty_cell -= size; }
                    auto & cell = cells[next_empty_cell];
                    if (cell.is_empty()) { break; }
                }
            }
        }
        if (min > seq_meta.tail) { min = seq_meta.tail; }
        if (max < seq_meta.tail) { max = seq_meta.tail; }
    }

    // relocate: the set is gathered at the first cell, and what it displaces lies up to the highest cell it held
    if (relocate) {
        min = 0;
        max = std::max<int32_t>(max, (int32_t) n_seqs - 1);
    }

    // gather and re-order
    for (uint32_t s = 0; s < n_seqs; ++s) {
        const uint32_t i = s*n_seq_tokens;
        const int32_t dst_id = s + min;
        const int32_t src_id = cells[ubatch.seq_id[i][0]].tail;
        if (dst_id != src_id) {
            auto & dst_cell = cells[dst_id];
            auto & src_cell = cells[src_id];

            std::swap(dst_cell.pos, src_cell.pos);
            std::swap(dst_cell.src, src_cell.src);
            std::swap(dst_cell.seq_id, src_cell.seq_id);

            // swap tails
            for (uint32_t j = 0; j < size; ++j) {
                int32_t & tail = cells[j].tail;
                if (tail == src_id) {
                    tail = dst_id;
                } else if (tail == dst_id) {
                    tail = src_id;
                }
            }
        }
    }

    // update the pos of the used seqs
    for (uint32_t s = 0; s < n_seqs; ++s) {
        const uint32_t i = s*n_seq_tokens;
        const llama_pos last_pos = ubatch.pos[i + n_seq_tokens - 1];
        const int32_t cell_id = s + min;
        auto & cell = cells[cell_id];

        if (cell.pos >= 0 && last_pos != cell.pos + (llama_pos) n_seq_tokens) {
            // What should happen when the pos backtracks or skips a value?
            // Clearing the state mid-batch would require special-casing which isn't done.
            LLAMA_LOG_WARN("%s: non-consecutive token position %d after %d for sequence %d with %u new tokens\n",
                __func__, last_pos, cell.pos, ubatch.seq_id[i][0], n_seq_tokens);
        }
        cell.pos = last_pos;
        cell.seq_id.clear();
        for (int32_t j = 0; j < ubatch.n_seq_id[i]; ++j) {
            const llama_seq_id seq_id = ubatch.seq_id[i][j];
            cell.seq_id.insert(seq_id);
            cells[seq_id].tail = cell_id;
        }
    }

    // Find first cell without src refs, to use as the zero-ed state
    {
        // TODO: bake-in src refcounts in the cell metadata
        std::vector<int32_t> refcounts(size, 0);
        for (size_t i = 0; i < size; ++i) {
            const int32_t src = cells[i].src;
            if (src >= 0) {
                refcounts[src] += 1;
            }
        }

        rs_z = -1;
        for (int i = min; i <= max; ++i) {
            if (refcounts[i] == 0) {
                rs_z = i;
                break;
            }
        }

        for (int i = min; i <= max; ++i) {
            if (cells[i].src < 0) {
                GGML_ASSERT(rs_z >= 0);
                cells[i].src0 = rs_z;
            } else {
                // Stage the source ids for all used cells to allow correct seq_* behavior
                // and still make these values available when setting the inputs
                cells[i].src0 = cells[i].src;
            }
            cells[i].src = i; // avoid moving or clearing twice
        }
    }

    // allow getting the range of used cells, from head to head + n
    head = min;
    n    = max - min + 1;

    // relocate: only a state another sequence keeps past the set's cells has to move (a state of the set is read from
    // the cell it was in and written to the cell it takes); with none, the decode's graph has the set's cells alone
    if (relocate) {
        for (int i = (int) n_seqs; i <= max; ++i) {
            relocate_pending = relocate_pending || (!cells[i].is_empty() && cells[i].src0 != i);
        }
        // and a state of the set that lies in a cell another sequence of the set is written to (a set in another order,
        // a set closing up over a cell that was left, a second new sequence reading the first one's zeroed cell)
        for (int i = 0; relocate_disjoint && i < (int) n_seqs; ++i) {
            relocate_pending = relocate_pending || (cells[i].src0 != i && cells[i].src0 < (int32_t) n_seqs);
        }
        if (!relocate_pending) {
            n = n_seqs;
        }
    }
    used = std::count_if(cells.begin(), cells.end(),
        [](const mem_cell & cell){ return !cell.is_empty(); });

    // sanity check
    return n >= n_seqs;
}

void llama_memory_recurrent::set_relocate(bool value) {
    // f32 states only: a row is moved through an f32 tensor of the graph's, which a block type would not survive bit
    // for bit
    bool f32 = true;
    for (const auto & layers : {r_l, s_l}) {
        for (const ggml_tensor * tensor : layers) {
            f32 = f32 && (tensor == nullptr || tensor->type == GGML_TYPE_F32);
        }
    }
    relocate = value && n_seq_devices == 1 && f32;
    relocate_disjoint = relocate && !ggml_env_switch("LLAMA_RS_GATHER_LEGACY");
}

void llama_memory_recurrent::relocate_states(uint32_t n_seqs) {
    if (!relocate_pending) {
        return;
    }
    relocate_pending = false;
    // n is n_seqs when only states of the set move (relocate_disjoint)
    GGML_ASSERT(owner_context != nullptr && head == 0 && n >= n_seqs && n <= size);

    // The row each row takes its state from, every other row its own. A cell of the batch takes its sequence's state,
    // at the snapshot of a pending rollback, which this takes as s_copy does: the decode writes its snapshots anew. A
    // cell another sequence keeps takes that sequence's state and its snapshots, each in its group, so a rollback it has
    // pending or is asked for later (a verify's, when a second ubatch of the same decode displaced it) reads its own
    const uint32_t n_groups = 1 + n_rs_seq;
    std::vector<int32_t> rows(size*n_groups);
    for (uint32_t r = 0; r < rows.size(); ++r) {
        rows[r] = r;
    }
    for (uint32_t i = 0; i < n; ++i) {
        if (cells[i].is_empty()) {
            continue;
        }
        if (i < n_seqs) {
            const llama_seq_id seq = *cells[i].seq_id.begin();
            rows[i] = (int32_t) (rs_idx[seq]*size) + cells[i].src0;
            rs_idx[seq] = 0;
        } else {
            for (uint32_t g = 0; g < n_groups; ++g) {
                rows[g*size + i] = (int32_t) (g*size) + cells[i].src0;
            }
        }
    }

    std::vector<ggml_tensor *> states;
    for (const auto & layers : {r_l, s_l}) {
        for (ggml_tensor * tensor : layers) {
            if (tensor) { states.push_back(tensor); }
        }
    }
    // a context whose layers keep no state here (the MTP head's) has cells and nothing to move
    if (!states.empty()) {
        llama_synchronize(owner_context);
        owner_context->rs_relocate(states, rows);
    }

    for (uint32_t i = 0; i < n; ++i) {
        cells[i].src0 = i;
    }
    n = n_seqs;
}

bool llama_memory_recurrent::get_can_shift() const {
    // shifting the pos is trivial for recurrent models
    return true;
}

size_t llama_memory_recurrent::total_size() const {
    size_t size = 0;
    for (const auto & [_, buf] : ctxs_bufs) {
        size += ggml_backend_buffer_get_size(buf.get());
    }

    return size;
}

size_t llama_memory_recurrent::size_r_bytes() const {
    size_t size_r_bytes = 0;

    for (const auto & r : r_l) {
        if (r != nullptr) {
            size_r_bytes += ggml_nbytes(r);
        }
    }

    return size_r_bytes;
}

size_t llama_memory_recurrent::size_s_bytes() const {
    size_t size_s_bytes = 0;

    for (const auto & s : s_l) {
        if (s != nullptr) {
            size_s_bytes += ggml_nbytes(s);
        }
    }

    return size_s_bytes;
}

void llama_memory_recurrent::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    GGML_UNUSED(flags);

    std::vector<std::pair<uint32_t, uint32_t>> cell_ranges; // ranges, from inclusive, to exclusive
    std::vector<std::pair<uint32_t, uint32_t>> cell_ranges_data; // logical source row ranges
    uint32_t cell_count = 0;

    // Count the number of cells with the specified seq_id
    // Find all the ranges of cells with this seq id (or all, when -1)
    uint32_t cell_range_begin = size;
    for (uint32_t i = 0; i < size; ++i) {
        const auto & cell = cells[i];
        // TODO: fix incosistent handling of `seq_id < 0` and `seq_id == -1` in the codebase [TAG_LLAMA_SEQ_ID_NEG]
        if ((seq_id == -1 && !cell.is_empty()) || cell.has_seq_id(seq_id)) {
            ++cell_count;
            uint32_t rs_idx_cur = 0;

            if (n_rs_seq != 0) {
                if (seq_id != -1) {
                    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < rs_idx.size());
                    rs_idx_cur = rs_idx[seq_id];
                } else {
                    bool has_rs_idx = false;
                    for (const llama_seq_id cell_seq_id : cell.seq_id) {
                        GGML_ASSERT(cell_seq_id >= 0 && (size_t) cell_seq_id < rs_idx.size());

                        const uint32_t seq_rs_idx = rs_idx[cell_seq_id];
                        if (!has_rs_idx) {
                            rs_idx_cur = seq_rs_idx;
                            has_rs_idx = true;
                        } else if (rs_idx_cur != seq_rs_idx) {
                            GGML_ABORT("cannot write shared recurrent state with different rollback indices");
                        }
                    }
                }
            }

            const uint32_t cell_id = state_row(cell.src >= 0 ? cell.src : (int32_t) i, rs_idx_cur);
            if (cell_ranges_data.empty() || cell_ranges_data.back().second != cell_id) {
                cell_ranges_data.emplace_back(cell_id, cell_id + 1);
            } else {
                cell_ranges_data.back().second++;
            }

            if (cell_range_begin == size) {
                cell_range_begin = i;
            }
        } else {
            if (cell_range_begin != size) {
                cell_ranges.emplace_back(cell_range_begin, i);
                cell_range_begin = size;
            }
        }
    }
    if (cell_range_begin != size) {
        cell_ranges.emplace_back(cell_range_begin, size);
    }

    if ((flags & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE) && cell_ranges.size() > 1) {
        GGML_ABORT("cannot save/load multiple ranges of cells to/from device memory\n");
    }

    // DEBUG CHECK: Sum of cell counts in ranges should equal the total cell count
    uint32_t cell_count_check = 0;
    for (const auto & range : cell_ranges) {
        cell_count_check += range.second - range.first;
    }
    GGML_ASSERT(cell_count == cell_count_check);

    cell_count_check = 0;
    for (const auto & range : cell_ranges_data) {
        cell_count_check += range.second - range.first;
    }
    GGML_ASSERT(cell_count == cell_count_check);

    io.write(&cell_count, sizeof(cell_count));

    state_write_meta(io, cell_ranges, seq_id);
    state_write_data(io, cell_ranges_data);
}

int64_t llama_memory_recurrent::stage_src(llama_seq_id seq_id) const {
    if (n_stage == 0 || seq_id < 0 || (size_t) seq_id >= cells.size() || (size_t) seq_id >= rs_idx.size()) {
        return -1;
    }
    const int32_t tail = cells[seq_id].tail;
    if (tail < 0 || cells[tail].seq_id.size() != 1 || !cells[tail].has_seq_id(seq_id)) {
        return -1;
    }
    // the row state_write reads for the cell
    return state_row(cells[tail].src >= 0 ? cells[tail].src : tail, n_rs_seq != 0 ? rs_idx[seq_id] : 0);
}

void llama_memory_recurrent::stage_tensors(uint32_t row, std::vector<ggml_tensor *> & states, std::vector<ggml_tensor *> & stages) const {
    GGML_ASSERT(row < n_stage);
    for (size_t il = 0; il < r_l.size(); ++il) {
        if (r_l[il] != nullptr) {
            states.push_back(r_l[il]);
            stages.push_back(r_stage[row][il]);
        }
    }
    for (size_t il = 0; il < s_l.size(); ++il) {
        if (s_l[il] != nullptr) {
            states.push_back(s_l[il]);
            stages.push_back(s_stage[row][il]);
        }
    }
}

void llama_memory_recurrent::state_write_staged(llama_io_write_i & io, llama_seq_id seq_id, uint32_t row) const {
    GGML_ASSERT(row < n_stage && stage_src(seq_id) >= 0);

    // as state_write for the sequence's one cell
    const uint32_t cell       = cells[seq_id].tail;
    const uint32_t cell_count = 1;

    io.write(&cell_count, sizeof(cell_count));

    state_write_meta(io, {{cell, cell + 1}}, seq_id);

    // as state_write_data, each tensor's row read from the spare row: these stay where they are, so a writer that
    // copies its tensors later may (write_tensor, not write_tensor_now)
    const uint32_t s_trans = 0;
    const uint32_t n_layer = hparams.n_layer();

    io.write(&s_trans, sizeof(s_trans));
    io.write(&n_layer, sizeof(n_layer));

    for (uint32_t il = 0; il < n_layer; ++il) {
        if (r_l[il] == nullptr) continue;

        const int32_t  r_type_i   = (int32_t) r_l[il]->type;
        const uint64_t r_size_row = ggml_row_size(r_l[il]->type, hparams.n_embd_r());

        io.write(&r_type_i,   sizeof(r_type_i));
        io.write(&r_size_row, sizeof(r_size_row));
        io.write_tensor(r_stage[row][il], 0, r_size_row);
    }

    for (uint32_t il = 0; il < n_layer; ++il) {
        if (s_l[il] == nullptr) continue;

        const int32_t  s_type_i   = (int32_t) s_l[il]->type;
        const uint64_t s_size_row = ggml_row_size(s_l[il]->type, hparams.n_embd_s());

        io.write(&s_type_i,   sizeof(s_type_i));
        io.write(&s_size_row, sizeof(s_size_row));
        io.write_tensor(s_stage[row][il], 0, s_size_row);
    }
}

void llama_memory_recurrent::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    GGML_UNUSED(flags);

    uint32_t cell_count;
    io.read(&cell_count, sizeof(cell_count));

    bool res = true;

    res = res && state_read_meta(io, cell_count, seq_id);

    try {
        res = res && state_read_data(io, cell_count);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: %s\n", __func__, err.what());
        res = false;
    } catch (...) {
        res = false;
    }

    if (!res) {
        // TODO: fix incosistent handling of `seq_id < 0` and `seq_id == -1` in the codebase [TAG_LLAMA_SEQ_ID_NEG]
        if (seq_id == -1) {
            clear(true);
        } else {
            seq_rm(seq_id, -1, -1);
        }
        throw std::runtime_error("failed to restore kv cache");
    }

    if (n_rs_seq != 0) {
        set_rs_idx(seq_id, 0);
    }

    // a restored state comes without the snapshots of the batch that reached it
    if (seq_id < 0) {
        std::fill(rs_reach.begin(), rs_reach.end(), 0);
    } else {
        rs_reach[seq_id] = 0;
    }
}

void llama_memory_recurrent::state_write_meta(llama_io_write_i & io, const std::vector<std::pair<uint32_t, uint32_t>> & cell_ranges, llama_seq_id seq_id) const {
    for (const auto & range : cell_ranges) {
        for (uint32_t i = range.first; i < range.second; ++i) {
            const auto & cell = cells[i];
            const llama_pos pos      = cell.pos;
            const uint32_t  n_seq_id = seq_id == -1 ? cell.seq_id.size() : 0;

            io.write(&pos,      sizeof(pos));
            io.write(&n_seq_id, sizeof(n_seq_id));

            if (n_seq_id) {
                for (auto seq_id : cell.seq_id) {
                    io.write(&seq_id, sizeof(seq_id));
                }
            }
        }
    }
}

void llama_memory_recurrent::state_write_data(llama_io_write_i & io, const std::vector<std::pair<uint32_t, uint32_t>> & cell_ranges) const {
    const uint32_t s_trans = 0;
    const uint32_t n_layer = hparams.n_layer();

    io.write(&s_trans, sizeof(s_trans));
    io.write(&n_layer, sizeof(n_layer));

    // Iterate and write all the R tensors first, each row is a cell
    // Get whole range at a time
    for (uint32_t il = 0; il < n_layer; ++il) {
        // skip null layers (read_data will handle this by checking "r_l" and "s_l" for null)
        if (r_l[il] == nullptr) continue;

        // Write R tensor type
        const int32_t r_type_i = (int32_t)r_l[il]->type;
        io.write(&r_type_i, sizeof(r_type_i));

        // Write row size of R tensor
        const uint64_t r_size_row = ggml_row_size(r_l[il]->type, hparams.n_embd_r());
        io.write(&r_size_row, sizeof(r_size_row));

        // Write each logical cell row range. With pending recurrent rollback,
        // the logical current state may live in a rollback snapshot plane.
        for (const auto & range : cell_ranges) {
            const size_t range_size = range.second - range.first;
            const size_t buf_size = range_size * r_size_row;
            io.write_tensor_now(r_l[il], range.first * r_size_row, buf_size);
        }
    }

    if (!s_trans) {
        for (uint32_t il = 0; il < n_layer; ++il) {
            // skip null layers (read_data will handle this by checking "r_l" and "s_l" for null)
            if (s_l[il] == nullptr) continue;

            // Write S tensor type
            const int32_t s_type_i = (int32_t)s_l[il]->type;
            io.write(&s_type_i, sizeof(s_type_i));

            // Write row size of S tensor
            const uint64_t s_size_row = ggml_row_size(s_l[il]->type, hparams.n_embd_s());
            io.write(&s_size_row, sizeof(s_size_row));

            // Write each logical cell row range. With pending recurrent rollback,
            // the logical current state may live in a rollback snapshot plane.
            for (const auto & range : cell_ranges) {
                const size_t range_size = range.second - range.first;
                const size_t buf_size = range_size * s_size_row;
                io.write_tensor_now(s_l[il], range.first * s_size_row, buf_size);
            }
        }
    } else {
        // When S tensor is transposed, we also need the element size and get the element ranges from each row
        const uint32_t mem_size = size;
        for (uint32_t il = 0; il < n_layer; ++il) {
            // skip null layers (read_data will handle this by checking "r_l" and "s_l" for null)
            if (s_l[il] == nullptr) continue;

            const uint32_t n_embd_s = hparams.n_embd_s();

            // Write S tensor type
            const int32_t s_type_i = (int32_t)s_l[il]->type;
            io.write(&s_type_i, sizeof(s_type_i));

            // Write element size
            const uint32_t s_size_el = ggml_type_size(s_l[il]->type);
            io.write(&s_size_el, sizeof(s_size_el));

            // Write GQA embedding size
            io.write(&n_embd_s, sizeof(n_embd_s));

            // For each row, we get the element values of each logical cell
            for (uint32_t j = 0; j < n_embd_s; ++j) {
                for (const auto & range : cell_ranges) {
                    const size_t range_size = range.second - range.first;
                    const size_t src_offset = (range.first + j * mem_size) * s_size_el;
                    const size_t buf_size = range_size * s_size_el;
                    io.write_tensor_now(s_l[il], src_offset, buf_size);
                }
            }
        }
    }
}

bool llama_memory_recurrent::state_read_meta(llama_io_read_i & io, uint32_t cell_count, llama_seq_id dest_seq_id) {
    if (n_seq_devices > 1) {
        if (cell_count > n_seq_max || (dest_seq_id >= 0 && (uint32_t(dest_seq_id) >= n_seq_max || cell_count > 1))) {
            return false;
        }
        if (dest_seq_id < 0) { clear(true); }
        else { seq_rm(dest_seq_id, -1, -1); }
        restore_cells.clear();
        for (uint32_t i = 0; i < cell_count; ++i) {
            llama_pos pos;
            uint32_t count;
            io.read(&pos, sizeof(pos));
            io.read(&count, sizeof(count));
            llama_seq_id seq = dest_seq_id;
            if (dest_seq_id < 0) {
                if (count != 1) { return false; }
                io.read(&seq, sizeof(seq));
            } else if (count != 0) { return false; }
            if (seq < 0 || uint32_t(seq) >= n_seq_max || cells[seq].tail >= 0) { return false; }
            const uint32_t slot = llama_seq_to_stream(seq, n_seq_max, n_seq_devices);
            auto & cell = cells[slot];
            cell.pos = pos;
            cell.src = slot;
            cell.seq_id.insert(seq);
            cells[seq].tail = slot;
            restore_cells.push_back(slot);
            ++used;
        }
        return true;
    }
    if (dest_seq_id != -1) {
        // single sequence
        seq_rm(dest_seq_id, -1, -1);

        if (cell_count == 0) {
            return true;
        }

        llama_batch_allocr balloc(hparams.n_pos_per_embd());

        llama_ubatch ubatch = balloc.ubatch_reserve(cell_count, 1);

        for (uint32_t i = 0; i < cell_count; ++i) {
            llama_pos pos;
            uint32_t n_seq_id;

            io.read(&pos,      sizeof(pos));
            io.read(&n_seq_id, sizeof(n_seq_id));

            if (n_seq_id != 0) {
                LLAMA_LOG_ERROR("%s: invalid seq_id-agnostic kv cell\n", __func__);
                return false;
            }

            ubatch.pos[i] = pos;
        }
        ubatch.n_seq_id[0] = 1;
        ubatch.seq_id[0] = &dest_seq_id;

        if (!find_slot(ubatch)) {
            LLAMA_LOG_ERROR("%s: failed to find available cells in kv cache\n", __func__);
            return false;
        }

        // DEBUG CHECK: kv.head should be our first cell, kv.head + cell_count - 1 should be our last cell (verify seq_id and pos values)
        // Assume that this is one contiguous block of cells
        GGML_ASSERT(head + cell_count <= size);
        GGML_ASSERT(cells[head].pos == ubatch.pos[0]);
        GGML_ASSERT(cells[head + cell_count - 1].pos == ubatch.pos[cell_count - 1]);
        GGML_ASSERT(cells[head].has_seq_id(dest_seq_id));
        GGML_ASSERT(cells[head + cell_count - 1].has_seq_id(dest_seq_id));
    } else {
        // whole KV cache restore

        if (cell_count > size) {
            LLAMA_LOG_ERROR("%s: not enough cells in kv cache\n", __func__);
            return false;
        }

        clear(true);

        for (uint32_t i = 0; i < cell_count; ++i) {
            auto & cell = cells[i];

            llama_pos pos;
            uint32_t  n_seq_id;

            io.read(&pos,      sizeof(pos));
            io.read(&n_seq_id, sizeof(n_seq_id));

            cell.pos = pos;

            for (uint32_t j = 0; j < n_seq_id; ++j) {
                llama_seq_id seq_id;
                io.read(&seq_id, sizeof(seq_id));

                if (seq_id < 0 || (uint32_t) seq_id >= this->n_seq_max) {
                    LLAMA_LOG_ERROR("%s: invalid seq_id, %d is out of range [0, %u)\n", __func__, seq_id, this->n_seq_max);
                    return false;
                }

                cell.seq_id.insert(seq_id);

                int32_t & tail = cells[seq_id].tail;
                if (tail != -1) {
                    LLAMA_LOG_ERROR("%s: duplicate tail for seq_id %d in cell %d and %d\n", __func__, seq_id, i, tail);
                    return false;
                }
                tail = i;
            }
        }

        head = 0;
        used = cell_count;
    }

    for (uint32_t i = 0; i < cell_count; ++i) {
        uint32_t cell_id = head + i;
        // make sure the recurrent states will keep their restored state
        cells[cell_id].src = cell_id;
    }

    return true;
}

bool llama_memory_recurrent::state_read_data(llama_io_read_i & io, uint32_t cell_count) {
    uint32_t s_trans;
    uint32_t n_layer;
    io.read(&s_trans, sizeof(s_trans));
    io.read(&n_layer, sizeof(n_layer));

    if (n_layer != hparams.n_layer()) {
        LLAMA_LOG_ERROR("%s: mismatched layer count (%u instead of %u)\n", __func__, n_layer, hparams.n_layer());
        return false;
    }
    if (cell_count > size) {
        LLAMA_LOG_ERROR("%s: not enough cells in kv cache to restore state (%u > %u)\n", __func__, cell_count, size);
        return false;
    }
    if (false != (bool) s_trans) {
        LLAMA_LOG_ERROR("%s: incompatible s transposition\n", __func__);
        return false;
    }

    // For each layer, read the keys for each cell, one row is one cell, read as one contiguous block
    for (uint32_t il = 0; il < n_layer; ++il) {
        // skip null layers
        if (r_l[il] == nullptr) continue;

        // Read type of key
        int32_t r_type_i_ref;
        io.read(&r_type_i_ref, sizeof(r_type_i_ref));
        const int32_t r_type_i = (int32_t) r_l[il]->type;
        if (r_type_i != r_type_i_ref) {
            LLAMA_LOG_ERROR("%s: mismatched r type (%d != %d, layer %d)\n", __func__, r_type_i, r_type_i_ref, il);
            return false;
        }

        // Read row size of key
        uint64_t r_size_row_ref;
        io.read(&r_size_row_ref, sizeof(r_size_row_ref));
        const size_t r_size_row = ggml_row_size(r_l[il]->type, hparams.n_embd_r());
        if (r_size_row != r_size_row_ref) {
            LLAMA_LOG_ERROR("%s: mismatched r row size (%zu != %zu, layer %d)\n", __func__, r_size_row, (size_t) r_size_row_ref, il);
            return false;
        }

        if (cell_count) {
            // Read and set the keys for the whole cell range
            if (n_seq_devices > 1) {
                for (const uint32_t cell : restore_cells) { io.read_tensor_now(r_l[il], state_row(cell, 0)*r_size_row, r_size_row); }
            } else {
                io.read_tensor_now(r_l[il], head * r_size_row, cell_count * r_size_row);
            }
        }
    }

    if (!s_trans) {
        for (uint32_t il = 0; il < n_layer; ++il) {
            // skip null layers
            if (s_l[il] == nullptr) continue;

            // Read type of value
            int32_t s_type_i_ref;
            io.read(&s_type_i_ref, sizeof(s_type_i_ref));
            const int32_t s_type_i = (int32_t)s_l[il]->type;

            if (s_type_i != s_type_i_ref) {
                LLAMA_LOG_ERROR("%s: mismatched s type (%d != %d, layer %d)\n", __func__, s_type_i, s_type_i_ref, il);
                return false;
            }

            // Read row size of value
            uint64_t s_size_row_ref;
            io.read(&s_size_row_ref, sizeof(s_size_row_ref));
            const size_t s_size_row = ggml_row_size(s_l[il]->type, hparams.n_embd_s());
            if (s_size_row != s_size_row_ref) {
                LLAMA_LOG_ERROR("%s: mismatched s row size (%zu != %zu, layer %d)\n", __func__, s_size_row, (size_t) s_size_row_ref, il);
                return false;
            }

            if (cell_count) {
                // Read and set the values for the whole cell range
                if (n_seq_devices > 1) {
                    for (const uint32_t cell : restore_cells) { io.read_tensor_now(s_l[il], state_row(cell, 0)*s_size_row, s_size_row); }
                } else {
                    io.read_tensor_now(s_l[il], head * s_size_row, cell_count * s_size_row);
                }
            }
        }
    } else {
        // For each layer, read the values for each cell (transposed)
        for (uint32_t il = 0; il < n_layer; ++il) {
            // skip null layers
            if (s_l[il] == nullptr) continue;

            const uint32_t n_embd_s = hparams.n_embd_s();

            // Read type of value
            int32_t s_type_i_ref;
            io.read(&s_type_i_ref, sizeof(s_type_i_ref));
            const int32_t s_type_i = (int32_t)s_l[il]->type;
            if (s_type_i != s_type_i_ref) {
                LLAMA_LOG_ERROR("%s: mismatched s type (%d != %d, layer %d)\n", __func__, s_type_i, s_type_i_ref, il);
                return false;
            }

            // Read element size of value
            uint32_t s_size_el_ref;
            io.read(&s_size_el_ref, sizeof(s_size_el_ref));
            const size_t s_size_el = ggml_type_size(s_l[il]->type);
            if (s_size_el != s_size_el_ref) {
                LLAMA_LOG_ERROR("%s: mismatched s element size (%zu != %zu, layer %d)\n", __func__, s_size_el, (size_t) s_size_el_ref, il);
                return false;
            }

            // Read state embedding size
            uint32_t n_embd_s_ref;
            io.read(&n_embd_s_ref, sizeof(n_embd_s_ref));
            if (n_embd_s != n_embd_s_ref) {
                LLAMA_LOG_ERROR("%s: mismatched s embedding size (%u != %u, layer %d)\n", __func__, n_embd_s, n_embd_s_ref, il);
                return false;
            }

            if (cell_count) {
                // For each row in the transposed matrix, read the values for the whole cell range
                for (uint32_t j = 0; j < n_embd_s; ++j) {
                    const size_t dst_offset = (head + j * size) * s_size_el;
                    io.read_tensor_now(s_l[il], dst_offset, cell_count * s_size_el);
                }
            }
        }
    }

    return true;
}

//
// llama_memory_recurrent_context
//

llama_memory_recurrent_context::llama_memory_recurrent_context(llama_memory_status status) : status(status) {}

llama_memory_recurrent_context::llama_memory_recurrent_context(
        llama_memory_recurrent * mem) : status(LLAMA_MEMORY_STATUS_SUCCESS), mem(mem), is_full(true) {
}

llama_memory_recurrent_context::llama_memory_recurrent_context(
        llama_memory_recurrent * mem,
        std::vector<llama_ubatch> ubatches) : status(LLAMA_MEMORY_STATUS_SUCCESS), mem(mem), ubatches(std::move(ubatches)) {}

llama_memory_recurrent_context::~llama_memory_recurrent_context() = default;

bool llama_memory_recurrent_context::next() {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    if (++i_next >= ubatches.size()) {
        return false;
    }

    return true;
}

bool llama_memory_recurrent_context::apply() {
    assert(!llama_memory_status_is_fail(status));

    // no ubatches -> this is an update
    if (ubatches.empty()) {
        // recurrent cache never performs updates
        assert(status == LLAMA_MEMORY_STATUS_NO_UPDATE);

        return true;
    }

    const llama_ubatch & ubatch = ubatches[i_next];

    mem->find_slot(ubatch);
    mem->zero_rs_z();
    mem->relocate_states(ubatch.n_seqs);

    for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
        mem->rs_reach[ubatch.seq_id_unq[s]] = std::min(mem->n_rs_seq, ubatch.n_seq_tokens - 1);
    }

    return true;
}

llama_memory_status llama_memory_recurrent_context::get_status() const {
    return status;
}

const llama_ubatch & llama_memory_recurrent_context::get_ubatch() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    return ubatches[i_next];
}

uint32_t llama_memory_recurrent_context::get_n_rs() const {
    return is_full ? mem->size : mem->n;
}

uint32_t llama_memory_recurrent_context::get_head() const {
    return is_full ? 0 : mem->head;
}

int32_t llama_memory_recurrent_context::get_rs_z() const {
    // relocate: the row is zeroed before the decode (zero_rs_z), not by its graph
    return is_full ? 0 : mem->relocate ? -1 : mem->rs_z;
}

bool llama_memory_recurrent_context::rows_disjoint() const {
    return !is_full && mem->relocate_disjoint;
}

uint32_t llama_memory_recurrent_context::get_size() const {
    return mem->size;
}

ggml_tensor * llama_memory_recurrent_context::get_r_l(int32_t il) const {
    return mem->r_l[il];
}

ggml_tensor * llama_memory_recurrent_context::get_s_l(int32_t il) const {
    return mem->s_l[il];
}

int32_t llama_memory_recurrent_context::s_copy(int i) const {
    const uint32_t cell_idx = mem->n_devices() > 1 ?
        llama_seq_to_stream(get_ubatch().seq_id_unq[i], mem->size, mem->n_devices()) : i + mem->head;
    const int32_t  src0     = mem->cells[cell_idx].src0;

    if (mem->n_rs_seq == 0) { return mem->n_devices() > 1 ? mem->state_row(src0, 0, true) : src0; }

    uint32_t idx = 0;
    if (!mem->cells[cell_idx].seq_id.empty()) {
        const llama_seq_id seq = *mem->cells[cell_idx].seq_id.begin();
        if (seq >= 0 && (size_t) seq < mem->rs_idx.size()) {
            idx = mem->rs_idx[seq];
            // reset rollback idx
            mem->rs_idx[seq] = 0;
        }
    }
    return mem->n_devices() > 1 ? mem->state_row(src0, idx, true) : (int32_t)(idx*mem->size) + src0;
}

uint32_t llama_memory_recurrent_context::s_write(llama_seq_id seq, uint32_t snapshot) const {
    GGML_ASSERT(mem->n_devices() > 1);
    return mem->state_row(llama_seq_to_stream(seq, mem->size, mem->n_devices()), snapshot, true);
}
