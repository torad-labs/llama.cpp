// A layer's current and rollback states must fit in their own addressable
// allocation, with unchanged canonical rows and no cross-layer or slot aliasing.
#include "../src/llama-model.h"
#include "../src/llama-batch.h"
#include "../src/llama-memory-recurrent.h"
#include "../src/llama-io.h"
#include "ggml-backend.h"
#include "../ggml/src/ggml-recurrent-arena.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <memory>
#include <random>
#include <set>
#include <vector>

static bool test_window_properties() {
    std::mt19937 rng(530);
    for (int step = 0; step < 1000; ++step) {
        const uint32_t slots = 1 + rng()%16, snapshots = rng()%17, layers = 1 + rng()%64;
        ggml_tensor root {};
        root.type = GGML_TYPE_F32;
        root.ne[0] = 128 * (1 + rng()%128);
        root.ne[1] = slots * (1 + snapshots);
        root.ne[2] = root.ne[3] = 1;
        root.nb[0] = sizeof(float);
        root.nb[1] = root.ne[0] * root.nb[0];
        root.nb[2] = root.ne[1] * root.nb[1];
        root.nb[3] = root.nb[2];
        std::vector<float> storage(root.nb[2]/sizeof(float));
        root.data = storage.data();
        ggml_recurrent_arena_mark(&root, slots, layers);
        const size_t current = slots*root.nb[1];
        for (const size_t limit : {size_t(0), current - 1, current, current + root.nb[1]}) {
            const auto w = ggml_recurrent_arena_window_of(&root, limit);
            if (limit < current) {
                if (w.base || w.bytes || w.active_bytes) { return false; }
            } else if (w.base != root.data || w.bytes != current || w.active_bytes != current*layers) {
                fprintf(stderr, "FAIL: current state must have one bounded window with all-layer budget\n");
                return false;
            }
        }
        for (uint32_t row = 0; row < slots*(1 + snapshots); ++row) {
            ggml_tensor view = root;
            view.op = GGML_OP_VIEW;
            view.view_src = &root;
            view.ne[1] = 1;
            view.data = (char *) root.data + row*root.nb[1];
            const auto w = ggml_recurrent_arena_window_of(&view, current);
            if (row < slots ? (w.base != root.data || w.bytes != current) : (w.bytes != 0)) { return false; }
        }
        root.op_params[0] = 0;
        if (ggml_recurrent_arena_window_of(&root, current).bytes) { return false; }
    }
    return true;
}

// A placement that leaves a state to move owes the move until it is made (llama_memory_recurrent::settle). One that
// is never made drops the sequences it was for and every sequence whose cell names a row its state is not in, and
// they stay dropped until each is removed; a sequence past the placement's cells is not touched, and a copy of a
// dropped sequence is dropped. A trial placement (prepare) owes nothing and leaves nothing behind, and a restore's
// placement moves no one.
static bool test_owed_moves(const llama_model & model) {
    llama_memory_recurrent mem(model, false, GGML_TYPE_F32, GGML_TYPE_F32, false, 4, 4, 1, {});
    mem.set_relocate(true);
    if (!mem.relocate) {
        fprintf(stderr, "FAIL: f32 states on one device do not relocate\n");
        return false;
    }

    llama_batch_allocr balloc(model.hparams.n_pos_per_embd());
    llama_seq_id ids[4]  = {0, 1, 2, 3};
    llama_pos    next[4] = {0, 0, 0, 0};
    // one more token of one sequence, as a decode's ubatch names it
    const auto one = [&](llama_seq_id seq) {
        llama_ubatch ubatch = balloc.ubatch_reserve(1, 1);
        ubatch.pos[0]      = next[seq]++;
        ubatch.n_seq_id[0] = 1;
        ubatch.seq_id[0]   = &ids[seq];
        return ubatch;
    };
    const auto fail = [](const char * what) {
        fprintf(stderr, "FAIL: owed moves: %s\n", what);
        return false;
    };

    // sequence 0 takes the first cell; sequence 1 is then gathered there and 0's state is left to move
    if (!mem.find_slot(one(0)) || mem.relocate_pending) { return fail("a first sequence owes a move"); }
    if (!mem.find_slot(one(1)) || !mem.relocate_pending) { return fail("a displaced state is not owed a move"); }

    // the move is never made: both cells of that placement name a row their state is not in
    if (mem.seq_lost(2) || !mem.seq_lost(0) || !mem.seq_lost(1)) { return fail("the sequences of a placement that was never completed are not dropped"); }
    if (mem.relocate_pending || mem.seq_pos_max(0) != -1 || mem.seq_pos_max(1) != -1 || mem.used != 0) { return fail("a dropped sequence keeps a cell"); }

    // they stay dropped through later placements, until each is removed
    if (!mem.find_slot(one(2)) || mem.relocate_pending || mem.seq_lost(2)) { return fail("a sequence placed after the drop"); }
    if (!mem.seq_lost(0) || !mem.seq_lost(1)) { return fail("a dropped sequence came back without being removed"); }
    mem.seq_rm(0, -1, -1);
    if (mem.seq_lost(0) || !mem.seq_lost(1)) { return fail("removing one dropped sequence"); }
    mem.seq_rm(1, -1, -1);
    if (mem.seq_lost(1)) { return fail("removing the other dropped sequence"); }

    // trial placements that would displace sequence 2 twice: nothing is owed after them and nothing is left behind
    const auto cells = mem.cells;
    const auto n = mem.n;
    const auto rs_z = mem.rs_z;
    const auto head = mem.head;
    const auto used = mem.used;
    if (!mem.prepare({one(3), one(0)})) { return fail("a trial placement"); }
    bool same = mem.n == n && mem.rs_z == rs_z && mem.head == head && mem.used == used && !mem.relocate_pending;
    for (size_t i = 0; same && i < cells.size(); ++i) {
        same = mem.cells[i].pos == cells[i].pos && mem.cells[i].src == cells[i].src && mem.cells[i].tail == cells[i].tail &&
               mem.cells[i].seq_id == cells[i].seq_id;
    }
    if (!same) { return fail("a trial placement left something behind"); }
    for (llama_seq_id seq = 0; seq < 4; ++seq) {
        if (mem.seq_lost(seq)) { return fail("a trial placement dropped a sequence"); }
    }

    // a restore's placement: sequence 3 takes a free cell and sequence 2 keeps its own
    const int32_t cell_2 = mem.cells[2].tail;
    if (!mem.find_slot(one(3), /*allow_relocate =*/ false) || mem.relocate_pending) { return fail("a restore's placement owes a move"); }
    if (mem.cells[2].tail != cell_2 || mem.cells[3].tail < 0 || mem.cells[3].tail == cell_2) { return fail("a restore's placement moved another sequence"); }
    for (llama_seq_id seq = 0; seq < 4; ++seq) {
        if (mem.seq_lost(seq)) { return fail("a restore's placement dropped a sequence"); }
    }

    // A placement for two of four sequences that is never completed. 0 keeps its row, at a position no decode took it
    // to; 2 takes the second row; 1 is displaced to the row 2 had; 3 lies past the placement's cells
    llama_memory_recurrent four(model, false, GGML_TYPE_F32, GGML_TYPE_F32, false, 4, 4, 1, {});
    four.set_relocate(true);
    llama_pos at[4] = {0, 0, 0, 0};
    const auto set = [&](std::initializer_list<llama_seq_id> seqs) {
        llama_ubatch ubatch = balloc.ubatch_reserve(1, (uint32_t) seqs.size());
        uint32_t s = 0;
        for (const llama_seq_id seq : seqs) {
            ubatch.pos[s]      = at[seq]++;
            ubatch.n_seq_id[s] = 1;
            ubatch.seq_id[s]   = &ids[seq];
            ++s;
        }
        return ubatch;
    };
    for (llama_seq_id seq = 0; seq < 4; ++seq) {
        if (!four.find_slot(set({seq}), /*allow_relocate =*/ false) || four.relocate_pending || four.cells[seq].tail != seq) {
            return fail("four sequences, each placed in a row of its own");
        }
    }
    if (!four.find_slot(set({0, 2})) || !four.relocate_pending) { return fail("a set that displaces a sequence owes no move"); }
    if (four.cells[0].tail != 0 || four.cells[2].tail != 1 || four.cells[1].tail != 2) { return fail("the set was not gathered at the first rows"); }
    if (!four.seq_lost(-1) || !four.seq_lost(0) || !four.seq_lost(2)) { return fail("a sequence of an abandoned set is not dropped, the one that kept its row included"); }
    if (!four.seq_lost(1)) { return fail("the sequence an abandoned set displaced is not dropped"); }
    if (four.seq_lost(3) || four.cells[3].tail != 3 || four.seq_pos_max(3) != 0 || four.used != 1) { return fail("a sequence past the placement's cells was touched"); }

    // a copy of a dropped sequence is dropped, into a sequence that was removed too; each is cleared by its removal
    four.seq_rm(1, -1, -1);
    if (four.seq_lost(1)) { return fail("removing a dropped sequence"); }
    four.seq_cp(0, 1, -1, -1);
    if (!four.seq_lost(1) || four.cells[1].tail != -1) { return fail("a copy of a dropped sequence is not dropped"); }
    for (llama_seq_id seq = 0; seq < 3; ++seq) {
        four.seq_rm(seq, -1, -1);
    }
    if (four.seq_lost(-1) || four.seq_lost(3) || four.cells[3].tail != 3) { return fail("the dropped sequences removed"); }
    return true;
}

struct state_buffer : llama_io_write_i, llama_io_read_i {
    std::vector<uint8_t> bytes;
    size_t offset = 0;
    void write(const void * src, size_t size) override {
        const auto * begin = static_cast<const uint8_t *>(src);
        bytes.insert(bytes.end(), begin, begin + size);
        offset += size;
    }
    void write_tensor(ggml_tensor * tensor, size_t at, size_t size) override {
        const size_t start = bytes.size();
        bytes.resize(start + size);
        ggml_backend_tensor_get(tensor, bytes.data() + start, at, size);
        offset += size;
    }
    void read(void * dst, size_t size) override {
        if (offset > bytes.size() || size > bytes.size() - offset) { throw std::runtime_error("short test state"); }
        std::memcpy(dst, bytes.data() + offset, size);
        offset += size;
    }
    void read_tensor(ggml_tensor * tensor, size_t at, size_t size) override {
        if (offset > bytes.size() || size > bytes.size() - offset) { throw std::runtime_error("short test tensor"); }
        ggml_backend_tensor_set(tensor, bytes.data() + offset, at, size);
        offset += size;
    }
    size_t n_bytes() override { return offset; }
};

struct fixed_rows_env {
    const bool present = getenv("LLAMA_RS_FIXED_ROWS") != nullptr;
    const std::string previous = present ? getenv("LLAMA_RS_FIXED_ROWS") : "";
    static void set(const char * value) {
#ifdef _WIN32
        _putenv_s("LLAMA_RS_FIXED_ROWS", value ? value : "");
#else
        if (value) { setenv("LLAMA_RS_FIXED_ROWS", value, 1); }
        else { unsetenv("LLAMA_RS_FIXED_ROWS"); }
#endif
    }
    ~fixed_rows_env() { set(present ? previous.c_str() : nullptr); }
};

static bool test_fixed_rows(const llama_model & model) {
    fixed_rows_env env;
    env.set("0");
    llama_memory_recurrent legacy(model, false, GGML_TYPE_F32, GGML_TYPE_F32, false, 8, 4, 2, {});
    env.set("1");
    llama_memory_recurrent fixed(model, false, GGML_TYPE_F32, GGML_TYPE_F32, false, 8, 4, 2, {});
    const auto fail = [](const char * what) {
        fprintf(stderr, "FAIL: fixed rows: %s\n", what);
        return false;
    };
    fixed.set_relocate(true);
    if (!fixed.is_fixed_rows() || fixed.size != 4 || fixed.relocate || fixed.relocate_disjoint) {
        return fail("one permanent row per sequence, without relocation");
    }
    llama_batch_allocr balloc(model.hparams.n_pos_per_embd());
    const auto batch = [&](std::vector<llama_seq_id> owners, llama_pos pos, uint32_t tokens) {
        auto b = balloc.ubatch_reserve(tokens, 1);
        b.data->seq_id_data = owners;
        b.data->seq_id_unq = owners;
        std::sort(b.data->seq_id_unq.begin(), b.data->seq_id_unq.end());
        b.seq_id_unq = b.data->seq_id_unq.data();
        b.n_seqs_unq = owners.size();
        for (uint32_t t = 0; t < tokens; ++t) {
            b.pos[t] = pos + t;
            b.n_seq_id[t] = owners.size();
            b.seq_id[t] = b.data->seq_id_data.data();
        }
        return b;
    };
    const auto dump = [](llama_memory_recurrent & mem, llama_seq_id seq) {
        state_buffer out;
        mem.state_write(out, seq);
        return out.bytes;
    };
    const auto fill = [](llama_memory_recurrent & mem, llama_seq_id seq) {
        const uint32_t cell = mem.cells[seq].tail;
        for (const auto & layers : {mem.r_l, mem.s_l}) {
            for (auto * tensor : layers) {
                if (!tensor) { continue; }
                for (uint32_t p = 0; p <= mem.n_rs_seq; ++p) {
                    std::vector<float> values(tensor->ne[0], 10.0f + p);
                    ggml_backend_tensor_set(tensor, values.data(), mem.state_row(cell, p)*tensor->nb[1], tensor->nb[1]);
                }
            }
        }
    };
    llama_memory_recurrent_context initial(&fixed, {batch({3}, 0, 4)});
    if (!initial.apply()) { return fail("initial placement"); }
    fill(fixed, 3);
    fixed.seq_cp(3, 1, -1, -1);
    const auto sibling = dump(fixed, 1);
    if (fixed.cells[3].tail != 3 || fixed.cells[1].tail != 1 ||
            fixed.cells[3].seq_id.size() != 1 || fixed.cells[1].seq_id.size() != 1 || sibling != dump(fixed, 3)) {
        return fail("a copy must own independent bytes and metadata");
    }
    if (!fixed.seq_rm(3, 3, -1) || fixed.rs_idx[3] != 1 || dump(fixed, 1) != sibling) {
        return fail("source rollback changed its copied sibling");
    }
    fixed.seq_cp(3, 2, -1, -1);
    const auto rolled = dump(fixed, 2);
    if (rolled != dump(fixed, 3) || fixed.rs_idx[2] != 0 || fixed.rs_reach[2] != 0) {
        return fail("copy of a pending rollback did not take its selected plane");
    }
    fixed.seq_rm(3, -1, -1);
    llama_memory_recurrent_context reused(&fixed, {batch({3}, 0, 1)});
    if (!reused.apply() || fixed.cells[3].tail != 3 || dump(fixed, 1) != sibling || dump(fixed, 2) != rolled) {
        return fail("reusing the source changed a copied sibling");
    }
    for (const auto & layers : {fixed.r_l, fixed.s_l}) {
        for (auto * tensor : layers) {
            if (!tensor) { continue; }
            std::vector<float> values(tensor->ne[0]);
            ggml_backend_tensor_get(tensor, values.data(), 3*tensor->nb[1], tensor->nb[1]);
            if (std::any_of(values.begin(), values.end(), [](float value) { return value != 0; })) {
                return fail("a reused sequence did not start with a zero state");
            }
        }
    }
    const auto cells = fixed.cells;
    const auto indices = fixed.rs_idx;
    if (!fixed.prepare({batch({0}, 0, 1), batch({2}, 3, 1)}) || fixed.rs_idx != indices) {
        return fail("trial placement changed rollback selection");
    }
    for (size_t i = 0; i < cells.size(); ++i) {
        const auto & a = cells[i];
        const auto & b = fixed.cells[i];
        if (a.pos != b.pos || a.src != b.src || a.src0 != b.src0 || a.tail != b.tail || a.seq_id != b.seq_id) {
            return fail("trial placement changed ownership");
        }
    }
    llama_memory_recurrent_context shared(&fixed, {batch({2, 0}, 3, 1)});
    if (!shared.apply() || shared.s_copy(0) != 2 || shared.s_write(0, 0) != 0 || shared.s_write(2, 1) != 6 ||
            !shared.rows_disjoint() || fixed.cells[0].tail != 0 || fixed.cells[2].tail != 2) {
        return fail("shared tokens must read their first owner and write each owner's row");
    }

    llama_memory_recurrent_context old(&legacy, {batch({2}, 0, 4)});
    if (!old.apply()) { return fail("legacy source placement"); }
    fill(legacy, 2);
    legacy.seq_cp(2, 0, -1, -1);
    const auto expected = dump(legacy, 2);
    state_buffer wire;
    legacy.state_write(wire);
    wire.offset = 0;
    fixed.state_read(wire);
    if (fixed.used != 2 || fixed.cells[0].tail != 0 || fixed.cells[2].tail != 2 ||
            dump(fixed, 0) != expected || dump(fixed, 2) != expected || wire.offset != wire.bytes.size()) {
        return fail("a legacy shared state must restore into independent fixed rows");
    }
    state_buffer reverse;
    fixed.state_write(reverse);
    reverse.offset = 0;
    legacy.state_read(reverse);
    if (dump(legacy, 0) != expected || dump(legacy, 2) != expected) {
        return fail("fixed-row state must remain readable by the legacy memory");
    }
    return true;
}

int main() {
    if (!test_window_properties()) { return 5; }
#ifdef _WIN32
    _putenv_s("GGML_CUDA_GDN_STATE_ARENA", "1");
#else
    setenv("GGML_CUDA_GDN_STATE_ARENA", "1", 1);
#endif
    // the cases test the contiguous layout unless they set the switch themselves (test_fixed_rows), whatever the
    // caller's environment holds: a suite run with LLAMA_RS_FIXED_ROWS=1 exported must not fail test_owed_moves
    fixed_rows_env::set(nullptr);
    std::unique_ptr<llama_model> model(llama_model_create(LLM_ARCH_GLM5NEXT, llama_model_default_params()));
    auto & h = model->hparams;
    h.n_layer_all = 5;
    h.n_embd = 256;
    h.n_head_arr.fill(2);
    h.n_head_kv_arr.fill(2);
    h.n_embd_head_kda = 128;
    h.ssm_d_conv = 4;
    if (!test_owed_moves(*model)) { return 6; }
    if (!test_fixed_rows(*model)) { return 7; }
    std::mt19937 rng(530);
    for (const uint32_t slots : {1u, 3u}) {
        for (const uint32_t snapshots : {0u, 1u, 8u}) {
            // Repeated lifetimes exercise reuse of addresses after teardown.
            for (int life = 0; life < 3; ++life) {
                llama_memory_recurrent mem(*model, false, GGML_TYPE_F32, GGML_TYPE_F32, false,
                                           slots, slots, snapshots, {});
                std::set<ggml_backend_buffer_t> buffers;
                std::vector<std::vector<float>> expected;
                for (auto * s : mem.s_l) {
                    if (!ggml_is_contiguous(s) || s->ne[1] != slots*(1 + snapshots) ||
                        !buffers.insert(s->buffer).second ||
                        ggml_backend_buffer_get_size(s->buffer) > ggml_nbytes(s) + 1024) {
                        fprintf(stderr, "FAIL: state layer does not have its own bounded arena\n");
                        return 1;
                    }
                    expected.emplace_back(ggml_nelements(s), 0.0f);
                    std::vector<float> initial(ggml_nelements(s), 1.0f);
                    ggml_backend_tensor_get(s, initial.data(), 0, ggml_nbytes(s));
                    if (initial != expected.back()) { return 2; }
                }
                for (int step = 0; step < 120; ++step) {
                    const size_t layer = rng() % mem.s_l.size();
                    auto * s = mem.s_l[layer];
                    const size_t row = rng() % s->ne[1];
                    std::vector<float> values(s->ne[0]);
                    for (auto & v : values) { v = (int(rng()%2049) - 1024) * (1.0f/128); }
                    ggml_backend_tensor_set(s, values.data(), row*s->nb[1], s->nb[1]);
                    std::copy(values.begin(), values.end(), expected[layer].begin() + row*s->ne[0]);
                    // Read every layer, not just the row just written: another
                    // sequence's rollback prefix must remain untouched.
                    for (size_t l = 0; l < mem.s_l.size(); ++l) {
                        std::vector<float> actual(expected[l].size());
                        ggml_backend_tensor_get(mem.s_l[l], actual.data(), 0, ggml_nbytes(mem.s_l[l]));
                        if (actual != expected[l]) { return 3; }
                    }
                }
                mem.clear(true);
                for (auto * s : mem.s_l) {
                    std::vector<float> actual(ggml_nelements(s));
                    ggml_backend_tensor_get(s, actual.data(), 0, ggml_nbytes(s));
                    for (float v : actual) { if (v != 0) { return 4; } }
                }
            }
        }
    }
    puts("recurrent arena rows, snapshots, clear and lifetime reuse passed");
    return 0;
}
