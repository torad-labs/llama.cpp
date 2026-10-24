// A layer's current and rollback states must fit in their own addressable
// allocation, with unchanged canonical rows and no cross-layer or slot aliasing.
#include "../src/llama-model.h"
#include "../src/llama-memory-recurrent.h"
#include "ggml-backend.h"
#include "../ggml/src/ggml-recurrent-arena.h"
#include <cstdio>
#include <cstdlib>
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

int main() {
    if (!test_window_properties()) { return 5; }
#ifdef _WIN32
    _putenv_s("GGML_CUDA_GDN_STATE_ARENA", "1");
#else
    setenv("GGML_CUDA_GDN_STATE_ARENA", "1", 1);
#endif
    std::unique_ptr<llama_model> model(llama_model_create(LLM_ARCH_GLM5NEXT, llama_model_default_params()));
    auto & h = model->hparams;
    h.n_layer_all = 5;
    h.n_embd = 256;
    h.n_head_arr.fill(2);
    h.n_head_kv_arr.fill(2);
    h.n_embd_head_kda = 128;
    h.ssm_d_conv = 4;
    std::mt19937 rng(530);
    for (const uint32_t slots : {1u, 3u}) {
        for (const uint32_t snapshots : {0u, 1u, 8u}) {
            // Repeated lifetimes exercise reuse of addresses after teardown.
            for (int life = 0; life < 3; ++life) {
                llama_memory_recurrent mem(*model, GGML_TYPE_F32, GGML_TYPE_F32, false,
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
