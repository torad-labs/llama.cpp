#include "llama-kv-cache.h"
#include "llama-model.h"
#include "models/models.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <vector>

static void init_model(llama_model & model, uint32_t n_layer) {
    model.hparams = {};
    model.hparams.n_layer_all = n_layer;
    model.hparams.n_embd_head_k_full = 4;
    model.hparams.n_embd_head_v_full = 4;
    model.hparams.n_head_arr.fill(1);
    model.hparams.n_head_kv_arr.fill(1);
    std::iota(model.hparams.kv_source_impl.begin(), model.hparams.kv_source_impl.end(), 0);
    model.layers.resize(n_layer);
}

int main() {
    for (int bit = 0; bit < 64; ++bit) {
        GGML_ASSERT(ctz64(uint64_t{1} << bit) == bit);
        GGML_ASSERT(ctz64(UINT64_MAX << bit) == bit);
    }

    llama_backend_init();

    {
        std::string errors;
        llama_log_set([](ggml_log_level level, const char * text, void * user_data) {
            if (level == GGML_LOG_LEVEL_ERROR) {
                *static_cast<std::string *>(user_data) += text;
            }
        }, &errors);

        for (llm_arch arch : { LLM_ARCH_DEEPSEEK2, LLM_ARCH_DEEPSEEK32, LLM_ARCH_DEEPSEEK4,
                LLM_ARCH_DEEPSEEK41, LLM_ARCH_GLM_DSA, LLM_ARCH_GLM5_NEXT, LLM_ARCH_DOTS3NOTE,
                LLM_ARCH_MISTRAL_LARGE4, LLM_ARCH_HY_V4, LLM_ARCH_MINIMAX_M3, LLM_ARCH_QWEN4EXP,
                LLM_ARCH_DFLASH }) {
            std::unique_ptr<llama_model> model(llama_model_create(arch, llama_model_default_params()));
            init_model(*model, 1);
            if (arch == LLM_ARCH_DEEPSEEK2) {
                model->hparams.n_embd_head_k_mla_impl = 4;
                model->hparams.n_embd_head_v_mla_impl = 4;
            }

            for (ggml_type type_k : { GGML_TYPE_F16, GGML_TYPE_F8_E4M3 }) {
                for (ggml_type type_v : { GGML_TYPE_F16, GGML_TYPE_F8_E4M3 }) {
                    if (type_k == GGML_TYPE_F16 && type_v == GGML_TYPE_F16) {
                        continue;
                    }
                    auto params = llama_context_default_params();
                    params.type_k = type_k;
                    params.type_v = type_v;
                    errors.clear();
                    llama_context * ctx = llama_init_from_model(model.get(), params);
                    GGML_ASSERT(ctx == nullptr);
                    if (type_k == type_v || (!model->hparams.is_mla() && arch != LLM_ARCH_DEEPSEEK4 && arch != LLM_ARCH_DEEPSEEK41)) {
                        GGML_ASSERT(errors.find("FP8 cache is not supported for architecture") != std::string::npos);
                    } else {
                        GGML_ASSERT(errors.find("model requires matching K and V cache types") != std::string::npos);
                    }
                }
            }
        }

        llama_log_set(nullptr, nullptr);
    }

    {
        std::unique_ptr<llama_model> target(llama_model_create(LLM_ARCH_GEMMA4, llama_model_default_params()));
        std::unique_ptr<llama_model> assistant(llama_model_create(LLM_ARCH_GEMMA4_ASSISTANT, llama_model_default_params()));
        init_model(*target, 3);
        init_model(*assistant, 2);

        ggml_context_ptr ctx(ggml_init({ 4096, nullptr, false }));
        target->layers[2].k_cache_s = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 1);
        target->layers[2].v_cache_s = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 1);
        assistant->layers[0].k_cache_s = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 1);
        assistant->layers[0].v_cache_s = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 1);
        assistant->layers[1].k_cache_s = assistant->layers[0].k_cache_s;
        assistant->layers[1].v_cache_s = assistant->layers[0].v_cache_s;
        *static_cast<float *>(target->layers[2].k_cache_s->data) = 0.5f;
        *static_cast<float *>(target->layers[2].v_cache_s->data) = 0.25f;
        *static_cast<float *>(assistant->layers[0].k_cache_s->data) = 2.0f;
        *static_cast<float *>(assistant->layers[0].v_cache_s->data) = 4.0f;

        for (ggml_type type_k : { GGML_TYPE_F16, GGML_TYPE_F8_E4M3 }) {
            for (ggml_type type_v : { GGML_TYPE_F16, GGML_TYPE_F8_E4M3 }) {
                llama_kv_cache source(*target, false, target->hparams, type_k, type_v,
                        false, false, true, 32, 1, 0, LLAMA_SWA_TYPE_NONE, nullptr, nullptr,
                        [](uint32_t il) { return il == 0 ? 2 : -1; }, nullptr);
                llama_kv_cache shared(*assistant, false, assistant->hparams, type_k, type_v,
                        false, false, true, 32, 1, 0, LLAMA_SWA_TYPE_NONE, &source, nullptr, nullptr,
                        [](int32_t il) { return il == 0 ? 2 : 1; });

                GGML_ASSERT(source.get_k_scale(2) == (type_k == GGML_TYPE_F8_E4M3 ? target->layers[2].k_cache_s : nullptr));
                GGML_ASSERT(source.get_v_scale(2) == (type_v == GGML_TYPE_F8_E4M3 ? target->layers[2].v_cache_s : nullptr));
                GGML_ASSERT(source.get_k_scale(0) == source.get_k_scale(2));
                GGML_ASSERT(source.get_v_scale(0) == source.get_v_scale(2));
                GGML_ASSERT(shared.get_k_storage(0) == source.get_k_storage(2));
                GGML_ASSERT(shared.get_k_scale(0) == source.get_k_scale(2));
                GGML_ASSERT(shared.get_v_scale(0) == source.get_v_scale(2));
                GGML_ASSERT(shared.get_k_scale(1) == nullptr);
                GGML_ASSERT(shared.get_v_scale(1) == nullptr);
            }
        }
    }

    // seq_rm removes [p0, p1) of one sequence, a negative bound is open
    {
        std::unique_ptr<llama_model> model(llama_model_create(LLM_ARCH_LLAMA, llama_model_default_params()));
        init_model(*model, 1);

        llama_kv_cache kv(*model, false, model->hparams, GGML_TYPE_F16, GGML_TYPE_F16,
                false, false, true, 32, 2, 0, LLAMA_SWA_TYPE_NONE, nullptr, nullptr, nullptr, nullptr);

        // returns the first cell that takes the tokens
        auto add_to = [](llama_kv_cache & cache, llama_seq_id seq_id, llama_pos p0, uint32_t n) {
            std::vector<llama_token>    token(n, 1);
            std::vector<llama_pos>      pos(n);
            std::vector<int32_t>        n_seq_id(n, 1);
            std::vector<llama_seq_id *> seq_ids(n, &seq_id);
            std::iota(pos.begin(), pos.end(), p0);

            llama_ubatch ubatch = {};
            ubatch.n_tokens     = n;
            ubatch.n_seq_tokens = n;
            ubatch.n_seqs       = 1;
            ubatch.n_seqs_unq   = 1;
            ubatch.n_pos        = 1;
            ubatch.token        = token.data();
            ubatch.pos          = pos.data();
            ubatch.n_seq_id     = n_seq_id.data();
            ubatch.seq_id       = seq_ids.data();
            ubatch.seq_id_unq   = &seq_id;

            const auto sinfo = cache.find_slot(ubatch, false);
            GGML_ASSERT(!sinfo.empty());
            cache.apply_ubatch(sinfo, ubatch);

            return sinfo.head();
        };

        auto add = [&](llama_seq_id seq_id, llama_pos p0, uint32_t n) { return add_to(kv, seq_id, p0, n); };

        GGML_ASSERT(add(0, 0, 8) == 0);
        GGML_ASSERT(add(1, 0, 4) == 8);

        // the tail of sequence 0, the other sequence keeps its cells
        GGML_ASSERT(kv.seq_rm(0, 5, -1));
        GGML_ASSERT(kv.seq_pos_min(0) == 0 && kv.seq_pos_max(0) == 4);
        GGML_ASSERT(kv.seq_pos_min(1) == 0 && kv.seq_pos_max(1) == 3);

        // the next tokens go to the lowest freed cell
        GGML_ASSERT(add(0, 5, 2) == 5);

        // p1 is not part of the range
        GGML_ASSERT(kv.seq_rm(0, -1, 2));
        GGML_ASSERT(kv.seq_pos_min(0) == 2 && kv.seq_pos_max(0) == 6);
        GGML_ASSERT(kv.seq_rm(0, 6, 6));
        GGML_ASSERT(kv.seq_pos_max(0) == 6);

        // freed cells are taken in cell order, so a higher position can sit in a lower cell
        GGML_ASSERT(add(0, 7, 3) == 0);
        GGML_ASSERT(kv.seq_rm(0, 6, -1));
        GGML_ASSERT(add(0, 6, 1) == 0);

        // a cell shared with another sequence stays for that sequence
        kv.seq_cp(0, 1, 5, -1);
        GGML_ASSERT(kv.seq_rm(0, 5, -1));
        GGML_ASSERT(kv.seq_pos_max(0) == 4 && kv.seq_pos_max(1) == 6);

        // every cell of a repeated position is removed
        add(0, 5, 1);
        add(0, 5, 1);
        GGML_ASSERT(kv.seq_rm(0, 5, 6));
        GGML_ASSERT(kv.seq_pos_max(0) == 4);

        llama_kv_cache swa(*model, false, model->hparams, GGML_TYPE_F16, GGML_TYPE_F16,
                false, false, true, 8, 2, 4, LLAMA_SWA_TYPE_STANDARD, nullptr, nullptr, nullptr, nullptr);
        add_to(swa, 0, 0, 4);
        swa.seq_cp(0, 1, -1, -1);
        add_to(swa, 0, 4, 4);
        // Both owners have moved past cell 0. The slower owner still needs cells 1 to 3.
        GGML_ASSERT(add_to(swa, 0, 8, 1) == 0);
        GGML_ASSERT(add_to(swa, 0, 9, 1) == 4);
        GGML_ASSERT(swa.seq_pos_min(1) == 1 && swa.seq_pos_max(1) == 3);
    }

    // A call cut into several micro-batches leaves every row the positions its window reads, whichever sequences
    // share the windowed pool, and finding the cells of a call changes nothing
    {
        std::unique_ptr<llama_model> model(llama_model_create(LLM_ARCH_LLAMA, llama_model_default_params()));
        init_model(*model, 1);

        struct rows { // one micro-batch
            std::vector<llama_token>    token;
            std::vector<llama_pos>      pos;
            std::vector<int32_t>        n_seq_id;
            std::vector<llama_seq_id>   seq;
            std::vector<llama_seq_id *> seq_ptr;
            std::vector<llama_seq_id>   seq_unq;
        };

        // as the simple split hands it to the cache: every row a sequence set of its own
        const auto as_ubatch = [](rows & r) {
            const uint32_t n = r.pos.size();
            r.token.assign(n, 1);
            r.n_seq_id.assign(n, 1);
            for (uint32_t i = 0; i < n; ++i) {
                r.seq_ptr.push_back(&r.seq[i]);
                if (std::find(r.seq_unq.begin(), r.seq_unq.end(), r.seq[i]) == r.seq_unq.end()) {
                    r.seq_unq.push_back(r.seq[i]);
                }
            }
            llama_ubatch ubatch = {};
            ubatch.n_tokens     = n;
            ubatch.n_seq_tokens = 1;
            ubatch.n_seqs       = n;
            ubatch.n_seqs_unq   = r.seq_unq.size();
            ubatch.n_pos        = 1;
            ubatch.token        = r.token.data();
            ubatch.pos          = r.pos.data();
            ubatch.n_seq_id     = r.n_seq_id.data();
            ubatch.seq_id       = r.seq_ptr.data();
            ubatch.seq_id_unq   = r.seq_unq.data();
            return ubatch;
        };

        // the position and owners of every cell, then the first and last position of every sequence
        const auto snapshot = [](const llama_kv_cache & kv, uint32_t n_seq) {
            const auto & cells = kv.get_cells(0);
            std::vector<int64_t> ret;
            for (uint32_t i = 0; i < cells.size(); ++i) {
                int64_t cell = -1;
                if (!cells.is_empty(i)) {
                    cell = cells.pos_get(i);
                    for (uint32_t s = 0; s < n_seq; ++s) {
                        cell = cell*2 + cells.seq_has(i, s);
                    }
                }
                ret.push_back(cell);
            }
            for (uint32_t s = 0; s < n_seq; ++s) {
                ret.push_back(kv.seq_pos_min(s));
                ret.push_back(kv.seq_pos_max(s));
            }
            return ret;
        };

        size_t n_calls = 0, n_rows = 0, changed = 0, lost = 0;
        std::mt19937 rng(1234);
        for (int round = 0; round < 400; ++round) {
            const uint32_t n_seq = 1 + rng() % 4;
            const uint32_t n_swa = 2 + rng() % 7;
            const uint32_t n_ub  = 1 + rng() % 12;
            const auto     type  = rng() % 4 ? LLAMA_SWA_TYPE_STANDARD : LLAMA_SWA_TYPE_CHUNKED;
            // the least pool a windowed layer is given: a window a sequence and one micro-batch
            llama_kv_cache kv(*model, false, model->hparams, GGML_TYPE_F16, GGML_TYPE_F16, false, false, true,
                    n_swa*n_seq + n_ub, n_seq, n_swa, type, nullptr, nullptr, nullptr, nullptr);

            std::vector<llama_pos> next(n_seq, 0);
            for (int call = 0; call < 10; ++call, ++n_calls) {
                // runs of consecutive positions, a run from one sequence, cut every n_ub rows
                std::vector<rows> parts;
                for (uint32_t n_runs = 1 + rng() % 4; n_runs > 0; --n_runs) {
                    const llama_seq_id s = rng() % n_seq;
                    for (uint32_t n = 1 + rng() % (2*n_ub); n > 0; --n) {
                        if (parts.empty() || parts.back().pos.size() == n_ub) {
                            parts.emplace_back();
                        }
                        parts.back().pos.push_back(next[s]++);
                        parts.back().seq.push_back(s);
                    }
                }
                std::vector<llama_ubatch> ubatches;
                for (auto & r : parts) {
                    ubatches.push_back(as_ubatch(r));
                }

                const auto before = snapshot(kv, n_seq);
                const auto sinfos = kv.prepare(ubatches);
                GGML_ASSERT(sinfos.size() == ubatches.size());
                changed += snapshot(kv, n_seq) != before;

                for (size_t k = 0; k < ubatches.size(); ++k) {
                    kv.apply_ubatch(sinfos[k], ubatches[k]);
                    const auto & cells = kv.get_cells(0);
                    for (size_t i = 0; i < parts[k].pos.size(); ++i, ++n_rows) {
                        const auto & held = cells.seq_pos_get(parts[k].seq[i]);
                        for (llama_pos p0 = 0; p0 <= parts[k].pos[i]; ++p0) {
                            if (llama_hparams::is_masked_swa(n_swa, type, p0, parts[k].pos[i])) {
                                continue;
                            }
                            const auto it = held.lower_bound({p0, 0});
                            lost += it == held.end() || it->first != p0;
                        }
                    }
                }
            }
        }
        fprintf(stderr, "windowed pool: %zu calls, %zu changed by finding their cells; %zu rows, %zu positions of a window lost\n",
                n_calls, changed, n_rows, lost);
        GGML_ASSERT(changed == 0);
        GGML_ASSERT(lost == 0);
    }

    {
        std::unique_ptr<llama_model> model(llama_model_create(LLM_ARCH_INKLING2, llama_model_default_params()));
        init_model(*model, 1);
        for (auto type : {GGML_TYPE_F16, GGML_TYPE_F32}) {
            for (uint32_t window : {0u, 1024u}) {
                llama_kv_cache kv(*model, false, model->hparams, type, type, false, false, true, 43008, 1,
                        window, window ? LLAMA_SWA_TYPE_STANDARD : LLAMA_SWA_TYPE_NONE, nullptr, nullptr, nullptr, nullptr);
                llama_seq_id seq = 0;
                llama_seq_id * seq_ptr = &seq;
                llama_pos pos = 0;
                int32_t n_id = 1;
                llama_ubatch ub = {};
                ub.n_tokens = ub.n_seq_tokens = ub.n_seqs = ub.n_seqs_unq = ub.n_pos = 1;
                ub.pos = &pos;
                ub.n_seq_id = &n_id;
                ub.seq_id = &seq_ptr;
                ub.seq_id_unq = &seq;
                llama_kv_cache::slot_info slot{};
                slot.strm = {0};
                const bool bucket = !window && type == GGML_TYPE_F16 && ggml_env_switch("LLAMA_INKLING2_BUCKET_PREFILL_KV");
                const uint32_t cells = (uint32_t) ggml_env_int("LLAMA_INKLING2_BUCKET_CELLS");
                for (uint32_t i : {1024u, 32768u, 33792u, 42752u}) {
                    slot.idxs = {{i}};
                    kv.apply_ubatch(slot, ub);
                    auto prompt = ub;
                    prompt.n_tokens = prompt.n_seq_tokens = 1024;
                    const uint32_t narrow = i + 256;
                    GGML_ASSERT(kv.get_n_kv(slot, &ub) == narrow);
                    const uint32_t expected = bucket && i >= 32768 ? std::min(43008u, uint32_t(GGML_PAD(narrow, cells))) : narrow;
                    GGML_ASSERT(kv.get_n_kv(slot, &prompt) == expected);
                    GGML_ASSERT(kv.get_cells(seq).seq_n_cells(seq) == uint32_t(pos + 1));
                    ++pos;
                }
            }
        }
    }

    // Inkling 2 sessions that share a pool each add a cell a step, and the padded length shapes every decode graph: it
    // holds for a power of two while the pool is small and for 256 cells a session after. LLAMA_KV_PAD_SHARED_LEGACY=1: 256.
    // The pool of its DSpark draft pads the same
    for (const bool draft : { false, true }) {
        std::unique_ptr<llama_model> model(llama_model_create(draft ? LLM_ARCH_DFLASH : LLM_ARCH_INKLING2, llama_model_default_params()));
        init_model(*model, 1);
        if (draft) {
            static_cast<llama_model_dflash &>(*model).dspark_markov_rank = 1;
            GGML_ASSERT(llama_model_is_inkling_draft(*model));
        }
        const uint32_t n_seq = 23, size = 65536, n_steps = 2048;
        llama_kv_cache kv(*model, false, model->hparams, GGML_TYPE_F16, GGML_TYPE_F16, false, false, true, size, n_seq,
                0, LLAMA_SWA_TYPE_NONE, nullptr, nullptr, nullptr, nullptr);
        std::vector<llama_seq_id> seq(n_seq);
        std::iota(seq.begin(), seq.end(), 0);
        std::vector<llama_seq_id *> seq_ptr(n_seq);
        std::vector<llama_pos> pos(n_seq);
        std::vector<int32_t> n_id(n_seq, 1);
        for (uint32_t i = 0; i < n_seq; ++i) {
            seq_ptr[i] = &seq[i];
        }
        llama_ubatch ub = {};
        ub.n_tokens = ub.n_seqs = ub.n_seqs_unq = n_seq;
        ub.n_seq_tokens = ub.n_pos = 1;
        ub.pos = pos.data();
        ub.n_seq_id = n_id.data();
        ub.seq_id = seq_ptr.data();
        ub.seq_id_unq = seq.data();
        llama_kv_cache::slot_info slot{};
        slot.strm = {0};
        slot.idxs.resize(1);
        const char * legacy = getenv("LLAMA_KV_PAD_SHARED_LEGACY");
        const bool narrow = legacy && atoi(legacy) == 1;
        uint32_t lengths = 0, last = 0;
        for (uint32_t step = 0; step < n_steps; ++step) {
            slot.idxs[0].clear();
            for (uint32_t i = 0; i < n_seq; ++i) {
                slot.idxs[0].push_back(step*n_seq + i);
                pos[i] = step;
            }
            kv.apply_ubatch(slot, ub);
            const uint32_t cells = (step + 1)*n_seq, n_kv = kv.get_n_kv(slot, &ub);
            GGML_ASSERT(n_kv >= cells && n_kv <= size && n_kv % 256 == 0);
            // the empty cells the padding adds stay under 8,192, the next power of two over 256 a session
            GGML_ASSERT(n_kv - cells < (narrow ? 256u : 8192u));
            lengths += n_kv != last;
            last = n_kv;
        }
        fprintf(stderr, "shared Inkling 2%s pool: %u padded lengths in %u steps of %u sessions\n", draft ? " draft" : "", lengths, n_steps, n_seq);
        // 47,104 cells: 256 to 8,192 by powers of two, then every 8,192; or every 256
        GGML_ASSERT(lengths == (narrow ? 184u : 11u));
    }

    llama_backend_free();
    return 0;
}
