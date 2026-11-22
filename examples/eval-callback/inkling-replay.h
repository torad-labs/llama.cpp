#pragma once

#include "ggml-backend.h"
#include "hash/hash.h"
#include "nlohmann/json.hpp"
#include "sampling.h"
#include "../../ggml/src/ggml-attn-mask.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <set>
#include <stdexcept>
#include <string_view>

// Diagnostic only. A fixed batch drives the same token prefix in every arm.
// The callback reads the compact op's input, without requesting API probabilities.
class inkling_replay {
    using json = nlohmann::json;

    const std::filesystem::path out;
    json input;
    std::ofstream scores;
    json rows = json::array();
    json attention = json::array();
    std::string layout;
    std::set<const ggml_tensor *> seen;
    std::vector<uint8_t> scratch;
    std::vector<float> logits;
    int n_seq = 0;
    int vocab = 0;
    bool recording = false;
    bool wrote_scores = false;
    bool compact;

    void hash_tensor(const ggml_tensor * t) {
        if (!t || !seen.insert(t).second) {
            return;
        }
        scratch.resize(ggml_nbytes(t));
        ggml_backend_tensor_get(t, scratch.data(), 0, scratch.size());
        // Use logical shape and bytes, not the buffer address.
        layout += json(t->ne).dump() + ":" + ggml_type_name(t->type) + ":" +
            hash_sha256_hex(scratch.data(), scratch.size()) + "\n";
    }

    bool capture(ggml_tensor * t, bool ask) {
        if (!recording) {
            return !ask;
        }
        const bool is_attn = t->op == GGML_OP_FLASH_ATTN_EXT;
        const bool is_score = std::string_view(t->name) == (compact ? "result_candidates" : "result_output");
        if (ask) {
            return is_attn || is_score;
        }
        if (is_attn) {
            hash_tensor(t->src[3]);
            hash_tensor(t->src[7]);
            hash_tensor(t->src[8]);
            attention.push_back({
                {"queries", t->src[0]->ne[1]}, {"query_heads", t->src[0]->ne[2]},
                {"kv_cells", t->src[1]->ne[1]}, {"kv_heads", t->src[1]->ne[2]},
                {"low_mean", ggml_flash_attn_ext_has_low_mean_mask(t)},
            });
        } else if (is_score) {
            const ggml_tensor * head = compact ? t->src[0] : t;
            if (wrote_scores || !head || head->type != GGML_TYPE_F32 || !ggml_is_contiguous(head) ||
                    head->ne[0] < vocab || head->ne[1] != n_seq || head->ne[2] != 1 || head->ne[3] != 1) {
                throw std::runtime_error("replay did not expose one full head row per slot");
            }
            // Meta reads whole padded rows. Drop the invalid suffix only after the read.
            logits.resize(head->ne[0]);
            for (int i = 0; i < n_seq; ++i) {
                ggml_backend_tensor_get(head, logits.data(), i*head->nb[1], head->ne[0]*sizeof(float));
                scores.write(reinterpret_cast<const char *>(logits.data()), vocab*sizeof(float));
            }
            if (!scores) {
                throw std::runtime_error("cannot write replay scores");
            }
            wrote_scores = true;
        }
        return true;
    }

    static std::string digest_tokens(const llama_tokens & tokens) {
        // Matches the comma-separated token encoding used by the client probe.
        std::string wire;
        for (const auto token : tokens) {
            if (!wire.empty()) {
                wire += ',';
            }
            wire += std::to_string(token);
        }
        return hash_sha256_hex(wire.data(), wire.size());
    }

public:
    inkling_replay(const std::filesystem::path & manifest, const std::filesystem::path & out, bool compact)
        : out(out), compact(compact) {
        std::ifstream spec(manifest);
        spec >> input;
    }

    void configure(common_params & params) {
        n_seq = input.at("private").size();
        if (n_seq <= 0 || params.n_parallel != n_seq) {
            throw std::runtime_error("set --parallel to the number of replay slots");
        }
        // A fixture may test the same Meta readback on logical CPU ranks.
        if (input.value("logical_cpu_ranks", 0) == 2) {
            params.devices = {ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU),
                              ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU), nullptr};
            params.split_mode = LLAMA_SPLIT_MODE_TENSOR;
        }
        params.cb_eval = [](ggml_tensor * t, bool ask, void * opaque) {
            return static_cast<inkling_replay *>(opaque)->capture(t, ask);
        };
        params.cb_eval_user_data = this;
        params.warmup = false;
        params.kv_unified = true;
        params.sampling.backend_sampling = false;
        params.sampling.ignore_eos = true;
        params.sampling.temp = 0.0f;
        params.sampling.top_k = 0;
        params.sampling.top_p = 1.0f;
        params.sampling.min_p = 0.0f;
    }

    void run(llama_context * ctx, const common_params & params) {
        const auto * model = llama_get_model(ctx);
        const auto * v = llama_model_get_vocab(model);
        vocab = llama_vocab_n_tokens(v);
        const auto common = input.at("common").get<llama_tokens>();
        const auto tails = input.at("private").get<std::vector<llama_tokens>>();
        const auto forced = input.at("forced").get<std::vector<llama_tokens>>();
        if (common.empty() || tails.size() != forced.size() ||
                digest_tokens(common) != input.at("common_sha256") ||
                llama_n_seq_max(ctx) < uint32_t(n_seq) || llama_n_ubatch(ctx) < uint32_t(n_seq)) {
            throw std::runtime_error("replay prefixes or context capacity differ from the manifest");
        }
        for (int i = 0; i < n_seq; ++i) {
            if (tails[i].empty() || forced[i].size() != forced.front().size() ||
                    digest_tokens(tails[i]) != input.at("private_sha256").at(i)) {
                throw std::runtime_error("a private prefix or forced length differs from the manifest");
            }
        }
        const uint32_t endian = 1;
        if (*reinterpret_cast<const uint8_t *>(&endian) != 1 || !std::filesystem::create_directory(out)) {
            throw std::runtime_error("replay requires a new output directory and little-endian F32");
        }
        scores.open(out / "logits.f32le", std::ios::binary);
        if (!scores) {
            throw std::runtime_error("cannot open replay scores");
        }

        auto sampling = params.sampling;
        sampling.samplers = {COMMON_SAMPLER_TYPE_TEMPERATURE};
        sampling.temp = 0.0f;
        sampling.dynatemp_range = 0.0f;
        sampling.top_k = 0;
        sampling.n_probs = 0;
        sampling.mirostat = 0;
        common_sampler_ptr sampler(common_sampler_init(model, sampling));
        const int k = common_sampler_n_greedy_candidates(sampler.get());
        if (k <= 0) {
            throw std::runtime_error("replay requires the neutral greedy chain with static exclusions");
        }
        std::set<llama_token> suppress;
        for (const auto & bias : sampling.logit_bias) {
            suppress.insert(bias.token);
        }
        int32_t n_model_suppress = 0;
        const llama_token * model_suppress = llama_vocab_get_suppress_tokens(v, &n_model_suppress);
        for (int i = 0; i < n_model_suppress; ++i) {
            suppress.insert(model_suppress[i]);
        }

        common_batch batch(ctx);
        auto submit = [&] {
            if (batch.size() && llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
                throw std::runtime_error("replay decode failed");
            }
        };

        for (size_t i = 0; i < common.size(); ++i) {
            batch.add(common[i], i, 0, false);
            if (batch.size() == int32_t(llama_n_ubatch(ctx))) {
                submit();
                batch.clear();
            }
        }
        submit();
        batch.clear();
        for (int i = 1; i < n_seq; ++i) {
            llama_memory_seq_cp(llama_get_memory(ctx), 0, i, -1, -1);
        }

        const size_t chunk = std::min<size_t>(128, llama_n_ubatch(ctx)/n_seq);
        for (size_t offset = 0;; offset += chunk) {
            for (int i = 0; i < n_seq; ++i) {
                const size_t stop = std::min(offset + chunk, tails[i].size() - 1);
                for (size_t j = offset; j < stop; ++j) {
                    batch.add(tails[i][j], common.size() + j, i, false);
                }
            }
            if (!batch.size()) {
                break;
            }
            submit();
            batch.clear();
        }

        if (compact) {
            llama_set_output_candidates(ctx, k);
            for (int i = 0; i < n_seq; ++i) {
                if (!llama_set_seq_greedy_candidates(ctx, i)) {
                    throw std::runtime_error("compact replay requires a split Inkling head");
                }
            }
        }

        for (size_t step = 0; step <= forced.front().size(); ++step) {
            json order = json::array();
            for (int i = 0; i < n_seq; ++i) {
                const auto token = step ? forced[i][step-1] : tails[i].back();
                const auto pos = common.size() + tails[i].size() - 1 + step;
                order.push_back({i, token, pos});
                batch.add(token, pos, i, true);
            }

            layout.clear();
            seen.clear();
            attention = json::array();
            wrote_scores = false;
            recording = true;
            submit();
            recording = false;
            if (!wrote_scores || layout.empty()) {
                throw std::runtime_error("replay missed the head or attention identities");
            }

            const auto batch_key = order.dump();
            const auto batch_sha = hash_sha256_hex(batch_key.data(), batch_key.size());
            const auto kv_sha = hash_sha256_hex(layout.data(), layout.size());
            for (int i = 0; i < n_seq; ++i) {
                const auto sample = llama_get_sampling_row_ith(ctx, i);
                if (compact ? sample.logits != nullptr || sample.n_candidates == 0 : sample.logits == nullptr) {
                    throw std::runtime_error("replay took the wrong sampling path");
                }
                rows.push_back({
                    {"client", i}, {"token_position", step + 1},
                    {"selected_id", common_sampler_sample(sampler.get(), v, sample, false, true)},
                    {"condition", {
                        {"prompt_sha256", input.at("common_sha256").get<std::string>() + ":" +
                            input.at("private_sha256").at(i).get<std::string>()},
                        {"forced_tokens", llama_tokens(forced[i].begin(), forced[i].begin() + step)},
                        {"query_position", common.size() + tails[i].size() - 1 + step},
                        {"batch_sha256", batch_sha}, {"kv_mask_sha256", kv_sha}, {"suppressed", suppress},
                    }},
                    {"attention_inputs", attention}, {"compact", compact},
                });
            }
            batch.clear();
        }

        scores.close();
        std::ofstream manifest(out / "rows.json");
        manifest << json({{"build_sha256", input.at("build_sha256")}, {"model_sha256", input.at("model_sha256")},
                          {"n_vocab", vocab}, {"rows", rows}}).dump(2) << '\n';
        manifest.close();
        if (!scores || !manifest) {
            throw std::runtime_error("cannot finish writing replay outputs");
        }
    }
};
