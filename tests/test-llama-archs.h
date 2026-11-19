// The helpers of test-llama-archs that its architecture tests in other files share (test-llama-archs-inkling.cpp)
#pragma once

#include "llama-cpp.h"
#include "gguf.h"

#include <cstdio>
#include <string>
#include <vector>

// normalized mean squared error = mse(a, b) / mse(a, 0)
double nmse(const std::vector<float> & a, const std::vector<float> & b);

// n_tokens tokens below n_vocab drawn from seed
std::vector<llama_token> get_tokens(uint32_t n_tokens, uint32_t n_vocab, size_t seed);

// the fixture model of gguf_ctx (or file) with its weights drawn from seed, on devs split as split_mode says
llama_model_ptr get_model(
        gguf_context * gguf_ctx, FILE * file, size_t seed, float stdev, const std::vector<ggml_backend_dev_t> & devs,
        llama_split_mode split_mode = LLAMA_SPLIT_MODE_LAYER,
        const llama_model_tensor_buft_override * tensor_buft_overrides = nullptr,
        const std::vector<float> & tensor_split = {}, bool mtp = false);

void set_env(const char * name, const char * value);
void batch_clear(llama_batch & batch);
void batch_add(llama_batch & batch, llama_token id, llama_pos pos, const std::vector<llama_seq_id> & seq_ids, bool logits);

// --inkling-split <case>: the case of Inkling 2 run on the ranks of devs under -sm tensor and on the first alone
int test_inkling_split(const std::string & name, size_t seed, const std::vector<ggml_backend_dev_t> & devs);

// --inkling-path <feature>/<case>: an optimized path of Inkling 2 against the path it replaced, on dev
int test_inkling_path(const std::string & path, size_t seed, ggml_backend_dev_t dev);

// --inkling-window <device>: sessions of Inkling 2 that share one cache against each session alone, on dev
int test_inkling_window(size_t seed, ggml_backend_dev_t dev);
