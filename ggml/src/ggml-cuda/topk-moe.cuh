#include "common.cuh"
#include "ggml.h"

#include <initializer_list>

struct ggml_cuda_topk_moe_args {
    bool sigmoid{};
    bool sqrt_softplus{};
    bool softmax{};
    bool delayed_softmax{};
    bool prob_bias{};
    bool norm{};
    bool scale{};
};

void ggml_cuda_op_topk_moe(ggml_backend_cuda_context &     ctx,
                           const ggml_tensor *             logits,
                           ggml_tensor *                   weights,
                           ggml_tensor *                   ids,
                           const ggml_tensor *             clamp,
                           const ggml_tensor *             scale,
                           const ggml_tensor *             bias,
                           const ggml_cuda_topk_moe_args & args);

// Whether ggml_cuda_op_topk_moe over n_rows reads every row's logits before it writes a weight or an id: its rows fit one
// block, whose warps meet at a barrier past their reads, so its weights and ids may lie over its logits (a decode's, an
// MTP verify's).
bool ggml_cuda_topk_moe_reads_before_writes(int64_t n_rows);

bool ggml_cuda_should_use_topk_moe(const ggml_tensor * gating_op,
                                   const ggml_tensor * weights,
                                   const ggml_tensor * logits,
                                   const ggml_tensor * ids);
