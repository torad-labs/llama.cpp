#include "common.cuh"
#include "ggml.h"

#include <initializer_list>

// Rows that one CUDA block handles.
#define TOPK_MOE_ROWS_PER_BLOCK 8
// Rows of one wide block, which reads every row's logits before it writes (up to 32 warps)
#define TOPK_MOE_WIDE_ROWS 32

struct ggml_cuda_topk_moe_args {
    bool sigmoid{};
    bool sqrt_softplus{};
    bool softmax{};
    bool delayed_softmax{};
    bool prob_bias{};
    bool norm{};
    bool norm_eps{};     // the norm adds an epsilon to the sum (a SCALE node) instead of clamping it (a CLAMP node)
    bool scale{};
    int  n_groups{};     // > 0: pick experts only from the top n_group_used expert groups
    int  n_group_used{};
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
// MTP verify's). Up to TOPK_MOE_WIDE_ROWS rows it launches one wide block when they do lie over the logits (a 23-session
// decode's); GGML_CUDA_TOPK_MOE_WIDE_LEGACY=1 keeps the rows of one ordinary block.
bool ggml_cuda_topk_moe_reads_before_writes(int64_t n_rows);

bool ggml_cuda_should_use_topk_moe(const ggml_tensor * gating_op,
                                   const ggml_tensor * weights,
                                   const ggml_tensor * logits,
                                   const ggml_tensor * ids);
