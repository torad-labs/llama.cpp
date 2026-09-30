#include "common.cuh"
#include "ggml.h"

void ggml_cuda_op_dsv4_hc_comb(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_dsv4_hc_pre(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_dsv4_hc_post(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_dsv4_hc_weights(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// The front of a hyper-connection cycle, six nodes in two kernels: the weightless RMS_NORM of the flat streams
// (rms_flat), their MUL_MAT by hc_fn (mm), DSV4_HC_WEIGHTS (weights), DSV4_HC_PRE (pre) on its pre weights' view, and
// the RMS_NORM (rms) and MUL (mul) of the sublayer's norm; it writes weights and mul. Supported at up to 16 tokens, the
// flat streams being pre's streams, contiguous, and hc_fn F32, F16 or BF16. weights_outlive: the weights are read past
// the front (a view of them besides pre's is in the graph, or they are an output), so their bytes stay theirs until a
// join; only then may the comb be written beside the stream, after the front's launch.
bool ggml_cuda_dsv4_hc_pre_fused_supported(const ggml_tensor * rms_flat, const ggml_tensor * mm,
        const ggml_tensor * weights, const ggml_tensor * pre, const ggml_tensor * rms, const ggml_tensor * mul);
void ggml_cuda_op_dsv4_hc_pre_fused(ggml_backend_cuda_context & ctx, const ggml_tensor * rms_flat,
        const ggml_tensor * mm, ggml_tensor * weights, const ggml_tensor * pre, const ggml_tensor * rms,
        ggml_tensor * mul, bool weights_outlive);

// The previous sublayer's DSV4_HC_POST (post) and the front after it, seven nodes in two kernels: the front's first
// makes the streams as the post does and writes them to post's output, pre's streams. Supported where the front is and
// the post's output and streams are contiguous along n_embd, not under GGML_CUDA_HC_PRE_GRAM_LEGACY; it writes post,
// weights and mul.
bool ggml_cuda_dsv4_hc_post_pre_fused_supported(const ggml_tensor * post, const ggml_tensor * rms_flat,
        const ggml_tensor * mm, const ggml_tensor * weights, const ggml_tensor * pre, const ggml_tensor * rms,
        const ggml_tensor * mul);
void ggml_cuda_op_dsv4_hc_post_pre_fused(ggml_backend_cuda_context & ctx, ggml_tensor * post,
        const ggml_tensor * rms_flat, const ggml_tensor * mm, ggml_tensor * weights, const ggml_tensor * pre,
        const ggml_tensor * rms, ggml_tensor * mul, bool weights_outlive);

// Whether node is a front's mul, the sublayer's normed mix: the fused front writes its q8_1 copy beside it when the
// evaluation holds one for it (ggml_cuda_mmvq_shared_q8_1::produce) and n_embd is a multiple of MATRIX_ROW_PADDING,
// not under GGML_CUDA_HC_PRE_GRAM_LEGACY. Where the front does not run fused, its readers quantize it themselves.
bool ggml_cuda_dsv4_hc_writes_q8_1(const ggml_tensor * node);

// The fused front's comb weights are made beside the evaluation's stream (GGML_CUDA_HC_COMB_STREAM), off the front's
// chain, where the evaluation allows it (ctx.hc_comb_side) and not under GGML_CUDA_HC_COMB_SIDE_LEGACY: the stream waits
// for them before a node that reads the weights (ggml_cuda_dsv4_hc_comb_reads, and every DSV4_HC_POST) and at the end of
// the evaluation (ggml_cuda_dsv4_hc_comb_join, a no-op with none pending).
#define GGML_CUDA_HC_COMB_STREAM (GGML_CUDA_MAX_STREAMS - 2)
bool ggml_cuda_dsv4_hc_comb_reads(const ggml_backend_cuda_context & ctx, const ggml_tensor * node);
void ggml_cuda_dsv4_hc_comb_join(ggml_backend_cuda_context & ctx);
