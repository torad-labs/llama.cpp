#include "common.cuh"
#include "ggml.h"

void ggml_cuda_op_dsv4_hc_comb(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_dsv4_hc_pre(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_dsv4_hc_post(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_dsv4_hc_weights(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// A DSV4_HC_POST (post) and the next front's weightless RMS_NORM of its flat streams (rms_flat) in one kernel, a block a
// token: it writes post's output and the norm's BF16 copy y, the values ggml_cuda_op_rms_norm_bf16 would write from it,
// for a MUL_MAT that runs on cuBLAS in BF16. Supported where post's x, residual streams and output are F32 and contiguous
// along n_embd, n_embd a multiple of 1024 up to 8192, and rms_flat reads post's output whole as one row a token; not under
// GGML_CUDA_HC_POST_ROWS_LEGACY.
bool ggml_cuda_dsv4_hc_post_norm_supported(const ggml_tensor * post, const ggml_tensor * rms_flat);
void ggml_cuda_op_dsv4_hc_post_norm_bf16(ggml_backend_cuda_context & ctx, ggml_tensor * post,
        const ggml_tensor * rms_flat, nv_bfloat16 * y);

// The front of a hyper-connection cycle, six nodes in two kernels: the weightless RMS_NORM of the flat streams
// (rms_flat), their MUL_MAT by hc_fn (mm), DSV4_HC_WEIGHTS (weights), DSV4_HC_PRE (pre) on its pre weights' view, and
// the RMS_NORM (rms) and MUL (mul) of the sublayer's norm; it writes weights and mul. Supported at up to 16 tokens, the
// flat streams being pre's streams, contiguous, and hc_fn F32, F16 or BF16.
bool ggml_cuda_dsv4_hc_pre_fused_supported(const ggml_tensor * rms_flat, const ggml_tensor * mm,
        const ggml_tensor * weights, const ggml_tensor * pre, const ggml_tensor * rms, const ggml_tensor * mul);
void ggml_cuda_op_dsv4_hc_pre_fused(ggml_backend_cuda_context & ctx, const ggml_tensor * rms_flat,
        const ggml_tensor * mm, ggml_tensor * weights, const ggml_tensor * pre, const ggml_tensor * rms,
        ggml_tensor * mul);

// The previous sublayer's DSV4_HC_POST (post) and the front after it, seven nodes in two kernels: the front's first
// makes the streams as the post does and writes them to post's output, pre's streams. Supported where the front is and
// the post's output and streams are contiguous along n_embd, not under GGML_CUDA_HC_PRE_GRAM_LEGACY; it writes post,
// weights and mul.
bool ggml_cuda_dsv4_hc_post_pre_fused_supported(const ggml_tensor * post, const ggml_tensor * rms_flat,
        const ggml_tensor * mm, const ggml_tensor * weights, const ggml_tensor * pre, const ggml_tensor * rms,
        const ggml_tensor * mul);
void ggml_cuda_op_dsv4_hc_post_pre_fused(ggml_backend_cuda_context & ctx, ggml_tensor * post,
        const ggml_tensor * rms_flat, const ggml_tensor * mm, ggml_tensor * weights, const ggml_tensor * pre,
        const ggml_tensor * rms, ggml_tensor * mul);

// Whether node is a front's mul, the sublayer's normed mix: the fused front writes its q8_1 copy beside it when the
// evaluation holds one for it (ggml_cuda_mmvq_shared_q8_1::produce) and n_embd is a multiple of MATRIX_ROW_PADDING,
// not under GGML_CUDA_HC_PRE_GRAM_LEGACY. Where the front does not run fused, its readers quantize it themselves.
bool ggml_cuda_dsv4_hc_writes_q8_1(const ggml_tensor * node);
