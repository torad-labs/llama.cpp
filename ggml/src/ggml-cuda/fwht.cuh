#include "common.cuh"

// Returns whether the Fast Walsh-Hadamard transform could be used.
bool ggml_cuda_op_fwht(ggml_backend_cuda_context & ctx, const ggml_tensor * src, ggml_tensor * dst);
bool ggml_cuda_op_fwht_signed(ggml_backend_cuda_context & ctx, const ggml_tensor * src,
                              const ggml_tensor * signs, ggml_tensor * dst);
// The signed transform of the contiguous copy of src, a view of F32 data (as a CONT would copy it), read through the
// view itself, so the copy is never made.
bool ggml_cuda_op_fwht_view(ggml_backend_cuda_context & ctx, const ggml_tensor * src, const ggml_tensor * signs,
                            ggml_tensor * dst);

// rms_norm (eps from rms_norm), its weight multiply by w, the sign flip and the transform into dst in one launch, at up to 8 tokens.
// normed: the multiply's result when other nodes read it, else nullptr. release_to: a handoff slot each block releases to
// once its normed values are stored (ggml_cuda_ssm_conv_ab_slots), or nullptr. Returns the blocks it launched.
bool ggml_cuda_rms_norm_fwht_supported(const ggml_tensor * rms_norm, const ggml_tensor * w, const ggml_tensor * normed,
                                       const ggml_tensor * signs, const ggml_tensor * dst);
int  ggml_cuda_op_rms_norm_fwht(ggml_backend_cuda_context & ctx, const ggml_tensor * rms_norm, const ggml_tensor * w,
                                ggml_tensor * normed, const ggml_tensor * signs, ggml_tensor * dst,
                                unsigned int * release_to = nullptr);
