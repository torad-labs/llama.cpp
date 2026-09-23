#include "common.cuh"

#define MMVF_MAX_BATCH_SIZE 8 // Max. batch size for which to use MMVF kernels.

void ggml_cuda_mul_mat_vec_f(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst,
    const ggml_cuda_mm_fusion_args_host * fusion = nullptr);

void ggml_cuda_op_mul_mat_vec_f(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream);

// Whether the kernel can run src0 at all (type, alignment); ggml_cuda_should_use_mmvf adds when it is the fastest choice.
bool ggml_cuda_mmvf_supports(enum ggml_type type, const int64_t * src0_ne, const size_t * src0_nb);

// Whether to run src0 x src1 on the kernel: both operands pass ggml_cuda_mmvf_supports (src1 as f32) and the kernel is
// the fastest choice for src0 at ne11 columns.
bool ggml_cuda_should_use_mmvf(enum ggml_type type, int cc, const int64_t * src0_ne, const size_t * src0_nb,
        const int64_t * src1_ne, const size_t * src1_nb, int64_t ne11);
