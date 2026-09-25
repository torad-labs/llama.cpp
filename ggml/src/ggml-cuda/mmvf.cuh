#include "common.cuh"

#define MMVF_MAX_BATCH_SIZE 8 // Max. batch size for which to use MMVF kernels.

void ggml_cuda_mul_mat_vec_f(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst,
    const ggml_cuda_mm_fusion_args_host * fusion = nullptr);

void ggml_cuda_op_mul_mat_vec_f(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream);

// Whether the kernel can run src0 x src1 at all: src0 f32/f16/bf16, src1 f32, and both laid out and placed so that its
// paired reads are aligned (data address and strides, checked at dispatch, when the data is allocated). A fused gate is
// read like src0 and is checked as ggml_cuda_mmvf_supports(gate, src1). ggml_cuda_should_use_mmvf adds when the kernel
// is the fastest choice.
bool ggml_cuda_mmvf_supports(const ggml_tensor * src0, const ggml_tensor * src1);

// Whether to run src0 x src1 on the kernel: ggml_cuda_mmvf_supports holds and the kernel is the fastest choice for src0
// at ne11 columns.
bool ggml_cuda_should_use_mmvf(const ggml_tensor * src0, const ggml_tensor * src1, int cc, int64_t ne11);

// dst_a = src0_a x src1 and dst_b = src0_b x src1 in one launch, as the kernel's two channels: channel 1 reaches src0_b
// and dst_b through the channel strides. Each product keeps the kernel, block size and summation order it has alone.
// ggml_cuda_mmvf_pair_supports must hold.
void ggml_cuda_mul_mat_vec_f_pair(ggml_backend_cuda_context & ctx, const ggml_tensor * src0_a, const ggml_tensor * src0_b,
    const ggml_tensor * src1, ggml_tensor * dst_a, ggml_tensor * dst_b);

// Whether the kernel can run the pair: both products 2-D with up to MMVF_MAX_BATCH_SIZE columns and supported, the
// weights of one type, shape and strides, the outputs F32 of one shape, strides and precision, and each second tensor in
// the same buffer as the first at an element offset that fits the kernel's int channel stride.
bool ggml_cuda_mmvf_pair_supports(const ggml_tensor * src0_a, const ggml_tensor * src0_b, const ggml_tensor * src1,
    const ggml_tensor * dst_a, const ggml_tensor * dst_b);
