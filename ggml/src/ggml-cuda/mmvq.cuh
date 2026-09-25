#include "common.cuh"

#define MMVQ_MAX_BATCH_SIZE 8 // Max. batch size for which to use MMVQ kernels.

// Max. columns for which a MUL_MAT (not MUL_MAT_ID) fuses a bias or residual add into MMVQ, reading one per column at the
// output's column stride: an MTP verify's width at two drafts. A gate and GLU fuse at one column only. Measured on
// sm_120 (5070 Ti), net of the launches the fusion removes, per step: the add -302 / -92 / +768 us at 2 / 3 / 4 columns,
// the gate/up + GLU +69 / +393 / +162 us.
#define MMVQ_MAX_FUSED_NCOLS 3

// Whether a fused MUL_MAT operand (a bias added after the matmul) can be read at dst's strides, as the fused MMVQ kernel
// reads it: dst's shape, f32, contiguous rows, and dst's stride in every dimension that has more than one element.
static inline bool ggml_cuda_mmvq_fusion_operand_ok(const ggml_tensor * operand, const ggml_tensor * dst) {
    if (operand->type != GGML_TYPE_F32 || operand->nb[0] != sizeof(float) || !ggml_are_same_shape(operand, dst)) {
        return false;
    }
    for (int d = 1; d < GGML_MAX_DIMS; ++d) {
        if (dst->ne[d] > 1 && operand->nb[d] != dst->nb[d]) {
            return false;
        }
    }
    return true;
}

bool ggml_cuda_should_use_mmvq(enum ggml_type type, int cc, int64_t ne11);

// Returns the maximum batch size for which MMVQ should be used for MUL_MAT_ID,
// based on the quantization type and GPU architecture (compute capability).
int get_mmvq_mmid_max_batch(ggml_type type, int cc);

void ggml_cuda_mul_mat_vec_q(ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst, const ggml_cuda_mm_fusion_args_host * fusion = nullptr);

void ggml_cuda_op_mul_mat_vec_q(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream);
