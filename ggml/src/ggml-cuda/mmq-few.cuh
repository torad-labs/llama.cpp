#pragma once

#include "common.cuh"

// ids != nullptr: MUL_MAT_ID
bool ggml_cuda_should_use_mmq_few(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, int cc);

// the bytes of the few-row MMQ copy of src1 for src0's type (MUL_MAT only, no ids)
size_t ggml_cuda_mmq_few_shared_q8_size(const ggml_tensor * src0, const ggml_tensor * src1);

void ggml_cuda_mul_mat_q_few(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst);

// src1_q_shared != nullptr: the copy is shared with the other MUL_MATs that read src1, and quantize_src1 says whether
// this call makes it
void ggml_cuda_mul_mat_q_few(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst,
    char * src1_q_shared, bool quantize_src1);

// A group: MUL_MATs of one src1 by weights of one K-quant type, each of which ggml_cuda_should_use_mmq_few takes, run as
// one launch over all their row tiles, each output the same bit for bit as its own launch's
static constexpr int MMQ_FEW_GROUP_MAX = 8;

// whether mm joins the group that first leads (first itself included): src1, type, row length and tile shape agree
bool ggml_cuda_mmq_few_group_fits(const ggml_tensor * first, const ggml_tensor * mm);

// the n MUL_MATs dsts as one launch on the shared copy of their src1, which quantize_src1 says whether this call makes
void ggml_cuda_mul_mat_q_few_group(ggml_backend_cuda_context & ctx, ggml_tensor * const * dsts, int n,
    char * src1_q_shared, bool quantize_src1);
