#pragma once

// The q8_1 wire for the data-parallel row gather (plan 17 D1, ggml_cuda_ar_allgather): each rank quantizes the rows it
// owns into the fork's 36-byte block_q8_1, the gather carries 1.125 bytes a value instead of F32's 4, and every rank
// dequantizes the gathered payload into its F32 tensor. A row is a whole number of 32-value blocks (n_embd 4,608
// divides by 32) and every rank's part of the gather starts and ends on a row, so no block is shared between ranks.
//
// The payload must requantize, in the MoE's own quantizer, to exactly the q and d the F32 payload would. Quantize uses
// quantize_mmq_q8_1's arithmetic (d_inv = 127/amax, q = roundf(x*d_inv), d = 1/d_inv kept as f16); dequantize restores
// x = q*d exactly in f32; the consumer then finds amax = 127*d, d_inv' = 1/d and q' = q, and the f16 d round-trips.
// That holds where the consumer's q8_1 scale block is 32 values (MMQ layouts D4 and DS4, which cover every served type
// except Q2_K, whose D2S6 layout scales 64 values and must run with the wire off).

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

void ggml_cuda_prefill_sp_q8_1_quantize(cudaStream_t stream, const float * src, void * dst, int64_t ne);
void ggml_cuda_prefill_sp_q8_1_dequantize(cudaStream_t stream, const void * src, float * dst, int64_t ne);

// The payload of ne F32 values: one 36-byte block every 32 values.
size_t ggml_cuda_prefill_sp_q8_1_bytes(int64_t ne);

// Per-device staging payload for ne values, kept for the process so a captured step allocates nothing: on a capturing
// stream only an existing buffer is returned, else nullptr and the caller keeps the stock F32 path. The cache grows
// (cudaMalloc) only outside capture and never shrinks. The buffer holds whatever the caller wrote; zero the parts
// outside the rank's own before an OR-gather.
void * ggml_cuda_prefill_sp_q8_1_staging(int device, int64_t ne, cudaStream_t stream, int rank = 0);
