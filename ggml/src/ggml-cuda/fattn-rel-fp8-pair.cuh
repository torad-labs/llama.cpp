#pragma once

// FP8 accumulation for the relative decode, behind GGML_CUDA_FATTN_FP8_ACCUM (off by default).
//
// The reference design is QGRE's lossless FP8 accumulator (campaign row inkling-fp8-accumulation):
// ksl pair_kslice_sum.cuh and pair_partial_row_combine_two_term.cuh in qgre-silicon, and the scalar pair /
// block-scaled MMA headers they include. Ported laws, in their words at our engine's sizes:
//
// - One e4m3 code carries ONE value under ONE E8M0 block scale of 32 codes: value = code * 2^(sf - 127),
//   the pair. A block of 32 values shares its scale (fattn_fp8_encode_block), the ceil-pow2 byte picked
//   from the block's amax so a code can never leave e4m3's 448.
// - A rounded value that is later an addend is carried TWO-TERM: hi is the RNE pair of v, lo the RNE pair
//   of v - decode(hi), so lo holds exactly what hi cannot (fattn_fp8_encode_two, one rounding each).
// - Sums ride the sm_120a block-scaled MMA (m16n8k32, kind::mxf8f6f4, ue8m0 scales): the fp32 lives only
//   in the tensor-core accumulator fragment and NEVER crosses a memory seam; the fold exits through ONE
//   encode (fattn_fp8_encode_two), so a carried partial's lo carries what its hi cannot.
// - Every scale operand's FINAL PRODUCER is an LDS from shared memory (fattn_fp8_lds_sf). The sm_120a
//   QMMA scale-collector erratum makes an ALU-assembled scale adjacent to the MMA latch a stale byte; the
//   broadcast into a shared word at staging and the load back are the whole fix (QGRE CP-ROWDOT-HALF).
// - A stored partial keeps today's 4 bytes an element: [hi code, lo code, hi sf, lo sf] in one u32 word
//   (fattn_fp8_slot_pack). The partial plane's traffic is unchanged from the f32 it replaces; the win this
//   path is measured on is the fp8 MMA rate and the single rounding per fold, not bytes. The arm's
//   falsifier is hi+lo quality against f16 and prefill time (row note, 2026-11-22).
//
// The fold shrinks to our geometry: one warp folds 32 two-term partials per trip (two K32 blocks, hi and
// lo), up to 256 through eight retained group sums; the per-term form sums up to 16 terms of ONE element,
// each under its own scale (four code words, four scale words - a 128-value head is four K32 blocks, so
// four terms an element slot against QGRE's sixteen). The rel-mma and merge wiring lands after
// inkling-small's D2 hunks leave the shared files; this header is the primitive both sides meet on.
//
// Provenance: the PTX, the SFA/SFB selector layout and the one-hot probes are QGRE's verified sm_120a
// sequence (CUTLASS cute/arch/mma_sm120.hpp SM120_16x8x32_TN_VS<e4m3,e4m3,f32,ue8m0>); the encode math is
// QGRE's e4m3_pair_encode / e4m3_pair_encode_two_term, ported to ggml types.
//
// Verified (the test): tests/test-inkling-fp8-pair.cpp on the box. Not wired into the decode yet.

#include "common.cuh"

// GGML_CUDA_FATTN_FP8_ACCUM: the relative decode's folds accumulate in e4m3 hi+lo pairs on the block-scaled
// MMA where the term count fits. Declared in tunables.toml + ggml-tunables.inc; off, every path above is
// byte-identical.
static inline bool fattn_fp8_accum() {
    static const bool on = ggml_env_switch("GGML_CUDA_FATTN_FP8_ACCUM");
    return on;
}

#if defined(BLACKWELL_MMA_AVAILABLE)

// One e4m3 value: value = code * 2^(sf - 127). sf 0 with code 0 is exact zero.
struct fattn_fp8_pair {
    uint8_t code;
    uint8_t sf;
};

// A value as hi + lo, both pairs, both RNE (the two-term carrier of every partial and every fold exit).
struct fattn_fp8_two {
    fattn_fp8_pair hi;
    fattn_fp8_pair lo;
};

static constexpr uint8_t FATTN_FP8_UNIT_SF  = 127;   // 2^0
static constexpr uint8_t FATTN_FP8_ONE_CODE = 0x38;  // e4m3 1.0
// A fold's capacity: four code words and four scale words a lane, sixteen terms an element.
static constexpr int FATTN_FP8_TERMS_MAX = 16;
// The block-cooperative group: one scale a 32-lane block of partials, the MMA's scale_vec::1X grain.
static constexpr int FATTN_FP8_GROUP = 32;
// 256 partials an element fold in two rounds of eight group sums.
static constexpr int FATTN_FP8_GROUPS_MAX = 8;

// The exact zero pair: code 0 under scale byte 0 (0 * 2^-127 = +0; the scale byte of a zero is never read
// as a multiplier of a nonzero code).
__device__ __forceinline__ fattn_fp8_pair fattn_fp8_zero_pair() {
    return fattn_fp8_pair{0u, 0u};
}

__device__ __forceinline__ fattn_fp8_two fattn_fp8_zero_two() {
    return fattn_fp8_two{{0u, 0u}, {0u, 0u}};
}

// e4m3 code -> f32, exact integer bit construction (every e4m3 value is an f32). The OCP NaN code 0x7F
// traps: no encode stores it (the 448 clamp below turns it into 0x7E), so a decoded one is a foreign byte.
__device__ __forceinline__ float fattn_fp8_code_to_f32(uint8_t b) {
    if ((b & 0x7Fu) == 0x7Fu) {
        __trap();
    }
    const uint32_t e = (uint32_t) (b >> 3) & 0xFu;
    const uint32_t m = (uint32_t) b & 0x7u;
    if (e != 0u) {
        return __uint_as_float((((uint32_t) b >> 7) << 31) | ((e - 7u + 127u) << 23) | (m << 20));
    }
    if (m == 0u) {
        return __uint_as_float(((uint32_t) b >> 7) << 31);   // +/-0
    }
    // subnormal: m * 2^-9
    const uint32_t top = 31u - __clz(m);
    return __uint_as_float((((uint32_t) b >> 7) << 31) | ((127u - 9u + top) << 23) |
                           ((m << (23u - top)) & 0x7FFFFFu));
}

// The E8M0 byte IS the f32 exponent field (bias 127): 2^(sf - 127).
__device__ __forceinline__ float fattn_fp8_sf_to_f32(uint8_t sf) {
    return __uint_as_float((uint32_t) sf << 23);
}

__device__ __forceinline__ float fattn_fp8_decode(fattn_fp8_pair p) {
    return fattn_fp8_code_to_f32(p.code) * fattn_fp8_sf_to_f32(p.sf);
}

// The ceil-pow2 E8M0 byte a block amax needs: the smallest sf with amax <= 448 * 2^(sf - 127). A mantissa
// over 0.75 climbs a binade, which keeps 1.mmm * 2^8 <= 448 exactly (e4m3's max). 0 for a zero block.
__device__ __forceinline__ uint8_t fattn_fp8_ceil_sf_bits(uint32_t amax) {
    const uint32_t e = (amax >> 23) & 0xFFu;
    const uint32_t bump = ((amax & 0x7FFFFFu) > 0x600000u) ? 1u : 0u;
    return (e + bump <= 8u) ? 0u : (uint8_t) ((e + bump - 8u) < 255u ? (e + bump - 8u) : 255u);
}

// One finite f32 (its bits) as an e4m3 code at the given scale: RNE on the 3-bit mantissa. The round adds
// half a step plus the kept mantissa's lsb, so an exact half carries only when the kept mantissa is odd;
// a carry walks into the exponent; e4 past 15 saturates at 0x7E (448, the NaN code 0x7F's neighbour).
// A value below the block's first normal binade (e4 <= 0) lands on the subnormal grid (step 2^-9 of the
// scale) by a fresh RNE of its exact significand - never by flooring the exponent, which would keep the
// normal mantissa bits and read far off (the block-cooperative encode of a lane far below its block's
// amax is exactly this case, and the box test reds on it).
__device__ __forceinline__ uint8_t fattn_fp8_code_at_bits(uint32_t bits, uint8_t sf) {
    const uint32_t e = (bits >> 23) & 0xFFu;
    if (e == 0xFFu) {
        __trap();   // the caller's value was not finite: a stale or foreign pair byte
    }
    const uint32_t sign = bits >> 31;
    if (e == 0u) {
        return (uint8_t) (sign << 7);   // f32 zero and subnormals are below any block's grid
    }
    const uint32_t mantissa = bits & 0x7FFFFFu;
    const uint32_t rounded = mantissa + 0x7FFFFu + ((mantissa >> 20) & 1u);
    const int e4 = (int) e - (int) sf + 7 + (int) (rounded >> 23);
    if (e4 >= 16) {
        return (uint8_t) ((sign << 7) | 0x7Eu);   // 448: the NaN code 0x7F's neighbour
    }
    if (e4 <= 0) {
        // Subnormal grid: value = (m / 8) * 2^-6 at this scale, step 2^-9, so m = RNE(sig * 2^(e4 - 21))
        // with sig = 1.M in Q23. m == 8 carries to e4 = 1's mantissa 0, which is the same code byte.
        const uint32_t sig = 0x800000u | mantissa;
        const int sh = 21 - e4;   // 21..147; past 31 the exact value is below half a step
        uint32_t m = 0u;
        if (sh < 32) {
            const uint32_t base = sig >> sh;
            const uint32_t rem = sig & ((1u << sh) - 1u);
            const uint32_t half = 1u << (sh - 1);
            m = base + ((rem > half || (rem == half && (base & 1u))) ? 1u : 0u);
        }
        return (uint8_t) ((sign << 7) | (uint8_t) m);
    }
    const uint8_t mag = (uint8_t) (((uint32_t) e4 << 3) | ((rounded >> 20) & 7u));
    return (uint8_t) ((sign << 7) | mag);
}

// The one-shot pair of a finite f32: its own ceil-pow2 scale, RNE. The scale is the carrier; a reader that
// holds a block scale (the block-cooperative encode) uses fattn_fp8_code_at_bits directly.
__device__ __forceinline__ fattn_fp8_pair fattn_fp8_encode(float v) {
    const uint32_t bits = __float_as_uint(v);
    const uint32_t e = (bits >> 23) & 0xFFu;
    if (e == 0xFFu) {
        __trap();
    }
    if (e == 0u) {
        return fattn_fp8_pair{(uint8_t) ((bits >> 31) << 7), 0u};
    }
    const uint32_t bump = ((bits & 0x7FFFFFu) > 0x600000u) ? 1u : 0u;
    const uint8_t sf = (e + bump <= 8u) ? 0u : (uint8_t) ((e + bump - 8u) < 255u ? (e + bump - 8u) : 255u);
    return fattn_fp8_pair{fattn_fp8_code_at_bits(bits, sf), sf};
}

// The two-term encode: hi = RNE pair of v, lo = RNE pair of the residue. decode(hi) is exact (an e4m3
// value times a power of two), so the subtract rounds once; |v - (hi + lo)| <= half a lo step.
__device__ __forceinline__ fattn_fp8_two fattn_fp8_encode_two(float v) {
    fattn_fp8_two r;
    r.hi = fattn_fp8_encode(v);
    r.lo = fattn_fp8_encode(v - fattn_fp8_decode(r.hi));
    return r;
}

// The partial slot: today's 4 bytes an element, [hi code, lo code, hi sf, lo sf] in one word. The merge
// reads the word whole; nothing downstream sees a wider value.
__device__ __forceinline__ uint32_t fattn_fp8_slot_pack(fattn_fp8_two p) {
    return (uint32_t) p.hi.code | ((uint32_t) p.lo.code << 8) | ((uint32_t) p.hi.sf << 16) |
           ((uint32_t) p.lo.sf << 24);
}

__device__ __forceinline__ fattn_fp8_two fattn_fp8_slot_unpack(uint32_t w) {
    return fattn_fp8_two{{(uint8_t) (w & 0xFFu), (uint8_t) ((w >> 16) & 0xFFu)},
                         {(uint8_t) ((w >> 8) & 0xFFu), (uint8_t) ((w >> 24) & 0xFFu)}};
}

// The block-cooperative encode (every lane of the warp calls it, the amax is warp-wide): ONE ceil-pow2
// E8M0 byte for the block of 32 values, the lane's code RNE at that scale. A block whose scale derives to
// zero (all-zero, or below the smallest E8M0 step) is the exact zero pair - the divide by 2^-127 that
// e8m0 byte 0 would spell is never entered.
__device__ __forceinline__ fattn_fp8_pair fattn_fp8_encode_block(float v) {
    const uint32_t amax = __reduce_max_sync(0xFFFFFFFFu, __float_as_uint(v) & 0x7FFFFFFFu);
    const uint8_t sf = fattn_fp8_ceil_sf_bits(amax);
    if (sf == 0u) {
        return fattn_fp8_zero_pair();
    }
    return fattn_fp8_pair{fattn_fp8_code_at_bits(__float_as_uint(v), sf), sf};
}

// ── the block-scaled MMA ─────────────────────────────────────────────────────────────────────────────────
// m16n8k32, e4m3 x e4m3, fp32 accumulator, E8M0 block scales (scale_vec::1X: one byte a 32-code block).
// A: four ldmatrix fragment registers; B: two. SFA/SFB: four-byte scale words, register-direct into the
// instruction (the caller's final producer must be fattn_fp8_lds_sf: the erratum law above).
// byte_id selects the scale byte a K32 block reads of the supplied word; thread_id is 0 at our shapes.
__device__ __forceinline__ void fattn_fp8_mma_word_sf(
        const uint32_t A[4], const uint32_t B[2], float C[4], uint32_t SFA, uint16_t byte_id_a,
        uint32_t SFB, uint16_t byte_id_b) {
    const uint16_t thread_id = 0u;
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.f32.e4m3.e4m3.f32.ue8m0 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, "
        "{%10}, {%11, %12}, {%13}, {%14, %15};"
        : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]),
          "r"(SFA), "h"(byte_id_a), "h"(thread_id),
          "r"(SFB), "h"(byte_id_b), "h"(thread_id));
}

// The erratum-law scale loader: the scale operand's final producer is this shared load, never an ALU
// broadcast assembled at the MMA call (sm_120a QMMA scale-collector erratum).
__device__ __forceinline__ uint32_t fattn_fp8_lds_sf(uint32_t addr) {
    uint32_t w;
    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(w) : "r"(addr));
    return w;
}

// ldmatrix x4 and the XOR bank-conflict swizzle the staged tiles use (16-byte chunks a row; STRIDE bytes
// a row, 64 here).
__device__ __forceinline__ void fattn_fp8_ldmatrix4(uint32_t r[4], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}

template <int STRIDE>
__device__ __forceinline__ uint32_t fattn_fp8_swizzle(uint32_t row, uint32_t col) {
    if constexpr (STRIDE > 16) {
        col ^= (row % 8u) / (128 / STRIDE);
    }
    return row * STRIDE + col * 16;
}

// A warp's own fold scratch: the 16-row x 64-B A image, the 8-row x 64-B B image (also the scale table's
// 32 x 16 B span), and the four scale words the block sum leaves for the MMAs to load.
struct alignas(16) fattn_fp8_fold_stage {
    uint8_t  a[16 * 64];
    uint8_t  b[8 * 64];
    uint32_t sf[4];
};

// Zero both images (16-byte stores) and close any preceding reduction's reads.
__device__ __forceinline__ void fattn_fp8_zero_stage(int lane, fattn_fp8_fold_stage & s) {
    __syncwarp();
    uint4 * va = (uint4 *) s.a;
    uint4 * vb = (uint4 *) s.b;
    const uint4 z = make_uint4(0u, 0u, 0u, 0u);
    for (int i = lane; i < 64; i += 32) {
        va[i] = z;
    }
    for (int i = lane; i < 32; i += 32) {
        vb[i] = z;
    }
    __syncwarp();
}

// The block sum: sum over the warp's 32 two-term partials (one slot word a lane), exiting two-term.
// Each value's hi and lo are re-encoded at the block's ONE shared scale (two serial MMAs - the hi block
// then the lo block - into one fp32 accumulator), so the low terms are never rounded away before the sum,
// and the sum leaves through one fattn_fp8_encode_two. Unity B (e4m3 1.0) makes the MMA read the whole
// K32 block; every column of the result is the block's sum and the exit takes column 0.
__device__ __forceinline__ fattn_fp8_two fattn_fp8_fold32(uint32_t slot, int lane, fattn_fp8_fold_stage & s) {
    fattn_fp8_zero_stage(lane, s);

    const fattn_fp8_two v = fattn_fp8_slot_unpack(slot);
    const fattn_fp8_pair hi = fattn_fp8_encode_block(fattn_fp8_decode(v.hi));
    const fattn_fp8_pair lo = fattn_fp8_encode_block(fattn_fp8_decode(v.lo));

    // A row 0: block 0 = the block's hi codes, block 1 = its lo codes, one byte a lane; every other row
    // stays zero (it feeds D rows and columns nothing reads).
    s.a[lane]      = hi.code;
    s.a[32 + lane] = lo.code;
    // B: the unity code in row 0's words; the four scale words of the two MMAs, replicated x4: A0 the hi
    // block, B0 unity, A1 the lo block, B1 unity.
    if (lane < 16) {
        ((uint32_t *) s.b)[lane] = (uint32_t) FATTN_FP8_ONE_CODE * 0x01010101u;
    }
    if (lane == 0) {
        s.sf[0] = (uint32_t) hi.sf * 0x01010101u;
        s.sf[1] = (uint32_t) FATTN_FP8_UNIT_SF * 0x01010101u;
        s.sf[2] = (uint32_t) lo.sf * 0x01010101u;
        s.sf[3] = (uint32_t) FATTN_FP8_UNIT_SF * 0x01010101u;
    }
    __syncwarp();

    const uint32_t a_base = (uint32_t) __cvta_generic_to_shared(s.a);
    const uint32_t b_base = (uint32_t) __cvta_generic_to_shared(s.b);
    const uint32_t sf_addr = (uint32_t) __cvta_generic_to_shared(s.sf);
    const uint32_t a_addr = a_base + fattn_fp8_swizzle<64>((uint32_t) (lane % 16), (uint32_t) (lane / 16));
    const uint32_t b_addr = b_base + fattn_fp8_swizzle<64>((uint32_t) (lane % 8), (uint32_t) (lane / 8));

    uint32_t a0[4], a1[4], b01[4];
    fattn_fp8_ldmatrix4(a0, a_addr);
    fattn_fp8_ldmatrix4(a1, a_addr ^ 32u);   // the row's other K32 block: the lo codes at bytes 32..63
    fattn_fp8_ldmatrix4(b01, b_addr);

    float c[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    fattn_fp8_mma_word_sf(a0, &b01[0], c, fattn_fp8_lds_sf(sf_addr), 0u,
                          fattn_fp8_lds_sf(sf_addr + 4u), 0u);
    fattn_fp8_mma_word_sf(a1, &b01[2], c, fattn_fp8_lds_sf(sf_addr + 8u), 0u,
                          fattn_fp8_lds_sf(sf_addr + 12u), 0u);
    __syncwarp();

    // Row 0 of the result sits in c[0] of lanes 0..3; every column equals the block sum under the unity B.
    return fattn_fp8_encode_two(__shfl_sync(0xFFFFFFFFu, c[0], 0));
}

// The element fold: n two-term slot words of ONE element (n <= 256: streams, slices, partial banks),
// folded in groups of 32 - each group's sum exits two-term into lane `group` - then the retained group
// sums fold once more, one exit. The caller stages every warp on its own scratch and passes its lane.
// Lanes past n feed the exact zero pair.
__device__ __forceinline__ fattn_fp8_two fattn_fp8_fold_slots(
        const uint32_t * slots, int n, int lane, fattn_fp8_fold_stage & s) {
    const int groups = (n + FATTN_FP8_GROUP - 1) / FATTN_FP8_GROUP;
    if (groups > FATTN_FP8_GROUPS_MAX) {
        __trap();   // the two-round fold holds 256 partials; a wider bank is a contract change
    }
    const uint32_t zero = fattn_fp8_slot_pack(fattn_fp8_zero_two());
    fattn_fp8_two group = fattn_fp8_zero_two();
    fattn_fp8_two retained = fattn_fp8_zero_two();
    for (int g = 0; g < groups; ++g) {
        const int i = g * FATTN_FP8_GROUP + lane;
        group = fattn_fp8_fold32(i < n ? slots[i] : zero, lane, s);
        if (lane == g) {
            retained = group;
        }
        __syncwarp();
    }
    if (groups <= 1) {
        return group;
    }
    return fattn_fp8_fold32(fattn_fp8_slot_pack(retained), lane, s);
}

// ── the one-hot per-term fold ────────────────────────────────────────────────────────────────────────────
// Up to FATTN_FP8_TERMS_MAX terms of ONE element, each term a pair under its OWN scale, summed by MMAs
// whose B operand is a register one-hot (e4m3 1.0 at the term's K row): MMA t reads exactly column t of
// the lane's term codes under term t's scale byte, so no term is rescaled to a shared scale and no term is
// merged before the silicon sum. This is QGRE's packed contraction at our capacity: an element's 16 terms
// are the four code words and four scale words of fattn_fp8_terms, each word four terms a byte.
struct fattn_fp8_terms {
    uint32_t code[4];
    uint32_t sf[4];
};

// Every lane folds its OWN element (32 elements a warp in two 16-row tiles). The exit is one two-term
// encode a lane. rank: live terms, guarded at compile time by the caller's term count.
template <int N_TERMS>
__device__ __forceinline__ fattn_fp8_two fattn_fp8_sum_terms(
        const fattn_fp8_terms & terms, int lane, fattn_fp8_fold_stage & s) {
    static_assert(N_TERMS >= 1 && N_TERMS <= FATTN_FP8_TERMS_MAX,
                  "the fold holds four code words and four scale words");
    constexpr uint32_t sf_words = ((uint32_t) N_TERMS + 3u) >> 2u;

    fattn_fp8_zero_stage(lane, s);
    // Element e's row (e & 15 of its tile's 16, block e >> 4, columns 0..15): its four code words then
    // zeros, each 16-byte chunk at its own swizzled address (the XOR may swap the pair's order). The scale
    // table takes its live scale words at e * 16.
    const uint32_t row = (uint32_t) lane & 15u;
    const uint32_t block_col = ((uint32_t) lane >> 4u) << 1u;
    *(uint4 *) (s.a + fattn_fp8_swizzle<64>(row, block_col)) =
        make_uint4(terms.code[0], terms.code[1], terms.code[2], terms.code[3]);
    *(uint4 *) (s.a + fattn_fp8_swizzle<64>(row, block_col + 1u)) = make_uint4(0u, 0u, 0u, 0u);
    for (uint32_t w = 0; w < sf_words; ++w) {
        ((uint32_t *) (s.b + ((uint32_t) lane << 4)))[w] = terms.sf[w];
    }
    if (lane == 0) {
        s.sf[0] = (uint32_t) FATTN_FP8_UNIT_SF * 0x01010101u;   // the unity SFB word
    }
    __syncwarp();

    const uint32_t a_base = (uint32_t) __cvta_generic_to_shared(s.a);
    const uint32_t table  = (uint32_t) __cvta_generic_to_shared(s.b);
    uint32_t a_t0[4], a_t1[4];
    fattn_fp8_ldmatrix4(a_t0, a_base + fattn_fp8_swizzle<64>(row, (uint32_t) (lane / 16)));
    fattn_fp8_ldmatrix4(a_t1, a_base + fattn_fp8_swizzle<64>(row, (uint32_t) (lane / 16) + 2u));
    // The scale supplier map (m16n8k32 scale operands): lane 4q supplies row q, lane 4q + 1 row q + 8.
    const uint32_t supplier_row = ((uint32_t) lane >> 2u) + (((uint32_t) lane & 1u) << 3u);
    uint32_t sfa_t0[4] = {0u, 0u, 0u, 0u};
    uint32_t sfa_t1[4] = {0u, 0u, 0u, 0u};
    for (uint32_t w = 0; w < sf_words; ++w) {
        sfa_t0[w] = fattn_fp8_lds_sf(table + (supplier_row << 4) + (w << 2));
        sfa_t1[w] = fattn_fp8_lds_sf(table + ((16u + supplier_row) << 4) + (w << 2));
    }
    const uint32_t sfb_unit = fattn_fp8_lds_sf((uint32_t) __cvta_generic_to_shared(s.sf));
    const uint32_t b_quad = (uint32_t) lane & 3u;

    float c_t0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float c_t1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
    for (int t = 0; t < N_TERMS; ++t) {
        // Term t's one-hot B column: e4m3 1.0 at K row t, a register constant at byte t & 3 of the word
        // the quad t >> 2 stages; an all-ones B would sum every column under one term's scale.
        const uint32_t b_one_hot[2] = {
            (b_quad == ((uint32_t) t >> 2u)) ? ((uint32_t) FATTN_FP8_ONE_CODE << (((uint32_t) t & 3u) << 3u)) : 0u,
            0u};
        const uint16_t byte_id = (uint16_t) (t & 3);
        fattn_fp8_mma_word_sf(a_t0, b_one_hot, c_t0, sfa_t0[t >> 2], byte_id, sfb_unit, 0u);
        fattn_fp8_mma_word_sf(a_t1, b_one_hot, c_t1, sfa_t1[t >> 2], byte_id, sfb_unit, 0u);
    }
    __syncwarp();

    // Row r of a tile sits in c[0] of lanes 4r..4r+3 (r < 8) or c[2] of lanes 4(r-8)..; every column is
    // equal under the one-hot B. Lane e takes row e & 15 of tile e >> 4 from lane 4 * (e & 7), encodes once.
    const uint32_t src = ((uint32_t) lane & 7u) << 2u;
    const float v00 = __shfl_sync(0xFFFFFFFFu, c_t0[0], src);
    const float v02 = __shfl_sync(0xFFFFFFFFu, c_t0[2], src);
    const float v10 = __shfl_sync(0xFFFFFFFFu, c_t1[0], src);
    const float v12 = __shfl_sync(0xFFFFFFFFu, c_t1[2], src);
    const float own = ((uint32_t) lane & 16u) ? (((uint32_t) lane & 8u) ? v12 : v10)
                                              : (((uint32_t) lane & 8u) ? v02 : v00);
    return fattn_fp8_encode_two(own);
}

#endif // BLACKWELL_MMA_AVAILABLE
