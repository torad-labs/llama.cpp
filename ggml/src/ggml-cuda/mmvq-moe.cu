#include "mmvq-moe.cuh"
#include "mmvq.cuh"
#include "unary.cuh"
#include "vecdotq.cuh"

#include <algorithm>

// A token's routed experts are a few thousand rows each (GLM-5.3-Flash: 8 experts x 2,048 rows of 4,096 weights, 1,568
// bytes a row at IQ3_XXS), and the mat-vec kernels read them a row or two a block: a block's loads are its only bytes
// in flight, so an SM holds a few KB of requests and DRAM idles between them (RTX 5080, cold L2: 84.6 % of its
// bandwidth at gate/up, 61.1 % at down, where a 2,048-weight row leaves half of each block's threads without a block).
// Here each SM runs one block, and the block's producer warp streams its tiles of rows through a ring of shared-memory
// slots, a tile a slot, with 1D bulk copies (cp.async.bulk): the ring's slots are the SM's bytes in flight, 50-90 KB,
// whatever the row length.
//
// The work: the launch's tokens route to experts, and an expert that several tokens route to (an MTP verify's drafts
// share many) is one: every block reads ids after the dependency wait and lists the distinct experts in the order they
// first appear, each with the token/slot pairs that route to it, and a tile is 8*RPW rows of one of them. Tile t is
// expert t / ntr's row tile t % ntr. A block's first tiles, as many as its ring holds, are its own (blockIdx.x, then
// every gridDim.x-th); the rest go to whichever block asks next, by a ticket on the stream's counter
// (ggml_cuda_pq2_tile_counters), so the blocks end within a tile of each other where
// owned tiles left the launch's end to the slowest (2,048 tiles on 84 SMs: 25 against 24, a tile's 1.3 us). Two teams of
// eight consumer warps take the block's tiles in turn (team 0 the even ones). In a team, warp g takes the tile's rows
// [g*RPW, (g+1)*RPW), its 32 lanes along k as mul_mat_vec_q's (vec_dot_q_cuda at the type's vdr), for each of the
// expert's pairs: the rows are read from DRAM once.
//
// A gate (vgate) is the up rows' twin, the same rows of the gate matrix beside them in the slot under one barrier, and
// the GLU (with the SwiGLU limit and the biases) is applied as mul_mat_vec_q applies it.
//
// The bytes in flight are the rate (qgre's bulk-copy ladder on sm_120: 861 B/ns at 16 KB an SM, 877 at 32, 884 at 64),
// and a slot a team holds is not in flight: a team that kept its slot through its dot products held two of four, so the
// ring streamed ~25 KB an SM and 87.5 % of DRAM with the math on against 94 % with it compiled out. So at IQ3_XXS a warp
// takes its lanes' share of its rows out of the slot into registers (iq3_xxs_frag: the 8 grid indices, the signs and
// scale, and d of each block a lane reads, 4 registers each) and releases the slot before the math, which then runs on
// registers while the producer refills it; the plan keeps a lane's fragments to 8 (mmvq_moe_fits), past which the
// kernel spills. The math is bound by shared memory and L1, not by the dot products: the grid is gathered at random,
// so past the ring the shared memory holds it a copy per lane (grid_rep[i*32 + lane]: a warp's 32 gathers in 32 banks,
// whatever the indices), and a lane keeps its q8_1 fragments in registers, loaded once per token vector, not once per
// row. In a launch of several tokens whose pairs are as many as the experts or more, a fragment is decoded once (its grid
// gathered, its signs applied) and met with the vectors of its expert's pairs in turn (pairs_once). The other types read as mul_mat_vec_q does (vec_dot_q_cuda), from
// the slot, which their team releases after it.
//
// The tokens' q8_1 vectors come from a copy in shared memory past the ring when it fits there, which the producer's
// first bulk copy makes, before any tile: a global load that a consumer issues behind the ring's copies waits for them.
// Loaded from global memory, a gate/up's first tile took 7 us to its results where the next took 1.8 (GLM-5.3's proxy
// under -sm tensor, CUDA graphs and PDL, RTX 5070 Ti: both teams at once, the ring full meanwhile and DRAM idle for
// ~3.5 us), and at 3 tokens every change of vector paid it.
//
// No pointer carries __restrict__: with PDL a restrict load may compile to ld.global.nc, which the compiler can move
// above the grid dependency wait (upstream #24030). Nothing is read before that wait but ids when they are whole before
// the launch (ids_ready), and the weights of the tiles they name; nothing is written but the stop word of the L2 issue
// beside the stream (ggml_cuda_l2_issue_stop), an atomic add that no kernel on the stream reads.
//
// A launch triggers the next only past its dependency wait, so whatever the kernels before it wrote is whole before any
// kernel after it on the stream starts. The ids a ring launch reads (a MoE layer's top-k, written by the kernels before
// it) are then whole when a later ring launch reading the same ids starts: the down projection after a gate/up. That
// launch lists its experts and issues its block's first tiles before its wait, so its first rows stream during the
// gate/up's last tiles and the kernels between (the q8_1 of the GLU's output), where they started only past both (RTX
// 5070 Ti, GLM-5.3's proxy under -sm tensor: the down 18.75 us a layer against 14.5 at the DRAM peak, the gate/up
// 34.34 against 29.0).

#define MMVQ_MOE_NG          8                 // a team's warps: a tile's row groups
#define MMVQ_MOE_NT          2                 // teams, each on its own tiles
#define MMVQ_MOE_NW          (MMVQ_MOE_NG * MMVQ_MOE_NT) // consumer warps, and one producer warp
#define MMVQ_MOE_MAX_PAIRS   64                // tokens x experts used
#define MMVQ_MOE_MAX_SLOTS   8
#define MMVQ_MOE_SMEM_MAX    (99 * 1024)       // the shared memory a block may take on sm_90 - sm_120
#define MMVQ_MOE_SLOT_TARGET (16 * 1024)       // a slot's bytes past one row a warp: more slots, a shorter tail
#define MMVQ_MOE_PB          4                 // an expert's pairs a decoded fragment meets at once: an MTP verify's
                                               // 1-3 drafts and their token in one pass

static_assert(MMVQ_MOE_MAX_PAIRS == 64, "the producer warp lists the pairs, two a lane");
static_assert(MMVQ_MOE_MAX_PAIRS <= 64, "an expert's pairs are a 64-bit mask");

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_HOPPER && !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#define MMVQ_MOE_AVAILABLE
#endif

struct mmvq_moe_dev_args {
    const char *       vx;
    const char *       vgate;
    const block_q8_1 * y;
    const int32_t *    ids;
    const float *      x_bias;
    const float *      gate_bias;
    float *            dst;
    int64_t            stride_channel_x_bytes;
    int64_t            stride_bias;
    int                row_bytes;
    int                box_bytes;   // a matrix's part of a slot: 8*RPW rows, padded to 128 bytes
    int                nrows;
    int                nb;          // quant blocks a row
    int                ntr;         // row tiles an expert
    uint3              ntr_fd;      // ntr and nslots as fast_div_modulo takes them: a tile's expert and row tile, and
    uint3              nslots_fd;   // a sequence's slot and phase, each a multiply and a shift in place of a division
    int                n_used;
    int                ntokens;
    int                ids_stride;
    int                glu_op;
    float              glu_limit;
    int                nslots;
    int *              tile_ctr;
    int                y_bytes;     // > 0: the tokens' vectors, [y, y + y_bytes), copied into shared memory past the ring
    const float *      y_f32;       // non-null: the vectors in f32 (slot s of token t at s*y_f32_s1 + t*y_f32_s2), which
    int                y_f32_s1;    // the consumers quantize into that copy (mmvq_moe_quantize_y) where the producer
    int                y_f32_s2;    // would copy y
    int                ids_ready;   // ids whole before the launch (ggml_cuda_mmvq_moe_args): read before the dependency wait
    unsigned int *     l2_issue_stop; // bumped as the reads start (ggml_cuda_l2_issue_stop), or nullptr
    // pair p's (token p / n_used, slot p % n_used) vector, in blocks past y_base (its slot's % nchannels_y), and its
    // first dst element: made on the host, where each pair of a tile took them from 3 divisions (~15 % of the down's
    // instructions at 3 tokens). Read at a pair index the warp shares, a constant load each
    int                pair_y[MMVQ_MOE_MAX_PAIRS];
    int64_t            pair_dst[MMVQ_MOE_MAX_PAIRS];
};

// IQ3_XXS: a lane's k iterations at rpw rows a warp, 4 blocks of a row an iteration, the most its registers hold. The
// plan's rpw bounds a row (MMVQ_MOE_SLOT_TARGET: 8*rpw rows in 16 KB) and so the iterations it needs: rpw 1, 16 blocks
// (K 4,096, the most taken); rpw 2, 1,024-byte rows, 10 blocks; rpw 4, 512 bytes, 5
static constexpr __host__ __device__ int mmvq_moe_iq3_nit(const int rpw) {
    return rpw == 1 ? 4 : rpw == 2 ? 3 : 2;
}

// Whether a warp's rows fit its lanes' registers: at IQ3_XXS a lane holds rpw*nmat*nit fragments of the slot (4 registers
// each) through the math, and past 8 the kernel spills (a thread has 96 registers, 17 warps an SM). The plan takes only
// what fits, and only what fits is built.
static constexpr bool mmvq_moe_fits(const ggml_type type, const int nmat, const int rpw) {
    return type != GGML_TYPE_IQ3_XXS || rpw*nmat*mmvq_moe_iq3_nit(rpw) <= 8;
}

// Whether an instance may decode each fragment once for all the pairs of its expert (pairs_once, a launch of several
// tokens): IQ3_XXS, whose decode is most of its math, where the decoded fragments and the pairs' sums fit the registers
// beside the slot's fragments. Its own instance: beside the one-pair path's registers (a vector's fragments kept across
// tiles) the kernel spilled, and so did the one-pair path (GLM-5.3's proxy at 3 tokens: the gate/up 108 -> 139 us)
static constexpr __host__ __device__ bool mmvq_moe_pairs_fits(const ggml_type type, const int nmat, const int rpw) {
    return type == GGML_TYPE_IQ3_XXS && rpw*nmat <= 2;
}

#ifdef MMVQ_MOE_AVAILABLE
static __device__ __forceinline__ uint32_t mmvq_moe_smem_u32(const void * p) {
    return (uint32_t) __cvta_generic_to_shared(p);
}

static __device__ __forceinline__ void mmvq_moe_mbar_init(uint64_t * bar, uint32_t count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(mmvq_moe_smem_u32(bar)), "r"(count) : "memory");
}

static __device__ __forceinline__ void mmvq_moe_mbar_arrive_expect_tx(uint64_t * bar, uint32_t bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
        :: "r"(mmvq_moe_smem_u32(bar)), "r"(bytes) : "memory");
}

static __device__ __forceinline__ void mmvq_moe_mbar_arrive(uint64_t * bar) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(mmvq_moe_smem_u32(bar)) : "memory");
}

static __device__ __forceinline__ void mmvq_moe_mbar_wait(uint64_t * bar, uint32_t parity) {
    const uint32_t addr = mmvq_moe_smem_u32(bar);
    uint32_t done = 0;
    do {
        asm volatile(
            "{\n"
            ".reg .pred p;\n"
            "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
            "selp.u32 %0, 1, 0, p;\n"
            "}\n"
            : "=r"(done) : "r"(addr), "r"(parity) : "memory");
    } while (!done);
}

// Into this block's own shared memory, as mmvq-pq2-mma.cu's TMA loads: .shared::cluster would compile on sm_120 to a
// branch through a driver syscall that costs every resident thread a 14.5 KB stack. .shared::cta needs PTX ISA 8.6.
#if __CUDACC_VER_MAJOR__ > 12 || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 8)
#define MMVQ_MOE_BULK_DST "shared::cta"
#else
#define MMVQ_MOE_BULK_DST "shared::cluster"
#endif

// bytes (a multiple of 16) from src (16-byte aligned) to dst (16-byte aligned), counted on bar
static __device__ __forceinline__ void mmvq_moe_bulk_load(void * dst, const void * src, uint32_t bytes, uint64_t * bar,
                                                          uint64_t policy) {
    asm volatile(
        "cp.async.bulk." MMVQ_MOE_BULK_DST ".global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;"
        :: "r"(mmvq_moe_smem_u32(dst)), "l"((uint64_t) src), "r"(bytes), "r"(mmvq_moe_smem_u32(bar)), "l"(policy)
        : "memory");
}
#endif // MMVQ_MOE_AVAILABLE

// pair p's q8_1 vector: token p / n_used's, for expert slot p % n_used, in y or in its copy at y_base
static __device__ __forceinline__ const block_q8_1 * mmvq_moe_y(const mmvq_moe_dev_args & a, const block_q8_1 * y_base,
                                                               const int p) {
    return y_base + a.pair_y[p];
}

#define MMVQ_MOE_QY_CHUNK 8 // float4s a consumer thread loads before it quantizes them: a round trip to L2 for all

// a.y_f32: the consumer threads quantize the pairs' vectors into the copy at y_smem, bit for bit as quantize_q8_1 writes
// them. A thread takes a float4 of a vector, so 8 lanes take a q8_1 block, and the block's max and sum take
// warp_reduce_max/sum<QK8_1>'s tree: its xor 16, 8 and 4 are xor 4, 2 and 1 across the 8 lanes (the block's lane
// 4m + i is lane m's element i), its xor 2 and 1 add the lane's elements 0 + 2 and 1 + 3, then those two.
static __device__ __forceinline__ void mmvq_moe_quantize_y(const mmvq_moe_dev_args & a, char * y_smem, const int ncols) {
    constexpr int NT    = MMVQ_MOE_NW*32;
    const int     nf4   = ncols / 4;
    const int     total = a.ntokens*a.n_used*nf4; // a whole number of q8_1 blocks: 8 lanes are all in or all out
    for (int j0 = threadIdx.x; j0 < total; j0 += MMVQ_MOE_QY_CHUNK*NT) {
        float4 v[MMVQ_MOE_QY_CHUNK];
#pragma unroll
        for (int c = 0; c < MMVQ_MOE_QY_CHUNK; ++c) {
            const int j = j0 + c*NT;
            v[c] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (j < total) {
                const int p = j / nf4;
                v[c] = *(const float4 *) (a.y_f32 + (p % a.n_used)*a.y_f32_s1 + (p / a.n_used)*a.y_f32_s2 + 4*(j - p*nf4));
            }
        }
#pragma unroll
        for (int c = 0; c < MMVQ_MOE_QY_CHUNK; ++c) {
            float s[4] = { v[c].x, v[c].y, v[c].z, v[c].w };
            float m    = fmaxf(fmaxf(fabsf(s[0]), fabsf(s[1])), fmaxf(fabsf(s[2]), fabsf(s[3])));
#pragma unroll
            for (int o = 4; o > 0; o >>= 1) {
                m = fmaxf(m, __shfl_xor_sync(0xFFFFFFFF, m, o, 8));
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    s[i] += __shfl_xor_sync(0xFFFFFFFF, s[i], o, 8);
                }
            }
            const float sum = (s[0] + s[2]) + (s[1] + s[3]);

            const int j = j0 + c*NT;
            if (j < total) {
                const float d  = m / 127.0f;
                const float xs[4] = { v[c].x, v[c].y, v[c].z, v[c].w };
                char4 q;
                q.x = m == 0.0f ? 0 : (int8_t) roundf(xs[0] / d);
                q.y = m == 0.0f ? 0 : (int8_t) roundf(xs[1] / d);
                q.z = m == 0.0f ? 0 : (int8_t) roundf(xs[2] / d);
                q.w = m == 0.0f ? 0 : (int8_t) roundf(xs[3] / d);
                const int    p  = j / nf4;
                const int    c4 = j - p*nf4;
                block_q8_1 * yb = (block_q8_1 *) y_smem + a.pair_y[p] + c4/8;
                *(char4 *) &yb->qs[4*(c4 % 8)] = q;
                if (c4 % 8 == 0) {
                    yb->ds = make_half2(d, sum);
                }
            }
        }
    }
}

template <ggml_type type, int nmat, int rpw, bool pairs_once>
__launch_bounds__((MMVQ_MOE_NW + 1)*32, 1)
static __global__ void mmvq_moe(const mmvq_moe_dev_args a) {
    static_assert(!pairs_once || mmvq_moe_pairs_fits(type, nmat, rpw), "an instance that decodes once for the pairs");
#ifdef MMVQ_MOE_AVAILABLE
    constexpr int qk              = ggml_cuda_type_traits<type>::qk;
    constexpr int qi              = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr             = get_vdr_mmvq(type);
    constexpr int lanes_per_block = qi / vdr;
    constexpr int blocks_per_iter = 32 / lanes_per_block;
    constexpr int R               = MMVQ_MOE_NG * rpw;
    constexpr vec_dot_q_cuda_t vec_dot = get_vec_dot_q_cuda(type);
    static_assert(32 % lanes_per_block == 0, "a warp takes whole blocks of a row");
    static_assert(nmat == 1 || nmat == 2, "a matrix, or a gate beside it");

    extern __shared__ __align__(128) char ring[];
    __shared__ uint64_t full[MMVQ_MOE_MAX_SLOTS];
    __shared__ uint64_t empty[MMVQ_MOE_MAX_SLOTS];
    __shared__ uint64_t y_full;                                           // the tokens' vectors copied (y_bytes > 0)
    __shared__ int      held[MMVQ_MOE_MAX_SLOTS];                         // the tile in each slot, -1: no more
    __shared__ int      expert[MMVQ_MOE_MAX_PAIRS];                       // the distinct experts
    __shared__ int      pair_e[32];                                       // pair t*n_used + s's expert, the first 32
    __shared__ int      pair_u[MMVQ_MOE_MAX_PAIRS];                       // a first pair's index in expert
    __shared__ unsigned long long pairs_of[MMVQ_MOE_MAX_PAIRS];       // each distinct expert's pairs, a bit each
    __shared__ __align__(16) uint32_t grid_rep[type == GGML_TYPE_IQ3_XXS ? 256*32 : 1];   // iq3xxs_grid, a copy a lane
    __shared__ uint64_t ksigns_s[type == GGML_TYPE_IQ3_XXS ? 128 : 1];

    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;

    if (threadIdx.x == 0) {
        for (int s = 0; s < a.nslots; ++s) {
            mmvq_moe_mbar_init(&full[s],  1);
            mmvq_moe_mbar_init(&empty[s], MMVQ_MOE_NG); // the warps of the team the slot's tile goes to
        }
        mmvq_moe_mbar_init(&y_full, 1);
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads(); // the barriers, before any arrival or wait

    // the tokens' vectors, past the ring (a.y_bytes > 0) or in global memory
    char * const y_smem = ring + (size_t) a.nslots*nmat*a.box_bytes;
    const block_q8_1 * const y_base = a.y_bytes > 0 ? (const block_q8_1 *) y_smem : a.y;

    if (warp == MMVQ_MOE_NW) {
        // The producer warp alone lists the distinct experts, in the order of their first pairs, as soon as it may read
        // ids, and starts the loads while the consumers fill their tables (they read the lists after a slot's barrier,
        // which lane 0's arrival publishes): pairs lane and lane + 32, a pair past the launch's with an expert of its own.
        // ids are a previous kernel's result: read past the dependency wait, or before it when they were whole before
        // the launch (ids_ready), from L2 (ld.global.cg), past any line of theirs an earlier kernel left in this SM's L1.
        // The tokens' vectors are previous kernels' results too, copied past the wait before any tile: a global load that
        // the consumers issue behind the ring's bulk copies waits for them (a gate/up's first tile took 7 us to its
        // results where the next took 1.8, both teams at once, while the full ring kept the SM's rows from being asked
        // for); with ids_ready the block's first tiles are ahead of the copy, asked for before the wait.
        uint64_t unchanged;
        asm volatile("createpolicy.fractional.L2::evict_unchanged.b64 %0, 1.0;" : "=l"(unchanged));
        if (!a.ids_ready) {
            ggml_cuda_pdl_sync();
            ggml_cuda_pdl_lc(); // past the wait: whatever the kernels before this one wrote is whole for the next
            if (lane == 0 && a.y_bytes > 0 && a.y_f32 == nullptr) {
                mmvq_moe_mbar_arrive_expect_tx(&y_full, a.y_bytes);
                mmvq_moe_bulk_load(y_smem, a.y, a.y_bytes, &y_full, unchanged);
            }
        }
        // the launch's reads start here (past the wait, or before it with ids_ready): an L2 issue beside the stream stops
        if (a.l2_issue_stop != nullptr && blockIdx.x == 0 && lane == 0) {
            atomicAdd(a.l2_issue_stop, 1u);
        }
        const int npairs = a.ntokens * a.n_used;
        const int p1     = lane + 32;
        const int e0     = lane < npairs ? __ldcg(&a.ids[lane % a.n_used + (lane / a.n_used)*a.ids_stride]) : -1 - lane;
        const int e1     = p1   < npairs ? __ldcg(&a.ids[p1   % a.n_used + (p1   / a.n_used)*a.ids_stride]) : -1 - p1;
        pair_e[lane]   = e0;
        pairs_of[lane] = 0;
        pairs_of[p1]   = 0;
        __syncwarp();
        const unsigned same0  = __match_any_sync(0xFFFFFFFF, e0);
        const unsigned same1  = __match_any_sync(0xFFFFFFFF, e1);
        const int      first0 = __ffs(same0) - 1;
        int            first1 = 32 + __ffs(same1) - 1;
        for (int q = 0; q < 32; ++q) {
            if (pair_e[q] == e1) {
                first1 = q;
                break;
            }
        }
        const unsigned heads0 = __ballot_sync(0xFFFFFFFF, lane < npairs && first0 == lane);
        const unsigned heads1 = __ballot_sync(0xFFFFFFFF, p1   < npairs && first1 == p1);
        const unsigned below  = (1u << lane) - 1;
        if (heads0 & (1u << lane)) {
            expert[__popc(heads0 & below)] = e0;
            pair_u[lane] = __popc(heads0 & below);
        }
        if (heads1 & (1u << lane)) {
            expert[__popc(heads0) + __popc(heads1 & below)] = e1;
            pair_u[p1] = __popc(heads0) + __popc(heads1 & below);
        }
        __syncwarp();
        if (lane < npairs) {
            atomicOr(&pairs_of[pair_u[first0]], 1ull << lane);
        }
        if (p1 < npairs) {
            atomicOr(&pairs_of[pair_u[first1]], 1ull << p1);
        }
        __syncwarp();
        const int n_tiles = (__popc(heads0) + __popc(heads1)) * a.ntr;

        // the producer: tile i of the block's sequence into slot i % nslots, once the consumers have released it. The
        // block's own tiles first (blockIdx.x, then every gridDim.x-th: the blocks stream the tiles in waves, a wave a
        // few MB of neighbouring rows), all the waves but the last; then, with a counter, a ticket a tile as
        // mmvq-pq2-mma.cu takes them: each block's last ticket is past the tiles, so the launch takes
        // n_tiles - dyn_base + gridDim.x of them, and the block that uses the last sets the counter back to 0. A ticket
        // is asked for once the tile before it is issued and used a sequence later, so its round trip to L2 overlaps the
        // wait for a slot: the barrier arrival that issues a tile is a release, which waits for the atomic before it
        // (a ticket a tile, each asked for before its tile's arrival, cost GLM's up projection 43.0 against 40.9 us on
        // an RTX 5080)
        if (lane == 0) {
            uint64_t policy; // the rows stream through L2 once
            asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(policy));
            const int n_own    = a.tile_ctr != nullptr ? max(a.nslots, n_tiles / (int) gridDim.x - 1) : n_tiles;
            const int dyn_base = a.tile_ctr != nullptr ? min(n_own * (int) gridDim.x, n_tiles) : n_tiles;
            const int last     = n_tiles - dyn_base + (int) gridDim.x - 1; // the launch's last ticket
            int n_end  = 0;
            int ticket = -1; // the next sequence's, once asked for

            // tile into sequence i's slot
            const auto issue = [&](const int i, const int tile) {
                const int s = (int) fastmodulo((uint32_t) i, a.nslots_fd);
                held[s] = tile;
                const uint2    ur    = fast_div_modulo((uint32_t) tile, a.ntr_fd);
                const int      u     = (int) ur.x;
                const int      row0  = (int) ur.y * R;
                const uint32_t bytes = (uint32_t) (min(R, a.nrows - row0) * a.row_bytes);
                const int64_t  off   = expert[u]*a.stride_channel_x_bytes + (int64_t) row0*a.row_bytes;
                char * slot = ring + (size_t) s*nmat*a.box_bytes;
                mmvq_moe_mbar_arrive_expect_tx(&full[s], nmat*bytes);
                mmvq_moe_bulk_load(slot, a.vx + off, bytes, &full[s], policy);
                if constexpr (nmat == 2) {
                    mmvq_moe_bulk_load(slot + a.box_bytes, a.vgate + off, bytes, &full[s], policy);
                }
            };

            int i = 0;
            if (a.ids_ready) {
                // the block's first own tiles before the wait, a ring's at most: weights, which no kernel writes. Not the
                // last own tile, whose issue asks for the launch's first ticket from the stream's counter, which the
                // launch before may still be taking
                for (; i < a.nslots && i + 1 < n_own; ++i) {
                    const int t = (int) blockIdx.x + i * (int) gridDim.x;
                    if (t >= dyn_base) {
                        break;
                    }
                    issue(i, t);
                }
                ggml_cuda_pdl_sync();
                ggml_cuda_pdl_lc();
                if (a.y_bytes > 0 && a.y_f32 == nullptr) {
                    mmvq_moe_mbar_arrive_expect_tx(&y_full, a.y_bytes);
                    mmvq_moe_bulk_load(y_smem, a.y, a.y_bytes, &y_full, unchanged);
                }
            }
            for (;; ++i) {
                const uint2 ph = fast_div_modulo((uint32_t) i, a.nslots_fd);
                const int   s  = (int) ph.y;
                if (i >= a.nslots) {
                    mmvq_moe_mbar_wait(&empty[s], (ph.x - 1) & 1);
                }
                int tile = -1;
                if (n_end == 0) {
                    if (i < n_own) {
                        const int t = (int) blockIdx.x + i * (int) gridDim.x;
                        tile = t < dyn_base ? t : -1;
                    } else if (dyn_base < n_tiles) {
                        if (ticket == last) {
                            atomicExch(a.tile_ctr, 0);
                        }
                        tile = dyn_base + ticket < n_tiles ? dyn_base + ticket : -1;
                    }
                }
                if (tile < 0) {
                    // the end: a slot with no rows for each team (the next NT in the sequence are one a team), its barrier
                    // completed by a plain arrival
                    held[s] = -1;
                    mmvq_moe_mbar_arrive(&full[s]);
                    if (++n_end == MMVQ_MOE_NT) {
                        break;
                    }
                    continue;
                }
                issue(i, tile);
                if (i + 1 >= n_own && dyn_base < n_tiles) {
                    ticket = atomicAdd(a.tile_ctr, 1); // the next sequence's tile, asked for now this one is issued
                }
            }
        }
        return;
    }

    // the consumers: the tables depend on nothing, so they are filled under the previous kernel, before the dependency
    // wait, and the consumer warps meet on barrier 1 once they are whole
    if constexpr (type == GGML_TYPE_IQ3_XXS) {
        for (int i = threadIdx.x; i < 256*8; i += MMVQ_MOE_NW*32) {
            const uint32_t v = iq3xxs_grid[i / 8];
            *(uint4 *) &grid_rep[4*i] = make_uint4(v, v, v, v); // entry i/8's copies 4*(i%8) to 4*(i%8) + 3
        }
        for (int i = threadIdx.x; i < 128; i += MMVQ_MOE_NW*32) {
            ksigns_s[i] = ksigns64[i];
        }
        asm volatile("bar.sync 1, %0;" :: "n"(MMVQ_MOE_NW*32) : "memory");
    }
    ggml_cuda_pdl_sync(); // the tokens are the previous kernels' results, and dst may still be read
    ggml_cuda_pdl_lc();
    if (a.y_f32 != nullptr) {
        mmvq_moe_quantize_y(a, y_smem, a.nb*qk);
        asm volatile("bar.sync 1, %0;" :: "n"(MMVQ_MOE_NW*32) : "memory"); // every consumer's blocks, before any is read
    } else if (a.y_bytes > 0) {
        mmvq_moe_mbar_wait(&y_full, 0); // the producer's copy of the tokens' vectors
    }

    const int team = warp / MMVQ_MOE_NG;                                 // the block's tiles team, team + NT, ...
    const int g    = warp % MMVQ_MOE_NG;                                 // the warp's rows of a tile, [g*rpw, (g+1)*rpw)
    const int kb0  = lane / lanes_per_block;                             // the lane's first block of a row
    const int kqs  = vdr * (lane % lanes_per_block);                     // and its quant ints in each

    // IQ3_XXS: the lane's q8_1 fragments, a k iteration each (its block's 8 ints and scale), of the vector y_key (its
    // pair_y), kept across pairs and tiles until the vector changes (not with pairs_once)
    constexpr int nit = type == GGML_TYPE_IQ3_XXS ? mmvq_moe_iq3_nit(rpw) : 1;
    [[maybe_unused]] int   yu[nit][8];
    [[maybe_unused]] float yd[nit];
    [[maybe_unused]] int   y_key = -1;
    [[maybe_unused]] const auto grid   = [&](const int i) { return grid_rep[i*32 + lane]; };
    [[maybe_unused]] const auto ksigns = [&](const int i) { return ksigns_s[i]; };

    for (int i = team;; i += MMVQ_MOE_NT) {
        const uint2 ph = fast_div_modulo((uint32_t) i, a.nslots_fd);
        const int   s  = (int) ph.y;
        mmvq_moe_mbar_wait(&full[s], ph.x & 1);
        const int tile = held[s];
        if (tile < 0) {
            break;
        }
        const uint2  ur   = fast_div_modulo((uint32_t) tile, a.ntr_fd);
        const int    u    = (int) ur.x;
        const int    e    = expert[u];
        const int    row0 = (int) ur.y * R + g*rpw; // this warp's first row
        const char * box  = ring + (size_t) s*nmat*a.box_bytes;

        // IQ3_XXS: the lane's fragments of its rows out of the slot, and the slot released before the math, so the
        // producer refills it while the warps compute: a slot is held for these loads, not for the dot products, and
        // the ring's slots are nearly all in flight at once
        [[maybe_unused]] iq3_xxs_frag wf[rpw][nmat][nit];
        if constexpr (type == GGML_TYPE_IQ3_XXS) {
#pragma unroll
            for (int it = 0; it < nit; ++it) {
                const int kb = kb0 + it*blocks_per_iter;
                if (kb < a.nb) {
#pragma unroll
                    for (int r = 0; r < rpw; ++r) {
#pragma unroll
                        for (int m = 0; m < nmat; ++m) {
                            wf[r][m][it] = iq3_xxs_frag_load(
                                (const block_iq3_xxs *) (box + m*a.box_bytes) + (g*rpw + r)*a.nb + kb, kqs);
                        }
                    }
                }
            }
            __syncwarp(); // every lane has read this slot: release it
            if (lane == 0) {
                mmvq_moe_mbar_arrive(&empty[s]);
            }
        }

        // a pair's rows: each lane's sums added over the warp, and lane 0 writes them, through the GLU with a gate
        const auto finish = [&](const int p, float (&acc)[rpw][nmat]) {
            float * const dst = a.dst + a.pair_dst[p];
#pragma unroll
            for (int r = 0; r < rpw; ++r) {
#pragma unroll
                for (int m = 0; m < nmat; ++m) {
                    acc[r][m] = warp_reduce_sum<32>(acc[r][m]);
                }
            }

#pragma unroll
            for (int r = 0; r < rpw; ++r) {
                const float * v   = acc[r];
                const int     row = row0 + r;
                if (lane == 0 && row < a.nrows) {
                    float result = v[0];
                    if (a.x_bias != nullptr) {
                        result += a.x_bias[e*a.stride_bias + row];
                    }
                    if constexpr (nmat == 2) {
                        float gate_value = v[1];
                        if (a.gate_bias != nullptr) {
                            gate_value += a.gate_bias[e*a.stride_bias + row];
                        }
                        if (a.glu_limit > 0.0f) {
                            gate_value = fminf(gate_value, a.glu_limit);
                            result     = fminf(fmaxf(result, -a.glu_limit), a.glu_limit);
                        }
                        switch (a.glu_op) {
                            case GGML_GLU_OP_SWIGLU:
                                result *= ggml_cuda_op_silu_single(gate_value);
                                break;
                            case GGML_GLU_OP_GEGLU:
                                result *= ggml_cuda_op_gelu_single(gate_value);
                                break;
                            case GGML_GLU_OP_SWIGLU_OAI:
                                result = ggml_cuda_op_swiglu_oai_single(gate_value, result);
                                break;
                            default:
                                result = result * gate_value;
                                break;
                        }
                    }
                    dst[row] = result;
                }
            }
        };

        // pairs_once (IQ3_XXS, a launch of several tokens): the expert's pairs MMVQ_MOE_PB at a time, each fragment
        // decoded once, its grid gathered and its signs applied, and met with the pairs' vectors in turn, where a pair at
        // a time decoded it again for each (GLM-5.3's proxy at 3 tokens, every token on each of its 8 experts, RTX 5070
        // Ti under ncu: the gate/up 113-116 us at 51-53 % of DRAM, its memory pipes 72-74 % busy). A vector's fragments
        // are read for each tile, not kept across tiles. Each pair's sums are added in the same order, so the same bits
        if constexpr (pairs_once) {
            for (unsigned long long pairs = pairs_of[u]; pairs != 0;) {
                int pc[MMVQ_MOE_PB]; // the pass's pairs, -1 past them
                int yo[MMVQ_MOE_PB]; // and their vectors, in blocks past y_base
#pragma unroll
                for (int q = 0; q < MMVQ_MOE_PB; ++q) {
                    pc[q] = pairs != 0 ? __ffsll(pairs) - 1 : -1;
                    yo[q] = pc[q] >= 0 ? (int) (mmvq_moe_y(a, y_base, pc[q]) - y_base) : 0;
                    pairs &= pairs - 1;
                }
                float acc[MMVQ_MOE_PB][rpw][nmat] = {};
#pragma unroll
                for (int it = 0; it < nit; ++it) {
                    const int kb = kb0 + it*blocks_per_iter;
                    if (kb < a.nb) {
                        iq3_xxs_frag_q8 wq[rpw][nmat];
#pragma unroll
                        for (int r = 0; r < rpw; ++r) {
#pragma unroll
                            for (int m = 0; m < nmat; ++m) {
                                wq[r][m] = iq3_xxs_frag_decode(wf[r][m][it], grid, ksigns);
                            }
                        }
#pragma unroll
                        for (int q = 0; q < MMVQ_MOE_PB; ++q) {
                            if (pc[q] >= 0) {
                                const block_q8_1 * yb = y_base + yo[q] + kb*(qk/QK8_1) + kqs/2;
                                int u8[8];
#pragma unroll
                                for (int l = 0; l < 8; ++l) {
                                    u8[l] = get_int_b4(yb->qs, l);
                                }
                                const float d8 = __low2float(yb->ds);
#pragma unroll
                                for (int r = 0; r < rpw; ++r) {
#pragma unroll
                                    for (int m = 0; m < nmat; ++m) {
                                        acc[q][r][m] += vec_dot_iq3_xxs_frag_q8(wq[r][m], u8, d8);
                                    }
                                }
                            }
                        }
                    }
                }
#pragma unroll
                for (int q = 0; q < MMVQ_MOE_PB; ++q) {
                    if (pc[q] >= 0) {
                        finish(pc[q], acc[q]);
                    }
                }
            }
        }

        // every pair that routes to the expert, each with its own tokens (a down projection's differ by slot): with
        // pairs_once, none past the passes above
        for (unsigned long long pairs = pairs_once ? 0 : pairs_of[u]; pairs != 0; pairs &= pairs - 1) {
            const int p = __ffsll(pairs) - 1;
            const block_q8_1 * y = mmvq_moe_y(a, y_base, p);

            float acc[rpw][nmat] = {{0.0f}};
            if constexpr (type == GGML_TYPE_IQ3_XXS) {
                const int key = a.pair_y[p]; // the vector, whichever pairs share it
                if (key != y_key) {
                    y_key = key;
#pragma unroll
                    for (int it = 0; it < nit; ++it) {
                        const int kb = kb0 + it*blocks_per_iter;
                        if (kb < a.nb) {
                            const block_q8_1 * yb = y + kb*(qk/QK8_1) + kqs/2;
#pragma unroll
                            for (int l = 0; l < 8; ++l) {
                                yu[it][l] = get_int_b4(yb->qs, l);
                            }
                            yd[it] = __low2float(yb->ds);
                        }
                    }
                }
#pragma unroll
                for (int it = 0; it < nit; ++it) {
                    const int kb = kb0 + it*blocks_per_iter;
                    if (kb < a.nb) {
#pragma unroll
                        for (int r = 0; r < rpw; ++r) {
#pragma unroll
                            for (int m = 0; m < nmat; ++m) {
                                acc[r][m] += vec_dot_iq3_xxs_frag(wf[r][m][it], yu[it], yd[it], grid, ksigns);
                            }
                        }
                    }
                }
            } else {
#pragma unroll 4
                for (int kb = kb0; kb < a.nb; kb += blocks_per_iter) {
                    const int kby = kb * (qk/QK8_1);
#pragma unroll
                    for (int r = 0; r < rpw; ++r) {
#pragma unroll
                        for (int m = 0; m < nmat; ++m) {
                            acc[r][m] += vec_dot(box + m*a.box_bytes, &y[kby], (g*rpw + r)*a.nb + kb, kqs);
                        }
                    }
                }
            }
            finish(p, acc);
        }

        if constexpr (type != GGML_TYPE_IQ3_XXS) {
            __syncwarp(); // every lane has read this slot: release it
            if (lane == 0) {
                mmvq_moe_mbar_arrive(&empty[s]);
            }
        }
    }
#else
    GGML_UNUSED(a);
    NO_DEVICE_CODE;
#endif // MMVQ_MOE_AVAILABLE
}

// ---------------------------------------------------------------------------------------------------------------------
// host

static bool mmvq_moe_legacy() {
    static const bool legacy = ggml_env_switch("GGML_CUDA_MMVQ_MOE_LEGACY");
    return legacy;
}

// The types, at the token counts, where the ring measured faster than mul_mat_vec_q in place, or level with it (moe-graph:
// GLM-5.3-Flash's routed FFN on 32 experts, CUDA graphs and PDL, RTX 5080). IQ3_XXS: 815 against 818 us for 8 layers at
// 1 token, 1,823 against 1,953 at 3. Q8_0: 989 against 978 us for 4 layers at 1 token, 2,265 against 2,382 at 3; its dot
// products read the slot, so a team holds the slot through them, and only an MTP verify's shared experts make up for
// that. IQ4_XS was slower at both, 527 against 505 and 1,241 against 1,199, so it keeps mul_mat_vec_q.
static bool mmvq_moe_takes(ggml_type type, int64_t ntokens) {
    return type == GGML_TYPE_IQ3_XXS || (type == GGML_TYPE_Q8_0 && ntokens > 1);
}

static int64_t mmvq_moe_row_bytes(ggml_type type, int64_t ncols_x) {
    return ncols_x / ggml_blck_size(type) * (int64_t) ggml_type_size(type);
}

// The shared memory the ring and the tokens' vectors share: MMVQ_MOE_SMEM_MAX past the instance's static shared memory
// as compiled (barriers and lists, and at IQ3_XXS its tables: 36,232 bytes on sm_120), read once a device. A 36 KiB
// budget for it left an MTP verify's gate/up without the room for its 3 tokens' vectors beside its ring. A pairs_once
// instance declares the same shared memory as its shape's other.
template <ggml_type type, int nmat, int rpw>
static int mmvq_moe_dyn_max() {
    if constexpr (mmvq_moe_fits(type, nmat, rpw)) {
        static int dyn_max[GGML_CUDA_MAX_DEVICES] = { 0 };
        const int  id                             = ggml_cuda_get_device();
        if (dyn_max[id] == 0) {
            cudaFuncAttributes attr;
            CUDA_CHECK(cudaFuncGetAttributes(&attr, mmvq_moe<type, nmat, rpw, false>));
            dyn_max[id] = MMVQ_MOE_SMEM_MAX - (int) attr.sharedSizeBytes;
        }
        return dyn_max[id];
    } else {
        return 0;
    }
}

template <ggml_type type>
static int mmvq_moe_dyn_max_type(const int nmat, const int rpw) {
    switch (nmat*8 + rpw) {
        case 1*8 + 1: return mmvq_moe_dyn_max<type, 1, 1>();
        case 1*8 + 2: return mmvq_moe_dyn_max<type, 1, 2>();
        case 1*8 + 4: return mmvq_moe_dyn_max<type, 1, 4>();
        case 2*8 + 1: return mmvq_moe_dyn_max<type, 2, 1>();
        case 2*8 + 2: return mmvq_moe_dyn_max<type, 2, 2>();
        case 2*8 + 4: return mmvq_moe_dyn_max<type, 2, 4>();
        default:      return 0;
    }
}

static int mmvq_moe_dyn_max(const ggml_type type, const int nmat, const int rpw) {
    switch (type) {
        case GGML_TYPE_IQ3_XXS: return mmvq_moe_dyn_max_type<GGML_TYPE_IQ3_XXS>(nmat, rpw);
        case GGML_TYPE_Q8_0:    return mmvq_moe_dyn_max_type<GGML_TYPE_Q8_0>(nmat, rpw);
        default:                return 0;
    }
}

// rpw: rows a row group, the tile 8*rpw rows; the most whose slot stays under MMVQ_MOE_SLOT_TARGET (one row a group in
// any case), and as many slots as fit beside y_bytes of the tokens' vectors, a whole number a team: sequence i's slot is
// i % nslots and its team i % NT, so a team always uses the same slots, and waits on each slot's full barrier one phase
// after the last it consumed (a team waiting on a slot another team has not consumed yet could see the parity of the
// phase before and pass early)
struct mmvq_moe_plan {
    int rpw       = 0;
    int nslots    = 0;
    int box_bytes = 0;
};

static mmvq_moe_plan mmvq_moe_make_plan(ggml_type type, int64_t row_bytes, int nmat, int64_t y_bytes = 0) {
    for (int rpw : { 4, 2, 1 }) {
        const int64_t box  = GGML_PAD(MMVQ_MOE_NG * rpw * row_bytes, 128);
        const int64_t slot = nmat * box;
        if (rpw > 1 && (slot > MMVQ_MOE_SLOT_TARGET || !mmvq_moe_fits(type, nmat, rpw))) {
            continue;
        }
        const int fit = (int) std::min<int64_t>((mmvq_moe_dyn_max(type, nmat, rpw) - y_bytes) / slot,
            MMVQ_MOE_MAX_SLOTS) / MMVQ_MOE_NT * MMVQ_MOE_NT;
        if (fit < MMVQ_MOE_NT) {
            return {};
        }
        return { rpw, fit, (int) box };
    }
    return {};
}

bool ggml_cuda_mmvq_moe_usable(int cc, ggml_type type, const void * vx, const void * vgate, int64_t ncols_x,
                               int64_t nrows_x, int64_t stride_row_x, int64_t stride_channel_x, int64_t n_used,
                               int64_t ntokens) {
    if (mmvq_moe_legacy() || !GGML_CUDA_CC_IS_NVIDIA(cc) || cc < GGML_CUDA_CC_HOPPER ||
            ggml_cuda_highest_compiled_arch(cc) < GGML_CUDA_CC_HOPPER || !mmvq_moe_takes(type, ntokens)) {
        return false;
    }
    const int64_t       ts        = ggml_type_size(type);
    const int64_t       row_bytes = mmvq_moe_row_bytes(type, ncols_x);
    const mmvq_moe_plan p         = mmvq_moe_make_plan(type, row_bytes, vgate != nullptr ? 2 : 1);
    // a tile is 8*rpw dense rows: 16-byte aligned wherever it starts when the rows are an even number of bytes, and the
    // last tile of an expert when the expert is a whole number of 16-byte units. IQ3_XXS: a lane's k iterations (4
    // blocks of a row each) fit its fragments
    return ncols_x % ggml_blck_size(type) == 0 && stride_row_x*ts == row_bytes && row_bytes % 2 == 0 &&
        (nrows_x*row_bytes) % 16 == 0 && (stride_channel_x*ts) % 16 == 0 &&
        (uintptr_t) vx % 16 == 0 && (uintptr_t) vgate % 16 == 0 &&
        ntokens >= 1 && ntokens <= MMVQ_MAX_BATCH_SIZE && n_used >= 1 && ntokens*n_used <= MMVQ_MOE_MAX_PAIRS &&
        nrows_x < (1 << 30) && row_bytes < (1 << 20) && p.rpw > 0 &&
        (type != GGML_TYPE_IQ3_XXS || (ncols_x/QK_K + 3)/4 <= mmvq_moe_iq3_nit(p.rpw));
}

template <ggml_type type, int nmat, int rpw, bool pairs_once>
static void mmvq_moe_launch(const mmvq_moe_dev_args & a, const int nblocks, cudaStream_t stream) {
    if constexpr (mmvq_moe_fits(type, nmat, rpw)) {
        const size_t smem = (size_t) a.nslots * nmat * a.box_bytes + a.y_bytes;
        CUDA_SET_SHARED_MEMORY_LIMIT((mmvq_moe<type, nmat, rpw, pairs_once>), (mmvq_moe_dyn_max<type, nmat, rpw>())); // every plan's
        const ggml_cuda_kernel_launch_params params(dim3(nblocks), dim3((MMVQ_MOE_NW + 1)*32), smem, stream);
        ggml_cuda_kernel_launch(mmvq_moe<type, nmat, rpw, pairs_once>, params, a);
    } else {
        GGML_ABORT("%s: no plan takes %d matrices at %d rows a warp for %s", __func__, nmat, rpw, ggml_type_name(type));
    }
}

// the shape's pairs_once instance where it has one (mmvq_moe_pairs_fits) and pairs_once, its other otherwise
template <ggml_type type, int nmat, int rpw>
static void mmvq_moe_launch_pairs(const mmvq_moe_dev_args & a, const bool pairs_once, const int nblocks,
                                  cudaStream_t stream) {
    if constexpr (mmvq_moe_pairs_fits(type, nmat, rpw)) {
        if (pairs_once) {
            mmvq_moe_launch<type, nmat, rpw, true>(a, nblocks, stream);
            return;
        }
    }
    mmvq_moe_launch<type, nmat, rpw, false>(a, nblocks, stream);
}

template <ggml_type type>
static void mmvq_moe_launch_type(const mmvq_moe_dev_args & a, const int nmat, const int rpw, const bool pairs_once,
                                 const int nblocks, cudaStream_t stream) {
    switch (nmat*8 + rpw) {
        case 1*8 + 1: mmvq_moe_launch_pairs<type, 1, 1>(a, pairs_once, nblocks, stream); break;
        case 1*8 + 2: mmvq_moe_launch_pairs<type, 1, 2>(a, pairs_once, nblocks, stream); break;
        case 1*8 + 4: mmvq_moe_launch_pairs<type, 1, 4>(a, pairs_once, nblocks, stream); break;
        case 2*8 + 1: mmvq_moe_launch_pairs<type, 2, 1>(a, pairs_once, nblocks, stream); break;
        case 2*8 + 2: mmvq_moe_launch_pairs<type, 2, 2>(a, pairs_once, nblocks, stream); break;
        case 2*8 + 4: mmvq_moe_launch_pairs<type, 2, 4>(a, pairs_once, nblocks, stream); break;
        default: GGML_ABORT("%s: no instance for %d matrices at %d rows a warp", __func__, nmat, rpw);
    }
}

int64_t ggml_cuda_mmvq_moe_y_bytes(int64_t ncols_x, int64_t n_used, int64_t ntokens, int64_t nchannels_y,
                                   int64_t stride_col_y, int64_t stride_channel_y) {
    const int64_t nchannels = std::min(nchannels_y, n_used); // pair p's vector is its slot's % nchannels_y
    const int64_t end       = (nchannels - 1)*stride_channel_y + (ntokens - 1)*stride_col_y + ncols_x/QK8_1; // blocks
    const int64_t bytes     = end * (int64_t) sizeof(block_q8_1);
    return bytes % 16 == 0 ? bytes : 0;
}

bool ggml_cuda_mmvq_moe_keeps_y(ggml_type type, int64_t ncols_x, bool gate, int64_t y_bytes) {
    static const bool y_global = ggml_env_switch("GGML_CUDA_MMVQ_MOE_Y_GLOBAL");
    const int           nmat      = gate ? 2 : 1;
    const int64_t       row_bytes = mmvq_moe_row_bytes(type, ncols_x);
    const mmvq_moe_plan p         = mmvq_moe_make_plan(type, row_bytes, nmat);
    return !y_global && y_bytes > 0 && p.rpw > 0 && mmvq_moe_make_plan(type, row_bytes, nmat, y_bytes).rpw == p.rpw;
}

bool ggml_cuda_mmvq_moe_quantizes_y(ggml_type type, int64_t ncols_x, bool gate, int64_t n_used, int64_t ntokens,
                                    int64_t nchannels_y, const void * y_f32, int64_t y_f32_s1, int64_t y_f32_s2) {
    static const bool legacy = ggml_env_switch("GGML_CUDA_MMVQ_MOE_QUANTIZE_LEGACY");
    if (legacy || nchannels_y != n_used || ncols_x % QK8_1 != 0 || (uintptr_t) y_f32 % 16 != 0 || y_f32_s1 % 4 != 0 ||
            y_f32_s2 % 4 != 0 || (n_used - 1)*y_f32_s1 + (ntokens - 1)*y_f32_s2 + ncols_x > INT_MAX) {
        return false;
    }
    // the q8_1 layout the launch reads (mmvq_src1_q8_1's: each vector's row padded to MATRIX_ROW_PADDING)
    const int64_t s11 = GGML_PAD(ncols_x, MATRIX_ROW_PADDING) / QK8_1;
    return ggml_cuda_mmvq_moe_keeps_y(type, ncols_x, gate, ggml_cuda_mmvq_moe_y_bytes(ncols_x, n_used, ntokens, nchannels_y,
        n_used*s11, s11));
}

void ggml_cuda_mmvq_moe(const ggml_cuda_mmvq_moe_args & args, cudaStream_t stream) {
    const int     nmat      = args.vgate != nullptr ? 2 : 1;
    const int64_t row_bytes = mmvq_moe_row_bytes(args.type, args.ncols_x);

    // routed experts read the tokens' vectors from the producer's copy past the ring when the plan keeps them there, or
    // from the copy the consumers quantize there (y_f32, which ggml_cuda_mmvq_moe_quantizes_y admitted)
    int64_t y_bytes = (uintptr_t) args.y % 16 != 0 && args.y_f32 == nullptr ? 0 : ggml_cuda_mmvq_moe_y_bytes(args.ncols_x,
        args.n_used, args.ntokens, args.nchannels_y, args.stride_col_y, args.stride_channel_y);
    if (!ggml_cuda_mmvq_moe_keeps_y(args.type, args.ncols_x, nmat == 2, y_bytes)) {
        GGML_ASSERT(args.y_f32 == nullptr && "ggml_cuda_mmvq_moe_quantizes_y keeps the quantized vectors in shared memory");
        y_bytes = 0;
    }
    GGML_ASSERT(args.y_f32 == nullptr || args.nchannels_y == args.n_used);
    const mmvq_moe_plan p = mmvq_moe_make_plan(args.type, row_bytes, nmat, y_bytes);
    GGML_ASSERT(p.rpw > 0 && "ggml_cuda_mmvq_moe_usable holds a plan");

    const int R   = MMVQ_MOE_NG * p.rpw;
    const int ntr = (int) ((args.nrows_x + R - 1) / R);

    mmvq_moe_dev_args a;
    a.vx                     = (const char *) args.vx;
    a.vgate                  = (const char *) args.vgate;
    a.y                      = (const block_q8_1 *) args.y;
    a.ids                    = args.ids;
    a.x_bias                 = args.x_bias;
    a.gate_bias              = args.gate_bias;
    a.dst                    = args.dst;
    a.stride_channel_x_bytes = args.stride_channel_x_bytes;
    a.stride_bias            = args.stride_bias;
    a.row_bytes              = (int) row_bytes;
    a.box_bytes              = p.box_bytes;
    a.nrows                  = (int) args.nrows_x;
    a.nb                     = (int) (args.ncols_x / ggml_blck_size(args.type));
    a.ntr                    = ntr;
    a.ntr_fd                 = init_fastdiv_values(ntr);
    a.nslots_fd              = init_fastdiv_values(p.nslots);
    a.n_used                 = (int) args.n_used;
    a.ntokens                = (int) args.ntokens;
    a.ids_stride             = (int) args.ids_stride;
    a.glu_op                 = (int) args.glu_op;
    a.glu_limit              = args.glu_limit;
    a.nslots                 = p.nslots;
    a.y_bytes                = (int) y_bytes;
    a.y_f32                  = args.y_f32;
    a.y_f32_s1               = (int) args.y_f32_s1;
    a.y_f32_s2               = (int) args.y_f32_s2;
    a.tile_ctr               = args.tile_ctr;
    a.ids_ready              = args.ids_ready ? 1 : 0;
    a.l2_issue_stop          = args.l2_issue_stop;
    for (int pair = 0; pair < args.ntokens*args.n_used; ++pair) {
        const int64_t t    = pair / args.n_used;
        const int64_t slot = pair % args.n_used;
        const int64_t yo   = (slot % args.nchannels_y)*args.stride_channel_y + t*args.stride_col_y;
        GGML_ASSERT(yo <= INT_MAX);
        a.pair_y[pair]   = (int) yo;
        a.pair_dst[pair] = t*args.stride_col_dst + slot*args.stride_channel_dst;
    }

    // a launch of several tokens decodes each fragment once for an expert's pairs where the shape has that instance
    // (mmvq_moe_pairs_fits) and its pairs are at least as many as the experts. Decoding once pays for an expert that
    // meets several pairs, and costs its instance's bookkeeping on one that meets one: where 3 tokens route to 8 of 288
    // experts (GLM-5.3-Flash) nearly every expert a launch reads meets one pair, and that instance ran the gate/up 1.2 %
    // and the down 2.1 % slower than the one-pair one (a 4-layer proxy with its 288 experts, random routing, RTX 5080),
    // and GLM-5.3-Flash at 3 and 4 tokens 2.4 and 1.4 % slower end to end (2 RTX PRO 6000, -sm tensor); where each of 3
    // tokens routes to all 8 of 8 (the 44-layer proxy) its gate/up is 27 % faster. GGML_CUDA_MMVQ_MOE_PAIRS_LEGACY=1:
    // each pair decodes it, whatever the counts
    static const bool pairs_legacy = ggml_env_switch("GGML_CUDA_MMVQ_MOE_PAIRS_LEGACY");
    const bool        pairs_once   = !pairs_legacy && args.ntokens > 1 && args.ntokens*args.n_used >= args.n_experts;

    // one block an SM, never more than the tiles of the most distinct experts the pairs can name
    const int nsm     = ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;
    const int nblocks = (int) std::min<int64_t>(nsm, args.ntokens*args.n_used*ntr);

    switch (args.type) {
        case GGML_TYPE_IQ3_XXS: mmvq_moe_launch_type<GGML_TYPE_IQ3_XXS>(a, nmat, p.rpw, pairs_once, nblocks, stream); break;
        case GGML_TYPE_Q8_0:    mmvq_moe_launch_type<GGML_TYPE_Q8_0>   (a, nmat, p.rpw, pairs_once, nblocks, stream); break;
        default: GGML_ABORT("%s: no instance for %s", __func__, ggml_type_name(args.type));
    }
}
