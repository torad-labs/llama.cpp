// The KQ mask taken from the sequences' bits (llama_kv_cells::seq_cells64 and pos_max64) against the scans it replaces.
//
// Random cell states the way a server makes them: sequences in one unified pool or one stream each, prompts appended,
// drafts rolled back and their cells taken again, conversations ended and their cells reused from position 0, prefixes
// shared by two sequences, cells removed, kept, saved and restored (llama_kv_cache::prepare), positions shifted down with
// the shift applied or still pending. On each state: the cells' bits and bounds against their definition, then the mask
// of a decode or verify ubatch built three ways - cell by cell (LLAMA_KQ_MASK_SCAN_LEGACY), 16 cells at a time
// (LLAMA_KQ_MASK_SEQ_BITS_LEGACY) and 64 at a time from the bits - compared byte for byte, bit-packed and f16, causal
// and not, with and without a window, with and without 2D positions.

#include "../src/models/inkling2-attention.h"
#include "../src/llama-hparams.h"
#include "../src/llama-host-threads.h"
#include "../src/llama-kv-cache.h"
#include "../src/llama-kv-cells.h"

#include "ggml.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

static int n_fail = 0;

#define CHECK(cond, ...) do { if (!(cond)) { ++n_fail; if (n_fail <= 20) { fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } } } while (0)

struct pool {
    std::vector<llama_kv_cells> v_cells;       // one per stream
    std::vector<uint32_t>       seq_to_stream;
    std::vector<uint32_t>       head;          // per stream
    std::vector<llama_pos>      p_next;        // per sequence
    bool                        pos_2d;
    std::mt19937 &              rng;

    pool(uint32_t size, uint32_t n_seq, bool unified, bool pos_2d, std::mt19937 & rng) : pos_2d(pos_2d), rng(rng) {
        const uint32_t n_stream = unified ? 1 : n_seq;
        v_cells.resize(n_stream);
        for (auto & cells : v_cells) {
            cells.resize(size);
        }
        head.assign(n_stream, 0);
        p_next.assign(n_seq, 0);
        for (uint32_t s = 0; s < n_seq; ++s) {
            seq_to_stream.push_back(unified ? 0 : s);
        }
    }

    llama_kv_cells & cells_of(llama_seq_id s) { return v_cells[seq_to_stream[s]]; }

    // the next empty cell from the stream's head on, wrapping (llama_kv_cache::find_slot, not contiguous)
    int64_t find_empty(llama_seq_id s) {
        auto & cells = cells_of(s);
        uint32_t & h = head[seq_to_stream[s]];
        for (uint32_t n = 0; n < cells.size(); ++n) {
            const uint32_t i = (h + n) % cells.size();
            if (cells.is_empty(i)) {
                h = (i + 1) % cells.size();
                return i;
            }
        }
        return -1;
    }

    std::vector<uint32_t> append(llama_seq_id s, uint32_t n) {
        std::vector<uint32_t> placed;
        auto & cells = cells_of(s);
        for (uint32_t k = 0; k < n; ++k) {
            const int64_t i = find_empty(s);
            if (i < 0) {
                break;
            }
            cells.pos_set(i, p_next[s]);
            if (pos_2d) {
                cells.ext_set(i, { (llama_pos) (rng() % 4), (llama_pos) (rng() % 4) });
            }
            cells.seq_add(i, s);
            placed.push_back(i);
            ++p_next[s];
        }
        return placed;
    }

    // drop the sequence's positions from p0 on (a rejected draft: llama_kv_cache::seq_rm)
    void rollback(llama_seq_id s, llama_pos p0) {
        auto & cells = cells_of(s);
        uint32_t new_head = cells.size();
        for (uint32_t i = 0; i < cells.size(); ++i) {
            if (cells.pos_in(i, p0, INT32_MAX) && cells.seq_has(i, s) && cells.seq_rm(i, s)) {
                new_head = std::min(new_head, i);
            }
        }
        uint32_t & h = head[seq_to_stream[s]];
        if (new_head < h) {
            h = new_head;
        }
        p_next[s] = cells.seq_pos_max(s) + 1;
    }

    void random_op(uint32_t n_seq) {
        const llama_seq_id s = rng() % n_seq;
        auto & cells = cells_of(s);
        switch (rng() % 12) {
            case 0: case 1: case 2: // a prompt chunk or a few decoded tokens
                append(s, 1 + rng() % (rng() % 4 == 0 ? 300 : 6));
                break;
            case 3: case 4: // a verify's rejected draft
                if (p_next[s] > 0) {
                    rollback(s, std::max<llama_pos>(0, p_next[s] - 1 - (llama_pos) (rng() % 4)));
                }
                break;
            case 5: // the conversation ends; the next one starts at 0 in whatever cells are free
                if (rng() % 3 == 0) {
                    rollback(s, 0);
                }
                break;
            case 6: { // a second sequence of the same stream takes this one's prefix (a cached prompt: seq_cp)
                const llama_seq_id t = rng() % n_seq;
                if (t != s && seq_to_stream[t] == seq_to_stream[s] && cells.seq_pos_max(t) < 0 && p_next[s] > 1) {
                    const llama_pos p1 = 1 + rng() % p_next[s];
                    for (uint32_t i = 0; i < cells.size(); ++i) {
                        if (cells.pos_in(i, 0, p1) && cells.seq_has(i, s)) {
                            cells.seq_add(i, t);
                        }
                    }
                    p_next[t] = cells.seq_pos_max(t) + 1;
                }
            } break;
            case 7: { // one cell removed whatever holds it (seq_rm of every sequence)
                const uint32_t i = rng() % cells.size();
                if (!cells.is_empty(i)) {
                    cells.rm(i);
                    for (uint32_t t = 0; t < n_seq; ++t) {
                        if (seq_to_stream[t] == seq_to_stream[s]) {
                            p_next[t] = cells.seq_pos_max(t) + 1;
                        }
                    }
                }
            } break;
            case 8: { // prepare(): cells saved, a ubatch placed, the cells restored (never with a shift pending)
                if (cells.get_has_shift()) {
                    break;
                }
                std::vector<uint32_t> idxs;
                if (rng() % 2) { // a whole group of 64: emptied below, only the restore can bound it again
                    const uint32_t g0 = 64*(rng() % ((cells.size() + 63)/64));
                    for (uint32_t i = g0; i < std::min<uint32_t>(cells.size(), g0 + 64); ++i) {
                        idxs.push_back(i);
                    }
                } else {
                    const uint32_t n = 1 + rng() % 8;
                    for (uint32_t k = 0; k < n; ++k) {
                        idxs.push_back(rng() % cells.size());
                    }
                    std::sort(idxs.begin(), idxs.end());
                    idxs.erase(std::unique(idxs.begin(), idxs.end()), idxs.end());
                }
                const llama_kv_cells saved = cells.cp(idxs);
                const bool place = rng() % 2; // else the cells are only emptied: the restore alone must bound them again
                for (const uint32_t i : idxs) {
                    if (!cells.is_empty(i)) {
                        cells.rm(i);
                    }
                    if (place) {
                        cells.pos_set(i, 100000 + (llama_pos) (rng() % 1000));
                        cells.seq_add(i, s);
                    }
                }
                cells.set(idxs, saved);
            } break;
            case 9: { // positions shifted down (a context shift) or up (a reused prompt chunk), applied (reset_shift) or not yet
                const llama_pos d = (rng() % 2 ? 1 : -1) * (llama_pos) (1 + rng() % 64);
                const llama_pos p0 = p_next[s] > 0 ? rng() % p_next[s] : 0;
                for (uint32_t i = 0; i < cells.size(); ++i) {
                    if (cells.pos_in(i, p0, INT32_MAX) && cells.seq_has(i, s) && cells.seq_count(i) == 1) {
                        cells.pos_add(i, d);
                    }
                }
                if (rng() % 2) {
                    cells.reset_shift();
                }
                p_next[s] = cells.seq_pos_max(s) + 1;
            } break;
            case 10: { // seq_keep of one sequence
                if (rng() % 8 == 0) {
                    for (uint32_t i = 0; i < cells.size(); ++i) {
                        cells.seq_keep(i, s);
                    }
                    for (uint32_t t = 0; t < n_seq; ++t) {
                        if (seq_to_stream[t] == seq_to_stream[s]) {
                            p_next[t] = cells.seq_pos_max(t) + 1;
                        }
                    }
                }
            } break;
            case 11: // positions divided (self-extend), pending
                if (rng() % 8 == 0) {
                    for (uint32_t i = 0; i < cells.size(); ++i) {
                        if (!cells.is_empty(i) && cells.seq_has(i, s) && cells.seq_count(i) == 1) {
                            cells.pos_div(i, 2);
                        }
                    }
                    p_next[s] = cells.seq_pos_max(s) + 1;
                }
                break;
        }
    }
};

// the bits and bounds against their definition
static void check_cells(const llama_kv_cells & cells, uint32_t n_seq) {
    for (uint32_t g = 0; g*64 < cells.size(); ++g) {
        llama_pos pmax = -1;
        for (uint32_t i = g*64; i < std::min<uint32_t>(cells.size(), g*64 + 64); ++i) {
            if (!cells.is_empty(i)) {
                pmax = std::max(pmax, cells.pos_get(i));
            }
        }
        CHECK(cells.pos_max64(g) >= pmax, "group %u: bound %d below position %d", g, cells.pos_max64(g), pmax);
        CHECK(pmax >= 0 || cells.pos_max64(g) == -1, "group %u: empty, bound %d", g, cells.pos_max64(g));
        for (uint32_t s = 0; s < n_seq; ++s) {
            uint64_t want = 0;
            for (uint32_t i = g*64; i < std::min<uint32_t>(cells.size(), g*64 + 64); ++i) {
                if (!cells.is_empty(i) && cells.seq_has(i, s)) {
                    want |= (uint64_t) 1 << (i - g*64);
                }
            }
            CHECK(cells.seq_cells64(s, g) == want, "group %u seq %u: bits %016llx, cells %016llx", g, s,
                    (unsigned long long) cells.seq_cells64(s, g), (unsigned long long) want);
        }
    }
}

static void check_inkling2_decode_masks() {
    using mode = inkling2_decode_mask;
    std::vector<uint32_t> cells(23, 8192);
    // as LLAMA_INKLING2_DECODE_CLASS_LEGACY=1 serves them, from 8 rows
    const inkling2_mask_shapes served = {32, 8, 32, 8192, 8, 23, 262144, 6, 3};
    auto shapes = served;
    const auto classify = [&](uint32_t pool, bool mean) {
        return inkling2_classify_decode_mask(pool, cells.size(), mean, [&](uint32_t i) { return cells[i]; }, shapes);
    };
    CHECK(classify(262144, false) == mode::bounded_rows, "private rows must preserve the old bound");
    cells[0] = 262144;
    CHECK(classify(327680, false) == mode::dense, "opt-out must keep a long reader dense");
    CHECK(classify(327680, true) == mode::low_mean, "one long reader has low mean visibility");
    cells.assign(23, 262144);
    CHECK(classify(1703936, true) == mode::low_mean, "23 sessions sharing 196608 prefix cells");
    CHECK(classify(262144, true) == mode::dense, "a fully shared pool is dense");
    cells.assign(23, 65536);
    CHECK(classify(393216, true) == mode::low_mean, "at one sixth mean visibility");
    ++cells.back();
    CHECK(classify(393216, true) == mode::dense, "crossing the mean threshold changes the reusable graph class");
    cells.pop_back();
    CHECK(classify(393216, true) == mode::dense, "new path is qualified for 23 query rows");
    cells.assign(23, 262145);
    CHECK(classify(2097152, true) == mode::low_mean, "one cell past the old max is a distinct hint");
    cells.assign(23, 262144);
    CHECK(classify(2097152, true) == mode::bounded_rows, "the original bounded mode takes precedence");

    // each shape a search can move (tunables.toml) moves its own bound
    cells.assign(23, 16385);
    CHECK(classify(262144, false) == mode::bounded_rows, "a row past a sixteenth of the pool is within an eighth");
    shapes.pool_share = 16;
    CHECK(classify(262144, false) == mode::dense, "a share of 16 bounds a row at a sixteenth of the pool");
    cells.assign(23, 16384);
    CHECK(classify(262144, false) == mode::bounded_rows, "a row at the share's bound keeps the class");
    shapes = served;
    shapes.rows_min = 24;
    CHECK(classify(262144, false) == mode::dense, "a step under the fewest rows is dense");
    shapes = served;
    shapes.rows_max = 22;
    CHECK(classify(262144, false) == mode::dense, "a step over the most rows is dense");
    shapes = served;
    shapes.pool_min = 524288;
    CHECK(classify(262144, false) == mode::dense, "a pool under the smallest is not bounded");
    cells.assign(23, 8192);
    cells[0] = 262144;
    CHECK(classify(327680, true) == mode::low_mean, "the smallest bounded pool leaves the low-mean class alone");
    shapes = served;
    cells.assign(23, 65536);
    shapes.mean_pool_share = 7;
    CHECK(classify(393216, true) == mode::dense, "a mean share of 7 turns away rows at a sixth");
    shapes.mean_pool_share = 4;
    ++cells.back();
    CHECK(classify(393216, true) == mode::low_mean, "a mean share of 4 takes rows past a sixth");
    shapes = served;
    cells.assign(16, 8192);
    cells[0] = 131072;
    CHECK(classify(196608, true) == mode::dense, "a 16-row step is dense while the low-mean class takes 23 rows");
    shapes.mean_rows = 16;
    CHECK(classify(196608, true) == mode::dense, "and while its pool is under the low-mean class's smallest");
    shapes.mean_pool_min = 131072;
    CHECK(classify(196608, true) == mode::low_mean, "16 rows in a pool the low-mean class takes are low mean");

    llama_kv_cells shared;
    shared.resize(32769);
    for (uint32_t i = 0; i < shared.size(); ++i) {
        shared.pos_set(i, i);
        shared.seq_add(i, 0);
    }
    shared.seq_add(32768, 1);
    CHECK(shared.seq_n_cells(0) == 32769 && shared.seq_n_cells(1) == 1, "shared cell contributes to both sequence counts");
    const auto shared_mode = [&]() {
        return inkling2_classify_decode_mask(262144, 23, true, [&](uint32_t s) { return shared.seq_n_cells(s); }, served);
    };
    CHECK(shared_mode() == mode::low_mean, "shared reader exceeds the per-row bound");
    shared.seq_rm(32768, 0);
    CHECK(shared_mode() == mode::bounded_rows, "removing one membership restores the per-row bound");
}

// the default class: any row count the decode kernels take, any share a row holds, until the rows read pool_reads pools
static void check_inkling2_decode_rows() {
    using mode = inkling2_decode_mask;
    const inkling2_mask_shapes shapes = {32, 1, 32, 8192, 8, 23, 262144, 6, 3};
    std::vector<uint32_t> cells;
    const auto classify = [&](uint32_t pool, bool mean) {
        return inkling2_classify_decode_rows(pool, cells.size(), mean, [&](uint32_t i) { return cells[i]; }, shapes);
    };
    // G7: one session of 199,967 cells among 23 in a pool of 860,160, of which 1, 4 or 23 decode
    for (size_t rows : {1, 4, 7, 8, 23}) {
        cells.assign(rows, 28400);
        CHECK(classify(860160, true) == mode::bounded_rows, "rows within an eighth of the pool are bounded at any count");
        cells[0] = 199967;
        CHECK(classify(860160, true) == mode::low_mean, "one long session reads its own cells at any row count");
        CHECK(classify(860160, false) == mode::dense, "uncounted rows past an eighth stay dense");
        cells[0] = 860000;
        CHECK(classify(860160, true) == mode::low_mean, "a session holding most of the pool reads only its cells");
    }
    cells.assign(1, 8192);
    CHECK(classify(8160, true) == mode::dense, "a pool under the decode kernels' smallest is dense");
    cells.assign(33, 1);
    CHECK(classify(860160, true) == mode::dense, "a step past the most rows is dense");
    cells.clear();
    CHECK(classify(860160, true) == mode::dense, "a step with no rows is dense");
    // 23 sessions sharing a long prefix: past three pools together the dense kernel's one read a query head costs less
    cells.assign(23, 30000);
    CHECK(classify(230000, true) == mode::low_mean, "the rows read three pools together");
    ++cells.back();
    CHECK(classify(230000, true) == mode::dense, "one cell more is dense");
}

// a ubatch of the given (sequence, position) rows, in that order
struct ubatch_builder {
    std::vector<llama_pos>      pos;
    std::vector<llama_seq_id>   seq;
    std::vector<llama_seq_id *> seq_ptr;
    std::vector<int32_t>        n_seq_id;
    std::vector<llama_pos>      pos_all;
    llama_ubatch                ub = {};

    void add(llama_seq_id s, llama_pos p) {
        pos.push_back(p);
        seq.push_back(s);
    }

    const llama_ubatch & get() {
        const uint32_t n = pos.size();

        seq_ptr.resize(n);
        n_seq_id.assign(n, 1);
        pos_all = pos;

        for (uint32_t i = 0; i < n; ++i) {
            seq_ptr[i] = &seq[i];
        }

        ub.n_tokens = n;
        ub.n_pos    = 1;
        ub.pos      = pos_all.data();
        ub.n_seq_id = n_seq_id.data();
        ub.seq_id   = seq_ptr.data();

        return ub;
    }
};

struct mask_counts {
    int n_masks = 0;
    int n_chunk = 0;
};

// the ubatch's mask built four ways - cell by cell, 16 cells at a time, 64 at a time from the bits, and the chunk rows -
// compared byte for byte; expect_chunk: whether the chunk path must build this layout
static void check_masks_equal(const pool & p, const llama_ubatch & ub, uint32_t n_kv, uint32_t n_stream, ggml_type type,
        bool causal, uint32_t n_swa, llama_swa_type swa_type, const int32_t * rows, bool block_uniform,
        bool expect_chunk, const char * what, mask_counts & counts,
        llama_non_causal_type nct = LLAMA_NON_CAUSAL_TYPE_ALL) {
    llama_hparams hparams{};
    hparams.non_causal_type = nct;

    ggml_init_params ip = { /*.mem_size =*/ 64*1024*1024, /*.mem_buffer =*/ nullptr, /*.no_alloc =*/ false };

    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * masks[4];
    for (int m = 0; m < 4; ++m) {
        masks[m] = ggml_new_tensor_4d(ctx, type, type == GGML_TYPE_I16 ? n_kv/16 : n_kv, ub.n_tokens/n_stream, 1, n_stream);
        memset(masks[m]->data, 0xa5, ggml_nbytes(masks[m])); // a cell no path writes shows as a difference
    }

    llama_kv_cache_set_input_kq_mask(masks[0], &ub, causal, hparams, p.v_cells, p.seq_to_stream, n_swa, swa_type, true,  false, rows, block_uniform, false);
    llama_kv_cache_set_input_kq_mask(masks[1], &ub, causal, hparams, p.v_cells, p.seq_to_stream, n_swa, swa_type, false, false, rows, block_uniform, false);
    llama_kv_cache_set_input_kq_mask(masks[2], &ub, causal, hparams, p.v_cells, p.seq_to_stream, n_swa, swa_type, false, true,  rows, block_uniform, false);
    const bool chunk = llama_kv_cache_set_input_kq_mask(masks[3], &ub, causal, hparams, p.v_cells, p.seq_to_stream, n_swa, swa_type, false, true, rows, block_uniform, true);

    CHECK(chunk == expect_chunk, "%s: the chunk rows %s the rows, expected they %s them", what,
            chunk ? "built" : "left", expect_chunk ? "build" : "leave");

    const size_t nb = ggml_nbytes(masks[0]);
    CHECK(memcmp(masks[0]->data, masks[1]->data, nb) == 0, "%s: the 16-cell scan differs from the cell scan", what);
    CHECK(memcmp(masks[0]->data, masks[2]->data, nb) == 0, "%s: the scans from the bits differ from the cell scan", what);
    CHECK(memcmp(masks[0]->data, masks[3]->data, nb) == 0, "%s: the chunk rows differ from the cell scan", what);

    counts.n_masks += 4;
    counts.n_chunk += chunk;

    ggml_free(ctx);
}

// The two layouts the chunk rows must leave to the scans: the cell rows an attention reads through an index list
// (llm_attn_mask::rows, the kv_rows attention) and the rows a block of draft tokens shares (block_uniform, dspark)
static void check_rows_and_blocks(std::mt19937 & rng, mask_counts & counts) {
    for (int round = 0; round < 40; ++round) {
        const uint32_t size  = std::vector<uint32_t>{ 2048, 4096, 16384 }[rng() % 3];
        const uint32_t n_seq = 2 + rng() % 3;

        pool p(size, n_seq, /*unified =*/ true, /*pos_2d =*/ false, rng);

        for (int k = 0; k < 10; ++k) {
            p.random_op(n_seq);
        }

        // the ubatch: n_tps tokens of each sequence, placed first, at the tip
        const uint32_t n_tps = 1 + rng() % 4;

        ubatch_builder ub;
        bool placed_all = true;
        for (llama_seq_id s = 0; s < (llama_seq_id) n_seq && placed_all; ++s) {
            const llama_pos p0 = p.p_next[s];
            placed_all = p.append(s, n_tps).size() == n_tps;
            for (uint32_t k = 0; k < n_tps; ++k) {
                ub.add(s, p0 + k);
            }
        }
        if (!placed_all) {
            continue; // the pool is full
        }

        uint32_t used_max = 0;
        for (const auto & cells : p.v_cells) {
            used_max = std::max(used_max, cells.used_max_p1());
        }
        const uint32_t n_kv = std::min<uint32_t>(size, std::max(256u, GGML_PAD(used_max, 256)));

        char what[160];

        // the shared block rows: every token of a sequence reads the first token's row
        for (const ggml_type type : { GGML_TYPE_I16, GGML_TYPE_F16 }) {
            for (const bool causal : { true, false }) {
                snprintf(what, sizeof(what), "round %d shared rows (%s, %s)", round, ggml_type_name(type), causal ? "causal" : "non-causal");
                check_masks_equal(p, ub.get(), n_kv, 1, type, causal, 0,   LLAMA_SWA_TYPE_NONE,     nullptr, true, false, what, counts);
                check_masks_equal(p, ub.get(), n_kv, 1, type, causal, 128, LLAMA_SWA_TYPE_STANDARD, nullptr, true, false, what, counts);
            }
        }

        // the caller's causal flag flipped by the model's non-causal type: a dense layer's caller flag leaves a windowed
        // cache causal (SWA_ONLY) or a windowed layer's leaves a dense cache causal (DENSE_ONLY), and both builders follow.
        // Either way the effective attention is causal here, so only a one-row ubatch at the tip takes the tip path
        for (const bool causal : { true, false }) {
            snprintf(what, sizeof(what), "round %d causal flip (%s)", round, causal ? "causal" : "non-causal");
            check_masks_equal(p, ub.get(), n_kv, 1, GGML_TYPE_I16, causal, 0,  LLAMA_SWA_TYPE_NONE,     nullptr, false,
                    n_tps > 1, what, counts, LLAMA_NON_CAUSAL_TYPE_SWA_ONLY);
            check_masks_equal(p, ub.get(), n_kv, 1, GGML_TYPE_I16, causal, 64, LLAMA_SWA_TYPE_STANDARD, nullptr, false,
                    n_tps > 1, what, counts, LLAMA_NON_CAUSAL_TYPE_DENSE_ONLY);
        }

        // the cell rows: a slice a sequence, its cells in cell order (F16, as the kv_rows attention reads them)
        uint32_t n_rows = 0;
        for (llama_seq_id s = 0; s < (llama_seq_id) n_seq; ++s) {
            n_rows = std::max(n_rows, p.v_cells[0].seq_n_cells(s));
        }
        n_rows = std::max<uint32_t>(16u, GGML_PAD(n_rows, 16));

        std::vector<int32_t> row_data((size_t) n_rows*n_seq, -1);
        std::vector<llama_kv_rows_slice> slices(n_seq);
        for (llama_seq_id s = 0; s < (llama_seq_id) n_seq; ++s) {
            slices[s].seq_ids = { s };
        }
        p.v_cells[0].get_seq_cells(slices, row_data.data(), n_rows);

        for (const bool block_uniform : { false, true }) {
            for (const bool causal : { true, false }) {
                snprintf(what, sizeof(what), "round %d cell rows (%s%s)", round, causal ? "causal" : "non-causal",
                        block_uniform ? ", shared rows" : "");
                check_masks_equal(p, ub.get(), n_rows, n_seq, GGML_TYPE_F16, causal, 0,  LLAMA_SWA_TYPE_NONE,     row_data.data(), block_uniform, false, what, counts);
                check_masks_equal(p, ub.get(), n_rows, n_seq, GGML_TYPE_F16, causal, 64, LLAMA_SWA_TYPE_STANDARD, row_data.data(), block_uniform, false, what, counts);
            }
        }

        // one cell row over every sequence (an interleaved ubatch that cannot split): the union of their cells, each once
        {
            const uint32_t n_rows_all = std::max<uint32_t>(16u, GGML_PAD(used_max, 16));

            std::vector<int32_t> row_all((size_t) n_rows_all, -1);
            std::vector<llama_kv_rows_slice> slice_all(1);
            for (llama_seq_id s = 0; s < (llama_seq_id) n_seq; ++s) {
                slice_all[0].seq_ids.push_back(s);
            }
            p.v_cells[0].get_seq_cells(slice_all, row_all.data(), n_rows_all);

            for (const bool causal : { true, false }) {
                snprintf(what, sizeof(what), "round %d one cell-row slice (%s)", round, causal ? "causal" : "non-causal");
                check_masks_equal(p, ub.get(), n_rows_all, 1, GGML_TYPE_F16, causal, 0,  LLAMA_SWA_TYPE_NONE,     row_all.data(), false, false, what, counts);
                check_masks_equal(p, ub.get(), n_rows_all, 1, GGML_TYPE_F16, causal, 64, LLAMA_SWA_TYPE_STANDARD, row_all.data(), false, false, what, counts);
            }
        }
    }
}

// A pool at the served prefill's depth (reports/inkling2-prefill-roofline.md (a)): 1.65M cells, a second sequence sharing
// half of the first one's prefix, prompt chunks mid-sequence with cells of their own sequence above them, rows inside the
// shared prefix, an empty sequence, and the same chunk built at one thread and at four
static void check_large_layouts(std::mt19937 & rng, mask_counts & counts) {
    const uint32_t size = 1703936;

    pool p(size, 4, /*unified =*/ true, /*pos_2d =*/ false, rng); // sequences 0-2 hold cells; sequence 3 stays empty

    p.append(0, 1000000);

    // sequence 1 takes the first half of sequence 0's prefix (a cached prompt), then has its own tail
    for (uint32_t i = 0; i < p.v_cells[0].size(); ++i) {
        if (p.v_cells[0].pos_in(i, 0, 500000) && p.v_cells[0].seq_has(i, 0)) {
            p.v_cells[0].seq_add(i, 1);
        }
    }
    p.p_next[1] = p.cells_of(1).seq_pos_max(1) + 1;

    p.append(1, 150000);
    p.append(2, 550000); // 1,700,000 cells: the served phase's widest pool (reports/inkling2-prefill-roofline.md)

    check_cells(p.v_cells[0], 4);

    uint32_t used_max = 0;
    for (const auto & cells : p.v_cells) {
        used_max = std::max(used_max, cells.used_max_p1());
    }
    uint32_t n_kv = std::min<uint32_t>(size, std::max(256u, GGML_PAD(used_max, 256)));
    n_kv -= n_kv % 16;

    char what[160];

    // sequence 0's chunk mid-sequence (300,000 of its cells above it), the tips of the other two in the same ubatch
    ubatch_builder chunk;
    for (uint32_t k = 0; k < 16; ++k) {
        chunk.add(0, 700000 + k);
        chunk.add(1, 650000 - 16 + k);
        chunk.add(2, 550000 - 16 + k);
    }

    for (const llama_swa_type swa : { LLAMA_SWA_TYPE_NONE, LLAMA_SWA_TYPE_STANDARD }) {
        snprintf(what, sizeof(what), "large chunk (%s)", swa == LLAMA_SWA_TYPE_NONE ? "no window" : "window");
        check_masks_equal(p, chunk.get(), n_kv, 1, GGML_TYPE_I16, true,
                swa == LLAMA_SWA_TYPE_STANDARD ? 2048 : 0, swa, nullptr, false, true, what, counts);
    }

    // rows inside the shared prefix: both sequences see the same cells, each with cells above its own position
    ubatch_builder shared;
    for (uint32_t k = 0; k < 3; ++k) {
        shared.add(0, 100000 + k);
        shared.add(1, 100000 + k);
    }

    snprintf(what, sizeof(what), "large shared prefix");
    check_masks_equal(p, shared.get(), n_kv, 1, GGML_TYPE_I16, true, 0, LLAMA_SWA_TYPE_NONE, nullptr, false, true, what, counts);

    // an empty sequence: no cells of its own, its row is all zero
    ubatch_builder empty;
    empty.add(0, 5);
    empty.add(3, 0);

    snprintf(what, sizeof(what), "large empty sequence");
    check_masks_equal(p, empty.get(), n_kv, 1, GGML_TYPE_I16, true, 0, LLAMA_SWA_TYPE_NONE, nullptr, false, true, what, counts);

    // the same chunk at one thread and at four: every byte the same output, and the build really splits
    {
        llama_hparams hparams{};

        const llama_ubatch & cub = chunk.get();

        ggml_init_params ip = { /*.mem_size =*/ 64*1024*1024, /*.mem_buffer =*/ nullptr, /*.no_alloc =*/ false };
        ggml_context * ctx = ggml_init(ip);

        ggml_tensor * one   = ggml_new_tensor_4d(ctx, GGML_TYPE_I16, n_kv/16, cub.n_tokens, 1, 1);
        ggml_tensor * split = ggml_new_tensor_4d(ctx, GGML_TYPE_I16, n_kv/16, cub.n_tokens, 1, 1);
        memset(one->data,   0xa5, ggml_nbytes(one));
        memset(split->data, 0xa5, ggml_nbytes(split));

        llama_host_threads_set(1);
        const bool built_one = llama_kv_cache_set_input_kq_mask(one, &cub, true, hparams, p.v_cells, p.seq_to_stream, 2048, LLAMA_SWA_TYPE_STANDARD, false, true, nullptr, false, true);
        const int used_one = llama_host_threads_last();
        llama_host_threads_set(4);
        const bool built_split = llama_kv_cache_set_input_kq_mask(split, &cub, true, hparams, p.v_cells, p.seq_to_stream, 2048, LLAMA_SWA_TYPE_STANDARD, false, true, nullptr, false, true);
        const int used_split = llama_host_threads_last();
        llama_host_threads_set(0);

        CHECK(built_one && built_split && used_one == 1 && used_split == 4, "large chunk: the chunk rows at 1 and 4 threads (%d, %d), split across %d and %d",
                (int) built_one, (int) built_split, used_one, used_split);
        CHECK(memcmp(one->data, split->data, ggml_nbytes(one)) == 0, "large chunk: the chunk rows at 4 threads differ from one thread");

        counts.n_masks += 2;
        counts.n_chunk += built_one + built_split;

        ggml_free(ctx);
    }
}

// The builder's host time for the served prefill shape: n_seq sessions of n_tps rows a pass over a pool of n_pref+n_tps
// cells a session (reports/inkling2-prefill-roofline.md (a): 8 x 128 rows, 983,040 cells), the scans as served and the
// chunk rows, one thread and the host threads
static int bench_builder(int argc, char ** argv) {
    const uint32_t n_seq  = argc > 2 ? (uint32_t) atoi(argv[2]) : 8;
    const uint32_t n_tps  = argc > 3 ? (uint32_t) atoi(argv[3]) : 128;
    const uint32_t n_pref = argc > 4 ? (uint32_t) atoi(argv[4]) : 122752;
    const uint32_t n_swa  = argc > 5 ? (uint32_t) atoi(argv[5]) : 0;

    std::mt19937 rng(20261122);

    const uint32_t size = n_seq*(n_pref + n_tps) + 256;

    pool p(size, n_seq, /*unified =*/ true, /*pos_2d =*/ false, rng);

    ubatch_builder ub;
    for (llama_seq_id s = 0; s < (llama_seq_id) n_seq; ++s) {
        p.append(s, n_pref);
        const llama_pos p0 = p.p_next[s];
        p.append(s, n_tps);
        for (uint32_t k = 0; k < n_tps; ++k) {
            ub.add(s, p0 + k);
        }
    }

    uint32_t used_max = 0;
    for (const auto & cells : p.v_cells) {
        used_max = std::max(used_max, cells.used_max_p1());
    }
    const uint32_t n_kv = std::min<uint32_t>(size, std::max(256u, GGML_PAD(used_max, 256)));

    llama_hparams hparams{};

    const size_t mask_bytes = (size_t) n_kv/16*(n_seq*n_tps)*sizeof(uint16_t);

    ggml_init_params ip = { /*.mem_size =*/ mask_bytes + 64*1024*1024, /*.mem_buffer =*/ nullptr, /*.no_alloc =*/ false };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * mask = ggml_new_tensor_4d(ctx, GGML_TYPE_I16, n_kv/16, n_seq*n_tps, 1, 1);

    const llama_swa_type swa_type = n_swa > 0 ? LLAMA_SWA_TYPE_STANDARD : LLAMA_SWA_TYPE_NONE;

    // warm both, and check the chunk rows against the scans at this size
    llama_kv_cache_set_input_kq_mask(mask, &ub.get(), true, hparams, p.v_cells, p.seq_to_stream, n_swa, swa_type, false, true, nullptr, false, false);
    const std::vector<uint16_t> scanned((const uint16_t *) mask->data, (const uint16_t *) mask->data + ggml_nelements(mask));
    llama_kv_cache_set_input_kq_mask(mask, &ub.get(), true, hparams, p.v_cells, p.seq_to_stream, n_swa, swa_type, false, true, nullptr, false, true);
    CHECK(memcmp(scanned.data(), mask->data, mask_bytes) == 0, "bench: the chunk rows differ from the scans at %u rows and %u cells", n_seq*n_tps, n_kv);

    const auto time_it = [&](bool chunk, int threads) {
        llama_host_threads_set(threads);
        const int64_t t0 = ggml_time_us();
        llama_kv_cache_set_input_kq_mask(mask, &ub.get(), true, hparams, p.v_cells, p.seq_to_stream, n_swa, swa_type, false, true, nullptr, false, chunk);
        const int64_t t1 = ggml_time_us();
        llama_host_threads_set(0);
        return (t1 - t0)/1000.0;
    };

    for (const int threads : { 1, 0 }) {
        double scan_ms = 1e9, chunk_ms = 1e9;
        for (int r = 0; r < 3; ++r) {
            scan_ms  = std::min(scan_ms,  time_it(false, threads));
            chunk_ms = std::min(chunk_ms, time_it(true,  threads));
        }
        printf("bench: %u rows, %u cells, window %u, %s: scans %.1f ms, chunk rows %.1f ms (%.2fx)\n",
                n_seq*n_tps, n_kv, n_swa, threads == 1 ? "1 thread" : "host threads", scan_ms, chunk_ms, scan_ms/chunk_ms);
    }

    ggml_free(ctx);

    return 0;
}

int main(int argc, char ** argv) {
    if (argc > 1 && strcmp(argv[1], "--bench") == 0) {
        return bench_builder(argc, argv);
    }

    check_inkling2_decode_masks();
    check_inkling2_decode_rows();
    std::mt19937 rng(20260925);

    ggml_init_params ip = { /*.mem_size =*/ 64*1024*1024, /*.mem_buffer =*/ nullptr, /*.no_alloc =*/ false };

    int n_masks = 0, n_whole = 0, n_chunk = 0;

    for (int round = 0; round < 1200; ++round) {
        const uint32_t size    = std::vector<uint32_t>{ 256, 1024, 4096, 5000 }[rng() % 4];
        const uint32_t n_seq   = 1 + rng() % 4;
        const bool     unified = n_seq == 1 || rng() % 3 != 0;
        const bool     pos_2d  = rng() % 3 == 0;

        pool p(size, n_seq, unified, pos_2d, rng);

        const int n_ops = rng() % 400;
        for (int k = 0; k < n_ops; ++k) {
            p.random_op(n_seq);
        }
        for (const auto & cells : p.v_cells) {
            check_cells(cells, n_seq);
        }

        // the ubatch: a decode (1 token), a verify (up to 5) or a prompt chunk (up to 64 tokens) of some sequences,
        // placed in the cells first as the cache does (apply_ubatch before the inputs are set); split streams take the
        // same count from each sequence. A prompt row is rarely at its sequence's tip: sometimes the sequence keeps its
        // own cells above the ubatch too (a prompt chunk under a restored tail, a rejected draft's cells)
        const uint32_t n_tps_seq = rng() % 4 == 0 ? 8 + rng() % 57 : rng() % 2 ? 1 : 2 + rng() % 4;
        std::vector<llama_seq_id> seqs;
        for (uint32_t s = 0; s < n_seq; ++s) {
            if (!unified || rng() % 2 == 0 || s == n_seq - 1) {
                seqs.push_back(s);
            }
        }
        // the tokens, grouped by sequence first: a real batch interleaves its sequences, and a sequence's own rows can
        // arrive in any order, so both are generated below
        std::vector<std::pair<llama_seq_id, llama_pos>> rows;
        bool placed_all = true;
        for (const llama_seq_id s : seqs) {
            const llama_pos p0 = p.p_next[s];
            // the cells of the batch, and sometimes cells of the same sequence above every row of it
            const uint32_t n_ahead = rng() % 4 == 0 ? 1 + rng() % 8 : 0;
            placed_all &= p.append(s, n_tps_seq + n_ahead).size() == n_tps_seq + n_ahead;
            for (uint32_t k = 0; k < n_tps_seq; ++k) {
                rows.emplace_back(s, p0 + (llama_pos) k);
            }
        }
        if (!placed_all) {
            continue; // the pool is full
        }

        // the rows of a sequence out of order, within its own block (a split stream's block is its token range)
        if (seqs.size() > 0 && rng() % 2 == 0) {
            for (size_t j = 0; j < seqs.size(); ++j) {
                std::shuffle(rows.begin() + j*n_tps_seq, rows.begin() + (j + 1)*n_tps_seq, rng);
            }
        }

        // sequences interleaved: a unified pool's rows belong to no stream, so their order is free
        if (unified && seqs.size() > 1 && rng() % 2 == 0) {
            std::vector<std::pair<llama_seq_id, llama_pos>> mixed;
            mixed.reserve(rows.size());
            for (uint32_t k = 0; k < n_tps_seq; ++k) {
                for (size_t j = 0; j < seqs.size(); ++j) {
                    mixed.push_back(rows[j*n_tps_seq + k]);
                }
            }
            rows = std::move(mixed);
        }

        std::vector<llama_pos>    pos;
        std::vector<llama_seq_id> seq_ids;
        for (const auto & r : rows) {
            seq_ids.push_back(r.first);
            pos.push_back(r.second);
        }
        const uint32_t n_tokens = pos.size();
        const uint32_t n_pos    = pos_2d ? 4 : 1;
        std::vector<llama_pos> pos_all(n_tokens*n_pos);
        for (uint32_t i = 0; i < n_tokens; ++i) {
            pos_all[i] = pos[i];
            for (uint32_t d = 1; d < n_pos; ++d) {
                pos_all[i + d*n_tokens] = rng() % 4;
            }
        }
        std::vector<llama_seq_id *> seq_id_ptr(n_tokens);
        std::vector<int32_t> n_seq_id(n_tokens, 1);
        for (uint32_t i = 0; i < n_tokens; ++i) {
            seq_id_ptr[i] = &seq_ids[i];
        }
        llama_ubatch ubatch = {};
        ubatch.n_tokens = n_tokens;
        ubatch.n_pos    = n_pos;
        ubatch.pos      = pos_all.data();
        ubatch.n_seq_id = n_seq_id.data();
        ubatch.seq_id   = seq_id_ptr.data();

        const uint32_t n_stream = unified ? 1 : n_seq;
        uint32_t used_max = 0;
        for (const auto & cells : p.v_cells) {
            used_max = std::max(used_max, cells.used_max_p1());
        }

        for (const ggml_type type : { GGML_TYPE_I16, GGML_TYPE_F16 }) {
            // n_kv: the cache's padded extent (llama_kv_cache::get_n_kv), or one ending inside a group of 64
            uint32_t n_kv = std::min(size, std::max(256u, GGML_PAD(used_max, 256)));
            if (rng() % 3 == 0) {
                n_kv = std::min<uint32_t>(size, GGML_PAD(used_max + 1 + (uint32_t) (rng() % 100), 16));
            }
            if (type == GGML_TYPE_I16) {
                n_kv -= n_kv % 16;
            }
            if (n_kv == 0) {
                continue;
            }

            for (const llama_swa_type swa_type : { LLAMA_SWA_TYPE_NONE, LLAMA_SWA_TYPE_STANDARD }) {
                for (const bool causal : { true, false }) {
                    const uint32_t n_swa = swa_type == LLAMA_SWA_TYPE_NONE ? 0 : 1 + rng() % 300;

                    // the caller's causal flag, flipped for a cache by the model's non-causal type like llama_set_causal_attn
                    llama_hparams hparams{};
                    hparams.non_causal_type = (llama_non_causal_type) (rng() % 4);

                    ggml_context * ctx = ggml_init(ip);
                    ggml_tensor * masks[4];
                    for (int m = 0; m < 4; ++m) {
                        masks[m] = ggml_new_tensor_4d(ctx, type, type == GGML_TYPE_I16 ? n_kv/16 : n_kv, n_tokens/n_stream, 1, n_stream);
                        memset(masks[m]->data, 0xa5, ggml_nbytes(masks[m])); // a cell no path writes shows as a difference
                    }

                    // cell by cell; 16 cells at a time; 64 at a time from the bits, all with chunk_rows off; then the
                    // chunk rows, a row from the row before it (the scans build the prompt chunks' rows, as before)
                    CHECK(!llama_kv_cache_set_input_kq_mask(masks[0], &ubatch, causal, hparams, p.v_cells, p.seq_to_stream, n_swa, swa_type, true,  false, nullptr, false, false),
                            "round %d: the cell scan reports the chunk rows", round);
                    CHECK(!llama_kv_cache_set_input_kq_mask(masks[1], &ubatch, causal, hparams, p.v_cells, p.seq_to_stream, n_swa, swa_type, false, false, nullptr, false, false),
                            "round %d: the 16-cell scan reports the chunk rows", round);
                    CHECK(!llama_kv_cache_set_input_kq_mask(masks[2], &ubatch, causal, hparams, p.v_cells, p.seq_to_stream, n_swa, swa_type, false, true,  nullptr, false, false),
                            "round %d: the bits scan reports the chunk rows", round);
                    const bool chunk = llama_kv_cache_set_input_kq_mask(masks[3], &ubatch, causal, hparams, p.v_cells, p.seq_to_stream, n_swa, swa_type, false, true, nullptr, false, true);

                    const size_t nb = ggml_nbytes(masks[0]);
                    CHECK(memcmp(masks[0]->data, masks[1]->data, nb) == 0, "round %d: 16-cell scan differs from the cell scan (%s, n_kv %u, %s, %s)",
                            round, ggml_type_name(type), n_kv, causal ? "causal" : "non-causal", swa_type == LLAMA_SWA_TYPE_NONE ? "no window" : "window");
                    CHECK(memcmp(masks[0]->data, masks[2]->data, nb) == 0, "round %d: the bits differ from the cell scan (%s, n_kv %u, %s, %s, %u seqs, %s, %u tokens)",
                            round, ggml_type_name(type), n_kv, causal ? "causal" : "non-causal", swa_type == LLAMA_SWA_TYPE_NONE ? "no window" : "window",
                            n_seq, unified ? "unified" : "split", n_tokens);
                    CHECK(memcmp(masks[0]->data, masks[3]->data, nb) == 0, "round %d: the chunk rows differ from the cell scan (%s, n_kv %u, %s, %s, %u seqs, %s, %u tokens, built %d)",
                            round, ggml_type_name(type), n_kv, causal ? "causal" : "non-causal", swa_type == LLAMA_SWA_TYPE_NONE ? "no window" : "window",
                            n_seq, unified ? "unified" : "split", n_tokens, (int) chunk);
                    // the chunk path takes only layouts its rows are the scans' rows for
                    CHECK(!chunk || (type == GGML_TYPE_I16 && !pos_2d), "round %d: the chunk path built a %s mask%s",
                            round, ggml_type_name(type), pos_2d ? " with 2D positions" : "");

                    ++n_masks;
                    n_chunk += chunk;
                    n_whole += swa_type == LLAMA_SWA_TYPE_NONE;

                    ggml_free(ctx);
                }
            }
        }

        for (const auto & cells : p.v_cells) {
            check_cells(cells, n_seq);
        }
    }

    // the layouts the chunk rows leave to the scans, and the served phase's depth
    mask_counts counts;
    check_rows_and_blocks(rng, counts);
    check_large_layouts(rng, counts);

    n_masks += counts.n_masks;
    n_chunk += counts.n_chunk;

    printf("test-kv-mask: %d masks compared (%d through the bits, %d by the chunk rows), %d failures\n",
            n_masks, n_whole, n_chunk, n_fail);

    return n_fail == 0 && n_chunk > 0 ? 0 : 1;
}
