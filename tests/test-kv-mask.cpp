// The KQ mask taken from the sequences' bits (llama_kv_cells::seq_cells64 and pos_max64) against the scans it replaces.
//
// Random cell states the way a server makes them: sequences in one unified pool or one stream each, prompts appended,
// drafts rolled back and their cells taken again, conversations ended and their cells reused from position 0, prefixes
// shared by two sequences, cells removed, kept, saved and restored (llama_kv_cache::prepare), positions shifted down with
// the shift applied or still pending. On each state: the cells' bits and bounds against their definition, then the mask
// of a decode or verify ubatch built three ways - cell by cell (LLAMA_KQ_MASK_SCAN_LEGACY), 16 cells at a time
// (LLAMA_KQ_MASK_SEQ_BITS_LEGACY) and 64 at a time from the bits - compared byte for byte, bit-packed and f16, causal
// and not, with and without a window, with and without 2D positions.

#include "../src/llama-hparams.h"
#include "../src/llama-kv-cache.h"
#include "../src/llama-kv-cells.h"

#include "ggml.h"

#include <algorithm>
#include <cstdio>
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

int main() {
    std::mt19937 rng(20260925);

    ggml_init_params ip = { /*.mem_size =*/ 64*1024*1024, /*.mem_buffer =*/ nullptr, /*.no_alloc =*/ false };

    int n_masks = 0, n_whole = 0;

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

        // the ubatch: a decode (1 token) or a verify (up to 5) of some sequences, placed in the cells first as the cache
        // does (apply_ubatch before the inputs are set); split streams take the same count from each sequence
        const uint32_t n_tps_seq = rng() % 2 ? 1 : 2 + rng() % 4;
        std::vector<llama_seq_id> seqs;
        for (uint32_t s = 0; s < n_seq; ++s) {
            if (!unified || rng() % 2 == 0 || s == n_seq - 1) {
                seqs.push_back(s);
            }
        }
        std::vector<llama_pos>    pos;
        std::vector<llama_seq_id> seq_ids;
        bool placed_all = true;
        for (const llama_seq_id s : seqs) {
            const llama_pos p0 = p.p_next[s];
            placed_all &= p.append(s, n_tps_seq).size() == n_tps_seq;
            for (uint32_t k = 0; k < n_tps_seq; ++k) {
                pos.push_back(p0 + k);
                seq_ids.push_back(s);
            }
        }
        if (!placed_all) {
            continue; // the pool is full
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

                    llama_hparams hparams{};

                    ggml_context * ctx = ggml_init(ip);
                    ggml_tensor * masks[3];
                    for (int m = 0; m < 3; ++m) {
                        masks[m] = ggml_new_tensor_4d(ctx, type, type == GGML_TYPE_I16 ? n_kv/16 : n_kv, n_tokens/n_stream, 1, n_stream);
                        memset(masks[m]->data, 0xa5, ggml_nbytes(masks[m])); // a cell no path writes shows as a difference
                    }

                    // cell by cell; 16 cells at a time; 64 at a time from the bits
                    llama_kv_cache_set_input_kq_mask(masks[0], &ubatch, causal, hparams, p.v_cells, p.seq_to_stream, n_swa, swa_type, true,  false);
                    llama_kv_cache_set_input_kq_mask(masks[1], &ubatch, causal, hparams, p.v_cells, p.seq_to_stream, n_swa, swa_type, false, false);
                    llama_kv_cache_set_input_kq_mask(masks[2], &ubatch, causal, hparams, p.v_cells, p.seq_to_stream, n_swa, swa_type, false, true);

                    const size_t nb = ggml_nbytes(masks[0]);
                    CHECK(memcmp(masks[0]->data, masks[1]->data, nb) == 0, "round %d: 16-cell scan differs from the cell scan (%s, n_kv %u, %s, %s)",
                            round, ggml_type_name(type), n_kv, causal ? "causal" : "non-causal", swa_type == LLAMA_SWA_TYPE_NONE ? "no window" : "window");
                    CHECK(memcmp(masks[0]->data, masks[2]->data, nb) == 0, "round %d: the bits differ from the cell scan (%s, n_kv %u, %s, %s, %u seqs, %s, %u tokens)",
                            round, ggml_type_name(type), n_kv, causal ? "causal" : "non-causal", swa_type == LLAMA_SWA_TYPE_NONE ? "no window" : "window",
                            n_seq, unified ? "unified" : "split", n_tokens);

                    ++n_masks;
                    n_whole += swa_type == LLAMA_SWA_TYPE_NONE;

                    ggml_free(ctx);
                }
            }
        }

        for (const auto & cells : p.v_cells) {
            check_cells(cells, n_seq);
        }
    }

    printf("test-kv-mask: %d masks compared (%d through the bits), %d failures\n", n_masks, n_whole, n_fail);

    return n_fail == 0 ? 0 : 1;
}
