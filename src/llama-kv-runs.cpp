#include "llama-kv-runs.h"

#include "llama-batch.h"
#include "llama-impl.h"
#include "llama-kv-cells.h"

#include <algorithm>
#include <map>
#include <tuple>

// whether run b holds cell c, and as which of its cells
static bool run_holds(const llama_kv_run & b, uint32_t c, uint32_t & offset) {
    if (c < b.cell || (c - b.cell) % b.stride != 0) {
        return false;
    }
    offset = (c - b.cell)/b.stride;
    return offset < b.len;
}

// run b takes cell c at position p when c comes after its last cell at its stride (any stride for a run of one cell)
static bool run_extend(llama_kv_run & b, uint32_t c, llama_pos p, bool shared) {
    const uint32_t last = b.cell + (b.len - 1)*b.stride;
    if (b.shared != shared || p != b.pos + llama_pos(b.len) || c <= last || (b.len > 1 && c - last != b.stride) ||
            c - last > INT32_MAX) {
        return false;
    }
    b.stride = c - last;
    b.len++;
    return true;
}

void llama_kv_runs::build(const llama_kv_cells & cells, llama_seq_id seq, seq_runs & sr) const {
    sr = seq_runs();
    const auto & pos = cells.positions();
    // Keep 7 extra older cells on SWA so drafted verify groups (up to 8 tokens) retain their window.
    const llama_pos pos_swa = n_swa > 0 ? std::max<llama_pos>(0, cells.seq_pos_max(seq) - 7) : -1;
    const uint32_t n_groups = (cells.used_max_p1() + 63)/64;
    for (uint32_t g = 0; g < n_groups; ++g) {
        for (uint64_t bits = cells.seq_cells64(seq, g); bits; bits &= bits - 1) {
            const uint32_t c = 64*g + ctz64(bits);
            if (n_swa > 0 && llama_hparams::is_masked_swa(n_swa, swa_type, pos[c], pos_swa)) {
                continue;
            }
            const bool shared = cells.seq_count(c) > 1;
            if (sr.runs.empty() || !run_extend(sr.runs.back(), c, pos[c], shared)) {
                sr.runs.push_back({c, pos[c], 1, 1, shared});
            }
            if (pos[c] >= sr.pos_end) {
                sr.pos_end = pos[c];
                sr.tail = sr.runs.size() - 1;
            }
            sr.n_cells++;
            sr.cell_max = c;
        }
    }
    sr.built = true;
}

void llama_kv_runs::sync(const llama_kv_cells & cells) {
    if (seqs.empty()) {
        seqs.resize(LLAMA_MAX_SEQ);
    }
    // a list past 1/64 of the cells costs more to apply than the runs cost to build again
    cells.changes().track(std::max<uint32_t>(256, cells.size()/64));
    const bool listed = cells.changes().take(chg) && synced == &cells;
    synced = &cells;
    if (!listed) {
        for (auto & sr : seqs) {
            sr.built = false;
        }
        return;
    }
    if (chg.empty()) {
        return;
    }
    std::sort(chg.begin(), chg.end());
    chg.erase(std::unique(chg.begin(), chg.end()), chg.end());

    const auto & pos = cells.positions();
    std::vector<uint32_t> added;
    for (llama_seq_id s = 0; s < LLAMA_MAX_SEQ; ++s) {
        seq_runs & sr = seqs[s];
        if (!sr.built) {
            continue;
        }
        bool stale = false;
        added.clear();
        for (const uint32_t c : chg) {
            const bool holds = !cells.is_empty(c) && cells.seq_has(c, s);
            bool held = false;
            for (size_t i = 0; c <= sr.cell_max && i < sr.runs.size() && !held; ++i) {
                uint32_t offset = 0;
                if (run_holds(sr.runs[i], c, offset)) {
                    // a cell the runs hold stays as it was, whatever other sequences hold it
                    held = true;
                    stale = !holds || pos[c] != sr.runs[i].pos + llama_pos(offset) ||
                        (cells.seq_count(c) > 1) != sr.runs[i].shared;
                }
            }
            if (stale) {
                break;
            }
            if (holds && !held) {
                added.push_back(c);
            }
        }
        // cells added at the end of the sequence, in the order of their positions
        std::sort(added.begin(), added.end(), [&](uint32_t a, uint32_t b) { return pos[a] < pos[b]; });
        for (size_t i = 0; i < added.size() && !stale; ++i) {
            const uint32_t c = added[i];
            const bool shared = cells.seq_count(c) > 1;
            stale = pos[c] != sr.pos_end + 1;
            if (!stale && (sr.runs.empty() || !run_extend(sr.runs[sr.tail], c, pos[c], shared))) {
                sr.runs.push_back({c, pos[c], 1, 1, shared});
                sr.tail = sr.runs.size() - 1;
            }
            sr.n_cells++;
            sr.pos_end = pos[c];
            sr.cell_max = std::max(sr.cell_max, c);
        }
        if (!stale && n_swa > 0 && !added.empty()) {
            // Trim cells older than the sliding window plus 7-token draft verify slack.
            const llama_pos pos_swa = std::max<llama_pos>(0, sr.pos_end - 7);
            for (size_t ri = 0; ri < sr.runs.size(); ) {
                auto & b = sr.runs[ri];
                while (b.len > 0 && llama_hparams::is_masked_swa(n_swa, swa_type, b.pos, pos_swa)) {
                    b.cell += b.stride;
                    b.pos  += 1;
                    b.len  -= 1;
                    sr.n_cells--;
                }
                if (b.len == 0) {
                    sr.runs.erase(sr.runs.begin() + ri);
                    if (sr.tail > ri) {
                        sr.tail--;
                    } else if (sr.tail == ri) {
                        sr.tail = sr.runs.empty() ? 0 : sr.runs.size() - 1;
                    }
                } else {
                    ++ri;
                }
            }
        }
        sr.built = !stale && (n_swa > 0 || sr.n_cells == cells.seq_n_cells(s));
    }
}

const std::vector<llama_kv_run> * llama_kv_runs::get(const llama_kv_cells & cells, llama_seq_id seq, uint32_t n_max) {
    sync(cells);
    seq_runs & sr = seqs[seq];
    if (!sr.built) {
        build(cells, seq, sr);
    }
    return sr.runs.size() <= n_max ? &sr.runs : nullptr;
}

void llama_kv_runs::write(const llama_kv_cells & cells, const llama_ubatch & ubatch, int32_t * data, int64_t n_entries) {
    const uint32_t n_rows = ubatch.n_tokens;
    std::vector<const std::vector<llama_kv_run> *> lists(n_rows);
    std::vector<uint32_t> seq_of(n_rows);
    for (uint32_t r = 0; r < n_rows; ++r) {
        const llama_seq_id seq = ubatch.seq_id[r][0];
        lists[r] = get(cells, seq, uint32_t(n_entries - 2));
        seq_of[r] = uint32_t(ubatch.seq_idx[seq]);
        GGML_ASSERT(lists[r] && seq_of[r] < 32 && "the step was classed by llama_kv_cache::kv_runs_fit");
    }

    // Shared list: stride-1 runs held whole by at least two distinct sequences (skipped on SWA).
    using key = std::tuple<uint32_t, llama_pos, uint32_t>;
    std::map<key, std::pair<uint32_t, uint32_t>> held; // sequences, rows
    if (n_swa == 0) {
        for (uint32_t r = 0; r < n_rows; ++r) {
            for (const auto & b : *lists[r]) {
                if (b.stride == 1) {
                    auto & h = held[{b.cell, b.pos, b.len}];
                    h.first |= 1u << seq_of[r];
                    h.second++;
                }
            }
        }
    }
    std::fill(data, data + 4*n_entries*(n_rows + 1), 0);
    int32_t * list = data + 4*n_entries*n_rows;
    int32_t n_list = 0, at_list = 0;
    for (auto it = held.begin(); it != held.end();) {
        const auto & [cell, pos, len] = it->first;
        const uint32_t seq_mask = it->second.first;
        const bool multi_seq = (seq_mask & (seq_mask - 1u)) != 0;
        if (!multi_seq || n_list == n_entries - 2) {
            it = held.erase(it);
            continue;
        }
        int32_t * e = list + 4*(1 + n_list);
        e[0] = int32_t(cell);
        e[1] = pos;
        e[2] = at_list;
        e[3] = int32_t(it->second.first);
        at_list += int32_t(len);
        n_list++;
        ++it;
    }
    list[0] = n_list;
    list[1] = at_list;
    list[4*(1 + n_list) + 2] = at_list;

    // each row: the runs of the shared list first, then the others, in cell order as the sequence holds them
    for (uint32_t r = 0; r < n_rows; ++r) {
        int32_t * row = data + 4*n_entries*r;
        int32_t n = 0, at = 0, n_shared = 0;
        for (const bool in_list : {true, false}) {
            for (const auto & b : *lists[r]) {
                if ((b.stride == 1 && held.count({b.cell, b.pos, b.len})) != in_list) {
                    continue;
                }
                int32_t * e = row + 4*(1 + n);
                e[0] = int32_t(b.cell);
                e[1] = b.pos;
                e[2] = at;
                e[3] = int32_t(b.stride | (in_list ? 1u << 31 : 0u));
                at += int32_t(b.len);
                n_shared += in_list ? int32_t(b.len) : 0;
                n++;
            }
        }
        row[0] = n;
        row[1] = at;
        row[2] = n_shared;
        row[3] = int32_t(seq_of[r]);
        row[4*(1 + n) + 2] = at;
    }
}
