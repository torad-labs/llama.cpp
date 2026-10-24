#include "server-common.h"
#include <cstdio>

int main() {
    server_slot_stats stats;
    // Nine delivered decode tokens over three verification rounds. The first
    // output belongs to prompt timing and is excluded from decode throughput.
    stats.n_gen = 19; // includes unrelated plain decode tokens
    stats.n_draft_tokens = 11;
    stats.n_draft_accepted = 6;
    stats.n_draft_verif_steps = 3;
    stats.n_draft_delivered = 9;
    for (const size_t rows : {3, 4, 7}) stats.record_draft_verification(rows, false);
    const auto result = stats.to_json();
    if (result.value("draft_rounds", 0) != 3 ||
        result.value("draft_delivered_n", 0) != 9 ||
        result.value("draft_delivered_per_round", 0.0) != 3.0) {
        fprintf(stderr, "FAIL: sampled response omits delivered-token/round accounting\n");
        return 1;
    }
    // Cancellation before the first output never underflows a token count.
    stats = {};
    stats.n_draft_tokens = 2;
    const auto cancelled = stats.to_json();
    if (cancelled.value("draft_delivered_n", -1) != 0 ||
        cancelled.value("draft_delivered_per_round", -1.0) != 0.0) { return 2; }
    // Generated verification/replay sequences conserve attempts and work even
    // if a checkpoint restore produces no delivered tokens in that attempt.
    uint64_t rows_total = 0, replay_total = 0;
    std::array<uint64_t, 9> histogram = {};
    uint32_t rng = 530;
    for (int i = 0; i < 512; ++i) {
        rng = 1664525u*rng + 1013904223u;
        const size_t rows = 1 + rng%17;
        const bool replay = (rng >> 8)%3 == 0;
        stats.record_draft_verification(rows, replay);
        rows_total += rows;
        replay_total += replay;
        histogram[std::min<size_t>(rows, 9)-1]++;
    }
    const auto work = stats.to_json();
    if (work.at("draft_verify_attempts") != 512 || work.at("draft_verify_rows_total") != rows_total ||
        work.at("draft_replay_attempts") != replay_total) return 3;
    for (size_t row = 0; row < histogram.size(); ++row) {
        if (work.at("draft_verify_rows").at(row < 8 ? std::to_string(row+1) : ">8") != histogram[row]) return 4;
    }
    return 0;
}
