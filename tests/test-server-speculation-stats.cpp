#include "server-common.h"
#include "server-task.h"
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
#include <unistd.h>
#endif

struct server_prompt_cache_test_access {
    static void insert(server_prompt_cache & cache, llama_token token) {
        server_prompt_cache_state state;
        state.prompt.tokens.push_back(token);
        state.data->main.resize(100);
        cache.insert_state(std::move(state));
    }
    static size_t count(const server_prompt_cache & cache) { return cache.states.size(); }
    static const server_prompt_cache_state & front(const server_prompt_cache & cache) { return cache.states.front(); }
    static const server_prompt_cache_state & back(const server_prompt_cache & cache) { return cache.states.back(); }
    static void erase_front(server_prompt_cache & cache) { cache.erase_state(cache.states.begin()); }
    static void insert(server_prompt_cache & cache, server_prompt_cache_state && state) { cache.insert_state(std::move(state)); }
    static const std::list<server_prompt_cache_state> & states(const server_prompt_cache & cache) { return cache.states; }
    static server_prompt_cache_refusal refusal(const server_prompt_cache & cache, server_prompt_cache_result result) {
        return { &cache, cache.revision, cache.limit_size, cache.limit_tokens, result };
    }
};

static bool test_update_eviction_accounting() {
    for (const bool bytes : {true, false}) {
        server_prompt_cache cache(0, 0);
        cache.limit_size = bytes ? 250 : 0;
        cache.limit_tokens = bytes ? 0 : 2;
        server_prompt_cache_test_access::insert(cache, 1);
        server_prompt_cache_test_access::insert(cache, 2);
        server_prompt_cache_test_access::insert(cache, 3);
        const auto * protect = &server_prompt_cache_test_access::front(cache);
        cache.update(protect);
        if (server_prompt_cache_test_access::count(cache) != 2 || cache.size() != 200 || cache.n_tokens() != 2) {
            fprintf(stderr, "FAIL: %s update eviction left %zu states, %zu bytes, %zu tokens\n",
                bytes ? "byte" : "token", server_prompt_cache_test_access::count(cache), cache.size(), cache.n_tokens());
            return false;
        }
        if (server_prompt_cache_test_access::front(cache).prompt.tokens.get_tokens().front() != 1 ||
                server_prompt_cache_test_access::back(cache).prompt.tokens.get_tokens().front() != 3) {
            fprintf(stderr, "FAIL: %s update eviction did not retain the protected and newest states\n", bytes ? "byte" : "token");
            return false;
        }
    }
    return true;
}

static bool test_idle_refusal_invalidation() {
    server_prompt_cache cache(1, 100);
    server_prompt empty;
    const auto check = [&](server_prompt_cache_refusal & refusal, server_prompt_cache_result expected) {
        // An uncached call reaches the empty-prompt guard without reading a context.
        return cache.save_idle(empty, nullptr, nullptr, nullptr, 0, refusal) == expected;
    };
    for (auto result : {server_prompt_cache_result::too_large, server_prompt_cache_result::no_space}) {
        auto refusal = server_prompt_cache_test_access::refusal(cache, result);
        if (!check(refusal, result) || !check(refusal, result)) { return false; }
        ++cache.limit_size;
        if (!check(refusal, server_prompt_cache_result::empty)) { return false; }
        refusal = server_prompt_cache_test_access::refusal(cache, result);
        ++cache.limit_tokens;
        if (!check(refusal, server_prompt_cache_result::empty)) { return false; }

        server_prompt_cache_test_access::insert(cache, 1);
        refusal = server_prompt_cache_test_access::refusal(cache, result);
        server_prompt_cache_test_access::erase_front(cache);
        server_prompt_cache_test_access::insert(cache, 2);
        // Equal occupancy can still contain a newly recoverable snapshot.
        if (!check(refusal, server_prompt_cache_result::empty)) { return false; }
        refusal = server_prompt_cache_test_access::refusal(cache, result);
        server_prompt_cache_test_access::erase_front(cache);
        if (!check(refusal, server_prompt_cache_result::empty)) { return false; }
        refusal = {};
        if (!check(refusal, server_prompt_cache_result::empty)) { return false; }
    }
    auto failed = server_prompt_cache_test_access::refusal(cache, server_prompt_cache_result::failed);
    return check(failed, server_prompt_cache_result::empty);
}

#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
// one snapshot of the store/open test: session s has s + 1 tokens, a checkpoint, and no draft state when s is 1
static server_prompt_cache_state stored_state(int s) {
    const auto fill = [](uint8_t * ptr, size_t n, int seed) {
        for (size_t i = 0; i < n; i++) {
            ptr[i] = (uint8_t) (seed + 7 * i);
        }
    };
    server_prompt_cache_state state;
    for (int i = 0; i <= s; i++) {
        state.prompt.tokens.push_back(100 * s + i);
    }
    state.pos_min = s;
    state.data->main.resize(9000 + s);
    fill(state.data->main.data(), state.data->main.size(), s);
    if (s != 1) {
        state.data->drft.resize(300 + s);
        fill(state.data->drft.data(), state.data->drft.size(), 50 + s);
    }
    state.data->spec = { 1, 2, 3, (uint8_t) s };
    common_prompt_checkpoint checkpoint;
    checkpoint.n_tokens = s + 1;
    checkpoint.pos_min  = 0;
    checkpoint.pos_max  = s;
    checkpoint.pinned   = s == 2;
    checkpoint.data_tgt.assign(70 + s, (uint8_t) (10 + s));
    checkpoint.data_dft.assign(s, (uint8_t) (20 + s));
    checkpoint.data_spec.assign(5, (uint8_t) (30 + s));
    state.prompt.checkpoints.push_back(std::move(checkpoint));
    return state;
}

static bool same_state(const server_prompt_cache_state & a, const server_prompt_cache_state & b) {
    const auto same = [](const uint8_t * x, size_t n_x, const uint8_t * y, size_t n_y) {
        return n_x == n_y && (n_x == 0 || memcmp(x, y, n_x) == 0);
    };
    if (a.prompt.tokens.get_tokens() != b.prompt.tokens.get_tokens() || a.pos_min != b.pos_min || a.size() != b.size() ||
            !same(a.data->main.data(), a.data->main.size(), b.data->main.data(), b.data->main.size()) ||
            !same(a.data->drft.data(), a.data->drft.size(), b.data->drft.data(), b.data->drft.size()) ||
            a.data->spec != b.data->spec || a.prompt.checkpoints.size() != b.prompt.checkpoints.size()) {
        return false;
    }
    auto it = b.prompt.checkpoints.begin();
    for (const auto & cur : a.prompt.checkpoints) {
        if (cur.n_tokens != it->n_tokens || cur.pos_min != it->pos_min || cur.pos_max != it->pos_max || cur.pinned != it->pinned ||
                cur.data_tgt != it->data_tgt || cur.data_dft != it->data_dft || cur.data_spec != it->data_spec) {
            return false;
        }
        ++it;
    }
    return true;
}

static size_t count_files(const std::string & dir) {
    size_t n = 0;
    std::error_code ec;
    for (const auto & entry : std::filesystem::directory_iterator(dir, ec)) {
        (void) entry;
        n++;
    }
    return n;
}

// LLAMA_PROMPT_CACHE_DIR: what a stop stores is what the next start's cache holds, in the same order
static bool test_store_open() {
    const std::string dir = (std::filesystem::temp_directory_path() / ("test-prompt-file-" + std::to_string(getpid()))).string();
    std::filesystem::remove_all(dir);
    const auto fail = [&](const char * what) {
        fprintf(stderr, "FAIL: prompt files: %s\n", what);
        std::filesystem::remove_all(dir);
        return false;
    };
    const auto store = [&](int n) {
        server_prompt_cache cache(0, 0);
        for (int s = 0; s < n; s++) {
            server_prompt_cache_test_access::insert(cache, stored_state(s));
        }
        return cache.store(dir, 42) == (size_t) n && server_prompt_cache_test_access::count(cache) == 0 && cache.size() == 0;
    };

    if (!store(3) || count_files(dir) != 3) { return fail("three snapshots are not three files and an empty cache"); }
    {
        // other models: nothing is taken and nothing removed
        server_prompt_cache cache(0, 0);
        if (cache.open(dir, 43, false) != 0 || count_files(dir) != 3) { return fail("files of other models were taken or removed"); }
    }
    {
        server_prompt_cache cache(0, 0);
        if (cache.open(dir, 42, false) != 3 || count_files(dir) != 0) { return fail("the files were not all taken and removed"); }
        int s = 0;
        size_t n_bytes = 0;
        for (const auto & state : server_prompt_cache_test_access::states(cache)) {
            const auto expected = stored_state(s++);
            n_bytes += expected.size();
            if (!same_state(state, expected)) { return fail("a snapshot differs from the one stored"); }
        }
        if (cache.size() != n_bytes || cache.n_tokens() != 6) { return fail("the cache's accounting differs"); }
        // stored again, from the mapped files this time
        if (cache.store(dir, 42) != 3 || count_files(dir) != 3) { return fail("mapped snapshots were not stored again"); }
    }
    {
        // a second stop's files come after the first's, and the limits of the cache that opens them apply
        if (!store(2)) { return fail("a second stop stored nothing"); }
        server_prompt_cache cache(0, 0);
        cache.limit_tokens = 5;
        if (cache.open(dir, 42, false) != 5 || server_prompt_cache_test_access::count(cache) != 2 ||
                !same_state(server_prompt_cache_test_access::front(cache), stored_state(0)) ||
                !same_state(server_prompt_cache_test_access::back(cache), stored_state(1))) {
            return fail("the newest stop's snapshots are not the ones kept");
        }
    }
    {
        // a file cut short is removed, the whole ones are taken
        if (!store(3)) { return fail("a third stop stored nothing"); }
        std::string first;
        for (const auto & entry : std::filesystem::directory_iterator(dir)) {
            if (first.empty() || entry.path().string() < first) {
                first = entry.path().string();
            }
        }
        std::filesystem::resize_file(first, std::filesystem::file_size(first) - 1);
        server_prompt_cache cache(0, 0);
        if (cache.open(dir, 42, false) != 2 || count_files(dir) != 0 ||
                !same_state(server_prompt_cache_test_access::front(cache), stored_state(1))) {
            return fail("a file cut short was taken, or kept, or took the others with it");
        }
    }
    std::filesystem::remove_all(dir);
    return true;
}
#else
static bool test_store_open() { return true; }
#endif

int main() {
    if (!test_update_eviction_accounting()) { return 5; }
    if (!test_idle_refusal_invalidation()) { return 6; }
    if (!test_store_open()) { return 7; }

    server_slot_stats stats;
    // Nine delivered decode tokens over three verification rounds. The first
    // output belongs to prompt timing and is excluded from decode throughput.
    stats.n_gen = 19; // includes unrelated plain decode tokens
    stats.n_draft_tokens = 11;
    stats.n_draft_accepted = 6;
    stats.n_draft_verif_steps = 3;
    stats.n_draft_delivered = 9;
    for (const size_t rows : {3, 4, 7}) stats.record_draft_verification(rows, false);
    stats.record_draft_verification(3, true);
    stats.record_draft_verification(4, true);
    const auto result = stats.to_json();
    if (result.value("draft_rounds", 0) != 3 ||
        result.value("draft_delivered_n", 0) != 9 ||
        result.value("draft_delivered_per_round", 0.0) != 3.0 ||
        result.value("draft_verify_attempts", 0) != 5 || result.value("draft_replay_attempts", 0) != 2) {
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
