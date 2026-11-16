// The server's result queue, driven the way the server drives it: the decode loop sends each request's results as it
// makes them, and one HTTP thread per request waits for its own.
//
// Goal: a result wakes only the reader waiting for it. Every request reads its own results in the order they were
// sent, a reader of several tasks reads them in send order, a result for a task nobody waits for is dropped, removing
// a task drops its pending results, and a broadcast reaches every waiting task. Each result used to wake every waiting
// reader, and each of them took the queue's lock to scan it: at 36 streaming sessions the decode loop spent 21 ms of a
// 148 ms round waiting for that lock (four RTX PRO 6000 cards).

#include "server-queue.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <map>
#include <thread>
#include <random>
#include <set>
#include <vector>

namespace {

struct test_result : server_task_result {
    int seq = -1;

    json to_json() override { return json{ { "seq", seq } }; }

    server_task_result * clone() const override { return new test_result(*this); }
};

server_task_result_ptr make_result(int id_task, int seq) {
    auto res = std::make_unique<test_result>();
    res->id  = id_task;
    res->seq = seq;
    return res;
}

int seq_of(const server_task_result_ptr & res) {
    return res ? static_cast<const test_result *>(res.get())->seq : -1;
}

bool fail(const char * what) {
    fprintf(stderr, "FAIL: %s\n", what);
    return false;
}

// n_readers HTTP threads each wait for n_results of their own task; the decode loop sends one result at a time and
// waits until its reader has it, so every other reader is waiting when a result arrives
bool test_one_wakeup_per_result() {
    constexpr int n_readers = 36;
    constexpr int n_results = 20;
    server_response queue;
    for (int r = 0; r < n_readers; ++r) {
        queue.add_waiting_task_id(r);
    }

    std::vector<std::atomic<int>> received(n_readers);
    std::vector<std::atomic<bool>> in_order(n_readers);
    for (int r = 0; r < n_readers; ++r) {
        received[r] = 0;
        in_order[r] = true;
    }
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);

    std::vector<std::thread> readers;
    for (int r = 0; r < n_readers; ++r) {
        readers.emplace_back([&, r] {
            const std::unordered_set<int> ids = { r };
            while (received[r] < n_results && std::chrono::steady_clock::now() < deadline) {
                auto res = queue.recv_with_timeout(ids, 1);
                if (!res) {
                    continue;
                }
                if (res->id != r || seq_of(res) != received[r]) {
                    in_order[r] = false;
                }
                received[r]++;
            }
        });
    }

    // let every reader reach its wait before the first result
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    const uint64_t wakeups_before = queue.n_wakeups;

    bool delivered = true;
    for (int k = 0; k < n_results && delivered; ++k) {
        for (int r = 0; r < n_readers && delivered; ++r) {
            queue.send(make_result(r, k));
            while (received[r] < k + 1) {
                if (std::chrono::steady_clock::now() >= deadline) {
                    delivered = false;
                    break;
                }
                std::this_thread::yield();
            }
        }
    }
    for (auto & t : readers) {
        t.join();
    }
    const uint64_t wakeups = queue.n_wakeups - wakeups_before;

    if (!delivered) {
        return fail("a reader did not receive its result");
    }
    for (int r = 0; r < n_readers; ++r) {
        if (!in_order[r]) {
            return fail("a reader received another task's result, or its own out of order");
        }
    }
    const uint64_t total = (uint64_t) n_readers * n_results;
    printf("%d readers, %llu results: %llu reader wakeups\n", n_readers, (unsigned long long) total, (unsigned long long) wakeups);
    // a reader wakes for its own result; a few wakeups the platform makes up are allowed
    if (wakeups > total + total / 10) {
        return fail("results woke readers that were waiting for other tasks");
    }
    return true;
}

// one reader of two tasks reads both in the order they were sent
bool test_reader_of_several_tasks() {
    server_response queue;
    queue.add_waiting_task_ids({ 1, 2 });
    queue.send(make_result(2, 0));
    queue.send(make_result(1, 1));
    queue.send(make_result(2, 2));
    const std::unordered_set<int> ids = { 1, 2 };
    for (int seq = 0; seq < 3; ++seq) {
        auto res = queue.recv(ids);
        if (seq_of(res) != seq) {
            return fail("a reader of several tasks did not read them in send order");
        }
    }

    // a reader blocked on two tasks wakes for either
    std::atomic<int> got{-1};
    std::thread reader([&] { got = seq_of(queue.recv(ids)); });
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    queue.send(make_result(1, 7));
    reader.join();
    if (got != 7) {
        return fail("a reader waiting on two tasks did not wake for one of them");
    }
    return true;
}

bool test_dropped_results() {
    server_response queue;
    // nobody waits for task 5: its result is dropped
    queue.send(make_result(5, 0));
    queue.add_waiting_task_id(5);
    if (queue.recv_with_timeout({ 5 }, 0)) {
        return fail("a result sent before anyone waited for its task was kept");
    }

    // removing a task drops its pending results
    queue.add_waiting_task_id(7);
    queue.send(make_result(7, 0));
    queue.remove_waiting_task_id(7);
    queue.add_waiting_task_id(7);
    if (queue.recv_with_timeout({ 7 }, 0)) {
        return fail("removing a task kept its pending result");
    }
    return true;
}

bool test_broadcast() {
    server_response queue;
    queue.add_waiting_task_ids({ 3, 4 });
    std::atomic<int> got3{-1}, got4{-1};
    std::thread r3([&] { got3 = seq_of(queue.recv(3)); });
    std::thread r4([&] { got4 = seq_of(queue.recv(4)); });
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    queue.broadcast(make_result(-1, 9));
    r3.join();
    r4.join();
    if (got3 != 9 || got4 != 9) {
        return fail("a broadcast did not reach every waiting task");
    }
    return true;
}

// Generated sparse arrivals: each owner's oldest readers get a turn, even if another
// owner has only one late-numbered slot. No reader may wait forever behind that slot.
bool test_prompt_admission_fairness() {
    std::mt19937 rng(53);
    for (int trial = 0; trial < 64; ++trial) {
        const uint32_t owners = 2 + rng() % 3;
        std::vector<llama_seq_id> pending;
        for (llama_seq_id id = 0; id < 40; ++id) {
            if (id % owners == 0 || rng() % 5 == 0) {
                pending.push_back(id);
            }
        }
        server_prompt_scheduler scheduler;
        std::set<llama_seq_id> seen;
        for (size_t round = 0; round < pending.size(); ++round) {
            auto order = scheduler.order(pending, owners);
            if (std::set<llama_seq_id>(order.begin(), order.end()) !=
                    std::set<llama_seq_id>(pending.begin(), pending.end()) || order.size() != pending.size()) {
                return fail("prompt admission lost or duplicated a pending slot");
            }
            std::set<uint32_t> active_owners;
            for (auto id : pending) { active_owners.insert(id % owners); }
            std::set<uint32_t> admitted_owners;
            for (size_t i = 0; i < active_owners.size(); ++i) {
                admitted_owners.insert(order[i] % owners);
                seen.insert(order[i]);
                scheduler.admitted(order[i]);
            }
            if (admitted_owners != active_owners) {
                return fail("prompt admission gave one owner a second turn before another got one");
            }
        }
        if (seen.size() != pending.size()) {
            return fail("a sparse owner starved another owner's prompt");
        }
    }
    return true;
}

// --prompt-step-ms, driven the way the server drives it: a read beside one reply. A step of the reply alone takes 10 ms,
// and a step's prompt tokens fixed_us and per_token_us(step) a token more; every spike-th step waits 500 ms more.
struct budget_run {
    int32_t settled = 0; // the allowance of most of the last 300 steps
    double  over    = 0; // the share of the steps after the first 300 whose prompt part was over the target
};

template <typename F>
budget_run run_prompt_budget(int target_ms, int32_t n_ceiling, int64_t fixed_us, F per_token_us, int n_steps, int spike = 0) {
    server_prompt_budget budget;
    budget.init(target_ms, n_ceiling);
    int64_t t_us = 0;
    for (int i = 0; i < 8; ++i) {
        budget.step(t_us, 1, 0);
        t_us += 10000;
    }
    std::map<int32_t, int> count;
    int n_over = 0;
    for (int i = 0; i < n_steps; ++i) {
        const int32_t n = budget.allowance();
        budget.step(t_us, 1, n);
        const int64_t cost_us = fixed_us + (int64_t) per_token_us(i)*n;
        t_us += 10000 + cost_us + (spike > 0 && i % spike == 0 ? 500000 : 0);
        if (i >= n_steps - 300) {
            count[n]++;
        }
        if (i >= 300) {
            n_over += cost_us > (int64_t) target_ms*1000;
        }
    }
    budget_run res;
    for (const auto & [n, c] : count) {
        if (c > count[res.settled]) {
            res.settled = n;
        }
    }
    res.over = n_steps > 300 ? (double) n_over/(n_steps - 300) : 0;
    return res;
}

bool test_prompt_budget() {
    const auto flat = [](int64_t us) { return [us](int) { return us; }; };

    // a cost in proportion to the tokens: 128 tokens take 12.8 ms and 256 take 25.6
    budget_run res = run_prompt_budget(20, 1024, 0, flat(100), 2000);
    if (res.settled != 128 || res.over > 0.05) {
        return fail("the prompt budget did not settle on the largest allowance under its target");
    }
    // the same with a slow step in every seven: a wait is not the allowance's cost
    res = run_prompt_budget(20, 1024, 0, flat(100), 2000, 7);
    if (res.settled != 128 || res.over > 0.05) {
        return fail("slow steps moved the prompt budget");
    }
    // a fixed part and a part a token: 256 tokens take 22.8 ms and 512 take 35.6
    res = run_prompt_budget(30, 1024, 10000, flat(50), 2000);
    if (res.settled != 256 || res.over > 0.05) {
        return fail("the prompt budget did not find the allowance under its target past a fixed cost");
    }
    // a pass that takes 50 ms whatever its tokens: smaller steps are not shorter, so the limit of tokens stands
    res = run_prompt_budget(20, 1024, 50000, flat(20), 2000);
    if (res.settled != 1024) {
        return fail("the prompt budget kept small steps that a fixed cost made no shorter");
    }
    // the smallest step, 19 ms, over the target: the steps stay within twice its cost (128 tokens take 31 ms, 256 take 47)
    res = run_prompt_budget(10, 1024, 15000, flat(125), 2000);
    if (res.settled != 128) {
        return fail("the prompt budget went past twice the cost of its smallest step");
    }
    // a prompt that costs more a token as it gets deeper: the allowance comes down with it
    res = run_prompt_budget(20, 1024, 0, [](int i) { return 100 + std::min(i, 3000)/10; }, 3400);
    if (res.settled != 32) {
        return fail("the prompt budget did not come down as a token's cost went up");
    }
    // a token that comes to cost a third of what it did: the allowance goes back up, and no further than fits
    res = run_prompt_budget(20, 1024, 0, [](int i) { return i < 600 ? 300 : 100; }, 1200);
    if (res.settled != 128 || res.over > 0.05) {
        return fail("the prompt budget did not go back up as a token's cost came down");
    }
    // the limit of tokens is the largest allowance, a power of two or not
    res = run_prompt_budget(1000, 300, 0, flat(100), 600);
    if (res.settled != 300) {
        return fail("the prompt budget did not reach its limit of tokens");
    }

    // the wait for the next request is no step's time
    server_prompt_budget budget;
    budget.init(20, 1024);
    int64_t t_us = 0;
    for (int i = 0; i < 8; ++i) {
        if (i == 4) {
            budget.idle();
            t_us += 5000000;
        }
        budget.step(t_us, 1, 0);
        t_us += 10000;
    }
    const auto & replies = budget.base[1];
    if (replies.n != 4) {
        return fail("the prompt budget did not time the steps of replies on each side of an idle wait");
    }
    for (int64_t i = 0; i < replies.n; ++i) {
        if (replies.v[i] != 10000) {
            return fail("the prompt budget took an idle wait for a step of replies");
        }
    }
    return true;
}

} // namespace

int main() {
    bool ok = true;
    ok = test_prompt_admission_fairness() && ok;
    ok = test_prompt_budget() && ok;
    ok = test_reader_of_several_tasks() && ok;
    ok = test_dropped_results() && ok;
    ok = test_broadcast() && ok;
    ok = test_one_wakeup_per_result() && ok;
    printf("%s\n", ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}
