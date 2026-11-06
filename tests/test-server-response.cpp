// The server's result queue, driven the way the server drives it: the decode loop sends each request's results as it
// makes them, and one HTTP thread per request waits for its own.
//
// Goal: a result wakes only the reader waiting for it. Every request reads its own results in the order they were
// sent, a reader of several tasks reads them in send order, a result for a task nobody waits for is dropped, removing
// a task drops its pending results, and a broadcast reaches every waiting task. Each result used to wake every waiting
// reader, and each of them took the queue's lock to scan it: at 36 streaming sessions the decode loop spent 21 ms of a
// 148 ms round waiting for that lock (four RTX PRO 6000 cards).

#include "server-queue.h"

#include <chrono>
#include <cstdio>
#include <thread>
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

} // namespace

int main() {
    bool ok = true;
    ok = test_reader_of_several_tasks() && ok;
    ok = test_dropped_results() && ok;
    ok = test_broadcast() && ok;
    ok = test_one_wakeup_per_result() && ok;
    printf("%s\n", ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}
