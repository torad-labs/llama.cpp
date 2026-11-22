#include "ggml-ledger.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>

namespace {

constexpr int MAX_SLOTS = 4096;
constexpr int BUCKETS   = 44; // bucket k holds the values below 2^k and not below 2^(k-1); 0 holds zero and less

struct slot_data {
    std::atomic<int64_t> count;
    std::atomic<int64_t> sum;
    std::atomic<int64_t> max;
    std::atomic<int64_t> hist[BUCKETS];
    std::atomic<bool>    gauge;
    char                 name[112];
};

// static storage: zero before any code runs, so a slot can be made from another unit's initializer
slot_data             g_slots[MAX_SLOTS];
std::atomic<int>      g_n_slots{0};
std::atomic<uint64_t> g_steps{0};
std::atomic<uint64_t> g_timed_steps{0};
std::atomic<bool>     g_timed{false};

struct index_t {
    std::mutex                           mutex;
    std::unordered_map<std::string, int> slots;
};

index_t & index() {
    static index_t * i = new index_t; // never freed: a slot may be asked for while the process exits
    return *i;
}

void write_at_exit() {
    const char * path = ggml_env_str("GGML_LEDGER_FILE");
    if (path != nullptr && path[0] != '\0') {
        ggml_ledger_write(path);
    }
}

int64_t period() {
    static const int64_t n = ggml_env_int("GGML_LEDGER_PERIOD");
    return n > 0 ? n : 1;
}

int64_t now_us() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
}

int bucket_of(int64_t value) {
    if (value <= 0) {
        return 0;
    }
    int k = 0;
    for (uint64_t v = (uint64_t) value; v != 0; v >>= 1) {
        k++;
    }
    return k < BUCKETS ? k : BUCKETS - 1;
}

void append(std::string & out, const char * format, long long value) {
    char text[64];
    snprintf(text, sizeof(text), format, value);
    out += text;
}

} // namespace

bool ggml_ledger_on(void) {
    static const bool on = [] {
        const bool value = ggml_env_switch("GGML_LEDGER");
        if (value) {
            const char * path = ggml_env_str("GGML_LEDGER_FILE");
            if (path != nullptr && path[0] != '\0') {
                atexit(write_at_exit);
            }
        }
        return value;
    }();
    return on;
}

int ggml_ledger_slot(const char * name) {
    if (!ggml_ledger_on() || name == nullptr) {
        return -1;
    }
    index_t & idx = index();
    const std::lock_guard<std::mutex> lock(idx.mutex);
    const auto found = idx.slots.find(name);
    if (found != idx.slots.end()) {
        return found->second;
    }
    const int slot = g_n_slots.load(std::memory_order_relaxed);
    if (slot >= MAX_SLOTS) {
        return -1;
    }
    snprintf(g_slots[slot].name, sizeof(g_slots[slot].name), "%s", name);
    idx.slots.emplace(g_slots[slot].name, slot);
    g_n_slots.store(slot + 1, std::memory_order_release);
    return slot;
}

int ggml_ledger_slot_at(const char * name, int index) {
    if (!ggml_ledger_on()) {
        return -1;
    }
    char full[112];
    snprintf(full, sizeof(full), "%s[%d]", name, index);
    return ggml_ledger_slot(full);
}

int ggml_ledger_slot_of(const char * name, const char * detail) {
    if (!ggml_ledger_on()) {
        return -1;
    }
    char full[112];
    snprintf(full, sizeof(full), "%s.%s", name, detail != nullptr ? detail : "none");
    return ggml_ledger_slot(full);
}

void ggml_ledger_add(int slot, int64_t value) {
    if (slot < 0) {
        return;
    }
    slot_data & s = g_slots[slot];
    s.count.fetch_add(1, std::memory_order_relaxed);
    s.sum.fetch_add(value, std::memory_order_relaxed);
    s.hist[bucket_of(value)].fetch_add(1, std::memory_order_relaxed);
    int64_t seen = s.max.load(std::memory_order_relaxed);
    while (value > seen && !s.max.compare_exchange_weak(seen, value, std::memory_order_relaxed)) {
    }
}

void ggml_ledger_set(int slot, int64_t value) {
    if (slot < 0) {
        return;
    }
    slot_data & s = g_slots[slot];
    s.gauge.store(true, std::memory_order_relaxed);
    s.sum.store(value, std::memory_order_relaxed);
    int64_t seen = s.max.load(std::memory_order_relaxed);
    while (value > seen && !s.max.compare_exchange_weak(seen, value, std::memory_order_relaxed)) {
    }
}

bool ggml_ledger_step(void) {
    if (!ggml_ledger_on()) {
        return false;
    }
    const uint64_t step  = g_steps.fetch_add(1, std::memory_order_relaxed);
    const bool     timed = step % (uint64_t) period() == 0;
    if (timed) {
        g_timed_steps.fetch_add(1, std::memory_order_relaxed);
    }
    g_timed.store(timed, std::memory_order_relaxed);
    return timed;
}

bool ggml_ledger_step_timed(void) {
    return g_timed.load(std::memory_order_relaxed);
}

size_t ggml_ledger_json(char * buf, size_t size) {
    std::string out;
    out.reserve(1 << 16);
    out += "{\"version\":1";
    append(out, ",\"on\":%lld", ggml_ledger_on() ? 1 : 0);
    append(out, ",\"t_us\":%lld", now_us());
    append(out, ",\"steps\":%lld", (long long) g_steps.load(std::memory_order_relaxed));
    append(out, ",\"timed_steps\":%lld", (long long) g_timed_steps.load(std::memory_order_relaxed));
    append(out, ",\"period\":%lld", ggml_ledger_on() ? period() : 0);
    out += ",\"slots\":{";
    const int n = g_n_slots.load(std::memory_order_acquire);
    for (int i = 0; i < n; i++) {
        const slot_data & s = g_slots[i];
        if (i > 0) {
            out += ',';
        }
        out += '"';
        for (const char * c = s.name; *c != '\0'; c++) {
            if (*c == '"' || *c == '\\') {
                out += '\\';
            }
            out += (unsigned char) *c < 0x20 ? ' ' : *c;
        }
        out += "\":{";
        if (s.gauge.load(std::memory_order_relaxed)) {
            append(out, "\"g\":%lld", s.sum.load(std::memory_order_relaxed));
            append(out, ",\"max\":%lld", s.max.load(std::memory_order_relaxed));
        } else {
            append(out, "\"n\":%lld", s.count.load(std::memory_order_relaxed));
            append(out, ",\"sum\":%lld", s.sum.load(std::memory_order_relaxed));
            append(out, ",\"max\":%lld", s.max.load(std::memory_order_relaxed));
            out += ",\"h\":{";
            bool first = true;
            for (int k = 0; k < BUCKETS; k++) {
                const int64_t c = s.hist[k].load(std::memory_order_relaxed);
                if (c == 0) {
                    continue;
                }
                if (!first) {
                    out += ',';
                }
                first = false;
                append(out, "\"%lld\":", k);
                append(out, "%lld", c);
            }
            out += '}';
        }
        out += '}';
    }
    out += "}}";
    if (buf != nullptr && size > 0) {
        const size_t n_copy = out.size() < size - 1 ? out.size() : size - 1;
        memcpy(buf, out.data(), n_copy);
        buf[n_copy] = '\0';
    }
    return out.size() + 1;
}

bool ggml_ledger_write(const char * path) {
    const size_t size = ggml_ledger_json(nullptr, 0);
    std::string  text(size, '\0');
    ggml_ledger_json(text.data(), size);
    FILE * f = fopen(path, "w");
    if (f == nullptr) {
        return false;
    }
    const bool ok = fwrite(text.data(), 1, size - 1, f) == size - 1 && fputc('\n', f) != EOF;
    return fclose(f) == 0 && ok;
}
