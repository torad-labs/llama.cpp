#pragma once

// The ledger: where a step's time and a device's resources go, kept by the engine itself.
//
// Any code names what it counts or times and adds to it; nothing in here knows a model. A reader takes two
// snapshots and reads the difference between them, so one tool serves every model, backend and workload:
//
//   GGML_LEDGER=1           turns it on (off by default, so a served model pays nothing for profiling paths it
//                           does not use; an enabled entry costs two or three relaxed atomic adds)
//   GGML_LEDGER_FILE=path   writes a snapshot when the process exits
//   GGML_LEDGER_PERIOD=N    device time is read on one decode in N (default 8), since reading it costs two
//                           events on the device's stream for each graph evaluation
//
// A slot holds how many times it was added to, the sum and the largest of what was added, and how many of the
// values fell in each power of two, which is what tells a steady cost from a rare large one. A gauge holds the
// last value it was set to. Names are dotted paths; a device's entries end in [device].

#include "ggml.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

    GGML_API bool ggml_ledger_on(void);

    // The slot of a name, made at its first use. -1 when the ledger is off or full: adding to -1 does nothing.
    GGML_API int ggml_ledger_slot   (const char * name);
    GGML_API int ggml_ledger_slot_at(const char * name, int index);           // "name[index]"
    GGML_API int ggml_ledger_slot_of(const char * name, const char * detail); // "name.detail"

    GGML_API void ggml_ledger_add(int slot, int64_t value); // one more of this value: microseconds, bytes, rows
    GGML_API void ggml_ledger_set(int slot, int64_t value); // a gauge's value now

    // A decode begins. True when this one's device time is read (one in GGML_LEDGER_PERIOD).
    GGML_API bool ggml_ledger_step(void);
    GGML_API bool ggml_ledger_step_timed(void);

    // A snapshot as JSON. Returns the bytes it needs, the terminator included; writes at most size of them.
    GGML_API size_t ggml_ledger_json(char * buf, size_t size);
    GGML_API bool   ggml_ledger_write(const char * path);

#ifdef __cplusplus
}

#include <chrono>

// Adds the microseconds between its making and its end to a slot.
struct ggml_ledger_timer {
    int                                   slot;
    std::chrono::steady_clock::time_point t0;

    explicit ggml_ledger_timer(int slot) : slot(slot) {
        if (slot >= 0) {
            t0 = std::chrono::steady_clock::now();
        }
    }
    ggml_ledger_timer(const ggml_ledger_timer &) = delete;
    ggml_ledger_timer & operator=(const ggml_ledger_timer &) = delete;

    ~ggml_ledger_timer() {
        stop();
    }

    void stop() {
        if (slot >= 0) {
            ggml_ledger_add(slot, std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - t0).count());
            slot = -1;
        }
    }
};
#endif
