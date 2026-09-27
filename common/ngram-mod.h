#pragma once

#include <cstdint>
#include <vector>
#include <cstddef>

//
// common_ngram_mod
// ref: https://github.com/ggml-org/llama.cpp/pull/19164
//

// basic n-gram hasher
struct common_ngram_mod {
    using entry_t = int32_t;

    static constexpr entry_t EMPTY = -1;

    common_ngram_mod(uint16_t n, size_t size);

    size_t  idx(const entry_t * tokens) const;
    void    add(const entry_t * tokens);
    entry_t get(const entry_t * tokens) const; // return -1 if not found, a slot another n-gram holds included

    void reset();

    size_t get_n()    const;
    size_t get_used() const;

    size_t size()       const;
    size_t size_bytes() const;

private:
    size_t n; // ngram size to hash

    size_t used;

    // a slot keeps the high half of its n-gram's hash beside the token that followed it: the slot is chosen by the low
    // half, so without the key a lookup landing on a slot another n-gram filled returned that n-gram's token, at a rate
    // equal to the occupancy (up to the 25% the drafter resets at), and each such draft took a draft model's round
    struct slot {
        uint32_t key;
        entry_t  tok;
    };

    uint64_t hash(const entry_t * tokens) const;

    std::vector<slot> entries;
};
