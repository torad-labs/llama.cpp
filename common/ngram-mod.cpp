#include "ngram-mod.h"

#include <algorithm>

//
// common_ngram_mod
//

common_ngram_mod::common_ngram_mod(uint16_t n, size_t size) : n(n), used(0) {
    entries.resize(size);

    reset();
}

uint64_t common_ngram_mod::hash(const entry_t * tokens) const {
    uint64_t res = 0;

    for (size_t i = 0; i < n; ++i) {
        res = res*6364136223846793005ULL + tokens[i];
    }

    return res;
}

size_t common_ngram_mod::idx(const entry_t * tokens) const {
    return hash(tokens) % entries.size();
}

void common_ngram_mod::add(const entry_t * tokens) {
    const uint64_t h = hash(tokens);
    slot & s = entries[h % entries.size()];

    if (s.tok == EMPTY) {
        used++;
    }

    s = { (uint32_t) (h >> 32), tokens[n] };
}

common_ngram_mod::entry_t common_ngram_mod::get(const entry_t * tokens) const {
    const uint64_t h = hash(tokens);
    const slot & s = entries[h % entries.size()];

    return s.key == (uint32_t) (h >> 32) ? s.tok : EMPTY;
}

void common_ngram_mod::reset() {
    std::fill(entries.begin(), entries.end(), slot { 0, EMPTY });
    used = 0;
}

size_t common_ngram_mod::get_n() const {
    return n;
}

size_t common_ngram_mod::get_used() const {
    return used;
}

size_t common_ngram_mod::size() const {
    return entries.size();
}

size_t common_ngram_mod::size_bytes() const {
    return entries.size() * sizeof(entries[0]);
}
