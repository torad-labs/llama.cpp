// utf8_replace_malformed (tools/server/server-common.cpp): the server's repair of generated text before the chat parser
// reads it. The parser's JSON dump is strict, so a sequence this lets through as well-formed answers the request 500.
// Each case is the input, the `from` it is read from, the text it must leave and where it says the incomplete tail starts.

#include "server-common.h"

#include <cstdio>
#include <string>

static int failures = 0;

static std::string show(const std::string & s) {
    std::string out;
    char buf[8];
    for (unsigned char c : s) {
        snprintf(buf, sizeof(buf), c >= 0x20 && c < 0x7F ? "%c" : "\\x%02X", c);
        out += buf;
    }
    return out;
}

static void check(const char * name, const std::string & in, size_t from, const std::string & want, size_t want_ret) {
    std::string text = in;
    const size_t ret = utf8_replace_malformed(text, from);
    if (text != want || ret != want_ret) {
        fprintf(stderr, "FAIL %s: \"%s\" -> \"%s\" (%zu), want \"%s\" (%zu)\n", name, show(in).c_str(),
                show(text).c_str(), ret, show(want).c_str(), want_ret);
        failures++;
    } else {
        printf("ok   %s\n", name);
    }
}

int main() {
    const std::string R = "\xEF\xBF\xBD"; // U+FFFD

    // well-formed text is left alone, every length and range edge included
    const std::string good = "a\xC2\x80\xDF\xBF\xE0\xA0\x80\xED\x9F\xBF\xEE\x80\x80\xF0\x90\x80\x80\xF4\x8F\xBF\xBF";
    check("well-formed", good, 0, good, good.size());

    // the shapes the pattern check already caught
    check("stray continuation", "a\x80" "b", 0, "a" + R + "b", 3 + 2);
    check("lead then ascii", "\xC3" "A", 0, R + "A", 4);
    check("lead then lead", "\xE4\xE4\xE4", 0, R + R + "\xE4", 6);

    // what a pattern check let through as well-formed, each one U+FFFD for its maximal subpart (Unicode 3.9, U+FFFD
    // substitution of maximal subparts): a byte that leads nothing, or a second byte outside its lead's range
    check("surrogate", "\xED\xA0\x80", 0, R + R + R, 9);
    check("overlong 2", "\xC0\x80", 0, R + R, 6);
    check("overlong 3", "\xE0\x80\x80", 0, R + R + R, 9);
    check("overlong 4", "\xF0\x80\x80\x80", 0, R + R + R + R, 12);
    check("past U+10FFFF", "\xF4\x90\x80\x80", 0, R + R + R + R, 12);
    check("lead F5", "\xF5\x80\x80\x80", 0, R + R + R + R, 12);
    check("lead FF", "x\xFF" "y", 0, "x" + R + "y", 5);

    // a maximal subpart is one U+FFFD, and the byte that ended it is read afresh
    check("subpart of two", "\xE4\xB8" "A", 0, R + "A", 4);
    check("subpart of three", "\xF0\x9F\x98" "A", 0, R + "A", 4);

    // an incomplete tail is kept back only while every byte of it is right so far
    check("tail of a 3-byte", "ok\xE4\xB8", 0, "ok\xE4\xB8", 2);
    check("tail E0 A0", "\xE0\xA0", 0, "\xE0\xA0", 0);
    check("not a tail: E0 80", "\xE0\x80", 0, R + R, 6);
    check("not a tail: ED A0", "\xED\xA0", 0, R + R, 6);

    // `from` reads only the unsent part; what is before it is not looked at again
    check("from", "\x80" "ab\x80", 3, "\x80" "ab" + R, 6);

    if (failures > 0) {
        fprintf(stderr, "%d case(s) failed\n", failures);
        return 1;
    }
    return 0;
}
