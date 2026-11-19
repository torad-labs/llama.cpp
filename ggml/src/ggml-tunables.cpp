// The knob registry: every environment variable the engine reads is declared in tunables.toml, which
// scripts/tunables.py turns into ggml-tunables.inc, and read here, so a value is typed, range-checked and recorded.

#include "ggml.h"
#include "ggml-impl.h"

#include <atomic>
#include <cctype>
#include <cerrno>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#ifdef _WIN32
#    include <stdlib.h>
#    define GGML_TUNABLES_ENVIRON _environ
#else
extern char ** environ;
#    define GGML_TUNABLES_ENVIRON environ
#endif

namespace {

enum : uint8_t {
    FLAG_AUTO = 1, // the default depends on the device or the model, so the reader passes it
    FLAG_RAW  = 2, // still read with getenv somewhere: declared, not checked here
};

enum : uint8_t {
    SOURCE_NONE    = 0, // not read by this run
    SOURCE_DEFAULT = 1,
    SOURCE_AUTO    = 2,
    SOURCE_ENV     = 3,
};

struct knob_decl {
    const char * name;
    uint8_t      type; // ggml_tunable_type
    uint8_t      flags;
    double       def;
    double       min;
    double       max;
    const char * values; // the allowed values as a comma list, or NULL
};

#define BOOL  GGML_TUNABLE_BOOL
#define INT   GGML_TUNABLE_INT
#define FLOAT GGML_TUNABLE_FLOAT
#define STR   GGML_TUNABLE_STR
#define AUTO  FLAG_AUTO
#define RAW   FLAG_RAW

// sorted by name
const knob_decl g_knobs[] = {
#define GGML_TUNABLE(name, type, flags, def, min, max, values) { name, type, flags, def, min, max, values },
#define GGML_TUNABLE_IGNORE(prefix)
#include "ggml-tunables.inc"
#undef GGML_TUNABLE
#undef GGML_TUNABLE_IGNORE
};

// variables with these prefixes belong to someone else (the argument parser, other backends, the tests)
const char * const g_ignore[] = {
#define GGML_TUNABLE(name, type, flags, def, min, max, values)
#define GGML_TUNABLE_IGNORE(prefix) prefix,
#include "ggml-tunables.inc"
#undef GGML_TUNABLE
#undef GGML_TUNABLE_IGNORE
};

#undef BOOL
#undef INT
#undef FLOAT
#undef STR
#undef AUTO
#undef RAW

const size_t g_n_knobs = sizeof(g_knobs)/sizeof(g_knobs[0]);

struct knob_state {
    std::atomic<uint32_t> reads{0};
    std::atomic<uint8_t>  source{SOURCE_NONE};
    std::atomic<bool>     warned{false};
    std::atomic<int64_t>  ival{0};
    std::atomic<double>   fval{0.0};
};

knob_state g_state[sizeof(g_knobs)/sizeof(g_knobs[0])];

bool g_strict = false;

const char * type_name(uint8_t type) {
    switch (type) {
        case GGML_TUNABLE_BOOL:  return "bool";
        case GGML_TUNABLE_INT:   return "int";
        case GGML_TUNABLE_FLOAT: return "float";
        case GGML_TUNABLE_STR:   return "str";
    }
    return "?";
}

int find_knob(const char * name) {
    size_t lo = 0;
    size_t hi = g_n_knobs;
    while (lo < hi) {
        const size_t mid = lo + (hi - lo)/2;
        const int    cmp = strcmp(name, g_knobs[mid].name);
        if (cmp == 0) {
            return (int) mid;
        }
        if (cmp < 0) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    return -1;
}

// the value without the spaces around it: an env file with CRLF lines ends every value with \r
std::string trimmed(const char * value) {
    const char * end = value + strlen(value);
    while (value < end && isspace((unsigned char) *value)) {
        ++value;
    }
    while (end > value && isspace((unsigned char) end[-1])) {
        --end;
    }
    return std::string(value, end);
}

bool is_word(const std::string & value, const char * word) {
    if (value.size() != strlen(word)) {
        return false;
    }
    for (size_t i = 0; i < value.size(); ++i) {
        if (tolower((unsigned char) value[i]) != word[i]) {
            return false;
        }
    }
    return true;
}

// 1/0, true/false, yes/no, on/off in any case, or a number (on unless 0); empty is off
bool parse_bool(const std::string & value, bool * out) {
    for (const char * off : { "", "0", "false", "no", "off" }) {
        if (is_word(value, off)) {
            *out = false;
            return true;
        }
    }
    for (const char * on : { "1", "true", "yes", "on" }) {
        if (is_word(value, on)) {
            *out = true;
            return true;
        }
    }
    char * end = nullptr;
    const long number = strtol(value.c_str(), &end, 10);
    if (end != value.c_str() && *end == '\0') {
        *out = number != 0;
        return true;
    }
    return false;
}

bool parse_int(const std::string & value, int64_t * out) {
    char * end = nullptr;
    errno = 0;
    const long long number = strtoll(value.c_str(), &end, 10);
    if (end == value.c_str() || *end != '\0' || errno == ERANGE) {
        return false;
    }
    *out = number;
    return true;
}

bool parse_float(const std::string & value, double * out) {
    char * end = nullptr;
    errno = 0;
    const double number = strtod(value.c_str(), &end);
    if (end == value.c_str() || *end != '\0' || errno == ERANGE || !std::isfinite(number)) {
        return false;
    }
    *out = number;
    return true;
}

bool in_values(const char * values, const std::string & value) {
    const size_t n = value.size();
    for (const char * p = values; *p != '\0';) {
        const char * comma = strchr(p, ',');
        const size_t len   = comma != nullptr ? (size_t) (comma - p) : strlen(p);
        if (len == n && strncmp(p, value.c_str(), n) == 0) {
            return true;
        }
        p += len + (comma != nullptr ? 1 : 0);
    }
    return false;
}

// What is wrong with value for the knob, or empty. A lenient run then uses the nearest bound for a number out of
// range and "on" for a switch (usable stays true, with that in ival/fval), and takes anything else as unset.
std::string check_value(const knob_decl & k, const std::string & value, int64_t * ival, double * fval, bool * usable) {
    char buf[256];
    *usable = true;
    switch (k.type) {
        case GGML_TUNABLE_BOOL: {
            bool on = true;
            if (!parse_bool(value, &on)) {
                *ival = 1;
                return "is none of 1/0, true/false, yes/no, on/off: taken as on";
            }
            *ival = on ? 1 : 0;
            return "";
        }
        case GGML_TUNABLE_INT: {
            int64_t number = 0;
            if (!parse_int(value, &number)) {
                *usable = false;
                return "is not a whole number: taken as unset";
            }
            if (k.values != nullptr) {
                if (!in_values(k.values, std::to_string(number))) {
                    *usable = false;
                    snprintf(buf, sizeof(buf), "is not one of %s: taken as unset", k.values);
                    return buf;
                }
            } else if (number < (int64_t) k.min || number > (int64_t) k.max) {
                *ival = number < (int64_t) k.min ? (int64_t) k.min : (int64_t) k.max;
                snprintf(buf, sizeof(buf), "is outside %" PRId64 "..%" PRId64 ": %" PRId64 " is used", (int64_t) k.min, (int64_t) k.max, *ival);
                return buf;
            }
            *ival = number;
            return "";
        }
        case GGML_TUNABLE_FLOAT: {
            double number = 0.0;
            if (!parse_float(value, &number)) {
                *usable = false;
                return "is not a number: taken as unset";
            }
            if (number < k.min || number > k.max) {
                *fval = number < k.min ? k.min : k.max;
                snprintf(buf, sizeof(buf), "is outside %g..%g: %g is used", k.min, k.max, *fval);
                return buf;
            }
            *fval = number;
            return "";
        }
        case GGML_TUNABLE_STR: {
            if (k.values != nullptr && !in_values(k.values, value)) {
                *usable = false;
                snprintf(buf, sizeof(buf), "is not one of %s: taken as unset", k.values);
                return buf;
            }
            return "";
        }
    }
    return "";
}

// What a knob's value is checked as. A word is taken whole, since the reader gets the variable's own text back.
std::string knob_text(const knob_decl & k, const char * value) {
    return k.type == GGML_TUNABLE_STR ? std::string(value) : trimmed(value);
}

// a strict run ends here: straight to stderr, since the process is gone before a log thread would write it
[[noreturn]] void refuse(const char * text) {
    fflush(stdout);
    fprintf(stderr, "%s\nGGML_TUNABLES_STRICT is set: stopping. The knobs are listed in tunables.toml\n", text);
    fflush(stderr);
    _Exit(1);
}

bool is_ours(const char * name, size_t n) {
    return (n > 5 && strncmp(name, "GGML_", 5) == 0) || (n > 6 && strncmp(name, "LLAMA_", 6) == 0);
}

bool is_ignored(const std::string & name) {
    for (const char * prefix : g_ignore) {
        if (name.compare(0, strlen(prefix), prefix) == 0) {
            return true;
        }
    }
    return false;
}

// the engine's variables in the environment, as (name, value)
std::vector<std::pair<std::string, std::string>> our_environment() {
    std::vector<std::pair<std::string, std::string>> out;
    for (char ** e = GGML_TUNABLES_ENVIRON; e != nullptr && *e != nullptr; ++e) {
        const char * eq = strchr(*e, '=');
        if (eq == nullptr || !is_ours(*e, (size_t) (eq - *e))) {
            continue;
        }
        out.emplace_back(std::string(*e, (size_t) (eq - *e)), std::string(eq + 1));
    }
    return out;
}

void json_string(FILE * f, const std::string & s) {
    fputc('"', f);
    for (const char c : s) {
        if (c == '"' || c == '\\') {
            fprintf(f, "\\%c", c);
        } else if ((unsigned char) c < 0x20) {
            fprintf(f, "\\u%04x", c);
        } else {
            fputc(c, f);
        }
    }
    fputc('"', f);
}

void dump_at_exit() {
    const char * path = getenv("GGML_TUNABLES_DUMP");
    if (path != nullptr && path[0] != '\0') {
        ggml_tunables_dump(path);
    }
}

// The first read checks the whole environment, so a wrong value or a misspelt name shows when the process starts
// and not when the path that reads it first runs.
void init() {
    static std::once_flag once;
    std::call_once(once, [] {
        const char * strict = getenv("GGML_TUNABLES_STRICT");
        bool         on     = false;
        g_strict = strict != nullptr && parse_bool(trimmed(strict), &on) && on;

        std::string problems;
        for (const auto & [name, value] : our_environment()) {
            const int i = find_knob(name.c_str());
            if (i < 0) {
                if (!is_ignored(name)) {
                    problems += name + " is set, and no knob has that name\n";
                }
                continue;
            }
            if (g_knobs[i].flags & FLAG_RAW) {
                continue;
            }
            int64_t ival   = 0;
            double  fval   = 0.0;
            bool    usable = true;
            const std::string problem = check_value(g_knobs[i], knob_text(g_knobs[i], value.c_str()), &ival, &fval, &usable);
            if (!problem.empty()) {
                problems += name + "=" + value + " " + problem + "\n";
                g_state[i].warned = true;
            }
        }
        if (!problems.empty()) {
            problems.pop_back();
            if (g_strict) {
                refuse(problems.c_str());
            }
            GGML_LOG_WARN("%s\n", problems.c_str());
        }

        const char * dump = getenv("GGML_TUNABLES_DUMP");
        if (dump != nullptr && dump[0] != '\0') {
            atexit(dump_at_exit);
        }
    });
}

struct knob_read {
    bool    is_set; // the environment has a usable value
    int64_t ival;
    double  fval;
};

// reads a declared knob of the given type; the caller turns an unset one into its default
knob_read read_knob(const char * name, uint8_t type, bool want_auto, int * index) {
    init();
    const int i = find_knob(name);
    if (i < 0) {
        GGML_ABORT("%s is read as a knob and tunables.toml does not declare it", name);
    }
    const knob_decl & k = g_knobs[i];
    if (k.type != type || ((k.flags & FLAG_AUTO) != 0) != want_auto) {
        GGML_ABORT("%s is declared %s%s in tunables.toml and read as %s%s", name, type_name(k.type), (k.flags & FLAG_AUTO) ? " with default auto" : "",
                type_name(type), want_auto ? " with a default of the reader" : "");
    }
    *index = i;

    knob_read res = { false, (int64_t) k.def, k.def };
    const char * value = getenv(name);
    if (value == nullptr) {
        return res;
    }
    const std::string text = knob_text(k, value);
    // a variable set to nothing is unset, except for a switch, which it turns off
    if (text.empty() && type != GGML_TUNABLE_BOOL) {
        return res;
    }
    knob_read set = { true, res.ival, res.fval };
    const std::string problem = check_value(k, text, &set.ival, &set.fval, &set.is_set);
    if (problem.empty()) {
        return set;
    }
    if (g_strict) {
        refuse((std::string(name) + "=" + value + " " + problem).c_str());
    }
    if (!g_state[i].warned.exchange(true)) {
        GGML_LOG_WARN("%s=%s %s\n", name, value, problem.c_str());
    }
    return set.is_set ? set : res;
}

void record(int i, uint8_t source, int64_t ival, double fval) {
    knob_state & s = g_state[i];
    s.ival.store(ival, std::memory_order_relaxed);
    s.fval.store(fval, std::memory_order_relaxed);
    s.source.store(source, std::memory_order_relaxed);
    s.reads.fetch_add(1, std::memory_order_relaxed);
}

} // namespace

bool ggml_env_switch(const char * name) {
    // a name no knob declares keeps the plain reading: tests and tools have switches of their own
    if (find_knob(name) < 0) {
        const char * value = getenv(name);
        bool         on    = false;
        if (value == nullptr) {
            return false;
        }
        if (!parse_bool(trimmed(value), &on)) {
            GGML_LOG_WARN("%s=%s is none of 1/0, true/false, yes/no, on/off: taken as on\n", name, value);
            return true;
        }
        return on;
    }
    int i = 0;
    const knob_read r = read_knob(name, GGML_TUNABLE_BOOL, false, &i);
    const bool on = r.is_set ? r.ival != 0 : g_knobs[i].def != 0;
    record(i, r.is_set ? SOURCE_ENV : SOURCE_DEFAULT, on ? 1 : 0, 0.0);
    return on;
}

bool ggml_env_switch_or(const char * name, bool fallback) {
    int i = 0;
    const knob_read r = read_knob(name, GGML_TUNABLE_BOOL, true, &i);
    const bool on = r.is_set ? r.ival != 0 : fallback;
    record(i, r.is_set ? SOURCE_ENV : SOURCE_AUTO, on ? 1 : 0, 0.0);
    return on;
}

int64_t ggml_env_int(const char * name) {
    int i = 0;
    const knob_read r = read_knob(name, GGML_TUNABLE_INT, false, &i);
    record(i, r.is_set ? SOURCE_ENV : SOURCE_DEFAULT, r.ival, 0.0);
    return r.ival;
}

int64_t ggml_env_int_or(const char * name, int64_t fallback) {
    int i = 0;
    const knob_read r = read_knob(name, GGML_TUNABLE_INT, true, &i);
    const int64_t v = r.is_set ? r.ival : fallback;
    record(i, r.is_set ? SOURCE_ENV : SOURCE_AUTO, v, 0.0);
    return v;
}

double ggml_env_float(const char * name) {
    int i = 0;
    const knob_read r = read_knob(name, GGML_TUNABLE_FLOAT, false, &i);
    record(i, r.is_set ? SOURCE_ENV : SOURCE_DEFAULT, 0, r.fval);
    return r.fval;
}

double ggml_env_float_or(const char * name, double fallback) {
    int i = 0;
    const knob_read r = read_knob(name, GGML_TUNABLE_FLOAT, true, &i);
    const double v = r.is_set ? r.fval : fallback;
    record(i, r.is_set ? SOURCE_ENV : SOURCE_AUTO, 0, v);
    return v;
}

const char * ggml_env_str(const char * name) {
    int i = 0;
    const knob_read r = read_knob(name, GGML_TUNABLE_STR, false, &i);
    record(i, r.is_set ? SOURCE_ENV : SOURCE_DEFAULT, 0, 0.0);
    return r.is_set ? getenv(name) : nullptr;
}

size_t ggml_tunables_count(void) {
    return g_n_knobs;
}

bool ggml_tunables_get(size_t i, struct ggml_tunable * out) {
    if (i >= g_n_knobs || out == nullptr) {
        return false;
    }
    const knob_decl & k = g_knobs[i];
    out->name       = k.name;
    out->type       = (enum ggml_tunable_type) k.type;
    out->is_auto    = (k.flags & FLAG_AUTO) != 0;
    out->is_checked = (k.flags & FLAG_RAW) == 0;
    out->def        = k.def;
    out->min        = k.min;
    out->max        = k.max;
    out->values     = k.values;
    return true;
}

bool ggml_tunables_dump(const char * path) {
    init();
    FILE * f = fopen(path, "w");
    if (f == nullptr) {
        GGML_LOG_ERROR("%s: cannot write %s\n", __func__, path);
        return false;
    }

    // read: each knob this run read, the value it used and where the value came from (env, default, or auto for a
    // default the reader worked out)
    fprintf(f, "{\n\"strict\": %s,\n\"read\": [", g_strict ? "true" : "false");
    bool first = true;
    for (size_t i = 0; i < g_n_knobs; ++i) {
        const knob_state & s      = g_state[i];
        const uint8_t      source = s.source.load();
        if (source == SOURCE_NONE) {
            continue;
        }
        const knob_decl & k = g_knobs[i];
        fprintf(f, "%s\n  {\"name\": \"%s\", \"type\": \"%s\", \"value\": ", first ? "" : ",", k.name, type_name(k.type));
        first = false;
        switch (k.type) {
            case GGML_TUNABLE_BOOL:  fprintf(f, "%s", s.ival.load() != 0 ? "true" : "false"); break;
            case GGML_TUNABLE_INT:   fprintf(f, "%" PRId64, s.ival.load()); break;
            case GGML_TUNABLE_FLOAT: fprintf(f, "%.17g", s.fval.load()); break;
            case GGML_TUNABLE_STR: {
                const char * value = source == SOURCE_ENV ? getenv(k.name) : nullptr;
                if (value != nullptr) {
                    json_string(f, value);
                } else {
                    fprintf(f, "null");
                }
            } break;
        }
        fprintf(f, ", \"source\": \"%s\", \"reads\": %u}", source == SOURCE_ENV ? "env" : source == SOURCE_AUTO ? "auto" : "default", s.reads.load());
    }
    fprintf(f, "\n],\n");

    // unread: set and never read, so it changed nothing here. unchecked: a knob still read with getenv, taken as set.
    const auto env = our_environment();
    for (const char * section : { "unread", "unchecked" }) {
        fprintf(f, "\"%s\": {", section);
        first = true;
        for (const auto & [name, value] : env) {
            const int i = find_knob(name.c_str());
            const char * is = i < 0 ? "" : (g_knobs[i].flags & FLAG_RAW) ? "unchecked" : g_state[i].source.load() == SOURCE_NONE ? "unread" : "";
            if (strcmp(is, section) != 0) {
                continue;
            }
            fprintf(f, "%s\n  ", first ? "" : ",");
            first = false;
            json_string(f, name);
            fprintf(f, ": ");
            json_string(f, value);
        }
        fprintf(f, "%s},\n", first ? "" : "\n");
    }
    // unknown: no knob has the name. Names only: what an undeclared variable holds may be a secret.
    fprintf(f, "\"unknown\": [");
    first = true;
    for (const auto & [name, value] : env) {
        if (find_knob(name.c_str()) >= 0 || is_ignored(name)) {
            continue;
        }
        fprintf(f, "%s", first ? "" : ", ");
        first = false;
        json_string(f, name);
    }
    fprintf(f, "]\n");
    fprintf(f, "}\n");
    fclose(f);
    return true;
}
