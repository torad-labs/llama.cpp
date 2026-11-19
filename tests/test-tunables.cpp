// The knobs of tunables.toml as a search sets them: a knob reads its declared default when unset and any value in its
// range when set, a value outside the range or a variable no knob declares is refused under GGML_TUNABLES_STRICT=1,
// and GGML_TUNABLES_DUMP records what the run read. The cases run over every knob the registry checks.

#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#ifdef _WIN32

int main(void) {
    fprintf(stderr, "test-tunables : skipping on windows, which has no setenv\n");
    return 0;
}

#else

#include <sys/wait.h>
#include <unistd.h>

static int n_fail = 0;

#define CHECK(cond, ...)                              \
    do {                                              \
        if (!(cond)) {                                \
            fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
            fprintf(stderr, __VA_ARGS__);             \
            fprintf(stderr, "\n");                    \
            ++n_fail;                                 \
        }                                             \
    } while (0)

static std::vector<std::string> split_values(const char * values) {
    std::vector<std::string> out;
    std::stringstream ss(values != nullptr ? values : "");
    for (std::string item; std::getline(ss, item, ',');) {
        out.push_back(item);
    }
    return out;
}

// what a reader of the knob's own type returns, as text
static std::string read_knob(const ggml_tunable & t) {
    char buf[64];
    switch (t.type) {
        case GGML_TUNABLE_BOOL:
            return (t.is_auto ? ggml_env_switch_or(t.name, false) : ggml_env_switch(t.name)) ? "1" : "0";
        case GGML_TUNABLE_INT:
            snprintf(buf, sizeof(buf), "%lld", (long long) (t.is_auto ? ggml_env_int_or(t.name, -7) : ggml_env_int(t.name)));
            return buf;
        case GGML_TUNABLE_FLOAT:
            snprintf(buf, sizeof(buf), "%.17g", t.is_auto ? ggml_env_float_or(t.name, -7.5) : ggml_env_float(t.name));
            return buf;
        case GGML_TUNABLE_STR: {
            const char * s = ggml_env_str(t.name);
            return s != nullptr ? s : "(unset)";
        }
    }
    return "?";
}

struct child_result {
    int         status;
    std::string output;
};

// runs this program again as `--read <knob>` with the variables of env set, and returns what it printed and how it ended
static child_result run_child(const std::string & self, const std::string & env, const std::string & knob) {
    const std::string cmd = "env " + env + " '" + self + "' --read " + knob + " 2>&1";
    child_result res = { -1, "" };
    FILE * pipe = popen(cmd.c_str(), "r");
    if (pipe == nullptr) {
        return res;
    }
    char buf[512];
    while (fgets(buf, sizeof(buf), pipe) != nullptr) {
        res.output += buf;
    }
    const int rc = pclose(pipe);
    res.status = WIFEXITED(rc) ? WEXITSTATUS(rc) : 128;
    return res;
}

static const ggml_tunable * find_knob(const std::vector<ggml_tunable> & knobs, const char * name) {
    for (const auto & t : knobs) {
        if (strcmp(t.name, name) == 0) {
            return &t;
        }
    }
    return nullptr;
}

int main(int argc, char ** argv) {
    std::vector<ggml_tunable> knobs(ggml_tunables_count());
    for (size_t i = 0; i < knobs.size(); ++i) {
        if (!ggml_tunables_get(i, &knobs[i])) {
            fprintf(stderr, "FAIL: knob %zu of %zu is not listed\n", i, knobs.size());
            return 1;
        }
    }

    if (argc == 3 && strcmp(argv[1], "--read") == 0) {
        const ggml_tunable * t = find_knob(knobs, argv[2]);
        if (t == nullptr) {
            fprintf(stderr, "no knob %s\n", argv[2]);
            return 2;
        }
        printf("value=%s\n", read_knob(*t).c_str());
        return 0;
    }

    const std::string self = argv[0];

    const ggml_tunable * an_int = nullptr; // a checked int knob with a range and a declared default
    size_t n_checked = 0;

    for (const auto & t : knobs) {
        if (!t.is_checked) {
            continue;
        }
        ++n_checked;

        // a knob left unset reads its declared default
        unsetenv(t.name);
        const std::string unset = read_knob(t);
        switch (t.type) {
            case GGML_TUNABLE_BOOL:
                CHECK(unset == (t.is_auto ? "0" : (t.def != 0 ? "1" : "0")), "%s unset reads %s", t.name, unset.c_str());
                break;
            case GGML_TUNABLE_INT:
                CHECK(strtoll(unset.c_str(), nullptr, 10) == (t.is_auto ? -7 : (long long) t.def), "%s unset reads %s", t.name, unset.c_str());
                break;
            case GGML_TUNABLE_FLOAT:
                CHECK(strtod(unset.c_str(), nullptr) == (t.is_auto ? -7.5 : t.def), "%s unset reads %s", t.name, unset.c_str());
                break;
            case GGML_TUNABLE_STR:
                CHECK(unset == "(unset)", "%s unset reads %s", t.name, unset.c_str());
                break;
        }

        // every value a knob allows is read back as it was set
        std::vector<std::string> allowed = split_values(t.values);
        char buf[64];
        if (allowed.empty() && t.type == GGML_TUNABLE_INT) {
            snprintf(buf, sizeof(buf), "%lld", (long long) t.min);
            allowed.push_back(buf);
            snprintf(buf, sizeof(buf), "%lld", (long long) t.max);
            allowed.push_back(buf);
        }
        if (allowed.empty() && t.type == GGML_TUNABLE_FLOAT) {
            snprintf(buf, sizeof(buf), "%.17g", t.min);
            allowed.push_back(buf);
            snprintf(buf, sizeof(buf), "%.17g", t.max);
            allowed.push_back(buf);
        }
        for (const auto & value : allowed) {
            setenv(t.name, value.c_str(), 1);
            const std::string got = read_knob(t);
            if (t.type == GGML_TUNABLE_FLOAT) {
                CHECK(strtod(got.c_str(), nullptr) == strtod(value.c_str(), nullptr), "%s=%s reads %s", t.name, value.c_str(), got.c_str());
            } else {
                CHECK(got == value, "%s=%s reads %s", t.name, value.c_str(), got.c_str());
            }
        }
        if (t.type == GGML_TUNABLE_BOOL) {
            for (const char * on : { "1", "true", "YES", " on " }) {
                setenv(t.name, on, 1);
                CHECK(read_knob(t) == "1", "%s=%s reads off", t.name, on);
            }
            for (const char * off : { "0", "false", "No", "off\r" }) {
                setenv(t.name, off, 1);
                CHECK(read_knob(t) == "0", "%s=%s reads on", t.name, off);
            }
        }
        unsetenv(t.name);

        if (an_int == nullptr && t.type == GGML_TUNABLE_INT && t.values == nullptr && !t.is_auto && t.max > t.min && t.max < 1e15) {
            an_int = &t;
        }
    }

    CHECK(n_checked > 0, "the registry checks no knob");
    CHECK(an_int != nullptr, "no int knob with a range to test refusals on");

    if (an_int != nullptr) {
        char above[96];
        snprintf(above, sizeof(above), "%s=%lld", an_int->name, (long long) an_int->max + 1);
        char top[64];
        snprintf(top, sizeof(top), "value=%lld\n", (long long) an_int->max);

        // a value above a knob's range is replaced by the bound and named in a warning
        const child_result lenient = run_child(self, above, an_int->name);
        CHECK(lenient.status == 0, "%s ended with %d: %s", above, lenient.status, lenient.output.c_str());
        CHECK(lenient.output.find(top) != std::string::npos, "%s did not read the upper bound: %s", above, lenient.output.c_str());
        CHECK(lenient.output.find(an_int->name) != std::string::npos, "%s was replaced without a word: %s", above, lenient.output.c_str());

        // a strict run refuses it
        const child_result strict = run_child(self, std::string("GGML_TUNABLES_STRICT=1 ") + above, an_int->name);
        CHECK(strict.status != 0, "a strict run took %s: %s", above, strict.output.c_str());
        CHECK(strict.output.find(an_int->name) != std::string::npos, "a strict run refused %s without naming it: %s", above, strict.output.c_str());

        // a value that is no number is refused the same way
        const child_result junk = run_child(self, std::string("GGML_TUNABLES_STRICT=1 ") + an_int->name + "=many", an_int->name);
        CHECK(junk.status != 0, "a strict run took %s=many: %s", an_int->name, junk.output.c_str());

        // a variable with the engine's prefix that no knob declares is named, and stops a strict run
        const child_result typo = run_child(self, "GGML_NO_SUCH_KNOB=1", an_int->name);
        CHECK(typo.status == 0, "an undeclared variable ended the run with %d: %s", typo.status, typo.output.c_str());
        CHECK(typo.output.find("GGML_NO_SUCH_KNOB") != std::string::npos, "an undeclared variable went unreported: %s", typo.output.c_str());
        const child_result typo_strict = run_child(self, "GGML_TUNABLES_STRICT=1 LLAMA_NO_SUCH_KNOB=1", an_int->name);
        CHECK(typo_strict.status != 0, "a strict run took an undeclared variable: %s", typo_strict.output.c_str());
        CHECK(typo_strict.output.find("LLAMA_NO_SUCH_KNOB") != std::string::npos, "a strict run did not name the undeclared variable: %s", typo_strict.output.c_str());

        // the dump records the knob a run read, the value it used and that it came from the environment
        const std::string dump_path = "/tmp/test-tunables-" + std::to_string((long long) getpid()) + ".json";
        char inside[96];
        snprintf(inside, sizeof(inside), "%s=%lld", an_int->name, (long long) an_int->min);
        const child_result dumped = run_child(self, "GGML_TUNABLES_DUMP=" + dump_path + " " + inside, an_int->name);
        CHECK(dumped.status == 0, "the dumping run ended with %d: %s", dumped.status, dumped.output.c_str());
        std::ifstream in(dump_path);
        std::stringstream text;
        text << in.rdbuf();
        const std::string dump = text.str();
        char record[256];
        snprintf(record, sizeof(record), "{\"name\": \"%s\", \"type\": \"int\", \"value\": %lld, \"source\": \"env\"", an_int->name, (long long) an_int->min);
        CHECK(dump.find(record) != std::string::npos, "the dump has no record %s: %s", record, dump.c_str());
        remove(dump_path.c_str());
    }

    if (n_fail == 0) {
        printf("test-tunables: %zu knobs listed, %zu checked: OK\n", knobs.size(), n_checked);
    }
    return n_fail == 0 ? 0 : 1;
}

#endif // _WIN32
