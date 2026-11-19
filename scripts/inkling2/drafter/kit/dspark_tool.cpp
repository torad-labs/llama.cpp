// dspark_tool: records what Inkling does on Claude Code requests, to train its draft head from.
//
//   dspark_tool prepare [options] -- [llama-server flags]
//   dspark_tool capture [options] -- [llama-server flags]
//
// Both turn an Anthropic Messages body into prompt tokens and sampler settings on the path llama-server takes for
// /v1/messages (conversion, chat template, tool-call grammar, tokenizer), so the recording sees what a live session does.
//
// prepare loads the vocabulary alone. For every request it writes the token count, the positions where a model turn
// begins (cuts) and the length of the prefix it shares with a donor request.
//
// capture loads the model as the server would and runs a plan: every request is read through all slots at once, the
// hidden states of the tapped layers are written for every position, and at the cuts the plan chose the model writes
// its own continuation, which is recorded with its states and the top of its distribution, then rewound.

#include "arg.h"
#include "chat.h"
#include "common.h"
#include "json.h"
#include "llama.h"
#include "log.h"
#include "sampling.h"
#include "speculative.h"

#include "server-chat.h"
#include "server-common.h"
#include "server-schema.h"
#include "server-task.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <sys/stat.h>
#include <unistd.h>

// ---------------------------------------------------------------------------------------------------------------------
// small helpers
// ---------------------------------------------------------------------------------------------------------------------

static double now_s() {
    using clock = std::chrono::steady_clock;
    static const clock::time_point t0 = clock::now();
    return std::chrono::duration<double>(clock::now() - t0).count();
}

static std::mutex g_say_mutex;

static void say(const char * fmt, ...) {
    std::lock_guard<std::mutex> lock(g_say_mutex);
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "[dspark %9.1f] ", now_s());
    vfprintf(stderr, fmt, ap);
    fprintf(stderr, "\n");
    va_end(ap);
    fflush(stderr);
}

[[noreturn]] static void dsp_die(const char * fmt, ...) {
    {
        std::lock_guard<std::mutex> lock(g_say_mutex);
        va_list ap;
        va_start(ap, fmt);
        fprintf(stderr, "[dspark %9.1f] FATAL: ", now_s());
        vfprintf(stderr, fmt, ap);
        fprintf(stderr, "\n");
        va_end(ap);
        fflush(stderr);
    }
    std::_Exit(1);
}

static std::string jq(const std::string & s) {
    return json::make(s).dump_safe();
}

static std::string ints_json(const std::vector<int> & v) {
    std::string out = "[";
    for (size_t i = 0; i < v.size(); i++) {
        if (i > 0) {
            out += ",";
        }
        out += std::to_string(v[i]);
    }
    out += "]";
    return out;
}

static void parallel_for(int n_threads, size_t n, const std::function<void(size_t, size_t)> & fn) {
    if (n == 0) {
        return;
    }
    const size_t nt = std::min<size_t>((size_t) std::max(1, n_threads), n);
    if (nt == 1) {
        fn(0, n);
        return;
    }
    const size_t per = (n + nt - 1) / nt;
    std::vector<std::thread> workers;
    workers.reserve(nt - 1);
    for (size_t t = 1; t < nt; t++) {
        const size_t b = t * per;
        const size_t e = std::min(n, b + per);
        if (b >= e) {
            break;
        }
        workers.emplace_back([&fn, b, e] { fn(b, e); });
    }
    fn(0, std::min(n, per));
    for (auto & w : workers) {
        w.join();
    }
}

static std::vector<std::string> read_lines(const std::vector<std::string> & paths) {
    std::vector<std::string> out;
    for (const auto & path : paths) {
        std::ifstream in(path);
        if (!in) {
            dsp_die("cannot open %s", path.c_str());
        }
        std::string line;
        while (std::getline(in, line)) {
            if (!line.empty()) {
                out.push_back(std::move(line));
            }
        }
    }
    return out;
}

static long long file_size(const std::string & path) {
    struct stat st;
    if (stat(path.c_str(), &st) != 0) {
        return -1;
    }
    return (long long) st.st_size;
}

static void make_dir(const std::string & path) {
    if (mkdir(path.c_str(), 0755) != 0 && errno != EEXIST) {
        dsp_die("cannot create folder %s: %s", path.c_str(), strerror(errno));
    }
}

static FILE * open_file(const std::string & path, const char * mode) {
    FILE * f = fopen(path.c_str(), mode);
    if (f == nullptr) {
        dsp_die("cannot open %s (%s): %s", path.c_str(), mode, strerror(errno));
    }
    return f;
}

static inline uint16_t f32_to_bf16(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, 4);
    const uint32_t lsb = (bits >> 16) & 1;
    bits += 0x7fff + lsb;
    return (uint16_t) (bits >> 16);
}

static inline void row_to_bf16(const float * src, uint16_t * dst, int n) {
    for (int i = 0; i < n; i++) {
        dst[i] = f32_to_bf16(src[i]);
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// options
// ---------------------------------------------------------------------------------------------------------------------

struct tool_opts {
    std::string mode;
    std::vector<std::string> requests;
    std::string plan;
    std::string out;
    std::string donor;
    int  max_tokens      = 200000;
    int  threads         = 8;
    std::vector<int> taps = {5, 11, 17, 23, 29, 35, 41, 48};
    int  topk            = 64;
    bool no_cuts         = false;
    bool stops           = false;
    bool trace           = false;
    bool verify          = false;
    int  trip_every      = 0;
    int  step_rows       = 0;
    int  max_requests    = -1;
    int  margin          = 64;
    int  show_cuts       = 0;
    int  log_secs        = 30;
    int  write_buffer_gb = 8;
};

static void usage() {
    fprintf(stderr,
        "usage: dspark_tool <prepare|capture> [options] -- [llama-server flags]\n"
        "  --requests FILE      a requests jsonl ({id, source, prefix, body}, or {id, tokens} for a request with no cut in the\n"
        "                       plan); may be given more than once\n"
        "  --out PATH           prepare: the jsonl it writes; capture: the folder of the recording\n"
        "  --plan FILE          capture: the plan jsonl\n"
        "  --donor ID           prepare: the request whose prefix the others are compared with\n"
        "  --max-tokens N       prepare: a longer request is cut at its last model turn that starts within N (200000)\n"
        "  --threads N          host threads for parsing, conversion and sampling (8)\n"
        "  --taps A,B,...       the layer inputs recorded (5,11,17,23,29,35,41,48)\n"
        "  --topk K             how much of each distribution is kept (64)\n"
        "  --no-cuts            capture: read the requests only, nothing generated\n"
        "  --stops              capture: with --no-cuts, still end a read at every cut of the plan, as a run with cuts does\n"
        "  --trace              capture: write trace.txt, one line for every part of every prompt batch\n"
        "  --verify-states      capture: look at every sequence's saved state around every decode and write verify.txt:\n"
        "                       a sequence the decode did not hold, and the tool did not load or cut, keeps its bytes\n"
        "  --trip-every N       capture: at every Nth cut, the state is read again after the way back, and the run stops\n"
        "                       when it is not the state saved before the continuation\n"
        "  --step-rows N        capture: the prompt rows one decode holds at most (the micro-batch, as the server's own\n"
        "                       decodes; more is a probe's setting, not a recording's)\n"
        "  --max-requests N     capture: only the first N requests of the plan\n"
        "  --margin N           capture: spare cells kept per request in the shared pool (64)\n"
        "  --show-cuts N        prepare: print the tokens around the donor's first N cuts\n"
        "  --log-secs N         capture: seconds between progress lines (30)\n"
        "  --write-buffer-gb N  capture: how much unwritten data may wait in memory (8)\n");
}

static std::vector<int> parse_int_list(const std::string & text) {
    std::vector<int> out;
    size_t start = 0;
    while (start <= text.size()) {
        const size_t comma = text.find(',', start);
        const std::string part = text.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
        if (!part.empty()) {
            out.push_back(std::stoi(part));
        }
        if (comma == std::string::npos) {
            break;
        }
        start = comma + 1;
    }
    return out;
}

static bool parse_tool_opts(int argc, char ** argv, tool_opts & opt) {
    if (argc < 1) {
        return false;
    }
    opt.mode = argv[0];
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        auto need = [&](const char * name) -> std::string {
            if (i + 1 >= argc) {
                dsp_die("%s needs a value", name);
            }
            return argv[++i];
        };
        if (a == "--requests") {
            opt.requests.push_back(need("--requests"));
        } else if (a == "--out") {
            opt.out = need("--out");
        } else if (a == "--plan") {
            opt.plan = need("--plan");
        } else if (a == "--donor") {
            opt.donor = need("--donor");
        } else if (a == "--max-tokens") {
            opt.max_tokens = std::stoi(need("--max-tokens"));
        } else if (a == "--threads") {
            opt.threads = std::stoi(need("--threads"));
        } else if (a == "--taps") {
            opt.taps = parse_int_list(need("--taps"));
        } else if (a == "--topk") {
            opt.topk = std::stoi(need("--topk"));
        } else if (a == "--no-cuts") {
            opt.no_cuts = true;
        } else if (a == "--stops") {
            opt.stops = true;
        } else if (a == "--trace") {
            opt.trace = true;
        } else if (a == "--verify-states") {
            opt.verify = true;
        } else if (a == "--trip-every") {
            opt.trip_every = std::stoi(need("--trip-every"));
        } else if (a == "--step-rows") {
            opt.step_rows = std::stoi(need("--step-rows"));
        } else if (a == "--max-requests") {
            opt.max_requests = std::stoi(need("--max-requests"));
        } else if (a == "--margin") {
            opt.margin = std::stoi(need("--margin"));
        } else if (a == "--show-cuts") {
            opt.show_cuts = std::stoi(need("--show-cuts"));
        } else if (a == "--log-secs") {
            opt.log_secs = std::stoi(need("--log-secs"));
        } else if (a == "--write-buffer-gb") {
            opt.write_buffer_gb = std::stoi(need("--write-buffer-gb"));
        } else {
            fprintf(stderr, "unknown option: %s\n", a.c_str());
            return false;
        }
    }
    return opt.mode == "prepare" || opt.mode == "capture";
}

// ---------------------------------------------------------------------------------------------------------------------
// the server's own path from a request body to prompt tokens and sampler settings
// ---------------------------------------------------------------------------------------------------------------------

struct chat_env {
    const llama_vocab *   vocab  = nullptr;
    const common_params * params = nullptr;
    server_chat_params    chat_params;
    llama_token           tok_model = LLAMA_TOKEN_NULL; // <|message_model|>
    llama_token           tok_end   = LLAMA_TOKEN_NULL; // <|content_model_end_sampling|>
    int                   max_tokens = 0;
};

static llama_token single_token(const llama_vocab * vocab, const char * text) {
    const std::vector<llama_token> toks = common_tokenize(vocab, text, false, true);
    if (toks.size() != 1) {
        dsp_die("%s is %zu tokens in this vocabulary, expected 1", text, toks.size());
    }
    return toks[0];
}

// as server_context::load_model fills it, for a server with no projector
static void init_chat_env(chat_env & env, const llama_model * model, const common_params & params, int max_tokens) {
    env.vocab      = llama_model_get_vocab(model);
    env.params     = &params;
    env.max_tokens = max_tokens;

    common_chat_templates_ptr tmpls = common_chat_templates_init(model, params.chat_template);
    const bool supports_thinking = common_chat_templates_support_enable_thinking(tmpls.get());
    const json caps = common_chat_templates_get_caps(tmpls.get());

    server_chat_params & cp = env.chat_params;
    cp.use_jinja                = params.use_jinja;
    cp.prefill_assistant        = params.prefill_assistant;
    cp.reasoning_format         = params.reasoning_format;
    cp.chat_template_kwargs     = params.default_template_kwargs;
    cp.tmpls                    = std::move(tmpls);
    cp.allow_image              = false;
    cp.allow_audio              = false;
    cp.allow_video              = false;
    cp.enable_thinking          = params.enable_reasoning != 0 && params.use_jinja && supports_thinking;
    cp.reasoning_budget         = params.sampling.reasoning_budget_tokens;
    cp.reasoning_budget_message = params.sampling.reasoning_budget_message;
    cp.media_path               = params.media_path;
    cp.media_allow_private_urls = params.media_allow_private_urls;
    cp.force_pure_content       = params.force_pure_content_parser;

    if (params.enable_reasoning == 0 && caps.contains("reasoning_effort_none") && caps.at("reasoning_effort_none").get<bool>()) {
        cp.chat_template_kwargs.emplace("enable_thinking", "false");
    }

    env.tok_model = single_token(env.vocab, "<|message_model|>");
    env.tok_end   = single_token(env.vocab, "<|content_model_end_sampling|>");
}

struct parsed_request {
    llama_tokens     tokens;            // what the model reads, cut to max_tokens
    int              n_raw     = 0;     // the token count before that cut
    bool             truncated = false;
    std::vector<int> cuts;              // positions where a model turn begins; n_tokens when the request ends before one
    common_params_sampling sampling;    // the sampler settings the server would give this request's slot
    bool             grammar   = false;
    int              n_stops   = 0;
    std::string      error;
};

static parsed_request parse_request(const chat_env & env, const json & body_anthropic) {
    parsed_request out;
    try {
        json body = server_chat_convert_anthropic_to_oai(body_anthropic);

        std::vector<raw_buffer> files;
        common_chat_session     session;
        json parsed = oaicompat_chat_params_parse(env.vocab, body, env.chat_params, files, session);

        task_params tp = server_schema::eval_llama_cmpl_schema(env.vocab, *env.params, env.params->sampling.logit_bias_eog, parsed);

        // server_task::apply_chat_session()
        if (session.has_template()) {
            session.apply_sampling(tp.sampling);
        }

        out.sampling = tp.sampling;
        out.grammar  = !tp.sampling.grammar.grammar.empty();
        out.n_stops  = (int) session.additional_stops().size();

        llama_tokens toks = tokenize_mixed(env.vocab, parsed.at("prompt"), true, true);
        out.n_raw = (int) toks.size();

        bool inside = false;
        for (int i = 0; i < (int) toks.size(); i++) {
            if (toks[i] == env.tok_model) {
                if (!inside) {
                    if (i > 0) {
                        out.cuts.push_back(i);
                    }
                    inside = true;
                }
            } else if (toks[i] == env.tok_end) {
                inside = false;
            }
        }
        if (!inside) {
            out.cuts.push_back((int) toks.size());
        }

        if (env.max_tokens > 0 && (int) toks.size() > env.max_tokens) {
            int keep = -1;
            for (int c : out.cuts) {
                if (c <= env.max_tokens) {
                    keep = c;
                }
            }
            if (keep <= 0) {
                out.error = "no model turn starts within max_tokens";
                return out;
            }
            toks.resize((size_t) keep);
            std::vector<int> kept;
            for (int c : out.cuts) {
                if (c <= keep) {
                    kept.push_back(c);
                }
            }
            out.cuts      = kept;
            out.truncated = true;
        }

        out.tokens = std::move(toks);
    } catch (const std::exception & e) {
        out.error = e.what();
        if (out.error.empty()) {
            out.error = "exception";
        }
    }
    return out;
}

// ---------------------------------------------------------------------------------------------------------------------
// prepare
// ---------------------------------------------------------------------------------------------------------------------

struct prepared_row {
    std::string id, source, prefix;
    int  n_tokens  = 0;
    int  n_raw     = 0;
    bool truncated = false;
    int  lcp       = 0;
    bool grammar   = false;
    int  n_stops   = 0;
    std::vector<int> cuts;
    std::string error;
};

static int run_prepare(const tool_opts & opt, common_params & params) {
    if (opt.requests.empty() || opt.out.empty()) {
        dsp_die("prepare needs --requests and --out");
    }

    llama_model_params mparams = llama_model_default_params();
    mparams.vocab_only = true;
    llama_model * model = llama_model_load_from_file(params.model.path.c_str(), mparams);
    if (model == nullptr) {
        dsp_die("cannot load the vocabulary of %s", params.model.path.c_str());
    }

    chat_env env;
    init_chat_env(env, model, params, opt.max_tokens);

    const std::vector<std::string> lines = read_lines(opt.requests);
    say("prepare: %zu requests, <|message_model|> = %d, <|content_model_end_sampling|> = %d", lines.size(), env.tok_model, env.tok_end);

    // the donor: the request named, else the first one whose prefix class carries the system prompt
    std::string    donor_id;
    parsed_request donor;
    for (size_t i = 0; i < lines.size() && donor_id.empty(); i++) {
        const json line = json::parse(lines[i]);
        const std::string id     = line.at("id").get<std::string>();
        const std::string prefix = line.value("prefix", std::string());
        if (opt.donor.empty() ? prefix == "cc25-system" : id == opt.donor) {
            donor = parse_request(env, line.at("body"));
            if (!donor.error.empty()) {
                dsp_die("the donor %s does not parse: %s", id.c_str(), donor.error.c_str());
            }
            donor_id = id;
        }
    }
    if (donor_id.empty()) {
        dsp_die("no donor request found");
    }
    say("prepare: donor %s, %zu tokens, %zu cuts, grammar %d", donor_id.c_str(), donor.tokens.size(), donor.cuts.size(), donor.grammar ? 1 : 0);

    if (opt.show_cuts > 0) {
        for (int k = 0; k < opt.show_cuts && k < (int) donor.cuts.size(); k++) {
            const int c = donor.cuts[k];
            std::string text;
            for (int i = std::max(0, c - 8); i < std::min((int) donor.tokens.size(), c + 8); i++) {
                if (i == c) {
                    text += " >>CUT<< ";
                }
                text += "[" + common_token_to_piece(env.vocab, donor.tokens[i], true) + "]";
            }
            say("prepare: donor cut %d at %d: %s", k, c, jq(text).c_str());
        }
        std::string head;
        for (int i = 0; i < std::min((int) donor.tokens.size(), 12); i++) {
            head += "[" + common_token_to_piece(env.vocab, donor.tokens[i], true) + "]";
        }
        say("prepare: donor head: %s", jq(head).c_str());
    }

    std::vector<prepared_row> rows(lines.size());
    std::atomic<size_t> next{0};
    std::atomic<size_t> done{0};
    const int n_workers = std::max(1, opt.threads);
    std::vector<std::thread> workers;
    for (int w = 0; w < n_workers; w++) {
        workers.emplace_back([&] {
            while (true) {
                const size_t i = next.fetch_add(1);
                if (i >= lines.size()) {
                    break;
                }
                prepared_row & row = rows[i];
                try {
                    const json line = json::parse(lines[i]);
                    row.id     = line.at("id").get<std::string>();
                    row.source = line.value("source", std::string());
                    row.prefix = line.value("prefix", std::string());
                    parsed_request pr = parse_request(env, line.at("body"));
                    row.error     = pr.error;
                    row.n_tokens  = (int) pr.tokens.size();
                    row.n_raw     = pr.n_raw;
                    row.truncated = pr.truncated;
                    row.grammar   = pr.grammar;
                    row.n_stops   = pr.n_stops;
                    row.cuts      = pr.cuts;
                    const size_t n = std::min(pr.tokens.size(), donor.tokens.size());
                    size_t lcp = 0;
                    while (lcp < n && pr.tokens[lcp] == donor.tokens[lcp]) {
                        lcp++;
                    }
                    row.lcp = (int) lcp;
                } catch (const std::exception & e) {
                    row.error = std::string("line: ") + e.what();
                }
                const size_t d = done.fetch_add(1) + 1;
                if (d % 200 == 0) {
                    say("prepare: %zu / %zu", d, lines.size());
                }
            }
        });
    }
    for (auto & w : workers) {
        w.join();
    }

    FILE * f = open_file(opt.out, "wb");
    fprintf(f, "{\"type\":\"header\",\"donor\":%s,\"donor_tokens\":%zu,\"tok_model\":%d,\"tok_end\":%d,\"max_tokens\":%d}\n",
            jq(donor_id).c_str(), donor.tokens.size(), env.tok_model, env.tok_end, opt.max_tokens);
    long long total = 0;
    size_t n_err = 0;
    for (const prepared_row & row : rows) {
        fprintf(f, "{\"type\":\"request\",\"id\":%s,\"source\":%s,\"prefix\":%s,\"n_tokens\":%d,\"n_raw\":%d,\"truncated\":%s,\"lcp\":%d,"
                   "\"grammar\":%s,\"stops\":%d,\"cuts\":%s,\"error\":%s}\n",
                jq(row.id).c_str(), jq(row.source).c_str(), jq(row.prefix).c_str(), row.n_tokens, row.n_raw,
                row.truncated ? "true" : "false", row.lcp, row.grammar ? "true" : "false", row.n_stops,
                ints_json(row.cuts).c_str(), jq(row.error).c_str());
        total += row.n_tokens;
        if (!row.error.empty()) {
            n_err++;
            if (n_err <= 5) {
                say("prepare: %s: %s", row.id.c_str(), row.error.c_str());
            }
        }
    }
    if (fclose(f) != 0) {
        dsp_die("cannot write %s", opt.out.c_str());
    }
    say("prepare: wrote %s: %zu requests, %lld tokens, %zu errors", opt.out.c_str(), rows.size(), total, n_err);

    llama_model_free(model);
    return 0;
}

// ---------------------------------------------------------------------------------------------------------------------
// capture: the plan
// ---------------------------------------------------------------------------------------------------------------------

struct plan_cut {
    int      pos   = 0;
    int      g_max = 0;
    uint32_t seed  = 0;
};

struct plan_req {
    std::string id, file;
    int n_tokens   = 0;
    int shared_len = 0;
    std::vector<plan_cut> cuts;
};

struct plan_t {
    std::string donor;
    int k_tools    = 0;
    int k_sys      = 0;
    int max_tokens = 0;
    std::vector<plan_req> reqs;
};

static plan_t load_plan(const std::string & path) {
    plan_t plan;
    bool have_header = false;
    for (const std::string & text : read_lines({path})) {
        const json j = json::parse(text);
        const std::string type = j.at("type").get<std::string>();
        if (type == "header") {
            plan.donor      = j.at("donor").get<std::string>();
            plan.k_tools    = j.at("k_tools").get<int>();
            plan.k_sys      = j.at("k_sys").get<int>();
            plan.max_tokens = j.at("max_tokens").get<int>();
            have_header     = true;
        } else if (type == "request") {
            plan_req r;
            r.id         = j.at("id").get<std::string>();
            r.file       = j.at("file").get<std::string>();
            r.n_tokens   = j.at("n_tokens").get<int>();
            r.shared_len = j.at("shared_len").get<int>();
            for (const auto & c : j.at("cuts")) {
                plan_cut pc;
                pc.pos   = c.at("pos").get<int>();
                pc.g_max = c.at("g_max").get<int>();
                pc.seed  = c.at("seed").get<unsigned int>();
                r.cuts.push_back(pc);
            }
            for (size_t i = 0; i < r.cuts.size(); i++) {
                const plan_cut & pc = r.cuts[i];
                if (pc.pos <= r.shared_len || pc.pos > r.n_tokens || pc.g_max < 1 || (i > 0 && pc.pos <= r.cuts[i - 1].pos)) {
                    dsp_die("plan: request %s has a bad cut at %d", r.id.c_str(), pc.pos);
                }
            }
            if (r.n_tokens <= r.shared_len) {
                dsp_die("plan: request %s has nothing after its shared prefix", r.id.c_str());
            }
            if (r.file.empty() || r.file.find('/') != std::string::npos) {
                dsp_die("plan: request %s has a bad file name", r.id.c_str());
            }
            plan.reqs.push_back(std::move(r));
        }
    }
    if (!have_header) {
        dsp_die("plan: %s has no header", path.c_str());
    }
    if (plan.k_tools < 0 || plan.k_sys < plan.k_tools) {
        dsp_die("plan: keep points %d and %d are out of order", plan.k_tools, plan.k_sys);
    }
    return plan;
}

// ---------------------------------------------------------------------------------------------------------------------
// capture: one thread writes everything, in order
// ---------------------------------------------------------------------------------------------------------------------

struct write_op {
    enum kind_t { DATA, CLOSE, LINE } kind = DATA;
    FILE * f = nullptr;
    std::unique_ptr<uint8_t[]> raw;
    size_t                     raw_n = 0;
    std::vector<uint8_t>       vec;
    std::string                text;

    size_t bytes() const { return raw_n + vec.size() + text.size(); }
};

class async_writer {
  public:
    explicit async_writer(size_t max_pending) : max_pending(max_pending), worker([this] { run(); }) {}

    void data(FILE * f, std::unique_ptr<uint8_t[]> && raw, size_t n) {
        write_op op;
        op.kind  = write_op::DATA;
        op.f     = f;
        op.raw   = std::move(raw);
        op.raw_n = n;
        push(std::move(op));
    }

    void data(FILE * f, std::vector<uint8_t> && vec) {
        write_op op;
        op.kind = write_op::DATA;
        op.f    = f;
        op.vec  = std::move(vec);
        push(std::move(op));
    }

    void close(FILE * f) {
        write_op op;
        op.kind = write_op::CLOSE;
        op.f    = f;
        push(std::move(op));
    }

    // appended and flushed once everything queued before it is on its way to disk
    void line(FILE * f, std::string text) {
        write_op op;
        op.kind = write_op::LINE;
        op.f    = f;
        op.text = std::move(text);
        push(std::move(op));
    }

    void finish() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            stopping = true;
        }
        cv_work.notify_all();
        if (worker.joinable()) {
            worker.join();
        }
    }

    bool failed() const { return has_failed.load(); }
    std::string error() {
        std::lock_guard<std::mutex> lock(mutex);
        return error_text;
    }
    size_t pending() {
        std::lock_guard<std::mutex> lock(mutex);
        return pending_bytes;
    }
    long long written() const { return written_bytes.load(); }

  private:
    void push(write_op && op) {
        std::unique_lock<std::mutex> lock(mutex);
        cv_room.wait(lock, [&] { return pending_bytes < max_pending || has_failed.load(); });
        pending_bytes += op.bytes();
        queue.push_back(std::move(op));
        lock.unlock();
        cv_work.notify_one();
    }

    void fail(const std::string & what) {
        std::lock_guard<std::mutex> lock(mutex);
        if (!has_failed.load()) {
            error_text = what + ": " + strerror(errno);
            has_failed.store(true);
        }
    }

    void run() {
        while (true) {
            write_op op;
            {
                std::unique_lock<std::mutex> lock(mutex);
                cv_work.wait(lock, [&] { return stopping || !queue.empty(); });
                if (queue.empty()) {
                    return;
                }
                op = std::move(queue.front());
                queue.pop_front();
            }
            const size_t n = op.bytes();
            if (!has_failed.load()) {
                switch (op.kind) {
                    case write_op::DATA: {
                        const uint8_t * p = op.raw ? op.raw.get() : op.vec.data();
                        if (n > 0 && fwrite(p, 1, n, op.f) != n) {
                            fail("short write");
                        }
                        written_bytes.fetch_add((long long) n);
                    } break;
                    case write_op::CLOSE: {
                        if (fclose(op.f) != 0) {
                            fail("close");
                        }
                    } break;
                    case write_op::LINE: {
                        if (fwrite(op.text.data(), 1, op.text.size(), op.f) != op.text.size() || fflush(op.f) != 0) {
                            fail("index write");
                        }
                    } break;
                }
            }
            {
                std::lock_guard<std::mutex> lock(mutex);
                pending_bytes -= n;
            }
            cv_room.notify_all();
        }
    }

    size_t                  max_pending;
    std::mutex              mutex;
    std::condition_variable cv_work, cv_room;
    std::deque<write_op>    queue;
    size_t                  pending_bytes = 0;
    bool                    stopping      = false;
    std::atomic<bool>       has_failed{false};
    std::atomic<long long>  written_bytes{0};
    std::string             error_text;
    std::thread             worker;
};

// ---------------------------------------------------------------------------------------------------------------------
// capture: the recorder
// ---------------------------------------------------------------------------------------------------------------------

struct job_t {
    const plan_req * req = nullptr;
    parsed_request   pr;
    int              g_max = 0; // the longest continuation its cuts ask for
};

struct segment_t {
    int      cut      = 0;
    int      g_max    = 0;
    uint32_t seed     = 0;
    int      n_states = 0;      // generated tokens that were decoded
    long long states_row0 = 0;
    long long topk_row0   = 0;
    std::string stop;
    std::vector<int> tokens;    // n_states + 1: the last one was sampled and not decoded
};

enum slot_state { S_IDLE, S_WARMUP, S_PREFILL, S_GEN };

struct slot_t {
    llama_seq_id seq   = 0;
    slot_state   state = S_IDLE;

    // the keep points of the donor prefix this slot holds a checkpoint of
    int level = 0;
    std::vector<common_prompt_checkpoint> keep;

    // warming up to a keep point
    int  warm_need     = 0;
    int  warm_pos      = 0;
    bool writes_prefix = false;

    // the request
    std::unique_ptr<job_t> job;
    int    pos      = 0;
    size_t next_cut = 0;
    FILE * f_states     = nullptr;
    FILE * f_gen_states = nullptr;
    FILE * f_gen_topk   = nullptr;
    long long states_rows = 0;
    long long gen_rows    = 0;
    long long topk_rows   = 0;
    bool sampler_fallback = false;
    std::vector<segment_t> segments;

    // the continuation being written
    common_sampler *         smpl = nullptr;
    common_prompt_checkpoint cut_ckpt;
    bool                     have_cut_ckpt = false;
    bool                     cut_trip = false;
    segment_t                seg;
    llama_token              pending = LLAMA_TOKEN_NULL;
    llama_token              sampled = LLAMA_TOKEN_NULL;
    int32_t                  out_idx = -1;
    std::vector<uint8_t>     gen_buf;
    std::vector<uint8_t>     topk_buf;
};

struct stats_t {
    long long n_done      = 0;
    long long tok_prefill = 0;
    long long tok_warm    = 0;
    long long tok_gen     = 0;
    long long p_steps     = 0;
    long long g_steps     = 0;
    double    t_p_decode  = 0;
    double    t_p_post    = 0;
    double    t_g_decode  = 0;
    double    t_g_post    = 0;
    long long n_ckpt      = 0;
    double    t_ckpt      = 0;
    long long bytes_ckpt  = 0;
    long long n_keep      = 0;
    double    t_keep      = 0;
    long long bytes_keep  = 0;
    long long n_restore   = 0;
    double    t_restore   = 0;
    long long n_trip      = 0;
    double    t_trip      = 0;
    long long n_host      = 0; // states copied to or from host memory, and the tensor copies they took
    long long copies_host = 0;
    long long n_smpl      = 0;
    double    t_smpl      = 0;
    long long n_fallback  = 0;
    long long n_grammar_off = 0;
    long long seg_stop    = 0;
    long long seg_length  = 0;
};

class recorder {
  public:
    recorder(const tool_opts & opt, common_params & params, const std::vector<std::string> & server_argv)
        : opt(opt), params(params), server_argv(server_argv) {}

    int run();

  private:
    // setup
    void load_inputs();
    void open_outputs();
    void feeder_main();

    // scheduling
    void      record_prefix();
    bool      assign_jobs();
    long long held(const slot_t & s) const;
    void      start_request(slot_t & s, std::unique_ptr<job_t> job);
    int       next_stop(const slot_t & s) const;
    void      p_step();
    void      g_step();

    // per slot
    bool state_of(llama_seq_id seq, std::vector<uint8_t> & out);
    void verify_look(const char * when, const std::vector<char> & in_batch);
    void verify_trip(slot_t & s, const char * what, const uint8_t * want, size_t want_size);
    void verify_touch(const slot_t & s) {
        if (opt.verify && (size_t) s.seq < watch.size()) {
            watch[s.seq].touched = true;
        }
    }
    void take_keep(slot_t & s);
    void restore_keep(slot_t & s, int index);
    void draw_first(slot_t & s, int32_t out_idx);
    void accept(slot_t & s, llama_token id);
    void begin_segment(slot_t & s);
    void end_segment(slot_t & s, const char * why);
    void finish_request(slot_t & s);
    void topk_row(const float * logits, std::vector<uint8_t> & out) const;
    bool is_stop(llama_token id) const { return id == env.tok_end || llama_vocab_is_eog(vocab, id); }

    void progress(bool force);
    void write_summary();

    const tool_opts &                opt;
    common_params &                  params;
    const std::vector<std::string> & server_argv;

    common_init_result_ptr llama_init;
    llama_model *          model = nullptr;
    llama_context *        ctx   = nullptr;
    const llama_vocab *    vocab = nullptr;
    llama_memory_t         mem   = nullptr;
    llama_batch_ext *      batch = nullptr;
    chat_env               env;

    int    n_embd    = 0;
    int    n_vocab   = 0;
    int    n_batch   = 0;
    int    n_step    = 0;        // the prompt rows one decode holds at most
    int    n_chunk   = 0;        // the server's prompt chunk: a sequence's share of a step is whole chunks
    int    p_cursor  = 0;        // the sequence the next prompt decode starts from
    int    n_slots   = 0;
    FILE * f_trace   = nullptr;  // --trace: what every prompt batch held

    // --verify-states: each sequence's partial state as the engine last gave it
    struct watch_t {
        std::vector<uint8_t> blob;
        bool have    = false;
        bool touched = false;  // the tool loaded or cut it since: it may differ
    };
    std::vector<watch_t> watch;
    FILE *    f_verify   = nullptr;
    long long v_compared = 0, v_moved = 0, v_trips = 0, v_trips_differ = 0, v_dumped = 0;
    long long pool   = 0;
    long long seq_limit = 0;
    size_t row_bytes  = 0;  // one position: every tap, bf16
    size_t topk_bytes = 0;  // one distribution: logsumexp, ids, logits

    plan_t             plan;
    std::vector<int>   keeps;       // the donor positions a slot can be rewound to, ascending
    llama_tokens       donor;       // the donor's tokens up to the last keep point
    std::unordered_map<std::string, std::string> raw;  // request id -> its line
    std::unordered_set<std::string> done_ids;
    std::vector<const plan_req *>   todo;

    std::vector<slot_t> slots;

    // the feeder parses requests ahead of the slots, in plan order
    std::thread             feeder;
    std::mutex              q_mutex;
    std::mutex              smpl_mutex;
    std::condition_variable q_cv_room, q_cv_item;
    std::deque<std::unique_ptr<job_t>> q;
    bool                    feeder_done = false;

    std::unique_ptr<async_writer> writer;
    FILE *    f_index  = nullptr;
    FILE *    f_prefix = nullptr;
    int       prefix_have = 0;      // donor positions whose states an earlier run wrote

    stats_t stats, stats_prev;
    double  t_start = 0, t_prev = 0;
};

void recorder::load_inputs() {
    plan = load_plan(opt.plan);
    if (plan.k_tools > 0) {
        keeps.push_back(plan.k_tools);
    }
    if (plan.k_sys > plan.k_tools) {
        keeps.push_back(plan.k_sys);
    }

    std::unordered_set<std::string> wanted;
    for (const plan_req & r : plan.reqs) {
        wanted.insert(r.id);
        if (r.shared_len != 0 && std::find(keeps.begin(), keeps.end(), r.shared_len) == keeps.end()) {
            dsp_die("plan: request %s shares %d tokens, which is no keep point", r.id.c_str(), r.shared_len);
        }
    }
    wanted.insert(plan.donor);

    const std::vector<std::string> lines = read_lines(opt.requests);
    std::vector<std::string> ids(lines.size());
    parallel_for(opt.threads, lines.size(), [&](size_t b, size_t e) {
        for (size_t i = b; i < e; i++) {
            try {
                ids[i] = json::parse(lines[i]).at("id").get<std::string>();
            } catch (const std::exception & ex) {
                dsp_die("requests line %zu: %s", i, ex.what());
            }
        }
    });
    for (size_t i = 0; i < lines.size(); i++) {
        if (wanted.count(ids[i]) > 0) {
            raw[ids[i]] = lines[i];
        }
    }
    for (const std::string & id : wanted) {
        if (raw.count(id) == 0) {
            dsp_die("request %s of the plan is in none of the --requests files", id.c_str());
        }
    }

    // the donor prefix
    if (!keeps.empty()) {
        const json line = json::parse(raw.at(plan.donor));
        parsed_request pr = parse_request(env, line.at("body"));
        if (!pr.error.empty()) {
            dsp_die("the donor does not parse: %s", pr.error.c_str());
        }
        if ((int) pr.tokens.size() < keeps.back()) {
            dsp_die("the donor has %zu tokens, fewer than the keep point %d", pr.tokens.size(), keeps.back());
        }
        donor.assign(pr.tokens.begin(), pr.tokens.begin() + keeps.back());
    }

    // what an earlier run finished
    const std::string index_path = opt.out + "/index.jsonl";
    if (file_size(index_path) > 0) {
        for (const std::string & text : read_lines({index_path})) {
            const json j = json::parse_no_throw(text);
            if (j.is_discarded() || !j.contains("id")) {
                continue;
            }
            done_ids.insert(j.at("id").get<std::string>());
        }
    }

    for (const plan_req & r : plan.reqs) {
        if (opt.max_requests >= 0 && (int) todo.size() + (int) done_ids.size() >= opt.max_requests) {
            break;
        }
        if (done_ids.count(r.id) > 0) {
            continue;
        }
        todo.push_back(&r);
    }
    say("capture: plan has %zu requests, %zu already recorded, %zu to do; keep points:%s", plan.reqs.size(), done_ids.size(), todo.size(),
        keeps.empty() ? " none" : (" " + ints_json(keeps)).c_str());
}

void recorder::open_outputs() {
    make_dir(opt.out);
    make_dir(opt.out + "/req");

    // meta.json
    {
        json meta = json::object();
        json taps = json::array();
        for (int t : opt.taps) {
            taps.push_back(t);
        }
        json argv = json::array();
        for (const std::string & a : server_argv) {
            argv.push_back(a);
        }
        meta["taps"]       = taps;
        meta["n_embd"]     = n_embd;
        meta["n_vocab"]    = n_vocab;
        meta["dtype"]      = "bf16";
        meta["row_bytes"]  = (long long) row_bytes;
        meta["topk"]       = opt.topk;
        meta["topk_bytes"] = (long long) topk_bytes;
        meta["k_tools"]    = plan.k_tools;
        meta["k_sys"]      = plan.k_sys;
        meta["donor"]      = plan.donor;
        meta["tok_model"]  = (int) env.tok_model;
        meta["tok_end"]    = (int) env.tok_end;
        meta["n_slots"]    = n_slots;
        meta["n_batch"]    = n_batch;
        meta["n_step"]     = n_step;
        meta["n_chunk"]    = n_chunk;
        meta["pool"]       = pool;
        meta["no_cuts"]    = opt.no_cuts;
        meta["stops"]      = opt.stops;
        meta["server_argv"] = argv;
        FILE * f = open_file(opt.out + "/meta.json", "wb");
        const std::string text = meta.dump(2) + "\n";
        fwrite(text.data(), 1, text.size(), f);
        fclose(f);
    }

    // the donor prefix: tokens now, states by the first slot that reads each part
    if (!keeps.empty()) {
        FILE * f = open_file(opt.out + "/prefix.tokens", "wb");
        fwrite(donor.data(), sizeof(llama_token), donor.size(), f);
        fclose(f);

        const std::string path = opt.out + "/prefix.states";
        const long long size = file_size(path);
        const long long rows = size > 0 ? size / (long long) row_bytes : 0;
        int have = 0;
        for (int k : keeps) {
            if (rows >= k) {
                have = k;
            }
        }
        if (size > 0 && size != (long long) have * (long long) row_bytes) {
            if (truncate(path.c_str(), (off_t) ((long long) have * (long long) row_bytes)) != 0) {
                dsp_die("cannot cut %s back to %d rows", path.c_str(), have);
            }
        }
        prefix_have = have;
        f_prefix = open_file(path, "ab");
    }

    f_index = open_file(opt.out + "/index.jsonl", "ab");
    writer.reset(new async_writer((size_t) opt.write_buffer_gb << 30));
}

void recorder::feeder_main() {
    for (const plan_req * r : todo) {
        std::unique_ptr<job_t> job(new job_t());
        job->req = r;
        try {
            const json line = json::parse(raw.at(r->id));
            if (line.contains("tokens")) {
                // a request that is its tokens already: a continuation of an earlier recording read back as a prompt
                job->pr.tokens   = line.at("tokens").get<llama_tokens>();
                job->pr.n_raw    = (int) job->pr.tokens.size();
                job->pr.sampling = env.params->sampling;
                for (llama_token t : job->pr.tokens) {
                    if (t < 0 || t >= llama_vocab_n_tokens(env.vocab)) {
                        dsp_die("request %s holds token %d, which the vocabulary does not have", r->id.c_str(), t);
                    }
                }
                if (!r->cuts.empty()) {
                    dsp_die("request %s is given as tokens and has cuts in the plan", r->id.c_str());
                }
            } else {
                job->pr = parse_request(env, line.at("body"));
            }
        } catch (const std::exception & e) {
            dsp_die("request %s: %s", r->id.c_str(), e.what());
        }
        if (!job->pr.error.empty()) {
            dsp_die("request %s does not parse: %s", r->id.c_str(), job->pr.error.c_str());
        }
        for (const plan_cut & pc : r->cuts) {
            job->g_max = std::max(job->g_max, pc.g_max);
        }
        if (opt.no_cuts) {
            job->g_max = 0;
        }
        std::unique_lock<std::mutex> lock(q_mutex);
        q_cv_room.wait(lock, [&] { return q.size() < (size_t) (2 * n_slots); });
        q.push_back(std::move(job));
        lock.unlock();
        q_cv_item.notify_all();
    }
    {
        std::lock_guard<std::mutex> lock(q_mutex);
        feeder_done = true;
    }
    q_cv_item.notify_all();
}

// cells of the shared pool a slot holds or will hold
long long recorder::held(const slot_t & s) const {
    const long long kept = s.level > 0 ? keeps[s.level - 1] : 0;
    if (s.state == S_IDLE) {
        return kept;
    }
    const long long warm = s.state == S_WARMUP ? keeps[s.warm_need - 1] : 0;
    const long long own  = s.job ? (long long) s.job->req->n_tokens + s.job->g_max + opt.margin : 0;
    return std::max(kept, std::max(warm, own));
}

bool recorder::assign_jobs() {
    bool any = false;
    for (slot_t & s : slots) {
        if (s.state != S_IDLE) {
            continue;
        }
        std::unique_ptr<job_t> job;
        {
            std::lock_guard<std::mutex> lock(q_mutex);
            if (q.empty()) {
                break;
            }
            const job_t & head = *q.front();
            const long long need = std::max((long long) head.req->shared_len, (long long) head.req->n_tokens + head.g_max + opt.margin);
            if (need > seq_limit) {
                dsp_die("request %s needs %lld cells, one sequence holds %lld", head.req->id.c_str(), need, seq_limit);
            }
            long long others = 0;
            bool busy = false;
            for (const slot_t & t : slots) {
                if (&t == &s) {
                    continue;
                }
                others += held(t);
                busy = busy || t.state != S_IDLE;
            }
            if (others + need > pool) {
                if (!busy) {
                    dsp_die("request %s needs %lld cells and the other slots keep %lld of the pool's %lld", head.req->id.c_str(), need, others, pool);
                }
                break; // wait for a slot to finish
            }
            job = std::move(q.front());
            q.pop_front();
        }
        q_cv_room.notify_all();
        start_request(s, std::move(job));
        any = true;
    }
    return any;
}

// the donor prefix is read once by slot 0 alone, before any request, so its states are written once and in order
void recorder::record_prefix() {
    if (keeps.empty() || prefix_have >= keeps.back()) {
        return;
    }
    slot_t & s = slots[0];
    llama_memory_seq_rm(mem, s.seq, -1, -1);
    s.level         = 0;
    s.warm_need     = (int) keeps.size();
    s.warm_pos      = 0;
    s.writes_prefix = true;
    s.state         = S_WARMUP;
    while (s.state == S_WARMUP) {
        p_step();
    }
    say("capture: the donor prefix is recorded, %d positions (%d from an earlier run)", keeps.back(), prefix_have);
}

// the engine's partial state of a sequence (its sliding-window rows and its recurrent rows), or nothing if it holds none
bool recorder::state_of(llama_seq_id seq, std::vector<uint8_t> & out) {
    out.clear();
    if (llama_memory_seq_pos_max(mem, seq) < 0) {
        return false;
    }
    const size_t n = llama_state_seq_get_size_ext(ctx, seq, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    out.resize(n);
    const size_t got = n == 0 ? 0 : llama_state_seq_get_data_ext(ctx, out.data(), n, seq, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    out.resize(got);
    return got > 0;
}

static void blob_diff(const uint8_t * a, size_t na, const uint8_t * b, size_t nb, long long & first, long long & last,
                      long long & count) {
    first = last = -1;
    count = 0;
    const size_t n = std::min(na, nb);
    for (size_t i = 0; i < n; i++) {
        if (a[i] != b[i]) {
            if (first < 0) {
                first = (long long) i;
            }
            last = (long long) i;
            count++;
        }
    }
    if (na != nb) {
        if (first < 0) {
            first = (long long) n;
        }
        last   = (long long) std::max(na, nb) - 1;
        count += (long long) (std::max(na, nb) - n);
    }
}

// Every sequence's state now. One that was not in the decode just made (in_batch), and that the tool did not load or
// cut since the last look, must hold the bytes it held then; one that does not is written to verify.txt.
void recorder::verify_look(const char * when, const std::vector<char> & in_batch) {
    if (f_verify == nullptr) {
        f_verify = open_file(opt.out + "/verify.txt", "ab");
    }
    std::vector<uint8_t> cur;
    for (slot_t & s : slots) {
        watch_t & w = watch[s.seq];
        const bool have = state_of(s.seq, cur);
        const bool held = (size_t) s.seq < in_batch.size() && in_batch[s.seq];
        if (have && w.have && !w.touched && !held) {
            v_compared++;
            if (cur != w.blob) {
                long long first, last, count;
                blob_diff(w.blob.data(), w.blob.size(), cur.data(), cur.size(), first, last, count);
                v_moved++;
                fprintf(f_verify, "M %s %lld %lld %d %d %zu %zu %lld %lld %lld %d %d\n", when, stats.p_steps, stats.g_steps,
                        s.seq, (int) s.state, w.blob.size(), cur.size(), first, last, count,
                        (int) llama_memory_seq_pos_min(mem, s.seq), (int) llama_memory_seq_pos_max(mem, s.seq));
                if (v_dumped < 8) {
                    const std::string base = opt.out + "/verify-" + std::to_string(v_dumped++);
                    FILE * fa = open_file(base + "-before.bin", "wb");
                    fwrite(w.blob.data(), 1, w.blob.size(), fa);
                    fclose(fa);
                    FILE * fb = open_file(base + "-after.bin", "wb");
                    fwrite(cur.data(), 1, cur.size(), fb);
                    fclose(fb);
                }
                fflush(f_verify);
            }
        }
        w.have    = have;
        w.touched = false;
        w.blob.swap(cur);
    }
}

// a state the tool just loaded, read back: it should be the bytes that were loaded
void recorder::verify_trip(slot_t & s, const char * what, const uint8_t * want, size_t want_size) {
    if (f_verify == nullptr) {
        f_verify = open_file(opt.out + "/verify.txt", "ab");
    }
    std::vector<uint8_t> cur;
    state_of(s.seq, cur);
    v_trips++;
    if (cur.size() != want_size || std::memcmp(cur.data(), want, want_size) != 0) {
        long long first, last, count;
        blob_diff(want, want_size, cur.data(), cur.size(), first, last, count);
        v_trips_differ++;
        fprintf(f_verify, "T %s %lld %lld %d %zu %zu %lld %lld %lld\n", what, stats.p_steps, stats.g_steps, s.seq, want_size,
                cur.size(), first, last, count);
        if (v_dumped < 8) {
            const std::string base = opt.out + "/verify-" + std::to_string(v_dumped++);
            FILE * fa = open_file(base + "-before.bin", "wb");
            fwrite(want, 1, want_size, fa);
            fclose(fa);
            FILE * fb = open_file(base + "-after.bin", "wb");
            fwrite(cur.data(), 1, cur.size(), fb);
            fclose(fb);
        }
        fflush(f_verify);
    }
}

void recorder::take_keep(slot_t & s) {
    const double t0 = now_s();
    common_prompt_checkpoint & ck = s.keep[s.level];
    ck.clear();
    ck.update_pos(keeps[s.level], llama_memory_seq_pos_min(mem, s.seq), llama_memory_seq_pos_max(mem, s.seq));
    ck.update_tgt(ctx, s.seq, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    stats.n_keep++;
    stats.t_keep     += now_s() - t0;
    stats.bytes_keep += (long long) ck.size();
    s.level++;
}

// the slot back at its keep point `index`: the state at that position, and nothing after it
void recorder::restore_keep(slot_t & s, int index) {
    const double t0 = now_s();
    if (!s.keep[index].load_tgt(ctx, s.seq, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY)) {
        dsp_die("slot %d: the keep point at %d does not load", s.seq, keeps[index]);
    }
    if (!llama_memory_seq_rm(mem, s.seq, keeps[index], -1)) {
        dsp_die("slot %d: cannot drop the positions from %d on", s.seq, keeps[index]);
    }
    stats.n_restore++;
    stats.t_restore += now_s() - t0;
    verify_touch(s);
    if (opt.verify) {
        verify_trip(s, "keep", s.keep[index].data_tgt.data(), s.keep[index].data_tgt.size());
    }
}

void recorder::start_request(slot_t & s, std::unique_ptr<job_t> job) {
    const plan_req &       r  = *job->req;
    const parsed_request & pr = job->pr;

    // the plan was made from these same requests: anything else means the two runs differ
    if ((int) pr.tokens.size() != r.n_tokens) {
        dsp_die("request %s is %zu tokens here and %d in the plan", r.id.c_str(), pr.tokens.size(), r.n_tokens);
    }
    for (const plan_cut & pc : r.cuts) {
        if (!std::binary_search(pr.cuts.begin(), pr.cuts.end(), pc.pos)) {
            dsp_die("request %s has no model turn at %d", r.id.c_str(), pc.pos);
        }
    }
    if (r.shared_len > 0 && !std::equal(donor.begin(), donor.begin() + r.shared_len, pr.tokens.begin())) {
        dsp_die("request %s does not start with the donor's first %d tokens", r.id.c_str(), r.shared_len);
    }

    int need = 0;
    if (r.shared_len > 0) {
        need = (int) (std::find(keeps.begin(), keeps.end(), r.shared_len) - keeps.begin()) + 1;
    }

    const std::string base = opt.out + "/req/" + r.file;
    {
        FILE * f = open_file(base + ".tokens", "wb");
        fwrite(pr.tokens.data(), sizeof(llama_token), pr.tokens.size(), f);
        fclose(f);
    }
    s.f_states     = open_file(base + ".states", "wb");
    s.f_gen_states = opt.no_cuts ? nullptr : open_file(base + ".gen.states", "wb");
    s.f_gen_topk   = opt.no_cuts ? nullptr : open_file(base + ".gen.topk", "wb");

    s.job              = std::move(job);
    s.pos              = r.shared_len;
    s.next_cut         = 0;
    s.states_rows      = 0;
    s.gen_rows         = 0;
    s.topk_rows        = 0;
    s.sampler_fallback = false;
    s.segments.clear();

    if (need == 0) {
        llama_memory_seq_rm(mem, s.seq, -1, -1);
        verify_touch(s);
        s.level = 0;
        s.state = S_PREFILL;
    } else if (need <= s.level) {
        if (need < s.level) {
            restore_keep(s, need - 1);
            s.level = need;
        }
        s.state = S_PREFILL;
    } else {
        if (s.level == 0) {
            llama_memory_seq_rm(mem, s.seq, -1, -1);
            verify_touch(s);
        }
        s.warm_need     = need;
        s.warm_pos      = s.level > 0 ? keeps[s.level - 1] : 0;
        s.writes_prefix = false;
        s.state         = S_WARMUP;
    }
}

int recorder::next_stop(const slot_t & s) const {
    if (s.state == S_WARMUP) {
        return keeps[s.level];
    }
    const plan_req & r = *s.job->req;
    if ((!opt.no_cuts || opt.stops) && s.next_cut < r.cuts.size()) {
        return r.cuts[s.next_cut].pos;
    }
    return r.n_tokens;
}

void recorder::topk_row(const float * logits, std::vector<uint8_t> & out) const {
    const int k = opt.topk;
    // logsumexp over the whole row
    float vmax = logits[0];
    for (int i = 1; i < n_vocab; i++) {
        vmax = std::max(vmax, logits[i]);
    }
    double sum = 0.0;
    for (int i = 0; i < n_vocab; i++) {
        sum += std::exp((double) (logits[i] - vmax));
    }
    const float lse = (float) ((double) vmax + std::log(sum));

    // the k largest, by a small heap whose top is the smallest kept
    std::vector<std::pair<float, int>> heap;
    heap.reserve((size_t) k + 1);
    auto smaller = [](const std::pair<float, int> & a, const std::pair<float, int> & b) {
        // a ranks before b: larger logit, then smaller id
        return a.first > b.first || (a.first == b.first && a.second < b.second);
    };
    for (int i = 0; i < n_vocab; i++) {
        const std::pair<float, int> item(logits[i], i);
        if ((int) heap.size() < k) {
            heap.push_back(item);
            std::push_heap(heap.begin(), heap.end(), smaller);
        } else if (smaller(item, heap.front())) {
            std::pop_heap(heap.begin(), heap.end(), smaller);
            heap.back() = item;
            std::push_heap(heap.begin(), heap.end(), smaller);
        }
    }
    std::sort(heap.begin(), heap.end(), smaller);

    const size_t at = out.size();
    out.resize(at + topk_bytes);
    uint8_t * p = out.data() + at;
    std::memcpy(p, &lse, 4);
    p += 4;
    for (int i = 0; i < k; i++) {
        const int32_t id = i < (int) heap.size() ? heap[i].second : -1;
        std::memcpy(p + 4 * i, &id, 4);
    }
    p += 4 * (size_t) k;
    for (int i = 0; i < k; i++) {
        const float v = i < (int) heap.size() ? heap[i].first : 0.0f;
        std::memcpy(p + 4 * i, &v, 4);
    }
}

void recorder::p_step() {
    std::vector<slot_t *> act;
    for (slot_t & s : slots) {
        if (s.state == S_WARMUP || s.state == S_PREFILL) {
            act.push_back(&s);
        }
    }

    // One decode holds what the server gives one (server-context.cpp, prepare_batch_view): at most a micro-batch of prompt
    // rows, in runs of one width, each the next whole chunks of a sequence, and a shorter run in a decode of its own. A
    // decode of several micro-batches is no shape the server makes, and it is not safe in a pool that sequences share: the
    // sliding-window cache places all of them on trial first, a later one may take a cell an earlier one gave a sequence,
    // the cells of that sequence below it are purged with it, and the trial's undo brings back only the cells the
    // micro-batches took (llama_kv_cache::prepare, apply_ubatch). The window the sequence's first rows read is then gone
    // before they are computed.
    std::vector<int> take(act.size(), 0), room(act.size(), 0);
    size_t first = 0;
    for (size_t k = 0; k < act.size(); k++) {
        const slot_t & s = *act[k];
        room[k] = next_stop(s) - (s.state == S_WARMUP ? s.warm_pos : s.pos);
        if (room[k] <= 0) {
            dsp_die("slot %d has nothing to read", s.seq);
        }
        if (act[first]->seq < p_cursor && s.seq >= p_cursor) {
            first = k;
        }
    }
    const int share = std::max(n_chunk, n_batch / (int) act.size() / n_chunk * n_chunk);
    const auto run  = [&](size_t k) { return std::min(std::min(share, n_step), room[k]); };
    const int width = run(first);
    int used = 0;
    for (size_t i = 0; i < act.size() && used + width <= n_step; i++) {
        const size_t k = (first + i) % act.size();
        if (run(k) != width) {
            continue;
        }
        take[k]  = width;
        used    += width;
        p_cursor = act[k]->seq + 1;
    }

    struct chunk_t {
        slot_t * s;
        int32_t  idx0;
        int      p0;
        int      n;
        bool     to_cut;
        bool     record;
        std::unique_ptr<uint8_t[]> buf;
    };
    std::vector<chunk_t> chunks;

    llama_batch_ext_clear(batch);
    for (size_t k = 0; k < act.size(); k++) {
        slot_t & s = *act[k];
        const int n = take[k];
        if (n == 0) {
            continue;
        }
        const bool warm = s.state == S_WARMUP;
        const llama_token * src = warm ? donor.data() : s.job->pr.tokens.data();
        const int p0 = warm ? s.warm_pos : s.pos;

        chunk_t c;
        c.s      = &s;
        c.idx0   = -1;
        c.p0     = p0;
        c.n      = n;
        c.to_cut = false;
        c.record = warm ? (s.writes_prefix && p0 >= prefix_have) : true;
        for (int i = 0; i < n; i++) {
            const int32_t idx = llama_batch_ext_add_token(batch, s.seq, src[p0 + i]);
            if (idx < 0) {
                dsp_die("slot %d: the batch refused a token (%d)", s.seq, idx);
            }
            const llama_pos p = p0 + i;
            llama_batch_ext_set_pos(batch, idx, &p);
            if (i == 0) {
                c.idx0 = idx;
            } else if (idx != c.idx0 + i) {
                dsp_die("slot %d: batch rows are not in a row", s.seq);
            }
        }
        if (!warm && !opt.no_cuts && s.next_cut < s.job->req->cuts.size() && p0 + n == s.job->req->cuts[s.next_cut].pos) {
            llama_batch_ext_set_output_logits(batch, c.idx0 + n - 1, true);
            c.to_cut = true;
        }
        chunks.push_back(std::move(c));
    }

    if (opt.verify) {
        verify_look("before-P", {});
    }
    double t0 = now_s();
    const int32_t rc = llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch);
    if (rc != 0) {
        dsp_die("decode of %d prompt tokens failed (%d)", used, rc);
    }
    double t1 = now_s();
    stats.p_steps++;
    stats.t_p_decode += t1 - t0;
    if (opt.trace) {
        if (f_trace == nullptr) {
            f_trace = open_file(opt.out + "/trace.txt", "ab");
        }
        // step, generating steps so far, tokens in the batch, request file or - for the shared prefix, sequence, first row
        // of the batch, first position, count, whether it ends at a cut
        for (const chunk_t & c : chunks) {
            fprintf(f_trace, "P %lld %lld %d %s %d %d %d %d %d\n", stats.p_steps, stats.g_steps, used,
                    c.s->state == S_WARMUP ? "-" : c.s->job->req->file.c_str(), c.s->seq, c.idx0, c.p0, c.n, c.to_cut ? 1 : 0);
        }
        fflush(f_trace);
    }

    // the tapped states of every recorded position, as bf16
    std::vector<const float *> tap(opt.taps.size());
    for (size_t t = 0; t < opt.taps.size(); t++) {
        tap[t] = llama_get_embeddings_layer_inp(ctx, (uint32_t) opt.taps[t]);
        if (tap[t] == nullptr) {
            dsp_die("the context has no states for layer input %d", opt.taps[t]);
        }
    }
    struct item_t {
        int32_t    idx;
        uint16_t * dst;
    };
    std::vector<item_t> items;
    items.reserve((size_t) used);
    for (chunk_t & c : chunks) {
        if (!c.record) {
            continue;
        }
        c.buf.reset(new uint8_t[(size_t) c.n * row_bytes]);
        for (int i = 0; i < c.n; i++) {
            items.push_back({c.idx0 + i, (uint16_t *) (c.buf.get() + (size_t) i * row_bytes)});
        }
    }
    parallel_for(opt.threads, items.size(), [&](size_t b, size_t e) {
        for (size_t k = b; k < e; k++) {
            for (size_t t = 0; t < tap.size(); t++) {
                row_to_bf16(tap[t] + (size_t) items[k].idx * n_embd, items[k].dst + t * (size_t) n_embd, n_embd);
            }
        }
    });

    // a continuation's first token is drawn from a row that borrows the context's output storage, so every one of them
    // is drawn before any slot's state is saved, loaded or cut
    for (chunk_t & c : chunks) {
        if (c.to_cut) {
            draw_first(*c.s, c.idx0 + c.n - 1);
        }
    }
    if (opt.verify) {
        std::vector<char> held(slots.size(), 0);
        for (const chunk_t & c : chunks) {
            held[c.s->seq] = 1;
        }
        verify_look("after-P", held);
    }

    for (chunk_t & c : chunks) {
        slot_t & s = *c.s;
        if (s.state == S_WARMUP) {
            if (c.record) {
                writer->data(f_prefix, std::move(c.buf), (size_t) c.n * row_bytes);
            }
            s.warm_pos     += c.n;
            stats.tok_warm += c.n;
            if (s.warm_pos == keeps[s.level]) {
                take_keep(s);
                if (s.level == s.warm_need) {
                    s.state         = s.job ? S_PREFILL : S_IDLE;
                    s.writes_prefix = false;
                }
            }
        } else {
            writer->data(s.f_states, std::move(c.buf), (size_t) c.n * row_bytes);
            s.pos             += c.n;
            s.states_rows     += c.n;
            stats.tok_prefill += c.n;
            if (c.to_cut) {
                begin_segment(s);
            } else {
                if (opt.no_cuts && opt.stops) {
                    // a read that only ends where a run with cuts would end it
                    const auto & cuts = s.job->req->cuts;
                    while (s.next_cut < cuts.size() && cuts[s.next_cut].pos <= s.pos) {
                        s.next_cut++;
                    }
                }
                if (s.pos == s.job->req->n_tokens) {
                    finish_request(s);
                }
            }
        }
    }
    stats.t_p_post += now_s() - t1;
}

static common_sampler * make_sampler(const llama_model * model, const common_params_sampling & base, uint32_t seed, bool & fallback) {
    common_sampler * smpl = nullptr;
    {
        common_params_sampling sp = base;
        sp.seed = seed;
        try {
            smpl = common_sampler_init(model, sp);
        } catch (const std::exception & e) {
            say("capture: sampler: %s", e.what());
        }
    }
    if (smpl != nullptr) {
        return smpl;
    }
    fallback = true;
    common_params_sampling plain = base;
    plain.seed         = seed;
    plain.grammar      = {};
    plain.grammar_lazy = false;
    plain.grammar_triggers.clear();
    smpl = common_sampler_init(model, plain);
    if (smpl == nullptr) {
        dsp_die("the sampler cannot be built, with or without its grammar");
    }
    return smpl;
}

// A request with tools carries a lazy grammar that starts on what the model wrote: any first text that is no content
// token starts it. When that text is no tool name the engine throws, and the grammar is spent. The token is the model's
// own and stays in the record; the continuation goes on with the same sampler less its grammar, and the index says so.
void recorder::accept(slot_t & s, llama_token id) {
    try {
        common_sampler_accept(s.smpl, id, true);
        return;
    } catch (const std::exception & ex) {
        std::lock_guard<std::mutex> lock(smpl_mutex);
        if (stats.n_grammar_off++ < 5) {
            say("capture: slot %d: the grammar is off for the rest of this continuation: %s", s.seq, ex.what());
        }
    }
    std::lock_guard<std::mutex> lock(smpl_mutex);
    common_params_sampling plain = s.job->pr.sampling;
    plain.seed         = s.seg.seed;
    plain.grammar      = {};
    plain.grammar_lazy = false;
    plain.grammar_triggers.clear();
    common_sampler * smpl = common_sampler_init(model, plain);
    if (smpl == nullptr) {
        dsp_die("slot %d: the sampler cannot be built without its grammar", s.seq);
    }
    const llama_tokens & toks = s.job->pr.tokens;
    const int n_history = common_sampler_n_history(smpl);
    for (int i = std::max(0, s.seg.cut - n_history); i < s.seg.cut; i++) {
        common_sampler_accept(smpl, toks[i], false);
    }
    for (const llama_token t : s.seg.tokens) {
        common_sampler_accept(smpl, t, true);
    }
    common_sampler_accept(smpl, id, true);
    common_sampler_free(s.smpl);
    s.smpl = smpl;
    s.sampler_fallback = true;
}

// the request was read up to a cut the plan chose: the model writes from here. Its sampler is the one the server
// gives the slot, primed with the end of the prompt as slot::init_sampler() does
void recorder::draw_first(slot_t & s, int32_t out_idx) {
    const plan_req & r  = *s.job->req;
    const plan_cut & pc = r.cuts[s.next_cut];

    s.seg       = segment_t();
    s.seg.cut   = pc.pos;
    s.seg.g_max = pc.g_max;
    s.seg.seed  = pc.seed;
    s.gen_buf.clear();
    s.topk_buf.clear();
    s.have_cut_ckpt = false;

    double t0 = now_s();
    bool fallback = false;
    s.smpl = make_sampler(model, s.job->pr.sampling, pc.seed, fallback);
    if (fallback) {
        s.sampler_fallback = true;
        stats.n_fallback++;
    }
    {
        const llama_tokens & toks = s.job->pr.tokens;
        const int n_history = common_sampler_n_history(s.smpl);
        for (int i = std::max(0, pc.pos - n_history); i < pc.pos; i++) {
            common_sampler_accept(s.smpl, toks[i], false);
        }
    }
    stats.n_smpl++;
    stats.t_smpl += now_s() - t0;

    const llama_sampling_row row = llama_get_sampling_row_ith(ctx, out_idx);
    if (row.logits == nullptr) {
        dsp_die("slot %d: the context gave no whole row of logits at the cut", s.seq);
    }
    topk_row(row.logits, s.topk_buf);
    const llama_token id = common_sampler_sample(s.smpl, vocab, row, false);
    accept(s, id);
    s.seg.tokens.push_back(id);
}

void recorder::begin_segment(slot_t & s) {
    const plan_req & r  = *s.job->req;
    const plan_cut & pc = r.cuts[s.next_cut];
    const llama_token id = s.seg.tokens[0];

    if (is_stop(id)) {
        end_segment(s, "stop");
        return;
    }

    // the way back to the cut, unless the request ends here
    if (pc.pos < r.n_tokens) {
        s.cut_trip = opt.trip_every > 0 && stats.n_ckpt % opt.trip_every == 0;
        const double t0 = now_s();
        s.cut_ckpt.clear();
        s.cut_ckpt.update_pos(pc.pos, llama_memory_seq_pos_min(mem, s.seq), llama_memory_seq_pos_max(mem, s.seq));
        s.cut_ckpt.update_tgt(ctx, s.seq, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        s.have_cut_ckpt = true;
        stats.n_ckpt++;
        stats.t_ckpt     += now_s() - t0;
        stats.bytes_ckpt += (long long) s.cut_ckpt.size();
        stats.n_host++;
        stats.copies_host += (long long) llama_state_seq_n_copies(ctx);
    }

    s.pending = id;
    s.state   = S_GEN;
}

void recorder::end_segment(slot_t & s, const char * why) {
    const plan_req & r = *s.job->req;

    s.seg.stop        = why;
    s.seg.states_row0 = s.gen_rows;
    s.seg.topk_row0   = s.topk_rows;
    if (std::string(why) == "stop") {
        stats.seg_stop++;
    } else {
        stats.seg_length++;
    }
    if (s.gen_buf.size() != (size_t) s.seg.n_states * row_bytes || s.topk_buf.size() != (size_t) (s.seg.n_states + 1) * topk_bytes) {
        dsp_die("slot %d: a continuation's rows do not add up", s.seq);
    }
    s.gen_rows  += s.seg.n_states;
    s.topk_rows += s.seg.n_states + 1;
    if (!s.gen_buf.empty()) {
        writer->data(s.f_gen_states, std::move(s.gen_buf));
    }
    writer->data(s.f_gen_topk, std::move(s.topk_buf));
    s.gen_buf  = std::vector<uint8_t>();
    s.topk_buf = std::vector<uint8_t>();

    common_sampler_free(s.smpl);
    s.smpl = nullptr;

    const int  cut     = s.seg.cut;
    const bool decoded = s.seg.n_states > 0;
    s.segments.push_back(std::move(s.seg));
    s.seg = segment_t();
    s.next_cut++;

    if (cut == r.n_tokens) {
        finish_request(s);
        return;
    }
    if (decoded) {
        const double t0 = now_s();
        if (!s.have_cut_ckpt || !s.cut_ckpt.load_tgt(ctx, s.seq, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY)) {
            dsp_die("slot %d: the state at the cut %d does not load", s.seq, cut);
        }
        if (!llama_memory_seq_rm(mem, s.seq, cut, -1)) {
            dsp_die("slot %d: cannot drop the continuation from %d on", s.seq, cut);
        }
        stats.n_restore++;
        stats.t_restore += now_s() - t0;
        verify_touch(s);
        if (s.cut_trip) {
            const double    t1     = now_s();
            const long long differ = v_trips_differ;
            verify_trip(s, "cut", s.cut_ckpt.data_tgt.data(), s.cut_ckpt.data_tgt.size());
            stats.n_host++;
            stats.copies_host += (long long) llama_state_seq_n_copies(ctx);
            stats.n_trip++;
            stats.t_trip += now_s() - t1;
            if (v_trips_differ != differ) {
                dsp_die("slot %d: the state after the way back to the cut %d is not the state saved before it (verify.txt)", s.seq, cut);
            }
        } else if (opt.verify) {
            verify_trip(s, "cut", s.cut_ckpt.data_tgt.data(), s.cut_ckpt.data_tgt.size());
        }
    }
    s.cut_ckpt.clear();
    s.have_cut_ckpt = false;
    s.cut_trip = false;
    s.state = S_PREFILL;
}

void recorder::finish_request(slot_t & s) {
    const plan_req & r = *s.job->req;

    std::string text = "{\"id\":" + jq(r.id) + ",\"file\":" + jq(r.file) +
        ",\"n_tokens\":" + std::to_string(r.n_tokens) +
        ",\"shared_len\":" + std::to_string(r.shared_len) +
        ",\"states_rows\":" + std::to_string(s.states_rows) +
        ",\"gen_rows\":" + std::to_string(s.gen_rows) +
        ",\"topk_rows\":" + std::to_string(s.topk_rows) +
        ",\"grammar\":" + (s.job->pr.grammar ? "true" : "false") +
        ",\"sampler_fallback\":" + (s.sampler_fallback ? "true" : "false") +
        ",\"segments\":[";
    for (size_t i = 0; i < s.segments.size(); i++) {
        const segment_t & g = s.segments[i];
        if (i > 0) {
            text += ",";
        }
        text += "{\"cut\":" + std::to_string(g.cut) +
            ",\"g_max\":" + std::to_string(g.g_max) +
            ",\"seed\":" + std::to_string(g.seed) +
            ",\"n_states\":" + std::to_string(g.n_states) +
            ",\"states_row0\":" + std::to_string(g.states_row0) +
            ",\"topk_row0\":" + std::to_string(g.topk_row0) +
            ",\"stop\":" + jq(g.stop) +
            ",\"tokens\":" + ints_json(g.tokens) + "}";
    }
    text += "]}\n";

    if (s.states_rows != (long long) r.n_tokens - r.shared_len) {
        dsp_die("request %s: %lld state rows for %d positions", r.id.c_str(), s.states_rows, r.n_tokens - r.shared_len);
    }

    writer->close(s.f_states);
    if (s.f_gen_states != nullptr) {
        writer->close(s.f_gen_states);
    }
    if (s.f_gen_topk != nullptr) {
        writer->close(s.f_gen_topk);
    }
    writer->line(f_index, std::move(text));
    s.f_states = s.f_gen_states = s.f_gen_topk = nullptr;

    // the slot back at what it keeps of the donor prefix, its other cells returned to the pool
    if (s.level > 0) {
        restore_keep(s, s.level - 1);
    } else {
        llama_memory_seq_rm(mem, s.seq, -1, -1);
        verify_touch(s);
    }

    s.job.reset();
    s.segments.clear();
    s.state = S_IDLE;
    stats.n_done++;
}

void recorder::g_step() {
    std::vector<slot_t *> gen;
    for (slot_t & s : slots) {
        if (s.state == S_GEN) {
            gen.push_back(&s);
        }
    }

    llama_batch_ext_clear(batch);
    for (slot_t * sp : gen) {
        slot_t & s = *sp;
        const int32_t idx = llama_batch_ext_add_token(batch, s.seq, s.pending);
        if (idx < 0) {
            dsp_die("slot %d: the batch refused a generated token (%d)", s.seq, idx);
        }
        const llama_pos p = s.seg.cut + s.seg.n_states;
        llama_batch_ext_set_pos(batch, idx, &p);
        llama_batch_ext_set_output_logits(batch, idx, true);
        s.out_idx = idx;
    }

    if (opt.verify) {
        verify_look("before-G", {});
    }
    double t0 = now_s();
    const int32_t rc = llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch);
    if (rc != 0) {
        dsp_die("decode of %zu generated tokens failed (%d)", gen.size(), rc);
    }
    double t1 = now_s();
    stats.g_steps++;
    stats.t_g_decode += t1 - t0;
    stats.tok_gen    += (long long) gen.size();

    std::vector<const float *> tap(opt.taps.size());
    for (size_t t = 0; t < opt.taps.size(); t++) {
        tap[t] = llama_get_embeddings_layer_inp(ctx, (uint32_t) opt.taps[t]);
        if (tap[t] == nullptr) {
            dsp_die("the context has no states for layer input %d", opt.taps[t]);
        }
    }
    std::vector<llama_sampling_row> rows(gen.size());
    for (size_t k = 0; k < gen.size(); k++) {
        rows[k] = llama_get_sampling_row_ith(ctx, gen[k]->out_idx);
        if (rows[k].logits == nullptr) {
            dsp_die("slot %d: the context gave no whole row of logits", gen[k]->seq);
        }
    }

    // each slot's row: its states, the top of its distribution, and its next token from its own sampler
    parallel_for(opt.threads, gen.size(), [&](size_t b, size_t e) {
        for (size_t k = b; k < e; k++) {
            slot_t & s = *gen[k];
            try {
                const size_t at = s.gen_buf.size();
                s.gen_buf.resize(at + row_bytes);
                uint16_t * dst = (uint16_t *) (s.gen_buf.data() + at);
                for (size_t t = 0; t < tap.size(); t++) {
                    row_to_bf16(tap[t] + (size_t) s.out_idx * n_embd, dst + t * (size_t) n_embd, n_embd);
                }
                topk_row(rows[k].logits, s.topk_buf);
                s.sampled = common_sampler_sample(s.smpl, vocab, rows[k], false);
                accept(s, s.sampled);
            } catch (const std::exception & ex) {
                dsp_die("slot %d: sampling failed: %s", s.seq, ex.what());
            }
        }
    });

    if (opt.verify) {
        std::vector<char> held(slots.size(), 0);
        for (const slot_t * sp : gen) {
            held[sp->seq] = 1;
        }
        verify_look("after-G", held);
    }

    for (slot_t * sp : gen) {
        slot_t & s = *sp;
        s.seg.n_states++;
        s.seg.tokens.push_back(s.sampled);
        if (is_stop(s.sampled)) {
            end_segment(s, "stop");
        } else if (s.seg.n_states >= s.seg.g_max) {
            end_segment(s, "length");
        } else {
            s.pending = s.sampled;
        }
    }
    stats.t_g_post += now_s() - t1;
}

void recorder::progress(bool force) {
    const double t = now_s();
    if (!force && t - t_prev < opt.log_secs) {
        return;
    }
    const double dt = std::max(1e-6, t - t_prev);
    const stats_t & a = stats_prev;
    const stats_t & b = stats;
    int n_warm = 0, n_pre = 0, n_gen = 0;
    long long cells = 0;
    for (const slot_t & s : slots) {
        n_warm += s.state == S_WARMUP;
        n_pre  += s.state == S_PREFILL;
        n_gen  += s.state == S_GEN;
        cells  += held(s);
    }
    const long long d_p = (b.tok_prefill + b.tok_warm) - (a.tok_prefill + a.tok_warm);
    const long long d_g = b.tok_gen - a.tok_gen;
    const long long d_ps = b.p_steps - a.p_steps;
    const long long d_gs = b.g_steps - a.g_steps;
    const long long d_ck = b.n_ckpt - a.n_ckpt;
    const long long d_rs = b.n_restore - a.n_restore;
    const long long d_h  = b.n_host - a.n_host;
    say("capture: %lld/%zu requests | read %lld (+%lld warm) gen %lld | %.0fs: read %.0f tok/s, gen %.0f tok/s | "
        "P %lld steps %.0f ms + %.0f ms host | G %lld steps %.1f rows %.1f ms + %.1f ms host | "
        "cut ckpt %lld x %.1f ms %.1f MB | restore %lld x %.1f ms | now ckpt %.0f ms restore %.0f ms, %.0f copies a host state | "
        "trips %lld x %.0f ms | sampler %lld x %.1f ms (%lld fallbacks) | "
        "slots warm %d read %d gen %d | pool %lld/%lld | unwritten %.2f GB | written %.1f GB",
        b.n_done, todo.size(), b.tok_prefill, b.tok_warm, b.tok_gen,
        dt, d_p / dt, d_g / dt,
        d_ps, d_ps > 0 ? 1e3 * (b.t_p_decode - a.t_p_decode) / d_ps : 0.0, d_ps > 0 ? 1e3 * (b.t_p_post - a.t_p_post) / d_ps : 0.0,
        d_gs, d_gs > 0 ? (double) d_g / d_gs : 0.0, d_gs > 0 ? 1e3 * (b.t_g_decode - a.t_g_decode) / d_gs : 0.0,
        d_gs > 0 ? 1e3 * (b.t_g_post - a.t_g_post) / d_gs : 0.0,
        b.n_ckpt, b.n_ckpt > 0 ? 1e3 * b.t_ckpt / b.n_ckpt : 0.0, b.n_ckpt > 0 ? b.bytes_ckpt / 1e6 / b.n_ckpt : 0.0,
        b.n_restore, b.n_restore > 0 ? 1e3 * b.t_restore / b.n_restore : 0.0,
        d_ck > 0 ? 1e3 * (b.t_ckpt - a.t_ckpt) / d_ck : 0.0, d_rs > 0 ? 1e3 * (b.t_restore - a.t_restore) / d_rs : 0.0,
        d_h > 0 ? (double) (b.copies_host - a.copies_host) / d_h : 0.0,
        b.n_trip, b.n_trip > 0 ? 1e3 * b.t_trip / b.n_trip : 0.0,
        b.n_smpl, b.n_smpl > 0 ? 1e3 * b.t_smpl / b.n_smpl : 0.0, b.n_fallback,
        n_warm, n_pre, n_gen, cells, pool,
        writer->pending() / 1073741824.0, writer->written() / 1e9);
    stats_prev = stats;
    t_prev     = t;
}

void recorder::write_summary() {
    const double dt = now_s() - t_start;
    json j = json::object();
    j["seconds"]        = dt;
    j["requests"]       = stats.n_done;
    j["tokens_read"]    = stats.tok_prefill;
    j["tokens_warm"]    = stats.tok_warm;
    j["tokens_gen"]     = stats.tok_gen;
    j["p_steps"]        = stats.p_steps;
    j["g_steps"]        = stats.g_steps;
    j["p_decode_s"]     = stats.t_p_decode;
    j["p_host_s"]       = stats.t_p_post;
    j["g_decode_s"]     = stats.t_g_decode;
    j["g_host_s"]       = stats.t_g_post;
    j["cut_ckpts"]      = stats.n_ckpt;
    j["cut_ckpt_s"]     = stats.t_ckpt;
    j["cut_ckpt_bytes"] = stats.bytes_ckpt;
    j["keeps"]          = stats.n_keep;
    j["keep_s"]         = stats.t_keep;
    j["keep_bytes"]     = stats.bytes_keep;
    j["restores"]       = stats.n_restore;
    j["restore_s"]      = stats.t_restore;
    j["cut_trips"]      = stats.n_trip;
    j["cut_trip_s"]     = stats.t_trip;
    j["host_states"]    = stats.n_host;
    j["host_state_copies"] = stats.copies_host;
    j["samplers"]       = stats.n_smpl;
    j["sampler_s"]      = stats.t_smpl;
    j["sampler_fallbacks"] = stats.n_fallback;
    j["grammar_off"]       = stats.n_grammar_off;
    j["segments_stop"]  = stats.seg_stop;
    j["segments_length"] = stats.seg_length;
    j["bytes_written"]  = writer->written();
    FILE * f = open_file(opt.out + "/summary.json", "wb");
    const std::string text = j.dump(2) + "\n";
    fwrite(text.data(), 1, text.size(), f);
    fclose(f);
}

int recorder::run() {
    if (opt.requests.empty() || opt.out.empty() || opt.plan.empty()) {
        dsp_die("capture needs --requests, --plan and --out");
    }

    // server_output_limits()
    {
        const common_speculative_output_limits lim =
            common_speculative_get_output_limits(params.n_batch, params.n_parallel, common_speculative_n_max(&params.speculative));
        params.n_outputs_max         = std::max<int32_t>(1, lim.total);
        params.n_outputs_max_per_seq = std::max<int32_t>(1, lim.per_seq);
        say("capture: outputs a batch %d, a sequence %d", (int) params.n_outputs_max, (int) params.n_outputs_max_per_seq);
        if ((int) params.n_outputs_max < params.n_parallel) {
            dsp_die("the context gives %d outputs a batch, fewer than the %d slots", (int) params.n_outputs_max, params.n_parallel);
        }
    }

    llama_init = common_init_from_params(params);
    model = llama_init ? llama_init->model() : nullptr;
    ctx   = llama_init ? llama_init->context() : nullptr;
    if (model == nullptr || ctx == nullptr) {
        dsp_die("cannot load %s", params.model.path.c_str());
    }
    vocab   = llama_model_get_vocab(model);
    mem     = llama_get_memory(ctx);
    n_embd  = llama_model_n_embd(model);
    n_vocab = llama_vocab_n_tokens(vocab);
    n_batch = (int) llama_n_batch(ctx);
    n_step  = std::min(n_batch, opt.step_rows > 0 ? opt.step_rows : (int) llama_n_ubatch(ctx));
    n_chunk = std::min(n_step, params.n_prompt_chunk > 0 ? (int) params.n_prompt_chunk : n_step);
    n_slots = params.n_parallel;
    pool    = (long long) llama_n_ctx(ctx);
    seq_limit = (long long) llama_n_ctx_seq(ctx);
    row_bytes  = opt.taps.size() * (size_t) n_embd * 2;
    topk_bytes = 4 + 8 * (size_t) opt.topk;

    const int n_layer = llama_model_n_layer(model);
    for (int t : opt.taps) {
        if (t < 0 || t > n_layer) {
            dsp_die("tap %d is outside the model's %d layers", t, n_layer);
        }
        llama_set_embeddings_layer_inp(ctx, (uint32_t) t, true);
    }

    init_chat_env(env, model, params, 0);
    say("capture: %d slots, batch %d, %d prompt rows a decode in chunks of %d, pool %lld cells, %lld a sequence, width %d, "
        "vocabulary %d, %zu bytes a position",
        n_slots, n_batch, n_step, n_chunk, pool, seq_limit, n_embd, n_vocab, row_bytes);

    load_inputs();
    env.max_tokens = plan.max_tokens;
    open_outputs();

    slots.resize((size_t) n_slots);
    for (int i = 0; i < n_slots; i++) {
        slots[i].seq = i;
        slots[i].keep.resize(keeps.size());
    }
    watch.resize((size_t) n_slots);
    batch = llama_batch_ext_init(ctx);
    if (batch == nullptr) {
        dsp_die("cannot make a batch");
    }

    t_start = t_prev = now_s();
    record_prefix();
    feeder = std::thread([this] { feeder_main(); });

    while (true) {
        if (writer->failed()) {
            dsp_die("writing the recording failed: %s", writer->error().c_str());
        }
        assign_jobs();

        bool any_read = false, any_gen = false;
        for (const slot_t & s : slots) {
            any_read = any_read || s.state == S_WARMUP || s.state == S_PREFILL;
            any_gen  = any_gen  || s.state == S_GEN;
        }
        if (any_read) {
            p_step();
        } else if (any_gen) {
            g_step();
        } else {
            std::unique_lock<std::mutex> lock(q_mutex);
            if (q.empty()) {
                if (feeder_done) {
                    break;
                }
                q_cv_item.wait_for(lock, std::chrono::milliseconds(200));
            }
        }
        progress(false);
    }

    feeder.join();
    progress(true);
    if (f_prefix != nullptr) {
        writer->close(f_prefix);
    }
    writer->close(f_index);
    writer->finish();
    if (writer->failed()) {
        dsp_die("writing the recording failed: %s", writer->error().c_str());
    }
    write_summary();
    if (opt.trip_every > 0) {
        say("capture: %lld ways back to a cut read again, each the state saved before its continuation", stats.n_trip);
    }
    if (opt.verify) {
        if (f_verify == nullptr) {
            f_verify = open_file(opt.out + "/verify.txt", "ab");
        }
        fprintf(f_verify, "S %lld %lld %lld %lld\n", v_compared, v_moved, v_trips, v_trips_differ);
        fclose(f_verify);
        f_verify = nullptr;
        say("capture: states looked at: %lld sequences left alone by a decode or between two, %lld of them changed; %lld loaded "
            "states read back, %lld of them not as loaded",
            v_compared, v_moved, v_trips, v_trips_differ);
    }
    say("capture: done, %lld requests in %.0f s", stats.n_done, now_s() - t_start);

    llama_batch_ext_free(batch);
    return 0;
}

// ---------------------------------------------------------------------------------------------------------------------

int main(int argc, char ** argv) {
    int split = argc;
    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--") == 0) {
            split = i;
            break;
        }
    }

    tool_opts opt;
    if (argc < 2 || !parse_tool_opts(split - 1, argv + 1, opt)) {
        usage();
        return 2;
    }

    std::vector<std::string> server_argv;
    std::vector<char *> argv2;
    argv2.push_back(argv[0]);
    for (int i = split + 1; i < argc; i++) {
        argv2.push_back(argv[i]);
        server_argv.push_back(argv[i]);
    }

    common_init();

    common_params params;
    if (!common_params_parse((int) argv2.size(), argv2.data(), params, LLAMA_EXAMPLE_SERVER)) {
        return 2;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    int rc = 0;
    if (opt.mode == "prepare") {
        rc = run_prepare(opt, params);
    } else {
        recorder rec(opt, params, server_argv);
        rc = rec.run();
    }

    llama_backend_free();
    return rc;
}
