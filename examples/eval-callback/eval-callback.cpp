#include "arg.h"
#include "common.h"
#include "debug.h"
#include "log.h"
#include "llama.h"
#include "ggml-backend.h"

#include <cctype>
#include <clocale>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <regex>
#include <string>
#include <vector>

// Opt-in raw captures for comparing the first differing activation across device layouts. The budget bounds disk and readback.
struct tensor_dump {
    std::filesystem::path directory;
    std::regex filter;
    std::ofstream manifest;
    size_t bytes = 0;
    size_t count = 0;
    bool failed = false;
    static constexpr size_t budget = size_t(2)*1024*1024*1024;

    tensor_dump(const char * path, const char * pattern) : directory(path), filter(pattern) {
        std::filesystem::create_directories(directory);
        manifest.open(directory / "tensors.tsv", std::ios::trunc);
        if (!manifest) { throw std::runtime_error("cannot open tensor manifest"); }
        manifest << "index\tname\ttype\tne0\tne1\tne2\tne3\tnb0\tnb1\tnb2\tnb3\tfile\n";
    }

    static bool callback(ggml_tensor * tensor, bool ask, void * userdata) {
        auto & self = *static_cast<tensor_dump *>(userdata);
        if (ask) { return std::regex_search(tensor->name, self.filter); }
        const size_t size = ggml_nbytes(tensor);
        if (size > budget - self.bytes) {
            LOG_ERR("tensor dump exceeded its 2 GiB budget at %s\n", tensor->name);
            self.failed = true;
            return false;
        }
        std::string name = tensor->name;
        for (char & c : name) { if (!std::isalnum((unsigned char) c) && c != '_' && c != '-' && c != '.') c = '_'; }
        const std::string file = std::to_string(self.count) + "-" + name + ".bin";
        std::vector<char> data(size);
        ggml_backend_tensor_get(tensor, data.data(), 0, size);
        std::ofstream output(self.directory / file, std::ios::binary | std::ios::trunc);
        output.write(data.data(), data.size());
        output.close();
        self.manifest << self.count << '\t' << name << '\t' << ggml_type_name(tensor->type);
        for (const auto n : tensor->ne) { self.manifest << '\t' << n; }
        for (const auto n : tensor->nb) { self.manifest << '\t' << n; }
        self.manifest << '\t' << file << '\n';
        self.manifest.flush();
        self.failed = !output || !self.manifest;
        self.bytes += size;
        ++self.count;
        return !self.failed;
    }
};

static bool run(llama_context * ctx, const common_params & params) {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    const bool add_bos = llama_vocab_get_add_bos(vocab);

    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, add_bos, true);

    if (tokens.empty()) {
        LOG_ERR("%s : there are not input tokens to process - (try to provide a prompt with '-p')\n", __func__);
        return false;
    }

    LOG_INF("number of input tokens = %zu\n", tokens.size());
    for (size_t i = 0; i < tokens.size(); ++i) {
        LOG_INF("  %d\n", tokens[i]);
    }

    common_batch batch = common_batch_get_one(ctx, tokens);
    if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get())) {
        LOG_ERR("%s : failed to eval\n", __func__);
        return false;
    }

    return true;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_debug_cb_user_data cb_data;

    common_params params;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    // pass the callback to the backend scheduler
    // it will be executed for each node during the graph computation
    params.cb_eval = common_debug_cb_eval;
    params.cb_eval_user_data = &cb_data;
    params.warmup = false;
    std::unique_ptr<tensor_dump> dump;
    const char * dump_path = std::getenv("LLAMA_EVAL_DUMP");
    if (dump_path) {
        const char * filter = std::getenv("LLAMA_EVAL_FILTER");
        if (!filter || !filter[0]) {
            LOG_ERR("LLAMA_EVAL_DUMP requires LLAMA_EVAL_FILTER to select tensor names\n");
            return 1;
        }
        try {
            dump = std::make_unique<tensor_dump>(dump_path, filter);
        } catch (const std::exception & e) {
            LOG_ERR("tensor dump: %s\n", e.what());
            return 1;
        }
        params.cb_eval = tensor_dump::callback;
        params.cb_eval_user_data = dump.get();
    }

    // init
    auto llama_init = common_init_from_params(params);

    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();

    if (model == nullptr || ctx == nullptr) {
        LOG_ERR("%s : failed to init\n", __func__);
        return 1;
    }

    // print system information
    {
        LOG_INF("\n");
        LOG_INF("%s\n", common_params_get_system_info(params).c_str());
        LOG_INF("\n");
    }

    bool OK = run(ctx, params);
    if (!OK || (dump && (dump->failed || dump->count == 0))) {
        return 1;
    }

    LOG("\n");
    llama_perf_context_print(ctx);

    llama_backend_free();

    return 0;
}
