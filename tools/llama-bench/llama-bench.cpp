#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cinttypes>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iterator>
#include <map>
#include <numeric>
#include <regex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <unordered_set>

#include "arg.h"
#include "build-info.h"
#include "common.h"
#include "download.h"
#include "fit.h"
#include "ggml.h"
#include "llama.h"
#include "log.h"

#ifdef _WIN32
#    define WIN32_LEAN_AND_MEAN
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#endif

// utils
static uint64_t get_time_ns() {
    using clock = std::chrono::high_resolution_clock;
    return std::chrono::nanoseconds(clock::now().time_since_epoch()).count();
}

static bool tensor_buft_override_equal(const llama_model_tensor_buft_override& a, const llama_model_tensor_buft_override& b) {
    if (a.pattern != b.pattern) {
        // cString comparison that may be null
        if (a.pattern == nullptr || b.pattern == nullptr) {
            return false;
        }
        if (strcmp(a.pattern, b.pattern) != 0) {
            return false;
        }
    }
    if (a.buft != b.buft) {
        return false;
    }
    return true;
}

static bool vec_tensor_buft_override_equal(const std::vector<llama_model_tensor_buft_override>& a, const std::vector<llama_model_tensor_buft_override>& b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (size_t i = 0; i < a.size(); i++) {
        if (!tensor_buft_override_equal(a[i], b[i])) {
            return false;
        }
    }
    return true;
}

static bool vec_vec_tensor_buft_override_equal(const std::vector<std::vector<llama_model_tensor_buft_override>>& a, const std::vector<std::vector<llama_model_tensor_buft_override>>& b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (size_t i = 0; i < a.size(); i++) {
        if (!vec_tensor_buft_override_equal(a[i], b[i])) {
            return false;
        }
    }
    return true;
}

template <class T> static std::string join(const std::vector<T> & values, const std::string & delim) {
    std::ostringstream str;
    for (size_t i = 0; i < values.size(); i++) {
        str << values[i];
        if (i < values.size() - 1) {
            str << delim;
        }
    }
    return str.str();
}

template <typename T, typename F> static std::vector<std::string> transform_to_str(const std::vector<T> & values, F f) {
    std::vector<std::string> str_values;
    std::transform(values.begin(), values.end(), std::back_inserter(str_values), f);
    return str_values;
}

template <typename T> static T avg(const std::vector<T> & v) {
    if (v.empty()) {
        return 0;
    }
    T sum = std::accumulate(v.begin(), v.end(), T(0));
    return sum / (T) v.size();
}

template <typename T> static T stdev(const std::vector<T> & v) {
    if (v.size() <= 1) {
        return 0;
    }
    T mean   = avg(v);
    T sq_sum = std::inner_product(v.begin(), v.end(), v.begin(), T(0));
    T stdev  = std::sqrt(sq_sum / (T) (v.size() - 1) - mean * mean * (T) v.size() / (T) (v.size() - 1));
    return stdev;
}

static std::string get_cpu_info() {
    std::vector<std::string> cpu_list;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        auto * dev      = ggml_backend_dev_get(i);
        auto   dev_type = ggml_backend_dev_type(dev);
        if (dev_type == GGML_BACKEND_DEVICE_TYPE_CPU || dev_type == GGML_BACKEND_DEVICE_TYPE_ACCEL) {
            cpu_list.push_back(ggml_backend_dev_description(dev));
        }
    }
    return join(cpu_list, ", ");
}

static std::string get_gpu_info() {
    std::vector<std::string> gpu_list;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        auto * dev      = ggml_backend_dev_get(i);
        auto   dev_type = ggml_backend_dev_type(dev);
        if (dev_type == GGML_BACKEND_DEVICE_TYPE_GPU || dev_type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
            gpu_list.push_back(ggml_backend_dev_description(dev));
        }
    }
    return join(gpu_list, ", ");
}

static std::vector<ggml_backend_dev_t> parse_devices_arg(const std::string & value) {
    std::vector<ggml_backend_dev_t> devices;
    std::string                     trimmed = string_strip(value);
    if (trimmed.empty()) {
        throw std::invalid_argument("no devices specified");
    }
    if (trimmed == "auto") {
        return devices;
    }

    auto dev_names = string_split<std::string>(trimmed, '/');
    if (dev_names.size() == 1 && string_strip(dev_names[0]) == "none") {
        devices.push_back(nullptr);
        return devices;
    }

    for (auto & name : dev_names) {
        std::string dev_name = string_strip(name);
        if (dev_name.empty()) {
            throw std::invalid_argument("invalid device specification");
        }
        auto * dev = ggml_backend_dev_by_name(dev_name.c_str());
        if (!dev || ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
            throw std::invalid_argument(string_format("invalid device: %s", dev_name.c_str()));
        }
        devices.push_back(dev);
    }

    devices.push_back(nullptr);
    return devices;
}

static void register_rpc_server_list(const std::string & servers) {
    auto rpc_servers = string_split<std::string>(servers, ',');
    if (rpc_servers.empty()) {
        throw std::invalid_argument("no RPC servers specified");
    }

    auto * rpc_reg = ggml_backend_reg_by_name("RPC");
    if (!rpc_reg) {
        throw std::invalid_argument("failed to find RPC backend");
    }

    using add_rpc_server_fn = ggml_backend_reg_t (*)(const char * endpoint);
    auto * ggml_backend_rpc_add_server_fn = (add_rpc_server_fn) ggml_backend_reg_get_proc_address(rpc_reg, "ggml_backend_rpc_add_server");
    if (!ggml_backend_rpc_add_server_fn) {
        throw std::invalid_argument("failed to find RPC add server function");
    }
    for (const auto & server : rpc_servers) {
        auto reg = ggml_backend_rpc_add_server_fn(server.c_str());
        ggml_backend_register(reg);
    }
}

static std::string devices_to_string(const std::vector<ggml_backend_dev_t> & devices) {
    if (devices.empty()) {
        return "auto";
    }

    if (devices.size() == 1 && devices[0] == nullptr) {
        return "none";
    }

    std::vector<std::string> names;
    for (auto * dev : devices) {
        if (dev == nullptr) {
            break;
        }
        names.push_back(ggml_backend_dev_name(dev));
    }

    return join(names, "/");
}

// command line params
enum output_formats { NONE, CSV, JSON, JSONL, MARKDOWN, SQL };

static const char * output_format_str(output_formats format) {
    switch (format) {
        case NONE:
            return "none";
        case CSV:
            return "csv";
        case JSON:
            return "json";
        case JSONL:
            return "jsonl";
        case MARKDOWN:
            return "md";
        case SQL:
            return "sql";
        default:
            GGML_ABORT("invalid output format");
    }
}

static bool output_format_from_str(const std::string & s, output_formats & format) {
    if (s == "none") {
        format = NONE;
    } else if (s == "csv") {
        format = CSV;
    } else if (s == "json") {
        format = JSON;
    } else if (s == "jsonl") {
        format = JSONL;
    } else if (s == "md") {
        format = MARKDOWN;
    } else if (s == "sql") {
        format = SQL;
    } else {
        return false;
    }
    return true;
}

static const char * split_mode_str(llama_split_mode mode) {
    switch (mode) {
        case LLAMA_SPLIT_MODE_NONE:
            return "none";
        case LLAMA_SPLIT_MODE_LAYER:
            return "layer";
        case LLAMA_SPLIT_MODE_ROW:
            return "row";
        case LLAMA_SPLIT_MODE_TENSOR:
            return "tensor";
        default:
            GGML_ABORT("invalid split mode");
    }
}

static const char * lazy_mode_str(llama_lazy_mode mode) {
    switch (mode) {
        case LLAMA_LAZY_MODE_OFF:
            return "off";
        case LLAMA_LAZY_MODE_AUTO:
            return "auto";
        case LLAMA_LAZY_MODE_ON:
            return "on";
        default:
            GGML_ABORT("invalid lazy mode");
    }
}

static std::string pair_str(const std::pair<int, int> & p) {
    static char buf[32];
    snprintf(buf, sizeof(buf), "%d,%d", p.first, p.second);
    return buf;
}

static std::vector<int> parse_int_range(const std::string & s, bool allow_negative = false) {
    // first[-last[(+|*)step]]
    std::regex range_regex(allow_negative
        ? R"(^(-?\d+)(?:-(\d+)(?:([\+|\*])(\d+))?)?(?:,|$))"
        : R"(^(\d+)(?:-(\d+)(?:([\+|\*])(\d+))?)?(?:,|$))");

    std::smatch match;
    std::string::const_iterator search_start(s.cbegin());
    std::vector<int> result;
    while (std::regex_search(search_start, s.cend(), match, range_regex)) {
        int  first = std::stoi(match[1]);
        int  last  = match[2].matched ? std::stoi(match[2]) : first;
        char op    = match[3].matched ? match[3].str()[0] : '+';
        int  step  = match[4].matched ? std::stoi(match[4]) : 1;

        for (int i = first; i <= last;) {
            result.push_back(i);

            int prev_i = i;

            if (op == '+') {
                i += step;
            } else if (op == '*') {
                i *= step;
            } else {
                throw std::invalid_argument("invalid range format");
            }

            if (i <= prev_i) {
                throw std::invalid_argument("invalid range");
            }
        }
        search_start = match.suffix().first;
    }

    if (search_start != s.cend()) {
        throw std::invalid_argument("invalid range format");
    }

    return result;
}

struct cmd_params {
    std::vector<std::string>         model;
    std::vector<std::string>         hf_repo;
    std::vector<std::string>         hf_file;
    std::string                      hf_token;
    bool                             offline;
    std::vector<int>                 n_prompt;
    std::vector<int>                 n_gen;
    std::vector<std::pair<int, int>> n_pg;
    std::vector<int>                 n_depth;
    std::vector<int>                 n_batch;
    std::vector<int>                 n_ubatch;
    std::vector<ggml_type>           type_k;
    std::vector<ggml_type>           type_v;
    std::vector<int>                 n_threads;
    std::vector<std::string>         cpu_mask;
    std::vector<bool>                cpu_strict;
    std::vector<int>                 poll;
    std::vector<int>                 n_gpu_layers;
    std::vector<int>                 n_cpu_moe;
    std::vector<llama_split_mode>    split_mode;
    std::vector<llama_load_mode>     load_mode;
    std::vector<llama_lazy_mode>     lazy_mode;
    std::vector<int>                 main_gpu;
    std::vector<bool>                no_kv_offload;
    std::vector<llama_flash_attn_type> flash_attn;
    std::vector<std::vector<ggml_backend_dev_t>> devices;
    std::vector<std::vector<float>>  tensor_split;
    std::vector<std::vector<llama_model_tensor_buft_override>> tensor_buft_overrides;
    std::vector<bool>                embeddings;
    std::vector<bool>                no_op_offload;
    std::vector<bool>                no_host;
    std::vector<bool>                repack;
    std::vector<size_t>              fit_params_target;
    std::vector<uint32_t>            fit_params_min_ctx;
    ggml_numa_strategy               numa;
    int                              reps;
    ggml_sched_priority              prio;
    int                              delay;
    bool                             verbose;
    bool                             progress;
    bool                             no_warmup;
    output_formats                   output_format;
    output_formats                   output_format_stderr;
};

static const cmd_params cmd_params_defaults = {
    /* model                */ { "models/7B/ggml-model-q4_0.gguf" },
    /* hf_repo              */ {},
    /* hf_file              */ {},
    /* hf_token             */ "",
    /* offline              */ false,
    /* n_prompt             */ { 512 },
    /* n_gen                */ { 128 },
    /* n_pg                 */ {},
    /* n_depth              */ { 0 },
    /* n_batch              */ { 2048 },
    /* n_ubatch             */ { 512 },
    /* type_k               */ { GGML_TYPE_F16 },
    /* type_v               */ { GGML_TYPE_F16 },
    /* n_threads            */ { common_cpu_get_num_math() },
    /* cpu_mask             */ { "0x0" },
    /* cpu_strict           */ { false },
    /* poll                 */ { 50 },
    /* n_gpu_layers         */ { -1 },
    /* n_cpu_moe            */ { 0 },
    /* split_mode           */ { LLAMA_SPLIT_MODE_LAYER },
    /* load_mode            */ { LLAMA_LOAD_MODE_AUTO },
    /* lazy_mode            */ { LLAMA_LAZY_MODE_AUTO },
    /* main_gpu             */ { 0 },
    /* no_kv_offload        */ { false },
    /* flash_attn           */ { LLAMA_FLASH_ATTN_TYPE_AUTO },
    /* devices              */ { {} },
    /* tensor_split         */ { std::vector<float>(llama_max_devices(), 0.0f) },
    /* tensor_buft_overrides*/ { std::vector<llama_model_tensor_buft_override>{ { nullptr, nullptr } } },
    /* embeddings           */ { false },
    /* no_op_offload        */ { false },
    /* no_host              */ { false },
    /* repack               */ { llama_model_default_params().use_extra_bufts },
    /* fit_params_target    */ { 0 },
    /* fit_params_min_ctx   */ { 0 },
    /* numa                 */ GGML_NUMA_STRATEGY_DISABLED,
    /* reps                 */ 5,
    /* prio                 */ GGML_SCHED_PRIO_NORMAL,
    /* delay                */ 0,
    /* verbose              */ false,
    /* progress             */ false,
    /* no_warmup            */ false,
    /* output_format        */ MARKDOWN,
    /* output_format_stderr */ NONE,
};

static void print_usage(int /* argc */, char ** argv) {
    printf("usage: %s [options]\n", argv[0]);
    printf("\n");
    printf("options:\n");
    printf("  -h, --help\n");
    printf("  --version                                   show version and build info\n");
    printf("  --numa <distribute|isolate|numactl|mirror>  numa mode (default: disabled)\n");
