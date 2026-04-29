#include "gpukmeans/io.hpp"
#include "gpukmeans/kmeans.hpp"
#include "gpukmeans/synthetic.hpp"

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct BenchmarkRow {
    int rows = 0;
    int dims = 0;
    int clusters = 0;
    int max_iterations = 0;
    int repeat = 0;
    std::uint64_t seed = 0;
    std::string backend;
    std::string kernel;
    std::string status = "ok";
    std::string error;
    gpukmeans::FitResult result;
};

struct BenchmarkConfig {
    std::vector<int> rows = {10000};
    std::vector<int> dims = {16};
    std::vector<int> clusters = {8};
    std::vector<gpukmeans::KernelStrategy> kernels = {
        gpukmeans::KernelStrategy::Naive,
        gpukmeans::KernelStrategy::Shared};
    std::string backend = "both";
    std::string output_path = "benchmarks.csv";
    std::string format = "csv";
    int max_iterations = 50;
    int repeats = 1;
    int device_id = 0;
    int threads_per_block = 256;
    float tolerance = 1.0e-4f;
    float cluster_stddev = 0.6f;
    std::uint64_t seed = 42;
};

void print_help() {
    std::cout
        << "GPU K-means benchmark runner\n\n"
        << "Usage:\n"
        << "  kmeans_benchmark --rows 10000,100000 --dims 2,16 --k 4,16 [options]\n\n"
        << "Options:\n"
        << "  --rows LIST                  Dataset sizes (default: 10000)\n"
        << "  --dims LIST                  Dimensionalities (default: 16)\n"
        << "  --k LIST                     Cluster counts (default: 8)\n"
        << "  --iterations N               Max iterations per run (default: 50)\n"
        << "  --repeats N                  Repeats per configuration (default: 1)\n"
        << "  --backend cpu|cuda|both      Backend set (default: both)\n"
        << "  --kernels LIST               CUDA kernels: naive,shared,warp,fused\n"
        << "  --seed N                     Base deterministic seed (default: 42)\n"
        << "  --tolerance VALUE            Convergence tolerance (default: 1e-4)\n"
        << "  --device N                   CUDA device id (default: 0)\n"
        << "  --threads N                  CUDA threads per block (default: 256)\n"
        << "  --cluster-stddev VALUE       Synthetic data cluster stddev (default: 0.6)\n"
        << "  --output PATH                Result path (default: benchmarks.csv)\n"
        << "  --format csv|json            Output format (default: csv)\n";
}

std::string require_value(int& i, int argc, char** argv) {
    if (i + 1 >= argc) {
        throw std::invalid_argument(std::string("missing value for ") + argv[i]);
    }
    return argv[++i];
}

int parse_int(const std::string& value, const std::string& name) {
    char* end = nullptr;
    const long parsed = std::strtol(value.c_str(), &end, 10);
    if (end == value.c_str() || *end != '\0') {
        throw std::invalid_argument("invalid integer for " + name + ": " + value);
    }
    return static_cast<int>(parsed);
}

float parse_float(const std::string& value, const std::string& name) {
    char* end = nullptr;
    const float parsed = std::strtof(value.c_str(), &end);
    if (end == value.c_str() || *end != '\0') {
        throw std::invalid_argument("invalid float for " + name + ": " + value);
    }
    return parsed;
}

std::uint64_t parse_u64(const std::string& value, const std::string& name) {
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(value.c_str(), &end, 10);
    if (end == value.c_str() || *end != '\0') {
        throw std::invalid_argument("invalid unsigned integer for " + name + ": " + value);
    }
    return static_cast<std::uint64_t>(parsed);
}

std::vector<std::string> split_list(const std::string& value) {
    std::vector<std::string> items;
    std::stringstream stream(value);
    std::string item;
    while (std::getline(stream, item, ',')) {
        if (!item.empty()) {
            items.push_back(item);
        }
    }
    return items;
}

std::vector<int> parse_int_list(const std::string& value, const std::string& name) {
    std::vector<int> values;
    for (const std::string& item : split_list(value)) {
        values.push_back(parse_int(item, name));
    }
    if (values.empty()) {
        throw std::invalid_argument("empty list for " + name);
    }
    return values;
}

std::vector<gpukmeans::KernelStrategy> parse_kernel_list(const std::string& value) {
    std::vector<gpukmeans::KernelStrategy> values;
    for (const std::string& item : split_list(value)) {
        values.push_back(gpukmeans::parse_kernel_strategy(item));
    }
    if (values.empty()) {
        throw std::invalid_argument("empty kernel list");
    }
    return values;
}

BenchmarkConfig parse_args(int argc, char** argv) {
    BenchmarkConfig config;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_help();
            std::exit(0);
        } else if (arg == "--rows") {
            config.rows = parse_int_list(require_value(i, argc, argv), arg);
        } else if (arg == "--dims") {
            config.dims = parse_int_list(require_value(i, argc, argv), arg);
        } else if (arg == "--k") {
            config.clusters = parse_int_list(require_value(i, argc, argv), arg);
        } else if (arg == "--iterations") {
            config.max_iterations = parse_int(require_value(i, argc, argv), arg);
        } else if (arg == "--repeats") {
            config.repeats = parse_int(require_value(i, argc, argv), arg);
        } else if (arg == "--backend") {
            config.backend = require_value(i, argc, argv);
        } else if (arg == "--kernels") {
            config.kernels = parse_kernel_list(require_value(i, argc, argv));
        } else if (arg == "--seed") {
            config.seed = parse_u64(require_value(i, argc, argv), arg);
        } else if (arg == "--tolerance") {
            config.tolerance = parse_float(require_value(i, argc, argv), arg);
        } else if (arg == "--device") {
            config.device_id = parse_int(require_value(i, argc, argv), arg);
        } else if (arg == "--threads") {
            config.threads_per_block = parse_int(require_value(i, argc, argv), arg);
        } else if (arg == "--cluster-stddev") {
            config.cluster_stddev = parse_float(require_value(i, argc, argv), arg);
        } else if (arg == "--output") {
            config.output_path = require_value(i, argc, argv);
        } else if (arg == "--format") {
            config.format = require_value(i, argc, argv);
        } else {
            throw std::invalid_argument("unknown argument: " + arg);
        }
    }
    return config;
}

bool wants_cpu(const std::string& backend) {
    return backend == "cpu" || backend == "both";
}

bool wants_cuda(const std::string& backend) {
    return backend == "cuda" || backend == "gpu" || backend == "both";
}

BenchmarkRow run_one(const gpukmeans::Matrix& data,
                     int clusters,
                     int max_iterations,
                     int repeat,
                     std::uint64_t seed,
                     gpukmeans::Backend backend,
                     gpukmeans::KernelStrategy kernel,
                     const BenchmarkConfig& config) {
    BenchmarkRow row;
    row.rows = data.rows;
    row.dims = data.cols;
    row.clusters = clusters;
    row.max_iterations = max_iterations;
    row.repeat = repeat;
    row.seed = seed;
    row.backend = gpukmeans::to_string(backend);
    row.kernel = backend == gpukmeans::Backend::Cpu ? "cpu" : gpukmeans::to_string(kernel);

    gpukmeans::KMeansOptions options;
    options.clusters = clusters;
    options.max_iterations = max_iterations;
    options.tolerance = config.tolerance;
    options.seed = seed;
    options.backend = backend;
    options.kernel = kernel;
    options.device_id = config.device_id;
    options.threads_per_block = config.threads_per_block;
    options.collect_history = false;

    try {
        row.result = gpukmeans::fit(data, clusters, options);
    } catch (const std::exception& ex) {
        row.status = "error";
        row.error = ex.what();
    }
    return row;
}

void write_csv(const std::string& path, const std::vector<BenchmarkRow>& rows) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("failed to open output file: " + path);
    }
    out << "rows,dims,k,max_iterations,repeat,seed,backend,kernel,status,error,"
           "iterations,converged,inertia,total_ms,init_ms,cpu_ms,h2d_ms,kernel_ms,d2h_ms\n";
    for (const BenchmarkRow& row : rows) {
        out << row.rows << ','
            << row.dims << ','
            << row.clusters << ','
            << row.max_iterations << ','
            << row.repeat << ','
            << row.seed << ','
            << row.backend << ','
            << row.kernel << ','
            << row.status << ','
            << '"' << row.error << '"' << ','
            << row.result.iterations << ','
            << (row.result.converged ? "true" : "false") << ','
            << row.result.inertia << ','
            << row.result.timing.total_ms << ','
            << row.result.timing.init_ms << ','
            << row.result.timing.cpu_ms << ','
            << row.result.timing.h2d_ms << ','
            << row.result.timing.kernel_ms << ','
            << row.result.timing.d2h_ms << '\n';
    }
}

void write_json_string(std::ostream& out, const std::string& value) {
    out << '"';
    for (char c : value) {
        if (c == '"' || c == '\\') {
            out << '\\' << c;
        } else {
            out << c;
        }
    }
    out << '"';
}

void write_json(const std::string& path, const std::vector<BenchmarkRow>& rows) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("failed to open output file: " + path);
    }
    out << "[\n";
    for (std::size_t i = 0; i < rows.size(); ++i) {
        const BenchmarkRow& row = rows[i];
        out << "  {\"rows\": " << row.rows
            << ", \"dims\": " << row.dims
            << ", \"k\": " << row.clusters
            << ", \"max_iterations\": " << row.max_iterations
            << ", \"repeat\": " << row.repeat
            << ", \"seed\": " << row.seed
            << ", \"backend\": ";
        write_json_string(out, row.backend);
        out << ", \"kernel\": ";
        write_json_string(out, row.kernel);
        out << ", \"status\": ";
        write_json_string(out, row.status);
        out << ", \"error\": ";
        write_json_string(out, row.error);
        out << ", \"iterations\": " << row.result.iterations
            << ", \"converged\": " << (row.result.converged ? "true" : "false")
            << ", \"inertia\": " << row.result.inertia
            << ", \"timing_ms\": {\"total\": " << row.result.timing.total_ms
            << ", \"init\": " << row.result.timing.init_ms
            << ", \"cpu\": " << row.result.timing.cpu_ms
            << ", \"h2d\": " << row.result.timing.h2d_ms
            << ", \"kernel\": " << row.result.timing.kernel_ms
            << ", \"d2h\": " << row.result.timing.d2h_ms << "}}";
        out << (i + 1 == rows.size() ? "\n" : ",\n");
    }
    out << "]\n";
}

} // namespace

int main(int argc, char** argv) {
    try {
        BenchmarkConfig config = parse_args(argc, argv);
        if (!wants_cpu(config.backend) && !wants_cuda(config.backend)) {
            throw std::invalid_argument("--backend must be cpu, cuda, or both");
        }

        std::vector<BenchmarkRow> results;
        for (int rows : config.rows) {
            for (int dims : config.dims) {
                for (int clusters : config.clusters) {
                    for (int repeat = 0; repeat < config.repeats; ++repeat) {
                        const std::uint64_t seed = config.seed + static_cast<std::uint64_t>(repeat);
                        gpukmeans::Matrix data =
                            gpukmeans::generate_blobs(rows, dims, clusters, seed, config.cluster_stddev);

                        if (wants_cpu(config.backend)) {
                            results.push_back(run_one(data,
                                                      clusters,
                                                      config.max_iterations,
                                                      repeat,
                                                      seed,
                                                      gpukmeans::Backend::Cpu,
                                                      gpukmeans::KernelStrategy::Naive,
                                                      config));
                        }
                        if (wants_cuda(config.backend)) {
                            for (gpukmeans::KernelStrategy kernel : config.kernels) {
                                results.push_back(run_one(data,
                                                          clusters,
                                                          config.max_iterations,
                                                          repeat,
                                                          seed,
                                                          gpukmeans::Backend::Cuda,
                                                          kernel,
                                                          config));
                            }
                        }
                    }
                }
            }
        }

        if (config.format == "json") {
            write_json(config.output_path, results);
        } else if (config.format == "csv") {
            write_csv(config.output_path, results);
        } else {
            throw std::invalid_argument("--format must be csv or json");
        }

        std::cout << "wrote " << results.size() << " benchmark rows to " << config.output_path << '\n';
        if (wants_cuda(config.backend) && !gpukmeans::cuda_backend_compiled()) {
            std::cout << "note: CUDA backend was not compiled; CUDA rows contain error status\n";
        }
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "error: " << ex.what() << '\n';
        std::cerr << "run with --help for usage\n";
        return 1;
    }
}
