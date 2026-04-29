#include "gpukmeans/io.hpp"
#include "gpukmeans/kmeans.hpp"
#include "gpukmeans/synthetic.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

struct CliConfig {
    gpukmeans::KMeansOptions options;
    gpukmeans::CsvReadOptions csv;
    std::string input_path;
    std::string assignments_path;
    std::string centroids_path;
    std::string json_path;
    int synthetic_rows = 0;
    int synthetic_cols = 0;
    float synthetic_stddev = 0.6f;
};

void print_help() {
    std::cout
        << "GPU K-means CLI\n\n"
        << "Usage:\n"
        << "  kmeans_cli --input data.csv --k 8 [options]\n"
        << "  kmeans_cli --rows 10000 --dims 16 --k 8 [options]\n\n"
        << "Data options:\n"
        << "  --input PATH                 Read CSV/whitespace matrix input\n"
        << "  --rows N --dims D            Generate synthetic blob data when --input is omitted\n"
        << "  --cluster-stddev VALUE       Synthetic cluster standard deviation (default: 0.6)\n"
        << "  --has-header                 Skip first input row\n"
        << "  --drop-first-column          Drop a leading id/index column\n"
        << "  --skip-first-count-line      Skip legacy first line containing row count\n\n"
        << "K-means options:\n"
        << "  --k N                        Number of clusters\n"
        << "  --iterations N               Maximum Lloyd iterations (default: 100)\n"
        << "  --tolerance VALUE            Convergence tolerance (default: 1e-4)\n"
        << "  --seed N                     Deterministic random seed (default: 42)\n"
        << "  --init random|kmeans++       Initialization strategy (default: kmeans++)\n"
        << "  --variant lloyd|mini-batch   Algorithm variant (default: lloyd)\n"
        << "  --mini-batch N               Mini-batch size for --variant mini-batch\n\n"
        << "Execution options:\n"
        << "  --backend cpu|cuda           Execution backend (default: cpu)\n"
        << "  --kernel naive|shared|warp|fused\n"
        << "  --device N                   CUDA device id (default: 0)\n"
        << "  --threads N                  CUDA threads per block (default: 256)\n\n"
        << "Output options:\n"
        << "  --output PATH                Write assignments CSV\n"
        << "  --centroids PATH             Write centroids CSV\n"
        << "  --json PATH                  Write fit summary JSON\n"
        << "  --help                       Show this message\n";
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

std::uint64_t parse_u64(const std::string& value, const std::string& name) {
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(value.c_str(), &end, 10);
    if (end == value.c_str() || *end != '\0') {
        throw std::invalid_argument("invalid unsigned integer for " + name + ": " + value);
    }
    return static_cast<std::uint64_t>(parsed);
}

float parse_float(const std::string& value, const std::string& name) {
    char* end = nullptr;
    const float parsed = std::strtof(value.c_str(), &end);
    if (end == value.c_str() || *end != '\0') {
        throw std::invalid_argument("invalid float for " + name + ": " + value);
    }
    return parsed;
}

CliConfig parse_args(int argc, char** argv) {
    CliConfig config;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_help();
            std::exit(0);
        } else if (arg == "--input") {
            config.input_path = require_value(i, argc, argv);
        } else if (arg == "--rows") {
            config.synthetic_rows = parse_int(require_value(i, argc, argv), arg);
        } else if (arg == "--dims") {
            config.synthetic_cols = parse_int(require_value(i, argc, argv), arg);
        } else if (arg == "--cluster-stddev") {
            config.synthetic_stddev = parse_float(require_value(i, argc, argv), arg);
        } else if (arg == "--has-header") {
            config.csv.has_header = true;
        } else if (arg == "--drop-first-column") {
            config.csv.drop_first_column = true;
        } else if (arg == "--skip-first-count-line") {
            config.csv.skip_first_count_line = true;
        } else if (arg == "--k") {
            config.options.clusters = parse_int(require_value(i, argc, argv), arg);
        } else if (arg == "--iterations") {
            config.options.max_iterations = parse_int(require_value(i, argc, argv), arg);
        } else if (arg == "--tolerance") {
            config.options.tolerance = parse_float(require_value(i, argc, argv), arg);
        } else if (arg == "--seed") {
            config.options.seed = parse_u64(require_value(i, argc, argv), arg);
        } else if (arg == "--init") {
            config.options.init = gpukmeans::parse_init_method(require_value(i, argc, argv));
        } else if (arg == "--variant") {
            config.options.variant = gpukmeans::parse_variant(require_value(i, argc, argv));
        } else if (arg == "--mini-batch") {
            config.options.mini_batch_size = parse_int(require_value(i, argc, argv), arg);
            config.options.variant = gpukmeans::KMeansVariant::MiniBatch;
        } else if (arg == "--backend") {
            config.options.backend = gpukmeans::parse_backend(require_value(i, argc, argv));
        } else if (arg == "--kernel") {
            config.options.kernel = gpukmeans::parse_kernel_strategy(require_value(i, argc, argv));
        } else if (arg == "--device") {
            config.options.device_id = parse_int(require_value(i, argc, argv), arg);
        } else if (arg == "--threads") {
            config.options.threads_per_block = parse_int(require_value(i, argc, argv), arg);
        } else if (arg == "--output") {
            config.assignments_path = require_value(i, argc, argv);
        } else if (arg == "--centroids") {
            config.centroids_path = require_value(i, argc, argv);
        } else if (arg == "--json") {
            config.json_path = require_value(i, argc, argv);
        } else {
            throw std::invalid_argument("unknown argument: " + arg);
        }
    }
    return config;
}

gpukmeans::Matrix load_data(const CliConfig& config) {
    if (!config.input_path.empty()) {
        return gpukmeans::read_matrix(config.input_path, config.csv);
    }
    if (config.synthetic_rows > 0 && config.synthetic_cols > 0) {
        return gpukmeans::generate_blobs(config.synthetic_rows,
                                         config.synthetic_cols,
                                         config.options.clusters,
                                         config.options.seed,
                                         config.synthetic_stddev);
    }
    throw std::invalid_argument("provide --input or synthetic --rows and --dims");
}

void print_summary(const gpukmeans::FitResult& result, const gpukmeans::Matrix& data) {
    std::cout << "backend=" << result.backend
              << " kernel=" << result.kernel
              << " init=" << result.init
              << " variant=" << result.variant << '\n';
    std::cout << "rows=" << data.rows
              << " dims=" << data.cols
              << " clusters=" << result.centroids.rows
              << " iterations=" << result.iterations
              << " converged=" << (result.converged ? "true" : "false")
              << " inertia=" << result.inertia << '\n';
    std::cout << "timing_ms total=" << result.timing.total_ms
              << " init=" << result.timing.init_ms
              << " cpu=" << result.timing.cpu_ms
              << " h2d=" << result.timing.h2d_ms
              << " kernel=" << result.timing.kernel_ms
              << " d2h=" << result.timing.d2h_ms << '\n';

    const std::size_t preview = std::min<std::size_t>(result.labels.size(), 20U);
    std::cout << "labels_preview";
    for (std::size_t i = 0; i < preview; ++i) {
        std::cout << (i == 0 ? "=" : ",") << result.labels[i];
    }
    std::cout << '\n';
}

} // namespace

int main(int argc, char** argv) {
    try {
        CliConfig config = parse_args(argc, argv);
        gpukmeans::Matrix data = load_data(config);
        gpukmeans::FitResult result = gpukmeans::fit(data, config.options.clusters, config.options);

        if (!config.assignments_path.empty()) {
            gpukmeans::write_assignments_csv(config.assignments_path, result.labels);
        }
        if (!config.centroids_path.empty()) {
            gpukmeans::write_centroids_csv(config.centroids_path, result.centroids);
        }
        if (!config.json_path.empty()) {
            gpukmeans::write_fit_result_json(config.json_path, result);
        }

        print_summary(result, data);
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "error: " << ex.what() << '\n';
        std::cerr << "run with --help for usage\n";
        return 1;
    }
}
