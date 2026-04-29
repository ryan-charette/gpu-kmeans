#include "gpukmeans/io.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace gpukmeans {
namespace {

std::string trim(const std::string& value) {
    const auto first = std::find_if_not(value.begin(), value.end(),
                                        [](unsigned char c) { return std::isspace(c) != 0; });
    const auto last = std::find_if_not(value.rbegin(), value.rend(),
                                       [](unsigned char c) { return std::isspace(c) != 0; }).base();
    if (first >= last) {
        return {};
    }
    return std::string(first, last);
}

std::vector<Scalar> parse_numeric_row(std::string line) {
    for (char& c : line) {
        if (c == ',' || c == ';' || c == '\t') {
            c = ' ';
        }
    }

    std::stringstream stream(line);
    std::vector<Scalar> values;
    double parsed = 0.0;
    while (stream >> parsed) {
        values.push_back(static_cast<Scalar>(parsed));
    }

    if (values.empty() && !trim(line).empty()) {
        throw std::invalid_argument("non-numeric row encountered while reading matrix");
    }
    return values;
}

void require_output_stream(const std::ofstream& out, const std::string& path) {
    if (!out) {
        throw std::runtime_error("failed to open output file: " + path);
    }
}

void write_json_string(std::ostream& out, const std::string& value) {
    out << '"';
    for (char c : value) {
        if (c == '"' || c == '\\') {
            out << '\\' << c;
        } else if (c == '\n') {
            out << "\\n";
        } else {
            out << c;
        }
    }
    out << '"';
}

} // namespace

Matrix read_matrix(const std::string& path, CsvReadOptions options) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("failed to open input file: " + path);
    }

    std::vector<Scalar> values;
    int rows = 0;
    int cols = -1;
    std::string line;
    bool first_data_line = true;

    if (options.has_header) {
        std::getline(in, line);
    }

    while (std::getline(in, line)) {
        line = trim(line);
        if (line.empty() || line.front() == '#') {
            continue;
        }

        std::vector<Scalar> row = parse_numeric_row(line);
        if (first_data_line && options.skip_first_count_line && row.size() == 1U) {
            first_data_line = false;
            continue;
        }
        first_data_line = false;

        if (options.drop_first_column) {
            if (row.size() <= 1U) {
                throw std::invalid_argument("cannot drop first column from a row with fewer than two columns");
            }
            row.erase(row.begin());
        }

        if (cols < 0) {
            cols = static_cast<int>(row.size());
        }
        if (static_cast<int>(row.size()) != cols) {
            throw std::invalid_argument("input rows have inconsistent column counts");
        }

        values.insert(values.end(), row.begin(), row.end());
        ++rows;
    }

    if (rows == 0 || cols <= 0) {
        throw std::invalid_argument("input matrix is empty");
    }
    return Matrix(std::move(values), rows, cols);
}

void write_assignments_csv(const std::string& path, const std::vector<int>& labels) {
    std::ofstream out(path);
    require_output_stream(out, path);
    out << "row,label\n";
    for (std::size_t i = 0; i < labels.size(); ++i) {
        out << i << ',' << labels[i] << '\n';
    }
}

void write_centroids_csv(const std::string& path, const Matrix& centroids) {
    std::ofstream out(path);
    require_output_stream(out, path);
    out << "cluster";
    for (int d = 0; d < centroids.cols; ++d) {
        out << ",x" << d;
    }
    out << '\n';

    out << std::setprecision(9);
    for (int c = 0; c < centroids.rows; ++c) {
        out << c;
        for (int d = 0; d < centroids.cols; ++d) {
            out << ',' << centroids.row(c)[d];
        }
        out << '\n';
    }
}

void write_fit_result_json(const std::string& path, const FitResult& result) {
    std::ofstream out(path);
    require_output_stream(out, path);
    out << std::setprecision(9);
    out << "{\n";
    out << "  \"backend\": ";
    write_json_string(out, result.backend);
    out << ",\n  \"kernel\": ";
    write_json_string(out, result.kernel);
    out << ",\n  \"init\": ";
    write_json_string(out, result.init);
    out << ",\n  \"variant\": ";
    write_json_string(out, result.variant);
    out << ",\n  \"iterations\": " << result.iterations;
    out << ",\n  \"converged\": " << (result.converged ? "true" : "false");
    out << ",\n  \"inertia\": " << result.inertia;
    out << ",\n  \"timing_ms\": {";
    out << "\"total\": " << result.timing.total_ms;
    out << ", \"init\": " << result.timing.init_ms;
    out << ", \"cpu\": " << result.timing.cpu_ms;
    out << ", \"h2d\": " << result.timing.h2d_ms;
    out << ", \"kernel\": " << result.timing.kernel_ms;
    out << ", \"d2h\": " << result.timing.d2h_ms << "},\n";
    out << "  \"history\": [";
    for (std::size_t i = 0; i < result.history.size(); ++i) {
        const IterationRecord& record = result.history[i];
        if (i != 0U) {
            out << ',';
        }
        out << "\n    {\"iteration\": " << record.iteration
            << ", \"inertia\": " << record.inertia
            << ", \"centroid_shift\": " << record.centroid_shift
            << ", \"iteration_ms\": " << record.iteration_ms << '}';
    }
    if (!result.history.empty()) {
        out << '\n';
    }
    out << "  ]\n";
    out << "}\n";
}

} // namespace gpukmeans
