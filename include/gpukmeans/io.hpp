#pragma once

#include "gpukmeans/kmeans.hpp"

#include <string>

namespace gpukmeans {

struct CsvReadOptions {
    bool has_header = false;
    bool drop_first_column = false;
    bool skip_first_count_line = false;
};

Matrix read_matrix(const std::string& path, CsvReadOptions options = {});
void write_assignments_csv(const std::string& path, const std::vector<int>& labels);
void write_centroids_csv(const std::string& path, const Matrix& centroids);
void write_fit_result_json(const std::string& path, const FitResult& result);

} // namespace gpukmeans
