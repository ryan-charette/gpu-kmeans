#pragma once

#include "gpukmeans/kmeans.hpp"

#include <random>
#include <vector>

namespace gpukmeans::detail {

void validate_fit_input(const Matrix& data, const KMeansOptions& options);
Matrix initialize_centroids(const Matrix& data, const KMeansOptions& options, std::mt19937& rng);
std::vector<int> assign_labels(const Matrix& data, const Matrix& centroids, double* inertia_out);
Matrix recompute_centroids(const Matrix& data,
                           const std::vector<int>& labels,
                           const Matrix& previous_centroids);
Scalar max_centroid_shift(const Matrix& a, const Matrix& b);

FitResult fit_cpu(const Matrix& data, KMeansOptions options);
FitResult fit_cuda(const Matrix& data, KMeansOptions options);

} // namespace gpukmeans::detail
