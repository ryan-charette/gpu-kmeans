#include "kmeans_internal.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cctype>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace gpukmeans {
namespace {

using Clock = std::chrono::high_resolution_clock;

double elapsed_ms(Clock::time_point begin, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

std::string lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

int nearest_centroid(const Scalar* point, const Matrix& centroids, double* distance_out) {
    int best = 0;
    double best_distance = std::numeric_limits<double>::infinity();

    for (int c = 0; c < centroids.rows; ++c) {
        const Scalar* center = centroids.row(c);
        double distance = 0.0;
        for (int d = 0; d < centroids.cols; ++d) {
            const double diff = static_cast<double>(point[d]) - static_cast<double>(center[d]);
            distance += diff * diff;
        }
        if (distance < best_distance) {
            best_distance = distance;
            best = c;
        }
    }

    if (distance_out) {
        *distance_out = best_distance;
    }
    return best;
}

Matrix random_centroids(const Matrix& data, const KMeansOptions& options, std::mt19937& rng) {
    std::vector<int> indices(data.rows);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), rng);

    Matrix centroids(options.clusters, data.cols);
    for (int c = 0; c < options.clusters; ++c) {
        std::copy(data.row(indices[c]), data.row(indices[c]) + data.cols, centroids.row(c));
    }
    return centroids;
}

Matrix kmeans_plus_plus_centroids(const Matrix& data, const KMeansOptions& options, std::mt19937& rng) {
    Matrix centroids(options.clusters, data.cols);
    std::uniform_int_distribution<int> first_distribution(0, data.rows - 1);
    const int first = first_distribution(rng);
    std::copy(data.row(first), data.row(first) + data.cols, centroids.row(0));

    std::vector<double> min_distances(data.rows, std::numeric_limits<double>::infinity());
    for (int c = 1; c < options.clusters; ++c) {
        double total_distance = 0.0;
        for (int r = 0; r < data.rows; ++r) {
            const Scalar* point = data.row(r);
            const Scalar* last_center = centroids.row(c - 1);
            double distance = 0.0;
            for (int d = 0; d < data.cols; ++d) {
                const double diff = static_cast<double>(point[d]) - static_cast<double>(last_center[d]);
                distance += diff * diff;
            }
            min_distances[r] = std::min(min_distances[r], distance);
            total_distance += min_distances[r];
        }

        int next_index = data.rows - 1;
        if (total_distance <= std::numeric_limits<double>::epsilon()) {
            std::uniform_int_distribution<int> fallback_distribution(0, data.rows - 1);
            next_index = fallback_distribution(rng);
        } else {
            std::uniform_real_distribution<double> weighted_distribution(0.0, total_distance);
            const double target = weighted_distribution(rng);
            double cumulative = 0.0;
            for (int r = 0; r < data.rows; ++r) {
                cumulative += min_distances[r];
                if (cumulative >= target) {
                    next_index = r;
                    break;
                }
            }
        }

        std::copy(data.row(next_index), data.row(next_index) + data.cols, centroids.row(c));
    }

    return centroids;
}

FitResult fit_lloyd_cpu(const Matrix& data, KMeansOptions options) {
    const auto total_begin = Clock::now();
    std::mt19937 rng(static_cast<std::mt19937::result_type>(options.seed));

    const auto init_begin = Clock::now();
    Matrix centroids = detail::initialize_centroids(data, options, rng);
    const auto init_end = Clock::now();

    std::vector<int> labels(data.rows, 0);
    double current_inertia = 0.0;
    bool converged = false;
    int iterations = 0;
    std::vector<IterationRecord> history;

    const auto cpu_begin = Clock::now();
    for (int iter = 0; iter < options.max_iterations; ++iter) {
        const auto iteration_begin = Clock::now();
        labels = detail::assign_labels(data, centroids, nullptr);
        Matrix next_centroids = detail::recompute_centroids(data, labels, centroids);
        const Scalar shift = detail::max_centroid_shift(centroids, next_centroids);
        centroids = std::move(next_centroids);
        labels = detail::assign_labels(data, centroids, &current_inertia);
        iterations = iter + 1;
        converged = shift <= options.tolerance;

        if (options.collect_history) {
            history.push_back({iterations,
                               current_inertia,
                               shift,
                               elapsed_ms(iteration_begin, Clock::now())});
        }

        if (converged) {
            break;
        }
    }
    const auto cpu_end = Clock::now();

    FitResult result;
    result.centroids = std::move(centroids);
    result.labels = std::move(labels);
    result.inertia = current_inertia;
    result.iterations = iterations;
    result.converged = converged;
    result.history = std::move(history);
    result.backend = to_string(Backend::Cpu);
    result.kernel = "cpu";
    result.init = to_string(options.init);
    result.variant = to_string(options.variant);
    result.timing.init_ms = elapsed_ms(init_begin, init_end);
    result.timing.cpu_ms = elapsed_ms(cpu_begin, cpu_end);
    result.timing.total_ms = elapsed_ms(total_begin, Clock::now());
    return result;
}

FitResult fit_minibatch_cpu(const Matrix& data, KMeansOptions options) {
    const auto total_begin = Clock::now();
    std::mt19937 rng(static_cast<std::mt19937::result_type>(options.seed));
    const int batch_size = std::min(options.mini_batch_size, data.rows);

    const auto init_begin = Clock::now();
    Matrix centroids = detail::initialize_centroids(data, options, rng);
    const auto init_end = Clock::now();

    std::uniform_int_distribution<int> row_distribution(0, data.rows - 1);
    std::vector<std::int64_t> update_counts(options.clusters, 0);
    std::vector<int> labels(data.rows, 0);
    std::vector<IterationRecord> history;
    double current_inertia = 0.0;
    bool converged = false;
    int iterations = 0;

    const auto cpu_begin = Clock::now();
    for (int iter = 0; iter < options.max_iterations; ++iter) {
        const auto iteration_begin = Clock::now();
        Matrix previous_centroids = centroids;

        for (int b = 0; b < batch_size; ++b) {
            const int row_index = row_distribution(rng);
            const Scalar* point = data.row(row_index);
            const int label = nearest_centroid(point, centroids, nullptr);
            update_counts[label] += 1;
            const Scalar eta = static_cast<Scalar>(1.0 / static_cast<double>(update_counts[label]));
            Scalar* center = centroids.row(label);
            for (int d = 0; d < data.cols; ++d) {
                center[d] = static_cast<Scalar>((1.0f - eta) * center[d] + eta * point[d]);
            }
        }

        const Scalar shift = detail::max_centroid_shift(previous_centroids, centroids);
        labels = detail::assign_labels(data, centroids, &current_inertia);
        iterations = iter + 1;
        converged = shift <= options.tolerance;

        if (options.collect_history) {
            history.push_back({iterations,
                               current_inertia,
                               shift,
                               elapsed_ms(iteration_begin, Clock::now())});
        }

        if (converged) {
            break;
        }
    }
    const auto cpu_end = Clock::now();

    FitResult result;
    result.centroids = std::move(centroids);
    result.labels = std::move(labels);
    result.inertia = current_inertia;
    result.iterations = iterations;
    result.converged = converged;
    result.history = std::move(history);
    result.backend = to_string(Backend::Cpu);
    result.kernel = "cpu";
    result.init = to_string(options.init);
    result.variant = to_string(options.variant);
    result.timing.init_ms = elapsed_ms(init_begin, init_end);
    result.timing.cpu_ms = elapsed_ms(cpu_begin, cpu_end);
    result.timing.total_ms = elapsed_ms(total_begin, Clock::now());
    return result;
}

} // namespace

Matrix::Matrix(int row_count, int col_count) {
    if (row_count < 0 || col_count < 0) {
        throw std::invalid_argument("matrix dimensions must be non-negative");
    }
    rows = row_count;
    cols = col_count;
    values.assign(static_cast<std::size_t>(row_count) * static_cast<std::size_t>(col_count), 0.0f);
}

Matrix::Matrix(std::vector<Scalar> data, int row_count, int col_count)
    : values(std::move(data)), rows(row_count), cols(col_count) {
    if (row_count < 0 || col_count < 0) {
        throw std::invalid_argument("matrix dimensions must be non-negative");
    }
    if (values.size() != static_cast<std::size_t>(row_count) * static_cast<std::size_t>(col_count)) {
        throw std::invalid_argument("matrix value count does not match rows * cols");
    }
}

bool Matrix::empty() const {
    return rows == 0 || cols == 0 || values.empty();
}

std::size_t Matrix::size() const {
    return values.size();
}

const Scalar* Matrix::row(int index) const {
    return values.data() + static_cast<std::size_t>(index) * static_cast<std::size_t>(cols);
}

Scalar* Matrix::row(int index) {
    return values.data() + static_cast<std::size_t>(index) * static_cast<std::size_t>(cols);
}

KMeans::KMeans(KMeansOptions options) : options_(options) {}

FitResult KMeans::fit(const Matrix& data) {
    result_ = gpukmeans::fit(data, options_.clusters, options_);
    fitted_ = true;
    return result_;
}

std::vector<int> KMeans::predict(const Matrix& data) const {
    if (!fitted_) {
        throw std::logic_error("predict() called before fit()");
    }
    return gpukmeans::predict(data, result_.centroids);
}

double KMeans::inertia(const Matrix& data) const {
    if (!fitted_) {
        throw std::logic_error("inertia() called before fit()");
    }
    return gpukmeans::inertia(data, result_.centroids);
}

const FitResult& KMeans::result() const {
    if (!fitted_) {
        throw std::logic_error("result() called before fit()");
    }
    return result_;
}

const Matrix& KMeans::centroids() const {
    if (!fitted_) {
        throw std::logic_error("centroids() called before fit()");
    }
    return result_.centroids;
}

const KMeansOptions& KMeans::options() const {
    return options_;
}

FitResult fit(const Matrix& data, int k, KMeansOptions options) {
    options.clusters = k;
    detail::validate_fit_input(data, options);

    if (options.backend == Backend::Cuda) {
        return detail::fit_cuda(data, options);
    }
    return detail::fit_cpu(data, options);
}

std::vector<int> predict(const Matrix& data, const Matrix& centroids) {
    if (centroids.rows <= 0 || centroids.cols <= 0 || centroids.empty()) {
        throw std::invalid_argument("centroids must be non-empty");
    }
    if (data.cols != centroids.cols) {
        throw std::invalid_argument("data and centroid dimensionality differ");
    }
    return detail::assign_labels(data, centroids, nullptr);
}

double inertia(const Matrix& data, const Matrix& centroids) {
    if (centroids.rows <= 0 || centroids.cols <= 0 || centroids.empty()) {
        throw std::invalid_argument("centroids must be non-empty");
    }
    if (data.cols != centroids.cols) {
        throw std::invalid_argument("data and centroid dimensionality differ");
    }
    double value = 0.0;
    detail::assign_labels(data, centroids, &value);
    return value;
}

std::string to_string(Backend backend) {
    return backend == Backend::Cuda ? "cuda" : "cpu";
}

std::string to_string(KernelStrategy strategy) {
    switch (strategy) {
    case KernelStrategy::Naive:
        return "naive";
    case KernelStrategy::Shared:
        return "shared";
    case KernelStrategy::Warp:
        return "warp";
    case KernelStrategy::Fused:
        return "fused";
    }
    return "unknown";
}

std::string to_string(InitMethod init) {
    return init == InitMethod::Random ? "random" : "kmeans++";
}

std::string to_string(KMeansVariant variant) {
    return variant == KMeansVariant::MiniBatch ? "mini-batch" : "lloyd";
}

Backend parse_backend(const std::string& value) {
    const std::string normalized = lower(value);
    if (normalized == "cpu") {
        return Backend::Cpu;
    }
    if (normalized == "cuda" || normalized == "gpu") {
        return Backend::Cuda;
    }
    throw std::invalid_argument("unknown backend: " + value);
}

KernelStrategy parse_kernel_strategy(const std::string& value) {
    const std::string normalized = lower(value);
    if (normalized == "naive") {
        return KernelStrategy::Naive;
    }
    if (normalized == "shared") {
        return KernelStrategy::Shared;
    }
    if (normalized == "warp") {
        return KernelStrategy::Warp;
    }
    if (normalized == "fused") {
        return KernelStrategy::Fused;
    }
    throw std::invalid_argument("unknown kernel strategy: " + value);
}

InitMethod parse_init_method(const std::string& value) {
    const std::string normalized = lower(value);
    if (normalized == "random") {
        return InitMethod::Random;
    }
    if (normalized == "kmeans++" || normalized == "kmeans-plus-plus" || normalized == "plusplus") {
        return InitMethod::KMeansPlusPlus;
    }
    throw std::invalid_argument("unknown initialization method: " + value);
}

KMeansVariant parse_variant(const std::string& value) {
    const std::string normalized = lower(value);
    if (normalized == "lloyd" || normalized == "full") {
        return KMeansVariant::Lloyd;
    }
    if (normalized == "mini-batch" || normalized == "minibatch") {
        return KMeansVariant::MiniBatch;
    }
    throw std::invalid_argument("unknown K-means variant: " + value);
}

namespace detail {

void validate_fit_input(const Matrix& data, const KMeansOptions& options) {
    if (data.rows <= 0 || data.cols <= 0 || data.empty()) {
        throw std::invalid_argument("data matrix must be non-empty");
    }
    if (data.values.size() != static_cast<std::size_t>(data.rows) * static_cast<std::size_t>(data.cols)) {
        throw std::invalid_argument("data matrix shape does not match its value count");
    }
    if (options.clusters <= 0) {
        throw std::invalid_argument("number of clusters must be positive");
    }
    if (options.clusters > data.rows) {
        throw std::invalid_argument("number of clusters cannot exceed number of rows");
    }
    if (options.max_iterations <= 0) {
        throw std::invalid_argument("max_iterations must be positive");
    }
    if (options.tolerance < 0.0f) {
        throw std::invalid_argument("tolerance must be non-negative");
    }
    if (options.threads_per_block <= 0) {
        throw std::invalid_argument("threads_per_block must be positive");
    }
    if (options.variant == KMeansVariant::MiniBatch && options.mini_batch_size <= 0) {
        throw std::invalid_argument("mini_batch_size must be positive for mini-batch K-means");
    }
}

Matrix initialize_centroids(const Matrix& data, const KMeansOptions& options, std::mt19937& rng) {
    if (options.init == InitMethod::Random) {
        return random_centroids(data, options, rng);
    }
    return kmeans_plus_plus_centroids(data, options, rng);
}

std::vector<int> assign_labels(const Matrix& data, const Matrix& centroids, double* inertia_out) {
    std::vector<int> labels(data.rows);
    double total_inertia = 0.0;
    for (int r = 0; r < data.rows; ++r) {
        double distance = 0.0;
        labels[r] = nearest_centroid(data.row(r), centroids, &distance);
        total_inertia += distance;
    }
    if (inertia_out) {
        *inertia_out = total_inertia;
    }
    return labels;
}

Matrix recompute_centroids(const Matrix& data,
                           const std::vector<int>& labels,
                           const Matrix& previous_centroids) {
    Matrix centroids(previous_centroids.rows, previous_centroids.cols);
    std::vector<int> counts(previous_centroids.rows, 0);

    for (int r = 0; r < data.rows; ++r) {
        const int label = labels[r];
        counts[label] += 1;
        Scalar* center = centroids.row(label);
        const Scalar* point = data.row(r);
        for (int d = 0; d < data.cols; ++d) {
            center[d] += point[d];
        }
    }

    for (int c = 0; c < centroids.rows; ++c) {
        Scalar* center = centroids.row(c);
        if (counts[c] == 0) {
            std::copy(previous_centroids.row(c), previous_centroids.row(c) + centroids.cols, center);
            continue;
        }
        for (int d = 0; d < centroids.cols; ++d) {
            center[d] = static_cast<Scalar>(center[d] / static_cast<Scalar>(counts[c]));
        }
    }

    return centroids;
}

Scalar max_centroid_shift(const Matrix& a, const Matrix& b) {
    if (a.rows != b.rows || a.cols != b.cols) {
        throw std::invalid_argument("centroid matrices have different shapes");
    }
    double max_shift = 0.0;
    for (int c = 0; c < a.rows; ++c) {
        double sum = 0.0;
        for (int d = 0; d < a.cols; ++d) {
            const double diff = static_cast<double>(a.row(c)[d]) - static_cast<double>(b.row(c)[d]);
            sum += diff * diff;
        }
        max_shift = std::max(max_shift, std::sqrt(sum));
    }
    return static_cast<Scalar>(max_shift);
}

FitResult fit_cpu(const Matrix& data, KMeansOptions options) {
    if (options.variant == KMeansVariant::MiniBatch) {
        return fit_minibatch_cpu(data, options);
    }
    return fit_lloyd_cpu(data, options);
}

} // namespace detail
} // namespace gpukmeans
