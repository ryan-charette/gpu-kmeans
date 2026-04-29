#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace gpukmeans {

using Scalar = float;

struct Matrix {
    std::vector<Scalar> values;
    int rows = 0;
    int cols = 0;

    Matrix() = default;
    Matrix(int row_count, int col_count);
    Matrix(std::vector<Scalar> data, int row_count, int col_count);

    bool empty() const;
    std::size_t size() const;
    const Scalar* row(int index) const;
    Scalar* row(int index);
};

enum class Backend {
    Cpu,
    Cuda
};

enum class KernelStrategy {
    Naive,
    Shared,
    Warp,
    Fused
};

enum class InitMethod {
    Random,
    KMeansPlusPlus
};

enum class KMeansVariant {
    Lloyd,
    MiniBatch
};

struct KMeansOptions {
    int clusters = 2;
    int max_iterations = 100;
    Scalar tolerance = 1.0e-4f;
    std::uint64_t seed = 42;
    InitMethod init = InitMethod::KMeansPlusPlus;
    KMeansVariant variant = KMeansVariant::Lloyd;
    Backend backend = Backend::Cpu;
    KernelStrategy kernel = KernelStrategy::Naive;
    int mini_batch_size = 0;
    int device_id = 0;
    int threads_per_block = 256;
    bool collect_history = true;
};

struct TimingBreakdown {
    double total_ms = 0.0;
    double init_ms = 0.0;
    double cpu_ms = 0.0;
    double h2d_ms = 0.0;
    double kernel_ms = 0.0;
    double d2h_ms = 0.0;
};

struct IterationRecord {
    int iteration = 0;
    double inertia = 0.0;
    Scalar centroid_shift = 0.0f;
    double iteration_ms = 0.0;
};

struct FitResult {
    Matrix centroids;
    std::vector<int> labels;
    double inertia = 0.0;
    int iterations = 0;
    bool converged = false;
    TimingBreakdown timing;
    std::vector<IterationRecord> history;
    std::string backend;
    std::string kernel;
    std::string init;
    std::string variant;
};

class KMeans {
public:
    explicit KMeans(KMeansOptions options = {});

    FitResult fit(const Matrix& data);
    std::vector<int> predict(const Matrix& data) const;
    double inertia(const Matrix& data) const;

    const FitResult& result() const;
    const Matrix& centroids() const;
    const KMeansOptions& options() const;

private:
    KMeansOptions options_;
    FitResult result_;
    bool fitted_ = false;
};

FitResult fit(const Matrix& data, int k, KMeansOptions options = {});
std::vector<int> predict(const Matrix& data, const Matrix& centroids);
double inertia(const Matrix& data, const Matrix& centroids);

std::string to_string(Backend backend);
std::string to_string(KernelStrategy strategy);
std::string to_string(InitMethod init);
std::string to_string(KMeansVariant variant);

Backend parse_backend(const std::string& value);
KernelStrategy parse_kernel_strategy(const std::string& value);
InitMethod parse_init_method(const std::string& value);
KMeansVariant parse_variant(const std::string& value);

bool cuda_backend_compiled();
int cuda_device_count();

} // namespace gpukmeans
