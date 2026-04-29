#include "gpukmeans/kmeans.hpp"
#include "gpukmeans/synthetic.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

gpukmeans::Matrix tiny_two_cluster_data() {
    return gpukmeans::Matrix({
                                 0.0f, 0.0f,
                                 0.1f, 0.0f,
                                 -0.1f, 0.1f,
                                 10.0f, 10.0f,
                                 10.2f, 9.9f,
                                 9.8f, 10.1f,
                             },
                             6,
                             2);
}

template <typename Fn>
void expect_throw(Fn&& fn) {
    bool threw = false;
    try {
        fn();
    } catch (const std::exception&) {
        threw = true;
    }
    assert(threw);
}

void test_invalid_inputs() {
    const gpukmeans::Matrix data = tiny_two_cluster_data();
    gpukmeans::KMeansOptions options;

    expect_throw([&] { gpukmeans::fit(data, 0, options); });
    expect_throw([&] { gpukmeans::fit(data, 7, options); });

    gpukmeans::KMeans model;
    expect_throw([&] { (void)model.predict(data); });
}

void test_cpu_converges_on_simple_data() {
    gpukmeans::KMeansOptions options;
    options.clusters = 2;
    options.seed = 7;
    options.max_iterations = 50;
    options.tolerance = 1.0e-5f;
    options.init = gpukmeans::InitMethod::KMeansPlusPlus;

    const gpukmeans::Matrix data = tiny_two_cluster_data();
    const gpukmeans::FitResult result = gpukmeans::fit(data, 2, options);

    assert(result.labels.size() == static_cast<std::size_t>(data.rows));
    assert(result.centroids.rows == 2);
    assert(result.centroids.cols == 2);
    assert(result.iterations > 0);
    assert(result.converged);
    assert(result.inertia < 0.2);

    assert(result.labels[0] == result.labels[1]);
    assert(result.labels[3] == result.labels[4]);
    assert(result.labels[0] != result.labels[3]);
}

void test_deterministic_seed() {
    const gpukmeans::Matrix data = gpukmeans::generate_blobs(200, 4, 3, 123, 0.3f);
    gpukmeans::KMeansOptions options;
    options.clusters = 3;
    options.seed = 99;
    options.max_iterations = 20;
    options.collect_history = false;

    const gpukmeans::FitResult a = gpukmeans::fit(data, 3, options);
    const gpukmeans::FitResult b = gpukmeans::fit(data, 3, options);

    assert(a.labels == b.labels);
    assert(a.centroids.values == b.centroids.values);
    assert(std::fabs(a.inertia - b.inertia) < 1.0e-6);
}

void test_predict_and_inertia_api() {
    const gpukmeans::Matrix data = tiny_two_cluster_data();
    gpukmeans::KMeansOptions options;
    options.clusters = 2;
    options.seed = 4;

    gpukmeans::KMeans model(options);
    const gpukmeans::FitResult result = model.fit(data);
    const std::vector<int> predicted = model.predict(data);
    assert(predicted == result.labels);
    assert(std::fabs(model.inertia(data) - result.inertia) < 1.0e-6);
}

void test_minibatch_smoke() {
    const gpukmeans::Matrix data = gpukmeans::generate_blobs(300, 3, 4, 777, 0.4f);
    gpukmeans::KMeansOptions options;
    options.clusters = 4;
    options.seed = 5;
    options.variant = gpukmeans::KMeansVariant::MiniBatch;
    options.mini_batch_size = 32;
    options.max_iterations = 25;

    const gpukmeans::FitResult result = gpukmeans::fit(data, 4, options);
    assert(result.labels.size() == static_cast<std::size_t>(data.rows));
    assert(result.centroids.rows == 4);
    assert(result.inertia >= 0.0);
}

void test_cuda_parity_when_available() {
#if GPUKMEANS_HAVE_CUDA
    if (gpukmeans::cuda_device_count() <= 0) {
        return;
    }
    const gpukmeans::Matrix data = gpukmeans::generate_blobs(512, 4, 4, 11, 0.5f);
    gpukmeans::KMeansOptions cpu;
    cpu.clusters = 4;
    cpu.seed = 101;
    cpu.max_iterations = 10;
    cpu.collect_history = false;

    gpukmeans::KMeansOptions gpu = cpu;
    gpu.backend = gpukmeans::Backend::Cuda;
    gpu.kernel = gpukmeans::KernelStrategy::Naive;

    const gpukmeans::FitResult cpu_result = gpukmeans::fit(data, 4, cpu);
    const gpukmeans::FitResult gpu_result = gpukmeans::fit(data, 4, gpu);
    assert(gpu_result.labels.size() == cpu_result.labels.size());
    assert(std::fabs(gpu_result.inertia - cpu_result.inertia) / std::max(1.0, cpu_result.inertia) < 0.05);
#endif
}

} // namespace

int main() {
    test_invalid_inputs();
    test_cpu_converges_on_simple_data();
    test_deterministic_seed();
    test_predict_and_inertia_api();
    test_minibatch_smoke();
    test_cuda_parity_when_available();
    std::cout << "gpukmeans tests passed\n";
    return 0;
}
