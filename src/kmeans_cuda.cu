#include "kmeans_internal.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cfloat>
#include <cmath>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace gpukmeans {
namespace {

using Clock = std::chrono::high_resolution_clock;

void cuda_check(cudaError_t status, const char* expression, const char* file, int line) {
    if (status != cudaSuccess) {
        std::ostringstream message;
        message << "CUDA call failed at " << file << ':' << line << " for " << expression
                << ": " << cudaGetErrorString(status);
        throw std::runtime_error(message.str());
    }
}

#define GPUKMEANS_CUDA_CHECK(expr) cuda_check((expr), #expr, __FILE__, __LINE__)

double elapsed_ms(Clock::time_point begin, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

float event_elapsed_ms(cudaEvent_t begin, cudaEvent_t end) {
    float ms = 0.0f;
    GPUKMEANS_CUDA_CHECK(cudaEventElapsedTime(&ms, begin, end));
    return ms;
}

class ScopedEvent {
public:
    ScopedEvent() {
        GPUKMEANS_CUDA_CHECK(cudaEventCreate(&event_));
    }

    ~ScopedEvent() {
        if (event_ != nullptr) {
            cudaEventDestroy(event_);
        }
    }

    cudaEvent_t get() const {
        return event_;
    }

private:
    cudaEvent_t event_ = nullptr;
};

template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;

    explicit DeviceBuffer(std::size_t count) {
        reset(count);
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept : ptr_(other.ptr_), count_(other.count_) {
        other.ptr_ = nullptr;
        other.count_ = 0;
    }

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            release();
            ptr_ = other.ptr_;
            count_ = other.count_;
            other.ptr_ = nullptr;
            other.count_ = 0;
        }
        return *this;
    }

    ~DeviceBuffer() {
        release();
    }

    void reset(std::size_t count) {
        release();
        count_ = count;
        if (count_ > 0U) {
            GPUKMEANS_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr_), count_ * sizeof(T)));
        }
    }

    T* get() {
        return ptr_;
    }

    const T* get() const {
        return ptr_;
    }

    std::size_t bytes() const {
        return count_ * sizeof(T);
    }

private:
    void release() noexcept {
        if (ptr_ != nullptr) {
            cudaFree(ptr_);
            ptr_ = nullptr;
            count_ = 0;
        }
    }

    T* ptr_ = nullptr;
    std::size_t count_ = 0;
};

__global__ void assign_naive_kernel(const Scalar* data,
                                    const Scalar* centroids,
                                    int* labels,
                                    Scalar* distances,
                                    int rows,
                                    int cols,
                                    int clusters) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }

    const Scalar* point = data + row * cols;
    int best = 0;
    Scalar best_distance = FLT_MAX;
    for (int c = 0; c < clusters; ++c) {
        const Scalar* center = centroids + c * cols;
        Scalar distance = 0.0f;
        for (int d = 0; d < cols; ++d) {
            const Scalar diff = point[d] - center[d];
            distance += diff * diff;
        }
        if (distance < best_distance) {
            best_distance = distance;
            best = c;
        }
    }

    labels[row] = best;
    distances[row] = best_distance;
}

__global__ void assign_shared_kernel(const Scalar* data,
                                     const Scalar* centroids,
                                     int* labels,
                                     Scalar* distances,
                                     int rows,
                                     int cols,
                                     int clusters) {
    extern __shared__ Scalar shared_centroids[];

    const int centroid_values = clusters * cols;
    for (int i = threadIdx.x; i < centroid_values; i += blockDim.x) {
        shared_centroids[i] = centroids[i];
    }
    __syncthreads();

    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }

    const Scalar* point = data + row * cols;
    int best = 0;
    Scalar best_distance = FLT_MAX;
    for (int c = 0; c < clusters; ++c) {
        const Scalar* center = shared_centroids + c * cols;
        Scalar distance = 0.0f;
        for (int d = 0; d < cols; ++d) {
            const Scalar diff = point[d] - center[d];
            distance += diff * diff;
        }
        if (distance < best_distance) {
            best_distance = distance;
            best = c;
        }
    }

    labels[row] = best;
    distances[row] = best_distance;
}

__global__ void accumulate_kernel(const Scalar* data,
                                  const int* labels,
                                  Scalar* sums,
                                  int* counts,
                                  int rows,
                                  int cols) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }

    const int label = labels[row];
    const Scalar* point = data + row * cols;
    Scalar* sum = sums + label * cols;
    for (int d = 0; d < cols; ++d) {
        atomicAdd(sum + d, point[d]);
    }
    atomicAdd(counts + label, 1);
}

__global__ void finalize_centroids_kernel(Scalar* centroids,
                                          const Scalar* old_centroids,
                                          const Scalar* sums,
                                          const int* counts,
                                          int cols,
                                          int clusters) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = clusters * cols;
    if (index >= total) {
        return;
    }

    const int cluster = index / cols;
    const int count = counts[cluster];
    centroids[index] = count > 0 ? sums[index] / static_cast<Scalar>(count) : old_centroids[index];
}

void launch_assignment(KernelStrategy strategy,
                       const DeviceBuffer<Scalar>& data,
                       const DeviceBuffer<Scalar>& centroids,
                       DeviceBuffer<int>& labels,
                       DeviceBuffer<Scalar>& distances,
                       int rows,
                       int cols,
                       int clusters,
                       int blocks,
                       int threads,
                       std::size_t shared_bytes) {
    if (strategy == KernelStrategy::Naive) {
        assign_naive_kernel<<<blocks, threads>>>(data.get(), centroids.get(), labels.get(), distances.get(),
                                                 rows, cols, clusters);
    } else if (strategy == KernelStrategy::Shared) {
        assign_shared_kernel<<<blocks, threads, shared_bytes>>>(data.get(), centroids.get(), labels.get(),
                                                                distances.get(), rows, cols, clusters);
    } else {
        throw std::runtime_error("warp and fused CUDA strategies are scaffolded but not implemented yet");
    }
    GPUKMEANS_CUDA_CHECK(cudaGetLastError());
}

double timed_copy_h2d(const void* host, void* device, std::size_t bytes) {
    ScopedEvent begin;
    ScopedEvent end;
    GPUKMEANS_CUDA_CHECK(cudaEventRecord(begin.get()));
    GPUKMEANS_CUDA_CHECK(cudaMemcpy(device, host, bytes, cudaMemcpyHostToDevice));
    GPUKMEANS_CUDA_CHECK(cudaEventRecord(end.get()));
    GPUKMEANS_CUDA_CHECK(cudaEventSynchronize(end.get()));
    return event_elapsed_ms(begin.get(), end.get());
}

double timed_copy_d2h(void* host, const void* device, std::size_t bytes) {
    ScopedEvent begin;
    ScopedEvent end;
    GPUKMEANS_CUDA_CHECK(cudaEventRecord(begin.get()));
    GPUKMEANS_CUDA_CHECK(cudaMemcpy(host, device, bytes, cudaMemcpyDeviceToHost));
    GPUKMEANS_CUDA_CHECK(cudaEventRecord(end.get()));
    GPUKMEANS_CUDA_CHECK(cudaEventSynchronize(end.get()));
    return event_elapsed_ms(begin.get(), end.get());
}

} // namespace

bool cuda_backend_compiled() {
    return true;
}

int cuda_device_count() {
    int count = 0;
    const cudaError_t status = cudaGetDeviceCount(&count);
    if (status != cudaSuccess) {
        return 0;
    }
    return count;
}

namespace detail {

FitResult fit_cuda(const Matrix& data, KMeansOptions options) {
    if (options.variant == KMeansVariant::MiniBatch) {
        throw std::runtime_error("CUDA mini-batch K-means is not implemented; use backend=cpu for mini-batch");
    }
    if (options.kernel == KernelStrategy::Warp || options.kernel == KernelStrategy::Fused) {
        throw std::runtime_error(
            "selected CUDA strategy is scaffolded for future work; implemented CUDA strategies are naive and shared");
    }

    const int device_count = cuda_device_count();
    if (device_count <= 0) {
        throw std::runtime_error("no CUDA devices are available");
    }
    if (options.device_id < 0 || options.device_id >= device_count) {
        throw std::invalid_argument("requested CUDA device id is out of range");
    }

    GPUKMEANS_CUDA_CHECK(cudaSetDevice(options.device_id));
    cudaDeviceProp props{};
    GPUKMEANS_CUDA_CHECK(cudaGetDeviceProperties(&props, options.device_id));
    if (options.threads_per_block > props.maxThreadsPerBlock) {
        throw std::invalid_argument("threads_per_block exceeds device maximum");
    }

    const std::size_t centroid_values =
        static_cast<std::size_t>(options.clusters) * static_cast<std::size_t>(data.cols);
    const std::size_t shared_bytes = centroid_values * sizeof(Scalar);
    if (options.kernel == KernelStrategy::Shared &&
        shared_bytes > static_cast<std::size_t>(props.sharedMemPerBlock)) {
        throw std::runtime_error("centroid cache does not fit in per-block shared memory for this device");
    }

    const auto total_begin = Clock::now();
    std::mt19937 rng(static_cast<std::mt19937::result_type>(options.seed));

    const auto init_begin = Clock::now();
    Matrix centroids = initialize_centroids(data, options, rng);
    const auto init_end = Clock::now();

    DeviceBuffer<Scalar> d_data(data.values.size());
    DeviceBuffer<Scalar> d_centroids(centroids.values.size());
    DeviceBuffer<Scalar> d_old_centroids(centroids.values.size());
    DeviceBuffer<int> d_labels(data.rows);
    DeviceBuffer<Scalar> d_distances(data.rows);
    DeviceBuffer<Scalar> d_sums(centroids.values.size());
    DeviceBuffer<int> d_counts(options.clusters);

    TimingBreakdown timing;
    timing.init_ms = elapsed_ms(init_begin, init_end);
    timing.h2d_ms += timed_copy_h2d(data.values.data(), d_data.get(), d_data.bytes());
    timing.h2d_ms += timed_copy_h2d(centroids.values.data(), d_centroids.get(), d_centroids.bytes());

    const int threads = options.threads_per_block;
    const int point_blocks = (data.rows + threads - 1) / threads;
    const int centroid_blocks = (static_cast<int>(centroid_values) + threads - 1) / threads;

    std::vector<int> labels(data.rows, 0);
    std::vector<Scalar> distances(data.rows, 0.0f);
    std::vector<IterationRecord> history;
    double current_inertia = 0.0;
    bool converged = false;
    int iterations = 0;

    for (int iter = 0; iter < options.max_iterations; ++iter) {
        const auto iteration_begin = Clock::now();
        Matrix previous_centroids = centroids;

        ScopedEvent kernel_begin;
        ScopedEvent kernel_end;
        GPUKMEANS_CUDA_CHECK(cudaEventRecord(kernel_begin.get()));

        launch_assignment(options.kernel, d_data, d_centroids, d_labels, d_distances, data.rows, data.cols,
                          options.clusters, point_blocks, threads, shared_bytes);
        GPUKMEANS_CUDA_CHECK(cudaMemcpy(d_old_centroids.get(), d_centroids.get(), d_centroids.bytes(),
                                        cudaMemcpyDeviceToDevice));
        GPUKMEANS_CUDA_CHECK(cudaMemset(d_sums.get(), 0, d_sums.bytes()));
        GPUKMEANS_CUDA_CHECK(cudaMemset(d_counts.get(), 0, d_counts.bytes()));
        accumulate_kernel<<<point_blocks, threads>>>(d_data.get(), d_labels.get(), d_sums.get(), d_counts.get(),
                                                     data.rows, data.cols);
        GPUKMEANS_CUDA_CHECK(cudaGetLastError());
        finalize_centroids_kernel<<<centroid_blocks, threads>>>(d_centroids.get(), d_old_centroids.get(),
                                                                d_sums.get(), d_counts.get(), data.cols,
                                                                options.clusters);
        GPUKMEANS_CUDA_CHECK(cudaGetLastError());

        GPUKMEANS_CUDA_CHECK(cudaEventRecord(kernel_end.get()));
        GPUKMEANS_CUDA_CHECK(cudaEventSynchronize(kernel_end.get()));
        timing.kernel_ms += event_elapsed_ms(kernel_begin.get(), kernel_end.get());

        timing.d2h_ms += timed_copy_d2h(centroids.values.data(), d_centroids.get(), d_centroids.bytes());
        const Scalar shift = max_centroid_shift(previous_centroids, centroids);
        iterations = iter + 1;
        converged = shift <= options.tolerance;

        if (options.collect_history) {
            detail::assign_labels(data, centroids, &current_inertia);
            history.push_back({iterations,
                               current_inertia,
                               shift,
                               elapsed_ms(iteration_begin, Clock::now())});
        }

        if (converged) {
            break;
        }
    }

    ScopedEvent final_kernel_begin;
    ScopedEvent final_kernel_end;
    GPUKMEANS_CUDA_CHECK(cudaEventRecord(final_kernel_begin.get()));
    launch_assignment(options.kernel, d_data, d_centroids, d_labels, d_distances, data.rows, data.cols,
                      options.clusters, point_blocks, threads, shared_bytes);
    GPUKMEANS_CUDA_CHECK(cudaEventRecord(final_kernel_end.get()));
    GPUKMEANS_CUDA_CHECK(cudaEventSynchronize(final_kernel_end.get()));
    timing.kernel_ms += event_elapsed_ms(final_kernel_begin.get(), final_kernel_end.get());

    timing.d2h_ms += timed_copy_d2h(labels.data(), d_labels.get(), d_labels.bytes());
    timing.d2h_ms += timed_copy_d2h(distances.data(), d_distances.get(), d_distances.bytes());
    current_inertia = std::accumulate(distances.begin(), distances.end(), 0.0);

    timing.total_ms = elapsed_ms(total_begin, Clock::now());

    FitResult result;
    result.centroids = std::move(centroids);
    result.labels = std::move(labels);
    result.inertia = current_inertia;
    result.iterations = iterations;
    result.converged = converged;
    result.timing = timing;
    result.history = std::move(history);
    result.backend = to_string(Backend::Cuda);
    result.kernel = to_string(options.kernel);
    result.init = to_string(options.init);
    result.variant = to_string(options.variant);
    return result;
}

} // namespace detail
} // namespace gpukmeans
