#include "kmeans_internal.hpp"

#include <stdexcept>

namespace gpukmeans {

bool cuda_backend_compiled() {
    return false;
}

int cuda_device_count() {
    return 0;
}

namespace detail {

FitResult fit_cuda(const Matrix&, KMeansOptions) {
    throw std::runtime_error(
        "CUDA backend was not compiled. Reconfigure with a CUDA compiler and GPUKMEANS_ENABLE_CUDA=ON.");
}

} // namespace detail
} // namespace gpukmeans
