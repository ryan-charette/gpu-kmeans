#pragma once

#include "gpukmeans/kmeans.hpp"

#include <cstdint>

namespace gpukmeans {

Matrix generate_blobs(int rows,
                      int cols,
                      int clusters,
                      std::uint64_t seed,
                      Scalar cluster_stddev = 0.6f);

} // namespace gpukmeans
