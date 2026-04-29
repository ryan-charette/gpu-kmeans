#include "gpukmeans/synthetic.hpp"

#include <algorithm>
#include <random>
#include <stdexcept>

namespace gpukmeans {

Matrix generate_blobs(int rows,
                      int cols,
                      int clusters,
                      std::uint64_t seed,
                      Scalar cluster_stddev) {
    if (rows <= 0 || cols <= 0) {
        throw std::invalid_argument("synthetic data dimensions must be positive");
    }
    if (clusters <= 0 || clusters > rows) {
        throw std::invalid_argument("cluster count must be in [1, rows]");
    }
    if (cluster_stddev <= 0.0f) {
        throw std::invalid_argument("cluster_stddev must be positive");
    }

    std::mt19937 rng(static_cast<std::mt19937::result_type>(seed));
    std::uniform_real_distribution<Scalar> center_distribution(-10.0f, 10.0f);
    std::normal_distribution<Scalar> noise_distribution(0.0f, cluster_stddev);

    Matrix centers(clusters, cols);
    for (Scalar& value : centers.values) {
        value = center_distribution(rng);
    }

    std::vector<int> cluster_ids(rows);
    for (int r = 0; r < rows; ++r) {
        cluster_ids[r] = r % clusters;
    }
    std::shuffle(cluster_ids.begin(), cluster_ids.end(), rng);

    Matrix data(rows, cols);
    for (int r = 0; r < rows; ++r) {
        const Scalar* center = centers.row(cluster_ids[r]);
        Scalar* point = data.row(r);
        for (int d = 0; d < cols; ++d) {
            point[d] = center[d] + noise_distribution(rng);
        }
    }
    return data;
}

} // namespace gpukmeans
