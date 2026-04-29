#include "gpukmeans/distributed.hpp"

#include <stdexcept>

namespace gpukmeans {

Matrix SingleProcessCentroidReducer::reduce(
    const std::vector<Matrix>& partial_sums,
    const std::vector<std::vector<int>>& partial_counts) const {
    if (partial_sums.empty() || partial_sums.size() != partial_counts.size()) {
        throw std::invalid_argument("partial sums and counts must be non-empty and aligned");
    }

    const int clusters = partial_sums.front().rows;
    const int cols = partial_sums.front().cols;
    Matrix sums(clusters, cols);
    std::vector<int> counts(clusters, 0);

    for (std::size_t part = 0; part < partial_sums.size(); ++part) {
        if (partial_sums[part].rows != clusters || partial_sums[part].cols != cols ||
            static_cast<int>(partial_counts[part].size()) != clusters) {
            throw std::invalid_argument("all partial centroid buffers must have the same shape");
        }
        for (int c = 0; c < clusters; ++c) {
            counts[c] += partial_counts[part][c];
            for (int d = 0; d < cols; ++d) {
                sums.row(c)[d] += partial_sums[part].row(c)[d];
            }
        }
    }

    for (int c = 0; c < clusters; ++c) {
        if (counts[c] == 0) {
            continue;
        }
        for (int d = 0; d < cols; ++d) {
            sums.row(c)[d] = static_cast<Scalar>(sums.row(c)[d] / static_cast<Scalar>(counts[c]));
        }
    }
    return sums;
}

std::vector<DeviceDescriptor> discover_devices() {
    return {{0, "cpu", 0}};
}

std::vector<RowPartition> partition_rows(int rows, const std::vector<int>& device_ids) {
    if (rows < 0) {
        throw std::invalid_argument("rows must be non-negative");
    }
    if (device_ids.empty()) {
        throw std::invalid_argument("at least one device id is required");
    }

    std::vector<RowPartition> partitions;
    partitions.reserve(device_ids.size());
    const int base = rows / static_cast<int>(device_ids.size());
    int remainder = rows % static_cast<int>(device_ids.size());
    int begin = 0;
    for (int device_id : device_ids) {
        const int extra = remainder > 0 ? 1 : 0;
        if (remainder > 0) {
            --remainder;
        }
        const int end = begin + base + extra;
        partitions.push_back({device_id, begin, end});
        begin = end;
    }
    return partitions;
}

} // namespace gpukmeans
