#pragma once

#include "gpukmeans/kmeans.hpp"

#include <string>
#include <vector>

namespace gpukmeans {

struct DeviceDescriptor {
    int id = 0;
    std::string name = "cpu";
    std::size_t global_memory_bytes = 0;
};

struct RowPartition {
    int device_id = 0;
    int row_begin = 0;
    int row_end = 0;
};

class CentroidReducer {
public:
    virtual ~CentroidReducer() = default;
    // Future multi-GPU implementations should replace this with NCCL/all-reduce backed reducers.
    virtual Matrix reduce(const std::vector<Matrix>& partial_sums,
                          const std::vector<std::vector<int>>& partial_counts) const = 0;
};

class SingleProcessCentroidReducer final : public CentroidReducer {
public:
    Matrix reduce(const std::vector<Matrix>& partial_sums,
                  const std::vector<std::vector<int>>& partial_counts) const override;
};

std::vector<DeviceDescriptor> discover_devices();
std::vector<RowPartition> partition_rows(int rows, const std::vector<int>& device_ids);

} // namespace gpukmeans
