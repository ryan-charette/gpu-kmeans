# GPU K-means

GPU K-means is a small C++17/CUDA clustering library, command-line tool, and benchmark runner for comparing CPU K-means with multiple CUDA kernel strategies.

This project is a reusable open-source project with a library API, CPU reference implementation, selectable GPU kernels, benchmarking output, tests, and release-oriented documentation.

## What Is Included

- Reusable C++ API in `include/gpukmeans/`
- CPU Lloyd K-means reference implementation
- CPU mini-batch K-means implementation
- Random and K-means++ initialization
- Optional CUDA backend with `naive` and `shared` kernels
- Scaffolded `warp` and `fused` strategy enum values for future implementations
- CLI frontend for CSV input or synthetic data generation
- Benchmark runner with CSV/JSON output
- Basic correctness tests
- Python plotting utilities for 2D clusters and benchmark CSVs
- Device-selection and row-partitioning hooks for future multi-GPU work
- CPU-only GitHub Actions CI workflow

## Build

Prerequisites:

- CMake 3.20+
- A C++17 compiler
- Optional: NVIDIA CUDA Toolkit for GPU builds

CPU-only build:

```bash
cmake -S . -B build -DGPUKMEANS_ENABLE_CUDA=OFF
cmake --build build --config Release
ctest --test-dir build --output-on-failure
```

CUDA build:

```bash
cmake -S . -B build -DGPUKMEANS_ENABLE_CUDA=ON -DGPUKMEANS_CUDA_ARCHITECTURES=75
cmake --build build --config Release
ctest --test-dir build --output-on-failure
```

If CUDA is not found and `GPUKMEANS_REQUIRE_CUDA` is `OFF`, CMake builds a CPU-only library with a CUDA stub that reports a clear runtime error when selected.

## CLI Examples

Generate synthetic data and run CPU K-means:

```bash
./build/kmeans_cli --rows 10000 --dims 16 --k 8 --iterations 50 --seed 42 \
  --backend cpu --output assignments.csv --centroids centroids.csv --json fit.json
```

Run CUDA with the naive global-memory strategy:

```bash
./build/kmeans_cli --rows 100000 --dims 32 --k 16 --backend cuda --kernel naive
```

Run CUDA with shared-memory centroid caching:

```bash
./build/kmeans_cli --rows 100000 --dims 32 --k 16 --backend cuda --kernel shared
```

Read a CSV file:

```bash
./build/kmeans_cli --input data.csv --k 4 --has-header --backend cpu
```

Read the original coursework-style sample files, which have a row-count line and a leading record id:

```bash
./build/kmeans_cli --input input/random-n2048-d16-c16.txt --k 16 \
  --skip-first-count-line --drop-first-column --backend cpu
```

Mini-batch K-means is currently implemented for the CPU backend:

```bash
./build/kmeans_cli --rows 50000 --dims 8 --k 12 --variant mini-batch --mini-batch 512
```

## C++ API Example

```cpp
#include "gpukmeans/kmeans.hpp"
#include "gpukmeans/synthetic.hpp"

int main() {
    gpukmeans::Matrix data = gpukmeans::generate_blobs(10000, 8, 4, 42);

    gpukmeans::KMeansOptions options;
    options.clusters = 4;
    options.init = gpukmeans::InitMethod::KMeansPlusPlus;
    options.backend = gpukmeans::Backend::Cpu;
    options.max_iterations = 100;

    gpukmeans::KMeans model(options);
    gpukmeans::FitResult result = model.fit(data);
    std::vector<int> labels = model.predict(data);
    double loss = model.inertia(data);
}
```

The free function API is also available:

```cpp
gpukmeans::FitResult result = gpukmeans::fit(data, 4, options);
```

## Benchmarks

Run a CPU vs GPU strategy comparison over multiple dataset sizes:

```bash
./build/kmeans_benchmark --rows 10000,100000 --dims 8,32 --k 8,16 \
  --iterations 50 --backend both --kernels naive,shared \
  --output benchmarks.csv --format csv
```

JSON output:

```bash
./build/kmeans_benchmark --rows 10000 --dims 16 --k 8 \
  --backend cpu --output benchmarks.json --format json
```

Benchmark rows include:

- dataset rows
- dimensionality
- number of clusters
- max iterations
- repeat index and seed
- backend and kernel strategy
- convergence metadata
- inertia
- total time
- initialization time
- CPU time
- host-to-device transfer time
- CUDA kernel time
- device-to-host transfer time
- status and error message

Plot benchmark CSV output:

```bash
python tools/plot_benchmarks.py --input benchmarks.csv --output benchmarks.png --x rows
```

Plot 2D clusters:

```bash
python tools/plot_clusters.py --data data.csv --assignments assignments.csv \
  --centroids centroids.csv --output clusters.png
```

## Architecture

```text
include/gpukmeans/
  kmeans.hpp        Public API, options, result metadata, strategy enums
  io.hpp            Matrix and result file IO helpers
  synthetic.hpp     Synthetic blob generator
  distributed.hpp   Device, partitioning, and centroid reduction hooks

src/
  kmeans.cpp        CPU reference implementation and public API glue
  kmeans_cuda.cu    CUDA backend and kernels
  kmeans_cuda_stub.cpp
  io.cpp
  synthetic.cpp
  distributed.cpp

apps/
  kmeans_cli.cpp
  kmeans_benchmark.cpp

tests/
  test_kmeans.cpp

tools/
  plot_clusters.py
  plot_benchmarks.py
```

## Kernel Strategies

`naive`

One thread assigns one point to its nearest centroid. Centroids are read from global memory. Updates use global atomics into centroid sums and counts. This is simple and useful as the baseline GPU implementation.

`shared`

Each block caches the centroid matrix in shared memory before assignment. This can reduce repeated global-memory reads when `k * dimensions * sizeof(float)` fits in per-block shared memory. The update step remains a simple atomic accumulation.

`warp`

Scaffolded strategy enum and CLI flag. Intended future direction: use warp-level primitives to reduce per-point distance computations or reduce centroid update contention.

`fused`

Scaffolded strategy enum and CLI flag. Intended future direction: combine assignment and partial update using block-local accumulation to reduce global memory traffic. This is not advertised as working yet.

## Correctness And Stability

- Deterministic seed support uses `std::mt19937`.
- CPU tests cover invalid inputs, convergence on simple data, deterministic runs, prediction/inertia APIs, and mini-batch smoke behavior.
- CUDA parity tests run only when the project is compiled with CUDA and a CUDA device is available.
- Empty clusters keep their previous centroid, which avoids introducing hidden nondeterministic reseeding inside the iteration loop.
- Convergence uses maximum centroid L2 shift rather than division by old centroid values.

## Multi-GPU Design Hooks

Full multi-GPU execution is not implemented yet. The public scaffolding includes:

- `DeviceDescriptor` for device discovery
- `RowPartition` and `partition_rows()` for partitioning rows across devices
- `CentroidReducer` abstraction for combining partial centroid sums and counts
- `SingleProcessCentroidReducer` as the current local reducer

Future distributed work should add CUDA-aware reducers and NCCL/all-reduce support behind the reducer interface.

## Performance Discussion Template

When publishing benchmark results, include:

- GPU model, CPU model, CUDA version, compiler, and build type
- Dataset rows, dimensions, clusters, and initialization method
- Number of iterations and convergence tolerance
- Separate host-to-device, kernel, and device-to-host timings
- CPU baseline timing and speedup
- Strategy comparison for `naive` vs `shared`
- Notes on shared-memory capacity and atomic update contention
- Whether timings include synthetic data generation

## Limitations And Future Work

- `warp` and `fused` strategies are clean scaffolds, not implemented kernels.
- CUDA mini-batch K-means is not implemented.
- Multi-GPU support is architectural only; NCCL/all-reduce is TODO.
- The CUDA update path uses atomics and favors clarity over peak performance.
- CSV parsing is intentionally lightweight and numeric-only.
- Only float32 is implemented today.
- There is no package manager integration yet.

## License

MIT. See `LICENSE`.
