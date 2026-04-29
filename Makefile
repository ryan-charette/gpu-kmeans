BUILD_DIR ?= build
CMAKE ?= cmake

.PHONY: all configure build test clean

all: configure build

configure:
	$(CMAKE) -S . -B $(BUILD_DIR) -DGPUKMEANS_ENABLE_CUDA=ON

build:
	$(CMAKE) --build $(BUILD_DIR) --config Release

test:
	$(CMAKE) --build $(BUILD_DIR) --config Release --target gpukmeans_tests
	ctest --test-dir $(BUILD_DIR) --output-on-failure

clean:
	$(CMAKE) -E rm -rf $(BUILD_DIR)
