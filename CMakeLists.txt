cmake_minimum_required(VERSION 3.20)

project(GPUKMeans
    VERSION 0.1.0
    DESCRIPTION "Modular CPU/CUDA K-means clustering library and benchmark suite"
    LANGUAGES CXX)

include(CheckLanguage)
include(GNUInstallDirs)

option(GPUKMEANS_ENABLE_CUDA "Try to build the CUDA backend" ON)
option(GPUKMEANS_REQUIRE_CUDA "Fail configuration if CUDA is unavailable" OFF)
set(GPUKMEANS_CUDA_ARCHITECTURES "75" CACHE STRING "CUDA architectures for gpukmeans")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

check_language(CUDA)
set(GPUKMEANS_HAVE_CUDA OFF)
if(GPUKMEANS_ENABLE_CUDA AND CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(GPUKMEANS_HAVE_CUDA ON)
elseif(GPUKMEANS_REQUIRE_CUDA)
    message(FATAL_ERROR "CUDA requested but no CUDA compiler was found")
else()
    message(STATUS "CUDA compiler not found or disabled; building CPU-only gpukmeans")
endif()

add_library(gpukmeans
    src/distributed.cpp
    src/io.cpp
    src/kmeans.cpp
    src/synthetic.cpp)

target_include_directories(gpukmeans
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

target_compile_features(gpukmeans PUBLIC cxx_std_17)

if(MSVC)
    target_compile_options(gpukmeans PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W4>)
else()
    target_compile_options(gpukmeans PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall>
        $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
        $<$<COMPILE_LANGUAGE:CXX>:-Wpedantic>)
endif()

if(GPUKMEANS_HAVE_CUDA)
    target_sources(gpukmeans PRIVATE src/kmeans_cuda.cu)
    set_target_properties(gpukmeans PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        CUDA_ARCHITECTURES "${GPUKMEANS_CUDA_ARCHITECTURES}"
        POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(gpukmeans PUBLIC GPUKMEANS_HAVE_CUDA=1)
else()
    target_sources(gpukmeans PRIVATE src/kmeans_cuda_stub.cpp)
    target_compile_definitions(gpukmeans PUBLIC GPUKMEANS_HAVE_CUDA=0)
endif()

add_executable(kmeans_cli apps/kmeans_cli.cpp)
target_link_libraries(kmeans_cli PRIVATE gpukmeans)

add_executable(kmeans_benchmark apps/kmeans_benchmark.cpp)
target_link_libraries(kmeans_benchmark PRIVATE gpukmeans)

enable_testing()
add_executable(gpukmeans_tests tests/test_kmeans.cpp)
target_link_libraries(gpukmeans_tests PRIVATE gpukmeans)
add_test(NAME gpukmeans_tests COMMAND gpukmeans_tests)

install(TARGETS gpukmeans kmeans_cli kmeans_benchmark
    EXPORT GPUKMeansTargets
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT GPUKMeansTargets
    NAMESPACE GPUKMeans::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/GPUKMeans)
