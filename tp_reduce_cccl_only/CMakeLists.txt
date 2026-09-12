cmake_minimum_required(VERSION 3.26.4 FATAL_ERROR)
project(tp_reduce LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

include(FetchContent)

FetchContent_Declare(googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG main)

FetchContent_Declare(googlebenchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG main)

FetchContent_MakeAvailable(
        googletest
        googlebenchmark)

# This example uses the CMake Package Manager (CPM) to simplify fetching CCCL from GitHub
# For more information, see https://github.com/cpm-cmake/CPM.cmake
include(cmake/CPM.cmake)

CPMAddPackage(
        NAME CCCL
        GITHUB_REPOSITORY "nvidia/cccl"
        GIT_TAG "v3.5.0.dev"
        )
              

# Set the CUDA standard
set(CMAKE_CUDA_STANDARD 17)
# Set the C++ standard
set(CXX_STANDARD 17)
# Add fconcept for lambdas
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fconcepts")
# Generate cubin for this specific GPU
set(CMAKE_CUDA_ARCHITECTURES "native")
# Set fpic for Google Benchmark
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_library(to_bench STATIC
  src/to_bench.cu)
target_link_libraries(to_bench PUBLIC benchmark::benchmark PRIVATE CCCL::CCCL)

add_library(test_helpers STATIC
  bench/test_helpers.cu)
target_link_libraries(test_helpers PUBLIC benchmark::benchmark PRIVATE CCCL::CCCL)

include_directories(src)

add_executable(bench bench/main.cc)
target_link_libraries(bench
    PRIVATE CUDA::cudart
    to_bench
    test_helpers
    benchmark::benchmark
    CCCL::CCCL)