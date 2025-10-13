cmake_minimum_required(VERSION 3.26.4 FATAL_ERROR)
project(tp_scan LANGUAGES CXX CUDA)

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

include(cmake/CPM.cmake)

# Get the latest CCCL version
CPMAddPackage(
  NAME CCCL
  GITHUB_REPOSITORY "nvidia/cccl"
  GIT_TAG "main"
)

CPMAddPackage(NAME rmm
              GITHUB_REPOSITORY rapidsai/rmm
              VERSION 25.12.00a
              SOURCE_SUBDIR cpp
              SYSTEM Off)

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


add_library(to_bench
  src/to_bench.cu)
target_link_libraries(to_bench PUBLIC rmm::rmm CCCL::CCCL benchmark::benchmark)

add_library(test_helpers
  bench/test_helpers.cu)
target_link_libraries(test_helpers PUBLIC rmm::rmm CCCL::CCCL benchmark::benchmark)

include_directories(src)

add_executable(bench bench/main.cc)
target_link_libraries(bench to_bench test_helpers benchmark::benchmark)