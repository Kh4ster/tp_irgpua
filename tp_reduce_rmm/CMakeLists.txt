cmake_minimum_required(VERSION 3.25)
project(tp_reduce LANGUAGES CXX CUDA)

find_package(rmm REQUIRED)
find_package(raft REQUIRED)
find_package(benchmark REQUIRED)

# Set the CUDA standard
set(CMAKE_CUDA_STANDARD 17)
set(CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fconcepts")
set(CUDA_ARCHITECTURES "75")

add_library(to_bench
  src/to_bench.cu)
target_link_libraries(to_bench PUBLIC rmm::rmm raft::raft)

add_library(test_helpers
  bench/test_helpers.cu)
target_link_libraries(test_helpers PUBLIC rmm::rmm benchmark::benchmark)

include_directories(src)

add_executable(bench bench/main.cc)
target_link_libraries(bench to_bench test_helpers benchmark::benchmark)