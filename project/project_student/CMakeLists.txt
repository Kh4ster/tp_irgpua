cmake_minimum_required(VERSION 3.25)
project(IRGPUA LANGUAGES CXX CUDA)

find_package(OpenMP REQUIRED)
find_package(rmm REQUIRED)
find_package(raft REQUIRED)
find_package(benchmark REQUIRED)


# Set the CUDA standard
set(CMAKE_CUDA_STANDARD 17)
set(CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fconcepts")
set(CUDA_ARCHITECTURES "75")

include_directories(src)

add_executable(main src/main.cu src/fix_cpu.cu)
target_compile_options(main PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${OpenMP_CXX_FLAGS}>)
target_link_libraries(main PUBLIC OpenMP::OpenMP_CXX rmm::rmm raft::raft)
