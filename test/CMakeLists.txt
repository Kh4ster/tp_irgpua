cmake_minimum_required(VERSION 3.30)
project(MyCudaProject LANGUAGES CXX CUDA)

set(CUDA_ARCHITECTURES "75")

# Find the RMM library
find_package(rmm REQUIRED)

# Set the CUDA standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

# Add the executable
add_executable(test_rmm test_rmm.cu)

# Link against the RMM library
target_link_libraries(test_rmm PRIVATE rmm::rmm)