cmake_minimum_required(VERSION 3.25)
project(MyCudaProject LANGUAGES CXX CUDA)

find_package(rmm REQUIRED)
find_package(raft REQUIRED)
find_package(benchmark REQUIRED)

# Set the CUDA standard
set(CMAKE_CUDA_STANDARD 17)
set(CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fconcepts")
set(CMAKE_CUDA_ARCHITECTURES  "75")

include_directories(src)

add_compile_options(-W -Wall -Wextra -fconcepts)

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Ofast -march=native -flto -funroll-loops -ffast-math")
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
set(CMAKE_CMAKE_CUDA_ARCHITECTURES  70)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_FLAGS "--extended-lambda --generate-line-info --expt-relaxed-constexpr ${CMAKE_CUDA_FLAGS}")
#set(CMAKE_CUDA_FLAGS "--expt-relaxed-constexpr --ptxas-options=-v --generate-line-info")

add_executable(main src/main.cu)

# BUILD_TYPE
if (NOT CMAKE_BUILD_TYPE)
    string(REGEX REPLACE ".*/build_" "" BUILD_NAME ${CMAKE_BINARY_DIR})
    # Default BUILD_TYPE is Release
    set(CMAKE_BUILD_TYPE Release)
    # If folder is named "build_debug" BUILD_TYPE is set to: Debug
    if(${BUILD_NAME} STREQUAL "Debug" OR ${BUILD_NAME} STREQUAL "debug")
        set(CMAKE_BUILD_TYPE Debug)
    endif()
    message(STATUS "Build type : ${CMAKE_BUILD_TYPE}")
endif()