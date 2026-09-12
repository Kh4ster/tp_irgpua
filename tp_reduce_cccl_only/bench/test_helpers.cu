#include "test_helpers.cuh"
#include "cuda_tools/cuda_error_checking.cuh"

#include <benchmark/benchmark.h>

#include <iostream>

#include <cuda/buffer>
#include <cuda/algorithm>

#include <thrust/uninitialized_fill.h>

template <typename T>
void check_buffer(const cuda::device_buffer<T>& scalar,
                  T expected,
                  benchmark::State& st)
{
    T value;
    // Bring back value from device to host
    // Copy bytes only supports buffer/span so need to wrap value in a span
    cuda::copy_bytes(scalar.stream(), scalar, cuda::std::span<T>{&value, 1});
    // Need to sync as copy_bytes is asynchronous
    scalar.stream().sync();

    if (value != expected)
    {
        std::cout << "Expected " << expected << ", got " << value << std::endl;
        st.SkipWithError("Failed test");
    }
}

template <typename T>
void fill_buffer(cuda::device_buffer<T>& buffer,
                 T val)
{
    thrust::uninitialized_fill(thrust::cuda::par.on(buffer.stream().get()),
                               buffer.begin(),
                               buffer.end(),
                               val);
}

template void check_buffer(const cuda::device_buffer<int>& scalar,
                           int expected,
                           benchmark::State& st);

template void fill_buffer(cuda::device_buffer<int>& buffer,
                          int val);