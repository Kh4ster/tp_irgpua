#include "test_helpers.cuh"
#include "cuda_tools/cuda_error_checking.cuh"

#include <benchmark/benchmark.h>

#include <iostream>

#include <cuda/algorithm>
#include <cuda/buffer>
#include <cuda/std/span>

#include <algorithm>
#include <vector>

#include <thrust/uninitialized_fill.h>

template <typename T>
static auto host_copy(const cuda::device_buffer<T>& device_buffer)
{
  std::vector<T> host_vec(device_buffer.size());
  cuda::copy_bytes(device_buffer.stream(),
                   device_buffer,
                   host_vec);
  device_buffer.stream().sync();
  return host_vec;
}

template <typename T>
void check_buffer(const cuda::device_buffer<T>& buffer,
                  const std::vector<T>& expected,
                  benchmark::State& st)
{
    const auto& host_buffer = host_copy(buffer);

    if (!std::equal(host_buffer.cbegin(),
                    host_buffer.cend(),
                    expected.cbegin()))
    {
        auto [first, second] = std::mismatch(host_buffer.cbegin(),
                                             host_buffer.cend(),
                                             expected.cbegin());
        std::cout << "Error at " << first - host_buffer.cbegin() << ": "
                  << *first << " " << *second << std::endl;
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

template void check_buffer(const cuda::device_buffer<int>& vector,
                           const std::vector<int>& expected,
                           benchmark::State& st);

template void fill_buffer(cuda::device_buffer<int>& buffer,
                          int val);