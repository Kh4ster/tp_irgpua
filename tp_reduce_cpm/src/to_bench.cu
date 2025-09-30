#include "to_bench.cuh"

#include "cuda_tools/cuda_error_checking.cuh"

#include <cuda/std/span>

#include <rmm/device_uvector.hpp>
#include <rmm/device_scalar.hpp>


template <typename T>
__global__
void kernel_reduce_baseline(cuda::std::span<const T> buffer, cuda::std::span<T> total)
{
    for (int i = 0; i < buffer.size(); ++i)
        *total.data() += buffer[i];
}

void baseline_reduce(rmm::device_uvector<int>& buffer,
                     rmm::device_scalar<int>& total)
{
	kernel_reduce_baseline<int><<<1, 1, 0, buffer.stream()>>>(
        cuda::std::span<int>(buffer.data(), buffer.size()),
        cuda::std::span<int>(total.data(), 1));

    CUDA_CHECK_ERROR(cudaStreamSynchronize(buffer.stream()));
}

template <typename T>
__global__
void kernel_your_reduce(cuda::std::span<const T> buffer, cuda::std::span<T> total)
{
    // TODO
    // ...
}

void your_reduce(rmm::device_uvector<int>& buffer,
                 rmm::device_scalar<int>& total)
{
    // TODO
    // ...

	kernel_your_reduce<int><<<1, 1, 0, buffer.stream()>>>(
        cuda::std::span<const int>(buffer.data(), buffer.size()),
        cuda::std::span<int>(total.data(), 1));

    CUDA_CHECK_ERROR(cudaStreamSynchronize(buffer.stream()));
}