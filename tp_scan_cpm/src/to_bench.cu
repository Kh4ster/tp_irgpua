#include "to_bench.cuh"

#include "cuda_tools/cuda_error_checking.cuh"

#include <cuda/std/span>

#include <rmm/device_uvector.hpp>

template <typename T>
__global__
void kernel_scan_baseline(cuda::std::span<T> buffer)
{
    for (int i = 1; i < buffer.size(); ++i)
        buffer[i] += buffer[i - 1];
}

void baseline_scan(rmm::device_uvector<int>& buffer)
{
	kernel_scan_baseline<int><<<1, 1, 0, buffer.stream()>>>(
        cuda::std::span<int>(buffer.data(), buffer.size()));

    CUDA_CHECK_ERROR(cudaStreamSynchronize(buffer.stream()));
}

template <typename T>
__global__
void kernel_your_scan(cuda::std::span<T> buffer)
{
    // TODO
    // ...
}

void your_scan(rmm::device_uvector<int>& buffer)
{
    // TODO
    // ...

	kernel_your_scan<int><<<1, 1, 0, buffer.stream()>>>(
        cuda::std::span<int>(buffer.data(), buffer.size()));

    CUDA_CHECK_ERROR(cudaStreamSynchronize(buffer.stream()));
}