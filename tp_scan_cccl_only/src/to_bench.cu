#include "to_bench.cuh"

#include "cuda_tools/cuda_error_checking.cuh"

#include <cuda/std/span>

#include <cuda/buffer>

template <typename T>
__global__
void kernel_scan_baseline(cuda::std::span<T> buffer)
{
    for (int i = 1; i < buffer.size(); ++i)
        buffer[i] += buffer[i - 1];
}

void baseline_scan(cuda::device_buffer<int>& buffer)
{
	kernel_scan_baseline<int><<<1, 1, 0, buffer.stream().get()>>>(buffer);

    buffer.stream().sync();
}

template <typename T>
__global__
void kernel_your_scan(cuda::std::span<T> buffer)
{
    // TODO
    // ...
}

void your_scan(cuda::device_buffer<int>& buffer)
{
    // TODO
    // ...

	kernel_your_scan<int><<<1, 1, 0, buffer.stream().get()>>>(buffer);

    buffer.stream().sync();
}