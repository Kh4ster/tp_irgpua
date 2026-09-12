#include "to_bench.cuh"

#include "cuda_tools/cuda_error_checking.cuh"

#include <cuda/std/span>


template <typename T>
__global__
void kernel_reduce_baseline(cuda::std::span<const T> buffer, cuda::std::span<T> total)
{
    for (int i = 0; i < buffer.size(); ++i)
        *total.data() += buffer[i];
}

void baseline_reduce(cuda::device_buffer<int>& buffer,
                     cuda::device_buffer<int>& total)
{
	kernel_reduce_baseline<int><<<1, 1, 0, buffer.stream().get()>>>(buffer, total);

    buffer.stream().sync();
}

template <typename T>
__global__
void kernel_your_reduce(cuda::std::span<const T> buffer, cuda::std::span<T> total)
{
    // Help: odd size
    // When treating an odd size think about two things
    // 1. How could a thread sum two values and have the second (that we don't want for the odd case) not have any impact on the sum?
    // 2. Once 1. is achived, could we use a fixed even size while still achieving the same 

    // TODO
    // Your reduce code
}

void your_reduce(cuda::device_buffer<int>& buffer,
                 cuda::device_buffer<int>& total)
{
    // Help: more than 1 thread block
    // When treating the 2 thread block case you need to create a temporary array
    // To do so use the following API : cuda::device_buffer<int> tmp(<SIZE>, buffer.stream())

    // Help: very large case
    // Using only 2 kernels, what is the biggest buffer size we can handle?

    // TODO fill in blocks, threads, and shared memory
    // Help: To properly compute the amount of block, use the following API: cuda::ceil_div(<PROBLEM_SIZE>, <BLOCK_SIZE>)
	kernel_your_reduce<int><<<1 /*TODO FILL*/, 1 /*TODO FILL*/, 0 /*TODO FILL*/, buffer.stream().get()>>>(buffer, total);

    buffer.stream().sync();
}