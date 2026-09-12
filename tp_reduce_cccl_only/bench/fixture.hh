#pragma once

#include <benchmark/benchmark.h>

#include <cuda/stream>
#include <cuda/buffer>
#include <cuda/algorithm>
#include <cuda/buffer>
#include <cuda/memory_resource>
#include <cuda/stream>
#include <cuda/memory>
#include <cuda/launch>
#include <cuda/algorithm>
#include <cuda/std/mdspan>
#include <cuda/std/span>
#include <cuda/cmath>

#include "test_helpers.cuh"

class Fixture
{
  public:
    static bool no_check;

    template <typename FUNC, typename... Args>
    void
    bench_reduce(benchmark::State& st, FUNC callback, int size, Args&&... args)
    {
        cuda::stream stream{cuda::device_ref{0}};
        // Resource to handle the GPU memory allocations
        cuda::device_memory_pool_ref device_resource = cuda::device_default_memory_pool(cuda::device_ref{0});
        cuda::device_buffer<int> buffer = cuda::make_buffer<int>(stream, device_resource, size, cuda::no_init);
        cuda::device_buffer<int> total = cuda::make_buffer<int>(stream, device_resource, 1, cuda::no_init);
        fill_buffer(buffer, 1);

        for (auto _ : st)
        {
            st.PauseTiming();
            cuda::fill_bytes(stream, total, 0);
            st.ResumeTiming();
            callback(buffer, total);
        }

        st.SetBytesProcessed(int64_t(st.iterations()) *
                             int64_t(size * sizeof(int)));

        if (!no_check)
            check_buffer(total, size, st);
    }

    template <typename FUNC>
    void register_reduce(benchmark::State& st, FUNC func)
    {
        int size = st.range(0);
        this->bench_reduce(st, func, size);
    }
};

bool Fixture::no_check = false;