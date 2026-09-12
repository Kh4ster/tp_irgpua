#pragma once

#include <benchmark/benchmark.h>
#include <cuda/stream>
#include <cuda/buffer>
#include <cuda/memory_resource>
#include <numeric>
#include <vector>

#include "test_helpers.cuh"

class Fixture
{
  public:
    static bool no_check;

    template <typename FUNC, typename... Args>
    void
    bench_scan(benchmark::State& st, FUNC callback, int size, Args&&... args)
    {
        cuda::stream stream{cuda::device_ref{0}};
        auto device_resource = cuda::device_default_memory_pool(cuda::device_ref{0});
        auto buffer = cuda::make_buffer<int>(stream, device_resource, size, cuda::no_init);
        fill_buffer(buffer, 1);

        for (auto _ : st)
        {
            st.PauseTiming();
            fill_buffer(buffer, 1);
            st.ResumeTiming();
            callback(buffer);
        }

        st.SetBytesProcessed(int64_t(st.iterations()) *
                             int64_t(size * sizeof(int)));

        std::vector<int> expected(size);
        std::iota(expected.begin(), expected.end(), 1);
        if (!no_check)
            check_buffer(buffer, expected, st);
    }

    template <typename FUNC>
    void register_scan(benchmark::State& st, FUNC func)
    {
        int size = st.range(0);
        this->bench_scan(st, func, size);
    }
};

bool Fixture::no_check = false;