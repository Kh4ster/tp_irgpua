#include "benchmark_registerer.hh"
#include "cuda_tools/cuda_error_checking.cuh"
#include "fixture.hh"
#include "to_bench.cuh"

#include <benchmark/benchmark.h>

#include <rmm/mr/device/cuda_async_memory_resource.hpp>
#include <rmm/mr/device/owning_wrapper.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include <cmath>
#include <tuple>

template <typename Tuple>
constexpr auto tuple_length(Tuple)
{
    return std::tuple_size_v<Tuple>;
}

inline auto make_async()
{
    return std::make_shared<rmm::mr::cuda_async_memory_resource>();
}
inline auto make_pool()
{
    // Allocate 2 Go
    size_t initial_pool_size = std::pow(2, 31);
    return rmm::mr::make_owning_wrapper<rmm::mr::pool_memory_resource>(
        make_async(),
        initial_pool_size);
}

int main(int argc, char** argv)
{
    // Google bench setup
    using benchmark_t = benchmark::internal::Benchmark;
    ::benchmark::Initialize(&argc, argv);
    bool bench_nsight = false;

    // RMM Setup
    auto memory_resource = make_pool();
    rmm::mr::set_current_device_resource(memory_resource.get());

    // Argument parsing
    for (int i = 1; i < argc; i++)
    {
        if (argv[i] == std::string_view("--no-check"))
        {
            Fixture::no_check = true;
            std::swap(argv[i], argv[--argc]);
        }
        // Set iteration number to 1 not to mess with nsight
        if (argv[i] == std::string_view("--bench-nsight"))
        {
            bench_nsight = true;
            std::swap(argv[i], argv[--argc]);
        }
    }

    // Benchmarks registration
    Fixture fx;
    {
        // Add the sizes to benchmark here
        // Start with 1 block of 64 (block reduce)
        // Then 2 blocks of 64 each (grid reduce)
        // Then and odd size
        // Finally the true sizes
        // TODO
        constexpr std::array sizes = {
            64
            // 128,
            // 129,
            // 524288,
            // 1048576
        };

        // Add the name and function to benchmark here
        // TODO
        constexpr std::tuple reduce_to_bench{
            "Baseline_reduce",
            &baseline_reduce,
            "Your_reduce",
            &your_reduce,
        };

        //  / 2 because we store name + function pointer
        benchmark_t* b[tuple_length(reduce_to_bench) / 2];
        int function_index = 0;

        // Call to registerer
        registerer_reduce(&fx,
                          b,
                          function_index,
                          sizes,
                          bench_nsight,
                          reduce_to_bench);
    }
    ::benchmark::RunSpecifiedBenchmarks();
}
