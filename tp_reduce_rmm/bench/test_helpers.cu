#include "test_helpers.cuh"
#include "cuda_tools/cuda_error_checking.cuh"

#include <benchmark/benchmark.h>

#include <iostream>

#include <raft/core/handle.hpp>

#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/cuda_async_memory_resource.hpp>
#include <rmm/mr/device/owning_wrapper.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

template <typename T>
void check_buffer(rmm::device_scalar<T>& scalar,
                  T expected,
                  benchmark::State& st)
{
    const T value = scalar.value(scalar.stream());

    if (value != expected)
    {
        std::cout << "Expected " << expected << ", got " << value << std::endl;
        st.SkipWithError("Failed test");
    }
}

template <typename T>
void fill_buffer(const raft::handle_t& handle,
                 rmm::device_uvector<T>& buffer,
                 T val)
{
    thrust::uninitialized_fill(handle.get_thrust_policy(),
                               buffer.begin(),
                               buffer.end(),
                               val);
}

/*
inline auto make_async()
{
    return std::make_shared<rmm::mr::cuda_async_memory_resource>();
}

auto make_pool() -> decltype(rmm::mr::make_owning_wrapper<rmm::mr::pool_memory_resource>(std::make_shared<rmm::mr::cuda_async_memory_resource>(), 0))
{
    size_t free_mem, total_mem;
    CUDA_CHECK_ERROR(cudaMemGetInfo(&free_mem, &total_mem));
    size_t rmm_alloc_gran = 256;
    double alloc_ratio = 0.8;
    // 80% of the GPU memory is the recommanded amount
    size_t initial_pool_size =
        (size_t(free_mem * alloc_ratio) / rmm_alloc_gran) * rmm_alloc_gran;
    return rmm::mr::make_owning_wrapper<rmm::mr::pool_memory_resource>(
        make_async(),
        initial_pool_size);
}*/

template void check_buffer(rmm::device_scalar<int>& scalar,
                           int expected,
                           benchmark::State& st);

template void fill_buffer(const raft::handle_t& handle,
                          rmm::device_uvector<int>& buffer,
                          int val);