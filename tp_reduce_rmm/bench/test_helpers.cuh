#pragma once

#include <benchmark/benchmark.h>

#include <raft/core/handle.hpp>

#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>
//#include <rmm/mr/device/cuda_async_memory_resource.hpp>
//#include <rmm/mr/device/owning_wrapper.hpp>
//#include <rmm/mr/device/pool_memory_resource.hpp>

template <typename T>
void check_buffer(rmm::device_scalar<T>& buffer,
                  T expected,
                  benchmark::State& st);

template <typename T>
void fill_buffer(const raft::handle_t& handle,
                 rmm::device_uvector<T>& buffer,
                 T value);

//auto make_pool() -> decltype(rmm::mr::make_owning_wrapper<rmm::mr::pool_memory_resource>(std::make_shared<rmm::mr::cuda_async_memory_resource>(), 0));