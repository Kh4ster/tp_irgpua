#pragma once

#include <benchmark/benchmark.h>

#include <cuda/buffer>

#include <vector>

template <typename T>
void check_buffer(const cuda::device_buffer<T>& buffer,
                  const std::vector<T>& expected,
                  benchmark::State& st);

template <typename T>
void fill_buffer(cuda::device_buffer<T>& buffer,
                 T value);