#pragma once

#include <benchmark/benchmark.h>

#include <cuda/buffer>

template <typename T>
void check_buffer(const cuda::device_buffer<T>& buffer,
                  T expected,
                  benchmark::State& st);

template <typename T>
void fill_buffer(cuda::device_buffer<T>& buffer,
                 T value);