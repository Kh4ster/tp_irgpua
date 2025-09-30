#pragma once

#include <benchmark/benchmark.h>

#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

template <typename T>
void check_buffer(const rmm::device_scalar<T>& buffer,
                  T expected,
                  benchmark::State& st);

template <typename T>
void fill_buffer(rmm::device_uvector<T>& buffer,
                 T value);