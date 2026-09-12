#pragma once

#include <cuda/buffer>

void baseline_reduce(cuda::device_buffer<int>& buffer,
                     cuda::device_buffer<int>& total);

void your_reduce(cuda::device_buffer<int>& buffer,
                 cuda::device_buffer<int>& total);