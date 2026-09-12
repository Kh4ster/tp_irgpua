#pragma once

#include <cuda/buffer>

void baseline_scan(cuda::device_buffer<int>& buffer);

void your_scan(cuda::device_buffer<int>& buffer);