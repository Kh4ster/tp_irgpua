#include "image.hh"

#include <random>
#include <vector>
#include <iostream>
#include <numeric>
#include <array>
#include <algorithm>

#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>

#include <cub/cub.cuh> 

#include <filesystem>

void generate_broken_image(Image& in)
{
    for (int i = 0; i < in.height * in.width; ++i)
    {
        if (i % 4 == 0)
            in.buffer[i] -= 1;
        else if (i % 4 == 1)
            in.buffer[i] += 5;
        else if (i % 4 == 2)
            in.buffer[i] -= 3;
        else if (i % 4 == 3)
            in.buffer[i] += 8;
    }

    std::default_random_engine rd(in.buffer[5]);
    std::uniform_int_distribution<int> proba_holes(0, 20);
    std::uniform_int_distribution<int> size_holes(1, 100);
    std::vector<int> out;
    for (int i = 0; i < in.height * in.width; ++i)
    {
        if (proba_holes(rd) == 5)
        {
            int nb_holes = size_holes(rd);
            for (int j = 0; j < nb_holes; ++j)
                out.emplace_back(-1);
        }
        out.emplace_back(in.buffer[i]);
    }

    in.buffer = std::move(out);
}

void fix_image_cpu(Image& to_fix)
{
    const int image_size = to_fix.width * to_fix.height;

    // #1 Compact

    // Build predicate vector
    std::vector<int> predicate(to_fix.buffer.size(), 0);

    // A mettre en for each / transform / generate
    for (std::size_t i = 0; i < to_fix.buffer.size(); ++i)
        if (to_fix.buffer[i] != -1)
            predicate[i] = 1;

    // Compute the exclusive sum of the predicate
    std::exclusive_scan(predicate.begin(), predicate.end(), predicate.begin(), 0);

    // Scatter to the corresponding addresses
    for (std::size_t i = 0; i < predicate.size(); ++i)
        if (to_fix.buffer[i] != -1)
            to_fix.buffer[predicate[i]] = to_fix.buffer[i];


    // #2 Apply map to fix pixels
    for (int i = 0; i < image_size; ++i)
    {
        if (i % 4 == 0)
            to_fix.buffer[i] += 1;
        else if (i % 4 == 1)
            to_fix.buffer[i] -= 5;
        else if (i % 4 == 2)
            to_fix.buffer[i] += 3;
        else if (i % 4 == 3)
            to_fix.buffer[i] -= 8;
    }

    // #3 Histogram
    std::array<int, 256> histo;
    std::generate(histo.begin(), histo.end(), [](){ return 0; });
    for (int i = 0; i < image_size; ++i)
        ++histo[to_fix.buffer[i]];

    // Compute the inclusive sum of the histogram
    std::inclusive_scan(histo.begin(), histo.end(), histo.begin());

    // Find the first non zero value in the cumulative histogram
    auto first_none_zero = std::find_if(histo.begin(), histo.end(), [](auto v) { return v != 0; });

    const int cdf_min = *first_none_zero;
    //std::cout << "cdf_min cpu " << cdf_min << std::endl;

    std::transform(to_fix.buffer.data(), to_fix.buffer.data() + image_size, to_fix.buffer.data(),
        [image_size, cdf_min, &histo](int pixel)
            { 
                return std::roundf( ((histo[pixel] - cdf_min) / static_cast<float>(image_size - cdf_min)) * 255.0f);
            }
    );
}

__global__
void map(int* fixed, int size)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size)
    {
        if (index % 4 == 0)
            fixed[index] += 1;
        else if (index % 4 == 1)
            fixed[index] -= 5;
        else if (index % 4 == 2)
            fixed[index] += 3;
        else if (index % 4 == 3)
            fixed[index] -= 8;
    }
}

__global__
void histogram(int* fixed, int* hist, int size)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size)
    {
        atomicAdd(&hist[fixed[index]], 1);
    }
}

__global__
void find_min(int* min, int* hist)
{
    for (int i = 0; i <= 255; ++i)
    {
        if (hist[i] > 0)
        {
            *min = hist[i];
            break;
        }
    }
}

__global__
void map_apply(int* fixed, int image_size, int* hist, int cdf_min)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < image_size)
    {
        int result = std::roundf( ((hist[fixed[index]] - cdf_min) / static_cast<float>(image_size - cdf_min)) * 255.0f);
        fixed[index] = result;
    }
}

struct is_valid
    {
        __host__ __device__
        bool operator()(const int x)
        {
            return x != -1;
        }
    };

int* fix_image_gpu(int* to_fix, int image_size, int total_size)
{
    // #1 Compact

    int* fixed;
    cudaMalloc(&fixed, sizeof(int) * total_size);


    thrust::copy_if(thrust::device, to_fix, to_fix + total_size, fixed, is_valid());


    // #2 Apply map to fix pixels
    constexpr int TILE_WIDTH = 64;
    const int gx = (image_size + TILE_WIDTH - 1) / TILE_WIDTH;

    const dim3 block(TILE_WIDTH);
    const dim3 grid(gx);

    map<<<grid, block>>>(fixed, image_size);
    cudaDeviceSynchronize();

    // #3 Histogram
    int* hist;
    cudaMalloc(&hist, sizeof(int) * 256);
    cudaMemset(hist, 0, sizeof(int) * 256);
    histogram<<<grid, block>>>(fixed, hist, image_size);
    cudaDeviceSynchronize();

    
    // Compute the inclusive sum of the histogram
    thrust::inclusive_scan(thrust::device, hist, hist + 256, hist); 

    
    // Find the first non zero value in the cumulative histogram
    int* cdf_min;
    cudaMalloc(&cdf_min, sizeof(int));
    find_min<<<1, 1>>>(cdf_min, hist);

    int h_cdf_min;
    cudaMemcpy(&h_cdf_min, cdf_min, sizeof(int), cudaMemcpyDeviceToHost);

    //std::cout << "h_cdf_min " << h_cdf_min << std::endl;


    map_apply<<<grid, block>>>(fixed, image_size, hist, h_cdf_min);
    cudaDeviceSynchronize();

    return fixed;


    /*std::transform(to_fix.buffer.data(), to_fix.buffer.data() + image_size, to_fix.buffer.data(),
        [image_size, cdf_min, &histo](int pixel)
            { 
                return std::roundf( ((histo[pixel] - cdf_min) / static_cast<float>(image_size - cdf_min)) * 255.0f);
            }
    );*/
}

/*
__global__
void map(int* fixed, int size)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size)
    {
        if (index % 4 == 0)
            fixed[index] += 1;
        else if (index % 4 == 1)
            fixed[index] -= 5;
        else if (index % 4 == 2)
            fixed[index] += 3;
        else if (index % 4 == 3)
            fixed[index] -= 8;
    }
}

__global__
void histogram(int* fixed, int* hist, int size)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size)
    {
        atomicAdd(&hist[fixed[index]], 1);
    }
}

__global__
void find_min(int* min, int* hist)
{
    for (int i = 0; i <= 255; ++i)
    {
        if (hist[i] > 0)
        {
            *min = hist[i];
            break;
        }
    }
}

__global__
void map_apply(int* fixed, int image_size, int* hist, int cdf_min)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < image_size)
    {
        int result = std::roundf( ((hist[fixed[index]] - cdf_min) / static_cast<float>(image_size - cdf_min)) * 255.0f);
        fixed[index] = result;
    }
}

struct is_valid
    {
        __host__ __device__
        bool operator()(const int x)
        {
            return x != -1;
        }
    };

int* fix_image_gpu(int* to_fix, int image_size, int total_size)
{
    // #1 Compact

    int* fixed;
    cudaMalloc(&fixed, sizeof(int) * total_size);


    thrust::copy_if(thrust::device, to_fix, to_fix + total_size, fixed, is_valid());


    // #2 Apply map to fix pixels
    constexpr int TILE_WIDTH = 64;
    const int gx = (image_size + TILE_WIDTH - 1) / TILE_WIDTH;

    const dim3 block(TILE_WIDTH);
    const dim3 grid(gx);

    map<<<grid, block>>>(fixed, image_size);
    cudaDeviceSynchronize();

    // #3 Histogram
    int* hist;
    cudaMalloc(&hist, sizeof(int) * 256);
    cudaMemset(hist, 0, sizeof(int) * 256);
    histogram<<<grid, block>>>(fixed, hist, image_size);
    cudaDeviceSynchronize();

    
    // Compute the inclusive sum of the histogram
    thrust::inclusive_scan(thrust::device, hist, hist + 256, hist); 

    
    // Find the first non zero value in the cumulative histogram
    int* cdf_min;
    cudaMalloc(&cdf_min, sizeof(int));
    find_min<<<1, 1>>>(cdf_min, hist);

    int h_cdf_min;
    cudaMemcpy(&h_cdf_min, cdf_min, sizeof(int), cudaMemcpyDeviceToHost);

    map_apply<<<grid, block>>>(fixed, image_size, hist, h_cdf_min);
    cudaDeviceSynchronize();

    return fixed;
}*/

int main()
{
    /*Image in_cpu("../../Unequalized.pgm");
    generate_broken_image(in_cpu);
    
    int* in_gpu;
    cudaMalloc(&in_gpu, sizeof(int) * in_cpu.buffer.size());
    cudaMemcpy(in_gpu, in_cpu.buffer.data(), sizeof(int) * in_cpu.buffer.size(), cudaMemcpyHostToDevice);
    
    fix_image_cpu(in_cpu);
    int* fixed_gpu = fix_image_gpu(in_gpu, in_cpu.width * in_cpu.height, in_cpu.buffer.size());


    std::vector<int> out_gpu(in_cpu.buffer.size(), -5);
    cudaMemcpy(out_gpu.data(), fixed_gpu, sizeof(int) * in_cpu.width * in_cpu.height, cudaMemcpyDeviceToHost);
    
    Image in_gpu_wrapper(std::move(out_gpu), in_cpu.height, in_cpu.width);

    in_gpu_wrapper.write("Test_gpu.pgm");
    in_cpu.write("Test_cpu.pgm");*/


    using recursive_directory_iterator = std::filesystem::recursive_directory_iterator;
    int i = 0;
    for (const auto& dirEntry : recursive_directory_iterator("../good_images_pgm"))
    {
        Image in_cpu(dirEntry.path());
        generate_broken_image(in_cpu);

        std::ostringstream oss;
        oss << "../corrupted_images/Broken#" << i << ".pgm";
        std::string str = oss.str();

        in_cpu.write_bad(str);
        ++i;
    }
    return 0;
}