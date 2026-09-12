#pragma once

#include <cmath>
#include <tuple>


template <typename Tuple>
constexpr auto tuple_length(Tuple)
{
    return std::tuple_size_v<Tuple>;
}

bool parse_arguments(int argc, char* argv[])
{
    bool bench_nsight = false;
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

    return bench_nsight;
}