#pragma once

#include <string>
#include <fstream>
#include <iostream>
#include <cstring>
#include <vector>

struct Image
{
    Image(const std::string& filepath)
    {
        std::ifstream infile(filepath, std::ifstream::binary);

        if (!infile.is_open()) 
            throw std::runtime_error("Failed to open");

        std::string magic;
        infile >> magic;
        infile.seekg(1, infile.cur);
        char c;
        infile.get(c);
        if (c == '#')
        {
            while (c != '\n')
                infile.get(c);
        }
        else
            infile.seekg(-1, infile.cur);
        
        int max;
        infile >> width >> height >> max;
        if (max != 255)
            throw std::runtime_error("Bad max value");

        buffer.reserve(width * height);

        if (magic == "P5")
        {
            infile.seekg(1, infile.cur);
            for (int i = 0; i < width * height; ++i)
            {
                uint8_t pixel_char;
                infile >> pixel_char;
                buffer[i] = pixel_char;
            }
        }
        else
            throw std::runtime_error("Bad PPM value");
    }

    Image(std::vector<int>&& b, int h, int w): buffer(std::move(b)), height(h), width(w)
    {
    }

    void write(const std::string & filepath) const
    {
        std::ofstream outfile(filepath, std::ofstream::binary);
        if (outfile.fail())
            throw std::runtime_error("Failed to open");
        outfile << "P5" << "\n" << width << " " << height << "\n" << 255 << "\n";

        for (int i = 0; i < height; ++i)
        {
            for (int j = 0; j < width; ++j)
            {
                int val = buffer[i * width + j];
                if (val < 0 || val > 255)
                {
                    std::cout << std::endl;
                    std::cout << i * width + j << " " << val << std::endl;
                    throw std::runtime_error("Invalid image format");
                }
                outfile << static_cast<uint8_t>(val);
            }
        }
    }

    void write_bad(const std::string & filepath) const
    {
        std::ofstream outfile(filepath, std::ofstream::binary);
        if (outfile.fail())
            throw std::runtime_error("Failed to open");
        int a = -1;
        outfile << "P?" << "\n" << width << " " << height << "\n" << a << "\n";

        for (int i = 0; i < buffer.size(); ++i)
        {
            int val = buffer[i];
            outfile << val;
            outfile << ";";
        }
    }

    std::vector<int> buffer;
    int height = -1;
    int width = -1;
    int total = -1;
};