#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>
#include "cuda.h"
#include "cuda_runtime.h"

int
main(int argc, char **argv) {
    std::string filename = argv[1];
    std::ifstream f;
    size_t data_size;
    f.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    f.open(filename, std::ios::in | std::ios::binary);
    f.seekg(0, std::ios::end);
    data_size = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<char> buffer(data_size);
    uint8_t *ptr_h = (uint8_t*) buffer.data();

    auto start = std::chrono::high_resolution_clock::now();
    f.read(reinterpret_cast<char *>(ptr_h), data_size);
    f.close();
    auto end = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    double throughput = (data_size / (1024 * 1024 * 1024)) / (duration/1000.0);
    std::cout << "Benchmark: F2H read in " << duration << " msec at "
              << throughput << " GB/s" << std::endl;

    uint8_t *ptr_d;
    cudaHostRegister(ptr_h, data_size, cudaHostRegisterDefault);
    cudaMalloc((void **)&ptr_d, data_size);
    start = std::chrono::high_resolution_clock::now();
    cudaMemcpy(ptr_d, ptr_h, data_size, cudaMemcpyHostToDevice);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    throughput = (data_size / (1024 * 1024 * 1024)) / (duration/1000.0);
    std::cout << "Benchmark: H2D cpy of all data in " << duration << " msec at "
              << throughput << " GB/s" << std::endl;
}