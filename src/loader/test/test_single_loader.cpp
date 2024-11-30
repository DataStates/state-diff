#include "data_loader.hpp"
#include "liburing_reader.hpp"
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <typeinfo>
#include <vector>

#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_timer.hpp"

#define THREAD_PER_BLOCK 256

__global__ void
mult_by_two_d(uint32_t *data, size_t n_ele_batch) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_ele_batch)
        data[idx] *= 2;
}

void
mult_by_two_h(uint32_t *data, size_t n_ele_batch) {
    for (size_t idx = 0; idx < n_ele_batch; idx++)
        data[idx] *= 2;
}

void
process_data(uint32_t *data_ptr, uint32_t *data_h_ptr, size_t n_ele_batch,
             size_t proc_ele) {
    cudaPointerAttributes attributes;
    cudaError_t err = cudaPointerGetAttributes(&attributes, data_ptr);
    size_t cpy_size = n_ele_batch * sizeof(uint32_t);
    uint32_t *dst_ptr = data_h_ptr + proc_ele;
    if (err == cudaSuccess && attributes.type == cudaMemoryTypeDevice) {
        int tpb = THREAD_PER_BLOCK;
        int num_blocks = (n_ele_batch + tpb - 1) / tpb;
        mult_by_two_d<<<num_blocks, tpb, 0>>>(data_ptr, n_ele_batch);
        cudaMemcpy(dst_ptr, data_ptr, cpy_size, cudaMemcpyDeviceToHost);
    } else {
        mult_by_two_h(data_ptr, n_ele_batch);
        std::memcpy(dst_ptr, data_ptr, cpy_size);
    }
}

int
validate(uint32_t *data, uint32_t *dev_out, size_t data_len) {

    auto start = std::chrono::high_resolution_clock::now();
    assert(data_len == dev_output.size());
    for (size_t idx = 0; idx < data_len; idx++)
        data[idx] *= 2;
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "CPU Validation completed in "
              << std::chrono::duration_cast<Duration>(end - start).count()
              << " seconds." << std::endl;

    for (size_t i = 0; i < data_len; i++) {
        if (data[i] != dev_out[i]) {
            std::cout << "Mismatch at index " << i << ": HOST = " << data[i]
                      << ", DEV = " << dev_out[i] << "\n";
            return -1;
        }
    }
    return 0;
}

void
read_host_verify(std::string filename, uint32_t *data_veri_h,
                 size_t num_elements, std::streampos start_offset = 0) {
    std::ifstream f;
    f.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    try {
        f.open(filename, std::ios::in | std::ios::binary);
        f.seekg(start_offset);
        f.read(reinterpret_cast<char *>(data_veri_h),
               num_elements * sizeof(uint32_t));
        f.close();
    } catch (const std::ifstream::failure &e) {
        std::cerr << "Error reading file: " << e.what() << std::endl;
    }
}

int
main(int argc, char **argv) {
    size_t host_cache_size = std::stol(argv[1]);
    size_t dev_cache_size = std::stol(argv[2]);
    std::string filename = argv[3];
    size_t seg_size = std::stol(argv[4]);
    size_t batch_size = std::stol(argv[5]);   // segments per batch

    int MB = 1024 * 1024;
    size_t start_foffset = 0;
    // TransferType trans_type = TransferType::FileToHost;
    TransferType trans_type = TransferType::FileToDevice;

    // Create reader
    liburing_io_reader_t uring_reader(filename);
    size_t data_size = uring_reader.size();   // size in bytes
    size_t num_ele_total = data_size / sizeof(uint32_t);
    std::vector<uint32_t> data_h(num_ele_total);
    std::vector<uint32_t> data_veri_h(num_ele_total);
    uint32_t *data_h_ptr = data_h.data();
    cudaHostRegister(data_h_ptr, data_size, cudaHostRegisterDefault);

    printf("Client - Configuration: Filename = %s; Data Size = %zu MB; Host "
           "Cache = %zu MB; "
           "Device Cache = %zu MB;\n\tBatch Size = %zu segs; Seg "
           "Size = %zu MB;\n\tNEleTot = %zu\n",
           filename.c_str(), data_size / MB, host_cache_size / MB,
           dev_cache_size / MB, batch_size, seg_size / MB, num_ele_total);

    // create loader
    data_loader_t data_loader(host_cache_size, dev_cache_size);

    auto start = std::chrono::high_resolution_clock::now();

    // start async data loading from file to device
    auto st_init = std::chrono::high_resolution_clock::now();
    data_loader.file_load(uring_reader, start_foffset, seg_size, batch_size,
                          trans_type);
    auto nd_init = std::chrono::high_resolution_clock::now();

    // start computation
    double load_time = 0;
    double proc_time = 0;
    size_t proc_elements = 0;
    size_t i = 0;

    while (proc_elements < num_ele_total) {
        printf("Client - Processing batch %zu\n", ++i);
        auto st_ld = std::chrono::high_resolution_clock::now();
        auto next_batch = data_loader.next(trans_type);
        uint32_t *data_ptr = (uint32_t *)next_batch.first;
        size_t ready_size = next_batch.second;
        size_t num_ele_batch = ready_size / sizeof(uint32_t);
        auto nd_ld = std::chrono::high_resolution_clock::now();
        load_time +=
            std::chrono::duration_cast<Duration>(nd_ld - st_ld).count();

        auto st_cp = std::chrono::high_resolution_clock::now();
        process_data(data_ptr, data_h_ptr, num_ele_batch, proc_elements);
        auto nd_cp = std::chrono::high_resolution_clock::now();
        proc_time +=
            std::chrono::duration_cast<Duration>(nd_cp - st_cp).count();
        proc_elements += num_ele_batch;
    }
    printf("Client - Processed all batches\n");
    auto end = std::chrono::high_resolution_clock::now();

    printf("Validating the results for correctness\n");
    try {
        read_host_verify(filename, data_veri_h.data(), num_ele_total);
        std::cout << "Validation data read successfully." << std::endl;
    } catch (const std::ifstream::failure &e) {
        std::cerr << "Exception occurred while reading file: " << e.what()
                  << std::endl;
    }
    validate(data_veri_h.data(), data_h.data(), num_ele_total);

    std::cout << "Stats: \nInit time = "
              << std::chrono::duration_cast<Duration>(nd_init - st_init).count()
              << " s"
              << "\nLoad throughput = "
              << (data_size / (1024 * 1024 * 1024)) / load_time << " GB/s"
              << "\nCompute throughput = "
              << (data_size / (1024 * 1024 * 1024)) / proc_time << " GB/s"
              << " \nTotal Exec time = "
              << std::chrono::duration_cast<Duration>(end - start).count()
              << " s" << std::endl;
    return 0;
}