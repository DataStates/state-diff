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
compare_blocks_d(uint32_t *data0, uint32_t *data1, size_t n_ele_batch,
                 uint32_t *diff_local) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_ele_batch) {
        if (data0[idx] != data1[idx])
            atomicAdd(diff_local, 1);
    }
}

size_t
compare_blocks_h(uint32_t *data0, uint32_t *data1, size_t n_ele_batch) {
    size_t diff = 0;
    for (size_t idx = 0; idx < n_ele_batch; idx++) {
        if (data0[idx] != data1[idx])
            diff += 1;
    }
    return diff;
}

void
compare_data(uint32_t *data_ptr0, uint32_t *data_ptr1, size_t n_ele_batch,
             size_t proc_ele) {
    cudaPointerAttributes attributes;
    cudaError_t err = cudaPointerGetAttributes(&attributes, data_ptr0);
    uint32_t h_diff_local = 0;
    if (err == cudaSuccess && attributes.type == cudaMemoryTypeDevice) {
        int tpb = THREAD_PER_BLOCK;
        int num_blocks = (n_ele_batch + tpb - 1) / tpb;
        // pass variable and copy result to host
        uint32_t *d_diff_local;
        cudaMalloc(&d_diff_local, sizeof(uint32_t));
        cudaMemset(d_diff_local, 0, sizeof(uint32_t));
        compare_blocks_d<<<num_blocks, tpb, 0>>>(data_ptr0, data_ptr1,
                                                 n_ele_batch, d_diff_local);
        cudaDeviceSynchronize();

        cudaMemcpy(&h_diff_local, d_diff_local, sizeof(uint32_t),
                   cudaMemcpyDeviceToHost);
        cudaFree(d_diff_local);
    } else {
        h_diff_local = compare_blocks_h(data_ptr0, data_ptr1, n_ele_batch);
    }
    if (h_diff_local > 0) {
        std::cout << "Mismatch detected" << std::endl;
    }
}

int
validate(uint32_t *data0, uint32_t *data1, size_t data_len) {

    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < data_len; i++) {
        if (data0[i] != data1[i]) {
            std::cout << "Mismatch at index " << i << ": File0 = " << data0[i]
                      << ", File1 = " << data1[i] << "\n";
            return -1;
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "CPU Validation completed in "
              << std::chrono::duration_cast<Duration>(end - start).count()
              << " seconds." << std::endl;
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
    std::string filename0 = argv[3];
    std::string filename1 = argv[4];
    size_t seg_size = std::stol(argv[5]);
    size_t batch_size = std::stol(argv[6]);   // segments per batch

    int MB = 1024 * 1024;
    size_t start_foffset = 0;
    // TransferType trans_type = TransferType::FileToHost;
    TransferType trans_type = TransferType::FileToDevice;

    // Create reader
    liburing_io_reader_t uring_reader0(filename0);
    liburing_io_reader_t uring_reader1(filename1);
    size_t data_size = uring_reader0.size();   // size in bytes
    size_t num_ele_total = data_size / sizeof(uint32_t);

    printf("Client - Configuration: Filename0 = %s; Filename1 = %s; Data Size "
           "= %zu MB; Host "
           "Cache = %zu MB; "
           "Device Cache = %zu MB;\n\tBatch Size = %zu segs; Seg "
           "Size = %zu MB;\n\tNEleTot = %zu\n",
           filename0.c_str(), filename1.c_str(), data_size / MB,
           host_cache_size / MB, dev_cache_size / MB, batch_size, seg_size / MB,
           num_ele_total);

    // create loader
    data_loader_t data_loader(host_cache_size, dev_cache_size);

    auto start = std::chrono::high_resolution_clock::now();

    // start async data loading from file to device
    auto st_init = std::chrono::high_resolution_clock::now();
    int ld0 = data_loader.file_load(uring_reader0, start_foffset, seg_size,
                                    batch_size, trans_type);
    int ld1 = data_loader.file_load(uring_reader1, start_foffset, seg_size,
                                    batch_size, trans_type);
    auto nd_init = std::chrono::high_resolution_clock::now();

    // start computation
    double load_time = 0;
    double proc_time = 0;
    size_t proc_elements = 0;
    size_t i = 0;

    while (proc_elements < num_ele_total) {
        printf("Client - Processing batch %zu\n", ++i);
        auto st_ld = std::chrono::high_resolution_clock::now();
        auto next_batch0 = data_loader.next(ld0, trans_type);
        auto next_batch1 = data_loader.next(ld1, trans_type);
        uint32_t *data_ptr0 = (uint32_t *)next_batch0.first;
        uint32_t *data_ptr1 = (uint32_t *)next_batch1.first;
        size_t ready_size = next_batch0.second;
        size_t num_ele_batch = ready_size / sizeof(uint32_t);
        auto nd_ld = std::chrono::high_resolution_clock::now();
        load_time +=
            std::chrono::duration_cast<Duration>(nd_ld - st_ld).count();

        auto st_cp = std::chrono::high_resolution_clock::now();
        compare_data(data_ptr0, data_ptr1, num_ele_batch, proc_elements);
        auto nd_cp = std::chrono::high_resolution_clock::now();
        proc_time +=
            std::chrono::duration_cast<Duration>(nd_cp - st_cp).count();
        proc_elements += num_ele_batch;
    }
    printf("Client - Processed all batches\n");
    auto end = std::chrono::high_resolution_clock::now();

    printf("Validating the results for correctness\n");
    std::vector<uint32_t> data_veri_0(num_ele_total);
    std::vector<uint32_t> data_veri_1(num_ele_total);
    read_host_verify(filename0, data_veri_0.data(), num_ele_total);
    read_host_verify(filename1, data_veri_1.data(), num_ele_total);
    validate(data_veri_0.data(), data_veri_1.data(), num_ele_total);

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