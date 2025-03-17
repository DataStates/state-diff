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

using Duration = std::chrono::duration<double>;

template <typename DataType>
int
validate(DataType *data, DataType *loader_out, size_t data_len) {
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < data_len; i++) {
        // std::cout << "Loader at index " << i << " = " << loader_out[i] << "
        // vs real val = " << data[i] << "\n";
        if (data[i] != loader_out[i]) {
            std::cout << "Mismatch at index " << i << ": ifstream = " << data[i]
                      << ", loader = " << loader_out[i] << "\n";
            return -1;
        }
    }
    std::cout << "Bingo! No Mismatch \n";
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "CPU Validation completed in "
              << std::chrono::duration_cast<Duration>(end - start).count()
              << " seconds." << std::endl;
    return 0;
}

template <typename DataType>
void
read_ifstream(std::string filename, DataType *data_veri_h, size_t num_elements,
              std::streampos start_offset = 0) {
    std::ifstream f;
    f.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    try {
        f.open(filename, std::ios::in | std::ios::binary);
        f.seekg(start_offset);
        f.read(reinterpret_cast<char *>(data_veri_h),
               num_elements * sizeof(DataType));
        f.close();
    } catch (const std::ifstream::failure &e) {
        std::cerr << "Error reading file: " << e.what() << std::endl;
    }
}

int
main(int argc, char **argv) {
    std::string filename = argv[1];
    std::string dtype = argv[2];

    int MB = 1024 * 1024;
    size_t read_size = 128 * MB;
    TransferType trans_type = TransferType::FileToHost;

    // Create reader
    liburing_io_reader_t uring_reader(filename);
    size_t data_size = uring_reader.size();   // size in bytes
    printf("Data size = %zu\n", data_size);
    std::vector<uint8_t> data_h(data_size);
    std::vector<uint8_t> data_veri_h(data_size);

    // create loader
    // data_loader_t data_loader(host_cache_size);
    data_loader_t data_loader;
    int ld = data_loader.file_load(uring_reader, read_size, trans_type);

    // start loading
    size_t read_bytes = 0;
    size_t i = 0;

    while (read_bytes < data_size) {
        // auto next_batch = data_loader.next(ld, trans_type);
        // uint8_t *data_ptr = next_batch.first;
        // size_t ready_size = next_batch.second;
        next_batch_t batch = data_loader.next(ld, trans_type);
        uint8_t *data_ptr = batch.ptr;
        size_t ready_size = batch.size;
        printf("Client - Loaded batch %zu of size %zu bytes\n", ++i,
               ready_size);
        std::memcpy(data_h.data() + read_bytes, data_ptr, ready_size);
        read_bytes += ready_size;
    }
    printf("Client - Loaded %zu batches (%zu bytes) out of %zu bytes of data\n",
           i, read_bytes, data_size);

    printf("Validating the results for correctness\n");
    try {
        if (dtype.compare("-f") == 0) {
            read_ifstream<float>(filename, (float *)data_veri_h.data(),
                                 data_size / sizeof(float));
        } else {
            read_ifstream<uint32_t>(filename, (uint32_t *)data_veri_h.data(),
                                    data_size / sizeof(uint32_t));
        }
        std::cout << "Validation data read successfully." << std::endl;
    } catch (const std::ifstream::failure &e) {
        std::cerr << "Exception occurred while reading file: " << e.what()
                  << std::endl;
    }
    if (dtype.compare("-f") == 0) {
        validate<float>((float *)data_veri_h.data(), (float *)data_h.data(),
                        data_size / sizeof(float));
    } else {
        validate<uint32_t>((uint32_t *)data_veri_h.data(),
                           (uint32_t *)data_h.data(),
                           data_size / sizeof(uint32_t));
    }
    return 0;
}