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
    assert(data_len == loader_out.size());
    for (size_t i = 0; i < data_len; i++) {
        // std::cout << "Loader at index " << i << " = " << loader_out[i] << " vs real val = " << data[i] << "\n";
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
read_ifstream(std::string filename, DataType *data_veri_h,
                 size_t num_elements, std::streampos start_offset = 0) {
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
    size_t host_cache_size = std::stol(argv[1]);
    size_t dev_cache_size = std::stol(argv[2]);
    std::string filename = argv[3];
    std::string dtype = argv[4];

    using DataType = uint32_t;  // Default type
    if (dtype.compare("-f") == 0) {
        using DataType = float;  // If dtype is not "-f", set DataType to uint32_t
    }
    
    int MB = 1024 * 1024;
    size_t read_size = 128*MB;
    size_t start_foffset = 0;
    TransferType trans_type = TransferType::FileToHost;

    // Create reader
    liburing_io_reader_t uring_reader(filename);
    size_t data_size = uring_reader.size();   // size in bytes
    printf("Data size = %zu\n", data_size);
    std::vector<uint8_t> data_h(data_size);
    std::vector<uint8_t> data_veri_h(data_size);

    // create loader
    data_loader_t data_loader(host_cache_size, dev_cache_size);
    int ld = data_loader.file_load(uring_reader, start_foffset, read_size, trans_type);
    auto nd_init = std::chrono::high_resolution_clock::now();

    // start computation
    double load_time = 0;
    double proc_time = 0;
    size_t proc_elements = 0;
    size_t i = 0;

    while (proc_elements < data_size) {
        printf("Client - Processing batch %zu\n", ++i);
        auto next_batch = data_loader.next(ld, trans_type);
        uint8_t *data_ptr = next_batch.first;
        size_t ready_size = next_batch.second;
        // std::vector<uint8_t> ptr(data_size);
        // uint8_t *data_ptr = ptr.data();
        // data_loader.next(ld, data_ptr);
        // size_t ready_size = data_size;
        std::memcpy(data_h.data()+proc_elements, data_ptr, ready_size);
        proc_elements += ready_size;
    }
    printf("Client - Processed all batches\n");
    auto end = std::chrono::high_resolution_clock::now();

    printf("Validating the results for correctness\n");
    try {
        read_ifstream<DataType>(filename, (DataType *)data_veri_h.data(), data_size/sizeof(DataType));
        std::cout << "Validation data read successfully." << std::endl;
    } catch (const std::ifstream::failure &e) {
        std::cerr << "Exception occurred while reading file: " << e.what()
                  << std::endl;
    }
    validate<DataType>((DataType *)data_veri_h.data(), (DataType *)data_h.data(), data_size/sizeof(DataType));
    return 0;
}