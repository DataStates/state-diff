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
validate(DataType *data0, DataType *data1, size_t data_len) {
    for (size_t i = 0; i < data_len; i++) {
        // std::cout << "Loader at index " << i << " = " << loader_out[i] << "
        // vs real val = " << data[i] << "\n";
        if (data0[i] != data1[i]) {
            std::cout << "Mismatch at index " << i << ": File0 = " << data0[i]
                      << ", File1 = " << data1[i] << "\n";
            return -1;
        }
    }
    std::cout << "Bingo! No Mismatch \n";
    return 0;
}

int
main(int argc, char **argv) {
    std::string filename0 = argv[1];
    std::string filename1 = argv[2];
    std::string dtype = argv[3];
    int offset_pct = std::stoi(argv[4]);

    size_t chunk_size = 8196;
    TransferType trans_type = TransferType::FileToHost;

    // Create reader
    liburing_io_reader_t uring_reader0(filename0);
    liburing_io_reader_t uring_reader1(filename1);
    size_t data_size = uring_reader0.size();   // size in bytes
    assert(data_size == uring_reader1.size());

    // Generate offsets
    size_t num_chunks = data_size / chunk_size;
    if (num_chunks * chunk_size < data_size)
        num_chunks += 1;

    size_t num_offsets = static_cast<size_t>(num_chunks * (offset_pct / 100.0));
    std::unordered_set<size_t> unique_offsets;
    std::mt19937 gen(0);
    std::uniform_int_distribution<size_t> int_dis(0, num_chunks - 1);

    while (unique_offsets.size() < num_offsets) {
        unique_offsets.insert(int_dis(gen));
    }

    std::vector<size_t> chunk_offsets(unique_offsets.begin(),
                                      unique_offsets.end());
    std::sort(chunk_offsets.begin(), chunk_offsets.end());
    printf("Generated %zu offsets (%d percent of %zu)\n", num_offsets,
           offset_pct, num_chunks);

    // create loader
    // data_loader_t data_loader(host_cache_size);
    data_loader_t data_loader;
    std::pair<int, std::vector<size_t>> lid_offst_pair =
        data_loader.file_load(uring_reader0, uring_reader1, chunk_offsets,
                              chunk_size, trans_type, 2, 134217728);
    int ld = lid_offst_pair.first;

    // load data
    size_t total_read_size = static_cast<size_t>(num_offsets * chunk_size);
    printf("Data size = %zu | Total Read Size = %zu\n", data_size,
           total_read_size);
    std::vector<uint8_t> data_0(total_read_size);
    std::vector<uint8_t> data_1(total_read_size);
    size_t read_bytes = 0;
    size_t i = 0;
    while (read_bytes < total_read_size) {
        next_batch_t batch = data_loader.next(ld, trans_type);
        uint8_t *data_ptr0 = batch.ptr;
        size_t ready_size = batch.size / 2;

        // auto next_batch = data_loader.next(ld, trans_type);
        if (data_ptr0 == nullptr) {
            printf("Client - Received a null pointer. Exiting loop.\n");
            break;
        }
        if (ready_size == 0) {
            printf("Client - Received an empty batch. Exiting loop.\n");
            break;
        }
        // uint8_t *data_ptr0 = next_batch.first;
        // a batch for the two file loader has a batch size of 2, i.e., two
        // segments per batch each corresponding to a file
        // size_t ready_size = next_batch.second / 2;
        uint8_t *data_ptr1 = data_ptr0 + ready_size;
        // copy loaded data into a buffer reserved for comparison of the files
        std::memcpy(data_0.data() + read_bytes, data_ptr0, ready_size);
        std::memcpy(data_1.data() + read_bytes, data_ptr1, ready_size);
        read_bytes += ready_size;
        printf("Client - Loaded batch %zu of size %zu bytes\n", ++i,
               batch.size);
    }
    printf("Client - Loaded %zu batches (%zu bytes) out of %zu bytes of data\n",
           i, read_bytes, data_size);

    printf("Validating the results for correctness\n");
    if (dtype.compare("-f") == 0) {
        validate<float>((float *)data_0.data(), (float *)data_1.data(),
                        total_read_size / sizeof(float));
    } else {
        validate<uint32_t>((uint32_t *)data_0.data(), (uint32_t *)data_1.data(),
                           total_read_size / sizeof(uint32_t));
    }
    return 0;
}