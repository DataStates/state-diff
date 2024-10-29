#include "io_uring_stream.hpp"
#include "statediff.hpp"
#include <cereal/archives/binary.hpp>
#include <chrono>
#include <fstream>
#include <iostream>
#include <omp.h>
#include <random>
#include <string>
#include <vector>
#include <numeric>

bool
write_file(const std::string &fn, uint8_t *buffer, size_t size) {
    int fd = open(fn.c_str(), O_CREAT | O_TRUNC | O_WRONLY, 0644);
    if (fd == -1) {
        FATAL("cannot open " << fn << ", error = " << strerror(errno));
        return false;
    }
    size_t transferred = 0, remaining = size;
    while (remaining > 0) {
        size_t ret = write(fd, buffer + transferred, remaining);
        remaining -= ret;
        transferred += ret;
    }
    close(fd);
    return true;
}

int
main(int argc, char **argv) {

    if (argc != 5) {
        std::cerr << "Usage: ./benchmark_tree_creation <data_size> "
                     "<dev_buf_size> <chunk_size>"
                     "<csv_fn>"
                  << std::endl;
        return 1;
    }

    // Parse input arguments
    uint32_t data_size = static_cast<uint32_t>(std::stoul(argv[1]));
    uint32_t dev_buf_size = static_cast<uint32_t>(std::stoul(argv[2]));
    uint32_t chunk_size = static_cast<uint32_t>(std::stoul(argv[3]));
    std::string csv_fn = argv[4];

    std::string fname = "checkpoint.dat";
    float error_tolerance = 1e-4;
    bool fuzzy_hash = true;
    char dtype = 'f';
    int root_level = 13;
    int KB = 1024;
    int MB = 1024 * KB;
    int GB = 1024 * MB;

    int num_chunks = data_size / chunk_size;
    Kokkos::initialize(argc, argv);
    {
        // Create synthetic datasets
        float max_float = 100.0;
        float min_float = 0.0;
        int seed = 0x123;
        int data_len = data_size / sizeof(float);
        std::vector<float> run_data(data_len);
#pragma omp parallel
        {
            std::mt19937 prng(seed + omp_get_thread_num());
            std::uniform_real_distribution<float> prng_dist(min_float,
                                                            max_float);
#pragma omp for
            for (int i = 0; i < data_len; ++i) {
                run_data[i] = prng_dist(prng);
            }
        }

        // save checkpoint for offline tree cretion and comparison
        write_file(fname, (uint8_t *)run_data.data(), data_size);
        std::cout << "EXEC STATE:: File saved" << std::endl;

        io_uring_stream_t<float> reader(fname, chunk_size / sizeof(float));
        state_diff::client_t<float, io_uring_stream_t> client(
            1, reader, data_size, error_tolerance, dtype, chunk_size,
            root_level, fuzzy_hash, dev_buf_size);

        auto start_create = std::chrono::high_resolution_clock::now();
        client.create(run_data);
        auto end_create = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> create_time = end_create - start_create;
        std::vector<double> timings = client.get_create_time();
        auto measured_create_time = std::reduce(timings.begin(), timings.end());
        std::cout << "Tree created with data_size=" << data_size / GB
                  << "GB, dev_buf_size=" << dev_buf_size / MB
                  << "MB, chunk_size=" << chunk_size / KB << "KB with "
                  << num_chunks << " chunks in " << create_time.count()
                  << " sec (Measured = " << measured_create_time/1000.0 << " sec), throughput: "
                  << (data_size / create_time.count()) / (1024 * 1024 * 1024)
                  << " GB/s" << std::endl;

        // Append results to the CSV file
        std::ofstream csv_file(csv_fn, std::ios_base::app);
        if (csv_file.is_open()) {
            csv_file << data_size << "," << dev_buf_size << "," << chunk_size
                     << "," << timings[0] << "," << timings[1] << ","
                     << timings[2] << "," << timings[3] << "\n";
            csv_file.close();
        } else {
            std::cerr << "Unable to open the output CSV file." << std::endl;
        }
    }
    Kokkos::finalize();
    return 0;
}
