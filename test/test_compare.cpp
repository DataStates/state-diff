#include "io_uring_stream.hpp"
#include "statediff.hpp"
#include <chrono>
#include <fstream>
#include <iostream>
#include <omp.h>
#include <random>
#include <string>
#include <vector>

bool
write_file(const std::string &fn, uint8_t *buffer, size_t size) {
    int fd = open(fn.c_str(), O_CREAT | O_TRUNC | O_WRONLY, 0644);
    if (fd == -1) {
	std::cout << "cannot open " << fn << ", error = " << strerror(errno) << std::endl;
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

    int test_status = 0;

    // Define the parameters
    // maximum floating-point (FP) value in synthetic data
    float max_float = 100.0;
    // minimum FP value in synthetic data
    float min_float = 0.0;
    // size in bytes of the synthetic data (1GB)
    // size_t data_size = 1024 * 1024 * 1024;
    size_t data_size = 1024 * 1024;   // 1MB
    // Application error tolerance
    float error_tolerance = 1e-4;
    // Target chunk size. This example uses 16 bytes
    size_t chunk_size = 512;
    // Use our rounding hash algorithm or exact hash.
    bool fuzzy_hash = true;
    char dtype = 'f';   // float
    // Random number seed to generate the synthetic data
    int seed = 0x123;
    // builds the tree from leaves to root level, can be 12 or 13.
    int root_level = 1;
    std::string fn_0 = "data_0.dat", fn_1 = "data_1.dat", fn_2 = "data_2.dat";

    int num_chunks = data_size / chunk_size;
    std::cout << "Nunber of leaf nodes = " << num_chunks << std::endl;

    Kokkos::initialize(argc, argv);
    {
        // Create synthetic datasets
        size_t data_len = data_size / sizeof(float);
        std::vector<float> run_0_data(data_len), run_1_data(data_len),
            run_2_data(data_len);
#pragma omp parallel
        {
            std::mt19937 prng(seed + omp_get_thread_num());
            std::uniform_real_distribution<float> prng_dist(min_float,
                                                            max_float);
            std::uniform_real_distribution<float> in_error_dist(
                0.0, 0.5 * error_tolerance);
            std::uniform_real_distribution<float> outof_error_dist(
                1.5 * error_tolerance, 2 * error_tolerance);
#pragma omp for
            for (size_t i = 0; i < data_len; ++i) {
                run_0_data[i] = prng_dist(prng);
                run_1_data[i] = run_0_data[i] + in_error_dist(prng);
                run_2_data[i] = run_0_data[i] + outof_error_dist(prng);
            }
        }

        // save checkpoint for offline tree cretion and comparison
        write_file(fn_0, (uint8_t *)run_0_data.data(), data_size);
        write_file(fn_1, (uint8_t *)run_1_data.data(), data_size);
        write_file(fn_2, (uint8_t *)run_2_data.data(), data_size);
        std::cout << "EXEC STATE:: Files saved" << std::endl;

        // read data, build tree and save
        io_uring_stream_t<float> reader_0(fn_0, chunk_size / sizeof(float));
        io_uring_stream_t<float> reader_1(fn_1, chunk_size / sizeof(float));
        io_uring_stream_t<float> reader_2(fn_2, chunk_size / sizeof(float));

        state_diff::client_t<float, io_uring_stream_t> client_0(
            0, reader_0, data_size, error_tolerance, dtype, chunk_size,
            root_level, fuzzy_hash);
        state_diff::client_t<float, io_uring_stream_t> client_1(
            1, reader_1, data_size, error_tolerance, dtype, chunk_size,
            root_level, fuzzy_hash);
        state_diff::client_t<float, io_uring_stream_t> client_2(
            2, reader_2, data_size, error_tolerance, dtype, chunk_size,
            root_level, fuzzy_hash);
        client_0.create(run_0_data);
        client_1.create(run_1_data);
        client_2.create(run_2_data);
        std::cout << "EXEC STATE:: Trees created" << std::endl;

        // compare the checkpoints one-to-one
        client_0.compare_with(client_0);
        std::cout << "EXEC STATE:: (0-0) Comparison completed" << std::endl;
        std::cout << "(0-0) Number of mismatch = " << client_1.get_num_changes()
                  << std::endl;
        if (client_0.get_num_changes() != 0) {
            test_status = -1;
        }

        client_1.compare_with(client_0);
        std::cout << "EXEC STATE:: (1-0) Comparison completed" << std::endl;
        std::cout << "(1-0) Number of mismatch = " << client_1.get_num_changes()
                  << std::endl;
        if (client_1.get_num_changes() != 0) {
            test_status = -1;
        }

        client_2.compare_with(client_0);
        std::cout << "EXEC STATE:: (2-0) comparison completed" << std::endl;
        std::cout << "(2-0) Number of mismatch = " << client_2.get_num_changes()
                  << std::endl;
        if (client_2.get_num_changes() != data_len) {
            test_status = -1;
        }
    }
    Kokkos::finalize();
    return test_status;
}
