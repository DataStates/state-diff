#include "io_uring_stream.hpp"
#include "statediff.hpp"
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
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
        FATAL("cannot open " << fn << ", error = " << strerror(errno));
        return false;
    }
    size_t transferred = 0, remaining = size;
    while (remaining > 0) {
        size_t ret = write(fd, buffer + transferred, remaining);
        if (ret < 0)
            FATAL("cannot write " << size << " bytes to " << fn
                                  << " , error = " << std::strerror(errno));
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
    int data_size = 1024 * 1024 * 1024;
    // Application error tolerance
    float error_tolerance = 1e-4;
    // Target chunk size. This example uses 16 bytes
    int chunk_size = 512;
    // Use our rounding hash algorithm or exact hash.
    bool fuzzy_hash = true;
    char dtype = 'f';   // float
    // Random number seed to generate the synthetic data
    int seed = 0x123;
    // builds the tree from leaves to root level, can be 12 or 13.
    int root_level = 1;
    std::string fname = "checkpoint.dat";
    std::string metadata_fn = "checkpoint.tree";

    int num_chunks = data_size / chunk_size;
    std::cout << "Nunber of leaf nodes = " << num_chunks << std::endl;

    Kokkos::initialize(argc, argv);
    {
        // Create synthetic datasets
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

        // read data, build tree and save
        // liburing_io_reader_t reader(fname);
        io_uring_stream_t<float> reader(fname, chunk_size / sizeof(float));
        state_diff::client_t<float, io_uring_stream_t> client(
            1, reader, data_size, error_tolerance, dtype, chunk_size,
            root_level, fuzzy_hash);
        client.create(run_data);
        auto start_serialize = std::chrono::high_resolution_clock::now();
        {
            std::ofstream ofs(metadata_fn, std::ios::binary);
            boost::archive::binary_oarchive oa(ofs);
            oa << client;
            ofs.close();
        }
        auto end_serialize = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> serialize_duration =
            end_serialize - start_serialize;

        std::cout << "EXEC STATE:: Tree created and saved" << std::endl;

        // load metadata file, deserialize tree
        state_diff::client_t<float, io_uring_stream_t> new_client(1, reader);
        auto start_deserialize = std::chrono::high_resolution_clock::now();
        {
            std::ifstream ifs(metadata_fn, std::ios::binary);
            boost::archive::binary_iarchive ia(ifs);
            ia >> new_client;
            ifs.close();
        }
        auto end_deserialize = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> deserialize_duration =
            end_deserialize - start_deserialize;
        std::cout << "EXEC STATE:: Tree deserialized" << std::endl;

        auto client_info = client.get_client_info();
        auto new_client_info = new_client.get_client_info();
        if (!(client_info == new_client_info)) {
            test_status = -1;
        }
        std::cout << "BOOST Serialization took " << serialize_duration.count()
                  << " seconds" << std::endl;
        std::cout << "BOOST Deserialization took " << deserialize_duration.count()
                  << " seconds" << std::endl;
    }
    Kokkos::finalize();
    return test_status;
}
