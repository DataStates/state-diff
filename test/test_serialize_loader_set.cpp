#include "liburing_reader.hpp"
#include "common/direct_io.hpp"
#include "statediff.hpp"
#include <cereal/archives/binary.hpp>
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
    float error_tolerance = 1e-4;
    int chunk_size = 4096;
    // Use our rounding hash algorithm or exact hash.
    bool fuzzy_hash = true;
    char dtype = 'f';   // float
    // builds the tree from leaves to root level, can be 12 or 13.
    int root_level = 1;
    std::string fname = "/lus/eagle/projects/RECUP/kassogba/veloc-ckpt/haac/sc-experiments/4gpus/np796-500mil/run1/m000p.mpirestart-combined-0-10.dat";
    std::string metadata_fn = "/lus/eagle/projects/RECUP/kassogba/veloc-ckpt/haac/sc-experiments/4gpus/np796-500mil/run1/m000p.mpirestart-combined-0-10.dat.tree";
    off_t filesize;
    get_file_size(fname, &filesize);
    size_t data_size = static_cast<size_t>(filesize);
    int num_chunks = data_size / chunk_size;
    std::cout << "Nunber of leaf nodes = " << num_chunks << std::endl;

    Kokkos::initialize(argc, argv);
    {
        // read data, build tree and save
	auto start_create = std::chrono::high_resolution_clock::now();
        liburing_io_reader_t reader(fname);
        state_diff::client_t<float> client(
            1, data_size, error_tolerance, dtype, chunk_size,
            root_level, fuzzy_hash, 16ULL*1024*1024*1024, 8ULL*1024*1024*1024);
        client.create(reader);
	auto end_create = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> create_duration =
            end_create - start_create;

        auto start_serialize = std::chrono::high_resolution_clock::now();
        {
            std::ofstream ofs(metadata_fn, std::ios::binary);
            cereal::BinaryOutputArchive oa(ofs);
            oa(client);
            ofs.close();
        }
        auto end_serialize = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> serialize_duration =
            end_serialize - start_serialize;

        std::cout << "EXEC STATE:: Tree created and saved" << std::endl;

        // load metadata file, deserialize tree
        state_diff::client_t<float> new_client;
        auto start_deserialize = std::chrono::high_resolution_clock::now();
        {
            std::ifstream ifs(metadata_fn, std::ios::binary);
            cereal::BinaryInputArchive ia(ifs);
            ia(new_client);
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
	std::cout << "Creation took " << create_duration.count()
                  << " seconds" << std::endl;
        std::cout << "Serialization took " << serialize_duration.count()
                  << " seconds" << std::endl;
        std::cout << "Deserialization took " << deserialize_duration.count()
                  << " seconds" << std::endl;
    }
    Kokkos::finalize();
    return test_status;
}
