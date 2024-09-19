#include "adios_reader.hpp"
#include "statediff.hpp"
#include <adios2.h>
#include <cereal/archives/binary.hpp>
#include <chrono>
#include <fstream>
#include <iostream>
#include <omp.h>
#include <random>
#include <string>
#include <vector>

template <typename D, typename T>
void
adios_write(adios2::IO &io, std::vector<D> &data_buffer,
            std::vector<T> &tree_buffer, int run_id, std::string fn) {
    io.SetEngine("BP5");
    adios2::Variable<D> adios_data = io.DefineVariable<D>(
        "data", {data_buffer.size()}, {0}, {data_buffer.size()});
    adios2::Variable<T> adios_tree = io.DefineVariable<T>(
        "tree", {tree_buffer.size()}, {0}, {tree_buffer.size()});
    adios2::Engine writer = io.Open(fn, adios2::Mode::Write);
    writer.Put<D>(adios_data, data_buffer.data());
    writer.Put<T>(adios_tree, tree_buffer.data());
    writer.Close();
}

template <typename T>
std::vector<T>
adios_read(adios2::IO &io, std::string tag, std::string fn) {
    io.SetEngine("BP5");
    adios2::Engine reader = io.Open(fn, adios2::Mode::Read);
    reader.BeginStep();
    adios2::Variable<T> variable = io.InquireVariable<T>(tag);
    std::vector<T> deserialized_buffer;
    reader.Get<T>(variable, deserialized_buffer);
    reader.EndStep();
    reader.Close();
    return deserialized_buffer;
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
    std::string fname = "checkpoint.bp";
    adios_reader_t reader(fname);
    adios2::ADIOS adios_client;
    adios2::IO io_writer = adios_client.DeclareIO("writer");
    adios2::IO io_reader = adios_client.DeclareIO("reader");

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

        // read data, build tree and save (tree + data) to BP file
        state_diff::client_t<float, adios_reader_t> client(
            1, reader, data_size, error_tolerance, dtype, chunk_size,
            root_level, fuzzy_hash);
        client.create(run_data);
        std::vector<uint8_t> ser_buf;
        {
            std::stringstream out_string_stream;
            cereal::BinaryOutputArchive out_archive(out_string_stream);
            out_archive(client);
            std::string str = out_string_stream.str();
            ser_buf = std::vector<uint8_t>(str.begin(), str.end());
        }
        adios_write<float, uint8_t>(io_writer, run_data, ser_buf, 1, fname);
        std::cout
            << "EXEC STATE:: Tree created and serialized to ADIOS2's BP file"
            << std::endl;

        // load metadata only from BP, deserialize tree
        std::vector<uint8_t> tree_buf;
        tree_buf = adios_read<uint8_t>(io_reader, "tree", fname);
        state_diff::client_t<float, adios_reader_t> new_client(1, reader);
        {
            std::istringstream in_string_stream(
                std::string(tree_buf.begin(), tree_buf.end()));
            cereal::BinaryInputArchive in_archive(in_string_stream);
            in_archive(new_client);
        }
        std::cout << "EXEC STATE:: Tree deserialized from ADIOS2's BP file"
                  << std::endl;

        auto client_info = client.get_client_info();
        auto new_client_info = new_client.get_client_info();
        if (!(client_info == new_client_info)) {
            test_status = -1;
        }
    }
    Kokkos::finalize();
    return test_status;
}