#include "statediff.hpp"
#include "stdio.h"
#include <Kokkos_Core.hpp>
#include <adios2.h>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <omp.h>
#include <random>
#include <string>
#include <vector>

void
adios_writer(adios2::ADIOS &adios, std::vector<uint8_t> &buffer) {
    adios2::IO io = adios.DeclareIO("tree-writer");
    io.SetEngine("BP5");
    adios2::Variable<uint8_t> adios_tree = io.DefineVariable<uint8_t>(
        "tree0", {buffer.size()}, {0}, {buffer.size()});
    adios2::Engine writer = io.Open("run0_tree.bp", adios2::Mode::Write);
    writer.Put<uint8_t>(adios_tree, buffer.data());
    writer.Close();
}

std::vector<uint8_t>
adios_reader(adios2::ADIOS &adios) {
    adios2::IO io = adios.DeclareIO("tree-reader");
    io.SetEngine("BP5");
    adios2::Engine reader = io.Open("run0_tree.bp", adios2::Mode::Read);
    reader.BeginStep();
    adios2::Variable<uint8_t> variable = io.InquireVariable<uint8_t>("tree0");
    std::vector<uint8_t> deserialized_buffer;
    reader.Get<uint8_t>(variable, deserialized_buffer);
    reader.EndStep();
    reader.Close();
    return deserialized_buffer;
}

int
main(int argc, char **argv) {
    int test_status = 0;

    // Define the parameters
    float max_float =
        100.0;   // maximum floating-point (FP) value in synthetic data
    float min_float = 0.0;   // minimum FP value in synthetic data
    int data_size =
        1024 * 1024 * 1024;         // size in bytes of the synthetic data (1GB)
    float error_tolerance = 1e-4;   // Application error tolerance
    int chunk_size = 512;     // Target chunk size. This example uses 16 bytes
    bool fuzzy_hash = true;   // Set to true to use our rounding hash algorithm.
                              // Otherwise, directly hash blocks of FP values
    char dtype = 'f';         // float
    int seed = 0x123;   // Random number seed to generate the synthetic data
    int root_level =
        1;   // builds the tree from the leaf level to level 1 (root level). For
             // better parallelism, set root_level to 12 or 13.

    int num_chunks = data_size / chunk_size;
    std::cout << "Nunber of leaf nodes = " << num_chunks << std::endl;

    Kokkos::initialize(argc, argv);
    {
        // Create synthetic datasets
        Timer::time_point start_dataloading = Timer::now();
        int data_len = data_size / sizeof(float);
        std::vector<float> data_run0_h(data_len);
#pragma omp parallel
        {
            std::mt19937 prng(seed + omp_get_thread_num());
            std::uniform_real_distribution<float> prng_dist(min_float,
                                                            max_float);
#pragma omp for
            for (int i = 0; i < data_len; ++i) {
                data_run0_h[i] = prng_dist(prng);
            }
        }

        float *data_run0_ptr = (float *) data_run0_h.data();
        Kokkos::View<float *> data_run0_d("Run0 Data", data_run0_h.size());
        Kokkos::View<float *, Kokkos::HostSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>
            data_run0_h_kokkos(data_run0_ptr, data_run0_h.size());
        Kokkos::deep_copy(data_run0_d, data_run0_h_kokkos);
        std::cout << "EXEC STATE:: Data loaded and transfered to GPU"
                  << std::endl;

        // Create tree_run0
        CompareTreeDeduplicator tree_object(chunk_size, root_level, fuzzy_hash,
                                            error_tolerance, dtype);
        tree_object.setup(data_size);
        tree_object.create_tree((uint8_t *) data_run0_d.data(),
                                data_run0_d.size());
        std::cout << "EXEC STATE:: Tree created" << std::endl;

        // Serialize tree
        std::vector<uint8_t> serialized_buffer;
        serialized_buffer = tree_object.serialize();
        adios2::ADIOS adios_client;
        adios_writer(adios_client, serialized_buffer);
        std::cout << "EXEC STATE:: Tree serialized to ADIOS2's BP file"
                  << std::endl;

        // Deserialize tree
        std::vector<uint8_t> deserialized_buffer = adios_reader(adios_client);
        CompareTreeDeduplicator new_tree_object(
            chunk_size, root_level, fuzzy_hash, error_tolerance, dtype);
        new_tree_object.setup(data_size);
        new_tree_object.deserialize(deserialized_buffer);
        std::cout
            << "EXEC STATE:: Tree Read from ADIOS2's BP file and deserialized"
            << std::endl;

        if (tree_object.num_nodes != new_tree_object.num_nodes ||
            serialized_buffer.size() != deserialized_buffer.size()) {
            test_status = -1;
        }
    }
    Kokkos::finalize();

    return test_status;
}
