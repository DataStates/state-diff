#include "statediff.hpp"
#include <scissor/readers.hpp>
#include <chrono>
#include <iostream>
#include <string>

int
main(int argc, char **argv) {

    std::string fname = argv[1];
    size_t chunk_size = std::stoi(argv[2]);

    // Define the parameters
    size_t host_cache = 2 * GB, dev_cache = 2 * GB;
    size_t data_size = chunk_size;
    float error_tolerance = 0.01;
    bool fuzzy_hash = true;
    char dtype = 'f';
    int root_level = 1;

    TransferType creation_cache_tier = TransferType::FileToHost;
    int flag = 0;
#ifdef __NVCC__
    creation_cache_tier = TransferType::FileToDevice;
    flag = 1;
#endif

    Kokkos::initialize(argc, argv);
    {
        scissor::LiburingReader reader;
        state_diff::client_t<float> client(1, reader, data_size, error_tolerance,
                                           dtype, chunk_size, root_level, fuzzy_hash);

        auto start_create = std::chrono::high_resolution_clock::now();
        client.create(fname, chunk_size, creation_cache_tier);
        auto end_create = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> create_time = end_create - start_create;

        std::vector<double> create_timings = client.get_create_time();
        chunk_size = client.get_client_info().chunk_size;

        // Writing timing to log file
        std::fstream benchmark_stream;
        std::string log_fname = "create_timings.csv";
        benchmark_stream.open(log_fname, std::fstream::ate | std::fstream::out |
                                             std::fstream::app);
        if (benchmark_stream.tellp() == 0) {
            benchmark_stream
                << "Create GPU,Chunk Size,Data Size,Number of Leaves,Number of "
                   "Nodes,"
                << "Setup time,Leaves time,Rest time,Load time,Hashing time"
                << std::endl;
        }

        int num_leaves = data_size / chunk_size;
        if (num_leaves * chunk_size < data_size) {
            num_leaves += 1;
        }

        benchmark_stream << flag << "," << chunk_size << ","   // chunk size
                         << data_size << ","                   // data size
                         << num_leaves << ","           // number of leaves
                         << 2 * num_leaves + 1 << ","   // number of nodes
                         << create_timings[0] << ","    // setup time
                         << create_timings[1] << ","    // create leaves time
                         << create_timings[2] << "," << create_timings[3] << ","
                         << create_timings[4]
                         << std::endl;   // create rest of tree time
        benchmark_stream.close();

        std::cout << "(" << flag << ") Chunk size: " << chunk_size
                  << ", Creation time: " << create_time.count()
                  << " seconds, throughput: "
                  << (data_size / create_time.count()) / (1024 * MB) << " GB/s"
                  << std::endl;
    }
    Kokkos::finalize();
    return 0;
}
