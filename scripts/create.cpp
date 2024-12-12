#include "liburing_reader.hpp"
#include "statediff.hpp"
#include <chrono>
#include <iostream>
#include <string>

int
main(int argc, char **argv) {

    std::string fname = argv[1];
    int chunk_size = std::stoi(argv[2]) * MB;

    // Define the parameters
    int host_cache = 1 * GB, dev_cache = 1 * GB, data_size = 1 * GB;
    // int min_chunk_size = 16 * MB, max_chunk_size = 1025 * MB;
    float error_tolerance = 0.01;
    bool fuzzy_hash = true;
    char dtype = 'f';
    int root_level = 1;
    
    TransferType creation_cache_tier = TransferType::FileToHost;
    // TransferType creation_cache_tier = TransferType::FileToDevice;

    liburing_io_reader_t reader(fname);

    Kokkos::initialize(argc, argv);
    {
        // for (int chunk_size = min_chunk_size; chunk_size < max_chunk_size;
        //      chunk_size *= 2) {
        state_diff::client_t<float, liburing_io_reader_t> client(
            1, data_size, error_tolerance, dtype, chunk_size, root_level,
            fuzzy_hash, host_cache, dev_cache);

        auto start_create = std::chrono::high_resolution_clock::now();
        client.create(reader, creation_cache_tier);
        auto end_create = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> create_time =
            end_create - start_create;

        std::vector<double> create_timings = client.get_create_time();

        // Writing timing to log file
        std::fstream benchmark_stream;
        std::string log_fname = "create_timings.csv";
        benchmark_stream.open(log_fname, std::fstream::ate | std::fstream::out | std::fstream::app);
        if(benchmark_stream.tellp() == 0) {
        benchmark_stream << "Chunk Size,Data Size,Number of Leaves,Number of Nodes,"
                    << "Setup time,Leaves time,Rest time"
                    << std::endl;
        }  

        int num_leaves = data_size/chunk_size;
        if(num_leaves * chunk_size < data_size) {
            num_leaves += 1;
        }

        benchmark_stream << chunk_size << "," // chunk size
                    << data_size << "," // data size
                    << num_leaves << "," // number of leaves
                    << 2*num_leaves  + 1 << "," // number of nodes
                    << create_timings[0] << "," // setup time
                    << create_timings[1] << "," // create leaves time
                    << create_timings[2] << std::endl; // create rest of tree time
        benchmark_stream.close();

        std::cout << "Chunk size: " << chunk_size
                    << ", Creation time: " << create_time.count()
                    << " seconds, throughput: "
                    << (data_size / create_time.count()) / (1024 * MB)
                    << " GB/s" << std::endl;
        // }
    }
    Kokkos::finalize();
    return 0;
}
