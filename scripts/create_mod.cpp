#include "liburing_reader.hpp"
#include "statediff.hpp"
#include "common/direct_io.hpp"
#include <chrono>
#include <iostream>
#include <string>

int
main(int argc, char **argv) {

    std::string fname = argv[1];
    size_t chunk_size = std::stol(argv[2]);
    double error_tolerance = std::stod(argv[3]);

    // Define the parameters
    size_t host_cache = 16ULL * GB, dev_cache = 16ULL * GB, data_size = 0;
    off_t filesize;
    get_file_size(fname, &filesize);
    data_size = static_cast<size_t>(filesize);
    
    bool fuzzy_hash = true;
    char dtype = 'f';
    int root_level = 13;
    
    Kokkos::initialize(argc, argv);
    {
	TransferType creation_cache_tier = TransferType::FileToHost;

	liburing_io_reader_t reader(fname);

        state_diff::client_t<float> client(
            1, data_size, error_tolerance, dtype, chunk_size, root_level,
            fuzzy_hash, host_cache, dev_cache);

        auto start_create = std::chrono::high_resolution_clock::now();
        client.create(reader, creation_cache_tier);
        auto end_create = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> create_time =
            end_create - start_create;

        std::vector<double> create_timings = client.get_create_time();
	chunk_size = client.get_client_info().chunk_size;

        // Writing timing to log file
        std::fstream benchmark_stream;
        std::string log_fname = "create_timings.csv";
        benchmark_stream.open(log_fname, std::fstream::ate | std::fstream::out | std::fstream::app);
        if(benchmark_stream.tellp() == 0) {
        benchmark_stream << "Chunk Size,Error,Data Size,Number of Leaves,Number of Nodes,"
                    << "Setup time,Leaves time,Rest time,Load time,Hashing time"
                    << std::endl;
        }  

        int num_leaves = data_size/chunk_size;
        if(num_leaves * chunk_size < data_size) {
            num_leaves += 1;
        }

        benchmark_stream << chunk_size << "," // chunk size
		    << error_tolerance << "," // error tolerance
                    << data_size << "," // data size
                    << num_leaves << "," // number of leaves
                    << 2*num_leaves  + 1 << "," // number of nodes
                    << create_timings[0] << "," // setup time
                    << create_timings[1] << "," // create leaves time
                    << create_timings[2] << ","
		    << create_timings[3] << ","
		    << create_timings[4] << std::endl; // create rest of tree time
        benchmark_stream.close();

        std::cout << "(" << error_tolerance << ") Chunk size: " << chunk_size
                    << ", Creation time: " << create_time.count()
                    << " seconds, throughput: "
                    << (data_size / create_time.count()) / (1024 * MB)
                    << " GB/s" << std::endl;
    }
    Kokkos::finalize();
    return 0;
}
