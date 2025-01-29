#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include "common/direct_io.hpp"


void oneforall(std::string &filename, size_t chunk_size) {

    // Get size of source file
    off_t filesize;
    get_file_size(filename, &filesize);
    size_t dsize = static_cast<size_t>(filesize);

    // Create a buffer to hold the data loaded by liburing
    std::vector<float> buffer(chunk_size / sizeof(float), 0);

    // Compute the number of segments to read the entire data
    int n_segs = dsize / chunk_size;
    if (n_segs * chunk_size < dsize)
        n_segs += 1;

    // Create segments and initialize reader
    std::ifstream f;
    f.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    f.open(filename, std::ios::in | std::ios::binary);

    double total_time = 0;
    for (int i = 0; i < n_segs; i++) {
        // Handle last segment
        size_t worksize = chunk_size;
        if(i == n_segs - 1) {
            worksize = dsize - (chunk_size * i);
        }

        // Create request
        auto start = std::chrono::high_resolution_clock::now();
        uint8_t *ptr_h = (uint8_t*) buffer.data();
        f.seekg(i * chunk_size);
        f.read(reinterpret_cast<char *>(ptr_h), worksize);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> load_time = end - start;
        total_time += load_time.count();
    }
    f.close();
    double total_thrupt = (dsize / (1024 * 1024 * 1024)) / total_time;

    // Writing timing to log file
    std::fstream benchmark_stream;
    std::string log_fname = "validate_liburing.csv";
    benchmark_stream.open(log_fname, std::fstream::ate | std::fstream::out | std::fstream::app);
    if (!benchmark_stream.is_open()) {
        throw std::runtime_error("Failed to open log file: " + log_fname);
    }
    if (benchmark_stream.tellp() == 0) {
        benchmark_stream << "API,Chunk Size,Data Size,Load time,Load thrupt" << std::endl;
    }

    benchmark_stream << "Posix-1xN," << chunk_size << ","  // chunk size
                     << dsize << ","      // data size
                     << total_time << ","  // ld time
                     << total_thrupt << std::endl;  // ld throughput
    benchmark_stream.close();

    std::cout << "(" << chunk_size << ") Posix throughput for one chunk at a time = " << total_thrupt << " GB/s" << std::endl;
}

void allforone(std::string &filename, size_t chunk_size) {

    // Get size of source file
    off_t filesize;
    get_file_size(filename, &filesize);
    size_t dsize = static_cast<size_t>(filesize);

    // Create a buffer to hold the data
    std::vector<float> buffer(dsize / sizeof(float), 0);

    // Read all data at once
    std::ifstream f;
    f.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    f.open(filename, std::ios::in | std::ios::binary);
    uint8_t *ptr_h = (uint8_t*) buffer.data();

    auto start = std::chrono::high_resolution_clock::now();
    f.read(reinterpret_cast<char *>(ptr_h), dsize);
    f.close();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    double throughput = (dsize / (1024 * 1024 * 1024)) / duration.count();

    // Writing timing to log file
    std::fstream benchmark_stream;
    std::string log_fname = "validate_liburing.csv";
    benchmark_stream.open(log_fname, std::fstream::ate | std::fstream::out | std::fstream::app);
    if (!benchmark_stream.is_open()) {
        throw std::runtime_error("Failed to open log file: " + log_fname);
    }
    if (benchmark_stream.tellp() == 0) {
        benchmark_stream << "API,Chunk Size,Data Size,Load time,Load thrupt" << std::endl;
    }

    benchmark_stream << "Posix-Nx1," << dsize << ","  // chunk size
                     << dsize << ","      // data size
                     << duration.count() << ","  // ld time
                     << throughput << std::endl;  // ld throughput
    benchmark_stream.close();

    std::cout << "Posix throughput for all data requested at once = " << throughput << " GB/s" << std::endl;
}

int
main(int argc, char **argv) {

    std::string filename = argv[1];
    size_t chunk_size = std::stoi(argv[2]);
    int type = std::stoi(argv[3]);

    if(type == 0) {
        oneforall(filename, chunk_size);
    } else {
        allforone(filename, chunk_size);
    }
    return 0;
}
