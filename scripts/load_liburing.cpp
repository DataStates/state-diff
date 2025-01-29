#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include "liburing_reader.hpp"
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
    std::vector<segment_t> segments(1);
    liburing_io_reader_t reader(filename);

    double total_time = 0;

    for (int i = 0; i < n_segs; i++) {
        // Handle last segment
        size_t worksize = chunk_size;
        if(i == n_segs - 1) {
            worksize = dsize - (chunk_size * i);
        }

        // Create request
        segment_t seg;
        seg.buffer = (uint8_t*)buffer.data();
        seg.offset = 0;
        seg.size = worksize;
        segments[0] = seg;

        // Issue and measure read latency
        auto start_load = std::chrono::high_resolution_clock::now();
        reader.enqueue_reads(segments);
        reader.wait_all();
        auto end_load = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> load_time = end_load - start_load;
        total_time += load_time.count();
    }
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

    benchmark_stream << "Uring-1xN," << chunk_size << ","  // chunk size
                     << dsize << ","      // data size
                     << total_time << ","  // ld time
                     << total_thrupt << std::endl;  // ld throughput
    benchmark_stream.close();

    std::cout << "(" << chunk_size << ") Liburing throughput for one chunk at a time = " << total_thrupt << " GB/s" << std::endl;
}

void allforone(std::string &filename, size_t chunk_size) {

    // Get size of source file
    off_t filesize;
    get_file_size(filename, &filesize);
    size_t dsize = static_cast<size_t>(filesize);

    // Create a buffer to hold the data loaded by liburing
    std::vector<float> buffer(dsize / sizeof(float), 0);

    // Compute the number of segments to read the entire data
    int n_segs = dsize / chunk_size;
    if (n_segs * chunk_size < dsize)
        n_segs += 1;

    // Create segments and initialize reader
    std::vector<segment_t> segments(n_segs);
    liburing_io_reader_t reader(filename);

    for (int i = 0; i < n_segs; i++) {
        segment_t seg;
        seg.buffer = (uint8_t*)(buffer.data() + (chunk_size * i) / sizeof(float));  // Adjust for float size
        seg.offset = chunk_size * i;
        seg.size = (i == n_segs - 1) ? dsize - (chunk_size * i) : chunk_size;  // Handle last segment
        segments[i] = seg;
    }

    auto start_load = std::chrono::high_resolution_clock::now();
    reader.enqueue_reads(segments);
    reader.wait_all();
    auto end_load = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> load_time = end_load - start_load;
    double throughput = (dsize / (1024 * 1024 * 1024)) / load_time.count();

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

    benchmark_stream << "Uring-Nx1," << chunk_size << ","  // chunk size
                     << dsize << ","      // data size
                     << load_time.count() << ","  // ld time
                     << throughput << std::endl;  // ld throughput
    benchmark_stream.close();

    std::cout << "(" << chunk_size << ")  Liburing throughput for all chunks at once = " << throughput << " GB/s" << std::endl;
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
