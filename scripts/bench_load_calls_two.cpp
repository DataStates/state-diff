#include "liburing_reader.hpp"
#include <vector>
#include <chrono>
#include <fstream>
#include <stdexcept>
#include <string>
#include <cmath>

void benchmark_liburing_reads(std::string& pfs_filename, std::string& ssd_filename, size_t dsize,
                               size_t chunk_size,
                               const std::string& log_fname = "benchmark_load_two.csv") {
    // Open log file
    std::fstream benchmark_stream(log_fname, std::fstream::ate | std::fstream::out | std::fstream::app);
    if (!benchmark_stream.is_open()) {
        throw std::runtime_error("Failed to open log file: " + log_fname);
    }

    // Write header if file is empty
    if (benchmark_stream.tellp() == 0) {
        benchmark_stream << "IOP,Chunk size (B),Total size (B),Load time (s),Throughput (GiB/s)\n";
    }

    // Create data buffer
    std::vector<float> pfs_buffer(dsize / sizeof(float), 0);
    std::vector<float> ssd_buffer(dsize / sizeof(float), 0);

    // Compute number of segments
    size_t n_segs = dsize / chunk_size;
    if (n_segs * chunk_size < dsize) n_segs += 1;

    // Create segments
    std::vector<segment_t> pfs_segments(n_segs);
    std::vector<segment_t> ssd_segments(n_segs);
    
    for (size_t i = 0; i < n_segs; ++i) {
        segment_t seg0;
        segment_t seg1;
        seg0.offset = chunk_size * i;
        seg0.size = (i == n_segs - 1) ? dsize - seg0.offset : chunk_size;
        seg0.buffer = reinterpret_cast<uint8_t*>(pfs_buffer.data()) + seg0.offset;
        pfs_segments[i] = seg0;

        seg1.offset = chunk_size * i;
        seg1.size = (i == n_segs - 1) ? dsize - seg1.offset : chunk_size;
        seg1.buffer = reinterpret_cast<uint8_t*>(ssd_buffer.data()) + seg1.offset;
        ssd_segments[i] = seg1;
    }

    // Load and time
    liburing_io_reader_t pfs_reader(pfs_filename);
    liburing_io_reader_t ssd_reader(ssd_filename);
    auto start_load = std::chrono::high_resolution_clock::now();
    pfs_reader.enqueue_reads(pfs_segments);
    ssd_reader.enqueue_reads(ssd_segments);
    ssd_reader.wait_all();
    pfs_reader.wait_all();
    auto end_load = std::chrono::high_resolution_clock::now();

    // Compute throughput
    std::chrono::duration<double> load_time = end_load - start_load;
    double throughput = (double(dsize) / (1024.0 * 1024.0 * 1024.0)) / load_time.count();

    // Log result
    benchmark_stream << n_segs << ","
                        << chunk_size << ","
                        << dsize << ","
                        << load_time.count() << ","
                        << throughput << "\n";
 
    benchmark_stream.close();
}

int main(int argc, char* argv[]) {
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << " <filename> <dsize> <chunk_size>\n";
        return 1;
    }
    std::string pfs_fname = argv[1];
    std::string ssd_fname = argv[2];
    size_t chunk_size = std::stoull(argv[3]);
    size_t nsegs = std::stoull(argv[4]);
    size_t dsize = chunk_size * nsegs;

    try {
        benchmark_liburing_reads(pfs_fname, ssd_fname, dsize, chunk_size);
    } catch (const std::exception& e) {
        std::cerr << "Benchmark failed: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

