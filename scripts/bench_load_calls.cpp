#include "liburing_reader.hpp"
#include <vector>
#include <chrono>
#include <fstream>
#include <stdexcept>
#include <string>
#include <cmath>

void benchmark_liburing_reads(std::string& filename, size_t dsize,
                               size_t chunk_size,
                               const std::string& log_fname = "benchmark_load.csv") {
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
    std::vector<float> buffer(dsize / sizeof(float), 0);

    // Compute number of segments
    size_t n_segs = dsize / chunk_size;
    if (n_segs * chunk_size < dsize) n_segs += 1;

    // Create segments
    std::vector<segment_t> segments(n_segs);
    for (size_t i = 0; i < n_segs; ++i) {
        segment_t seg;
        seg.offset = chunk_size * i;
        seg.size = (i == n_segs - 1) ? dsize - seg.offset : chunk_size;
        seg.buffer = reinterpret_cast<uint8_t*>(buffer.data()) + seg.offset;
        segments[i] = seg;
    }

    // Load and time
    liburing_io_reader_t reader(filename);
    auto start_load = std::chrono::high_resolution_clock::now();
    reader.enqueue_reads(segments);
    reader.wait_all();
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
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <filename> <dsize> <chunk_size>\n";
        return 1;
    }

    std::string fname = argv[1];
    size_t dsize = std::stoull(argv[2]);
    size_t chunk_size = std::stoull(argv[3]);

    try {
        benchmark_liburing_reads(fname, dsize, chunk_size);
    } catch (const std::exception& e) {
        std::cerr << "Benchmark failed: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

