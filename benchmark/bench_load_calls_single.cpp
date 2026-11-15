#include <scissor/io_reader.hpp>
#include <scissor/readers.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <stdexcept>
#include <string>
#include <cmath>
#include <fcntl.h>
#include <unistd.h>

void benchmark_liburing_reads(std::string& filename, size_t dsize,
                               size_t chunk_size,
                               const std::string& log_fname = "benchmark_load_single.csv") {
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

    // Open file and build segments with explicit fd
    int fd = open(filename.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("Cannot open file: " + filename);

    std::vector<scissor::Segment> segments(n_segs);
    for (size_t i = 0; i < n_segs; ++i) {
        scissor::Segment seg{};
        seg.id     = i;
        seg.fd     = fd;
        seg.offset = chunk_size * i;
        seg.size   = (i == n_segs - 1) ? dsize - seg.offset : chunk_size;
        seg.buffer = reinterpret_cast<uint8_t*>(buffer.data()) + seg.offset;
        segments[i] = seg;
    }

    // Load and time via scissor::LiburingReader
    scissor::LiburingReader reader;

    auto start_load = std::chrono::high_resolution_clock::now();
    reader.enqueue_reads(segments);
    reader.wait_all();
    auto end_load = std::chrono::high_resolution_clock::now();
    close(fd);

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
    size_t chunk_size = std::stoull(argv[2]);
    size_t nsegs = std::stoull(argv[3]);
    size_t dsize = chunk_size * nsegs;

    try {
        benchmark_liburing_reads(fname, dsize, chunk_size);
    } catch (const std::exception& e) {
        std::cerr << "Benchmark failed: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
