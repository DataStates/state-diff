#include "liburing_reader.hpp"
#include <vector>
#include <chrono>
#include <fstream>
#include <stdexcept>
#include <string>
#include <cmath>
#include <random>
#include <numeric>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <algorithm>

std::vector<size_t> get_offsets(size_t total_chunks, size_t n_segs, bool sequential) {
    if(sequential) {
        std::vector<size_t> idx(n_segs);
        std::iota(idx.begin(), idx.end(), 0);
        return idx;
    } else {
        std::vector<size_t> idx(total_chunks);
        std::iota(idx.begin(), idx.end(), 0);
        std::mt19937_64 rng(std::random_device{}());
        std::shuffle(idx.begin(), idx.end(), rng);
        idx.resize(n_segs);
        return idx;
    }
}

void benchmark_liburing_reads(std::string& file_1, std::string& file_2, size_t dsize,
                               size_t chunk_size, bool sequential_offt,
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
    std::vector<float> buffer_1(dsize / sizeof(float), 0);
    std::vector<float> buffer_2(dsize / sizeof(float), 0);

    // Compute number of segments
    size_t n_segs = dsize / chunk_size;
    if (n_segs * chunk_size < dsize) n_segs += 1;

    // Open files and get size
    int fd_1 = open(file_1.c_str(), O_RDONLY);
    int fd_2 = open(file_2.c_str(), O_RDONLY);
    size_t fsize_1 = lseek(fd_1, 0, SEEK_END);
    size_t fsize_2 = lseek(fd_2, 0, SEEK_END);
    lseek(fd_1, 0, SEEK_SET);
    lseek(fd_2, 0, SEEK_SET);
    if (fd_1 == -1 || fd_2 == -1) {
        std::cerr << "cannot open files, error = " << std::strerror(errno) << std::endl;
        return;
    }
    if( fsize_1 != fsize_2) {
        std::cerr << "Mismatching file sizes " << std::strerror(errno) << std::endl;
        close(fd_1); close(fd_2);
        return;
    }

    size_t total_chunks = fsize_1 / chunk_size;
    size_t n_reads = n_segs*2;
    std::vector<segment_t> segments(n_reads);
    liburing_io_reader_t file_reader("", dsize);
    std::vector<size_t> file_offsets = get_offsets(total_chunks, n_segs, sequential_offt);
    // std::cout << "Loader initialized" << std::endl;

    // Create segments
    for (size_t j = 0; j < n_segs; ++j) {
        const size_t off = file_offsets[j] * chunk_size;
        if (off >= fsize_1) break;

        // File 1
        segment_t seg0;
        seg0.fd     = fd_1;
        seg0.offset = off;
        seg0.size   = std::min(chunk_size, fsize_1 - off);
        seg0.buffer = reinterpret_cast<uint8_t*>(buffer_1.data()) + off;
        segments[2*j] = seg0;

        // File 2
        segment_t seg1;
        seg1.fd     = fd_2;
        seg1.offset = off;
        seg1.size   = std::min(chunk_size, fsize_1 - off);
        seg1.buffer = reinterpret_cast<uint8_t*>(buffer_2.data()) + off;
        segments[2*j + 1] = seg1;
    }
    // printf("%zu Segments created\n", n_reads);
    // std::cout << n_reads << " Segments created" << std::endl;

    // Load and time
    auto start_load = std::chrono::high_resolution_clock::now();
    file_reader.enqueue_reads(segments);
    // printf("Segments enqueued. Waiting...\n");
    // std::cout << "Segments enqueued. Waiting..." << std::endl;
    file_reader.wait_all();
    auto end_load = std::chrono::high_resolution_clock::now();
    // printf("Data loading completed\n");
    // std::cout << "Data loading completed" << std::endl;

    // Compute throughput
    std::chrono::duration<double> load_time = end_load - start_load;
    double throughput = (double(dsize*2) / (1024.0 * 1024.0 * 1024.0)) / load_time.count();

    // Log result
    benchmark_stream << n_segs << ","
                        << chunk_size << ","
                        << dsize << ","
                        << load_time.count() << ","
                        << throughput << "\n";
 
    benchmark_stream.close();
    close(fd_1);
    close(fd_2);
}

int main(int argc, char* argv[]) {
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] << " <filename> <dsize> <chunk_size> [--rand]\n";
        return 1;
    }
    std::string pfs_fname = argv[1];
    std::string ssd_fname = argv[2];
    size_t chunk_size = std::stoull(argv[3]);
    size_t nsegs = std::stoull(argv[4]);
    bool sequential_offt = true;
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--rand") {
            sequential_offt = false;
        }
    }
    size_t dsize = chunk_size * nsegs;

    try {
        benchmark_liburing_reads(pfs_fname, ssd_fname, dsize, chunk_size, sequential_offt);
    } catch (const std::exception& e) {
        std::cerr << "Benchmark failed: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

