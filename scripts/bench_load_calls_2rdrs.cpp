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
        // printf("Here\n");
        return idx;
    } else {
        std::vector<size_t> idx(total_chunks);
        std::iota(idx.begin(), idx.end(), 0);
        std::mt19937_64 rng(std::random_device{}());
        std::shuffle(idx.begin(), idx.end(), rng);
        idx.resize(n_segs);
        // printf("There\n");
        return idx;
    }
}

void benchmark_liburing_reads(std::string& fname0, std::string& fname1, size_t dsize,
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
    std::vector<float> buffer1(dsize / sizeof(float), 0);
    std::vector<float> buffer2(dsize / sizeof(float), 0);

    // Compute number of segments
    size_t n_segs = dsize / chunk_size;
    if (n_segs * chunk_size < dsize) n_segs += 1;

    // Open files and get size, close files
    int fd0 = open(fname0.c_str(), O_RDONLY);
    int fd1 = open(fname1.c_str(), O_RDONLY);
    if (fd0 == -1 || fd1 == -1) {
        std::cerr << "cannot open files, error = " << std::strerror(errno) << std::endl;
        return;
    }
    size_t fsize0 = lseek(fd0, 0, SEEK_END);
    size_t fsize1 = lseek(fd1, 0, SEEK_END);
    lseek(fd0, 0, SEEK_SET);
    lseek(fd1, 0, SEEK_SET);
    if( fsize0 != fsize1) {
        std::cerr << "Mismatching file sizes " << std::strerror(errno) << std::endl;
        close(fd0); close(fd1);
        return;
    }
    size_t total_chunks = fsize0 / chunk_size;

    // Create segments
    std::vector<segment_t> segments1(n_segs);
    std::vector<segment_t> segments2(n_segs);
    std::vector<size_t> file_offsets = get_offsets(total_chunks, n_segs, sequential_offt);
    
    for (size_t i = 0; i < n_segs; ++i) {
        segment_t seg0;
        segment_t seg1;
        const size_t off = file_offsets[i] * chunk_size;
        const size_t buf_off  = i * chunk_size;
        seg0.id     = i;
        seg0.fd     = fd0;
        seg0.offset = off;
        seg0.size   = std::min(chunk_size, fsize0 - off);
        seg0.buffer = reinterpret_cast<uint8_t*>(buffer1.data()) + buf_off;
        segments1[i] = seg0;

        seg1.id     = i;
        seg1.fd     = fd1;
        seg1.offset = off;
        seg1.size   = std::min(chunk_size, fsize0 - off);
        seg1.buffer = reinterpret_cast<uint8_t*>(buffer2.data()) + buf_off;
        segments2[i] = seg1;
    }

    // Load and time
    // liburing_io_reader_t reader1(file1);
    // liburing_io_reader_t reader2(file2);
    liburing_io_reader_t reader1;
    liburing_io_reader_t reader2;
    auto start_load = std::chrono::high_resolution_clock::now();
    reader1.enqueue_reads(segments1);
    reader2.enqueue_reads(segments2);
    reader2.wait_all();
    reader1.wait_all();
    auto end_load = std::chrono::high_resolution_clock::now();

    // Compute throughput
    std::chrono::duration<double> load_time = end_load - start_load;
    double throughput = (double(dsize*2) / (1024.0 * 1024.0 * 1024.0)) / load_time.count();

    // Log result
    close(fd0); close(fd1);
    benchmark_stream << n_segs << ","
                        << chunk_size << ","
                        << dsize << ","
                        << load_time.count() << ","
                        << throughput << "\n";
 
    benchmark_stream.close();
}

int main(int argc, char* argv[]) {
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] << " <filename> <dsize> <chunk_size> [--rand]\n";
        return 1;
    }
    std::string fname1 = argv[1];
    std::string fname2 = argv[2];
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
        benchmark_liburing_reads(fname1, fname2, dsize, chunk_size, sequential_offt);
    } catch (const std::exception& e) {
        std::cerr << "Benchmark failed: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

