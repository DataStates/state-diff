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
    // std::vector<float> buffer_1(dsize / sizeof(float), 0);
    // std::vector<float> buffer_2(dsize / sizeof(float), 0);
    std::vector<uint8_t> buffer_1(dsize), buffer_2(dsize);

    // Compute number of segments
    size_t n_segs = dsize / chunk_size;
    if (n_segs * chunk_size < dsize) n_segs += 1;

    // Open files and get size
    // int fd0 = open(fname0.c_str(), O_RDONLY);
    // int fd1 = open(fname1.c_str(), O_RDONLY);
    int fd0 = open(fname0.c_str(), O_RDONLY | O_DIRECT);
    int fd1 = open(fname1.c_str(), O_RDONLY | O_DIRECT);
    if (fd0 == -1 || fd1 == -1) {
        std::cerr << "cannot open files, error = " << std::strerror(errno) << std::endl;
        return;
    }
    size_t fsize0 = lseek(fd0, 0, SEEK_END);
    size_t fsize1 = lseek(fd1, 0, SEEK_END);
    // off_t fsize_1 = lseek(fd_1, 0, SEEK_END);
    // off_t fsize_2 = lseek(fd_2, 0, SEEK_END);
    if (fsize0 < 0 || fsize1 < 0) { perror("lseek"); close(fd0); close(fd1); return; }
    if( fsize0 != fsize1) {
        std::cerr << "Mismatching file sizes " << std::strerror(errno) << std::endl;
        close(fd0); close(fd1);
        return;
    }

    size_t total_chunks = fsize0 / chunk_size;
    if (n_segs > total_chunks) {
        std::cerr << "Requested " << n_segs << " segments but file holds only "
                << total_chunks << " chunks of size " << chunk_size << "\n";
        close(fd0); close(fd1); return;
    }
    size_t n_reads = n_segs*2;
    std::vector<segment_t> segments(n_reads);
    liburing_io_reader_t file_reader;
    std::vector<size_t> file_offsets = get_offsets(total_chunks, n_segs, sequential_offt);
    // std::cout << "Loader initialized" << std::endl;

    // Create segments
    for (size_t j = 0; j < n_segs; ++j) {
        const size_t off = file_offsets[j] * chunk_size;
        const size_t buf_off  = j * chunk_size;
        if (off >= fsize0) 
            break;

        // File 1
        segment_t seg0;
        seg0.id     = j;
        seg0.fd     = fd0;
        seg0.offset = off;
        seg0.size   = std::min(chunk_size, dsize - buf_off);
        seg0.buffer = reinterpret_cast<uint8_t*>(buffer_1.data()) + buf_off;
        posix_memalign((void**)&seg0.buffer, 4096, seg0.size);
        segments[2*j] = seg0;

        // File 2
        segment_t seg1;
        seg1.id     = j;
        seg1.fd     = fd1;
        seg1.offset = off;
        seg1.size   = std::min(chunk_size,  dsize - buf_off);
        seg1.buffer = reinterpret_cast<uint8_t*>(buffer_2.data()) + buf_off;
        posix_memalign((void**)&seg1.buffer, 4096, seg1.size);
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
    close(fd0);
    close(fd1);
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

