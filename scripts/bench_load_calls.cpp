#include <scissor/readers.hpp>
#include <iostream>
#include <cstring>
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
#include <memory>

struct FreeDeleter {
    void operator()(void *p) const { free(p); }
};

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
                               size_t chunk_size, bool sequential_offt, bool grouped, 
                               std::string log_fname) {
    // Open log file
    std::fstream benchmark_stream(log_fname, std::fstream::ate | std::fstream::out | std::fstream::app);
    if (!benchmark_stream.is_open()) {
        throw std::runtime_error("Failed to open log file: " + log_fname);
    }

    // Write header if file is empty
    if (benchmark_stream.tellp() == 0) {
        benchmark_stream << "IOP,Chunk size (B),Total size (B),Load time (s),Compare time (s),Throughput (GiB/s)\n";
    }

    // Create 4 KiB aligned data buffers (O_DIRECT)
    std::unique_ptr<uint8_t, FreeDeleter> buffer_1, buffer_2;
    for (auto *buffer : {&buffer_1, &buffer_2}) {
        void *ptr = nullptr;
        if (posix_memalign(&ptr, 4096, dsize) != 0)
            throw std::runtime_error("Failed to allocate " + std::to_string(dsize) + " B buffer");
        buffer->reset(static_cast<uint8_t*>(ptr));
    }

    // Compute number of segments
    size_t n_segs = dsize / chunk_size;
    if (n_segs * chunk_size < dsize) n_segs += 1;

    // Open files and get size
    int fd0 = open(fname0.c_str(), O_RDONLY | O_DIRECT);
    int fd1 = open(fname1.c_str(), O_RDONLY | O_DIRECT);
    if (fd0 == -1 || fd1 == -1) {
        std::cerr << "cannot open files, error = " << std::strerror(errno) << std::endl;
        return;
    }
    off_t end0 = lseek(fd0, 0, SEEK_END);
    off_t end1 = lseek(fd1, 0, SEEK_END);
    if (end0 < 0 || end1 < 0) { perror("lseek"); close(fd0); close(fd1); return; }
    size_t fsize0 = end0, fsize1 = end1;
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
    std::vector<scissor::Segment> segments(n_reads);
    scissor::LiburingReader file_reader;
    std::vector<size_t> file_offsets = get_offsets(total_chunks, n_segs, sequential_offt);
    // std::cout << "Loader initialized" << std::endl;

    // Create segments
    if(grouped) {
        for (size_t j = 0; j < n_segs; ++j) {
            const size_t off = file_offsets[j] * chunk_size;
            const size_t buf_off  = j * chunk_size;
            if (off >= fsize0) 
                break;

            // File 1
            scissor::Segment seg0;
            seg0.id     = j;
            seg0.fd     = fd0;
            seg0.offset = off;
            seg0.size   = std::min(chunk_size, dsize - buf_off);
            seg0.buffer = buffer_1.get() + buf_off;
            segments[j] = seg0;
        }
        for (size_t j = 0; j < n_segs; ++j) {
            const size_t off = file_offsets[j] * chunk_size;
            const size_t buf_off  = j * chunk_size;
            if (off >= fsize1) 
                break;
            // File 2
            scissor::Segment seg1;
            seg1.id     = n_segs + j;
            seg1.fd     = fd1;
            seg1.offset = off;
            seg1.size   = std::min(chunk_size,  dsize - buf_off);
            seg1.buffer = buffer_2.get() + buf_off;
            segments[n_segs + j] = seg1;
        }
    } else {
        for (size_t j = 0; j < n_segs; ++j) {
            const size_t off = file_offsets[j] * chunk_size;
            const size_t buf_off  = j * chunk_size;
            if (off >= fsize0) 
                break;

            // File 1
            scissor::Segment seg0;
            seg0.id     = 2*j;
            seg0.fd     = fd0;
            seg0.offset = off;
            seg0.size   = std::min(chunk_size, dsize - buf_off);
            seg0.buffer = buffer_1.get() + buf_off;
            segments[2*j] = seg0;

            // File 2
            scissor::Segment seg1;
            seg1.id     = 2*j + 1;
            seg1.fd     = fd1;
            seg1.offset = off;
            seg1.size   = std::min(chunk_size,  dsize - buf_off);
            seg1.buffer = buffer_2.get() + buf_off;
            segments[2*j + 1] = seg1;
        }
    }
    // std::cout << n_reads << " Segments created" << std::endl;

    // Load and time
    auto start_load = std::chrono::high_resolution_clock::now();
    file_reader.enqueue_reads(segments);
    // std::cout << "Segments enqueued. Waiting..." << std::endl;
    file_reader.wait_all();
    auto end_load = std::chrono::high_resolution_clock::now();
    // std::cout << "Data loading completed" << std::endl;

    auto start_compare = std::chrono::high_resolution_clock::now();
    size_t n_floats = dsize / sizeof(float);
    float* buf1 = reinterpret_cast<float*>(buffer_1.get());
    float* buf2 = reinterpret_cast<float*>(buffer_2.get());
    size_t count = 0;
    for(size_t i = 0; i < n_floats; ++i) {
        if(buf1[i] != buf2[i])
            count++;
    }
    auto end_compare = std::chrono::high_resolution_clock::now();

    // Compute throughput
    std::chrono::duration<double> load_time = end_load - start_load;
    std::chrono::duration<double> compare_time = end_compare - start_compare;
    double throughput = (double(dsize*2) / (1024.0 * 1024.0 * 1024.0)) / load_time.count();

    // Log result
    benchmark_stream << n_segs << ","
                        << chunk_size << ","
                        << dsize << ","
                        << load_time.count() << ","
                        << compare_time.count() << ","
                        << throughput << "\n";
 
    benchmark_stream.close();
    close(fd0);
    close(fd1);
}

int main(int argc, char* argv[]) {
    if (argc < 6) {
        std::cerr << "Usage: " << argv[0] << " <file1> <file2> <chunk_size> <nchunks> <outfile> [--rand] [--interleaved]\n";
        return 1;
    }
    std::string fname1 = argv[1];
    std::string fname2 = argv[2];
    size_t chunk_size = std::stoull(argv[3]);
    size_t nsegs = std::stoull(argv[4]);
    std::string log_file = argv[5];
    bool sequential_offt = true;
    bool grouped = true;
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--rand") {
            sequential_offt = false;
        }
	if (std::string(argv[i]) == "--interleaved") {
            grouped = false;
        }
    }
    size_t dsize = chunk_size * nsegs;

    try {
        std::cout << "Reading from " << dsize << " B with Csize = " << chunk_size
                    << " Sequential: " << sequential_offt << ", Grouped = " << grouped << std::endl;
        benchmark_liburing_reads(fname1, fname2, dsize, chunk_size, sequential_offt, grouped, log_file);
    } catch (const std::exception& e) {
        std::cerr << "Benchmark failed: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
