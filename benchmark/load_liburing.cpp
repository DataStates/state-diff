#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <scissor/io_reader.hpp>
#include <scissor/readers.hpp>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

static int get_file_size(const std::string& filename, off_t* size_out) {
    struct stat file_stat;
    if (stat(filename.c_str(), &file_stat) < 0) return -1;
    if (!S_ISREG(file_stat.st_mode)) return -1;
    *size_out = file_stat.st_size;
    return 0;
}

void oneforall(std::string &filename, size_t chunk_size) {

    // Get size of source file
    off_t filesize;
    get_file_size(filename, &filesize);
    size_t dsize = static_cast<size_t>(filesize);

    // Create a buffer to hold the data loaded
    std::vector<float> buffer(chunk_size / sizeof(float), 0);

    // Compute the number of segments to read the entire data
    int n_segs = dsize / chunk_size;
    if (n_segs * chunk_size < dsize)
        n_segs += 1;

    int reader_fd = open(filename.c_str(), O_RDONLY);
    if (reader_fd < 0) throw std::runtime_error("Cannot open: " + filename);

    scissor::LiburingReader reader;

    std::vector<scissor::Segment> segments(1);
    double total_time = 0;

    for (int i = 0; i < n_segs; i++) {
        // Handle last segment
        size_t worksize = chunk_size;
        if(i == n_segs - 1) {
            worksize = dsize - (chunk_size * i);
        }

        // Create request
        scissor::Segment seg{};
        seg.id     = 0;
        seg.fd     = reader_fd;
        seg.buffer = (uint8_t*)buffer.data();
        seg.offset = chunk_size * static_cast<size_t>(i);
        seg.size   = worksize;
        segments[0] = seg;

        // Issue and measure read latency
        auto start_load = std::chrono::high_resolution_clock::now();
        reader.enqueue_reads(segments);
        reader.wait_all();
        auto end_load = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> load_time = end_load - start_load;
        total_time += load_time.count();
    }
    close(reader_fd);
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

    // Create a buffer to hold the data loaded
    std::vector<float> buffer(dsize / sizeof(float), 0);

    // Compute the number of segments to read the entire data
    int n_segs = dsize / chunk_size;
    if (n_segs * chunk_size < dsize)
        n_segs += 1;

    int reader_fd = open(filename.c_str(), O_RDONLY);
    if (reader_fd < 0) throw std::runtime_error("Cannot open: " + filename);

    scissor::LiburingReader reader;

    std::vector<scissor::Segment> segments(n_segs);

    for (int i = 0; i < n_segs; i++) {
        scissor::Segment seg{};
        seg.id     = static_cast<size_t>(i);
        seg.fd     = reader_fd;
        seg.buffer = (uint8_t*)(buffer.data() + (chunk_size * i) / sizeof(float));
        seg.offset = chunk_size * static_cast<size_t>(i);
        seg.size   = (i == n_segs - 1) ? dsize - (chunk_size * i) : chunk_size;
        segments[i] = seg;
    }

    auto start_load = std::chrono::high_resolution_clock::now();
    reader.enqueue_reads(segments);
    reader.wait_all();
    auto end_load = std::chrono::high_resolution_clock::now();
    close(reader_fd);
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
