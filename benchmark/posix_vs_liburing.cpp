#include <scissor/io_reader.hpp>
#include <scissor/readers.hpp>
#include "stdio.h"
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

int get_file_size(const std::string& filename, off_t *size) {
    struct stat st;

    if (stat(filename.c_str(), &st) < 0 )
        return -1;
    if(S_ISREG(st.st_mode)) {
        *size = st.st_size;
        return 0;
    }
    return -1;
}

uint8_t* read_file_syscall(const char* filename, size_t& num_bytes) {
    int fd = open(filename, O_RDONLY);
    if (fd == -1) {
        perror("open");
        return nullptr;
    }

    struct stat sb;
    if (fstat(fd, &sb) == -1) {
        perror("fstat");
        close(fd);
        return nullptr;
    }

    size_t file_size = sb.st_size;
    num_bytes = file_size;

    uint8_t* buffer = static_cast<uint8_t*>(std::malloc(file_size));
    if (!buffer) {
        std::cerr << "Memory allocation failed for " << file_size << " bytes." << std::endl;
        perror("malloc failed");
        close(fd);
        return nullptr;
    }

    size_t total_read = 0;
    while (total_read < file_size) {
        ssize_t bytes_read = read(fd, buffer + total_read, file_size - total_read);
        if (bytes_read <= 0) {
            perror("read");
            std::free(buffer);
            close(fd);
            return nullptr;
        }
        total_read += bytes_read;
    }

    close(fd);
    return buffer;
}


int
main(int argc, char **argv) {
    std::string ref_file = argv[1];

    std::vector<uint8_t> data0_h;
    size_t data_len = 0;
    off_t filesize;
    get_file_size(ref_file, &filesize);
    data_len = static_cast<size_t>(filesize);
    if(data0_h.size() < data_len) {
        data0_h.resize(data_len);
    }
    printf("Data length = %zu\n", data_len);

    int fd0 = open(ref_file.c_str(), O_RDONLY, 0644);
    if (fd0 == -1) {
        std::cerr << "cannot open " << ref_file << ", error = " << strerror(errno) << std::endl;
        std::abort();
    }
    size_t transferred = 0, remaining = data_len;
    while (remaining > 0) {
        auto ret = read(fd0, data0_h.data() + transferred, remaining);
        if (ret < 0) {
            std::cerr << "cannot read " << data_len << " bytes from " << ref_file
                      << " , error = " << std::strerror(errno) << std::endl;
            std::abort();
        }
        remaining -= ret;
        transferred += ret;
    }
    fsync(fd0);
    close(fd0);
    float *reader_buffer_if = (float*) data0_h.data();

    std::vector<uint8_t> buffer(data_len, 0);
    size_t buffer_size = 2*1024*1024;
    size_t n_iter = data_len/buffer_size;
    if(n_iter * buffer_size < data_len)
        n_iter += 1;
    std::vector<scissor::Segment> segments(n_iter);

    // Open file for liburing reads via scissor
    int uring_fd = open(ref_file.c_str(), O_RDONLY);
    if (uring_fd == -1) {
        std::cerr << "cannot open " << ref_file << " for liburing, error = " << strerror(errno) << std::endl;
        std::abort();
    }

    for(size_t i = 0; i < n_iter; i++) {
        scissor::Segment seg{};
        seg.id     = i;
        seg.fd     = uring_fd;
        seg.buffer = buffer.data()+(buffer_size*i);
        seg.offset = buffer_size*i;
        seg.size   = buffer_size;
        if(seg.offset+seg.size > data_len)
            seg.size = data_len - seg.offset;
        segments[i] = seg;
    }
    scissor::LiburingReader reader_cur;
    reader_cur.enqueue_reads(segments);
    reader_cur.wait_all();
    close(uring_fd);
    float *reader_buffer = (float*)segments[0].buffer;

    if (!reader_buffer) {
        std::cerr << "Null pointer access!" << std::endl;
    }

    // Compare read data
    for (size_t i = 0; i < data_len/sizeof(float); i++) {
        if (reader_buffer_if[i] != reader_buffer[i]) {
            std::cout << "Mismatch at index " << i << ": ifstream = " << reader_buffer_if[i]
                    << ", loader = " << reader_buffer[i] << "\n";
            return -1;
        }
    }
    std::cout << "Bingo! No Mismatch \n";
    return 0;
}
