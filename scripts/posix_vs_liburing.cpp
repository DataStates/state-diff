#include "liburing_reader.hpp"
#include "stdio.h"
#include <chrono>
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
    liburing_io_reader_t reader_cur(ref_file);

    std::vector<uint8_t> data0_h;
    size_t data_len = 0;
    off_t filesize;
    get_file_size(ref_file, &filesize);
    data_len = static_cast<size_t>(filesize);
    if(data0_h.size() < data_len) {
        data0_h.resize(data_len);
    }
    printf("Data length = %zu; Reader size = %zu\n", data_len, reader_cur.size());

    int fd0 = open(ref_file.c_str(), O_RDONLY, 0644);
    if (fd0 == -1) {
        FATAL("cannot open " << ref_file << ", error = " << strerror(errno));
    }
    size_t transferred = 0, remaining = data_len;
    while (remaining > 0) {
        auto ret = read(fd0, data0_h.data() + transferred, remaining);
        if (ret < 0)
            FATAL("cannot read " << data_len << " bytes from " << ref_file
                                << " , error = " << std::strerror(errno));
        remaining -= ret;
        transferred += ret;
    }
    fsync(fd0);
    close(fd0);
    float *reader_buffer_if = (float*) data0_h.data();
    // uint8_t *reader_buffer_if = (uint8_t*) data0_h.data();

    // std::vector<uint8_t> buffer(data_len, 0);
    // std::vector<segment_t> segments(1);
    // segment_t seg;
    // seg.buffer = buffer.data();
    // seg.offset = 0;
    // seg.size = data_len;
    // segments[0] = seg;
    // reader_cur.enqueue_reads(segments);
    // reader_cur.wait_all();
    // float *reader_buffer = (float*)segments[0].buffer;

    std::vector<uint8_t> buffer(data_len, 0);
    size_t buffer_size = 2*1024*1024;
    size_t n_iter = data_len/buffer_size;
    if(n_iter * buffer_size < data_len)
        n_iter += 1;
    std::vector<segment_t> segments(n_iter);

    for(size_t i = 0; i < n_iter; i++) {
        segment_t seg;
        seg.buffer = buffer.data()+(buffer_size*i);
        // seg.buffer = static_cast<uint8_t*>(std::malloc(data_len));
        // posix_memalign((void**)&seg.buffer, 4096, data_len);
        seg.offset = buffer_size*i;
        seg.size = buffer_size;
        if(seg.offset+seg.size > data_len)
            seg.size = data_len - seg.offset;
        segments[i] = seg;
    }
    reader_cur.enqueue_reads(segments);
    reader_cur.wait_all();
    float *reader_buffer = (float*)segments[0].buffer;

    // uint8_t *reader_buffer = read_file_syscall(ref_file.c_str(), data_len);

    if (!reader_buffer) {
        std::cerr << "Null pointer access!" << std::endl;
    }

    // Compare read data
    for (size_t i = 0; i < data_len/sizeof(float); i++) {
        // std::cout << "Loader at index " << i << " = " << reader_buffer[i] << " vs real val = " << reader_buffer_if[i] << "\n";
        if (reader_buffer_if[i] != reader_buffer[i]) {
            std::cout << "Mismatch at index " << i << ": ifstream = " << reader_buffer_if[i]
                    << ", loader = " << reader_buffer[i] << "\n";
            return -1;
        }
    }
    std::cout << "Bingo! No Mismatch \n";
    return 0;
}
