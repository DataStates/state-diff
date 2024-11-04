#include <Kokkos_Core.hpp>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <string>
#include <unistd.h>
#include <vector>
//#define DEBUG
//#define STDOUT
#include "argparse/argparse.hpp"
#include "statediff.hpp"
#include "io_reader.hpp"

using namespace state_diff;

int
get_file_size(const std::string &filename, off_t *size) {
    struct stat st;

    if (stat(filename.c_str(), &st) < 0)
        return -1;
    if (S_ISREG(st.st_mode)) {
        *size = st.st_size;
        return 0;
    }
    return -1;
}

void
read_file(const std::string &filename, std::vector<uint8_t> &data,
          size_t data_len) {
    int fd = open(filename.c_str(), O_RDONLY);
    if (fd == -1) {
        std::cerr << "Cannot open " << filename
                  << ", error = " << std::strerror(errno) << std::endl;
        exit(1);
    }

    size_t transferred = 0, remaining = data_len;
    while (remaining > 0) {
        ssize_t ret = read(fd, data.data() + transferred, remaining);
        if (ret < 0) {
            std::cerr << "Cannot read from " << filename
                      << ", error = " << std::strerror(errno) << std::endl;
            close(fd);
            exit(1);
        }
        remaining -= ret;
        transferred += ret;
    }
    fsync(fd);
    posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
    close(fd);
}

int
main(int argc, char **argv) {
    Kokkos::initialize(argc, argv);
    {
    argparse::ArgumentParser program("statediff");

    program.add_argument("run0")
        .help("First file to compare").required();

    program.add_argument("run1")
        .help("Second file to compare").required();

    program.add_argument("--help")
        .help("Print help message")
        .default_value(false)
        .implicit_value(true);
    
    program.add_argument("--verbose")
        .help("Verbose output")
        .default_value(false)
        .implicit_value(true);

    program.add_argument("-e", "--error")
        .help("Error tolerance")
        .default_value(static_cast<double>(0.0f))
        .scan<'g', double>();

    program.add_argument("-t", "--type")
        .required()
        .help("Data type")
        .default_value(std::string("byte"))
        .choices("byte", "float", "double");

    program.add_argument("-c", "--chunk_size")
        .help("Chunk size")
        .default_value(size_t(4096))
        .scan<'u', size_t>();

    program.add_argument("-b", "--buffer-len")
        .help("Buffer length")
        .default_value(size_t(1024 * 1024))
        .scan<'u', size_t>();

    program.add_argument("-s", "--start-level")
        .help("Start level")
        .default_value(size_t(13))
        .scan<'u', size_t>();

    program.add_argument("-a", "--approx-hash")
        .help("Approximate hashing")
        .default_value(false)
        .implicit_value(true);

    program.add_argument("-n", "--num-threads")
        .help("Number of threads for reading and comparing data")
        .default_value(1)
        .scan<'u', size_t>();
    
    program.add_argument("--new-reader")
        .help("Use new readers for I/O")
        .default_value(false)
        .implicit_value(true);

    try {
        program.parse_args(argc, argv);
    } catch (const std::runtime_error &err) {
        std::cerr << err.what() << std::endl;
        std::cout << program;
        return 1;
    }

    bool print_help = program.get<bool>("help");
    if(print_help) {
        Kokkos::finalize();
        std::cout << program;
        return 0;
    }
    std::string file0 = program.get<std::string>("run0");
    std::string file1 = program.get<std::string>("run1");
    std::string dtype = program.get<std::string>("type");
    double error = program.get<double>("error");
    size_t chunk_size = program.get<size_t>("chunk_size");
    size_t buffer_len = program.get<size_t>("buffer-len");
    size_t start_level = program.get<size_t>("start-level");
    size_t num_threads = program.get<size_t>("num-threads");
    bool approx_hash = program.get<bool>("approx-hash");
    bool use_new_reader = program.get<bool>("new-reader");
    bool verbose = program.get<bool>("verbose");

    if(verbose) {
        std::cout << "File 0: " << file0 << std::endl;
        std::cout << "File 1: " << file1 << std::endl;
        std::cout << "Error: " << error << std::endl;
        std::cout << "Data type: " << dtype << std::endl;
        std::cout << "Chunk size: " << chunk_size << std::endl;
        std::cout << "Buffer length: " << buffer_len << std::endl;
        std::cout << "Start level: " << start_level << std::endl;
        std::cout << "Number of threads: " << num_threads << std::endl;
        std::cout << "Approx hash: " << approx_hash << std::endl;
    }

    off_t filesize;
    get_file_size(file0, &filesize);
    size_t data_len = static_cast<size_t>(filesize);

    std::vector<uint8_t> data0(data_len), data1(data_len);
    std::vector<int> metadata0;
    read_file(file0, data0, data_len);
    read_file(file1, data1, data_len);

    size_t elem_per_buffer = buffer_len/sizeof(float);
    size_t elem_per_chunk = chunk_size/sizeof(float);
    liburing_reader_t<float> reader0(file0, elem_per_buffer, elem_per_chunk, 
                                     false, false, num_threads);
    liburing_reader_t<float> reader1(file1, elem_per_buffer, elem_per_chunk, 
                                     false, false, num_threads);

    // Create client for file 0
    client_t<float, liburing_reader_t> client0(1, reader0, data_len, error, 
                          dtype[0], chunk_size, start_level, approx_hash);

    // Create merkle tree
    client0.create((uint8_t *) data0.data());

    // Create client for file 1
    client_t<float, liburing_reader_t> client1(2, reader1, data_len, error, 
                          dtype[0], chunk_size, start_level, approx_hash);

    // Create merkle tree
    client1.create((uint8_t *) data1.data());

    // Compare files
    if(use_new_reader) {
        client0.compare_with_new_reader(client1);
    } else {
        client0.compare_with(client1);
    }

    if (client0.get_num_changes() == 0) {
        std::cout << "SUCCESS::Files " << file0 << " and " << file1
                  << " are within error tolerance." << std::endl;
    } else {
        std::cout << "FAILURE::Files " << file0 << " and " << file1
                  << " are NOT within error tolerance. Found " << client0.get_num_changes() 
                  << " changes." << std::endl;
    }

    if(verbose) {
        std::cout << "Number of changed elements: " 
                  << client0.get_num_changes() << std::endl;
        std::cout << "Tree comparison time: " 
                  << client0.get_tree_comparison_time() << std::endl;
        std::cout << "Direct comparison time: " 
                  << client0.get_compare_time() << std::endl;
    }

    }
    Kokkos::finalize();
    return 0;
}
