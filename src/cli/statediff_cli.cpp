#include "argparse/argparse.hpp"
#include "statediff.hpp"
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <string>
#include <unistd.h>
#include <vector>

using namespace statediff;

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

    size_t transferred = 0, remaining = data_len * sizeof(int);
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
    close(fd);
}

int
main(int argc, char **argv) {
    argparse::ArgumentParser program("statediff");

    program.add_argument("run0").help("Run0 Checkpoint data file").required();

    program.add_argument("run1").help("Run1 Checkpoint data file").required();

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

    program.add_argument("-b", "--buffer_len")
        .help("Buffer length")
        .default_value(size_t(1024 * 1024))
        .scan<'u', size_t>();

    program.add_argument("-s", "--start_level")
        .help("Start level")
        .default_value(size_t(13))
        .scan<'u', size_t>();

    program.add_argument("-a", "--approx_hash")
        .help("Approximate hashing")
        .default_value(true)
        .action(argparse::Action::StoreTrue);

    try {
        program.parse_args(argc, argv);
    } catch (const std::runtime_error &err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        return 1;
    }

    std::string file0 = program.get<std::string>("run0");
    std::string file1 = program.get<std::string>("run1");
    double error = program.get<double>("error");
    char dtype = program.get<char>("type");
    size_t chunk_size = program.get<size_t>("chunk_size");
    size_t buffer_len = program.get<size_t>("buffer_len");
    size_t start_level = program.get<size_t>("start_level");
    bool approx_hash = program.get<bool>("approx_hash");

    off_t filesize;
    get_file_size(file0, &filesize);
    size_t data_len = static_cast<size_t>(filesize);

    std::vector<uint8_t> data0(data_len), data1(data_len);
    read_file(file0, data0, data_len);
    read_file(file1, data1, data_len);

    client_t<int> client0(1, file0, data_len, error, dtype, chunk_size,
                          buffer_len, start_level, approx_hash);
    client0.create(data);

    client_t<int> client1(2, file1, data_len, error, dtype, chunk_size,
                          buffer_len, start_level, approx_hash);

    client0.compare_with(client1);

    if (client0.get_num_changes() == 0) {
        std::cout << "SUCCESS::Files " << file0 << " and " << file1
                  << " are within error tolerance." << std::endl;
    } else {
        std::cout << "FAILURE::Files " << file0 << " and " << file1
                  << " are NOT within error tolerance." << std::endl;
    }

    return 0;
}
