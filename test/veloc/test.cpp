#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <vector>
#include <iostream>
#include <fcntl.h>
#include <fstream>
#include <map>

void read_chkp(const std::string &filename, std::vector<uint8_t> &buffer) {
    std::ifstream basefile;
    basefile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    basefile.open(filename, std::ifstream::in | std::ifstream::binary);
    printf("Here1\n");
    int id;
    size_t num_regions, region_size, expected_size = 0;
    std::map<int, size_t> region_info;
    printf("Here2\n");
    basefile.read(reinterpret_cast<char *>(&num_regions), sizeof(size_t));
    for (uint32_t i = 0; i < num_regions; i++) {
        basefile.read(reinterpret_cast<char *>(&id), sizeof(int));
        basefile.read(reinterpret_cast<char *>(&region_size), sizeof(size_t));
        region_info.insert(std::make_pair(id, region_size));
        expected_size += region_size;
    }
    printf("Here3\n");
    size_t header_size = basefile.tellg();
    basefile.seekg(0, basefile.end);
    size_t file_size = static_cast<size_t>(basefile.tellg()) - header_size;
    if (file_size != expected_size) {
        std::cerr << "File size " << file_size
                  << " does not match expected size " << expected_size
                  << std::endl;
    }
    buffer.resize(expected_size);
    basefile.seekg(header_size);
    printf("Here4\n");
    basefile.read(reinterpret_cast<char *>(buffer.data()), expected_size);
    basefile.close();
}

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

int main(int argc, char **argv) {
    
    std::string filename = argv[1];

    // off_t file_size;
    // get_file_size(filename, &file_size);
    // size_t data_size = static_cast<size_t>(file_size);

    std::vector<uint8_t> local_data;
    read_chkp(filename, local_data);
    std::cout << "Done" << std::endl;

}