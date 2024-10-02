#include <fstream>
#include <iostream>
#include <map>
#include <vector>
#include <cstdint>

int
main(int argc, char **argv) {
    std::ifstream basefile;

    basefile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    basefile.open(argv[1], std::ifstream::in | std::ifstream::binary);

    int id;
    size_t num_regions, region_size, expected_size = 0;
    std::map<int, size_t> region_info;

    basefile.read(reinterpret_cast<char *>(&num_regions), sizeof(size_t));
    for (uint32_t i = 0; i < num_regions; i++) {
        basefile.read(reinterpret_cast<char *>(&id), sizeof(int));
        basefile.read(reinterpret_cast<char *>(&region_size), sizeof(size_t));
        region_info.insert(std::make_pair(id, region_size));
        expected_size += region_size;
    }
    size_t header_size = basefile.tellg();
    basefile.seekg(0, basefile.end);
    size_t file_size = static_cast<size_t>(basefile.tellg()) - header_size;
    if (file_size != expected_size) {
        std::cerr << "File size " << file_size
                  << " does not match expected size " << expected_size
                  << std::endl;
    }

    std::vector<std::string> names = {"NumParticles", "X",   "Y",  "Z",
                                      "VX",           "VY",  "VZ", "Potential",
                                      "GlobalIds",    "Mask"};
    basefile.seekg(header_size);
    for (auto &e : region_info) {
        std::vector<uint8_t> buffer(e.second, 0);
        basefile.read(reinterpret_cast<char *>(buffer.data()), e.second);
        std::string outfile =
            std::string(argv[1]) + "." + names[e.first] + ".dat";
        std::ofstream outputfile(outfile,
                                 std::ofstream::out | std::ofstream::binary);
        outputfile.write(reinterpret_cast<char *>(buffer.data()), e.second);
        outputfile.close();
    }
    basefile.close();
}
