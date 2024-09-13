#ifndef __ADIOS_READER_HPP
#define __ADIOS_READER_HPP

#include "io_reader.hpp"
#include <adios2.h>

class adios_reader_t : public base_io_reader_t {

    std::string fname;
    std::vector<uint8_t> buffer;

    size_t read_all();

  public:
    adios_reader_t();                    // default
    adios_reader_t(std::string &name);   // open file
    ~adios_reader_t() override;
    adios_reader_t(const adios_reader_t &other);
    adios_reader_t &operator=(const adios_reader_t &other);
    int enqueue_reads(const std::vector<segment_t> &segments)
        override;                   // Add segments to read queue
    int wait(size_t id) override;   // Wait for id to finish
    int wait_all() override;        // wait for all pending reads to finish
    size_t wait_any() override;     // wait for any available read to finish
};

#endif   // __ADIOS_READER_HPP
