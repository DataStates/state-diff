#ifndef __IO_READER_HPP
#define __IO_READER_HPP

#include "common/debug.hpp"
#include "common/io_utils.hpp"
#include <cassert>
#include <cerrno>
#include <chrono>
#include <fcntl.h>
#include <future>
#include <stdlib.h>
#include <string>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <type_traits>
#include <unistd.h>
#include <vector>
#include "common/segment.hpp"

class base_io_reader_t {
  public:
    base_io_reader_t() = default; // default
    base_io_reader_t(std::string& name); // open file
    virtual ~base_io_reader_t() = default; // default
    virtual int enqueue_reads(const std::vector<segment_t>& segments) = 0; // Add segments to read queue
    virtual int wait(size_t id) = 0; // Wait for id to finish
    virtual int wait_all() = 0; // wait for all pending reads to finish
    virtual size_t wait_any() = 0; // wait for any available read to finish
    virtual size_t size() = 0; // Get the size of the corresponding file
};

#endif   // __IO_READER_HPP
