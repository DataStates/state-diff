#ifndef __POSIX_READER_HPP
#define __POSIX_READER_HPP

//#define __DEBUG
#include "io_reader.hpp"
#include <unordered_set>
#include <chrono>

class posix_io_reader_t : public base_io_reader_t {
  size_t fsize;
  int fd, num_threads = 4;
  std::string fname;
  std::vector<segment_t> reads;
  std::vector<bool> segment_status;
  std::vector<std::future<int>> futures;

  int read_data(size_t beg, size_t end);

  public:
    posix_io_reader_t(); // default
    posix_io_reader_t(std::string& name); // open file
    ~posix_io_reader_t() override; 
    int enqueue_reads(const std::vector<segment_t>& segments) override; // Add segments to read queue
    int wait(size_t id) override; // Wait for id to finish
    int wait_all() override; // wait for all pending reads to finish
    size_t wait_any() override; // wait for any available read to finish
    size_t size() {
        return fsize;
    }
};
#endif // __POSIX_READER_HPP
