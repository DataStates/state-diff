#ifndef __LIBURING_READER_HPP
#define __LIBURING_READER_HPP

#include <liburing.h>
#include "io_reader.hpp"
#include <thread>
#include <queue>
#include <map>
#include <unordered_set>

struct file_info_t {
    int fd;
    size_t fsize;
};

class liburing_io_reader_t : public base_io_reader_t {
  const size_t MAX_RING_SIZE = 32768;
  size_t nrings = 1;
  //size_t req_submitted[2], req_completed[2];
  size_t *req_submitted, *req_completed;
  size_t fsize;
  std::map<std::string,file_info_t> file_info;
  int fd;
  std::queue<segment_t> submissions;
  std::unordered_set<size_t> completions;
  io_uring *ring;
  struct io_uring_cqe *cqe[32768], *cqe2[32768];
  std::mutex m;
  std::condition_variable cv;
  std::thread th;
  bool active, wait_all_mode;

  uint32_t request_completion();
  uint32_t request_submission();
  int io_thread();

  public:
  std::string fname;
    liburing_io_reader_t(); // default
    liburing_io_reader_t(std::string& name, size_t num_rings=1); // open file
    ~liburing_io_reader_t() override; 
    int enqueue_reads(const std::vector<segment_t>& segments) override; // Add segments to read queue
    int enqueue_reads(const std::string& fname, const std::vector<segment_t>& segments); // Add segments to read queue
    int wait(size_t id) override; // Wait for id to finish
    int wait_all() override; // wait for all pending reads to finish
    size_t wait_any() override; // wait for any available read to finish
    size_t size() {
        return file_info[fname].fsize;
    }
};
#endif // __LIBURING_READER_HPP
