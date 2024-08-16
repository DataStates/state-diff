#ifndef __LIBURING_READER_HPP
#define __LIBURING_READER_HPP

#include <liburing.h>
#include "io_reader.hpp"

class liburing_io_reader_t : public base_io_reader_t {
  size_t fsize;
  int fd, num_threads = 4;
  size_t max_ring_size = 32768;
  std::string fname;
  std::vector<segment_t> reads;
  std::vector<bool> segment_status;
  std::vector<std::future<int>> futures;

  int read_data(size_t beg, size_t end);

  public:
    liburing_io_reader_t(); // default
    ~liburing_io_reader_t(); 
    liburing_io_reader_t(std::string& name); // open file
    int setup_context(uint32_t entries, struct io_uring* ring);
    int enqueue_reads(const std::vector<segment_t>& segments) override; // Add segments to read queue
    int wait(size_t id) override; // Wait for id to finish
    int wait_all() override; // wait for all pending reads to finish
};

// struct ring_state_t {
//     int id;
//     struct io_uring ring;
//     uint32_t ring_size=32768;
//     uint32_t num_cqe=0;
// };

struct read_call_data_t {
    uint64_t beg;
    uint64_t end;
    uint64_t fileoffset;
    uint64_t offset;
};

template<typename DataType>
class liburing_reader_t : public io_reader_t<DataType> { 

    size_t *host_offsets = NULL;
    size_t *file_offsets = NULL;
    size_t num_offsets = 0;
    size_t transferred_chunks = 0;
    size_t active_slice_len = 0;
    size_t transfer_slice_len = 0;
    size_t elem_per_slice = 0;
    size_t bytes_per_slice = 0;
    size_t elem_per_chunk = 0;
    size_t bytes_per_chunk = 0;
    size_t chunks_per_slice = 0;
    int num_threads = 16;
    bool async = true;
    bool full_transfer = true;
    bool done = false;
    std::string filename;
    DataType *active_buffer = NULL;
    DataType *transfer_buffer = NULL;
    DataType *host_buffer = NULL;
    std::vector<std::future<int>> futures;
    std::future<int> fut;
    std::vector<double> timer;
#ifdef __NVCC__
    cudaStream_t transfer_stream; // Stream for data transfers
#endif

    size_t filesize;
    int file;
    int max_ring_size=32768;
    int32_t num_cqe=0;
    struct io_uring ring;
    DataType *mmapped_file=NULL;

    int setup_context(uint32_t entries, struct io_uring* ring);
    int get_file_size(int fd, size_t *size);
    int queue_read(struct io_uring *ring, off_t size, off_t offset, void* dst_buf, uint64_t data_index);
    size_t* get_offset_ptr() const;
    int async_memcpy();
    int read_chunks(int tid, size_t beg, size_t end);
    size_t prepare_slice();

public:
    liburing_reader_t(size_t buff_len, const std::string& file_name, const size_t chunk_size, bool async_memcpy = true, bool transfer_all = true, int nthreads = 2);

    liburing_reader_t& operator=(const liburing_reader_t& other);
    liburing_reader_t(const liburing_reader_t&) = default;

    void start_stream(size_t* offset_ptr, const size_t n_offsets, const size_t chunk_size) override;
    size_t get_file_size() const override;
    DataType* next_slice() override;
    size_t get_slice_len() const override;
    std::vector<double> get_timer() override;
    void end_stream() override;

    size_t get_chunks_per_slice() {
        return chunks_per_slice;
    }

    ~liburing_reader_t() override;

};

#endif // __LIBURING_READER_HPP
