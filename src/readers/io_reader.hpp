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

struct read_offsets_t {
    size_t *host_offsets;
    size_t *file_offsets;
};

struct slice_t {
    size_t active_len;
    size_t transfer_len;
    size_t elements;
    size_t chunks;
    size_t bytes;
};

struct chunk_t {
    size_t elements;
    size_t bytes;
};

template <typename DataType> class io_reader_t {

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
    cudaStream_t transfer_stream;   // Stream for data transfers
#endif

  public:
    io_reader_t() = default;
    io_reader_t(const std::string &name, size_t buffer_length,
                size_t chunk_size, bool async_memcpy, bool transfer_all,
                int nthreads);
//    virtual ~io_reader_t() = 0;

    virtual void start_stream(size_t *offset_ptr, const size_t n_offsets,
                              const size_t chunk_size) = 0;
    virtual size_t get_file_size() const = 0;
    virtual DataType *next_slice() = 0;
    virtual size_t get_slice_len() const = 0;
    virtual std::vector<double> get_timer() = 0;
    virtual void end_stream() = 0;
    virtual size_t get_chunks_per_slice() = 0;

  private:
    // Delete copy constructor and copy assignment operator to prevent copying
//    io_reader_t(const io_reader_t &) = delete;
//    io_reader_t &operator=(const io_reader_t &) = delete;
    io_reader_t(const io_reader_t &) = default;
    io_reader_t &operator=(const io_reader_t &) = default;
};

template <typename DataType>
io_reader_t<DataType>::io_reader_t(const std::string &name,
                                   size_t buffer_length, size_t chunk_size,
                                   bool async_memcpy, bool transfer_all,
                                   int nthreads)
    : filename(name), elem_per_slice(buffer_length), elem_per_chunk(chunk_size),
      async(async_memcpy), full_transfer(transfer_all), num_threads(nthreads) {}

#endif   // __IO_READER_HPP
