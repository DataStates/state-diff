#ifndef __POSIX_READER_HPP
#define __POSIX_READER_HPP

#include "io_reader.hpp"

struct buff_state_t {
    uint8_t *buff;
    size_t size;
};

template<typename DataType>
class posix_reader_t : public io_reader_t<DataType> {
    
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

    buff_state_t file_buffer;

    buff_state_t map_file(const std::string &fn);
    size_t get_chunk(const buff_state_t &ckpt, size_t offset, DataType** ptr);
    size_t* get_offset_ptr() const;
    int async_memcpy();
    int read_chunks(int tid, size_t beg, size_t end);
    size_t prepare_slice();

public:
    posix_reader_t(size_t buff_len, const std::string& file_name, const size_t chunk_size, bool async_memcpy = true, bool transfer_all = true, int nthreads = 2);

    posix_reader_t& operator=(const posix_reader_t& other);
    posix_reader_t(const posix_reader_t&) = default;

    void start_stream(size_t* offset_ptr, const size_t n_offsets, const size_t chunk_size) override;
    size_t get_file_size() const override;
    DataType* next_slice() override;
    size_t get_slice_len() const override;
    std::vector<double> get_timer() override;
    void end_stream() override;

    size_t get_chunks_per_slice() {
        return chunks_per_slice;
    }

    ~posix_reader_t() override;

};

#endif // __POSIX_READER_HPP
