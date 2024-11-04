#ifndef __POSIX_READER_HPP
#define __POSIX_READER_HPP

//#define __DEBUG
#include "io_reader.hpp"
#include <unordered_set>
#include <chrono>

class posix_io_reader_t : public base_io_reader_t {
  size_t fsize;
  int fd, num_threads = 1;
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
};

struct buff_state_t {
    uint8_t *buff;
    size_t size;
};

template<typename DataType>
class posix_reader_t : public io_reader_t<DataType> {
    
public:
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
    uint32_t num_threads = 16;
    bool async = true;
    bool full_transfer = false;
    bool done = false;
    DataType *active_buffer = NULL;
    DataType *transfer_buffer = NULL;
    DataType *host_buffer = NULL;
    std::vector<std::future<int>> futures;
    std::future<int> fut;
    std::vector<double> timer;
    std::string filename;
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
    posix_reader_t(const std::string& file_name, size_t buff_len, size_t chunk_size, bool async_memcpy = true, bool transfer_all = true, int nthreads = 1);

    posix_reader_t& operator=(const posix_reader_t& other);
    posix_reader_t(const posix_reader_t&);

    void start_stream(size_t* offset_ptr, const size_t n_offsets, const size_t chunk_size) override;
    size_t get_file_size() const override;
    DataType* next_slice() override;
    size_t get_slice_len() const override;
    std::vector<double> get_timer() override;
    void end_stream() override;

    size_t get_chunks_per_slice() {
        return chunks_per_slice;
    }

//    ~posix_reader_t() override;
    ~posix_reader_t();

};

template <typename DataType>
posix_reader_t<DataType>::posix_reader_t(const std::string &file_name,
                                         size_t buff_len,
                                         size_t chunk_size,
                                         bool async_memcpy, bool transfer_all,
                                         int nthreads)
    : elem_per_slice(buff_len), filename(file_name), elem_per_chunk(chunk_size),
      async(async_memcpy), full_transfer(transfer_all), num_threads(nthreads) {

    timer = std::vector<double>(3, 0.0);
    bytes_per_slice = elem_per_slice * sizeof(DataType);
    active_slice_len = elem_per_slice;
    transfer_slice_len = elem_per_slice;
    // Calculate useful values
    if (full_transfer) {
        elem_per_chunk = elem_per_slice;
        bytes_per_chunk = bytes_per_slice;
        chunks_per_slice = 1;
    } else {
        bytes_per_chunk = elem_per_chunk * sizeof(DataType);
        chunks_per_slice = elem_per_slice / elem_per_chunk;
    }

    file_buffer = map_file(filename);
    if (transfer_all) {
        madvise(file_buffer.buff, file_buffer.size, MADV_SEQUENTIAL);
    } else {
        madvise(file_buffer.buff, file_buffer.size, MADV_RANDOM);
    }
    ASSERT(file_buffer.size % sizeof(DataType) == 0);
#ifdef __NVCC__
    active_buffer = device_alloc<DataType>(bytes_per_slice);
    transfer_buffer = device_alloc<DataType>(bytes_per_slice);
    host_buffer = host_alloc<DataType>(bytes_per_slice);
    size_t max_offsets = file_buffer.size / bytes_per_chunk;
    if (bytes_per_chunk * max_offsets < file_buffer.size)
        max_offsets += 1;
    host_offsets = host_alloc<size_t>(sizeof(size_t) * max_offsets);
    gpuErrchk(cudaStreamCreate(&transfer_stream));
#else
    if (!full_transfer) {
        active_buffer = device_alloc<DataType>(bytes_per_slice);
        transfer_buffer = device_alloc<DataType>(bytes_per_slice);
    }
    host_buffer = transfer_buffer;
#endif

    DEBUG_PRINT("Constructor: Filename: %s\n", filename.c_str());
    DEBUG_PRINT("Constructor: File size: %zu\n", file_buffer.size);
    DEBUG_PRINT("Constructor: Full transfer? %d\n", full_transfer);
    DEBUG_PRINT("Constructor: Async transfer? %d\n", async);
    DEBUG_PRINT("Constructor: Elem per chunk: %zu\n", elem_per_chunk);
    DEBUG_PRINT("Constructor: Bytes per chunk: %zu\n", bytes_per_chunk);
    DEBUG_PRINT("Constructor: Elem per slice: %zu\n", elem_per_slice);
    DEBUG_PRINT("Constructor: Bytes per slice: %zu\n", bytes_per_slice);
    DEBUG_PRINT("Constructor: Active slice len: %zu\n", active_slice_len);
    DEBUG_PRINT("Constructor: Transfer slice len: %zu\n", transfer_slice_len);
}

template <typename DataType>
buff_state_t
posix_reader_t<DataType>::map_file(const std::string &fn) {
    int fd = open(fn.c_str(), O_RDONLY | O_DIRECT);
    if (fd == -1)
        FATAL("cannot open " << fn << ", error = " << std::strerror(errno));
    size_t size = lseek(fd, 0, SEEK_END);
    uint8_t *buff = (uint8_t *) mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (buff == MAP_FAILED)
        FATAL("cannot mmap " << fn << ", error (" << errno << ") = " << std::strerror(errno));
    return buff_state_t{buff, size};
}

template <typename DataType>
size_t
posix_reader_t<DataType>::get_chunk(const buff_state_t &ckpt, size_t offset,
                                    DataType **ptr) {
    size_t byte_offset = offset * sizeof(DataType);
    if(byte_offset >= ckpt.size) {
      printf("%zu * %zu < %zu\n", offset, sizeof(DataType), ckpt.size);
    }
    ASSERT(byte_offset < ckpt.size);
    size_t ret = byte_offset + bytes_per_chunk >= ckpt.size
                     ? ckpt.size - byte_offset
                     : bytes_per_chunk;
    ASSERT(ret % sizeof(DataType) == 0);
    *ptr = (DataType *) (ckpt.buff + byte_offset);
    return ret / sizeof(DataType);
}

template <typename DataType>
size_t *
posix_reader_t<DataType>::get_offset_ptr() const {
    return file_offsets;
}

template <typename DataType>
int
posix_reader_t<DataType>::async_memcpy() {
    for (uint32_t i = 0; i < futures.size(); i++) {
        int res = futures[i].get();
        if (res < 0)
            fprintf(stderr, "read_chunks: %s\n", strerror(-res));
    }
    futures.clear();

    Timer::time_point beg_copy = Timer::now();
#ifdef __NVCC__
    if (async) {
        gpuErrchk(cudaMemcpyAsync(transfer_buffer, host_buffer,
                                  transfer_slice_len * sizeof(DataType),
                                  cudaMemcpyHostToDevice, transfer_stream));
    } else {
        gpuErrchk(cudaMemcpy(transfer_buffer, host_buffer,
                             transfer_slice_len * sizeof(DataType),
                             cudaMemcpyHostToDevice));
    }
#endif
    Timer::time_point end_copy = Timer::now();
    timer[2] +=
        std::chrono::duration_cast<Duration>(end_copy - beg_copy).count();
    return 0;
}

template <typename DataType>
int
posix_reader_t<DataType>::read_chunks(int tid, size_t beg, size_t end) {
    for (size_t i = beg; i < end; i++) {
        if (transferred_chunks + i < num_offsets) {
            DataType *chunk;
            size_t len = get_chunk(
                file_buffer,
                host_offsets[transferred_chunks + i] * elem_per_chunk, &chunk);
            if (len != elem_per_chunk)
                transfer_slice_len -= elem_per_chunk - len;
            assert(len <= elem_per_chunk);
            assert(i * elem_per_chunk + len * sizeof(DataType) <=
                   bytes_per_slice);
            assert(host_buffer != NULL);
            assert((size_t) host_buffer + i * elem_per_chunk + len <=
                   (size_t) host_buffer + elem_per_slice);
            if (len > 0)
                memcpy(host_buffer + i * elem_per_chunk, chunk,
                       len * sizeof(DataType));
        }
    }
    return 0;
}

template <typename DataType>
size_t
posix_reader_t<DataType>::prepare_slice() {
    if (full_transfer) {
        // Offset into file
        Timer::time_point beg_read = Timer::now();
        size_t offset = host_offsets[transferred_chunks] * elem_per_slice;
#ifdef __NVCC__
        futures.push_back(std::async(std::launch::async | std::launch::deferred,
                                     &posix_reader_t::read_chunks, this, 0, 0,
                                     1));
        fut = std::async(std::launch::async | std::launch::deferred,
                         &posix_reader_t::async_memcpy, this);
#endif
        transfer_slice_len = get_chunk(file_buffer, offset, &transfer_buffer);
        Timer::time_point end_read = Timer::now();
        timer[1] +=
            std::chrono::duration_cast<Duration>(end_read - beg_read).count();
    } else {
        // Calculate number of elements to read
        transfer_slice_len = elem_per_slice;
        size_t elements_read = elem_per_chunk * transferred_chunks;
        if (elements_read + transfer_slice_len > num_offsets * elem_per_chunk) {
            transfer_slice_len = num_offsets * elem_per_chunk - elements_read;
        }
        Timer::time_point beg_read = Timer::now();

        size_t chunks_left = chunks_per_slice;
        if (transferred_chunks + chunks_left > num_offsets) {
            chunks_left = num_offsets - transferred_chunks;
        }

        size_t chunks_per_thread = chunks_left / num_threads;
        if (chunks_per_thread * num_threads < chunks_left)
            chunks_per_thread += 1;

        int active_threads =
            num_threads > chunks_left ? chunks_left : num_threads;
        for (int tid = 0; tid < active_threads; tid++) {
            size_t beg = tid * chunks_per_thread;
            size_t end = (tid + 1) * chunks_per_thread;
            if (end > chunks_left)
                end = chunks_left;
            futures.push_back(
                std::async(std::launch::async | std::launch::deferred,
                           &posix_reader_t::read_chunks, this, tid, beg, end));
        }
        Timer::time_point end_read = Timer::now();
        timer[1] +=
            std::chrono::duration_cast<Duration>(end_read - beg_read).count();
    }
    fut = std::async(std::launch::async | std::launch::deferred,
                     &posix_reader_t::async_memcpy, this);
    return transfer_slice_len;
}

template <typename DataType>
posix_reader_t<DataType>::posix_reader_t(const posix_reader_t& other) {
    file_buffer = other.file_buffer;
    host_offsets = other.host_offsets;
    file_offsets = other.file_offsets;
    num_offsets = other.num_offsets;
    transferred_chunks = other.transferred_chunks;
    active_slice_len = other.active_slice_len;
    transfer_slice_len = other.transfer_slice_len;
    elem_per_slice = other.elem_per_slice;
    bytes_per_slice = other.bytes_per_slice;
    elem_per_chunk = other.elem_per_chunk;
    bytes_per_chunk = other.bytes_per_chunk;
    chunks_per_slice = other.chunks_per_slice;
    async = other.async;
    full_transfer = other.full_transfer;
    done = other.done;
    filename = other.filename;
    active_buffer = other.active_buffer;
    transfer_buffer = other.transfer_buffer;
    host_buffer = other.host_buffer;
    num_threads = other.num_threads;
    timer = other.timer;
#ifdef __NVCC__
    transfer_stream = other.transfer_stream;
#endif
}

template <typename DataType>
posix_reader_t<DataType> &
posix_reader_t<DataType>::operator=(const posix_reader_t &other) {
    if (this == &other) {
        return *this;
    }
    file_buffer = other.file_buffer;
    host_offsets = other.host_offsets;
    file_offsets = other.file_offsets;
    num_offsets = other.num_offsets;
    transferred_chunks = other.transferred_chunks;
    active_slice_len = other.active_slice_len;
    transfer_slice_len = other.transfer_slice_len;
    elem_per_slice = other.elem_per_slice;
    bytes_per_slice = other.bytes_per_slice;
    elem_per_chunk = other.elem_per_chunk;
    bytes_per_chunk = other.bytes_per_chunk;
    chunks_per_slice = other.chunks_per_slice;
    async = other.async;
    full_transfer = other.full_transfer;
    done = other.done;
    filename = other.filename;
    active_buffer = other.active_buffer;
    transfer_buffer = other.transfer_buffer;
    host_buffer = other.host_buffer;
    num_threads = other.num_threads;
    timer = other.timer;
#ifdef __NVCC__
    transfer_stream = other.transfer_stream;
#endif
    return *this;
}

template <typename DataType> posix_reader_t<DataType>::~posix_reader_t() {
    if (done && (file_buffer.buff != NULL)) {
        munmap(file_buffer.buff, file_buffer.size);
        file_buffer.buff = NULL;
    }
#ifdef __NVCC__
    if (done && (active_buffer != NULL)) {
        device_free<DataType>(active_buffer);
        active_buffer = NULL;
    }
    if (done && (transfer_buffer != NULL)) {
        device_free<DataType>(transfer_buffer);
        transfer_buffer = NULL;
    }
    if (done && (host_buffer != NULL)) {
        host_free<DataType>(host_buffer);
        host_buffer = NULL;
    }
    if (done && (transfer_stream != 0)) {
        gpuErrchk(cudaStreamDestroy(transfer_stream));
        transfer_stream = 0;
    }
    if (done && (host_offsets != NULL)) {
        host_free<size_t>(host_offsets);
        host_offsets = NULL;
    }
#else
    if (!full_transfer) {
        if (done && (active_buffer != NULL)) {
            device_free<DataType>(active_buffer);
            active_buffer = NULL;
        }
        if (done && (transfer_buffer != NULL)) {
            device_free<DataType>(transfer_buffer);
            transfer_buffer = NULL;
        }
    }
#endif
}

template <typename DataType>
void
posix_reader_t<DataType>::start_stream(size_t *offset_ptr,
                                       const size_t n_offsets,
                                       const size_t chunk_size) {
    transferred_chunks = 0;      // Initialize stream
    file_offsets = offset_ptr;   // Store pointer to device offsets
    num_offsets = n_offsets;     // Store number of offsets
    DEBUG_PRINT("Elem per chunk: %zu\n", elem_per_chunk);
    DEBUG_PRINT("Bytes per chunk: %zu\n", bytes_per_chunk);
    DEBUG_PRINT("Elem per slice: %zu\n", elem_per_slice);
    DEBUG_PRINT("Bytes per slice: %zu\n", bytes_per_slice);
    DEBUG_PRINT("Chunks per slice: %zu\n", chunks_per_slice);
    DEBUG_PRINT("Num offsets: %zu\n", num_offsets);

    // Copy offsets to device if necessary
#ifdef __NVCC__
    gpuErrchk(cudaMemcpy(host_offsets, offset_ptr, n_offsets * sizeof(size_t),
                         cudaMemcpyDeviceToHost));
#else
    host_offsets = offset_ptr;
#endif
    // Start copying data into buffer
    Timer::time_point beg = Timer::now();
    prepare_slice();
    Timer::time_point end = Timer::now();
    timer[0] += std::chrono::duration_cast<Duration>(end - beg).count();
    return;
}

template <typename DataType>
size_t
posix_reader_t<DataType>::get_file_size() const {
    return file_buffer.size;
}

template <typename DataType>
DataType *
posix_reader_t<DataType>::next_slice() {
    int ret = fut.get();
    if (ret < 0)
        FATAL("failed to get future"
              << ", error = " << std::strerror(errno));
#ifdef __NVCC__
    if (async) {
        gpuErrchk(cudaStreamSynchronize(
            transfer_stream));   // Wait for slice to finish async copy
    }
#endif
    // Swap device buffers
    DataType *temp = active_buffer;
    active_buffer = transfer_buffer;
    transfer_buffer = temp;
#ifndef __NVCC__
    if (!full_transfer) {
        host_buffer = transfer_buffer;
    }
#endif
    active_slice_len = transfer_slice_len;
    // Update number of chunks transferred
    size_t nchunks = active_slice_len / elem_per_chunk;
    if (elem_per_chunk * nchunks < active_slice_len)
        nchunks += 1;
    transferred_chunks += nchunks;
    // Start reading next slice if there are any left
    if (transferred_chunks < num_offsets) {
        Timer::time_point beg = Timer::now();
        prepare_slice();
        Timer::time_point end = Timer::now();
        timer[0] += std::chrono::duration_cast<Duration>(end - beg).count();
    }
    return active_buffer;
}

template <typename DataType>
size_t
posix_reader_t<DataType>::get_slice_len() const {
    return active_slice_len;
}

template <typename DataType>
std::vector<double>
posix_reader_t<DataType>::get_timer() {
    return timer;
}

template <typename DataType>
void
posix_reader_t<DataType>::end_stream() {
    done = true;
    timer = {0.0};
}

#endif // __POSIX_READER_HPP
