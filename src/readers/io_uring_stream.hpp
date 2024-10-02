#ifndef __IO_URING_STREAM_HPP
#define __IO_URING_STREAM_HPP

#include "common/debug.hpp"
#include "common/io_utils.hpp"
#include <cassert>
#include <cerrno>
#include <fcntl.h>
#include <future>
#include <liburing.h>
#include <stdlib.h>
#include <string>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <thread>
#include <type_traits>
#include <unistd.h>
#include <vector>

template <typename DataType> class io_uring_stream_t {

    struct ring_state_t {
        int id;
        struct io_uring ring;
        uint32_t ring_size = 32768;
        uint32_t num_cqe = 0;
    };

    struct read_call_data_t {
        uint64_t beg;
        uint64_t end;
        uint64_t fileoffset;
        uint64_t offset;
    };

    int setup_context(uint32_t entries, struct io_uring *ring);
    int get_file_size(int fd, size_t *size);
    int queue_read(struct io_uring *ring, off_t size, off_t offset,
                   void *dst_buf, uint64_t data_index);

  public:
    size_t *host_offsets = NULL,
           *file_offsets = NULL;   // Track where to make slice
    size_t num_offsets, transferred_chunks = 0;
    size_t active_slice_len = 0,
           transfer_slice_len = 0;   // Track the length of the inflight slice
    size_t elem_per_slice = 0,
           bytes_per_slice = 0;   // Max number of elements or bytes per slice
    size_t elem_per_chunk = 0,
           bytes_per_chunk = 0;   // Number of elements or bytes per chunk
    size_t chunks_per_slice = 0;
    bool async = true, full_transfer = true, done = false;
    std::string fname;
    size_t filesize;
    int file;
    int num_threads = 16;
    int max_ring_size = 32768;
    int32_t num_cqe = 0;
    struct io_uring ring;
    DataType *active_buffer = NULL,
             *transfer_buffer = NULL;   // Convenient pointers
    DataType *host_buffer = NULL;
    std::vector<std::future<int>> futures;
    std::future<int> fut;
    std::vector<double> timer;
#ifdef __NVCC__
    cudaStream_t transfer_stream;   // Stream for data transfers
#endif

  public:
    io_uring_stream_t();
    io_uring_stream_t(std::string &file_name, const size_t chunk_size,
                      bool async_memcpy = true, bool transfer_all = false,
                      int nthreads = 2);
    io_uring_stream_t &operator=(const io_uring_stream_t &other);
    ~io_uring_stream_t();

    // Get slice length for active buffer
    size_t get_slice_len() const;
    size_t *get_offset_ptr() const;
    size_t get_file_size() const;
    int async_memcpy();
    int read_chunks(int tid, size_t beg, size_t end);
    size_t prepare_slice();
    // Start streaming data from Host to Device
    void start_stream(size_t *offset_ptr, size_t n_offsets, size_t chunk_size);
    // Get next slice of data on Device buffer
    DataType *next_slice();
    // Reset Host to Device stream
    void end_stream();
    std::vector<double> get_timer();

    size_t get_chunks_per_slice() { return chunks_per_slice; }
};

template <typename DataType>
int
io_uring_stream_t<DataType>::setup_context(uint32_t entries,
                                           struct io_uring *ring) {
    int ret;
    ret = io_uring_queue_init(entries, ring, 0);
    if (ret < 0) {
        fprintf(stderr, "queue_init: %s\n", strerror(-ret));
        return -1;
    }
    return 0;
}

template <typename DataType>
int
io_uring_stream_t<DataType>::get_file_size(int fd, size_t *size) {
    struct stat st;
    off_t fsize;

    if (fstat(fd, &st) < 0)
        return -1;
    if (S_ISREG(st.st_mode)) {
        fsize = st.st_size;
        *size = static_cast<size_t>(fsize);
        return 0;
    } else if (S_ISBLK(st.st_mode)) {
        unsigned long long bytes;

        if (ioctl(fd, BLKGETSIZE64, &bytes) != 0)
            return -1;

        *size = bytes;
        return 0;
    }
    return -1;
}

template <typename DataType>
int
io_uring_stream_t<DataType>::queue_read(struct io_uring *ring, off_t size,
                                        off_t offset, void *dst_buf,
                                        uint64_t data_index) {
    struct io_uring_sqe *sqe;

    sqe = io_uring_get_sqe(ring);
    if (!sqe) {
        fprintf(stderr, "Could not get sqe\n");
        return -1;
    }

    io_uring_sqe_set_data64(sqe, data_index);

    io_uring_prep_read(sqe, file, dst_buf, size, offset);
    return 0;
}

template <typename DataType> io_uring_stream_t<DataType>::io_uring_stream_t() {}

// Constructor for Host -> Device stream
template <typename DataType>
io_uring_stream_t<DataType>::io_uring_stream_t(std::string &file_name,
                                               const size_t chunk_size,
                                               bool async_memcpy,
                                               bool transfer_all,
                                               int nthreads) {
    num_threads = nthreads;
    timer = std::vector<double>(3, 0.0);
    full_transfer = transfer_all;
    async = async_memcpy;
    elem_per_slice =
        chunk_size * 2;   // adapted to have buff_len = chunk_size*2
    bytes_per_slice = elem_per_slice * sizeof(DataType);
    fname = file_name;   // Save file name
    active_slice_len = elem_per_slice;
    transfer_slice_len = elem_per_slice;

    // Calculate useful values
    if (full_transfer) {
        elem_per_chunk = elem_per_slice;
        bytes_per_chunk = bytes_per_slice;
        chunks_per_slice = 1;
    } else {
        elem_per_chunk = chunk_size;
        bytes_per_chunk = elem_per_chunk * sizeof(DataType);
        chunks_per_slice = elem_per_slice / elem_per_chunk;
    }

    // Open file
    file = open(fname.c_str(), O_RDONLY);
    if (file < 0) {
        fprintf(stderr, "Failed to open file %s\n", fname.c_str());
    }
    if (get_file_size(file, &filesize) != 0) {
        fprintf(stderr, "Failed to retrieve file size for %s\n", fname.c_str());
    }

#ifdef __NVCC__
    active_buffer = device_alloc<DataType>(bytes_per_slice);
    transfer_buffer = device_alloc<DataType>(bytes_per_slice);
    host_buffer = host_alloc<DataType>(bytes_per_slice);
    size_t max_offsets = filesize / bytes_per_chunk;
    if (bytes_per_chunk * max_offsets < filesize)
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

    // DEBUG_PRINT("Constructor: Filename: %s\n", fname.c_str());
    // DEBUG_PRINT("File size: %zu\n", filesize);
    // DEBUG_PRINT("Constructor: Full transfer? %d\n", full_transfer);
    // DEBUG_PRINT("Constructor: Async transfer? %d\n", async);
    // DEBUG_PRINT("Constructor: Elem per slice: %zu\n", elem_per_slice);
    // DEBUG_PRINT("Constructor: Bytes per slice: %zu\n", bytes_per_slice);
    // DEBUG_PRINT("Constructor: Active slice len: %zu\n", active_slice_len);
    // DEBUG_PRINT("Constructor: Transfer slice len: %zu\n", transfer_slice_len);
}

template <typename DataType>
io_uring_stream_t<DataType> &
io_uring_stream_t<DataType>::operator=(
    const io_uring_stream_t<DataType> &other) {
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
    fname = other.fname;
    filesize = other.filesize;
    file = other.file;
    num_threads = other.num_threads;
    max_ring_size = other.max_ring_size;
    num_cqe = other.num_cqe;
    ring = other.ring;
    active_buffer = other.active_buffer;
    transfer_buffer = other.transfer_buffer;
    host_buffer = other.host_buffer;
    timer = other.timer;
#ifdef __NVCC__
    transfer_stream = other.transfer_stream;
#endif
    return *this;
}

template <typename DataType> io_uring_stream_t<DataType>::~io_uring_stream_t() {
    if (done && (file != -1)) {
        close(file);
        file = -1;
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

// Get slice length for active buffer
template <typename DataType>
size_t
io_uring_stream_t<DataType>::get_slice_len() const {
    return active_slice_len;
}

template <typename DataType>
size_t *
io_uring_stream_t<DataType>::get_offset_ptr() const {
    return file_offsets;
}

template <typename DataType>
size_t
io_uring_stream_t<DataType>::get_file_size() const {
    return filesize;
}

template <typename DataType>
int
io_uring_stream_t<DataType>::async_memcpy() {
    for (size_t i = 0; i < futures.size(); i++) {
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
io_uring_stream_t<DataType>::read_chunks(int tid, size_t beg, size_t end) {
    // Create submission queue
    io_uring ring;
    int ret = setup_context(max_ring_size, &ring);
    if (ret < 0) {
        FATAL("Failed to setup ring of size " << max_ring_size);
    }

    std::vector<read_call_data_t> read_data;

    // Fill submission queue with reads
    uint32_t num_cqe = 0;
    for (size_t i = beg; i < end; i++) {
        size_t fileoffset =
            host_offsets[transferred_chunks + i] * bytes_per_chunk;
        size_t len = bytes_per_chunk;
        if (fileoffset + len > filesize)
            len = filesize - fileoffset;
        read_call_data_t data;
        data.beg = beg;
        data.end = end;
        data.fileoffset = fileoffset;
        data.offset = i * elem_per_chunk;
        read_data.push_back(data);
        queue_read(&ring, len, fileoffset, host_buffer + i * elem_per_chunk, i);
        num_cqe++;
    }

    // Submit queue
    ret = io_uring_submit(&ring);
    if (ret < 0) {
        fprintf(stderr, "io_uring_submit: %s\n", strerror(-ret));
        return ret;
    }

    // Wait for queue to finish
    io_uring_cqe *cqe[32768];
    ret = io_uring_wait_cqe_nr(&ring, &cqe[0], num_cqe);
    if (ret < 0) {
        fprintf(stderr, "io_uring_wait_cqe_nr: %s\n", strerror(-ret));
        return ret;
    }
    io_uring_cq_advance(&ring, num_cqe);

    // Destroy queue
    io_uring_queue_exit(&ring);
    return 0;
}

template <typename DataType>
size_t
io_uring_stream_t<DataType>::prepare_slice() {
    Timer::time_point beg_read = Timer::now();
    transfer_slice_len = elem_per_slice;
    size_t elements_read = elem_per_chunk * transferred_chunks;
    if (elements_read + transfer_slice_len > num_offsets * elem_per_chunk) {
        transfer_slice_len = num_offsets * elem_per_chunk - elements_read;
    }
    if ((full_transfer) && (elements_read + transfer_slice_len > filesize)) {
        transfer_slice_len = filesize - elements_read;
    }
    size_t chunks_left = chunks_per_slice;
    if (transferred_chunks + chunks_left > num_offsets) {
        chunks_left = num_offsets - transferred_chunks;
    }

    size_t chunks_per_thread = chunks_left / num_threads;
    if (chunks_per_thread * num_threads < chunks_left)
        chunks_per_thread += 1;

    int active_threads =
        num_threads > (int)chunks_left ? chunks_left : num_threads;
    for (int tid = 0; tid < active_threads; tid++) {
        size_t beg = tid * chunks_per_thread;
        size_t end = (tid + 1) * chunks_per_thread;
        if (end > chunks_left)
            end = chunks_left;
        futures.push_back(std::async(std::launch::async | std::launch::deferred,
                                     &io_uring_stream_t::read_chunks, this, tid,
                                     beg, end));
    }

    Timer::time_point end_read = Timer::now();
    timer[1] +=
        std::chrono::duration_cast<Duration>(end_read - beg_read).count();

    fut = std::async(std::launch::async | std::launch::deferred,
                     &io_uring_stream_t::async_memcpy, this);
    return transfer_slice_len;
}

// Start streaming data from Host to Device
template <typename DataType>
void
io_uring_stream_t<DataType>::start_stream(size_t *offset_ptr, size_t n_offsets,
                                          size_t chunk_size) {
    transferred_chunks = 0;   // Initialize stream
    file_offsets = offset_ptr;
    num_offsets = n_offsets;
    if (full_transfer) {
        elem_per_chunk = elem_per_slice;
        bytes_per_chunk = bytes_per_slice;
        chunks_per_slice = 1;
    } else {
        elem_per_chunk = chunk_size;
        bytes_per_chunk = elem_per_chunk * sizeof(DataType);
        if (elem_per_slice > max_ring_size * num_threads * elem_per_chunk)
            elem_per_slice = max_ring_size * num_threads * elem_per_chunk;
        chunks_per_slice = elem_per_slice / elem_per_chunk;
    }
    // DEBUG_PRINT("Elem per chunk: %zu\n", elem_per_chunk);
    // DEBUG_PRINT("Bytes per chunk: %zu\n", bytes_per_chunk);
    // DEBUG_PRINT("Elem per slice: %zu\n", elem_per_slice);
    // DEBUG_PRINT("Bytes per slice: %zu\n", bytes_per_slice);
    // DEBUG_PRINT("Chunks per slice: %zu\n", chunks_per_slice);
    // DEBUG_PRINT("Num offsets: %zu\n", num_offsets);

    if (get_file_size(file, &filesize) != 0) {
        fprintf(stderr, "Failed to get file size\n");
    }

#ifdef __NVCC__
    gpuErrchk(cudaMemcpy(host_offsets, offset_ptr, n_offsets * sizeof(size_t),
                         cudaMemcpyDeviceToHost));
#else
    host_offsets = offset_ptr;
#endif
    Timer::time_point beg = Timer::now();
    prepare_slice();
    Timer::time_point end = Timer::now();
    timer[0] += std::chrono::duration_cast<Duration>(end - beg).count();
    return;
}

// Get next slice of data on Device buffer
template <typename DataType>
DataType *
io_uring_stream_t<DataType>::next_slice() {
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
    host_buffer = transfer_buffer;
#endif
    active_slice_len = transfer_slice_len;
    size_t nchunks = active_slice_len / elem_per_chunk;
    if (elem_per_chunk * nchunks < active_slice_len)
        nchunks += 1;
    transferred_chunks += nchunks;
    if (transferred_chunks < num_offsets) {
        Timer::time_point beg = Timer::now();
        prepare_slice();
        Timer::time_point end = Timer::now();
        timer[0] += std::chrono::duration_cast<Duration>(end - beg).count();
    }
    return active_buffer;
}

// Reset Host to Device stream
template <typename DataType>
void
io_uring_stream_t<DataType>::end_stream() {
    done = true;
}

template <typename DataType>
std::vector<double>
io_uring_stream_t<DataType>::get_timer() {
    return timer;
}

#endif   // __IO_URING_STREAM_HPP
