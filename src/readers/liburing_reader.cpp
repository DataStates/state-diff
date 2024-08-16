#include "liburing_reader.hpp"

liburing_io_reader_t::liburing_io_reader_t() {
}

liburing_io_reader_t::~liburing_io_reader_t() {
    wait_all();   
    close(fd);
}

liburing_io_reader_t::liburing_io_reader_t(std::string& name) {
    fname = name;
    fd = open(name.c_str(), O_RDONLY | O_DIRECT);
    if (fd == -1) {
        FATAL("cannot open " << fname << ", error = " << std::strerror(errno));
    }
    fsize = lseek(fd, 0, SEEK_END);
    lseek(fd, 0, SEEK_SET);
}

int liburing_io_reader_t::read_data(size_t beg, size_t end) {
    // Create submission queue
    io_uring ring;
    int ret = setup_context(max_ring_size, &ring);
    if (ret < 0)
        FATAL("Failed to setup ring of size max_ring_size");

    // Fill submission queue with reads
    size_t submit_reads = 0;
    size_t num_reads = end-beg;
    size_t idx = beg;
    while(submit_reads < num_reads) {
        // Queue reads
        size_t stop = idx + max_ring_size;
        if(stop > end)
            stop = end;
        for(idx; idx<stop; idx++) {
            segment_t& seg = reads[idx];
            ssize_t len = 0;
            if(seg.size + seg.offset > fsize)
              seg.size = fsize - seg.offset;

            struct io_uring_sqe *sqe;
            sqe = io_uring_get_sqe(&ring);
            if (!sqe) {
                fprintf(stderr, "Could not get sqe\n");
                return -1;
            }
            io_uring_sqe_set_data64(sqe, idx);
            io_uring_prep_read(sqe, fd, seg.buffer, seg.size, seg.offset);
            submit_reads += 1;
        }

        // Submit queue
        ret = io_uring_submit(&ring);
        if(ret < 0) {
            fprintf(stderr, "io_uring_submit: %s\n", strerror(-ret));
        }

        // Wait for some completion events 
        io_uring_cqe *cqe;
        size_t num_cqe = max_ring_size;
        if(num_reads - idx < num_cqe)
            num_cqe = num_reads - idx;
        for(size_t i=0; i<num_cqe; i++) {
            ret = io_uring_wait_cqe(&ring, &cqe);
            if(ret < 0) {
                fprintf(stderr, "io_uring_wait_cqe: %s\n", strerror(-ret));
                return ret;
            }
            segment_status[io_uring_cqe_get_data64(cqe)] = true;
        }
        io_uring_cq_advance(&ring, num_cqe);
    }

    // Destory queue
    io_uring_queue_exit(&ring);

    return 0;
}

int liburing_io_reader_t::enqueue_reads(const std::vector<segment_t>& segments) {
    size_t start = reads.size();
    segment_status.insert(segment_status.end(), segments.size(), false);
    reads.insert(reads.end(), segments.begin(), segments.end());
    size_t per_thread = segments.size() / num_threads;
    if(per_thread*num_threads < segments.size())
        per_thread += 1;
    for(size_t i=0; i<num_threads; i++) {
        size_t beg = i*per_thread;
        size_t end = (i+1)*per_thread;
        if(end > segments.size())
            end = segments.size();
        futures.push_back(std::async(std::launch::async | std::launch::deferred,
                                     &liburing_io_reader_t::read_data, this, beg, end));
    }
    return 0;
}

int liburing_io_reader_t::wait(size_t id) {
    size_t pos = 0;
    for(pos; pos < reads.size(); pos++) {
        if(reads[pos].id == id)
            break;
    }
    while(segment_status[pos] == false) {
      std::this_thread::sleep_for(std::chrono::duration<double, std::milli>(10));
    }
    return 0;
}

int liburing_io_reader_t::wait_all() {
    for (uint32_t i = 0; i < futures.size(); i++) {
        int res = futures[i].get();
        if (res < 0)
            fprintf(stderr, "read_chunks: %s\n", strerror(-res));
    }
    futures.clear();
    reads.clear();
    segment_status.clear();
    return 0;
}

template <typename DataType>
int
liburing_reader_t<DataType>::setup_context(uint32_t entries,
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
liburing_reader_t<DataType>::get_file_size(int fd, size_t *size) {
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
liburing_reader_t<DataType>::queue_read(struct io_uring *ring, off_t size,
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

template <typename DataType>
size_t *
liburing_reader_t<DataType>::get_offset_ptr() const {
    return file_offsets;
}

template <typename DataType>
liburing_reader_t<DataType>::liburing_reader_t(size_t buff_len,
                                               const std::string &file_name,
                                               const size_t chunk_size,
                                               bool async_memcpy,
                                               bool transfer_all, int nthreads)
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

    file = open(filename.c_str(), O_RDONLY);
    if (file < 0) {
        fprintf(stderr, "Failed to open file %s\n", filename.c_str());
    }
    if (get_file_size(file, &filesize) != 0) {
        fprintf(stderr, "Failed to retrieve file size for %s\n",
                filename.c_str());
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

    DEBUG_PRINT("Constructor: Filename: %s\n", filename.c_str());
    DEBUG_PRINT("Constructor: File size: %zu\n", filesize);
    DEBUG_PRINT("Constructor: Full transfer? %d\n", full_transfer);
    DEBUG_PRINT("Constructor: Async transfer? %d\n", async);
    DEBUG_PRINT("Constructor: Elem per slice: %zu\n", elem_per_slice);
    DEBUG_PRINT("Constructor: Bytes per slice: %zu\n", bytes_per_slice);
    DEBUG_PRINT("Constructor: Active slice len: %zu\n", active_slice_len);
    DEBUG_PRINT("Constructor: Transfer slice len: %zu\n", transfer_slice_len);
}

template <typename DataType>
liburing_reader_t<DataType> &
liburing_reader_t<DataType>::operator=(const liburing_reader_t &other) {
    if (this == &other) {
        return *this;
    }
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
    filesize = other.filesize;
    file = other.file;
    active_buffer = other.active_buffer;
    transfer_buffer = other.transfer_buffer;
    host_buffer = other.host_buffer;
    num_threads = other.num_threads;
    max_ring_size = other.max_ring_size;
    num_cqe = other.num_cqe;
    ring = other.ring;
    mmapped_file = other.mmapped_file;
    timer = other.timer;
#ifdef __NVCC__
    transfer_stream = other.transfer_stream;
#endif
    return *this;
}

template <typename DataType> liburing_reader_t<DataType>::~liburing_reader_t() {
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

template <typename DataType>
int
liburing_reader_t<DataType>::async_memcpy() {
    for (int i = 0; i < futures.size(); i++) {
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
liburing_reader_t<DataType>::read_chunks(int tid, size_t beg, size_t end) {
    // Create submission queue
    io_uring ring;
    int ret = setup_context(max_ring_size, &ring);
    if (ret < 0)
        FATAL("Failed to setup ring of size " << max_ring_size);

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
liburing_reader_t<DataType>::prepare_slice() {
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

    int active_threads = num_threads > chunks_left ? chunks_left : num_threads;
    for (int tid = 0; tid < active_threads; tid++) {
        size_t beg = tid * chunks_per_thread;
        size_t end = (tid + 1) * chunks_per_thread;
        if (end > chunks_left)
            end = chunks_left;
        futures.push_back(std::async(std::launch::async | std::launch::deferred,
                                     &liburing_reader_t::read_chunks, this, tid,
                                     beg, end));
    }
    Timer::time_point end_read = Timer::now();
    timer[1] +=
        std::chrono::duration_cast<Duration>(end_read - beg_read).count();

    fut = std::async(std::launch::async | std::launch::deferred,
                     &liburing_reader_t::async_memcpy, this);
    return transfer_slice_len;
}

template <typename DataType>
void
liburing_reader_t<DataType>::start_stream(size_t *offset_ptr,
                                          const size_t n_offsets,
                                          const size_t chunk_size) {
    transferred_chunks = 0;      // Initialize stream
    file_offsets = offset_ptr;   // Store pointer to device offsets
    num_offsets = n_offsets;     // Store number of offsets
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
    DEBUG_PRINT("Elem per chunk: %zu\n", elem_per_chunk);
    DEBUG_PRINT("Bytes per chunk: %zu\n", bytes_per_chunk);
    DEBUG_PRINT("Elem per slice: %zu\n", elem_per_slice);
    DEBUG_PRINT("Bytes per slice: %zu\n", bytes_per_slice);
    DEBUG_PRINT("Chunks per slice: %zu\n", chunks_per_slice);
    DEBUG_PRINT("Num offsets: %zu\n", num_offsets);

    if (get_file_size(file, &filesize) != 0) {
        fprintf(stderr, "Failed to get file size\n");
    }

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
liburing_reader_t<DataType>::get_file_size() const {
    return filesize;
}

template <typename DataType>
DataType *
liburing_reader_t<DataType>::next_slice() {
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
liburing_reader_t<DataType>::get_slice_len() const {
    return active_slice_len;
}

template <typename DataType>
std::vector<double>
liburing_reader_t<DataType>::get_timer() {
    return timer;
}

template <typename DataType>
void
liburing_reader_t<DataType>::end_stream() {
    done = true;
    timer = {0.0};
}
