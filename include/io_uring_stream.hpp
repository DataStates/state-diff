#ifndef __IO_URING_STREAM_HPP
#define __IO_URING_STREAM_HPP
#include <type_traits>
#include <string>
#include <future>
#include <mutex>
#include <thread>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <cerrno>
//#undef NDEBUG
#include <cassert>
#include <sys/stat.h>
#include <sys/ioctl.h>
#include <stdio.h>
#include <stdlib.h>
#include <liburing.h>
#include <Kokkos_Core.hpp>
#include "utils.hpp"
#include "debug.hpp"

#define __DEBUG
#define __ASSERT

template<typename DataType>
class IOUringStream {

  public:
    size_t *host_offsets=NULL, *file_offsets=NULL; // Track where to make slice
    size_t num_offsets, transferred_chunks=0;
    size_t active_slice_len=0, transfer_slice_len=0; // Track the length of the inflight slice
    size_t elem_per_slice=0, bytes_per_slice=0; // Max number of elements or bytes per slice
    size_t elem_per_chunk=0, bytes_per_chunk=0; // Number of elements or bytes per chunk
    size_t chunks_per_slice=0;
    bool async=true, full_transfer=true, done=false;
    std::string filename;
    off_t filesize;
    int file;
    uint32_t ring_size=32768;
    int num_cqe=0;
    struct io_uring ring;
    DataType *mmapped_file=NULL; // Pointer to host data
    DataType *active_buffer=NULL, *transfer_buffer=NULL; // Convenient pointers
    DataType *host_buffer=NULL;
    std::future<int> fut;
    std::vector<double> timer;
#ifdef __NVCC__
    cudaStream_t transfer_stream; // Stream for data transfers
#endif

  private:
    int setup_context(uint32_t entries, struct io_uring* ring) {
      int ret;
      ret = io_uring_queue_init(entries, ring, 0);
      if( ret<0 ) {
        fprintf(stderr, "queue_init: %s\n", strerror(-ret));
        return -1;
      }
      return 0;
    }

    int get_file_size(int fd, off_t *size) {
        struct stat st;
    
        if (fstat(fd, &st) < 0 )
            return -1;
        if(S_ISREG(st.st_mode)) {
            *size = st.st_size;
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

    int queue_read(struct io_uring *ring, off_t size, off_t offset, void* dst_buf) {
        struct io_uring_sqe *sqe;

        sqe = io_uring_get_sqe(ring);
        if (!sqe) {
            return 1;
        }
    
        io_uring_prep_read(sqe, file, dst_buf, size, offset);
        return 0;
    }

  public:
    IOUringStream() {}

    // Constructor for Host -> Device stream
    IOUringStream(size_t buff_len, std::string& file_name, const size_t chunk_size,
                  bool async_memcpy=true, bool transfer_all=true) {
      timer = std::vector<double>(3, 0.0);
      full_transfer = transfer_all;
      async = async_memcpy;
      elem_per_slice = buff_len;
      bytes_per_slice = elem_per_slice*sizeof(DataType);
      filename = file_name; // Save file name
      active_slice_len = elem_per_slice;
      transfer_slice_len = elem_per_slice;

      // Calculate useful values
      if(full_transfer) {
        elem_per_chunk = elem_per_slice;
        bytes_per_chunk = bytes_per_slice;
        chunks_per_slice = 1;
      } else {
        elem_per_chunk = chunk_size;
        bytes_per_chunk = elem_per_chunk*sizeof(DataType);
        chunks_per_slice = elem_per_slice/elem_per_chunk;
      }

      file = open(filename.c_str(), O_RDONLY);
      if( file < 0 ) {
        fprintf(stderr, "Failed to open file %s\n", filename.c_str());
      }
      if( get_file_size(file, &filesize) != 0) {
        fprintf(stderr, "Failed to retrieve file size for %s\n", filename.c_str());
      }
      INFO("mapping " << first.size / sizeof(DataType) << " elements");
#ifdef __NVCC__
      active_buffer = device_alloc<DataType>(bytes_per_slice);
      transfer_buffer = device_alloc<DataType>(bytes_per_slice);
      host_buffer = host_alloc<DataType>(bytes_per_slice);
      size_t max_offsets = filesize / bytes_per_chunk;
      if(bytes_per_chunk*max_offsets < filesize)
        max_offsets += 1;
      host_offsets = host_alloc<size_t>(sizeof(size_t)*max_offsets);
      gpuErrchk( cudaStreamCreate(&transfer_stream) );
#else
      if(!full_transfer) {
        active_buffer = device_alloc<DataType>(bytes_per_slice);
        transfer_buffer = device_alloc<DataType>(bytes_per_slice);
      }
      host_buffer = transfer_buffer;
#endif
      
      DEBUG_PRINT("Constructor: Filename: %s\n", filename.c_str());
      DEBUG_PRINT("File size: %zu\n", filesize);
      DEBUG_PRINT("Constructor: Full transfer? %d\n", full_transfer);
      DEBUG_PRINT("Constructor: Async transfer? %d\n", async);
      DEBUG_PRINT("Constructor: Elem per slice: %zu\n", elem_per_slice);
      DEBUG_PRINT("Constructor: Bytes per slice: %zu\n", bytes_per_slice);
      DEBUG_PRINT("Constructor: Active slice len: %zu\n", active_slice_len);
      DEBUG_PRINT("Constructor: Transfer slice len: %zu\n", transfer_slice_len);
    }

    ~IOUringStream() {
      if(done && (file != -1)) {
        close(file);
        file = -1;
      }
#ifdef __NVCC__
      if(done && (active_buffer != NULL)) {
        device_free<DataType>(active_buffer);
        active_buffer = NULL;
      }
      if(done && (transfer_buffer != NULL)) {
        device_free<DataType>(transfer_buffer);
        transfer_buffer = NULL;
      }
      if(done && (host_buffer != NULL)) {
        host_free<DataType>(host_buffer);
        host_buffer = NULL;
      }
      if(done && (transfer_stream != 0)) {
        gpuErrchk( cudaStreamDestroy(transfer_stream) );
        transfer_stream = 0;
      }
      if(done && (host_offsets != NULL)) {
        host_free<size_t>(host_offsets);
        host_offsets = NULL;
      }
#else
      if(!full_transfer) {
        if(active_buffer != NULL) {
          device_free<DataType>(active_buffer);
          active_buffer = NULL;
        }
        if(transfer_buffer != NULL) {
          device_free<DataType>(transfer_buffer);
          transfer_buffer = NULL;
        }
      }
#endif
    }

    // Get slice length for active buffer
    size_t get_slice_len() const {
      return active_slice_len;
    }
 
    size_t* get_offset_ptr() const {
      return file_offsets;
    }

    size_t get_file_size() const {
      return filesize;
    }

    int async_memcpy() {
      struct io_uring_cqe *cqe;
      int ret = io_uring_wait_cqe_nr(&ring, &cqe, num_cqe);
      if(ret < 0) {
        fprintf(stderr, "io_uring_wait_cqe_nr: %s\n", strerror(-ret));
        return ret;
      }
      io_uring_cq_advance(&ring, num_cqe);
      
      Timer::time_point beg_copy = Timer::now();
#ifdef __NVCC__
      if(async) {
        gpuErrchk( cudaMemcpyAsync(transfer_buffer, 
                                   host_buffer, 
                                   transfer_slice_len*sizeof(DataType), 
                                   cudaMemcpyHostToDevice, 
                                   transfer_stream) );
      } else {
        gpuErrchk( cudaMemcpy(transfer_buffer, 
                              host_buffer, 
                              transfer_slice_len*sizeof(DataType), 
                              cudaMemcpyHostToDevice) );
      }
#endif
      Timer::time_point end_copy = Timer::now();
      timer[2] += std::chrono::duration_cast<Duration>(end_copy - beg_copy).count();
      return 0;
    }

    size_t prepare_slice() {
      Timer::time_point beg_read = Timer::now();
      int ret;
      uint32_t reads=0;
      transfer_slice_len = elem_per_slice;
      size_t elements_read = elem_per_chunk*transferred_chunks;
      if(elements_read+transfer_slice_len > num_offsets*elem_per_chunk) { 
        transfer_slice_len = num_offsets*elem_per_chunk - elements_read;
      }
      size_t chunks_left = chunks_per_slice;
      if(transferred_chunks+chunks_left>num_offsets) {
        chunks_left = num_offsets - transferred_chunks;
      }

      num_cqe = 0;
      for(size_t i=0; i<chunks_left; i++) {
        if(reads >= ring_size)
          break;
        size_t fileoffset = host_offsets[transferred_chunks+i]*bytes_per_chunk;
        size_t len = bytes_per_chunk;
        if(fileoffset+len > filesize)
          len = filesize - fileoffset;
        if(queue_read(&ring, len, fileoffset, host_buffer+i*elem_per_chunk))
          break;
        reads++;
        num_cqe++;
      }
      ret = io_uring_submit(&ring);
      if(ret < 0) {
        fprintf(stderr, "io_uring_submit: %s\n", strerror(-ret));
      }
      Timer::time_point end_read = Timer::now();
      timer[1] += std::chrono::duration_cast<Duration>(end_read - beg_read).count();
      
      fut = std::async(std::launch::async, &IOUringStream::async_memcpy, this);
      return transfer_slice_len;
    }

    // Start streaming data from Host to Device
    void start_stream(size_t* offset_ptr, size_t n_offsets, size_t chunk_size) {
      transferred_chunks = 0; // Initialize stream
      file_offsets = offset_ptr;
      num_offsets = n_offsets;
      if(full_transfer) {
        elem_per_chunk = elem_per_slice;
        bytes_per_chunk = bytes_per_slice;
        chunks_per_slice = 1;
      } else {
        elem_per_chunk = chunk_size;
        bytes_per_chunk = elem_per_chunk*sizeof(DataType);
        if(elem_per_slice > ring_size*elem_per_chunk)
          elem_per_slice = ring_size*elem_per_chunk;
        chunks_per_slice = elem_per_slice/elem_per_chunk;
      }
      DEBUG_PRINT("Elem per chunk: %zu\n", elem_per_chunk);
      DEBUG_PRINT("Bytes per chunk: %zu\n", bytes_per_chunk);
      DEBUG_PRINT("Elem per slice: %zu\n", elem_per_slice);
      DEBUG_PRINT("Bytes per slice: %zu\n", bytes_per_slice);
      DEBUG_PRINT("Chunks per slice: %zu\n", chunks_per_slice);
      DEBUG_PRINT("Num offsets: %zu\n", num_offsets);

      if (setup_context(ring_size, &ring)) {
        fprintf(stderr, "Failed to setup ring of size %u\n", ring_size);
      }
      if (get_file_size(file, &filesize) != 0) {
        fprintf(stderr, "Failed to get file size\n");
      }

#ifdef __NVCC__
      gpuErrchk( cudaMemcpy(host_offsets, offset_ptr, n_offsets*sizeof(size_t), cudaMemcpyDeviceToHost) );
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
    DataType* next_slice() {
      int ret = fut.get();
      if(ret < 0)
        fprintf(stderr, "async_memcpy: %s\n", strerror(-ret));
#ifdef __NVCC__
      if(async) {
        gpuErrchk( cudaStreamSynchronize(transfer_stream) ); // Wait for slice to finish async copy
      }
#endif
      // Swap device buffers
      DataType* temp = active_buffer;
      active_buffer = transfer_buffer;
      transfer_buffer = temp;
#ifndef __NVCC__
      host_buffer = transfer_buffer;
#endif
      active_slice_len = transfer_slice_len;
      size_t nchunks = active_slice_len/elem_per_chunk;
      if(elem_per_chunk*nchunks < active_slice_len)
        nchunks += 1;
      transferred_chunks += nchunks;
      if(transferred_chunks < num_offsets) {
        Timer::time_point beg = Timer::now();
        prepare_slice();
        Timer::time_point end = Timer::now();
        timer[0] += std::chrono::duration_cast<Duration>(end - beg).count();
      }
      return active_buffer;
    }

    // Reset Host to Device stream
    void end_stream() {
      done = true;
    }

    std::vector<double> get_timer() {
      return timer;
    }
};

#endif // __IO_URING_STREAM_HPP


