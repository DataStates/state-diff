#ifndef __MMAP_STREAM_HPP
#define __MMAP_STREAM_HPP
#include <type_traits>
#include <string>
#include <chrono>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <cerrno>
//#undef NDEBUG
#include <cassert>
#include <sys/stat.h>
#include <Kokkos_Core.hpp>
#include "utils.hpp"
#include "debug.hpp"

#define __DEBUG
#define __ASSERT

template<typename DataType>
class MMapStream {
  struct buff_state_t {
    uint8_t *buff;
    size_t size;
  };

  public:
    buff_state_t file_buffer;
    size_t *host_offsets=NULL, *file_offsets=NULL; // Track where to make slice
    size_t num_offsets, transferred_chunks=0;
    size_t active_slice_len=0, transfer_slice_len=0; // Track the length of the inflight slice
    size_t elem_per_slice=0, bytes_per_slice=0; // Max number of elements or bytes per slice
    size_t elem_per_chunk=0, bytes_per_chunk=0; // Number of elements or bytes per chunk
    size_t chunks_per_slice=0;
    bool async=true, full_transfer=true, done=false;
    std::string filename;
    DataType *mmapped_file=NULL; // Pointer to host data
    DataType *active_buffer=NULL, *transfer_buffer=NULL; // Convenient pointers
    DataType *host_buffer=NULL;
    double timer = 0;
#ifdef __NVCC__
    cudaStream_t transfer_stream; // Stream for data transfers
#endif

    buff_state_t map_file(const std::string &fn) {
      int fd = open(fn.c_str(), O_RDONLY | O_DIRECT);
      if (fd == -1)
        FATAL("cannot open " << fn << ", error = " << std::strerror(errno));
      size_t size = lseek(fd, 0, SEEK_END);
      uint8_t *buff = (uint8_t *)mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);
      close(fd);
      if (buff == MAP_FAILED)
        FATAL("cannot mmap " << fn << ", error = " << std::strerror(errno));
      return buff_state_t{buff, size};
    }

    size_t get_chunk(const buff_state_t &ckpt, size_t offset, DataType** ptr) {
      size_t byte_offset = offset * sizeof(DataType);
      assert(byte_offset < ckpt.size);
      size_t ret = byte_offset + bytes_per_chunk >= ckpt.size ? ckpt.size - byte_offset : bytes_per_chunk;
      assert(ret % sizeof(DataType) == 0);
      *ptr = (DataType *)(ckpt.buff + byte_offset);
      return ret / sizeof(DataType);
    }


  public:
    MMapStream() {}

    // Constructor for Host -> Device stream
    MMapStream(size_t buff_len, std::string& file_name, bool async_memcpy=true, bool transfer_all=true) {
      full_transfer = transfer_all;
      async = async_memcpy;
      elem_per_slice = buff_len;
      bytes_per_slice = elem_per_slice*sizeof(DataType);
      filename = file_name; // Save file name
      active_slice_len = elem_per_slice;
      transfer_slice_len = elem_per_slice;
      file_buffer = map_file(filename);
      if(transfer_all) {
        madvise(file_buffer.buff, file_buffer.size, MADV_SEQUENTIAL);
      } else {
        madvise(file_buffer.buff, file_buffer.size, MADV_RANDOM);
      }
      ASSERT(file_buffer.size % sizeof(DataType) == 0);
      INFO("mapping " << first.size / sizeof(DataType) << " elements");
#ifdef __NVCC__
      active_buffer = device_alloc<DataType>(bytes_per_slice);
      transfer_buffer = device_alloc<DataType>(bytes_per_slice);
      host_buffer = host_alloc<DataType>(bytes_per_slice);
      gpuErrchk( cudaStreamCreate(&transfer_stream) );
#else
      if(!full_transfer) {
        active_buffer = device_alloc<DataType>(bytes_per_slice);
        transfer_buffer = device_alloc<DataType>(bytes_per_slice);
      }
      host_buffer = transfer_buffer;
#endif
      DEBUG_PRINT("Constructor: Filename: %s\n", filename.c_str());
      DEBUG_PRINT("Constructor: File size: %zu\n", file_buffer.size);
      DEBUG_PRINT("Constructor: Full transfer? %d\n", full_transfer);
      DEBUG_PRINT("Constructor: Async transfer? %d\n", async);
      DEBUG_PRINT("Constructor: Elem per slice: %zu\n", elem_per_slice);
      DEBUG_PRINT("Constructor: Bytes per slice: %zu\n", bytes_per_slice);
      DEBUG_PRINT("Constructor: Active slice len: %zu\n", active_slice_len);
      DEBUG_PRINT("Constructor: Transfer slice len: %zu\n", transfer_slice_len);
    }

    ~MMapStream() {
      if(done && (file_buffer.buff != NULL)) {
        munmap(file_buffer.buff, file_buffer.size);
        file_buffer.buff = NULL;
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
//      if(done && file_buffer.buff != NULL) {
//        munmap(file_buffer.buff, file_buffer.size);
//        file_buffer.buff = NULL;
//      }
//      if(!full_transfer) {
//        if(active_buffer != NULL)
//          device_free<DataType>(active_buffer);
//        if(transfer_buffer != NULL)
//          device_free<DataType>(transfer_buffer);
//#ifdef __NVCC__
//        if(host_buffer != NULL)
//          host_free<DataType>(host_buffer);
//#endif
//      }
//#ifdef __NVCC__
//      if(done && transfer_stream != 0) {
//        gpuErrchk( cudaStreamDestroy(transfer_stream) );
//      }
//#endif
    }

    // Get slice length for active buffer
    size_t get_slice_len() const {
      return active_slice_len;
    }
 
    size_t* get_offset_ptr() const {
      return file_offsets;
    }

    size_t get_file_size() const {
      return file_buffer.size;
    }

    size_t prepare_slice() {
      if(full_transfer) {
        // Offset into file
        size_t offset = host_offsets[transferred_chunks]*elem_per_slice;
#ifdef __NVCC__
        // Get pointer to chunk
        DataType* slice;
        transfer_slice_len = get_chunk(file_buffer, offset, &slice);
        // Copy chunk to host buffer
        if(transfer_slice_len > 0)
          memcpy(host_buffer, slice, transfer_slice_len*sizeof(DataType));
        // Transfer buffer to GPU if needed
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
#else
        // Get pointer to chunk
        transfer_slice_len = get_chunk(file_buffer, offset, &transfer_buffer);
#endif
      } else {
        // Calculate number of elements to read
        transfer_slice_len = elem_per_slice;
        size_t elements_read = elem_per_chunk*transferred_chunks;
        if(elements_read+transfer_slice_len > num_offsets*elem_per_chunk) { 
          transfer_slice_len = num_offsets*elem_per_chunk - elements_read;
        }
//        bool* chunk_read = (bool*) malloc(sizeof(bool)*chunks_per_slice);
//        for(size_t i=0; i<chunks_per_slice; i++) {
//          chunk_read[i] = false;
//        }
//        #pragma omp parallel
//        #pragma omp single
//{
//        #pragma omp taskloop
        #pragma omp parallel for 
        for(size_t i=0; i<chunks_per_slice; i++) {
//#pragma omp task depend(out: chunk_read[i])
//{
          if(transferred_chunks+i<num_offsets) {
            DataType* chunk;
            size_t len = get_chunk(file_buffer, host_offsets[transferred_chunks+i]*elem_per_chunk, &chunk);
            if(len != elem_per_chunk)
              transfer_slice_len -= elem_per_chunk-len;
            assert(len <= elem_per_chunk);
            assert(i*elem_per_chunk+len*sizeof(DataType) <= bytes_per_slice);
            assert((size_t)host_buffer+i*elem_per_chunk+len <= (size_t)host_buffer+elem_per_slice);
            if(len > 0)
              memcpy(host_buffer+i*elem_per_chunk, chunk, len*sizeof(DataType));
          }
//          chunk_read[i] = true;
//}
        }
#ifdef __NVCC__
        // Transfer buffer to GPU if needed
//#pragma omp task depend(in: chunk_read[0:chunks_per_slice])
//{
//        #pragma omp taskwait
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
//        free(chunk_read);
//}
#endif
//}
      }
      return transfer_slice_len;
    }

    // Start streaming data from Host to Device
    void start_stream(size_t* offset_ptr, const size_t n_offsets, const size_t chunk_size) {
      transferred_chunks = 0; // Initialize stream
      file_offsets = offset_ptr; // Store pointer to device offsets
      num_offsets = n_offsets; // Store number of offsets
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
      DEBUG_PRINT("Elem per chunk: %zu\n", elem_per_chunk);
      DEBUG_PRINT("Bytes per chunk: %zu\n", bytes_per_chunk);
      DEBUG_PRINT("Elem per slice: %zu\n", elem_per_slice);
      DEBUG_PRINT("Bytes per slice: %zu\n", bytes_per_slice);
      DEBUG_PRINT("Chunks per slice: %zu\n", chunks_per_slice);
      DEBUG_PRINT("Num offsets: %zu\n", num_offsets);

      // Copy offsets to device if necessary
#ifdef __NVCC__
      host_offsets = host_alloc<size_t>(n_offsets*sizeof(size_t));
      gpuErrchk( cudaMemcpy(host_offsets, offset_ptr, n_offsets*sizeof(size_t), cudaMemcpyDeviceToHost) );
#else
      host_offsets = offset_ptr;
#endif
      // Start copying data into buffer
      Timer::time_point beg = Timer::now();
      prepare_slice();
      Timer::time_point end = Timer::now();
      timer += std::chrono::duration_cast<Duration>(end - beg).count();
      return;
    }
    
    // Get next slice of data on Device buffer
    DataType* next_slice() {
//#pragma omp taskwait
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
      if(!full_transfer) {
        host_buffer = transfer_buffer;
      }
#endif
      active_slice_len = transfer_slice_len;
      // Update number of chunks transferred
      size_t nchunks = active_slice_len/elem_per_chunk;
      if(elem_per_chunk*nchunks < active_slice_len)
        nchunks += 1;
      transferred_chunks += nchunks;
      // Start reading next slice if there are any left
      if(transferred_chunks < num_offsets) {
        Timer::time_point beg = Timer::now();
        prepare_slice();
        Timer::time_point end = Timer::now();
        timer += std::chrono::duration_cast<Duration>(end - beg).count();
      }
      return active_buffer;
    }

    // Reset Host to Device stream
    void end_stream() {
      done = true;
      timer = 0.0;
    }

    double get_timer() {
      return timer;
    }
};

#endif // __MMAP_STREAM_HPP

