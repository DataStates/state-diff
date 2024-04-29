#ifndef __MMAP_STREAM_HPP
#define __MMAP_STREAM_HPP
#include <type_traits>
#include <string>
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
    size_t filesize;
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
      ASSERT(byte_offset < ckpt.size);
      size_t ret = byte_offset + bytes_per_chunk >= ckpt.size ? ckpt.size - byte_offset : bytes_per_chunk;
      ASSERT(ret % sizeof(DataType) == 0);
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
//        madvise(file_buffer.buff, file_buffer.size, MADV_SEQUENTIAL);
      }
      ASSERT(file_buffer.size % sizeof(DataType) == 0);
      INFO("mapping " << first.size / sizeof(DataType) << " elements");
#ifdef __NVCC__
      active_buffer = device_alloc<DataType>(bytes_per_slice);
      transfer_buffer = device_alloc<DataType>(bytes_per_slice);
      if(!full_transfer) {
        host_buffer = host_alloc<DataType>(bytes_per_slice);
      } else {
        host_buffer = transfer_buffer;
      }
#else
      if(!full_transfer) {
        active_buffer = device_alloc<DataType>(bytes_per_slice);
        transfer_buffer = device_alloc<DataType>(bytes_per_slice);
      }
      host_buffer = transfer_buffer;
#endif
      
#ifdef __NVCC__
      gpuErrchk( cudaStreamCreate(&transfer_stream) );
#endif
      DEBUG_PRINT("Constructor: Filename: %s\n", filename.c_str());
      DEBUG_PRINT("File size: %zu\n", file_buffer.size);
      DEBUG_PRINT("Constructor: Full transfer? %d\n", full_transfer);
      DEBUG_PRINT("Constructor: Async transfer? %d\n", async);
      DEBUG_PRINT("Constructor: Elem per slice: %zu\n", elem_per_slice);
      DEBUG_PRINT("Constructor: Bytes per slice: %zu\n", bytes_per_slice);
      DEBUG_PRINT("Constructor: Active slice len: %zu\n", active_slice_len);
      DEBUG_PRINT("Constructor: Transfer slice len: %zu\n", transfer_slice_len);
    }

    // Move assignment operator
    MMapStream& operator=(MMapStream&& right) {
      file_buffer.buff = right.file_buffer.buff;
      file_buffer.size = right.file_buffer.size;
      right.file_buffer.buff = NULL;
      right.file_buffer.size = 0;
      host_offsets = right.host_offsets;
      right.host_offsets = NULL;
      file_offsets = right.file_offsets;
      right.file_offsets = NULL;
      num_offsets = right.num_offsets;
      active_slice_len = right.active_slice_len;
      transfer_slice_len = right.transfer_slice_len;
      elem_per_slice = right.elem_per_slice;
      bytes_per_slice = right.bytes_per_slice;
      elem_per_chunk = right.elem_per_chunk;
      bytes_per_chunk = right.bytes_per_chunk;
      chunks_per_slice = right.chunks_per_slice;
      async = right.async;
      full_transfer = right.full_transfer;
      done = right.done;
      filename = right.filename;
      filesize = right.filesize;
      mmapped_file = right.mmapped_file;
      right.mmapped_file = NULL;
      host_buffer = right.host_buffer;
      right.host_buffer = NULL;
      active_buffer = right.active_buffer;
      right.active_buffer = NULL;
      transfer_buffer = right.transfer_buffer;
      right.transfer_buffer = NULL;
#ifdef __NVCC__
      transfer_stream = right.transfer_stream;
      right.transfer_stream = 0;
#endif
      return *this;
    }

    // Move constructor
    MMapStream(MMapStream&& src) {
      *this = src;
    }

    ~MMapStream() {
      if(done && file_buffer.buff != NULL) {
        munmap(file_buffer.buff, file_buffer.size);
        file_buffer.buff = NULL;
      }
      if(!full_transfer) {
        if(active_buffer != NULL)
          device_free<DataType>(active_buffer);
        if(transfer_buffer != NULL)
          device_free<DataType>(transfer_buffer);
#ifdef __NVCC__
        if(host_buffer != NULL)
          host_free<DataType>(host_buffer);
#endif
      }
#ifdef __NVCC__
      if(done && transfer_stream != 0) {
        gpuErrchk( cudaStreamDestroy(transfer_stream) );
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
      return file_buffer.size;
    }

    size_t prepare_slice() {
      if(full_transfer) {
        transfer_slice_len = get_chunk(file_buffer, host_offsets[transferred_chunks]*elem_per_slice, &transfer_buffer);
      } else {
        if(elem_per_chunk*(transferred_chunks+chunks_per_slice) > filesize/sizeof(DataType)) { 
          transfer_slice_len = filesize/sizeof(DataType) - transferred_chunks*elem_per_chunk;
        } else {
          transfer_slice_len = elem_per_chunk*chunks_per_slice;
        }
//#pragma omp task depend(inout: host_buffer[0:transfer_slice_len])
//{
//        #pragma omp parallel for
        #pragma omp taskloop
        for(size_t i=0; i<chunks_per_slice; i++) {
          if(transferred_chunks+i<num_offsets) {
            DataType* chunk;
            size_t len = get_chunk(file_buffer, host_offsets[transferred_chunks+i]*elem_per_chunk, &chunk);
            assert(len <= elem_per_chunk);
            assert(i*elem_per_chunk+len*sizeof(DataType) <= bytes_per_slice);
            assert((size_t)host_buffer+i*elem_per_chunk+len <= (size_t)host_buffer+elem_per_slice);
            if(len > 0)
              memcpy(host_buffer+i*elem_per_chunk, chunk, len*sizeof(DataType));
          }
        }
//}

//        size_t per_threads = chunks_per_slice / (Kokkos::num_threads());
//        if(per_threads*Kokkos::num_threads() < chunks_per_slice)
//          per_threads += 1;
//        for(int thread=0; thread<Kokkos::num_threads(); thread++) {
////#pragma omp task depend(out: host_buffer[thread*per_threads : (thread+1)*per_threads])
//#pragma omp task 
//{
//          for(size_t i=per_threads*thread; i<(thread+1)*per_threads; i++) {
//            if((i<chunks_per_slice) && transferred_chunks+i<num_offsets) {
//              DataType* chunk;
//              size_t len = get_chunk(file_buffer, host_offsets[transferred_chunks+i]*elem_per_chunk, &chunk);
//              assert(len <= elem_per_chunk);
//              assert(i*elem_per_chunk+len*sizeof(DataType) <= bytes_per_slice);
//              assert((size_t)host_buffer+i*elem_per_chunk+len <= (size_t)host_buffer+elem_per_slice);
//              if(len > 0)
//                memcpy(host_buffer+i*elem_per_chunk, chunk, len*sizeof(DataType));
//            }
//          }
//}
//        }
#ifdef __NVCC__
#pragma omp task depend(in: host_buffer[0:transfer_slice_len])
{
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
}
#endif
      }
      return transfer_slice_len;
    }

    // Start streaming data from Host to Device
    void start_stream(size_t* offset_ptr, const size_t n_offsets, const size_t chunk_size) {
      transferred_chunks = 0; // Initialize stream
      file_offsets = offset_ptr;
      filesize = file_buffer.size;
      num_offsets = n_offsets;
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

#ifdef __NVCC__
      host_offsets = host_alloc<size_t>(n_offsets*sizeof(size_t));
      gpuErrchk( cudaMemcpy(host_offsets, offset_ptr, n_offsets*sizeof(size_t), cudaMemcpyDeviceToHost) );
#else
      host_offsets = offset_ptr;
#endif
      Timer::time_point beg = Timer::now();
      prepare_slice();
      Timer::time_point end = Timer::now();
      timer += std::chrono::duration_cast<Duration>(end - beg).count();
      return;
    }
    
    // Get next slice of data on Device buffer
    DataType* next_slice() {
#pragma omp taskwait
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
      size_t nchunks = active_slice_len/elem_per_chunk;
      if(elem_per_chunk*nchunks < active_slice_len)
        nchunks += 1;
      transferred_chunks += nchunks;
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

