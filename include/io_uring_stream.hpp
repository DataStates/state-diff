#ifndef __IO_URING_STREAM_HPP
#define __IO_URING_STREAM_HPP
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
#include <liburing.h>
#include <Kokkos_Core.hpp>
#include "utils.hpp"
#include "debug.hpp"

#define __DEBUG
#define __ASSERT

template<typename DataType>
class IOUringStream {
  struct buff_state_t {
    uint8_t *buff;
    size_t size;
  };

  public:
    size_t *host_offsets=NULL, *file_offsets=NULL; // Track where to make slice
    size_t num_offsets, transferred_chunks=0;
    size_t active_slice_len=0, transfer_slice_len=0; // Track the length of the inflight slice
    size_t elem_per_slice=0, bytes_per_slice=0; // Max number of elements or bytes per slice
    size_t elem_per_chunk=0, bytes_per_chunk=0; // Number of elements or bytes per chunk
    size_t chunks_per_slice=0;
    bool async=true, full_transfer=true, done=false;
    std::string filename;
    size_t filesize;
    int file;
    uint32_t ring_size=32768;
    struct io_uring ring;
    DataType *mmapped_file=NULL; // Pointer to host data
    DataType *active_buffer=NULL, *transfer_buffer=NULL; // Convenient pointers
    DataType *host_buffer=NULL;
#ifdef __NVCC__
    cudaStream_t transfer_stream; // Stream for data transfers
#endif

    struct io_data {
        int read;
        off_t first_offset, offset;
        size_t first_len;
        struct iovec iov;
    };

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

    void queue_prepped(struct io_uring *ring, struct io_data *data) {
        struct io_uring_sqe *sqe;
    
        sqe = io_uring_get_sqe(ring);
        assert(sqe);
    
        if (data->read)
            io_uring_prep_readv(sqe, infd, &data->iov, 1, data->offset);
        else
            io_uring_prep_writev(sqe, outfd, &data->iov, 1, data->offset);
    
        io_uring_sqe_set_data(sqe, data);
    }


    int queue_read(struct io_uring *ring, off_t size, off_t offset) {
        struct io_uring_sqe *sqe;
        struct io_data *data;
    
        data = malloc(size + sizeof(*data));
        if (!data)
            return 1;
    
        sqe = io_uring_get_sqe(ring);
        if (!sqe) {
            free(data);
            return 1;
        }
    
        data->read = 1;
        data->offset = data->first_offset = offset;
    
        data->iov.iov_base = data + 1;
        data->iov.iov_len = size;
        data->first_len = size;
    
        io_uring_prep_readv(sqe, infd, &data->iov, 1, offset);
        io_uring_sqe_set_data(sqe, data);
        return 0;
    }

//    buff_state_t map_file(const std::string &fn) {
//      int fd = open(fn.c_str(), O_RDONLY | O_DIRECT);
//      if (fd == -1)
//        FATAL("cannot open " << fn << ", error = " << std::strerror(errno));
//      size_t size = lseek(fd, 0, SEEK_END);
//      uint8_t *buff = (uint8_t *)mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);
//      close(fd);
//      if (buff == MAP_FAILED)
//        FATAL("cannot mmap " << fn << ", error = " << std::strerror(errno));
//      return buff_state_t{buff, size};
//    }

//    size_t get_chunk(const buff_state_t &ckpt, size_t offset, DataType** ptr) {
//      size_t byte_offset = offset * sizeof(DataType);
//      ASSERT(byte_offset < ckpt.size);
//      size_t ret = byte_offset + bytes_per_chunk >= ckpt.size ? ckpt.size - byte_offset : bytes_per_chunk;
//      ASSERT(ret % sizeof(DataType) == 0);
//      *ptr = (DataType *)(ckpt.buff + byte_offset);
//      return ret / sizeof(DataType);
//    }


  public:
    IOUringStream() {}

    // Constructor for Host -> Device stream
    IOUringStream(size_t buff_len, std::string& file_name, bool async_memcpy=true, bool transfer_all=true) {
      full_transfer = transfer_all;
      async = async_memcpy;
      elem_per_slice = buff_len;
      bytes_per_slice = elem_per_slice*sizeof(DataType);
      filename = file_name; // Save file name
      active_slice_len = elem_per_slice;
      transfer_slice_len = elem_per_slice;
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
//      if(!full_transfer) {
//        active_buffer = device_alloc<DataType>(bytes_per_slice);
//        transfer_buffer = device_alloc<DataType>(bytes_per_slice);
//#ifdef __NVCC__
//        host_buffer = host_alloc<DataType>(bytes_per_slice);
//#else
//        host_buffer = transfer_buffer;
//#endif
//      } else {
//        host_buffer = transfer_buffer;
//      }
      
#ifdef __NVCC__
      gpuErrchk( cudaStreamCreate(&transfer_stream) );
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

    // Move assignment operator
    IOUringStream& operator=(IOUringStream&& right) {
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
      file = right.file;
      right.file = 0;
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
    IOUringStream(IOUringStream&& src) {
      *this = src;
    }

    ~IOUringStream() {
      if(done && file != 0 {
        close(file);
        file=0;
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
      return filsize;
    }

    size_t prepare_slice() {
      uint64_t reads, writes;
DataType* test_buff = host_buffer;
      if(full_transfer) {
        transfer_slice_len = get_chunk(file_buffer, host_offsets[transferred_chunks]*elem_per_slice, &host_buffer);
      } else {
        if(elem_per_chunk*(transferred_chunks+chunks_per_slice) > filesize/sizeof(DataType)) { 
          transfer_slice_len = filesize/sizeof(DataType) - transferred_chunks*elem_per_chunk;
        } else {
          transfer_slice_len = elem_per_chunk*chunks_per_slice;
        }
        size_t chunks_left = chunks_per_slice;
        size_t chunks_to_read = chunks_per_slice;
        while (chunks_to_read || chunks_left) {
          int had_read, got_comp;
          /* Queue up as many reads as we can */
          while (chunks_to_read) {
            off_t this_size = chunks_to_read;
            if (reads+write >= ring_size)
              break;
            if (this_size > buffer
          }
        }
//#pragma omp task depend(inout: test_buff[0:transfer_slice_len])
//{
        #pragma omp parallel for
        for(size_t i=0; i<chunks_per_slice; i++) {
          if(transferred_chunks+i<num_offsets) {
            DataType* chunk;
            size_t len = get_chunk(file_buffer, host_offsets[transferred_chunks+i]*elem_per_chunk, &chunk);
            assert(len <= elem_per_chunk);
            assert(i*elem_per_chunk+len*sizeof(DataType) <= bytes_per_slice);
            assert((size_t)test_buff+i*elem_per_chunk+len <= (size_t)test_buff+elem_per_slice);
            if(len > 0)
              memcpy(test_buff+i*elem_per_chunk, chunk, len*sizeof(DataType));
          }
        }
//}

//        size_t per_threads = chunks_per_slice / (Kokkos::num_threads()/2);
//        if(per_threads*Kokkos::num_threads() < chunks_per_slice)
//          per_threads += 1;
//        for(size_t thread=0; thread<Kokkos::num_threads()/2; thread++) {
//#pragma omp task depend(out: test_buff[thread*per_threads : (thread+1)*per_threads])
//{
//          for(size_t i=per_threads*thread; i<(thread+1)*per_threads; i++) {
//            if((i<chunks_per_slice) && transferred_chunks+i<num_offsets) {
//              DataType* chunk;
//              size_t len = get_chunk(file_buffer, host_offsets[transferred_chunks+i]*elem_per_chunk, &chunk);
//              assert(len <= elem_per_chunk);
//              assert(i*elem_per_chunk+len*sizeof(DataType) <= bytes_per_slice);
//              assert((size_t)test_buff+i*elem_per_chunk+len <= (size_t)test_buff+elem_per_slice);
//              if(len > 0)
//                memcpy(test_buff+i*elem_per_chunk, chunk, len*sizeof(DataType));
//            }
//          }
//}
//        }
#ifdef __NVCC__
//#pragma omp task depend(in: test_buff[0:transfer_slice_len])
//{
      if(async) {
        gpuErrchk( cudaMemcpyAsync(transfer_buffer, 
                                   test_buff, 
                                   transfer_slice_len*sizeof(DataType), 
                                   cudaMemcpyHostToDevice, 
                                   transfer_stream) );
      } else {
        gpuErrchk( cudaMemcpy(transfer_buffer, 
                              test_buff, 
                              transfer_slice_len*sizeof(DataType), 
                              cudaMemcpyHostToDevice) );
      }
//}
#endif
      }
      return transfer_slice_len;
    }

    // Start streaming data from Host to Device
    void start_stream(size_t* offset_ptr, size_t n_offsets, size_t chunk_size) {
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

      if (setup_context(ring_size, &ring) {
        fprintf(stderr, "Failed to setup ring of size %u\n", ring_size);
      }

#ifdef __NVCC__
      host_offsets = host_alloc<size_t>(n_offsets*sizeof(size_t));
      gpuErrchk( cudaMemcpy(host_offsets, offset_ptr, n_offsets*sizeof(size_t), cudaMemcpyDeviceToHost) );
#else
      host_offsets = offset_ptr;
#endif
      prepare_slice();
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
      size_t nchunks = active_slice_len/elem_per_chunk;
      if(elem_per_chunk*nchunks < active_slice_len)
        nchunks += 1;
      transferred_chunks += nchunks;
      if(transferred_chunks < num_offsets)
        prepare_slice();
      return active_buffer;
    }

    // Reset Host to Device stream
    void end_stream() {
      done = true;
    }
};

#endif // __IO_URING_STREAM_HPP


