#ifndef DIRECT_COMPARER_HPP
#define DIRECT_COMPARER_HPP
#include <Kokkos_Core.hpp>
#include <Kokkos_Bitset.hpp>
#include <Kokkos_ScatterView.hpp>
#include <climits>
#include "utils.hpp"
#include "kokkos_vector.hpp"
#include "mmap_stream.hpp"
#ifdef IO_URING_STREAM
#include "io_uring_stream.hpp"
#endif

template<typename DataType, typename ExecutionDevice=Kokkos::DefaultExecutionSpace>
class DirectComparer {
  public:
    using scalar_type = DataType;
    using exec_space = ExecutionDevice;
    double tol;
    size_t d_stream_buf_len = 1024*1024*1024/sizeof(DataType);
    Kokkos::Bitset<Kokkos::DefaultExecutionSpace> changed_entries;
    size_t num_comparisons=0;
    std::string file0, file1;
    size_t block_size = 0;
    std::vector<double> io_timer0, io_timer1;
    double compare_timer=0.0;

  public:
    // Default empty constructor
    DirectComparer();

    DirectComparer(const double err_tol, 
                   const uint32_t chunk_size,
                   const size_t device_buf_len=1024*1024*1024/sizeof(DataType));

    ~DirectComparer() {}

    /**
     * Setup deduplicator based on the provided data. Calculate necessary values and allocates
     * memory for data structures 
     *
     * \param data_device_ptr   Data to be deduplicated
     * \param data_device_len   Length of data in bytes
     */
    void setup(const uint64_t  data_device_len);

    /**
     * Main deduplicate function. Given a Device pointer, create an incremental diff using 
     * the chosen deduplication strategy. Returns a host pointer with the diff.
     *
     * \param data_device_ptr   Data to be deduplicated
     * \param data_device_len   Length of data in bytes
     */
    template< template<typename> typename CompareFunc> 
    uint64_t
    compare(const DataType* data_device_ptr, 
            const uint64_t  data_device_len,
            size_t*   offsets,
            const size_t    noffsets);

    template< template<typename> typename CompareFunc> 
    uint64_t compare(size_t* offsets, const size_t noffsets);

    template< template<typename> typename CompareFunc> 
    uint64_t compare(const DataType* data_a, const DataType* data_b, const size_t data_length);

    /**
     * Serialize the current Merkle tree as well as needed values for checkpoint metadata
     */
    std::vector<uint8_t> serialize(); 

    /**
     * Deserialize the current Merkle tree as well as needed values for checkpoint metadata
     *
     * \param buffer Buffer containing serialized structure
     */
    int deserialize(std::string& filename);
    int deserialize(std::string& filename0, std::string& filename1);

    double get_io_time() const;
    double get_compare_time() const;
    uint64_t get_num_comparisons() const ;
    uint64_t get_num_changed_blocks() const ;
};

template<typename DataType, typename ExecutionDevice>
DirectComparer<DataType,ExecutionDevice>::DirectComparer() {
  tol = 0.0;
  io_timer0 = std::vector<double>(3, 0.0);
  io_timer1 = std::vector<double>(3, 0.0);
}

template<typename DataType, typename ExecutionDevice>
DirectComparer<DataType,ExecutionDevice>::DirectComparer(double tolerance, 
                                                         const uint32_t chunk_size,
                                                         size_t device_buf_len
                                                         ) {
  tol = tolerance;
  d_stream_buf_len = device_buf_len;
  block_size = chunk_size;
  io_timer0 = std::vector<double>(3, 0.0);
  io_timer1 = std::vector<double>(3, 0.0);
}

/**
 * Setup deduplicator based on the provided data. Calculate necessary values and allocates
 * memory for data structures 
 *
 * \param data_device_ptr   Data to be deduplicated
 * \param data_device_len   Length of data in bytes
 */
template<typename DataType, typename ExecutionDevice>
void DirectComparer<DataType,ExecutionDevice>::setup(const uint64_t data_device_len) {
}

/**
 * Direct Comparison Function. Given a data source and length, will compare data
 * using a user defined comparison function. returns number of differing 
 * elements. 
 *
 * \param data_device_ptr   Data to be deduplicated
 * \param data_device_len   Length of data in bytes
 */
template<typename DataType, typename ExecutionDevice>
template<template<typename> typename CompareFunc> 
uint64_t DirectComparer<DataType,ExecutionDevice>::compare(const DataType* data_device_ptr, 
                                                           const uint64_t  data_device_len,
                                                           size_t*   offsets,
                                                           const size_t    noffsets) {
  // Start streaming data
#ifdef IO_URING_STREAM
  IOUringStream<DataType> file_stream0(d_stream_buf_len, file0);
#else
  MMapStream<DataType> file_stream0(d_stream_buf_len, file0); 
#endif
  file_stream0.start_stream(offsets, noffsets, block_size);

  changed_entries = Kokkos::Bitset<Kokkos::DefaultExecutionSpace>(noffsets);
  num_comparisons = 0;
  CompareFunc<DataType> compFunc;
  uint64_t num_diff = 0;
  size_t offset_idx = 0;
  double err_tol = tol;
  while(offset_idx < data_device_len/block_size) {
    uint64_t ndiff = 0;
    DataType* slice = file_stream0.next_slice();
    size_t slice_len = file_stream0.get_slice_len();
    size_t blocksize = block_size;
    size_t nblocks = slice_len/block_size;
    if(block_size*nblocks < slice_len)
      nblocks += 1;
  
    auto mdrange_policy = Kokkos::MDRangePolicy<size_t, Kokkos::Rank<2>>({0,0}, {nblocks, blocksize});
    Kokkos::parallel_reduce("Count differences", mdrange_policy, 
    KOKKOS_LAMBDA(const size_t i, const size_t j, uint64_t& update) {
      size_t data_idx = blocksize*offsets[offset_idx+i] + j;
      if(data_idx < data_device_len) {
        DataType curr = data_device_ptr[data_idx];
        DataType prev = slice[i*blocksize + j];
        if(!compFunc(curr, prev, err_tol)) {
          update += 1;
        }
      }
    }, Kokkos::Sum<uint64_t>(ndiff));
    Kokkos::fence();
  
    offset_idx += slice_len/block_size;
    num_diff += ndiff;
    num_comparisons += slice_len;
  }
  io_timer0 = file_stream0.get_timer();
  file_stream0.end_stream();
  return num_diff;
}

template<typename DataType, typename ExecutionDevice>
template<template<typename> typename CompareFunc> 
uint64_t DirectComparer<DataType,ExecutionDevice>::compare(size_t* offsets, const size_t noffsets) {

  Kokkos::Profiling::pushRegion("Direct: Compare: create streams");
#ifdef IO_URING_STREAM
  IOUringStream<DataType> file_stream0(d_stream_buf_len, file0, true, true);
  IOUringStream<DataType> file_stream1(d_stream_buf_len, file1, true, true);
#else
  MMapStream<DataType> file_stream0(d_stream_buf_len, file0, true, true); 
  MMapStream<DataType> file_stream1(d_stream_buf_len, file1, true, true); 
#endif
  Kokkos::Profiling::popRegion();
  Kokkos::Profiling::pushRegion("Direct: Compare: start streaming");
  file_stream0.start_stream(offsets, noffsets, block_size);
  file_stream1.start_stream(offsets, noffsets, block_size);
  Kokkos::Profiling::popRegion();

  Kokkos::Profiling::pushRegion("Direct: Compare: prep");
  CompareFunc<DataType> compFunc;
  uint64_t num_diff = 0;
  double err_tol = tol;
  size_t data_processed=0;
  // Stats
  num_comparisons = 0;
  changed_entries = Kokkos::Bitset<Kokkos::DefaultExecutionSpace>(noffsets);
  changed_entries.reset();
  auto& changes = changed_entries;
  auto elem_per_block = block_size;
  // Calculate number of iterations
  size_t num_iter = file_stream0.num_offsets/file_stream0.chunks_per_slice;
  if(num_iter * file_stream0.chunks_per_slice < file_stream0.num_offsets)
    num_iter += 1;
  DEBUG_PRINT("Number of iterations: %zu\n", num_iter);
  Kokkos::Profiling::popRegion();

  for(size_t iter=0; iter<num_iter; iter++) {
    Kokkos::Profiling::pushRegion("Direct: Compare: get slices");
    DataType* sliceA = file_stream0.next_slice();
    DataType* sliceB = file_stream1.next_slice();
    size_t slice_len = file_stream0.get_slice_len();
    uint64_t ndiff = 0;
    Kokkos::Profiling::popRegion();

    Kokkos::Profiling::pushRegion("Direct: Compare: compare slices");
    Timer::time_point beg = Timer::now();
    // Parallel comparison
    auto range_policy = Kokkos::RangePolicy<size_t>(0, slice_len);
    Kokkos::parallel_reduce("Count differences", range_policy, 
    KOKKOS_LAMBDA(const size_t i, uint64_t& update) {
      size_t data_idx = data_processed + i;
      if(!compFunc(sliceA[i], sliceB[i], err_tol)) {
        update += 1;
        changes.set(data_idx/elem_per_block);
      }
    }, Kokkos::Sum<uint64_t>(ndiff));
    Kokkos::fence();
    data_processed += slice_len;
    num_diff += ndiff;
    Timer::time_point end = Timer::now();
    compare_timer += std::chrono::duration_cast<Duration>(end - beg).count();
    Kokkos::Profiling::popRegion();
  }
  num_comparisons = data_processed;
  io_timer0 = file_stream0.get_timer();
  io_timer1 = file_stream1.get_timer();
  file_stream0.end_stream();
  file_stream1.end_stream();
  return num_diff;
}

template<typename DataType, typename ExecutionDevice>
template<template<typename> typename CompareFunc> 
uint64_t DirectComparer<DataType,ExecutionDevice>::compare(const DataType* data_a, const DataType* data_b, const size_t data_length) {
  CompareFunc<DataType> compFunc;
  uint64_t num_diff = 0;
  double err_tol = tol;
  uint32_t num_chunks = data_length/block_size;
  if(num_chunks*block_size < data_length)
    num_chunks += 1;
  changed_entries = Kokkos::Bitset<Kokkos::DefaultExecutionSpace>(num_chunks);
  changed_entries.reset();
  auto& changes = changed_entries;
  auto elem_per_block = block_size;

  // Parallel comparison
  auto range_policy = Kokkos::RangePolicy<size_t>(0, data_length);
  Kokkos::parallel_reduce("Count differences", range_policy, 
  KOKKOS_LAMBDA(const size_t i, uint64_t& update) {
    if(!compFunc(data_a[i], data_b[i], err_tol)) {
      update += 1;
      changes.set(i/elem_per_block);
    }
  }, Kokkos::Sum<uint64_t>(num_diff));
  Kokkos::fence();
  num_comparisons = data_length;
  return num_diff;
}

/**
 * Serialize the current Merkle tree as well as needed values for checkpoint metadata
 */
template<typename DataType, typename ExecutionDevice>
std::vector<uint8_t> DirectComparer<DataType,ExecutionDevice>::serialize() {
  std::vector<uint8_t> buffer;
  return buffer;
}

/**
 * Deserialize the current Merkle tree as well as needed values for checkpoint metadata
 *
 * \param buffer Buffer containing serialized structure
 */
template<typename DataType, typename ExecDevice>
int
DirectComparer<DataType,ExecDevice>::deserialize(std::string& filename) {
  file0 = filename;
  return 0;
}

template<typename DataType, typename ExecDevice>
int 
DirectComparer<DataType,ExecDevice>::deserialize(std::string& filename0, std::string& filename1) {
  file0 = filename0;
  file1 = filename1;
  return 0;
}

template<typename DataType, typename ExecDevice>
uint64_t DirectComparer<DataType,ExecDevice>::get_num_comparisons() const {
  return num_comparisons; 
}

template<typename DataType, typename ExecDevice>
uint64_t DirectComparer<DataType,ExecDevice>::get_num_changed_blocks() const {
  return changed_entries.count();
  //uint64_t num_diff = 0;
  //size_t nblocks = changed_entries.size()/block_size;
  //size_t elem_per_block = block_size;
  //auto& changes = changed_entries;
  //Kokkos::parallel_reduce("Count differences", Kokkos::RangePolicy<size_t>(0, nblocks),
  //KOKKOS_LAMBDA(const size_t i, uint64_t& update) {
  //  bool changed = false;
  //  for(size_t j=0; j<elem_per_block; j++) {
  //    size_t idx = i*elem_per_block + j;
  //    if(changes.test(idx))
  //      changed = true;
  //  }
  //  if(changed) {
////printf("Direct: Block %zu changed\n", i);
  //    update += 1;
  //  }
  //}, Kokkos::Sum<uint64_t>(num_diff));
  //Kokkos::fence();
  //return num_diff;
}

template<typename DataType, typename ExecDevice>
double DirectComparer<DataType, ExecDevice>::get_io_time() const {
  return io_timer0[0];
}

template<typename DataType, typename ExecDevice>
double DirectComparer<DataType, ExecDevice>::get_compare_time() const {
  return compare_timer;
}

#endif // DIRECT_COMPARER_HPP

