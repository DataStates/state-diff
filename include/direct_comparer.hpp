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
    uint32_t current_id;
    double tol;
    size_t d_stream_buf_len = 1024*1024*1024/sizeof(DataType);
#ifdef IO_URING_STREAM
    IOUringStream<scalar_type> file_stream0, file_stream1;
#else
    MMapStream<scalar_type> file_stream0, file_stream1;
#endif
    Kokkos::Bitset<Kokkos::DefaultExecutionSpace> changed_entries;
    Kokkos::View<uint64_t[1]> num_comparisons;
    bool file_stream=false;
    size_t block_size = 0;
    uint32_t num_threads=1;

  public:
    // Default empty constructor
    DirectComparer();

    DirectComparer(double err_tol, size_t device_buf_len=1024*1024*1024/sizeof(DataType), uint32_t nthreads=1);

    ~DirectComparer() {}

    /**
     * Setup deduplicator based on the provided data. Calculate necessary values and allocates
     * memory for data structures 
     *
     * \param data_device_ptr   Data to be deduplicated
     * \param data_device_len   Length of data in bytes
     * \param make_baseline     Flag determining whether to make a baseline checkpoint
     */
    void setup(const uint64_t  data_device_len,
               const bool      make_baseline);

    /**
     * Main deduplicate function. Given a Device pointer, create an incremental diff using 
     * the chosen deduplication strategy. Returns a host pointer with the diff.
     *
     * \param data_device_ptr   Data to be deduplicated
     * \param data_device_len   Length of data in bytes
     * \param make_baseline     Flag determining whether to make a baseline checkpoint
     */
    template< template<typename> typename CompareFunc> 
    uint64_t
    compare(const DataType* data_device_ptr, 
            const uint64_t  data_device_len,
            const bool      make_baseline);

    template< template<typename> typename CompareFunc> 
    uint64_t compare();

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
    uint64_t deserialize(size_t* offsets, size_t noffsets, size_t blocksize, std::string& filename);
    uint64_t deserialize(size_t* offsets, size_t noffsets, size_t blocksize, std::string& file0, std::string& file1);

    uint64_t get_num_comparisons() const ;
    uint64_t get_num_changed_blocks() const ;
};

template<typename DataType, typename ExecutionDevice>
DirectComparer<DataType,ExecutionDevice>::DirectComparer() {
  current_id = 0;
  tol = 0.0;
  num_comparisons = Kokkos::View<uint64_t[1]>("Num comparisons");
}

template<typename DataType, typename ExecutionDevice>
DirectComparer<DataType,ExecutionDevice>::DirectComparer(double tolerance, 
                                                         size_t device_buf_len,
                                                         uint32_t nthreads) {
  current_id = 0;
  tol = tolerance;
  d_stream_buf_len = device_buf_len;
  num_threads= nthreads;
  num_comparisons = Kokkos::View<uint64_t[1]>("Num comparisons");
}

/**
 * Setup deduplicator based on the provided data. Calculate necessary values and allocates
 * memory for data structures 
 *
 * \param data_device_ptr   Data to be deduplicated
 * \param data_device_len   Length of data in bytes
 * \param make_baseline     Flag determining whether to make a baseline checkpoint
 */
template<typename DataType, typename ExecutionDevice>
void DirectComparer<DataType,ExecutionDevice>::setup(const uint64_t  data_device_len,
                                                     const bool      make_baseline) {
}

/**
 * Direct Comparison Function. Given a data source and length, will compare data
 * using a user defined comparison function. returns number of differing 
 * elements. If make_baseline is set, will return data_device_len.
 *
 * \param data_device_ptr   Data to be deduplicated
 * \param data_device_len   Length of data in bytes
 * \param make_baseline     Flag determining whether to make a baseline checkpoint
 */
template<typename DataType, typename ExecutionDevice>
template<template<typename> typename CompareFunc> 
uint64_t DirectComparer<DataType,ExecutionDevice>::compare(const DataType* data_device_ptr, 
                                                           const uint64_t  data_device_len,
                                                           const bool      make_baseline) {
  Kokkos::deep_copy(num_comparisons, 0);
  if(!make_baseline) {
    CompareFunc<DataType> compFunc;
    uint64_t num_diff = 0;
    size_t offset_idx = 0;
    double err_tol = tol;
    while(offset_idx < data_device_len/block_size) {
      uint64_t ndiff = 0;
      DataType* slice = NULL;
      size_t slice_len = 0;
      size_t* offsets = NULL;
      slice = file_stream0.next_slice();
      slice_len = file_stream0.get_slice_len();
      offsets = file_stream0.get_offset_ptr();
      size_t blocksize = block_size;
      size_t nblocks = slice_len/block_size;
      if(block_size*nblocks < slice_len)
        nblocks += 1;
  
      auto& num_comp = num_comparisons;
      auto mdrange_policy = Kokkos::MDRangePolicy<size_t, Kokkos::Rank<2>>({0,0}, {nblocks, blocksize});
      Kokkos::parallel_reduce("Count differences", mdrange_policy, 
      [slice, data_device_ptr, data_device_len, offsets, offset_idx, blocksize, compFunc, err_tol, num_comp] 
      KOKKOS_FUNCTION (const size_t i, const size_t j, uint64_t& update) {
        size_t data_idx = blocksize*offsets[offset_idx+i] + j;
        if(data_idx < data_device_len) {
          DataType curr = data_device_ptr[data_idx];
          DataType prev = slice[i*blocksize + j];
          if(!compFunc(curr, prev, err_tol)) {
            update += 1;
          }
          Kokkos::atomic_add(&(num_comp(0)), 1);
        }
      }, Kokkos::Sum<uint64_t>(ndiff));
      Kokkos::fence();
  
      offset_idx += slice_len/block_size;
      num_diff += ndiff;
    }
    file_stream0.end_stream();
    file_stream1.end_stream();
    return num_diff;
  } else {
    return data_device_len;
  }
}

template<typename DataType, typename ExecutionDevice>
template<template<typename> typename CompareFunc> 
uint64_t DirectComparer<DataType,ExecutionDevice>::compare() {
  Kokkos::Profiling::pushRegion("Direct: Compare: prep");
  CompareFunc<DataType> compFunc;
  uint64_t num_diff = 0;
  double err_tol = tol;
  DataType *sliceA=NULL, *sliceB=NULL;
  size_t slice_len=0, data_processed=0;
  Kokkos::deep_copy(num_comparisons, 0);
  changed_entries.reset();
  auto& changes = changed_entries;
  size_t num_iter = file_stream0.num_offsets/file_stream0.chunks_per_slice;
  if(num_iter * file_stream0.chunks_per_slice < file_stream0.num_offsets)
    num_iter += 1;
  Kokkos::Experimental::ScatterView<uint64_t[1]> num_comp(num_comparisons);
  Kokkos::Profiling::popRegion();
  for(size_t iter=0; iter<num_iter; iter++) {
    Kokkos::Profiling::pushRegion("Direct: Compare: get slices");
    sliceA = file_stream0.next_slice();
    sliceB = file_stream1.next_slice();
    slice_len = file_stream0.get_slice_len();
    uint64_t ndiff = 0;
    Kokkos::Profiling::popRegion();

    Kokkos::Profiling::pushRegion("Direct: Compare: compare slices");
    // Parallel comparison
    auto range_policy = Kokkos::RangePolicy<size_t>(0, slice_len);
    Kokkos::parallel_reduce("Count differences", range_policy, 
    KOKKOS_LAMBDA(const size_t i, uint64_t& update) {
      size_t data_idx = data_processed + i;
      if(!compFunc(sliceA[i], sliceB[i], err_tol)) {
        update += 1;
        changes.set(data_idx);
      }
      auto ncomp_access = num_comp.access();
      ncomp_access(0) += 1;
    }, Kokkos::Sum<uint64_t>(ndiff));
    Kokkos::fence();
    data_processed += slice_len;
    num_diff += ndiff;
    Kokkos::Profiling::popRegion();
  }
  Kokkos::Profiling::pushRegion("Direct: Compare: contribute and finalize");
  Kokkos::Experimental::contribute(num_comparisons, num_comp);
  file_stream0.end_stream();
  file_stream1.end_stream();
  Kokkos::Profiling::popRegion();
  return num_diff;
}

template<typename DataType, typename ExecutionDevice>
template<template<typename> typename CompareFunc> 
uint64_t DirectComparer<DataType,ExecutionDevice>::compare(const DataType* data_a, const DataType* data_b, const size_t data_length) {
  CompareFunc<DataType> compFunc;
  uint64_t num_diff = 0;
  double err_tol = tol;
  auto& changes = changed_entries;

  // Parallel comparison
  auto range_policy = Kokkos::RangePolicy<size_t>(0, data_length);
  Kokkos::parallel_reduce("Count differences", range_policy, 
  KOKKOS_LAMBDA(const size_t i, uint64_t& update) {
    if(!compFunc(data_a[i], data_b[i], err_tol)) {
      update += 1;
      changes.set(i);
    }
  }, Kokkos::Sum<uint64_t>(num_diff));
  Kokkos::fence();
  Kokkos::deep_copy(num_comparisons, data_length);
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
uint64_t 
DirectComparer<DataType,ExecDevice>::deserialize(size_t* offsets, size_t noffsets, 
                                                 size_t blocksize, std::string& filename) {
  size_t host_len = blocksize*noffsets;
  block_size = blocksize;
  changed_entries = Kokkos::Bitset<Kokkos::DefaultExecutionSpace>(blocksize*noffsets);
#ifdef IO_URING_STREAM
  file_stream0 = IOUringStream<DataType>(d_stream_buf_len, filename);
#else
  file_stream0 = MMapStream<DataType>(d_stream_buf_len, filename); 
#endif
  file_stream0.start_stream(offsets, noffsets, blocksize);
  file_stream=true;
  return noffsets*blocksize;
}

template<typename DataType, typename ExecDevice>
uint64_t 
DirectComparer<DataType,ExecDevice>::deserialize(size_t* offsets, size_t noffsets, size_t blocksize, 
                                                 std::string& file0, std::string& file1) {
  block_size = blocksize;
  changed_entries = Kokkos::Bitset<Kokkos::DefaultExecutionSpace>(blocksize*noffsets);
  file_stream=true;

#ifdef IO_URING_STREAM
  file_stream0 = IOUringStream<DataType>(d_stream_buf_len, file0, true, false);
  file_stream1 = IOUringStream<DataType>(d_stream_buf_len, file1, true, false);
#else
  file_stream0 = MMapStream<DataType>(d_stream_buf_len, file0, true, true); 
  file_stream1 = MMapStream<DataType>(d_stream_buf_len, file1, true, true); 
#endif
  file_stream0.start_stream(offsets, noffsets, blocksize);
  file_stream1.start_stream(offsets, noffsets, blocksize);
  return noffsets*blocksize;
}

template<typename DataType, typename ExecDevice>
uint64_t DirectComparer<DataType,ExecDevice>::get_num_comparisons() const {
  auto num_comparisons_h = Kokkos::create_mirror_view(num_comparisons);
  Kokkos::deep_copy(num_comparisons_h, num_comparisons);
  return num_comparisons_h(0); 
}

template<typename DataType, typename ExecDevice>
uint64_t DirectComparer<DataType,ExecDevice>::get_num_changed_blocks() const {
  uint64_t num_diff = 0;
  size_t nblocks = changed_entries.size()/block_size;
  size_t elem_per_block = block_size;
  auto& changes = changed_entries;
  Kokkos::parallel_reduce("Count differences", Kokkos::RangePolicy<size_t>(0, nblocks),
  KOKKOS_LAMBDA(const size_t i, uint64_t& update) {
    bool changed = false;
    for(size_t j=0; j<elem_per_block; j++) {
      size_t idx = i*elem_per_block + j;
      if(changes.test(idx))
        changed = true;
    }
    if(changed) {
//printf("Direct: Block %zu changed\n", i);
      update += 1;
    }
  }, Kokkos::Sum<uint64_t>(num_diff));
  Kokkos::fence();
//printf("Direct: num diff %lu\n", num_diff);
  return num_diff;
}

#endif // DIRECT_COMPARER_HPP

