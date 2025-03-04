#ifndef __DATA_LOADER_HPP
#define __DATA_LOADER_HPP

// #include "common/debug.hpp"
#include "cuda_timer.hpp"
#include "debug.hpp"
#include "host_cache.hpp"
#include "io_reader.hpp"
// #include "async_data_reader.hpp"
#include <cassert>
#include <cmath>
#include <optional>
// #include <utility>
// #include <iostream>

#include <algorithm>

#ifdef __NVCC__
#include "device_cache.hpp"
#define EXEC_IF_NVCC(instruction) instruction
#else
#define EXEC_IF_NVCC(instruction)
#endif

enum TransferType : int {
    FileToHost = 0,
    FileToDevice = 1,
    HostToDevice = 2,
    HostPinned = 3,
};

class data_loader_t {

    using FileReader = base_io_reader_t;

    uint8_t *data_ptr_;
    size_t host_cache_size_;
    size_t device_cache_size_;
    host_cache_t *host_cache_;
    EXEC_IF_NVCC(device_cache_t *device_cache_;);
    int gpu_id = 0;
    int last_retrieving_id = 0;
    std::atomic<int> instance_count{0};
    std::unordered_map<int, size_t> ready_count;

    void coalesce(int id, std::vector<size_t> offsets, size_t seg_size,
                  uint32_t gap);
    void enqueue_reads(int id, std::vector<size_t> offsets, size_t seg_size,
                       size_t n_segs_per_read, size_t total_n_segs,
                       size_t total_read_size);

  public:
    data_loader_t(){};
    data_loader_t(size_t host_cache_size, size_t device_cache_size);

    ~data_loader_t();

    int file_load(FileReader &io_reader, size_t start_foffset, size_t seg_size,
                  TransferType trans_type,
                  std::optional<std::vector<size_t>> offsets = std::nullopt,
                  bool merge_seg = false, uint32_t gap = 0);
    int file_load(FileReader &io_reader0, FileReader &io_reader1,
                  size_t start_foffset, size_t seg_size,
                  TransferType trans_type,
                  std::optional<std::vector<size_t>> offsets, uint32_t gap = 0);
    size_t next(int loader_id, void *ptr);
    std::pair<uint8_t *, size_t> next(int loader_id, TransferType trans_type);
    size_t get_chunksize(size_t data_size);
};
#endif   // __DATA_LOADER_HPP
