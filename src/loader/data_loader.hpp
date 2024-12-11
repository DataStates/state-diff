#ifndef __DATA_LOADER_HPP
#define __DATA_LOADER_HPP

#include "common/debug.hpp"
#include "cuda_timer.hpp"
#include "device_cache.hpp"
#include "host_cache.hpp"
#include "io_reader.hpp"
#include <cassert>
#include <optional>
#include <cmath>

#include <algorithm>

enum TransferType : int {
    FileToHost = 0,
    FileToDevice = 1,
    HostToDevice = 2,
    HostPinned = 3,
};

class data_loader_t {

    using FileReader = base_io_reader_t;

    uint8_t *data_ptr_;
    host_cache_t *host_cache_;
    device_cache_t *device_cache_;
    size_t host_cache_size_;
    size_t device_cache_size_;
    int gpu_id = 0;
    std::atomic<int> instance_count{0};
    std::unordered_map<int, size_t> ready_count;

    size_t max_batch_size(size_t seg_size);
    void merge_create_seg(int id, std::vector<size_t> &offsets, size_t total_segs,
                          size_t batch_size_, size_t seg_size);

  public:
    data_loader_t(size_t host_cache_size, size_t device_cache_size);

    ~data_loader_t();

    int file_load(FileReader &io_reader, size_t start_foffset, size_t seg_size,
                   size_t batch_size, TransferType trans_type,
                   std::optional<std::vector<size_t>> offsets = std::nullopt,
                   bool merge_seg = false);
    void mem_load(int loader_id, std::vector<uint8_t> &data, size_t start_foffset,
                  size_t seg_size, size_t batch_size, TransferType trans_type,
                  std::optional<std::vector<size_t>> offsets = std::nullopt);
    size_t next(int loader_id, void *ptr);
    std::pair<uint8_t *, size_t> next(int loader_id, TransferType trans_type);
    size_t get_chunksize(size_t data_size);
};
#endif   // __DATA_LOADER_HPP