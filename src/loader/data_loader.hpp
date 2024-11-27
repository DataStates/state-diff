#ifndef __DATA_LOADER_HPP
#define __DATA_LOADER_HPP

#include "cuda_timer.hpp"
#include "device_cache.hpp"
#include "host_cache.hpp"
#include "io_reader.hpp"
#include "common/debug.hpp"
#include <cassert>
#include <optional>

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

    size_t max_batch_size(size_t seg_size);

  public:
    data_loader_t(size_t host_cache_size, size_t device_cache_size);

    ~data_loader_t();

    void file_load(FileReader &io_reader, size_t start_foffset, size_t seg_size,
                   size_t batch_size, TransferType trans_type,
                   std::optional<std::vector<size_t>> offsets = std::nullopt);
    void mem_load(std::vector<uint8_t> &data, size_t start_foffset,
                  size_t seg_size, size_t batch_size, TransferType trans_type,
                  std::optional<std::vector<size_t>> offsets = std::nullopt);
    void next(void* ptr);
    void wait();
};
#endif   // __DATA_LOADER_HPP