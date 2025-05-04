#ifndef __DATA_LOADER_HPP
#define __DATA_LOADER_HPP

#include "debug.hpp"
#include "host_cache.hpp"
#include "io_reader.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <optional>

enum TransferType : int {
    FileToHost = 0,
    FileToDevice = 1,
    HostToDevice = 2,
    HostPinned = 3,
};

struct next_batch_t {
    uint8_t *ptr;        // Pointer to start offset
    size_t size;         // Size of the batch in bytes
    size_t offt_count;   // Number of offsets to process
};

struct loader_info_t {
    int ld_id;
    std::vector<size_t> read_offsets;
    int best_gap;
    size_t best_block_size;
};

class data_loader_t {

    using FileReader = base_io_reader_t;

    host_cache_t *host_cache_;
    int gpu_id = 0;
    int last_retrieving_id = 0;
    std::atomic<int> instance_count{0};
    std::unordered_map<int, size_t> ready_count;
    std::unordered_map<int, size_t> wasted_bytes;
    std::unordered_map<int, size_t> IOP_count;
    std::unordered_map<int, size_t> IO_time;

    std::vector<size_t> coalesce(int id, std::vector<size_t> offsets,
                                 size_t seg_size, uint32_t gap, int n_readers,
                                 size_t used_chks_per_read);
    void enqueue_reads(int id, size_t seg_size, size_t n_segs_per_read,
                       size_t total_n_segs, size_t total_read_size);
    void stage_batch_if_ready(int id, std::vector<segment_t> &segments,
                              size_t &wait_for_count, size_t &n_seg_in_segvec,
                              int n_readers, size_t used_chks_per_read);
    void stage_final_batch(int id, std::vector<segment_t> &segments,
                           size_t wait_for_count, size_t n_seg_in_segvec,
                           int n_readers);

  public:
    data_loader_t();

    ~data_loader_t();

    int file_load(FileReader &io_reader, size_t seg_size,
                  TransferType trans_type);
    // std::pair<int, std::vector<size_t>>
    loader_info_t 
    file_load(FileReader &io_reader0, FileReader &io_reader1,
              std::vector<size_t> offsets, size_t seg_size,
              TransferType trans_type, uint32_t gap, size_t block_size);
    loader_info_t 
    file_load(FileReader &io_reader0, FileReader &io_reader1,
              std::vector<size_t> offsets, size_t seg_size,
              TransferType trans_type, int nthreads = 16);
    next_batch_t next(int loader_id, TransferType trans_type);
    size_t get_wasted_bytes_count(int id);
    size_t get_IOP_count(int id);
};
#endif   // __DATA_LOADER_HPP
