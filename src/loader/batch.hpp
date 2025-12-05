#ifndef __BATCH_HPP
#define __BATCH_HPP

#include "debug.hpp"
#include "io_reader.hpp"
#include <cassert>
#include <iostream>
#include <vector>

struct batch_t {
    std::vector<segment_t> data;
    size_t batch_len; // Max #of segments in a batch
    size_t proc_offt; // #of requested (logical) offsets in this batch
    size_t count;     // #segments actually pushed
    size_t size;      // total bytes (sum of segment sizes)
    uint8_t* batch_ptr = nullptr;

    size_t chunk_size_bytes_ = 0;           // bytes per chunk
    std::vector<size_t> seg_first_chunk_;   // Position of 1st chunk in each seg (in chunks)
    std::vector<size_t> seg_num_chunks_;    // #of chunks in each seg
    size_t total_n_chunks_ = 0;              // Total #of seg in batch

    size_t interleave_ = 1;      // set to 2 for {prev,cur}
    size_t update_counter_ = 0;  // 0 => first of pair, 1 => second

    batch_t(size_t size_, size_t offt_count = 0, size_t chunk_size_bytes = 0, size_t interleave = 1)
        : data(size_), batch_len(size_), proc_offt(offt_count),
          count(0), size(0), chunk_size_bytes_(chunk_size_bytes),
          interleave_(interleave) {
        assert((chunk_size_bytes_ % 4096) == 0 && "chunk_size must be O_DIRECT aligned");
    }

    batch_t &operator=(const batch_t &) = delete;

    batch_t(const batch_t &other)
        : data(other.batch_len), batch_len(other.batch_len),
          proc_offt(other.proc_offt), count(0), size(other.size),
          chunk_size_bytes_(other.chunk_size_bytes_),
          seg_first_chunk_(other.seg_first_chunk_),
          seg_num_chunks_(other.seg_num_chunks_),
          total_n_chunks_(other.total_n_chunks_) {

        for (size_t i = 0; i < count; ++i) {
            DBG("Building batch of size "
                 << other.batch_len
                 << " with items at offset = " << other.data[i].offset
                 << ", size = " << other.data[i].size / 1024 << "KB");
            data[i] = other.data[i];
            // buffer isn't shared as batch obj is reusable across mem tiers
            data[i].buffer = nullptr;
        }
    }

    batch_t(batch_t *other) : batch_t(*other) {}

    ~batch_t() {
        if (batch_ptr) 
            free(batch_ptr); 
        batch_ptr=nullptr;
    }

    // Push one segment and update mapping counters
    void push(segment_t item) {
        assert(count < batch_len);

        // Enforce O_DIRECT alignment
        // size_t lblk_size = 4096;
        assert((item.offset % 4096) == 0 && "segment offset must be 4K-aligned");
        assert((item.size   % 4096) == 0 && "segment size must be 4K-multiple");
        assert(chunk_size_bytes_ > 0 && "chunk_size_bytes_ must be set");
        assert((item.size % chunk_size_bytes_) == 0 && "segment size must be multiple of chunk size");

        if(update_counter_ == 0) {
            seg_first_chunk_.push_back(total_n_chunks_);
            size_t seg_chunks = item.size / chunk_size_bytes_;
            seg_num_chunks_.push_back(seg_chunks);
            total_n_chunks_ += seg_chunks;
        }

        data[count++] = item;
        size += item.size;
        update_counter_ = (update_counter_ + 1) % interleave_;
    }

    void allocate() {
        DBG("Batch - Allocating memory resources to " << batch_len
                                                      << " segments in batch");
        size_t lblk_size = 4096; // Logical block size of 4K used to align O_DIRECT
        assert((size & lblk_size) == 0 && "batch total must be 4K-multiple");

        if (batch_ptr) { 
            free(batch_ptr); 
            batch_ptr = nullptr; 
        }

        void* ptr = nullptr;
        int ret = posix_memalign(&ptr, lblk_size, size);
        if (ret != 0) {
            fprintf(stderr, "posix_memalign failed: %s\n", strerror(ret));
            exit(1);
        }
        batch_ptr = static_cast<uint8_t*>(ptr);
        size_t cur_alloc = 0;
        for (size_t i = 0; i < count; ++i) {
            segment_t& seg = data[i];
            seg.buffer = batch_ptr + cur_alloc;  
            cur_alloc += seg.size;
        }
        assert(cur_alloc == size);
    }

    std::vector<segment_t> to_vec() {
        return std::vector<segment_t>(data.begin(), data.begin() + count);
    }
    std::vector<segment_t> left_vec() {
        return std::vector<segment_t>(data.begin(), data.begin() + count / 2);
    }
    std::vector<segment_t> right_vec() {
        return std::vector<segment_t>(data.begin() + count / 2,
                                      data.begin() + count);
    }
    const size_t n_segments() const { return count; }
    const size_t n_chunks()   const { return total_n_chunks_; }
    const std::vector<size_t>& first_chunk_idx() const { return seg_first_chunk_; }
    const std::vector<size_t>& seg_num_chunks()  const { return seg_num_chunks_; }
};

#endif   //__BATCH_HPP