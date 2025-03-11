#ifndef __BATCH_HPP
#define __BATCH_HPP

#include "debug.hpp"
#include "io_reader.hpp"
#include <cassert>
#include <iostream>
#include <vector>

struct batch_t {
    segment_t *data;
    // Number of segments expected to be in a batch
    size_t batch_len;
    // count of offsets to process in the batch. proc_offt is by default 0,
    // i.e., all bytes, unless specified in the constructor
    size_t proc_offt;
    // used to track objects added to the batch (post-incremented, i.e. ends as
    // batch_size)
    size_t count;
    size_t size;

    batch_t(size_t size_, size_t offt_count = 0)
        : batch_len(size_), proc_offt(offt_count), count(0), size(0) {
        data = new segment_t[batch_len];
    }
    batch_t &operator=(const batch_t &) = delete;
    batch_t(const batch_t &other)
        : batch_len(other.batch_len), proc_offt(other.proc_offt), count(0),
          size(other.size) {
        data = new segment_t[batch_len];
        for (size_t i = 0; i < other.batch_len; i++) {
            INFO("Building batch of size "
                 << other.batch_len
                 << " with items at offset = " << other.data[i].offset
                 << ", size = " << other.data[i].size / 1024 << "KB");
            data[i] = other.data[i];
            data[i].buffer = nullptr;   // Buffer not allocated yet
            count++;
        }
    }
    batch_t(batch_t *other) : batch_t(*other) {}
    ~batch_t() { delete[] data; }
    void push(segment_t item) {
        assert(count < batch_len);
        data[count++] = item;
        size += item.size;
    }
    void inc_last(size_t inc_size) { data[count - 1].size += inc_size; }
    std::vector<segment_t> to_vec() {
        return std::vector<segment_t>(data, data + batch_len);
    }
    std::vector<segment_t> left_vec() {
        size_t half_size = batch_len / 2;
        return std::vector<segment_t>(data, data + half_size);
    }
    std::vector<segment_t> right_vec() {
        size_t half_size = batch_len / 2;
        return std::vector<segment_t>(data + half_size, data + batch_len);
    }
};

#endif   //__BATCH_HPP