#ifndef __SEGMENT_BATCH_HPP
#define __SEGMENT_BATCH_HPP

#include "common/debug.hpp"
#include <cassert>
#include <iostream>
#include <vector>

struct segment_t {
    size_t offset;     // Start offset in file
    size_t size;       // Size of the memory region
    int fd;            // Descriptor of corresponding file
    uint8_t *buffer;   // Pointer of this segment's memory region (GPU or CPU)

    segment_t() = default;
    segment_t(size_t offset_, size_t size_)
        : offset(offset_), size(size_), fd(0), buffer(nullptr){};
    segment_t(segment_t *other)
        : offset(other->offset), size(other->size), fd(other->fd),
          buffer(nullptr){};
};

struct batch_t {
    segment_t *data;
    size_t batch_size;   // Number of segments in a batch
    size_t count;

    batch_t(size_t size_) : batch_size(size_), count(0) {
        data = new segment_t[batch_size];
    }
    batch_t &operator=(const batch_t &) = delete;
    batch_t(batch_t *other) : batch_size(other->batch_size), count(0) {
        data = new segment_t[batch_size];
        for (size_t i = 0; i < other->batch_size; i++) {
            DBG("Building batch of size "
                << other->batch_size
                << " with items at offset = " << other->data[i].offset
                << ", size = " << other->data[i].size / 1024 << "KB");
            data[i] = segment_t(other->data[i]);
            data[i].buffer = nullptr;
            count++;
        }
    }
    ~batch_t() { delete[] data; }
    void push(segment_t item) {
        assert(count < batch_size);
        data[count++] = item;
    }
    void inc_last(size_t inc_size) { data[count - 1].size += inc_size; }
    std::vector<segment_t> to_vec() {
        return std::vector<segment_t>(data, data + count);
    }
};

#endif   //__SEGMENT_BATCH_HPP