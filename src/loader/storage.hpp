#ifndef __STORAGE_HPP
#define __STORAGE_HPP
#include "common/debug.hpp"
#include "common/segment.hpp"
#include <cassert>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <queue>

class storage_t {

    uint8_t *start_ = nullptr;
    std::mutex mtx_;
    std::condition_variable cv_;
    size_t head_;
    size_t tail_;
    size_t total_size_;
    size_t curr_size_;
    bool storage_ful_ = false;
    std::deque<segment_t *> stored_segs_;

    bool can_allocate(size_t req_size);

  public:
    storage_t(uint8_t *start, size_t total_size);
    ~storage_t();
    void allocate(batch_t *seg);
    void deallocate(batch_t *seg);
    size_t get_free_size();
    size_t get_capacity();
};
#endif   //__STORAGE_HPP