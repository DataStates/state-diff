#include "storage.hpp"

storage_t::storage_t(uint8_t *start, size_t total_size)
    : start_(start), head_(0), tail_(0), total_size_(total_size),
      curr_size_(0) {
    DBG("Store - Storage initialized with start as uint8_t pointer");
}

storage_t::~storage_t() {
    cv_.notify_all();
    stored_segs_.clear();
}

bool
storage_t::can_allocate(size_t req_size) {
    DBG("Store - Trying to allocate "
        << req_size / 1024 << " KB given current size of "
        << curr_size_ / (1024 * 1024) << " MB and total size of "
        << total_size_ / (1024 * 1024) << " MB when storage state is "
        << (curr_size_ == total_size_ ? "full" : "not full"));
    if (req_size > get_free_size()) {
        return false;
    }
    size_t next_head = (head_ + req_size) % total_size_;
    return next_head != tail_;
}

size_t
storage_t::get_free_size() {
    return total_size_ - curr_size_;
}

size_t
storage_t::get_capacity() {
    return total_size_;
}

void
storage_t::allocate(batch_t *seg_batch) {
    std::unique_lock<std::mutex> lck(mtx_);
    for (size_t i = 0; i < seg_batch->batch_len; i++) {
        segment_t &seg = seg_batch->data[i];
        assert(seg.size < total_size_);
        DBG("Store - Waiting for resources to allocate batch item "
            << i << "/" << seg_batch->batch_len);
        cv_.wait(lck, [this, &seg] { return can_allocate(seg.size); });
        DBG("Store - Resources are now available for batch item "
            << i << "/" << seg_batch->batch_len);
        seg.buffer = start_ + head_;
        head_ = (head_ + seg.size) % total_size_;
        curr_size_ += seg.size;
        stored_segs_.push_back(&seg_batch->data[i]);
        bool was_full = (head_ == tail_);
        if (was_full) {
            head_ = 0;
        }
    }
    lck.unlock();
    cv_.notify_one();
}

void
storage_t::deallocate(batch_t *seg_batch) {
    std::unique_lock<std::mutex> lck(mtx_);
    if (stored_segs_.empty()) {
        FATAL("Deallocate called with no stored segments in storage.");
        return;
    }
    for (size_t i = 0; i < seg_batch->batch_len; i++) {
        segment_t &curr_seg = seg_batch->data[i];
        segment_t *oldest = stored_segs_.front();
        if (oldest->offset != curr_seg.offset) {
            FATAL("FIFO violation: Attempted to deallocate out of order. "
                << "Expected offset: " << oldest->offset
                << ", but got: " << curr_seg.offset);
            return;
        }
        tail_ = (tail_ + curr_seg.size) % total_size_;
        if (tail_ > total_size_)
            tail_ = 0;
        curr_size_ -= curr_seg.size;
        if (curr_size_ == 0)
            head_ = tail_ = 0;
        stored_segs_.pop_front();
        DBG("Store - Deallocated batch item " << i 
            << "/" << seg_batch->batch_len 
            << ", new tail: " << tail_ 
            << ", free space: " << get_free_size());
    }
    lck.unlock();
    cv_.notify_one();
}
