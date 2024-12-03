#include "storage.hpp"

storage_t::storage_t(uint8_t *start, size_t total_size)
    : start_(start), total_size_(total_size), curr_size_(0), head_(0),
      tail_(0) {
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
        << storage_ful_);
    // return !storage_ful_ && curr_size_ + req_size <= total_size_;
    return !storage_ful_ && get_free_size() >= req_size;
}

size_t
storage_t::get_free_size() {
    // std::unique_lock<std::mutex> lck(mtx_);
    return total_size_ - curr_size_;
}

size_t
storage_t::get_capacity() {
    return total_size_;
}

void
storage_t::allocate(batch_t *seg_batch) {
    std::unique_lock<std::mutex> lck(mtx_);
    for (size_t i = 0; i < seg_batch->batch_size; i++) {
        segment_t &seg = seg_batch->data[i];
        assert(seg.size < total_size_);
        DBG("Store - Waiting for resources to allocate batch item " << i << "/" << seg_batch->batch_size);
        cv_.wait(lck, [this, &seg] { return can_allocate(seg.size); });
        DBG("Store - Resources are now available for batch item " << i << "/" << seg_batch->batch_size);
        seg.buffer = start_ + head_;
        head_ = (head_ + seg.size) % total_size_;
        curr_size_ += seg.size;
        stored_segs_.push_back(&seg_batch->data[i]);
        storage_ful_ = (head_ == tail_);
        if (storage_ful_)
            head_ = 0;
    }
    lck.unlock();
    cv_.notify_one();
}

void
storage_t::deallocate(batch_t *seg_batch) {
    std::unique_lock<std::mutex> lck(mtx_);
    if (stored_segs_.empty()) {
        FATAL("Invalid request to deallocate non-existing segment");
        return;
    }
    for (size_t i = 0; i < seg_batch->batch_size; i++) {
        segment_t &curr_seg = seg_batch->data[i];
        segment_t *front = stored_segs_.front();
        if (front->offset != curr_seg.offset) {
            FATAL("Should deallocate the oldest segment first. FIFO enforced!");
            return;
        }
        tail_ = (tail_ + curr_seg.size) % total_size_;
        curr_size_ -= curr_seg.size;
        storage_ful_ = false;
        if (head_ == tail_)
            tail_ = 0;
        stored_segs_.pop_front();
        DBG("Store - Deallocated resources for batch item " << i << "/" << seg_batch->batch_size);
    }
    lck.unlock();
    cv_.notify_one();
}
