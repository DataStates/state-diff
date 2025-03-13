#ifndef __META_QUEUE_HPP
#define __META_QUEUE_HPP

#include "batch.hpp"
#include <condition_variable>
#include <deque>
#include <thread>

class metadata_queue {

    std::deque<batch_t *> meta_q_;
    std::mutex mtx_;
    std::condition_variable cv_;
    bool is_active_ = true;

  public:
    metadata_queue() = default;

    ~metadata_queue() = default;

    void push(batch_t *seg) {
        std::unique_lock<std::mutex> lock(mtx_);
        meta_q_.push_back(seg);
        cv_.notify_one();
    }

    void pop() {
        std::unique_lock<std::mutex> lock(mtx_);
        cv_.wait(lock, [this] { return !meta_q_.empty() || !is_active_; });
        if (!meta_q_.empty()) {
            meta_q_.pop_front();
            cv_.notify_one();
        }
    }

    batch_t *front() {
        std::unique_lock<std::mutex> lock(mtx_);
        if (meta_q_.empty())
            return nullptr;
        return meta_q_.front();
    }

    // Method to swap the internal queue and return the batches
    bool swap_batches(std::deque<batch_t *> &batches) {
        std::unique_lock<std::mutex> lock(mtx_);
        if (meta_q_.empty()) {
            return false;
        }
        // Swap to avoid acquiring and releasing lock multiple time during
        // processing
        batches.swap(meta_q_);
        return true;
    }

    bool wait_until_empty() {
        std::unique_lock<std::mutex> lock(mtx_);
        cv_.wait(lock, [this] { return meta_q_.empty() || !is_active_; });
        return true;
    }

    bool wait_any() {
        std::unique_lock<std::mutex> lock(mtx_);
        cv_.wait(lock, [this] { return !meta_q_.empty() || !is_active_; });
        return true;
    }

    bool wait_for(size_t count) {
        std::unique_lock<std::mutex> lock(mtx_);
        cv_.wait(lock, [this, count] {
            return meta_q_.size() >= count || !is_active_;
        });
        return true;
    }

    size_t size() {
        std::unique_lock<std::mutex> lock(mtx_);
        return meta_q_.size();
    }

    void set_inactive() {
        std::unique_lock<std::mutex> lock(mtx_);
        is_active_ = false;
        cv_.notify_all();
    };
};

#endif   //__META_QUEUE_HPP