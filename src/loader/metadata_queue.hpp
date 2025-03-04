#ifndef __META_QUEUE_HPP
#define __META_QUEUE_HPP

#include <thread>
#include <condition_variable>
#include <deque>
// #include "common/segment.hpp"
#include "batch.hpp"

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
        cv_.notify_all();
    }

    void pop() {
        std::unique_lock<std::mutex> lock(mtx_);
        if(!meta_q_.empty()) {
            meta_q_.pop_front();
            cv_.notify_all();
        }
    }

    batch_t *front() {
        std::unique_lock<std::mutex> lock(mtx_);
        if (meta_q_.empty()) return nullptr;
        return meta_q_.front();
    }

    bool wait_for_completion() {
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
        while (meta_q_.size() < count && is_active_) {  // Use a while loop
            cv_.wait(lock);
        }
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