#ifndef __BASE_CACHE_HPP
#define __BASE_CACHE_HPP

#include "metadata_queue.hpp"
#include "storage.hpp"
#include <shared_mutex>
#include <unordered_map>

class base_cache_t {
  protected:
    size_t tot_cache_size_ = 0;
    // std::thread fetch_thread_;
    // std::thread flush_thread_;
    std::unordered_map<int, std::thread> fetch_thread_;
    std::unordered_map<int, metadata_queue> fetch_q_;
    std::unordered_map<int, metadata_queue> ready_q_;
    // metadata_queue fetch_q_;
    // metadata_queue ready_q_;
    // bool is_active_ = true;
    std::atomic<bool> is_active_ = true;
    // std::condition_variable cv;
    std::unordered_map<int, std::shared_mutex> fetch_q_mutex_;
    std::unordered_map<int, std::shared_mutex> ready_q_mutex_;

  public:
    storage_t *data_store_ = nullptr;
    base_cache_t(size_t total_size) : tot_cache_size_(total_size) {};
    virtual ~base_cache_t() {};
    virtual void activate(int id) = 0;
    virtual void stage_in(int id, batch_t *seg_batch) = 0;
    virtual void stage_out(int id, batch_t *seg_batch) = 0;
    virtual void fetch_(int id) = 0;
    virtual bool wait_for_completion() = 0;
    virtual batch_t *get_completed(int id) = 0;
    virtual bool release(int id) = 0;
    virtual void coalesce_and_copy(batch_t *consumed_item, void *ptr) = 0;
};
#endif   //__BASE_CACHE_HPP