#ifndef __DEVICE_CACHE_HPP
#define __DEVICE_CACHE_HPP

#include "base_cache.hpp"
#include "common/io_utils.hpp"
#include "cuda_timer.hpp"
#include <cuda_runtime.h>

class device_cache_t : public base_cache_t {
    uint8_t *start_ptr_ = nullptr;
    cudaStream_t h2d_stream_;

  public:
    device_cache_t(int gpu_id, size_t tot_cache_size);
    ~device_cache_t();
    void activate(int id);
    void stage_in(int id, batch_t *seg_batch);
    void stage_out(int id, batch_t *seg_batch);
    void fetch_(int id);
    void flush_(int id);
    void set_next_tier(int id, base_cache_t *cache_tier);
    bool wait_for_completion();
    batch_t *get_completed(int id);
    bool release(int id);
    void coalesce_and_copy(batch_t *consumed_item, void *ptr);
};
#endif   // __DEVICE_CACHE_HPP