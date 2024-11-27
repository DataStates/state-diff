#ifndef __HOST_CACHE_HPP
#define __HOST_CACHE_HPP

#include "base_cache.hpp"
#include "io_reader.hpp"

class host_cache_t : public base_cache_t {
    using FileReader = base_io_reader_t;

    FileReader *freader_ = nullptr;
    uint8_t *start_ptr_ = nullptr;

  public:
    host_cache_t(int gpu_id, size_t tot_cache_size);
    ~host_cache_t();
    void set_reader(FileReader *io_reader);
    void stage_in(batch_t *seg_batch);
    void stage_out(batch_t *seg_batch);
    void fetch_();
    void flush_();
    void set_next_tier(base_cache_t* cache_tier);
    bool wait_for_completion();
    batch_t* get_completed();
    void coalesce_and_copy(batch_t* consumed_item, void *ptr);
};
#endif   // __HOST_CACHE_HPP