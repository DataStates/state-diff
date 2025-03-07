#ifndef __HOST_CACHE_HPP
#define __HOST_CACHE_HPP

#include "base_cache.hpp"
#include "io_reader.hpp"
#include <variant>
// #include "async_data_reader.hpp"

class host_cache_t : public base_cache_t {
    using FileReader = base_io_reader_t;
    using SingleOrDoubleReader = std::variant<FileReader*, std::pair<FileReader*, FileReader*>>;

    // FileReader *freader_ = nullptr;
    // std::unordered_map<int, FileReader *> freader_;
    std::unordered_map<int, SingleOrDoubleReader> freader_;
    uint8_t *start_ptr_ = nullptr;

  public:
    host_cache_t(int gpu_id, size_t tot_cache_size);
    ~host_cache_t();
    void set_reader(int id, FileReader *io_reader);
    void set_reader(int id, FileReader *io_reader0, FileReader *io_reader1);
    void activate(int id);
    void stage_in(int id, batch_t *seg_batch);
    void stage_out(int id, batch_t *seg_batch);
    void fetch_(int id);
    bool wait_for_completion();
    batch_t* get_completed(int id);
    bool release(int id);
    void coalesce_and_copy(batch_t* consumed_item, void *ptr);
};
#endif   // __HOST_CACHE_HPP