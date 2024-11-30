#include "host_cache.hpp"

host_cache_t::host_cache_t(int gpu_id, size_t tot_cache_size)
    : base_cache_t(gpu_id, tot_cache_size) {
    gpuErrchk(cudaSetDevice(gpu_id_));
    gpuErrchk(cudaMallocHost((void **)&start_ptr_, tot_cache_size_));
    INFO("Host - Creating a cache of size " << tot_cache_size / (1024 * 1024)
                                            << " MB");
    data_store_ = new storage_t(start_ptr_, tot_cache_size_);
    fetch_thread_ = std::thread([&] { fetch_(); });
    fetch_thread_.detach();
    INFO("Host - Started fetch thread");
}

host_cache_t::~host_cache_t() {
    wait_for_completion();
    is_active_ = false;
    fetch_q_.set_inactive();
    ready_q_.set_inactive();
    DBG("Host - Cache destroyed");
};

void
host_cache_t::stage_in(batch_t *seg_batch) {
    TIMER_START(hst_stagein);
    fetch_q_.push(seg_batch);
    DBG("Host - batch staged in");
    TIMER_STOP(hst_stagein, "Staged batch for f2h copy");
}

void
host_cache_t::stage_out(batch_t *seg_batch) {
    TIMER_START(hst_stageout);
    ready_q_.push(seg_batch);
    DBG("Host - batch staged out");
    TIMER_STOP(hst_stageout, "Host staged batch out for h2d copy");
}

void
host_cache_t::set_reader(FileReader *io_reader) {
    INFO("Host - Setting reader to read from file");
    assert(io_reader != nullptr);
    freader_ = io_reader;
}

void
host_cache_t::set_next_tier(base_cache_t *cache_tier) {
    DBG("Host - Setting next tier for outgoing transfers");
    next_cache_tier_ = cache_tier;
    flush_thread_ = std::thread([&] { flush_(); });
    flush_thread_.detach();
    INFO("Host - Started flush threads");
}

void
host_cache_t::fetch_() {
    while (is_active_) {
        // wait for item
        DBG("Host - Waiting for items to be pushed onto the fetch_q");
        TIMER_START(hst_waitfetch);
        bool res = fetch_q_.wait_any();
        TIMER_STOP(hst_waitfetch, "waited any batch for host fetch");
        if (!res)
            FATAL("Undefined behavior in fetch metadata queue of host cache");
        TIMER_START(hst_fetch);
        size_t curr_capacity = fetch_q_.size();
        for (size_t i = 0; i < curr_capacity; i++) {
            batch_t *item = fetch_q_.front();
            DBG("Host - Allocating memory to front batch of size"
                << item->batch_size);
            data_store_->allocate(item);
            DBG("Host - Enqueuing for read from file");
            freader_->enqueue_reads(item->to_vec());
            freader_->wait_all();
            DBG("Host - Adding item to host ready queue");
            stage_out(item);
            fetch_q_.pop();
        }
        TIMER_STOP(hst_fetch,
                   "fetched " << curr_capacity << " batches to host cache");
    }
    INFO("Host - Fetch thread exiting");
}

void
host_cache_t::flush_() {
    while (is_active_) {
        TIMER_START(dev_waitflush);
        bool res = ready_q_.wait_any();
        TIMER_STOP(dev_waitflush, "waited any batch for host flush");
        if (!res)
            FATAL("Undefined behavior in flush metadata queue of host cache");
        DBG("Host - Waiting for item to be loaded on ready queue");
        TIMER_START(hst_flush);
        size_t curr_capacity = ready_q_.size();
        for (size_t i = 0; i < curr_capacity; i++) {
            batch_t *item = ready_q_.front();
            next_cache_tier_->stage_in(item);
            ready_q_.pop();
            data_store_->deallocate(item);
        }
        TIMER_STOP(hst_flush, "flushed and deallocated "
                                  << curr_capacity
                                  << " batches from host cache");
    }
    INFO("Host - Flush thread exiting\n");
}

bool
host_cache_t::wait_for_completion() {
    INFO("Host - Waiting for all jobs on fetch_q to be completed");
    return fetch_q_.wait_for_completion();
}

batch_t *
host_cache_t::get_completed() {
    DBG("Host - Getting completed jobs from ready_q");
    ready_q_.wait_any();
    batch_t *front_batch = ready_q_.front();
    ready_q_.pop();
    return front_batch;
}

void
host_cache_t::coalesce_and_copy(batch_t *consumed_item, void *ptr) {
    uint8_t *destination = static_cast<uint8_t *>(ptr);
    for (size_t i = 0; i < consumed_item->batch_size; i++) {
        DBG("Host - Coalescing batch item "
            << i << "/" << consumed_item->batch_size << " on host");
        segment_t &segment = consumed_item->data[i];
        std::memcpy(destination, segment.buffer, segment.size);
        destination += segment.size;
    }
    data_store_->deallocate(consumed_item);
}