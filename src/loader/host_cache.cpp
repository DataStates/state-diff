#include "host_cache.hpp"

host_cache_t::host_cache_t(int gpu_id, size_t tot_cache_size)
    : base_cache_t(gpu_id, tot_cache_size) {
    gpuErrchk(cudaSetDevice(gpu_id_));
    gpuErrchk(cudaMallocHost((void **)&start_ptr_, tot_cache_size_));
    INFO("Host - Creating a cache of size " << tot_cache_size / (1024 * 1024)
                                            << " MB");
    data_store_ = new storage_t(start_ptr_, tot_cache_size_);
}

host_cache_t::~host_cache_t() {
    wait_for_completion();
    is_active_ = false;
    for (auto& fqueue : fetch_q_)
        fqueue.second.set_inactive();

    for (auto& rqueue : ready_q_)
        rqueue.second.set_inactive();
    
    DBG("Host - Cache destroyed");
};

void
host_cache_t::activate(int id) {
    fetch_thread_[id] = std::thread([this, id] { fetch_(id); });
    fetch_thread_[id].detach();
    INFO("Host (" << id << ")- Started fetch thread on host cache");
}

void
host_cache_t::stage_in(int id, batch_t *seg_batch) {
    fetch_q_[id].push(seg_batch);
    INFO("Host (" << id << ")- Staged batch of size " << seg_batch->batch_size << " for f2h copy");
}

void
host_cache_t::stage_out(int id, batch_t *seg_batch) {
    ready_q_[id].push(seg_batch);
    INFO("Host (" << id << ")- Staged batch out for h2d copy");
}

void
host_cache_t::set_reader(int id, FileReader *io_reader) {
    INFO("Host (" << id << ")- Setting reader to read from file");
    assert(io_reader != nullptr);
    freader_[id] = io_reader;
    activate(id);
}

void
host_cache_t::set_next_tier(int id, base_cache_t *cache_tier) {
    DBG("Host (" << id << ")- Setting next tier for outgoing transfers");
    if(next_cache_tier_ == nullptr)
        next_cache_tier_ = cache_tier;
    next_cache_tier_->activate(id);
    flush_thread_[id] = std::thread([this, id] { flush_(id); });
    flush_thread_[id].detach();
    INFO("Host (" << id << ")- Started flush threads");
}

void
host_cache_t::fetch_(int id) {
    while (is_active_) {
        // wait for item
        DBG("Host (" << id << ")- Waiting for items to be pushed onto the fetch_q");
        TIMER_START(hst_waitfetch);
        bool res = fetch_q_[id].wait_any();
        TIMER_STOP(hst_waitfetch, "(" << id << ")- Waited any batch for host fetch");
        if (!res)
            FATAL("Undefined behavior in fetch metadata queue of host cache");
        TIMER_START(hst_fetch);
        size_t curr_capacity = fetch_q_[id].size();
        for (size_t i = 0; i < curr_capacity; i++) {
            batch_t *item = fetch_q_[id].front();
            DBG("Host (" << id << ")- Allocating memory to front batch of size "
                << item->batch_size);
            data_store_->allocate(item);
            DBG("Host (" << id << ")- Enqueuing for read from file");
            freader_[id]->enqueue_reads(item->to_vec());
            freader_[id]->wait_all();
            DBG("Host (" << id << ")- Adding item to host ready queue");
            stage_out(id, item);
            fetch_q_[id].pop();
        }
        TIMER_STOP(hst_fetch,
                   "(" << id << ")- Fetched " << curr_capacity << " batches to host cache");
    }
    INFO("Host (" << id << ")- Fetch thread exiting");
}

void
host_cache_t::flush_(int id) {
    while (is_active_) {
        DBG("Host (" << id << ")- Waiting for item to be loaded on ready queue");
        TIMER_START(dev_waitflush);
        bool res = ready_q_[id].wait_any();
        TIMER_STOP(dev_waitflush, "(" << id << ")- Waited any batch for host flush");
        if (!res)
            FATAL("Undefined behavior in flush metadata queue of host cache");
        TIMER_START(hst_flush);
        size_t curr_capacity = ready_q_[id].size();
        for (size_t i = 0; i < curr_capacity; i++) {
            batch_t *item = ready_q_[id].front();
            next_cache_tier_->stage_in(id, item);
            data_store_->deallocate(item);
            ready_q_[id].pop();
        }
        TIMER_STOP(hst_flush, "(" << id << ")- Flushed and deallocated "
                                  << curr_capacity
                                  << " batches from host cache");
    }
    INFO("Host (" << id << ")- Flush thread exiting\n");
}

bool
host_cache_t::wait_for_completion() {
    INFO("Host - Waiting for all jobs on fetch_q to be completed");
    for (auto& fqueue : fetch_q_)
        fqueue.second.wait_for_completion();
    
    for (auto& rqueue : ready_q_)
        rqueue.second.wait_for_completion();
    return true;
}

batch_t *
host_cache_t::get_completed(int id) {
    DBG("Host (" << id << ")- Getting completed jobs from ready_q");
    ready_q_[id].wait_any();
    batch_t *front_batch = ready_q_[id].front();
    ready_q_[id].pop();
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