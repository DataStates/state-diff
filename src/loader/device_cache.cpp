#include "device_cache.hpp"

device_cache_t::device_cache_t(int gpu_id, size_t tot_cache_size)
    : base_cache_t(gpu_id, tot_cache_size) {
    gpuErrchk(cudaSetDevice(gpu_id_));
    gpuErrchk(cudaMalloc((void **)&start_ptr_, tot_cache_size_));
    INFO("Device - Creating a cache of size " << tot_cache_size / (1024 * 1024)
                                              << " MB");
    data_store_ = new storage_t(start_ptr_, tot_cache_size_);
    gpuErrchk(cudaStreamCreateWithFlags(&h2d_stream_, cudaStreamNonBlocking));
}

device_cache_t::~device_cache_t() {
    wait_for_completion();
    is_active_ = false;
    for (auto &fqueue : fetch_q_)
        fqueue.second.set_inactive();

    for (auto &rqueue : ready_q_)
        rqueue.second.set_inactive();

    gpuErrchk(cudaFree(start_ptr_));
    gpuErrchk(cudaStreamDestroy(h2d_stream_));
    DBG("Device - Cache destroyed");
};

void
device_cache_t::activate(int id) {
    fetch_thread_[id] = std::thread([this, id] { fetch_(id); });   // H2D thread
    fetch_thread_[id].detach();
    INFO("Device (" << id << ")- Started fetch thread on device cache");
}

void
device_cache_t::stage_in(int id, batch_t *seg_batch) {
    fetch_q_[id].push(seg_batch);
    DBG("Device (" << id << ")- Staged batch in for h2d copy");
}

void
device_cache_t::stage_out(int id, batch_t *seg_batch) {
    ready_q_[id].push(seg_batch);
    DBG("Device (" << id << ")- Staged batch for d2h copy");
}

void
device_cache_t::set_next_tier(int id, base_cache_t *cache_tier) {
    DBG("Device (" << id << ")- Setting next tier for outgoing transfers");
    if (next_cache_tier_ == nullptr)
        next_cache_tier_ = cache_tier;
    next_cache_tier_->activate(id);
    flush_thread_[id] = std::thread([&, id] { flush_(id); });
    flush_thread_[id].detach();
    INFO("Device (" << id << ")- Started flush threads on device cache");
}

void
device_cache_t::fetch_(int id) {
    while (is_active_) {
        // wait for item
        DBG("Device (" << id
                       << ")- Waiting for items to be pushed onto the fetch_q");
        TIMER_START(dev_waitfetch);
        bool res = fetch_q_[id].wait_any();
        TIMER_STOP(dev_waitfetch,
                   "Device (" << id << ")- Waited any batch for device fetch");
        if (!res)
            FATAL("Undefined behavior in fetch metadata queue of device cache");
        // get item and transfer item to destination
        CudaTimer transfer_timer("h2d_cpy");
        TIMER_START(dev_fetch);
        size_t curr_capacity = fetch_q_[id].size();
        for (size_t i = 0; i < curr_capacity; i++) {
            batch_t *host_item = fetch_q_[id].front();
            batch_t *dev_item = new batch_t(host_item);
            DBG("Device (" << id << ")- Allocating memory to front batch");
            data_store_->allocate(dev_item);
            DBG("Device (" << id << ")- H2D transfer of batch");
            transfer_timer.start(h2d_stream_);
            for (size_t i = 0; i < dev_item->batch_size; i++) {
                gpuErrchk(cudaMemcpyAsync(dev_item->data[i].buffer,
                                          host_item->data[i].buffer,
                                          host_item->data[i].size,
                                          cudaMemcpyHostToDevice, h2d_stream_));
            }
            gpuErrchk(cudaStreamSynchronize(h2d_stream_));
            transfer_timer.stop(h2d_stream_);
            DBG("Device (" << id << ")- Adding item to ready queue");
            stage_out(id, dev_item);   // not needed at this stage
            fetch_q_[id].pop();
        }
        transfer_timer.finalize();
        float h2d_time = transfer_timer.getTotalTime();
        TIMER_STOP(dev_fetch, "Device ("
                                  << id << ")- Fetched " << curr_capacity
                                  << " batches to device with h2d_cpy of "
                                  << h2d_time << " ms");
    }
    INFO("Device (" << id << ")- Fetch thread exiting\n");
}

void
device_cache_t::flush_(int id) {
    while (is_active_) {
        DBG("Device (" << id
                       << ")- Waiting for item to be loaded on ready queue");
        TIMER_START(dev_waitflush);
        bool res = ready_q_[id].wait_any();
        TIMER_STOP(dev_waitflush,
                   "Device (" << id << ")- Waited any batch for device flush");
        if (!res)
            FATAL("Undefined behavior in flush metadata queue of device cache");
        TIMER_START(dev_flush);
        size_t curr_capacity = ready_q_[id].size();
        for (size_t i = 0; i < curr_capacity; i++) {
            batch_t *item = ready_q_[id].front();
            next_cache_tier_->stage_in(id, item);
            data_store_->deallocate(item);
            ready_q_[id].pop();
        }
        TIMER_STOP(dev_flush, "Device (" << id << ")- Flushed and deallocated "
                                         << curr_capacity
                                         << " batches from device cache");
    }
    INFO("Device (" << id << ")- Flush thread exiting");
}

bool
device_cache_t::wait_for_completion() {
    INFO("Device - Waiting for all jobs on fetch_q to be completed");
    for (auto &fqueue : fetch_q_)
        fqueue.second.wait_for_completion();

    for (auto &rqueue : ready_q_)
        rqueue.second.wait_for_completion();
    return true;
}

batch_t *
device_cache_t::get_completed(int id) {
    INFO("Device (" << id << ")- Getting completed jobs from ready_q");
    ready_q_[id].wait_any();
    // batch_t *front_batch = ready_q_[id].front();
    // ready_q_[id].pop();
    // return front_batch;
    return ready_q_[id].front();
}

bool
device_cache_t::release(int id) {
    INFO("Device (" << id
                    << ")- Releasing memory used by previous processed batch");
    batch_t *consumed_item = ready_q_[id].front();
    data_store_->deallocate(consumed_item);
    ready_q_[id].pop();
    return true;
}

// run this in a kernel not on CPU
void
device_cache_t::coalesce_and_copy(batch_t *consumed_item, void *ptr) {
    DBG("Device - Coalescing batch on device");
    uint8_t *destination = static_cast<uint8_t *>(ptr);
    size_t trans_size = consumed_item->data[0].size * consumed_item->batch_size;
    gpuErrchk(cudaMemcpy(destination, &consumed_item->data[0], trans_size,
                         cudaMemcpyDeviceToDevice));
    data_store_->deallocate(consumed_item);
}