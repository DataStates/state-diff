#include "device_cache.hpp"

device_cache_t::device_cache_t(int gpu_id, size_t tot_cache_size)
    : base_cache_t(gpu_id, tot_cache_size) {
    gpuErrchk(cudaSetDevice(gpu_id_));
    gpuErrchk(cudaMalloc((void **)&start_ptr_, tot_cache_size_));
    INFO("Device - Creating a cache of size " << tot_cache_size / (1024 * 1024) << " MB");
    data_store_ = new storage_t(start_ptr_, tot_cache_size_);
    fetch_thread_ = std::thread([&] { fetch_(); });   // H2D thread
    fetch_thread_.detach();
    gpuErrchk(cudaStreamCreateWithFlags(&h2d_stream_, cudaStreamNonBlocking));
    INFO("Device - Started fetch thread on device cache");
}

device_cache_t::~device_cache_t() {
    wait_for_completion();
    fetch_q_.set_inactive();
    ready_q_.set_inactive();
    if (fetch_thread_.joinable()) {
        fetch_thread_.join();
    }
    is_active_ = false;
    gpuErrchk(cudaFree(start_ptr_));
    gpuErrchk(cudaStreamDestroy(h2d_stream_));
    DBG("Device - Cache destroyed");
};

void
device_cache_t::stage_in(batch_t *seg_batch) {
    fetch_q_.push(seg_batch);
    DBG("Device - batch staged in");
}

void
device_cache_t::stage_out(batch_t *seg_batch) {
    ready_q_.push(seg_batch);
    DBG("Device - batch staged out");
}

void
device_cache_t::set_next_tier(base_cache_t *cache_tier) {
    DBG("Device - Setting next tier for outgoing transfers");
    next_cache_tier_ = cache_tier;
    flush_thread_ = std::thread([&] { flush_(); });
    flush_thread_.detach();
    INFO("Host - Started flush threads");
}

void
device_cache_t::fetch_() {
    while (is_active_) {
        // wait for item
        DBG("Device - Waiting for items to be pushed onto the fetch_q");
        bool res = fetch_q_.wait_any();
        if (!res)
            FATAL("Undefined behavior in fetch metadata queue of device cache");
        // get item and transfer item to destination
        size_t curr_capacity = fetch_q_.size();
        for (size_t i = 0; i < curr_capacity; i++) {
            batch_t *host_item = fetch_q_.front();
            batch_t *dev_item = new batch_t(host_item);
            DBG("Device - Allocating memory to front batch");
            data_store_->allocate(dev_item);
            DBG("Device - H2D transfer of batch");
            for (size_t i = 0; i < dev_item->batch_size; i++) {
                gpuErrchk(cudaMemcpyAsync(dev_item->data[i].buffer,
                                          host_item->data[i].buffer,
                                          host_item->data[i].size,
                                          cudaMemcpyHostToDevice, h2d_stream_));
            }
            gpuErrchk(cudaStreamSynchronize(h2d_stream_));
            DBG("Device - Adding item to ready queue");
            stage_out(dev_item);
            fetch_q_.pop();
        }
    }
    INFO("Device - Fetch thread exiting\n");
}

void
device_cache_t::flush_() {
    while (is_active_) {
        bool res = ready_q_.wait_any();
        if (!res)
            FATAL("Undefined behavior in flush metadata queue of device cache");
        DBG("Device - Waiting for item to be loaded on ready queue");
        size_t curr_capacity = ready_q_.size();
        for (size_t i = 0; i < curr_capacity; i++) {
            batch_t *item = ready_q_.front();
            next_cache_tier_->stage_in(item);
            ready_q_.pop();
            data_store_->deallocate(item);
        }
    }
    INFO("Device - Flush thread exiting");
}

bool
device_cache_t::wait_for_completion() {
    INFO("Device - Waiting for all jobs on fetch_q to be completed");
    return fetch_q_.wait_for_completion();
}

batch_t *
device_cache_t::get_completed() {
    DBG("Device - Getting completed jobs from ready_q");
    ready_q_.wait_any();
    batch_t *front_batch = ready_q_.front();
    ready_q_.pop();
    return front_batch;
}

// run this in a kernel not on CPU
void
device_cache_t::coalesce_and_copy(batch_t *consumed_item, void *ptr) {
    uint8_t *destination = static_cast<uint8_t *>(ptr);
    for (size_t i = 0; i < consumed_item->batch_size; i++) {
        DBG("Device - Coalescing batch item " << i << "/" << consumed_item->batch_size << " on device");
        segment_t &segment = consumed_item->data[i];
        gpuErrchk(cudaMemcpy(destination, segment.buffer, segment.size,
                             cudaMemcpyDeviceToDevice));
        destination += segment.size;
    }
    data_store_->deallocate(consumed_item);
}