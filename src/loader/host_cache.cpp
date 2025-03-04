#include "host_cache.hpp"

host_cache_t::host_cache_t(int gpu_id, size_t tot_cache_size)
    : base_cache_t(gpu_id, tot_cache_size) {
#ifdef __NVCC__
    gpuErrchk(cudaSetDevice(gpu_id_));
    gpuErrchk(cudaMallocHost((void **)&start_ptr_, tot_cache_size_));
#else
    start_ptr_ = (uint8_t *)malloc(tot_cache_size_);
#endif
    INFO("Host - Creating a cache of size " << tot_cache_size / (1024 * 1024)
                                            << " MB");
    data_store_ = new storage_t(start_ptr_, tot_cache_size_);
}

host_cache_t::~host_cache_t() {
    wait_for_completion();
    is_active_ = false;
    for (auto &fqueue : fetch_q_)
        fqueue.second.set_inactive();

    for (auto &rqueue : ready_q_)
        rqueue.second.set_inactive();

    for (auto &thread : fetch_thread_) {
        if (thread.second.joinable()) {
            thread.second.join();   // Join thread if it is joinable
        }
    }

    for (auto &thread : flush_thread_) {
        if (thread.second.joinable()) {
            thread.second.join();   // Join thread if it is joinable
        }
    }

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
    bool singlereader = std::holds_alternative<FileReader *>(freader_[id]);
    fetch_q_[id].push(seg_batch);
    int count = 1;
    if (!singlereader) {
        batch_t *seg_batch_cpy = new batch_t(seg_batch);
        // batch_t seg_batch_cpy = seg_batch;
        fetch_q_[id].push(seg_batch_cpy);
        count++;
    }
    DBG("Host (" << id << ")- Staged batch of size " << seg_batch->batch_size
                 << " (" << count << " readers) for f2h copy");
}

void
host_cache_t::stage_out(int id, batch_t *seg_batch) {
    ready_q_[id].push(seg_batch);
    DBG("Host (" << id << ")- Staged batch out for h2d copy");
}

void
host_cache_t::set_reader(int id, FileReader *io_reader) {
    INFO("Host (" << id << ")- Setting reader to read from file");
    assert(io_reader != nullptr);
    freader_[id] = io_reader;
    activate(id);
}

void
host_cache_t::set_reader(int id, FileReader *io_reader0,
                         FileReader *io_reader1) {
    DBG("Host (" << id << ")- Setting reader to read from file");
    assert(io_reader0 != nullptr && io_reader1 != nullptr);
    freader_[id] = std::make_pair(io_reader0, io_reader1);
    activate(id);
}

void
host_cache_t::set_next_tier(int id, base_cache_t *cache_tier) {
    DBG("Host (" << id << ")- Setting next tier for outgoing transfers");
    if (next_cache_tier_ == nullptr)
        next_cache_tier_ = cache_tier;
    next_cache_tier_->activate(id);
    flush_thread_[id] = std::thread([this, id] { flush_(id); });
    flush_thread_[id].detach();
    DBG("Host (" << id << ")- Started flush threads on host cache");
}

void
host_cache_t::fetch_(int id) {
    bool use_reader0 = true;

    while (is_active_) {
        // wait for item
        DBG("Host (" << id
                     << ")- Waiting for items to be pushed onto the fetch_q");
        TIMER_START(hst_waitfetch);
        bool res = fetch_q_[id].wait_any();
        TIMER_STOP(hst_waitfetch,
                   "Host (" << id << ")- Waited any batch for host fetch");
        if (!res) {
            DBG("Error in fetch metadata queue of host cache, retrying...");
            continue;
        }
        TIMER_START(hst_fetch);
        size_t curr_capacity = fetch_q_[id].size();
        for (size_t i = 0; i < curr_capacity; i++) {
            batch_t *item = fetch_q_[id].front();
            DBG("Host (" << id << ")- Allocating memory to front batch of size "
                         << item->batch_size);
            data_store_->allocate(item);

            if (auto *reader_pair =
                    std::get_if<std::pair<FileReader *, FileReader *>>(
                        &freader_[id])) {
                FileReader *selected_reader =
                    use_reader0 ? reader_pair->first : reader_pair->second;
                // switch to use the other reader for next batch
                use_reader0 = !use_reader0;
                DBG("Host (" << id << ")- Enqueuing for read from "
                             << (use_reader0 ? "reader0" : "reader1"));
                selected_reader->enqueue_reads(item->to_vec());
                selected_reader->wait_n(item->batch_size);
            } else if (auto *single_reader =
                           std::get_if<FileReader *>(&freader_[id])) {
                DBG("Host (" << id << ")- Enqueuing for read from file");
                (*single_reader)->enqueue_reads(item->to_vec());
                (*single_reader)->wait_n(item->batch_size);
            } else {
                DBG("Error: No valid reader found!");
                continue;
            }

            DBG("Host (" << id << ")- Adding item to host ready queue");
            stage_out(id, item);
            fetch_q_[id].pop();
        }
        TIMER_STOP(hst_fetch, "Host (" << id << ")- Fetched " << curr_capacity
                                       << " batches to host cache");
    }
    DBG("Host (" << id << ")- Fetch thread exiting");
}

void
host_cache_t::flush_(int id) {
    while (is_active_) {
        DBG("Host (" << id
                     << ")- Waiting for item to be loaded on ready queue");
        TIMER_START(dev_waitflush);
        bool res = ready_q_[id].wait_any();
        TIMER_STOP(dev_waitflush,
                   "Host (" << id << ")- Waited any batch for host flush");
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
        TIMER_STOP(hst_flush, "Host (" << id << ")- Flushed and deallocated "
                                       << curr_capacity
                                       << " batches from host cache");
    }
    DBG("Host (" << id << ")- Flush thread exiting\n");
}

bool
host_cache_t::wait_for_completion() {
    DBG("Host - Waiting for all jobs on fetch_q to be completed");
    for (auto &fqueue : fetch_q_)
        fqueue.second.wait_for_completion();

    for (auto &rqueue : ready_q_)
        rqueue.second.wait_for_completion();
    return true;
}

batch_t *
host_cache_t::get_completed(int id) {
    DBG("Host (" << id << ")- Getting completed jobs from ready_q");
    bool singlereader = std::holds_alternative<FileReader *>(freader_[id]);
    if(singlereader){
        ready_q_[id].wait_any();
    } else {
        ready_q_[id].wait_for(2); // 2 for two readers (one batch per reader)
    }
    return ready_q_[id].front();
}

bool
host_cache_t::release(int id) {
    DBG("Host (" << id
                  << ")- Releasing memory used by previous processed batch");
    int nreaders = std::holds_alternative<FileReader *>(freader_[id]) ? 1 : 2;
    for(int i = 0; i < nreaders; i++) {
        batch_t *consumed_item = ready_q_[id].front();
        data_store_->deallocate(consumed_item);
        ready_q_[id].pop();
        delete consumed_item; // Prevent memory leak
    }
    return true;
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