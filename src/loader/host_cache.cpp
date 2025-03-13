#include "host_cache.hpp"

host_cache_t::host_cache_t(int gpu_id, size_t tot_cache_size)
    : base_cache_t(gpu_id, tot_cache_size) {
    start_ptr_ = (uint8_t *)malloc(tot_cache_size_);
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
    delete start_ptr_;
    delete data_store_;
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
    DBG("Host (" << id << ")- Staged batch of size " << seg_batch->batch_size
                 << " for f2h copy");
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
host_cache_t::fetch_(int id) {
    while (is_active_) {
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
        std::deque<batch_t *> batches;
        // Swap batches from the queue (without acquiring and releasing lock
        // multiple times in the for loop)
        if (!fetch_q_[id].swap_batches(batches)) {
            DBG("Error: No batches to fetch.");
            continue;
        }

        size_t curr_capacity = batches.size();
        for (size_t i = 0; i < curr_capacity; i++) {
            batch_t *item = batches[i];
            DBG("Host (" << id << ")- Allocating memory to front batch of size "
                         << item->batch_size);
            data_store_->allocate(item);

            if (auto *single_reader =
                    std::get_if<FileReader *>(&freader_[id])) {
                DBG("Host (" << id << ")- Enqueuing for read from file");
                (*single_reader)->enqueue_reads(item->to_vec());
                (*single_reader)->wait_n(item->batch_len);
            } else if (auto *reader_pair =
                           std::get_if<std::pair<FileReader *, FileReader *>>(
                               &freader_[id])) {
                std::vector<segment_t> left = item->left_vec();
                std::vector<segment_t> right = item->right_vec();
                reader_pair->first->enqueue_reads(left);
                reader_pair->second->enqueue_reads(right);
                DBG("Host (" << id << ")- Enqueuing for read from two files");
                reader_pair->first->wait_n(item->batch_len / 2);
                reader_pair->second->wait_n(item->batch_len / 2);
            } else {
                DBG("Error: No valid reader found!");
                continue;
            }
            DBG("Host (" << id << ")- Adding item to host ready queue");
            stage_out(id, item);
        }
        TIMER_STOP(hst_fetch, "Host (" << id << ")- Fetched " << curr_capacity
                                       << " batches to host cache");
    }
    DBG("Host (" << id << ")- Fetch thread exiting");
}

bool
host_cache_t::wait_for_completion() {
    DBG("Host - Waiting for all jobs on fetch_q to be completed");
    for (auto &fqueue : fetch_q_)
        fqueue.second.wait_until_empty();

    for (auto &rqueue : ready_q_)
        rqueue.second.wait_until_empty();
    return true;
}

batch_t *
host_cache_t::get_completed(int id) {
    DBG("Host (" << id << ")- Getting completed jobs from ready_q");
    ready_q_[id].wait_any();
    return ready_q_[id].front();
}

bool
host_cache_t::release(int id) {
    DBG("Host (" << id
                 << ")- Releasing memory used by previous processed batch");
    batch_t *consumed_item = ready_q_[id].front();
    data_store_->deallocate(consumed_item);
    ready_q_[id].pop();
    return true;
}

void
host_cache_t::coalesce_and_copy(batch_t *consumed_item, void *ptr) {
    uint8_t *destination = static_cast<uint8_t *>(ptr);
    for (size_t i = 0; i < consumed_item->batch_len; i++) {
        DBG("Host - Coalescing batch item "
            << i << "/" << consumed_item->batch_size << " on host");
        segment_t &segment = consumed_item->data[i];
        std::memcpy(destination, segment.buffer, segment.size);
        destination += segment.size;
    }
    data_store_->deallocate(consumed_item);
}