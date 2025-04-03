#include "host_cache.hpp"

host_cache_t::~host_cache_t() {
    wait_for_completion();
    is_active_ = false;
    for (auto &fqueue : fetch_q_)
        fqueue.second.set_inactive();

    for (auto &rqueue : ready_q_)
        rqueue.second.set_inactive();

    for (auto &thread : fetch_thread_) {
        if (thread.second.joinable()) {
            thread.second.join();
        }
    }
    DBG("Host - Cache destroyed");
};

void
host_cache_t::stage_in(int id, batch_t *seg_batch) {
    DBG("Host (" << id << ")- Allocating memory to batch of size "
                 << seg_batch->batch_len);
    seg_batch->allocate();
    fetch_q_[id].push(seg_batch);
    DBG("Host (" << id << ")- Staged batch of size " << seg_batch->batch_len
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

    // enqueue first read request
    batch_t *item = fetch_q_[id].front();
    io_reader->enqueue_reads(item->to_vec());
}

void
host_cache_t::set_reader(int id, FileReader *io_reader0,
                         FileReader *io_reader1) {
    DBG("Host (" << id << ")- Setting reader to read from file");
    assert(io_reader0 != nullptr && io_reader1 != nullptr);
    freader_[id] = std::make_pair(io_reader0, io_reader1);

    // enqueue first read requests
    batch_t *item = fetch_q_[id].front();
    io_reader0->enqueue_reads(item->left_vec());
    io_reader1->enqueue_reads(item->right_vec());
}

void
host_cache_t::submit_work(int id) {
    FileReader *single_reader = nullptr;
    std::pair<FileReader *, FileReader *> *reader_pair = nullptr;
    if (auto *reader = std::get_if<FileReader *>(&freader_[id])) {
        single_reader = *reader;
    } else if (auto *pair_reader =
                   std::get_if<std::pair<FileReader *, FileReader *>>(
                       &freader_[id])) {
        reader_pair = pair_reader;
    } else {
        DBG("Error: No valid reader found!");
        return;
    }

    // Wait for previous batch to complete loading
    batch_t *prev_batch = fetch_q_[id].front();
    if (single_reader) {
        single_reader->wait_all();
    } else {
        reader_pair->first->wait_all();
        reader_pair->second->wait_all();
    }
    // stage out previous batch and pop from fetch queue
    DBG("Host (" << id << ")- Adding ready batch to host ready queue");
    stage_out(id, prev_batch);
    fetch_q_[id].pop();

    // Enqueue new batch for read if there are more work
    if (fetch_q_[id].size() > 0) {
        batch_t *new_batch = fetch_q_[id].front();
        if (single_reader) {
            single_reader->enqueue_reads(new_batch->to_vec());
        } else {
            reader_pair->first->enqueue_reads(new_batch->left_vec());
            reader_pair->second->enqueue_reads(new_batch->right_vec());
        }
    }
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
    submit_work(id);
    return ready_q_[id].front();
}

bool
host_cache_t::release(int id) {
    DBG("Host (" << id
                 << ")- Releasing memory used by previous processed batch");
    ready_q_[id].pop();
    return true;
}

void
host_cache_t::coalesce_and_copy(batch_t *consumed_item, void *ptr) {
    uint8_t *destination = static_cast<uint8_t *>(ptr);
    for (size_t i = 0; i < consumed_item->batch_len; i++) {
        DBG("Host - Coalescing batch item "
            << i << "/" << consumed_item->batch_len << " on host");
        segment_t &segment = consumed_item->data[i];
        std::memcpy(destination, segment.buffer, segment.size);
        destination += segment.size;
    }
}