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

// void
// host_cache_t::activate(int id, size_t used_chks_per_read) {
//     fetch_thread_[id] = std::thread([this, id, used_chks_per_read] {
//     fetch_(id, used_chks_per_read); }); fetch_thread_[id].detach();
//     INFO("Host (" << id << ")- Started fetch thread on host cache");
// }

void
host_cache_t::activate(int id) {
    fetch_thread_[id] = std::thread([this, id] { fetch_(id); });
    fetch_thread_[id].detach();
    INFO("Host (" << id << ")- Started fetch thread on host cache");
}

void
host_cache_t::stage_in(int id, batch_t *seg_batch) {
    fetch_q_[id].push(seg_batch);
    DBG("Host (" << id << ")- Staged batch of size " << seg_batch->batch_len
                 << " for f2h copy");
}


// void
// host_cache_t::stage_in(int id, batch_t *seg_batch) {
//     if (auto *reader_pair =
//             std::get_if<std::pair<FileReader *, FileReader *>>(&freader_[id])) {
//         DBG("Host (" << id << ")- Allocating memory to batch of size "
//                      << seg_batch->batch_len);
//         seg_batch->allocate();
//         DBG("Host (" << id << ")- Enqueuing for read from two files");
//         reader_pair->first->enqueue_reads(seg_batch->left_vec());
//         reader_pair->second->enqueue_reads(seg_batch->right_vec());
//     }
//     fetch_q_[id].push(seg_batch);
//     DBG("Host (" << id << ")- Staged batch of size " << seg_batch->batch_len
//                  << " for f2h copy");
// }

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

    while (is_active_) {
        DBG("Host (" << id
                     << ")- Waiting for items to be pushed onto the fetch_q");
        TIMER_START(hst_waitfetch);
        if (!fetch_q_[id].wait_any()) {
            DBG("Error in fetch metadata queue of host cache, retrying...");
            // continue;
        }
        TIMER_STOP(hst_waitfetch,
                   "Host (" << id << ")- Waited any batch for host fetch");

        TIMER_START(hst_fetch);
        std::deque<batch_t *> batches;
        // Swap batches from the queue (without acquiring and releasing lock
        // multiple times in the for loop)
        if (!fetch_q_[id].swap_batches(batches)) {
            DBG("Error: No batches to fetch.");
            // continue;
        }
        size_t curr_capacity = batches.size();
        // size_t nsegs_inbatch;
        for (size_t i = 0; i < curr_capacity; i++) {
            batch_t *item = batches[i];
            if (single_reader) {
                DBG("Host (" << id << ")- Allocating memory to batch of size "
                             << item->batch_len);
                item->allocate();
                DBG("Host (" << id << ")- Enqueuing for read from file");
                single_reader->enqueue_reads(item->to_vec());
                DBG("Host (" << id << ")- Waiting for batch read from file");
                // nsegs_inbatch = item->batch_len;
                // single_reader->wait_n(nsegs_inbatch);
                single_reader->wait_all();
            } else if (reader_pair) {
                DBG("Host (" << id << ")- Allocating memory to batch of size "
                     << item->batch_len);
                item->allocate();
                DBG("Host (" << id << ")- Enqueuing for read from two files");
                reader_pair->first->enqueue_reads(item->left_vec());
                reader_pair->second->enqueue_reads(item->right_vec());
                // size_t nsegs_inbatch = item->batch_len / 2;
                DBG("Host (" << id << ")- Waiting for read of " << item->batch_len / 2
                             << " segments from files");
                // Using reader.wait_n(nsegs_inbatch) allows waiting for n
                // segments but as segments have different sizes, the wait_n
                // call leads to out-of-order access affecting correctness of
                // which chunk is actually different
                reader_pair->first->wait_all();
                reader_pair->second->wait_all();
            }
            DBG("Host (" << id << ")- Adding item to host ready queue");
            stage_out(id, item);
        }
        TIMER_STOP(hst_fetch, "Host (" << id << ")- Fetched " << curr_capacity
                                       << " batches to host cache");
    }
    DBG("Host (" << id << ")- Fetch thread exiting");
}

// void
// host_cache_t::fetch_(int id) {

//     FileReader *single_reader = nullptr;
//     std::pair<FileReader *, FileReader *> *reader_pair = nullptr;

//     if (auto *reader = std::get_if<FileReader *>(&freader_[id])) {
//         single_reader = *reader;
//     } else if (auto *pair_reader =
//                    std::get_if<std::pair<FileReader *, FileReader *>>(
//                        &freader_[id])) {
//         reader_pair = pair_reader;
//     } else {
//         DBG("Error: No valid reader found!");
//         return;
//     }

//     while (is_active_) {
//         DBG("Host (" << id
//                      << ")- Waiting for items to be pushed onto the
//                      fetch_q");
//         TIMER_START(hst_waitfetch);
//         if (!fetch_q_[id].wait_any()) {
//             DBG("Error in fetch metadata queue of host cache, retrying...");
//             continue;
//         }
//         TIMER_STOP(hst_waitfetch,
//                    "Host (" << id << ")- Waited any batch for host fetch");

//         TIMER_START(hst_fetch);
//         std::deque<batch_t *> batches;
//         // Swap batches from the queue (without acquiring and releasing lock
//         // multiple times in the for loop)
//         if (!fetch_q_[id].swap_batches(batches)) {
//             DBG("Error: No batches to fetch.");
//             continue;
//         }
//         size_t curr_capacity = batches.size();
//         size_t nsegs_inbatch;
//         for (size_t i = 0; i < curr_capacity; i++) {
//             batch_t *item = batches[i];
//             if (single_reader) {
//                 DBG("Host (" << id << ")- Allocating memory to batch of size
//                 "
//                              << item->batch_len);
//                 item->allocate();
//                 DBG("Host (" << id << ")- Enqueuing for read from file");
//                 single_reader->enqueue_reads(item->to_vec());
//                 DBG("Host (" << id << ")- Waiting for batch read from file");
//                 nsegs_inbatch = item->batch_len;
//                 single_reader->wait_n(nsegs_inbatch);
//             } else if (reader_pair) {
//                 DBG("Host (" << id << ")- Waiting for batch read from
//                 files"); nsegs_inbatch = item->batch_len / 2;
//                 // reader_pair->first->wait_n(nsegs_inbatch);
//                 // reader_pair->second->wait_n(nsegs_inbatch);
//                 reader_pair->first->wait(item->data[0].offset);
//                 reader_pair->second->wait(item->data[0].offset);
//             }
//             DBG("Host (" << id << ")- Adding item to host ready queue");
//             stage_out(id, item);
//         }
//         TIMER_STOP(hst_fetch, "Host (" << id << ")- Fetched " <<
//         curr_capacity
//                                        << " batches to host cache");
//     }
//     DBG("Host (" << id << ")- Fetch thread exiting");
// }

// void
// host_cache_t::fetch_(int id, size_t used_chks_per_read) {

//     FileReader *single_reader = nullptr;
//     std::pair<FileReader *, FileReader *> *reader_pair = nullptr;

//     if (auto *reader = std::get_if<FileReader *>(&freader_[id])) {
//         single_reader = *reader;
//     } else if (auto *pair_reader =
//                    std::get_if<std::pair<FileReader *, FileReader *>>(
//                        &freader_[id])) {
//         reader_pair = pair_reader;
//     } else {
//         DBG("Error: No valid reader found!");
//         return;
//     }

//     while (is_active_) {
//         DBG("Host (" << id
//                      << ")- Waiting for items to be pushed onto the
//                      fetch_q");
//         TIMER_START(hst_waitfetch);
//         if (!fetch_q_[id].wait_any()) {
//             DBG("Error in fetch metadata queue of host cache, retrying...");
//             continue;
//         }
//         TIMER_STOP(hst_waitfetch,
//                    "Host (" << id << ")- Waited any batch for host fetch");

//         TIMER_START(hst_fetch);
//         std::deque<batch_t *> batches;
//         // Swap batches from the queue (without acquiring and releasing lock
//         // multiple times in the for loop)
//         if (!fetch_q_[id].swap_batches(batches)) {
//             DBG("Error: No batches to fetch.");
//             continue;
//         }
//         size_t curr_capacity = batches.size();
//         size_t nsegs_inbatch;
//         if (single_reader) {
//             for (size_t i = 0; i < curr_capacity; i++) {
//                 batch_t *item = batches[i];
//                 DBG("Host (" << id << ")- Allocating memory to batch of size
//                 "
//                              << item->batch_len);
//                 item->allocate();
//                 DBG("Host (" << id << ")- Enqueuing for read from file");
//                 single_reader->enqueue_reads(item->to_vec());
//                 DBG("Host (" << id << ")- Waiting for batch read from file");
//                 nsegs_inbatch = item->batch_len;
//                 single_reader->wait_n(nsegs_inbatch);
//                 DBG("Host (" << id << ")- Adding item to host ready queue");
//                 stage_out(id, item);
//             }
//         } else if (reader_pair) {
//             size_t wait_for_count = 0;
//             size_t start = 0;
//             for (size_t iter = 0; iter < curr_capacity; iter++) {
//                 wait_for_count += batches[iter]->proc_offt;
//                 if(wait_for_count >= used_chks_per_read) {
//                     DBG("Host (" << id << ")- Waiting for batch read from
//                     files"); size_t nbatch_forcount = iter+1;
//                     reader_pair->first->wait_n(nbatch_forcount);
//                     reader_pair->second->wait_n(nbatch_forcount);
//                     for(size_t j = start; j < iter; j++) {
//                         DBG("Host (" << id << ")- Adding item to host ready
//                         queue"); stage_out(id, batches[j]);
//                     }
//                     wait_for_count = 0;
//                     start = iter;
//                 }
//             }
//         }
//         TIMER_STOP(hst_fetch, "Host (" << id << ")- Fetched " <<
//         curr_capacity
//                                        << " batches to host cache");
//     }
//     DBG("Host (" << id << ")- Fetch thread exiting");
// }

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
    return ready_q_[id].front();
}

bool
host_cache_t::release(int id) {
    DBG("Host (" << id
                 << ")- Releasing memory used by previous processed batch");
    // batch_t *consumed_item = ready_q_[id].front();
    // free(consumed_item->ptr);
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
    // free(consumed_item->ptr);
}