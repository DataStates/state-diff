#include "data_loader.hpp"

data_loader_t::data_loader_t(size_t host_cache_size, size_t device_cache_size)
    : data_ptr_(nullptr), host_cache_size_(host_cache_size),
      device_cache_size_(device_cache_size) {
    TIMER_START(init_loader);
    host_cache_ = new host_cache_t(gpu_id, host_cache_size_);
    INFO("Loader - host caches initialized");
    TIMER_STOP(init_loader, "Initialized data loader");
}

data_loader_t::~data_loader_t() {
    host_cache_ = nullptr;
    DBG("Loader - destroyed");
};

void
data_loader_t::coalesce(int id, std::vector<size_t> offsets, size_t seg_size,
                        uint32_t gap, int n_readers) {
    INFO("Loader (" << id << ")- Coalescing offsets with gap = " << gap
                    << " for large reads");
    if (offsets.empty())
        return;
    size_t start = offsets[0];
    size_t lastOffset = start;
    size_t in_group_offt = 1;
    for (size_t i = 1; i < offsets.size(); ++i) {
        if (offsets[i] - lastOffset <= gap) {   // include new offset
            lastOffset = offsets[i];
            in_group_offt += 1;
        } else {   // create batch and start a new group
            batch_t *seg_batch = new batch_t(n_readers, in_group_offt);
            size_t combined_size = (lastOffset - start + 1) * seg_size;
            for (int j = 0; j < n_readers; j++) {
                segment_t seg(start, combined_size);
                seg_batch->push(seg);
            }
            host_cache_->stage_in(id, seg_batch);
            start = offsets[i];
            lastOffset = offsets[i];
            in_group_offt = 1;
        }
    }
    // Account for the last group
    // what if the size of the last chunk was not seg_size?
    batch_t *seg_batch = new batch_t(n_readers, in_group_offt);
    size_t combined_size = (lastOffset - start + 1) * seg_size;
    for (int i = 0; i < n_readers; i++) {
        segment_t seg(start, combined_size);
        seg_batch->push(seg);
    }
    host_cache_->stage_in(id, seg_batch);
}

void
data_loader_t::enqueue_reads(int id, size_t seg_size, size_t n_segs_per_read,
                             size_t total_n_segs, size_t total_read_size) {
    size_t n_read_call = total_n_segs / n_segs_per_read;
    n_read_call = (n_read_call * n_segs_per_read < total_n_segs)
                      ? (n_read_call + 1)
                      : n_read_call;
    batch_t *seg_batch = new batch_t(n_segs_per_read);
    for (size_t i = 0; i < total_n_segs; i++) {
        if (i > 0 && i % n_segs_per_read == 0) {
            DBG("Loader (" << id << ")- Staging batch "
                           << (i / n_segs_per_read) - 1 << " of size "
                           << n_segs_per_read << " for read from file");
            host_cache_->stage_in(id, seg_batch);
            n_read_call -= 1;
            if (n_read_call == 1)
                n_segs_per_read = total_n_segs - i;
            seg_batch = new batch_t(n_segs_per_read);
        }
        size_t offset = i * seg_size;
        if (i == total_n_segs - 1)
            seg_size = total_read_size - i * seg_size;
        segment_t seg(offset, seg_size);
        seg_batch->push(seg);
    }
    // stage last batch
    DBG("Loader (" << id << ")- Staging batch "
                           << (total_n_segs / n_segs_per_read) - 1 << " of size "
                           << n_segs_per_read << " for read from file");
    host_cache_->stage_in(id, seg_batch);
}

/**
 * @brief Load data from a file
 *
 * This function takes a liburing reader and a destination of the data
 * as input (trans_type) and loads data from a file to the destination.
 *
 * @param io_reader The reference of the liburing reader for file IO.
 * @param start_foffset Offset of the file from which the reader starts reading
 * from.
 * @param seg_size  Size of each segment enqueued for read operations
 * @param batch_size Number of segments to submit at a time to the liburing
 * reader
 * @param trans_type Definition of the source and destination of the data
 * @param offsets List of file offsets to read data from. If empty, all
 * data are read.
 * @param merge_seg Temporary boolean parameter used to define if non-contiguous
 * offsets should be merged.
 * @return ID of the file loading request. It is used to coordinate multiple
 * concurrent file loading requests.
 */
int
data_loader_t::file_load(FileReader &io_reader, size_t seg_size,
                         TransferType trans_type) {
    TIMER_START(file_load);
    assert((trans_type == TransferType::FileToHost ||
            trans_type == TransferType::FileToDevice) &&
           "Invalid TransferType: Must be FileToHost or FileToDevice");

    // Assign an ID to this loader call
    int loader_id = instance_count++;
    ready_count[loader_id] = 0;

    INFO("Loader (" << loader_id
                    << ")- Creating segments without given file offsets");

    // Create segments to read
    size_t total_read_size = io_reader.size();
    assert(seg_size > 0 && total_read_size > 0);
    size_t total_n_segs = total_read_size / seg_size;
    total_n_segs = (total_n_segs * seg_size < total_read_size)
                       ? (total_n_segs + 1)
                       : total_n_segs;
    size_t base_segcnt = 2;
    size_t n_segs_per_read = std::min({base_segcnt, total_n_segs});
    enqueue_reads(loader_id, seg_size, n_segs_per_read, total_n_segs,
                  total_read_size);
    INFO("Loader (" << loader_id
                    << ")- All batches staged in for read from file");
    TIMER_STOP(file_load, "Created segments and staged for file read");

    // // Set the reader to use for file IO.
    // // Making sure metadata are enqueue before setting the reader to avoid
    // // having the thread wait and constantly pool for enqueue metadata.
    host_cache_->set_reader(loader_id, &io_reader);
    return loader_id;
}

int
data_loader_t::file_load(FileReader &io_reader0, FileReader &io_reader1,
                         std::vector<size_t> offsets, size_t seg_size,
                         TransferType trans_type, uint32_t gap) {
    TIMER_START(file_load);
    assert((trans_type == TransferType::FileToHost ||
            trans_type == TransferType::FileToDevice) &&
           "Invalid TransferType: Must be FileToHost or FileToDevice");
    // Assign an ID to this loader call
    int loader_id = instance_count++;
    ready_count[loader_id] = 0;
    int n_readers = 2;

    INFO("Loader ("
         << loader_id
         << ")- Creating segments for two readers given file offsets");
    coalesce(loader_id, offsets, seg_size, gap, n_readers);
    INFO("Loader (" << loader_id
                    << ")- All batches staged in for read with two readers");
    TIMER_STOP(file_load, "Created segments and staged for file read");

    // Set two reader to use for file IO
    host_cache_->set_reader(loader_id, &io_reader0, &io_reader1);
    return loader_id;
}

// This implementation of next works because the memory for the segments in a
// batch point are allocated contiguously from the data_store. However, once the
// pointer to the ready batch is returned to the user, new data may be loaded
// inplace because the loader does not know when the kernel completed
// computation, affecting accuracy. To address that issue, we implement an
// approach that keeps track of the number of batches returned to deallocate the
// previous batch before returning a pointer to the next batch. The reason why
// we are returning a pointer to the user instead of receiving a pointer from
// the user (and copying data to the user pointer) is because computation is
// faster than data movement, i.e., the time it would take to copy the data is
// higher than the time to process it.

batch_t *
get_next(int id, base_cache_t *cache_tier, bool should_release) {
    if (should_release)
        cache_tier->release(id);
    return cache_tier->get_completed(id);
}

next_batch_t
data_loader_t::next(int id, TransferType trans_type) {
    size_t call_count = ready_count[id]++;
    TIMER_START(next);
    assert((trans_type == TransferType::HostToDevice ||
            trans_type == TransferType::HostPinned ||
            trans_type == TransferType::FileToHost ||
            trans_type == TransferType::FileToDevice) &&
           "Invalid TransferType!");

    base_cache_t *cache_tier = host_cache_;
    // Because of the FIFO queue implementation, we need to make sure the first
    // retrieved batch of the next ID is not the last batch the previous
    // retrieving ID had read.
    if (id != last_retrieving_id) {
        INFO("Releasing last batch of ID = " << last_retrieving_id
                                             << " from storage")
        cache_tier->release(last_retrieving_id);
        last_retrieving_id = id;
    }
    // Retrieve the next batch for the current ID
    batch_t *front_batch = get_next(id, cache_tier, call_count > 0);
    TIMER_STOP(next, "Retrieved pointer to next batch of data for computation");
    size_t front_size = front_batch->size;
    assert(front_batch->data[0].buffer != nullptr && front_size > 0);
    // assert(front_size == front_batch->data[0].size*2);
    DBG("Retrieved pointer to offset " << front_batch->data[0].offset << " for "
                                       << front_size << " bytes");
    next_batch_t batch = {front_batch->data[0].buffer, front_size,
                          front_batch->proc_offt};
    return batch;
}