#include "data_loader.hpp"

data_loader_t::data_loader_t(size_t host_cache_size, size_t device_cache_size)
    : data_ptr_(nullptr), host_cache_size_(host_cache_size),
      device_cache_size_(device_cache_size) {
    TIMER_START(init_loader);
    host_cache_ = new host_cache_t(gpu_id, host_cache_size_);
    EXEC_IF_NVCC(device_cache_ =
                     new device_cache_t(gpu_id, device_cache_size_););
    INFO("Loader - host and device caches initialized");
    TIMER_STOP(init_loader, "Initialized data loader");
}

data_loader_t::~data_loader_t() {
    host_cache_ = nullptr;
    EXEC_IF_NVCC(device_cache_ = nullptr;);
    DBG("Loader - destroyed");
};

void
data_loader_t::coalesce(int id, std::vector<size_t> offsets, size_t seg_size,
                        uint32_t gap) {
    INFO("Loader (" << id << ")- Coalescing offsets with gap = " << gap
                    << " for large reads");

    if (offsets.empty())
        return;
    size_t start = offsets[0];
    size_t lastOffset = start;
    for (size_t i = 1; i < offsets.size(); ++i) {
        if (offsets[i] - lastOffset - 1 <= gap) {
            lastOffset = offsets[i];
        } else {
            batch_t *seg_batch = new batch_t(1);
            size_t combined_size = (lastOffset - start + 1) * seg_size;
            segment_t seg(start, combined_size);
            seg_batch->push(seg);
            host_cache_->stage_in(id, seg_batch);
            start = offsets[i];
            lastOffset = offsets[i];
        }
    }

    // Account for the last group
    // what if the size of the last chunk was not seg_size?
    batch_t *seg_batch = new batch_t(1);
    size_t combined_size = (lastOffset - start + 1) * seg_size;
    segment_t seg(start, combined_size);
    seg_batch->push(seg);
    host_cache_->stage_in(id, seg_batch);
}

void
data_loader_t::enqueue_reads(int id, std::vector<size_t> offsets,
                             size_t seg_size, size_t n_segs_per_read,
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
        size_t offset;
        if (offsets.empty()) {
            offset = i * seg_size;
            if (i == total_n_segs - 1)
                seg_size = total_read_size - i * seg_size;
        } else {
            offset = offsets[i];
        }
        segment_t seg(offset, seg_size);
        seg_batch->push(seg);
    }
    host_cache_->stage_in(id, seg_batch);   // stage last batch
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
 * @param offsets Optional list of file offsets to read data from. If empty, all
 * data are read.
 * @param merge_seg Temporary boolean parameter used to define if non-contiguous
 * offsets should be merged.
 * @return ID of the file loading request. It is used to coordinate multiple
 * concurrent file loading requests.
 */
int
data_loader_t::file_load(FileReader &io_reader, size_t start_foffset,
                         size_t seg_size, TransferType trans_type,
                         std::optional<std::vector<size_t>> offsets,
                         bool merge_seg, uint32_t gap) {
    TIMER_START(file_load);
    assert(trans_type == TransferType::FileToHost ||
           trans_type == TransferType::FileToDevice &&
               "Invalid TransferType: Must be FileToHost or FileToDevice");

    int loader_id = instance_count++;
    ready_count[loader_id] = 0;
    host_cache_->set_reader(loader_id, &io_reader);

    EXEC_IF_NVCC(if (trans_type == TransferType::FileToDevice) {
        host_cache_->set_next_tier(loader_id, device_cache_);
    });

    size_t n_segs_per_read;
    size_t total_n_segs;
    // create segments
    if (offsets.has_value()) {
        INFO("Loader (" << loader_id
                        << ")- Creating segments given file offsets");
        if (merge_seg) {
            coalesce(loader_id, *offsets, seg_size, gap);
        } else {
            total_n_segs = offsets->size();
            n_segs_per_read = 1;
            enqueue_reads(loader_id, *offsets, seg_size, n_segs_per_read,
                          total_n_segs, total_n_segs * seg_size);
        }
    } else {
        INFO("Loader (" << loader_id
                        << ")- Creating segments without given file offsets");
        size_t total_read_size = io_reader.size() - start_foffset;
        n_segs_per_read = 1;
        total_n_segs = total_read_size / seg_size;
        total_n_segs = (total_n_segs * seg_size < total_read_size)
                           ? (total_n_segs + 1)
                           : total_n_segs;
        enqueue_reads(loader_id, *offsets, seg_size, n_segs_per_read,
                      total_n_segs, total_read_size);
    }
    INFO("Loader (" << loader_id
                    << ")- All batches staged in for read from file");
    TIMER_STOP(file_load, "Created segments and staged for file read");
    return loader_id;
}

int
data_loader_t::file_load(FileReader &io_reader0, FileReader &io_reader1,
                         size_t start_foffset, size_t seg_size,
                         TransferType trans_type,
                         std::optional<std::vector<size_t>> offsets,
                         uint32_t gap) {
    TIMER_START(file_load);
    assert(trans_type == TransferType::FileToHost ||
           trans_type == TransferType::FileToDevice &&
               "Invalid TransferType: Must be FileToHost or FileToDevice");

    int loader_id = instance_count++;
    ready_count[loader_id] = 0;
    host_cache_->set_reader(loader_id, &io_reader0, &io_reader1);
    INFO("Loader ("
         << loader_id
         << ")- Creating segments for two readers given file offsets");
    coalesce(loader_id, *offsets, seg_size, gap);
    INFO("Loader (" << loader_id
                    << ")- All batches staged in for read with two readers");
    TIMER_STOP(file_load, "Created segments and staged for file read");
    return loader_id;
}

size_t
data_loader_t::next(int id, void *ptr) {
    // NB: Ensure that each segment in batch is of size seg_size
    TIMER_START(next);
    batch_t *front_batch;
    bool request_processed = false;

    EXEC_IF_NVCC(
        cudaPointerAttributes attributes;
        cudaError_t err = cudaPointerGetAttributes(&attributes, ptr);
        if (err == cudaSuccess && attributes.type == cudaMemoryTypeDevice) {
            front_batch = device_cache_->get_completed(id);
            device_cache_->coalesce_and_copy(front_batch, ptr);
            device_cache_->release(id);
            request_processed = true;
        });

    if (!request_processed) {
        front_batch = host_cache_->get_completed(id);
        host_cache_->coalesce_and_copy(front_batch, ptr);
        host_cache_->release(id);
    }
    TIMER_STOP(next, "Retrieved pointer to next batch of data for computation");
    size_t ready_size = front_batch->data->size * front_batch->batch_size;
    return ready_size;
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

std::pair<uint8_t *, size_t>
data_loader_t::next(int id, TransferType trans_type) {
    size_t call_count = ready_count[id]++;
    TIMER_START(next);
    assert(trans_type == TransferType::HostToDevice ||
           trans_type == TransferType::HostPinned ||
           trans_type == TransferType::FileToHost ||
           trans_type == TransferType::FileToDevice && "Invalid TransferType!");

    base_cache_t *cache_tier = host_cache_;
    batch_t *front_batch;
    if (trans_type == TransferType::HostToDevice ||
        trans_type == TransferType::FileToDevice) {
        EXEC_IF_NVCC(cache_tier = device_cache_;);
    }
    // Because of the FIFO queue implementation, we need to make sure the first
    // retrieved batch of the next ID is not the last batch the previous
    // retrieving ID had read.
    if (id != last_retrieving_id) {
        INFO("Releasing last batch of ID = " << last_retrieving_id << " from storage")
        cache_tier->release(last_retrieving_id);
        last_retrieving_id = id;
    }
    // Retrieve the next batch for the current ID
    front_batch = get_next(id, cache_tier, call_count > 0);
    TIMER_STOP(next, "Retrieved pointer to next batch of data for computation");
    size_t front_size = front_batch->data->size * front_batch->batch_size;
    DBG("Retrieved pointer to offset " << front_batch->data[0].offset << " for "
                                       << front_size << " bytes");
    ASSERT(front_batch->data[0].buffer != nullptr);
    return {front_batch->data[0].buffer, front_size};
}