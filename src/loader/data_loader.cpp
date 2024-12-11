#include "data_loader.hpp"

data_loader_t::data_loader_t(size_t host_cache_size, size_t device_cache_size)
    : host_cache_size_(host_cache_size), device_cache_size_(device_cache_size),
      data_ptr_(nullptr) {
    TIMER_START(init_loader);
    host_cache_ = new host_cache_t(gpu_id, host_cache_size_);
    device_cache_ = new device_cache_t(gpu_id, device_cache_size_);
    INFO("Loader - host and device caches initialized");
    TIMER_STOP(init_loader, "Initialized data loader");
}

data_loader_t::~data_loader_t() {
    host_cache_ = nullptr;
    device_cache_ = nullptr;
    DBG("Loader - destroyed");
};

size_t
data_loader_t::max_batch_size(size_t seg_size) {
    size_t max_payload = std::min(host_cache_size_, device_cache_size_);
    size_t n_segs = max_payload / seg_size;
    n_segs = (n_segs * seg_size < max_payload) ? (n_segs + 1) : n_segs;
    return n_segs;
}

std::vector<std::pair<size_t, int>>
coalesce(std::vector<size_t> offsets) {
    // std::sort(offsets.begin(), offsets.end());
    std::vector<std::pair<size_t, int>> new_offsets;
    for (size_t offset : offsets) {
        if (new_offsets.empty()) {
            new_offsets.emplace_back(offset, 1);
        } else {
            auto &lastPair = new_offsets.back();
            size_t lastKey = lastPair.first;
            int lastValue = lastPair.second;
            if (lastKey + lastValue == offset) {
                ++lastPair.second;
            } else {
                new_offsets.emplace_back(offset, 1);
            }
        }
    }
    return new_offsets;
}

void
data_loader_t::merge_create_seg(int id, std::vector<size_t> &offsets,
                                size_t total_segs, size_t batch_size_,
                                size_t seg_size) {
    INFO("Loader - Merging contiguous segments for file reading");
    // std::sort(offsets.begin(), offsets.end());
    std::vector<std::pair<size_t, int>> new_offsets;
    size_t i = 0;
    batch_t *seg_batch = new batch_t(batch_size_);

    while (i < total_segs) {
        if (i == batch_size_) {
            host_cache_->stage_in(id, seg_batch);
            seg_batch = new batch_t(batch_size_);
        }
        if (i == 0) {
            new_offsets.emplace_back(offsets[i], 1);
            segment_t seg(offsets[i], seg_size);
            seg_batch->push(seg);
        } else {
            auto &lastPair = new_offsets.back();
            size_t lastKey = lastPair.first;
            int lastValue = lastPair.second;
            if (lastKey + lastValue == offsets[i]) {
                ++lastPair.second;
                seg_batch->inc_last(seg_size);
            } else {
                new_offsets.emplace_back(offsets[i], 1);
                segment_t seg(offsets[i], seg_size);
                seg_batch->push(seg);
            }
        }
    }
}

int
data_loader_t::file_load(FileReader &io_reader, size_t start_foffset,
                         size_t seg_size, size_t batch_size,
                         TransferType trans_type,
                         std::optional<std::vector<size_t>> offsets,
                         bool merge_seg) {
    TIMER_START(file_load);
    assert(trans_type == TransferType::FileToHost ||
           trans_type == TransferType::FileToDevice &&
               "Invalid TransferType: Must be FileToHost or FileToDevice");

    int loader_id = instance_count++;
    ready_count[loader_id] = 0;
    host_cache_->set_reader(loader_id, &io_reader);
    if (trans_type == TransferType::FileToDevice) {
        host_cache_->set_next_tier(loader_id, device_cache_);
    }

    size_t batch_size_ =
        (batch_size < 1) ? max_batch_size(seg_size) : batch_size;

    // create segments
    if (offsets.has_value()) {
        INFO("Loader (" << loader_id
                        << ")- Creating segments given file offsets");
        size_t total_segs = offsets->size();
        if (merge_seg) {
            merge_create_seg(loader_id, *offsets, total_segs, batch_size_,
                             seg_size);
        } else {
            size_t n_iter = total_segs / batch_size_;
            n_iter =
                (n_iter * batch_size_ < total_segs) ? (n_iter + 1) : n_iter;
            for (size_t i = 0; i < n_iter; i++) {
                batch_t *seg_batch = new batch_t(batch_size_);
                DBG("Loader (" << loader_id << ")- Staging batch " << i + 1
                               << " of size " << batch_size_
                               << " for read from file");
                for (size_t j = 0; j < batch_size_; j++) {
                    size_t index = i * batch_size_ + j;
                    segment_t seg((*offsets)[index], seg_size);
                    seg_batch->push(seg);
                }
                host_cache_->stage_in(loader_id, seg_batch);
            }
        }
    } else {
        INFO("Loader (" << loader_id
                        << ")- Creating segments without given file offsets");
        size_t data_size = io_reader.size() - start_foffset;
        // batch_size_ = 1;
        // seg_size *= max_batch_size(seg_size);
        size_t total_segs = data_size / seg_size;
        total_segs =
            (total_segs * seg_size < data_size) ? (total_segs + 1) : total_segs;
        size_t n_iter = total_segs / batch_size_;
        n_iter = (n_iter * batch_size_ < total_segs) ? (n_iter + 1) : n_iter;

        for (size_t i = 0; i < n_iter; i++) {
            batch_t *seg_batch = new batch_t(batch_size_);
            DBG("Loader (" << loader_id << ")- Staging batch " << i + 1
                           << " of size " << batch_size_
                           << " for read from file");
            for (size_t j = 0; j < batch_size_; j++) {
                size_t index = i * batch_size_ + j;
                segment_t seg(index * seg_size, seg_size);
                seg_batch->push(seg);
            }
            host_cache_->stage_in(loader_id, seg_batch);
        }
    }
    TIMER_STOP(file_load, "Created segments and staged for file read");
    return loader_id;
}

size_t
data_loader_t::next(int id, void *ptr) {
    // NB: Ensure that each segment in batch is of size seg_size
    TIMER_START(next);
    cudaPointerAttributes attributes;
    cudaError_t err = cudaPointerGetAttributes(&attributes, ptr);
    batch_t *front_batch;
    if (err == cudaSuccess && attributes.type == cudaMemoryTypeDevice) {
        front_batch = device_cache_->get_completed(id);
        device_cache_->coalesce_and_copy(front_batch, ptr);
        device_cache_->release(id);
    } else {
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
// pointer to the ready batch is returned to the user, we can load new data
// inplace because the loader does not know when the kernel completed
// computation. To address that issue, we implement an approach that keeps track
// of the number of batches returned to deallocate the previous batch before
// returning a pointer to the next batch.

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

    batch_t *front_batch;
    if (trans_type == TransferType::HostToDevice ||
        trans_type == TransferType::FileToDevice) {
        front_batch = get_next(id, device_cache_, call_count > 0);
    } else {
        front_batch = get_next(id, host_cache_, call_count > 0);
    }
    TIMER_STOP(next, "Retrieved pointer to next batch of data for computation");
    size_t ready_size = front_batch->data->size * front_batch->batch_size;
    return {front_batch->data[0].buffer, ready_size};
}

size_t
data_loader_t::get_chunksize(size_t data_size) {
    double peak_bw = 25;
    float rate_of_change = 0.5;
    size_t opt_chksize = data_size / (exp(data_size / peak_bw*rate_of_change) + 1);
    printf("Optimum chun size is %zu\n", opt_chksize);
    return 1024;
}

void
data_loader_t::mem_load(int loader_id, std::vector<uint8_t> &data,
                        size_t start_foffset, size_t seg_size,
                        size_t batch_size, TransferType trans_type,
                        std::optional<std::vector<size_t>> offsets) {

    assert(trans_type == TransferType::HostToDevice ||
           trans_type == TransferType::HostPinned &&
               "Invalid TransferType: Must be HostToDevice or HostPinned");
    data_ptr_ = data.data();
}