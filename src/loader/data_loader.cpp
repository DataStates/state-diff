#include "data_loader.hpp"

data_loader_t::data_loader_t(size_t host_cache_size, size_t device_cache_size)
    : host_cache_size_(host_cache_size), device_cache_size_(device_cache_size),
      data_ptr_(nullptr) {

    host_cache_ = new host_cache_t(gpu_id, host_cache_size_);
    device_cache_ = new device_cache_t(gpu_id, device_cache_size_);
    INFO("Loader - host and device caches initialized");
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

void
data_loader_t::file_load(FileReader &io_reader, size_t start_foffset,
                         size_t seg_size, size_t batch_size,
                         TransferType trans_type,
                         std::optional<std::vector<size_t>> offsets) {

    assert(trans_type == TransferType::FileToHost ||
           trans_type == TransferType::FileToDevice &&
               "Invalid TransferType: Must be FileToHost or FileToDevice");

    host_cache_->set_reader(&io_reader);
    if (trans_type == TransferType::FileToDevice) {
        host_cache_->set_next_tier(device_cache_);
    }
    
    size_t batch_size_ =
        (batch_size < 1) ? max_batch_size(seg_size) : batch_size;

    // create segments
    if (offsets.has_value()) {
        INFO("Loader - Creating segments given offsets");
        size_t total_segs = offsets->size();
        size_t n_iter = total_segs / batch_size_;
        n_iter = (n_iter * batch_size_ < total_segs) ? (n_iter + 1) : n_iter;

        for (size_t i = 0; i < n_iter; i++) {
            batch_t seg_batch(batch_size_);
            DBG("Loader - Staging batch " << i+1 << " of size " << batch_size_ << " for read from file");
            for (size_t j = 0; j < batch_size_; j++) {
                size_t index = i * batch_size_ + j;
                segment_t seg((*offsets)[index], seg_size);
                seg_batch.push(seg);
            }
            host_cache_->stage_in(&seg_batch);
        }
    } else {
        INFO("Loader - Creating segments without given offsets");
        size_t data_size = io_reader.size() - start_foffset;
        size_t total_segs = data_size / seg_size;
        total_segs =
            (total_segs * seg_size < data_size) ? (total_segs + 1) : total_segs;
        size_t n_iter = total_segs / batch_size_;
        n_iter = (n_iter * batch_size_ < total_segs) ? (n_iter + 1) : n_iter;

        for (size_t i = 0; i < n_iter; i++) {
            batch_t *seg_batch = new batch_t(batch_size_);
            DBG("Loader - Staging batch" << i+1 << " of size " << batch_size_ << " for read from file");
            for (size_t j = 0; j < batch_size_; j++) {
                size_t index = i * batch_size_ + j;
                segment_t seg(index * seg_size, seg_size);
                seg_batch->push(seg);
            }
            host_cache_->stage_in(seg_batch);
        }
    }
}

void
data_loader_t::next(void *ptr) {
    cudaPointerAttributes attributes;
    cudaError_t err = cudaPointerGetAttributes(&attributes, ptr);
    if (err == cudaSuccess && attributes.type == cudaMemoryTypeDevice) {
        batch_t *front_batch = device_cache_->get_completed();
        device_cache_->coalesce_and_copy(front_batch, ptr);
    } else {
        batch_t *front_batch = host_cache_->get_completed();
        host_cache_->coalesce_and_copy(front_batch, ptr);
    }
}

void
data_loader_t::mem_load(std::vector<uint8_t> &data, size_t start_foffset,
                        size_t seg_size, size_t batch_size,
                        TransferType trans_type,
                        std::optional<std::vector<size_t>> offsets) {

    assert(trans_type == TransferType::HostToDevice ||
           trans_type == TransferType::HostPinned &&
               "Invalid TransferType: Must be HostToDevice or HostPinned");
    data_ptr_ = data.data();
}

void
data_loader_t::wait() {}