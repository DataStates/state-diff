#ifndef __DATA_LOADER_HPP
#define __DATA_LOADER_HPP
#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_timer.hpp"
#include "io_reader.hpp"

template <typename FileReader> class data_loader {
  private:
    FileReader &io_reader_;
    std::vector<segment_t> segments_;
    size_t num_host_seg_;
    size_t num_dev_seg_;
    size_t num_batches_;
    size_t transfered_batch_;
    size_t curr_host_seg_;
    size_t transfered_seg_;
    bool stream_finalized = false;
    static const int num_dev_buf_ = 2;
    uint8_t *device_buffers_[num_dev_buf_];

    // Variables to hold timing events
    CudaTimer load_timer;
    CudaTimer wait_timer;

#ifdef __NVCC__
    cudaEvent_t transfer_events_[num_dev_buf_];
    cudaStream_t active_stream_[num_dev_buf_];

#endif

  public:
    data_loader(FileReader &reader, std::vector<segment_t> &segments,
                size_t host_buf_size, size_t dev_buf_size, size_t num_batch);
    ~data_loader();

    segment_t *load();
    uint8_t *to_device();
    void finalize();
#ifdef __NVCC__
    cudaStream_t getStream() const {
        return active_stream_[transfered_batch_ % num_dev_buf_];
    }
    std::vector<float> getTimings() const {
        return {load_timer.getTotalTime(), wait_timer.getTotalTime()};
    }
#endif
};

template <typename FileReader>
data_loader<FileReader>::data_loader(FileReader &reader,
                                     std::vector<segment_t> &segments,
                                     size_t host_buf_size, size_t dev_buf_size,
                                     size_t num_batch)
    : io_reader_(reader), segments_(segments), num_batches_(num_batch),
      load_timer("loading_data"), wait_timer("waiting_for_data") {

    // Assuming the segment sizes are same
    size_t chunk_size = segments_[0].size;
    num_host_seg_ = host_buf_size / chunk_size;
    if (num_host_seg_ * chunk_size < host_buf_size)
        num_host_seg_ += 1;
    num_dev_seg_ = dev_buf_size / chunk_size;
    if (num_dev_seg_ * chunk_size < dev_buf_size)
        num_dev_seg_ += 1;
    curr_host_seg_ = 0;
    transfered_seg_ = 0;
    transfered_batch_ = 0;
#ifdef __NVCC__
    // Allocate two device buffers for double buffering
    for (int i = 0; i < num_dev_buf_; i++) {
        gpuErrchk(cudaMalloc(&device_buffers_[i], num_dev_seg_ * chunk_size));
        gpuErrchk(cudaEventCreate(&transfer_events_[i]));
        gpuErrchk(cudaStreamCreate(&active_stream_[i]));
    }
#else
    for (int i = 0; i < num_dev_buf_; i++) {
        device_buffers_[i] = (uint8_t *) malloc(num_dev_seg_ * chunk_size);
    }
#endif
}

template <typename FileReader> data_loader<FileReader>::~data_loader() {
#ifdef __NVCC__
    if (!stream_finalized) {
        finalize();
    }
    for (int i = 0; i < num_dev_buf_; i++) {
        gpuErrchk(cudaEventDestroy(transfer_events_[i]));
        gpuErrchk(cudaStreamDestroy(active_stream_[i]));
        gpuErrchk(cudaFree(device_buffers_[i]));
    }
#else
    for (int i = 0; i < num_dev_buf_; i++) {
        free(device_buffers_[i]);
    }
#endif
}

template <typename FileReader>
segment_t *
data_loader<FileReader>::load() {
    // return nullptr to mark that all segments were read
    if (curr_host_seg_ >= segments_.size()) {
        return nullptr;
    }
    size_t end_seg = std::min(curr_host_seg_ + num_host_seg_, segments_.size());
    std::vector<segment_t> seg_block(segments_.begin() + curr_host_seg_,
                                     segments_.begin() + end_seg);
    // Read segments until they fill the available host buffer space
    io_reader_.enqueue_reads(seg_block);
    io_reader_.wait_all();

    segment_t *cur_block_start = &segments_[curr_host_seg_];
    // Move the current position to prepare for the next call to load
    curr_host_seg_ = end_seg;
    // Return the pointer to the beginning of loaded segments
    return cur_block_start;
}

template <typename FileReader>
uint8_t *
data_loader<FileReader>::to_device() {
    assert(transfered_batch_ < num_batches_);
    assert(transfered_seg_ < num_batches_ * num_dev_seg_);
    int buffer_idx = transfered_batch_ % num_dev_buf_;
    segment_t *host_segment = segments_.data();
    size_t start = transfered_seg_;
    uint8_t *out_buffer = NULL;
    size_t end = std::min(start + num_dev_seg_, segments_.size());
#ifdef __NVCC__
    load_timer.start(active_stream_[buffer_idx]);
    size_t offset = 0;
    for (size_t i = start; i < end; ++i) {
        gpuErrchk(cudaMemcpyAsync(device_buffers_[buffer_idx] + offset,
                                  host_segment[i].buffer, host_segment[i].size,
                                  cudaMemcpyHostToDevice,
                                  active_stream_[buffer_idx]));
        offset += host_segment[i].size;
    }
    gpuErrchk(cudaEventRecord(transfer_events_[buffer_idx],
                              active_stream_[buffer_idx]));
    load_timer.stop(active_stream_[buffer_idx]);
    wait_timer.start(active_stream_[buffer_idx]);
    gpuErrchk(cudaStreamWaitEvent(active_stream_[buffer_idx],
                                  transfer_events_[buffer_idx], 0));
    wait_timer.stop(active_stream_[buffer_idx]);
    while (cudaEventQuery(transfer_events_[buffer_idx]) == cudaErrorNotReady) {
        std::this_thread::sleep_for(std::chrono::microseconds(10));
    }
    transfered_seg_ += (end - start);
    transfered_batch_ += 1;
    out_buffer = device_buffers_[buffer_idx];
#endif
    return out_buffer;
}

template <typename FileReader>
void
data_loader<FileReader>::finalize() {
    for (int i = 0; i < num_dev_buf_; i++) {
        gpuErrchk(cudaStreamSynchronize(active_stream_[i]));
    }
    wait_timer.finalize();
    load_timer.finalize();
    stream_finalized = true;
}

#endif   // __DATA_LOADER_HPP
