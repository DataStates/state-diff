#ifndef __CUDA_TIMER_HPP
#define __CUDA_TIMER_HPP

#ifdef __NVCC__

#include "cuda.h"
#include "cuda_runtime.h"
#include "nvtx3/nvToolsExt.h"
#include <iostream>
#include <vector>

class CudaTimer {
  public:
    CudaTimer(std::string event)
        : event_name_(event), event_counter_(0), total_time_(0.0) {}

    ~CudaTimer() {}

    void start(cudaStream_t &stream) {
        cudaEvent_t event;
        cudaEventCreate(&event);
        start_.push_back(event);
        nvtxRangePush(event_name_.c_str());
        cudaEventRecord(event, stream);
    }

    void stop(cudaStream_t &stream) {
        cudaEvent_t event;
        cudaEventCreate(&event);
        stop_.push_back(event);
        cudaEventRecord(event, stream);
        nvtxRangePop();
        event_counter_ += 1;
    }

    void finalize() {
        for (size_t i = 0; i < event_counter_; i++) {
            // Synchronize to make sure all recorded events are complete
            cudaEventSynchronize(stop_[i]);
            float milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start_[i], stop_[i]);
            total_time_ += milliseconds;
            cudaEventDestroy(start_[i]);
            cudaEventDestroy(stop_[i]);
        }
    }

    float getTotalTime() const { return total_time_; }

    float getThroughput(float data_size_gb) const {
        return data_size_gb / (total_time_ / 1000.0f);   // Throughput in GB/s
    }

  private:
    std::vector<cudaEvent_t> start_, stop_;
    size_t event_counter_;
    float total_time_;
    std::string event_name_;
};

#endif   // __NVCC__
#endif   // __CUDA_TIMER_HPP
