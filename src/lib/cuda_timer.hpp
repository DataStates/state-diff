#include "cuda.h"
#include "cuda_runtime.h"
#include "nvtx3/nvToolsExt.h"

class CudaTimer {
  public:
    CudaTimer(std::string event, cudaStream_t stream = 0)
        : event_(event), stream_(stream), total_time_(0.0) {
        cudaEventCreate(&start_);
        cudaEventCreate(&stop_);
    }

    ~CudaTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

    void start() { 
        nvtxRangePush(event_.c_str());
        cudaEventRecord(start_, stream_); 
    }

    void stop() {
        cudaEventRecord(stop_, stream_);
        cudaEventSynchronize(stop_);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start_, stop_);
        total_time_ += milliseconds;
        nvtxRangePop();
    }

    float getTotalTime() const { return total_time_; }

    float getThroughput(float data_size_gb) const {
        return data_size_gb / (total_time_ / 1000);   // Throughput in GB/s
    }

  private:
    cudaEvent_t start_, stop_;
    cudaStream_t stream_;
    float total_time_;
    std::string event_;
};
