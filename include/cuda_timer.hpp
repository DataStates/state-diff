#include "cuda.h"
#include "cuda_runtime.h"
#include "nvtx3/nvToolsExt.h"

class CudaTimer {
public:
    CudaTimer(std::string event)
        : event_(event), total_time_(0.0) {
        cudaEventCreate(&start_);
        cudaEventCreate(&stop_);
    }

    ~CudaTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

    void start(cudaStream_t &stream) {
        nvtxRangePush(event_.c_str());
        cudaEventRecord(start_, stream);
    }

    void stop(cudaStream_t &stream) {
        cudaEventRecord(stop_, stream);
        nvtxRangePop();
    }

    void finalize() {
        // Synchronize to make sure all recorded events are complete before measuring
        cudaEventSynchronize(stop_);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start_, stop_);
        total_time_ += milliseconds;
    }

    float getTotalTime() const { return total_time_; }

    float getThroughput(float data_size_gb) const {
        return data_size_gb / (total_time_ / 1000);   // Throughput in GB/s
    }

private:
    cudaEvent_t start_, stop_;
    float total_time_;
    std::string event_;
};


// class CudaTimer {
//   public:
//     CudaTimer(std::string event)
//         : event_(event), total_time_(0.0) {
//         cudaEventCreate(&start_);
//         cudaEventCreate(&stop_);
//     }

//     ~CudaTimer() {
//         cudaEventDestroy(start_);
//         cudaEventDestroy(stop_);
//     }

//     void start(cudaStream_t &stream) { 
//         nvtxRangePush(event_.c_str());
//         cudaEventRecord(start_, stream); 
//     }

//     void stop(cudaStream_t &stream) {
//         cudaEventRecord(stop_, stream);
//         cudaEventSynchronize(stop_);
//         float milliseconds = 0;
//         cudaEventElapsedTime(&milliseconds, start_, stop_);
//         total_time_ += milliseconds;
//         nvtxRangePop();
//     }

//     float getTotalTime() const { return total_time_; }

//     float getThroughput(float data_size_gb) const {
//         return data_size_gb / (total_time_ / 1000);   // Throughput in GB/s
//     }

//   private:
//     cudaEvent_t start_, stop_;
//     float total_time_;
//     std::string event_;
// };
