#include "data_loader.hpp"
#include "liburing_reader.hpp"
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <typeinfo>
#include <vector>

#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_timer.hpp"

#ifdef __NVCC__
__global__ void
mult_by_two(uint32_t *data, size_t block_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < block_size)
        data[idx] *= 2;
}
#endif

int
validate(uint32_t *data, size_t data_len,
         const std::vector<uint32_t> &dev_output, bool eval_correctness) {
    if (!eval_correctness)
        return 0;

    assert(data_len == dev_output.size());
    for (size_t idx = 0; idx < data_len; ++idx)
        data[idx] *= 2;

    for (size_t i = 0; i < data_len; ++i) {
        if (data[i] != dev_output[i]) {
            std::cerr << "Mismatch at index " << i << ": HOST = " << data[i]
                      << ", DEV = " << dev_output[i] << "\n";
            return -1;
        }
    }
    return 0;
}

template <typename Reader>
std::vector<uint32_t>
test(Reader &reader, std::vector<segment_t> &segments, size_t host_buf_size,
     size_t dev_buf_size, size_t num_batch, bool eval_correctness) {

    auto start = std::chrono::high_resolution_clock::now();
    data_loader<Reader> chkpt_data(reader, segments, host_buf_size,
                                   dev_buf_size, num_batch);
    CudaTimer compute_timer("kernel_exec");
    cudaStream_t compute_stream;
    gpuErrchk(cudaStreamCreate(&compute_stream));
    // Pre-load all data from file to host
    printf("H-Buf-size=%zu; D-Buf-size=%zu; N-seg=%zu; N-batch=%zu\n",
           host_buf_size, dev_buf_size, segments.size(), num_batch);
    size_t batch_len = dev_buf_size / sizeof(uint32_t);
    segment_t *data_ptr = chkpt_data.load();
    std::vector<uint32_t> data_h(batch_len * num_batch, 0);
    if (data_ptr == nullptr) {
        printf("Data has a null ptr\n");
    } else {
        int thread_per_block = 256;
        int num_blocks = (batch_len + thread_per_block - 1) / thread_per_block;

        for (size_t i = 0; i < num_batch; i++) {
            compute_stream = chkpt_data.getStream();
            uint8_t *data_d = chkpt_data.to_device();
            printf("Processing batch %zu\n", i);
            uint32_t *fp_data_d = reinterpret_cast<uint32_t *>(data_d);
            compute_timer.start(compute_stream);
            mult_by_two<<<num_blocks, thread_per_block, 0, compute_stream>>>(
                fp_data_d, batch_len);
            compute_timer.stop(compute_stream);

            if (eval_correctness)
                cudaMemcpy(data_h.data() + i * batch_len, fp_data_d,
                           dev_buf_size, cudaMemcpyDeviceToHost);
        }
        printf("Processed all batches\n");
    }
    compute_timer.finalize();
    chkpt_data.finalize();    
    auto end = std::chrono::high_resolution_clock::now();
    std::vector<float> timers = chkpt_data.getTimings();
    const size_t GB = 1ULL << 30;
    float data_size_gb = static_cast<float>(host_buf_size) / GB;
    printf("Data Loading Time: %.3f ms\n", timers[0]);
    printf("Data Loading Throughput: %.3f GBps\n",
           data_size_gb / (timers[0] / 1000.0f));
    printf("Total Compute Time: %.3f ms\n", compute_timer.getTotalTime());
    printf("Compute Throughput: %.3f GBps\n",
           compute_timer.getThroughput(data_size_gb));
    printf("Total Wait Time: %.3f ms\n", timers[1]);
    std::cout << "Number of Batches: " << num_batch << " - Exec time = "
              << std::chrono::duration_cast<Duration>(end - start).count()
              << std::endl;
    return data_h;
}

int
main(int argc, char **argv) {
    int result = 0;
    std::string filename = argv[1];
    size_t data_size = std::stol(argv[2]);    // size in bytes
    size_t batch_size = std::stol(argv[3]);   // bytes per batch
    bool eval_correctness = false;
    if (argc > 4)
        eval_correctness = (std::stoi(argv[4]) == 0) ? false : true;

    size_t data_len = data_size / sizeof(uint32_t);   // number of elements
    size_t host_buf_size = data_size;
    size_t num_batch = data_size / batch_size;
    if (num_batch * batch_size < data_size)
        num_batch += 1;

    std::vector<segment_t> segments(num_batch);
    std::vector<uint8_t> buffer(data_size, 0);

    // Prepare segments
    for (size_t i = 0; i < num_batch; i++) {
        segment_t seg;
        seg.id = i;
        seg.size = batch_size;
        seg.offset = seg.id * seg.size;
        seg.buffer = buffer.data() + seg.offset;
        segments[i] = seg;
    }

    // Evaluate loader performance
    liburing_io_reader_t uring_reader(filename);
    std::vector<uint32_t> dev_output =
        test(uring_reader, segments, host_buf_size, batch_size, num_batch,
             eval_correctness);

    result += validate((uint32_t *) buffer.data(), data_len, dev_output,
                       eval_correctness);

    return result;
}