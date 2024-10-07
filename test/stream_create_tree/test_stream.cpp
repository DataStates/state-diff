#include <Kokkos_Core.hpp>
#include <cuda_runtime.h>
#include <iostream>

void process_data_with_kokkos_and_cuda(uint8_t* host_data, size_t total_size, size_t chunk_size) {
    // Number of chunks to process
    size_t total_chunks = total_size / chunk_size;

    // Allocate device memory (one chunk buffer to reuse)
    Kokkos::View<uint8_t*> device_buffer("device_buffer", chunk_size);

    // Create CUDA events and streams
    cudaStream_t cuda_stream_1;
    cudaStreamCreate(&cuda_stream_1);

    std::vector<cudaEvent_t> events(total_chunks);

    // Step 1: Asynchronous Memory Transfer and Event Creation
    for (size_t i = 0; i < total_chunks; ++i) {
        // Asynchronously copy a chunk from host to device
        cudaMemcpyAsync(device_buffer.data(), &host_data[i * chunk_size], chunk_size, cudaMemcpyHostToDevice, cuda_stream_1);

        // Create an event to signal when the copy is done
        cudaEventCreateWithFlags(&events[i], cudaEventDisableTiming);
        cudaEventRecord(events[i], cuda_stream_1);  // Record event on stream
    }

    // Step 2: Wait for Events and Launch Kokkos Kernels
    for (size_t i = 0; i < total_chunks; ++i) {
        // Wait for the corresponding data transfer to complete
        cudaStreamWaitEvent(cuda_stream_1, events[i], 0);

        // Launch Kokkos parallel_for to process the chunk once the data is available
        Kokkos::parallel_for("process_chunk", Kokkos::RangePolicy<>(0, chunk_size), KOKKOS_LAMBDA(const int idx) {
            // Process the data in the chunk
            device_buffer(idx) *= 2;  // Example: simple computation
        });

        // Optionally, wait for the kernel to finish before moving to the next iteration
        Kokkos::fence();
    }

    // Clean up events and streams
    for (size_t i = 0; i < total_chunks; ++i) {
        cudaEventDestroy(events[i]);
    }
    cudaStreamDestroy(cuda_stream_1);
}

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);

    const size_t total_size = 1024 * 1024;  // 1MB of data
    const size_t chunk_size = 256 * 1024;   // 256KB chunks

    // Host data to transfer and process
    uint8_t* host_data = new uint8_t[total_size];
    std::fill_n(host_data, total_size, 1);  // Fill host data with 1s

    // Process data using CUDA events and Kokkos
    process_data_with_kokkos_and_cuda(host_data, total_size, chunk_size);

    delete[] host_data;
    Kokkos::finalize();
}
