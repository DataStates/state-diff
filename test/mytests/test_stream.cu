#include <iostream>
#include <string>
#include <cuda_runtime.h>
#include <cassert>
#include <vector>

// Dummy structure to simulate client_info_t
struct client_info_t {
    size_t data_size;
    size_t device_buff_size;
};

// Dummy tree structure
struct tree_t {
    __host__ __device__ void hash_leaves_kernel(uint8_t *data_ptr, client_info_t client_info, uint32_t left_leaf, uint32_t idx, uint8_t *hash_output);

    void create_leaves_cuda(uint8_t *data_ptr, client_info_t client_info, uint32_t left_leaf, std::string diff_label, uint8_t *hash_output);
};

// Kernel declaration (this is your actual kernel)
__global__ void _hash_leaves_kernel(uint8_t *data_ptr, client_info_t client_info, tree_t tree_obj, uint32_t left_leaf, uint8_t *hash_output);

__host__ __device__ void tree_t::hash_leaves_kernel(uint8_t *data_ptr, client_info_t client_info, uint32_t left_leaf, uint32_t idx, uint8_t *hash_output) {
        if (idx < client_info.data_size / 64) {  // Assuming 64-byte chunks for this example
            uint8_t hash_value = 0;
            for (int i = 0; i < 64; ++i) {
                hash_value ^= data_ptr[idx * 64 + i];  // Simple XOR hash over 64 bytes
            }
            hash_output[idx] = hash_value;  // Store hash value for this thread
        }
    }

// Function definition
void tree_t::create_leaves_cuda(uint8_t *data_ptr, client_info_t client_info, uint32_t left_leaf, std::string diff_label, uint8_t *hash_output) {

    auto data_size = client_info.data_size;
    auto chunksize = 64; // Process data in 64-byte chunks
    size_t device_buff_size = client_info.device_buff_size;
    size_t transfer_size = device_buff_size;
    int kernel_block_size = 256;   // Threads per block
    cudaStream_t io_stream, compute_stream;
    cudaStreamCreate(&io_stream);
    cudaStreamCreate(&compute_stream);

    uint32_t num_transfers = data_size / device_buff_size;
    if (num_transfers * device_buff_size < data_size) {
        num_transfers += 1;
    }

    std::vector<cudaEvent_t> events(num_transfers);
    int deviceId;
    cudaGetDevice(&deviceId);
    cudaMemPool_t device_buffer;
    cudaMemPoolProps poolProps = {};
    poolProps.location.type = cudaMemLocationTypeDevice;
    poolProps.location.id = deviceId;
    poolProps.maxSize = device_buff_size;
    cudaMemPoolCreate(&device_buffer, &poolProps);

    for (size_t i = 0; i < num_transfers; ++i) {
        printf("Processing a new transfer\n");
        if (i == (num_transfers - 1)) {
            transfer_size = data_size - (device_buff_size * i);
        }
        size_t offset = i * transfer_size;
        assert(transfer_size % chunksize == 0);

        // Allocate GPU buffer for data to transfer
        uint8_t *device_ptr;
        cudaMallocFromPoolAsync(&device_ptr, transfer_size, device_buffer, io_stream);

        // Transfer data from host to device using the io_stream
        printf("Data transfer\n");
        cudaMemcpyAsync(device_ptr, &data_ptr[offset], transfer_size, cudaMemcpyHostToDevice, io_stream);

        // Using cudaEvents for synchronization
        cudaEventCreateWithFlags(&events[i], cudaEventDisableTiming);
        cudaEventRecord(events[i], io_stream);

        // Wait on the previous event before kernel execution
        if (i > 0)
            cudaStreamWaitEvent(compute_stream, events[i-1], 0);

        // Launch hashing kernel
        printf("Kernel launch\n");
        int num_blocks = ((transfer_size / chunksize) + kernel_block_size - 1) / kernel_block_size;
        _hash_leaves_kernel<<<num_blocks, kernel_block_size, 0, compute_stream>>>(device_ptr, client_info, *this, left_leaf, hash_output);
        cudaPeekAtLastError();

        // Free memory back to the pool
        cudaFreeAsync(device_ptr, compute_stream);
    }

    // Clean up
    for (size_t i = 0; i < num_transfers; ++i) {
        cudaEventDestroy(events[i]);
    }
    cudaStreamDestroy(io_stream);
    cudaStreamDestroy(compute_stream);
}

// Kernel definition
__global__ void
_hash_leaves_kernel(uint8_t *data_ptr, client_info_t client_info, tree_t tree_obj, uint32_t left_leaf, uint8_t *hash_output) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < client_info.data_size / 64) {
        printf("Data at idx %u: %u\n", idx, data_ptr[idx * 64]); // Print first byte of each 64-byte chunk
        tree_obj.hash_leaves_kernel(data_ptr, client_info, left_leaf, idx, hash_output);
    }
}

// Main function to test
int main() {
    size_t data_size = 1024;  // 1KB test data
    size_t device_buff_size = 512;  // Buffer size 512 bytes
    uint32_t left_leaf = 0;
    std::string diff_label = "test";

    // Allocate and initialize host data
    uint8_t *data_ptr = new uint8_t[data_size];
    for (size_t i = 0; i < data_size; ++i) {
        data_ptr[i] = static_cast<uint8_t>(i % 256);
    }

    // Allocate host memory for hash output
    size_t num_hashes = data_size / 64;
    uint8_t *host_hash_output = new uint8_t[num_hashes];

    // Create a dummy client_info object
    client_info_t client_info = {data_size, device_buff_size};

    // Allocate device memory for hash output
    uint8_t *device_hash_output;
    cudaMalloc(&device_hash_output, num_hashes * sizeof(uint8_t));

    // Create a tree object
    tree_t tree;

    // Call the function
    tree.create_leaves_cuda(data_ptr, client_info, left_leaf, diff_label, device_hash_output);

    // Copy hash results back to the host
    cudaMemcpy(host_hash_output, device_hash_output, num_hashes * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    // Display the hash output
    for (size_t i = 0; i < num_hashes; ++i) {
        printf("Hash for chunk %zu: %u\n", i, host_hash_output[i]);
    }

    // Free host and device memory
    delete[] data_ptr;
    delete[] host_hash_output;
    cudaFree(device_hash_output);

    cudaDeviceSynchronize();
    return 0;
}
