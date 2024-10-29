#include "merkle_tree.hpp"
#include "common/debug.hpp"

using ExecSpace = Kokkos::DefaultExecutionSpace;

template void
tree_t::save<cereal::BinaryOutputArchive>(cereal::BinaryOutputArchive &,
                                          const unsigned int) const;

template void
tree_t::load<cereal::BinaryInputArchive>(cereal::BinaryInputArchive &,
                                         const unsigned int);

void
tree_t::digest_to_hex(const uint8_t *digest, char *output) {
    int i, j;
    char *c = output;
    for (i = 0; i < static_cast<int>(sizeof(HashDigest) / 4); i++) {
        for (j = 0; j < 4; j++) {
            sprintf(c, "%02X", digest[i * 4 + j]);
            c += 2;
        }
        sprintf(c, " ");
        c += 1;
    }
    *(c - 1) = '\0';
}

KOKKOS_INLINE_FUNCTION
bool
tree_t::calc_hash(uint32_t u) const {
    uint32_t child_l = 2 * u + 1, child_r = 2 * u + 2;
    HashDigest temp[2];
    memcpy((uint8_t *)(&temp[0]), (uint8_t *)(&tree_d(child_l)),
           sizeof(HashDigest));
    memcpy((uint8_t *)(&temp[1]), (uint8_t *)(&tree_d(child_r)),
           sizeof(HashDigest));
    kokkos_murmur3::hash(&temp, 2 * sizeof(HashDigest),
                         (uint8_t *)(&tree_d(u)));
    return false;
}

KOKKOS_INLINE_FUNCTION
bool
tree_t::calc_leaf_hash(const void *data, uint64_t size, uint32_t u) const {
    kokkos_murmur3::hash(data, size, (uint8_t *)(&tree_d(u)));
    return false;
}

KOKKOS_INLINE_FUNCTION
bool
tree_t::calc_leaf_fuzzy_hash(const void *data, uint64_t size, float errorValue,
                             const char dataType, uint32_t u) const {

    HashDigest digests[2] = {0};
    roundinghash(data, size, dataType, errorValue, digests);
    // Set the bit in the hashnum_bitset if both hashes are valid
    tree_d(u) = digests[0];
    return false;
}

/**
 * Allocate space for list of hashes on device and host. Tree is complete and
 * binary so # of nodes is 2*num_leaves-1
 *
 * \param num_leaves Number of leaves in the tree
 */
tree_t::tree_t(const size_t data_size, const size_t c_size, bool fuzzyhash)
    : chunk_size(c_size), use_fuzzyhash(fuzzyhash) {
    num_leaves = data_size / c_size;
    if (num_leaves * c_size < data_size)
        num_leaves += 1;
    num_nodes = 2 * num_leaves - 1;
    tree_d = Kokkos::View<HashDigest *>("Merkle tree", num_nodes);
}

tree_t::tree_t() {}

KOKKOS_INLINE_FUNCTION
void
tree_t::hash_leaves_kernel(uint8_t *data_ptr, client_info_t client_info,
                           uint32_t left_leaf, uint32_t idx) const {
    auto nchunks = num_leaves;
    auto nnodes = num_nodes;
    auto chunksize = chunk_size;
    auto data_size = client_info.data_size;
    auto dtype = client_info.data_type;
    auto err_tol = client_info.error_tolerance;
    bool use_fuzzy_hash = use_fuzzyhash;
    auto &curr_tree = *this;

    // Calculate leaf node
    uint32_t leaf = left_leaf + idx;
    // Adjust leaf if not on the lowest level
    if (leaf >= nnodes) {
        const uint32_t diff = leaf - nnodes;
        leaf = ((nnodes - 1) / 2) + diff;
    }
    // Determine which chunk of data to hash
    uint32_t num_bytes = chunksize;
    uint64_t offset =
        static_cast<uint64_t>(idx) * static_cast<uint64_t>(chunksize);
    if (idx == nchunks - 1)   // Calculate how much data to hash
        num_bytes = data_size - offset;
    // Hash chunk
    if (use_fuzzy_hash) {
        curr_tree.calc_leaf_fuzzy_hash(data_ptr + offset, num_bytes, err_tol,
                                       dtype, leaf);
    } else {
        curr_tree.calc_leaf_hash(data_ptr + offset, num_bytes, leaf);
    }
}

void
tree_t::create_leaves(uint8_t *data_ptr, client_info_t client_info,
                      uint32_t left_leaf, std::string diff_label) {
    auto data_size = client_info.data_size;
    Kokkos::View<uint8_t *> data_d("Device pointer", data_size);
    Kokkos::View<uint8_t *, Kokkos::HostSpace,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>
        data_h(data_ptr, data_size);
    Kokkos::deep_copy(data_d, data_h);
    auto &curr_tree = *this;

    Kokkos::parallel_for(
        diff_label + std::string("Hash leaves"),
        Kokkos::RangePolicy<>(0, num_leaves), KOKKOS_LAMBDA(uint32_t idx) {
            curr_tree.hash_leaves_kernel(data_d.data(), client_info, left_leaf,
                                         idx);
        });
}

// #ifdef __NVCC__
__global__ void
_hash_leaves_kernel(uint8_t *data_ptr, client_info_t client_info,
                    tree_t tree_obj, uint32_t left_leaf) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    tree_obj.hash_leaves_kernel(data_ptr, client_info, left_leaf, idx);
}

void
tree_t::create_leaves_cuda(uint8_t *data_ptr, client_info_t client_info,
                           uint32_t left_leaf, std::string diff_label) {

    auto data_size = client_info.data_size;
    auto chunksize = chunk_size;
    size_t device_buff_size = client_info.device_buff_size;
    size_t transfer_size = device_buff_size;
    auto &curr_tree = *this;
    int kernel_block_size = 256;

    uint32_t num_transfers = data_size / device_buff_size;
    if (num_transfers * device_buff_size < data_size) {
        num_transfers += 1;
    }

    // Cuda environment setup
    int n_streams = 2;
    uint8_t *device_ptrs[n_streams];
    cudaStream_t streams[n_streams];
    cudaEvent_t events[n_streams];
    // gpuErrchk(cudaHostRegister(data_ptr, data_size, cudaHostRegisterDefault));

    // Variables to hold timing events
    CudaTimer data_load_timer("loading data");
    CudaTimer compute_timer("hashing leaves");
    CudaTimer wait_timer("kernel awaits data");

    for (int i = 0; i < n_streams; i++) {
        gpuErrchk(cudaMalloc((void **)&device_ptrs[i], device_buff_size));
        gpuErrchk(cudaStreamCreate(&streams[i]));
        gpuErrchk(cudaEventCreate(&events[i]));
    }

    for (int i = 0; i < num_transfers; i++) {
        int stream_idx = i % n_streams;

        // Copy the block of data to the current buffer asynchronously
        if (i == (num_transfers - 1)) {
            transfer_size = data_size - (device_buff_size * i);
        }
        size_t offset = i * transfer_size;
        assert(transfer_size % chunksize == 0);
        data_load_timer.start(streams[stream_idx]);
        // nvtxRangePush("data_loading");
        gpuErrchk(cudaMemcpyAsync(device_ptrs[stream_idx], &data_ptr[offset],
                                  device_buff_size, cudaMemcpyHostToDevice,
                                  streams[stream_idx]));
        data_load_timer.stop(streams[stream_idx]);
        // nvtxRangePop();

        // Record event after data transfer is done
        gpuErrchk(cudaEventRecord(events[stream_idx], streams[stream_idx]));

        // Wait for kernel to complete before launching a new kernel
        if (i > 0) {
            int prev_stream_idx = (i - 1) % n_streams;
            wait_timer.start(streams[stream_idx]);
            // nvtxRangePush("kernel_waiting");
            cudaStreamWaitEvent(streams[stream_idx], events[prev_stream_idx],
                                0);
            wait_timer.stop(streams[stream_idx]);
            // nvtxRangePop();
        }

        int num_blocks = ((transfer_size / chunksize) + kernel_block_size - 1) /
                         kernel_block_size;
        compute_timer.start(streams[stream_idx]);
        // nvtxRangePush("kernel_hashing");
        _hash_leaves_kernel<<<num_blocks, kernel_block_size, 0,
                              streams[stream_idx]>>>(
            device_ptrs[stream_idx], client_info, curr_tree, left_leaf);
        compute_timer.record(streams[stream_idx]);
        // nvtxRangePop();
    }

    for (int i = 0; i < n_streams; i++) {
        gpuErrchk(cudaStreamSynchronize(streams[i]));
    }

    // data_load_timer.finalize();
    // wait_timer.finalize();
    compute_timer.finalize();

    for (int i = 0; i < n_streams; i++) {
        gpuErrchk(cudaStreamDestroy(streams[i]));
        gpuErrchk(cudaEventDestroy(events[i]));
        gpuErrchk(cudaFree(device_ptrs[i]));
    }
    // gpuErrchk(cudaHostUnregister(data_ptr));

    // Final result
    timers[0] = data_load_timer.getTotalTime();
    timers[1] = wait_timer.getTotalTime();
    timers[2] = compute_timer.getTotalTime() * num_transfers;
    float data_size_gb =
        static_cast<float>(data_size) / (1024 * 1024 * 1024);   // GB
    printf("Total Data Loading Time: %.3f ms\n", timers[0]);
    printf("Total Data Loading Throughput: %.3f GBps\n",
           data_size_gb / (timers[0] / 1000));
    printf("Total Wait Time: %.3f ms\n", timers[1]);
    printf("Total Compute Time: %.3f ms\n", timers[2]);
    printf("Total Compute Throughput: %.3f GBps\n",
           data_size_gb / (timers[2] / 1000));
}

// #endif   //__NVCC__

void
tree_t::create(uint8_t *data_ptr, client_info_t client_info) {

    // Get number of chunks and nodes
    STDOUT_PRINT("Num chunks: %zu\n", num_leaves);
    STDOUT_PRINT("Num nodes: %zu\n", num_nodes);
    Kokkos::Timer create_timer;

    // Setup markers for beginning and end of tree level
    uint32_t level_beg = 0, level_end = 0;
    while (level_beg < num_nodes) {
        level_beg = 2 * level_beg + 1;
        level_end = 2 * level_end + 2;
    }
    level_beg = (level_beg - 1) / 2;
    level_end = (level_end - 2) / 2;
    uint32_t left_leaf = level_beg;
    uint32_t last_lvl_beg = (1 << client_info.start_level) - 1;

    // Temporary values to avoid capturing this object in the lambda
    auto nchunks = num_leaves;
    auto &curr_tree = *this;

    std::string diff_label = std::string("Diff: ");
    Kokkos::Profiling::pushRegion(diff_label + std::string("Construct Tree"));
    printf("Create Params Init: %.3f ms\n", create_timer.seconds()*1000.0);
    create_timer.reset();
    // Build the tree leaves
#ifdef __NVCC__
    create_leaves_cuda(data_ptr, client_info, left_leaf, diff_label);
#else
    create_leaves(data_ptr, client_info, left_leaf, diff_label);
#endif

    printf("Leaves Creation: %.3f ms\n", create_timer.seconds()*1000.0);
    create_timer.reset();
    // Kokkos::Timer rest_tree_time;
    auto start_rest = std::chrono::high_resolution_clock::now();
    // Build up tree level by level until last_lvl_beg
    while (level_beg >= last_lvl_beg) {
        std::string tree_constr_label =
            diff_label + std::string("Construct level [") +
            std::to_string(level_beg) + std::string(",") +
            std::to_string(level_end) + std::string("]");
        Kokkos::parallel_for(
            tree_constr_label, Kokkos::RangePolicy<>(level_beg, level_end + 1),
            KOKKOS_LAMBDA(const uint32_t node) {
                // Check if node is non leaf
                if (node < nchunks - 1) {
                    curr_tree.calc_hash(node);
                }
            });
        level_beg = (level_beg - 1) / 2;
        level_end = (level_end - 2) / 2;
    }
    printf("Rest of Tree Creation: %.3f ms\n", create_timer.seconds()*1000.0);
    Kokkos::Profiling::popRegion();
    // timers[3] = rest_tree_time.seconds()*1000.0;
    // rest_tree_time.reset();
    auto end_rest = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> create_rest = end_rest - start_rest;
    timers[3] = create_rest.count();
    printf("Remainding tree level build time: %.3f ms\n", create_rest.count());
}

/**
 * Access hash digest in tree
 *
 * \param i Index of tree node
 *
 * \return Reference to hash digest at node i
 */
KOKKOS_INLINE_FUNCTION
HashDigest &
tree_t::operator[](uint32_t i) const {
    return tree_d(i);
}

/**
 * Implementation of Cereal's main Serialize function
 */
template <class Archive>
void
tree_t::save(Archive &ar, const unsigned int version) const {
    std::vector<HashDigest> tree_h(num_nodes);
    Kokkos::View<HashDigest *, Kokkos::HostSpace,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>
        temp_view(tree_h.data(), num_nodes);
    Kokkos::deep_copy(temp_view, tree_d);

    ar(num_leaves);
    ar(cereal::binary_data(tree_h.data(), num_nodes * sizeof(HashDigest)));
}

/**
 * Implementation of Cereal's main Deserialize function
 */
template <class Archive>
void
tree_t::load(Archive &ar, const unsigned int version) {
    ar(num_leaves);
    num_nodes = 2 * num_leaves - 1;
    tree_d = Kokkos::View<HashDigest *>("Merkle tree", num_nodes);

    // temporary buffer for deserialization
    std::vector<HashDigest> tree_h(num_nodes);
    ar(cereal::binary_data(tree_h.data(), num_nodes * sizeof(HashDigest)));

    // Create a host view with the deserialized data and copy to GPU
    Kokkos::View<HashDigest *, Kokkos::HostSpace,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>
        temp_view(tree_h.data(), num_nodes);
    Kokkos::deep_copy(tree_d, temp_view);
}

// Calculate the number of leaves for the tree rooted at node
KOKKOS_INLINE_FUNCTION
uint32_t
tree_t::num_leaf_descendents(uint32_t node, uint32_t num_nodes) {
    uint32_t leftmost = (2 * node) + 1;
    uint32_t rightmost = (2 * node) + 2;
    uint32_t num_leaves = 0;
    while (leftmost < num_nodes) {
        leftmost = (2 * leftmost) + 1;
        rightmost = (2 * rightmost) + 2;
    }
    leftmost = (leftmost - 1) / 2;
    rightmost = (rightmost - 2) / 2;
    uint32_t old_right = rightmost;
    bool split_flag = false;
    if (rightmost > num_nodes - 1) {
        rightmost = num_nodes - 1;
        split_flag = true;
    }
    num_leaves += rightmost - leftmost + 1;
    if (split_flag) {
        leftmost = ((num_nodes - 1) / 2);
        rightmost = (old_right - 2) / 2;
        num_leaves += rightmost - leftmost + 1;
    }
    return num_leaves;
}

// Get the leftmost leaf of the tree rooted at node
KOKKOS_INLINE_FUNCTION
uint32_t
tree_t::leftmost_leaf(uint32_t node, uint32_t num_nodes) {
    uint32_t leftmost = (2 * node) + 1;
    while (leftmost < num_nodes) {
        leftmost = (2 * leftmost) + 1;
    }
    leftmost = (leftmost - 1) / 2;
    return static_cast<uint32_t>(leftmost);
}

// Get the rightmost leaf of the tree rooted at node
KOKKOS_INLINE_FUNCTION
uint32_t
tree_t::rightmost_leaf(uint32_t node, uint32_t num_nodes) {
    uint32_t leftmost = (2 * node) + 1;
    uint32_t rightmost = (2 * node) + 2;
    while (leftmost < num_nodes) {
        leftmost = (2 * leftmost) + 1;
        rightmost = (2 * rightmost) + 2;
    }
    leftmost = (leftmost - 1) / 2;
    rightmost = (rightmost - 2) / 2;
    if (rightmost > num_nodes)
        rightmost = num_nodes - 1;
    return static_cast<uint32_t>(rightmost);
}

/**
 * Print leaves of tree in hex
 */
void
tree_t::print_leaves() {
    Kokkos::View<HashDigest *>::HostMirror tree_h =
        Kokkos::View<HashDigest *>::HostMirror("tree_h", 2 * num_leaves - 1);
    Kokkos::deep_copy(tree_h, tree_d);
    uint32_t num_leaves = (tree_h.extent(0) + 1) / 2;
    printf("============================================================\n");
    char buffer[64];
    unsigned int counter = 2;
    for (unsigned int i = num_leaves - 1; i < tree_h.extent(0); i++) {
        digest_to_hex((uint8_t *)(tree_h(i).digest), buffer);
        printf("Node: %u: %s \n", i, buffer);
        if (i == counter) {
            printf("\n");
            counter += 2 * counter;
        }
    }
    printf("============================================================\n");
}

const double*
tree_t::get_timers() const {
    return timers;
}