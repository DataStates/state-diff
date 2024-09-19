#ifndef __STATE_DIFF_HPP
#define __STATE_DIFF_HPP

#include "Kokkos_Bitset.hpp"
#include "Kokkos_Core.hpp"
#include "Kokkos_ScatterView.hpp"
#include "Kokkos_Sort.hpp"
#include "common/compare_utils.hpp"
#include "common/debug.hpp"
#include "common/statediff_bitset.hpp"
#include "common/merkle_tree.hpp"
#include "io_reader.hpp"
#include "mmap_reader.hpp"
#include "reader_factory.hpp"
#include <climits>
#include <cstddef>
#include <vector>

namespace state_diff {

template <typename DataType, template<typename> typename Reader> class client_t {
    int client_id;   // current_id
    tree_t *tree;
    size_t data_len;
    size_t chunk_size;
    size_t num_chunks;    // tree
    size_t num_nodes;     // tree
    size_t start_level;   // tree
    bool use_fuzzyhash;
    double errorValue;
    char dataType;
    std::string data_fn;
    Reader<DataType> io_reader;

    CompareOp comp_op = Absolute;
    Queue working_queue;
    Kokkos::Bitset<>
        changed_chunks;   // Bitset for tracking which chunks have been changed
    Vector<size_t> diff_hash_vec;   // Vec of idx of chunks that are marked
                                    // different during the 1st phase

    // Defaults
    static const size_t DEFAULT_CHUNK_SIZE = 4096;
    static const size_t DEFAULT_START_LEVEL = 13;
    static const bool DEFAULT_FUZZY_HASH = true;
    static const char DEFAULT_DTYPE = 'f';

    // Internal implementations
    size_t compare_trees(const client_t &prev, Queue &working_queue,
                         Vector<size_t> &diff_hash_vec,
                         Kokkos::View<uint64_t[1]> &num_hash_comp);
    size_t compare_data(client_t &prev, Vector<size_t> &diff_hash_vec,
                        Kokkos::Bitset<> &changed_chunks,
                        Kokkos::View<uint64_t[1]> &num_changed,
                        Kokkos::View<uint64_t[1]> &num_comparisons);
    template<typename FileReader>
    size_t compare_data_new_reader(client_t &prev, 
                        FileReader& reader0,
                        FileReader& reader1,
                        Vector<size_t> &diff_hash_vec,
                        Kokkos::Bitset<> &changed_chunks,
                        Kokkos::View<uint64_t[1]> &num_changed,
                        Kokkos::View<uint64_t[1]> &num_comparisons);
  public:

    // timers
    //std::vector<double> io_timer;
    double timers[2];
    double read_timer;
    double compare_timer;

    // Stats
    Kokkos::View<uint64_t[1]> num_comparisons =
        Kokkos::View<uint64_t[1]>("Num comparisons");
    Kokkos::View<uint64_t[1]> num_changed =
        Kokkos::View<uint64_t[1]>("Num changed");
    Kokkos::View<uint64_t[1]> num_hash_comp =
        Kokkos::View<uint64_t[1]>("Num hash comparisons");
    size_t nchange=0;

    // Constructor and destructor
    client_t(int client_id, Reader<DataType> &reader, size_t data_len,
             double error, char dtype = DEFAULT_DTYPE,
             size_t chunk_size = DEFAULT_CHUNK_SIZE,
             size_t start_level = DEFAULT_START_LEVEL,
             bool fuzzyhash = DEFAULT_FUZZY_HASH);

    ~client_t();

    // Stats getters
    size_t get_num_hash_comparisons() const;
    size_t get_num_comparisons() const;
    size_t get_num_changes() const;
    double get_tree_comparison_time() const;
    double get_compare_time() const;


    void create(const std::vector<uint8_t> &data);
    template <class Archive> void serialize(Archive &ar);
    std::vector<uint8_t> serialize();
    size_t deserialize(std::vector<uint8_t> &tree_data);
    bool compare_with(client_t &prev);
    bool compare_with_new_reader(client_t &prev);

};

template <typename DataType, template<typename> typename Reader>
client_t<DataType, Reader>::client_t(int id, Reader<DataType> &reader,
                             size_t data_length, double error, char dtype,
                             size_t chunksize, size_t start, bool fuzzyhash)
    : client_id(id), io_reader(reader), data_len(data_length),
      errorValue(error), dataType(dtype), chunk_size(chunksize),
      start_level(start), use_fuzzyhash(fuzzyhash)  {
    DEBUG_PRINT("Begin setup\n");
    std::string setup_region_name = std::string("StateDiff:: Checkpoint ") +
                                    std::to_string(client_id) +
                                    std::string(": Setup");
    Kokkos::Profiling::pushRegion(setup_region_name.c_str());
    num_chunks = data_len / chunk_size;
    if (num_chunks * chunk_size < data_len)
        num_chunks += 1;
    num_nodes = 2 * num_chunks - 1;
    working_queue = Queue(num_chunks);
    tree = new tree_t(num_chunks);

    changed_chunks = Kokkos::Bitset<>(num_chunks);
    changed_chunks.reset();
    tree->dual_hash_d.clear();
    if (tree->tree_d.size() < num_nodes) {
        Kokkos::resize(tree->tree_d, num_nodes);
        Kokkos::resize(tree->tree_h, num_nodes);
        tree->dual_hash_d =
            Dedupe::Bitset<Kokkos::DefaultExecutionSpace>(num_nodes);
        tree->dual_hash_h =
            Dedupe::Bitset<Kokkos::DefaultHostExecutionSpace>(num_nodes);
    }
    Kokkos::resize(diff_hash_vec.vector_d, num_chunks);
    Kokkos::resize(diff_hash_vec.vector_h, num_chunks);

    // Clear stats
    timers[0] = 0;
    timers[1] = 0;
    read_timer = 0;
    compare_timer = 0;
    diff_hash_vec.clear();
    Kokkos::deep_copy(num_comparisons, 0);
    Kokkos::deep_copy(num_hash_comp, 0);
    Kokkos::deep_copy(num_changed, 0);

    Kokkos::Profiling::popRegion();
    DEBUG_PRINT("Finished setup\n");
}

template <typename DataType, template<typename> typename Reader> client_t<DataType, Reader>::~client_t() {
    delete tree;
    tree = nullptr;
}

template <typename DataType, template<typename> typename Reader>
void
client_t<DataType, Reader>::create(const std::vector<uint8_t> &data) {

    // Get a uint8_t pointer to the data
    const DataType *data_ptr = (DataType*)data.data();
    const uint8_t *uint8_ptr = reinterpret_cast<const uint8_t *>(data_ptr);

    // Grab references to tree object
    tree_t &tree_curr = *tree;

    // Get number of chunks and nodes
    STDOUT_PRINT("Chunk size: %zu\n", chunk_size);
    STDOUT_PRINT("Num chunks: %zu\n", num_chunks);
    STDOUT_PRINT("Num nodes: %zu\n", num_nodes);

    // Setup markers for beginning and end of tree level
    uint32_t level_beg = 0, level_end = 0;
    while (level_beg < num_nodes) {
        level_beg = 2 * level_beg + 1;
        level_end = 2 * level_end + 2;
    }
    level_beg = (level_beg - 1) / 2;
    level_end = (level_end - 2) / 2;
    uint32_t left_leaf = level_beg;
    uint32_t last_lvl_beg = (1 << start_level) - 1;

    // Temporary values to avoid capturing this object in the lambda
    auto nchunks = num_chunks;
    auto nnodes = num_nodes;
    auto chunksize = chunk_size;
    auto data_size = data_len;
    auto dtype = dataType;
    auto err_tol = errorValue;
    bool use_fuzzy_hash = use_fuzzyhash && (comp_op != Equivalence);

    std::string diff_label =
        std::string("Diff ") + std::to_string(client_id) + std::string(": ");
    Kokkos::Profiling::pushRegion(diff_label + std::string("Construct Tree"));
    Kokkos::parallel_for(
        diff_label + std::string("Hash leaves"),
        Kokkos::RangePolicy<>(0, num_chunks), KOKKOS_LAMBDA(uint32_t idx) {
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
                tree_curr.calc_leaf_fuzzy_hash(uint8_ptr + offset, num_bytes,
                                               err_tol, dtype, leaf);
            } else {
                tree_curr.calc_leaf_hash(uint8_ptr + offset, num_bytes, leaf);
            }
        });
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
                    tree_curr.calc_hash(node);
                }
            });
        level_beg = (level_beg - 1) / 2;
        level_end = (level_end - 2) / 2;
    }
    Kokkos::Profiling::popRegion();
    return;
}

template <typename DataType, template<typename> typename Reader>
std::vector<uint8_t>
client_t<DataType, Reader>::serialize() {
    Kokkos::Timer timer_total;
    Kokkos::Timer timer_section;

    // Copy tree to host
    timer_section.reset();
    Kokkos::deep_copy(tree->tree_h, tree->tree_d);

    HashDigest *tree_ptr = tree->tree_h.data();
    uint64_t size = tree->tree_h.extent(0) * tree->tree_h.extent(1);
    uint32_t hashes_per_node = static_cast<uint32_t>(tree->tree_h.extent(1));
    STDOUT_PRINT("Time for copying tree to host: %f seconds.\n",
                 timer_section.seconds());

    size_t buffer_len = sizeof(client_id) + sizeof(chunk_size) +
                        sizeof(num_chunks) + sizeof(hashes_per_node) +
                        (sizeof(HashDigest) * size);
    if (hashes_per_node > 1) {
        Dedupe::deep_copy(tree->dual_hash_h, tree->dual_hash_d);
        buffer_len +=
            sizeof(unsigned int) * ((tree->dual_hash_h.size() + 31) / 32);
    }

    timer_section.reset();
    // preallocating the vector to simplify things
    //  size of current id, chunk size, hashes/node, size + size of data in tree
    //  (16 times number of nodes as 1 digest is 16 bytes) + the size of
    //  dual_hash rounded up to the nearest byte.
    std::vector<uint8_t> buffer(buffer_len);

    size_t offset = 0;
    STDOUT_PRINT("Time for preallocating vector: %f seconds.\n",
                 timer_section.seconds());

    // inserting the current id
    timer_section.reset();
    memcpy(buffer.data() + offset, &client_id, sizeof(client_id));
    offset += sizeof(client_id);
    STDOUT_PRINT("Time for inserting the current id: %f seconds.\n",
                 timer_section.seconds());

    // inserting the chunk size
    timer_section.reset();
    memcpy(buffer.data() + offset, &chunk_size, sizeof(chunk_size));
    offset += sizeof(chunk_size);
    STDOUT_PRINT("Time for inserting the chunk size: %f seconds.\n",
                 timer_section.seconds());

    // inserting the number of chunks
    timer_section.reset();
    memcpy(buffer.data() + offset, &num_chunks, sizeof(num_chunks));
    offset += sizeof(num_chunks);
    STDOUT_PRINT("Time for inserting the number of chunks: %f seconds.\n",
                 timer_section.seconds());

    // inserting the number of hashes per node
    timer_section.reset();
    memcpy(buffer.data() + offset, &hashes_per_node, sizeof(hashes_per_node));
    offset += sizeof(hashes_per_node);
    STDOUT_PRINT(
        "Time for inserting the number of hashes per node: %f seconds.\n",
        timer_section.seconds());

    // inserting tree_h
    timer_section.reset();
    memcpy(buffer.data() + offset, tree_ptr, size * sizeof(HashDigest));
    offset += size * sizeof(HashDigest);
    STDOUT_PRINT("Time for inserting tree_h: %f seconds.\n",
                 timer_section.seconds());

    // inserting dual_hash_h
    if (hashes_per_node > 1) {
        timer_section.reset();
        memcpy(buffer.data() + offset, tree->dual_hash_h.data(),
               sizeof(unsigned int) * ((tree->dual_hash_h.size() + 31) / 32));
        STDOUT_PRINT("Time for inserting dual_hash_h: %f seconds.\n",
                     timer_section.seconds());
    }
    STDOUT_PRINT("Total time for serialize function: %f seconds.\n",
                 timer_total.seconds());

    return buffer;
}

/**
 * Deserialize the current Merkle tree as well as needed metadata
 */
template <typename DataType, template<typename> typename Reader>
size_t
client_t<DataType, Reader>::deserialize(std::vector<uint8_t> &buffer) {
    size_t offset = 0;
    uint32_t t_id, t_chunksize;

    memcpy(&t_id, buffer.data() + offset, sizeof(t_id));
    offset += sizeof(t_id);
    if (client_id != t_id) {
        std::cerr << "deserialize_tree: Tree IDs do not match (" << client_id
                  << " vs " << t_id << ").\n";
        return 0;
    }

    memcpy(&t_chunksize, buffer.data() + offset, sizeof(t_chunksize));
    offset += sizeof(t_chunksize);
    if (chunk_size != t_chunksize) {
        std::cerr << "deserialize_tree: Tree chunk sizes do not match ("
                  << chunk_size << " vs " << t_chunksize << ").\n";
        return 0;
    }

    memcpy(&num_chunks, buffer.data() + offset, sizeof(num_chunks));
    num_nodes = 2 * num_chunks - 1;
    offset += sizeof(num_chunks);
    changed_chunks = Kokkos::Bitset<>(num_chunks);

    uint32_t hashes_per_node = 1;
    memcpy(&hashes_per_node, buffer.data() + offset, sizeof(hashes_per_node));
    offset += sizeof(hashes_per_node);

    if (tree != nullptr && tree->tree_h.extent(0) < num_nodes &&
        tree->tree_h.extent(1) < hashes_per_node)
        tree = new tree_t(num_chunks, hashes_per_node);

    using HashDigest2DView =
        Kokkos::View<HashDigest **, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    HashDigest *raw_ptr =
        reinterpret_cast<HashDigest *>(buffer.data() + offset);
    HashDigest2DView unmanaged_view(raw_ptr, num_nodes, hashes_per_node);
    offset += static_cast<size_t>(num_nodes) *
              static_cast<size_t>(hashes_per_node) * sizeof(HashDigest);
    Kokkos::deep_copy(tree->tree_d, unmanaged_view);

    if (hashes_per_node > 1) {
        memcpy(tree->dual_hash_h.data(), buffer.data() + offset,
               sizeof(unsigned int) * ((num_nodes + 31) / 32));
        offset += sizeof(unsigned int) * ((tree->dual_hash_h.size() + 31) / 32);
        Dedupe::deep_copy(tree->dual_hash_d, tree->dual_hash_h);
    }

    Kokkos::fence();
    return buffer.size();
}

template <typename DataType, template<typename> typename Reader>
bool
client_t<DataType, Reader>::compare_with(client_t &prev) {
    ASSERT(tree != nullptr && prev.tree != nullptr);
    ASSERT(num_chunks == prev.num_chunks);
    ASSERT(num_nodes == prev.num_nodes);
    ASSERT(start_level == prev.start_level);
    ASSERT(chunk_size == prev.chunk_size);
    ASSERT(dataType == prev.dataType);
    ASSERT(errorValue == prev.errorValue);

    Timer::time_point beg = Timer::now();
    compare_trees(prev, working_queue, diff_hash_vec, num_hash_comp);
    Timer::time_point end = Timer::now();
    timers[0] =
        std::chrono::duration_cast<Duration>(end - beg).count();

    // Validate first occurences with direct comparison
    beg = Timer::now();
std::cout << "Number of different hashes after phase 1: " << diff_hash_vec.size() << std::endl;
    if (diff_hash_vec.size() > 0)
        compare_data(prev, diff_hash_vec, changed_chunks, num_changed,
                     num_comparisons);
    //io_timer = io_reader.get_timer();
    // prev.io_timer = prev.io_reader.get_timer();
    end = Timer::now();
    timers[1] =
        std::chrono::duration_cast<Duration>(end - beg).count();
    return get_num_changes() == 0;
}

template <typename DataType, template<typename> typename Reader>
bool
client_t<DataType, Reader>::compare_with_new_reader(client_t &prev) {
    ASSERT(tree != nullptr && prev.tree != nullptr);
    ASSERT(num_chunks == prev.num_chunks);
    ASSERT(num_nodes == prev.num_nodes);
    ASSERT(start_level == prev.start_level);
    ASSERT(chunk_size == prev.chunk_size);
    ASSERT(dataType == prev.dataType);
    ASSERT(errorValue == prev.errorValue);

    liburing_io_reader_t reader0(prev.io_reader.filename);
    liburing_io_reader_t reader1(io_reader.filename);

    Timer::time_point beg = Timer::now();
    auto ndifferent = compare_trees(prev, working_queue, diff_hash_vec, num_hash_comp);
std::cout << "Different hashes: " << ndifferent << std::endl;
    Timer::time_point end = Timer::now();
    timers[0] =
        std::chrono::duration_cast<Duration>(end - beg).count();

    // Validate first occurences with direct comparison
    beg = Timer::now();
std::cout << "Number of different hashes after phase 1: " << diff_hash_vec.size() << std::endl;
    if (diff_hash_vec.size() > 0)
        compare_data_new_reader(prev, reader0, reader1, diff_hash_vec, changed_chunks, num_changed,
                     num_comparisons);
    //io_timer = io_reader.get_timer();
    // prev.io_timer = prev.io_reader.get_timer();
    end = Timer::now();
    timers[1] =
        std::chrono::duration_cast<Duration>(end - beg).count();
    return get_num_changes() == 0;
}

template <typename DataType, template<typename> typename Reader>
size_t
client_t<DataType, Reader>::compare_trees(const client_t &prev, Queue &working_queue,
                                  Vector<size_t> &diff_hash_vec,
                                  Kokkos::View<uint64_t[1]> &num_hash_comp) {
    std::string diff_label = std::string("Chkpt ") +
                             std::to_string(prev.client_id) + std::string(": ");
    Kokkos::Profiling::pushRegion(diff_label + std::string("Compare Trees"));

    Kokkos::Profiling::pushRegion(diff_label +
                                  std::string("Compare Trees setup"));
    // Grab references to current and previous tree
    tree_t &tree_curr = *tree;
    tree_t &tree_prev = *prev.tree;
    // Setup markers for beginning and end of tree level
    uint32_t level_beg = 0;
    uint32_t level_end = 0;
    while (level_beg < prev.num_nodes) {
        level_beg = 2 * level_beg + 1;
        level_end = 2 * level_end + 2;
    }
    level_beg = (level_beg - 1) / 2;
    level_end = (level_end - 2) / 2;
    uint32_t left_leaf = level_beg;
    uint32_t right_leaf = level_end;
    uint32_t last_lvl_beg = (1 << prev.start_level) - 1;
    uint32_t last_lvl_end = (1 << (prev.start_level + 1)) - 2;
    if (last_lvl_beg > left_leaf)
        last_lvl_beg = left_leaf;
    if (last_lvl_end > right_leaf)
        last_lvl_end = right_leaf;
    DEBUG_PRINT("Leaf range [%u,%u]\n", left_leaf, right_leaf);
    DEBUG_PRINT("Start level [%u,%u]\n", last_lvl_beg, last_lvl_end);
    Kokkos::Experimental::ScatterView<uint64_t[1]> nhash_comp(num_hash_comp);
    Kokkos::Profiling::popRegion();

    // Fills up queue with nodes in the stop level or leavs in case of num
    // levels < 13
    Kokkos::Profiling::pushRegion(diff_label + "Compare Trees with queue");
    level_beg = last_lvl_beg;
    level_end = last_lvl_end;
    auto &work_queue = working_queue;
    auto fill_policy = Kokkos::RangePolicy<>(level_beg, level_end + 1);
    Kokkos::parallel_for(
        "Fill up queue with every node in the stop_level", fill_policy,
        KOKKOS_LAMBDA(const uint32_t i) { work_queue.push(i); });
    // Temporary values to pass to the lambda
    auto &prev_dual_hash = tree_prev.dual_hash_d;
    auto &curr_dual_hash = tree_curr.dual_hash_d;
    auto &diff_hashes = diff_hash_vec;
    auto n_chunks = prev.num_chunks;
    auto n_nodes = prev.num_nodes;

    // Compare trees level by level
    while (work_queue.size() > 0) {
        Kokkos::parallel_for(
            "Process queue", Kokkos::RangePolicy<>(0, work_queue.size()),
            KOKKOS_LAMBDA(uint32_t i) {
                auto nhash_comp_access = nhash_comp.access();
                uint32_t node = work_queue.pop();
                bool identical = false;
                if (curr_dual_hash.test(node) && prev_dual_hash.test(node)) {
                    identical =
                        digests_same(tree_curr(node, 0), tree_prev(node, 0)) ||
                        digests_same(tree_curr(node, 0), tree_prev(node, 1)) ||
                        digests_same(tree_curr(node, 1), tree_prev(node, 0)) ||
                        digests_same(tree_curr(node, 1), tree_prev(node, 1));
                    nhash_comp_access(0) += 4;
                } else if (curr_dual_hash.test(node)) {
                    identical =
                        digests_same(tree_curr(node, 0), tree_prev(node, 0)) ||
                        digests_same(tree_curr(node, 1), tree_prev(node, 0));
                    nhash_comp_access(0) += 2;
                } else if (prev_dual_hash.test(node)) {
                    identical =
                        digests_same(tree_curr(node, 0), tree_prev(node, 0)) ||
                        digests_same(tree_curr(node, 0), tree_prev(node, 1));
                    nhash_comp_access(0) += 2;
                } else {
                    identical = digests_same(tree_curr(node), tree_prev(node));
                    nhash_comp_access(0) += 1;
                }
                if (!identical) {
                    if ((n_chunks - 1 <= node) && (node < n_nodes)) {
                        if (node < left_leaf) {   // Leaf is not on the last level
                            size_t entry = (n_nodes - left_leaf) +
                                           (node - ((n_nodes - 1) / 2));
//                            ASSERT(entry < n_chunks);
                            diff_hashes.push(entry);
                        } else {   // Leaf is on the last level
//                            ASSERT(node - left_leaf < n_chunks);
                            diff_hashes.push(node - left_leaf);
                        }
                    } else {
                        uint32_t child_l = 2 * node + 1;
                        uint32_t child_r = 2 * node + 2;
                        if (child_l < n_nodes) {
                            work_queue.push(child_l);
                        }
                        if (child_r < n_nodes) {
                            work_queue.push(child_r);
                        }
                    }
                }
            });
        Kokkos::fence();
    }
    Kokkos::Profiling::popRegion();
    Kokkos::Profiling::pushRegion(
        diff_label + std::string("Contribute hash comparison count"));
    Kokkos::Experimental::contribute(num_hash_comp, nhash_comp);
    Kokkos::Profiling::popRegion();
    Kokkos::Profiling::popRegion();
    return diff_hash_vec.size();
}

template <typename DataType, template<typename> typename Reader>
template<typename FileReader>
size_t
client_t<DataType, Reader>::compare_data_new_reader(client_t &prev,
                                 FileReader& reader0,
                                 FileReader& reader1,
                                 Vector<size_t> &diff_hash_vec,
                                 Kokkos::Bitset<> &changed_chunks,
                                 Kokkos::View<uint64_t[1]> &num_changed,
                                 Kokkos::View<uint64_t[1]> &num_comparisons) {
    STDOUT_PRINT("Number of first occurrences (Leaves) - Phase One: %u\n",
                 diff_hash_vec.size());
    std::string diff_label = std::string("Chkpt ") +
                             std::to_string(prev.client_id) + std::string(": ");
    Kokkos::Profiling::pushRegion(
        diff_label + std::string("Compare Trees direct comparison"));

    // Sort indices for better performance
    Kokkos::Profiling::pushRegion(diff_label +
                                  std::string("Compare Tree sort indices"));
    size_t num_diff_hash = static_cast<size_t>(diff_hash_vec.size());
    auto subview_bounds = Kokkos::make_pair((size_t) (0), num_diff_hash);
    auto diff_hash_subview =
        Kokkos::subview(diff_hash_vec.vector_d, subview_bounds);
    Kokkos::sort(diff_hash_vec.vector_d, 0, num_diff_hash);
    size_t elemPerChunk = prev.chunk_size / sizeof(DataType);
    Kokkos::Profiling::popRegion();
 
//    auto &num_changes = num_changed;
//    auto &changed_blocks = changed_chunks;
    double err_tol = prev.errorValue;
    AbsoluteComp<DataType> abs_comp;
    if (num_diff_hash > 0) {

        Timer::time_point read_beg = Timer::now();

        Kokkos::deep_copy(diff_hash_vec.vector_h, diff_hash_vec.vector_d);
        std::vector<segment_t> segments0(num_diff_hash), segments1(num_diff_hash);
        std::vector<DataType> buffer0(num_diff_hash*elemPerChunk), buffer1(num_diff_hash*elemPerChunk);
        Kokkos::parallel_for("Fill segment vectors", 
          Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,num_diff_hash), 
          [&](size_t i) {
            segments0[i].id = i;
            segments0[i].buffer = (uint8_t*)(buffer0.data())+i*chunk_size;
            segments0[i].offset = diff_hash_vec.vector_h(i)*chunk_size;
            segments0[i].size = chunk_size;

            segments1[i].id = i;
            segments1[i].buffer = (uint8_t*)(buffer1.data())+i*chunk_size;
            segments1[i].offset = diff_hash_vec.vector_h(i)*chunk_size;
            segments1[i].size = chunk_size;
        });
        Timer::time_point seg_end = Timer::now();
        std::cout << "Segment preparation time: " <<  
            std::chrono::duration_cast<Duration>(seg_end - read_beg).count() << std::endl;

        //liburing_io_reader_t* reader0 = new liburing_io_reader_t(prev.io_reader.filename);
        //liburing_io_reader_t* reader1 = new liburing_io_reader_t(io_reader.filename);
        //mmap_io_reader_t* reader0 = new mmap_io_reader_t(prev.io_reader.filename);
        //mmap_io_reader_t* reader1 = new mmap_io_reader_t(io_reader.filename);
        //posix_io_reader_t* reader0 = new posix_io_reader_t(prev.io_reader.filename);
        //posix_io_reader_t* reader1 = new posix_io_reader_t(io_reader.filename);

        reader0.enqueue_reads(segments0);
        reader1.enqueue_reads(segments1);
        reader0.wait_all();
        reader1.wait_all();

        Timer::time_point read_end = Timer::now();
        read_timer +=
            std::chrono::duration_cast<Duration>(read_end - read_beg).count();

        Kokkos::Profiling::pushRegion(
            diff_label + std::string("Compare Tree direct comparison"));
        Timer::time_point beg = Timer::now();
        uint64_t ndiff = 0;
        // Parallel comparison
        using PolicyType = Kokkos::RangePolicy<size_t, Kokkos::DefaultHostExecutionSpace>;
        auto range_policy = PolicyType(0, num_diff_hash*elemPerChunk);
        const segment_t* segments = segments0.data();
        const DataType* prev_buffer = buffer0.data();
        const DataType* curr_buffer = buffer1.data();
        Kokkos::parallel_reduce("Count differences", range_policy,
            KOKKOS_LAMBDA(const size_t idx, uint64_t &update) {
                size_t i = idx / elemPerChunk;   // Block
                size_t j = idx % elemPerChunk;   // Element in block
                size_t data_idx = segments[i].offset + j * sizeof(DataType);
                if(data_idx < data_len) {
                    if(!abs_comp(prev_buffer[idx], curr_buffer[idx], err_tol)) {
                        update += 1;
                    }
                }
            },
            Kokkos::Sum<uint64_t>(ndiff));
        Kokkos::fence();
        nchange += ndiff;
        Kokkos::Profiling::popRegion();
        Timer::time_point end = Timer::now();
        compare_timer +=
            std::chrono::duration_cast<Duration>(end - beg).count();
        //delete reader0;
        //delete reader1;
    }

    STDOUT_PRINT("Number of changed elements - Phase Two: %lu\n", nchange);
    Kokkos::Profiling::popRegion();
    return nchange;
}

template <typename DataType, template<typename> typename Reader>
size_t
client_t<DataType, Reader>::compare_data(client_t &prev,
                                 Vector<size_t> &diff_hash_vec,
                                 Kokkos::Bitset<> &changed_chunks,
                                 Kokkos::View<uint64_t[1]> &num_changed,
                                 Kokkos::View<uint64_t[1]> &num_comparisons) {
    STDOUT_PRINT("Number of first occurrences (Leaves) - Phase One: %u\n",
                 diff_hash_vec.size());
    std::string diff_label = std::string("Chkpt ") +
                             std::to_string(prev.client_id) + std::string(": ");
    Kokkos::Profiling::pushRegion(
        diff_label + std::string("Compare Trees direct comparison"));

    // Sort indices for better performance
    Kokkos::Profiling::pushRegion(diff_label +
                                  std::string("Compare Tree sort indices"));
    size_t num_diff_hash = static_cast<size_t>(diff_hash_vec.size());
    auto subview_bounds = Kokkos::make_pair((size_t) (0), num_diff_hash);
    auto diff_hash_subview =
        Kokkos::subview(diff_hash_vec.vector_d, subview_bounds);
    Kokkos::sort(diff_hash_vec.vector_d, 0, num_diff_hash);
    size_t blocksize = prev.chunk_size / sizeof(DataType);
    Kokkos::Profiling::popRegion();

    uint64_t num_diff = 0;
    auto &num_changes = num_changed;
    auto &changed_blocks = changed_chunks;
    if (num_diff_hash > 0) {

        Timer::time_point read_beg = Timer::now();

        Kokkos::Profiling::pushRegion(
            diff_label + std::string("Compare Tree start file streams"));
        io_reader.start_stream(diff_hash_vec.vector_d.data(), num_diff_hash,
                               blocksize);
        prev.io_reader.start_stream(diff_hash_vec.vector_d.data(),
                                    num_diff_hash, blocksize);
        Kokkos::Profiling::popRegion();

        Kokkos::Profiling::pushRegion(
            diff_label +
            std::string("Compare Tree setup counters and variables"));
        AbsoluteComp<DataType> abs_comp;
        size_t offset_idx = 0;
        double err_tol = prev.errorValue;
        DataType *sliceA = NULL, *sliceB = NULL;
        size_t slice_len = 0;
        size_t *offsets = diff_hash_vec.vector_d.data();
        size_t filesize = io_reader.get_file_size();
        size_t num_iter = num_diff_hash / io_reader.get_chunks_per_slice();
        if (num_iter * io_reader.get_chunks_per_slice() < num_diff_hash)
            num_iter += 1;
        Kokkos::Experimental::ScatterView<uint64_t[1]> num_comp(
            num_comparisons);
        Kokkos::Profiling::popRegion();
        for (size_t iter = 0; iter < num_iter; iter++) {
            Kokkos::Profiling::pushRegion("Next slice");
            sliceA = io_reader.next_slice();
            sliceB = prev.io_reader.next_slice();
            slice_len = io_reader.get_slice_len();
            size_t slice_len_b = prev.io_reader.get_slice_len();

            ASSERT(slice_len == slice_len_b);
            Kokkos::Profiling::popRegion();

            Timer::time_point read_end = Timer::now();
            read_timer +=
                std::chrono::duration_cast<Duration>(read_end - read_beg).count();
            Timer::time_point read_beg = Timer::now();

            Kokkos::Profiling::pushRegion(
                diff_label + std::string("Compare Tree direct comparison"));
            Timer::time_point beg = Timer::now();
            uint64_t ndiff = 0;
            // Parallel comparison
            auto range_policy = Kokkos::RangePolicy<size_t>(0, slice_len);
            Kokkos::parallel_reduce(
                "Count differences", range_policy,
                KOKKOS_LAMBDA(const size_t idx, uint64_t &update) {
                    auto ncomp_access = num_comp.access();
                    size_t i = idx / blocksize;   // Block
                    size_t j = idx % blocksize;   // Element in block
//                    ASSERT(offset_idx + i < num_diff_hash);
                    size_t data_idx =
                        blocksize * sizeof(DataType) * offsets[offset_idx + i] +
                        j * sizeof(DataType);
//                    ASSERT(data_idx < filesize);
                    if ((offset_idx + i < num_diff_hash) &&
                        (data_idx < filesize)) {
                        if (!abs_comp(sliceA[idx], sliceB[idx], err_tol)) {
                            update += 1;
                            changed_blocks.set(offsets[offset_idx + i]);
                        }
                        ncomp_access(0) += 1;
                    }
                },
                Kokkos::Sum<uint64_t>(ndiff));
            Kokkos::fence();
            num_diff += ndiff;
            offset_idx += slice_len / blocksize;
            if (slice_len % blocksize > 0)
                offset_idx += 1;
            Kokkos::Profiling::popRegion();
            Timer::time_point end = Timer::now();
            compare_timer +=
                std::chrono::duration_cast<Duration>(end - beg).count();
        }
        Kokkos::Profiling::pushRegion(diff_label +
                                      std::string("Compare Tree finalize"));
        Kokkos::Experimental::contribute(num_comparisons, num_comp);
        io_reader.end_stream();
        prev.io_reader.end_stream();
        Kokkos::deep_copy(num_changes, num_diff);
        Kokkos::Profiling::popRegion();
    }
    nchange = num_diff;
    STDOUT_PRINT("Number of changed elements - Phase Two: %lu\n", num_diff);
    Kokkos::Profiling::popRegion();
    return nchange;
}

template <typename DataType, template<typename> typename Reader>
size_t
client_t<DataType, Reader>::get_num_hash_comparisons() const {
    auto num_hash_comp_h = Kokkos::create_mirror_view(num_hash_comp);
    Kokkos::deep_copy(num_hash_comp_h, num_hash_comp);
    return num_hash_comp_h(0);
}

template <typename DataType, template<typename> typename Reader>
size_t
client_t<DataType, Reader>::get_num_comparisons() const {
    auto num_comparisons_h = Kokkos::create_mirror_view(num_comparisons);
    Kokkos::deep_copy(num_comparisons_h, num_comparisons);
    return num_comparisons_h(0);
}

template <typename DataType, template<typename> typename Reader>
size_t
client_t<DataType, Reader>::get_num_changes() const {
//    auto num_changed_h = Kokkos::create_mirror_view(num_changed);
//    Kokkos::deep_copy(num_changed_h, num_changed);
//    return num_changed_h(0);
    return nchange;
}

template <typename DataType, template<typename> typename Reader>
double
client_t<DataType, Reader>::get_tree_comparison_time() const {
    return timers[0];
//    return io_timer[0];
}

template <typename DataType, template<typename> typename Reader>
double
client_t<DataType, Reader>::get_compare_time() const {
    return timers[1];
//    return compare_timer;
}

}   // namespace state_diff

#endif   // __STATE_DIFF_HPP
