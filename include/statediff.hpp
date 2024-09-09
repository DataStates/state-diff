#ifndef __STATE_DIFF_HPP
#define __STATE_DIFF_HPP

#include "Kokkos_Bitset.hpp"
#include "Kokkos_Core.hpp"
#include "Kokkos_ScatterView.hpp"
#include "Kokkos_Sort.hpp"
#include "common/compare_utils.hpp"
#include "common/debug.hpp"
#include "common/merkle_tree.hpp"
#include "common/statediff_bitset.hpp"
#include "io_reader.hpp"
#include "mmap_reader.hpp"
#include "reader_factory.hpp"
#include <climits>
#include <cstddef>
#include <fstream>
#include <functional>
#include <iostream>
#include <vector>

namespace state_diff {

template <typename DataType, template <typename> typename Reader>
class client_t {

    struct client_info_t {
        int id;
        char data_type;
        size_t data_len;
        size_t chunk_size;
        size_t start_level;
        double error_tolerance;

        // Ensuring client_info_t is serializable
        template <class Archive>
        void serialize(Archive &ar, const unsigned int version) {
            ar << id;
            ar << data_type;
            ar << chunk_size;
            ar << error_tolerance;
        }

        // operator to assess two clients to make sure metadata match
        bool operator==(const client_info_t &other) const {
            return (data_type == other.data_type) &&
                   (data_len == other.data_len) &&
                   (chunk_size == other.chunk_size) &&
                   (start_level == other.start_level) &&
                   (error_tolerance == other.error_tolerance);
        }
    };

    // Defaults
    static const size_t DEFAULT_CHUNK_SIZE = 4096;
    static const size_t DEFAULT_START_LEVEL = 13;
    static const bool DEFAULT_FUZZY_HASH = true;
    static const char DEFAULT_DTYPE = 'f';

    // client variables
    tree_t *tree;
    Reader<DataType> io_reader;
    client_info_t client_info;

    // comparison state
    Queue working_queue;
    // Bitset for tracking which chunks have been changed
    Kokkos::Bitset<> changed_chunks;
    // Vec of idx of chunks that are marked different during the 1st phase
    Vector<size_t> diff_hash_vec;
    Kokkos::View<uint64_t[1]> num_comparisons =
        Kokkos::View<uint64_t[1]>("Num comparisons");
    Kokkos::View<uint64_t[1]> num_changed =
        Kokkos::View<uint64_t[1]>("Num changed");
    Kokkos::View<uint64_t[1]> num_hash_comp =
        Kokkos::View<uint64_t[1]>("Num hash comparisons");
    size_t nchange = 0;

    // timers
    std::vector<double> io_timer;
    double compare_timer;

  public:
    // Stats getters
    size_t get_num_hash_comparisons() const;
    size_t get_num_comparisons() const;
    size_t get_num_changes() const;
    double get_io_time() const;
    double get_compare_time() const;

    client_t(int client_id, Reader<DataType> &reader, size_t data_len,
             double error, char dtype = DEFAULT_DTYPE,
             size_t chunk_size = DEFAULT_CHUNK_SIZE,
             size_t start_level = DEFAULT_START_LEVEL,
             bool fuzzyhash = DEFAULT_FUZZY_HASH);
    ~client_t();

    void create(const std::vector<uint8_t> &data);
    void save(const std::string &filename);
    void load(const std::string &filename);

    bool compare_with(client_t &prev);
    size_t compare_trees(const client_t &prev, Queue &working_queue,
                         Vector<size_t> &diff_hash_vec,
                         Kokkos::View<uint64_t[1]> &num_hash_comp);
    size_t compare_data(client_t &prev, Vector<size_t> &diff_hash_vec,
                        Kokkos::Bitset<> &changed_chunks,
                        Kokkos::View<uint64_t[1]> &num_changed,
                        Kokkos::View<uint64_t[1]> &num_comparisons);
};

template <typename DataType, template <typename> typename Reader>
client_t<DataType, Reader>::client_t(int id, Reader<DataType> &reader,
                                     size_t data_length, double error,
                                     char dtype, size_t chunksize, size_t start,
                                     bool fuzzyhash)
    : io_reader(reader) {
    DEBUG_PRINT("Begin setup\n");
    std::string setup_region_name = std::string("StateDiff:: Checkpoint ") +
                                    std::to_string(id) + std::string(": Setup");
    Kokkos::Profiling::pushRegion(setup_region_name.c_str());
    client_info =
        client_info_t{id, dtype, data_length, chunksize, start, error};
    tree = new tree_t(data_length, chunksize, fuzzyhash);

    size_t n_chunks = data_length / chunksize;
    if (n_chunks * chunksize < data_length)
        n_chunks += 1;

    working_queue = Queue(n_chunks);
    changed_chunks = Kokkos::Bitset<>(n_chunks);
    changed_chunks.reset();
    Kokkos::resize(diff_hash_vec.vector_d, n_chunks);
    Kokkos::resize(diff_hash_vec.vector_h, n_chunks);

    // Clear stats
    diff_hash_vec.clear();
    Kokkos::deep_copy(num_comparisons, 0);
    Kokkos::deep_copy(num_hash_comp, 0);
    Kokkos::deep_copy(num_changed, 0);

    Kokkos::Profiling::popRegion();
    DEBUG_PRINT("Finished setup\n");
}

template <typename DataType, template <typename> typename Reader>
client_t<DataType, Reader>::~client_t() {
    delete tree;
    tree = nullptr;
}

/**
 * Create the corresponding tree
 */
template <typename DataType, template <typename> typename Reader>
void
client_t<DataType, Reader>::create(const std::vector<uint8_t> &data) {
    tree->create(data, client_info.error_tolerance, client_info.data_type,
                 client_info.start_level);
}

/**
 * Serialize and save {clientInfo, tree} to file
 */
template <typename DataType, template <typename> typename Reader>
void
client_t<DataType, Reader>::save(const std::string &filename) {
    std::ofstream ofs(filename);
    boost::archive::binary_oarchive oa(ofs);
    oa << client_info;
    oa << tree;
}

/**
 * load and deserialize {clientInfo, tree} from file
 */
template <typename DataType, template <typename> typename Reader>
void
client_t<DataType, Reader>::load(const std::string &filename) {
    std::ifstream ifs(filename);
    boost::archive::binary_iarchive ia(ifs);
    ia >> client_info;
    ia >> tree;
}

template <typename DataType, template <typename> typename Reader>
bool
client_t<DataType, Reader>::compare_with(client_t &prev) {
    ASSERT(tree != nullptr && prev.tree != nullptr);
    ASSERT(client_info == prev.client_info);

    compare_trees(prev, working_queue, diff_hash_vec, num_hash_comp);

    // Validate first occurences with direct comparison
    if (diff_hash_vec.size() > 0)
        compare_data(prev, diff_hash_vec, changed_chunks, num_changed,
                     num_comparisons);
    io_timer = io_reader.get_timer();
    return get_num_changes() == 0;
}

template <typename DataType, template <typename> typename Reader>
size_t
client_t<DataType, Reader>::compare_trees(
    const client_t &prev, Queue &working_queue, Vector<size_t> &diff_hash_vec,
    Kokkos::View<uint64_t[1]> &num_hash_comp) {
    std::string diff_label = std::string("Chkpt ") +
                             std::to_string(client_info.id) + std::string(": ");
    Kokkos::Profiling::pushRegion(diff_label + std::string("Compare Trees"));

    Kokkos::Profiling::pushRegion(diff_label +
                                  std::string("Compare Trees setup"));
    // Grab references to current and previous tree
    tree_t &tree_curr = *tree;
    tree_t &tree_prev = *prev.tree;
    // Setup markers for beginning and end of tree level
    uint32_t level_beg = 0;
    uint32_t level_end = 0;
    while (level_beg < prev.tree->num_nodes) {
        level_beg = 2 * level_beg + 1;
        level_end = 2 * level_end + 2;
    }
    level_beg = (level_beg - 1) / 2;
    level_end = (level_end - 2) / 2;
    uint32_t left_leaf = level_beg;
    uint32_t right_leaf = level_end;
    uint32_t last_lvl_beg = (1 << client_info.start_level) - 1;
    uint32_t last_lvl_end = (1 << (client_info.start_level + 1)) - 2;
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
    auto &diff_hashes = diff_hash_vec;
    auto n_chunks = prev.tree->num_leaves;
    auto n_nodes = prev.tree->num_nodes;

    // Compare trees level by level
    while (work_queue.size() > 0) {
        Kokkos::parallel_for(
            "Process queue", Kokkos::RangePolicy<>(0, work_queue.size()),
            KOKKOS_LAMBDA(uint32_t i) {
                auto nhash_comp_access = nhash_comp.access();
                uint32_t node = work_queue.pop();
                bool identical = digests_same(tree_curr[node], tree_prev[node]);
                nhash_comp_access(0) += 1;
                if (!identical) {
                    if ((n_chunks - 1 <= node) && (node < n_nodes)) {
                        if (node < left_leaf) {
                            // Leaf is not on the last level
                            size_t entry = (n_nodes - left_leaf) +
                                           (node - ((n_nodes - 1) / 2));
                            diff_hashes.push(entry);
                        } else {
                            // Leaf is on the last level
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

template <typename DataType, template <typename> typename Reader>
size_t
client_t<DataType, Reader>::compare_data(
    client_t &prev, Vector<size_t> &diff_hash_vec,
    Kokkos::Bitset<> &changed_chunks, Kokkos::View<uint64_t[1]> &num_changed,
    Kokkos::View<uint64_t[1]> &num_comparisons) {
    STDOUT_PRINT("Number of first occurrences (Leaves) - Phase One: %u\n",
                 diff_hash_vec.size());
    std::string diff_label = std::string("Chkpt ") +
                             std::to_string(client_info.id) + std::string(": ");
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
    size_t elemPerChunk = client_info.chunk_size / sizeof(DataType);
    Kokkos::Profiling::popRegion();

    double err_tol = client_info.error_tolerance;
    AbsoluteComp<DataType> abs_comp;
    if (num_diff_hash > 0) {
        Kokkos::deep_copy(diff_hash_vec.vector_h, diff_hash_vec.vector_d);
        std::vector<segment_t> segments0, segments1;
        std::vector<DataType> buffer0(num_diff_hash * elemPerChunk),
            buffer1(num_diff_hash * elemPerChunk);
        for (size_t i = 0; i < num_diff_hash; i++) {
            segment_t seg0, seg1;
            seg0.id = i;
            seg0.buffer =
                (uint8_t *) (buffer0.data()) + i * client_info.chunk_size;
            seg0.offset = diff_hash_vec.vector_h(i) * client_info.chunk_size;
            seg0.size = client_info.chunk_size;
            segments0.push_back(seg0);
            seg1.id = i;
            seg1.buffer =
                (uint8_t *) (buffer1.data()) + i * client_info.chunk_size;
            seg1.offset = diff_hash_vec.vector_h(i) * client_info.chunk_size;
            seg1.size = client_info.chunk_size;
            segments1.push_back(seg1);
        }

        liburing_io_reader_t *reader0 =
            new liburing_io_reader_t(prev.io_reader.filename);
        liburing_io_reader_t *reader1 =
            new liburing_io_reader_t(io_reader.filename);
        // mmap_io_reader_t* reader0 = new
        // mmap_io_reader_t(prev.io_reader.filename); mmap_io_reader_t* reader1
        // = new mmap_io_reader_t(io_reader.filename); posix_io_reader_t*
        // reader0 = new posix_io_reader_t(prev.io_reader.filename);
        // posix_io_reader_t* reader1 = new
        // posix_io_reader_t(io_reader.filename);
        reader0->enqueue_reads(segments0);
        reader1->enqueue_reads(segments1);
        reader0->wait_all();
        reader1->wait_all();

        Kokkos::Profiling::pushRegion(
            diff_label + std::string("Compare Tree direct comparison"));
        Timer::time_point beg = Timer::now();
        uint64_t ndiff = 0;
        // Parallel comparison
        using PolicyType =
            Kokkos::RangePolicy<size_t, Kokkos::DefaultHostExecutionSpace>;
        auto range_policy = PolicyType(0, num_diff_hash * elemPerChunk);
        const segment_t *segments = segments0.data();
        const DataType *prev_buffer = buffer0.data();
        const DataType *curr_buffer = buffer1.data();
        Kokkos::parallel_reduce(
            "Count differences", range_policy,
            KOKKOS_LAMBDA(const size_t idx, uint64_t &update) {
                size_t i = idx / elemPerChunk;   // Block
                size_t j = idx % elemPerChunk;   // Element in block
                size_t data_idx = segments[i].offset + j * sizeof(DataType);
                if (data_idx < client_info.data_len) {
                    if (!abs_comp(prev_buffer[idx], curr_buffer[idx],
                                  err_tol)) {
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
        delete reader0;
        delete reader1;
    }

    STDOUT_PRINT("Number of changed elements - Phase Two: %lu\n", num_diff);
    Kokkos::Profiling::popRegion();
    return nchange;
}

template <typename DataType, template <typename> typename Reader>
size_t
client_t<DataType, Reader>::get_num_hash_comparisons() const {
    auto num_hash_comp_h = Kokkos::create_mirror_view(num_hash_comp);
    Kokkos::deep_copy(num_hash_comp_h, num_hash_comp);
    return num_hash_comp_h(0);
}

template <typename DataType, template <typename> typename Reader>
size_t
client_t<DataType, Reader>::get_num_comparisons() const {
    auto num_comparisons_h = Kokkos::create_mirror_view(num_comparisons);
    Kokkos::deep_copy(num_comparisons_h, num_comparisons);
    return num_comparisons_h(0);
}

template <typename DataType, template <typename> typename Reader>
size_t
client_t<DataType, Reader>::get_num_changes() const {
    return nchange;
}

template <typename DataType, template <typename> typename Reader>
double
client_t<DataType, Reader>::get_io_time() const {
    return io_timer[0];
}

template <typename DataType, template <typename> typename Reader>
double
client_t<DataType, Reader>::get_compare_time() const {
    return compare_timer;
}

}   // namespace state_diff

#endif   // __STATE_DIFF_HPP
