#ifndef __STATE_DIFF_HPP
#define __STATE_DIFF_HPP

#include "Kokkos_Bitset.hpp"
#include "Kokkos_Core.hpp"
#include "Kokkos_ScatterView.hpp"
#include "Kokkos_Sort.hpp"
#include "common/compare_utils.hpp"
#include "common/debug.hpp"
#include "common/statediff_bitset.hpp"
#include "data_loader.hpp"
#include "io_reader.hpp"
// #include "reader_factory.hpp"
#include "merkle_tree.hpp"
#include <chrono>
#include <climits>
#include <cstddef>
#include <functional>
#include <iostream>
#include <vector>

#define KB 1024
#define MB (1024 * KB)
#define GB (1024ULL * MB)

namespace state_diff {

template <typename DataType, typename Reader> class client_t {

    // Defaults
    static const size_t DEFAULT_CHUNK_SIZE = 4096;
    static const size_t DEFAULT_START_LEVEL = 13;
    static const bool DEFAULT_FUZZY_HASH = true;
    static const char DEFAULT_DTYPE = 'f';
    static const size_t DEFAULT_HOST_CACHE = 2ULL * GB;
    static const size_t DEFAULT_DEVICE_CACHE = 2ULL * GB;

    // client variables
    client_info_t client_info;
    tree_t tree;
    Reader &io_reader;
    data_loader_t data_loader;

    // comparison state
    Queue working_queue;   // device
    // Bitset for tracking which chunks have been changed
    Kokkos::Bitset<> changed_chunks;   // host
    // Vec of idx of chunks that are marked different during the 1st phase
    Vector<size_t> diff_hash_vec;   // device
    Kokkos::View<uint64_t[1]> num_comparisons =
        Kokkos::View<uint64_t[1]>("Num comparisons");   // host
    Kokkos::View<uint64_t[1]> num_changed =
        Kokkos::View<uint64_t[1]>("Num changed");   // host
    Kokkos::View<uint64_t[1]> num_hash_comp =
        Kokkos::View<uint64_t[1]>("Num hash comparisons");   // device
    size_t nchange = 0;

    // timers
    double timers[2];
    double read_timer;
    double compare_timer;

    void initialize(size_t n_chunks);
    void create_(std::vector<DataType> &data);

  public:
    client_t(int client_id, Reader &reader,
             size_t host_cache_size = DEFAULT_HOST_CACHE,
             size_t dev_cache_size = DEFAULT_DEVICE_CACHE);
    client_t(int client_id, Reader &reader, size_t data_size, double error,
             char dtype = DEFAULT_DTYPE, size_t chunk_size = DEFAULT_CHUNK_SIZE,
             size_t start_level = DEFAULT_START_LEVEL,
             bool fuzzyhash = DEFAULT_FUZZY_HASH,
             size_t host_cache_size = DEFAULT_HOST_CACHE,
             size_t dev_cache_size = DEFAULT_DEVICE_CACHE);
    ~client_t();

    void create(std::vector<DataType> &data);
    void create(uint8_t *data_ptr);
    void create(Reader &reader,
                std::optional<TransferType> cache_tier = std::nullopt);
    template <class Archive>
    void save(Archive &ar, const unsigned int version) const;
    template <class Archive> void load(Archive &ar, const unsigned int version);

    bool compare_with(client_t &prev);

    // Internal implementations
    size_t compare_trees(const client_t &prev, Queue &working_queue,
                         Vector<size_t> &diff_hash_vec,
                         Kokkos::View<uint64_t[1]> &num_hash_comp);
    size_t compare_data(client_t &prev, Vector<size_t> &diff_hash_vec,
                        Kokkos::Bitset<> &changed_chunks,
                        Kokkos::View<uint64_t[1]> &num_changed,
                        Kokkos::View<uint64_t[1]> &num_comparisons);

    // Stats getters
    size_t get_num_hash_comparisons() const;
    size_t get_num_comparisons() const;
    size_t get_num_changes() const;
    double get_tree_comparison_time() const;
    double get_compare_time() const;
    std::vector<double> get_create_time() const;
    // double get_create_time() const;
    client_info_t get_client_info() const;
};

template <typename DataType, typename Reader>
client_t<DataType, Reader>::client_t(int client_id, Reader &reader,
                                     size_t host_cache_size,
                                     size_t dev_cache_size)
    : io_reader(reader), data_loader(host_cache_size, dev_cache_size) {}

// TO-DO: Remove reader from params because we do not it. Do that after
// refactoring the comparison code to use the data_loader.
template <typename DataType, typename Reader>
client_t<DataType, Reader>::client_t(int client_id, Reader &reader,
                                     size_t data_size, double error, char dtype,
                                     size_t min_chunk_size, size_t start,
                                     bool fuzzyhash, size_t host_cache_size,
                                     size_t dev_cache_size)
    : io_reader(reader), data_loader(host_cache_size, dev_cache_size) {
    DBG("Begin client setup");
    std::string setup_region_name = std::string("StateDiff:: Checkpoint ") +
                                    std::to_string(client_id) +
                                    std::string(": Setup");
    Kokkos::Profiling::pushRegion(setup_region_name.c_str());
    size_t optim_chksize = data_loader.get_chunksize(data_size);
    client_info = client_info_t{client_id,      dtype, data_size,
                                min_chunk_size, start, error};
    tree = tree_t(data_size, optim_chksize, fuzzyhash);

    size_t n_chunks = data_size / optim_chksize;
    if (n_chunks * optim_chksize < data_size)
        n_chunks += 1;

    initialize(n_chunks);

    Kokkos::Profiling::popRegion();
    DBG("Finished client setup");
}

template <typename DataType, typename Reader>
void
client_t<DataType, Reader>::initialize(size_t n_chunks) {
    working_queue = Queue(n_chunks);
    changed_chunks = Kokkos::Bitset<>(n_chunks);
    changed_chunks.reset();
    Kokkos::resize(diff_hash_vec.vector_d, n_chunks);
    Kokkos::resize(diff_hash_vec.vector_h, n_chunks);

    // Clear stats
    timers[0] = 0;
    timers[1] = 0;
    read_timer = 0;
    compare_timer = 0;
    diff_hash_vec.clear();
    Kokkos::deep_copy(num_comparisons, 0);
    Kokkos::deep_copy(num_hash_comp, 0);
    Kokkos::deep_copy(num_changed, 0);
}

template <typename DataType, typename Reader>
client_t<DataType, Reader>::~client_t() {}

/**
 * Create the corresponding tree
 */
template <typename DataType, typename Reader>
void
client_t<DataType, Reader>::create(std::vector<DataType> &data) {
    uint8_t *data_ptr = reinterpret_cast<uint8_t *>(data.data());
    tree.create(data_ptr, client_info);
}

template <typename DataType, typename Reader>
void
client_t<DataType, Reader>::create(uint8_t *data_ptr) {
    tree.create(data_ptr, client_info);
}

template <typename DataType, typename Reader>
void
client_t<DataType, Reader>::create(Reader &reader,
                                   std::optional<TransferType> cache_tier) {
    TransferType create_tree_tier =
        cache_tier.value_or(TransferType::FileToHost);
    int ld = data_loader.file_load(reader, 0, client_info.chunk_size, 0,
                                   create_tree_tier);
    tree.create(client_info, data_loader, ld, create_tree_tier);
}

/**
 * Serialize a client into an archive
 */
template <typename DataType, typename Reader>
template <class Archive>
void
client_t<DataType, Reader>::save(Archive &ar,
                                 const unsigned int version) const {
    ar(client_info);
    ar(tree);
}

/**
 * Serialize a client into an archive
 */
template <typename DataType, typename Reader>
template <class Archive>
void
client_t<DataType, Reader>::load(Archive &ar, const unsigned int version) {
    ar(client_info);
    ar(tree);
    initialize(tree.num_leaves);
}

template <typename DataType, typename Reader>
bool
client_t<DataType, Reader>::compare_with(client_t &prev) {
    ASSERT(client_info == prev.client_info);

    Timer::time_point beg = Timer::now();
    auto ndifferent =
        compare_trees(prev, working_queue, diff_hash_vec, num_hash_comp);
    std::cout << "Different hashes: " << ndifferent << std::endl;
    Timer::time_point end = Timer::now();
    timers[0] = std::chrono::duration_cast<Duration>(end - beg).count();

    // Validate first occurences with direct comparison
    beg = Timer::now();
    std::cout << "Number of different hashes after phase 1: "
              << diff_hash_vec.size() << std::endl;
    if (diff_hash_vec.size() > 0)
        compare_data(prev, diff_hash_vec, changed_chunks, num_changed,
                     num_comparisons);
    // compare_data(prev, reader0, reader1, diff_hash_vec, changed_chunks,
    // num_changed,
    //              num_comparisons);
    end = Timer::now();
    timers[1] = std::chrono::duration_cast<Duration>(end - beg).count();
    return get_num_changes() == 0;
}

template <typename DataType, typename Reader>
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
    const tree_t &tree_curr = tree;
    const tree_t &tree_prev = prev.tree;
    // Setup markers for beginning and end of tree level
    uint32_t level_beg = 0;
    uint32_t level_end = 0;
    while (level_beg < tree_prev.num_nodes) {
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
    DBG("Leaf range [" << left_leaf << "," << right_leaf << "]");
    DBG("Start level [" << last_lvl_beg << "," << last_lvl_end << "]");
    Kokkos::Experimental::ScatterView<uint64_t[1]> nhash_comp(num_hash_comp);
    Kokkos::Profiling::popRegion();

    // Fills up queue with nodes in the stop level or leavs if num_level < 13
    Kokkos::Profiling::pushRegion(diff_label + "Compare Trees with queue");
    level_beg = last_lvl_beg;
    level_end = last_lvl_end;
    auto &work_queue = working_queue;
    auto fill_policy = Kokkos::RangePolicy<>(level_beg, level_end + 1);
    Kokkos::parallel_for(
        "Fill up queue with every node in the stop_level", fill_policy,
        KOKKOS_LAMBDA(const uint32_t i) { work_queue.push(i); });
    auto &diff_hashes = diff_hash_vec;
    auto n_chunks = tree_prev.num_leaves;
    auto n_nodes = tree_prev.num_nodes;

    // Compare trees level by level
    while (work_queue.size() > 0) {
        Kokkos::parallel_for(
            "Process queue", Kokkos::RangePolicy<>(0, work_queue.size()),
            KOKKOS_LAMBDA(uint32_t i) {
                auto nhash_comp_access = nhash_comp.access();
                uint32_t node = work_queue.pop();
                bool identical = digests_same(tree_curr.tree_d(node),
                                              tree_prev.tree_d(node));
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

template <typename DataType, typename Reader>
size_t
client_t<DataType, Reader>::compare_data(
    client_t &prev, Vector<size_t> &diff_hash_vec,
    Kokkos::Bitset<> &changed_chunks, Kokkos::View<uint64_t[1]> &num_changed,
    Kokkos::View<uint64_t[1]> &num_comparisons) {
    Timer::time_point setup_beg = Timer::now();
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
    auto subview_bounds = Kokkos::make_pair((size_t)(0), num_diff_hash);
    auto diff_hash_subview =
        Kokkos::subview(diff_hash_vec.vector_d, subview_bounds);
    Kokkos::sort(diff_hash_vec.vector_d, 0, num_diff_hash);
    size_t elemPerChunk = client_info.chunk_size / sizeof(DataType);
    Kokkos::Profiling::popRegion();
    Timer::time_point setup_end = Timer::now();
    std::cout
        << "Setup time: "
        << std::chrono::duration_cast<Duration>(setup_end - setup_beg).count()
        << std::endl;
    Timer::time_point vec_beg = Timer::now();
    std::vector<segment_t> segments0(num_diff_hash), segments1(num_diff_hash);
    std::vector<DataType> buffer0(num_diff_hash * elemPerChunk),
        buffer1(num_diff_hash * elemPerChunk);
    Timer::time_point vec_end = Timer::now();
    std::cout << "Vector allocation time: "
              << std::chrono::duration_cast<Duration>(vec_end - vec_beg).count()
              << std::endl;

    double err_tol = client_info.error_tolerance;
    size_t d_size = client_info.data_size;
    AbsoluteComp<DataType> abs_comp;
    if (num_diff_hash > 0) {

        Timer::time_point total_beg = Timer::now();
        Timer::time_point read_beg = Timer::now();

        Kokkos::deep_copy(diff_hash_vec.vector_h, diff_hash_vec.vector_d);
        Kokkos::parallel_for(
            "Fill segment vectors",
            Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(
                0, num_diff_hash),
            [&](size_t i) {
                // segments0[i].id = i;
                segments0[i].buffer =
                    (uint8_t *)(buffer0.data()) + i * client_info.chunk_size;
                segments0[i].offset =
                    diff_hash_vec.vector_h(i) * client_info.chunk_size;
                segments0[i].size = client_info.chunk_size;

                // segments1[i].id = i;
                segments1[i].buffer =
                    (uint8_t *)(buffer1.data()) + i * client_info.chunk_size;
                segments1[i].offset =
                    diff_hash_vec.vector_h(i) * client_info.chunk_size;
                segments1[i].size = client_info.chunk_size;
            });
        Kokkos::fence();
        Timer::time_point seg_end = Timer::now();
        std::cout
            << "Segment preparation time: "
            << std::chrono::duration_cast<Duration>(seg_end - read_beg).count()
            << std::endl;

        double enq_time = 0, wait_time = 0;
        Timer::time_point enq_beg = Timer::now();
        prev.io_reader.enqueue_reads(segments0);
        Timer::time_point enq_end = Timer::now();
        enq_time +=
            std::chrono::duration_cast<Duration>(enq_end - enq_beg).count();

        Timer::time_point wait_beg = Timer::now();
        prev.io_reader.wait_all();
        Timer::time_point wait_end = Timer::now();
        wait_time +=
            std::chrono::duration_cast<Duration>(wait_end - wait_beg).count();

        enq_beg = Timer::now();
        io_reader.enqueue_reads(segments1);
        enq_end = Timer::now();
        enq_time +=
            std::chrono::duration_cast<Duration>(enq_end - enq_beg).count();

        wait_beg = Timer::now();
        io_reader.wait_all();
        wait_end = Timer::now();
        wait_time +=
            std::chrono::duration_cast<Duration>(wait_end - wait_beg).count();

        Timer::time_point read_end = Timer::now();
        read_timer +=
            std::chrono::duration_cast<Duration>(read_end - read_beg).count();

        Kokkos::Profiling::pushRegion(
            diff_label + std::string("Compare Tree direct comparison"));
        Timer::time_point comp_beg = Timer::now();
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
                if (data_idx < d_size) {
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
        Timer::time_point comp_end = Timer::now();
        compare_timer +=
            std::chrono::duration_cast<Duration>(comp_end - comp_beg).count();

        Timer::time_point total_end = Timer::now();

        printf("2nd Phase: Enqueue reads: %2.10f\n", enq_time);
        printf("2nd Phase: Wait time:     %2.10f\n", wait_time);
        printf(
            "2nd Phase: Compare time:  %2.10f\n",
            std::chrono::duration_cast<Duration>(comp_end - comp_beg).count());
        printf("2nd Phase: Total time:    %2.10f\n",
               std::chrono::duration_cast<Duration>(total_end - total_beg)
                   .count());
    }

    STDOUT_PRINT("Number of changed elements - Phase Two: %lu\n", nchange);
    Kokkos::Profiling::popRegion();
    return nchange;
}

template <typename DataType, typename Reader>
size_t
client_t<DataType, Reader>::get_num_hash_comparisons() const {
    auto num_hash_comp_h = Kokkos::create_mirror_view(num_hash_comp);
    Kokkos::deep_copy(num_hash_comp_h, num_hash_comp);
    return num_hash_comp_h(0);
}

template <typename DataType, typename Reader>
size_t
client_t<DataType, Reader>::get_num_comparisons() const {
    auto num_comparisons_h = Kokkos::create_mirror_view(num_comparisons);
    Kokkos::deep_copy(num_comparisons_h, num_comparisons);
    return num_comparisons_h(0);
}

template <typename DataType, typename Reader>
size_t
client_t<DataType, Reader>::get_num_changes() const {
    return nchange;
}

template <typename DataType, typename Reader>
std::vector<double>
client_t<DataType, Reader>::get_create_time() const {
    const double *timers = tree.get_timers();
    return {timers[0], timers[1], timers[2], timers[3]};
}

template <typename DataType, typename Reader>
double
client_t<DataType, Reader>::get_tree_comparison_time() const {
    return timers[0];
}

template <typename DataType, typename Reader>
double
client_t<DataType, Reader>::get_compare_time() const {
    return timers[1];
}

template <typename DataType, typename Reader>
client_info_t
client_t<DataType, Reader>::get_client_info() const {
    return client_info;
}

}   // namespace state_diff

#endif   // __STATE_DIFF_HPP
