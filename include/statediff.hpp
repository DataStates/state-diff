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
    static const size_t DEFAULT_DEVICE_CACHE = 1ULL * GB;
    static const TransferType DEFAULT_CACHE_TIER = TransferType::FileToHost;

    // client variables
    client_info_t client_info;
    tree_t tree;
    data_loader_t data_loader;
    int curr_chkpt_id = -1;

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

    // timers (setup, compare_tree, compare_direct)
    double timers[3];

    void initialize(size_t n_chunks);

  public:
    client_t(int client_id, size_t host_cache_size = DEFAULT_HOST_CACHE,
             size_t dev_cache_size = DEFAULT_DEVICE_CACHE);
    client_t(int client_id, size_t data_size, double error,
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

    bool compare_with(int chkpt_id, Reader &curr_reader, client_t &prev,
                      Reader &prev_reader,
                      std::optional<TransferType> cache_tier = std::nullopt);

    // Internal implementations
    size_t compare_trees(const client_t &prev, Queue &working_queue,
                         Vector<size_t> &diff_hash_vec,
                         Kokkos::View<uint64_t[1]> &num_hash_comp);
    size_t compare_data(client_t &prev, int ld_prev, int ld_curr,
                        Vector<size_t> &diff_hash_vec,
                        Kokkos::Bitset<> &changed_chunks,
                        Kokkos::View<uint64_t[1]> &num_changed,
                        Kokkos::View<uint64_t[1]> &num_comparisons,
                        TransferType cache_tier = DEFAULT_CACHE_TIER);

    // Stats getters
    size_t get_num_hash_comparisons() const;
    size_t get_num_comparisons() const;
    size_t get_num_changes() const;
    double get_tree_comparison_time() const;
    double get_data_compare_time() const;
    std::vector<double> get_create_time() const;
    std::vector<double> get_compare_time() const;
    client_info_t get_client_info() const;
};

template <typename DataType, typename Reader>
client_t<DataType, Reader>::client_t(int client_id, size_t host_cache_size,
                                     size_t dev_cache_size)
    : data_loader(host_cache_size, dev_cache_size) {}

template <typename DataType, typename Reader>
client_t<DataType, Reader>::client_t(int client_id, size_t data_size,
                                     double error, char dtype,
                                     size_t min_chunk_size, size_t start,
                                     bool fuzzyhash, size_t host_cache_size,
                                     size_t dev_cache_size)
    : data_loader(host_cache_size, dev_cache_size) {
    TIMER_START(client_init);
    DBG("Begin client setup");
    std::string setup_region_name = std::string("StateDiff:: Checkpoint ") +
                                    std::to_string(client_id) +
                                    std::string(": Setup");
    Kokkos::Profiling::pushRegion(setup_region_name.c_str());
    // size_t optim_chksize = data_loader.get_chunksize(data_size);
    size_t optim_chksize = min_chunk_size;
    client_info = client_info_t{client_id,      dtype, data_size,
                                min_chunk_size, start, error};
    tree = tree_t(data_size, optim_chksize, fuzzyhash);

    size_t n_chunks = data_size / optim_chksize;
    if (n_chunks * optim_chksize < data_size)
        n_chunks += 1;

    initialize(n_chunks);

    Kokkos::Profiling::popRegion();
    TIMER_STOP(client_init,
               "State-diff client " << client_id << " initialized");
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
    timers[2] = 0;
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
    TIMER_START(client_create_tree);
    uint8_t *data_ptr = reinterpret_cast<uint8_t *>(data.data());
    tree.create(data_ptr, client_info);
    TIMER_STOP(client_create_tree,
               "State-diff tree " << curr_chkpt_id << " created from vector");
    curr_chkpt_id++;
}

template <typename DataType, typename Reader>
void
client_t<DataType, Reader>::create(uint8_t *data_ptr) {
    TIMER_START(client_create_tree);
    tree.create(data_ptr, client_info);
    TIMER_STOP(client_create_tree,
               "State-diff tree " << curr_chkpt_id << " created from pointer");
    curr_chkpt_id++;
}

template <typename DataType, typename Reader>
void
client_t<DataType, Reader>::create(Reader &reader,
                                   std::optional<TransferType> cache_tier) {
    TIMER_START(client_create_tree);
    TransferType create_tree_tier = cache_tier.value_or(DEFAULT_CACHE_TIER);
    int ld = data_loader.file_load(reader, 0, client_info.chunk_size, 0,
                                   create_tree_tier);
    tree.create(client_info, data_loader, ld, create_tree_tier);
    TIMER_STOP(client_create_tree,
               "State-diff tree " << curr_chkpt_id << " created from reader");
    curr_chkpt_id++;
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
client_t<DataType, Reader>::compare_with(
    int chkpt_id, Reader &curr_reader, client_t &prev, Reader &prev_reader,
    std::optional<TransferType> cache_tier) {
    // ASSERT(client_info == prev.client_info && curr_chkpt_id == chkpt_id);
    TIMER_START(client_compare_with);
    ASSERT(client_info == prev.client_info ||
           "Comparing two clients with different metadata characteristics.");
    ASSERT(curr_chkpt_id == chkpt_id ||
           "Comparing two checkpoints with different IDs.");

    auto ndifferent =
        compare_trees(prev, working_queue, diff_hash_vec, num_hash_comp);

    // Validate hash mismatches with direct comparison
    DBG("Number of different hashes after phase 1: " << diff_hash_vec.size());
    if (diff_hash_vec.size() > 0) {
        TransferType compare_data_tier =
            cache_tier.value_or(DEFAULT_CACHE_TIER);
        int ld_prev = data_loader.file_load(
            prev_reader, 0, client_info.chunk_size, 0, compare_data_tier);
        int ld_curr = data_loader.file_load(
            curr_reader, 0, client_info.chunk_size, 0, compare_data_tier);
        auto ndifferent_direct =
            compare_data(prev, ld_prev, ld_curr, diff_hash_vec, changed_chunks,
                         num_changed, num_comparisons, compare_data_tier);
        DBG("Number of different hashes after phase 2: " << ndifferent_direct);
    }
    TIMER_STOP(client_compare_with, "State-diff tree and data for chkpt "
                                        << curr_chkpt_id << " compared");
    return get_num_changes() == 0;
}

template <typename DataType, typename Reader>
size_t
client_t<DataType, Reader>::compare_trees(
    const client_t &prev, Queue &working_queue, Vector<size_t> &diff_hash_vec,
    Kokkos::View<uint64_t[1]> &num_hash_comp) {

    Timer::time_point setup_beg = Timer::now();
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
    Timer::time_point setup_end = Timer::now();
    timers[0] +=
        std::chrono::duration_cast<Duration>(setup_end - setup_beg).count();

    Timer::time_point compare_beg = Timer::now();
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
    Timer::time_point compare_end = Timer::now();
    timers[1] +=
        std::chrono::duration_cast<Duration>(compare_end - compare_beg).count();
    return diff_hash_vec.size();
}

template <typename DataType, typename Reader>
size_t
client_t<DataType, Reader>::compare_data(
    client_t &prev, int ld_prev, int ld_curr, Vector<size_t> &diff_hash_vec,
    Kokkos::Bitset<> &changed_chunks, Kokkos::View<uint64_t[1]> &num_changed,
    Kokkos::View<uint64_t[1]> &num_comparisons, TransferType cache_tier) {
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
    timers[0] +=
        std::chrono::duration_cast<Duration>(setup_end - setup_beg).count();

    Timer::time_point compare_beg = Timer::now();
    double err_tol = client_info.error_tolerance;
    size_t chunk_size = client_info.chunk_size;
    AbsoluteComp<DataType> abs_comp;
    size_t work_start = 0;
    while (work_start < num_diff_hash) {
        auto prev_batch = data_loader.next(ld_prev, cache_tier);
        auto curr_batch = data_loader.next(ld_curr, cache_tier);
        DataType *prev_ptr = (DataType *)prev_batch.first;
        DataType *curr_ptr = (DataType *)curr_batch.first;
        size_t ready_size = prev_batch.second;
        size_t curr_n_chunks = ready_size / chunk_size;
        if (curr_n_chunks * chunk_size < ready_size)
            curr_n_chunks += 1;
        work_start += curr_n_chunks;

        Kokkos::Profiling::pushRegion(
            diff_label + std::string("Compare Tree direct comparison"));
        uint64_t ndiff = 0;
        // Parallel comparison
        using PolicyType =
            Kokkos::RangePolicy<size_t, Kokkos::DefaultHostExecutionSpace>;
        auto range_policy = PolicyType(0, curr_n_chunks * elemPerChunk);
        Kokkos::parallel_reduce(
            "Count differences", range_policy,
            KOKKOS_LAMBDA(const size_t idx, uint64_t &update) {
                if (!abs_comp(prev_ptr[idx], curr_ptr[idx], err_tol)) {
                    update += 1;
                }
            },
            Kokkos::Sum<uint64_t>(ndiff));
        Kokkos::fence();
        nchange += ndiff;
        Kokkos::Profiling::popRegion();
    }
    Timer::time_point compare_end = Timer::now();
    timers[1] +=
        std::chrono::duration_cast<Duration>(compare_end - compare_beg).count();
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
    // setup, leaves, rest of tree (their sum gives the total creation time)
    return {timers[0], timers[1], timers[2]};
}

template <typename DataType, typename Reader>
std::vector<double>
client_t<DataType, Reader>::get_compare_time() const {
    // setup, compare_tree, compare_direct (their sum gives the total creation time)
    return {timers[0], timers[1], timers[2]};
}

template <typename DataType, typename Reader>
double
client_t<DataType, Reader>::get_tree_comparison_time() const {
    return timers[1];
}

template <typename DataType, typename Reader>
double
client_t<DataType, Reader>::get_data_compare_time() const {
    return timers[2];
}

template <typename DataType, typename Reader>
client_info_t
client_t<DataType, Reader>::get_client_info() const {
    return client_info;
}

}   // namespace state_diff

#endif   // __STATE_DIFF_HPP
