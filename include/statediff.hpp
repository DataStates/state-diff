#ifndef __STATE_DIFF_HPP
#define __STATE_DIFF_HPP

#include "Kokkos_Bitset.hpp"
#include "Kokkos_Core.hpp"
#include "Kokkos_ScatterView.hpp"
#include "Kokkos_Sort.hpp"
#include "compare_utils.hpp"
#include "data_loader.hpp"
#include "debug.hpp"
#include "io_reader.hpp"
#include "merkle_tree.hpp"
#include "statediff_bitset.hpp"
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

template <typename DataType> class client_t {

    // Defaults
    static const char DEFAULT_DTYPE = 'f';
    static const bool DEFAULT_FUZZY_HASH = true;
    static const size_t DEFAULT_START_LEVEL = 13;
    static const size_t DEFAULT_CHUNK_SIZE = 4 * KB;
    static const size_t DEFAULT_HOST_CACHE = 2ULL * GB;
    static const size_t DEFAULT_DEVICE_CACHE = 1ULL * GB;
    static const size_t DEFAULT_CREATE_READ_SIZE = 128 * MB;
    static const TransferType DEFAULT_CACHE_TIER = TransferType::FileToHost;

    // client variables
    client_info_t client_info;
    tree_t tree;
    data_loader_t data_loader;
    int curr_chkpt_id = -1;

    // comparison state
    Queue working_queue;
    // Bitset for tracking which chunks have been changed
    Kokkos::Bitset<> changed_chunks;   // host
    // Vec of idx of chunks that are marked different during the 1st phase
    Vector<size_t> diff_hash_vec;
    Kokkos::View<size_t[1]> num_comparisons =
        Kokkos::View<size_t[1]>("Num comparisons");
    Kokkos::View<size_t[1]> num_changed =
        Kokkos::View<size_t[1]>("Num changed");
    Kokkos::View<size_t[1]> num_hash_comp =
        Kokkos::View<size_t[1]>("Num hash comparisons");
    size_t nchange = 0;

    // timers (setup, compare_tree, compare_direct, load_direct,
    // elementwise_compare)
    double timers[5];

    void initialize(size_t n_chunks);

  public:
    client_t() {};
    client_t(int client_id, size_t data_size, double error,
             char dtype = DEFAULT_DTYPE, size_t chunk_size = DEFAULT_CHUNK_SIZE,
             size_t start_level = DEFAULT_START_LEVEL,
             bool fuzzyhash = DEFAULT_FUZZY_HASH,
             size_t host_cache_size = DEFAULT_HOST_CACHE,
             size_t dev_cache_size = DEFAULT_DEVICE_CACHE);
    ~client_t();

    void create(std::vector<DataType> &data);
    void create(uint8_t *data_ptr);

    template <typename Reader>
    void create(Reader &reader, size_t read_blk_size = DEFAULT_CREATE_READ_SIZE,
                TransferType create_tier = DEFAULT_CACHE_TIER);
    template <class Archive>
    void save(Archive &ar, const unsigned int version) const;
    template <class Archive> void load(Archive &ar, const unsigned int version);

    template <typename Reader>
    bool compare_with(int chkpt_id, Reader &curr_reader, client_t &prev,
                      Reader &prev_reader, uint32_t offt_gap = 0,
                      TransferType compare_tier = DEFAULT_CACHE_TIER);

    // Internal implementations
    size_t compare_trees(const client_t &prev, Queue &working_queue,
                         Vector<size_t> &diff_hash_vec,
                         Kokkos::View<size_t[1]> &num_hash_comp);
    size_t compare_data(client_t &prev, int ld_id,
                        Vector<size_t> &diff_hash_vec,
                        Kokkos::Bitset<> &changed_chunks,
                        Kokkos::View<size_t[1]> &num_changed,
                        Kokkos::View<size_t[1]> &num_comparisons,
                        TransferType cache_tier);

    // Stats getters
    size_t get_num_hash_comparisons() const;
    size_t get_num_comparisons() const;
    size_t get_num_changes() const;
    size_t get_filtered_blocks() const;
    size_t get_validated_diffs() const;
    double get_tree_comparison_time() const;
    double get_data_compare_time() const;
    std::vector<double> get_create_time() const;
    std::vector<double> get_compare_time() const;
    std::vector<double> get_direct_compare_time() const;
    client_info_t get_client_info() const;
};

template <typename DataType>
client_t<DataType>::client_t(int client_id, size_t data_size, double error,
                             char dtype, size_t chunk_size, size_t start,
                             bool fuzzyhash, size_t host_cache_size,
                             size_t dev_cache_size)
    : data_loader(host_cache_size, dev_cache_size) {
    TIMER_START(client_init);
    DBG("Begin client setup");
    std::string setup_region_name = std::string("StateDiff:: Checkpoint ") +
                                    std::to_string(client_id) +
                                    std::string(": Setup");
    Kokkos::Profiling::pushRegion(setup_region_name.c_str());
    client_info =
        client_info_t{client_id, dtype, data_size, chunk_size, start, error};

    size_t n_chunks = data_size / chunk_size;
    if (n_chunks * chunk_size < data_size)
        n_chunks += 1;

    tree = tree_t(n_chunks, chunk_size, fuzzyhash);
    initialize(n_chunks);

    Kokkos::Profiling::popRegion();
    TIMER_STOP(client_init,
               "State-diff client " << client_id << " initialized");
    DBG("Finished client setup");
}

template <typename DataType>
void
client_t<DataType>::initialize(size_t n_chunks) {
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

template <typename DataType> client_t<DataType>::~client_t() {}

/**
 * Create the corresponding tree
 */
template <typename DataType>
void
client_t<DataType>::create(std::vector<DataType> &data) {
    TIMER_START(client_create_tree);
    uint8_t *data_ptr = reinterpret_cast<uint8_t *>(data.data());
    tree.create(data_ptr, client_info);
    TIMER_STOP(client_create_tree,
               "State-diff tree " << curr_chkpt_id << " created from vector");
    curr_chkpt_id++;
}

template <typename DataType>
void
client_t<DataType>::create(uint8_t *data_ptr) {
    TIMER_START(client_create_tree);
    tree.create(data_ptr, client_info);
    TIMER_STOP(client_create_tree,
               "State-diff tree " << curr_chkpt_id << " created from pointer");
    curr_chkpt_id++;
}

template <typename DataType>
template <typename Reader>
void
client_t<DataType>::create(Reader &reader, size_t read_blk_size,
                           TransferType create_tree_tier) {
    TIMER_START(client_create_tree);
    int ld = data_loader.file_load(reader, read_blk_size, create_tree_tier);
    tree.create(client_info, data_loader, ld, create_tree_tier);
    TIMER_STOP(client_create_tree,
               "State-diff tree " << curr_chkpt_id << " created from reader");
    curr_chkpt_id++;
}

/**
 * Serialize a client into an archive
 */
template <typename DataType>
template <class Archive>
void
client_t<DataType>::save(Archive &ar, const unsigned int version) const {
    ar(client_info);
    ar(tree);
}

/**
 * Serialize a client into an archive
 */
template <typename DataType>
template <class Archive>
void
client_t<DataType>::load(Archive &ar, const unsigned int version) {
    ar(client_info);
    ar(tree);
    initialize(tree.num_leaves);
}

template <typename DataType>
template <typename Reader>
bool
client_t<DataType>::compare_with(int chkpt_id, Reader &curr_reader,
                                 client_t &prev, Reader &prev_reader,
                                 uint32_t offt_gap, TransferType compare_tier) {
    TIMER_START(client_compare_with);
    ASSERT(client_info == prev.client_info ||
           "Comparing two clients with different metadata characteristics.");
    ASSERT(curr_chkpt_id == chkpt_id ||
           "Comparing two checkpoints with different IDs.");

    compare_trees(prev, working_queue, diff_hash_vec, num_hash_comp);

    // Validate hash mismatches with direct comparison
    DBG("Number of different hashes after phase 1: " << diff_hash_vec.size());
    if (diff_hash_vec.size() > 0) {
        Timer::time_point setup_beg = Timer::now();

        // Sort indices for coalescing and better performance
        std::string diff_label = std::string("Chkpt ") +
                                 std::to_string(client_info.id) +
                                 std::string(": ");
        Kokkos::Profiling::pushRegion(diff_label +
                                      std::string("Compare Tree sort indices"));
        Kokkos::sort(diff_hash_vec.vector_d, 0,
                     static_cast<size_t>(diff_hash_vec.size()));
        std::vector<size_t> diff_offsets(
            diff_hash_vec.data(), diff_hash_vec.data() + diff_hash_vec.size());
        Kokkos::Profiling::popRegion();

        int ld_id = data_loader.file_load(prev_reader, curr_reader,
                                          diff_offsets, client_info.chunk_size,
                                          compare_tier, offt_gap);
        Timer::time_point setup_end = Timer::now();
        double setup_time =
            std::chrono::duration_cast<Duration>(setup_end - setup_beg).count();
        timers[0] += setup_time;   // total setup time
        timers[2] += setup_time;   // total compare time
        compare_data(prev, ld_id, diff_hash_vec, changed_chunks, num_changed,
                     num_comparisons, compare_tier);
        DBG("Number of different hashes after phase 2: " << nchange);
    }
    TIMER_STOP(client_compare_with, "State-diff tree and data for chkpt "
                                        << curr_chkpt_id << " compared");
    return get_num_changes() == 0;
}

template <typename DataType>
size_t
client_t<DataType>::compare_trees(const client_t &prev, Queue &working_queue,
                                  Vector<size_t> &diff_hash_vec,
                                  Kokkos::View<size_t[1]> &num_hash_comp) {

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
    Kokkos::Experimental::ScatterView<size_t[1]> nhash_comp(num_hash_comp);
    Kokkos::Profiling::popRegion();
    Timer::time_point setup_end = Timer::now();
    double setup_time =
        std::chrono::duration_cast<Duration>(setup_end - setup_beg).count();
    timers[0] += setup_time;

    Timer::time_point compare_beg = Timer::now();
    // Fills up queue with nodes in the stop level
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
                bool identical = digests_same(tree_curr[node], tree_prev[node]);
                nhash_comp_access(0) += 1;
                if (!identical) {
                    if ((n_chunks - 1 <= node) && (node < n_nodes)) {
                        if (node < left_leaf) {
                            // Leaf is not on the last level
                            size_t entry = (n_nodes - left_leaf) +
                                           (node - ((n_nodes - 1) / 2));
                            ASSERT(entry < n_chunks);
                            diff_hashes.push(entry);
                        } else {
                            // Leaf is on the last level
                            ASSERT(node - left_leaf < n_chunks);
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
        setup_time +
        std::chrono::duration_cast<Duration>(compare_end - compare_beg).count();
    return diff_hash_vec.size();
}

template <typename DataType>
size_t
client_t<DataType>::compare_data(client_t &prev, int ld_id,
                                 Vector<size_t> &diff_hash_vec,
                                 Kokkos::Bitset<> &changed_chunks,
                                 Kokkos::View<size_t[1]> &num_changed,
                                 Kokkos::View<size_t[1]> &num_comparisons,
                                 TransferType compare_tier) {
    std::string diff_label = std::string("Chkpt ") +
                             std::to_string(client_info.id) + std::string(": ");
    Kokkos::Profiling::pushRegion(
        diff_label + std::string("Compare Trees direct comparison"));

    size_t num_diff_hash = static_cast<size_t>(diff_hash_vec.size());
    size_t elemPerChunk = client_info.chunk_size / sizeof(DataType);
    DBG("Phase 1 filtered hashes: " << num_diff_hash
                                    << "; Element per chunk: " << elemPerChunk);
    Timer::time_point compare_beg = Timer::now();
    double err_tol = client_info.error_tolerance;
    AbsoluteComp<DataType> abs_comp;
    Kokkos::Experimental::ScatterView<size_t[1]> num_comp(num_comparisons);
    auto &changed_blocks = changed_chunks;
    int n_files = 2;
    size_t work_done = 0;
    DataType *prev_ptr = NULL, *curr_ptr = NULL;
    while (work_done < num_diff_hash) {
        Timer::time_point read_beg = Timer::now();
        next_batch_t front_batch = data_loader.next(ld_id, compare_tier);
        prev_ptr = reinterpret_cast<DataType *>(front_batch.ptr);
        size_t ready_size = front_batch.size / n_files;
        curr_ptr = prev_ptr + ready_size / sizeof(DataType);
        size_t proc_offt = front_batch.offt_count;
        DBG("Work Done : " << work_done << "; Offsets to process: " << proc_offt
                           << "; offsets read: "
                           << ready_size / client_info.chunk_size);
        Timer::time_point read_end = Timer::now();
        timers[3] +=
            std::chrono::duration_cast<Duration>(read_end - read_beg).count();
        Timer::time_point cmp_beg = Timer::now();

        // Using (ready_size / chunk_size) would serve all data for comparison
        // even those that do not correspond to offsets in diff_hash_vec. We
        // adjust that by corelating with diff_hash_vec to only compare chunks
        // that need to be validated.
        auto subview_bounds =
            Kokkos::make_pair(work_done, work_done + proc_offt);
        auto diff_hash_subview =
            Kokkos::subview(diff_hash_vec.vector_d, subview_bounds);

        Kokkos::Profiling::pushRegion(
            diff_label + std::string("Statediff direct compare iter"));
        size_t ndiff = 0;
        // Parallel comparison
        auto range_policy =
            Kokkos::RangePolicy<size_t>(0, proc_offt * elemPerChunk);
        Kokkos::parallel_reduce(
            "Count differences", range_policy,
            KOKKOS_LAMBDA(const size_t idx, size_t &update) {
                auto ncomp_access = num_comp.access();
                size_t blk_idx = idx / elemPerChunk;   // Block idx
                size_t gap = diff_hash_subview[blk_idx] - diff_hash_subview[0];
                size_t blk_start = gap * elemPerChunk;   // Block start
                size_t elm_idx =
                    blk_start + (idx % elemPerChunk);   // Element in block

                if (!abs_comp(prev_ptr[elm_idx], curr_ptr[elm_idx], err_tol)) {
                    update += 1;
                    changed_blocks.set(diff_hash_subview[blk_idx]);
                }
                ncomp_access(0) += 1;
            },
            Kokkos::Sum<size_t>(ndiff));
        nchange += ndiff;
        work_done += proc_offt;
        Timer::time_point cmp_end = Timer::now();
        timers[4] +=
            std::chrono::duration_cast<Duration>(cmp_end - cmp_beg).count();
    }
    Timer::time_point compare_end = Timer::now();
    Kokkos::Experimental::contribute(num_comparisons, num_comp);
    timers[2] +=
        std::chrono::duration_cast<Duration>(compare_end - compare_beg).count();
    Kokkos::Profiling::popRegion();
    return nchange;
}

template <typename DataType>
size_t
client_t<DataType>::get_num_hash_comparisons() const {
    auto num_hash_comp_h = Kokkos::create_mirror_view(num_hash_comp);
    Kokkos::deep_copy(num_hash_comp_h, num_hash_comp);
    return num_hash_comp_h(0);
}

template <typename DataType>
size_t
client_t<DataType>::get_num_comparisons() const {
    auto num_comparisons_h = Kokkos::create_mirror_view(num_comparisons);
    Kokkos::deep_copy(num_comparisons_h, num_comparisons);
    return num_comparisons_h(0);
}

template <typename DataType>
size_t
client_t<DataType>::get_num_changes() const {
    return nchange;
}

template <typename DataType>
size_t
client_t<DataType>::get_filtered_blocks() const {
    return diff_hash_vec.size();
}

template <typename DataType>
size_t
client_t<DataType>::get_validated_diffs() const {
    return changed_chunks.count();
}

template <typename DataType>
std::vector<double>
client_t<DataType>::get_create_time() const {
    const double *create_timers = tree.get_timers();
    // setup, leaves, rest of tree (their sum gives the total creation time),
    // load, hash
    return {create_timers[0], create_timers[1], create_timers[2],
            create_timers[3], create_timers[4]};
}

template <typename DataType>
std::vector<double>
client_t<DataType>::get_compare_time() const {
    // setup, compare_tree, compare_direct (their sum gives the total creation
    // time)
    return {timers[0], timers[1], timers[2]};
}

template <typename DataType>
std::vector<double>
client_t<DataType>::get_direct_compare_time() const {
    // load time and comparison time during direct comparison
    return {timers[3], timers[4]};
}

template <typename DataType>
double
client_t<DataType>::get_tree_comparison_time() const {
    return timers[1];
}

template <typename DataType>
double
client_t<DataType>::get_data_compare_time() const {
    return timers[2];
}

template <typename DataType>
client_info_t
client_t<DataType>::get_client_info() const {
    return client_info;
}

}   // namespace state_diff

#endif   // __STATE_DIFF_HPP
