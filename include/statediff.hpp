#ifndef __STATE_DIFF_HPP
#define __STATE_DIFF_HPP

#include "Kokkos_Bitset.hpp"
#include "Kokkos_Core.hpp"
#include "Kokkos_ScatterView.hpp"
#include "Kokkos_Sort.hpp"
#include "compare_utils.hpp"
#include "merkle_tree.hpp"
#include "statediff_bitset.hpp"
#include <scissor/io_reader.hpp>
#include <scissor/loader.hpp>
#include <scissor/batch.hpp>
#include <stdexcept>
#include <scissor/types.hpp>
#include <statediff/log.hpp>
#include <cassert>
#include <memory>
#include <stdexcept>
#include <chrono>
#include <climits>
#include <cstddef>
#include <functional>
#include <iostream>
#include <vector>

#define KB 1024
#define MB (1024 * KB)
#define GB (1024ULL * MB)

// Bring scissor enums into global scope so existing code that uses
// TransferType::FileToHost, FileSrc::PFS, etc. continues to compile.
using scissor::TransferType;
using scissor::FileSrc;

namespace state_diff {

// How the direct comparison picks its (offset gap, block size):
//   Model -- the scissor Regressor selects them (Kokkos::num_threads() cores);
//            the gap/block_size arguments are ignored.
//   Fixed -- the given gap/block_size are used as-is (gap 0 = no coalescing),
//            loader runs single-threaded (nthreads = 1), for profiling.
enum class Tuning { Model, Fixed };

template <typename DataType> class client_t {

    // Defaults
    static const char DEFAULT_DTYPE = 'f';
    static const bool DEFAULT_FUZZY_HASH = true;
    static const size_t DEFAULT_START_LEVEL = 13;
    static const size_t DEFAULT_CHUNK_SIZE = 4 * KB;
    static const size_t DEFAULT_CREATE_READ_SIZE = 128 * MB;
    static const size_t DEFAULT_COMPARE_READ_SIZE = 4 * MB;
    static const TransferType DEFAULT_CACHE_TIER = TransferType::FileToHost;
    static const FileSrc DEFAULT_FILE_SRC = FileSrc::PFS;

    // client variables
    client_info_t client_info;
    tree_t tree;
    // The reader is owned by the client's creator (one per process) and
    // shared by all clients. The loader is built from it once.
    scissor::IReader *reader_ = nullptr;
    std::unique_ptr<scissor::Loader> data_loader;   // empty without a reader
    int curr_chkpt_id = -1;
    bool is_initialized_ = false;

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
    size_t wasted_bytes = 0;
    size_t IOP_count = 0;
    int best_gap = 0;
    size_t best_batch_size = 0;

    // timers (setup, compare_tree, compare_direct, load_direct,
    // elementwise_compare, wait_direct)
    double timers[6];

    void resize(size_t n_chunks);
    scissor::Loader &loader();

  public:
    client_t() = default;  // required by cereal deserialization
    ~client_t();

    // The reader is used for all file I/O of this client (tree creation from
    // file and direct comparison). It must outlive the client.
    client_t(int client_id, scissor::IReader &reader, size_t data_size,
             double error, char dtype = DEFAULT_DTYPE,
             size_t chunk_size = DEFAULT_CHUNK_SIZE,
             size_t start_level = DEFAULT_START_LEVEL,
             bool fuzzyhash = DEFAULT_FUZZY_HASH);

    void create(std::vector<DataType> &data);
    void create(uint8_t *data_ptr);

    void create(const std::string& fname, size_t read_blk_size = DEFAULT_CREATE_READ_SIZE,
                TransferType create_tier = DEFAULT_CACHE_TIER);
    template <class Archive>
    void save(Archive &ar, const unsigned int version) const;
    template <class Archive> void load(Archive &ar, const unsigned int version);

    bool compare_with(int chkpt_id, const std::string& curr_fname, client_t &prev,
                      const std::string& prev_fname, uint32_t offt_gap = 0,
                      size_t block_size = DEFAULT_COMPARE_READ_SIZE,
                      bool ideal_compare = false, bool exec_compare = true,
                      TransferType compare_tier = DEFAULT_CACHE_TIER,
                      FileSrc file_src_loc = DEFAULT_FILE_SRC,
                      Tuning tuning = Tuning::Model);

    // Internal implementations
    size_t compare_trees(const client_t &prev, Queue &working_queue,
                         Vector<size_t> &diff_hash_vec,
                         Kokkos::View<size_t[1]> &num_hash_comp);
    size_t compare_data(client_t &prev, const std::string& prev_fname, const std::string& curr_fname,
                        std::vector<size_t> &diff_offsets,
                        Kokkos::Bitset<> &changed_chunks,
                        Kokkos::View<size_t[1]> &num_changed,
                        Kokkos::View<size_t[1]> &num_comparisons,
                        uint32_t offt_gap, size_t block_size,
                        TransferType cache_tier, FileSrc file_src_loc, bool exec_compare,
                        Tuning tuning);
    size_t compare_data_ideal_compute(client_t &prev,
                                      Vector<size_t> &diff_hash_vec,
                                      Kokkos::Bitset<> &changed_chunks,
                                      Kokkos::View<size_t[1]> &num_changed,
                                      Kokkos::View<size_t[1]> &num_comparisons,
                                      TransferType compare_tier,
                                      const std::string& prev_fname, const std::string& curr_fname);
    // Stats getters
    size_t get_num_hash_comparisons() const;
    size_t get_num_comparisons() const;
    size_t get_num_changes() const;
    size_t get_filtered_blocks() const;
    size_t get_validated_diffs() const;
    size_t get_wastedbytes_count() const;
    size_t get_IOP_count() const;
    double get_tree_comparison_time() const;
    double get_data_compare_time() const;
    std::vector<double> get_create_time() const;
    std::vector<double> get_compare_time() const;
    std::vector<double> get_direct_compare_time() const;
    client_info_t get_client_info() const;
    std::pair<int, size_t> get_loader_info() const;
};

template <typename DataType>
client_t<DataType>::client_t(int client_id, scissor::IReader &reader,
                              size_t data_size, double error, char dtype,
                              size_t chunk_size, size_t start, bool fuzzyhash) {
    STATEDIFF_TIMER_START(client_init);
    std::string setup_region_name = std::string("StateDiff:: Checkpoint ") +
                                    std::to_string(client_id) +
                                    std::string(": Setup");
    Kokkos::Profiling::pushRegion(setup_region_name.c_str());
    client_info = client_info_t{client_id, dtype, data_size,
                                chunk_size, start, error};

    size_t n_chunks = data_size / chunk_size;
    if (n_chunks * chunk_size < data_size)
        n_chunks += 1;
    // The file reader is created without specific file assignment
    // i.e., the reader can load data from any file
    reader_ = &reader;
    data_loader = std::make_unique<scissor::Loader>(reader);
    tree = tree_t(n_chunks, chunk_size, fuzzyhash);
    resize(n_chunks);
    Kokkos::Profiling::popRegion();
    STATEDIFF_TIMER_PERF(client_init, "State-diff client initialized",
                         "client_id", client_id);
    STATEDIFF_DEBUG("Finished client setup");
}

template <typename DataType>
void
client_t<DataType>::resize(size_t n_chunks) {
    working_queue = Queue(n_chunks);
    changed_chunks = Kokkos::Bitset<>(n_chunks);
    changed_chunks.reset();
    Kokkos::resize(diff_hash_vec.vector_d, n_chunks);
    Kokkos::resize(diff_hash_vec.vector_h, n_chunks);

    // Clear stats
    // timers[0] = 0;
    // timers[1] = 0;
    // timers[2] = 0;
    std::fill(std::begin(timers), std::end(timers), 0.0);
    diff_hash_vec.clear();
    Kokkos::deep_copy(num_comparisons, 0);
    Kokkos::deep_copy(num_hash_comp, 0);
    Kokkos::deep_copy(num_changed, 0);
    nchange = 0;
    wasted_bytes = 0;
    IOP_count = 0;
    best_gap = 0;
    best_batch_size = 0;
    is_initialized_ = true;
}

template <typename DataType> client_t<DataType>::~client_t() {}

template <typename DataType>
scissor::Loader &
client_t<DataType>::loader() {
    if (!data_loader) {
        throw std::logic_error("data_loader used before init(): client_t "
                               "was constructed without a reader");
    }
    return *data_loader;
}

/**
 * Create the corresponding tree
 */
template <typename DataType>
void
client_t<DataType>::create(std::vector<DataType> &data) {
    STATEDIFF_TIMER_START(client_create_tree);
    uint8_t *data_ptr = reinterpret_cast<uint8_t *>(data.data());
    tree.create(data_ptr, client_info);
    STATEDIFF_TIMER_PERF(client_create_tree, "State-diff tree created from vector",
                         "chkpt_id", curr_chkpt_id);
    curr_chkpt_id++;
}

template <typename DataType>
void
client_t<DataType>::create(uint8_t *data_ptr) {
    STATEDIFF_TIMER_START(client_create_tree);
    tree.create(data_ptr, client_info);
    STATEDIFF_TIMER_PERF(client_create_tree, "State-diff tree created from pointer",
                         "chkpt_id", curr_chkpt_id);
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
    resize(tree.num_leaves);
}

template <typename DataType>
bool
client_t<DataType>::compare_with(int chkpt_id, const std::string& curr_fname,
                                 client_t &prev, const std::string& prev_fname,
                                 uint32_t offt_gap, size_t block_size,
                                 bool ideal_compare, bool exec_compare,
                                 TransferType compare_tier, FileSrc file_src_loc,
                                 Tuning tuning) {
    STATEDIFF_TIMER_START(client_compare_with);
    if (!(client_info == prev.client_info))
        throw std::invalid_argument(
            "statediff: comparing two clients with different metadata characteristics");
    // d1282a9 asserted this with ASSERT(x || "msg"), which never fired,
    // deserialized clients keep curr_chkpt_id == -1, so only warn.
    if (!(curr_chkpt_id == chkpt_id))
        STATEDIFF_WARN("Comparing two checkpoints with different IDs",
                       "chkpt_id", chkpt_id, "curr_chkpt_id", curr_chkpt_id);

    compare_trees(prev, working_queue, diff_hash_vec, num_hash_comp);

    // Validate hash mismatches with direct comparison
    STATEDIFF_DEBUG("Number of different hashes after phase 1");
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
        Timer::time_point setup_end = Timer::now();
        double setup_time =
            std::chrono::duration_cast<Duration>(setup_end - setup_beg).count();
        timers[0] += setup_time;

        // Throws std::logic_error if this client has no reader
        loader();
        if (ideal_compare) {
            std::cout << "Comparing data in an ideal comparison case"
                      << std::endl;
            compare_data_ideal_compute(prev, diff_hash_vec, changed_chunks,
                                       num_changed, num_comparisons,
                                       compare_tier, prev_fname, curr_fname);
        } else {
            std::cout << "Comparing data in the default comparison case with "
                         "exec_compare = "
                      << exec_compare << std::endl;
            compare_data(prev, prev_fname, curr_fname, diff_offsets,
                         changed_chunks, num_changed, num_comparisons, offt_gap,
                         block_size, compare_tier, file_src_loc, exec_compare,
                         tuning);
        }
        STATEDIFF_DEBUG("Number of different hashes after phase 2");
    }
    STATEDIFF_TIMER_PERF(client_compare_with, "State-diff tree and data compared",
                         "chkpt_id", curr_chkpt_id);
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
    STATEDIFF_DEBUG("Leaf range", "range", std::to_string(left_leaf) + "," + std::to_string(right_leaf));
    STATEDIFF_DEBUG("Start level", "range", std::to_string(last_lvl_beg) + "," + std::to_string(last_lvl_end));
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
                            assert(entry < n_chunks);
                            diff_hashes.push(entry);
                        } else {
                            // Leaf is on the last level
                            assert(node - left_leaf < n_chunks);
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
size_t
client_t<DataType>::get_wastedbytes_count() const {
    return wasted_bytes;
}

template <typename DataType>
size_t
client_t<DataType>::get_IOP_count() const {
    return IOP_count;
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
    // load time, comparison time and wait time during direct comparison
    return {timers[3], timers[4], timers[5]};
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

template <typename DataType>
std::pair<int, size_t>
client_t<DataType>::get_loader_info() const {
    return std::make_pair(best_gap, best_batch_size);
}

}   // namespace state_diff

#endif   // __STATE_DIFF_HPP