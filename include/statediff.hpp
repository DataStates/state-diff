#ifndef __STATE_DIFF_HPP
#define __STATE_DIFF_HPP

#include "Kokkos_Bitset.hpp"
#include "Kokkos_Core.hpp"
#include "Kokkos_ScatterView.hpp"
#include "Kokkos_Sort.hpp"
#include "common/compare_utils.hpp"
#include "common/debug.hpp"
#include "common/statediff_bitset.hpp"
#include "io_reader.hpp"
#include "merkle_tree.hpp"
#include "reader_factory.hpp"
#include <climits>
#include <cstddef>
#include <vector>

namespace state_diff {
template <typename DataType> class client_t {
    static const size_t DEFAULT_CHUNK_SIZE = 4096;
    static const size_t DEFAULT_START_LEVEL = 13;
    static const bool DEFAULT_FUZZY_HASH = true;
    static const char DEFAULT_DTYPE = 'f';

    int client_id;   // current_id
    tree_t *tree = nullptr;
    size_t data_len;
    size_t chunk_size;
    size_t num_chunks;    // tree
    size_t num_nodes;     // tree
    size_t start_level;   // tree
    bool use_fuzzyhash;
    double errorValue;
    char dataType;
    std::string data_fn;
    io_reader_t<DataType> io_reader;

    CompareOp comp_op = Equivalence;
    Queue working_queue;
    Kokkos::Bitset<>
        changed_chunks;   // Bitset for tracking which chunks have been changed
    Vector<size_t> diff_hash_vec;   // Vec of idx of chunks that are marked
                                    // different during the 1st phase

    // timers
    std::vector<double> io_timer;
    double compare_timer;

    // Stats
    Kokkos::View<uint64_t[1]> num_comparisons =
        Kokkos::View<uint64_t[1]>("Num comparisons");
    Kokkos::View<uint64_t[1]> num_changed =
        Kokkos::View<uint64_t[1]>("Num changed");
    Kokkos::View<uint64_t[1]> num_hash_comp =
        Kokkos::View<uint64_t[1]>("Num hash comparisons");

    void setup();

    // Stats getters
    size_t get_num_hash_comparisons() const;
    size_t get_num_comparisons() const;
    size_t get_num_changes() const;
    double get_io_time() const;
    double get_compare_time() const;

  public:
    client_t(int client_id, io_reader_t<DataType> &io_reader, size_t data_len,
             double error, char dtype = DEFAULT_DTYPE,
             size_t chunk_size = DEFAULT_CHUNK_SIZE,
             size_t start_level = DEFAULT_START_LEVEL,
             bool fuzzyhash = DEFAULT_FUZZY_HASH);

    ~client_t();

    void create(const std::vector<DataType> &data);
    template <class Archive> void serialize(Archive &ar);
    std::vector<uint8_t> serialize();
    size_t deserialize(std::vector<uint8_t> &tree_data);
    bool compare_with(const client_t &prev);

    size_t compare_trees(const client_t &prev, Queue &working_queue,
                         Vector<size_t> &diff_hash_vec,
                         Kokkos::View<uint64_t[1]> &num_hash_comp);
    size_t compare_data(const client_t &prev, Vector<size_t> &diff_hash_vec,
                        Kokkos::Bitset<> &changed_chunks,
                        Kokkos::View<uint64_t[1]> &num_changed,
                        Kokkos::View<uint64_t[1]> &num_comparisons);
};

}   // namespace state_diff

#endif   // __STATE_DIFF_HPP