#ifndef __COMPARE_TREE_APPROACH_HPP
#define __COMPARE_TREE_APPROACH_HPP
#include <Kokkos_Core.hpp>
#include <Kokkos_ScatterView.hpp>
#include <Kokkos_Bitset.hpp>
#include <Kokkos_Sort.hpp>
#include <climits>
#include "kokkos_merkle_tree.hpp"
#include "utils.hpp"
#include "kokkos_vector.hpp"
#include "comp_func.hpp"
#include "kokkos_queue.hpp"
#include "modified_kokkos_bitset.hpp"
#include "mmap_stream.hpp"
#ifdef IO_URING_STREAM
#include "io_uring_stream.hpp"
#endif

class CompareTreeDeduplicator {
  public:
    uint32_t chunk_size;
    uint32_t current_id, baseline_id;
    uint64_t data_len;
    // timers and data sizes
    std::pair<uint64_t,uint64_t> datasizes;
    double timers[4];
    double restore_timers[2];
    std::vector<double> io_timer0, io_timer1;
    double compare_timer=0.0;
    // Merkle trees
    MerkleTree tree1, tree2;
    MerkleTree *curr_tree, *prev_tree;
    uint32_t num_chunks, num_nodes;
    uint32_t start_level=12;
    bool fuzzyhash = false;
    double errorValue;
    char dataType = static_cast<char>(*("f"));
    CompareOp comp_op=Equivalence;
    // Stats
    Kokkos::View<uint64_t[1]> num_comparisons;
    Kokkos::View<uint64_t[1]> num_changed;
    Kokkos::View<uint64_t[1]> num_hash_comp;
    // Bitset for tracking which chunks have been changed
    Kokkos::Bitset<> changed_chunks;
    Vector<size_t> diff_hash_vec; // Vec of idx of chunks that are marked different during the 1st phase 
    std::string file0, file1; // Filenames for file streaming
    size_t stream_buffer_len=1024*1024;

    void setup(const size_t data_len, std::string& filename0, std::string& filename1);

    void setup(const size_t data_len);

    void create_tree(const uint8_t* data_ptr, const size_t len);

  public:
    CompareTreeDeduplicator();

    CompareTreeDeduplicator(uint32_t bytes_per_chunk);

    CompareTreeDeduplicator(uint32_t bytes_per_chunk, uint32_t limit, bool fuzzy=false, 
                            float errorValue=0, const char dataType=*("f"));

    ~CompareTreeDeduplicator() {};

    /**
     * Calculate Merkle tree and compare with the previous tree if available. Returns 
     * number of different chunks.
     *
     * \param data_device_ptr   Data to be deduplicated
     * \param data_device_len   Length of data in bytes
     * \param make_baseline     Flag determining whether to make a baseline checkpoint
     */
    size_t
    compare_trees_phase1();
    size_t
    compare_trees_phase2();

    /**
     * Serialize the current Merkle tree as well as needed metadata
     */
    std::vector<uint8_t> serialize(); 

    /**
     * Deserialize the current Merkle tree as well as needed metadata
     */
    uint64_t deserialize(std::vector<uint8_t>& buffer); 
    uint64_t deserialize(std::vector<uint8_t>& run0_buffer, std::vector<uint8_t>& run1_buffer); 
    int deserialize(uint8_t* run0_buffer, uint8_t* run1_buffer);

    uint64_t get_num_hash_comparisons() const ;
    uint64_t get_num_comparisons() const ;
    uint64_t get_num_changes() const ;
    double get_io_time() const ;
    double get_compare_time() const ;
};

#endif // TREE_APPROACH_HPP

