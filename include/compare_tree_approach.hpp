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
    header_t header;
    uint32_t chunk_size;
    uint32_t current_id;
    uint32_t baseline_id;
    uint64_t data_len;
    // timers and data sizes
    std::pair<uint64_t,uint64_t> datasizes;
    double timers[4];
    double restore_timers[2];
    double io_timer=0.0;
    double compare_timer=0.0;
    MerkleTree tree1;
    MerkleTree tree2;
    MerkleTree* curr_tree;
    MerkleTree* prev_tree;
    Vector<uint32_t> first_ocur_vec; // First occurrence root offsets
    uint32_t num_chunks;
    uint32_t num_nodes;
    uint32_t start_level=30;
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
    Vector<size_t> diff_hash_vec; // Vec of idx of chunks that are different during the 1st phase of the hybrid hashing approach
    // Filenames for file streaming
    std::string file0, file1;
    size_t stream_buffer_len=1024*1024;
    size_t num_threads=1;
    size_t num_change_hashes_phase_1=0;

    void setup(const size_t data_len, std::string& filename0, std::string& filename1);

    void setup(const size_t data_len);

    void create_tree(const uint8_t* data_ptr, const size_t len);

    void dedup_data(const uint8_t* data_ptr, 
                    const size_t len,
                    bool baseline);

  public:
    CompareTreeDeduplicator();

    CompareTreeDeduplicator(uint32_t bytes_per_chunk);

    CompareTreeDeduplicator(uint32_t bytes_per_chunk, uint32_t limit, bool fuzzy=false, 
                            float errorValue=0, const char dataType=*("f"));

    KOKKOS_INLINE_FUNCTION
    ~CompareTreeDeduplicator() {};

    size_t num_first_ocur() const;
    //size_t num_first_ocur() {
    //  return first_ocur_vec.size();
    //}

    /**
     * Calculate Merkle tree and compare with the previous tree if available. Returns 
     * number of different chunks.
     *
     * \param data_device_ptr   Data to be deduplicated
     * \param data_device_len   Length of data in bytes
     * \param make_baseline     Flag determining whether to make a baseline checkpoint
     */
    size_t
    compare(uint8_t*  data_device_ptr, 
            size_t    data_device_len,
            bool      make_baseline);
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

    /**
     * Write logs for the diff metadata/data breakdown, runtimes, and the overall summary.
     *
     * \param logname Base filename for the logs
     */
    void write_diff_log(std::string& logname);

    /**
     * Function for writing the restore log.
     *
     * \param logname      Filename for writing log
     */
    void write_restore_log(std::string& logname);

    uint64_t get_num_hash_comparisons() const ;
    uint64_t get_num_comparisons() const ;
    uint64_t get_num_changes() const ;
    double get_io_time() const ;
    double get_compare_time() const ;
};

#endif // TREE_APPROACH_HPP

