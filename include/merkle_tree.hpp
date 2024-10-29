#ifndef __KOKKOS_MERKLE_TREE_HPP
#define __KOKKOS_MERKLE_TREE_HPP
#include "common/compare_utils.hpp"
#include "common/io_utils.hpp"
#include "common/statediff_bitset.hpp"
#include "rounding_hash.hpp"
#include <Kokkos_Core.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/types/vector.hpp>
#include <fstream>

#ifdef __NVCC__
#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_timer.hpp"
#include "nvtx3/nvToolsExt.h"
#endif   //__NVCC__

/** \class Merkle Tree class
 *  Merkle tree class. Merkle trees are trees where the leaves contain the
 * hashes of data chunks and the non leaves contain the hash of the children
 * nodes hashes concatenated
 */
class tree_t {
  private:
    double timers[4];
    double create_time;
    void digest_to_hex(const uint8_t *digest, char *output);
    KOKKOS_INLINE_FUNCTION bool calc_hash(uint32_t u) const;
    KOKKOS_INLINE_FUNCTION bool calc_leaf_hash(const void *data, uint64_t size,
                                               uint32_t u) const;
    KOKKOS_INLINE_FUNCTION bool
    calc_leaf_fuzzy_hash(const void *data, uint64_t size, float errorValue,
                         const char dataType, uint32_t u) const;
    // KOKKOS_INLINE_FUNCTION void hash_leaves_kernel(uint8_t *data_ptr,
    //                                                client_info_t client_info,
    //                                                uint32_t left_leaf,
    //                                                uint32_t idx);
    // void _hash_leaves_kernel(uint8_t *data_ptr, client_info_t client_info,
    //                 uint32_t left_leaf);
    void create_leaves_cuda(uint8_t *data_ptr, client_info_t client_info,
                            uint32_t left_leaf, std::string diff_label);

  public:
    size_t num_leaves;
    size_t num_nodes;
    size_t chunk_size;
    bool use_fuzzyhash;
    Kokkos::View<HashDigest *> tree_d;

    tree_t(const size_t data_size, const size_t chunk_size, bool fuzzyhash);
    tree_t();

    void create(uint8_t *data_ptr, client_info_t client_info);
    template <class Archive>
    void save(Archive &ar, const unsigned int version) const;
    template <class Archive> void load(Archive &ar, const unsigned int version);

    KOKKOS_INLINE_FUNCTION HashDigest &operator[](uint32_t i) const;
    void create_leaves(uint8_t *data_ptr, client_info_t client_info,
                       uint32_t left_leaf, std::string diff_label);

    KOKKOS_INLINE_FUNCTION void hash_leaves_kernel(uint8_t *data_ptr,
                                                   client_info_t client_info,
                                                   uint32_t left_leaf,
                                                   uint32_t idx) const;
    KOKKOS_INLINE_FUNCTION uint32_t num_leaf_descendents(uint32_t node,
                                                         uint32_t num_nodes);
    KOKKOS_INLINE_FUNCTION uint32_t leftmost_leaf(uint32_t node,
                                                  uint32_t num_nodes);
    KOKKOS_INLINE_FUNCTION uint32_t rightmost_leaf(uint32_t node,
                                                   uint32_t num_nodes);

    void print_leaves();
    // double get_time() const;
    const double* get_timers() const;
};
#endif   //  __KOKKOS_MERKLE_TREE_HPP