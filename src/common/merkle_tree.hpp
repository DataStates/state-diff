#ifndef __KOKKOS_MERKLE_TREE_HPP
#define __KOKKOS_MERKLE_TREE_HPP
#include "common/compare_utils.hpp"
#include "common/statediff_bitset.hpp"
#include "rounding_hash.hpp"
#include "common/debug.hpp"
#include <Kokkos_Core.hpp>
#include <fstream>

#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/array.hpp>
#include <boost/serialization/split_member.hpp>
#include <boost/serialization/base_object.hpp>

/** \class Merkle Tree class
 *  Merkle tree class. Merkle trees are trees where the leaves contain the
 * hashes of data chunks and the non leaves contain the hash of the children
 * nodes hashes concatenated
 */
class tree_t {
  private:
    void digest_to_hex(const uint8_t *digest, char *output);
    KOKKOS_FUNCTION bool calc_hash(uint32_t u) const;
    KOKKOS_FUNCTION bool calc_leaf_hash(const void *data, uint64_t len,
                                        uint32_t u) const;
    KOKKOS_FUNCTION bool calc_leaf_fuzzy_hash(const void *data, uint64_t len,
                                              float errorValue,
                                              const char dataType,
                                              uint32_t u) const;

  public:
    size_t client_id;
    size_t num_leaves;
    size_t num_nodes;
    size_t chunk_size;
    bool use_fuzzyhash;
    Kokkos::View<HashDigest *> tree_d;

    tree_t(const size_t data_len, const size_t chunk_size,
           bool fuzzyhash);
    tree_t();
    KOKKOS_FUNCTION HashDigest &operator[](uint32_t i) const;

    void create(const std::vector<uint8_t> &data, double errorValue,
                   char dataType, size_t start_level);
    template <class Archive>
    void serialize(Archive &ar, const unsigned int version);
    template <class Archive>
    void deserialize(Archive &ar, const unsigned int version);

    KOKKOS_FUNCTION uint32_t num_leaf_descendents(uint32_t node,
                                                  uint32_t num_nodes);
    KOKKOS_FUNCTION uint32_t leftmost_leaf(uint32_t node, uint32_t num_nodes);
    KOKKOS_FUNCTION uint32_t rightmost_leaf(uint32_t node, uint32_t num_nodes);

    void print_leaves();
    
};
#endif   //  __KOKKOS_MERKLE_TREE_HPP