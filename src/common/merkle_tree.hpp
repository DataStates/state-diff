#ifndef __KOKKOS_MERKLE_TREE_HPP
#define __KOKKOS_MERKLE_TREE_HPP
#include "common/compare_utils.hpp"
#include "common/debug.hpp"
#include "common/statediff_bitset.hpp"
#include "rounding_hash.hpp"
#include <Kokkos_Core.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/array.hpp>
#include <boost/serialization/serialization.hpp>
#include <fstream>

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
    size_t num_leaves;
    size_t num_nodes;
    size_t chunk_size;
    bool use_fuzzyhash;
    Kokkos::View<HashDigest *> tree_d;

    tree_t(const size_t data_len, const size_t chunk_size, bool fuzzyhash);
    tree_t();

    void create(const uint8_t *data_ptr, client_info_t client_info);
    template <class Archive>
    void save(Archive &ar, const unsigned int version) const;
    template <class Archive> void load(Archive &ar, const unsigned int version);

    KOKKOS_FUNCTION HashDigest &operator[](uint32_t i) const;
    KOKKOS_FUNCTION uint32_t num_leaf_descendents(uint32_t node,
                                                  uint32_t num_nodes);
    KOKKOS_FUNCTION uint32_t leftmost_leaf(uint32_t node, uint32_t num_nodes);
    KOKKOS_FUNCTION uint32_t rightmost_leaf(uint32_t node, uint32_t num_nodes);

    void print_leaves();

    // Macro for boost to split serialization into save/load
    BOOST_SERIALIZATION_SPLIT_MEMBER()
};
#endif   //  __KOKKOS_MERKLE_TREE_HPP