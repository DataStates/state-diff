#ifndef __KOKKOS_MERKLE_TREE_HPP
#define __KOKKOS_MERKLE_TREE_HPP
#include <Kokkos_Core.hpp>
#include "fuzzy_hash.hpp"
#include "truncating_hash.hpp"
#include "utils.hpp"
#include "modified_kokkos_bitset.hpp"

/** \class Merkle Tree class
 *  Merkle tree class. Merkle trees are trees where the leaves contain the hashes of
 *  data chunks and the non leaves contain the hash of the children nodes hashes concatenated
 */
class MerkleTree {
private:
  /**
   * Helper function for converting a hash digest to hex
   */
  void digest_to_hex_(const uint8_t* digest, char* output) {
    int i,j;
    char* c = output;
    for(i=0; i<static_cast<int>(sizeof(HashDigest)/4); i++) {
      for(j=0; j<4; j++) {
        sprintf(c, "%02X", digest[i*4 + j]);
        c += 2;
      }
      sprintf(c, " ");
      c += 1;
    }
    *(c-1) = '\0';
  }

public:
  using tree_type = Kokkos::View<HashDigest**>;
  Kokkos::View<HashDigest**> tree_d; ///< Device tree
  Kokkos::View<HashDigest**>::HostMirror tree_h; ///< Host mirror of tree
  Dedupe::Bitset<Kokkos::DefaultExecutionSpace> dual_hash_d;
  Dedupe::Bitset<Kokkos::DefaultHostExecutionSpace> dual_hash_h;

  /// empty constructor
  MerkleTree() {}

  /**
   * Allocate space for list of hashes on device and host. Tree is complete and binary
   * so # of nodes is 2*num_leaves-1
   *
   * \param num_leaves Number of leaves in the tree
   */
  MerkleTree(const uint32_t num_leaves) {
    tree_d = Kokkos::View<HashDigest**>("Merkle tree", (2*num_leaves-1), 1);
    tree_h = Kokkos::create_mirror_view(tree_d);
    dual_hash_d = Dedupe::Bitset<Kokkos::DefaultExecutionSpace>(2*num_leaves-1);
    dual_hash_h = Dedupe::Bitset<Kokkos::DefaultHostExecutionSpace>(2*num_leaves-1);
    dual_hash_d.reset();
    dual_hash_h.reset();
  }

  /**
   * Allocate space for tree of hashes on device and host. Tree is complete and binary
   * so # of nodes is 2*num_leaves-1
   * Set number of hash digests per node.
   *
   * \param num_leaves Number of leaves in the tree
   */
  MerkleTree(const uint32_t num_leaves, const uint32_t hpn) {
    tree_d = Kokkos::View<HashDigest**>("Merkle tree", (2*num_leaves-1), hpn);
    tree_h = Kokkos::create_mirror_view(tree_d);
    dual_hash_d = Dedupe::Bitset<Kokkos::DefaultExecutionSpace>(2*num_leaves-1);
    dual_hash_h = Dedupe::Bitset<Kokkos::DefaultHostExecutionSpace>(2*num_leaves-1);
    dual_hash_d.reset();
    dual_hash_h.reset();
  }

  /**
   * Access hash digest in tree
   *
   * \param i Index of tree node
   *
   * \return Reference to hash digest at node i
   */
  KOKKOS_INLINE_FUNCTION HashDigest& operator()(uint32_t i) const {
    return tree_d(i, 0);
  }

  KOKKOS_INLINE_FUNCTION HashDigest& operator()(uint32_t i, uint32_t j) const {
    return tree_d(i, j);
  }

  ///**
  // * Helper function for calculating hash digest for node u
  // */
  //KOKKOS_INLINE_FUNCTION bool calc_hash(uint32_t u, HashDigest* dig) const {
  //  uint32_t child_l=2*u+1, child_r=2*u+2;
  //  bool dual_hash = false;
  //  HashDigest temp[2];
  //  bool child_l_dual = dual_hash_d.test(child_l);
  //  bool child_r_dual = dual_hash_d.test(child_r);
  //  uint8_t* temp0 = (uint8_t*)(&temp[0]);
  //  uint8_t* temp1 = (uint8_t*)(&temp[1]);
  //  if(!child_l_dual && !child_r_dual) { // Both have 1 hash each
  //    memcpy(temp0, (uint8_t*)(&tree_d(child_l,0)), sizeof(HashDigest)); 
  //    memcpy(temp1, (uint8_t*)(&tree_d(child_r,0)), sizeof(HashDigest)); 
  //    kokkos_murmur3::hash(&temp, 2*sizeof(HashDigest), dig[0].digest);
  //  } else if(child_l_dual && !child_r_dual) { // Left has 2 hashes
  //    memcpy(temp0, (uint8_t*)(&tree_d(child_l,0)), sizeof(HashDigest)); 
  //    memcpy(temp1, (uint8_t*)(&tree_d(child_r,0)), sizeof(HashDigest)); 
  //    kokkos_murmur3::hash(&temp, 2*sizeof(HashDigest), dig[0].digest);
  //    memcpy(temp0, (uint8_t*)(&tree_d(child_l,1)), sizeof(HashDigest)); 
  //    memcpy(temp1, (uint8_t*)(&tree_d(child_r,0)), sizeof(HashDigest)); 
  //    kokkos_murmur3::hash(&temp, 2*sizeof(HashDigest), dig[1].digest);
  //    dual_hash = true;
  //  } else if(!child_l_dual && child_r_dual) { // Right has 2 hashes
  //    memcpy(temp0, (uint8_t*)(&tree_d(child_l,0)), sizeof(HashDigest)); 
  //    memcpy(temp1, (uint8_t*)(&tree_d(child_r,0)), sizeof(HashDigest)); 
  //    kokkos_murmur3::hash(&temp, 2*sizeof(HashDigest), dig[0].digest);
  //    memcpy(temp0, (uint8_t*)(&tree_d(child_l,0)), sizeof(HashDigest)); 
  //    memcpy(temp1, (uint8_t*)(&tree_d(child_r,1)), sizeof(HashDigest)); 
  //    kokkos_murmur3::hash(&temp, 2*sizeof(HashDigest), dig[1].digest);
  //    dual_hash = true;
  //  } else if(child_l_dual && child_r_dual) { // Both have 2 hashes
  //    memcpy(temp0, (uint8_t*)(&tree_d(child_l,0)), sizeof(HashDigest)); 
  //    memcpy(temp1, (uint8_t*)(&tree_d(child_r,0)), sizeof(HashDigest)); 
  //    kokkos_murmur3::hash(&temp, 2*sizeof(HashDigest), dig[0].digest);
  //    memcpy(temp0, (uint8_t*)(&tree_d(child_l,1)), sizeof(HashDigest)); 
  //    memcpy(temp1, (uint8_t*)(&tree_d(child_r,1)), sizeof(HashDigest)); 
  //    kokkos_murmur3::hash(&temp, 2*sizeof(HashDigest), dig[1].digest);
  //    dual_hash = true;
  //  }
  //  if (dual_hash)   {
  //    // Set the bit in the hashnum_bitset if both hashes are valid
  //    dual_hash_d.set(u);
  //  }
  //  return dual_hash;
  //}
 
  KOKKOS_INLINE_FUNCTION bool calc_hash(uint32_t u) const {
    uint32_t child_l=2*u+1, child_r=2*u+2;
    HashDigest temp[2];
    bool dual_hash = false;
    if(!dual_hash_d.test(child_l) && !dual_hash_d.test(child_r)) { // Both have 1 hash each
      memcpy((uint8_t*)(&temp[0]), (uint8_t*)(&tree_d(child_l,0)), sizeof(HashDigest)); 
      memcpy((uint8_t*)(&temp[1]), (uint8_t*)(&tree_d(child_r,0)), sizeof(HashDigest)); 
      kokkos_murmur3::hash(&temp, 2*sizeof(HashDigest), (uint8_t*)(&tree_d(u,0)));
    } else if(dual_hash_d.test(child_l) && !dual_hash_d.test(child_r)) { // Left has 2 hashes
      memcpy((uint8_t*)(&temp[0]), (uint8_t*)(&tree_d(child_l,0)), sizeof(HashDigest)); 
      memcpy((uint8_t*)(&temp[1]), (uint8_t*)(&tree_d(child_r,0)), sizeof(HashDigest)); 
      kokkos_murmur3::hash(&temp, 2*sizeof(HashDigest), (uint8_t*)(&tree_d(u,0)));

      memcpy((uint8_t*)(&temp[0]), (uint8_t*)(&tree_d(child_l,1)), sizeof(HashDigest)); 
      memcpy((uint8_t*)(&temp[1]), (uint8_t*)(&tree_d(child_r,0)), sizeof(HashDigest)); 
      kokkos_murmur3::hash(&temp, 2*sizeof(HashDigest), (uint8_t*)(&tree_d(u,1)));
      dual_hash = true;
    } else if(!dual_hash_d.test(child_l) && dual_hash_d.test(child_r)) { // Right has 2 hashes
      memcpy((uint8_t*)(&temp[0]), (uint8_t*)(&tree_d(child_l,0)), sizeof(HashDigest)); 
      memcpy((uint8_t*)(&temp[1]), (uint8_t*)(&tree_d(child_r,0)), sizeof(HashDigest)); 
      kokkos_murmur3::hash(&temp, 2*sizeof(HashDigest), (uint8_t*)(&tree_d(u,0)));

      memcpy((uint8_t*)(&temp[0]), (uint8_t*)(&tree_d(child_l,0)), sizeof(HashDigest)); 
      memcpy((uint8_t*)(&temp[1]), (uint8_t*)(&tree_d(child_r,1)), sizeof(HashDigest)); 
      kokkos_murmur3::hash(&temp, 2*sizeof(HashDigest), (uint8_t*)(&tree_d(u,1)));
      dual_hash = true;
    } else if(dual_hash_d.test(child_l) && dual_hash_d.test(child_r)) { // Both have 2 hashes
      memcpy((uint8_t*)(&temp[0]), (uint8_t*)(&tree_d(child_l,0)), sizeof(HashDigest)); 
      memcpy((uint8_t*)(&temp[1]), (uint8_t*)(&tree_d(child_r,0)), sizeof(HashDigest)); 
      kokkos_murmur3::hash(&temp, 2*sizeof(HashDigest), (uint8_t*)(&tree_d(u,0)));

      memcpy((uint8_t*)(&temp[0]), (uint8_t*)(&tree_d(child_l,1)), sizeof(HashDigest)); 
      memcpy((uint8_t*)(&temp[1]), (uint8_t*)(&tree_d(child_r,1)), sizeof(HashDigest)); 
      kokkos_murmur3::hash(&temp, 2*sizeof(HashDigest), (uint8_t*)(&tree_d(u,1)));
      dual_hash = true;
    }
    if (dual_hash)   {
      // Set the bit in the hashnum_bitset if both hashes are valid
      dual_hash_d.set(u);
    }
    return dual_hash;
  }
 
  KOKKOS_INLINE_FUNCTION bool calc_leaf_hash(const void* data, uint64_t len, HashDigest& digest) const {
    kokkos_murmur3::hash(data, len, (uint8_t*)(&digest));
    return false;
  }
 
  KOKKOS_INLINE_FUNCTION bool calc_leaf_hash(const void* data, uint64_t len, uint32_t u) const {
    kokkos_murmur3::hash(data, len, (uint8_t*)(&tree_d(u,0)));
    dual_hash_d.reset(u);
    return false;
  }

  KOKKOS_INLINE_FUNCTION bool 
  calc_leaf_fuzzy_hash(const void* data, uint64_t len, 
                       float errorValue, const char dataType, uint32_t u) const {
    
    HashDigest digests[2] = {0};
    //bool dualValid = fuzzyhash(data, len, dataType, errorValue, digests);
    bool dualValid = trunchash(data, len, dataType, errorValue, digests);

    // Set the bit in the hashnum_bitset if both hashes are valid
    tree_d(u,0) = digests[0];
    if (dualValid) {
      dual_hash_d.set(u);
      tree_d(u,1) = digests[1];
    }
    return dualValid;
  }
 
  KOKKOS_INLINE_FUNCTION bool 
  calc_leaf_fuzzy_hash(const void* data, uint64_t len, 
                       float errorValue, const char dataType, HashDigest* digests, uint32_t u) const {
    
    bool dualValid = fuzzyhash(data, len, dataType, errorValue, digests);

    // Set the bit in the hashnum_bitset if both hashes are valid
    if (dualValid) {
      dual_hash_d.set(u);
    }
    return dualValid;
  }
 
  /**
   * Print leaves of tree in hex
   */
  void print_leaves() {
    Kokkos::deep_copy(tree_h, tree_d);
    uint32_t num_leaves = (tree_h.extent(0)+1)/2;
    printf("============================================================\n");
    char buffer[64];
    unsigned int counter = 2;
    for(unsigned int i=num_leaves-1; i<tree_h.extent(0); i++) {
      digest_to_hex_((uint8_t*)(tree_h(i,0).digest), buffer);
      printf("Node: %u: %s \n", i, buffer);
      if(i == counter) {
        printf("\n");
        counter += 2*counter;
      }
    }
    printf("============================================================\n");
  }

  void print() {
    Kokkos::deep_copy(tree_h, tree_d);
    printf("============================================================\n");
    char buffer[64];
    unsigned int counter = 2;
    for(unsigned int i=16777215; i<16777315; i++) {
      digest_to_hex_((uint8_t*)(tree_h(i,0).digest), buffer);
      printf("Node: %u: %s \n", i, buffer);
      if(i == counter) {
        printf("\n");
        counter += 2*counter;
      }
    }
    printf("============================================================\n");
  }
};

// Calculate the number of leaves for the tree rooted at node
KOKKOS_INLINE_FUNCTION uint32_t num_leaf_descendents(uint32_t node, uint32_t num_nodes) {
  uint32_t leftmost = (2*node)+1;
  uint32_t rightmost = (2*node)+2;
  uint32_t num_leaves = 0;
  while(leftmost < num_nodes) {
    leftmost = (2*leftmost)+1;
    rightmost = (2*rightmost)+2;
  }
  leftmost = (leftmost-1)/2;
  rightmost = (rightmost-2)/2;
  uint32_t old_right = rightmost;
  bool split_flag = false;
  if(rightmost > num_nodes-1) {
    rightmost = num_nodes-1;
    split_flag = true;
  }
  num_leaves += rightmost-leftmost+1;
  if(split_flag) {
    leftmost = ((num_nodes-1)/2);
    rightmost = (old_right-2)/2;
    num_leaves += rightmost-leftmost+1;
  }
  return num_leaves;
}

// Get the leftmost leaf of the tree rooted at node
KOKKOS_INLINE_FUNCTION uint32_t leftmost_leaf(uint32_t node, uint32_t num_nodes) {
  uint32_t leftmost = (2*node)+1;
  while(leftmost < num_nodes) {
    leftmost = (2*leftmost)+1;
  }
  leftmost = (leftmost-1)/2;
  return static_cast<uint32_t>(leftmost);
}

// Get the rightmost leaf of the tree rooted at node
KOKKOS_INLINE_FUNCTION uint32_t rightmost_leaf(uint32_t node, uint32_t num_nodes) {
  uint32_t leftmost = (2*node)+1;
  uint32_t rightmost = (2*node)+2;
  while(leftmost < num_nodes) {
    leftmost = (2*leftmost)+1;
    rightmost = (2*rightmost)+2;
  }
  leftmost = (leftmost-1)/2;
  rightmost = (rightmost-2)/2;
  if(rightmost > num_nodes)
    rightmost = num_nodes-1;
  return static_cast<uint32_t>(rightmost);
}

#endif // KOKKOS_MERKLE_TREE_HPP

