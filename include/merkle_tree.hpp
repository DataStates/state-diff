#ifndef __KOKKOS_MERKLE_TREE_HPP
#define __KOKKOS_MERKLE_TREE_HPP
#include "common/compare_utils.hpp"
#include "common/statediff_bitset.hpp"
#include "rounding_hash.hpp"
#include <Kokkos_Core.hpp>

/** \class Merkle Tree class
 *  Merkle tree class. Merkle trees are trees where the leaves contain the
 * hashes of data chunks and the non leaves contain the hash of the children
 * nodes hashes concatenated
 */
class tree_t {
  private:
    void digest_to_hex_(const uint8_t *digest, char *output);

  public:
    using tree_type = Kokkos::View<HashDigest **>;
    Kokkos::View<HashDigest **> tree_d;
    Kokkos::View<HashDigest **>::HostMirror tree_h;
    Dedupe::Bitset<Kokkos::DefaultExecutionSpace> dual_hash_d;
    Dedupe::Bitset<Kokkos::DefaultHostExecutionSpace> dual_hash_h;

    tree_t();
    tree_t(const uint32_t num_leaves);
    tree_t(const uint32_t num_leaves, const uint32_t hpn);

    KOKKOS_INLINE_FUNCTION HashDigest &operator()(uint32_t i) const;
    KOKKOS_INLINE_FUNCTION HashDigest &operator()(uint32_t i, uint32_t j) const;

    KOKKOS_INLINE_FUNCTION bool calc_hash(uint32_t u) const;
    KOKKOS_INLINE_FUNCTION bool calc_leaf_hash(const void *data, uint64_t len,
                                               HashDigest &digest) const;
    KOKKOS_INLINE_FUNCTION bool calc_leaf_hash(const void *data, uint64_t len,
                                               uint32_t u) const;
    KOKKOS_INLINE_FUNCTION bool
    calc_leaf_fuzzy_hash(const void *data, uint64_t len, float errorValue,
                         const char dataType, uint32_t u) const;

    KOKKOS_INLINE_FUNCTION uint32_t num_leaf_descendents(uint32_t node,
                                                         uint32_t num_nodes);
    KOKKOS_INLINE_FUNCTION uint32_t leftmost_leaf(uint32_t node,
                                                  uint32_t num_nodes);
    KOKKOS_INLINE_FUNCTION uint32_t rightmost_leaf(uint32_t node,
                                                   uint32_t num_nodes);

    void print_leaves();
    void print();
};

void
tree_t::digest_to_hex_(const uint8_t *digest, char *output) {
    int i, j;
    char *c = output;
    for (i = 0; i < static_cast<int>(sizeof(HashDigest) / 4); i++) {
        for (j = 0; j < 4; j++) {
            sprintf(c, "%02X", digest[i * 4 + j]);
            c += 2;
        }
        sprintf(c, " ");
        c += 1;
    }
    *(c - 1) = '\0';
}

tree_t::tree_t() {}

/**
 * Allocate space for list of hashes on device and host. Tree is complete and
 * binary so # of nodes is 2*num_leaves-1
 *
 * \param num_leaves Number of leaves in the tree
 */
tree_t::tree_t(const uint32_t num_leaves) {
    tree_d =
        Kokkos::View<HashDigest **>("Merkle tree", (2 * num_leaves - 1), 1);
    tree_h = Kokkos::create_mirror_view(tree_d);
    dual_hash_d =
        Dedupe::Bitset<Kokkos::DefaultExecutionSpace>(2 * num_leaves - 1);
    dual_hash_h =
        Dedupe::Bitset<Kokkos::DefaultHostExecutionSpace>(2 * num_leaves - 1);
    dual_hash_d.reset();
    dual_hash_h.reset();
}

/**
 * Allocate space for tree of hashes on device and host. Tree is complete and
 * binary so # of nodes is 2*num_leaves-1 Set number of hash digests per node.
 *
 * \param num_leaves Number of leaves in the tree
 */
tree_t::tree_t(const uint32_t num_leaves, const uint32_t hpn) {
    tree_d =
        Kokkos::View<HashDigest **>("Merkle tree", (2 * num_leaves - 1), hpn);
    tree_h = Kokkos::create_mirror_view(tree_d);
    dual_hash_d =
        Dedupe::Bitset<Kokkos::DefaultExecutionSpace>(2 * num_leaves - 1);
    dual_hash_h =
        Dedupe::Bitset<Kokkos::DefaultHostExecutionSpace>(2 * num_leaves - 1);
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
KOKKOS_INLINE_FUNCTION
HashDigest &
tree_t::operator()(uint32_t i) const {
    return tree_d(i, 0);
}

KOKKOS_INLINE_FUNCTION
HashDigest &
tree_t::operator()(uint32_t i, uint32_t j) const {
    return tree_d(i, j);
}

KOKKOS_INLINE_FUNCTION
bool
tree_t::calc_hash(uint32_t u) const {
    uint32_t child_l = 2 * u + 1, child_r = 2 * u + 2;
    HashDigest temp[2];
    bool dual_hash = false;
    if (!dual_hash_d.test(child_l) &&
        !dual_hash_d.test(child_r)) {   // Both have 1 hash each
        memcpy((uint8_t *) (&temp[0]), (uint8_t *) (&tree_d(child_l, 0)),
               sizeof(HashDigest));
        memcpy((uint8_t *) (&temp[1]), (uint8_t *) (&tree_d(child_r, 0)),
               sizeof(HashDigest));
        kokkos_murmur3::hash(&temp, 2 * sizeof(HashDigest),
                             (uint8_t *) (&tree_d(u, 0)));
    } else if (dual_hash_d.test(child_l) &&
               !dual_hash_d.test(child_r)) {   // Left has 2 hashes
        memcpy((uint8_t *) (&temp[0]), (uint8_t *) (&tree_d(child_l, 0)),
               sizeof(HashDigest));
        memcpy((uint8_t *) (&temp[1]), (uint8_t *) (&tree_d(child_r, 0)),
               sizeof(HashDigest));
        kokkos_murmur3::hash(&temp, 2 * sizeof(HashDigest),
                             (uint8_t *) (&tree_d(u, 0)));

        memcpy((uint8_t *) (&temp[0]), (uint8_t *) (&tree_d(child_l, 1)),
               sizeof(HashDigest));
        memcpy((uint8_t *) (&temp[1]), (uint8_t *) (&tree_d(child_r, 0)),
               sizeof(HashDigest));
        kokkos_murmur3::hash(&temp, 2 * sizeof(HashDigest),
                             (uint8_t *) (&tree_d(u, 1)));
        dual_hash = true;
    } else if (!dual_hash_d.test(child_l) &&
               dual_hash_d.test(child_r)) {   // Right has 2 hashes
        memcpy((uint8_t *) (&temp[0]), (uint8_t *) (&tree_d(child_l, 0)),
               sizeof(HashDigest));
        memcpy((uint8_t *) (&temp[1]), (uint8_t *) (&tree_d(child_r, 0)),
               sizeof(HashDigest));
        kokkos_murmur3::hash(&temp, 2 * sizeof(HashDigest),
                             (uint8_t *) (&tree_d(u, 0)));

        memcpy((uint8_t *) (&temp[0]), (uint8_t *) (&tree_d(child_l, 0)),
               sizeof(HashDigest));
        memcpy((uint8_t *) (&temp[1]), (uint8_t *) (&tree_d(child_r, 1)),
               sizeof(HashDigest));
        kokkos_murmur3::hash(&temp, 2 * sizeof(HashDigest),
                             (uint8_t *) (&tree_d(u, 1)));
        dual_hash = true;
    } else if (dual_hash_d.test(child_l) &&
               dual_hash_d.test(child_r)) {   // Both have 2 hashes
        memcpy((uint8_t *) (&temp[0]), (uint8_t *) (&tree_d(child_l, 0)),
               sizeof(HashDigest));
        memcpy((uint8_t *) (&temp[1]), (uint8_t *) (&tree_d(child_r, 0)),
               sizeof(HashDigest));
        kokkos_murmur3::hash(&temp, 2 * sizeof(HashDigest),
                             (uint8_t *) (&tree_d(u, 0)));

        memcpy((uint8_t *) (&temp[0]), (uint8_t *) (&tree_d(child_l, 1)),
               sizeof(HashDigest));
        memcpy((uint8_t *) (&temp[1]), (uint8_t *) (&tree_d(child_r, 1)),
               sizeof(HashDigest));
        kokkos_murmur3::hash(&temp, 2 * sizeof(HashDigest),
                             (uint8_t *) (&tree_d(u, 1)));
        dual_hash = true;
    }
    if (dual_hash) {
        // Set the bit in the hashnum_bitset if both hashes are valid
        dual_hash_d.set(u);
    }
    return dual_hash;
}

KOKKOS_INLINE_FUNCTION
bool
tree_t::calc_leaf_hash(const void *data, uint64_t len,
                       HashDigest &digest) const {
    kokkos_murmur3::hash(data, len, (uint8_t *) (&digest));
    return false;
}

KOKKOS_INLINE_FUNCTION
bool
tree_t::calc_leaf_hash(const void *data, uint64_t len, uint32_t u) const {
    kokkos_murmur3::hash(data, len, (uint8_t *) (&tree_d(u, 0)));
    dual_hash_d.reset(u);
    return false;
}

KOKKOS_INLINE_FUNCTION
bool
tree_t::calc_leaf_fuzzy_hash(const void *data, uint64_t len, float errorValue,
                             const char dataType, uint32_t u) const {

    HashDigest digests[2] = {0};
    bool dualValid = roundinghash(data, len, dataType, errorValue, digests);
    // Set the bit in the hashnum_bitset if both hashes are valid
    tree_d(u, 0) = digests[0];
    if (dualValid) {
        dual_hash_d.set(u);
        tree_d(u, 1) = digests[1];
    }
    return dualValid;
}

/**
 * Print leaves of tree in hex
 */
void
tree_t::print_leaves() {
    Kokkos::deep_copy(tree_h, tree_d);
    uint32_t num_leaves = (tree_h.extent(0) + 1) / 2;
    printf("============================================================\n");
    char buffer[64];
    unsigned int counter = 2;
    for (unsigned int i = num_leaves - 1; i < tree_h.extent(0); i++) {
        digest_to_hex_((uint8_t *) (tree_h(i, 0).digest), buffer);
        printf("Node: %u: %s \n", i, buffer);
        if (i == counter) {
            printf("\n");
            counter += 2 * counter;
        }
    }
    printf("============================================================\n");
}

void
tree_t::print() {
    Kokkos::deep_copy(tree_h, tree_d);
    printf("============================================================\n");
    char buffer[64];
    unsigned int counter = 2;
    for (unsigned int i = 16777215; i < 16777315; i++) {
        digest_to_hex_((uint8_t *) (tree_h(i, 0).digest), buffer);
        printf("Node: %u: %s \n", i, buffer);
        if (i == counter) {
            printf("\n");
            counter += 2 * counter;
        }
    }
    printf("============================================================\n");
}

// Calculate the number of leaves for the tree rooted at node
KOKKOS_INLINE_FUNCTION
uint32_t
tree_t::num_leaf_descendents(uint32_t node, uint32_t num_nodes) {
    uint32_t leftmost = (2 * node) + 1;
    uint32_t rightmost = (2 * node) + 2;
    uint32_t num_leaves = 0;
    while (leftmost < num_nodes) {
        leftmost = (2 * leftmost) + 1;
        rightmost = (2 * rightmost) + 2;
    }
    leftmost = (leftmost - 1) / 2;
    rightmost = (rightmost - 2) / 2;
    uint32_t old_right = rightmost;
    bool split_flag = false;
    if (rightmost > num_nodes - 1) {
        rightmost = num_nodes - 1;
        split_flag = true;
    }
    num_leaves += rightmost - leftmost + 1;
    if (split_flag) {
        leftmost = ((num_nodes - 1) / 2);
        rightmost = (old_right - 2) / 2;
        num_leaves += rightmost - leftmost + 1;
    }
    return num_leaves;
}

// Get the leftmost leaf of the tree rooted at node
KOKKOS_INLINE_FUNCTION
uint32_t
tree_t::leftmost_leaf(uint32_t node, uint32_t num_nodes) {
    uint32_t leftmost = (2 * node) + 1;
    while (leftmost < num_nodes) {
        leftmost = (2 * leftmost) + 1;
    }
    leftmost = (leftmost - 1) / 2;
    return static_cast<uint32_t>(leftmost);
}

// Get the rightmost leaf of the tree rooted at node
KOKKOS_INLINE_FUNCTION
uint32_t
tree_t::rightmost_leaf(uint32_t node, uint32_t num_nodes) {
    uint32_t leftmost = (2 * node) + 1;
    uint32_t rightmost = (2 * node) + 2;
    while (leftmost < num_nodes) {
        leftmost = (2 * leftmost) + 1;
        rightmost = (2 * rightmost) + 2;
    }
    leftmost = (leftmost - 1) / 2;
    rightmost = (rightmost - 2) / 2;
    if (rightmost > num_nodes)
        rightmost = num_nodes - 1;
    return static_cast<uint32_t>(rightmost);
}

#endif   // KOKKOS_MERKLE_TREE_HPP
