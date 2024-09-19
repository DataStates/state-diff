#ifndef __COMP_FUNC_HPP
#define __COMP_FUNC_HPP

#include <Kokkos_Core.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/types/vector.hpp>
#include <common/statediff_queue.hpp>
#include <common/statediff_vector.hpp>

enum CompareOp { Equivalence = 0, Relative = 1, Absolute = 2 };

template <typename T> struct BaseComp {
    KOKKOS_INLINE_FUNCTION
    bool operator()(const T &a, const T &b, double tol) const { return true; }
};

template <typename T> struct EquivalenceComp : public BaseComp<T> {
    KOKKOS_INLINE_FUNCTION
    bool operator()(const T &a, const T &b, double tol) const { return a == b; }
};

template <typename T> struct RelativeComp : public BaseComp<T> {
    KOKKOS_INLINE_FUNCTION
    bool operator()(const T &expect, const T &approx, double tol) const {
        return Kokkos::abs((approx - expect) / expect) <= tol;
    }
};

template <typename T> struct AbsoluteComp : public BaseComp<T> {
    KOKKOS_INLINE_FUNCTION
    bool operator()(const T &a, const T &b, double tol) const {
        return Kokkos::abs(static_cast<double>(b) - static_cast<double>(a)) <=
               tol;
    }
};

using Timer = std::chrono::high_resolution_clock;
using Duration = std::chrono::duration<double>;

struct alignas(16) HashDigest {
    uint8_t digest[16] = {0};

    template <class Archive>
    void serialize(Archive &ar, const unsigned int version) {
        ar(cereal::binary_data(digest, sizeof(digest)));
    }
};

// Helper function for checking if two hash digests are identical
KOKKOS_INLINE_FUNCTION
bool
digests_same(const HashDigest &lhs, const HashDigest &rhs) {
    uint64_t *l_ptr = (uint64_t *)(lhs.digest);
    uint64_t *r_ptr = (uint64_t *)(rhs.digest);
    for (size_t i = 0; i < sizeof(HashDigest) / 8; i++) {
        if (l_ptr[i] != r_ptr[i]) {
            return false;
        }
    }
    return true;
}

template <typename TeamMember>
KOKKOS_FORCEINLINE_FUNCTION void
team_memcpy(uint8_t *dst, uint8_t *src, size_t len, TeamMember &team_member) {
    uint32_t *src_u32 = (uint32_t *)(src);
    uint32_t *dst_u32 = (uint32_t *)(dst);
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, len / 4),
                         [&](const uint64_t &j) { dst_u32[j] = src_u32[j]; });
    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team_member, len % 4), [&](const uint64_t &j) {
            dst[((len / 4) * 4) + j] = src[((len / 4) * 4) + j];
        });
}

typedef struct header_t {
    uint32_t ref_id;            // ID of reference diff
    uint32_t cur_id;            // ID of current diff
    uint64_t datalen;           // Length of memory region in bytes
    uint32_t chunk_size;        // Size of chunks
    uint32_t num_first_ocur;    // Number of first occurrence entries
    uint32_t num_prior_diffs;   // Number of prior diffs needed for restoration
    uint32_t num_shift_dupl;    // Number of duplicate entries
} header_t;

struct client_info_t {
    int id;
    char data_type;
    size_t data_size;
    size_t chunk_size;
    size_t start_level;
    double error_tolerance;

    // Ensuring client_info_t is serializable
    template <class Archive>
    void serialize(Archive &archive, const unsigned int version) {
        archive(id, data_type, data_size, chunk_size, start_level,
                error_tolerance);
    }

    // operator to assess two clients to make sure metadata match
    bool operator==(const client_info_t &other) const {
        return (data_type == other.data_type) && (data_size == other.data_size) &&
               (chunk_size == other.chunk_size) &&
               (start_level == other.start_level) &&
               (error_tolerance == other.error_tolerance);
    }
};

#endif   // __COMP_FUNC_HPP