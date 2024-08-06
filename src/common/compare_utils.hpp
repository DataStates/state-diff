#ifndef __COMP_FUNC_HPP
#define __COMP_FUNC_HPP

#include <Kokkos_Core.hpp>
#include <common/statediff_queue.hpp>
#include <common/statediff_vector.hpp>

enum CompareOp {
  Equivalence=0,
  Relative=1,
  Absolute=2
};

template<typename T>
struct BaseComp {
    KOKKOS_INLINE_FUNCTION
    bool operator()(const T& a, const T& b, double tol) const {
      return true;
    }
};

template<typename T>
struct EquivalenceComp : public BaseComp<T> {
    KOKKOS_INLINE_FUNCTION
    bool operator()(const T& a, const T& b, double tol) const {
        return a == b;
    }
};

template<typename T>
struct RelativeComp : public BaseComp<T> {
    KOKKOS_INLINE_FUNCTION
    bool operator()(const T& expect, const T& approx, double tol) const {
        return Kokkos::abs((approx-expect)/expect) <= tol;
    }
};

template<typename T>
struct AbsoluteComp : public BaseComp<T> {
    KOKKOS_INLINE_FUNCTION
    bool operator()(const T& a, const T& b, double tol) const {
        return Kokkos::abs(static_cast<double>(b)-static_cast<double>(a)) <= tol;
    }
};


using Timer = std::chrono::high_resolution_clock;
using Duration = std::chrono::duration<double>;

struct alignas(16) HashDigest {
  uint8_t digest[16] = {0};
};

// Helper function for checking if two hash digests are identical
KOKKOS_INLINE_FUNCTION
bool digests_same(const HashDigest& lhs, const HashDigest& rhs) {
  uint64_t* l_ptr = (uint64_t*)(lhs.digest);
  uint64_t* r_ptr = (uint64_t*)(rhs.digest);
  for(size_t i=0; i<sizeof(HashDigest)/8; i++) {
    if(l_ptr[i] != r_ptr[i]) {
      return false;
    }
  }
  return true;
}

template <typename TeamMember>
KOKKOS_FORCEINLINE_FUNCTION
void team_memcpy(uint8_t* dst, uint8_t* src, size_t len, TeamMember& team_member) {
  uint32_t* src_u32 = (uint32_t*)(src);
  uint32_t* dst_u32 = (uint32_t*)(dst);
  Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, len/4), [&] (const uint64_t& j) {
    dst_u32[j] = src_u32[j];
  });
  Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, len%4), [&] (const uint64_t& j) {
    dst[((len/4)*4)+j] = src[((len/4)*4)+j];
  });
}

typedef struct header_t {
  uint32_t ref_id;           // ID of reference diff
  uint32_t cur_id;         // ID of current diff
  uint64_t datalen;          // Length of memory region in bytes
  uint32_t chunk_size;       // Size of chunks
  uint32_t num_first_ocur;    // Number of first occurrence entries
  uint32_t num_prior_diffs;   // Number of prior diffs needed for restoration
  uint32_t num_shift_dupl;      // Number of duplicate entries
} header_t;

#endif // __COMP_FUNC_HPP