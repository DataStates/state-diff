#ifndef ROUNDING_HASH_HPP
#define ROUNDING_HASH_HPP
#include "kokkos_murmur3.hpp"
#include "common/compare_utils.hpp"

#if defined(KOKKOS_CORE_HPP)
  #define KOKKOS_INLINE KOKKOS_INLINE_FUNCTION
#else
  #define KOKKOS_INLINE inline
#endif

// Hashing approach tolerant to variations of floating point numbers
template <typename T1>
KOKKOS_INLINE
bool roundProcessData(const T1* data, uint64_t len,  T1 errorValue, HashDigest* digests);

KOKKOS_INLINE
bool roundinghash(const void* data, uint64_t len, const char dataType,
                double errorValue, HashDigest* digests)  {
  if (dataType == *("d")) {
    const double* doubleData = static_cast<const double*>(data);
    return roundProcessData(doubleData, len, errorValue, digests);
  } else if(dataType == *("f")) {
    const float* floatData = static_cast<const float*>(data);
    return roundProcessData(floatData, len, static_cast<float>(errorValue), digests);
  } else {
    kokkos_murmur3::MurmurHash3_x64_128(data, len, 0, &(digests[0]));
    return false;
  }
}

template <typename T1>
KOKKOS_INLINE
bool roundProcessData(const T1* data, uint64_t len, T1 errorValue, HashDigest* digests) {
  // Given that every data point has two hashes, compute the modified
  // binary representations per data point and compute the hashes
  // for the entire chunk in a streaming fashion.
  const size_t blockSize = 16;
  constexpr uint32_t elementsPerBlock = blockSize/sizeof(T1);

  T1 dataLower[elementsPerBlock];
  T1 dataUpper[elementsPerBlock];
  uint64_t seedLower[2] = {0, 0};
  uint64_t seedUpper[2] = {0, 0};
  uint32_t offset;

  for(offset=0; offset<len; offset+=blockSize) {
    // Declare the digest for the current block
    uint64_t* digestLower = (uint64_t*)&digests[0];
    uint64_t* digestUpper = (uint64_t*)&digests[1];
    
    // Create a copy of data for the current block
    size_t bytes_to_copy = blockSize;
    if(bytes_to_copy+offset > len) {
      for(uint32_t i=0; i<elementsPerBlock; i++) {
        dataLower[i] = 0;
        dataUpper[i] = 0;
      }
      bytes_to_copy = len - offset;
    }
    memcpy(dataLower, (const uint8_t*)(data)+offset, bytes_to_copy);
    memcpy(dataUpper, (const uint8_t*)(data)+offset, bytes_to_copy);

    // Process each element
    for(uint32_t j=0; j<elementsPerBlock; j++) {
      dataLower[j] = round(dataLower[j]*(1.0/errorValue))*(errorValue);
    }
    
    // Compute the hashes for the current block of 128-bit data
    kokkos_murmur3::MurmurHash3_x64_128(dataLower, blockSize, seedLower, digestLower);
    kokkos_murmur3::MurmurHash3_x64_128(dataUpper, blockSize, seedUpper, digestUpper);

    // Copy the content of digestLower and digestUpper into seedLower and seedUpper
    seedLower[0] = digestLower[0];
    seedLower[1] = digestLower[1];
    seedUpper[0] = digestUpper[0];
    seedUpper[1] = digestUpper[1];
  }
  return false;
}

#endif // ROUNDING_HASH_HPP
