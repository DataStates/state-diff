#ifndef ROUNDING_HASH_HPP
#define ROUNDING_HASH_HPP
#include "kokkos_murmur3.hpp"
#include "utils.hpp"

// Hashing approach tolerant to variations of floating point numbers
template <typename T1, typename T2>
KOKKOS_INLINE_FUNCTION
bool roundProcessData(const T1* data, T2 combinedBytes, uint64_t len,  T1 errorValue, HashDigest* digests);

KOKKOS_INLINE_FUNCTION
bool roundinghash(const void* data, uint64_t len, const char dataType,
                double errorValue, HashDigest* digests)  {
  if (dataType == *("d")) {
    const double* doubleData = static_cast<const double*>(data);
    uint64_t bitsDataType = 0;
    return roundProcessData(doubleData, bitsDataType, len, errorValue, digests);
  } else if(dataType == *("f")) {
    const float* floatData = static_cast<const float*>(data);
    uint32_t bitsDataType = 0;
    return roundProcessData(floatData, bitsDataType, len, static_cast<float>(errorValue), digests);
  } else {
    kokkos_murmur3::MurmurHash3_x64_128(data, len, 0, &(digests[0]));
    return false;
  }
}

template <typename T1, typename T2>
KOKKOS_INLINE_FUNCTION
bool roundProcessData(const T1* data, T2 bitsDataType, uint64_t len, T1 errorValue, HashDigest* digests) {
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
//      dataUpper[j] = (dataUpper[j]*(1.0/errorValue))*(errorValue);
//      dataUpper[j] = round(dataUpper[j]*(1.0/errorValue))*(errorValue);
//      dataLower[j] = floor(dataLower[j]*(1.0/errorValue))*(errorValue);
//      dataUpper[j] = ceil(dataUpper[j]*(1.0/errorValue))*(errorValue);
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
  // ------------------------------------------------------------------------
  // Handled data at the tail for fuzzy hash
  // ------------------------------------------------------------------------
//  if ( (seedUpper[0] != seedLower[0]) || (seedUpper[1] != seedLower[1]) ) {
//    // The hashes are not identical. We need them both to proceed.
//    return true;
//  }
  return false;
}

#endif // ROUNDING_HASH_HPP
