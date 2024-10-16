#ifndef __FUZZY_HASH_HPP
#define __FUZZY_HASH_HPP

#include <bits/stdc++.h>
#include <iostream>
#include "kokkos_murmur3.hpp"
#include "MurmurHash3.h"

#ifdef __USEKOKKOS
  #define KOKKOS_INLINE KOKKOS_INLINE_FUNCTION
#else
  #define KOKKOS_INLINE inline
#endif

template <typename T>
struct DataTypeInfo;

template <>
struct DataTypeInfo<double> {
  static const int MANTISSABITS = 52;
  static const int BIAS = 1023;
  using MaskType = std::uint64_t;
  static const MaskType EXPONENTMASK = 0x7FF; // 11 bits
  static const MaskType MANTISSAMASK = 0xFFFFFFFFFFFFF; // 52 bits (13 Fs with F = 4 bits)
};

template <>
struct DataTypeInfo<float> {
  static const int MANTISSABITS = 23;
  static const int BIAS = 127;
  using MaskType = std::uint32_t;
  static const MaskType EXPONENTMASK = 0xFF; // 8 bits
  static const MaskType MANTISSAMASK = 0x7FFFFF; // 23 bits
};

struct alignas(16) HashDigest {
  uint8_t digest[16] = {0};
};

// Hashing approach tolerant to variations of floating point numbers
template <typename T1, typename T2>
KOKKOS_INLINE
void processData(const T1* data, T2 combinedBytes, uint64_t len, float errorValue, HashDigest* digests, bool usekokkos);

template <typename T1, typename T2>
KOKKOS_INLINE
void processElement(T1& fpvalueLower, T1& fpvalueUpper, T2 bitsDataType, float errorValue);

// Keep MSBs and clear LSBs given exponents of float and error values
template <typename T1, typename T2>
KOKKOS_INLINE
bool clearLSBs(T1 fpvalue, T2 *bits, T2 mantissaFP, int numBitsToKeep);

// Check if the number is a NaN, Inf or other special cases
template <typename T1, typename T2>
KOKKOS_INLINE
bool specialCase(T1 fpvalue, T2 exponentBits, T2 mantissaBits);

// Function to add two uint32_t binary representations
template <typename T2>
KOKKOS_INLINE
void addBinary(T2 a, T2 b, T2* newFPbits);

template <typename T2>
KOKKOS_INLINE
void printBits(T2 value);

KOKKOS_INLINE
bool digests_same(const HashDigest& lhs, const HashDigest& rhs);

KOKKOS_INLINE
void printMismatch(std::string prefix, const HashDigest& dig);

// Main hash function used by the code
KOKKOS_INLINE
void fuzzyhash(const void* data, uint64_t len, const char dataType,
                float errorValue, HashDigest* digests, bool usekokkos)  {
  if (dataType == *("d")) {
    const double* doubleData = static_cast<const double*>(data);
    uint64_t bitsDataType = 0;
    return processData(doubleData, bitsDataType, len, errorValue, digests, usekokkos);
  } else {
    const float* floatData = static_cast<const float*>(data);
    uint32_t bitsDataType = 0;
    return processData(floatData, bitsDataType, len, errorValue, digests, usekokkos);
  }    
}

template <typename T1, typename T2>
KOKKOS_INLINE
void processData(const T1* data, T2 bitsDataType, uint64_t len, float errorValue, HashDigest* digests, bool usekokkos) {
  // Given that every data point has two hashes, compute the modified
  // binary representations per data point and compute the hashes
  // for the entire chunk in a streaming fashion.
  const size_t blockSize = 16;
  constexpr uint32_t elementsPerBlock = blockSize/sizeof(T1);
  uint32_t blocksPerChunk = len/blockSize;

  T1 dataLower[elementsPerBlock];
  T1 dataUpper[elementsPerBlock];
  uint64_t seedLower[2] = {0, 0};
  uint64_t seedUpper[2] = {0, 0};

  for(uint32_t i=0; i<blocksPerChunk; i++) {

    // Declare the digest for the current block
    uint64_t* digestLower = (uint64_t*)&digests[0];
    uint64_t* digestUpper = (uint64_t*)&digests[1];
    
    // Create a copy of data for the current block
    memcpy(dataLower, (const uint8_t*)(data)+i*blockSize, blockSize);
    memcpy(dataUpper, (const uint8_t*)(data)+i*blockSize, blockSize);

    // Process each element
    for(uint32_t j=0; j<elementsPerBlock; j++) {
      float tmp = dataLower[j];
      processElement(dataLower[j], dataUpper[j], bitsDataType, errorValue);
      // printf("Input Value: %.9f\n", tmp);
      // printf("After Truncation: %.9f & After Addition:: %.9f\n", dataLower[j], dataUpper[j]);
      // printf("Error -> Truncated: %.9f & Added: %.9f\n", Kokkos::fabs(tmp-dataLower[j]), Kokkos::fabs(tmp-dataUpper[j]));
    }
    
    // Compute the hashes for the current block of 128-bit data
    if( usekokkos ) {
      kokkos_murmur3::MurmurHash3_x64_128(dataLower, blockSize, seedLower, digestLower);
      kokkos_murmur3::MurmurHash3_x64_128(dataUpper, blockSize, seedUpper, digestUpper);
    } else {
      base_murmur3::MurmurHash3_x64_128(dataLower, blockSize, seedLower, digestLower);
      base_murmur3::MurmurHash3_x64_128(dataUpper, blockSize, seedUpper, digestUpper);
    }

    // Copy the content of digestLower and digestUpper into seedLower and seedUpper
    seedLower[0] = digestLower[0];
    seedLower[1] = digestLower[1];
    seedUpper[0] = digestUpper[0];
    seedUpper[1] = digestUpper[1];

    // printf("After Truncation: %lu and %lu\n", seedLower[0], seedLower[1]);
    // printf("After Addition: %lu and %lu\n", seedUpper[0], seedUpper[1]);
  }
}

// Hashing approach tolerant to variations of floating point numbers
template <typename T1, typename T2>
KOKKOS_INLINE
void processElement(T1& fpvalueLower, T1& fpvalueUpper, T2 bitsDataType, float errorValue) {

  using Info = DataTypeInfo<T1>;
  T1 fpvalue = fpvalueLower;
  T2* oribits = reinterpret_cast<T2 *>(&fpvalue);
  // Get the binary representation of the error value and its exponent
  T2* errorBits = reinterpret_cast<T2 *>(&errorValue);
  int errorExponent = (((*errorBits) >> Info::MANTISSABITS) & Info::EXPONENTMASK) - Info::BIAS;

  // Interpret the float value as a uint32_t to access its binary representation
  T2* bits = reinterpret_cast<T2 *>(&fpvalueLower);
  T2 exponentBits = ((*bits) >> Info::MANTISSABITS) & Info::EXPONENTMASK;
  T2 mantissaBits = (*bits) & Info::MANTISSAMASK;
  int exponent = exponentBits - Info::BIAS;
  T2* newFPbits = reinterpret_cast<T2 *>(&fpvalueUpper);
  if (specialCase(fpvalueLower, exponentBits, mantissaBits)) {
    // Verify that the numbers are not NaNs or infinity. 
    // If so, directly apply the murmur hash instead of the fuzzy hash
    *newFPbits = *bits;
  }
  else {
    int numBitsToKeep = exponent - errorExponent;
    bool unchangedMantissa = clearLSBs(fpvalueLower, bits, mantissaBits, numBitsToKeep);
    T2 bitmask;
    if (unchangedMantissa) {
      // sequence of 1s where #ofones = #LSBCleared
      bitmask = (1 << (Info::MANTISSABITS - numBitsToKeep)) - 1;
    } else {
      // sequence of 1 and 0s where #ofzeros = #LSBCleared
      bitmask = (1 << (Info::MANTISSABITS - numBitsToKeep));
    }
    addBinary(*bits, bitmask, newFPbits);
  }
}

template <typename T2>
KOKKOS_INLINE
void printBits(T2 value) {
  int numBits = sizeof(T2) * 8;
  for (int i = numBits - 1; i >= 0; i--) {
    T2 bit = (value >> i) & 1u;
    printf("%u", (uint32_t)bit);
  }
  printf("\n");
}

// Keep MSBs and clear LSBs given exponents of float and error values
template <typename T1, typename T2>
KOKKOS_INLINE
bool clearLSBs(T1 fpvalue, T2 *bits, T2 mantissaFP, int numBitsToKeep) {

  using Info = DataTypeInfo<T1>;
  T2 modifiedMantissa = 0b0;
  // drop least significant bits only if the number of bits to keep is smaller
  // than the total number of bits in the mantissa
  int M = Info::MANTISSABITS - numBitsToKeep;
  if (M <= Info::MANTISSABITS)    {      
    // Use bitwise left-shift to set the N most significant bits to 1
    T2 lowerboundmask = (1ULL << numBitsToKeep) - 1;

    // Left-shift the mask by M positions to add M zeros at the end
    lowerboundmask <<= M;

    // Using the AND operation between the mantissa and the bitmask to clear bits.
    modifiedMantissa |= (mantissaFP & lowerboundmask);

    // We now clear the original mantissa
    // and define the average as the new mantissa
    (*bits) &= ~Info::MANTISSAMASK;
    (*bits) |= modifiedMantissa;
  }
  return (mantissaFP == modifiedMantissa);
}

template <typename T1, typename T2>
KOKKOS_INLINE
bool specialCase(T1 fpvalue, T2 exponentBits, T2 mantissaBits) {
  /*
    If the exponent bits are all zeros and the mantissa bits are not all zeros.
    This condition identifies subnormal (or denormalized) numbers.
    These numbers are used to represent values close to zero that
    cannot be accurately represented using the regular normalized format.

    If the exponent bits are all ones.
    This condition is used to identify special values such as
    infinity and NaN (Not-a-Number) in IEEE 754.
    If the mantissa bits are all zeros, it typically represents positive or negative infinity.
    If the mantissa bits are not all zeros, it represents NaN.
  */
  using Info = DataTypeInfo<T1>;
  if ((exponentBits == 0 && mantissaBits != 0) || exponentBits == Info::EXPONENTMASK) {
    // std::cout << "Special case detected." << std::endl;
    return true;
  }

  int exponent = exponentBits - Info::BIAS;
  if ((exponent > Info::BIAS) || (exponent < 1 - Info::BIAS)) {
    // std::cout << "Mantissa Overflow/Underflow detected." << std::endl;
    return true;
  }
  return false;
}

// Function to add two T2 (uint32_t or uint64_t) binary representations
template <typename T2>
void addBinary(T2 a, T2 b, T2* newFPbits) {
  *newFPbits = 0;
  T2 carry = 0;
  // Iterate over each bit
  int numofbits = sizeof(T2)*8;
  for (int i = 0; i < numofbits; ++i) {
    T2 bitA = (a >> i) & 1;
    T2 bitB = (b >> i) & 1;
    // Calculate the sum and carry
    T2 sum = bitA ^ bitB ^ carry;
    carry = (bitA & bitB) | ((bitA ^ bitB) & carry);
    // Set the bit in the result
    (*newFPbits) |= (sum << i);
  }
  // If there is a carry after the loop, set it in the result
  (*newFPbits) |= (carry << (numofbits-1));
}

// Helper function for checking if two hash digests are identical
KOKKOS_INLINE
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

KOKKOS_INLINE
void printMismatch(const HashDigest& dig) {
  uint64_t* dig_ptr = (uint64_t*)(dig.digest);
  for(size_t i=0; i<sizeof(HashDigest)/8; i++) {
    printf("%lu ", dig_ptr[i]);
  }
  printf("\n--------------------------\n");
}

int areAbsoluteEqual(const float* data_a, const float* data_b, int size, float error, int id) {
  int result = 0;
  for (int i = 0; i < size & result < 1; ++i) {
    if (fabs(data_a[i] - data_b[i]) > error) {
      result = 1;
    }
  }
  return result;
}

#endif // __FUZZY_HASH_HPP
