#ifndef UTILS_HPP
#define UTILS_HPP

#include <Kokkos_Core.hpp>

//#define STDOUT
//#define DEBUG
//#define STATS

#ifdef DEBUG
#define DEBUG_PRINT(...) do{ printf( __VA_ARGS__ ); } while( false )
#else
#define DEBUG_PRINT(...) do{ } while ( false )
#endif

#ifdef STDOUT
#define STDOUT_PRINT(...) do{ printf( __VA_ARGS__ ); } while( false )
#else
#define STDOUT_PRINT(...) do{ } while ( false )
#endif

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

#ifdef __NVCC__
#include "cuda.h"
#include "cuda_runtime.h"

#define DEBUG_GPU
#ifdef DEBUG_GPU
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
#else
#define gpuErrchk(ans) ans
#endif

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

template<typename T>
T* device_alloc(size_t len) {
  T* device_ptr = NULL;
  gpuErrchk( cudaMalloc(&device_ptr, len) );
  return device_ptr;
}

template<typename T>
T* host_alloc(size_t len) {
  T* host_ptr = NULL;
  gpuErrchk( cudaHostAlloc(&host_ptr, len, cudaHostAllocDefault) );
  return host_ptr;
}

template<typename T>
void device_free(T* device_ptr) {
  gpuErrchk( cudaFree(device_ptr) );
}

template<typename T>
void host_free(T* host_ptr) {
  gpuErrchk( cudaFreeHost(host_ptr) );
}

template<typename Stream>
void create_stream(Stream& stream) {
  gpuErrchk( cudaStreamCreate(&stream) );
}
#else
template<typename T>
T* device_alloc(size_t len) {
  T* device_ptr = (T*) malloc(len);;
  return device_ptr;
}

template<typename T>
T* host_alloc(size_t len) {
  T* host_ptr = (T*) malloc(len);
  return host_ptr;
}

template<typename T>
void device_free(T* device_ptr) {
  free(device_ptr);
}

template<typename T>
void host_free(T* host_ptr) {
  free(host_ptr);
}
#endif


#endif
