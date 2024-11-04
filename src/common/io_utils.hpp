#ifndef __IO_UTILS_HPP
#define __IO_UTILS_HPP

#include <chrono>

using Timer = std::chrono::high_resolution_clock;
using Duration = std::chrono::duration<double>;


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