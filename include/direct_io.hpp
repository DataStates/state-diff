#ifndef __DIRECT_IO_HPP
#define __DIRECT_IO_HPP

#include <string>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include "debug.hpp"

int get_file_size(const std::string& filename, off_t *size) {
    struct stat st;

    if (stat(filename.c_str(), &st) < 0 )
        return -1;
    if(S_ISREG(st.st_mode)) {
        *size = st.st_size;
        return 0;
    } 
    return -1;
}

ssize_t aligned_direct_write(const std::string& filename, void* data, size_t size) {
  // Open file with Direct IO
  int fd = open(filename.c_str(), O_CREAT | O_TRUNC | O_WRONLY | O_DIRECT, 0644);
  if (fd == -1)
    FATAL("cannot open " << filename << ", error = " << std::strerror(errno));

  // Write data
  size_t transferred = 0, remaining = size;
  while (remaining > 0) {
  	auto ret = write(fd, (uint8_t*)(data) + transferred, remaining);
  	if (ret < 0)
  	    FATAL("cannot write " << size << " bytes to " << filename << " , error = " << std::strerror(errno));
  	remaining -= ret;
  	transferred += ret;
  }

  // Close file
  int ret = close(fd);
  if(ret == -1)
    FATAL("close failed for " << filename << ", error = " << std::strerror(errno));

  return transferred;  
}

ssize_t unaligned_direct_write(const std::string& filename, void* data, size_t size) {
  // Allocate aligned temporary buffer
  size_t pagesize = sysconf(_SC_PAGESIZE);
  size_t npages = size/pagesize;
  if(npages*pagesize < size)
    npages += 1;
  uint8_t* data_ptr = (uint8_t*) aligned_alloc(pagesize, npages*pagesize);
  memset(data_ptr, 0, npages*pagesize);
  // Copy aligned data into pointer
  memcpy(data_ptr, data, size);
  // Read with direct I/O
  ssize_t nwrite = aligned_direct_write(filename, data_ptr, pagesize*npages);
  // Free temporary buffer
  free(data_ptr);
  return nwrite;  
}

ssize_t aligned_direct_read(const std::string& filename, void* data, size_t size) {
  // Open file with Direct IO
  int fd = open(filename.c_str(), O_RDONLY | O_DIRECT, 0644);
  if (fd == -1)
    FATAL("cannot open " << filename << ", error = " << std::strerror(errno));

  // Read data
  ssize_t nread = read(fd, data, size);
  if(nread == -1)
    FATAL("read failed for " << filename << ", error = " << std::strerror(errno));
  if(nread != (ssize_t)size) {
    nread += read(fd, data, size-nread);
  }
  if(nread != (ssize_t)size) {
    FATAL("read returned " << nread << " bytes instead of " << size << std::endl);
  }

  // Close file
  int ret = close(fd);
  if(ret == -1)
    FATAL("close failed for " << filename << ", error = " << std::strerror(errno));

  return nread;  
}

ssize_t unaligned_direct_read(const std::string& filename, void* data, size_t size) {
  // Allocate aligned temporary buffer
  size_t pagesize = sysconf(_SC_PAGESIZE);
  uint8_t* data_ptr = (uint8_t*) aligned_alloc(pagesize, size);
  // Read with direct I/O
  ssize_t nread = aligned_direct_read(filename, data_ptr, size);
  // Copy aligned data into pointer
  memcpy(data, data_ptr, size);
  // Free temporary buffer
  free(data_ptr);
  return nread;  
}

#endif // __DIRECT_IO_HPP
