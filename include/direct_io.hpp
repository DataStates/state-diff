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

int try_free_page_cache(const std::string& filename) {
  int fd, ret=0;
  fd = open(filename.c_str(), O_RDONLY);
  if (fd == -1)
    FATAL("cannot open " << filename << ", error = " << std::strerror(errno));
  ret = fsync(fd);
  if(ret == -1)
    FATAL("fsync failed for " << filename << ", error = " << std::strerror(errno));
  ret = posix_fadvise(fd, 0,0,POSIX_FADV_DONTNEED);
  if(ret != 0)
    FATAL("fadvise POSIX_FADV_DONTNEED failed for " << filename << ", error = " << std::strerror(ret));
  close(fd);
  return ret;
}

ssize_t aligned_direct_write(const std::string& filename, void* data, size_t size) {
  int ret;
  size_t pagesize = sysconf(_SC_PAGESIZE);
  size_t direct_size = size - (size % pagesize);
printf("pagesize: %zu, size: %zu, direct_size: %zu\n", pagesize, size, direct_size);

  // Write majority of data with O_DIRECT

  // Open file with Direct IO
  int fd = open(filename.c_str(), O_CREAT | O_TRUNC | O_WRONLY, 0644);
  if (fd == -1)
    FATAL("cannot open " << filename << ", error = " << std::strerror(errno));

  // Write data
  size_t transferred = 0, remaining = direct_size;
  while (remaining > 0) {
  	auto bytes_written = write(fd, (uint8_t*)(data) + transferred, remaining);
  	if (bytes_written < 0)
  	    FATAL("cannot write " << size << " bytes to " << filename << " , error = " << std::strerror(errno));
  	remaining -= bytes_written;
  	transferred += bytes_written;
  }

  // Close file
  ret = fsync(fd);
  if(ret == -1)
    FATAL("fsync failed for " << filename << ", error = " << std::strerror(errno));
   
  posix_fadvise(fd, 0,0,POSIX_FADV_DONTNEED);
  ret = close(fd);
  if(ret == -1)
    FATAL("close failed for " << filename << ", error = " << std::strerror(errno));
printf("Wrote %zu bytes directly to file, %zu bytes remaining.\n", transferred, size-transferred);

  // Write remaining bytes if any
  if(transferred < size) {
    fd = open(filename.c_str(), O_CREAT | O_TRUNC | O_WRONLY, 0644);
    if (fd == -1)
      FATAL("cannot open " << filename << ", error = " << std::strerror(errno));

    // Seek previous starting point
    ret = lseek(fd, transferred, SEEK_SET);
    if(ret == -1)
      FATAL("lseek failed for " << filename << ", error = " << std::strerror(errno));

    // Write data
    remaining = size - transferred;
    while (remaining > 0) {
    	auto bytes_written = write(fd, (uint8_t*)(data) + transferred, remaining);
    	if (bytes_written < 0)
    	    FATAL("cannot write " << size << " bytes to " << filename << " , error = " << std::strerror(errno));
    	remaining -= bytes_written;
    	transferred += bytes_written;
    }

    // Close file
    ret = fsync(fd);
    if(ret == -1)
      FATAL("fsync failed for " << filename << ", error = " << std::strerror(errno));
     
    // Tell kernel to drop cached pages
    posix_fadvise(fd, 0,0,POSIX_FADV_DONTNEED);

    ret = close(fd);
    if(ret == -1)
      FATAL("close failed for " << filename << ", error = " << std::strerror(errno));
printf("Wrote %zu bytes to file, %zu bytes remaining.\n", transferred, size-transferred);
  }

  return transferred;  
}

ssize_t unaligned_direct_write(const std::string& filename, void* data, size_t size) {
  // Allocate aligned temporary buffer
  size_t pagesize = sysconf(_SC_PAGESIZE);
  size_t npages = size/pagesize;
  if(npages*pagesize < size)
    npages += 1;
  uint8_t* data_ptr = (uint8_t*) aligned_alloc(pagesize, npages*pagesize);
  if(data_ptr == NULL)
    FATAL("aligned_alloc(" << pagesize << ", " << npages*pagesize << ") failed, "<< ", error = " << std::strerror(errno));
  memset(data_ptr, 0, npages*pagesize);
  // Copy aligned data into pointer
  memcpy(data_ptr, data, size);
  // Read with direct I/O
  ssize_t nwrite = aligned_direct_write(filename, data_ptr, size);
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
  ssize_t total_read = nread;
  if(nread == -1)
    FATAL("read failed for " << filename << ", error = " << std::strerror(errno));
  while((total_read != (ssize_t)size) && (nread != -1)) {
    nread = read(fd, data, size-nread);
    total_read += nread;
  }
  if(total_read == -1) {
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
  size_t npages = size/pagesize;
  if(npages*pagesize < size)
    npages += 1;
  uint8_t* data_ptr = (uint8_t*) aligned_alloc(pagesize, npages*pagesize);
  // Read with direct I/O
  ssize_t nread = aligned_direct_read(filename, data_ptr, npages*pagesize);
  // Copy aligned data into pointer
  memcpy(data, data_ptr, size);
  // Free temporary buffer
  free(data_ptr);
  return nread;  
}

#endif // __DIRECT_IO_HPP
