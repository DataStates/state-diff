#ifndef __THREADED_CMP
#define __THREADED_CMP

#define __DEBUG
#include "debug.hpp"

#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <atomic>
#include <omp.h>

template <typename T> class threaded_cmp_t {
    size_t map_file(const std::string &fn, T* &buff) {
        int fd = open(fn.c_str(), O_RDONLY | O_DIRECT);
	if (fd == -1)
	    FATAL("cannot open " << fn << ", error = " << std::strerror(errno));
	size_t size = lseek(fd, 0, SEEK_END);
        ASSERT(size % sizeof(T) == 0);
	buff = (T *)mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);
	close(fd);
	if (buff == MAP_FAILED)
	    FATAL("cannot mmap " << fn << ", error = " << std::strerror(errno));
	return size / sizeof(T);
    }
    inline size_t get_chunk_size(size_t offset) {
	ASSERT(offset < ckpt_size);
	return offset + chunk_size >= ckpt_size ? ckpt_size - offset : chunk_size;
    }

    T *first, *second;
    size_t ckpt_size, chunk_size;
    double error;
    std::vector<size_t> offsets;

public:
    threaded_cmp_t(const std::string &f1, const std::string &f2, const std::vector<size_t> &off, size_t cs, double err) : chunk_size(cs), error(err), offsets(off) {
	ckpt_size = map_file(f1, first);
	[[maybe_unused]] size_t second_size = map_file(f2, second);
	ASSERT(ckpt_size == second_size);
	INFO("mapping " << ckpt_size  << " floating points from the runs");
    }
    size_t run_comparisons() {
	std::atomic<size_t> mismatches{0};
	size_t no_chunks = offsets.size();
        #pragma omp parallel for
	for (size_t i = 0; i < no_chunks; i++) {
	    T *first_chunk = first + offsets[i];
            T *second_chunk = second + offsets[i];
	    size_t len = get_chunk_size(offsets[i]);
	    for (size_t j = 0; j < len; j++)
		if (std::abs(first_chunk[j] - second_chunk[j]) > error)
		    mismatches++;
	}
	return mismatches;
    }
    ~threaded_cmp_t() {
	munmap(first, ckpt_size * sizeof(T));
	munmap(second, ckpt_size * sizeof(T));
    }

};

#endif // __THREADED_CMP
