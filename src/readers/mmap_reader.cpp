#include "mmap_reader.hpp"
#include <limits>

mmap_io_reader_t::mmap_io_reader_t() {
}

mmap_io_reader_t::~mmap_io_reader_t() {
    wait_all();   
}

mmap_io_reader_t::mmap_io_reader_t(std::string& name) {
    fname = name;
    fd = open(name.c_str(), O_RDONLY);
    if (fd == -1) {
        FATAL("cannot open " << fname << ", error = " << std::strerror(errno));
    }
    fsize = lseek(fd, 0, SEEK_END);
    buffer = (uint8_t *) mmap(NULL, fsize, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (buffer == MAP_FAILED)
        FATAL("cannot mmap " << fname << ", error (" << errno << ") = " << std::strerror(errno));
    madvise(buffer, fsize, MADV_RANDOM);
}

int mmap_io_reader_t::read_data(size_t beg, size_t end) {
    for (size_t i = beg; i < end; i++) {
        segment_t& seg = reads[i];
        if(seg.size + seg.offset > fsize)
          seg.size = fsize - seg.offset;
        memcpy(seg.buffer, buffer+seg.offset, seg.size);
        segment_status[i] = true;
    }
    return 0;
}

int mmap_io_reader_t::enqueue_reads(const std::vector<segment_t>& segments) {
    segment_status.insert(segment_status.end(), segments.size(), false);
    reads.insert(reads.end(), segments.begin(), segments.end());
    size_t per_thread = segments.size() / num_threads;
    if(per_thread*num_threads < segments.size())
        per_thread += 1;
    for(size_t i=0; i<(size_t)num_threads; i++) {
        size_t beg = i*per_thread;
        size_t end = (i+1)*per_thread;
        if(end > segments.size())
            end = segments.size();
        futures.push_back(std::async(std::launch::async | std::launch::deferred,
                                     &mmap_io_reader_t::read_data, this, beg, end));
    }
    return 0;
}

int mmap_io_reader_t::wait(size_t id) {
    size_t pos = 0;
    for(pos=0; pos < reads.size(); pos++) {
        // if(reads[pos].id == id)
        if(reads[pos].offset == id)
            break;
    }
    while(segment_status[pos] == false) {
      std::this_thread::sleep_for(std::chrono::duration<double, std::milli>(10));
    }
    return 0;
}

int mmap_io_reader_t::wait_all() {
    for (uint32_t i = 0; i < futures.size(); i++) {
        int res = futures[i].get();
        if (res < 0)
            fprintf(stderr, "read_chunks: %s\n", strerror(-res));
    }
    futures.clear();
    reads.clear();
    segment_status.clear();
    return 0;
}

size_t mmap_io_reader_t::wait_any() {
    size_t id = std::numeric_limits<size_t>::max();;
    do {
        for(size_t pos=0; pos<reads.size(); pos++) {
            if(segment_status[pos]) {
                // id = reads[pos].id;
                id = reads[pos].offset;
            }
        }
    } while(id == std::numeric_limits<size_t>::max());
    return id;
}
