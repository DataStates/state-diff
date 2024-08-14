#include "posix_reader.hpp"

posix_io_reader_t::posix_io_reader_t() {
}

posix_io_reader_t::~posix_io_reader_t() {
    wait_all();   
    close(fd);
}

posix_io_reader_t::posix_io_reader_t(std::string& name) {
    fname = name;
    fd = open(name.c_str(), O_RDONLY | O_DIRECT);
    if (fd == -1) {
        FATAL("cannot open " << fname << ", error = " << std::strerror(errno));
    }
    fsize = lseek(fd, 0, SEEK_END);
    lseek(fd, 0, SEEK_SET);
}

int posix_io_reader_t::read_data(size_t beg, size_t end) {
    for (size_t i = beg; i < end; i++) {
        segment_t& seg = reads[i];
        ssize_t len = 0;
        if(seg.size + seg.offset > fsize)
          seg.size = fsize - seg.offset;
        while(len < seg.size) {
            ssize_t data_read = pread(fd, seg.buffer+len, seg.size-len, seg.offset+len); 
            if(data_read < 0) {
                FATAL("pread(fd, " << seg.buffer+len << ", " << seg.size-len << ", " 
                      << seg.offset+len << ") failed" << fname << ", error = " 
                      << std::strerror(errno));
            } else {
                len += data_read;
            }
        }
        in_progress.erase(in_progress.find(seg.id));
    }
    return 0;
}

int posix_io_reader_t::enqueue_reads(const std::vector<segment_t>& segments) {
    size_t start = reads.size();
    reads.insert(reads.end(), segments.begin(), segments.end());
    for(const segment_t& seg : segments) {
        in_progress.insert(seg.id);
    }
    size_t per_thread = segments.size() / num_threads;
    if(per_thread*num_threads < segments.size())
        per_thread += 1;
printf("Launch %d threads with %zu segments per thread\n", num_threads, per_thread);
    for(size_t i=0; i<num_threads; i++) {
        size_t beg = i*per_thread;
        size_t end = (i+1)*per_thread;
        if(end > segments.size())
            end = segments.size();
        futures.push_back(std::async(std::launch::async | std::launch::deferred,
                                     &posix_io_reader_t::read_data, this, beg, end));
    }
printf("Launched async tasks\n");
    return 0;
}

int posix_io_reader_t::wait(size_t id) {
    while(in_progress.find(id) != in_progress.end()) {
      std::this_thread::sleep_for(std::chrono::duration<double, std::milli>(10));
    }
    return 0;
}

int posix_io_reader_t::wait_all() {
printf("Trying to call get on %lu futures\n", futures.size());
    for (uint32_t i = 0; i < futures.size(); i++) {
        int res = futures[i].get();
printf("Called get on future %u\n", i);
        if (res < 0)
            fprintf(stderr, "read_chunks: %s\n", strerror(-res));
    }
    futures.clear();
    reads.clear();
    return 0;
}
