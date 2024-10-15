#include <stdio.h>
#include <map>
#include "liburing_reader.hpp"

//#define BACKGROUND_THREAD

liburing_io_reader_t::liburing_io_reader_t() {}

liburing_io_reader_t::~liburing_io_reader_t() {
    // Wait for any remaining operations
    wait_all();   
#ifdef BACKGROUND_THREAD
    // Flag background thread that everything is done
    std::unique_lock<std::mutex> lk(m);
//    cv.wait(lk, [this] { return (submissions.size() == 0) && (req_completed[0] == req_submitted[0]); });
    // Signal IO thread to be inactive
    active = false;
    lk.unlock();
    cv.notify_one();
    // Join background thread
    th.join();
#endif
    // Destory queue
    for(size_t i=0; i<nrings; i++) {
        io_uring_queue_exit(&ring[i]);
    }
    // Close file
    close(fd);
    delete[] ring;
    delete[] req_completed;
    delete[] req_submitted;
}

liburing_io_reader_t::liburing_io_reader_t(std::string& name, size_t num_rings) {
    // Open file
    nrings = num_rings;
    ring = new io_uring[nrings];
    req_submitted = new size_t[nrings];
    req_completed = new size_t[nrings];
    fname = name;
    fd = open(name.c_str(), O_RDONLY);
    if (fd == -1) {
        FATAL("cannot open " << fname << ", error = " << std::strerror(errno));
    }

    // Get file size
    fsize = lseek(fd, 0, SEEK_END);
    lseek(fd, 0, SEEK_SET);

    // Initialize ring 
//    struct io_uring_params params;
//    memset(&params, 0, sizeof(params));
//    params.flags |= IORING_SETUP_SQPOLL;
//    params.sq_thread_idle = 2000;
//
//    int ret = io_uring_queue_init_params(MAX_RING_SIZE, &ring, &params);
    for(size_t i=0; i<nrings; i++) {
        int ret = io_uring_queue_init(MAX_RING_SIZE, &ring[i], 0);
        if (ret < 0) {
            FATAL("queue_init: " << std::strerror(-ret));
        }
        req_submitted[i] = 0;
        req_completed[i] = 0;
    }

#ifdef BACKGROUND_THREAD
    // Launch background thread for submitting/completing reads
    active = true;
    wait_all_mode = false;
    th = std::thread(&liburing_io_reader_t::io_thread, this);
#endif
    return;
}

// Process any available completions
uint32_t liburing_io_reader_t::request_completion() {
    uint32_t tot_ready = 0;

//    std::multimap<uint32_t,size_t> req_ready;
//    for(size_t i=0; i<nrings; i++) {
//        req_ready.insert(std::pair{io_uring_cq_ready(&ring[i]), i});
//    }
//    
//    for(const auto& entry : req_ready) {
//        uint32_t nwait = entry.first;
//        size_t i = entry.second;
//printf("%u requests ready for ring %zu\n", nwait, i);

//    std::multimap<uint32_t,size_t> req_ready;
//    size_t ndone = 0;
//    while(ndone < nrings) {
//        req_ready.clear();
//        for(size_t i=0; i<nrings; i++) {
//            req_ready.insert(std::pair{io_uring_cq_ready(&ring[i]), i});
//        }
//    }

    for(size_t i=0; i<nrings; i++) {
        uint32_t nwait = MAX_RING_SIZE/8;
        if(req_submitted[i] - req_completed[i] < nwait)
            nwait = (req_submitted[i] - req_completed[i]);
//        uint32_t nwait = io_uring_cq_ready(&ring[i]);

        if(nwait > 0) {
            int ret = io_uring_wait_cqe_nr(&ring[i], &cqe[0], nwait);
            if(ret < 0) {
                fprintf(stderr, "io_uring_wait_cqe(&ring, &cqe[0]): %s\n", strerror(-ret));
            }
            uint32_t nready = 0;
            uint32_t head;
            struct io_uring_cqe *cqe_ptr;
            io_uring_for_each_cqe(&ring[i], head, cqe_ptr) {
                size_t id = io_uring_cqe_get_data64(cqe_ptr);
                completions.insert(id);
                nready += 1;
            }
//printf("Completing %u requests for ring 1\n", nready);
            // Update ring and count of total requests completed
            io_uring_cq_advance(&ring[i], nready);
            req_completed[i] += nready;
            tot_ready += nready;
//printf("%u requests completed for ring %zu\n", nready, i);
        }
    }
    return tot_ready;
}

uint32_t liburing_io_reader_t::request_submission() {
    uint32_t tot_submit = 0;
    uint32_t per_ring = submissions.size() / nrings;
    if(per_ring*nrings < submissions.size())
        per_ring += 1;
printf("Per ring submissions: %u\n", per_ring);
    for(size_t i=0; i<nrings; i++) {
//        uint32_t nsubmit = MAX_RING_SIZE/8;
        uint32_t nsubmit = per_ring < MAX_RING_SIZE ? per_ring : MAX_RING_SIZE;
        if(req_submitted[i] < req_completed[i])
            nsubmit = req_completed[i] - req_submitted[i];
        if(nsubmit > submissions.size())
            nsubmit = submissions.size();
    
        if(nsubmit > 0) {
            // Prep reads
            for(size_t j=0; j<nsubmit; j++) {
                // Get segment from queue
                segment_t seg = submissions.front();
                if(seg.size + seg.offset > fsize)
                    seg.size = fsize - seg.offset;
                // Prepare submission queue entry
                auto sqe = io_uring_get_sqe(&ring[i]);
                if (!sqe) {
                    fprintf(stderr, "Could not get sqe\n");
                    return -1;
                }
                // Save ID
                io_uring_sqe_set_data64(sqe, seg.id);
                io_uring_prep_read(sqe, fd, seg.buffer, seg.size, seg.offset);
                // Remove segment from queue
                submissions.pop();
            }
//printf("Submitting %u requests for ring %zu\n", nsubmit, i);
    
            // Submit queued reads
            int ret = io_uring_submit(&ring[i]);
            if(ret < 0) {
                fprintf(stderr, "io_uring_submit: %s\n", strerror(-ret));
            }
        
            // Update count of submitted requests
            req_submitted[i] += nsubmit;
            tot_submit += nsubmit;
        }
    }
    return tot_submit;
}

int liburing_io_reader_t::io_thread() {
    int ret;
    uint32_t avail_cqe=MAX_RING_SIZE;
    while(active) {
        // Pause until one of three scenarios
        // 1. No more work
        // 2. Active flag disabled by destructor
        // 3. Done submitting request but some completions need to be done
        std::unique_lock lk(m);
        auto completed = 0, submitted = 0;
        for(size_t i=0; i<nrings; i++) {
            completed += req_completed[i];
            submitted += req_submitted[i];
        }
        cv.wait(lk, [this, submitted, completed] { return !active || (submissions.size() > 0) || ((submissions.size() == 0) && (submitted > completed)); });
        // Destructor called
        if(!active) {
            lk.unlock();
            cv.notify_one();
            break;
        }
        // Process completions and submit buffered segments
        // If doing a wait all then loop to avoid unnecessary wake ups
        do {
            // Process any available completions
            uint32_t nready = request_completion();
    
            // Need to finish up remaining completions
            
            completed = 0, submitted = 0;
            for(size_t i=0; i<nrings; i++) {
                completed += req_completed[i];
                submitted += req_submitted[i];
            }

            if( (submissions.size() == 0) && (submitted > completed) ) {
                avail_cqe += nready;
                if(submitted == completed) {
                    break;
                } else {
                    continue;
                }
            }
    
            uint32_t nsubm = request_submission();

            completed = 0, submitted = 0;
            for(size_t i=0; i<nrings; i++) {
                completed += req_completed[i];
                submitted += req_submitted[i];
            }
//            // Calculate how many reads to submit
//            avail_cqe += nready;
//            size_t num_reads = submissions.size();
//            if(num_reads > avail_cqe) 
//                num_reads = avail_cqe;
//    
//            // Prep reads
//            for(size_t i=0; i<num_reads; i++) {
//                // Get segment from queue
//                segment_t seg = submissions.front();
//                if(seg.size + seg.offset > fsize)
//                    seg.size = fsize - seg.offset;
//    
//                // Prepare submission queue entry
//                auto sqe = io_uring_get_sqe(&ring[0]);
//                if (!sqe) {
//                    fprintf(stderr, "Could not get sqe\n");
//                    return -1;
//                }
//                // Save ID
//                io_uring_sqe_set_data64(sqe, seg.id);
//                io_uring_prep_read(sqe, fd, seg.buffer, seg.size, seg.offset);
//    
//                // Remove segment from queue
//                submissions.pop();
//            }
//            avail_cqe -= num_reads;
//            req_submitted[0] += num_reads;
//    
//            // Submit queued reads
//            ret = io_uring_submit(&ring[0]);
//            if(ret < 0) {
//                fprintf(stderr, "io_uring_submit: %s\n", strerror(-ret));
//            }
            // Loop if waiting for all segments 
        } while(wait_all_mode && ((submissions.size() > 0) || submitted > completed));

        // Acknowledge work and let main thread continue
        lk.unlock();
        cv.notify_one();
    }
    return 0;
}

int liburing_io_reader_t::enqueue_reads(const std::vector<segment_t>& segments) {
#ifdef BACKGROUND_THREAD
    // Acquire lock
    std::unique_lock<std::mutex> lock(m);

    // Push segments onto queue and add ID to the in progress set
    for(size_t i=0; i<segments.size(); i++) {
        submissions.push(segments[i]);
    }
    
    // Notify background thread that there is work
    lock.unlock();
    cv.notify_one();
#else
    // Push segments onto queue and add ID to the in progress set
    for(size_t i=0; i<segments.size(); i++) {
        submissions.push(segments[i]);
    }

    request_submission();
#endif
    
    return 0;
}

int liburing_io_reader_t::wait(size_t id) {
#ifdef BACKGROUND_THREAD
    // Acquire lock
    std::unique_lock<std::mutex> lock(m);
    // Search for ID in already completed requests
    cv.wait(lock, [this, id] { return completions.find(id) != completions.end(); });
    // Remove ID from vector of completed request IDs
    completions.erase(id);
    lock.unlock();
    cv.notify_one();
#else
    while(completions.find(id) == completions.end()) {
        uint32_t ncomp = request_completion();
        uint32_t nsubm = request_submission();
    }
    completions.erase(id);
#endif
    return 0;
}

int liburing_io_reader_t::wait_all() {
#ifdef BACKGROUND_THREAD
    // Wait till there are no queued submissions and all requests are done
    std::unique_lock<std::mutex> lock(m);
    cv.wait(lock, [this] { return (submissions.size() == 0) && (req_submitted[0] == req_completed[0]); });
    // Clear completion set
    completions.clear();
    wait_all_mode = true;
    lock.unlock();
    cv.notify_one();
#else
    double com_time = 0, sub_time = 0, check_time = 0, clear_time = 0;
    Timer::time_point check_beg = Timer::now();
    bool req_done = true;
    for(size_t i=0; i<nrings; i++) {
        req_done = req_done && (req_submitted[i] == req_completed[i]);
    }
    Timer::time_point check_end = Timer::now();
    check_time += 
        std::chrono::duration_cast<Duration>(check_end - check_beg).count();
    while( !( (submissions.size() == 0) && req_done ) ) {
        Timer::time_point com_beg = Timer::now();
        uint32_t ncomp = request_completion();
        Timer::time_point com_end = Timer::now();

        Timer::time_point sub_beg = Timer::now();
        uint32_t nsubm = request_submission();
        Timer::time_point sub_end = Timer::now();

        Timer::time_point check_beg = Timer::now();
        req_done = true;
        for(size_t i=0; i<nrings; i++) {
            req_done = req_done && (req_submitted[i] == req_completed[i]);
        }
        Timer::time_point check_end = Timer::now();

        com_time += 
            std::chrono::duration_cast<Duration>(com_end - com_beg).count();
        sub_time += 
            std::chrono::duration_cast<Duration>(sub_end - sub_beg).count();
        check_time += 
            std::chrono::duration_cast<Duration>(check_end - check_beg).count();
    } 
    Timer::time_point clear_beg = Timer::now();
    completions.clear();
    Timer::time_point clear_end = Timer::now();
    clear_time += 
        std::chrono::duration_cast<Duration>(clear_end - clear_beg).count();
    printf("\tCheck time:      %f\n", check_time);
    printf("\tSubmission time: %f\n", sub_time);
    printf("\tCompletion time: %f\n", com_time);
    printf("\tClear time:      %f\n", clear_time);
#endif
    return 0;
}

size_t liburing_io_reader_t::wait_any() {
#ifdef BACKGROUND_THREAD
    // Wait for any request to finish
    std::unique_lock<std::mutex> lock(m);
    cv.wait(lock, [this] { return completions.size() > 0; });
    // Get ID of finished op and remove from set
    size_t id = *(completions.begin());
    completions.erase(completions.begin());
    lock.unlock();
    cv.notify_one();
#else
    while(completions.size() == 0) {
        uint32_t ncomp = request_completion();
        uint32_t nsubm = request_submission();
    }
    size_t id = *(completions.begin());
    completions.erase(completions.begin());
#endif
    return id;
}

