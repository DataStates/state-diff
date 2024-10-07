#include <stdio.h>
#include "liburing_reader.hpp"

//#define BACKGROUND_THREAD

liburing_io_reader_t::liburing_io_reader_t() {}

liburing_io_reader_t::~liburing_io_reader_t() {
    // Wait for any remaining operations
    wait_all();   
#ifdef BACKGROUND_THREAD
    // Flag background thread that everything is done
    std::unique_lock<std::mutex> lk(m);
    cv.wait(lk, [this] { return (submissions.size() == 0) && (req_completed == req_submitted); });
    // Signal IO thread to be inactive
    active = false;
    lk.unlock();
    cv.notify_one();
    // Join background thread
    th.join();
#endif
    // Destory queue
    io_uring_queue_exit(&ring);
    io_uring_queue_exit(&ring2);
    // Close file
    close(fd);
}

liburing_io_reader_t::liburing_io_reader_t(std::string& name) {
    // Open file
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
    int ret = io_uring_queue_init(MAX_RING_SIZE, &ring, 0);
    if (ret < 0) {
        FATAL("queue_init: " << std::strerror(-ret));
    }
    ret = io_uring_queue_init(MAX_RING_SIZE, &ring2, 0);
    if (ret < 0) {
        FATAL("queue_init: " << std::strerror(-ret));
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
    // Get # of ready completions
    //uint32_t nready = io_uring_cq_ready(&ring); 
//    uint32_t nwait = 32768/2;
//    if(req_submitted - req_completed < nwait)
//        nwait = req_submitted - req_completed;
//    int ret = io_uring_wait_cqe_nr(&ring, &cqe[0], nwait);
//    if(ret < 0) {
//        fprintf(stderr, "io_uring_wait_cqe_nr(&ring, &cqe[0], %u): %s\n", nwait, strerror(-ret));
//        return ret;
//    }
//    uint32_t nready = nwait;
//    if(nready > 0) {
//        uint32_t head;
//        struct io_uring_cqe* temp_cqe;
//        io_uring_for_each_cqe(&ring, head, temp_cqe) {
//            size_t id = io_uring_cqe_get_data64(temp_cqe);
//            completions.insert(id);
//        }
//
////        uint32_t ret = io_uring_peek_batch_cqe(&ring, &cqe[0], nready);
////        if(ret != nready) {
////            fprintf(stderr, "io_uring_peek_batch_cqe: %u\n vs %u", ret, nready);
////            return ret;
////        }
////        // Get IDs of completed segments and push to completion vector
////        for(uint32_t i=0; i<nready; i++) {
////            size_t id = io_uring_cqe_get_data64(cqe[i]);
////            completions.insert(id);
////        }
//        // Update ring and count of total requests completed
//        io_uring_cq_advance(&ring, nready);
//        req_completed += nready;
//    }

    // Ring 1
    uint32_t nwait = 32768/2;
//    uint32_t nwait = io_uring_cq_ready(&ring);;
    if(req_submitted - req_completed < nwait)
        nwait = req_submitted - req_completed;

    int ret = io_uring_wait_cqe_nr(&ring, &cqe[0], nwait/2);
    if(ret < 0) {
        fprintf(stderr, "io_uring_wait_cqe(&ring, &cqe[0]): %s\n", strerror(-ret));
    }
    uint32_t nready = 0;
    uint32_t head;
    struct io_uring_cqe *cqe_ptr;
    io_uring_for_each_cqe(&ring, head, cqe_ptr) {
        size_t id = io_uring_cqe_get_data64(cqe_ptr);
        completions.insert(id);
        nready += 1;
    }
    // Update ring and count of total requests completed
    io_uring_cq_advance(&ring, nready);

    // Ring 2
    if(req_submitted - req_completed < nwait)
        nwait = req_submitted - req_completed;
    ret = io_uring_wait_cqe_nr(&ring2, &cqe[0], nwait/2);
    io_uring_for_each_cqe(&ring2, head, cqe_ptr) {
        size_t id = io_uring_cqe_get_data64(cqe_ptr);
        completions.insert(id);
        nready += 1;
    }
    // Update ring and count of total requests completed
    io_uring_cq_advance(&ring2, nready);

    req_completed += nready;
    return nready;
}

uint32_t liburing_io_reader_t::request_submission() {
    // Get # of ready completions
//    uint32_t nsubmit = io_uring_sq_space_left(&ring); 
//    uint32_t nsubmit = io_uring_sqring_wait(&ring);
    uint32_t nsubmit = MAX_RING_SIZE;
    if(req_submitted < req_completed)
        nsubmit = req_completed - req_submitted;
    if(nsubmit > submissions.size())
        nsubmit = submissions.size();
    if(nsubmit > 0) {
        // Prep reads
        for(size_t i=0; i<nsubmit; i++) {
            // Get segment from queue
            segment_t seg = submissions.front();
            if(seg.size + seg.offset > fsize)
                seg.size = fsize - seg.offset;
    
            // Prepare submission queue entry
            auto sqe = io_uring_get_sqe(&ring);
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

        // Submit queued reads
        int ret = io_uring_submit(&ring);
        if(ret < 0) {
            fprintf(stderr, "io_uring_submit: %s\n", strerror(-ret));
        }
    
        // Update count of submitted requests
        req_submitted += nsubmit;
    }

    if(req_submitted < req_completed)
        nsubmit = req_completed - req_submitted;
    if(nsubmit > submissions.size())
        nsubmit = submissions.size();
    if(nsubmit > 0) {
        // Prep reads
        for(size_t i=0; i<nsubmit; i++) {
            // Get segment from queue
            segment_t seg = submissions.front();
            if(seg.size + seg.offset > fsize)
                seg.size = fsize - seg.offset;
    
            // Prepare submission queue entry
            auto sqe = io_uring_get_sqe(&ring2);
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

        // Submit queued reads
        int ret = io_uring_submit(&ring2);
        if(ret < 0) {
            fprintf(stderr, "io_uring_submit: %s\n", strerror(-ret));
        }
    
        // Update count of submitted requests
        req_submitted += nsubmit;
    }
    return nsubmit;
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
        cv.wait(lk, [this] { return !active || (submissions.size() > 0) || ((submissions.size() == 0) && (req_submitted > req_completed)); });
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
            if( (submissions.size() == 0) && (req_submitted > req_completed) ) {
                avail_cqe += nready;
                if(req_submitted == req_completed) {
                    break;
                } else {
                    continue;
                }
            }
    
            // Calculate how many reads to submit
            avail_cqe += nready;
            size_t num_reads = submissions.size();
            if(num_reads > avail_cqe) 
                num_reads = avail_cqe;
    
            // Prep reads
            for(size_t i=0; i<num_reads; i++) {
                // Get segment from queue
                segment_t seg = submissions.front();
                if(seg.size + seg.offset > fsize)
                    seg.size = fsize - seg.offset;
    
                // Prepare submission queue entry
                auto sqe = io_uring_get_sqe(&ring);
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
            avail_cqe -= num_reads;
            req_submitted += num_reads;
    
            // Submit queued reads
            ret = io_uring_submit(&ring);
            if(ret < 0) {
                fprintf(stderr, "io_uring_submit: %s\n", strerror(-ret));
            }
            // Loop if waiting for all segments 
        } while(wait_all_mode && ((submissions.size() > 0) || (req_submitted > req_completed)));

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
    cv.wait(lock, [this] { return (submissions.size() == 0) && (req_submitted == req_completed); });
    // Clear completion set
    completions.clear();
    wait_all_mode = true;
    lock.unlock();
    cv.notify_one();
#else
//    do {
//        uint32_t ncomp = request_completion();
//        uint32_t nsubm = request_submission();
//    } while( !( (submissions.size() == 0) && (req_submitted == req_completed) ) );
    while( !( (submissions.size() == 0) && (req_submitted == req_completed) ) ) {
        uint32_t ncomp = request_completion();
        uint32_t nsubm = request_submission();
    }
    completions.clear();
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

