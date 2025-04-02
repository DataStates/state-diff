#include "data_loader.hpp"
#include <iostream>
#include <string>

data_loader_t::data_loader_t() {
    TIMER_START(init_loader);
    host_cache_ = new host_cache_t();
    INFO("Loader - host caches initialized");
    TIMER_STOP(init_loader, "Initialized data loader");
}

data_loader_t::~data_loader_t() {
    host_cache_ = nullptr;
    DBG("Loader - destroyed");
};

// void
// data_loader_t::coalesce(int id, std::vector<size_t> offsets, size_t seg_size,
//                         uint32_t gap, int n_readers) {
//     INFO("Loader (" << id << ")- Coalescing offsets with gap = " << gap
//                     << " for large reads");
//     if (offsets.empty())
//         return;
//     size_t start = offsets[0];
//     size_t lastOffset = start;
//     size_t in_group_offt = 1;
//     size_t wasted_read_bytes = 0;
//     for (size_t i = 1; i < offsets.size(); i++) {
//         size_t curr_offset = offsets[i];
//         if (curr_offset - lastOffset <= gap) {   // include new offset
//             lastOffset = curr_offset;
//             in_group_offt++;
//         } else {   // create batch and start a new group
//             batch_t *seg_batch = new batch_t(n_readers, in_group_offt);
//             size_t segment_start = start * seg_size;
//             size_t combined_size = (lastOffset - start + 1) * seg_size;
//             // Compute wasted bytes
//             size_t needed_size = in_group_offt * seg_size;
//             wasted_read_bytes += (combined_size - needed_size);
//             for (int j = 0; j < n_readers; j++) {
//                 seg_batch->push(segment_t(segment_start, combined_size));
//             }
//             host_cache_->stage_in(id, seg_batch);
//             start = curr_offset;
//             lastOffset = curr_offset;
//             in_group_offt = 1;
//         }
//     }
//     // Account for the last group
//     batch_t *seg_batch = new batch_t(n_readers, in_group_offt);
//     size_t segment_start = start * seg_size;
//     size_t combined_size = (lastOffset - start + 1) * seg_size;
//     // Compute wasted bytes
//     size_t needed_size = in_group_offt * seg_size;
//     wasted_read_bytes += (combined_size - needed_size);
//     for (int j = 0; j < n_readers; j++) {
//         seg_batch->push(segment_t(segment_start, combined_size));
//     }
//     host_cache_->stage_in(id, seg_batch);
//     wasted_bytes[id] = wasted_read_bytes;
// }

void
data_loader_t::coalesce(int id, std::vector<size_t> offsets, size_t seg_size,
                        uint32_t gap, int n_readers,
                        size_t used_chks_per_read) {
    INFO("Loader (" << id << ")- Coalescing offsets with gap = " << gap
                    << " for large reads");
    if (offsets.empty())
        return;
    size_t start = offsets[0];
    size_t lastOffset = start;
    size_t in_group_offt = 1;
    size_t wasted_read_bytes = 0;
    size_t wait_for_count = 0;
    size_t n_seg_in_segvec = 0;
    std::vector<segment_t> segments;
    for (size_t i = 1; i < offsets.size(); i++) {
        size_t curr_offset = offsets[i];
        if (curr_offset - lastOffset <= gap) {   // include new offset
            lastOffset = curr_offset;
            in_group_offt++;
        } else {   // create batch and start a new group
            size_t segment_start = start * seg_size;
            size_t n_segs = lastOffset - start + 1;
            size_t combined_size = n_segs * seg_size;
            // Compute wasted bytes
            size_t needed_size = in_group_offt * seg_size;
            wasted_read_bytes += (combined_size - needed_size);
            n_seg_in_segvec += n_segs;
            segments.push_back(segment_t(segment_start, combined_size));
            // Create batch and stage in when we reach minimum IO and compute
            wait_for_count += in_group_offt;
            if (wait_for_count >= used_chks_per_read) {
                batch_t *seg_batch =
                    new batch_t(segments.size() * n_readers, wait_for_count);
                for (int j = 0; j < n_readers; j++) {
                    for (segment_t segment : segments) {
                        seg_batch->push(segment);
                    }
                }
                host_cache_->stage_in(id, seg_batch);
                segments.clear();
                wait_for_count = 0;
                n_seg_in_segvec = 0;
            }
            start = curr_offset;
            lastOffset = curr_offset;
            in_group_offt = 1;
        }
    }
    // Account for the last group
    size_t segment_start = start * seg_size;
    size_t n_segs = lastOffset - start + 1;
    size_t combined_size = n_segs * seg_size;
    // Compute wasted bytes
    size_t needed_size = in_group_offt * seg_size;
    wasted_read_bytes += (combined_size - needed_size);
    n_seg_in_segvec += n_segs;
    segments.push_back(segment_t(segment_start, combined_size));
    wait_for_count += in_group_offt;
    // Create last batch and stage in for read
    batch_t *seg_batch =
        new batch_t(segments.size() * n_readers, wait_for_count);
    for (int j = 0; j < n_readers; j++) {
        for (segment_t segment : segments) {
            seg_batch->push(segment);
        }
    }
    host_cache_->stage_in(id, seg_batch);
    wasted_bytes[id] = wasted_read_bytes;
}

void
create_segment_group(int id, size_t start, size_t lastOffset,
                     size_t in_group_offt, size_t seg_size,
                     size_t &wasted_read_bytes, size_t &n_seg_in_segvec,
                     std::vector<segment_t> &segments) {
    size_t segment_start = start * seg_size;
    size_t n_segs = lastOffset - start + 1;
    size_t combined_size = n_segs * seg_size;
    size_t needed_size = in_group_offt * seg_size;
    wasted_read_bytes += (combined_size - needed_size);
    n_seg_in_segvec += n_segs;
    segments.push_back(segment_t(segment_start, combined_size));
}

void
data_loader_t::stage_batch_if_ready(int id, std::vector<segment_t> &segments,
                                    size_t &wait_for_count,
                                    size_t &n_seg_in_segvec, int n_readers,
                                    size_t used_chks_per_read) {
    if (wait_for_count >= used_chks_per_read) {
        batch_t *seg_batch =
            new batch_t(segments.size() * n_readers, wait_for_count);
        // batch_t *seg_batch =
        //     new batch_t(n_seg_in_segvec * n_readers, wait_for_count);
        for (int i = 0; i < n_readers; i++) {
            for (segment_t &segment : segments) {
                seg_batch->push(segment);
            }
        }
        host_cache_->stage_in(id, seg_batch);
        segments.clear();
        wait_for_count = 0;
        n_seg_in_segvec = 0;
    }
}

void
data_loader_t::stage_final_batch(int id, std::vector<segment_t> &segments,
                                 size_t wait_for_count, size_t n_seg_in_segvec,
                                 int n_readers) {
    // batch_t *seg_batch =
    //     new batch_t(n_seg_in_segvec * n_readers, wait_for_count);
    batch_t *seg_batch =
        new batch_t(segments.size() * n_readers, wait_for_count);
    for (int i = 0; i < n_readers; i++) {
        for (segment_t &segment : segments) {
            seg_batch->push(segment);
        }
    }
    host_cache_->stage_in(id, seg_batch);
}

// void
// data_loader_t::coalesce(int id, std::vector<size_t> offsets, size_t seg_size,
//                         uint32_t gap, int n_readers,
//                         size_t used_chks_per_read) {
//     INFO("Loader (" << id << ")- Coalescing offsets with gap = " << gap
//                     << " for large reads");
//     if (offsets.empty())
//         return;
//     size_t start = offsets[0];
//     size_t lastOffset = start;
//     size_t in_group_offt = 1;
//     size_t wasted_read_bytes = 0;
//     size_t wait_for_count = 0;
//     size_t n_seg_in_segvec = 0;
//     std::vector<segment_t> segments;
//     for (size_t i = 1; i < offsets.size(); i++) {
//         size_t curr_offset = offsets[i];
//         if (curr_offset - lastOffset <= gap) {   // Group nearby offsets
//             lastOffset = curr_offset;
//             in_group_offt++;
//             // std::cout << "Found one" << std::endl;
//         } else {   // Create segment and start a new group
//             create_segment_group(id, start, lastOffset, in_group_offt, seg_size,
//                                  wasted_read_bytes, n_seg_in_segvec, segments);
//             wait_for_count += in_group_offt;
//             stage_batch_if_ready(id, segments, wait_for_count, n_seg_in_segvec, n_readers,
//                                  used_chks_per_read);
//             // Reset for the next group
//             start = curr_offset;
//             lastOffset = curr_offset;
//             in_group_offt = 1;
//         }
//     }
//     // Handle the final segment group
//     create_segment_group(id, start, lastOffset, in_group_offt, seg_size,
//                          wasted_read_bytes, n_seg_in_segvec, segments);
//     wait_for_count += in_group_offt;
//     stage_final_batch(id, segments, wait_for_count, n_seg_in_segvec, n_readers);
//     wasted_bytes[id] = wasted_read_bytes;
// }

void
data_loader_t::enqueue_reads(int id, size_t seg_size, size_t n_segs_per_read,
                             size_t total_n_segs, size_t total_read_size) {

    size_t n_read_call = (total_n_segs + n_segs_per_read - 1) / n_segs_per_read;
    size_t staged_size = 0;
    size_t batch_len = n_segs_per_read;
    batch_t *seg_batch = new batch_t(batch_len);
    for (size_t i = 0; i < total_n_segs; i++) {
        if (i % n_segs_per_read == 0 && i > 0) {

            DBG("Loader (" << id << ")- Staging batch "
                           << (i / n_segs_per_read) - 1 << " of size "
                           << batch_len << " for read from file");
            host_cache_->stage_in(id, seg_batch);
            n_read_call--;
            // Adjust the segment count for the last batch
            size_t remaining = total_n_segs - i;
            batch_len = (n_read_call == 1) ? remaining : batch_len;
            seg_batch = new batch_t(batch_len);
        }
        size_t offset = i * seg_size;
        size_t current_seg_size =
            (i == total_n_segs - 1) ? total_read_size - staged_size : seg_size;
        segment_t seg(offset, current_seg_size);
        seg_batch->push(seg);
        staged_size += current_seg_size;
    }
    // stage last batch
    DBG("Loader (" << id << ")- Staging batch "
                   << total_n_segs / n_segs_per_read << " of size " << batch_len
                   << " for read from file");
    host_cache_->stage_in(id, seg_batch);
}

/**
 * @brief Load data from a file
 *
 * This function takes a liburing reader and a destination of the data
 * as input (trans_type) and loads data from a file to the destination.
 *
 * @param io_reader The reference of the liburing reader for file IO.
 * @param start_foffset Offset of the file from which the reader starts reading
 * from.
 * @param seg_size  Size of each segment enqueued for read operations
 * @param batch_size Number of segments to submit at a time to the liburing
 * reader
 * @param trans_type Definition of the source and destination of the data
 * @param offsets List of file offsets to read data from. If empty, all
 * data are read.
 * @param merge_seg Temporary boolean parameter used to define if non-contiguous
 * offsets should be merged.
 * @return ID of the file loading request. It is used to coordinate multiple
 * concurrent file loading requests.
 */
int
data_loader_t::file_load(FileReader &io_reader, size_t seg_size,
                         TransferType trans_type) {
    TIMER_START(file_load);
    assert((trans_type == TransferType::FileToHost ||
            trans_type == TransferType::FileToDevice) &&
           "Invalid TransferType: Must be FileToHost or FileToDevice");

    // Assign an ID to this loader call
    int loader_id = instance_count++;
    ready_count[loader_id] = 0;
    wasted_bytes[loader_id] = 0;

    INFO("Loader (" << loader_id
                    << ")- Creating segments without given file offsets");

    // Create segments to read
    size_t total_read_size = io_reader.size();
    assert(seg_size > 0 && total_read_size > 0);
    size_t total_n_segs = (total_read_size + seg_size - 1) / seg_size;
    size_t base_segcnt = 2;
    size_t n_segs_per_read = std::min({base_segcnt, total_n_segs});
    enqueue_reads(loader_id, seg_size, n_segs_per_read, total_n_segs,
                  total_read_size);
    INFO("Loader (" << loader_id
                    << ")- All batches staged in for read from file");
    TIMER_STOP(file_load, "Created segments and staged for file read");

    // Set the reader to use for file IO.
    // Making sure metadata are enqueue before setting the reader to avoid
    // having the thread wait and constantly pool for enqueue metadata.
    host_cache_->set_reader(loader_id, &io_reader);
    return loader_id;
}

int
data_loader_t::file_load(FileReader &io_reader0, FileReader &io_reader1,
                         std::vector<size_t> offsets, size_t seg_size,
                         TransferType trans_type, uint32_t gap, size_t block_size) {
    TIMER_START(file_load);
    assert((trans_type == TransferType::FileToHost ||
            trans_type == TransferType::FileToDevice) &&
           "Invalid TransferType: Must be FileToHost or FileToDevice");
    // Assign an ID to this loader call
    int loader_id = instance_count++;
    ready_count[loader_id] = 0;
    // Set two reader to use for file IO
    // host_cache_->set_reader(loader_id, &io_reader0, &io_reader1);

    int n_readers = 2;
    // size_t wait_for_size = 128 * 1024 * 1024;
    size_t wait_for_size = block_size;
    size_t used_chks_per_read = (wait_for_size + seg_size - 1) / seg_size;

    INFO("Loader ("
         << loader_id
         << ")- Creating segments for two readers given file offsets");
    // auto start_coalesce = std::chrono::high_resolution_clock::now();
    coalesce(loader_id, offsets, seg_size, gap, n_readers, used_chks_per_read);
    // auto end_coalesce = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> coalesce_time = end_coalesce - start_coalesce;
    // double c_time = coalesce_time.count();
    // printf("Coalesce time = %f ms\n", c_time*1000);
    INFO("Loader (" << loader_id
                    << ")- All batches staged in for read with two readers");
    TIMER_STOP(file_load, "Created segments and staged for file read");

    // host_cache_->activate(loader_id);
    host_cache_->set_reader(loader_id, &io_reader0, &io_reader1);

    return loader_id;
}

// This implementation of next works because the memory for the segments in a
// batch point are allocated contiguously from the data_store. However, once the
// pointer to the ready batch is returned to the user, new data may be loaded
// inplace because the loader does not know when the kernel completed
// computation, affecting accuracy. To address that issue, we implement an
// approach that keeps track of the number of batches returned to deallocate the
// previous batch before returning a pointer to the next batch. The reason why
// we are returning a pointer to the user instead of receiving a pointer from
// the user (and copying data to the user pointer) is because computation is
// faster than data movement, i.e., the time it would take to copy the data is
// higher than the time to process it.

batch_t *
get_next(int id, base_cache_t *cache_tier, bool should_release) {
    if (should_release)
        cache_tier->release(id);
    return cache_tier->get_completed(id);
}

next_batch_t
data_loader_t::next(int id, TransferType trans_type) {
    size_t call_count = ready_count[id]++;
    TIMER_START(next);
    assert((trans_type == TransferType::HostToDevice ||
            trans_type == TransferType::HostPinned ||
            trans_type == TransferType::FileToHost ||
            trans_type == TransferType::FileToDevice) &&
           "Invalid TransferType!");

    base_cache_t *cache_tier = host_cache_;
    // Because of the FIFO queue implementation, we need to make sure the first
    // retrieved batch of the next ID is not the last batch the previous
    // retrieving ID had read.
    if (id != last_retrieving_id) {
        INFO("Releasing last batch of ID = " << last_retrieving_id
                                             << " from storage")
        cache_tier->release(last_retrieving_id);
        last_retrieving_id = id;
    }
    // Retrieve the next batch for the current ID
    batch_t *front_batch = get_next(id, cache_tier, call_count > 0);
    TIMER_STOP(next, "Retrieved pointer to next batch of data for computation");
    assert(front_batch->data[0].buffer != nullptr && front_batch->size > 0);
    DBG("Retrieved pointer to offset " << front_batch->data[0].offset << " for "
                                       << front_batch->size << " bytes");
    next_batch_t batch = {front_batch->data[0].buffer, front_batch->size,
                          front_batch->proc_offt};
    return batch;
}

size_t
data_loader_t::get_wasted_bytes_count(int id) {
    return wasted_bytes[id];
}