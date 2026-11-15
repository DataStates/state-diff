#include "statediff.hpp"
#include <statediff/log.hpp>
#include <scissor/regressor.hpp>
#include <fcntl.h>
#include <unistd.h>

namespace state_diff {

// ======================================================
// create(fname): build the Merkle tree by reading a file
// through the client's scissor Loader (backed by the
// reader given at construction).
// ======================================================
template <typename DataType>
void
client_t<DataType>::create(const std::string& fname, size_t read_blk_size,
                           TransferType create_tree_tier) {
    STATEDIFF_TIMER_START(client_create_tree);
    scissor::Loader &ld = loader();
    auto load_handle = ld.load(fname, client_info.data_size, read_blk_size, create_tree_tier);
    tree.create(client_info, ld, load_handle, create_tree_tier);
    ld.close(load_handle);
    STATEDIFF_TIMER_PERF(client_create_tree, "State-diff tree created from file",
                         "chkpt_id", curr_chkpt_id + 1);
    curr_chkpt_id++;
}

// ======================================================
// compare_data: validate hash mismatches via direct I/O
// through the scissor pipeline (liburing-backed).
// ======================================================
template <typename DataType>
size_t
client_t<DataType>::compare_data(client_t &prev, const std::string& prev_fname,
                                 const std::string& curr_fname,
                                 std::vector<size_t> &diff_offsets,
                                 Kokkos::Bitset<> &changed_chunks,
                                 Kokkos::View<size_t[1]> &num_changed,
                                 Kokkos::View<size_t[1]> &num_comparisons,
                                 uint32_t offt_gap, size_t block_size,
                                 TransferType compare_tier, FileSrc file_src_loc, bool exec_compare,
                                 Tuning tuning) {
    //  This implementation of compare_data works for the following memory layout:
    // {prev_a, curr_a, prev_b, curr_b, prev_c, curr_c, ...}
    std::string diff_label = std::string("Chkpt ") +
                             std::to_string(client_info.id) + std::string(": ");
    Kokkos::Profiling::pushRegion(
        diff_label + std::string("Compare Trees direct comparison"));

    // reused parameters
    double err_tol = client_info.error_tolerance;
    size_t num_diff_hash = static_cast<size_t>(diff_offsets.size());
    size_t elemPerChunk = client_info.chunk_size / sizeof(DataType);
    AbsoluteComp<DataType> abs_comp;
    Kokkos::Experimental::ScatterView<size_t[1]> num_comp(num_comparisons);
    auto &changed_blocks = changed_chunks;
    size_t n_files = 2;   // number of readers (prev + curr)
    size_t work_done = 0;
    auto first_offset = diff_offsets.begin();
    DataType *batch_ptr = nullptr;

    // start loading data from the two files
    //   Tuning::Fixed -- using the input gap and block size
    //                    (nthreads = 1, for benchmarking/profiling purposes only)
    //   Tuning::Model -- using our performance model with Kokkos::num_threads()
    int nthreads = (tuning == Tuning::Model) ? Kokkos::num_threads() : 1;

    scissor::Loader &loader = this->loader();
    auto load_handle = loader.load(prev_fname, curr_fname, diff_offsets,
                                   client_info.chunk_size, compare_tier, file_src_loc,
                                   nthreads, offt_gap, block_size);
    // Gap and block size the Loader actually used (the Regressor's
    // choice in model mode)
    scissor::LoadParams used = loader.params(load_handle);
    best_gap = static_cast<int>(used.gap);
    best_batch_size = used.block_size;

    while (true) {
        Timer::time_point iter_beg = Timer::now();
        // Read one batch of data
        // Iterations are IO bound with a non-zero wait time after
        // comparison. Therefore, the next load iteration will
        // start after comparison+wait time. Thus load time is
        // comparison + wait times, except the 1st iterarion where
        // load time is wait time
        Timer::time_point wait_beg = Timer::now();
        scissor::common::Batch* front_batch = loader.next(load_handle, compare_tier);
        if (!front_batch) break;
        batch_ptr = reinterpret_cast<DataType*>(front_batch->batch_ptr);
        // number of used offsets per read, approx work size
        size_t proc_offt = front_batch->proc_offt;
        Timer::time_point wait_end = Timer::now();

        // Compare data in batch
        Timer::time_point cmp_beg = Timer::now();
        size_t work_end = work_done + proc_offt;
        assert(front_batch->size % (n_files * client_info.chunk_size) == 0);
        auto seg_first_chunk = front_batch->first_chunk_idx();
        auto seg_num_chunks  = front_batch->chunk_counts();
        size_t n_chunks = front_batch->n_chunks();

        // Build per-batch chunk offsets from segment data.
        // Each segment pair s: prev data at data[2*s], curr data at data[2*s+1].
        size_t chunk_size = client_info.chunk_size;
        size_t n_seg_pairs = seg_num_chunks.size();
        std::vector<size_t> batch_read_offts(n_chunks);
        for (size_t s = 0; s < n_seg_pairs; ++s) {
            size_t base_offset = front_batch->data[2 * s].offset / chunk_size;
            for (size_t c = 0; c < seg_num_chunks[s]; ++c) {
                batch_read_offts[seg_first_chunk[s] + c] = base_offset + c;
            }
        }

        STATEDIFF_INFO("batch segment count",
                       "n_seg_pairs", n_seg_pairs,
                       "num_diff_hash", num_diff_hash);
        STATEDIFF_INFO("batch progress",
                       "work_done", work_done,
                       "proc_offt", proc_offt,
                       "n_chunks", n_chunks);

        // exec_compare is a boolean used for benchmarking when comparison is
        // overlapped with data loading
        if (exec_compare) {
            size_t ndiff = 0;
            auto range_policy = Kokkos::RangePolicy<size_t>(0, n_chunks);
            Kokkos::parallel_reduce(
                "Count differences", range_policy,
                KOKKOS_LAMBDA(const size_t idx, size_t &update) {
                    size_t chunk_offt = batch_read_offts[idx];

                    // filtering requested read only
                    bool relevant =
                        std::binary_search(first_offset + work_done,
                                           first_offset + work_end, chunk_offt);
                    if (!relevant)
                        return;

                    // Compute chunk indices in batch given interleaved layout
                    // chunk idx (upper_bound on seg_first_chunk)
                    size_t lo = 0, hi = seg_first_chunk.size();
                    while (lo + 1 < hi) {
                        size_t mid = (lo + hi) >> 1;
                        if (seg_first_chunk[mid] <= idx) {
                            lo = mid;
                        } else {
                            hi = mid;
                        }
                    }
                    size_t segment_idx = lo;
                    size_t chunk_idx = idx - seg_first_chunk[segment_idx];

                    size_t prev_start = (n_files * seg_first_chunk[segment_idx] + chunk_idx) * elemPerChunk;
                    size_t curr_start = prev_start + seg_num_chunks[segment_idx] * elemPerChunk;

                    bool diff_found = false;
                    for (size_t i = 0; i < elemPerChunk; ++i) {
                        if (!abs_comp(batch_ptr[prev_start + i], batch_ptr[curr_start + i], err_tol)) {
                            update += 1;
                            diff_found = true;
                        }
                    }
                    if (diff_found) {
                        changed_blocks.set(chunk_offt);
                    }
                    auto ncomp_access = num_comp.access();
                    ncomp_access(0) += elemPerChunk;
                },
                Kokkos::Sum<size_t>(ndiff));
            nchange += ndiff;
        }
        Timer::time_point cmp_end = Timer::now();
        Timer::time_point iter_end = Timer::now();

        // update timers
        double comp_time_iter =
            std::chrono::duration_cast<Duration>(cmp_end - cmp_beg).count();
        double wait_time_iter =
            std::chrono::duration_cast<Duration>(wait_end - wait_beg).count();
        double total_time_iter =
            std::chrono::duration_cast<Duration>(iter_end - iter_beg).count();
        timers[2] += total_time_iter;   // total comparison time
        // load time
        if (work_done == 0) {
            timers[3] += wait_time_iter;
        } else {
            timers[3] += total_time_iter;
        }
        timers[4] += comp_time_iter;    // comparison time
        timers[5] += wait_time_iter;    // wait time

        // update iterator
        work_done = work_end;
    }
    Kokkos::Experimental::contribute(num_comparisons, num_comp);
    wasted_bytes = loader.wasted_bytes(load_handle);
    IOP_count = loader.iop_count(load_handle);
    loader.close(load_handle);
    Kokkos::Profiling::popRegion();
    return nchange;
}

// ======================================================
// compare_data_ideal_compute: ideal (non-pipelined) direct
// comparison using reads through the client's reader, used
// for benchmarking the theoretical compute-only cost.
// ======================================================
template <typename DataType>
size_t
client_t<DataType>::compare_data_ideal_compute(
    client_t &prev, Vector<size_t> &diff_hash_vec,
    Kokkos::Bitset<> &changed_chunks, Kokkos::View<size_t[1]> &num_changed,
    Kokkos::View<size_t[1]> &num_comparisons, TransferType compare_tier,
    const std::string& prev_fname, const std::string& curr_fname) {

    std::string diff_label = std::string("Chkpt ") +
                             std::to_string(client_info.id) + std::string(": ");
    Kokkos::Profiling::pushRegion(
        diff_label + std::string("Compare Trees direct comparison"));

    // Sort indices for better performance
    Kokkos::Profiling::pushRegion(diff_label +
                                  std::string("Compare Tree sort indices"));
    size_t num_diff_hash = static_cast<size_t>(diff_hash_vec.size());
    auto subview_bounds = Kokkos::make_pair((size_t)(0), num_diff_hash);
    auto diff_hash_subview =
        Kokkos::subview(diff_hash_vec.vector_d, subview_bounds);
    Kokkos::sort(diff_hash_vec.vector_d, 0, num_diff_hash);
    size_t elemPerChunk = client_info.chunk_size / sizeof(DataType);
    Kokkos::Profiling::popRegion();

    Timer::time_point compare_beg = Timer::now();
    std::vector<scissor::Segment> segments(2 * num_diff_hash);
    std::vector<DataType> buffer0(num_diff_hash * elemPerChunk),
        buffer1(num_diff_hash * elemPerChunk);
    double err_tol = client_info.error_tolerance;
    auto data_size = client_info.data_size;
    auto chunk_size = client_info.chunk_size;
    auto &changed_blocks = changed_chunks;
    size_t *offsets = diff_hash_vec.vector_d.data();
    Kokkos::Experimental::ScatterView<size_t[1]> num_comp(num_comparisons);
    AbsoluteComp<DataType> abs_comp;
    if (num_diff_hash > 0) {
        Kokkos::deep_copy(diff_hash_vec.vector_h, diff_hash_vec.vector_d);
        int fd0 = open(prev_fname.c_str(), O_RDONLY);
        int fd1 = open(curr_fname.c_str(), O_RDONLY);
        Kokkos::parallel_for(
            "Fill segment vectors",
            Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(
                0, num_diff_hash),
            [&](size_t i) {
                scissor::Segment &seg0 = segments[i];
                seg0.id = i;
                seg0.fd = fd0;
                seg0.buffer =
                    (uint8_t *)(buffer0.data()) + i * chunk_size;
                seg0.offset = diff_hash_vec.vector_h(i) * chunk_size;
                seg0.size = chunk_size;

                scissor::Segment &seg1 = segments[num_diff_hash + i];
                seg1.id = num_diff_hash + i;
                seg1.fd = fd1;
                seg1.buffer =
                    (uint8_t *)(buffer1.data()) + i * chunk_size;
                seg1.offset = diff_hash_vec.vector_h(i) * chunk_size;
                seg1.size = chunk_size;
            });
        Kokkos::fence();
        // One reader serves both files (segments carry their fd)
        reader_->enqueue_reads(segments);
        std::cout << "Segment enqueued successfully" << std::endl;
        reader_->wait_all();
        std::cout << "Segment loaded successfully" << std::endl;

        Kokkos::Profiling::pushRegion(
            diff_label + std::string("Compare Tree direct comparison"));
        Timer::time_point comp_beg = Timer::now();
        uint64_t ndiff = 0;
        // Parallel comparison
        const DataType *prev_buffer = buffer0.data();
        const DataType *curr_buffer = buffer1.data();
        using PolicyType =
            Kokkos::RangePolicy<size_t, Kokkos::DefaultHostExecutionSpace>;
        auto range_policy = PolicyType(0, num_diff_hash);
        Kokkos::parallel_reduce(
            "Count differences", range_policy,
            KOKKOS_LAMBDA(const size_t idx, uint64_t &update) {
                size_t blk_offset = offsets[idx];
                size_t blk_start = idx * elemPerChunk;
                bool diff_found = false;
                for (size_t i = 0; i < elemPerChunk; i++) {
                    size_t elem_idx = blk_start + i;
                    if ((elem_idx < data_size) &&
                        (!abs_comp(prev_buffer[elem_idx], curr_buffer[elem_idx],
                                   err_tol))) {
                        update += 1;
                        diff_found = true;
                    }
                }
                if (diff_found) {
                    changed_blocks.set(blk_offset);
                }
                auto ncomp_access = num_comp.access();
                ncomp_access(0) += elemPerChunk;
            },
            Kokkos::Sum<uint64_t>(ndiff));
        Kokkos::fence();
        nchange += ndiff;
        Kokkos::Profiling::popRegion();
        Timer::time_point comp_end = Timer::now();
        timers[4] +=
            std::chrono::duration_cast<Duration>(comp_end - comp_beg).count();
        close(fd0), close(fd1);
    }
    Timer::time_point compare_end = Timer::now();
    Kokkos::Experimental::contribute(num_comparisons, num_comp);
    timers[2] +=
        std::chrono::duration_cast<Duration>(compare_end - compare_beg).count();
    Kokkos::Profiling::popRegion();
    return nchange;
}

// Explicit instantiations: all callers use float.
template void   state_diff::client_t<float>::create(const std::string&, size_t, scissor::TransferType);
template size_t state_diff::client_t<float>::compare_data(
    client_t<float>&, const std::string&, const std::string&,
    std::vector<size_t>&, Kokkos::Bitset<>&,
    Kokkos::View<size_t[1]>&, Kokkos::View<size_t[1]>&,
    uint32_t, size_t, scissor::TransferType, scissor::FileSrc, bool, Tuning);
template size_t state_diff::client_t<float>::compare_data_ideal_compute(
    client_t<float>&, Vector<size_t>&,
    Kokkos::Bitset<>&, Kokkos::View<size_t[1]>&,
    Kokkos::View<size_t[1]>&, scissor::TransferType,
    const std::string&, const std::string&);

} // namespace state_diff
