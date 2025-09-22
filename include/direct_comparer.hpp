#ifndef DIRECT_COMPARER_HPP
#define DIRECT_COMPARER_HPP
#include "Kokkos_Bitset.hpp"
#include "io_reader.hpp"
#include "statediff_vector.hpp"
#include <Kokkos_Core.hpp>
#include <Kokkos_ScatterView.hpp>
#include <climits>
#include <cstdlib>
#include <iostream>

using Timer = std::chrono::high_resolution_clock;
using Duration = std::chrono::duration<double>;

template <typename DataType> class DirectComparer {
  public:
    double tol;
    static const size_t default_buf_len = 1ULL * 1024 * 1024 * 1024;
    Kokkos::Bitset<Kokkos::DefaultExecutionSpace> changed_entries;
    size_t num_comparisons = 0;
    std::string file0, file1;
    double compare_time = 0.0;
    double total_time = 0.0;
    size_t buf_len;
    size_t d_size;

  public:
    DirectComparer(size_t data_size, double tolerance,
                   size_t batch_size = default_buf_len);

    ~DirectComparer() {};

    template <typename Reader>
    size_t compare(Reader &reader, const std::string& fname0, const std::string& fname1);
    // size_t compare(Reader &reader_prev, Reader &reader_cur);

    double get_total_time() const;
    double get_compare_time() const;
    size_t get_num_comparisons() const;
    size_t get_num_changed_blocks() const;
};

template <typename DataType>
DirectComparer<DataType>::DirectComparer(size_t data_size, double tolerance,
                                         size_t batch_size) {
    d_size = data_size;
    tol = tolerance;
    buf_len = batch_size;
}

template <typename DataType>
template <typename Reader>
size_t
DirectComparer<DataType>::compare(Reader &reader, const std::string& fname0, const std::string& fname1) {
    Timer::time_point e2e_beg = Timer::now();
    Kokkos::Profiling::pushRegion("Direct: Compare: start streaming");

    size_t num_iops = (d_size + buf_len - 1) / buf_len;
    std::vector<segment_t> segments0(num_iops), segments1(num_iops);
    int fd0 = open(fname0.c_str(), O_RDONLY);
    int fd1 = open(fname1.c_str(), O_RDONLY);
    Kokkos::parallel_for(
        "Fill segment vectors",
        Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, num_iops),
        [&](size_t i) {
            size_t start = i * buf_len;
            size_t c_size = (i < num_iops - 1) ? buf_len : d_size - start;
            segments0[i].id = i;
            segments0[i].fd = fd0;
            segments0[i].offset = start;
            segments0[i].size = c_size;
            segments0[i].buffer = (uint8_t *)malloc(c_size);

            segments1[i].id = i;
            segments1[i].fd = fd1;
            segments1[i].offset = start;
            segments1[i].size = c_size;
            segments1[i].buffer = (uint8_t *)malloc(c_size);
        });
    Kokkos::fence();
    Kokkos::Profiling::popRegion();

    Kokkos::Profiling::pushRegion("Direct: Compare: prep");
    size_t num_diff = 0;
    double err_tol = tol;
    changed_entries = Kokkos::Bitset<Kokkos::DefaultExecutionSpace>(num_iops);
    changed_entries.reset();
    auto &changes = changed_entries;
    Kokkos::Profiling::popRegion();

    // Start with first segments
    reader.enqueue_reads({segments0[0], segments1[0]});
    // reader_prev.enqueue_reads({segments0[0]});
    // reader_cur.enqueue_reads({segments1[0]});

    for (size_t iter = 0; iter < num_iops; ++iter) {
        Kokkos::Profiling::pushRegion("Direct: Compare: get slices");
        reader.wait_all();
        // reader_cur.wait_all();
        // reader_prev.wait_all();

        // Enqueue next read if not at the last iteration
        if (iter + 1 < num_iops) {
            reader.enqueue_reads({segments0[iter + 1], segments1[iter + 1]});
            // reader_prev.enqueue_reads({segments0[iter + 1]});
            // reader_cur.enqueue_reads({segments1[iter + 1]});
        }

        segment_t &prev_seg = segments0[iter];
        segment_t &cur_seg = segments1[iter];
        DataType *sliceA = reinterpret_cast<DataType *>(prev_seg.buffer);
        DataType *sliceB = reinterpret_cast<DataType *>(cur_seg.buffer);
        size_t slice_len = prev_seg.size / sizeof(DataType);
        size_t ndiff = 0;
        Kokkos::Profiling::popRegion();

        Kokkos::Profiling::pushRegion("Direct: Compare: compare slices");
        Timer::time_point beg = Timer::now();
        Kokkos::parallel_reduce(
            "Count differences", Kokkos::RangePolicy<size_t>(0, slice_len),
            KOKKOS_LAMBDA(const size_t i, size_t &update) {
                bool diff = false;
                if (Kokkos::abs(sliceA[i] - sliceB[i]) > err_tol) {
                    update += 1;
                    diff = true;
                }
                if (diff) {
                    changes.set(iter);
                }
            },
            Kokkos::Sum<size_t>(ndiff));
        Kokkos::fence();
        num_comparisons += slice_len;
        num_diff += ndiff;
        Timer::time_point end = Timer::now();
        compare_time += std::chrono::duration_cast<Duration>(end - beg).count();
        Kokkos::Profiling::popRegion();

        free(prev_seg.buffer);
        free(cur_seg.buffer);
    }

    Timer::time_point e2e_end = Timer::now();
    close(fd0), close(fd1);
    total_time +=
        std::chrono::duration_cast<Duration>(e2e_end - e2e_beg).count();
    return num_diff;
}

template <typename DataType>
size_t
DirectComparer<DataType>::get_num_comparisons() const {
    return num_comparisons;
}

template <typename DataType>
size_t
DirectComparer<DataType>::get_num_changed_blocks() const {
    return changed_entries.count();
}

template <typename DataType>
double
DirectComparer<DataType>::get_total_time() const {
    return total_time;
}

template <typename DataType>
double
DirectComparer<DataType>::get_compare_time() const {
    return compare_time;
}

#endif   // DIRECT_COMPARER_HPP
