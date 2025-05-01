#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <tuple>
#include <vector>

// Abstract class model which contains IO and compute models (parameterized) and extend the mode here.
class ThroughputOptimizer {

  private:
    // double comp_curve_max_;
    // double comp_growth_rate_;
    // double comp_midpoint_;
    double compute_slope_ = 0.743;
    double compute_intercept_ = -17.0f;
    std::vector<double> io_slope_ = {0.281, 0.058};
    double io_intercept_ = -5.525;
    std::vector<int> gaps = {0, 1, 2, 3};

  public:
    ThroughputOptimizer() {}

    double compute_model(double batch_size) const {
        double log_x = std::log(batch_size);
        return std::exp(compute_intercept_ + compute_slope_ * log_x);
        // return comp_curve_max_ /
        //        (1 + std::exp(-comp_growth_rate_ * (log_x - comp_midpoint_)));
    }

    double io_model(double num_ops, double op_size) const {
        double log_ops = std::log(num_ops);
        double log_size = std::log(op_size);
        double log_y =
            io_intercept_ + (io_slope_[0] * log_ops + io_slope_[1] * log_size);
        return std::exp(log_y);
    }

    std::pair<std::vector<size_t>, std::vector<size_t>>
    analyze_contiguity(const std::vector<size_t> &offsets, int gap,
                       size_t chunk_size, size_t batch_size) const {
        // Convert gap from byte to chunk units
        size_t start = offsets[0];
        size_t last_offset = start;
        size_t in_group_offt = 1;
        size_t wait_for_count = 0;
        size_t used_chks_per_read = std::ceil(static_cast<double>(batch_size) / chunk_size);

        std::vector<size_t> segments;
        std::vector<size_t> IOP_count;     // Number of segments in each batch
        std::vector<size_t> size_of_iop;   // Size of each batch in bytes
        size_t diff = static_cast<size_t>(gap);

        for (size_t i = 1; i < offsets.size(); ++i) {
            size_t curr_offset = offsets[i];
            if (curr_offset - last_offset <= diff) {
                last_offset = curr_offset;
                ++in_group_offt;
            } else {
                size_t n_segs = last_offset - start + 1;
                size_t combined_size = n_segs * chunk_size;
                segments.push_back(combined_size);
                wait_for_count += in_group_offt;

                if (wait_for_count >= used_chks_per_read) {
                    size_t batch_data_size = std::accumulate(
                        segments.begin(), segments.end(), size_t{0});
                    size_of_iop.push_back(batch_data_size);
                    IOP_count.push_back(segments.size());

                    segments.clear();
                    wait_for_count = 0;
                }

                start = curr_offset;
                last_offset = curr_offset;
                in_group_offt = 1;
            }
        }

        // Final batch
        size_t n_segs = last_offset - start + 1;
        size_t combined_size = n_segs * chunk_size;
        segments.push_back(combined_size);
        wait_for_count += in_group_offt;

        if (!segments.empty()) {
            size_t batch_data_size =
                std::accumulate(segments.begin(), segments.end(), size_t{0});
            size_of_iop.push_back(batch_data_size);
            IOP_count.push_back(segments.size());
        }

        return {IOP_count, size_of_iop};
    }

    double score_function(size_t batch_size, const std::vector<size_t> &offsets,
                          int gap, size_t chunk_size, int n_comp_cores) const {

        auto [iops_counts, iops_sizes] =
            analyze_contiguity(offsets, gap, chunk_size, batch_size);
        std::vector<size_t> work_size_per_core;
        for (size_t size : iops_sizes) {
            work_size_per_core.push_back(size / n_comp_cores);
        }

        // estimating the compute time
        std::vector<double> pred_compute_time;
        for (size_t work : work_size_per_core) {
            pred_compute_time.push_back(compute_model(work));
        }

        // estimating the data loading time
        std::vector<double> pred_io_time;
        for (size_t i = 0; i < iops_counts.size(); ++i) {
            pred_io_time.push_back(io_model(iops_counts[i], iops_sizes[i]));
        }

        double total_data_size =
            std::accumulate(iops_sizes.begin(), iops_sizes.end(), 0.0) /
            (1024.0 * 1024.0 * 1024.0);
        double sum_compute_time = std::accumulate(pred_compute_time.begin(),
                                                  pred_compute_time.end(), 0.0);
        double sum_io_time =
            std::accumulate(pred_io_time.begin(), pred_io_time.end(), 0.0);

        if (sum_compute_time > sum_io_time) {
            return total_data_size / sum_compute_time;
        } else {
            return total_data_size / sum_io_time;
        }
    }

    std::pair<double, double> binary_search(std::vector<size_t> batch_sizes,
                                            const std::vector<size_t> &offsets,
                                            int gap, size_t chunk_size,
                                            int n_cores) const {

        double best_bsize = batch_sizes[0];
        double best_score = -std::numeric_limits<double>::infinity();

        int left = 0;
        int right = static_cast<int>(batch_sizes.size()) - 1;

        while (left <= right) {
            int mid = (left + right) / 2;
            double bsize = batch_sizes[mid];

            double score =
                score_function(bsize, offsets, gap, chunk_size, n_cores);
            if (score > best_score) {
                best_score = score;
                best_bsize = bsize;
            }

            if (mid + 1 < static_cast<int>(batch_sizes.size())) {
                double next_score = score_function(
                    batch_sizes[mid + 1], offsets, gap, chunk_size, n_cores);
                if (next_score > score) {
                    left = mid + 1;
                } else {
                    right = mid - 1;
                }
            } else {
                break;
            }
        }

        return {best_bsize, best_score};
    }

    std::pair<int, size_t>
    optimize_throughput(const std::vector<size_t> &offsets, size_t chunk_size, int n_cores,
                        bool verbose = false) const {

        // std::sort(batch_sizes.begin(), batch_sizes.end());

        std::vector<size_t> batch_sizes;
        for (size_t mb = 1; mb <= 128; mb *= 2) {
            batch_sizes.push_back(mb * 1024 * 1024);
        }
        double opt_thrpt = -std::numeric_limits<double>::infinity();
        int opt_gap = -1;
        size_t opt_bsize = 0;

        for (int gap : gaps) {
            auto [batch_size, total_thrpt] =
                binary_search(batch_sizes, offsets, gap, chunk_size, n_cores);
            if (total_thrpt > opt_thrpt) {
                opt_thrpt = total_thrpt;
                opt_gap = gap;
                opt_bsize = batch_size;
            }
        }

        if (verbose) {
            std::cout << "Optimal Throughput: " << opt_thrpt << " GiB/s\n";
            std::cout << "Optimal Gap: " << opt_gap << " KiB\n";
            std::cout << "Optimal Batch size: " << (int)(opt_bsize / 1024)
                      << " KiB\n";
        }

        return {opt_gap, opt_bsize};
    }
};
