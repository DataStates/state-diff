#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <tuple>
#include <vector>

// Abstract class model which contains IO and compute models (parameterized)
class ThroughputOptimizer {

  private:
    double GB = 1024.0 * 1024.0 * 1024.0;
    double compute_slope_ = 0.748;
    double compute_intercept_ = -17.0366;
    std::vector<double> io_params_;
    std::vector<int> gaps = {0, 1, 2, 4, 8, 16}; // {0, 1, 2, 3};

  public:
    ThroughputOptimizer(FileSrc file_src_loc) {
        switch (file_src_loc) {
            case PFS:
                io_params_ = {4.29e+00, -3.33e-02, 7.26e-06, 4.54e-01, 1.37e-01};
                break;
            case SSD:
                io_params_ = {1.13e+01, -5.58e-03, 1.82e-05, 1.64e-01, 7.48e-02};
                break;
            case Hybrid: 
                io_params_ = {4.26e+00, -3.47e-02, 4.35e-06, 3.86e-01, -2.72e-02};
                break;
            default:
                io_params_ = {4.29e+00, -3.33e-02, 7.26e-06, 4.54e-01, 1.37e-01};
        }
        std::cout << "Optimizing batch size and gap to read from " << file_src_loc << std::endl;
    }

    // batch_size = Work per core in B
    double compute_model(double batch_size) const {
        assert(batch_size > 0.0);
        double log_x = std::log(batch_size);
        return std::exp(compute_intercept_ + compute_slope_ * log_x);
    }

    // num_ops = IOP (number of operations) & op_size = s (op size in GB)
    double io_model(double num_ops, double op_size) const {
        double term_iop_pow = std::pow(num_ops, io_params_[1]);
        double term_iop_exp = std::exp(-io_params_[2] * num_ops);

        double term_size_pow = std::pow(op_size, io_params_[3]);
        double term_size_exp = std::exp(-io_params_[4] * op_size);

        return io_params_[0] * term_iop_pow * term_iop_exp * term_size_pow * term_size_exp;
    }

    std::tuple<std::vector<size_t>, std::vector<size_t>, std::vector<size_t>>
    analyze_contiguity(const std::vector<size_t> &offsets, int gap,
                    size_t chunk_size, size_t batch_size) const {
        
        if (offsets.empty()) return {{}, {}, {}};
        // Offsets are chunks IDs
        size_t start = offsets[0];
        size_t last_offset = start;
        size_t in_group_offt = 1;
        size_t wait_for_count = 0;

        size_t used_chks_per_read =
            std::ceil(static_cast<double>(batch_size) / chunk_size);

        std::vector<size_t> segments;
        std::vector<size_t> IOP_count;
        std::vector<size_t> size_of_iop;
        std::vector<size_t> effective_reads;

        size_t effective_reads_in_batch = 0;
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
                effective_reads_in_batch += in_group_offt;

                if (wait_for_count >= used_chks_per_read) {
                    size_t batch_data_size =
                        std::accumulate(segments.begin(), segments.end(), size_t{0});

                    size_of_iop.push_back(batch_data_size);
                    IOP_count.push_back(segments.size());
                    effective_reads.push_back(effective_reads_in_batch);

                    segments.clear();
                    wait_for_count = 0;
                    effective_reads_in_batch = 0;
                }

                start = curr_offset;
                last_offset = curr_offset;
                in_group_offt = 1;
            }
        }
        // Final group
        size_t n_segs = last_offset - start + 1;
        size_t combined_size = n_segs * chunk_size;
        segments.push_back(combined_size);
        wait_for_count += in_group_offt;
        effective_reads_in_batch += in_group_offt;

        if (!segments.empty()) {
            size_t batch_data_size =
                std::accumulate(segments.begin(), segments.end(), size_t{0});
            size_of_iop.push_back(batch_data_size);
            IOP_count.push_back(segments.size());
            effective_reads.push_back(effective_reads_in_batch);
        }

        return {IOP_count, size_of_iop, effective_reads};
    }

    double score_function(size_t batch_size, const std::vector<size_t> &offsets,
                          int gap, size_t chunk_size, int n_comp_cores) const {

        auto [iops_counts, iops_sizes, effective_reads] =
            analyze_contiguity(offsets, gap, chunk_size, batch_size);

        // Estimate total throughput
        double total_time = 0.0;
        double total_data_size =
            std::accumulate(iops_sizes.begin(), iops_sizes.end(), 0.0) / GB;

        for (size_t i = 0; i < iops_counts.size(); ++i) {
            // estimating the compute time
            double work_bytes = static_cast<double>(effective_reads[i]) * chunk_size;
            double per_core_work = work_bytes / static_cast<double>(n_comp_cores);
            double comp_t = compute_model(per_core_work);
            
            // estimating the data loading time
            double io_size_gb = static_cast<double>(iops_sizes[i]) / GB;
            double io_thrpt = io_model(static_cast<double>(iops_counts[i]), io_size_gb);
            double io_t = io_size_gb / io_thrpt;

            total_time += std::max(comp_t, io_t);
        }
        double throughput = total_data_size / total_time;
        return throughput;
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
    optimize_throughput(const std::vector<size_t> &offsets,
                        size_t chunk_size,
                        int n_cores,
                        bool verbose = false) const {

        std::vector<size_t> batch_sizes;
        for (size_t mb = 1; mb <= 1024; mb *= 2) {
            batch_sizes.push_back(mb * 1024 * 1024);
        }

        double best_throughput = -std::numeric_limits<double>::infinity();
        int best_gap = -1;
        size_t best_batch_size = 0;

        for (int gap : gaps) {
            for (size_t batch_size : batch_sizes) {
                double throughput = score_function(batch_size, offsets, gap, chunk_size, n_cores);
                if (throughput > best_throughput) {
                    best_throughput = throughput;
                    best_gap = gap;
                    best_batch_size = batch_size;
                }
            }
        }

        if (verbose) {
            std::cout << "Optimal Throughput: "
                    << best_throughput << " GiB/s\n";
            std::cout << "Optimal Gap: "
                    << best_gap << " (gap units)\n";
            std::cout << "Optimal Batch size: "
                    << (best_batch_size >> 20) << " MB\n";
        }
        return {best_gap, best_batch_size};
    }
};
