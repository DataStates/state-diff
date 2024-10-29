#ifndef CS_SCHEDULER_H
#define CS_SCHEDULER_H

#include <vector>

class CS_Scheduler {
private:
    double alpha;
    double beta;
    double gamma;
    int min_chunk_size;
    int max_chunk_size;

    double estimate_total_time(int chunk_size, int segment_size) const;
    int binary_search_chunk_size(int segment_size) const;

public:
    CS_Scheduler(double alpha, double beta, double gamma, int min_chunk, int max_chunk);

    std::vector<int> optimize_chunk_sizes(const std::vector<int> &segment_sizes) const;
    double compute_total_time(const std::vector<int> &segment_sizes, const std::vector<int> &chunk_sizes) const;
};


#include "cs_scheduler.h"
#include <cmath>

// Constructor to initialize the CS_Scheduler with necessary parameters
CS_Scheduler::CS_Scheduler(double alpha, double beta, double gamma, int min_chunk, int max_chunk)
    : alpha(alpha), beta(beta), gamma(gamma), min_chunk_size(min_chunk), max_chunk_size(max_chunk) {}

// Function to estimate the total time based on a given chunk size for one segment
double CS_Scheduler::estimate_total_time(int chunk_size, int segment_size) const {
    int num_chunks = segment_size / chunk_size;

    double stage1_time = alpha * (segment_size / chunk_size);   // Tree construction
    double stage2_time = beta * std::log2(num_chunks);          // Hash comparison
    double stage3_time = gamma * (num_chunks * chunk_size);     // Bitwise float comparison

    return stage1_time + stage2_time + stage3_time;
}

// Binary search to find the optimal chunk size for a specific segment
int CS_Scheduler::binary_search_chunk_size(int segment_size) const {
    int optimal_chunk_size = min_chunk_size;
    double min_total_time = estimate_total_time(min_chunk_size, segment_size);

    int low = min_chunk_size;
    int high = max_chunk_size;

    while (low <= high) {
        int mid_chunk_size = low + (high - low) / 2;
        double total_time = estimate_total_time(mid_chunk_size, segment_size);

        if (total_time < min_total_time) {
            min_total_time = total_time;
            optimal_chunk_size = mid_chunk_size;
        }

        // Binary search logic to minimize total time
        if (total_time < min_total_time) {
            high = mid_chunk_size - 1;
        } else {
            low = mid_chunk_size + 1;
        }
    }

    return optimal_chunk_size;
}

// Multi-segment chunk size optimization
std::vector<int> CS_Scheduler::optimize_chunk_sizes(const std::vector<int> &segment_sizes) const {
    std::vector<int> optimal_chunk_sizes;

    for (int segment_size : segment_sizes) {
        int optimal_chunk_size = binary_search_chunk_size(segment_size);
        optimal_chunk_sizes.push_back(optimal_chunk_size);
    }

    return optimal_chunk_sizes;
}

// Total time for the entire file using the optimal chunk sizes for each segment
double CS_Scheduler::compute_total_time(const std::vector<int> &segment_sizes, const std::vector<int> &chunk_sizes) const {
    double total_time = 0.0;

    for (size_t i = 0; i < segment_sizes.size(); ++i) {
        total_time += estimate_total_time(chunk_sizes[i], segment_sizes[i]);
    }

    return total_time;
}


#endif // CS_SCHEDULER_H
