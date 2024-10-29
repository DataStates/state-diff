#include "cs_scheduler.h"
#include <iostream>

int main() {
    // Example: We divide a file of size 1,000,000 bytes into 3 segments
    std::vector<int> segment_sizes = {300000, 400000, 300000}; // Each segment has a different size
    int min_chunk_size = 1024;     // Minimum chunk size in bytes
    int max_chunk_size = 65536;    // Maximum chunk size in bytes

    // Define the coefficients for stage 1, 2, and 3 calculations
    double alpha = 1.0;
    double beta = 1.5;
    double gamma = 0.5;

    // Create an instance of CS_Scheduler
    CS_Scheduler scheduler(alpha, beta, gamma, min_chunk_size, max_chunk_size);

    // Optimize chunk sizes for each segment
    std::vector<int> optimal_chunk_sizes = scheduler.optimize_chunk_sizes(segment_sizes);

    // Output the optimal chunk sizes for each segment
    std::cout << "Optimal chunk sizes for each segment:" << std::endl;
    for (size_t i = 0; i < optimal_chunk_sizes.size(); ++i) {
        std::cout << "Segment " << i + 1 << ": " << optimal_chunk_sizes[i] << " bytes" << std::endl;
    }

    // Compute the total time for the file with these optimal chunk sizes
    double total_time = scheduler.compute_total_time(segment_sizes, optimal_chunk_sizes);
    std::cout << "Total time: " << total_time << " units" << std::endl;

    return 0;
}
