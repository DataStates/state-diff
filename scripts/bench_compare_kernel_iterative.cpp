#include "Kokkos_Core.hpp"
#include <chrono>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

void test(size_t nthreads) {
    using Timer = std::chrono::high_resolution_clock;
    using Duration = std::chrono::duration<double>;
    uint64_t diff_count = 0;
    double init_time = 0.0;
    auto range_policy = Kokkos::RangePolicy<
        size_t, Kokkos::DefaultHostExecutionSpace>(0, nthreads);
    Timer::time_point beg_init = Timer::now();
    Kokkos::parallel_reduce(
        "Count differences", range_policy,
        KOKKOS_LAMBDA(const size_t idx, uint64_t &update) {
            update += 1;
        },
        Kokkos::Sum<uint64_t>(diff_count));
    Kokkos::fence();
    Timer::time_point end_init = Timer::now();
    init_time = std::chrono::duration_cast<Duration>(
                end_init - beg_init)
                .count();
    std::cout << "Init time = " << init_time*1000.0 << " ms\n";
}

int
main(int argc, char **argv) {
    using Timer = std::chrono::high_resolution_clock;
    using Duration = std::chrono::duration<double>;
    Kokkos::initialize(argc, argv);
    {
        size_t n_threads = Kokkos::num_threads();
	    // test(n_threads);

        size_t min_worksize = n_threads;
        size_t min_csize = 16; //1024;
        size_t max_csize = 1024; //32768;
        double error_tol = 0.001;
        std::mt19937 gen(0);
        std::uniform_real_distribution<float> dis(-1.0, 1.0);

        // Generate 1GB data to iterate over for comparison
        int n_gb = std::stoi(argv[1]);
        std::string logname = argv[2];
        size_t data_size = n_gb * 1ULL * 1024 * 1024 * 1024;
        size_t total_data_len = data_size / sizeof(float);
        std::vector<float> data0(total_data_len), data1(total_data_len);
        for (size_t j = 0; j < total_data_len; j++) {
            data0[j] = dis(gen);
            data1[j] = dis(gen);
        }
        
        std::ofstream logfile;
        logfile.precision(10);
        logfile.open(logname, std::ofstream::out | std::ofstream::app);
        if (logfile.tellp() == logfile.beg) {
            logfile << "Data size (B),Chunk size (B),Work size (chunk),Nthreads,Compute time (ms)\n";
        }

        for(size_t c_size = min_csize; c_size <= max_csize; c_size *= 2) {
            size_t tot_n_chunks = (data_size + c_size - 1) / c_size;
            for (size_t w_size = min_worksize; w_size <= tot_n_chunks; w_size *= 2) {
                printf("-------------------------------------------\n");
                printf("Running with Dsize = %zu, nthreads = %zu, chunk = %zu B, work size = %zu chunks\n", data_size, n_threads, c_size, w_size);
                size_t n_iter = (tot_n_chunks + w_size - 1) / w_size;
                size_t ndiff = 0;
                double compute_time = 0.0;
                for (size_t iter = 0; iter < n_iter; iter++) {
                    size_t chunk_len = c_size / sizeof(float);
                    size_t pos = iter * w_size * chunk_len;
                    const float *buff0 = data0.data() + pos;
                    const float *buff1 = data1.data() + pos;

                    uint64_t diff_count = 0;
                    Timer::time_point beg_compare = Timer::now();
                    auto range_policy = Kokkos::RangePolicy<
                        size_t, Kokkos::DefaultHostExecutionSpace>(0, w_size);
                    Kokkos::parallel_reduce(
                        "Count differences", range_policy,
                        KOKKOS_LAMBDA(const size_t idx, uint64_t &update) {
                            size_t start = idx * chunk_len;
                            for (size_t i = 0; i < chunk_len; i++) {
                                if (!(Kokkos::abs(buff0[start + i] -
                                                    buff1[start + i]) <=
                                        error_tol)) {
                                    update += 1;
                                }
                            }
                        },
                        Kokkos::Sum<uint64_t>(diff_count));
                    Kokkos::fence();
                    ndiff += diff_count;
                    Timer::time_point end_compare = Timer::now();
                    compute_time += std::chrono::duration_cast<Duration>(
                                end_compare - beg_compare)
                                .count();
                }
                logfile << data_size << ",";
                logfile << c_size << ",";
                logfile << w_size << ",";
		logfile << n_threads << ",";
                logfile << compute_time*1000.0 << std::endl; // seconds to millisecons
            }
        }
        logfile.close();    

    }
    Kokkos::finalize();
    return 0;
}
