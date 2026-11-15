#include "Kokkos_Core.hpp"
#include <chrono>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

int
main(int argc, char **argv) {
    using Timer = std::chrono::high_resolution_clock;
    using Duration = std::chrono::duration<double>;
    Kokkos::initialize(argc, argv);
    {
        size_t n_threads = Kokkos::num_threads();
        size_t min_worksize = n_threads;
        size_t max_worksize = 65536;
        size_t min_csize = 128;
        size_t max_csize = 131072;
        double error_tol = 0.001f;
        std::mt19937 gen(0);
        std::uniform_real_distribution<float> dis(-1.0, 1.0);
        
        std::ofstream logfile;
        logfile.precision(10);
        std::string logname = "compute_benchmark.csv";
        logfile.open(logname, std::ofstream::out | std::ofstream::app);
        if (logfile.tellp() == logfile.beg) {
            logfile << "Nthreads,Work size (B),Compute time (ms)\n";
        }

        for (size_t w_size = min_worksize; w_size <= max_worksize; w_size *= 2) {
            for(size_t c_size = min_csize; c_size <= max_csize; c_size *= 2) {
                printf("-------------------------------------------\n");
                size_t data_size = w_size * c_size;
                size_t chunk_len = c_size / sizeof(float);
                size_t data_len = w_size*chunk_len;
                std::vector<float> data0(data_len), data1(data_len);
                for (size_t j = 0; j < data_len; j++) {
                    data0[j] = dis(gen);
                    data1[j] = dis(gen);
                }
                const float *buff0 = data0.data();
                const float *buff1 = data1.data();

                std::vector<size_t> offsets(w_size);
                for(size_t i = 0; i < w_size; i++) {
                    offsets.push_back(i);
                }
                uint64_t diff_count = 0;
                printf("Running with nthreads = %zu, work size = %zu B\n", n_threads, data_size);
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
                Timer::time_point end_compare = Timer::now();
                double compute_time = std::chrono::duration_cast<Duration>(
                                            end_compare - beg_compare)
                                            .count();
                logfile << n_threads << ",";
                logfile << data_size << ",";
                logfile << compute_time*1000.0 << std::endl;
            }
        }
        logfile.close();    
    }
    Kokkos::finalize();
}