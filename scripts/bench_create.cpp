#include "Kokkos_Core.hpp"
#include <chrono>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

size_t
elm_work_sharing(const float *buff0, const float *buff1, float error_tol,
                 int work_start, int work) {
    size_t ndiff = 0;
    auto range_policy =
        Kokkos::RangePolicy<int, Kokkos::DefaultHostExecutionSpace>(0, work);
    Kokkos::parallel_reduce(
        "Count differences", range_policy,
        KOKKOS_LAMBDA(const int idx, uint64_t &update) {
            int elm_idx = work_start + idx;
            if (!(Kokkos::abs(buff0[elm_idx] - buff1[elm_idx]) <= error_tol)) {
                update += 1;
            }
        },
        Kokkos::Sum<uint64_t>(ndiff));
    Kokkos::fence();
    return ndiff;
}

int
main(int argc, char **argv) {
    using Timer = std::chrono::high_resolution_clock;
    using Duration = std::chrono::duration<double>;
    Kokkos::initialize(argc, argv);
    {
        int base_data_len = 4 * 1024 * 1024;   // number of element
        int exp_count = 3;
        int chunk_len = 1024;

        for (int i = 0; i < exp_count; i++) {
            printf("-------------------------------------------\n");
            int data_len = base_data_len * (i + 1) * 4;
            std::vector<float> data0(data_len), data1(data_len);
            float error_tol = 0.001;

            // data gen
            std::mt19937 gen(0);
            std::uniform_real_distribution<float> dis(-1.0, 1.0);
            for (int j = 0; j < data_len; j++) {
                data0[j] = dis(gen);
                data1[j] = dis(gen);
            }
            const float *buff0 = data0.data();
            const float *buff1 = data1.data();

            std::vector<int> test_cases = {1, 2, 3, 4, 5};

            for (size_t i = 0; i < test_cases.size(); i++) {
                int approach = test_cases[i];
                size_t diff_count = 0;
                std::string share_type = "";
                Timer::time_point beg_compare = Timer::now();
                if (approach == 1) {
                    share_type = "Element-based partitioning where 1 chunk is processed "
                                 "by N cores at a time";
                    int n_proc = data_len / chunk_len;
                    for (int cur = 0; cur < n_proc; cur++) {
                        int work = chunk_len;
                        int work_start = cur * work;
                        diff_count += elm_work_sharing(buff0, buff1, error_tol,
                                                       work_start, work);
                    }
                } else if (approach == 2) {
                    share_type = "Element-based partitioning where N chunks are processed "
                                 "by N cores at a time";
                    int n_proc = Kokkos::num_threads();
                    int n_chunks = data_len / chunk_len;
                    int n_iter = n_chunks / n_proc;
                    if (n_iter * n_proc < n_chunks)
                        n_iter += 1;

                    int work_done = 0;
                    int work_start = 0;
                    for (int cur = 0; cur < n_iter; cur++) {
                        int blk_size =
                            (cur == n_iter - 1) ? n_chunks - work_done : n_proc;
                        int work = blk_size * chunk_len;
                        diff_count += elm_work_sharing(buff0, buff1, error_tol,
                                                       work_start, work);
                        work_done += blk_size;
                        work_start += work;
                    }
                } else if (approach == 3) {
                    share_type = "Element-based partitioning where all chunks are "
                                 "processed by N cores at once";
                    int work = data_len;
                    int work_start = 0;
                    diff_count = elm_work_sharing(buff0, buff1, error_tol,
                                                  work_start, work);
                } else if (approach == 4) {
                    share_type = "Chunk-based partitioning where N chunks are processed "
                                 "by N cores at a time";
                    int n_proc = Kokkos::num_threads();
                    int n_chunks = data_len / chunk_len;
                    int n_iter = n_chunks / n_proc;
                    if (n_iter * n_proc < n_chunks)
                        n_iter += 1;

                    int work_done = 0;
                    for (int cur = 0; cur < n_iter; cur++) {
                        int work =
                            (cur == n_iter - 1) ? n_chunks - work_done : n_proc;
                        size_t ndiff = 0;
                        auto range_policy = Kokkos::RangePolicy<
                            int, Kokkos::DefaultHostExecutionSpace>(0, work);
                        Kokkos::parallel_reduce(
                            "Count differences", range_policy,
                            KOKKOS_LAMBDA(const int idx, uint64_t &update) {
                                int chunk_offt = work_done + idx;
                                int start = chunk_offt * chunk_len;
                                for (int i = 0; i < chunk_len; i++) {
                                    if (!(Kokkos::abs(buff0[start + i] -
                                                      buff1[start + i]) <=
                                          error_tol)) {
                                        update += 1;
                                    }
                                }
                            },
                            Kokkos::Sum<uint64_t>(ndiff));
                        Kokkos::fence();
                        work_done += work;
                        diff_count += ndiff;
                    }
                } else {
                    share_type = "Chunk-based partitioning where all chunks are processed "
                                 "by N cores at once";
                    int n_chunks = data_len / chunk_len;
                    auto range_policy = Kokkos::RangePolicy<
                        int, Kokkos::DefaultHostExecutionSpace>(0, n_chunks);
                    Kokkos::parallel_reduce(
                        "Count differences", range_policy,
                        KOKKOS_LAMBDA(const int idx, uint64_t &update) {
                            int start = idx * chunk_len;
                            for (int i = 0; i < chunk_len; i++) {
                                if (!(Kokkos::abs(buff0[start + i] -
                                                  buff1[start + i]) <=
                                      error_tol)) {
                                    update += 1;
                                }
                            }
                        },
                        Kokkos::Sum<uint64_t>(diff_count));
                    Kokkos::fence();
                }
                Timer::time_point end_compare = Timer::now();
                double compute_time = std::chrono::duration_cast<Duration>(
                                          end_compare - beg_compare)
                                          .count();
                std::cout << "(Case " << approach
                          << ", data len = " << data_len
                          << ", chunk len = " << chunk_len << ") " << share_type
                          << ": Ndiff = " << diff_count
                          << ", Compute time = " << compute_time * 1000.0
                          << " msecs" << std::endl;
            }
        }
    }
    Kokkos::finalize();
}

// int
// main(int argc, char **argv) {
//     using Timer = std::chrono::high_resolution_clock;
//     using Duration = std::chrono::duration<double>;
//     Kokkos::initialize(argc, argv);
//     {
//         int base_data_len = 4 * 1024;   // number of element
//         int exp_count = 3;
//         int chunk_size = 1024;
//         int approach = std::stoi(argv[1]);

//         for (int i = 0; i < exp_count; i++) {
//             int data_len = base_data_len * (i + 1) * 4;
//             std::vector<float> data0(data_len), data1(data_len);
//             float error_tol = 0.001;

//             // data gen
//             std::mt19937 gen(0);
//             std::uniform_real_distribution<float> dis(-1.0, 1.0);
//             for (int j = 0; j < data_len; j++) {
//                 data0[j] = dis(gen);
//                 data1[j] = dis(gen);
//             }
//             float *buff0 = data0.data();
//             float *buff1 = data1.data();

//             Timer::time_point beg_compare = Timer::now();
//             if (approach == 1) {
//                 // all procs are processing one chunk at a time
//                 std::cout << "All procs are processing one chunk at a time"
//                           << std::endl;
//                 int n_proc = data_len / chunk_size;
//                 int tot_diff = 0;
//                 for (int cur = 0; cur < n_proc; cur++) {
//                     size_t ndiff = 0;
//                     auto range_policy =
//                         Kokkos::RangePolicy<size_t,
//                                             Kokkos::DefaultHostExecutionSpace>(
//                             0, chunk_size);
//                     Kokkos::parallel_reduce(
//                         "Count differences", range_policy,
//                         KOKKOS_LAMBDA(const size_t idx, uint64_t &update) {
//                             int start = cur * chunk_size;
//                             bool inrange =
//                                 Kokkos::abs(buff0[start + idx] -
//                                             buff1[start + idx]) <= error_tol;
//                             if (!inrange) {
//                                 update += 1;
//                             }
//                         },
//                         Kokkos::Sum<uint64_t>(ndiff));
//                     tot_diff += ndiff;
//                 }
//             } else if (approach == 2) {   // all procs are processing all
//             data
//                                           // (one element per proc)
//                 std::cout << "All procs are processing all data (one element
//                 "
//                              "per proc)"
//                           << std::endl;
//                 size_t ndiff = 0;
//                 auto range_policy = Kokkos::RangePolicy<
//                     size_t, Kokkos::DefaultHostExecutionSpace>(0, data_len);
//                 Kokkos::parallel_reduce(
//                     "Count differences", range_policy,
//                     KOKKOS_LAMBDA(const size_t idx, uint64_t &update) {
//                         bool inrange =
//                             Kokkos::abs(buff0[idx] - buff1[idx]) <=
//                             error_tol;
//                         if (!inrange) {
//                             update += 1;
//                         }
//                     },
//                     Kokkos::Sum<uint64_t>(ndiff));
//             } else if (approach == 3) {   // A core is processing a full
//             chunk
//                 std::cout << "A core is processing a full chunk" <<
//                 std::endl; size_t n_proc = data_len / chunk_size; size_t
//                 ndiff = 0; auto range_policy = Kokkos::RangePolicy<
//                     size_t, Kokkos::DefaultHostExecutionSpace>(0, n_proc);
//                 Kokkos::parallel_reduce(
//                     "Count differences", range_policy,
//                     KOKKOS_LAMBDA(const size_t idx, uint64_t &update) {
//                         int start = idx * chunk_size;
//                         for (int i = 0; i < chunk_size; i++) {
//                             if (!(Kokkos::abs(buff0[start + i] -
//                                               buff1[start + i]) <=
//                                               error_tol)) {
//                                 update += 1;
//                             }
//                         }
//                     },
//                     Kokkos::Sum<uint64_t>(ndiff));
//             } else if (approach ==
//                        4) {   // Similar to approach 4 but using a team
//                        policy
//                 std::cout << "A core is processing a full chunk using a team"
//                           << std::endl;
//                 size_t n_proc = data_len / chunk_size;
//                 size_t ndiff = 0;
//                 using TeamPolicy =
//                     Kokkos::TeamPolicy<Kokkos::DefaultHostExecutionSpace>;
//                 using MemberType = TeamPolicy::member_type;

//                 auto team_policy = TeamPolicy(n_proc, Kokkos::AUTO);
//                 Kokkos::parallel_reduce(
//                     "Count differences", team_policy,
//                     KOKKOS_LAMBDA(const MemberType &team, uint64_t &update) {
//                         int idx = team.league_rank();
//                         int start = idx * chunk_size;
//                         uint64_t local_count = 0;

//                         Kokkos::parallel_reduce(
//                             Kokkos::TeamThreadRange(team, chunk_size),
//                             [&](int i, uint64_t &local_update) {
//                                 if (!(Kokkos::abs(buff0[start + i] -
//                                                   buff1[start + i]) <=
//                                       error_tol)) {
//                                     local_update += 1;
//                                 }
//                             },
//                             local_count);

//                         Kokkos::single(Kokkos::PerTeam(team),
//                                        [&]() { update += local_count; });
//                     },
//                     Kokkos::Sum<uint64_t>(ndiff));
//             }
//             Timer::time_point end_compare = Timer::now();
//             double compute_time =
//                 std::chrono::duration_cast<Duration>(end_compare -
//                 beg_compare)
//                     .count();

//             printf(
//                 "(Approach %i) Compute time with chunk size = %i is %f
//                 msecs\n", approach, data_len, compute_time * 1000.0);
//         }
//     }
//     Kokkos::finalize();
// }