#include "common/direct_io.hpp"
#include "liburing_reader.hpp"
#include "mpi.h"
#include "statediff.hpp"
#include "stdio.h"
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <argparse/argparse.hpp>
#include <cereal/archives/binary.hpp>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

int
main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    using Timer = std::chrono::high_resolution_clock;
    using Duration = std::chrono::duration<double>;
    Kokkos::initialize(argc, argv);
    {
        STDOUT_PRINT(
            "------------------------------------------------------\n");

        // Setup argument parser
        argparse::ArgumentParser program("statediff");
        program.add_argument("-v", "--verbose")
            .help("Compute differences between immutable data states")
            .default_value(false)
            .implicit_value(true);
        program.add_argument("-c", "--chunk-size")
            .required()
            .help("Chunk size in bytes")
            .scan<'u', uint32_t>();
        program.add_argument("-t", "--type")
            .required()
            .help("Data type")
            .default_value(std::string("float"))
            .choices("byte", "float", "double");
        program.add_argument("-e", "--error")
            .help("Error tolerance for comparing floating-point data")
            .default_value(static_cast<double>(0.0f))
            .scan<'g', double>();
        program.add_argument("-l", "--level")
            .help("Level to start/stop processing the tree. Root is level 0.")
            .default_value(static_cast<uint32_t>(13))
            .scan<'u', uint32_t>();
        program.add_argument("--host-cache")
            .help("Size of host cache for data transfers. (bytes)")
            .default_value(static_cast<size_t>(1073741824))
            .scan<'u', size_t>();
        program.add_argument("--dev-cache")
            .help("Size of device cache for data transfers. (bytes)")
            .default_value(static_cast<size_t>(1073741824))
            .scan<'u', size_t>();
        program.add_argument("--run0")
            .help("Checkpoint files for run 0")
            .nargs(argparse::nargs_pattern::any)
            .default_value(std::vector<std::string>());
        program.add_argument("--run1")
            .help("Checkpoint files for run 1")
            .nargs(argparse::nargs_pattern::any)
            .default_value(std::vector<std::string>());
        program.add_argument("--run0-full")
            .help("Full checkpoint files for run 0")
            .nargs(argparse::nargs_pattern::any)
            .default_value(std::vector<std::string>());
        program.add_argument("--run1-full")
            .help("Full checkpoint files for run 1")
            .nargs(argparse::nargs_pattern::any)
            .default_value(std::vector<std::string>());
        program.add_argument("-o", "--output-filename")
            .help("Save tree data to file")
            .default_value(std::string(""));
        program.add_argument("-r", "--result-logname")
            .help("Filename for storing csv logs")
            .default_value(std::string("result_log"));

        // Parse and retrieve arguments
        try {
            program.parse_args(argc, argv);
        } catch (const std::exception &err) {
            std::cerr << err.what() << std::endl;
            std::cerr << program;
            std::exit(1);
        }
        // Load arguments into convenience variables
        uint32_t chunk_size = program.get<uint32_t>("-c");
        std::string dtype = program.get<std::string>("--type");
        double err_tol = program.get<double>("--error");
        uint32_t level = program.get<uint32_t>("-l");
        size_t host_cache = program.get<size_t>("--host-cache");
        size_t dev_cache = program.get<size_t>("--dev-cache");
        auto run0_all_files = program.get<std::vector<std::string>>("--run0");
        auto run1_all_files = program.get<std::vector<std::string>>("--run1");
        auto run0_all_full_files =
            program.get<std::vector<std::string>>("--run0-full");
        auto run1_all_full_files =
            program.get<std::vector<std::string>>("--run1-full");
        std::string output_fname =
            program.get<std::string>("--output-filename");
        std::string logname = program.get<std::string>("--result-logname");
        STDOUT_PRINT("Chunk Size: %u\n", chunk_size);
        STDOUT_PRINT("Data Type:  %s\n", dtype.c_str());
        STDOUT_PRINT("Error Tol:  %s\n", err_tol);
        STDOUT_PRINT("Start Level %u\n", level);
        STDOUT_PRINT("Host Cache:  %s\n", dtype.c_str());
        STDOUT_PRINT("Dev Cache:  %s\n", dtype.c_str());

        std::sort(run0_all_files.begin(), run0_all_files.end());
        std::sort(run1_all_files.begin(), run1_all_files.end());
        std::sort(run0_all_full_files.begin(), run0_all_full_files.end());
        std::sort(run1_all_full_files.begin(), run1_all_full_files.end());

        int world_rank = 0, world_size = 1;
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        logname += "." + std::to_string(world_rank) + ".csv";
        if (world_rank == 0) {
            if (run0_all_files.size() > 0) {
                for (std::string str : run0_all_files) {
                    printf("Run 0 File: %s\n", str.c_str());
                }
            }
            if (run1_all_files.size() > 0) {
                for (std::string str : run1_all_files) {
                    printf("Run 1 File: %s\n", str.c_str());
                }
            }
            if (run0_all_full_files.size() > 0) {
                for (std::string str : run0_all_full_files) {
                    printf("Run 0 Full File: %s\n", str.c_str());
                }
            }
            if (run1_all_full_files.size() > 0) {
                for (std::string str : run1_all_full_files) {
                    printf("Run 1 Full File: %s\n", str.c_str());
                }
            }
        }
        std::string rank_str = "-" + std::to_string(world_rank) + "-";
        std::vector<std::string> run0_files, run1_files, run0_full_files,
            run1_full_files;
        if (run0_all_files.size() > 0) {
            for (uint32_t i = 0; i < run0_all_files.size(); i++) {
                if ((int)i % world_size == world_rank) {
                    run0_files.push_back(run0_all_files[i]);
                    try_free_page_cache(run0_all_files[i]);
                }
            }
            for (uint32_t i = 0; i < run0_all_full_files.size(); i++) {
                if ((int)i % world_size == world_rank) {
                    run0_full_files.push_back(run0_all_full_files[i]);
                    try_free_page_cache(run0_all_full_files[i]);
                }
            }
        }

        if (run1_all_files.size() > 0) {
            for (uint32_t i = 0; i < run1_all_files.size(); i++) {
                if ((int)i % world_size == world_rank) {
                    run1_files.push_back(run1_all_files[i]);
                    try_free_page_cache(run1_all_files[i]);
                }
            }
            for (uint32_t i = 0; i < run1_all_full_files.size(); i++) {
                if ((int)i % world_size == world_rank) {
                    run1_full_files.push_back(run1_all_full_files[i]);
                    try_free_page_cache(run1_all_full_files[i]);
                }
            }
        }
        uint32_t num_file_per_run = run0_files.size();
        bool comparing_runs = run1_files.size() == num_file_per_run;
        for (uint32_t i = 0; i < run0_files.size(); i++) {
            printf("Rank %d: Run 0 File %d: %s\n", world_rank, i,
                   run0_files[i].c_str());
        }
        for (uint32_t i = 0; i < run1_files.size(); i++) {
            printf("Rank %d: Run 1 File %d: %s\n", world_rank, i,
                   run1_files[i].c_str());
        }

        double timers[7] = {0.0};
        size_t elem_changed = 0;
        uint64_t changed_blocks = 0;
        uint64_t filtered_blocks = 0;
        uint64_t n_comparisons = 0;
        uint64_t n_hash_comp = 0;

        double setup_time = 0;
        double read_time = 0;
        double deserialize_time = 0;
        double compare_time1 = 0;
        double compare_time2 = 0;
        double serialize_time = 0;
        double write_time = 0;

        // Create statediff clients
        bool fuzzy_hash = true;
        size_t base_data_size = 0;
        std::string ref_file =
            comparing_runs ? run1_full_files[0] : run0_files[0];
        off_t filesize;
        get_file_size(ref_file, &filesize);
        size_t data_size = static_cast<size_t>(filesize);
        state_diff::client_t<float> client_cur(1, data_size, err_tol, dtype[0],
                                               chunk_size, level, fuzzy_hash, host_cache, dev_cache);
        if (comparing_runs) {
            off_t filesize;
            get_file_size(run0_files[0], &filesize);
            data_size = static_cast<size_t>(filesize);
            // assert(base_data_size == data_size);
        }
        state_diff::client_t<float> client_prev;

        
        MPI_Barrier(MPI_COMM_WORLD);
        // Iterate through files
        for (uint32_t idx = 0; idx < num_file_per_run; idx++) {
            std::cout << "Rank " << world_rank << ": Checkpoint " << idx
                      << std::endl;

            if (!comparing_runs) {
                // ================================================================
                // Setup
                // ================================================================
                Timer::time_point beg_setup = Timer::now();
                Kokkos::Profiling::pushRegion("Setup");
                liburing_io_reader_t reader_cur(run0_files[idx]);
                Kokkos::Profiling::popRegion();
                Timer::time_point end_setup = Timer::now();
                setup_time =
                    std::chrono::duration_cast<Duration>(end_setup - beg_setup)
                        .count();
                std::cout << "\tRank " << world_rank
                          << ": Setup: " << setup_time << std::endl;

                // ================================================================
                // Create tree
                // ================================================================
                Timer::time_point beg_create = Timer::now();
                Kokkos::Profiling::pushRegion("Create tree");
                client_cur.create(reader_cur);
                Kokkos::Profiling::popRegion();
                Timer::time_point end_create = Timer::now();
                double create_time = std::chrono::duration_cast<Duration>(
                                         end_create - beg_create)
                                         .count();
                std::cout << "\tRank " << world_rank
                          << ": Create Tree: " << create_time << std::endl;
                compare_time1 = create_time;

                // ================================================================
                // Serialize
                // ================================================================
                Timer::time_point beg_serialize = Timer::now();
                Kokkos::Profiling::pushRegion("Serialize");
                std::string outname = run0_files[idx] + std::string(".") +
                                      std::to_string(idx) +
                                      std::string(".compare-tree");
                {
                    std::ofstream ofs(outname, std::ios::binary);
                    cereal::BinaryOutputArchive oa(ofs);
                    oa(client_cur);
                    ofs.close();
                }
                Kokkos::Profiling::popRegion();
                Timer::time_point end_serialize = Timer::now();
                serialize_time = std::chrono::duration_cast<Duration>(
                                     end_serialize - beg_serialize)
                                     .count();
                std::cout << "\tRank " << world_rank
                          << ": Serialize: " << serialize_time << std::endl;
            } else {
                // ================================================================
                // Setup
                // ================================================================
                Timer::time_point beg_setup = Timer::now();
                Kokkos::Profiling::pushRegion("Setup");
                liburing_io_reader_t reader_prev(run0_full_files[idx]);
                liburing_io_reader_t reader_cur(run1_full_files[idx]);
                Kokkos::Profiling::popRegion();
                Timer::time_point end_setup = Timer::now();
                setup_time =
                    std::chrono::duration_cast<Duration>(end_setup - beg_setup)
                        .count();
                std::cout << "\tRank " << world_rank
                          << ": Setup: " << setup_time << std::endl;

                // ================================================================
                // Deserialize
                // ================================================================
                Timer::time_point beg_deserialize = Timer::now();
                Kokkos::Profiling::pushRegion("Deserialize");
                {
                    std::ifstream ifs(run0_files[idx], std::ios::binary);
                    cereal::BinaryInputArchive ia(ifs);
                    ia(client_prev);
                    ifs.close();
                }
                {
                    std::ifstream ifs(run1_files[idx], std::ios::binary);
                    cereal::BinaryInputArchive ia(ifs);
                    ia(client_cur);
                    ifs.close();
                }
                Kokkos::Profiling::popRegion();
                Timer::time_point end_deserialize = Timer::now();
                deserialize_time = std::chrono::duration_cast<Duration>(
                                       end_deserialize - beg_deserialize)
                                       .count();
                std::cout << "\tRank " << world_rank
                          << ": Deserialize: " << deserialize_time << std::endl;

                // ================================================================
                // Compare
                // ================================================================
                Kokkos::Profiling::pushRegion("Compare phase");
                client_cur.compare_with(0, reader_cur, client_prev,
                                        reader_prev);
                compare_time1 = client_cur.get_tree_comparison_time();
                compare_time2 = client_cur.get_data_compare_time();
                Kokkos::Profiling::popRegion();
                std::cout << "\tRank " << world_rank
                          << ": Compare Tree Phase 1: " << compare_time1
                          << std::endl;

                std::cout << "\tRank " << world_rank
                          << ": Compare Tree Phase 2: " << compare_time2
                          << std::endl;

                std::vector<double> compare_time =
                    client_cur.get_compare_time();
                std::cout << "\t\tRank " << world_rank << ": Compare Time: "
                          << std::reduce(compare_time.begin(),
                                         compare_time.end())
                          << std::endl;
            }

            // ========================================================================================
            // Collect stats for logs
            // ========================================================================================
            timers[0] = setup_time;
            timers[1] = read_time;
            timers[2] = deserialize_time;
            timers[3] = compare_time1;
            timers[4] = compare_time2;
            timers[5] = serialize_time;
            timers[6] = write_time;
            n_comparisons = client_cur.get_num_comparisons();
            n_hash_comp = client_cur.get_num_hash_comparisons();
            elem_changed = client_cur.get_num_changes();
            filtered_blocks = client_cur.get_filtered_blocks();
            changed_blocks = client_cur.get_validated_diffs();
            printf("Rank %d: Number of different elements %zu\n", world_rank,
                   elem_changed);
            printf("Rank %d: Number of comparisons %lu\n", world_rank,
                   n_comparisons);
            printf("Rank %d: Number of hash comparisons %lu\n", world_rank,
                   n_hash_comp);
            printf("Rank %d: Number of different hashes (Phase 1) %zu\n",
                   world_rank, filtered_blocks);
            printf("Rank %d: Number of different hashes (Phase 2) %zu\n\n",
                   world_rank, changed_blocks);

            Kokkos::fence();

            // ========================================================================================
            // Write log
            // ========================================================================================
            std::ofstream logfile;
            logfile.open(logname, std::ofstream::out | std::ofstream::app);
            logfile.precision(10);
            if (logfile.tellp() == logfile.beg) {
                logfile << "Rank,File,File size,Baseline file,Baseline file "
                           "size,Hash function,Chunk size,Data type,";
                logfile
                    << "Error tolerance,Start level,Host cache,Device cache,";
                logfile
                    << "Setup time,Read time,Deserialization time,Construction "
                       "time,Compare tree time,Compare direct "
                       "time,Serialization time,Write time,";
                logfile << "Elements different,Hashes different,Num "
                           "comparisons,Num hash comparisons,Filtered hashes\n";
            }
            logfile << world_rank << ",";
            if (comparing_runs) {
                logfile << run1_files[idx] << ",";
            } else {
                logfile << run0_files[idx] << ",";
            }
            logfile << data_size << ",";
            if (comparing_runs) {
                logfile << run0_files[idx] << ",";
                logfile << base_data_size << ",";
            } else {
                logfile << ",,";
            }
            if (fuzzy_hash) {
                logfile << "Fuzzy hash,";
            } else {
                logfile << "Murmur3,";
            }
            logfile << chunk_size << ",";
            logfile << dtype << ",";
            logfile << err_tol << ",";
            logfile << level << ",";
            logfile << host_cache << ",";
            logfile << dev_cache << ",";
            logfile << timers[0] << ",";
            logfile << timers[1] << ",";
            logfile << timers[2] << ",";
            if (comparing_runs) {
                logfile << "0," << timers[3] << "," << timers[4] << ",";
            } else {
                logfile << timers[3] << ",0,0,";
            }
            logfile << timers[5] << ",";
            logfile << timers[6] << ",";
            logfile << elem_changed << ",";
            logfile << changed_blocks << ",";
            logfile << n_comparisons << ",";
            logfile << n_hash_comp << ",";
            logfile << filtered_blocks << std::endl;
            logfile.close();
        }
    }
    Kokkos::finalize();
    DEBUG_PRINT("Done finalizing Kokkos\n");
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;
}
