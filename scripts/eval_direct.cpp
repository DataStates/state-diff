#include "direct_comparer.hpp"
#include "direct_io.hpp"
#include "liburing_reader.hpp"
#include "mpi.h"
#include "stdio.h"
#include <Kokkos_Core.hpp>
#include <argparse/argparse.hpp>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

int
main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    Kokkos::initialize(argc, argv);
    {
        STDOUT_PRINT(
            "------------------------------------------------------\n");

        // Setup argument parser
        argparse::ArgumentParser program("statediff");
        program.add_argument("-v", "--verbose")
            .help(
                "Compute differences between two files using direct comparison")
            .default_value(false)
            .implicit_value(true);
        program.add_argument("-b", "--block_size")
            .help("Batch size in bytes")
            .default_value(static_cast<size_t>(1073741824))
            .scan<'u', size_t>();
        program.add_argument("-t", "--type")
            .required()
            .help("Data type")
            .default_value(std::string("float"))
            .choices("byte", "float", "double");
        program.add_argument("-e", "--error")
            .help("Error tolerance for comparing floating-point data")
            .default_value(static_cast<double>(0.0f))
            .scan<'g', double>();
        program.add_argument("--run0")
            .help("Checkpoint files for run 0")
            .nargs(argparse::nargs_pattern::any)
            .default_value(std::vector<std::string>());
        program.add_argument("--run1")
            .help("Checkpoint files for run 1")
            .nargs(argparse::nargs_pattern::any)
            .default_value(std::vector<std::string>());
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
        // Load arguments into convenience variables1
        std::string dtype = program.get<std::string>("--type");
        double err_tol = program.get<double>("--error");
        size_t block_size = program.get<size_t>("-b");
        auto run0_all_files = program.get<std::vector<std::string>>("--run0");
        auto run1_all_files = program.get<std::vector<std::string>>("--run1");
        std::string logname = program.get<std::string>("--result-logname");

        std::sort(run0_all_files.begin(), run0_all_files.end());
        std::sort(run1_all_files.begin(), run1_all_files.end());

        int world_rank = 0, world_size = 1;
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
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
        }

        if (run1_all_files.size() > 0) {
            for (uint32_t i = 0; i < run1_all_files.size(); i++) {
                if ((int)i % world_size == world_rank) {
                    run1_files.push_back(run1_all_files[i]);
                    try_free_page_cache(run1_all_files[i]);
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

        size_t elem_changed = 0;
        uint64_t changed_blocks = 0;
        uint64_t n_comparisons = 0;
        double compare_time = 0;
        double total_time = 0;

        off_t filesize;
        get_file_size(run0_files[0], &filesize);
        size_t data_size = static_cast<size_t>(filesize);
        if (comparing_runs) {
            off_t meta_filesize;
            get_file_size(run0_files[0], &meta_filesize);
        }

        DirectComparer<float> comparator(data_size, err_tol, block_size);

        MPI_Barrier(MPI_COMM_WORLD);
        // Iterate through files
        for (uint32_t idx = 0; idx < num_file_per_run; idx++) {
            std::cout << "Rank " << world_rank << ": Checkpoint " << idx
                      << std::endl;
            // ================================================================
            // Compare
            // ================================================================
            Kokkos::Profiling::pushRegion("Compare");
            liburing_io_reader_t reader_prev(run0_files[idx]);
            liburing_io_reader_t reader_cur(run1_files[idx]);
            elem_changed = comparator.compare(reader_prev, reader_cur);
            changed_blocks = comparator.get_num_changed_blocks();
            n_comparisons = comparator.get_num_comparisons();
            compare_time = comparator.get_compare_time();
            total_time = comparator.get_total_time();
            std::cout << "Rank " << world_rank
                      << ": Compare Time: " << total_time << std::endl;

            // ========================================================================================
            // Collect stats for logs
            // ========================================================================================
            printf("Rank %d: Number of different elements %zu\n", world_rank,
                   elem_changed);
            printf("Rank %d: Number of comparisons %lu\n", world_rank,
                   n_comparisons);
            printf("Rank %d: Number of different blocks %zu\n\n", world_rank,
                   changed_blocks);

            Kokkos::fence();

            // ========================================================================================
            // Write log
            // ========================================================================================
            std::ofstream logfile;
            logfile.precision(10);
            logname += "." + std::to_string(world_rank) + ".compare.csv";
            logfile.open(logname, std::ofstream::out | std::ofstream::app);
            if (logfile.tellp() == logfile.beg) {
                logfile << "File,Data filesize,Error tolerance,Block "
                           "size,Elements different,Block different,"
                           "Num comparisons,Compute time,Total time\n";
            }
            logfile << run1_files[idx] << ",";
            logfile << data_size << ",";
            logfile << err_tol << ",";
            logfile << block_size << ",";
            logfile << elem_changed << ",";
            logfile << changed_blocks << ",";
            logfile << n_comparisons << ",";
            logfile << compare_time << ",";
            logfile << total_time << std::endl;
            logfile.close();
        }
    }
    Kokkos::finalize();
    DEBUG_PRINT("Done finalizing Kokkos\n");
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;
}
