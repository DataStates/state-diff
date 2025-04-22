#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <string>
#include <vector>
#include <fstream>
#include <chrono>
#include <iostream>
#include <filesystem>
#include <argparse/argparse.hpp>
#include "stdio.h"
#include "direct_io.hpp"
#include "state_diff.hpp"
#include "mpi.h"

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  using Timer = std::chrono::high_resolution_clock;
  using Duration = std::chrono::duration<double>;
  Kokkos::initialize(argc, argv);
  {
    STDOUT_PRINT("------------------------------------------------------\n");

    // Setup argument parser
    argparse::ArgumentParser program("Dedup Files");
    program.add_argument("-v", "--verbose")
      .help("Deduplicate files")
      .default_value(false)
      .implicit_value(true);
    program.add_argument("-c", "--chunk-size")
      .required()
      .help("Chunk size in bytes")
      .scan<'u', uint32_t>();
    program.add_argument("-a", "--alg")
      .required()
      .help("Select algorithm")
      .default_value(std::string("direct"))
      .choices("direct", "compare-tree");
    program.add_argument("-l", "--level")
      .help("Level to start/stop processing the tree. Root is level 0.")
      .default_value(static_cast<uint32_t>(13))
      .scan<'u', uint32_t>();
    program.add_argument("--enable-file-streaming")
      .help("Use file streaming for direct comparisons.")
      .default_value(false)
      .implicit_value(true);
    program.add_argument("--buffer-len")
      .help("Size of device buffers used for asynchronous data transfers. (bytes)")
      .default_value(static_cast<size_t>(1073741824))
      .scan<'u', size_t>();
    program.add_argument("--run0")
      .help("Checkpoint files for run 0")
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
    program.add_argument("--run1")
      .help("Checkpoint files for run 1")
      .nargs(argparse::nargs_pattern::any)
      .default_value(std::vector<std::string>());
    program.add_argument("-o", "--output-filename")
      .help("Save tree data to file")
      .default_value(std::string(""));
    program.add_argument("-r", "--result-logname")
      .help("Filename for storing csv logs")
      .default_value(std::string("result_log"));
    program.add_argument("-e", "--error")
      .help("Error tolerance for comparing floating-point data")
      .default_value(static_cast<double>(0.0f))
      .scan<'g', double>();

    // Parse and retrieve arguments
    try {
      program.parse_args(argc, argv);
    } catch (const std::exception& err) {
      std::cerr << err.what() << std::endl;
      std::cerr << program;
      std::exit(1);
    }
    // Load arguments into convenience variables
    uint32_t chunk_size = program.get<uint32_t>("-c");
    uint32_t level = program.get<uint32_t>("-l");
    bool fuzzy_hash = true;
    bool async_stream = true;
    bool enable_file_streaming = program["--enable-file-streaming"] == true;
    size_t buffer_len = program.get<size_t>("--buffer-len");
    std::string alg     = program.get<std::string>("--alg");
    std::string dtype   = "float";
    std::string logname = program.get<std::string>("--result-logname");
    auto run0_all_files = program.get<std::vector<std::string>>("--run0");
    auto run1_all_files = program.get<std::vector<std::string>>("--run1");
    auto run0_all_full_files = program.get<std::vector<std::string>>("--run0-full");
    auto run1_all_full_files = program.get<std::vector<std::string>>("--run1-full");
    std::sort(run0_all_files.begin(), run0_all_files.end());
    std::sort(run1_all_files.begin(), run1_all_files.end());
    std::sort(run0_all_full_files.begin(), run0_all_full_files.end());
    std::sort(run1_all_full_files.begin(), run1_all_full_files.end());

    double err_tol = program.get<double>("--error");
    std::string output_fname = program.get<std::string>("--output-filename");
    STDOUT_PRINT("Chunk size: %u\n", chunk_size);
    STDOUT_PRINT("Start level %u\n", level);
    STDOUT_PRINT("Algorithm:  %s\n", alg.c_str());
    STDOUT_PRINT("Data type:  %s\n", dtype.c_str());

    int world_rank=0, world_size=1;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    logname += "." + std::to_string(world_rank) + ".csv";
    //logname += ".csv";
    if(world_rank ==0) {
      if(run0_all_files.size() > 0) {
        for(std::string str: run0_all_files) {
          printf("Run 0 File: %s\n", str.c_str());
        }
      }
      if(run1_all_files.size() > 0) {
        for(std::string str: run1_all_files) {
          printf("Run 1 File: %s\n", str.c_str());
        }
      }
      if(run0_all_full_files.size() > 0) {
        for(std::string str: run0_all_full_files) {
          printf("Run 0 Full File: %s\n", str.c_str());
        }
      }
      if(run1_all_full_files.size() > 0) {
        for(std::string str: run1_all_full_files) {
          printf("Run 1 Full File: %s\n", str.c_str());
        }
      }
    }
    std::string rank_str = "-" + std::to_string(world_rank) + "-";
    std::vector<std::string> run0_files, run1_files, run0_full_files, run1_full_files;
    if(run0_all_files.size() > 0) {
      for(uint32_t i=0; i<run0_all_files.size(); i++) {
        if((int)i % world_size == world_rank) {
          run0_files.push_back(run0_all_files[i]);
          try_free_page_cache(run0_all_files[i]);
        }
      }
      for(uint32_t i=0; i<run0_all_full_files.size(); i++) {
        if((int)i % world_size == world_rank) {
          run0_full_files.push_back(run0_all_full_files[i]);
          try_free_page_cache(run0_all_full_files[i]);
        }
      }
    }
    if(run1_all_files.size() > 0) {
      for(uint32_t i=0; i<run1_all_files.size(); i++) {
        if((int)i % world_size == world_rank) {
          run1_files.push_back(run1_all_files[i]);
          try_free_page_cache(run1_all_files[i]);
        }
      }
      for(uint32_t i=0; i<run1_all_full_files.size(); i++) {
        if((int)i % world_size == world_rank) {
          run1_full_files.push_back(run1_all_full_files[i]);
          try_free_page_cache(run1_all_full_files[i]);
        }
      }
    }
    uint32_t num_diffs = run0_files.size();
    bool comparing_runs = run1_files.size() == num_diffs;
    for(uint32_t i=0; i<run0_files.size(); i++) {
      printf("Rank %d: Run 0 File %d: %s\n", world_rank, i, run0_files[i].c_str());
    }
    for(uint32_t i=0; i<run1_files.size(); i++) {
      printf("Rank %d: Run 1 File %d: %s\n", world_rank, i, run1_files[i].c_str());
    }

    double timers[7] = {0.0};
    size_t elem_changed = 0;
    uint64_t changed_blocks = 0;
    uint64_t filtered_blocks = 0;
    uint64_t n_comparisons = 0;
    uint64_t n_hash_comp = 0;

    // Create deduplicators
    CompareTreeDeduplicator comp_deduplicator(chunk_size, level, fuzzy_hash, err_tol, dtype[0]);
    comp_deduplicator.comp_op = Absolute;
    DirectComparer<float> f32_comparer(err_tol, chunk_size/sizeof(float), buffer_len/sizeof(float));

    MPI_Barrier(MPI_COMM_WORLD);
    using ByteDeviceView = Kokkos::View<uint8_t*>;
    using UnmanagedByteHostView = Kokkos::View<uint8_t*, Kokkos::DefaultHostExecutionSpace, 
                                           Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

    Kokkos::View<uint8_t*> data0_d("Run 0 region", 0), data1_d("Run 1 region", 0);
    Kokkos::View<uint8_t*>::HostMirror data0_h = Kokkos::create_mirror_view(data0_d);
    Kokkos::View<uint8_t*>::HostMirror data1_h = Kokkos::create_mirror_view(data1_d);

    Kokkos::View<size_t*> offsets;

    std::vector<double> io_time0(3,0.0), io_time1(3,0.0);
    double compare_time = 0.0;

    // Iterate through files
    for(uint32_t idx=0; idx<num_diffs; idx++) {
      std::cout << "Rank " << world_rank << ": Checkpoint " << idx << std::endl;
      size_t data_len = 0, base_data_len=0;
      // Get length of file
      off_t filesize;
      get_file_size(run0_files[idx], &filesize);
      data_len = static_cast<size_t>(filesize);

      if(alg.compare("direct") == 0) { // Compare data element by element directly
        // ========================================================================================
        // Setup
        // ========================================================================================
        Timer::time_point beg_setup = Timer::now();
        Kokkos::Profiling::pushRegion("Setup");
        if(comparing_runs) {
          if(enable_file_streaming) { // Start loading of data from files
            f32_comparer.setup(data_len, buffer_len/(Kokkos::num_threads()*sizeof(float)), run0_files[idx], run1_files[idx]);
          }
        }
        if(!comparing_runs || !enable_file_streaming) { // Setup comparer
          if(data0_h.size() < data_len) {
            Kokkos::resize(data0_h, data_len);
            Kokkos::resize(data0_d, data_len);
          }
          if(comparing_runs) {
            if(data1_h.size() < data_len) {
              Kokkos::resize(data1_h, data_len);
              Kokkos::resize(data1_d, data_len);
            }
          }
        }

        // Create offsets for loading data
        size_t blocksize = buffer_len;
        blocksize /= Kokkos::num_threads();
        size_t noffsets = data_len/blocksize;
        if(noffsets*blocksize < data_len)
          noffsets += 1;
        if(offsets.size() < noffsets)
          Kokkos::resize(offsets, noffsets);
        Kokkos::parallel_for("Create offsets", Kokkos::RangePolicy<size_t>(0,noffsets), 
        KOKKOS_LAMBDA(const size_t i) {
          offsets(i) = i;
        });
        Kokkos::Profiling::popRegion();
        Timer::time_point end_setup = Timer::now();
        double setup_time = std::chrono::duration_cast<Duration>(end_setup - beg_setup).count();
        std::cout << "\tRank " << world_rank << ": Setup: " << setup_time << std::endl;
        timers[0] = setup_time;

        // ========================================================================================
        // Open file and read/calc important values
        // ========================================================================================
        Timer::time_point beg_read = Timer::now();
        Kokkos::Profiling::pushRegion("Read");
        if(!enable_file_streaming) { // Read files if not using file data streaming
          int fd0 = open(run0_files[idx].c_str(), O_RDONLY, 0644);
          if (fd0 == -1) {
            FATAL("cannot open " << run0_files[idx] << ", error = " << strerror(errno));
          }
          size_t transferred = 0, remaining = data_len;
          while (remaining > 0) {
          	auto ret = read(fd0, data0_h.data() + transferred, remaining);
          	if (ret < 0)
          	  FATAL("cannot read " << data_len << " bytes from " << run0_files[idx] << " , error = " << std::strerror(errno));
          	remaining -= ret;
          	transferred += ret;
          }
          fsync(fd0);
          close(fd0);

          if(comparing_runs) {
            int fd1 = open(run1_files[idx].c_str(), O_RDONLY, 0644);
            if (fd1 == -1) {
              FATAL("cannot open " << run0_files[idx] << ", error = " << strerror(errno));
            }
            transferred = 0, remaining = data_len;
            while (remaining > 0) {
            	auto ret = read(fd1, data1_h.data() + transferred, remaining);
            	if (ret < 0)
            	  FATAL("cannot read " << data_len << " bytes from " << run1_files[idx] << " , error = " << std::strerror(errno));
            	remaining -= ret;
            	transferred += ret;
            }
            fsync(fd1);
            close(fd1);
          }
        } else {
          if(!comparing_runs) {
            int fd0 = open(run0_files[idx].c_str(), O_RDONLY, 0644);
            if (fd0 == -1) {
              FATAL("cannot open " << run0_files[idx] << ", error = " << strerror(errno));
            }
            size_t transferred = 0, remaining = data_len;
            while (remaining > 0) {
            	auto ret = read(fd0, data0_h.data() + transferred, remaining);
            	if (ret < 0)
            	  FATAL("cannot read " << data_len << " bytes from " << run0_files[idx] << " , error = " << std::strerror(errno));
            	remaining -= ret;
            	transferred += ret;
            }
            fsync(fd0);
            close(fd0);
          }
        }

        Kokkos::Profiling::popRegion();
        Timer::time_point end_read = Timer::now();
        double read_time = std::chrono::duration_cast<Duration>(end_read - beg_read).count();
        std::cout << "\tRank " << world_rank << ": Read prior data from file: " << read_time << std::endl;
        timers[1] = read_time;

        // ========================================================================================
        // Deserialize
        // ========================================================================================
        Timer::time_point beg_deserialize = Timer::now();
        Kokkos::Profiling::pushRegion("Deserialization");
        if(comparing_runs) {
          if(enable_file_streaming) { // Start loading of data from files
            f32_comparer.deserialize(run0_files[idx], run1_files[idx]);
          } else { // Copy data to device
            Kokkos::deep_copy(data0_d, data0_h);
            Kokkos::deep_copy(data1_d, data1_h);
          }
        } else {
          if(!enable_file_streaming) { // Copy data to device
            Kokkos::deep_copy(data0_d, data0_h);
          }
        }
        Kokkos::Profiling::popRegion();
        Timer::time_point end_deserialize = Timer::now();
        double deserialize_time = std::chrono::duration_cast<Duration>(end_deserialize - beg_deserialize).count();
        std::cout << "\tRank " << world_rank << ": Deserialize: " << deserialize_time << std::endl;
        timers[2] = deserialize_time;

        // ========================================================================================
        // Compare data
        // ========================================================================================
        Timer::time_point beg_compare = Timer::now();
        Kokkos::Profiling::pushRegion("Compare");
        uint64_t nchanges = 0;

        if(enable_file_streaming) { // Compare data as it is streamed from the files to the device
          nchanges = f32_comparer.compare<AbsoluteComp>(offsets.data(), noffsets);
          io_time0 = f32_comparer.io_timer0;
          io_time1 = f32_comparer.io_timer1;
          compare_time = f32_comparer.get_compare_time();
        } else { // Compare data already on device
          nchanges = f32_comparer.compare<AbsoluteComp>((float*)(data0_d.data()), (float*)(data1_d.data()), data_len/sizeof(float));
        }

        Kokkos::Profiling::popRegion();
        Timer::time_point end_compare = Timer::now();
        double comparison_time = std::chrono::duration_cast<Duration>(end_compare - beg_compare).count();
        std::cout << "\tRank " << world_rank << ": Compare: " << comparison_time << std::endl;
        std::cout << "\t\tRank " << world_rank << ": IO Time (File 0): " << io_time0[0] << std::endl;
        std::cout << "\t\t\tRank " << world_rank << ": Read Time (File 0): " << io_time0[1] << std::endl;
        std::cout << "\t\t\tRank " << world_rank << ": cudaMemcpy Time (File 0): " << io_time0[2] << std::endl;
        std::cout << "\t\tRank " << world_rank << ": IO Time (File 1): " << io_time1[0] << std::endl;
        std::cout << "\t\t\tRank " << world_rank << ": Read Time (File 1): " << io_time1[1] << std::endl;
        std::cout << "\t\t\tRank " << world_rank << ": cudaMemcpy Time (File 1): " << io_time1[2] << std::endl;
        std::cout << "\t\tRank " << world_rank << ": Compare Time: " << compare_time << std::endl;
        timers[4] = comparison_time;

        // ========================================================================================
        // Serialize (does nothing since data is already serialized)
        // ========================================================================================
        timers[5] = 0.0;

        // ========================================================================================
        // Write serialized data
        // ========================================================================================
        timers[6] = 0.0;

        // ========================================================================================
        // Collect stats for log
        // ========================================================================================
        elem_changed = nchanges;
        n_comparisons = f32_comparer.get_num_comparisons();
        changed_blocks = f32_comparer.get_num_changed_blocks();
        printf("Rank %d: Number of different elements %zu\n", world_rank, elem_changed);
        printf("Rank %d: Number of comparisons %lu\n", world_rank, n_comparisons);
        printf("Rank %d: Number of different blocks %zu\n\n", world_rank, changed_blocks);
      } else if(alg.compare("compare-tree") == 0) {
        // ========================================================================================
        //  Setup
        // ========================================================================================
        Timer::time_point beg_setup = Timer::now();
        Kokkos::Profiling::pushRegion("Setup");
        if(comparing_runs) {
          comp_deduplicator.setup(data_len, buffer_len/sizeof(float), run0_full_files[idx], run1_full_files[idx]);
          if(data0_h.size() < data_len) {
            Kokkos::resize(data0_h, data_len);
            Kokkos::resize(data1_h, data_len);
          }
        } else {
          comp_deduplicator.setup(data_len);
          if(data0_h.size() < data_len) {
            Kokkos::resize(data0_h, data_len);
            Kokkos::resize(data0_d, data_len);
          }
        }
        Kokkos::Profiling::popRegion();
        Timer::time_point end_setup = Timer::now();
        double setup_time = std::chrono::duration_cast<Duration>(end_setup - beg_setup).count();
        std::cout << "\tRank " << world_rank << ": Setup: " << setup_time << std::endl;
        timers[0] = setup_time;

        // ========================================================================================
        // Open file and read/calc important values
        // ========================================================================================
        Timer::time_point beg_read = Timer::now();
        Kokkos::Profiling::pushRegion("Read");
        if(comparing_runs) {
          
          int fd0 = open(run0_files[idx].c_str(), O_RDONLY, 0644);
          if (fd0 == -1) {
            FATAL("cannot open " << run0_files[idx] << ", error = " << strerror(errno));
          }
          size_t transferred = 0, remaining = data_len;
          while (remaining > 0) {
          	auto ret = read(fd0, data0_h.data() + transferred, remaining);
          	if (ret < 0)
          	  FATAL("cannot read " << data_len << " bytes from " << run0_files[idx] << " , error = " << std::strerror(errno));
          	remaining -= ret;
          	transferred += ret;
          }
          fsync(fd0);
          close(fd0);

          int fd1 = open(run1_files[idx].c_str(), O_RDONLY, 0644);
          if (fd1 == -1) {
            FATAL("cannot open " << run1_files[idx] << ", error = " << strerror(errno));
          }
          transferred = 0, remaining = data_len;
          while (remaining > 0) {
          	auto ret = read(fd1, data1_h.data() + transferred, remaining);
          	if (ret < 0)
          	  FATAL("cannot read " << data_len << " bytes from " << run1_files[idx] << " , error = " << std::strerror(errno));
          	remaining -= ret;
          	transferred += ret;
          }
          fsync(fd1);
          close(fd1);
        } else {
          int fd0 = open(run0_files[idx].c_str(), O_RDONLY, 0644);
          if (fd0 == -1) {
            FATAL("cannot open " << run0_files[idx] << ", error = " << strerror(errno));
          }
          size_t transferred = 0, remaining = data_len;
          while (remaining > 0) {
          	auto ret = read(fd0, data0_h.data() + transferred, remaining);
          	if (ret < 0)
          	  FATAL("cannot read " << data_len << " bytes from " << run0_files[idx] << " , error = " << std::strerror(errno));
          	remaining -= ret;
          	transferred += ret;
          }
          fsync(fd0);
          close(fd0);
        }
        Kokkos::Profiling::popRegion();
        Timer::time_point end_read = Timer::now();
        double read_time = std::chrono::duration_cast<Duration>(end_read - beg_read).count();
        std::cout << "\tRank " << world_rank << ": Read prior run file: " << read_time << std::endl;
        timers[1] = read_time;

        // ========================================================================================
        // Deserialize
        // ========================================================================================
        Timer::time_point beg_deserialize = Timer::now();
        Kokkos::Profiling::pushRegion("Deserialize");
        if(comparing_runs) {
          comp_deduplicator.deserialize(data0_h.data(), data1_h.data());
        } else {
          Kokkos::deep_copy(data0_d, data0_h);
        }
        Kokkos::Profiling::popRegion();
        Timer::time_point end_deserialize = Timer::now();
        double deserialize_time = std::chrono::duration_cast<Duration>(end_deserialize - beg_deserialize).count();
        std::cout << "\tRank " << world_rank << ": Deserialize: " << deserialize_time << std::endl;
        timers[2] = deserialize_time;

        // ========================================================================================
        // Compare
        // ========================================================================================
        if(!comparing_runs) {
          Timer::time_point beg_compare1 = Timer::now();
          Kokkos::Profiling::pushRegion("Create tree");
          comp_deduplicator.create_tree((uint8_t*)(data0_d.data()), data0_d.size());
          Kokkos::Profiling::popRegion();
          Timer::time_point end_compare1 = Timer::now();
          double compare_time1 = std::chrono::duration_cast<Duration>(end_compare1 - beg_compare1).count();
          std::cout << "\tRank " << world_rank << ": Create Tree: " << compare_time1 << std::endl;
          timers[3] = compare_time1;
        } else {
          Timer::time_point beg_compare1 = Timer::now();
          Kokkos::Profiling::pushRegion("Compare phase 1");
          comp_deduplicator.compare_trees_phase1();
          Kokkos::Profiling::popRegion();
          Timer::time_point end_compare1 = Timer::now();
          double compare_time1 = std::chrono::duration_cast<Duration>(end_compare1 - beg_compare1).count();
          std::cout << "\tRank " << world_rank << ": Compare Tree Phase 1: " << compare_time1 << std::endl;
          timers[3] = compare_time1;

          Timer::time_point beg_compare2 = Timer::now();
          Kokkos::Profiling::pushRegion("Compare phase 2");
          if(comp_deduplicator.diff_hash_vec.size() > 0) {
            comp_deduplicator.compare_trees_phase2();
          }
          Kokkos::Profiling::popRegion();
          Timer::time_point end_compare2 = Timer::now();
          double compare_time2 = std::chrono::duration_cast<Duration>(end_compare2 - beg_compare2).count();
          std::cout << "\tRank " << world_rank << ": Compare Tree Phase 2: " << compare_time2 << std::endl;
          timers[4] = compare_time2;

          std::cout << "\t\tRank " << world_rank << ": IO Time (File 0): " << comp_deduplicator.io_timer0[0] << std::endl;
          std::cout << "\t\t\tRank " << world_rank << ": Read Time (File 0): " << comp_deduplicator.io_timer0[1] << std::endl;
          std::cout << "\t\t\tRank " << world_rank << ": cudaMemcpy Time (File 0): " << comp_deduplicator.io_timer0[2] << std::endl;
          std::cout << "\t\tRank " << world_rank << ": IO Time (File 1): " << comp_deduplicator.io_timer1[0] << std::endl;
          std::cout << "\t\t\tRank " << world_rank << ": Read Time (File 1): " << comp_deduplicator.io_timer1[1] << std::endl;
          std::cout << "\t\t\tRank " << world_rank << ": cudaMemcpy Time (File 1): " << comp_deduplicator.io_timer1[2] << std::endl;
          std::cout << "\t\tRank " << world_rank << ": Compare Time: " << comp_deduplicator.get_compare_time() << std::endl;
        }

        // ========================================================================================
        // Serialize
        // ========================================================================================
        std::vector<uint8_t> serialized_buffer;
        Timer::time_point beg_serialize = Timer::now();
        Kokkos::Profiling::pushRegion("Serialize");
        serialized_buffer = comp_deduplicator.serialize();
        Kokkos::Profiling::popRegion();
        Timer::time_point end_serialize = Timer::now();
        double serialize_time = std::chrono::duration_cast<Duration>(end_serialize - beg_serialize).count();
        std::cout << "\tRank " << world_rank << ": Serialize: " << serialize_time << std::endl;
        timers[5] = serialize_time;

        // ========================================================================================
        // Write
        // ========================================================================================
        Timer::time_point beg_write_tree = Timer::now();
        Kokkos::Profiling::pushRegion("Write");
        std::string outname;
        if(comparing_runs) {
          outname = run1_files[idx] + std::string(".") + std::to_string(idx) + std::string(".compare-tree");
        } else {
          outname = run0_files[idx] + std::string(".") + std::to_string(idx) + std::string(".compare-tree");
        }
        if(output_fname.size() > 0) {
          outname = output_fname;
        }
        if(!comparing_runs) {
          int fd = open(outname.c_str(), O_CREAT | O_TRUNC | O_WRONLY, 0644);
          if (fd == -1) {
              FATAL("cannot open " << outname << ", error = " << strerror(errno));
          }
          size_t transferred = 0, remaining = serialized_buffer.size();
          while (remaining > 0) {
          	auto ret = write(fd, serialized_buffer.data() + transferred, remaining);
          	if (ret < 0)
          	    FATAL("cannot write " << serialized_buffer.size() << " bytes to " << outname << " , error = " << std::strerror(errno));
          	remaining -= ret;
          	transferred += ret;
          }
          fsync(fd);
          posix_fadvise(fd, 0,0,POSIX_FADV_DONTNEED);
          close(fd);
        }
        Kokkos::Profiling::popRegion();
        Timer::time_point end_write_tree = Timer::now();
        double write_tree_time = std::chrono::duration_cast<Duration>(end_write_tree - beg_write_tree).count();
        std::cout << "\tRank " << world_rank << ": Write: " << write_tree_time << std::endl;
        timers[6] = write_tree_time;
        // ========================================================================================
        // Collect stats for logs
        // ========================================================================================
        n_comparisons = comp_deduplicator.get_num_comparisons();
        n_hash_comp = comp_deduplicator.get_num_hash_comparisons();
        elem_changed = comp_deduplicator.get_num_changes();
        filtered_blocks = comp_deduplicator.diff_hash_vec.size();
        changed_blocks = comp_deduplicator.changed_chunks.count();
        printf("Rank %d: Number of different elements %zu\n", world_rank, elem_changed);
        printf("Rank %d: Number of comparisons %lu\n", world_rank, n_comparisons);
        printf("Rank %d: Number of hash comparisons %lu\n", world_rank, n_hash_comp);
        printf("Rank %d: Number of different hashes (Phase 1) %zu\n", world_rank, filtered_blocks);
        printf("Rank %d: Number of different hashes (Phase 2) %zu\n\n", world_rank, changed_blocks);
      }
      Kokkos::fence();
      // Write log
      std::ofstream logfile;
      logfile.open(logname, std::ofstream::out | std::ofstream::app);
      logfile.precision(10);
      if(logfile.tellp() == logfile.beg) {
        logfile << "Rank,File,File size,Baseline file,Baseline file size,Chunk size,";
        logfile << "Error tolerance,";
        logfile << "Setup time,Read time,Deserialization time,Construction time,Compare tree time,Compare direct time,Serialization time,Write time,";
        logfile << "Elements different,Hashes different,Num comparisons,Num hash comparisons,Filtered hashes,Create leaves time\n";
      }
      logfile << world_rank << ",";
      if(comparing_runs) {
        logfile << run1_files[idx] << ",";
      } else {
        logfile << run0_files[idx] << ",";
      }
      logfile << data_len << ",";
      if(comparing_runs) {
        logfile << run0_files[idx] << ",";
        logfile << base_data_len << ",";
      } else {
        logfile << ",,";
      }
      logfile << chunk_size << ",";
      logfile << err_tol << ",";
      logfile << timers[0] << ",";
      logfile << timers[1] << ",";
      logfile << timers[2] << ",";
      if(comparing_runs) {
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
      logfile << filtered_blocks << ",";
      logfile << comp_deduplicator.get_createleaves_time() << std::endl;
      logfile.close();
    }
  }
  Kokkos::finalize();
  DEBUG_PRINT("Done finalizing Kokkos\n");
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
  return 0;
}
