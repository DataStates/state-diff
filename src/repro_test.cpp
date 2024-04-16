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
//#include "deduplicator.hpp"
//#include "compare_tree_approach.hpp"
//#include "direct_comparer.hpp"
//#include "comp_func.hpp"
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
    program.add_argument("--fuzzy-hash")
      .help("Decide whether to use fuzzy hash")
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
    program.add_argument("-t", "--type")
      .required()
      .help("Data type")
      .default_value(std::string("byte"))
      .choices("byte", "float", "double");
    program.add_argument("--comp")
      .required()
      .help("Comparison Algorithm for Direct Comparison")
      .default_value(std::string("equivalence"))
      .choices("equivalence", "absolute", "relative");
    program.add_argument("-l", "--level")
      .help("Level to start/stop processing the tree. Root is level 0.")
      .default_value(static_cast<uint32_t>(13))
      .scan<'u', uint32_t>();
    program.add_argument("--sync-data-stream")
      .help("Use synchronous data copies for streaming data. Asynchronous by default.")
      .default_value(false)
      .implicit_value(true);
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
      .default_value(std::string("result_log.csv"));
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
    bool fuzzy_hash = false;
    if(program["--fuzzy-hash"] == true) 
      fuzzy_hash = true;
    bool async_stream = !(program["--sync-data-stream"]==true);
    bool enable_file_streaming = program["--enable-file-streaming"] == true;
    size_t buffer_len = program.get<size_t>("--buffer-len");
    std::string alg     = program.get<std::string>("--alg");
    std::string dtype   = program.get<std::string>("--type");
    std::string comp    = program.get<std::string>("--comp");
    std::string logname = program.get<std::string>("--result-logname");
//    logname = "Rank."logname;
    auto run0_all_files = program.get<std::vector<std::string>>("--run0");
    auto run1_all_files = program.get<std::vector<std::string>>("--run1");
    auto run0_all_full_files = program.get<std::vector<std::string>>("--run0-full");
    auto run1_all_full_files = program.get<std::vector<std::string>>("--run1-full");

    double err_tol = program.get<double>("--error");
    std::string output_fname = program.get<std::string>("--output-filename");
    STDOUT_PRINT("Chunk size: %u\n", chunk_size);
    STDOUT_PRINT("Start level %u\n", level);
    STDOUT_PRINT("Fuzzy hash? %d\n", fuzzy_hash);
    STDOUT_PRINT("Algorithm:  %s\n", alg.c_str());
    STDOUT_PRINT("Data type:  %s\n", dtype.c_str());
    STDOUT_PRINT("Numb diffs: %u\n", num_diffs);

    int world_rank=0, world_size=1;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    std::string rank_str = "mpirestart-" + std::to_string(world_rank);
    std::vector<std::string> run0_files, run1_files, run0_full_files, run1_full_files;
    if(run0_all_files.size() > 0) {
      for(uint32_t i=0; i<run0_all_files.size(); i++) {
        if(run0_all_files[i].find(rank_str) != std::string::npos) {
          run0_files.push_back(run0_all_files[i]);
        }
      }
      for(uint32_t i=0; i<run0_all_full_files.size(); i++) {
        if(run0_all_full_files[i].find(rank_str) != std::string::npos) {
          run0_full_files.push_back(run0_all_full_files[i]);
        }
      }
    }
    if(run1_all_files.size() > 0) {
      for(uint32_t i=0; i<run1_all_files.size(); i++) {
        if(run1_all_files[i].find(rank_str) != std::string::npos) {
          run1_files.push_back(run1_all_files[i]);
        }
      }
      for(uint32_t i=0; i<run1_all_full_files.size(); i++) {
        if(run1_all_full_files[i].find(rank_str) != std::string::npos) {
          run1_full_files.push_back(run1_all_full_files[i]);
        }
      }
    }
    uint32_t num_diffs = run0_files.size();
    bool comparing_runs = run1_files.size() == num_diffs;
    for(uint32_t i=0; i<run0_files.size(); i++) {
      STDOUT_PRINT("File %d: %s\n", i, run0_files[i].c_str());
    }

    double timers[6] = {0.0};
    size_t elem_changed = 0;
    uint64_t changed_blocks = 0;
    uint64_t n_comparisons = 0;
    uint64_t n_hash_comp = 0;
    size_t data_type_size = 1;
    if(dtype.compare("float") == 0) {
      data_type_size = sizeof(float);
    } else if(dtype.compare("double") == 0) {
      data_type_size = sizeof(double);
    }

    // Create deduplicators
    CompareTreeDeduplicator comp_deduplicator = CompareTreeDeduplicator(chunk_size, level, fuzzy_hash, err_tol, dtype[0]);
    comp_deduplicator.num_threads = Kokkos::num_threads();
    if (comp.compare("absolute") ==  0) { 
      comp_deduplicator.comp_op = Absolute;
    } else if(comp.compare("relative") == 0) {
      comp_deduplicator.comp_op = Relative;
    } else if(comp.compare("equivalence") == 0) {
      comp_deduplicator.comp_op = Equivalence;
    }
    DirectComparer<uint8_t,Kokkos::DefaultExecutionSpace> u8_comparer( err_tol, buffer_len, Kokkos::num_threads());  
    DirectComparer<float,Kokkos::DefaultExecutionSpace>   f32_comparer(err_tol, buffer_len/sizeof(float), Kokkos::num_threads()); 
    DirectComparer<double,Kokkos::DefaultExecutionSpace>  f64_comparer(err_tol, buffer_len/sizeof(double), Kokkos::num_threads()); 

    // Iterate through files
    for(uint32_t idx=0; idx<num_diffs; idx++) {
      std::cout << "Rank " << world_rank << ": Checkpoint " << idx << std::endl;
      size_t data_len = 0, base_data_len=0;
      if(alg.compare("direct") == 0) { // Compare data element by element directly
        // ========================================================================================
        // Open file and read/calc important values
        // ========================================================================================
        Timer::time_point beg_read = Timer::now();
        Kokkos::Profiling::pushRegion("Read");

        // Get length of file
        std::ifstream f;
        f.exceptions(std::ifstream::failbit | std::ifstream::badbit);
        f.open(run0_files[idx], std::ifstream::in | std::ifstream::binary);
        f.seekg(0, f.end);
        data_len = f.tellg();
        f.seekg(0, f.beg);

        Kokkos::View<uint8_t*>::HostMirror data0_h, data1_h;
        if(!enable_file_streaming) { // Read files if not using file data streaming
          data0_h = Kokkos::View<uint8_t*>::HostMirror("Run 0 region mirror", data_len);
          f.read((char*)(data0_h.data()), data_len);
          f.close();
          if(comparing_runs) {
            data1_h = Kokkos::View<uint8_t*>::HostMirror("Run 1 region mirror", data_len);
            f.open(run1_files[idx], std::ifstream::in | std::ifstream::binary);
            f.read((char*)(data1_h.data()), data_len);
            f.close();
          }
        } else {
          if(comparing_runs) { // Read current file for later write
//            data1_h = Kokkos::View<uint8_t*>::HostMirror("Run 0 region mirror", data_len);
//            f.read((char*)(data1_h.data()), data_len);
            f.close();
          } else {
            data0_h = Kokkos::View<uint8_t*>::HostMirror("Run 0 region mirror", data_len);
            f.read((char*)(data0_h.data()), data_len);
            f.close();
          }
        }

        Kokkos::Profiling::popRegion();
        Timer::time_point end_read = Timer::now();
        double read_time = std::chrono::duration_cast<Duration>(end_read - beg_read).count();
        std::cout << "\tRank " << world_rank << ": Read prior data from file: " << read_time << std::endl;
        timers[0] = read_time;

        // ========================================================================================
        // Setup
        // ========================================================================================
        Timer::time_point beg_setup = Timer::now();
        Kokkos::Profiling::pushRegion("Setup");
        Kokkos::View<uint8_t*> data0_d, data1_d;
        if(enable_file_streaming) { // Setup comparer
          if(dtype.compare("float") == 0) {
            f32_comparer.setup(data_len/sizeof(float), !comparing_runs);
          } else if(dtype.compare("double") == 0) {
            f64_comparer.setup(data_len/sizeof(double), !comparing_runs);
          } else {
            u8_comparer.setup(data_len, !comparing_runs);
          }
        } else { // Allocate Views if not using file streaming
          data0_d = Kokkos::View<uint8_t*>("Run 0 region mirror", data_len);
          if(comparing_runs)
            data1_d = Kokkos::View<uint8_t*>("Run 1 region mirror", data_len);
        }
        Kokkos::Profiling::popRegion();
        Timer::time_point end_setup = Timer::now();
        double setup_time = std::chrono::duration_cast<Duration>(end_setup - beg_setup).count();
        std::cout << "\tRank " << world_rank << ": Setup: " << setup_time << std::endl;
        timers[1] = setup_time;

        // Create offsets for loading data
        size_t blocksize = chunk_size/data_type_size;
//        size_t blocksize = buffer_len/(data_type_size);
        size_t noffsets = (data_len/data_type_size)/blocksize;
        if(noffsets*blocksize < data_len/data_type_size)
          noffsets += 1;
        Kokkos::View<size_t*> offsets("File offsets", noffsets);
        Kokkos::parallel_for("Create offsets", Kokkos::RangePolicy<size_t>(0,noffsets), 
        KOKKOS_LAMBDA(const size_t i) {
          offsets(i) = i;
        });

        // ========================================================================================
        // Deserialize
        // ========================================================================================
        Timer::time_point beg_deserialize = Timer::now();
        Kokkos::Profiling::pushRegion("Deserialization");
        if(comparing_runs) {
          if(enable_file_streaming) { // Start loading of data from files
            if(dtype.compare("float") == 0) {
              f32_comparer.deserialize(offsets.data(), noffsets, blocksize, run0_files[idx], run1_files[idx]);
            } else if(dtype.compare("double") == 0) {
              f64_comparer.deserialize(offsets.data(), noffsets, blocksize, run0_files[idx], run1_files[idx]);
            } else {
              u8_comparer.deserialize(offsets.data(), noffsets, blocksize, run0_files[idx], run1_files[idx]);
            }
          } else { // Copy data to device
            Kokkos::deep_copy(data0_d, data0_h);
            Kokkos::deep_copy(data1_d, data1_h);
          }
        } else {
          f32_comparer.block_size = blocksize;
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

        if(dtype.compare("float") == 0) {
          if(comparing_runs) {
            if(enable_file_streaming) { // Compare data as it is streamed from the files to the device

              if (comp.compare("absolute") ==  0) { 
                nchanges = f32_comparer.compare<AbsoluteComp>();
              } else if (comp.compare("relative") == 0) {
                nchanges = f32_comparer.compare<RelativeComp>();
              } else {
                nchanges = f32_comparer.compare<EquivalenceComp>();
              }
            } else { // Compare data already on device
              if (comp.compare("absolute") ==  0) { 
                nchanges = f32_comparer.compare<AbsoluteComp>((float*)(data0_d.data()), (float*)(data1_d.data()), data_len/sizeof(float));
              } else if (comp.compare("relative") == 0) {
                nchanges = f32_comparer.compare<RelativeComp>((float*)(data0_d.data()), (float*)(data1_d.data()), data_len/sizeof(float));
              } else {
                nchanges = f32_comparer.compare<EquivalenceComp>((float*)(data0_d.data()), (float*)(data1_d.data()), data_len/sizeof(float));
              }
            }
          } else { // Compare initial case. Only for recording times to contrast with I/O and data movement
            float* f32_data = (float*)(data0_d.data());
            size_t f32_data_len = data_len/sizeof(float);
            if (comp.compare("absolute") ==  0) { 
              nchanges = f32_comparer.compare<AbsoluteComp>(f32_data, f32_data_len, !comparing_runs);
            } else if (comp.compare("relative") == 0) {
              nchanges = f32_comparer.compare<RelativeComp>(f32_data, f32_data_len, !comparing_runs);
            } else {
              nchanges = f32_comparer.compare<EquivalenceComp>(f32_data, f32_data_len, !comparing_runs);
            }
          }
        } else if(dtype.compare("double") == 0) {
          if(comparing_runs) {
            if(enable_file_streaming) {
              if (comp.compare("absolute") ==  0) { 
                nchanges = f64_comparer.compare<AbsoluteComp>();
              } else if (comp.compare("relative") == 0) {
                nchanges = f64_comparer.compare<RelativeComp>();
              } else {
                nchanges = f64_comparer.compare<EquivalenceComp>();
              }
            } else {
              if (comp.compare("absolute") ==  0) { 
                nchanges = f64_comparer.compare<AbsoluteComp>((double*)(data0_d.data()), (double*)(data1_d.data()), data_len/sizeof(double));
              } else if (comp.compare("relative") == 0) {
                nchanges = f64_comparer.compare<RelativeComp>((double*)(data0_d.data()), (double*)(data1_d.data()), data_len/sizeof(double));
              } else {
                nchanges = f64_comparer.compare<EquivalenceComp>((double*)(data0_d.data()), (double*)(data1_d.data()), data_len/sizeof(double));
              }
            }
          } else {
            double* f64_data = (double*)(data0_d.data());
            size_t f64_data_len = data_len/sizeof(double);
            if (comp.compare("absolute") ==  0) { 
              nchanges = f64_comparer.compare<AbsoluteComp>(f64_data, f64_data_len, !comparing_runs);
            } else if (comp.compare("relative") == 0) {
              nchanges = f64_comparer.compare<RelativeComp>(f64_data, f64_data_len, !comparing_runs);
            } else {
              nchanges = f64_comparer.compare<EquivalenceComp>(f64_data, f64_data_len, !comparing_runs);
            }
          }
        } else {
          if(comparing_runs) {
            if(enable_file_streaming) {
              if (comp.compare("absolute") ==  0) { 
                nchanges = u8_comparer.compare<AbsoluteComp>();
              } else if (comp.compare("relative") == 0) {
                nchanges = u8_comparer.compare<RelativeComp>();
              } else {
                nchanges = u8_comparer.compare<EquivalenceComp>();
              }
            } else {
              if (comp.compare("absolute") ==  0) { 
                nchanges = u8_comparer.compare<AbsoluteComp>((uint8_t*)(data0_d.data()), (uint8_t*)(data1_d.data()), data_len/sizeof(uint8_t));
              } else if (comp.compare("relative") == 0) {
                nchanges = u8_comparer.compare<RelativeComp>((uint8_t*)(data0_d.data()), (uint8_t*)(data1_d.data()), data_len/sizeof(uint8_t));
              } else {
                nchanges = u8_comparer.compare<EquivalenceComp>((uint8_t*)(data0_d.data()), (uint8_t*)(data1_d.data()), data_len/sizeof(uint8_t));
              }
            }
          } else {
            if (comp.compare("absolute") ==  0) { 
              nchanges = u8_comparer.compare<AbsoluteComp>((uint8_t *)(data0_d.data()), data_len, !comparing_runs);
            } else if (comp.compare("relative") == 0) {
              nchanges = u8_comparer.compare<RelativeComp>((uint8_t *)(data0_d.data()), data_len, !comparing_runs);
            } else {
              nchanges = u8_comparer.compare<EquivalenceComp>((uint8_t *)(data0_d.data()), data_len, !comparing_runs);
            }
          }
        }

        Kokkos::Profiling::popRegion();
        Timer::time_point end_compare = Timer::now();
        double compare_time = std::chrono::duration_cast<Duration>(end_compare - beg_compare).count();
        std::cout << "\tRank " << world_rank << ": Compare: " << compare_time << std::endl;
        timers[3] = compare_time;

        // ========================================================================================
        // Serialize (does nothing since data is already serialized)
        // ========================================================================================
        Timer::time_point beg_serialize = Timer::now();
        Kokkos::Profiling::pushRegion("Serialize");
        Kokkos::Profiling::popRegion();
        Timer::time_point end_serialize = Timer::now();
        double serialize_time = std::chrono::duration_cast<Duration>(end_serialize - beg_serialize).count();
        std::cout << "\tRank " << world_rank << ": Serialize: " << serialize_time << std::endl;
        timers[4] = serialize_time;

        // ========================================================================================
        // Write serialized data
        // ========================================================================================
        Timer::time_point beg_write_tree = Timer::now();
        Kokkos::Profiling::pushRegion("Write");
        std::string outname;
        if(comparing_runs) {
          outname = run1_files[idx] + std::string(".") + std::to_string(idx) + std::string(".direct");
        } else {
          outname = run0_files[idx] + std::string(".") + std::to_string(idx) + std::string(".direct");
        }
        if(output_fname.size() > 0) {
          outname = output_fname;
        }
        if(!comparing_runs) {
//          int fd = open(outname.c_str(), O_CREAT | O_TRUNC | O_WRONLY | O_DIRECT, 0644);
          int fd = open(outname.c_str(), O_CREAT | O_TRUNC | O_WRONLY, 0644);
          if (fd == -1) {
              FATAL("cannot open " << outname << ", error = " << strerror(errno));
          }
          size_t transferred = 0, remaining = data_len;
          while (remaining > 0) {
          	auto ret = write(fd, data0_h.data() + transferred, remaining);
          	if (ret < 0)
          	    FATAL("cannot write " << data_len << " bytes to " << outname << " , error = " << std::strerror(errno));
          	remaining -= ret;
          	transferred += ret;
          }
          close(fd);
        }
        
//        std::ofstream log;
//        if(!comparing_runs) {
//          log.exceptions(std::ofstream::failbit | std::ofstream::badbit);
//          log.open(outname, std::ofstream::out | std::ofstream::binary);
//          log.write((char*)(data0_h.data()), data_len);
//          log.close();
//        }
        Kokkos::Profiling::popRegion();
        Timer::time_point end_write_tree = Timer::now();
        double write_tree_time = std::chrono::duration_cast<Duration>(end_write_tree - beg_write_tree).count();
        std::cout << "\tRank " << world_rank << ": Write: " << write_tree_time << std::endl;
        timers[5] = deserialize_time;

        // ========================================================================================
        // Collect stats for log
        // ========================================================================================
        elem_changed = nchanges;
        if(dtype.compare("float") == 0) {
          n_comparisons = f32_comparer.get_num_comparisons();
          changed_blocks = f32_comparer.get_num_changed_blocks();
        } else if(dtype.compare("double") == 0) {
          n_comparisons = f64_comparer.get_num_comparisons();
          changed_blocks = f64_comparer.get_num_changed_blocks();
        } else if(dtype.compare("byte") == 0) {
          n_comparisons = u8_comparer.get_num_comparisons();
          changed_blocks = u8_comparer.get_num_changed_blocks();
        }
        printf("Rank %d: Number of different elements %zu\n", world_rank, elem_changed);
        printf("Rank %d: Number of comparisons %lu\n", world_rank, n_comparisons);
        printf("Rank %d: Number of different blocks %zu\n\n", world_rank, changed_blocks);
      } else if(alg.compare("compare-tree") == 0) {
        // ========================================================================================
        // Open file and read/calc important values
        // ========================================================================================
        Timer::time_point beg_read = Timer::now();
        Kokkos::Profiling::pushRegion("Read");

        std::ifstream f;
        if(comparing_runs) {
          f.exceptions(std::ifstream::failbit | std::ifstream::badbit);
          f.open(run1_files[idx], std::ifstream::in | std::ifstream::binary);
        } else {
          f.exceptions(std::ifstream::failbit | std::ifstream::badbit);
          f.open(run0_files[idx], std::ifstream::in | std::ifstream::binary);
        }
        f.seekg(0, f.end);
        data_len = f.tellg();
        f.seekg(0, f.beg);

        // Read diff file and load it into the device
        std::vector<uint8_t> run0_buffer, run1_buffer;
        Kokkos::View<uint8_t*> run_view_d;
        Kokkos::View<uint8_t*>::HostMirror run_view_h;
        if(comparing_runs) {
          run1_buffer = std::vector<uint8_t>(data_len, 0);
          run0_buffer = std::vector<uint8_t>(data_len, 0);
          f.read((char*)(run1_buffer.data()), data_len);
          f.close();
          f.open(run0_files[idx], std::ifstream::in | std::ifstream::binary);
          f.read((char*)(run0_buffer.data()), data_len);
          f.close();
        } else {
          run_view_h = Kokkos::View<uint8_t*>::HostMirror("Host data", data_len);
          f.read((char*)(run_view_h.data()), data_len);
          f.close();
        }
        Kokkos::Profiling::popRegion();
        Timer::time_point end_read = Timer::now();
        double read_time = std::chrono::duration_cast<Duration>(end_read - beg_read).count();
        std::cout << "\tRank " << world_rank << ": Read prior run file: " << read_time << std::endl;
        timers[0] = read_time;

        // ========================================================================================
        //  Setup
        // ========================================================================================
        Timer::time_point beg_setup = Timer::now();
        Kokkos::Profiling::pushRegion("Setup");
        if(comparing_runs) {
          comp_deduplicator.setup(data_len, !comparing_runs, run0_full_files[idx], run1_full_files[idx]);
          comp_deduplicator.stream_buffer_len = buffer_len/sizeof(float);
        } else {
          comp_deduplicator.setup(data_len, !comparing_runs);
        }
        if(!comparing_runs) {
          run_view_d = Kokkos::View<uint8_t*>("Device data", data_len);
        }
        Kokkos::Profiling::popRegion();
        Timer::time_point end_setup = Timer::now();
        double setup_time = std::chrono::duration_cast<Duration>(end_setup - beg_setup).count();
        std::cout << "\tRank " << world_rank << ": Setup: " << setup_time << std::endl;
        timers[1] = setup_time;

        // ========================================================================================
        // Deserialize
        // ========================================================================================
        Timer::time_point beg_deserialize = Timer::now();
        Kokkos::Profiling::pushRegion("Deserialize");
        if(comparing_runs) {
          comp_deduplicator.deserialize(run0_buffer, run1_buffer);
        } else {
          Kokkos::deep_copy(run_view_d, run_view_h);
//          comp_deduplicator.deserialize(run0_buffer);
        }
        Kokkos::Profiling::popRegion();
        Timer::time_point end_deserialize = Timer::now();
        double deserialize_time = std::chrono::duration_cast<Duration>(end_deserialize - beg_deserialize).count();
        std::cout << "\tRank " << world_rank << ": Deserialize: " << deserialize_time << std::endl;
        timers[2] = deserialize_time;

        // ========================================================================================
        // Compare
        // ========================================================================================
        Timer::time_point beg_compare = Timer::now();
        Kokkos::Profiling::pushRegion("Compare");
        if(comparing_runs) {
          auto ncomp = comp_deduplicator.compare_trees();
          //printf("Ncomp: %zu\n", comp_deduplicator.num_first_ocur());
        } else {
          comp_deduplicator.create_tree((uint8_t*)(run_view_d.data()), run_view_d.size());
        }
        Kokkos::Profiling::popRegion();
        Timer::time_point end_compare = Timer::now();
        double compare_time = std::chrono::duration_cast<Duration>(end_compare - beg_compare).count();
        std::cout << "\tRank " << world_rank << ": Compare: " << compare_time << std::endl;
        timers[3] = compare_time;

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
        timers[4] = serialize_time;

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
//          int fd = open(outname.c_str(), O_CREAT | O_TRUNC | O_WRONLY | O_DIRECT, 0644);
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
          close(fd);
        }
//        if(!comparing_runs) {
//          std::ofstream log;
//          log.exceptions(std::ofstream::failbit | std::ofstream::badbit);
//          log.open(outname, std::ofstream::out | std::ofstream::binary);
//          log.write((char*)(serialized_buffer.data()), serialized_buffer.size());
//          log.close();
//        }
        Kokkos::Profiling::popRegion();
        Timer::time_point end_write_tree = Timer::now();
        double write_tree_time = std::chrono::duration_cast<Duration>(end_write_tree - beg_write_tree).count();
        std::cout << "\tRank " << world_rank << ": Write: " << write_tree_time << std::endl;
        timers[5] = write_tree_time;
        // ========================================================================================
        // Collect stats for logs
        // ========================================================================================
        size_t changed_blocks_phase1;
        if(fuzzy_hash && (comp_deduplicator.comp_op != Equivalence)) {
          n_comparisons = comp_deduplicator.get_num_comparisons();
          n_hash_comp = comp_deduplicator.get_num_hash_comparisons();
          elem_changed = comp_deduplicator.get_num_changes();
          changed_blocks = comp_deduplicator.changed_chunks.count();
          changed_blocks_phase1 = comp_deduplicator.num_change_hashes_phase_1;
        } else {
          n_comparisons = 0;
          n_hash_comp = comp_deduplicator.get_num_hash_comparisons();
          elem_changed = comp_deduplicator.num_first_ocur()*chunk_size/sizeof(float);
          changed_blocks_phase1 = comp_deduplicator.num_first_ocur();
          changed_blocks = changed_blocks_phase1;
        }
        printf("Rank %d: Number of different elements %zu\n", world_rank, elem_changed);
        printf("Rank %d: Number of comparisons %lu\n", world_rank, n_comparisons);
        printf("Rank %d: Number of hash comparisons %lu\n", world_rank, n_hash_comp);
        printf("Rank %d: Number of different hashes (Phase 1) %zu\n", world_rank, changed_blocks_phase1);
        printf("Rank %d: Number of different hashes (Phase 2) %zu\n\n", world_rank, changed_blocks);
      }
      Kokkos::fence();
      // Write log
      std::ofstream logfile;
      logfile.open(logname, std::ofstream::out | std::ofstream::app);
      logfile.precision(10);
      if(logfile.tellp() == logfile.beg) {
        logfile << "Rank,File,File size,Baseline file,Baseline file size,Hash function,Chunk size,Algorithm,Data type,";
        logfile << "Comparison operator,Error tolerance,Start level,Synchronous,Device buffer length,";
        logfile << "Read time,Setup time,Deserialization time,Construction time,Comparison time,Serialization time,Write time,";
        logfile << "Elements different,Hashes different,Num comparisons,Num hash comparisons\n";
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
      if(fuzzy_hash) {
        logfile << "Fuzzy hash,";
      } else {
        logfile << "Murmur3,";
      }
      logfile << chunk_size << ",";
      logfile << alg << ",";
      logfile << dtype << ",";
      logfile << comp << ",";
      logfile << err_tol << ",";
      logfile << level << ",";
      if(async_stream) {
        logfile << "async,";
      } else {
        logfile << "sync,";
      }
      logfile << buffer_len << ",";
      logfile << timers[0] << ",";
      logfile << timers[1] << ",";
      logfile << timers[2] << ",";
      if(comparing_runs) {
        logfile << "0," << timers[3] << ",";
      } else {
        logfile << timers[3] << ",0,";
      }
      logfile << timers[4] << ",";
      logfile << timers[5] << ",";
      logfile << elem_changed << ",";
      logfile << changed_blocks << ",";
      logfile << n_comparisons << ",";
      logfile << n_hash_comp << std::endl;
      logfile.close();
    }
  }
  Kokkos::finalize();
  DEBUG_PRINT("Done finalizing Kokkos\n");
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
  return 0;
}
