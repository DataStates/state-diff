#include <vector>
#include <random>
#include <algorithm>
#include <chrono>
#include <string>
#include "stdio.h"
#include <omp.h>
#include <iostream>
#include "state_diff.hpp"
#include <Kokkos_Core.hpp>

bool write_file(std::string filename, uint8_t *buffer, size_t size) {
  int fd = open(filename.c_str(), O_CREAT | O_TRUNC | O_WRONLY, 0644);
  if (fd == -1) {
    FATAL("cannot open " << filename << ", error = " << strerror(errno));
    return false;
  }
  size_t transferred = 0, remaining = size;
  while (remaining > 0) {
    ssize_t ret = write(fd, buffer + transferred, remaining);
    if (ret == -1)
      FATAL("cannot write " << size << " bytes to " << filename << " , error = " << std::strerror(errno));
    remaining -= ret;
    transferred += ret;
  }
  close(fd);
  return true;
}

int main(int argc, char** argv) 
{
  int test_status = 0;
  
  // Define the parameters
  float max_float = 100.0; // maximum floating-point (FP) value in synthetic data
  float min_float = 0.0; // minimum FP value in synthetic data
  size_t data_size = 1024*1024*1024; // size in bytes of the synthetic data (1GB)
  size_t stream_buffer_len = 512*1024*1024/sizeof(float); // how much data element to read in during the second phase of the comparison.
  float error_tolerance = 1e-4; // Application error tolerance
  int chunk_size = 1024; // Target chunk size. This example uses 16 bytes
  bool fuzzy_hash = true; // Set to true to use our rounding hash algorithm. Otherwise, directly hash blocks of FP values
  std::string dtype = "float"; // float
  int seed = 0x123; // Random number seed to generate the synthetic data
  int root_level = 1; // builds the tree from the leaf level to level 1 (root level). For better parallelism, set root_level to 12 or 13.

  int num_chunks =  data_size / chunk_size;
  std::cout << "Nunber of leaf nodes = " << num_chunks << std::endl;

  Kokkos::initialize(argc, argv);
  {
    // Create synthetic datasets
    Timer::time_point start_dataloading = Timer::now();
    size_t data_len = data_size / sizeof(float);
    std::vector<float> data_run0(data_len), data_run1(data_len), data_run2(data_len);
    #pragma omp parallel
    {
      std::mt19937 prng(seed + omp_get_thread_num());
      std::uniform_real_distribution<float> prng_dist(min_float, max_float);
      std::uniform_real_distribution<float> in_error_dist(0.0, 0.5*error_tolerance);
      std::uniform_real_distribution<float> outof_error_dist(1.5*error_tolerance, 2*error_tolerance);
      #pragma omp for
      for (int i = 0; i < data_len; ++i) {
        data_run0[i] = prng_dist(prng);
        data_run1[i] = data_run0[i] + in_error_dist(prng);
        data_run2[i] = data_run0[i] + outof_error_dist(prng);
      }
    }
    // Save checkpoint data for offline comparison
    std::string run0_file = "run0_data.dat", run1_file = "run1_data.dat", run2_file = "run2_data.dat";
    write_file(run0_file, (uint8_t *)data_run0.data(), data_size);
    write_file(run1_file, (uint8_t *)data_run1.data(), data_size);
    write_file(run2_file, (uint8_t *)data_run2.data(), data_size);

    // Copy data from host to device
    Kokkos::View<uint8_t*> data_run0_d("Run0", data_size), data_run1_d("Run1", data_size), data_run2_d("Run2", data_size);
    using DataHostView = Kokkos::View<uint8_t*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    DataHostView data_run0_h((uint8_t*)data_run0.data(), data_size);
    DataHostView data_run1_h((uint8_t*)data_run1.data(), data_size);
    DataHostView data_run2_h((uint8_t*)data_run2.data(), data_size);
    Kokkos::deep_copy(data_run0_d, data_run0_h);
    Kokkos::deep_copy(data_run1_d, data_run1_h);
    Kokkos::deep_copy(data_run2_d, data_run2_h);
    std::cout << "EXEC STATE:: Data loaded and transfered to GPU" << std::endl;
    
    // Create trees
    CompareTreeDeduplicator tree_object(chunk_size, root_level, fuzzy_hash, error_tolerance, dtype[0]);
    tree_object.setup(data_size);

    tree_object.create_tree((uint8_t*)data_run0_d.data(), data_run0_d.size());
    std::vector<uint8_t> tree0_data = tree_object.serialize();

    tree_object.create_tree((uint8_t*)data_run1_d.data(), data_run1_d.size());
    std::vector<uint8_t> tree1_data = tree_object.serialize();

    tree_object.create_tree((uint8_t*)data_run2_d.data(), data_run2_d.size());
    std::vector<uint8_t> tree2_data = tree_object.serialize();
    std::cout << "EXEC STATE:: Trees created" << std::endl;

    // Compare checkpoints 0 and 1. These two checkpoints should match
    CompareTreeDeduplicator comparator_01(chunk_size, root_level, fuzzy_hash, error_tolerance, dtype[0]);
    comparator_01.setup(data_size, stream_buffer_len/sizeof(float), run0_file, run1_file);
    comparator_01.deserialize(tree0_data, tree1_data); // merkle trees of each run
    comparator_01.compare_trees_phase1(); // compare trees
    if(comparator_01.diff_hash_vec.size() > 0) 
      comparator_01.compare_trees_phase2(); // compare data for mismatched data chunks   
    std::cout << "Number of mismatch (0-1) = " << comparator_01.get_num_changes() << std::endl;
    
    // Compare checkpoints 0 and 2. These two checkpoints should NOT match
    CompareTreeDeduplicator comparator_02(chunk_size, root_level, fuzzy_hash, error_tolerance, dtype[0]);
    comparator_02.setup(data_size, stream_buffer_len/sizeof(float), run0_file, run2_file);
    comparator_02.deserialize(tree0_data, tree2_data); // merkle trees of each run
    comparator_02.compare_trees_phase1(); // compare trees
    if(comparator_02.diff_hash_vec.size() > 0) 
      comparator_02.compare_trees_phase2(); // compare data for mismatched data chunks
    std::cout << "Number of mismatch (0-2) = " << comparator_02.get_num_changes() << std::endl;

    if(comparator_01.get_num_changes() != 0 || comparator_02.get_num_changes() != data_len)
      test_status = -1;
  }
  Kokkos::finalize();

  return test_status;
}
