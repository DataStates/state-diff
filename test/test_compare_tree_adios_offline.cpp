#include <vector>
#include <random>
#include <algorithm>
#include <chrono>
#include <string>
#include "stdio.h"
#include <omp.h>
#include <iostream>
#include "state_diff.hpp"
#include <adios2.h>
#include <Kokkos_Core.hpp>

template<typename T>
void adios_writer(adios2::IO &io, std::vector<T> &buffer, std::string tag, std::string fn) {
  io.SetEngine("BP5");
  adios2::Variable<T> adios_tree = 
    io.DefineVariable<T>(tag, {buffer.size()}, {0}, {buffer.size()});
  adios2::Engine writer = io.Open(fn, adios2::Mode::Write);
  writer.Put<T>(adios_tree, buffer.data());
  writer.Close();
}

template<typename T>
std::vector<T> adios_reader(adios2::IO &io, std::string tag, std::string fn) {
  io.SetEngine("BP5");
  adios2::Engine reader = io.Open(fn, adios2::Mode::Read);
  reader.BeginStep();
  adios2::Variable<T> variable = io.InquireVariable<T>(tag);    
  std::vector<T> deserialized_buffer;
  reader.Get<T>(variable, deserialized_buffer);
  reader.EndStep();
  reader.Close();
  return deserialized_buffer;
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
  int chunk_size = 512; // Target chunk size. This example uses 16 bytes
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
    std::vector<float> data_run0(data_len), data_run1(data_len);
    #pragma omp parallel
    {
      std::mt19937 prng(seed + omp_get_thread_num());
      std::uniform_real_distribution<float> prng_dist(min_float, max_float);
      std::uniform_real_distribution<float> outof_error_dist(1.5*error_tolerance, 2*error_tolerance);
      #pragma omp for
      for (int i = 0; i < data_len; ++i) {
        data_run0[i] = prng_dist(prng);
        data_run1[i] = data_run0[i] + outof_error_dist(prng);
      }
    }
    
    // Save checkpoint data for offline comparison
    adios2::ADIOS adios_client;
    adios2::IO io_writer = adios_client.DeclareIO("writer");
    adios2::IO io_reader = adios_client.DeclareIO("reader");
    std::string run0_file = "run0_data.bp", run1_file = "run1_data.bp";
    adios_writer<float>(io_writer, data_run0, "data0", run0_file);
    adios_writer<float>(io_writer, data_run1, "data1", run1_file);

    // Copy data from host to device
    Kokkos::View<uint8_t*> data_run0_d("Run0", data_size), data_run1_d("Run1", data_size);
    using DataHostView = Kokkos::View<uint8_t*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    DataHostView data_run0_h((uint8_t*)data_run0.data(), data_size);
    DataHostView data_run1_h((uint8_t*)data_run1.data(), data_size);
    Kokkos::deep_copy(data_run0_d, data_run0_h);
    Kokkos::deep_copy(data_run1_d, data_run1_h);
    std::cout << "EXEC STATE:: Data loaded and transfered to GPU" << std::endl;
    
    // Create and save trees
    CompareTreeDeduplicator tree_object(chunk_size, root_level, fuzzy_hash, error_tolerance, dtype[0]);
    tree_object.setup(data_size);

    tree_object.create_tree((uint8_t*)data_run0_d.data(), data_run0_d.size());
    std::vector<uint8_t> tree0_data = tree_object.serialize();

    tree_object.create_tree((uint8_t*)data_run1_d.data(), data_run1_d.size());
    std::vector<uint8_t> tree1_data = tree_object.serialize();

    adios_writer<uint8_t>(io_writer, tree0_data, "tree0", "run0_tree.bp");
    adios_writer<uint8_t>(io_writer, tree1_data, "tree1", "run1_tree.bp");
    std::cout << "EXEC STATE:: Trees created and saved to ADIOS2's BP file" << std::endl;

    // Read and deserialize trees for comparison. These two checkpoints should NOT match
    std::vector<uint8_t> deserialized_tree0 = adios_reader<uint8_t>(io_reader, "tree0", "run0_tree.bp");
    std::vector<uint8_t> deserialized_tree1 = adios_reader<uint8_t>(io_reader, "tree1", "run1_tree.bp");
    std::cout << "Successfully deserialized" << std::endl;
    CompareTreeDeduplicator comparator(chunk_size, root_level, fuzzy_hash, error_tolerance, dtype[0]);
    comparator.setup(data_size, stream_buffer_len/sizeof(float), run0_file, run1_file);
    comparator.deserialize(deserialized_tree0, deserialized_tree1); // merkle trees of each run
    comparator.compare_trees_phase1(); // compare trees
    if(comparator.diff_hash_vec.size() > 0) 
      comparator.compare_trees_phase2(); // compare data for mismatched data chunks 
    std::cout << "Number of Different Values = " << comparator.get_num_changes() << std::endl;
    std::cout << "EXEC STATE:: Tree Read from ADIOS2's BP file and compared" << std::endl;

    if(comparator.get_num_changes() != data_len)
      test_status = -1;
  }
  Kokkos::finalize();

  return test_status;
}