#define __DEBUG
#include "debug.hpp"

#include <vector>
#include <random>
#include <cstring>
#include <Kokkos_Core.hpp>
#include <Kokkos_Bitset.hpp>
#include "kokkos_vector.hpp"
#include "fuzzy_hash.hpp"

void gatherMismatchChunks_cpu(const float* data_a, const float* data_b, std::vector<size_t> val_hash_vec, size_t num_first_occ,
                          int chunk_size, std::vector<float>& region_a_data, std::vector<float>& region_b_data) {
  region_a_data.clear();
  region_b_data.clear();

  for (size_t i = 0; i < num_first_occ; i++) {
    size_t from = val_hash_vec[i]*chunk_size / sizeof(float);
    size_t to = from + chunk_size / sizeof(float);
    std::vector<float> buffer_a(data_a + from, data_a + to);
    std::vector<float> buffer_b(data_b + from, data_b + to);

    std::copy(buffer_a.begin(), buffer_a.end(), std::back_inserter(region_a_data));
    std::copy(buffer_b.begin(), buffer_b.end(), std::back_inserter(region_b_data));
  }
}

void gatherMismatchChunks_gpu(const float* data_a, const float* data_b, 
                                Vector<size_t> val_hash_vec,
                                size_t num_first_occ, int chunk_size, 
                                Kokkos::View<uint8_t*> region_a_d, Kokkos::View<uint8_t*> region_b_d) {
  // Load data from the checkpoint into the corresponsing region
  std::vector<float> region_a_data;
  std::vector<float> region_b_data;

  for (size_t i = 0; i < num_first_occ; i++)  {
    // move the pointer to the corresponding offset to read data
    size_t from = val_hash_vec[i]*chunk_size / sizeof(float);
    size_t to = from + chunk_size/sizeof(float);
    std::vector<float> buffer_a(data_a + from, data_a + to);
    std::vector<float> buffer_b(data_b + from, data_b + to);

    std::copy(buffer_a.begin(), buffer_a.end(), std::back_inserter(region_a_data));
    std::copy(buffer_b.begin(), buffer_b.end(), std::back_inserter(region_b_data));
  }

  uint8_t* region_a_ptr = reinterpret_cast<uint8_t*>(region_a_data.data());
  uint8_t* region_b_ptr = reinterpret_cast<uint8_t*>(region_b_data.data());
  size_t num_bytes = region_a_data.size() * sizeof(float);

  // For kokkos, we can now create an unmanaged view of the data
  Kokkos::View<uint8_t*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> > region_a_h(region_a_ptr, num_bytes);
  Kokkos::View<uint8_t*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> > region_b_h(region_b_ptr, num_bytes);
  Kokkos::deep_copy(region_a_d, region_a_h);
  Kokkos::deep_copy(region_b_d, region_b_h);

  region_a_data.clear();
  region_b_data.clear();
}

void hybridHash_cpu(size_t num_chunks, int chunk_size, float* data_a, float* data_b, float error,
                  size_t& num_mismatch_one, size_t& num_mismatch_two) {

  // STAGE 1: Hash Generation
  size_t num_mismatch = 0;
  std::vector<size_t> mismatch_vec;
  for (size_t idx = 0; idx < num_chunks; idx++)    {
    uint64_t num_bytes = chunk_size;
    int num_element = num_bytes / sizeof(float);
    size_t offset = idx*num_element;
    float* cur_data_a = data_a + offset;
    float* cur_data_b = data_b + offset;
    HashDigest digests_a[2] = {HashDigest(), HashDigest()};
    HashDigest digests_b[2] = {HashDigest(), HashDigest()};
    fuzzyhash(cur_data_a, num_bytes, 'f', error, digests_a, false);
    fuzzyhash(cur_data_b, num_bytes, 'f', error, digests_b, false);
    bool same = digests_same(digests_a[0], digests_b[0]) || 
                digests_same(digests_a[0], digests_b[1]) || 
                digests_same(digests_a[1], digests_b[0]) || 
                digests_same(digests_a[1], digests_b[1]);
    if (!same) {
      num_mismatch += 1;
      mismatch_vec.push_back(idx);
    } else {
      // Sanity check in case we missed any chunk that should have been a mismatch
      int dsum = 0;
      for (int i = 0; i < num_element && dsum < 1; ++i) {
        if (Kokkos::fabs(cur_data_a[i] - cur_data_b[i]) > error) {
          dsum = 1;
        }
      }
      if(dsum == 1) {
        for (int i = 0; i < num_element; ++i) {
          printf("(%zu) Mismatch -> %.6f vs %.6f\n", idx, cur_data_a[i], cur_data_b[i]);
        }
      }
    }
  }

  // STAGE 2: HASH MISMATCH VALIDATION THROUGH DIRECT COMPARISON
  size_t num_mismatch_final = 0;
  if (num_mismatch > 0) {
    std::vector<float> val_data_a, val_data_b;
    gatherMismatchChunks_cpu(data_a, data_b, mismatch_vec, num_mismatch, chunk_size, val_data_a, val_data_b);
    for(size_t idx = 0; idx < num_mismatch; idx++) {
      int num_element = chunk_size / sizeof(float);
      size_t offset = idx * num_element;
      float* cur_data_a = val_data_a.data()+offset;
      float* cur_data_b = val_data_b.data()+offset;
      int dsum = 0;
      for (int i = 0; i < num_element && dsum < 1; ++i) {
        if (Kokkos::fabs(cur_data_a[i] - cur_data_b[i]) > error) {
          dsum = 1;
        }
      }
      if(dsum > 0) { num_mismatch_final += 1; }
    }
  }
  num_mismatch_one = num_mismatch;
  num_mismatch_two = num_mismatch_final;
}

void hybridHash_gpu(size_t num_chunks, int chunk_size, 
                  Kokkos::View<float*> data_a, Kokkos::View<float*> data_b, 
                  float error, size_t& num_mismatch_one, size_t& num_mismatch_two) {
  Vector<size_t> val_hash_vec(num_chunks);
  
  // STAGE 1: Hash Generation
  Kokkos::parallel_reduce("Hybrid::Hash", Kokkos::RangePolicy<>(0, num_chunks), 
  KOKKOS_LAMBDA (size_t idx, size_t& lsum) {
    uint64_t num_bytes = chunk_size;
    int num_element = num_bytes / sizeof(float);
    size_t offset = (int)idx*num_element;
    float* cur_data_a = data_a.data() + offset;
    float* cur_data_b = data_b.data() + offset;
    HashDigest digests_a[2] = {HashDigest(), HashDigest()};
    HashDigest digests_b[2] = {HashDigest(), HashDigest()};
    fuzzyhash(cur_data_a, num_bytes, 'f', error, digests_a, true);
    fuzzyhash(cur_data_b, num_bytes, 'f', error, digests_b, true);
    bool same = digests_same(digests_a[0], digests_b[0]) || 
                digests_same(digests_a[0], digests_b[1]) || 
                digests_same(digests_a[1], digests_b[0]) || 
                digests_same(digests_a[1], digests_b[1]);
    if (!same) {
      lsum += 1;
      val_hash_vec.push(idx);
    } else {
      // Sanity check in case we missed any chunk that should have been a mismatch
      int dsum = 0;
      for (int i = 0; i < num_element && dsum < 1; ++i) {
        if (Kokkos::fabs(cur_data_a[i] - cur_data_b[i]) > error) {
          dsum = 1;
        }
      }
      if(dsum == 1) {
        for (int i = 0; i < num_element; ++i) {
          printf("(%zu) Mismatch -> %.6f vs %.6f\n", idx, cur_data_a[i], cur_data_b[i]);
        }
      }
    }
  }, num_mismatch_one);
  Kokkos::fence();

  // STAGE 2: HASH MISMATCH VALIDATION THROUGH DIRECT COMPARISON
  if (num_mismatch_one > 0) {
    val_hash_vec.to_host();
    Kokkos::View<float*>::HostMirror data_a_h = Kokkos::create_mirror_view(data_a);
    Kokkos::View<float*>::HostMirror data_b_h = Kokkos::create_mirror_view(data_b);
    Kokkos::deep_copy(data_a_h, data_a);
    Kokkos::deep_copy(data_b_h, data_b);
    size_t val_data_len = num_mismatch_one*chunk_size;
    Kokkos::View<uint8_t*> val_data_a_d("Target regionA", val_data_len);
    Kokkos::View<uint8_t*> val_data_b_d("Target regionB", val_data_len);
    gatherMismatchChunks_gpu(data_a_h.data(), data_b_h.data(), val_hash_vec, 
                                num_mismatch_one, chunk_size, val_data_a_d, val_data_b_d);

    // given the value of bitset at the index, change the digest or leave it as is.
    Kokkos::parallel_reduce("HybridMethod-Direct", Kokkos::RangePolicy<>(0, num_mismatch_one), 
    KOKKOS_LAMBDA(size_t idx, size_t& lsum) {
      int num_element = chunk_size / sizeof(float);
      size_t offset = idx * num_element;
      float* cur_data_a = (float*)val_data_a_d.data()+offset;
      float* cur_data_b = (float*)val_data_b_d.data()+offset;
      int dsum = 0;
      for (int i = 0; i < num_element && dsum < 1; ++i) {
        if (Kokkos::fabs(cur_data_a[i] - cur_data_b[i]) > error) {
          dsum = 1;
        }
      }
      if(dsum > 0) {
        lsum += 1;
      }
    }, num_mismatch_two);
  }
}

std::vector<float> load_data_from_file(std::string filename, size_t start_offset) {
  std::vector<float> data;
  std::ifstream f;
  f.exceptions(std::ifstream::failbit | std::ifstream::badbit);
  try {
    f.open(filename, std::ifstream::binary);
    if (!f.is_open()) throw std::runtime_error("Failed to open file");

    f.seekg(0, f.end);
    size_t file_size = f.tellg();
    int data_len = file_size - start_offset;
    data.resize(data_len / sizeof(float));

    f.seekg(start_offset, f.beg);  
    f.read((char*)(data.data()), data_len);
    f.close();
  } catch (const std::ifstream::failure& e) {
      std::cerr << "Exception opening/reading/closing file: " << e.what() << '\n';
  }
  return data;
}

int main(int argc, char** argv) {
  
  float error = std::stof(argv[1]);
  int chunk_size = std::atoi(argv[2]);
  std::string file_a = argv[3];
  std::string file_b = argv[4];

  size_t file_start_offset = 0;
  std::vector<float> data_a = load_data_from_file(file_a, file_start_offset);
  std::vector<float> data_b = load_data_from_file(file_b, file_start_offset);
  size_t data_len = data_a.size();
  size_t num_chunks =  data_len*sizeof(float) / chunk_size;
  if(num_chunks*chunk_size < data_len*sizeof(float))
    num_chunks += 1;
  INFO("Params:: DataSize=" << data_len << ", ChunkSize=" << chunk_size << ", NofChunks=" << num_chunks);

  int num_tests = (error > 0) ? 1 : 10;
  const int seed = 0x4321;
  std::mt19937 prng(seed);
	std::uniform_real_distribution<float> error_dist(1e-6, 1e-1);

  Kokkos::initialize(argc, argv);
  {
    Kokkos::View<float*> data_a_d("RunOneData", data_len);
    Kokkos::View<float*> data_b_d("RunTwoData", data_len);
    Kokkos::View<float*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> > data_a_h((float*)data_a.data(), data_len);
    Kokkos::View<float*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> > data_b_h((float*)data_b.data(), data_len);
    Kokkos::deep_copy(data_a_d, data_a_h);
    Kokkos::deep_copy(data_b_d, data_b_h);

    int fail_count = 0;
    for(int i = 0; i < num_tests; i++) {
      error = (num_tests == 1) ? error : error_dist(prng);
      
      // DIRECT COMPARISON
      size_t direct_mismatch = 0;
      for (size_t idx = 0; idx < num_chunks; idx++)    {
        int num_element = chunk_size / sizeof(float);
        size_t offset = idx*num_element;
        direct_mismatch += areAbsoluteEqual(data_a.data()+offset, data_b.data()+offset, num_element, error, idx);
      }

      // OUR APPROACH (HYBRID COOMPARISON - CPU)
      /*
      int hybrid_mismatch_cpu_one, hybrid_mismatch_cpu_two;
      hybridHash_cpu(num_chunks, chunk_size, 
        (float*)data_a.data(), (float*)data_b.data(), 
        error, hybrid_mismatch_cpu_one, hybrid_mismatch_cpu_two);
      */

      // OUR APPROACH (HYBRID COMPARISON - GPU)
      size_t hybrid_mismatch_gpu_one, hybrid_mismatch_gpu_two;
      hybridHash_gpu(num_chunks, chunk_size, data_a_d, data_b_d, error, 
                    hybrid_mismatch_gpu_one, hybrid_mismatch_gpu_two);

      // LOGGING RESULTS
      if(direct_mismatch == hybrid_mismatch_gpu_two) {
        INFO("Test" << i << " (e=" << error << "): Dir=" << 
          direct_mismatch << "; Ours=" << hybrid_mismatch_gpu_two << ".....OK");
      } else {
        INFO("Test" << i << " (e=" << error << "): Dir=" << 
          direct_mismatch << "; Ours=" << hybrid_mismatch_gpu_two << ".....FAIL");
        fail_count += 1;
      }
    }
    INFO("Validation result: " << fail_count << "/" << num_tests << " tests failed");
  }
  Kokkos::finalize();
  return 0;
}
