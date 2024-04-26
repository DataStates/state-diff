#ifndef DATA_GENERATION_HPP
#define DATA_GENERATION_HPP

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <Kokkos_UnorderedMap.hpp>
#include <Kokkos_Bitset.hpp>
#include "stdio.h"
#include <string>
#include <fstream>
#include <libgen.h>
#include <random>
#include "debug.hpp"
//#include "state_diff.hpp"
#include <type_traits>
#include <unistd.h>
#include <fcntl.h>

enum DataGenerationMode {
  Perturb,
};

template<typename DataType, typename Generator>
KOKKOS_INLINE_FUNCTION
DataType generate_random(Generator& rand_gen) {
  if(std::is_same<DataType, uint8_t>::value) {
    return static_cast<uint8_t>(rand_gen.urand() % 256);
  } else if(std::is_same<DataType, int32_t>::value) {
    return rand_gen.rand();
  } else if(std::is_same<DataType, int64_t>::value) {
    return rand_gen.rand64();
  } else if(std::is_same<DataType, uint32_t>::value) {
    return rand_gen.urand();
  } else if(std::is_same<DataType, uint64_t>::value) {
    return rand_gen.urand64();
  } else if(std::is_same<DataType, size_t>::value) {
    return static_cast<size_t>(rand_gen.urand64());
  } else if(std::is_same<DataType, float>::value) {
    return rand_gen.frand();
  } else if(std::is_same<DataType, double>::value) {
    return rand_gen.drand();
  }
}

template<typename DataType, typename Generator>
KOKKOS_INLINE_FUNCTION
DataType generate_random(Generator& rand_gen, DataType beg, DataType end) {
  if(std::is_same<DataType, uint8_t>::value) {
    return static_cast<uint8_t>(rand_gen.urand() % 256);
  } else if(std::is_same<DataType, int32_t>::value) {
    return rand_gen.rand(beg, end);
  } else if(std::is_same<DataType, int64_t>::value) {
    return rand_gen.rand64(beg, end);
  } else if(std::is_same<DataType, uint32_t>::value) {
    return rand_gen.urand(beg, end);
  } else if(std::is_same<DataType, uint64_t>::value) {
    return rand_gen.urand64(beg, end);
  } else if(std::is_same<DataType, size_t>::value) {
    return static_cast<size_t>(rand_gen.urand64(beg, end));
  } else if(std::is_same<DataType, float>::value) {
    return rand_gen.frand(beg, end);
  } else if(std::is_same<DataType, double>::value) {
    return rand_gen.drand(beg, end);
  }
}

template<typename DataType, typename Generator>
KOKKOS_INLINE_FUNCTION
DataType generate_random(Generator& rand_gen, DataType range) {
  return generate_random(rand_gen, static_cast<DataType>(0), range);
}

template<typename DataType>
Kokkos::View<DataType*> generate_initial_data(size_t max_data_len) {
  Kokkos::Random_XorShift64_Pool<> rand_pool(1931);
  Kokkos::View<DataType*> data("Data", max_data_len);
  auto policy = Kokkos::RangePolicy<size_t>(0LLU, max_data_len);
  Kokkos::parallel_for("Fill random", policy, KOKKOS_LAMBDA(const size_t i) {
    auto rand_gen = rand_pool.get_state();
    data(i) = generate_random<DataType>(rand_gen, static_cast<DataType>(1));
    rand_pool.free_state(rand_gen);
  });
  return data;
}

template<typename DataType>
void perturb_data(Kokkos::View<DataType*>&          data0, 
                  const size_t                      num_changes, 
                  DataGenerationMode                mode, 
                  Kokkos::Random_XorShift64_Pool<>& rand_pool, 
                  std::default_random_engine&       generator,
                  DataType                          perturb=static_cast<DataType>(0)) {
  if(mode == Perturb) {
        Kokkos::View<DataType*> original("Original copy", data0.size());
        Kokkos::deep_copy(original, data0);
        printf("Perturbing %zu out of %zu elements in the data\n", num_changes, data0.size());
        Kokkos::Bitset<>bitset(data0.size());
        bitset.reset();
        while(bitset.count() < num_changes-1) {
            auto policy = Kokkos::RangePolicy<uint64_t>(0, num_changes - bitset.count());
            Kokkos::parallel_for("Gen random indicies", policy, KOKKOS_LAMBDA(const uint64_t j) {
                bitset.set(data0.size()-1);
                auto rand_gen = rand_pool.get_state();
                auto index = rand_gen.rand64() % data0.size();
                bitset.set(index);
                rand_pool.free_state(rand_gen);
            });
            Kokkos::fence();
        }

        auto policy = Kokkos::RangePolicy<uint64_t>(0, data0.size());
        Kokkos::parallel_for("Gen random indicies", policy, KOKKOS_LAMBDA(const uint64_t j) {
            auto rand_gen = rand_pool.get_state();
            if(bitset.test(j)) {
                while( (data0(j) == original(j)) || (Kokkos::abs((double)data0(j) - (double)original(j)) >= perturb)) {
                  data0(j) = original(j) + generate_random(rand_gen, (DataType)(-perturb), (DataType)(perturb));
                }
            }
            rand_pool.free_state(rand_gen);
        });
        Kokkos::fence();

        uint64_t ndiff=0;
        Kokkos::parallel_reduce("Verify perturbations", policy, KOKKOS_LAMBDA(const uint64_t i, uint64_t& update) {
          if( (data0(i) != original(i)) )
            update += 1;
        }, Kokkos::Sum<uint64_t>(ndiff));
        Kokkos::fence();
        printf("Number of mismatches in the error bounds : %lu\n", ndiff);
  }

}

bool write_file(const std::string &fn, uint8_t *buffer, size_t size) {
    bool ret=true;
//    int fd = open(fn.c_str(), O_CREAT | O_TRUNC | O_WRONLY | O_DIRECT, 0644);
    int fd = open(fn.c_str(), O_CREAT | O_TRUNC | O_WRONLY , 0644);
    if (fd == -1) {
//        FATAL("cannot open " << fn << ", error = " << strerror(errno));
        return false;
    }
    size_t transferred = 0, remaining = size;
    while (remaining > 0) {
    	auto ret = write(fd, buffer + transferred, remaining);
//    	if (ret < 0)
//    	    FATAL("cannot write " << size << " bytes to " << fn << " , error = " << std::strerror(errno));
    	remaining -= ret;
    	transferred += ret;
    }
    close(fd);
    return ret;
}

template<typename DataType>
void write_data(const std::string& filename, Kokkos::View<DataType*>& data) {
  typename Kokkos::View<DataType*>::HostMirror data_h = Kokkos::create_mirror_view(data);
  Kokkos::deep_copy(data_h, data);
  write_file(filename, (uint8_t*)(data_h.data()), data_h.size()*sizeof(DataType));

//  using HostView = Kokkos::View<DataType*, 
//                                Kokkos::DefaultHostExecutionSpace, 
//                                Kokkos::MemoryTraits<Kokkos::Unmanaged>
//                                >;
//  size_t npages = data.size()*sizeof(DataType)/getpagesize();
//  if(npages*getpagesize() < data.size()*sizeof(DataType))
//    npages += 1;
//  DataType* data_h_ptr = (DataType*) aligned_alloc(getpagesize(), npages*getpagesize());
//  HostView data_h(data_h_ptr, data.size());
//  Kokkos::deep_copy(data_h, data);
//  write_file(filename, (uint8_t*)(data_h.data()), npages*getpagesize());
//  free(data_h_ptr);

//  FILE *data_file;
//  data_file = fopen(filename.c_str(), "wb");
//  if(data_file == NULL) {
//    printf("Failed to open data file %s\n", filename.c_str());
//  } else {
//    fwrite(data_h.data(), sizeof(DataType), data_h.size(), data_file);
//    fflush(data_file);
//    fclose(data_file);
//  }
}
#endif // DATA_GENERATION_HPP
