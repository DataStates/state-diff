#ifndef DATA_GENERATION_HPP
#define DATA_GENERATION_HPP

#include "common/debug.hpp"
#include "common/direct_io.hpp"
#include "stdio.h"
#include <Kokkos_Bitset.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <Kokkos_UnorderedMap.hpp>
#include <fcntl.h>
#include <fstream>
#include <libgen.h>
#include <random>
#include <string>
#include <type_traits>
#include <unistd.h>

enum DataGenerationMode {
    Perturb,
};

template <typename DataType, typename Generator>
KOKKOS_INLINE_FUNCTION DataType
generate_random(Generator &rand_gen) {
    if (std::is_same<DataType, int32_t>::value) {
        return rand_gen.rand();
    } else if (std::is_same<DataType, int64_t>::value) {
        return rand_gen.rand64();
    } else if (std::is_same<DataType, uint32_t>::value) {
        return rand_gen.urand();
    } else if (std::is_same<DataType, uint64_t>::value) {
        return rand_gen.urand64();
    } else if (std::is_same<DataType, size_t>::value) {
        return static_cast<size_t>(rand_gen.urand64());
    } else if (std::is_same<DataType, float>::value) {
        return rand_gen.frand();
    } else if (std::is_same<DataType, double>::value) {
        return rand_gen.drand();
    } else {
      return static_cast<uint8_t>(rand_gen.urand() % 256);
    }
}

template <typename DataType, typename Generator>
KOKKOS_INLINE_FUNCTION DataType
generate_random(Generator &rand_gen, DataType beg, DataType end) {
    if (std::is_same<DataType, int32_t>::value) {
        return rand_gen.rand(beg, end);
    } else if (std::is_same<DataType, int64_t>::value) {
        return rand_gen.rand64(beg, end);
    } else if (std::is_same<DataType, uint32_t>::value) {
        return rand_gen.urand(beg, end);
    } else if (std::is_same<DataType, uint64_t>::value) {
        return rand_gen.urand64(beg, end);
    } else if (std::is_same<DataType, size_t>::value) {
        return static_cast<size_t>(rand_gen.urand64(beg, end));
    } else if (std::is_same<DataType, float>::value) {
        return rand_gen.frand(beg, end);
    } else if (std::is_same<DataType, double>::value) {
        return rand_gen.drand(beg, end);
    } else {
      return static_cast<uint8_t>(rand_gen.urand() % 256);
    }
}

template <typename DataType, typename Generator>
KOKKOS_INLINE_FUNCTION DataType
generate_random(Generator &rand_gen, DataType range) {
    return generate_random(rand_gen, static_cast<DataType>(0), range);
}

template <typename DataType>
Kokkos::View<DataType *>
generate_initial_data(size_t max_data_len) {
    Kokkos::Random_XorShift64_Pool<> rand_pool(1931);
    Kokkos::View<DataType *> data("Data", max_data_len);
    auto policy = Kokkos::RangePolicy<size_t>(0LLU, max_data_len);
    Kokkos::parallel_for(
        "Fill random", policy, KOKKOS_LAMBDA(const size_t i) {
            auto rand_gen = rand_pool.get_state();
            data(i) =
                generate_random<DataType>(rand_gen, static_cast<DataType>(1));
            rand_pool.free_state(rand_gen);
        });
    return data;
}

template <typename DataType>
void
perturb_data(Kokkos::View<DataType *> &data0, const size_t num_changes,
             DataGenerationMode mode,
             Kokkos::Random_XorShift64_Pool<> &rand_pool,
             std::default_random_engine &generator,
             DataType perturb = static_cast<DataType>(0)) {
    if (mode == Perturb) {
        Kokkos::View<DataType *> original("Original copy", data0.size());
        Kokkos::deep_copy(original, data0);
        printf("Perturbing %zu out of %zu elements in the data\n", num_changes,
               data0.size());
        Kokkos::Bitset<> bitset(data0.size());
        bitset.reset();
        while (bitset.count() < num_changes) {
            auto policy =
                Kokkos::RangePolicy<uint64_t>(0, num_changes - bitset.count());
            Kokkos::parallel_for(
                "Gen random indicies", policy, KOKKOS_LAMBDA(const uint64_t j) {
                    bitset.set(data0.size() - 1);
                    auto rand_gen = rand_pool.get_state();
                    auto index = rand_gen.rand64() % data0.size();
                    bitset.set(index);
                    rand_pool.free_state(rand_gen);
                });
            Kokkos::fence();
        }

        auto policy = Kokkos::RangePolicy<uint64_t>(0, data0.size());
        Kokkos::parallel_for(
            "Gen random indicies", policy, KOKKOS_LAMBDA(const uint64_t j) {
                auto rand_gen = rand_pool.get_state();
                if (bitset.test(j)) {
                    while ((data0(j) == original(j)) ||
                           (Kokkos::abs((double)data0(j) -
                                        (double)original(j)) < perturb)) {
                        data0(j) =
                            original(j) +
                            generate_random(rand_gen, (DataType)(perturb),
                                            (DataType)(2 * perturb));
                    }
                }
                rand_pool.free_state(rand_gen);
            });
        Kokkos::fence();

        uint64_t ndiff = 0;
        Kokkos::parallel_reduce(
            "Verify perturbations", policy,
            KOKKOS_LAMBDA(const uint64_t i, uint64_t &update) {
                if (Kokkos::abs((double)data0(i) - (double)original(i)) <
                    perturb)
                    update += 1;
            },
            Kokkos::Sum<uint64_t>(ndiff));
        Kokkos::fence();
        printf("Number of elements within the error bounds : %lu\n", ndiff);
    }
}

bool
write_file(const std::string &fn, uint8_t *buffer, size_t size) {
    bool ret = true;
    // int fd = open(fn.c_str(), O_CREAT | O_TRUNC | O_WRONLY | O_DIRECT, 0644);
    int fd = open(fn.c_str(), O_CREAT | O_TRUNC | O_WRONLY, 0644);
    if (fd == -1) {
        std::cout << "cannot open " << fn << ", error = " << strerror(errno);
        exit(-1);
    }
    size_t transferred = 0, remaining = size;
    while (remaining > 0) {
        auto ret = write(fd, buffer + transferred, remaining);
        if (ret < 0)
            std::cout << "cannot write " << size << " bytes to " << fn
                      << " , error = " << std::strerror(errno);
        exit(-1);
        remaining -= ret;
        transferred += ret;
    }
    close(fd);
    return ret;
}

template <typename DataType>
void
write_data(const std::string &filename, Kokkos::View<DataType *> &data) {
    typename Kokkos::View<DataType *>::HostMirror data_h =
        Kokkos::create_mirror_view(data);
    Kokkos::deep_copy(data_h, data);
    unaligned_direct_write(filename, (uint8_t *)(data_h.data()),
                           data.size() * sizeof(DataType));
}
#endif   // DATA_GENERATION_HPP
