#include <Kokkos_Core.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/types/vector.hpp>
#include <chrono>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

struct alignas(16) HashDigest {
    uint8_t digest[16] = {0};

    // Operator to print HashDigest for debugging
    friend std::ostream &operator<<(std::ostream &os, const HashDigest &hd) {
        // Print the digest in hexadecimal
        for (int i = 0; i < 16; ++i) {
            os << std::hex << static_cast<int>(hd.digest[i]) << " ";
        }

        // Switch back to decimal formatting
        os << std::dec;

        return os;
    }

    template <class Archive> void serialize(Archive &ar) {
        ar(cereal::binary_data(digest, sizeof(digest)));
    }
};

bool
compare_trees(const Kokkos::View<HashDigest *> &tree1,
              const Kokkos::View<HashDigest *> &tree2) {
    if (tree1.extent(0) != tree2.extent(0)) {
        std::cout << "Trees have different sizes: " << tree1.extent(0) << " vs "
                  << tree2.extent(0) << std::endl;
        return false;
    }

    // Create host views to copy data from device for comparison
    Kokkos::View<HashDigest *, Kokkos::HostSpace> tree1_h("tree1_h",
                                                          tree1.extent(0));
    Kokkos::View<HashDigest *, Kokkos::HostSpace> tree2_h("tree2_h",
                                                          tree2.extent(0));

    Kokkos::deep_copy(tree1_h, tree1);
    Kokkos::deep_copy(tree2_h, tree2);

    // Compare element by element
    for (size_t i = 0; i < tree1_h.extent(0); ++i) {
        for (int j = 0; j < 16; ++j) {
            if (tree1_h(i).digest[j] != tree2_h(i).digest[j]) {
                std::cout << "Mismatch at index " << i << ", byte " << j
                          << std::endl;
                return false;
            }
        }
    }
    return true;
}

void
create_array(uint32_t size, Kokkos::View<HashDigest *> &array_d) {
    std::vector<HashDigest> data(size);
    std::mt19937 prng(1234);
    std::uniform_int_distribution<uint8_t> prng_dist(0, 255);
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < 16; ++j) {
            data[i].digest[j] = prng_dist(prng);
        }
    }

    // Create an unmanaged host view for the data and copy to GPU
    HashDigest *data_ptr = data.data();
    Kokkos::View<HashDigest *, Kokkos::HostSpace,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>
        data_h(data_ptr, data.size());
    Kokkos::deep_copy(array_d, data_h);
}

class test_cereal_t {
  public:
    Kokkos::View<HashDigest *> tree_d;
    int test_var;
    uint32_t n_hashes;

    test_cereal_t() : test_var(0), n_hashes(0) {}

    test_cereal_t(uint32_t _size)
        : tree_d("tree", _size), test_var(0), n_hashes(_size) {}

    test_cereal_t(uint32_t size, Kokkos::View<HashDigest *> &array_d)
        : tree_d("tree", size), test_var(12085) {
        Kokkos::deep_copy(tree_d, array_d);
        n_hashes = tree_d.extent(0);
    }

    template <class Archive> void save(Archive &ar) const {
        // Copy GPU data to a temporary host buffer
        std::vector<HashDigest> temp_data(n_hashes);
        Kokkos::View<HashDigest *, Kokkos::HostSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>
            temp_view(temp_data.data(), n_hashes);
        Kokkos::deep_copy(temp_view, tree_d);
        ar(n_hashes);
        ar(test_var);
        ar(cereal::binary_data(temp_data.data(),
                               n_hashes * sizeof(HashDigest)));
    }

    template <class Archive> void load(Archive &ar) {
        ar(n_hashes);
        ar(test_var);

        tree_d = Kokkos::View<HashDigest *>("tree", n_hashes);
        std::vector<HashDigest> temp_data(n_hashes);
        ar(cereal::binary_data(temp_data.data(),
                               n_hashes * sizeof(HashDigest)));

        Kokkos::View<HashDigest *, Kokkos::HostSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>
            temp_view(temp_data.data(), n_hashes);
        Kokkos::deep_copy(tree_d, temp_view);
    }
};

int
main() {
    Kokkos::initialize();
    {
        uint32_t size = 67108864;   // 1GB
        Kokkos::View<HashDigest *> array_d("GPU array", size);
        create_array(size, array_d);

        // Create and serialize object
        test_cereal_t cereal_obj(size, array_d);
        // printf("CEREAL::Before\n");
        // for (int i = 0; i < size; i++) {
        //     std::cout << cereal_obj.tree_d(i) << std::endl;
        // }

        auto start_serialize = std::chrono::high_resolution_clock::now();
        {
            // Serialize to file
            std::ofstream ofs("serialized_data.cereal");
            if (!ofs) {
                std::cerr << "Error opening file for writing!" << std::endl;
                return -1;
            }
            cereal::BinaryOutputArchive oa(ofs);
            oa(cereal_obj);
        }
        auto end_serialize = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> serialize_duration =
            end_serialize - start_serialize;

        test_cereal_t new_cereal_obj;
        auto start_deserialize = std::chrono::high_resolution_clock::now();
        {
            // Deserialize object
            std::ifstream ifs("serialized_data.cereal");
            if (!ifs) {
                std::cerr << "Error opening file for reading!" << std::endl;
                return -1;
            }
            cereal::BinaryInputArchive ia(ifs);
            ia(new_cereal_obj);
        }
        auto end_deserialize = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> deserialize_duration =
            end_deserialize - start_deserialize;

        // printf("CEREAL::After\n");
        // for (int i = 0; i < size; i++) {
        //     std::cout << new_cereal_obj.tree_d(i) << std::endl;
        // }
        bool result = compare_trees(cereal_obj.tree_d, new_cereal_obj.tree_d);
        std::cout << "Comparison result: " << (result ? "Equal" : "Not Equal")
                  << std::endl;
        std::cout << "Serialization took " << serialize_duration.count()
                  << " seconds" << std::endl;
        std::cout << "Deserialization took " << deserialize_duration.count()
                  << " seconds" << std::endl;
    }
    Kokkos::finalize();
    return 0;
}
