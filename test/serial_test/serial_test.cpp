#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/array.hpp>
#include <Kokkos_Core.hpp>
#include <fstream>
#include <vector>
#include <random>
#include <iostream>

// Custom data structure
struct alignas(16) HashDigest {
    uint8_t digest[16] = {0};

    // Optional operator to print HashDigest for debugging
    friend std::ostream &operator<<(std::ostream &os, const HashDigest &hd) {
        for (int i = 0; i < 16; ++i) {
            os << std::hex << static_cast<int>(hd.digest[i]) << " ";
        }
        return os;
    }
};

// Example class with Kokkos::View
class MyGpuArray {
  public:
    Kokkos::View<HashDigest *> gpu_array;   // GPU memory for HashDigest

    // Constructor for MyGpuArray
    MyGpuArray(uint32_t size)
        : gpu_array("gpu_array", size) {
            
        std::vector<HashDigest> data(size);

        // Fill the data with random bytes
        std::mt19937 prng(1234);
        std::uniform_int_distribution<uint8_t> prng_dist(0, 255);
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < 16; ++j) {
                data[i].digest[j] = prng_dist(prng);
            }
        }

        // Create an unmanaged host view for the data and copy to GPU
        HashDigest* data_ptr = data.data();
        Kokkos::View<HashDigest*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> data_h(data_ptr, data.size());
        Kokkos::deep_copy(gpu_array, data_h);
    }

    // Serialize function
    template <class Archive>
    void serialize(Archive &ar, const unsigned int version) {
        // Copy GPU data to a temporary host buffer
        size_t size = gpu_array.extent(0);
        std::vector<HashDigest> temp_data(size);

        // Create a host view with the same size as the GPU array
        Kokkos::View<HashDigest*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> temp_view(temp_data.data(), size);
        
        // Copy data from GPU to the host buffer
        Kokkos::deep_copy(temp_view, gpu_array);

        // Serialize the size and the array data
        ar & size;
        ar & boost::serialization::make_array(reinterpret_cast<uint8_t*>(temp_data.data()), size * sizeof(HashDigest));
    }

    // Deserialize function
    template <class Archive>
    void deserialize(Archive &ar, const unsigned int version) {
        // Deserialize the size and the array data
        uint32_t size;
        ar & size;

        // Allocate GPU arrays if needed
        gpu_array = Kokkos::View<HashDigest *>("gpu_array", size);

        // Temporary buffer for deserialization
        std::vector<HashDigest> temp_data(size);
        ar & boost::serialization::make_array(reinterpret_cast<uint8_t*>(temp_data.data()), size * sizeof(HashDigest));

        // Create a host view with the deserialized data and copy to GPU
        Kokkos::View<HashDigest *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> temp_view(temp_data.data(), size);
        Kokkos::deep_copy(gpu_array, temp_view);
    }
};

int main() {
    Kokkos::initialize();

    {
        // Create and serialize object
        MyGpuArray gpu_data(10);
        printf("Before\n");
        for (int i = 0; i < 10; i++) {
            std::cout << gpu_data.gpu_array(i) << std::endl;
        }

        // Serialize to file
        std::ofstream ofs("gpu_data.bin");
        boost::archive::binary_oarchive oa(ofs);
        oa << gpu_data;
    }

    {
        // Deserialize object
        MyGpuArray gpu_data_loaded(10);
        std::ifstream ifs("gpu_data.bin");
        boost::archive::binary_iarchive ia(ifs);
        ia >> gpu_data_loaded;

        printf("After\n");
        for (int i = 0; i < 10; i++) {
            std::cout << gpu_data_loaded.gpu_array(i) << std::endl;
        }
    }

    Kokkos::finalize();
    return 0;
}
