#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/array.hpp>
#include <Kokkos_Core.hpp>
#include <fstream>
#include <vector>
#include <random>

// Example class with Kokkos::View
class MyGpuArray {
  public:
    Kokkos::View<float *> gpu_array;   // GPU memory
    Kokkos::View<float *>
        host_array;   // Host memory (temporary for serialization)

    MyGpuArray(uint32_t size)
        : gpu_array("gpu_array", size), host_array("host_array", size) {
            
        std::vector<float> data(size);
        std::mt19937 prng(1234);
        std::uniform_real_distribution<float> prng_dist(1, 10);
        for (int i = 0; i < size; ++i) {
            data[i] = prng_dist(prng);
        }
        float* data_ptr = (float*)data.data();
        Kokkos::View<float*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> > data_h(data_ptr, data.size());
        Kokkos::deep_copy(gpu_array, data_h);
    }

    // Serialize function
    template <class Archive>
    void serialize(Archive &ar, const unsigned int version) {
        // Ensure the host array is large enough
        Kokkos::deep_copy(host_array, gpu_array);

        // Get the size of the array
        uint32_t size = host_array.extent(0);

        // Serialize the size and the array data
        ar & size;
        ar & boost::serialization::make_array(host_array.data(), size);
    }

    // Deserialize function
    template <class Archive>
    void deserialize(Archive &ar, const unsigned int version) {
        // Deserialize the size and the array data
        uint32_t size;
        ar & size;

        // Allocate GPU arrays if needed
        gpu_array = Kokkos::View<float *>("gpu_array", size);

        std::vector<float> temp_data(size);
        ar & boost::serialization::make_array(temp_data.data(), size);

        Kokkos::View<float *, Kokkos::HostSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>
            host_array((float *) temp_data.data(), size);

        Kokkos::deep_copy(gpu_array, host_array);
    }
};

int
main() {
    Kokkos::initialize();

    {
        // Create and serialize object
        MyGpuArray gpu_data(10);
        printf("Before\n");
        for (int i = 0; i < 10; i++) {
            printf("%f ", gpu_data.gpu_array(i));
        }
        printf("\n");

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
            printf("%f ", gpu_data_loaded.gpu_array(i));
        }
        printf("\n");
    }

    Kokkos::finalize();
    return 0;
}
