#include <Kokkos_Core.hpp>
#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include "liburing_reader.hpp"
#include "common/direct_io.hpp"

std::vector<float>
load_data_from_file(const std::string &filename, size_t start_offset) {
    std::vector<float> data;
    std::ifstream f;
    f.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    try {
        f.open(filename, std::ifstream::binary);
        if (!f.is_open())
            throw std::runtime_error("Failed to open file");

        f.seekg(0, f.end);
        size_t file_size = f.tellg();
        if (start_offset >= file_size)
            throw std::runtime_error("Start offset exceeds file size");

        size_t data_len = file_size - start_offset;
        if (data_len % sizeof(float) != 0)
            throw std::runtime_error("File size is not aligned to float size");

        data.resize(data_len / sizeof(float));
        f.seekg(start_offset, f.beg);
        f.read(reinterpret_cast<char *>(data.data()), data_len);
        f.close();
    } catch (const std::exception &e) {
        std::cerr << "Error loading file: " << e.what() << '\n';
    }
    return data;
}

void load_io_uring(std::string &filename) {
	off_t filesize;
	get_file_size(filename, &filesize);
	size_t dsize = static_cast<size_t>(filesize);

	std::vector<float> buffer(dsize/sizeof(float), 0);
	int n_segs = 4;
	std::vector<segment_t> segments(n_segs);
	liburing_io_reader_t reader(filename);

	int c_size = dsize / n_segs;

	for(int i = 0; i < n_segs; i++) {
		segment_t seg;
		seg.buffer = (uint8_t*)(buffer.data()+c_size*i);
		seg.offset = c_size*i;
		seg.size = dsize;
		segments[i] = seg;
	}
	
	auto start_load = std::chrono::high_resolution_clock::now();
	reader.enqueue_reads(segments);
	reader.wait_all();
	auto end_load = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> load_time = end_load - start_load;
	std::cout << "File loading throughput (uring) = " << (dsize/1024/1024/1024)/load_time.count() << " GB/s" << std::endl;
}

int
main(int argc, char **argv) {

    // std::string file_a = argv[1];
    // std::string file_b = argv[2];
    std::string file_a =
        "/eagle/projects/RECUP/kassogba/veloc-ckpt/haac/sc-experiments/4gpus/"
        "np796-500mil/run1/m000p.mpirestart-combined-0-10.dat";
    std::string file_b =
        "/eagle/projects/RECUP/kassogba/veloc-ckpt/haac/sc-experiments/4gpus/"
        "np796-500mil/run2/m000p.mpirestart-combined-0-10.dat";

    load_io_uring(file_a);
    load_io_uring(file_b);
    exit(0);

    auto start_load = std::chrono::high_resolution_clock::now();
    std::vector<float> data_a = load_data_from_file(file_a, 0);
    auto end_load = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> load_time = end_load - start_load;
    std::vector<float> data_b = load_data_from_file(file_b, 0);
    size_t data_size = data_a.size();
    size_t data_len = data_size * sizeof(float);
    std::cout << "File loading throughput = " << (data_len/1024/1024/1024)/load_time.count() << " GB/s" << std::endl;

    if (data_size != data_b.size()) {
        std::cerr << "Error: Input files have mismatched sizes\n";
        return 1;
    }

    Kokkos::initialize(argc, argv);
    {
        const int start_chunk_size = 128;

        Kokkos::View<float *, Kokkos::HostSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>
            data_a_h(data_a.data(), data_size);
        Kokkos::View<float *, Kokkos::HostSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>
            data_b_h(data_b.data(), data_size);

        //Kokkos::View<float *> data_a_d("RunOneData", data_size);
        //Kokkos::View<float *> data_b_d("RunTwoData", data_size);
        auto start_copy = std::chrono::high_resolution_clock::now();
        //Kokkos::deep_copy(data_a_d, data_a_h);
        //Kokkos::deep_copy(data_b_d, data_b_h);
        auto end_copy = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> copy_time = end_copy - start_copy;

        float *data_a = (float *) data_a_h.data();
        float *data_b = (float *) data_b_h.data();

        //float *data_a = (float *)data_a_d.data();
        //float *data_b = (float *)data_b_d.data();

        std::fstream benchmark_stream("compare_timings.csv",
                                      std::ios::out | std::ios::app);
        if (!benchmark_stream.is_open()) {
            std::cerr << "Failed to open benchmark log file\n";
            Kokkos::finalize();
            return 1;
        }

        if (benchmark_stream.tellp() == 0) {
            benchmark_stream << "data size,chunk size,copy time,compare time\n";
        }

        for (size_t chunk_size = start_chunk_size; chunk_size <= data_len;
             chunk_size *= 2) {
            std::cout << "Params:: DataSize=" << data_len
                      << ", ChunkSize=" << chunk_size << std::endl;

            auto start_compare = std::chrono::high_resolution_clock::now();
            int num_element = chunk_size / sizeof(float);
            uint64_t num_mismatch = 0;
            using PolicyType = Kokkos::RangePolicy<size_t,
            	Kokkos::DefaultHostExecutionSpace>;
            //using PolicyType =
            //   Kokkos::RangePolicy<size_t, Kokkos::DefaultExecutionSpace>;
            auto range_policy = PolicyType(0, num_element);
            Kokkos::parallel_reduce(
                "Count differences", range_policy,
                KOKKOS_LAMBDA(size_t idx, size_t & update) {
                    if (Kokkos::fabs(data_a[idx] - data_b[idx]) > 0.001) {
                        update += 1;
                    }
                },
                Kokkos::Sum<uint64_t>(num_mismatch));

            auto end_compare = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> compare_time =
                end_compare - start_compare;

            benchmark_stream << data_len << "," << chunk_size << ","
                             << copy_time.count() << "," << compare_time.count()
                             << "\n";
        }
        benchmark_stream.close();
    }
    Kokkos::finalize();
    return 0;
}
