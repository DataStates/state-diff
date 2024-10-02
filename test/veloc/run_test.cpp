#include "veloc.hpp"
#include <omp.h>
#include <random>

int
main(int argc, char **argv) {

    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <run_id> <config_file>"
                  << std::endl;
        return -1;
    }

    // maximum floating-point (FP) value in synthetic data
    float max_float = 100.0;
    // minimum FP value in synthetic data
    float min_float = 0.0;
    // size in bytes of the synthetic data (1GB)
    size_t data_size = 1024 * 1024 * 1024;
    // Application error tolerance
    float error_tolerance = 1e-4;
    // Random number seed to generate the synthetic data
    int seed = 0x123;
    std::string fn = "veloc.statediff.ckpt";

    // Create synthetic datasets
    size_t data_len = data_size / sizeof(float);
    std::vector<float> run_0_data(data_len), run_1_data(data_len),
        run_2_data(data_len);
#pragma omp parallel
    {
        std::mt19937 prng(seed + omp_get_thread_num());
        std::uniform_real_distribution<float> prng_dist(min_float, max_float);
        std::uniform_real_distribution<float> in_error_dist(
            0.0, 0.5 * error_tolerance);
        std::uniform_real_distribution<float> outof_error_dist(
            1.5 * error_tolerance, 2 * error_tolerance);
#pragma omp for
        for (size_t i = 0; i < data_len; ++i) {
            run_0_data[i] = prng_dist(prng);
            run_1_data[i] = run_0_data[i] + in_error_dist(prng);
            run_2_data[i] = run_0_data[i] + outof_error_dist(prng);
        }
    }

    //  checkpoint with VeloC
    int run_id = std::stoi(argv[1]);
    std::string cfg_file = argv[2];
    veloc::client_t *ckpt = veloc::get_client(run_id, cfg_file);
    if (!ckpt) {
        throw std::runtime_error("VeloC client initialization failed");
    }

    if (run_id == 1) {
        ckpt->mem_protect(0, run_0_data.data(), data_len,
                          sizeof(run_0_data[0]));
    } else {
        ckpt->mem_protect(0, run_1_data.data(), data_len,
                          sizeof(run_1_data[0]));
    }

    if (!ckpt->checkpoint(fn, 0))
        throw std::runtime_error("checkpointing failed");

    std::cout << "Checkpointing successful for run " << run_id << std::endl;
    return 0;
}
