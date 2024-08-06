#define __DEBUG
#include "debug.hpp"

#include <cstring>
#include <random>
#include <vector>

#include "threaded_cmp.hpp"

bool
write_file(const std::string &fn, uint8_t *buffer, size_t size) {
    bool ret;
    int fd = open(fn.c_str(), O_CREAT | O_TRUNC | O_WRONLY, 0644);
    if (fd == -1) {
        FATAL("cannot open " << fn << ", error = " << strerror(errno));
        return false;
    }
    size_t transferred = 0, remaining = size;
    while (remaining > 0) {
        size_t ret = write(fd, buffer + transferred, remaining);
        if (ret < 0)
            FATAL("cannot write " << size << " bytes to " << fn
                                  << " , error = " << std::strerror(errno));
        remaining -= ret;
        transferred += ret;
    }
    close(fd);
    return ret;
}

int
main() {
    const int N = 50000000, SEED = 0x1234, CHUNK_SIZE = 128;
    double ERROR = 0.01;

    std::vector<double> run_1(N), run_2(N);

    // generate the test data
    TIMER_START(gen_data);
#pragma omp parallel
    {
        std::mt19937 prng(SEED + omp_get_thread_num());
        std::uniform_real_distribution<double> prng_dist(0.0, 10.0);
        std::uniform_real_distribution<double> error_dist(0.0, ERROR);
#pragma omp for
        for (int i = 0; i < N; i++) {
            run_1[i] = prng_dist(prng);
            run_2[i] = run_1[i] + error_dist(prng);
        }
    }

    write_file("checkpoint_1.dat", (uint8_t *) run_1.data(),
               N * sizeof(double));
    write_file("checkpoint_2.dat", (uint8_t *) run_2.data(),
               N * sizeof(double));
    run_1.clear();
    run_2.clear();
    TIMER_STOP(gen_data, "finished generating the test data");

    // run the comparisons
    std::vector<size_t> offsets;
    for (int i = 0; i < N; i += CHUNK_SIZE)
        offsets.push_back(i);
    TIMER_START(cmp_data);
    threaded_cmp_t<double> threaded_cmp("checkpoint_1.dat", "checkpoint_2.dat",
                                        offsets, CHUNK_SIZE, ERROR);
    size_t mismatches = threaded_cmp.run_comparisons();
    TIMER_STOP(cmp_data, "finished direct comparison");
    INFO("Found " << mismatches << "/" << N << " mismatching values");

    return 0;
}
