#include "stdio.h"
#include <string>
#include <fstream>
#include <libgen.h>
#include <random>
#include <argparse/argparse.hpp>
#include "data_generation.hpp"

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    argparse::ArgumentParser program("Dedup Files");
    program.add_argument("-v", "--verbose")
      .help("Generate files")
      .default_value(false)
      .implicit_value(true);
    program.add_argument("--data-len")
      .help("Length of data in elements")
      .default_value(1048576ULL)
      .nargs(1)
      .scan<'u', uint64_t>();
    program.add_argument("-n", "--num-files")
      .help("Number of data files to generate")
      .default_value(static_cast<uint32_t>(10))
      .nargs(1)
      .scan<'u', uint32_t>();
    program.add_argument("-p", "--perturb-mode")
      .help("Mode for perturbing the data")
      .choices("R", "B", "C", "I", "S", "Z", "W", "P")
      .nargs(1)
      .default_value(std::string("P"));
    program.add_argument("-e", "--error")
      .help("How much error to perturb data")
      .nargs(1)
      .default_value(static_cast<double>(1e-5))
      .scan<'g', double>();
    program.add_argument("--num-changes")
      .help("Number of elements to change")
      .default_value(static_cast<uint64_t>(262144))
      .nargs(1)
      .scan<'u', uint64_t>();
    program.add_argument("--data-type")
      .help("Data type to generate")
      .choices("bytes", "uint32_t", "uint64_t", "float", "double")
      .default_value(std::string("bytes"))
      .nargs(1);
    program.add_argument("outname")
      .help("Output filename")
      .nargs(1)
      .default_value(std::string("generated_data"));
    // Parse and retrieve arguments
    try {
      program.parse_args(argc, argv);
    } catch (const std::exception& err) {
      std::cerr << err.what() << std::endl;
      std::cerr << program;
      std::exit(1);
    }
    uint64_t    data_len       = program.get<uint64_t>("--data-len");
    printf("Data length in elements: %lu\n", data_len);
    uint32_t    num_files      = program.get<uint32_t>("-n");
    printf("Number of files:         %d\n",  num_files);
    std::string generator_mode = program.get<std::string>("-p");
    printf("Perturb mode:            %s\n",  generator_mode.c_str());
    uint64_t    num_changes    = program.get<uint64_t>("--num-changes");
    printf("Num changes:             %lu\n", num_changes);
    std::string data_type      = program.get<std::string>("--data-type");
    printf("Data type:               %s\n",  data_type.c_str());
    std::string out_filename   = program.get<std::string>("outname");
    printf("File name:               %s\n",  out_filename.c_str());
    double      error          = program.get<double>("--error");
    printf("Error:                   %f\n",  error);
    DataGenerationMode mode = Perturb;
    printf("Data length in elements: %lu\n", data_len);
    printf("Number of files:         %d\n",  num_files);
    printf("Perturb mode:            %s\n",  generator_mode.c_str());
    printf("Num changes:             %lu\n", num_changes);
    printf("Data type:               %s\n",  data_type.c_str());
    printf("File name:               %s\n",  out_filename.c_str());

    Kokkos::Random_XorShift64_Pool<> rand_pool(1931);
    std::default_random_engine generator(1931);

    if(data_type.compare("bytes") == 0) {
      Kokkos::View<uint8_t*> data = generate_initial_data<uint8_t>(data_len);
      Kokkos::fence();
      write_data<uint8_t>(out_filename + std::to_string(0) + std::string(".dat"), data);

      for(uint32_t i=1; i<num_files; i++) {
        perturb_data<uint8_t>(data, num_changes, mode, rand_pool, generator, static_cast<uint8_t>(error));
        Kokkos::fence();
        write_data<uint8_t>(out_filename + std::to_string(i) + std::string(".dat"), data);
      }
    } else if (data_type.compare("uint32_t") == 0) {
      Kokkos::View<uint32_t*> data = generate_initial_data<uint32_t>(data_len);
      Kokkos::fence();
      write_data<uint32_t>(out_filename + std::to_string(0) + std::string(".dat"), data);

      for(uint32_t i=1; i<num_files; i++) {
        perturb_data<uint32_t>(data, num_changes, mode, rand_pool, generator, static_cast<uint32_t>(error));
        Kokkos::fence();
        write_data<uint32_t>(out_filename + std::to_string(i) + std::string(".dat"), data);
      }
    } else if (data_type.compare("uint64_t") == 0) {
      Kokkos::View<uint64_t*> data = generate_initial_data<uint64_t>(data_len);
      Kokkos::fence();
      write_data<uint64_t>(out_filename + std::to_string(0) + std::string(".dat"), data);

      for(uint32_t i=1; i<num_files; i++) {
        perturb_data<uint64_t>(data, num_changes, mode, rand_pool, generator, static_cast<uint64_t>(error));
        Kokkos::fence();
        write_data<uint64_t>(out_filename + std::to_string(i) + std::string(".dat"), data);
      }
    } else if (data_type.compare("float") == 0) {
      Kokkos::View<float*> data = generate_initial_data<float>(data_len);
      Kokkos::fence();
      write_data<float>(out_filename + std::to_string(0) + std::string(".dat"), data);

      for(uint32_t i=1; i<num_files; i++) {
        perturb_data<float>(data, num_changes, mode, rand_pool, generator, static_cast<float>(error));
        Kokkos::fence();
        write_data<float>(out_filename + std::to_string(i) + std::string(".dat"), data);
      }
    } else if (data_type.compare("double") == 0) {
      Kokkos::View<double*> data = generate_initial_data<double>(data_len);
      Kokkos::fence();
      write_data<double>(out_filename + std::to_string(0) + std::string(".dat"), data);

      for(uint32_t i=1; i<num_files; i++) {
        perturb_data<double>(data, num_changes, mode, rand_pool, generator, static_cast<double>(error));
        Kokkos::fence();
        write_data<double>(out_filename + std::to_string(i) + std::string(".dat"), data);
      }
    }
  }
  Kokkos::finalize();
}
