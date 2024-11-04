#include <random>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <typeinfo>
#include <chrono>
#include <algorithm>
#include <unistd.h>
#include "argparse/argparse.hpp"
#include "io_reader.hpp"
#include "liburing_reader.hpp"
#include "posix_reader.hpp"
#include "mmap_reader.hpp"

bool vectors_match(std::vector<float>& expected, std::vector<float>& buffer, size_t start=0, size_t end=0) {
  if(end == 0)
    end = expected.size();
  for(size_t i=start; i<end; i++) {
    if(expected[i] != buffer[i]) {
      return false;
    }
  }
  return true;
}

void clear_pages(std::string& filename) {
  int fd = open(filename.c_str(), O_RDWR);
  off_t len = lseek(fd, 0, SEEK_END);
  lseek(fd, 0, SEEK_SET);
  posix_fadvise(fd, 0, len, POSIX_FADV_DONTNEED);
  fsync(fd);
  close(fd);
}

template<typename Reader>
bool wait_test(Reader& reader, std::vector<segment_t>& segments, std::vector<float>& expected, std::vector<float>& buffer, size_t chunk_size) {
  bool success = true;
  std::vector<size_t> ids(segments.size(), 0);
  for(size_t i=0; i<ids.size(); i++) {
    ids[i] = i;
  }
  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(ids.begin(), ids.end(), g);
  auto start = std::chrono::high_resolution_clock::now();  
  reader.enqueue_reads(segments);
  for(size_t i: ids) {
    reader.wait(i);
    success = success && vectors_match(expected, buffer, i*chunk_size, (i+1)*chunk_size);
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::cout << typeid(reader).name() << ": test wait()     time: " << std::chrono::duration_cast<Duration>(end - start).count() << std::endl;
  return success;
}

template<typename Reader>
bool wait_all_test(Reader& reader, std::vector<segment_t>& segments, std::vector<float>& expected, std::vector<float>& buffer) {
  auto start = std::chrono::high_resolution_clock::now();  
  reader.enqueue_reads(segments);
  reader.wait_all();
  auto end = std::chrono::high_resolution_clock::now();
  std::cout << typeid(reader).name() << ": test wait_all() time: " << std::chrono::duration_cast<Duration>(end - start).count() << std::endl;
  return vectors_match(expected, buffer);
}

template<typename Reader>
bool wait_any_test(Reader& reader, std::vector<segment_t>& segments, std::vector<float>& expected, std::vector<float>& buffer, size_t chunk_size) {
  bool success = true;
  auto start = std::chrono::high_resolution_clock::now();  
  reader.enqueue_reads(segments);
  for(size_t i=0; i<segments.size(); i++) {
    size_t id = reader.wait_any();
    success = success && vectors_match(expected, buffer, id*chunk_size, (id+1)*chunk_size);
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::cout << typeid(reader).name() << ": test wait_any() time: " << std::chrono::duration_cast<Duration>(end - start).count() << std::endl;
  return success;
}

int main(int argc, char** argv) {
  argparse::ArgumentParser program("statediff");
  program.add_argument("--help")
      .help("Print help message")
      .default_value(false)
      .implicit_value(true);
  
  program.add_argument("-v", "--verbose")
      .help("Verbose output")
      .default_value(false)
      .implicit_value(true);

  program.add_argument("--data-length")
      .help("Length of data in bytes")
      .default_value(size_t(1024*1024*1024))
      .scan<'u', size_t>();

  program.add_argument("-c", "--chunk-size")
      .help("Chunk size")
      .default_value(size_t(128))
      .scan<'u', size_t>();

  program.add_argument("-n", "--num-chunks")
      .help("Number of random chunks to read")
      .default_value(size_t(1024 * 64))
      .scan<'u', size_t>();

  program.add_argument("-r", "--reader")
      .help("Use new readers for I/O")
      .default_value(std::string("posix"))
      .choices("posix", "mmap", "io-uring", "old-mmap", "old-io-uring");

  try {
      program.parse_args(argc, argv);
  } catch (const std::runtime_error &err) {
      std::cerr << err.what() << std::endl;
      std::cout << program;
      return 1;
  }

  int result = 0;
  bool print_help = program.get<bool>("help");
  if(print_help) {
      std::cout << program;
      return 0;
  }
  bool verbose_out = program.get<bool>("verbose");
  std::string reader = program.get<std::string>("reader");
  size_t num_offsets = program.get<size_t>("num-chunks");
  size_t data_len = program.get<size_t>("data-length") / sizeof(float);
  size_t chunk_size = program.get<size_t>("chunk-size") / sizeof(float);

  if(verbose_out) {
    std::cout << "Reader: " << reader << std::endl;
    std::cout << "Data length: " << data_len << std::endl;
    std::cout << "Chunk size: " << chunk_size << std::endl;
    std::cout << "Num chunks: " << num_offsets << std::endl;
  }

  std::vector<float> data(data_len), expected(num_offsets*chunk_size), buffer(chunk_size * num_offsets, 0);
  std::vector<size_t> offsets(num_offsets);
  std::vector<segment_t> segments(num_offsets);

  //Data Gen
  std::mt19937 gen(0);
  std::uniform_real_distribution<float> dis(-1.0, 1.0);
  std::uniform_int_distribution<size_t> int_dis(0, data_len/chunk_size);
  for(size_t i=0; i<data_len; i++) {
    data[i] = dis(gen);
  }

  // Prepare segments
  for(size_t i=0; i<num_offsets; i++) {
    offsets[i] = int_dis(gen);
    for(size_t j=0; j<chunk_size; j++) {
      expected[i*chunk_size + j] = data[offsets[i] * chunk_size + j];
    }
    segment_t seg;
    seg.id = i;
    seg.buffer = (uint8_t*)(buffer.data()+chunk_size*i);
    seg.offset = offsets[i]*chunk_size*sizeof(float);
    seg.size = chunk_size*sizeof(float);
    segments[i] = seg;
  }

  std::string filename("test.dat");

  // Write test data
  std::fstream filestream(filename, std::ios::binary | std::ios::out | std::ios::trunc);
  filestream.write((char*)(data.data()), data.size()*sizeof(float));
  filestream.close();
  clear_pages(filename);

  if(reader.compare("posix") == 0) {
    clear_pages(filename);
    posix_io_reader_t posix_reader(filename);
    if(!wait_test(posix_reader, segments, expected, buffer, chunk_size))
      result |= 1;
    clear_pages(filename);
    if(!wait_all_test(posix_reader, segments, expected, buffer)) 
      result |= 2;
    clear_pages(filename);
    if(!wait_any_test(posix_reader, segments, expected, buffer, chunk_size))
      result |= 4;
    clear_pages(filename);
  } else if(reader.compare("mmap") == 0) {
    clear_pages(filename);
    mmap_io_reader_t mmap_reader(filename);
    if(!wait_test(mmap_reader, segments, expected, buffer, chunk_size))
      result |= 1;
    clear_pages(filename);
    if(!wait_all_test(mmap_reader, segments, expected, buffer)) 
      result |= 2;
    clear_pages(filename);
    if(!wait_any_test(mmap_reader, segments, expected, buffer, chunk_size))
      result |= 4;
    clear_pages(filename);
  } else if(reader.compare("io-uring") == 0) {
    clear_pages(filename);
    liburing_io_reader_t uring_reader(filename);
    if(!wait_test(uring_reader, segments, expected, buffer, chunk_size))
      result |= 1;
    clear_pages(filename);
    if(!wait_all_test(uring_reader, segments, expected, buffer)) 
      result |= 2;
    clear_pages(filename);
    if(!wait_any_test(uring_reader, segments, expected, buffer, chunk_size))
      result |= 4;
    clear_pages(filename);
  } else if(reader.compare("old-mmap") == 0) {
    clear_pages(filename);
    MMapStream<float> io_reader(expected.size(), filename, chunk_size, false, false, 1);
    auto start = std::chrono::high_resolution_clock::now();  
    io_reader.start_stream(offsets.data(), offsets.size(), chunk_size);
    auto slice = io_reader.next_slice();
    std::vector<float> data_read(slice, slice+io_reader.get_slice_len());
    auto end = std::chrono::high_resolution_clock::now();
    auto runtime = std::chrono::duration_cast<Duration>(end - start).count();
    std::cout << typeid(io_reader).name() << ": test wait_all() time: " << runtime << std::endl;
    if(vectors_match(expected, buffer)) {
      result |= 2;
    }
    clear_pages(filename);
  } else if(reader.compare("old-io-uring") == 0) {
    clear_pages(filename);
    liburing_reader_t<float> io_reader(filename, expected.size(), chunk_size, false, false, 1);
    auto start = std::chrono::high_resolution_clock::now();  
    io_reader.start_stream(offsets.data(), offsets.size(), chunk_size);
    auto slice = io_reader.next_slice();
    std::vector<float> data_read(slice, slice+io_reader.get_slice_len());
    auto end = std::chrono::high_resolution_clock::now();
    auto runtime = std::chrono::duration_cast<Duration>(end - start).count();
    std::cout << typeid(io_reader).name() << ": test wait_all() time: " << runtime << std::endl;
    if(vectors_match(expected, buffer)) {
      result |= 2;
    }
    clear_pages(filename);
  }
  std::remove("test.dat");

  if(result & 1)
    std::cerr << "Wait failed!\n";
  if(result & 2)
    std::cerr << "Wait all failed!\n";
  if(result & 4)
    std::cerr << "Wait any failed!\n";
  return result;
}
