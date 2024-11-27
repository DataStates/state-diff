#include <random>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <typeinfo>
#include <chrono>
#include <algorithm>
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
  int result = 0;
  size_t num_offsets = 1024*64;
  size_t data_len = 1024*1024*1024 / sizeof(float);
  size_t chunk_size = 128 / sizeof(float);

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
    // seg.id = i;
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

  posix_io_reader_t posix_reader(filename);
  if(!wait_test(posix_reader, segments, expected, buffer, chunk_size))
    result |= 1;
  if(!wait_all_test(posix_reader, segments, expected, buffer)) 
    result |= 2;
  if(!wait_any_test(posix_reader, segments, expected, buffer, chunk_size))
    result |= 4;
  buffer.assign(buffer.size(), 0.0f);

  mmap_io_reader_t mmap_reader(filename);
  if(!wait_test(mmap_reader, segments, expected, buffer, chunk_size))
    result |= 8;
  if(!wait_all_test(mmap_reader, segments, expected, buffer)) 
    result |= 16;
  if(!wait_any_test(mmap_reader, segments, expected, buffer, chunk_size))
    result |= 32;
  buffer.assign(buffer.size(), 0.0f);

  liburing_io_reader_t uring_reader(filename);
  if(!wait_test(uring_reader, segments, expected, buffer, chunk_size))
    result |= 64;
  if(!wait_all_test(uring_reader, segments, expected, buffer)) 
    result |= 128;
  if(!wait_any_test(uring_reader, segments, expected, buffer, chunk_size))
    result |= 256;
  buffer.assign(buffer.size(), 0.0f);

  std::remove("test.dat");

  if(result & 1)
    std::cerr << "Posix reader failed!\n";
  if(result & 2)
    std::cerr << "MMap reader failed!\n";
  if(result & 4)
    std::cerr << "IO Uring reader failed!\n";

  return result;
}
