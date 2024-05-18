#include "kokkos_merkle_tree.hpp"
#include "compare_tree_approach.hpp"

#define __ASSERT

CompareTreeDeduplicator::CompareTreeDeduplicator() {}

CompareTreeDeduplicator::CompareTreeDeduplicator(uint32_t bytes_per_chunk) {
  curr_tree = &tree2;
  prev_tree = &tree1;
  chunk_size = bytes_per_chunk;
  current_id = 0;
  start_level = 30;
  num_comparisons = Kokkos::View<uint64_t[1]>("Num comparisons");
  num_hash_comp = Kokkos::View<uint64_t[1]>("Num hash comparisons");
  num_changed = Kokkos::View<uint64_t[1]>("Num changed");
}

CompareTreeDeduplicator::CompareTreeDeduplicator(uint32_t bytes_per_chunk, uint32_t start, bool fuzzy, float error, const char dtype) {
  curr_tree = &tree2;
  prev_tree = &tree1;
  chunk_size = bytes_per_chunk;
  current_id = 0;
  start_level = start;
  fuzzyhash = fuzzy;
  errorValue = error;
  dataType = dtype;
  num_comparisons = Kokkos::View<uint64_t[1]>("Num comparisons");
  num_hash_comp = Kokkos::View<uint64_t[1]>("Num hash comparisons");
  num_changed = Kokkos::View<uint64_t[1]>("Num changed");
}

void 
CompareTreeDeduplicator::setup(const size_t data_size) {
  DEBUG_PRINT("Begin setup\n");
  // ==========================================================================================
  // Deduplicate data
  // ==========================================================================================
  std::string setup_region_name = std::string("Deduplication chkpt ") + 
                                  std::to_string(current_id) + std::string(": Setup");
  Kokkos::Profiling::pushRegion(setup_region_name.c_str());

  // Set important values
  data_len = data_size;
  num_chunks = data_len/chunk_size;
  if(static_cast<uint64_t>(num_chunks)*static_cast<uint64_t>(chunk_size) < data_len)
    num_chunks += 1;
  num_nodes = 2*num_chunks-1;

  DEBUG_PRINT("Setup constants\n");

  // Allocate or resize necessary variables for each approach
  if(prev_tree->tree_d.size() == 0) {
//    uint32_t hashes_per_node = fuzzyhash ? 2:1;
    uint32_t hashes_per_node = 1;
    *prev_tree = MerkleTree(num_chunks, hashes_per_node);
    *curr_tree = MerkleTree(num_chunks, hashes_per_node);
  }
  changed_chunks = Kokkos::Bitset<>(num_chunks);
  curr_tree->dual_hash_d.clear();
  prev_tree->dual_hash_d.clear();

  DEBUG_PRINT("Cleared vectors and bitsets\n");

  std::string resize_tree_label = std::string("Deduplication chkpt ") + 
                                  std::to_string(current_id) + 
                                  std::string(": Setup: Resize Tree");
  Kokkos::Profiling::pushRegion(resize_tree_label.c_str());
  if((*prev_tree).tree_d.size() < num_nodes) {
    Kokkos::resize((*prev_tree).tree_d, num_nodes);
    Kokkos::resize((*prev_tree).tree_h, num_nodes);
    prev_tree->dual_hash_d = Dedupe::Bitset<Kokkos::DefaultExecutionSpace>(num_nodes);
    prev_tree->dual_hash_h = Dedupe::Bitset<Kokkos::DefaultHostExecutionSpace>(num_nodes);  
  }
  DEBUG_PRINT("Resized previous tree\n");
  if((*curr_tree).tree_d.size() < num_nodes) {
    Kokkos::resize((*curr_tree).tree_d, num_nodes);
    Kokkos::resize((*curr_tree).tree_h, num_nodes);
    curr_tree->dual_hash_d = Dedupe::Bitset<Kokkos::DefaultExecutionSpace>(num_nodes);
    curr_tree->dual_hash_h = Dedupe::Bitset<Kokkos::DefaultHostExecutionSpace>(num_nodes);  
  }
  Kokkos::resize(diff_hash_vec.vector_d, num_chunks);
  Kokkos::resize(diff_hash_vec.vector_h, num_chunks);
  Kokkos::Profiling::popRegion(); // Resize tree
  Kokkos::Profiling::popRegion(); // Setup
  DEBUG_PRINT("Finished setup\n");
}

void 
CompareTreeDeduplicator::setup(const size_t data_size,
               std::string& run0_file, std::string& run1_file) {
  file0 = run0_file;
  file1 = run1_file;
  setup(data_size);
}

void 
CompareTreeDeduplicator::create_tree(const uint8_t* data_ptr, const size_t data_size) {
  // ==============================================================================================
  // Setup 
  // ==============================================================================================
  // Get number of chunks and nodes
  STDOUT_PRINT("Chunk size: %u\n", chunk_size);
  STDOUT_PRINT("Num chunks: %u\n", num_chunks);
  STDOUT_PRINT("Num nodes: %u\n", num_nodes);
  DEBUG_PRINT("Baseline deduplication\n");

  // Grab references to current and previous tree
  MerkleTree& tree_curr = *curr_tree;

  // Setup markers for beginning and end of tree level
  uint32_t level_beg = 0, level_end = 0;
  while(level_beg < num_nodes) {
    level_beg = 2*level_beg + 1;
    level_end = 2*level_end + 2;
  }
  level_beg = (level_beg-1)/2;
  level_end = (level_end-2)/2;
  uint32_t left_leaf = level_beg;
  //uint32_t right_leaf = level_end;
  uint32_t last_lvl_beg = (1 <<  start_level) - 1;
  //uint32_t last_lvl_end = (1 << (start_level + 1)) - 2;
  //DEBUG_PRINT("Leaf range [%u,%u]\n", left_leaf, right_leaf);
  //DEBUG_PRINT("Start level [%u,%u]\n", last_lvl_beg, last_lvl_end);

  // Temporary values to avoid capturing this object in the lambda
  auto nchunks        = num_chunks;
  auto nnodes         = num_nodes;
  auto chunksize      = chunk_size;
  auto dtype          = dataType;
  auto err_tol        = errorValue;
  bool use_fuzzy_hash = fuzzyhash && (comp_op != Equivalence);

  // ==============================================================================================
  // Construct tree
  // ==============================================================================================
  std::string diff_label = std::string("Diff ") + std::to_string(current_id) + std::string(": ");
  Kokkos::Profiling::pushRegion(diff_label + std::string("Construct Tree"));
  Kokkos::parallel_for(diff_label + std::string("Hash leaves"), Kokkos::RangePolicy<>(0,num_chunks),
  KOKKOS_LAMBDA(uint32_t idx) {
    // Calculate leaf node
    uint32_t leaf = left_leaf + idx;
    // Adjust leaf if not on the lowest level
    if(leaf >= nnodes) {
      const uint32_t diff = leaf - nnodes;
      leaf = ((nnodes-1)/2) + diff;
    }
    // Determine which chunk of data to hash
    uint32_t num_bytes = chunksize;
    uint64_t offset = static_cast<uint64_t>(idx)*static_cast<uint64_t>(chunksize);
    if(idx == nchunks-1) // Calculate how much data to hash
      num_bytes = data_size-offset;
    // Hash chunk
    if(use_fuzzy_hash) {
      tree_curr.calc_leaf_fuzzy_hash(data_ptr+offset, num_bytes, err_tol, dtype, leaf);
    } else {
      tree_curr.calc_leaf_hash(data_ptr+offset, num_bytes, leaf);
    }
  });

  // Build up tree level by level
  // Iterate through each level of tree and build First occurrence trees
  // This should stop building at last_lvl_beg
  while(level_beg >= last_lvl_beg) { // Intentional unsigned integer underflow
    std::string tree_constr_label = diff_label + std::string("Construct level [") 
                                               + std::to_string(level_beg) + std::string(",") 
                                               + std::to_string(level_end) + std::string("]");
    Kokkos::parallel_for(tree_constr_label, Kokkos::RangePolicy<>(level_beg, level_end+1), 
    KOKKOS_LAMBDA(const uint32_t node) {
      // Check if node is non leaf
      if(node < nchunks-1) {
        tree_curr.calc_hash(node);
      }
    });
    level_beg = (level_beg-1)/2;
    level_end = (level_end-2)/2;
  }
  Kokkos::Profiling::popRegion();
  return;
}


size_t
CompareTreeDeduplicator::compare_trees_phase1() {
  // ==============================================================================================
  // Compare Trees tree
  // ==============================================================================================

  std::string diff_label = std::string("Chkpt ") + std::to_string(current_id) + std::string(": ");
  Kokkos::Profiling::pushRegion(diff_label + std::string("Compare Trees"));

  Kokkos::Profiling::pushRegion(diff_label + std::string("Compare Trees setup"));
  // Grab references to current and previous tree
  MerkleTree& tree_prev = *prev_tree;
  MerkleTree& tree_curr = *curr_tree;

  // Setup markers for beginning and end of tree level
  uint32_t level_beg = 0;
  uint32_t level_end = 0;
  while(level_beg < num_nodes) {
    level_beg = 2*level_beg + 1;
    level_end = 2*level_end + 2;
  }
  level_beg = (level_beg-1)/2;
  level_end = (level_end-2)/2;
  uint32_t left_leaf = level_beg;
  uint32_t right_leaf = level_end;
  uint32_t last_lvl_beg = (1 <<  start_level) - 1;
  uint32_t last_lvl_end = (1 << (start_level + 1)) - 2;
  if(last_lvl_beg > left_leaf)
    last_lvl_beg = left_leaf;
  if(last_lvl_end > right_leaf)
    last_lvl_end = right_leaf;
  DEBUG_PRINT("Leaf range [%u,%u]\n", left_leaf, right_leaf);
  DEBUG_PRINT("Start level [%u,%u]\n", last_lvl_beg, last_lvl_end);
  
  diff_hash_vec.clear();
  Kokkos::deep_copy(num_comparisons, 0);
  Kokkos::deep_copy(num_hash_comp, 0);
  Kokkos::deep_copy(num_changed, 0);
  Kokkos::Experimental::ScatterView<uint64_t[1]> nhash_comp(num_hash_comp);
  Kokkos::Profiling::popRegion();
  
  // Fills up queue with nodes in the stop level or leavs in case of num levels < 12
  Kokkos::Profiling::pushRegion(diff_label + "Compare Trees with queue");
  level_beg = last_lvl_beg;
  level_end = last_lvl_end;
  Queue working_queue(num_chunks);
  auto fill_policy = Kokkos::RangePolicy<>(level_beg, level_end + 1);
  Kokkos::parallel_for("Fill up queue with every node in the stop_level", fill_policy,
  KOKKOS_LAMBDA(const uint32_t i) {
    working_queue.push(i);
  });
  // Temporary values to pass to the lambda
  auto& prev_dual_hash = tree_prev.dual_hash_d;
  auto& curr_dual_hash = tree_curr.dual_hash_d;
  auto& diff_hashes = diff_hash_vec;
  auto n_chunks = num_chunks;
  auto n_nodes = num_nodes;

  // Compare trees level by level
  while(working_queue.size() > 0){
    Kokkos::parallel_for("Process queue", Kokkos::RangePolicy<>(0,working_queue.size()), 
    KOKKOS_LAMBDA(uint32_t i){
      auto nhash_comp_access = nhash_comp.access();
      uint32_t node = working_queue.pop();
      bool identical = false;
      if(curr_dual_hash.test(node) && prev_dual_hash.test(node)) {
        identical = digests_same(tree_curr(node,0), tree_prev(node,0)) || 
                    digests_same(tree_curr(node,0), tree_prev(node,1)) ||
                    digests_same(tree_curr(node,1), tree_prev(node,0)) ||
                    digests_same(tree_curr(node,1), tree_prev(node,1));
        nhash_comp_access(0) += 4;
      } else if(curr_dual_hash.test(node)) {
        identical = digests_same(tree_curr(node,0), tree_prev(node,0)) || 
                    digests_same(tree_curr(node,1), tree_prev(node,0));
        nhash_comp_access(0) += 2;
      } else if(prev_dual_hash.test(node)) {
        identical = digests_same(tree_curr(node,0), tree_prev(node,0)) || 
                    digests_same(tree_curr(node,0), tree_prev(node,1));
        nhash_comp_access(0) += 2;
      } else {
        identical = digests_same(tree_curr(node), tree_prev(node));
        nhash_comp_access(0) += 1;
      }
      if(!identical) {
        if( (n_chunks-1 <= node) && (node < n_nodes) ) {
          if(node < left_leaf) { // Leaf is not on the last level
            size_t entry = (n_nodes-left_leaf) + (node - ((n_nodes-1)/2));
            assert(entry < n_chunks);
            diff_hashes.push(entry);
//printf("Tree: Block %zu (%zu) changed\n", node, (n_nodes-left_leaf) + (node - ((n_nodes-1)/2)));
          } else { // Leaf is on the last level
            assert(node-left_leaf < n_chunks);
            diff_hashes.push(node-left_leaf);
//printf("Tree: Block %zu (%zu) changed\n", node, node-left_leaf);
          }
        } else {
          uint32_t child_l = 2 * node + 1;
          uint32_t child_r = 2 * node + 2;
          if(child_l < n_nodes) {
            working_queue.push(child_l);
          }
          if(child_r < n_nodes) {
            working_queue.push(child_r);
          }
        }
      }
    });
    // printf("Ready to synchronize\n");
    Kokkos::fence();
    // printf("Synchronized\n");
  }
  Kokkos::Profiling::popRegion();
  Kokkos::Profiling::pushRegion(diff_label + std::string("Contribute hash comparison count"));
  Kokkos::Experimental::contribute(num_hash_comp, nhash_comp);
  Kokkos::Profiling::popRegion();
//printf("Diff hash capacity: %u\n", diff_hash_vec.capacity());
//Kokkos::parallel_for(num_chunks, KOKKOS_CLASS_LAMBDA(const uint32_t i) {
//  diff_hash_vec.push(static_cast<size_t>(i));
//});
//Kokkos::fence();
//printf("Diff hash vec pointer %p\n", diff_hash_vec.vector_d.data());
  Kokkos::Profiling::popRegion();
  return diff_hash_vec.size();
}

size_t
CompareTreeDeduplicator::compare_trees_phase2() {
  // ==============================================================================================
  // Validate first occurences with direct comparison
  // ==============================================================================================
  STDOUT_PRINT("Number of first occurrences (Leaves) - Phase One: %u\n", diff_hash_vec.size());
  std::string diff_label = std::string("Chkpt ") + std::to_string(current_id) + std::string(": ");
  Kokkos::Profiling::pushRegion(diff_label + std::string("Compare Trees direct comparison"));
  
  // Sort indices for better performance
  Kokkos::Profiling::pushRegion(diff_label + std::string("Compare Tree sort indices"));
  size_t num_diff_hash = static_cast<size_t>(diff_hash_vec.size());
  auto subview_bounds = Kokkos::make_pair((size_t)(0),num_diff_hash);
  auto diff_hash_subview = Kokkos::subview(diff_hash_vec.vector_d, subview_bounds);
  Kokkos::sort(diff_hash_vec.vector_d, 0, num_diff_hash);
  size_t blocksize = chunk_size;
  if(dataType == *"f") {
    blocksize /= sizeof(float);
  } else if(dataType == *"d") {
    blocksize /= sizeof(double);
  }
  Kokkos::Profiling::popRegion();

  uint64_t num_diff = 0;
  auto& num_changes = num_changed;
  auto& changed_blocks = changed_chunks;
  if (num_diff_hash > 0) {
    if(dataType == 'f') {
      Kokkos::Profiling::pushRegion(diff_label + std::string("Compare Tree start file streams"));
      changed_chunks.reset();

      size_t buffer_length = stream_buffer_len < num_diff_hash*blocksize ? stream_buffer_len : num_diff_hash*blocksize;
#ifdef IO_URING_STREAM
      IOUringStream<float> file_stream0(buffer_length, file0, true, false); 
      IOUringStream<float> file_stream1(buffer_length, file1, true, false); 
#else
      MMapStream<float> file_stream0(buffer_length, file0, true, false); 
      MMapStream<float> file_stream1(buffer_length, file1, true, false); 
#endif
      file_stream0.start_stream(diff_hash_vec.vector_d.data(), num_diff_hash, blocksize);
      file_stream1.start_stream(diff_hash_vec.vector_d.data(), num_diff_hash, blocksize);

      AbsoluteComp<float> abs_comp;
      size_t offset_idx = 0;
      double err_tol = errorValue;
      float *sliceA=NULL, *sliceB=NULL;
      size_t slice_len=0;
      size_t* offsets = diff_hash_vec.vector_d.data();
      size_t filesize = file_stream0.get_file_size();
      Kokkos::deep_copy(num_comparisons, 0);
      Kokkos::deep_copy(num_changed, 0);
      size_t num_iter = num_diff_hash/file_stream0.chunks_per_slice;
      if(num_iter * file_stream0.chunks_per_slice < num_diff_hash)
        num_iter += 1;
      Kokkos::Experimental::ScatterView<uint64_t[1]> num_comp(num_comparisons);
      Kokkos::Profiling::popRegion();
      for(size_t iter=0; iter<num_iter; iter++) {
        Kokkos::Profiling::pushRegion("Next slice");
        sliceA = file_stream0.next_slice();
        sliceB = file_stream1.next_slice();
        slice_len = file_stream0.get_slice_len();
        size_t slice_len_b = file_stream1.get_slice_len();
        assert(slice_len == slice_len_b);
        Kokkos::Profiling::popRegion();
        Kokkos::Profiling::pushRegion(diff_label + std::string("Compare Tree direct comparison"));
        Timer::time_point beg = Timer::now();
        uint64_t ndiff = 0;
        // Parallel comparison
        auto range_policy = Kokkos::RangePolicy<size_t>(0, slice_len);
        Kokkos::parallel_reduce("Count differences", range_policy, 
        KOKKOS_LAMBDA (const size_t idx, uint64_t& update) {
          auto ncomp_access = num_comp.access();
          size_t i = idx / blocksize; // Block
          size_t j = idx % blocksize; // Element in block
          assert(offset_idx+i < num_diff_hash);
          size_t data_idx = blocksize*sizeof(float)*offsets[offset_idx+i] + j*sizeof(float);
          assert(data_idx < filesize);
          if( (offset_idx+i < num_diff_hash) && (data_idx<filesize) ) {
            if(!abs_comp(sliceA[idx], sliceB[idx], err_tol)) {
              update += 1;
              changed_blocks.set(offsets[offset_idx+i]);
            }
            ncomp_access(0) += 1;
          }
        }, Kokkos::Sum<uint64_t>(ndiff));
        Kokkos::fence();

        num_diff += ndiff;
        offset_idx += slice_len/blocksize;
        if(slice_len % blocksize > 0)
          offset_idx += 1;
        Kokkos::Profiling::popRegion();
        Timer::time_point end = Timer::now();
        compare_timer += std::chrono::duration_cast<Duration>(end - beg).count();
      }
      Kokkos::Profiling::pushRegion(diff_label + std::string("Compare Tree finalize"));
      Kokkos::Experimental::contribute(num_comparisons, num_comp);
      io_timer = file_stream0.get_timer();
      if(file_stream1.get_timer() > io_timer) {
        io_timer = file_stream1.get_timer();
      }
      file_stream0.end_stream();
      file_stream1.end_stream();
      Kokkos::deep_copy(num_changes, num_diff);
      Kokkos::Profiling::popRegion();
    }
  }
  // ==============================================================================================
  // End of Validation
  // ==============================================================================================
  Kokkos::Profiling::popRegion();
  return num_diff;
}

/**
 * Serialize the current Merkle tree as well as needed metadata
 */
std::vector<uint8_t> 
CompareTreeDeduplicator::serialize() {
  Kokkos::Timer timer_total;
  Kokkos::Timer timer_section;

  // Copy tree to host
  timer_section.reset();
  Kokkos::deep_copy(curr_tree->tree_h, curr_tree->tree_d);
  Dedupe::deep_copy(curr_tree->dual_hash_h, curr_tree->dual_hash_d);

  HashDigest* tree_ptr = curr_tree->tree_h.data();
  uint64_t size = curr_tree->tree_h.extent(0) * curr_tree->tree_h.extent(1);
  uint32_t hashes_per_node = static_cast<uint32_t>(curr_tree->tree_h.extent(1));
  STDOUT_PRINT("Time for copying tree to host: %f seconds.\n", timer_section.seconds() );
  
  timer_section.reset();
  //preallocating the vector to simplify things
  // size of current id, chunk size, hashes/node, size + size of data in tree (16 times number of 
  // nodes as 1 digest is 16 bytes) + the size of dual_hash rounded up to the nearest byte. 
  std::vector<uint8_t> buffer(  sizeof(current_id) + sizeof(chunk_size) 
                              + sizeof(num_chunks) + sizeof(hashes_per_node) 
                              + (sizeof(HashDigest)*size) 
                              + sizeof(unsigned int)*((curr_tree->dual_hash_h.size()+31)/32));
  
  size_t offset = 0;
  STDOUT_PRINT("Time for preallocating vector: %f seconds.\n", timer_section.seconds() );
  
  //inserting the current id 
  timer_section.reset();
  memcpy(buffer.data() + offset, &current_id, sizeof(current_id));
  offset += sizeof(current_id);
  STDOUT_PRINT("Time for inserting the current id: %f seconds.\n", timer_section.seconds() );
                
  //inserting the chunk size
  timer_section.reset();
  memcpy(buffer.data() + offset, &chunk_size, sizeof(chunk_size));
  offset += sizeof(chunk_size);
  STDOUT_PRINT("Time for inserting the chunk size: %f seconds.\n", timer_section.seconds() );
  
  // inserting the number of chunks 
  timer_section.reset();
  memcpy(buffer.data() + offset, &num_chunks, sizeof(num_chunks));
  offset += sizeof(num_chunks);
  STDOUT_PRINT("Time for inserting the number of chunks: %f seconds.\n", timer_section.seconds() );
  
  //inserting the number of hashes per node 
  timer_section.reset();
  memcpy(buffer.data() + offset, &hashes_per_node, sizeof(hashes_per_node));
  offset += sizeof(hashes_per_node);
  STDOUT_PRINT("Time for inserting the number of hashes per node: %f seconds.\n", timer_section.seconds() );
  
  //inserting tree_h
  timer_section.reset();
  memcpy(buffer.data() + offset, tree_ptr, size*sizeof(HashDigest));
  offset += size*sizeof(HashDigest);
  STDOUT_PRINT("Time for inserting tree_h: %f seconds.\n", timer_section.seconds() );
  
  //inserting dual_hash_h
  timer_section.reset();
  memcpy(buffer.data() + offset, curr_tree->dual_hash_h.data(), sizeof(unsigned int)*((curr_tree->dual_hash_h.size()+31)/32));
  STDOUT_PRINT("Time for inserting dual_hash_h: %f seconds.\n", timer_section.seconds() );

  STDOUT_PRINT("Total time for serialize function: %f seconds.\n", timer_total.seconds() );

  return buffer;
}

/**
 * Deserialize the current Merkle tree as well as needed metadata
 */
uint64_t CompareTreeDeduplicator::deserialize(std::vector<uint8_t>& buffer) {

    size_t offset = 0;
    uint32_t t_id, t_chunksize;

    memcpy(&t_id, buffer.data() + offset, sizeof(t_id));
    offset += sizeof(t_id);
    if (current_id != t_id) {
        std::cerr << "deserialize_tree: Tree IDs do not match (" << current_id << " vs " << t_id << ").\n";
        return 0;
    }

    memcpy(&t_chunksize, buffer.data() + offset, sizeof(t_chunksize));
    offset += sizeof(t_chunksize);
    if (chunk_size != t_chunksize) {
        std::cerr << "deserialize_tree: Tree chunk sizes do not match (" << chunk_size << " vs " << t_chunksize << ").\n";
        return 0;
    }

    memcpy(&num_chunks, buffer.data() + offset, sizeof(num_chunks));
    num_nodes = 2 * num_chunks - 1;
    offset += sizeof(num_chunks);
    changed_chunks = Kokkos::Bitset<>(num_chunks);

    uint32_t hashes_per_node = 1;
    memcpy(&hashes_per_node, buffer.data() + offset, sizeof(hashes_per_node));
    offset += sizeof(hashes_per_node);

    if (prev_tree != nullptr && prev_tree->tree_h.extent(0) < num_nodes 
                             && prev_tree->tree_h.extent(1) < hashes_per_node) 
      *prev_tree = MerkleTree(num_chunks, hashes_per_node);
    if (curr_tree != nullptr && curr_tree->tree_h.extent(0) < num_nodes 
                             && curr_tree->tree_h.extent(1) < hashes_per_node) 
      *curr_tree = MerkleTree(num_chunks, hashes_per_node);

    using HashDigest2DView = Kokkos::View<HashDigest**, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    HashDigest* raw_ptr = reinterpret_cast<HashDigest*>(buffer.data() + offset);
    HashDigest2DView unmanaged_view(raw_ptr, num_nodes, hashes_per_node);
    offset += static_cast<size_t>(num_nodes)*static_cast<size_t>(hashes_per_node) * sizeof(HashDigest);
  
    memcpy(prev_tree->dual_hash_h.data(), buffer.data() + offset, sizeof(unsigned int)*((num_nodes+31)/32));
    offset += sizeof(unsigned int)*((prev_tree->dual_hash_h.size() + 31) / 32);

    Kokkos::deep_copy(prev_tree->tree_d, unmanaged_view);
    Dedupe::deep_copy(prev_tree->dual_hash_d, prev_tree->dual_hash_h);
    Kokkos::fence();

    return buffer.size();
}

uint64_t CompareTreeDeduplicator::deserialize(std::vector<uint8_t>& run0_buffer, std::vector<uint8_t>& run1_buffer) {

    Kokkos::Profiling::pushRegion("Deserialize: copy constants");
    size_t offset0 = 0, offset1 = 0;
    uint32_t t_id0, t_id1, t_chunksize0, t_chunksize1;

    memcpy(&t_id0, run0_buffer.data() + offset0, sizeof(t_id0));
    offset0 += sizeof(t_id0);
    memcpy(&t_id1, run1_buffer.data() + offset1, sizeof(t_id1));
    offset1 += sizeof(t_id1);
    if (current_id != t_id1) {
        std::cerr << "deserialize_tree: Tree IDs do not match (" << current_id << " vs " << t_id1 << ").\n";
        return 0;
    }

    memcpy(&t_chunksize0, run0_buffer.data() + offset0, sizeof(t_chunksize0));
    offset0 += sizeof(t_chunksize0);
    if (chunk_size != t_chunksize0) {
        std::cerr << "deserialize_tree: Tree chunk sizes do not match (" << chunk_size << " vs " << t_chunksize0 << ").\n";
        return 0;
    }
    memcpy(&t_chunksize1, run1_buffer.data() + offset1, sizeof(t_chunksize1));
    offset1 += sizeof(t_chunksize1);
    if (chunk_size != t_chunksize1) {
        std::cerr << "deserialize_tree: Tree chunk sizes do not match (" << chunk_size << " vs " << t_chunksize1 << ").\n";
        return 0;
    }

    memcpy(&num_chunks, run0_buffer.data() + offset0, sizeof(num_chunks));
    num_nodes = 2 * num_chunks - 1;
    offset0 += sizeof(num_chunks);
    memcpy(&num_chunks, run1_buffer.data() + offset1, sizeof(num_chunks));
    num_nodes = 2 * num_chunks - 1;
    offset1 += sizeof(num_chunks);
    changed_chunks = Kokkos::Bitset<>(num_chunks);

    uint32_t hashes_per_node0 = 1, hashes_per_node1 = 1;
    memcpy(&hashes_per_node0, run0_buffer.data() + offset0, sizeof(hashes_per_node0));
    offset0 += sizeof(hashes_per_node0);
    memcpy(&hashes_per_node1, run1_buffer.data() + offset1, sizeof(hashes_per_node1));
    offset1 += sizeof(hashes_per_node1);
    if(hashes_per_node0 || hashes_per_node1)
      fuzzyhash = true;
    
    Kokkos::Profiling::popRegion();
    Kokkos::Profiling::pushRegion("Deserialize: create/resize trees");

    if (prev_tree != nullptr && (prev_tree->tree_d.extent(0) != num_nodes 
                             || prev_tree->tree_d.extent(1) != hashes_per_node0)) 
      *prev_tree = MerkleTree(num_chunks, hashes_per_node0);
    if (curr_tree != nullptr && (curr_tree->tree_d.extent(0) != num_nodes 
                             || curr_tree->tree_d.extent(1) != hashes_per_node1)) 
      *curr_tree = MerkleTree(num_chunks, hashes_per_node1);

    Kokkos::Profiling::popRegion();
    Kokkos::Profiling::pushRegion("Deserialize: create unmanaged host views");

    using HashDigest2DView = Kokkos::View<HashDigest**, MerkleTree::tree_type::array_layout, Kokkos::DefaultHostExecutionSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    HashDigest* raw_ptr0 = reinterpret_cast<HashDigest*>(run0_buffer.data() + offset0);
    HashDigest2DView unmanaged_view0(raw_ptr0, num_nodes, hashes_per_node0);
    offset0 += static_cast<size_t>(num_nodes)*static_cast<size_t>(hashes_per_node0) * sizeof(HashDigest);
    HashDigest* raw_ptr1 = reinterpret_cast<HashDigest*>(run1_buffer.data() + offset1);
    HashDigest2DView unmanaged_view1(raw_ptr1, num_nodes, hashes_per_node1);
    offset1 += static_cast<size_t>(num_nodes)*static_cast<size_t>(hashes_per_node1) * sizeof(HashDigest);

    Kokkos::Profiling::popRegion();
    Kokkos::Profiling::pushRegion("Deserialize: copy bitset to host copy");

    memcpy(prev_tree->dual_hash_h.data(), run0_buffer.data() + offset0, sizeof(unsigned int)*((num_nodes+31)/32));
    offset0 += sizeof(unsigned int)*((prev_tree->dual_hash_h.size() + 31) / 32);
    memcpy(curr_tree->dual_hash_h.data(), run1_buffer.data() + offset1, sizeof(unsigned int)*((num_nodes+31)/32));
    offset1 += sizeof(unsigned int)*((curr_tree->dual_hash_h.size() + 31) / 32);

    Kokkos::Profiling::popRegion();
    Kokkos::Profiling::pushRegion("Deserialize: copy data to device");

    Kokkos::deep_copy(prev_tree->tree_d, unmanaged_view0);
    Dedupe::deep_copy(prev_tree->dual_hash_d, prev_tree->dual_hash_h);
    Kokkos::deep_copy(curr_tree->tree_d, unmanaged_view1);
    Dedupe::deep_copy(curr_tree->dual_hash_d, curr_tree->dual_hash_h);
    Kokkos::resize(diff_hash_vec.vector_d, num_chunks);
    Kokkos::resize(diff_hash_vec.vector_h, num_chunks);
    Kokkos::fence();

    Kokkos::Profiling::popRegion();

    return run1_buffer.size();
}

uint64_t CompareTreeDeduplicator::get_num_hash_comparisons() const {
  auto num_hash_comp_h = Kokkos::create_mirror_view(num_hash_comp);
  Kokkos::deep_copy(num_hash_comp_h, num_hash_comp);
  return num_hash_comp_h(0); 
}

uint64_t CompareTreeDeduplicator::get_num_comparisons() const {
  auto num_comparisons_h = Kokkos::create_mirror_view(num_comparisons);
  Kokkos::deep_copy(num_comparisons_h, num_comparisons);
  return num_comparisons_h(0); 
}

uint64_t CompareTreeDeduplicator::get_num_changes() const {
  auto num_changed_h = Kokkos::create_mirror_view(num_changed);
  Kokkos::deep_copy(num_changed_h, num_changed);
  return num_changed_h(0); 
}

double CompareTreeDeduplicator::get_io_time() const {
  return io_timer;
}

double CompareTreeDeduplicator::get_compare_time() const {
  return compare_timer;
}
