
## Repository structure

```
state-diff
  - src
   |- common (tree_utils, hashing_utils, io_utils, comparison_utils)
   |- lib (client_implementation)
   |- modules (all modules)
  - include
   |- state-diff.hpp
  - test (unit tests)
  - examples (How to use state-diff)
  - docs (all markdowns)
  - artifacts (contains the main source file for our proposed optimized direct comparison.)
```

## APIs

```
namespace StateDiff
{
  class client 
  {
    typedef std::function<void (std::ostream &)> serializer_t;
    typedef std::function<bool (std::istream &)> deserializer_t;

    struct stats // contains the results (#of differences, etc.)

    client()
    ~client()
    init(std:string cfg) // cfg is config file containing parameters (chunk size, buffer len, etc.)

    Tree* create(void* data_ptr_h, size_t size)
    void createSave(void* data_ptr_h, size_t size) // use our serializer to write to file
    void createSave(void* data_ptr_h, size_t size, serializer_t ser) // use the serializer to write to file

    Template<typename T, U> // T and U can either be Tree* (GPU ptr to tree) or std:string (path to tree file)
    void compare(T curr_tree, U prev_tree) // uses our deserializer and prints whether app is reproducible or not
    void compare(T curr_tree, U prev_tree, deserializer_t deser) 

    void saveStats(std::string prefix) // writes the content of stats to prefix.json
    void printStats() // prints stats to CLI
  }

}
```

## New API

```
namespace state_diff {
    
template <typename T> class random_input_reader_t {
public:
    random_input_reader_t(const std::string &name);
    std::vector<T> read_offsets(std::vector<size_t> offsets);
};
    
template <typename T> class client_t {
    static const size_t CHUNK_SIZE = 4096;
    
    tree_t tree;
public:
    client_t(random_input_reader_t &reader, size_t chunk_size = CHUNK_SIZE);
    ~client_t();
    
    create(const std::vector<T> &data);
    // use ar & basic_data_structure[i] for all i in tree_t - code separately in the CPP
    template<class Archive> void serialize(Archive &ar);
    bool compare_with(const client_t &prev);
};
}
```

## Modules

Each child module inherits from the base module.

```
* tree_module
  |- merkle_tree
  |- others

* hashing_module
  |- rounding_hash
  |- fuzzy_hash
  |- others

* io_module
  |- posix_module
  |- mmap_module
  |- liburing_module
  |- adios_module
  |- others
```

