# state-diff
Compute differences between immutable data states


## Overview

state-diff is an innovative method to compute differences between immutable data states, offering scalable capture and comparison of intermediate multi-run results. It follows three key design principles: 

* GPU-optimized hashing technique for groups of floating point values organized into chunks that need to match within a given error bound; 
* Hierarchic organization of hashes using GPU-optimized data structures (Merkle trees) to accelerate the comparison of identical contiguous regions; 
* Multi-level I/O pipelining for data transfers and overlapping with GPU computations to maximize parallelization and enable scalability  

By capitalizing on intermediate checkpoints and hash-based techniques with user-defined error bounds, state-diff identifies divergences between data states early in application execution.

For detailed description about design principles, implementation, and performance evaluation against state-of-the-art data states comparison approaches, please refer our Middleware'24 paper.
> Nigel Tan, Kevin Assogba, Jay Ashworth, Befikir Bogale, Franck Cappello, M. Mustafa Rafique, Michela Taufer, and Bogdan Nicolae. "Towards Affordable Reproducibility Using Scalable Capture and Comparison of Intermediate Multi-Run Results". MIDDLEWARE'24: The 25th ACM/IFIP International Middleware Conference (Hong Kong, China, 2024).

## Building and installing state-diff

### Dependencies

* [CMake](https://cmake.org/)
* [Kokkos](https://github.com/kokkos/kokkos)
* [Liburing](https://github.com/axboe/liburing)
* [Cereal](https://github.com/USCiLab/cereal)

### Building with CMake (Recommended)

In the following example, we first install kokkos (following the [official repository](https://github.com/kokkos/kokkos?tab=readme-ov-file#building-kokkos)) and clone state-diff into the `$HOME` directory. You can adjust the installation instructions based on the location of your Kokkos installation and that of state-diff.

```
git clone https://github.com/DataStates/state-diff $HOME/state-diff
cd $HOME/state-diff
mkdir build

cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER=$HOME/kokkos/bin/nvcc_wrapper \
    -DCMAKE_CXX_FLAGS="-fPIC" \
    -DCMAKE_INSTALL_PREFIX=$HOME/state-diff/build \
    -DENABLE_TESTS=ON \
    -DKokkos_DIR=$HOME/kokkos/build/install/lib64/cmake/Kokkos \
    ..

make -j8
make install
```

### Automation with Spack

```
git clone https://github.com/DataStates/state-diff
cd state-diff
spack repo add ./spack-repo
spack install statediff
```

## Testing state-diff


This repository also contains a few text examples located in the `state-diff/test` directory illustrating metadata creation and data states comparison processes.

* `test_serialize.cpp`: Test the metadata creation process and its serialization to an archive using the [Cereal](https://github.com/USCiLab/cereal) library.
* `test_compare.cpp`: Test the metadata creation and comparison processes. It highlights the complete workflow of state-diff in an offline comparison scenario.

To run both test, execute the `make test` command from the build directory.

## Contacts

In case of questions and comments, please contact the authors on the paper.

## Release
For release details and restrictions, please read the [LICENSE](https://github.com/DataStates/state-diff/blob/main/LICENSE) file.