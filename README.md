# state-diff
Library for computing differences between immutable data states. 

# Installation
## Dependencies
* [Kokkos](https://github.com/kokkos/kokkos)
* [argparse](https://github.com/p-ranav/argparse)
* [ADIOS 2] (https://adios2.readthedocs.io/en/latest) (Optional)
* CMake
* io\_uring

## Build ##
```
mkdir build
cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=g++ \
  -DCMAKE_INSTALL_PREFIX=$HOME/state-diff/build/install \
  -DKokkos_DIR=$KOKKOS_DIR/install/lib/cmake/Kokkos \
  ..
```
Note that the exact install directory and Kokkos directory will vary depending on the system and where you install Kokkos.

## Running and Tests ##

# Authors and Contacts
**University of Tennessee Knoxville**: Nigel Tan (nphtan2@gmail.com), Walter J. Ashworth, Befikir Bogale, Michela Taufer (taufer@gmail.com)

**Rochester Institute of Technology**: Kevin Assogba and M. Mustafa Rafique

**Argonne National Laboratory**: Bogdan Nicolae (bnicolae@anl.gov) and Franck Cappello

# Citing
Nigel Tan, Kevin Assogba, Walter J. Ashworth, Befikir Bogale, Franck Cappello, M. Mustafa Rafique, Michela Taufer, and Bogdan Nicolae. 2024. Towards Affordable Reproducibility Using Scalable Capture and Comparison of Intermediate Multi-Run Results. In Proceedings of the 25th International Middleware Conference (MIDDLEWARE '24). Association for Computing Machinery, New York, NY, USA, 392–403. https://doi.org/10.1145/3652892.3700780

# Copyright and License
See LICENSE for more details.

# Acknowledgements
This material is based upon work supported by: the U.S. Department of Energy (DOE), Office of Science, Office of Advanced Scientific Computing Research, under Contract DE-AC02-06CH11357; the National Science Foundation under Grants #1900888, #1900765, #2223704, #2331152, #2411386, #2411387, #2106635.
