# state-diff
Compute differences between immutable data states


## Overview

state-diff is an innovative method to compute differences between immutable data states, offering scalable capture and comparison of intermediate multi-run results. It follows three key design principles: 

* GPU-optimized hashing technique for groups of floating point values organized into chunks that need to match within a given error bound.
* Hierarchic organization of hashes using GPU-optimized data structures (Merkle trees) to accelerate the comparison of identical contiguous regions.
* Multi-level I/O pipelining for data transfers and overlapping with GPU computations to maximize parallelization and enable scalability  

By capitalizing on intermediate checkpoints and hash-based techniques with user-defined error bounds, state-diff identifies divergences between data states early in application execution.

For a detailed description of the design principles, implementation, and performance evaluation, please refer to our Middleware'24 paper.
> Nigel Tan, Kevin Assogba, Jay Ashworth, Befikir Bogale, Franck Cappello, M. Mustafa Rafique, Michela Taufer, and Bogdan Nicolae. "Towards Affordable Reproducibility Using Scalable Capture and Comparison of Intermediate Multi-Run Results". MIDDLEWARE'24: The 25th ACM/IFIP International Middleware Conference (Hong Kong, China, 2024).

## Dependencies

| Dependency | Notes |
|------------|-------|
| [CMake](https://cmake.org/) ≥ 3.14 | Build system |
| [Kokkos](https://github.com/kokkos/kokkos) | Parallel compute backend (CPU or GPU) |
| [OpenSSL](https://www.openssl.org/) | SHA-256 hashing |
| [scissor-io](https://github.com/DataStates/scissor-io) (tag `v0.0.1`) | I/O readers (liburing, POSIX, mmap), staged Loader, gap/block-size Regressor and logger. Links liburing ≥ 2.6. Found via `find_package(scissor 0.0.1)`, otherwise fetched automatically |

## Building and installing state-diff

### Option A: Automated build (recommended)

The included `build.py` script builds scissor-io and state-diff in a single
command:

```bash
# Build everything from source (scissor-io expected at ../scissor-io)
python3 state-diff/build.py \
    --kokkos-dir /path/to/kokkos/lib/cmake/Kokkos

# Use a pre-installed scissor-io
python3 state-diff/build.py \
    --scissor-prefix /path/to/scissor/install \
    --kokkos-dir /path/to/kokkos/lib/cmake/Kokkos

# Production install
python3 state-diff/build.py \
    --prefix /opt/statediff \
    --scissor-prefix /opt/scissor \
    --kokkos-dir /opt/kokkos/lib/cmake/Kokkos \
    --build-type RelWithDebInfo
```

Run `python3 state-diff/build.py --help` for the full list of options.

### Option B: Manual CMake build

Build and install scissor-io first (plain CMake, see scissor-io's README), then:

```bash
cmake -S state-diff -B state-diff/_build \
      -DCMAKE_PREFIX_PATH=/path/to/scissor/install \
      -DKokkos_DIR=/path/to/kokkos/lib/cmake/Kokkos \
      -DCMAKE_INSTALL_PREFIX=/path/to/statediff/install
cmake --build state-diff/_build -j$(nproc)
cmake --install state-diff/_build
```

### Logging level

The compiled-in minimum log level is inherited from scissor-io
(`SCISSOR_MIN_LOG_LEVEL`, exported by the `scissor::scissor` target. Set it when
building scissor-io).

## Unit testing

Tests are located in `state-diff/test/`.  Run the full suite from the build
directory:

```bash
cd state-diff/_build
ctest --output-on-failure
```

| CTest name | Source file | Description |
|------------|-------------|-------------|
| `test_serialize` | `test_serialize.cpp` | Builds a Merkle tree and round-trips it through Cereal serialization. |
| `test_create_stream` | `test_create_stream.cpp` | Builds a Merkle tree from a file stream (no in-memory data copy). |
| `test_compare_0` | `test_compare.cpp 0` | Self-comparison: compares a dataset against itself, expects 0 mismatches. |
| `test_compare_1` | `test_compare.cpp 1` | Within-tolerance comparison: perturbed data within error bound, expects 0 mismatches. |
| `test_compare_2` | `test_compare.cpp 2` | Out-of-tolerance comparison: all elements perturbed beyond error bound, expects `data_len` mismatches. |

**Note:** `chunk_size` must be a multiple of 4096 to satisfy scissor's O_DIRECT
alignment requirement.  The tests use `chunk_size = 4096`.

## Command-line tool (statediff-cli)

After building, the CLI is at `state-diff/_build/src/cli/statediff-cli`.

```
Usage: statediff run0 run1 [options]

Positional arguments:
  run0        First file to compare
  run1        Second file to compare

Options:
  -t, --type  <byte|float|double>   Data type (required)
  -e, --error <double>              Error tolerance (default: 0.0)
  -c, --chunk_size <size_t>         Chunk size in bytes; must be multiple of 4096 (default: 4096)
  -b, --buffer-len <size_t>         I/O buffer length (default: 1048576)
  -s, --start-level <size_t>        Merkle tree start level (default: 13)
  -a, --approx-hash                 Use approximate (fuzzy) hashing
  -n, --num-threads <size_t>        Reader threads (default: 1)
      --verbose                     Print parameter summary and timing
      --help                        Print help message
```

**Example: generate test data and compare**

```bash
cd state-diff/_build

# Generate two files of 1 M floats: test0.dat (base) and test1.dat
# (10000 values perturbed by 1e-3)
src/tools/data_generator --data-len 1000000 -n 2 -t float \
    -e 0.001 --num-changes 10000 test --kokkos-num-threads=2

# Compare the files
src/cli/statediff-cli test0.dat test1.dat \
    --error 0.0001 --type float -c 4096 -s 1 \
    --kokkos-num-threads=2
```

Exit code is 0 whether or not mismatches are found. The result is printed to
stdout as either `SUCCESS::Files ... are within error tolerance.` or
`FAILURE::Files ... are NOT within error tolerance. Found N changes.`

## Logging

state-diff uses the `scissor/log.hpp` structured JSONL logger provided by
scissor-io.  scissor and state-diff write to the same output destination and
are distinguished by the `comp` field in each JSON record (`scissor::<func>`
or `statediff::<func>`).

Runtime environment variables:

| Variable | Description |
|----------|-------------|
| `SCISSOR_LOG_LEVEL` | Runtime log level: `DEBUG`, `PERF`, `INFO`, `WARN`, `ERROR`, `FATAL` |
| `SCISSOR_LOG_FILE` | Log output file. Use `%r` for MPI rank |
| `SCISSOR_LOG_RANK0_ONLY` | If set, ranks > 0 suppress below `WARN` |
| `SCISSOR_MAX_OPEN_FDS` | FD LRU cache size used by scissor's Loader (default: 64) |
| `SCISSOR_MAX_STAGING_MB` | Staging memory cap in MiB for scissor's Loader (default: 4096) |
| `SCISSOR_MODEL_CONFIG` | Path to a custom scissor `models.conf` |

## Contacts

In case of questions and comments, please contact the authors on the paper.

**University of Tennessee Knoxville**: Nigel Tan (nphtan2@gmail.com), Walter J. Ashworth, Befikir Bogale, Michela Taufer (taufer@gmail.com)

**Rochester Institute of Technology**: Kevin Assogba and M. Mustafa Rafique

**Argonne National Laboratory**: Bogdan Nicolae (bnicolae@anl.gov) and Franck Cappello

## Citing
Nigel Tan, Kevin Assogba, Walter J. Ashworth, Befikir Bogale, Franck Cappello, M. Mustafa Rafique, Michela Taufer, and Bogdan Nicolae. 2024. Towards Affordable Reproducibility Using Scalable Capture and Comparison of Intermediate Multi-Run Results. In Proceedings of the 25th International Middleware Conference (MIDDLEWARE '24). Association for Computing Machinery, New York, NY, USA, 392-403. https://doi.org/10.1145/3652892.3700780

## Acknowledgements
This material is based upon work supported by: the U.S. Department of Energy (DOE), Office of Science, Office of Advanced Scientific Computing Research, under Contract DE-AC02-06CH11357; the National Science Foundation under Grants #1900888, #1900765, #2223704, #2331152, #2411386, #2411387, #2106635.

## Copyright and License
For release details and restrictions, please read the [LICENSE](https://github.com/DataStates/state-diff/blob/main/LICENSE) file.
