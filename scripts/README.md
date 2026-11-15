# state-diff scripts

Benchmark and evaluation executables built alongside state-diff.  All targets
are compiled when state-diff is built (they live in `scripts/CMakeLists.txt`).
After building, the binaries are in `state-diff/_build/scripts/`.

## Built executables

| Executable | Source | Description |
|------------|--------|-------------|
| `eval_statediff` | `eval_statediff.cpp` | End-to-end MPI benchmark: builds the Merkle trees of checkpoint files (create step), then compares the trees of two runs and validates the flagged chunks with direct I/O through scissor's Loader (compare step). |
| `eval_direct` | `eval_direct.cpp` | Direct comparison benchmark: compares two checkpoint files using scissor's staged I/O loader without the Merkle tree layer. Useful for isolating I/O performance from tree traversal cost. |
| `bench_load` | `bench_load_calls.cpp` | io_uring read microbenchmark: reads N chunks at the same offsets of two files through one `scissor::LiburingReader` and logs the load time and the throughput of both files. |
| `bench_load_2rdrs` | `bench_load_calls_2rdrs.cpp` | Like `bench_load` with one reader per file. |
| `bench_load_genoft` | `bench_load_calls_genoft.cpp` | Like `bench_load` with generated sparse offsets coalesced with a given gap. |
| `bench_compute` | `bench_compare_kernel.cpp` | Kokkos comparison-kernel microbenchmark: element-wise comparison of two buffers over a grid of chunk counts and chunk sizes, independent of I/O. |

## Benchmark workflow

### 1. End-to-end state-diff evaluation

```bash
cd state-diff/_build

# Two files of 1 M floats: test0.dat and test1.dat (10000 values perturbed)
src/tools/data_generator --data-len 1000000 -n 2 -t float \
    -e 0.001 --num-changes 10000 test --kokkos-num-threads=2

# Create step: writes test0.dat.0.compare-tree and test1.dat.1.compare-tree
scripts/eval_statediff -c 4096 --type float --error 0.0001 -r demo \
    --run0 test0.dat test1.dat --kokkos-num-threads=2

# Compare step: appends to demo.0.compare.csv
scripts/eval_statediff -c 4096 --type float --error 0.0001 -r demo \
    --run0 test0.dat.0.compare-tree --run1 test1.dat.1.compare-tree \
    --run0-full test0.dat --run1-full test1.dat --kokkos-num-threads=2
```

#### eval_statediff I/O flags

Each MPI rank creates one reader before the checkpoint loop
(`scissor::LiburingReader(qdepth, cqdepth)`, or `scissor::MmapReader` for
`--engine mmap`). It is shared by all clients and checkpoints of that rank.

| Flag | Default | Description |
|------|---------|-------------|
| `-q, --qdepth` | `32768` | io_uring submission queue depth (ring size). Written to the `Queue depth` CSV column. |
| `-z, --cqdepth` | `1024` | Completions reaped per batch by the reader. Written to the `CQueue depth` CSV column. |
| `--tuning` | `model` | `model`: scissor's Regressor picks offset gap and block size for `Kokkos::num_threads()` cores, `-g`/`-b` are ignored. `fixed`: use `-g` (gap in chunks, `0` = no coalescing) and `-b` (block = `b` x chunk size x threads) as given, single-threaded loader (profiling/breakdown runs). |
| `--engine` | `scissor` | `scissor`: coalesced reads over io_uring. `iouring`: one read per chunk over io_uring. `mmap`: one read per chunk over mmap (`MADV_RANDOM`, 4 copy threads). The `iouring` and `mmap` baselines use offset gap 0, `fixed` tuning and a 1 GiB block size unless `-b` is given. |

`Block size` and `Offset gap` in the compare CSV are the values actually used
(the Regressor's pick in `model` mode).

### 2. Direct I/O comparison (no Merkle tree)

```bash
scripts/eval_direct --run0 test0.dat --run1 test1.dat \
    --type float --error 0.0001 -r direct
```

### 3. Read microbenchmarks

```bash
# <file0> <file1> <chunk size> <number of chunks> <log file> [--rand] [--interleaved]
scripts/bench_load test0.dat test1.dat 4096 256 read.csv --rand --interleaved

# <file0> <file1> <chunk size> <number of chunks> [--rand]
scripts/bench_load_2rdrs test0.dat test1.dat 4096 256

# <file0> <file1> <chunk size> <number of chunks> <gap>
scripts/bench_load_genoft test0.dat test1.dat 4096 256 4
```

### 4. Compute kernel microbenchmark

```bash
# Appends the timings to compute_benchmark.csv
scripts/bench_compute --kokkos-num-threads=8
```

## Logging and environment variables

All executables inherit the shared JSONL logger.  Set `SCISSOR_LOG_LEVEL=PERF`
to capture detailed per-batch timing emitted by scissor's Loader.

```bash
SCISSOR_LOG_LEVEL=PERF SCISSOR_LOG_FILE=/tmp/eval_%r.jsonl \
    scripts/eval_direct --run0 test0.dat --run1 test1.dat --type float
```

Useful scissor tuning variables:

| Variable | Description |
|----------|-------------|
| `SCISSOR_MAX_OPEN_FDS` | FD LRU cache size (default: 64) |
| `SCISSOR_MAX_STAGING_MB` | Staging memory cap in MiB (default: 4096) |
| `SCISSOR_MODEL_CONFIG` | Path to custom `models.conf` for regressor |
