# state-diff scripts

Benchmark and evaluation executables built alongside state-diff.  All targets
are compiled when state-diff is built (they live in `scripts/CMakeLists.txt`).
After building, the binaries are in `state-diff/_build/scripts/`.

## Built executables

| Executable | Source | Description |
|------------|--------|-------------|
| `eval_statediff` | `eval_statediff.cpp` | End-to-end MPI benchmark: generates synthetic checkpoints, builds Merkle trees, and compares them using the full state-diff pipeline. |
| `eval_direct` | `eval_direct.cpp` | Direct comparison benchmark: compares two checkpoint files using scissor's staged I/O loader without the Merkle tree layer. Useful for isolating I/O performance from tree traversal cost. |
| `bench_load` | `bench_load_calls.cpp` | Microbenchmark for scissor's `Loader`: measures batched load throughput for a single reader across varying chunk sizes and batch sizes. |
| `bench_load_2rdrs` | `bench_load_calls_2rdrs.cpp` | Like `bench_load` but drives two concurrent readers (e.g. liburing + posix) to compare backend throughput head-to-head. |
| `bench_load_genoft` | `bench_load_calls_genoft.cpp` | Like `bench_load` but generates random sparse offset patterns to stress the aggregator's coalescing logic at different contiguity gap values. |
| `bench_compute` | `bench_compare_kernel.cpp` | Microbenchmark for the Kokkos comparison kernel: measures hash comparison throughput on the CPU independent of I/O. |

## Benchmark workflow

### 1. End-to-end state-diff evaluation

```bash
cd state-diff/_build

# Generate two 1 M-float checkpoints with 1% of values perturbed beyond 1e-4
scripts/eval_statediff \
    --data-len 1000000 -n 2 \
    -e 0.0001 --num-changes 10000 \
    --type float reproducibility_test \
    --kokkos-num-threads=2

# Compare using statediff-cli
src/cli/statediff-cli \
    reproducibility_test0.dat reproducibility_test1.dat \
    --error 0.0001 --type float -c 4096 -s 1 \
    --kokkos-num-threads=2
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
scripts/eval_direct \
    --file0 reproducibility_test0.dat \
    --file1 reproducibility_test1.dat \
    --type float -c 4096
```

### 3. Loader microbenchmark

```bash
# Single reader
scripts/bench_load --file reproducibility_test0.dat -c 4096

# Two readers
scripts/bench_load_2rdrs \
    --file0 reproducibility_test0.dat \
    --file1 reproducibility_test1.dat \
    -c 4096

# Sparse offsets with varying gaps
scripts/bench_load_genoft --file reproducibility_test0.dat -c 4096 --gap 4
```

### 4. Compute kernel microbenchmark

```bash
scripts/bench_compute --chunks 4096 --chunk-size 4096
```

## Logging and environment variables

All executables inherit the shared JSONL logger.  Set `SCISSOR_LOG_LEVEL=PERF`
to capture detailed per-batch timing emitted by scissor's Loader.

```bash
SCISSOR_LOG_LEVEL=PERF SCISSOR_LOG_FILE=/tmp/bench_%r.jsonl \
    scripts/bench_load --file reproducibility_test0.dat -c 4096
```

Useful scissor tuning variables:

| Variable | Description |
|----------|-------------|
| `SCISSOR_MAX_OPEN_FDS` | FD LRU cache size (default: 64) |
| `SCISSOR_MAX_STAGING_MB` | Staging memory cap in MiB (default: 4096) |
| `SCISSOR_MODEL_CONFIG` | Path to custom `models.conf` for regressor |
