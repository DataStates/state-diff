#!/bin/bash
# Runs bench_compute REPEAT times per thread count. Each run appends the
# full grid (work items x chunk size) to compute_benchmark.csv.
set -x
REPEAT=3
THREADS=(8 16)
LOGFILE=compute_benchmark.csv

for thread in "${THREADS[@]}"; do
    for r in $(seq 1 "$REPEAT"); do
        echo "Running: ../build/scripts/bench_compute $LOGFILE --kokkos-num-threads=$thread (run $r)"
        ./../build/scripts/bench_compute "$LOGFILE" --kokkos-num-threads="$thread"
    done
done
set +x
