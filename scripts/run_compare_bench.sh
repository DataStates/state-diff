#!/bin/bash
set -x
REPEAT=3
DSIZE=(4)
THREADS=(1 4 16) #(1 4 16)

for i in "${!DSIZE[@]}"; do
    ds=${DSIZE[$i]}
    filename="compute_time_benchmark_${ds}_sm.csv"

    for thread in "${THREADS[@]}"; do
        for r in $(seq 1 "$REPEAT"); do
            echo "Running: ../build/scripts/bench_compute $ds $filename --kokkos-num-threads=$thread (run $r)"       
            ./../build/scripts/bench_compute "$ds" "$filename" --kokkos-num-threads="$thread"
        done
    done
done
set +x
