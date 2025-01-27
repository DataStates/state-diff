#!/bin/bash

BUILD_DIR="$HOME/research/recup/veloc/apps/state-diff/buildcpu"
DATA_DIR="/lus/eagle/projects/RECUP/kassogba/veloc-ckpt/haac/sc-experiments/4gpus/np796-500mil/run1"
num_runs=1

echo "==============================================================================="
echo " Benchmarking Liburing loading time to CPU "
echo "==============================================================================="
chunk_size=(32 64 128 256 512 1024 2048 4096 8192 16384 32768 65536 131072 262144 524288 1048576 2097152 4194304 8388608 16777216 33554432)

worksize=${#chunk_size[@]}

for test_id in $(seq 1 $num_runs)
do
    for((i=worksize-1; i>=0; i--))
    do
        /home/keveltun/install/vmtouch/bin/vmtouch -ve $DATA_DIR/m000p.mpirestart-combined-0-10.dat
        $BUILD_DIR/scripts/benchmark_liburing $DATA_DIR/m000p.mpirestart-combined-0-10.dat ${chunk_size[i]}
	
	sleep 5
    done
done
