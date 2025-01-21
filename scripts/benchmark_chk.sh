#!/bin/bash

BUILD_DIR="$HOME/research/recup/veloc/apps/state-diff/build"
BUILD_DIR_CPU="$HOME/research/recup/veloc/apps/state-diff/buildcpu"
DATA_DIR="/lus/eagle/projects/RECUP/kassogba/veloc-ckpt/haac/sc-experiments"
KB=1024
MB=$((1024 * 1024))
GB=$((1024 * $MB))
data_size=$((1 * $GB))
outname="$DATA_DIR/test"
num_runs=3
nthreads=32

chunk_size=(4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768 655361 131072)

for chunk in "${chunk_size[@]}"
    do
echo "==============================================================================="
echo " Generate data "
echo "==============================================================================="
$BUILD_DIR/src/tools/data_generator --data-len $(($chunk * $KB)) -n 1 -e 0.01 --num-changes 0 $outname \
  --kokkos-num-threads=2 --kokkos-map-device-id-by=mpi_rank

for test_id in $(seq 1 $num_runs)
do
echo "==============================================================================="
echo " Benchmarking the tree creation time per chunk size on GPU "
echo "==============================================================================="
/home/keveltun/install/vmtouch/bin/vmtouch -ve $DATA_DIR/test0.dat
$BUILD_DIR/scripts/benchmark_create $DATA_DIR/test0.dat $(($chunk * $KB))

echo "==============================================================================="
echo " Benchmarking the tree creation time per chunk size on CPU "
echo "==============================================================================="
/home/keveltun/install/vmtouch/bin/vmtouch -ve $DATA_DIR/test0.dat
$BUILD_DIR_CPU/scripts/benchmark_create $DATA_DIR/test0.dat $(($chunk * $KB))

done
rm $DATA_DIR/test0.dat
done
