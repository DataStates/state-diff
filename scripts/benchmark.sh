#!/bin/bash

BUILD_DIR="$HOME/research/recup/veloc/apps/state-diff/buildcpu"
DATA_DIR="/lus/eagle/projects/RECUP/kassogba/veloc-ckpt/haac/sc-experiments"
MB=$((1024 * 1024))
GB=$((1024 * $MB))
data_size=$((1 * $GB))
outname="$DATA_DIR/test"
num_runs=3

echo "==============================================================================="
echo " Generate data "
echo "==============================================================================="
$BUILD_DIR/src/tools/data_generator --data-len $data_size -n 1 -e 0.01 --num-changes 0 $outname \
  --kokkos-num-threads=2 --kokkos-map-device-id-by=mpi_rank

echo "==============================================================================="
echo " Peak bandwidth verification "
echo "==============================================================================="
for test_id in $(seq 1 $num_runs)
do 
    /home/keveltun/install/vmtouch/bin/vmtouch -ve $DATA_DIR/test0.dat
    $BUILD_DIR/scripts/benchmark_thrpt $DATA_DIR/test0.dat

    # output two numbers (f2h, h2d). read the two numbers, add them to a list and sort to find max of both
done

echo "==============================================================================="
echo " Benchmarking the tree creation time per chunk size  "
echo "==============================================================================="
chunk_size=(1 2 4 8 16 32 64 128 256 512)
#chunk_size=(512)
for test_id in $(seq 1 $num_runs)
do 
    for chunk in "${chunk_size[@]}"
    do
        /home/keveltun/install/vmtouch/bin/vmtouch -ve $DATA_DIR/test0.dat
        $BUILD_DIR/scripts/benchmark_create $DATA_DIR/test0.dat $chunk
    done
done
rm $DATA_DIR/test0.dat
