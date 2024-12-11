#!/bin/bash

BUILD_DIR="$HOME/research/anl/state-diff/build"

MB=$((1024 * 1024))
GB=$((1024 * $MB))
data_size=$((1 * $GB))
outname="test"
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
    /home/kta7930/research/anl/install/vmtouch/usr/local/bin/vmtouch -ve test0.dat
    $BUILD_DIR/benchmark/benchmark_thrpt test0.dat

    # output two numbers (f2h, h2d). read the two numbers, add them to a list and sort to find max of both
done

echo "==============================================================================="
echo " Benchmarking the tree creation time per chunk size  "
echo "==============================================================================="
chunk_size=( 16 32 64 128 256 512 1024)
for test_id in $(seq 1 $num_runs)
do 
    for chunk in "${chunk_size[@]}"
    do
        /home/kta7930/research/anl/install/vmtouch/usr/local/bin/vmtouch -ve test0.dat
        $BUILD_DIR/benchmark/benchmark_create test0.dat $chunk
    done
done
rm test0.dat