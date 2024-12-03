#!/bin/bash

BUILD_DIR="$HOME/research/anl/state-diff/build"

MB=$((1024 * 1024))
GB=$((1024 * $MB))
data_size=$((1 * $GB))
host_cache=$((2 * $GB))
dev_cache=$((2 * $GB))
seg_size=$((128 * $MB))

# data_size=$((1024))
# seg_size=$((128))
# host_cache=$((128 * $MB))
# dev_cache=$((128 * $MB))

batch_size=4
max_batch_size=8
# max_batch_size=$(( $dev_cache / $seg_size))
outname="test"

echo "==============================================================================="
echo " Generate data "
echo "==============================================================================="
$BUILD_DIR/src/tools/data_generator --data-len $data_size -n 2 -e 0.01 --num-changes 0 $outname \
  --kokkos-num-threads=2 --kokkos-map-device-id-by=mpi_rank
/home/kta7930/research/anl/install/vmtouch/usr/local/bin/vmtouch -ve test0.dat test1.dat

echo "==============================================================================="
echo " Test data loader "
echo "==============================================================================="
while [ $batch_size -le $max_batch_size ]
do
  # cmd="$BUILD_DIR/src/loader/test/single_loader_test $host_cache $dev_cache test0.dat $seg_size $batch_size"
  cmd="$BUILD_DIR/src/loader/test/two_loader_test $host_cache $dev_cache test0.dat test1.dat $seg_size $batch_size"
  
  echo "Batch Size = $batch_size"
  echo $cmd
  # nsys profile --trace=cuda,nvtx -o profile_loader_${batch_size} --force-overwrite true 
  $cmd
  /home/kta7930/research/anl/install/vmtouch/usr/local/bin/vmtouch -ve test0.dat test1.dat
  batch_size=$(( $batch_size * 2 ))
done

$BUILD_DIR/src/loader/test/benchmark test0.dat
rm test0.dat test1.dat