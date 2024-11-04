#!/bin/bash

BUILD_DIR="$HOME/research/anl/state-diff/build"

MB=$((1024 * 1024))
GB=$((1024 * $MB))
data_size=$((2 * $GB))
# data_size=$((16 * $MB))
ratio=1
max_ratio=4;
outname="test"

echo "==============================================================================="
echo " Generate data "
echo "==============================================================================="
$BUILD_DIR/src/tools/data_generator --data-len $data_size -n 1 -e 0 --num-changes 0 $outname \
  --kokkos-num-threads=2 --kokkos-map-device-id-by=mpi_rank

echo "==============================================================================="
echo " Test data loader "
echo "==============================================================================="
while [ $ratio -le $max_ratio ]
do
  batch_size=$(($data_size / $ratio))
  echo "DataSize = $data_size is $ratio times BatchSize = $batch_size"
  nsys profile --trace=cuda,nvtx -o profile_loader_${ratio} --force-overwrite true \
    $BUILD_DIR/test/loader_test test0.dat $data_size $batch_size $1
  /home/kta7930/research/anl/install/vmtouch/usr/local/bin/vmtouch -ve test0.dat
  ratio=$(( $ratio * 2 ))
done

rm test0.dat