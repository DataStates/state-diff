#!/bin/bash

BUILD_DIR="$HOME/research/anl/state-diff/build"

MB=$((1024 * 1024))
GB=$((1024 * $MB))
data_size=$((4 * $GB))
seg_size=$((128 * $MB))

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
# run_file="/data/8gpus/np796-500mil/run1/m000p.mpirestart-combined-0-10.dat"
# flags="-f"

run1_file="test0.dat"
run2_file="test1.dat"
flags="-u"

pct=10

# cmd="$BUILD_DIR/src/loader/test/test_loader $run1_file $flags"

cmd="$BUILD_DIR/src/loader/test/test_loader_2files $run1_file $run2_file $flags $pct"

echo $cmd
eval $cmd
/home/kta7930/research/anl/install/vmtouch/usr/local/bin/vmtouch -ve test0.dat test1.dat
rm test0.dat test1.dat