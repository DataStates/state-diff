#!/usr/bin/env bash

nthreads=2
nprocs=1
niters=1
export OMP_NUM_THREADS=$nthreads
#export OMP_PLACES=threads
#export OMP_PROC_BIND=spread

#export NVCC_WRAPPER_DEFAULT_COMPILER=g++
#df -h
#lsblk -o KNAME,TYPE,SIZE,MODEL

#export KOKKOS_TOOLS_LIBS=/home/ntan1/kokkos-tools/kp_kernel_logger.so
#export KOKKOS_TOOLS_LIBS=/home/ntan1/kokkos-tools/profiling/simple-kernel-timer/kp_kernel_timer.so
#export KOKKOS_TOOLS_LIBS=/home/ntan1/kokkos-tools/profiling/space-time-stack/kp_space_time_stack.so
#export KOKKOS_TOOLS_LIBS=/home/ntan1/kokkos-tools/profiling/memory-events/kp_memory_events.so
#export KOKKOS_TOOLS_LIBS=/home/ntan1/kokkos-tools/kp_nvtx_connector.so
#kokkos_tools_libs=/home/ntan1/kokkos-tools/profiling/space-time-stack/kp_space_time_stack.so

# =============================================================================
# Generate new data
# =============================================================================
datalen=$((256 * 1024 * 1024 + 3)) #$((10 * 256 * 1024 * 1024 + 3))
err_tol=0.001
nchanges=$((1 * 256 * 1024 * 1024 / 100))
filestr="${datalen}_floats_abs_perturb_${nchanges}_by_${err_tol}"
outname="${filestr}-mpirestart-0."
echo "COMMAND: ./data_generator --data-len $datalen -n 2 \
  -e $err_tol --num-changes $nchanges --data-type float $outname \
  --kokkos-num-threads=$nthreads --kokkos-map-device-id-by=mpi_rank"

rm ${outname}*.dat
rm *.direct *.compare-tree
echo "==============================================================================="
echo " Generate data "
echo "==============================================================================="
#nvidia-smi
#compute-sanitizer --force-blocking-launches yes \
./data_generator --data-len $datalen -n 2 \
  -e $err_tol --num-changes $nchanges --data-type float $outname \
  --kokkos-num-threads=$nthreads --kokkos-map-device-id-by=mpi_rank

# =============================================================================
# Compare data
# =============================================================================
#chunk_sizes=(128 256 512 1024 2048 4096)
chunk_sizes=(192)
#project_dir=/data/gclab/dedup
project_dir=/home/kta7930/research/anl/state-diff/build
#filestr=2684354560_floats_abs_perturb_2684354_by_0.001
#filestr=2684354563_floats_abs_perturb_2684354_by_0.001
dedup_approaches=('direct' 'compare-tree')
#dedup_approaches=('direct')
#dedup_approaches=('compare-tree')
buffer_sizes=( $((1 * 8 * 1024 * 1024)) )
data_type=('float')
cmd_flags="--fuzzy-hash --enable-file-streaming --comp absolute --error 0.0005369000"
# cmd_flags="--fuzzy-hash --enable-file-streaming --comp absolute --error 0.000000000 --level 5"
#cmd_flags="--fuzzy-hash --enable-file-streaming --comp equivalence --error 0.0000000 --level 3"
#cmd_flags="--enable-file-streaming --comp equivalence --error 0.000000 --level 3"
#cmd_flags="--comp equivalence --error 0.000000 --level 3"

# Get filenames for different runs
dat_files=( $(ls ${project_dir}/${filestr}-mpirestart-*.dat) )
run0_full_files=( $(ls ${project_dir}/${filestr}-mpirestart-*.0.dat) )
run1_full_files=( $(ls ${project_dir}/${filestr}-mpirestart-*.1.dat) )
echo "All data full files: ${dat_files[@]}"
echo "Run 0 data full files: ${run0_full_files[@]}"
echo "Run 1 data full files: ${run1_full_files[@]}"


# =============================================================================
# Part 1: Create trees
# =============================================================================
for chunk_size in "${chunk_sizes[@]}";
do
  for approach in "${dedup_approaches[@]}";
  do
    for buffer_size in "${buffer_sizes[@]}";
    do
      for dtype in "${data_type[@]}";
      do
        echo "========================================================================"
        echo " Method $approach, Data type $dtype, Buffer size $buffer_size :  Prepare"
        echo "========================================================================"
#        ./time -v \
        ./repro_test -c $chunk_size --alg $approach --type $dtype --buffer-len $buffer_size $cmd_flags \
          --run0 ${dat_files[@]} \
          --kokkos-num-threads=$nthreads --kokkos-map-device-id-by=mpi_rank
      done
    done
  done
done

# =============================================================================
# Part 2: Compare data
# =============================================================================
for iter in $(seq 1 $niters)
do
  for chunk_size in "${chunk_sizes[@]}";
  do
    for approach in "${dedup_approaches[@]}";
    do
      # Delete full files to avoid caching
      rm ${dat_files[@]}
      echo "==============================================================================="
      echo " Generate data "
      echo "==============================================================================="
      ./data_generator --data-len $datalen -n 2 \
        -e $err_tol --num-changes $nchanges --data-type float $outname \
        --kokkos-num-threads=$nthreads --kokkos-map-device-id-by=mpi_rank

      # Get runs specific files for trees and metadata
      run0_files=( $(ls ${project_dir}/${filestr}-mpirestart-*.0.dat.0*${approach}) )
      run1_files=( $(ls ${project_dir}/${filestr}-mpirestart-*.1.dat.1*${approach}) )
      for buffer_size in "${buffer_sizes[@]}";
      do
        for dtype in "${data_type[@]}";
        do
          echo "========================================================================"
          echo " Method $approach, Data type $dtype, Buffer size $buffer_size :  Compare"
          echo "========================================================================"
  #        nsys profile --trace cuda,nvtx -o trace \
  #        ./time -v \
  #        compute-sanitizer --force-blocking-launches yes \
  #        valgrind \
  #        ncu --set full --call-stack --nvtx -o full_set_profile \
          ./repro_test -c $chunk_size --alg $approach --type $dtype --buffer-len $buffer_size $cmd_flags \
            --run0-full ${run0_full_files[@]} \
            --run1-full ${run1_full_files[@]} \
            --run0 ${run0_files[@]} \
            --run1 ${run1_files[@]} \
            --kokkos-num-threads=$nthreads \
            --kokkos-map-device-id-by=mpi_rank
        done
      done
    done
  done
done

