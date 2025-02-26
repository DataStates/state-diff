#!/usr/bin/env bash

export NVCC_WRAPPER_DEFAULT_COMPILER=CC
export CRAYPE_LINK_TYPE=dynamic

nthreads=8
niters=1
export OMP_NUM_THREADS=$nthreads

# =============================================================================
# Compare data
# =============================================================================
build_dir="$HOME/research/anl/state-diff/build/scripts"
project_dir="/data/8gpus"
vmtouch_dir="$HOME/research/anl/install/vmtouch"

run0_dir=${project_dir}/np796-500mil/run1
run1_dir=${project_dir}/np796-500mil/run2

#dedup_approaches=('direct' 'compare-tree')
# dedup_approaches=('direct')
dedup_approaches=('compare-tree')

outname="logs/np796_tree_compare_log"

MB=$(( 1024 * 1024 ))
GB=$(( 1024 * 1024 * 1024 ))

# chunk_sizes=(8192 16384 32768 65536 131072)
chunk_sizes=(8192)
# error_tols=(0.0000001 0.00001 0.001)
# error_tols=(0.0000001)
error_tols=(0.001)

data_type=('float')
cmd_flags="--host-cache $((16 * $GB)) --dev-cache $((16 * $GB)) --level 13 -r $outname"

# Get filenames for different runs
var_name=combined-0-10
run0_full_files=( $(ls ${run0_dir}/m000p.mpirestart-${var_name}*.dat) )
run1_full_files=( $(ls ${run1_dir}/m000p.mpirestart-${var_name}*.dat) )
echo "Run 0 data full files: ${run0_full_files[@]}"
echo "Run 1 data full files: ${run1_full_files[@]}"

for chunk_size in "${chunk_sizes[@]}";
do
  for tol in "${error_tols[@]}";
  do
    for dtype in "${data_type[@]}";
    do
      for iter in $(seq 1 $niters)
      do
        echo "========================================================================"
        echo " Data type $dtype"
        echo " Chunk size $chunk_size"
        echo " Error tolerance $tol"
        echo "========================================================================"
        for approach in "${dedup_approaches[@]}";
        do
          # =============================================================================
          # Part 1: Create trees and a softlink for direct checkpoints
          # =============================================================================
          echo "------------------------------------------------------------------------"
          echo " Method $approach : Prepare Checkpoints"
          echo "------------------------------------------------------------------------"
          if [ "$approach" == "compare-tree" ]; then
            rm ${run0_dir}/*.${approach}
            rm ${run1_dir}/*.${approach}
            $vmtouch_dir/vmtouch -ve "${run0_dir}/"
            $vmtouch_dir/vmtouch -ve "${run1_dir}/"
            sleep 5s
            echo "$build_dir/eval_statediff -c $chunk_size --type $dtype $cmd_flags --error $tol
              --run0 ${run0_full_files[@]} ${run1_full_files[@]}
              --kokkos-num-threads=$nthreads --kokkos-map-device-id-by=mpi_rank"
            $build_dir/eval_statediff -c $chunk_size --type $dtype $cmd_flags --error $tol \
              --run0 ${run0_full_files[@]} ${run1_full_files[@]} \
              --kokkos-num-threads=$nthreads --kokkos-map-device-id-by=mpi_rank
          else
            for r0_file in "${run0_full_files[@]}";
            do
              echo "Deleting ${r0_file}*.0.direct and creating link ${r0_file}.0.direct"
              rm ${r0_file}*.0.direct
              ln -s ${r0_file} ${r0_file}.0.direct
            done
            for r1_file in "${run1_full_files[@]}";
            do
              echo "Deleting ${r1_file}*.1.direct and creating link ${r1_file}.1.direct"
              rm ${r1_file}*.1.direct
              ln -s ${r1_file} ${r1_file}.1.direct
            done
          fi
        done
        for approach in "${dedup_approaches[@]}";
        do
          # =============================================================================
          # Part 2: Compare data
          # =============================================================================
          run0_files=( $(ls ${run0_dir}/*.${approach}) )
          run1_files=( $(ls ${run1_dir}/*.${approach}) )
          echo "------------------------------------------------------------------------"
          echo " Method $approach : Compare Checkpoints"
          echo "------------------------------------------------------------------------"
          echo "Run 0 files: ${run0_files[@]}"
          echo "Run 1 files: ${run1_files[@]}"
          echo "Run 0 full files: ${run0_full_files[@]}"
          echo "Run 1 full files: ${run1_full_files[@]}"
          $vmtouch_dir/vmtouch -ve "${run0_dir}/"
          $vmtouch_dir/vmtouch -ve "${run1_dir}/"
          sleep 5s
          echo "$build_dir/eval_statediff -c $chunk_size --type $dtype $cmd_flags --error $tol
            --run0-full ${run0_full_files[@]}
            --run1-full ${run1_full_files[@]}
            --run0 ${run0_files[@]}
            --run1 ${run1_files[@]}
            --kokkos-num-threads=$nthreads
            --kokkos-map-device-id-by=mpi_rank"
          $build_dir/eval_statediff -c $chunk_size --type $dtype $cmd_flags --error $tol \
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
