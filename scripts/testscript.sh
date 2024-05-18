#!/usr/bin/env bash
#PBS -l select=2:system=polaris
#PBS -l place=scatter:exclhost
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:eagle:grand
#PBS -q debug
#PBS -A Veloc
#PBS -o output/testing_2nodes.out
#PBS -e output/testing_2nodes.err
##PBS -o output/scratch_np1260_combined_gpu_roundhash_tree_perftest_polaris.out
##PBS -e output/scratch_np1260_combined_gpu_roundhash_tree_perftest_polaris.err

cd ${PBS_O_WORKDIR}

export NVCC_WRAPPER_DEFAULT_COMPILER=CC
export CRAYPE_LINK_TYPE=dynamic

#NNODES=`wc -l < $PBS_NODEFILE`
#NRANKS_PER_NODE=1
#NDEPTH=2
#NTHREADS=2

ndepth=16
nprocs=8
nthreads=16
procs_per_node=4
niters=1
export OMP_NUM_THREADS=$nthreads
#export OMP_PLACES=threads
#export OMP_PROC_BIND=spread

#export NVCC_WRAPPER_DEFAULT_COMPILER=g++
#df -h
#lsblk -o KNAME,TYPE,SIZE,MODEL

#export KOKKOS_TOOLS_LIBS=/home/nphtan/kokkos-tools/kp_kernel_logger.so
#export KOKKOS_TOOLS_LIBS=/home/nphtan/kokkos-tools/profiling/simple-kernel-timer/kp_kernel_timer.so
#export KOKKOS_TOOLS_LIBS=/home/nphtan/kokkos-tools/profiling/space-time-stack/kp_space_time_stack.so
#export KOKKOS_TOOLS_LIBS=/home/nphtan/kokkos-tools/profiling/memory-events/kp_memory_events.so
#export KOKKOS_TOOLS_LIBS=/home/nphtan/kokkos-tools/kp_nvtx_connector.so
#kokkos_tools_libs=/home/ntan1/kokkos-tools/profiling/space-time-stack/kp_space_time_stack.so

## =============================================================================
## Generate new data
## =============================================================================
#datalen=$((256 * 1024 * 1024 + 3)) #$((10 * 256 * 1024 * 1024 + 3))
#err_tol=0.001
#nchanges=$((1 * 256 * 1024 * 1024 / 100))
#filestr="${datalen}_floats_abs_perturb_${nchanges}_by_${err_tol}"
#outname="${filestr}.mpirestart-0."
#echo "COMMAND: ./data_generator --data-len $datalen -n 2 \
#  -e $err_tol --num-changes $nchanges --data-type float $outname \
#  --kokkos-num-threads=$nthreads --kokkos-map-device-id-by=mpi_rank"

#rm ${outname}*.dat
#rm *.direct *.compare-tree
#echo "==============================================================================="
#echo " Generate data "
#echo "==============================================================================="
#
#./data_generator --data-len $datalen -n 2 \
#  -e $err_tol --num-changes $nchanges --data-type float $outname \
#  --kokkos-num-threads=$nthreads --kokkos-map-device-id-by=mpi_rank

# =============================================================================
# Compare data
# =============================================================================
#chunk_sizes=(64 128 256 512 1024 2048 4096)
#chunk_sizes=(512 1024 2048 4096)
chunk_sizes=(4096)
#project_dir=/home/nphtan/state-diff/build
project_dir=/lus/grand/projects/VeloC/nphtan
run0_dir=${project_dir}/run1
run1_dir=${project_dir}/run2
#project_dir=/lus/eagle/projects/RECUP/kassogba/veloc-ckpt/haac/sc-experiments/8gpus/np1260-02bil
#filestr=m000p
dedup_approaches=('direct' 'compare-tree')
#dedup_approaches=('direct')
#dedup_approaches=('compare-tree')
MB=$(( 1024 * 1024 ))
GB=$(( 1024 * 1024 * 1024 ))
#buffer_sizes=( 64 128 256 512 1024 )
#buffer_sizes=( 1024 2048 4096 8192 )
buffer_sizes=( 1024 )
#error_tols=( 0.0 0.0000001 0.000001 0.00001 0.0001 0.001 )
error_tols=( 0.0000001 )
data_type=('float')
cmd_flags="--fuzzy-hash --enable-file-streaming --comp absolute --level 13"
#cmd_flags="--enable-file-streaming --comp absolute --level 6"
#cmd_flags="--fuzzy-hash --enable-file-streaming --comp equivalence --error 0.0000000 --level 3"
#ecmd_flags="--enable-file-streaming --comp equivalence --error 0.000000 --level 3"
#cmd_flags="--comp equivalence --error 0.000000 --level 3"

# Get filenames for different runs
var_name=combined
#run0_filestr=run1_m000p
#run1_filestr=run2_m000p
#dat_files=( $(ls ${project_dir}/*${filestr}.mpirestart-*.${var_name}.dat) )
#run0_full_files=( $(ls ${project_dir}/${run0_filestr}.mpirestart-*.${var_name}.dat) )
#run1_full_files=( $(ls ${project_dir}/${run1_filestr}.mpirestart-*.${var_name}.dat) )
run0_full_files=( $(ls ${project_dir}/run1/m000p.mpirestart-${var_name}-*.dat) )
run1_full_files=( $(ls ${project_dir}/run2/m000p.mpirestart-${var_name}-*.dat) )
#run0_full_files=$project_dir/run1/m000p.mpirestart-combined-0-10.dat
#run1_full_files=$project_dir/run2/m000p.mpirestart-combined-0-10.dat
#echo "All data full files: ${dat_files[@]}"
echo "Run 0 data full files: ${run0_full_files[@]}"
echo "Run 1 data full files: ${run1_full_files[@]}"

for chunk_size in "${chunk_sizes[@]}";
do
  for tol in "${error_tols[@]}";
  do
    for buffer_size in "${buffer_sizes[@]}";
    do
      for dtype in "${data_type[@]}";
      do
        for iter in $(seq 1 $niters)
        do
          echo "========================================================================"
          echo " Data type $dtype"
          echo " Buffer size $buffer_size MB"
          echo " Chunk size $chunk_size"
          echo " Error tolerance $tol"
          echo "========================================================================"
          for approach in "${dedup_approaches[@]}";
          do
            # =============================================================================
            # Part 1: Create trees
            # =============================================================================
            echo "------------------------------------------------------------------------"
            echo " Method $approach : Prepare Checkpoints"
            echo "------------------------------------------------------------------------"
            if [ "$approach" == "compare-tree" ]; then
              rm ${run0_dir}/*.${approach}
              rm ${run1_dir}/*.${approach}
              mpiexec -n $nprocs --ppn $procs_per_node -d $ndepth --cpu-bind depth \
                --env OMP_NUM_THREADS=$nthreads \
                ./repro_test -c $chunk_size --alg $approach --type $dtype \
                --buffer-len $(( $buffer_size * $MB)) $cmd_flags --error $tol \
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
            #run0_files=( $(ls ${project_dir}/${run0_filestr}.mpirestart-*.${var_name}.dat.*${approach}) )
            #run1_files=( $(ls ${project_dir}/${run1_filestr}.mpirestart-*.${var_name}.dat.*${approach}) )
            run0_files=( $(ls ${run0_dir}/*.${approach}) )
            run1_files=( $(ls ${run1_dir}/*.${approach}) )
            echo "------------------------------------------------------------------------"
            echo " Method $approach : Compare Checkpoints"
            echo "------------------------------------------------------------------------"
            echo "Run 0 files: ${run0_files[@]}"
            echo "Run 1 files: ${run1_files[@]}"
            echo "Run 0 full files: ${run0_full_files[@]}"
            echo "Run 1 full files: ${run1_full_files[@]}"
            mpiexec -n $nprocs --ppn $procs_per_node -d $ndepth --cpu-bind depth \
            --env OMP_NUM_THREADS=$nthreads \
            ./repro_test -c $chunk_size --alg $approach --type $dtype \
              --buffer-len $(( $buffer_size * $MB )) $cmd_flags --error $tol \
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
done


#            --depth=8 --cpu-bind depth \
