#!/bin/bash

ndepth=16
nprocs=1
nthreads=16
procs_per_node=1
export CRAYPE_LINK_TYPE=dynamic
export OMP_NUM_THREADS=$nthreads

BUILD_DIR="$HOME/research/recup/veloc/apps/state-diff/buildcpu"
DATA_DIR="/lus/eagle/projects/RECUP/kassogba/veloc-ckpt/haac/sc-experiments/4gpus/np796-500mil/run1"
KB=1024
num_runs=1

echo " Benchmarking the tree creation time for modeling "
echo "==============================================================================="
KB=1
#chunk_size=(32 64 128 256 512 1024 2048)
chunk_size=(1024 2048)
#chunk_size=(4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768 65536 131072)
#chunk_size=(262144 524288 1048576 2097152 4194304)
errors=(0.01 0.0001 0.0000001)


for test_id in $(seq 1 $num_runs)
do
    for chunk in "${chunk_size[@]}"
    do
	for error in "${errors[@]}"
	do
        	mpiexec -n $nprocs --ppn $procs_per_node -d $ndepth --cpu-bind depth \
			/home/keveltun/install/vmtouch/bin/vmtouch -ve $DATA_DIR
        	mpiexec -n $nprocs --ppn $procs_per_node -d $ndepth --cpu-bind depth \
                	--env OMP_NUM_THREADS=$nthreads \
			$BUILD_DIR/scripts/benchmark_create_mod \
			$DATA_DIR/m000p.mpirestart-combined-0-10.dat \
			$(($chunk * $KB)) $error \
			--kokkos-num-threads=$nthreads \
			--kokkos-map-device-id-by=mpi_rank
	done
    done
done
