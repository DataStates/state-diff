#!/bin/bash

BUILD_DIR="$HOME/research/recup/veloc/apps/state-diff/build/scripts"
VMTOUCH_BIN="$HOME/install/vmtouch/bin/"
#BUILD_DIR="$HOME/research/recup/veloc/apps/state-diff/build_sophia/scripts"
#VMTOUCH_BIN="$HOME/install/sophia/vmtouch/usr/local/bin/"
# BUILD_DIR="$HOME/research/anl/state-diff/build/scripts"
# VMTOUCH_BIN="$HOME/research/anl/install/vmtouch/"
KB=1024
MB=$((1024 * $KB))
GB=$((1024 * $MB))
NTHREADS=8
export OMP_NUM_THREADS=$NTHREADS

rnd_data_gen() {
    local destination=$1
    local data_size=$2

    echo "==============================================================================="
    echo " Copying target file from PFS to SSD "
    echo "==============================================================================="
    $BUILD_DIR/../src/tools/data_generator --data-len $data_size -n 2 -e 0 --num-changes 0 $destination \
        --kokkos-num-threads=$NTHREADS
    $VMTOUCH_BIN/vmtouch -ve "${destination}0.dat" "${destination}1.dat"
}

pfs2ssd() {
    local source_file=$1
    local destination=$2

    echo "==============================================================================="
    echo " Copying target file from PFS to SSD "
    echo "==============================================================================="
    cp $source_file $destination
    $VMTOUCH_BIN/vmtouch -ve $source_file
    $VMTOUCH_BIN/vmtouch -ve $destination
}

dd_thrupt_test() {
    local source_file=$1
    local chunk_size=$2
    local csv_file=$3
    $VMTOUCH_BIN/vmtouch -ve $source_file

    # echo "*******************************************************************************"
    # echo " Testing DD throughput with chunk size: $(($chunk_size * $KB)) "
    # echo "*******************************************************************************"
    dd_output=$(dd if="$source_file" of=/dev/null bs="${chunk_size}K" status=progress 2>&1)
    local throughput=$(echo "$dd_output" | \
        awk '/[MG]B\/s/ {value=$(NF-1); unit=$NF} END {if (unit=="MB/s") value /= 1024; print value}')
    
    echo "($(($chunk_size * $KB))) DD throughput for all data = $throughput GB/s"
    echo "DD,$(($chunk_size * $KB)),0,0,${throughput}" >> "$csv_file"
}

liburing_thrupt_test() {
    local source_file=$1
    local chunk_size=$2
    local type=$3
    $VMTOUCH_BIN/vmtouch -ve $source_file

    # echo "*******************************************************************************"
    # echo " Benchmarking Liburing loading throughput with chunk size: $(($chunk_size * $KB)) "
    # echo "*******************************************************************************"
    $BUILD_DIR/benchmark_liburing $source_file $(($chunk_size * $KB)) $type
}

posix_thrupt_test() {
    local source_file=$1
    local chunk_size=$2
    local type=$3
    $VMTOUCH_BIN/vmtouch -ve $source_file

    # echo "*******************************************************************************"
    # echo " Benchmarking Posix loading throughput with chunk size: $(($chunk_size * $KB)) "
    # echo "*******************************************************************************"
    $BUILD_DIR/benchmark_posix $source_file $(($chunk_size * $KB)) $type
}

tree_create_thrupt_bench() {
	local source_file=$1
	local chunk_size=$2
	local error=$3
	$VMTOUCH_BIN/vmtouch -ve $source_file

	# echo "*******************************************************************************"
	# echo " Benchmarking Creation for modeling with chunk size: $(($chunk_size * $KB)) "
	# echo "*******************************************************************************"
	$BUILD_DIR/benchmark_create_mod $source_file $(($chunk_size * $KB)) $error \
               --kokkos-num-threads=$NTHREADS
}

validate_liburing() {
    echo "==============================================================================="
    echo " Validating Liburing Implementation with variable read sizes "
    echo "==============================================================================="
    local source_file=$1
    local min_chunk_size=4  # Minimum chunk size in KB (4KB)
    local max_chunk_size=32768 # Maximum chunk size in KB (32MB)
    # local csv_file="dd_throughput_results.csv"
    local csv_file="validate_liburing.csv"

    if [[ ! -f "$csv_file" ]]; then
        echo "API,Chunk Size,Data Size,Load time,Load thrupt" > "$csv_file"
    fi

    # Test DD throughput for each chunk size
    local size=$min_chunk_size
    while [[ $size -le $max_chunk_size ]]; do
        dd_thrupt_test "$source_file" "$size" "$csv_file"
        size=$((size * 2))
    done

    # Test Liburing throughput for each chunk size
    # This experiment creates and issues one request for chunk_size data at a time (1 req N times).
    local size=$min_chunk_size
    while [[ $size -le $max_chunk_size ]]; do
        liburing_thrupt_test "$source_file" "$size" 0 
        size=$((size * 2))
    done

    # This experiment creates N requests of chunk_size and issues one call for the entire dataset (N req 1 time).
    local size=$min_chunk_size
    while [[ $size -le $max_chunk_size ]]; do
        liburing_thrupt_test "$source_file" "$size" 1
        size=$((size * 2))
    done

    # Test Posix throughput for each chunk size
    # This first test measures the throughput in the case of 1 req N times
    local size=$min_chunk_size
    while [[ $size -le $max_chunk_size ]]; do
        posix_thrupt_test "$source_file" "$size" 0 
        size=$((size * 2))
    done

    # This test measures when all data is read in one call
    posix_thrupt_test "$source_file" 0 1

    echo "Validation tests (DD + Liburing) for read throughput completed."
}

benchmark_creation() {
    echo "==============================================================================="
    echo " Benchmarking tree creation "
    echo "==============================================================================="
    local source_file=$1
    local min_chunk_size=4  # Minimum chunk size in KB (4KB)
    local max_chunk_size=131072  # Maximum chunk size in KB (128MB selected as batch size is 128MB)
    local size=$min_chunk_size
    while [[ $size -le $max_chunk_size ]]; do
        tree_create_thrupt_bench "$source_file" "$size" 0.0000001
        size=$((size * 2))
    done
}

benchmark_comparison() {
    echo "==============================================================================="
    echo " Benchmarking direct comparison "
    echo "==============================================================================="

}

# Help function to display usage
usage() {
    echo "Usage: $0 {local|polaris} {read|create|compare}"
    echo "    local   - Randomly generates data and executes procedures on the data"
    echo "    polaris - Transfers data from data_dir to ssd and executes procedures"
    echo "    read    - Executes procedure to analyze read throughput"
    echo "    create  - Executes procedure to benchmark tree ceation"
    echo "    compare - Executes procedure to benchmark direct comparison"
}

# if [[ $# -lt 2 ]]; then
#     echo "Error: No benchmarking task specified."
#     usage
#     exit 1
# fi

case "$1" in
    local)
        ckpt_size=$((3 * $GB))
        ckpt_name="/lus/eagle/projects/RECUP/kassogba/veloc-ckpt/rand-sample/rand_sample"
        rnd_data_gen $ckpt_name $ckpt_size
        SOURCE_FILE="/lus/eagle/projects/RECUP/kassogba/veloc-ckpt/rand-sample/rand_sample0.dat"
        ;;
    polaris)
        # Ensure the data is on the SSD at the start of the experiments
        DATA_DIR="/lus/eagle/projects/RECUP/kassogba/veloc-ckpt/haac/sc-experiments/4gpus/np796-500mil/run1"
        pfs2ssd "$DATA_DIR/m000p.mpirestart-combined-0-10.dat" "/local/scratch"
        SOURCE_FILE="/local/scratch/m000p.mpirestart-combined-0-10.dat"
        ;;
    *)
        echo "Error: Invalid benchmarking location."
        usage
        exit 1
        ;;
esac

if [[ ! -f "$SOURCE_FILE" ]]; then
    echo "Error: Source file does not exist: $SOURCE_FILE"
    exit 1
fi

# case "$2" in
#     read)
#         validate_liburing "$SOURCE_FILE"
#         ;;
#     create)
#         benchmark_creation "$SOURCE_FILE"
#         ;;
#     compare)
#         benchmark_comparison
#         ;;
#     *)
#         echo "Error: Invalid benchmarking task name."
#         usage
#         exit 1
#         ;;
# esac
