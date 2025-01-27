#!/bin/bash

nthreads=32
export CRAYPE_LINK_TYPE=dynamic
export OMP_NUM_THREADS=$nthreads

BUILD_DIR="$HOME/research/recup/veloc/apps/state-diff/buildcpu/scripts"
DATA_DIR="/lus/eagle/projects/RECUP/kassogba/veloc-ckpt/haac/sc-experiments/4gpus/np796-500mil/run1"
VMTOUCH_BIN="$HOME/install/vmtouch/bin/"

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

    echo "*******************************************************************************"
    echo " Testing DD throughput with chunk size: $(($chunk_size * 1024)) "
    echo "*******************************************************************************"
    local throughput=$(dd if="$source_file" of=/dev/null bs="${chunk_size}K" status=progress 2>&1 | \
        awk '/GB\/s/ {value=$(NF-1)} END {print value}')
    echo $throughput
    echo "$(($chunk_size * 1024)),${throughput}" >> "$csv_file"
}

liburing_thrupt_test() {
    local source_file=$1
    local chunk_size=$2
    local type=$3
    $VMTOUCH_BIN/vmtouch -ve $source_file

    echo "*******************************************************************************"
    echo " Benchmarking Liburing loading throughput with chunk size: $(($chunk_size * 1024)) "
    echo "*******************************************************************************"
    $BUILD_DIR/benchmark_liburing $source_file $(($chunk_size * 1024)) $type
}

posix_thrupt_test() {
    local source_file=$1
    local chunk_size=$2
    local type=$3
    $VMTOUCH_BIN/vmtouch -ve $source_file

    echo "*******************************************************************************"
    echo " Benchmarking Posix loading throughput with chunk size: $(($chunk_size * 1024)) "
    echo "*******************************************************************************"
    $BUILD_DIR/benchmark_posix $source_file $(($chunk_size * 1024)) $type
}

validate_liburing() {
    echo "==============================================================================="
    echo " Validating Liburing Implementation with variable read sizes "
    echo "==============================================================================="
    local source_file=$1
    local min_chunk_size=4  # Minimum chunk size in KB (4KB)
    local max_chunk_size=1048576  # Maximum chunk size in KB (256MB)
    # local max_chunk_size=262144  # Maximum chunk size in KB (256MB)
    local csv_file="dd_throughput_results.csv"

    # Generate chunk sizes
    if [[ ! -f "$csv_file" ]]; then
        echo "Chunk size (KB),throughput (GB/s)" > "$csv_file"
    fi

    # Test DD throughput for each chunk size
    local size=$min_chunk_size
    while [[ $size -le $max_chunk_size ]]; do
        dd_thrupt_test "$source_file" "$size" "$csv_file"
        size=$((size * 2))
    done

    # Test Liburing throughput for each chunk size
    local size=$min_chunk_size
    while [[ $size -le $max_chunk_size ]]; do
        liburing_thrupt_test "$source_file" "$size" 0 
        size=$((size * 2))
    done

    local size=$min_chunk_size
    while [[ $size -le $max_chunk_size ]]; do
        liburing_thrupt_test "$source_file" "$size" 1
        size=$((size * 2))
    done

    # Test Posix throughput for each chunk size
    # local size=$min_chunk_size
    # while [[ $size -le $max_chunk_size ]]; do
    #     posix_thrupt_test "$source_file" "$size" 0 
    #     size=$((size * 2))
    # done

    # posix_thrupt_test "$source_file" 0 1

    echo "Validation tests (DD + Liburing) for read throughput completed."
}

benchmark_creation() {
    echo "==============================================================================="
    echo " Benchmarking tree creation "
    echo "==============================================================================="

}

benchmark_comparison() {
    echo "==============================================================================="
    echo " Benchmarking direct comparison "
    echo "==============================================================================="

}

# Help function to display usage
usage() {
    echo "Usage: $0 {read|create|compare}"
    echo "    read    - Executes procedure to analyze read throughput"
    echo "    create  - Executes procedure to benchmark tree ceation"
    echo "    compare - Executes procedure to benchmark direct comparison"
}

if [[ $# -eq 0 ]]; then
    echo "Error: No benchmarking task specified."
    usage
    exit 1
fi

# Ensure the data is on the SSD at the start of the experiments
pfs2ssd "$DATA_DIR/m000p.mpirestart-combined-0-10.dat" "/local/scratch"

SOURCE_FILE="/local/scratch/m000p.mpirestart-combined-0-10.dat"
if [[ ! -f "$SOURCE_FILE" ]]; then
    echo "Error: Source file does not exist: $SOURCE_FILE"
    exit 1
fi

case "$1" in
    read)
        validate_liburing "$SOURCE_FILE"
        ;;
    create)
        benchmark_creation
        ;;
    compare)
        benchmark_comparison
        ;;
    *)
        echo "Error: Invalid benchmarking task name."
        usage
        exit 1
        ;;
esac