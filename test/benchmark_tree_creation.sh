#!/bin/bash

BUILD_DIR=../build/test
KB=$((1024))
MB=$((1024 * KB))
GB=$((1024 * MB))

# data_sizes=($((1 * GB)) $((2 * GB)) $((3 * GB)))
# dev_buf_sizes=($((256 * MB)) $((512 * MB)) $((1024 * MB)))
# chunk_sizes=($((4 * KB)) $((32 * KB)) $((256 * KB)))

data_sizes=($((1 * GB)))
dev_buf_sizes=($((256 * MB)) $((512 * MB)) $((1024 * MB)))
chunk_sizes=($((4 * KB)))

n_run=1

output_csv="../test/benchmark_tree_results.csv"
# echo "data_size,dev_buf_size,chunk_size,total_create,data_loading,create_kernel,kernel_wait" > "$output_csv"
echo "data_size,dev_buf_size,chunk_size,leaves_data_loading,leaves_kernel_wait,leaves_create_kernel,rest_create" > "$output_csv"

for ((i = 1; i <= n_run; i++)); do
    echo "Iteration $i/$n_run"
    for data_size in "${data_sizes[@]}"; do
        for dev_buf_size in "${dev_buf_sizes[@]}"; do
            for chunk_size in "${chunk_sizes[@]}"; do
                echo "Running benchmark with data_size=$data_size, dev_buf_size=$dev_buf_size, chunk_size=$chunk_size"

                $BUILD_DIR/benchmark_tree_creation "$data_size" "$dev_buf_size" "$chunk_size" "$output_csv"
                rm checkpoint.dat
            done
        done
    done
done

# for ((i = 1; i <= n_run; i++)); do
#     echo "Iteration $i/$n_run"
#     for data_size in "${data_sizes[@]}"; do
#         for dev_buf_size in "${dev_buf_sizes[@]}"; do
#             for chunk_size in "${chunk_sizes[@]}"; do
#                 echo "Running benchmark with data_size=$data_size, dev_buf_size=$dev_buf_size, chunk_size=$chunk_size"
#                 fname=../test/profile_${data_size}_${dev_buf_size}_${chunk_size}

#                 # Profile execution
#                 nsys profile --trace=cuda,nvtx -o ${fname} \
#                     $BUILD_DIR/benchmark_tree_creation "$data_size" "$dev_buf_size" "$chunk_size" "$output_csv"

#                 # Generate sqlite file
#                 echo "Generating sqlite"
#                 nsys stats --force-export=true --quiet ${fname}.nsys-rep

#                 # Execute the SQL queries and store the results
#                 echo "Executing SQL queries"
#                 data_loading=$(sqlite3 "${fname}.sqlite" "SELECT SUM(end-start) FROM NVTX_EVENTS WHERE text='data_loading';")
#                 kernel_exec=$(sqlite3 "${fname}.sqlite" "SELECT SUM(end-start) FROM NVTX_EVENTS WHERE text='kernel_hashing';")
#                 kernel_wait=$(sqlite3 "${fname}.sqlite" "SELECT SUM(end-start) FROM NVTX_EVENTS WHERE text='kernel_waiting';")

#                 # Check if the queries return valid results (non-empty)
#                 echo "Updating output file"
#                 if [[ -z "$data_loading" ]]; then
#                     data_loading=0
#                 fi
#                 if [[ -z "$kernel_exec" ]]; then
#                     kernel_exec=0
#                 fi
#                 if [[ -z "$kernel_wait" ]]; then
#                     kernel_wait=0
#                 fi

#                 # Read the last line of the CSV
#                 last_line=$(tail -n 1 "$output_csv")

#                 # Remove the last line from the file
#                 sed -i '$ d' "$output_csv"

#                 # Append the results to the last line and write it back to the CSV
#                 echo "$last_line,$data_loading,$kernel_exec,$kernel_wait" >> "$output_csv"

#                 # Clear space
#                 # echo "Clearing up files"
#                 # rm checkpoint.dat ${fname}.nsys-rep ${fname}.sqlite
#             done
#         done
#     done
# done