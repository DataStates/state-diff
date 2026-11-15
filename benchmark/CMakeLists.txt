# ============================================================
# Benchmarks: exploratory I/O and compute benchmarks
# ============================================================

# --- Pure Kokkos compute benchmarks ---

add_executable(bench_compare_kernel_iterative bench_compare_kernel_iterative.cpp)
target_link_libraries(bench_compare_kernel_iterative PRIVATE Kokkos::kokkos)
set_target_properties(bench_compare_kernel_iterative PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS OFF
)

add_executable(bench_compare_approaches bench_create.cpp)
target_link_libraries(bench_compare_approaches PRIVATE Kokkos::kokkos)
set_target_properties(bench_compare_approaches PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS OFF
)

# --- Pure POSIX I/O benchmarks ---

add_executable(bench_read_posix read.cpp)
set_target_properties(bench_read_posix PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS OFF
)
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    target_link_libraries(bench_read_posix PRIVATE CUDA::cudart)
endif()

add_executable(bench_load_posix load_posix.cpp)
set_target_properties(bench_load_posix PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS OFF
)

# --- scissor I/O benchmarks ---

add_executable(bench_load_single bench_load_calls_single.cpp)
target_link_libraries(bench_load_single PRIVATE statediff scissor::scissor)
set_target_properties(bench_load_single PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS OFF
)

add_executable(bench_load_two_files bench_load_calls_working.cpp)
target_link_libraries(bench_load_two_files PRIVATE statediff scissor::scissor)
set_target_properties(bench_load_two_files PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS OFF
)

add_executable(bench_load_liburing load_liburing.cpp)
target_link_libraries(bench_load_liburing PRIVATE statediff scissor::scissor)
set_target_properties(bench_load_liburing PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS OFF
)

add_executable(bench_posix_vs_liburing posix_vs_liburing.cpp)
target_link_libraries(bench_posix_vs_liburing PRIVATE statediff scissor::scissor)
set_target_properties(bench_posix_vs_liburing PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS OFF
)

# --- Direct comparison benchmark (scissor + Kokkos) ---

add_executable(bench_compare_direct compare.cpp)
target_link_libraries(bench_compare_direct PRIVATE statediff scissor::scissor)
target_link_libraries(bench_compare_direct PRIVATE Kokkos::kokkos)
set_target_properties(bench_compare_direct PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS OFF
)

# --- statediff tree creation benchmarks ---

add_executable(bench_create_tree create.cpp)
target_include_directories(bench_create_tree PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(bench_create_tree PRIVATE statediff)
target_link_libraries(bench_create_tree PRIVATE Kokkos::kokkos)
target_link_libraries(bench_create_tree PRIVATE OpenSSL::SSL)
set_target_properties(bench_create_tree PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS OFF
)

add_executable(bench_create_mod create_mod.cpp)
target_include_directories(bench_create_mod PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(bench_create_mod PRIVATE statediff)
target_link_libraries(bench_create_mod PRIVATE Kokkos::kokkos)
target_link_libraries(bench_create_mod PRIVATE OpenSSL::SSL)
set_target_properties(bench_create_mod PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS OFF
)
