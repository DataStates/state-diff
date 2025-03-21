add_executable(benchmark_thrpt read.cpp)
set_target_properties(benchmark_thrpt PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS OFF
)

# add_executable(benchmark_create create.cpp)
# target_link_libraries(benchmark_create PUBLIC Kokkos::kokkos)
# target_link_libraries(benchmark_create PRIVATE OpenSSL::SSL)
# target_link_libraries(benchmark_create PRIVATE statediff)
# set_target_properties(benchmark_create PROPERTIES
#     CXX_STANDARD 17
#     CXX_STANDARD_REQUIRED YES
#     CXX_EXTENSIONS OFF
# )

# add_executable(benchmark_create_mod create_mod.cpp)
# target_link_libraries(benchmark_create_mod PUBLIC Kokkos::kokkos)
# target_link_libraries(benchmark_create_mod PRIVATE OpenSSL::SSL)
# target_link_libraries(benchmark_create_mod PRIVATE statediff)
# set_target_properties(benchmark_create_mod PROPERTIES
#     CXX_STANDARD 17
#     CXX_STANDARD_REQUIRED YES
#     CXX_EXTENSIONS OFF
# )

# add_executable(benchmark_compare compare.cpp)
# target_link_libraries(benchmark_compare PUBLIC Kokkos::kokkos)
# target_link_libraries(benchmark_compare PRIVATE OpenSSL::SSL)
# target_link_libraries(benchmark_compare PRIVATE statediff)
# set_target_properties(benchmark_compare PROPERTIES
#     CXX_STANDARD 17
#     CXX_STANDARD_REQUIRED YES
#     CXX_EXTENSIONS OFF
# )

add_executable(benchmark_liburing load_liburing.cpp)
target_link_libraries(benchmark_liburing PUBLIC Kokkos::kokkos)
target_link_libraries(benchmark_liburing PRIVATE OpenSSL::SSL)
target_link_libraries(benchmark_liburing PRIVATE statediff)
set_target_properties(benchmark_liburing PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS OFF
)

add_executable(benchmark_posix load_posix.cpp)
target_link_libraries(benchmark_posix PUBLIC Kokkos::kokkos)
target_link_libraries(benchmark_posix PRIVATE OpenSSL::SSL)
target_link_libraries(benchmark_posix PRIVATE statediff)
set_target_properties(benchmark_posix PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS OFF
)

find_package(MPI REQUIRED)

add_executable(eval_statediff eval_statediff.cpp)
target_include_directories(eval_statediff PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(eval_statediff PRIVATE MPI::MPI_CXX)
target_link_libraries(eval_statediff PRIVATE Kokkos::kokkos)
target_link_libraries(eval_statediff PRIVATE OpenSSL::SSL)
target_link_libraries(eval_statediff PRIVATE statediff)
target_link_libraries(eval_statediff PRIVATE argparse)
set_target_properties(eval_statediff PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS OFF
)

add_executable(posix_vs_liburing posix_vs_liburing.cpp)
target_include_directories(posix_vs_liburing PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(posix_vs_liburing PRIVATE OpenSSL::SSL)
target_link_libraries(posix_vs_liburing PRIVATE statediff)
set_target_properties(eval_statediff PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS OFF
)

add_executable(bench_create bench_create.cpp)
target_include_directories(bench_create PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(bench_create PRIVATE Kokkos::kokkos)
target_link_libraries(bench_create PRIVATE OpenSSL::SSL)
set_target_properties(bench_create PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS OFF
)