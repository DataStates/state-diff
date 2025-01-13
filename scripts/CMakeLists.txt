add_executable(benchmark_thrpt read.cpp)
set_target_properties(benchmark_thrpt PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS OFF
)

add_executable(benchmark_create create.cpp)
target_link_libraries(benchmark_create PUBLIC Kokkos::kokkos)
target_link_libraries(benchmark_create PRIVATE OpenSSL::SSL)
target_link_libraries(benchmark_create PRIVATE statediff)
set_target_properties(benchmark_create PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS OFF
)

add_executable(benchmark_compare compare.cpp)
target_link_libraries(benchmark_compare PUBLIC Kokkos::kokkos)
target_link_libraries(benchmark_compare PRIVATE OpenSSL::SSL)
target_link_libraries(benchmark_compare PRIVATE statediff)
set_target_properties(benchmark_compare PROPERTIES
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
