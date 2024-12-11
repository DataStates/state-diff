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