add_executable(test_case_create_tree test_case_create_tree.cpp)
target_include_directories(test_case_create_tree PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(test_case_create_tree PRIVATE Kokkos::kokkos)
target_link_libraries(test_case_create_tree PRIVATE OpenSSL::SSL)
target_link_libraries(test_case_create_tree PRIVATE StateDiff)
set_target_properties(test_case_create_tree PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS OFF
)

add_test(NAME test_case_create_tree COMMAND test_case_create_tree --kokkos-num-threads=2 --kokkos-map-device-id-by=mpi_rank)
