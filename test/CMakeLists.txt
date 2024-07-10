add_executable(test_create_tree test_create_tree.cpp)
target_include_directories(test_create_tree PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(test_create_tree PRIVATE Kokkos::kokkos)
target_link_libraries(test_create_tree PRIVATE OpenSSL::SSL)
target_link_libraries(test_create_tree PRIVATE StateDiff)
set_target_properties(test_create_tree PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS OFF
)

add_executable(test_create_tree_adios test_create_tree_adios.cpp)
target_include_directories(test_create_tree_adios PUBLIC ${CMAKE_SOURCE_DIR}/include /home/kta7930/research/install/adios2/include)
target_link_libraries(test_create_tree_adios PRIVATE Kokkos::kokkos)
target_link_libraries(test_create_tree_adios PRIVATE OpenSSL::SSL)
target_link_libraries(test_create_tree_adios PRIVATE StateDiff)
target_link_libraries(test_create_tree_adios PRIVATE adios2::cxx11)
set_target_properties(test_create_tree_adios PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS OFF
)

add_executable(test_compare_tree_offline test_compare_tree_offline.cpp)
target_include_directories(test_compare_tree_offline PUBLIC ${CMAKE_SOURCE_DIR}/include /home/kta7930/research/install/adios2/include)
target_link_libraries(test_compare_tree_offline PRIVATE Kokkos::kokkos)
target_link_libraries(test_compare_tree_offline PRIVATE OpenSSL::SSL)
target_link_libraries(test_compare_tree_offline PRIVATE StateDiff)
target_link_libraries(test_compare_tree_offline PRIVATE adios2::cxx11)
set_target_properties(test_compare_tree_offline PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS OFF
)

add_executable(test_compare_tree_adios_offline test_compare_tree_adios_offline.cpp)
target_include_directories(test_compare_tree_adios_offline PUBLIC ${CMAKE_SOURCE_DIR}/include /home/kta7930/research/install/adios2/include)
target_link_libraries(test_compare_tree_adios_offline PRIVATE Kokkos::kokkos)
target_link_libraries(test_compare_tree_adios_offline PRIVATE OpenSSL::SSL)
target_link_libraries(test_compare_tree_adios_offline PRIVATE StateDiff)
target_link_libraries(test_compare_tree_adios_offline PRIVATE adios2::cxx11)
set_target_properties(test_compare_tree_adios_offline PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS OFF
)

add_test(NAME test_create_tree COMMAND test_create_tree --kokkos-num-threads=2 --kokkos-map-device-id-by=mpi_rank)
add_test(NAME test_create_tree_adios COMMAND test_create_tree_adios --kokkos-num-threads=2 --kokkos-map-device-id-by=mpi_rank)
add_test(NAME test_compare_tree_offline COMMAND test_compare_tree_offline --kokkos-num-threads=2 --kokkos-map-device-id-by=mpi_rank)
# add_test(NAME test_compare_tree_adios_offline COMMAND test_compare_tree_adios_offline --kokkos-num-threads=2 --kokkos-map-device-id-by=mpi_rank)
