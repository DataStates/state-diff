add_executable(test_serialize test_serialize.cpp)
target_include_directories(test_serialize PUBLIC ${CMAKE_SOURCE_DIR}/include ${Boost_INCLUDE_DIR})
target_link_libraries(test_serialize PRIVATE Kokkos::kokkos)
target_link_libraries(test_serialize PRIVATE OpenSSL::SSL)
target_link_libraries(test_serialize PUBLIC statediff)
set_target_properties(test_serialize PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS OFF
)

# add_executable(test_create_tree test_create_tree.cpp)
# target_include_directories(test_create_tree PUBLIC ${CMAKE_SOURCE_DIR}/include)
# target_link_libraries(test_create_tree PRIVATE Kokkos::kokkos)
# target_link_libraries(test_create_tree PRIVATE OpenSSL::SSL)
# target_link_libraries(test_create_tree PRIVATE statediff)
# set_target_properties(test_create_tree PROPERTIES
#     CXX_STANDARD 14
#     CXX_STANDARD_REQUIRED YES
#     CXX_EXTENSIONS OFF
# )

# add_executable(test_compare_tree_offline test_compare_tree_offline.cpp)
# target_include_directories(test_compare_tree_offline PUBLIC ${CMAKE_SOURCE_DIR}/include)
# target_link_libraries(test_compare_tree_offline PRIVATE Kokkos::kokkos)
# target_link_libraries(test_compare_tree_offline PRIVATE OpenSSL::SSL)
# target_link_libraries(test_compare_tree_offline PRIVATE statediff)
# target_link_libraries(test_compare_tree_offline PRIVATE adios2::cxx11)
# set_target_properties(test_compare_tree_offline PROPERTIES
#     CXX_STANDARD 14
#     CXX_STANDARD_REQUIRED YES
#     CXX_EXTENSIONS OFF
# )

# add_test(NAME test_compare_tree_offline COMMAND test_compare_tree_offline --kokkos-num-threads=2 --kokkos-map-device-id-by=mpi_rank)
# add_test(NAME test_create_tree COMMAND test_create_tree --kokkos-num-threads=2 --kokkos-map-device-id-by=mpi_rank)
add_test(NAME test_serialize COMMAND test_serialize --kokkos-num-threads=2 --kokkos-map-device-id-by=mpi_rank)

##### Make sure ADIOS2 is installed for testing
# ADIOS2
find_package(ADIOS2)
if(ADIOS2_FOUND)
    find_package(MPI REQUIRED)
    add_executable(test_serialize_adios test_serialize_adios.cpp)
    target_include_directories(test_serialize_adios PUBLIC ${CMAKE_SOURCE_DIR}/include ${ADIOS2_INCLUDE_DIR} ${Boost_INCLUDE_DIR})
    target_link_libraries(test_serialize_adios PRIVATE Kokkos::kokkos)
    target_link_libraries(test_serialize_adios PRIVATE OpenSSL::SSL)
    target_link_libraries(test_serialize_adios PUBLIC statediff)
    target_link_libraries(test_serialize_adios PRIVATE adios2::cxx11)
    set_target_properties(test_serialize_adios PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS OFF
    )

    # add_executable(test_compare_tree_adios_offline test_compare_tree_adios_offline.cpp)
    # target_include_directories(test_compare_tree_adios_offline PUBLIC ${CMAKE_SOURCE_DIR}/include ${ADIOS2_INCLUDE_DIR})
    # target_link_libraries(test_compare_tree_adios_offline PRIVATE Kokkos::kokkos)
    # target_link_libraries(test_compare_tree_adios_offline PRIVATE OpenSSL::SSL)
    # target_link_libraries(test_compare_tree_adios_offline PRIVATE statediff)
    # target_link_libraries(test_compare_tree_adios_offline PRIVATE adios2::cxx11)
    # set_target_properties(test_compare_tree_adios_offline PROPERTIES
    #     CXX_STANDARD 14
    #     CXX_STANDARD_REQUIRED YES
    #     CXX_EXTENSIONS OFF
    # )
    add_test(NAME test_serialize_adios COMMAND test_serialize_adios --kokkos-num-threads=2 --kokkos-map-device-id-by=mpi_rank)

    # add_test(NAME test_create_tree_adios COMMAND test_create_tree_adios --kokkos-num-threads=2 --kokkos-map-device-id-by=mpi_rank)
    # add_test(NAME test_compare_tree_adios_offline COMMAND test_compare_tree_adios_offline --kokkos-num-threads=2 --kokkos-map-device-id-by=mpi_rank)
endif()