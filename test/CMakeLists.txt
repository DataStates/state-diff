add_executable(test_serialize test_serialize.cpp)
target_include_directories(test_serialize PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(test_serialize PUBLIC Kokkos::kokkos)
target_link_libraries(test_serialize PRIVATE OpenSSL::SSL)
target_link_libraries(test_serialize PUBLIC statediff)
set_target_properties(test_serialize PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS OFF
)

add_executable(test_compare test_compare.cpp)
target_include_directories(test_compare PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(test_compare PUBLIC Kokkos::kokkos)
target_link_libraries(test_compare PRIVATE OpenSSL::SSL)
target_link_libraries(test_compare PRIVATE statediff)
set_target_properties(test_compare PROPERTIES
   CXX_STANDARD 17
   CXX_STANDARD_REQUIRED YES
   CXX_EXTENSIONS OFF
)

add_executable(test_create_stream test_create_stream.cpp)
target_include_directories(test_create_stream PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(test_create_stream PUBLIC Kokkos::kokkos)
target_link_libraries(test_create_stream PRIVATE OpenSSL::SSL)
target_link_libraries(test_create_stream PRIVATE statediff)
set_target_properties(test_create_stream PROPERTIES
   CXX_STANDARD 17
   CXX_STANDARD_REQUIRED YES
   CXX_EXTENSIONS OFF
)

add_executable(loader_test test_data_loader.cpp)
target_include_directories(loader_test PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(loader_test PRIVATE Kokkos::kokkos)
target_link_libraries(loader_test PRIVATE statediff-io)
set_target_properties(loader_test PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS OFF
)

add_test(NAME test_serialize COMMAND test_serialize --kokkos-num-threads=2 --kokkos-map-device-id-by=mpi_rank)
add_test(NAME test_compare COMMAND test_compare --kokkos-num-threads=2 --kokkos-map-device-id-by=mpi_rank)
add_test(NAME test_create_stream COMMAND test_create_stream --kokkos-map-device-id-by=mpi_rank)
add_test(NAME loader_test COMMAND loader_test)
