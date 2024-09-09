add_executable(reader_test reader_test.cpp)
target_include_directories(reader_test PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(reader_test PRIVATE Kokkos::kokkos)
target_link_libraries(reader_test PRIVATE statediff-io)
set_target_properties(reader_test PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS OFF
)

add_test(NAME ReaderTest COMMAND reader_test --kokkos-num-threads=1)
