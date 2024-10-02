add_executable(statediff-cli statediff_cli.cpp
  ${CMAKE_SOURCE_DIR}/src/common/merkle_tree.cpp
  ${CMAKE_SOURCE_DIR}/src/readers/io_uring_stream.cpp)

add_dependencies(statediff-cli statediff-io)

target_include_directories(statediff-cli PRIVATE ${CMAKE_SOURCE_DIR}/include)

target_link_libraries(statediff-cli PRIVATE statediff)
target_link_libraries(statediff-cli PRIVATE statediff-io)
target_link_libraries(statediff-cli PRIVATE argparse)
target_link_libraries(statediff-cli PRIVATE Kokkos::kokkos)
set_target_properties(statediff-cli PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
)

# Install executables
install (TARGETS statediff-cli
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
