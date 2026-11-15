add_executable(statediff-cli statediff_cli.cpp)

target_include_directories(statediff-cli PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(statediff-cli PUBLIC statediff)
target_link_libraries(statediff-cli PUBLIC argparse)
target_link_libraries(statediff-cli PUBLIC Kokkos::kokkos)
set_target_properties(statediff-cli PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
)

# Install executables
install (TARGETS statediff-cli
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
