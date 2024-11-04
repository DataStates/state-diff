add_executable (statediff-extract-hacc statediff-extract-hacc.cpp)

# Install executables
install (TARGETS statediff-extract-hacc
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

add_executable (data_generator data_generation.cpp)
target_link_libraries(data_generator PRIVATE Kokkos::kokkos)
target_link_libraries(data_generator PRIVATE argparse)
target_include_directories(data_generator PRIVATE 
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
)

# Install executables
install (TARGETS data_generator
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
