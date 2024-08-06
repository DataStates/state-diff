add_library (statediff STATIC statediff.cpp)

# Link the executable to the necessary libraries.
target_include_directories(statediff 
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(statediff PRIVATE Kokkos::kokkos)
target_link_libraries(statediff PUBLIC statediff-io)

# Add -fPIC
# target_compile_options(statediff PRIVATE -fPIC)

# Install libraries
install (TARGETS statediff EXPORT statediffTargets
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install (DIRECTORY
  ${PROJECT_SOURCE_DIR}/include
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)