add_library (statediff SHARED 
  statediff.cpp 
  # Merkle tree implementation
  ${CMAKE_SOURCE_DIR}/src/common/merkle_tree.cpp)

add_library(statediff::client ALIAS statediff)

# Link the executable to the necessary libraries.
target_link_libraries(statediff PRIVATE Kokkos::kokkos)
target_link_libraries(statediff PUBLIC statediff-io)

# Add -fPIC
target_compile_options(statediff PRIVATE -fPIC)

set_target_properties(statediff PROPERTIES PUBLIC_HEADER ${CMAKE_SOURCE_DIR}/include/statediff.hpp)

target_include_directories(statediff PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# Install libraries
install (TARGETS statediff EXPORT statediffTargets
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

#install (DIRECTORY
#  ${PROJECT_SOURCE_DIR}/include
#  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
#)
