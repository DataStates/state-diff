add_library (statediff-io SHARED
    liburing_reader.cpp
    posix_reader.cpp
    mmap_reader.cpp
)

##### Find dependencies
# Liburing
# Add the external liburing CMake configuration
add_subdirectory(external)

# Link the liburing static library to your project
target_link_libraries(statediff-io PRIVATE statediff_liburing)
#target_include_directories(statediff-io PUBLIC ${LIBURING_INCLUDE_DIR})

# find_package(liburing REQUIRED)
add_definitions(-DWITH_LIBURING)

# ADIOS2
find_package(ADIOS2)
if(ADIOS2_FOUND)
    find_package(MPI REQUIRED)
    add_definitions(-DWITH_ADIOS2)
endif()

##### Add include directories
target_include_directories (statediff-io PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src/readers>
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/external/liburing/src/include>
  #$<BUILD_INTERFACE:${LIBURING_INCLUDE_DIR}>
)

###### Install libraries
install (TARGETS statediff-io EXPORT statediffTargets
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

###### Unit tests
# add_subdirectory(test)
