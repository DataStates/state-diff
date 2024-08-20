add_library (statediff-io SHARED
    liburing_reader.cpp
    posix_reader.cpp
    mmap_reader.cpp
)

##### Find dependencies
# Liburing
find_library(LIBURING_LIBRARY NAMES uring)

# If liburing is not found, download and build it
if(NOT LIBURING_LIBRARY)
    include(ExternalProject)

    # Add external project for liburing
    ExternalProject_Add(
      liburing
      GIT_REPOSITORY https://github.com/axboe/liburing.git
      GIT_TAG liburing-2.6
      PREFIX ${CMAKE_BINARY_DIR}/liburing
      CONFIGURE_COMMAND ./configure --prefix=${CMAKE_BINARY_DIR}/liburing/install
      BUILD_COMMAND make
      INSTALL_COMMAND make install
      UPDATE_COMMAND ""
      BUILD_IN_SOURCE 1
    #   STAMP_DIR ${CMAKE_BINARY_DIR}/liburing-stamp
    )

    # Specify the install directory
    set(LIBURING_INSTALL_DIR ${CMAKE_BINARY_DIR}/liburing/install)

    # Manually set the include directories and libraries
    set(LIBURING_INCLUDE_DIR ${LIBURING_INSTALL_DIR}/include)
    set(LIBURING_LIBRARY ${LIBURING_INSTALL_DIR}/lib/liburing.so)

    # Ensure liburing is built before proceeding
    add_dependencies(statediff-io liburing)
    target_link_libraries(statediff-io uring)
else()
    # If found, set the include directories and libraries
    find_path(LIBURING_INCLUDE_DIR NAMES liburing.h)
    set(LIBURING_LIBRARY uring)
    target_link_libraries(statediff-io uring)
endif()

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
  $<BUILD_INTERFACE:${LIBURING_INCLUDE_DIR}>
)

###### Install libraries
install (TARGETS statediff-io EXPORT statediffTargets
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
