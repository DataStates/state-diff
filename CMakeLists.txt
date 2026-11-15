cmake_minimum_required(VERSION 3.14)
project(statediff VERSION 0.0.1 LANGUAGES CXX)

##### Global settings
include(GNUInstallDirs)
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
set(CMAKE_CXX_STANDARD 20)

# The statediff shared library links the static scissor archive.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

##### Find dependencies

# Threads and OpenSSL are standard system libraries available on all HPC
# platforms. No auto-fetch fallback is needed.
find_package(Threads REQUIRED)
find_package(OpenSSL REQUIRED)

# Kokkos must be pre-installed (via Spack, module, or manual build).
# Pass -DKokkos_DIR=<prefix>/lib/cmake/Kokkos if not on CMAKE_PREFIX_PATH.
# No auto-fetch: Kokkos is a large framework with platform-specific tuning
# that should be configured for the target system before building state-diff.
find_package(Kokkos REQUIRED)

# scissor-io (standalone SCISSOR library: provides the I/O readers, the
# Loader and the structured logger, and links liburing transitively).
# Pass -DCMAKE_PREFIX_PATH=<scissor prefix> to use an installed copy,
# otherwise it is fetched automatically from:
#   https://github.com/DataStates/scissor-io.git  @  v0.0.1
find_package(scissor 0.0.1 QUIET)
if(NOT scissor_FOUND)
    message(STATUS "scissor not found via find_package, fetching https://github.com/DataStates/scissor-io.git @ v0.0.1")
    include(FetchContent)
    set(SCISSOR_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
    set(SCISSOR_ENABLE_TOOLS OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(scissor
        GIT_REPOSITORY https://github.com/DataStates/scissor-io.git
        GIT_TAG        v0.0.1
        GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(scissor)
else()
    message(STATUS "Found scissor ${scissor_VERSION} (pre-installed)")
endif()

##### Enable testing
enable_testing()

##### Subdirectories
add_subdirectory(src)

##### State-diff Testing
add_subdirectory(test)

##### State-diff Benchmarking
add_subdirectory(benchmark)
add_subdirectory(scripts)

###### Export sets and Configs
install(EXPORT statediffTargets
        FILE statediffTargets.cmake
        NAMESPACE statediff::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/statediff)

# Add the export set to the build directory
export(EXPORT statediffTargets NAMESPACE statediff:: FILE statediffTargets.cmake)

# Setup the configs
include(CMakePackageConfigHelpers)
configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/statediffConfig.cmake.in
                              ${CMAKE_CURRENT_BINARY_DIR}/statediffConfig.cmake
                              INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/statediff)
write_basic_package_version_file(
        ${CMAKE_CURRENT_BINARY_DIR}/statediffConfigVersion.cmake
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY AnyNewerVersion
)

###### Install header files
install(DIRECTORY 
        ${PROJECT_SOURCE_DIR}/include/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(FILES 
        ${CMAKE_CURRENT_BINARY_DIR}/statediffConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/statediffConfigVersion.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/statediff)

