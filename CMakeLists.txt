cmake_minimum_required(VERSION 3.14)
project(statediff VERSION 0.0.1 LANGUAGES CXX)

##### Global settings
include(GNUInstallDirs)
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
add_definitions(-O2 -g -Wall -D__BENCHMARK -D__ASSERT -D__INFO)
set(CMAKE_CXX_STANDARD 23)

##### Find dependencies
find_package(Kokkos REQUIRED)
find_package(OpenSSL REQUIRED)

##### Enable testing
enable_testing()

##### Subdirectories
add_subdirectory(src)

##### Testing
# enable_testing()
# add_subdirectory(test)

###### Export sets and Configs
install(EXPORT statediffTargets
        FILE statediffTargets.cmake
        NAMESPACE statediff::
        DESTINATION share/statediff/cmake)

# Add the export set to the build directory
export(EXPORT statediffTargets NAMESPACE statediff:: FILE statediffTargets.cmake)

# Setup the configs
include(CMakePackageConfigHelpers)
configure_package_config_file(cmake/statediffConfig.cmake.in
                              ${CMAKE_CURRENT_BINARY_DIR}/statediffConfig.cmake
                              INSTALL_DESTINATION share/statediff/cmake)
write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/statediffConfigVersion.cmake
                                 COMPATIBILITY SameMajorVersion)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/statediffConfig.cmake
              ${CMAKE_CURRENT_BINARY_DIR}/statediffConfigVersion.cmake
        DESTINATION share/statediff/cmake)

