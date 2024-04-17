cmake_minimum_required(VERSION 3.12)

project(state-diff VERSION 0.0.1 LANGUAGES CXX)

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")

#find_package(argparse REQUIRED)
find_package(Kokkos REQUIRED)
find_package(OpenSSL REQUIRED)

include(GNUInstallDirs)

option(ENABLE_TESTS "enable tests" OFF)

#==============================================================================
# Create library
#==============================================================================
add_library(StateDiff STATIC)
set_target_properties(StateDiff PROPERTIES PUBLIC_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/include/state_diff.hpp)

target_sources(StateDiff
  PRIVATE
    src/compare_tree_approach.cpp
    src/logger_state.cpp
    src/debug.cpp
)
target_include_directories(StateDiff 
  PUBLIC
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(StateDiff PRIVATE Kokkos::kokkos)
target_link_libraries(StateDiff PRIVATE uring)

include(CMakePackageConfigHelpers)
write_basic_package_version_file("${PROJECT_BINARY_DIR}/StateDiffConfigVersion.cmake"
                                 VERSION 0.0.0
                                 COMPATIBILITY SameMajorVersion
                                 )
install(TARGETS StateDiff
        EXPORT StateDiffTargets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} 
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} 
        BUNDLE DESTINATION ${CMAKE_INSTALL_BINDIR}
)

configure_package_config_file(
  "${PROJECT_SOURCE_DIR}/cmake/StateDiffConfig.cmake.in"
  "${PROJECT_BINARY_DIR}/StateDiffConfig.cmake"
  INSTALL_DESTINATION lib/cmake/StateDiff)

install(EXPORT StateDiffTargets FILE StateDiffTargets.cmake NAMESPACE Deduplicator:: DESTINATION lib/cmake/StateDiff)
install(FILES "${PROJECT_BINARY_DIR}/StateDiffConfigVersion.cmake"
              "${PROJECT_BINARY_DIR}/StateDiffConfig.cmake"
        DESTINATION lib/cmake/StateDiff)
install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/ DESTINATION include)

# Argparse
include(FetchContent)
FetchContent_Declare(
  argparse
  GIT_REPOSITORY https://github.com/p-ranav/argparse.git
)
FetchContent_MakeAvailable(argparse)
#add_library(argparse INTERFACE)

add_executable(data_generator src/data_generation.cpp)
target_include_directories(data_generator PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(data_generator PRIVATE Kokkos::kokkos)
target_link_libraries(data_generator PRIVATE argparse)
#target_include_directories(data_generator PRIVATE $<TARGET_PROPERTY:argparse,INTERFACE_INCLUDE_DIRECTORIES>)
set_target_properties(data_generator PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS OFF
)

add_executable(repro_test src/repro_test.cpp)
target_include_directories(repro_test PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(repro_test PRIVATE Kokkos::kokkos)
target_link_libraries(repro_test PRIVATE OpenSSL::SSL)
target_link_libraries(repro_test PRIVATE StateDiff)
target_link_libraries(repro_test PRIVATE argparse)
#target_include_directories(data_generator PRIVATE $<TARGET_PROPERTY:argparse,INTERFACE_INCLUDE_DIRECTORIES>)
set_target_properties(repro_test PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS OFF
)

#if(ENABLE_TESTS)
#  enable_testing()
#  add_subdirectory(test)
#endif(ENABLE_TESTS)
