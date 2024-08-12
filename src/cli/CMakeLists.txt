# fetch latest argparse
include(FetchContent)
FetchContent_Declare(
    argparse
    GIT_REPOSITORY https://github.com/p-ranav/argparse.git
)
FetchContent_MakeAvailable(argparse)

#add_executable(statediff-cli statediff_cli.cpp)
add_executable(statediff-cli statediff_cli.cpp
  ${CMAKE_SOURCE_DIR}/src/common/merkle_tree.cpp)

target_include_directories(statediff-cli PUBLIC ${CMAKE_SOURCE_DIR}/include)
#target_include_directories(statediff-cli PUBLIC ${MPI_INCLUDE_PATH})
target_link_libraries(statediff-cli PUBLIC statediff)
target_link_libraries(statediff-cli PUBLIC statediff-io)
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
