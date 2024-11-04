# fetch latest argparse
include(FetchContent)
FetchContent_Declare(
    argparse
    GIT_REPOSITORY https://github.com/p-ranav/argparse.git
)
FetchContent_MakeAvailable(argparse)
add_executable(reader_test reader_test.cpp)
target_include_directories(reader_test PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(reader_test PRIVATE Kokkos::kokkos)
target_link_libraries(reader_test PRIVATE statediff-io)
target_link_libraries(reader_test PRIVATE argparse)
set_target_properties(reader_test PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS OFF
)

add_test(NAME PosixReaderTest COMMAND reader_test -r posix)
add_test(NAME MMapReaderTest COMMAND reader_test -r mmap)
add_test(NAME IOUringReaderTest COMMAND reader_test -r io-uring)
add_test(NAME OldMMapReaderTest COMMAND reader_test -r old-mmap)
add_test(NAME OldIOUringReaderTest COMMAND reader_test -r old-io-uring)
