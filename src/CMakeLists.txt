# argparse
include(FetchContent)
FetchContent_Declare(
    argparse
    GIT_REPOSITORY https://github.com/p-ranav/argparse.git
)
FetchContent_MakeAvailable(argparse)

add_subdirectory(cli)
add_subdirectory(lib)
add_subdirectory(readers)
add_subdirectory(loader)
add_subdirectory(tools)
