# argparse
include(FetchContent)
FetchContent_Declare(
    argparse
    GIT_REPOSITORY https://github.com/p-ranav/argparse.git
)
FetchContent_MakeAvailable(argparse)

#### Dependencies
# Cereal
set(CEREAL_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/src/extern/cereal/include")
if(NOT EXISTS "${CEREAL_INCLUDE_DIR}/cereal/cereal.hpp")
    message(STATUS "Cereal library not found. Downloading...")
    FetchContent_Declare(
        cereal
        GIT_REPOSITORY https://github.com/USCiLab/cereal.git
        GIT_TAG        v1.3.2
        SOURCE_DIR     ${CMAKE_SOURCE_DIR}/src/extern/cereal
    )
    set(BUILD_SANDBOX OFF CACHE BOOL "Disable Cereal tests")
    set(SKIP_PERFORMANCE_COMPARISON ON CACHE BOOL "Disable Cereal performance comparison")
    FetchContent_MakeAvailable(cereal)
endif()

add_subdirectory(cli)
add_subdirectory(lib)
add_subdirectory(tools)
