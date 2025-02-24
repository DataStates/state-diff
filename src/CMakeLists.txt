# argparse
include(FetchContent)
FetchContent_Declare(
    argparse
    GIT_REPOSITORY https://github.com/p-ranav/argparse.git
)
FetchContent_MakeAvailable(argparse)

#Liburing-loader
set(READER_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/src/extern/liburing-loader/include")
if(NOT EXISTS "${READER_INCLUDE_DIR}/async_data_reader.hpp")
    message(STATUS "Liburing-loader library not found. Downloading...")
    FetchContent_Declare(
        statediff-io
        GIT_REPOSITORY git@github.com:DataStates/liburing-loader.git
        GIT_TAG        rev-id2offset
        SOURCE_DIR     ${CMAKE_SOURCE_DIR}/src/extern/liburing-loader
    )
    FetchContent_Populate(statediff-io)
endif()

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
add_subdirectory(extern/liburing-loader)
# add_subdirectory(readers)
add_subdirectory(loader)
add_subdirectory(tools)
