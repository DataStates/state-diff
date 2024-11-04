# Define the statediff library
add_library(statediff SHARED 
  statediff.cpp 
  merkle_tree.cpp
  # ${CMAKE_SOURCE_DIR}/src/common/merkle_tree.cpp
)

# Alias for statediff
add_library(statediff::client ALIAS statediff)

#### Dependencies
# Cereal Settings
set(CEREAL_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/extern/cereal/include")

if(NOT EXISTS "${CEREAL_INCLUDE_DIR}/cereal/cereal.hpp")
    message(STATUS "Cereal library not found. Downloading...")
    include(FetchContent)

    # Declare the Boost content
    FetchContent_Declare(
        cereal
        GIT_REPOSITORY https://github.com/USCiLab/cereal.git
        GIT_TAG        v1.3.2
        SOURCE_DIR     ${CMAKE_SOURCE_DIR}/extern/cereal
    )

    # Fetch the content
    FetchContent_MakeAvailable(cereal)
endif()

# Link the executable to the necessary libraries.
target_link_libraries(statediff PUBLIC statediff-io Kokkos::kokkos)

# Add -fPIC
# target_compile_options(statediff PUBLIC -fPIC)

set_target_properties(statediff PROPERTIES PUBLIC_HEADER ${CMAKE_SOURCE_DIR}/include/statediff.hpp)

target_include_directories(statediff PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CEREAL_INCLUDE_DIR}>
    $<INSTALL_INTERFACE:include>
    # $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# Install libraries

install(DIRECTORY 
        ${PROJECT_SOURCE_DIR}/src/common/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/common
)

install(DIRECTORY 
        ${CEREAL_INCLUDE_DIR}/cereal
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install (TARGETS statediff EXPORT statediffTargets
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
