#### Dependencies
# Boost
set(Boost_USE_STATIC_LIBS OFF) 
set(Boost_USE_MULTITHREADED ON)  
set(Boost_USE_STATIC_RUNTIME OFF) 
find_package(Boost 1.74.0 COMPONENTS serialization)

if(Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS})
else()
    include(FetchContent)

    FetchContent_Declare(
        boost
        GIT_REPOSITORY https://github.com/boostorg/boost.git
        GIT_TAG        boost-1.86.0
    )

    # Fetch the content
    FetchContent_MakeAvailable(boost)

    # Set Boost include and library paths
    set(BOOST_ROOT "${boost_SOURCE_DIR}")
    set(BOOST_LIBRARYDIR "${boost_BINARY_DIR}/stage/lib")
    set(Boost_INCLUDE_DIR "${BOOST_ROOT}")
    set(Boost_LIBRARY_DIR "${BOOST_LIBRARYDIR}")

    # Build boost with specific libraries
    add_custom_target(boost_build ALL
        COMMAND ./bootstrap.sh
        COMMAND ./b2 install link=static threading=multi variant=release
        WORKING_DIRECTORY ${boost_SOURCE_DIR}
    )
    include_directories(${Boost_INCLUDE_DIR})
    link_directories(${Boost_LIBRARY_DIR})
    message(STATUS "Boost was not found on system, so it was downloaded and built.")
endif()

add_library (statediff SHARED 
  statediff.cpp 
  # Merkle tree implementation
  ${CMAKE_SOURCE_DIR}/src/common/merkle_tree.cpp)

# add_library (statediff STATIC 
#   statediff.cpp 
#   # Merkle tree implementation
#   ${CMAKE_SOURCE_DIR}/src/common/merkle_tree.cpp)

add_library(statediff::client ALIAS statediff)

# Link the executable to the necessary libraries.
target_link_libraries(statediff PRIVATE Kokkos::kokkos)
target_link_libraries(statediff PUBLIC statediff-io)
target_link_libraries(statediff PUBLIC ${Boost_LIBRARIES})

# Add -fPIC
target_compile_options(statediff PRIVATE -fPIC)

set_target_properties(statediff PROPERTIES PUBLIC_HEADER ${CMAKE_SOURCE_DIR}/include/statediff.hpp)

target_include_directories(statediff PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# Install libraries
install (TARGETS statediff EXPORT statediffTargets
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

#install (DIRECTORY
#  ${PROJECT_SOURCE_DIR}/include
#  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
#)
