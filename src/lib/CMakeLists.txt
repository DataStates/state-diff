# Define the statediff library
add_library(statediff SHARED 
  statediff.cpp 
  ${CMAKE_SOURCE_DIR}/src/common/merkle_tree.cpp)

# Alias for statediff
add_library(statediff::client ALIAS statediff)

#### Dependencies
# Boost Settings
set(Boost_USE_STATIC_LIBS OFF) 
set(Boost_USE_MULTITHREADED ON)  
set(Boost_USE_STATIC_RUNTIME OFF)
find_package(Boost 1.74.0 COMPONENTS serialization)

if(Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS})
else()

    include(FetchContent)

    # Declare the Boost content
    FetchContent_Declare(
        boost
        GIT_REPOSITORY https://github.com/boostorg/boost.git
        GIT_TAG        boost-1.86.0
        SOURCE_DIR     ${CMAKE_SOURCE_DIR}/extern/boost
        BINARY_DIR     ${CMAKE_BINARY_DIR}/boost
    )

    # Fetch the content
    FetchContent_MakeAvailable(boost)

    # Define the Boost build directory
    set(BOOST_BUILD_DIR "${CMAKE_SOURCE_DIR}/extern/boost/stage")
    set(BOOST_SERIALIZATION_LIB "${BOOST_BUILD_DIR}/lib/libboost_serialization.a")

    # Custom command to build only the serialization component of Boost with -fPIC
    if(NOT EXISTS "${BOOST_SERIALIZATION_LIB}")
        message(STATUS "Boost serialization library not found. Building Boost serialization library.")
        
        add_custom_target(boost_build ALL
            COMMAND ./bootstrap.sh --with-libraries=serialization
            COMMAND ./b2 cxxflags=-fPIC link=static threading=multi variant=release --with-serialization
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/extern/boost
            BYPRODUCTS ${BOOST_SERIALIZATION_LIB}
            COMMENT "Building Boost serialization library"
        )
        
        # Ensure Boost is built before linking
        add_dependencies(statediff boost_build)
    else()
        message(STATUS "Boost serialization library already built and found.")
    endif()

    # Set Boost paths for include and library
    set(BOOST_ROOT "${CMAKE_SOURCE_DIR}/extern/boost")
    set(BOOST_INCLUDE_DIR "${BOOST_ROOT}")
    set(BOOST_LIBRARY_DIR "${BOOST_BUILD_DIR}/lib")
    set(Boost_LIBRARIES "${BOOST_SERIALIZATION_LIB}")

    # Include Boost directories and link Boost libraries
    include_directories(${BOOST_INCLUDE_DIR})
    link_directories(${BOOST_LIBRARY_DIR})
endif()

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