# Define the statediff library
add_library(statediff SHARED
  statediff.cpp
  merkle_tree.cpp
)

# Alias for statediff
add_library(statediff::client ALIAS statediff)

#### Library linking
target_link_libraries(statediff PUBLIC scissor::scissor Kokkos::kokkos)
set_target_properties(statediff PROPERTIES POSITION_INDEPENDENT_CODE ON)

set_target_properties(statediff PROPERTIES PUBLIC_HEADER ${CMAKE_SOURCE_DIR}/include/statediff.hpp)

target_include_directories(statediff PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/common>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CEREAL_INCLUDE_DIR}>
    $<INSTALL_INTERFACE:include>
)

install(DIRECTORY
        ${PROJECT_SOURCE_DIR}/src/common/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/common
)

install(DIRECTORY
        ${CEREAL_INCLUDE_DIR}/cereal
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(FILES
        ${PROJECT_SOURCE_DIR}/include/statediff/log.hpp
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/statediff
)

install (TARGETS statediff EXPORT statediffTargets
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
