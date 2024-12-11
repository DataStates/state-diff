add_definitions(-D__DEBUG)

add_library (statediff-loader SHARED
    data_loader.cpp
    host_cache.cpp
    device_cache.cpp
    storage.cpp
)

target_link_libraries(statediff-loader PUBLIC statediff-io)

##### Add include directories
target_include_directories (statediff-loader PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src/loader>
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src/common>
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
)

###### Install libraries
install (TARGETS statediff-loader EXPORT statediffTargets
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

###### Unit tests
add_subdirectory(test)
