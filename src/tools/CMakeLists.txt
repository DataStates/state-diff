add_executable (statediff-extract-hacc statediff-extract-hacc.cpp)

# Install executables
install (TARGETS statediff-extract-hacc
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
