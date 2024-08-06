add_executable (statedif-extract-hacc statedif-extract-hacc.cpp)

# Install executables
install (TARGETS statedif-extract-hacc
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)