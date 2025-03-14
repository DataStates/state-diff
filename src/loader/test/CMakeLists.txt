add_executable(test_loader load_test.cpp)
target_include_directories(test_loader PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(test_loader PRIVATE asyncDataReader)
target_link_libraries(test_loader PRIVATE statediff-loader)
set_target_properties(test_loader PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS OFF
)

add_executable(test_loader_2files load_test_2readers.cpp)
target_include_directories(test_loader_2files PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(test_loader_2files PRIVATE asyncDataReader)
target_link_libraries(test_loader_2files PRIVATE statediff-loader)
set_target_properties(test_loader_2files PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS OFF
)