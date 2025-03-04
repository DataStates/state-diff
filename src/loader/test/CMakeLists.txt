# add_executable(single_loader_test test_single_loader.cpp)
# target_include_directories(single_loader_test PUBLIC ${CMAKE_SOURCE_DIR}/include)
# target_link_libraries(single_loader_test PRIVATE statediff-io)
# target_link_libraries(single_loader_test PRIVATE statediff-loader)
# set_target_properties(single_loader_test PROPERTIES
#     CXX_STANDARD 17
#     CXX_STANDARD_REQUIRED YES
#     CXX_EXTENSIONS OFF
# )

# add_executable(two_loader_test test_two_loader.cpp)
# target_include_directories(two_loader_test PUBLIC ${CMAKE_SOURCE_DIR}/include)
# target_link_libraries(two_loader_test PRIVATE statediff-io)
# target_link_libraries(two_loader_test PRIVATE statediff-loader)
# set_target_properties(two_loader_test PROPERTIES
#     CXX_STANDARD 17
#     CXX_STANDARD_REQUIRED YES
#     CXX_EXTENSIONS OFF
# )

add_executable(test_loader load_test.cpp)
target_include_directories(test_loader PUBLIC ${CMAKE_SOURCE_DIR}/include)
# target_link_libraries(test_loader PRIVATE statediff-io)
target_link_libraries(test_loader PRIVATE asyncDataReader)
target_link_libraries(test_loader PRIVATE statediff-loader)
set_target_properties(test_loader PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS OFF
)