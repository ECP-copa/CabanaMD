
add_executable(MD main.cpp)
target_link_libraries(MD LINK_PUBLIC CabanaMD)
install(TARGETS MD DESTINATION bin)
