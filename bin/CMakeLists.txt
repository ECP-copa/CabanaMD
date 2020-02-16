
add_executable(cbnMD main.cpp)
target_link_libraries(cbnMD LINK_PUBLIC CabanaMD)
install(TARGETS cbnMD DESTINATION bin)
