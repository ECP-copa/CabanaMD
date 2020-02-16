
add_executable(LatNbrs lattice_neighbors.cpp)
target_link_libraries(LatNbrs LINK_PUBLIC CabanaMD)
add_test(NAME Lattice_Neighbors COMMAND
         LatNbrs -il ${CMAKE_CURRENT_SOURCE_DIR}/../input/in.lj)

