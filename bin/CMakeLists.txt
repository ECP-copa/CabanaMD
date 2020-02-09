
link_directories(${CABANA_DIR}/lib64)
link_directories(${CABANA_DIR}/lib)

add_executable(MD main.cpp)
target_link_libraries(MD LINK_PUBLIC CabanaMD cabanacore dl hwloc MPI::MPI_CXX)
target_include_directories(MD PUBLIC ${CABANA_DIR}/include)

#include_directories(SYSTEM ${KOKKOS_DIR}/include)
#link_directories(${KOKKOS_DIR}/lib)

#if(CabanaMD_ENABLE_NNP)
#  target_include_directories(MD ${N2P2_DIR}/include)
#  target_link_directories(MD ${N2P2_DIR}/lib)
#  target_link_libraries(MD nnp)
#endif()

install(TARGETS MD DESTINATION bin)
