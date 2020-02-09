configure_file(CabanaMD_config.hpp.cmakein CabanaMD_config.hpp)

#------------------------------------------------------------

file(GLOB HEADERS_PUBLIC
     GLOB *.h force_types/*.h
    )

file(GLOB SOURCES
     *.cpp
    )

if(CabanaMD_ENABLE_NNP)
  file(GLOB FORCE_TYPES force_types/*.cpp)
else()
  file(GLOB FORCE_TYPES force_types/force_lj*.cpp)
endif()
list(APPEND SOURCES ${FORCE_TYPES})

install(FILES ${HEADERS_PUBLIC} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

#------------------------------------------------------------

add_library(CabanaMD ${SOURCES})

# Sources linking against CabanaMD will implicitly include these directories:
target_include_directories(CabanaMD PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/force_types)

#------------------------------------------------------------

target_include_directories(CabanaMD PUBLIC ${CABANA_DIR}/include)
link_directories(${CABANA_DIR}/lib64)
link_directories(${CABANA_DIR}/lib)
target_link_libraries(CabanaMD cabanacore Kokkos::kokkos dl hwloc MPI::MPI_CXX)

if(CabanaMD_ENABLE_NNP)
  target_include_directories(CabanaMD PUBLIC ${N2P2_DIR}/include)
  link_directories(${N2P2_DIR}/lib)
  target_link_libraries(CabanaMD nnp)
endif()

install(TARGETS CabanaMD DESTINATION lib)

