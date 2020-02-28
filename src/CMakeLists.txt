configure_file(CabanaMD_config.hpp.cmakein CabanaMD_config.hpp)

#------------------------------------------------------------

file(GLOB HEADERS_PUBLIC
     GLOB *.h force_types/*.h longrange_force_types/*.h
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

if( CabanaMD_ENABLE_LongRange )
  file(GLOB LR_FORCE_TYPES longrange_force_types/*.cpp)
  list(APPEND SOURCES ${LR_FORCE_TYPES})
endif()

install(FILES ${HEADERS_PUBLIC} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

#------------------------------------------------------------

add_library(CabanaMD ${SOURCES})

# Sources linking against CabanaMD will implicitly include these directories:
target_include_directories(CabanaMD PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/force_types ${CMAKE_CURRENT_SOURCE_DIR}/longrange_force_types)

#------------------------------------------------------------

target_link_libraries(CabanaMD Cabana::cabanacore Kokkos::kokkos MPI::MPI_CXX dl hwloc)

if(CabanaMD_ENABLE_NNP)
  target_include_directories(CabanaMD PUBLIC ${N2P2_DIR}/include)
  find_library(N2P2_LIB nnp PATHS ${N2P2_DIR}/lib NO_DEFAULT_PATH)
  target_link_libraries(CabanaMD ${N2P2_LIB})
endif()

if( CabanaMD_ENABLE_LongRange )
  target_include_directories(CabanaMD PUBLIC ${CAJITA_DIR}/include)
  find_library(CAJITA_LIB Cajita PATHS ${CAJITA_DIR}/lib NO_DEFAULT_PATH)

  target_include_directories(CabanaMD PUBLIC ${HEFFTE_DIR}/include)
  find_library(HEFFTE_LIB heffte PATHS ${HEFFTE_DIR}/lib NO_DEFAULT_PATH)

  find_package(FFTW)
  if(FFTW_FOUND)
    target_link_libraries(CabanaMD ${CAJITA_LIB} ${HEFFTE_LIB} ${FFTW_LIB} ${FFTWF_LIB})
  else()
    message(FATAL_ERROR "FFTW not found! Set CabanaMD_ENABLE_Longrange=OFF or install FFTW")
  endif()
endif()

install(TARGETS CabanaMD DESTINATION lib)

