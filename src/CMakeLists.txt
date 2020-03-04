#------------------------------------------------------------
# Set up layout and vector length options
#------------------------------------------------------------
if(NOT CabanaMD_LAYOUT_1AoSoA AND NOT CabanaMD_LAYOUT_6AoSoA AND NOT CabanaMD_LAYOUT_2AoSoA)
  set(CabanaMD_LAYOUT_6AoSoA ON)
endif()
if( CabanaMD_LAYOUT_1AoSoA )
  message( STATUS "Using layout:        1 combined AoSoA" )
elseif( CabanaMD_LAYOUT_6AoSoA )
  message( STATUS "Using layout:        6 separate AoSoA" )
elseif( CabanaMD_LAYOUT_2AoSoA )
  message( STATUS "Using layout:        2 AoSoA (XFType, VIdQ)" )
endif()

if(NOT CabanaMD_VECTORLENGTH)
  if(Kokkos_ENABLE_CUDA)
    set(CabanaMD_VECTORLENGTH 32)
  else()
    set(CabanaMD_VECTORLENGTH 1)
  endif()
endif()

if( CabanaMD_VECTORLENGTH )
  string(REGEX MATCH "(^[0-9]*[02468]$)" _ ${CabanaMD_VECTORLENGTH})
  if(NOT (${CMAKE_MATCH_1}) AND NOT (${CabanaMD_VECTORLENGTH} EQUAL 1))
    message( FATAL_ERROR "Vector length must be a factor of 2 or equal to 1" )
  endif()
endif()
message( STATUS "Using vector length: ${CabanaMD_VECTORLENGTH}" )

if(CabanaMD_ENABLE_NNP)
 if(NOT CabanaMD_LAYOUT_NNP_1AoSoA AND NOT CabanaMD_LAYOUT_NNP_3AoSoA)
    set(CabanaMD_LAYOUT_NNP_3AoSoA ON)
  endif()
  if( CabanaMD_LAYOUT_NNP STREQUAL 1AoSoA )
    message( STATUS "Using layout:        1 combined AoSoA (NNP)" )
  elseif( CabanaMD_LAYOUT_NNP STREQUAL 3AoSoA )
    message( STATUS "Using layout:        3 separate AoSoA (NNP)" )
  endif()

  if(NOT CabanaMD_VECTORLENGTH_NNP)
    if(Kokko_ENABLE_CUDA)
      set(CabanaMD_VECTORLENGTH_NNP 32)
    else()
      set(CabanaMD_VECTORLENGTH_NNP 1)
    endif()
  endif()

  if( CabanaMD_VECTORLENGTH_NNP )
    string(REGEX MATCH "(^[0-9]*[02468]$)" _ ${CabanaMD_VECTORLENGTH_NNP})
    if(NOT (${CMAKE_MATCH_1}) AND NOT (${CabanaMD_VECTORLENGTH_NNP} EQUAL 1))
      message( FATAL_ERROR "NNP vector length must be a factor of 2 or equal to 1" )
    endif()
  endif()
  message( STATUS "Using vector length: ${CabanaMD_VECTORLENGTH_NNP} (NNP)" )
endif()

#------------------------------------------------------------

configure_file(CabanaMD_config.hpp.cmakein CabanaMD_config.hpp @ONLY)

#------------------------------------------------------------

file(GLOB HEADERS_PUBLIC
     GLOB *.h force_types/*.h system_types/*.h
    )

file(GLOB SOURCES
     *.cpp
    )

if( CabanaMD_LAYOUT_1AoSoA )
  file(GLOB SYSTEM_TYPES system_types/system_1aosoa.cpp)
elseif( CabanaMD_LAYOUT_2AoSoA )
  file(GLOB SYSTEM_TYPES system_types/system_2aosoa.cpp)
elseif( CabanaMD_LAYOUT_6AoSoA )
  file(GLOB SYSTEM_TYPES system_types/system_6aosoa.cpp)
endif()

if(CabanaMD_ENABLE_NNP)
  file(GLOB FORCE_TYPES force_types/*.cpp)
  if( CabanaMD_LAYOUT_NNP_1AoSoA )
    file(GLOB SYSNNP_TYPES system_types/system_nnp_1aosoa.cpp)
  elseif( CabanaMD_LAYOUT_NNP_3AoSoA )
    file(GLOB SYSNNP_TYPES system_types/system_nnp_3aosoa.cpp)
  endif()
else()
  file(GLOB FORCE_TYPES force_types/force_lj*.cpp)
endif()

list(APPEND SOURCES ${FORCE_TYPES} ${SYSTEM_TYPES} ${SYSNNP_TYPES})

install(FILES ${HEADERS_PUBLIC} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/CabanaMD_config.hpp DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

#------------------------------------------------------------

add_library(CabanaMD ${SOURCES})

# Sources linking against CabanaMD will implicitly include these directories:
target_include_directories(CabanaMD PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
                                    ${CMAKE_CURRENT_SOURCE_DIR}/force_types
                                    ${CMAKE_CURRENT_SOURCE_DIR}/system_types
                                    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
                                    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>)

#------------------------------------------------------------

target_link_libraries(CabanaMD Cabana::cabanacore Kokkos::kokkos MPI::MPI_CXX dl hwloc)

if(CabanaMD_ENABLE_NNP)
  target_include_directories(CabanaMD PUBLIC ${N2P2_DIR}/include)
  find_library(N2P2_LIB nnp PATHS ${N2P2_DIR}/lib NO_DEFAULT_PATH)
  target_link_libraries(CabanaMD ${N2P2_LIB})
endif()

install(TARGETS CabanaMD DESTINATION lib)
