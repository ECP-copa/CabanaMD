#------------------------------------------------------------
# Set up vector length options
#------------------------------------------------------------
macro(CabanaMD_layout)
  cmake_parse_arguments(LAYOUT "" "TYPE;PRINT" "ALLOWED" ${ARGN})
  set(CabanaMD_${LAYOUT_TYPE} 1 CACHE STRING "Number of AoSoA for${LAYOUT_PRINT} atom properties")
  set(_match 0)
  foreach(_l ${LAYOUT_ALLOWED})
    if(CabanaMD_${LAYOUT_TYPE} EQUAL _l)
      set(_match 1)
      message(STATUS "Number of AoSoA${LAYOUT_PRINT}: ${CabanaMD_${LAYOUT_TYPE}}")
    endif()
  endforeach()
  if(NOT _match)
    message(FATAL_ERROR "Layout${LAYOUT_PRINT} must be one of ${LAYOUT_ALLOWED}")
  endif()
endmacro()

macro(CabanaMD_vector_length)
  cmake_parse_arguments(VL "" "TYPE;LAYOUT;PRINT" "" ${ARGN})

  set(CabanaMD_${VL_TYPE} 1 CACHE STRING "Vector length for all${VL_PRINT} atom AoSoA(s) or semi-colon separated list of vector lengths")
  set(CabanaMD_${VL_TYPE}_0 1)

  list(LENGTH CabanaMD_${VL_TYPE} _nv_list)
  if(_nv_list EQUAL 1)
    math(EXPR _nv_loop "${VL_LAYOUT}-1")
    foreach(_v RANGE 0 ${_nv_loop})
      set(CabanaMD_${VL_TYPE}_${_v} ${CabanaMD_${VL_TYPE}})
    endforeach()
  elseif(_nv_list EQUAL ${VL_LAYOUT})
    set(_count 0)
    foreach(_v ${CabanaMD_${VL_TYPE}})
      set(CabanaMD_${VL_TYPE}_${_count} ${_v})
      math(EXPR _count "${_count}+1")
    endforeach()
  else()
    message(FATAL_ERROR "Vector length${VL_PRINT} list must be length 1 or match CabanaMD_LAYOUT length")
  endif()
  message(STATUS "Using vector length(s)${VL_PRINT}: ${CabanaMD_${VL_TYPE}}")
endmacro()

CabanaMD_layout(TYPE LAYOUT ALLOWED "1;2;6")
CabanaMD_vector_length(TYPE VECTORLENGTH LAYOUT ${CabanaMD_LAYOUT})

if(CabanaMD_ENABLE_NNP)
  CabanaMD_layout(TYPE LAYOUT_NNP PRINT " NNP" ALLOWED "1;3")
  CabanaMD_vector_length(TYPE VECTORLENGTH_NNP PRINT " NNP" LAYOUT ${CabanaMD_LAYOUT_NNP})

  if(NOT CabanaMD_MAXSYMMFUNC_NNP)
    set(CabanaMD_MAXSYMMFUNC_NNP 30)
  endif()
  message(STATUS "Maximum symmetry functions NNP: ${CabanaMD_MAXSYMMFUNC_NNP}")
endif()

#------------------------------------------------------------

configure_file(CabanaMD_config.hpp.cmakein CabanaMD_config.hpp @ONLY)

#------------------------------------------------------------

file(GLOB HEADERS_PUBLIC
     GLOB *.h force_types/*.h longrange_force_types/*.h neighbor_types/*.h system_types/*.h
    )

file(GLOB SOURCES
     *.cpp
    )

if(CabanaMD_ENABLE_NNP)
  file(GLOB FORCE_TYPES force_types/*.cpp)
else()
  file(GLOB FORCE_TYPES force_types/force_lj*.cpp)
endif()

list(APPEND SOURCES ${FORCE_TYPES} ${SYSTEM_TYPES} ${SYSNNP_TYPES})

if( Cabana_ENABLE_HEFFTE )
  file(GLOB LR_FORCE_TYPES longrange_force_types/*.cpp)
  list(APPEND SOURCES ${LR_FORCE_TYPES})
endif()

install(FILES ${HEADERS_PUBLIC} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/CabanaMD_config.hpp DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

#------------------------------------------------------------

add_library(CabanaMD ${SOURCES})

# Sources linking against CabanaMD will implicitly include these directories:
target_include_directories(CabanaMD PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
                                    ${CMAKE_CURRENT_SOURCE_DIR}/force_types
                                    ${CMAKE_CURRENT_SOURCE_DIR}/longrange_force_types
                                    ${CMAKE_CURRENT_SOURCE_DIR}/neighbor_types
                                    ${CMAKE_CURRENT_SOURCE_DIR}/system_types
                                    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
                                    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>)

#------------------------------------------------------------

target_link_libraries(CabanaMD Cabana::cabanacore Cabana::Cajita)

if(CabanaMD_ENABLE_NNP AND N2P2_DIR)
  target_include_directories(CabanaMD PUBLIC ${N2P2_DIR}/include)
  find_library(N2P2_LIB nnp PATHS ${N2P2_DIR}/lib NO_DEFAULT_PATH)
  target_link_libraries(CabanaMD ${N2P2_LIB})
endif()

if(Cabana_ENABLE_HEFFTE)
   target_link_libraries(CabanaMD Cabana::Cajita)
endif()

install(TARGETS CabanaMD DESTINATION lib)
