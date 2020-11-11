#------------------------------------------------------------
# Set up vector length options
#------------------------------------------------------------
option(CabanaMD_LAYOUT "Number of AoSoA for atom properties." 6)
if(CabanaMD_LAYOUT EQUAL 6 OR CabanaMD_LAYOUT EQUAL 2 OR CabanaMD_LAYOUT EQUAL 1)
  message(STATUS "Number of AoSoA: ${CabanaMD_LAYOUT}")
else()
  message(FATAL_ERROR "Layout must be 1, 2, or 6.")
endif()

option(CabanaMD_VECTORLENGTH "Vector length for all atom AoSoA(s) or semi-colon separated list of vector lengths")
if(NOT CabanaMD_VECTORLENGTH)
  if(Kokkos_ENABLE_CUDA)
    set(CabanaMD_VECTORLENGTH_0 32)
    set(CabanaMD_VECTORLENGTH   32)
  else()
    set(CabanaMD_VECTORLENGTH_0 1)
    set(CabanaMD_VECTORLENGTH   1)
  endif()
  set(_nv_list 1)
else()
  list(LENGTH CabanaMD_VECTORLENGTH _nv_list)
  math(EXPR _nv_loop "${CabanaMD_LAYOUT}-1")
  if(_nv_list EQUAL 1)
    math(EXPR _nv_loop "${CabanaMD_LAYOUT}-1")
    #list(GET CabanaMD_VECTORLENGTH 0 _vl)
    foreach(_v RANGE 0 ${_nv_loop})
      set(CabanaMD_VECTORLENGTH_${_v} ${CabanaMD_VECTORLENGTH})
      message(STATUS ${_vl})
    endforeach()
  elseif(_nv_list EQUAL CabanaMD_LAYOUT)
    set(_count 0)
    foreach(_v ${CabanaMD_VECTORLENGTH})
      #list(GET CabanaMD_VECTORLENGTH _v _vl)
      set(CabanaMD_VECTORLENGTH_${_count} ${_v})
      message(STATUS "${_count} ${CabanaMD_VECTORLENGTH_${_count}}")
      math(EXPR _count "${_count}+1")
    endforeach()
  else()
    message(FATAL_ERROR "Vector length list must be 1 or match LAYOUT.")
  endif()
endif()

message( STATUS "Using vector length(s): ${CabanaMD_VECTORLENGTH}" )

if(CabanaMD_ENABLE_NNP)
  if(NOT CabanaMD_VECTORLENGTH_NNP)
    if(Kokkos_ENABLE_CUDA)
      set(CabanaMD_VECTORLENGTH_NNP 32)
    else()
      set(CabanaMD_VECTORLENGTH_NNP 1)
    endif()
  endif()
  message( STATUS "Using vector length: ${CabanaMD_VECTORLENGTH_NNP} (NNP)" )

  if(NOT CabanaMD_MAXSYMMFUNC_NNP)
    set(CabanaMD_MAXSYMMFUNC_NNP 30)
  endif()
  message( STATUS "Maximum symmetry functions: ${CabanaMD_MAXSYMMFUNC_NNP} (NNP)" )
endif()

#------------------------------------------------------------

configure_file(CabanaMD_config.hpp.cmakein CabanaMD_config.hpp @ONLY)

#------------------------------------------------------------

file(GLOB HEADERS_PUBLIC
     GLOB *.h force_types/*.h neighbor_types/*.h system_types/*.h
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

install(FILES ${HEADERS_PUBLIC} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/CabanaMD_config.hpp DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

#------------------------------------------------------------

add_library(CabanaMD ${SOURCES})

# Sources linking against CabanaMD will implicitly include these directories:
target_include_directories(CabanaMD PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
                                    ${CMAKE_CURRENT_SOURCE_DIR}/force_types
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

install(TARGETS CabanaMD DESTINATION lib)
