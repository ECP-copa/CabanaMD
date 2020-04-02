#------------------------------------------------------------
# Set up vector length options
#------------------------------------------------------------
if(NOT CabanaMD_VECTORLENGTH)
  if(Kokkos_ENABLE_CUDA)
    set(CabanaMD_VECTORLENGTH 32)
  else()
    set(CabanaMD_VECTORLENGTH 1)
  endif()
endif()
message( STATUS "Using vector length: ${CabanaMD_VECTORLENGTH}" )

if(CabanaMD_ENABLE_NNP)
  if(NOT CabanaMD_VECTORLENGTH_NNP)
    if(Kokkos_ENABLE_CUDA)
      set(CabanaMD_VECTORLENGTH_NNP 32)
    else()
      set(CabanaMD_VECTORLENGTH_NNP 1)
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
                                    ${CMAKE_CURRENT_SOURCE_DIR}/system_types
                                    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
                                    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>)

#------------------------------------------------------------

target_link_libraries(CabanaMD Cabana::cabanacore Kokkos::kokkos MPI::MPI_CXX dl hwloc)

if(CabanaMD_ENABLE_NNP AND N2P2_DIR)
  target_include_directories(CabanaMD PUBLIC ${N2P2_DIR}/include)
  find_library(N2P2_LIB nnp PATHS ${N2P2_DIR}/lib NO_DEFAULT_PATH)
  target_link_libraries(CabanaMD ${N2P2_LIB})
endif()

install(TARGETS CabanaMD DESTINATION lib)
