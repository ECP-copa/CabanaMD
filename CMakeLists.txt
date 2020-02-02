cmake_minimum_required(VERSION 3.9)
project(CabanaMD LANGUAGES CXX VERSION 0.1.0)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 11)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CXX_FLAGS)
  #release comes with -O3 by default
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CXX_FLAGS)

option(CMAKE_VERBOSE_MAKEFILE "Generate verbose Makefiles" OFF)
include(GNUInstallDirs)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

##---------------------------------------------------------------------------##
# Set up main options
##---------------------------------------------------------------------------##
option(CabanaMD_ENABLE_Serial "Build CabanaMD with Serial support" ON)
if( CabanaMD_ENABLE_Serial )
  message( STATUS "Using backend: Serial")
  add_definitions(-DCabanaMD_ENABLE_Serial=ON)
endif()

#option(CabanaMD_ENABLE_Pthread "Build CabanaMD with Pthread support" OFF)
#if( CabanaMD_ENABLE_Pthread )
#  add_definitions(-DCabanaMD_ENABLE_Pthread=ON)
#  find_package(Threads)
#endif()

option(CabanaMD_ENABLE_OpenMP "Build CabanaMD with OpenMP support" OFF)
if( CabanaMD_ENABLE_OpenMP )
  message( STATUS "Using backend: OpenMP")
  add_definitions(-DCabanaMD_ENABLE_OpenMP=ON)
  find_package(OpenMP)
  if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
  endif()
endif()

option(CabanaMD_ENABLE_Cuda "Build CabanaMD with Cuda support" OFF)
if( CabanaMD_ENABLE_Cuda )
  add_definitions(-DCabanaMD_ENABLE_Cuda=ON)
  find_package(CUDA)
  if (CUDA_FOUND)
    message("Found CUDA")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --expt-extended-lambda -g")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored")
  else()
    message( FATAL_ERROR "Requested CUDA but cant find it!")
  endif(CUDA_FOUND)
  message( STATUS "Using backend: CUDA")
endif()

find_package(MPI REQUIRED)

if(NOT DEFINED CABANA_DIR)
  set(CABANA_DIR ~/install/cabana/)
endif()

if(NOT DEFINED KOKKOS_DIR)
  set(KOKKOS_DIR ~/install/kokkos/)
endif()

##---------------------------------------------------------------------------##
# Set up layout and vector length options
##---------------------------------------------------------------------------##
if(NOT CabanaMD_LAYOUT)
  set(CabanaMD_LAYOUT 6AoSoA CACHE STRING "Cabana AoSoA particle property layouts; options are 1AoSoA 6AoSoA 2AoSoA (separate AoSoA denoted by underscores)")
endif()
if( CabanaMD_LAYOUT STREQUAL 1AoSoA )
  add_definitions(-DCabanaMD_LAYOUT_1AoSoA=ON)
  message( STATUS "Using layout: 1 combined AoSoA" )
elseif( CabanaMD_LAYOUT STREQUAL 6AoSoA )
  add_definitions(-DCabanaMD_LAYOUT_6AoSoA=ON)
  message( STATUS "Using layout: 6 separate AoSoA" )
elseif( CabanaMD_LAYOUT STREQUAL 2AoSoA )
  add_definitions(-DCabanaMD_LAYOUT_2AoSoA=ON)
  message( STATUS "Using layout: 2 AoSoA (XFType, VIdQ)" )
else()
  message( FATAL_ERROR "Layout option not supported: ${CabanaMD_LAYOUT}" )
endif()

if(NOT CabanaMD_VECTORLENGTH)
  if(CUDA_FOUND)
    set(CabanaMD_VECTORLENGTH 32)
  else()
    set(CabanaMD_VECTORLENGTH 1)
  endif()
endif()
message( STATUS "Using vector length: ${CabanaMD_VECTORLENGTH}" )
if( CabanaMD_VECTORLENGTH )
  string(REGEX MATCH "(^[0-9]*[02468]$)" _ ${CabanaMD_VECTORLENGTH})
  if(${CMAKE_MATCH_1})
    add_definitions(-DCabanaMD_VECTORLENGTH=${CabanaMD_VECTORLENGTH})
  else()
    if( ${CabanaMD_VECTORLENGTH} EQUAL 1 )
      add_definitions(-DCabanaMD_VECTORLENGTH=${CabanaMD_VECTORLENGTH})
    else()
      message( FATAL_ERROR "Vector length must be a factor of 2 or equal to 1" )
    endif()
  endif()
endif()

##---------------------------------------------------------------------------##
# Set up neural network options
##---------------------------------------------------------------------------##
option(CabanaMD_ENABLE_NNP "Build CabanaMD with neural network potential (and n2p2 libnnp)" OFF)
if( CabanaMD_ENABLE_NNP )
  add_definitions(-DCabanaMD_ENABLE_NNP=ON)

  if(NOT CabanaMD_LAYOUT_NNP)
    set(CabanaMD_LAYOUT_NNP 3AoSoA CACHE STRING "Cabana AoSoA NNP particle property layouts; options are 3AoSoA 1AoSoA (separate AoSoA denoted by underscores)")
  endif()
  if( CabanaMD_LAYOUT_NNP STREQUAL 1AoSoA )
    add_definitions(-DCabanaMD_LAYOUT_NNP_1AoSoA=ON)
    message( STATUS "Using layout: 1 combined AoSoA (NNP)" )
  elseif( CabanaMD_LAYOUT_NNP STREQUAL 3AoSoA )
    add_definitions(-DCabanaMD_LAYOUT_NNP_3AoSoA=ON)
    message( STATUS "Using layout: 3 separate AoSoA (NNP)" )
  else()
    message( FATAL_ERROR "NNP layout option not supported: ${CabanaMD_LAYOUT_NNP}" )
  endif()
endif()
if(NOT DEFINED N2P2_DIR)
  set(N2P2_DIR ~/install/n2p2/)
endif()

if(NOT CabanaMD_VECTORLENGTH_NNP)
  if(CUDA_FOUND)
    set(CabanaMD_VECTORLENGTH_NNP 32)
  else()
    set(CabanaMD_VECTORLENGTH_NNP 1)
  endif()
endif()
message( STATUS "Using vector length: ${CabanaMD_VECTORLENGTH_NNP} (NNP)" )
if( CabanaMD_VECTORLENGTH_NNP )
  string(REGEX MATCH "(^[0-9]*[02468]$)" _ ${CabanaMD_VECTORLENGTH_NNP})
  if(${CMAKE_MATCH_1})
    add_definitions(-DCabanaMD_VECTORLENGTH_NNP=${CabanaMD_VECTORLENGTH_NNP})
  else()
    if( ${CabanaMD_VECTORLENGTH_NNP} EQUAL 1 )
      add_definitions(-DCabanaMD_VECTORLENGTH_NNP=${CabanaMD_VECTORLENGTH_NNP})
    else()
      message( FATAL_ERROR "NNP vector length must be a factor of 2 or equal to 1" )
    endif()
  endif()
endif()

##---------------------------------------------------------------------------##
## Print the Git revision number to stdout
##---------------------------------------------------------------------------##
FIND_PACKAGE(Git)
IF(GIT_FOUND AND IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/.git)
    EXECUTE_PROCESS(
        COMMAND           ${GIT_EXECUTABLE} log --pretty=format:%H -n 1
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        OUTPUT_VARIABLE   CabanaMD_GIT_COMMIT_HASH
        )
ELSE()
    SET(CabanaMD_GIT_COMMIT_HASH "Not a git repository")
ENDIF()
MESSAGE(STATUS "CabanaMD Revision = '${CabanaMD_GIT_COMMIT_HASH}'")

##---------------------------------------------------------------------------##
## Build CabanaMD
##---------------------------------------------------------------------------##
include_directories(src src/force_types src/system_types)
if( CabanaMD_ENABLE_NNP )
  file(GLOB SOURCES_SR src/force_types/*.cpp)
  if( CabanaMD_LAYOUT_NNP STREQUAL 1AoSoA )
    file(GLOB SOURCES_SYSNNP src/system_types/system_nnp_1aosoa.cpp)
  elseif( CabanaMD_LAYOUT_NNP STREQUAL 3AoSoA )
    file(GLOB SOURCES_SYSNNP src/system_types/system_nnp_3aosoa.cpp)
  endif()
else()
  file(GLOB SOURCES_SR src/force_types/force_lj*.cpp)
endif()
if( CabanaMD_LAYOUT STREQUAL 1AoSoA )
  file(GLOB SOURCES_SYS src/system_types/system_1aosoa.cpp)
elseif( CabanaMD_LAYOUT STREQUAL 2AoSoA )
  file(GLOB SOURCES_SYS src/system_types/system_2aosoa.cpp)
elseif( CabanaMD_LAYOUT STREQUAL 6AoSoA )
  file(GLOB SOURCES_SYS src/system_types/system_6aosoa.cpp)
endif()

file(GLOB SOURCES src/*.cpp)
list(APPEND SOURCES ${SOURCES_SR} ${SOURCES_SYS} ${SOURCES_SYSNNP})

include_directories(SYSTEM ${KOKKOS_DIR}/include)
link_directories(${KOKKOS_DIR}/lib)
include_directories(${CABANA_DIR}/include)
link_directories(${CABANA_DIR}/lib64)
link_directories(${CABANA_DIR}/lib)
include_directories(${N2P2_DIR}/include)
link_directories(${N2P2_DIR}/lib)

add_executable(CabanaMD ${SOURCES})

if(CabanaMD_ENABLE_NNP)
  target_link_libraries(CabanaMD cabanacore kokkos dl hwloc MPI::MPI_CXX nnp)
else()
  target_link_libraries(CabanaMD cabanacore kokkos dl hwloc MPI::MPI_CXX)
endif()

install(TARGETS CabanaMD DESTINATION bin)

##---------------------------------------------------------------------------##
## Clang format
##---------------------------------------------------------------------------##
find_package(CLANG_FORMAT)
if(CLANG_FORMAT_FOUND)
  file(GLOB_RECURSE FORMAT_SOURCES src/*.cpp src/*.h)
  add_custom_target(format
    COMMAND ${CLANG_FORMAT_EXECUTABLE} -i -style=file ${FORMAT_SOURCES}
    DEPENDS ${FORMAT_SOURCES})
endif()
