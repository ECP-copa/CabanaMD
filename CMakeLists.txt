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

##---------------------------------------------------------------------------##
# Set up main options
##---------------------------------------------------------------------------##
option(CabanaMD_ENABLE_Serial "Build CabanaMD with Serial support" ON)
if( CabanaMD_ENABLE_Serial )
  add_definitions(-DCabanaMD_ENABLE_Serial=ON)
endif()

#option(CabanaMD_ENABLE_Pthread "Build CabanaMD with Pthread support" OFF)
#if( CabanaMD_ENABLE_Pthread )
#  add_definitions(-DCabanaMD_ENABLE_Pthread=ON)
#  find_package(Threads)
#endif()

option(CabanaMD_ENABLE_OpenMP "Build CabanaMD with OpenMP support" OFF)
if( CabanaMD_ENABLE_OpenMP )
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
    message( FATAL_ERROR "Requested CUDA, but cant find it!")
  endif(CUDA_FOUND)
endif()

find_package(MPI REQUIRED)

if(NOT DEFINED CABANA_DIR)
  set(CABANA_DIR ~/install/cabana/)
endif()

if(NOT DEFINED KOKKOS_DIR)
  set(KOKKOS_DIR ~/install/kokkos/)
endif()

option(CabanaMD_ENABLE_LongRange "Build CabanaMD with longrange forces (Ewald)" OFF)
if( CabanaMD_ENABLE_LongRange )
  add_definitions(-DCabanaMD_ENABLE_LongRange=ON)

  if(NOT DEFINED CAJITA_DIR)
    set(CAJITA_DIR ~/install/cajita/)
  endif()

  if(NOT CabanaMD_ENABLE_Cuda)
    set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
    find_package(FFTW3)
  endif()

  if(CabanaMD_ENABLE_Cuda)
  elseif(FFTW3_FOUND)
  else()
    message(FATAL_ERROR "Requsted long range, but FFTW not found and Cuda not enabled!")
  endif()
endif()

option(CabanaMD_ENABLE_NNP "Build CabanaMD with neural network potential (and n2p2 libnnp)" OFF)
if( CabanaMD_ENABLE_NNP )
  add_definitions(-DCabanaMD_ENABLE_NNP=ON)
endif()
if(NOT DEFINED N2P2_DIR)
  set(N2P2_DIR ~/install/n2p2/)
endif()

##---------------------------------------------------------------------------##
## Print the revision number to stdout
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
if( CabanaMD_ENABLE_LongRange )
  include_directories(src src/force_types src/longrange_force_types)
  file(GLOB SOURCES_LR src/longrange_force_types/*.cpp)
else()
  include_directories(src src/force_types)
endif()

if( CabanaMD_ENABLE_NNP )
  file(GLOB SOURCES_SR src/force_types/*.cpp)
else()
  file(GLOB SOURCES src/*.cpp src/force_types/force_lj*.cpp)
endif()

file(GLOB SOURCES src/*.cpp)
list(APPEND SOURCES ${SOURCES_SR} ${SOURCES_LR})


include_directories(SYSTEM ${KOKKOS_DIR}/include)
link_directories(${KOKKOS_DIR}/lib)
include_directories(${CABANA_DIR}/include)
link_directories(${CABANA_DIR}/lib64)
link_directories(${CABANA_DIR}/lib)
include_directories(${CAJITA_DIR}/include)
link_directories(${CAJITA_DIR}/lib)
include_directories(${N2P2_DIR}/include)
link_directories(${N2P2_DIR}/lib)

add_executable(CabanaMD ${SOURCES})

if( CabanaMD_ENABLE_LongRange )
  if(FFTW3_FOUND)
    target_link_libraries(CabanaMD Cajita FFTW3::FFTW3)
  elseif( CabanaMD_ENABLE_Cuda )
    target_link_libraries(CabanaMD Cajita ${CUDA_CUFFT_LIBRARIES})
  endif()
endif()

if(CabanaMD_ENABLE_NNP)
  target_link_libraries(CabanaMD nnp)
endif()

target_link_libraries(CabanaMD cabanacore kokkos dl hwloc MPI::MPI_CXX)

install(TARGETS CabanaMD DESTINATION bin)
