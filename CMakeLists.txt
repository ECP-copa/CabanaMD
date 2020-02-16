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
# Set up main options (inherit from Kokkos and Cabana CMake)
##---------------------------------------------------------------------------##
find_package(Cabana REQUIRED)
find_package(MPI REQUIRED)

option(CabanaMD_ENABLE_Serial "Build CabanaMD with Serial support" ON)
if( CabanaMD_ENABLE_Serial )
  if( NOT Kokkos_ENABLE_SERIAL )
    message( FATAL_ERROR "Requested SERIAL, but not enabled in Kokkos!" )
  endif()
  add_definitions(-DCabanaMD_ENABLE_Serial=ON)
endif()

option(CabanaMD_ENABLE_Pthread "Build CabanaMD with Pthread support" OFF)
if( CabanaMD_ENABLE_Pthread )
  if( NOT Kokkos_ENABLE_PTHREAD )
    message( FATAL_ERROR "Requested PTHREAD, but not enabled in Kokkos!" )
  endif()
  add_definitions(-DCabanaMD_ENABLE_Pthread=ON)
endif()

option(CabanaMD_ENABLE_OpenMP "Build CabanaMD with OpenMP support" OFF)
if( CabanaMD_ENABLE_OpenMP )
  if( NOT Kokkos_ENABLE_OPENMP )
    message( FATAL_ERROR "Requested OPENMP, but not enabled in Kokkos!" )
  endif()

  find_package(OpenMP)
  if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
  endif()

  add_definitions(-DCabanaMD_ENABLE_OpenMP=ON)
endif()

option(CabanaMD_ENABLE_Cuda "Build CabanaMD with Cuda support" OFF)
if( CabanaMD_ENABLE_Cuda )
  if( NOT Kokkos_ENABLE_CUDA )
    message( FATAL_ERROR "Requested CUDA, but not enabled in Kokkos!" )
  endif()

  find_package(CUDA REQUIRED)
  if (CUDA_FOUND)
    message("Found CUDA")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --expt-extended-lambda -g")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored")
  else()
    message( FATAL_ERROR "Requested CUDA but cant find it!")
  endif(CUDA_FOUND)

  add_definitions(-DCabanaMD_ENABLE_Cuda=ON)
endif()

##---------------------------------------------------------------------------##
# Set up optional libraries
##---------------------------------------------------------------------------##

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

add_subdirectory(src)
add_subdirectory(bin)

##---------------------------------------------------------------------------##
## Enable Tests?
##---------------------------------------------------------------------------##

option(CabanaMD_ENABLE_TESTING "Build tests" OFF)
if(CabanaMD_ENABLE_TESTING)
  enable_testing()
  add_subdirectory(unit_test)
endif()

##---------------------------------------------------------------------------##
## Pretty-printing wizardry:
##---------------------------------------------------------------------------##
find_package(CLANG_FORMAT)
if(CLANG_FORMAT_FOUND)
  file(GLOB_RECURSE FORMAT_SOURCES src/*.cpp src/*.h)
  add_custom_target(format
    COMMAND ${CLANG_FORMAT_EXECUTABLE} -i -style=file ${FORMAT_SOURCES}
    DEPENDS ${FORMAT_SOURCES})
endif()
