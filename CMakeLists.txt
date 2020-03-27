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
option(CabanaMD_ENABLE_Threads "Build CabanaMD with Threads support" OFF)
option(CabanaMD_ENABLE_OpenMP "Build CabanaMD with OpenMP support" OFF)
option(CabanaMD_ENABLE_Cuda "Build CabanaMD with Cuda support" OFF)

set(CABANAMD_SUPPORTED_DEVICES Serial Threads OpenMP Cuda)
foreach(_device ${CABANAMD_SUPPORTED_DEVICES})
  if(CabanaMD_ENABLE_${_device})
    string(TOUPPER ${_device} _uppercase_device)
    kokkos_check( DEVICES ${_uppercase_device} )
  endif()
endforeach()

##---------------------------------------------------------------------------##
# Set up neural network options
##---------------------------------------------------------------------------##
option(CabanaMD_ENABLE_NNP "Build CabanaMD with neural network potential (and n2p2 libnnp)" OFF)
if(NOT DEFINED N2P2_DIR)
  set(N2P2_DIR ~/n2p2/)
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
## Clang format
##---------------------------------------------------------------------------##
find_package(CLANG_FORMAT)
if(CLANG_FORMAT_FOUND)
  file(GLOB_RECURSE FORMAT_SOURCES src/*.cpp src/*.h bin/*.cpp)
  add_custom_target(format
    COMMAND ${CLANG_FORMAT_EXECUTABLE} -i -style=file ${FORMAT_SOURCES}
    DEPENDS ${FORMAT_SOURCES})
endif()
