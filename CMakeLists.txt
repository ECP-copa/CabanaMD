cmake_minimum_required(VERSION 3.14)
project(CabanaMD LANGUAGES CXX VERSION 0.1.0)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 14)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CXX_FLAGS)
  #release comes with -O3 by default
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CXX_FLAGS)

option(CMAKE_VERBOSE_MAKEFILE "Generate verbose Makefiles" OFF)
include(GNUInstallDirs)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

##---------------------------------------------------------------------------##
# Download and unpack googletest
##---------------------------------------------------------------------------##
include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        release-1.10.0
)
FetchContent_GetProperties(googletest)
if(NOT googletest_POPULATED)
  FetchContent_Populate(googletest)
  add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR})
endif()

# Prevent GoogleTest from overriding our compiler/linker options
# when building with Visual Studio
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

##---------------------------------------------------------------------------##
# Set up main options (inherit from Kokkos and Cabana CMake)
##---------------------------------------------------------------------------##
find_package(Cabana REQUIRED)

if( NOT Cabana_ENABLE_MPI )
  message( FATAL_ERROR "Cabana must be compiled with MPI" )
endif()
if( NOT Cabana_ENABLE_CAJITA )
  message( FATAL_ERROR "Cabana must be compiled with Cajita" )
endif()

##---------------------------------------------------------------------------##
# Macro for optional dependencies (taken from Cabana)
##---------------------------------------------------------------------------##
macro(CabanaMD_add_dependency)
  cmake_parse_arguments(CABANAMD_DEPENDENCY "" "PACKAGE;VERSION" "" ${ARGN})
  find_package( ${CABANAMD_DEPENDENCY_PACKAGE} ${CABANAMD_DEPENDENCY_VERSION} QUIET )
  string(TOUPPER "${CABANAMD_DEPENDENCY_PACKAGE}" CABANAMD_DEPENDENCY_OPTION )
  option(
    CabanaMD_REQUIRE_${CABANAMD_DEPENDENCY_OPTION}
    "Require CabanaMD to build with ${CABANAMD_DEPENDENCY_PACKAGE} support" ${CABANAMD_DEPENDENCY_PACKAGE}_FOUND)
  if(CabanaMD_REQUIRE_${CABANAMD_DEPENDENCY_OPTION})
    find_package( ${CABANAMD_DEPENDENCY_PACKAGE} ${CABANAMD_DEPENDENCY_VERSION} REQUIRED )
  endif()
  set(CabanaMD_ENABLE_${CABANAMD_DEPENDENCY_OPTION} ${${CABANAMD_DEPENDENCY_PACKAGE}_FOUND})
endmacro()

##---------------------------------------------------------------------------##
# Set up optional libraries
##---------------------------------------------------------------------------##
option(CabanaMD_ENABLE_NNP "Build CabanaMD with neural network potential (and n2p2 libnnp)" OFF)
if(NOT DEFINED N2P2_DIR)
  set(N2P2_DIR ~/n2p2/)
endif()

# todo(sschulz): All versions from 0.9.2 onwards should be fine.
#CabanaMD_add_dependency( PACKAGE "ALL" VERSION 0.9.2 )

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
## Unit tests
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
  file(GLOB_RECURSE FORMAT_SOURCES src/*.cpp src/*.h bin/*.cpp unit_test/*.cpp unit_test/*.hpp)
  add_custom_target(format
    COMMAND ${CLANG_FORMAT_EXECUTABLE} -i -style=file ${FORMAT_SOURCES}
    DEPENDS ${FORMAT_SOURCES})
endif()
