cmake_minimum_required(VERSION 3.9)
project(CabanaMD LANGUAGES CXX VERSION 0.1.0)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Release)
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
    message( FATAL_ERROR "Requested CUDA but cant find it!")
  endif(CUDA_FOUND)
endif()

if(NOT DEFINED CABANA_DIR)
  set(CABANA_DIR ~/install/cabana/)
endif()

if(NOT DEFINED KOKKOS_DIR)
  set(KOKKOS_DIR ~/install/kokkos/)
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
include_directories(src src/force_types src/libnnp)
file(GLOB SOURCES src/libnnp/*.cpp src/*.cpp src/force_types/*.cpp)

include_directories(SYSTEM ${KOKKOS_DIR}/include)
link_directories(${KOKKOS_DIR}/lib)
include_directories(${CABANA_DIR}/include)
link_directories(${CABANA_DIR}/lib64)
link_directories(${CABANA_DIR}/lib)

add_executable(CabanaMD ${SOURCES})
target_link_libraries(CabanaMD cabanacore kokkos dl)
install(TARGETS CabanaMD DESTINATION bin)
