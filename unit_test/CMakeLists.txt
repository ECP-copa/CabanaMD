set(gtest_args --gtest_color=yes)

##--------------------------------------------------------------------------##
## On-node tests
##--------------------------------------------------------------------------##
macro(CabanaMD_add_tests)
  cmake_parse_arguments(CABANAMD_UNIT_TEST "MPI" "" "NAMES" ${ARGN})
  set(CABANAMD_UNIT_TEST_MPIEXEC_NUMPROCS 1 2)
  if(MPIEXEC_MAX_NUMPROCS GREATER 2)
    list(APPEND CABANAMD_UNIT_TEST_MPIEXEC_NUMPROCS ${MPIEXEC_MAX_NUMPROCS})
  endif()
  set(CABANAMD_UNIT_TEST_NUMTHREADS 1 2)
  set(CABANAMD_UNIT_TEST_MAIN unit_test_main.cpp)

  foreach(_device Serial OpenMP Cuda)
    if(Kokkos_ENABLE_${_device})
      string(TOUPPER ${_device} _uppercase_device)
      set(_dir ${CMAKE_CURRENT_BINARY_DIR}/${_uppercase_device})
      file(MAKE_DIRECTORY ${_dir})
      foreach(_test ${CABANAMD_UNIT_TEST_NAMES})
        set(_file ${_dir}/tst${_test}_${_uppercase_device}.cpp)
        file(WRITE ${_file}
          "#include <Test${_uppercase_device}_Category.hpp>\n"
          "#include <tst${_test}.hpp>\n"
          )
        set(_target ${_test}_test_${_uppercase_device})
        add_executable(${_target} ${_file} ${CABANAMD_UNIT_TEST_MAIN})
        target_include_directories(${_target} PRIVATE ${_dir} ${CMAKE_CURRENT_SOURCE_DIR})
        target_link_libraries(${_target} PRIVATE CabanaMD gtest CabanaMD)

        foreach(_np ${CABANAMD_UNIT_TEST_MPIEXEC_NUMPROCS})
          if(_device STREQUAL THREADS OR _device STREQUAL OpenMP)
            foreach(_thread ${CABANAMD_UNIT_TEST_NUMTHREADS})
              add_test(NAME ${_target}_${_np}_${_thread} COMMAND
                ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${_np} ${MPIEXEC_PREFLAGS}
                ${_target} ${MPIEXEC_POSTFLAGS} ${gtest_args} --kokkos-threads=${_thread})
            endforeach()
          else()
            add_test(NAME ${_target}_${_np} COMMAND
              ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${_np} ${MPIEXEC_PREFLAGS}
              ${_target} ${MPIEXEC_POSTFLAGsS} ${gtest_args})
          endif()
        endforeach()
      endforeach()
    endif()
  endforeach()
endmacro()

CabanaMD_add_tests(MPI NAMES Integrator)
