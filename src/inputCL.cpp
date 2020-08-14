/****************************************************************************
 * Copyright (c) 2018-2020 by the Cabana authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Cabana library. Cabana is distributed under a   *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

//************************************************************************
//  ExaMiniMD v. 1.0
//  Copyright (2018) National Technology & Engineering Solutions of Sandia,
//  LLC (NTESS).
//
//  Under the terms of Contract DE-NA-0003525 with NTESS, the U.S. Government
//  retains certain rights in this software.
//
//  ExaMiniMD is licensed under 3-clause BSD terms of use: Redistribution and
//  use in source and binary forms, with or without modification, are
//  permitted provided that the following conditions are met:
//
//    1. Redistributions of source code must retain the above copyright notice,
//       this list of conditions and the following disclaimer.
//
//    2. Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//    3. Neither the name of the Corporation nor the names of the contributors
//       may be used to endorse or promote products derived from this software
//       without specific prior written permission.
//
//  THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY EXPRESS OR IMPLIED
//  WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
//  MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
//  IN NO EVENT SHALL NTESS OR THE CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
//  INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
//  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
//  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
//  HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
//  STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
//  IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
//  POSSIBILITY OF SUCH DAMAGE.
//
//************************************************************************

#include <Cabana_Core.hpp>

#include <inputCL.h>
#include <output.h>
#include <types.h>

#include <iostream>

InputCL::InputCL()
{
    device_type = SERIAL;
    layout_type = AOSOA_6;
    nnp_layout_type = AOSOA_3;
    neighbor_type = NEIGH_VERLET_2D;
    force_iteration_type = FORCE_ITER_NEIGH_FULL;
    set_force_iteration = false;
    force_neigh_parallel_type = FORCE_PARALLEL_NEIGH_SERIAL;
}

InputCL::~InputCL() {}

void InputCL::read_args( int argc, char *argv[] )
{
    for ( int i = 1; i < argc; i++ )
    {
        // Help command
        if ( ( strcmp( argv[i], "-h" ) == 0 ) ||
             ( strcmp( argv[i], "--help" ) == 0 ) )
        {
            log( std::cout, "CabanaMD 0.1\n", "Options:" );
            log( std::cout,
                 "  -il [FILE] (OR)\n"
                 "  --input-lammps [FILE]:    Provide LAMMPS input file\n" );
            log(
                std::cout,
                "  --device-type [TYPE]:     Kokkos device type to run ",
                "with\n",
                "                                (SERIAL, OPENMP, CUDA, HIP)" );
            log( std::cout,
                 "  --layout-type [TYPE]:     Number of AoSoA for particle ",
                 "properties\n",
                 "                                (1AOSOA, 2AOSOA, 6AOSOA)" );
            log( std::cout,
                 "  --nnp-layout-type [TYPE]: Number of AoSoA for neural ",
                 "network potential particle properties\n",
                 "                                (1AOSOA, 3AOSOA)" );
            log( std::cout,
                 "  --force-iteration [TYPE]: Specify iteration style for ",
                 "force calculations\n",
                 "                                (NEIGH_FULL, NEIGH_HALF)" );
            log(
                std::cout,
                "  --neigh-parallel [TYPE]:  Specify neighbor parallelism ",
                "and, if applicable, angular neighbor parallelism\n",
                "                                (SERIAL, TEAM, TEAM_VECTOR)" );
            log( std::cout,
                 "  --neigh-type [TYPE]:      Specify Neighbor Routines ",
                 "implementation\n",
                 "                                (VERLET_2D, VERLET_CSR, "
                 "TREE)" );
            log( std::cout,
                 "  --dumpbinary [N] [PATH]:  Request that binary output ",
                 "files PATH/output* be generated every N steps\n",
                 "                                (N = positive integer)\n",
                 "                                (PATH = location of ",
                 "directory)" );
            log( std::cout,
                 "  --correctness [N] [PATH] [FILE]: Request that "
                 "correctness check against files PATH/output* be performed "
                 "every N "
                 "steps, correctness data written to FILE\n",
                 "                                (N = positive integer)\n",
                 "                                (PATH = location of ",
                 "directory)\n" );
        }

        // Read Lammps input deck
        else if ( ( strcmp( argv[i], "-il" ) == 0 ) ||
                  ( strcmp( argv[i], "--input-lammps" ) == 0 ) )
        {
            input_file = argv[i + 1];
            input_file_type = INPUT_LAMMPS;
            ++i;
        }

        // Output file names
        else if ( ( strcmp( argv[i], "-o" ) == 0 ) ||
                  ( strcmp( argv[i], "--output-file" ) == 0 ) )
        {
            output_file = argv[i + 1];
            ++i;
        }
        else if ( ( strcmp( argv[i], "-e" ) == 0 ) ||
                  ( strcmp( argv[i], "--error-file" ) == 0 ) )
        {
            error_file = argv[i + 1];
            ++i;
        }

        // Kokkos device type
        else if ( ( strcmp( argv[i], "--device-type" ) == 0 ) )
        {
            if ( ( strcmp( argv[i + 1], "SERIAL" ) == 0 ) )
                device_type = SERIAL;
            else if ( ( strcmp( argv[i + 1], "OPENMP" ) == 0 ) )
                device_type = OPENMP;
            else if ( ( strcmp( argv[i + 1], "CUDA" ) == 0 ) )
                device_type = CUDA;
            else if ( ( strcmp( argv[i + 1], "HIP" ) == 0 ) )
                device_type = HIP;
            else
                log_err( std::cout, "Unknown commandline option: ", argv[i],
                         " ", argv[i + 1] );
            ++i;
        }

        // AoSoA layout type
        else if ( ( strcmp( argv[i], "--layout-type" ) == 0 ) )
        {
            if ( ( strcmp( argv[i + 1], "1AOSOA" ) == 0 ) )
                layout_type = AOSOA_1;
            else if ( ( strcmp( argv[i + 1], "2AOSOA" ) == 0 ) )
                layout_type = AOSOA_2;
            else if ( ( strcmp( argv[i + 1], "6AOSOA" ) == 0 ) )
                layout_type = AOSOA_6;
            else
                log_err( std::cout, "Unknown commandline option: ", argv[i],
                         " ", argv[i + 1] );
            ++i;
        }
        else if ( ( strcmp( argv[i], "--nnp-layout-type" ) == 0 ) )
        {
            if ( ( strcmp( argv[i + 1], "1AOSOA" ) == 0 ) )
                layout_type = AOSOA_1;
            else if ( ( strcmp( argv[i + 1], "3AOSOA" ) == 0 ) )
                layout_type = AOSOA_3;
            else
                log_err( std::cout, "Unknown commandline option: ", argv[i],
                         " ", argv[i + 1] );
            ++i;
        }

        // Force Iteration Type Related
        else if ( ( strcmp( argv[i], "--force-iteration" ) == 0 ) )
        {
            set_force_iteration = true;
            if ( ( strcmp( argv[i + 1], "NEIGH_FULL" ) == 0 ) )
                force_iteration_type = FORCE_ITER_NEIGH_FULL;
            else if ( ( strcmp( argv[i + 1], "NEIGH_HALF" ) == 0 ) )
                force_iteration_type = FORCE_ITER_NEIGH_HALF;
            else
                log_err( std::cout, "Unknown commandline option: ", argv[i],
                         " ", argv[i + 1] );
            ++i;
        }

        // Neighbor Type
        else if ( ( strcmp( argv[i], "--neigh-type" ) == 0 ) )
        {
            if ( ( strcmp( argv[i + 1], "VERLET_2D" ) == 0 ) )
                neighbor_type = NEIGH_VERLET_2D;
            else if ( ( strcmp( argv[i + 1], "VERLET_CSR" ) == 0 ) )
                neighbor_type = NEIGH_VERLET_CSR;
            else if ( ( strcmp( argv[i + 1], "TREE" ) == 0 ) )
                neighbor_type = NEIGH_TREE;
            else
                log_err( std::cout, "Unknown commandline option: ", argv[i],
                         " ", argv[i + 1] );
            ++i;
#ifndef Cabana_ENABLE_ARBORX
            if ( neighbor_type == NEIGH_TREE )
            {
                log_err( std::cout,
                         "ArborX requested, but not compiled with Cabana!" );
            }
#endif
        }

        // Neighbor parallel
        else if ( ( strcmp( argv[i], "--neigh-parallel" ) == 0 ) )
        {
            if ( ( strcmp( argv[i + 1], "SERIAL" ) == 0 ) )
                force_neigh_parallel_type = FORCE_PARALLEL_NEIGH_SERIAL;
            else if ( ( strcmp( argv[i + 1], "TEAM" ) == 0 ) )
                force_neigh_parallel_type = FORCE_PARALLEL_NEIGH_TEAM;
            else if ( ( strcmp( argv[i + 1], "TEAM_VECTOR" ) == 0 ) )
                force_neigh_parallel_type = FORCE_PARALLEL_NEIGH_VECTOR;
            else
                log_err( std::cout, "Unknown commandline option: ", argv[i],
                         " ", argv[i + 1] );
            ++i;
        }

        // Dump Binary
        else if ( ( strcmp( argv[i], "--dumpbinary" ) == 0 ) )
        {
            dumpbinary_rate = atoi( argv[i + 1] );
            dumpbinary_path = argv[i + 2];
            dumpbinaryflag = true;
            i += 2;
        }

        // Correctness Check
        else if ( ( strcmp( argv[i], "--correctness" ) == 0 ) )
        {
            correctness_rate = atoi( argv[i + 1] );
            reference_path = argv[i + 2];
            correctness_file = argv[i + 3];
            correctnessflag = true;
            i += 3;
        }

        else if ( ( strstr( argv[i], "--kokkos-" ) == NULL ) )
        {
            log_err( std::cout, "Unknown command line argument: ", argv[i] );
        }
    }
}
