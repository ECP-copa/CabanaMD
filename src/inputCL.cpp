/****************************************************************************
 * Copyright (c) 2018-2019 by the Cabana authors                            *
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

#include <inputCL.h>
#include <system.h>

#include <fstream>
#include <iostream>
#include <typeinfo>

InputCL::InputCL()
{
    int proc_rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &proc_rank );
    do_print = proc_rank == 0;

    layout_type = AOSOA_6;
    nnp_layout_type = AOSOA_3;
    neighbor_type = NEIGH_2D;
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
            if ( do_print )
            {
                printf( "CabanaMD 0.1 \n\n" );
                printf( "Options:\n" );
                printf( "  -il [file] / --input-lammps [FILE]: Provide LAMMPS "
                        "input file\n" );
                printf( "  --layout-type [TYPE]:       Number of AoSoA for "
                        "particle properties\n                              "
                        "(1AOSOA, 2AOSOA, 6AOSOA)\n" );
                printf( "  --nnp-layout-type [TYPE]:   Number of AoSoA for "
                        "neural network potential particle properties\n        "
                        "                      "
                        "(1AOSOA, 3AOSOA)\n" );
                printf( "  --force-iteration [TYPE]:   Specify iteration style "
                        "for force calculations\n" );
                printf( "                              (NEIGH_FULL, "
                        "NEIGH_HALF)\n" );
                printf( "  --neigh-parallel [TYPE]:    Specify neighbor "
                        "parallelism and, if applicable, angular neighbor "
                        "parallelism\n" );
                printf( "                              (SERIAL, TEAM, "
                        "TEAM_VECTOR)\n" );
                printf( "  --neigh-type [TYPE]:        Specify Neighbor "
                        "Routines implementation \n" );
                printf(
                    "                              (NEIGH_2D, NEIGH_CSR)\n" );
                printf( "  --comm-type [TYPE]:         Specify Communication "
                        "Routines implementation \n" );
                printf( "                              (MPI, SERIAL)\n" );
                printf(
                    "  --dumpbinary [N] [PATH]:    Request that binary output "
                    "files PATH/output* be generated every N steps\n" );
                printf(
                    "                              (N = positive integer)\n" );
                printf( "                              (PATH = location of "
                        "directory)\n" );
                printf(
                    "  --correctness [N] [PATH] [FILE]:   Request that "
                    "correctness check against files PATH/output* be performed "
                    "every N steps, correctness data written to FILE\n" );
                printf(
                    "                              (N = positive integer)\n" );
                printf( "                              (PATH = location of "
                        "directory)\n" );
            }
        }
        // Read Lammps input deck
        else if ( ( strcmp( argv[i], "-il" ) == 0 ) ||
                  ( strcmp( argv[i], "--input-lammps" ) == 0 ) )
        {
            input_file = argv[i + 1];
            input_file_type = INPUT_LAMMPS;
            ++i;
        }

        // AoSoA layout type
        else if ( ( strcmp( argv[i], "--layout-type" ) == 0 ) )
        {
            if ( ( strcmp( argv[i + 1], "1AOSOA" ) == 0 ) )
            {
                layout_type = AOSOA_1;
            }
            if ( ( strcmp( argv[i + 1], "2AOSOA" ) == 0 ) )
            {
                layout_type = AOSOA_2;
            }
            if ( ( strcmp( argv[i + 1], "6AOSOA" ) == 0 ) )
            {
                layout_type = AOSOA_6;
            }
            ++i;
        }
        else if ( ( strcmp( argv[i], "--nnp-layout-type" ) == 0 ) )
        {
            if ( ( strcmp( argv[i + 1], "1AOSOA" ) == 0 ) )
            {
                layout_type = AOSOA_1;
            }
            if ( ( strcmp( argv[i + 1], "3AOSOA" ) == 0 ) )
            {
                layout_type = AOSOA_3;
            }
            ++i;
        }

        // Force Iteration Type Related
        else if ( ( strcmp( argv[i], "--force-iteration" ) == 0 ) )
        {
            set_force_iteration = true;
            if ( ( strcmp( argv[i + 1], "NEIGH_FULL" ) == 0 ) )
                force_iteration_type = FORCE_ITER_NEIGH_FULL;
            if ( ( strcmp( argv[i + 1], "NEIGH_HALF" ) == 0 ) )
                force_iteration_type = FORCE_ITER_NEIGH_HALF;
            ++i;
        }

        // Neighbor Type
        else if ( ( strcmp( argv[i], "--neigh-type" ) == 0 ) )
        {
            if ( ( strcmp( argv[i + 1], "NEIGH_2D" ) == 0 ) )
                neighbor_type = NEIGH_2D;
            if ( ( strcmp( argv[i + 1], "NEIGH_CSR" ) == 0 ) )
                neighbor_type = NEIGH_CSR;
            ++i;
        }

        // Neighbor parallel
        else if ( ( strcmp( argv[i], "--neigh-parallel" ) == 0 ) )
        {
            if ( ( strcmp( argv[i + 1], "SERIAL" ) == 0 ) )
                force_neigh_parallel_type = FORCE_PARALLEL_NEIGH_SERIAL;
            if ( ( strcmp( argv[i + 1], "TEAM" ) == 0 ) )
                force_neigh_parallel_type = FORCE_PARALLEL_NEIGH_TEAM;
            if ( ( strcmp( argv[i + 1], "TEAM_VECTOR" ) == 0 ) )
                force_neigh_parallel_type = FORCE_PARALLEL_NEIGH_VECTOR;
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
            if ( do_print )
                printf( "ERROR: Unknown command line argument: %s\n", argv[i] );
            exit( 1 );
        }
    }
}
