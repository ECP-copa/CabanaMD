/****************************************************************************
 * Copyright (c) 2018-2021 by the Cabana authors                            *
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

#ifndef INPUT_H
#define INPUT_H

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>

#include <comm_mpi.h>
#include <inputCL.h>
#include <output.h>
#include <system.h>
#include <types.h>

#include <array>
#include <fstream>
#include <vector>

// Class replicating LAMMPS Random velocity initialization with GEOM option
#define IA 16807
#define IM 2147483647
#define AM ( 1.0 / IM )
#define IQ 127773
#define IR 2836

class LAMMPS_RandomVelocityGeom
{
  private:
    int seed;

  public:
    KOKKOS_INLINE_FUNCTION
    LAMMPS_RandomVelocityGeom()
        : seed( 0 ){};

    KOKKOS_INLINE_FUNCTION
    double uniform()
    {
        int k = seed / IQ;
        seed = IA * ( seed - k * IQ ) - IR * k;
        if ( seed < 0 )
            seed += IM;
        double ans = AM * seed;
        return ans;
    }

    KOKKOS_INLINE_FUNCTION
    double gaussian()
    {
        double v1, v2, rsq;
        do
        {
            v1 = 2.0 * uniform() - 1.0;
            v2 = 2.0 * uniform() - 1.0;
            rsq = v1 * v1 + v2 * v2;
        } while ( ( rsq >= 1.0 ) || ( rsq == 0.0 ) );

        const double fac = sqrt( -2.0 * log( rsq ) / rsq );
        return v2 * fac;
    }

    KOKKOS_INLINE_FUNCTION
    void reset( int ibase, double *coord )
    {
        int i;

        char *str = (char *)&ibase;
        int n = sizeof( int );

        unsigned int hash = 0;
        for ( i = 0; i < n; i++ )
        {
            hash += str[i];
            hash += ( hash << 10 );
            hash ^= ( hash >> 6 );
        }

        str = (char *)coord;
        n = 3 * sizeof( double );
        for ( i = 0; i < n; i++ )
        {
            hash += str[i];
            hash += ( hash << 10 );
            hash ^= ( hash >> 6 );
        }

        hash += ( hash << 3 );
        hash ^= ( hash >> 11 );
        hash += ( hash << 15 );

        // keep 31 bits of unsigned int as new seed
        // do not allow seed = 0, since will cause hang in gaussian()
        seed = hash & 0x7ffffff;
        if ( !seed )
            seed = 1;

        // warm up the RNG
        for ( i = 0; i < 5; i++ )
            uniform();
    }
};

template <class t_System>
class InputFile
{
  private:
    bool timestepflag = false; // input timestep?
  public:
    InputCL commandline;
    t_System *system;

    bool _print_rank;

    // defaults match ExaMiniMD LJ example
    int units_style = UNITS_LJ;
    int lattice_style = LATTICE_FCC;
    double lattice_constant = 0.8442, lattice_offset_x = 0.0,
           lattice_offset_y = 0.0, lattice_offset_z = 0.0;

    struct Block
    {
        double xlo, xhi, ylo, yhi, zlo, zhi;
    };
    Block box = { 0.0, 40.0, 0.0, 40.0, 0.0, 40.0 };

    char *data_file;
    int data_file_type;

    std::string output_file;
    std::string error_file;

    double temperature_target = 1.4;
    int temperature_seed = 87287;

    int integrator_type = INTEGRATOR_NVE;
    int nsteps = 100;

    int binning_type = BINNING_LINKEDCELL;

    int comm_type = COMM_MPI;
    int comm_exchange_rate = 20;
    double comm_ghost_cutoff;

    int force_type = FORCE_LJ;
    int force_iteration_type;
    int force_neigh_parallel_type;

    T_F_FLOAT force_cutoff = 2.5;
    std::vector<std::vector<std::string>> force_coeff_lines;

    T_F_FLOAT neighbor_skin = 0.0;
    int neighbor_type = NEIGH_VERLET_2D;
    T_INT max_neigh_guess = 50;

    int layout_type;
    int nnp_layout_type;

    int thermo_rate = 10, dumpbinary_rate = 0, correctness_rate = 0;
    bool dumpbinaryflag = false, correctnessflag = false;
    char *dumpbinary_path, *reference_path, *correctness_file;
    std::string input_data_file;
    std::string output_data_file;
    bool read_data_flag = false;
    bool write_data_flag = false;
    bool write_vtk_flag = false;
    int vtk_rate;
    std::string vtk_file;

    InputFile( InputCL cl, t_System *s );
    void read_file( const char *filename = NULL );
    void read_lammps_file( std::ifstream &in, std::ofstream &out,
                           std::ofstream &err );
    void check_lammps_command( std::string line, std::ofstream &err );
    void create_lattice( Comm<t_System> *comm );
};

#include <inputFile_impl.h>
#endif
