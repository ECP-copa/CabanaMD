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
    bool timestepflag; // input timestep?

  public:
    InputCL commandline;
    t_System *system;

    bool _print_rank;
    int units_style;
    int lattice_style;
    std::vector<double> lattice_constant;
    std::vector<double> lattice_offset_x, lattice_offset_y, lattice_offset_z;
    std::vector<int> lattice_nx, lattice_ny, lattice_nz;
    std::vector<double> charge;

    int num_lattice;

    char *data_file;
    int data_file_type;

    std::string output_file;
    std::string error_file;

    double temperature_target;
    int temperature_seed;

    int integrator_type;
    int nsteps;

    int binning_type;

    int comm_type;
    int comm_exchange_rate;

    int force_type;

    int lrforce_type;
    int force_iteration_type;
    int lrforce_iteration_type;
    int force_neigh_parallel_type;

    T_F_FLOAT force_cutoff;
    T_F_FLOAT lrforce_cutoff;
    std::vector<std::vector<std::string>> force_coeff_lines;
    std::vector<std::vector<std::string>> lrforce_coeff_lines;

    T_F_FLOAT neighbor_skin;
    int neighbor_type;
    T_INT max_neigh_guess;

    int layout_type;
    int nnp_layout_type;

    int thermo_rate, dumpbinary_rate, correctness_rate;
    bool dumpbinaryflag, correctnessflag;
    char *dumpbinary_path, *reference_path, *correctness_file;
    std::string input_data_file;
    std::string output_data_file;
    bool read_data_flag = false;
    bool create_velocity_flag = false;
    bool write_data_flag = false;

    InputFile( InputCL cl, t_System *s );
    void read_file( std::ofstream &out, std::ofstream &err,
                    const char *filename = NULL );
    void read_lammps_file( std::ifstream &in, std::ofstream &out,
                           std::ofstream &err );
    void check_lammps_command( std::string line, std::ofstream &err );
    void create_lattices( Comm<t_System> *comm, std::ofstream &out );
    template <class t_HostSystem>
    void create_one_lattice( Comm<t_System> *comm, std::ofstream &out,
                             const int type, t_HostSystem &host_system );
    void create_velocities( Comm<t_System> *comm, std::ofstream &out );
};

#include <inputFile_impl.h>
#endif
