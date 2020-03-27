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

/* ----------------------------------------------------------------------
   miniMD is a simple, parallel molecular dynamics (MD) code.   miniMD is
   an MD microapplication in the Mantevo project at Sandia National
   Laboratories ( http://www.mantevo.org ). The primary
   authors of miniMD are Steve Plimpton (sjplimp@sandia.gov) , Paul Crozier
   (pscrozi@sandia.gov) and Christian Trott (crtrott@sandia.gov).

   Copyright (2008) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This library is free software; you
   can redistribute it and/or modify it under the terms of the GNU Lesser
   General Public License as published by the Free Software Foundation;
   either version 3 of the License, or (at your option) any later
   version.

   This library is distributed in the hope that it will be useful, but
   WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with this software; if not, write to the Free Software
   Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
   USA.  See also: http://www.gnu.org/licenses/lgpl.txt .

   Please read the accompanying LICENSE_MINIMD file.
---------------------------------------------------------------------- */

#include <comm_mpi.h>
#include <system.h>
#include <types.h>

#include <Cabana_Core.hpp>

#include <cstring>
#include <fstream>
#include <iostream>
#include <string>

void skip_empty( std::ifstream &file, std::string &line )
{
    // skip blank lines
    while ( 1 )
    {
        std::getline( file, line );
        if ( !line.empty() )
            break;
    }
}

std::string read_lammps_parse_keyword( std::ifstream &file )
{
    // Read up to first non-blank line plus 1 following
    std::string line;
    std::string keyword = "";
    while ( 1 )
    {
        // exit if end-of-file is encountered
        if ( file.eof() )
            break;
        std::getline( file, line );
        // get substring prior to # (ignore comments)
        line = line.substr( 0, line.find( '#' ) );
        // for non-blank lines, skip leading whitespaces and read word
        std::size_t pos = line.find_first_not_of( " \r\t\n" );
        if ( pos != std::string::npos )
        {
            keyword = line;
            int end = keyword.find_last_not_of( " " ) + 1;
            if ( end < line.length() )
                keyword.erase( end );
            if ( ( keyword.compare( "Atoms" ) == 0 ) ||
                 ( keyword.compare( "Velocities" ) == 0 ) ||
                 ( keyword.compare( "Masses" ) == 0 ) ||
                 ( keyword.compare( "Pair Coeffs" ) == 0 ) )
                return keyword;
            else
            {
                std::cout << "ERROR: Unknown identifier in data file: "
                          << keyword.data() << std::endl;
                return keyword;
            }
        }
        else
            continue;
    }
    return keyword;
}

template <class t_System>
void read_lammps_header( std::ifstream &file, t_System *s )
{
    std::string line;
    // skip 1st line of file
    if ( !std::getline( file, line ) )
        std::cout
            << "ERROR: could not read line from file. Please check for a valid "
               "file and ensure that file path is less than 32 characters"
            << std::endl;
    while ( 1 )
    {

        std::getline( file, line );
        // skip blank lines
        std::size_t pos = line.find_first_not_of( " \r\t\n" );
        if ( pos == std::string::npos )
            continue;

        // ignore anything after # by getting substring till # (takes care of
        // comment lines too)
        line = line.substr( 0, line.find( '#' ) );

        int natoms, ntypes;
        double xlo, xhi, ylo, yhi, zlo, zhi;
        const char *temp = line.data(); // convert to C-string for sscanf
                                        // utility
        // search line for header keyword and set corresponding variable
        if ( line.find( "atoms" ) != std::string::npos )
        {
            std::sscanf( temp, "%i", &natoms );
            s->N = natoms;
        }
        else if ( line.find( "atom types" ) != std::string::npos )
        {
            std::sscanf( temp, "%i", &ntypes );
            s->ntypes = ntypes;
        }
        else if ( line.find( "xlo xhi" ) != std::string::npos )
        {
            std::sscanf( temp, "%lg %lg", &xlo, &xhi );
            s->domain_lo_x = xlo;
            s->domain_hi_x = xhi;
            s->domain_x = xhi - xlo;
        }
        else if ( line.find( "ylo yhi" ) != std::string::npos )
        {
            std::sscanf( temp, "%lg %lg", &ylo, &yhi );
            s->domain_lo_y = ylo;
            s->domain_hi_y = yhi;
            s->domain_y = yhi - ylo;
        }
        else if ( line.find( "zlo zhi" ) != std::string::npos )
        {
            std::sscanf( temp, "%lg %lg", &zlo, &zhi );
            s->domain_lo_z = zlo;
            s->domain_hi_z = zhi;
            s->domain_z = zhi - zlo;
            break;
        }
    }
}

template <class t_System>
void read_lammps_atoms( std::ifstream &file, t_System *s )
{
    std::string line;

    s->slice_all();
    auto x = s->x;
    auto id = s->id;
    auto type = s->type;
    auto q = s->q;

    skip_empty( file, line );

    T_INT id_tmp, type_tmp;
    T_FLOAT x_tmp, y_tmp, z_tmp, q_tmp;
    T_INT counter = 0;
    for ( int n = 0; n < s->N; n++ )
    {
        // TODO: error if atom_style doesn't match data
        const char *temp = line.data();
        if ( s->atom_style == "atomic" )
        {
            std::sscanf( temp, "%i %i %lg %lg %lg", &id_tmp, &type_tmp, &x_tmp,
                         &y_tmp, &z_tmp );
            if ( ( x_tmp >= s->sub_domain_lo_x ) &&
                 ( y_tmp >= s->sub_domain_lo_y ) &&
                 ( z_tmp >= s->sub_domain_lo_z ) &&
                 ( x_tmp < s->sub_domain_hi_x ) &&
                 ( y_tmp < s->sub_domain_hi_y ) &&
                 ( z_tmp < s->sub_domain_hi_z ) )
            {
                id( n ) = id_tmp;
                type( n ) = type_tmp - 1;
                x( n, 0 ) = x_tmp;
                x( n, 1 ) = y_tmp;
                x( n, 2 ) = z_tmp;
                q( n ) = 0;
                counter++;
            }
            std::getline( file, line );
        }
        if ( s->atom_style == "charge" )
        {
            std::sscanf( temp, "%i %i %lg %lg %lg %lg", &id_tmp, &type_tmp,
                         &q_tmp, &x_tmp, &y_tmp, &z_tmp );
            if ( ( x_tmp >= s->sub_domain_lo_x ) &&
                 ( y_tmp >= s->sub_domain_lo_y ) &&
                 ( z_tmp >= s->sub_domain_lo_z ) &&
                 ( x_tmp < s->sub_domain_hi_x ) &&
                 ( y_tmp < s->sub_domain_hi_y ) &&
                 ( z_tmp < s->sub_domain_hi_z ) )
            {
                id( n ) = id_tmp;
                type( n ) = type_tmp - 1;
                q( n ) = q_tmp;
                x( n, 0 ) = x_tmp;
                x( n, 1 ) = y_tmp;
                x( n, 2 ) = z_tmp;
                counter++;
            }
            // getline pushed to the end of loop because line already stores the
            // 1st non-blank line after exiting while loop
            std::getline( file, line );
        }
    }
    s->N_local = counter;
}

template <class t_System>
void read_lammps_velocities( std::ifstream &file, t_System *s )
{
    std::string line;

    s->slice_v();
    auto v = s->v;

    skip_empty( file, line );

    T_INT id_tmp;
    T_FLOAT vx_tmp, vy_tmp, vz_tmp;
    for ( int n = 0; n < s->N; n++ )
    {
        const char *temp = line.data();
        std::sscanf( temp, "%i %lg %lg %lg", &id_tmp, &vx_tmp, &vy_tmp,
                     &vz_tmp );
        v( n, 0 ) = vx_tmp;
        v( n, 1 ) = vy_tmp;
        v( n, 2 ) = vz_tmp;
        std::getline( file, line );
    }
}

template <class t_System>
void read_lammps_masses( std::ifstream &file, t_System *s )
{
    std::string line;
    s->mass = t_mass( "System::mass", s->ntypes );

    skip_empty( file, line );

    T_INT type_tmp;
    T_FLOAT m_tmp;
    for ( int n = 0; n < s->ntypes; n++ )
    {
        const char *temp = line.data();
        std::sscanf( temp, "%i %lg", &type_tmp, &m_tmp );
        Kokkos::View<T_V_FLOAT> mass_one( s->mass, type_tmp - 1 );
        Kokkos::deep_copy( mass_one, m_tmp );
        std::getline( file, line );
    }
}

template <class t_System>
void read_lammps_pair( std::ifstream &file, t_System *s )
{
    // Parse through pair_coeff lines to avoid error, but ignore values
    std::string line;
    skip_empty( file, line );

    for ( int n = 0; n < s->ntypes; n++ )
        std::getline( file, line );
}

template <class t_System>
void read_lammps_data_file( std::string filename, t_System *s,
                            Comm<t_System> *comm )
{

    int atomflag = 0;
    std::string keyword;
    std::ifstream file( filename );

    read_lammps_header<t_System>( file, s );

    // TODO: this won't scale
    s->resize( s->N );

    // perform domain decomposition and get access to subdomains
    comm->create_domain_decomposition();

    // check that the next string is a valid section keyword
    keyword = read_lammps_parse_keyword( file );
    while ( keyword.length() )
    {
        if ( keyword.compare( "Atoms" ) == 0 )
        {
            read_lammps_atoms<t_System>( file, s );
            atomflag = 1;
        }
        else if ( keyword.compare( "Velocities" ) == 0 )
        {
            if ( atomflag == 0 )
                std::cout << "ERROR: Must read Atoms before Velocities"
                          << std::endl;

            read_lammps_velocities<t_System>( file, s );
        }
        else if ( keyword.compare( "Masses" ) == 0 )
        {
            read_lammps_masses<t_System>( file, s );
        }
        else if ( keyword.compare( "Pair Coeffs" ) == 0 )
        {
            std::cout << "WARNING: Ignoring potential parameters in data file. "
                         "CabanaMD only reads pair_coeff in the input file."
                      << std::endl;
            read_lammps_pair<t_System>( file, s );
        }
        else
        {
            std::cout << "ERROR: Unknown identifier in data file: " << keyword
                      << std::endl;
        }

        keyword = read_lammps_parse_keyword( file );
    }

    // check that correct # of atoms were created
    int natoms = s->N_local;
    comm->reduce_int( &natoms, 1 );

    if ( natoms != s->N )
    {
        if ( s->do_print )
            std::cout << "ERROR: Created incorrect # of atoms" << std::endl;
    }

    s->resize( s->N_local );
}
