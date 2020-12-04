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

#include <Kokkos_Core.hpp>

#include <comm_mpi.h>
#include <types.h>

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

std::string read_lammps_parse_keyword( std::ifstream &file, std::ofstream &err )
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
            std::size_t end = keyword.find_last_not_of( " " ) + 1;
            if ( end < line.length() )
                keyword.erase( end );
            if ( ( keyword.compare( "Atoms" ) == 0 ) ||
                 ( keyword.compare( "Velocities" ) == 0 ) ||
                 ( keyword.compare( "Masses" ) == 0 ) ||
                 ( keyword.compare( "Pair Coeffs" ) == 0 ) )
                return keyword;
            else
            {
                log_err( err, "Unknown data file keyword: ", keyword.data() );
                return keyword;
            }
        }
        else
            continue;
    }
    return keyword;
}

template <class t_System>
void read_lammps_header( std::ifstream &file, std::ofstream &err, t_System *s )
{
    std::string line;
    // skip 1st line of file
    if ( !std::getline( file, line ) )
        log_err(
            err,
            "Could not read from data file. Please check for a valid file and "
            "ensure that file path is less than 32 characters." );

    std::array<double, 3> low_corner;
    std::array<double, 3> high_corner;
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
            low_corner[0] = xlo;
            high_corner[0] = xhi;
        }
        else if ( line.find( "ylo yhi" ) != std::string::npos )
        {
            std::sscanf( temp, "%lg %lg", &ylo, &yhi );
            low_corner[1] = ylo;
            high_corner[1] = yhi;
        }
        else if ( line.find( "zlo zhi" ) != std::string::npos )
        {
            std::sscanf( temp, "%lg %lg", &zlo, &zhi );
            low_corner[2] = zlo;
            high_corner[2] = zhi;
            break;
        }
    }

    // Create mesh
    s->create_domain( low_corner, high_corner );
}

template <class t_System, class t_HostSystem>
void read_lammps_atoms( std::ifstream &file, t_System *s, t_HostSystem &host_s )
{
    std::string line;

    host_s.slice_all();
    auto h_x = host_s.x;
    auto h_id = host_s.id;
    auto h_type = host_s.type;
    auto h_q = host_s.q;

    auto local_mesh_lo_x = s->local_mesh_lo_x;
    auto local_mesh_lo_y = s->local_mesh_lo_y;
    auto local_mesh_lo_z = s->local_mesh_lo_z;
    auto local_mesh_hi_x = s->local_mesh_hi_x;
    auto local_mesh_hi_y = s->local_mesh_hi_y;
    auto local_mesh_hi_z = s->local_mesh_hi_z;

    skip_empty( file, line );

    T_INT id_tmp, type_tmp;
    T_FLOAT x_tmp, y_tmp, z_tmp, q_tmp;
    std::size_t count = 0;
    for ( int n = 0; n < s->N; n++ )
    {
        // Resize if needed
        if ( count >= h_x.size() - 1 )
        {
            host_s.resize( count * 2 );
            host_s.slice_all();
            h_x = host_s.x;
            h_id = host_s.id;
            h_type = host_s.type;
            h_q = host_s.q;
        }

        // TODO: error if atom_style doesn't match data
        const char *temp = line.data();
        if ( s->atom_style == "atomic" )
        {
            std::sscanf( temp, "%i %i %lg %lg %lg", &id_tmp, &type_tmp, &x_tmp,
                         &y_tmp, &z_tmp );
            if ( ( x_tmp >= local_mesh_lo_x ) && ( y_tmp >= local_mesh_lo_y ) &&
                 ( z_tmp >= local_mesh_lo_z ) && ( x_tmp < local_mesh_hi_x ) &&
                 ( y_tmp < local_mesh_hi_y ) && ( z_tmp < local_mesh_hi_z ) )
            {
                h_id( count ) = id_tmp;
                h_type( count ) = type_tmp - 1;
                h_x( count, 0 ) = x_tmp;
                h_x( count, 1 ) = y_tmp;
                h_x( count, 2 ) = z_tmp;
                h_q( count ) = 0;
                count++;
            }
        }
        if ( s->atom_style == "charge" )
        {
            std::sscanf( temp, "%i %i %lg %lg %lg %lg", &id_tmp, &type_tmp,
                         &q_tmp, &x_tmp, &y_tmp, &z_tmp );
            if ( ( x_tmp >= local_mesh_lo_x ) && ( y_tmp >= local_mesh_lo_y ) &&
                 ( z_tmp >= local_mesh_lo_z ) && ( x_tmp < local_mesh_hi_x ) &&
                 ( y_tmp < local_mesh_hi_y ) && ( z_tmp < local_mesh_hi_z ) )
            {
                h_id( count ) = id_tmp;
                h_type( count ) = type_tmp - 1;
                h_q( count ) = q_tmp;
                h_x( count, 0 ) = x_tmp;
                h_x( count, 1 ) = y_tmp;
                h_x( count, 2 ) = z_tmp;
                count++;
            }
        }
        // getline pushed to the end of loop because line already stores the
        // 1st non-blank line after exiting while loop
        std::getline( file, line );
    }
    s->N_local = count;
    host_s.resize( s->N_local );
}

template <class t_System, class t_HostSystem>
void read_lammps_velocities( std::ifstream &file, t_System *s,
                             t_HostSystem &host_s )
{
    std::string line;

    host_s.slice_v();
    host_s.slice_id();
    auto h_v = host_s.v;
    auto h_id = host_s.id;

    skip_empty( file, line );

    T_INT id_tmp;
    T_FLOAT vx_tmp, vy_tmp, vz_tmp;
    std::size_t count = 0;
    for ( int n = 0; n < s->N; n++ )
    {
        const char *temp = line.data();
        std::sscanf( temp, "%i %lg %lg %lg", &id_tmp, &vx_tmp, &vy_tmp,
                     &vz_tmp );
        // Only extract velocities on this rank (order matches positions)
        if ( id_tmp == h_id( count ) )
        {
            h_v( count, 0 ) = vx_tmp;
            h_v( count, 1 ) = vy_tmp;
            h_v( count, 2 ) = vz_tmp;
            count++;
        }
        std::getline( file, line );
    }
}

template <class t_System>
void read_lammps_masses( std::ifstream &file, t_System *s )
{
    std::string line;
    using t_mass = typename t_System::t_mass;
    s->mass = t_mass( "System::mass", s->ntypes );

    skip_empty( file, line );

    T_INT type_tmp;
    T_FLOAT m_tmp;
    for ( int n = 0; n < s->ntypes; n++ )
    {
        const char *temp = line.data();
        std::sscanf( temp, "%i %lg", &type_tmp, &m_tmp );
        using exe_space = typename t_System::execution_space;
        Kokkos::View<T_V_FLOAT, exe_space> mass_one( s->mass, type_tmp - 1 );
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
void read_lammps_data_file( InputFile<t_System> *input, t_System *s,
                            Comm<t_System> *comm )
{
    int atomflag = 0;
    std::string keyword;
    std::ifstream file( input->input_data_file );
    std::ofstream out( input->output_file, std::ofstream::app );
    std::ofstream err( input->error_file, std::ofstream::app );

    read_lammps_header<t_System>( file, err, s );

    // Use a host mirror for reading from data file
    using t_layout = typename t_System::layout_type;
    System<Kokkos::Device<Kokkos::DefaultHostExecutionSpace, Kokkos::HostSpace>,
           t_layout>
        host_system;
    // Assume near load balance and resize as necessary
    host_system.resize( s->N / comm->num_processes() );

    // check that the next string is a valid section keyword
    keyword = read_lammps_parse_keyword( file, err );
    while ( keyword.length() )
    {
        if ( keyword.compare( "Atoms" ) == 0 )
        {
            read_lammps_atoms<t_System>( file, s, host_system );
            atomflag = 1;
        }
        else if ( keyword.compare( "Velocities" ) == 0 )
        {
            if ( atomflag == 0 )
                log_err( err, "Must read Atoms before Velocities" );

            read_lammps_velocities<t_System>( file, s, host_system );
        }
        else if ( keyword.compare( "Masses" ) == 0 )
        {
            read_lammps_masses<t_System>( file, s );
        }
        else if ( keyword.compare( "Pair Coeffs" ) == 0 )
        {
            read_lammps_pair<t_System>( file, s );
            log( err, "Warning: Ignoring potential parameters in data file. "
                      "CabanaMD only reads pair_coeff in the input file." );
            read_lammps_pair<t_System>( file, s );
        }
        else
        {
            log_err( err, "Unknown identifier in data file: ", keyword );
        }

        keyword = read_lammps_parse_keyword( file, err );
    }

    s->resize( s->N_local );
    s->deep_copy( host_system );

    // check that correct # of atoms were created
    int natoms = s->N_local;
    comm->reduce_int( &natoms, 1 );

    if ( natoms != s->N )
    {
        log_err( err, "Created incorrect # of atoms." );
    }
    else
    {
        log( out, "Atoms: ", s->N, " ", s->N_local );
    }
}

template <class t_System>
void write_data( t_System *s, std::string data_file )
{
    std::ofstream data( data_file );

    using t_layout = typename t_System::layout_type;
    System<Kokkos::Device<Kokkos::DefaultHostExecutionSpace, Kokkos::HostSpace>,
           t_layout>
        host_s;
    s->slice_x();
    auto x = s->x;
    host_s.resize( x.size() );
    host_s.slice_x();
    auto h_x = host_s.x;
    host_s.deep_copy( *s );

    log( data, "LAMMPS data file from CabanaMD\n" );
    log( data, s->N, " atoms" );
    log( data, s->ntypes, " atom types\n" );

    log( data, s->local_mesh_lo_x, " ", s->local_mesh_hi_x, " xlo xhi" );
    log( data, s->local_mesh_lo_y, " ", s->local_mesh_hi_y, " ylo yhi" );
    log( data, s->local_mesh_lo_z, " ", s->local_mesh_hi_z, " zlo zhi\n" );
    log( data, "Atoms # atomic\n" );

    host_s.slice_all();
    h_x = host_s.x;
    auto h_id = host_s.id;
    auto h_type = host_s.type;
    auto h_v = host_s.v;

    for ( int n = 0; n < s->N_local; n++ )
    {
        log( data, h_id( n ), " ", h_type( n ) + 1, " ", h_x( n, 0 ), " ",
             h_x( n, 1 ), " ", h_x( n, 2 ) );
    }
    log( data, "\nVelocities\n" );
    for ( int n = 0; n < s->N_local; n++ )
    {
        log( data, h_id( n ), " ", h_v( n, 0 ), " ", h_v( n, 1 ), " ",
             h_v( n, 2 ) );
    }
}
