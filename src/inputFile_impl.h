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

#include <inputFile.h>
#include <property_temperature.h>

#include <fstream>
#include <iostream>
#include <regex>

std::vector<std::string> split( const std::string &line )
{
    // Split line on spaces and tabs
    std::regex re( "[ \r\t\n]" );
    std::sregex_token_iterator first{line.begin(), line.end(), re, -1}, last;
    std::vector<std::string> words{first, last};
    // Remove empty
    words.erase(
        std::remove_if( words.begin(), words.end(),
                        []( std::string const &s ) { return s.empty(); } ),
        words.end() );
    return words;
}

template <class t_System>
InputFile<t_System>::InputFile( InputCL commandline_, t_System *system_ )
    : commandline( commandline_ )
    , system( system_ )
{
    comm_type = COMM_MPI;
    integrator_type = INTEGRATOR_NVE;
    neighbor_type = NEIGH_VERLET_2D;
    force_type = FORCE_LJ;
    lrforce_type = FORCE_NONE;
    binning_type = BINNING_LINKEDCELL;

    layout_type = commandline.layout_type;
    nnp_layout_type = commandline.nnp_layout_type;
    neighbor_type = commandline.neighbor_type;
    force_iteration_type = commandline.force_iteration_type;
    force_neigh_parallel_type = commandline.force_neigh_parallel_type;
    lrforce_iteration_type = force_iteration_type;

    output_file = commandline.output_file;
    error_file = commandline.error_file;

    // set defaults (matches ExaMiniMD LJ example)

    nsteps = 0;

    thermo_rate = 0;
    dumpbinary_rate = 0;
    correctness_rate = 0;
    dumpbinaryflag = false;
    correctnessflag = false;
    timestepflag = false;

    lattice_offset_x = 0.0;
    lattice_offset_y = 0.0;
    lattice_offset_z = 0.0;
    box[0] = 0;
    box[2] = 0;
    box[4] = 0;
    box[1] = 40;
    box[3] = 40;
    box[5] = 40;

    units_style = UNITS_LJ;
    lattice_style = LATTICE_FCC;
    lattice_constant = 0.8442;

    num_lattice = -1;
    curr_lattice = 0;
    done_lattice = 1;
    type_lattice = 0;

    temperature_target = 1.4;
    temperature_seed = 87287;

    nsteps = 100;
    thermo_rate = 10;

    neighbor_skin = 0.3;
    neighbor_skin = 0.0; // for metal and real units
    comm_exchange_rate = 20;

    force_cutoff = 2.5;
}

template <class t_System>
void InputFile<t_System>::read_file( const char *filename )
{
    // This is the first time the streams are used - overwrite any previous
    // output
    std::ofstream out( output_file, std::ofstream::out );
    std::ofstream err( error_file, std::ofstream::out );

    if ( filename == NULL )
    {
        filename = commandline.input_file;
    }
    if ( commandline.input_file_type == INPUT_LAMMPS )
    {
        std::ifstream in( filename );
        read_lammps_file( in, out, err );
        return;
    }
    log_err( err, "Unknown input file type: ", filename );
    err.close();
    out.close();
}

template <class t_System>
void InputFile<t_System>::read_lammps_file( std::ifstream &in,
                                            std::ofstream &out,
                                            std::ofstream &err )
{
    if ( num_lattice == -1 )
        log( out, "\n#InputFile:\n",
             "#=========================================================" );

    curr_lattice = 0;
    std::string line;
    while ( std::getline( in, line ) )
    {
        check_lammps_command( line, err );

        if ( num_lattice == -1 )
            log( out, line );

        if ( curr_lattice == done_lattice and done_lattice < num_lattice )
        {
            done_lattice += 1;
            return;
        }
    }

    // Re-read input file to build multiple lattices
    if ( num_lattice == -1 )
    {
        num_lattice = curr_lattice;
        log( out,
             "#=========================================================\n" );
    }
}

template <class t_System>
void InputFile<t_System>::check_lammps_command( std::string line,
                                                std::ofstream &err )
{
    std::string keyword = "";
    bool known = false;
    auto words = split( line );

    // Ignore empty lines and all comment lines
    if ( words.size() == 0 )
        known = true;
    else if ( words.at( 0 )[0] == '#' )
        known = true;
    else
        keyword = words.at( 0 );

    if ( keyword.compare( "variable" ) == 0 )
    {
        log_err( err, "LAMMPS-Command: 'variable' keyword is not supported in "
                      "CabanaMD" );
    }
    if ( keyword.compare( "units" ) == 0 )
    {
        if ( words.at( 1 ).compare( "metal" ) == 0 )
        {
            known = true;
            units_style = UNITS_METAL;
            system->boltz = 8.617343e-5;
            // hplanck = 95.306976368;
            system->mvv2e = 1.0364269e-4;
            system->dt = 0.001;
        }
        else if ( words.at( 1 ).compare( "real" ) == 0 )
        {
            known = true;
            units_style = UNITS_REAL;
            system->boltz = 0.0019872067;
            // hplanck = 95.306976368;
            system->mvv2e = 48.88821291 * 48.88821291;
            if ( !timestepflag )
                system->dt = 1.0;
        }
        else if ( words.at( 1 ).compare( "lj" ) == 0 )
        {
            known = true;
            units_style = UNITS_LJ;
            system->boltz = 1.0;
            // hplanck = 0.18292026;
            system->mvv2e = 1.0;
            if ( !timestepflag )
                system->dt = 0.005;
        }
        else
        {
            log_err( err, "LAMMPS-Command: 'units' command only supports "
                          "'metal', 'real', and 'lj' in CabanaMD" );
        }
    }
    if ( keyword.compare( "atom_style" ) == 0 )
    {
        if ( words.at( 1 ).compare( "atomic" ) == 0 )
        {
            known = true;
        }
        else if ( words.at( 1 ).compare( "charge" ) == 0 )
        {
            known = true;
            system->atom_style = "charge";
        }
        else
        {
            log_err( err, "LAMMPS-Command: 'atom_style' command only supports "
                          "'atomic' and 'charge' in CabanaMD" );
        }
    }
    if ( keyword.compare( "lattice" ) == 0 )
    {
        if ( words.at( 1 ).compare( "sc" ) == 0 )
        {
            known = true;
            lattice_style = LATTICE_SC;
            lattice_constant = std::stod( words.at( 2 ) );
        }
        else if ( words.at( 1 ).compare( "fcc" ) == 0 )
        {
            known = true;
            lattice_style = LATTICE_FCC;
            if ( units_style == UNITS_LJ )
                lattice_constant = std::pow(
                    ( 4.0 / std::stod( words.at( 2 ) ) ), ( 1.0 / 3.0 ) );
            else
                lattice_constant = std::stod( words.at( 2 ) );
        }
        else
        {
            log_err( err,
                     "LAMMPS-Command: 'lattice' command only supports 'sc' "
                     "and 'fcc' in CabanaMD" );
        }
        system->lattice_constant = lattice_constant;

        if ( words.size() > 3 )
        {
            if ( words.at( 3 ).compare( "origin" ) == 0 )
            {
                lattice_offset_x = std::stod( words.at( 4 ) );
                lattice_offset_y = std::stod( words.at( 5 ) );
                lattice_offset_z = std::stod( words.at( 6 ) );
            }
            else
            {
                log_err( err, "LAMMPS-Command: 'lattice' command only supports "
                              "'origin' additional option in CabanaMD" );
            }
        }
        else
        {
            lattice_offset_x = 0.0;
            lattice_offset_y = 0.0;
            lattice_offset_z = 0.0;
        }
    }
    if ( keyword.compare( "region" ) == 0 )
    {
        if ( words.at( 2 ).compare( "block" ) == 0 )
        {
            known = true;
            int box[6];
            box[0] = std::stoi( words.at( 3 ) );
            box[1] = std::stoi( words.at( 4 ) );
            box[2] = std::stoi( words.at( 5 ) );
            box[3] = std::stoi( words.at( 6 ) );
            box[4] = std::stoi( words.at( 7 ) );
            box[5] = std::stoi( words.at( 8 ) );
            if ( ( box[0] != 0 ) || ( box[2] != 0 ) || ( box[4] != 0 ) )
                log_err( err, "LAMMPS-Command: region only allows for boxes "
                              "with 0,0,0 offset in CabanaMD" );
            lattice_nx = box[1];
            lattice_ny = box[3];
            lattice_nz = box[5];
        }
        else
        {
            log_err( err, "LAMMPS-Command: 'region' command only supports "
                          "'block' option in CabanaMD" );
        }
    }
    if ( keyword.compare( "create_box" ) == 0 )
    {
        known = true;
        // Avoid resetting arrays if creating multiple lattices
        if ( num_lattice == -1 )
        {
            system->ntypes = std::stoi( words.at( 1 ) );
            using t_mass = typename t_System::t_mass;
            system->mass = t_mass( "System::mass", system->ntypes );
            system->charge = t_mass( "System::charge", system->ntypes );
        }
    }
    if ( keyword.compare( "create_atoms" ) == 0 )
    {
        known = true;
        type_lattice = std::stoi( words.at( 1 ) ) - 1;
        curr_lattice += 1;
    }
    if ( keyword.compare( "mass" ) == 0 )
    {
        known = true;
        int type = std::stoi( words.at( 1 ) ) - 1;
        using exe_space = typename t_System::execution_space;
        Kokkos::View<T_V_FLOAT, exe_space> mass_one( system->mass, type );
        T_V_FLOAT mass = std::stod( words.at( 2 ) );
        Kokkos::deep_copy( mass_one, mass );
    }
    if ( keyword.compare( "set" ) == 0 )
    {
        known = true;
        if ( words.at( 1 ).compare( "type" ) == 0 and
             words.at( 3 ).compare( "charge" ) == 0 )
        {
            int type = stoi( words.at( 2 ) ) - 1;
            Kokkos::View<T_V_FLOAT> charge_one( system->charge, type );
            T_V_FLOAT charge = std::stod( words.at( 4 ) );
            Kokkos::deep_copy( charge_one, charge );
        }
        else
        {
            log_err( err, "LAMMPS-Command: 'set' command only supports setting "
                          "charges by type CabanaMD\n" );
        }
    }
    if ( keyword.compare( "read_data" ) == 0 )
    {
        known = true;
        read_data_flag = true;
        lammps_data_file = words.at( 1 );
    }
    if ( keyword.compare( "pair_style" ) == 0 )
    {
        if ( words.at( 1 ).compare( "lj/cut" ) == 0 or
             words.at( 1 ).compare( "lj/cut/coul/long" ) == 0 )
        {
            known = true;
            force_type = FORCE_LJ;
            force_cutoff = std::stod( words.at( 2 ) );
            if ( words.at( 1 ).compare( "lj/cut/coul/long" ) == 0 )
            {
                lrforce_cutoff = std::stod( words.at( 2 ) );
            }
        }
        if ( words.at( 1 ).compare( "snap" ) == 0 )
        {
            known = false;
            force_type = FORCE_SNAP;
            force_cutoff = 4.73442; // std::stod(words.at(2));
        }
        if ( words.at( 1 ).compare( "nnp" ) == 0 )
        {
            known = true;
            force_type = FORCE_NNP;
            force_coeff_lines.resize( 1 );
            force_coeff_lines.at( 0 ) = split( line );
        }
        if ( !known )
            log_err(
                err,
                "LAMMPS-Command: 'pair_style' command only supports 'lj/cut', "
                "'lj/cut/coul/long', and 'nnp' style in CabanaMD" );
    }
    if ( keyword.compare( "pair_coeff" ) == 0 )
    {
        known = true;
        if ( force_type == FORCE_NNP )
            force_cutoff = std::stod( words.at( 3 ) );
        else
        {
            int nlines = force_coeff_lines.size();
            force_coeff_lines.resize( nlines + 1 );
            force_coeff_lines.at( nlines ) = split( line );
        }
    }
    if ( keyword.compare( "kspace_style" ) == 0 )
    {
        if ( words.at( 1 ).compare( "ewald" ) == 0 )
        {
            known = true;
            lrforce_type = FORCE_EWALD;
        }
        else if ( words.at( 1 ).compare( "spme" ) == 0 )
        {
            known = true;
            lrforce_type = FORCE_SPME;
        }
        else
        {
            log_err( err, "LAMMPS-Command: 'kspace_style' command only "
                          "supports 'ewald' or 'spme' in CabanaMD\n" );
        }
    }
    if ( keyword.compare( "velocity" ) == 0 )
    {
        known = true;
        if ( words.at( 1 ).compare( "all" ) != 0 )
        {
            log_err( err, "LAMMPS-Command: 'velocity' command can only be "
                          "applied to 'all' in CabanaMD" );
        }
        if ( words.at( 2 ).compare( "create" ) != 0 )
        {
            log_err( err, "LAMMPS-Command: 'velocity' command can only be used "
                          "with option 'create' in CabanaMD" );
        }
        temperature_target = std::stod( words.at( 3 ) );
        temperature_seed = std::stoi( words.at( 4 ) );
    }
    if ( keyword.compare( "neighbor" ) == 0 )
    {
        known = true;
        neighbor_skin = std::stod( words.at( 1 ) );
    }
    if ( keyword.compare( "neigh_modify" ) == 0 )
    {
        known = true;
        for ( std::size_t i = 0; i < words.size(); i++ )
            if ( words.at( i ).compare( "every" ) == 0 )
            {
                comm_exchange_rate = std::stoi( words.at( i + 1 ) );
            }
    }
    if ( keyword.compare( "fix" ) == 0 )
    {
        if ( words.at( 3 ).compare( "nve" ) == 0 )
        {
            known = true;
            integrator_type = INTEGRATOR_NVE;
        }
        else
        {
            log_err( err, "LAMMPS-Command: 'fix' command only supports 'nve' "
                          "style in CabanaMD" );
        }
    }
    if ( keyword.compare( "run" ) == 0 )
    {
        known = true;
        nsteps = std::stoi( words.at( 1 ) );
    }
    if ( keyword.compare( "thermo" ) == 0 )
    {
        known = true;
        thermo_rate = std::stoi( words.at( 1 ) );
    }
    if ( keyword.compare( "timestep" ) == 0 )
    {
        known = true;
        system->dt = std::stod( words.at( 1 ) );
        timestepflag = true;
    }
    if ( keyword.compare( "newton" ) == 0 )
    {
        known = true;
        // File setting overriden by commandline
        if ( !commandline.set_force_iteration )
        {
            if ( words.at( 1 ).compare( "on" ) == 0 )
            {
                force_iteration_type = FORCE_ITER_NEIGH_HALF;
            }
            else if ( words.at( 1 ).compare( "off" ) == 0 )
            {
                force_iteration_type = FORCE_ITER_NEIGH_FULL;
            }
            else
            {
                log_err( err, "LAMMPS-Command: 'newton' must be followed by "
                              "'on' or 'off'" );
            }
        }
        else
        {
            log( err,
                 "Warning: Overriding LAMMPS-Command: 'newton' replaced by "
                 "commandline --force-iteration" );
        }
    }

    if ( !known )
    {
        log_err( err, "Unknown input file keyword: ", line );
    }
}

template <class t_System>
void InputFile<t_System>::create_lattice( Comm<t_System> *comm )
{
    std::ofstream out( output_file, std::ofstream::app );

    t_System s = *system;
    using t_layout = typename t_System::layout_type;
    System<Kokkos::Device<Kokkos::DefaultHostExecutionSpace, Kokkos::HostSpace>,
           t_layout>
        host_system;

    using h_t_mass = typename t_System::h_t_mass;
    h_t_mass h_mass = Kokkos::create_mirror_view( s.mass );
    Kokkos::deep_copy( h_mass, s.mass );
    h_t_mass h_charge = Kokkos::create_mirror_view( s.charge );
    Kokkos::deep_copy( h_charge, s.charge );

    // Create Simple Cubic Lattice
    if ( lattice_style == LATTICE_SC )
    {
        // If creating the first lattice
        if ( curr_lattice == 1 )
        {
            system->domain_x = lattice_constant * lattice_nx;
            system->domain_y = lattice_constant * lattice_ny;
            system->domain_z = lattice_constant * lattice_nz;
            system->domain_hi_x = system->domain_x;
            system->domain_hi_y = system->domain_y;
            system->domain_hi_z = system->domain_z;

            comm->create_domain_decomposition();
        }
        s = *system;

        T_INT ix_start = s.sub_domain_lo_x / s.domain_x * lattice_nx - 1.5;
        T_INT iy_start = s.sub_domain_lo_y / s.domain_y * lattice_ny - 1.5;
        T_INT iz_start = s.sub_domain_lo_z / s.domain_z * lattice_nz - 1.5;

        T_INT ix_end = s.sub_domain_hi_x / s.domain_x * lattice_nx + 1.5;
        T_INT iy_end = s.sub_domain_hi_y / s.domain_y * lattice_ny + 1.5;
        T_INT iz_end = s.sub_domain_hi_z / s.domain_z * lattice_nz + 1.5;

        T_INT n = 0;

        for ( T_INT iz = iz_start; iz <= iz_end; iz++ )
        {
            T_FLOAT ztmp = lattice_constant * ( iz + lattice_offset_z );
            for ( T_INT iy = iy_start; iy <= iy_end; iy++ )
            {
                T_FLOAT ytmp = lattice_constant * ( iy + lattice_offset_y );
                for ( T_INT ix = ix_start; ix <= ix_end; ix++ )
                {
                    T_FLOAT xtmp = lattice_constant * ( ix + lattice_offset_x );
                    if ( ( xtmp >= s.sub_domain_lo_x ) &&
                         ( ytmp >= s.sub_domain_lo_y ) &&
                         ( ztmp >= s.sub_domain_lo_z ) &&
                         ( xtmp < s.sub_domain_hi_x ) &&
                         ( ytmp < s.sub_domain_hi_y ) &&
                         ( ztmp < s.sub_domain_hi_z ) )
                    {
                        n++;
                    }
                }
            }
        }
        system->resize( system->N_local + n );
        system->slice_all();
        s = *system;
        auto x = s.x;
        auto v = s.v;
        auto id = s.id;
        auto type = s.type;
        auto q = s.q;

        n = system->N_local;
        for ( T_INT iz = iz_start; iz <= iz_end; iz++ )
        {
            T_FLOAT ztmp = lattice_constant * ( iz + lattice_offset_z );
            for ( T_INT iy = iy_start; iy <= iy_end; iy++ )
            {
                T_FLOAT ytmp = lattice_constant * ( iy + lattice_offset_y );
                for ( T_INT ix = ix_start; ix <= ix_end; ix++ )
                {
                    T_FLOAT xtmp = lattice_constant * ( ix + lattice_offset_x );
                    if ( ( xtmp >= s.sub_domain_lo_x ) &&
                         ( ytmp >= s.sub_domain_lo_y ) &&
                         ( ztmp >= s.sub_domain_lo_z ) &&
                         ( xtmp < s.sub_domain_hi_x ) &&
                         ( ytmp < s.sub_domain_hi_y ) &&
                         ( ztmp < s.sub_domain_hi_z ) )
                    {
                        x( n, 0 ) = xtmp;
                        x( n, 1 ) = ytmp;
                        x( n, 2 ) = ztmp;
                        type( n ) = type_lattice;
                        id( n ) = n + 1;
                        q( n ) = h_charge( type( n ) );
                        n++;
                    }
                }
            }
        }
        system->N_local = n;
        system->N = n;
        comm->reduce_int( &system->N, 1 );

        // Make ids unique over all processes
        T_INT N_local_offset = n;
        comm->scan_int( &N_local_offset, 1 );
        for ( T_INT i = 0; i < n; i++ )
            id( i ) += N_local_offset - n;
    }

    // Create Face Centered Cubic (FCC) Lattice
    if ( lattice_style == LATTICE_FCC )
    {
        // If creating the first lattice
        if ( curr_lattice == 1 )
        {
            system->domain_x = lattice_constant * lattice_nx;
            system->domain_y = lattice_constant * lattice_ny;
            system->domain_z = lattice_constant * lattice_nz;
            system->domain_hi_x = system->domain_x;
            system->domain_hi_y = system->domain_y;
            system->domain_hi_z = system->domain_z;

            comm->create_domain_decomposition();
        }
        s = *system;

        double basis[4][3];
        basis[0][0] = 0.0;
        basis[0][1] = 0.0;
        basis[0][2] = 0.0;
        basis[1][0] = 0.5;
        basis[1][1] = 0.5;
        basis[1][2] = 0.0;
        basis[2][0] = 0.5;
        basis[2][1] = 0.0;
        basis[2][2] = 0.5;
        basis[3][0] = 0.0;
        basis[3][1] = 0.5;
        basis[3][2] = 0.5;

        for ( int i = 0; i < 4; i++ )
        {
            basis[i][0] += lattice_offset_x;
            basis[i][1] += lattice_offset_y;
            basis[i][2] += lattice_offset_z;
        }

        T_INT ix_start = s.sub_domain_lo_x / s.domain_x * lattice_nx - 1.5;
        T_INT iy_start = s.sub_domain_lo_y / s.domain_y * lattice_ny - 1.5;
        T_INT iz_start = s.sub_domain_lo_z / s.domain_z * lattice_nz - 1.5;

        T_INT ix_end = s.sub_domain_hi_x / s.domain_x * lattice_nx + 1.5;
        T_INT iy_end = s.sub_domain_hi_y / s.domain_y * lattice_ny + 1.5;
        T_INT iz_end = s.sub_domain_hi_z / s.domain_z * lattice_nz + 1.5;

        T_INT n = 0;

        for ( T_INT iz = iz_start; iz <= iz_end; iz++ )
        {
            for ( T_INT iy = iy_start; iy <= iy_end; iy++ )
            {
                for ( T_INT ix = ix_start; ix <= ix_end; ix++ )
                {
                    for ( int k = 0; k < 4; k++ )
                    {
                        T_FLOAT xtmp =
                            lattice_constant * ( 1.0 * ix + basis[k][0] );
                        T_FLOAT ytmp =
                            lattice_constant * ( 1.0 * iy + basis[k][1] );
                        T_FLOAT ztmp =
                            lattice_constant * ( 1.0 * iz + basis[k][2] );
                        if ( ( xtmp >= s.sub_domain_lo_x ) &&
                             ( ytmp >= s.sub_domain_lo_y ) &&
                             ( ztmp >= s.sub_domain_lo_z ) &&
                             ( xtmp < s.sub_domain_hi_x ) &&
                             ( ytmp < s.sub_domain_hi_y ) &&
                             ( ztmp < s.sub_domain_hi_z ) )
                        {
                            n++;
                        }
                    }
                }
            }
        }
        system->resize( system->N_local + n );

        host_system.resize( system->N_local + n );
        host_system.slice_all();
        auto h_x = host_system.x;
        auto h_id = host_system.id;
        auto h_type = host_system.type;
        auto h_q = host_system.q;

        n = system->N_local;
        for ( T_INT iz = iz_start; iz <= iz_end; iz++ )
        {
            for ( T_INT iy = iy_start; iy <= iy_end; iy++ )
            {
                for ( T_INT ix = ix_start; ix <= ix_end; ix++ )
                {
                    for ( int k = 0; k < 4; k++ )
                    {
                        T_FLOAT xtmp =
                            lattice_constant * ( 1.0 * ix + basis[k][0] );
                        T_FLOAT ytmp =
                            lattice_constant * ( 1.0 * iy + basis[k][1] );
                        T_FLOAT ztmp =
                            lattice_constant * ( 1.0 * iz + basis[k][2] );
                        if ( ( xtmp >= s.sub_domain_lo_x ) &&
                             ( ytmp >= s.sub_domain_lo_y ) &&
                             ( ztmp >= s.sub_domain_lo_z ) &&
                             ( xtmp < s.sub_domain_hi_x ) &&
                             ( ytmp < s.sub_domain_hi_y ) &&
                             ( ztmp < s.sub_domain_hi_z ) )
                        {
                            h_x( n, 0 ) = xtmp;
                            h_x( n, 1 ) = ytmp;
                            h_x( n, 2 ) = ztmp;
                            h_type( n ) = rand() % s.ntypes;
                            h_id( n ) = n + 1;
                            h_q( n ) = h_charge( h_type( n ) );
                            n++;
                        }
                    }
                }
            }
        }
        system->N_local = n;
        system->N = n;
        comm->reduce_int( &system->N, 1 );

        // Make ids unique over all processes
        T_INT N_local_offset = n;
        comm->scan_int( &N_local_offset, 1 );
        for ( T_INT i = 0; i < n; i++ )
            h_id( i ) += N_local_offset - n;

        system->deep_copy( host_system );
    }
    log( out, "Atoms: ", system->N, " ", system->N_local );

    // Initialize velocity using the equivalent of the LAMMPS
    // velocity geom option, i.e. uniform random kinetic energies.
    // zero out momentum of the whole system afterwards, to eliminate
    // drift (bad for energy statistics)

    // already sliced
    auto h_x = host_system.x;
    auto h_v = host_system.v;
    auto h_q = host_system.q;
    auto h_type = host_system.type;

    T_FLOAT total_mass = 0.0;
    T_FLOAT total_momentum_x = 0.0;
    T_FLOAT total_momentum_y = 0.0;
    T_FLOAT total_momentum_z = 0.0;

    for ( T_INT i = 0; i < system->N_local; i++ )
    {
        LAMMPS_RandomVelocityGeom random;
        double x_i[3] = {h_x( i, 0 ), h_x( i, 1 ), h_x( i, 2 )};
        random.reset( temperature_seed, x_i );

        T_FLOAT mass_i = h_mass( h_type( i ) );
        T_FLOAT vx = random.uniform() - 0.5;
        T_FLOAT vy = random.uniform() - 0.5;
        T_FLOAT vz = random.uniform() - 0.5;

        h_v( i, 0 ) = vx / sqrt( mass_i );
        h_v( i, 1 ) = vy / sqrt( mass_i );
        h_v( i, 2 ) = vz / sqrt( mass_i );

        h_q( i ) = 0.0;

        total_mass += mass_i;
        total_momentum_x += mass_i * h_v( i, 0 );
        total_momentum_y += mass_i * h_v( i, 1 );
        total_momentum_z += mass_i * h_v( i, 2 );
    }
    comm->reduce_float( &total_momentum_x, 1 );
    comm->reduce_float( &total_momentum_y, 1 );
    comm->reduce_float( &total_momentum_z, 1 );
    comm->reduce_float( &total_mass, 1 );

    T_FLOAT system_vx = total_momentum_x / total_mass;
    T_FLOAT system_vy = total_momentum_y / total_mass;
    T_FLOAT system_vz = total_momentum_z / total_mass;

    for ( T_INT i = 0; i < system->N_local; i++ )
    {
        h_v( i, 0 ) -= system_vx;
        h_v( i, 1 ) -= system_vy;
        h_v( i, 2 ) -= system_vz;
    }
    // temperature computed on the device
    system->deep_copy( host_system );

    Temperature<t_System> temp( comm );
    T_V_FLOAT T = temp.compute( system );

    T_V_FLOAT T_init_scale = sqrt( temperature_target / T );

    for ( T_INT i = 0; i < system->N_local; i++ )
    {
        h_v( i, 0 ) *= T_init_scale;
        h_v( i, 1 ) *= T_init_scale;
        h_v( i, 2 ) *= T_init_scale;
    }
    system->deep_copy( host_system );

    out.close();
}
