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

    units_style = UNITS_LJ;
    lattice_style = LATTICE_FCC;

    num_lattice = 0;

    temperature_target = 1.4;
    temperature_seed = 87287;

    nsteps = 100;
    thermo_rate = 10;

    neighbor_skin = 0.3;
    neighbor_skin = 0.0; // for metal and real units
    max_neigh_guess = 50;
    comm_exchange_rate = 20;

    force_cutoff = 2.5;
}

template <class t_System>
void InputFile<t_System>::read_file( std::ofstream &out, std::ofstream &err,
                                     const char *filename )
{
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
}

template <class t_System>
void InputFile<t_System>::read_lammps_file( std::ifstream &in,
                                            std::ofstream &out,
                                            std::ofstream &err )
{
    log( out, "\n#InputFile:\n",
         "#=========================================================" );

    std::string line;
    while ( std::getline( in, line ) )
        check_lammps_command( line, err );

    log( out, "#=========================================================\n" );
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
        int curr_latt = lattice_constant.size();
        lattice_constant.resize( curr_latt + 1 );
        if ( words.at( 1 ).compare( "sc" ) == 0 )
        {
            known = true;
            lattice_style = LATTICE_SC;
            lattice_constant.at( curr_latt ) = std::stod( words.at( 2 ) );
        }
        else if ( words.at( 1 ).compare( "fcc" ) == 0 )
        {
            known = true;
            lattice_style = LATTICE_FCC;
            if ( units_style == UNITS_LJ )
                lattice_constant.at( curr_latt ) = std::pow(
                    ( 4.0 / std::stod( words.at( 2 ) ) ), ( 1.0 / 3.0 ) );
            else
                lattice_constant.at( curr_latt ) = std::stod( words.at( 2 ) );
        }
        else
        {
            log_err( err,
                     "LAMMPS-Command: 'lattice' command only supports 'sc' "
                     "and 'fcc' in CabanaMD" );
        }
        system->lattice_constant = lattice_constant;

        lattice_offset_x.resize( curr_latt + 1 );
        lattice_offset_y.resize( curr_latt + 1 );
        lattice_offset_z.resize( curr_latt + 1 );
        if ( words.size() > 3 )
        {
            if ( words.at( 3 ).compare( "origin" ) == 0 )
            {
                lattice_offset_x.at( curr_latt ) = std::stod( words.at( 4 ) );
                lattice_offset_y.at( curr_latt ) = std::stod( words.at( 5 ) );
                lattice_offset_z.at( curr_latt ) = std::stod( words.at( 6 ) );
            }
            else
            {
                log_err( err, "LAMMPS-Command: 'lattice' command only supports "
                              "'origin' additional option in CabanaMD" );
            }
        }
        else
        {
            lattice_offset_x.at( curr_latt ) = 0.0;
            lattice_offset_y.at( curr_latt ) = 0.0;
            lattice_offset_z.at( curr_latt ) = 0.0;
        }
    }
    if ( keyword.compare( "region" ) == 0 )
    {
        int curr_latt = lattice_nx.size();
        lattice_nx.resize( curr_latt + 1 );
        lattice_ny.resize( curr_latt + 1 );
        lattice_nz.resize( curr_latt + 1 );
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
            lattice_nx.at( curr_latt ) = box[1];
            lattice_ny.at( curr_latt ) = box[3];
            lattice_nz.at( curr_latt ) = box[5];
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
        system->ntypes = std::stoi( words.at( 1 ) );
    }
    if ( keyword.compare( "create_atoms" ) == 0 )
    {
        known = true;
        num_lattice += 1;
    }
    if ( keyword.compare( "mass" ) == 0 )
    {
        known = true;
        int type = std::stoi( words.at( 1 ) ) - 1;
        if ( type >= (int)system->mass.extent( 0 ) )
            Kokkos::resize( system->mass, type + 1 );
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
            charge.resize( charge.size() + 1 );
            charge.at( type ) = std::stod( words.at( 4 ) );
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
        input_data_file = words.at( 1 );
    }
    if ( keyword.compare( "write_data" ) == 0 )
    {
        known = true;
        write_data_flag = true;
        output_data_file = words.at( 1 );
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
            lrforce_coeff_lines.resize( 1 );
            lrforce_coeff_lines.at( 0 ) = split( line );
        }
        else if ( words.at( 1 ).compare( "spme" ) == 0 )
        {
            known = true;
            lrforce_type = FORCE_SPME;
            lrforce_coeff_lines.resize( 1 );
            lrforce_coeff_lines.at( 0 ) = split( line );
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
        create_velocity_flag = true;
    }
    if ( keyword.compare( "neighbor" ) == 0 )
    {
        known = true;
        neighbor_skin = std::stod( words.at( 1 ) );
    }
    if ( keyword.compare( "neigh_modify" ) == 0 )
    {
        known = true;
        std::size_t i = 1;
        while ( i < words.size() )
        {
            if ( words.at( i ).compare( "every" ) == 0 )
            {
                comm_exchange_rate = std::stoi( words.at( i + 1 ) );
                i += 2;
            }
            else if ( words.at( i ).compare( "one" ) == 0 )
            {
                max_neigh_guess = std::stoi( words.at( i + 1 ) );
                i += 2;
            }
            else
            {
                log_err( err, "LAMMPS-Command: 'neigh_modify' only supports "
                              "'every' and 'one' in CabanaMD" );
            }
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
void InputFile<t_System>::create_lattices( Comm<t_System> *comm,
                                           std::ofstream &out )
{
    t_System s = *system;
    using t_layout = typename t_System::layout_type;
    System<Kokkos::Device<Kokkos::DefaultHostExecutionSpace, Kokkos::HostSpace>,
           t_layout>
        host_system;

    for ( int i = 0; i < num_lattice; i++ )
        create_one_lattice( comm, out, i, host_system );

    system->deep_copy( host_system );
}

template <class t_System>
template <class t_HostSystem>
void InputFile<t_System>::create_one_lattice( Comm<t_System> *comm,
                                              std::ofstream &out,
                                              const int create_type,
                                              t_HostSystem &host_system )
{
    auto curr_charge = charge.at( create_type );
    auto curr_constant = lattice_constant.at( create_type );
    auto curr_offset_x = lattice_offset_x.at( create_type );
    auto curr_offset_y = lattice_offset_y.at( create_type );
    auto curr_offset_z = lattice_offset_z.at( create_type );
    auto curr_nx = lattice_nx.at( create_type );
    auto curr_ny = lattice_ny.at( create_type );
    auto curr_nz = lattice_nz.at( create_type );

    // Create the mesh.
    T_X_FLOAT max_x = curr_constant * curr_nx;
    T_X_FLOAT max_y = curr_constant * curr_ny;
    T_X_FLOAT max_z = curr_constant * curr_nz;
    std::array<T_X_FLOAT, 3> global_low = {0.0, 0.0, 0.0};
    std::array<T_X_FLOAT, 3> global_high = {max_x, max_y, max_z};
    system->create_domain( global_low, global_high );
    t_System s = *system;

    auto local_mesh_lo_x = s.local_mesh_lo_x;
    auto local_mesh_lo_y = s.local_mesh_lo_y;
    auto local_mesh_lo_z = s.local_mesh_lo_z;
    auto local_mesh_hi_x = s.local_mesh_hi_x;
    auto local_mesh_hi_y = s.local_mesh_hi_y;
    auto local_mesh_hi_z = s.local_mesh_hi_z;

    T_INT ix_start = local_mesh_lo_x / s.global_mesh_x * curr_nx - 1.5;
    T_INT iy_start = local_mesh_lo_y / s.global_mesh_y * curr_ny - 1.5;
    T_INT iz_start = local_mesh_lo_z / s.global_mesh_z * curr_nz - 1.5;
    T_INT ix_end = local_mesh_hi_x / s.global_mesh_x * curr_nx + 1.5;
    T_INT iy_end = local_mesh_hi_y / s.global_mesh_y * curr_ny + 1.5;
    T_INT iz_end = local_mesh_hi_z / s.global_mesh_z * curr_nz + 1.5;

    // Create Simple Cubic Lattice
    if ( lattice_style == LATTICE_SC )
    {
        T_INT n = 0;

        for ( T_INT iz = iz_start; iz <= iz_end; iz++ )
        {
            T_FLOAT ztmp = curr_constant * ( iz + curr_offset_z );
            for ( T_INT iy = iy_start; iy <= iy_end; iy++ )
            {
                T_FLOAT ytmp = curr_constant * ( iy + curr_offset_y );
                for ( T_INT ix = ix_start; ix <= ix_end; ix++ )
                {
                    T_FLOAT xtmp = curr_constant * ( ix + curr_offset_x );
                    if ( ( xtmp >= local_mesh_lo_x ) &&
                         ( ytmp >= local_mesh_lo_y ) &&
                         ( ztmp >= local_mesh_lo_z ) &&
                         ( xtmp < local_mesh_hi_x ) &&
                         ( ytmp < local_mesh_hi_y ) &&
                         ( ztmp < local_mesh_hi_z ) )
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
            T_FLOAT ztmp = curr_constant * ( iz + curr_offset_z );
            for ( T_INT iy = iy_start; iy <= iy_end; iy++ )
            {
                T_FLOAT ytmp = curr_constant * ( iy + curr_offset_y );
                for ( T_INT ix = ix_start; ix <= ix_end; ix++ )
                {
                    T_FLOAT xtmp = curr_constant * ( ix + curr_offset_x );
                    if ( ( xtmp >= local_mesh_lo_x ) &&
                         ( ytmp >= local_mesh_lo_y ) &&
                         ( ztmp >= local_mesh_lo_z ) &&
                         ( xtmp < local_mesh_hi_x ) &&
                         ( ytmp < local_mesh_hi_y ) &&
                         ( ztmp < local_mesh_hi_z ) )
                    {
                        x( n, 0 ) = xtmp;
                        x( n, 1 ) = ytmp;
                        x( n, 2 ) = ztmp;
                        type( n ) = create_type;
                        id( n ) = n + 1;
                        q( n ) = curr_charge;
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
            basis[i][0] += curr_offset_x;
            basis[i][1] += curr_offset_y;
            basis[i][2] += curr_offset_z;
        }

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
                            curr_constant * ( 1.0 * ix + basis[k][0] );
                        T_FLOAT ytmp =
                            curr_constant * ( 1.0 * iy + basis[k][1] );
                        T_FLOAT ztmp =
                            curr_constant * ( 1.0 * iz + basis[k][2] );
                        if ( ( xtmp >= local_mesh_lo_x ) &&
                             ( ytmp >= local_mesh_lo_y ) &&
                             ( ztmp >= local_mesh_lo_z ) &&
                             ( xtmp < local_mesh_hi_x ) &&
                             ( ytmp < local_mesh_hi_y ) &&
                             ( ztmp < local_mesh_hi_z ) )
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
                            curr_constant * ( 1.0 * ix + basis[k][0] );
                        T_FLOAT ytmp =
                            curr_constant * ( 1.0 * iy + basis[k][1] );
                        T_FLOAT ztmp =
                            curr_constant * ( 1.0 * iz + basis[k][2] );
                        if ( ( xtmp >= local_mesh_lo_x ) &&
                             ( ytmp >= local_mesh_lo_y ) &&
                             ( ztmp >= local_mesh_lo_z ) &&
                             ( xtmp < local_mesh_hi_x ) &&
                             ( ytmp < local_mesh_hi_y ) &&
                             ( ztmp < local_mesh_hi_z ) )
                        {
                            h_x( n, 0 ) = xtmp;
                            h_x( n, 1 ) = ytmp;
                            h_x( n, 2 ) = ztmp;
                            h_type( n ) = create_type;
                            h_id( n ) = n + 1;
                            h_q( n ) = curr_charge;
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
    }
    log( out, "Atoms: ", system->N, " ", system->N_local );
}

template <class t_System>
void InputFile<t_System>::create_velocities( Comm<t_System> *comm,
                                             std::ofstream &out )
{
    // Initialize velocity using the equivalent of the LAMMPS
    // velocity geom option, i.e. uniform random kinetic energies.
    // zero out momentum of the whole system afterwards, to eliminate
    // drift (bad for energy statistics)

    t_System s = *system;
    using t_layout = typename t_System::layout_type;
    System<Kokkos::Device<Kokkos::DefaultHostExecutionSpace, Kokkos::HostSpace>,
           t_layout>
        host_system;
    host_system.slice_all();

    using h_t_mass = typename t_System::h_t_mass;
    h_t_mass h_mass = Kokkos::create_mirror_view( s.mass );
    Kokkos::deep_copy( h_mass, s.mass );

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

    log( out, "Created velocities" );
}
