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

#include <inputFile.h>
#include <property_temperature.h>

#include <fstream>
#include <iostream>

ItemizedFile::ItemizedFile()
{
    nlines = 0;
    max_nlines = 0;
    words = NULL;
    words_per_line = 32;
    max_word_size = 32;
}

void ItemizedFile::allocate_words( int num_lines )
{
    nlines = 0;

    if ( max_nlines >= num_lines )
    {
        for ( int i = 0; i < max_nlines; i++ )
            for ( int j = 0; j < words_per_line; j++ )
                words[i][j][0] = 0;
        return;
    }

    free_words();
    max_nlines = num_lines;
    words = new char **[max_nlines];
    for ( int i = 0; i < max_nlines; i++ )
    {
        words[i] = new char *[words_per_line];
        for ( int j = 0; j < words_per_line; j++ )
        {
            words[i][j] = new char[max_word_size];
            words[i][j][0] = 0;
        }
    }
}

void ItemizedFile::free_words()
{
    for ( int i = 0; i < max_nlines; i++ )
    {
        for ( int j = 0; j < words_per_line; j++ )
            delete[] words[i][j];
        delete[] words[i];
    }
    delete[] words;
}

void ItemizedFile::print_line( int i )
{
    for ( int j = 0; j < words_per_line; j++ )
    {
        if ( words[i][j][0] )
            std::cout << words[i][j] << " ";
    }
    std::cout << std::endl;
}

int ItemizedFile::words_in_line( int i )
{
    int count = 0;
    for ( int j = 0; j < words_per_line; j++ )
        if ( words[i][j][0] )
            count++;
    return count;
}
void ItemizedFile::print()
{
    for ( int l = 0; l < nlines; l++ )
        print_line( l );
}

void ItemizedFile::add_line( const char *const line )
{
    const char *pos = line;
    if ( nlines < max_nlines )
    {
        int j = 0;
        while ( ( *pos ) && ( j < words_per_line ) )
        {
            while ( ( ( *pos == ' ' ) || ( *pos == '\t' ) ) && *pos )
                pos++;
            int k = 0;
            while ( ( ( *pos != ' ' ) && ( *pos != '\t' ) ) && ( *pos ) &&
                    ( k < max_word_size ) )
            {
                words[nlines][j][k] = *pos;
                k++;
                pos++;
            }
            words[nlines][j][k] = 0;
            j++;
        }
    }
    nlines++;
}

template <class t_System>
InputFile<t_System>::InputFile( InputCL commandline_, t_System *system_ )
    : commandline( commandline_ )
    , system( system_ )
    , input_data( ItemizedFile() )
    , data_file_data( ItemizedFile() )
{
    comm_type = COMM_MPI;
    integrator_type = INTEGRATOR_NVE;
    neighbor_type = NEIGH_VERLET_2D;
    force_type = FORCE_LJ;
    binning_type = BINNING_LINKEDCELL;

    layout_type = commandline.layout_type;
    nnp_layout_type = commandline.nnp_layout_type;
    neighbor_type = commandline.neighbor_type;
    force_iteration_type = commandline.force_iteration_type;
    force_neigh_parallel_type = commandline.force_neigh_parallel_type;

    // set defaults (matches ExaMiniMD LJ example)

    nsteps = 0;
    force_coeff_lines = Kokkos::View<int *, Kokkos::HostSpace>(
        "InputFile::force_coeff_lines", 0 );
    data_file_type = -1;

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
    if ( filename == NULL )
    {
        filename = commandline.input_file;
    }
    if ( commandline.input_file_type == INPUT_LAMMPS )
    {
        read_lammps_file( filename );
        return;
    }
    if ( system->do_print )
        printf( "ERROR: Unknown input file type\n" );
    exit( 1 );
}

template <class t_System>
void InputFile<t_System>::read_lammps_file( const char *filename )
{
    input_data.allocate_words( 100 );

    std::ifstream file( filename );
    char *line = new char[512];
    line[0] = 0;
    file.getline( line, std::streamsize( 511 ) );
    while ( file.good() )
    {
        input_data.add_line( line );
        line[0] = 0;
        file.getline( line, 511 );
    }
    if ( system->do_print )
    {
        printf( "\n" );
        printf( "#InputFile:\n" );
        printf(
            "#=========================================================\n" );

        input_data.print();

        printf(
            "#=========================================================\n" );
        printf( "\n" );
    }
    for ( int l = 0; l < input_data.nlines; l++ )
        check_lammps_command( l );
}

template <class t_System>
void InputFile<t_System>::check_lammps_command( int line )
{
    bool known = false;

    if ( input_data.words[line][0][0] == 0 )
    {
        known = true;
    }
    if ( strstr( input_data.words[line][0], "#" ) )
    {
        known = true;
    }
    if ( strcmp( input_data.words[line][0], "variable" ) == 0 )
    {
        if ( system->do_print )
            printf( "LAMMPS-Command: 'variable' keyword is not supported in "
                    "CabanaMD\n" );
    }
    if ( strcmp( input_data.words[line][0], "units" ) == 0 )
    {
        if ( strcmp( input_data.words[line][1], "metal" ) == 0 )
        {
            known = true;
            units_style = UNITS_METAL;
            system->boltz = 8.617343e-5;
            // hplanck = 95.306976368;
            system->mvv2e = 1.0364269e-4;
            system->dt = 0.001;
        }
        else if ( strcmp( input_data.words[line][1], "real" ) == 0 )
        {
            known = true;
            units_style = UNITS_REAL;
            system->boltz = 0.0019872067;
            // hplanck = 95.306976368;
            system->mvv2e = 48.88821291 * 48.88821291;
            if ( !timestepflag )
                system->dt = 1.0;
        }
        else if ( strcmp( input_data.words[line][1], "lj" ) == 0 )
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
            if ( system->do_print )
                printf( "LAMMPS-Command: 'units' command only supports "
                        "'metal', 'real', and 'lj' in CabanaMD\n" );
        }
    }
    if ( strcmp( input_data.words[line][0], "atom_style" ) == 0 )
    {
        if ( strcmp( input_data.words[line][1], "atomic" ) == 0 )
        {
            known = true;
        }
        else if ( strcmp( input_data.words[line][1], "charge" ) == 0 )
        {
            known = true;
            system->atom_style = "charge";
        }
        else
        {
            if ( system->do_print )
                printf( "LAMMPS-Command: 'atom_style' command only supports "
                        "'atomic' and 'charge' in CabanaMD\n" );
        }
    }
    if ( strcmp( input_data.words[line][0], "lattice" ) == 0 )
    {
        if ( strcmp( input_data.words[line][1], "sc" ) == 0 )
        {
            known = true;
            lattice_style = LATTICE_SC;
            lattice_constant = atof( input_data.words[line][2] );
        }
        else if ( strcmp( input_data.words[line][1], "fcc" ) == 0 )
        {
            known = true;
            lattice_style = LATTICE_FCC;
            if ( units_style == UNITS_LJ )
                lattice_constant =
                    std::pow( ( 4.0 / atof( input_data.words[line][2] ) ),
                              ( 1.0 / 3.0 ) );
            else
                lattice_constant = atof( input_data.words[line][2] );
        }
        else
        {
            if ( system->do_print )
                printf( "LAMMPS-Command: 'lattice' command only supports 'sc' "
                        "and 'fcc' in CabanaMD\n" );
        }
        if ( strcmp( input_data.words[line][3], "origin" ) == 0 )
        {
            lattice_offset_x = atof( input_data.words[line][4] );
            lattice_offset_y = atof( input_data.words[line][5] );
            lattice_offset_z = atof( input_data.words[line][6] );
        }
    }
    if ( strcmp( input_data.words[line][0], "region" ) == 0 )
    {
        if ( strcmp( input_data.words[line][2], "block" ) == 0 )
        {
            known = true;
            int box[6];
            box[0] = atoi( input_data.words[line][3] );
            box[1] = atoi( input_data.words[line][4] );
            box[2] = atoi( input_data.words[line][5] );
            box[3] = atoi( input_data.words[line][6] );
            box[4] = atoi( input_data.words[line][7] );
            box[5] = atoi( input_data.words[line][8] );
            if ( ( box[0] != 0 ) || ( box[2] != 0 ) || ( box[4] != 0 ) )
                if ( system->do_print )
                    printf( "LAMMPS-Command: region only allows for boxes with "
                            "0,0,0 offset in CabanaMD\n" );
            lattice_nx = box[1];
            lattice_ny = box[3];
            lattice_nz = box[5];
        }
        else
        {
            if ( system->do_print )
                printf( "LAMMPS-Command: 'region' command only supports "
                        "'block' option in CabanaMD\n" );
        }
    }
    if ( strcmp( input_data.words[line][0], "create_box" ) == 0 )
    {
        known = true;
        system->ntypes = atoi( input_data.words[line][1] );
        system->mass = t_mass( "System::mass", system->ntypes );
    }
    if ( strcmp( input_data.words[line][0], "create_atoms" ) == 0 )
    {
        known = true;
    }
    if ( strcmp( input_data.words[line][0], "mass" ) == 0 )
    {
        known = true;
        int type = atoi( input_data.words[line][1] ) - 1;
        Kokkos::View<T_V_FLOAT> mass_one( system->mass, type );
        T_V_FLOAT mass = atof( input_data.words[line][2] );
        Kokkos::deep_copy( mass_one, mass );
    }
    if ( strcmp( input_data.words[line][0], "read_data" ) == 0 )
    {
        known = true;
        read_data_flag = true;
        lammps_data_file = input_data.words[line][1];
    }

    if ( strcmp( input_data.words[line][0], "pair_style" ) == 0 )
    {
        if ( strcmp( input_data.words[line][1], "lj/cut" ) == 0 )
        {
            known = true;
            force_type = FORCE_LJ;
            force_cutoff = atof( input_data.words[line][2] );
            force_line = line;
        }
        if ( strcmp( input_data.words[line][1], "snap" ) == 0 )
        {
            known = false;
            force_type = FORCE_SNAP;
            force_cutoff = 4.73442; // atof(input_data.words[line][2]);
            force_line = line;
        }
        if ( strcmp( input_data.words[line][1], "nnp" ) == 0 )
        {
            known = true;
            force_type = FORCE_NNP;
            force_line = line; // TODO: process this line to read in the right
                               // directories
            Kokkos::resize( force_coeff_lines, 1 );
            force_coeff_lines( 0 ) = line;
        }
        if ( system->do_print && !known )
            printf( "LAMMPS-Command: 'pair_style' command only supports "
                    "'lj/cut' and 'nnp' style in CabanaMD\n" );
    }
    if ( strcmp( input_data.words[line][0], "pair_coeff" ) == 0 )
    {
        known = true;
        if ( force_type == FORCE_NNP )
            force_cutoff = atof( input_data.words[line][3] );
        else
        {
            int n_coeff_lines = force_coeff_lines.extent( 0 );
            Kokkos::resize( force_coeff_lines, n_coeff_lines + 1 );
            force_coeff_lines( n_coeff_lines ) = line;
            n_coeff_lines++;
        }
    }
    if ( strcmp( input_data.words[line][0], "velocity" ) == 0 )
    {
        known = true;
        if ( strcmp( input_data.words[line][1], "all" ) != 0 )
        {
            if ( system->do_print )
                printf( "LAMMPS-Command: 'velocity' command can only be "
                        "applied to 'all' in CabanaMD\n" );
        }
        if ( strcmp( input_data.words[line][2], "create" ) != 0 )
        {
            if ( system->do_print )
                printf( "LAMMPS-Command: 'velocity' command can only be used "
                        "with option 'create' in CabanaMD\n" );
        }
        temperature_target = atof( input_data.words[line][3] );
        temperature_seed = atoi( input_data.words[line][4] );
    }
    if ( strcmp( input_data.words[line][0], "neighbor" ) == 0 )
    {
        known = true;
        neighbor_skin = atof( input_data.words[line][1] );
    }
    if ( strcmp( input_data.words[line][0], "neigh_modify" ) == 0 )
    {
        known = true;
        for ( int i = 1; i < input_data.words_per_line - 1; i++ )
            if ( strcmp( input_data.words[line][i], "every" ) == 0 )
            {
                comm_exchange_rate = atoi( input_data.words[line][i + 1] );
            }
    }
    if ( strcmp( input_data.words[line][0], "fix" ) == 0 )
    {
        if ( strcmp( input_data.words[line][3], "nve" ) == 0 )
        {
            known = true;
            integrator_type = INTEGRATOR_NVE;
        }
        else
        {
            if ( system->do_print )
                printf( "LAMMPS-Command: 'fix' command only supports 'nve' "
                        "style in CabanaMD\n" );
        }
    }
    if ( strcmp( input_data.words[line][0], "run" ) == 0 )
    {
        known = true;
        nsteps = atoi( input_data.words[line][1] );
    }
    if ( strcmp( input_data.words[line][0], "thermo" ) == 0 )
    {
        known = true;
        thermo_rate = atoi( input_data.words[line][1] );
    }
    if ( strcmp( input_data.words[line][0], "timestep" ) == 0 )
    {
        known = true;
        system->dt = atof( input_data.words[line][1] );
        timestepflag = true;
    }
    if ( strcmp( input_data.words[line][0], "newton" ) == 0 )
    {
        known = true;
        // File setting overriden by commandline
        if ( !commandline.set_force_iteration )
        {
            if ( strcmp( input_data.words[line][1], "on" ) == 0 )
            {
                force_iteration_type = FORCE_ITER_NEIGH_HALF;
            }
            else if ( strcmp( input_data.words[line][1], "off" ) == 0 )
            {
                force_iteration_type = FORCE_ITER_NEIGH_FULL;
            }
            else
            {
                if ( system->do_print )
                    printf(
                        "LAMMPS-Command: 'newton' must be followed by 'on' or "
                        "'off'\n" );
            }
        }
        else
        {
            if ( system->do_print )
                printf( "Overriding LAMMPS-Command: 'newton' replaced by "
                        "commandline --force-iteration " );
        }
    }
    if ( input_data.words[line][0][0] == '#' )
    {
        known = true;
    }
    if ( !known && system->do_print )
    {
        printf( "ERROR: unknown keyword\n" );
        input_data.print_line( line );
    }
}

template <class t_System>
void InputFile<t_System>::create_lattice( Comm<t_System> *comm )
{
    t_System s = *system;

    t_mass::HostMirror h_mass = Kokkos::create_mirror_view( s.mass );
    Kokkos::deep_copy( h_mass, s.mass );

    // Create Simple Cubic Lattice
    if ( lattice_style == LATTICE_SC )
    {
        system->domain_x = lattice_constant * lattice_nx;
        system->domain_y = lattice_constant * lattice_ny;
        system->domain_z = lattice_constant * lattice_nz;
        system->domain_hi_x = system->domain_x;
        system->domain_hi_y = system->domain_y;
        system->domain_hi_z = system->domain_z;

        comm->create_domain_decomposition();
        s = *system;

        T_INT ix_start = s.sub_domain_lo_x / s.domain_x * lattice_nx - 0.5;
        T_INT iy_start = s.sub_domain_lo_y / s.domain_y * lattice_ny - 0.5;
        T_INT iz_start = s.sub_domain_lo_z / s.domain_z * lattice_nz - 0.5;

        T_INT ix_end = s.sub_domain_hi_x / s.domain_x * lattice_nx + 0.5;
        T_INT iy_end = s.sub_domain_hi_y / s.domain_y * lattice_ny + 0.5;
        T_INT iz_end = s.sub_domain_hi_z / s.domain_z * lattice_nz + 0.5;

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
        system->N_local = n;
        system->N = n;
        system->resize( n );
        system->slice_all();
        s = *system;
        auto x = s.x;
        auto v = s.v;
        auto id = s.id;
        auto type = s.type;
        auto q = s.q;

        n = 0;
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
                        type( n ) = rand() % s.ntypes;
                        id( n ) = n + 1;
                        n++;
                    }
                }
            }
        }
        comm->reduce_int( &system->N, 1 );

        // Make ids unique over all processes
        T_INT N_local_offset = n;
        comm->scan_int( &N_local_offset, 1 );
        for ( T_INT i = 0; i < n; i++ )
            id( i ) += N_local_offset - n;

        if ( system->do_print )
            printf( "Atoms: %i %i\n", system->N, system->N_local );
    }

    // Create Face Centered Cubic (FCC) Lattice
    if ( lattice_style == LATTICE_FCC )
    {
        system->domain_x = lattice_constant * lattice_nx;
        system->domain_y = lattice_constant * lattice_ny;
        system->domain_z = lattice_constant * lattice_nz;
        system->domain_hi_x = system->domain_x;
        system->domain_hi_y = system->domain_y;
        system->domain_hi_z = system->domain_z;

        comm->create_domain_decomposition();
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

        T_INT ix_start = s.sub_domain_lo_x / s.domain_x * lattice_nx - 0.5;
        T_INT iy_start = s.sub_domain_lo_y / s.domain_y * lattice_ny - 0.5;
        T_INT iz_start = s.sub_domain_lo_z / s.domain_z * lattice_nz - 0.5;

        T_INT ix_end = s.sub_domain_hi_x / s.domain_x * lattice_nx + 0.5;
        T_INT iy_end = s.sub_domain_hi_y / s.domain_y * lattice_ny + 0.5;
        T_INT iz_end = s.sub_domain_hi_z / s.domain_z * lattice_nz + 0.5;

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

        system->N_local = n;
        system->N = n;
        system->resize( n );
        system->slice_all();
        s = *system;
        auto x = s.x;
        auto v = s.v;
        auto id = s.id;
        auto type = s.type;
        auto q = s.q;

        n = 0;

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
                            x( n, 0 ) = xtmp;
                            x( n, 1 ) = ytmp;
                            x( n, 2 ) = ztmp;
                            type( n ) = rand() % s.ntypes;
                            id( n ) = n + 1;
                            n++;
                        }
                    }
                }
            }
        }

        // Make ids unique over all processes
        T_INT N_local_offset = n;
        comm->scan_int( &N_local_offset, 1 );
        for ( T_INT i = 0; i < n; i++ )
            id( i ) += N_local_offset - n;

        comm->reduce_int( &system->N, 1 );

        if ( system->do_print )
            printf( "Atoms: %i %i\n", system->N, system->N_local );
    }
    // Initialize velocity using the equivalent of the LAMMPS
    // velocity geom option, i.e. uniform random kinetic energies.
    // zero out momentum of the whole system afterwards, to eliminate
    // drift (bad for energy statistics)

    { // Scope s
        system->slice_all();
        t_System s = *system;
        auto x = s.x;
        auto v = s.v;
        auto type = s.type;
        auto q = s.q;

        T_FLOAT total_mass = 0.0;
        T_FLOAT total_momentum_x = 0.0;
        T_FLOAT total_momentum_y = 0.0;
        T_FLOAT total_momentum_z = 0.0;

        for ( T_INT i = 0; i < system->N_local; i++ )
        {
            LAMMPS_RandomVelocityGeom random;
            double x_i[3] = {x( i, 0 ), x( i, 1 ), x( i, 2 )};
            random.reset( temperature_seed, x_i );

            T_FLOAT mass_i = h_mass( type( i ) );
            T_FLOAT vx = random.uniform() - 0.5;
            T_FLOAT vy = random.uniform() - 0.5;
            T_FLOAT vz = random.uniform() - 0.5;

            v( i, 0 ) = vx / sqrt( mass_i );
            v( i, 1 ) = vy / sqrt( mass_i );
            v( i, 2 ) = vz / sqrt( mass_i );

            q( i ) = 0.0;

            total_mass += mass_i;
            total_momentum_x += mass_i * v( i, 0 );
            total_momentum_y += mass_i * v( i, 1 );
            total_momentum_z += mass_i * v( i, 2 );
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
            v( i, 0 ) -= system_vx;
            v( i, 1 ) -= system_vy;
            v( i, 2 ) -= system_vz;
        }

        Temperature<t_System> temp( comm );
        T_V_FLOAT T = temp.compute( system );

        T_V_FLOAT T_init_scale = sqrt( temperature_target / T );

        for ( T_INT i = 0; i < system->N_local; i++ )
        {
            v( i, 0 ) *= T_init_scale;
            v( i, 1 ) *= T_init_scale;
            v( i, 2 ) *= T_init_scale;
        }
    }
}
