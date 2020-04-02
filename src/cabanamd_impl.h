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

#include <property_kine.h>
#include <property_pote.h>
#include <property_temperature.h>
#include <read_data.h>

#define MAXPATHLEN 1024

template <class t_System, class t_Neighbor>
void CbnMD<t_System, t_Neighbor>::init( InputCL commandline )
{
    // Create the System class: atom properties (AoSoA) and simulation box
    system = new t_System;
    system->init();

    // Create the Input class: Command line and LAMMPS input file
    input = new InputFile<t_System>( commandline, system );

    if ( system->do_print )
    {
        Kokkos::DefaultExecutionSpace::print_configuration( std::cout );
    }
    // Read input file
    input->read_file();
    nsteps = input->nsteps;
    if ( system->do_print )
    {
        printf( "Read input file\n" );
    }
    // Check that the requested pair_style was compiled
#ifndef CabanaMD_ENABLE_NNP
    if ( input->force_type == FORCE_NNP )
    {
        if ( system->do_print )
        {
            std::cout << "NNP requested, but not compiled!" << std::endl;
        }
        std::exit( 1 );
    }
#endif
    auto neigh_cutoff = input->force_cutoff + input->neighbor_skin;
    bool half_neigh = input->force_iteration_type == FORCE_ITER_NEIGH_HALF;

    // Create Communication class: MPI
    comm = new Comm<t_System>( system, neigh_cutoff );

    // Create Integrator class: NVE ensemble
    integrator = new Integrator<t_System>( system );

    // Create Binning class: linked cell bin sort
    binning = new Binning<t_System>( system );

    // Create Neighbor class: create neighbor list
    neighbor = new t_Neighbor( neigh_cutoff, half_neigh );

    // Create Force class: potential options in force_types/ folder
    if ( input->force_type == FORCE_LJ )
    {
        force = new ForceLJ<t_System, t_Neighbor>( system );
    }
#ifdef CabanaMD_ENABLE_NNP
#include <system_nnp.h>
    else if ( input->force_type == FORCE_NNP )
    {
        bool serial_neigh =
            input->force_neigh_parallel_type == FORCE_PARALLEL_NEIGH_SERIAL;
        bool team_neigh =
            input->force_neigh_parallel_type == FORCE_PARALLEL_NEIGH_TEAM;
        bool vector_angle =
            input->force_neigh_parallel_type == FORCE_PARALLEL_NEIGH_VECTOR;
        if ( half_neigh )
            throw std::runtime_error( "Half neighbor list not implemented "
                                      "for the neural network potential." );
        else
        {
            if ( input->nnp_layout_type == AOSOA_1 )
            {
                if ( serial_neigh )
                    force =
                        new ForceNNP<t_System, System_NNP<AoSoA1>, t_Neighbor,
                                     Cabana::SerialOpTag, Cabana::SerialOpTag>(
                            system );
                if ( team_neigh )
                    force =
                        new ForceNNP<t_System, System_NNP<AoSoA1>, t_Neighbor,
                                     Cabana::TeamOpTag, Cabana::TeamOpTag>(
                            system );
                if ( vector_angle )
                    force = new ForceNNP<t_System, System_NNP<AoSoA1>,
                                         t_Neighbor, Cabana::TeamOpTag,
                                         Cabana::TeamVectorOpTag>( system );
            }
            if ( input->nnp_layout_type == AOSOA_3 )
            {
                if ( serial_neigh )
                    force =
                        new ForceNNP<t_System, System_NNP<AoSoA3>, t_Neighbor,
                                     Cabana::SerialOpTag, Cabana::SerialOpTag>(
                            system );
                if ( team_neigh )
                    force =
                        new ForceNNP<t_System, System_NNP<AoSoA3>, t_Neighbor,
                                     Cabana::TeamOpTag, Cabana::TeamOpTag>(
                            system );
                if ( vector_angle )
                    force = new ForceNNP<t_System, System_NNP<AoSoA3>,
                                         t_Neighbor, Cabana::TeamOpTag,
                                         Cabana::TeamVectorOpTag>( system );
            }
        }
    }
#endif
    else
        comm->error( "Invalid ForceType" );

    for ( std::size_t line = 0; line < input->force_coeff_lines.extent( 0 );
          line++ )
    {
        force->init_coeff(
            input->input_data.words[input->force_coeff_lines( line )] );
    }

    if ( system->do_print )
    {
        printf( "Using: SystemVectorLength:%i %s\n", CabanaMD_VECTORLENGTH,
                system->name() );
        printf( "Using: %s %s %s %s %s\n", force->name(), neighbor->name(),
                comm->name(), binning->name(), integrator->name() );
    }

    // Create atoms - from LAMMPS data file or create FCC/SC lattice
    if ( system->N == 0 && input->read_data_flag == true )
    {
        read_lammps_data_file<t_System>( input->lammps_data_file, system,
                                         comm );
    }
    else if ( system->N == 0 )
    {
        input->create_lattice( comm );
    }

    // Exchange atoms across MPI ranks
    comm->exchange();

    // Sort atoms
    binning->create_binning( neigh_cutoff, neigh_cutoff, neigh_cutoff, 1, true,
                             false, true );

    // Add ghost atoms from other MPI ranks (gather)
    comm->exchange_halo();

    // Compute atom neighbors
    neighbor->create( system );

    // Compute initial forces
    system->slice_f();
    auto f = system->f;
    Cabana::deep_copy( f, 0.0 );
    force->compute( system, neighbor );

    // Scatter ghost atom forces back to original MPI rank
    //   (update force for pair_style nnp even if full neighbor list)
    if ( half_neigh or input->force_type == FORCE_NNP )
    {
        comm->update_force();
    }

    // Initial output
    int step = 0;
    if ( input->thermo_rate > 0 )
    {
        Temperature<t_System> temp( comm );
        PotE<t_System, t_Neighbor> pote( comm );
        KinE<t_System> kine( comm );
        auto T = temp.compute( system );
        auto PE = pote.compute( system, force, neighbor ) / system->N;
        auto KE = kine.compute( system ) / system->N;
        if ( system->do_print )
        {
            if ( !system->print_lammps )
            {
                printf( "\n" );
                printf( "#Timestep Temperature PotE ETot Time Atomsteps/s\n" );
                printf( "%i %lf %lf %lf %lf %e\n", step, T, PE, PE + KE, 0.0,
                        0.0 );
            }
            else
            {
                printf( "\n" );
                printf( "Step Temp E_pair TotEng CPU\n" );
                printf( "     %i %lf %lf %lf %e\n", step, T, PE, PE + KE, 0.0 );
            }
        }
    }

    if ( input->dumpbinaryflag )
    {
        dump_binary( step );
    }
    if ( input->correctnessflag )
    {
        check_correctness( step );
    }
}

template <class t_System, class t_Neighbor>
void CbnMD<t_System, t_Neighbor>::run()
{
    auto neigh_cutoff = input->force_cutoff + input->neighbor_skin;
    bool half_neigh = input->force_iteration_type == FORCE_ITER_NEIGH_HALF;

    Temperature<t_System> temp( comm );
    PotE<t_System, t_Neighbor> pote( comm );
    KinE<t_System> kine( comm );

    double force_time = 0;
    double comm_time = 0;
    double neigh_time = 0;
    double integrate_time = 0;
    double other_time = 0;

    double last_time = 0;
    Kokkos::Timer timer, force_timer, comm_timer, neigh_timer, integrate_timer,
        other_timer;

    // Main timestep loop
    for ( int step = 1; step <= nsteps; step++ )
    {
        // Integrate atom positions - velocity Verlet first half
        integrate_timer.reset();
        integrator->initial_integrate( system );
        integrate_time += integrate_timer.seconds();

        if ( step % input->comm_exchange_rate == 0 && step > 0 )
        {
            // Exchange atoms across MPI ranks
            comm_timer.reset();
            comm->exchange();
            comm_time += comm_timer.seconds();

            // Sort atoms
            other_timer.reset();
            binning->create_binning( neigh_cutoff, neigh_cutoff, neigh_cutoff,
                                     1, true, false, true );
            other_time += other_timer.seconds();

            // Update ghost atoms (gather)
            comm_timer.reset();
            comm->exchange_halo();
            comm_time += comm_timer.seconds();

            // Compute atom neighbors
            neigh_timer.reset();
            neighbor->create( system );
            neigh_time += neigh_timer.seconds();
        }
        else
        {
            // Update ghost atom positions (scatter)
            comm_timer.reset();
            comm->update_halo();
            comm_time += comm_timer.seconds();
        }

        // Reset forces
        force_timer.reset();
        system->slice_f();
        auto f = system->f;
        Cabana::deep_copy( f, 0.0 );

        // Compute short range force
        force->compute( system, neighbor );
        force_time += force_timer.seconds();

        // This is where Bonds, Angles, and KSpace should go eventually

        // Scatter ghost atom forces back to original MPI rank
        //   (update force for pair_style nnp even if full neighbor list)
        if ( half_neigh or input->force_type == FORCE_NNP )
        {
            comm_timer.reset();
            comm->update_force();
            comm_time += comm_timer.seconds();
        }

        // Integrate atom positions - velocity Verlet second half
        integrate_timer.reset();
        integrator->final_integrate( system );
        integrate_time += integrate_timer.seconds();

        other_timer.reset();

        // Print output
        if ( step % input->thermo_rate == 0 )
        {
            auto T = temp.compute( system );
            auto PE = pote.compute( system, force, neighbor ) / system->N;
            auto KE = kine.compute( system ) / system->N;
            if ( system->do_print )
            {
                if ( !system->print_lammps )
                {
                    double time = timer.seconds();
                    printf( "%i %lf %lf %lf %lf %e\n", step, T, PE, PE + KE,
                            timer.seconds(),
                            1.0 * system->N * input->thermo_rate /
                                ( time - last_time ) );
                    last_time = time;
                }
                else
                {
                    double time = timer.seconds();
                    printf( "     %i %lf %lf %lf %lf\n", step, T, PE, PE + KE,
                            timer.seconds() );
                    last_time = time;
                }
            }
        }

        if ( input->dumpbinaryflag )
            dump_binary( step );

        if ( input->correctnessflag )
            check_correctness( step );

        other_time += other_timer.seconds();
    }

    double time = timer.seconds();

    // Final output and timings
    if ( system->do_print )
    {
        if ( !system->print_lammps )
        {
            printf( "\n" );
            printf( "#Procs Particles | Time T_Force T_Neigh T_Comm T_Int "
                    "T_Other | Steps/s Atomsteps/s Atomsteps/(proc*s)\n" );
            printf( "%i %i | %lf %lf %lf %lf %lf %lf | %lf %e %e PERFORMANCE\n",
                    comm->num_processes(), system->N, time, force_time,
                    neigh_time, comm_time, integrate_time, other_time,
                    1.0 * nsteps / time, 1.0 * system->N * nsteps / time,
                    1.0 * system->N * nsteps / time / comm->num_processes() );
            printf( "%i %i | %lf %lf %lf %lf %lf %lf | FRACTION\n",
                    comm->num_processes(), system->N, 1.0, force_time / time,
                    neigh_time / time, comm_time / time, integrate_time / time,
                    other_time / time );
        }
        else
        {
            printf( "Loop time of %f on %i procs for %i steps with %i atoms\n",
                    time, comm->num_processes(), nsteps, system->N );
        }
    }
}

template <class t_System, class t_Neighbor>
void CbnMD<t_System, t_Neighbor>::dump_binary( int step )
{

    // On dump steps print configuration

    if ( step % input->dumpbinary_rate )
        return;

    FILE *fp;
    T_INT n = system->N_local;

    char *filename = new char[MAXPATHLEN];
    sprintf( filename, "%s%s.%010d.%03d", input->dumpbinary_path, "/output",
             step, comm->process_rank() );
    fp = fopen( filename, "wb" );
    if ( fp == NULL )
    {
        char str[MAXPATHLEN];
        sprintf( str, "Cannot open dump file %s", filename );
        // comm->error(str);
    }

    system->slice_all();
    t_System s = *system;
    auto h_x = s.x;
    auto h_v = s.v;
    auto h_f = s.f;
    auto h_id = s.id;
    auto h_type = s.type;
    auto h_q = s.q;

    fwrite( &n, sizeof( T_INT ), 1, fp );
    fwrite( h_id.data(), sizeof( T_INT ), n, fp );
    fwrite( h_type.data(), sizeof( T_INT ), n, fp );
    fwrite( h_q.data(), sizeof( T_FLOAT ), n, fp );
    fwrite( h_x.data(), sizeof( T_X_FLOAT ), 3 * n, fp );
    fwrite( h_v.data(), sizeof( T_V_FLOAT ), 3 * n, fp );
    fwrite( h_f.data(), sizeof( T_F_FLOAT ), 3 * n, fp );

    fclose( fp );
}

// TODO: 1. Add path to Reference [DONE]
//     2. Add MPI Rank file ids in Reference [DONE]
//     3. Move to separate class
//     4. Add pressure to thermo output
//     5. basis_offset [DONE]
//     6. correctness output to file [DONE]

template <class t_System, class t_Neighbor>
void CbnMD<t_System, t_Neighbor>::check_correctness( int step )
{

    if ( step % input->correctness_rate )
        return;

    FILE *fpref;
    T_INT n = system->N_local;
    T_INT ntmp;

    system->slice_all();
    t_System s = *system;
    auto x = s.x;
    auto v = s.v;
    auto f = s.f;
    auto id = s.id;
    auto type = s.type;
    auto q = s.q;

    char *filename = new char[MAXPATHLEN];
    sprintf( filename, "%s%s.%010d.%03d", input->reference_path, "/output",
             step, comm->process_rank() );
    fpref = fopen( filename, "rb" );
    if ( fpref == NULL )
    {
        char str[MAXPATHLEN];
        sprintf( str, "Cannot open input file %s", filename );
        // comm->error(str);
    }

    fread( &ntmp, sizeof( T_INT ), 1, fpref );
    if ( ntmp != n )
    {
        // comm->error("Mismatch in current and reference atom counts");
        printf( "Mismatch in current and reference atom counts\n" );
    }

    Kokkos::View<T_INT *, Kokkos::LayoutRight> idref( "Correctness::id", n );
    Kokkos::View<T_INT *, Kokkos::LayoutRight> typeref( "Correctness::type",
                                                        n );
    Kokkos::View<T_FLOAT *, Kokkos::LayoutRight> qref( "Correctness::q", n );
    Kokkos::View<T_X_FLOAT **, Kokkos::LayoutRight> xref( "Correctness::x", 3,
                                                          n );
    Kokkos::View<T_V_FLOAT **, Kokkos::LayoutRight> vref( "Correctness::v", 3,
                                                          n );
    Kokkos::View<T_F_FLOAT **, Kokkos::LayoutRight> fref( "Correctness::f", 3,
                                                          n );

    fread( idref.data(), sizeof( T_INT ), n, fpref );
    fread( typeref.data(), sizeof( T_INT ), n, fpref );
    fread( qref.data(), sizeof( T_FLOAT ), n, fpref );
    fread( xref.data(), sizeof( T_X_FLOAT ), 3 * n, fpref );
    fread( vref.data(), sizeof( T_V_FLOAT ), 3 * n, fpref );
    fread( fref.data(), sizeof( T_F_FLOAT ), 3 * n, fpref );

    T_FLOAT sumdelrsq = 0.0;
    T_FLOAT sumdelvsq = 0.0;
    T_FLOAT sumdelfsq = 0.0;
    T_FLOAT maxdelr = 0.0;
    T_FLOAT maxdelv = 0.0;
    T_FLOAT maxdelf = 0.0;
    for ( int i = 0; i < n; i++ )
    {
        int ii = -1;
        if ( id( i ) != idref( i ) )
            for ( int j = 0; j < n; j++ )
            {
                if ( id( j ) == idref( i ) )
                {
                    ii = j;
                    break;
                }
            }
        else
            ii = i;

        if ( ii == -1 )
            printf( "Unable to find current id matchinf reference id %d \n",
                    idref( i ) );
        else
        {
            T_FLOAT delx, dely, delz, delrsq;
            delx = x( ii, 0 ) - xref( i, 0 );
            dely = x( ii, 1 ) - xref( i, 1 );
            delz = x( ii, 2 ) - xref( i, 2 );
            delrsq = delx * delx + dely * dely + delz * delz;
            sumdelrsq += delrsq;
            maxdelr = MAX( fabs( delx ), maxdelr );
            maxdelr = MAX( fabs( dely ), maxdelr );
            maxdelr = MAX( fabs( delz ), maxdelr );

            delx = v( ii, 0 ) - vref( i, 0 );
            dely = v( ii, 1 ) - vref( i, 1 );
            delz = v( ii, 2 ) - vref( i, 2 );
            delrsq = delx * delx + dely * dely + delz * delz;
            sumdelvsq += delrsq;
            maxdelv = MAX( fabs( delx ), maxdelv );
            maxdelv = MAX( fabs( dely ), maxdelv );
            maxdelv = MAX( fabs( delz ), maxdelv );

            delx = f( ii, 0 ) - fref( i, 0 );
            dely = f( ii, 1 ) - fref( i, 1 );
            delz = f( ii, 2 ) - fref( i, 2 );
            delrsq = delx * delx + dely * dely + delz * delz;
            sumdelfsq += delrsq;
            maxdelf = MAX( fabs( delx ), maxdelf );
            maxdelf = MAX( fabs( dely ), maxdelf );
            maxdelf = MAX( fabs( delz ), maxdelf );
        }
    }

    fclose( fpref );

    // Can't use this with current CommMPI::reduce_float()
    // T_FLOAT buf[3];
    // buf[0] = sumdelrsq;
    // buf[1] = sumdelvsq;
    // buf[2] = sumdelfsq;
    // comm->reduce_float(&buf[0],3);
    // sumdelrsq = buf[0];
    // sumdelrsq = buf[1];
    // sumdelrsq = buf[2];
    // buf[0] = maxdelr;
    // buf[1] = maxdelv;
    // buf[2] = maxdelf;
    // comm->reduce_max_float(buf,3);
    // maxdelr =   buf[0];
    // maxdelv =   buf[1];
    // maxdelf =   buf[2];

    comm->reduce_float( &sumdelrsq, 1 );
    comm->reduce_float( &sumdelvsq, 1 );
    comm->reduce_float( &sumdelfsq, 1 );
    comm->reduce_max_float( &maxdelr, 1 );
    comm->reduce_max_float( &maxdelv, 1 );
    comm->reduce_max_float( &maxdelf, 1 );

    if ( system->do_print )
    {
        if ( step == 0 )
        {
            FILE *fpout = fopen( input->correctness_file, "w" );
            fprintf( fpout, "# timestep deltarnorm maxdelr deltavnorm maxdelv "
                            "deltafnorm maxdelf\n" );
            fprintf( fpout, "%d %g %g %g %g %g %g\n", step, sqrt( sumdelrsq ),
                     maxdelr, sqrt( sumdelvsq ), maxdelv, sqrt( sumdelfsq ),
                     maxdelf );
            fclose( fpout );
        }
        else
        {
            FILE *fpout = fopen( input->correctness_file, "a" );
            fprintf( fpout, "%d %g %g %g %g %g %g\n", step, sqrt( sumdelrsq ),
                     maxdelr, sqrt( sumdelvsq ), maxdelv, sqrt( sumdelfsq ),
                     maxdelf );
            fclose( fpout );
        }
    }
}
