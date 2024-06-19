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

#include <CabanaMD_config.hpp>

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>

#include <mpi.h>

#include <output.h>
#include <property_kine.h>
#include <property_pote.h>
#include <property_temperature.h>
#include <read_data.h>
#include <vtk_writer.h>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

#define MAXPATHLEN 1024

template <class t_System, class t_Neighbor>
void CbnMD<t_System, t_Neighbor>::init( InputCL commandline )
{
    // Create the System class: atom properties (AoSoA) and simulation box
    system = new t_System;
    system->init();

    // Create the Input class: Command line and LAMMPS input file
    input = new InputFile<t_System>( commandline, system );
    // Read input file
    input->read_file();
    nsteps = input->nsteps;
    std::ofstream out( input->output_file, std::ofstream::app );
    std::ofstream err( input->error_file, std::ofstream::app );
    log( out, "Read input file." );

    using exe_space = typename t_System::execution_space;
    if ( print_rank() )
        exe_space().print_configuration( out );

#ifndef CabanaMD_ENABLE_NNP
    // Check that the requested pair_style was compiled
    if ( input->force_type == FORCE_NNP )
    {
        log_err( err, "NNP requested, but not compiled!" );
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
    neighbor =
        new t_Neighbor( neigh_cutoff, half_neigh, input->max_neigh_guess );

    // Create Force class: potential options in force_types/ folder
    bool serial_neigh =
        input->force_neigh_parallel_type == FORCE_PARALLEL_NEIGH_SERIAL;
    bool team_neigh =
        input->force_neigh_parallel_type == FORCE_PARALLEL_NEIGH_TEAM;
    if ( input->force_type == FORCE_LJ )
    {
        if ( serial_neigh )
            force = new ForceLJ<t_System, t_Neighbor, Cabana::SerialOpTag>(
                system );
        else if ( team_neigh )
            force =
                new ForceLJ<t_System, t_Neighbor, Cabana::TeamOpTag>( system );
    }
#ifdef CabanaMD_ENABLE_NNP
#include <system_nnp.h>
    else if ( input->force_type == FORCE_NNP )
    {
        bool vector_angle =
            input->force_neigh_parallel_type == FORCE_PARALLEL_NEIGH_VECTOR;
        if ( half_neigh )
            log_err( err, "Half neighbor list not implemented "
                          "for the neural network potential." );
        else
        {
            using t_device = typename t_System::device_type;
#if ( CabanaMD_LAYOUT_NNP == 1 )
            if ( serial_neigh )
                force =
                    new ForceNNP<t_System, System_NNP<t_device, 1>, t_Neighbor,
                                 Cabana::SerialOpTag, Cabana::SerialOpTag>(
                        system );
            if ( team_neigh )
                force =
                    new ForceNNP<t_System, System_NNP<t_device, 1>, t_Neighbor,
                                 Cabana::TeamOpTag, Cabana::TeamOpTag>(
                        system );
            if ( vector_angle )
                force =
                    new ForceNNP<t_System, System_NNP<t_device, 1>, t_Neighbor,
                                 Cabana::TeamOpTag, Cabana::TeamVectorOpTag>(
                        system );
#elif ( CabanaMD_LAYOUT_NNP == 3 )
            if ( serial_neigh )
                force =
                    new ForceNNP<t_System, System_NNP<t_device, 3>, t_Neighbor,
                                 Cabana::SerialOpTag, Cabana::SerialOpTag>(
                        system );
            if ( team_neigh )
                force =
                    new ForceNNP<t_System, System_NNP<t_device, 3>, t_Neighbor,
                                 Cabana::TeamOpTag, Cabana::TeamOpTag>(
                        system );
            if ( vector_angle )
                force =
                    new ForceNNP<t_System, System_NNP<t_device, 3>, t_Neighbor,
                                 Cabana::TeamOpTag, Cabana::TeamVectorOpTag>(
                        system );
#endif
        }
    }
#endif
    else
    {
        log_err( err, "Invalid ForceType" );
    }
    force->init_coeff( input->force_coeff_lines );

    log( out, "Using: SystemVectorLength: ", CabanaMD_VECTORLENGTH, " ",
         system->name() );
#ifdef CabanaMD_ENABLE_NNP
    if ( input->force_type == FORCE_NNP )
        log( out, "Using: SystemNNPVectorLength: ", CabanaMD_VECTORLENGTH_NNP,
             " ", force->system_name() );
#endif
    log( out, "Using: ", force->name(), " ", neighbor->name(), " ",
         comm->name(), " ", binning->name(), " ", integrator->name() );

    // Create atoms - from LAMMPS data file or create FCC/SC lattice
    if ( system->N == 0 && input->read_data_flag == true )
    {
        read_lammps_data_file<t_System>( input, system, comm );
    }
    else if ( system->N == 0 )
    {
        input->create_lattice( comm );
    }
    log( out, "Created atoms." );

    // Set MPI rank neighbors
    comm->create_domain_decomposition();

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

#ifdef CabanaMD_ENABLE_LB
    lb = new Cabana::Grid::Experimental::LoadBalancer<Cabana::Grid::UniformMesh<double>>(
        MPI_COMM_WORLD, system->global_grid );
#endif

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
        if ( !_print_lammps )
        {
#ifdef CabanaMD_ENABLE_LB
            log( out, "\n", std::fixed, std::setprecision( 6 ),
                 "#Timestep Temperature PotE ETot Time Atomsteps/s "
                 "LBImbalance\n",
                 step, " ", T, " ", PE, " ", PE + KE, " ",
                 std::setprecision( 2 ), 0.0, " ", std::scientific, 0.0, " ",
                 std::setprecision( 2 ), 0.0 );
#else
            log( out, "\n", std::fixed, std::setprecision( 6 ),
                 "#Timestep Temperature PotE ETot Time Atomsteps/s\n", step,
                 " ", T, " ", PE, " ", PE + KE, " ", std::setprecision( 2 ),
                 0.0, " ", std::scientific, 0.0 );
#endif
        }
        else
        {
            log( out, "\nStep Temp E_pair TotEng CPU\n", step, " ", T, " ", PE,
                 " ", PE + KE, " ", 0.0 );
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
    out.close();
    err.close();
}

template <class t_System, class t_Neighbor>
void CbnMD<t_System, t_Neighbor>::run()
{
    std::ofstream out( input->output_file, std::ofstream::app );
    std::ofstream err( input->error_file, std::ofstream::app );

    auto neigh_cutoff = input->force_cutoff + input->neighbor_skin;
    bool half_neigh = input->force_iteration_type == FORCE_ITER_NEIGH_HALF;

    Temperature<t_System> temp( comm );
    PotE<t_System, t_Neighbor> pote( comm );
    KinE<t_System> kine( comm );

    std::string vtk_actual_domain_basename( "domain_act" );
    std::string vtk_lb_domain_basename( "domain_lb" );

    double force_time = 0;
    double comm_time = 0;
    double neigh_time = 0;
    double integrate_time = 0;
    double lb_time = 0;
    double other_time = 0;

    double last_time = 0;
    Kokkos::Timer timer, force_timer, comm_timer, neigh_timer, integrate_timer,
        lb_timer, other_timer;

    // Main timestep loop
    for ( int step = 1; step <= nsteps; step++ )
    {
        // Integrate atom positions - velocity Verlet first half
        integrate_timer.reset();
        integrator->initial_integrate( system );
        integrate_time += integrate_timer.seconds();

        if ( step % input->comm_exchange_rate == 0 && step > 0 )
        {
            // Update domain decomposition
            lb_timer.reset();
#ifdef CabanaMD_ENABLE_LB
            double work = system->N_local + system->N_ghost;
            auto new_global_grid = lb->createBalancedGlobalGrid(
                system->global_mesh, *system->partitioner, work );
            system->update_global_grid( new_global_grid );
#endif
            lb_time += lb_timer.seconds();

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

            if ( !_print_lammps )
            {
                double time = timer.seconds();
                double rate =
                    1.0 * system->N * input->thermo_rate / ( time - last_time );
#ifdef CabanaMD_ENABLE_LB
                log( out, std::fixed, std::setprecision( 6 ), step, " ", T, " ",
                     PE, " ", PE + KE, " ", std::setprecision( 2 ), time, " ",
                     std::scientific, rate, " ", std::setprecision( 2 ),
                     lb->getImbalance() );
#else
                log( out, std::fixed, std::setprecision( 6 ), step, " ", T, " ",
                     PE, " ", PE + KE, " ", std::setprecision( 2 ), time, " ",
                     std::scientific, rate );
#endif
                last_time = time;
            }
            else
            {
                double time = timer.seconds();
                log( out, std::fixed, std::setprecision( 6 ), "     ", step,
                     " ", T, " ", PE, " ", PE + KE, " ", time );
                last_time = time;
            }
#ifdef CabanaMD_ENABLE_LB
            double work = system->N_local + system->N_ghost;
            std::array<double, 6> vertices;
            vertices = lb->getVertices();
            VTKWriter::writeDomain( MPI_COMM_WORLD, step, vertices, work,
                                    vtk_actual_domain_basename );
            vertices = lb->getInternalVertices();
            VTKWriter::writeDomain( MPI_COMM_WORLD, step, vertices, work,
                                    vtk_lb_domain_basename );
#endif
        }

        if ( step % input->vtk_rate == 0 )
            VTKWriter::writeParticles( MPI_COMM_WORLD, step, system,
                                       input->vtk_file, err );

        if ( input->dumpbinaryflag )
            dump_binary( step );

        if ( input->correctnessflag )
            check_correctness( step );

        other_time += other_timer.seconds();
    }

    double time = timer.seconds();

    // Final output and timings
    if ( !_print_lammps )
    {
        double steps_per_sec = 1.0 * nsteps / time;
        double atom_steps_per_sec = system->N * steps_per_sec;
        // todo(sschulz): Properly remove lb timing if not enabled.
        log( out, std::fixed, std::setprecision( 2 ),
             "\n#Procs Atoms | Time T_Force T_Neigh T_Comm T_Int T_lb ",
             "T_Other |\n", comm->num_processes(), " ", system->N, " | ", time,
             " ", force_time, " ", neigh_time, " ", comm_time, " ",
             integrate_time, " ", lb_time, " ", other_time, " | PERFORMANCE\n",
             std::fixed, comm->num_processes(), " ", system->N, " | ", 1.0, " ",
             force_time / time, " ", neigh_time / time, " ", comm_time / time,
             " ", integrate_time / time, " ", lb_time / time, " ",
             other_time / time, " | FRACTION\n\n",
             "#Steps/s Atomsteps/s Atomsteps/(proc*s)\n", std::scientific,
             steps_per_sec, " ", atom_steps_per_sec, " ",
             atom_steps_per_sec / comm->num_processes() );
    }
    else
    {
        log( out, "Loop time of ", time, " on ", comm->num_processes(),
             " procs for ", nsteps, " steps with ", system->N, " atoms" );
    }
    out.close();
    err.close();

    if ( input->write_data_flag )
        write_data( system, input->output_data_file );
}

template <class t_System, class t_Neighbor>
void CbnMD<t_System, t_Neighbor>::dump_binary( int step )
{
    std::ofstream err( input->error_file, std::ofstream::app );

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
        log_err( err, "Cannot open dump file: ", filename );
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
    err.close();
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
    std::ofstream err( input->error_file, std::ofstream::app );

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
        log_err( err, "Cannot open input file: ", filename );
    }

    T_INT n_ret = fread( &ntmp, sizeof( T_INT ), 1, fpref );
    if ( ntmp != n )
    {
        log_err( err, "Mismatch in current and reference atom counts" );
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

    T_INT id_ret = fread( idref.data(), sizeof( T_INT ), n, fpref );
    T_INT type_ret = fread( typeref.data(), sizeof( T_INT ), n, fpref );
    T_INT q_ret = fread( qref.data(), sizeof( T_FLOAT ), n, fpref );
    T_INT x_ret = fread( xref.data(), sizeof( T_X_FLOAT ), 3 * n, fpref );
    T_INT v_ret = fread( vref.data(), sizeof( T_V_FLOAT ), 3 * n, fpref );
    T_INT f_ret = fread( fref.data(), sizeof( T_F_FLOAT ), 3 * n, fpref );
    if ( id_ret != n || type_ret != n || q_ret != n || x_ret != 3 * n ||
         v_ret != 3 * n || f_ret != 3 * n || n_ret != 1 )
        log_err( err, "Error reading reference data." );

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
            log_err( err, "Unable to find current id matchinf reference id: ",
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
    err.close();
}
