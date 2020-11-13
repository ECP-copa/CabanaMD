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

#include <mpi.h>

#include <algorithm>

template <class t_System>
Comm<t_System>::Comm( t_System *s, T_X_FLOAT comm_depth_ )
    : neighbors_halo( 6 )
    , neighbors_dist( 6 )
    , halo_all( 6 )
    , system( s )
    , comm_depth( comm_depth_ )
{
    MPI_Comm_size( MPI_COMM_WORLD, &proc_size );
    MPI_Comm_rank( MPI_COMM_WORLD, &proc_rank );

    pack_count = Kokkos::View<int, Kokkos::LayoutRight, device_type>(
        "CommMPI::pack_count" );
    pack_indicies_all =
        Kokkos::View<T_INT **, Kokkos::LayoutRight, device_type>(
            "CommMPI::pack_indicies_all", 6, 200 );
    pack_ranks_all = Kokkos::View<T_INT **, Kokkos::LayoutRight, device_type>(
        "CommMPI::pack_ranks_all", 6, 200 );
}

template <class t_System>
void Comm<t_System>::init()
{
}

template <class t_System>
void Comm<t_System>::create_domain_decomposition()
{
    for ( int d = 0; d < 3; d++ )
    {
        proc_grid[d] = system->ranks_per_dim[d];
        proc_pos[d] = system->rank_dim_pos[d];
    }

    proc_neighbors_send[0] = system->local_grid->neighborRank( 1, 0, 0 );
    proc_neighbors_send[1] = system->local_grid->neighborRank( -1, 0, 0 );
    proc_neighbors_send[2] = system->local_grid->neighborRank( 0, 1, 0 );
    proc_neighbors_send[3] = system->local_grid->neighborRank( 0, -1, 0 );
    proc_neighbors_send[4] = system->local_grid->neighborRank( 0, 0, 1 );
    proc_neighbors_send[5] = system->local_grid->neighborRank( 0, 0, -1 );

    proc_neighbors_recv[0] = proc_neighbors_send[1];
    proc_neighbors_recv[1] = proc_neighbors_send[0];
    proc_neighbors_recv[2] = proc_neighbors_send[3];
    proc_neighbors_recv[3] = proc_neighbors_send[2];
    proc_neighbors_recv[4] = proc_neighbors_send[5];
    proc_neighbors_recv[5] = proc_neighbors_send[4];

    for ( int p = 0; p < 6; p++ )
    {
        neighbors_dist[p] = {proc_rank, proc_neighbors_send[p],
                             proc_neighbors_recv[p]};
        neighbors_halo[p] = {proc_neighbors_send[p], proc_neighbors_recv[p]};

        std::sort( neighbors_halo[p].begin(), neighbors_halo[p].end() );
        auto unique_end =
            std::unique( neighbors_halo[p].begin(), neighbors_halo[p].end() );
        neighbors_halo[p].resize(
            std::distance( neighbors_halo[p].begin(), unique_end ) );

        std::sort( neighbors_dist[p].begin(), neighbors_dist[p].end() );
        unique_end =
            std::unique( neighbors_dist[p].begin(), neighbors_dist[p].end() );
        neighbors_dist[p].resize(
            std::distance( neighbors_dist[p].begin(), unique_end ) );
    }
}

template <class t_System>
void Comm<t_System>::scan_int( T_INT *vals, T_INT count )
{
    if ( std::is_same<T_INT, int>::value )
    {
        MPI_Scan( MPI_IN_PLACE, vals, count, MPI_INT, MPI_SUM, MPI_COMM_WORLD );
    }
}

template <class t_System>
void Comm<t_System>::reduce_int( T_INT *vals, T_INT count )
{
    if ( std::is_same<T_INT, int>::value )
    {
        MPI_Allreduce( MPI_IN_PLACE, vals, count, MPI_INT, MPI_SUM,
                       MPI_COMM_WORLD );
    }
}

template <class t_System>
void Comm<t_System>::reduce_float( T_FLOAT *vals, T_INT count )
{
    if ( std::is_same<T_FLOAT, double>::value )
    {
        // This generates MPI_ERR_BUFFER for count>1
        MPI_Allreduce( MPI_IN_PLACE, vals, count, MPI_DOUBLE, MPI_SUM,
                       MPI_COMM_WORLD );
    }
}

template <class t_System>
void Comm<t_System>::reduce_max_int( T_INT *vals, T_INT count )
{
    if ( std::is_same<T_INT, int>::value )
    {
        MPI_Allreduce( MPI_IN_PLACE, vals, count, MPI_INT, MPI_MAX,
                       MPI_COMM_WORLD );
    }
}

template <class t_System>
void Comm<t_System>::reduce_max_float( T_FLOAT *vals, T_INT count )
{
    if ( std::is_same<T_FLOAT, double>::value )
    {
        MPI_Allreduce( MPI_IN_PLACE, vals, count, MPI_DOUBLE, MPI_MAX,
                       MPI_COMM_WORLD );
    }
}

template <class t_System>
void Comm<t_System>::reduce_min_int( T_INT *vals, T_INT count )
{
    if ( std::is_same<T_INT, int>::value )
    {
        MPI_Allreduce( MPI_IN_PLACE, vals, count, MPI_INT, MPI_MAX,
                       MPI_COMM_WORLD );
    }
}

template <class t_System>
void Comm<t_System>::reduce_min_float( T_FLOAT *vals, T_INT count )
{
    if ( std::is_same<T_FLOAT, double>::value )
    {
        MPI_Allreduce( MPI_IN_PLACE, vals, count, MPI_DOUBLE, MPI_MAX,
                       MPI_COMM_WORLD );
    }
}

template <class t_System>
void Comm<t_System>::exchange()
{

    Kokkos::Profiling::pushRegion( "Comm::exchange" );

    N_local = system->N_local;
    system->resize( N_local );
    system->slice_x();
    s = *system;
    x = s.x;

    max_local = x.size() * 1.1;

    std::shared_ptr<Cabana::Distributor<device_type>> distributor;

    pack_ranks_migrate_all =
        Kokkos::View<T_INT *, Kokkos::LayoutRight, device_type>(
            "pack_ranks_migrate", max_local );
    Kokkos::parallel_for(
        "CommMPI::exchange_self",
        Kokkos::RangePolicy<exe_space, TagExchangeSelf,
                            Kokkos::IndexType<T_INT>>( 0, N_local ),
        *this );

    T_INT N_total_recv = 0;
    T_INT N_total_send = 0;

    for ( phase = 0; phase < 6; phase++ )
    {
        proc_num_send[phase] = 0;
        proc_num_recv[phase] = 0;

        T_INT count = 0;
        Kokkos::deep_copy( pack_count, 0 );

        if ( proc_grid[phase / 2] > 1 )
        {
            // If a previous phase resized the AoSoA, export ranks needs to be
            // resized as well
            if ( pack_ranks_migrate_all.extent( 0 ) < x.size() )
            {
                max_local *= 1.1;
                Kokkos::realloc( pack_ranks_migrate_all, max_local );
            }
            pack_ranks_migrate =
                Kokkos::subview( pack_ranks_migrate_all,
                                 std::pair<size_t, size_t>( 0, x.size() ) );
            Kokkos::deep_copy( pack_ranks_migrate, proc_rank );

            Kokkos::parallel_for(
                "CommMPI::exchange_pack",
                Kokkos::RangePolicy<exe_space, TagExchangePack,
                                    Kokkos::IndexType<T_INT>>( 0, x.size() ),
                *this );

            Kokkos::deep_copy( count, pack_count );
            proc_num_send[phase] = count;

            distributor = std::make_shared<Cabana::Distributor<device_type>>(
                MPI_COMM_WORLD, pack_ranks_migrate, neighbors_dist[phase] );
            system->migrate( distributor );
            system->resize(
                distributor->totalNumImport() ); // Resized by migrate, but not
                                                 // within System
            system->slice_x();
            s = *system;
            x = s.x;

            proc_num_recv[phase] = distributor->totalNumImport() + count -
                                   distributor->totalNumExport();
            count = proc_num_recv[phase];
        }

        N_total_recv += proc_num_recv[phase];
        N_total_send += proc_num_send[phase];
    }

    N_local = N_local + N_total_recv - N_total_send;

    system->N_local = N_local;
    system->N_ghost = 0;

    Kokkos::Profiling::popRegion();
}

template <class t_System>
void Comm<t_System>::exchange_halo()
{

    Kokkos::Profiling::pushRegion( "Comm::exchange_halo" );

    N_local = system->N_local;
    N_ghost = 0;

    system->slice_x();
    s = *system;
    x = s.x;

    for ( phase = 0; phase < 6; phase++ )
    {
        pack_indicies =
            Kokkos::subview( pack_indicies_all, phase, Kokkos::ALL() );
        pack_ranks = Kokkos::subview( pack_ranks_all, phase, Kokkos::ALL() );

        T_INT count = 0;
        Kokkos::deep_copy( pack_count, 0 );

        T_INT nparticles =
            N_local + N_ghost -
            ( ( phase % 2 == 1 ) ? proc_num_recv[phase - 1] : 0 );
        Kokkos::parallel_for(
            "CommMPI::halo_exchange_pack",
            Kokkos::RangePolicy<exe_space, TagHaloPack,
                                Kokkos::IndexType<T_INT>>( 0, nparticles ),
            *this );

        Kokkos::deep_copy( count, pack_count );
        if ( (unsigned)count > pack_indicies.extent( 0 ) )
        {
            Kokkos::resize( pack_indicies_all, 6, count * 1.1 );
            pack_indicies =
                Kokkos::subview( pack_indicies_all, phase, Kokkos::ALL() );
            Kokkos::resize( pack_ranks_all, 6, count * 1.1 );
            pack_ranks =
                Kokkos::subview( pack_ranks_all, phase, Kokkos::ALL() );

            Kokkos::deep_copy( pack_count, 0 );
            Kokkos::parallel_for(
                "CommMPI::halo_exchange_pack",
                Kokkos::RangePolicy<exe_space, TagHaloPack,
                                    Kokkos::IndexType<T_INT>>( 0, nparticles ),
                *this );
        }
        proc_num_send[phase] = count;

        pack_indicies = Kokkos::subview(
            pack_indicies,
            std::pair<size_t, size_t>( 0, proc_num_send[phase] ) );
        pack_ranks = Kokkos::subview(
            pack_ranks, std::pair<size_t, size_t>( 0, proc_num_send[phase] ) );

        halo_all[phase] = std::make_shared<Cabana::Halo<device_type>>(
            MPI_COMM_WORLD, N_local + N_ghost, pack_indicies, pack_ranks,
            neighbors_halo[phase] );
        system->resize( halo_all[phase]->numLocal() +
                        halo_all[phase]->numGhost() );
        system->slice_x();
        system->slice_type();
        s = *system;
        x = s.x;
        type = s.type;

        Cabana::gather( *halo_all[phase], x );
        Cabana::gather( *halo_all[phase], type );

        proc_num_recv[phase] = halo_all[phase]->numGhost();
        count = proc_num_recv[phase];

        Kokkos::deep_copy( pack_count, 0 );
        Kokkos::parallel_for(
            "CommMPI::halo_exchange_pack_wrap",
            Kokkos::RangePolicy<exe_space, TagHaloPBC,
                                Kokkos::IndexType<T_INT>>(
                halo_all[phase]->numLocal(),
                halo_all[phase]->numLocal() + halo_all[phase]->numGhost() ),
            *this );

        N_ghost += count;
    }

    system->N_ghost = N_ghost;

    Kokkos::Profiling::popRegion();
}

template <class t_System>
void Comm<t_System>::update_halo()
{

    Kokkos::Profiling::pushRegion( "Comm::update_halo" );

    N_local = system->N_local;
    N_ghost = 0;
    system->slice_x();
    s = *system;
    x = s.x;

    for ( phase = 0; phase < 6; phase++ )
    {
        pack_indicies = Kokkos::subview(
            pack_indicies_all, phase,
            std::pair<size_t, size_t>( 0, proc_num_send[phase] ) );
        pack_ranks = Kokkos::subview(
            pack_ranks_all, phase,
            std::pair<size_t, size_t>( 0, proc_num_send[phase] ) );

        system->resize( halo_all[phase]->numLocal() +
                        halo_all[phase]->numGhost() );
        system->slice_x();
        s = *system;
        x = s.x;
        Cabana::gather( *halo_all[phase], x );

        Kokkos::parallel_for(
            "CommMPI::halo_update_PBC",
            Kokkos::RangePolicy<exe_space, TagHaloPBC,
                                Kokkos::IndexType<T_INT>>(
                halo_all[phase]->numLocal(),
                halo_all[phase]->numLocal() + halo_all[phase]->numGhost() ),
            *this );

        N_ghost += proc_num_recv[phase];
    }

    Kokkos::Profiling::popRegion();
}

template <class t_System>
void Comm<t_System>::update_force()
{

    Kokkos::Profiling::pushRegion( "Comm::update_force" );

    N_local = system->N_local;
    N_ghost = 0;
    system->slice_f();
    s = *system;
    f = s.f;

    for ( phase = 5; phase >= 0; phase-- )
    {
        pack_indicies = Kokkos::subview(
            pack_indicies_all, phase,
            std::pair<size_t, size_t>( 0, proc_num_send[phase] ) );
        pack_ranks = Kokkos::subview(
            pack_ranks_all, phase,
            std::pair<size_t, size_t>( 0, proc_num_send[phase] ) );

        system->resize( halo_all[phase]->numLocal() +
                        halo_all[phase]->numGhost() );
        system->slice_f();
        s = *system;
        f = s.f;
        Cabana::scatter( *halo_all[phase], f );

        N_ghost += proc_num_recv[phase];
    }

    Kokkos::Profiling::popRegion();
}

template <class t_System>
const char *Comm<t_System>::name()
{
    return "Comm:CabanaMPI";
}

template <class t_System>
int Comm<t_System>::process_rank()
{
    return proc_rank;
}

template <class t_System>
int Comm<t_System>::num_processes()
{
    return proc_size;
}
