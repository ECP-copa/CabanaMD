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

template <class t_System, class t_Neighbor, class t_parallel>
ForceLJ<t_System, t_Neighbor, t_parallel>::ForceLJ( t_System *system )
    : Force<t_System, t_Neighbor>( system )
{
    ntypes = system->ntypes;

    lj1 = t_fparams( "ForceLJCabanaNeigh::lj1", ntypes, ntypes );
    lj2 = t_fparams( "ForceLJCabanaNeigh::lj2", ntypes, ntypes );
    cutsq = t_fparams( "ForceLJCabanaNeigh::cutsq", ntypes, ntypes );

    N_local = 0;
    step = 0;
}

template <class t_System, class t_Neighbor, class t_parallel>
void ForceLJ<t_System, t_Neighbor, t_parallel>::init_coeff(
    std::vector<std::vector<std::string>> args )
{
    for ( std::size_t a = 0; a < args.size(); a++ )
    {
        auto pair = args.at( a );
        int i = std::stoi( pair.at( 1 ) ) - 1;
        int j = std::stoi( pair.at( 2 ) ) - 1;
        double eps = std::stod( pair.at( 3 ) );
        double sigma = std::stod( pair.at( 4 ) );
        double cut = std::stod( pair.at( 5 ) );

        stack_lj1[i][j] = 48.0 * eps * pow( sigma, 12.0 );
        stack_lj2[i][j] = 24.0 * eps * pow( sigma, 6.0 );
        stack_cutsq[i][j] = cut * cut;
        stack_lj1[j][i] = stack_lj1[i][j];
        stack_lj2[j][i] = stack_lj2[i][j];
        stack_cutsq[j][i] = stack_cutsq[i][j];
    }
}

template <class t_System, class t_Neighbor, class t_parallel>
void ForceLJ<t_System, t_Neighbor, t_parallel>::compute( t_System *system,
                                                         t_Neighbor *neighbor )
{
    N_local = system->N_local;
    system->slice_force();
    auto x = system->x;
    auto f = system->f;
    t_f_a f_a = system->f;
    auto type = system->type;

    auto neigh_list = neighbor->get();

    if ( neighbor->half_neigh )
    {
        // Forces must be atomic for half list
        compute_force_half( f_a, x, type, neigh_list );
    }
    else
    {
        // Forces only atomic if using team threading
        if ( std::is_same<t_parallel, Cabana::TeamOpTag>::value )
            compute_force_full( f_a, x, type, neigh_list );
        else
            compute_force_full( f, x, type, neigh_list );
    }
    Kokkos::fence();

    step++;
}

template <class t_System, class t_Neighbor, class t_parallel>
T_F_FLOAT ForceLJ<t_System, t_Neighbor, t_parallel>::compute_energy(
    t_System *system, t_Neighbor *neighbor )
{
    N_local = system->N_local;
    system->slice_force();
    auto x = system->x;
    auto f = system->f;
    auto type = system->type;

    auto neigh_list = neighbor->get();

    T_F_FLOAT energy;
    if ( neighbor->half_neigh )
        energy = compute_energy_half( x, type, neigh_list );
    else
        energy = compute_energy_full( x, type, neigh_list );
    Kokkos::fence();

    step++;
    return energy;
}

template <class t_System, class t_Neighbor, class t_parallel>
const char *ForceLJ<t_System, t_Neighbor, t_parallel>::name()
{
    return "Force:LJCabana";
}

template <class t_System, class t_Neighbor, class t_parallel>
template <class t_f, class t_x, class t_type, class t_neigh>
void ForceLJ<t_System, t_Neighbor, t_parallel>::compute_force_full(
    t_f f, const t_x x, const t_type type, const t_neigh neigh_list )
{
    auto force_full = KOKKOS_LAMBDA( const int i, const int j )
    {
        const T_F_FLOAT x_i = x( i, 0 );
        const T_F_FLOAT y_i = x( i, 1 );
        const T_F_FLOAT z_i = x( i, 2 );
        const int type_i = type( i );

        T_F_FLOAT fxi = 0.0;
        T_F_FLOAT fyi = 0.0;
        T_F_FLOAT fzi = 0.0;

        const T_F_FLOAT dx = x_i - x( j, 0 );
        const T_F_FLOAT dy = y_i - x( j, 1 );
        const T_F_FLOAT dz = z_i - x( j, 2 );

        const int type_j = type( j );
        const T_F_FLOAT rsq = dx * dx + dy * dy + dz * dz;

        const T_F_FLOAT cutsq_ij = stack_cutsq[type_i][type_j];

        if ( rsq < cutsq_ij )
        {
            const T_F_FLOAT lj1_ij = stack_lj1[type_i][type_j];
            const T_F_FLOAT lj2_ij = stack_lj2[type_i][type_j];

            T_F_FLOAT r2inv = 1.0 / rsq;
            T_F_FLOAT r6inv = r2inv * r2inv * r2inv;
            T_F_FLOAT fpair = ( r6inv * ( lj1_ij * r6inv - lj2_ij ) ) * r2inv;
            fxi += dx * fpair;
            fyi += dy * fpair;
            fzi += dz * fpair;
        }

        f( i, 0 ) += fxi;
        f( i, 1 ) += fyi;
        f( i, 2 ) += fzi;
    };

    Kokkos::RangePolicy<exe_space> policy( 0, N_local );
    t_parallel neigh_parallel;
    Cabana::neighbor_parallel_for( policy, force_full, neigh_list,
                                   Cabana::FirstNeighborsTag(), neigh_parallel,
                                   "ForceLJCabanaNeigh::compute_full" );
}

template <class t_System, class t_Neighbor, class t_parallel>
template <class t_f, class t_x, class t_type, class t_neigh>
void ForceLJ<t_System, t_Neighbor, t_parallel>::compute_force_half(
    t_f f_a, const t_x x, const t_type type, const t_neigh neigh_list )
{
    auto force_half = KOKKOS_LAMBDA( const int i, const int j )
    {
        const T_F_FLOAT x_i = x( i, 0 );
        const T_F_FLOAT y_i = x( i, 1 );
        const T_F_FLOAT z_i = x( i, 2 );
        const int type_i = type( i );

        T_F_FLOAT fxi = 0.0;
        T_F_FLOAT fyi = 0.0;
        T_F_FLOAT fzi = 0.0;

        const T_F_FLOAT dx = x_i - x( j, 0 );
        const T_F_FLOAT dy = y_i - x( j, 1 );
        const T_F_FLOAT dz = z_i - x( j, 2 );

        const int type_j = type( j );
        const T_F_FLOAT rsq = dx * dx + dy * dy + dz * dz;

        const T_F_FLOAT cutsq_ij = stack_cutsq[type_i][type_j];

        if ( rsq < cutsq_ij )
        {
            const T_F_FLOAT lj1_ij = stack_lj1[type_i][type_j];
            const T_F_FLOAT lj2_ij = stack_lj2[type_i][type_j];

            T_F_FLOAT r2inv = 1.0 / rsq;
            T_F_FLOAT r6inv = r2inv * r2inv * r2inv;
            T_F_FLOAT fpair = ( r6inv * ( lj1_ij * r6inv - lj2_ij ) ) * r2inv;
            fxi += dx * fpair;
            fyi += dy * fpair;
            fzi += dz * fpair;
            f_a( j, 0 ) -= dx * fpair;
            f_a( j, 1 ) -= dy * fpair;
            f_a( j, 2 ) -= dz * fpair;
        }
        f_a( i, 0 ) += fxi;
        f_a( i, 1 ) += fyi;
        f_a( i, 2 ) += fzi;
    };

    Kokkos::RangePolicy<exe_space> policy( 0, N_local );
    t_parallel neigh_parallel;
    Cabana::neighbor_parallel_for( policy, force_half, neigh_list,
                                   Cabana::FirstNeighborsTag(), neigh_parallel,
                                   "ForceLJCabanaNeigh::compute_half" );
}

template <class t_System, class t_Neighbor, class t_parallel>
template <class t_x, class t_type, class t_neigh>
T_F_FLOAT ForceLJ<t_System, t_Neighbor, t_parallel>::compute_energy_full(
    const t_x x, const t_type type, const t_neigh neigh_list )
{
    auto energy_full = KOKKOS_LAMBDA( const int i, const int j, T_F_FLOAT &PE )
    {
        const T_F_FLOAT x_i = x( i, 0 );
        const T_F_FLOAT y_i = x( i, 1 );
        const T_F_FLOAT z_i = x( i, 2 );
        const int type_i = type( i );
        const bool shift_flag = true;

        const T_F_FLOAT dx = x_i - x( j, 0 );
        const T_F_FLOAT dy = y_i - x( j, 1 );
        const T_F_FLOAT dz = z_i - x( j, 2 );

        const int type_j = type( j );
        const T_F_FLOAT rsq = dx * dx + dy * dy + dz * dz;

        const T_F_FLOAT cutsq_ij = stack_cutsq[type_i][type_j];

        if ( rsq < cutsq_ij )
        {
            const T_F_FLOAT lj1_ij = stack_lj1[type_i][type_j];
            const T_F_FLOAT lj2_ij = stack_lj2[type_i][type_j];

            T_F_FLOAT r2inv = 1.0 / rsq;
            T_F_FLOAT r6inv = r2inv * r2inv * r2inv;
            PE += 0.5 * r6inv * ( 0.5 * lj1_ij * r6inv - lj2_ij ) /
                  6.0; // optimize later

            if ( shift_flag )
            {
                T_F_FLOAT r2invc = 1.0 / cutsq_ij;
                T_F_FLOAT r6invc = r2invc * r2invc * r2invc;
                PE -= 0.5 * r6invc * ( 0.5 * lj1_ij * r6invc - lj2_ij ) /
                      6.0; // optimize later
            }
        }
    };

    T_F_FLOAT energy = 0.0;
    Kokkos::RangePolicy<exe_space> policy( 0, N_local );
    t_parallel neigh_parallel;
    Cabana::neighbor_parallel_reduce(
        policy, energy_full, neigh_list, Cabana::FirstNeighborsTag(),
        neigh_parallel, energy, "ForceLJCabanaNeigh::compute_half" );
    return energy;
}

template <class t_System, class t_Neighbor, class t_parallel>
template <class t_x, class t_type, class t_neigh>
T_F_FLOAT ForceLJ<t_System, t_Neighbor, t_parallel>::compute_energy_half(
    const t_x x, const t_type type, const t_neigh neigh_list )
{
    auto energy_half = KOKKOS_LAMBDA( const int i, const int j, T_F_FLOAT &PE )
    {

        const T_F_FLOAT x_i = x( i, 0 );
        const T_F_FLOAT y_i = x( i, 1 );
        const T_F_FLOAT z_i = x( i, 2 );
        const int type_i = type( i );
        const bool shift_flag = true;

        const T_F_FLOAT dx = x_i - x( j, 0 );
        const T_F_FLOAT dy = y_i - x( j, 1 );
        const T_F_FLOAT dz = z_i - x( j, 2 );

        const int type_j = type( j );
        const T_F_FLOAT rsq = dx * dx + dy * dy + dz * dz;

        const T_F_FLOAT cutsq_ij = stack_cutsq[type_i][type_j];

        if ( rsq < cutsq_ij )
        {
            const T_F_FLOAT lj1_ij = stack_lj1[type_i][type_j];
            const T_F_FLOAT lj2_ij = stack_lj2[type_i][type_j];

            T_F_FLOAT r2inv = 1.0 / rsq;
            T_F_FLOAT r6inv = r2inv * r2inv * r2inv;
            T_F_FLOAT fac;
            if ( j < N_local )
                fac = 1.0;
            else
                fac = 0.5;

            PE += fac * r6inv * ( 0.5 * lj1_ij * r6inv - lj2_ij ) /
                  6.0; // optimize later

            if ( shift_flag )
            {
                T_F_FLOAT r2invc = 1.0 / cutsq_ij;
                T_F_FLOAT r6invc = r2invc * r2invc * r2invc;
                PE -= fac * r6invc * ( 0.5 * lj1_ij * r6invc - lj2_ij ) /
                      6.0; // optimize later
            }
        }
    };

    T_F_FLOAT energy = 0.0;
    Kokkos::RangePolicy<exe_space> policy( 0, N_local );
    t_parallel neigh_parallel;
    Cabana::neighbor_parallel_reduce(
        policy, energy_half, neigh_list, Cabana::FirstNeighborsTag(),
        neigh_parallel, energy, "ForceLJCabanaNeigh::compute_half" );
    return energy;
}
