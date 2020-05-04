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

template <class t_System, class t_Neighbor>
ForceLJ<t_System, t_Neighbor>::ForceLJ( t_System *system )
    : Force<t_System, t_Neighbor>( system )
{
    ntypes = system->ntypes;

    lj1 = t_fparams( "ForceLJCabanaNeigh::lj1", ntypes, ntypes );
    lj2 = t_fparams( "ForceLJCabanaNeigh::lj2", ntypes, ntypes );
    cutsq = t_fparams( "ForceLJCabanaNeigh::cutsq", ntypes, ntypes );

    N_local = 0;
    step = 0;
}

template <class t_System, class t_Neighbor>
void ForceLJ<t_System, t_Neighbor>::init_coeff( char **args )
{
    step = 0;

    double eps = atof( args[3] );
    double sigma = atof( args[4] );
    double cut = atof( args[5] );

    for ( int i = 0; i < ntypes; i++ )
    {
        for ( int j = 0; j < ntypes; j++ )
        {
            stack_lj1[i][j] = 48.0 * eps * pow( sigma, 12.0 );
            stack_lj2[i][j] = 24.0 * eps * pow( sigma, 6.0 );
            stack_cutsq[i][j] = cut * cut;
        }
    }
}

template <class t_System, class t_Neighbor>
void ForceLJ<t_System, t_Neighbor>::compute( t_System *system,
                                             t_Neighbor *neighbor )
{
    N_local = system->N_local;
    system->slice_force();
    x = system->x;
    f = system->f;
    f_a = f;
    type = system->type;

    neigh_list = neighbor->get();

    if ( neighbor->half_neigh )
    {
        Kokkos::parallel_for(
            "ForceLJCabanaNeigh::compute",
            t_policy_half_neigh_stackparams( 0, system->N_local ), *this );
    }
    else
    {
        Kokkos::parallel_for(
            "ForceLJCabanaNeigh::compute",
            t_policy_full_neigh_stackparams( 0, system->N_local ), *this );
    }
    Kokkos::fence();

    step++;
}

template <class t_System, class t_Neighbor>
T_V_FLOAT ForceLJ<t_System, t_Neighbor>::compute_energy( t_System *system,
                                                         t_Neighbor *neighbor )
{
    N_local = system->N_local;
    system->slice_force();
    x = system->x;
    f = system->f;
    f_a = f;
    type = system->type;

    neigh_list = neighbor->get();

    T_V_FLOAT energy;

    if ( neighbor->half_neigh )
        Kokkos::parallel_reduce(
            "ForceLJCabanaNeigh::compute_energy",
            t_policy_half_neigh_pe_stackparams( 0, system->N_local ), *this,
            energy );
    else
        Kokkos::parallel_reduce(
            "ForceLJCabanaNeigh::compute_energy",
            t_policy_full_neigh_pe_stackparams( 0, system->N_local ), *this,
            energy );

    Kokkos::fence();

    step++;
    return energy;
}

template <class t_System, class t_Neighbor>
const char *ForceLJ<t_System, t_Neighbor>::name()
{
    return "Force:LJCabana";
}

template <class t_System, class t_Neighbor>
KOKKOS_INLINE_FUNCTION void ForceLJ<t_System, t_Neighbor>::
operator()( TagFullNeigh, const T_INT &i ) const
{
    const T_F_FLOAT x_i = x( i, 0 );
    const T_F_FLOAT y_i = x( i, 1 );
    const T_F_FLOAT z_i = x( i, 2 );
    const int type_i = type( i );

    int num_neighs =
        Cabana::NeighborList<t_neigh_list>::numNeighbor( neigh_list, i );

    T_F_FLOAT fxi = 0.0;
    T_F_FLOAT fyi = 0.0;
    T_F_FLOAT fzi = 0.0;

    for ( int jj = 0; jj < num_neighs; jj++ )
    {
        int j = Cabana::NeighborList<t_neigh_list>::getNeighbor( neigh_list, i,
                                                                 jj );

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
    }

    f( i, 0 ) += fxi;
    f( i, 1 ) += fyi;
    f( i, 2 ) += fzi;
}

template <class t_System, class t_Neighbor>
KOKKOS_INLINE_FUNCTION void ForceLJ<t_System, t_Neighbor>::
operator()( TagHalfNeigh, const T_INT &i ) const
{
    const T_F_FLOAT x_i = x( i, 0 );
    const T_F_FLOAT y_i = x( i, 1 );
    const T_F_FLOAT z_i = x( i, 2 );
    const int type_i = type( i );

    int num_neighs =
        Cabana::NeighborList<t_neigh_list>::numNeighbor( neigh_list, i );

    T_F_FLOAT fxi = 0.0;
    T_F_FLOAT fyi = 0.0;
    T_F_FLOAT fzi = 0.0;
    for ( int jj = 0; jj < num_neighs; jj++ )
    {
        int j = Cabana::NeighborList<t_neigh_list>::getNeighbor( neigh_list, i,
                                                                 jj );

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
    }
    f_a( i, 0 ) += fxi;
    f_a( i, 1 ) += fyi;
    f_a( i, 2 ) += fzi;
}

template <class t_System, class t_Neighbor>
KOKKOS_INLINE_FUNCTION void ForceLJ<t_System, t_Neighbor>::
operator()( TagFullNeighPE, const T_INT &i, T_V_FLOAT &PE ) const
{
    const T_F_FLOAT x_i = x( i, 0 );
    const T_F_FLOAT y_i = x( i, 1 );
    const T_F_FLOAT z_i = x( i, 2 );
    const int type_i = type( i );
    const bool shift_flag = true;

    int num_neighs =
        Cabana::NeighborList<t_neigh_list>::numNeighbor( neigh_list, i );

    for ( int jj = 0; jj < num_neighs; jj++ )
    {
        int j = Cabana::NeighborList<t_neigh_list>::getNeighbor( neigh_list, i,
                                                                 jj );

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
    }
}

template <class t_System, class t_Neighbor>
KOKKOS_INLINE_FUNCTION void ForceLJ<t_System, t_Neighbor>::
operator()( TagHalfNeighPE, const T_INT &i, T_V_FLOAT &PE ) const
{
    const T_F_FLOAT x_i = x( i, 0 );
    const T_F_FLOAT y_i = x( i, 1 );
    const T_F_FLOAT z_i = x( i, 2 );
    const int type_i = type( i );
    const bool shift_flag = true;

    int num_neighs =
        Cabana::NeighborList<t_neigh_list>::numNeighbor( neigh_list, i );

    for ( int jj = 0; jj < num_neighs; jj++ )
    {
        int j = Cabana::NeighborList<t_neigh_list>::getNeighbor( neigh_list, i,
                                                                 jj );

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
    }
}
