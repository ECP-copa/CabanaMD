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

#ifdef MODULES_OPTION_CHECK
if ( ( strcmp( argv[i], "--force-iteration" ) == 0 ) )
{
    if ( ( strcmp( argv[i + 1], "NEIGH_FULL" ) == 0 ) )
        force_iteration_type = FORCE_ITER_NEIGH_FULL;
    if ( ( strcmp( argv[i + 1], "NEIGH_HALF" ) == 0 ) )
        force_iteration_type = FORCE_ITER_NEIGH_HALF;
}
if ( ( strcmp( argv[i], "--neigh-parallel" ) == 0 ) )
{
    if ( ( strcmp( argv[i + 1], "SERIAL" ) == 0 ) )
        force_neigh_parallel_type = FORCE_PARALLEL_NEIGH_SERIAL;
    if ( ( strcmp( argv[i + 1], "TEAM" ) == 0 ) )
        force_neigh_parallel_type = FORCE_PARALLEL_NEIGH_TEAM;
    if ( ( strcmp( argv[i + 1], "TEAM_VECTOR" ) == 0 ) )
        force_neigh_parallel_type = FORCE_PARALLEL_NEIGH_VECTOR;
}
if ( ( strcmp( argv[i], "--neigh-type" ) == 0 ) )
{
    if ( ( strcmp( argv[i + 1], "NEIGH_VERLET_2D" ) == 0 ) )
        neighbor_type = NEIGH_VERLET_2D;
    if ( ( strcmp( argv[i + 1], "NEIGH_VERLET_CSR" ) == 0 ) )
        neighbor_type = NEIGH_VERLET_CSR;
}
#endif
#ifdef FORCE_MODULES_INSTANTIATION
else if ( input->force_type == FORCE_NNP )
{
    bool half_neigh = input->force_iteration_type == FORCE_ITER_NEIGH_HALF;
    bool serial_neigh =
        input->force_neigh_parallel_type == FORCE_PARALLEL_NEIGH_SERIAL;
    bool team_neigh =
        input->force_neigh_parallel_type == FORCE_PARALLEL_NEIGH_TEAM;
    bool vector_angle =
        input->force_neigh_parallel_type == FORCE_PARALLEL_NEIGH_VECTOR;
    if ( input->neighbor_type == NEIGH_VERLET_2D )
    {
        if ( half_neigh )
            throw std::runtime_error( "Half neighbor list not implemented "
                                      "for the neural network potential." );
        else
        {
            if ( serial_neigh )
                force = new ForceNNP<t_verletlist_full_2D, t_neighborop_serial,
                                     t_neighborop_serial>( system );
            if ( team_neigh )
                force = new ForceNNP<t_verletlist_full_2D, t_neighborop_team,
                                     t_neighborop_team>( system );
            if ( vector_angle )
                force = new ForceNNP<t_verletlist_full_2D, t_neighborop_team,
                                     t_neighborop_vector>( system );
        }
    }
    else if ( input->neighbor_type == NEIGH_VERLET_CSR )
    {
        if ( half_neigh )
            throw std::runtime_error( "Half neighbor list not implemented "
                                      "for the neural network potential." );
        else
        {
            if ( serial_neigh )
                force = new ForceNNP<t_verletlist_full_CSR, t_neighborop_serial,
                                     t_neighborop_serial>( system );
            if ( team_neigh )
                force = new ForceNNP<t_verletlist_full_CSR, t_neighborop_team,
                                     t_neighborop_team>( system );
            if ( vector_angle )
                force = new ForceNNP<t_verletlist_full_CSR, t_neighborop_team,
                                     t_neighborop_vector>( system );
        }
    }
#undef FORCETYPE_ALLOCATION_MACRO
}
#endif

#if !defined( MODULES_OPTION_CHECK ) && !defined( FORCE_MODULES_INSTANTIATION )

#ifndef FORCE_NNP_CABANA_NEIGH_H
#define FORCE_NNP_CABANA_NEIGH_H

#include <nnp_mode_impl.h>

#include <force.h>
#include <neighbor.h>
#include <system.h>
#include <types.h>

#include <Cabana_Core.hpp>

template <class t_neighbor, class t_neigh_parallel, class t_angle_parallel>
class ForceNNP : public Force
{
  private:
    int N_local, ntypes;
    typename AoSoA::member_slice_type<Positions> x;
    typename AoSoA::member_slice_type<Forces> f;
    typename AoSoA::member_slice_type<Forces>::atomic_access_slice f_a;
    typename AoSoA::member_slice_type<IDs> id;
    typename AoSoA::member_slice_type<Types> type;

    int step;

  public:
    struct TagFullNeigh
    {
    };

    struct TagHalfNeigh
    {
    };

    struct TagFullNeighPE
    {
    };

    struct TagHalfNeighPE
    {
    };

    typedef Kokkos::RangePolicy<TagFullNeigh, Kokkos::IndexType<T_INT>>
        t_policy_full_neigh_stackparams;
    typedef Kokkos::RangePolicy<TagHalfNeigh, Kokkos::IndexType<T_INT>>
        t_policy_half_neigh_stackparams;
    typedef Kokkos::RangePolicy<TagFullNeighPE, Kokkos::IndexType<T_INT>>
        t_policy_full_neigh_pe_stackparams;
    typedef Kokkos::RangePolicy<TagHalfNeighPE, Kokkos::IndexType<T_INT>>
        t_policy_half_neigh_pe_stackparams;

    nnpCbn::Mode *mode;

    /// AoSoAs of use to compute energy and force
    /// Allow storage of G, dEdG and energy (per atom properties)
    AoSoA_NNP nnp_data;
    // numSymmetryFunctionsPerElement (per type property)
    t_mass d_numSFperElem;
    h_t_mass h_numSFperElem, atomicEnergyOffset;

    ForceNNP( System *system );
    void init_coeff( char **args );

    void compute( System *system, Neighbor *neighbor );
    T_F_FLOAT compute_energy( System *system, Neighbor *neighbor );

    const char *name();

    bool showew;
    bool resetew;
    T_INT showewsum;
    T_INT maxew;
    long numExtrapolationWarningsTotal;
    long numExtrapolationWarningsSummary;
    T_FLOAT maxCutoffRadius;
    char *directory;
};

#endif
#endif
