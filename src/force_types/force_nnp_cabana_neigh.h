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
    if ( input->nnp_layout_type == AOSOA_1 )
    {
        if ( input->neighbor_type == NEIGH_2D )
        {
            if ( half_neigh )
                throw std::runtime_error( "Half neighbor list not implemented "
                                          "for the neural network potential." );
            else
            {
                if ( serial_neigh )
                    force =
                        new ForceNNP<t_System, System_NNP<AoSoA1>,
                                     t_verletlist_full_2D, t_neighborop_serial,
                                     t_neighborop_serial>( system, half_neigh );
                if ( team_neigh )
                    force =
                        new ForceNNP<t_System, System_NNP<AoSoA1>,
                                     t_verletlist_full_2D, t_neighborop_team,
                                     t_neighborop_team>( system, half_neigh );
                if ( vector_angle )
                    force =
                        new ForceNNP<t_System, System_NNP<AoSoA1>,
                                     t_verletlist_full_2D, t_neighborop_team,
                                     t_neighborop_vector>( system, half_neigh );
            }
        }
        else if ( input->neighbor_type == NEIGH_CSR )
        {
            if ( half_neigh )
                throw std::runtime_error( "Half neighbor list not implemented "
                                          "for the neural network potential." );
            else
            {
                if ( serial_neigh )
                    force =
                        new ForceNNP<t_System, System_NNP<AoSoA1>,
                                     t_verletlist_full_CSR, t_neighborop_serial,
                                     t_neighborop_serial>( system, half_neigh );
                if ( team_neigh )
                    force =
                        new ForceNNP<t_System, System_NNP<AoSoA1>,
                                     t_verletlist_full_CSR, t_neighborop_team,
                                     t_neighborop_team>( system, half_neigh );
                if ( vector_angle )
                    force =
                        new ForceNNP<t_System, System_NNP<AoSoA1>,
                                     t_verletlist_full_CSR, t_neighborop_team,
                                     t_neighborop_vector>( system, half_neigh );
            }
        }
    }
    else if ( input->nnp_layout_type == AOSOA_3 )
    {
        if ( input->neighbor_type == NEIGH_2D )
        {
            if ( half_neigh )
                throw std::runtime_error( "Half neighbor list not implemented "
                                          "for the neural network potential." );
            else
            {
                if ( serial_neigh )
                    force =
                        new ForceNNP<t_System, System_NNP<AoSoA3>,
                                     t_verletlist_full_2D, t_neighborop_serial,
                                     t_neighborop_serial>( system, half_neigh );
                if ( team_neigh )
                    force =
                        new ForceNNP<t_System, System_NNP<AoSoA3>,
                                     t_verletlist_full_2D, t_neighborop_team,
                                     t_neighborop_team>( system, half_neigh );
                if ( vector_angle )
                    force =
                        new ForceNNP<t_System, System_NNP<AoSoA3>,
                                     t_verletlist_full_2D, t_neighborop_team,
                                     t_neighborop_vector>( system, half_neigh );
            }
        }
        else if ( input->neighbor_type == NEIGH_CSR )
        {
            if ( half_neigh )
                throw std::runtime_error( "Half neighbor list not implemented "
                                          "for the neural network potential." );
            else
            {
                if ( serial_neigh )
                    force =
                        new ForceNNP<t_System, System_NNP<AoSoA3>,
                                     t_verletlist_full_CSR, t_neighborop_serial,
                                     t_neighborop_serial>( system, half_neigh );
                if ( team_neigh )
                    force =
                        new ForceNNP<t_System, System_NNP<AoSoA3>,
                                     t_verletlist_full_CSR, t_neighborop_team,
                                     t_neighborop_team>( system, half_neigh );
                if ( vector_angle )
                    force =
                        new ForceNNP<t_System, System_NNP<AoSoA3>,
                                     t_verletlist_full_CSR, t_neighborop_team,
                                     t_neighborop_vector>( system, half_neigh );
            }
        }
    }
#undef FORCETYPE_ALLOCATION_MACRO
}
#endif

#if !defined( MODULES_OPTION_CHECK ) && !defined( FORCE_MODULES_INSTANTIATION )

#ifndef FORCE_NNP_CABANA_NEIGH_H
#define FORCE_NNP_CABANA_NEIGH_H

#include <nnp_mode_impl.h>
#include <system_nnp.h>

#include <force.h>
#include <system.h>
#include <types.h>

#include <Cabana_Core.hpp>

template <class t_System, class t_System_NNP, class t_neighbor,
          class t_neigh_parallel, class t_angle_parallel>
class ForceNNP : public Force<t_System>
{
  private:
    int N_local, ntypes;
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

    bool half_neigh;
    T_X_FLOAT neigh_cut;

    nnpCbn::Mode *mode;
    t_neighbor neigh_list;

    // NNP-specific System class for AoSoAs
    // Storage of G, dEdG and energy (per atom properties)
    t_System_NNP *system_nnp;
    // numSymmetryFunctionsPerElement (per type property)
    t_mass d_numSFperElem;
    h_t_mass h_numSFperElem, atomicEnergyOffset;

    ForceNNP( t_System *system, bool half_neigh_ );
    void init_coeff( T_X_FLOAT neigh_cutoff, char **args ) override;

    void create_neigh_list( t_System *system ) override;

    void compute( t_System *system ) override;
    T_F_FLOAT compute_energy( t_System *system ) override;

    const char *name() override;

    bool showew;
    bool resetew;
    T_INT showewsum;
    T_INT maxew;
    long numExtrapolationWarningsTotal;
    long numExtrapolationWarningsSummary;
    T_FLOAT maxCutoffRadius;
    char *directory;
};

#include <force_nnp_cabana_neigh_impl.h>
#endif
#endif
