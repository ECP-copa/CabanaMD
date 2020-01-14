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
        lrforce_iteration_type = FORCE_ITER_NEIGH_FULL;
    if ( ( strcmp( argv[i + 1], "NEIGH_HALF" ) == 0 ) )
        lrforce_iteration_type = FORCE_ITER_NEIGH_HALF;
}
if ( ( strcmp( argv[i], "--neigh-type" ) == 0 ) )
{
    if ( ( strcmp( argv[i + 1], "NEIGH_2D" ) == 0 ) )
        neighbor_type = NEIGH_2D;
    if ( ( strcmp( argv[i + 1], "NEIGH_CSR" ) == 0 ) )
        neighbor_type = NEIGH_CSR;
}
#endif
#ifdef LONGRANGE_FORCE_MODULES_INSTANTIATION
else if ( input->lrforce_type == FORCE_EWALD )
{
    bool half_neigh = input->lrforce_iteration_type == FORCE_ITER_NEIGH_HALF;
    if ( input->neighbor_type == NEIGH_2D )
    {
        if ( half_neigh )
            lrforce =
                new ForceEwald<t_verletlist_half_2D>( system, half_neigh );
        else
            throw std::runtime_error( "Full neighbor list not implemented "
                                      "for the Ewald longrange solver." );
    }
    else if ( input->neighbor_type == NEIGH_CSR )
    {
        if ( half_neigh )
            lrforce =
                new ForceEwald<t_verletlist_half_CSR>( system, half_neigh );
        else
            throw std::runtime_error( "Full neighbor list not implemented "
                                      "for the Ewald longrange solver." );
    }
#undef FORCETYPE_ALLOCATION_MACRO
}
#endif

#if !defined( MODULES_OPTION_CHECK ) &&                                        \
    !defined( LONGRANGE_FORCE_MODULES_INSTANTIATION )

#ifndef FORCE_EWALD_CABANA_NEIGH_H
#define FORCE_EWALD_CABANA_NEIGH_H

#include <assert.h>

#include <Cabana_Core.hpp>
#include <comm_mpi.h>
#include <force.h>
#include <system.h>
#include <types.h>

template <class t_neighbor>
class ForceEwald : public Force
{
  private:
    int N_local, ntypes;
    typename AoSoA::member_slice_type<Positions> x;
    typename AoSoA::member_slice_type<Forces> f;
    typename AoSoA::member_slice_type<Forces>::atomic_access_slice f_a;
    typename AoSoA::member_slice_type<IDs> id;
    typename AoSoA::member_slice_type<Types> type;
    typename AoSoA::member_slice_type<Charges> q;
    typename AoSoA::member_slice_type<Potentials> p;

    Kokkos::View<T_F_FLOAT *, DeviceType> U_trigonometric;

    double _alpha;
    double _r_max;
    double _k_max;

    double lx, ly, lz;

    // dielectric constant
    double _eps_r = 1.0; // Assume 1 for now (vacuum)

    MPI_Comm cart_comm;

  public:
    bool half_neigh;

    t_neighbor neigh_list;

    ForceEwald( System *system, bool half_neigh );

    void init_coeff( System *system, T_X_FLOAT cut, char **args );
    void tune( System *system, T_F_FLOAT accuracy );

    void create_neigh_list( System *system );

    void compute( System *system );
    T_F_FLOAT compute_energy( System *system );

    const char *name();
};

#endif
#endif
