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

#ifndef FORCE_EWALD_CABANA_NEIGH_H
#define FORCE_EWALD_CABANA_NEIGH_H

#include <assert.h>

#include <Cabana_Core.hpp>
#include <comm_mpi.h>
#include <force.h>
#include <system.h>
#include <types.h>

template <class t_System, class t_Neighbor>
    class ForceEwald : public Force<t_System, t_Neighbor>
{
  private:
    int N_local, ntypes;
    typename t_System::t_x x;
    typename t_System::t_f f;
    typename t_System::t_f::atomic_access_slice f_a;
    typename t_System::t_type type;

    typedef typename t_Neighbor::t_neigh_list t_neigh_list;

    Kokkos::View<T_F_FLOAT *, DeviceType> U_trigonometric;

    double _alpha;
    double _r_max;
    double _k_max;

    double lx, ly, lz;

    // dielectric constant
    double _eps_r = 1.0; // Assume 1 for now (vacuum)

    MPI_Comm cart_comm;

  public:
    ForceEwald( t_System *system );

    void init_coeff( t_System *system, char **args );
    void tune( t_System *system, T_F_FLOAT accuracy );

    void compute( t_System *system, t_Neighbor *neighbor );
    T_F_FLOAT compute_energy( t_System *system, t_Neighbor *neighbor );

    const char *name();
};

#include <force_ewald_cabana_neigh_impl.h>
#endif
