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

#ifndef FORCE_SPME_CABANA_NEIGH_H
#define FORCE_SPME_CABANA_NEIGH_H
#include <Cabana_Core.hpp>
#include <Cajita.hpp>
#include <comm_mpi.h>
#include <force.h>
#include <system.h>
#include <types.h>

template <class t_System, class t_Neighbor>
class ForceSPME : public Force<t_System, t_Neighbor>
{
  private:
    int N_local, ntypes;
    typename t_System::t_x x;
    typename t_System::t_f f;
    typename t_System::t_f::atomic_access_slice f_a;
    typename t_System::t_type type;

    typedef typename t_Neighbor::t_neigh_list t_neigh_list;

    double _alpha;
    double _r_max;
    double _k_max;

    // dielectric constant
    double _eps_r = 1.0; // Assume 1 for now (vacuum)

  public:
    std::shared_ptr<Cajita::LocalGrid<Cajita::UniformMesh<double>>> local_grid;
    std::shared_ptr<Cajita::Array<double, Cajita::Node,
                                  Cajita::UniformMesh<double>, DeviceType>>
        Q;
    std::shared_ptr<Cajita::Halo<double, DeviceType>> Q_halo;
    std::shared_ptr<Cajita::Array<Kokkos::complex<double>, Cajita::Node,
                                  Cajita::UniformMesh<double>, DeviceType>>
        BC_array;

    ForceSPME( t_System *system );

    void init_coeff( t_System *system, char **args );
    void tune( t_System *system, T_F_FLOAT accuracy );
    void create_mesh( t_System *system );

    double oneDspline( double x );
    double oneDsplinederiv( double x );
    double oneDeuler( int k, int meshwidth );

    void compute( t_System *system, t_Neighbor *neighbor );
    T_F_FLOAT compute_energy( t_System *system, t_Neighbor *neighbor );

    const char *name();
};

#include <force_spme_cabana_neigh_impl.h>
#endif
