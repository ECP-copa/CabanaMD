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

    T_F_FLOAT _alpha;
    T_X_FLOAT _r_max;
    T_X_FLOAT _k_max;
    T_F_FLOAT accuracy;
    T_INT _k_int;
    T_INT n_kvec;

    // dielectric constant
    T_X_FLOAT _eps_r = 1.0; // Assume 1 for now (vacuum)

    using exe_space = typename t_System::execution_space;
    using device_type = typename t_System::device_type;

  public:
    std::shared_ptr<Cajita::LocalGrid<Cajita::UniformMesh<double>>> local_grid;
    std::shared_ptr<Cajita::Array<double, Cajita::Node,
                                  Cajita::UniformMesh<double>, device_type>>
        Q;
    std::shared_ptr<Cajita::Halo<double, device_type>> Q_halo;
    std::shared_ptr<Cajita::Array<double, Cajita::Node,
                                  Cajita::UniformMesh<double>, device_type>>
        BC_array;
    std::shared_ptr<Cajita::Array<Kokkos::complex<double>, Cajita::Node,
                                  Cajita::UniformMesh<double>, device_type>>
        Qcomplex;

    ForceSPME( t_System *system );

    void init_coeff( std::vector<std::vector<std::string>> args ) override;
    void init_longrange( t_System *system, T_X_FLOAT r_max ) override;
    void tune( t_System *system );
    void create_mesh( t_System *system );

    double oneDspline( double x );
    double oneDsplinederiv( double x );
    double oneDeuler( int k, int meshwidth );

    void compute( t_System *system, t_Neighbor *neighbor ) override;
    T_F_FLOAT compute_energy( t_System *system, t_Neighbor *neighbor ) override;

    const char *name() override;
};

#include <force_spme_cabana_neigh_impl.h>
#endif
