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

#ifndef FORCE_NNP_CABANA_NEIGH_H
#define FORCE_NNP_CABANA_NEIGH_H

#include <nnp_mode.h>
#include <system_nnp.h>

#include <force.h>
#include <neighbor.h>
#include <system.h>
#include <types.h>

#include <Cabana_Core.hpp>

template <class t_System, class t_System_NNP, class t_Neighbor,
          class t_neigh_parallel, class t_angle_parallel>
class ForceNNP : public Force<t_System, t_Neighbor>
{
  private:
    int N_local, ntypes;
    int step;

    typedef typename t_System::t_x t_x;
    // Must be atomic
    typedef typename t_System::t_f::atomic_access_slice t_f_a;
    typedef typename t_System::t_type t_type;

    typedef typename t_System_NNP::t_G t_G;
    typedef typename t_System_NNP::t_G::atomic_access_slice t_G_a;
    typedef typename t_System_NNP::t_dEdG t_dEdG;
    typedef typename t_System_NNP::t_E t_E;

    typedef typename t_Neighbor::t_neigh_list t_neigh_list;

    // NNP-specific System class for AoSoAs
    // Storage of G, dEdG and energy (per atom properties)
    t_System_NNP *system_nnp;

    using device_type = typename t_System::device_type;
    using exe_space = typename t_System::execution_space;

  public:
    nnpCbn::Mode<device_type> *mode;

    // numSymmetryFunctionsPerElement (per type property)
    using t_mass = typename t_System::t_mass;
    using h_t_mass = typename t_System::h_t_mass;
    t_mass d_numSFperElem;
    h_t_mass h_numSFperElem, atomicEnergyOffset;

    ForceNNP( t_System *system );

    void init_coeff( char **args ) override;
    void compute( t_System *system, t_Neighbor *neighbor ) override;
    T_F_FLOAT compute_energy( t_System *system, t_Neighbor *neighbor ) override;

    const char *name() override;
    const char *system_name() override;

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
