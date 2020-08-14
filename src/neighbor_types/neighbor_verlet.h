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

#ifndef NEIGHBOR_VERLET_H
#define NEIGHBOR_VERLET_H

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>

#include <neighbor.h>

template <class t_System, class t_iteration, class t_layout>
class NeighborVerlet : public Neighbor<t_System>
{
    using memory_space = typename t_System::memory_space;

  public:
    T_X_FLOAT neigh_cut;
    bool half_neigh;
    T_INT max_neigh = 50;

#ifdef KOKKOS_ENABLE_HIP
    using t_build = Cabana::TeamOpTag;
#else
    using t_build = Cabana::TeamVectorOpTag;
#endif
    using t_neigh_list =
        Cabana::VerletList<memory_space, t_iteration, t_layout, t_build>;

    NeighborVerlet( T_X_FLOAT neigh_cut_, bool half_neigh_ )
        : Neighbor<t_System>( neigh_cut_, half_neigh_ )
        , neigh_cut( neigh_cut_ )
        , half_neigh( half_neigh_ )
    {
    }

    void create( t_System *system ) override
    {
        T_INT N_local = system->N_local;

        double grid_min[3] = {system->sub_domain_lo_x - system->sub_domain_x,
                              system->sub_domain_lo_y - system->sub_domain_y,
                              system->sub_domain_lo_z - system->sub_domain_z};
        double grid_max[3] = {system->sub_domain_hi_x + system->sub_domain_x,
                              system->sub_domain_hi_y + system->sub_domain_y,
                              system->sub_domain_hi_z + system->sub_domain_z};

        system->slice_x();
        auto x = system->x;

        list = t_neigh_list( x, 0, N_local, neigh_cut, 1.0, grid_min, grid_max,
                             max_neigh );
        max_neigh =
            Cabana::NeighborList<t_neigh_list>::maxNeighbor( list ) * 1.1;
    }

    t_neigh_list &get() { return list; }

    const char *name() override
    {
        return half_neigh ? "Neighbor:CabanaVerletHalf"
                          : "Neighbor:CabanaVerletFull";
    }

  private:
    t_neigh_list list;
};

#endif
