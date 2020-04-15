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

#ifndef NEIGHBOR_TREE_H
#define NEIGHBOR_TREE_H

#include <neighbor.h>
#include <system.h>
#include <types.h>

#include <Cabana_Core.hpp>

template <class t_System, class t_iteration, class t_layout>
class NeighborTree : public Neighbor<t_System, t_iteration, t_layout>
{
  public:
    T_X_FLOAT neigh_cut;
    bool half_neigh;

    using t_neigh_list =
        Cabana::Experimental::CrsGraph<MemorySpace, t_iteration>;

    NeighborTree( T_X_FLOAT neigh_cut_, bool half_neigh_ )
        : Neighbor<t_System, t_iteration, t_layout>( neigh_cut_, half_neigh_ )
        , neigh_cut( neigh_cut_ )
        , half_neigh( half_neigh_ )
    {
    }

    void create( t_System *system ) override
    {
        T_INT N_local = system->N_local;

        system->slice_x();
        auto x = system->x;

        t_iteration tag;
        list = Cabana::Experimental::makeNeighborList<MemorySpace>(
            tag, x, 0, N_local, neigh_cut );
    }

    t_neigh_list &get() { return list; }

    const char *name() override
    {
        return half_neigh ? "Neighbor:CabanaTreeHalf"
                          : "Neighbor:CabanaTreeFull";
    }

  private:
    t_neigh_list list;
};

#endif
