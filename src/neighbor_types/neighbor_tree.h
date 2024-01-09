/****************************************************************************
 * Copyright (c) 2018-2021 by the Cabana authors                            *
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

#include <Cabana_Core.hpp>

#include <neighbor.h>

template <class t_System, class t_iteration, class t_layout>
class NeighborTree : public Neighbor<t_System>
{
};

template <class t_System, class t_iteration>
class NeighborTree<t_System, t_iteration, Cabana::VerletLayoutCSR>
    : public Neighbor<t_System>
{
    using device_type = typename t_System::device_type;
    using memory_space = typename t_System::memory_space;

  public:
    T_X_FLOAT neigh_cut;
    bool half_neigh;
    T_INT max_neigh_guess;

    using t_neigh_list =
        Cabana::Experimental::CrsGraph<memory_space, t_iteration>;

    NeighborTree( T_X_FLOAT neigh_cut_, bool half_neigh_,
                  T_INT max_neigh_guess_ )
        : Neighbor<t_System>( neigh_cut_, half_neigh_ )
        , neigh_cut( neigh_cut_ )
        , half_neigh( half_neigh_ )
        , max_neigh_guess( max_neigh_guess_ )
    {
    }

    void create( t_System *system ) override
    {
        T_INT N_local = system->N_local;

        system->slice_x();
        auto x = system->x;

        t_iteration tag;
        list = Cabana::Experimental::makeNeighborList(
            tag, x, 0, N_local, neigh_cut, max_neigh_guess );
    }

    t_neigh_list &get() { return list; }

    const char *name() override
    {
        return half_neigh ? "Neighbor:CabanaTreeHalfCSR"
                          : "Neighbor:CabanaTreeFullCSR";
    }

  private:
    t_neigh_list list;
};

template <class t_System, class t_iteration>
class NeighborTree<t_System, t_iteration, Cabana::VerletLayout2D>
    : public Neighbor<t_System>
{
    using device_type = typename t_System::device_type;
    using memory_space = typename t_System::memory_space;

  public:
    T_X_FLOAT neigh_cut;
    bool half_neigh;
    T_INT max_neigh_guess;

    using t_neigh_list = Cabana::Experimental::Dense<memory_space, t_iteration>;

    NeighborTree( T_X_FLOAT neigh_cut_, bool half_neigh_,
                  T_INT max_neigh_guess_ )
        : Neighbor<t_System>( neigh_cut_, half_neigh_ )
        , neigh_cut( neigh_cut_ )
        , half_neigh( half_neigh_ )
        , max_neigh_guess( max_neigh_guess_ )
    {
    }

    void create( t_System *system ) override
    {
        T_INT N_local = system->N_local;

        system->slice_x();
        auto x = system->x;

        t_iteration tag;
        list = Cabana::Experimental::make2DNeighborList(
            tag, x, 0, N_local, neigh_cut, max_neigh_guess );
    }

    t_neigh_list &get() { return list; }

    const char *name() override
    {
        return half_neigh ? "Neighbor:CabanaTreeHalf2D"
                          : "Neighbor:CabanaTreeFull2D";
    }

  private:
    t_neigh_list list;
};

#endif
