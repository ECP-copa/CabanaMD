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

//************************************************************************
//  ExaMiniMD v. 1.0
//  Copyright (2018) National Technology & Engineering Solutions of Sandia,
//  LLC (NTESS).
//
//  Under the terms of Contract DE-NA-0003525 with NTESS, the U.S. Government
//  retains certain rights in this software.
//
//  ExaMiniMD is licensed under 3-clause BSD terms of use: Redistribution and
//  use in source and binary forms, with or without modification, are
//  permitted provided that the following conditions are met:
//
//    1. Redistributions of source code must retain the above copyright notice,
//       this list of conditions and the following disclaimer.
//
//    2. Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//    3. Neither the name of the Corporation nor the names of the contributors
//       may be used to endorse or promote products derived from this software
//       without specific prior written permission.
//
//  THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY EXPRESS OR IMPLIED
//  WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
//  MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
//  IN NO EVENT SHALL NTESS OR THE CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
//  INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
//  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
//  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
//  HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
//  STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
//  IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
//  POSSIBILITY OF SUCH DAMAGE.
//
//************************************************************************

#ifndef SYSTEM_H
#define SYSTEM_H

#include <Cabana_Core.hpp>
#include <Cajita.hpp>
#include <Kokkos_Core.hpp>

#include <types.h>

#include <memory>
#include <string>

template <class t_device>
class SystemCommon
{
  public:
    using device_type = t_device;
    using memory_space = typename device_type::memory_space;
    using execution_space = typename device_type::execution_space;

    T_INT N;       // Number of Global Particles
    T_INT N_max;   // Number of Particles I could have in available storage
    T_INT N_local; // Number of owned Particles
    T_INT N_ghost; // Number of non-owned Particles

    int ntypes;
    std::string atom_style;

    // Per Type Property
    // typedef typename t_device::array_layout layout;
    typedef Kokkos::View<T_V_FLOAT *, t_device> t_mass;
    typedef Kokkos::View<const T_V_FLOAT *, t_device> t_mass_const;
    typedef typename t_mass::HostMirror h_t_mass;
    t_mass mass;
    t_mass charge;

    // Simulation total domain
    T_X_FLOAT global_mesh_x, global_mesh_y, global_mesh_z;
    T_X_FLOAT lattice_constant;

    // Simulation sub domain (single MPI rank)
    T_X_FLOAT local_mesh_x, local_mesh_y, local_mesh_z;
    T_X_FLOAT local_mesh_lo_x, local_mesh_lo_y, local_mesh_lo_z;
    T_X_FLOAT local_mesh_hi_x, local_mesh_hi_y, local_mesh_hi_z;
    T_X_FLOAT ghost_mesh_lo_x, ghost_mesh_lo_y, ghost_mesh_lo_z;
    T_X_FLOAT ghost_mesh_hi_x, ghost_mesh_hi_y, ghost_mesh_hi_z;
    std::shared_ptr<Cajita::LocalGrid<Cajita::UniformMesh<T_X_FLOAT>>>
        local_grid;

    // Only needed for current comm
    std::array<int, 3> ranks_per_dim;
    std::array<int, 3> rank_dim_pos;

    // Units
    T_FLOAT boltz, mvv2e, dt;

    SystemCommon()
    {
        N = 0;
        N_max = 0;
        N_local = 0;
        N_ghost = 0;
        ntypes = 1;
        atom_style = "atomic";

        mass = t_mass();

        global_mesh_x = global_mesh_y = global_mesh_z = 0.0;
        local_mesh_lo_x = local_mesh_lo_y = local_mesh_lo_z = 0.0;
        local_mesh_hi_x = local_mesh_hi_y = local_mesh_hi_z = 0.0;
        ghost_mesh_lo_x = ghost_mesh_lo_y = ghost_mesh_lo_z = 0.0;
        ghost_mesh_hi_x = ghost_mesh_hi_y = ghost_mesh_hi_z = 0.0;
        local_mesh_x = local_mesh_y = local_mesh_z = 0.0;

        mvv2e = boltz = dt = 0.0;

        mass = t_mass( "System::mass", ntypes );
    }

    ~SystemCommon() {}

    void create_domain( std::array<double, 3> low_corner,
                        std::array<double, 3> high_corner )
    {
        // Create the MPI partitions.
        Cajita::UniformDimPartitioner partitioner;
        ranks_per_dim = partitioner.ranksPerDimension( MPI_COMM_WORLD, {} );

        // Create global mesh of MPI partitions.
        auto global_mesh = Cajita::createUniformGlobalMesh(
            low_corner, high_corner, ranks_per_dim );

        global_mesh_x = global_mesh->extent( 0 );
        global_mesh_y = global_mesh->extent( 1 );
        global_mesh_z = global_mesh->extent( 2 );

        // Create the global grid.
        std::array<bool, 3> is_periodic = {true, true, true};
        auto global_grid = Cajita::createGlobalGrid(
            MPI_COMM_WORLD, global_mesh, is_periodic, partitioner );

        for ( int d = 0; d < 3; d++ )
        {
            rank_dim_pos[d] = global_grid->dimBlockId( d );
        }

        // Create a local mesh
        int halo_width = 1;
        local_grid = Cajita::createLocalGrid( global_grid, halo_width );
        auto local_mesh = Cajita::createLocalMesh<t_device>( *local_grid );

        local_mesh_lo_x = local_mesh.lowCorner( Cajita::Own(), 0 );
        local_mesh_lo_y = local_mesh.lowCorner( Cajita::Own(), 1 );
        local_mesh_lo_z = local_mesh.lowCorner( Cajita::Own(), 2 );
        local_mesh_hi_x = local_mesh.highCorner( Cajita::Own(), 0 );
        local_mesh_hi_y = local_mesh.highCorner( Cajita::Own(), 1 );
        local_mesh_hi_z = local_mesh.highCorner( Cajita::Own(), 2 );
        ghost_mesh_lo_x = local_mesh.lowCorner( Cajita::Ghost(), 0 );
        ghost_mesh_lo_y = local_mesh.lowCorner( Cajita::Ghost(), 1 );
        ghost_mesh_lo_z = local_mesh.lowCorner( Cajita::Ghost(), 2 );
        ghost_mesh_hi_x = local_mesh.highCorner( Cajita::Ghost(), 0 );
        ghost_mesh_hi_y = local_mesh.highCorner( Cajita::Ghost(), 1 );
        ghost_mesh_hi_z = local_mesh.highCorner( Cajita::Ghost(), 2 );
        local_mesh_x = local_mesh.extent( Cajita::Own(), 0 );
        local_mesh_y = local_mesh.extent( Cajita::Own(), 1 );
        local_mesh_z = local_mesh.extent( Cajita::Own(), 2 );
    }

    void slice_all()
    {
        slice_x();
        slice_v();
        slice_f();
        slice_type();
        slice_id();
        slice_q();
    }
    void slice_integrate()
    {
        slice_x();
        slice_v();
        slice_f();
        slice_type();
    }
    void slice_force()
    {
        slice_x();
        slice_f();
        slice_type();
    }
    void slice_properties()
    {
        slice_v();
        slice_type();
    }
    virtual void slice_x() = 0;
    virtual void slice_v() = 0;
    virtual void slice_f() = 0;
    virtual void slice_type() = 0;
    virtual void slice_id() = 0;
    virtual void slice_q() = 0;

    virtual void init() = 0;
    virtual void resize( T_INT N_new ) = 0;
    virtual void permute( Cabana::LinkedCellList<t_device> cell_list ) = 0;
    virtual void
    migrate( std::shared_ptr<Cabana::Distributor<t_device>> distributor ) = 0;
    virtual void gather( std::shared_ptr<Cabana::Halo<t_device>> halo ) = 0;
    virtual const char *name() { return "SystemNone"; }
};

template <class t_device, class t_layout>
class System : public SystemCommon<t_device>
{
  public:
    using SystemCommon<t_device>::SystemCommon;
};

#include <modules_system.h>
#endif
