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
#include <Cabana_Grid.hpp>
#include <Kokkos_Core.hpp>

#include <types.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

template <class DeviceType>
class SystemCommon
{
  public:
    using device_type = DeviceType;
    using memory_space = typename device_type::memory_space;
    using execution_space = typename device_type::execution_space;

    T_INT N;       // Number of Global Particles
    T_INT N_max;   // Number of Particles I could have in available storage
    T_INT N_local; // Number of owned Particles
    T_INT N_ghost; // Number of non-owned Particles

    int ntypes;
    std::string atom_style;

    // Per Type Property
    typedef Kokkos::View<T_V_FLOAT *, memory_space> t_mass;
    typedef Kokkos::View<const T_V_FLOAT *, memory_space> t_mass_const;
    typedef typename t_mass::HostMirror h_t_mass;
    t_mass mass;

    // Simulation total domain
    T_X_FLOAT global_mesh_x, global_mesh_y, global_mesh_z;

    // Simulation sub domain (single MPI rank)
    T_X_FLOAT local_mesh_x, local_mesh_y, local_mesh_z;
    T_X_FLOAT local_mesh_lo_x, local_mesh_lo_y, local_mesh_lo_z;
    T_X_FLOAT local_mesh_hi_x, local_mesh_hi_y, local_mesh_hi_z;
    T_X_FLOAT ghost_mesh_lo_x, ghost_mesh_lo_y, ghost_mesh_lo_z;
    T_X_FLOAT ghost_mesh_hi_x, ghost_mesh_hi_y, ghost_mesh_hi_z;
    T_X_FLOAT halo_width;
    std::shared_ptr<Cabana::Grid::DimBlockPartitioner<3>> partitioner;
    std::shared_ptr<
        Cabana::Grid::GlobalMesh<Cabana::Grid::UniformMesh<T_X_FLOAT>>>
        global_mesh;
    std::shared_ptr<
        Cabana::Grid::LocalGrid<Cabana::Grid::UniformMesh<T_X_FLOAT>>>
        local_grid;
    std::shared_ptr<
        Cabana::Grid::GlobalGrid<Cabana::Grid::UniformMesh<T_X_FLOAT>>>
        global_grid;

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

        mass = t_mass( "System::mass", ntypes );

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
        double ghost_cutoff =
            std::max( std::max( high_corner[0] - low_corner[0],
                                high_corner[2] - low_corner[1] ),
                      high_corner[2] - low_corner[2] );
        create_domain( low_corner, high_corner, ghost_cutoff );
    }
    void create_domain( std::array<double, 3> low_corner,
                        std::array<double, 3> high_corner, double ghost_cutoff )
    {
        halo_width = ghost_cutoff;
        // Create the MPI partitions.
        partitioner = std::make_shared<Cabana::Grid::DimBlockPartitioner<3>>();
        ranks_per_dim = partitioner->ranksPerDimension( MPI_COMM_WORLD, {} );
        int cells_per_dim_per_rank = 1;

        // The load balancing will be able to change the local domains with a
        // resolution of 1/cells_per_dim_per_rank
        cells_per_dim_per_rank = 100;
        std::array<int, 3> cells_per_rank = {
            cells_per_dim_per_rank * ranks_per_dim[0],
            cells_per_dim_per_rank * ranks_per_dim[1],
            cells_per_dim_per_rank * ranks_per_dim[2] };

        // Create global mesh of MPI partitions.
        global_mesh = Cabana::Grid::createUniformGlobalMesh(
            low_corner, high_corner, cells_per_rank );

        global_mesh_x = global_mesh->extent( 0 );
        global_mesh_y = global_mesh->extent( 1 );
        global_mesh_z = global_mesh->extent( 2 );

        // Create the global grid.
        std::array<bool, 3> is_periodic = { true, true, true };
        global_grid = Cabana::Grid::createGlobalGrid(
            MPI_COMM_WORLD, global_mesh, is_periodic, *partitioner );

        for ( int d = 0; d < 3; d++ )
        {
            rank_dim_pos[d] = global_grid->dimBlockId( d );
        }

        // Create a local mesh
        double minimum_cell_size = std::min(
            std::min( global_mesh->cellSize( 0 ), global_mesh->cellSize( 1 ) ),
            global_mesh->cellSize( 2 ) );
        halo_width = std::ceil( ghost_cutoff / minimum_cell_size );
        // Update local_mesh_* and ghost_mesh_* info as well as create
        // local_grid.
        update_global_grid( global_grid );
    }

    // Update domain info according to new global grid. We assume that the
    // number of ranks (per dim) does not change. We also assume that the
    // position of this rank in the cartesian grid of ranks does not change.
    void update_global_grid(
        const std::shared_ptr<
            Cabana::Grid::GlobalGrid<Cabana::Grid::UniformMesh<T_X_FLOAT>>>
            &new_global_grid )
    {
        global_grid = new_global_grid;
        local_grid = Cabana::Grid::createLocalGrid( global_grid, halo_width );
        update_mesh_info();
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
    virtual void permute( Cabana::LinkedCellList<memory_space> cell_list ) = 0;
    virtual void migrate(
        std::shared_ptr<Cabana::Distributor<memory_space>> distributor ) = 0;
    virtual void gather( std::shared_ptr<Cabana::Halo<memory_space>> halo ) = 0;
    virtual const char *name() { return "SystemNone"; }

  private:
    // Update local_mesh_* and ghost_mesh* info from global grid
    void update_mesh_info()
    {
        auto local_mesh =
            Cabana::Grid::createLocalMesh<memory_space>( *local_grid );

        local_mesh_lo_x = local_mesh.lowCorner( Cabana::Grid::Own(), 0 );
        local_mesh_lo_y = local_mesh.lowCorner( Cabana::Grid::Own(), 1 );
        local_mesh_lo_z = local_mesh.lowCorner( Cabana::Grid::Own(), 2 );
        local_mesh_hi_x = local_mesh.highCorner( Cabana::Grid::Own(), 0 );
        local_mesh_hi_y = local_mesh.highCorner( Cabana::Grid::Own(), 1 );
        local_mesh_hi_z = local_mesh.highCorner( Cabana::Grid::Own(), 2 );
        ghost_mesh_lo_x = local_mesh.lowCorner( Cabana::Grid::Ghost(), 0 );
        ghost_mesh_lo_y = local_mesh.lowCorner( Cabana::Grid::Ghost(), 1 );
        ghost_mesh_lo_z = local_mesh.lowCorner( Cabana::Grid::Ghost(), 2 );
        ghost_mesh_hi_x = local_mesh.highCorner( Cabana::Grid::Ghost(), 0 );
        ghost_mesh_hi_y = local_mesh.highCorner( Cabana::Grid::Ghost(), 1 );
        ghost_mesh_hi_z = local_mesh.highCorner( Cabana::Grid::Ghost(), 2 );
        local_mesh_x = local_mesh.extent( Cabana::Grid::Own(), 0 );
        local_mesh_y = local_mesh.extent( Cabana::Grid::Own(), 1 );
        local_mesh_z = local_mesh.extent( Cabana::Grid::Own(), 2 );
    }
};

template <class DeviceType, int layout>
class System : public SystemCommon<DeviceType>
{
  public:
    using SystemCommon<DeviceType>::SystemCommon;
};

#include <modules_system.h>
#endif
