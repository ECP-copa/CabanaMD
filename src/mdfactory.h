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

#include <CabanaMD_config.hpp>

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>

#include <cabanamd.h>
#include <inputCL.h>
#include <neighbor.h>
#include <system.h>

class MDfactory
{
    template <typename t_sys, typename t_lay>
    static CabanaMD *createImplVerlet( int half_neigh )
    {
        if ( half_neigh )
        {
            using t_half = Cabana::HalfNeighborTag;
            using t_neigh = NeighborVerlet<t_sys, t_half, t_lay>;
            return new CbnMD<t_sys, t_neigh>;
        }
        else
        {
            using t_full = Cabana::FullNeighborTag;
            using t_neigh = NeighborVerlet<t_sys, t_full, t_lay>;
            return new CbnMD<t_sys, t_neigh>;
        }
    }
#ifdef Cabana_ENABLE_ARBORX
    template <typename t_sys, typename t_lay>
    static CabanaMD *createImplTree( int half_neigh )
    {
        if ( half_neigh )
        {
            using t_half = Cabana::HalfNeighborTag;
            using t_neigh = NeighborTree<t_sys, t_half, t_lay>;
            return new CbnMD<t_sys, t_neigh>;
        }
        else
        {
            using t_full = Cabana::FullNeighborTag;
            using t_neigh = NeighborTree<t_sys, t_full, t_lay>;
            return new CbnMD<t_sys, t_neigh>;
        }
    }
#endif
    template <typename t_sys>
    static CabanaMD *createImplSystem( int neigh, int half_neigh )
    {
        if ( neigh == NEIGH_VERLET_2D )
            return createImplVerlet<t_sys, Cabana::VerletLayout2D>(
                half_neigh );
        else if ( neigh == NEIGH_VERLET_CSR )
            return createImplVerlet<t_sys, Cabana::VerletLayoutCSR>(
                half_neigh );
#ifdef Cabana_ENABLE_ARBORX
        else if ( neigh == NEIGH_TREE_2D )
            return createImplTree<t_sys, Cabana::VerletLayout2D>( half_neigh );
        else if ( neigh == NEIGH_TREE_CSR )
            return createImplTree<t_sys, Cabana::VerletLayoutCSR>( half_neigh );
#endif // ArborX
        return nullptr;
    }

    template <typename t_device>
    static CabanaMD *createImpl( InputCL commandline )
    {
        int neigh = commandline.neighbor_type;
        bool half_neigh =
            commandline.force_iteration_type == FORCE_ITER_NEIGH_HALF;

        return createImplSystem<System<t_device, CabanaMD_LAYOUT>>(
            neigh, half_neigh );

        return nullptr;
    }

  public:
    static CabanaMD *create( InputCL commandline )
    {
        int device = commandline.device_type;

        if ( device == CUDA )
        {
            // Cuda is the first default, so just use that.
            device = DEFAULT;
#ifndef KOKKOS_ENABLE_CUDA
            throw std::runtime_error(
                "CabanaMD not compiled with Kokkos::Cuda" );
#endif
        }
        if ( device == DEFAULT )
        {
            using t_exe = Kokkos::DefaultExecutionSpace;
            using t_mem = typename t_exe::memory_space;
            using t_device = Kokkos::Device<t_exe, t_mem>;

            return createImpl<t_device>( commandline );
        }
        if ( device == SERIAL )
        {
#ifdef KOKKOS_ENABLE_SERIAL
            using t_device = Kokkos::Device<Kokkos::Serial, Kokkos::HostSpace>;
            return createImpl<t_device>( commandline );

#else
            throw std::runtime_error(
                "CabanaMD not compiled with Kokkos::Serial" );
#endif
        }
        else if ( device == PTHREAD )
        {
#ifdef KOKKOS_ENABLE_THREADS
            using t_device = Kokkos::Threads::device_type;
            return createImpl<t_device>( commandline );
#else
            throw std::runtime_error(
                "CabanaMD not compiled with Kokkos::Threads" );
#endif
        }
        else if ( device == OPENMP )
        {
#ifdef KOKKOS_ENABLE_OPENMP
            using t_device = Kokkos::Device<Kokkos::OpenMP, Kokkos::HostSpace>;
            return createImpl<t_device>( commandline );
#else
            throw std::runtime_error(
                "CabanaMD not compiled with Kokkos::OpenMP" );
#endif
        }
        else if ( device == HIP )
        {
#ifdef KOKKOS_ENABLE_HIP
            using t_device = Kokkos::Device<Kokkos::Experimental::HIP,
                                            Kokkos::Experimental::HIPSpace>;
            return createImpl<t_device>( commandline );
#else
            throw std::runtime_error(
                "CabanaMD not compiled with Kokkos::Experimental::HIP" );
#endif
        }

        return nullptr;
    }
};
