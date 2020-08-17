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

#include <CabanaMD_config.hpp>

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>

#include <cabanamd.h>
#include <inputCL.h>
#include <neighbor.h>
#include <system.h>

class MDfactory
{
  public:
    static CabanaMD *create( InputCL commandline )
    {
        int device = commandline.device_type;
        int layout = commandline.layout_type;
        int neigh = commandline.neighbor_type;
        bool half_neigh =
            commandline.force_iteration_type == FORCE_ITER_NEIGH_HALF;

        if ( device == SERIAL )
        {
#ifdef KOKKOS_ENABLE_SERIAL
            using t_device = Kokkos::Device<Kokkos::Serial, Kokkos::HostSpace>;
            if ( layout == AOSOA_1 )
            {
                using t_sys = System<t_device, AoSoA1>;
                if ( neigh == NEIGH_VERLET_2D )
                {
                    using t_lay = Cabana::VerletLayout2D;
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
                else if ( neigh == NEIGH_VERLET_CSR )
                {
                    using t_lay = Cabana::VerletLayoutCSR;
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
                else if ( neigh == NEIGH_TREE_2D )
                {
                    using t_lay = Cabana::VerletLayout2D;
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
                else if ( neigh == NEIGH_TREE_CSR )
                {
                    using t_lay = Cabana::VerletLayoutCSR;
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
#endif // ArborX
            }
            else if ( layout == AOSOA_2 )
            {
                using t_sys = System<t_device, AoSoA2>;
                if ( neigh == NEIGH_VERLET_2D )
                {
                    using t_lay = Cabana::VerletLayout2D;
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
                else if ( neigh == NEIGH_VERLET_CSR )
                {
                    using t_lay = Cabana::VerletLayoutCSR;
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
                else if ( neigh == NEIGH_TREE_2D )
                {
                    using t_lay = Cabana::VerletLayout2D;
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
                else if ( neigh == NEIGH_TREE_CSR )
                {
                    using t_lay = Cabana::VerletLayoutCSR;
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
#endif // ArborX
            }
            else if ( layout == AOSOA_6 )
            {
                using t_sys = System<t_device, AoSoA6>;
                if ( neigh == NEIGH_VERLET_2D )
                {
                    using t_lay = Cabana::VerletLayout2D;
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
                else if ( neigh == NEIGH_VERLET_CSR )
                {
                    using t_lay = Cabana::VerletLayoutCSR;
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
                else if ( neigh == NEIGH_TREE_2D )
                {
                    using t_lay = Cabana::VerletLayout2D;
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
                else if ( neigh == NEIGH_TREE_CSR )
                {
                    using t_lay = Cabana::VerletLayoutCSR;
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
#endif // ArborX
            }
#else
            throw std::runtime_error(
                "CabanaMD not compiled with Kokkos::Serial" );
#endif // Serial
        }
        else if ( device == OPENMP )
        {
#ifdef KOKKOS_ENABLE_OPENMP
            using t_device = Kokkos::Device<Kokkos::OpenMP, Kokkos::HostSpace>;
            if ( layout == AOSOA_1 )
            {
                using t_sys = System<t_device, AoSoA1>;
                if ( neigh == NEIGH_VERLET_2D )
                {
                    using t_lay = Cabana::VerletLayout2D;
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
                else if ( neigh == NEIGH_VERLET_CSR )
                {
                    using t_lay = Cabana::VerletLayoutCSR;
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
                else if ( neigh == NEIGH_TREE_2D )
                {
                    using t_lay = Cabana::VerletLayout2D;
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
                else if ( neigh == NEIGH_TREE_CSR )
                {
                    using t_lay = Cabana::VerletLayoutCSR;
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
#endif // ArborX
            }
            else if ( layout == AOSOA_2 )
            {
                using t_sys = System<t_device, AoSoA2>;
                if ( neigh == NEIGH_VERLET_2D )
                {
                    using t_lay = Cabana::VerletLayout2D;
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
                else if ( neigh == NEIGH_VERLET_CSR )
                {
                    using t_lay = Cabana::VerletLayoutCSR;
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
                else if ( neigh == NEIGH_TREE_2D )
                {
                    using t_lay = Cabana::VerletLayout2D;
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
                else if ( neigh == NEIGH_TREE_CSR )
                {
                    using t_lay = Cabana::VerletLayoutCSR;
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
#endif // ArborX
            }
            else if ( layout == AOSOA_6 )
            {
                using t_sys = System<t_device, AoSoA6>;
                if ( neigh == NEIGH_VERLET_2D )
                {
                    using t_lay = Cabana::VerletLayout2D;
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
                else if ( neigh == NEIGH_VERLET_CSR )
                {
                    using t_lay = Cabana::VerletLayoutCSR;
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
                else if ( neigh == NEIGH_TREE_2D )
                {
                    using t_lay = Cabana::VerletLayout2D;
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
                else if ( neigh == NEIGH_TREE_CSR )
                {
                    using t_lay = Cabana::VerletLayoutCSR;
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
#endif // ArborX
            }
#else
            throw std::runtime_error(
                "CabanaMD not compiled with Kokkos::OpenMP" );
#endif // OpenMP
        }
        else if ( device == CUDA )
        {
#ifdef KOKKOS_ENABLE_CUDA
            using t_device = Kokkos::Device<Kokkos::Cuda, Kokkos::CudaSpace>;
            if ( layout == AOSOA_1 )
            {
                using t_sys = System<t_device, AoSoA1>;
                if ( neigh == NEIGH_VERLET_2D )
                {
                    using t_lay = Cabana::VerletLayout2D;
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
                else if ( neigh == NEIGH_VERLET_CSR )
                {
                    using t_lay = Cabana::VerletLayoutCSR;
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
                else if ( neigh == NEIGH_TREE_2D )
                {
                    using t_lay = Cabana::VerletLayout2D;
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
                else if ( neigh == NEIGH_TREE_CSR )
                {
                    using t_lay = Cabana::VerletLayoutCSR;
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
#endif // ArborX
            }
            else if ( layout == AOSOA_2 )
            {
                using t_sys = System<t_device, AoSoA2>;
                if ( neigh == NEIGH_VERLET_2D )
                {
                    using t_lay = Cabana::VerletLayout2D;
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
                else if ( neigh == NEIGH_VERLET_CSR )
                {
                    using t_lay = Cabana::VerletLayoutCSR;
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
                else if ( neigh == NEIGH_TREE_2D )
                {
                    using t_lay = Cabana::VerletLayout2D;
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
                else if ( neigh == NEIGH_TREE_CSR )
                {
                    using t_lay = Cabana::VerletLayoutCSR;
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
#endif // ArborX
            }
            else if ( layout == AOSOA_6 )
            {
                using t_sys = System<t_device, AoSoA6>;
                if ( neigh == NEIGH_VERLET_2D )
                {
                    using t_lay = Cabana::VerletLayout2D;
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
                else if ( neigh == NEIGH_VERLET_CSR )
                {
                    using t_lay = Cabana::VerletLayoutCSR;
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
                else if ( neigh == NEIGH_TREE_2D )
                {
                    using t_lay = Cabana::VerletLayout2D;
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
                else if ( neigh == NEIGH_TREE_CSR )
                {
                    using t_lay = Cabana::VerletLayoutCSR;
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
#endif // ArborX
            }
#else
            throw std::runtime_error(
                "CabanaMD not compiled with Kokkos::Cuda" );
#endif // Cuda
        }
        else if ( device == HIP )
        {
#ifdef KOKKOS_ENABLE_HIP
            using t_device = Kokkos::Device<Kokkos::Experimental::HIP,
                                            Kokkos::Experimental::HIPSpace>;
            if ( layout == AOSOA_1 )
            {
                using t_sys = System<t_device, AoSoA1>;
                if ( neigh == NEIGH_VERLET_2D )
                {
                    using t_lay = Cabana::VerletLayout2D;
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
                else if ( neigh == NEIGH_VERLET_CSR )
                {
                    using t_lay = Cabana::VerletLayoutCSR;
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
                else if ( neigh == NEIGH_TREE_2D )
                {
                    using t_lay = Cabana::VerletLayout2D;
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
                else if ( neigh == NEIGH_TREE_CSR )
                {
                    using t_lay = Cabana::VerletLayoutCSR;
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
#endif // ArborX
            }
            else if ( layout == AOSOA_2 )
            {
                using t_sys = System<t_device, AoSoA2>;
                if ( neigh == NEIGH_VERLET_2D )
                {
                    using t_lay = Cabana::VerletLayout2D;
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
                else if ( neigh == NEIGH_VERLET_CSR )
                {
                    using t_lay = Cabana::VerletLayoutCSR;
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
                else if ( neigh == NEIGH_TREE_2D )
                {
                    using t_lay = Cabana::VerletLayout2D;
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
                else if ( neigh == NEIGH_TREE_CSR )
                {
                    using t_lay = Cabana::VerletLayoutCSR;
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
#endif // ArborX
            }
            else if ( layout == AOSOA_6 )
            {
                using t_sys = System<t_device, AoSoA6>;
                if ( neigh == NEIGH_VERLET_2D )
                {
                    using t_lay = Cabana::VerletLayout2D;
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
                else if ( neigh == NEIGH_VERLET_CSR )
                {
                    using t_lay = Cabana::VerletLayoutCSR;
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
                else if ( neigh == NEIGH_TREE_2D )
                {
                    using t_lay = Cabana::VerletLayout2D;
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
                else if ( neigh == NEIGH_TREE_CSR )
                {
                    using t_lay = Cabana::VerletLayoutCSR;
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
#endif // ArborX
            }
#else
            throw std::runtime_error(
                "CabanaMD not compiled with Kokkos::Experimental::HIP" );
#endif // HIP
        }

        return nullptr;
    }
};
