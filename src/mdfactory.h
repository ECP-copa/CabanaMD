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

#include <cabanamd.h>
#include <inputCL.h>
#include <neighbor.h>
#include <system.h>

#include <CabanaMD_config.hpp>

class MDfactory
{
  public:
    static CabanaMD *create( InputCL commandline )
    {
        bool half_neigh =
            commandline.force_iteration_type == FORCE_ITER_NEIGH_HALF;
        int layout = commandline.layout_type;
        int neigh = commandline.neighbor_type;

        if ( neigh == NEIGH_VERLET_2D )
        {
            if ( half_neigh )
            {
                if ( layout == AOSOA_1 )
                    return new CbnMD<
                        System<AoSoA1>,
                        NeighborVerlet<System<AoSoA1>, Cabana::HalfNeighborTag,
                                       Cabana::VerletLayout2D>>;
                else if ( layout == AOSOA_2 )
                    return new CbnMD<
                        System<AoSoA2>,
                        NeighborVerlet<System<AoSoA2>, Cabana::HalfNeighborTag,
                                       Cabana::VerletLayout2D>>;
                else if ( layout == AOSOA_6 )
                    return new CbnMD<
                        System<AoSoA6>,
                        NeighborVerlet<System<AoSoA6>, Cabana::HalfNeighborTag,
                                       Cabana::VerletLayout2D>>;
            }
            else
            {
                if ( layout == AOSOA_1 )
                    return new CbnMD<
                        System<AoSoA1>,
                        NeighborVerlet<System<AoSoA1>, Cabana::FullNeighborTag,
                                       Cabana::VerletLayout2D>>;
                else if ( layout == AOSOA_2 )
                    return new CbnMD<
                        System<AoSoA2>,
                        NeighborVerlet<System<AoSoA2>, Cabana::FullNeighborTag,
                                       Cabana::VerletLayout2D>>;
                else if ( layout == AOSOA_6 )
                    return new CbnMD<
                        System<AoSoA6>,
                        NeighborVerlet<System<AoSoA6>, Cabana::FullNeighborTag,
                                       Cabana::VerletLayout2D>>;
            }
        }
        if ( neigh == NEIGH_VERLET_CSR )
        {
            if ( half_neigh )
            {
                if ( layout == AOSOA_1 )
                    return new CbnMD<
                        System<AoSoA1>,
                        NeighborVerlet<System<AoSoA1>, Cabana::HalfNeighborTag,
                                       Cabana::VerletLayoutCSR>>;
                else if ( layout == AOSOA_2 )
                    return new CbnMD<
                        System<AoSoA2>,
                        NeighborVerlet<System<AoSoA2>, Cabana::HalfNeighborTag,
                                       Cabana::VerletLayoutCSR>>;
                else if ( layout == AOSOA_6 )
                    return new CbnMD<
                        System<AoSoA6>,
                        NeighborVerlet<System<AoSoA6>, Cabana::HalfNeighborTag,
                                       Cabana::VerletLayoutCSR>>;
            }
            else
            {
                if ( layout == AOSOA_1 )
                    return new CbnMD<
                        System<AoSoA1>,
                        NeighborVerlet<System<AoSoA1>, Cabana::FullNeighborTag,
                                       Cabana::VerletLayoutCSR>>;
                else if ( layout == AOSOA_2 )
                    return new CbnMD<
                        System<AoSoA2>,
                        NeighborVerlet<System<AoSoA2>, Cabana::FullNeighborTag,
                                       Cabana::VerletLayoutCSR>>;
                else if ( layout == AOSOA_6 )
                    return new CbnMD<
                        System<AoSoA6>,
                        NeighborVerlet<System<AoSoA6>, Cabana::FullNeighborTag,
                                       Cabana::VerletLayoutCSR>>;
            }
        }
#ifdef CabanaMD_ENABLE_ARBORX
        if ( neigh == NEIGH_TREE )
        {
            if ( half_neigh )
            {
                if ( layout == AOSOA_1 )
                    return new CbnMD<
                        System<AoSoA1>,
                        NeighborTree<System<AoSoA1>, Cabana::HalfNeighborTag,
                                     Cabana::VerletLayoutCSR>>;
                else if ( layout == AOSOA_2 )
                    return new CbnMD<
                        System<AoSoA2>,
                        NeighborTree<System<AoSoA2>, Cabana::HalfNeighborTag,
                                     Cabana::VerletLayoutCSR>>;
                else if ( layout == AOSOA_6 )
                    return new CbnMD<
                        System<AoSoA6>,
                        NeighborTree<System<AoSoA6>, Cabana::HalfNeighborTag,
                                     Cabana::VerletLayoutCSR>>;
            }
            else
            {
                if ( layout == AOSOA_1 )
                    return new CbnMD<
                        System<AoSoA1>,
                        NeighborTree<System<AoSoA1>, Cabana::FullNeighborTag,
                                     Cabana::VerletLayoutCSR>>;
                else if ( layout == AOSOA_2 )
                    return new CbnMD<
                        System<AoSoA2>,
                        NeighborTree<System<AoSoA2>, Cabana::FullNeighborTag,
                                     Cabana::VerletLayoutCSR>>;
                else if ( layout == AOSOA_6 )
                    return new CbnMD<
                        System<AoSoA6>,
                        NeighborTree<System<AoSoA6>, Cabana::FullNeighborTag,
                                     Cabana::VerletLayoutCSR>>;
            }
        }
#endif
        return nullptr;
    }
};
