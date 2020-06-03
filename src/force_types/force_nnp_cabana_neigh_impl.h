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

#include <iostream>
#include <string.h>
#include <string>

template <class t_System, class t_System_NNP, class t_Neighbor,
          class t_neigh_parallel, class t_angle_parallel>
ForceNNP<t_System, t_System_NNP, t_Neighbor, t_neigh_parallel,
         t_angle_parallel>::ForceNNP( t_System *system )
    : Force<t_System, t_Neighbor>( system )
{
    ntypes = system->ntypes;
    N_local = 0;
    step = 0;

    system_nnp = new t_System_NNP;
}

template <class t_System, class t_System_NNP, class t_Neighbor,
          class t_neigh_parallel, class t_angle_parallel>
const char *ForceNNP<t_System, t_System_NNP, t_Neighbor, t_neigh_parallel,
                     t_angle_parallel>::system_name()
{
    return system_nnp->name();
}

template <class t_System, class t_System_NNP, class t_Neighbor,
          class t_neigh_parallel, class t_angle_parallel>
const char *ForceNNP<t_System, t_System_NNP, t_Neighbor, t_neigh_parallel,
                     t_angle_parallel>::name()
{
    return "Force:NNPCabana";
}

template <class t_System, class t_System_NNP, class t_Neighbor,
          class t_neigh_parallel, class t_angle_parallel>
void ForceNNP<t_System, t_System_NNP, t_Neighbor, t_neigh_parallel,
              t_angle_parallel>::init_coeff( char **args )
{
    mode = new ( nnpCbn::Mode<device_type> );

    mode->initialize();
    std::string settingsfile =
        std::string( args[3] ) + "/input.nn"; // arg[3] gives directory path
    mode->loadSettingsFile( settingsfile );
    mode->setupNormalization();
    mode->setupElementMap();
    mode->setupElements();
    mode->setupCutoff();
    mode->setupSymmetryFunctions();
    mode->setupSymmetryFunctionGroups();
    mode->setupNeuralNetwork();
    std::string scalingfile = std::string( args[3] ) + "/scaling.data";
    mode->setupSymmetryFunctionScaling( scalingfile );
    std::string weightsfile = std::string( args[3] ) + "/weights.%03zu.data";
    mode->setupSymmetryFunctionStatistics( false, false, true, false );
    mode->setupNeuralNetworkWeights( weightsfile );
}

template <class t_System, class t_System_NNP, class t_Neighbor,
          class t_neigh_parallel, class t_angle_parallel>
void ForceNNP<t_System, t_System_NNP, t_Neighbor, t_neigh_parallel,
              t_angle_parallel>::compute( t_System *s, t_Neighbor *neighbor )
{
    N_local = s->N_local;

    auto neigh_list = neighbor->get();

    system_nnp->resize( N_local );

    s->slice_force();
    auto x = s->x;
    // Atomic force slice
    t_f_a f_a = s->f;
    auto type = s->type;

    system_nnp->slice_G();
    system_nnp->slice_dEdG();
    system_nnp->slice_E();
    t_G G = system_nnp->G;
    t_G_a G_a = G;
    auto dEdG = system_nnp->dEdG;
    auto E = system_nnp->E;

    mode->calculateSymmetryFunctionGroups( x, type, G_a, neigh_list, N_local,
                                           t_neigh_parallel(),
                                           t_angle_parallel() );
    mode->calculateAtomicNeuralNetworks( type, G, dEdG, E, N_local );
    mode->calculateForces( x, f_a, type, dEdG, neigh_list, N_local,
                           t_neigh_parallel(), t_angle_parallel() );
}

template <class t_System, class t_System_NNP, class t_Neighbor,
          class t_neigh_parallel, class t_angle_parallel>
T_V_FLOAT ForceNNP<t_System, t_System_NNP, t_Neighbor, t_neigh_parallel,
                   t_angle_parallel>::compute_energy( t_System *s,
                                                      t_Neighbor * )
{
    system_nnp->slice_E();
    auto energy = system_nnp->E;

    T_V_FLOAT system_energy = 0.0;
    // Loop over all atoms and add atomic contributions to total energy.
    Kokkos::RangePolicy<exe_space> policy( 0, s->N_local );
    Kokkos::parallel_reduce(
        "ForceNNPCabanaNeigh::compute_energy", policy,
        KOKKOS_LAMBDA( const size_t i, T_V_FLOAT &updated_energy ) {
            updated_energy += energy( i );
        },
        system_energy );

    Kokkos::fence();

    // TODO: replace ( 0 ): hardcoded
    system_energy += s->N_local * mode->atomicEnergyOffset( 0 );
    system_energy /= mode->convEnergy;
    system_energy += s->N * mode->meanEnergy;
    // TODO: generalize (hartree to eV conversion)
    system_energy *= 27.211384021355236;
    step++;
    return system_energy;
}
