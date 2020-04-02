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

#include <iostream>
#include <string.h>
#include <string>

template <class t_System, class t_System_NNP, class t_neighbor,
          class t_neigh_parallel, class t_angle_parallel>
ForceNNP<t_System, t_System_NNP, t_neighbor, t_neigh_parallel,
         t_angle_parallel>::ForceNNP( t_System *system, bool half_neigh_ )
    : Force<t_System>( system, half_neigh )
{
    ntypes = system->ntypes;
    N_local = 0;
    step = 0;
    half_neigh = half_neigh_;

    system_nnp = new t_System_NNP;

    if ( system->do_print )
        printf( "Using: NNPVectorLength:%i %s\n", CabanaMD_VECTORLENGTH_NNP,
                system_nnp->name() );
}

template <class t_System, class t_System_NNP, class t_neighbor,
          class t_neigh_parallel, class t_angle_parallel>
void ForceNNP<t_System, t_System_NNP, t_neighbor, t_neigh_parallel,
              t_angle_parallel>::create_neigh_list( t_System *system )
{
    N_local = system->N_local;
    double grid_min[3] = {system->sub_domain_lo_x - system->sub_domain_x,
                          system->sub_domain_lo_y - system->sub_domain_y,
                          system->sub_domain_lo_z - system->sub_domain_z};
    double grid_max[3] = {system->sub_domain_hi_x + system->sub_domain_x,
                          system->sub_domain_hi_y + system->sub_domain_y,
                          system->sub_domain_hi_z + system->sub_domain_z};

    system->slice_x();
    auto x = system->x;

    t_neighbor list( x, 0, N_local, neigh_cut, 1.0, grid_min, grid_max );
    neigh_list = list;
}

template <class t_System, class t_System_NNP, class t_neighbor,
          class t_neigh_parallel, class t_angle_parallel>
const char *ForceNNP<t_System, t_System_NNP, t_neighbor, t_neigh_parallel,
                     t_angle_parallel>::name()
{
    return half_neigh ? "Force:NNPCabanaVerletHalf"
                      : "Force:NNPCabanaVerletFull";
}

template <class t_System, class t_System_NNP, class t_neighbor,
          class t_neigh_parallel, class t_angle_parallel>
void ForceNNP<t_System, t_System_NNP, t_neighbor, t_neigh_parallel,
              t_angle_parallel>::init_coeff( T_X_FLOAT neigh_cutoff,
                                             char **args )
{
    neigh_cut = neigh_cutoff;
    mode = new ( nnpCbn::Mode );
    mode->initialize();
    std::string settingsfile =
        std::string( args[3] ) + "/input.nn"; // arg[3] gives directory path
    mode->loadSettingsFile( settingsfile );
    mode->setupNormalization();
    mode->setupElementMap();
    atomicEnergyOffset = mode->setupElements( atomicEnergyOffset );
    mode->setupCutoff();
    h_numSFperElem = mode->setupSymmetryFunctions( h_numSFperElem );
    d_numSFperElem =
        t_mass( "ForceNNP::numSymmetryFunctionsPerElement", mode->numElements );
    mode->setupSymmetryFunctionGroups();
    mode->setupNeuralNetwork();
    std::string scalingfile = std::string( args[3] ) + "/scaling.data";
    mode->setupSymmetryFunctionScaling( scalingfile );
    std::string weightsfile = std::string( args[3] ) + "/weights.%03zu.data";
    mode->setupSymmetryFunctionStatistics( false, false, true, false );
    mode->setupNeuralNetworkWeights( weightsfile );
}

template <class t_System, class t_System_NNP, class t_neighbor,
          class t_neigh_parallel, class t_angle_parallel>
void ForceNNP<t_System, t_System_NNP, t_neighbor, t_neigh_parallel,
              t_angle_parallel>::compute( t_System *s )
{
    system_nnp->resize( s->N_local );
    Kokkos::deep_copy( d_numSFperElem, h_numSFperElem );
    mode->calculateSymmetryFunctionGroups<t_System, t_System_NNP, t_neighbor,
                                          t_neigh_parallel, t_angle_parallel>(
        s, system_nnp, neigh_list );
    mode->calculateAtomicNeuralNetworks<t_System, t_System_NNP, t_neighbor,
                                        t_neigh_parallel, t_angle_parallel>(
        s, system_nnp, d_numSFperElem );
    mode->calculateForces<t_System, t_System_NNP, t_neighbor, t_neigh_parallel,
                          t_angle_parallel>( s, system_nnp, neigh_list );
}

template <class t_System, class t_System_NNP, class t_neighbor,
          class t_neigh_parallel, class t_angle_parallel>
T_V_FLOAT ForceNNP<t_System, t_System_NNP, t_neighbor, t_neigh_parallel,
                   t_angle_parallel>::compute_energy( t_System *s )
{
    system_nnp->slice_E();
    auto energy = system_nnp->E;
    T_V_FLOAT system_energy = 0.0;
    // Loop over all atoms and add atomic contributions to total energy.
    Kokkos::parallel_reduce(
        "ForceNNPCabanaNeigh::compute_energy", s->N_local,
        KOKKOS_LAMBDA( const size_t i, T_V_FLOAT &updated_energy ) {
            updated_energy += energy( i );
        },
        system_energy );

    Kokkos::fence();

    int proc_rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &proc_rank );
    if ( proc_rank == 0 ) // only add offset once
        system_energy +=
            s->N * atomicEnergyOffset( 0 ); // TODO: replace hardcoded

    system_energy /= mode->convEnergy;
    system_energy += s->N * mode->meanEnergy;
    system_energy *=
        27.211384021355236; // hartree to eV conversion (TODO: look into this)
    step++;
    return system_energy;
}
