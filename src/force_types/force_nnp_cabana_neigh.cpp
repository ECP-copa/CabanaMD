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
//    2. Redistributions in binary form must reproduce the above copyright notice,
//       this list of conditions and the following disclaimer in the documentation
//       and/or other materials provided with the distribution.
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
//  Questions? Contact Christian R. Trott (crtrott@sandia.gov)
//************************************************************************

#include <force_nnp_cabana_neigh.h>
#include <string.h>
#include <string>
#include <iostream>
#define VECLEN 16
#define MAX_SF 30 //TODO: hardcoded 

ForceNNP::ForceNNP(System* system, bool half_neigh_):Force(system,half_neigh) {
  ntypes = system->ntypes;
  N_local = 0;
  step = 0;
  half_neigh = half_neigh_;
  //resize nnp_data to have size as big as number of atoms in system
  AoSoA_NNP nnp_data ("ForceNNP::nnp_data", system->N);
  dEdG = t_mass("ForceNNP::dEdG", MAX_SF);
}


void ForceNNP::create_neigh_list(System* system) {
  N_local = system->N_local;

  double grid_min[3] = {-system->domain_x,-system->domain_y,-system->domain_z};
  double grid_max[3] = {2*system->domain_x,2*system->domain_y,2*system->domain_z};

  auto x = Cabana::slice<Positions>(system->xvf);
  auto id = Cabana::slice<IDs>(system->xvf);
  t_verletlist_full_2D list( x, 0, N_local, neigh_cut, 1.0, grid_min, grid_max );
  neigh_list = list;
}


const char* ForceNNP::name() {return half_neigh?"Force:NNPCabanaVerletHalf":"Force:NNPCabanaVerletFull";}

void ForceNNP::init_coeff(T_X_FLOAT neigh_cutoff, char** args) {
  neigh_cut = neigh_cutoff;
  mode = new(nnp::Mode);
  mode->initialize();
  std::string settingsfile = std::string(args[3]) + "/input.nn"; //arg[3] gives directory path
  mode->loadSettingsFile(settingsfile);
  mode->setupNormalization();
  numSymmetryFunctionsPerElement = mode->setupElementMap(numSymmetryFunctionsPerElement);
  mode->setupElements();
  mode->setupCutoff();
  numSymmetryFunctionsPerElement = mode->setupSymmetryFunctions(numSymmetryFunctionsPerElement);
  mode->setupSymmetryFunctionGroups();
  mode->setupNeuralNetwork();
  std::string scalingfile = std::string(args[3]) + "/scaling.data";
  mode->setupSymmetryFunctionScaling(scalingfile);
  std::string weightsfile = std::string(args[3]) + "/weights.%03zu.data";
  mode->setupSymmetryFunctionStatistics(false, false, true, false);
  mode->setupNeuralNetworkWeights(weightsfile);

}


void ForceNNP::compute(System* s) {
  //nnp::Mode* mode = new(nnp::Mode);
  nnp_data.resize(s->N);
  mode->calculateSymmetryFunctionGroups(s, nnp_data, neigh_list, true);
  mode->calculateAtomicNeuralNetworks(s, nnp_data, dEdG, true);
  mode->calculateForces(s, numSymmetryFunctionsPerElement, nnp_data, dEdG, neigh_list);
}

T_V_FLOAT ForceNNP::compute_energy(System* s) {
    
    auto energy = Cabana::slice<NNPNames::energy>(nnp_data);
    T_V_FLOAT system_energy=0.0;
    // Loop over all atoms and add atomic contributions to total energy.
    for (int i = 0; i < energy.size(); ++i)
    {
        //std::cout << "i = " << energy(i) << std::endl;
        system_energy += energy(i);
    }

  /*if(half_neigh)
    Kokkos::parallel_reduce("ForceNNPCabanaNeigh::compute_energy", t_policy_half_neigh_pe_stackparams(0, system->N_local), *this, energy);
  else
    Kokkos::parallel_reduce("ForceNNPCabanaNeigh::compute_energy", t_policy_full_neigh_pe_stackparams(0, system->N_local), *this, energy);

  Kokkos::fence();
  */
  system_energy /= s->cfenergy;
  if (s->normalize)
    system_energy /= s->convEnergy;
  step++;
  return system_energy;
}
