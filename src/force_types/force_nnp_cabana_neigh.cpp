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
#include <iostream>

template class ForceNNP<t_verletlist_half_2D>;
template class ForceNNP<t_verletlist_full_2D>;
template class ForceNNP<t_verletlist_half_CSR>;
template class ForceNNP<t_verletlist_full_CSR>;


template<class t_neighbor>
ForceNNP<t_neighbor>::ForceNNP(System* system, bool half_neigh_):Force(system,half_neigh_) {
  ntypes = system->ntypes;
  N_local = 0;
  step = 0;
}


template<class t_neighbor>
void ForceNNP<t_neighbor>::create_neigh_list(System* system) {
  N_local = system->N_local;

  double grid_min[3] = {-system->domain_x,-system->domain_y,-system->domain_z};
  double grid_max[3] = {2*system->domain_x,2*system->domain_y,2*system->domain_z};

  auto x = Cabana::slice<Positions>(system->xvf);

  t_neighbor list( x, 0, N_local, neigh_cut, 1.0, grid_min, grid_max );
  neigh_list = list;
}


template<class t_neighbor>
const char* ForceNNP<t_neighbor>::name() {return half_neigh?"Force:NNPCabanaVerletHalf":"Force:NNPCabanaVerletFull";}


template<class t_neighbor>
void ForceNNP<t_neighbor>::setup_pair_style(nnp::Mode*) {
  mode->initialize();
  mode->loadSettingsFile("input.nn");
  mode->setupElementMap();
  mode->setupCutoff();
  mode->setupSymmetryFunctions();
  mode->setupSymmetryFunctionGroups();
  mode->setupSymmetryFunctionStatistics(false, false, true, false);
  mode->setupNeuralNetwork();
  mode->setupSymmetryFunctionScaling();
  mode->setupNeuralNetworkWeights();
}
