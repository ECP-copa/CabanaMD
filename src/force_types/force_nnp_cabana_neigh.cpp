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


template <class t_neighbor>
void ForceNNP<t_neighbor>::settings(int narg, char **arg)
{
  if (narg == 0)
    printf("ERROR: Illegal pair_style command");

  // default settings
  int len = strlen("nnp/") + 1; //for \0 I assume
  directory = new char[len];
  strcpy(directory,"nnp/");
  showew = true;
  showewsum = 0;
  maxew = 0;
  resetew = false;
  cflength = 1.0;
  cfenergy = 1.0;
  numExtrapolationWarningsTotal = 0;
  numExtrapolationWarningsSummary = 0;

  int iarg = 0;
  while(iarg < narg) {
    // set NNP directory
    if (strcmp(arg[iarg],"dir") == 0) {
      if (iarg+2 > narg)
        printf("ERROR: Illegal pair_style command");
      delete[] directory;
      len = strlen(arg[iarg+1]) + 2; //for / and \0 I assume
      directory = new char[len];
      printf(directory, "%s/", arg[iarg+1]);
      iarg += 2;
    } 
    // show extrapolation warnings
    else if (strcmp(arg[iarg],"showew") == 0) {
      if (iarg+2 > narg)
        printf("ERROR: Illegal pair_style command");
      if (strcmp(arg[iarg+1],"yes") == 0)
        showew = true;
      else if (strcmp(arg[iarg+1],"no") == 0)
        showew = false;
      else
        printf("ERROR: Illegal pair_style command");
      iarg += 2;
    // show extrapolation warning summary
    }
    else if (strcmp(arg[iarg],"showewsum") == 0) {
      if (iarg+2 > narg)
        printf("ERROR: Illegal pair_style command");
      showewsum = atoi(arg[iarg+1]);
      iarg += 2;
    }
    // maximum allowed extrapolation warnings
    else if (strcmp(arg[iarg],"maxew") == 0) {
      if (iarg+2 > narg)
        printf("ERROR: Illegal pair_style command");
      maxew = atoi(arg[iarg+1]);
      iarg += 2;
    }
    // reset extrapolation warning counter
    else if (strcmp(arg[iarg],"resetew") == 0) {
      if (iarg+2 > narg)
        printf("ERROR: Illegal pair_style command");
      if (strcmp(arg[iarg+1],"yes") == 0)
        resetew = true;
      else if (strcmp(arg[iarg+1],"no") == 0)
        resetew = false;
      else
        printf("ERROR: Illegal pair_style command");
      iarg += 2;
    }
    // length unit conversion factor
    else if (strcmp(arg[iarg],"cflength") == 0) {
      if (iarg+2 > narg)
        printf("ERROR: Illegal pair_style command");
      cflength = atof(arg[iarg+1]);
      iarg += 2;
    }
    // energy unit conversion factor
    else if (strcmp(arg[iarg],"cfenergy") == 0) {
      if (iarg+2 > narg)
        printf("ERROR: Illegal pair_style command");
      cfenergy = atof(arg[iarg+1]);
      iarg += 2;
    }
    else
      printf("ERROR: Illegal pair_style command");
  }
}


template <class t_neighbor>
void ForceNNP<t_neighbor>::coeff(int narg, char **arg)
{
  //if (!allocated)
  //  allocate();

  if (narg != 3)
    printf("ERROR: Incorrect args for pair coefficients");

  int ilo,ihi,jlo,jhi;
  // TODO: look into this
  //force->bounds(FLERR,arg[0],atom->ntypes,ilo,ihi);
  //force->bounds(FLERR,arg[1],atom->ntypes,jlo,jhi);

  maxCutoffRadius = atoi(arg[2]);

  // TODO: Check how this flag is set. (set by n2p2, not by us)
  /*int count = 0;
  for(int i=ilo; i<=ihi; i++) {
    for(int j=MAX(jlo,i); j<=jhi; j++) {
      setflag[i][j] = 1;
      count++;
    }
  }

  if (count == 0)
    printf("ERROR: Incorrect args for pair coefficients");
  */
}


template <class t_neighbor>
void ForceNNP<t_neighbor>::compute(System* s)
{
  
  interface.setLocalAtoms(s);

  // Transfer local neighbor list to NNP interface.
  //transferNeighborList();

  // Compute symmetry functions, atomic neural networks and add up energy.
  interface.process();

  // Do all stuff related to extrapolation warnings.
  //TODO: requires MPI
  //if(showew == true || showewsum > 0 || maxew >= 0) {
  //  handleExtrapolationWarnings();
  //}

  // Calculate forces of local and ghost atoms.
  //interface.getForces();

  // Add energy contribution to total energy.
  //TODO: Rethink this
  //if (eflag_global)
  //   ev_tally(0,0,atom->nlocal,1,interface.getEnergy(),0.0,0.0,0.0,0.0,0.0);

  //TODO: think about printing atomic energy and virial
}
