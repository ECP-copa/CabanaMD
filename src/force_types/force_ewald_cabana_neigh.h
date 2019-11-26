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

//TODO: I don't really understand this first part
#ifdef MODULES_OPTION_CHECK
  if( (strcmp(argv[i], "--force-iteration") == 0) ) {
    if( (strcmp(argv[i+1], "NEIGH_FULL") == 0) )
      lrforce_iteration_type = FORCE_ITER_NEIGH_FULL;
    if( (strcmp(argv[i+1], "NEIGH_HALF") == 0) )
      lrforce_iteration_type = FORCE_ITER_NEIGH_HALF;
  }
  if( (strcmp(argv[i], "--neigh-type") == 0) ) {
    if( (strcmp(argv[i+1], "NEIGH_2D") == 0) )
      neighbor_type = NEIGH_2D;
    if( (strcmp(argv[i+1], "NEIGH_CSR") == 0) )
      neighbor_type = NEIGH_CSR;
  }
#endif
#ifdef FORCE_MODULES_INSTANTIATION
    else if (input->lrforce_type == FORCE_EWALD) {
      bool half_neigh = input->lrforce_iteration_type == FORCE_ITER_NEIGH_HALF;
      if (input->neighbor_type == NEIGH_2D) {
        if (half_neigh)
          lrforce = new ForceEwald<t_verletlist_half_2D>(system,half_neigh);
        else
          lrforce = new ForceEwald<t_verletlist_full_2D>(system,half_neigh);
      }
      else if (input->neighbor_type == NEIGH_CSR) {
	if (half_neigh)
          lrforce = new ForceEwald<t_verletlist_half_CSR>(system,half_neigh);
        else
          lrforce = new ForceEwald<t_verletlist_full_CSR>(system,half_neigh);
      }
      #undef FORCETYPE_ALLOCATION_MACRO
    }
#endif


#if !defined(MODULES_OPTION_CHECK) && \
    !defined(FORCE_MODULES_INSTANTIATION)

#ifndef FORCE_EWALD_CABANA_NEIGH_H
#define FORCE_EWALD_CABANA_NEIGH_H
#include <Cabana_Core.hpp>

#include<lrforce.h>
#include<types.h>
#include<system.h>

template<class t_neighbor>
class ForceEwald: public LRForce {
private:
  //int N_local,ntypes;
  //typename AoSoA::member_slice_type<Positions> x;
  //typename AoSoA::member_slice_type<Forces> f;
  //typename AoSoA::member_slice_type<Forces>::atomic_access_slice f_a;//TODO: what is this for?
  //typename AoSoA::member_slice_type<IDs> id;
  //typename AoSoA::member_slice_type<Types> type;
  typename AoSoA::member_slice_type<Charges> q;

  double _alpha;
  double _r_max;
  double _k_max;

  //dielectric constant
  double _eps_r = 1.0; //Assume 1 for now (vacuum)

  double *EwaldUk_coeffs;
  
  //Kokkos::View<double *, MemorySpace> domain_width;

  //MPI_Comm comm;

public:

  bool half_neigh;
  T_X_FLOAT neigh_cut;

  t_neighbor neigh_list;

  ForceEwald(System* system, bool half_neigh);

  void tune(char** args);

  //void create_neigh_list(System* system);

  void compute(System* system);
  T_F_FLOAT compute_energy(System* system);

  const char* name();
};

#endif
#endif
