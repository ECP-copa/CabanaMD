/****************************************************************************
 * Copyright (c) 2018 by the Cabana authors                                 *
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

#ifndef FORCE_LJ_CABANA_NEIGH_H
#define FORCE_LJ_CABANA_NEIGH_H
#include<vector>
#include<types.h>
#include<system.h>
#include<binning.h>
#include<neighbor.h>

template<class NeighborClass>
class Force {
private:
  int N_local,ntypes;
  t_x_const_rnd x;
  t_f f;
  t_f_atomic f_a;
  t_id id;
  t_type_const_rnd type;
  Binning::t_bincount bin_count;
  Binning::t_binoffsets bin_offsets;
  T_INT nbinx,nbiny,nbinz,nhalo;
  int step;
  bool use_stackparams;

  typedef Kokkos::View<T_F_FLOAT**> t_fparams;
  typedef Kokkos::View<const T_F_FLOAT**,
      Kokkos::MemoryTraits<Kokkos::RandomAccess>> t_fparams_rnd;
  t_fparams lj1,lj2,cutsq;
  t_fparams_rnd rnd_lj1,rnd_lj2,rnd_cutsq;

  T_F_FLOAT stack_lj1[MAX_TYPES_STACKPARAMS+1][MAX_TYPES_STACKPARAMS+1]; // hardwired space for 12 atom types
  T_F_FLOAT stack_lj2[MAX_TYPES_STACKPARAMS+1][MAX_TYPES_STACKPARAMS+1];
  T_F_FLOAT stack_cutsq[MAX_TYPES_STACKPARAMS+1][MAX_TYPES_STACKPARAMS+1];

  typedef typename NeighborClass::t_neigh_list t_neigh_list;
  t_neigh_list neigh_list;

public:
  typedef T_V_FLOAT value_type;

  template<bool STACKPARAMS>
  struct TagFullNeigh {};

  template<bool STACKPARAMS>
  struct TagHalfNeigh {};

  template<bool STACKPARAMS>
  struct TagFullNeighPE {};

  template<bool STACKPARAMS>
  struct TagHalfNeighPE {};

  typedef Kokkos::RangePolicy<TagFullNeigh<false>,Kokkos::IndexType<T_INT> > t_policy_full_neigh;
  typedef Kokkos::RangePolicy<TagHalfNeigh<false>,Kokkos::IndexType<T_INT> > t_policy_half_neigh;
  typedef Kokkos::RangePolicy<TagFullNeighPE<false>,Kokkos::IndexType<T_INT> > t_policy_full_neigh_pe;
  typedef Kokkos::RangePolicy<TagHalfNeighPE<false>,Kokkos::IndexType<T_INT> > t_policy_half_neigh_pe;

  typedef Kokkos::RangePolicy<TagFullNeigh<true>,Kokkos::IndexType<T_INT> > t_policy_full_neigh_stackparams;
  typedef Kokkos::RangePolicy<TagHalfNeigh<true>,Kokkos::IndexType<T_INT> > t_policy_half_neigh_stackparams;
  typedef Kokkos::RangePolicy<TagFullNeighPE<true>,Kokkos::IndexType<T_INT> > t_policy_full_neigh_pe_stackparams;
  typedef Kokkos::RangePolicy<TagHalfNeighPE<true>,Kokkos::IndexType<T_INT> > t_policy_half_neigh_pe_stackparams;

  bool half_neigh, comm_newton;

  Force(System* system, bool half_neigh_);

  void init_coeff(std::vector<int> force_types, std::vector<double> force_coeff);

  void compute(System* system, Binning* binning, Neighbor* neighbor );
  T_F_FLOAT compute_energy(System* system, Binning* binning, Neighbor* neighbor);

  template<bool STACKPARAMS>
  KOKKOS_INLINE_FUNCTION
  void operator() (TagFullNeigh<STACKPARAMS>, const T_INT& i) const;

  template<bool STACKPARAMS>
  KOKKOS_INLINE_FUNCTION
  void operator() (TagHalfNeigh<STACKPARAMS>, const T_INT& i) const;

  template<bool STACKPARAMS>
  KOKKOS_INLINE_FUNCTION
  void operator() (TagFullNeighPE<STACKPARAMS>, const T_INT& i, T_V_FLOAT& PE) const;

  template<bool STACKPARAMS>
  KOKKOS_INLINE_FUNCTION
  void operator() (TagHalfNeighPE<STACKPARAMS>, const T_INT& i, T_V_FLOAT& PE) const;

  const char* name();
};

#endif
