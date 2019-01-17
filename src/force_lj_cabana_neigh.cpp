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

#include<force_lj_cabana_neigh.h>

Force::Force(System* system, bool half_neigh_):neigh_list_full(init_fullneigh_list()),
					       neigh_list_half(init_halfneigh_list()),
					       half_neigh(half_neigh_),neigh_cut(0.0) {
  ntypes = system->ntypes;

  lj1 = t_fparams("ForceLJCabanaNeigh::lj1",ntypes,ntypes);
  lj2 = t_fparams("ForceLJCabanaNeigh::lj2",ntypes,ntypes);
  cutsq = t_fparams("ForceLJCabanaNeigh::cutsq",ntypes,ntypes);

  N_local = 0;
  step = 0;
}

void Force::init_coeff(T_X_FLOAT neigh_cut_, std::vector<int> force_types, std::vector<double> force_coeff) {
  neigh_cut = neigh_cut_;
  step = 0;

  int one_based_type = 1;
  int t1 = force_types[0]-one_based_type;
  int t2 = force_types[1]-one_based_type;
  double eps = force_coeff[0];
  double sigma = force_coeff[1];
  double cut = force_coeff[2];

  for (int i = 0; i < ntypes; i++) {
    for (int j = 0; j < ntypes; j++) {
      stack_lj1[i][j] = 48.0 * eps * pow(sigma,12.0);
      stack_lj2[i][j] = 24.0 * eps * pow(sigma,6.0);
      stack_cutsq[i][j] = cut*cut;
    }
  }
}

// No default constructor for Cabana::VerletList
t_verletlist_full Force::init_fullneigh_list() {
  AoSoA xvf (1);
  t_slice_x x = xvf.slice<Positions>();
  t_verletlist_full neigh_list_full( x, 0, x.size(), 1.0, 1.0, (const double[]){0.0,0.0,0.0}, (const double[]){1.0,1.0,1.0} );
  return neigh_list_full;
}
t_verletlist_half Force::init_halfneigh_list() {
  AoSoA xvf (1);
  t_slice_x x = xvf.slice<Positions>();
  t_verletlist_half neigh_list_half( x, 0, x.size(), 1.0, 1.0, (const double[]){0.0,0.0,0.0}, (const double[]){1.0,1.0,1.0} );
  return neigh_list_half;
}

void Force::create_neigh_list(System* system) {
  N_local = system->N_local;

  double grid_min[3] = {-system->domain_x,-system->domain_y,-system->domain_z};
  double grid_max[3] = {2*system->domain_x,2*system->domain_y,2*system->domain_z};

  auto x = system->xvf.slice<Positions>();

  if(half_neigh) {
    t_verletlist_half half( x, 0, x.size(), neigh_cut, 1.0, grid_min, grid_max );
    neigh_list_half = half;
  }
  else {
    t_verletlist_full full( x, 0, x.size(), neigh_cut, 1.0, grid_min, grid_max );
    neigh_list_full = full;
  }
}

void Force::compute(System* system) {
  N_local = system->N_local;
  x = system->xvf.slice<Positions>();
  f = system->xvf.slice<Forces>();
  f_a = system->xvf.slice<Forces>();
  id = system->xvf.slice<IDs>();
  type = system->xvf.slice<Types>();

  if(half_neigh)
    Kokkos::parallel_for("ForceLJCabanaNeigh::compute", t_policy_half_neigh_stackparams(0, system->N_local), *this);
  else
    Kokkos::parallel_for("ForceLJCabanaNeigh::compute", t_policy_full_neigh_stackparams(0, system->N_local), *this);

  Kokkos::fence();

  step++;
}

T_V_FLOAT Force::compute_energy(System* system) {
  N_local = system->N_local;
  x = system->xvf.slice<Positions>();
  f = system->xvf.slice<Forces>();
  f_a = system->xvf.slice<Forces>();
  id = system->xvf.slice<IDs>();
  type = system->xvf.slice<Types>();

  T_V_FLOAT energy;

  if(half_neigh)
    Kokkos::parallel_reduce("ForceLJCabanaNeigh::compute_energy", t_policy_half_neigh_pe_stackparams(0, system->N_local), *this, energy);
  else
    Kokkos::parallel_reduce("ForceLJCabanaNeigh::compute_energy", t_policy_full_neigh_pe_stackparams(0, system->N_local), *this, energy);

  Kokkos::fence();

  step++;
  return energy;
}

const char* Force::name() { return half_neigh?"ForceLJCabanaNeighHalf":"ForceLJCabanaNeighFull"; }

KOKKOS_INLINE_FUNCTION
void Force::operator() (TagFullNeigh, const T_INT& i) const {
  const T_F_FLOAT x_i = x(i,0);
  const T_F_FLOAT y_i = x(i,1);
  const T_F_FLOAT z_i = x(i,2);
  const int type_i = type(i);

  int num_neighs = Cabana::NeighborList<t_verletlist_full>::numNeighbor(neigh_list_full, i);

  T_F_FLOAT fxi = 0.0;
  T_F_FLOAT fyi = 0.0;
  T_F_FLOAT fzi = 0.0;

  for(int jj = 0; jj < num_neighs; jj++) {
    int j = Cabana::NeighborList<t_verletlist_full>::getNeighbor(neigh_list_full, i, jj);

    const T_F_FLOAT dx = x_i - x(j,0);
    const T_F_FLOAT dy = y_i - x(j,1);
    const T_F_FLOAT dz = z_i - x(j,2);

    const int type_j = type(j);
    const T_F_FLOAT rsq = dx*dx + dy*dy + dz*dz;

    const T_F_FLOAT cutsq_ij = stack_cutsq[type_i][type_j];

    if( rsq < cutsq_ij ) {
      const T_F_FLOAT lj1_ij = stack_lj1[type_i][type_j];
      const T_F_FLOAT lj2_ij = stack_lj2[type_i][type_j];

      T_F_FLOAT r2inv = 1.0/rsq;
      T_F_FLOAT r6inv = r2inv*r2inv*r2inv;
      T_F_FLOAT fpair = (r6inv * (lj1_ij*r6inv - lj2_ij)) * r2inv;
      fxi += dx*fpair;
      fyi += dy*fpair;
      fzi += dz*fpair;
    }
  }

  f(i,0) += fxi;
  f(i,1) += fyi;
  f(i,2) += fzi;

}

KOKKOS_INLINE_FUNCTION
void Force::operator() (TagHalfNeigh, const T_INT& i) const {
  const T_F_FLOAT x_i = x(i,0);
  const T_F_FLOAT y_i = x(i,1);
  const T_F_FLOAT z_i = x(i,2);
  const int type_i = type(i);

  int num_neighs = Cabana::NeighborList<t_verletlist_half>::numNeighbor(neigh_list_half, i);

  T_F_FLOAT fxi = 0.0;
  T_F_FLOAT fyi = 0.0;
  T_F_FLOAT fzi = 0.0;
  for(int jj = 0; jj < num_neighs; jj++) {
    int j = Cabana::NeighborList<t_verletlist_half>::getNeighbor(neigh_list_half, i, jj);

    const T_F_FLOAT dx = x_i - x(j,0);
    const T_F_FLOAT dy = y_i - x(j,1);
    const T_F_FLOAT dz = z_i - x(j,2);

    const int type_j = type(j);
    const T_F_FLOAT rsq = dx*dx + dy*dy + dz*dz;

    const T_F_FLOAT cutsq_ij = stack_cutsq[type_i][type_j];

    if( rsq < cutsq_ij ) {
      const T_F_FLOAT lj1_ij = stack_lj1[type_i][type_j];
      const T_F_FLOAT lj2_ij = stack_lj2[type_i][type_j];

      T_F_FLOAT r2inv = 1.0/rsq;
      T_F_FLOAT r6inv = r2inv*r2inv*r2inv;
      T_F_FLOAT fpair = (r6inv * (lj1_ij*r6inv - lj2_ij)) * r2inv;
      fxi += dx*fpair;
      fyi += dy*fpair;
      fzi += dz*fpair;
      f_a(j,0) -= dx*fpair;
      f_a(j,1) -= dy*fpair;
      f_a(j,2) -= dz*fpair;
    }
  }
  f_a(i,0) += fxi;
  f_a(i,1) += fyi;
  f_a(i,2) += fzi;

}

KOKKOS_INLINE_FUNCTION
void Force::operator() (TagFullNeighPE, const T_INT& i, T_V_FLOAT& PE) const {
  const T_F_FLOAT x_i = x(i,0);
  const T_F_FLOAT y_i = x(i,1);
  const T_F_FLOAT z_i = x(i,2);
  const int type_i = type(i);
  const bool shift_flag = true;

  int num_neighs = Cabana::NeighborList<t_verletlist_full>::numNeighbor(neigh_list_full, i);

  for(int jj = 0; jj < num_neighs; jj++) {
    int j = Cabana::NeighborList<t_verletlist_full>::getNeighbor(neigh_list_full, i, jj);

    const T_F_FLOAT dx = x_i - x(j,0);
    const T_F_FLOAT dy = y_i - x(j,1);
    const T_F_FLOAT dz = z_i - x(j,2);

    const int type_j = type(j);
    const T_F_FLOAT rsq = dx*dx + dy*dy + dz*dz;

    const T_F_FLOAT cutsq_ij = stack_cutsq[type_i][type_j];

    if( rsq < cutsq_ij ) {
      const T_F_FLOAT lj1_ij = stack_lj1[type_i][type_j];
      const T_F_FLOAT lj2_ij = stack_lj2[type_i][type_j];

      T_F_FLOAT r2inv = 1.0/rsq;
      T_F_FLOAT r6inv = r2inv*r2inv*r2inv;
      PE += 0.5*r6inv * (0.5*lj1_ij*r6inv - lj2_ij) / 6.0; // optimize later

      if (shift_flag) {
        T_F_FLOAT r2invc = 1.0/cutsq_ij;
        T_F_FLOAT r6invc = r2invc*r2invc*r2invc;
        PE -= 0.5*r6invc * (0.5*lj1_ij*r6invc - lj2_ij) / 6.0; // optimize later
      }
    }
  }
}

KOKKOS_INLINE_FUNCTION
void Force::operator() (TagHalfNeighPE, const T_INT& i, T_V_FLOAT& PE) const {
  const T_F_FLOAT x_i = x(i,0);
  const T_F_FLOAT y_i = x(i,1);
  const T_F_FLOAT z_i = x(i,2);
  const int type_i = type(i);
  const bool shift_flag = true;

  int num_neighs = Cabana::NeighborList<t_verletlist_half>::numNeighbor(neigh_list_half, i);

  for(int jj = 0; jj < num_neighs; jj++) {
    int j = Cabana::NeighborList<t_verletlist_half>::getNeighbor(neigh_list_half, i, jj);

    const T_F_FLOAT dx = x_i - x(j,0);
    const T_F_FLOAT dy = y_i - x(j,1);
    const T_F_FLOAT dz = z_i - x(j,2);

    const int type_j = type(j);
    const T_F_FLOAT rsq = dx*dx + dy*dy + dz*dz;

    const T_F_FLOAT cutsq_ij = stack_cutsq[type_i][type_j];

    if( rsq < cutsq_ij ) {
      const T_F_FLOAT lj1_ij = stack_lj1[type_i][type_j];
      const T_F_FLOAT lj2_ij = stack_lj2[type_i][type_j];

      T_F_FLOAT r2inv = 1.0/rsq;
      T_F_FLOAT r6inv = r2inv*r2inv*r2inv;
      T_F_FLOAT fac;
      if(j<N_local) fac = 1.0;
      else fac = 0.5;

      PE += fac * r6inv * (0.5*lj1_ij*r6inv - lj2_ij) / 6.0;  // optimize later

      if (shift_flag) {
        T_F_FLOAT r2invc = 1.0/cutsq_ij;
        T_F_FLOAT r6invc = r2invc*r2invc*r2invc;
        PE -= fac * r6invc * (0.5*lj1_ij*r6invc - lj2_ij) / 6.0;  // optimize later
      }
    }
  }

}
