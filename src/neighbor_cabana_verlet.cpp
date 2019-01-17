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

#include <neighbor_cabana_verlet.h>

Neighbor::Neighbor():neigh_cut(0.0) {
  neigh_type = NEIGH_CABANA_VERLET;
};
~Neighbor::Neighbor() {};

void Neighbor::init(T_X_FLOAT neigh_cut_) { neigh_cut = neigh_cut_; };

void Neighbor::create_neigh_list(System* system, Binning* binning, bool half_neigh_, bool) {
  // Get some data handles
  N_local = system->N_local;
  x = system->x;
  type = system->type;
  id = system->id;
  half_neigh = half_neigh_;

  T_INT total_num_neighs;

  // Reset the neighbor count array
  if( num_neighs.extent(0) < N_local + 1 ) {
    num_neighs = Kokkos::View<T_INT*, MemorySpace>("NeighborsCSR::num_neighs", N_local + 1);
    neigh_offsets = Kokkos::View<T_INT*, MemorySpace>("NeighborsCSR::neigh_offsets", N_local + 1);
  } else
    Kokkos::deep_copy(num_neighs,0);
  num_neighs_atomic = num_neighs;

  // Create the pair list
  nhalo = binning->nhalo;
  nbinx = binning->nbinx - 2*nhalo;
  nbiny = binning->nbiny - 2*nhalo;
  nbinz = binning->nbinz - 2*nhalo;

  T_INT nbins = nbinx*nbiny*nbinz;

  bin_offsets = binning->binoffsets;
  bin_count = binning->bincount;
  permute_vector = binning->permute_vector;

  if(half_neigh)
    Kokkos::parallel_for("NeighborCSR::count_neighbors_half", t_policy_cnh(nbins,Kokkos::AUTO,8),*this);
  else
    Kokkos::parallel_for("NeighborCSR::count_neighbors_full", t_policy_cnf(nbins,Kokkos::AUTO,8),*this);
  Kokkos::fence();

  // Create the Offset list for neighbors of atoms
  Kokkos::parallel_scan("NeighborCSR::create_offsets", t_policy_co(0, N_local), *this);
  Kokkos::fence();

  // Get the total neighbor count
  Kokkos::View<T_INT,MemorySpace> d_total_num_neighs(neigh_offsets,N_local);
  Kokkos::deep_copy(total_num_neighs,d_total_num_neighs);

  // Resize NeighborList
  if( neighs.extent(0) < total_num_neighs )
    neighs = Kokkos::View<T_INT*, MemorySpace> ("NeighborCSR::neighs", total_num_neighs);

  // Copy entries from the PairList to the actual NeighborList
  Kokkos::deep_copy(num_neighs,0);

  if(half_neigh)
    Kokkos::parallel_for("NeighborCSR::fill_neigh_list_half",t_policy_fnlh(nbins,Kokkos::AUTO,8),*this);
  else
    Kokkos::parallel_for("NeighborCSR::fill_neigh_list_full",t_policy_fnlf(nbins,Kokkos::AUTO,8),*this);

  Kokkos::fence();

  // Create actual CSR NeighList
  neigh_list = t_neigh_list(
      Kokkos::View<T_INT*, MemorySpace>( neighs,     Kokkos::pair<T_INT,T_INT>(0,total_num_neighs)),
      Kokkos::View<T_INT*, MemorySpace>( neigh_offsets, Kokkos::pair<T_INT,T_INT>(0,N_local+1)));

}

t_neigh_list Neighbor::get_neigh_list() { return neigh_list; }
const char* Neighbor::name() {return "NeighborCabanaVerlet";}
