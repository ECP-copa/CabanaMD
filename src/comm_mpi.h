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

#ifndef COMM_MPI_H
#define COMM_MPI_H

#include "mpi.h"

#include <Cabana_Core.hpp>

#include <types.h>
#include <system.h>

#include <algorithm>
#include <memory>
#include <vector>

class Comm {

  // Variables Comm doesn't own but requires for computations

  T_INT N_local;
  T_INT N_ghost;

  System s;
  typename AoSoA::member_slice_type<Positions> x;
  typename AoSoA::member_slice_type<Forces> f;
  typename AoSoA::member_slice_type<Velocities> v;
  typename AoSoA::member_slice_type<IDs> id;
  typename AoSoA::member_slice_type<Types> type;
  typename AoSoA::member_slice_type<Charges> q;

  // Owned Variables

  int phase; // Communication Phase
  int proc_neighbors_recv[6]; // Neighbor for each phase
  int proc_neighbors_send[6]; // Neighbor for each phase
  int proc_num_recv[6];  // Number of received particles in each phase
  int proc_num_send[6];  // Number of send particles in each phase
  int proc_pos[3];       // My process position
  int proc_grid[3];      // Process Grid size
  int proc_rank;         // My Process rank
  int proc_size;         // Number of processes

  Kokkos::View<int, Kokkos::MemoryTraits<Kokkos::Atomic> > pack_count;

  Kokkos::View<T_INT**,Kokkos::LayoutRight,DeviceType> pack_indicies_all;
  Kokkos::View<T_INT*,Kokkos::LayoutRight,DeviceType> pack_indicies;
  Kokkos::View<T_INT**,Kokkos::LayoutRight,DeviceType> pack_ranks_all;
  Kokkos::View<T_INT*,Kokkos::LayoutRight,DeviceType> pack_ranks;
  Kokkos::View<T_INT*,Kokkos::LayoutRight,DeviceType> pack_ranks_migrate;
  std::vector<int> neighbors;

  std::vector<std::shared_ptr<Cabana::Halo<DeviceType>>> halo_all;

protected:
  System* system;

  T_X_FLOAT comm_depth;

public:

  struct TagExchangeSelf {};
  struct TagExchangePack {};
  
  struct TagHaloPack {};
  struct TagHaloPBC {};

  Comm(System* s, T_X_FLOAT comm_depth_);
  void init();
  void create_domain_decomposition();
  void exchange();
  void exchange_halo();
  void update_halo();
  void update_force();
  void scan_int(T_INT* vals, T_INT count);
  void reduce_int(T_INT* vals, T_INT count);
  void reduce_float(T_FLOAT* vals, T_INT count);
  void reduce_max_int(T_INT* vals, T_INT count);
  void reduce_max_float(T_FLOAT* vals, T_INT count);
  void reduce_min_int(T_INT* vals, T_INT count);
  void reduce_min_float(T_FLOAT* vals, T_INT count);

  // local atom periodic shift
  KOKKOS_INLINE_FUNCTION
  void operator() (const TagExchangeSelf, 
                   const T_INT& i) const {
    if(proc_grid[0]==1) {
      const T_X_FLOAT x1 = x(i,0);
      if(x1>s.domain_x) x(i,0) -= s.domain_x;
      if(x1<0)          x(i,0) += s.domain_x;
    }
    if(proc_grid[1]==1) {
      const T_X_FLOAT y1 = x(i,1);
      if(y1>s.domain_y) x(i,1) -= s.domain_y;
      if(y1<0)          x(i,1) += s.domain_y;
    }
    if(proc_grid[2]==1) {
      const T_X_FLOAT z1 = x(i,2);
      if(z1>s.domain_z) x(i,2) -= s.domain_z;
      if(z1<0)          x(i,2) += s.domain_z;
    }   
  }

  // Mark for Cabana-migrate (periodic shift if needed)
  KOKKOS_INLINE_FUNCTION
  void operator() (const TagExchangePack, 
                   const T_INT& i) const {
    if( (phase == 0) && (x(i,0)>s.sub_domain_hi_x)) {
      const std::size_t pack_idx = pack_count()++;
      if( pack_idx < pack_ranks_migrate.extent(0) ) {
        pack_ranks_migrate(i) = proc_neighbors_send[phase];
        if(proc_pos[0] == proc_grid[0]-1)
          x(i,0) -= s.domain_x;
      }
    }  

    if( (phase == 1) && (x(i,0)<s.sub_domain_lo_x)) {
      const std::size_t pack_idx = pack_count()++;
      if( pack_idx < pack_ranks_migrate.extent(0) ) {
        pack_ranks_migrate(i) = proc_neighbors_send[phase];
        if(proc_pos[0] == 0)
          x(i,0) += s.domain_x;
      }
    }

    if( (phase == 2) && (x(i,1)>s.sub_domain_hi_y)) {
      const std::size_t pack_idx = pack_count()++;
      if( pack_idx < pack_ranks_migrate.extent(0) ) {
        pack_ranks_migrate(i) = proc_neighbors_send[phase];
        if(proc_pos[1] == proc_grid[1]-1)
          x(i,1) -= s.domain_y;
      }
    }
    if( (phase == 3) && (x(i,1)<s.sub_domain_lo_y)) {
      const std::size_t pack_idx = pack_count()++;
      if( pack_idx < pack_ranks_migrate.extent(0) ) {
        pack_ranks_migrate(i) = proc_neighbors_send[phase];
        if(proc_pos[1] == 0)
          x(i,1) += s.domain_y;
      }
    }

    if( (phase == 4) && (x(i,2)>s.sub_domain_hi_z)) {
      const std::size_t pack_idx = pack_count()++;
      if( pack_idx < pack_ranks_migrate.extent(0) ) {
        pack_ranks_migrate(i) = proc_neighbors_send[phase];
        if(proc_pos[2] == proc_grid[2]-1)
          x(i,2) -= s.domain_z;
      }
    }
    if( (phase == 5) && (x(i,2)<s.sub_domain_lo_z)) {
      const std::size_t pack_idx = pack_count()++;
      if( pack_idx < pack_ranks_migrate.extent(0) ) {
        pack_ranks_migrate(i) = proc_neighbors_send[phase];
        if(proc_pos[2] == 0)
          x(i,2) += s.domain_z;
      }
    }
  } 
  
  // Add ghosts to Cabana-gather
  KOKKOS_INLINE_FUNCTION
  void operator() (const TagHaloPack,
                   const T_INT& i) const {
    int proc_send = proc_neighbors_send[phase];
    if (proc_send < 0)
      proc_send = proc_rank;

    if(phase == 0) {
      if( x(i,0)>=s.sub_domain_hi_x - comm_depth ) {
        const std::size_t pack_idx = pack_count()++;
        if(pack_idx < pack_indicies.extent(0)) {
          pack_indicies(pack_idx) = i;
          pack_ranks(pack_idx) = proc_send;
        }
      }
    }
    if(phase == 1) {
      if( x(i,0)<=s.sub_domain_lo_x + comm_depth ) {
        const std::size_t pack_idx = pack_count()++;
        if(pack_idx < pack_indicies.extent(0)) {
          pack_indicies(pack_idx) = i;
          pack_ranks(pack_idx) = proc_send;
        }
      }
    }
    if(phase == 2) {
      if( x(i,1)>=s.sub_domain_hi_y - comm_depth ) {
        const std::size_t pack_idx = pack_count()++;
        if(pack_idx < pack_indicies.extent(0)) {
          pack_indicies(pack_idx) = i;
          pack_ranks(pack_idx) = proc_send;
        }
      }
    }
    if(phase == 3) {
      if( x(i,1)<=s.sub_domain_lo_y + comm_depth ) {
        const std::size_t pack_idx = pack_count()++;
        if(pack_idx < pack_indicies.extent(0)) {
          pack_indicies(pack_idx) = i;
          pack_ranks(pack_idx) = proc_send;
        }
      }
    }
    if(phase == 4) {
      if( x(i,2)>=s.sub_domain_hi_z - comm_depth ) {
        const std::size_t pack_idx = pack_count()++;
        if(pack_idx < pack_indicies.extent(0)) {
          pack_indicies(pack_idx) = i;
          pack_ranks(pack_idx) = proc_send;
        }
      }
    }
    if(phase == 5) {
      if( x(i,2)<=s.sub_domain_lo_z + comm_depth ) {
        const std::size_t pack_idx = pack_count()++;
        if(pack_idx < pack_indicies.extent(0)) {
          pack_indicies(pack_idx) = i;
          pack_ranks(pack_idx) = proc_send;
        }
      }
    }
  }

  // Wrap ghosts after update from local counterpart
  // (after MPI, from the perspective of receiving rank)
  KOKKOS_INLINE_FUNCTION
  void operator() (const TagHaloPBC,
                   const T_INT& i) const {

    switch (phase) {
      case 0: if(proc_pos[0] == 0)              x(i,0) -= s.domain_x; break;
      case 1: if(proc_pos[0] == proc_grid[0]-1) x(i,0) += s.domain_x; break;
      case 2: if(proc_pos[1] == 0)              x(i,1) -= s.domain_y; break;
      case 3: if(proc_pos[1] == proc_grid[1]-1) x(i,1) += s.domain_y; break;
      case 4: if(proc_pos[2] == 0)              x(i,2) -= s.domain_z; break;
      case 5: if(proc_pos[2] == proc_grid[2]-1) x(i,2) += s.domain_z; break;
    }
  }

  const char* name();
  int process_rank();
  int num_processes();
  void error(const char *);
};

#endif
