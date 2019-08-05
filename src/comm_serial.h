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

#ifndef COMM_SERIAL_H
#define COMM_SERIAL_H
#include <Cabana_Slice.hpp>

#include <types.h>
#include <system.h>

class Comm {

  // Variables Comm doesn't own but requires for computations
  T_INT N_local;
  T_INT N_ghost;

  System s;
  typename t_AoSoA_x::member_slice_type<0> x;
  typename t_AoSoA_x::member_slice_type<0> f;

  // Owned Variables

  int phase; // Communication Phase
  T_INT num_ghost[6];

  T_INT ghost_offsets[6];

  T_INT num_packed;
  Kokkos::View<int, Kokkos::MemoryTraits<Kokkos::Atomic> > pack_count;

  Kokkos::View<T_INT**,Kokkos::LayoutRight> pack_indicies_all;
  Kokkos::View<T_INT*,Kokkos::LayoutRight,Kokkos::MemoryTraits<Kokkos::Unmanaged> > pack_indicies;

protected:
  System* system;

  T_X_FLOAT comm_depth;

public:

  struct TagUnpack {};

  struct TagExchangeSelf {};
  
  struct TagHaloSelf {};
  struct TagHaloUpdateSelf {};
  struct TagHaloForceSelf {};

  Comm(System* s, T_X_FLOAT comm_depth_);
  ~Comm();

  // Move particles which left local domain
  void exchange();
  // Exchange ghost particles
  void exchange_halo();
  // Update ghost particles
  void update_halo();
  // Reverse communication of forces
  void update_force();

  KOKKOS_INLINE_FUNCTION
  void operator() (const TagExchangeSelf, 
                   const T_INT& i) const {
    const T_X_FLOAT x1 = x(i,0);
    if(x1>s.domain_x) x(i,0) -= s.domain_x;
    if(x1<0)          x(i,0) += s.domain_x;

    const T_X_FLOAT y1 = x(i,1);
    if(y1>s.domain_y) x(i,1) -= s.domain_y;
    if(y1<0)          x(i,1) += s.domain_y;

    const T_X_FLOAT z1 = x(i,2);
    if(z1>s.domain_z) x(i,2) -= s.domain_z;
    if(z1<0)          x(i,2) += s.domain_z;
  }


  KOKKOS_INLINE_FUNCTION
  void operator() (const TagHaloSelf,
                   const T_INT& i) const {
    if(phase == 0) {
      if( x(i,0)>=s.sub_domain_hi_x - comm_depth ) {
        const int pack_idx = pack_count()++;
        if(((unsigned) pack_idx < pack_indicies.extent(0)) &&
           (((unsigned) N_local+N_ghost+pack_idx) < x.size())) {
          pack_indicies(pack_idx) = i;
          s.set_particle(i, N_local + N_ghost + pack_idx);
          x(N_local + N_ghost + pack_idx,0) -= s.domain_x;
        }
      }
    }
    if(phase == 1) {
      if( x(i,0)<=s.sub_domain_lo_x + comm_depth ) {
        const int pack_idx = pack_count()++;
        if(((unsigned) pack_idx < pack_indicies.extent(0)) &&
           (((unsigned) N_local+N_ghost+pack_idx) < x.size())) {
          pack_indicies(pack_idx) = i;
          s.set_particle(i, N_local + N_ghost + pack_idx);
          x(N_local + N_ghost + pack_idx,0) += s.domain_x;
        }
      }
    }
    if(phase == 2) {
      if( x(i,1)>=s.sub_domain_hi_y - comm_depth ) {
        const int pack_idx = pack_count()++;
        if(((unsigned) pack_idx < pack_indicies.extent(0)) &&
           (((unsigned) N_local+N_ghost+pack_idx) < x.size())) {
          pack_indicies(pack_idx) = i;
          s.set_particle(i, N_local + N_ghost + pack_idx);
          x(N_local + N_ghost + pack_idx,1) -= s.domain_y;
        }
      }
    }
    if(phase == 3) {
      if( x(i,1)<=s.sub_domain_lo_y + comm_depth ) {
        const int pack_idx = pack_count()++;
        if(((unsigned) pack_idx < pack_indicies.extent(0)) &&
           (((unsigned) N_local+N_ghost+pack_idx) < x.size())) {
          pack_indicies(pack_idx) = i;
          s.set_particle(i, N_local + N_ghost + pack_idx);
          x(N_local + N_ghost + pack_idx,1) += s.domain_y;
        }
      }
    }
    if(phase == 4) {
      if( x(i,2)>=s.sub_domain_hi_z - comm_depth ) {
        const int pack_idx = pack_count()++;
        if(((unsigned) pack_idx < pack_indicies.extent(0)) &&
           (((unsigned) N_local+N_ghost+pack_idx) < x.size())) {
          pack_indicies(pack_idx) = i;
          s.set_particle(i, N_local + N_ghost + pack_idx);
          x(N_local + N_ghost + pack_idx,2) -= s.domain_z;
        }
      }
    }
    if(phase == 5) {
      if( x(i,2)<=s.sub_domain_lo_z + comm_depth ) {
        const int pack_idx = pack_count()++;
        if(((unsigned) pack_idx < pack_indicies.extent(0)) &&
           (((unsigned) N_local+N_ghost+pack_idx) < x.size())) {
          pack_indicies(pack_idx) = i;
          s.set_particle(i, N_local + N_ghost + pack_idx);
          x(N_local + N_ghost + pack_idx,2) += s.domain_z;
        }
      }
    }

  }

  KOKKOS_INLINE_FUNCTION
  void operator() (const TagHaloUpdateSelf,
                   const T_INT& i) const {

    s.set_particle(pack_indicies(i), N_local + N_ghost + i);
    switch (phase) {
      case 0: x(N_local + N_ghost + i,0) -= s.domain_x; break;
      case 1: x(N_local + N_ghost + i,0) += s.domain_x; break;
      case 2: x(N_local + N_ghost + i,1) -= s.domain_y; break;
      case 3: x(N_local + N_ghost + i,1) += s.domain_y; break;
      case 4: x(N_local + N_ghost + i,2) -= s.domain_z; break;
      case 5: x(N_local + N_ghost + i,2) += s.domain_z; break;
    }
  }

  KOKKOS_INLINE_FUNCTION
  void operator() (const TagHaloForceSelf,
                   const T_INT& ii) const {

    const T_INT i = pack_indicies(ii);
    T_F_FLOAT fx_i = f(ghost_offsets[phase] + ii,0);
    T_F_FLOAT fy_i = f(ghost_offsets[phase] + ii,1);
    T_F_FLOAT fz_i = f(ghost_offsets[phase] + ii,2);

    //printf("FORCESELF %i %i %i %lf %lf\n",i,ghost_offsets[phase] + ii,ghost_offsets[phase],f(i,0),fx_i);
    f(i, 0) += fx_i;
    f(i, 1) += fy_i;
    f(i, 2) += fz_i;

  }

  // Do a sum reduction over floats
  void reduce_float(T_FLOAT* values, T_INT N);
  // Do a sum reduction over integers
  void reduce_int(T_INT* values, T_INT N);
  // Do a max reduction over floats
  void reduce_max_float(T_FLOAT* values, T_INT N);
  // Do a max reduction over integers
  void reduce_max_int(T_INT* values, T_INT N);
  // Do a min reduction over floats
  void reduce_min_float(T_FLOAT* values, T_INT N);
  // Do a min reduction over integers
  void reduce_min_int(T_INT* values, T_INT N);
  // Do an inclusive scan over integers
  void scan_int(T_INT* values, T_INT N);
  // Do a sum reduction over floats with weights
  void weighted_reduce_float(T_FLOAT* values, T_INT* weight, T_INT N);
  // Create a processor grid
  void create_domain_decomposition();
  // Get Processor rank
  int process_rank();
  // Get number of processors
  int num_processes();

  void error(const char *);
  const char* name();
};
#endif
