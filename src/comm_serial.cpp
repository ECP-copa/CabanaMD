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

#include<comm_serial.h>

Comm::Comm(System* s, T_X_FLOAT comm_depth_):system(s),comm_depth(comm_depth_) {
  pack_count = Kokkos::View<int>("CommSerial::pack_count");
  pack_indicies_all = Kokkos::View<T_INT**,Kokkos::LayoutRight>("CommSerial::pack_indicies_all",6,0);
}
Comm::~Comm() {}

void Comm::exchange() {
  s = *system;
  N_local = system->N_local;
  x = Cabana::slice<0>(s.aosoa_x);

  Kokkos::parallel_for("CommSerial::exchange_self",
            Kokkos::RangePolicy<TagExchangeSelf, Kokkos::IndexType<T_INT> >(0,N_local), *this);
}

void Comm::exchange_halo() {

  N_local = system->N_local;
  N_ghost = 0;

  s = *system;
  x = Cabana::slice<0>(s.aosoa_x);

  for(phase = 0; phase < 6; phase ++) {
    pack_indicies = Kokkos::subview(pack_indicies_all,phase,Kokkos::ALL());

    T_INT count = 0;
    Kokkos::deep_copy(pack_count,0);

    T_INT nparticles = N_local + N_ghost - ( (phase%2==1) ? num_ghost[phase-1]:0 );
    Kokkos::parallel_for("CommSerial::halo_exchange_self",
              Kokkos::RangePolicy<TagHaloSelf, Kokkos::IndexType<T_INT> >(0,nparticles),
              *this);
    Kokkos::deep_copy(count,pack_count);
    bool redo = false;
    if((unsigned) N_local+N_ghost+count > x.size()) {
      system->resize(N_local + N_ghost + count);
      s = *system;
      x = Cabana::slice<0>(s.aosoa_x); // reslice after resize
      redo = true;
    }
    if((unsigned) count > pack_indicies.extent(0)) {
      Kokkos::resize(pack_indicies_all,6,count*1.1);
      pack_indicies = Kokkos::subview(pack_indicies_all,phase,Kokkos::ALL());
      redo = true;
    }
    if(redo) {
      Kokkos::deep_copy(pack_count,0);
      Kokkos::parallel_for("CommSerial::halo_exchange_self",
                Kokkos::RangePolicy<TagHaloSelf, Kokkos::IndexType<T_INT> >(0,nparticles),
                *this);
    }

    num_ghost[phase] = count;

    N_ghost += count;
  }
  system->N_ghost = N_ghost;

  if ((unsigned) N_local+N_ghost < x.size()) {
    system->resize(N_local+N_ghost);
  }
}

void Comm::update_halo() {
  N_ghost = 0;
  s=*system;
  x = Cabana::slice<0>(s.aosoa_x);
  for(phase = 0; phase<6; phase++) {
    pack_indicies = Kokkos::subview(pack_indicies_all,phase,Kokkos::ALL());

    Kokkos::parallel_for("CommSerial::halo_update_self",
      Kokkos::RangePolicy<TagHaloUpdateSelf, Kokkos::IndexType<T_INT> >(0,num_ghost[phase]),
      *this);
    N_ghost += num_ghost[phase];
  }
}

void Comm::update_force() {
  //printf("Update Force\n");
  s=*system;
  f = Cabana::slice<0>(s.aosoa_f);
  ghost_offsets[0] = s.N_local;
  for(phase = 1; phase<6; phase++) {
    ghost_offsets[phase] = ghost_offsets[phase-1] + num_ghost[phase-1];
  }

  for(phase = 5; phase>=0; phase--) {
    pack_indicies = Kokkos::subview(pack_indicies_all,phase,Kokkos::ALL());

    Kokkos::parallel_for("CommSerial::halo_force_self",
      Kokkos::RangePolicy<TagHaloForceSelf, Kokkos::IndexType<T_INT> >(0,num_ghost[phase]),
      *this);
  }
}

void Comm::reduce_float(T_FLOAT*, T_INT) {}
void Comm::reduce_int(T_INT*, T_INT) {}
void Comm::reduce_max_float(T_FLOAT*, T_INT) {}
void Comm::reduce_max_int(T_INT*, T_INT) {}
void Comm::reduce_min_float(T_FLOAT*, T_INT) {}
void Comm::reduce_min_int(T_INT*, T_INT) {}
void Comm::scan_int(T_INT*, T_INT) {}
void Comm::weighted_reduce_float(T_FLOAT* , T_INT* , T_INT ) {}

void Comm::create_domain_decomposition() {
  system->sub_domain_lo_x = system->domain_lo_x;
  system->sub_domain_lo_y = system->domain_lo_y;
  system->sub_domain_lo_z = system->domain_lo_z;
  system->sub_domain_hi_x = system->domain_hi_x;
  system->sub_domain_hi_y = system->domain_hi_y;
  system->sub_domain_hi_z = system->domain_hi_z;
  system->sub_domain_x = system->domain_x;
  system->sub_domain_y = system->domain_y;
  system->sub_domain_z = system->domain_z;
}
int Comm::process_rank() {return 0;}
int Comm::num_processes() {return 1;}
void Comm::error(const char *errormsg) {
  printf("%s\n",errormsg);
  exit(1);
}
const char* Comm::name() { return "Comm:Serial"; }
