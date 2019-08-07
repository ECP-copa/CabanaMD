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

#include<comm_mpi.h>

Comm::Comm(System* s, T_X_FLOAT comm_depth_):neighbors(7),system(s),comm_depth(comm_depth_) {
  pack_count = Kokkos::View<int>("CommMPI::pack_count");
  pack_indicies_all = Kokkos::View<T_INT**,Kokkos::LayoutRight,DeviceType>("CommMPI::pack_indicies_all",6,200);
  pack_ranks_all = Kokkos::View<T_INT**,Kokkos::LayoutRight,DeviceType>("CommMPI::pack_ranks_all",6,200);
}

void Comm::init() {}

void Comm::create_domain_decomposition() {

  MPI_Comm_size(MPI_COMM_WORLD, &proc_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);

  int ipx = 1;

  double area_xy = system->domain_x * system->domain_y;
  double area_xz = system->domain_x * system->domain_z;
  double area_yz = system->domain_y * system->domain_z;

  double smallest_surface = 2.0 * (area_xy + area_xz + area_yz);

  while(ipx <= proc_size) {
    if(proc_size % ipx == 0) {
      int nremain = proc_size / ipx;
      int ipy = 1;

      while(ipy <= nremain) {
        if(nremain % ipy == 0) {
          int ipz = nremain / ipy;
          double surface = area_xy / ipx / ipy + area_xz / ipx / ipz + area_yz / ipy / ipz;

          if(surface < smallest_surface) {
            smallest_surface = surface;
            proc_grid[0] = ipx;
            proc_grid[1] = ipy;
            proc_grid[2] = ipz;
          }
        }

        ipy++;
      }
    }

    ipx++;
  }
  proc_pos[2] = proc_rank / (proc_grid[0] * proc_grid[1]);
  proc_pos[1] = (proc_rank % (proc_grid[0] * proc_grid[1])) / proc_grid[0];
  proc_pos[0] = proc_rank % proc_grid[0];

  if(proc_grid[0]>1) {
    proc_neighbors_send[1] = ( proc_pos[0] > 0 )                ? proc_rank - 1 :
                                                                proc_rank + (proc_grid[0]-1);
    proc_neighbors_send[0] = ( proc_pos[0] < (proc_grid[0]-1) ) ? proc_rank + 1 :
                                                                proc_rank - (proc_grid[0]-1);

  } else {
    proc_neighbors_send[0] = -1;
    proc_neighbors_send[1] = -1;
  }

  if(proc_grid[1]>1) {
    proc_neighbors_send[3] = ( proc_pos[1] > 0 )                ? proc_rank - proc_grid[0] :
                                                                proc_rank + proc_grid[0]*(proc_grid[1]-1);
    proc_neighbors_send[2] = ( proc_pos[1] < (proc_grid[1]-1) ) ? proc_rank + proc_grid[0] :
                                                                proc_rank - proc_grid[0]*(proc_grid[1]-1);
  } else {
    proc_neighbors_send[2] = -1;
    proc_neighbors_send[3] = -1;
  }

  if(proc_grid[2]>1) {
    proc_neighbors_send[5] = ( proc_pos[2] > 0 )                ? proc_rank - proc_grid[0]*proc_grid[1] :
                                                                proc_rank + proc_grid[0]*proc_grid[1]*(proc_grid[2]-1);
    proc_neighbors_send[4] = ( proc_pos[2] < (proc_grid[2]-1) ) ? proc_rank + proc_grid[0]*proc_grid[1] :
                                                                proc_rank - proc_grid[0]*proc_grid[1]*(proc_grid[2]-1);
  } else {
    proc_neighbors_send[4] = -1;
    proc_neighbors_send[5] = -1;
  }

  proc_neighbors_recv[0] = proc_neighbors_send[1];
  proc_neighbors_recv[1] = proc_neighbors_send[0];
  proc_neighbors_recv[2] = proc_neighbors_send[3];
  proc_neighbors_recv[3] = proc_neighbors_send[2];
  proc_neighbors_recv[4] = proc_neighbors_send[5];
  proc_neighbors_recv[5] = proc_neighbors_send[4];

  system->sub_domain_x = system->domain_x / proc_grid[0];
  system->sub_domain_y = system->domain_y / proc_grid[1];
  system->sub_domain_z = system->domain_z / proc_grid[2];
  system->sub_domain_lo_x = proc_pos[0] * system->sub_domain_x + system->domain_lo_x;
  system->sub_domain_lo_y = proc_pos[1] * system->sub_domain_y + system->domain_lo_y;
  system->sub_domain_lo_z = proc_pos[2] * system->sub_domain_z + system->domain_lo_z;
  system->sub_domain_hi_x = ( proc_pos[0] + 1 ) * system->sub_domain_x + system->domain_lo_x;
  system->sub_domain_hi_y = ( proc_pos[1] + 1 ) * system->sub_domain_y + system->domain_lo_y;
  system->sub_domain_hi_z = ( proc_pos[2] + 1 ) * system->sub_domain_z + system->domain_lo_z;

  for(int p = 0; p < 6; p ++)
    neighbors[p] = proc_neighbors_send[p];
  neighbors[6] = proc_rank;

  std::sort( neighbors.begin(), neighbors.end() );
  auto unique_end = std::unique( neighbors.begin(), neighbors.end() );
  neighbors.resize( std::distance(neighbors.begin(), unique_end) );
  if (neighbors[0] < 0)
    neighbors.erase(neighbors.begin());
}


void Comm::scan_int(T_INT* vals, T_INT count) {
  if(std::is_same<T_INT,int>::value) {
    MPI_Scan(MPI_IN_PLACE,vals,count,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
  }
}

void Comm::reduce_int(T_INT* vals, T_INT count) {
  if(std::is_same<T_INT,int>::value) {
    MPI_Allreduce(MPI_IN_PLACE,vals,count,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
  }
}

void Comm::reduce_float(T_FLOAT* vals, T_INT count) {
  if(std::is_same<T_FLOAT,double>::value) {
    // This generates MPI_ERR_BUFFER for count>1
    MPI_Allreduce(MPI_IN_PLACE,vals,count,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
  }
}

void Comm::reduce_max_int(T_INT* vals, T_INT count) {
  if(std::is_same<T_INT,int>::value) {
    MPI_Allreduce(MPI_IN_PLACE,vals,count,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
  }
}

void Comm::reduce_max_float(T_FLOAT* vals, T_INT count) {
  if(std::is_same<T_FLOAT,double>::value) {
    MPI_Allreduce(MPI_IN_PLACE,vals,count,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
  }
}

void Comm::reduce_min_int(T_INT* vals, T_INT count) {
  if(std::is_same<T_INT,int>::value) {
    MPI_Allreduce(MPI_IN_PLACE,vals,count,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
  }
}

void Comm::reduce_min_float(T_FLOAT* vals, T_INT count) {
  if(std::is_same<T_FLOAT,double>::value) {
    MPI_Allreduce(MPI_IN_PLACE,vals,count,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
  }
}

void Comm::exchange() {

  Kokkos::Profiling::pushRegion("Comm::exchange");

  N_local = system->N_local;
  system->resize(N_local);
  s = *system;
  x = Cabana::slice<Positions>(s.xvf);

  pack_ranks_migrate = Kokkos::View<T_INT*,Kokkos::LayoutRight,DeviceType>( "pack_ranks_migrate", x.size());
  Kokkos::parallel_for("CommMPI::exchange_self",
            Kokkos::RangePolicy<TagExchangeSelf, Kokkos::IndexType<T_INT> >(0,N_local), *this);

  T_INT N_total_recv = 0;
  T_INT N_total_send = 0;

  for(phase = 0; phase < 6; phase ++) {
    proc_num_send[phase] = 0;
    proc_num_recv[phase] = 0;
    Kokkos::deep_copy(pack_ranks_migrate,proc_rank);

    T_INT count = 0;
    Kokkos::deep_copy(pack_count,0);

    if(proc_grid[phase/2]>1) {
      // If a previous phase resized the AoSoA, export ranks needs to be resized as well
      if(pack_ranks_migrate.extent(0) != x.size()) {
        Kokkos::resize(pack_ranks_migrate, x.size());
        Kokkos::deep_copy(pack_ranks_migrate,proc_rank);
      }

      Kokkos::parallel_for("CommMPI::exchange_pack",
                Kokkos::RangePolicy<TagExchangePack, Kokkos::IndexType<T_INT> >(0,x.size()),
                *this);

      Kokkos::deep_copy(count,pack_count);
      proc_num_send[phase] = count;

      Cabana::Distributor<DeviceType> distributor( MPI_COMM_WORLD, pack_ranks_migrate, neighbors );
      Cabana::migrate( distributor, s.xvf );
      system->resize(distributor.totalNumImport()); // Resized by migrate, but not within System
      s = *system;
      x = Cabana::slice<Positions>(s.xvf);

      proc_num_recv[phase] = distributor.totalNumImport()+count-distributor.totalNumExport();
      count = proc_num_recv[phase];
    }

    N_total_recv += proc_num_recv[phase];
    N_total_send += proc_num_send[phase];
  }

  N_local = N_local + N_total_recv - N_total_send;

  system->N_local = N_local;
  system->N_ghost = 0;

  Kokkos::Profiling::popRegion();
}

void Comm::exchange_halo() {

  Kokkos::Profiling::pushRegion("Comm::exchange_halo");

  N_local = system->N_local;
  N_ghost = 0;

  s = *system;
  x = Cabana::slice<Positions>(s.xvf);

  for(phase = 0; phase < 6; phase ++) {
    pack_indicies = Kokkos::subview(pack_indicies_all,phase,Kokkos::ALL());
    pack_ranks = Kokkos::subview(pack_ranks_all,phase,Kokkos::ALL());

    T_INT count = 0;
    Kokkos::deep_copy(pack_count,0);

    T_INT nparticles = N_local + N_ghost - ( (phase%2==1) ? proc_num_recv[phase-1]:0 );
    Kokkos::parallel_for("CommMPI::halo_exchange_pack",
                Kokkos::RangePolicy<TagHaloPack, Kokkos::IndexType<T_INT> >(0,nparticles),
                *this);

    Kokkos::deep_copy(count,pack_count);
    if((unsigned) count > pack_indicies.extent(0)) {
      Kokkos::resize(pack_indicies_all,6,count*1.1);
      pack_indicies = Kokkos::subview(pack_indicies_all,phase,Kokkos::ALL());
      Kokkos::resize(pack_ranks_all,6,count*1.1);
      pack_ranks = Kokkos::subview(pack_ranks_all,phase,Kokkos::ALL());

      Kokkos::deep_copy(pack_count,0);
      Kokkos::parallel_for("CommMPI::halo_exchange_pack",
                           Kokkos::RangePolicy<TagHaloPack, Kokkos::IndexType<T_INT> >(0,nparticles),
                  *this);
    }
    proc_num_send[phase] = count;

    pack_indicies = Kokkos::subview(pack_indicies, std::pair<size_t, size_t>(0,proc_num_send[phase]));
    pack_ranks = Kokkos::subview(pack_ranks, std::pair<size_t, size_t>(0,proc_num_send[phase]));

    Cabana::Halo<DeviceType> halo(
        MPI_COMM_WORLD, N_local+N_ghost, pack_indicies, pack_ranks, neighbors );
    system->resize( halo.numLocal() + halo.numGhost() );
    s=*system;
    x = Cabana::slice<Positions>(s.xvf);
    Cabana::gather( halo, s.xvf );

    proc_num_recv[phase] = halo.numGhost();
    count = proc_num_recv[phase];

    Kokkos::deep_copy(pack_count,0);
    Kokkos::parallel_for("CommMPI::halo_exchange_pack_wrap",
                Kokkos::RangePolicy<TagHaloPBC, Kokkos::IndexType<T_INT> >(
                         halo.numLocal(),halo.numLocal()+halo.numGhost()),
                *this);

    N_ghost += count;
  }

  system->N_ghost = N_ghost;

  Kokkos::Profiling::popRegion();
}

void Comm::update_halo() {

  Kokkos::Profiling::pushRegion("Comm::update_halo");

  N_local = system->N_local;
  N_ghost = 0;
  s=*system;
  x = Cabana::slice<Positions>(s.xvf);

  for(phase = 0; phase<6; phase++) {
    pack_indicies = Kokkos::subview(pack_indicies_all, phase, std::pair<size_t, size_t>(0,proc_num_send[phase]));
    pack_ranks = Kokkos::subview(pack_ranks_all, phase, std::pair<size_t, size_t>(0,proc_num_send[phase]));

    Cabana::Halo<DeviceType> halo(
        MPI_COMM_WORLD, N_local+N_ghost, pack_indicies, pack_ranks, neighbors );
    system->resize( halo.numLocal() + halo.numGhost() );
    s=*system;
    x = Cabana::slice<Positions>(s.xvf);
    Cabana::gather( halo, s.xvf );

    Kokkos::parallel_for("CommMPI::halo_update_PBC",
                Kokkos::RangePolicy<TagHaloUpdatePBC, Kokkos::IndexType<T_INT> >(
                                  halo.numLocal(),halo.numLocal()+halo.numGhost()),
                *this);

    N_ghost += proc_num_recv[phase];
  }

  Kokkos::Profiling::popRegion();
}

void Comm::update_force() {

  Kokkos::Profiling::pushRegion("Comm::update_force");

  N_local = system->N_local;
  N_ghost = 0;
  s=*system;
  f = Cabana::slice<Forces>(s.xvf);

  for(phase = 5; phase>=0; phase--) {
    pack_indicies = Kokkos::subview(pack_indicies_all, phase, std::pair<size_t, size_t>(0,proc_num_send[phase]));
    pack_ranks = Kokkos::subview(pack_ranks_all, phase, std::pair<size_t, size_t>(0,proc_num_send[phase]));

    Cabana::Halo<DeviceType> halo(
        MPI_COMM_WORLD, N_local+N_ghost, pack_indicies, pack_ranks, neighbors );
    system->resize( halo.numLocal() + halo.numGhost() );
    s=*system;
    f = Cabana::slice<Forces>(s.xvf);
    Cabana::scatter( halo, f );

    N_ghost += proc_num_recv[phase];
  }

  Kokkos::Profiling::popRegion();
}

const char* Comm::name() { return "Comm:CabanaMPI"; }

int Comm::process_rank() { return proc_rank; }
int Comm::num_processes() { return proc_size; }
void Comm::error(const char *errormsg) {
  if(proc_rank==0)
  printf("%s\n",errormsg);
  MPI_Abort(MPI_COMM_WORLD,1);
}
