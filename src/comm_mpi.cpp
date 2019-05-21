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
  pack_buffer = Kokkos::View<t_particle*>("CommMPI::pack_buffer",200);
  unpack_buffer = Kokkos::View<t_particle*>("CommMPI::pack_buffer",200);
  pack_indicies_all = Kokkos::View<T_INT**,Kokkos::LayoutRight>("CommMPI::pack_indicies_all",6,200);
  exchange_dest_list = Kokkos::View<T_INT*,Kokkos::LayoutRight >("CommMPI::exchange_dest_list",200);
  pack_ranks_all = Kokkos::View<T_INT**,Kokkos::LayoutRight>("CommMPI::pack_ranks_all",6,200);
}

void Comm::init() {};

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
  system->sub_domain_lo_x = proc_pos[0] * system->sub_domain_x;
  system->sub_domain_lo_y = proc_pos[1] * system->sub_domain_y;
  system->sub_domain_lo_z = proc_pos[2] * system->sub_domain_z;
  system->sub_domain_hi_x = ( proc_pos[0] + 1 ) * system->sub_domain_x;
  system->sub_domain_hi_y = ( proc_pos[1] + 1 ) * system->sub_domain_y;
  system->sub_domain_hi_z = ( proc_pos[2] + 1 ) * system->sub_domain_z;
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

  s = *system;
  N_local = system->N_local;
  N_ghost = 0;
  x = s.xvf.slice<Positions>();
  v = s.xvf.slice<Velocities>();
  q = s.xvf.slice<Charges>();
  id = s.xvf.slice<IDs>();
  type = s.xvf.slice<Types>();
  Kokkos::parallel_for("CommMPI::exchange_self",
            Kokkos::RangePolicy<TagExchangeSelf, Kokkos::IndexType<T_INT> >(0,N_local), *this);

  T_INT N_total_recv = 0;
  T_INT N_total_send = 0;

  for(phase = 0; phase < 6; phase ++) {
    proc_num_send[phase] = 0;
    proc_num_recv[phase] = 0;
    pack_indicies = Kokkos::subview(pack_indicies_all,phase,Kokkos::ALL());

    T_INT count = 0;
    Kokkos::deep_copy(pack_count,0);

    if(proc_grid[phase/2]>1) {
      Kokkos::parallel_for("CommMPI::exchange_pack",
                Kokkos::RangePolicy<TagExchangePack, Kokkos::IndexType<T_INT> >(0,N_local+N_ghost),
                *this);

      Kokkos::deep_copy(count,pack_count);
      if(count > pack_indicies.extent(0)) {
        Kokkos::realloc(pack_buffer,count*1.1);
        Kokkos::realloc(pack_indicies_all,6,count*1.1);
        pack_indicies = Kokkos::subview(pack_indicies_all,phase,Kokkos::ALL());
        Kokkos::deep_copy(pack_count,0);
        Kokkos::parallel_for("CommMPI::exchange_pack",
                  Kokkos::RangePolicy<TagExchangePack, Kokkos::IndexType<T_INT> >(0,N_local+N_ghost),
                  *this);
      }
      proc_num_send[phase] = count;
      if(pack_buffer.extent(0)<count)
        pack_buffer = Kokkos::View<t_particle*>("Comm::pack_buffer",count);
      MPI_Request request;
      MPI_Irecv(&proc_num_recv[phase],1,MPI_INT, proc_neighbors_recv[phase],100001,MPI_COMM_WORLD,&request);
      MPI_Send(&proc_num_send[phase],1,MPI_INT, proc_neighbors_send[phase],100001,MPI_COMM_WORLD);
      MPI_Status status;
      MPI_Wait(&request,&status);
      count = proc_num_recv[phase];

      if(unpack_buffer.extent(0)<count) {
        unpack_buffer = Kokkos::View<t_particle*>("Comm::unpack_buffer",count);
      }
      if(proc_num_recv[phase]>0)
        MPI_Irecv(unpack_buffer.data(),proc_num_recv[phase]*sizeof(t_particle)/sizeof(int),MPI_INT, proc_neighbors_recv[phase],100002,MPI_COMM_WORLD,&request);
      if(proc_num_send[phase]>0)
        MPI_Send (pack_buffer.data(),proc_num_send[phase]*sizeof(t_particle)/sizeof(int),MPI_INT, proc_neighbors_send[phase],100002,MPI_COMM_WORLD);
      system->resize(N_local + N_ghost + count);
      s = *system;
      x = s.xvf.slice<Positions>();
      type = s.xvf.slice<Types>();
      if(proc_num_recv[phase]>0)
        MPI_Wait(&request,&status);
      Kokkos::parallel_for("CommMPI::exchange_unpack",
                Kokkos::RangePolicy<TagUnpack, Kokkos::IndexType<T_INT> >(0,proc_num_recv[phase]),
                *this);

    }

    N_ghost += count;
    if (x.size() > N_local+N_ghost) {
      system->resize(N_local+N_ghost);
    }

    N_total_recv += proc_num_recv[phase];
    N_total_send += proc_num_send[phase];
  }
  T_INT N_local_start = N_local;
  T_INT N_exchange = N_ghost;

  N_local = N_local + N_total_recv - N_total_send;
  N_ghost = N_local_start + N_exchange - N_local;

  if(exchange_dest_list.extent(0)<N_ghost)
    Kokkos::realloc(exchange_dest_list,N_ghost);

  Kokkos::parallel_scan("CommMPI::exchange_create_dest_list",
            Kokkos::RangePolicy<TagExchangeCreateDestList, Kokkos::IndexType<T_INT> >(0,N_local),
            *this);
  Kokkos::parallel_scan("CommMPI::exchange_compact",
            Kokkos::RangePolicy<TagExchangeCompact, Kokkos::IndexType<T_INT> >(0,N_ghost),
            *this);

  system->N_local = N_local;
  system->N_ghost = 0;

  Kokkos::Profiling::popRegion();
};

void Comm::exchange_halo() {

  Kokkos::Profiling::pushRegion("Comm::exchange_halo");

  N_local = system->N_local;
  N_ghost = 0;

  s = *system;
  x = s.xvf.slice<Positions>();

  for(phase = 0; phase < 6; phase ++) {
    pack_indicies = Kokkos::subview(pack_indicies_all,phase,Kokkos::ALL());

    T_INT count = 0;
    Kokkos::deep_copy(pack_count,0);

    if(proc_grid[phase/2]>1) {
      Kokkos::parallel_for("CommMPI::halo_exchange_pack",
                Kokkos::RangePolicy<TagHaloPack, Kokkos::IndexType<T_INT> >(0,N_local+N_ghost - ( (phase%2==1) ? proc_num_recv[phase-1]:0)),
                *this);

      Kokkos::deep_copy(count,pack_count);
      if(count > pack_indicies.extent(0)) {
        Kokkos::realloc(pack_buffer,count*1.1);
        Kokkos::resize(pack_indicies_all,6,count*1.1);
        pack_indicies = Kokkos::subview(pack_indicies_all,phase,Kokkos::ALL());
        Kokkos::deep_copy(pack_count,0);
        Kokkos::parallel_for("CommMPI::halo_exchange_pack",
                  Kokkos::RangePolicy<TagHaloPack, Kokkos::IndexType<T_INT> >(0,N_local+N_ghost- ( (phase%2==1) ? proc_num_recv[phase-1]:0)),
                  *this);
      }
      proc_num_send[phase] = count;
      if(pack_buffer.extent(0)<count)
        pack_buffer = Kokkos::View<t_particle*>("Comm::pack_buffer",count);
      MPI_Request request;
      MPI_Irecv(&proc_num_recv[phase],1,MPI_INT, proc_neighbors_recv[phase],100001,MPI_COMM_WORLD,&request);
      MPI_Send(&proc_num_send[phase],1,MPI_INT, proc_neighbors_send[phase],100001,MPI_COMM_WORLD);
      MPI_Status status;
      MPI_Wait(&request,&status);
      count = proc_num_recv[phase];
      if(unpack_buffer.extent(0)<count) {
        unpack_buffer = Kokkos::View<t_particle*>("Comm::unpack_buffer",count);
      }
      MPI_Irecv(unpack_buffer.data(),proc_num_recv[phase]*sizeof(t_particle)/sizeof(int),MPI_INT, proc_neighbors_recv[phase],100002,MPI_COMM_WORLD,&request);
      MPI_Send (pack_buffer.data(),proc_num_send[phase]*sizeof(t_particle)/sizeof(int),MPI_INT, proc_neighbors_send[phase],100002,MPI_COMM_WORLD);
      system->resize(N_local + N_ghost + count);
      s = *system;
      x = s.xvf.slice<Positions>();
      MPI_Wait(&request,&status);
      Kokkos::parallel_for("CommMPI::halo_exchange_unpack",
                Kokkos::RangePolicy<TagUnpack, Kokkos::IndexType<T_INT> >(0,proc_num_recv[phase]),
                *this);

    } else {
      T_INT nparticles = N_local + N_ghost - ( (phase%2==1) ? proc_num_recv[phase-1]:0 );
      Kokkos::parallel_for("CommMPI::halo_exchange_self",
                Kokkos::RangePolicy<TagHaloSelf, Kokkos::IndexType<T_INT> >(0,nparticles),
                *this);
      Kokkos::deep_copy(count,pack_count);
      bool redo = false;
      if(N_local+N_ghost+count>x.size()) {
        system->resize(N_local + N_ghost + count);
        s = *system;
	x = s.xvf.slice<Positions>(); // reslice after resize
        redo = true;
      }
      if(count > pack_indicies.extent(0)) {
        Kokkos::realloc(pack_buffer,count*1.1);
        Kokkos::resize(pack_indicies_all,6,count*1.1);
        pack_indicies = Kokkos::subview(pack_indicies_all,phase,Kokkos::ALL());
        redo = true;
      }
      if(redo) {
        Kokkos::deep_copy(pack_count,0);
        Kokkos::parallel_for("CommMPI::halo_exchange_self",
                  Kokkos::RangePolicy<TagHaloSelf, Kokkos::IndexType<T_INT> >(0,nparticles),
                  *this);
      }
      proc_num_send[phase] = count;
      proc_num_recv[phase] = count;
    }

    num_ghost[phase] = count;
    N_ghost += count;
    if (x.size() > N_local+N_ghost) {
      system->resize(N_local+N_ghost);
    }

  }
  static int step = 0;
  step++;

  system->N_ghost = N_ghost;

  Kokkos::Profiling::popRegion();
};

void Comm::update_halo() {

  Kokkos::Profiling::pushRegion("Comm::update_halo");

  N_ghost = 0;
  s=*system;
  x = s.xvf.slice<Positions>();

  pack_buffer_update = t_buffer_update((T_X_FLOAT*)pack_buffer.data(),pack_indicies_all.extent(1));
  unpack_buffer_update = t_buffer_update((T_X_FLOAT*)unpack_buffer.data(),pack_indicies_all.extent(1));

  for(phase = 0; phase<6; phase++) {
    pack_indicies = Kokkos::subview(pack_indicies_all,phase,Kokkos::ALL());
    if(proc_grid[phase/2]>1) {  
      
      Kokkos::parallel_for("CommMPI::halo_update_pack",
         Kokkos::RangePolicy<TagHaloUpdatePack, Kokkos::IndexType<T_INT> >(0,proc_num_send[phase]),
         *this);
      MPI_Request request;
      MPI_Status status;
      MPI_Irecv(unpack_buffer.data(),proc_num_recv[phase]*sizeof(T_X_FLOAT)*3/sizeof(int),MPI_INT, proc_neighbors_recv[phase],100002,MPI_COMM_WORLD,&request);
      MPI_Send (pack_buffer.data(),proc_num_send[phase]*sizeof(T_X_FLOAT)*3/sizeof(int),MPI_INT, proc_neighbors_send[phase],100002,MPI_COMM_WORLD);
      s = *system;
      x = s.xvf.slice<Positions>();
      MPI_Wait(&request,&status);
      const int count = proc_num_recv[phase];
      if(unpack_buffer_update.extent(0)<count) {
        unpack_buffer_update = t_buffer_update((T_X_FLOAT*)unpack_buffer.data(),count);
      }
      Kokkos::parallel_for("CommMPI::halo_update_unpack",
                Kokkos::RangePolicy<TagHaloUpdateUnpack, Kokkos::IndexType<T_INT> >(0,proc_num_recv[phase]),
                *this);

    } else {
      Kokkos::parallel_for("CommMPI::halo_update_self",
        Kokkos::RangePolicy<TagHaloUpdateSelf, Kokkos::IndexType<T_INT> >(0,proc_num_send[phase]),
        *this);
    }
    N_ghost += proc_num_recv[phase];
    if (x.size() > N_local+N_ghost) {
      system->resize(N_local+N_ghost);
    }

  }

  Kokkos::Profiling::popRegion();
};

void Comm::update_force() {

  Kokkos::Profiling::pushRegion("Comm::update_force");

  N_ghost = 0;
  s=*system;
  f = s.xvf.slice<Forces>();

  ghost_offsets[0] = s.N_local;
  for(phase = 1; phase<6; phase++) {
    ghost_offsets[phase] = ghost_offsets[phase-1] + proc_num_recv[phase-1];
  }

  pack_buffer_update = t_buffer_update((T_X_FLOAT*)pack_buffer.data(),pack_indicies_all.extent(1));
  unpack_buffer_update = t_buffer_update((T_X_FLOAT*)unpack_buffer.data(),pack_indicies_all.extent(1));

  for(phase = 5; phase>=0; phase--) {
    pack_indicies = Kokkos::subview(pack_indicies_all,phase,Kokkos::ALL());
    if(proc_grid[phase/2]>1) {

      Kokkos::parallel_for("CommMPI::halo_force_pack",
         Kokkos::RangePolicy<TagHaloForcePack, Kokkos::IndexType<T_INT> >(0,proc_num_recv[phase]),
         *this);
      MPI_Request request;
      MPI_Status status;
      MPI_Irecv(pack_buffer.data(),proc_num_send[phase]*sizeof(T_X_FLOAT)*3/sizeof(int),MPI_INT, proc_neighbors_send[phase],100002,MPI_COMM_WORLD,&request);
      MPI_Send (unpack_buffer.data(),proc_num_recv[phase]*sizeof(T_X_FLOAT)*3/sizeof(int),MPI_INT, proc_neighbors_recv[phase],100002,MPI_COMM_WORLD);
      s = *system;
      f = s.xvf.slice<Forces>();
      MPI_Wait(&request,&status);
      Kokkos::parallel_for("CommMPI::halo_force_unpack",
                Kokkos::RangePolicy<TagHaloForceUnpack, Kokkos::IndexType<T_INT> >(0,proc_num_send[phase]),
                *this);

    } else {
      Kokkos::parallel_for("CommMPI::halo_force_self",
        Kokkos::RangePolicy<TagHaloForceSelf, Kokkos::IndexType<T_INT> >(0,proc_num_send[phase]),
        *this);
    }
  }

  Kokkos::Profiling::popRegion();
};

const char* Comm::name() { return "CommMPI"; }

int Comm::process_rank() { return proc_rank; }
int Comm::num_processes() { return proc_size; }
void Comm::error(const char *errormsg) {
  if(proc_rank==0)
  printf("%s\n",errormsg);
  MPI_Abort(MPI_COMM_WORLD,1);
};
